"""Metric depth and cameras for a set of views: Depth-Anything-3 "nested giant-large", as InSpatio runs it.

InSpatio-World lifts its source views (one image, a few images, or the frames of a video) to a
point cloud from per-view depth and cameras. `DepthEstimator` reproduces the upstream call
`DA3NESTED-GIANT-LARGE.inference(views, use_ray_pose=False, process_res=504)` and InSpatio's
post-processing of its result:

1. resize every view so its longest side is 504 px and both sides are multiples of 14
   (`preprocess.py`);
2. the any-view ViT-g (cross-view attention, 40 blocks) gives relative depth, a confidence and a
   camera per view; the metric ViT-L runs each view alone and gives depth at a 300 px focal
   length plus a sky score;
3. the metric depth is rescaled by each view's predicted focal length and one scale is fitted,
   by least squares over confident non-sky pixels, to bring the relative depth and camera
   translations to metres; sky pixels get the 99th percentile of the non-sky depth (at most
   200 m);
4. depth is resized back to the input size (nearest), K is scaled to it, and the cameras are
   re-expressed relative to view 0 and Gaussian-smoothed over the view index (sigma 2).

Numerics follow upstream's bf16 autocast, made explicit: the ViT linear layers and patch
embedding hold bf16 weights and run in bf16, with float32 layer norms, LayerScale and residual
stream (`transformer.py`); heads, camera decoder and the metric alignment run in float32; the
trajectory post-processing runs in float64. The weights file stores exactly those dtypes.

Deliberate deviation: upstream estimates the median confidence and the non-sky 99th percentile
from up to 100k randomly drawn pixels, so repeated runs differ slightly; here both quantiles
are exact (the same `torch.quantile` interpolation over all pixels) and the result is
deterministic.

Ref: Depth Anything 3 arXiv:2511.10647
# Adapted from https://github.com/ByteDance-Seed/depth-anything-3/blob/main/src/depth_anything_3/model/da3.py
# and .../utils/alignment.py; post-processing from https://github.com/inspatio/inspatio-world-v1.5 (pipeline/depth_estimation.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from ..types import ConfidenceMaps, DepthMaps, Intrinsics, Poses, SkyMaps, SourceViews
from ..utils.hub import HubModule
from .backbone import VisionTransformer, ViTConfig
from .camera import CameraDecoder, decode_pose, smooth_trajectory
from .heads import DepthHead, MetricHead
from .preprocess import preprocess, resize_nearest

METRIC_FOCAL = 300.0  # focal length (px) the metric head's depth is expressed for
SKY_THRESHOLD = 0.3
MIN_NON_SKY_PIXELS = 10
MIN_DEPTH = 1e-3  # relative depth below this is left out of the scale fit
MIN_METRIC_DEPTH = 1e-2
SCALE_EPS = 1e-12
MAX_SKY_DEPTH = 200.0
SKY_QUANTILE = 0.99


@dataclass(frozen=True, slots=True)
class DepthEstimatorConfig:
    """Architecture quantities of DA3NESTED-GIANT-LARGE plus InSpatio's processing constants."""

    anyview_dim: int = 1536
    anyview_depth: int = 40
    anyview_heads: int = 24
    anyview_out_layers: tuple[int, ...] = (19, 27, 33, 39)
    alternating_start: int = 13  # first cross-view / camera-token / rotary / QK-norm block
    metric_dim: int = 1024
    metric_depth: int = 24
    metric_heads: int = 16
    metric_out_layers: tuple[int, ...] = (4, 11, 17, 23)
    head_features: int = 256
    head_channels: tuple[int, ...] = (256, 512, 1024, 1024)
    process_res: int = 504  # longest side of the processed views
    smoothing_sigma: float = 2.0  # temporal Gaussian on the camera path, in views

    @property
    def anyview(self) -> ViTConfig:
        return ViTConfig(
            dim=self.anyview_dim,
            depth=self.anyview_depth,
            num_heads=self.anyview_heads,
            out_layers=self.anyview_out_layers,
            swiglu=True,
            alternating_start=self.alternating_start,
        )

    @property
    def metric(self) -> ViTConfig:
        return ViTConfig(
            dim=self.metric_dim,
            depth=self.metric_depth,
            num_heads=self.metric_heads,
            out_layers=self.metric_out_layers,
            swiglu=False,
            alternating_start=None,
        )


@dataclass(frozen=True, slots=True)
class DepthEstimate:
    """Per-view metric depth and cameras, in pixels of the input views (OpenCV convention)."""

    depth: DepthMaps  # float32 z-depth in metres, (V, H, W)
    intrinsics: Intrinsics  # (V, 3, 3)
    world_to_camera: Poses  # (V, 4, 4); view 0 is the identity before smoothing


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    """`torch.quantile(values, q)` (linear interpolation) without its 16M-element limit."""

    ordered = values.flatten().sort().values
    rank = torch.tensor(q, dtype=values.dtype, device=values.device) * (ordered.numel() - 1)
    below = rank.long()
    return torch.lerp(ordered[below], ordered[rank.ceil().long()], rank - below)


def align_to_metric(
    depth: DepthMaps,
    confidence: ConfidenceMaps,
    intrinsics: Intrinsics,
    metric_depth: DepthMaps,
    sky: SkyMaps,
) -> tuple[DepthMaps, torch.Tensor]:
    """Scale relative depth to metres with the metric branch; returns `(depth, scale)`."""

    focal = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2
    metric_depth = metric_depth * (focal[:, None, None] / METRIC_FOCAL)
    non_sky = sky < SKY_THRESHOLD
    assert non_sky.sum() > MIN_NON_SKY_PIXELS, "depth estimator: every pixel was classified as sky"

    # Least squares `metric ~ scale * depth` over confident, valid, non-sky pixels
    median_confidence = _quantile(confidence[non_sky], 0.5)
    mask = (
        (confidence >= median_confidence)
        & non_sky
        & (metric_depth > MIN_METRIC_DEPTH)
        & (depth > MIN_DEPTH)
    )
    relative, metric = depth[mask], metric_depth[mask]
    scale = (metric * relative).sum() / (relative * relative).sum().clamp_min(SCALE_EPS)
    depth = depth * scale

    sky_depth = _quantile(depth[non_sky], SKY_QUANTILE).clamp(max=MAX_SKY_DEPTH)
    return torch.where(non_sky, depth, sky_depth), scale


class DepthEstimator(nn.Module, HubModule):
    """DA3NESTED-GIANT-LARGE: any-view ViT-g depth + cameras, scaled by a metric ViT-L."""

    config_class: ClassVar[type] = DepthEstimatorConfig

    def __init__(self, config: DepthEstimatorConfig) -> None:
        super().__init__()
        self.config = config
        anyview, metric = config.anyview, config.metric
        self.backbone = VisionTransformer(anyview)
        self.head = DepthHead(2 * anyview.dim, config.head_features, config.head_channels)
        self.camera_decoder = CameraDecoder(2 * anyview.dim)
        self.metric_backbone = VisionTransformer(metric)
        self.metric_head = MetricHead(metric.dim, config.head_features, config.head_channels)

    @torch.inference_mode()
    def estimate(self, views: SourceViews | np.ndarray) -> DepthEstimate:
        """Depth, intrinsics and smoothed world-to-camera poses for uint8 RGB views `(V, H, W, 3)`."""

        device = next(self.parameters()).device
        views = torch.as_tensor(views).to(device)
        height, width = views.shape[1:3]
        images = preprocess(views, self.config.process_res)
        size = (images.shape[-2], images.shape[-1])

        features, camera_tokens = self.backbone(images)
        depth, confidence = self.head(features, size)
        world_to_camera, intrinsics = decode_pose(self.camera_decoder(camera_tokens), size)
        metric_features, _ = self.metric_backbone(images)
        metric_depth, sky = self.metric_head(metric_features, size)
        depth, scale = align_to_metric(depth, confidence, intrinsics, metric_depth, sky)
        world_to_camera = torch.cat(
            [world_to_camera[:, :, :3], world_to_camera[:, :, 3:] * scale], dim=-1
        )

        # Back to the input resolution, cameras relative to view 0 and smoothed over time
        intrinsics[:, 0] *= width / size[1]
        intrinsics[:, 1] *= height / size[0]
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64, device=device)
        world_to_camera = torch.cat(
            [world_to_camera.double(), bottom.expand(len(views), 1, 4)], dim=1
        )
        camera_to_world = torch.linalg.inv(world_to_camera)
        camera_to_world = torch.linalg.inv(camera_to_world[0]) @ camera_to_world
        camera_to_world = smooth_trajectory(camera_to_world, self.config.smoothing_sigma)
        return DepthEstimate(
            depth=resize_nearest(depth, (height, width)),
            intrinsics=intrinsics,
            world_to_camera=torch.linalg.inv(camera_to_world).float(),
        )
