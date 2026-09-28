"""DPT dense heads of DA3: relative depth + confidence (any-view model) and depth + sky (metric).

Both heads share one feature pyramid (`FeaturePyramid`): the four tapped ViT layers are folded
back onto the 14 px patch grid, projected to (256, 512, 1024, 1024) channels, resampled x4 / x2 /
x1 / x0.5 (strides 3.5, 7, 14 and 28 px), fused top-down by RefineNet blocks,
reduced to 128 channels and bilinearly upsampled to the processing resolution. The any-view
head layer-norms the tokens first and adds a fixed sinusoidal UV embedding after each
projection and after the upsampling; the metric head does neither. Both then apply a small
convolutional output stack: `exp` gives depth, `exp + 1` the confidence and `relu` the sky score.

The any-view checkpoint also carries an auxiliary ray head, used only for DA3's ray-based
pose; InSpatio decodes the camera from the camera token instead, so that branch is not built.
Views run through the heads eight at a time, as upstream does to bound activation memory.
Everything here runs in float32 (upstream disables autocast around its heads).

# Adapted from https://github.com/ByteDance-Seed/depth-anything-3/blob/main/src/depth_anything_3/model/dualdpt.py
# and .../model/dpt.py, .../model/utils/head_utils.py
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..types import ConfidenceMaps, DepthMaps, FeatureMaps, PatchFeatures, SkyMaps
from .backbone import PATCH_SIZE

HEAD_CHUNK = 8  # views per head pass
OUTPUT_FEATURES = 32
UV_EMBED_SCALE = 0.1
UV_EMBED_BASE = 100.0


def _sincos(positions: torch.Tensor, dim: int) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=torch.float32, device=positions.device)
    omega /= dim / 2.0
    omega = 1.0 / UV_EMBED_BASE**omega
    angles = positions[:, None] * omega[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def uv_embedding(
    channels: int, rows: int, cols: int, aspect: float, device: torch.device
) -> FeatureMaps:
    """Sinusoidal embedding of a centred UV grid normalized by the image diagonal, `(1, C, H, W)`."""

    diagonal = (aspect**2 + 1.0) ** 0.5
    span_x, span_y = aspect / diagonal, 1.0 / diagonal
    xs = torch.linspace(
        -span_x * (cols - 1) / cols, span_x * (cols - 1) / cols, cols, device=device
    )
    ys = torch.linspace(
        -span_y * (rows - 1) / rows, span_y * (rows - 1) / rows, rows, device=device
    )
    u, v = torch.meshgrid(xs, ys, indexing="xy")
    embedding = torch.cat(
        [_sincos(u.flatten(), channels // 2), _sincos(v.flatten(), channels // 2)], dim=-1
    )
    return rearrange(embedding, "(h w) c -> 1 c h w", h=rows) * UV_EMBED_SCALE


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    @override
    def forward(self, x: FeatureMaps) -> FeatureMaps:
        return self.conv2(F.relu(self.conv1(F.relu(x)))) + x


class FusionBlock(nn.Module):
    """RefineNet step: add the refined lateral map, refine, upsample, 1x1 projection."""

    def __init__(self, features: int, *, lateral: bool) -> None:
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(features) if lateral else None
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    @override
    def forward(
        self, x: FeatureMaps, lateral: FeatureMaps | None, size: tuple[int, int]
    ) -> FeatureMaps:
        if self.resConfUnit1 is not None and lateral is not None:
            x = x + self.resConfUnit1(lateral)
        x = F.interpolate(self.resConfUnit2(x), size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class FeaturePyramid(nn.Module):
    """Tapped ViT layers -> one 128-channel map at the processing resolution."""

    def __init__(
        self, dim_in: int, features: int, channels: tuple[int, ...], *, uv_embedding: bool
    ) -> None:
        super().__init__()
        self.uv_embedding = uv_embedding
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, c, 1) for c in channels])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[0], channels[0], 4, stride=4),
                nn.ConvTranspose2d(channels[1], channels[1], 2, stride=2),
                nn.Identity(),
                nn.Conv2d(channels[3], channels[3], 3, stride=2, padding=1),
            ]
        )
        self.layer1_rn, self.layer2_rn, self.layer3_rn, self.layer4_rn = (
            nn.Conv2d(c, features, 3, padding=1, bias=False) for c in channels
        )
        self.refinenet1, self.refinenet2, self.refinenet3 = (
            FusionBlock(features, lateral=True) for _ in range(3)
        )
        self.refinenet4 = FusionBlock(features, lateral=False)
        self.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)

    def _embed(self, x: FeatureMaps, aspect: float) -> FeatureMaps:
        if not self.uv_embedding:
            return x
        return x + uv_embedding(x.shape[1], x.shape[2], x.shape[3], aspect, x.device)

    @override
    def forward(self, grids: list[FeatureMaps], size: tuple[int, int]) -> FeatureMaps:
        aspect = size[1] / size[0]
        maps = [
            resize(self._embed(project(grid), aspect))
            for grid, project, resize in zip(grids, self.projects, self.resize_layers, strict=True)
        ]

        l1, l2, l3, l4 = (
            rn(m)
            for rn, m in zip(
                (self.layer1_rn, self.layer2_rn, self.layer3_rn, self.layer4_rn), maps, strict=True
            )
        )
        x = self.refinenet4(l4, None, l3.shape[2:])
        x = self.refinenet3(x, l3, l2.shape[2:])
        x = self.refinenet2(x, l2, l1.shape[2:])
        x = self.refinenet1(x, l1, (l1.shape[2] * 2, l1.shape[3] * 2))
        x = self.output_conv1(x)

        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self._embed(x, aspect)


def _fold(tokens: PatchFeatures, size: tuple[int, int]) -> FeatureMaps:
    """Tokens back onto the patch grid: a channels-last view, as upstream's permute-then-reshape.

    Transposing before splitting the token axis reproduces upstream's strides exactly; for a
    single view they are ambiguous and the convolutions then run as NCHW, as upstream's do.
    """

    channels_first = rearrange(tokens, "v n c -> v c n")
    return rearrange(channels_first, "v c (h w) -> v c h w", h=size[0] // PATCH_SIZE)


def _output_stack(features: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(features // 2, OUTPUT_FEATURES, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(OUTPUT_FEATURES, out_channels, 1),
    )


class DepthHead(nn.Module):
    """Any-view DualDPT main branch: relative depth and its confidence."""

    def __init__(self, dim_in: int, features: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.pyramid = FeaturePyramid(dim_in, features, channels, uv_embedding=True)
        self.output_conv2 = _output_stack(features, 2)

    @override
    def forward(
        self, features: list[PatchFeatures], size: tuple[int, int]
    ) -> tuple[DepthMaps, ConfidenceMaps]:
        depths, confidences = [], []
        for start in range(0, features[0].shape[0], HEAD_CHUNK):
            grids = [_fold(self.norm(f[start : start + HEAD_CHUNK]), size) for f in features]
            logits = self.output_conv2(self.pyramid(grids, size))
            depths.append(logits[:, 0].exp())
            confidences.append(logits[:, 1].exp() + 1)
        return torch.cat(depths), torch.cat(confidences)


class MetricHead(nn.Module):
    """Metric DPT: depth for a focal length of 300 px, and a sky score."""

    def __init__(self, dim_in: int, features: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.pyramid = FeaturePyramid(dim_in, features, channels, uv_embedding=False)
        self.output_conv2 = _output_stack(features, 1)
        self.sky_output_conv2 = _output_stack(features, 1)

    @override
    def forward(
        self, features: list[PatchFeatures], size: tuple[int, int]
    ) -> tuple[DepthMaps, SkyMaps]:
        depths, skies = [], []
        for start in range(0, features[0].shape[0], HEAD_CHUNK):
            # NOTE: upstream's metric DPT copies the folded grid to contiguous NCHW, while the
            # any-view head keeps the channels-last view; the layout picks the cuDNN kernels
            grids = [_fold(f[start : start + HEAD_CHUNK], size).contiguous() for f in features]
            fused = self.pyramid(grids, size)
            depths.append(self.output_conv2(fused)[:, 0].exp())
            skies.append(F.relu(self.sky_output_conv2(fused)[:, 0]))
        return torch.cat(depths), torch.cat(skies)
