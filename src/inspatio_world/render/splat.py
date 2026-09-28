"""Depth forward-splatting: source views + depth -> the rendered condition of target cameras.

The world model is conditioned on what the target camera would see of the known scene: every
source pixel is lifted to 3D with its depth, moved into the target camera, projected, and
splatted bilinearly onto the four neighbouring target pixels. Overlapping splats are blended
with weights that fall off steeply with depth (`exp(-50 * log(1 + z) / max log(1 + z))`), so the
nearest surface wins; pixels no splat reaches are "unknown" and form the render mask. With several
source views, each view is splatted separately and the nearest known depth wins per pixel.

    views      images (V, 3, H, W) in [-1, 1], depth (V, H, W), Tcw (V, 4, 4), K (V, 3, 3)
    targets    Tcw (F, 4, 4) with the shared target K, and the K views each frame sees
    render     rgb (F, 3, H, W) in [-1, 1] (-1 where unknown), mask (F, H, W),
               and per (frame, view) the count of pixels the view covers (view selection)

Everything runs in float32 on the device of the inputs. Pixel coordinates are integer pixel
indices (no half-pixel offset), as upstream.

# Adapted from https://github.com/inspatio/inspatio-world-v1.5/blob/main/pipeline/depth_warper.py
# and pipeline/render_scene.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..types import (
    DepthMaps,
    Intrinsics,
    PixelScores,
    PointsHomogeneous,
    Poses,
    RenderedFrames,
    RenderMasks,
    SourceImages,
)

MAX_SPLAT_DEPTH = 1000.0  # depths are saturated here before the log-depth blending weight
DEPTH_SHARPNESS = 50.0  # exponent scale of the nearest-surface-wins blending weight
MIN_DEPTH = 1e-6

type SplatValues = Float[Tensor, "P C H W"]  # per (frame, view) pair, the values to splat
type PairDepths = Float[Tensor, "P H W"]


@dataclass(frozen=True, slots=True)
class LiftedViews:
    """Source views lifted to camera-frame points once, rendered into any target camera."""

    images: SourceImages
    points: PointsHomogeneous  # camera-frame (x, y, z, 1) of every source pixel
    valid: Bool[Tensor, "V H W"]  # pixels with positive depth
    camera_to_world: Poses  # (V, 4, 4) inverse of the source Tcw


@dataclass(frozen=True, slots=True)
class Render:
    rgb: RenderedFrames
    mask: RenderMasks
    view_pixels: PixelScores  # (F, K) pixels each of the frame's views covers


def lift_views(
    images: SourceImages, depth: DepthMaps, intrinsics: Intrinsics, world_to_camera: Poses
) -> LiftedViews:
    """Lift every source pixel to its camera-frame 3D point with the view's depth."""

    _, h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=torch.float32),
        torch.arange(w, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    rays = einsum(torch.linalg.inv(intrinsics.float()), pixels, "v i j, h w j -> v h w i")
    local = rays * depth.float()[..., None]
    return LiftedViews(
        images=images.float(),
        points=torch.cat([local, torch.ones_like(local[..., :1])], dim=-1),
        valid=depth > 0,
        camera_to_world=torch.linalg.inv(world_to_camera.float()),
    )


def render(
    views: LiftedViews,
    view_index: Int[Tensor, "F K"],
    world_to_camera: Poses,
    intrinsics: Float[Tensor, "3 3"],
) -> Render:
    """Splat each frame's K source views into its target camera and keep the nearest surface.

    `view_index[f]` lists the views frame f sees: every view for image scenes, the frame's own
    source frame for videos.
    """

    k = view_index.shape[1]
    # Source camera -> world -> target camera, for every (frame, view) pair
    transform = einsum(
        world_to_camera.float(), views.camera_to_world[view_index], "f i m, f k m j -> f k i j"
    )
    moved = einsum(transform, views.points[view_index], "f k i j, f k h w j -> f k h w i")[..., :3]
    projected = einsum(intrinsics.float(), moved, "i j, f k h w j -> f k h w i")
    z = projected[..., 2]
    coordinates = projected[..., :2] / z.clamp_min(MIN_DEPTH)[..., None]

    values = torch.cat([views.images[view_index], z[:, :, None]], dim=2)
    valid = views.valid[view_index] & (z > 0)
    splatted, known = splat(
        rearrange(values, "f k c h w -> (f k) c h w"),
        rearrange(valid, "f k h w -> (f k) h w").float(),
        rearrange(z, "f k h w -> (f k) h w"),
        rearrange(coordinates, "f k h w two -> (f k) h w two"),
    )
    splatted = rearrange(splatted, "(f k) c h w -> f k c h w", k=k)
    known = rearrange(known, "(f k) h w -> f k h w", k=k)

    # Nearest known depth wins; ties keep the lower view index, as upstream's sequential fusion
    rgb, depth = splatted[:, :, :3].clamp(-1, 1), splatted[:, :, 3]
    nearest = torch.where(known, depth, torch.inf).argmin(dim=1)
    rgb = torch.gather(rgb, 1, repeat(nearest, "f h w -> f 1 c h w", c=3))[:, 0]
    mask = known.any(dim=1)
    return Render(
        rgb=torch.where(mask[:, None], rgb, -1.0),
        mask=mask,
        view_pixels=known.sum(dim=(2, 3)),
    )


def splat(
    values: SplatValues,
    weights: PairDepths,
    depth: PairDepths,
    coordinates: Float[Tensor, "P H W 2"],
) -> tuple[SplatValues, Float[Tensor, "P H W"]]:
    """Bilinear forward splatting with depth-weighted blending; returns (values, known mask).

    The target canvas has a one-pixel border so out-of-frame splats clamp onto it and are cropped
    away. When a coordinate is an exact integer its four corners coincide, as upstream.
    """

    p, c, h, w = values.shape
    offset = coordinates + 1
    floor, ceil = torch.floor(offset).long(), torch.ceil(offset).long()
    bounds = torch.tensor([w + 1, h + 1], device=values.device)
    offset = torch.minimum(offset.clamp_min(0), bounds)
    floor = torch.minimum(floor.clamp_min(0), bounds)
    ceil = torch.minimum(ceil.clamp_min(0), bounds)

    # Proximity of the splat centre to each corner, before the clamps' distortions
    near_x, near_y = 1 - (offset[..., 0] - floor[..., 0]), 1 - (offset[..., 1] - floor[..., 1])
    far_x, far_y = 1 - (ceil[..., 0] - offset[..., 0]), 1 - (ceil[..., 1] - offset[..., 1])
    log_depth = torch.log1p(depth.clamp(0, MAX_SPLAT_DEPTH))
    scale = log_depth.amax(dim=(1, 2), keepdim=True).clamp_min(MIN_DEPTH)
    depth_weight = torch.exp(log_depth / scale * DEPTH_SHARPNESS)
    base = weights * (depth >= 0) / depth_weight

    # One canvas row per (pair, row, column) holding [weighted values | weight]
    canvas = values.new_zeros(p * (h + 2) * (w + 2), c + 1)
    pair = torch.arange(p, device=values.device)[:, None, None]
    channels_last = rearrange(values, "p c h w -> p h w c")
    corners = (
        (floor[..., 1], floor[..., 0], near_y * near_x),
        (ceil[..., 1], floor[..., 0], far_y * near_x),
        (floor[..., 1], ceil[..., 0], near_y * far_x),
        (ceil[..., 1], ceil[..., 0], far_y * far_x),
    )
    for row, column, proximity in corners:
        weight = proximity * base
        # Splats on the border ring are cropped below and zero weights add nothing: dropping
        # them first keeps out-of-frame points from serializing atomics on the border pixels
        keep = (weight > 0) & (row > 0) & (row <= h) & (column > 0) & (column <= w)
        index = ((pair * (h + 2) + row) * (w + 2) + column)[keep]
        contribution = torch.cat(
            [channels_last[keep] * weight[keep][:, None], weight[keep][:, None]], dim=1
        )
        canvas.index_add_(0, index, contribution)

    canvas = rearrange(canvas, "(p h w) c -> p h w c", p=p, h=h + 2)
    canvas, total = canvas[..., :c], canvas[..., c:]
    canvas, total = canvas[:, 1:-1, 1:-1], total[:, 1:-1, 1:-1]
    known = total[..., 0] > 0
    out = torch.where(known[..., None], canvas / total, 0.0)
    return rearrange(out, "p h w c -> p c h w"), known
