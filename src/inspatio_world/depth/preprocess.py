"""Resize source views for DA3 and resize its depth back, reproducing OpenCV bit for bit in torch.

DA3's input processor ("upper_bound_resize") scales each view so its longest side is
`process_res`, then nudges each side to the nearest multiple of the 14-pixel patch, both steps
with `cv2.resize` on uint8 RGB: `INTER_AREA` when shrinking and `INTER_CUBIC` when any side
grows. InSpatio then brings the depth back to the source size with `INTER_NEAREST`. OpenCV is
not a dependency here, so the three kernels are re-derived from its source:

- area: per-axis tap tables from `computeResizeAreaTab`, accumulated in float32 with fused
  multiply-adds (the aarch64 build contracts `buf + S * alpha`), rounded half-to-even;
- cubic: A = -0.75 coefficients quantized to 11-bit fixed point, integer horizontal pass, and a
  vertical pass rounded half-to-even (the SIMD path);
- nearest: `floor(x / (dst / src))` in float64, clamped to the last source pixel.

On the InSpatio example views (832x480 -> 504x291 area -> 504x294 cubic) the output equals
OpenCV's exactly; an upscaling cubic resize can differ by one level on a handful of pixels,
where OpenCV's scalar row tail rounds half up. Integer-ratio area shrinks (OpenCV's fast path,
e.g. 1008 -> 504) are resampled with the generic area weights, which agree up to the same
half-level rounding.

# Adapted from https://github.com/ByteDance-Seed/depth-anything-3/blob/main/src/depth_anything_3/utils/io/input_processor.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
from einops import rearrange

from ..types import DepthMaps, ProcessedViews, ResampleIndex, ResampleWeights, SourceViews

PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CUBIC_A = -0.75  # OpenCV's bicubic sharpness
CUBIC_COEF_SCALE = 2048  # INTER_RESIZE_COEF_SCALE: cubic weights in 11-bit fixed point
CUBIC_TAPS = 4
AREA_EPS = 1e-3  # partial-coverage threshold of computeResizeAreaTab
MAX_LEVEL = 255


def processing_size(height: int, width: int, process_res: int) -> tuple[int, int]:
    """Size after the longest-side resize, before the patch-multiple adjustment."""

    scale = process_res / float(max(height, width))
    if max(height, width) == process_res:
        return height, width
    return max(1, round(height * scale)), max(1, round(width * scale))


def _nearest_multiple(size: int) -> int:
    down = size // PATCH_SIZE * PATCH_SIZE
    up = down + PATCH_SIZE
    return max(1, up if abs(up - size) <= abs(size - down) else down)


def _area_taps(src: int, dst: int) -> tuple[ResampleIndex, ResampleWeights]:
    """OpenCV's `computeResizeAreaTab` as dense `(dst, taps)` tables, zero-padded."""

    scale = 1.0 / (dst / src)
    rows: list[list[tuple[int, float]]] = []
    for dx in range(dst):
        start = dx * scale
        end = start + scale
        cell = min(scale, src - start)
        first = math.ceil(start)
        last = min(math.floor(end), src - 1)
        first = min(first, last)
        row = []
        if first - start > AREA_EPS:
            row.append((first - 1, (first - start) / cell))
        row += [(sx, 1.0 / cell) for sx in range(first, last)]
        if end - last > AREA_EPS:
            row.append((last, min(min(end - last, 1.0), cell) / cell))
        rows.append(row)

    taps = max(len(row) for row in rows)
    index = np.zeros((dst, taps), dtype=np.int64)
    weight = np.zeros((dst, taps), dtype=np.float32)  # padded taps weigh 0 and add exactly 0
    for dx, row in enumerate(rows):
        for tap, (sx, alpha) in enumerate(row):
            index[dx, tap] = sx
            weight[dx, tap] = alpha
    return torch.from_numpy(index), torch.from_numpy(weight)


def _cubic_coefficients(frac: np.float32) -> list[np.float32]:
    """OpenCV's `interpolateCubic`, in float32 like the C++ code."""

    a, one = np.float32(CUBIC_A), np.float32(1)
    before, after = frac + one, one - frac
    c0 = ((a * before - np.float32(5) * a) * before + np.float32(8) * a) * before - np.float32(
        4
    ) * a
    c1 = ((a + np.float32(2)) * frac - (a + np.float32(3))) * frac * frac + one
    c2 = ((a + np.float32(2)) * after - (a + np.float32(3))) * after * after + one
    return [c0, c1, c2, one - c0 - c1 - c2]


def _cubic_taps(src: int, dst: int) -> tuple[ResampleIndex, ResampleWeights]:
    """Four clamped taps per output pixel with fixed-point weights (replicated border)."""

    scale = 1.0 / (dst / src)
    index = np.zeros((dst, CUBIC_TAPS), dtype=np.int64)
    weight = np.zeros((dst, CUBIC_TAPS), dtype=np.int64)
    for dx in range(dst):
        position = np.float32((dx + 0.5) * scale - 0.5)
        sx = math.floor(position)
        coefficients = _cubic_coefficients(np.float32(position - np.float32(sx)))
        for tap, coefficient in enumerate(coefficients):
            index[dx, tap] = min(max(sx - 1 + tap, 0), src - 1)
            weight[dx, tap] = int(np.rint(coefficient * np.float32(CUBIC_COEF_SCALE)))
    return torch.from_numpy(index), torch.from_numpy(weight)


def _area_pass(values: torch.Tensor, axis: int, size: int) -> torch.Tensor:
    """Resample one axis of a float32 `(V, H, W, 3)` stack with fused multiply-add accumulation."""

    index, weight = (t.to(values.device) for t in _area_taps(values.shape[axis], size))
    shape = [1] * values.dim()
    shape[axis] = size
    total = values.index_select(axis, index[:, 0]) * weight[:, 0].reshape(shape)
    for tap in range(1, index.shape[1]):
        term = (
            values.index_select(axis, index[:, tap]).double()
            * weight[:, tap].reshape(shape).double()
        )
        total = (total.double() + term).float()  # exact product + one rounding == fmaf
    return total


def _cubic_pass(values: torch.Tensor, axis: int, size: int) -> torch.Tensor:
    """Resample one axis of an int64 stack with the fixed-point cubic weights."""

    index, weight = (t.to(values.device) for t in _cubic_taps(values.shape[axis], size))
    shape = [1] * values.dim()
    shape[axis] = size
    return sum(
        (
            values.index_select(axis, index[:, tap]) * weight[:, tap].reshape(shape)
            for tap in range(CUBIC_TAPS)
        ),
        start=torch.zeros((), dtype=torch.int64, device=values.device),
    )


def _resize_uint8(views: SourceViews, size: tuple[int, int]) -> SourceViews:
    """`cv2.resize` of uint8 views: area when no side grows, bicubic otherwise."""

    height, width = size
    if (height, width) == tuple(views.shape[1:3]):
        return views
    if height <= views.shape[1] and width <= views.shape[2]:
        resized = _area_pass(_area_pass(views.float(), 2, width), 1, height)
        return torch.round(resized).clamp(0, MAX_LEVEL).to(torch.uint8)
    resized = _cubic_pass(_cubic_pass(views.long(), 2, width), 1, height)
    levels = torch.round(resized.double() / CUBIC_COEF_SCALE**2)  # two fixed-point passes
    return levels.clamp(0, MAX_LEVEL).to(torch.uint8)


def preprocess(views: SourceViews, process_res: int) -> ProcessedViews:
    """Resize like DA3's `upper_bound_resize`, then scale to [0, 1] and ImageNet-normalize."""

    height, width = processing_size(views.shape[1], views.shape[2], process_res)
    views = _resize_uint8(views, (height, width))
    views = _resize_uint8(views, (_nearest_multiple(height), _nearest_multiple(width)))

    # NOTE: a device tensor divisor keeps a true division; CUDA turns `x / 255` into
    # `x * (1 / 255)`, one ulp away from torchvision's CPU `ToTensor`
    levels = torch.tensor(float(MAX_LEVEL), device=views.device)
    pixels = rearrange(views, "v h w c -> v c h w").float() / levels
    mean = torch.tensor(IMAGENET_MEAN, device=pixels.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=pixels.device)[:, None, None]
    # NOTE: contiguous NCHW like upstream's stacked CHW tensors; a channels-last view selects
    # other cuDNN kernels for the bf16 patch embedding
    return ((pixels - mean) / std).contiguous()


def resize_nearest(depth: DepthMaps, size: tuple[int, int]) -> DepthMaps:
    """`cv2.resize(..., INTER_NEAREST)`: each output pixel takes `floor(x * src / dst)`."""

    height, width = size
    src_height, src_width = depth.shape[1:]
    rows = [
        min(math.floor(y * (1.0 / (height / src_height))), src_height - 1) for y in range(height)
    ]
    cols = [min(math.floor(x * (1.0 / (width / src_width))), src_width - 1) for x in range(width)]
    rows_t = torch.tensor(rows, device=depth.device)
    cols_t = torch.tensor(cols, device=depth.device)
    return depth.index_select(1, rows_t).index_select(2, cols_t)
