"""DA3's DINOv2 backbone: per-view ViT tokens, with cross-view attention for the any-view model.

The metric ViT-L is a plain DINOv2: every block attends within one view and the tapped layers
are layer-normed. The any-view ViT-g (`alternating_start = 13`) changes three things from its
13th block on:

- it alternates attention: even blocks attend within each view (rotary on the `(row, col)`
  patch grid), odd blocks attend across all views jointly (rotary with every patch at `(1, 1)`,
  so position only separates the special token from the patches);
- the class token of every view is replaced by a learned camera token, one for the reference
  view and one shared by all others; with three or more views the reference is picked before
  block 12 (`_reference_view`, DA3's "saddle_balanced") and moved to the front, and the tapped
  outputs are put back in input order afterwards;
- each tapped layer is the concatenation of the last view-local stream (raw) and the current
  stream (layer-normed), 2 x 1536 channels; its special token, unnormed, feeds the camera head.

Views are the batch axis; one call is one scene. The residual stream is float32 and the blocks
follow the bf16 recipe in `transformer.py`; the patch embedding is a bf16 convolution and the
position embedding is interpolated bicubically in float32, like upstream under autocast.

# Adapted from https://github.com/ByteDance-Seed/depth-anything-3/blob/main/src/depth_anything_3/model/dinov2/vision_transformer.py
# and .../model/reference_view_selector.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ..types import CameraTokens, PatchFeatures, ProcessedViews, TokenRotary, ViewTokens
from .transformer import Block, rotary_tables

PATCH_SIZE = 14
PRETRAIN_GRID = 37  # 518 / 14 patches per side of the learned position embedding
POS_EMBED_OFFSET = 0.1  # DINOv2's interpolation scale-factor kludge
REFERENCE_SELECTION_MIN_VIEWS = 3
NORMALIZE_EPS = 1e-8


@dataclass(frozen=True, slots=True)
class ViTConfig:
    """Architecture quantities of one DA3 backbone."""

    dim: int
    depth: int
    num_heads: int
    out_layers: tuple[int, ...]
    swiglu: bool  # SwiGLU feed-forward (ViT-g) instead of the GELU MLP (ViT-L)
    alternating_start: int | None  # first cross-view block; None for the single-view ViT-L


def _normalize(metric: torch.Tensor) -> torch.Tensor:
    return (metric - metric.min()) / (metric.max() - metric.min() + NORMALIZE_EPS)


def _reference_view(class_tokens: CameraTokens) -> int:
    """DA3's "saddle_balanced" choice: the view closest to the median of three statistics."""

    views = class_tokens.shape[0]
    unit = class_tokens / class_tokens.norm(dim=-1, keepdim=True)
    similarity = (unit.bfloat16() @ unit.bfloat16().T).float()  # bf16 under upstream autocast
    similarity = similarity - torch.eye(views, device=similarity.device)
    mean_similarity = similarity.sum(dim=-1) / (views - 1)
    balance = (
        (_normalize(mean_similarity) - 0.5).abs()
        + (_normalize(class_tokens.norm(dim=-1)) - 0.5).abs()
        + (_normalize(unit.var(dim=-1)) - 0.5).abs()
    )
    return int(balance.argmin())


class VisionTransformer(nn.Module):
    """DINOv2 ViT returning the tapped layers' patch features and camera tokens."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        start = config.alternating_start
        self.patch_embed = nn.Conv2d(3, dim, PATCH_SIZE, stride=PATCH_SIZE, dtype=torch.bfloat16)
        self.cls_token = nn.Parameter(torch.zeros(1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1 + PRETRAIN_GRID**2, dim))
        self.camera_token = nn.Parameter(torch.zeros(2, dim)) if start is not None else None
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    config.num_heads,
                    swiglu=config.swiglu,
                    qk_norm=start is not None and i >= start,
                )
                for i in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def embed(self, images: ProcessedViews) -> ViewTokens:
        """Patchify, prepend the class token and add the interpolated position embedding."""

        rows, cols = images.shape[-2] // PATCH_SIZE, images.shape[-1] // PATCH_SIZE
        patches = rearrange(self.patch_embed(images.bfloat16()), "v d h w -> v (h w) d")
        tokens = torch.cat(
            [repeat(self.cls_token, "1 d -> v 1 d", v=images.shape[0]), patches.float()], dim=1
        )

        grid = rearrange(self.pos_embed[1:], "(h w) d -> 1 d h w", h=PRETRAIN_GRID)
        grid = F.interpolate(
            grid,
            scale_factor=(
                (rows + POS_EMBED_OFFSET) / PRETRAIN_GRID,
                (cols + POS_EMBED_OFFSET) / PRETRAIN_GRID,
            ),
            mode="bicubic",
        )
        position = torch.cat([self.pos_embed[:1], rearrange(grid, "1 d h w -> (h w) d")])
        return tokens + position

    def rotary(
        self, views: int, rows: int, cols: int, device: torch.device
    ) -> tuple[tuple[TokenRotary, TokenRotary], ...]:
        """(view-local, cross-view) rotary tables; the special token sits at `(0, 0)`."""

        row_index, col_index = torch.arange(rows, device=device), torch.arange(cols, device=device)
        grid = torch.cartesian_prod(row_index, col_index) + 1  # patches start at (1, 1)
        special = torch.zeros(1, 2, dtype=grid.dtype, device=device)
        local = torch.cat([special, grid])
        joint = torch.cat([special, torch.ones_like(grid)])
        head_dim = self.config.dim // self.config.num_heads
        local_tables = rotary_tables(repeat(local, "n c -> v n c", v=views), head_dim)
        joint_tables = rotary_tables(repeat(joint, "n c -> 1 (v n) c", v=views), head_dim)
        return local_tables, joint_tables

    @override
    def forward(self, images: ProcessedViews) -> tuple[list[PatchFeatures], CameraTokens]:
        views = images.shape[0]
        x = self.embed(images)
        start = self.config.alternating_start
        local_rotary, joint_rotary = self.rotary(
            views, images.shape[-2] // PATCH_SIZE, images.shape[-1] // PATCH_SIZE, images.device
        )

        order = torch.arange(views, device=images.device)
        local_x, taps = x, []
        for i, block in enumerate(self.blocks):
            rotary = start is not None and i >= start
            if start is not None and i == start - 1 and views >= REFERENCE_SELECTION_MIN_VIEWS:
                reference = _reference_view(x[:, 0])
                order = torch.cat(
                    [order[reference : reference + 1], order[:reference], order[reference + 1 :]]
                )
                x, local_x = x[order], local_x[order]
            if start is not None and i == start:
                assert self.camera_token is not None
                x = x.clone()
                x[:, 0] = torch.cat(
                    [self.camera_token[:1], repeat(self.camera_token[1], "d -> v d", v=views - 1)]
                )

            if start is not None and i >= start and i % 2 == 1:
                joint = rearrange(x, "v n d -> 1 (v n) d")
                x = rearrange(block(joint, joint_rotary), "1 (v n) d -> v n d", v=views)
            else:
                x = local_x = block(x, local_rotary if rotary else None)

            if i in self.config.out_layers:
                taps.append(torch.cat([local_x, x], dim=-1) if start is not None else x)

        # Back to input order; only the current-stream half of a tap is layer-normed
        inverse = torch.argsort(order)
        taps = [tap[inverse] for tap in taps]
        dim = self.config.dim
        features = [
            torch.cat([tap[:, 1:, :-dim], self.norm(tap[:, 1:, -dim:])], dim=-1) for tap in taps
        ]
        return features, taps[-1][:, 0]
