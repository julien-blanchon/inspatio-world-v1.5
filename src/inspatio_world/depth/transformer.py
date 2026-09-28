"""The DINOv2 transformer block as DA3 runs it: optional QK-norm and 2D rotary on the query/key.

Both DA3 backbones stack this block. The any-view ViT-g switches on QK layer norms and 2D rotary
position embedding from its 13th block (where alternating view-local / cross-view attention
starts); the metric ViT-L never does. The feed-forward is SwiGLU for ViT-g and a GELU MLP for
ViT-L, the two released checkpoints' choice.

Numerics reproduce DA3's bf16 autocast explicitly: linear layers hold bf16 weights and see bf16
inputs; layer norms (including the QK norms) run in float32 with float32 weights; LayerScale
multiplies the bf16 branch output by a float32 gain, so the residual stream stays float32. The
rotary rotation runs in float32 and attention in bf16. The rotary angles are the bf16 product
`position * inverse_frequency` (autocast runs upstream's `einsum` in bf16) widened to float32.

# Adapted from https://github.com/ByteDance-Seed/depth-anything-3/blob/main/src/depth_anything_3/model/dinov2/layers/block.py
# and .../dinov2/layers/attention.py, .../dinov2/layers/rope.py, .../dinov2/layers/swiglu_ffn.py
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..types import TokenRotary, ViewTokens

BLOCK_NORM_EPS = 1e-6  # the blocks' pre-norms; the QK and final norms keep torch's 1e-5
ROPE_BASE = 100.0
MLP_RATIO = 4


def rotary_tables(positions: torch.Tensor, head_dim: int) -> tuple[TokenRotary, TokenRotary]:
    """cos / sin tables of 2D rotary angles for integer `(y, x)` positions `(..., N, 2)`.

    The first half of each head rotates with the row index, the second half with the column.
    """

    half = head_dim // 2
    exponents = torch.arange(0, half, 2, device=positions.device).float() / half
    inverse_frequency = 1.0 / torch.pow(ROPE_BASE, exponents)
    # bf16 product rounded like upstream's autocast einsum, then widened for cos / sin
    angles = (positions.bfloat16()[..., None] * inverse_frequency.bfloat16()).float()
    angles = torch.cat([angles, angles], dim=-1)  # (..., N, 2 axes, D / 2)
    angles = rearrange(angles, "... n p d -> ... n (p d)")
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_rotary(heads: torch.Tensor, cos: TokenRotary, sin: TokenRotary) -> torch.Tensor:
    """Rotate `(B, A, N, D)` heads: the row and column halves each get a 1D rotary."""

    rows, cols = heads.chunk(2, dim=-1)
    cos_rows, cos_cols = cos[:, None].chunk(2, dim=-1)
    sin_rows, sin_cols = sin[:, None].chunk(2, dim=-1)
    rows = rows * cos_rows + _rotate_half(rows) * sin_rows
    cols = cols * cos_cols + _rotate_half(cols) * sin_cols
    return torch.cat([rows, cols], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention, optionally with float32 QK layer norms and 2D rotary."""

    def __init__(self, dim: int, num_heads: int, *, qk_norm: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, dtype=torch.bfloat16)
        self.q_norm = nn.LayerNorm(head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(head_dim) if qk_norm else None
        self.proj = nn.Linear(dim, dim, dtype=torch.bfloat16)

    @override
    def forward(self, x: ViewTokens, rotary: tuple[TokenRotary, TokenRotary] | None) -> ViewTokens:
        qkv = self.qkv(x.bfloat16())
        query, key, value = rearrange(
            qkv, "b n (three a d) -> three b a n d", three=3, a=self.num_heads
        )
        if self.q_norm is not None and self.k_norm is not None:
            query, key = self.q_norm(query.float()), self.k_norm(key.float())
        if rotary is not None:
            query, key = apply_rotary(query, *rotary), apply_rotary(key, *rotary)

        out = F.scaled_dot_product_attention(query.bfloat16(), key.bfloat16(), value)
        return self.proj(rearrange(out, "b a n d -> b n (a d)"))


class SwiGLU(nn.Module):
    """DINOv2's fused SwiGLU: one projection to both gates, hidden width 2/3 * 4 * dim."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = (int(dim * MLP_RATIO * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(dim, 2 * hidden, dtype=torch.bfloat16)
        self.w3 = nn.Linear(hidden, dim, dtype=torch.bfloat16)

    @override
    def forward(self, x: ViewTokens) -> ViewTokens:
        gate, value = self.w12(x.bfloat16()).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


class Mlp(nn.Module):
    """Two-layer GELU feed-forward, hidden width 4 * dim."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, MLP_RATIO * dim, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(MLP_RATIO * dim, dim, dtype=torch.bfloat16)

    @override
    def forward(self, x: ViewTokens) -> ViewTokens:
        return self.fc2(F.gelu(self.fc1(x.bfloat16())))


class Block(nn.Module):
    """Pre-norm transformer block with LayerScale on both residual branches."""

    def __init__(self, dim: int, num_heads: int, *, swiglu: bool, qk_norm: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=BLOCK_NORM_EPS)
        self.attn = Attention(dim, num_heads, qk_norm=qk_norm)
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, eps=BLOCK_NORM_EPS)
        self.mlp = SwiGLU(dim) if swiglu else Mlp(dim)
        self.ls2 = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: ViewTokens, rotary: tuple[TokenRotary, TokenRotary] | None) -> ViewTokens:
        x = x + self.attn(self.norm1(x), rotary) * self.ls1
        return x + self.mlp(self.norm2(x)) * self.ls2
