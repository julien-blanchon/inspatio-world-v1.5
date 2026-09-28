"""Causal Wan2.1-1.3B diffusion transformer, split into a KV prefill and a cached denoise step.

InSpatio-World generates video autoregressively, three latent frames (one "block") at a time. For
each block the upstream model is called twice with different roles, which this module exposes as
two functions over one set of weights:

    prefill   clean context latents (the source block, plus the previous block's prediction) at
              timestep 0 -> the rotated keys and values of every self-attention layer (the "KV
              cache"); the model's output head is never evaluated on this pass
    denoise   the noisy block, concatenated with its rendered condition, at a timestep of the
              schedule -> the flow prediction, attending to the cached context and to itself

Shape walkthrough for one steady-state block at 832x480 (one token per 2x2 latent cells):

    context    (B, 36, 6, 60, 104)  source + previous latents, zero condition channels
    noisy      (B, 16, 3, 60, 104)  cat render (B, 20, 3, 60, 104) -> 36 channels
    patchify   Conv3d(36 -> 1536, kernel = stride = (1, 2, 2)); 1560 tokens per latent frame
    prefill    (B, 9360, 1536) through 30 blocks -> KV (30, 2, B, 9360, 12, 128)
    denoise    (B, 4680, 1536) through 30 blocks, keys = [cached context | own] -> (B, 16, 3, 60, 104)

Rotary positions are 3D (frame, row, column) over a 22 / 21 / 21 split of the 64 rotation pairs.
Context frames take temporal positions 0.., the noisy block always takes positions 6..8 (upstream
`freqs_offset = 2 * num_frame_per_block`), also on the first block whose context is 3 frames.

Numerics follow the upstream inference path, where the whole pipeline is cast to bf16: weights,
residual stream, adaLN modulation and layer-norm outputs are bf16; RMS norms compute in float32
and round back; the rotary rotation runs in float32 (upstream: float64) and rounds to bf16.

Ref: InSpatio-World arXiv:2604.07209; Wan arXiv:2503.20314; CausVid arXiv:2412.07772
# Adapted from https://github.com/inspatio/inspatio-world-v1.5/blob/main/wan/modules/causal_model.py
# and https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, cast, override

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from ..types import (
    AttentionHeads,
    CrossAttentionCache,
    KVCache,
    LatentGrid,
    Modulation,
    RotaryTable,
    TextContext,
    TimeEmbedding,
    Timesteps,
    Tokens,
    VideoLatents,
)
from ..utils.hub import HubModule

NUM_BLOCK_MODULATIONS = 6  # shift, scale, gate for self-attention, then for the feed-forward
ROPE_MAX_POSITIONS = 1024  # longest axis (latent frames or patches) the rotary tables cover
ROPE_THETA = 10_000.0
TIMESTEP_MAX_PERIOD = 10_000.0

type ConditionedLatents = Float[Tensor, "B C G H W"]  # latents with the condition channels (36)
type Condition = Float[Tensor, "B C G H W"]  # render mask (4) + render latents (16)
type BlockKV = tuple[AttentionHeads, AttentionHeads]


@dataclass(frozen=True, slots=True)
class WanDiTConfig:
    """Architecture quantities of the causal Wan2.1-1.3B model; defaults are the release."""

    in_channels: int = 36  # 16 latent + 4 render mask + 16 render latent channels
    out_channels: int = 16
    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30
    freq_dim: int = 256  # sinusoidal timestep features
    text_dim: int = 4096  # umT5-XXL hidden size
    text_length: int = 512  # the fixed context length the model was trained with
    patch_size: tuple[int, int, int] = (1, 2, 2)
    denoise_frame_offset: int = 6  # rotary frame position of the noisy block's first frame
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


def sinusoidal_embedding(timesteps: Timesteps, dim: int) -> TimeEmbedding:
    """Embed timesteps as `[cos | sin]` over geometrically spaced frequencies, in float64."""

    half = dim // 2
    exponents = torch.arange(half, device=timesteps.device, dtype=torch.float64) / half
    angles = torch.outer(timesteps.double(), torch.pow(TIMESTEP_MAX_PERIOD, -exponents))
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class RMSNorm(nn.Module):
    """RMS normalization in float32 with a learned gain, returned in the input dtype (qk-norm)."""

    def __init__(self, dim: int, *, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: Tokens) -> Tokens:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class RotaryEmbedding3D(nn.Module):
    """Rotary position embedding over the `(frame, height, width)` token grid.

    A token's rotation angles are its frame index in the first 22 pairs of the head, its row in
    the next 21 and its column in the last 21, each axis with its own geometric frequency ladder.
    Angles are tabulated once in float64 and kept as float32 cos / sin buffers.
    """

    cos: RotaryTable
    sin: RotaryTable

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        spatial_pairs = head_dim // 6
        self.pairs_per_axis = (head_dim // 2 - 2 * spatial_pairs, spatial_pairs, spatial_pairs)

        positions = torch.arange(ROPE_MAX_POSITIONS, dtype=torch.float64)
        angles = torch.cat(
            [_rotary_angles(positions, 2 * pairs) for pairs in self.pairs_per_axis], dim=1
        )
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)

    @override
    def forward(self, grid: LatentGrid, frame_offset: int) -> tuple[RotaryTable, RotaryTable]:
        return (
            _grid_table(self.cos, grid, frame_offset, self.pairs_per_axis),
            _grid_table(self.sin, grid, frame_offset, self.pairs_per_axis),
        )


def _rotary_angles(positions: Tensor, dim: int) -> RotaryTable:
    frequencies = 1.0 / ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    return torch.outer(positions, frequencies)


def _grid_table(
    table: RotaryTable, grid: LatentGrid, frame_offset: int, pairs_per_axis: tuple[int, int, int]
) -> RotaryTable:
    # Broadcast each axis' angles over the other two, laid out in (f, h, w) raster order
    f, h, w = grid
    per_axis = torch.split(table, list(pairs_per_axis), dim=1)
    return torch.cat(
        [
            repeat(per_axis[0][frame_offset : frame_offset + f], "f d -> (f h w) d", h=h, w=w),
            repeat(per_axis[1][:h], "h d -> (f h w) d", f=f, w=w),
            repeat(per_axis[2][:w], "w d -> (f h w) d", f=f, h=h),
        ],
        dim=-1,
    )


def apply_rotary(x: AttentionHeads, cos: RotaryTable, sin: RotaryTable) -> AttentionHeads:
    """Rotate adjacent `(even, odd)` channel pairs of a `(B, N, heads, D)` tensor in float32."""

    pairs = rearrange(x.float(), "b n h (d two) -> b n h d two", two=2)
    x1, x2 = pairs.unbind(dim=-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rearrange(rotated, "b n h d two -> b n h (d two)").type_as(x)


def attention(query: AttentionHeads, key: AttentionHeads, value: AttentionHeads) -> Tokens:
    """Full (non-causal) multi-head attention; `(B, N, heads, D)` in, `(B, N, heads * D)` out."""

    attended = F.scaled_dot_product_attention(
        rearrange(query, "b n h d -> b h n d"),
        rearrange(key, "b n h d -> b h n d"),
        rearrange(value, "b n h d -> b h n d"),
    )
    return rearrange(attended, "b h n d -> b n (h d)")


class SelfAttention(nn.Module):
    """Rotary self-attention with RMS-normalized queries and keys and a cached-context prefix.

    The fused `qkv` projection holds upstream's separate `q`, `k`, `v` layers stacked by rows.
    `forward` returns the output together with this call's rotated keys and values, which is the
    block's contribution to the KV cache when the call is a prefill.
    """

    def __init__(self, dim: int, num_heads: int, *, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.query_norm = RMSNorm(dim, eps=eps)
        self.key_norm = RMSNorm(dim, eps=eps)

    @override
    def forward(
        self, x: Tokens, rotary: tuple[RotaryTable, RotaryTable], prefix: BlockKV | None
    ) -> tuple[Tokens, BlockKV]:
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query, key = self.query_norm(query), self.key_norm(key)
        query, key, value = (
            rearrange(t, "b n (h d) -> b n h d", h=self.num_heads) for t in (query, key, value)
        )
        query, key = apply_rotary(query, *rotary), apply_rotary(key, *rotary)

        # The denoise pass sees the cached context first, then its own tokens
        keys, values = key, value
        if prefix is not None:
            keys = torch.cat([prefix[0], key], dim=1)
            values = torch.cat([prefix[1], value], dim=1)
        return self.output(attention(query, keys, values)), (key, value)


class CrossAttention(nn.Module):
    """Text cross-attention; the text keys and values are projected once per prompt (`project`)."""

    def __init__(self, dim: int, num_heads: int, *, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(dim, dim)
        self.query_norm = RMSNorm(dim, eps=eps)
        self.key_norm = RMSNorm(dim, eps=eps)

    def project(self, context: TextContext) -> BlockKV:
        key, value = self.kv(context).chunk(2, dim=-1)
        key = self.key_norm(key)
        return (
            rearrange(key, "b l (h d) -> b l h d", h=self.num_heads),
            rearrange(value, "b l (h d) -> b l h d", h=self.num_heads),
        )

    @override
    def forward(self, x: Tokens, text_kv: BlockKV) -> Tokens:
        query = rearrange(self.query_norm(self.query(x)), "b n (h d) -> b n h d", h=self.num_heads)
        return self.output(attention(query, *text_kv))


class WanBlock(nn.Module):
    """One DiT block: adaLN self-attention, text cross-attention, adaLN feed-forward."""

    def __init__(self, config: WanDiTConfig) -> None:
        super().__init__()
        dim, eps = config.dim, config.eps
        self.self_attention = SelfAttention(dim, config.num_heads, eps=eps)
        self.cross_attention_norm = nn.LayerNorm(dim, eps=eps)
        self.cross_attention = CrossAttention(dim, config.num_heads, eps=eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, config.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_dim, dim),
        )
        # Per-block adaLN table added to the shared timestep modulation
        self.modulation = nn.Parameter(torch.randn(1, NUM_BLOCK_MODULATIONS, dim) / math.sqrt(dim))
        self.eps = eps

    @override
    def forward(
        self,
        tokens: Tokens,
        modulation: Modulation,
        rotary: tuple[RotaryTable, RotaryTable],
        text_kv: BlockKV,
        prefix: BlockKV | None,
    ) -> tuple[Tokens, BlockKV]:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (self.modulation + modulation).chunk(
            NUM_BLOCK_MODULATIONS, dim=1
        )
        dim = tokens.shape[-1]

        # 1. Self-attention over [context |] block tokens, modulated in and gated out
        normed = F.layer_norm(tokens, (dim,), eps=self.eps) * (1 + scale_a) + shift_a
        attended, kv = self.self_attention(normed, rotary, prefix)
        tokens = tokens + attended * gate_a

        # 2. Cross-attention to the text: normalized (affine), neither modulated nor gated
        tokens = tokens + self.cross_attention(self.cross_attention_norm(tokens), text_kv)

        # 3. Feed-forward, modulated and gated by the second half of the vectors
        normed = F.layer_norm(tokens, (dim,), eps=self.eps) * (1 + scale_f) + shift_f
        return tokens + self.feed_forward(normed) * gate_f, kv


class OutputHead(nn.Module):
    """adaLN (shift, scale from the time embedding) + linear projection back to latent patches."""

    def __init__(self, config: WanDiTConfig) -> None:
        super().__init__()
        self.eps = config.eps
        self.linear = nn.Linear(config.dim, math.prod(config.patch_size) * config.out_channels)
        self.modulation = nn.Parameter(torch.randn(1, 2, config.dim) / math.sqrt(config.dim))

    @override
    def forward(self, tokens: Tokens, time_embedding: TimeEmbedding) -> Tokens:
        shift, scale = (self.modulation + time_embedding[:, None]).chunk(2, dim=1)
        normed = F.layer_norm(tokens, (tokens.shape[-1],), eps=self.eps)
        return self.linear(normed * (1 + scale) + shift)


class CausalWanDiT(nn.Module, HubModule):
    """Causal Wan2.1-1.3B: `prefill` builds the per-block KV cache, `denoise` predicts the flow."""

    config_class: ClassVar[type] = WanDiTConfig

    def __init__(self, config: WanDiTConfig) -> None:
        super().__init__()
        dim = config.dim
        self.config = config

        self.patch_embedding = nn.Conv3d(
            config.in_channels, dim, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, NUM_BLOCK_MODULATIONS * dim))
        self.rotary = RotaryEmbedding3D(config.head_dim)
        self.blocks = nn.ModuleList([WanBlock(config) for _ in range(config.num_layers)])
        self.head = OutputHead(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    def encode_text(self, context: TextContext) -> CrossAttentionCache:
        """Project a padded umT5 context into every block's cross-attention keys and values."""

        embedded = self.text_embedding(context.to(self.dtype))
        blocks = cast(list[WanBlock], list(self.blocks))
        per_block = [torch.stack(block.cross_attention.project(embedded)) for block in blocks]
        return torch.stack(per_block)

    def prefill(self, latents: ConditionedLatents, text_kv: CrossAttentionCache) -> KVCache:
        """Run the clean context at timestep 0 and return every block's rotated keys and values."""

        tokens, grid = self._patchify(latents)
        rotary = self.rotary(grid, 0)
        timesteps = latents.new_zeros(latents.shape[0], dtype=torch.float32)
        _, modulation = self._time(timesteps)

        cache = []
        for index, block in enumerate(self.blocks):
            tokens, kv = block(tokens, modulation, rotary, _unstack(text_kv[index]), None)
            cache.append(torch.stack(kv))
        return torch.stack(cache)

    def denoise(
        self,
        noisy: VideoLatents,
        condition: Condition,
        timesteps: Timesteps,
        context_kv: KVCache,
        text_kv: CrossAttentionCache,
    ) -> VideoLatents:
        """Predict the flow `noise - x0` of a noisy block given its condition and the cache."""

        tokens, grid = self._patchify(torch.cat([noisy, condition], dim=1))
        rotary = self.rotary(grid, self.config.denoise_frame_offset)
        time_embedding, modulation = self._time(timesteps)

        for index, block in enumerate(self.blocks):
            prefix = _unstack(context_kv[index])
            tokens, _ = block(tokens, modulation, rotary, _unstack(text_kv[index]), prefix)
        return self._unpatchify(self.head(tokens, time_embedding), grid)

    def _patchify(self, latents: ConditionedLatents) -> tuple[Tokens, LatentGrid]:
        patches = self.patch_embedding(latents.to(self.dtype))
        _, _, f, h, w = patches.shape
        return rearrange(patches, "b d f h w -> b (f h w) d"), (f, h, w)

    def _unpatchify(self, tokens: Tokens, grid: LatentGrid) -> VideoLatents:
        f, h, w = grid
        pt, ph, pw = self.config.patch_size
        return rearrange(
            tokens,
            "b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)",
            f=f, h=h, w=w, pt=pt, ph=ph, pw=pw,
        )  # fmt: skip

    def _time(self, timesteps: Timesteps) -> tuple[TimeEmbedding, Modulation]:
        # Upstream rounds the float64 sinusoid to the model dtype before the MLP
        embedding = self.time_embedding(
            sinusoidal_embedding(timesteps, self.config.freq_dim).to(self.dtype)
        )
        modulation = self.time_modulation(embedding)
        return embedding, rearrange(modulation, "b (k d) -> b k d", k=NUM_BLOCK_MODULATIONS)


def _unstack(pair: Tensor) -> BlockKV:
    return pair[0], pair[1]
