"""Wan2.1 causal video VAE, run as streams: frames -> latents chunk by chunk, and back.

The encoder turns the first frame alone into one latent and every following group of 4 frames
into one more (4x temporal, 8x spatial, 16 channels); the decoder inverts that, one latent at a
time. Both are causal: each temporal convolution reads the last frames of the previous chunk from
a `CausalCache`, so a video encoded in chunks is identical to the video encoded at once, and the
world model can encode renders and decode predictions as they are produced.

    frames   (B, 3, 1 or 4, 480, 832) in [-1, 1]
    encode   conv_in 96 -> 3 stages x (2 residual blocks + downsample) -> 384 x 60 x 104
             -> middle (residual, attention, residual) -> head -> mean of the 2x16 channels
    latents  (B, 16, 1, 60, 104), normalized per channel by the release statistics

Decoding one latent at a time is part of the contract, not an implementation detail: upstream's
first temporal upsampler refreshes its cache with the current frame duplicated (`where(previous
== 0, 0, current)`), so the output depends on the chunking. It is reproduced as shipped.

Numerics: bf16 throughout, as upstream casts the whole pipeline; nearest upsampling runs in
float32 and the latent normalization uses bf16-rounded statistics, both as upstream.

Ref: Wan arXiv:2503.20314
# Adapted from https://github.com/inspatio/inspatio-world-v1.5/blob/main/wan/modules/vae.py
# (itself from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import ClassVar, override

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from ..types import Frames, LatentStatistics, VAEFeatures, VideoLatents
from ..utils.hub import HubModule

CACHE_FRAMES = 2  # a kernel-3 causal convolution needs the previous chunk's last two frames
FRAMES_PER_LATENT = 4  # temporal compression after the first frame

# Release latent statistics of the Wan2.1 VAE (per channel)
LATENT_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)  # fmt: skip
LATENT_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class WanVAEConfig:
    """Architecture quantities of the Wan2.1 VAE; defaults are the release."""

    base_dim: int = 96
    latent_channels: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temporal_downsample: tuple[bool, ...] = (False, True, True)


@dataclass(slots=True)
class CausalCache:
    """The previous chunk's tail for every causal layer, in call order.

    The first chunk appends one entry per layer (`None` reads mean zero padding); every later
    chunk walks the same slots again from `rewind()`.
    """

    slots: list[Tensor | None] = field(default_factory=list)
    cursor: int = 0

    def rewind(self) -> None:
        self.cursor = 0

    def read(self) -> Tensor | None:
        return self.slots[self.cursor] if self.cursor < len(self.slots) else None

    def write(self, tail: Tensor | None) -> None:
        if self.cursor < len(self.slots):
            self.slots[self.cursor] = tail
        else:
            self.slots.append(tail)
        self.cursor += 1


class CausalConv3d(nn.Conv3d):
    """3D convolution padded only towards the past; the past comes from the cache when given."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int, int] | int,
        stride: tuple[int, int, int] | int = 1,
        padding: tuple[int, int, int] | int = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel, stride=stride, padding=0)
        pt, ph, pw = (padding,) * 3 if isinstance(padding, int) else padding
        self.causal_padding = (pw, pw, ph, ph, 2 * pt, 0)

    @override
    def forward(self, x: VAEFeatures, past: Tensor | None = None) -> VAEFeatures:  # pyright: ignore[reportIncompatibleMethodOverride]
        padding = list(self.causal_padding)
        if past is not None and padding[4] > 0:
            x = torch.cat([past, x], dim=2)
            padding[4] -= past.shape[2]
        return super().forward(F.pad(x, padding))


def cached_conv(conv: CausalConv3d, x: VAEFeatures, cache: CausalCache) -> VAEFeatures:
    """Run a kernel-3 causal convolution on a chunk and leave its last two frames in the cache."""

    past = cache.read()
    tail = x[:, :, -CACHE_FRAMES:].clone()
    if tail.shape[2] < CACHE_FRAMES and past is not None:
        tail = torch.cat([past[:, :, -1:], tail], dim=2)
    cache.write(tail)
    return conv(x, past)


class RMSNorm3d(nn.Module):
    """Channel RMS norm (`F.normalize` over channels, times sqrt(C) and a learned gain)."""

    def __init__(self, dim: int, spatial_dims: int = 3) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, *([1] * spatial_dims)))

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    """Two causal 3x3x3 convolutions with RMS norm + SiLU, and a 1x1x1 shortcut when widening."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm3d(in_dim)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMSNorm3d(out_dim)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    @override
    def forward(self, x: VAEFeatures, cache: CausalCache) -> VAEFeatures:
        hidden = cached_conv(self.conv1, F.silu(self.norm1(x)), cache)
        hidden = cached_conv(self.conv2, F.silu(self.norm2(hidden)), cache)
        return hidden + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention within each frame."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm3d(dim, spatial_dims=2)
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    @override
    def forward(self, x: VAEFeatures, cache: CausalCache) -> VAEFeatures:
        b, _, t, h, w = x.shape
        frames = rearrange(x, "b c t h w -> (b t) c h w")
        qkv = rearrange(self.qkv(self.norm(frames)), "n (k c) h w -> k n 1 (h w) c", k=3)
        # Fused attention kernels need unit stride on the channel axis
        qkv = qkv.contiguous()
        attended = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        attended = rearrange(attended, "n 1 (h w) c -> n c h w", h=h, w=w)
        return x + rearrange(self.proj(attended), "(b t) c h w -> b c t h w", b=b, t=t)


class Downsample(nn.Module):
    """2x spatial stride-2 convolution, then (temporal) a stride-2 causal convolution over time.

    The temporal convolution skips the first chunk (one frame stays one frame) and afterwards
    reads the previous chunk's last frame, so 4 frames become 2 and then 1 across two stages.
    """

    def __init__(self, dim: int, temporal: bool) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(dim, dim, 3, stride=2)
        self.temporal = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1)) if temporal else None

    @override
    def forward(self, x: VAEFeatures, cache: CausalCache) -> VAEFeatures:
        t = x.shape[2]
        frames = F.pad(rearrange(x, "b c t h w -> (b t) c h w"), (0, 1, 0, 1))
        x = rearrange(self.spatial(frames), "(b t) c h w -> b c t h w", t=t)
        if self.temporal is None:
            return x

        past = cache.read()
        cache.write(x[:, :, -1:].clone())
        if past is None:
            return x
        return self.temporal(torch.cat([past[:, :, -1:], x], dim=2))


class Upsample(nn.Module):
    """(Temporal) a causal convolution doubling the frames, then 2x nearest + 3x3 convolution.

    The temporal convolution skips the first latent (one frame stays one frame). Its cache update
    for a single-frame chunk keeps upstream's rule: the new tail is `[0, x]` right after the first
    latent and `[x, x]` afterwards (`where(previous_last == 0, 0, x)` elementwise).
    """

    def __init__(self, dim: int, temporal: bool) -> None:
        super().__init__()
        self.temporal = (
            CausalConv3d(dim, 2 * dim, (3, 1, 1), padding=(1, 0, 0)) if temporal else None
        )
        self.spatial = nn.Conv2d(dim, dim // 2, 3, padding=1)

    @override
    def forward(self, x: VAEFeatures, cache: CausalCache) -> VAEFeatures:
        if self.temporal is not None:
            x = self._double_frames(x, cache)
        t = x.shape[2]
        frames = rearrange(x, "b c t h w -> (b t) c h w")
        frames = F.interpolate(frames.float(), scale_factor=2.0, mode="nearest").type_as(frames)
        return rearrange(self.spatial(frames), "(b t) c h w -> b c t h w", t=t)

    def _double_frames(self, x: VAEFeatures, cache: CausalCache) -> VAEFeatures:
        past = cache.read()
        if past is None:
            cache.write(torch.zeros_like(x[:, :, :1]).expand(-1, -1, CACHE_FRAMES, -1, -1))
            return x

        tail = x[:, :, -CACHE_FRAMES:].clone()
        if tail.shape[2] < CACHE_FRAMES:
            padding = torch.where(past[:, :, -1:] == 0, 0, tail)
            tail = torch.cat([padding, tail], dim=2)
        cache.write(tail)
        assert self.temporal is not None
        doubled = self.temporal(x, past)
        return rearrange(doubled, "b (two c) t h w -> b c (t two) h w", two=2)


class Encoder(nn.Module):
    """Frames -> (mean, log-variance) features; `conv_out` has twice the latent channels."""

    def __init__(self, config: WanVAEConfig) -> None:
        super().__init__()
        dims = [config.base_dim * mult for mult in (1, *config.dim_mult)]
        self.conv_in = CausalConv3d(3, dims[0], 3, padding=1)

        blocks: list[nn.Module] = []
        for stage, (in_dim, out_dim) in enumerate(pairwise(dims)):
            blocks.extend(
                ResidualBlock(in_dim if index == 0 else out_dim, out_dim)
                for index in range(config.num_res_blocks)
            )
            if stage < len(config.dim_mult) - 1:
                blocks.append(Downsample(out_dim, config.temporal_downsample[stage]))
        self.down = nn.ModuleList(blocks)

        width = dims[-1]
        self.middle = nn.ModuleList(
            [ResidualBlock(width, width), AttentionBlock(width), ResidualBlock(width, width)]
        )
        self.norm_out = RMSNorm3d(width)
        self.conv_out = CausalConv3d(width, 2 * config.latent_channels, 3, padding=1)

    @override
    def forward(self, frames: Frames, cache: CausalCache) -> VAEFeatures:
        x = cached_conv(self.conv_in, frames, cache)
        for layer in (*self.down, *self.middle):
            x = layer(x, cache)
        return cached_conv(self.conv_out, F.silu(self.norm_out(x)), cache)


class Decoder(nn.Module):
    """Latents -> frames in roughly [-1, 1]; each latent after the first becomes 4 frames."""

    def __init__(self, config: WanVAEConfig) -> None:
        super().__init__()
        mults = config.dim_mult
        dims = [config.base_dim * mult for mult in (mults[-1], *reversed(mults))]
        temporal_upsample = tuple(reversed(config.temporal_downsample))
        self.conv_in = CausalConv3d(config.latent_channels, dims[0], 3, padding=1)
        self.middle = nn.ModuleList(
            [ResidualBlock(dims[0], dims[0]), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0])]
        )  # fmt: skip

        blocks: list[nn.Module] = []
        for stage, (in_dim, out_dim) in enumerate(pairwise(dims)):
            # Every upsampler halves the channels, so later stages start at half width
            in_dim = in_dim // 2 if stage > 0 else in_dim
            blocks.extend(
                ResidualBlock(in_dim if index == 0 else out_dim, out_dim)
                for index in range(config.num_res_blocks + 1)
            )
            if stage < len(mults) - 1:
                blocks.append(Upsample(out_dim, temporal_upsample[stage]))
        self.up = nn.ModuleList(blocks)

        self.norm_out = RMSNorm3d(dims[-1])
        self.conv_out = CausalConv3d(dims[-1], 3, 3, padding=1)

    @override
    def forward(self, latents: VAEFeatures, cache: CausalCache) -> Frames:
        x = cached_conv(self.conv_in, latents, cache)
        for layer in (*self.middle, *self.up):
            x = layer(x, cache)
        return cached_conv(self.conv_out, F.silu(self.norm_out(x)), cache)


class WanVAE(nn.Module, HubModule):
    """Wan2.1 VAE with chunked causal `encode` and `decode` over explicit caches."""

    config_class: ClassVar[type] = WanVAEConfig

    latent_mean: LatentStatistics
    latent_std: LatentStatistics

    def __init__(self, config: WanVAEConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.latent_channels
        self.encoder = Encoder(config)
        self.encoder_projection = CausalConv3d(2 * channels, 2 * channels, 1)
        self.decoder_projection = CausalConv3d(channels, channels, 1)
        self.decoder = Decoder(config)
        mean = torch.tensor(LATENT_MEAN, dtype=torch.float32)
        std = torch.tensor(LATENT_STD, dtype=torch.float32)
        self.register_buffer("latent_mean", rearrange(mean, "c -> 1 c 1 1 1"), persistent=False)
        self.register_buffer("latent_std", rearrange(std, "c -> 1 c 1 1 1"), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder_projection.weight.dtype

    def encode(self, frames: Frames, cache: CausalCache) -> VideoLatents:
        """Encode the next frames of a stream: the first frame alone, then groups of 4."""

        latents = []
        start = 0
        while start < frames.shape[2]:
            size = 1 if not cache.slots else FRAMES_PER_LATENT
            latents.append(self.encode_chunk(frames[:, :, start : start + size], cache))
            start += size
        return torch.cat(latents, dim=2)

    def decode(self, latents: VideoLatents, cache: CausalCache) -> Frames:
        """Decode the next latents of a stream, one at a time (see module docstring), unclamped."""

        chunks = [self._decode_chunk(latent, cache) for latent in latents.split(1, dim=2)]
        return torch.cat(chunks, dim=2)

    def encode_chunk(self, frames: Frames, cache: CausalCache) -> VideoLatents:
        """Encode one chunk (the first frame, or 4 frames) into one normalized latent."""

        cache.rewind()
        features = self.encoder_projection(self.encoder(frames.to(self.dtype), cache))
        mean = features[:, : self.config.latent_channels]
        # Upstream normalizes with bf16-rounded statistics and a bf16 reciprocal
        return (mean - self.latent_mean.to(mean.dtype)) * (1 / self.latent_std.to(mean.dtype))

    def _decode_chunk(self, latent: VideoLatents, cache: CausalCache) -> Frames:
        cache.rewind()
        latent = latent.to(self.dtype)
        latent = latent / (1 / self.latent_std.to(latent.dtype)) + self.latent_mean.to(latent.dtype)
        return self.decoder(self.decoder_projection(latent), cache)
