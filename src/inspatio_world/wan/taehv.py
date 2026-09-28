"""TAEHV decoder for the Wan2.1 latent space: a ~10x cheaper stand-in for the VAE decoder.

`taew2_1` is a tiny convolutional autoencoder distilled to the Wan2.1 VAE's (normalized) latents.
Only its decoder is used here, to turn predicted latents into frames when latency matters more
than the last bit of fidelity. It is 2D convolutions per frame plus two kinds of temporal glue:

    MemBlock   conv(cat(x_t, x_{t-1})): each frame also sees the previous frame's input
    TGrow      a 1x1 conv producing `stride` frames per input frame (2x twice: 1 latent -> 4)

Within a chunk of latents every layer runs over all timesteps at once; across chunks each
MemBlock's last input is carried in a `CausalCache`, so decoding a stream chunk by chunk equals
decoding it whole. The first latent yields 4 frames of which the first 3 are start-up frames and
are dropped, matching the VAE's 1 + 4k frame layout.

Numerics: bf16 weights and activations; output clamped to [0, 1] and returned in [-1, 1].

# Adapted from https://github.com/madebyollin/taehv/blob/main/taehv.py (MIT)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, cast, override

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..types import FrameFeatures, Frames, VideoLatents
from ..utils.hub import HubModule
from .vae import CausalCache

STARTUP_FRAMES = 3  # frames the first latent decodes to before the VAE-aligned first frame


@dataclass(frozen=True, slots=True)
class TaehvConfig:
    """Architecture quantities of the `taew2_1` decoder."""

    latent_channels: int = 16
    widths: tuple[int, int, int, int] = (256, 128, 64, 64)
    blocks_per_stage: int = 3
    temporal_upscale: tuple[bool, bool, bool] = (False, True, True)


def _conv(in_channels: int, out_channels: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)


class MemBlock(nn.Module):
    """Residual 3-conv block over `cat(frame, previous frame)` (zeros before the first frame)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(2 * dim, dim), nn.ReLU(), _conv(dim, dim), nn.ReLU(), _conv(dim, dim)
        )

    @override
    def forward(self, x: FrameFeatures, batch: int, cache: CausalCache) -> FrameFeatures:
        frames = rearrange(x, "(b t) c h w -> b t c h w", b=batch)
        past = cache.read()
        cache.write(frames[:, -1:])
        if past is None:
            past = torch.zeros_like(frames[:, :1])
        # Each frame's memory is the previous frame of the stream
        memory = torch.cat([past, frames[:, :-1]], dim=1)
        memory = rearrange(memory, "b t c h w -> (b t) c h w")
        return F.relu(self.conv(torch.cat([x, memory], dim=1)) + x)


class TGrow(nn.Module):
    """Temporal upsampling: a 1x1 convolution emitting `stride` consecutive frames per frame."""

    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(dim, dim * stride, 1, bias=False)

    @override
    def forward(self, x: FrameFeatures) -> FrameFeatures:
        return rearrange(self.conv(x), "n (s c) h w -> (n s) c h w", s=self.stride)


class TaehvDecoder(nn.Module, HubModule):
    """`taew2_1` decoder with a streaming `decode(latents, cache)` over chunks of latents."""

    config_class: ClassVar[type] = TaehvConfig

    def __init__(self, config: TaehvConfig) -> None:
        super().__init__()
        self.config = config
        widths = config.widths
        layers: list[nn.Module] = [_conv(config.latent_channels, widths[0]), nn.ReLU()]
        for stage in range(3):
            layers += [MemBlock(widths[stage]) for _ in range(config.blocks_per_stage)]
            layers += [
                nn.Upsample(scale_factor=2),
                TGrow(widths[stage], 2 if config.temporal_upscale[stage] else 1),
                _conv(widths[stage], widths[stage + 1], bias=False),
            ]
        layers += [nn.ReLU(), _conv(widths[3], 3)]
        self.layers = nn.ModuleList(layers)

    @property
    def dtype(self) -> torch.dtype:
        return cast(nn.Conv2d, self.layers[0]).weight.dtype

    def decode(self, latents: VideoLatents, cache: CausalCache) -> Frames:
        """Decode the next latents of a stream into frames in [-1, 1] (1 + 4k layout)."""

        cache.rewind()
        first_chunk = not cache.slots
        b = latents.shape[0]
        # Upstream's input clamp: a soft tanh squashing to [-3, 3]
        x = torch.tanh(latents.to(self.dtype) / 3) * 3
        x = rearrange(x, "b c t h w -> (b t) c h w")
        for layer in self.layers:
            x = layer(x, b, cache) if isinstance(layer, MemBlock) else layer(x)

        frames = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        if first_chunk:
            frames = frames[:, :, STARTUP_FRAMES:]
        return frames.clamp(0, 1) * 2 - 1
