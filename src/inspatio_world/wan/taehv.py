"""TAEHV for the Wan2.1 latent space: a ~10x cheaper stand-in for the VAE encoder and decoder.

`taew2_1` is a tiny convolutional autoencoder distilled to the Wan2.1 VAE's (normalized) latents,
used when latency matters more than the last bit of fidelity. Both halves are 2D convolutions per
frame plus three kinds of temporal glue:

    MemBlock   conv(cat(x_t, x_{t-1})): each frame also sees the previous frame's input
    TPool      a 1x1 conv over `stride` consecutive frames stacked as channels (2x twice: 4 -> 1)
    TGrow      a 1x1 conv producing `stride` frames per input frame (2x twice: 1 latent -> 4)

    encode   frames (B, 3, 4k, 480, 832) -> conv 64 -> 3 x (TPool, stride-2 conv, 3 MemBlocks)
             -> conv 16 -> latents (B, 16, k, 60, 104)
    decode   latents -> conv 256 -> 3 x (3 MemBlocks, 2x upsample, TGrow, conv) -> RGB

Within a chunk every layer runs over all timesteps at once; across chunks each MemBlock's last
input is carried in a `CausalCache`, so a stream coded chunk by chunk equals the stream coded
whole. TAEHV has no single-frame first latent: the decoder's first latent yields 4 frames of which
the first 3 are start-up frames and are dropped, matching the VAE's 1 + 4k frame layout. The
encoder's latents sit 3 frames later in time than the VAE's; `encode` realigns them (a stream
aligned like the VAE's makes the world model follow the camera 3 frames late).

Encoder chunks follow the VAE's stream: 1 + 4k frames first, then multiples of 4.

Numerics: bf16 weights and activations; frames in [-1, 1] at both ends (the network's own [0, 1]
is converted at the boundary; decoded frames are clamped).

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

FRAMES_PER_LATENT = 4
STARTUP_FRAMES = FRAMES_PER_LATENT - 1  # padding frames before the VAE-aligned first frame


@dataclass(frozen=True, slots=True)
class TaehvConfig:
    """Architecture quantities of `taew2_1`."""

    latent_channels: int = 16
    encoder_width: int = 64
    temporal_downscale: tuple[bool, bool, bool] = (True, True, False)
    widths: tuple[int, int, int, int] = (256, 128, 64, 64)
    blocks_per_stage: int = 3
    temporal_upscale: tuple[bool, bool, bool] = (False, True, True)


def _conv(in_channels: int, out_channels: int, stride: int = 1, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)


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


class TPool(nn.Module):
    """Temporal downsampling: a 1x1 convolution over `stride` consecutive frames as channels."""

    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(dim * stride, dim, 1, bias=False)

    @override
    def forward(self, x: FrameFeatures) -> FrameFeatures:
        return self.conv(rearrange(x, "(n s) c h w -> n (s c) h w", s=self.stride))


class TGrow(nn.Module):
    """Temporal upsampling: a 1x1 convolution emitting `stride` consecutive frames per frame."""

    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(dim, dim * stride, 1, bias=False)

    @override
    def forward(self, x: FrameFeatures) -> FrameFeatures:
        return rearrange(self.conv(x), "n (s c) h w -> (n s) c h w", s=self.stride)


def _run(layers: nn.ModuleList, x: FrameFeatures, batch: int, cache: CausalCache) -> FrameFeatures:
    for layer in layers:
        x = layer(x, batch, cache) if isinstance(layer, MemBlock) else layer(x)
    return x


class Taehv(nn.Module, HubModule):
    """`taew2_1` with streaming `encode(frames, cache)` and `decode(latents, cache)`."""

    config_class: ClassVar[type] = TaehvConfig

    def __init__(self, config: TaehvConfig) -> None:
        super().__init__()
        self.config = config
        blocks = config.blocks_per_stage

        width = config.encoder_width
        encoder: list[nn.Module] = [_conv(3, width), nn.ReLU()]
        for pooled in config.temporal_downscale:
            encoder += [TPool(width, 2 if pooled else 1), _conv(width, width, 2, bias=False)]
            encoder += [MemBlock(width) for _ in range(blocks)]
        encoder.append(_conv(width, config.latent_channels))
        self.encoder = nn.ModuleList(encoder)

        widths = config.widths
        decoder: list[nn.Module] = [_conv(config.latent_channels, widths[0]), nn.ReLU()]
        for stage, grown in enumerate(config.temporal_upscale):
            decoder += [MemBlock(widths[stage]) for _ in range(blocks)]
            decoder += [
                nn.Upsample(scale_factor=2),
                TGrow(widths[stage], 2 if grown else 1),
                _conv(widths[stage], widths[stage + 1], bias=False),
            ]
        decoder += [nn.ReLU(), _conv(widths[3], 3)]
        self.decoder = nn.ModuleList(decoder)

    @property
    def dtype(self) -> torch.dtype:
        return cast(nn.Conv2d, self.decoder[0]).weight.dtype

    def encode(self, frames: Frames, cache: CausalCache) -> VideoLatents:
        """Encode the next frames of a stream (1 + 4k frames in, 1 + k latents out overall).

        The Wan VAE's latent k covers frames 4k-3..4k; TAEHV's matches it when it reads frames
        4k..4k+3, 3 frames ahead. The last latent of a chunk therefore needs 3 frames that do not
        exist yet: it is encoded from the chunk's last frame held 4 times, on a copy of the cache,
        and re-encoded (and discarded) with the real frames at the next chunk, so the cached
        stream only ever holds real frames. Cache slot 0 carries that last frame.
        """

        cache.rewind()
        pending = cache.read()
        stream = frames if pending is None else torch.cat([pending, frames], dim=2)
        cache.write(stream[:, :, -1:])

        # Complete groups of 4 are committed to the cache, the look-ahead group runs on a copy
        committed = self._encode_frames(stream[:, :, :-1], cache)
        held = stream[:, :, -1:].expand(-1, -1, FRAMES_PER_LATENT, -1, -1)
        lookahead = self._encode_frames(held, CausalCache(list(cache.slots), cursor=1))
        if pending is not None:
            committed = committed[:, :, 1:]  # the previous chunk's look-ahead latent, redone
        return torch.cat([committed, lookahead], dim=2)

    def _encode_frames(self, frames: Frames, cache: CausalCache) -> VideoLatents:
        b = frames.shape[0]
        x = rearrange(frames.to(self.dtype) * 0.5 + 0.5, "b c t h w -> (b t) c h w")
        x = _run(self.encoder, x, b, cache)
        return rearrange(x, "(b t) c h w -> b c t h w", b=b)

    def decode(self, latents: VideoLatents, cache: CausalCache) -> Frames:
        """Decode the next latents of a stream into frames in [-1, 1] (1 + 4k layout)."""

        cache.rewind()
        first_chunk = not cache.slots
        b = latents.shape[0]
        # Upstream's input clamp: a soft tanh squashing to [-3, 3]
        x = torch.tanh(latents.to(self.dtype) / 3) * 3
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = _run(self.decoder, x, b, cache)

        frames = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        if first_chunk:
            frames = frames[:, :, STARTUP_FRAMES:]
        return frames.clamp(0, 1) * 2 - 1
