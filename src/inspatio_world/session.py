"""A streaming generation: target cameras in, frames out, one block of 3 latents at a time.

The world model is causal, so a whole video equals the concatenation of its blocks as long as
every stream keeps its state between them. A `Session` holds that state:

    render encoder cache    the latent encoder over the splatted condition frames
    source encoder cache    the latent encoder over the source frames the block is anchored to
    decoder cache           the VAE (or TAEHV) decoder over the predicted latents
    previous prediction     the last block's clean latents, which join the next block's context
    noise generator         seeded once per session

Each `step` takes the target cameras of the next block (9 frames for the first block, whose
first latent is a single frame, then 12), renders the scene into them, encodes the condition,
denoises 3 latents and decodes them. Frames stream out with a latency of one block, which is what
makes camera control interactive.

For image scenes with several views, each latent is anchored to one source view: per block the
three views covering most of the rendered pixels are picked and assigned one per latent to
maximize coverage, as upstream's `chunk_top3`.

# Adapted from https://github.com/inspatio/inspatio-world-v1.5/blob/main/inference.py,
# pipeline/causal_inference.py and pipeline/render_scene.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Protocol

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import UInt8
from torch import Tensor

from .render.splat import LiftedViews, lift_views, render
from .sampler import FlowSchedule, denoise_block
from .scene import Scene
from .types import CrossAttentionCache, Frames, MaskLatents, Poses, RenderMasks, VideoLatents
from .wan import CausalCache, CausalWanDiT

FRAMES_PER_LATENT = 4
LATENTS_PER_BLOCK = 3
FIRST_BLOCK_FRAMES = 1 + FRAMES_PER_LATENT * (LATENTS_PER_BLOCK - 1)  # 9
BLOCK_FRAMES = FRAMES_PER_LATENT * LATENTS_PER_BLOCK  # 12
LATENT_SCALE = 8  # VAE spatial compression
CONDITION_CHANNELS = 20  # 4 render-mask + 16 render-latent channels, zero in the context
ANCHOR_VIEWS = 3  # multi-view scenes: distinct source views per block, one per latent

type FramesUInt8 = UInt8[Tensor, "F H W 3"]


class LatentEncoder(Protocol):
    """Frames in [-1, 1] -> normalized latents, streaming over a `CausalCache` (Wan VAE or TAEHV)."""

    def encode(self, frames: Frames, cache: CausalCache) -> VideoLatents: ...


class FrameDecoder(Protocol):
    """Latents -> frames in [-1, 1], streaming over a `CausalCache` (Wan VAE or TAEHV)."""

    def decode(self, latents: VideoLatents, cache: CausalCache) -> Frames: ...


@dataclass(frozen=True, slots=True)
class BlockOutput:
    frames: FramesUInt8  # the generated frames of the block, on the model device
    render: FramesUInt8  # the splatted condition the block was generated from


class Session:
    """The causal state of one generation over one scene."""

    def __init__(
        self,
        scene: Scene,
        dit: CausalWanDiT,
        encoder: LatentEncoder,
        decoder: FrameDecoder,
        schedule: FlowSchedule,
        text_kv: CrossAttentionCache,
        seed: int,
    ) -> None:
        self.scene, self.dit, self.encoder, self.decoder = scene, dit, encoder, decoder
        self.schedule, self.text_kv = schedule, text_kv
        self.device = text_kv.device
        self.generator = torch.Generator(self.device).manual_seed(seed)
        self.frame_index = 0
        self.render_cache, self.source_cache, self.decoder_cache = (
            CausalCache(), CausalCache(), CausalCache(),
        )  # fmt: skip
        self.previous: VideoLatents | None = None
        self.target_intrinsics = scene.intrinsics[0].to(self.device)
        # Image scenes lift their few views once; video views are lifted frame by frame
        self.image_views = (
            self._lift(torch.arange(scene.num_views)) if scene.kind == "image" else None
        )

    @property
    def frames_needed(self) -> int:
        """Target cameras the next `step` expects."""

        return FIRST_BLOCK_FRAMES if self.frame_index == 0 else BLOCK_FRAMES

    @property
    def finished(self) -> bool:
        limit = self.scene.max_frames
        return limit is not None and self.frame_index >= limit

    @torch.inference_mode()
    def step(self, world_to_camera: Poses) -> BlockOutput:
        """Generate the next block for these target cameras (`frames_needed` of them)."""

        count = world_to_camera.shape[0]
        assert count == self.frames_needed, f"block needs {self.frames_needed} cameras, got {count}"
        frames = torch.arange(self.frame_index, self.frame_index + count)
        rendered, mask, anchors = self._render(frames, world_to_camera.to(self.device))

        # Condition: render latents + downsampled mask; context: source (+ previous) latents
        render_latents = self.encoder.encode(_to_video(rendered), self.render_cache)
        source = self._source_frames(frames, anchors)
        source_latents = self.encoder.encode(_to_video(source), self.source_cache)
        condition = torch.cat(
            [self._mask_latents(mask, render_latents.dtype), render_latents], dim=1
        )
        context = [source_latents] + ([self.previous] if self.previous is not None else [])
        context = torch.cat(
            [F.pad(c, (0, 0, 0, 0, 0, 0, 0, CONDITION_CHANNELS)) for c in context], dim=2
        )

        noise = [self._noise(source_latents) for _ in self.schedule.timesteps]
        latents = denoise_block(self.dit, self.schedule, context, condition, self.text_kv, noise)
        self.previous = latents

        decoded = self.decoder.decode(latents, self.decoder_cache)
        self.frame_index += count
        valid = (
            count
            if self.scene.max_frames is None
            else min(count, self.scene.max_frames - int(frames[0]))
        )
        return BlockOutput(frames=_to_uint8(decoded)[:valid], render=rendered[:valid])

    def _lift(self, indices: Tensor) -> LiftedViews:
        scene = self.scene
        images = rearrange(scene.images[indices].to(self.device), "v h w c -> v c h w")
        return lift_views(
            images.float() / 127.5 - 1,
            scene.depth[indices].to(self.device),
            scene.intrinsics[indices].to(self.device),
            scene.world_to_camera[indices].to(self.device),
        )

    def _render(
        self, frames: Tensor, world_to_camera: Poses
    ) -> tuple[FramesUInt8, RenderMasks, list[int]]:
        if self.image_views is not None:
            views = self.image_views
            index = torch.arange(self.scene.num_views, device=self.device).expand(len(frames), -1)
        else:
            # Past the end of a video, the last source frame repeats (upstream pads the same way)
            views = self._lift(frames.clamp(max=self.scene.num_views - 1))
            index = torch.arange(len(frames), device=self.device)[:, None]
        result = render(views, index, world_to_camera, self.target_intrinsics)
        # Upstream stores the render as 8-bit video before encoding it
        rendered = ((result.rgb + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return (
            rearrange(rendered, "f c h w -> f h w c"),
            result.mask,
            self._anchor_views(result.view_pixels),
        )

    def _anchor_views(self, view_pixels: Tensor) -> list[int]:
        """The source view each latent of the block is anchored to (multi-view image scenes)."""

        if self.scene.kind == "video" or self.scene.num_views == 1:
            return [0] * LATENTS_PER_BLOCK
        assert self.scene.num_views >= ANCHOR_VIEWS, "multi-view scenes need at least 3 views"
        per_latent = [group.sum(dim=0).tolist() for group in self._latent_groups(view_pixels)]
        totals = [
            sum(scores[view] for scores in per_latent) for view in range(self.scene.num_views)
        ]
        top = sorted(range(self.scene.num_views), key=lambda view: (-totals[view], view))[
            :ANCHOR_VIEWS
        ]
        best = min(
            permutations(top),
            key=lambda views: (-sum(per_latent[i][view] for i, view in enumerate(views)), views),
        )
        return list(best)

    def _latent_groups(self, per_frame: Tensor) -> list[Tensor]:
        """Split a block's per-frame rows into its 3 latents (1 + 4 + 4 frames first, then 4s)."""

        sizes = (
            [1, FRAMES_PER_LATENT, FRAMES_PER_LATENT]
            if self.frame_index == 0
            else [FRAMES_PER_LATENT] * 3
        )
        return list(per_frame.split(sizes))

    def _source_frames(self, frames: Tensor, anchors: list[int]) -> FramesUInt8:
        if self.scene.kind == "video":
            index = frames.clamp(max=self.scene.num_views - 1)
        else:
            groups = self._latent_groups(frames)
            index = torch.cat(
                [
                    torch.full((len(group),), view)
                    for group, view in zip(groups, anchors, strict=True)
                ]
            )
        return self.scene.images[index].to(self.device)

    def _mask_latents(self, mask: RenderMasks, dtype: torch.dtype) -> MaskLatents:
        """Render validity in {-1, 1}, bilinearly downsampled; 4 frames per latent as channels."""

        signed = (mask.to(dtype) * 2 - 1)[:, None]
        h, w = mask.shape[1] // LATENT_SCALE, mask.shape[2] // LATENT_SCALE
        small = F.interpolate(signed, size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        if self.frame_index == 0:
            # The first latent covers one frame; upstream repeats its mask four times
            small = torch.cat([small[:1].expand(FRAMES_PER_LATENT, -1, -1), small[1:]])
        return rearrange(small, "(g c) h w -> 1 c g h w", c=FRAMES_PER_LATENT)

    def _noise(self, like: VideoLatents) -> VideoLatents:
        return torch.randn(like.shape, generator=self.generator, device=self.device).to(like.dtype)


def _to_video(frames: FramesUInt8) -> Frames:
    """uint8 frames -> the VAE's `(1, 3, F, H, W)` input in [-1, 1] (normalized in float32)."""

    return rearrange(frames.float() / 255 * 2 - 1, "f h w c -> 1 c f h w")


def _to_uint8(frames: Frames) -> FramesUInt8:
    pixels = (frames[0].float().clamp(-1, 1) * 0.5 + 0.5) * 255
    return rearrange(pixels.to(torch.uint8), "c f h w -> f h w c")
