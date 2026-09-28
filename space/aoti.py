"""Ahead-of-time compilation of the per-block hot path for ZeroGPU.

ZeroGPU runs every `@spaces.GPU` call in a fresh worker, so `torch.compile`'s JIT cache would be
lost after each call; ahead-of-time (AoTI) compilation produces a shared library once at startup
that every later call reuses (https://huggingface.co/blog/zerogpu-aoti).

Only the shapes that repeat every block are compiled; the first block of a session (3-frame
context, 1-frame encoder chunk) keeps the eager path:

    prefill   DiT over the 6-frame context (source block + previous prediction)
    denoise   DiT denoising step of a 3-frame block against the 6-frame cache
    encode    VAE encoder over one steady-state 4-frame chunk, cache tensors in and out

Each graph is exported from a thin wrapper module, compiled with `spaces.aoti_compile`, and
dispatched by shape from the world model's methods (`install`). If compilation fails the Space
keeps running eagerly.
"""

from __future__ import annotations

import logging
from typing import Any

import spaces
import torch
from torch import Tensor, nn

from inspatio_world import WorldModel
from inspatio_world.wan import CausalCache, CausalWanDiT, WanVAE

logger = logging.getLogger(__name__)

STEADY_CONTEXT_FRAMES = 6
LATENT_HEIGHT, LATENT_WIDTH = 60, 104
FRAMES_PER_CHUNK = 4
STEADY_TAIL_FRAMES = 2  # encoder cache tails hold 2 frames from the third chunk on


class Prefill(nn.Module):
    def __init__(self, dit: CausalWanDiT) -> None:
        super().__init__()
        self.dit = dit

    def forward(self, latents: Tensor, text_kv: Tensor) -> Tensor:
        return self.dit.prefill(latents, text_kv)


class Denoise(nn.Module):
    def __init__(self, dit: CausalWanDiT) -> None:
        super().__init__()
        self.dit = dit

    def forward(
        self,
        noisy: Tensor,
        condition: Tensor,
        timesteps: Tensor,
        context_kv: Tensor,
        text_kv: Tensor,
    ) -> Tensor:
        return self.dit.denoise(noisy, condition, timesteps, context_kv, text_kv)


class EncodeChunk(nn.Module):
    def __init__(self, vae: WanVAE) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, frames: Tensor, slots: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
        cache = CausalCache(list(slots))
        latent = self.vae.encode_chunk(frames, cache)
        return latent, [slot for slot in cache.slots if slot is not None]


def example_inputs(world: WorldModel) -> dict[str, tuple[Any, ...]]:
    """Real-shaped inputs of the three steady-state calls (random values)."""

    device, dtype = world.device, world.dit.dtype
    # Export traces outside inference mode: inputs must be ordinary tensors
    text_kv = world.encode_prompt("").clone()
    context = torch.randn(
        1, 36, STEADY_CONTEXT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, device=device, dtype=dtype
    )
    with torch.no_grad():
        context_kv = world.dit.prefill(context, text_kv)
    noisy = torch.randn(1, 16, 3, LATENT_HEIGHT, LATENT_WIDTH, device=device, dtype=dtype)
    condition = torch.randn(1, 20, 3, LATENT_HEIGHT, LATENT_WIDTH, device=device, dtype=dtype)
    timesteps = torch.full((1,), 937.5, device=device)

    # The cache tails reach their steady length (2 frames) after the first two chunks
    frames = (
        torch.rand(1, 3, 1 + 2 * FRAMES_PER_CHUNK, 480, 832, device=device, dtype=dtype) * 2 - 1
    )
    cache = CausalCache()
    with torch.no_grad():
        world.vae.encode(frames[:, :, : 1 + FRAMES_PER_CHUNK], cache)
    slots = [slot for slot in cache.slots if slot is not None]
    assert len(slots) == len(cache.slots), "the encoder writes a tensor to every cache slot"
    return {
        "prefill": (context, text_kv),
        "denoise": (noisy, condition, timesteps, context_kv, text_kv),
        "encode": (frames[:, :, 1 + FRAMES_PER_CHUNK :], slots),
    }


def compile_graphs(world: WorldModel) -> dict[str, Any]:
    """Export and AoT-compile the three graphs (call inside `@spaces.GPU`)."""

    modules = {
        "prefill": Prefill(world.dit),
        "denoise": Denoise(world.dit),
        "encode": EncodeChunk(world.vae),
    }
    compiled = {}
    for name, args in example_inputs(world).items():
        with torch.no_grad():
            exported = torch.export.export(modules[name], args=args)
        compiled[name] = spaces.aoti_compile(exported)
        logger.info("compiled %s", name)
    return compiled


def install(world: WorldModel, compiled: dict[str, Any]) -> None:
    """Route the steady-state shapes to the compiled graphs, everything else to eager code."""

    dit, vae = world.dit, world.vae
    eager_prefill, eager_denoise, eager_encode = dit.prefill, dit.denoise, vae.encode_chunk

    def prefill(latents: Tensor, text_kv: Tensor) -> Tensor:
        if latents.shape[2] == STEADY_CONTEXT_FRAMES:
            return compiled["prefill"](latents, text_kv)
        return eager_prefill(latents, text_kv)

    def denoise(
        noisy: Tensor, condition: Tensor, timesteps: Tensor, context_kv: Tensor, text_kv: Tensor
    ) -> Tensor:
        if context_kv.shape[3] == STEADY_CONTEXT_FRAMES * (LATENT_HEIGHT // 2) * (
            LATENT_WIDTH // 2
        ):
            return compiled["denoise"](noisy, condition, timesteps, context_kv, text_kv)
        return eager_denoise(noisy, condition, timesteps, context_kv, text_kv)

    def encode_chunk(frames: Tensor, cache: CausalCache) -> Tensor:
        first = cache.slots[0] if cache.slots else None
        steady = first is not None and first.shape[2] == STEADY_TAIL_FRAMES
        if frames.shape[2] != FRAMES_PER_CHUNK or not steady:
            return eager_encode(frames, cache)
        latent, slots = compiled["encode"](frames, cache.slots)
        cache.slots[:] = slots
        return latent

    dit.prefill, dit.denoise, vae.encode_chunk = prefill, denoise, encode_chunk  # type: ignore[method-assign]
