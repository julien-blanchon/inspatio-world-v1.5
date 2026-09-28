"""Ahead-of-time compilation of the per-block hot path for ZeroGPU.

ZeroGPU runs every `@spaces.GPU` call in a fresh worker, so `torch.compile`'s JIT cache would be
lost after each call; ahead-of-time (AoTI) compilation produces a shared library once at startup
that every later call reuses (https://huggingface.co/blog/zerogpu-aoti).

Only the DiT shapes that repeat every block are compiled; the first block of a session
(3-frame context) keeps the eager path:

    prefill   DiT over the 6-frame context (source block + previous prediction)
    denoise   DiT denoising step of a 3-frame block against the 6-frame cache

The VAE encoder stays eager: its AoT graph (cache tails in and out) returned corrupted tails
(measured: the generation stops following the camera), so it was dropped.

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
from inspatio_world.wan import CausalWanDiT

logger = logging.getLogger(__name__)

STEADY_CONTEXT_FRAMES = 6
LATENT_HEIGHT, LATENT_WIDTH = 60, 104
STEADY_CONTEXT_TOKENS = STEADY_CONTEXT_FRAMES * (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)
# Pick Triton configs by heuristics instead of benchmarking them while compiling: the
# benchmark's CUDA event timing fails inside ZeroGPU's compile call
INDUCTOR_CONFIGS = {"triton.autotune_at_compile_time": False}


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


def example_inputs(world: WorldModel) -> dict[str, tuple[Any, ...]]:
    """Real-shaped inputs of the two steady-state DiT calls (random values)."""

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

    return {
        "prefill": (context, text_kv),
        "denoise": (noisy, condition, timesteps, context_kv, text_kv),
    }


def compile_graphs(world: WorldModel) -> dict[str, Any]:
    """Export and AoT-compile both graphs (call inside `@spaces.GPU`)."""

    modules = {
        "prefill": Prefill(world.dit),
        "denoise": Denoise(world.dit),
    }
    compiled = {}
    for name, args in example_inputs(world).items():
        with torch.no_grad():
            exported = torch.export.export(modules[name], args=args)
        compiled[name] = spaces.aoti_compile(exported, INDUCTOR_CONFIGS)
        logger.info("compiled %s", name)
    return compiled


def install(world: WorldModel, compiled: dict[str, Any]) -> None:
    """Route the steady-state shapes to the compiled graphs, everything else to eager code."""

    dit = world.dit
    eager_prefill, eager_denoise = dit.prefill, dit.denoise

    def prefill(latents: Tensor, text_kv: Tensor) -> Tensor:
        if latents.shape[2] == STEADY_CONTEXT_FRAMES:
            return compiled["prefill"](latents, text_kv)
        return eager_prefill(latents, text_kv)

    def denoise(
        noisy: Tensor, condition: Tensor, timesteps: Tensor, context_kv: Tensor, text_kv: Tensor
    ) -> Tensor:
        if context_kv.shape[3] == STEADY_CONTEXT_TOKENS:
            return compiled["denoise"](noisy, condition, timesteps, context_kv, text_kv)
        return eager_denoise(noisy, condition, timesteps, context_kv, text_kv)

    dit.prefill, dit.denoise = prefill, denoise  # type: ignore[method-assign]
