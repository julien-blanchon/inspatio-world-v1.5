"""Few-step flow-matching sampler for one autoregressive block.

The released model is distilled (Self-Forcing style) to denoise a block in 4 steps. Each step
predicts the flow `noise - x0`, converts it to a clean estimate `x0 = x_t - sigma_t * flow`, and,
except after the last step, re-noises that estimate to the next (lower) noise level with fresh
Gaussian noise. The schedule is the shifted flow-matching schedule of the training setup:

    sigma(s) = shift * s / (1 + (shift - 1) * s)  over s = 1, 0.999, ..., 0.001 (1000 steps)

and the denoising steps pick entries 0, 250, 500, 750 of it ("warped" steps), i.e. timesteps
1000, 937.5, 833.3, 625 for shift 5. Before the steps, the block's clean context is prefilled
into the KV cache once.

Numerics as upstream: schedule tables in float32, the x0 conversion in float64, the re-noising
mix in float32; latents are rounded to the model dtype between steps.

Ref: InSpatio-World arXiv:2604.07209; Self-Forcing arXiv:2506.08009
# Adapted from https://github.com/inspatio/inspatio-world-v1.5/blob/main/pipeline/causal_inference.py
# and pipeline/scheduler.py, pipeline/wan_wrapper.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import CrossAttentionCache, KVCache, VideoLatents
from .wan.dit import CausalWanDiT, Condition, ConditionedLatents

TRAIN_TIMESTEPS = 1000


@dataclass(frozen=True, slots=True)
class FlowSchedule:
    """The (timestep, sigma) pairs of the denoising steps, highest noise first."""

    timesteps: tuple[float, ...]
    sigmas: tuple[float, ...]

    @classmethod
    def warped(cls, step_indices: tuple[int, ...], shift: float) -> FlowSchedule:
        """Pick `TRAIN_TIMESTEPS - step` entries of the shifted schedule, as upstream warps them."""

        levels = torch.linspace(1.0, 0.0, TRAIN_TIMESTEPS + 1)[:-1]
        sigmas = shift * levels / (1 + (shift - 1) * levels)
        timesteps = sigmas * TRAIN_TIMESTEPS
        indices = [TRAIN_TIMESTEPS - step for step in step_indices]
        return cls(
            timesteps=tuple(timesteps[indices].tolist()),
            sigmas=tuple(sigmas[indices].tolist()),
        )


def denoise_block(
    dit: CausalWanDiT,
    schedule: FlowSchedule,
    context: ConditionedLatents,
    condition: Condition,
    text_kv: CrossAttentionCache,
    noise: list[VideoLatents],
) -> VideoLatents:
    """Prefill the context, then run the schedule on `noise[0]`; `noise[1:]` re-noise the steps."""

    context_kv: KVCache = dit.prefill(context, text_kv)
    noisy = noise[0]
    batch = noisy.shape[0]
    clean = noisy
    for step, (timestep, sigma) in enumerate(zip(schedule.timesteps, schedule.sigmas, strict=True)):
        timesteps = torch.full((batch,), timestep, device=noisy.device, dtype=torch.float32)
        flow = dit.denoise(noisy, condition, timesteps, context_kv, text_kv)
        clean = (noisy.double() - sigma * flow.double()).to(noisy.dtype)
        if step + 1 < len(schedule.sigmas):
            next_sigma, fresh = schedule.sigmas[step + 1], noise[step + 1]
            noisy = ((1 - next_sigma) * clean.float() + next_sigma * fresh.float()).to(fresh.dtype)
    return clean
