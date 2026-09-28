"""Frozen dataclass configs: quantities and the one checkpoint choice, never code paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WEIGHTS_REPO = "blanchon/inspatio-world-v1.5"

# The Wan2.1 VAE decoder as trained with, or TAEHV's distilled decoder (~20x cheaper, softer)
DecoderKind = Literal["wan", "taehv"]
# DiT linear layers in bf16 as released, or float8 (dynamic per-row activation and weight scales,
# torchao; the `fp8` extra) on GPUs with fp8 tensor cores (Hopper, Ada, Blackwell)
DiTPrecision = Literal["bf16", "fp8"]


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Which weights to load, where to run them, and the sampling schedule."""

    repo_id: str = WEIGHTS_REPO
    revision: str | None = None
    device: str = "cuda"
    decoder: DecoderKind = "wan"
    dit_precision: DiTPrecision = "bf16"
    # The 5.7B text encoder only runs once per prompt; "cpu" keeps its 11 GB off the GPU
    text_encoder_device: str = "cuda"
    # Regional torch.compile of the DiT blocks and VAE encoder stages (~1.4x; the first blocks of
    # each shape compile for a few minutes, cached on disk by inductor afterwards)
    compile: bool = False
    # Indices into the 1000-step shifted schedule; the release is distilled for these four
    denoising_steps: tuple[int, ...] = (1000, 750, 500, 250)
    timestep_shift: float = 5.0
