"""Ahead-of-time compilation of the DiT block for ZeroGPU, cached on the Hub.

ZeroGPU runs every `@spaces.GPU` call in a fresh worker, so `torch.compile`'s JIT cache would be
lost after each call; ahead-of-time (AoTI) compilation produces a shared library once that every
later call reuses (https://huggingface.co/blog/zerogpu-aoti).

Compilation is regional: one `WanBlock` is exported and compiled for the two per-block calls
that repeat every generated block, and the package is shared by all 30 blocks, each passing its
own weights (`LazyAOTIModel.with_weights`, as `spaces.aoti_blocks_load` does):

    prefill   the block over the 6-frame context (source block + previous prediction)
    denoise   the block denoising 3 frames against the 6-frame cache

The first block of a session (3-frame context) keeps the eager path. The VAE encoder stays eager:
its AoT graph (cache tails in and out) returned corrupted tails, so it was dropped.

Packages are cached in a Hub repository under a key of torch version, GPU architecture, DiT
precision and a hash of the DiT source, so only the first start on new hardware or code pays
the compilation; uploading needs an `HF_TOKEN` secret with write access.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import spaces
import torch
from huggingface_hub import HfApi, hf_hub_download
from spaces.zero.torch.aoti import LazyAOTIModel
from torch import Tensor, nn
from torch._functorch._aot_autograd.subclass_parametrization import (
    unwrap_tensor_subclass_parameters,
)

import inspatio_world.wan.dit
from inspatio_world import WorldModel

logger = logging.getLogger(__name__)

CACHE_REPO = os.environ.get("INSPATIO_AOTI_REPO", "blanchon/inspatio-world-v1.5-aoti")
GRAPHS = ("prefill", "denoise")
STEADY_CONTEXT_FRAMES = 6
LATENT_HEIGHT, LATENT_WIDTH = 60, 104
STEADY_CONTEXT_TOKENS = STEADY_CONTEXT_FRAMES * (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)


def cache_key(world: WorldModel) -> str:
    """Packages are only valid for one torch build, GPU architecture, precision and DiT code."""

    major, minor = torch.cuda.get_device_capability()
    source = inspect.getsource(inspatio_world.wan.dit).encode()
    code = hashlib.sha256(source).hexdigest()[:10]
    version = torch.__version__.replace("+", "-")
    return f"torch{version}-sm{major}{minor}-{world.config.dit_precision}-{code}"


def load_or_compile(world: WorldModel) -> dict[str, bytes]:
    """The block packages from the Hub cache, compiled and uploaded first if missing.

    Call inside `@spaces.GPU`: the key reads the GPU architecture and compiling needs the GPU.
    """

    key = cache_key(world)
    try:
        packages = {name: _download(f"{key}/{name}/package.pt2") for name in GRAPHS}
        logger.info("loaded AoT packages %s", key)
        return packages
    except Exception:  # not cached yet (or the Hub is unreachable): compile
        logger.info("no AoT packages for %s, compiling", key)

    packages = compile_packages(world)
    try:
        api = HfApi()
        api.create_repo(CACHE_REPO, exist_ok=True)
        for name, package in packages.items():
            api.upload_file(
                path_or_fileobj=package,
                path_in_repo=f"{key}/{name}/package.pt2",
                repo_id=CACHE_REPO,
                commit_message=f"AoT package {key}/{name}",
            )
    except Exception:  # a missing or read-only token must not break the demo
        logger.exception("could not upload the AoT packages")
    return packages


def compile_packages(world: WorldModel) -> dict[str, bytes]:
    """Export and AoT-compile one block for the steady-state prefill and denoise calls."""

    _synchronized_event_timing()
    block = _unwrapped(world.dit.blocks[0])
    packages = {}
    for name, (args, kwargs) in _block_calls(world).items():
        with torch.no_grad():
            exported = torch.export.export(block, args=args, kwargs=kwargs)
        archive = spaces.aoti_compile(exported).archive_file
        assert isinstance(archive, BytesIO), "spaces.aoti_compile packages into memory"
        packages[name] = archive.getvalue()
        logger.info("compiled block %s", name)
    return packages


def install(world: WorldModel, packages: dict[str, bytes]) -> None:
    """Route every block's steady-state calls to the shared packages, the rest to eager code."""

    # Each worker process loads the package from a file path (a stream reads only once)
    folder = Path(tempfile.mkdtemp(prefix="inspatio-aoti-"))
    models = {}
    for name, package in packages.items():
        path = folder / f"{name}.pt2"
        path.write_bytes(package)
        models[name] = LazyAOTIModel(str(path))
    for block in world.dit.blocks:
        weights = _unwrapped(block).state_dict()
        prefill = models["prefill"].with_weights(weights)
        denoise = models["denoise"].with_weights(weights)
        block.forward = _dispatch(block.forward, prefill, denoise)  # type: ignore[method-assign]


def _dispatch(eager: Any, prefill: Any, denoise: Any) -> Any:
    def forward(
        tokens: Tensor, modulation: Tensor, rotary: Any, text_kv: Any, prefix: Any
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if prefix is None and tokens.shape[1] == STEADY_CONTEXT_TOKENS:
            return prefill(tokens, modulation, rotary, text_kv, None)
        if prefix is not None and prefix[0].shape[1] == STEADY_CONTEXT_TOKENS:
            return denoise(tokens, modulation, rotary, text_kv, prefix)
        return eager(tokens, modulation, rotary, text_kv, prefix)

    return forward


def _block_calls(world: WorldModel) -> dict[str, tuple[tuple[Any, ...], dict[str, Any]]]:
    """Capture block 0's arguments in a steady-state prefill and denoise (random latents)."""

    device, dtype = world.device, world.dit.dtype
    # Export traces outside inference mode: inputs must be ordinary tensors
    text_kv = world.encode_prompt("").clone()
    context = torch.randn(
        1, 36, STEADY_CONTEXT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, device=device, dtype=dtype
    )
    noisy = torch.randn(1, 16, 3, LATENT_HEIGHT, LATENT_WIDTH, device=device, dtype=dtype)
    condition = torch.randn(1, 20, 3, LATENT_HEIGHT, LATENT_WIDTH, device=device, dtype=dtype)
    timesteps = torch.full((1,), 937.5, device=device)

    calls = {}
    with torch.no_grad():
        context_kv = world.dit.prefill(context, text_kv)
        with spaces.aoti_capture(world.dit.blocks[0]) as call:
            world.dit.prefill(context, text_kv)
        calls["prefill"] = (call.args, call.kwargs)
        with spaces.aoti_capture(world.dit.blocks[0]) as call:
            world.dit.denoise(noisy, condition, timesteps, context_kv, text_kv)
        calls["denoise"] = (call.args, call.kwargs)
    return calls


def _unwrapped(module: nn.Module) -> nn.Module:
    """A shallow copy whose tensor-subclass (fp8) parameters are plain tensors, for AoTI."""

    clone = _shallow_clone(module)
    unwrap_tensor_subclass_parameters(clone)
    return clone


def _shallow_clone(module: nn.Module) -> nn.Module:
    clone = object.__new__(module.__class__)
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = module._parameters.copy()
    clone._buffers = module._buffers.copy()
    clone._modules = {
        name: _shallow_clone(child) for name, child in module._modules.items() if child
    }
    return clone


def _download(path: str) -> bytes:
    return Path(hf_hub_download(CACHE_REPO, path)).read_bytes()


def _synchronized_event_timing() -> None:
    """Make CUDA event timing wait for both events.

    Inductor times candidate kernels with CUDA events while compiling; inside a ZeroGPU call
    the events can still be pending when their elapsed time is read ("Both events must be
    completed before calculating elapsed time"), which aborts the compilation.
    """

    elapsed_time = torch.cuda.Event.elapsed_time

    def synchronized(self: torch.cuda.Event, end_event: torch.cuda.Event) -> float:
        self.synchronize()
        end_event.synchronize()
        return elapsed_time(self, end_event)

    torch.cuda.Event.elapsed_time = synchronized  # type: ignore[method-assign]
