"""The world model: loads every component from the weights repository and starts sessions.

`WorldModel.from_pretrained` is the only reader of the weights repository. It holds the causal
DiT, the Wan2.1 VAE and TAEHV (either encoder conditions every block, either decoder renders the
prediction), and the umT5 text encoder, and caches each prompt's cross-attention keys and values
so a prompt is encoded once however many sessions use it.

    world = WorldModel.from_pretrained(WorldConfig())
    session = world.start(scene, seed=0)
    while not session.finished:
        block = session.step(next_cameras(session.frames_needed))
"""

from __future__ import annotations

import logging

import torch

from .config import WorldConfig
from .depth import DepthEstimator
from .sampler import FlowSchedule
from .scene import Scene
from .session import FrameDecoder, LatentEncoder, Session
from .types import CrossAttentionCache
from .utils.hub import component_dir
from .wan import CausalWanDiT, PromptTokenizer, Taehv, TextEncoder, UMT5Encoder, WanVAE

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16
TEXT_LENGTH = 512


class WorldModel:
    """Weights and schedule shared by every session."""

    def __init__(
        self,
        dit: CausalWanDiT,
        vae: WanVAE,
        encoder: LatentEncoder,
        decoder: FrameDecoder,
        text_encoder: TextEncoder,
        config: WorldConfig,
    ) -> None:
        self.dit, self.vae, self.text_encoder = dit, vae, text_encoder
        self.encoder, self.decoder = encoder, decoder
        self.config = config
        self.device = torch.device(config.device)
        self.schedule = FlowSchedule.warped(config.denoising_steps, config.timestep_shift)
        self._prompts: dict[str, CrossAttentionCache] = {}

    @classmethod
    def from_pretrained(cls, config: WorldConfig) -> WorldModel:
        def folder(name: str) -> str:
            return str(component_dir(config.repo_id, config.revision, name))

        device = torch.device(config.device)
        dit = CausalWanDiT.from_pretrained(folder("dit")).to(device, DTYPE).eval()
        if config.dit_precision == "fp8":
            _quantize_fp8(dit)
        vae = WanVAE.from_pretrained(folder("vae")).to(device, DTYPE).eval()
        taehv = Taehv.from_pretrained(folder("taehv")).to(device, DTYPE).eval()
        encoder: LatentEncoder = taehv if config.encoder == "taehv" else vae
        decoder: FrameDecoder = taehv if config.decoder == "taehv" else vae
        text_model = UMT5Encoder.from_pretrained(folder("text_encoder"))
        text_model = text_model.to(torch.device(config.text_encoder_device), DTYPE).eval()
        tokenizer_file = (
            component_dir(config.repo_id, config.revision, "tokenizer") / "tokenizer.json"
        )
        tokenizer = PromptTokenizer(tokenizer_file, TEXT_LENGTH)
        for module in (dit, vae, taehv, text_model):
            module.requires_grad_(False)
        world = cls(dit, vae, encoder, decoder, TextEncoder(text_model, tokenizer), config)
        if config.compile:
            world.compile()
        return world

    def compile(self) -> None:
        """Regionally compile the repeated DiT block and the VAE encoder stages in place.

        Shapes are static per call site (prefill of 3 or 6 latent frames, denoise against a
        3- or 6-frame cache, 1- or 4-frame encoder chunks), so each region compiles a handful
        of specialized graphs. The VAE decoder gains nothing measurable and stays eager.
        """

        torch._dynamo.config.cache_size_limit = 32
        for block in self.dit.blocks:
            block.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        for stage in (*self.vae.encoder.down, *self.vae.encoder.middle):
            stage.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    def encode_prompt(self, prompt: str) -> CrossAttentionCache:
        """The DiT's per-block text keys and values for a prompt, computed once per prompt."""

        if prompt not in self._prompts:
            context = self.text_encoder([prompt]).to(self.device)
            with torch.inference_mode():
                self._prompts[prompt] = self.dit.encode_text(context)
            logger.info("encoded prompt %r", prompt[:60])
        return self._prompts[prompt]

    def start(self, scene: Scene, seed: int = 0, prompt: str | None = None) -> Session:
        """A new generation over `scene` (its own prompt unless `prompt` overrides it)."""

        text_kv = self.encode_prompt(scene.prompt if prompt is None else prompt)
        return Session(scene, self.dit, self.encoder, self.decoder, self.schedule, text_kv, seed)


def load_depth_estimator(config: WorldConfig) -> DepthEstimator:
    """The depth + camera estimator that turns pictures into a `Scene` (kept in its own dtypes)."""

    folder = component_dir(config.repo_id, config.revision, "depth")
    estimator = DepthEstimator.from_pretrained(str(folder)).to(torch.device(config.device))
    return estimator.eval().requires_grad_(False)


def _quantize_fp8(dit: CausalWanDiT) -> None:
    """Swap the blocks' linear layers for float8 ones (per-row dynamic scales, torchao)."""

    from torchao.quantization import (  # the optional `fp8` extra
        Float8DynamicActivationFloat8WeightConfig,
        PerRow,
        quantize_,
    )

    quantize_(dit.blocks, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
