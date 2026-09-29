"""Repackage every upstream checkpoint into one weights repository, one folder per component.

Temporary tool: run once to build `blanchon/inspatio-world-v1.5` from the upstream releases, then
it is no longer needed at inference. It reads

    inspatio/world-1.5              InSpatio-World-1.5-1.3B.safetensors (fp32 EMA)  -> dit/
    Wan-AI/Wan2.1-T2V-1.3B          Wan2.1_VAE.pth                                  -> vae/
                                    models_t5_umt5-xxl-enc-bf16.pth                  -> text_encoder/
                                    google/umt5-xxl/tokenizer.json                   -> tokenizer/
    github madebyollin/taehv        taew2_1.pth                                     -> taehv/
    florence-community/Florence-2-base  (copied as is, transformers format)          -> captioner/
    depth-anything/DA3NESTED-GIANT-LARGE                                             -> depth/

renames every state-dict key to this package's module names (fusing the DiT's q/k/v and the
cross-attention k/v projections), casts to bf16 (the inference dtype), loads each component with
`strict=True` to prove the mapping complete, and writes `config.json` + `model.safetensors` per
folder with `save_pretrained`. `--push` uploads the folder to the Hub.

    uv run scripts/convert_checkpoints.py --output /path/to/staging [--push]
"""

from __future__ import annotations

import re
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import load_file
from torch import Tensor, nn

from inspatio_world.utils.hub import no_init
from inspatio_world.wan import (
    CausalWanDiT,
    TaehvConfig,
    TaehvDecoder,
    UMT5Config,
    UMT5Encoder,
    WanDiTConfig,
    WanVAE,
    WanVAEConfig,
)

WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
INSPATIO_REPO = "inspatio/world-1.5"
TAEHV_URL = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
CAPTIONER_REPO = "florence-community/Florence-2-base"  # transformers-native Florence-2 weights
DTYPE = torch.bfloat16

type StateDict = dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class ConvertConfig:
    output: Path
    """Staging folder of the weights repository (one sub-folder per component)."""
    repo_id: str = "blanchon/inspatio-world-v1.5"
    """Hub repository to upload to with `--push`."""
    push: bool = False
    """Upload the staging folder to `repo_id` after converting."""
    skip_depth: bool = False
    """Leave `depth/` as it is (it is converted by `convert_depth.py`)."""


def _rename(state: StateDict, rules: list[tuple[str, str]]) -> StateDict:
    renamed = {}
    for key, value in state.items():
        for pattern, replacement in rules:
            key = re.sub(pattern, replacement, key)
        renamed[key] = value
    return renamed


def dit_state(path: Path) -> StateDict:
    """InSpatio DiT: strip `model.`, fuse q/k/v and cross k/v, rename to `CausalWanDiT`."""

    state = {key.removeprefix("model."): value for key, value in load_file(path).items()}
    for index in range(WanDiTConfig().num_layers):
        prefix = f"blocks.{index}"
        for kind in ("weight", "bias"):
            qkv = [state.pop(f"{prefix}.self_attn.{name}.{kind}") for name in ("q", "k", "v")]
            state[f"{prefix}.self_attention.qkv.{kind}"] = torch.cat(qkv)
            kv = [state.pop(f"{prefix}.cross_attn.{name}.{kind}") for name in ("k", "v")]
            state[f"{prefix}.cross_attention.kv.{kind}"] = torch.cat(kv)
    return _rename(
        state,
        [
            (r"\.self_attn\.o\.", ".self_attention.output."),
            (r"\.self_attn\.norm_q\.", ".self_attention.query_norm."),
            (r"\.self_attn\.norm_k\.", ".self_attention.key_norm."),
            (r"\.norm3\.", ".cross_attention_norm."),
            (r"\.cross_attn\.q\.", ".cross_attention.query."),
            (r"\.cross_attn\.o\.", ".cross_attention.output."),
            (r"\.cross_attn\.norm_q\.", ".cross_attention.query_norm."),
            (r"\.cross_attn\.norm_k\.", ".cross_attention.key_norm."),
            (r"\.ffn\.", ".feed_forward."),
            (r"^time_projection\.", "time_modulation."),
            (r"^head\.head\.", "head.linear."),
        ],
    )


def vae_state(path: Path) -> StateDict:
    """Wan2.1 VAE: name the residual/resample sub-layers after `WanVAE`'s modules."""

    state = torch.load(path, map_location="cpu", weights_only=True)
    return _rename(
        state,
        [
            (r"^conv1\.", "encoder_projection."),
            (r"^conv2\.", "decoder_projection."),
            (r"^(encoder|decoder)\.conv1\.", r"\1.conv_in."),
            (r"^(encoder|decoder)\.head\.0\.", r"\1.norm_out."),
            (r"^(encoder|decoder)\.head\.2\.", r"\1.conv_out."),
            (r"^encoder\.downsamples\.", "encoder.down."),
            (r"^decoder\.upsamples\.", "decoder.up."),
            (r"\.residual\.0\.", ".norm1."),
            (r"\.residual\.2\.", ".conv1."),
            (r"\.residual\.3\.", ".norm2."),
            (r"\.residual\.6\.", ".conv2."),
            (r"\.resample\.1\.", ".spatial."),
            (r"\.time_conv\.", ".temporal."),
            (r"\.to_qkv\.", ".qkv."),
        ],
    )


def text_encoder_state(path: Path) -> StateDict:
    """umT5-XXL encoder: rename attention / feed-forward / norm layers after `UMT5Encoder`."""

    state = torch.load(path, map_location="cpu", weights_only=True)
    return _rename(
        state,
        [
            (r"\.norm1\.", ".attention_norm."),
            (r"\.norm2\.", ".ffn_norm."),
            (r"\.attn\.q\.", ".query."),
            (r"\.attn\.k\.", ".key."),
            (r"\.attn\.v\.", ".value."),
            (r"\.attn\.o\.", ".output."),
            (r"\.ffn\.gate\.0\.", ".gate."),
            (r"\.ffn\.fc1\.", ".fc1."),
            (r"\.ffn\.fc2\.", ".fc2."),
            (r"\.pos_embedding\.", ".position_bias."),
        ],
    )


def taehv_state(path: Path) -> StateDict:
    """taew2_1: keep the decoder, shift indices past the parameter-free input clamp.

    The first temporal-grow layer is stride 1 at inference; its checkpoint weight holds two
    timesteps' channels and upstream keeps the last one.
    """

    state = torch.load(path, map_location="cpu", weights_only=True)
    decoder = {}
    for key, value in state.items():
        match = re.match(r"decoder\.(\d+)\.(.*)", key)
        if match:
            decoder[f"layers.{int(match[1]) - 1}.{match[2]}"] = value
    width = TaehvConfig().widths[0]
    grow = "layers.6.conv.weight"
    decoder[grow] = decoder[grow][-width:]
    return decoder


def _save[M: nn.Module](build: Callable[[], M], state: StateDict, folder: Path) -> None:
    with no_init():
        module = build()
    module.load_state_dict(state, strict=True)
    module.to(DTYPE).save_pretrained(folder)  # pyright: ignore[reportAttributeAccessIssue]
    # The mixin writes a placeholder card per folder; the repository has one card at its root
    (folder / "README.md").unlink(missing_ok=True)


def main(config: ConvertConfig) -> None:
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    wan = lambda name: Path(hf_hub_download(WAN_REPO, name))  # noqa: E731

    dit_path = Path(hf_hub_download(INSPATIO_REPO, "InSpatio-World-1.5-1.3B.safetensors"))
    _save(lambda: CausalWanDiT(WanDiTConfig()), dit_state(dit_path), output / "dit")
    _save(lambda: WanVAE(WanVAEConfig()), vae_state(wan("Wan2.1_VAE.pth")), output / "vae")
    text_state = text_encoder_state(wan("models_t5_umt5-xxl-enc-bf16.pth"))
    _save(lambda: UMT5Encoder(UMT5Config()), text_state, output / "text_encoder")
    del text_state
    (output / "tokenizer").mkdir(exist_ok=True)
    shutil.copy(wan("google/umt5-xxl/tokenizer.json"), output / "tokenizer" / "tokenizer.json")

    taehv_path = output / ".cache" / "taew2_1.pth"
    taehv_path.parent.mkdir(exist_ok=True)
    if not taehv_path.exists():
        urllib.request.urlretrieve(TAEHV_URL, taehv_path)
    _save(lambda: TaehvDecoder(TaehvConfig()), taehv_state(taehv_path), output / "taehv")
    shutil.rmtree(taehv_path.parent)

    # Florence-2-base captions video prompts (demo / example preparation); mirrored unchanged
    snapshot_download(
        CAPTIONER_REPO,
        local_dir=output / "captioner",
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )

    if not config.skip_depth:
        from convert_depth import convert_depth  # pyright: ignore[reportMissingImports]

        convert_depth(output)

    shutil.copy(Path(__file__).with_name("MODEL_CARD.md"), output / "README.md")
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o644)

    if config.push:
        api = HfApi()
        api.create_repo(config.repo_id, exist_ok=True)
        api.upload_folder(repo_id=config.repo_id, folder_path=output)


if __name__ == "__main__":
    main(tyro.cli(ConvertConfig))
