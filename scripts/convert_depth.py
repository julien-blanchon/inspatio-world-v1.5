"""Repackage Depth-Anything-3 DA3NESTED-GIANT-LARGE as the `depth/` component of the weights repo.

Temporary tool, folded into `convert_checkpoints.py`: it reads the upstream snapshot

    depth-anything/DA3NESTED-GIANT-LARGE    model.safetensors (fp32)    -> depth/

keeps only what `DepthEstimator` runs (the any-view ViT-g with its DualDPT depth branch and camera
decoder, and the metric ViT-L with its DPT head), dropping the ray branch, the camera encoder and
the Gaussian-splatting head; renames keys to this package's modules; casts every tensor to the
dtype its module computes in (bf16 for the ViT linear layers and patch embeddings, float32 for
everything else); loads with `strict=True` to prove the mapping complete; and writes
`config.json` + `model.safetensors` with `save_pretrained`.

    uv run scripts/convert_depth.py --output /path/to/staging
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tyro
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor

from inspatio_world.depth import DepthEstimator, DepthEstimatorConfig
from inspatio_world.utils.hub import no_init

DA3_REPO = "depth-anything/DA3NESTED-GIANT-LARGE"
DA3_REVISION = "8615eefb62f2db4f8d6ebaa59160086981672829"

type StateDict = dict[str, Tensor]

DROPPED = re.compile(r"^model\.da3\.(cam_enc|gs_head|gs_adapter)\.|_aux\.")
RULES = [
    # Backbones: drop the DinoV2 wrapper levels, flatten LayerScale and the learned tokens
    (r"^model\.da3\.backbone\.pretrained\.", "backbone."),
    (r"^model\.da3_metric\.backbone\.pretrained\.", "metric_backbone."),
    (r"\.patch_embed\.proj\.", ".patch_embed."),
    (r"\.(ls[12])\.gamma$", r".\1"),
    # Heads: the output convolutions belong to the head, the rest to its feature pyramid
    (r"^model\.da3\.head\.", "head."),
    (r"^model\.da3_metric\.head\.", "metric_head."),
    (r"^(head|metric_head)\.scratch\.((sky_)?output_conv2)\.", r"\1.\2."),
    (r"^(head|metric_head)\.(scratch\.|(?=projects|resize_layers))", r"\1.pyramid."),
    # Camera decoder: the FoV layer loses its ReLU wrapper
    (r"^model\.da3\.cam_dec\.", "camera_decoder."),
    (r"\.fc_fov\.0\.", ".fc_fov."),
]
SQUEEZED = ("cls_token", "camera_token", "pos_embed")  # stored without upstream's leading 1


def depth_state(path: Path) -> StateDict:
    """Upstream nested DA3 state dict -> `DepthEstimator` keys (dtypes unchanged)."""

    state = {}
    for key, value in load_file(path).items():
        if DROPPED.search(key):
            continue
        for pattern, replacement in RULES:
            key = re.sub(pattern, replacement, key)
        state[key] = value[0] if key.rsplit(".", 1)[-1] in SQUEEZED else value
    return state


def convert_depth(output_dir: Path) -> None:
    """Write `output_dir/depth/{config.json, model.safetensors}` from the upstream snapshot."""

    path = Path(hf_hub_download(DA3_REPO, "model.safetensors", revision=DA3_REVISION))
    with no_init():
        model = DepthEstimator(DepthEstimatorConfig())
    targets = model.state_dict()
    state = {
        key: value.to(targets[key].dtype) if key in targets else value
        for key, value in depth_state(path).items()
    }
    model.load_state_dict(state, strict=True)
    model.save_pretrained(output_dir / "depth")


@dataclass(frozen=True, slots=True)
class ConvertDepthConfig:
    output: Path
    """Staging folder of the weights repository; `depth/` is written inside it."""


def main() -> None:
    convert_depth(tyro.cli(ConvertDepthConfig).output)


if __name__ == "__main__":
    main()
