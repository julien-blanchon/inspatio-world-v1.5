---
license: other
license_name: mixed
license_link: LICENSE
library_name: inspatio-world
pipeline_tag: image-to-video
tags:
  - world-model
  - novel-view-synthesis
  - video-generation
  - camera-control
  - wan2.1
base_model:
  - inspatio/world-1.5
  - Wan-AI/Wan2.1-T2V-1.3B
  - depth-anything/DA3NESTED-GIANT-LARGE
  - florence-community/Florence-2-base
---

# InSpatio-World 1.5 — all weights in one repository

Everything [InSpatio-World 1.5](https://github.com/inspatio/inspatio-world-v1.5)
([paper](https://arxiv.org/abs/2604.07209)) needs at inference, repackaged for the minimal
re-implementation [`inspatio-world`](https://github.com/julien-blanchon/inspatio-world-v1.5):
one folder per component, each a `config.json` + `model.safetensors` pair loaded with
`from_pretrained`. No other repository is downloaded.

| Folder | Component | Source | Params | Dtype |
|---|---|---|---|---|
| `dit/` | Causal Wan2.1-1.3B DiT (InSpatio-World 1.5, EMA) | [inspatio/world-1.5](https://huggingface.co/inspatio/world-1.5) | 1.42 B | bf16 |
| `vae/` | Wan2.1 VAE (causal video autoencoder) | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | 127 M | bf16 |
| `text_encoder/` | umT5-XXL encoder | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | 5.7 B | bf16 |
| `tokenizer/` | umT5 tokenizer (`tokenizer.json`) | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | — | — |
| `taehv/` | TAEHV `taew2_1` decoder (fast preview decoder) | [madebyollin/taehv](https://github.com/madebyollin/taehv) | 9.8 M | bf16 |
| `depth/` | Depth-Anything-3 nested giant-large (depth + cameras) | [depth-anything/DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) | 1.7 B | bf16 / fp32 |
| `captioner/` | Florence-2-base (video prompts for uploads / examples), transformers format | [florence-community/Florence-2-base](https://huggingface.co/florence-community/Florence-2-base) | 0.23 B | fp32 |
| `examples/` | Scene folders: the six upstream examples and nine scenes from the [project page](https://inspatio.github.io/inspatio-world-1.5/); `trajectories/`: camera-path presets with previews | [upstream examples](https://github.com/inspatio/inspatio-world-v1.5/tree/main/examples) | — | — |

The conversion only renames keys (fusing the DiT's q/k/v and cross-attention k/v projections),
drops unused branches (DA3's ray / Gaussian-splatting heads) and casts to the dtype each module
runs in; upstream runs the DiT, VAE and text encoder in bf16, so outputs are unchanged. It was
produced by [`scripts/convert_checkpoints.py`](https://github.com/julien-blanchon/inspatio-world-v1.5/blob/main/scripts/convert_checkpoints.py).

## Usage

```bash
pip install "inspatio-world @ git+https://github.com/julien-blanchon/inspatio-world-v1.5"
inspatio-world generate --source.paths photo.jpg --moves forward:36 turn-right+forward:36 --output out.mp4
```

```python
from inspatio_world import CameraAction, CameraRig, WorldConfig, WorldModel
from inspatio_world.data import load_scene

world = WorldModel.from_pretrained(WorldConfig(dit_precision="fp8", compile=True))
scene = load_scene(examples_dir / "image_example_00")
session, rig = world.start(scene), CameraRig(scene)
while True:
    block = session.step(rig.advance(CameraAction(forward=1.0, yaw=0.3), session.frames_needed))
    ...  # block.frames: (12, 480, 832, 3) uint8
```

Interactive demo: [spaces/blanchon/inspatio-world-v1.5](https://huggingface.co/spaces/blanchon/inspatio-world-v1.5).

## Licenses

Each component keeps its original license:

- `dit/`: InSpatio-World 1.5 weights, released by the InSpatio team with the Apache-2.0 code
  of [inspatio/inspatio-world-v1.5](https://github.com/inspatio/inspatio-world-v1.5).
- `vae/`, `text_encoder/`, `tokenizer/`: Wan2.1, Apache-2.0.
- `taehv/`: TAEHV, MIT.
- `captioner/`: Florence-2, MIT.
- `depth/`: Depth-Anything-3 DA3NESTED-GIANT-LARGE, **CC BY-NC 4.0 (non-commercial)**.
- `examples/`: from the upstream repository.

## Citation

```bibtex
@misc{inspatio-world,
  title={INSPATIO-WORLD: A Real-Time 4D World Simulator via Spatiotemporal Autoregressive Modeling},
  author={InSpatio Team},
  journal={arXiv preprint arXiv:2604.07209},
  year={2026}
}
```
