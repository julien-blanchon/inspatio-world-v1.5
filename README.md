# inspatio-world

A minimal, typed re-implementation of **[InSpatio-World 1.5](https://github.com/inspatio/inspatio-world-v1.5)**
([arXiv:2604.07209](https://arxiv.org/abs/2604.07209)) inference: a real-time, camera-controllable
4D world model. Give it a picture (or a few, or a video) and a camera path; it streams the scene
seen from that camera, 12 frames at a time, so the camera can be steered live.

[Weights](https://huggingface.co/blanchon/inspatio-world-v1.5) ·
[Demo (ZeroGPU)](https://huggingface.co/spaces/blanchon/inspatio-world-v1.5) ·
[Upstream](https://github.com/inspatio/inspatio-world-v1.5)

```bash
pip install "inspatio-world @ git+https://github.com/julien-blanchon/inspatio-world-v1.5"
# one image -> depth + cameras (Depth-Anything-3) -> a scene folder
inspatio-world prepare --source.paths photo.jpg --output scenes/photo
# fly through it with scripted moves (or --trajectory cameras.txt)
inspatio-world generate --source.paths scenes/photo \
    --moves forward:36 turn-right+forward:36 look-up:12 --output photo.mp4 \
    --world.decoder taehv --world.dit-precision fp8 --world.compile
```

```python
from inspatio_world import CameraAction, CameraRig, WorldConfig, WorldModel
from inspatio_world.data import load_scene

world = WorldModel.from_pretrained(WorldConfig(decoder="taehv", dit_precision="fp8"))
scene = load_scene(Path("scenes/photo"))
session, rig = world.start(scene, seed=0), CameraRig(scene)
while not session.finished:
    cameras = rig.advance(CameraAction(forward=1.0, yaw=0.2), session.frames_needed)
    block = session.step(cameras)  # block.frames: (12, 480, 832, 3) uint8, block.render: the condition
```

## How it works

Each block of 3 latents (12 frames at 832x480; the first block is 9 frames):

1. **Render** — the source views are lifted to 3D with their depth and forward-splatted into the
   target cameras (`render/splat.py`), giving a partial RGB render and a coverage mask.
2. **Encode** — the Wan2.1 VAE encodes the render and the source frames as causal streams.
3. **Denoise** — the causal Wan2.1-1.3B DiT prefills a KV cache from the clean context (source
   block + previous prediction) and denoises the block in 4 flow-matching steps, conditioned on
   the render latents and mask (`wan/dit.py`, `sampler.py`).
4. **Decode** — the Wan2.1 VAE decoder (or the ~20x cheaper TAEHV decoder) streams out frames.

All causal state lives in a `Session`, so a streamed video equals the offline one.

## Parity with upstream

| Component | vs upstream |
|---|---|
| umT5 tokenizer | token ids identical |
| DiT prefill / denoise | bf16 noise level: both are equally far from an fp32 run (flow mean abs 2.2e-2 vs 2.4e-2), 1e-2 apart |
| Wan2.1 VAE encode / decode stream | mean abs 2e-4 / 6e-4 (bf16) |
| TAEHV decoder | mean abs 2e-4 |
| Depth-Anything-3 nested (depth, K, poses) | bit-exact model outputs; post-processing within 5e-7 |
| Depth splatting render | identical coverage masks up to ≤ 7 borderline pixels of ~300k |

Deliberate deviations: the render is not round-tripped through a lossy H.264 file before encoding;
DA3 quantiles are computed exactly instead of on a random subsample; `ftfy` is dropped from prompt
cleaning.

## Speed (one GH200, per 12-frame block, steady state)

| Configuration | DiT (prefill + 4 steps) | 2x VAE encode | decode | total |
|---|---|---|---|---|
| eager, bf16, Wan decoder | 486 ms | 334 ms | 268 ms | ~1.1 s |
| compiled, bf16, TAEHV | 340 ms | 246 ms | 13 ms | ~0.62 s |
| compiled, fp8, TAEHV | 295 ms | 246 ms | 13 ms | ~0.57 s (21 fps) |

Scenes play at 15 fps (images) or their source rate (videos), so the compiled fp8 + TAEHV setup
generates faster than real time. Speed-ups: FP8 DiT linears (torchao), regional compilation
(`WorldConfig.compile`, or AoT on ZeroGPU: `space/aoti.py`), cached text keys/values, the TAEHV
decoder, and contention-free splatting.

## Layout

```
src/inspatio_world/   the package (see STYLE.md for conventions)
scripts/              one-off tools: convert_checkpoints.py (+ convert_depth.py) built the weights
                      repository, prepare_examples.py the example scenes
space/                the Hugging Face ZeroGPU Space: app.py, aoti.py, the WorldViewer component
```

## Licenses

Code: Apache-2.0 (upstream InSpatio-World, Wan2.1, DA3 code are Apache-2.0; TAEHV MIT). Weights
keep their licenses; note that the Depth-Anything-3 nested giant model is **CC BY-NC 4.0**.

## Citation

```bibtex
@misc{inspatio-world,
  title={INSPATIO-WORLD: A Real-Time 4D World Simulator via Spatiotemporal Autoregressive Modeling},
  author={InSpatio Team},
  journal={arXiv preprint arXiv:2604.07209},
  year={2026}
}
```
