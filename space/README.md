---
title: InSpatio-World 1.5
emoji: 🌍
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Fly through a scene in real time with a 4D world model
models:
  - blanchon/inspatio-world-v1.5
---

# InSpatio-World 1.5 — real-time camera control

Interactive demo of [InSpatio-World 1.5](https://github.com/inspatio/inspatio-world-v1.5)
(arXiv:2604.07209) running on ZeroGPU with the minimal re-implementation
[`inspatio-world`](https://github.com/julien-blanchon/inspatio-world-v1.5).

- Pick an example (or upload an image / a video: depth and cameras come from Depth-Anything-3).
- Press **Start**, click the viewer, and steer with **WASD**, **arrows / Q E**, **R F**, **Space / Shift**.
- Every 12 frames are generated from the camera path of your keys, a block (≈0.8 s) behind.

Speed-ups used here: FP8 (torchao) DiT linear layers, ahead-of-time Inductor compilation of one DiT
block shared by all 30 blocks and cached on the Hub (`aoti.py`), the TAEHV decoder, cached
text keys/values, and depth-splat rendering without border contention. The viewer is a custom
Svelte component (`component/`, built wheel in `wheels/`).
