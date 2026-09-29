"""WorldViewer demo with a fake world model.

A synthetic point-cloud "room" is splatted into a camera driven by the controls (the fake render
condition, with real holes); the fake "generated" frame is the upscaled splat with holes filled by a
scrolling gradient. Emits every optional field: renders, actions, control_seqs, poses and scene.
"""

from __future__ import annotations

import os
import sys
import time
import uuid

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from gradio_worldviewer import Chunk, SceneInfo, WorldViewer, WorldViewerData, encode_frames

W, H = 832, 480
RW, RH = 208, 120
K = [500.0, 500.0, 416.0, 240.0, float(W), float(H)]
FPS = 15.0
CHUNK = 12
N_CHUNKS = 40
LIMIT_S = 30.0
AUTOPILOT_CHUNKS = 3  # first blocks drift on autopilot (actions = None)

CONTROLS: dict[str, dict] = {}  # latest control payload per browser session (request.session_hash)
LOG = os.environ.get("WORLDVIEWER_CONTROL_LOG")


def make_room(n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Points on the floor / walls of a box in front of the origin camera (OpenCV: y down, z forward)."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1, z0, z1 = -3.0, 3.0, -2.0, 1.5, 1.0, 9.0
    parts = []
    m = n // 5
    parts.append(np.c_[rng.uniform(x0, x1, m), np.full(m, y1), rng.uniform(z0, z1, m)])  # floor
    parts.append(np.c_[rng.uniform(x0, x1, m), rng.uniform(y0, y1, m), np.full(m, z1)])  # back wall
    parts.append(np.c_[np.full(m, x0), rng.uniform(y0, y1, m), rng.uniform(z0, z1, m)])  # left wall
    parts.append(np.c_[np.full(m, x1), rng.uniform(y0, y1, m), rng.uniform(z0, z1, m)])  # right wall
    # a few boxes on the floor
    k = n - 4 * m
    c = rng.integers(0, 4, k)
    centers = np.array([[-1.5, 1.0, 4.0], [1.2, 0.9, 5.5], [0.0, 1.1, 7.0], [2.0, 1.0, 3.0]])
    sizes = np.array([0.5, 0.6, 0.4, 0.5])
    u = rng.uniform(-1, 1, (k, 3))
    face = rng.integers(0, 3, k)
    u[np.arange(k), face] = np.sign(u[np.arange(k), face])
    parts.append(centers[c] + u * sizes[c, None])
    pts = np.concatenate(parts).astype(np.float32)
    # colours: checker floor, gradient walls, saturated boxes
    col = np.zeros_like(pts)
    col[:, 0] = 0.5 + 0.5 * np.sin(pts[:, 0] * 1.3)
    col[:, 1] = 0.5 + 0.5 * np.sin(pts[:, 2] * 0.9 + 1.0)
    col[:, 2] = 0.5 + 0.5 * np.cos(pts[:, 1] * 1.7)
    floor = slice(0, m)
    chk = ((np.floor(pts[floor, 0]) + np.floor(pts[floor, 2])) % 2).astype(np.float32)
    col[floor] = np.c_[0.25 + 0.5 * chk, 0.25 + 0.5 * chk, 0.3 + 0.4 * chk]
    box_cols = np.array([[1, 0.3, 0.2], [0.2, 0.8, 1], [1, 0.85, 0.2], [0.5, 1, 0.4]])
    col[4 * m :] = box_cols[c]
    return pts, (col * 255).clip(0, 255).astype(np.uint8)


PTS, COLS = make_room()
SCENE = SceneInfo.from_arrays(PTS, COLS, source_poses=np.eye(4)[None], intrinsics=K, kind="image", title="Fake room")
_GX, _GY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def splat(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project the point cloud into the camera at RW x RH; returns (rgb, mask)."""
    R, t = pose[:3, :3], pose[:3, 3]
    pc = (PTS - t) @ R  # world -> camera (R^T (p - t))
    z = pc[:, 2]
    ok = z > 0.1
    s = RW / W
    u = (K[0] * s * pc[ok, 0] / z[ok] + K[2] * s).astype(np.int32)
    v = (K[1] * s * pc[ok, 1] / z[ok] + K[3] * s).astype(np.int32)
    col, zz = COLS[ok], z[ok]
    inb = (u >= 0) & (u < RW) & (v >= 0) & (v < RH)
    u, v, col, zz = u[inb], v[inb], col[inb], zz[inb]
    order = np.argsort(-zz)  # far first, near overwrites
    img = np.zeros((RH, RW, 3), np.uint8)
    mask = np.zeros((RH, RW), bool)
    img[v[order], u[order]] = col[order]
    mask[v[order], u[order]] = True
    return img, mask


def fake_frame(pose: np.ndarray, t: float, label: str) -> tuple[np.ndarray, np.ndarray]:
    rgb, mask = splat(pose)
    big = np.asarray(Image.fromarray(rgb).resize((W, H), Image.BILINEAR)).astype(np.float32)
    bm = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)).astype(np.float32)[..., None] / 255
    g = np.stack(
        [
            90 + 60 * np.sin(_GX / 70 + t),
            80 + 50 * np.sin(_GY / 50 + _GX / 110 - t * 0.7),
            110 + 60 * np.sin(np.hypot(_GX - W / 2, _GY - H / 2) / 45 - t),
        ],
        -1,
    )
    out = (big * np.clip(bm * 2.2, 0, 1) + g * (1 - np.clip(bm * 2.2, 0, 1))).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(out)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, H - 24, W, H], fill=(0, 0, 0))
    d.text((10, H - 18), label, fill=(255, 255, 255))
    return np.asarray(pil), rgb


def run(request: gr.Request):
    sh = request.session_hash
    sid = uuid.uuid4().hex[:8]
    yield WorldViewerData(status="loading", message="Warming up the fake model…", scene=SCENE)
    time.sleep(1.0)
    pose = np.eye(4)
    t_anim = 0.0
    n_frame = 0
    t0 = time.time()
    for i in range(N_CHUNKS):
        tb = time.time()
        ctrl = dict(CONTROLS.get(sh, {}))
        seq = ctrl.get("seq")
        autopilot = i < AUTOPILOT_CHUNKS
        act = None if autopilot else {k: float(ctrl.get(k, 0.0)) for k in ("forward", "right", "up", "yaw", "pitch")}
        frames, renders, poses = [], [], []
        for _ in range(CHUNK):
            a = act or {"forward": 0.4, "right": 0.0, "up": 0.0, "yaw": 0.15, "pitch": 0.0}
            R = pose[:3, :3] @ rot_y(0.035 * a["yaw"]) @ rot_x(0.025 * a["pitch"])
            pose[:3, :3] = R
            pose[:3, 3] += R @ (np.array([a["right"], -a["up"], a["forward"]]) * 0.06)
            t_anim += 0.12
            n_frame += 1
            label = f"frame {n_frame:4d}  " + ("AUTOPILOT" if act is None else "  ".join(f"{k}={v:+.0f}" for k, v in act.items()))
            f, r = fake_frame(pose, t_anim, label)
            frames.append(f)
            renders.append(r)
            poses.append(pose.copy())
        block_ms = (time.time() - tb) * 1000
        elapsed = time.time() - t0
        yield WorldViewerData(
            status="running",
            scene=SCENE,  # repeated on purpose: the diff makes repeats free, the viewer caches it
            chunk=Chunk(
                id=i,
                session=sid,
                fps=FPS,
                frames=encode_frames(frames, quality=80),
                renders=encode_frames(renders, quality=75),
                actions=[act] * CHUNK,
                control_seqs=[seq] * CHUNK,
                poses=np.stack(poses),
            ),
            stats={
                "block_ms": round(block_ms, 1),
                "gen_fps": round(CHUNK / max(1e-3, block_ms / 1000), 1),
                "frames": n_frame,
                "elapsed_s": round(elapsed, 2),
                "limit_s": LIMIT_S,
            },
        )
        if elapsed > LIMIT_S:
            break
        time.sleep(max(0.0, CHUNK / FPS * 0.95 - (time.time() - tb)))  # a bit faster than real time
    yield WorldViewerData(
        status="ended",
        message="Demo finished" if time.time() - t0 <= LIMIT_S else "Session ended: time limit",
        stats={"frames": n_frame, "elapsed_s": round(time.time() - t0, 2), "limit_s": LIMIT_S},
    )


def on_control(evt: gr.EventData, request: gr.Request):
    data = evt._data  # {"forward","right","up","yaw","pitch","seq","session"}
    CONTROLS[request.session_hash] = data
    line = f"[control] {time.time():.3f} {request.session_hash} {data}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")


def on_stop():
    return WorldViewerData(status="ended", message="Stopped by user")


with gr.Blocks(title="WorldViewer demo") as demo:
    gr.Markdown("## WorldViewer demo\nFake world model: press **Start**, click the viewer and drive with WASD / arrows.")
    viewer = WorldViewer(label="World", show_label=False)
    start_evt = viewer.start(run, inputs=None, outputs=viewer, show_progress="hidden")
    viewer.stop(on_stop, inputs=None, outputs=viewer, cancels=[start_evt], queue=False, show_progress="hidden")
    viewer.control(on_control, inputs=None, outputs=None, queue=False, show_progress="hidden", trigger_mode="multiple")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    demo.launch(server_name="127.0.0.1", server_port=port)
