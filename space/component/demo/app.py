"""WorldViewer demo with a fake world model (scrolling gradient driven by the camera controls)."""

from __future__ import annotations

import os
import sys
import time
import uuid

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from gradio_worldviewer import Chunk, WorldViewer, WorldViewerData, encode_frames

W, H = 832, 480
FPS = 15.0
CHUNK = 12
N_CHUNKS = 30
LIMIT_S = 30.0

# Latest control payload per browser session (request.session_hash).
CONTROLS: dict[str, dict] = {}
LOG = os.environ.get("WORLDVIEWER_CONTROL_LOG")

_X, _Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def render(state: dict, action: dict) -> tuple[np.ndarray, np.ndarray]:
    ox, oy, zoom = state["ox"], state["oy"], state["zoom"]
    s = 1.0 / zoom
    r = 127 + 127 * np.sin((_X * s + ox) / 60.0)
    g = 127 + 127 * np.sin((_Y * s + oy) / 45.0 + (_X * s + ox) / 90.0)
    b = 127 + 127 * np.sin(((_X - W / 2) ** 2 + (_Y - H / 2) ** 2) ** 0.5 * s / 40.0 - state["t"])
    img = np.stack([r, g, b], -1).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    txt = "  ".join(f"{k}={action.get(k, 0):+.0f}" for k in ("forward", "right", "up", "yaw", "pitch"))
    d.rectangle([0, H - 28, W, H], fill=(0, 0, 0))
    d.text((10, H - 22), f"frame {state['frame']:4d}   {txt}", fill=(255, 255, 255))
    small = pil.resize((208, 120)).convert("L").convert("RGB")
    return np.asarray(pil), np.asarray(small)


def run(request: gr.Request):
    sh = request.session_hash
    sid = uuid.uuid4().hex[:8]
    yield WorldViewerData(status="loading", message="Warming up the fake model…")
    time.sleep(1.0)
    state = {"ox": 0.0, "oy": 0.0, "zoom": 1.0, "t": 0.0, "frame": 0}
    t0 = time.time()
    for i in range(N_CHUNKS):
        tb = time.time()
        act = dict(CONTROLS.get(sh, {}))
        frames, renders = [], []
        for _ in range(CHUNK):
            state["ox"] += 12 * float(act.get("right", 0)) + 20 * float(act.get("yaw", 0))
            state["oy"] -= 12 * float(act.get("up", 0)) + 16 * float(act.get("pitch", 0))
            state["zoom"] = float(np.clip(state["zoom"] * (1 + 0.02 * float(act.get("forward", 0))), 0.3, 4))
            state["t"] += 0.15
            state["frame"] += 1
            f, r = render(state, act)
            frames.append(f)
            renders.append(r)
        enc = encode_frames(frames, quality=80)
        enc_r = encode_frames(renders, quality=70)
        block_ms = (time.time() - tb) * 1000
        elapsed = time.time() - t0
        yield WorldViewerData(
            status="running",
            chunk=Chunk(id=i, session=sid, fps=FPS, frames=enc, renders=enc_r),
            stats={
                "block_ms": round(block_ms, 1),
                "gen_fps": round(CHUNK / max(1e-3, block_ms / 1000), 1),
                "frames": state["frame"],
                "elapsed_s": round(elapsed, 2),
                "limit_s": LIMIT_S,
            },
        )
        if elapsed > LIMIT_S:
            break
        # a fake model a bit faster than real time
        time.sleep(max(0.0, CHUNK / FPS * 0.9 - (time.time() - tb)))
    yield WorldViewerData(
        status="ended",
        message="Demo finished" if time.time() - t0 <= LIMIT_S else "Session ended: time limit",
        stats={"frames": state["frame"], "elapsed_s": round(time.time() - t0, 2), "limit_s": LIMIT_S},
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
    viewer = WorldViewer(label="World", show_render=True)
    start_evt = viewer.start(run, inputs=None, outputs=viewer, show_progress="hidden")
    viewer.stop(on_stop, inputs=None, outputs=viewer, cancels=[start_evt], queue=False, show_progress="hidden")
    viewer.control(on_control, inputs=None, outputs=None, queue=False, show_progress="hidden", trigger_mode="multiple")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    demo.launch(server_name="127.0.0.1", server_port=port)
