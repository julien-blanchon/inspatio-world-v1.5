"""InSpatio-World 1.5 on ZeroGPU: drive a camera through a scene in real time.

Pick an example or upload a picture / video (depth and cameras are estimated with
Depth-Anything-3), press Start, click the viewer and fly with WASD / arrows. Each block of 12
frames is generated from the camera path the keys produced while the previous block was playing.

Process layout on ZeroGPU: the Gradio server (main process) receives the viewer's control events
and writes the latest camera action to a small file per browser session; the `@spaces.GPU`
generator runs in a GPU worker process, reads that file before every block, and yields frames
back. A stop flag file ends the loop early; the session also ends at its time limit.
"""

from __future__ import annotations

import spaces  # must be imported before torch on ZeroGPU

# isort: split

import itertools
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aoti  # pyright: ignore[reportImplicitRelativeImport]
import gradio as gr
import numpy as np
import simplejpeg
import torch
from gradio.themes import Soft
from gradio_worldviewer import Chunk, WorldViewer, WorldViewerData, encode_frame
from huggingface_hub import snapshot_download

from inspatio_world import (
    CameraAction,
    CameraRig,
    RigConfig,
    Scene,
    Session,
    WorldConfig,
    WorldModel,
    load_depth_estimator,
)
from inspatio_world.data import load_scene, read_image, read_trajectory, read_video, save_scene

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inspatio-space")

REPO_ID = os.environ.get("INSPATIO_REPO", "blanchon/inspatio-world-v1.5")
GPU_SIZE = os.environ.get("INSPATIO_GPU_SIZE", "xlarge")
SESSION_SECONDS = 90  # upper bound of a session's GPU time
SECONDS_ARG = 6  # position of `seconds` in run_session's arguments
MAX_LEAD_SECONDS = 1.0  # generation may run this far ahead of playback before it waits
MAX_UPLOAD_FRAMES = 180
RENDER_PREVIEW_STRIDE = 4  # picture-in-picture render at 1/4 resolution
STATE_DIR = Path(tempfile.gettempdir()) / "inspatio-sessions"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# --- models: placed on cuda at import, as ZeroGPU expects ---
CONFIG = WorldConfig(repo_id=REPO_ID, decoder="taehv", dit_precision="fp8")
world = WorldModel.from_pretrained(CONFIG)
depth_estimator = load_depth_estimator(CONFIG)
EXAMPLES = (
    Path(REPO_ID) / "examples"
    if Path(REPO_ID).is_dir()
    else Path(snapshot_download(REPO_ID, allow_patterns=["examples/*"])) / "examples"
)


@spaces.GPU(duration=1500)
def compile_world() -> dict[str, Any]:
    return aoti.compile_graphs(world)


if os.environ.get("INSPATIO_AOTI", "1") == "1":
    try:
        aoti.install(world, compile_world())
    except Exception:  # the demo stays usable eagerly if compilation fails
        logger.exception("ahead-of-time compilation failed, running eagerly")


# --- control / stop files shared between the server and the GPU worker ---
def _state_file(browser: str, name: str) -> Path:
    return STATE_DIR / f"{browser}.{name}.json"


def write_state(browser: str, name: str, value: dict[str, Any]) -> None:
    path = _state_file(browser, name)
    # Unique per writer: control events arrive concurrently
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def read_action(browser: str) -> CameraAction:
    try:
        data = json.loads(_state_file(browser, "control").read_text())
    except (OSError, ValueError):
        return CameraAction()
    return CameraAction(**{axis: float(data.get(axis, 0.0)) for axis in asdict(CameraAction())})


def stop_requested(browser: str, since: float) -> bool:
    """Whether Stop was pressed in this browser after `since` (wall-clock seconds)."""

    try:
        return json.loads(_state_file(browser, "stop").read_text())["time"] > since
    except (OSError, ValueError, KeyError):
        return False


# --- scenes ---
def example_choices() -> list[tuple[str, str]]:
    return [
        (str(folder / "poster.jpg"), folder.name)
        for folder in sorted(EXAMPLES.iterdir())
        if folder.is_dir()
    ]


@spaces.GPU(duration=60)
def estimate_scene(kind: str, pictures: np.ndarray, prompt: str, fps: float) -> str:
    scene = Scene.estimate(kind, pictures, depth_estimator, prompt=prompt, fps=fps)  # pyright: ignore[reportArgumentType]
    folder = STATE_DIR / f"scene-{uuid.uuid4().hex[:12]}"
    save_scene(scene, folder)
    return str(folder)


def build_upload(images: list[str] | None, video: str | None, prompt: str) -> tuple[str, str, str]:
    if video:
        frames, fps = read_video(Path(video), max_frames=MAX_UPLOAD_FRAMES)
        folder = estimate_scene("video", frames, prompt, fps)
    elif images:
        pictures = np.stack([read_image(Path(path)) for path in images[:4]])
        folder = estimate_scene("image", pictures, prompt, 15.0)
    else:
        raise gr.Error("Upload one to four images of a place, or a video.")
    return folder, str(Path(folder) / "view_00.png") if not video else "", prompt


def select_example(event: gr.SelectData) -> tuple[str, str, str]:
    folder = EXAMPLES / example_choices()[event.index][1]
    meta = json.loads((folder / "scene.json").read_text())
    return str(folder), str(folder / "poster.jpg"), meta["prompt"]


# --- the session ---
def session_duration(*args: object) -> int:
    """GPU time to reserve: the requested session length plus loading (same args as run_session)."""

    return int(float(args[SECONDS_ARG])) + 30  # pyright: ignore[reportArgumentType]


@spaces.GPU(duration=session_duration, size=GPU_SIZE)  # pyright: ignore[reportArgumentType]
def run_session(
    scene_path: str,
    prompt: str,
    quality: str,
    camera: str,
    speed: float,
    seed: int,
    seconds: float,
    browser: str,
) -> Iterator[dict[str, Any]]:
    """Yield {"frames", "renders", "stats"} per block, then {"ended": reason}."""

    folder = Path(scene_path)
    scene = load_scene(folder)
    text_kv = world.encode_prompt(prompt)
    decoder = world.vae if quality == "Quality (Wan VAE decoder)" else world.decoder
    session = Session(scene, world.dit, world.vae, decoder, world.schedule, text_kv, seed)
    rig = CameraRig(scene, RigConfig(move_speed=RigConfig().move_speed * speed))
    trajectory = None
    if camera == "Example trajectory" and (folder / "trajectory.txt").exists():
        trajectory = torch.from_numpy(read_trajectory(folder / "trajectory.txt"))

    started, started_wall = time.perf_counter(), time.time()
    while (
        reason := _end_reason(session, trajectory, time.perf_counter() - started, seconds)
    ) is None:
        if stop_requested(browser, started_wall):
            reason = "Stopped"
            break

        count = session.frames_needed
        if trajectory is not None:
            index = torch.arange(session.frame_index, session.frame_index + count).clamp(
                max=len(trajectory) - 1
            )
            poses = trajectory[index]
        else:
            poses = rig.advance(read_action(browser), count)
        block_start = time.perf_counter()
        block = session.step(poses)
        frames = block.frames.cpu().numpy()
        renders = block.render[:, ::RENDER_PREVIEW_STRIDE, ::RENDER_PREVIEW_STRIDE].cpu().numpy()
        block_seconds = time.perf_counter() - block_start
        yield {
            "frames": frames,
            "renders": renders,
            "stats": {
                "block_ms": round(block_seconds * 1000, 1),
                "gen_fps": round(len(frames) / block_seconds, 1),
                "frames": session.frame_index,
                "elapsed_s": round(time.perf_counter() - started, 1),
                "limit_s": seconds,
            },
        }

        # Stay at most MAX_LEAD_SECONDS ahead of playback so key presses show up quickly
        lead = session.frame_index / scene.fps - (time.perf_counter() - started)
        if lead > MAX_LEAD_SECONDS:
            time.sleep(lead - MAX_LEAD_SECONDS)
    yield {"ended": reason}


def _end_reason(
    session: Session, trajectory: torch.Tensor | None, elapsed: float, seconds: float
) -> str | None:
    if trajectory is not None and session.frame_index >= len(trajectory):
        return "End of the example trajectory"
    if session.finished:
        return "End of the source video"
    if elapsed > seconds:
        return "Time limit reached"
    return None


def start(
    scene_path: str | None,
    prompt: str,
    quality: str,
    camera: str,
    speed: float,
    seed: int,
    seconds: float,
    request: gr.Request,
) -> Iterator[WorldViewerData]:
    if not scene_path:
        yield WorldViewerData(
            status="error", message="Pick an example or build a scene from your upload first."
        )
        return
    browser, session_id = request.session_hash or "anonymous", uuid.uuid4().hex[:8]
    write_state(browser, "control", asdict(CameraAction()))
    fps = load_scene_fps(scene_path)
    yield WorldViewerData(status="loading", message="Waiting for a GPU…")

    chunk_ids = itertools.count()
    for item in run_session(
        scene_path, prompt, quality, camera, speed, int(seed), seconds, browser
    ):
        if "ended" in item:
            yield WorldViewerData(status="ended", message=item["ended"])
            return
        frames = _jpeg_uris(item["frames"], quality=85)
        renders = _jpeg_uris(item["renders"], quality=70)
        chunk = Chunk(
            id=next(chunk_ids), session=session_id, fps=fps, frames=frames, renders=renders
        )
        yield WorldViewerData(status="running", chunk=chunk, stats=item["stats"])


def _jpeg_uris(frames: np.ndarray, quality: int) -> list[str]:
    return [
        encode_frame(
            simplejpeg.encode_jpeg(np.ascontiguousarray(frame), quality=quality, colorspace="RGB")
        )
        for frame in frames
    ]


def load_scene_fps(scene_path: str) -> float:
    return float(json.loads((Path(scene_path) / "scene.json").read_text())["fps"])


def on_control(event: gr.EventData, request: gr.Request) -> None:
    write_state(request.session_hash or "anonymous", "control", event._data)


def on_stop(request: gr.Request) -> WorldViewerData:
    write_state(request.session_hash or "anonymous", "stop", {"time": time.time()})
    return WorldViewerData(status="ended", message="Stopped")


# --- UI ---
DESCRIPTION = """
# InSpatio-World 1.5 — a real-time 4D world model
Pick a scene, press **Start**, then click the viewer and fly with **W A S D** (move),
**← →** or **Q E** (turn), **R F** (look up / down), **Space / Shift** (up / down).
The model (Wan2.1-1.3B, causal, 4 steps per block) re-renders the scene from the camera you steer,
12 frames at a time. [Code](https://github.com/julien-blanchon/inspatio-world-v1.5) ·
[Weights](https://huggingface.co/blanchon/inspatio-world-v1.5) ·
[Paper](https://arxiv.org/abs/2604.07209) · [Upstream](https://github.com/inspatio/inspatio-world-v1.5)
"""

with gr.Blocks(title="InSpatio-World 1.5") as demo:
    gr.Markdown(DESCRIPTION)
    scene_path = gr.State(None)
    with gr.Row():
        with gr.Column(scale=3):
            viewer = WorldViewer(label="World", show_label=False, show_render=False)
        with gr.Column(scale=1, min_width=300):
            poster = gr.Image(label="Scene", interactive=False, height=180)
            prompt = gr.Textbox(
                label="Prompt", lines=3, placeholder="Optional description of the scene"
            )
            with gr.Accordion("Session", open=True):
                camera = gr.Radio(
                    ["Keyboard", "Example trajectory"], value="Keyboard", label="Camera"
                )
                quality = gr.Radio(
                    ["Fast (TAEHV decoder)", "Quality (Wan VAE decoder)"],
                    value="Fast (TAEHV decoder)",
                    label="Decoder",
                )
                speed = gr.Slider(0.25, 3.0, value=1.0, step=0.25, label="Movement speed")
                seconds = gr.Slider(
                    15, SESSION_SECONDS, value=60, step=5, label="Session length (s)"
                )
                seed = gr.Number(0, label="Seed", precision=0)

    gr.Markdown("### Examples — click one to load it")
    gallery = gr.Gallery(
        value=example_choices(), columns=6, height=160, allow_preview=False, show_label=False
    )
    with gr.Accordion("Your own scene", open=False):
        gr.Markdown(
            "One image (or up to 4 of the same place), or a short video. Depth and cameras are estimated with Depth-Anything-3."
        )
        with gr.Row():
            images = gr.File(file_count="multiple", file_types=["image"], label="Images")
            video = gr.Video(label="Video", sources=["upload"])
        build = gr.Button("Build scene")

    gallery.select(select_example, None, [scene_path, poster, prompt])
    build.click(build_upload, [images, video, prompt], [scene_path, poster, prompt])
    start_event = viewer.start(
        start,
        [scene_path, prompt, quality, camera, speed, seed, seconds],
        viewer,
        show_progress="hidden",
    )
    viewer.stop(on_stop, None, viewer, cancels=[start_event], queue=False, show_progress="hidden")
    viewer.control(
        on_control, None, None, queue=False, show_progress="hidden", trigger_mode="multiple"
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        theme=Soft(primary_hue="orange"),
        allowed_paths=[str(EXAMPLES), str(STATE_DIR)],
        ssr_mode=False,  # the SSR proxy drops the custom component's requests on Spaces
    )
