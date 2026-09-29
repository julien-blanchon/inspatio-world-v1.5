"""InSpatio-World 1.5 on ZeroGPU: drive a camera through a scene in real time.

Pick a scene (or upload a picture / video: depth and cameras come from Depth-Anything-3, video
prompts from Florence-2), pick a camera path (your keyboard, or a preset), press Start. The world
runs a fixed LATENCY_SECONDS behind you: each frame plays the keys you held that long before it
appears, and the viewer shows what the model conditions on (the point cloud splatted into your
camera), the camera path, and a timeline of your key presses against the frames that play them.

Process layout on ZeroGPU: the Gradio server (main process) receives the viewer's control events
and appends them, timestamped, to a small log file per browser session; the `@spaces.GPU`
generator runs in a GPU worker process, reads the keys held at each frame's time from that log
before every block, and yields frames back. A stop flag file ends the loop early; the session also
ends at its time limit.
"""

from __future__ import annotations

import spaces  # must be imported before torch on ZeroGPU

# isort: split

import base64
import bisect
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
from gradio_worldviewer import WorldViewer, encode_frame
from huggingface_hub import snapshot_download
from PIL import Image

from inspatio_world import (
    CameraAction,
    CameraRig,
    RigConfig,
    Scene,
    Session,
    WorldConfig,
    WorldModel,
    load_depth_estimator,
    parse_moves,
)
from inspatio_world.data import load_scene, read_image, read_trajectory, read_video, save_scene
from inspatio_world.utils import component_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inspatio-space")

REPO_ID = os.environ.get("INSPATIO_REPO", "blanchon/inspatio-world-v1.5")
GPU_SIZE = os.environ.get("INSPATIO_GPU_SIZE", "xlarge")
SESSION_SECONDS = 90  # upper bound of a session's GPU time
DEFAULT_SESSION_SECONDS = 45  # (45 + margin) x 2 (xlarge) fits the 120 s anonymous daily quota
SESSION_MARGIN_SECONDS = 10  # scene loading, prompt encoding and the eager first block
SECONDS_ARG = 6  # position of `seconds` in run_session's arguments
# Playback runs below the ~14 fps the GPU generates, so the viewer's buffer never drains
PLAYBACK_FPS = 12.0
# Fixed key-to-screen delay: frame f plays the keys held at t0 + f/fps and is shown at
# t0 + LATENCY_SECONDS + f/fps. A block (<= 1 s of frames) starts once the keys of its last frame
# are known and takes ~0.85 s to generate, which leaves ~1.7 s of slack before it is due.
LATENCY_SECONDS = 3.5
MAX_UPLOAD_FRAMES = 180
RENDER_PREVIEW_STRIDE = 2  # the condition panel shows the render at half resolution
SCENE_POINTS = 30_000  # sparse point cloud sent to the 3D camera view
POINT_VIEWS = 4  # views of a video sampled for that point cloud
KEYBOARD = "Keyboard"
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
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
PRESETS_FILE = EXAMPLES / "trajectories" / "presets.json"
# Camera paths: {"name", "kind": "keyboard" | "trajectory" (the scene's own) | "moves", ...}
PRESETS: list[dict[str, Any]] = (
    json.loads(PRESETS_FILE.read_text())
    if PRESETS_FILE.exists()
    else [{"name": KEYBOARD, "kind": "keyboard"}]
)


def _load_captioner() -> tuple[Any, Any]:
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    folder = component_dir(REPO_ID, None, "captioner")
    model = Florence2ForConditionalGeneration.from_pretrained(folder, dtype=torch.bfloat16)
    return AutoProcessor.from_pretrained(folder), model.to("cuda").eval()  # pyright: ignore[reportArgumentType]


caption_processor, caption_model = _load_captioner()


@spaces.GPU(duration=900)
def compile_world() -> dict[str, bytes]:
    return aoti.load_or_compile(world)


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


class ControlLog:
    """The browser's control events with their arrival times, read incrementally by the worker."""

    def __init__(self, browser: str) -> None:
        self.path = _state_file(browser, "controls")
        self.offset = 0
        self.times: list[float] = [float("-inf")]
        self.events: list[tuple[CameraAction, int | None]] = [(CameraAction(), None)]

    @staticmethod
    def reset(browser: str) -> None:
        _state_file(browser, "controls").write_text("")

    @staticmethod
    def append(browser: str, event: dict[str, Any]) -> None:
        line = json.dumps({**event, "time": time.time()}) + "\n"
        # One O_APPEND write per event: concurrent control events never interleave
        descriptor = os.open(
            _state_file(browser, "controls"), os.O_WRONLY | os.O_APPEND | os.O_CREAT
        )
        try:
            os.write(descriptor, line.encode())
        finally:
            os.close(descriptor)

    def at(self, moment: float) -> tuple[CameraAction, int | None]:
        """The action held at wall-clock `moment` and the sequence number of its event."""

        self._refresh()
        return self.events[bisect.bisect_right(self.times, moment) - 1]

    def _refresh(self) -> None:
        try:
            with self.path.open("rb") as file:
                file.seek(self.offset)
                data = file.read()
        except OSError:
            return
        complete = data[: data.rfind(b"\n") + 1]
        self.offset += len(complete)
        for line in complete.decode().splitlines():
            event = json.loads(line)
            fields = {axis: float(event.get(axis, 0.0)) for axis in asdict(CameraAction())}
            self.times.append(float(event["time"]))
            self.events.append((CameraAction(**fields), event.get("seq")))


def stop_requested(browser: str, since: float) -> bool:
    """Whether Stop was pressed in this browser after `since` (wall-clock seconds)."""

    try:
        return json.loads(_state_file(browser, "stop").read_text())["time"] > since
    except (OSError, ValueError, KeyError):
        return False


# --- scenes ---
def example_folders() -> list[Path]:
    """Scene folders, image scenes first (they run as long as you steer), then videos."""

    folders = [f for f in EXAMPLES.iterdir() if (f / "scene.json").exists()]
    kind = {f: json.loads((f / "scene.json").read_text())["kind"] for f in folders}
    return sorted(folders, key=lambda f: (kind[f] == "video", f.name))


def example_choices() -> list[tuple[str, str]]:
    return [(str(f / "poster.jpg"), f.name.replace("_", " ")) for f in example_folders()]


def preset_choices() -> list[tuple[str, str]]:
    return [
        (str(EXAMPLES / "trajectories" / p["preview"]), p["name"])
        for p in PRESETS
        if "preview" in p
    ]


def caption(image: np.ndarray) -> str:
    """Florence-2 detailed caption of a frame, as upstream v1 prompts its videos."""

    picture = Image.fromarray(image)
    inputs = caption_processor(text=CAPTION_TASK, images=picture, return_tensors="pt")
    inputs = inputs.to("cuda", torch.bfloat16)
    ids = caption_model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=3)
    text = caption_processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = caption_processor.post_process_generation(
        text, task=CAPTION_TASK, image_size=picture.size
    )
    return str(parsed[CAPTION_TASK]).strip()


@spaces.GPU(duration=90)
def estimate_scene(kind: str, pictures: np.ndarray, prompt: str, fps: float) -> tuple[str, str]:
    with torch.inference_mode():
        if kind == "video" and not prompt.strip():
            prompt = caption(pictures[0])
        scene = Scene.estimate(kind, pictures, depth_estimator, prompt=prompt, fps=fps)  # pyright: ignore[reportArgumentType]
    folder = STATE_DIR / f"scene-{uuid.uuid4().hex[:12]}"
    save_scene(scene, folder)
    return str(folder), prompt


def build_upload(
    images: list[str] | None, video: str | None, prompt: str
) -> tuple[str, np.ndarray, str]:
    if video:
        pictures, fps = read_video(Path(video), max_frames=MAX_UPLOAD_FRAMES)
        folder, prompt = estimate_scene("video", pictures, prompt, fps)
    elif images:
        pictures = np.stack([read_image(Path(path)) for path in images[:4]])
        folder, prompt = estimate_scene("image", pictures, prompt, 15.0)
    else:
        raise gr.Error("Upload one to four images of a place, or a video.")
    return folder, pictures[0], prompt


def select_example(event: gr.SelectData) -> tuple[str, np.ndarray, str]:
    # The poster goes out as pixels: Hub-cache paths are symlinks Gradio refuses to serve
    folder = example_folders()[event.index]
    meta = json.loads((folder / "scene.json").read_text())
    return str(folder), np.asarray(Image.open(folder / "poster.jpg")), meta["prompt"]


def select_preset(event: gr.SelectData) -> str:
    return PRESETS[event.index]["name"]


# --- what the viewer shows next to the video ---
def scene_info(scene: Scene, title: str) -> dict[str, Any]:
    """A sparse colored point cloud of the source views and their cameras, for the 3D view."""

    views = range(scene.num_views)
    if scene.kind == "video":
        views = np.linspace(0, scene.num_views - 1, min(POINT_VIEWS, scene.num_views)).astype(int)
    generator = np.random.default_rng(0)
    points, colors = [], []
    for view in views:
        depth = scene.depth[view].numpy()
        rows, cols = np.nonzero(depth > 0)
        pick = generator.choice(
            len(rows), min(len(rows), SCENE_POINTS // len(views)), replace=False
        )
        rows, cols = rows[pick], cols[pick]
        z = depth[rows, cols]
        k = scene.intrinsics[view].numpy()
        local = np.stack(
            [(cols - k[0, 2]) * z / k[0, 0], (rows - k[1, 2]) * z / k[1, 1], z, np.ones_like(z)]
        )
        camera_to_world = np.linalg.inv(scene.world_to_camera[view].numpy())
        points.append((camera_to_world @ local)[:3].T)
        colors.append(scene.images[view].numpy()[rows, cols])
    k = scene.intrinsics[0].numpy()
    return {
        "points": base64.b64encode(np.concatenate(points).astype("<f4").tobytes()).decode(),
        "colors": base64.b64encode(np.concatenate(colors).astype(np.uint8).tobytes()).decode(),
        "source_poses": [
            np.linalg.inv(p).reshape(-1).tolist() for p in scene.world_to_camera.numpy()
        ],
        "intrinsics": [
            float(k[0, 0]),
            float(k[1, 1]),
            float(k[0, 2]),
            float(k[1, 2]),
            832.0,
            480.0,
        ],
        "kind": scene.kind,
        "title": title,
    }


class CameraPath:
    """Per-frame target cameras from the keyboard, a preset script or the scene's original path
    (scenes without one fall back to the keyboard)."""

    def __init__(self, scene: Scene, folder: Path, camera: str, speed: float, browser: str) -> None:
        self.rig = CameraRig(scene, RigConfig(move_speed=RigConfig().move_speed * speed))
        self.controls = ControlLog(browser)
        self.trajectory: torch.Tensor | None = None
        self.script: list[CameraAction] | None = None
        preset = next((p for p in PRESETS if p["name"] == camera), {"kind": "keyboard"})
        if preset["kind"] == "trajectory" and (folder / "trajectory.txt").exists():
            self.trajectory = torch.from_numpy(read_trajectory(folder / "trajectory.txt"))
        elif preset["kind"] == "moves":
            plan = parse_moves(tuple(preset["moves"]))
            self.script = [action for action, frames in plan for _ in range(frames)]

    @property
    def length(self) -> int | None:
        if self.trajectory is not None:
            return len(self.trajectory)
        return len(self.script) if self.script is not None else None

    def next(
        self, start: int, count: int, sample_times: list[float]
    ) -> tuple[torch.Tensor, list[Any], list[int | None]]:
        """Target world-to-camera poses of frames `start..start+count`, their actions and seqs;
        the keyboard is read at each frame's wall-clock `sample_times`."""

        if self.trajectory is not None:
            index = torch.arange(start, start + count).clamp(max=len(self.trajectory) - 1)
            return self.trajectory[index], [None] * count, [None] * count
        if self.script is not None:
            actions = [self.script[min(start + i, len(self.script) - 1)] for i in range(count)]
            poses = torch.cat([self.rig.advance(action, 1) for action in actions])
            return poses, [asdict(a) for a in actions], [None] * count
        held = [self.controls.at(moment) for moment in sample_times]
        poses = torch.cat([self.rig.advance(action, 1) for action, _ in held])
        return poses, [asdict(action) for action, _ in held], [seq for _, seq in held]


# --- the session ---
def session_duration(*args: object) -> int:
    """GPU time to reserve: the requested session length plus loading (same args as run_session)."""

    return int(float(args[SECONDS_ARG])) + SESSION_MARGIN_SECONDS  # pyright: ignore[reportArgumentType]


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
    """Yield {"scene"} once, then {"frames", "renders", "poses", "actions", ...} per block,
    then {"ended": reason}."""

    folder = Path(scene_path)
    scene = load_scene(folder)
    yield {"scene": scene_info(scene, folder.name.replace("_", " "))}
    text_kv = world.encode_prompt(prompt)
    decoder = world.vae if quality == "Quality (Wan VAE decoder)" else world.decoder
    session = Session(scene, world.dit, world.vae, decoder, world.schedule, text_kv, seed)
    path = CameraPath(scene, folder, camera, speed, browser)
    fps = playback_fps(scene.fps)

    clock = time.time()  # t0: frame f plays the keys held at clock + f / fps
    while (reason := _end_reason(session, path, time.time() - clock, seconds)) is None:
        if stop_requested(browser, clock):
            reason = "Stopped"
            break

        # Start the block once the keys of its last frame are known
        first, count = session.frame_index, session.frames_needed
        sample_times = [clock + (first + i) / fps for i in range(count)]
        time.sleep(max(0.0, sample_times[-1] - time.time()))
        poses, actions, seqs = path.next(first, count, sample_times)
        block_start = time.perf_counter()
        block = session.step(poses)
        frames = block.frames.cpu().numpy()
        stride = RENDER_PREVIEW_STRIDE
        renders = block.render[:, ::stride, ::stride].cpu().numpy()
        block_seconds = time.perf_counter() - block_start
        valid = len(frames)
        yield {
            "frames": frames,
            "frame_start": first,
            "elapsed": time.time() - clock,
            "renders": renders,
            "poses": [np.linalg.inv(p).reshape(-1).tolist() for p in poses[:valid].numpy()],
            "actions": actions[:valid],
            "control_seqs": seqs[:valid],
            "stats": {
                "block_ms": round(block_seconds * 1000, 1),
                "gen_fps": round(valid / block_seconds, 1),
                "frames": session.frame_index,
                "elapsed_s": round(time.time() - clock, 1),
                "limit_s": seconds,
            },
        }

    yield {"ended": reason}


def _end_reason(session: Session, path: CameraPath, elapsed: float, seconds: float) -> str | None:
    if path.length is not None and session.frame_index >= path.length:
        return "End of the camera path"
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
) -> Iterator[dict[str, Any]]:
    if not scene_path:
        yield {"status": "error", "message": "Pick a scene first."}
        return
    browser, session_id = request.session_hash or "anonymous", uuid.uuid4().hex[:8]
    ControlLog.reset(browser)
    fps = playback_fps(load_scene_fps(scene_path))
    yield {"status": "loading", "message": "Waiting for a GPU…"}

    chunk_ids, scene = itertools.count(), None
    blocks = run_session(scene_path, prompt, quality, camera, speed, int(seed), seconds, browser)
    try:
        for item in blocks:
            if "scene" in item:
                scene = item["scene"]
                continue
            if "ended" in item:
                yield {"status": "ended", "message": item["ended"]}
                return
            chunk = {
                "id": next(chunk_ids),
                "session": session_id,
                "fps": fps,
                "frame_start": item["frame_start"],
                "elapsed": item["elapsed"],
                "latency": LATENCY_SECONDS,
                "frames": _jpeg_uris(item["frames"], quality=85),
                "renders": _jpeg_uris(item["renders"], quality=75),
                "poses": item["poses"],
                "actions": item["actions"],
                "control_seqs": item["control_seqs"],
            }
            yield {"status": "running", "chunk": chunk, "stats": item["stats"], "scene": scene}
            scene = None  # the point cloud goes out once, with the first chunk
    except gr.Error as error:
        # e.g. ZeroGPU's quota message: show it in the viewer instead of a bare "Error"
        yield {"status": "error", "message": str(error.message)}


def _jpeg_uris(frames: np.ndarray, quality: int) -> list[str]:
    return [
        encode_frame(
            simplejpeg.encode_jpeg(np.ascontiguousarray(frame), quality=quality, colorspace="RGB")
        )
        for frame in frames
    ]


def playback_fps(scene_fps: float) -> float:
    return min(scene_fps, PLAYBACK_FPS)


def load_scene_fps(scene_path: str) -> float:
    return float(json.loads((Path(scene_path) / "scene.json").read_text())["fps"])


def on_control(event: gr.EventData, request: gr.Request) -> None:
    ControlLog.append(request.session_hash or "anonymous", event._data)


def on_stop(request: gr.Request) -> dict[str, Any]:
    write_state(request.session_hash or "anonymous", "stop", {"time": time.time()})
    return {"status": "ended", "message": "Stopped"}


# --- UI ---
TITLE = """
<div style="display:flex;align-items:baseline;gap:.75rem;flex-wrap:wrap">
<h2 style="margin:0">InSpatio-World 1.5</h2>
<span style="opacity:.7;font-size:.9rem">
<a href="https://github.com/julien-blanchon/inspatio-world-v1.5">Code</a> ·
<a href="https://huggingface.co/blanchon/inspatio-world-v1.5">Weights</a> ·
<a href="https://arxiv.org/abs/2604.07209">Paper</a></span></div>
"""

with gr.Blocks(title="InSpatio-World 1.5") as demo:
    gr.HTML(TITLE)
    scene_path = gr.State(None)
    camera = gr.State(KEYBOARD)
    viewer = WorldViewer(show_label=False, show_render=True, latency=LATENCY_SECONDS)
    with gr.Row():
        with gr.Column(scale=3):
            gallery = gr.Gallery(
                value=example_choices(),
                label="Scenes",
                columns=5,
                height=230,
                allow_preview=False,
            )
        with gr.Column(scale=2):
            presets = gr.Gallery(
                value=preset_choices(),
                label="Camera path",
                columns=4,
                height=230,
                allow_preview=False,
                selected_index=0,  # the keyboard
            )
    with gr.Accordion("Scene & prompt", open=False):
        with gr.Row():
            poster = gr.Image(label="Scene", interactive=False, height=180)
            prompt = gr.Textbox(
                label="Prompt",
                lines=5,
                placeholder="Optional; video uploads are captioned automatically",
            )
        with gr.Row():
            images = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Your images (1 to 4 of one place)",
            )
            video = gr.Video(label="Your video", sources=["upload"])
        build = gr.Button("Build scene from upload")
    with gr.Accordion("Advanced", open=False):
        quality = gr.Radio(
            ["Fast (TAEHV decoder)", "Quality (Wan VAE decoder)"],
            value="Fast (TAEHV decoder)",
            label="Decoder",
        )
        speed = gr.Slider(0.25, 3.0, value=1.0, step=0.25, label="Movement speed")
        seconds = gr.Slider(
            15, SESSION_SECONDS, value=DEFAULT_SESSION_SECONDS, step=5, label="Session length (s)"
        )
        seed = gr.Number(0, label="Seed", precision=0)

    gallery.select(select_example, None, [scene_path, poster, prompt])
    presets.select(select_preset, None, camera)
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
    )
