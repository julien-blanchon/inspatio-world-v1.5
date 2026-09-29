"""Turn free stock videos from Mixkit into example scenes (one-off, like the weights).

Mixkit clips (https://mixkit.co/license/#videoFree) are full-HD footage with real depth: steady
shots with moving content (crowds, rain, waves, water) become video scenes, and single frames of
walkable places (paths, alleys, naves) become image scenes. Depth and cameras come from
Depth-Anything-3; video prompts are written by Florence-2-base (`<MORE_DETAILED_CAPTION>`), as the
upstream v1 pipeline captions its inputs with Florence-2; image scenes keep the empty prompt of the
upstream image examples.

    uv run --extra caption scripts/prepare_stock_examples.py --output /path/to/staging/examples
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
import torch
import tyro
from PIL import Image

from inspatio_world import Scene, WorldConfig, load_depth_estimator
from inspatio_world.data import fit_frame, save_scene
from inspatio_world.utils import component_dir

SOURCE = "https://assets.mixkit.co/videos/{id}/{id}-{height}.mp4"
# name -> (Mixkit video id, kind, first frame used)
CLIPS: dict[str, tuple[int, str, int]] = {
    "tokyo_night_street": (4451, "video", 0),
    "times_square_rain": (4332, "video", 0),
    "waves_on_rocks": (9294, "video", 0),
    "mossy_waterfall": (45315, "video", 0),
    "bamboo_raft": (1218, "video", 0),
    "venice_alley": (4600, "image", 0),
    "blossom_path": (44970, "image", 0),
    "forest_road": (41574, "image", 0),
    "gothic_nave": (22723, "image", 0),
    "castle_lawn": (4074, "image", 0),
}
IMAGE_FPS = 15.0
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
POSTER_WIDTH = 416


@dataclass(frozen=True, slots=True)
class StockConfig:
    output: Path
    """The examples folder of the weights repository."""
    max_video_frames: int = 120
    frame_stride: int = 2
    """Keep every n-th video frame: 25-30 fps footage then plays at ~12-15 fps like the demo."""
    videos: Path | None = None
    """Folder with the clips already downloaded as `<id>.mp4` (else they are downloaded)."""
    world: WorldConfig = field(default_factory=WorldConfig)


def download(video_id: int, folder: Path) -> Path:
    path = folder / f"{video_id}.mp4"
    for height in (1080, 720):
        if path.exists():
            break
        request = urllib.request.Request(
            SOURCE.format(id=video_id, height=height), headers={"User-Agent": "Mozilla/5.0"}
        )
        try:
            with urllib.request.urlopen(request) as response:
                path.write_bytes(response.read())
        except OSError:
            continue
    return path


def read_frames(path: Path, start: int, limit: int, stride: int) -> tuple[np.ndarray, float]:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24) / stride
        for index, frame in enumerate(container.decode(stream)):
            if index >= start and (index - start) % stride == 0:
                frames.append(fit_frame(frame.to_image()))
                if len(frames) == limit:
                    break
    return np.stack(frames), fps


class Captioner:
    """Florence-2-base detailed captions, from the `captioner/` folder of the weights repo."""

    def __init__(self, world: WorldConfig) -> None:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        folder = component_dir(world.repo_id, world.revision, "captioner")
        self.processor = AutoProcessor.from_pretrained(folder)
        dtype = torch.bfloat16 if world.device != "cpu" else torch.float32
        self.model = Florence2ForConditionalGeneration.from_pretrained(folder, dtype=dtype)
        self.model = self.model.to(world.device).eval()

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> str:
        picture = Image.fromarray(image)
        inputs = self.processor(text=CAPTION_TASK, images=picture, return_tensors="pt")
        inputs = inputs.to(self.model.device, self.model.dtype)
        ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=3)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=CAPTION_TASK, image_size=picture.size
        )
        return str(parsed[CAPTION_TASK]).strip()


def main(config: StockConfig) -> None:
    estimator = load_depth_estimator(config.world)
    captioner = Captioner(config.world)
    with tempfile.TemporaryDirectory() as downloads:
        for name, (video_id, kind, start) in CLIPS.items():
            path = download(video_id, config.videos or Path(downloads))
            limit = 1 if kind == "image" else config.max_video_frames
            frames, fps = read_frames(path, start, limit, config.frame_stride)
            prompt = "" if kind == "image" else captioner(frames[0])
            scene = Scene.estimate(
                kind, frames, estimator, prompt=prompt, fps=IMAGE_FPS if kind == "image" else fps
            )  # pyright: ignore[reportArgumentType]

            target = config.output / name
            shutil.rmtree(target, ignore_errors=True)
            save_scene(scene, target)
            poster = Image.fromarray(frames[0])
            poster.resize((POSTER_WIDTH, POSTER_WIDTH * poster.height // poster.width)).save(
                target / "poster.jpg", quality=85
            )
            print(f"{name}: {kind}, {len(frames)} frames, prompt: {prompt[:80]!r}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(StockConfig))
