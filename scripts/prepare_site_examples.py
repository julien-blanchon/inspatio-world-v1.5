"""Turn demos from the InSpatio-World 1.5 project page into scene folders (one-off, like the weights).

The project page (https://inspatio.github.io/inspatio-world-1.5/) shows generated videos. Their
first frames make good single-image scenes (scene roaming), and the "dynamic" results are
themselves realistic videos that work as video scenes once the small input inset in their
top-left corner is cropped away. Depth and cameras come from Depth-Anything-3; video prompts are
written by Florence-2-base (`<MORE_DETAILED_CAPTION>`), as the upstream v1 pipeline captions its
inputs with Florence-2; image scenes keep the empty prompt of the upstream image examples.

    uv run --extra caption scripts/prepare_site_examples.py --output /path/to/staging/examples
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

SITE = "https://inspatio.github.io/inspatio-world-1.5/static/videos"
# name -> (video, kind, crop box (left, top, right, bottom) on the 1280x720 frame or None)
DEMOS: dict[str, tuple[str, str, tuple[int, int, int, int] | None]] = {
    "castle": ("scene_castle", "image", None),
    "corridor": ("scene_corridor", "image", None),
    "bookstore": ("input_bookstore", "image", None),
    "celestial_palace": ("input_generated", "image", None),
    "boat_ride": ("dynamic_boat_combined", "video", (420, 238, 1280, 720)),
    "oranges": ("dynamic_oranges_combined", "video", (420, 238, 1280, 720)),
    "rainy_cottage": ("dynamic_rain_combined", "video", (420, 238, 1280, 720)),
    "tornado": ("dynamic_tornado_combined", "video", (420, 238, 1280, 720)),
    "flour": ("dynamic_flour_combined", "video", (420, 238, 1280, 720)),
}
IMAGE_FPS = 15.0
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
POSTER_WIDTH = 416


@dataclass(frozen=True, slots=True)
class SiteConfig:
    output: Path
    """The examples folder of the weights repository."""
    max_video_frames: int = 240
    videos: Path | None = None
    """Folder with the project page videos already downloaded (else they are downloaded)."""
    world: WorldConfig = field(default_factory=WorldConfig)


def read_frames(
    path: Path, crop: tuple[int, int, int, int] | None, limit: int
) -> tuple[np.ndarray, float]:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        for frame in container.decode(stream):
            image = frame.to_image()
            frames.append(fit_frame(image.crop(crop) if crop else image))
            if len(frames) == limit:
                break
    return np.stack(frames), fps


class Captioner:
    """Florence-2-base detailed captions, from the `captioner/` folder of the weights repo."""

    def __init__(self, world: WorldConfig) -> None:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        folder = component_dir(world.repo_id, world.revision, "captioner")
        self.processor = AutoProcessor.from_pretrained(folder)
        self.model = Florence2ForConditionalGeneration.from_pretrained(folder, dtype=torch.bfloat16)
        self.model = self.model.to(world.device).eval()

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> str:
        picture = Image.fromarray(image)
        inputs = self.processor(text=CAPTION_TASK, images=picture, return_tensors="pt")
        inputs = inputs.to(self.model.device, torch.bfloat16)
        ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=3)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=CAPTION_TASK, image_size=picture.size
        )
        return str(parsed[CAPTION_TASK]).strip()


def main(config: SiteConfig) -> None:
    estimator = load_depth_estimator(config.world)
    captioner = Captioner(config.world)
    with tempfile.TemporaryDirectory() as downloads:
        for name, (video, kind, crop) in DEMOS.items():
            path = (config.videos or Path(downloads)) / f"{video}.mp4"
            if not path.exists():
                urllib.request.urlretrieve(f"{SITE}/{video}.mp4", path)
            limit = 1 if kind == "image" else config.max_video_frames
            frames, fps = read_frames(path, crop, limit)
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
            print(f"{name}: {kind}, {len(frames)} frames, prompt: {prompt[:80]!r}")


if __name__ == "__main__":
    main(tyro.cli(SiteConfig))
