"""Convert upstream's example scenes into scene folders for the demo (one-off, like the weights).

Upstream ships each example as `input/` (views or a video, prompt, target trajectory) plus, for
images, `depth/` (uint16 PNG depth with a min/max file, intrinsics and poses as text). Image
examples keep that depth and those cameras exactly; video examples ship none, so depth and
cameras are estimated here with the same Depth-Anything-3 path as upstream's runner. Each output
folder is a scene folder (`data.scene_files`) plus `trajectory.txt` (upstream's target path) and
`poster.jpg` (the gallery thumbnail).

    uv run scripts/prepare_examples.py --upstream /path/to/inspatio-world-v1.5 --output examples
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image

from inspatio_world import Scene, WorldConfig, load_depth_estimator
from inspatio_world.data import read_video, save_scene, write_trajectory
from inspatio_world.data.scene_files import read_trajectory

UINT16_MAX = 65_535
POSTER_WIDTH = 416


@dataclass(frozen=True, slots=True)
class ExamplesConfig:
    upstream: Path
    """Checkout of https://github.com/inspatio/inspatio-world-v1.5."""
    output: Path
    """Folder receiving one scene folder per example."""
    max_video_frames: int = 240
    """Video examples are cut to this many frames (depth estimation attends across all)."""
    world: WorldConfig = field(default_factory=WorldConfig)


def image_example(folder: Path) -> Scene:
    meta = json.loads((folder / "scene.json").read_text())
    views = sorted((folder / "input").glob("view_*.png"))
    depth_dir = folder / "depth"
    depth_paths = (
        [depth_dir / "depth.png"]
        if len(views) == 1
        else [depth_dir / f"depth_{index:02d}.png" for index in range(len(views))]
    )
    low, high = np.loadtxt(depth_dir / "metadata.txt")
    depth = np.stack(
        [
            low + np.asarray(Image.open(path), dtype=np.float64) / UINT16_MAX * (high - low)
            for path in depth_paths
        ]
    )
    return Scene(
        kind="image",
        images=torch.from_numpy(
            np.stack([np.asarray(Image.open(p).convert("RGB")) for p in views])
        ),
        depth=torch.from_numpy(depth.astype(np.float32)),
        intrinsics=torch.from_numpy(
            np.loadtxt(depth_dir / "source_intrinsics.txt", ndmin=2).reshape(-1, 3, 3)
        ).float(),
        world_to_camera=torch.from_numpy(
            np.loadtxt(depth_dir / "source_tcw.txt", ndmin=2).reshape(-1, 4, 4)
        ).float(),
        prompt=(folder / "input" / "prompt.txt").read_text().strip(),
        fps=float(meta["fps"]),
    )


def main(config: ExamplesConfig) -> None:
    examples = config.upstream / "examples"
    estimator = None
    for name in json.loads((examples / "manifest.json").read_text()):
        folder = examples / name
        meta = json.loads((folder / "scene.json").read_text())
        trajectory = read_trajectory(folder / "input" / "target_tcw.txt")
        if meta["kind"] == "image":
            scene = image_example(folder)
            trajectory = trajectory[: meta["valid_frames"]]
        else:
            estimator = estimator or load_depth_estimator(config.world)
            frames, fps = read_video(
                folder / "input" / "video.mp4", max_frames=config.max_video_frames
            )
            prompt = (folder / "input" / "prompt.txt").read_text().strip()
            scene = Scene.estimate("video", frames, estimator, prompt=prompt, fps=fps)
            trajectory = trajectory[: len(frames)]

        target = config.output / name
        shutil.rmtree(target, ignore_errors=True)
        save_scene(scene, target)
        write_trajectory(target / "trajectory.txt", trajectory)
        poster = Image.fromarray(scene.images[0].numpy())
        poster.resize((POSTER_WIDTH, POSTER_WIDTH * poster.height // poster.width)).save(
            target / "poster.jpg", quality=85
        )
        print(f"{name}: {scene.kind}, {scene.num_views} views, {len(trajectory)} target frames")


if __name__ == "__main__":
    main(tyro.cli(ExamplesConfig))
