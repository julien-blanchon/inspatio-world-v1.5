"""Build a scene folder from pictures: depth and cameras are estimated with Depth-Anything-3.

inspatio-world prepare photo.jpg --output scenes/photo
inspatio-world prepare view0.png view1.png view2.png view3.png --output scenes/room
inspatio-world prepare clip.mp4 --prompt "A street at dusk" --output scenes/street
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from inspatio_world import Scene, WorldConfig, load_depth_estimator
from inspatio_world.data import load_scene, read_image, read_video, save_scene

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
IMAGE_FPS = 15.0  # playback rate of image scenes, as the upstream examples


@dataclass(frozen=True, slots=True)
class SceneSource:
    """Where the scene comes from."""

    paths: tuple[Path, ...]
    """A scene folder, one image, several images of one place, or one video."""
    prompt: str | None = None
    """Scene description (defaults to the folder's prompt, else empty)."""
    max_frames: int = 240
    """Longest video prefix to use (depth estimation attends across all frames)."""


@dataclass(frozen=True, slots=True)
class PrepareConfig:
    """Estimate depth and cameras for pictures and save them as a scene folder."""

    source: SceneSource
    output: Path
    world: WorldConfig = field(default_factory=WorldConfig)


def build_scene(source: SceneSource, world: WorldConfig) -> Scene:
    """The glue from any accepted input to a `Scene`; the only place that knows file kinds."""

    first = source.paths[0]
    if first.is_dir():
        scene = load_scene(first)
        return scene if source.prompt is None else _with_prompt(scene, source.prompt)

    estimator = load_depth_estimator(world)
    prompt = source.prompt or ""
    if first.suffix.lower() in VIDEO_SUFFIXES:
        frames, fps = read_video(first, max_frames=source.max_frames)
        return Scene.estimate("video", frames, estimator, prompt=prompt, fps=fps)
    images = np.stack([read_image(path) for path in source.paths])
    return Scene.estimate("image", images, estimator, prompt=prompt, fps=IMAGE_FPS)


def _with_prompt(scene: Scene, prompt: str) -> Scene:
    from dataclasses import replace

    return replace(scene, prompt=prompt)


def main(config: PrepareConfig) -> None:
    scene = build_scene(config.source, config.world)
    save_scene(scene, config.output)
    print(f"{scene.kind} scene with {scene.num_views} views -> {config.output}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(PrepareConfig))
