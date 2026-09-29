"""Generate a video: a scene (folder, image(s) or video) + a camera path -> MP4.

The camera path is either a trajectory file (one world-to-camera 4x4 per frame, see
`data.scene_files`) or a sequence of held moves, e.g. `--moves forward:36 turn-right+forward:24`,
driven through the same `CameraRig` as the interactive demo.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from inspatio_world import CameraRig, Scene, WorldConfig, WorldModel
from inspatio_world.controls import parse_moves
from inspatio_world.data import read_trajectory, write_video
from inspatio_world.scripts.prepare import SceneSource, build_scene


@dataclass(frozen=True, slots=True)
class GenerateConfig:
    """Generate a video from a scene and a camera path."""

    source: SceneSource
    output: Path = Path("output.mp4")
    """Where to write the generated MP4."""
    trajectory: Path | None = None
    """Per-frame world-to-camera matrices (16 numbers per line); overrides `moves`."""
    moves: tuple[str, ...] = ("forward:24", "turn-right+forward:36", "turn-left:24")
    """Held moves `name[*scale][+name[*scale]]:frames`, names from: forward back left right up
    down turn-left turn-right look-up look-down still (e.g. `right+turn-left*0.8:48` orbits)."""
    seed: int = 0
    save_render: bool = True
    """Also write the splatted condition next to the output (`*_render.mp4`)."""
    world: WorldConfig = field(default_factory=WorldConfig)


def camera_path(scene: Scene, config: GenerateConfig) -> torch.Tensor:
    if config.trajectory is not None:
        return torch.from_numpy(read_trajectory(config.trajectory))
    rig = CameraRig(scene)
    return torch.cat([rig.advance(action, frames) for action, frames in parse_moves(config.moves)])


def main(config: GenerateConfig) -> None:
    world = WorldModel.from_pretrained(config.world)
    scene = build_scene(config.source, config.world)
    path = camera_path(scene, config)
    total = len(path) if scene.max_frames is None else min(len(path), scene.max_frames)

    session = world.start(scene, seed=config.seed)
    frames, renders = [], []
    started = time.perf_counter()
    while session.frame_index < total:
        index = torch.arange(session.frame_index, session.frame_index + session.frames_needed)
        block = session.step(path[index.clamp(max=len(path) - 1)])
        frames.append(block.frames.cpu().numpy())
        renders.append(block.render.cpu().numpy())
        print(f"frame {min(session.frame_index, total)}/{total}", flush=True)
    elapsed = time.perf_counter() - started

    write_video(config.output, np.concatenate(frames)[:total], scene.fps)
    if config.save_render:
        render_path = config.output.with_name(f"{config.output.stem}_render.mp4")
        write_video(render_path, np.concatenate(renders)[:total], scene.fps)
    print(f"{total} frames in {elapsed:.1f}s ({total / elapsed:.1f} fps) -> {config.output}")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(GenerateConfig))
