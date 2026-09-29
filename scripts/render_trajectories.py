"""Render the demo's camera-path presets on one example scene, with animated previews.

Each preset is a script of held moves (`controls.parse_moves`), so it replays on any scene through
the same `CameraRig` as the keyboard; this renders every preset on `image_example_00` and writes

    trajectories/presets.json      [{"name", "moves": [...], "preview": "<name>.webp"}, ...]
    trajectories/<name>.webp       animated preview (the generated video, small)
    trajectories/<name>.mp4        the full-resolution generation

into the examples folder of the weights repository, where the Space picks them up.

    uv run scripts/render_trajectories.py --examples /path/to/staging/examples
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image

from inspatio_world import CameraRig, WorldConfig, WorldModel, parse_moves
from inspatio_world.data import load_scene, write_video

# Strafing right while turning left at 0.57 circles a point at the median scene depth
PRESETS: dict[str, tuple[str, ...]] = {
    "Orbit": ("right+turn-left*0.57:120",),
    "Dolly in": ("forward:108",),
    "Crane up": ("up+look-down*0.4:108",),
    "Look around": ("turn-left:36", "turn-right:72", "turn-left:36"),
    "Spiral": ("forward*0.6+turn-right*0.7+up*0.5:120",),
    "Zig-zag": ("left+forward*0.5:30", "right+forward*0.5:48", "left+forward*0.5:42"),
    "Reveal": ("back+up*0.5+look-down*0.2:108",),
    "Barrel": ("forward*0.5+look-up*0.6:36", "forward*0.5+look-down*0.6:72"),
}
PREVIEW_WIDTH, PREVIEW_STRIDE = 320, 2


@dataclass(frozen=True, slots=True)
class RenderConfig:
    examples: Path
    """The examples folder of the weights repository (scene folders)."""
    scene: str = "image_example_00"
    seed: int = 0
    world: WorldConfig = field(default_factory=lambda: WorldConfig(dit_precision="fp8"))


def render_preset(
    world: WorldModel, scene_dir: Path, moves: tuple[str, ...], seed: int
) -> np.ndarray:
    scene = load_scene(scene_dir)
    rig = CameraRig(scene)
    path = torch.cat([rig.advance(action, frames) for action, frames in parse_moves(moves)])
    session = world.start(scene, seed=seed)
    frames = []
    while session.frame_index < len(path):
        index = torch.arange(session.frame_index, session.frame_index + session.frames_needed)
        frames.append(session.step(path[index.clamp(max=len(path) - 1)]).frames.cpu().numpy())
    return np.concatenate(frames)[: len(path)]


def main(config: RenderConfig) -> None:
    world = WorldModel.from_pretrained(config.world)
    out = config.examples / "trajectories"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for name, moves in PRESETS.items():
        slug = name.lower().replace(" ", "-")
        frames = render_preset(world, config.examples / config.scene, moves, config.seed)
        write_video(out / f"{slug}.mp4", frames, fps=15)
        height = PREVIEW_WIDTH * frames.shape[1] // frames.shape[2]
        preview = [
            Image.fromarray(f).resize((PREVIEW_WIDTH, height)) for f in frames[::PREVIEW_STRIDE]
        ]
        preview[0].save(
            out / f"{slug}.webp",
            save_all=True,
            append_images=preview[1:],
            duration=1000 * PREVIEW_STRIDE // 15,
            loop=0,
            quality=70,
        )
        index.append({"name": name, "moves": list(moves), "preview": f"{slug}.webp"})
        print(f"{name}: {len(frames)} frames")
    (out / "presets.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))
