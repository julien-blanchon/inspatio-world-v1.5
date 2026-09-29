"""Render the demo's camera-path presets on one example scene, with animated previews.

Each preset is a script of held moves (`controls.parse_moves`), so it replays on any scene through
the same `CameraRig` as the keyboard; this renders every preset on `image_example_00` and writes

    trajectories/presets.json      [{"name", "kind", "moves"?, "preview"}, ...]: first the
                                   keyboard ("keyboard"), then the scene's own upstream path
                                   ("trajectory"), then the move scripts ("moves")
    trajectories/<name>.webp       animated preview (the generated video, small)
    trajectories/<name>.mp4        the full-resolution generation

With `--splat-only` the previews show the splatted point cloud along the path instead of the
model's output: no GPU or model needed, still enough to see each camera motion.

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
from PIL import Image, ImageDraw, ImageFont

from inspatio_world import CameraRig, WorldConfig, WorldModel, parse_moves
from inspatio_world.data import load_scene, read_trajectory, write_video
from inspatio_world.render import lift_views, render

# Forward-led paths, as upstream's example trajectories: the camera travels ~2-3 median scene
# depths with gentle turns, which keeps most of the splatted render valid (fewer holes to invent)
# than sideways or orbiting motion. 240 frames = 20 s at the demo's 12 fps.
PRESETS: dict[str, tuple[str, ...]] = {
    "Stroll": (
        "forward*0.9+turn-left*0.2:60",
        "forward*0.9+turn-right*0.2:120",
        "forward*0.9+turn-left*0.2:60",
    ),
    "Wander": (
        "forward*0.8+turn-right*0.25:40",
        "forward*0.9+left*0.25+turn-left*0.3:100",
        "forward*0.7+look-up*0.15:100",
    ),
    "Curve right": ("forward*0.9+turn-right*0.18+right*0.05:240",),
    "Slalom": (
        "forward*0.8+left*0.35+turn-left*0.15:40",
        "forward*0.8+right*0.35+turn-right*0.15:80",
        "forward*0.8+left*0.35+turn-left*0.15:80",
        "forward*0.8:40",
    ),
    "Rise": ("forward*0.8+up*0.3+look-down*0.1:240",),
    "Dive": ("forward*0.9+down*0.1+look-down*0.08:120", "forward*0.8+up*0.05+look-up*0.1:120"),
    "Glide & look up": ("forward*0.9:120", "forward*0.7+look-up*0.25:60", "forward*0.7:60"),
    "Arc": ("right*0.6+turn-left*0.35+forward*0.25:240",),
}
PREVIEW_WIDTH, PREVIEW_STRIDE = 320, 2
# Splat-only previews stop halfway: past that, a single-image scene is mostly behind the camera
# (the model invents it, the bare splat is black), for the upstream path as for the presets.
SPLAT_PREVIEW_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class RenderConfig:
    examples: Path
    """The examples folder of the weights repository (scene folders)."""
    scene: str = "image_example_00"
    seed: int = 0
    splat_only: bool = False
    """Preview the splatted point cloud instead of generating (CPU, no model)."""
    world: WorldConfig = field(default_factory=lambda: WorldConfig(dit_precision="fp8"))


def preset_path(scene_dir: Path, moves: tuple[str, ...]) -> torch.Tensor:
    """The target cameras of a move script on a scene, through the keyboard's `CameraRig`."""

    rig = CameraRig(load_scene(scene_dir))
    return torch.cat([rig.advance(action, frames) for action, frames in parse_moves(moves)])


def render_model(world: WorldModel, scene_dir: Path, path: torch.Tensor, seed: int) -> np.ndarray:
    session = world.start(load_scene(scene_dir), seed=seed)
    frames = []
    while session.frame_index < len(path):
        index = torch.arange(session.frame_index, session.frame_index + session.frames_needed)
        frames.append(session.step(path[index.clamp(max=len(path) - 1)]).frames.cpu().numpy())
    return np.concatenate(frames)[: len(path)]


def render_splat(scene_dir: Path, path: torch.Tensor) -> np.ndarray:
    """The point cloud splatted along the path, every `PREVIEW_STRIDE`-th frame."""

    scene = load_scene(scene_dir)
    images = scene.images.permute(0, 3, 1, 2).float() / 127.5 - 1
    views = lift_views(images, scene.depth, scene.intrinsics, scene.world_to_camera)
    index = torch.zeros(1, 1, dtype=torch.long)
    frames = []
    for pose in path[::PREVIEW_STRIDE]:
        rgb = render(views, index, pose[None], scene.intrinsics[0]).rgb[0]
        frames.append(((rgb.permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy())
    return np.stack(frames)


def keyboard_tile(path: Path) -> None:
    """The gallery tile of the keyboard mode: W A S D keycaps on a dark background."""

    image = Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_WIDTH * 480 // 832), (24, 26, 32))
    draw = ImageDraw.Draw(image)
    size, gap = 44, 8
    cx, cy = image.width // 2, image.height // 2
    keys = {"W": (cx - size // 2, cy - size - gap // 2)}
    for offset, key in zip((-1, 0, 1), "ASD", strict=True):
        keys[key] = (cx - size // 2 + offset * (size + gap), cy + gap // 2)
    font = ImageFont.load_default(size=22)
    for key, (x, y) in keys.items():
        draw.rounded_rectangle(
            (x, y, x + size, y + size),
            radius=8,
            fill=(236, 238, 242),
            outline=(249, 115, 22),
            width=3,
        )
        draw.text((x + size / 2, y + size / 2), key, fill=(24, 26, 32), font=font, anchor="mm")
    image.save(path)


def save_preview(frames: np.ndarray, path: Path) -> None:
    height = PREVIEW_WIDTH * frames.shape[1] // frames.shape[2]
    preview = [Image.fromarray(f).resize((PREVIEW_WIDTH, height)) for f in frames]
    preview[0].save(
        path,
        save_all=True,
        append_images=preview[1:],
        duration=1000 * PREVIEW_STRIDE // 15,
        loop=0,
        quality=70,
    )


def main(config: RenderConfig) -> None:
    world = None if config.splat_only else WorldModel.from_pretrained(config.world)
    out = config.examples / "trajectories"
    out.mkdir(parents=True, exist_ok=True)
    scene_dir = config.examples / config.scene

    keyboard_tile(out / "keyboard.png")
    entries = [
        {"name": "Keyboard", "kind": "keyboard", "preview": "keyboard.png"},
        {"name": "Original path", "kind": "trajectory", "preview": "original-path.webp"},
    ]
    paths = {"original-path": torch.from_numpy(read_trajectory(scene_dir / "trajectory.txt"))}
    for name, moves in PRESETS.items():
        slug = name.lower().replace(" ", "-").replace("&", "and")
        paths[slug] = preset_path(scene_dir, moves)
        entries.append(
            {"name": name, "kind": "moves", "moves": list(moves), "preview": f"{slug}.webp"}
        )

    for slug, path in paths.items():
        if world is None:
            shown = path[: max(1, int(len(path) * SPLAT_PREVIEW_FRACTION))]
            save_preview(render_splat(scene_dir, shown), out / f"{slug}.webp")
        else:
            frames = render_model(world, scene_dir, path, config.seed)
            write_video(out / f"{slug}.mp4", frames, fps=15)
            save_preview(frames[::PREVIEW_STRIDE], out / f"{slug}.webp")
        print(f"{slug}: {len(path)} frames")
    (out / "presets.json").write_text(json.dumps(entries, indent=1))


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))
