"""Scene folders: the on-disk form of a `Scene`, and camera trajectory text files.

A scene folder holds

    scene.json    {"kind", "fps", "prompt", "depth_range": [min, max],
                   "intrinsics": V x 3 x 3, "world_to_camera": V x 4 x 4}
    view_00.png   one lossless PNG per view (kind "image"), or
    video.mp4     the source frames (kind "video", H.264 crf 18)
    depth.mkv     every view's depth as lossless 16-bit gray FFV1: depth = min + q / 65535 * range
    trajectory.txt  (optional) a target camera path, see `read_trajectory`

A trajectory file has one target camera per line: 16 numbers, the row-major 4x4 world-to-camera
matrix (OpenCV axes), upstream's `target_tcw.txt` format.
"""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from ..scene import Scene
from ..types import DepthArray, PosesArray
from .video import read_video, write_video

DEPTH_LEVELS = 65_535


def save_scene(scene: Scene, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    depth = scene.depth.numpy()
    depth_range = (float(depth.min()), float(depth.max()))
    meta = {
        "kind": scene.kind,
        "fps": scene.fps,
        "prompt": scene.prompt,
        "depth_range": depth_range,
        "intrinsics": scene.intrinsics.tolist(),
        "world_to_camera": scene.world_to_camera.tolist(),
    }
    (folder / "scene.json").write_text(json.dumps(meta, indent=1))

    images = scene.images.numpy()
    if scene.kind == "video":
        write_video(folder / "video.mp4", images, scene.fps)
    else:
        for index, image in enumerate(images):
            Image.fromarray(image).save(folder / f"view_{index:02d}.png")
    _write_depth(folder / "depth.mkv", depth, depth_range)


def load_scene(folder: Path) -> Scene:
    meta = json.loads((folder / "scene.json").read_text())
    if meta["kind"] == "video":
        images, _ = read_video(folder / "video.mp4")
    else:
        paths = sorted(folder.glob("view_*.png"))
        images = np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths])
    depth = _read_depth(folder / "depth.mkv", tuple(meta["depth_range"]))
    return Scene(
        kind=meta["kind"],
        images=torch.from_numpy(images),
        depth=torch.from_numpy(depth),
        intrinsics=torch.tensor(meta["intrinsics"], dtype=torch.float32),
        world_to_camera=torch.tensor(meta["world_to_camera"], dtype=torch.float32),
        prompt=meta["prompt"],
        fps=float(meta["fps"]),
    )


def read_trajectory(path: Path) -> PosesArray:
    """One world-to-camera 4x4 per line, 16 row-major numbers (upstream `target_tcw.txt`)."""

    values = np.loadtxt(path, ndmin=2, dtype=np.float32)
    assert values.shape[1] == 16, f"{path}: expected 16 numbers per line, got {values.shape[1]}"
    return values.reshape(-1, 4, 4)


def write_trajectory(path: Path, world_to_camera: PosesArray) -> None:
    np.savetxt(path, world_to_camera.reshape(-1, 16), fmt="%.9g")


def _write_depth(path: Path, depth: DepthArray, depth_range: tuple[float, float]) -> None:
    low, high = depth_range
    scale = DEPTH_LEVELS / max(high - low, 1e-12)
    quantized = np.rint(np.clip((depth - low) * scale, 0, DEPTH_LEVELS)).astype(np.uint16)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=1)
        stream.pix_fmt = "gray16le"
        stream.width, stream.height = depth.shape[2], depth.shape[1]
        for plane in quantized:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(plane, format="gray16le")))
        container.mux(stream.encode())


def _read_depth(path: Path, depth_range: tuple[float, float]) -> DepthArray:
    low, high = depth_range
    with av.open(str(path)) as container:
        planes = [frame.to_ndarray(format="gray16le") for frame in container.decode(video=0)]
    return (low + np.stack(planes).astype(np.float32) / DEPTH_LEVELS * (high - low)).astype(
        np.float32
    )
