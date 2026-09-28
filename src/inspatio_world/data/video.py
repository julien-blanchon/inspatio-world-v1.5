"""Video and image files <-> uint8 RGB arrays at the model's 832x480 frame, via PyAV and Pillow.

The world model works at one resolution, so every picture entering the package passes through
`fit_frame`: scaled to cover 832x480 and center-cropped, as upstream's ffmpeg filter
`scale=832:480:force_original_aspect_ratio=increase,crop=832:480` does.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

from ..types import ClipArray, ImageArray

FRAME_WIDTH, FRAME_HEIGHT = 832, 480


def fit_frame(image: Image.Image) -> ImageArray:
    """Scale an image to cover the model frame and center-crop it to exactly 832x480."""

    scale = max(FRAME_WIDTH / image.width, FRAME_HEIGHT / image.height)
    width, height = round(image.width * scale), round(image.height * scale)
    image = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    left, top = (width - FRAME_WIDTH) // 2, (height - FRAME_HEIGHT) // 2
    return np.asarray(image.crop((left, top, left + FRAME_WIDTH, top + FRAME_HEIGHT)))


def read_image(path: Path) -> ImageArray:
    with Image.open(path) as image:
        return fit_frame(image)


def read_video(path: Path, max_frames: int | None = None) -> tuple[ClipArray, float]:
    """Decode a video to `(frames, fps)`, each frame fitted to 832x480."""

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 24)
        for frame in container.decode(stream):
            frames.append(fit_frame(frame.to_image()))
            if max_frames is not None and len(frames) == max_frames:
                break
    return np.stack(frames), fps


def write_video(path: Path, frames: Iterable[ImageArray], fps: float, crf: int = 18) -> None:
    """Encode RGB frames as H.264 (yuv420p) MP4."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        stream.pix_fmt = "yuv420p"
        stream.width, stream.height = FRAME_WIDTH, FRAME_HEIGHT
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for image in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(stream.encode())
