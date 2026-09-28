"""The scene: source views with depth and cameras, the only input form the world model knows.

A scene is one or more 832x480 RGB views, each with a depth map, pinhole intrinsics and a
world-to-camera pose (OpenCV axes, world = the first view's camera). It comes in two kinds:

    image   1 view (or 4 views of one place); every generated frame may see every view, and the
            generation can run for as long as the camera keeps moving
    video   one view per video frame; generated frame f is conditioned on source frame f, so the
            generation lasts at most as many frames as the video

Target cameras are expressed in the same world frame and share the first view's intrinsics.
Scenes are built from pictures with the depth estimator (`Scene.estimate`) or loaded from a scene
folder (`data.scene_files`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import UInt8
from torch import Tensor

from .depth import DepthEstimator
from .types import ClipArray, DepthMaps, Intrinsics, Poses

SceneKind = Literal["image", "video"]


@dataclass(frozen=True, slots=True)
class Scene:
    """Source views of one scene, on the CPU, in the canonical internal form."""

    kind: SceneKind
    images: UInt8[Tensor, "V H W 3"]
    depth: DepthMaps
    intrinsics: Intrinsics
    world_to_camera: Poses
    prompt: str
    fps: float

    @property
    def num_views(self) -> int:
        return self.images.shape[0]

    @property
    def max_frames(self) -> int | None:
        """Frames a generation can last: the video's length, or unbounded for images."""

        return self.num_views if self.kind == "video" else None

    @classmethod
    def estimate(
        cls,
        kind: SceneKind,
        images: ClipArray,
        estimator: DepthEstimator,
        prompt: str = "",
        fps: float = 15.0,
    ) -> Scene:
        """Build a scene from 832x480 pictures, estimating depth and cameras jointly."""

        estimate = estimator.estimate(np.ascontiguousarray(images))
        return cls(
            kind=kind,
            images=torch.from_numpy(np.ascontiguousarray(images)),
            depth=estimate.depth.float().cpu(),
            intrinsics=estimate.intrinsics.float().cpu(),
            world_to_camera=estimate.world_to_camera.float().cpu(),
            prompt=prompt,
            fps=fps,
        )
