"""Camera control: per-frame actions (move, turn) -> target world-to-camera poses.

A `CameraRig` flies a camera relative to the scene's source camera, like a first-person game:
forward / right / up translate along the camera's level heading, yaw turns around the world's
vertical axis of the source camera, pitch tilts the view. Speeds are relative to the scene: one
unit of `move_speed` per frame is that fraction of the source view's median depth, so the same
key press feels the same in a room and on a street.

    image scenes   the offset is applied to the (single, fixed) source camera
    video scenes   the offset rides on the source camera of each frame, so the user steers
                   relative to the original camera motion

Actions are smoothed with a first-order filter (`smoothing` per frame) so key presses ease in and
out instead of jerking the camera, which also keeps the render condition temporally coherent.
Poses use OpenCV axes (x right, y down, z forward).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch

from .scene import Scene
from .types import Poses

UP = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)  # OpenCV y points down


@dataclass(frozen=True, slots=True)
class CameraAction:
    """Desired motion per axis in [-1, 1]; positive = forward, right, up, turn right, look up."""

    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0


@dataclass(frozen=True, slots=True)
class RigConfig:
    move_speed: float = 0.012  # per frame, as a fraction of the source view's median depth
    turn_speed: float = 1.2  # degrees per frame at full yaw / pitch
    max_pitch: float = 60.0  # degrees
    smoothing: float = 0.25  # per-frame blend towards the requested action


@dataclass(slots=True)
class CameraRig:
    """Integrates actions frame by frame into target poses for a `Session`."""

    scene: Scene
    config: RigConfig = field(default_factory=RigConfig)
    position: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    yaw: float = 0.0
    pitch: float = 0.0
    velocity: CameraAction = field(default_factory=CameraAction)
    frame: int = 0
    scale: float = field(init=False)  # the source view's median depth

    def __post_init__(self) -> None:
        depth = self.scene.depth[0]
        self.scale = float(depth[depth > 0].median()) if (depth > 0).any() else 1.0

    def advance(self, action: CameraAction, frames: int) -> Poses:
        """Target world-to-camera poses of the next `frames` frames under a held action."""

        poses = []
        for _ in range(frames):
            self._integrate(action)
            poses.append(torch.linalg.inv(self._anchor() @ self._offset()))
            self.frame += 1
        return torch.stack(poses).float()

    def _integrate(self, action: CameraAction) -> None:
        blend = self.config.smoothing
        current = self.velocity
        self.velocity = replace(
            current,
            forward=current.forward + blend * (action.forward - current.forward),
            right=current.right + blend * (action.right - current.right),
            up=current.up + blend * (action.up - current.up),
            yaw=current.yaw + blend * (action.yaw - current.yaw),
            pitch=current.pitch + blend * (action.pitch - current.pitch),
        )
        turn = math.radians(self.config.turn_speed)
        self.yaw += self.velocity.yaw * turn
        limit = math.radians(self.config.max_pitch)
        self.pitch = min(max(self.pitch + self.velocity.pitch * turn, -limit), limit)

        # Translate along the level heading (yaw only), so looking up does not fly upwards
        heading = _rotation_y(self.yaw)
        step = self.config.move_speed * self.scale
        self.position = self.position + step * (
            self.velocity.forward * heading[:, 2]
            + self.velocity.right * heading[:, 0]
            + self.velocity.up * UP
        )

    def _offset(self) -> torch.Tensor:
        """The rig's camera-to-anchor transform: turn (yaw, then pitch), then translate."""

        offset = torch.eye(4, dtype=torch.float64)
        # Pitch about the camera x axis: a positive angle tilts +z towards -y, i.e. looks up
        offset[:3, :3] = _rotation_y(self.yaw) @ _rotation_x(self.pitch)
        offset[:3, 3] = self.position
        return offset

    def _anchor(self) -> torch.Tensor:
        """Camera-to-world of the source camera this frame is relative to."""

        view = min(self.frame, self.scene.num_views - 1) if self.scene.kind == "video" else 0
        return torch.linalg.inv(self.scene.world_to_camera[view].double())


def _rotation_y(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


def _rotation_x(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)
