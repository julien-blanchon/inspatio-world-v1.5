"""Cameras of the depth estimator: DA3's camera-token decoder and InSpatio's trajectory smoothing.

`CameraDecoder` maps each view's camera token to a 9-number pose encoding -- camera-to-world
translation, xyzw quaternion, vertical and horizontal field of view -- and `decode_pose` turns
it into OpenCV world-to-camera extrinsics and a pinhole K with the principal point at the image
centre, in pixels of the processing resolution. This is DA3's `use_ray_pose=False` path, run in
float32 as upstream (autocast is off around its camera head).

`smooth_trajectory` is InSpatio's post-processing of the recovered path: each camera-to-world
pose becomes a quaternion (SciPy's `Rotation.from_matrix` convention, sign included, so the
smoothing sees the same component signs) plus a translation, every component is Gaussian
filtered over time (SciPy's `gaussian_filter1d`: reflect padding, radius `int(4 sigma + 0.5)`)
and the quaternions are renormalized. It runs in float64.

# Adapted from https://github.com/ByteDance-Seed/depth-anything-3/blob/main/src/depth_anything_3/model/cam_dec.py
# and .../model/utils/transform.py; smoothing from https://github.com/inspatio/inspatio-world-v1.5 (pipeline/depth_utils.py)
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..types import CameraTokens, Intrinsics, PoseEncoding, Poses, Quaternions

FOV_TAN_MIN = 1e-6
GAUSSIAN_TRUNCATE = 4.0  # SciPy's default kernel radius, in sigmas
ORTHOGONAL_ATOL = 1e-12  # SciPy re-orthogonalizes a rotation whose Gram matrix is off by more


class CameraDecoder(nn.Module):
    """MLP from the camera token to the pose encoding `(t, q_xyzw, fov_y, fov_x)`."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU()
        )
        self.fc_t = nn.Linear(dim, 3)
        self.fc_qvec = nn.Linear(dim, 4)
        self.fc_fov = nn.Linear(dim, 2)

    @override
    def forward(self, tokens: CameraTokens) -> PoseEncoding:
        hidden = self.backbone(tokens)
        return torch.cat(
            [self.fc_t(hidden), self.fc_qvec(hidden), F.relu(self.fc_fov(hidden))], dim=-1
        )


def quaternion_to_matrix(quaternions: Quaternions) -> torch.Tensor:
    """DA3's xyzw quaternion -> rotation, normalizing through `2 / |q|^2`."""

    i, j, k, r = quaternions.unbind(-1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    matrix = torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        dim=-1,
    )
    return rearrange(matrix, "... (r c) -> ... r c", r=3)


def decode_pose(encoding: PoseEncoding, size: tuple[int, int]) -> tuple[torch.Tensor, Intrinsics]:
    """Pose encoding -> world-to-camera `(V, 3, 4)` and intrinsics for an image of `size`."""

    height, width = size
    rotation = quaternion_to_matrix(encoding[:, 3:7])  # camera-to-world
    translation = encoding[:, :3, None]
    world_to_camera = torch.cat([rotation.mT, -rotation.mT @ translation], dim=-1)

    intrinsics = torch.zeros(encoding.shape[0], 3, 3, device=encoding.device)
    intrinsics[:, 0, 0] = (width / 2.0) / torch.clamp(torch.tan(encoding[:, 8] / 2.0), FOV_TAN_MIN)
    intrinsics[:, 1, 1] = (height / 2.0) / torch.clamp(torch.tan(encoding[:, 7] / 2.0), FOV_TAN_MIN)
    intrinsics[:, 0, 2] = width / 2
    intrinsics[:, 1, 2] = height / 2
    intrinsics[:, 2, 2] = 1.0
    return world_to_camera, intrinsics


def matrix_to_quaternion(rotation: torch.Tensor) -> Quaternions:
    """SciPy's `Rotation.from_matrix(...).as_quat()`: nearest rotation, largest-pivot branch."""

    gram = rotation @ rotation.mT
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    orthogonal = torch.isclose(gram, eye, rtol=1e-5, atol=ORTHOGONAL_ATOL).flatten(1).all(dim=1)
    u, _, vt = torch.linalg.svd(rotation)
    rotation = torch.where(orthogonal[:, None, None], rotation, u @ vt)

    m = rotation
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    candidates = torch.stack(
        [
            torch.stack(
                [
                    1 - trace + 2 * m[:, 0, 0],
                    m[:, 1, 0] + m[:, 0, 1],
                    m[:, 2, 0] + m[:, 0, 2],
                    m[:, 2, 1] - m[:, 1, 2],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[:, 1, 0] + m[:, 0, 1],
                    1 - trace + 2 * m[:, 1, 1],
                    m[:, 2, 1] + m[:, 1, 2],
                    m[:, 0, 2] - m[:, 2, 0],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[:, 2, 0] + m[:, 0, 2],
                    m[:, 2, 1] + m[:, 1, 2],
                    1 - trace + 2 * m[:, 2, 2],
                    m[:, 1, 0] - m[:, 0, 1],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[:, 2, 1] - m[:, 1, 2],
                    m[:, 0, 2] - m[:, 2, 0],
                    m[:, 1, 0] - m[:, 0, 1],
                    1 + trace,
                ],
                -1,
            ),
        ],
        dim=1,
    )
    choice = torch.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], trace], dim=-1).argmax(dim=-1)
    quaternion = candidates[torch.arange(m.shape[0], device=m.device), choice]
    return quaternion / quaternion.norm(dim=-1, keepdim=True)


def quaternion_to_rotation(quaternions: Quaternions) -> torch.Tensor:
    """SciPy's `Rotation.from_quat(q).as_matrix()` for xyzw quaternions."""

    x, y, z, w = (quaternions / quaternions.norm(dim=-1, keepdim=True)).unbind(-1)
    matrix = torch.stack(
        [
            x * x - y * y - z * z + w * w,
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            -x * x + y * y - z * z + w * w,
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            -x * x - y * y + z * z + w * w,
        ],
        dim=-1,
    )
    return rearrange(matrix, "v (r c) -> v r c", r=3)


def gaussian_filter(values: torch.Tensor, sigma: float) -> torch.Tensor:
    """`scipy.ndimage.gaussian_filter1d` along the first axis of `(V, C)`, mode "reflect"."""

    radius = int(GAUSSIAN_TRUNCATE * sigma + 0.5)
    offsets = torch.arange(-radius, radius + 1, device=values.device)
    kernel = torch.exp(-0.5 / (sigma * sigma) * offsets.double() ** 2)
    kernel = kernel / kernel.sum()

    # Half-sample symmetric extension, periodic in 2V so it also covers V < radius
    views = values.shape[0]
    index = (torch.arange(views, device=values.device)[:, None] + offsets[None]) % (2 * views)
    index = torch.where(index >= views, 2 * views - 1 - index, index)
    return (values[index] * kernel[None, :, None]).sum(dim=1)


def smooth_trajectory(camera_to_world: Poses, sigma: float) -> Poses:
    """Gaussian-smooth camera-to-world poses over time, per quaternion / translation component."""

    quaternions = gaussian_filter(matrix_to_quaternion(camera_to_world[:, :3, :3]), sigma)
    translations = gaussian_filter(camera_to_world[:, :3, 3], sigma)

    smoothed = torch.eye(4, dtype=camera_to_world.dtype, device=camera_to_world.device).repeat(
        camera_to_world.shape[0], 1, 1
    )
    smoothed[:, :3, :3] = quaternion_to_rotation(
        quaternions / quaternions.norm(dim=-1, keepdim=True)
    )
    smoothed[:, :3, 3] = translations
    return smoothed
