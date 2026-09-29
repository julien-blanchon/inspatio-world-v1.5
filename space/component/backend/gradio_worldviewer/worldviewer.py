"""WorldViewer: a keyboard-driven, chunk-streaming video viewer for camera-controllable world models."""

from __future__ import annotations

import base64
import io
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from gradio.components.base import Component
from gradio.data_classes import GradioModel
from gradio.events import Events
from pydantic import field_validator

if TYPE_CHECKING:
    from gradio.components import Timer

Status = Literal["idle", "loading", "running", "ended", "error"]


def _flat_poses(value: Any) -> Any:
    """Accept (N,4,4)/(N,16) arrays or nested lists and return a list of 16-float lists."""
    if value is None:
        return None
    if hasattr(value, "tolist") and hasattr(value, "reshape"):  # numpy / torch (cpu)
        return value.reshape(len(value), -1).tolist()
    out = []
    for p in value:
        if hasattr(p, "tolist") and hasattr(p, "reshape"):
            out.append(p.reshape(-1).tolist())
        elif p and isinstance(p[0], (list, tuple)):
            out.append([float(x) for row in p for x in row])
        else:
            out.append([float(x) for x in p])
    return out


class Chunk(GradioModel):
    """A batch of new frames to append to the viewer's playback queue."""

    id: int
    """Monotonically increasing within a session; 0 = first chunk of a new session (resets the queue)."""
    session: str
    """Session id; a new id also resets the player."""
    fps: float = 15.0
    """Playback rate of these frames."""
    frames: list[str]
    """Data URIs, e.g. "data:image/jpeg;base64,..."."""
    renders: Optional[list[str]] = None
    """Optional same-length list of small data-URI previews of the render condition."""
    actions: Optional[list[Optional[dict[str, float]]]] = None
    """Optional per-frame camera action {forward,right,up,yaw,pitch} used to generate the frame (None = autopilot)."""
    control_seqs: Optional[list[Optional[int]]] = None
    """Optional per-frame `seq` of the latest control event the backend had received (latency measurement)."""
    poses: Optional[list[list[float]]] = None
    """Optional per-frame camera-to-world 4x4 matrices, row-major 16 floats, OpenCV axes."""

    @field_validator("poses", mode="before")
    @classmethod
    def validate_poses(cls, v: Any) -> Any:
        return _flat_poses(v)


class SceneInfo(GradioModel):
    """Static scene description shown in the 3D camera panel (send with the first chunk of a session)."""

    points: str
    """base64 of little-endian float32 N x 3 xyz (world, OpenCV axes)."""
    colors: str
    """base64 of uint8 N x 3 RGB."""
    source_poses: list[list[float]] = []
    """Camera-to-world 4x4 (row-major 16 floats) of the source view(s)."""
    intrinsics: Optional[list[float]] = None
    """[fx, fy, cx, cy, width, height] used to draw frustums."""
    kind: Literal["image", "video"] = "image"
    title: str = ""

    @field_validator("source_poses", mode="before")
    @classmethod
    def validate_source_poses(cls, v: Any) -> Any:
        return _flat_poses(v) or []

    @classmethod
    def from_arrays(
        cls,
        points: Any,
        colors: Any,
        source_poses: Any = (),
        intrinsics: Sequence[float] | None = None,
        kind: Literal["image", "video"] = "image",
        title: str = "",
        max_points: int = 30000,
        seed: int = 0,
    ) -> SceneInfo:
        """Build from numpy arrays: points (N,3) float, colors (N,3) uint8 or float in [0,1].

        Randomly subsamples to `max_points` and drops non-finite points.
        """
        import numpy as np

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        col = np.asarray(colors).reshape(-1, 3)
        if col.dtype != np.uint8:
            col = np.clip(col * (255.0 if col.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        keep = np.isfinite(pts).all(1)
        pts, col = pts[keep], col[keep]
        if len(pts) > max_points:
            idx = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
            pts, col = pts[idx], col[idx]
        return cls(
            points=base64.b64encode(pts.astype("<f4").tobytes()).decode("ascii"),
            colors=base64.b64encode(np.ascontiguousarray(col).tobytes()).decode("ascii"),
            source_poses=_flat_poses(np.asarray(source_poses, dtype=np.float64).reshape(-1, 16))
            if len(source_poses)
            else [],
            intrinsics=[float(x) for x in intrinsics] if intrinsics is not None else None,
            kind=kind,
            title=title,
        )


class WorldViewerData(GradioModel):
    status: Status = "idle"
    message: str = ""
    chunk: Optional[Chunk] = None
    stats: dict[str, Union[float, int, str]] = {}
    scene: Optional[SceneInfo] = None


def encode_frame(frame: Any, quality: int = 85, fmt: str = "JPEG") -> str:
    """Encode one frame (HxWx3 uint8 numpy array, PIL image, or raw JPEG/PNG bytes) as a data URI."""
    if isinstance(frame, (bytes, bytearray)):
        raw = bytes(frame)
        mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
    from PIL import Image

    img = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else f"image/{fmt.lower()}"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def encode_frames(frames: Iterable[Any], quality: int = 85, fmt: str = "JPEG") -> list[str]:
    """Encode a sequence of frames (e.g. a TxHxWx3 uint8 array) as a list of data URIs."""
    return [encode_frame(f, quality=quality, fmt=fmt) for f in frames]


class WorldViewer(Component):
    """
    Interactive real-time viewer for camera-controllable video world models.

    The value is a `WorldViewerData` (status, message, optional `Chunk` of new frames, stats, optional
    `SceneInfo`). Bind a generator to `.start()` that yields one `WorldViewerData` per generated chunk;
    read the user's camera action from `.control()` events (`evt._data`), and bind `.stop()` with
    `cancels=[start_event]`.
    """

    EVENTS = [Events.change, "start", "stop", "control"]
    data_model = WorldViewerData

    def __init__(
        self,
        value: WorldViewerData | dict | Callable | None = None,
        *,
        label: str | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        visible: bool | Literal["hidden"] = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | tuple[int | str, ...] | None = None,
        preserved_by_key: list[str] | str | None = "value",
        every: Timer | float | None = None,
        inputs: Component | Sequence[Component] | set[Component] | None = None,
        show_render: bool = True,
        show_camera: bool = True,
        show_timeline: bool = True,
        prebuffer_frames: int = 8,
        catchup_rate: float = 1.15,
        heartbeat_ms: int = 250,
        max_control_hz: float = 20.0,
        timeline_seconds: float = 8.0,
        placeholder: str | None = None,
        aspect_ratio: float = 832 / 480,
    ):
        """
        Parameters:
            value: initial WorldViewerData / dict (usually None -> idle).
            show_render: whether the "What the model sees" panel (chunk.renders) is initially expanded.
            show_camera: whether the 3D "Camera" panel is initially expanded.
            show_timeline: whether the input/playback timeline is shown.
            prebuffer_frames: frames to buffer before playback starts (and after a stall).
            catchup_rate: playback speed multiplier used when more than ~2 chunks are buffered.
            heartbeat_ms: interval of the `control` heartbeat while running/loading.
            max_control_hz: maximum rate of `control` events on action changes.
            timeline_seconds: visible window of the timeline.
            placeholder: idle-state hint text.
            aspect_ratio: canvas aspect ratio before the first frame arrives.
        """
        self.show_render = show_render
        self.show_camera = show_camera
        self.show_timeline = show_timeline
        self.prebuffer_frames = prebuffer_frames
        self.catchup_rate = catchup_rate
        self.heartbeat_ms = heartbeat_ms
        self.max_control_hz = max_control_hz
        self.timeline_seconds = timeline_seconds
        self.placeholder = placeholder
        self.aspect_ratio = aspect_ratio
        super().__init__(
            value=value,
            label=label,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            key=key,
            preserved_by_key=preserved_by_key,
            every=every,
            inputs=inputs,
        )

    def preprocess(self, payload: WorldViewerData | None) -> dict | None:
        """
        Parameters:
            payload: the current viewer value.
        Returns:
            the value as a plain dict (or None).
        """
        if payload is None:
            return None
        return payload.model_dump()

    def postprocess(self, value: WorldViewerData | dict | None) -> WorldViewerData | None:
        """
        Parameters:
            value: a WorldViewerData, a dict of the same shape, or None.
        Returns:
            the validated WorldViewerData sent to the frontend.
        """
        if value is None:
            return None
        if isinstance(value, WorldViewerData):
            return value
        if isinstance(value, dict):
            return WorldViewerData(**value)
        raise TypeError(f"WorldViewer value must be WorldViewerData, dict or None, got {type(value)!r}")

    def example_payload(self) -> Any:
        return WorldViewerData(status="idle", message="").model_dump()

    def example_value(self) -> Any:
        return WorldViewerData(
            status="running",
            message="",
            chunk=Chunk(
                id=0,
                session="example",
                fps=15.0,
                frames=[
                    # 1x1 grey JPEG
                    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACP/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
                ],
                actions=[{"forward": 1.0, "right": 0.0, "up": 0.0, "yaw": 0.0, "pitch": 0.0}],
                control_seqs=[1],
                poses=[[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
            ),
            stats={"frames": 1},
        )
