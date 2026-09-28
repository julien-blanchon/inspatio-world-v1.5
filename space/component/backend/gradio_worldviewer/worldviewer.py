"""WorldViewer: a keyboard/touch-driven, chunk-streaming video viewer for world models."""

from __future__ import annotations

import base64
import io
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

from gradio.components.base import Component
from gradio.data_classes import GradioModel
from gradio.events import Events

if TYPE_CHECKING:
    from gradio.components import Timer

Status = Literal["idle", "loading", "running", "ended", "error"]


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
    renders: Union[list[str], None] = None
    """Optional same-length list of small data-URI previews (shown picture-in-picture)."""


class WorldViewerData(GradioModel):
    status: Status = "idle"
    message: str = ""
    chunk: Union[Chunk, None] = None
    stats: dict[str, Union[float, int, str]] = {}


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

    The value is a `WorldViewerData` (status, message, optional `Chunk` of new frames, stats).
    Bind a generator to `.start()` that yields one `WorldViewerData` per generated chunk; read
    the user's camera action from `.control()` events (`evt._data`), and bind `.stop()` with
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
        show_render: bool = False,
        show_pad: bool | None = None,
        prebuffer_frames: int = 6,
        catchup_rate: float = 1.15,
        heartbeat_ms: int = 250,
        max_control_hz: float = 20.0,
        placeholder: str | None = None,
        aspect_ratio: float = 832 / 480,
    ):
        """
        Parameters:
            value: initial WorldViewerData / dict (usually None -> idle).
            show_render: whether the picture-in-picture of `chunk.renders` is initially visible.
            show_pad: show the on-screen control pad. None = auto (shown on touch devices, collapsible on desktop).
            prebuffer_frames: frames to buffer before playback starts (and after a stall).
            catchup_rate: playback speed multiplier used when more than ~2 chunks are buffered.
            heartbeat_ms: interval of the `control` heartbeat while running/loading.
            max_control_hz: maximum rate of `control` events on action changes.
            placeholder: idle-state hint text.
            aspect_ratio: canvas aspect ratio before the first frame arrives.
        """
        self.show_render = show_render
        self.show_pad = show_pad
        self.prebuffer_frames = prebuffer_frames
        self.catchup_rate = catchup_rate
        self.heartbeat_ms = heartbeat_ms
        self.max_control_hz = max_control_hz
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
            ),
            stats={"frames": 1},
        )
