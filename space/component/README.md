# gradio_worldviewer

`WorldViewer` is a Gradio 6 custom component for real-time, camera-controllable video world
models (built for InSpatio-World). The backend streams generated video in chunks of frames.
The frontend plays them through a jitter buffer on a `<canvas>` while the user steers the camera
with the keyboard (WASD and the arrow keys) or an on-screen touch pad.

Built and tested with **gradio 6.28.0** (`gradio>=6.0,<7.0`).

## Install

```bash
pip install wheels/gradio_worldviewer-0.1.0-py3-none-any.whl
```

## Value / data model

```python
from gradio_worldviewer import WorldViewer, WorldViewerData, Chunk, encode_frames

WorldViewerData(
    status="idle" | "loading" | "running" | "ended" | "error",
    message="",                # HUD / overlay text
    chunk=Chunk(               # or None (status-only update)
        id=0,                  # increases within a session; 0 = new session (resets the player)
        session="abc123",      # a new session id also resets the player
        fps=15.0,              # playback rate of these frames
        frames=[...],          # data URIs: "data:image/jpeg;base64,..."
        renders=None,          # optional same-length previews, shown as picture-in-picture
    ),
    stats={"block_ms": 612.3, "gen_fps": 19.6, "frames": 240, "elapsed_s": 16.2, "limit_s": 90},
)
```

`postprocess` accepts a `WorldViewerData`, a plain dict with the same shape, or `None`.
`preprocess` returns a plain dict. `encode_frames(frames, quality=85)` turns a `T×H×W×3`
uint8 array, a list of PIL images, or a list of JPEG bytes into data URIs. `encode_frame` does
the same for a single frame.

The HUD shows `stats` keys generically. Keys ending in `_ms` or `_s` are formatted with that unit.
`elapsed_s` and `limit_s` drive the session progress bar, and `elapsed_s` is interpolated
client-side between chunks.

## Constructor

```python
WorldViewer(
    value=None, *, label=None, show_label=None, container=True, scale=None, min_width=160,
    visible=True, elem_id=None, elem_classes=None, render=True, key=None,
    show_render=False,      # PiP of chunk.renders initially visible (toggle button in the viewer)
    show_pad=None,          # on-screen pad: None = auto (open on touch devices), True/False to force
    prebuffer_frames=6,     # frames buffered before playback (re)starts
    catchup_rate=1.15,      # playback speed when > 2 chunks are buffered (squared when > 4 chunks)
    heartbeat_ms=250,       # control heartbeat while running/loading
    max_control_hz=20,      # throttle for control events on action changes
    placeholder=None,       # idle hint text
    aspect_ratio=832/480,   # canvas aspect before the first frame (then taken from the frames)
)
```

## Events

| event | fired by | payload (`evt._data`) |
| --- | --- | --- |
| `start` | Start button, big play button, Enter key | `{}` |
| `stop` | Stop button, Escape key | `{}` |
| `control` | action change (throttled to `max_control_hz`), plus a heartbeat every `heartbeat_ms` while running/loading | `{"forward","right","up","yaw","pitch": float in [-1,1], "seq": int, "session": str}` |
| `change` | any value change | none |

Key bindings apply only while the viewer is focused (click it; Start also focuses it). The mapped
keys do not scroll the page, and every held key is released on blur or when the tab is hidden.

| keys | axis |
| --- | --- |
| W / S, ArrowUp / ArrowDown | forward +1 / -1 |
| A / D | right -1 / +1 |
| Q / E, ArrowLeft / ArrowRight | yaw -1 / +1 |
| R / F | pitch +1 / -1 |
| Space / Shift | up +1 / -1 |

Keys are matched with `KeyboardEvent.code`, so they follow physical key positions: on AZERTY,
the physical WASD keys are ZQSD.

## Wiring (verified with gradio 6.28.0)

```python
import gradio as gr
from gradio_worldviewer import WorldViewer, WorldViewerData, Chunk, encode_frames

CONTROLS = {}   # latest action per browser session

def on_control(evt: gr.EventData, request: gr.Request):
    CONTROLS[request.session_hash] = evt._data     # dict: forward/right/up/yaw/pitch/seq/session
    # evt.forward etc. also work (EventData.__getattr__ reads _data)

def run(request: gr.Request):
    sid = uuid.uuid4().hex[:8]
    yield WorldViewerData(status="loading", message="Compiling…")
    for i in range(n_chunks):
        action = CONTROLS.get(request.session_hash, {})
        frames = model.step(action)                 # T×H×W×3 uint8
        yield WorldViewerData(status="running",
                              chunk=Chunk(id=i, session=sid, fps=16, frames=encode_frames(frames)),
                              stats={"elapsed_s": ..., "limit_s": 90})
    yield WorldViewerData(status="ended", message="Session ended: time limit")   # chunk=None!

def on_stop():
    return WorldViewerData(status="ended", message="Stopped")

with gr.Blocks() as demo:
    viewer = WorldViewer()
    start_evt = viewer.start(run, inputs=None, outputs=viewer, show_progress="hidden")
    viewer.stop(on_stop, None, viewer, cancels=[start_evt], queue=False, show_progress="hidden")
    viewer.control(on_control, None, None, queue=False, show_progress="hidden", trigger_mode="multiple")
```

## Gotchas

- **Generator outputs are diffed.** Gradio sends each yield after the first as a JSON diff against
  the previous yield. Each frame string that changes is sent again in full as a `replace` edit, so
  there are no bandwidth savings for frames. When the generator finishes, Gradio re-sends the
  **last full value**. The viewer ignores chunk ids it has already seen, but you should still make
  the final yield a status-only value (`chunk=None`) so the last chunk's frames are not sent twice.
- **Use a fresh `session` id per Start.** The player resets on a new session id, or on `id == 0`
  once a higher id has been seen. If you reuse a session id after a single-chunk run, the new
  chunk 0 is treated as a duplicate.
- **Stop and `cancels`.** Cancelling the generator does not send a final value. The viewer shows
  "Stopped" right away when Stop is pressed, and `on_stop` should return an `ended` value so every
  client agrees. A chunk that was already in flight may still arrive and play.
- **`control` with `queue=False`.** Control events then skip the queue and are not throttled by
  the start listener's concurrency. With `queue=True`, set `concurrency_limit=None`; otherwise
  controls from different users are serialized behind the default limit of 1.
- **Loading status.** The viewer does not render Gradio's `StatusTracker` overlay. It shows a
  "Waiting in queue (position N)" message while a Start is pending, and an error overlay if the
  `start` event errors.
- **Payload size.** A 12-frame 832×480 JPEG chunk at q≈85 is about 0.6–1.5 MB base64 (natural
  images are roughly 50–120 KB per frame). At 16 fps this is about 1–2 MB/s per user over SSE.
  Lower the JPEG quality, or send `renders` at a small size (for example 208×120), to keep
  latency down.
- **ZeroGPU.** A `@spaces.GPU` function runs in a separate worker process. A plain dict that the
  `control` handler updates in the main process is **not visible** inside that worker. Read the
  latest action in the main-process generator and pass it to a per-chunk GPU call, or share it
  across processes (for example with a `multiprocessing` Manager, or a small file keyed by
  `session_hash`).
