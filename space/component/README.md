# gradio_worldviewer

`WorldViewer` is a Gradio 6 custom component for real-time, camera-controllable video world
models such as InSpatio-World. The backend streams the generated video in chunks, typically one
12-frame block per chunk. The frontend:

- plays the frames through a jitter buffer while the user steers the camera with the keyboard;
- shows what the model is conditioned on;
- shows where the camera is in the scene (a three.js view);
- shows how far the picture on screen lags behind the keys the user pressed.

Built and tested with **gradio 6.28.0** (`gradio>=6.0,<7.0`). Version 0.2.0.

## Layout

- **Main stage:** the generated video, with a minimal HUD (status, playback fps, generation fps,
  buffered frames, latency). Start and Stop sit in a toolbar underneath, along with key hints that
  light up while a key is held.
- **What the model sees:** `chunk.renders` (the point cloud splatted into the target camera),
  synchronized with the frame that is playing. The panel can be hidden.
- **Camera:** a threlte (three.js) view of `scene.points`, the source camera frustum(s) in grey, and the
  generated camera path from `chunk.poses`:
  - frames already played are drawn solid;
  - frames generated but still buffered are drawn faint;
  - the current frustum is highlighted.

  Drag to orbit and scroll to zoom. "Reset view" re-frames the scene. The view never listens to
  keys.
- **Timeline:** the last `timeline_seconds` (8 s by default), with "now" at the right edge.
  - The **You** lane shows your key presses as they happen, with a tick for each `control` event
    sent.
  - The **Played** lane shows `chunk.actions` for each frame at the moment it is displayed.
    Autopilot frames are hatched, and block boundaries are dashed.
  - A bracket between the two lanes shows the current latency.

On narrow screens (container width of 760 px or less) the panels stack vertically.

## Data model

```python
from gradio_worldviewer import WorldViewer, WorldViewerData, Chunk, SceneInfo, encode_frames

WorldViewerData(
    status="idle" | "loading" | "running" | "ended" | "error",
    message="",
    stats={"block_ms": 612.3, "gen_fps": 19.6, "frames": 240, "elapsed_s": 16.2, "limit_s": 90},
    scene=SceneInfo(...) | None,        # optional; see below
    chunk=Chunk(
        id=0,                            # increases within a session; 0 = new session
        session="abc123",                # a new id resets the player, camera path and latency
        fps=16.0,
        frames=[...],                    # data URIs (JPEG)
        renders=[...] | None,            # same length: small JPEG data URIs of the render condition
        actions=[{...} | None] | None,   # same length: {"forward","right","up","yaw","pitch"} floats in [-1,1]; None entry = autopilot
        control_seqs=[int | None] | None,  # same length: `seq` of the latest control event the backend had when generating the frame
        poses=[[16 floats]] | None,      # same length: camera-to-world 4x4 row-major, OpenCV axes (x right, y down, z forward)
    ),
)

SceneInfo(
    points="<base64 little-endian float32 N×3 xyz>",   # world frame, OpenCV axes; N ≈ 20–40k
    colors="<base64 uint8 N×3 RGB>",
    source_poses=[[16 floats], ...],   # 1 for an image, 4 for multi-view, per-frame cameras for a video
    intrinsics=[fx, fy, cx, cy, width, height],
    kind="image" | "video",
    title="",
)
SceneInfo.from_arrays(points, colors, source_poses=(), intrinsics=None, kind="image", title="",
                      max_points=30000)  # numpy helper: drops non-finite points, subsamples, encodes
```

All new fields are optional, so v0.1 payloads still work. `Chunk.poses` and
`SceneInfo.source_poses` also accept numpy arrays shaped `(N,4,4)` or `(N,16)`, or nested 4×4
lists, and flatten them. `postprocess` accepts a `WorldViewerData`, a dict of the same shape, or
`None`.

**Scene caching:** the frontend keeps the last scene it received. It rebuilds the point cloud only
when the scene content changes (title, size and a sample of the base64 string). You can therefore
send the scene with every yield. Gradio's generator diffing sends unchanged strings as nothing, so
repeats cost nothing except in the final full value re-send. A value with `scene=None` keeps the
cached scene.

**Latency:** the frontend records `sent_at[seq]` when it dispatches each `control` event. When a
frame with `control_seqs[i] == S` is displayed, `lag = now - sent_at[S]`. The HUD and timeline
show an exponential moving average (α = 0.15). The timeline and camera path need `actions` and
`poses` respectively; latency needs `control_seqs`.

## Constructor

```python
WorldViewer(
    value=None, *, label=None, show_label=None, container=True, scale=None, min_width=160,
    visible=True, elem_id=None, elem_classes=None, render=True, key=None,
    show_render=True,        # "What the model sees" panel initially expanded
    show_camera=True,        # 3D "Camera" panel initially expanded
    show_timeline=True,      # show the timeline
    prebuffer_frames=8,      # frames buffered before playback (re)starts
    catchup_rate=1.15,       # playback speed when > 2 chunks are buffered (squared when > 4 chunks)
    heartbeat_ms=250,        # control heartbeat while running/loading
    max_control_hz=20,       # throttle for control events on action changes
    timeline_seconds=8.0,    # visible timeline window
    placeholder=None,        # idle hint text
    aspect_ratio=832/480,    # canvas aspect before the first frame
)
```

The on-screen touch pad and `show_pad` from v0.1 were removed.

## Events

| event | fired by | payload (`evt._data`) |
| --- | --- | --- |
| `start` | Start button, big play button, Enter | `{}` |
| `stop` | Stop button, Escape | `{}` |
| `control` | action change (throttled to `max_control_hz`), plus a heartbeat every `heartbeat_ms` while running/loading | `{"forward","right","up","yaw","pitch": float, "seq": int, "session": str}` |
| `change` | any value change | none |

Keys only work while the main stage has focus (click it; Start also focuses it). Clicking or
dragging in the 3D view moves focus away, which releases every held key. The mapped keys do not
scroll the page.

| keys | axis |
| --- | --- |
| W / S, ArrowUp / ArrowDown | forward +1 / -1 |
| A / D | right -1 / +1 |
| Q / E, ArrowLeft / ArrowRight | yaw -1 / +1 |
| R / F | pitch +1 / -1 |
| Space / Shift | up +1 / -1 |

Keys are matched with `KeyboardEvent.code`, i.e. by physical position.

## Wiring

```python
def on_control(evt: gr.EventData, request: gr.Request):
    CONTROLS[request.session_hash] = evt._data     # includes "seq": echo it back in chunk.control_seqs

def run(request: gr.Request):
    yield WorldViewerData(status="loading", message="Compiling…", scene=scene_info)
    for i in range(n):
        ctrl = CONTROLS.get(request.session_hash, {})
        frames, renders, poses = model.step(ctrl)
        yield WorldViewerData(status="running", chunk=Chunk(
            id=i, session=sid, fps=16, frames=encode_frames(frames), renders=encode_frames(renders, 70),
            actions=[action] * len(frames), control_seqs=[ctrl.get("seq")] * len(frames), poses=poses))
    yield WorldViewerData(status="ended", message="Session ended")  # status-only final yield

with gr.Blocks() as demo:
    viewer = WorldViewer()
    start_evt = viewer.start(run, None, viewer, show_progress="hidden")
    viewer.stop(on_stop, None, viewer, cancels=[start_evt], queue=False, show_progress="hidden")
    viewer.control(on_control, None, None, queue=False, show_progress="hidden", trigger_mode="multiple")
```

`demo/app.py` is a complete fake model: a synthetic point-cloud room splatted into a controllable
camera, with autopilot for the first three blocks.

## Gotchas

- Gradio sends each generator yield as a diff against the previous one, then re-sends the last full
  value when the generator finishes. The viewer ignores chunk ids it has already seen. Still, make
  the final yield status-only so the last chunk's frames are not sent twice.
- Use a new `session` id for every Start.
- Cancelling the generator via Stop sends no final value, so `on_stop` should return an `ended`
  value. A chunk already in flight may still arrive and play.
- Payload size: 12 frames of 832×480 JPEG are about 0.6–1.5 MB of base64 per chunk. Renders at
  208×120 are about 4 KB each. A 30k-point scene is about 640 KB of base64, sent once per session
  (plus the final re-send).
- The 3D panel is built with [threlte](https://threlte.xyz) 8 (`@threlte/core` plus `OrbitControls` from
  `@threlte/extras`) on three.js 0.186. It renders on demand and is code-split: `CameraView.svelte`
  is loaded with a dynamic `import()`, so that chunk (about 978 KB minified, about 221 KB gzipped)
  does not block the viewer. The main component code is about 124 KB (about 38 KB gzipped).
- Build note: `frontend/tsconfig.json` sets `"target": "ESNext"`. The gradio build runs
  svelte-preprocess TypeScript over dependency `.svelte` files too. With a lower target, object rest
  in threlte's `$props()` destructuring is down-leveled to `__rest(...)`, and the Svelte compiler
  then rejects `$bindable()`.
- ZeroGPU: `@spaces.GPU` code runs in a separate worker process. Share the latest control action
  through something that crosses process boundaries, for example files keyed by `session_hash`.
