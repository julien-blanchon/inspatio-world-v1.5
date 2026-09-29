# gradio_worldviewer

`WorldViewer` is a Gradio 6 custom component for real-time, camera-controllable video world
models such as InSpatio-World. The backend streams the generated video in chunks, typically one
12-frame block per chunk. The frontend:

- plays the frames through a jitter buffer while the user steers the camera with the keyboard;
- shows what the model is conditioned on;
- shows where the camera is in the scene (a three.js view);
- shows how far the picture on screen lags behind the keys the user pressed.

Built and tested with **gradio 6.28.0** (`gradio>=6.0,<7.0`). Version 0.3.1.

## Layout

- **Main stage:** the generated video, with a minimal HUD (status, playback fps, generation fps,
  buffered frames, the scheduled delay and the measured lag). The stage and every layer on it are
  clipped to the block's rounded frame (`var(--block-radius)`), including blurred and transformed
  layers. Start and Stop sit in a toolbar underneath, along with key hints that
  light up while a key is held.
- **What the model sees:** `chunk.renders` (the point cloud splatted into the target camera),
  synchronized with the frame that is playing. The panel can be hidden.
- **Camera** (Map / 3D toggle, Map by default):
  - **Map:** a top-down 2D view of the source camera's ground plane, with "ahead" (the source
    camera's forward direction) pointing up. It shows:
    - the point cloud as faint, desaturated context;
    - a **start** marker;
    - the path as a thick trace, with a bigger dot at each 12-frame block (played: solid orange;
      upcoming/buffered: dashed and faded, ending in a ring where the camera is heading);
    - one bold FOV wedge for the current camera and its heading;
    - a readout of heading, pitch and height relative to the source camera, since a top-down view
      hides vertical motion.

    The view auto-fits the source camera, the region ahead of it and the whole path, and follows
    smoothly with time-based easing. Drag pans and the wheel zooms on top of the auto-follow;
    "Reset view" clears that. The map is Canvas 2D: no three.js, cached cloud layer, redraws capped
    at about 30 Hz.
  - **3D:** a threlte (three.js) orbit view. It is code-split and fetched only the first time 3D is
    selected. It shows a faint point cloud, the source frustum, the path as a thick line with
    per-frame dots (bigger per block; played solid, buffered faded), and a single bright frustum
    for the current camera.

  Neither view listens to keys.
- **Timeline:** a single lane with a fixed window (`timeline_seconds`, 10 s by default) and two
  cursors.
  - The **You** cursor sits at about 80 % of the width and marks "now". Key presses grow there as
    bars and scroll left.
  - The **Model** cursor sits to its left by the delay. In fixed-latency mode this is the scheduled
    delay (for example "3.5 s"), which grows only after a stall. In legacy mode it is the measured
    lag (smoothed), or a 2 s estimate labelled as such before the first measurement. An input bar
    reaching the Model cursor is when its effect appears on screen.
  - Bars are drawn full strength left of the Model cursor and dimmer between the cursors (in
    flight).
  - A slim strip under the bars shows each frame's `chunk.actions`, with ticks at block boundaries.
    Played frames are solid and buffered frames dimmed. In fixed-latency mode, frame `f` is placed
    at the moment its action was sampled (`t0 + f/fps` on the input axis). It therefore lines up
    with the key press that caused it and crosses the Model cursor exactly when it is displayed.
  - With no keyboard input in view (presets or scripted cameras), the frames' own actions are also
    drawn in the bar rows, so upcoming scripted moves are visible.
  - `None` actions (autopilot) are drawn as a hatched band.
  - The timeline scrolls only while the status is `running`. In every other state it freezes at its
    last view, or shows an empty static axis before the first run. Pressing Start clears it.

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
        frame_start=int | None,          # fixed-latency playback: session frame index of frames[0]
        elapsed=float | None,            # fixed-latency playback: seconds since the server's session clock t0 at yield
        latency=float | None,            # fixed-latency playback: frame f is shown at t0 + latency + f/fps
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

**Fixed-latency playback** (when chunks carry `frame_start` and `elapsed`):

- **The clock.** On the first such chunk of a session, the client sets `base = now - elapsed` as its
  estimate of `t0`. Frame `f` is then shown at `base + latency + f/fps` from a rAF loop. `latency`
  comes from the chunk, else from `WorldViewer(latency=3.5)`.
- **Early and late frames.** Frames that arrive early wait. If a frame is missing at its due time,
  the last frame is held and a "buffering" chip appears. When the frame arrives, the schedule
  shifts forward by the stall, so the delay grows only in that case. Frames that were on time but
  missed because rAF paused (for example a background tab) are dropped to catch up.
- **Start and end.** Before the first frame, a "first frame in X s" chip counts down. After
  `status="ended"`, the frames already buffered keep playing, and the ended overlay appears once
  they run out. Stop clears the buffer immediately and ignores chunks still in flight.
- **Server side.** Frame `f`'s camera should be driven by the keys held at `t0 + f/fps`. Sample
  the controls when that time arrives, as `demo/app.py` does.
- **Legacy mode.** Chunks without these fields use the older jitter buffer (`prebuffer_frames`,
  `catchup_rate`).

**Measured lag:** the frontend records `sent_at[seq]` when it dispatches each `control` event. When a
frame with `control_seqs[i] == S` is displayed, `lag = now - sent_at[S]`. The HUD and timeline
show an exponential moving average (α = 0.15). The timeline and camera path need `actions` and
`poses` respectively; latency needs `control_seqs`.

## Constructor

```python
WorldViewer(
    value=None, *, label=None, show_label=None, container=True, scale=None, min_width=160,
    visible=True, elem_id=None, elem_classes=None, render=True, key=None,
    show_render=True,        # "What the model sees" panel initially expanded
    show_camera=True,        # "Camera" panel (Map / 3D) initially expanded
    show_timeline=True,      # show the timeline
    latency=3.5,             # fixed-latency playback: default delay (s); chunk.latency overrides it
    prebuffer_frames=8,      # legacy jitter buffer only (chunks without frame_start/elapsed)
    catchup_rate=1.15,       # legacy jitter buffer only: speed-up when > 2 chunks are buffered
    heartbeat_ms=250,        # control heartbeat while running/loading
    max_control_hz=20,       # throttle for control events on action changes
    timeline_seconds=10.0,   # visible timeline window
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
    t0 = time.time()                                   # session clock
    for i in range(n):
        frame_start = i * 12
        time.sleep(max(0, t0 + frame_start / fps - time.time()))  # sample keys at t0 + f/fps
        ctrl = CONTROLS.get(request.session_hash, {})
        frames, renders, poses = model.step(ctrl)
        yield WorldViewerData(status="running", chunk=Chunk(
            id=i, session=sid, fps=fps, frames=encode_frames(frames), renders=encode_frames(renders, 70),
            actions=[action] * len(frames), control_seqs=[ctrl.get("seq")] * len(frames), poses=poses,
            frame_start=frame_start, elapsed=time.time() - t0, latency=3.5))
    yield WorldViewerData(status="ended", message="Session ended")  # status-only final yield

with gr.Blocks() as demo:
    viewer = WorldViewer()
    start_evt = viewer.start(run, None, viewer, show_progress="hidden")
    viewer.stop(on_stop, None, viewer, cancels=[start_evt], queue=False, show_progress="hidden")
    viewer.control(on_control, None, None, queue=False, show_progress="hidden", trigger_mode="multiple")
```

`demo/app.py` is a complete fake model: a synthetic point-cloud room splatted into a controllable
camera. It has a "Keyboard" mode (the first two blocks are `None` autopilot) and a "Scripted" mode
(a preset action per block). It samples the controls at `t0 + f/fps` and emits `frame_start`,
`elapsed` and `latency=3.5`.

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
- The Map view is plain Canvas 2D and part of the main bundle. The 3D panel is built with [threlte](https://threlte.xyz) 8 (`@threlte/core` plus `OrbitControls` from
  `@threlte/extras`) on three.js 0.186. It renders on demand and is code-split: `CameraView.svelte`
  is loaded with a dynamic `import()` only when "3D" is first selected, so that chunk (about 997 KB minified, about 227 KB gzipped)
  does not block the viewer. The main component code (including the map) is about 140 KB (about 42 KB gzipped).
- Build note: `frontend/tsconfig.json` sets `"target": "ESNext"`. The gradio build runs
  svelte-preprocess TypeScript over dependency `.svelte` files too. With a lower target, object rest
  in threlte's `$props()` destructuring is down-leveled to `__rest(...)`, and the Svelte compiler
  then rejects `$bindable()`.
- ZeroGPU: `@spaces.GPU` code runs in a separate worker process. Share the latest control action
  through something that crosses process boundaries, for example files keyed by `session_hash`.
