<script lang="ts">
	import { Gradio } from "@gradio/utils";
	import { Block } from "@gradio/atoms";
	import { onDestroy, onMount, untrack } from "svelte";
	import Timeline from "./Timeline.svelte";
	import MapView from "./MapView.svelte";
	import type {
		Action,
		Axis,
		CamStore,
		Chunk,
		ControlPayload,
		Frame,
		Seg,
		Status,
		TimelineStore,
		VKey,
		WorldViewerEvents,
		WorldViewerProps,
		WorldViewerValue
	} from "./types";

	const props = $props();
	const gradio = new Gradio<WorldViewerEvents, WorldViewerProps>(props);
	gradio.watch_for_change();

	const AXES: Axis[] = ["forward", "right", "yaw", "pitch", "up"];

	// ---------------------------------------------------------------- config
	const prebuffer = $derived(Math.max(1, gradio.props.prebuffer_frames ?? 8));
	const catchup = $derived(gradio.props.catchup_rate ?? 1.15);
	const heartbeat_ms = $derived(gradio.props.heartbeat_ms ?? 250);
	const min_ctrl_interval = $derived(1000 / Math.max(1, gradio.props.max_control_hz ?? 20));
	const timeline_seconds = $derived(gradio.props.timeline_seconds ?? 10);
	const default_latency_ms = $derived(1000 * (gradio.props.latency ?? 3.5));
	const show_timeline = $derived(gradio.props.show_timeline ?? true);
	const placeholder_text = $derived(
		gradio.props.placeholder ?? "Press Start, then click here and use WASD / arrows to move"
	);

	// ------------------------------------------------------------ UI state
	let root: HTMLDivElement | undefined = $state();
	let stage: HTMLDivElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();
	let cond: HTMLCanvasElement | undefined = $state();
	let cam_reset = $state(0);
	// Camera panel: 2D top-down map by default; the three.js/threlte 3D view is code-split and only
	// fetched the first time "3D" is selected
	let cam_mode: "map" | "3d" = $state("map");
	let camera_view: Promise<typeof import("./CameraView.svelte")> | null = $state(null);
	function set_cam_mode(m: "map" | "3d"): void {
		cam_mode = m;
		if (m === "3d" && !camera_view) camera_view = import("./CameraView.svelte");
	}

	let pending: "start" | "stop" | null = $state(null);
	let focused = $state(false);
	let is_fullscreen = $state(false);
	let cond_open = $state(untrack(() => gradio.props.show_render ?? true));
	let cam_open = $state(untrack(() => gradio.props.show_camera ?? true));
	let has_frame = $state(false);
	let has_renders = $state(false);
	let buffering = $state(false);
	let buffered = $state(0);
	let play_fps = $state(0);
	let latency = $state<number | null>(null); // measured key -> display lag (ms)
	let delay = $state<number | null>(null); // scheduled delay (s), scheduled mode only
	let first_in = $state<number | null>(null); // seconds until the first frame is due
	let aspect = $state(untrack(() => gradio.props.aspect_ratio || 832 / 480));
	let now = $state(performance.now());
	let stats_at = $state(performance.now());
	let scene_title = $state("");

	const value: WorldViewerValue | null = $derived(gradio.props.value ?? null);
	const server_status: Status = $derived(value?.status ?? "idle");
	const loading_status = $derived(gradio.shared.loading_status);
	const ls_error = $derived(
		loading_status?.status === "error" &&
			(pending === "start" || server_status === "loading" || server_status === "running")
	);
	const status: Status = $derived(
		ls_error
			? "error"
			: pending === "start"
				? "loading"
				: pending === "stop"
					? "ended"
					: server_status
	);
	const message = $derived(
		ls_error
			? loading_status?.message || value?.message || "Something went wrong"
			: pending === "start"
				? queue_text() || "Starting…"
				: pending === "stop"
					? "Stopped"
					: (value?.message ?? "")
	);
	const active = $derived(status === "running" || status === "loading");
	const can_start = $derived(!active);
	const can_stop = $derived(active);
	const stats = $derived(value?.stats ?? {});
	const gen_fps = $derived(typeof stats.gen_fps === "number" ? (stats.gen_fps as number) : null);
	const limit_s = $derived(typeof stats.limit_s === "number" ? (stats.limit_s as number) : null);
	const elapsed_s = $derived.by(() => {
		const base = typeof stats.elapsed_s === "number" ? (stats.elapsed_s as number) : 0;
		const extra = server_status === "running" ? (now - stats_at) / 1000 : 0;
		return limit_s ? Math.min(limit_s, base + extra) : base + extra;
	});

	function queue_text(): string {
		const ls = gradio.shared.loading_status;
		if (ls?.status === "pending" && ls.queue_position != null && ls.queue_position > 0)
			return `Waiting in queue (position ${ls.queue_position + 1}${ls.queue_size ? ` / ${ls.queue_size}` : ""})…`;
		return "";
	}

	// ----------------------------------------------------- shared stores
	const tl: TimelineStore = { you: [], frames: [], queue: [], latency_ms: null, session: 0, sched: null };
	const cam: CamStore = { scene: null, scene_version: 0, poses: [], played: 0, version: 0 };
	let scene_key = "";

	// ------------------------------------------------------- player state
	type QFrame = Frame & { pose_index: number };
	let queue: QFrame[] = [];
	let gen = 0;
	let decode_chain: Promise<void> = Promise.resolve();
	let cur_session: string | null = null;
	let seen = new Set<number>();
	let max_seen = -1;
	let chunk_len = 12;
	let playing = false;
	let last_t = 0;
	let acc = 0;
	let current: QFrame | null = null;
	let drawn_times: number[] = [];
	let raf = 0;
	let ui_last = 0;
	let latency_ema: number | null = null;
	const sent_at = new Map<number, number>();
	// fixed-latency schedule (set from the first chunk carrying `elapsed`): frame f is due at
	// base0 + lat_ms + shift + f * 1000 / sched_fps; `shift` only grows when a frame arrives late
	let base0: number | null = null;
	let lat_ms = 3500;
	let shift = 0;
	let sched_fps = 15;
	let last_index = -1;
	let drawn_this_session = false;
	let stopped_session: string | null = null;

	function close_frame(f: Frame | null): void {
		if (!f) return;
		if ("close" in f.bm) f.bm.close();
		if (f.rb && "close" in f.rb) f.rb.close();
	}

	function reset_player(session: string | null): void {
		gen++;
		for (const f of queue) close_frame(f);
		queue = [];
		tl.queue = queue;
		seen = new Set();
		max_seen = -1;
		cur_session = session;
		playing = false;
		acc = 0;
		drawn_times = [];
		buffered = 0;
		latency_ema = null;
		tl.latency_ms = null;
		latency = null;
		base0 = null;
		shift = 0;
		last_index = -1;
		drawn_this_session = false;
		tl.sched = null;
		cam.poses = [];
		cam.played = 0;
		cam.version++;
	}

	function data_uri_to_blob(uri: string): Blob {
		const comma = uri.indexOf(",");
		const mime = uri.slice(5, comma).split(";")[0] || "image/jpeg";
		const bin = atob(uri.slice(comma + 1));
		const bytes = new Uint8Array(bin.length);
		for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
		return new Blob([bytes], { type: mime });
	}

	async function decode(uri: string): Promise<ImageBitmap | HTMLImageElement> {
		if (typeof createImageBitmap === "function" && uri.startsWith("data:")) {
			return await createImageBitmap(data_uri_to_blob(uri));
		}
		const img = new Image();
		img.decoding = "async";
		img.src = uri;
		await img.decode();
		return img;
	}

	function norm_action(a: Partial<Action> | null | undefined): Action | null {
		if (!a) return null;
		return {
			forward: +(a.forward ?? 0),
			right: +(a.right ?? 0),
			up: +(a.up ?? 0),
			yaw: +(a.yaw ?? 0),
			pitch: +(a.pitch ?? 0)
		};
	}

	function enqueue_chunk(c: Chunk): void {
		const my_gen = gen;
		const fps = c.fps > 0 ? c.fps : 15;
		const start = typeof c.frame_start === "number" ? c.frame_start : null;
		const frames_p = Promise.all(c.frames.map(decode));
		const renders_p = c.renders?.length
			? Promise.all(c.renders.map((r) => decode(r).catch(() => null)))
			: Promise.resolve(null);
		decode_chain = decode_chain.then(async () => {
			let bms: (ImageBitmap | HTMLImageElement)[];
			let rbs: (ImageBitmap | HTMLImageElement | null)[] | null;
			try {
				[bms, rbs] = await Promise.all([frames_p, renders_p]);
			} catch (e) {
				console.warn("[WorldViewer] failed to decode chunk", c.id, e);
				return;
			}
			if (my_gen !== gen) {
				bms.forEach((bm) => close_frame({ bm, rb: null } as Frame));
				rbs?.forEach((rb) => rb && close_frame({ bm: rb, rb: null } as Frame));
				return;
			}
			const has_action = Array.isArray(c.actions);
			const frames: QFrame[] = bms.map((bm, i) => {
				const pose = c.poses?.[i];
				let pose_index = -1;
				if (Array.isArray(pose) && pose.length >= 12) {
					cam.poses.push(pose);
					pose_index = cam.poses.length - 1;
				}
				return {
					bm,
					rb: rbs?.[i] ?? null,
					fps,
					action: has_action ? norm_action(c.actions![i]) : null,
					has_action,
					seq: c.control_seqs?.[i] ?? null,
					pose: pose ?? null,
					chunk_id: c.id,
					first_in_chunk: i === 0,
					index: start != null ? start + i : null,
					arrived: performance.now(),
					pose_index
				};
			});
			if (c.poses?.length) cam.version++;
			if (rbs && rbs.some(Boolean)) has_renders = true;
			chunk_len = Math.max(1, frames.length);
			queue.push(...frames);
			buffered = queue.length;
		});
	}

	function set_scene(v: WorldViewerValue): void {
		const s = v.scene;
		if (!s || !s.points) return;
		const key = `${s.title}|${s.points.length}|${s.points.slice(0, 64)}|${s.points.slice(-64)}|${s.source_poses?.length}`;
		if (key === scene_key) return;
		scene_key = key;
		cam.scene = s;
		cam.scene_version++;
		scene_title = s.title || "";
	}

	function on_value(v: WorldViewerValue | null): void {
		if (!v) return;
		if (pending === "start") pending = null;
		else if (pending === "stop" && v.status !== "running" && v.status !== "loading") pending = null;
		stats_at = performance.now();
		set_scene(v);
		const c = v.chunk;
		if (!c || !Array.isArray(c.frames) || c.frames.length === 0) return;
		if (c.session === stopped_session) return; // chunks still in flight after Stop
		if (c.session !== cur_session) reset_player(c.session);
		else if (c.id === 0 && max_seen > 0) reset_player(c.session);
		else if (seen.has(c.id)) return; // repeated value update (e.g. final re-send)
		seen.add(c.id);
		max_seen = Math.max(max_seen, c.id);
		if (base0 == null && typeof c.elapsed === "number" && typeof c.frame_start === "number") {
			// client estimate of the server's session clock t0 (network delay only adds latency)
			base0 = performance.now() - c.elapsed * 1000;
			sched_fps = c.fps > 0 ? c.fps : 15;
		}
		if (base0 != null) {
			lat_ms = 1000 * (typeof c.latency === "number" ? c.latency : default_latency_ms / 1000);
			tl.sched = { base0, fps: sched_fps, delay_ms: lat_ms + shift };
		}
		enqueue_chunk(c);
	}

	$effect(() => {
		const v = gradio.props.value;
		untrack(() => on_value(v));
	});

	// ------------------------------------------------------ timeline data
	function record_played(f: QFrame, t: number): void {
		tl.frames.push({
			t,
			dur: 1000 / (f.fps || 15),
			action: f.action,
			has_action: f.has_action,
			fps: f.fps,
			first_in_chunk: f.first_in_chunk,
			index: f.index
		});
	}

	/** New session: clear the timeline (keys still held restart at the current time). */
	function clear_timeline(): void {
		tl.you.length = 0;
		tl.frames.length = 0;
		tl.latency_ms = null;
		tl.session++;
		for (const ax of AXES) delete you_open[ax];
		record_you(axes(held));
	}

	// ------------------------------------------------------------ drawing
	function draw(f: QFrame, t: number): void {
		if (!canvas) return;
		const w = f.bm.width,
			h = f.bm.height;
		if (w && h && (canvas.width !== w || canvas.height !== h)) {
			canvas.width = w;
			canvas.height = h;
			aspect = w / h;
		}
		canvas.getContext("2d")?.drawImage(f.bm, 0, 0, canvas.width, canvas.height);
		draw_cond(f);
		const prev = current;
		current = f;
		if (prev && prev !== f) close_frame(prev);
		if (!has_frame) has_frame = true;

		// latency: time since the control event the backend had when it generated this frame
		if (f.seq != null && sent_at.has(f.seq)) {
			const lag = t - sent_at.get(f.seq)!;
			latency_ema = latency_ema == null ? lag : latency_ema * 0.85 + lag * 0.15;
			tl.latency_ms = latency_ema;
		}
		record_played(f, t);
		if (f.pose_index >= 0) {
			cam.played = f.pose_index + 1;
			cam.version++;
		}
	}

	function draw_cond(f: QFrame | null): void {
		if (!f?.rb || !cond) return;
		const rw = f.rb.width,
			rh = f.rb.height;
		if (rw && rh && (cond.width !== rw || cond.height !== rh)) {
			cond.width = rw;
			cond.height = rh;
		}
		cond.getContext("2d")?.drawImage(f.rb, 0, 0, cond.width, cond.height);
	}

	function redraw_current(): void {
		if (!current || !canvas) return;
		canvas.getContext("2d")?.drawImage(current.bm, 0, 0, canvas.width, canvas.height);
		draw_cond(current);
	}

	function tick(t: number): void {
		raf = requestAnimationFrame(tick);
		const dt = last_t ? Math.min(t - last_t, 250) : 0;
		last_t = t;
		const streaming = server_status === "running" || server_status === "loading";

		if (base0 != null && (queue.length === 0 || queue[0].index != null)) {
			tick_scheduled(t, streaming);
		} else {
			tick_jitter(t, dt, streaming);
		}

		if (t - ui_last > 120) {
			ui_last = t;
			let b: boolean;
			if (base0 != null) {
				const next_due = base0 + lat_ms + shift + ((last_index + 1) * 1000) / sched_fps;
				b = streaming && drawn_this_session && queue.length === 0 && t > next_due + 30;
				const fi = !drawn_this_session && streaming ? Math.max(0, (next_due - t) / 1000) : null;
				if (fi !== first_in && (fi == null || first_in == null || Math.abs(fi - first_in) > 0.05)) first_in = fi;
				const d = (lat_ms + shift) / 1000;
				if (d !== delay) delay = d;
			} else {
				b = streaming && !playing && has_frame;
				if (first_in != null) first_in = null;
				if (delay != null) delay = null;
			}
			if (b !== buffering) buffering = b;
			if (buffered !== queue.length) buffered = queue.length;
			let fps = 0;
			if (drawn_times.length > 2 && t - drawn_times[drawn_times.length - 1] < 1000) {
				fps = ((drawn_times.length - 1) * 1000) / (drawn_times[drawn_times.length - 1] - drawn_times[0]);
			}
			if (Math.abs(fps - play_fps) > 0.05) play_fps = fps;
			const l = latency_ema;
			if (l !== latency && (l == null || latency == null || Math.abs(l - latency) > 15)) latency = l;
			now = performance.now();
			for (const [s, ts] of sent_at) if (t - ts > 60000) sent_at.delete(s);
		}
	}

	/** Fixed-latency playback: show frame f exactly at its due time; wait if early, stall if late. */
	function tick_scheduled(t: number, streaming: boolean): void {
		const due = (f: QFrame) => base0! + lat_ms + shift + (f.index! * 1000) / (f.fps || sched_fps);
		// catch up (e.g. after a background tab paused rAF): drop frames that were on time but missed
		while (queue.length > 1 && due(queue[1]) <= t && queue[1].arrived <= due(queue[1])) {
			const d = queue.shift()!;
			record_played(d, due(d));
			if (d.pose_index >= 0) cam.played = d.pose_index + 1;
			close_frame(d);
		}
		const head = queue[0];
		if (!head || t < due(head)) return;
		const late = t - due(head);
		if (head.arrived > due(head) && late > 0) {
			// stall: the frame came after its due time -> move the schedule forward by the stall
			shift += late;
			if (tl.sched) tl.sched.delay_ms = lat_ms + shift;
		}
		queue.shift();
		playing = true;
		drawn_this_session = true;
		last_index = head.index!;
		draw(head, t);
		drawn_times.push(t);
		if (drawn_times.length > 30) drawn_times.shift();
	}

	/** Legacy playback (chunks without frame_start/elapsed): jitter buffer + catch-up rate. */
	function tick_jitter(t: number, dt: number, streaming: boolean): void {
		if (!playing) {
			if (queue.length >= prebuffer || (queue.length > 0 && !streaming)) {
				playing = true;
				acc = 1000 / (queue[0]?.fps || 15); // show first frame immediately
			}
		}
		if (playing) {
			if (queue.length === 0) {
				playing = false;
			} else {
				const n = queue.length;
				const rate = n > 4 * chunk_len ? catchup * catchup : n > 2 * chunk_len ? catchup : 1;
				acc += dt * rate;
				const interval = 1000 / (queue[0].fps || 15);
				if (acc >= interval) {
					acc = Math.min(acc - interval, interval);
					draw(queue.shift()!, t);
					drawn_times.push(t);
					if (drawn_times.length > 30) drawn_times.shift();
				}
			}
		}
	}

	// ------------------------------------------------------------ controls
	const KEYMAP: Record<string, VKey> = {
		KeyW: "fwd",
		ArrowUp: "fwd",
		KeyS: "back",
		ArrowDown: "back",
		KeyA: "left",
		KeyD: "right",
		KeyQ: "yawL",
		ArrowLeft: "yawL",
		KeyE: "yawR",
		ArrowRight: "yawR",
		KeyR: "pitchU",
		KeyF: "pitchD",
		Space: "up",
		ShiftLeft: "down",
		ShiftRight: "down"
	};

	let kbd: Record<string, VKey> = $state({});
	const held = $derived(new Set<VKey>(Object.values(kbd)));

	function axes(h: Set<VKey>): Action {
		const d = (a: VKey, b: VKey) => (h.has(a) ? 1 : 0) - (h.has(b) ? 1 : 0);
		return {
			forward: d("fwd", "back"),
			right: d("right", "left"),
			up: d("up", "down"),
			yaw: d("yawR", "yawL"),
			pitch: d("pitchU", "pitchD")
		};
	}

	// "You" lane: one open segment per axis while that axis is non-zero
	const you_open: Partial<Record<Axis, Seg>> = {};
	function record_you(a: Action): void {
		const t = performance.now();
		for (const ax of AXES) {
			const open = you_open[ax];
			if (open && open.val === a[ax]) continue;
			if (open) {
				open.t1 = t;
				delete you_open[ax];
			}
			if (a[ax] !== 0) {
				const s: Seg = { axis: ax, val: a[ax], t0: t, t1: null };
				tl.you.push(s);
				you_open[ax] = s;
			}
		}
	}

	let seq = 0;
	let last_sent_key = "";
	let last_sent_at = 0;
	let ctrl_timer: ReturnType<typeof setTimeout> | null = null;

	function send_control(force = false): void {
		const a = axes(held);
		const key = `${a.forward},${a.right},${a.up},${a.yaw},${a.pitch}`;
		if (!force && key === last_sent_key) return;
		const t = performance.now();
		const wait = last_sent_at + min_ctrl_interval - t;
		if (wait > 0) {
			if (!ctrl_timer)
				ctrl_timer = setTimeout(() => {
					ctrl_timer = null;
					send_control(true);
				}, wait);
			return;
		}
		last_sent_key = key;
		last_sent_at = t;
		const payload: ControlPayload = { ...a, seq: ++seq, session: cur_session ?? "" };
		sent_at.set(payload.seq, t);
		gradio.dispatch("control", payload);
	}

	$effect(() => {
		const h = held;
		untrack(() => {
			record_you(axes(h));
			send_control(false);
		});
	});

	$effect(() => {
		if (!active) return;
		const id = setInterval(() => send_control(true), heartbeat_ms);
		return () => clearInterval(id);
	});

	function do_start(): void {
		if (!can_start) return;
		pending = "start";
		stats_at = performance.now();
		stopped_session = null;
		clear_timeline();
		gradio.dispatch("start", {});
		stage?.focus({ preventScroll: true });
	}
	function do_stop(): void {
		if (!can_stop) return;
		pending = "stop";
		// Stop clears immediately: drop buffered frames and ignore chunks still in flight
		stopped_session = cur_session;
		gen++;
		for (const f of queue) close_frame(f);
		queue.length = 0;
		buffered = 0;
		gradio.dispatch("stop", {});
	}

	function on_keydown(e: KeyboardEvent): void {
		if (e.ctrlKey || e.metaKey || e.altKey) return;
		if (e.code === "Enter" || e.code === "NumpadEnter") {
			e.preventDefault();
			do_start();
			return;
		}
		if (e.code === "Escape") {
			e.preventDefault();
			do_stop();
			return;
		}
		const vk = KEYMAP[e.code];
		if (!vk) return;
		e.preventDefault();
		if (kbd[e.code] !== vk) kbd[e.code] = vk;
	}
	function on_keyup(e: KeyboardEvent): void {
		if (KEYMAP[e.code]) {
			e.preventDefault();
			delete kbd[e.code];
		}
	}
	function release_all(): void {
		if (Object.keys(kbd).length) kbd = {};
	}
	function on_blur(): void {
		focused = false;
		release_all();
	}

	// ------------------------------------------------------------ fullscreen
	async function toggle_fullscreen(): Promise<void> {
		try {
			if (document.fullscreenElement) await document.exitFullscreen();
			else await root?.requestFullscreen();
		} catch (e) {
			console.warn("[WorldViewer] fullscreen failed", e);
		}
	}
	function on_visibility(): void {
		if (document.hidden) release_all();
	}
	function on_fs(): void {
		is_fullscreen = document.fullscreenElement === root;
		requestAnimationFrame(redraw_current);
	}

	onMount(() => {
		raf = requestAnimationFrame(tick);
		document.addEventListener("fullscreenchange", on_fs);
		document.addEventListener("visibilitychange", on_visibility);
		window.addEventListener("blur", release_all);
	});
	onDestroy(() => {
		cancelAnimationFrame(raf);
		if (ctrl_timer) clearTimeout(ctrl_timer);
		document.removeEventListener("fullscreenchange", on_fs);
		document.removeEventListener("visibilitychange", on_visibility);
		window.removeEventListener("blur", release_all);
		reset_player(null);
		close_frame(current);
	});

	const TIP_COND = "Your scene's point cloud seen from your camera; black holes are what the model imagines.";
	const TIP_CAM = "Top-down map of your camera path from the start (dashed: not played yet); switch to 3D to orbit.";
	const TIP_TL = "Your keys (right cursor) reach the screen at the model cursor, one latency later.";
</script>

{#snippet info(tip: string, side: "left" | "right")}
	<span class="info {side}" tabindex="0" role="note" aria-label={tip} data-tip={tip}>i</span>
{/snippet}

<Block
	visible={gradio.shared.visible}
	elem_id={gradio.shared.elem_id}
	elem_classes={gradio.shared.elem_classes}
	container={gradio.shared.container}
	scale={gradio.shared.scale}
	min_width={gradio.shared.min_width}
	padding={false}
	allow_overflow={true}
>
	<div class="wv" class:fullscreen={is_fullscreen} bind:this={root}>
		<div class="grid">
			<div class="main">
				<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
				<div
					class="stage"
					class:focused
					bind:this={stage}
					tabindex="0"
					role="application"
					aria-label="World viewer. Click, then use W A S D, arrows, Q E, R F, Space and Shift to move the camera. Enter starts, Escape stops."
					style:--ar={aspect}
					onkeydown={on_keydown}
					onkeyup={on_keyup}
					onfocus={() => (focused = true)}
					onblur={on_blur}
					onpointerdown={() => stage?.focus({ preventScroll: true })}
					data-testid="worldviewer-stage"
				>
					<canvas bind:this={canvas} class="screen" class:hidden={!has_frame} data-testid="worldviewer-canvas"></canvas>
					{#if !has_frame}
						<div class="poster"><div class="poster-grid"></div></div>
					{/if}

					<div class="hud glass" class:hidden={status === "idle" && !has_frame} aria-live="polite" data-testid="worldviewer-hud">
						<div class="hud-row status-row">
							<span class="dot {status}"></span>
							<span class="status-name">{status}</span>
							{#if message && status !== "loading" && status !== "error"}
								<span class="msg" title={message}>{message}</span>
							{/if}
						</div>
						<div class="hud-row mono">
							<span><span class="k">play</span> {play_fps.toFixed(1)}</span>
							{#if gen_fps != null}<span><span class="k">gen</span> {gen_fps.toFixed(1)} fps</span>{/if}
							<span><span class="k">buf</span> {buffered}</span>
							{#if delay != null}<span class="lat"><span class="k">delay</span> {delay.toFixed(2)} s</span>{/if}
							{#if latency != null}<span class={delay != null ? "" : "lat"}><span class="k">{delay != null ? "lag" : "latency"}</span> {(latency / 1000).toFixed(2)} s</span>{/if}
						</div>
					</div>

					<div class="tools">
						<button
							class="icon-btn glass"
							title={is_fullscreen ? "Exit fullscreen" : "Fullscreen"}
							onclick={toggle_fullscreen}
							aria-label="Toggle fullscreen"
						>
							{#if is_fullscreen}
								<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3M21 8h-3a2 2 0 0 1-2-2V3M3 16h3a2 2 0 0 1 2 2v3M16 21v-3a2 2 0 0 1 2-2h3" /></svg>
							{:else}
								<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3" /></svg>
							{/if}
						</button>
					</div>

					{#if status === "loading"}
						<div class="overlay dim">
							<div class="spinner" aria-hidden="true"></div>
							<div class="ov-text">{message || "Loading…"}</div>
						</div>
					{:else if status === "error"}
						<div class="overlay error">
							<div class="ov-title">Error</div>
							<div class="ov-text">{message}</div>
							<button class="btn primary small" onclick={do_start}>Try again</button>
						</div>
					{:else if status === "ended" && buffered === 0}
						<div class="overlay dim">
							<div class="ov-title">Session ended</div>
							{#if message}<div class="ov-text">{message}</div>{/if}
							<button class="btn primary small" onclick={do_start}>
								<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
								Start again
							</button>
						</div>
					{:else if status === "idle"}
						<div class="overlay" class:dim={has_frame}>
							<button class="big-play" onclick={do_start} aria-label="Start">
								<svg viewBox="0 0 24 24" width="30" height="30" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
							</button>
							<div class="ov-text">{message || placeholder_text}</div>
						</div>
					{/if}

					{#if buffering && status === "running"}
						<div class="chip buffering glass"><span class="mini-spin"></span>buffering</div>
					{/if}
					{#if first_in != null && status === "running"}
						<div class="chip buffering glass"><span class="mini-spin"></span>first frame in {first_in.toFixed(1)} s</div>
					{/if}
					{#if status === "running" && !focused}
						<div class="chip hint glass">Click here to control</div>
					{/if}
					{#if limit_s}
						<div class="session-bar" title="Session time {elapsed_s.toFixed(0)}s / {limit_s.toFixed(0)}s">
							<div class="fill" style:width="{Math.min(100, (100 * elapsed_s) / limit_s)}%"></div>
						</div>
					{/if}
				</div>

				<div class="toolbar">
					<div class="left">
						<button class="btn primary" disabled={!can_start} onclick={do_start} data-testid="worldviewer-start">
							<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
							Start
						</button>
						<button class="btn stop" disabled={!can_stop} onclick={do_stop} data-testid="worldviewer-stop">
							<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
							Stop
						</button>
						{#if limit_s}
							<span class="time mono">{elapsed_s.toFixed(0)}s / {limit_s.toFixed(0)}s</span>
						{/if}
					</div>
					<div class="hints" aria-hidden="true">
						<span><kbd class:on={held.has("fwd")}>W</kbd><kbd class:on={held.has("left")}>A</kbd><kbd class:on={held.has("back")}>S</kbd><kbd class:on={held.has("right")}>D</kbd> move</span>
						<span><kbd class:on={held.has("yawL")}>Q</kbd><kbd class:on={held.has("yawR")}>E</kbd>/<kbd>←</kbd><kbd>→</kbd> turn</span>
						<span><kbd class:on={held.has("pitchU")}>R</kbd><kbd class:on={held.has("pitchD")}>F</kbd> look</span>
						<span><kbd class:on={held.has("up")}>Space</kbd><kbd class:on={held.has("down")}>⇧</kbd> up/down</span>
						<span><kbd>↵</kbd> start <kbd>Esc</kbd> stop</span>
					</div>
				</div>
			</div>

			<div class="side">
				<section class="panel cond-panel" class:closed={!cond_open}>
					<header>
						<span class="title">What the model sees</span>
						{@render info(TIP_COND, "right")}
						<span class="grow"></span>
						<button class="mini" onclick={() => { cond_open = !cond_open; requestAnimationFrame(redraw_current); }} aria-pressed={cond_open} data-testid="worldviewer-cond-toggle">
							{cond_open ? "Hide" : "Show"}
						</button>
					</header>
					<div class="panel-body" class:hidden={!cond_open}>
						<div class="cond-wrap" style:--ar={aspect}>
							<canvas bind:this={cond} class="cond" class:hidden={!has_renders} data-testid="worldviewer-cond"></canvas>
							{#if !has_renders}<div class="empty">No render condition yet</div>{/if}
						</div>
						<div class="caption">Point cloud splatted into your camera — black holes are what the model imagines</div>
					</div>
				</section>

				<section class="panel cam-panel" class:closed={!cam_open}>
					<header>
						<span class="title">Camera</span>
						{@render info(TIP_CAM, "right")}
						{#if scene_title}<span class="sub" title={scene_title}>{scene_title}</span>{/if}
						<span class="grow"></span>
						{#if cam_open}
							<div class="seg" role="group" aria-label="Camera view">
								<button class:on={cam_mode === "map"} aria-pressed={cam_mode === "map"} onclick={() => set_cam_mode("map")} data-testid="worldviewer-cam-map">Map</button>
								<button class:on={cam_mode === "3d"} aria-pressed={cam_mode === "3d"} onclick={() => set_cam_mode("3d")} data-testid="worldviewer-cam-3d">3D</button>
							</div>
							<button class="mini" onclick={() => cam_reset++} title="Re-frame the view">Reset view</button>
						{/if}
						<button class="mini" onclick={() => (cam_open = !cam_open)} aria-pressed={cam_open}>
							{cam_open ? "Hide" : "Show"}
						</button>
					</header>
					<div class="panel-body cam-body" class:hidden={!cam_open}>
						{#if cam_mode === "map"}
							<MapView store={cam} visible={cam_open} reset={cam_reset} />
						{:else if camera_view}
							{#await camera_view then m}
								<m.default store={cam} visible={cam_open} reset={cam_reset} />
							{:catch}
								<div class="empty">3D view unavailable</div>
							{/await}
						{/if}
						<div class="cam-legend">
							<span><i class="mk start"></i>start</span>
							<span><i class="sw played"></i>played</span>
							<span><i class="sw buf"></i>upcoming</span>
							<span><i class="mk cur"></i>camera</span>
						</div>
					</div>
				</section>
			</div>
		</div>

		{#if show_timeline}
			<section class="panel tl-panel">
				<header>
					<span class="title">Timeline</span>
					{@render info(TIP_TL, "left")}
					<span class="sub">your keys vs. what is on screen</span>
					<span class="grow"></span>
					{#if latency != null}<span class="lat-badge mono" title="Measured time from a control event to the frame generated with it">measured lag {(latency / 1000).toFixed(2)} s</span>{/if}
				</header>
				<Timeline store={tl} seconds={timeline_seconds} running={status === "running"} />
			</section>
		{/if}
	</div>
</Block>

<style>
	.wv {
		--wv-radius: var(--block-radius, 14px);
		--wv-accent: var(--color-accent, #f97316);
		--wv-glass: rgba(14, 16, 22, 0.55);
		--wv-glass-border: rgba(255, 255, 255, 0.12);
		--wv-panel: #0e1117;
		--wv-panel-border: rgba(255, 255, 255, 0.08);
		--wv-fg: #f4f5f7;
		--wv-muted: rgba(244, 245, 247, 0.62);
		container-type: inline-size;
		container-name: wv;
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 10px;
		width: 100%;
		box-sizing: border-box;
		color: var(--body-text-color);
		/* clip everything (incl. blurred / transformed layers) to the block's rounded frame */
		border-radius: var(--block-radius, 14px);
		overflow: hidden;
		isolation: isolate;
	}
	.wv.fullscreen {
		background: #07080b;
		border-radius: 0;
		height: 100vh;
		overflow: auto;
		padding: 14px;
		color: var(--wv-fg);
	}
	.grid {
		display: grid;
		grid-template-columns: minmax(0, 65fr) minmax(0, 35fr);
		gap: 10px;
		align-items: stretch;
	}
	.main {
		display: flex;
		flex-direction: column;
		gap: 10px;
		min-width: 0;
	}
	.side {
		display: flex;
		flex-direction: column;
		gap: 10px;
		min-width: 0;
	}
	@container wv (max-width: 760px) {
		.grid {
			grid-template-columns: minmax(0, 1fr);
		}
		.cam-panel .cam-body {
			height: 260px;
			flex: none;
		}
		.hints {
			display: none !important;
		}
	}

	/* ---------------------------------------------------------------- stage */
	.stage {
		position: relative;
		width: 100%;
		aspect-ratio: var(--ar);
		border-radius: var(--wv-radius);
		overflow: hidden;
		isolation: isolate;
		/* backdrop-filter / transformed children can escape a rounded overflow clip in Chromium;
		   clip-path + paint containment keep every layer inside the rounded box */
		clip-path: inset(0 round var(--wv-radius));
		contain: paint;
		background: radial-gradient(120% 120% at 50% 0%, #1b1f2a 0%, #0b0d12 60%, #06070a 100%);
		outline: none;
		box-shadow:
			0 0 0 1px rgba(255, 255, 255, 0.06) inset,
			0 10px 30px -12px rgba(0, 0, 0, 0.45);
		transition: box-shadow 0.18s ease;
		touch-action: none;
		user-select: none;
		-webkit-user-select: none;
		color: var(--wv-fg);
		font-family: var(--font, ui-sans-serif, system-ui, sans-serif);
	}
	.stage.focused {
		box-shadow:
			0 0 0 2px var(--wv-accent),
			0 10px 30px -12px rgba(0, 0, 0, 0.45);
	}
	.screen {
		display: block;
		width: 100%;
		height: 100%;
		object-fit: contain;
		border-radius: inherit;
	}
	.hidden {
		display: none !important;
	}
	.poster {
		position: absolute;
		inset: 0;
		overflow: hidden; /* the grid below is scaled/transformed: keep it inside */
		border-radius: inherit;
	}
	.poster-grid {
		position: absolute;
		inset: 0;
		background-image:
			linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px),
			linear-gradient(90deg, rgba(255, 255, 255, 0.05) 1px, transparent 1px);
		background-size: 40px 40px;
		mask-image: radial-gradient(ellipse at 50% 60%, black 20%, transparent 75%);
		-webkit-mask-image: radial-gradient(ellipse at 50% 60%, black 20%, transparent 75%);
		transform: perspective(500px) rotateX(55deg) translateY(18%) scale(1.6);
		transform-origin: 50% 100%;
	}
	.glass {
		background: var(--wv-glass);
		border: 1px solid var(--wv-glass-border);
		backdrop-filter: blur(10px) saturate(140%);
		-webkit-backdrop-filter: blur(10px) saturate(140%);
	}

	/* ------------------------------------------------------------------ HUD */
	.hud {
		position: absolute;
		top: 10px;
		left: 10px;
		z-index: 4;
		padding: 7px 10px;
		border-radius: 10px;
		font-size: 11px;
		line-height: 1.35;
		color: var(--wv-fg);
		max-width: min(70%, 420px);
		display: flex;
		flex-direction: column;
		gap: 3px;
		pointer-events: none;
	}
	.hud-row {
		display: flex;
		flex-wrap: wrap;
		gap: 2px 10px;
		align-items: center;
		color: var(--wv-fg);
	}
	.hud .k {
		color: rgba(244, 245, 247, 0.5);
	}
	.hud .lat {
		color: #fdba74;
	}
	.mono {
		font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
		font-variant-numeric: tabular-nums;
	}
	.status-row {
		gap: 6px;
	}
	.status-name {
		text-transform: uppercase;
		letter-spacing: 0.08em;
		font-weight: 600;
		font-size: 10px;
	}
	.msg {
		color: var(--wv-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 100%;
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: #9ca3af;
		flex: none;
	}
	.dot.running {
		background: #22c55e;
		animation: pulse 1.6s infinite;
	}
	.dot.loading {
		background: #eab308;
	}
	.dot.error {
		background: #ef4444;
	}
	.dot.ended {
		background: #60a5fa;
	}
	@keyframes pulse {
		0% {
			box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55);
		}
		70% {
			box-shadow: 0 0 0 6px rgba(34, 197, 94, 0);
		}
		100% {
			box-shadow: 0 0 0 0 rgba(34, 197, 94, 0);
		}
	}
	.session-bar {
		position: absolute;
		left: 0;
		right: 0;
		bottom: 0;
		border-radius: 0 0 var(--wv-radius) var(--wv-radius);
		overflow: hidden;
		height: 3px;
		background: rgba(255, 255, 255, 0.1);
		z-index: 3;
	}
	.session-bar .fill {
		height: 100%;
		background: linear-gradient(90deg, var(--wv-accent), #fbbf24);
		transition: width 0.25s linear;
	}

	/* ---------------------------------------------------------------- tools */
	.tools {
		position: absolute;
		top: 10px;
		right: 10px;
		z-index: 5;
		display: flex;
		gap: 6px;
	}
	.icon-btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		height: 30px;
		padding: 0 9px;
		border-radius: 8px;
		color: var(--wv-fg);
		cursor: pointer;
		font-size: 12px;
	}
	.icon-btn:hover {
		background: rgba(30, 34, 44, 0.75);
	}

	/* ------------------------------------------------------------- overlays */
	.overlay {
		position: absolute;
		inset: 0;
		border-radius: inherit;
		z-index: 2;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 12px;
		text-align: center;
		padding: 16px;
	}
	.overlay.dim {
		background: rgba(5, 6, 9, 0.55);
		backdrop-filter: blur(2px);
	}
	.overlay.error {
		background: rgba(60, 8, 12, 0.72);
		backdrop-filter: blur(2px);
	}
	.ov-title {
		font-size: 18px;
		font-weight: 650;
	}
	.ov-text {
		font-size: 13px;
		color: var(--wv-muted);
		max-width: 520px;
	}
	.overlay.error .ov-text {
		color: #fecaca;
	}
	.big-play {
		width: 68px;
		height: 68px;
		border-radius: 50%;
		display: grid;
		place-items: center;
		cursor: pointer;
		color: #fff;
		background: var(--wv-accent);
		border: none;
		box-shadow:
			0 0 0 8px rgba(255, 255, 255, 0.06),
			0 10px 30px rgba(0, 0, 0, 0.45);
		transition: transform 0.15s ease;
	}
	.big-play:hover {
		transform: scale(1.06);
	}
	.big-play svg {
		margin-left: 4px;
	}
	.spinner {
		width: 40px;
		height: 40px;
		border-radius: 50%;
		border: 3px solid rgba(255, 255, 255, 0.15);
		border-top-color: var(--wv-accent);
		animation: spin 0.9s linear infinite;
	}
	.mini-spin {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		border: 2px solid rgba(255, 255, 255, 0.25);
		border-top-color: #fff;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
	.chip {
		position: absolute;
		z-index: 4;
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		border-radius: 999px;
		font-size: 11px;
		color: var(--wv-fg);
		pointer-events: none;
	}
	.chip.buffering {
		right: 12px;
		bottom: 12px;
	}
	.chip.hint {
		left: 50%;
		bottom: 12px;
		transform: translateX(-50%);
	}

	/* -------------------------------------------------------------- toolbar */
	.toolbar {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
	}
	.toolbar .left {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.time {
		font-size: 12px;
		color: var(--body-text-color-subdued, inherit);
		margin-left: 4px;
	}
	.btn {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 7px;
		height: 40px;
		padding: 0 20px;
		border-radius: 10px;
		font-size: 14px;
		font-weight: 600;
		cursor: pointer;
		border: 1px solid transparent;
		transition:
			filter 0.15s,
			transform 0.05s,
			opacity 0.15s;
	}
	.btn:active:not(:disabled) {
		transform: translateY(1px);
	}
	.btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.btn.primary {
		background: var(--wv-accent);
		color: #fff;
	}
	.btn.primary:hover:not(:disabled) {
		filter: brightness(1.08);
	}
	.btn.stop {
		background: var(--button-secondary-background-fill, rgba(127, 127, 127, 0.15));
		color: var(--button-secondary-text-color, var(--body-text-color));
		border-color: var(--border-color-primary, rgba(127, 127, 127, 0.3));
	}
	.btn.stop:hover:not(:disabled) {
		background: #dc2626;
		border-color: #dc2626;
		color: #fff;
	}
	.btn.small {
		height: 32px;
		padding: 0 14px;
		font-size: 13px;
	}
	.hints {
		flex: 1;
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: 4px 12px;
		font-size: 12px;
		color: var(--body-text-color-subdued, rgba(127, 127, 127, 0.9));
	}
	.hints span {
		display: inline-flex;
		align-items: center;
		gap: 3px;
		white-space: nowrap;
	}
	kbd {
		display: inline-grid;
		place-items: center;
		min-width: 20px;
		height: 20px;
		padding: 0 5px;
		box-sizing: border-box;
		font-family: var(--font-mono, ui-monospace, monospace);
		font-size: 10.5px;
		font-weight: 600;
		border-radius: 5px;
		color: var(--body-text-color);
		background: var(--background-fill-secondary, rgba(127, 127, 127, 0.12));
		border: 1px solid var(--border-color-primary, rgba(127, 127, 127, 0.35));
		border-bottom-width: 2px;
		transition: background 0.08s;
	}
	kbd.on {
		background: var(--wv-accent);
		border-color: var(--wv-accent);
		color: #fff;
	}
	.fullscreen .hints,
	.fullscreen .time {
		color: rgba(244, 245, 247, 0.6);
	}
	.fullscreen kbd:not(.on) {
		color: #eee;
		background: rgba(255, 255, 255, 0.08);
		border-color: rgba(255, 255, 255, 0.2);
	}

	/* --------------------------------------------------------------- panels */
	.panel {
		background: var(--wv-panel);
		border: 1px solid var(--wv-panel-border);
		border-radius: 12px;
		color: var(--wv-fg);
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.panel header {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 7px 10px;
		font-size: 11px;
		min-height: 20px;
	}
	.panel .title {
		text-transform: uppercase;
		letter-spacing: 0.09em;
		font-weight: 650;
		font-size: 10.5px;
		color: rgba(244, 245, 247, 0.82);
		white-space: nowrap;
	}
	.panel .sub {
		color: rgba(244, 245, 247, 0.45);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		min-width: 0;
	}
	.grow {
		flex: 1;
	}
	.mini {
		height: 22px;
		padding: 0 8px;
		border-radius: 6px;
		font-size: 10.5px;
		color: rgba(244, 245, 247, 0.8);
		background: rgba(255, 255, 255, 0.06);
		border: 1px solid rgba(255, 255, 255, 0.1);
		cursor: pointer;
		white-space: nowrap;
	}
	.mini:hover {
		background: rgba(255, 255, 255, 0.12);
	}
	.panel-body {
		padding: 0 10px 10px;
		min-height: 0;
	}
	.cond-wrap {
		position: relative;
		width: 100%;
		aspect-ratio: var(--ar);
		border-radius: 8px;
		overflow: hidden;
		isolation: isolate;
		background: #000;
	}
	.cond {
		display: block;
		width: 100%;
		height: 100%;
		image-rendering: auto;
	}
	.empty {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		font-size: 11px;
		color: rgba(244, 245, 247, 0.4);
	}
	.caption {
		margin-top: 6px;
		font-size: 11px;
		line-height: 1.35;
		color: rgba(244, 245, 247, 0.55);
	}
	.cam-panel {
		flex: 1 1 auto;
		min-height: 0;
	}
	.cam-panel.closed {
		flex: none;
	}
	.cam-body {
		position: relative;
		flex: 1 1 auto;
		min-height: 200px;
		padding: 0;
		border-radius: 0 0 12px 12px;
		overflow: hidden;
		isolation: isolate;
		clip-path: inset(0 round 0 0 12px 12px);
		background: radial-gradient(100% 100% at 50% 30%, #161a23 0%, #0b0d12 100%);
	}
	.cam-legend {
		position: absolute;
		left: 8px;
		bottom: 6px;
		display: flex;
		gap: 10px;
		font-size: 10px;
		color: rgba(244, 245, 247, 0.7);
		pointer-events: none;
		padding: 2px 7px;
		border-radius: 6px;
		background: rgba(11, 13, 18, 0.7);
	}
	.cam-legend span {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.sw {
		display: inline-block;
		width: 12px;
		height: 3px;
	}
	.mk {
		display: inline-block;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		box-sizing: border-box;
	}
	.mk.start {
		border: 2px solid #f4f5f7;
	}
	.mk.cur {
		background: var(--wv-accent);
		border: 1.5px solid #fff;
	}
	.seg {
		display: inline-flex;
		border: 1px solid rgba(255, 255, 255, 0.12);
		border-radius: 6px;
		overflow: hidden;
	}
	.seg button {
		height: 22px;
		padding: 0 8px;
		font-size: 10.5px;
		color: rgba(244, 245, 247, 0.7);
		background: transparent;
		border: none;
		cursor: pointer;
	}
	.seg button.on {
		background: rgba(249, 115, 22, 0.22);
		color: #fff;
	}
	.sw.played {
		background: var(--wv-accent);
	}
	.sw.buf {
		background: repeating-linear-gradient(90deg, var(--wv-accent) 0 4px, transparent 4px 7px);
		opacity: 0.5;
	}
	.tl-panel {
		padding-bottom: 8px;
	}
	.lat-badge {
		font-size: 11px;
		color: #fdba74;
		background: rgba(249, 115, 22, 0.12);
		border: 1px solid rgba(249, 115, 22, 0.35);
		padding: 1px 7px;
		border-radius: 999px;
		white-space: nowrap;
	}

	/* ------------------------------------------------------------- tooltips */
	.info {
		position: relative;
		display: inline-grid;
		place-items: center;
		width: 14px;
		height: 14px;
		flex: none;
		border-radius: 50%;
		font-size: 9px;
		font-weight: 700;
		font-style: italic;
		font-family: Georgia, serif;
		color: rgba(244, 245, 247, 0.7);
		border: 1px solid rgba(244, 245, 247, 0.35);
		cursor: help;
		outline: none;
	}
	.info:hover::after,
	.info:focus::after {
		content: attr(data-tip);
		position: absolute;
		top: 20px;
		z-index: 50;
		width: 260px;
		padding: 8px 10px;
		border-radius: 8px;
		background: #1a1e27;
		border: 1px solid rgba(255, 255, 255, 0.14);
		box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
		color: #e5e7eb;
		font: 400 11.5px/1.45 var(--font, ui-sans-serif, system-ui, sans-serif);
		font-style: normal;
		white-space: normal;
		text-align: left;
	}
	.info.left:hover::after,
	.info.left:focus::after {
		left: -6px;
	}
	.info.right:hover::after,
	.info.right:focus::after {
		right: auto;
		left: -120px;
	}

	@media (max-width: 640px) {
		.wv {
			padding: 6px;
		}
		.hud {
			font-size: 10px;
			padding: 6px 8px;
		}
		.toolbar {
			justify-content: flex-start;
		}
		.btn {
			padding: 0 16px;
		}
		.info:hover::after,
		.info:focus::after {
			width: 220px;
		}
	}
</style>
