<script lang="ts">
	import { Gradio } from "@gradio/utils";
	import { Block } from "@gradio/atoms";
	import { onDestroy, onMount, untrack } from "svelte";
	import type {
		Chunk,
		ControlPayload,
		Frame,
		Status,
		VKey,
		WorldViewerEvents,
		WorldViewerProps,
		WorldViewerValue
	} from "./types";

	const props = $props();
	const gradio = new Gradio<WorldViewerEvents, WorldViewerProps>(props);
	gradio.watch_for_change();

	// ---------------------------------------------------------------- config
	const prebuffer = $derived(Math.max(1, gradio.props.prebuffer_frames ?? 6));
	const catchup = $derived(gradio.props.catchup_rate ?? 1.15);
	const heartbeat_ms = $derived(gradio.props.heartbeat_ms ?? 250);
	const min_ctrl_interval = $derived(
		1000 / Math.max(1, gradio.props.max_control_hz ?? 20)
	);
	const placeholder_text = $derived(
		gradio.props.placeholder ??
			"Press Start, then click here and use WASD / arrows to move"
	);

	// ------------------------------------------------------------ UI state
	let root: HTMLDivElement | undefined = $state();
	let stage: HTMLDivElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();
	let pip: HTMLCanvasElement | undefined = $state();

	let pending: "start" | "stop" | null = $state(null);
	let focused = $state(false);
	let is_fullscreen = $state(false);
	let show_render = $state(untrack(() => !!gradio.props.show_render));
	let pad_open = $state(false);
	let has_frame = $state(false);
	let has_renders = $state(false);
	let buffering = $state(false);
	let buffered = $state(0);
	let play_fps = $state(0);
	let aspect = $state(untrack(() => gradio.props.aspect_ratio || 832 / 480));
	let now = $state(performance.now());
	let stats_at = $state(performance.now());

	const value: WorldViewerValue | null = $derived(gradio.props.value ?? null);
	const server_status: Status = $derived(value?.status ?? "idle");
	const loading_status = $derived(gradio.shared.loading_status);
	const ls_error = $derived(
		loading_status?.status === "error" &&
			(pending === "start" ||
				server_status === "loading" ||
				server_status === "running")
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
					: value?.message ?? ""
	);
	const active = $derived(status === "running" || status === "loading");
	const can_start = $derived(!active);
	const can_stop = $derived(active);
	const stats = $derived(value?.stats ?? {});
	const limit_s = $derived(
		typeof stats.limit_s === "number" ? (stats.limit_s as number) : null
	);
	const elapsed_s = $derived.by(() => {
		const base = typeof stats.elapsed_s === "number" ? (stats.elapsed_s as number) : 0;
		const extra = server_status === "running" ? (now - stats_at) / 1000 : 0;
		return limit_s ? Math.min(limit_s, base + extra) : base + extra;
	});
	const extra_stats = $derived(
		Object.entries(stats).filter(([k]) => k !== "elapsed_s" && k !== "limit_s")
	);

	function queue_text(): string {
		const ls = gradio.shared.loading_status;
		if (ls?.status === "pending" && ls.queue_position != null && ls.queue_position > 0)
			return `Waiting in queue (position ${ls.queue_position + 1}${ls.queue_size ? ` / ${ls.queue_size}` : ""})…`;
		return "";
	}

	function fmt(k: string, v: number | string): string {
		if (typeof v !== "number") return String(v);
		if (k.endsWith("_ms")) return `${v.toFixed(0)} ms`;
		if (k.endsWith("_s")) return `${v.toFixed(1)} s`;
		return Number.isInteger(v) ? String(v) : v.toFixed(1);
	}
	function label_of(k: string): string {
		return k.replace(/_(ms|s)$/, "").replace(/_/g, " ");
	}

	// ------------------------------------------------------- player state
	let queue: Frame[] = [];
	let gen = 0;
	let decode_chain: Promise<void> = Promise.resolve();
	let cur_session: string | null = null;
	let seen = new Set<number>();
	let max_seen = -1;
	let chunk_len = 12;
	let playing = false;
	let last_t = 0;
	let acc = 0;
	let current: Frame | null = null;
	let drawn_times: number[] = [];
	let raf = 0;
	let ui_last = 0;

	function close_frame(f: Frame | null): void {
		if (!f) return;
		if ("close" in f.bm) f.bm.close();
		if (f.rb && "close" in f.rb) f.rb.close();
	}

	function reset_player(session: string | null): void {
		gen++;
		for (const f of queue) close_frame(f);
		queue = [];
		seen = new Set();
		max_seen = -1;
		cur_session = session;
		playing = false;
		acc = 0;
		drawn_times = [];
		buffered = 0;
	}

	function data_uri_to_blob(uri: string): Blob {
		const comma = uri.indexOf(",");
		const header = uri.slice(5, comma);
		const mime = header.split(";")[0] || "image/jpeg";
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

	function enqueue_chunk(c: Chunk): void {
		const my_gen = gen;
		const fps = c.fps > 0 ? c.fps : 15;
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
			const frames = bms.map((bm, i) => ({ bm, rb: rbs?.[i] ?? null, fps }));
			if (my_gen !== gen) {
				frames.forEach(close_frame);
				return;
			}
			if (rbs && rbs.some(Boolean)) has_renders = true;
			chunk_len = Math.max(1, frames.length);
			queue.push(...frames);
			buffered = queue.length;
		});
	}

	function on_value(v: WorldViewerValue | null): void {
		if (!v) return;
		if (pending === "start") pending = null;
		else if (pending === "stop" && v.status !== "running" && v.status !== "loading")
			pending = null;
		stats_at = performance.now();
		const c = v.chunk;
		if (!c || !Array.isArray(c.frames) || c.frames.length === 0) return;
		if (c.session !== cur_session) reset_player(c.session);
		else if (c.id === 0 && max_seen > 0) reset_player(c.session);
		else if (seen.has(c.id)) return; // repeated value update (e.g. final re-send)
		seen.add(c.id);
		max_seen = Math.max(max_seen, c.id);
		enqueue_chunk(c);
	}

	$effect(() => {
		const v = gradio.props.value;
		untrack(() => on_value(v));
	});

	function draw(f: Frame): void {
		if (!canvas) return;
		const w = f.bm.width,
			h = f.bm.height;
		if (w && h && (canvas.width !== w || canvas.height !== h)) {
			canvas.width = w;
			canvas.height = h;
			aspect = w / h;
		}
		canvas.getContext("2d")?.drawImage(f.bm, 0, 0, canvas.width, canvas.height);
		if (f.rb && pip) {
			const rw = f.rb.width,
				rh = f.rb.height;
			if (rw && rh && (pip.width !== rw || pip.height !== rh)) {
				pip.width = rw;
				pip.height = rh;
			}
			pip.getContext("2d")?.drawImage(f.rb, 0, 0, pip.width, pip.height);
		}
		const prev = current;
		current = f;
		if (prev && prev !== f) close_frame(prev);
		if (!has_frame) has_frame = true;
	}

	function redraw_current(): void {
		if (!current || !canvas) return;
		canvas.getContext("2d")?.drawImage(current.bm, 0, 0, canvas.width, canvas.height);
		if (current.rb && pip)
			pip.getContext("2d")?.drawImage(current.rb, 0, 0, pip.width, pip.height);
	}

	function tick(t: number): void {
		raf = requestAnimationFrame(tick);
		const dt = last_t ? Math.min(t - last_t, 250) : 0;
		last_t = t;
		const streaming = server_status === "running" || server_status === "loading";

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
				const rate =
					n > 4 * chunk_len ? catchup * catchup : n > 2 * chunk_len ? catchup : 1;
				acc += dt * rate;
				const interval = 1000 / (queue[0].fps || 15);
				if (acc >= interval) {
					acc = Math.min(acc - interval, interval);
					draw(queue.shift()!);
					drawn_times.push(t);
					if (drawn_times.length > 30) drawn_times.shift();
				}
			}
		}

		if (t - ui_last > 120) {
			ui_last = t;
			const b = streaming && !playing && has_frame;
			if (b !== buffering) buffering = b;
			if (buffered !== queue.length) buffered = queue.length;
			let fps = 0;
			if (drawn_times.length > 2 && t - drawn_times[drawn_times.length - 1] < 1000) {
				fps =
					((drawn_times.length - 1) * 1000) /
					(drawn_times[drawn_times.length - 1] - drawn_times[0]);
			}
			if (Math.abs(fps - play_fps) > 0.05) play_fps = fps;
			now = performance.now();
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
	let ptr: Record<number, VKey> = $state({});
	const held = $derived(new Set<VKey>([...Object.values(kbd), ...Object.values(ptr)]));

	function axes(h: Set<VKey>) {
		const d = (a: VKey, b: VKey) => (h.has(a) ? 1 : 0) - (h.has(b) ? 1 : 0);
		return {
			forward: d("fwd", "back"),
			right: d("right", "left"),
			up: d("up", "down"),
			yaw: d("yawR", "yawL"),
			pitch: d("pitchU", "pitchD")
		};
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
		gradio.dispatch("control", payload);
	}

	$effect(() => {
		held; // re-run when the held set changes
		untrack(() => send_control(false));
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
		gradio.dispatch("start", {});
		stage?.focus({ preventScroll: true });
	}
	function do_stop(): void {
		if (!can_stop) return;
		pending = "stop";
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
		if (Object.keys(ptr).length) ptr = {};
	}
	function on_blur(): void {
		focused = false;
		if (Object.keys(kbd).length) kbd = {};
	}

	function pad_down(e: PointerEvent, vk: VKey): void {
		e.preventDefault();
		(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
		ptr[e.pointerId] = vk;
		stage?.focus({ preventScroll: true });
	}
	function pad_up(e: PointerEvent): void {
		if (e.pointerId in ptr) delete ptr[e.pointerId];
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
		const sp = gradio.props.show_pad;
		pad_open =
			sp === true ||
			(sp == null && window.matchMedia?.("(pointer: coarse)").matches === true);
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

	const HUD_KEYS: { k: string; vk: VKey }[][] = [
		[
			{ k: "Q", vk: "yawL" },
			{ k: "W", vk: "fwd" },
			{ k: "E", vk: "yawR" },
			{ k: "R", vk: "pitchU" }
		],
		[
			{ k: "A", vk: "left" },
			{ k: "S", vk: "back" },
			{ k: "D", vk: "right" },
			{ k: "F", vk: "pitchD" }
		]
	];
</script>

<Block
	visible={gradio.shared.visible}
	elem_id={gradio.shared.elem_id}
	elem_classes={gradio.shared.elem_classes}
	container={gradio.shared.container}
	scale={gradio.shared.scale}
	min_width={gradio.shared.min_width}
	padding={false}
	allow_overflow={false}
>
	<div class="wv" class:fullscreen={is_fullscreen} bind:this={root}>
		<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
		<div
			class="stage"
			class:focused
			class:active
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
			<canvas bind:this={canvas} class="screen" class:hidden={!has_frame} data-testid="worldviewer-canvas"
			></canvas>
			{#if !has_frame}
				<div class="poster">
					<div class="poster-grid"></div>
				</div>
			{/if}

			<canvas
				bind:this={pip}
				class="pip"
				class:hidden={!(show_render && has_renders)}
				aria-label="Render condition preview"
			></canvas>

			<!-- HUD -->
			<div class="hud glass" class:hidden={status === "idle" && !has_frame} aria-live="polite">
				<div class="hud-row status-row">
					<span class="dot {status}"></span>
					<span class="status-name">{status}</span>
					{#if message && status !== "loading" && status !== "error"}
						<span class="msg" title={message}>{message}</span>
					{/if}
				</div>
				<div class="hud-row mono">
					<span>play {play_fps.toFixed(1)} fps</span>
					<span class="sep">·</span>
					<span>buf {buffered}</span>
				</div>
				{#if extra_stats.length}
					<div class="hud-row mono stats">
						{#each extra_stats as [k, v] (k)}
							<span><span class="k">{label_of(k)}</span> {fmt(k, v)}</span>
						{/each}
					</div>
				{/if}
				<div class="hud-keys" aria-hidden="true">
					{#each HUD_KEYS as row}
						<div class="kr">
							{#each row as key (key.k)}
								<span class="mk" class:on={held.has(key.vk)}>{key.k}</span>
							{/each}
						</div>
					{/each}
					<div class="kr">
						<span class="mk wide" class:on={held.has("down")}>⇧</span>
						<span class="mk wider" class:on={held.has("up")}>␣</span>
					</div>
				</div>
				{#if limit_s}
					<div class="progress" title="Session time">
						<div class="bar"><div class="fill" style:width="{Math.min(100, (100 * elapsed_s) / limit_s)}%"></div></div>
						<span class="mono">{elapsed_s.toFixed(0)}s / {limit_s.toFixed(0)}s</span>
					</div>
				{/if}
			</div>

			<!-- top-right tools -->
			<div class="tools">
				{#if has_renders}
					<button
						class="icon-btn glass"
						class:on={show_render}
						title={show_render ? "Hide render" : "Show render"}
						aria-pressed={show_render}
						onclick={() => {
							show_render = !show_render;
							requestAnimationFrame(redraw_current);
						}}
					>
						<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 2 7l10 5 10-5-10-5z" /><path d="m2 17 10 5 10-5" /><path d="m2 12 10 5 10-5" /></svg>
						<span class="lbl">{show_render ? "Hide render" : "Show render"}</span>
					</button>
				{/if}
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

			<!-- overlays -->
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
			{#if status === "running" && !focused}
				<div class="chip hint glass">Click to control</div>
			{/if}
		</div>

		<!-- bottom toolbar -->
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
				{#if gradio.shared.show_label && gradio.shared.label}
					<span class="label">{gradio.shared.label}</span>
				{/if}
			</div>
			<div class="hints" aria-hidden="true">
				<span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span>
				<span><kbd>Q</kbd><kbd>E</kbd>/<kbd>←</kbd><kbd>→</kbd> turn</span>
				<span><kbd>R</kbd><kbd>F</kbd> look</span>
				<span><kbd>Space</kbd><kbd>⇧</kbd> up/down</span>
				<span><kbd>↵</kbd> start <kbd>Esc</kbd> stop</span>
			</div>
			<button
				class="icon-btn plain"
				class:on={pad_open}
				title={pad_open ? "Hide control pad" : "Show control pad"}
				aria-pressed={pad_open}
				onclick={() => (pad_open = !pad_open)}
			>
				<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 11h4M8 9v4M15 12h.01M18 10h.01" /><rect x="2" y="6" width="20" height="12" rx="6" /></svg>
				<span class="lbl">Pad</span>
			</button>
		</div>

		{#if pad_open}
			<div class="pad" data-testid="worldviewer-pad">
				{#snippet pbtn(vk: VKey, label: string, title: string, cls: string)}
					<button
						class="pb {cls}"
						class:on={held.has(vk)}
						{title}
						aria-label={title}
						onpointerdown={(e) => pad_down(e, vk)}
						onpointerup={pad_up}
						onpointercancel={pad_up}
						onlostpointercapture={pad_up}
						oncontextmenu={(e) => e.preventDefault()}>{label}</button
					>
				{/snippet}
				<div class="cluster">
					<div class="cap">move</div>
					<div class="cross">
						{@render pbtn("fwd", "▲", "Forward (W)", "n")}
						{@render pbtn("left", "◀", "Strafe left (A)", "w")}
						{@render pbtn("right", "▶", "Strafe right (D)", "e")}
						{@render pbtn("back", "▼", "Backward (S)", "s")}
					</div>
				</div>
				<div class="cluster">
					<div class="cap">height</div>
					<div class="col">
						{@render pbtn("up", "↑", "Up (Space)", "")}
						{@render pbtn("down", "↓", "Down (Shift)", "")}
					</div>
				</div>
				<div class="cluster">
					<div class="cap">look</div>
					<div class="cross">
						{@render pbtn("pitchU", "⌃", "Look up (R)", "n")}
						{@render pbtn("yawL", "↶", "Turn left (Q / ←)", "w")}
						{@render pbtn("yawR", "↷", "Turn right (E / →)", "e")}
						{@render pbtn("pitchD", "⌄", "Look down (F)", "s")}
					</div>
				</div>
			</div>
		{/if}
	</div>
</Block>

<style>
	.wv {
		--wv-radius: 14px;
		--wv-accent: var(--color-accent, #f97316);
		--wv-glass: rgba(14, 16, 22, 0.55);
		--wv-glass-border: rgba(255, 255, 255, 0.12);
		--wv-fg: #f4f5f7;
		--wv-muted: rgba(244, 245, 247, 0.62);
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 10px;
		width: 100%;
		box-sizing: border-box;
		color: var(--body-text-color);
	}
	.wv.fullscreen {
		background: #07080b;
		height: 100vh;
		padding: 14px;
		color: var(--wv-fg);
	}

	/* ---------------------------------------------------------------- stage */
	.stage {
		position: relative;
		width: 100%;
		aspect-ratio: var(--ar);
		border-radius: var(--wv-radius);
		overflow: hidden;
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
	.fullscreen .stage {
		flex: 1 1 auto;
		aspect-ratio: auto;
		min-height: 0;
		background: #000;
	}
	.screen {
		display: block;
		width: 100%;
		height: 100%;
		object-fit: contain;
		image-rendering: auto;
	}
	.hidden {
		display: none !important;
	}
	.poster {
		position: absolute;
		inset: 0;
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

	.pip {
		position: absolute;
		left: 12px;
		bottom: 12px;
		width: 24%;
		min-width: 110px;
		max-width: 260px;
		height: auto;
		border-radius: 10px;
		border: 1px solid var(--wv-glass-border);
		box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
		background: #000;
		z-index: 3;
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
		padding: 8px 10px;
		border-radius: 10px;
		font-size: 11px;
		line-height: 1.35;
		color: var(--wv-fg);
		max-width: min(46%, 300px);
		display: flex;
		flex-direction: column;
		gap: 4px;
		pointer-events: none;
	}
	.hud-row {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 6px;
		align-items: center;
		color: var(--wv-muted);
	}
	.mono {
		font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
		font-variant-numeric: tabular-nums;
	}
	.stats .k {
		color: rgba(244, 245, 247, 0.45);
	}
	.status-row {
		color: var(--wv-fg);
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
		box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6);
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
	.hud-keys {
		display: flex;
		flex-direction: column;
		gap: 2px;
		margin-top: 2px;
	}
	.kr {
		display: flex;
		gap: 2px;
	}
	.kr:nth-child(2) {
		padding-left: 6px;
	}
	.mk {
		width: 17px;
		height: 16px;
		display: inline-grid;
		place-items: center;
		font-size: 9px;
		font-weight: 600;
		border-radius: 4px;
		background: rgba(255, 255, 255, 0.07);
		border: 1px solid rgba(255, 255, 255, 0.12);
		color: rgba(255, 255, 255, 0.55);
		transition:
			background 0.08s,
			color 0.08s;
	}
	.mk.wide {
		width: 26px;
	}
	.mk.wider {
		width: 48px;
	}
	.mk.on {
		background: var(--wv-accent);
		border-color: var(--wv-accent);
		color: #fff;
	}
	.progress {
		display: flex;
		align-items: center;
		gap: 6px;
		color: var(--wv-muted);
	}
	.progress .bar {
		flex: 1;
		min-width: 70px;
		height: 4px;
		border-radius: 2px;
		background: rgba(255, 255, 255, 0.14);
		overflow: hidden;
	}
	.progress .fill {
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
		transition:
			background 0.15s,
			border-color 0.15s;
	}
	.icon-btn:hover {
		background: rgba(30, 34, 44, 0.75);
	}
	.icon-btn.on {
		border-color: var(--wv-accent);
	}
	.icon-btn.plain {
		background: transparent;
		border: 1px solid var(--border-color-primary, rgba(127, 127, 127, 0.3));
		color: var(--body-text-color);
	}
	.icon-btn.plain.on {
		border-color: var(--wv-accent);
		color: var(--wv-accent);
	}

	/* ------------------------------------------------------------- overlays */
	.overlay {
		position: absolute;
		inset: 0;
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
		letter-spacing: 0.01em;
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
	.label {
		font-size: 13px;
		font-weight: 600;
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
	}
	.fullscreen .hints,
	.fullscreen .label {
		color: rgba(244, 245, 247, 0.6);
	}
	.fullscreen kbd {
		color: #eee;
		background: rgba(255, 255, 255, 0.08);
		border-color: rgba(255, 255, 255, 0.2);
	}
	.fullscreen .icon-btn.plain {
		color: #eee;
	}

	/* ------------------------------------------------------------------ pad */
	.pad {
		display: flex;
		justify-content: space-between;
		align-items: flex-end;
		gap: 12px;
		padding: 4px 2px 2px;
		touch-action: none;
		user-select: none;
		-webkit-user-select: none;
	}
	.cluster {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 4px;
	}
	.cap {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--body-text-color-subdued, rgba(127, 127, 127, 0.9));
	}
	.cross {
		display: grid;
		grid-template-columns: repeat(3, 48px);
		grid-template-rows: repeat(3, 44px);
		gap: 4px;
	}
	.cross :global(.pb.n) {
		grid-area: 1 / 2;
	}
	.cross :global(.pb.w) {
		grid-area: 2 / 1;
	}
	.cross :global(.pb.e) {
		grid-area: 2 / 3;
	}
	.cross :global(.pb.s) {
		grid-area: 3 / 2;
	}
	.col {
		display: grid;
		grid-template-rows: repeat(2, 66px);
		gap: 4px;
	}
	.col .pb {
		width: 48px;
	}
	.pb {
		display: grid;
		place-items: center;
		border-radius: 10px;
		font-size: 17px;
		cursor: pointer;
		color: var(--body-text-color);
		background: var(--background-fill-secondary, rgba(127, 127, 127, 0.12));
		border: 1px solid var(--border-color-primary, rgba(127, 127, 127, 0.3));
		border-bottom-width: 3px;
		touch-action: none;
		-webkit-tap-highlight-color: transparent;
		transition:
			background 0.08s,
			transform 0.05s;
	}
	.pb.on {
		background: var(--wv-accent);
		border-color: var(--wv-accent);
		color: #fff;
		transform: translateY(1px);
	}
	.fullscreen .pb {
		color: #eee;
		background: rgba(255, 255, 255, 0.08);
		border-color: rgba(255, 255, 255, 0.2);
	}

	/* --------------------------------------------------------------- mobile */
	@media (max-width: 640px) {
		.wv {
			padding: 6px;
		}
		.hud {
			font-size: 10px;
			padding: 6px 8px;
			max-width: 60%;
		}
		.hud-keys,
		.stats {
			display: none;
		}
		.hints {
			display: none;
		}
		.toolbar {
			justify-content: space-between;
		}
		.btn {
			padding: 0 16px;
		}
		.icon-btn .lbl {
			display: none;
		}
		.pad {
			gap: 6px;
		}
		.cross {
			grid-template-columns: repeat(3, 36px);
			grid-template-rows: repeat(3, 38px);
			gap: 3px;
		}
		.col {
			grid-template-rows: repeat(2, 58px);
			gap: 4px;
		}
		.col .pb {
			width: 38px;
		}
	}
</style>
