<script lang="ts">
	// Top-down "map" of the camera motion (Canvas 2D, no three.js).
	// Map frame = the source camera's ground plane: x = its right axis, up-on-screen = its forward axis,
	// height = its up axis (-y in OpenCV). The view auto-fits the source camera, the region ahead of it
	// and the whole path, and follows smoothly (exponential easing, no per-frame snapping).
	import { onDestroy, onMount } from "svelte";
	import type { CamStore } from "./types";

	let {
		store,
		visible = true,
		reset = 0,
		accent = "#f97316"
	}: { store: CamStore; visible?: boolean; reset?: number; accent?: string } = $props();

	type V3 = [number, number, number];
	let wrap: HTMLDivElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();
	let readout = $state("");
	let empty = $state(true);
	let raf = 0;
	let ro: ResizeObserver | null = null;
	let W = 300,
		H = 200,
		dpr = 1;

	// reference frame (from the first source camera, else the first path pose)
	let c0: V3 = [0, 0, 0],
		right0: V3 = [1, 0, 0],
		down0: V3 = [0, 1, 0],
		fwd0: V3 = [0, 0, 1];
	let has_ref = false;
	let depth = 1; // median scene depth ahead of the source camera
	let K = [500, 500, 416, 240, 832, 480];

	// context cloud in map coords, bucketed by (desaturated, quantized) colour for cheap drawing
	let cloud: { style: string; uv: Float32Array }[] = [];
	// path in map coords
	let path_u = new Float32Array(0),
		path_v = new Float32Array(0);

	// view (map units): current and target centre / scale (px per unit); user pan / zoom on top
	let view = { u: 0, v: 0, s: 60 };
	let target = { u: 0, v: 0, s: 60 };
	let user = { du: 0, dv: 0, zoom: 1 };
	let seen_scene = -1,
		seen_path = -1,
		seen_reset = 0;
	let dirty = true;
	let last_t = 0;
	let last_draw = 0;
	// cached cloud layer: re-rendered only when the zoom changes noticeably or the view pans far;
	// otherwise blitted with an offset/scale (keeps the main thread free for video playback)
	let cache: { c: HTMLCanvasElement; u: number; v: number; s: number; ox: number; oy: number } | null = null;

	const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
	const center_of = (p: number[]): V3 => [p[3], p[7], p[11]];
	const col = (p: number[], k: number): V3 => [p[k], p[4 + k], p[8 + k]];
	const to_map = (x: V3): [number, number, number] => {
		const d = sub(x, c0);
		return [dot(d, right0), dot(d, fwd0), -dot(d, down0)]; // u (right), v (forward), h (up)
	};

	function b64_bytes(b64: string): Uint8Array {
		const bin = atob(b64);
		const out = new Uint8Array(bin.length);
		for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
		return out;
	}

	function set_ref(): void {
		const src = store.scene?.source_poses?.[0] ?? store.poses[0] ?? null;
		has_ref = !!src;
		if (!src) return;
		c0 = center_of(src);
		right0 = col(src, 0);
		down0 = col(src, 1);
		fwd0 = col(src, 2);
	}

	function build_cloud(): void {
		cloud = [];
		const s = store.scene;
		if (s?.intrinsics?.length === 6) K = s.intrinsics.slice();
		set_ref();
		depth = 1;
		if (!s?.points) return;
		let xyz: Float32Array, cb: Uint8Array;
		try {
			const pb = b64_bytes(s.points);
			xyz = new Float32Array(pb.buffer, 0, Math.floor(pb.byteLength / 12) * 3);
			cb = b64_bytes(s.colors);
		} catch {
			return;
		}
		const n = Math.min(xyz.length / 3, Math.floor(cb.length / 3));
		const step = Math.max(1, Math.floor(n / 12000));
		const buckets = new Map<string, number[]>();
		const zs: number[] = [];
		for (let i = 0; i < n; i += step) {
			const p: V3 = [xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]];
			if (!Number.isFinite(p[0] + p[1] + p[2])) continue;
			const [u, v] = to_map(p);
			if (v > 0) zs.push(v);
			// desaturate towards grey and quantize (4 levels / channel -> <= 64 fill styles)
			const r = cb[i * 3],
				g = cb[i * 3 + 1],
				b = cb[i * 3 + 2];
			const l = 0.3 * r + 0.59 * g + 0.11 * b;
			const q = (c: number) => Math.round(((0.45 * c + 0.55 * l) / 255) * 3) * 85;
			const key = `rgb(${q(r)},${q(g)},${q(b)})`;
			let arr = buckets.get(key);
			if (!arr) buckets.set(key, (arr = []));
			arr.push(u, v);
		}
		cloud = [...buckets].map(([style, a]) => ({ style, uv: new Float32Array(a) }));
		if (zs.length) {
			zs.sort((a, b) => a - b);
			depth = Math.max(1e-3, zs[Math.floor(zs.length / 2)]);
		}
	}

	function build_path(): void {
		if (!has_ref) set_ref();
		const n = store.poses.length;
		path_u = new Float32Array(n);
		path_v = new Float32Array(n);
		for (let i = 0; i < n; i++) {
			const [u, v] = to_map(center_of(store.poses[i]));
			path_u[i] = u;
			path_v[i] = v;
		}
	}

	/** Fit: source camera, the region ahead of it, and the whole path (played + buffered), padded. */
	function compute_target(): void {
		let u0 = -0.35 * depth,
			u1 = 0.35 * depth,
			v0 = -0.15 * depth,
			v1 = 0.6 * depth;
		for (let i = 0; i < path_u.length; i++) {
			u0 = Math.min(u0, path_u[i]);
			u1 = Math.max(u1, path_u[i]);
			v0 = Math.min(v0, path_v[i]);
			v1 = Math.max(v1, path_v[i]);
		}
		const pad = 0.18 * Math.max(u1 - u0, v1 - v0) + 0.05 * depth;
		u0 -= pad;
		u1 += pad;
		v0 -= pad;
		v1 += pad;
		const s = Math.min((W - 24) / (u1 - u0), (H - 44) / (v1 - v0));
		target = { u: (u0 + u1) / 2, v: (v0 + v1) / 2, s: Math.max(1e-6, s) };
	}

	function resize(): void {
		if (!wrap || !canvas) return;
		dpr = Math.min(2, window.devicePixelRatio || 1);
		W = Math.max(120, wrap.clientWidth);
		H = Math.max(120, wrap.clientHeight);
		canvas.width = Math.round(W * dpr);
		canvas.height = Math.round(H * dpr);
		compute_target();
		cache = null;
		dirty = true;
	}

	function nice_step(px_per_unit: number): number {
		const raw = 60 / px_per_unit;
		const p = 10 ** Math.floor(Math.log10(raw));
		for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
		return 10 * p;
	}

	function draw(): void {
		if (!canvas) return;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;
		const s = view.s * user.zoom;
		const cu = view.u + user.du,
			cv = view.v + user.dv;
		const X = (u: number) => W / 2 + (u - cu) * s;
		const Y = (v: number) => H / 2 + 8 - (v - cv) * s; // forward = up on screen
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, W, H);

		// grid (+ scale bar)
		const step = nice_step(s);
		ctx.strokeStyle = "rgba(255,255,255,0.05)";
		ctx.lineWidth = 1;
		ctx.beginPath();
		const ua = Math.floor((cu - W / 2 / s) / step) * step;
		for (let u = ua; X(u) < W; u += step) {
			const x = Math.round(X(u)) + 0.5;
			ctx.moveTo(x, 0);
			ctx.lineTo(x, H);
		}
		const va = Math.floor((cv - H / 2 / s) / step) * step;
		for (let v = va; Y(v) > 0; v += step) {
			const y = Math.round(Y(v)) + 0.5;
			ctx.moveTo(0, y);
			ctx.lineTo(W, y);
		}
		ctx.stroke();

		// context cloud: faint, desaturated (cached layer)
		if (cloud.length) {
			const stale =
				!cache ||
				Math.abs(Math.log(s / cache.s)) > 0.04 ||
				Math.abs((cu - cache.u) * s) > W * 0.2 ||
				Math.abs((cv - cache.v) * s) > H * 0.2;
			if (stale) cache = render_cloud(cu, cv, s);
			const r = s / cache!.s;
			// cache pixel (px, py) shows map point u = cache.u + (px - ox) / cache.s
			const dx = W / 2 + (cache!.u - cu) * s - cache!.ox * r;
			const dy = H / 2 + 8 - (cache!.v - cv) * s - cache!.oy * r;
			ctx.globalAlpha = 0.3;
			ctx.drawImage(cache!.c, dx, dy, (cache!.c.width / dpr) * r, (cache!.c.height / dpr) * r);
			ctx.globalAlpha = 1;
		}

		// source camera: faint grey wedge at the origin
		if (has_ref) wedge(ctx, X(0), Y(0), 0, 26, "rgba(203,213,225,0.55)", "rgba(203,213,225,0.10)", 1.2);

		const n = path_u.length;
		const played = Math.min(store.played, n);
		// upcoming (buffered) trace: lighter, dashed
		if (n - Math.max(0, played - 1) >= 2) {
			ctx.strokeStyle = accent;
			ctx.globalAlpha = 0.45;
			ctx.lineWidth = 3;
			ctx.setLineDash([6, 5]);
			ctx.lineCap = "round";
			ctx.beginPath();
			for (let i = Math.max(0, played - 1); i < n; i++) {
				const x = X(path_u[i]),
					y = Y(path_v[i]);
				i === Math.max(0, played - 1) ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
			}
			ctx.stroke();
			ctx.setLineDash([]);
			ctx.globalAlpha = 1;
		}
		// played trace: thick solid
		if (played >= 2) {
			ctx.strokeStyle = accent;
			ctx.lineWidth = 4;
			ctx.lineCap = "round";
			ctx.lineJoin = "round";
			ctx.beginPath();
			for (let i = 0; i < played; i++) {
				const x = X(path_u[i]),
					y = Y(path_v[i]);
				i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
			}
			ctx.stroke();
		}
		// dots: one per block (12 frames), bigger; played solid, upcoming faded & hollow
		for (let i = 0; i < n; i += 12) {
			const x = X(path_u[i]),
				y = Y(path_v[i]);
			ctx.beginPath();
			ctx.arc(x, y, 5, 0, Math.PI * 2);
			if (i < played) {
				ctx.fillStyle = accent;
				ctx.fill();
			} else {
				ctx.fillStyle = "rgba(14,17,23,0.9)";
				ctx.fill();
				ctx.globalAlpha = 0.55;
				ctx.strokeStyle = accent;
				ctx.lineWidth = 2;
				ctx.stroke();
				ctx.globalAlpha = 1;
			}
		}
		// where it is going: end of the buffered path
		if (n > played && n > 0) {
			const x = X(path_u[n - 1]),
				y = Y(path_v[n - 1]);
			ctx.globalAlpha = 0.7;
			ctx.strokeStyle = accent;
			ctx.lineWidth = 2;
			ctx.beginPath();
			ctx.arc(x, y, 8, 0, Math.PI * 2);
			ctx.stroke();
			ctx.globalAlpha = 1;
		}

		// start marker
		if (has_ref) {
			const x = n ? X(path_u[0]) : X(0),
				y = n ? Y(path_v[0]) : Y(0);
			ctx.beginPath();
			ctx.arc(x, y, 7, 0, Math.PI * 2);
			ctx.fillStyle = "#0e1117";
			ctx.fill();
			ctx.lineWidth = 2.5;
			ctx.strokeStyle = "#f4f5f7";
			ctx.stroke();
			ctx.beginPath();
			ctx.arc(x, y, 2.5, 0, Math.PI * 2);
			ctx.fillStyle = "#f4f5f7";
			ctx.fill();
			label(ctx, "start", x + 11, y + 4, "#f4f5f7");
		}

		// current camera: one bold heading wedge (FOV) + dot
		let text = "";
		if (played > 0) {
			const p = store.poses[played - 1];
			const f = col(p, 2);
			const hx = dot(f, right0),
				hy = dot(f, fwd0);
			const heading = Math.atan2(hx, hy); // 0 = source forward, + = to the right
			const x = X(path_u[played - 1]),
				y = Y(path_v[played - 1]);
			wedge(ctx, x, y, heading, 44, accent, "rgba(249,115,22,0.28)", 2.5);
			ctx.beginPath();
			ctx.arc(x, y, 6, 0, Math.PI * 2);
			ctx.fillStyle = accent;
			ctx.fill();
			ctx.lineWidth = 2;
			ctx.strokeStyle = "#fff";
			ctx.stroke();
			const [, , h] = to_map(center_of(p));
			const pitch = Math.asin(Math.max(-1, Math.min(1, -dot(f, down0))));
			const deg = (r: number) => `${r >= 0 ? "+" : "−"}${Math.abs((r * 180) / Math.PI).toFixed(0)}°`;
			text = `heading ${deg(heading)} · pitch ${deg(pitch)} · height ${h >= 0 ? "+" : "−"}${Math.abs(h).toFixed(2)}`;
		}
		if (text !== readout) readout = text;

		// north arrow ("forward" of the source camera) + scale bar
		ctx.fillStyle = "rgba(244,245,247,0.45)";
		ctx.font = "9px ui-monospace, monospace";
		const bar = step * s;
		ctx.fillRect(W - 12 - bar, H - 12, bar, 2);
		ctx.textAlign = "right";
		ctx.fillText(`${+step.toPrecision(2)} u`, W - 12, H - 16);
		ctx.textAlign = "left";
	}

	function render_cloud(cu: number, cv: number, s: number) {
		const mw = Math.round(W * 1.6),
			mh = Math.round(H * 1.6);
		const c = cache?.c ?? document.createElement("canvas");
		c.width = Math.round(mw * dpr);
		c.height = Math.round(mh * dpr);
		const g = c.getContext("2d")!;
		g.setTransform(dpr, 0, 0, dpr, 0, 0);
		g.clearRect(0, 0, mw, mh);
		const ox = mw / 2,
			oy = mh / 2 + 8;
		const ps = Math.max(1, Math.min(2.2, s * depth * 0.004));
		for (const b of cloud) {
			g.fillStyle = b.style;
			g.beginPath();
			const a = b.uv;
			for (let i = 0; i < a.length; i += 2) {
				const x = ox + (a[i] - cu) * s,
					y = oy - (a[i + 1] - cv) * s;
				if (x < -2 || y < -2 || x > mw + 2 || y > mh + 2) continue;
				g.rect(x - ps / 2, y - ps / 2, ps, ps);
			}
			g.fill();
		}
		return { c, u: cu, v: cv, s, ox, oy };
	}

	function label(ctx: CanvasRenderingContext2D, t: string, x: number, y: number, color: string): void {
		ctx.font = "700 10px ui-sans-serif, system-ui, sans-serif";
		const w = ctx.measureText(t).width;
		ctx.fillStyle = "rgba(14,17,23,0.8)";
		ctx.fillRect(x - 3, y - 10, w + 6, 14);
		ctx.fillStyle = color;
		ctx.fillText(t, x, y);
	}

	/** FOV wedge at (x, y) pointing at `heading` (radians, 0 = up on screen, + = clockwise). */
	function wedge(
		ctx: CanvasRenderingContext2D,
		x: number,
		y: number,
		heading: number,
		len: number,
		stroke: string,
		fill: string,
		lw: number
	): void {
		const half = Math.atan2(K[4] / 2, K[0]);
		const a0 = heading - half - Math.PI / 2,
			a1 = heading + half - Math.PI / 2;
		ctx.beginPath();
		ctx.moveTo(x, y);
		ctx.arc(x, y, len, a0, a1);
		ctx.closePath();
		ctx.fillStyle = fill;
		ctx.fill();
		ctx.lineWidth = lw;
		ctx.lineJoin = "round";
		ctx.strokeStyle = stroke;
		ctx.stroke();
	}

	function tick(t: number): void {
		raf = requestAnimationFrame(tick);
		const dt = last_t ? Math.min(100, t - last_t) : 16;
		last_t = t;
		if (!visible || !canvas) return;
		if (store.scene_version !== seen_scene) {
			seen_scene = store.scene_version;
			build_cloud();
			cache = null;
			build_path();
			compute_target();
			view = { ...target };
			dirty = true;
		}
		if (store.version !== seen_path) {
			seen_path = store.version;
			if (store.poses.length === 0 || !has_ref) set_ref();
			build_path();
			compute_target();
			if (store.poses.length === 0) view = { ...target };
			dirty = true;
		}
		if (reset !== seen_reset) {
			seen_reset = reset;
			user = { du: 0, dv: 0, zoom: 1 };
			dirty = true;
		}
		// smooth follow (time-based easing; ~250 ms time constant)
		const k = 1 - Math.exp(-dt / 250);
		const du = target.u - view.u,
			dv = target.v - view.v,
			ds = Math.log(target.s / view.s);
		if (Math.abs(du) * view.s > 0.2 || Math.abs(dv) * view.s > 0.2 || Math.abs(ds) > 0.002) {
			view.u += du * k;
			view.v += dv * k;
			view.s *= Math.exp(ds * k);
			dirty = true;
		}
		const e = !has_ref && store.poses.length === 0;
		if (e !== empty) empty = e;
		if (dirty && t - last_draw >= 33) {
			dirty = false;
			last_draw = t;
			draw();
		}
	}

	// user pan (drag) / zoom (wheel) on top of the auto-follow; "Reset view" clears it
	let drag: { x: number; y: number } | null = null;
	function on_down(e: PointerEvent): void {
		drag = { x: e.clientX, y: e.clientY };
		(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
	}
	function on_move(e: PointerEvent): void {
		if (!drag) return;
		const s = view.s * user.zoom;
		user.du -= (e.clientX - drag.x) / s;
		user.dv += (e.clientY - drag.y) / s;
		drag = { x: e.clientX, y: e.clientY };
		dirty = true;
	}
	function on_up(): void {
		drag = null;
	}
	function on_wheel(e: WheelEvent): void {
		e.preventDefault();
		user.zoom = Math.min(20, Math.max(0.2, user.zoom * Math.exp(-e.deltaY * 0.0015)));
		dirty = true;
	}

	onMount(() => {
		ro = new ResizeObserver(resize);
		if (wrap) ro.observe(wrap);
		canvas?.addEventListener("wheel", on_wheel, { passive: false });
		resize();
		raf = requestAnimationFrame(tick);
	});
	onDestroy(() => {
		cancelAnimationFrame(raf);
		ro?.disconnect();
		canvas?.removeEventListener("wheel", on_wheel);
	});
	$effect(() => {
		if (visible) {
			dirty = true;
			requestAnimationFrame(resize);
		}
	});
</script>

<div class="map" bind:this={wrap}>
	<canvas
		bind:this={canvas}
		data-testid="worldviewer-map"
		onpointerdown={on_down}
		onpointermove={on_move}
		onpointerup={on_up}
		onpointercancel={on_up}
	></canvas>
	<div class="compass" aria-hidden="true">▲ ahead</div>
	{#if readout}<div class="readout mono">{readout}</div>{/if}
	{#if empty}
		<div class="note">The map of your camera path appears here once a session starts.</div>
	{/if}
</div>

<style>
	.map {
		position: absolute;
		inset: 0;
		touch-action: none;
		cursor: grab;
	}
	.map:active {
		cursor: grabbing;
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
	}
	.compass {
		position: absolute;
		top: 6px;
		left: 8px;
		font-size: 9.5px;
		letter-spacing: 0.06em;
		color: rgba(244, 245, 247, 0.45);
		pointer-events: none;
	}
	.readout {
		position: absolute;
		top: 5px;
		right: 8px;
		font-size: 10px;
		padding: 2px 7px;
		border-radius: 6px;
		color: rgba(244, 245, 247, 0.85);
		background: rgba(11, 13, 18, 0.7);
		pointer-events: none;
		font-family: var(--font-mono, ui-monospace, monospace);
		font-variant-numeric: tabular-nums;
	}
	.note {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		text-align: center;
		padding: 12px;
		font-size: 12px;
		color: rgba(244, 245, 247, 0.5);
		pointer-events: none;
	}
</style>
