<script lang="ts">
	// Single-lane input/latency timeline.
	//  - "You" cursor (~85 % of the width) is now: key presses grow there and scroll left.
	//  - "Model" cursor sits left of it by the measured latency: an input reaching it is on screen.
	//  - A slim strip under the bars shows the actions of the frames actually displayed (left of the
	//    Model cursor) and of the frames still buffered (right of it, dimmed).
	// Frozen (no scrolling) whenever `running` is false.
	import { onDestroy, onMount } from "svelte";
	import type { Axis, TimelineStore, TlFrame } from "./types";

	let {
		store,
		seconds = 10,
		running = false,
		default_latency_ms = 2000
	}: {
		store: TimelineStore;
		seconds?: number;
		running?: boolean;
		default_latency_ms?: number;
	} = $props();

	const AXES: { axis: Axis; color: string; pos: string; neg: string }[] = [
		{ axis: "forward", color: "#f97316", pos: "W", neg: "S" },
		{ axis: "right", color: "#22d3ee", pos: "D", neg: "A" },
		{ axis: "yaw", color: "#a78bfa", pos: "E", neg: "Q" },
		{ axis: "pitch", color: "#4ade80", pos: "R", neg: "F" },
		{ axis: "up", color: "#facc15", pos: "␣", neg: "⇧" }
	];
	const ROW = new Map(AXES.map((a, i) => [a.axis, i]));

	let wrap: HTMLDivElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();
	let raf = 0;
	let ro: ResizeObserver | null = null;
	let W = 600,
		H = 120,
		dpr = 1;
	let hatch: CanvasPattern | null = null;
	let frozen_now: number | null = null; // time the view froze at (null = never ran)
	let dirty = true;
	let seen_session = -1;
	let was_running = false;

	const PAD = 10;
	const YOU_FRAC = 0.8;
	const TOP = 22; // cursor labels + latency bracket
	const ROW_H = 9;
	const ROW_GAP = 2;
	const BARS_Y = TOP;
	const BARS_H = AXES.length * (ROW_H + ROW_GAP) - ROW_GAP;
	const STRIP_ROW = 3;
	const STRIP_Y = BARS_Y + BARS_H + 7;
	const STRIP_H = AXES.length * (STRIP_ROW + 1) - 1;
	const AXIS_Y = STRIP_Y + STRIP_H + 13;

	function resize(): void {
		if (!wrap || !canvas) return;
		dpr = Math.min(2, window.devicePixelRatio || 1);
		W = Math.max(240, wrap.clientWidth);
		H = AXIS_Y + 6;
		canvas.width = Math.round(W * dpr);
		canvas.height = Math.round(H * dpr);
		canvas.style.height = `${H}px`;
		hatch = null;
		dirty = true;
	}

	function make_hatch(ctx: CanvasRenderingContext2D): CanvasPattern | null {
		const c = document.createElement("canvas");
		c.width = c.height = Math.round(6 * dpr);
		const g = c.getContext("2d")!;
		g.strokeStyle = "rgba(255,255,255,0.4)";
		g.lineWidth = 1.2 * dpr;
		g.beginPath();
		g.moveTo(0, c.height);
		g.lineTo(c.width, 0);
		g.stroke();
		return ctx.createPattern(c, "repeat");
	}

	function fill_hatch(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, alpha: number): void {
		if (!hatch) hatch = make_hatch(ctx);
		ctx.globalAlpha = alpha;
		ctx.fillStyle = "rgba(255,255,255,0.06)";
		ctx.fillRect(x, y, w, h);
		if (hatch) {
			ctx.fillStyle = hatch;
			ctx.fillRect(x, y, w, h);
		}
		ctx.globalAlpha = 1;
	}

	/** Draw one frame's action as bars (rows of height rh) between xa and xb. */
	function draw_action(
		ctx: CanvasRenderingContext2D,
		f: TlFrame,
		xa: number,
		xb: number,
		y0: number,
		rh: number,
		gap: number,
		alpha: number
	): void {
		const w = xb - xa;
		if (w <= 0) return;
		if (f.has_action && !f.action) {
			fill_hatch(ctx, xa, y0, w, AXES.length * (rh + gap) - gap, alpha);
			return;
		}
		if (!f.action) return;
		for (const a of AXES) {
			const v = f.action[a.axis];
			if (!v) continue;
			const y = y0 + (ROW.get(a.axis) ?? 0) * (rh + gap);
			ctx.globalAlpha = alpha * (v > 0 ? 0.35 + 0.65 * Math.min(1, v) : 0.3 + 0.35 * Math.min(1, -v));
			ctx.fillStyle = a.color;
			ctx.fillRect(xa, y, w, rh);
		}
		ctx.globalAlpha = 1;
	}

	function prune(t_min: number): void {
		let k = 0;
		while (k < store.you.length && (store.you[k].t1 ?? Infinity) < t_min) k++;
		if (k) store.you.splice(0, k);
		k = 0;
		while (k < store.frames.length && store.frames[k].t + store.frames[k].dur < t_min) k++;
		if (k) store.frames.splice(0, k);
	}

	function draw(now: number, active: boolean): void {
		if (!canvas) return;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;
		const span = seconds * 1000;
		const X0 = PAD,
			X1 = W - PAD;
		const pw = X1 - X0;
		const scale = pw / span;
		const x_you = X0 + YOU_FRAC * pw;
		const sched = store.sched;
		const measured = store.latency_ms != null && Number.isFinite(store.latency_ms);
		// scheduled mode: fixed delay (grows only on stalls); legacy: measured lag or an estimate
		const lat = Math.min(
			sched ? sched.delay_ms : measured ? store.latency_ms! : default_latency_ms,
			0.85 * YOU_FRAC * span
		);
		const x_model = x_you - lat * scale;
		const x_in = (t: number) => x_you - (now - t) * scale; // input timeline
		const x_disp = (t: number) => x_model - (now - t) * scale; // display timeline
		if (active) prune(now - span * 1.2);

		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, W, H);

		// regions: played | in flight | future
		const lane_y = BARS_Y - 3,
			lane_h = STRIP_Y + STRIP_H + 3 - lane_y;
		ctx.fillStyle = "rgba(255,255,255,0.035)";
		ctx.fillRect(X0, lane_y, x_model - X0, lane_h);
		ctx.fillStyle = "rgba(249,115,22,0.07)";
		ctx.fillRect(x_model, lane_y, x_you - x_model, lane_h);
		ctx.fillStyle = "rgba(255,255,255,0.015)";
		ctx.fillRect(x_you, lane_y, X1 - x_you, lane_h);

		// second grid + axis labels (relative to now)
		ctx.strokeStyle = "rgba(255,255,255,0.06)";
		ctx.lineWidth = 1;
		ctx.font = "9px ui-monospace, monospace";
		ctx.fillStyle = "rgba(244,245,247,0.4)";
		ctx.textAlign = "center";
		for (let k = -Math.floor(YOU_FRAC * seconds); k <= Math.floor((1 - YOU_FRAC) * seconds); k++) {
			const x = Math.round(x_you + k * 1000 * scale) + 0.5;
			ctx.beginPath();
			ctx.moveTo(x, lane_y);
			ctx.lineTo(x, lane_y + lane_h);
			ctx.stroke();
			if (k !== 0) ctx.fillText(`${k > 0 ? "+" : ""}${k}s`, x, AXIS_Y);
		}
		ctx.textAlign = "left";

		if (active || frozen_now != null) {
			// frames as [frame, x0, x1, played]. Scheduled mode: frame f sits at the time its action
			// was sampled (base0 + f/fps) on the input axis, so it lines up with the key press that
			// caused it and crosses the Model cursor exactly when it is displayed. Legacy mode: played
			// frames at display time (relative to the Model cursor), buffered frames queued after it.
			const items: [TlFrame, number, number, boolean][] = [];
			const sx = (f: TlFrame) => x_in(sched!.base0 + (f.index! * 1000) / sched!.fps);
			for (const f of store.frames) {
				const dur = f.dur * scale;
				const xa = sched && f.index != null ? sx(f) : x_disp(f.t);
				if (xa + dur >= X0) items.push([f, xa, xa + dur, true]);
			}
			let cum = 0;
			for (const f of store.queue) {
				const dur = 1000 / (f.fps || 15);
				const xa = sched && f.index != null ? sx(f) : x_model + cum * scale;
				cum += dur;
				if (xa > X1) break;
				items.push([f, xa, Math.min(X1, xa + dur * scale), false]);
			}
			for (const [f, xa, xb, played] of items) {
				draw_action(ctx, f, Math.max(X0, xa), xb + 0.5, STRIP_Y, STRIP_ROW, 1, played ? 1 : 0.4);
				if (f.first_in_chunk && xa >= X0) {
					ctx.fillStyle = played ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.35)";
					ctx.fillRect(Math.round(xa), STRIP_Y - 3, 1, STRIP_H + 6);
				}
			}

			// key-press bars; with no keyboard input in view (presets / autopilot), mirror the
			// frames' own actions into the bar rows so upcoming scripted moves stay readable
			const t_left = now - (x_you - X0) / scale;
			const has_keys = store.you.some((s) => (s.t1 ?? now) >= t_left);
			if (has_keys) {
				for (const s of store.you) {
					const t1 = s.t1 == null ? now : Math.min(s.t1, now);
					if (t1 <= s.t0) continue;
					const xa = Math.max(X0, x_in(s.t0));
					const xb = Math.min(x_you, x_in(t1));
					if (xb <= X0) continue;
					const a = AXES[ROW.get(s.axis) ?? 0];
					const y = BARS_Y + (ROW.get(s.axis) ?? 0) * (ROW_H + ROW_GAP);
					const mag = Math.min(1, Math.abs(s.val));
					for (const [pa, pb, dim] of [
						[xa, Math.min(xb, x_model), 1],
						[Math.max(xa, x_model), xb, 0.45]
					] as [number, number, number][]) {
						const w = pb - pa;
						if (w <= 0) continue;
						ctx.globalAlpha = dim * (0.35 + 0.65 * mag);
						if (s.val > 0) {
							ctx.fillStyle = a.color;
							ctx.fillRect(pa, y, Math.max(1.5, w), ROW_H);
						} else {
							ctx.fillStyle = a.color + "40";
							ctx.fillRect(pa, y, Math.max(1.5, w), ROW_H);
							ctx.strokeStyle = a.color;
							ctx.strokeRect(pa + 0.5, y + 0.5, Math.max(1, w - 1), ROW_H - 1);
						}
					}
					ctx.globalAlpha = 1;
					if (xb - xa > 11) {
						ctx.fillStyle = s.val > 0 ? "rgba(0,0,0,0.75)" : a.color;
						ctx.font = "700 8px ui-monospace, monospace";
						ctx.fillText(s.val > 0 ? a.pos : a.neg, xa + 2.5, y + ROW_H - 1.5);
					}
				}
			} else {
				for (const [f, xa, xb, played] of items)
					draw_action(ctx, f, Math.max(X0, xa), xb + 0.5, BARS_Y, ROW_H, ROW_GAP, played ? 0.9 : 0.35);
				if (store.frames.some((f) => f.has_action && !f.action) || store.queue.some((f) => f.has_action && !f.action)) {
					ctx.fillStyle = "rgba(244,245,247,0.75)";
					ctx.font = "600 9px ui-sans-serif, system-ui, sans-serif";
					ctx.fillText("autopilot", X0 + 6, BARS_Y + BARS_H / 2 + 3);
				}
			}
		} else {
			ctx.fillStyle = "rgba(244,245,247,0.4)";
			ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
			ctx.textAlign = "center";
			ctx.fillText("Press Start — your key presses and the model's response appear here", (X0 + X1) / 2, BARS_Y + BARS_H / 2 + 4);
			ctx.textAlign = "left";
		}

		// cursors
		ctx.lineWidth = 1.5;
		ctx.strokeStyle = "rgba(244,245,247,0.85)";
		ctx.beginPath();
		ctx.moveTo(Math.round(x_model) + 0.5, TOP - 6);
		ctx.lineTo(Math.round(x_model) + 0.5, lane_y + lane_h + 2);
		ctx.stroke();
		ctx.strokeStyle = "#f97316";
		ctx.beginPath();
		ctx.moveTo(Math.round(x_you) + 0.5, TOP - 6);
		ctx.lineTo(Math.round(x_you) + 0.5, lane_y + lane_h + 2);
		ctx.stroke();
		ctx.lineWidth = 1;

		// labels + latency bracket above the lane
		ctx.font = "700 9.5px ui-sans-serif, system-ui, sans-serif";
		ctx.fillStyle = "#f97316";
		ctx.fillText("You", x_you + 4, 10);
		ctx.fillStyle = "rgba(244,245,247,0.9)";
		const mw = ctx.measureText("Model").width;
		ctx.fillText("Model", x_model - mw - 4, 10);
		const by = 13;
		ctx.strokeStyle = "rgba(244,245,247,0.55)";
		ctx.beginPath();
		ctx.moveTo(x_model + 2, by);
		ctx.lineTo(x_you - 2, by);
		ctx.stroke();
		const label = sched
			? `${(lat / 1000).toFixed(1)} s`
			: measured
				? `latency ${(lat / 1000).toFixed(2)} s`
				: `latency ≈${(lat / 1000).toFixed(1)} s (estimate)`;
		ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
		const tw = ctx.measureText(label).width;
		const lx = Math.max(x_model + 4, (x_model + x_you) / 2 - tw / 2);
		ctx.fillStyle = "#0e1117";
		ctx.fillRect(lx - 5, by - 8, tw + 10, 14);
		ctx.fillStyle = sched || measured ? "#fdba74" : "rgba(244,245,247,0.6)";
		ctx.fillText(label, lx, by + 3.5);
		ctx.fillStyle = "rgba(244,245,247,0.4)";
		ctx.font = "9px ui-monospace, monospace";
		ctx.textAlign = "center";
		ctx.fillText("now", x_you, AXIS_Y);
		ctx.textAlign = "left";
	}

	function frame(): void {
		raf = requestAnimationFrame(frame);
		if (store.session !== seen_session) {
			seen_session = store.session;
			frozen_now = null;
			dirty = true;
		}
		if (running) {
			const now = performance.now();
			frozen_now = now;
			was_running = true;
			draw(now, true);
		} else if (dirty || was_running) {
			was_running = false;
			dirty = false;
			draw(frozen_now ?? performance.now(), false);
		}
	}

	onMount(() => {
		ro = new ResizeObserver(resize);
		if (wrap) ro.observe(wrap);
		resize();
		raf = requestAnimationFrame(frame);
	});
	onDestroy(() => {
		cancelAnimationFrame(raf);
		ro?.disconnect();
	});
</script>

<div class="tl" bind:this={wrap}>
	<canvas bind:this={canvas} data-testid="worldviewer-timeline" data-running={running ? "1" : "0"}></canvas>
	<div class="legend">
		{#each AXES as a (a.axis)}
			<span><i style:background={a.color}></i>{a.pos}/{a.neg} {a.axis}</span>
		{/each}
		<span><i class="hatch"></i>autopilot</span>
		<span><i class="strip"></i>displayed frames</span>
		<span><i class="blk"></i>block</span>
	</div>
</div>

<style>
	.tl {
		width: 100%;
	}
	canvas {
		display: block;
		width: 100%;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
		padding: 4px 10px 0;
		font-size: 10px;
		color: rgba(244, 245, 247, 0.55);
	}
	.legend span {
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.legend i {
		display: inline-block;
		width: 10px;
		height: 6px;
		border-radius: 2px;
	}
	.legend i.hatch {
		background: repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.4) 0 1.5px, transparent 1.5px 4px);
	}
	.legend i.strip {
		height: 3px;
		background: linear-gradient(90deg, #f97316 0 50%, #a78bfa 50%);
	}
	.legend i.blk {
		width: 1px;
		height: 10px;
		background: rgba(255, 255, 255, 0.7);
		border-radius: 0;
	}
</style>
