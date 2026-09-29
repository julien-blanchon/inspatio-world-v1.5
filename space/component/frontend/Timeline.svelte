<script lang="ts">
	import { onDestroy, onMount } from "svelte";
	import type { Axis, Seg, TimelineStore } from "./types";

	let {
		store,
		seconds = 8
	}: { store: TimelineStore; seconds?: number } = $props();

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
		H = 148,
		dpr = 1;
	let hatch: CanvasPattern | null = null;

	const GUTTER = 58;
	const ROW_H = 8;
	const ROW_GAP = 2;
	const LANE_H = AXES.length * (ROW_H + ROW_GAP) - ROW_GAP;
	const YOU_Y = 10;
	const MID_Y = YOU_Y + LANE_H + 10;
	const MID_H = 22;
	const PLAY_Y = MID_Y + MID_H + 4;
	const AXIS_Y = PLAY_Y + LANE_H + 8;

	function resize(): void {
		if (!wrap || !canvas) return;
		dpr = Math.min(2, window.devicePixelRatio || 1);
		W = Math.max(200, wrap.clientWidth);
		H = AXIS_Y + 16;
		canvas.width = Math.round(W * dpr);
		canvas.height = Math.round(H * dpr);
		canvas.style.height = `${H}px`;
		hatch = null;
	}

	function make_hatch(ctx: CanvasRenderingContext2D): CanvasPattern | null {
		const c = document.createElement("canvas");
		c.width = c.height = 8 * dpr;
		const g = c.getContext("2d")!;
		g.strokeStyle = "rgba(255,255,255,0.28)";
		g.lineWidth = 1.5 * dpr;
		g.beginPath();
		g.moveTo(0, 8 * dpr);
		g.lineTo(8 * dpr, 0);
		g.stroke();
		return ctx.createPattern(c, "repeat");
	}

	function prune(arr: { t1?: number | null }[] | number[], t_min: number): void {
		// arrays are append-ordered; drop from the front while fully out of the window
		let k = 0;
		while (k < arr.length) {
			const e: any = arr[k];
			const end = typeof e === "number" ? e : (e.t1 ?? Infinity);
			if (end >= t_min) break;
			k++;
		}
		if (k > 0) arr.splice(0, k);
	}

	function draw_segs(
		ctx: CanvasRenderingContext2D,
		segs: Seg[],
		y0: number,
		now: number,
		x_of: (t: number) => number
	): void {
		for (const s of segs) {
			const t1 = s.t1 == null ? now : Math.min(s.t1, now);
			if (t1 <= s.t0) continue;
			const x0 = Math.max(GUTTER, x_of(s.t0));
			const x1 = x_of(t1);
			if (x1 <= GUTTER) continue;
			const w = Math.max(1.5, x1 - x0);
			if (s.axis === "auto") {
				if (!hatch) hatch = make_hatch(ctx);
				ctx.fillStyle = "rgba(255,255,255,0.05)";
				ctx.fillRect(x0, y0, w, LANE_H);
				if (hatch) {
					ctx.fillStyle = hatch;
					ctx.fillRect(x0, y0, w, LANE_H);
				}
				if (w > 60) {
					ctx.fillStyle = "rgba(255,255,255,0.6)";
					ctx.font = "600 9px ui-sans-serif, system-ui, sans-serif";
					ctx.fillText("autopilot", x0 + 5, y0 + LANE_H / 2 + 3);
				}
				continue;
			}
			const row = ROW.get(s.axis) ?? 0;
			const a = AXES[row];
			const y = y0 + row * (ROW_H + ROW_GAP);
			const mag = Math.min(1, Math.abs(s.val));
			ctx.globalAlpha = 0.35 + 0.65 * mag;
			if (s.val > 0) {
				ctx.fillStyle = a.color;
				ctx.fillRect(x0, y, w, ROW_H);
			} else {
				ctx.fillStyle = a.color + "40";
				ctx.fillRect(x0, y, w, ROW_H);
				ctx.strokeStyle = a.color;
				ctx.lineWidth = 1;
				ctx.strokeRect(x0 + 0.5, y + 0.5, w - 1, ROW_H - 1);
			}
			ctx.globalAlpha = 1;
			if (w > 11) {
				ctx.fillStyle = s.val > 0 ? "rgba(0,0,0,0.75)" : a.color;
				ctx.font = "700 7.5px ui-monospace, monospace";
				ctx.fillText(s.val > 0 ? a.pos : a.neg, x0 + 2.5, y + ROW_H - 1.3);
			}
		}
	}

	function frame(): void {
		raf = requestAnimationFrame(frame);
		if (!canvas) return;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;
		const now = performance.now();
		const span = seconds * 1000;
		const plot_w = W - GUTTER - 8;
		const x_of = (t: number) => GUTTER + plot_w - ((now - t) / span) * plot_w;
		const t_min = now - span - 500;
		prune(store.you, t_min);
		prune(store.played, t_min);
		prune(store.ticks, t_min);
		prune(store.blocks, t_min);

		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, W, H);

		// lane backgrounds + labels
		ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
		for (const [y, name] of [
			[YOU_Y, "You"],
			[PLAY_Y, "Played"]
		] as [number, string][]) {
			ctx.fillStyle = "rgba(255,255,255,0.035)";
			ctx.fillRect(GUTTER, y - 2, plot_w, LANE_H + 4);
			ctx.fillStyle = "rgba(244,245,247,0.75)";
			ctx.fillText(name, 8, y + LANE_H / 2 + 4);
		}
		// second grid
		ctx.strokeStyle = "rgba(255,255,255,0.06)";
		ctx.fillStyle = "rgba(244,245,247,0.38)";
		ctx.font = "9px ui-monospace, monospace";
		ctx.lineWidth = 1;
		const first = Math.ceil((now - span) / 1000);
		for (let s = first; s <= Math.floor(now / 1000); s++) {
			const x = Math.round(x_of(s * 1000)) + 0.5;
			if (x < GUTTER) continue;
			ctx.beginPath();
			ctx.moveTo(x, YOU_Y - 2);
			ctx.lineTo(x, PLAY_Y + LANE_H + 2);
			ctx.stroke();
		}
		for (let k = 0; k <= seconds; k += 2) {
			const x = GUTTER + plot_w - (k / seconds) * plot_w;
			ctx.fillText(k === 0 ? "now" : `-${k}s`, x - (k === 0 ? 16 : 8), AXIS_Y + 8);
		}

		// block boundaries (played lane)
		ctx.strokeStyle = "rgba(255,255,255,0.28)";
		ctx.setLineDash([2, 2]);
		for (const t of store.blocks) {
			const x = Math.round(x_of(t)) + 0.5;
			if (x < GUTTER) continue;
			ctx.beginPath();
			ctx.moveTo(x, PLAY_Y - 4);
			ctx.lineTo(x, PLAY_Y + LANE_H + 4);
			ctx.stroke();
		}
		ctx.setLineDash([]);

		draw_segs(ctx, store.you, YOU_Y, now, x_of);
		draw_segs(ctx, store.played, PLAY_Y, now, x_of);

		// control event ticks (just under the You lane)
		ctx.fillStyle = "rgba(244,245,247,0.45)";
		for (const t of store.ticks) {
			const x = x_of(t);
			if (x < GUTTER) continue;
			ctx.fillRect(x - 0.5, YOU_Y + LANE_H + 3, 1, 4);
		}

		// latency bracket between the lanes
		const lat = store.latency_ms;
		if (lat != null && Number.isFinite(lat)) {
			const xr = x_of(now);
			const xl = Math.max(GUTTER, x_of(now - lat));
			const y = MID_Y + MID_H / 2 + 1;
			ctx.strokeStyle = "rgba(244,245,247,0.55)";
			ctx.beginPath();
			ctx.moveTo(xl, y);
			ctx.lineTo(xr, y);
			ctx.moveTo(xl + 0.5, y - 4);
			ctx.lineTo(xl + 0.5, y + 4);
			ctx.moveTo(xr - 0.5, y - 4);
			ctx.lineTo(xr - 0.5, y + 4);
			ctx.stroke();
			const label = `latency ${(lat / 1000).toFixed(2)} s`;
			ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
			const tw = ctx.measureText(label).width;
			const lx = Math.min(Math.max((xl + xr) / 2 - tw / 2, GUTTER + 2), xr - tw - 2);
			ctx.fillStyle = "#11141b";
			ctx.fillRect(lx - 5, y - 8, tw + 10, 15);
			ctx.fillStyle = "#f4f5f7";
			ctx.fillText(label, lx, y + 3.5);
		}
		// now marker
		ctx.strokeStyle = "rgba(249,115,22,0.8)";
		ctx.beginPath();
		ctx.moveTo(x_of(now) - 0.5, YOU_Y - 4);
		ctx.lineTo(x_of(now) - 0.5, PLAY_Y + LANE_H + 4);
		ctx.stroke();
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
	<canvas bind:this={canvas} data-testid="worldviewer-timeline"></canvas>
	<div class="legend">
		{#each AXES as a (a.axis)}
			<span><i style:background={a.color}></i>{a.pos}/{a.neg} {a.axis}</span>
		{/each}
		<span><i class="hatch"></i>autopilot</span>
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
		padding: 2px 8px 0 58px;
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
		background: repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.35) 0 1.5px, transparent 1.5px 4px);
	}
	.legend i.blk {
		width: 1px;
		height: 10px;
		border-left: 1px dashed rgba(255, 255, 255, 0.5);
		border-radius: 0;
	}
	@media (max-width: 520px) {
		.legend {
			padding-left: 8px;
		}
	}
</style>
