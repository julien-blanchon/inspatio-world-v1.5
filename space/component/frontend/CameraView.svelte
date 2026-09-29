<script lang="ts">
	// Threlte canvas wrapper. This module (and three.js with it) is loaded lazily by Index.svelte.
	import { Canvas } from "@threlte/core";
	import { NoToneMapping, WebGLRenderer } from "three";
	import CameraScene from "./CameraScene.svelte";
	import type { CamStore } from "./types";

	let {
		store,
		visible = true,
		reset = 0,
		accent = "#f97316"
	}: { store: CamStore; visible?: boolean; reset?: number; accent?: string } = $props();

	let empty = $state(true);

	function create_renderer(canvas: HTMLCanvasElement): WebGLRenderer {
		canvas.setAttribute("data-testid", "worldviewer-3d");
		// preserveDrawingBuffer keeps the last on-demand frame readable (screenshots / tests)
		return new WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
	}
</script>

<div class="cv">
	<Canvas renderMode="on-demand" dpr={[1, 2]} toneMapping={NoToneMapping} createRenderer={create_renderer}>
		<CameraScene {store} {visible} {reset} {accent} bind:empty />
	</Canvas>
	{#if empty}
		<div class="note">The scene and your camera path appear here once a session starts.</div>
	{/if}
</div>

<style>
	.cv {
		position: absolute;
		inset: 0;
		cursor: grab;
		touch-action: none;
	}
	.cv:active {
		cursor: grabbing;
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
