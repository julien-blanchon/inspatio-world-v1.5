<script lang="ts">
	import type { WorldViewerValue } from "./types";
	let {
		value,
		type,
		selected = false
	}: {
		value: WorldViewerValue | null;
		type: "gallery" | "table";
		selected?: boolean;
	} = $props();
	const first = $derived(value?.chunk?.frames?.[0] ?? null);
</script>

<div class:table={type === "table"} class:gallery={type === "gallery"} class:selected>
	{#if first}
		<img src={first} alt="first frame" />
	{:else}
		{value?.status ?? ""}
	{/if}
</div>

<style>
	.gallery {
		padding: var(--size-1) var(--size-2);
	}
	img {
		max-height: 48px;
		border-radius: 4px;
	}
</style>
