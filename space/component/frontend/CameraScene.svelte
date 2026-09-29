<script lang="ts">
	// Scene graph of the "Camera" panel: point cloud, source frustums, generated camera path.
	// The player writes into a plain mutable `store` at up to 60 Hz; a threlte task polls its version
	// counters and only invalidates (re-renders on demand) when something changed.
	import { T, useTask, useThrelte } from "@threlte/core";
	import { OrbitControls } from "@threlte/extras";
	import {
		BufferAttribute,
		BufferGeometry,
		DynamicDrawUsage,
		Float32BufferAttribute,
		type PerspectiveCamera
	} from "three";
	import { untrack } from "svelte";
	import type { CamStore } from "./types";

	let {
		store,
		visible = true,
		reset = 0,
		accent = "#f97316",
		empty = $bindable(true)
	}: {
		store: CamStore;
		visible?: boolean;
		reset?: number;
		accent?: string;
		empty?: boolean;
	} = $props();

	const { invalidate, dpr } = useThrelte();

	let cam: PerspectiveCamera | undefined = $state.raw();
	let controls: any = $state.raw();

	let points_geom: BufferGeometry | null = $state.raw(null);
	let src_frustums: BufferGeometry | null = $state.raw(null);
	let src_path: BufferGeometry | null = $state.raw(null);
	let played_geom: BufferGeometry = $state.raw(new BufferGeometry());
	let buf_geom: BufferGeometry = $state.raw(new BufferGeometry());
	const cur_geom = new BufferGeometry();
	const marks_geom = new BufferGeometry();
	let cur_visible = $state(false);

	let path_cap = 0;
	let path_attr: BufferAttribute | null = null;
	let seen_scene = -1;
	let seen_path = -1;
	let radius = 1;
	let center = [0, 0, 0];
	let K = [500, 500, 416, 240, 832, 480];
	let framed_by_path = false;

	// ------------------------------------------------------------- helpers
	function b64_bytes(b64: string): Uint8Array {
		const bin = atob(b64);
		const out = new Uint8Array(bin.length);
		for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
		return out;
	}
	const SRGB_TO_LINEAR = new Float32Array(256).map((_, i) => {
		const c = i / 255;
		return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
	});
	function cam_center(p: number[]): [number, number, number] {
		return [p[3], p[7], p[11]];
	}
	function to_three(v: number[]): number[] {
		// the world group is rotated 180 deg about x: OpenCV (x, y, z) -> three (x, -y, -z)
		return [v[0], -v[1], -v[2]];
	}
	/** Line-segment vertices of a camera frustum at depth d (OpenCV world coords). */
	function frustum_verts(p: number[], d: number, out: number[]): void {
		const [fx, fy, cx, cy, w, h] = K;
		const c = cam_center(p);
		const corners = [
			[0, 0],
			[w, 0],
			[w, h],
			[0, h]
		].map(([u, v]) => {
			const x = ((u - cx) / fx) * d,
				y = ((v - cy) / fy) * d,
				z = d;
			return [
				p[0] * x + p[1] * y + p[2] * z + c[0],
				p[4] * x + p[5] * y + p[6] * z + c[1],
				p[8] * x + p[9] * y + p[10] * z + c[2]
			];
		});
		for (let i = 0; i < 4; i++) out.push(...c, ...corners[i]);
		for (let i = 0; i < 4; i++) out.push(...corners[i], ...corners[(i + 1) % 4]);
		// "up" tick above the image's top edge (image up = -y in OpenCV)
		const top = [0, 1, 2].map((k) => (corners[0][k] + corners[1][k]) / 2);
		const tip = [top[0] - p[1] * d * 0.25, top[1] - p[5] * d * 0.25, top[2] - p[9] * d * 0.25];
		out.push(...corners[0], ...tip, ...tip, ...corners[1]);
	}
	function geom_from(verts: number[]): BufferGeometry {
		const g = new BufferGeometry();
		g.setAttribute("position", new Float32BufferAttribute(verts, 3));
		return g;
	}

	function robust_bounds(xyz: Float32Array, extra: number[][]): void {
		const n = xyz.length / 3;
		const step = Math.max(1, Math.floor(n / 6000));
		const ax: number[][] = [[], [], []];
		for (let i = 0; i < n; i += step)
			for (let k = 0; k < 3; k++) {
				const v = xyz[i * 3 + k];
				if (Number.isFinite(v)) ax[k].push(v);
			}
		for (const e of extra) for (let k = 0; k < 3; k++) ax[k].push(e[k]);
		const lo: number[] = [],
			hi: number[] = [];
		for (let k = 0; k < 3; k++) {
			const a = ax[k].sort((x, y) => x - y);
			lo.push(a.length ? a[Math.floor(a.length * 0.03)] : -1);
			hi.push(a.length ? a[Math.min(a.length - 1, Math.floor(a.length * 0.97))] : 1);
		}
		center = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
		radius = Math.max(1e-3, 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
	}

	function frame_view(): void {
		if (!cam || !controls) return;
		const src = store.scene?.source_poses?.[0] ?? store.poses[0] ?? null;
		const c3 = to_three(center);
		let eye: number[];
		if (src) {
			// behind and above the (first) source camera, looking at the scene centre
			const sc = to_three(cam_center(src));
			const fwd = to_three([src[2], src[6], src[10]]);
			eye = [
				sc[0] - fwd[0] * radius * 0.9,
				sc[1] - fwd[1] * radius * 0.9 + radius * 0.75,
				sc[2] - fwd[2] * radius * 0.9
			];
		} else {
			eye = [c3[0] + radius, c3[1] + radius, c3[2] + radius * 1.5];
		}
		cam.position.set(eye[0], eye[1], eye[2]);
		cam.near = radius / 200;
		cam.far = radius * 60;
		cam.updateProjectionMatrix();
		controls.target.set(c3[0], c3[1], c3[2]);
		controls.update();
		invalidate();
	}

	// ------------------------------------------------------------- builders
	function build_scene(): void {
		points_geom?.dispose();
		src_frustums?.dispose();
		src_path?.dispose();
		points_geom = src_frustums = src_path = null;
		const s = store.scene;
		if (!s) return;
		if (s.intrinsics?.length === 6) K = s.intrinsics.slice();
		let xyz = new Float32Array(0);
		try {
			const pb = b64_bytes(s.points);
			xyz = new Float32Array(pb.buffer, 0, Math.floor(pb.byteLength / 12) * 3);
			const cb = b64_bytes(s.colors);
			const n = Math.min(xyz.length / 3, Math.floor(cb.length / 3));
			const lin = new Float32Array(n * 3);
			for (let i = 0; i < n * 3; i++) lin[i] = SRGB_TO_LINEAR[cb[i]];
			const g = new BufferGeometry();
			g.setAttribute("position", new BufferAttribute(xyz.subarray(0, n * 3), 3));
			g.setAttribute("color", new BufferAttribute(lin, 3));
			points_geom = g;
		} catch (e) {
			console.warn("[WorldViewer] bad scene points", e);
		}
		const sp = s.source_poses ?? [];
		robust_bounds(xyz, sp.map(cam_center));
		const verts: number[] = [];
		const every = sp.length > 12 ? 12 : 1;
		sp.forEach((p, i) => {
			if (i % every === 0 || i === sp.length - 1) frustum_verts(p, radius * 0.12, verts);
		});
		if (verts.length) src_frustums = geom_from(verts);
		if (sp.length > 1) src_path = geom_from(sp.flatMap(cam_center));
		framed_by_path = false;
		frame_view();
	}

	function ensure_path_capacity(n: number): void {
		if (n <= path_cap && path_attr) return;
		path_cap = Math.max(1024, 2 ** Math.ceil(Math.log2(Math.max(1, n))));
		path_attr = new BufferAttribute(new Float32Array(path_cap * 3), 3);
		path_attr.setUsage(DynamicDrawUsage);
		played_geom.dispose();
		buf_geom.dispose();
		const a = new BufferGeometry();
		a.setAttribute("position", path_attr);
		const b = new BufferGeometry();
		b.setAttribute("position", path_attr); // both lines share one attribute, split by draw range
		played_geom = a;
		buf_geom = b;
	}

	function update_path(): void {
		const poses = store.poses;
		const n = poses.length;
		ensure_path_capacity(n);
		const arr = path_attr!.array as Float32Array;
		for (let i = 0; i < n; i++) {
			const p = poses[i];
			arr[i * 3] = p[3];
			arr[i * 3 + 1] = p[7];
			arr[i * 3 + 2] = p[11];
		}
		path_attr!.needsUpdate = true;
		const played = Math.min(store.played, n);
		played_geom.setDrawRange(0, played);
		const b0 = Math.max(0, played - 1);
		buf_geom.setDrawRange(b0, n - b0);

		const cur = played > 0 ? poses[played - 1] : null;
		cur_visible = !!cur;
		if (cur) {
			const v: number[] = [];
			frustum_verts(cur, radius * 0.1, v);
			cur_geom.setAttribute("position", new Float32BufferAttribute(v, 3));
		}
		// faint frustum at the start of each 12-frame block of the path
		const fv: number[] = [];
		for (let i = 0; i < n; i += 12) if (i !== played - 1) frustum_verts(poses[i], radius * 0.05, fv);
		marks_geom.setAttribute("position", new Float32BufferAttribute(fv, 3));

		if (!store.scene && n > 0 && !framed_by_path) {
			robust_bounds(new Float32Array(0), poses.map(cam_center));
			radius = Math.max(radius, 1);
			framed_by_path = true;
			frame_view();
		}
	}

	useTask(
		() => {
			if (!visible || !cam || !controls) return;
			let changed = false;
			if (store.scene_version !== seen_scene) {
				seen_scene = store.scene_version;
				build_scene();
				changed = true;
			}
			if (store.version !== seen_path) {
				seen_path = store.version;
				update_path();
				changed = true;
			}
			if (changed) {
				const e = !store.scene && store.poses.length === 0;
				if (e !== empty) empty = e;
				invalidate();
			}
		},
		{ autoInvalidate: false }
	);

	$effect(() => {
		reset;
		untrack(() => frame_view());
	});
	$effect(() => {
		if (visible) invalidate();
	});
	$effect(() => {
		if (cam && controls) untrack(() => frame_view());
	});
</script>

<T.PerspectiveCamera makeDefault fov={50} near={0.01} far={1000} bind:ref={cam}>
	<!-- pointer-only controls: no keyboard listeners, WASD stays with the world model -->
	<OrbitControls bind:ref={controls} />
</T.PerspectiveCamera>

<T.Group rotation.x={Math.PI}>
	{#if points_geom}
		<T.Points>
			<T is={points_geom} />
			<T.PointsMaterial size={1.7 * dpr.current} sizeAttenuation={false} vertexColors />
		</T.Points>
	{/if}
	{#if src_frustums}
		<T.LineSegments>
			<T is={src_frustums} />
			<T.LineBasicMaterial color="#9ca3af" transparent opacity={0.8} />
		</T.LineSegments>
	{/if}
	{#if src_path}
		<T.Line>
			<T is={src_path} />
			<T.LineBasicMaterial color="#9ca3af" transparent opacity={0.6} />
		</T.Line>
	{/if}
	<T.Line frustumCulled={false}>
		<T is={played_geom} />
		<T.LineBasicMaterial color={accent} />
	</T.Line>
	<T.Line frustumCulled={false}>
		<T is={buf_geom} />
		<T.LineBasicMaterial color={accent} transparent opacity={0.3} />
	</T.Line>
	<T.LineSegments frustumCulled={false}>
		<T is={marks_geom} />
		<T.LineBasicMaterial color={accent} transparent opacity={0.35} />
	</T.LineSegments>
	<T.LineSegments visible={cur_visible} frustumCulled={false}>
		<T is={cur_geom} />
		<T.LineBasicMaterial color={accent} />
	</T.LineSegments>
</T.Group>
