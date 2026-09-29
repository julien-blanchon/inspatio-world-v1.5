<script lang="ts">
	// Scene graph of the "Camera" panel: a faint point cloud, source frustums, and the generated camera
	// path drawn with thick lines (played = solid, buffered = faded; dots per frame, bigger dots per block;
	// a single frustum for the current camera).
	// The player writes into a plain mutable `store` at up to 60 Hz; a threlte task polls its version
	// counters and only invalidates (re-renders on demand) when something changed.
	import { T, useTask, useThrelte } from "@threlte/core";
	import { OrbitControls } from "@threlte/extras";
	import {
		BufferAttribute,
		BufferGeometry,
		CanvasTexture,
		Float32BufferAttribute,
		Points,
		PointsMaterial,
		type PerspectiveCamera
	} from "three";
	import { Line2 } from "three/examples/jsm/lines/Line2.js";
	import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
	import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
	import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
	import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
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

	const { invalidate, dpr, size } = useThrelte();

	let cam: PerspectiveCamera | undefined = $state.raw();
	let controls: any = $state.raw();

	let points_geom: BufferGeometry | null = $state.raw(null);
	let src_frustums: BufferGeometry | null = $state.raw(null);
	let src_path: BufferGeometry | null = $state.raw(null);

	// thick lines (screen-space width in CSS px); objects are created once, geometries swapped
	const mat_played = new LineMaterial({ color: accent, linewidth: 3, worldUnits: false });
	const mat_buf = new LineMaterial({ color: accent, linewidth: 2.5, transparent: true, opacity: 0.3, depthWrite: false });
	const mat_cur = new LineMaterial({ color: "#ffd2a8", linewidth: 2.5 });
	const played_line = new Line2(new LineGeometry(), mat_played);
	const buf_line = new Line2(new LineGeometry(), mat_buf);
	const cur_frustum = new LineSegments2(new LineSegmentsGeometry(), mat_cur);
	for (const o of [played_line, buf_line, cur_frustum]) {
		o.frustumCulled = false;
		o.visible = false;
	}
	cur_frustum.renderOrder = 4;
	played_line.renderOrder = 2;

	// position markers along the path: round screen-space dots per frame, bigger ones per block start
	function disc_texture(): CanvasTexture | null {
		if (typeof document === "undefined") return null;
		const c = document.createElement("canvas");
		c.width = c.height = 64;
		const g = c.getContext("2d")!;
		g.fillStyle = "#fff";
		g.beginPath();
		g.arc(32, 32, 28, 0, Math.PI * 2);
		g.fill();
		return new CanvasTexture(c);
	}
	const disc = disc_texture();
	const dot_mat = (px: number, opacity: number) =>
		new PointsMaterial({
			color: accent,
			size: px * dpr.current,
			sizeAttenuation: false,
			map: disc,
			alphaTest: 0.5,
			transparent: opacity < 1,
			opacity,
			depthWrite: opacity >= 1
		});
	const played_dots = new Points(new BufferGeometry(), dot_mat(5, 1));
	const buf_dots = new Points(new BufferGeometry(), dot_mat(5, 0.3));
	const played_blk = new Points(new BufferGeometry(), dot_mat(10, 1));
	const buf_blk = new Points(new BufferGeometry(), dot_mat(10, 0.3));
	for (const o of [played_dots, buf_dots, played_blk, buf_blk]) {
		o.frustumCulled = false;
		o.renderOrder = 3;
	}

	$effect(() => {
		const s = $size;
		for (const m of [mat_played, mat_buf, mat_cur]) m.resolution.set(s.width, s.height);
		invalidate();
	});

	let seen_scene = -1;
	let seen_path = -1;
	let depth = 1; // typical scene depth in front of the source camera (sets frustum sizes / framing)
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
	type V3 = [number, number, number];
	const cam_center = (p: number[]): V3 => [p[3], p[7], p[11]];
	const col = (p: number[], k: number): V3 => [p[k], p[4 + k], p[8 + k]]; // rotation column k
	const add = (a: V3, b: V3, s = 1): V3 => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
	// the world group is rotated 180 deg about x: OpenCV (x, y, z) -> three (x, -y, -z)
	const to_three = (v: V3): V3 => [v[0], -v[1], -v[2]];

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
	function swap_line(obj: Line2, pts: number[]): void {
		const old = obj.geometry;
		if (pts.length >= 6) {
			const g = new LineGeometry();
			g.setPositions(pts);
			obj.geometry = g;
			obj.visible = true;
		} else {
			obj.geometry = new LineGeometry();
			obj.visible = false;
		}
		old.dispose();
	}
	function swap_points(obj: Points, verts: number[]): void {
		const old = obj.geometry;
		obj.geometry = geom_from(verts);
		obj.visible = verts.length >= 3;
		old.dispose();
	}
	function swap_segments(obj: LineSegments2, verts: number[]): void {
		const old = obj.geometry;
		const g = new LineSegmentsGeometry();
		if (verts.length >= 6) g.setPositions(verts);
		obj.geometry = g;
		obj.visible = verts.length >= 6;
		old.dispose();
	}

	/** Median depth of the points in front of camera p (sampled). */
	function median_depth(xyz: Float32Array, p: number[]): number {
		const n = xyz.length / 3;
		if (!n) return 1;
		const c = cam_center(p),
			f = col(p, 2);
		const step = Math.max(1, Math.floor(n / 5000));
		const zs: number[] = [];
		for (let i = 0; i < n; i += step) {
			const z = (xyz[i * 3] - c[0]) * f[0] + (xyz[i * 3 + 1] - c[1]) * f[1] + (xyz[i * 3 + 2] - c[2]) * f[2];
			if (z > 0 && Number.isFinite(z)) zs.push(z);
		}
		if (!zs.length) return 1;
		zs.sort((a, b) => a - b);
		return Math.max(1e-3, zs[Math.floor(zs.length / 2)]);
	}

	/**
	 * 3/4 elevated view from behind/above the source (or first path) camera, looking at the region
	 * just in front of it, so forward motion and turns read clearly. Frames the path if it has
	 * already wandered further than that region.
	 */
	function frame_view(): void {
		if (!cam || !controls) return;
		const ref = store.scene?.source_poses?.[0] ?? store.poses[0] ?? null;
		let target: V3, eye: V3;
		if (ref) {
			const c = cam_center(ref);
			const right = col(ref, 0),
				down = col(ref, 1),
				fwd = col(ref, 2);
			let reach = depth * 0.35;
			for (const p of store.poses) {
				const q = cam_center(p);
				reach = Math.max(reach, Math.hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]) * 0.6);
			}
			target = add(c, fwd, Math.min(reach, depth * 0.6));
			const r = Math.max(reach, depth * 0.35);
			eye = add(add(add(target, fwd, -2.0 * r), down, -1.3 * r), right, 0.9 * r);
		} else {
			target = [0, 0, 0];
			eye = [1.5, -1.5, -2.5];
		}
		const t3 = to_three(target),
			e3 = to_three(eye);
		cam.position.set(e3[0], e3[1], e3[2]);
		cam.near = depth / 500;
		cam.far = depth * 100;
		cam.updateProjectionMatrix();
		controls.target.set(t3[0], t3[1], t3[2]);
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
		depth = sp.length ? median_depth(xyz, sp[0]) : 1;
		const verts: number[] = [];
		const every = sp.length > 12 ? 12 : 1;
		sp.forEach((p, i) => {
			if (i % every === 0 || i === sp.length - 1) frustum_verts(p, depth * 0.12, verts);
		});
		if (verts.length) src_frustums = geom_from(verts);
		if (sp.length > 1) src_path = geom_from(sp.flatMap(cam_center));
		framed_by_path = false;
		frame_view();
	}

	function update_path(): void {
		const poses = store.poses;
		const n = poses.length;
		const played = Math.min(store.played, n);
		const pts = poses.flatMap(cam_center);
		swap_line(played_line, pts.slice(0, played * 3));
		swap_line(buf_line, pts.slice(Math.max(0, played - 1) * 3));

		const cur = played > 0 ? poses[played - 1] : null;
		const cv: number[] = [];
		if (cur) frustum_verts(cur, depth * 0.1, cv);
		swap_segments(cur_frustum, cv);
		// dots: every frame, and bigger ones at block starts (every 12 frames); played solid, buffered faded
		swap_points(played_dots, pts.slice(0, played * 3));
		swap_points(buf_dots, pts.slice(played * 3));
		const pb: number[] = [],
			bb: number[] = [];
		for (let i = 0; i < n; i += 12) (i < played ? pb : bb).push(...cam_center(poses[i]));
		swap_points(played_blk, pb);
		swap_points(buf_blk, bb);

		if (!store.scene && n > 0 && !framed_by_path) {
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
		<!-- faint cloud so the camera path and frustums stand out -->
		<T.Points renderOrder={0}>
			<T is={points_geom} />
			<T.PointsMaterial
				size={1.2 * dpr.current}
				sizeAttenuation={false}
				vertexColors
				transparent
				opacity={0.3}
				depthWrite={false}
			/>
		</T.Points>
	{/if}
	{#if src_frustums}
		<T.LineSegments>
			<T is={src_frustums} />
			<T.LineBasicMaterial color="#cbd5e1" transparent opacity={0.9} />
		</T.LineSegments>
	{/if}
	{#if src_path}
		<T.Line>
			<T is={src_path} />
			<T.LineBasicMaterial color="#cbd5e1" transparent opacity={0.6} />
		</T.Line>
	{/if}
	<T is={buf_line} />
	<T is={played_line} />
	<T is={buf_dots} />
	<T is={buf_blk} />
	<T is={played_dots} />
	<T is={played_blk} />
	<T is={cur_frustum} />
</T.Group>
