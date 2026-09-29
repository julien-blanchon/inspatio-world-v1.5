import type { ILoadingStatus as LoadingStatus } from "@gradio/statustracker";

export type Status = "idle" | "loading" | "running" | "ended" | "error";

export interface Action {
	forward: number;
	right: number;
	up: number;
	yaw: number;
	pitch: number;
}

export interface Chunk {
	id: number;
	session: string;
	fps: number;
	frames: string[];
	renders?: string[] | null;
	actions?: (Partial<Action> | null)[] | null;
	control_seqs?: (number | null)[] | null;
	poses?: number[][] | null;
}

export interface SceneInfo {
	points: string;
	colors: string;
	source_poses?: number[][];
	intrinsics?: number[] | null;
	kind?: "image" | "video";
	title?: string;
}

export interface WorldViewerValue {
	status: Status;
	message?: string;
	chunk?: Chunk | null;
	stats?: Record<string, number | string>;
	scene?: SceneInfo | null;
}

export interface ControlPayload extends Action {
	seq: number;
	session: string;
}

export interface WorldViewerProps {
	value: WorldViewerValue | null;
	show_render: boolean;
	show_camera: boolean;
	show_timeline: boolean;
	prebuffer_frames: number;
	catchup_rate: number;
	heartbeat_ms: number;
	max_control_hz: number;
	timeline_seconds: number;
	placeholder: string | null;
	aspect_ratio: number;
}

export interface WorldViewerEvents {
	change: never;
	start: Record<string, never>;
	stop: Record<string, never>;
	control: ControlPayload;
	clear_status: LoadingStatus;
}

export type VKey =
	| "fwd"
	| "back"
	| "left"
	| "right"
	| "yawL"
	| "yawR"
	| "pitchU"
	| "pitchD"
	| "up"
	| "down";

export interface Frame {
	bm: ImageBitmap | HTMLImageElement;
	rb: ImageBitmap | HTMLImageElement | null;
	fps: number;
	action: Action | null; // null = autopilot / unknown
	has_action: boolean; // whether the chunk carried actions at all
	seq: number | null;
	pose: number[] | null;
	chunk_id: number;
	first_in_chunk: boolean;
}

export type Axis = "forward" | "right" | "yaw" | "pitch" | "up";

/** A timeline bar: axis value held from t0 to t1 (ms, performance.now clock); t1 null = still open. */
export interface Seg {
	axis: Axis | "auto";
	val: number;
	t0: number;
	t1: number | null;
}

/** Shared mutable timeline store (written by the player/controls, read by the Timeline rAF loop). */
export interface TimelineStore {
	you: Seg[];
	played: Seg[];
	ticks: number[]; // control events sent
	blocks: number[]; // block boundary display times
	latency_ms: number | null;
}

/** Shared mutable store written by the player; polled by CameraView in rAF (no Svelte reactivity at 15-60 Hz). */
export interface CamStore {
	scene: SceneInfo | null;
	scene_version: number;
	poses: number[][]; // generated camera path of the current session (queue order)
	played: number; // number of poses already displayed
	version: number;
}
