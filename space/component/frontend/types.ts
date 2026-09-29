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
	// fixed-latency scheduling (optional; without them the viewer falls back to a jitter buffer)
	frame_start?: number | null; // session frame index of frames[0]
	elapsed?: number | null; // seconds since the server's session clock t0 when the chunk was yielded
	latency?: number | null; // seconds: frame f is due at t0 + latency + f / fps
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
	latency: number;
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
	index: number | null; // session frame index (scheduled mode)
	arrived: number; // performance.now() when decoded and queued
}

export type Axis = "forward" | "right" | "yaw" | "pitch" | "up";

/** A key-press bar: axis value held from t0 to t1 (ms, performance.now clock); t1 null = still held. */
export interface Seg {
	axis: Axis;
	val: number;
	t0: number;
	t1: number | null;
}

/** A frame as the timeline needs it (displayed or still buffered). */
export interface TlFrame {
	action: Action | null;
	has_action: boolean;
	fps: number;
	first_in_chunk: boolean;
	index: number | null; // session frame index (scheduled mode)
}

/** A displayed frame: shown at time t (performance.now) for dur ms. */
export interface PlayedFrame extends TlFrame {
	t: number;
	dur: number;
}

/** Shared mutable timeline store (written by the player/controls, read by the Timeline rAF loop). */
export interface TimelineStore {
	you: Seg[]; // key presses
	frames: PlayedFrame[]; // frames actually displayed
	queue: TlFrame[]; // buffered frames, in play order (same array the player consumes)
	latency_ms: number | null; // smoothed measured key -> display lag (null until measured)
	session: number; // bumped when a new session starts (clears the view)
	/** Fixed-latency schedule: frame f was sampled at base0 + f*1000/fps and is shown delay_ms later. */
	sched: { base0: number; fps: number; delay_ms: number } | null;
}

/** Shared mutable store written by the player; polled by CameraView in rAF (no Svelte reactivity at 15-60 Hz). */
export interface CamStore {
	scene: SceneInfo | null;
	scene_version: number;
	poses: number[][]; // generated camera path of the current session (queue order)
	played: number; // number of poses already displayed
	version: number;
}
