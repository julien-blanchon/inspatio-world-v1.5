import type { ILoadingStatus as LoadingStatus } from "@gradio/statustracker";

export type Status = "idle" | "loading" | "running" | "ended" | "error";

export interface Chunk {
	id: number;
	session: string;
	fps: number;
	frames: string[];
	renders?: string[] | null;
}

export interface WorldViewerValue {
	status: Status;
	message?: string;
	chunk?: Chunk | null;
	stats?: Record<string, number | string>;
}

export interface ControlPayload {
	forward: number;
	right: number;
	up: number;
	yaw: number;
	pitch: number;
	seq: number;
	session: string;
}

export interface WorldViewerProps {
	value: WorldViewerValue | null;
	show_render: boolean;
	show_pad: boolean | null;
	prebuffer_frames: number;
	catchup_rate: number;
	heartbeat_ms: number;
	max_control_hz: number;
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
}
