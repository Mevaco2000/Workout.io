from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .config import CHART_POINTS
from .counters import ExerciseRepCounter
from .exercise_prediction import create_exercise_predictor
from .models import DisplaySettings, create_pipeline_models, resize_for_display
from .storage import JsonlPointStorage


@dataclass
class AnalysisSettings:
    source_type: Literal["video", "camera"]
    source_path: str | None
    camera_index: int
    pose_selection: str
    show_pose_preview: bool
    auxiliary_point_radius: int
    use_barbell_tracking: bool
    barbell_selection: str
    use_auxiliary_yolo: bool
    auxiliary_yolo_path: str
    pose_confidence_threshold: float
    barbell_confidence_threshold: float
    auxiliary_yolo_confidence_threshold: float
    exercise: str
    auto_exercise_model_path: str | None
    aspect_ratio: str
    scale_label: str
    reset_barbell_path_each_rep: bool


@dataclass
class FramePacket:
    frame_bgr: np.ndarray
    frame_index: int
    total_frames: int
    progress: float
    reps: int
    status_text: str
    points_path: str
    side_predictions: list[str]
    active_exercise: str


@dataclass
class StatePacket:
    kind: Literal["status", "error", "done", "paused", "running"]
    message: str
    points_path: str | None = None


class VideoAnalysisWorker(threading.Thread):
    def __init__(self, settings: AnalysisSettings, frame_queue: queue.Queue[FramePacket], state_queue: queue.Queue[StatePacket]) -> None:
        super().__init__(daemon=True)
        self.settings = settings
        self.frame_queue = frame_queue
        self.state_queue = state_queue
        self.pause_event = threading.Event()
        self.cancel_event = threading.Event()
        self._display_lock = threading.Lock()
        self._display_settings = DisplaySettings(
            settings.aspect_ratio,
            settings.scale_label,
            settings.show_pose_preview,
            settings.auxiliary_point_radius,
        )
        self._storage: JsonlPointStorage | None = None

    def update_display_settings(
        self,
        aspect_ratio: str,
        scale_label: str,
        show_pose_preview: bool,
        auxiliary_point_radius: int,
    ) -> None:
        with self._display_lock:
            self._display_settings = DisplaySettings(
                aspect_ratio,
                scale_label,
                show_pose_preview,
                auxiliary_point_radius,
            )

    @property
    def points_path(self) -> str | None:
        return None if self._storage is None else str(self._storage.path)

    def run(self) -> None:
        capture = None
        self._storage = JsonlPointStorage()
        try:
            source = self.settings.camera_index if self.settings.source_type == "camera" else str(self.settings.source_path)
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError("Failed to open the video source.")

            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 25.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self.settings.source_type == "video" else 0
            frame_interval = 1.0 / fps if fps > 0 else 0.04

            models = create_pipeline_models(
                self.settings.pose_selection,
                self.settings.pose_confidence_threshold,
                self.settings.use_barbell_tracking,
                self.settings.barbell_selection,
                self.settings.barbell_confidence_threshold,
                self.settings.use_auxiliary_yolo,
                self.settings.auxiliary_yolo_path,
                self.settings.auxiliary_yolo_confidence_threshold,
            )
            predictor = create_exercise_predictor(
                self.settings.exercise,
                model_path=self.settings.auto_exercise_model_path,
            )
            rep_counter = None if predictor is not None else ExerciseRepCounter(self.settings.exercise)
            previous_reps = 0
            active_exercise = self.settings.exercise
            frame_index = 0
            self._push_state("running", "Processing started.")

            while not self.cancel_event.is_set():
                if self.pause_event.is_set():
                    self._push_state("paused", "Processing paused.")
                    time.sleep(0.05)
                    continue

                started_at = time.perf_counter()
                success, frame = capture.read()
                if not success:
                    break

                with self._display_lock:
                    display_settings = self._display_settings

                annotated_frame, tracked_points, side_predictions = models.process(
                    frame,
                    show_pose_overlay=display_settings.show_pose_overlay,
                    auxiliary_point_radius=display_settings.auxiliary_point_radius,
                )
                predicted_top_labels: list[str] = []
                if predictor is not None:
                    prediction = predictor.update(tracked_points)
                    if prediction is not None:
                        active_exercise = prediction["exercise"]
                        predicted_top_labels = prediction["top_predictions"]
                        if rep_counter is None or rep_counter.exercise != active_exercise:
                            rep_counter = ExerciseRepCounter(active_exercise)
                    else:
                        active_exercise = "auto (waiting)"

                sample = self._build_sample(frame_index, fps, tracked_points)
                barbell_prediction = self._build_barbell_prediction(frame_index, fps, tracked_points)
                reps = rep_counter.update(sample) if rep_counter is not None else 0
                if rep_counter is not None:
                    sample["reps_completed"] = int(reps)
                if len(sample) > 2:
                    self._storage.append(sample)
                if barbell_prediction is not None:
                    self._storage.append_barbell_prediction(barbell_prediction)
                if self.settings.reset_barbell_path_each_rep and reps > previous_reps:
                    models.clear_barbell_trajectory()
                previous_reps = reps

                display_frame = resize_for_display(annotated_frame, display_settings)
                total = max(total_frames, frame_index + 1)
                progress = min((frame_index + 1) / total_frames, 1.0) if total_frames > 0 else 0.0
                status_text = f"Frame {frame_index + 1}/{total}"
                if predictor is not None:
                    if predicted_top_labels:
                        status_text += f" | auto: {predicted_top_labels[0]}"
                    else:
                        status_text += " | auto: collecting sequence"
                packet = FramePacket(
                    frame_bgr=display_frame,
                    frame_index=frame_index + 1,
                    total_frames=total,
                    progress=progress,
                    reps=reps,
                    status_text=status_text,
                    points_path=str(self._storage.path),
                    side_predictions=side_predictions + predicted_top_labels,
                    active_exercise=active_exercise,
                )
                self._push_frame(packet)
                frame_index += 1

                if self.settings.source_type == "video":
                    elapsed = time.perf_counter() - started_at
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            if self.cancel_event.is_set():
                self._push_state("status", "Processing cancelled.", str(self._storage.path))
            else:
                self._push_state("done", "Processing finished.", str(self._storage.path))
        except Exception as exc:
            self._push_state("error", str(exc), self.points_path)
        finally:
            if capture is not None:
                capture.release()
            if self._storage is not None:
                self._storage.close()

    def _build_sample(self, frame_index: int, fps: float, tracked_points: dict[str, dict[str, float]]) -> dict[str, float]:
        sample: dict[str, float] = {"frame_index": float(frame_index), "time_s": frame_index / fps if fps > 0 else float(frame_index)}
        for point_name in CHART_POINTS:
            point = tracked_points.get(point_name)
            if point is None:
                continue
            sample[f"{point_name}_x"] = float(point["x"])
            sample[f"{point_name}_y"] = float(point["y"])
        return sample

    def _build_barbell_prediction(self, frame_index: int, fps: float, tracked_points: dict[str, dict[str, float]]) -> dict[str, float] | None:
        point = tracked_points.get("barbell_center")
        if point is None:
            return None
        return {
            "frame_index": float(frame_index),
            "time_s": frame_index / fps if fps > 0 else float(frame_index),
            "center_x": float(point["x"]),
            "center_y": float(point["y"]),
            "confidence": float(point.get("confidence", 0.0)),
        }

    def _push_frame(self, packet: FramePacket) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.frame_queue.put_nowait(packet)

    def _push_state(self, kind: Literal["status", "error", "done", "paused", "running"], message: str, points_path: str | None = None) -> None:
        while not self.state_queue.empty():
            try:
                self.state_queue.get_nowait()
            except queue.Empty:
                break
        self.state_queue.put_nowait(StatePacket(kind=kind, message=message, points_path=points_path))
