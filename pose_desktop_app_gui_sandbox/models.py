from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np
from ultralytics import YOLO

from .config import (
    AUXILIARY_KEYPOINT_CONNECTIONS,
    AUXILIARY_KEYPOINT_NAMES,
    BARBELL_MODEL_OPTIONS,
    COCO_KEYPOINT_NAMES,
    DISPLAY_ASPECT_OPTIONS,
    DISPLAY_SCALE_OPTIONS,
    MEDIAPIPE_KEYPOINT_NAMES,
    MEDIAPIPE_MODEL_DIR,
    MEDIAPIPE_MODEL_URLS,
    POSE_MODEL_OPTIONS,
    SKELETON_CONNECTIONS,
)


@dataclass
class DisplaySettings:
    aspect_ratio: str
    scale_label: str
    show_pose_overlay: bool = True
    auxiliary_point_radius: int = 4


def resize_for_display(frame: np.ndarray, settings: DisplaySettings) -> np.ndarray:
    target = DISPLAY_ASPECT_OPTIONS.get(settings.aspect_ratio)
    scale = DISPLAY_SCALE_OPTIONS.get(settings.scale_label, 1.0)
    height, width = frame.shape[:2]
    if target is None:
        target_width = max(1, int(width * scale))
        target_height = max(1, int(height * scale))
    else:
        target_width = max(1, int(target[0] * scale))
        target_height = max(1, int(target[1] * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def validate_pose_selection(selection: str) -> None:
    if selection not in POSE_MODEL_OPTIONS:
        raise ValueError(f"Unsupported pose model: {selection}")
    option = POSE_MODEL_OPTIONS[selection]
    if option["backend"] == "yolo":
        path = Path(option["reference"])
        if not path.exists():
            raise FileNotFoundError(
                f"Pose model file is missing: {path}. Run download_optional_models.py to fetch optional pose models."
            )


def validate_barbell_selection(selection: str) -> None:
    if selection not in BARBELL_MODEL_OPTIONS:
        raise ValueError(f"Unsupported barbell model: {selection}")
    path = BARBELL_MODEL_OPTIONS[selection]
    if not path.exists():
        raise FileNotFoundError(f"Barbell model file is missing: {path}")


def validate_aux_yolo_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Spinal curvature estimation model file is missing: {path}")
    return path


class YoloPoseModel:
    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold

    def predict(self, frame: np.ndarray, draw_overlay: bool = True) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        annotated = frame.copy()
        tracked: dict[str, dict[str, float]] = {}
        result = self.model.predict(source=frame, verbose=False, conf=self.confidence_threshold)[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return annotated, tracked

        xy = keypoints.xy[0].cpu().numpy()
        confidences = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(len(xy))
        for index, (x_value, y_value) in enumerate(xy):
            confidence = float(confidences[index]) if index < len(confidences) else 1.0
            if confidence < self.confidence_threshold:
                continue
            point_name = COCO_KEYPOINT_NAMES.get(index)
            if point_name is None:
                continue
            tracked[point_name] = {"x": float(x_value), "y": float(y_value), "confidence": confidence}
            if draw_overlay:
                cv2.circle(annotated, (int(x_value), int(y_value)), 4, (40, 220, 120), -1)

        if draw_overlay:
            for start_name, end_name in SKELETON_CONNECTIONS:
                if start_name not in tracked or end_name not in tracked:
                    continue
                start_point = tracked[start_name]
                end_point = tracked[end_name]
                cv2.line(
                    annotated,
                    (int(start_point["x"]), int(start_point["y"])),
                    (int(end_point["x"]), int(end_point["y"])),
                    (255, 170, 0),
                    2,
                )
        return annotated, tracked


class MediaPipePoseModel:
    def __init__(self, variant: str, confidence_threshold: float) -> None:
        mp, mp_python, mp_vision = _load_mediapipe_modules()
        model_path = ensure_mediapipe_model_file(variant)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=confidence_threshold,
            min_pose_presence_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
        )
        self.detector = mp_vision.PoseLandmarker.create_from_options(options)
        self.confidence_threshold = confidence_threshold
        self._mp = mp

    def predict(self, frame: np.ndarray, draw_overlay: bool = True) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        annotated = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        tracked: dict[str, dict[str, float]] = {}
        if not result.pose_landmarks:
            return annotated, tracked

        landmarks = result.pose_landmarks[0]
        height, width = frame.shape[:2]
        for index, point_name in MEDIAPIPE_KEYPOINT_NAMES.items():
            landmark = landmarks[index]
            visibility = float(getattr(landmark, "visibility", 1.0))
            if visibility < self.confidence_threshold:
                continue
            x_value = float(landmark.x * width)
            y_value = float(landmark.y * height)
            tracked[point_name] = {"x": x_value, "y": y_value, "confidence": visibility}
            if draw_overlay:
                cv2.circle(annotated, (int(x_value), int(y_value)), 4, (40, 220, 120), -1)

        if draw_overlay:
            for start_name, end_name in SKELETON_CONNECTIONS:
                if start_name not in tracked or end_name not in tracked:
                    continue
                start_point = tracked[start_name]
                end_point = tracked[end_name]
                cv2.line(
                    annotated,
                    (int(start_point["x"]), int(start_point["y"])),
                    (int(end_point["x"]), int(end_point["y"])),
                    (255, 170, 0),
                    2,
                )
        return annotated, tracked


class BarbellTracker:
    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold
        self.trajectory_points: deque[tuple[int, int]] = deque(maxlen=2500)
        self.missing_frames = 0

    def annotate(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        annotated = frame.copy()
        tracked: dict[str, dict[str, float]] = {}
        result = self.model.predict(source=frame, verbose=False, conf=self.confidence_threshold)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            self.missing_frames += 1
            self._draw_trajectory(annotated)
            return annotated, tracked

        best_confidence = -1.0
        best_center: tuple[float, float] | None = None
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        for coords, confidence_value in zip(xyxy, confidences):
            confidence = float(confidence_value)
            if confidence < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = coords.astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 90, 255), 2)
            cv2.putText(annotated, f"barbell {confidence:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 90, 255), 2)
            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)
            if confidence > best_confidence:
                best_confidence = confidence
                best_center = (center_x, center_y)

        if best_center is not None:
            tracked["barbell_center"] = {"x": best_center[0], "y": best_center[1], "confidence": best_confidence}
            self._append_trajectory_point((int(best_center[0]), int(best_center[1])))
            self._draw_trajectory(annotated)
            cv2.circle(annotated, (int(best_center[0]), int(best_center[1])), 5, (0, 90, 255), -1)
        else:
            self.missing_frames += 1
            self._draw_trajectory(annotated)
        return annotated, tracked

    def _draw_trajectory(self, frame: np.ndarray) -> None:
        if not self.trajectory_points:
            return
        for point in self.trajectory_points:
            point_x, point_y = point
            cv2.circle(frame, (point_x, point_y), 2, (0, 160, 255), -1)
        self._draw_trajectory_segment(frame, list(self.trajectory_points))

    def _draw_trajectory_segment(self, frame: np.ndarray, segment: list[tuple[int, int]]) -> None:
        if len(segment) < 2:
            return
        trajectory = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [trajectory], isClosed=False, color=(0, 160, 255), thickness=2)

    def _append_trajectory_point(self, point: tuple[int, int]) -> None:
        if not self.trajectory_points:
            self.trajectory_points.append(point)
            self.missing_frames = 0
            return

        if self.missing_frames > 0:
            last_x, last_y = self.trajectory_points[-1]
            for step in range(1, self.missing_frames + 1):
                factor = step / (self.missing_frames + 1)
                interpolated_x = int(round(last_x + (point[0] - last_x) * factor))
                interpolated_y = int(round(last_y + (point[1] - last_y) * factor))
                self.trajectory_points.append((interpolated_x, interpolated_y))

        self.trajectory_points.append(point)
        self.missing_frames = 0

    def clear_trajectory(self) -> None:
        self.trajectory_points.clear()
        self.missing_frames = 0


class AuxiliaryYoloDetector:
    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold

    def annotate(self, frame: np.ndarray, point_radius: int = 4) -> tuple[np.ndarray, list[str], dict[str, dict[str, float]]]:
        annotated = frame.copy()
        result = self.model.predict(source=frame, verbose=False, conf=self.confidence_threshold)[0]
        detections: list[str] = []
        tracked_points: dict[str, dict[str, float]] = {}
        point_radius = max(1, int(point_radius))
        label_offset = point_radius + 2
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(boxes.xyxy), dtype=int)
            confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(boxes.xyxy))
            for coords, class_id, confidence_value in zip(xyxy, class_ids, confidences):
                confidence = float(confidence_value)
                if confidence < self.confidence_threshold:
                    continue
                label = str(names.get(int(class_id), class_id))
                detections.append(f"{label} ({confidence:.2f})")
                x1, y1, x2, y2 = coords.astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 210, 255), 2)
                cv2.putText(annotated, f"{label} {confidence:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 2)

        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None and keypoints.xy is not None and len(keypoints.xy) > 0:
            xy_batches = keypoints.xy.cpu().numpy()
            if keypoints.conf is not None:
                confidence_batches = keypoints.conf.cpu().numpy()
            else:
                confidence_batches = [np.ones(len(points)) for points in xy_batches]

            for person_index, (xy_points, point_confidences) in enumerate(zip(xy_batches, confidence_batches), start=1):
                detections.append(f"pose #{person_index}")
                person_points = self._build_tracked_points(xy_points, point_confidences)
                if person_index == 1:
                    tracked_points.update(person_points)
                for index, (x_value, y_value) in enumerate(xy_points):
                    confidence = float(point_confidences[index]) if index < len(point_confidences) else 1.0
                    if confidence < self.confidence_threshold:
                        continue
                    point_name = self._resolve_keypoint_name(index, len(xy_points))
                    if point_name is None:
                        continue
                    cv2.circle(annotated, (int(x_value), int(y_value)), point_radius, (255, 80, 80), -1)
                    cv2.putText(
                        annotated,
                        point_name,
                        (int(x_value) + label_offset, int(y_value) - label_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 80, 80),
                        1,
                    )

                for start_name, end_name in self._resolve_connections(len(xy_points)):
                    if start_name not in person_points or end_name not in person_points:
                        continue
                    start_point = person_points[start_name]
                    end_point = person_points[end_name]
                    cv2.line(
                        annotated,
                        (int(start_point["x"]), int(start_point["y"])),
                        (int(end_point["x"]), int(end_point["y"])),
                        (255, 80, 80),
                        2,
                    )

        return annotated, detections, tracked_points

    def _build_tracked_points(self, xy_points: np.ndarray, point_confidences: np.ndarray) -> dict[str, dict[str, float]]:
        tracked_points: dict[str, dict[str, float]] = {}
        for index, (x_value, y_value) in enumerate(xy_points):
            confidence = float(point_confidences[index]) if index < len(point_confidences) else 1.0
            if confidence < self.confidence_threshold:
                continue
            point_name = self._resolve_keypoint_name(index, len(xy_points))
            if point_name is None:
                continue
            tracked_points[point_name] = {"x": float(x_value), "y": float(y_value), "confidence": confidence}
        return tracked_points

    def _resolve_keypoint_name(self, index: int, keypoint_count: int) -> str | None:
        if keypoint_count == len(AUXILIARY_KEYPOINT_NAMES):
            return AUXILIARY_KEYPOINT_NAMES.get(index)
        return COCO_KEYPOINT_NAMES.get(index)

    def _resolve_connections(self, keypoint_count: int) -> list[tuple[str, str]]:
        if keypoint_count == len(AUXILIARY_KEYPOINT_NAMES):
            return AUXILIARY_KEYPOINT_CONNECTIONS
        return SKELETON_CONNECTIONS


class PipelineModels:
    def __init__(self, pose_model: Any, barbell_tracker: BarbellTracker | None, auxiliary_detector: AuxiliaryYoloDetector | None) -> None:
        self.pose_model = pose_model
        self.barbell_tracker = barbell_tracker
        self.auxiliary_detector = auxiliary_detector

    def process(
        self,
        frame: np.ndarray,
        show_pose_overlay: bool = True,
        auxiliary_point_radius: int = 4,
    ) -> tuple[np.ndarray, dict[str, dict[str, float]], list[str]]:
        annotated, tracked = self.pose_model.predict(frame, draw_overlay=show_pose_overlay)
        if self.barbell_tracker is not None:
            annotated, barbell_points = self.barbell_tracker.annotate(annotated)
            tracked.update(barbell_points)
        side_predictions: list[str] = []
        if self.auxiliary_detector is not None:
            annotated, side_predictions, auxiliary_points = self.auxiliary_detector.annotate(
                annotated,
                point_radius=auxiliary_point_radius,
            )
            tracked.update(auxiliary_points)
        return annotated, tracked, side_predictions

    def clear_barbell_trajectory(self) -> None:
        if self.barbell_tracker is not None:
            self.barbell_tracker.clear_trajectory()


def create_pipeline_models(
    pose_selection: str,
    pose_confidence_threshold: float,
    use_barbell_tracking: bool,
    barbell_selection: str,
    barbell_confidence_threshold: float,
    use_auxiliary_yolo: bool,
    auxiliary_yolo_path: str,
    auxiliary_yolo_confidence_threshold: float,
) -> PipelineModels:
    validate_pose_selection(pose_selection)
    option = POSE_MODEL_OPTIONS[pose_selection]
    if option["backend"] == "yolo":
        pose_model = YoloPoseModel(Path(option["reference"]), pose_confidence_threshold)
    else:
        pose_model = MediaPipePoseModel(str(option["reference"]), pose_confidence_threshold)

    barbell_tracker = None
    if use_barbell_tracking:
        validate_barbell_selection(barbell_selection)
        barbell_tracker = BarbellTracker(BARBELL_MODEL_OPTIONS[barbell_selection], barbell_confidence_threshold)

    auxiliary_detector = None
    if use_auxiliary_yolo:
        model_path = validate_aux_yolo_path(auxiliary_yolo_path)
        auxiliary_detector = AuxiliaryYoloDetector(model_path, auxiliary_yolo_confidence_threshold)
    return PipelineModels(pose_model, barbell_tracker, auxiliary_detector)


def ensure_mediapipe_model_file(variant: str) -> Path:
    if variant not in MEDIAPIPE_MODEL_URLS:
        raise ValueError(f"Unsupported MediaPipe variant: {variant}")
    MEDIAPIPE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_url = MEDIAPIPE_MODEL_URLS[variant]
    model_path = MEDIAPIPE_MODEL_DIR / Path(model_url).name
    if model_path.exists():
        return model_path
    urlretrieve(model_url, model_path)
    return model_path


def _load_mediapipe_modules() -> tuple[Any, Any, Any]:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as exc:
        raise RuntimeError(
            "Failed to load MediaPipe. Select a YOLO model or fix the "
            "MediaPipe/TensorFlow/protobuf installation in the environment."
        ) from exc
    return mp, mp_python, mp_vision
