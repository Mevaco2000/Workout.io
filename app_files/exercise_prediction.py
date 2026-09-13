from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .sequence_runtime import (
    CNN1DClassifier,
    GRUClassifier,
    LSTMClassifier,
    MLPClassifier,
    STGCNClassifier,
    TransformerEncoderClassifier,
    apply_standardization,
    normalize_xy_positions,
)

from .config import get_default_exercise_classifier_model


BODY_POINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

EXERCISE_NAME_MAP = {
    "deadlift": "deadlift",
    "squat": "squat",
    "OHP": "OHP",
    "rows": "rows",
    "Lateral_raises": "lateral raises",
    "lateral raises": "lateral raises",
    "Bicep_curls": "bicep curls",
    "bicep curls": "bicep curls",
}


def normalize_exercise_name(name: str) -> str:
    if name in EXERCISE_NAME_MAP:
        return EXERCISE_NAME_MAP[name]
    return name.replace("_", " ").strip()


MODEL_CLASS_BUILDERS = {
    "mlp": MLPClassifier,
    "cnn1d": CNN1DClassifier,
    "gru": GRUClassifier,
    "lstm": LSTMClassifier,
    "transformer_encoder": TransformerEncoderClassifier,
    "stgcn": STGCNClassifier,
}


class ExercisePredictor:
    def __init__(self, model_path: Path, smoothing_window: int = 8) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Exercise auto-classification model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        class_names = list(checkpoint["class_names"])
        model = self._build_model(checkpoint, len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names
        self.mean = np.array(checkpoint["normalization_mean"], dtype=np.float32)
        self.std = np.array(checkpoint["normalization_std"], dtype=np.float32)
        self.position_normalization_mode = str(checkpoint.get("position_normalization_mode", "none"))
        self.model_type = str(checkpoint.get("model_type", "lstm"))
        self.data_format = str(checkpoint.get("data_format", "sequence"))
        self.feature_indices = checkpoint.get("feature_indices")
        if self.feature_indices is not None:
            self.feature_indices = tuple(int(index) for index in self.feature_indices)

        input_shape = checkpoint.get("input_shape")
        if input_shape is not None:
            self.sequence_length = int(input_shape[0])
        else:
            self.sequence_length = int(checkpoint["sequence_length"])
        self.sequence_buffer: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.probability_buffer: deque[np.ndarray] = deque(maxlen=max(1, int(smoothing_window)))

    def _build_model(self, checkpoint: dict[str, object], num_classes: int) -> torch.nn.Module:
        model_type = str(checkpoint.get("model_type", "lstm"))
        model_config = checkpoint.get("model_config")
        if isinstance(model_config, dict) and model_type in MODEL_CLASS_BUILDERS:
            return MODEL_CLASS_BUILDERS[model_type](**model_config)

        if model_type == "gru":
            return GRUClassifier(
                input_size=int(checkpoint["input_size"]),
                hidden_size=int(checkpoint["hidden_size"]),
                num_layers=int(checkpoint["num_layers"]),
                num_classes=num_classes,
            )

        return LSTMClassifier(
            input_size=int(checkpoint["input_size"]),
            hidden_size=int(checkpoint["hidden_size"]),
            num_layers=int(checkpoint["num_layers"]),
            num_classes=num_classes,
        )

    def update(self, tracked_points: dict[str, dict[str, float]]) -> dict[str, object] | None:
        self.sequence_buffer.append(self._vectorize_points(tracked_points))
        if len(self.sequence_buffer) < self.sequence_length:
            return None

        sequence = np.stack(self.sequence_buffer).astype(np.float32)
        normalized = apply_standardization(sequence, self.mean, self.std)
        inputs = torch.from_numpy(normalized[None, ...]).to(self.device)

        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)

        self.probability_buffer.append(probabilities)
        smoothed_probabilities = np.mean(np.stack(self.probability_buffer), axis=0)
        best_index = int(np.argmax(smoothed_probabilities))
        ranked_indices = np.argsort(smoothed_probabilities)[::-1][:3]

        top_predictions = [
            f"{normalize_exercise_name(self.class_names[int(index)])} {float(smoothed_probabilities[int(index)]):.0%}"
            for index in ranked_indices
        ]
        return {
            "exercise": normalize_exercise_name(self.class_names[best_index]),
            "confidence": float(smoothed_probabilities[best_index]),
            "top_predictions": top_predictions,
        }

    def _vectorize_points(self, tracked_points: dict[str, dict[str, float]]) -> np.ndarray:
        values: list[float] = []
        for point_name in BODY_POINT_NAMES:
            point = tracked_points.get(point_name)
            if point is None:
                values.extend([0.0, 0.0, 0.0])
                continue
            values.extend(
                [
                    float(point.get("x", 0.0)),
                    float(point.get("y", 0.0)),
                    float(point.get("confidence", 0.0)),
                ]
            )
        keypoints = np.array(values, dtype=np.float32).reshape(len(BODY_POINT_NAMES), 3)
        normalized_keypoints = normalize_xy_positions(
            keypoints[None, ...],
            self.position_normalization_mode,
        )[0]
        if self.feature_indices is not None:
            normalized_keypoints = normalized_keypoints[:, list(self.feature_indices)]
        if self.data_format == "graph":
            return normalized_keypoints.astype(np.float32)
        return normalized_keypoints.reshape(-1).astype(np.float32)


def create_exercise_predictor(
    exercise_mode: str,
    model_path: str | Path | None = None,
    smoothing_window: int = 8,
) -> ExercisePredictor | None:
    if not exercise_mode.startswith("auto"):
        return None

    resolved_model_path = Path(model_path) if model_path is not None else get_default_exercise_classifier_model()
    if resolved_model_path is None:
        raise FileNotFoundError("No model found for exercise auto-classification.")
    return ExercisePredictor(model_path=resolved_model_path, smoothing_window=smoothing_window)