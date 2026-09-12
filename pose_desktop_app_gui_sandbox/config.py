from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = ROOT_DIR / "pose_desktop_runtime"

POSE_MODEL_OPTIONS = {
    "YOLO v26 n": {"backend": "yolo", "reference": ROOT_DIR / "yolo26n-pose.pt"},
    "YOLO v26 s": {"backend": "yolo", "reference": ROOT_DIR / "yolo26s-pose.pt"},
    "YOLO v26 m": {"backend": "yolo", "reference": ROOT_DIR / "yolo26m-pose.pt"},
    "YOLO v26 l": {"backend": "yolo", "reference": ROOT_DIR / "yolo26l-pose.pt"},
}

MEDIAPIPE_MODEL_URLS = {
    "MediaPipe Lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "MediaPipe Medium": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "MediaPipe Heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

MEDIAPIPE_MODEL_DIR = ROOT_DIR / "mediapipe_models"

BARBELL_MODEL_OPTIONS = {
    "Barbell tracker YOLO v26 n": ROOT_DIR / "barbell_yolo26n.pt",
}

DEFAULT_AUX_YOLO_MODEL_PATH = ROOT_DIR / "model_spinal_9pts.pt"
DEFAULT_EXERCISE_LSTM_MODEL_PATH = ROOT_DIR / "artifacts" / "exercise_lstm" / "exercise_lstm.pt"
DEFAULT_EXERCISE_LSTM_BBOX_MODEL_PATH = ROOT_DIR / "artifacts" / "exercise_lstm" / "exercise_lstm_normalized.pt"
AUTO_EXERCISE_MODEL_SEARCH_DIRS = [
    ROOT_DIR / "artifacts" / "all_sequence_models_benchmark",
    ROOT_DIR / "artifacts" / "exercise_lstm",
    ROOT_DIR / "artifacts" / "exercise_gru",
]

EXERCISES = [
    "auto",
    "deadlift",
    "squat",
    "OHP",
    "rows",
    "lateral raises",
    "bicep curls",
]

DISPLAY_ASPECT_OPTIONS = {
    "Original": None,
    "16:9": (960, 540),
    "4:3": (960, 720),
    "1:1": (720, 720),
}

DISPLAY_SCALE_OPTIONS = {
    "75%": 0.75,
    "100%": 1.0,
    "125%": 1.25,
    "150%": 1.5,
}

COCO_KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

MEDIAPIPE_KEYPOINT_NAMES = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

AUXILIARY_KEYPOINT_MODELS = {
    "4-point": {
        "names": {
            0: "neck_end",
            1: "shoulder_blade",
            2: "lumbar_start",
            3: "lumbar_end",
        },
        "connections": [
            ("neck_end", "shoulder_blade"),
            ("shoulder_blade", "lumbar_start"),
            ("lumbar_start", "lumbar_end"),
        ],
    },
    "9-point": {
        "names": {
            0: "neck_start",
            1: "neck_end",
            2: "thoracic_start",
            3: "thoracic_1",
            4: "thoracic_2",
            5: "thoracic_end",
            6: "lumbar_start",
            7: "lumbar_1",
            8: "lumbar_end",
        },
        "connections": [
            ("neck_start", "neck_end"),
            ("neck_end", "thoracic_start"),
            ("thoracic_start", "thoracic_1"),
            ("thoracic_1", "thoracic_2"),
            ("thoracic_2", "thoracic_end"),
            ("thoracic_end", "lumbar_start"),
            ("lumbar_start", "lumbar_1"),
            ("lumbar_1", "lumbar_end"),
        ],
    },
}

AUXILIARY_KEYPOINT_MODEL_OPTIONS = list(AUXILIARY_KEYPOINT_MODELS.keys())
DEFAULT_AUXILIARY_KEYPOINT_MODEL = "4-point"

CHART_POINTS = [
    "nose",
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
    "barbell_center",
    "neck_start",
    "neck_end",
    "shoulder_blade",
    "thoracic_start",
    "thoracic_1",
    "thoracic_2",
    "thoracic_end",
    "lumbar_start",
    "lumbar_1",
    "lumbar_end",
]

BARBELL_SPEED_CHART = "barbell_speed"
BARBELL_POSITION_CHART = "barbell_position"
CHART_OPTIONS = CHART_POINTS + [BARBELL_SPEED_CHART, BARBELL_POSITION_CHART]


def get_auxiliary_keypoint_schema(selection: str) -> tuple[dict[int, str], list[tuple[str, str]]]:
    schema = AUXILIARY_KEYPOINT_MODELS.get(selection) or AUXILIARY_KEYPOINT_MODELS[DEFAULT_AUXILIARY_KEYPOINT_MODEL]
    names = dict(schema["names"])
    connections = list(schema["connections"])
    return names, connections


def discover_exercise_classifier_models() -> dict[str, Path]:
    discovered_models: dict[str, Path] = {}
    for base_dir in AUTO_EXERCISE_MODEL_SEARCH_DIRS:
        if not base_dir.exists():
            continue
        for model_path in sorted(base_dir.rglob("*.pt")):
            label = model_path.relative_to(ROOT_DIR).as_posix()
            discovered_models[label] = model_path
    return discovered_models


def get_default_exercise_classifier_model() -> Path | None:
    preferred_paths = [
        ROOT_DIR / "artifacts" / "all_sequence_models_benchmark" / "lstm" / "lstm_after_eval.pt",
        ROOT_DIR / "artifacts" / "all_sequence_models_benchmark" / "lstm" / "lstm.pt",
        ROOT_DIR / "artifacts" / "exercise_lstm" / "exercise_lstm_normalized_last.pt",
        ROOT_DIR / "artifacts" / "exercise_lstm" / "exercise_lstm_normalized.pt",
        ROOT_DIR / "artifacts" / "exercise_gru" / "exercise_gru.pt",
        ROOT_DIR / "artifacts" / "exercise_lstm" / "exercise_lstm.pt",
    ]
    for model_path in preferred_paths:
        if model_path.exists():
            return model_path

    discovered_models = discover_exercise_classifier_models()
    if not discovered_models:
        return None
    return next(iter(discovered_models.values()))
