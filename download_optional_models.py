from __future__ import annotations

from pathlib import Path
from shutil import copy2
from urllib.request import urlretrieve


ROOT_DIR = Path(__file__).resolve().parent
DOWNLOAD_YOLO_POSE_MODELS = True
DOWNLOAD_MEDIAPIPE_MODELS = True
OVERWRITE_EXISTING_FILES = False

YOLO_POSE_SOURCES = {
    "yolo26n-pose.pt": "yolo26n-pose.pt",
    "yolo26s-pose.pt": "yolo26s-pose.pt",
    "yolo26m-pose.pt": "yolo26m-pose.pt",
    "yolo26l-pose.pt": "yolo26l-pose.pt",
}

MEDIAPIPE_MODEL_URLS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def download_file(url: str, target_path: Path, overwrite: bool) -> None:
    if target_path.exists() and not overwrite:
        print(f"Skipping existing file: {target_path.name}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {target_path.name} from URL...")
    urlretrieve(url, target_path)
    print(f"Saved: {target_path}")


def download_yolo_pose_model(source: str, target_path: Path, overwrite: bool) -> None:
    if target_path.exists() and not overwrite:
        print(f"Skipping existing file: {target_path.name}")
        return

    if source.startswith(("https://", "http://")):
        download_file(source, target_path, overwrite)
        return

    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required to auto-download YOLO pose models. "
            "Install dependencies from requirements.txt or replace YOLO_POSE_SOURCES values with direct URLs."
        ) from exc

    print(f"Resolving YOLO pose model: {source}")
    resolved_path = Path(attempt_download_asset(source))
    if not resolved_path.exists():
        raise FileNotFoundError(f"Ultralytics did not provide the requested model: {source}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(resolved_path, target_path)
    print(f"Saved: {target_path}")


def main() -> int:
    try:
        if DOWNLOAD_YOLO_POSE_MODELS:
            for file_name, source in YOLO_POSE_SOURCES.items():
                download_yolo_pose_model(source, ROOT_DIR / file_name, OVERWRITE_EXISTING_FILES)

        if DOWNLOAD_MEDIAPIPE_MODELS:
            mediapipe_dir = ROOT_DIR / "mediapipe_models"
            for file_name, url in MEDIAPIPE_MODEL_URLS.items():
                download_file(url, mediapipe_dir / file_name, OVERWRITE_EXISTING_FILES)

    except Exception as exc:
        print(f"Download failed: {exc}")
        return 1

    print("Optional model download finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())