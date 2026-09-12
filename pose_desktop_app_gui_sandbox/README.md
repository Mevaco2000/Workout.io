# Desktop System for Strength Training Monitoring and Analysis

Desktop application for video/camera analysis based on OpenCV, YOLO and MediaPipe.

## Features
- video file input or live camera input
- pose model selection: YOLO v26 n/s/m/l or MediaPipe Lite/Full/Heavy
- optional barbell tracking with selectable model
- optional additional YOLO model loaded from `.pt`
- pause, resume and cancel processing
- live preview scaling and aspect ratio changes
- JSONL keypoint export
- time charts after pause or after run
- repetition counting for: deadlift, squat, OHP, rows, lateral raises, bicep curls, pushups

## Reproducible Setup From Git Clone

1. Clone repository:

```powershell
git clone https://github.com/Mevaco2000/Workout.io.git
cd Workout.io
```

2. Create and activate virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. Download optional pose and MediaPipe models:

```powershell
python download_optional_models.py
```

5. Run application:

```powershell
python run_pose_desktop_app_gui_sandbox.py
```

## Assets (Splash and Info)

- UI icons and splash/info graphics are loaded from `apka_treningowa/` in repository root.
- If this folder is not present in the repo checkout, application falls back to `../apka_treningowa/`.

## Notes

- Runtime outputs are saved in `pose_desktop_runtime/`.
- If auto-download via ultralytics fails, set direct URLs in `YOLO_POSE_SOURCES` in `download_optional_models.py`.
