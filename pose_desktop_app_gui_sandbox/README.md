# Pose Desktop App

Nowa aplikacja desktopowa do analizy wideo i kamerki z wykorzystaniem OpenCV, YOLO v26 i MediaPipe.

## Funkcje
- wczytanie pliku wideo lub start z kamerki
- wybor modelu pose: YOLO v26 n/s/m/l lub MediaPipe Lite/Full/Heavy
- opcjonalne sledzenie sztangi z wyborem modelu
- opcjonalny dodatkowy model YOLO wskazywany z pliku `.pt`, nanoszony bezposrednio na podglad; dla modeli pose rysowane sa tez keypointy i szkielet
- pause, wznowienie i anulowanie przetwarzania
- zmiana proporcji i rozmiaru podgladu podczas pracy
- zapis punktow do pliku `jsonl` na dysku
- wykresy czasowe punktow po pauzie lub po zakonczeniu
- liczenie powtorzen dla cwiczen: deadlift, squat, OHP, rows, lateral raises, bicep curls, pushups

## Start
```powershell
pip install -r requirements.txt
python download_optional_models.py
python run_pose_desktop_app_gui_sandbox.py
```

## Uwagi
- W paczce na stale zostaja tylko `yolo26_barbell_tracker_no_validation.pt` oraz `spinal_s.pt`.
- Modele YOLO pose i MediaPipe mozna pobrac pozniej przez `python download_optional_models.py`.
- Jesli automatyczne pobieranie YOLO pose przez `ultralytics` nie zadziala, wpisz bezposrednie URL-e w `YOLO_POSE_SOURCES` na gorze skryptu.
- Jesli dodatkowy model pose zwraca 4 keypointy, sa one sledzone i zapisywane jako: `koniec_szyi`, `lopatka`, `poczatek_odcinka_ledzwiowego`, `koniec_odcinka_ledzwiowego`.
- Punkty sa zapisywane w katalogu `pose_desktop_runtime/`.
