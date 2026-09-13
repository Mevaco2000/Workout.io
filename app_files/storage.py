from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RUNTIME_ROOT


class JsonlPointStorage:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = RUNTIME_ROOT / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "points.jsonl"
        self.barbell_path = self.session_dir / "barbell_predictions.jsonl"
        self._file = self.path.open("a", encoding="utf-8")
        self._barbell_file = self.barbell_path.open("a", encoding="utf-8")

    def append(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._file.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._file.flush()

    def append_barbell_prediction(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._barbell_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._barbell_file.flush()

    def load_dataframe(self) -> pd.DataFrame:
        with self._lock:
            self._file.flush()
            if not self.path.exists() or self.path.stat().st_size == 0:
                return pd.DataFrame()
            return pd.read_json(self.path, lines=True)

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()
            if not self._barbell_file.closed:
                self._barbell_file.flush()
                self._barbell_file.close()
