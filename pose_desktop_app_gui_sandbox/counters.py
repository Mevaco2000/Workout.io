from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, sqrt
from typing import Iterable


Point = tuple[float, float]


def average_y(sample: dict[str, float], names: Iterable[str]) -> float | None:
    values = [sample.get(f"{name}_y") for name in names]
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def average_x(sample: dict[str, float], names: Iterable[str]) -> float | None:
    values = [sample.get(f"{name}_x") for name in names]
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def point(sample: dict[str, float], name: str) -> Point | None:
    x_value = sample.get(f"{name}_x")
    y_value = sample.get(f"{name}_y")
    if x_value is None or y_value is None:
        return None
    return (x_value, y_value)


def angle(a: Point | None, b: Point | None, c: Point | None) -> float | None:
    if a is None or b is None or c is None:
        return None

    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    ab_len = sqrt(ab[0] ** 2 + ab[1] ** 2)
    cb_len = sqrt(cb[0] ** 2 + cb[1] ** 2)
    if ab_len == 0 or cb_len == 0:
        return None

    cosine = (ab[0] * cb[0] + ab[1] * cb[1]) / (ab_len * cb_len)
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))


@dataclass
class ExerciseRepCounter:
    exercise: str
    reps: int = 0
    phase: str = "start"
    min_metric: float | None = None
    max_metric: float | None = None

    def update(self, sample: dict[str, float]) -> int:
        metric = self._metric(sample)
        if metric is None:
            return self.reps

        self.min_metric = metric if self.min_metric is None else min(self.min_metric, metric)
        self.max_metric = metric if self.max_metric is None else max(self.max_metric, metric)
        span = (self.max_metric - self.min_metric) if self.min_metric is not None and self.max_metric is not None else 0.0
        if span < self._minimum_span():
            return self.reps

        normalized = (metric - self.min_metric) / span
        high_is_work = self.exercise in {"deadlift", "squat", "pushups"}

        if high_is_work:
            if normalized >= 0.68:
                self.phase = "work"
            elif self.phase == "work" and normalized <= 0.34:
                self.reps += 1
                self.phase = "return"
        else:
            if normalized <= 0.32:
                self.phase = "work"
            elif self.phase == "work" and normalized >= 0.66:
                self.reps += 1
                self.phase = "return"

        return self.reps

    def _minimum_span(self) -> float:
        if self.exercise in {"bicep curls"}:
            return 12.0
        if self.exercise in {"OHP", "rows", "lateral raises"}:
            return 20.0
        return 25.0

    def _metric(self, sample: dict[str, float]) -> float | None:
        if self.exercise == "deadlift":
            return average_y(sample, ["left_hip", "right_hip"])
        if self.exercise == "squat":
            return average_y(sample, ["left_hip", "right_hip"])
        if self.exercise == "pushups":
            return average_y(sample, ["left_shoulder", "right_shoulder"])
        if self.exercise == "OHP":
            shoulder_y = average_y(sample, ["left_shoulder", "right_shoulder"])
            wrist_y = average_y(sample, ["left_wrist", "right_wrist"])
            if shoulder_y is None or wrist_y is None:
                return None
            return wrist_y - shoulder_y
        if self.exercise == "rows":
            hip_y = average_y(sample, ["left_hip", "right_hip"])
            wrist_y = average_y(sample, ["left_wrist", "right_wrist"])
            if hip_y is None or wrist_y is None:
                return None
            return wrist_y - hip_y
        if self.exercise == "lateral raises":
            shoulder_y = average_y(sample, ["left_shoulder", "right_shoulder"])
            wrist_y = average_y(sample, ["left_wrist", "right_wrist"])
            if shoulder_y is None or wrist_y is None:
                return None
            return wrist_y - shoulder_y
        if self.exercise == "bicep curls":
            left_angle = angle(point(sample, "left_shoulder"), point(sample, "left_elbow"), point(sample, "left_wrist"))
            right_angle = angle(point(sample, "right_shoulder"), point(sample, "right_elbow"), point(sample, "right_wrist"))
            angles = [value for value in [left_angle, right_angle] if value is not None]
            if not angles:
                return None
            return sum(angles) / len(angles)
        return average_y(sample, ["left_hip", "right_hip"])
