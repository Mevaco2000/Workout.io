from __future__ import annotations

from collections import deque
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
    preferred_arm_side: str | None = None
    smoothed_metric: float | None = None
    metric_window: deque[float] | None = None
    ohp_cycle_min: float | None = None
    ohp_cycle_max: float | None = None
    last_metric: float | None = None
    last_span: float | None = None
    last_normalized: float | None = None
    last_down_threshold: float | None = None
    last_up_threshold: float | None = None
    last_elbow_angle: float | None = None
    last_deadlift_aux_angle: float | None = None
    deadlift_cooldown_frames: int = 0
    # Angle-based deadlift phase tracking: instead of fixed y-position/angle
    # thresholds, a rep is detected purely from how the hip (torso) angle rises
    # and falls relative to its own recent top/bottom - this adapts to each
    # lifter's actual range of motion instead of relying on an absolute degree
    # cutoff that may never be reached (or reached too easily) for everyone.
    deadlift_angle_smoothed: float | None = None
    deadlift_last_smoothed_angle: float | None = None
    deadlift_angle_state: str = "top"
    deadlift_ref_top_angle: float | None = None
    deadlift_min_angle: float | None = None
    deadlift_rising_streak: int = 0
    deadlift_falling_streak: int = 0
    rows_cycle_min: float | None = None
    rows_cycle_max: float | None = None
    rows_prework_window: deque[float] | None = None
    rows_cooldown_frames: int = 0
    rows_work_hold_frames: int = 0
    rows_return_hold_frames: int = 0
    lateral_cycle_min: float | None = None
    lateral_cycle_max: float | None = None
    lateral_prework_window: deque[float] | None = None
    lateral_cooldown_frames: int = 0
    lateral_work_hold_frames: int = 0
    lateral_return_hold_frames: int = 0

    def update(self, sample: dict[str, float]) -> int:
        if self.exercise == "deadlift":
            return self._update_deadlift(sample)

        if self.rows_cooldown_frames > 0:
            self.rows_cooldown_frames -= 1
        if self.lateral_cooldown_frames > 0:
            self.lateral_cooldown_frames -= 1

        metric = self._metric(sample)
        if metric is None:
            self.last_metric = None
            self.last_span = None
            self.last_normalized = None
            self.last_elbow_angle = None
            self.last_deadlift_aux_angle = None
            self.rows_work_hold_frames = 0
            self.rows_return_hold_frames = 0
            self.lateral_work_hold_frames = 0
            self.lateral_return_hold_frames = 0
            return self.reps

        if self.exercise == "OHP":
            metric = self._smooth_metric(metric)
            # Use the range established by *previous* frames (not this one) so that
            # a sample which is itself a new extreme doesn't trivially normalize to
            # exactly 0/1 every time - it still shows real progress toward that edge.
            if self.ohp_cycle_min is None or self.ohp_cycle_max is None:
                self.ohp_cycle_min = metric
                self.ohp_cycle_max = metric
            self.min_metric = self.ohp_cycle_min
            self.max_metric = self.ohp_cycle_max
            # Extend the range unconditionally (not gated behind the span check
            # below), otherwise the range could never grow past its first sample.
            self.ohp_cycle_min = min(self.ohp_cycle_min, metric)
            self.ohp_cycle_max = max(self.ohp_cycle_max, metric)
        elif self.exercise == "rows":
            metric = self._smooth_metric(metric)
            if self.phase == "work":
                if self.rows_cycle_min is None or self.rows_cycle_max is None:
                    self.rows_cycle_min = metric
                    self.rows_cycle_max = metric
                else:
                    self.rows_cycle_min = min(self.rows_cycle_min, metric)
                    self.rows_cycle_max = max(self.rows_cycle_max, metric)
                self.min_metric = self.rows_cycle_min
                self.max_metric = self.rows_cycle_max
            else:
                if self.rows_prework_window is None:
                    self.rows_prework_window = deque(maxlen=24)
                self.rows_prework_window.append(metric)
                self.min_metric = min(self.rows_prework_window)
                self.max_metric = max(self.rows_prework_window)
        elif self.exercise == "lateral raises":
            metric = self._smooth_metric(metric)
            if self.phase == "work":
                if self.lateral_cycle_min is None or self.lateral_cycle_max is None:
                    self.lateral_cycle_min = metric
                    self.lateral_cycle_max = metric
                else:
                    self.lateral_cycle_min = min(self.lateral_cycle_min, metric)
                    self.lateral_cycle_max = max(self.lateral_cycle_max, metric)
                self.min_metric = self.lateral_cycle_min
                self.max_metric = self.lateral_cycle_max
            else:
                if self.lateral_prework_window is None:
                    self.lateral_prework_window = deque(maxlen=24)
                self.lateral_prework_window.append(metric)
                self.min_metric = min(self.lateral_prework_window)
                self.max_metric = max(self.lateral_prework_window)
        else:
            self.min_metric = metric if self.min_metric is None else min(self.min_metric, metric)
            self.max_metric = metric if self.max_metric is None else max(self.max_metric, metric)

        span = (self.max_metric - self.min_metric) if self.min_metric is not None and self.max_metric is not None else 0.0
        self.last_metric = metric
        self.last_span = span
        if span < self._minimum_span():
            self.last_normalized = None
            return self.reps

        normalized = (metric - self.min_metric) / span
        self.last_normalized = normalized
        high_is_work = self.exercise in {"squat", "pushups"}

        if high_is_work:
            self.last_deadlift_aux_angle = None
            if normalized >= 0.68:
                self.phase = "work"
            elif self.phase == "work" and normalized <= 0.34:
                self.reps += 1
                self.phase = "return"
        else:
            down_threshold = 0.32
            up_threshold = 0.66
            if self.exercise == "OHP":
                down_threshold = 0.40
                up_threshold = 0.58

                self.last_down_threshold = down_threshold
                self.last_up_threshold = up_threshold

                # No separate "arming" step: as soon as a real range (span) is
                # established, react to the down/up thresholds directly. Requiring
                # a return-to-rack before arming used to swallow the very first rep.
                if normalized <= down_threshold:
                    self.phase = "work"
                elif self.phase == "work" and normalized >= up_threshold:
                    self.reps += 1
                    self.phase = "return"
                    # Data was held since the last rep; start a fresh range for the next one.
                    self.ohp_cycle_min = metric
                    self.ohp_cycle_max = metric
                return self.reps

            self.last_down_threshold = down_threshold
            self.last_up_threshold = up_threshold

            if self.exercise == "rows":
                elbow_angle = self._selected_elbow_angle(sample)
                self.last_elbow_angle = elbow_angle
                if elbow_angle is None:
                    self.rows_work_hold_frames = 0
                    self.rows_return_hold_frames = 0
                    return self.reps

                rows_down_angle_max = 145.0
                rows_up_angle_min = 160.0
                required_work_frames = 2
                required_return_frames = 2

                if self.phase != "work" and self.rows_cooldown_frames == 0 and normalized <= down_threshold and elbow_angle <= rows_down_angle_max:
                    self.rows_work_hold_frames += 1
                else:
                    self.rows_work_hold_frames = 0

                if self.rows_work_hold_frames >= required_work_frames:
                    self.rows_cycle_min = metric
                    self.rows_cycle_max = metric
                    self.phase = "work"
                    self.rows_work_hold_frames = 0

                if self.phase == "work" and normalized >= up_threshold and elbow_angle >= rows_up_angle_min:
                    self.rows_return_hold_frames += 1
                else:
                    self.rows_return_hold_frames = 0

                if self.phase == "work" and self.rows_return_hold_frames >= required_return_frames and self.rows_cooldown_frames == 0:
                    self.reps += 1
                    self.phase = "return"
                    self.rows_cooldown_frames = 8
                    self.rows_return_hold_frames = 0
                    self.rows_cycle_min = None
                    self.rows_cycle_max = None
                    self.rows_prework_window = deque([metric], maxlen=24)
                    self.smoothed_metric = None
                return self.reps

            if self.exercise == "lateral raises":
                elbow_angle = self._selected_elbow_angle(sample)
                self.last_elbow_angle = elbow_angle
                if elbow_angle is None:
                    self.lateral_work_hold_frames = 0
                    self.lateral_return_hold_frames = 0
                    return self.reps

                # Distinguish true lateral raise from elbow flexion-driven wrist motion.
                lateral_up_threshold = 0.74
                self.last_up_threshold = lateral_up_threshold
                lateral_min_straight_angle = 152.0
                required_work_frames = 2
                required_return_frames = 2

                if self.phase != "work" and self.lateral_cooldown_frames == 0 and normalized <= down_threshold and elbow_angle >= lateral_min_straight_angle:
                    self.lateral_work_hold_frames += 1
                else:
                    self.lateral_work_hold_frames = 0

                if self.lateral_work_hold_frames >= required_work_frames:
                    self.lateral_cycle_min = metric
                    self.lateral_cycle_max = metric
                    self.phase = "work"
                    self.lateral_work_hold_frames = 0

                if self.phase == "work" and normalized >= lateral_up_threshold and elbow_angle >= lateral_min_straight_angle:
                    self.lateral_return_hold_frames += 1
                else:
                    self.lateral_return_hold_frames = 0

                if self.phase == "work" and self.lateral_return_hold_frames >= required_return_frames and self.lateral_cooldown_frames == 0:
                    self.reps += 1
                    self.phase = "return"
                    self.lateral_cooldown_frames = 8
                    self.lateral_return_hold_frames = 0
                    self.lateral_cycle_min = None
                    self.lateral_cycle_max = None
                    self.lateral_prework_window = deque([metric], maxlen=24)
                    self.smoothed_metric = None
                return self.reps

            self.last_elbow_angle = None

            if normalized <= down_threshold:
                self.phase = "work"
            elif self.phase == "work" and normalized >= up_threshold:
                self.reps += 1
                self.phase = "return"

        return self.reps

    def debug_snapshot(self) -> dict[str, float | str | bool | None]:
        return {
            "exercise": self.exercise,
            "phase": self.phase,
            "reps": float(self.reps),
            "preferred_side": self.preferred_arm_side,
            "metric": self.last_metric,
            "span": self.last_span,
            "normalized": self.last_normalized,
            "down_threshold": self.last_down_threshold,
            "up_threshold": self.last_up_threshold,
            "elbow_angle": self.last_elbow_angle,
            "deadlift_aux_angle": self.last_deadlift_aux_angle,
            "deadlift_angle_state": self.deadlift_angle_state,
            "deadlift_ref_top_angle": self.deadlift_ref_top_angle,
            "deadlift_min_angle": self.deadlift_min_angle,
            "deadlift_rising_streak": float(self.deadlift_rising_streak),
            "deadlift_falling_streak": float(self.deadlift_falling_streak),
            "rows_cooldown_frames": float(self.rows_cooldown_frames),
            "rows_work_hold_frames": float(self.rows_work_hold_frames),
            "rows_return_hold_frames": float(self.rows_return_hold_frames),
            "lateral_cooldown_frames": float(self.lateral_cooldown_frames),
            "lateral_work_hold_frames": float(self.lateral_work_hold_frames),
            "lateral_return_hold_frames": float(self.lateral_return_hold_frames),
        }

    def _smooth_metric(self, metric: float) -> float:
        if self.smoothed_metric is None:
            self.smoothed_metric = metric
            return metric
        alpha = 0.35
        self.smoothed_metric = alpha * metric + (1.0 - alpha) * self.smoothed_metric
        return self.smoothed_metric

    def _update_deadlift(self, sample: dict[str, float]) -> int:
        """Detect deadlift reps from the hip (torso) angle and its direction of
        change, instead of fixed absolute angle/position thresholds.

        The hip angle (shoulder-hip-knee) is smoothed, then classified into a
        rising or falling streak. A rep only starts counting once the angle has
        genuinely folded down by a meaningful amount relative to the lifter's
        own recent top position (not an absolute degree cutoff), and only
        completes once the angle has climbed back close to that same top - so
        the criterion adapts to each lifter's own range of motion instead of
        requiring everyone to reach e.g. exactly 160 degrees of lockout.
        """
        if self.deadlift_cooldown_frames > 0:
            self.deadlift_cooldown_frames -= 1

        raw_angle = self._deadlift_aux_angle(sample)
        self.last_deadlift_aux_angle = raw_angle
        if raw_angle is None:
            self.deadlift_rising_streak = 0
            self.deadlift_falling_streak = 0
            return self.reps

        alpha = 0.35
        if self.deadlift_angle_smoothed is None:
            self.deadlift_angle_smoothed = raw_angle
        else:
            self.deadlift_angle_smoothed = alpha * raw_angle + (1.0 - alpha) * self.deadlift_angle_smoothed
        angle_value = self.deadlift_angle_smoothed

        previous_angle = self.deadlift_last_smoothed_angle
        self.deadlift_last_smoothed_angle = angle_value

        # Ignore sub-noise jitter so a single flickering frame can't flip the
        # rising/falling streak back and forth.
        min_move = 0.3
        if previous_angle is not None:
            delta = angle_value - previous_angle
            if delta <= -min_move:
                self.deadlift_falling_streak += 1
                self.deadlift_rising_streak = 0
            elif delta >= min_move:
                self.deadlift_rising_streak += 1
                self.deadlift_falling_streak = 0

        required_streak = 3
        min_depth_drop = 15.0
        return_tolerance = 10.0

        if self.deadlift_angle_state == "top":
            self.deadlift_ref_top_angle = angle_value if self.deadlift_ref_top_angle is None else max(self.deadlift_ref_top_angle, angle_value)

            if self.deadlift_falling_streak >= required_streak and (self.deadlift_ref_top_angle - angle_value) >= min_depth_drop / 2:
                self.deadlift_angle_state = "descending"
                self.deadlift_min_angle = angle_value
                self.phase = "work"

        elif self.deadlift_angle_state == "descending":
            self.deadlift_min_angle = angle_value if self.deadlift_min_angle is None else min(self.deadlift_min_angle, angle_value)

            if self.deadlift_rising_streak >= required_streak:
                depth = self.deadlift_ref_top_angle - self.deadlift_min_angle
                if depth >= min_depth_drop:
                    self.deadlift_angle_state = "ascending"
                else:
                    # Too shallow to be a genuine rep (noise/adjustment) - resume
                    # tracking the top without counting anything.
                    self.deadlift_angle_state = "top"
                    self.deadlift_ref_top_angle = angle_value
                    self.deadlift_min_angle = None
                    self.phase = "start"

        elif self.deadlift_angle_state == "ascending":
            if self.deadlift_min_angle is not None and angle_value < self.deadlift_min_angle:
                # Dipped back down before finishing the ascent - keep tracking
                # the true bottom of the movement.
                self.deadlift_min_angle = angle_value
                self.deadlift_angle_state = "descending"
            elif angle_value >= self.deadlift_ref_top_angle - return_tolerance and self.deadlift_cooldown_frames == 0:
                self.reps += 1
                self.phase = "return"
                self.deadlift_cooldown_frames = 6
                self.deadlift_angle_state = "top"
                self.deadlift_ref_top_angle = angle_value
                self.deadlift_min_angle = None
                self.deadlift_rising_streak = 0
                self.deadlift_falling_streak = 0

        return self.reps

    def _minimum_span(self) -> float:
        if self.exercise in {"bicep curls"}:
            return 12.0
        if self.exercise in {"OHP"}:
            return 12.0
        if self.exercise in {"lateral raises"}:
            return 80.0
        if self.exercise in {"rows"}:
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
            return self._single_arm_vertical_metric(sample)
        if self.exercise == "rows":
            return self._single_side_rows_metric(sample)
        if self.exercise == "lateral raises":
            return self._single_side_lateral_metric(sample)
        if self.exercise == "bicep curls":
            left_angle = angle(point(sample, "left_shoulder"), point(sample, "left_elbow"), point(sample, "left_wrist"))
            right_angle = angle(point(sample, "right_shoulder"), point(sample, "right_elbow"), point(sample, "right_wrist"))
            angles = [value for value in [left_angle, right_angle] if value is not None]
            if not angles:
                return None
            return sum(angles) / len(angles)
        return average_y(sample, ["left_hip", "right_hip"])

    def _single_arm_vertical_metric(self, sample: dict[str, float]) -> float | None:
        metrics: dict[str, float | None] = {
            "left": self._arm_vertical_metric(sample, "left"),
            "right": self._arm_vertical_metric(sample, "right"),
        }
        available_sides = [side for side, value in metrics.items() if value is not None]
        if not available_sides:
            return None
        if len(available_sides) == 1:
            chosen_side = available_sides[0]
            self.preferred_arm_side = chosen_side
            return metrics[chosen_side]

        left_score = self._arm_visibility_score(sample, "left")
        right_score = self._arm_visibility_score(sample, "right")
        scores = {"left": left_score, "right": right_score}

        if self.preferred_arm_side in scores:
            preferred_side = self.preferred_arm_side
            other_side = "right" if preferred_side == "left" else "left"
            if scores[preferred_side] + 0.05 >= scores[other_side]:
                return metrics[preferred_side]

        chosen_side = "left" if left_score >= right_score else "right"
        self.preferred_arm_side = chosen_side
        return metrics[chosen_side]

    def _arm_vertical_metric(self, sample: dict[str, float], side: str) -> float | None:
        shoulder_y = sample.get(f"{side}_shoulder_y")
        wrist_y = sample.get(f"{side}_wrist_y")
        if shoulder_y is None or wrist_y is None:
            return None
        return wrist_y - shoulder_y

    def _single_side_rows_metric(self, sample: dict[str, float]) -> float | None:
        metrics: dict[str, float | None] = {
            "left": self._rows_side_metric(sample, "left"),
            "right": self._rows_side_metric(sample, "right"),
        }
        available_sides = [side for side, value in metrics.items() if value is not None]
        if not available_sides:
            return None
        if len(available_sides) == 1:
            chosen_side = available_sides[0]
            self.preferred_arm_side = chosen_side
            return metrics[chosen_side]

        left_score = self._arm_visibility_score(sample, "left")
        right_score = self._arm_visibility_score(sample, "right")
        scores = {"left": left_score, "right": right_score}

        if self.preferred_arm_side in scores:
            preferred_side = self.preferred_arm_side
            other_side = "right" if preferred_side == "left" else "left"
            if scores[preferred_side] + 0.05 >= scores[other_side]:
                return metrics[preferred_side]

        chosen_side = "left" if left_score >= right_score else "right"
        self.preferred_arm_side = chosen_side
        return metrics[chosen_side]

    def _rows_side_metric(self, sample: dict[str, float], side: str) -> float | None:
        hip_y = sample.get(f"{side}_hip_y")
        wrist_y = sample.get(f"{side}_wrist_y")
        if hip_y is None or wrist_y is None:
            return None
        return wrist_y - hip_y

    def _single_side_lateral_metric(self, sample: dict[str, float]) -> float | None:
        metrics: dict[str, float | None] = {
            "left": self._arm_vertical_metric(sample, "left"),
            "right": self._arm_vertical_metric(sample, "right"),
        }
        available_sides = [side for side, value in metrics.items() if value is not None]
        if not available_sides:
            return None
        if len(available_sides) == 1:
            chosen_side = available_sides[0]
            self.preferred_arm_side = chosen_side
            return metrics[chosen_side]

        left_score = self._arm_visibility_score(sample, "left")
        right_score = self._arm_visibility_score(sample, "right")
        scores = {"left": left_score, "right": right_score}

        if self.preferred_arm_side in scores:
            preferred_side = self.preferred_arm_side
            other_side = "right" if preferred_side == "left" else "left"
            if scores[preferred_side] + 0.05 >= scores[other_side]:
                return metrics[preferred_side]

        chosen_side = "left" if left_score >= right_score else "right"
        self.preferred_arm_side = chosen_side
        return metrics[chosen_side]

    def _selected_elbow_angle(self, sample: dict[str, float]) -> float | None:
        if self.preferred_arm_side in {"left", "right"}:
            preferred_angle = self._elbow_angle(sample, self.preferred_arm_side)
            if preferred_angle is not None:
                return preferred_angle

        left_angle = self._elbow_angle(sample, "left")
        right_angle = self._elbow_angle(sample, "right")
        if left_angle is None:
            return right_angle
        if right_angle is None:
            return left_angle

        left_score = self._arm_visibility_score(sample, "left")
        right_score = self._arm_visibility_score(sample, "right")
        return left_angle if left_score >= right_score else right_angle

    def _elbow_angle(self, sample: dict[str, float], side: str) -> float | None:
        return angle(
            point(sample, f"{side}_shoulder"),
            point(sample, f"{side}_elbow"),
            point(sample, f"{side}_wrist"),
        )

    def _deadlift_aux_angle(self, sample: dict[str, float]) -> float | None:
        left_angle = angle(
            point(sample, "left_shoulder"),
            point(sample, "left_hip"),
            point(sample, "left_knee"),
        )
        right_angle = angle(
            point(sample, "right_shoulder"),
            point(sample, "right_hip"),
            point(sample, "right_knee"),
        )
        angles = [value for value in [left_angle, right_angle] if value is not None]
        if not angles:
            return None
        return sum(angles) / len(angles)

    def _arm_visibility_score(self, sample: dict[str, float], side: str) -> float:
        shoulder = point(sample, f"{side}_shoulder")
        elbow = point(sample, f"{side}_elbow")
        wrist = point(sample, f"{side}_wrist")

        score = 0.0
        if shoulder is not None:
            score += 1.0
        if elbow is not None:
            score += 1.0
        if wrist is not None:
            score += 1.0

        if shoulder is not None and elbow is not None:
            upper_arm_len = sqrt((shoulder[0] - elbow[0]) ** 2 + (shoulder[1] - elbow[1]) ** 2)
            score += upper_arm_len
        if elbow is not None and wrist is not None:
            forearm_len = sqrt((elbow[0] - wrist[0]) ** 2 + (elbow[1] - wrist[1]) ** 2)
            score += forearm_len

        return score
