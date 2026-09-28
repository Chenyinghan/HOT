"""Bounded physical-progress reward for Scoop BASS guidance."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


_STAGES = ("approach", "sweep", "lift")


def _finite(value: Any, path: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"scoop task-progress diagnostic {path} must be numeric"
        ) from exc
    if not math.isfinite(numeric):
        raise ValueError(
            f"scoop task-progress diagnostic {path} must be finite"
        )
    return numeric


def _clip01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return min(1.0, max(0.0, numeric))


def _upper_bound_progress(value: Any, limit: Any, path: str) -> float:
    numeric = max(0.0, _finite(value, path))
    bound = _finite(limit, f"{path}_limit")
    if bound <= 0.0:
        raise ValueError(
            f"scoop task-progress limit for {path} must be positive"
        )
    if numeric <= bound:
        return 1.0
    return _clip01(bound / numeric)


def _lower_bound_progress(
    value: Any,
    threshold: Any,
    path: str,
    *,
    scale: Any | None = None,
) -> float:
    numeric = _finite(value, path)
    bound = _finite(threshold, f"{path}_threshold")
    if numeric >= bound:
        return 1.0
    denominator = (
        abs(bound)
        if scale is None
        else abs(_finite(scale, f"{path}_scale"))
    )
    denominator = max(denominator, 1.0e-12)
    return _clip01(1.0 - (bound - numeric) / denominator)


def _count_progress(value: Any, required: int, path: str) -> float:
    if required <= 0:
        raise ValueError("scoop required_ball_count must be positive")
    return _clip01(_finite(value, path) / float(required))


def _minimum(values: Any, path: str) -> float:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(
            f"scoop task-progress diagnostic {path} must be a sequence"
        )
    if not values:
        raise ValueError(
            f"scoop task-progress diagnostic {path} must not be empty"
        )
    return min(_finite(value, f"{path}[]") for value in values)


def _maximum(values: Any, path: str) -> float:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(
            f"scoop task-progress diagnostic {path} must be a sequence"
        )
    if not values:
        raise ValueError(
            f"scoop task-progress diagnostic {path} must not be empty"
        )
    return max(_finite(value, f"{path}[]") for value in values)


def _mean(parts: Sequence[float]) -> float:
    if not parts:
        return 0.0
    return _clip01(sum(parts) / float(len(parts)))


def _approach_progress(gate: Mapping[str, Any]) -> float:
    return _mean(
        (
            _upper_bound_progress(
                gate["horizontal_target_distance"],
                gate["position_tolerance"],
                "approach.horizontal_target_distance",
            ),
            _upper_bound_progress(
                gate["height_error"],
                gate["height_tolerance"],
                "approach.height_error",
            ),
            _lower_bound_progress(
                gate["function_below_ball_bottom_min"],
                gate["function_below_ball_bottom_required"],
                "approach.function_below_ball_bottom_min",
                scale=gate["height_tolerance"],
            ),
            _upper_bound_progress(
                gate["angle_error_rad"],
                gate["angle_tolerance_rad"],
                "approach.angle_error_rad",
            ),
            _upper_bound_progress(
                gate["source_ball_xy_motion_max"],
                gate["source_ball_xy_motion_tolerance"],
                "approach.source_ball_xy_motion_max",
            ),
        )
    )


def _sweep_progress(gate: Mapping[str, Any], required: int) -> float:
    below = _minimum(
        gate["function_below_ball_bottom"],
        "sweep.function_below_ball_bottom",
    )
    below_tolerance = _finite(
        gate["function_below_ball_bottom_tolerance"],
        "sweep.function_below_ball_bottom_tolerance",
    )
    return _mean(
        (
            _upper_bound_progress(
                gate["horizontal_target_distance"],
                gate["position_tolerance"],
                "sweep.horizontal_target_distance",
            ),
            _upper_bound_progress(
                gate["height_error"],
                gate["height_tolerance"],
                "sweep.height_error",
            ),
            _lower_bound_progress(
                below,
                -below_tolerance,
                "sweep.function_below_ball_bottom",
                scale=below_tolerance,
            ),
            _upper_bound_progress(
                gate["angle_error_rad"],
                gate["angle_tolerance_rad"],
                "sweep.angle_error_rad",
            ),
            _upper_bound_progress(
                _maximum(
                    gate["payload_xy_distance"],
                    "sweep.payload_xy_distance",
                ),
                gate["capture_xy_distance_tolerance"],
                "sweep.payload_xy_distance",
            ),
            _count_progress(
                gate["captured_count"], required, "sweep.captured_count"
            ),
        )
    )


def _lift_progress(gate: Mapping[str, Any], required: int) -> float:
    return _mean(
        (
            _lower_bound_progress(
                gate["height_above_source_support"],
                gate["height_required"],
                "lift.height_above_source_support",
            ),
            _upper_bound_progress(
                gate["horizontal_position_drift"],
                gate["horizontal_position_tolerance"],
                "lift.horizontal_position_drift",
            ),
            _lower_bound_progress(
                _minimum(gate["payload_clearance"], "lift.payload_clearance"),
                gate["payload_clearance_required"],
                "lift.payload_clearance",
            ),
            _lower_bound_progress(
                gate["pitch_rad"],
                gate["pitch_required_rad"],
                "lift.pitch_rad",
            ),
            _count_progress(
                gate["retained_count"], required, "lift.retained_count"
            ),
        )
    )


def scoop_bass_reward(
    *,
    score: float,
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Map ordered Approach, Sweep, and Lift evidence to ``[0, 1]``."""

    alpha = float(config.get("alpha", 0.8))
    beta = float(config.get("beta", 0.1))
    if alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        raise ValueError(
            "task-progress reward requires alpha >= 0, beta >= 0, "
            "and alpha + beta < 1"
        )
    loss_reference = float(config.get("loss_reference", 1.0e4))
    if loss_reference <= 0.0 or not math.isfinite(loss_reference):
        raise ValueError("loss_reference must be finite and positive")

    stage_gates = diagnostics.get("stage_gates")
    if not isinstance(stage_gates, Mapping):
        raise ValueError("scoop task progress requires stage_gates diagnostics")
    missing = [
        stage
        for stage in _STAGES
        if not isinstance(stage_gates.get(stage), Mapping)
    ]
    if missing:
        raise ValueError(
            "scoop task progress requires complete stage gates; "
            f"missing={missing}"
        )
    required = int(diagnostics.get("required_ball_count", 0))
    if required <= 0:
        raise ValueError("scoop task progress requires required_ball_count > 0")

    approach = stage_gates["approach"]
    sweep = stage_gates["sweep"]
    lift = stage_gates["lift"]
    raw_task_success = bool(diagnostics.get("task_success", False))
    approach_met = bool(approach.get("accepted", False))
    sweep_met = bool(approach_met and sweep.get("accepted", False))
    lift_met = bool(
        sweep_met and lift.get("accepted", False)
    )
    lift_carry_met = bool(
        diagnostics.get("lift_carry_success", False)
    )
    task_success = bool(lift_met and lift_carry_met and raw_task_success)
    gates = (approach_met, sweep_met, lift_met)

    milestone = 0
    for gate in gates:
        if not gate:
            break
        milestone += 1

    if milestone == 0:
        progress = _approach_progress(approach)
    elif milestone == 1:
        progress = _sweep_progress(sweep, required)
    elif milestone == 2:
        progress = _lift_progress(lift, required)
    else:
        progress = 1.0

    finite_score = math.isfinite(float(score))
    loss_quality = (
        loss_reference / (loss_reference + max(0.0, float(score)))
        if finite_score
        else 0.0
    )
    reward = (
        1.0
        if task_success
        else (milestone + alpha * progress + beta * loss_quality)
        / float(len(gates) + 1)
    )
    return {
        "bass_reward": _clip01(reward),
        "task_success": task_success,
        "raw_terminal_task_success": raw_task_success,
        "task_milestone": milestone,
        "task_stage_count": len(gates),
        "task_progress": progress,
        "task_feasible": True,
        "loss_quality": loss_quality,
        "task_progress_gates": {
            "approach": approach_met,
            "sweep": sweep_met,
            "lift": lift_met,
            "lift_carry": lift_carry_met,
            "task_success": task_success,
        },
        "task_progress_status": "ok",
    }


__all__ = ["scoop_bass_reward"]
