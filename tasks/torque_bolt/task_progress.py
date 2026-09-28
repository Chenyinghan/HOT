"""Bounded physical-progress reward for torque BASS guidance."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _clip01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return min(1.0, max(0.0, numeric))


def _approach_progress(distance: Any, reference: float) -> float:
    try:
        numeric = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return _clip01(1.0 - numeric / reference)


def _finite_nonnegative(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, numeric)


def torque_bass_reward(
    *,
    score: float,
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Map nested Align, Engage, Turn progress to a reward in [0, 1]."""

    alpha = float(config.get("alpha", 0.8))
    beta = float(config.get("beta", 0.1))
    if alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        raise ValueError(
            "task-progress reward requires alpha >= 0, beta >= 0, "
            "and alpha + beta < 1"
        )
    loss_reference = float(config.get("loss_reference", 1e5))
    align_reference = float(config.get("align_distance_reference", 10.0))
    engage_reference = float(config.get("engage_distance_reference", 1.3))
    geometry_penalty_scale = float(
        config.get("geometry_penalty_scale", 2.0)
    )
    failure_ceiling_enabled = bool(
        config.get("failure_ceiling_enabled", False)
    )
    failure_ceiling = float(config.get("failure_ceiling", 0.45))
    for name, value in (
        ("loss_reference", loss_reference),
        ("align_distance_reference", align_reference),
        ("engage_distance_reference", engage_reference),
    ):
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and positive")
    if geometry_penalty_scale < 0.0 or not math.isfinite(
        geometry_penalty_scale
    ):
        raise ValueError(
            "geometry_penalty_scale must be finite and nonnegative"
        )
    if (
        not math.isfinite(failure_ceiling)
        or failure_ceiling < 0.0
        or failure_ceiling >= 0.5
    ):
        raise ValueError(
            "failure_ceiling must be finite and lie in [0, 0.5)"
        )

    align_distance = float(diagnostics["align_distance"])
    engage_distance = float(diagnostics["engage_distance"])
    align_met = (
        math.isfinite(align_distance)
        and align_distance <= float(diagnostics["align_tolerance"])
    )
    engage_met = (
        align_met
        and math.isfinite(engage_distance)
        and engage_distance <= float(diagnostics["engage_tolerance"])
    )
    geometry_audit = dict(
        diagnostics.get("symmetric_geometry_audit", {}) or {}
    )
    severe_geometry_overlap = bool(
        geometry_audit.get("severe_overlap", False)
    )
    raw_task_success = bool(
        diagnostics.get(
            "raw_task_success_before_geometry_audit",
            diagnostics["task_success"],
        )
    )
    geometry_valid_terminal_success = bool(
        raw_task_success and not severe_geometry_overlap
    )
    turn_met = bool(engage_met and geometry_valid_terminal_success)
    gates = (align_met, engage_met, turn_met)

    milestone = 0
    for gate in gates:
        if not gate:
            break
        milestone += 1

    if milestone == 0:
        progress = _approach_progress(align_distance, align_reference)
    elif milestone == 1:
        progress = _approach_progress(engage_distance, engage_reference)
    elif milestone == 2:
        progress = _clip01(
            max(0.0, float(diagnostics["nail_turn_deg"]))
            / float(diagnostics["success_nail_deg"])
        )
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
        if milestone == len(gates)
        else (milestone + alpha * progress + beta * loss_quality)
        / float(len(gates) + 1)
    )
    geometry_severity = _finite_nonnegative(
        geometry_audit.get("max_severity", 0.0)
    )
    geometry_reference = max(
        1.0e-12,
        _finite_nonnegative(
            geometry_audit.get("containment_fraction_limit", 0.2)
        )
        * _finite_nonnegative(
            geometry_audit.get("normalized_depth_limit", 1.0)
        ),
    )
    geometry_excess_ratio = (
        max(1.0, geometry_severity / geometry_reference)
        if severe_geometry_overlap
        else 0.0
    )
    geometry_reward_multiplier = 1.0 / (
        1.0 + geometry_penalty_scale * geometry_excess_ratio
    )
    reward *= geometry_reward_multiplier
    raw_process_reward = _clip01(reward)
    failure_upper = (
        (len(gates) - 1) + alpha + beta
    ) / float(len(gates) + 1)
    normalized_failure_reward = (
        _clip01(raw_process_reward / failure_upper)
        if not turn_met
        else 1.0
    )
    if failure_ceiling_enabled:
        reward = (
            1.0
            if turn_met
            else failure_ceiling * normalized_failure_reward
        )
    else:
        reward = raw_process_reward
    return {
        "bass_reward": _clip01(reward),
        # The public success flag is the conjunction of all ordered gates.
        "task_success": turn_met,
        "raw_terminal_task_success": raw_task_success,
        "task_milestone": milestone,
        "task_stage_count": len(gates),
        "task_progress": progress,
        "task_feasible": not severe_geometry_overlap,
        "loss_quality": loss_quality,
        "task_progress_gates": {
            "align": align_met,
            "engage": engage_met,
            "turn": turn_met,
            "task_success": turn_met,
        },
        "symmetric_geometry_ok": not severe_geometry_overlap,
        "geometry_overlap_severity": geometry_severity,
        "geometry_overlap_excess_ratio": geometry_excess_ratio,
        "geometry_reward_multiplier": geometry_reward_multiplier,
        "geometry_penalty_applied": severe_geometry_overlap,
        "raw_process_reward": raw_process_reward,
        "failure_reward_upper_bound": failure_upper,
        "normalized_failure_reward": normalized_failure_reward,
        "failure_ceiling_enabled": failure_ceiling_enabled,
        "failure_ceiling": failure_ceiling,
        "final_scalar_reward": _clip01(reward),
        "task_progress_status": "ok",
    }


__all__ = ["torque_bass_reward"]
