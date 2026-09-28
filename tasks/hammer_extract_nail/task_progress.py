"""Bounded physical-progress reward for hammer_extract_nail BASS guidance."""

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


def hammer_extract_nail_bass_reward(
    *,
    score: float,
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Map nested physical task milestones to a reward in [0, 1]."""

    alpha = float(config.get("alpha", 0.8))
    beta = float(config.get("beta", 0.1))
    if alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        raise ValueError(
            "task-progress reward requires alpha >= 0, beta >= 0, "
            "and alpha + beta < 1"
        )
    loss_reference = float(config.get("loss_reference", 1e5))
    hammer_distance_reference = float(
        config.get("hammer_distance_reference", 5.0)
    )
    extract_distance_reference = float(
        config.get("extract_distance_reference", 15.0)
    )
    penetration_penalty_scale = float(
        config.get("penetration_penalty_scale", 1.0)
    )
    geometry_penalty_scale = float(
        config.get("geometry_penalty_scale", 2.0)
    )
    failure_ceiling_enabled = bool(
        config.get("failure_ceiling_enabled", False)
    )
    failure_ceiling = float(config.get("failure_ceiling", 0.45))
    for name, value in (
        ("loss_reference", loss_reference),
        ("hammer_distance_reference", hammer_distance_reference),
        ("extract_distance_reference", extract_distance_reference),
    ):
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and positive")
    if penetration_penalty_scale < 0.0 or not math.isfinite(
        penetration_penalty_scale
    ):
        raise ValueError(
            "penetration_penalty_scale must be finite and nonnegative"
        )
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

    terminal = dict(diagnostics.get("terminal", {}) or {})
    hammer_contact = bool(terminal.get("hammer_contact_met", False))
    down_goal = bool(diagnostics.get("nail_down_goal_met", False))
    transfer_distance = diagnostics.get("min_extract_target_dist")
    transfer_threshold = float(
        diagnostics.get(
            "transfer_approach_distance",
            config.get("transfer_approach_distance", 0.3),
        )
    )
    transfer_met = (
        transfer_distance is not None
        and math.isfinite(float(transfer_distance))
        and float(transfer_distance) <= transfer_threshold
    )
    engage_distance = terminal.get(
        "engagement_distance",
        diagnostics.get("extract_target_dist_end"),
    )
    max_engage_distance = float(
        diagnostics.get("maximum_engage_gate_distance", 1.0)
    )
    engage_met = (
        engage_distance is not None
        and math.isfinite(float(engage_distance))
        and float(engage_distance) <= max_engage_distance
    )
    extract_contact_diagnostic = bool(
        terminal.get("extract_contact_met", False)
    )
    geometry_audit = dict(
        terminal.get("symmetric_geometry_audit", {}) or {}
    )
    severe_geometry_overlap = bool(
        geometry_audit.get("severe_overlap", False)
    )
    task_feasible = not severe_geometry_overlap
    task_success = bool(
        diagnostics.get("task_success", False)
        and not severe_geometry_overlap
    )

    gates = [
        hammer_contact,
        hammer_contact and down_goal,
        hammer_contact and down_goal and transfer_met,
        hammer_contact and down_goal and transfer_met and engage_met,
        (
            hammer_contact
            and down_goal
            and transfer_met
            and engage_met
            and task_success
        ),
    ]
    milestone = 0
    for gate in gates:
        if not gate:
            break
        milestone += 1

    if milestone == 0:
        # Use the same approach-boundary measurement as the optimizer gate.
        # A minimum over the full moving target trajectory can become zero
        # even for a stationary tool when the target passes through it.
        progress = _approach_progress(
            terminal.get("approach_distance"),
            hammer_distance_reference,
        )
    elif milestone == 1:
        target = float(diagnostics.get("hammer_completion_depth", 1.0))
        progress = _clip01(float(diagnostics.get("nail_down_depth", 0.0)) / target)
    elif milestone == 2:
        progress = _approach_progress(
            transfer_distance,
            extract_distance_reference,
        )
    elif milestone == 3:
        progress = _approach_progress(
            engage_distance,
            max_engage_distance,
        )
    elif milestone == 4:
        target = float(diagnostics.get("target_up_lift", 1.0))
        progress = _clip01(float(diagnostics.get("nail_up_lift", 0.0)) / target)
    else:
        progress = 1.0

    finite_score = math.isfinite(float(score))
    loss_quality = (
        loss_reference
        / (loss_reference + max(0.0, float(score)))
        if finite_score
        else 0.0
    )
    reward = (
        1.0
        if milestone == len(gates)
        else (milestone + alpha * progress + beta * loss_quality)
        / float(len(gates) + 1)
    )
    penetration_limit = float(
        diagnostics.get(
            "max_contact_penetration",
            config.get("max_contact_penetration", 0.15),
        )
    )
    if penetration_limit <= 0.0 or not math.isfinite(penetration_limit):
        raise ValueError("max_contact_penetration must be finite and positive")
    max_penetration = max(
        _finite_nonnegative(terminal.get(name, 0.0))
        for name in (
            "max_head_nail_down_penetration_1ms",
            "max_head_nail_up_penetration_1ms",
            "max_head_nail_down_penetration",
            "max_head_nail_up_engage_penetration",
            "max_head_nail_up_pull_penetration",
        )
    )
    pull_limit = float(diagnostics.get(
        "max_pull_penetration", config.get("max_pull_penetration", penetration_limit)
    ))
    if pull_limit <= 0.0 or not math.isfinite(pull_limit):
        raise ValueError("max_pull_penetration must be finite and positive")
    # Old diagnostics have only a whole-trajectory dense maximum. Keep their
    # conservative interpretation; new audits provide separate phase maxima.
    pre_pull_max = max(_finite_nonnegative(terminal.get(name, 0.0)) for name in (
        "max_head_nail_down_penetration_1ms",
        "max_head_nail_down_penetration",
        "max_head_nail_up_engage_penetration",
        "max_head_nail_up_pre_pull_penetration_1ms",
    ))
    if "max_head_nail_up_pull_penetration_1ms" not in terminal:
        pre_pull_max = max(pre_pull_max, _finite_nonnegative(
            terminal.get("max_head_nail_up_penetration_1ms", 0.0)
        ))
    pull_max = max(_finite_nonnegative(terminal.get(name, 0.0)) for name in (
        "max_head_nail_up_pull_penetration", "max_head_nail_up_pull_penetration_1ms",
    ))
    penetration_excess_ratio = max(
        0.0, pre_pull_max / penetration_limit - 1.0, pull_max / pull_limit - 1.0
    )
    penetration_reward_multiplier = 1.0 / (
        1.0 + penetration_penalty_scale * penetration_excess_ratio
    )
    reward *= penetration_reward_multiplier
    geometry_severity = _finite_nonnegative(
        geometry_audit.get("max_severity", 0.0)
    )
    geometry_reference = max(
        1e-12,
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
        if not task_success
        else 1.0
    )
    if failure_ceiling_enabled:
        reward = (
            1.0
            if task_success
            else failure_ceiling * normalized_failure_reward
        )
    else:
        reward = raw_process_reward
    return {
        "bass_reward": _clip01(reward),
        "task_success": task_success,
        "task_milestone": milestone,
        "task_stage_count": len(gates),
        "task_progress": progress,
        "task_feasible": task_feasible,
        "loss_quality": loss_quality,
        "task_progress_gates": {
            "hammer_contact": gates[0],
            "nail_down": gates[1],
            "extract_transfer": gates[2],
            "extract_engage_geometry": gates[3],
            "extract_contact": bool(
                gates[2] and extract_contact_diagnostic
            ),
            "task_success": gates[4],
        },
        "max_operated_object_penetration": max_penetration,
        "penetration_limit": penetration_limit,
        "penetration_excess_ratio": penetration_excess_ratio,
        "penetration_reward_multiplier": penetration_reward_multiplier,
        "penetration_penalty_applied": penetration_excess_ratio > 0.0,
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


__all__ = ["hammer_extract_nail_bass_reward"]
