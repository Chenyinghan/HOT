"""Bounded physical-progress reward for Target_container BASS guidance."""

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


def sweep_balls_bass_reward(
    *,
    score: float,
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Map ordered sweep, contain, and hold progress to ``[0, 1]``."""

    alpha = float(config.get("alpha", 0.8))
    beta = float(config.get("beta", 0.1))
    if alpha < 0.0 or beta < 0.0 or alpha + beta >= 1.0:
        raise ValueError(
            "task-progress reward requires alpha >= 0, beta >= 0, "
            "and alpha + beta < 1"
        )
    loss_reference = float(config.get("loss_reference", 1e5))
    if loss_reference <= 0.0 or not math.isfinite(loss_reference):
        raise ValueError("loss_reference must be finite and positive")

    sweep_progress = _clip01(diagnostics["mean_ball_progress_ratio"])
    safe_fraction = _clip01(diagnostics["safe_ball_fraction"])
    hold_progress = _clip01(diagnostics["success_hold_fraction"])
    terminal_success = bool(diagnostics["task_success"])

    # Reaching the exact depth target is the dense sweep milestone.  All balls
    # already inside the task's radius-aware safe region is stronger physical
    # evidence, so it must also unlock the containment/hold gates.
    sweep_met = bool(
        terminal_success or sweep_progress >= 1.0 or safe_fraction >= 1.0
    )
    contain_met = bool(
        sweep_met and (terminal_success or safe_fraction >= 1.0)
    )
    hold_met = bool(contain_met and terminal_success)
    gates = (sweep_met, contain_met, hold_met)

    milestone = 0
    for gate in gates:
        if not gate:
            break
        milestone += 1

    if milestone == 0:
        progress = sweep_progress
    elif milestone == 1:
        progress = safe_fraction
    elif milestone == 2:
        progress = hold_progress
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
    return {
        "bass_reward": _clip01(reward),
        "task_success": milestone == len(gates),
        "task_milestone": milestone,
        "task_stage_count": len(gates),
        "task_progress": progress,
        "loss_quality": loss_quality,
        "task_progress_gates": {
            "sweep_progress": gates[0],
            "balls_contained": gates[1],
            "success_hold": gates[2],
            "task_success": milestone == len(gates),
        },
        "task_progress_status": "ok",
    }


__all__ = ["sweep_balls_bass_reward"]
