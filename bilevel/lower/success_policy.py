"""Shared stage-aware physical-success policies."""

from __future__ import annotations

from typing import Any, Dict


def stage_aware_penetration_success(
    *,
    physical_success: bool,
    optimize_design: bool,
    soft_penetration_ok: bool,
    severe_geometry_ok: bool,
) -> Dict[str, Any]:
    """Resolve success without hiding Stage-2 soft-penetration warnings.

    Action-only Stage 1 strictly enforces both the task's soft-contact
    penetration threshold and its severe geometry audit.  Design-enabled
    Stage 2 keeps reporting the soft threshold but only the severe geometry
    audit remains a hard success gate.
    """

    penetration_success_enforced = not bool(optimize_design)
    soft_penetration_ok = bool(soft_penetration_ok)
    severe_geometry_ok = bool(severe_geometry_ok)
    physical_success = bool(physical_success)
    return {
        "penetration_success_enforced": penetration_success_enforced,
        "soft_penetration_ok": soft_penetration_ok,
        "penetration_warning": not soft_penetration_ok,
        "task_success_without_soft_penetration": bool(
            physical_success and severe_geometry_ok
        ),
        "task_success": bool(
            physical_success
            and severe_geometry_ok
            and (
                soft_penetration_ok
                or not penetration_success_enforced
            )
        ),
    }


__all__ = ["stage_aware_penetration_success"]
