"""Central configuration precedence for legacy and canonical task adapters."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple


Converter = Callable[[Any], Any]
OverrideRule = Tuple[str, Converter]

COMMON_CONTEXT_OVERRIDES: Mapping[str, OverrideRule] = {
    "low_level_num_steps": ("num_steps", int),
    "low_level_sub_steps": ("sub_steps", int),
    "low_level_grad_clip": ("low_level_grad_clip", float),
    "low_level_step_scale": ("low_level_step_scale", float),
    "low_level_lr": ("low_level_lr", float),
}


def merge_task_config(
    defaults: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    extra_context_overrides: Optional[Mapping[str, OverrideRule]] = None,
) -> Dict[str, Any]:
    """Apply the shared precedence without inventing task-specific defaults.

    Precedence, from lowest to highest:

    1. task adapter defaults;
    2. ``context.task_json.task_config``;
    3. non-``None`` runtime context overrides.
    """

    merged = dict(defaults)
    task_json = dict(context.get("task_json", {}) or {})
    merged.update(dict(task_json.get("task_config", {}) or {}))

    rules = dict(COMMON_CONTEXT_OVERRIDES)
    if extra_context_overrides:
        overlap = set(rules).intersection(extra_context_overrides)
        if overlap:
            raise ValueError(
                "extra_context_overrides duplicates common keys: "
                f"{sorted(overlap)}"
            )
        rules.update(extra_context_overrides)

    for context_name, (config_name, converter) in rules.items():
        value = context.get(context_name)
        if value is not None:
            merged[config_name] = converter(value)

    if context.get("low_level_redmax_verbose"):
        merged["verbose"] = True
    return merged


__all__ = [
    "COMMON_CONTEXT_OVERRIDES",
    "Converter",
    "OverrideRule",
    "merge_task_config",
]
