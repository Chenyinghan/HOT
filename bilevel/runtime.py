"""Canonical orchestration for task loading, optimization, and replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .config import merge_task_config
from .lower.runtime_args import build_runner_args


def task_config_from_context(task: Any, context: Mapping[str, Any]) -> dict:
    """Merge one task's canonical config with explicit runtime overrides."""

    return merge_task_config(
        task.config,
        context,
        extra_context_overrides={
            "low_level_maxls": ("low_level_maxls", int),
            "force_connectivity": ("force_connectivity", bool),
            "generic_design_protocol": (
                "generic_design_protocol",
                str,
            ),
        },
    )


def load_task_from_context(
    task_name: str,
    context: Mapping[str, Any],
) -> Any:
    """Create one canonical plugin using the request's task configuration."""

    from tasks import load_task

    task_json = dict(context.get("task_json", {}) or {})
    config = dict(task_json.get("task_config", {}) or {})
    task = load_task(task_name, config=config)
    merged = task_config_from_context(task, context)
    if merged != task.config:
        task = load_task(task_name, config=merged)
    return task


def run_low_level_optimization(
    task: Any,
    xml_path: str,
    context: dict,
) -> dict:
    """Run RedMax through the shared engine for one canonical task."""

    from .lower import engine
    from tasks import load_task

    task_name = task.name

    def create_numerical_task(config: dict) -> Any:
        return load_task(task_name, config=config).numerical_task

    return engine.run_low_level_optimization(
        xml_path=xml_path,
        context=context,
        create_task=create_numerical_task,
        config_from_context=lambda value: task_config_from_context(
            task,
            value,
        ),
        runner_args=build_runner_args,
        motion_diagnostics=None,
    )


def optimize_xml(
    task: Any,
    xml_path: str,
    context: dict,
) -> dict:
    """Evaluate one XML in the isolated canonical lower-level subprocess."""

    from .lower import engine
    from tasks import canonical_task_module_path

    config = task_config_from_context(task, context)
    return engine.optimize_xml_subprocess(
        task_name=task.name,
        task_module_path=str(canonical_task_module_path(task.name)),
        xml_path=xml_path,
        context=context,
        low_level_maxiter_default=int(
            config.get("low_level_maxiter", 100)
        ),
    )


def visualize_xml(
    task: Any,
    xml_path: str,
    context: dict,
) -> dict:
    """Replay one canonical task result through the shared engine."""

    from .lower import engine
    from tasks import load_task

    task_name = task.name

    def create_numerical_task(config: dict) -> Any:
        return load_task(task_name, config=config).numerical_task

    return engine.visualize_xml(
        xml_path=xml_path,
        context=context,
        create_task=create_numerical_task,
        config_from_context=lambda value: task_config_from_context(
            task,
            value,
        ),
        runner_args=build_runner_args,
    )


def runner_args_for_task(
    task: Any,
    context: Mapping[str, Any],
    rollout_dir: Path,
) -> Any:
    """Expose the exact runner construction used by optimization and tests."""

    config = task_config_from_context(task, context)
    return build_runner_args(config, context, rollout_dir)


__all__ = [
    "load_task_from_context",
    "optimize_xml",
    "run_low_level_optimization",
    "runner_args_for_task",
    "task_config_from_context",
    "visualize_xml",
]
