"""Instantiate the universal TaskDefinition class at a registered location."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping, Optional

from .base import TaskBase
from .registry import TASK_REGISTRY, TaskRegistry


def load_task(name: str, *, config: Optional[Mapping[str, Any]] = None,
              registry: Optional[TaskRegistry] = None) -> TaskBase:
    registry = registry or TASK_REGISTRY
    module_name = registry.module(name)
    definition = getattr(import_module(module_name), "TaskDefinition", None)
    if not isinstance(definition, type) or not issubclass(definition, TaskBase):
        raise TypeError(f"{module_name} must export TaskDefinition derived from TaskBase")
    task = definition(config)
    if task.name != registry._name(name):
        raise ValueError(f"registered name {name!r} does not match task name {task.name!r}")
    return task


__all__ = ["load_task"]
