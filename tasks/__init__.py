"""Task plugins and canonical task interfaces."""

from .base import TaskBase
from .contexts import (
    ActionContext,
    DiagnosticsContext,
    GradientContext,
    SceneBindings,
    SceneContext,
    StepState,
)
from .objective import BaseObjective, TaskObjective
from .loader import load_task
from .registry import (
    CANONICAL_TASK_NAMES,
    TASK_REGISTRY,
    TaskRegistry,
    canonical_task_module,
    canonical_task_module_path,
    register_task,
)

__all__ = [
    "ActionContext",
    "TaskBase",
    "BaseObjective",
    "TaskObjective",
    "CANONICAL_TASK_NAMES",
    "canonical_task_module",
    "canonical_task_module_path",
    "DiagnosticsContext",
    "GradientContext",
    "SceneBindings",
    "SceneContext",
    "StepState",
    "TASK_REGISTRY",
    "TaskRegistry",
    "load_task",
    "register_task",
]
