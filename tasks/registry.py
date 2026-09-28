"""Task names mapped to implementation locations; no factories or task logic."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional


class TaskRegistry:
    """Store only the import path of each task's task.py module."""

    def __init__(self, modules: Optional[Mapping[str, str]] = None) -> None:
        self._modules: dict[str, str] = {}
        for name, module in (modules or {}).items():
            self.register(name, module)

    @staticmethod
    def _name(name: str) -> str:
        value = str(name).strip()
        if not value:
            raise ValueError("task name must be non-empty")
        return value

    def register(self, name: str, module: str, *, replace: bool = False) -> None:
        name = self._name(name)
        if not isinstance(module, str) or not all(
            part.isidentifier() for part in module.split(".")
        ):
            raise TypeError("task location must be a dotted module path")
        if name in self._modules and not replace:
            raise ValueError(f"task {name!r} is already registered")
        self._modules[name] = module

    def unregister(self, name: str) -> None:
        name = self._name(name)
        self.module(name)
        del self._modules[name]

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._modules))

    def module(self, name: str) -> str:
        name = self._name(name)
        try:
            return self._modules[name]
        except KeyError as exc:
            raise KeyError(f"unknown task {name!r}; registered tasks: {', '.join(self.names())}") from exc

    def module_path(self, name: str) -> Path:
        return Path(*self.module(name).split(".")).with_suffix(".py")


TASK_REGISTRY = TaskRegistry({
    "sweep_balls": "tasks.sweep_balls.task",
    "torque_bolt": "tasks.torque_bolt.task",
    "scoop_balls": "tasks.scoop_balls.task",
    "hammer_extract_nail": "tasks.hammer_extract_nail.task",
})
CANONICAL_TASK_NAMES = tuple(TASK_REGISTRY.names())


def canonical_task_module(name: str) -> str:
    return TASK_REGISTRY.module(name)


def canonical_task_module_path(name: str) -> Path:
    return TASK_REGISTRY.module_path(name)


def register_task(name: str, module: str, *, replace: bool = False,
                  registry: Optional[TaskRegistry] = None) -> None:
    (registry or TASK_REGISTRY).register(name, module, replace=replace)


__all__ = ["TaskRegistry", "TASK_REGISTRY", "CANONICAL_TASK_NAMES",
           "canonical_task_module", "canonical_task_module_path", "register_task"]
