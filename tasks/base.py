"""Shared public interface and lifecycle for every HOT task."""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from bilevel import HandleSpec, MorphologySpec, RuntimeSpec, SearchSpec, TaskSpec
from .contexts import SceneBindings, SceneContext
from .objective import TaskObjective


class TaskBase(ABC):
    """Configure a task once and expose the same API to every framework caller.

    Each task package exports ``TaskDefinition(TaskBase)``. Derived classes
    implement ``_configure`` to construct their numerical task and typed specs;
    they override optional behavior only when the task needs it.
    """

    config_path: Optional[Path] = None

    def __init__(self, config: Optional[Mapping[str, Any]] = None, *,
                 payload: Optional[Mapping[str, Any]] = None) -> None:
        if payload is None:
            payload = {} if self.config_path is None else json.loads(
                self.config_path.read_text(encoding="utf-8")
            )
        if not isinstance(payload, Mapping):
            raise TypeError("task payload must be a mapping")
        self.payload = dict(payload)
        self.config = dict(self.payload.get("task_config", {}))
        if config is not None:
            self.config.update(config)
        self._configure()

    @abstractmethod
    def _configure(self) -> None:
        """Construct numerical_task and the five typed specifications."""

    @property
    def name(self) -> str:
        return self.task_spec().name

    def task_spec(self) -> TaskSpec:
        return self._task_spec

    def search_spec(self) -> SearchSpec:
        return self._search_spec

    def handle_spec(self) -> HandleSpec:
        return self._handle_spec

    def morphology_spec(self) -> MorphologySpec:
        return self._morphology_spec

    def runtime_spec(self) -> RuntimeSpec:
        return self._runtime_spec

    def bind_scene(self, model: Any) -> SceneBindings:
        context = SceneContext(Path(model.model_path), model.simulation)
        self.numerical_task.configure_model(str(context.model_path), context.simulation)
        return SceneBindings(task=self.numerical_task, context=context)

    def create_objective(self, bindings: SceneBindings) -> TaskObjective:
        if not isinstance(bindings, SceneBindings):
            raise TypeError("task objective requires SceneBindings")
        if bindings.task is not self.numerical_task:
            raise ValueError("scene bindings belong to a different task")
        return TaskObjective(self.numerical_task)

    def seed_action(self, context: Any) -> Any:
        return self.numerical_task.init_action(
            int(context.ndof_u), int(context.num_ctrl_steps), seed=int(context.seed)
        )

    def action_bounds(self, *, ndof_u: int, num_ctrl_steps: int,
                      ndof_cage: int, optimize_design: bool) -> Any:
        return self.numerical_task.bounds(
            int(ndof_u), int(num_ctrl_steps), int(ndof_cage), bool(optimize_design)
        )

    def initialize_morphology(self, model_path: Path, simulation: Any) -> Any:
        return self.numerical_task.init_design(str(Path(model_path)), simulation)

    def validate_candidate(self, candidate: Any) -> dict[str, Any]:
        """Optional task-specific validity checks, in addition to framework gates."""
        return {"ok": True, "errors": []}

    def diagnostics(self, rollout: Any) -> dict[str, Any]:
        numerical_task = getattr(self, "numerical_task", None)
        return {} if numerical_task is None else TaskObjective(numerical_task).diagnostics(rollout)

    def bass_evaluation(self, *, score: float,
                        run_result: dict[str, Any]) -> dict[str, Any] | None:
        """Optional milestone/reward metadata for structural search."""
        return None


__all__ = ["TaskBase"]
