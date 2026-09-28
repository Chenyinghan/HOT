"""Typed contexts shared by every canonical bi-level task plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ActionContext:
    """Arguments required to seed one lower-level action trajectory."""

    ndof_u: int
    num_ctrl_steps: int
    seed: int = 0


@dataclass(frozen=True)
class SceneContext:
    """Generated model path and loaded RedMax simulation."""

    model_path: Path
    simulation: Any


@dataclass(frozen=True)
class SceneBindings:
    """A numerical task bound to one generated scene."""

    task: Any
    context: SceneContext

    @property
    def model_path(self) -> Path:
        return self.context.model_path

    @property
    def simulation(self) -> Any:
        return self.context.simulation


@dataclass(frozen=True)
class StepState:
    """State fields consumed by one task objective step."""

    num_ctrl_steps: int
    variables: Any
    q: Any


@dataclass(frozen=True)
class GradientContext:
    """Arguments consumed while writing task-owned analytic gradients."""

    step: int
    num_ctrl_steps: int
    control: Any
    variables: Any
    q: Any
    ndof_u: int
    ndof_var: int
    ndof_r: int
    sub_steps: int
    coefficients: Mapping[str, float]
    df_du: Any
    df_dvar: Any
    df_dq: Any


@dataclass(frozen=True)
class DiagnosticsContext:
    """Runner and parameter vector used by task diagnostics."""

    runner: Any
    params: Any


__all__ = [
    "ActionContext",
    "DiagnosticsContext",
    "GradientContext",
    "SceneBindings",
    "SceneContext",
    "StepState",
]
