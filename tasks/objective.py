"""Common objective interface for differentiable lower-level task losses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping


class BaseObjective(ABC):
    """Task-owned loss terms with explicit gradient hooks."""

    @abstractmethod
    def running_terms(
        self,
        step: int,
        state: Any,
        control: Any,
    ) -> Mapping[str, float]:
        """Return named loss terms for one running-cost step."""

    @abstractmethod
    def terminal_terms(self, state: Any) -> Mapping[str, float]:
        """Return named loss terms evaluated at the terminal state."""

    @abstractmethod
    def write_gradients(self, context: Any) -> None:
        """Write task loss derivatives into the supplied gradient context."""

    def diagnostics(self, rollout: Any) -> Dict[str, Any]:
        """Return optional task-specific diagnostics for one rollout."""

        _ = rollout
        return {}


class TaskObjective(BaseObjective):
    """Universal objective adapter over task-owned numerical equations."""

    def __init__(self, numerical_task: Any) -> None:
        self.numerical_task = numerical_task

    def running_terms(self, step, state, control):
        return dict(
            self.numerical_task.compute_terms(
                int(step),
                int(state.num_ctrl_steps),
                control,
                state.variables,
                state.q,
            )
        )

    def terminal_terms(self, state):
        _ = state
        return {}

    def write_gradients(self, context) -> None:
        self.numerical_task.write_terminal_grads(
            i=int(context.step),
            num_ctrl_steps=int(context.num_ctrl_steps),
            u_i=context.control,
            variables=context.variables,
            q=context.q,
            ndof_u=int(context.ndof_u),
            ndof_var=int(context.ndof_var),
            ndof_r=int(context.ndof_r),
            sub_steps=int(context.sub_steps),
            coef=dict(context.coefficients),
            df_du=context.df_du,
            df_dvar=context.df_dvar,
            df_dq=context.df_dq,
        )

    def diagnostics(self, rollout):
        callback = getattr(
            self.numerical_task,
            "rollout_diagnostics",
            None,
        )
        runner = getattr(rollout, "runner", None)
        params = getattr(rollout, "params", None)
        if callback is None or runner is None or params is None:
            return {}
        return dict(callback(runner, params) or {})


__all__ = ["BaseObjective", "TaskObjective"]
