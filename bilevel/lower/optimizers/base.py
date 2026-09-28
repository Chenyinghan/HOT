"""Public optimizer strategy contracts for the canonical bilevel runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, FrozenSet, Optional

import numpy as np


class OptimizationMode(str, Enum):
    """The lower-level problems exposed by the framework."""

    CO_REFINEMENT = "co_refinement"
    ACTION_ONLY = "action_only"
    SHAPE_ONLY = "shape_only"


class OptimizerStrategy(ABC):
    """One registered numerical strategy.

    Strategies receive the already constructed runner so task objectives,
    parameterization, RedMax state, and parameter layouts stay outside the
    optimizer-selection layer.
    """

    name: str
    supported_modes: FrozenSet[OptimizationMode]
    required_protocol: Optional[str] = None

    def validate(
        self,
        *,
        mode: OptimizationMode,
        design_protocol: Optional[str],
    ) -> None:
        if mode not in self.supported_modes:
            supported = ", ".join(
                sorted(value.value for value in self.supported_modes)
            )
            raise ValueError(
                f"optimizer {self.name!r} does not support mode "
                f"{mode.value!r}; supported modes: {supported}"
            )
        if (
            self.required_protocol is not None
            and design_protocol != self.required_protocol
        ):
            raise ValueError(
                f"optimizer {self.name!r} requires design protocol "
                f"{self.required_protocol!r}, got {design_protocol!r}"
            )

    @abstractmethod
    def optimize(self, runner: Any, params0: np.ndarray) -> np.ndarray:
        """Return optimized parameters without changing their public layout."""


__all__ = ["OptimizationMode", "OptimizerStrategy"]
