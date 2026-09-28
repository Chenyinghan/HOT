"""Registration and dispatch for lower-level optimizer strategies."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from .base import OptimizationMode, OptimizerStrategy


class OptimizerRegistry:
    """Store optimizer implementations without framework-level branches."""

    def __init__(self) -> None:
        self._strategies: Dict[str, OptimizerStrategy] = {}

    @staticmethod
    def _name(name: str) -> str:
        value = str(name).strip()
        if not value:
            raise ValueError("optimizer name must be non-empty")
        return value

    def register(
        self,
        strategy: OptimizerStrategy,
        *,
        replace: bool = False,
    ) -> None:
        if not isinstance(strategy, OptimizerStrategy):
            raise TypeError("optimizer strategy must implement OptimizerStrategy")
        name = self._name(strategy.name)
        if name in self._strategies and not replace:
            raise ValueError(f"optimizer {name!r} is already registered")
        self._strategies[name] = strategy

    def get(self, name: str) -> OptimizerStrategy:
        optimizer_name = self._name(name)
        try:
            return self._strategies[optimizer_name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"unknown optimizer {optimizer_name!r}; registered "
                f"optimizers: {available}"
            ) from exc

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._strategies))

    def optimize(
        self,
        name: str,
        runner,
        params0: np.ndarray,
        *,
        mode: OptimizationMode,
        design_protocol: Optional[str],
    ) -> np.ndarray:
        strategy = self.get(name)
        strategy.validate(
            mode=mode,
            design_protocol=design_protocol,
        )
        return strategy.optimize(runner, params0)


OPTIMIZER_REGISTRY = OptimizerRegistry()


def register_optimizer(
    strategy: OptimizerStrategy,
    *,
    replace: bool = False,
    registry: Optional[OptimizerRegistry] = None,
) -> None:
    (registry or OPTIMIZER_REGISTRY).register(
        strategy,
        replace=replace,
    )


__all__ = [
    "OPTIMIZER_REGISTRY",
    "OptimizerRegistry",
    "register_optimizer",
]
