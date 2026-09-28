"""Explicit optimizer policies for HOT's two maintained stages."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from .base import OptimizationMode

ACTION_TRUST_REGION = "action_trust_region"
STAGED_ACTION_TRUST_REGION = "staged_action_trust_region"
SUCCESS_CONSTRAINED_TARGET_SHELL = "success_constrained_target_shell"
_NAMES = {ACTION_TRUST_REGION, STAGED_ACTION_TRUST_REGION, SUCCESS_CONSTRAINED_TARGET_SHELL}


def canonical_optimizer_name(name: str) -> str:
    value = str(name).strip()
    if value not in _NAMES:
        raise ValueError(f"Unsupported HOT optimizer {value!r}; choose from {sorted(_NAMES)}")
    return value


@dataclass(frozen=True)
class OptimizerPolicy:
    co_refinement: str = SUCCESS_CONSTRAINED_TARGET_SHELL
    action_only: str = STAGED_ACTION_TRUST_REGION

    def __post_init__(self):
        canonical_optimizer_name(self.co_refinement)
        canonical_optimizer_name(self.action_only)
        if self.co_refinement != SUCCESS_CONSTRAINED_TARGET_SHELL:
            raise ValueError("Stage 2 requires success_constrained_target_shell")
        if self.action_only not in {ACTION_TRUST_REGION, STAGED_ACTION_TRUST_REGION}:
            raise ValueError("Stage 1 requires an action optimizer")

    def for_mode(self, mode):
        if mode is OptimizationMode.CO_REFINEMENT:
            return self.co_refinement
        if mode is OptimizationMode.ACTION_ONLY:
            return self.action_only
        raise ValueError(f"Unsupported optimization mode {mode}")

    def to_dict(self):
        return {"co_refinement": self.co_refinement, "action_only": self.action_only}


def optimizer_policy_from_config(config: Mapping[str, Any], **kwargs) -> OptimizerPolicy:
    for key in ("direct_planar_optimizer", "action_optimizer"):
        if config.get(key) is not None:
            raise ValueError(f"Removed optimizer field {key!r}; use optimizer policy")
    values = config.get("optimizer", {})
    if not isinstance(values, Mapping):
        raise TypeError("optimizer must be an object")
    return OptimizerPolicy(**dict(values))


def optimizer_selection_from_runner(args: Any, *, mode: OptimizationMode,
                                    design_protocol: Optional[str]) -> str:
    explicit = getattr(args, "optimizer_strategy", None)
    if explicit is not None:
        return canonical_optimizer_name(explicit)
    policy = getattr(args, "optimizer_policy", None) or OptimizerPolicy()
    if not isinstance(policy, OptimizerPolicy):
        raise TypeError("optimizer_policy must be OptimizerPolicy")
    return policy.for_mode(mode)


def optimizer_contract_from_config(config: Mapping[str, Any], *, default_maxiter: int) -> dict:
    return {**optimizer_policy_from_config(config).to_dict(),
            "low_level_maxiter": int(config.get("low_level_maxiter", default_maxiter)),
            "geometry_protocol": str(config["generic_design_protocol"])}
