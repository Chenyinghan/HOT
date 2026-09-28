"""Typed, behavior-neutral configuration contracts for the canonical framework."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def _nonempty(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _positive_int(value: int, field_name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _nonnegative_int(value: int, field_name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _finite_tuple(
    values: Sequence[float],
    size: int,
    field_name: str,
) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{field_name} must contain exactly {size} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain only finite values")
    return result


def _string_tuple(values: Sequence[str], field_name: str) -> Tuple[str, ...]:
    result = tuple(_nonempty(value, field_name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class TaskSpec:
    """Task identity, objective ownership, scene, and simulation horizon."""

    name: str
    functions: Tuple[str, ...]
    scene_path: Path
    horizon: int
    substeps: int
    objective_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "name"))
        functions = _string_tuple(self.functions, "functions")
        if not functions:
            raise ValueError("functions must contain at least one task function")
        object.__setattr__(self, "functions", functions)
        object.__setattr__(self, "scene_path", Path(self.scene_path))
        object.__setattr__(self, "horizon", _positive_int(self.horizon, "horizon"))
        object.__setattr__(self, "substeps", _positive_int(self.substeps, "substeps"))
        weights = {
            _nonempty(key, "objective_weights key"): float(value)
            for key, value in self.objective_weights.items()
        }
        if not all(math.isfinite(value) and value >= 0.0 for value in weights.values()):
            raise ValueError("objective_weights values must be finite and non-negative")
        object.__setattr__(self, "objective_weights", weights)


@dataclass(frozen=True)
class SearchSpec:
    """Discrete Handle-root BASS search space and policy."""

    allowed_head_asset_ids: Tuple[str, ...]
    max_head_links: int
    function_count: int
    function_count_margin: int = 0
    bass: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assets = _string_tuple(
            self.allowed_head_asset_ids,
            "allowed_head_asset_ids",
        )
        if not assets:
            raise ValueError("allowed_head_asset_ids must not be empty")
        object.__setattr__(self, "allowed_head_asset_ids", assets)
        object.__setattr__(
            self,
            "max_head_links",
            _positive_int(self.max_head_links, "max_head_links"),
        )
        object.__setattr__(
            self,
            "function_count",
            _positive_int(self.function_count, "function_count"),
        )
        object.__setattr__(
            self,
            "function_count_margin",
            _nonnegative_int(self.function_count_margin, "function_count_margin"),
        )
        object.__setattr__(self, "bass", dict(self.bass))


@dataclass(frozen=True)
class HandleSpec:
    """Fixed Handle asset, task-world pose, and root actuator contract.

    ``mount_face`` is the Handle face occupied by its root-side mounting
    interface.  It is intentionally excluded from ``open_faces``, which are
    the faces exposed to Head search.
    """

    root_asset_id: str
    mount_face: int
    open_faces: Tuple[int, ...]
    position: Vec3
    orientation: Quat
    actuator_type: str
    actuator_bounds: Vec2
    actuator_gains: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "root_asset_id",
            _nonempty(self.root_asset_id, "root_asset_id"),
        )
        mount_face = int(self.mount_face)
        open_faces = tuple(int(face) for face in self.open_faces)
        if mount_face < 0:
            raise ValueError("mount_face must be non-negative")
        if not open_faces or any(face < 0 for face in open_faces):
            raise ValueError("open_faces must contain non-negative face IDs")
        if len(set(open_faces)) != len(open_faces):
            raise ValueError("open_faces must not contain duplicates")
        if mount_face in open_faces:
            raise ValueError("mount_face must not also be an open face")
        object.__setattr__(self, "mount_face", mount_face)
        object.__setattr__(self, "open_faces", open_faces)
        object.__setattr__(
            self,
            "position",
            _finite_tuple(self.position, 3, "position"),
        )
        orientation = _finite_tuple(self.orientation, 4, "orientation")
        if math.sqrt(sum(value * value for value in orientation)) <= 1e-12:
            raise ValueError("orientation quaternion must be non-zero")
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(
            self,
            "actuator_type",
            _nonempty(self.actuator_type, "actuator_type"),
        )
        bounds = _finite_tuple(self.actuator_bounds, 2, "actuator_bounds")
        if bounds[0] >= bounds[1]:
            raise ValueError("actuator_bounds must satisfy lower < upper")
        object.__setattr__(self, "actuator_bounds", bounds)
        gains = {
            _nonempty(key, "actuator_gains key"): float(value)
            for key, value in self.actuator_gains.items()
        }
        if not all(math.isfinite(value) and value >= 0.0 for value in gains.values()):
            raise ValueError("actuator_gains values must be finite and non-negative")
        object.__setattr__(self, "actuator_gains", gains)


@dataclass(frozen=True)
class MorphologySpec:
    """Continuous morphology parameterization, constraints, and optimizer."""

    parameterization_id: str
    constraints: Mapping[str, Any] = field(default_factory=dict)
    optimizer: Mapping[str, Any] = field(default_factory=dict)
    collision_policy: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameterization_id",
            _nonempty(self.parameterization_id, "parameterization_id"),
        )
        object.__setattr__(self, "constraints", dict(self.constraints))
        object.__setattr__(self, "optimizer", dict(self.optimizer))
        object.__setattr__(self, "collision_policy", dict(self.collision_policy))


@dataclass(frozen=True)
class RuntimeSpec:
    """Execution resources, artifact locations, and diagnostic policy."""

    timeout_seconds: Optional[float]
    numeric_threads: int
    output_dir: Path
    cache_dir: Path
    replay_dir: Path
    diagnostics_level: str = "standard"
    seed: int = 0
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timeout = (
            None
            if self.timeout_seconds is None
            else float(self.timeout_seconds)
        )
        if timeout is not None and (
            not math.isfinite(timeout) or timeout <= 0.0
        ):
            raise ValueError(
                "timeout_seconds must be None or a finite positive value"
            )
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(
            self,
            "numeric_threads",
            _positive_int(self.numeric_threads, "numeric_threads"),
        )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "replay_dir", Path(self.replay_dir))
        object.__setattr__(
            self,
            "diagnostics_level",
            _nonempty(self.diagnostics_level, "diagnostics_level"),
        )
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "environment",
            {
                _nonempty(key, "environment key"): str(value)
                for key, value in self.environment.items()
            },
        )


__all__ = [
    "HandleSpec",
    "MorphologySpec",
    "RuntimeSpec",
    "SearchSpec",
    "TaskSpec",
]
