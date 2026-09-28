"""Runtime-facing composition for action and unified Head morphology."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import (
    MorphologyBlock,
    MorphologyCollisionDecision,
    MorphologyCollisionPolicy,
    MorphologyLayout,
    MorphologyParameterization,
    MorphologyRetractionResult,
)


def _finite_vector(
    values: np.ndarray,
    *,
    expected_dim: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if array.shape != (int(expected_dim),):
        raise ValueError(
            f"{name} has dimension {array.size}, expected exactly "
            f"{int(expected_dim)}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class JointParameterLayout:
    """Exact concatenation contract for lower-level action and morphology."""

    action_dim: int
    morphology_layout: MorphologyLayout | None = None
    action_slice: slice = field(init=False)
    morphology_slice: slice = field(init=False)
    morphology_dim: int = field(init=False)
    total_dim: int = field(init=False)

    def __post_init__(self) -> None:
        action_dim = int(self.action_dim)
        if action_dim < 0:
            raise ValueError("action_dim must be nonnegative")
        morphology_dim = (
            0
            if self.morphology_layout is None
            else int(self.morphology_layout.morphology_dim)
        )
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "morphology_dim", morphology_dim)
        object.__setattr__(self, "total_dim", action_dim + morphology_dim)
        object.__setattr__(self, "action_slice", slice(0, action_dim))
        object.__setattr__(
            self,
            "morphology_slice",
            slice(action_dim, action_dim + morphology_dim),
        )

    @property
    def morphology_parameterization(self) -> str | None:
        if self.morphology_layout is None:
            return None
        return self.morphology_layout.parameterization_id

    def validate(self, params: np.ndarray) -> np.ndarray:
        return _finite_vector(
            params,
            expected_dim=self.total_dim,
            name="joint parameter vector",
        )

    def pack(
        self,
        action: np.ndarray,
        morphology: np.ndarray | None = None,
    ) -> np.ndarray:
        action_values = _finite_vector(
            action,
            expected_dim=self.action_dim,
            name="action vector",
        )
        if self.morphology_layout is None:
            if morphology is not None and np.asarray(morphology).size:
                raise ValueError(
                    "morphology values were provided for an action-only layout"
                )
            return action_values.copy()
        if morphology is None:
            raise ValueError(
                "morphology vector is required by the joint parameter layout"
            )
        morphology_values = _finite_vector(
            morphology,
            expected_dim=self.morphology_dim,
            name="morphology vector",
        )
        return np.concatenate((action_values, morphology_values))

    def unpack(
        self,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        values = self.validate(params)
        action = values[self.action_slice].copy()
        morphology = (
            None
            if self.morphology_layout is None
            else values[self.morphology_slice].copy()
        )
        return action, morphology

    def to_manifest(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "action_slice": [
                int(self.action_slice.start),
                int(self.action_slice.stop),
            ],
            "morphology_dim": self.morphology_dim,
            "morphology_slice": [
                int(self.morphology_slice.start),
                int(self.morphology_slice.stop),
            ],
            "total_dim": self.total_dim,
            "morphology_parameterization": self.morphology_parameterization,
            "morphology_layout": (
                None
                if self.morphology_layout is None
                else self.morphology_layout.to_manifest()
            ),
        }


@dataclass(frozen=True)
class MorphologyRuntimeContext:
    """Stable task-diagnostic view without legacy optimizer coordinates."""

    parameterization_id: str
    layout: MorphologyLayout
    specification: Any
    head_topology: Any | None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "parameterization_id": self.parameterization_id,
            "layout": self.layout.to_manifest(),
            "has_specification": self.specification is not None,
            "has_head_topology": self.head_topology is not None,
        }


class MorphologyRuntimeBridge:
    """Validated runtime surface for one morphology parameterization."""


    def __init__(self, parameterization: MorphologyParameterization):
        if not isinstance(parameterization, MorphologyParameterization):
            raise TypeError(
                "parameterization must implement MorphologyParameterization"
            )
        self._parameterization = parameterization
        self._context = MorphologyRuntimeContext(
            parameterization_id=parameterization.parameterization_id,
            layout=parameterization.layout,
            specification=getattr(parameterization, "specification", None),
            head_topology=getattr(parameterization, "head_topology", None),
        )
        initial = parameterization.validate_parameters(
            parameterization.initial_parameters
        )
        bounds = tuple(parameterization.bounds())
        if len(bounds) != initial.size:
            raise ValueError(
                f"morphology bounds dimension {len(bounds)} != "
                f"initial morphology dimension {initial.size}"
            )
        for index, bound in enumerate(bounds):
            if len(bound) != 2:
                raise ValueError(
                    f"morphology bound {index} must contain lower and upper"
                )
            lower, upper = float(bound[0]), float(bound[1])
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise ValueError(
                    f"morphology bound {index} contains non-finite values"
                )
            if lower > upper:
                raise ValueError(
                    f"morphology bound {index} has lower > upper"
                )
            if initial[index] < lower or initial[index] > upper:
                raise ValueError(
                    f"initial morphology coordinate {index} is outside bounds"
                )

    @property
    def parameterization_id(self) -> str:
        return self._parameterization.parameterization_id

    @property
    def layout(self) -> MorphologyLayout:
        return self._parameterization.layout

    @property
    def morphology_dim(self) -> int:
        return int(self.layout.morphology_dim)

    @property
    def redmax_parameter_dim(self) -> int:
        return int(self.layout.redmax_parameter_dim)

    @property
    def active_blocks(self) -> tuple[MorphologyBlock, ...]:
        return self.layout.blocks

    @property
    def initial_morphology(self) -> np.ndarray:
        return self._parameterization.initial_parameters

    @property
    def diagnostic_context(self) -> MorphologyRuntimeContext:
        return self._context

    def bounds(
        self,
        *,
        head_parameter_margin: float = 2.0,
    ) -> tuple[tuple[float, float], ...]:
        bounds = tuple(
            self._parameterization.bounds(
                head_parameter_margin=head_parameter_margin,
            )
        )
        if len(bounds) != self.morphology_dim:
            raise ValueError(
                f"morphology bounds dimension {len(bounds)} != "
                f"layout dimension {self.morphology_dim}"
            )
        return bounds

    def joint_layout(self, action_dim: int) -> JointParameterLayout:
        return JointParameterLayout(
            action_dim=action_dim,
            morphology_layout=self.layout,
        )

    def validate_morphology(self, morphology: np.ndarray) -> np.ndarray:
        return self._parameterization.validate_parameters(morphology)

    def migrate_legacy_morphology(
        self,
        legacy_morphology: np.ndarray,
    ) -> np.ndarray:
        migration = getattr(
            self._parameterization,
            "migrate_legacy_numpy",
            None,
        )
        if migration is None:
            raise ValueError(
                f"{self.parameterization_id!r} does not support legacy "
                "morphology migration"
            )
        migrated = migration(legacy_morphology)
        return self.validate_morphology(migrated)

    def parameterize_numpy(
        self,
        morphology: np.ndarray,
        *,
        generate_mesh: bool = False,
    ):
        result = self._parameterization.parameterize_numpy(
            morphology,
            generate_mesh=generate_mesh,
        )
        design_params = result[0] if generate_mesh else result
        self._validate_redmax_numpy(design_params)
        return result

    def parameterize_torch(
        self,
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        design_params = self._parameterization.parameterize_torch(morphology)
        if design_params.ndim != 1:
            raise ValueError(
                "Torch RedMax design parameters must be one-dimensional"
            )
        if design_params.numel() != self.redmax_parameter_dim:
            raise ValueError(
                f"Torch RedMax parameter dimension {design_params.numel()} != "
                f"{self.redmax_parameter_dim}"
            )
        return design_params

    def release_torch_graph_refs(self) -> None:
        self._parameterization.release_torch_graph_refs()

    def connection_diagnostics(
        self,
        morphology: np.ndarray,
    ):
        return self._parameterization.connection_diagnostics(morphology)

    def apply(
        self,
        simulation: Any,
        morphology: np.ndarray,
        *,
        generate_mesh: bool = False,
    ):
        result = self._parameterization.apply(
            simulation,
            morphology,
            generate_mesh=generate_mesh,
        )
        design_params = result[0] if isinstance(result, tuple) else result
        self._validate_redmax_numpy(design_params)
        return result

    def project_tangent(
        self,
        morphology: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        return self._parameterization.project_tangent(
            morphology,
            vector,
        )

    def retract(
        self,
        morphology: np.ndarray,
        step: np.ndarray,
        *,
        max_shape_displacement: float = 0.125,
        mount_face_tolerance: float = 1e-7,
    ) -> MorphologyRetractionResult:
        return self._parameterization.retract(
            morphology,
            step,
            max_shape_displacement=max_shape_displacement,
            mount_face_tolerance=mount_face_tolerance,
        )

    def validate_geometry(
        self,
        morphology: np.ndarray,
        *,
        max_shape_displacement: float = 0.125,
        mount_face_tolerance: float = 1e-7,
    ) -> None:
        self._parameterization.validate_geometry(
            morphology,
            max_shape_displacement=max_shape_displacement,
            mount_face_tolerance=mount_face_tolerance,
        )

    def collision_report(
        self,
        morphology: np.ndarray,
        *,
        policy: MorphologyCollisionPolicy,
        xml_path: str | Path | None = None,
    ) -> Any | None:
        return self._parameterization.collision_report(
            morphology,
            policy=policy,
            xml_path=xml_path,
        )

    def validate_collision_transition(
        self,
        previous: np.ndarray,
        candidate: np.ndarray,
        *,
        policy: MorphologyCollisionPolicy,
        xml_path: str | Path | None = None,
    ) -> MorphologyCollisionDecision:
        return self._parameterization.validate_collision_transition(
            previous,
            candidate,
            policy=policy,
            xml_path=xml_path,
        )

    def _validate_redmax_numpy(self, design_params: np.ndarray) -> None:
        values = np.asarray(design_params)
        if values.ndim != 1:
            raise ValueError(
                "NumPy RedMax design parameters must be one-dimensional"
            )
        if values.size != self.redmax_parameter_dim:
            raise ValueError(
                f"NumPy RedMax parameter dimension {values.size} != "
                f"{self.redmax_parameter_dim}"
            )


__all__ = [
    "JointParameterLayout",
    "MorphologyRuntimeBridge",
    "MorphologyRuntimeContext",
]
