"""Task-independent contracts for continuous Head morphology."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


UNIFIED_PARAMETERIZATION_ID = "unified_connected_head_morphology"


@dataclass(frozen=True)
class RedMaxParameterSlice:
    """One named slice in RedMax's global type-major parameter vector."""

    group: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.group not in {"p1", "p2", "p3", "p4", "p5", "p6"}:
            raise ValueError(f"unknown RedMax parameter group {self.group!r}")
        if int(self.start) < 0 or int(self.stop) <= int(self.start):
            raise ValueError(
                f"invalid RedMax slice {self.group}[{self.start}:{self.stop}]"
            )

    @property
    def dimension(self) -> int:
        return int(self.stop) - int(self.start)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "start": int(self.start),
            "stop": int(self.stop),
            "dimension": self.dimension,
        }


@dataclass(frozen=True)
class MorphologyBlock:
    """One active, deformable Head block in the canonical optimization vector."""

    node_id: int
    asset_id: str
    parameterization: str
    start: int
    stop: int
    redmax_design_mask: int
    redmax_slices: tuple[RedMaxParameterSlice, ...]
    link_name: str
    body_name: str
    parent_node_id: int | None
    parent_face: int | None
    child_face: int | None
    endeffector_face: int | None
    connected_face_mask: tuple[bool, ...]
    direct_handle_mount: bool
    frozen_parameter_indices: tuple[int, ...] = ()
    reference_length: float = 1.0

    def __post_init__(self) -> None:
        if int(self.node_id) < 0:
            raise ValueError(f"node_id must be nonnegative, got {self.node_id}")
        if not str(self.asset_id).strip():
            raise ValueError("asset_id must be non-empty")
        if not str(self.parameterization).strip():
            raise ValueError("block parameterization must be non-empty")
        if int(self.start) < 0 or int(self.stop) <= int(self.start):
            raise ValueError(
                f"invalid morphology slice [{self.start}:{self.stop}]"
            )
        if int(self.redmax_design_mask) <= 0:
            raise ValueError("active Head block must have a positive RedMax mask")
        if len(self.connected_face_mask) != 6:
            raise ValueError("connected_face_mask must contain six faces")
        frozen = tuple(int(index) for index in self.frozen_parameter_indices)
        if len(set(frozen)) != len(frozen):
            raise ValueError("frozen_parameter_indices must be unique")
        if any(index < 0 or index >= self.dimension for index in frozen):
            raise ValueError(
                "frozen_parameter_indices must lie inside the morphology block"
            )
        if frozen and not self.direct_handle_mount:
            raise ValueError(
                "only a direct Handle mount may freeze morphology coordinates"
            )
        reference_length = float(self.reference_length)
        if not np.isfinite(reference_length) or reference_length <= 0.0:
            raise ValueError(
                "reference_length must be finite and positive"
            )
        object.__setattr__(self, "frozen_parameter_indices", frozen)
        object.__setattr__(self, "reference_length", reference_length)

    @property
    def dimension(self) -> int:
        return int(self.stop) - int(self.start)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "node_id": int(self.node_id),
            "asset_id": self.asset_id,
            "parameterization": self.parameterization,
            "start": int(self.start),
            "stop": int(self.stop),
            "dimension": self.dimension,
            "redmax_design_mask": int(self.redmax_design_mask),
            "redmax_slices": [
                parameter_slice.to_manifest()
                for parameter_slice in self.redmax_slices
            ],
            "link_name": self.link_name,
            "body_name": self.body_name,
            "parent_node_id": self.parent_node_id,
            "parent_face": self.parent_face,
            "child_face": self.child_face,
            "endeffector_face": self.endeffector_face,
            "connected_face_mask": list(self.connected_face_mask),
            "direct_handle_mount": bool(self.direct_handle_mount),
            "reference_length": self.reference_length,
            "frozen_parameter_indices": list(
                self.frozen_parameter_indices
            ),
        }


@dataclass(frozen=True)
class MorphologyLayout:
    """Deterministic concatenation of active Head morphology blocks."""

    parameterization_id: str
    blocks: tuple[MorphologyBlock, ...]
    morphology_dim: int
    redmax_parameter_dim: int

    def __post_init__(self) -> None:
        if self.parameterization_id != UNIFIED_PARAMETERIZATION_ID:
            raise ValueError(
                "unexpected morphology parameterization "
                f"{self.parameterization_id!r}"
            )
        if int(self.morphology_dim) < 0:
            raise ValueError("morphology_dim must be nonnegative")
        if int(self.redmax_parameter_dim) < 0:
            raise ValueError("redmax_parameter_dim must be nonnegative")
        cursor = 0
        seen_nodes: set[int] = set()
        for block in self.blocks:
            if block.start != cursor:
                raise ValueError(
                    "morphology blocks must be contiguous and deterministic: "
                    f"expected start {cursor}, got {block.start}"
                )
            if block.node_id in seen_nodes:
                raise ValueError(f"duplicate morphology node_id {block.node_id}")
            seen_nodes.add(block.node_id)
            cursor = block.stop
        if cursor != int(self.morphology_dim):
            raise ValueError(
                f"block dimension {cursor} != morphology_dim "
                f"{self.morphology_dim}"
            )

    def block_for_node(self, node_id: int) -> MorphologyBlock:
        for block in self.blocks:
            if block.node_id == int(node_id):
                return block
        raise KeyError(f"no morphology block for node_id {node_id}")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "parameterization_id": self.parameterization_id,
            "morphology_dim": int(self.morphology_dim),
            "redmax_parameter_dim": int(self.redmax_parameter_dim),
            "blocks": [block.to_manifest() for block in self.blocks],
        }


@dataclass(frozen=True)
class MorphologyRetractionResult:
    """Result of retracting a full active-Head morphology step."""

    morphology: np.ndarray
    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class MorphologyCollisionPolicy:
    """External design-preflight policy; never part of morphology variables."""

    enabled: bool = True
    margin: float = 1e-4
    max_report: int = 8
    check_ground: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.margin) or self.margin < 0.0:
            raise ValueError(
                "collision margin must be finite and nonnegative"
            )
        if int(self.max_report) <= 0:
            raise ValueError("collision max_report must be positive")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "margin", float(self.margin))
        object.__setattr__(self, "max_report", int(self.max_report))
        object.__setattr__(self, "check_ground", bool(self.check_ground))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "margin": self.margin,
            "max_report": self.max_report,
            "check_ground": self.check_ground,
        }


@dataclass(frozen=True)
class MorphologyCollisionDecision:
    """Runner-equivalent accept/reject result for a morphology transition."""

    accepted: bool
    report: Any | None
    skipped_unchanged: bool


class MorphologyParameterization(ABC):
    """Common forward and derivative surface for every canonical task."""

    parameterization_id = UNIFIED_PARAMETERIZATION_ID

    @property
    @abstractmethod
    def layout(self) -> MorphologyLayout:
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_parameters(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def bounds(
        self,
        *,
        head_parameter_margin: float = 2.0,
    ) -> tuple[tuple[float, float], ...]:
        """Return bounds in the exact active morphology layout order."""

        raise NotImplementedError

    @abstractmethod
    def parameterize_numpy(
        self,
        morphology: np.ndarray,
        *,
        generate_mesh: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def parameterize_torch(self, morphology: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        simulation: Any,
        morphology: np.ndarray,
        *,
        generate_mesh: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def project_tangent(
        self,
        morphology: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def retract(
        self,
        morphology: np.ndarray,
        step: np.ndarray,
        *,
        max_shape_displacement: float = 0.125,
        mount_face_tolerance: float = 1e-7,
    ) -> MorphologyRetractionResult:
        raise NotImplementedError

    @abstractmethod
    def validate_geometry(
        self,
        morphology: np.ndarray,
        *,
        max_shape_displacement: float = 0.125,
        mount_face_tolerance: float = 1e-7,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def collision_report(
        self,
        morphology: np.ndarray,
        *,
        policy: MorphologyCollisionPolicy,
        xml_path: str | Path | None = None,
    ) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def validate_collision_transition(
        self,
        previous: np.ndarray,
        candidate: np.ndarray,
        *,
        policy: MorphologyCollisionPolicy,
        xml_path: str | Path | None = None,
    ) -> MorphologyCollisionDecision:
        raise NotImplementedError

    def validate_parameters(self, morphology: np.ndarray) -> np.ndarray:
        values = np.asarray(morphology, dtype=np.float64).reshape(-1)
        expected = int(self.layout.morphology_dim)
        if values.shape != (expected,):
            raise ValueError(
                f"morphology shape {values.shape} != expected ({expected},)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("morphology contains non-finite values")
        return values

    def manifest(self) -> Mapping[str, Any]:
        return self.layout.to_manifest()

    def release_torch_graph_refs(self) -> None:
        """Release implementation-owned non-leaf Torch graph references."""

        return None

    def connection_diagnostics(
        self,
        morphology: np.ndarray,
    ) -> Mapping[str, Any] | None:
        """Return optional topology diagnostics in active coordinates."""

        self.validate_parameters(morphology)
        return None


def redmax_slices_from_record(record: Any) -> tuple[RedMaxParameterSlice, ...]:
    slices = []
    for group in ("p1", "p2", "p3", "p4", "p5", "p6"):
        value = getattr(record, f"{group}_slice", None)
        if value is None:
            continue
        slices.append(
            RedMaxParameterSlice(
                group=group,
                start=int(value.start),
                stop=int(value.stop),
            )
        )
    return tuple(slices)


def total_block_dimension(blocks: Iterable[MorphologyBlock]) -> int:
    return sum(block.dimension for block in blocks)
