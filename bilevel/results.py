"""Canonical, structured result schema for one evaluated skeleton."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


class EvaluationStatus(str, Enum):
    SUCCESS = "success"
    COMPILATION_FAILED = "compilation_failed"
    INVALID_INITIAL_GEOMETRY = "invalid_initial_geometry"
    REDMAX_LOAD_FAILED = "redmax_load_failed"
    INVALID_PARAMETERIZATION = "invalid_parameterization"
    OPTIMIZER_FAILED = "optimizer_failed"
    TIMEOUT = "timeout"
    NATIVE_CRASH = "native_crash"


def _name(value: str, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must be non-empty")
    return result


def _optional_finite(value: Optional[float], field_name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite when provided")
    return result


def _finite_values(values: Sequence[float], field_name: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain only finite values")
    return result


@dataclass(frozen=True)
class BilevelResult:
    """Versioned result shared by search, replay, diagnostics, and tests."""

    task: str
    status: EvaluationStatus
    skeleton: Mapping[str, Any]
    skeleton_hash: str
    root_asset_id: str
    head_asset_ids: Tuple[str, ...]
    score: Optional[float] = None
    loss: Optional[float] = None
    loss_terms: Mapping[str, float] = field(default_factory=dict)
    action_parameterization: str = ""
    action: Tuple[float, ...] = ()
    morphology_parameterization: str = ""
    morphology: Tuple[float, ...] = ()
    action_dim: Optional[int] = None
    morphology_dim: Optional[int] = None
    total_dim: Optional[int] = None
    morphology_layout_fingerprint: str = ""
    evaluation_key: str = ""
    evaluation_identity: Mapping[str, Any] = field(default_factory=dict)
    simulator_identity: Mapping[str, Any] = field(default_factory=dict)
    optimizer: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] = field(default_factory=dict)
    xml_path: Optional[Path] = None
    replay_path: Optional[Path] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise ValueError("schema_version must be 1")
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(self, "task", _name(self.task, "task"))
        status = (
            self.status
            if isinstance(self.status, EvaluationStatus)
            else EvaluationStatus(str(self.status))
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "skeleton", dict(self.skeleton))
        object.__setattr__(
            self,
            "skeleton_hash",
            _name(self.skeleton_hash, "skeleton_hash"),
        )
        object.__setattr__(
            self,
            "root_asset_id",
            _name(self.root_asset_id, "root_asset_id"),
        )
        head_ids = tuple(_name(value, "head_asset_ids") for value in self.head_asset_ids)
        object.__setattr__(self, "head_asset_ids", head_ids)
        score = _optional_finite(self.score, "score")
        loss = _optional_finite(self.loss, "loss")
        if status is EvaluationStatus.SUCCESS and (score is None or loss is None):
            raise ValueError("successful results require finite score and loss")
        if status is not EvaluationStatus.SUCCESS and not self.error:
            raise ValueError("failed results require structured error details")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "loss", loss)
        loss_terms = {
            _name(key, "loss_terms key"): float(value)
            for key, value in self.loss_terms.items()
        }
        if not all(math.isfinite(value) for value in loss_terms.values()):
            raise ValueError("loss_terms values must be finite")
        object.__setattr__(self, "loss_terms", loss_terms)
        object.__setattr__(
            self,
            "action",
            _finite_values(self.action, "action"),
        )
        object.__setattr__(
            self,
            "morphology",
            _finite_values(self.morphology, "morphology"),
        )
        if self.action and not str(self.action_parameterization).strip():
            raise ValueError("non-empty action requires action_parameterization")
        if self.morphology and not str(self.morphology_parameterization).strip():
            raise ValueError(
                "non-empty morphology requires morphology_parameterization"
            )
        object.__setattr__(
            self,
            "action_parameterization",
            str(self.action_parameterization).strip(),
        )
        object.__setattr__(
            self,
            "morphology_parameterization",
            str(self.morphology_parameterization).strip(),
        )
        for field_name in ("action_dim", "morphology_dim", "total_dim"):
            value = getattr(self, field_name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{field_name} must be nonnegative")
            object.__setattr__(
                self,
                field_name,
                None if value is None else int(value),
            )
        if self.action_dim is not None and self.action:
            if self.action_dim != len(self.action):
                raise ValueError("action_dim does not match action length")
        if self.morphology_dim is not None and self.morphology:
            if self.morphology_dim != len(self.morphology):
                raise ValueError(
                    "morphology_dim does not match morphology length"
                )
        if (
            self.total_dim is not None
            and self.action_dim is not None
            and self.morphology_dim is not None
            and self.total_dim != self.action_dim + self.morphology_dim
        ):
            raise ValueError(
                "total_dim must equal action_dim + morphology_dim"
            )
        object.__setattr__(
            self,
            "morphology_layout_fingerprint",
            str(self.morphology_layout_fingerprint).strip(),
        )
        object.__setattr__(
            self,
            "evaluation_key",
            str(self.evaluation_key).strip(),
        )
        object.__setattr__(
            self,
            "evaluation_identity",
            dict(self.evaluation_identity),
        )
        object.__setattr__(
            self,
            "simulator_identity",
            dict(self.simulator_identity),
        )
        if self.evaluation_identity:
            from .lower.evaluation import EvaluationIdentity

            identity = EvaluationIdentity.from_dict(
                self.evaluation_identity
            )
            if self.evaluation_key != identity.key:
                raise ValueError(
                    "evaluation_key does not match evaluation_identity"
                )
        object.__setattr__(self, "optimizer", dict(self.optimizer))
        object.__setattr__(self, "validation", dict(self.validation))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "error", dict(self.error))
        if self.xml_path is not None:
            object.__setattr__(self, "xml_path", Path(self.xml_path))
        if self.replay_path is not None:
            object.__setattr__(self, "replay_path", Path(self.replay_path))

    @property
    def ok(self) -> bool:
        return self.status is EvaluationStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible mapping with stable canonical field names."""

        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "status": self.status.value,
            "skeleton": dict(self.skeleton),
            "skeleton_hash": self.skeleton_hash,
            "root_asset_id": self.root_asset_id,
            "head_asset_ids": list(self.head_asset_ids),
            "score": self.score,
            "loss": self.loss,
            "loss_terms": dict(self.loss_terms),
            "action_parameterization": self.action_parameterization,
            "action": list(self.action),
            "morphology_parameterization": self.morphology_parameterization,
            "morphology": list(self.morphology),
            "action_dim": self.action_dim,
            "morphology_dim": self.morphology_dim,
            "total_dim": self.total_dim,
            "morphology_layout_fingerprint": (
                self.morphology_layout_fingerprint
            ),
            "evaluation_key": self.evaluation_key,
            "evaluation_identity": dict(self.evaluation_identity),
            "simulator_identity": dict(self.simulator_identity),
            "optimizer": dict(self.optimizer),
            "validation": dict(self.validation),
            "diagnostics": dict(self.diagnostics),
            "error": dict(self.error),
            "xml_path": None if self.xml_path is None else str(self.xml_path),
            "replay_path": None if self.replay_path is None else str(self.replay_path),
        }


__all__ = ["BilevelResult", "EvaluationStatus"]
