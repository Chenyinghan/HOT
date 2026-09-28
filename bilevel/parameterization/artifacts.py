"""Versioned, self-describing artifacts for joint lower-level parameters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import UNIFIED_PARAMETERIZATION_ID


PARAMETER_ARTIFACT_SCHEMA = "bilevel_joint_parameters"
PARAMETER_ARTIFACT_SCHEMA_VERSION = 1
CONNECTED_DIRECT_PARAMETERIZATION_ID = (
    "connected_direct_planar_hexahedron"
)

_REQUIRED_METADATA_KEYS = frozenset(
    {
        "parameter_artifact_schema",
        "schema_version",
        "action_parameterization",
        "morphology_parameterization",
        "action_dim",
        "morphology_dim",
        "total_dim",
        "action_slice",
        "morphology_slice",
        "morphology_layout_fingerprint",
        "morphology_layout",
    }
)
_VERSIONED_METADATA_MARKERS = _REQUIRED_METADATA_KEYS - {
    "action_parameterization"
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("artifact metadata contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, slice):
        if value.step not in (None, 1):
            raise ValueError("artifact slices must have unit stride")
        return [int(value.start), int(value.stop)]
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(
        "artifact metadata is not JSON-serializable: "
        f"{type(value).__name__}"
    )


def stable_layout_fingerprint(layout_manifest: Any) -> str:
    """Hash a canonical JSON representation of one morphology layout."""

    canonical = json.dumps(
        _jsonable(layout_manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if integer < 0 or integer != value:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def _strict_slice(value: Any, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a two-element [start, stop] list")
    start = _strict_nonnegative_int(value[0], name=f"{name}[0]")
    stop = _strict_nonnegative_int(value[1], name=f"{name}[1]")
    if stop < start:
        raise ValueError(f"{name} stop must not precede start")
    return start, stop


@dataclass(frozen=True)
class JointParameterArtifactMetadata:
    """Complete interpretation contract for one saved joint vector."""

    action_parameterization: str
    morphology_parameterization: str | None
    action_dim: int
    morphology_dim: int
    total_dim: int
    action_slice: tuple[int, int]
    morphology_slice: tuple[int, int]
    morphology_layout_fingerprint: str
    morphology_layout: Any
    schema_version: int = PARAMETER_ARTIFACT_SCHEMA_VERSION
    parameter_artifact_schema: str = PARAMETER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.parameter_artifact_schema != PARAMETER_ARTIFACT_SCHEMA:
            raise ValueError(
                "unsupported parameter artifact schema "
                f"{self.parameter_artifact_schema!r}"
            )
        schema_version = _strict_nonnegative_int(
            self.schema_version,
            name="schema_version",
        )
        if schema_version != PARAMETER_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported parameter artifact schema version "
                f"{self.schema_version!r}"
            )
        if not str(self.action_parameterization).strip():
            raise ValueError("action_parameterization must be non-empty")

        action_dim = _strict_nonnegative_int(
            self.action_dim,
            name="action_dim",
        )
        morphology_dim = _strict_nonnegative_int(
            self.morphology_dim,
            name="morphology_dim",
        )
        total_dim = _strict_nonnegative_int(
            self.total_dim,
            name="total_dim",
        )
        action_slice = _strict_slice(
            self.action_slice,
            name="action_slice",
        )
        morphology_slice = _strict_slice(
            self.morphology_slice,
            name="morphology_slice",
        )
        if action_slice != (0, action_dim):
            raise ValueError(
                "action_slice must be exactly [0, action_dim]"
            )
        if morphology_slice != (
            action_dim,
            action_dim + morphology_dim,
        ):
            raise ValueError(
                "morphology_slice must immediately follow action_slice"
            )
        if total_dim != action_dim + morphology_dim:
            raise ValueError(
                "total_dim must equal action_dim + morphology_dim"
            )
        morphology_parameterization = self.morphology_parameterization
        if morphology_dim == 0:
            if morphology_parameterization is not None:
                raise ValueError(
                    "action-only metadata cannot name a morphology "
                    "parameterization"
                )
            if self.morphology_layout is not None:
                raise ValueError(
                    "action-only metadata must have a null morphology layout"
                )
        else:
            if not str(morphology_parameterization or "").strip():
                raise ValueError(
                    "morphology_parameterization is required when "
                    "morphology_dim is nonzero"
                )
            if not isinstance(self.morphology_layout, Mapping):
                raise ValueError(
                    "morphology_layout must be an object when morphology is "
                    "present"
                )
            layout_parameterization = self.morphology_layout.get(
                "parameterization_id"
            )
            if layout_parameterization != morphology_parameterization:
                raise ValueError(
                    "morphology layout parameterization does not match "
                    "morphology_parameterization"
                )
            if self.morphology_layout.get("morphology_dim") != morphology_dim:
                raise ValueError(
                    "morphology layout dimension does not match "
                    "morphology_dim"
                )

        fingerprint = stable_layout_fingerprint(self.morphology_layout)
        if self.morphology_layout_fingerprint != fingerprint:
            raise ValueError(
                "morphology layout fingerprint does not match its manifest"
            )
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "morphology_dim", morphology_dim)
        object.__setattr__(self, "total_dim", total_dim)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "action_slice", action_slice)
        object.__setattr__(self, "morphology_slice", morphology_slice)
        object.__setattr__(
            self,
            "morphology_layout",
            _jsonable(self.morphology_layout),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_artifact_schema": self.parameter_artifact_schema,
            "schema_version": self.schema_version,
            "action_parameterization": self.action_parameterization,
            "morphology_parameterization": (
                self.morphology_parameterization
            ),
            "action_dim": self.action_dim,
            "morphology_dim": self.morphology_dim,
            "total_dim": self.total_dim,
            "action_slice": list(self.action_slice),
            "morphology_slice": list(self.morphology_slice),
            "morphology_layout_fingerprint": (
                self.morphology_layout_fingerprint
            ),
            "morphology_layout": _jsonable(self.morphology_layout),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "JointParameterArtifactMetadata":
        if not isinstance(payload, Mapping):
            raise ValueError("parameter artifact metadata must be an object")
        missing = sorted(_REQUIRED_METADATA_KEYS - set(payload))
        if missing:
            raise ValueError(
                "incomplete parameter artifact metadata; missing "
                + ", ".join(missing)
            )
        return cls(
            parameter_artifact_schema=str(
                payload["parameter_artifact_schema"]
            ),
            schema_version=payload["schema_version"],
            action_parameterization=str(
                payload["action_parameterization"]
            ),
            morphology_parameterization=(
                None
                if payload["morphology_parameterization"] is None
                else str(payload["morphology_parameterization"])
            ),
            action_dim=payload["action_dim"],
            morphology_dim=payload["morphology_dim"],
            total_dim=payload["total_dim"],
            action_slice=_strict_slice(
                payload["action_slice"],
                name="action_slice",
            ),
            morphology_slice=_strict_slice(
                payload["morphology_slice"],
                name="morphology_slice",
            ),
            morphology_layout_fingerprint=str(
                payload["morphology_layout_fingerprint"]
            ),
            morphology_layout=payload["morphology_layout"],
        )


def parse_parameter_artifact_metadata(
    payload: Mapping[str, Any] | None,
) -> JointParameterArtifactMetadata | None:
    """Parse complete metadata; permit only the pre-Phase-D action-only form."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("parameter artifact metadata must be an object")
    keys = set(payload)
    if not (keys & _VERSIONED_METADATA_MARKERS):
        return None
    return JointParameterArtifactMetadata.from_dict(payload)


def read_parameter_artifact_metadata(
    path: str | Path,
) -> dict[str, Any] | None:
    metadata_path = Path(path)
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(
            f"parameter artifact metadata must be an object: {metadata_path}"
        )
    return payload


def _record_asset_id(record: Any) -> str:
    explicit = str(
        getattr(record, "attrs", {}).get("asset_id", "")
    ).strip()
    if explicit:
        return explicit
    mesh_path = getattr(record, "mesh_path", None)
    if mesh_path is not None:
        return f"legacy_asset/{Path(mesh_path).stem}"
    return f"legacy_record/{getattr(record, 'link_name', 'unknown')}"


def _record_redmax_slices(record: Any) -> list[dict[str, Any]]:
    slices = []
    for group in ("p1", "p2", "p3", "p4", "p5", "p6"):
        value = getattr(record, f"{group}_slice", None)
        if value is None:
            continue
        slices.append(
            {
                "group": group,
                "start": int(value.start),
                "stop": int(value.stop),
            }
        )
    return slices


def legacy_morphology_layout_manifest(bundle: Any) -> dict[str, Any]:
    """Describe the exact optimizer-facing layout of a legacy DesignBundle."""

    parameterization_id = str(
        getattr(bundle, "generic_design_protocol", "")
    ).strip()
    if not parameterization_id:
        parameterization_id = (
            f"legacy/{type(bundle).__module__}.{type(bundle).__name__}"
        )
    initial = np.asarray(
        getattr(bundle, "init_cage_params", np.zeros(0)),
        dtype=np.float64,
    )
    if initial.ndim != 1 or not np.all(np.isfinite(initial)):
        raise ValueError(
            "legacy bundle initial morphology must be a finite vector"
        )
    redmax_dim = int(getattr(getattr(bundle, "spec", None), "ndof_p", 0))
    blocks = []
    fixed_indices: set[int] = set()
    records = list(
        getattr(getattr(bundle, "spec", None), "tool_records", ()) or ()
    )
    source_blocks = list(
        getattr(bundle, "direct_planar_blocks", ()) or ()
    )
    if parameterization_id == CONNECTED_DIRECT_PARAMETERIZATION_ID:
        if len(records) != len(source_blocks):
            raise ValueError(
                "connected-direct legacy record/block count mismatch"
            )
    for index, source in enumerate(source_blocks):
        start = int(source["start"])
        stop = int(source["stop"])
        if start < 0 or stop <= start or stop > initial.size:
            raise ValueError(
                f"invalid legacy morphology block [{start}:{stop}]"
            )
        deformable = bool(source.get("deformable", True))
        if not deformable:
            fixed_indices.update(range(start, stop))
        record = records[index] if index < len(records) else None
        blocks.append(
            {
                "tool_index": int(source.get("tool_index", index)),
                "node_id": (
                    None
                    if record is None
                    or getattr(record, "node_id", None) is None
                    else int(record.node_id)
                ),
                "asset_id": (
                    None if record is None else _record_asset_id(record)
                ),
                "start": start,
                "stop": stop,
                "dimension": stop - start,
                "deformable": deformable,
                "redmax_design_mask": (
                    None
                    if record is None
                    else int(getattr(record, "mask", 0))
                ),
                "redmax_slices": (
                    [] if record is None else _record_redmax_slices(record)
                ),
                "link_name": str(source.get("link_name", "")),
                "body_name": str(source.get("body_name", "")),
                "connected_face_mask": [
                    bool(value)
                    for value in source.get("face_mask", ())
                ],
                "parent_face": (
                    None
                    if record is None
                    else getattr(record, "planar_parent_face", None)
                ),
                "child_face": (
                    None
                    if record is None
                    else getattr(record, "planar_child_face", None)
                ),
                "endeffector_face": (
                    None
                    if record is None
                    else getattr(
                        record,
                        "planar_endeffector_face",
                        None,
                    )
                ),
            }
        )
    return {
        "parameterization_id": parameterization_id,
        "morphology_dim": int(initial.size),
        "redmax_parameter_dim": redmax_dim,
        "blocks": blocks,
        "fixed_coordinates": [
            {
                "index": int(index),
                "value": float(initial[index]),
            }
            for index in sorted(fixed_indices)
        ],
    }


def metadata_for_runner(runner: Any) -> JointParameterArtifactMetadata:
    action_dim = int(runner.action_parameter_dim)
    morphology_dim = int(runner.ndof_morphology)
    if morphology_dim == 0:
        parameterization = None
        layout_manifest = None
    elif runner.morphology_runtime is not None:
        parameterization = runner.morphology_runtime.parameterization_id
        layout_manifest = runner.morphology_runtime.layout.to_manifest()
    else:
        if runner.design_bundle is None:
            raise ValueError(
                "runner reports morphology coordinates without a design bundle"
            )
        layout_manifest = legacy_morphology_layout_manifest(
            runner.design_bundle
        )
        parameterization = str(
            layout_manifest["parameterization_id"]
        )
    total_dim = action_dim + morphology_dim
    return JointParameterArtifactMetadata(
        action_parameterization=runner.action_parameterization,
        morphology_parameterization=parameterization,
        action_dim=action_dim,
        morphology_dim=morphology_dim,
        total_dim=total_dim,
        action_slice=(0, action_dim),
        morphology_slice=(action_dim, total_dim),
        morphology_layout_fingerprint=stable_layout_fingerprint(
            layout_manifest
        ),
        morphology_layout=layout_manifest,
    )


def validate_parameter_vector(
    params: np.ndarray,
    *,
    expected_dim: int,
) -> np.ndarray:
    values = np.asarray(params, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"joint parameter vector must be one-dimensional, got "
            f"{values.shape}"
        )
    if values.size != int(expected_dim):
        raise ValueError(
            f"joint parameter vector has dimension {values.size}, expected "
            f"exactly {int(expected_dim)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("joint parameter vector contains non-finite values")
    return values


def normalize_parameters_for_runner(
    runner: Any,
    params: np.ndarray,
    metadata_payload: Mapping[str, Any] | None,
    *,
    legacy_action_parameterization: str | None = None,
) -> np.ndarray:
    """Validate, migrate if explicitly safe, and convert action coordinates."""

    source = parse_parameter_artifact_metadata(metadata_payload)
    target = metadata_for_runner(runner)
    if source is None:
        values = validate_parameter_vector(
            params,
            expected_dim=target.total_dim,
        )
        if (
            target.morphology_parameterization
            == UNIFIED_PARAMETERIZATION_ID
            and target.morphology_dim
        ):
            raise ValueError(
                "unversioned morphology parameters are ambiguous for the "
                "unified layout; complete artifact metadata is required"
            )
        source_action = legacy_action_parameterization
        if source_action is None and isinstance(metadata_payload, Mapping):
            source_action = metadata_payload.get(
                "action_parameterization"
            )
        return runner.convert_action_parameterization(
            values,
            None if source_action is None else str(source_action),
        )

    values = validate_parameter_vector(
        params,
        expected_dim=source.total_dim,
    )
    if source.action_dim != target.action_dim:
        raise ValueError(
            "saved action dimension does not match the current runner: "
            f"{source.action_dim} != {target.action_dim}"
        )
    action = values[
        source.action_slice[0] : source.action_slice[1]
    ].copy()
    source_morphology = values[
        source.morphology_slice[0] : source.morphology_slice[1]
    ].copy()

    if (
        source.morphology_parameterization
        == target.morphology_parameterization
    ):
        if source.morphology_dim != target.morphology_dim:
            raise ValueError(
                "saved morphology dimension does not match the current "
                "runner"
            )
        if (
            source.morphology_layout_fingerprint
            != target.morphology_layout_fingerprint
        ):
            raise ValueError(
                "saved morphology layout does not match the current runner"
            )
        morphology = source_morphology
    elif (
        source.morphology_parameterization
        == CONNECTED_DIRECT_PARAMETERIZATION_ID
        and target.morphology_parameterization
        == UNIFIED_PARAMETERIZATION_ID
    ):
        if runner.morphology_runtime is None:
            raise ValueError(
                "unified migration requires a morphology runtime"
            )
        current_legacy_manifest = legacy_morphology_layout_manifest(
            runner.design_bundle
        )
        current_legacy_fingerprint = stable_layout_fingerprint(
            current_legacy_manifest
        )
        if (
            source.morphology_layout_fingerprint
            != current_legacy_fingerprint
        ):
            raise ValueError(
                "legacy morphology layout does not match the current "
                "Handle-root model"
            )
        morphology = runner.morphology_runtime.migrate_legacy_morphology(
            source_morphology
        )
    else:
        raise ValueError(
            "unsupported morphology parameterization conversion: "
            f"{source.morphology_parameterization!r} -> "
            f"{target.morphology_parameterization!r}"
        )

    packed = runner.pack_params(
        action,
        morphology if target.morphology_dim else None,
    )
    return runner.convert_action_parameterization(
        packed,
        source.action_parameterization,
    )
