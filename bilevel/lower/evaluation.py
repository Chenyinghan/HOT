"""Versioned identity and provenance for one lower-level evaluation.

Skeleton and XML identity intentionally live in the upper-level search.  This
module identifies the *evaluation* of an already generated XML so scores from
different tasks, optimizers, parameter layouts, or simulator builds cannot be
silently reused.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


EVALUATION_IDENTITY_SCHEMA = "bilevel-evaluation-identity"
EVALUATION_IDENTITY_SCHEMA_VERSION = 1
EVALUATION_RESULT_SCHEMA = "bilevel-evaluation-result"
EVALUATION_RESULT_SCHEMA_VERSION = 1

_EPHEMERAL_CONTEXT_KEYS = {
    "cache_dir",
    "output_dir",
    "replay_dir",
    "xml_path",
    "best_xml",
    "best_run_json",
    "sequence",
    "bass_config",
    "debug",
    "evaluation",
    "phase",
    "visualize_step",
    "render",
    "allow_replay_param_mismatch",
    "evaluation_identity",
    "evaluation_key",
    "task_module_path",
    "low_level_spawn_method",
}


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _canonical(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if hasattr(value, "tolist"):
        return _canonical(value.tolist())
    if hasattr(value, "item"):
        try:
            return _canonical(value.item())
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        _canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=64)
def _file_identity_cached(
    resolved_path: str,
    size: int,
    mtime_ns: int,
) -> dict[str, Any]:
    _ = (size, mtime_ns)
    path = Path(resolved_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "sha256": digest,
    }


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _file_identity_cached(
        str(resolved),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def redmax_pythonpath_entries(repo_root: str | Path) -> list[str]:
    """Prefer the current inplace extension; use build output only as fallback."""

    override = os.environ.get("HOT_REDMAX_DIR")
    if override:
        directories = [
            str(Path(entry).expanduser().resolve())
            for entry in override.split(os.pathsep)
            if entry.strip()
        ]
        if not directories:
            raise ValueError("HOT_REDMAX_DIR is empty")
        missing = [
            directory
            for directory in directories
            if not tuple(Path(directory).glob("redmax_py*.so"))
        ]
        if missing:
            raise FileNotFoundError(
                "HOT_REDMAX_DIR contains no redmax_py extension: "
                + ", ".join(missing)
            )
        return directories

    core_dir = Path(repo_root).resolve() / "core"
    if tuple(core_dir.glob("redmax_py*.so")):
        return [str(core_dir)]
    build_root = core_dir / "build"
    if not build_root.exists():
        return []
    return [
        str(path)
        for path in sorted(build_root.glob("lib.*"))
        if path.is_dir()
    ]


def simulator_identity(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    candidates = []
    for directory in redmax_pythonpath_entries(root):
        candidates.extend(sorted(Path(directory).glob("redmax_py*.so")))
    if not candidates:
        return {
            "engine": "redmax",
            "binary": None,
            "available": False,
        }
    binary = file_identity(candidates[0])
    try:
        binary["path"] = str(
            candidates[0].resolve().relative_to(root)
        )
    except ValueError:
        pass
    return {
        "engine": "redmax",
        "binary": binary,
        "available": True,
    }


def _morphology_implementation_identity(repo_root: Path) -> dict[str, Any]:
    morphology_dir = repo_root / "bilevel" / "parameterization"
    files = sorted(morphology_dir.glob("*.py"))
    manifest = [
        {
            "path": str(path.resolve().relative_to(repo_root)),
            "sha256": file_identity(path)["sha256"],
        }
        for path in files
    ]
    return {
        "files": manifest,
        "sha256": stable_digest(manifest),
    }


@dataclass(frozen=True)
class EvaluationIdentity:
    """Canonical request identity used for lower-level score reuse."""

    payload: Mapping[str, Any]
    digest: str
    schema: str = EVALUATION_IDENTITY_SCHEMA
    schema_version: int = EVALUATION_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        canonical_payload = _canonical(self.payload)
        expected = stable_digest(canonical_payload)
        if str(self.digest) != expected:
            raise ValueError("evaluation identity digest does not match payload")
        if self.schema != EVALUATION_IDENTITY_SCHEMA:
            raise ValueError("unexpected evaluation identity schema")
        if int(self.schema_version) != EVALUATION_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unexpected evaluation identity schema version")
        object.__setattr__(self, "payload", canonical_payload)

    @property
    def key(self) -> str:
        return self.digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": int(self.schema_version),
            "digest": self.digest,
            "payload": _canonical(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationIdentity":
        return cls(
            payload=dict(value["payload"]),
            digest=str(value["digest"]),
            schema=str(value["schema"]),
            schema_version=int(value["schema_version"]),
        )


def build_evaluation_identity(
    *,
    repo_root: str | Path,
    xml_path: str | Path,
    task_module_path: str | Path,
    task_json: Mapping[str, Any],
    context: Mapping[str, Any],
) -> EvaluationIdentity:
    """Build an identity without changing upper-level skeleton/XML identity."""

    root = Path(repo_root).resolve()
    xml = Path(xml_path).resolve()
    task_module = Path(task_module_path)
    if not task_module.is_absolute():
        task_module = root / task_module
    task_config = dict(task_json.get("task_config", {}) or {})
    behavior_context = {
        str(key): value
        for key, value in context.items()
        if key not in _EPHEMERAL_CONTEXT_KEYS and key != "task_json"
    }
    if not bool(behavior_context.get("experimental", False)):
        behavior_context.pop("experimental_best_loss", None)
    morphology_id = context.get(
        "morphology_parameterization",
        task_config.get("morphology_parameterization"),
    )
    xml_identity = file_identity(xml)
    xml_identity.pop("path", None)
    task_module_identity = file_identity(task_module)
    try:
        task_module_identity["path"] = str(
            task_module.resolve().relative_to(root)
        )
    except ValueError:
        pass
    morphology_implementation = _morphology_implementation_identity(root)
    morphology_layout_identity = stable_digest(
        {
            "xml_sha256": xml_identity["sha256"],
            "requested_parameterization": (
                None
                if morphology_id is None or not str(morphology_id).strip()
                else str(morphology_id).strip()
            ),
            "generic_design_protocol": context.get(
                "generic_design_protocol",
                task_config.get("generic_design_protocol"),
            ),
            "implementation_sha256": morphology_implementation["sha256"],
        }
    )
    payload = {
        "xml": xml_identity,
        "task": {
            "mission_name": task_json.get(
                "mission_name",
                task_json.get("task_name"),
            ),
            "functions": task_json.get("functions"),
            "task_config": task_config,
            "task_module": task_module_identity,
        },
        "runtime": behavior_context,
        "parameterization": {
            "morphology_parameterization": (
                None
                if morphology_id is None or not str(morphology_id).strip()
                else str(morphology_id).strip()
            ),
            "generic_design_protocol": context.get(
                "generic_design_protocol",
                task_config.get("generic_design_protocol"),
            ),
            "force_connectivity": bool(
                context.get(
                    "force_connectivity",
                    task_config.get("force_connectivity", False),
                )
            ),
            "implementation": morphology_implementation,
            "morphology_layout_identity": (
                morphology_layout_identity
            ),
        },
        "simulator": simulator_identity(root),
    }
    return EvaluationIdentity(
        payload=payload,
        digest=stable_digest(payload),
    )


def result_parameter_fields(
    metadata_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not metadata_payload:
        return {
            "action_dim": None,
            "morphology_dim": None,
            "total_dim": None,
            "morphology_parameterization": None,
            "morphology_layout_fingerprint": None,
        }
    return {
        "action_dim": int(metadata_payload["action_dim"]),
        "morphology_dim": int(metadata_payload["morphology_dim"]),
        "total_dim": int(metadata_payload["total_dim"]),
        "morphology_parameterization": metadata_payload.get(
            "morphology_parameterization"
        ),
        "morphology_layout_fingerprint": metadata_payload.get(
            "morphology_layout_fingerprint"
        ),
    }


def enrich_evaluation_result(
    result: Mapping[str, Any],
    *,
    identity: EvaluationIdentity,
    metadata_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach the shared result/replay/cache contract without changing scores."""

    enriched = dict(result)
    if metadata_payload is None:
        embedded = enriched.get("parameter_artifact_metadata")
        if isinstance(embedded, Mapping):
            metadata_payload = embedded
    error = enriched.get("error")
    score = enriched.get("score", enriched.get("loss"))
    success = error is None
    try:
        success = success and score is not None and float(score) < float("inf")
    except (TypeError, ValueError):
        success = False
    error_text = str(error or "").lower()
    try:
        returncode = int(enriched.get("returncode", 0) or 0)
    except (TypeError, ValueError):
        returncode = 0
    if success:
        status = "success"
    elif "timeout" in error_text:
        status = "timeout"
    elif (
        "initial" in error_text
        or "preflight" in error_text
        or "collision_invalid" in error_text
    ):
        status = "invalid_initial_geometry"
    elif "parameter" in error_text or "morphology" in error_text:
        status = "invalid_parameterization"
    elif "redmax" in error_text and "load" in error_text:
        status = "redmax_load_failed"
    elif (
        "native" in error_text
        or returncode < 0
    ):
        status = "native_crash"
    else:
        status = "optimizer_failed"
    enriched.update(
        {
            "result_schema": EVALUATION_RESULT_SCHEMA,
            "result_schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
            "status": status,
            "evaluation_key": identity.key,
            "evaluation_identity": identity.to_dict(),
            "simulator_identity": identity.payload["simulator"],
            **result_parameter_fields(metadata_payload),
        }
    )
    if metadata_payload is not None:
        enriched["parameter_artifact_metadata"] = _canonical(
            metadata_payload
        )
    return enriched


def validate_result_identity(
    result: Mapping[str, Any],
    expected: EvaluationIdentity,
) -> None:
    payload = result.get("evaluation_identity")
    if not isinstance(payload, Mapping):
        raise ValueError("result has no versioned evaluation identity")
    actual = EvaluationIdentity.from_dict(payload)
    if actual.key != expected.key:
        raise ValueError(
            "result evaluation identity does not match the requested evaluation"
        )
    if str(result.get("evaluation_key", "")) != expected.key:
        raise ValueError("result evaluation_key does not match its identity")


def validate_replay_provenance(
    evaluation: Mapping[str, Any],
    *,
    repo_root: str | Path,
    allow_simulator_mismatch: bool = False,
) -> dict[str, Any]:
    """Validate saved identity and simulator provenance before replay."""

    nested = (
        evaluation.get("result")
        if isinstance(evaluation.get("result"), Mapping)
        else None
    )
    candidate = evaluation
    if not isinstance(candidate.get("evaluation_identity"), Mapping):
        candidate = nested or {}
    identity_payload = candidate.get("evaluation_identity")
    if not isinstance(identity_payload, Mapping):
        return {
            "verified": False,
            "legacy_unversioned": True,
            "reason": "evaluation identity is absent",
        }
    identity = EvaluationIdentity.from_dict(identity_payload)
    if str(candidate.get("evaluation_key", "")) != identity.key:
        raise ValueError(
            "saved evaluation_key does not match evaluation identity"
        )
    current_simulator = simulator_identity(repo_root)
    saved_simulator = identity.payload.get("simulator")
    simulator_matches = False
    if isinstance(saved_simulator, Mapping):
        saved_binary = saved_simulator.get("binary")
        current_binary = current_simulator.get("binary")
        if isinstance(saved_binary, Mapping) and isinstance(
            current_binary, Mapping
        ):
            simulator_matches = (
                saved_simulator.get("engine")
                == current_simulator.get("engine")
                and bool(saved_simulator.get("available"))
                == bool(current_simulator.get("available"))
                and saved_binary.get("sha256")
                == current_binary.get("sha256")
                and int(saved_binary.get("size", -1))
                == int(current_binary.get("size", -2))
            )
        else:
            simulator_matches = (
                stable_digest(saved_simulator)
                == stable_digest(current_simulator)
            )
    if not simulator_matches:
        if allow_simulator_mismatch:
            return {
                "verified": False,
                "legacy_unversioned": False,
                "evaluation_key": identity.key,
                "reason": "simulator_binary_mismatch_explicitly_allowed",
                "saved_simulator_identity": saved_simulator,
                "simulator_identity": current_simulator,
            }
        raise ValueError(
            "saved evaluation used a different RedMax simulator binary"
        )
    return {
        "verified": True,
        "legacy_unversioned": False,
        "evaluation_key": identity.key,
        "simulator_identity": current_simulator,
    }


__all__ = [
    "EVALUATION_IDENTITY_SCHEMA",
    "EVALUATION_IDENTITY_SCHEMA_VERSION",
    "EVALUATION_RESULT_SCHEMA",
    "EVALUATION_RESULT_SCHEMA_VERSION",
    "EvaluationIdentity",
    "build_evaluation_identity",
    "enrich_evaluation_result",
    "file_identity",
    "redmax_pythonpath_entries",
    "result_parameter_fields",
    "simulator_identity",
    "stable_digest",
    "validate_result_identity",
    "validate_replay_provenance",
]
