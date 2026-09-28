"""Load an optimized action trajectory without importing saved morphology."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np


@dataclass(frozen=True)
class ActionArtifact:
    action: np.ndarray
    action_parameterization: str
    source: Path


def _json_object(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        value = json.load(fp)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _scalar_string(value: Any, *, name: str) -> str:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{name} must contain one string")
        value = value.reshape(()).item()
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _validated_action(
    action: Any,
    *,
    expected_dim: Optional[int],
    source: Path,
) -> np.ndarray:
    result = np.asarray(action, dtype=np.float64).reshape(-1)
    if expected_dim is not None and result.size != int(expected_dim):
        raise ValueError(
            f"action from {source} has dimension {result.size}; "
            f"expected {int(expected_dim)}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"action from {source} contains non-finite values")
    return result.copy()


def _load_params(path: Path, *, expected_dim: Optional[int]) -> ActionArtifact:
    sidecar_names = {
        "params.npy": "params_meta.json",
        "params_checkpoint.npy": "params_checkpoint_meta.json",
        "params_initial.npy": "params_initial_meta.json",
    }
    meta_path = path.with_name(sidecar_names.get(path.name, "params_meta.json"))
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"parameter metadata is required to extract action: {meta_path}"
        )
    metadata = _json_object(meta_path)
    try:
        start, stop = metadata["action_slice"]
        parameterization = metadata["action_parameterization"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid action metadata in {meta_path}") from exc
    params = np.load(str(path), allow_pickle=False)
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    start, stop = int(start), int(stop)
    if start != 0 or stop < start or stop > params.size:
        raise ValueError(f"invalid action_slice in {meta_path}: {[start, stop]}")
    action = _validated_action(
        params[start:stop],
        expected_dim=expected_dim,
        source=path,
    )
    return ActionArtifact(
        action=action,
        action_parameterization=_scalar_string(
            parameterization,
            name="action_parameterization",
        ),
        source=path.resolve(),
    )


def _load_finalized(path: Path, *, expected_dim: Optional[int]) -> ActionArtifact:
    with np.load(str(path), allow_pickle=False) as payload:
        if "action_params" not in payload:
            raise ValueError(f"{path} does not contain action_params")
        if "action_parameterization" not in payload:
            raise ValueError(
                f"{path} does not contain action_parameterization"
            )
        action = _validated_action(
            payload["action_params"],
            expected_dim=expected_dim,
            source=path,
        )
        parameterization = _scalar_string(
            payload["action_parameterization"],
            name="action_parameterization",
        )
    return ActionArtifact(action, parameterization, path.resolve())


def _load_result_json(
    path: Path,
    *,
    expected_dim: Optional[int],
) -> ActionArtifact:
    payload = _json_object(path)
    if "action" not in payload:
        raise ValueError(f"{path} does not contain an action")
    parameterization = payload.get("action_parameterization")
    if parameterization is None:
        metadata = payload.get("parameter_artifact_metadata")
        if isinstance(metadata, Mapping):
            parameterization = metadata.get("action_parameterization")
    if parameterization is None:
        raise ValueError(f"{path} does not identify action_parameterization")
    return ActionArtifact(
        action=_validated_action(
            payload["action"],
            expected_dim=expected_dim,
            source=path,
        ),
        action_parameterization=_scalar_string(
            parameterization,
            name="action_parameterization",
        ),
        source=path.resolve(),
    )


def load_action_artifact(
    source: Union[str, Path],
    *,
    expected_dim: Optional[int] = None,
) -> ActionArtifact:
    """Load the final optimized action from a rollout directory or file."""

    path = Path(source).expanduser()
    if path.is_dir():
        candidates = (
            path / "params.npy",
            path / "finalized_state.npz",
            path / "low_level_result.json",
            path / "params_checkpoint.npy",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"no final action artifact or checkpoint found in {source}"
            )
    if not path.is_file():
        raise FileNotFoundError(f"action artifact does not exist: {path}")
    if path.name in {
        "params.npy",
        "params_checkpoint.npy",
        "params_initial.npy",
    }:
        return _load_params(path, expected_dim=expected_dim)
    if path.suffix == ".npz":
        return _load_finalized(path, expected_dim=expected_dim)
    if path.suffix == ".json":
        return _load_result_json(path, expected_dim=expected_dim)
    raise ValueError(
        "unsupported action artifact; expected a rollout directory, "
        "params*.npy, finalized_state.npz, or low_level_result.json"
    )


__all__ = ["ActionArtifact", "load_action_artifact"]
