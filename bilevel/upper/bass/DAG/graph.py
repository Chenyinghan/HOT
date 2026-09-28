"""Offline-compiled immutable grammar tree and lightweight runtime search.

The builder is the only code in this module that operates on ``SearchState``.
The runtime consumes structure-of-arrays artifacts and never calls
``valid_actions``, ``apply_action``, collision checking, or physical
canonicalization while selecting candidates.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import subprocess
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ..actions import Action, AddLink, End, SelectRootRotation, action_from_dict
from ..bayesian_lookahead import (
    MilestonePosterior,
    MilestonePosteriorRegistry,
    MilestoneSchema,
)
from ..feedback import RewardStatistics, StageEvidence
from ..config import BASSConfig
from ..io_assets import AssetSpec
from .canonical import physical_function_signature, physical_signature_digest
from ..scheduler_common import (
    ReadyReservoir,
    SchedulerOccupancyMonitor,
    SchedulerSupplyCriticalGuard,
)
from ..state import (
    SearchState,
    apply_action,
    function_group_constraints_satisfied,
    initial_state,
    valid_actions,
    within_function_count_margin,
)


FORMAT_VERSION = "hot-static-dag-v1"
ACTION_SCHEMA_VERSION = 1
PHYSICAL_SIGNATURE_VERSION = "physical-function-v2"
PARTIAL_SIGNATURE_VERSION = "physical-partial-state-v1"
POSITIONAL_FUNCTION_BINDING = "positional-v1"
UINT32_MAX = np.iinfo(np.uint32).max
UINT64_MAX = np.iinfo(np.uint64).max

NODE_FILES = {
    "first_edge": "node_first_edge.npy",
    "edge_count": "node_edge_count.npy",
    "physical_id": "node_physical_id.npy",
    "rep_parent": "node_rep_parent.npy",
    "rep_action_id": "node_rep_action_id.npy",
}

ACTION_DTYPE = np.dtype(
    [
        ("kind", "u1"),
        ("asset_index", "<u4"),
        ("p", "u1"),
        ("d", "<u4"),
        ("f", "u1"),
        ("q", "u1"),
        ("child_dock_id", "<u4"),
        ("flags", "u1"),
        ("root_rotation_mode", "u1"),
    ],
    align=False,
)


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _box_payload(box: Any) -> Any:
    if box is None:
        return None
    return [[float(value) for value in point] for point in box]


def _asset_payload(asset: AssetSpec) -> dict[str, Any]:
    """Return only asset fields that affect grammar or rigid realization."""

    return {
        "asset_id": asset.asset_id,
        "searchable": bool(asset.searchable),
        "enabled": bool(asset.enabled),
        "half_extents": [float(value) for value in asset.half_extents],
        "out_docks": asset.resolved_out_docks_per_face(),
        "in_docks": asset.resolved_in_docks_per_face(),
    }


def _grammar_payload(config: BASSConfig) -> dict[str, Any]:
    """Return the exact configuration subset that changes reachable states."""

    return {
        "max_total_links": int(config.max_total_links),
        "max_actions_per_expansion": int(config.max_actions_per_expansion),
        "target_function_count": config.target_function_count,
        "function_count_margin": int(config.function_count_margin),
        "function_group_depth_delta": config.function_group_depth_delta,
        "root_asset_id": str(config.root_asset_id),
        "root_blocked_face": config.root_blocked_face,
        "root_rotation_options": list(config.root_rotation_options),
        "function_semantic_labels": list(config.function_semantic_labels),
    }


def structural_dag_fingerprint(
    assets: Sequence[AssetSpec],
    config: BASSConfig,
    *,
    initial_root_box: Any = None,
    initial_forbidden_boxes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return strict semantic fingerprints for one static search space."""

    root_asset = next(
        (asset for asset in assets if asset.asset_id == config.root_asset_id),
        None,
    )
    if root_asset is None:
        raise ValueError(
            "root asset id {!r} is absent from structural-dag assets".format(
                config.root_asset_id
            )
        )
    asset_payload = [_asset_payload(asset) for asset in assets]
    return {
        "partial_state_signature_version": PARTIAL_SIGNATURE_VERSION,
        "asset_catalog_sha256": _sha256_json(asset_payload),
        "grammar_config_sha256": _sha256_json(_grammar_payload(config)),
        "root_geometry_sha256": _sha256_json(
            {
                "root_asset": _asset_payload(root_asset),
                "root_box": _box_payload(initial_root_box),
            }
        ),
        "forbidden_geometry_sha256": _sha256_json(
            [_box_payload(box) for box in (initial_forbidden_boxes or ())]
        ),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _action_key(action: Action) -> Tuple[Any, ...]:
    if isinstance(action, SelectRootRotation):
        return ("SelectRootRotation", action.mode)
    if isinstance(action, AddLink):
        return (
            "AddLink",
            action.asset_id,
            int(action.p),
            int(action.d),
            int(action.f),
            int(action.q),
            int(action.child_dock_id),
            bool(action.start_function_group),
        )
    if isinstance(action, End):
        return ("End",)
    raise ValueError("structural DAGs only accept grammar actions, got {!r}".format(action))


class ActionInterner:
    """Intern exact grammar actions and serialize them as fixed records."""

    def __init__(self, actions: Sequence[Action] | None = None) -> None:
        self.actions: List[Action] = []
        self._ids: Dict[Tuple[Any, ...], int] = {}
        for action in actions or ():
            self.intern(action)

    def intern(self, action: Action) -> int:
        key = _action_key(action)
        found = self._ids.get(key)
        if found is not None:
            return found
        action_id = len(self.actions)
        if action_id >= int(UINT32_MAX):
            raise OverflowError("static action table exceeds uint32 capacity")
        # Strip prepared geometry before checkpointing or serialization.
        stored = action_from_dict(action.to_dict())
        self.actions.append(stored)
        self._ids[key] = action_id
        return action_id

    def to_array(self, asset_ids: Sequence[str]) -> np.ndarray:
        asset_index = {asset_id: index for index, asset_id in enumerate(asset_ids)}
        result = np.zeros(len(self.actions), dtype=ACTION_DTYPE)
        result["asset_index"] = UINT32_MAX
        for index, action in enumerate(self.actions):
            if isinstance(action, SelectRootRotation):
                result[index]["kind"] = 0
                result[index]["root_rotation_mode"] = {
                    "roll": 0,
                    "pitch": 1,
                    "yaw": 2,
                }[action.mode]
            elif isinstance(action, AddLink):
                if action.asset_id not in asset_index:
                    raise ValueError("action references unknown asset {!r}".format(action.asset_id))
                result[index]["kind"] = 1
                result[index]["asset_index"] = asset_index[action.asset_id]
                result[index]["p"] = action.p
                result[index]["d"] = action.d
                result[index]["f"] = action.f
                result[index]["q"] = action.q
                result[index]["child_dock_id"] = action.child_dock_id
                result[index]["flags"] = int(action.start_function_group)
            elif isinstance(action, End):
                result[index]["kind"] = 2
            else:  # pragma: no cover - guarded by intern
                raise ValueError("unsupported static action")
        return result


def decode_action(record: np.void, asset_ids: Sequence[str]) -> Action:
    kind = int(record["kind"])
    if kind == 0:
        mode = {0: "roll", 1: "pitch", 2: "yaw"}.get(
            int(record["root_rotation_mode"])
        )
        if mode is None:
            raise ValueError("invalid static root rotation mode")
        return SelectRootRotation(mode)
    if kind == 1:
        asset_index = int(record["asset_index"])
        if not (0 <= asset_index < len(asset_ids)):
            raise ValueError("static action asset index is out of range")
        return AddLink(
            asset_id=asset_ids[asset_index],
            p=int(record["p"]),
            d=int(record["d"]),
            f=int(record["f"]),
            q=int(record["q"]),
            child_dock_id=int(record["child_dock_id"]),
            start_function_group=bool(int(record["flags"]) & 1),
        )
    if kind == 2:
        return End()
    raise ValueError("unknown static action kind {}".format(kind))


@dataclass(frozen=True)
class StructuralDAGBuildResult:
    output_dir: str
    node_count: int
    edge_count: int
    terminal_alias_count: int
    physical_count: int
    action_count: int
    generated_node_count: int
    generated_transition_count: int
    merged_transition_count: int
    elapsed_seconds: float


def _initial_builder_state(
    assets: Sequence[AssetSpec],
    config: BASSConfig,
    initial_forbidden_boxes: Sequence[Any] | None,
    initial_root_box: Any,
) -> SearchState:
    root_asset = next(
        asset for asset in assets if asset.asset_id == config.root_asset_id
    )
    forbidden = list(initial_forbidden_boxes or ())
    root_box = initial_root_box
    if initial_root_box is not None and config.root_blocked_face is not None:
        blocked_face = int(config.root_blocked_face)
        face_axis = {0: 2, 1: 0, 2: 1, 3: 0, 4: 2, 5: 1}
        face_sign = {0: 1.0, 1: 1.0, 2: 1.0, 3: -1.0, 4: -1.0, 5: -1.0}
        opposite = {0: 4, 1: 3, 2: 5, 3: 1, 4: 0, 5: 2}
        parent_face = opposite[blocked_face]
        parent_axis = face_axis[parent_face]
        parent_sign = face_sign[parent_face]
        child_axis = face_axis[blocked_face]
        tip_min, tip_max = initial_root_box
        center = [
            0.5 * (float(tip_min[index]) + float(tip_max[index]))
            for index in range(3)
        ]
        face_coord = (
            float(tip_max[parent_axis])
            if parent_sign > 0.0
            else float(tip_min[parent_axis])
        )
        center[parent_axis] = (
            face_coord + parent_sign * root_asset.half_extents[child_axis]
        )
        half = root_asset.half_extents
        root_box = (
            tuple(center[index] - half[index] for index in range(3)),
            tuple(center[index] + half[index] for index in range(3)),
        )
        forbidden.append(initial_root_box)
    return initial_state(
        forbidden_boxes=forbidden,
        root_box=root_box,
        root_asset=root_asset,
        root_blocked_face=config.root_blocked_face,
        root_rotation_options=config.root_rotation_options,
        function_semantic_labels=config.function_semantic_labels,
    )


def _configured_valid_actions(
    state: SearchState,
    assets: Sequence[AssetSpec],
    config: BASSConfig,
) -> List[Action]:
    return valid_actions(
        state,
        list(assets),
        max_depth=config.max_total_links,
        max_actions=(
            None
            if int(config.max_actions_per_expansion) <= 0
            else int(config.max_actions_per_expansion)
        ),
        target_function_count=config.target_function_count,
        function_count_margin=config.function_count_margin,
        function_group_depth_delta=config.function_group_depth_delta,
        rng=None,
    )


def _valid_completed_state(state: SearchState, config: BASSConfig) -> bool:
    return bool(
        state.is_complete
        and within_function_count_margin(
            state,
            config.target_function_count,
            config.function_count_margin,
        )
        and function_group_constraints_satisfied(
            state,
            config.function_group_depth_delta,
        )
    )


def _save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def build_structural_dag(
    output_dir: str | Path,
    assets: Sequence[AssetSpec],
    config: BASSConfig,
    *,
    initial_forbidden_boxes: Sequence[Any] | None = None,
    initial_root_box: Any = None,
    resume: bool = False,
    progress_interval: int = 100_000,
    scratch_dir: str | Path | None = None,
    builder_workers: int = 1,
    builder_job_parents: int = 64,
    builder_inflight_multiplier: int = 4,
    builder_reducer_workers: int = 16,
    builder_shards: int = 256,
    virtual_root_rotations: bool = False,
) -> StructuralDAGBuildResult:
    """Compile root rotations into the DAG unless virtual storage is requested."""
    from .builder import build_layered_dag

    modes = tuple(config.root_rotation_options)
    result = build_layered_dag(
        output_dir, assets,
        replace(config, root_rotation_options=()) if virtual_root_rotations else config,
        initial_forbidden_boxes=initial_forbidden_boxes,
        initial_root_box=initial_root_box, resume=resume,
        progress_interval=progress_interval, scratch_dir=scratch_dir,
        builder_workers=builder_workers, builder_job_parents=builder_job_parents,
        builder_inflight_multiplier=builder_inflight_multiplier,
        builder_reducer_workers=builder_reducer_workers, builder_shards=builder_shards,
    )
    metadata_path = Path(result.output_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["root_rotation_representation"] = (
        "virtual" if virtual_root_rotations and modes
        else "embedded" if modes else "none"
    )
    metadata["virtual_root_rotation_options"] = (
        list(modes) if virtual_root_rotations else []
    )
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, metadata_path)
    return result


@dataclass(frozen=True)
class StructuralDAG:
    root: Path
    metadata: Mapping[str, Any]
    first_edge: np.ndarray
    edge_count: np.ndarray
    physical_id: np.ndarray
    rep_parent: np.ndarray
    rep_action_id: np.ndarray
    edge_child: np.ndarray
    edge_action_id: np.ndarray
    parent_offsets: np.ndarray
    parent_nodes: np.ndarray
    actions: np.ndarray
    asset_ids: Tuple[str, ...]
    physical_rep_node: np.ndarray
    physical_signature_sha256: np.ndarray

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = True) -> "StructuralDAG":
        root = Path(path).resolve()
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != FORMAT_VERSION:
            raise ValueError("unsupported structural-dag format")
        if int(metadata.get("action_schema_version", -1)) != ACTION_SCHEMA_VERSION:
            raise ValueError("unsupported structural-dag action schema")
        if metadata.get("physical_signature_version") != PHYSICAL_SIGNATURE_VERSION:
            raise ValueError("unsupported structural-dag physical signature version")
        if metadata.get("partial_state_signature_version") != PARTIAL_SIGNATURE_VERSION:
            raise ValueError("unsupported static-DAG partial-state signature version")
        binding = metadata.get("function_binding")
        if binding not in (None, POSITIONAL_FUNCTION_BINDING):
            raise ValueError("unsupported structural-dag function binding")
        if binding == POSITIONAL_FUNCTION_BINDING and "function_semantic_labels" in metadata:
            raise ValueError("positional structural DAG must not store task function labels")
        mode = "r" if mmap else None
        arrays = {
            key: np.load(root / filename, mmap_mode=mode, allow_pickle=False)
            for key, filename in NODE_FILES.items()
        }
        return cls(
            root=root,
            metadata=metadata,
            first_edge=arrays["first_edge"],
            edge_count=arrays["edge_count"],
            physical_id=arrays["physical_id"],
            rep_parent=arrays["rep_parent"],
            rep_action_id=arrays["rep_action_id"],
            edge_child=np.load(
                root / "edge_child.npy", mmap_mode=mode, allow_pickle=False
            ),
            edge_action_id=np.load(
                root / "edge_action_id.npy", mmap_mode=mode, allow_pickle=False
            ),
            parent_offsets=np.load(
                root / "node_parent_offsets.npy", mmap_mode=mode, allow_pickle=False
            ),
            parent_nodes=np.load(
                root / "node_parent_nodes.npy", mmap_mode=mode, allow_pickle=False
            ),
            actions=np.load(root / "actions.npy", mmap_mode=mode, allow_pickle=False),
            asset_ids=tuple(json.loads((root / "asset_ids.json").read_text())),
            physical_rep_node=np.load(
                root / "physical_rep_node.npy", mmap_mode=mode, allow_pickle=False
            ),
            physical_signature_sha256=np.load(
                root / "physical_signature_sha256.npy", mmap_mode=mode, allow_pickle=False
            ),
        )

    @property
    def node_count(self) -> int:
        return len(self.physical_id)

    @property
    def edge_count_total(self) -> int:
        return len(self.edge_child)

    @property
    def physical_count(self) -> int:
        return len(self.physical_rep_node)

    @property
    def terminal_alias_count(self) -> int:
        # Kept as a compatibility name. Physical DAG terminals are unique.
        return self.physical_count

    def child_edges(self, node: int) -> range:
        begin = int(self.first_edge[node])
        return range(begin, begin + int(self.edge_count[node]))

    def children(self, node: int) -> Iterable[int]:
        return (int(self.edge_child[edge]) for edge in self.child_edges(node))

    def parents(self, node: int) -> np.ndarray:
        begin = int(self.parent_offsets[node])
        end = int(self.parent_offsets[node + 1])
        return self.parent_nodes[begin:end]

    def action(self, action_id: int) -> Action:
        return decode_action(self.actions[action_id], self.asset_ids)

    def sequence_for_node(self, node: int) -> List[Action]:
        action_ids: List[int] = []
        current = int(node)
        while int(self.rep_parent[current]) != int(UINT64_MAX):
            action_ids.append(int(self.rep_action_id[current]))
            current = int(self.rep_parent[current])
        action_ids.reverse()
        return [self.action(action_id) for action_id in action_ids]

    def representative_node_path(self, node: int) -> List[int]:
        """Return the deterministic representative root-to-node DAG path."""

        nodes = [int(node)]
        while int(self.rep_parent[nodes[-1]]) != int(UINT64_MAX):
            nodes.append(int(self.rep_parent[nodes[-1]]))
        nodes.reverse()
        return nodes

    def representative_sequence(self, physical_id: int) -> List[Action]:
        return self.sequence_for_node(int(self.physical_rep_node[physical_id]))

    def rotation_spec(self, fallback_modes: Sequence[str] = ()) -> tuple[str, tuple[str, ...]]:
        """Infer rotation storage from root actions, then read virtual choices."""
        root_actions = [self.action(int(self.edge_action_id[e])) for e in self.child_edges(0)]
        embedded = tuple(a.mode for a in root_actions if isinstance(a, SelectRootRotation))
        declared = tuple(self.metadata.get("root_rotation_options", ()))
        representation = self.metadata.get("root_rotation_representation")
        if embedded:
            if len(embedded) != len(root_actions) or len(set(embedded)) != len(embedded):
                raise ValueError("DAG root mixes rotation and geometry actions or repeats rotations")
            if representation in {"virtual", "none"}:
                raise ValueError("DAG rotation metadata contradicts embedded root actions")
            if declared and not set(embedded).issubset(declared):
                raise ValueError("DAG embedded root rotations disagree with metadata")
            modes = tuple(mode for mode in ("roll", "pitch", "yaw") if mode in embedded)
            representation = "embedded"
        else:
            if declared or representation == "embedded":
                raise ValueError("DAG declares embedded rotations but has no rotation root actions")
            modes = tuple(self.metadata.get("virtual_root_rotation_options", fallback_modes))
            if representation == "none" and modes:
                raise ValueError("DAG declares no rotation but supplies virtual rotation choices")
            representation = "virtual" if modes else "none"
        if len(set(modes)) != len(modes) or any(m not in ("roll", "pitch", "yaw") for m in modes):
            raise ValueError("DAG rotation choices must be unique roll/pitch/yaw modes")
        return representation, modes

    def topology_config(self, config: BASSConfig) -> BASSConfig:
        """Use stored construction choices for fingerprinting and graph replay."""
        representation, modes = self.rotation_spec(config.root_rotation_options)
        topology_modes = tuple(self.metadata.get("root_rotation_options", modes)) if representation == "embedded" else ()
        return replace(config, root_rotation_options=topology_modes)

    def assert_compatible(
        self,
        assets: Sequence[AssetSpec],
        config: BASSConfig,
        *,
        initial_root_box: Any = None,
        initial_forbidden_boxes: Sequence[Any] | None = None,
    ) -> None:
        config = self.topology_config(config)
        expected = structural_dag_fingerprint(
            assets,
            config,
            initial_root_box=initial_root_box,
            initial_forbidden_boxes=initial_forbidden_boxes,
        )
        positional = self.metadata.get("function_binding") == POSITIONAL_FUNCTION_BINDING
        if positional:
            grammar = _grammar_payload(config)
            grammar.pop("function_semantic_labels")
            expected["grammar_config_sha256"] = _sha256_json(grammar)
        mismatches = {
            key: (self.metadata.get(key), value)
            for key, value in expected.items()
            if self.metadata.get(key) != value
        }
        if not positional and set(mismatches) == {"grammar_config_sha256"}:
            artifact_labels = tuple(
                str(value)
                for value in self.metadata.get("function_semantic_labels", ())
            )
            runtime_labels = tuple(config.function_semantic_labels)
            artifact_rotations = tuple(
                str(value)
                for value in self.metadata.get("root_rotation_options", ())
            )
            runtime_rotations = tuple(config.root_rotation_options)
            if len(artifact_labels) == len(runtime_labels):
                rebound_grammar = _grammar_payload(config)
                rebound_grammar["function_semantic_labels"] = list(
                    artifact_labels
                )
                virtual_rotation_rebind = (
                    not artifact_rotations and bool(runtime_rotations)
                )
                if virtual_rotation_rebind:
                    rebound_grammar["root_rotation_options"] = []
                if (
                    _sha256_json(rebound_grammar)
                    == self.metadata.get("grammar_config_sha256")
                ):
                    mismatches.pop("grammar_config_sha256")
                    warnings.warn(
                        "static DAG runtime-only metadata rebound positionally: "
                        "function labels {} -> {}, virtual root rotations "
                        "{} -> {}; geometry topology is unchanged".format(
                            artifact_labels,
                            runtime_labels,
                            artifact_rotations,
                            runtime_rotations,
                        ),
                        RuntimeWarning,
                    )
        if mismatches:
            raise ValueError(
                "structural-dag artifact is incompatible with runtime configuration: {}".format(
                    mismatches
                )
            )
        if float(self.metadata.get("physical_signature_eps", -1.0)) != float(
            config.physical_signature_eps
        ):
            raise ValueError("structural-dag physical signature epsilon mismatch")

    @staticmethod
    def _runtime_sample_indices(length: int, samples: int) -> np.ndarray:
        if length <= 0:
            return np.empty(0, dtype=np.int64)
        count = min(int(length), max(1, int(samples)))
        return np.unique(
            np.linspace(0, int(length) - 1, num=count, dtype=np.int64)
        )

    def validate_runtime(self, *, samples: int = 10_000) -> None:
        """Perform bounded startup checks on an offline-accepted artifact.

        ``validate()`` remains the exhaustive acceptance check. Reconstructing
        Python sets for every edge of a production DAG on every search launch
        is unnecessary and can consume many gigabytes before BASS starts.
        """

        count = self.node_count
        edge_total = self.edge_count_total
        if count < 1:
            raise ValueError("static DAG must contain a root")
        if any(
            len(array) != count
            for array in (
                self.first_edge,
                self.edge_count,
                self.physical_id,
                self.rep_parent,
                self.rep_action_id,
            )
        ):
            raise ValueError("static node arrays have inconsistent lengths")
        expected_dtypes = (
            (self.first_edge, np.dtype("uint64")),
            (self.edge_count, np.dtype("uint32")),
            (self.physical_id, np.dtype("uint64")),
            (self.rep_parent, np.dtype("uint64")),
            (self.rep_action_id, np.dtype("uint32")),
            (self.edge_child, np.dtype("uint64")),
            (self.edge_action_id, np.dtype("uint32")),
            (self.parent_offsets, np.dtype("uint64")),
            (self.parent_nodes, np.dtype("uint64")),
        )
        if any(array.dtype != dtype for array, dtype in expected_dtypes):
            raise ValueError("static node array dtype mismatch")
        if self.actions.dtype != ACTION_DTYPE:
            raise ValueError("static action table dtype mismatch")
        if len(self.edge_child) != len(self.edge_action_id):
            raise ValueError("static DAG edge arrays have inconsistent lengths")
        if len(self.parent_offsets) != count + 1:
            raise ValueError("static DAG reverse CSR offset length is invalid")
        if len(self.parent_nodes) != edge_total:
            raise ValueError("static DAG reverse edge count mismatch")
        if int(self.rep_parent[0]) != int(UINT64_MAX):
            raise ValueError("root representative-parent sentinel is invalid")
        if int(self.rep_action_id[0]) != int(UINT32_MAX):
            raise ValueError("root action sentinel is invalid")
        if int(self.parent_offsets[0]) != 0 or int(self.parent_offsets[-1]) != edge_total:
            raise ValueError("static reverse CSR bounds are invalid")
        if self.physical_signature_sha256.shape != (self.physical_count, 32):
            raise ValueError("physical signature digest count mismatch")

        metadata_counts = {
            "node_count": count,
            "edge_count": edge_total,
            "terminal_alias_count": self.terminal_alias_count,
            "physical_count": self.physical_count,
            "action_count": len(self.actions),
        }
        for key, expected in metadata_counts.items():
            if key in self.metadata and int(self.metadata[key]) != expected:
                raise ValueError("static metadata {} mismatch".format(key))

        node_indices = self._runtime_sample_indices(count, samples)
        starts = np.asarray(self.first_edge[node_indices], dtype=np.uint64)
        child_counts = np.asarray(self.edge_count[node_indices], dtype=np.uint64)
        if np.any(starts > edge_total) or np.any(starts + child_counts > edge_total):
            raise ValueError("sampled static edge interval is out of range")
        terminal = np.asarray(self.physical_id[node_indices]) != UINT64_MAX
        if np.any(terminal & (child_counts != 0)):
            raise ValueError("sampled terminal static node has children")
        if np.any((~terminal) & (child_counts == 0)):
            raise ValueError("sampled nonterminal static leaf is unproductive")
        non_root = node_indices[node_indices > 0]
        if len(non_root):
            parents = np.asarray(self.rep_parent[non_root], dtype=np.uint64)
            actions = np.asarray(self.rep_action_id[non_root], dtype=np.uint32)
            if np.any(parents >= non_root.astype(np.uint64)):
                raise ValueError("sampled representative parent is not topological")
            if np.any(actions >= len(self.actions)):
                raise ValueError("sampled representative action is out of range")

        edge_indices = self._runtime_sample_indices(edge_total, samples)
        if len(edge_indices):
            children = np.asarray(self.edge_child[edge_indices], dtype=np.uint64)
            actions = np.asarray(self.edge_action_id[edge_indices], dtype=np.uint32)
            if np.any(children >= count):
                raise ValueError("sampled static edge child is out of range")
            if np.any(actions >= len(self.actions)):
                raise ValueError("sampled static edge action is out of range")

        offset_indices = self._runtime_sample_indices(count, samples)
        lower = np.asarray(self.parent_offsets[offset_indices], dtype=np.uint64)
        upper = np.asarray(self.parent_offsets[offset_indices + 1], dtype=np.uint64)
        if np.any(upper < lower) or np.any(upper > edge_total):
            raise ValueError("sampled reverse CSR interval is invalid")
        parent_indices = self._runtime_sample_indices(len(self.parent_nodes), samples)
        if len(parent_indices) and np.any(self.parent_nodes[parent_indices] >= count):
            raise ValueError("sampled reverse parent is out of range")

        physical_indices = self._runtime_sample_indices(self.physical_count, samples)
        if len(physical_indices):
            representative_nodes = np.asarray(
                self.physical_rep_node[physical_indices], dtype=np.uint64
            )
            if np.any(representative_nodes >= count):
                raise ValueError("sampled physical representative is out of range")
            represented_ids = np.asarray(
                self.physical_id[representative_nodes], dtype=np.uint64
            )
            if np.any(represented_ids != physical_indices.astype(np.uint64)):
                raise ValueError("sampled physical representative ID mismatch")

    def validate(self) -> None:
        count = self.node_count
        if count < 1:
            raise ValueError("static DAG must contain a root")
        if any(len(array) != count for array in (
            self.first_edge,
            self.edge_count,
            self.physical_id,
            self.rep_parent,
            self.rep_action_id,
        )):
            raise ValueError("static node arrays have inconsistent lengths")
        expected_dtypes = (
            (self.first_edge, np.dtype("uint64")),
            (self.edge_count, np.dtype("uint32")),
            (self.physical_id, np.dtype("uint64")),
            (self.rep_parent, np.dtype("uint64")),
            (self.rep_action_id, np.dtype("uint32")),
            (self.edge_child, np.dtype("uint64")),
            (self.edge_action_id, np.dtype("uint32")),
            (self.parent_offsets, np.dtype("uint64")),
            (self.parent_nodes, np.dtype("uint64")),
        )
        if any(array.dtype != dtype for array, dtype in expected_dtypes):
            raise ValueError("static node array dtype mismatch")
        if len(self.edge_child) != len(self.edge_action_id):
            raise ValueError("static DAG edge arrays have inconsistent lengths")
        if len(self.parent_offsets) != count + 1:
            raise ValueError("static DAG reverse CSR offset length is invalid")
        if len(self.parent_nodes) != len(self.edge_child):
            raise ValueError("static DAG reverse edge count mismatch")
        if self.actions.dtype != ACTION_DTYPE:
            raise ValueError("static action table dtype mismatch")
        metadata_counts = {
            "node_count": count,
            "edge_count": self.edge_count_total,
            "terminal_alias_count": self.terminal_alias_count,
            "physical_count": self.physical_count,
            "action_count": len(self.actions),
        }
        for key, expected in metadata_counts.items():
            if key in self.metadata and int(self.metadata[key]) != expected:
                raise ValueError("static metadata {} mismatch".format(key))
        if int(self.rep_parent[0]) != int(UINT64_MAX):
            raise ValueError("root representative-parent sentinel is invalid")
        if int(self.rep_action_id[0]) != int(UINT32_MAX):
            raise ValueError("root action sentinel is invalid")
        if count > 1:
            if np.any(self.rep_parent[1:] >= count):
                raise ValueError("representative parent index is out of range")
            if np.any(self.rep_action_id[1:] >= len(self.actions)):
                raise ValueError("representative action ID is out of range")
        if np.any(self.first_edge > self.edge_count_total):
            raise ValueError("static first-edge index is out of range")
        ends = self.first_edge.astype(np.uint64) + self.edge_count.astype(np.uint64)
        if np.any(ends > self.edge_count_total):
            raise ValueError("static edge interval is out of range")
        if int(np.sum(self.edge_count, dtype=np.uint64)) != self.edge_count_total:
            raise ValueError("static forward CSR does not cover every edge")
        if np.any(self.edge_child >= count):
            raise ValueError("static edge child is out of range")
        if np.any(self.edge_action_id >= len(self.actions)):
            raise ValueError("static edge action is out of range")
        if np.any(self.parent_offsets[1:] < self.parent_offsets[:-1]):
            raise ValueError("static reverse CSR offsets are not monotonic")
        if int(self.parent_offsets[0]) != 0 or int(self.parent_offsets[-1]) != self.edge_count_total:
            raise ValueError("static reverse CSR bounds are invalid")
        if np.any(self.parent_nodes >= count):
            raise ValueError("static reverse parent is out of range")

        reverse_pairs = set()
        for child in range(count):
            parents = list(map(int, self.parents(child)))
            if parents != sorted(set(parents)):
                raise ValueError("static reverse parent list is not unique and sorted")
            reverse_pairs.update((parent, child) for parent in parents)
        forward_pairs = set()
        for parent in range(count):
            children = list(self.children(parent))
            if len(children) != len(set(children)):
                raise ValueError("static DAG parent has duplicate physical children")
            for child in children:
                if child <= parent:
                    raise ValueError("static DAG is not topologically ordered")
                forward_pairs.add((parent, child))
        if forward_pairs != reverse_pairs:
            raise ValueError("static forward and reverse CSR disagree")
        for node in range(1, count):
            parent = int(self.rep_parent[node])
            action = int(self.rep_action_id[node])
            if not any(
                int(self.edge_child[edge]) == node
                and int(self.edge_action_id[edge]) == action
                for edge in self.child_edges(parent)
            ):
                raise ValueError("representative edge is absent from the DAG")
        terminal_mask = self.physical_id != UINT64_MAX
        if np.any(terminal_mask & (self.edge_count != 0)):
            raise ValueError("terminal static node has children")
        if np.any((~terminal_mask) & (self.edge_count == 0)):
            raise ValueError("nonterminal static leaf survived productive pruning")
        if np.any(self.physical_id[terminal_mask] >= self.physical_count):
            raise ValueError("static terminal physical ID is out of range")
        terminals = np.flatnonzero(terminal_mask).astype(np.uint64)
        if len(terminals) != self.physical_count:
            raise ValueError("physical DAG must contain one terminal per physical class")
        if len(np.unique(self.physical_id[terminal_mask])) != self.physical_count:
            raise ValueError("physical terminal IDs are not unique")
        for physical_id, node in enumerate(self.physical_rep_node):
            if int(self.physical_id[int(node)]) != physical_id:
                raise ValueError("physical representative points to the wrong terminal")
        if self.physical_signature_sha256.shape != (self.physical_count, 32):
            raise ValueError("physical signature digest count mismatch")

    def replay_validate(
        self,
        assets: Sequence[AssetSpec],
        config: BASSConfig,
        *,
        samples: int = 10_000,
        seed: int = 0,
        initial_root_box: Any = None,
        initial_forbidden_boxes: Sequence[Any] | None = None,
    ) -> None:
        self.assert_compatible(
            assets,
            config,
            initial_root_box=initial_root_box,
            initial_forbidden_boxes=initial_forbidden_boxes,
        )
        config = self.topology_config(config)
        if self.metadata.get("function_binding") == POSITIONAL_FUNCTION_BINDING:
            # The published artifact hashes function ordinals (function:0, ...)
            # instead of the task's display labels. Geometry and action paths
            # remain identical across tasks with the same grammar.
            config = replace(config, function_semantic_labels=())
        projection = self.metadata.get("virtual_root_projection")
        if projection is not None:
            config = replace(config, root_rotation_options=(str(projection["source_root_rotation"]),))
        terminals = np.flatnonzero(self.physical_id != UINT64_MAX)
        if not len(terminals):
            raise ValueError("structural DAG has no terminals")
        rng = random.Random(seed)
        chosen = (
            list(map(int, terminals))
            if len(terminals) <= samples
            else rng.sample(list(map(int, terminals)), samples)
        )
        for node in chosen:
            state = _initial_builder_state(
                assets,
                config,
                initial_forbidden_boxes,
                initial_root_box,
            )
            sequence = self.sequence_for_node(node)
            projection = self.metadata.get("virtual_root_projection")
            if projection is not None:
                sequence = [
                    SelectRootRotation(str(projection["source_root_rotation"]))
                ] + sequence
            for action in sequence:
                state = apply_action(state, action, list(assets))
            if not _valid_completed_state(state, config):
                raise ValueError("reconstructed static terminal is invalid")
            signature = physical_function_signature(
                state,
                assets,
                eps=float(config.physical_signature_eps),
            )
            digest = bytes.fromhex(physical_signature_digest(signature))
            physical_id = int(self.physical_id[node])
            if digest != bytes(
                np.asarray(self.physical_signature_sha256[physical_id], dtype=np.uint8)
            ):
                raise ValueError("reconstructed physical signature mismatch")




UNSEEN = 0
INFLIGHT = 1
DONE = 2


@dataclass
class StaticRuntimeOutcome:
    score: float
    reward: float
    valid: bool
    stage_evidence: StageEvidence | None
    task_success: bool | None = None


@dataclass(frozen=True)
class StaticProbe:
    """One evaluator dispatch selected from the immutable DAG."""

    physical_id: int
    dispatch_path: Tuple[int, ...]
    probe_start: int | None = None
    rollout_suffix: Tuple[int, ...] = ()
    warmup: bool = False
    probe_index: int | None = None
    root_rotation_mode: str | None = None


@dataclass
class _LookaheadEpoch:
    """Frozen two-step acquisition values between evaluator observations."""

    epoch_id: int = 0
    valid: bool = False
    heap: List[Tuple[float, int, int]] = field(default_factory=list)
    snapshot_scores: Dict[int, float] = field(default_factory=dict)
    snapshot_parallel_index: Dict[int, float] = field(default_factory=dict)
    node_versions: Dict[int, int] = field(default_factory=dict)

    def clear(self) -> None:
        self.valid = False
        self.heap.clear()
        self.snapshot_scores.clear()
        self.snapshot_parallel_index.clear()
        self.node_versions.clear()


@dataclass
class _StaticRolloutCursor:
    """Minimal grammar state needed by the frozen static-DAG rollout policy."""

    depths: List[int] = field(default_factory=lambda: [0])
    used_faces: List[set[int]] = field(default_factory=lambda: [set()])
    active_function_groups: List[bool] = field(default_factory=lambda: [False])
    function_count: int = 0

    @property
    def depth(self) -> int:
        return self.depths[-1] if self.depths else 0

    @property
    def active_function_group(self) -> bool:
        return bool(self.active_function_groups[-1]) if self.active_function_groups else False

    def is_branching(self, action: Action) -> bool:
        return isinstance(action, AddLink) and int(action.p) in self.used_faces[-1]

    def apply(self, action: Action) -> None:
        if isinstance(action, AddLink):
            self.used_faces[-1].add(int(action.p))
            active = self.active_function_group or bool(action.start_function_group)
            self.function_count += int(action.start_function_group)
            self.depths.append(self.depth + 1)
            self.used_faces.append(set())
            self.active_function_groups.append(active)
        elif isinstance(action, End) and len(self.depths) > 1:
            self.depths.pop()
            self.used_faces.pop()
            self.active_function_groups.pop()


@dataclass
class StaticSearchResult:
    best_sequence: List[Action]
    best_score: float
    best_reward: float
    total_iterations: int
    completed_candidates: int
    valid_completed_candidates: int
    best_function_count: int | None
    diagnostics: dict[str, Any]


def _coerce_runtime_outcome(value: Any) -> StaticRuntimeOutcome:
    if hasattr(value, "score") and hasattr(value, "reward"):
        score = float(value.score)
        reward = float(value.reward)
        valid = bool(getattr(value, "valid", True)) and math.isfinite(score)
        raw_task_success = getattr(value, "task_success", None)
        task_success = (
            None if raw_task_success is None else bool(raw_task_success)
        )
        stage = None
        milestone = getattr(value, "task_milestone", None)
        stage_count = getattr(value, "task_stage_count", None)
        if milestone is not None and stage_count is not None:
            stage = StageEvidence(
                milestone=int(milestone),
                stage_count=int(stage_count),
                progress=float(getattr(value, "task_progress", 0.0)),
                task_success=bool(task_success),
                feasible=bool(getattr(value, "task_feasible", True)),
                residual_reward=reward,
            )
        return StaticRuntimeOutcome(score, reward, valid, stage, task_success)
    score = float(value)
    return StaticRuntimeOutcome(
        score=score,
        reward=-score,
        valid=math.isfinite(score),
        stage_evidence=None,
        task_success=None,
    )


def _sample_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calibrate an BASS quantile from an empty sample")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _calibrate_milestones(
    outcomes: Sequence[StaticRuntimeOutcome],
    config: BASSConfig,
) -> dict[str, Any]:
    """Fit the immutable global BASS schema/prior before replaying warmup evidence."""

    accepted: List[StaticRuntimeOutcome] = []
    stage_count: int | None = None
    for outcome in outcomes:
        evidence = outcome.stage_evidence
        if not outcome.valid or outcome.task_success is None or evidence is None:
            continue
        evidence_stage_count = int(evidence.stage_count)
        if stage_count is None:
            stage_count = evidence_stage_count
        elif evidence_stage_count != stage_count:
            raise ValueError(
                "integrated BASS calibration observed inconsistent stage counts: "
                "{} and {}".format(stage_count, evidence_stage_count)
            )
        accepted.append(outcome)
    if not accepted:
        raise ValueError("integrated BASS calibration produced no valid staged outcomes")
    assert stage_count is not None

    configured_bins = tuple(
        int(value) for value in config.calibration_bins_per_stage
    )
    if configured_bins and len(configured_bins) != stage_count:
        raise ValueError(
            "integrated BASS calibration stage-count mismatch: evaluator={} config={}"
            .format(stage_count, len(configured_bins))
        )
    progress: List[List[float]] = [[] for _ in range(stage_count)]
    for outcome in accepted:
        if outcome.task_success:
            continue
        assert outcome.stage_evidence is not None
        milestone = int(outcome.stage_evidence.milestone)
        if not 0 <= milestone <= stage_count:
            raise ValueError("invalid BASS warmup milestone")
        if milestone == stage_count:
            # Terminal-filter failures have no unfinished stage to calibrate.
            # They still contribute failure evidence when encoded below.
            continue
        progress[milestone].append(
            min(1.0, max(0.0, float(outcome.stage_evidence.progress)))
        )

    threshold_samples: List[List[float]] = []
    for values in progress:
        if config.calibration_threshold_sample == "interior":
            values = [value for value in values if 0.0 < value < 1.0]
        threshold_samples.append(values)

    if configured_bins:
        bins = configured_bins
        bins_source = "configured"
    else:
        # Fresh static searches infer at most two informative micro-gates per
        # stage. Degenerate or unobserved stages correctly remain macro-only.
        inferred: List[int] = []
        for values in threshold_samples:
            unique_count = len(set(values))
            inferred.append(min(2, max(0, unique_count - 1)))
        bins = tuple(inferred)
        bins_source = "auto"

    thresholds: List[List[float]] = []
    threshold_counts: List[int] = []
    for stage, bin_count in enumerate(bins):
        values = threshold_samples[stage]
        threshold_counts.append(len(values))
        if bin_count and not values:
            raise ValueError(
                "integrated BASS calibration stage {} requests {} bins but has "
                "no informative progress outcomes".format(stage, bin_count)
            )
        row = [
            _sample_quantile(values, index / float(bin_count + 1))
            for index in range(1, bin_count + 1)
        ]
        if any(not 0.0 < value < 1.0 for value in row) or any(
            current <= previous for previous, current in zip(row, row[1:])
        ):
            raise ValueError(
                "integrated BASS calibration produced degenerate thresholds at "
                "stage {}: {}".format(stage, row)
            )
        thresholds.append(row)

    schema = MilestoneSchema(thresholds)
    category_counts = [0] * (schema.level_count + 1)
    for outcome in accepted:
        category = schema.encode(
            outcome.stage_evidence,
            bool(outcome.task_success),
        )
        category_counts[category] += 1
    smoothing = float(config.calibration_smoothing)
    continuation_means = []
    at_risk_counts = []
    survival_counts = []
    for level in range(schema.level_count):
        at_risk = sum(category_counts[level:])
        survived = sum(category_counts[level + 1 :])
        at_risk_counts.append(at_risk)
        survival_counts.append(survived)
        continuation_means.append(
            (survived + smoothing) / (at_risk + 2.0 * smoothing)
            if at_risk
            else 0.5
        )
    return {
        "mode": "integrated_warmup",
        "warmup_total_outcomes": len(outcomes),
        "source_accepted_ordinal_rows": len(accepted),
        "stage_count": stage_count,
        "bins_per_stage": list(bins),
        "bins_source": bins_source,
        "progress_thresholds": thresholds,
        "stage_sample_counts": [len(row) for row in progress],
        "threshold_sample_counts": threshold_counts,
        "threshold_sampling": config.calibration_threshold_sample,
        "ordinal_level_count": schema.level_count,
        "category_counts": category_counts,
        "hazard_at_risk_counts": at_risk_counts,
        "hazard_survival_counts": survival_counts,
        "continuation_prior_means": continuation_means,
        "prior_strengths": list(config.prior_strengths),
        "prior_success_mean": math.prod(continuation_means),
        "smoothing": smoothing,
        "calibration_reused_as_posterior_evidence": False,
    }


class StructuralDAGRuntime:
    """Per-run mutable state over one immutable physical-state DAG."""

    def __init__(self, tree: StructuralDAG, config: BASSConfig) -> None:
        self.tree = tree
        self.config = config
        count = tree.node_count
        self.physical_status = np.zeros(tree.physical_count, dtype=np.uint8)
        self.physical_reward = np.full(tree.physical_count, np.nan, dtype=np.float64)
        self.visits = np.zeros(count, dtype=np.uint32)
        self.inflight_visits = np.zeros(count, dtype=np.uint32)
        self.remaining_children = np.asarray(tree.edge_count, dtype=np.uint32).copy()
        self.remaining_selectable_children = np.asarray(
            tree.edge_count, dtype=np.uint32
        ).copy()
        self.resolved = np.zeros(count, dtype=np.bool_)
        self.selectable = np.ones(count, dtype=np.bool_)
        self.propagation_epoch = np.zeros(count, dtype=np.uint32)
        self.current_epoch = 0
        self.runtime_stats: Dict[int, RewardStatistics] = {}
        self.rng = random.Random(int(config.seed))
        self.started_at = time.monotonic()

        # The static files contain only immutable topology and physical IDs.
        # BASS search state is a sparse, per-run overlay allocated after load.
        self.bass: MilestonePosteriorRegistry | None = None
        self.materialized_probe_nodes: set[int] = set()
        self._bass_frontier_heap: List[Tuple[float, int, int]] = []
        self._bass_frontier_versions: Dict[int, int] = {}
        self.bass_selection_count = 0
        self.bass_proposal_count = 0
        self.bass_repeat_proposals = 0
        self.bass_accepted_probes = 0
        self.bass_evidence_updates = 0
        self.bass_duplicate_evidence_skips = 0
        self.bass_first_success_probe: int | None = None
        self.bass_first_success_wall_time: float | None = None
        self.warmup_bass_observations = 0
        self.warmup_bass_successes = 0
        self.bass_category_counts: List[int] = []
        self.bass_two_step_selection_calls = 0
        self.bass_two_step_selection_seconds = 0.0
        self.bass_two_step_scored_frontiers = 0
        self.bass_two_step_max_frontiers = 0
        self.bass_acquisition_epoch = _LookaheadEpoch()
        self.bass_calibration: dict[str, Any] | None = None
        self._replayed_calibration_terminals: set[int] = set()
        self.static_acquisition_epoch_count = 0
        self.static_acquisition_epoch_seconds = 0.0
        self.static_acquisition_frontiers_scored = 0
        self.static_candidates_from_epoch = 0
        self.static_rollout_dead_edges_filtered = 0
        self.static_rollout_exhausted_nodes = 0
        self.static_rollout_repeat_terminals = 0
        if config.acquisition in {
            "bass_n1",
            "bass_n2",
        }:
            if config.calibration_artifact:
                self._initialize_bass(
                    progress_thresholds=config.milestone_thresholds,
                    continuation_means=config.continuation_prior_means,
                )

    @property
    def bass_enabled(self) -> bool:
        return (
            self.bass is not None
            or self.calibration_pending
        )

    @property
    def calibration_pending(self) -> bool:
        return (
            self.config.acquisition
            in {"bass_n1", "bass_n2"}
            and self.bass is None
        )

    @property
    def posteriors_initialized(self) -> bool:
        return self.bass is not None

    def _initialize_bass(
        self,
        *,
        progress_thresholds: Sequence[Sequence[float]],
        continuation_means: Sequence[float],
    ) -> None:
        if self.bass is not None:
            raise RuntimeError("BASS overlay was initialized more than once")
        self.bass = MilestonePosteriorRegistry(
            schema=MilestoneSchema(progress_thresholds),
            continuation_means=continuation_means,
            prior_strengths=self.config.prior_strengths,
        )
        self.bass_category_counts = [0] * (self.bass.schema.level_count + 1)
        self.materialized_probe_nodes.add(0)

    def initialize_milestone_posteriors(
        self,
        outcomes: Sequence[StaticRuntimeOutcome],
    ) -> dict[str, Any]:
        if not self.calibration_pending:
            raise RuntimeError("integrated BASS calibration is not pending")
        calibration = _calibrate_milestones(outcomes, self.config)
        self._initialize_bass(
            progress_thresholds=calibration["progress_thresholds"],
            continuation_means=calibration["continuation_prior_means"],
        )
        self.bass_calibration = calibration
        return calibration

    def _bass_posterior(self, node: int) -> MilestonePosterior:
        if self.bass is None:
            raise RuntimeError("BASS posterior requested outside BASS mode")
        return self.bass.posterior(int(node))

    def _active_posterior(self, node: int) -> MilestonePosterior:
        if self.bass is not None:
            return self._bass_posterior(node)
        raise RuntimeError("BASS posterior is not initialized")

    def _active_parallel_index(self, node: int) -> float:
        return float(self._active_posterior(node).parallel_index())

    def _refresh_bass_frontier(self, node: int) -> None:
        node = int(node)
        if (
            node not in self.materialized_probe_nodes
            or not self.selectable[node]
            or int(self.tree.physical_id[node]) != int(UINT64_MAX)
        ):
            return
        version = self._bass_frontier_versions.get(node, 0) + 1
        self._bass_frontier_versions[node] = version
        heapq.heappush(
            self._bass_frontier_heap,
            (-self._active_parallel_index(node), node, version),
        )

    def _pop_bass_frontier(self) -> int | None:
        while self._bass_frontier_heap:
            _negative_index, node, version = heapq.heappop(
                self._bass_frontier_heap
            )
            if self._bass_frontier_versions.get(node) != version:
                continue
            if (
                not self.selectable[node]
                or int(self.tree.physical_id[node]) != int(UINT64_MAX)
            ):
                continue
            return int(node)
        return None

    def _invalidate_bass_acquisition_epoch(self) -> None:
        self.bass_acquisition_epoch.clear()

    def _push_bass_epoch_frontier(self, node: int, score: float) -> None:
        epoch = self.bass_acquisition_epoch
        version = epoch.node_versions.get(int(node), 0) + 1
        epoch.node_versions[int(node)] = version
        heapq.heappush(epoch.heap, (-float(score), int(node), version))

    def _rebuild_bass_acquisition_epoch(self) -> None:
        """Score the active frontier once for one Bayesian information state."""

        if self.bass is None:
            raise RuntimeError("BASS acquisition epoch requested outside BASS mode")
        candidates = sorted(
            int(node)
            for node in self.materialized_probe_nodes
            if self.selectable[int(node)]
            and int(self.tree.physical_id[int(node)]) == int(UINT64_MAX)
        )
        started = time.perf_counter()
        scores = self.bass.two_step_parallel_scores(candidates)
        elapsed = time.perf_counter() - started
        epoch = self.bass_acquisition_epoch
        epoch.clear()
        epoch.epoch_id += 1
        epoch.valid = True
        for node, raw_score in zip(candidates, scores):
            score = float(raw_score)
            epoch.snapshot_scores[node] = score
            epoch.snapshot_parallel_index[node] = self._bass_posterior(
                node
            ).parallel_index()
            self._push_bass_epoch_frontier(node, score)
        self.bass_two_step_selection_calls += 1
        self.bass_two_step_selection_seconds += elapsed
        self.bass_two_step_scored_frontiers += len(candidates)
        self.bass_two_step_max_frontiers = max(
            self.bass_two_step_max_frontiers,
            len(candidates),
        )
        self.static_acquisition_epoch_count += 1
        self.static_acquisition_epoch_seconds += elapsed
        self.static_acquisition_frontiers_scored += len(candidates)

    def _pop_bass_epoch_frontier(self) -> int | None:
        epoch = self.bass_acquisition_epoch
        if not epoch.valid:
            self._rebuild_bass_acquisition_epoch()
        while epoch.heap:
            _negative_score, node, version = heapq.heappop(epoch.heap)
            if epoch.node_versions.get(node) != version:
                continue
            if (
                not self.selectable[node]
                or int(self.tree.physical_id[node]) != int(UINT64_MAX)
            ):
                continue
            return int(node)
        if any(
            self.selectable[int(node)]
            and int(self.tree.physical_id[int(node)]) == int(UINT64_MAX)
            and int(node) not in epoch.snapshot_scores
            for node in self.materialized_probe_nodes
        ):
            self._rebuild_bass_acquisition_epoch()
            return self._pop_bass_epoch_frontier()
        return None

    def _refresh_bass_epoch_frontier_after_reservation(self, node: int) -> None:
        """Update only the selected arm's pending-worker diversification."""

        epoch = self.bass_acquisition_epoch
        node = int(node)
        if not epoch.valid or node not in epoch.snapshot_scores:
            return
        if (
            not self.selectable[node]
            or int(self.tree.physical_id[node]) != int(UINT64_MAX)
        ):
            return
        old_index = float(epoch.snapshot_parallel_index[node])
        new_index = float(self._bass_posterior(node).parallel_index())
        score = float(epoch.snapshot_scores[node])
        if old_index > 0.0:
            score *= new_index / old_index
        self._push_bass_epoch_frontier(node, score)


    @staticmethod
    def _depth_value(
        values: Sequence[float] | None,
        depth: int,
        *,
        default: float,
    ) -> float:
        if not values:
            return float(default)
        return float(values[min(max(0, int(depth)), len(values) - 1)])

    def _cursor_for_node(self, node: int) -> _StaticRolloutCursor:
        cursor = _StaticRolloutCursor()
        for action in self.tree.sequence_for_node(node):
            cursor.apply(action)
        return cursor

    def _sample_rollout_edge(
        self,
        node: int,
        cursor: _StaticRolloutCursor,
    ) -> int | None:
        all_edges = list(self.tree.child_edges(node))
        edges = [edge for edge in all_edges if self._rollout_edge_is_live(edge)]
        self.static_rollout_dead_edges_filtered += len(all_edges) - len(edges)
        if not edges:
            return None
        actions = [
            self.tree.action(int(self.tree.edge_action_id[edge]))
            for edge in edges
        ]

        if self.config.target_function_count is not None:
            minimum = max(
                0,
                int(self.config.target_function_count)
                - int(self.config.function_count_margin),
            )
            if cursor.function_count < minimum and not cursor.active_function_group:
                eligible = [
                    index
                    for index, action in enumerate(actions)
                    if isinstance(action, AddLink) and action.start_function_group
                ]
                if eligible:
                    return edges[self.rng.choice(eligible)]

        if self.config.rollout_policy == "uniform" and not self.config.structural_factorization:
            return self.rng.choice(edges)

        if self.config.structural_factorization:
            classes = [
                [index for index, action in enumerate(actions) if isinstance(action, End)],
                [index for index, action in enumerate(actions) if isinstance(action, AddLink)],
                [
                    index
                    for index, action in enumerate(actions)
                    if not isinstance(action, (End, AddLink))
                ],
            ]
            selected_class = self.rng.choice([group for group in classes if group])
            return edges[self.rng.choice(selected_class)]

        if self.config.rollout_policy == "end_biased":
            end_indices = [
                index for index, action in enumerate(actions) if isinstance(action, End)
            ]
            other_indices = [
                index for index, action in enumerate(actions) if not isinstance(action, End)
            ]
            probability = self._depth_value(
                self.config.rollout_end_prob_by_depth,
                cursor.depth,
                default=0.5,
            )
            if end_indices and (not other_indices or self.rng.random() < probability):
                return edges[self.rng.choice(end_indices)]
            return edges[self.rng.choice(other_indices or end_indices)]

        add_indices = [
            index for index, action in enumerate(actions) if isinstance(action, AddLink)
        ]
        end_indices = [
            index for index, action in enumerate(actions) if isinstance(action, End)
        ]
        add_probability = self._depth_value(
            self.config.rollout_addlink_prob_by_depth,
            cursor.depth,
            default=0.5,
        )
        if add_indices and self.rng.random() < add_probability:
            penalty = self._depth_value(
                self.config.rollout_branching_penalty_by_depth,
                cursor.depth,
                default=1.0,
            )
            weights = [
                max(0.0, penalty) if cursor.is_branching(actions[index]) else 1.0
                for index in add_indices
            ]
            total = sum(weights)
            if total > 0.0:
                threshold = self.rng.random() * total
                cumulative = 0.0
                for index, weight in zip(add_indices, weights):
                    cumulative += weight
                    if threshold <= cumulative:
                        return edges[index]
            return edges[self.rng.choice(add_indices)]
        if end_indices:
            return edges[self.rng.choice(end_indices)]
        return edges[self.rng.choice(add_indices or list(range(len(edges))))]

    def _rollout_edge_is_live(self, edge: int) -> bool:
        child = int(self.tree.edge_child[int(edge)])
        if not self.selectable[child]:
            return False
        physical_id = int(self.tree.physical_id[child])
        return (
            physical_id == int(UINT64_MAX)
            or self.physical_status[physical_id] == UNSEEN
        )

    def _sample_fixed_suffix(
        self,
        probe_start: int,
    ) -> Tuple[int, Tuple[int, ...]] | None:
        node = int(probe_start)
        suffix = [node]
        cursor = self._cursor_for_node(node)
        while int(self.tree.physical_id[node]) == int(UINT64_MAX):
            edge = self._sample_rollout_edge(node, cursor)
            if edge is None:
                self.static_rollout_exhausted_nodes += 1
                self._disable_node(node)
                self._invalidate_bass_acquisition_epoch()
                return None
            action = self.tree.action(int(self.tree.edge_action_id[edge]))
            cursor.apply(action)
            node = int(self.tree.edge_child[edge])
            suffix.append(node)
        return int(self.tree.physical_id[node]), tuple(suffix)

    def select_bass_probe(self) -> StaticProbe | None:
        """Select one sparse runtime frontier and draw a fixed-policy suffix."""

        if not self.bass_enabled:
            raise RuntimeError("BASS probe requested outside BASS mode")
        if not self.selectable[0]:
            return None
        if not self._bass_frontier_heap:
            for node in self.materialized_probe_nodes:
                self._refresh_bass_frontier(node)
        deferred: List[int] = []
        while True:
            if self.config.acquisition == "bass_n2":
                probe_start = self._pop_bass_epoch_frontier()
            else:
                probe_start = self._pop_bass_frontier()
            if probe_start is None:
                if not deferred:
                    return None
                for node in deferred:
                    self._refresh_bass_frontier(node)
                deferred.clear()
                continue
            self.bass_selection_count += 1
            sampled = self._sample_fixed_suffix(probe_start)
            if sampled is None:
                continue
            physical_id, suffix = sampled
            self.bass_proposal_count += 1
            if self.physical_status[physical_id] != UNSEEN:
                self.bass_repeat_proposals += 1
                self.static_rollout_repeat_terminals += 1
                self._disable_physical_aliases(physical_id)
                self._invalidate_bass_acquisition_epoch()
                continue
            for node in deferred:
                self._refresh_bass_frontier(node)
            prefix = self.tree.representative_node_path(probe_start)
            dispatch_path = tuple(prefix + list(suffix[1:]))
            return StaticProbe(
                physical_id=physical_id,
                dispatch_path=dispatch_path,
                probe_start=probe_start,
                rollout_suffix=suffix,
                probe_index=self.bass_accepted_probes + 1,
            )

    def select_warmup_probe(self) -> StaticProbe | None:
        """Select a uniformly random unseen physical terminal for calibration."""

        physical_id = None
        for _ in range(128):
            candidate = self.rng.randrange(self.tree.physical_count)
            if self.physical_status[candidate] == UNSEEN:
                physical_id = candidate
                break
        if physical_id is None:
            unseen = np.flatnonzero(self.physical_status == UNSEEN)
            if not len(unseen):
                return None
            physical_id = int(unseen[self.rng.randrange(len(unseen))])
        node = int(self.tree.physical_rep_node[physical_id])
        return StaticProbe(
            physical_id=physical_id,
            dispatch_path=tuple(self.tree.representative_node_path(node)),
            warmup=True,
        )

    def _stats(self, node: int) -> RewardStatistics:
        statistics = self.runtime_stats.get(node)
        if statistics is None:
            statistics = RewardStatistics()
            self.runtime_stats[node] = statistics
        return statistics

    def select_terminal(self):
        probe = self.select_warmup_probe()
        return None if probe is None else (probe.physical_id, list(probe.dispatch_path))

    def select_probe(self):
        return self.select_warmup_probe()

    def _disable_node(self, node: int) -> None:
        if not self.selectable[node]:
            return
        self.selectable[node] = False
        for raw_parent in self.tree.parents(node):
            parent = int(raw_parent)
            if self.remaining_selectable_children[parent] == 0:
                raise RuntimeError("selectable child counter underflow")
            self.remaining_selectable_children[parent] -= 1
            if self.remaining_selectable_children[parent] == 0:
                self._disable_node(parent)

    def _disable_physical_aliases(self, physical_id: int) -> None:
        self._disable_node(int(self.tree.physical_rep_node[physical_id]))

    def reserve(self, physical_id: int, path: Sequence[int]) -> None:
        if self.physical_status[physical_id] != UNSEEN:
            raise RuntimeError("physical terminal was reserved more than once")
        self.physical_status[physical_id] = INFLIGHT
        self._disable_physical_aliases(physical_id)
        for node in path:
            self.visits[node] += 1
            self.inflight_visits[node] += 1

    def reserve_probe(self, probe: StaticProbe) -> None:
        self.reserve(probe.physical_id, probe.dispatch_path)
        if not probe.warmup and probe.probe_start is not None:
            self._active_posterior(probe.probe_start).reserve()
            self.bass_accepted_probes += 1
            self.static_candidates_from_epoch += int(
                self.config.acquisition == "bass_n2"
            )
            for node in probe.rollout_suffix:
                if int(self.tree.physical_id[node]) == int(UINT64_MAX):
                    self.materialized_probe_nodes.add(int(node))
                    self._refresh_bass_frontier(int(node))
            if self.config.acquisition == "bass_n2":
                self._refresh_bass_epoch_frontier_after_reservation(
                    probe.probe_start
                )

    def _resolve_node(self, node: int) -> None:
        if self.resolved[node]:
            return
        self.resolved[node] = True
        for raw_parent in self.tree.parents(node):
            parent = int(raw_parent)
            if self.remaining_children[parent] == 0:
                raise RuntimeError("remaining child counter underflow")
            self.remaining_children[parent] -= 1
            if self.remaining_children[parent] == 0:
                self._resolve_node(parent)

    def complete(
        self,
        physical_id: int,
        dispatch_path: Sequence[int],
        outcome: StaticRuntimeOutcome,
    ) -> None:
        if self.physical_status[physical_id] != INFLIGHT:
            raise RuntimeError("physical completion without an in-flight reservation")
        self.physical_status[physical_id] = DONE
        self.physical_reward[physical_id] = outcome.reward
        for node in dispatch_path:
            if self.inflight_visits[node] == 0:
                raise RuntimeError("in-flight visit counter underflow")
            self.inflight_visits[node] -= 1
        self.current_epoch += 1
        if self.current_epoch >= int(UINT32_MAX):
            self.propagation_epoch.fill(0)
            self.current_epoch = 1
        backup_reward = outcome.reward
        if outcome.valid:
            pending = [int(self.tree.physical_rep_node[physical_id])]
            while pending:
                node = pending.pop()
                if self.propagation_epoch[node] == self.current_epoch:
                    continue
                self.propagation_epoch[node] = self.current_epoch
                self._stats(node).record(
                    backup_reward,
                    terminal_key=physical_id,
                    store_sample=False,
                    stage_evidence=outcome.stage_evidence,
                )
                pending.extend(map(int, self.tree.parents(node)))
        self._resolve_node(int(self.tree.physical_rep_node[physical_id]))

    def complete_probe(
        self,
        probe: StaticProbe,
        outcome: StaticRuntimeOutcome,
    ) -> None:
        self.complete(probe.physical_id, probe.dispatch_path, outcome)
        if probe.warmup or probe.probe_start is None:
            return
        self._active_posterior(probe.probe_start).release()
        self._observe_probe_outcome(probe, outcome)

    def replay_calibration_probe(
        self, probe: StaticProbe, outcome: StaticRuntimeOutcome,
    ) -> bool:
        """Replay a completed calibration terminal as a root-start BASS probe.

        Scalar backup, physical completion and dispatch counters were already
        handled during calibration. Only posterior evidence/frontiers change.
        """
        if self.bass is None:
            raise RuntimeError("calibration replay requires an initialized BASS prior")
        if not probe.warmup or self.physical_status[probe.physical_id] != DONE:
            raise RuntimeError("calibration replay requires a completed warmup probe")
        if probe.physical_id in self._replayed_calibration_terminals:
            return False
        self._replayed_calibration_terminals.add(probe.physical_id)
        virtual_probe = replace(
            probe, warmup=False, probe_start=0,
            rollout_suffix=probe.dispatch_path,
        )
        for node in virtual_probe.rollout_suffix:
            if int(self.tree.physical_id[node]) == int(UINT64_MAX):
                self.materialized_probe_nodes.add(int(node))
                self._refresh_bass_frontier(int(node))
        self._observe_probe_outcome(virtual_probe, outcome)
        return True

    def _observe_probe_outcome(
        self, probe: StaticProbe, outcome: StaticRuntimeOutcome,
    ) -> None:
        """Shared posterior backup for real and stored-metadata virtual probes."""
        if not outcome.valid or outcome.task_success is None:
            self._refresh_bass_frontier(probe.probe_start)
            self._invalidate_bass_acquisition_epoch()
            return
        category = None
        if self.bass is not None:
            category = self.bass.schema.encode(
                outcome.stage_evidence,
                bool(outcome.task_success),
            )
            self.bass_category_counts[category] += 1
        seen_posteriors: set[int] = set()
        for node in probe.rollout_suffix:
            posterior = self._active_posterior(int(node))
            identity = id(posterior)
            if identity in seen_posteriors:
                self.bass_duplicate_evidence_skips += 1
                continue
            seen_posteriors.add(identity)
            observed = (
                posterior.observe(category, probe.physical_id)
                if isinstance(posterior, MilestonePosterior)
                else posterior.observe(
                    bool(outcome.task_success),
                    probe.physical_id,
                )
            )
            if observed:
                self.bass_evidence_updates += 1
            else:
                self.bass_duplicate_evidence_skips += 1
            self._refresh_bass_frontier(int(node))
        if (
            outcome.task_success
            and probe.probe_index is not None
            and self.bass_first_success_probe is None
        ):
            self.bass_first_success_probe = probe.probe_index
            self.bass_first_success_wall_time = (
                time.monotonic() - self.started_at
            )
        self._invalidate_bass_acquisition_epoch()


class StaticVirtualRootProduct:
    """Runtime product of one geometry DAG and runtime-only root choices.

    Each root-rotation mode owns an independent mutable search overlay.  The
    immutable topology arrays remain shared through ``StructuralDAG`` mmap views.
    This prevents task evidence or physical evaluation status from leaking
    between actuator modes without duplicating the DAG artifact on disk.
    """

    def __init__(self, tree: StructuralDAG, config: BASSConfig) -> None:
        self.tree = tree
        self.config = config
        representation, modes = tree.rotation_spec(config.root_rotation_options)
        self.rotation_representation = representation
        self.artifact_rotation_modes = modes
        self.rotation_modes: Tuple[str | None, ...] = (None,) if representation == "embedded" else modes or (None,)
        self.virtual = self.rotation_modes != (None,)
        self.runtimes = tuple(
            StructuralDAGRuntime(tree, config) for _ in self.rotation_modes
        )
        for index, runtime in enumerate(self.runtimes):
            runtime.rng = random.Random(
                int(config.seed) + 104729 * (index + 1)
            )
        self._mode_index = {
            mode: index for index, mode in enumerate(self.rotation_modes)
        }
        self._root_stats = [RewardStatistics() for _ in self.rotation_modes]
        self._root_pending = np.zeros(len(self.rotation_modes), dtype=np.uint32)
        self._root_rng = random.Random(int(config.seed) ^ 0x5F3759DF)
        self._warmup_cursor = 0
        self._root_bass: MilestonePosteriorRegistry | None = None
        if self.primary.bass is not None:
            self._initialize_root_bass()
        self.started_at = time.monotonic()

    def _initialize_root_bass(self) -> None:
        """Give each rotation its own calibrated descendant-outcome posterior.

        Continuation-node posteriors only observe fixed-policy rollout suffixes;
        they are not aggregate evidence for the rotation that selected them.
        Warmup fits the shared prior but is not replayed as observations here.
        """
        if self._root_bass is not None or self.primary.bass is None:
            raise RuntimeError("root BASS must be initialized once after calibration")
        source = self.primary.bass
        self._root_bass = MilestonePosteriorRegistry(
            schema=source.schema,
            continuation_means=[
                beta / (alpha + beta)
                for alpha, beta in zip(
                    source.prior_stop_alpha, source.prior_survive_beta
                )
            ],
            prior_strengths=[
                alpha + beta
                for alpha, beta in zip(
                    source.prior_stop_alpha, source.prior_survive_beta
                )
            ],
        )

    def _root_bass_scores(self, available: Sequence[int]) -> List[float]:
        if self._root_bass is None:
            raise RuntimeError("root BASS scores requested before calibration")
        if self.config.acquisition == "bass_n2":
            return list(map(float, self._root_bass.two_step_parallel_scores(available)))
        return [self._root_bass.posterior(index).parallel_index() for index in available]

    def root_guidance_diagnostics(self) -> dict[str, Any]:
        """Bounded live telemetry without scanning the physical-state arrays."""
        available = self._available_mode_indices()
        scores = (
            dict(zip(available, self._root_bass_scores(available)))
            if self._root_bass is not None else {}
        )
        return {
            "static_root_guidance_mode": (
                self.config.acquisition if self._root_bass is not None
                else "calibration" if self.calibration_pending else "uniform_random"
            ),
            "static_root_bass_calibration": self.primary.bass_calibration,
            "static_root_guidance": [
                {
                    "mode": mode,
                    "available": index in available,
                    "completed_valid": self._root_stats[index].count,
                    "pending": int(self._root_pending[index]),
                    "acquisition_score": scores.get(index),
                    "posterior": (
                        self._root_bass.posterior(index).diagnostics()
                        if self._root_bass is not None else None
                    ),
                }
                for index, mode in enumerate(self.rotation_modes)
            ],
        }

    @property
    def primary(self) -> StructuralDAGRuntime:
        return self.runtimes[0]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)

    def _sum_runtime_counter(self, name: str) -> Any:
        return sum(getattr(runtime, name) for runtime in self.runtimes)

    @property
    def bass_selection_count(self) -> int:
        return int(self._sum_runtime_counter("bass_selection_count"))

    @property
    def bass_proposal_count(self) -> int:
        return int(self._sum_runtime_counter("bass_proposal_count"))

    @property
    def bass_repeat_proposals(self) -> int:
        return int(self._sum_runtime_counter("bass_repeat_proposals"))

    @property
    def bass_accepted_probes(self) -> int:
        return int(self._sum_runtime_counter("bass_accepted_probes"))

    @property
    def bass_evidence_updates(self) -> int:
        return int(self._sum_runtime_counter("bass_evidence_updates"))

    @property
    def bass_duplicate_evidence_skips(self) -> int:
        return int(
            self._sum_runtime_counter("bass_duplicate_evidence_skips")
        )

    @property
    def static_acquisition_epoch_count(self) -> int:
        return int(self._sum_runtime_counter("static_acquisition_epoch_count"))

    @property
    def static_acquisition_epoch_seconds(self) -> float:
        return float(self._sum_runtime_counter("static_acquisition_epoch_seconds"))

    @property
    def static_acquisition_frontiers_scored(self) -> int:
        return int(
            self._sum_runtime_counter("static_acquisition_frontiers_scored")
        )

    @property
    def static_candidates_from_epoch(self) -> int:
        return int(self._sum_runtime_counter("static_candidates_from_epoch"))

    @property
    def static_rollout_dead_edges_filtered(self) -> int:
        return int(
            self._sum_runtime_counter("static_rollout_dead_edges_filtered")
        )

    @property
    def static_rollout_exhausted_nodes(self) -> int:
        return int(self._sum_runtime_counter("static_rollout_exhausted_nodes"))

    @property
    def static_rollout_repeat_terminals(self) -> int:
        return int(self._sum_runtime_counter("static_rollout_repeat_terminals"))

    @property
    def bass_category_counts(self) -> List[int]:
        rows = [runtime.bass_category_counts for runtime in self.runtimes]
        length = max((len(row) for row in rows), default=0)
        return [
            sum(row[index] for row in rows if index < len(row))
            for index in range(length)
        ]

    def _runtime_for_probe(self, probe: StaticProbe) -> StructuralDAGRuntime:
        try:
            index = self._mode_index[probe.root_rotation_mode]
        except KeyError as exc:
            raise ValueError(
                "probe root rotation {!r} is outside {}".format(
                    probe.root_rotation_mode, self.rotation_modes
                )
            ) from exc
        return self.runtimes[index]

    def _tag(self, probe: StaticProbe, mode: str | None) -> StaticProbe:
        return replace(probe, root_rotation_mode=mode)

    def _available_mode_indices(self) -> List[int]:
        return [
            index
            for index, runtime in enumerate(self.runtimes)
            if not bool(runtime.resolved[0]) and bool(runtime.selectable[0])
        ]

    def _select_mode_index(self) -> int | None:
        available = self._available_mode_indices()
        if not available:
            return None

        if self._root_bass is not None:
            scores = self._root_bass_scores(available)
            maximum = max(scores)
            return self._root_rng.choice([
                index for index, score in zip(available, scores)
                if score == maximum
            ])

        weights = [int(np.count_nonzero(self.runtimes[i].physical_status == UNSEEN))
                   for i in available]
        return self._root_rng.choices(available, weights=weights, k=1)[0]

    @property
    def bass_enabled(self) -> bool:
        return self.primary.bass_enabled

    @property
    def calibration_pending(self) -> bool:
        return self.primary.calibration_pending

    @property
    def root_resolved(self) -> bool:
        return all(bool(runtime.resolved[0]) for runtime in self.runtimes)

    @property
    def evaluation_count(self) -> int:
        return self.tree.physical_count * len(self.rotation_modes)

    @property
    def physical_done_count(self) -> int:
        return sum(
            int(np.count_nonzero(runtime.physical_status == DONE))
            for runtime in self.runtimes
        )

    @property
    def sparse_statistics_node_count(self) -> int:
        return sum(len(runtime.runtime_stats) for runtime in self.runtimes)

    def select_warmup_probe(self) -> StaticProbe | None:
        for offset in range(len(self.runtimes)):
            index = (self._warmup_cursor + offset) % len(self.runtimes)
            probe = self.runtimes[index].select_warmup_probe()
            if probe is not None:
                self._warmup_cursor = (index + 1) % len(self.runtimes)
                return self._tag(probe, self.rotation_modes[index])
        return None

    def _select_from_runtime(self, *, bass: bool) -> StaticProbe | None:
        attempted: set[int] = set()
        while len(attempted) < len(self.runtimes):
            index = self._select_mode_index()
            if index is None or index in attempted:
                remaining = [
                    value
                    for value in self._available_mode_indices()
                    if value not in attempted
                ]
                if not remaining:
                    return None
                index = remaining[0]
            attempted.add(index)
            runtime = self.runtimes[index]
            probe = (
                runtime.select_bass_probe()
                if bass
                else runtime.select_probe()
            )
            if probe is not None:
                return self._tag(probe, self.rotation_modes[index])
        return None

    def select_bass_probe(self) -> StaticProbe | None:
        return self._select_from_runtime(bass=True)

    def select_probe(self) -> StaticProbe | None:
        return self._select_from_runtime(bass=False)

    def reserve_probe(self, probe: StaticProbe) -> None:
        index = self._mode_index[probe.root_rotation_mode]
        self.runtimes[index].reserve_probe(probe)
        self._root_pending[index] += 1
        if self._root_bass is not None and not probe.warmup:
            self._root_bass.posterior(index).reserve()

    def complete_probe(
        self,
        probe: StaticProbe,
        outcome: StaticRuntimeOutcome,
    ) -> None:
        index = self._mode_index[probe.root_rotation_mode]
        if self._root_pending[index] == 0:
            raise RuntimeError("virtual-root pending counter underflow")
        self._root_pending[index] -= 1
        self.runtimes[index].complete_probe(probe, outcome)
        if self._root_bass is not None and not probe.warmup:
            posterior = self._root_bass.posterior(index)
            posterior.release()
            self._observe_root_outcome(index, probe, outcome)
        if outcome.valid:
            value = outcome.reward
            self._root_stats[index].record(
                value,
                terminal_key=probe.physical_id,
                stage_evidence=outcome.stage_evidence,
            )

    def _observe_root_outcome(
        self, index: int, probe: StaticProbe, outcome: StaticRuntimeOutcome,
    ) -> None:
        if outcome.valid and outcome.task_success is not None:
            self._root_bass.posterior(index).observe(
                self._root_bass.schema.encode(
                    outcome.stage_evidence, bool(outcome.task_success)
                ),
                probe.physical_id,
            )

    def replay_calibration_probe(
        self, probe: StaticProbe, outcome: StaticRuntimeOutcome,
    ) -> bool:
        if self._root_bass is None:
            raise RuntimeError("calibration replay requires an initialized root BASS prior")
        index = self._mode_index[probe.root_rotation_mode]
        if not self.runtimes[index].replay_calibration_probe(probe, outcome):
            return False
        self._observe_root_outcome(index, probe, outcome)
        return True


    def initialize_milestone_posteriors(
        self,
        outcomes: Sequence[StaticRuntimeOutcome],
    ) -> dict[str, Any]:
        calibration = self.primary.initialize_milestone_posteriors(outcomes)
        for runtime in self.runtimes[1:]:
            runtime._initialize_bass(
                progress_thresholds=calibration["progress_thresholds"],
                continuation_means=calibration["continuation_prior_means"],
            )
            runtime.bass_calibration = calibration
        self._initialize_root_bass()
        return calibration

    def sequence_for_probe(self, probe: StaticProbe) -> List[Action]:
        sequence = self.tree.representative_sequence(probe.physical_id)
        if probe.root_rotation_mode is None:
            return sequence
        if sequence and isinstance(sequence[0], SelectRootRotation):
            raise RuntimeError(
                "virtual root rotation cannot wrap a DAG that already stores "
                "SelectRootRotation actions"
            )
        return [SelectRootRotation(probe.root_rotation_mode)] + sequence

    def diagnostics(self) -> dict[str, Any]:
        child_rows = []
        for index, (mode, statistics) in enumerate(
            zip(self.rotation_modes, self._root_stats)
        ):
            runtime = self.runtimes[index]
            row = {
                "mode": mode,
                "completed": statistics.count,
                "pending": int(self._root_pending[index]),
                "mean_value": statistics.mean,
                "resolved": bool(runtime.resolved[0]),
                "physical_done": int(
                    np.count_nonzero(runtime.physical_status == DONE)
                ),
            }
            if runtime.bass_enabled and not runtime.calibration_pending:
                row["root_posterior"] = runtime._active_posterior(0).diagnostics()
            child_rows.append(row)
        return {
            "dag_rotation_representation": self.rotation_representation,
            "dag_rotation_modes": list(self.artifact_rotation_modes),
            "static_virtual_root_active": self.virtual,
            **self.root_guidance_diagnostics(),
            "static_virtual_root_rotation_modes": [
                mode for mode in self.rotation_modes if mode is not None
            ],
            "static_virtual_root_evaluation_count": self.evaluation_count,
            "static_virtual_root_children": child_rows,
        }


def _better_static_outcome(
    candidate: StaticRuntimeOutcome,
    best: StaticRuntimeOutcome | None,
    config: BASSConfig,
) -> bool:
    if not candidate.valid:
        return False
    if best is None:
        return True
    if config.acquisition == "stage_monotonic":
        def key(value: StaticRuntimeOutcome) -> Tuple[float, float, float, float]:
            evidence = value.stage_evidence
            if evidence is None or not evidence.feasible:
                return (-1.0, 0.0, value.reward, -value.score)
            return (
                float(evidence.milestone),
                float(evidence.progress),
                float(evidence.residual_reward),
                -value.score,
            )
        return key(candidate) > key(best)
    if config.reward_mode == "bounded_task":
        return candidate.reward > best.reward + 1e-12 or (
            abs(candidate.reward - best.reward) <= 1e-12
            and candidate.score < best.score
        )
    return candidate.score < best.score


def run_structural_dag_search(
    tree: StructuralDAG,
    config: BASSConfig,
    evaluator: Callable[[List[Action]], Any],
) -> StaticSearchResult:
    """Search immutable topology with a disposable per-run statistics overlay."""

    runtime = StaticVirtualRootProduct(tree, config)
    workers = int(config.eval_workers or config.threads)
    budget = (
        None
        if int(config.iteration_budget) <= 0
        else int(config.iteration_budget) * int(config.threads)
    )
    deadline = (
        None
        if config.search_time_budget is None or config.search_time_budget <= 0
        else time.monotonic() + float(config.search_time_budget)
    )
    submitted = 0
    completed = 0
    valid = 0
    best_outcome: StaticRuntimeOutcome | None = None
    best_probe: StaticProbe | None = None
    warmup_submitted = 0
    warmup_completed = 0
    inflight: Dict[Any, StaticProbe] = {}
    ready_low = int(
        math.ceil(config.scheduler_ready_low_watermark_fraction * workers)
    )
    ready_high = max(
        1,
        int(
            math.ceil(
                config.scheduler_ready_high_watermark_fraction * workers
            )
        ),
    )
    ready: ReadyReservoir[StaticProbe] = ReadyReservoir(
        ready_low,
        ready_high,
    )
    refill_seconds = 0.0
    harvest_seconds = 0.0
    ready_low_watermark_hits = 0
    last_health_log = 0.0
    supply_critical_triggered = False
    evaluator_owner = getattr(evaluator, "__self__", None)
    evaluator_health_callback = getattr(
        evaluator_owner,
        "scheduler_health_snapshot",
        None,
    )
    supply_guard = SchedulerSupplyCriticalGuard(
        target_workers=workers,
        critical_fraction=config.scheduler_supply_critical_fraction,
        critical_seconds=config.scheduler_supply_critical_seconds,
        recent_new_seconds=config.scheduler_supply_recent_new_seconds,
        rate_window_seconds=config.scheduler_supply_rate_window_seconds,
    )
    occupancy_monitor = SchedulerOccupancyMonitor(
        target_workers=workers,
        low_fraction=config.scheduler_occupancy_warning_fraction,
        recovery_fraction=config.scheduler_occupancy_recovery_fraction,
        warning_duration_seconds=config.scheduler_occupancy_warning_seconds,
        recovery_duration_seconds=config.scheduler_occupancy_recovery_seconds,
    )
    diagnostics_path = (
        Path(config.diagnostics_jsonl).expanduser()
        if config.diagnostics_jsonl
        else None
    )
    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluator_health() -> dict[str, Any]:
        if not callable(evaluator_health_callback):
            return {}
        try:
            payload = evaluator_health_callback()
        except Exception as exc:
            return {"health_error": repr(exc)}
        return dict(payload) if isinstance(payload, dict) else {}

    def health_payload(event: str) -> dict[str, Any]:
        rates = supply_guard.rates()
        health = evaluator_health()
        return {
            "event": event,
            "elapsed_seconds": time.monotonic() - runtime.started_at,
            "submitted": submitted,
            "completed": completed,
            "static_active_evaluators": len(inflight),
            "static_actual_redmax_process_count": int(
                health.get("active", len(inflight)) or 0
            ),
            "static_ready_depth": len(ready),
            "static_ready_low_watermark": ready_low,
            "static_ready_high_watermark": ready_high,
            "static_candidate_supply_per_second": rates[
                "recent_candidate_supply_rate"
            ],
            "static_evaluator_demand_per_second": rates[
                "recent_evaluator_demand_rate"
            ],
            "static_supply_ratio": rates["recent_supply_ratio"],
            "static_acquisition_epoch_count": (
                runtime.static_acquisition_epoch_count
            ),
            "static_acquisition_epoch_seconds": (
                runtime.static_acquisition_epoch_seconds
            ),
            "static_acquisition_frontiers_scored": (
                runtime.static_acquisition_frontiers_scored
            ),
            "static_candidates_from_epoch": (
                runtime.static_candidates_from_epoch
            ),
            "static_rollout_dead_edges_filtered": (
                runtime.static_rollout_dead_edges_filtered
            ),
            "static_rollout_exhausted_nodes": (
                runtime.static_rollout_exhausted_nodes
            ),
            "static_rollout_repeat_terminals": (
                runtime.static_rollout_repeat_terminals
            ),
            "static_refill_seconds": refill_seconds,
            "static_harvest_seconds": harvest_seconds,
            "structural_dag_node_count": tree.node_count,
            "structural_dag_root_resolved": runtime.root_resolved,
            **runtime.root_guidance_diagnostics(),
            **occupancy_monitor.diagnostics(),
            **rates,
            **health,
        }

    def emit_health(event: str, *, force: bool = False) -> None:
        nonlocal last_health_log
        now = time.monotonic()
        interval = float(config.scheduler_health_log_interval)
        if not force and (interval <= 0.0 or now - last_health_log < interval):
            return
        last_health_log = now
        payload = health_payload(event)
        if diagnostics_path is not None:
            with diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if config.log_progress:
            print(
                "[static-scheduler] active={}/{} ready={}/{} "
                "supply={:.4f}/s demand={:.4f}/s ratio={:.3f} "
                "epochs={} candidates={}".format(
                    len(inflight),
                    workers,
                    len(ready),
                    ready_high,
                    payload["static_candidate_supply_per_second"],
                    payload["static_evaluator_demand_per_second"],
                    payload["static_supply_ratio"] or 0.0,
                    runtime.static_acquisition_epoch_count,
                    runtime.static_candidates_from_epoch,
                ),
                flush=True,
            )

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        if runtime.bass_enabled:
            warmup_target = min(
                int(config.calibration_budget),
                runtime.evaluation_count,
                (
                    int(config.calibration_budget)
                    if budget is None
                    else max(0, budget - submitted)
                ),
            )
            warmup_outcomes: List[StaticRuntimeOutcome] = []
            warmup_probes: List[StaticProbe] = []
            while warmup_completed < warmup_target:
                while (
                    len(inflight) < workers
                    and warmup_submitted < warmup_target
                    and (deadline is None or time.monotonic() < deadline)
                ):
                    probe = runtime.select_warmup_probe()
                    if probe is None:
                        break
                    runtime.reserve_probe(probe)
                    future = executor.submit(
                        evaluator,
                        runtime.sequence_for_probe(probe),
                    )
                    inflight[future] = probe
                    submitted += 1
                    warmup_submitted += 1
                if not inflight:
                    break
                done, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
                for future in done:
                    probe = inflight.pop(future)
                    try:
                        outcome = _coerce_runtime_outcome(future.result())
                    except Exception:
                        outcome = StaticRuntimeOutcome(
                            float("inf"), 0.0, False, None, None
                        )
                    runtime.complete_probe(probe, outcome)
                    warmup_outcomes.append(outcome)
                    warmup_probes.append(probe)
                    completed += 1
                    warmup_completed += 1
                    valid += int(outcome.valid)
                    if _better_static_outcome(outcome, best_outcome, config):
                        best_outcome = outcome
                        best_probe = probe
            warmup_observations = sum(
                outcome.valid and outcome.task_success is not None
                for outcome in warmup_outcomes
            )
            warmup_successes = sum(
                bool(outcome.task_success)
                for outcome in warmup_outcomes
                if outcome.valid and outcome.task_success is not None
            )
            if runtime.calibration_pending:
                calibration = runtime.initialize_milestone_posteriors(
                    warmup_outcomes
                )
                calibration_validator = getattr(
                    evaluator_owner, "validate_calibration", None
                )
                if callable(calibration_validator):
                    calibration_validator(calibration)
                replayed = sum(
                    runtime.replay_calibration_probe(probe, outcome)
                    for probe, outcome in zip(warmup_probes, warmup_outcomes)
                )
                calibration.update({
                    "calibration_reused_as_posterior_evidence": True,
                    "calibration_virtual_runs": replayed,
                    "calibration_virtual_valid_outcomes": warmup_observations,
                    "calibration_virtual_invalid_outcomes": (
                        len(warmup_outcomes) - warmup_observations
                    ),
                    "calibration_virtual_path_policy": "representative_root_to_terminal",
                    "calibration_virtual_lower_level_calls": 0,
                })
                if config.log_progress:
                    print(
                        "[static-bass] integrated BASS calibration "
                        "accepted={}/{} levels={} prior_success={:.8g}".format(
                            calibration["source_accepted_ordinal_rows"],
                            calibration["warmup_total_outcomes"],
                            calibration["ordinal_level_count"],
                            calibration["prior_success_mean"],
                        ),
                        flush=True,
                    )
                    print(
                        "[static-bass] replayed calibration metadata "
                        "virtual_runs={} valid={} lower_level_calls=0".format(
                            replayed, warmup_observations
                        ),
                        flush=True,
                    )
            # Keep neither paths nor outcomes alive during the guided phase.
            warmup_probes.clear()
            warmup_outcomes.clear()
            if config.log_progress:
                if runtime.bass is not None:
                    bass_diagnostics = runtime.bass.diagnostics()
                    print(
                        "[static-bass] BASS runtime overlay initialized "
                        "mode={} levels={} prior_success={:.8g} "
                        "materialized={}".format(
                            config.acquisition,
                            bass_diagnostics["bass_level_count"],
                            bass_diagnostics["bass_prior_success_mean"],
                            len(runtime.materialized_probe_nodes),
                        ),
                        flush=True,
                    )

        def harvest_completed(*, block: bool) -> int:
            nonlocal completed, valid, best_outcome, best_probe
            nonlocal harvest_seconds
            if not inflight:
                return 0
            started = time.perf_counter()
            if block:
                done, _ = wait(
                    tuple(inflight),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
            else:
                done = {future for future in inflight if future.done()}
            harvested = 0
            for future in done:
                probe = inflight.pop(future)
                try:
                    outcome = _coerce_runtime_outcome(future.result())
                except Exception:
                    outcome = StaticRuntimeOutcome(
                        score=float("inf"),
                        reward=0.0,
                        valid=False,
                        stage_evidence=None,
                    )
                runtime.complete_probe(probe, outcome)
                completed += 1
                harvested += 1
                valid += int(outcome.valid)
                if _better_static_outcome(outcome, best_outcome, config):
                    best_outcome = outcome
                    best_probe = probe
            supply_guard.record_completion(harvested)
            harvest_seconds += time.perf_counter() - started
            return harvested

        def submission_open() -> bool:
            planned = submitted + len(ready)
            return (
                (budget is None or planned < budget)
                and (deadline is None or time.monotonic() < deadline)
                and not runtime.root_resolved
            )

        def refill_ready() -> int:
            nonlocal refill_seconds, ready_low_watermark_hits
            if len(ready) >= ready_low or not submission_open():
                return 0
            ready_low_watermark_hits += 1
            started = time.perf_counter()
            produced = 0
            proposal_before = runtime.bass_proposal_count
            repeat_before = runtime.bass_repeat_proposals
            while len(ready) < ready_high and submission_open():
                if runtime.bass_enabled:
                    probe = runtime.select_bass_probe()
                else:
                    probe = runtime.select_probe()
                if probe is None:
                    break
                runtime.reserve_probe(probe)
                ready.append(probe)
                produced += 1
            proposal_count = runtime.bass_proposal_count - proposal_before
            supply_guard.record_proposals(
                total=max(produced, proposal_count),
                novel=produced,
                repeat=runtime.bass_repeat_proposals - repeat_before,
                invalid=0,
                cache=0,
            )
            refill_seconds += time.perf_counter() - started
            return produced

        def dispatch_ready() -> int:
            nonlocal submitted
            dispatched = 0
            while ready and len(inflight) < workers:
                probe = ready.popleft()
                sequence = runtime.sequence_for_probe(probe)
                future = executor.submit(evaluator, sequence)
                inflight[future] = probe
                submitted += 1
                dispatched += 1
            return dispatched

        def check_supply_critical() -> None:
            nonlocal supply_critical_triggered
            if not supply_guard.observe(
                active_futures=len(inflight),
                ready_depth=len(ready),
                submission_open=submission_open(),
                root_exhausted=runtime.root_resolved,
            ):
                return
            supply_critical_triggered = True
            emit_health("scheduler_supply_critical", force=True)
            owner = getattr(evaluator, "__self__", None)
            abort = getattr(owner, "scheduler_abort_low_level", None)
            if callable(abort):
                abort("scheduler_supply_critical")
            raise RuntimeError(
                "scheduler_supply_critical: static candidate supply remained "
                "chronically below real evaluator demand"
            )

        while True:
            harvest_completed(block=False)
            dispatch_ready()
            refill_ready()
            dispatch_ready()
            current_health = evaluator_health()
            occupancy_monitor.observe(
                int(current_health.get("active", len(inflight)) or 0),
                submission_open=submission_open(),
            )
            emit_health("static_scheduler_health")
            check_supply_critical()

            if not inflight and not ready:
                if not submission_open():
                    break
                # No selectable candidate remains even though the immutable
                # DAG has not yet propagated root exhaustion.
                if runtime.bass_enabled:
                    probe = runtime.select_bass_probe()
                else:
                    probe = runtime.select_probe()
                if probe is None:
                    break
                runtime.reserve_probe(probe)
                ready.append(probe)
                supply_guard.record_proposals(
                    total=1,
                    novel=1,
                    repeat=0,
                    invalid=0,
                    cache=0,
                )
                continue
            if inflight:
                harvest_completed(block=True)

        # READY entries are committed reservations; dispatch and drain them.
        while ready or inflight:
            dispatch_ready()
            if inflight:
                harvest_completed(block=True)
        emit_health("static_scheduler_complete", force=True)
    finally:
        executor.shutdown(wait=True)
    best_sequence = (
        runtime.sequence_for_probe(best_probe)
        if best_probe is not None
        else []
    )
    best_function_count = (
        sum(
            isinstance(action, AddLink) and action.start_function_group
            for action in best_sequence
        )
        if best_sequence
        else None
    )
    bass_diagnostics: dict[str, Any] = {}
    if runtime.bass is not None:
        root_posterior = runtime._bass_posterior(0)
        selectable_frontier = [
            int(node)
            for node in runtime.materialized_probe_nodes
            if runtime.selectable[int(node)]
            and int(tree.physical_id[int(node)]) == int(UINT64_MAX)
        ]
        if config.acquisition == "bass_n2":
            scores = runtime.bass.two_step_parallel_scores(selectable_frontier)
            top_frontiers = heapq.nlargest(
                20,
                zip(selectable_frontier, map(float, scores)),
                key=lambda item: (item[1], -item[0]),
            )
        else:
            top_frontiers = heapq.nlargest(
                20,
                (
                    (node, runtime._bass_posterior(node).parallel_index())
                    for node in selectable_frontier
                ),
                key=lambda item: (item[1], -item[0]),
            )
        frontier_rows = []
        for node, score in top_frontiers:
            posterior = runtime._bass_posterior(node)
            row = {
                "node_id": int(node),
                **posterior.diagnostics(),
            }
            if config.acquisition == "bass_n2":
                row["bass_two_step_value"] = float(score)
            frontier_rows.append(row)
        bass_diagnostics = {
            "bass_static_runtime": True,
            "bass_artifact_mutated": False,
            "bass_acquisition_mode": config.acquisition,
            "bass_two_step_independent_frontier_exact_without_pending": (
                config.acquisition == "bass_n2"
            ),
            "bass_suffix_sharing_acquisition_approximation": True,
            "bass_pending_diversification": "moment_matched_beta",
            "calibration_artifact": config.calibration_artifact,
            "bass_integrated_calibration": runtime.bass_calibration,
            **runtime.bass.diagnostics(),
            "bass_materialized_probe_nodes": len(runtime.materialized_probe_nodes),
            "bass_selection_count": runtime.bass_selection_count,
            "bass_proposal_count": runtime.bass_proposal_count,
            "bass_repeat_proposals": runtime.bass_repeat_proposals,
            "bass_accepted_probes": runtime.bass_accepted_probes,
            "bass_evidence_updates": runtime.bass_evidence_updates,
            "bass_duplicate_evidence_skips": (
                runtime.bass_duplicate_evidence_skips
            ),
            "bass_first_success_probe": runtime.bass_first_success_probe,
            "bass_first_success_wall_time": (
                runtime.bass_first_success_wall_time
            ),
            "bass_observed_category_counts": list(runtime.bass_category_counts),
            "bass_two_step_selection_calls": runtime.bass_two_step_selection_calls,
            "bass_two_step_selection_seconds": (
                runtime.bass_two_step_selection_seconds
            ),
            "bass_two_step_scored_frontiers": (
                runtime.bass_two_step_scored_frontiers
            ),
            "bass_two_step_max_frontiers": runtime.bass_two_step_max_frontiers,
            "static_acquisition_epoch_count": (
                runtime.static_acquisition_epoch_count
            ),
            "static_acquisition_epoch_seconds": (
                runtime.static_acquisition_epoch_seconds
            ),
            "static_acquisition_frontiers_scored": (
                runtime.static_acquisition_frontiers_scored
            ),
            "static_candidates_from_epoch": (
                runtime.static_candidates_from_epoch
            ),
            "static_frontiers_scored_per_candidate": (
                runtime.static_acquisition_frontiers_scored
                / max(1, runtime.static_candidates_from_epoch)
            ),
            "static_rollout_dead_edges_filtered": (
                runtime.static_rollout_dead_edges_filtered
            ),
            "static_rollout_exhausted_nodes": (
                runtime.static_rollout_exhausted_nodes
            ),
            "static_rollout_repeat_terminals": (
                runtime.static_rollout_repeat_terminals
            ),
            "bass_root_statistics": root_posterior.diagnostics(),
            "bass_top_frontiers": frontier_rows,
            "calibration_submitted": warmup_submitted,
            "calibration_completed": warmup_completed,
        }
    return StaticSearchResult(
        best_sequence=best_sequence,
        best_score=best_outcome.score if best_outcome else float("inf"),
        best_reward=best_outcome.reward if best_outcome else float("-inf"),
        total_iterations=submitted,
        completed_candidates=completed,
        valid_completed_candidates=valid,
        best_function_count=best_function_count,
        diagnostics={
            "structural_dag_enabled": True,
            "static_dag_enabled": True,
            "structural_dag_path": str(tree.root),
            "structural_dag_node_count": tree.node_count,
            "static_dag_edge_count": tree.edge_count_total,
            "structural_dag_terminal_alias_count": tree.terminal_alias_count,
            "structural_dag_physical_count": tree.physical_count,
            "structural_dag_physical_done": runtime.physical_done_count,
            "structural_dag_root_resolved": runtime.root_resolved,
            "structural_dag_sparse_statistics_nodes": (
                runtime.sparse_statistics_node_count
            ),
            "static_scheduler_v2_active": bool(config.scheduler_v2),
            "static_ready_low_watermark": ready_low,
            "static_ready_high_watermark": ready_high,
            "static_ready_low_watermark_hits": ready_low_watermark_hits,
            "static_refill_seconds": refill_seconds,
            "static_harvest_seconds": harvest_seconds,
            "scheduler_supply_critical_triggered": (
                supply_critical_triggered or supply_guard.triggered
            ),
            "static_ready_depth_final": len(ready),
            "static_active_evaluators_final": len(inflight),
            "static_candidate_supply_per_second": supply_guard.rates()[
                "recent_candidate_supply_rate"
            ],
            "static_evaluator_demand_per_second": supply_guard.rates()[
                "recent_evaluator_demand_rate"
            ],
            "static_supply_ratio": supply_guard.rates()[
                "recent_supply_ratio"
            ],
            **occupancy_monitor.diagnostics(),
            **runtime.diagnostics(),
            **bass_diagnostics,
        },
    )
