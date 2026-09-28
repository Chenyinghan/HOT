"""Layer-synchronous external compiler for the immutable physical search DAG.

The builder deliberately keeps canonical interning out of the expansion hot
path.  Every grammar depth is generated into append-only digest shards and is
then reduced exactly at a layer barrier.  SHA-256 is only a partitioning key;
complete compressed canonical descriptors decide equality for repeated
digests.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import struct
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from ..actions import action_from_dict
from .canonical import PACKED_SIGNATURE_VERSION
from .expansion import (
    _BatchExpansionTarget,
    _ParentExpansionBatchJob,
    _action_from_key,
    _action_key,
    _atomic_json,
    _directory_bytes,
    _expand_parent_batch_worker,
    _initialize_expansion_worker,
)


UINT32_MAX_INT = int(np.iinfo(np.uint32).max)
UINT64_MAX_INT = int(np.iinfo(np.uint64).max)
LAYERED_FORMAT = "hot-static-dag-layered-builder-v1"
RECORD_DTYPE = np.dtype(
    [
        ("parent", "<u8"),
        ("action_token", "<u4"),
        ("digest", "u1", (32,)),
        ("key_offset", "<u8"),
        ("key_length", "<u4"),
    ],
    align=False,
)
EDGE_DTYPE = np.dtype(
    [("parent", "<u8"), ("child", "<u8"), ("action", "<u4")],
    align=False,
)


@dataclass(frozen=True)
class _LayerInfo:
    depth: int
    base: int
    count: int
    expanded: bool
    edge_count: int


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _layer_dir(scratch: Path, depth: int) -> Path:
    return scratch / "layers" / "depth_{:03d}".format(int(depth))


def _load_manifest(scratch: Path) -> dict[str, Any]:
    return json.loads((scratch / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(scratch: Path, manifest: dict[str, Any]) -> None:
    payload = dict(manifest)
    payload["updated_unix"] = time.time()
    payload["scratch_bytes"] = _directory_bytes(scratch)
    _atomic_json(scratch / "manifest.json", payload)
    _fsync_directory(scratch)


def _action_payload(action: Any) -> str:
    return json.dumps(action.to_dict(), sort_keys=True, separators=(",", ":"))


def _load_actions(scratch: Path) -> list[Any]:
    path = scratch / "actions.json"
    if not path.exists():
        return []
    payloads = json.loads(path.read_text(encoding="utf-8"))
    return [action_from_dict(payload) for payload in payloads]


def _save_actions(scratch: Path, actions: Sequence[Any]) -> None:
    _atomic_text(
        scratch / "actions.json",
        json.dumps([action.to_dict() for action in actions], indent=2, sort_keys=True)
        + "\n",
    )


def _write_layer_nodes(
    directory: Path,
    *,
    rep_parent: np.ndarray,
    rep_action: np.ndarray,
    sequences: np.ndarray,
    physical_id: Optional[np.ndarray] = None,
    physical_digest: Optional[np.ndarray] = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    count = int(len(rep_parent))
    if rep_action.shape != (count,) or sequences.shape[0] != count:
        raise ValueError("inconsistent layered node arrays")
    _atomic_array(directory / "node_rep_parent.npy", np.asarray(rep_parent, dtype=np.uint64))
    _atomic_array(directory / "node_rep_action.npy", np.asarray(rep_action, dtype=np.uint32))
    _atomic_array(directory / "node_sequences.npy", np.asarray(sequences, dtype=np.uint32))
    if physical_id is None:
        physical_id = np.full(count, UINT64_MAX_INT, dtype=np.uint64)
    if physical_digest is None:
        physical_digest = np.zeros((count, 32), dtype=np.uint8)
    _write_terminal_annotations(directory, physical_id, physical_digest)
    _fsync_directory(directory)


def _write_terminal_annotations(
    directory: Path,
    physical_id: np.ndarray,
    physical_digest: np.ndarray,
) -> None:
    physical_id = np.asarray(physical_id, dtype=np.uint64)
    physical_digest = np.asarray(physical_digest, dtype=np.uint8)
    if physical_digest.shape != (len(physical_id), 32):
        raise ValueError("inconsistent layered terminal digest array")
    local = np.flatnonzero(physical_id != UINT64_MAX_INT).astype(np.uint64)
    _atomic_array(directory / "terminal_local.npy", local)
    _atomic_array(directory / "terminal_physical_id.npy", physical_id[local])
    _atomic_array(directory / "terminal_digest.npy", physical_digest[local])


def _write_layer_edges(
    directory: Path,
    *,
    parent_base: int,
    parent_count: int,
    child_base: int,
    child_count: int,
    edges: np.ndarray,
) -> None:
    if edges.dtype != EDGE_DTYPE:
        edges = np.asarray(edges, dtype=EDGE_DTYPE)
    if len(edges):
        order = np.lexsort((edges["child"], edges["parent"]))
        edges = edges[order]
    degrees = np.zeros(parent_count, dtype=np.uint64)
    if len(edges):
        local_parent = edges["parent"].astype(np.int64) - int(parent_base)
        if np.any(local_parent < 0) or np.any(local_parent >= parent_count):
            raise RuntimeError("layer edge parent is out of range")
        np.add.at(degrees, local_parent, 1)
    offsets = np.empty(parent_count + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(degrees, out=offsets[1:])

    reverse_order = (
        np.lexsort((edges["parent"], edges["child"]))
        if len(edges)
        else np.empty(0, dtype=np.int64)
    )
    reverse = edges[reverse_order]
    reverse_degrees = np.zeros(child_count, dtype=np.uint64)
    if len(reverse):
        local_child = reverse["child"].astype(np.int64) - int(child_base)
        if np.any(local_child < 0) or np.any(local_child >= child_count):
            raise RuntimeError("layer edge child is out of range")
        np.add.at(reverse_degrees, local_child, 1)
    reverse_offsets = np.empty(child_count + 1, dtype=np.uint64)
    reverse_offsets[0] = 0
    np.cumsum(reverse_degrees, out=reverse_offsets[1:])

    _atomic_array(directory / "edge_offsets.npy", offsets)
    _atomic_array(directory / "edge_child.npy", edges["child"].astype(np.uint64))
    _atomic_array(directory / "edge_action.npy", edges["action"].astype(np.uint32))
    _atomic_array(directory / "reverse_offsets.npy", reverse_offsets)
    _atomic_array(directory / "reverse_parent.npy", reverse["parent"].astype(np.uint64))
    _fsync_directory(directory)


class _ShardSpool:
    def __init__(self, root: Path, shard_count: int) -> None:
        self.root = root
        self.shard_count = int(shard_count)
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: Dict[int, Any] = {}
        self._keys: Dict[int, Any] = {}
        self.record_counts = np.zeros(self.shard_count, dtype=np.uint64)

    def append(
        self,
        *,
        parent: int,
        action_token: int,
        digest: bytes,
        canonical_key: bytes,
    ) -> None:
        shard = int(digest[0]) % self.shard_count
        record_handle = self._records.get(shard)
        key_handle = self._keys.get(shard)
        if record_handle is None:
            directory = self.root / "shard_{:03d}".format(shard)
            directory.mkdir(parents=True, exist_ok=True)
            record_handle = (directory / "records.bin").open("ab")
            key_handle = (directory / "keys.blob").open("ab")
            self._records[shard] = record_handle
            self._keys[shard] = key_handle
        offset = key_handle.tell()
        key_handle.write(canonical_key)
        record = np.zeros(1, dtype=RECORD_DTYPE)
        record["parent"] = int(parent)
        record["action_token"] = int(action_token)
        record["digest"][0, :] = np.frombuffer(digest, dtype=np.uint8)
        record["key_offset"] = int(offset)
        record["key_length"] = len(canonical_key)
        record_handle.write(record.tobytes())
        self.record_counts[shard] += 1

    def close(self) -> None:
        for handle in list(self._records.values()) + list(self._keys.values()):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._records.clear()
        self._keys.clear()
        _atomic_array(self.root / "record_counts.npy", self.record_counts)
        _fsync_directory(self.root)


def _read_exact_key(handle: Any, record: np.void) -> bytes:
    handle.seek(int(record["key_offset"]))
    value = handle.read(int(record["key_length"]))
    if len(value) != int(record["key_length"]):
        raise RuntimeError("truncated layered canonical-key blob")
    return value


def _reduce_shard_job(args: tuple[str, int, np.ndarray]) -> dict[str, Any]:
    shard_root_text, shard_id, action_map = args
    started = time.perf_counter()
    shard_root = Path(shard_root_text) / "shard_{:03d}".format(int(shard_id))
    records_path = shard_root / "records.bin"
    if not records_path.exists() or records_path.stat().st_size == 0:
        return {"shard": int(shard_id), "nodes": 0, "edges": 0, "seconds": 0.0}
    if records_path.stat().st_size % RECORD_DTYPE.itemsize:
        raise RuntimeError("corrupt layered proposal record file")
    records = np.fromfile(records_path, dtype=RECORD_DTYPE)
    digest_values = (
        np.ascontiguousarray(records["digest"].reshape((-1, 32)))
        .view("V32")
        .reshape(-1)
    )
    order = np.argsort(digest_values, kind="stable")
    sorted_digest = digest_values[order]
    starts = np.r_[0, np.flatnonzero(sorted_digest[1:] != sorted_digest[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    extras = np.zeros(len(starts), dtype=np.uint32)
    exact_subgroups: Dict[int, list[tuple[bytes, np.ndarray]]] = {}
    key_handle = (shard_root / "keys.blob").open("rb")
    try:
        for group_index in np.flatnonzero((ends - starts) > 1):
            indices = order[int(starts[group_index]) : int(ends[group_index])]
            grouped: Dict[bytes, list[int]] = {}
            for proposal_index in indices:
                key = _read_exact_key(key_handle, records[int(proposal_index)])
                grouped.setdefault(key, []).append(int(proposal_index))
            subgroups = [
                (key, np.asarray(grouped[key], dtype=np.int64))
                for key in sorted(grouped)
            ]
            exact_subgroups[int(group_index)] = subgroups
            extras[int(group_index)] = len(subgroups) - 1
    finally:
        key_handle.close()

    shifts = np.concatenate(
        (np.zeros(1, dtype=np.uint64), np.cumsum(extras[:-1], dtype=np.uint64))
    )
    group_bases = np.arange(len(starts), dtype=np.uint64) + shifts
    node_count = int(len(starts) + int(np.sum(extras, dtype=np.uint64)))
    proposal_rank = np.empty(len(records), dtype=np.uint64)
    repeated_groups = np.repeat(np.arange(len(starts), dtype=np.int64), ends - starts)
    proposal_rank[order] = group_bases[repeated_groups]
    global_actions = np.asarray(action_map, dtype=np.uint32)[records["action_token"]]

    rep_parent = np.empty(node_count, dtype=np.uint64)
    rep_action = np.empty(node_count, dtype=np.uint32)
    first_indices = order[starts]
    rep_parent[group_bases] = records["parent"][first_indices]
    rep_action[group_bases] = global_actions[first_indices]
    for group_index, subgroups in exact_subgroups.items():
        base = int(group_bases[group_index])
        for subgroup_offset, (_, indices) in enumerate(subgroups):
            pair_order = np.lexsort(
                (global_actions[indices], records["parent"][indices])
            )
            selected = int(indices[int(pair_order[0])])
            rank = base + subgroup_offset
            rep_parent[rank] = int(records["parent"][selected])
            rep_action[rank] = int(global_actions[selected])
            proposal_rank[indices] = rank

    edge_order = np.lexsort((global_actions, proposal_rank, records["parent"]))
    edge_parent = records["parent"][edge_order]
    edge_child = proposal_rank[edge_order]
    edge_action = global_actions[edge_order]
    keep = np.ones(len(edge_order), dtype=np.bool_)
    if len(keep) > 1:
        keep[1:] = (edge_parent[1:] != edge_parent[:-1]) | (
            edge_child[1:] != edge_child[:-1]
        )
    edges = np.empty(int(np.count_nonzero(keep)), dtype=EDGE_DTYPE)
    edges["parent"] = edge_parent[keep]
    edges["child"] = edge_child[keep]
    edges["action"] = edge_action[keep]
    _atomic_array(shard_root / "reduced_rep_parent.npy", rep_parent)
    _atomic_array(shard_root / "reduced_rep_action.npy", rep_action)
    _atomic_array(shard_root / "reduced_edges.npy", edges)
    return {
        "shard": int(shard_id),
        "nodes": node_count,
        "edges": int(len(edges)),
        "proposals": int(len(records)),
        "digest_groups": int(len(starts)),
        "singleton_digest_groups": int(np.count_nonzero((ends - starts) == 1)),
        "exact_key_reads": int(
            sum(
                sum(len(indices) for _, indices in subgroups)
                for subgroups in exact_subgroups.values()
            )
        ),
        "seconds": time.perf_counter() - started,
    }


def _batch_jobs(
    layer: _LayerInfo,
    sequences: np.ndarray,
    actions: Sequence[Any],
    job_parents: int,
    root_rotation_action_ids: Optional[set[int]] = None,
) -> Iterator[_ParentExpansionBatchJob]:
    for begin in range(0, layer.count, int(job_parents)):
        end = min(layer.count, begin + int(job_parents))
        targets = []
        required = set()
        for local in range(begin, end):
            action_ids = tuple(int(value) for value in sequences[local])
            if (
                root_rotation_action_ids is not None
                and (
                    not action_ids
                    or action_ids[0] not in root_rotation_action_ids
                )
            ):
                continue
            required.update(action_ids)
            targets.append(
                _BatchExpansionTarget(
                    parent_id=layer.base + local,
                    rep_sequence=np.asarray(action_ids, dtype=np.uint32).tobytes(),
                    action_ids=action_ids,
                )
            )
        if not targets:
            continue
        yield _ParentExpansionBatchJob(
            depth=layer.depth,
            targets=tuple(targets),
            action_catalog=tuple((index, actions[index]) for index in sorted(required)),
        )


def _intern_layer_actions(
    scratch: Path,
    actions: list[Any],
    temporary_keys: Sequence[Tuple[Any, ...]],
) -> np.ndarray:
    existing = {_action_key(action): index for index, action in enumerate(actions)}
    missing = sorted(
        (key for key in temporary_keys if key not in existing),
        key=lambda key: _action_payload(_action_from_key(key)),
    )
    for key in missing:
        existing[key] = len(actions)
        actions.append(_action_from_key(key))
    _save_actions(scratch, actions)
    return np.asarray([existing[key] for key in temporary_keys], dtype=np.uint32)


def _expand_layer(
    *,
    scratch: Path,
    layer: _LayerInfo,
    actions: list[Any],
    assets: Sequence[Any],
    config: Any,
    initial_forbidden_boxes: Sequence[Any] | None,
    initial_root_box: Any,
    workers: int,
    job_parents: int,
    inflight_multiplier: int,
    shard_count: int,
    progress_interval: int,
    continuation_root_rotation: Optional[str] = None,
) -> tuple[Path, np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    working = scratch / "working" / "depth_{:03d}".format(layer.depth)
    if working.exists():
        shutil.rmtree(working)
    spool = _ShardSpool(working / "proposals", shard_count)
    physical_id = np.full(layer.count, UINT64_MAX_INT, dtype=np.uint64)
    physical_digest = np.zeros((layer.count, 32), dtype=np.uint8)
    terminal_results: Dict[int, bytes] = {}
    temporary_keys: list[Tuple[Any, ...]] = []
    temporary_ids: Dict[Tuple[Any, ...], int] = {}
    sequences = np.load(_layer_dir(scratch, layer.depth) / "node_sequences.npy", mmap_mode="r")
    root_rotation_action_ids = None
    if continuation_root_rotation is not None and layer.depth > 0:
        root_rotation_action_ids = {
            index
            for index, action in enumerate(actions)
            if getattr(action, "kind", None) == "SelectRootRotation"
            and getattr(action, "mode", None) == continuation_root_rotation
        }
        if not root_rotation_action_ids:
            raise RuntimeError(
                "continuation root rotation {!r} has no interned action".format(
                    continuation_root_rotation
                )
            )
    jobs = iter(
        _batch_jobs(
            layer,
            sequences,
            actions,
            job_parents,
            root_rotation_action_ids,
        )
    )
    proposal_count = 0
    completed_parents = 0
    diagnostics = []
    next_progress = max(1, int(progress_interval))

    def consume(batch: Any) -> None:
        nonlocal proposal_count, completed_parents, next_progress
        diagnostics.append(
            {
                "parents": len(batch.results),
                "replay_seconds": batch.replay_seconds,
                "generation_seconds": batch.generation_seconds,
                "naive_replay_transitions": batch.naive_replay_transitions,
                "prefix_replay_transitions": batch.prefix_replay_transitions,
            }
        )
        for result in batch.results:
            completed_parents += 1
            local = int(result.parent_id) - layer.base
            if result.is_complete and result.valid_terminal:
                if result.terminal_digest is None:
                    raise RuntimeError("valid layered terminal is missing its digest")
                terminal_results[local] = bytes(result.terminal_digest)
            for proposal in result.children:
                token = temporary_ids.get(proposal.action_key)
                if token is None:
                    token = len(temporary_keys)
                    temporary_ids[proposal.action_key] = token
                    temporary_keys.append(proposal.action_key)
                spool.append(
                    parent=result.parent_id,
                    action_token=token,
                    digest=proposal.digest,
                    canonical_key=proposal.canonical_key,
                )
                proposal_count += 1
        if progress_interval > 0 and proposal_count >= next_progress:
            print(
                "[static-dag-layered-spool] depth={} parents={}/{} proposals={}".format(
                    layer.depth, completed_parents, layer.count, proposal_count
                ),
                flush=True,
            )
            while next_progress <= proposal_count:
                next_progress += max(1, int(progress_interval))

    if int(workers) == 1:
        _initialize_expansion_worker(assets, config, initial_forbidden_boxes, initial_root_box)
        for job in jobs:
            consume(_expand_parent_batch_worker(job))
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_expansion_worker,
            initargs=(assets, config, initial_forbidden_boxes, initial_root_box),
        ) as executor:
            pending = set()
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < int(workers) * int(inflight_multiplier):
                    try:
                        job = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(executor.submit(_expand_parent_batch_worker, job))
                if not pending:
                    continue
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    consume(future.result())
    spool.close()
    eligible_parents = sum(
        int(item["parents"])
        for item in diagnostics
        if "parents" in item
    )
    if completed_parents != eligible_parents:
        raise RuntimeError("layer expansion did not return every eligible parent")
    diagnostics.append(
        {
            "selection": "continuation_root_rotation",
            "mode": continuation_root_rotation,
            "eligible_parents": eligible_parents,
            "filtered_parents": layer.count - eligible_parents,
        }
    )
    next_physical = 0
    for local in sorted(terminal_results):
        physical_id[local] = next_physical
        physical_digest[local, :] = np.frombuffer(terminal_results[local], dtype=np.uint8)
        next_physical += 1
    action_map = _intern_layer_actions(scratch, actions, temporary_keys)
    _atomic_array(working / "action_map.npy", action_map)
    return working, physical_id, physical_digest, diagnostics, proposal_count


def _reduce_layer(
    *,
    scratch: Path,
    layer: _LayerInfo,
    working: Path,
    physical_id: np.ndarray,
    physical_digest: np.ndarray,
    reducer_workers: int,
    shard_count: int,
) -> tuple[Optional[_LayerInfo], int, list[dict[str, Any]]]:
    current_dir = _layer_dir(scratch, layer.depth)
    _write_terminal_annotations(current_dir, physical_id, physical_digest)
    action_map = np.load(working / "action_map.npy", mmap_mode="r")
    jobs = [
        (str(working / "proposals"), shard, np.asarray(action_map))
        for shard in range(int(shard_count))
        if (working / "proposals" / "shard_{:03d}".format(shard) / "records.bin").exists()
    ]
    if int(reducer_workers) == 1:
        reports = [_reduce_shard_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=int(reducer_workers),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            reports = list(executor.map(_reduce_shard_job, jobs))
    reports.sort(key=lambda item: item["shard"])
    node_counts = np.zeros(shard_count, dtype=np.uint64)
    edge_count = 0
    for report in reports:
        node_counts[int(report["shard"])] = int(report["nodes"])
        edge_count += int(report["edges"])
    total_nodes = int(np.sum(node_counts, dtype=np.uint64))
    if total_nodes == 0:
        _write_layer_edges(
            current_dir,
            parent_base=layer.base,
            parent_count=layer.count,
            child_base=layer.base + layer.count,
            child_count=0,
            edges=np.empty(0, dtype=EDGE_DTYPE),
        )
        return None, 0, reports

    shard_bases = np.r_[0, np.cumsum(node_counts[:-1], dtype=np.uint64)]
    next_base = layer.base + layer.count
    rep_parent = np.empty(total_nodes, dtype=np.uint64)
    rep_action = np.empty(total_nodes, dtype=np.uint32)
    edges = np.empty(edge_count, dtype=EDGE_DTYPE)
    edge_cursor = 0
    for report in reports:
        shard = int(report["shard"])
        directory = working / "proposals" / "shard_{:03d}".format(shard)
        count = int(report["nodes"])
        base = int(shard_bases[shard])
        rep_parent[base : base + count] = np.load(directory / "reduced_rep_parent.npy")
        rep_action[base : base + count] = np.load(directory / "reduced_rep_action.npy")
        shard_edges = np.load(directory / "reduced_edges.npy")
        size = len(shard_edges)
        edges[edge_cursor : edge_cursor + size] = shard_edges
        edges["child"][edge_cursor : edge_cursor + size] += next_base + base
        edge_cursor += size
    parent_sequences = np.load(current_dir / "node_sequences.npy", mmap_mode="r")
    sequences = np.empty((total_nodes, layer.depth + 1), dtype=np.uint32)
    if layer.depth:
        parent_local = rep_parent.astype(np.int64) - layer.base
        sequences[:, :-1] = parent_sequences[parent_local]
    sequences[:, -1] = rep_action
    next_dir_staging = scratch / "layers" / "depth_{:03d}.staging".format(layer.depth + 1)
    if next_dir_staging.exists():
        shutil.rmtree(next_dir_staging)
    _write_layer_nodes(
        next_dir_staging,
        rep_parent=rep_parent,
        rep_action=rep_action,
        sequences=sequences,
    )
    final_next = _layer_dir(scratch, layer.depth + 1)
    if final_next.exists():
        shutil.rmtree(final_next)
    os.replace(str(next_dir_staging), str(final_next))
    _write_layer_edges(
        current_dir,
        parent_base=layer.base,
        parent_count=layer.count,
        child_base=next_base,
        child_count=total_nodes,
        edges=edges,
    )
    return (
        _LayerInfo(layer.depth + 1, next_base, total_nodes, False, 0),
        edge_count,
        reports,
    )


def _manifest_layers(manifest: dict[str, Any]) -> list[_LayerInfo]:
    return [_LayerInfo(**item) for item in manifest.get("layers", [])]


def validate_layered_checkpoint(scratch_dir: str | Path) -> dict[str, int]:
    """Validate immutable layer boundaries and forward/reverse edge chunks."""

    scratch = Path(scratch_dir).resolve()
    manifest = _load_manifest(scratch)
    if manifest.get("format") != LAYERED_FORMAT:
        raise ValueError("unsupported layered checkpoint format")
    layers = _manifest_layers(manifest)
    if not layers or layers[0] != _LayerInfo(0, 0, 1, layers[0].expanded, layers[0].edge_count):
        raise RuntimeError("layered checkpoint root layer is invalid")
    next_base = 0
    next_physical = 0
    total_edges = 0
    terminal_count = 0
    for index, layer in enumerate(layers):
        if layer.depth != index or layer.base != next_base or layer.count <= 0:
            raise RuntimeError("layered checkpoint has noncontiguous layers")
        directory = _layer_dir(scratch, layer.depth)
        rep_parent = np.load(directory / "node_rep_parent.npy", mmap_mode="r")
        rep_action = np.load(directory / "node_rep_action.npy", mmap_mode="r")
        sequences = np.load(directory / "node_sequences.npy", mmap_mode="r")
        terminal_local = np.load(directory / "terminal_local.npy", mmap_mode="r")
        terminal_ids = np.load(directory / "terminal_physical_id.npy", mmap_mode="r")
        terminal_digest = np.load(directory / "terminal_digest.npy", mmap_mode="r")
        if rep_parent.shape != (layer.count,) or rep_action.shape != (layer.count,):
            raise RuntimeError("layered representative arrays have invalid shape")
        if sequences.shape != (layer.count, layer.depth):
            raise RuntimeError("layered representative sequence array has invalid shape")
        if terminal_ids.shape != (len(terminal_local),) or terminal_digest.shape != (len(terminal_local), 32):
            raise RuntimeError("layered terminal arrays have invalid shape")
        if len(terminal_local) and (
            np.any(terminal_local >= layer.count)
            or np.any(terminal_local[1:] <= terminal_local[:-1])
        ):
            raise RuntimeError("layered terminal indices are invalid")
        if len(terminal_ids) and not np.array_equal(
            terminal_ids,
            np.arange(
                next_physical,
                next_physical + len(terminal_ids),
                dtype=np.uint64,
            ),
        ):
            raise RuntimeError("layered physical IDs are not contiguous")
        next_physical += len(terminal_ids)
        if layer.depth == 0:
            if int(rep_parent[0]) != UINT64_MAX_INT or int(rep_action[0]) != UINT32_MAX_INT:
                raise RuntimeError("layered root representative is invalid")
        elif np.any(rep_parent < layers[layer.depth - 1].base) or np.any(
            rep_parent >= layer.base
        ):
            raise RuntimeError("layered representative parent is outside prior layer")
        terminal_count += len(terminal_local)
        if layer.expanded and index + 1 < len(layers):
            offsets = np.load(directory / "edge_offsets.npy", mmap_mode="r")
            children = np.load(directory / "edge_child.npy", mmap_mode="r")
            actions = np.load(directory / "edge_action.npy", mmap_mode="r")
            reverse_offsets = np.load(directory / "reverse_offsets.npy", mmap_mode="r")
            reverse_parent = np.load(directory / "reverse_parent.npy", mmap_mode="r")
            child_layer = layers[index + 1]
            if offsets.shape != (layer.count + 1,) or reverse_offsets.shape != (child_layer.count + 1,):
                raise RuntimeError("layered edge offsets have invalid shape")
            if len(children) != len(actions) or len(children) != len(reverse_parent):
                raise RuntimeError("layered forward/reverse edge counts disagree")
            if int(offsets[-1]) != len(children) or int(reverse_offsets[-1]) != len(children):
                raise RuntimeError("layered edge offsets have invalid extent")
            if len(children) and (
                np.any(children < child_layer.base)
                or np.any(children >= child_layer.base + child_layer.count)
                or np.any(reverse_parent < layer.base)
                or np.any(reverse_parent >= layer.base + layer.count)
            ):
                raise RuntimeError("layered edge endpoint is outside adjacent layers")
            total_edges += len(children)
        next_base += layer.count
    if next_base != int(manifest["next_node_id"]):
        raise RuntimeError("layered manifest node extent is stale")
    if terminal_count != int(manifest["next_physical_id"]):
        raise RuntimeError("layered manifest physical extent is stale")
    return {"layers": len(layers), "nodes": next_base, "edges": total_edges, "physical": terminal_count}


def _initialize_fresh(
    scratch: Path,
    fingerprint: dict[str, Any],
    *,
    shard_count: int,
) -> dict[str, Any]:
    scratch.mkdir(parents=True, exist_ok=False)
    root = _layer_dir(scratch, 0)
    _write_layer_nodes(
        root,
        rep_parent=np.asarray([UINT64_MAX_INT], dtype=np.uint64),
        rep_action=np.asarray([UINT32_MAX_INT], dtype=np.uint32),
        sequences=np.empty((1, 0), dtype=np.uint32),
    )
    _save_actions(scratch, [])
    manifest = {
        "format": LAYERED_FORMAT,
        "status": "building",
        "semantic_fingerprint": fingerprint,
        "packed_signature_version": PACKED_SIGNATURE_VERSION,
        "completed_depth": -1,
        "next_node_id": 1,
        "next_physical_id": 0,
        "generated_transition_count": 0,
        "layers": [
            {"depth": 0, "base": 0, "count": 1, "expanded": False, "edge_count": 0}
        ],
        "shard_count": int(shard_count),
        "started_unix": time.time(),
    }
    _write_manifest(scratch, manifest)
    return manifest




def _compile_layered_artifact(
    *,
    output: Path,
    scratch: Path,
    assets: Sequence[Any],
    config: Any,
    fingerprint: dict[str, Any],
    manifest: dict[str, Any],
    started: float,
) -> Any:
    from .graph import (
        ACTION_SCHEMA_VERSION,
        FORMAT_VERSION,
        PARTIAL_SIGNATURE_VERSION,
        PHYSICAL_SIGNATURE_VERSION,
        ActionInterner,
        StructuralDAG,
        StructuralDAGBuildResult,
        _git_commit,
        _save_array,
    )

    layers = _manifest_layers(manifest)
    raw_nodes = int(manifest["next_node_id"])
    productive_path = scratch / "productive.uint8"
    productive = np.memmap(productive_path, mode="w+", dtype=np.uint8, shape=(raw_nodes,))
    productive[:] = 0
    physical_count = int(manifest["next_physical_id"])
    for layer in layers:
        directory = _layer_dir(scratch, layer.depth)
        terminal_local = np.load(directory / "terminal_local.npy", mmap_mode="r")
        productive[layer.base + terminal_local] = 1
    for layer in reversed(layers[:-1]):
        directory = _layer_dir(scratch, layer.depth)
        offsets = np.load(directory / "edge_offsets.npy", mmap_mode="r")
        children = np.load(directory / "edge_child.npy", mmap_mode="r")
        for local_begin in range(0, layer.count, 1_000_000):
            local_end = min(layer.count, local_begin + 1_000_000)
            edge_begin = int(offsets[local_begin])
            edge_end = int(offsets[local_end])
            live_edges = productive[children[edge_begin:edge_end]] != 0
            cumulative = np.empty(len(live_edges) + 1, dtype=np.uint64)
            cumulative[0] = 0
            np.cumsum(live_edges, out=cumulative[1:])
            relative = offsets[local_begin : local_end + 1] - edge_begin
            live_counts = cumulative[relative[1:]] - cumulative[relative[:-1]]
            productive[
                layer.base + local_begin : layer.base + local_end
            ] |= live_counts != 0
    productive.flush()
    if not bool(productive[0]):
        raise ValueError("layered static DAG contains no valid terminal")
    old_to_new = np.memmap(
        scratch / "old_to_new.uint64", mode="w+", dtype=np.uint64, shape=(raw_nodes,)
    )
    old_to_new[:] = UINT64_MAX_INT
    next_id = 0
    chunk = 1_000_000
    for begin in range(0, raw_nodes, chunk):
        end = min(raw_nodes, begin + chunk)
        live = np.flatnonzero(productive[begin:end]) + begin
        old_to_new[live] = np.arange(next_id, next_id + len(live), dtype=np.uint64)
        next_id += len(live)
    node_count = int(next_id)
    edge_count = 0
    for layer in layers[:-1]:
        directory = _layer_dir(scratch, layer.depth)
        children = np.load(directory / "edge_child.npy", mmap_mode="r")
        offsets = np.load(directory / "edge_offsets.npy", mmap_mode="r")
        for local_begin in range(0, layer.count, 1_000_000):
            local_end = min(layer.count, local_begin + 1_000_000)
            counts = np.diff(offsets[local_begin : local_end + 1]).astype(np.int64)
            parent_live = np.repeat(
                productive[layer.base + local_begin : layer.base + local_end] != 0,
                counts,
            )
            edge_begin = int(offsets[local_begin])
            edge_end = int(offsets[local_end])
            edge_count += int(
                np.count_nonzero(
                    parent_live & (productive[children[edge_begin:edge_end]] != 0)
                )
            )

    staging = scratch / "artifact.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    first_edge = np.lib.format.open_memmap(staging / "node_first_edge.npy", mode="w+", dtype=np.uint64, shape=(node_count,))
    out_count = np.lib.format.open_memmap(staging / "node_edge_count.npy", mode="w+", dtype=np.uint32, shape=(node_count,))
    physical_out = np.lib.format.open_memmap(staging / "node_physical_id.npy", mode="w+", dtype=np.uint64, shape=(node_count,))
    rep_parent_out = np.lib.format.open_memmap(staging / "node_rep_parent.npy", mode="w+", dtype=np.uint64, shape=(node_count,))
    rep_action_out = np.lib.format.open_memmap(staging / "node_rep_action_id.npy", mode="w+", dtype=np.uint32, shape=(node_count,))
    edge_child_out = np.lib.format.open_memmap(staging / "edge_child.npy", mode="w+", dtype=np.uint64, shape=(edge_count,))
    edge_action_out = np.lib.format.open_memmap(staging / "edge_action_id.npy", mode="w+", dtype=np.uint32, shape=(edge_count,))
    first_edge[:] = 0
    out_count[:] = 0
    physical_out[:] = UINT64_MAX_INT
    rep_parent_out[:] = UINT64_MAX_INT
    rep_action_out[:] = UINT32_MAX_INT
    physical_rep = np.full(physical_count, UINT64_MAX_INT, dtype=np.uint64)
    physical_digest = np.zeros((physical_count, 32), dtype=np.uint8)
    for layer in layers:
        directory = _layer_dir(scratch, layer.depth)
        reps = np.load(directory / "node_rep_parent.npy", mmap_mode="r")
        actions = np.load(directory / "node_rep_action.npy", mmap_mode="r")
        terminal_local = np.load(directory / "terminal_local.npy", mmap_mode="r")
        terminal_ids = np.load(directory / "terminal_physical_id.npy", mmap_mode="r")
        terminal_digests = np.load(directory / "terminal_digest.npy", mmap_mode="r")
        for local_begin in range(0, layer.count, 1_000_000):
            local_end = min(layer.count, local_begin + 1_000_000)
            live_local = np.flatnonzero(
                productive[
                    layer.base + local_begin : layer.base + local_end
                ]
            ) + local_begin
            if not len(live_local):
                continue
            old_ids = layer.base + live_local
            new_ids = old_to_new[old_ids]
            live_reps = reps[live_local]
            has_rep = live_reps != UINT64_MAX_INT
            if np.any(has_rep):
                mapped_parent = old_to_new[live_reps[has_rep]]
                if np.any(mapped_parent == UINT64_MAX_INT):
                    raise RuntimeError(
                        "productive layered node has unproductive representative"
                    )
                rep_parent_out[new_ids[has_rep]] = mapped_parent
                rep_action_out[new_ids[has_rep]] = actions[live_local[has_rep]]
        if len(terminal_local):
            terminal_old_nodes = layer.base + terminal_local
            terminal_nodes = old_to_new[terminal_old_nodes]
            if np.any(terminal_nodes == UINT64_MAX_INT):
                raise RuntimeError("layered terminal was removed as unproductive")
            physical_out[terminal_nodes] = terminal_ids
            physical_rep[terminal_ids] = terminal_nodes
            physical_digest[terminal_ids, :] = terminal_digests
    edge_cursor = 0
    for layer in layers[:-1]:
        directory = _layer_dir(scratch, layer.depth)
        offsets = np.load(directory / "edge_offsets.npy", mmap_mode="r")
        children = np.load(directory / "edge_child.npy", mmap_mode="r")
        edge_actions = np.load(directory / "edge_action.npy", mmap_mode="r")
        for local_begin in range(0, layer.count, 1_000_000):
            local_end = min(layer.count, local_begin + 1_000_000)
            edge_begin = int(offsets[local_begin])
            edge_end = int(offsets[local_end])
            counts = np.diff(offsets[local_begin : local_end + 1]).astype(np.int64)
            old_parents = np.repeat(
                np.arange(
                    layer.base + local_begin,
                    layer.base + local_end,
                    dtype=np.uint64,
                ),
                counts,
            )
            old_children = children[edge_begin:edge_end]
            selected = (productive[old_parents] != 0) & (
                productive[old_children] != 0
            )
            mapped_parents = old_to_new[old_parents[selected]]
            mapped_children = old_to_new[old_children[selected]]
            size = len(mapped_children)
            edge_child_out[edge_cursor : edge_cursor + size] = mapped_children
            edge_action_out[edge_cursor : edge_cursor + size] = edge_actions[
                edge_begin:edge_end
            ][selected]
            if size:
                unique_parent, parent_counts = np.unique(
                    mapped_parents, return_counts=True
                )
                out_count[unique_parent] = parent_counts.astype(np.uint32)
            edge_cursor += size
    if edge_cursor != edge_count:
        raise RuntimeError("layered productive edge count changed")
    if node_count:
        first_edge[0] = 0
        np.cumsum(out_count[:-1], out=first_edge[1:], dtype=np.uint64)

    parent_offsets = np.lib.format.open_memmap(staging / "node_parent_offsets.npy", mode="w+", dtype=np.uint64, shape=(node_count + 1,))
    parent_offsets[:] = 0
    parent_nodes = np.lib.format.open_memmap(staging / "node_parent_nodes.npy", mode="w+", dtype=np.uint64, shape=(edge_count,))
    parent_cursor = 0
    for layer in layers[1:]:
        previous = _layer_dir(scratch, layer.depth - 1)
        reverse_offsets = np.load(previous / "reverse_offsets.npy", mmap_mode="r")
        reverse_parent = np.load(previous / "reverse_parent.npy", mmap_mode="r")
        for local_begin in range(0, layer.count, 1_000_000):
            local_end = min(layer.count, local_begin + 1_000_000)
            edge_begin = int(reverse_offsets[local_begin])
            edge_end = int(reverse_offsets[local_end])
            counts = np.diff(
                reverse_offsets[local_begin : local_end + 1]
            ).astype(np.int64)
            old_children = np.repeat(
                np.arange(
                    layer.base + local_begin,
                    layer.base + local_end,
                    dtype=np.uint64,
                ),
                counts,
            )
            old_parents = reverse_parent[edge_begin:edge_end]
            selected = (productive[old_children] != 0) & (
                productive[old_parents] != 0
            )
            mapped_children = old_to_new[old_children[selected]]
            mapped_parents = old_to_new[old_parents[selected]]
            size = len(mapped_parents)
            parent_nodes[parent_cursor : parent_cursor + size] = mapped_parents
            if size:
                unique_child, child_counts = np.unique(
                    mapped_children, return_counts=True
                )
                parent_offsets[unique_child + 1] = child_counts.astype(np.uint64)
            parent_cursor += size
    np.cumsum(parent_offsets, out=parent_offsets)
    if parent_cursor != edge_count:
        raise RuntimeError("layered reverse edge count changed")

    action_list = _load_actions(scratch)
    asset_ids = [asset.asset_id for asset in assets]
    _save_array(staging / "actions.npy", ActionInterner(action_list).to_array(asset_ids))
    _save_array(staging / "physical_rep_node.npy", physical_rep)
    _save_array(staging / "physical_signature_sha256.npy", physical_digest)
    (staging / "asset_ids.json").write_text(json.dumps(asset_ids, indent=2) + "\n", encoding="utf-8")
    multi_parent = np.diff(parent_offsets) > 1
    metadata = {
        "format": FORMAT_VERSION,
        "root_node_id": 0,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "physical_signature_version": PHYSICAL_SIGNATURE_VERSION,
        "physical_signature_eps": float(config.physical_signature_eps),
        "partial_state_signature_version": PARTIAL_SIGNATURE_VERSION,
        "builder_backend": "external-layered-v1",
        "packed_signature_version": PACKED_SIGNATURE_VERSION,
        "builder_git_commit": _git_commit(),
        **fingerprint,
        "root_rotation_options": list(config.root_rotation_options),
        "continuation_root_rotation_filter": manifest.get(
            "continuation_root_rotation_filter"
        ),
        "function_semantic_labels": list(config.function_semantic_labels),
        "max_links": int(config.max_total_links),
        "node_count": node_count,
        "edge_count": edge_count,
        "multi_parent_node_count": int(np.count_nonzero(multi_parent)),
        "multi_parent_nonterminal_count": int(np.count_nonzero(multi_parent & (physical_out == UINT64_MAX_INT))),
        "terminal_alias_count": physical_count,
        "terminal_count": physical_count,
        "physical_count": physical_count,
        "action_count": len(action_list),
        "generated_node_count": raw_nodes,
        "generated_transition_count": int(manifest["generated_transition_count"]),
        "merged_transition_count": int(manifest["generated_transition_count"]) - (raw_nodes - 1),
        "node_id_dtype": "uint64",
        "physical_id_dtype": "uint64",
        "action_id_dtype": "uint32",
    }
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for array in (first_edge, out_count, physical_out, rep_parent_out, rep_action_out, edge_child_out, edge_action_out, parent_offsets, parent_nodes):
        array.flush()
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError("refusing to overwrite non-empty DAG artifact {}".format(output))
        output.rmdir()
    os.replace(str(staging), str(output))
    tree = StructuralDAG.load(output, mmap=True)
    # Full set-based validation is prohibitive for billion-edge production DAGs.
    # The artifact is already atomically published; perform bounded structural
    # checks here and leave exhaustive validation as an explicit offline option.
    tree.validate_runtime()
    return StructuralDAGBuildResult(
        output_dir=str(output),
        node_count=tree.node_count,
        edge_count=tree.edge_count_total,
        terminal_alias_count=tree.terminal_alias_count,
        physical_count=tree.physical_count,
        action_count=len(tree.actions),
        generated_node_count=raw_nodes,
        generated_transition_count=int(manifest["generated_transition_count"]),
        merged_transition_count=int(manifest["generated_transition_count"]) - (raw_nodes - 1),
        elapsed_seconds=time.perf_counter() - started,
    )


def build_layered_dag(
    output_dir: str | Path,
    assets: Sequence[Any],
    config: Any,
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
    builder_continuation_root_rotation: Optional[str] = None,
) -> Any:
    """Compile a DAG by bulk-deduplicating complete grammar layers."""

    if int(builder_shards) != 256:
        raise ValueError("external-layered v1 requires 256 digest-prefix shards")
    if min(int(builder_workers), int(builder_job_parents), int(builder_inflight_multiplier), int(builder_reducer_workers)) < 1:
        raise ValueError("layered worker and batching controls must be positive")
    requested_rotation = (
        None
        if builder_continuation_root_rotation is None
        else str(builder_continuation_root_rotation).strip().lower()
    )
    if (
        requested_rotation is not None
        and requested_rotation not in config.root_rotation_options
    ):
        raise ValueError(
            "continuation root rotation {!r} is not enabled by the build "
            "configuration {}".format(
                requested_rotation, config.root_rotation_options
            )
        )
    from .graph import structural_dag_fingerprint

    started = time.perf_counter()
    asset_list = list(assets)
    config.validate()
    fingerprint = structural_dag_fingerprint(
        asset_list,
        config,
        initial_root_box=initial_root_box,
        initial_forbidden_boxes=initial_forbidden_boxes,
    )
    output = Path(output_dir).resolve()
    scratch = Path(scratch_dir or output.with_name(output.name + ".builder")).resolve()
    if resume:
        manifest = _load_manifest(scratch)
        if manifest.get("format") != LAYERED_FORMAT:
            raise ValueError("unsupported layered checkpoint format")
        if manifest.get("semantic_fingerprint") != fingerprint:
            raise ValueError("layered checkpoint fingerprint mismatch")
        validate_layered_checkpoint(scratch)
        working = scratch / "working"
        if working.exists():
            shutil.rmtree(working)
    else:
        if scratch.exists():
            raise FileExistsError("layered builder scratch already exists: {}".format(scratch))
        manifest = _initialize_fresh(scratch, fingerprint, shard_count=builder_shards)
    stored_filter = manifest.get("continuation_root_rotation_filter")
    if stored_filter is not None:
        stored_mode = str(stored_filter["mode"])
        if requested_rotation is not None and requested_rotation != stored_mode:
            raise ValueError(
                "layered checkpoint continuation root rotation is {!r}, "
                "not {!r}".format(stored_mode, requested_rotation)
            )
        requested_rotation = stored_mode
    elif requested_rotation is not None:
        manifest["continuation_root_rotation_filter"] = {
            "mode": requested_rotation,
            "start_depth": max(1, int(manifest["completed_depth"]) + 1),
        }
        _write_manifest(scratch, manifest)
    actions = _load_actions(scratch)

    while True:
        layers = _manifest_layers(manifest)
        depth = int(manifest["completed_depth"]) + 1
        layer = next((item for item in layers if item.depth == depth), None)
        if layer is None:
            break
        layer_started = time.perf_counter()
        continuation_filter = manifest.get("continuation_root_rotation_filter")
        continuation_mode = None
        if (
            continuation_filter is not None
            and layer.depth >= int(continuation_filter["start_depth"])
        ):
            continuation_mode = str(continuation_filter["mode"])
        working, physical_id, physical_digest, expansion_reports, proposals = _expand_layer(
            scratch=scratch,
            layer=layer,
            actions=actions,
            assets=asset_list,
            config=config,
            initial_forbidden_boxes=initial_forbidden_boxes,
            initial_root_box=initial_root_box,
            workers=builder_workers,
            job_parents=builder_job_parents,
            inflight_multiplier=builder_inflight_multiplier,
            shard_count=builder_shards,
            progress_interval=progress_interval,
            continuation_root_rotation=continuation_mode,
        )
        terminal_mask = physical_id != UINT64_MAX_INT
        terminal_count = int(np.count_nonzero(terminal_mask))
        if terminal_count:
            physical_id[terminal_mask] += int(manifest["next_physical_id"])
        next_layer, edge_count, reduction_reports = _reduce_layer(
            scratch=scratch,
            layer=layer,
            working=working,
            physical_id=physical_id,
            physical_digest=physical_digest,
            reducer_workers=builder_reducer_workers,
            shard_count=builder_shards,
        )
        existing = [dict(item) for item in manifest["layers"]]
        existing[layer.depth]["expanded"] = True
        existing[layer.depth]["edge_count"] = int(edge_count)
        if next_layer is not None:
            if len(existing) == next_layer.depth:
                existing.append(next_layer.__dict__)
            else:
                existing[next_layer.depth] = next_layer.__dict__
            manifest["next_node_id"] = next_layer.base + next_layer.count
        manifest["layers"] = existing
        manifest["completed_depth"] = layer.depth
        manifest["next_physical_id"] = int(manifest["next_physical_id"]) + terminal_count
        manifest["generated_transition_count"] = int(manifest["generated_transition_count"]) + int(proposals)
        manifest.setdefault("layer_diagnostics", []).append(
            {
                "depth": layer.depth,
                "parents": layer.count,
                "continuation_root_rotation": continuation_mode,
                "proposals": proposals,
                "unique_nodes": 0 if next_layer is None else next_layer.count,
                "edges": edge_count,
                "merged_proposals": proposals - (0 if next_layer is None else next_layer.count),
                "seconds": time.perf_counter() - layer_started,
                "expansion": expansion_reports,
                "reduction": reduction_reports,
            }
        )
        _write_manifest(scratch, manifest)
        shutil.rmtree(working)
        print(
            "[static-dag-layered] depth={} parents={} proposals={} unique={} edges={} seconds={:.3f}".format(
                layer.depth,
                layer.count,
                proposals,
                0 if next_layer is None else next_layer.count,
                edge_count,
                time.perf_counter() - layer_started,
            ),
            flush=True,
        )
        if next_layer is None:
            break
    manifest["status"] = "compiling"
    _write_manifest(scratch, manifest)
    result = _compile_layered_artifact(
        output=output,
        scratch=scratch,
        assets=asset_list,
        config=config,
        fingerprint=fingerprint,
        manifest=manifest,
        started=started,
    )
    manifest["status"] = "complete"
    manifest["artifact"] = str(output)
    _write_manifest(scratch, manifest)
    return result
