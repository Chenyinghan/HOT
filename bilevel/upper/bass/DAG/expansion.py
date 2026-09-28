"""Pure grammar expansion workers shared by the layered DAG builder."""
from __future__ import annotations
import json
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple
from ..actions import AddLink, End, SelectRootRotation
from .canonical import pack_physical_signature, packed_signature_digest, physical_function_signature, physical_partial_state_signature, physical_signature_digest, prepare_physical_partial_signature_context


@dataclass(frozen=True)
class _BatchExpansionTarget:
    parent_id: int
    rep_sequence: bytes
    action_ids: Tuple[int, ...]


@dataclass(frozen=True)
class _ParentExpansionBatchJob:
    depth: int
    targets: Tuple[_BatchExpansionTarget, ...]
    action_catalog: Tuple[Tuple[int, Any], ...]


@dataclass(frozen=True)
class _ChildProposal:
    action_key: Tuple[Any, ...]
    digest: bytes
    canonical_key: bytes


@dataclass(frozen=True)
class _ParentExpansionResult:
    parent_id: int
    depth: int
    rep_sequence: bytes
    is_complete: bool
    valid_terminal: bool
    terminal_digest: Optional[bytes]
    children: Tuple[_ChildProposal, ...]
    replay_seconds: float
    generation_seconds: float
    valid_actions_seconds: float
    apply_action_seconds: float
    canonicalization_seconds: float
    signature_seconds: float
    signature_pack_seconds: float
    signature_compress_seconds: float


@dataclass(frozen=True)
class _ParentExpansionBatchResult:
    results: Tuple[_ParentExpansionResult, ...]
    replay_seconds: float
    generation_seconds: float
    naive_replay_transitions: int
    prefix_replay_transitions: int


_WORKER_ROOT_STATE: Any = None


_WORKER_ASSETS: Optional[list[Any]] = None


_WORKER_CONFIG: Any = None


_WORKER_SIGNATURE_CONTEXT: Any = None


def _action_key(action: Any) -> Tuple[Any, ...]:
    if isinstance(action, SelectRootRotation):
        return (0, action.mode)
    if isinstance(action, AddLink):
        return (
            1,
            action.asset_id,
            int(action.p),
            int(action.d),
            int(action.f),
            int(action.q),
            int(action.child_dock_id),
            bool(action.start_function_group),
        )
    if isinstance(action, End):
        return (2,)
    raise TypeError("unsupported static grammar action: {}".format(type(action).__name__))


def _action_from_key(key: Tuple[Any, ...]) -> Any:
    if key[0] == 0:
        return SelectRootRotation(str(key[1]))
    if key[0] == 1:
        return AddLink(
            asset_id=str(key[1]),
            p=int(key[2]),
            d=int(key[3]),
            f=int(key[4]),
            q=int(key[5]),
            child_dock_id=int(key[6]),
            start_function_group=bool(key[7]),
        )
    if key[0] == 2:
        return End()
    raise ValueError("unknown compact action kind {!r}".format(key[0]))


def _directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _stored_canonical_key(packed: bytes) -> bytes:
    """Compress exact descriptors before disk storage and comparison."""

    return zlib.compress(packed, level=1)


def _expand_state(
    *,
    parent_id: int,
    depth: int,
    rep_sequence: bytes,
    state: Any,
    assets: Sequence[Any],
    config: Any,
) -> _ParentExpansionResult:
    """Expand an already reconstructed parent state."""

    from ..state import apply_action
    from .graph import _configured_valid_actions, _valid_completed_state

    generation_started = time.perf_counter()
    valid_actions_seconds = 0.0
    apply_action_seconds = 0.0
    canonicalization_seconds = 0.0
    signature_seconds = 0.0
    signature_pack_seconds = 0.0
    signature_compress_seconds = 0.0
    terminal_digest: Optional[bytes] = None
    children = []
    valid_terminal = False
    if state.is_complete:
        valid_terminal = _valid_completed_state(state, config)
        if valid_terminal:
            terminal_signature = physical_function_signature(
                state,
                assets,
                eps=float(config.physical_signature_eps),
            )
            terminal_digest = bytes.fromhex(
                physical_signature_digest(terminal_signature)
            )
    else:
        actions_started = time.perf_counter()
        actions = _configured_valid_actions(state, assets, config)
        valid_actions_seconds += time.perf_counter() - actions_started
        for action in actions:
            try:
                apply_started = time.perf_counter()
                child_state = apply_action(state, action, assets)
                apply_action_seconds += time.perf_counter() - apply_started
            except ValueError:
                apply_action_seconds += time.perf_counter() - apply_started
                continue
            canonical_started = time.perf_counter()
            signature_started = canonical_started
            child_signature = physical_partial_state_signature(
                child_state,
                assets,
                eps=float(config.physical_signature_eps),
                signature_context=(
                    _WORKER_SIGNATURE_CONTEXT if assets is _WORKER_ASSETS else None
                ),
            )
            signature_seconds += time.perf_counter() - signature_started
            pack_started = time.perf_counter()
            packed = pack_physical_signature(child_signature)
            digest = packed_signature_digest(packed)
            signature_pack_seconds += time.perf_counter() - pack_started
            compress_started = time.perf_counter()
            canonical_key = _stored_canonical_key(packed)
            signature_compress_seconds += time.perf_counter() - compress_started
            children.append(
                _ChildProposal(
                    action_key=_action_key(action),
                    digest=digest,
                    canonical_key=canonical_key,
                )
            )
            canonicalization_seconds += time.perf_counter() - canonical_started
    return _ParentExpansionResult(
        parent_id=int(parent_id),
        depth=int(depth),
        rep_sequence=bytes(rep_sequence),
        is_complete=bool(state.is_complete),
        valid_terminal=bool(valid_terminal),
        terminal_digest=terminal_digest,
        children=tuple(children),
        replay_seconds=0.0,
        generation_seconds=time.perf_counter() - generation_started,
        valid_actions_seconds=valid_actions_seconds,
        apply_action_seconds=apply_action_seconds,
        canonicalization_seconds=canonicalization_seconds,
        signature_seconds=signature_seconds,
        signature_pack_seconds=signature_pack_seconds,
        signature_compress_seconds=signature_compress_seconds,
    )


def _expand_parent_batch_job(
    job: _ParentExpansionBatchJob,
    root_state: Any,
    assets: Sequence[Any],
    config: Any,
) -> _ParentExpansionBatchResult:
    """Replay a representative-prefix trie and expand every target leaf."""

    from ..state import apply_action

    action_by_id = dict(job.action_catalog)
    trie: dict[Any, Any] = {}
    naive_replay_transitions = 0
    for target in job.targets:
        naive_replay_transitions += len(target.action_ids)
        cursor = trie
        for action_id in target.action_ids:
            cursor = cursor.setdefault(action_id, {})
        cursor.setdefault(None, []).append(target)

    results = []
    replay_seconds = 0.0
    generation_seconds = 0.0
    prefix_replay_transitions = 0

    def visit(cursor: dict[Any, Any], state: Any) -> None:
        nonlocal replay_seconds, generation_seconds, prefix_replay_transitions
        for target in cursor.get(None, ()):
            result = _expand_state(
                parent_id=target.parent_id,
                depth=job.depth,
                rep_sequence=target.rep_sequence,
                state=state,
                assets=assets,
                config=config,
            )
            generation_seconds += result.generation_seconds
            results.append(result)
        for action_id, child_cursor in cursor.items():
            if action_id is None:
                continue
            replay_started = time.perf_counter()
            child_state = apply_action(state, action_by_id[action_id], assets)
            replay_seconds += time.perf_counter() - replay_started
            prefix_replay_transitions += 1
            visit(child_cursor, child_state)

    visit(trie, root_state)
    results.sort(key=lambda result: result.parent_id)
    return _ParentExpansionBatchResult(
        results=tuple(results),
        replay_seconds=replay_seconds,
        generation_seconds=generation_seconds,
        naive_replay_transitions=naive_replay_transitions,
        prefix_replay_transitions=prefix_replay_transitions,
    )


def _initialize_expansion_worker(
    assets: Sequence[Any],
    config: Any,
    initial_forbidden_boxes: Sequence[Any] | None,
    initial_root_box: Any,
) -> None:
    """Initialize process-local immutable grammar inputs."""

    from .graph import _initial_builder_state

    global _WORKER_ASSETS, _WORKER_CONFIG, _WORKER_ROOT_STATE
    global _WORKER_SIGNATURE_CONTEXT
    _WORKER_ASSETS = list(assets)
    _WORKER_CONFIG = config
    _WORKER_ROOT_STATE = _initial_builder_state(
        _WORKER_ASSETS,
        config,
        initial_forbidden_boxes,
        initial_root_box,
    )
    _WORKER_SIGNATURE_CONTEXT = prepare_physical_partial_signature_context(
        _WORKER_ASSETS,
        _WORKER_ROOT_STATE.forbidden_boxes,
        eps=float(config.physical_signature_eps),
    )


def _expand_parent_batch_worker(
    job: _ParentExpansionBatchJob,
) -> _ParentExpansionBatchResult:
    if _WORKER_ASSETS is None or _WORKER_ROOT_STATE is None:
        raise RuntimeError("static-DAG expansion worker was not initialized")
    return _expand_parent_batch_job(
        job,
        _WORKER_ROOT_STATE,
        _WORKER_ASSETS,
        _WORKER_CONFIG,
    )
