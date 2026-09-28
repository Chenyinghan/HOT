"""Publish a legacy single-function DAG without a task function label.

The output preserves graph/action/physical IDs. It recomputes every terminal
signature with the generic ``function:0`` ordinal, then validates replay.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .build import _assets, _config, _load_task
from .canonical import physical_function_signature, physical_signature_digest
from .graph import (
    FORMAT_VERSION,
    POSITIONAL_FUNCTION_BINDING,
    StructuralDAG,
    _grammar_payload,
    _initial_builder_state,
    _sha256_json,
    _valid_completed_state,
    structural_dag_fingerprint,
)
from ..state import apply_action


ARRAY_NAMES = (
    "actions.npy", "edge_action_id.npy", "edge_child.npy",
    "node_edge_count.npy", "node_first_edge.npy", "node_parent_nodes.npy",
    "node_parent_offsets.npy", "node_physical_id.npy",
    "node_rep_action_id.npy", "node_rep_parent.npy",
    "physical_rep_node.npy", "physical_signature_sha256.npy",
)

_WORKER_TREE = None
_WORKER_ASSETS = None
_WORKER_CONFIG = None


def _init_worker(path: str, assets, config) -> None:
    global _WORKER_TREE, _WORKER_ASSETS, _WORKER_CONFIG
    _WORKER_TREE = StructuralDAG.load(path)
    _WORKER_ASSETS = list(assets)
    _WORKER_CONFIG = config


def _digest_chunk(start: int, stop: int):
    tree = _WORKER_TREE
    assets = _WORKER_ASSETS
    config = _WORKER_CONFIG
    result = np.empty((stop - start, 32), dtype=np.uint8)
    for offset, physical_id in enumerate(range(start, stop)):
        node = int(tree.physical_rep_node[physical_id])
        state = _initial_builder_state(assets, config, None, None)
        for action in tree.sequence_for_node(node):
            state = apply_action(state, action, assets)
        if not _valid_completed_state(state, config):
            raise ValueError("invalid terminal at physical ID {}".format(physical_id))
        signature = physical_function_signature(
            state, assets, eps=config.physical_signature_eps
        )
        digest = bytes.fromhex(physical_signature_digest(signature))
        result[offset] = np.frombuffer(digest, dtype=np.uint8)
    return start, result


def publish(source: Path, output: Path, task_json: Path, max_head_links: int,
            workers: int = 1) -> None:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    task_path, payload = _load_task(task_json)
    config = _config(payload, SimpleNamespace(max_head_links=max_head_links))
    assets = _assets(task_path, payload, config)
    metadata = json.loads((source / "metadata.json").read_text())
    labels = metadata.get("function_semantic_labels")
    if not isinstance(labels, list) or len(labels) != 1:
        raise ValueError("source must contain exactly one function label")
    if int(metadata.get("physical_count", -1)) != int(metadata.get("terminal_count", -2)):
        raise ValueError("source terminal/physical counts disagree")
    expected = structural_dag_fingerprint(assets, config)
    for key in ("partial_state_signature_version", "asset_catalog_sha256",
                "root_geometry_sha256", "forbidden_geometry_sha256"):
        if metadata.get(key) != expected[key]:
            raise ValueError("source fingerprint mismatch: " + key)
    legacy_grammar = _grammar_payload(config)
    legacy_grammar["function_semantic_labels"] = labels
    legacy_grammar["root_rotation_options"] = list(metadata["root_rotation_options"])
    if _sha256_json(legacy_grammar) != metadata.get("grammar_config_sha256"):
        raise ValueError("source grammar fingerprint mismatch")

    staging.mkdir(parents=True)
    for name in ARRAY_NAMES + ("asset_ids.json",):
        shutil.copy2(source / name, staging / name)
    neutral_grammar = dict(legacy_grammar)
    neutral_grammar.pop("function_semantic_labels")
    metadata["format"] = FORMAT_VERSION
    metadata["function_binding"] = POSITIONAL_FUNCTION_BINDING
    metadata["grammar_config_sha256"] = _sha256_json(neutral_grammar)
    metadata.pop("function_semantic_labels")
    for key in ("builder_backend", "builder_git_commit", "canonical_key_storage"):
        metadata.pop(key, None)
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    tree = StructuralDAG.load(staging)
    tree.validate_runtime(samples=10_000)
    neutral_config = replace(config, function_semantic_labels=())
    temporary = staging / "physical_signature_sha256.npy.tmp"
    digests = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=(tree.physical_count, 32)
    )
    chunk_size = 10_000
    spans = [(start, min(start + chunk_size, tree.physical_count))
             for start in range(0, tree.physical_count, chunk_size)]
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(staging), assets, neutral_config),
    ) as pool:
        futures = [pool.submit(_digest_chunk, start, stop) for start, stop in spans]
        for future in as_completed(futures):
            start, chunk = future.result()
            digests[start:start + len(chunk)] = chunk
            completed += len(chunk)
            if completed % 100_000 < chunk_size or completed == tree.physical_count:
                digests.flush()
                print("neutral digests {}/{}".format(completed, tree.physical_count), flush=True)
    digests.flush()
    del digests, tree
    os.replace(temporary, staging / "physical_signature_sha256.npy")
    tree = StructuralDAG.load(staging)
    tree.validate_runtime(samples=10_000)
    tree.replay_validate(assets, config, samples=10_000)
    staging.rename(output)
    print("published {} nodes, {} edges, {} terminals at {}".format(
        tree.node_count, tree.edge_count_total, tree.physical_count, output
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--max-head-links", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    publish(args.source, args.output, args.task_json, args.max_head_links,
            args.workers)


if __name__ == "__main__":
    main()
