#!/usr/bin/env python3
"""Build and validate canonical structural DAGs."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from bilevel.upper.bass.config import BASSConfig
from bilevel.upper.bass.io_assets import load_assets, resolve_asset_id, select_assets
from bilevel.upper.bass.DAG.graph import (
    StructuralDAG,
    build_structural_dag,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve(path: str | Path, *, relative_to: Path | None = None) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    base = relative_to or REPO_ROOT
    return (base / value).resolve()


def _load_task(path: str | Path) -> tuple[Path, dict[str, Any]]:
    task_path = _resolve(path)
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task JSON root must be an object")
    return task_path, payload


def _function_labels(payload: dict[str, Any]) -> tuple[str, ...]:
    labels = []
    for function in payload.get("functions", []):
        if isinstance(function, dict):
            label = function.get("function_name", function.get("name"))
        else:
            label = function
        if label is not None:
            labels.append(str(label))
    return tuple(labels)


def _config(payload: dict[str, Any], args: argparse.Namespace) -> BASSConfig:
    raw = dict(payload.get("bass", {}))
    task_config = dict(payload.get("task_config", {}))
    allowed = {item.name for item in fields(BASSConfig)}
    kwargs = {key: value for key, value in raw.items() if key in allowed}
    kwargs["target_function_count"] = int(
        payload.get("function_count", len(_function_labels(payload)))
    )
    kwargs["function_count_margin"] = int(
        payload.get(
            "function_count_margin",
            raw.get("function_count_margin", 0),
        )
    )
    kwargs["function_group_depth_delta"] = payload.get(
        "function_group_depth_delta",
        task_config.get(
            "function_group_depth_delta",
            raw.get("function_group_depth_delta", 0),
        ),
    )
    kwargs["function_semantic_labels"] = _function_labels(payload)
    kwargs["root_asset_id"] = str(
        payload.get("root_asset_id", raw.get("root_asset_id", "root/universal_handle"))
    )
    kwargs["root_blocked_face"] = payload.get(
        "root_blocked_face", raw.get("root_blocked_face", 2)
    )
    if payload.get("max_head_links") is not None:
        kwargs["max_head_links"] = int(payload["max_head_links"])
    if args.max_head_links is not None:
        kwargs["max_head_links"] = int(args.max_head_links)
    # Tuple normalization is required because JSON naturally supplies lists.
    for key in ("root_rotation_options", "function_semantic_labels"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    # Artifact construction is independent of runtime scheduler selection.
    # Keep the offline compiler usable against search-only revisions that do
    # not yet expose the optional structural-dag runtime selector.
    if "structural_dag_path" in allowed:
        kwargs["structural_dag_path"] = None
    config = BASSConfig(**kwargs)
    config.validate()
    return config


def _assets(task_path: Path, payload: dict[str, Any], config: BASSConfig):
    task_config = dict(payload.get("task_config", {}))
    raw_path = payload.get("assets_json", task_config.get("assets_json"))
    if raw_path is None:
        raise ValueError("task JSON must define assets_json")
    assets_path = _resolve(raw_path, relative_to=REPO_ROOT)
    all_assets = load_assets(str(assets_path))
    config.root_asset_id = resolve_asset_id(all_assets, config.root_asset_id)
    selected = select_assets(
        all_assets,
        payload.get("asset_selector", task_config.get("asset_selector")),
        required_ids=(config.root_asset_id,),
    )
    if any(getattr(asset, "role", None) is not None for asset in selected):
        selected = [
            asset
            for asset in selected
            if asset.asset_id == config.root_asset_id or asset.searchable
        ]
    return selected


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--max-head-links", type=int, default=None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build a layered DAG with embedded root rotations")
    _common_parser(build)
    build.add_argument("--resume", action="store_true")
    build.add_argument("--scratch-dir")
    build.add_argument("--progress-interval", type=int, default=100_000)
    build.add_argument("--builder-workers", type=int, default=1)
    build.add_argument("--builder-job-parents", type=int, default=64)
    build.add_argument("--builder-inflight-multiplier", type=int, default=4)
    build.add_argument("--builder-reducer-workers", type=int, default=16)
    build.add_argument("--builder-shards", type=int, default=256)
    build.add_argument("--virtual-root-rotations", action="store_true",
                       help="Store root rotation choices virtually")
    validate = sub.add_parser("validate", help="Validate and replay a completed DAG")
    _common_parser(validate)
    validate.add_argument("--samples", type=int, default=10_000)
    validate.add_argument("--seed", type=int, default=0)
    validate.add_argument("--no-mmap", action="store_true")
    validate.add_argument("--exhaustive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    task_path, payload = _load_task(args.task_json)
    config = _config(payload, args)
    assets = _assets(task_path, payload, config)
    tree_path = _resolve(args.tree)
    if args.command == "build":
        result = build_structural_dag(
            tree_path, assets, config, resume=args.resume,
            scratch_dir=args.scratch_dir, progress_interval=args.progress_interval,
            builder_workers=args.builder_workers,
            builder_job_parents=args.builder_job_parents,
            builder_inflight_multiplier=args.builder_inflight_multiplier,
            builder_reducer_workers=args.builder_reducer_workers,
            builder_shards=args.builder_shards,
            virtual_root_rotations=args.virtual_root_rotations,
        )
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 0
    tree = StructuralDAG.load(tree_path, mmap=not args.no_mmap)
    if args.exhaustive:
        tree.validate()
    else:
        tree.validate_runtime(samples=args.samples)
    tree.replay_validate(assets, config, samples=args.samples, seed=args.seed)
    representation, modes = tree.rotation_spec(config.root_rotation_options)
    print(json.dumps({"ok": True, "tree": str(tree.root), "nodes": tree.node_count,
                      "physical": tree.physical_count, "rotation_representation": representation,
                      "rotation_modes": modes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
