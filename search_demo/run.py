#!/usr/bin/env python3
"""Build a DAG, calibrate it, or search with frozen calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bilevel.upper.bass.config import BASSConfig
from bilevel.upper.bass.bayesian_lookahead import load_calibration_artifact
from bilevel.upper.bass.DAG.build import _assets as task_assets, _config as task_dag_config
from bilevel.upper.bass.DAG.graph import (
    NODE_FILES,
    StructuralDAG,
    _grammar_payload,
    _sha256_json,
    build_structural_dag,
)
from bilevel.upper.bass.io_assets import load_assets, select_assets


TASKS = ("sweep_balls", "torque_bolt")
REPLAY_COLUMNS = (
    "eval_number", "run_key", "status", "score", "task_stage_count",
    "task_milestone", "task_progress", "task_success", "task_feasible",
    "bass_reward",
)
DEFAULT_DAG = ROOT / "search_demo" / "dags" / "demo_dag"
DEFAULT_LIBRARY = ROOT / "assets" / "library" / "catalog.json"
MODE_OPTIONS = {
    "build": {"--mode", "--dag", "--links", "--functions", "--asset-library"},
    "calibrate": {"--mode", "--dag", "--output", "--asset-library", "--task",
                  "--runs", "--workers", "--seed", "--dry-run"},
    "search": {"--mode", "--dag", "--output", "--asset-library", "--task",
               "--evaluations", "--calibration-artifact", "--workers", "--seed",
               "--dry-run"},
}


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _guided_budget(value: str) -> int:
    result = int(value)
    if result != -1 and result < 1:
        raise argparse.ArgumentTypeError("must be -1 or a positive integer")
    return result


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _dag_metadata_sha256(dag: Path) -> str:
    return hashlib.sha256((dag / "metadata.json").read_bytes()).hexdigest()


def _write_calibration_json(path: Path, artifact: dict) -> None:
    rows = artifact["demo_replay_rows"]
    metadata = {key: value for key, value in artifact.items()
                if key != "demo_replay_rows"}
    prefix = json.dumps(metadata, indent=2, sort_keys=True)[:-2]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(prefix + ',\n  "demo_replay_rows": [\n')
        for index, row in enumerate(rows):
            handle.write("    " + json.dumps(row, separators=(",", ":")))
            handle.write(",\n" if index + 1 < len(rows) else "\n")
        handle.write("  ]\n}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("build", "calibrate", "search"),
                        help="Build a DAG, calibrate, or run guided search")
    parser.add_argument("--dag", type=Path,
                        help="DAG output in build mode; input in calibration or search")
    parser.add_argument("--output", type=Path,
                        help="Fresh calibration or search run directory")
    parser.add_argument("--links", type=_positive, default=3,
                        help="Maximum Head links, excluding the handle (default: 3)")
    parser.add_argument("--functions", type=_positive, default=1,
                        help="Required functional groups (default: 1)")
    parser.add_argument("--asset-library", type=Path, default=DEFAULT_LIBRARY,
                        help="Asset catalog for the DAG and task (default: full shared catalog)")
    parser.add_argument("--task", choices=TASKS, default="sweep_balls",
                        help="Physical task for calibration or search (default: sweep_balls)")
    parser.add_argument("--runs", type=_positive, default=9800,
                        help="Physical calibration evaluations (default: 9800)")
    parser.add_argument("--evaluations", type=_guided_budget, default=-1,
                        help="Maximum guided evaluations; -1 runs until stopped or exhausted (default: -1)")
    parser.add_argument("--calibration-artifact", type=Path,
                        help="Frozen artifact for search (default: packaged task artifact)")
    parser.add_argument("--workers", type=_positive, default=4,
                        help="Concurrent physical evaluators (default: 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Calibration or search random seed (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the run configuration without evaluating")
    return parser


def _build(args: argparse.Namespace) -> None:
    path = _path(args.dag or ROOT / "workspace" / "search_demo" / "dags" /
                 f"{args.links}_links_{args.functions}_functions")
    staging = path.with_name(path.name + ".building")
    if path.exists() or staging.exists():
        raise FileExistsError(f"DAG path or unfinished build already exists: {path}")
    catalog = load_assets(str(_path(args.asset_library)))
    assets = select_assets(catalog, required_ids=("root/universal_handle",))
    assets = [asset for asset in assets
              if asset.asset_id == "root/universal_handle" or asset.searchable]
    config = BASSConfig(
        max_head_links=args.links,
        target_function_count=args.functions,
        function_count_margin=0,
        function_group_depth_delta=0,
        root_asset_id="root/universal_handle",
        root_blocked_face=2,
        root_rotation_options=("roll", "pitch", "yaw"),
        function_semantic_labels=(),
    )
    config.validate()
    result = build_structural_dag(staging, assets, config, builder_workers=1,
                                  builder_reducer_workers=1)

    # Labels are display names belonging to tasks. The DAG uses function
    # ordinals, allowing the same geometry to serve different task names.
    metadata_path = staging / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    grammar = _grammar_payload(config)
    grammar.pop("function_semantic_labels")
    metadata["grammar_config_sha256"] = _sha256_json(grammar)
    metadata["function_binding"] = "positional-v1"
    metadata.pop("function_semantic_labels", None)
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, metadata_path)
    tree = StructuralDAG.load(staging)
    tree.replay_validate(assets, config, samples=min(100, tree.physical_count))
    staging.rename(path)
    print(json.dumps({"dag": str(path), "nodes": result.node_count,
                      "edges": result.edge_count, "physical_terminals": result.physical_count},
                     indent=2))


def _task_payload(args: argparse.Namespace, dag: Path) -> tuple[dict, int]:
    if not (dag / "metadata.json").is_file():
        raise FileNotFoundError(f"DAG not found: {dag}")
    required_arrays = (*NODE_FILES.values(), "edge_child.npy", "edge_action_id.npy",
                       "node_parent_offsets.npy", "node_parent_nodes.npy", "actions.npy",
                       "physical_rep_node.npy", "physical_signature_sha256.npy")
    unavailable = []
    for name in required_arrays:
        path = dag / name
        if not path.is_file():
            unavailable.append(name)
            continue
        with path.open("rb") as handle:
            if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                unavailable.append(name)
    if unavailable:
        remedy = (
            'Run: git lfs pull --include="search_demo/dags/demo_dag/**" --exclude=""'
            if dag.resolve() == DEFAULT_DAG.resolve()
            else "Restore or rebuild this custom DAG's complete array files."
        )
        raise ValueError(
            f"DAG arrays are missing or still Git LFS pointers: {', '.join(unavailable)}. "
            f"{remedy}"
        )
    payload = json.loads((ROOT / "tasks" / args.task / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((dag / "metadata.json").read_text(encoding="utf-8"))
    links = int(metadata["max_links"]) - 1
    if links < 1:
        raise ValueError("DAG has no Head links")
    library = _path(args.asset_library)
    asset_ids = json.loads((dag / "asset_ids.json").read_text(encoding="utf-8"))
    head_ids = [asset_id for asset_id in asset_ids
                if asset_id != "root/universal_handle"]
    if not head_ids:
        raise ValueError("DAG has no searchable Head assets")
    selector = {"ids": head_ids, "searchable_only": True}
    payload["assets_json"] = str(library)
    payload["asset_selector"] = selector
    payload["task_config"]["assets_json"] = str(library)
    payload["task_config"]["asset_selector"] = selector
    config = task_dag_config(payload, SimpleNamespace(max_head_links=links))
    assets = task_assets(ROOT / "tasks" / args.task / "config.json", payload, config)
    StructuralDAG.load(dag).assert_compatible(assets, config)
    return payload, links


def _write_run(payload: dict, output: Path, *, dry_run: bool, label: str) -> None:
    if output.exists():
        raise FileExistsError(f"{label} output already exists: {output}")
    payload.update(cache_dir=str(output / "cache"), output_dir=str(output / "output"),
                   replay_dir=str(output / "replay"), best_xml_out=str(output / "best.xml"),
                   best_run_json=str(output / "best_run.json"))
    output.mkdir(parents=True)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, str(ROOT / "run_bilevel_search.py"),
               "--task-json", str(config_path), "--no-visualize-best"]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"{label} run: {output}", flush=True)
    print("Command: " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _calibrate(args: argparse.Namespace) -> None:
    dag = _path(args.dag or DEFAULT_DAG)
    output = _path(args.output or ROOT / "workspace" / "search_demo" / "calibration" / args.task)
    if output.exists():
        raise FileExistsError(f"Calibration output already exists: {output}")
    payload, links = _task_payload(args, dag)
    # The task config owns its physics and stage definitions. This run only
    # changes the DAG, asset catalog, budget, seed, and output locations.
    search = payload.setdefault("bass", {})
    search.update(structural_dag_path=str(dag), max_head_links=links, threads=1,
                  eval_workers=args.workers, iteration_budget=args.runs,
                  calibration_budget=args.runs, seed=args.seed,
                  acquisition="bass_n2", parallelization_mode="shared_tree",
                  scheduler_v2=True, partial_state_transpositions=True,
                  physical_dedup_mode="enforce", reward_mode="bounded_task")
    search.pop("calibration_artifact", None)
    payload["post_search_rerank"] = False
    _write_run(payload, output, dry_run=args.dry_run, label="Calibration")
    if not args.dry_run:
        artifact = output / "output" / args.task / "calibration.json"
        if not artifact.is_file():
            raise RuntimeError(f"Calibration finished without artifact: {artifact}")
        frozen = load_calibration_artifact(artifact, expected_task_name=args.task)
        frozen["demo_dag_metadata_sha256"] = _dag_metadata_sha256(dag)
        frozen["demo_calibration_seed"] = args.seed
        source = Path(frozen.pop("source_eval_csv"))
        frozen.pop("source_eval_csv_sha256", None)
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(frozen["warmup_total_outcomes"]):
            raise ValueError("calibration CSV row count disagrees with the fitted artifact")
        frozen["demo_replay_columns"] = list(REPLAY_COLUMNS)
        frozen["demo_replay_rows"] = [
            [row[column] for column in REPLAY_COLUMNS]
            for row in sorted(rows, key=lambda row: int(row["eval_number"]))
        ]
        temporary = artifact.with_suffix(".json.tmp")
        _write_calibration_json(temporary, frozen)
        os.replace(temporary, artifact)
        print(f"Calibration artifact: {artifact}")


def _search(args: argparse.Namespace) -> None:
    dag = _path(args.dag or DEFAULT_DAG)
    if dag != DEFAULT_DAG and args.calibration_artifact is None:
        raise ValueError("a custom DAG requires --calibration-artifact from its calibration run")
    output = _path(args.output or ROOT / "workspace" / "search_demo" / "search" / args.task)
    if output.exists():
        raise FileExistsError(f"Search output already exists: {output}")
    payload, links = _task_payload(args, dag)
    artifact_path = _path(args.calibration_artifact or
                          ROOT / "search_demo" / "calibration" / args.task / "calibration.json")
    artifact = load_calibration_artifact(artifact_path, expected_task_name=args.task)
    recorded_dag = artifact.get("demo_dag_metadata_sha256")
    if dag != DEFAULT_DAG and recorded_dag is None:
        raise ValueError("custom DAG calibration artifact lacks demo_dag_metadata_sha256")
    if recorded_dag is not None and recorded_dag != _dag_metadata_sha256(dag):
        raise ValueError("calibration artifact was created for a different DAG")
    rows = artifact.get("demo_replay_rows", [])
    if artifact.get("demo_replay_columns") != list(REPLAY_COLUMNS):
        raise ValueError("calibration artifact has incompatible replay columns")
    if len(rows) != int(artifact["warmup_total_outcomes"]):
        raise ValueError("calibration artifact replay count disagrees with the fitted model")
    if any(len(row) != len(REPLAY_COLUMNS) for row in rows):
        raise ValueError("calibration artifact has incomplete replay outcomes")
    keys = [row[1] for row in rows]
    if len(set(keys)) != len(rows) or [int(row[0]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("calibration artifact has duplicate or unordered replay outcomes")
    calibration_seed = int(artifact["demo_calibration_seed"])
    if "--seed" in sys.argv[1:] or any(token.startswith("--seed=") for token in sys.argv[1:]):
        if args.seed != calibration_seed:
            raise ValueError("search seed must equal the calibration seed for exact replay")
    replay_count = int(artifact["warmup_total_outcomes"])
    search = payload.setdefault("bass", {})
    search.update(structural_dag_path=str(dag), max_head_links=links,
                  threads=1, eval_workers=args.workers,
                  iteration_budget=(0 if args.evaluations == -1 else
                                    replay_count + args.evaluations),
                  calibration_budget=replay_count,
                  seed=calibration_seed, acquisition="bass_n2",
                  calibration_bins_per_stage=([] if artifact["bins_source"] == "auto"
                                              else artifact["bins_per_stage"]),
                  calibration_threshold_sample=artifact["threshold_sampling"],
                  prior_strengths=artifact["prior_strengths"],
                  parallelization_mode="shared_tree", scheduler_v2=True,
                  partial_state_transpositions=True, physical_dedup_mode="enforce",
                  reward_mode="bounded_task")
    search.pop("calibration_artifact", None)
    payload["demo_calibration_replay"] = {
        "calibration_artifact": str(artifact_path),
    }
    payload["post_search_rerank"] = False
    _write_run(payload, output, dry_run=args.dry_run, label="Search")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    supplied = {token.split("=", 1)[0] for token in sys.argv[1:]
                if token.startswith("--")}
    misplaced = sorted(supplied - MODE_OPTIONS[args.mode])
    if misplaced:
        parser.error("{} cannot be used with --mode {}".format(
            ", ".join(misplaced), args.mode))
    if args.mode == "build":
        _build(args)
    elif args.mode == "calibrate":
        _calibrate(args)
    else:
        _search(args)


if __name__ == "__main__":
    main()
