#!/usr/bin/env python
"""Replay a Bilevel best_run.json result."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _configure_native_threads() -> None:
    numeric_threads = os.environ.get("BILEVEL_NUMERIC_THREADS", os.environ.get("BILEVEL_PARENT_NUMERIC_THREADS", "1"))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, numeric_threads)
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")


_configure_native_threads()


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve(path_text: str | None, default: str | None = None) -> Path:
    value = path_text or default
    if value is None:
        raise ValueError("missing required path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_migrated_path(path_text: str | None, default: str | None = None) -> Path:
    """Resolve paths from best_run.json, rebasing stale absolute paths.

    Bilevel runs are often moved between machines. Older best_run.json files may
    contain absolute paths from the source checkout. If that absolute path does
    not exist here, prefer the suffix starting at a known repo directory.
    """
    path = _resolve(path_text, default)
    if path.exists() or not path.is_absolute():
        return path

    parts = path.parts
    for marker in (
        "tasks",
        "workspace",
        "Bilevel",
        "bass",
        "assets",
    ):
        if marker not in parts:
            continue
        rebased = REPO_ROOT.joinpath(*parts[parts.index(marker):])
        if rebased.exists():
            return rebased
    return path


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _task_name_from_saved_task(
    task_json: dict,
    override: str | None = None,
) -> str:
    value = (
        override
        or task_json.get("mission_name")
        or task_json.get("task_name")
    )
    if not value:
        raise ValueError(
            "Cannot determine canonical task name from the saved result; "
            "pass --task explicitly"
        )
    return str(value)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a Bilevel best_run.json")
    parser.add_argument(
        "--best-run-json",
        default=(
            "workspace/bilevel/output/"
            "sweep_balls/best_run.json"
        ),
    )
    parser.add_argument("--run-key", default=None, help="Specific run key to replay (from replay_dir)")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory containing <run-key>.xml. Used when --run-key is set.",
    )
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="Replay directory containing <run-key>/ folders. Used when --run-key is set.",
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--phase", choices=["initial", "final"], default="final")
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override the RedMax replay camera position.",
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override the RedMax replay camera target.",
    )
    parser.add_argument(
        "--generic-design-protocol",
        choices=["connected_direct_planar_hexahedron"],
        default=None,
        help="Override the replayed generic_design protocol. Defaults to the saved low-level request.",
    )
    parser.add_argument("--no-render", action="store_true", help="Run replay setup/forward without opening the viewer")
    parser.add_argument(
        "--allow-simulator-mismatch",
        action="store_true",
        help=(
            "Explicitly allow replay with a different compiled RedMax binary. "
            "The replay is marked provenance-unverified."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    os.chdir(str(REPO_ROOT))
    args = parse_args(argv)

    from bilevel.runtime import visualize_xml
    from tasks import load_task

    best_run_path = _resolve(args.best_run_json)
    
    if args.run_key and not best_run_path.exists():
        payload = {}
    else:
        payload = json.loads(best_run_path.read_text(encoding="utf-8"))
    
    if args.run_key:
        cache_dir = _resolve_migrated_path(
            args.cache_dir
            or payload.get(
                "cache_dir",
                "workspace/bilevel/cache/sweep_balls",
            )
        )
        replay_dir = _resolve_migrated_path(
            args.replay_dir
            or payload.get(
                "replay_dir",
                "workspace/bilevel/output/"
                "sweep_balls/replay",
            )
        )
        
        run_result_path = replay_dir / args.run_key / "low_level_result.json"
        request_path = replay_dir / args.run_key / "low_level_request.json"
        cached_xml_path = cache_dir / f"{args.run_key}.xml"

        if cached_xml_path.exists():
            xml_path = cached_xml_path
        else:
            raise FileNotFoundError(
                f"Run {args.run_key} XML not found at {cached_xml_path}"
            )

        evaluation = _load_json_if_exists(run_result_path)
        low_level_request = _load_json_if_exists(request_path)
    else:
        cache_dir = _resolve_migrated_path(
            args.cache_dir
            or payload.get(
                "cache_dir",
                "workspace/bilevel/cache/sweep_balls",
            )
        )
        xml_path = _resolve_migrated_path(payload.get("best_xml"))
        if not xml_path.exists():
            evaluation = payload.get("evaluation", {})
            xml_path = _resolve_migrated_path(
                evaluation.get("xml_path") if isinstance(evaluation, dict) else None
            )
        evaluation = payload.get("evaluation", {})
        if not xml_path.exists():
            run_key = payload.get("run_key")
            cached_xml_path = cache_dir / f"{run_key}.xml" if run_key else None
            if cached_xml_path is not None and cached_xml_path.exists():
                xml_path = cached_xml_path
        if not xml_path.exists():
            raise FileNotFoundError(f"Best XML not found after migration-aware resolution: {xml_path}")
        run_key = payload.get("run_key")
        replay_dir = _resolve_migrated_path(
            payload.get(
                "replay_dir",
                "workspace/bilevel/output/"
                "sweep_balls/replay",
            )
        )
        request_path = replay_dir / str(run_key) / "low_level_request.json" if run_key else None
        low_level_request = _load_json_if_exists(request_path) if request_path is not None else {}

    request_context = {}
    if isinstance(low_level_request, dict):
        request_context = dict(low_level_request.get("context", {}) or {})
    request_task_json = dict(request_context.get("task_json", {}) or {})
    payload_task_json = {
        "mission_name": payload.get("mission_name"),
        "functions": payload.get("functions"),
    }
    task_json = request_task_json or payload_task_json

    # Fall back to persisted result metadata when old best_run files do not
    # include the original low-level request context.
    task_config = dict(task_json.get("task_config", {}) or {})
    if "num_steps" not in task_config and isinstance(evaluation, dict) and evaluation.get("num_steps") is not None:
        task_config["num_steps"] = int(evaluation["num_steps"])
    if "sub_steps" not in task_config and isinstance(evaluation, dict) and evaluation.get("sub_steps") is not None:
        task_config["sub_steps"] = int(evaluation["sub_steps"])
    if task_config:
        task_json["task_config"] = task_config
    if args.generic_design_protocol is not None:
        task_json.setdefault("task_config", {})
        task_json["task_config"]["generic_design_protocol"] = str(args.generic_design_protocol)

    task_name = _task_name_from_saved_task(task_json, args.task)
    task = load_task(
        task_name,
        config=dict(task_json.get("task_config", {}) or {}),
    )

    context = {
        "phase": args.phase,
        "visualize_step": int(args.step),
        "step": int(args.step),
        "best_run_json": str(best_run_path),
        "best_xml": str(xml_path),
        "run_key": args.run_key,
        "output_dir": request_context.get("output_dir", payload.get("output_dir")),
        "replay_dir": str(replay_dir) if 'replay_dir' in locals() else payload.get("replay_dir"),
        "cache_dir": str(cache_dir),
        "evaluation": evaluation,
        "render": not args.no_render,
        "allow_replay_simulator_mismatch": bool(
            args.allow_simulator_mismatch
        ),
        "task_json": task_json,
    }
    if args.camera_pos is not None:
        context["camera_pos"] = list(args.camera_pos)
    if args.camera_lookat is not None:
        context["camera_lookat"] = list(args.camera_lookat)
    for key in (
        "low_level_num_steps",
        "low_level_sub_steps",
        "low_level_grad_clip",
        "low_level_step_scale",
        "low_level_lr",
        "design_fd_grad",
        "design_fd_step",
        "design_fd_trigger_tol",
        "optimizer_strategy",
        "force_connectivity",
        "generic_design_protocol",
    ):
        if key in request_context:
            context[key] = request_context[key]
    if args.generic_design_protocol is not None:
        context["generic_design_protocol"] = str(args.generic_design_protocol)
        context["allow_replay_param_mismatch"] = True
    status = visualize_xml(task, str(xml_path), context)
    print(json.dumps(status, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
