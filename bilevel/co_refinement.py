#!/usr/bin/env python3
"""Run action-only BASS, then co-refine its successful Top-K structures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from bilevel.upper.refine_successful import (
    _manifest_update,
    _positive_int,
    default_output_dir,
    load_task_settings,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_JSON = Path("tasks/sweep_balls/config.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _resolved_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def resolve_search_run_dir(
    task_json_path: Path,
    output_dir_override: Optional[Path],
) -> Path:
    """Mirror bilevel.search._default_output_paths without importing search."""

    task_json = _load_json(task_json_path)
    mission = task_json.get("mission_name", task_json.get("task_name"))
    if not mission or not str(mission).strip():
        raise ValueError("task config requires mission_name or task_name")
    paths = task_json.get("paths", {})
    configured_output = (
        paths.get("output_dir")
        if isinstance(paths, dict) and "output_dir" in paths
        else task_json.get("output_dir")
    )
    output_base = (
        output_dir_override
        if output_dir_override is not None
        else Path(configured_output or "workspace/bilevel/output")
    )
    return (_resolved_repo_path(output_base) / str(mission)).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing BASS search in action-only mode using the task "
            "config's Na, then co-refine and rerank its successful Top-K. "
            "Unrecognized arguments are forwarded to run_bilevel_search.py."
        )
    )
    parser.add_argument("--task-json", type=Path, default=DEFAULT_TASK_JSON)
    parser.add_argument("--top-k", required=True, type=_positive_int)
    parser.add_argument("--refinement-maxiter", required=True, type=_positive_int)
    parser.add_argument("--refine-workers", type=_positive_int, default=1)
    parser.add_argument(
        "--refinement-output",
        type=Path,
        help="Exact output directory for the pipeline/refinement artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Existing Stage 1 search output-base override.",
    )
    return parser


def _managed_search_args(search_args: list[str]) -> list[str]:
    forbidden = {"--design-optim", "--low-level-maxiter"}
    conflicts = sorted(
        {
            token.split("=", 1)[0]
            for token in search_args
            if token.split("=", 1)[0] in forbidden
        }
    )
    if conflicts:
        raise ValueError(
            "pipeline-managed search arguments cannot be overridden: "
            + ", ".join(conflicts)
        )
    return [
        token
        for token in search_args
        if token.split("=", 1)[0] != "--no-design-optim"
    ]


def build_search_command(
    *,
    task_json_path: Path,
    output_dir: Optional[Path],
    search_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "run_bilevel_search.py"),
        "--task-json",
        str(task_json_path),
    ]
    if output_dir is not None:
        command.extend(("--output-dir", str(output_dir)))
    command.extend(_managed_search_args(search_args))
    command.append("--no-design-optim")
    return command


def build_refinement_command(
    *,
    task_json_path: Path,
    search_run_dir: Path,
    output_dir: Path,
    top_k: int,
    refinement_maxiter: int,
    workers: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "bilevel.upper.refine_successful",
        "--task-json",
        str(task_json_path),
        "--search-run-dir",
        str(search_run_dir),
        "--top-k",
        str(top_k),
        "--refinement-maxiter",
        str(refinement_maxiter),
        "--workers",
        str(workers),
        "--output-dir",
        str(output_dir),
    ]


def run_pipeline(
    argv: Optional[list[str]] = None,
    *,
    run_subprocess: Optional[Callable[..., Any]] = None,
) -> tuple[int, Path]:
    args, forwarded = build_parser().parse_known_args(argv)
    task_json_path = _resolved_repo_path(args.task_json)
    settings = load_task_settings(task_json_path)
    search_run_dir = resolve_search_run_dir(task_json_path, args.output_dir)
    pipeline_output = (
        default_output_dir(settings.mission_name)
        if args.refinement_output is None
        else _resolved_repo_path(args.refinement_output)
    )
    pipeline_output.mkdir(parents=True, exist_ok=True)
    search_command = build_search_command(
        task_json_path=task_json_path,
        output_dir=args.output_dir,
        search_args=forwarded,
    )
    refinement_command = build_refinement_command(
        task_json_path=task_json_path,
        search_run_dir=search_run_dir,
        output_dir=pipeline_output,
        top_k=args.top_k,
        refinement_maxiter=args.refinement_maxiter,
        workers=args.refine_workers,
    )
    _manifest_update(
        pipeline_output,
        mission_name=settings.mission_name,
        task_json=str(task_json_path),
        action_maxiter=settings.action_maxiter,
        requested_k=int(args.top_k),
        refinement_maxiter=int(args.refinement_maxiter),
        refine_workers=int(args.refine_workers),
        search_run_dir=str(search_run_dir),
        search_command=search_command,
        refinement_command=refinement_command,
        pipeline_started_at=_utc_now(),
        status="searching",
    )
    runner = subprocess.run if run_subprocess is None else run_subprocess
    search_completed = runner(search_command, cwd=str(REPO_ROOT))
    search_returncode = int(search_completed.returncode)
    _manifest_update(
        pipeline_output,
        search_returncode=search_returncode,
        search_finished_at=_utc_now(),
    )
    if search_returncode != 0:
        _manifest_update(
            pipeline_output,
            status="search_failed",
            pipeline_finished_at=_utc_now(),
            exit_code=search_returncode,
        )
        return search_returncode, pipeline_output

    _manifest_update(pipeline_output, status="starting_refinement")
    refinement_completed = runner(refinement_command, cwd=str(REPO_ROOT))
    refinement_returncode = int(refinement_completed.returncode)
    # refine_successful owns the detailed final status.  Preserve it and only add
    # wrapper-level completion information here.
    _manifest_update(
        pipeline_output,
        refinement_returncode=refinement_returncode,
        pipeline_finished_at=_utc_now(),
        exit_code=refinement_returncode,
    )
    return refinement_returncode, pipeline_output


def main(argv: Optional[list[str]] = None) -> int:
    try:
        exit_code, output_dir = run_pipeline(argv)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[search-co_refinement] error: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("[search-co_refinement] interrupted", file=sys.stderr, flush=True)
        return 130
    print(f"[search-co_refinement] output={output_dir}", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
