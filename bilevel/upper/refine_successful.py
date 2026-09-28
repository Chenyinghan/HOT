#!/usr/bin/env python3
"""Promote successful action-only BASS candidates into shape-action co-refinement.

This module deliberately sits outside the BASS implementation.  It consumes
persisted search artifacts, snapshots the selected structure/action pairs, and
invokes ``bilevel.lower.refine_structure`` once per selected candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from bilevel.lower.action_artifacts import ActionArtifact, load_action_artifact


REPO_ROOT = Path(__file__).resolve().parents[2]
REFINEMENT_SCHEMES = {
    "success_constrained_target_shell": "success-constrained-target-shell",
}


@dataclass(frozen=True)
class TaskSettings:
    task_json_path: Path
    mission_name: str
    action_maxiter: int
    refinement_optimizer: str
    scheme: str


@dataclass(frozen=True)
class Candidate:
    stage1_rank: int
    eval_number: int
    run_key: str
    xml_digest: str
    score: float
    low_level_score: float
    asset_volume_penalty: float
    task_success: bool
    xml_path: Path
    rollout_dir: Path
    diagnostics: Mapping[str, Any]
    low_level_result: Mapping[str, Any]
    params_metadata: Mapping[str, Any]
    action: ActionArtifact
    row: Mapping[str, str]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest_update(output_dir: Path, **updates: Any) -> dict[str, Any]:
    path = output_dir / "pipeline_manifest.json"
    payload: dict[str, Any] = {}
    if path.is_file():
        payload = _load_json(path)
    payload.setdefault("schema", "action_then_coopt_pipeline_v1")
    payload.update(updates)
    _atomic_write_json(path, payload)
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def load_task_settings(task_json_path: Path) -> TaskSettings:
    path = task_json_path.expanduser().resolve()
    payload = _load_json(path)
    mission = payload.get("mission_name", payload.get("task_name"))
    if not mission or not str(mission).strip():
        raise ValueError("task config requires mission_name or task_name")
    task_config = payload.get("task_config", {})
    if not isinstance(task_config, dict):
        raise ValueError("task_config must be a JSON object")
    maxiter = payload.get(
        "low_level_maxiter",
        task_config.get("low_level_maxiter"),
    )
    if maxiter is None or int(maxiter) <= 0:
        raise ValueError("task config requires positive low_level_maxiter")
    optimizer_policy = task_config.get("optimizer", {})
    if not isinstance(optimizer_policy, dict):
        raise ValueError("task_config.optimizer must be a JSON object")
    refinement_optimizer = str(optimizer_policy.get("co_refinement", "")).strip()
    scheme = REFINEMENT_SCHEMES.get(refinement_optimizer)
    if scheme is None:
        raise ValueError(
            "unsupported task co_refinement optimizer "
            f"{refinement_optimizer!r}; expected one of {sorted(REFINEMENT_SCHEMES)}"
        )
    return TaskSettings(
        task_json_path=path,
        mission_name=str(mission),
        action_maxiter=int(maxiter),
        refinement_optimizer=refinement_optimizer,
        scheme=scheme,
    )


def default_output_dir(mission_name: str) -> Path:
    return (
        REPO_ROOT
        / "workspace"
        / "bilevel"
        / "refinement"
        / mission_name
        / _run_id()
    )


def _resolve_artifact(value: str, search_run_dir: Path) -> Path:
    candidate = Path(value).expanduser()
    choices = [candidate]
    if not candidate.is_absolute():
        choices.extend((REPO_ROOT / candidate, search_run_dir / candidate))
    for choice in choices:
        if choice.exists():
            return choice.resolve()
    return choices[0].resolve()


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    return _finite_float(value, name="numeric CSV field")


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_summary(candidate: Candidate) -> dict[str, Any]:
    return {
        "stage1_rank": candidate.stage1_rank,
        "eval_number": candidate.eval_number,
        "run_key": candidate.run_key,
        "xml_digest": candidate.xml_digest,
        "stage1_score": candidate.score,
        "stage1_low_level_score": candidate.low_level_score,
        "asset_volume_penalty": candidate.asset_volume_penalty,
        "stage1_task_success": candidate.task_success,
        "xml_path": str(candidate.xml_path),
        "rollout_dir": str(candidate.rollout_dir),
        "action_source": str(candidate.action.source),
        "action_dim": int(candidate.action.action.size),
        "action_parameterization": candidate.action.action_parameterization,
    }


def _rejection(row: Mapping[str, str], reason: str) -> dict[str, Any]:
    return {
        "eval_number": row.get("eval_number", ""),
        "run_key": row.get("run_key", ""),
        "xml_path": row.get("xml_path", ""),
        "reason": reason,
    }


def _validate_row(
    row: Mapping[str, str],
    *,
    search_run_dir: Path,
    settings: TaskSettings,
) -> Candidate:
    if str(row.get("status", "")).strip().lower() != "ok":
        raise ValueError("Stage 1 status is not ok")
    score = _finite_float(row.get("score"), name="Stage 1 score")
    eval_text = str(row.get("eval_number", "")).strip()
    eval_number = int(eval_text) if eval_text else sys.maxsize
    run_key = str(row.get("run_key", "")).strip()
    if not run_key:
        raise ValueError("missing run_key")
    xml_value = str(row.get("xml_path", "")).strip()
    rollout_value = str(row.get("rollout_dir", "")).strip()
    if not xml_value or not rollout_value:
        raise ValueError("missing XML or rollout path")
    xml_path = _resolve_artifact(xml_value, search_run_dir)
    rollout_dir = _resolve_artifact(rollout_value, search_run_dir)
    if not xml_path.is_file():
        raise ValueError(f"XML does not exist: {xml_path}")
    if not rollout_dir.is_dir():
        raise ValueError(f"rollout does not exist: {rollout_dir}")

    actual_digest = _sha1(xml_path)

    diagnostics_path = rollout_dir / "diagnostics.json"
    diagnostics = _load_json(diagnostics_path) if diagnostics_path.is_file() else {}
    task_success = _optional_bool(diagnostics.get("task_success"))
    if task_success is None:
        task_success = _optional_bool(row.get("task_success"))
    if task_success is not True:
        raise ValueError("task_success is not true")

    params_path = rollout_dir / "params.npy"
    metadata_path = rollout_dir / "params_meta.json"
    result_path = rollout_dir / "low_level_result.json"
    if not params_path.is_file() or not metadata_path.is_file():
        raise ValueError("rollout is missing params.npy or params_meta.json")
    if not result_path.is_file():
        raise ValueError("rollout is missing low_level_result.json")
    params_metadata = _load_json(metadata_path)
    try:
        morphology_dim = int(params_metadata["morphology_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("params metadata does not prove morphology_dim") from exc
    if morphology_dim != 0:
        raise ValueError(
            f"Stage 1 is not action-only: morphology_dim={morphology_dim}"
        )
    action = load_action_artifact(rollout_dir)
    action_slice = params_metadata.get("action_slice")
    if (
        not isinstance(action_slice, list)
        or len(action_slice) != 2
        or int(action_slice[0]) != 0
        or int(action_slice[1]) != int(action.action.size)
    ):
        raise ValueError("action artifact disagrees with params metadata")

    low_level_result = _load_json(result_path)
    actual_maxiter = low_level_result.get("optimize_maxiter")
    if actual_maxiter is None or int(actual_maxiter) != settings.action_maxiter:
        raise ValueError(
            "Stage 1 optimize_maxiter does not equal config Na: "
            f"actual={actual_maxiter!r} expected={settings.action_maxiter}"
        )
    asset_penalty = _optional_float(row.get("asset_volume_penalty"), 0.0)
    low_level_score = _optional_float(
        row.get("low_level_score"), score - asset_penalty
    )
    return Candidate(
        stage1_rank=0,
        eval_number=eval_number,
        run_key=run_key,
        xml_digest=actual_digest,
        score=score,
        low_level_score=low_level_score,
        asset_volume_penalty=asset_penalty,
        task_success=True,
        xml_path=xml_path,
        rollout_dir=rollout_dir,
        diagnostics=diagnostics,
        low_level_result=low_level_result,
        params_metadata=params_metadata,
        action=action,
        row=dict(row),
    )


def select_topk_candidates(
    *,
    search_run_dir: Path,
    settings: TaskSettings,
    top_k: int,
) -> tuple[list[Candidate], list[dict[str, Any]], list[Candidate]]:
    evals_path = search_run_dir / "evals.csv"
    if not evals_path.is_file():
        raise FileNotFoundError(f"search evals.csv does not exist: {evals_path}")
    valid: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    with evals_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                valid.append(
                    _validate_row(
                        row,
                        search_run_dir=search_run_dir,
                        settings=settings,
                    )
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                rejected.append(_rejection(row, str(exc)))

    valid.sort(key=lambda item: (item.score, item.eval_number, item.run_key))
    unique: list[Candidate] = []
    seen_digests: set[str] = set()
    for candidate in valid:
        if candidate.xml_digest in seen_digests:
            rejected.append(
                _rejection(candidate.row, "duplicate XML digest")
            )
            continue
        seen_digests.add(candidate.xml_digest)
        unique.append(candidate)

    ranked = [
        Candidate(
            **{
                **candidate.__dict__,
                "stage1_rank": index,
            }
        )
        for index, candidate in enumerate(unique, start=1)
    ]
    return ranked[:top_k], rejected, ranked


def _safe_token(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )
    return cleaned[:80] or "candidate"


def _snapshot_candidate(
    candidate: Candidate,
    candidate_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    candidate_dir.mkdir(parents=True, exist_ok=False)
    xml_snapshot = candidate_dir / "source_structure.xml"
    action_snapshot = candidate_dir / "stage1_action.npz"
    shutil.copy2(candidate.xml_path, xml_snapshot)
    np.savez(
        str(action_snapshot),
        action_params=np.asarray(candidate.action.action, dtype=np.float64),
        action_parameterization=np.asarray(
            candidate.action.action_parameterization
        ),
    )
    diagnostics_snapshot = candidate_dir / "stage1_diagnostics.json"
    _atomic_write_json(diagnostics_snapshot, dict(candidate.diagnostics))
    provenance = {
        **_candidate_summary(candidate),
        "snapshot_xml": str(xml_snapshot.resolve()),
        "snapshot_action": str(action_snapshot.resolve()),
        "snapshot_diagnostics": str(diagnostics_snapshot.resolve()),
        "params_metadata": dict(candidate.params_metadata),
        "eval_row": dict(candidate.row),
    }
    _atomic_write_json(candidate_dir / "stage1_candidate.json", provenance)
    return xml_snapshot.resolve(), action_snapshot.resolve(), provenance


def build_reoptimization_command(
    *,
    settings: TaskSettings,
    xml_snapshot: Path,
    action_snapshot: Path,
    solve_dir: Path,
    refinement_maxiter: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "bilevel.lower.refine_structure",
        "--task",
        settings.mission_name,
        "--structure",
        str(xml_snapshot),
        "--action-source",
        str(action_snapshot),
        "--scheme",
        settings.scheme,
        "--save-dir",
        str(solve_dir.resolve()),
        "--",
        "--max-iters",
        str(refinement_maxiter),
        "--no-visualize",
    ]


def _final_objective(
    solve_dir: Path,
) -> tuple[float, dict[str, Any], Optional[bool]]:
    params_path = solve_dir / "params.npy"
    metadata_path = solve_dir / "params_meta.json"
    logs_path = solve_dir / "logs.npy"
    diagnostics_path = solve_dir / "diagnostics.json"
    for required in (params_path, metadata_path, logs_path):
        if not required.is_file():
            raise ValueError(f"Stage 2 artifact is missing: {required}")
    metadata = _load_json(metadata_path)
    params = np.asarray(np.load(str(params_path), allow_pickle=False))
    if params.ndim != 1 or not np.all(np.isfinite(params)):
        raise ValueError("Stage 2 params.npy must be a finite vector")
    if int(metadata.get("total_dim", -1)) != int(params.size):
        raise ValueError("Stage 2 params dimension disagrees with metadata")
    if int(metadata.get("morphology_dim", 0)) <= 0:
        raise ValueError("Stage 2 did not create trainable morphology parameters")
    logs = np.asarray(np.load(str(logs_path), allow_pickle=False))
    if logs.ndim != 2 or logs.shape[0] == 0 or logs.shape[1] < 2:
        raise ValueError("Stage 2 logs.npy must be a non-empty Nx2 array")
    objective = _finite_float(logs[-1, 1], name="Stage 2 final objective")
    diagnostics = _load_json(diagnostics_path) if diagnostics_path.is_file() else {}
    optimizer = diagnostics.get("optimizer", {})
    details = optimizer.get("details", {}) if isinstance(optimizer, dict) else {}
    diagnostic_objective = (
        details.get("final_objective") if isinstance(details, dict) else None
    )
    if diagnostic_objective is not None:
        diagnostic_value = _finite_float(
            diagnostic_objective,
            name="diagnostic final objective",
        )
        if not math.isclose(
            objective,
            diagnostic_value,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "Stage 2 logs and diagnostics final objectives disagree: "
                f"{objective} != {diagnostic_value}"
            )
    return objective, diagnostics, _optional_bool(diagnostics.get("task_success"))


def refine_candidate(
    candidate: Candidate,
    *,
    settings: TaskSettings,
    candidates_root: Path,
    refinement_maxiter: int,
    run_subprocess: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    runner = subprocess.run if run_subprocess is None else run_subprocess
    candidate_dir = candidates_root / (
        f"rank_{candidate.stage1_rank:03d}_{_safe_token(candidate.run_key)}"
    )
    base_record = {
        **_candidate_summary(candidate),
        "refinement_optimizer": settings.refinement_optimizer,
        "scheme": settings.scheme,
        "refinement_maxiter": int(refinement_maxiter),
        "candidate_dir": str(candidate_dir.resolve()),
        "status": "pending",
        "error": None,
        "returncode": None,
        "stage2_low_level_score": None,
        "final_score": None,
        "stage2_task_success": None,
        "final_rank": None,
    }
    try:
        xml_snapshot, action_snapshot, provenance = _snapshot_candidate(
            candidate,
            candidate_dir,
        )
        solve_dir = candidate_dir / "solve"
        command = build_reoptimization_command(
            settings=settings,
            xml_snapshot=xml_snapshot,
            action_snapshot=action_snapshot,
            solve_dir=solve_dir,
            refinement_maxiter=refinement_maxiter,
        )
        base_record.update(
            {
                "snapshot_xml": str(xml_snapshot),
                "snapshot_action": str(action_snapshot),
                "solve_dir": str(solve_dir.resolve()),
                "command": command,
                "provenance": provenance,
            }
        )
        completed = runner(command, cwd=str(REPO_ROOT))
        returncode = int(completed.returncode)
        base_record["returncode"] = returncode
        if returncode != 0:
            raise RuntimeError(
                f"refine_structure exited with return code {returncode}"
            )
        objective, diagnostics, task_success = _final_objective(solve_dir)
        final_score = objective + candidate.asset_volume_penalty
        base_record.update(
            {
                "status": "ok",
                "stage2_low_level_score": objective,
                "final_score": final_score,
                "stage2_task_success": task_success,
                "params_path": str((solve_dir / "params.npy").resolve()),
                "params_meta_path": str(
                    (solve_dir / "params_meta.json").resolve()
                ),
                "logs_path": str((solve_dir / "logs.npy").resolve()),
                "diagnostics_path": (
                    str((solve_dir / "diagnostics.json").resolve())
                    if (solve_dir / "diagnostics.json").is_file()
                    else None
                ),
                "stage2_diagnostics": diagnostics,
            }
        )
    except Exception as exc:
        base_record["status"] = "failed"
        base_record["error"] = str(exc)
    _atomic_write_json(candidate_dir / "refinement_result.json", base_record)
    return base_record


RANKING_FIELDS = (
    "final_rank",
    "status",
    "stage1_rank",
    "eval_number",
    "run_key",
    "xml_digest",
    "stage1_score",
    "stage1_low_level_score",
    "asset_volume_penalty",
    "stage1_task_success",
    "xml_path",
    "rollout_dir",
    "refinement_optimizer",
    "refinement_maxiter",
    "returncode",
    "stage2_low_level_score",
    "final_score",
    "stage2_task_success",
    "snapshot_xml",
    "snapshot_action",
    "solve_dir",
    "params_path",
    "params_meta_path",
    "logs_path",
    "diagnostics_path",
    "candidate_dir",
    "error",
)


def _write_results(
    output_dir: Path,
    *,
    requested_k: int,
    selected_k: int,
    results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = list(results)
    rankable = [
        record
        for record in records
        if record.get("status") == "ok"
        and record.get("stage2_task_success") is True
        and record.get("final_score") is not None
        and math.isfinite(float(record["final_score"]))
    ]
    rankable.sort(
        key=lambda record: (
            float(record["final_score"]),
            int(record["stage1_rank"]),
            str(record["run_key"]),
        )
    )
    for rank, record in enumerate(rankable, start=1):
        record["final_rank"] = rank
        result_path = Path(str(record["candidate_dir"])) / "refinement_result.json"
        _atomic_write_json(result_path, record)
    records.sort(
        key=lambda record: (
            record.get("final_rank") is None,
            record.get("final_rank") or int(record["stage1_rank"]),
        )
    )
    payload = {
        "schema": "topk_co_refinement_results_v1",
        "requested_k": int(requested_k),
        "selected_k": int(selected_k),
        "ranked_k": len(rankable),
        "ranking_policy": ["final_score", "stage1_rank", "run_key"],
        "results": records,
    }
    _atomic_write_json(output_dir / "final_ranking.json", payload)
    with (output_dir / "final_ranking.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in RANKING_FIELDS})
    if rankable:
        best = dict(rankable[0])
        _atomic_write_json(
            output_dir / "best_refined_run.json",
            {
                "schema": "best_refined_run_v1",
                "final_rank": 1,
                "final_score": best["final_score"],
                "stage2_low_level_score": best["stage2_low_level_score"],
                "stage2_task_success": best.get("stage2_task_success"),
                "source_structure_xml": best["snapshot_xml"],
                "initial_action_artifact": best["snapshot_action"],
                "final_params": best["params_path"],
                "final_params_metadata": best["params_meta_path"],
                "diagnostics": best.get("diagnostics_path"),
                "candidate_result": str(
                    Path(best["candidate_dir"]) / "refinement_result.json"
                ),
                "run_key": best["run_key"],
                "xml_digest": best["xml_digest"],
            },
        )
    return rankable


def run_refinement(
    *,
    task_json_path: Path,
    search_run_dir: Path,
    top_k: int,
    refinement_maxiter: int,
    workers: int,
    output_dir: Optional[Path] = None,
    run_subprocess: Optional[Callable[..., Any]] = None,
) -> tuple[int, Path]:
    settings = load_task_settings(task_json_path)
    search_dir = search_run_dir.expanduser().resolve()
    destination = (
        default_output_dir(settings.mission_name)
        if output_dir is None
        else output_dir.expanduser().resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    _manifest_update(
        destination,
        mission_name=settings.mission_name,
        task_json=str(settings.task_json_path),
        search_run_dir=str(search_dir),
        action_maxiter=settings.action_maxiter,
        requested_k=int(top_k),
        refinement_maxiter=int(refinement_maxiter),
        refine_workers=int(workers),
        refinement_optimizer=settings.refinement_optimizer,
        scheme=settings.scheme,
        refinement_started_at=_utc_now(),
        status="selecting",
    )
    selected, rejected, all_eligible = select_topk_candidates(
        search_run_dir=search_dir,
        settings=settings,
        top_k=top_k,
    )
    _atomic_write_json(
        destination / "selection.json",
        {
            "schema": "topk_action_only_selection_v1",
            "requested_k": int(top_k),
            "selected_k": len(selected),
            "eligible_unique_k": len(all_eligible),
            "selection_policy": ["score", "eval_number", "run_key"],
            "deduplication_key": "xml_digest",
            "selected": [_candidate_summary(item) for item in selected],
            "eligible": [_candidate_summary(item) for item in all_eligible],
            "rejected": rejected,
        },
    )
    if not selected:
        _manifest_update(
            destination,
            status="failed",
            selected_k=0,
            ranked_k=0,
            refinement_finished_at=_utc_now(),
            error="no successful action-only candidates are eligible",
        )
        return 1, destination

    candidates_root = destination / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=False)
    _manifest_update(destination, status="refining", selected_k=len(selected))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(
                refine_candidate,
                candidate,
                settings=settings,
                candidates_root=candidates_root,
                refinement_maxiter=refinement_maxiter,
                run_subprocess=run_subprocess,
            ): candidate
            for candidate in selected
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # defensive: refine_candidate is fail-closed
                results.append(
                    {
                        **_candidate_summary(candidate),
                        "status": "failed",
                        "error": f"unexpected refinement failure: {exc}",
                        "final_rank": None,
                        "final_score": None,
                    }
                )

    rankable = _write_results(
        destination,
        requested_k=top_k,
        selected_k=len(selected),
        results=results,
    )
    if not rankable:
        status, exit_code = "failed", 1
    elif len(selected) != top_k or len(rankable) != top_k:
        status, exit_code = "partial", 2
    else:
        status, exit_code = "complete", 0
    _manifest_update(
        destination,
        status=status,
        selected_k=len(selected),
        ranked_k=len(rankable),
        refinement_finished_at=_utc_now(),
        final_ranking=str((destination / "final_ranking.json").resolve()),
        best_refined_run=(
            str((destination / "best_refined_run.json").resolve())
            if rankable
            else None
        ),
        exit_code=exit_code,
    )
    return exit_code, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select successful action-only BASS candidates and warm-start "
            "action/shape shape-action co-refinement for the Top-K unique structures."
        )
    )
    parser.add_argument("--task-json", required=True, type=Path)
    parser.add_argument("--search-run-dir", required=True, type=Path)
    parser.add_argument("--top-k", required=True, type=_positive_int)
    parser.add_argument("--refinement-maxiter", required=True, type=_positive_int)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    destination: Optional[Path] = args.output_dir
    try:
        if destination is None:
            settings = load_task_settings(args.task_json)
            destination = default_output_dir(settings.mission_name)
        exit_code, destination = run_refinement(
            task_json_path=args.task_json,
            search_run_dir=args.search_run_dir,
            top_k=args.top_k,
            refinement_maxiter=args.refinement_maxiter,
            workers=args.workers,
            output_dir=destination,
        )
    except Exception as exc:
        if destination is not None:
            destination = destination.expanduser().resolve()
            destination.mkdir(parents=True, exist_ok=True)
            _manifest_update(
                destination,
                status="failed",
                refinement_finished_at=_utc_now(),
                error=str(exc),
                exit_code=1,
            )
        print(f"[refine-topk] error: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"[refine-topk] output={destination}", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
