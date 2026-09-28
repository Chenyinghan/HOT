"""Run BASS upper-level search with task-provided XML evaluation hooks."""

from __future__ import annotations

import argparse
import atexit
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping


def _configure_parent_native_threads() -> None:
    numeric_threads = os.environ.get("BILEVEL_PARENT_NUMERIC_THREADS", "1")
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


_configure_parent_native_threads()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bilevel.upper.bass.actions import Action
from bilevel.upper.bass.config import BASSConfig
from bilevel.upper.bass.io_assets import (
    load_assets,
    resolve_asset_id,
    select_assets,
)
from bilevel.upper.bass.search import BASSEvaluation, SearchResult, run_bass
from bilevel.upper.bass.assembly import compile_scene, sequence_to_tree, tree_to_dict
from bilevel.lower.evaluation import (
    build_evaluation_identity,
    enrich_evaluation_result,
    validate_result_identity,
)
from bilevel.runtime import optimize_xml, visualize_xml
from bilevel.serialization import jsonable
from tasks import (
    TaskBase,
    canonical_task_module_path,
    load_task,
)

XML_CACHE_VERSION = "root_rotation_search_v12"
_RUN_LOCK_HANDLES: list[Any] = []
_RERANK_ALLOWED_OVERRIDES = {
    "direct_planar_action_step_scale",
    "direct_planar_design_step_scale",
    "direct_planar_max_action_step",
    "low_level_maxls",
}


def _clean_stale_lock_files(lock_root: Path) -> None:
    """Remove lock files left behind by interrupted processes."""
    if not lock_root.exists():
        return
    for lock_path in lock_root.glob("*.lock"):
        try:
            with lock_path.open("a+", encoding="utf-8") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            continue


def _resolve_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return base / path


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_string_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    return values or None


def _parse_json_array(value: str) -> list[Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected a JSON array: {exc}") from exc
    if not isinstance(payload, list):
        raise argparse.ArgumentTypeError("expected a JSON array")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _post_search_rerank_config(
    task_json: dict[str, Any],
) -> dict[str, Any]:
    """Validate the optional shared post-search optimizer rerank config."""

    raw = task_json.get("post_search_rerank")
    if raw in (None, False):
        return {
            "enabled": False,
            "top_k": 0,
            "workers": 0,
            "optimizer_variants": [],
        }
    if not isinstance(raw, dict):
        raise TypeError("post_search_rerank must be an object or false")

    enabled = bool(raw.get("enabled", True))
    top_k = int(raw.get("top_k", 0))
    workers = int(raw.get("workers", 1))
    variants = raw.get("optimizer_variants", [])
    if top_k < 0:
        raise ValueError("post_search_rerank.top_k must be non-negative")
    if workers < 1:
        raise ValueError("post_search_rerank.workers must be positive")
    if not isinstance(variants, list):
        raise TypeError(
            "post_search_rerank.optimizer_variants must be a list"
        )

    normalized = []
    seen_names = set()
    for index, entry in enumerate(variants):
        if not isinstance(entry, dict):
            raise TypeError(
                "post_search_rerank optimizer variants must be objects"
            )
        name = str(entry.get("name", f"variant_{index}")).strip()
        if not name or name in seen_names:
            raise ValueError(
                "post_search_rerank optimizer variant names must be "
                "non-empty and unique"
            )
        overrides = entry.get("task_config_overrides", {})
        if not isinstance(overrides, dict) or not overrides:
            raise ValueError(
                f"post_search_rerank variant {name!r} requires non-empty "
                "task_config_overrides"
            )
        unsupported = set(overrides).difference(
            _RERANK_ALLOWED_OVERRIDES
        )
        if unsupported:
            raise ValueError(
                f"post_search_rerank variant {name!r} changes unsupported "
                f"task fields: {sorted(unsupported)}"
            )
        for key, value in overrides.items():
            if key == "low_level_maxls":
                if int(value) < 1:
                    raise ValueError(f"{name}.{key} must be positive")
            elif not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name}.{key} must be finite and positive")
        normalized.append(
            {
                "name": name,
                "task_config_overrides": dict(overrides),
            }
        )
        seen_names.add(name)

    return {
        "enabled": bool(enabled and top_k > 0 and normalized),
        "top_k": top_k,
        "workers": workers,
        "optimizer_variants": normalized,
    }


def _generic_design_protocol_from_inputs(
    args: argparse.Namespace,
    task_json: dict[str, Any],
    task_config: dict[str, Any],
) -> str:
    canonical = "connected_direct_planar_hexahedron"
    requested = _value(
        args.generic_design_protocol,
        task_json.get("generic_design_protocol"),
        task_config.get("generic_design_protocol"),
        default=canonical,
    )
    if str(requested) != canonical:
        raise ValueError(
            f"Unsupported generic_design_protocol {requested!r}; the canonical "
            f"framework only supports {canonical!r}"
        )
    return canonical


def _validate_mission_name(
    task_json: dict[str, Any],
    task: TaskBase,
) -> None:
    configured = task_json.get("mission_name", task_json.get("task_name"))
    declared = task.name
    if configured is not None and declared is not None and str(configured) != str(declared):
        raise ValueError(
            f"task.json mission_name={configured!r} does not match "
            f"canonical task name={declared!r}"
        )


def _bass_tool_assets(assets: list[Any], *, root_asset_id: str) -> list[Any]:
    """Return assets eligible for generated tool search."""
    if any(getattr(asset, "role", None) is not None for asset in assets):
        filtered = [
            asset
            for asset in assets
            if asset.asset_id == root_asset_id or asset.searchable
        ]
        if not any(
            asset.asset_id == root_asset_id
            for asset in filtered
        ):
            raise ValueError(
                f"root asset id {root_asset_id!r} is absent from the "
                "selected catalog assets"
            )
        return filtered

    any_tool_tag = any("tool" in set(getattr(asset, "tags", None) or []) for asset in assets)
    if any_tool_tag:
        filtered = [
            asset
            for asset in assets
            if "tool" in set(getattr(asset, "tags", None) or [])
        ]
    else:
        filtered = [
            asset
            for asset in assets
            if "finger" not in set(getattr(asset, "tags", None) or [])
        ]
    if not any(asset.asset_id == root_asset_id for asset in filtered):
        raise ValueError(
            f"root asset id {root_asset_id!r} is not enabled as a tool-search asset"
        )
    return filtered


def _bass_config_from_inputs(
    args: argparse.Namespace,
    task_json: dict[str, Any],
    task: TaskBase,
) -> BASSConfig:
    bass_defaults = dict(task_json.get("bass", {}))
    bass_modes = {"bass_n1", "bass_n2"}
    resolved_structural_dag_path = _value(
        getattr(args, "structural_dag_path", None),
        bass_defaults.get("structural_dag_path"),
    )
    default_acquisition = (
        "bass_n2"
    )
    resolved_acquisition = str(
        _value(
            getattr(args, "acquisition", None),
            bass_defaults.get("acquisition"),
            default=default_acquisition,
        )
    )
    bass_calibration = None
    if resolved_acquisition in bass_modes:
        from bilevel.upper.bass.bayesian_lookahead import load_calibration_artifact

        calibration_path = _value(
            getattr(args, "calibration_artifact", None),
            bass_defaults.get("calibration_artifact"),
        )
        if calibration_path:
            bass_calibration = load_calibration_artifact(
                calibration_path,
                expected_task_name=str(task_json.get("mission_name", "")),
            )
        if bass_calibration is not None:
            configured_fields = {
                "milestone_thresholds": "progress_thresholds",
                "continuation_prior_means": "continuation_prior_means",
                "prior_strengths": "prior_strengths",
            }
            for config_name, artifact_name in configured_fields.items():
                explicit = _value(
                    getattr(args, config_name, None),
                    bass_defaults.get(config_name),
                )
                artifact_value = bass_calibration[artifact_name]
                if explicit is not None and json.dumps(explicit) != json.dumps(
                    artifact_value
                ):
                    raise ValueError(
                        "{} disagrees with frozen BASS calibration artifact {}"
                        .format(config_name, calibration_path)
                    )
    task_config = dict(task_json.get("task_config", {}))
    generic_design_protocol = _generic_design_protocol_from_inputs(args, task_json, task_config)
    spec_function_count = task.search_spec().function_count
    function_count = _value(
        args.target_function_count,
        task_json.get("function_count"),
        spec_function_count,
    )
    eval_workers = _value(args.eval_workers, bass_defaults.get("eval_workers"))
    resolved_threads = int(
        _value(args.threads, bass_defaults.get("threads"), default=4)
    )
    resolved_eval_workers = (
        resolved_threads if eval_workers is None else int(eval_workers)
    )
    default_bass_warmup = (
        70 * resolved_eval_workers
        if (
            resolved_acquisition in bass_modes
            and bass_calibration is None
        )
        else 0
    )
    search_time_budget = _value(
        args.search_time_budget,
        bass_defaults.get("search_time_budget"),
        bass_defaults.get("time_budget"),
    )
    max_head_links = _value(
        args.max_head_links,
        task_json.get("max_head_links"),
        bass_defaults.get("max_head_links"),
    )
    root_asset_id = _value(
        args.root_asset_id,
        task_json.get("root_asset_id"),
        bass_defaults.get("root_asset_id"),
        default="root/universal_handle",
    )
    root_blocked_face = _value(
        args.root_blocked_face,
        task_json.get("root_blocked_face"),
        bass_defaults.get("root_blocked_face"),
        default=2,
    )
    resolved_scheduler_v2 = True
    resolved_parallelization_mode = "shared_tree"
    resolved_supply_policy = "baseline"
    return BASSConfig(
        asset_volume_coeff=float(bass_defaults.get("asset_volume_coeff", 0.0)),
        threads=resolved_threads,
        max_head_links=(
            None if max_head_links is None else int(max_head_links)
        ),
        max_depth=int(_value(args.max_depth, bass_defaults.get("max_depth"), default=6)),
        iteration_budget=int(_value(args.iteration_budget, bass_defaults.get("iteration_budget"), default=0)),
        search_time_budget=None if search_time_budget is None else float(search_time_budget),
        calibration_budget=int(
            _value(
                getattr(args, "calibration_budget", None),
                bass_defaults.get("calibration_budget"),
                default=default_bass_warmup,
            )
        ),
        seed=int(_value(args.seed, bass_defaults.get("seed"), default=0)),
        max_actions_per_expansion=int(_value(args.max_actions_per_expansion, bass_defaults.get("max_actions_per_expansion"), default=0)),
        partial_state_transpositions=bool(
            _value(
                getattr(args, "partial_state_transpositions", None),
                bass_defaults.get("partial_state_transpositions"),
                default=False,
            )
        ),
        structural_factorization=bool(
            _value(
                getattr(args, "structural_factorization", None),
                bass_defaults.get("structural_factorization"),
                default=False,
            )
        ),
        scheduler_v2=resolved_scheduler_v2,
        scheduler_health_log_interval=float(
            _value(
                getattr(args, "scheduler_health_log_interval", None),
                bass_defaults.get("scheduler_health_log_interval"),
                default=60.0,
            )
        ),
        scheduler_supply_critical_fraction=float(
            _value(
                getattr(args, "scheduler_supply_critical_fraction", None),
                bass_defaults.get("scheduler_supply_critical_fraction"),
                default=0.7,
            )
        ),
        scheduler_supply_critical_seconds=float(
            _value(
                getattr(args, "scheduler_supply_critical_seconds", None),
                bass_defaults.get("scheduler_supply_critical_seconds"),
                default=300.0,
            )
        ),
        scheduler_supply_recent_new_seconds=float(
            _value(
                getattr(args, "scheduler_supply_recent_new_seconds", None),
                bass_defaults.get("scheduler_supply_recent_new_seconds"),
                default=600.0,
            )
        ),
        scheduler_supply_rate_window_seconds=float(
            _value(
                getattr(args, "scheduler_supply_rate_window_seconds", None),
                bass_defaults.get("scheduler_supply_rate_window_seconds"),
                default=300.0,
            )
        ),
        scheduler_ready_low_watermark_fraction=float(
            _value(
                getattr(args, "scheduler_ready_low_watermark_fraction", None),
                bass_defaults.get("scheduler_ready_low_watermark_fraction"),
                default=0.5,
            )
        ),
        scheduler_ready_high_watermark_fraction=float(
            _value(
                getattr(args, "scheduler_ready_high_watermark_fraction", None),
                bass_defaults.get("scheduler_ready_high_watermark_fraction"),
                default=1.0,
            )
        ),
        scheduler_occupancy_warning_fraction=float(
            _value(
                getattr(args, "scheduler_occupancy_warning_fraction", None),
                bass_defaults.get("scheduler_occupancy_warning_fraction"),
                default=0.8,
            )
        ),
        scheduler_occupancy_recovery_fraction=float(
            _value(
                getattr(args, "scheduler_occupancy_recovery_fraction", None),
                bass_defaults.get("scheduler_occupancy_recovery_fraction"),
                default=0.9,
            )
        ),
        scheduler_occupancy_warning_seconds=float(
            _value(
                getattr(args, "scheduler_occupancy_warning_seconds", None),
                bass_defaults.get("scheduler_occupancy_warning_seconds"),
                default=30.0,
            )
        ),
        scheduler_occupancy_recovery_seconds=float(
            _value(
                getattr(args, "scheduler_occupancy_recovery_seconds", None),
                bass_defaults.get("scheduler_occupancy_recovery_seconds"),
                default=5.0,
            )
        ),
        acquisition=resolved_acquisition,
        milestone_thresholds=_value(
            None if bass_calibration is None else bass_calibration["progress_thresholds"],
            getattr(args, "milestone_thresholds", None),
            bass_defaults.get("milestone_thresholds"),
            default=(),
        ),
        continuation_prior_means=_value(
            None if bass_calibration is None else bass_calibration["continuation_prior_means"],
            getattr(args, "continuation_prior_means", None),
            bass_defaults.get("continuation_prior_means"),
            default=(),
        ),
        prior_strengths=_value(
            None if bass_calibration is None else bass_calibration["prior_strengths"],
            getattr(args, "prior_strengths", None),
            bass_defaults.get("prior_strengths"),
            default=(2.0,),
        ),
        calibration_artifact=_value(
            getattr(args, "calibration_artifact", None),
            bass_defaults.get("calibration_artifact"),
        ),
        calibration_bins_per_stage=tuple(
            _value(
                getattr(args, "calibration_bins_per_stage", None),
                bass_defaults.get("calibration_bins_per_stage"),
                default=(),
            )
        ),
        calibration_smoothing=float(
            _value(
                getattr(args, "calibration_smoothing", None),
                bass_defaults.get("calibration_smoothing"),
                default=0.5,
            )
        ),
        calibration_threshold_sample=str(
            _value(
                getattr(args, "calibration_threshold_sample", None),
                bass_defaults.get("calibration_threshold_sample"),
                default="interior",
            )
        ),
        physical_dedup_mode=str(
            _value(
                getattr(args, "physical_dedup_mode", None),
                bass_defaults.get("physical_dedup_mode"),
                default="off",
            )
        ),
        physical_signature_eps=float(
            _value(
                getattr(args, "physical_signature_eps", None),
                bass_defaults.get("physical_signature_eps"),
                default=1e-8,
            )
        ),
        function_semantic_labels=tuple(task.task_spec().functions),
        structural_dag_path=resolved_structural_dag_path,
        structural_dag_mmap=bool(
            _value(
                getattr(args, "structural_dag_mmap", None),
                bass_defaults.get("structural_dag_mmap"),
                default=True,
            )
        ),
        diagnostics_jsonl=_value(
            getattr(args, "bass_diagnostics_jsonl", None),
            bass_defaults.get("diagnostics_jsonl"),
        ),
        rollout_policy=str(_value(args.rollout_policy, bass_defaults.get("rollout_policy"), default="uniform")),
        rollout_end_prob_by_depth=_value(
            _parse_float_list(args.rollout_end_prob_by_depth),
            bass_defaults.get("rollout_end_prob_by_depth"),
        ),
        rollout_addlink_prob_by_depth=_value(
            _parse_float_list(args.rollout_addlink_prob_by_depth),
            bass_defaults.get("rollout_addlink_prob_by_depth"),
        ),
        rollout_branching_penalty_by_depth=_value(
            _parse_float_list(args.rollout_branching_penalty_by_depth),
            bass_defaults.get("rollout_branching_penalty_by_depth"),
        ),
        parallelization_mode=resolved_parallelization_mode,
        reward_mode=str(
            _value(
                bass_defaults.get("reward_mode"),
                default="negative_cost",
            )
        ),
        target_function_count=None if function_count is None else int(function_count),
        function_count_margin=int(_value(args.function_count_margin, task_json.get("function_count_margin"), bass_defaults.get("function_count_margin"), default=0)),
        function_group_depth_delta=(
            args.function_group_depth_delta
            if getattr(args, "function_group_depth_delta", None) is not None
            else task_json.get(
                "function_group_depth_delta",
                task_config.get(
                    "function_group_depth_delta",
                    bass_defaults.get("function_group_depth_delta", 0),
                ),
            )
        ),
        root_asset_id=str(root_asset_id),
        root_blocked_face=(
            None if root_blocked_face is None else int(root_blocked_face)
        ),
        root_rotation_options=tuple(
            _value(
                _parse_string_list(getattr(args, "root_rotation_options", None)),
                task_json.get("root_rotation_options"),
                bass_defaults.get("root_rotation_options"),
                default=(),
            )
        ),
        eval_workers=None if eval_workers is None else int(eval_workers),
        log_progress=bool(args.debug),
    )


def _structural_surrogate(sequence: list[Action], target_count: int | None) -> float:
    add_count = sum(1 for action in sequence if action.kind == "AddLink")
    group_count = sum(
        1
        for action in sequence
        if isinstance(action, Action)
        and action.kind == "AddLink"
        and bool(getattr(action, "start_function_group", False))
    )
    end_count = sum(1 for action in sequence if action.kind == "End")
    completion_penalty = 0.0 if end_count > add_count else 100.0
    target = group_count if target_count is None else target_count
    return float(abs(group_count - target) + 0.01 * len(sequence) + completion_penalty)


def _asset_volume(asset: Any) -> float:
    half_extents = getattr(asset, "half_extents", None)
    if half_extents is None:
        return 0.0
    if len(half_extents) != 3:
        return 0.0
    return float(8.0 * float(half_extents[0]) * float(half_extents[1]) * float(half_extents[2]))


class CachedXmlEvaluator:
    """Convert rollout sequences to cached XML and evaluate with task hooks."""

    def __init__(
        self,
        *,
        task: TaskBase,
        task_json: dict[str, Any],
        task_module_path: Path,
        cache_dir: Path,
        output_dir: Path,
        replay_dir: Path,
        assets_json: Path,
        scene_xml: Path,
        bass_config: BASSConfig,
        xml_options: dict[str, Any],
        low_level_maxiter: int | None = None,
        low_level_num_steps: int | None = None,
        low_level_sub_steps: int | None = None,
        low_level_timeout: float | None = None,
        low_level_numeric_threads: int | None = None,
        low_level_grad_clip: float | None = None,
        low_level_step_scale: float | None = None,
        low_level_redmax_verbose: bool = False,
        optimizer_strategy: str | None = None,
        force_connectivity: bool = False,
        generic_design_protocol: str = "connected_direct_planar_hexahedron",
        experimental: bool = False,
        experimental_alpha: float = 0.5,
        experimental_beta: float = 1.5,
        debug: bool = False,
        mission_name: str = "task",
        best_xml_path: Path | None = None,
        best_run_json_path: Path | None = None,
        eval_csv_path: Path | None = None,
    ) -> None:
        self.task = task
        self.task_json = task_json
        task_config = dict(self.task_json.get("task_config", {}) or {})
        self.task_module_path = task_module_path
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.replay_dir = replay_dir
        self.assets_json = assets_json
        self.scene_xml = scene_xml
        self.bass_config = bass_config
        loaded_assets = load_assets(str(self.assets_json))
        self._asset_volume_by_id = {
            asset.asset_id: _asset_volume(asset)
            for asset in loaded_assets
        }
        self.xml_options = xml_options
        self.low_level_maxiter = low_level_maxiter
        self.low_level_num_steps = low_level_num_steps
        self.low_level_sub_steps = low_level_sub_steps
        self.low_level_timeout = low_level_timeout
        self.low_level_numeric_threads = low_level_numeric_threads
        self.low_level_grad_clip = low_level_grad_clip
        self.low_level_step_scale = low_level_step_scale
        self.low_level_redmax_verbose = bool(low_level_redmax_verbose)
        self.low_level_spawn_method = str(
            task_config.get("low_level_spawn_method", "auto")
        ).strip().lower().replace("-", "_")
        if self.low_level_spawn_method not in {
            "auto",
            "posix_spawn",
            "popen",
        }:
            raise ValueError(
                "low_level_spawn_method must be auto, posix_spawn, or popen"
            )
        self.optimizer_strategy = optimizer_strategy
        self.force_connectivity = bool(force_connectivity)
        self.generic_design_protocol = str(generic_design_protocol)
        self.experimental = bool(experimental)
        self.experimental_alpha = float(experimental_alpha)
        self.experimental_beta = float(experimental_beta)
        self.debug = bool(debug)
        self.mission_name = str(mission_name)
        self.best_xml_path = best_xml_path
        self.best_run_json_path = best_run_json_path
        self.eval_csv_path = eval_csv_path
        self._best_checkpoint_score = float("inf")
        self._best_checkpoint_reward = float("-inf")
        self._best_run_key: str | None = None
        self._best_result: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._checkpoint_lock = threading.Lock()
        self._csv_lock = threading.Lock()
        self._xml_write_locks = tuple(
            threading.Lock() for _ in range(64)
        )
        self._eval_csv_fsync_interval = max(
            1,
            int(task_config.get("eval_csv_fsync_interval", 64)),
        )
        self._eval_csv_rows_since_sync = 0
        self._scores: dict[str, float] = {}
        self._run_results: dict[str, dict[str, Any]] = {}
        self._sequences: dict[str, list[Action]] = {}
        self._post_search_rerank_enabled = bool(
            _post_search_rerank_config(self.task_json)["enabled"]
        )
        self._inflight: dict[str, threading.Event] = {}
        self._xml_scores: dict[str, float] = {}
        self._xml_run_results: dict[str, dict[str, Any]] = {}
        self._xml_inflight: dict[str, threading.Event] = {}
        self._eval_count = 0
        if self.eval_csv_path is not None:
            self.eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.eval_csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._eval_csv_fields())
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())

    def _eval_csv_fields(self) -> list[str]:
        return [
            "eval_number",
            "bass_search_strategy",
            "run_key",
            "evaluation_key",
            "xml_digest",
            "score",
            "loss",
            "bass_reward",
            "task_success",
            "task_milestone",
            "task_stage_count",
            "task_progress",
            "task_feasible",
            "raw_terminal_task_success",
            "loss_quality",
            "raw_process_reward",
            "normalized_failure_reward",
            "failure_ceiling_enabled",
            "failure_ceiling",
            "final_scalar_reward",
            "task_progress_status",
            "optimizer_termination_reason",
            "last_completed_stage",
            "failed_stage",
            "status",
            "returncode",
            "action_count",
            "assets",
            "xml_path",
            "rollout_dir",
            "logs_path",
            "low_level_score",
            "asset_volume",
            "asset_volume_coeff",
            "asset_volume_penalty",
            "morphology_parameterization",
            "morphology_dim",
            "morphology_layout_fingerprint",
            "simulator_sha256",
        ]

    def scheduler_health_snapshot(self) -> dict[str, Any]:
        """Expose logical and native occupancy without coupling BASS to bilevel."""

        from bilevel.lower.engine import low_level_process_metrics

        with self._lock:
            logical_active = len(self._xml_inflight)
            retained_results = len(self._run_results)
            cached_xml_results = len(self._xml_run_results)
        return {
            **low_level_process_metrics(),
            "logical_active": int(logical_active),
            "retained_results": int(retained_results),
            "cached_xml_results": int(cached_xml_results),
        }

    def scheduler_abort_low_level(self, reason: str) -> int:
        """Terminate only RedMax workers owned by this controller process."""

        from bilevel.lower.engine import terminate_active_low_level_processes

        return terminate_active_low_level_processes(reason=reason)

    def _append_eval_csv(
        self,
        *,
        eval_count: int,
        key: str,
        sequence: list[Action],
        assets: list[str],
        score: float,
        status: str,
        run_result: dict[str, Any],
        xml_path: Path,
    ) -> None:
        if self.eval_csv_path is None:
            return
        optimizer_details = dict(
            (run_result.get("optimizer", {}) or {}).get("details", {})
            or {}
        )
        row = {
            "eval_number": eval_count,
            "bass_search_strategy": self.bass_config.acquisition,
            "run_key": key,
            "evaluation_key": run_result.get("evaluation_key", ""),
            "xml_digest": run_result.get("xml_digest", ""),
            "score": score,
            "loss": run_result.get("loss", score),
            "bass_reward": run_result.get("bass_reward", ""),
            "task_success": run_result.get("task_success", ""),
            "task_milestone": run_result.get("task_milestone", ""),
            "task_stage_count": run_result.get("task_stage_count", ""),
            "task_progress": run_result.get("task_progress", ""),
            "task_feasible": run_result.get("task_feasible", ""),
            "raw_terminal_task_success": run_result.get(
                "raw_terminal_task_success",
                "",
            ),
            "loss_quality": run_result.get("loss_quality", ""),
            "raw_process_reward": run_result.get(
                "raw_process_reward",
                "",
            ),
            "normalized_failure_reward": run_result.get(
                "normalized_failure_reward",
                "",
            ),
            "failure_ceiling_enabled": run_result.get(
                "failure_ceiling_enabled",
                "",
            ),
            "failure_ceiling": run_result.get("failure_ceiling", ""),
            "final_scalar_reward": run_result.get(
                "final_scalar_reward",
                "",
            ),
            "task_progress_status": run_result.get(
                "task_progress_status",
                "",
            ),
            "optimizer_termination_reason": optimizer_details.get(
                "termination_reason",
                "",
            ),
            "last_completed_stage": optimizer_details.get(
                "last_completed_stage",
                "",
            ),
            "failed_stage": optimizer_details.get("failed_stage", ""),
            "status": status,
            "returncode": run_result.get("returncode", ""),
            "action_count": len(sequence),
            "assets": " ".join(assets),
            "xml_path": str(xml_path),
            "rollout_dir": run_result.get("rollout_dir", ""),
            "logs_path": run_result.get("logs_path", ""),
            "low_level_score": run_result.get("low_level_score", run_result.get("raw_score", score)),
            "asset_volume": run_result.get("asset_volume", ""),
            "asset_volume_coeff": run_result.get("asset_volume_coeff", ""),
            "asset_volume_penalty": run_result.get("asset_volume_penalty", ""),
            "morphology_parameterization": run_result.get(
                "morphology_parameterization",
                "",
            ),
            "morphology_dim": run_result.get("morphology_dim", ""),
            "morphology_layout_fingerprint": run_result.get(
                "morphology_layout_fingerprint",
                "",
            ),
            "simulator_sha256": (
                (
                    run_result.get("simulator_identity", {})
                    .get("binary", {})
                    or {}
                ).get("sha256", "")
            ),
        }
        with self._csv_lock:
            with self.eval_csv_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._eval_csv_fields())
                writer.writerow(row)
                self._eval_csv_rows_since_sync += 1
                if (
                    self._eval_csv_rows_since_sync
                    >= self._eval_csv_fsync_interval
                ):
                    f.flush()
                    os.fsync(f.fileno())
                    self._eval_csv_rows_since_sync = 0

    def _sequence_key(self, sequence: list[Action]) -> str:
        payload = json.dumps(
            {
                "version": XML_CACHE_VERSION,
                "sequence": [action.to_dict() for action in sequence],
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _apply_current_experimental_gate(self, score: float) -> float:
        if not self.experimental or not math.isfinite(score):
            return score
        with self._lock:
            best_reference = self._best_checkpoint_score
        if math.isfinite(best_reference) and score > self.experimental_beta * best_reference:
            return float("inf")
        return score

    def xml_for_sequence(
        self,
        sequence: list[Action],
        *,
        name: str | None = None,
        out_path: Path | None = None,
    ) -> Path:
        key = name or self._sequence_key(sequence)
        xml_path = out_path if out_path is not None else self.cache_dir / f"{key}.xml"
        path_key = str(xml_path.resolve())
        path_lock = self._xml_write_locks[
            hash(path_key) % len(self._xml_write_locks)
        ]
        with path_lock:
            if out_path is None and xml_path.exists():
                return xml_path
            compile_scene(
                sequence,
                str(xml_path),
                scene_xml=str(self.scene_xml),
                assets_json=str(self.assets_json),
                root_asset_id=self.bass_config.root_asset_id,
                model_name=str(self.xml_options.get("model_name", "bilevel_rollout")),
                tool_design_params=int(self.xml_options.get("tool_design_params", 47)),
                root_design_params=int(
                    self.xml_options.get("root_design_params", 0)
                ),
                root_asset_pos=str(
                    self.xml_options.get("root_asset_pos", "0 0 0")
                ),
                root_asset_quat=str(
                    self.xml_options.get("root_asset_quat", "1 0 0 0")
                ),
                functions=self.xml_options.get("functions"),
                root_joint_type=str(self.xml_options.get("root_joint_type", "fixed")),
                root_joint_pos=str(self.xml_options.get("root_joint_pos", "0 0 0")),
                root_joint_quat=str(self.xml_options.get("root_joint_quat", "1 0 0 0")),
                root_joint_damping=self.xml_options.get("root_joint_damping"),
                root_aux_joint_type=self.xml_options.get("root_aux_joint_type"),
                root_aux_joint_name=str(self.xml_options.get("root_aux_joint_name", "freeform_aux_joint")),
                root_aux_joint_pos=str(self.xml_options.get("root_aux_joint_pos", "0 0 0")),
                root_aux_joint_quat=str(self.xml_options.get("root_aux_joint_quat", "1 0 0 0")),
                root_aux_joint_axis=str(self.xml_options.get("root_aux_joint_axis", "0 1 0")),
                root_aux_joint_axis1=str(self.xml_options.get("root_aux_joint_axis1", "0 0 1")),
                root_aux_joint_damping=self.xml_options.get("root_aux_joint_damping"),
                root_aux_joint_lim=self.xml_options.get("root_aux_joint_lim"),
                root_aux_joint_lim_stiffness=self.xml_options.get(
                    "root_aux_joint_lim_stiffness"
                ),
                generated_robot_placement=str(self.xml_options.get("generated_robot_placement", "after_scene")),
                root_body_size=str(self.xml_options.get("root_body_size", "0.05 0.05 0.05")),
                root_motor_ctrl=str(self.xml_options.get("root_motor_ctrl", "force")),
                root_motor_ctrl_range=str(self.xml_options.get("root_motor_ctrl_range", "-6e5 6e5")),
                root_motor_P=str(self.xml_options.get("root_motor_P", "2e4")),
                root_motor_D=str(self.xml_options.get("root_motor_D", "2e3")),
                root_aux_motor_ctrl=str(self.xml_options.get("root_aux_motor_ctrl", "position")),
                root_aux_motor_ctrl_range=str(self.xml_options.get("root_aux_motor_ctrl_range", "-5e4 5e4")),
                root_aux_motor_P=str(self.xml_options.get("root_aux_motor_P", "2e4")),
                root_aux_motor_D=str(self.xml_options.get("root_aux_motor_D", "2e3")),
                generated_endeffector_radius=str(
                    self.xml_options.get(
                        "generated_endeffector_radius",
                        "0.2",
                    )
                ),
                contact_model=self.xml_options.get("contact_model"),
            )
        return xml_path

    def result_for_sequence(self, sequence: list[Action]) -> dict[str, Any]:
        key = self._sequence_key(sequence)
        with self._lock:
            best_run_key = getattr(self, "_best_run_key", None)
            best_result = getattr(self, "_best_result", None)
            if key == best_run_key and best_result is not None:
                stored = dict(best_result)
            else:
                stored = dict(self._run_results.get(key, {}))
        result_metadata = dict(stored.get("result", {}))
        result_path = result_metadata.get("low_level_result_path")
        if result_path:
            try:
                restored = _load_json(Path(result_path))
                restored.update(result_metadata)
                stored["result"] = restored
                for field in ("params", "params_path", "logs_path", "finalized_state_path"):
                    if stored.get(field) is None:
                        stored[field] = restored.get(field)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return stored

    @staticmethod
    def _compact_run_result(run_result: dict[str, Any]) -> dict[str, Any]:
        retained_fields = (
            "score",
            "loss",
            "raw_score",
            "raw_loss",
            "low_level_score",
            "regularized_score",
            "asset_volume",
            "asset_volume_coeff",
            "asset_volume_penalty",
            "error",
            "exception",
            "returncode",
            "rollout_dir",
            "params_path",
            "logs_path",
            "finalized_state_path",
            "low_level_result_path",
            "num_log_entries",
            "num_steps",
            "sub_steps",
            "num_ctrl_steps",
            "ndof_u",
            "ndof_cage",
            "status",
            "result_schema",
            "result_schema_version",
            "evaluation_key",
            "evaluation_identity",
            "simulator_identity",
            "action_parameterization",
            "action_dim",
            "morphology_parameterization",
            "morphology_dim",
            "morphology_layout_fingerprint",
            "total_dim",
            "parameter_artifact_metadata",
            "task",
            "skeleton",
            "skeleton_hash",
            "root_asset_id",
            "head_asset_ids",
            "loss_terms",
            "optimizer",
            "validation",
            "diagnostics",
            "stage_loss_diagnostics",
            "xml_path",
            "replay_path",
            "post_search_rerank",
            "bass_reward",
            "task_success",
            "task_milestone",
            "task_stage_count",
            "task_progress",
            "task_feasible",
            "raw_terminal_task_success",
            "loss_quality",
            "raw_process_reward",
            "normalized_failure_reward",
            "failure_ceiling_enabled",
            "failure_ceiling",
            "final_scalar_reward",
            "task_progress_gates",
            "task_progress_status",
        )
        return {field: run_result.get(field) for field in retained_fields if field in run_result}

    @classmethod
    def _alias_cache_result(
        cls,
        stored_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Retain only fields required to resolve an equivalent XML later."""

        run_result = dict(stored_result.get("result", {}) or {})
        alias_fields = (
            "score",
            "loss",
            "raw_score",
            "raw_loss",
            "low_level_score",
            "regularized_score",
            "error",
            "returncode",
            "rollout_dir",
            "params_path",
            "logs_path",
            "finalized_state_path",
            "low_level_result_path",
            "evaluation_key",
            "evaluation_identity",
            "simulator_identity",
            "bass_reward",
            "task_success",
            "task_milestone",
            "task_stage_count",
            "task_progress",
            "task_feasible",
            "raw_terminal_task_success",
            "loss_quality",
            "raw_process_reward",
            "normalized_failure_reward",
            "failure_ceiling_enabled",
            "failure_ceiling",
            "final_scalar_reward",
            "task_progress_status",
            "validation",
        )
        compact_run_result = {
            field: run_result.get(field)
            for field in alias_fields
            if field in run_result
        }
        return {
            "eval_number": stored_result.get("eval_number"),
            "run_key": stored_result.get("run_key"),
            "evaluation_key": stored_result.get("evaluation_key"),
            "evaluation_identity": stored_result.get("evaluation_identity"),
            "xml_digest": stored_result.get("xml_digest"),
            "score": stored_result.get("score"),
            "loss": stored_result.get("loss"),
            "xml_path": stored_result.get("xml_path"),
            "params": None,
            "params_path": stored_result.get("params_path"),
            "logs_path": stored_result.get("logs_path"),
            "finalized_state_path": stored_result.get("finalized_state_path"),
            "result": compact_run_result,
        }

    def _release_transient_result(self, key: str) -> None:
        """Bound result memory once BASS has consumed a non-rerank record."""

        if getattr(self, "_post_search_rerank_enabled", False):
            return
        with self._lock:
            if key != getattr(self, "_best_run_key", None):
                self._run_results.pop(key, None)

    @staticmethod
    def _merged_rollout_diagnostics(
        run_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge engine metadata with task diagnostics persisted by Runner."""

        merged = dict(run_result.get("motion_diagnostics", {}) or {})
        rollout_dir = run_result.get("rollout_dir")
        if not rollout_dir:
            return merged
        diagnostics_path = Path(str(rollout_dir)) / "diagnostics.json"
        if not diagnostics_path.exists():
            return merged
        loaded = _load_json(diagnostics_path)
        if not isinstance(loaded, dict):
            raise TypeError(
                f"rollout diagnostics must be a JSON object: {diagnostics_path}"
            )
        # Runner diagnostics are the task authority. Engine-level metadata such
        # as mount_integrity remains available when keys do not overlap.
        merged.update(loaded)
        return merged

    def evaluate_for_bass(
        self,
        sequence: list[Action],
    ) -> float | BASSEvaluation:
        """Evaluate one sequence and expose task reward when configured."""

        score = float(self(sequence))
        key = self._sequence_key(sequence)
        if self.bass_config.reward_mode != "bounded_task":
            self._release_transient_result(key)
            return score
        with self._lock:
            stored = dict(self._run_results.get(key, {}))
        result = dict(stored.get("result", {}) or {})
        reward = (
            float(result.get("bass_reward", 0.0) or 0.0)
            if math.isfinite(score)
            else 0.0
        )
        milestone = result.get("task_milestone")
        stage_count = result.get("task_stage_count")
        # Preflight and initial-feasibility failures may carry a partial task
        # diagnostic, but they have not produced a complete staged evaluation.
        if milestone is None or stage_count is None:
            milestone = None
            stage_count = None
        evaluation = BASSEvaluation(
            score=score,
            reward=reward,
            valid=math.isfinite(score),
            task_milestone=milestone,
            task_stage_count=stage_count,
            task_progress=float(result.get("task_progress", 0.0) or 0.0),
            task_success=bool(result.get("task_success", False)),
            task_feasible=bool(result.get("task_feasible", True)),
        )
        self._release_transient_result(key)
        return evaluation

    def __call__(self, sequence: list[Action]) -> float:
        add_count = sum(1 for action in sequence if action.kind == "AddLink")
        end_count = sum(1 for action in sequence if action.kind == "End")
        if end_count <= add_count:
            return _structural_surrogate(sequence, self.bass_config.target_function_count) + 100.0

        key = self._sequence_key(sequence)
        with self._lock:
            cached = self._scores.get(key)
            if cached is not None:
                print(f"[bass] retry duplicate sequence key={key} score={cached}", flush=True)
                return float("inf")
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                owner = True
            else:
                owner = False

        if not owner:
            print(f"[bass] retry in-flight duplicate sequence key={key}", flush=True)
            return float("inf")

        try:
            return self._evaluate_uncached(sequence, key)
        finally:
            with self._lock:
                done = self._inflight.pop(key, None)
                if done is not None:
                    done.set()

    def _evaluate_uncached(self, sequence: list[Action], key: str) -> float:
        with self._lock:
            cached = self._scores.get(key)
            if getattr(self, "_post_search_rerank_enabled", False):
                self._sequences.setdefault(key, list(sequence))
        if cached is not None:
            return cached

        xml_path = self.xml_for_sequence(sequence)
        candidate_validation = self.task.validate_candidate(xml_path)
        if (
            not isinstance(candidate_validation, dict)
            or "ok" not in candidate_validation
        ):
            raise TypeError(
                "task.validate_candidate(...) must return a dict "
                "containing 'ok'"
            )
        xml_digest = hashlib.sha1(xml_path.read_bytes()).hexdigest()
        assets = [
            action.asset_id
            for action in sequence
            if hasattr(action, "asset_id")
        ]
        asset_volume = float(
            sum(
                self._asset_volume_by_id.get(asset_id, 0.0)
                for asset_id in assets
            )
        )
        asset_volume_coeff = float(
            getattr(self.bass_config, "asset_volume_coeff", 0.0) or 0.0
        )
        asset_volume_penalty = asset_volume_coeff * asset_volume
        with self._lock:
            best_reference_loss = self._best_checkpoint_score
        context = {
            "task_json": self.task_json,
            "task_module_path": str(self.task_module_path),
            "cache_dir": str(self.cache_dir),
            "output_dir": str(self.output_dir),
            "replay_dir": str(self.replay_dir),
            "xml_path": str(xml_path),
            "low_level_maxiter": self.low_level_maxiter,
            "low_level_num_steps": self.low_level_num_steps,
            "low_level_sub_steps": self.low_level_sub_steps,
            "low_level_timeout": self.low_level_timeout,
            "low_level_numeric_threads": self.low_level_numeric_threads,
            "low_level_grad_clip": self.low_level_grad_clip,
            "low_level_step_scale": self.low_level_step_scale,
            "low_level_redmax_verbose": self.low_level_redmax_verbose,
            "low_level_spawn_method": getattr(
                self,
                "low_level_spawn_method",
                "auto",
            ),
            "optimizer_strategy": getattr(
                self,
                "optimizer_strategy",
                None,
            ),
            "force_connectivity": self.force_connectivity,
            "generic_design_protocol": self.generic_design_protocol,
            "experimental": self.experimental,
            "experimental_alpha": self.experimental_alpha,
            "experimental_beta": self.experimental_beta,
            "experimental_best_loss": best_reference_loss,
            "debug": self.debug,
            "bass_config": self.bass_config,
            "sequence": [action.to_dict() for action in sequence],
        }
        evaluation_identity = build_evaluation_identity(
            repo_root=REPO_ROOT,
            xml_path=xml_path,
            task_module_path=self.task_module_path,
            task_json=self.task_json,
            context=context,
        )
        evaluation_key = evaluation_identity.key
        context["evaluation_identity"] = evaluation_identity.to_dict()
        context["evaluation_key"] = evaluation_key
        with self._lock:
            cached_low_level_score = self._xml_scores.get(evaluation_key)
            if cached_low_level_score is not None:
                regularized_score = (
                    cached_low_level_score + asset_volume_penalty
                    if math.isfinite(cached_low_level_score)
                    else cached_low_level_score
                )
                effective_score = self._apply_current_experimental_gate(
                    regularized_score
                )
                aliased = dict(
                    self._xml_run_results.get(evaluation_key, {})
                )
                aliased["aliased_from_xml_digest"] = xml_digest
                aliased["aliased_from_evaluation_key"] = evaluation_key
                aliased["aliased_run_key"] = aliased.get("run_key")
                aliased["run_key"] = key
                aliased["low_level_score"] = cached_low_level_score
                aliased["asset_volume"] = asset_volume
                aliased["asset_volume_coeff"] = asset_volume_coeff
                aliased["asset_volume_penalty"] = asset_volume_penalty
                aliased["regularized_score"] = effective_score
                aliased["score"] = effective_score
                aliased["loss"] = effective_score
                self._scores[key] = effective_score
                self._run_results[key] = aliased
                print(
                    f"[bass] retry duplicate evaluation key={key} "
                    f"xml_digest={xml_digest[:12]} "
                    f"evaluation_key={evaluation_key[:12]} "
                    f"score={effective_score}",
                    flush=True,
                )
                return float("inf")
            xml_event = self._xml_inflight.get(evaluation_key)
            if xml_event is None:
                xml_event = threading.Event()
                self._xml_inflight[evaluation_key] = xml_event
                xml_owner = True
            else:
                xml_owner = False

        if not xml_owner:
            print(
                f"[bass] retry in-flight duplicate evaluation key={key} "
                f"xml_digest={xml_digest[:12]} "
                f"evaluation_key={evaluation_key[:12]}",
                flush=True,
            )
            return float("inf")

        with self._lock:
            self._eval_count += 1
            eval_count = self._eval_count
            active_xml_evals = len(self._xml_inflight)
        print(
            f"[bass] eval#{eval_count} active={active_xml_evals} key={key} "
            f"evaluation_key={evaluation_key[:12]} "
            f"actions={len(sequence)} assets={assets} xml={xml_path}",
            flush=True,
        )
        try:
            if candidate_validation["ok"]:
                result = optimize_xml(self.task, str(xml_path), context)
            else:
                result = {
                    "score": float("inf"),
                    "loss": float("inf"),
                    "status": "candidate_validation_failed",
                    "error": "candidate_validation_failed",
                }
            if not isinstance(result, dict) or "score" not in result:
                raise ValueError(
                    "canonical optimize_xml(...) must return a dict "
                    "containing 'score'"
                )
            low_level_score = float(result["score"])
            score = (
                low_level_score + asset_volume_penalty
                if math.isfinite(low_level_score)
                else low_level_score
            )
            run_result = dict(result)
            if run_result.get("evaluation_identity") is not None:
                validate_result_identity(
                    run_result,
                    evaluation_identity,
                )
            run_result["low_level_score"] = low_level_score
            run_result["asset_volume"] = asset_volume
            run_result["asset_volume_coeff"] = asset_volume_coeff
            run_result["asset_volume_penalty"] = asset_volume_penalty
            run_result["regularized_score"] = score
            run_result["score"] = score
            run_result["loss"] = score
            run_result["xml_digest"] = xml_digest
            run_result = enrich_evaluation_result(
                run_result,
                identity=evaluation_identity,
            )
            optimizer_result = dict(
                run_result.get("optimizer", {}) or {}
            )
            optimizer_result.update(
                {
                    "maxiter": self.low_level_maxiter,
                    "grad_clip": self.low_level_grad_clip,
                    "step_scale": self.low_level_step_scale,
                    "numeric_threads": (
                        self.low_level_numeric_threads
                    ),
                }
            )
            run_result.update(
                {
                    "task": self.mission_name,
                    "skeleton": tree_to_dict(
                        sequence_to_tree(sequence)
                    ),
                    "skeleton_hash": key,
                    "root_asset_id": (
                        self.bass_config.root_asset_id
                    ),
                    "head_asset_ids": list(assets),
                    "loss_terms": dict(
                        run_result.get("final_terms", {}) or {}
                    ),
                    "optimizer": optimizer_result,
                    "validation": {
                        "candidate": candidate_validation,
                        "design_collision": run_result.get(
                            "design_collision"
                        ),
                    },
                    "diagnostics": self._merged_rollout_diagnostics(
                        run_result
                    ),
                    "xml_path": str(xml_path),
                    "replay_path": run_result.get("rollout_dir"),
                }
            )
            task_evaluation = self.task.bass_evaluation(
                score=score,
                run_result=run_result,
            )
            if task_evaluation is not None:
                if not isinstance(task_evaluation, Mapping):
                    raise TypeError(
                        "task.bass_evaluation(...) must return a mapping "
                        "or None"
                    )
                run_result.update(dict(task_evaluation))
        except BaseException:
            with self._lock:
                xml_done = self._xml_inflight.pop(
                    evaluation_key,
                    None,
                )
                if xml_done is not None:
                    xml_done.set()
            raise

        checkpoint_is_better = False
        with self._lock:
            self._scores[key] = score
            stored_result = {
                "eval_number": eval_count,
                "run_key": key,
                "evaluation_key": evaluation_key,
                "evaluation_identity": evaluation_identity.to_dict(),
                "xml_digest": xml_digest,
                "score": score,
                "loss": score,
                "xml_path": str(xml_path),
                "params": run_result.get("params"),
                "params_path": run_result.get("params_path"),
                "logs_path": run_result.get("logs_path"),
                "finalized_state_path": run_result.get("finalized_state_path"),
                "result": run_result,
            }
            compact_result = self._alias_cache_result(stored_result)
            self._run_results[key] = compact_result
            self._xml_scores[evaluation_key] = low_level_score
            self._xml_run_results[evaluation_key] = self._alias_cache_result(
                stored_result
            )
            xml_done = self._xml_inflight.pop(evaluation_key, None)
            if xml_done is not None:
                xml_done.set()
            reward = float(run_result.get("bass_reward", float("-inf")))
            checkpoint_is_better = (
                math.isfinite(score)
                and (
                    (
                        self.bass_config.reward_mode == "bounded_task"
                        and (
                            reward > self._best_checkpoint_reward + 1e-12
                            or (
                                abs(
                                    reward - self._best_checkpoint_reward
                                )
                                <= 1e-12
                                and score < self._best_checkpoint_score
                            )
                        )
                    )
                    or (
                        self.bass_config.reward_mode != "bounded_task"
                        and score < self._best_checkpoint_score
                    )
                )
            )
            if checkpoint_is_better:
                previous_best_key = getattr(self, "_best_run_key", None)
                self._best_checkpoint_score = score
                self._best_checkpoint_reward = reward
                self._best_run_key = key
                self._best_result = dict(stored_result)
                if (
                    not getattr(self, "_post_search_rerank_enabled", False)
                    and previous_best_key is not None
                    and previous_best_key != key
                ):
                    self._run_results.pop(previous_best_key, None)
        error = run_result.get("error")
        status = "ok" if error is None and math.isfinite(score) else str(error or "nonfinite_score")
        self._append_eval_csv(
            eval_count=eval_count,
            key=key,
            sequence=sequence,
            assets=assets,
            score=score,
            status=status,
            run_result=run_result,
            xml_path=xml_path,
        )
        if checkpoint_is_better:
            checkpoint_lock = getattr(
                self,
                "_checkpoint_lock",
                self._lock,
            )
            with checkpoint_lock:
                with self._lock:
                    still_best = self._best_run_key == key
                if still_best:
                    self._write_best_checkpoint_locked(
                        sequence,
                        key,
                        stored_result,
                    )
        extra = []
        if run_result.get("returncode") is not None:
            extra.append(f"returncode={run_result.get('returncode')}")
        if run_result.get("num_log_entries") is not None:
            extra.append(f"logs={run_result.get('num_log_entries')}")
        if run_result.get("rollout_dir"):
            extra.append(f"rollout_dir={run_result.get('rollout_dir')}")
        extra_text = "" if not extra else " " + " ".join(extra)
        print(
            f"[bass] done eval#{eval_count} key={key} score={score} "
            f"reward={run_result.get('bass_reward', 'legacy')} "
            f"status={status}{extra_text}",
            flush=True,
        )
        return score

    def ranked_sequences(
        self,
        limit: int,
    ) -> list[tuple[int, str, list[Action], float]]:
        """Return finite completed candidates ordered by follower score."""

        with self._lock:
            ranked = sorted(
                (
                    (key, list(self._sequences[key]), float(score))
                    for key, score in self._scores.items()
                    if key in self._sequences and math.isfinite(score)
                ),
                key=lambda item: (item[2], item[0]),
            )
        return [
            (rank, key, sequence, score)
            for rank, (key, sequence, score) in enumerate(
                ranked[: max(0, int(limit))],
                start=1,
            )
        ]

    def evaluate_rerank_variant(
        self,
        sequence: list[Action],
        *,
        variant_name: str,
        task_config_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Re-optimize one completed XML through the canonical runtime."""

        key = self._sequence_key(sequence)
        xml_path = self.xml_for_sequence(sequence)
        task_json = copy.deepcopy(self.task_json)
        task_config = dict(task_json.get("task_config", {}) or {})
        task_config.update(task_config_overrides)
        task_json["task_config"] = task_config
        rerank_task = load_task(self.mission_name, config=task_config)
        candidate_validation = rerank_task.validate_candidate(xml_path)
        if (
            not isinstance(candidate_validation, dict)
            or "ok" not in candidate_validation
        ):
            raise TypeError(
                "task.validate_candidate(...) must return a dict "
                "containing 'ok'"
            )
        if not candidate_validation["ok"]:
            raise ValueError(
                f"rerank candidate validation failed: {candidate_validation}"
            )

        variant_cache_dir = self.cache_dir / "rerank" / variant_name
        variant_output_dir = self.output_dir / "rerank" / variant_name
        variant_replay_dir = self.replay_dir / "rerank" / variant_name
        for directory in (
            variant_cache_dir,
            variant_output_dir,
            variant_replay_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        context = {
            "task_json": task_json,
            "task_module_path": str(self.task_module_path),
            "cache_dir": str(variant_cache_dir),
            "output_dir": str(variant_output_dir),
            "replay_dir": str(variant_replay_dir),
            "low_level_maxiter": self.low_level_maxiter,
            "low_level_num_steps": self.low_level_num_steps,
            "low_level_sub_steps": self.low_level_sub_steps,
            "low_level_timeout": self.low_level_timeout,
            "low_level_numeric_threads": self.low_level_numeric_threads,
            "low_level_grad_clip": self.low_level_grad_clip,
            "low_level_step_scale": self.low_level_step_scale,
            "low_level_redmax_verbose": self.low_level_redmax_verbose,
            "force_connectivity": self.force_connectivity,
            "generic_design_protocol": self.generic_design_protocol,
            "experimental": self.experimental,
            "experimental_alpha": self.experimental_alpha,
            "experimental_beta": self.experimental_beta,
            "debug": self.debug,
        }
        result = optimize_xml(rerank_task, str(xml_path), context)
        if not isinstance(result, dict) or "score" not in result:
            raise ValueError(
                "canonical optimize_xml(...) must return a dict "
                "containing 'score'"
            )

        baseline = self.result_for_sequence(sequence)
        baseline_result = dict(baseline.get("result", {}) or {})
        asset_volume = float(
            baseline_result.get("asset_volume", 0.0) or 0.0
        )
        asset_volume_coeff = float(
            baseline_result.get("asset_volume_coeff", 0.0) or 0.0
        )
        asset_volume_penalty = asset_volume * asset_volume_coeff
        low_level_score = float(result["score"])
        score = (
            low_level_score + asset_volume_penalty
            if math.isfinite(low_level_score)
            else low_level_score
        )
        xml_digest = hashlib.sha1(xml_path.read_bytes()).hexdigest()
        assets = [
            action.asset_id
            for action in sequence
            if hasattr(action, "asset_id")
        ]
        run_result = dict(result)
        rerank_metadata = {
            "variant": variant_name,
            "task_config_overrides": dict(task_config_overrides),
        }
        run_result.update(
            {
                "low_level_score": low_level_score,
                "asset_volume": asset_volume,
                "asset_volume_coeff": asset_volume_coeff,
                "asset_volume_penalty": asset_volume_penalty,
                "regularized_score": score,
                "score": score,
                "loss": score,
                "xml_digest": xml_digest,
                "task": self.mission_name,
                "skeleton": tree_to_dict(sequence_to_tree(sequence)),
                "skeleton_hash": key,
                "root_asset_id": self.bass_config.root_asset_id,
                "head_asset_ids": list(assets),
                "loss_terms": dict(
                    run_result.get("final_terms", {}) or {}
                ),
                "validation": {
                    "candidate": candidate_validation,
                    "design_collision": run_result.get(
                        "design_collision"
                    ),
                },
                "diagnostics": dict(
                    run_result.get("motion_diagnostics", {}) or {}
                ),
                "xml_path": str(xml_path),
                "replay_path": run_result.get("rollout_dir"),
                "post_search_rerank": rerank_metadata,
            }
        )
        stored_result = {
            "eval_number": baseline.get("eval_number"),
            "run_key": key,
            "evaluation_key": run_result.get("evaluation_key"),
            "evaluation_identity": run_result.get(
                "evaluation_identity"
            ),
            "xml_digest": xml_digest,
            "score": score,
            "loss": score,
            "xml_path": str(xml_path),
            "params": run_result.get("params"),
            "params_path": run_result.get("params_path"),
            "logs_path": run_result.get("logs_path"),
            "finalized_state_path": run_result.get(
                "finalized_state_path"
            ),
            "result": run_result,
            "post_search_rerank": rerank_metadata,
        }
        return {
            "run_key": key,
            "variant": variant_name,
            "task_config_overrides": dict(task_config_overrides),
            "score": score,
            "low_level_score": low_level_score,
            "status": (
                "ok"
                if math.isfinite(score) and not run_result.get("error")
                else str(run_result.get("error") or "nonfinite_score")
            ),
            "evaluation_key": run_result.get("evaluation_key"),
            "rollout_dir": run_result.get("rollout_dir"),
            "stored_result": stored_result,
        }

    def select_rerank_result(
        self,
        sequence: list[Action],
        stored_result: dict[str, Any],
    ) -> None:
        """Make a rerank result the exported result for a sequence."""

        key = self._sequence_key(sequence)
        compact_result = dict(stored_result)
        compact_result["params"] = None
        compact_result["result"] = self._compact_run_result(
            dict(stored_result.get("result", {}) or {})
        )
        with self._lock:
            self._scores[key] = float(stored_result["score"])
            self._run_results[key] = compact_result
            self._best_run_key = key
            self._best_result = dict(stored_result)
            self._best_checkpoint_score = min(
                self._best_checkpoint_score,
                float(stored_result["score"]),
            )

    def _write_best_checkpoint_locked(
        self,
        sequence: list[Action],
        key: str,
        run_result: dict[str, Any],
    ) -> None:
        if self.best_xml_path is not None:
            self.xml_for_sequence(sequence, out_path=self.best_xml_path)
        if self.best_run_json_path is None:
            return
        payload = {
            "checkpoint": True,
            "xml_cache_version": XML_CACHE_VERSION,
            "mission_name": self.mission_name,
            "best_xml": None if self.best_xml_path is None else str(self.best_xml_path),
            "cache_dir": str(self.cache_dir),
            "output_dir": str(self.output_dir),
            "replay_dir": str(self.replay_dir),
            "loss": run_result.get("loss"),
            "score": run_result.get("score"),
            "eval_number": run_result.get("eval_number"),
            "run_key": key,
            "evaluation_key": run_result.get("evaluation_key"),
            "evaluation_identity": run_result.get(
                "evaluation_identity"
            ),
            "xml_digest": run_result.get("xml_digest"),
            "params": run_result.get("params") or run_result.get("result", {}).get("params"),
            "evaluation": run_result,
            "bass": {
                "best_score": run_result.get("score"),
                "best_reward": run_result.get("result", {}).get(
                    "bass_reward"
                ),
                "completed_evaluations": len(self._scores),
                "config": self.bass_config,
            },
            "functions": self.task_json.get("functions"),
            "best_sequence": [action.to_dict() for action in sequence],
            "checkpoint_key": key,
        }
        self.best_run_json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.best_run_json_path.with_suffix(self.best_run_json_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
        tmp_path.replace(self.best_run_json_path)


class CalibrationReplayEvaluator:
    """Feed saved physical outcomes through the normal integrated BASS warmup."""

    def __init__(self, evaluator: CachedXmlEvaluator, artifact_path: Path) -> None:
        from bilevel.upper.bass.bayesian_lookahead import load_calibration_artifact

        self.evaluator = evaluator
        self.artifact = load_calibration_artifact(
            artifact_path, expected_task_name=evaluator.mission_name)
        self._lock = threading.Lock()
        columns = self.artifact["demo_replay_columns"]
        values = self.artifact["demo_replay_rows"]
        if any(len(values_row) != len(columns) for values_row in values):
            raise ValueError("calibration artifact has incomplete replay outcomes")
        rows = [dict(zip(columns, values_row)) for values_row in values]
        self.rows = {row["run_key"]: row for row in rows}
        expected = int(self.artifact["warmup_total_outcomes"])
        if len(rows) != expected or len(self.rows) != expected:
            raise ValueError("calibration replay requires distinct outcomes for every warmup candidate")
        self.seen: set[str] = set()
        self.validated = False
        # Keep compact provenance so best_run.json can identify a calibration
        # winner even though its original optimizer files are not packaged.
        evaluator._run_results.update({
            key: {
                "eval_number": int(row["eval_number"]),
                "run_key": key,
                "score": float(row["score"]),
                "loss": float(row["score"]),
                "result": {"calibration_replay": True,
                           "task_success": str(row.get("task_success", "")).lower() == "true",
                           "task_milestone": row.get("task_milestone"),
                           "task_progress": row.get("task_progress")},
            }
            for key, row in self.rows.items()
        })
        if evaluator.eval_csv_path is not None:
            fields = evaluator._eval_csv_fields()
            with evaluator.eval_csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                for row in sorted(rows, key=lambda item: int(item["eval_number"])):
                    writer.writerow({**row, "bass_search_strategy": "calibration_replay"})
        evaluator._eval_count = expected

    def evaluate(self, sequence: list[Action]) -> BASSEvaluation:
        key = self.evaluator._sequence_key(sequence)
        with self._lock:
            if not self.validated:
                row = self.rows.get(key)
                if row is None or key in self.seen:
                    raise ValueError(f"warmup candidate absent or duplicated in calibration artifact: {key}")
                self.seen.add(key)
            else:
                row = None
        if row is None:
            return self.evaluator.evaluate_for_bass(sequence)
        score = float(row["score"])
        valid = row["status"] == "ok" and math.isfinite(score)
        def optional_int(name: str) -> int | None:
            return int(row[name]) if row.get(name) not in (None, "") else None
        def boolean(name: str, default: bool) -> bool:
            value = row.get(name, "")
            return default if value == "" else value.lower() == "true"
        return BASSEvaluation(
            score=score, reward=float(row["bass_reward"]), valid=valid,
            task_milestone=optional_int("task_milestone"),
            task_stage_count=optional_int("task_stage_count"),
            task_progress=float(row.get("task_progress") or 0.0),
            task_success=boolean("task_success", False),
            task_feasible=boolean("task_feasible", True),
        )

    def validate_calibration(self, fitted: dict[str, Any]) -> None:
        if self.seen != self.rows.keys():
            raise ValueError("calibration replay did not consume every saved outcome")
        for field in ("progress_thresholds", "continuation_prior_means",
                      "prior_strengths", "bins_per_stage", "bins_source",
                      "category_counts",
                      "source_accepted_ordinal_rows"):
            if fitted[field] != self.artifact[field]:
                raise ValueError(f"replayed calibration differs from artifact: {field}")
        self.validated = True

    def scheduler_health_snapshot(self) -> dict[str, Any]:
        return self.evaluator.scheduler_health_snapshot()


def _write_debug_artifacts(cache_dir: Path, result: SearchResult, evaluator: CachedXmlEvaluator) -> None:
    payload = {
        "best_score": result.best_score,
        "best_reward": result.best_reward,
        "total_iterations": result.total_iterations,
        "completed_candidates": result.completed_candidates,
        "valid_completed_candidates": result.valid_completed_candidates,
        "rejected_count_mismatch": result.rejected_count_mismatch,
        "best_function_count": result.best_function_count,
        "constraint_diagnostics": result.constraint_diagnostics,
        "best_sequence": [action.to_dict() for action in result.best_sequence],
        "worker_summaries": result.worker_summaries,
    }
    (cache_dir / "debug_bass_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if result.best_sequence:
        tree_payload = tree_to_dict(sequence_to_tree(result.best_sequence))
        (cache_dir / "debug_best_tree.json").write_text(json.dumps(tree_payload, indent=2), encoding="utf-8")
        evaluator.xml_for_sequence(result.best_sequence, name="best")


def _run_post_search_rerank(
    *,
    result: SearchResult,
    evaluator: CachedXmlEvaluator,
    task_json: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run optional top-k optimizer variants without weakening BASS results."""

    config = _post_search_rerank_config(task_json)
    pre_rerank_best_score = float(result.best_score)
    summary: dict[str, Any] = {
        **config,
        "pre_rerank_best_score": pre_rerank_best_score,
        "post_rerank_best_score": pre_rerank_best_score,
        "attempted_evaluations": 0,
        "successful_evaluations": 0,
        "selected": None,
        "evaluations": [],
    }
    if not config["enabled"] or not result.best_sequence:
        return summary

    candidates = evaluator.ranked_sequences(config["top_k"])
    jobs = []
    for rank, run_key, sequence, baseline_score in candidates:
        for variant in config["optimizer_variants"]:
            jobs.append(
                (
                    rank,
                    run_key,
                    sequence,
                    baseline_score,
                    variant,
                )
            )
    summary["attempted_evaluations"] = len(jobs)
    if not jobs:
        return summary

    selected_sequence = None
    selected_stored_result = None
    selected_score = pre_rerank_best_score
    selected_metadata = None
    workers = min(config["workers"], len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_jobs = {
            executor.submit(
                evaluator.evaluate_rerank_variant,
                sequence,
                variant_name=variant["name"],
                task_config_overrides=variant[
                    "task_config_overrides"
                ],
            ): (rank, run_key, sequence, baseline_score, variant)
            for rank, run_key, sequence, baseline_score, variant in jobs
        }
        for future in as_completed(future_jobs):
            rank, run_key, sequence, baseline_score, variant = (
                future_jobs[future]
            )
            record = {
                "candidate_rank": rank,
                "run_key": run_key,
                "baseline_score": baseline_score,
                "variant": variant["name"],
                "task_config_overrides": variant[
                    "task_config_overrides"
                ],
            }
            try:
                evaluated = future.result()
                stored_result = evaluated.pop("stored_result")
                record.update(evaluated)
                if record.get("status") == "ok":
                    summary["successful_evaluations"] += 1
                score = float(record.get("score", float("inf")))
                if math.isfinite(score) and score < selected_score:
                    selected_score = score
                    selected_sequence = sequence
                    selected_stored_result = stored_result
                    selected_metadata = {
                        "candidate_rank": rank,
                        "run_key": run_key,
                        "variant": variant["name"],
                        "task_config_overrides": variant[
                            "task_config_overrides"
                        ],
                        "score": score,
                    }
            except Exception as exc:
                record.update(
                    {
                        "status": "error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            summary["evaluations"].append(record)
            print(
                "[rerank] "
                f"rank={rank} key={run_key} "
                f"variant={variant['name']} "
                f"score={record.get('score')} "
                f"status={record.get('status')}",
                flush=True,
            )

    summary["evaluations"].sort(
        key=lambda entry: (
            int(entry["candidate_rank"]),
            str(entry["variant"]),
        )
    )
    if selected_sequence is not None and selected_stored_result is not None:
        evaluator.select_rerank_result(
            selected_sequence,
            selected_stored_result,
        )
        result.best_sequence = list(selected_sequence)
        result.best_score = selected_score
        result.best_function_count = sum(
            1
            for action in selected_sequence
            if action.kind == "AddLink"
            and bool(getattr(action, "start_function_group", False))
        )
        summary["selected"] = selected_metadata
    summary["post_rerank_best_score"] = float(result.best_score)
    rerank_path = output_dir / "rerank.json"
    rerank_path.parent.mkdir(parents=True, exist_ok=True)
    rerank_path.write_text(
        json.dumps(jsonable(summary), indent=2),
        encoding="utf-8",
    )
    summary["artifact"] = str(rerank_path)
    return summary


def _mission_name(
    task_json: dict[str, Any],
    task: TaskBase,
) -> str:
    return str(
        task_json.get("mission_name")
        or task_json.get("task_name")
        or task.name
    )


def _json_path_value(task_json: dict[str, Any], key: str) -> Any:
    paths = task_json.get("paths", {})
    if isinstance(paths, dict) and key in paths:
        return paths.get(key)
    return task_json.get(key)


def _default_output_paths(args: argparse.Namespace, task_json: dict[str, Any], mission_name: str) -> tuple[Path, Path, Path, Path]:
    output_base = _resolve_path(
        args.output_dir
        or _json_path_value(task_json, "output_dir")
        or "workspace/bilevel/output"
    )
    task_output_dir = output_base / mission_name
    replay_dir = task_output_dir / "replay"
    best_xml_config = args.best_xml_out or _json_path_value(task_json, "best_xml_out")
    best_run_json_config = args.best_run_json or _json_path_value(task_json, "best_run_json")
    best_xml = (
        _resolve_path(best_xml_config, base=REPO_ROOT)
        if best_xml_config
        else task_output_dir / f"{mission_name}_best.xml"
    )
    best_run_json = (
        _resolve_path(best_run_json_config, base=REPO_ROOT)
        if best_run_json_config
        else task_output_dir / "best_run.json"
    )
    return task_output_dir, replay_dir, best_xml, best_run_json


def _clean_run_artifacts(
    *,
    cache_dir: Path,
    replay_dir: Path,
    best_xml_path: Path,
    best_run_json_path: Path,
    eval_csv_path: Path | None = None,
    rerank_output_dir: Path | None = None,
) -> None:
    """Remove stale generated artifacts so each search starts from a clean slate."""
    for directory in (cache_dir, replay_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    if rerank_output_dir is not None and rerank_output_dir.exists():
        shutil.rmtree(rerank_output_dir)
    for path in (best_xml_path, best_run_json_path):
        if path.exists():
            path.unlink()
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
    if eval_csv_path is not None and eval_csv_path.exists():
        eval_csv_path.unlink()


def _acquire_run_lock(*, mission_name: str, cache_dir: Path, output_dir: Path, replay_dir: Path) -> Path:
    """Prevent concurrent runs from clobbering shared cache/output artifacts."""
    lock_root = REPO_ROOT / "tmp" / ".lock"
    lock_root.mkdir(parents=True, exist_ok=True)
    _clean_stale_lock_files(lock_root)
    identity = json.dumps(
        {
            "mission_name": mission_name,
            "cache_dir": str(cache_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "replay_dir": str(replay_dir.resolve()),
        },
        sort_keys=True,
    )
    lock_name = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    lock_path = lock_root / f"{mission_name}_{lock_name}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        holder = handle.read().strip()
        handle.close()
        detail = f" Existing holder: {holder}" if holder else ""
        raise RuntimeError(
            "Another bilevel search is already using the same mission/cache/output paths. "
            "Use different --cache-dir and --output-dir for concurrent experiments, or wait "
            f"for the existing run to finish.{detail}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} mission={mission_name}\n")
    handle.write(f"cache_dir={cache_dir.resolve()}\n")
    handle.write(f"output_dir={output_dir.resolve()}\n")
    handle.write(f"replay_dir={replay_dir.resolve()}\n")
    handle.flush()
    _RUN_LOCK_HANDLES.append(handle)

    def _release_lock() -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass
        try:
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            pass

    atexit.register(_release_lock)
    return lock_path


def _write_best_run_json(
    path: Path,
    *,
    mission_name: str,
    result: SearchResult,
    evaluator: CachedXmlEvaluator,
    best_xml: Path,
    cache_dir: Path,
    output_dir: Path,
    replay_dir: Path,
    config: BASSConfig,
    task_json: dict[str, Any],
    visualization: dict[str, Any],
    post_search_rerank: dict[str, Any] | None = None,
) -> None:
    run_result = evaluator.result_for_sequence(result.best_sequence) if result.best_sequence else {}
    rerank_enabled = bool(
        post_search_rerank
        and post_search_rerank.get("enabled")
    )
    selected_replay_dir = str(replay_dir)
    if rerank_enabled:
        selected_rollout_dir = (
            dict(run_result.get("result", {}) or {}).get("rollout_dir")
            or run_result.get("rollout_dir")
        )
        if selected_rollout_dir:
            selected_replay_dir = str(Path(str(selected_rollout_dir)).parent)
    payload = {
        "mission_name": mission_name,
        "xml_cache_version": XML_CACHE_VERSION,
        "best_xml": str(best_xml),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "replay_dir": selected_replay_dir,
        "eval_number": run_result.get("eval_number"),
        "run_key": run_result.get("run_key"),
        "evaluation_key": run_result.get("evaluation_key"),
        "evaluation_identity": run_result.get("evaluation_identity"),
        "xml_digest": run_result.get("xml_digest"),
        "loss": run_result.get("loss", result.best_score),
        "score": run_result.get("score", result.best_score),
        "params": run_result.get("params") or run_result.get("result", {}).get("params"),
        "evaluation": run_result,
        "bass": {
            "strategy": {"resolved": config.acquisition},
            "best_score": result.best_score,
            "best_reward": result.best_reward,
            "total_iterations": result.total_iterations,
            "completed_candidates": result.completed_candidates,
            "valid_completed_candidates": result.valid_completed_candidates,
            "rejected_count_mismatch": result.rejected_count_mismatch,
            "best_function_count": result.best_function_count,
            "constraint_diagnostics": result.constraint_diagnostics,
            "config": config,
        },
        "functions": task_json.get("functions"),
        "best_sequence": [action.to_dict() for action in result.best_sequence],
        "worker_summaries": result.worker_summaries,
        "visualization": visualization,
    }
    if rerank_enabled:
        payload["post_search_rerank"] = post_search_rerank
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def _visualize_best_xml(
    task: TaskBase,
    xml_path: Path,
    *,
    step: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    status = {"attempted": True, "step": int(step), "ok": False}
    try:
        value = visualize_xml(task, str(xml_path), context)
        if isinstance(value, dict):
            status.update(value)
        status["ok"] = bool(status.get("ok", True))
    except Exception as exc:
        status["ok"] = False
        status["error"] = str(exc)
        status["traceback"] = traceback.format_exc()
        print(f"[WARN] Best XML visualization failed: {exc}", file=sys.stderr)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bilevel BASS/XML search")
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--best-xml-out", default=None)
    parser.add_argument("--best-run-json", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bass-log-interval", type=int, default=None)
    parser.add_argument("--no-visualize-best", action="store_true")
    parser.add_argument("--visualize-step", type=int, default=1000)
    parser.add_argument("--low-level-maxiter", type=int, default=None)
    parser.add_argument("--low-level-num-steps", type=int, default=None)
    parser.add_argument("--low-level-sub-steps", type=int, default=None)
    parser.add_argument("--low-level-timeout", type=float, default=None)
    parser.add_argument("--low-level-numeric-threads", type=int, default=None)
    parser.add_argument("--low-level-grad-clip", type=float, default=None)
    parser.add_argument("--low-level-step-scale", type=float, default=None)
    parser.add_argument(
        "--low-level-spawn-method",
        choices=("auto", "posix_spawn", "popen"),
        default=None,
    )
    parser.add_argument("--low-level-redmax-verbose", action="store_true", default=None)
    from bilevel.lower.optimizers import OPTIMIZER_REGISTRY

    parser.add_argument(
        "--optimizer",
        dest="optimizer_strategy",
        choices=tuple(OPTIMIZER_REGISTRY.names()),
        default=None,
        help="Explicitly override the active lower-level optimizer mode.",
    )
    design_group = parser.add_mutually_exclusive_group()
    design_group.add_argument(
        "--design-optim",
        dest="design_optim",
        action="store_true",
        default=None,
        help="Enable lower-level morphology optimization.",
    )
    design_group.add_argument(
        "--no-design-optim",
        dest="design_optim",
        action="store_false",
        help="Disable morphology optimization and optimize actions only.",
    )
    parser.add_argument("--contact-continuation-scales", type=str, default=None)
    parser.add_argument("--contact-continuation-weights", type=str, default=None)
    parser.add_argument("--force-connectivity", action="store_true", default=None)
    parser.add_argument(
        "--generic-design-protocol",
        choices=("connected_direct_planar_hexahedron",),
        default=None,
    )
    parser.add_argument("--experimental", action="store_true", default=None)
    parser.add_argument("--experimental-alpha", type=float, default=None)
    parser.add_argument("--experimental-beta", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--parallelization-mode", choices=["independent", "shared_tree"], default=None)
    parser.add_argument("--max-head-links", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--iteration-budget", type=int, default=None)
    parser.add_argument("--search-time-budget", "--time-budget", dest="search_time_budget", type=float, default=None)
    parser.add_argument("--calibration-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-actions-per-expansion", type=int, default=None)
    parser.add_argument(
        "--partial-state-transpositions",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--structural-factorization",
        action="store_true",
        default=None,
    )
    scheduler_group = parser.add_mutually_exclusive_group()
    scheduler_group.add_argument(
        "--scheduler-v2",
        dest="scheduler_v2",
        action="store_true",
    )
    scheduler_group.add_argument(
        "--no-scheduler-v2",
        dest="scheduler_v2",
        action="store_false",
    )
    parser.set_defaults(scheduler_v2=None)
    parser.add_argument(
        "--scheduler-health-log-interval",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--scheduler-supply-critical-fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--scheduler-supply-critical-seconds",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--scheduler-supply-recent-new-seconds",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--scheduler-supply-rate-window-seconds",
        type=float,
        default=None,
    )
    materialization_group = parser.add_mutually_exclusive_group()
    materialization_group.add_argument(
        "--scheduler-frontier-only-rollout",
        dest="scheduler_materialize_rollout_suffix",
        action="store_false",
    )
    parser.set_defaults(scheduler_materialize_rollout_suffix=None)
    parser.add_argument("--scheduler-ready-low-watermark-fraction", type=float, default=None)
    parser.add_argument("--scheduler-ready-high-watermark-fraction", type=float, default=None)
    parser.add_argument("--scheduler-occupancy-warning-fraction", type=float, default=None)
    parser.add_argument("--scheduler-occupancy-recovery-fraction", type=float, default=None)
    parser.add_argument("--scheduler-occupancy-warning-seconds", type=float, default=None)
    parser.add_argument("--scheduler-occupancy-recovery-seconds", type=float, default=None)
    parser.add_argument(
        "--acquisition",
        choices=["bass_n1", "bass_n2", "uniform_random"],
        default=None,
    )
    parser.add_argument(
        "--milestone-thresholds",
        type=_parse_json_array,
        default=None,
        help="Fixed per-stage threshold rows as a JSON array of arrays.",
    )
    parser.add_argument(
        "--continuation-prior-means",
        type=_parse_json_array,
        default=None,
        help="Ordinal continuation prior means as a JSON array.",
    )
    parser.add_argument(
        "--prior-strengths",
        type=_parse_json_array,
        default=None,
        help="One or one-per-level BASS Beta concentration as a JSON array.",
    )
    parser.add_argument("--calibration-artifact", default=None)
    parser.add_argument(
        "--calibration-bins-per-stage",
        type=_parse_json_array,
        default=None,
        help="Integrated calibration bin counts, one integer per task stage.",
    )
    parser.add_argument(
        "--calibration-smoothing", type=float, default=None
    )
    parser.add_argument(
        "--calibration-threshold-sample",
        choices=["all", "interior"],
        default=None,
    )
    parser.add_argument(
        "--physical-dedup-mode",
        choices=["off", "observe", "enforce"],
        default=None,
    )
    parser.add_argument("--physical-signature-eps", type=float, default=None)
    parser.add_argument(
        "--structural-dag-path",
        type=str,
        default=None,
        help=(
            "Load an offline-compiled immutable grammar tree and bypass "
            "online candidate generation."
        ),
    )
    structural_dag_mmap_group = parser.add_mutually_exclusive_group()
    structural_dag_mmap_group.add_argument(
        "--structural-dag-mmap",
        dest="structural_dag_mmap",
        action="store_true",
        help="Memory-map structural-dag arrays (default).",
    )
    structural_dag_mmap_group.add_argument(
        "--no-structural-dag-mmap",
        dest="structural_dag_mmap",
        action="store_false",
        help="Load structural-dag arrays into memory.",
    )
    parser.set_defaults(structural_dag_mmap=None)
    parser.add_argument("--bass-diagnostics-jsonl", type=str, default=None)
    parser.add_argument(
        "--bass-diagnostics-node-sample-interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--bass-diagnostics-flush-interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--bass-diagnostics-include-full-signatures",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--rollout-policy",
        choices=["uniform", "depth_biased", "end_biased"],
        default=None,
    )
    parser.add_argument("--rollout-end-prob-by-depth", type=str, default=None)
    parser.add_argument("--rollout-addlink-prob-by-depth", type=str, default=None)
    parser.add_argument("--rollout-branching-penalty-by-depth", type=str, default=None)
    parser.add_argument("--target-function-count", type=int, default=None)
    parser.add_argument("--function-count-margin", type=int, default=None)
    parser.add_argument("--function-group-depth-delta", type=int, default=None)
    parser.add_argument("--root-asset-id", type=str, default=None)
    parser.add_argument("--root-blocked-face", type=int, default=None)
    parser.add_argument(
        "--root-rotation-options",
        type=str,
        default=None,
        help="Comma-separated first-action choices from roll,pitch,yaw.",
    )
    parser.add_argument("--assets-json", type=str, default=None)
    parser.add_argument("--scene-xml", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_json_path = _resolve_path(args.task_json)
    task_json = _load_json(task_json_path)
    configured_mission = task_json.get(
        "mission_name",
        task_json.get("task_name"),
    )
    if not configured_mission:
        raise ValueError(
            "Canonical task config requires mission_name or task_name"
        )
    task = load_task(
        str(configured_mission),
        config=dict(task_json.get("task_config", {})),
    )
    _validate_mission_name(task_json, task)
    mission_name = _mission_name(task_json, task)
    task_config = dict(task.config)
    if args.design_optim is not None:
        task_config["optimize_design"] = bool(args.design_optim)
        task_json = dict(task_json)
        task_json["task_config"] = dict(task_config)
    if args.force_connectivity is not None:
        task_config["force_connectivity"] = bool(args.force_connectivity)
        task_json = dict(task_json)
        task_json["task_config"] = dict(task_config)
    if args.generic_design_protocol is not None:
        task_config["generic_design_protocol"] = str(args.generic_design_protocol)
        task_json = dict(task_json)
        task_json["task_config"] = dict(task_config)
    if args.contact_continuation_scales is not None:
        task_config["contact_continuation_scales"] = _parse_float_list(
            args.contact_continuation_scales
        )
    if args.contact_continuation_weights is not None:
        task_config["contact_continuation_weights"] = _parse_float_list(
            args.contact_continuation_weights
        )
    if args.low_level_spawn_method is not None:
        task_config["low_level_spawn_method"] = str(
            args.low_level_spawn_method
        )
    task_json = dict(task_json)
    task_json["task_config"] = dict(task_config)
    selected_generic_design_protocol = _generic_design_protocol_from_inputs(args, task_json, task_config)
    if task_config.get("generic_design_protocol") != selected_generic_design_protocol:
        task_config["generic_design_protocol"] = selected_generic_design_protocol
        task_json = dict(task_json)
        task_json["task_config"] = dict(task_config)
    task = load_task(mission_name, config=task_config)
    task_module_path = canonical_task_module_path(mission_name)
    post_search_rerank_config = _post_search_rerank_config(task_json)

    assets_json = _resolve_path(
        _value(
            args.assets_json,
            task_json.get("assets_json"),
            task_config.get("assets_json"),
            default="assets/library/catalog.json",
        )
    )
    scene_value = _value(
        args.scene_xml,
        task_json.get("scene_xml"),
        task_config.get("scene_xml"),
    )
    if not scene_value:
        raise ValueError(
            "Canonical bilevel search requires scene_xml in the task config "
            "or via --scene-xml."
        )
    scene_xml = _resolve_path(scene_value)
    output_dir, replay_dir, best_xml_path, best_run_json_path = _default_output_paths(args, task_json, mission_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    best_xml_path.parent.mkdir(parents=True, exist_ok=True)
    best_run_json_path.parent.mkdir(parents=True, exist_ok=True)
    eval_csv_path = output_dir / "evals.csv"
    cache_dir_config = args.cache_dir or _json_path_value(task_json, "cache_dir")
    cache_dir = (
        _resolve_path(cache_dir_config)
        if cache_dir_config
        else REPO_ROOT / "workspace" / "bilevel" / "cache" / mission_name
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_run_lock(
        mission_name=mission_name,
        cache_dir=cache_dir,
        output_dir=output_dir,
        replay_dir=replay_dir,
    )
    if args.debug:
        print(f"[bilevel] acquired run lock={lock_path}", flush=True)
    _clean_run_artifacts(
        cache_dir=cache_dir,
        replay_dir=replay_dir,
        best_xml_path=best_xml_path,
        best_run_json_path=best_run_json_path,
        eval_csv_path=eval_csv_path,
        rerank_output_dir=(
            output_dir / "rerank"
            if post_search_rerank_config["enabled"]
            else None
        ),
    )
    print(f"[bilevel] cleaned cache_dir={cache_dir} replay_dir={replay_dir} eval_csv={eval_csv_path}", flush=True)

    config = _bass_config_from_inputs(args, task_json, task)
    all_assets = load_assets(str(assets_json))
    config.root_asset_id = resolve_asset_id(all_assets, config.root_asset_id)
    asset_selector = task_json.get(
        "asset_selector",
        task_config.get("asset_selector"),
    )
    selected_assets = select_assets(
        all_assets,
        asset_selector,
        required_ids=(config.root_asset_id,),
    )
    assets = _bass_tool_assets(
        selected_assets,
        root_asset_id=config.root_asset_id,
    )
    dropped_assets = [asset.asset_id for asset in all_assets if asset not in assets]
    if dropped_assets:
        print(
            "[bilevel] BASS tool assets="
            f"{[asset.asset_id for asset in assets]} dropped_non_tool_assets={dropped_assets}",
            flush=True,
        )
    xml_options = dict(task_json.get("xml", {}))
    xml_options["functions"] = task_json.get("functions")
    xml_options["contact_model"] = task_config.get("contact_model")
    if args.model_name is not None:
        xml_options["model_name"] = args.model_name
    root_box, forbidden_boxes = None, []
    low_level_maxiter = _value(args.low_level_maxiter, task_json.get("low_level_maxiter"), task_config.get("low_level_maxiter"))
    low_level_num_steps = _value(args.low_level_num_steps, task_json.get("low_level_num_steps"), task_config.get("low_level_num_steps"))
    low_level_sub_steps = _value(args.low_level_sub_steps, task_json.get("low_level_sub_steps"), task_config.get("low_level_sub_steps"))
    low_level_timeout = _value(args.low_level_timeout, task_json.get("low_level_timeout"), task_config.get("low_level_timeout"), default=1800.0)
    low_level_numeric_threads = _value(
        args.low_level_numeric_threads,
        task_json.get("low_level_numeric_threads"),
        task_config.get("low_level_numeric_threads"),
        default=1,
    )
    low_level_grad_clip = _value(args.low_level_grad_clip, task_json.get("low_level_grad_clip"), task_config.get("low_level_grad_clip"))
    low_level_step_scale = _value(args.low_level_step_scale, task_json.get("low_level_step_scale"), task_config.get("low_level_step_scale"))
    low_level_redmax_verbose = bool(
        _value(args.low_level_redmax_verbose, task_json.get("low_level_redmax_verbose"), task_config.get("low_level_redmax_verbose"), default=False)
    )
    force_connectivity = bool(
        _value(args.force_connectivity, task_json.get("force_connectivity"), task_config.get("force_connectivity"), default=False)
    )
    generic_design_protocol = _generic_design_protocol_from_inputs(args, task_json, task_config)
    experimental = bool(_value(args.experimental, task_json.get("experimental"), task_config.get("experimental"), default=False))
    experimental_alpha = float(
        _value(args.experimental_alpha, task_json.get("experimental_alpha"), task_config.get("experimental_alpha"), default=0.5)
    )
    experimental_beta = float(
        _value(args.experimental_beta, task_json.get("experimental_beta"), task_config.get("experimental_beta"), default=1.5)
    )

    evaluator = CachedXmlEvaluator(
        task=task,
        task_json=task_json,
        task_module_path=task_module_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        replay_dir=replay_dir,
        assets_json=assets_json,
        scene_xml=scene_xml,
        bass_config=config,
        xml_options=xml_options,
        low_level_maxiter=None if low_level_maxiter is None else int(low_level_maxiter),
        low_level_num_steps=None if low_level_num_steps is None else int(low_level_num_steps),
        low_level_sub_steps=None if low_level_sub_steps is None else int(low_level_sub_steps),
        low_level_timeout=None if low_level_timeout is None else float(low_level_timeout),
        low_level_numeric_threads=None if low_level_numeric_threads is None else int(low_level_numeric_threads),
        low_level_grad_clip=None if low_level_grad_clip is None else float(low_level_grad_clip),
        low_level_step_scale=None if low_level_step_scale is None else float(low_level_step_scale),
        low_level_redmax_verbose=low_level_redmax_verbose,
        optimizer_strategy=args.optimizer_strategy,
        force_connectivity=force_connectivity,
        generic_design_protocol=generic_design_protocol,
        experimental=experimental,
        experimental_alpha=experimental_alpha,
        experimental_beta=experimental_beta,
        debug=args.debug,
        mission_name=mission_name,
        best_xml_path=best_xml_path,
        best_run_json_path=best_run_json_path,
        eval_csv_path=eval_csv_path,
    )
    bass_evaluator = evaluator.evaluate_for_bass
    replay_settings = task_json.get("demo_calibration_replay")
    if replay_settings:
        replay = CalibrationReplayEvaluator(
            evaluator, Path(replay_settings["calibration_artifact"]),
        )
        bass_evaluator = replay.evaluate
    result = run_bass(
        assets=assets,
        config=config,
        evaluator=bass_evaluator,
        initial_forbidden_boxes=forbidden_boxes,
        initial_root_box=root_box,
    )
    calibration = result.constraint_diagnostics.get("bass_integrated_calibration")
    if calibration:
        from bilevel.upper.bass.bayesian_lookahead import CALIBRATION_FORMAT
        if replay_settings:
            result.constraint_diagnostics["calibration_artifact"] = str(
                replay_settings["calibration_artifact"]
            )
        else:
            artifact = dict(calibration)
            artifact.update(format=CALIBRATION_FORMAT, task_name=mission_name,
                            source_eval_csv=str(eval_csv_path))
            calibration_output = output_dir / "calibration.json"
            calibration_output.write_text(json.dumps(artifact, indent=2) + "\n")
            result.constraint_diagnostics["calibration_artifact"] = str(calibration_output)
    post_search_rerank = _run_post_search_rerank(
        result=result,
        evaluator=evaluator,
        task_json=task_json,
        output_dir=output_dir,
    )
    visualization_status = {
        "initial": {"attempted": False, "ok": False, "step": 0},
        "final": {"attempted": False, "ok": False, "step": int(args.visualize_step)},
    }
    if result.best_sequence:
        evaluator.xml_for_sequence(result.best_sequence, out_path=best_xml_path)
        if not args.no_visualize_best:
            best_evaluation = evaluator.result_for_sequence(result.best_sequence)
            base_visualization_context = {
                "task_json": task_json,
                "cache_dir": str(cache_dir),
                "output_dir": str(output_dir),
                "replay_dir": str(replay_dir),
                "best_xml": str(best_xml_path),
                "best_run_json": str(best_run_json_path),
                "low_level_maxiter": low_level_maxiter,
                "low_level_num_steps": low_level_num_steps,
                "low_level_sub_steps": low_level_sub_steps,
                "low_level_timeout": low_level_timeout,
                "low_level_numeric_threads": low_level_numeric_threads,
                "low_level_grad_clip": low_level_grad_clip,
                "low_level_step_scale": low_level_step_scale,
                "low_level_redmax_verbose": low_level_redmax_verbose,
                "optimizer_strategy": args.optimizer_strategy,
                "force_connectivity": force_connectivity,
                "generic_design_protocol": generic_design_protocol,
                "experimental": experimental,
                "experimental_alpha": experimental_alpha,
                "experimental_beta": experimental_beta,
                "debug": args.debug,
                "evaluation": best_evaluation,
                "bass_config": config,
                "sequence": [action.to_dict() for action in result.best_sequence],
            }
            visualization_status["initial"] = _visualize_best_xml(
                task,
                best_xml_path,
                step=0,
                context={
                    **base_visualization_context,
                    "phase": "initial",
                    "visualize_step": 0,
                },
            )
            visualization_status["final"] = _visualize_best_xml(
                task,
                best_xml_path,
                step=int(args.visualize_step),
                context={
                    **base_visualization_context,
                    "phase": "final",
                    "visualize_step": int(args.visualize_step),
                },
            )
    if args.debug:
        _write_debug_artifacts(cache_dir, result, evaluator)
    _write_best_run_json(
        best_run_json_path,
        mission_name=mission_name,
        result=result,
        evaluator=evaluator,
        best_xml=best_xml_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        replay_dir=replay_dir,
        config=config,
        task_json=task_json,
        visualization=visualization_status,
        post_search_rerank=post_search_rerank,
    )

    print("Bilevel search complete")
    print(f"best_score={result.best_score}")
    print(f"best_reward={result.best_reward}")
    print(f"best_function_count={result.best_function_count}")
    print(f"completed_candidates={result.completed_candidates}")
    print(f"valid_completed_candidates={result.valid_completed_candidates}")
    print(f"rejected_count_mismatch={result.rejected_count_mismatch}")
    print(f"cache_dir={cache_dir}")
    print(f"best_xml={best_xml_path}")
    print(f"best_run_json={best_run_json_path}")
    print(f"eval_csv={eval_csv_path}")
    print(f"visualization_initial_ok={visualization_status.get('initial', {}).get('ok')}")
    print(f"visualization_final_ok={visualization_status.get('final', {}).get('ok')}")
    if post_search_rerank_config["enabled"]:
        print(
            "post_search_rerank_best_score="
            f"{post_search_rerank.get('post_rerank_best_score')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
