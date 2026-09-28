#!/usr/bin/env python
"""Visualize a saved run_main/bilevel result without running optimization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import redmax_py as redmax


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_main import (  # noqa: E402
    RUN_MAIN_TASKS,
    GenericDesignTaskAdapter,
    _default_model_path,
    _write_run_main_render_xml,
)
from bilevel.runner import CoOptRunner, SimRenderer  # noqa: E402
from bilevel.parameterization import (  # noqa: E402
    UNIFIED_PARAMETERIZATION_ID,
    parse_parameter_artifact_metadata,
    read_parameter_artifact_metadata,
    validate_parameter_vector,
)


def _resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_best_run(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"best_run.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _params_from_best_run(payload: dict[str, Any]) -> np.ndarray | None:
    candidates = [
        payload.get("params"),
        payload.get("evaluation", {}).get("params") if isinstance(payload.get("evaluation"), dict) else None,
        payload.get("evaluation", {}).get("result", {}).get("params")
        if isinstance(payload.get("evaluation"), dict) and isinstance(payload.get("evaluation", {}).get("result"), dict)
        else None,
    ]
    for value in candidates:
        if value is not None:
            return np.asarray(value, dtype=np.float64)

    path_candidates = [
        payload.get("params_path"),
        payload.get("evaluation", {}).get("params_path") if isinstance(payload.get("evaluation"), dict) else None,
        payload.get("evaluation", {}).get("result", {}).get("params_path")
        if isinstance(payload.get("evaluation"), dict)
        and isinstance(
            payload.get("evaluation", {}).get("result"),
            dict,
        )
        else None,
    ]
    for path_text in path_candidates:
        path = _resolve(path_text)
        if path is not None and path.exists():
            return np.load(path)
    return None


def _best_run_action_parameterization(payload: dict[str, Any]) -> str | None:
    evaluation = payload.get("evaluation")
    result = evaluation.get("result") if isinstance(evaluation, dict) else None
    for candidate in (payload, evaluation, result):
        if isinstance(candidate, dict) and candidate.get("action_parameterization"):
            return str(candidate["action_parameterization"])
    return None


def _params_file_action_parameterization(params_path: Path) -> str | None:
    metadata = _params_file_metadata(params_path)
    if metadata is None:
        return None
    value = metadata.get("action_parameterization")
    return None if value is None else str(value)


def _params_file_metadata(params_path: Path) -> dict[str, Any] | None:
    candidates = []
    if params_path.stem == "params_initial":
        candidates.append(params_path.parent / "params_initial_meta.json")
    candidates.append(params_path.parent / "params_meta.json")
    for meta_path in candidates:
        if not meta_path.exists():
            continue
        return read_parameter_artifact_metadata(meta_path)
    return None


def _load_params(
    args: argparse.Namespace, best_payload: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any] | None, str | None]:
    params_path = _resolve(args.params)
    if params_path is not None:
        if not params_path.exists():
            raise FileNotFoundError(f"params file not found: {params_path}")
        metadata = _params_file_metadata(params_path)
        return (
            np.load(params_path),
            metadata,
            _params_file_action_parameterization(params_path),
        )

    params = _params_from_best_run(best_payload)
    if params is not None:
        metadata = None
        evaluation = best_payload.get("evaluation")
        result = (
            evaluation.get("result")
            if isinstance(evaluation, dict)
            and isinstance(evaluation.get("result"), dict)
            else None
        )
        for candidate in (best_payload, evaluation, result):
            if not isinstance(candidate, dict):
                continue
            embedded = candidate.get("parameter_artifact_metadata")
            if embedded is not None:
                if not isinstance(embedded, dict):
                    raise ValueError(
                        "parameter_artifact_metadata must be an object"
                    )
                metadata = embedded
                break
            params_path_text = candidate.get("params_path")
            if params_path_text:
                resolved = _resolve(str(params_path_text))
                if resolved is not None:
                    metadata = _params_file_metadata(resolved)
                if metadata is not None:
                    break
        return (
            params,
            metadata,
            _best_run_action_parameterization(best_payload),
        )

    load_dir = _resolve(args.load_dir)
    if load_dir is None:
        raise ValueError("No params source: pass --params, --best-run-json, or --load-dir")
    params_path = load_dir / "params.npy"
    if not params_path.exists():
        raise FileNotFoundError(f"params file not found: {params_path}")
    metadata = _params_file_metadata(params_path)
    return (
        np.load(params_path),
        metadata,
        _params_file_action_parameterization(params_path),
    )


def _load_stage_stop_policy(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, Path | None]:
    if bool(args.ignore_stage_stop):
        return None, None
    diagnostics_path = _resolve(args.diagnostics)
    if diagnostics_path is None and args.params is not None:
        params_path = _resolve(args.params)
        if params_path is not None:
            candidate = params_path.parent / "diagnostics.json"
            if candidate.is_file():
                diagnostics_path = candidate
    if diagnostics_path is None and args.params is None:
        load_dir = _resolve(args.load_dir)
        if load_dir is not None:
            candidate = load_dir / "diagnostics.json"
            if candidate.is_file():
                diagnostics_path = candidate
    if diagnostics_path is None:
        return None, None
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"replay diagnostics not found: {diagnostics_path}"
        )
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    optimizer = payload.get("optimizer")
    details = optimizer.get("details") if isinstance(optimizer, dict) else None
    policy = payload.get("stage_stop")
    if not isinstance(policy, dict) and isinstance(details, dict):
        policy = details.get("stop_policy") or details.get("final_stage_stop")
    if isinstance(policy, dict):
        return dict(policy), diagnostics_path
    return None, diagnostics_path


def _make_task(args: argparse.Namespace):
    if args.task not in RUN_MAIN_TASKS:
        raise KeyError(
            f"Unknown task {args.task!r}. "
            f"Choices: {sorted(RUN_MAIN_TASKS.keys())}"
        )
    task = RUN_MAIN_TASKS[args.task]["factory"](args)
    task = GenericDesignTaskAdapter(
        task,
        freeze_finger_design=True,
        force_connectivity=bool(args.force_connectivity),
        generic_design_protocol=str(args.generic_design_protocol),
    )
    return task


def _model_path(args: argparse.Namespace, best_payload: dict[str, Any]) -> Path:
    explicit = _resolve(args.model_xml)
    if explicit is not None:
        return explicit
    best_xml = _resolve(best_payload.get("best_xml"))
    if best_xml is not None:
        return best_xml
    return Path(_default_model_path(str(REPO_ROOT), args.task)).resolve()


def _build_runner(args: argparse.Namespace, model_path: Path):
    task = _make_task(args)
    sim = redmax.Simulation(str(model_path), False)
    sim.reset()
    runner = CoOptRunner(
        sim,
        task,
        args=args,
        model_path=str(model_path),
        visualize=not bool(args.no_viewer),
        optimize_design=not bool(args.no_design_optim),
        visualize_every_n=None,
        morphology_parameterization=args.morphology_parameterization,
    )
    return runner


def _runner_for_params(
    args: argparse.Namespace,
    model_path: Path,
    params: np.ndarray,
    metadata_payload: dict[str, Any] | None,
):
    runner = _build_runner(args, model_path)
    metadata = parse_parameter_artifact_metadata(metadata_payload)
    if metadata is None:
        expected = (
            runner.action_parameter_dim + runner.ndof_morphology
        )
        if params.ndim == 1 and params.size == expected:
            return runner
        raise ValueError(
            "Saved parameter length differs from the current runner and no "
            "complete metadata identifies the action dimension; refusing to "
            "infer a control horizon from morphology length."
        )

    validate_parameter_vector(params, expected_dim=metadata.total_dim)
    if metadata.action_dim == runner.action_parameter_dim:
        return runner

    if (
        metadata.action_dim <= 0
        or metadata.action_dim % int(runner.ndof_u) != 0
    ):
        raise ValueError(
            f"Saved action_dim {metadata.action_dim} is incompatible with "
            f"ndof_u={runner.ndof_u}."
        )
    inferred_ctrl_steps = metadata.action_dim // int(runner.ndof_u)
    inferred_num_steps = inferred_ctrl_steps * int(args.sub_steps)
    print(
        f"[replay] restored num_steps={inferred_num_steps} from verified "
        f"artifact action_dim={metadata.action_dim}",
        flush=True,
    )
    args.num_steps = int(inferred_num_steps)
    return _build_runner(args, model_path)


def _replay_baked_design(runner: CoOptRunner, params: np.ndarray, *, open_viewer: bool) -> None:
    action, cage = runner.unpack_params(params)
    if cage is None:
        raise ValueError("saved params do not contain cage/design parameters")

    tmp_root = REPO_ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="run_main_final_replay_", dir=str(tmp_root)))
    try:
        design_params, meshes = runner.parameterize_morphology_numpy(
            cage,
            generate_mesh=True,
        )
        render_xml = _write_run_main_render_xml(
            runner.model_path,
            runner,
            meshes,
            str(tmp_dir),
            contact_cap=getattr(runner.args, "visualize_contact_cap", None),
        )
        sim = redmax.Simulation(str(render_xml), False)
        sim.set_design_params(np.asarray(design_params, dtype=np.float64))
        sim.reset()
        runner._reset_staged_motion_stop_runtime(sim=sim)

        u_all = runner.controls_from_action(action)
        u_knots = np.asarray(u_all, dtype=np.float64).reshape(
            runner.num_ctrl_steps,
            runner.ndof_u,
        )
        q_initial = np.asarray(sim.get_q(), dtype=np.float64).copy()
        print(
            "[replay] loaded action:"
            f" latent_norm={float(np.linalg.norm(action)):.9g}"
            f" control_norm={float(np.linalg.norm(u_knots)):.9g}"
            f" nonzero_knots={int(np.count_nonzero(np.linalg.norm(u_knots, axis=1) > 1e-12))}"
            f" control_abs_max={np.max(np.abs(u_knots), axis=0).tolist()}",
            flush=True,
        )
        remaining = max(0, min(int(runner.args.visualize_step), int(runner.num_steps)))
        total = int(remaining)
        ctrl_idx = 0
        t0 = time.time()
        print(
            f"[replay] baked XML: {render_xml} step={total} ctrl_steps={runner.num_ctrl_steps}",
            flush=True,
        )
        while remaining > 0 and ctrl_idx < runner.num_ctrl_steps:
            n = min(int(runner.sub_steps), remaining)
            u_i = u_all[ctrl_idx * runner.ndof_u : (ctrl_idx + 1) * runner.ndof_u]
            s0 = time.time()
            runner.advance_control_step(
                ctrl_idx,
                u_i,
                backward_flag=False,
                verbose=bool(runner.args.verbose),
                sim=sim,
                num_sub_steps=n,
            )
            remaining -= n
            ctrl_idx += 1
            print(
                f"[replay] ctrl {ctrl_idx}/{runner.num_ctrl_steps} "
                f"advanced={total - remaining}/{total} "
                f"dt={time.time() - s0:.3f}s elapsed={time.time() - t0:.3f}s",
                flush=True,
            )

        q_final = np.asarray(sim.get_q(), dtype=np.float64)
        q_delta = q_final - q_initial
        print(
            "[replay] state:"
            f" q_delta_norm={float(np.linalg.norm(q_delta)):.9g}"
            f" q_delta_max={float(np.max(np.abs(q_delta)) if q_delta.size else 0.0):.9g}",
            flush=True,
        )
        if np.linalg.norm(u_knots) > 1e-12 and np.linalg.norm(q_delta) <= 1e-12:
            print(
                "[WARN] Replay loaded nonzero controls but produced no generalized-coordinate motion.",
                flush=True,
            )

        if open_viewer:
            sim.viewer_options.speed = float(runner.args.replay_speed)
            SimRenderer.replay(
                sim,
                record=bool(runner.args.record),
                record_path=runner.args.record_file_name + "_optimized.gif",
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="sweep_balls_bass",
        choices=sorted(RUN_MAIN_TASKS.keys()),
    )
    parser.add_argument("--model-xml", default=None, help="XML to replay. Defaults to task XML or best_run best_xml.")
    parser.add_argument("--load-dir", default="results/tmp", help="Directory containing params.npy.")
    parser.add_argument("--params", default=None, help="Direct path to params.npy.")
    parser.add_argument("--best-run-json", default=None, help="Optional best_run.json containing best_xml and params.")
    parser.add_argument(
        "--diagnostics",
        default=None,
        help="Optional diagnostics.json containing the final stage-stop policy.",
    )
    parser.add_argument(
        "--ignore-stage-stop",
        action="store_true",
        help="Replay raw controls without restoring the optimizer's final stage stop.",
    )
    parser.add_argument("--num-steps", type=int, default=1200)
    parser.add_argument("--sub-steps", type=int, default=30)
    parser.add_argument("--visualize-step", type=int, default=1000)
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0.2,
        help="RedMax viewer playback speed multiplier.",
    )
    parser.add_argument(
        "--visualize-contact-cap",
        type=int,
        default=None,
        help="Visualization-only cap on contact points per body in the temporary baked XML.",
    )
    parser.add_argument("--design-source", choices=("generic",), default="generic")
    parser.add_argument(
        "--generic-design-protocol",
        choices=("connected_direct_planar_hexahedron",),
        default="connected_direct_planar_hexahedron",
    )
    parser.add_argument("--force-connectivity", action="store_true")
    parser.add_argument("--no-design-optim", action="store_true")
    parser.add_argument(
        "--morphology-parameterization",
        choices=(UNIFIED_PARAMETERIZATION_ID,),
        default=None,
        help=(
            "Explicit target morphology layout for replay. The default keeps "
            "the canonical legacy runtime."
        ),
    )
    parser.add_argument("--no-viewer", action="store_true", help="Run final replay setup/forward without opening viewer.")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-file-name", default="record")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxls", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--coef-sweep-pose", type=float, default=1.0)
    parser.add_argument("--coef-push-progress", type=float, default=8.0)
    parser.add_argument("--coef-control", type=float, default=0.01)
    parser.add_argument("--terminal-goal-multiplier", type=float, default=120.0)
    parser.add_argument("--freeform-max-wrench", type=float, nargs=6, default=None)
    parser.add_argument("--freeform-action-latent-bound", type=float, default=None)
    parser.add_argument(
        "--action-parameterization",
        choices=("bounded_linear_v1", "tanh_legacy_v1"),
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    os.chdir(str(REPO_ROOT))
    args = parse_args()
    args.visualize = not bool(args.no_viewer)
    args.visualize_every = None
    args.optimize_maxiter = int(args.max_iters)

    best_payload = _read_best_run(_resolve(args.best_run_json))
    model_path = _model_path(args, best_payload)
    params, metadata_payload, source_parameterization = _load_params(
        args,
        best_payload,
    )
    params = np.asarray(params, dtype=np.float64)
    stage_stop_policy, stage_stop_source = _load_stage_stop_policy(args)

    if not model_path.exists():
        raise FileNotFoundError(f"model XML not found: {model_path}")

    print("[replay] repo_root:", REPO_ROOT, flush=True)
    print("[replay] task:", args.task, flush=True)
    print("[replay] model_xml:", model_path, flush=True)
    print("[replay] params_shape:", params.shape, flush=True)
    print("[replay] visualize_step:", int(args.visualize_step), flush=True)
    print("[replay] force_connectivity:", bool(args.force_connectivity), flush=True)
    print("[replay] viewer:", not bool(args.no_viewer), flush=True)

    runner = _runner_for_params(
        args,
        model_path,
        params,
        metadata_payload,
    )
    params = runner.normalize_loaded_params(
        params,
        metadata_payload,
        legacy_action_parameterization=source_parameterization,
    )
    runner.configure_staged_replay_stop(stage_stop_policy)
    if stage_stop_policy is not None:
        print(
            "[replay] restored stage stop:"
            f" knot={int(stage_stop_policy['stop_knot'])}"
            f" failed_stage={stage_stop_policy.get('failed_stage')}"
            f" source={stage_stop_source}",
            flush=True,
        )

    if runner.optimize_design and runner.design_bundle is not None:
        _replay_baked_design(runner, params, open_viewer=not bool(args.no_viewer))
        return 0

    action, _ = runner.unpack_params(params)
    sim = runner.sim
    sim.reset()
    runner._reset_staged_motion_stop_runtime()
    u_all = runner.controls_from_action(action)
    remaining = max(0, min(int(args.visualize_step), runner.num_steps))
    total = remaining
    ctrl_idx = 0
    while remaining > 0 and ctrl_idx < runner.num_ctrl_steps:
        n = min(runner.sub_steps, remaining)
        runner.advance_control_step(
            ctrl_idx,
            u_all[ctrl_idx * runner.ndof_u : (ctrl_idx + 1) * runner.ndof_u],
            backward_flag=False,
            verbose=bool(args.verbose),
            num_sub_steps=n,
        )
        remaining -= n
        ctrl_idx += 1
        print(f"[replay] ctrl {ctrl_idx}/{runner.num_ctrl_steps} advanced={total - remaining}/{total}", flush=True)
    if not args.no_viewer:
        SimRenderer.replay(sim, record=bool(args.record), record_path=args.record_file_name + "_optimized.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
