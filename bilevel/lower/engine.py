from __future__ import annotations

import atexit
import errno
import itertools
import json
import math
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bilevel.lower.evaluation import redmax_pythonpath_entries
from bilevel.lower.runtime_args import configure_mount_runner_args

REPO_ROOT = Path(__file__).resolve().parents[2]
_ACTIVE_PROCESS_GROUPS: set[int] = set()
_ACTIVE_PROCESS_GROUPS_LOCK = threading.Lock()
_SUBPROCESS_SPAWN_LOCK = threading.Lock()
_LOW_LEVEL_PROCESS_METRICS_LOCK = threading.Lock()
_LOW_LEVEL_PROCESS_METRICS: dict[str, float | int | str] = {
    "active": 0,
    "peak_active": 0,
    "spawn_count": 0,
    "spawn_seconds_total": 0.0,
    "spawn_seconds_max": 0.0,
    "spawn_method": "none",
}

LOW_LEVEL_UNSTABLE_PATTERNS = (
    "Newton method did not converge",
    " g = nan",
    " g = inf",
    "objective = nan",
    "objective = inf",
    "Numerical issue",
)

OBJECTIVE_RE = re.compile(
    r"iteration\s+(\d+)\s*,\s*num_sim\s*=\s*([-+0-9.eE]+)\s*,.*?Objective\s*=\s*([-+0-9.eE]+)"
)
MAX_SUBPROCESS_CAPTURE_CHARS = 200_000


class _PosixSpawnProcess:
    """Small Popen-compatible handle backed by ``os.posix_spawn``.

    ``subprocess.Popen(start_new_session=True)`` uses fork/exec on Python 3.8.
    Forking the long-lived BASS controller becomes increasingly expensive as
    its tree and evaluation registries grow.  ``posix_spawn`` avoids copying
    that address space while retaining a dedicated process group for timeout
    cleanup.
    """

    def __init__(self, pid: int, stdout_fd: int) -> None:
        self.pid = int(pid)
        self.stdout = os.fdopen(
            stdout_fd,
            "r",
            encoding="utf-8",
            errors="replace",
            buffering=1,
        )
        self.returncode: int | None = None
        self._wait_lock = threading.Lock()

    @staticmethod
    def _decode_status(status: int) -> int:
        if hasattr(os, "waitstatus_to_exitcode"):
            return int(os.waitstatus_to_exitcode(status))
        if os.WIFSIGNALED(status):
            return -int(os.WTERMSIG(status))
        return int(os.WEXITSTATUS(status))

    def poll(self) -> int | None:
        with self._wait_lock:
            if self.returncode is not None:
                return self.returncode
            try:
                waited_pid, status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                return self.returncode
            if waited_pid == 0:
                return None
            self.returncode = self._decode_status(status)
            return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.poll()
            if result is not None:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(["posix_spawn", self.pid], timeout)
            time.sleep(0.01)

    def close(self) -> None:
        try:
            self.stdout.close()
        except OSError:
            pass


def _record_process_spawn(method: str, elapsed: float) -> None:
    with _LOW_LEVEL_PROCESS_METRICS_LOCK:
        metrics = _LOW_LEVEL_PROCESS_METRICS
        metrics["active"] = int(metrics["active"]) + 1
        metrics["peak_active"] = max(
            int(metrics["peak_active"]),
            int(metrics["active"]),
        )
        metrics["spawn_count"] = int(metrics["spawn_count"]) + 1
        metrics["spawn_seconds_total"] = (
            float(metrics["spawn_seconds_total"]) + float(elapsed)
        )
        metrics["spawn_seconds_max"] = max(
            float(metrics["spawn_seconds_max"]),
            float(elapsed),
        )
        metrics["spawn_method"] = str(method)


def _record_process_exit() -> None:
    with _LOW_LEVEL_PROCESS_METRICS_LOCK:
        _LOW_LEVEL_PROCESS_METRICS["active"] = max(
            0,
            int(_LOW_LEVEL_PROCESS_METRICS["active"]) - 1,
        )


def low_level_process_metrics() -> dict[str, float | int | str]:
    """Return process-admission metrics safe for live scheduler diagnostics."""

    with _LOW_LEVEL_PROCESS_METRICS_LOCK:
        result = dict(_LOW_LEVEL_PROCESS_METRICS)
    count = int(result["spawn_count"])
    result["spawn_seconds_mean"] = (
        float(result["spawn_seconds_total"]) / float(count)
        if count
        else 0.0
    )
    try:
        statm = Path("/proc/self/statm").read_text(
            encoding="ascii"
        ).split()
        result["controller_rss_bytes"] = (
            int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        )
    except (IndexError, OSError, ValueError):
        pass
    return result


def _spawn_with_posix_spawn(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
) -> _PosixSpawnProcess:
    """Spawn one isolated worker without forking the large controller."""

    read_fd, write_fd = os.pipe()
    try:
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, True)
        file_actions = [
            (os.POSIX_SPAWN_DUP2, write_fd, 1),
            (os.POSIX_SPAWN_DUP2, write_fd, 2),
            (os.POSIX_SPAWN_CLOSE, read_fd),
            (os.POSIX_SPAWN_CLOSE, write_fd),
        ]
        # Python 3.8's posix_spawn has no cwd file action.  The shell only
        # performs chdir and immediately execs the requested Python worker.
        argv = [
            "/bin/sh",
            "-c",
            'cd "$1" && shift && exec "$@"',
            "low-level-posix-spawn",
            str(cwd),
            *cmd,
        ]
        pid = os.posix_spawn(
            "/bin/sh",
            argv,
            env,
            file_actions=file_actions,
            setpgroup=0,
        )
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    os.close(write_fd)
    return _PosixSpawnProcess(pid, read_fd)


def _spawn_low_level_process(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    method: str,
) -> tuple[Any, str]:
    """Spawn one worker through the configured admission implementation."""

    normalized = str(method).strip().lower().replace("-", "_")
    if normalized not in {"auto", "posix_spawn", "popen"}:
        raise ValueError(
            "low_level_spawn_method must be auto, posix_spawn, or popen"
        )
    if normalized in {"auto", "posix_spawn"} and hasattr(os, "posix_spawn"):
        try:
            return _spawn_with_posix_spawn(cmd, cwd=cwd, env=env), "posix_spawn"
        except (NotImplementedError, OSError) as exc:
            error_number = getattr(exc, "errno", None)
            if normalized == "posix_spawn" or error_number not in {
                errno.ENOSYS,
                errno.EINVAL,
                errno.ENOTSUP,
                None,
            }:
                raise
    with _SUBPROCESS_SPAWN_LOCK:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
    return proc, "popen"


def _parameter_metadata_from_evaluation(
    evaluation: dict[str, Any],
) -> dict[str, Any] | None:
    """Find a parameter sidecar without adding fields to result dictionaries."""

    from bilevel.parameterization import read_parameter_artifact_metadata

    nested = (
        evaluation.get("result")
        if isinstance(evaluation.get("result"), dict)
        else None
    )
    for candidate in (evaluation, nested):
        if not isinstance(candidate, dict):
            continue
        embedded = candidate.get("parameter_artifact_metadata")
        if embedded is not None:
            if not isinstance(embedded, dict):
                raise ValueError(
                    "embedded parameter_artifact_metadata must be an object"
                )
            return embedded
        params_path = candidate.get("params_path")
        if params_path:
            sidecar = Path(str(params_path)).parent / "params_meta.json"
            payload = read_parameter_artifact_metadata(sidecar)
            if payload is not None:
                return payload
        rollout_dir = candidate.get("rollout_dir")
        if rollout_dir:
            payload = read_parameter_artifact_metadata(
                Path(str(rollout_dir)) / "params_meta.json"
            )
            if payload is not None:
                return payload
    return None


def _validated_replay_action_dim(
    params: np.ndarray,
    metadata_payload: dict[str, Any] | None,
) -> int | None:
    """Return an explicit source action dimension, never a length guess."""

    from bilevel.parameterization import (
        parse_parameter_artifact_metadata,
        validate_parameter_vector,
    )

    metadata = parse_parameter_artifact_metadata(metadata_payload)
    if metadata is None:
        return None
    validate_parameter_vector(params, expected_dim=metadata.total_dim)
    return metadata.action_dim


def _runner_class(config: dict[str, Any]):
    """Select the normal or fixed-root morphology runner explicitly."""

    from bilevel.runner import CoOptRunner

    if not bool(config.get("preserve_handle_mount", False)):
        return CoOptRunner
    from bilevel.parameterization.mount import MountPreservingCoOptRunner

    return MountPreservingCoOptRunner


def _morphology_parameterization(
    config: dict[str, Any],
    context: dict[str, Any],
) -> str | None:
    value = context.get(
        "morphology_parameterization",
        config.get("morphology_parameterization"),
    )
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _active_morphology_parameterization(
    config: dict[str, Any],
    context: dict[str, Any],
    *,
    optimize_design: bool,
) -> str | None:
    if not optimize_design:
        return None
    return _morphology_parameterization(config, context)


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Avoid empty/partial result files when a low-level process is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def _cleanup_active_process_groups() -> None:
    with _ACTIVE_PROCESS_GROUPS_LOCK:
        pgids = list(_ACTIVE_PROCESS_GROUPS)
        _ACTIVE_PROCESS_GROUPS.clear()
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            continue


def terminate_active_low_level_processes(
    *,
    reason: str = "scheduler_abort",
) -> int:
    """Request termination of all RedMax groups owned by this controller."""

    del reason  # Reserved for structured engine logging at the caller.
    with _ACTIVE_PROCESS_GROUPS_LOCK:
        pgids = list(_ACTIVE_PROCESS_GROUPS)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            continue
    return len(pgids)


atexit.register(_cleanup_active_process_groups)


def _parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = [float(part) for part in text.split()]
    return np.asarray(values, dtype=np.float64)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_matrix(pos_text: str | None, quat_text: str | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_matrix(_parse_vec(quat_text, (1.0, 0.0, 0.0, 0.0)))
    transform[:3, 3] = _parse_vec(pos_text, (0.0, 0.0, 0.0))
    return transform


def _read_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No vertices found in OBJ mesh: {path}")
    return np.asarray(vertices, dtype=np.float64)


def _aabb_from_vertices(vertices: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom = np.ones((vertices.shape[0], 4), dtype=np.float64)
    hom[:, :3] = vertices
    world = (transform @ hom.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def _aabb_for_body(body: ET.Element, transform: np.ndarray, xml_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    body_type = body.attrib.get("type", "")
    if body_type == "abstract":
        mesh_attr = body.attrib.get("mesh")
        if not mesh_attr:
            return None
        mesh_path = (xml_dir / mesh_attr).resolve()
        return _aabb_from_vertices(_read_obj_vertices(mesh_path), transform)
    if body_type == "sphere":
        radius = float(body.attrib.get("radius", "0"))
        center = transform[:3, 3]
        return center - radius, center + radius
    if body_type == "cuboid":
        size = _parse_vec(body.attrib.get("size"), (0.0, 0.0, 0.0))
        half = size * 0.5
        corners = np.asarray(
            [
                [sx * half[0], sy * half[1], sz * half[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        return _aabb_from_vertices(corners, transform)
    return None


def _collect_body_aabbs(root: ET.Element, xml_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    body_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def visit_link(link: ET.Element, parent_transform: np.ndarray) -> None:
        joint = link.find("joint")
        link_transform = parent_transform
        if joint is not None:
            link_transform = parent_transform @ _transform_matrix(joint.attrib.get("pos"), joint.attrib.get("quat"))
        for body in link.findall("body"):
            body_name = body.attrib.get("name")
            if not body_name:
                continue
            body_transform = link_transform @ _transform_matrix(body.attrib.get("pos"), body.attrib.get("quat"))
            aabb = _aabb_for_body(body, body_transform, xml_dir)
            if aabb is not None:
                body_aabbs[body_name] = aabb
        for child_link in link.findall("link"):
            visit_link(child_link, link_transform)

    for robot in root.findall("robot"):
        for link in robot.findall("link"):
            visit_link(link, np.eye(4, dtype=np.float64))
    return body_aabbs


def _is_generated_tool_body(name: str) -> bool:
    return name.startswith("body_tool_") or name.startswith("body_tip_")


def _generated_tool_structure(root: ET.Element) -> tuple[set[str], set[tuple[str, str]]]:
    """Return generated bodies and directly connected body pairs."""
    bodies: set[str] = set()
    adjacent: set[tuple[str, str]] = set()

    def visit(link: ET.Element, parent_generated_body: str | None) -> None:
        current = [
            body.attrib.get("name", "")
            for body in link.findall("body")
            if _is_generated_tool_body(body.attrib.get("name", ""))
        ]
        for body_name in current:
            bodies.add(body_name)
            if parent_generated_body:
                adjacent.add(tuple(sorted((parent_generated_body, body_name))))
        next_parent = current[0] if current else parent_generated_body
        for child in link.findall("link"):
            visit(child, next_parent)

    for robot in root.findall("robot"):
        for link in robot.findall("link"):
            visit(link, None)
    return bodies, adjacent


def preflight_xml_contact_overlaps(
    xml_path: str | Path,
    *,
    overlap_epsilon: float = 1e-6,
    max_report: int = 12,
) -> dict[str, Any]:
    """Reject XMLs that start with positive-volume overlap on explicit contact pairs.

    RedMax becomes very fragile when a contact pair is initialized in deep
    penetration. This is intentionally an AABB preflight, not a mesh-boolean
    test: it is cheap, deterministic, and conservative enough to catch the
    bad rollouts before the native simulator is launched.
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    body_aabbs = _collect_body_aabbs(root, xml_path.resolve().parent)
    overlaps: list[dict[str, Any]] = []
    contact = root.find("contact")
    pair_constraints = [] if contact is None else (
        list(contact.findall("general_contact")) + list(contact.findall("collision_constraint"))
    )
    checked_pairs: set[tuple[str, str]] = set()

    def check_pair(body1: str, body2: str, kind: str) -> None:
        pair = tuple(sorted((body1, body2)))
        if pair in checked_pairs or body1 not in body_aabbs or body2 not in body_aabbs:
            return
        checked_pairs.add(pair)
        lo1, hi1 = body_aabbs[body1]
        lo2, hi2 = body_aabbs[body2]
        overlap = np.minimum(hi1, hi2) - np.maximum(lo1, lo2)
        if np.all(overlap > overlap_epsilon):
            overlaps.append(
                {
                    "body1": body1,
                    "body2": body2,
                    "kind": kind,
                    "overlap": [float(v) for v in overlap],
                    "volume": float(np.prod(overlap)),
                }
            )

    for elem in pair_constraints:
        body1 = elem.attrib.get("body1")
        body2 = elem.attrib.get("body2")
        if not body1 or not body2:
            continue
        if not (_is_generated_tool_body(body1) or _is_generated_tool_body(body2)):
            continue
        check_pair(body1, body2, elem.tag)

    generated_bodies, adjacent_pairs = _generated_tool_structure(root)
    for body1, body2 in itertools.combinations(sorted(generated_bodies), 2):
        if (body1, body2) in adjacent_pairs:
            continue
        check_pair(body1, body2, "tool_tool_structural")
    overlaps.sort(key=lambda item: item["volume"], reverse=True)
    return {
        "ok": not overlaps,
        "overlaps": overlaps[:max_report],
        "num_overlaps": len(overlaps),
        "checked_pairs": len(checked_pairs),
        "overlap_epsilon": float(overlap_epsilon),
    }


def run_low_level_optimization(
    *,
    xml_path: str,
    context: dict,
    create_task: Callable[[dict], Any],
    config_from_context: Callable[[dict], dict],
    runner_args: Callable[[dict, dict, Path], Any],
    motion_diagnostics: Callable[[Any, np.ndarray], dict] | None = None,
) -> dict:
    """Run one RedMax lower-level shape-action co-refinement for a generated XML."""
    import redmax_py as redmax
    from bilevel.runner import initialize_action

    cfg = config_from_context(context)
    replay_dir = Path(
        context.get(
            "replay_dir",
            context.get(
                "cache_dir",
                "workspace/bilevel/output/task/replay",
            ),
        )
    )
    rollout_dir = replay_dir / Path(xml_path).stem
    rollout_dir.mkdir(parents=True, exist_ok=True)

    local_task = create_task(cfg)
    args = configure_mount_runner_args(
        runner_args(cfg, context, rollout_dir),
        cfg,
        context,
    )
    # RedMax's native constructor verbose path can segfault for generated
    # free3d/freeform XMLs. Keep construction quiet and use Python-side logs.
    sim = redmax.Simulation(xml_path, False)
    sim.reset()

    optimize_design = bool(cfg.get("optimize_design", False))
    runner = _runner_class(cfg)(
        sim,
        local_task,
        args=args,
        model_path=xml_path,
        visualize=False,
        optimize_design=optimize_design,
        morphology_parameterization=_active_morphology_parameterization(
            cfg,
            context,
            optimize_design=optimize_design,
        ),
    )
    action_initialization = {
        "mode": str(cfg.get("low_level_action_init_mode", "task")),
        "scale": float(cfg.get("low_level_action_init_scale", 1.0)),
        "smoothing_passes": int(
            cfg.get("low_level_action_init_smoothing_passes", 0)
        ),
        "seed": int(cfg.get("low_level_seed", 0)),
    }
    action0 = initialize_action(
        local_task,
        sim.ndof_u,
        runner.num_ctrl_steps,
        seed=action_initialization["seed"],
        mode=action_initialization["mode"],
        scale=action_initialization["scale"],
        smoothing_passes=action_initialization["smoothing_passes"],
    )
    cage0 = (
        runner.initial_morphology_parameters
        if optimize_design and runner.design_bundle is not None
        else None
    )
    params0 = runner.pack_params(action0, cage0)
    runner.save_initial(str(rollout_dir), params0)
    if hasattr(runner, "_design_collision_report_for_params"):
        try:
            initial_report = runner._design_collision_report_for_params(params0)
            initial_collision = initial_report.to_dict() if initial_report is not None else None
        except Exception as exc:
            initial_collision = {
                "ok": False,
                "error": "design_collision_check_failed",
                "exception": repr(exc),
            }
        if initial_collision is not None and not bool(initial_collision.get("ok", True)):
            return {
                "score": float("inf"),
                "loss": float("inf"),
                "error": "design_collision_invalid_initial",
                "design_collision": initial_collision,
                "action_initialization": action_initialization,
                "params": params0.tolist(),
                "action_parameterization": runner.action_parameterization,
                "rollout_dir": str(rollout_dir),
                "num_steps": int(local_task.num_steps()),
                "sub_steps": int(local_task.sub_steps()),
            }
    initial_loss, initial_terms = runner.forward(params0, backward_flag=False)

    params_opt = runner.optimize(params0)
    final_loss, final_terms = runner.forward(params_opt, backward_flag=False)
    design_collision_report = None
    if hasattr(runner, "_design_collision_report_for_params"):
        try:
            report = runner._design_collision_report_for_params(params_opt)
            design_collision_report = report.to_dict() if report is not None else None
        except Exception as exc:
            design_collision_report = {"ok": False, "error": "design_collision_check_failed", "exception": repr(exc)}
        if design_collision_report is not None and not bool(design_collision_report.get("ok", True)):
            return {
                "score": float("inf"),
                "loss": float("inf"),
                "error": "design_collision_invalid",
                "initial_loss": float(initial_loss),
                "initial_terms": {key: float(value) for key, value in initial_terms.items()},
                "final_terms": {key: float(value) for key, value in final_terms.items()},
                "design_collision": design_collision_report,
                "action_initialization": action_initialization,
                "params": params_opt.tolist(),
                "action_parameterization": runner.action_parameterization,
                "rollout_dir": str(rollout_dir),
                "num_steps": int(local_task.num_steps()),
                "sub_steps": int(local_task.sub_steps()),
            }
    diagnostics = motion_diagnostics(runner, params_opt) if motion_diagnostics is not None else {}
    if hasattr(runner, "mount_integrity_report"):
        diagnostics = dict(diagnostics or {})
        diagnostics["mount_integrity"] = runner.mount_integrity_report(
            params_opt
        )
    runner.save(str(rollout_dir), params_opt)

    action_opt, cage_opt = runner.unpack_params(params_opt)
    finalized_state_path = rollout_dir / "finalized_state.npz"
    parameter_artifact_metadata = (
        runner.parameter_artifact_metadata().to_dict()
    )
    finalized_payload = {
        "params": params_opt,
        "action_params": action_opt,
        "action_parameterization": np.asarray(
            runner.action_parameterization
        ),
        "parameter_artifact_metadata": np.asarray(
            json.dumps(
                parameter_artifact_metadata,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "loss": np.asarray([final_loss], dtype=np.float64),
    }
    if cage_opt is not None:
        finalized_payload["morphology_params"] = cage_opt
        # Compatibility key retained until downstream readers migrate.
        finalized_payload["cage_params"] = cage_opt
    if (
        optimize_design
        and runner.design_bundle is not None
        and cage_opt is not None
        and hasattr(runner.design_bundle, "design_np")
    ):
        task_json = context.get("task_json") if isinstance(context.get("task_json"), dict) else {}
        task_config = task_json.get("task_config") if isinstance(task_json.get("task_config"), dict) else {}
        save_finalized_meshes = bool(
            context.get(
                "save_finalized_meshes",
                task_json.get("save_finalized_meshes", task_config.get("save_finalized_meshes", False)),
            )
        )
        if save_finalized_meshes:
            design_params, meshes = runner.parameterize_morphology_numpy(
                cage_opt,
                generate_mesh=True,
            )
        else:
            design_params = runner.parameterize_morphology_numpy(
                cage_opt,
                generate_mesh=False,
            )
            meshes = []
        sim.set_design_params(design_params)
        finalized_payload["design_params"] = design_params
        if save_finalized_meshes:
            for mesh_idx, mesh in enumerate(meshes):
                finalized_payload[f"mesh_{mesh_idx}_V"] = mesh.V
                finalized_payload[f"mesh_{mesh_idx}_F"] = mesh.F
    elif optimize_design and runner.design_bundle is not None and cage_opt is not None:
        design_params, _ = runner.apply_morphology(
            cage_opt,
            generate_mesh=False,
        )
        finalized_payload["design_params"] = design_params
    np.savez(str(finalized_state_path), **finalized_payload)

    logs = np.asarray(runner.f_log) if runner.f_log else np.zeros((0, 2))
    return {
        "score": float(final_loss),
        "loss": float(final_loss),
        "initial_loss": float(initial_loss),
        "initial_terms": {key: float(value) for key, value in initial_terms.items()},
        "final_terms": {key: float(value) for key, value in final_terms.items()},
        "motion_diagnostics": diagnostics,
        "design_collision": design_collision_report,
        "action_initialization": action_initialization,
        "params": params_opt.tolist(),
        "action_parameterization": runner.action_parameterization,
        "action": action_opt.tolist(),
        "morphology": (
            [] if cage_opt is None else cage_opt.tolist()
        ),
        "morphology_parameterization": (
            parameter_artifact_metadata.get(
                "morphology_parameterization"
            )
        ),
        "parameter_artifact_metadata": parameter_artifact_metadata,
        "params_path": str(rollout_dir / "params.npy"),
        "logs_path": str(rollout_dir / "logs.npy"),
        "finalized_state_path": str(finalized_state_path),
        "finalized_meshes_saved": bool(
            any(key.startswith("mesh_") for key in finalized_payload)
        ),
        "rollout_dir": str(rollout_dir),
        "num_log_entries": int(logs.shape[0]),
        "optimizer": {
            **dict(getattr(runner, "_optimizer_metadata", None) or {}),
            "details": dict(
                getattr(runner, "_action_optimizer_diagnostics", None)
                or {}
            ),
        },
        "stage_loss_diagnostics": dict(
            getattr(runner, "_last_forward_diagnostics", None) or {}
        ),
        "optimize_maxiter": int(args.optimize_maxiter),
        "grad_clip": float(getattr(args, "grad_clip", 0.0) or 0.0),
        "step_scale": float(getattr(args, "step_scale", 1.0) or 1.0),
        "design_fd_grad": bool(getattr(args, "design_fd_grad", False)),
        "design_fd_step": float(getattr(args, "design_fd_step", 0.0) or 0.0),
        "redmax_verbose": bool(getattr(args, "verbose", False)),
        "num_steps": int(local_task.num_steps()),
        "sub_steps": int(local_task.sub_steps()),
        "num_ctrl_steps": int(runner.num_ctrl_steps),
        "ndof_u": int(runner.ndof_u),
        "ndof_cage": int(runner.ndof_cage),
    }


def optimize_xml_subprocess(
    *,
    task_name: str,
    task_module_path: str,
    xml_path: str,
    context: dict,
    low_level_maxiter_default: int = 100,
) -> dict:
    """Evaluate one XML in a subprocess so native simulator crashes are isolated."""
    from bilevel.lower.evaluation import (
        build_evaluation_identity,
        enrich_evaluation_result,
    )
    from bilevel.parameterization import read_parameter_artifact_metadata

    context = dict(context)
    identity = build_evaluation_identity(
        repo_root=REPO_ROOT,
        xml_path=xml_path,
        task_module_path=context.get(
            "task_module_path",
            task_module_path,
        ),
        task_json=(
            context.get("task_json")
            if isinstance(context.get("task_json"), dict)
            else {}
        ),
        context=context,
    )
    context["evaluation_identity"] = identity.to_dict()
    context["evaluation_key"] = identity.key
    replay_dir = Path(
        context.get(
            "replay_dir",
            context.get(
                "cache_dir",
                "workspace/bilevel/output/task/replay",
            ),
        )
    )
    rollout_dir = replay_dir / Path(xml_path).stem
    rollout_dir.mkdir(parents=True, exist_ok=True)
    request_path = rollout_dir / "low_level_request.json"
    result_path = rollout_dir / "low_level_result.json"

    def finalized_result(payload: dict[str, Any]) -> dict[str, Any]:
        metadata = read_parameter_artifact_metadata(
            rollout_dir / "params_meta.json"
        )
        if metadata is None:
            metadata = read_parameter_artifact_metadata(
                rollout_dir / "params_initial_meta.json"
            )
        enriched = enrich_evaluation_result(
            payload,
            identity=identity,
            metadata_payload=metadata,
        )
        _write_json_atomic(result_path, enriched)
        return enriched
    task_json = context.get("task_json")
    if not isinstance(task_json, dict):
        task_json = {}
    task_config = task_json.get("task_config")
    if not isinstance(task_config, dict):
        task_config = {}
    if bool(task_config.get("preserve_handle_mount", False)):
        try:
            from bilevel.lower.feasibility import (
                check_initial_feasibility,
            )
            from bilevel.lower.proxy_dynamics import (
                PROXY_POLICY,
                audit_proxy_dynamics,
            )

            proxy_dynamics = audit_proxy_dynamics(ET.parse(xml_path).getroot())
        except Exception as exc:
            fixed_root_result = {
                "score": float("inf"),
                "loss": float("inf"),
                "error": "fixed_root_validation_failed",
                "exception": repr(exc),
                "rollout_dir": str(rollout_dir),
            }
            return finalized_result(fixed_root_result)
        if not proxy_dynamics["ok"]:
            fixed_root_result = {
                "score": float("inf"),
                "loss": float("inf"),
                "error": "handle_root_proxy_dynamics_invalid",
                "required_proxy_policy": PROXY_POLICY,
                "proxy_dynamics": proxy_dynamics,
                "rollout_dir": str(rollout_dir),
            }
            return finalized_result(fixed_root_result)
        feasibility = check_initial_feasibility(
            xml_path,
            ground_tolerance=float(
                task_config.get("handle_root_initial_ground_tolerance", 1e-6)
            ),
            sphere_tolerance=float(
                task_config.get("handle_root_initial_sphere_tolerance", 1e-6)
            ),
        )
        if not feasibility["ok"]:
            fixed_root_result = {
                "score": float("inf"),
                "loss": float("inf"),
                "error": "initial_handle_root_feasibility_failed",
                "initial_feasibility": feasibility,
                "proxy_dynamics": proxy_dynamics,
                "rollout_dir": str(rollout_dir),
            }
            return finalized_result(fixed_root_result)
    if not bool(context.get("skip_xml_preflight", False)):
        try:
            preflight = preflight_xml_contact_overlaps(
                xml_path,
                overlap_epsilon=float(context.get("xml_preflight_overlap_epsilon", 1e-6)),
            )
        except Exception as exc:
            preflight_result = {
                "score": float("inf"),
                "loss": float("inf"),
                "error": "xml_preflight_failed",
                "exception": str(exc),
                "rollout_dir": str(rollout_dir),
            }
            return finalized_result(preflight_result)
        if not preflight.get("ok", True):
            preflight_result = {
                "score": float("inf"),
                "loss": float("inf"),
                "error": "xml_preflight_contact_overlap",
                "preflight": preflight,
                "rollout_dir": str(rollout_dir),
            }
            finalized = finalized_result(preflight_result)
            if bool(context.get("debug", False)):
                print(
                    "[low-level] rejected XML before RedMax: "
                    f"{preflight.get('num_overlaps', len(preflight.get('overlaps', [])))} contact AABB overlaps",
                    flush=True,
                )
            return finalized
    request = {
        "task_name": str(task_name),
        "task_module": context.get(
            "task_module_path",
            task_module_path,
        ),
        "xml_path": xml_path,
        "context": context,
    }
    request_path.write_text(json.dumps(request, indent=2, default=str), encoding="utf-8")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "bilevel.lower.subprocess_worker",
        "--request-json",
        str(request_path),
        "--result-json",
        str(result_path),
    ]
    timeout = context.get("low_level_timeout")
    if timeout is not None and float(timeout) <= 0.0:
        timeout = None
    debug = bool(context.get("debug", False))
    env = os.environ.copy()
    pythonpath_entries = [
        str(REPO_ROOT),
        *redmax_pythonpath_entries(REPO_ROOT),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    numeric_threads = str(context.get("low_level_numeric_threads") or os.environ.get("BILEVEL_NUMERIC_THREADS") or "1")
    env["BILEVEL_NUMERIC_THREADS"] = numeric_threads
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = numeric_threads
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    env["KMP_INIT_AT_FORK"] = "FALSE"
    env["KMP_BLOCKTIME"] = "0"
    env.setdefault("MALLOC_ARENA_MAX", "2")
    stdout_chunks: deque[str] = deque()
    stdout_chars = 0
    unstable_reason = None
    experimental = bool(context.get("experimental", False))
    experimental_alpha = float(context.get("experimental_alpha", 0.5))
    experimental_beta = float(context.get("experimental_beta", 1.5))
    experimental_best_loss = float(context.get("experimental_best_loss", float("inf")))
    maxiter = context.get("low_level_maxiter")
    if maxiter is None:
        maxiter = low_level_maxiter_default
    if maxiter is None:
        maxiter = 100
    experimental_iter_gate = max(0, int(math.ceil(experimental_alpha * int(maxiter))))
    low_level_metadata = {
        "grad_clip": context.get("low_level_grad_clip"),
        "step_scale": context.get("low_level_step_scale"),
        "redmax_verbose": bool(context.get("low_level_redmax_verbose", False)),
    }

    def _stdout_text() -> str:
        return "".join(stdout_chunks)

    def _stderr_text() -> str:
        return ""

    def _stdout_tail(limit: int = 12000) -> str:
        return _stdout_text()[-limit:]

    def _stderr_tail(limit: int = 12000) -> str:
        return _stderr_text()[-limit:]

    def _append_stdout(text: str) -> None:
        nonlocal stdout_chars
        if not text:
            return
        stdout_chunks.append(text)
        stdout_chars += len(text)
        while stdout_chars > MAX_SUBPROCESS_CAPTURE_CHARS and stdout_chunks:
            excess = stdout_chars - MAX_SUBPROCESS_CAPTURE_CHARS
            first = stdout_chunks[0]
            if len(first) <= excess:
                stdout_chars -= len(stdout_chunks.popleft())
                continue
            stdout_chunks[0] = first[excess:]
            stdout_chars -= excess
            break

    proc: Any | None = None
    process_counted = False

    def _kill_process_group(reason: str) -> None:
        nonlocal unstable_reason
        unstable_reason = reason
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _wait_process() -> int:
        nonlocal process_counted
        if proc is None:
            return -1
        try:
            return int(proc.wait(timeout=10.0))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return int(proc.wait())
        finally:
            with _ACTIVE_PROCESS_GROUPS_LOCK:
                _ACTIVE_PROCESS_GROUPS.discard(proc.pid)
            if process_counted:
                _record_process_exit()
                process_counted = False
            close = getattr(proc, "close", None)
            if close is not None:
                close()
            elif getattr(proc, "stdout", None) is not None:
                proc.stdout.close()

    try:
        spawn_started = time.perf_counter()
        proc, spawn_method = _spawn_low_level_process(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            method=str(context.get("low_level_spawn_method", "auto")),
        )
        _record_process_spawn(
            spawn_method,
            time.perf_counter() - spawn_started,
        )
        process_counted = True
        with _ACTIVE_PROCESS_GROUPS_LOCK:
            _ACTIVE_PROCESS_GROUPS.add(proc.pid)
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        assert proc.stdout is not None
        while True:
            if deadline is not None and time.monotonic() > deadline:
                if debug:
                    print(f"[low-level] killing timed-out rollout after {float(timeout)}s", flush=True)
                _kill_process_group("low_level_subprocess_timeout")
                break
            ready, _, _ = select.select([proc.stdout], [], [], 0.2)
            if ready:
                line = proc.stdout.readline()
                if line:
                    _append_stdout(line)
                    if debug:
                        print(line, end="", flush=True)
                    lowered = line.lower()
                    match = OBJECTIVE_RE.search(line)
                    if experimental and math.isfinite(experimental_best_loss) and match is not None:
                        opt_iter = int(float(match.group(1)))
                        num_sim = int(float(match.group(2)))
                        objective = float(match.group(3))
                        threshold = experimental_beta * experimental_best_loss
                        if max(opt_iter, num_sim) >= experimental_iter_gate and objective > threshold:
                            reason = (
                                "low_level_experimental_pruned: "
                                f"iter={opt_iter} num_sim={num_sim} objective={objective} threshold={threshold}"
                            )
                            if debug:
                                print(f"[low-level] killing pruned rollout: {reason}", flush=True)
                            _kill_process_group(reason)
                    if unstable_reason is None:
                        for pattern in LOW_LEVEL_UNSTABLE_PATTERNS:
                            if pattern.lower() in lowered:
                                reason = f"low_level_numerical_instability: {pattern}"
                                if debug:
                                    print(f"[low-level] killing unstable rollout: {reason}", flush=True)
                                _kill_process_group(reason)
                                break
                    if unstable_reason is not None:
                        break
                elif proc.poll() is not None:
                    break
            elif proc.poll() is not None:
                remaining = proc.stdout.read()
                if remaining:
                    _append_stdout(remaining)
                    if debug:
                        print(remaining, end="", flush=True)
                break
        returncode = _wait_process()
    except Exception as exc:
        if proc is not None:
            if proc.poll() is None:
                _kill_process_group(
                    f"low_level_subprocess_monitor_failed: {exc}"
                )
            if process_counted:
                try:
                    _wait_process()
                except Exception:
                    pass
        failed_result = {
            "score": float("inf"),
            "loss": float("inf"),
            "error": "low_level_subprocess_monitor_failed",
            "exception": str(exc),
            "stdout": "",
            "stderr": "",
            "stdout_tail": _stdout_tail(),
            "stderr_tail": _stderr_tail(),
            "rollout_dir": str(rollout_dir),
            **low_level_metadata,
        }
        return finalized_result(failed_result)
    if unstable_reason is not None:
        failed_result = {
            "score": float("inf"),
            "loss": float("inf"),
            "error": unstable_reason,
            "returncode": int(returncode),
            "timeout": None if timeout is None else float(timeout),
            "stdout": "",
            "stderr": "",
            "stdout_tail": _stdout_tail(),
            "stderr_tail": _stderr_tail(),
            "rollout_dir": str(rollout_dir),
            **low_level_metadata,
        }
        return finalized_result(failed_result)
    if returncode != 0:
        failed_result = {
            "score": float("inf"),
            "loss": float("inf"),
            "error": "low_level_subprocess_failed",
            "returncode": int(returncode),
            "stdout": "",
            "stderr": "",
            "stdout_tail": _stdout_tail(),
            "stderr_tail": _stderr_tail(),
            "rollout_dir": str(rollout_dir),
            **low_level_metadata,
        }
        return finalized_result(failed_result)
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        failed_result = {
            "score": float("inf"),
            "loss": float("inf"),
            "error": "low_level_result_read_failed",
            "exception": str(exc),
            "returncode": int(returncode),
            "stdout": "",
            "stderr": "",
            "stdout_tail": _stdout_tail(),
            "stderr_tail": _stderr_tail(),
            "rollout_dir": str(rollout_dir),
            **low_level_metadata,
        }
        return finalized_result(failed_result)
    result["stdout"] = ""
    result["stderr"] = ""
    result["stdout_tail"] = _stdout_tail()
    result["stderr_tail"] = _stderr_tail()
    result["captured_stdout_chars"] = int(stdout_chars)
    result["low_level_result_path"] = str(result_path)
    if experimental and math.isfinite(experimental_best_loss):
        threshold = experimental_beta * experimental_best_loss
        final_score = float(result.get("score", result.get("loss", float("inf"))))
        if final_score > threshold:
            pruned_result = {
                **result,
                "score": float("inf"),
                "loss": float("inf"),
                "raw_score": final_score,
                "raw_loss": result.get("loss", final_score),
                "error": (
                    "low_level_experimental_pruned_final: "
                    f"score={final_score} threshold={threshold}"
                ),
                "returncode": int(returncode),
                "experimental_threshold": threshold,
                "experimental_best_loss": experimental_best_loss,
                **low_level_metadata,
            }
            return finalized_result(pruned_result)
    return finalized_result(result)


def visualize_xml(
    *,
    xml_path: str,
    context: dict,
    create_task: Callable[[dict], Any],
    config_from_context: Callable[[dict], dict],
    runner_args: Callable[[dict, dict, Path], Any],
) -> dict:
    """Load and replay a generated task scene."""
    import redmax_py as redmax
    from bilevel.visualization.renderer import SimRenderer

    step = int(context.get("visualize_step", context.get("step", 1000)))
    phase = str(context.get("phase", "final"))
    cfg = config_from_context(context)
    local_task = create_task(cfg)
    replay_dir = Path(
        context.get(
            "replay_dir",
            context.get(
                "cache_dir",
                "workspace/bilevel/output/task/replay",
            ),
        )
    )
    args = configure_mount_runner_args(
        runner_args(cfg, context, replay_dir),
        cfg,
        context,
    )
    Runner = _runner_class(cfg)
    sim = redmax.Simulation(xml_path, False)
    sim.reset()
    saved_params_skipped = False
    saved_params_skip_reason = None
    replay_param_diagnostics: dict[str, Any] = {}
    if phase == "final":
        evaluation = dict(context.get("evaluation", {}))
        from bilevel.lower.evaluation import validate_replay_provenance

        replay_param_diagnostics["evaluation_provenance"] = (
            validate_replay_provenance(
                evaluation,
                repo_root=REPO_ROOT,
                allow_simulator_mismatch=bool(
                    context.get("allow_replay_simulator_mismatch", False)
                ),
            )
        )
        params = evaluation.get("params")
        if params is None and isinstance(evaluation.get("result"), dict):
            params = evaluation["result"].get("params")
        if params is not None:
            params_arr = np.asarray(params, dtype=np.float64)
            runner = Runner(
                sim,
                local_task,
                args=args,
                model_path=xml_path,
                visualize=False,
                optimize_design=bool(cfg.get("optimize_design", False)),
                morphology_parameterization=_active_morphology_parameterization(
                    cfg,
                    context,
                    optimize_design=bool(
                        cfg.get("optimize_design", False)
                    ),
                ),
            )
            metadata_payload = _parameter_metadata_from_evaluation(
                evaluation
            )
            source_parameterization = evaluation.get(
                "action_parameterization"
            )
            if (
                source_parameterization is None
                and isinstance(evaluation.get("result"), dict)
            ):
                source_parameterization = evaluation["result"].get(
                    "action_parameterization"
                )
            try:
                source_action_dim = _validated_replay_action_dim(
                    params_arr,
                    metadata_payload,
                )
                if (
                    source_action_dim is not None
                    and source_action_dim != runner.action_parameter_dim
                ):
                    if (
                        source_action_dim <= 0
                        or source_action_dim % runner.ndof_u != 0
                    ):
                        raise ValueError(
                            "saved action_dim is incompatible with ndof_u: "
                            f"{source_action_dim} vs {runner.ndof_u}"
                        )
                    inferred_ctrl_steps = (
                        source_action_dim // runner.ndof_u
                    )
                    cfg["num_steps"] = int(
                        inferred_ctrl_steps
                        * int(
                            cfg.get(
                                "sub_steps",
                                local_task.sub_steps(),
                            )
                        )
                    )
                    local_task = create_task(cfg)
                    args = configure_mount_runner_args(
                        runner_args(cfg, context, replay_dir),
                        cfg,
                        context,
                    )
                    sim = redmax.Simulation(xml_path, False)
                    sim.reset()
                    runner = Runner(
                        sim,
                        local_task,
                        args=args,
                        model_path=xml_path,
                        visualize=False,
                        optimize_design=bool(cfg.get("optimize_design", False)),
                        morphology_parameterization=(
                            _active_morphology_parameterization(
                                cfg,
                                context,
                                optimize_design=bool(
                                    cfg.get("optimize_design", False)
                                ),
                            )
                        ),
                    )
                params_arr = runner.normalize_loaded_params(
                    params_arr,
                    metadata_payload,
                    legacy_action_parameterization=source_parameterization,
                )
            except ValueError as exc:
                if not bool(
                    context.get("allow_replay_param_mismatch", False)
                ):
                    raise ValueError(
                        "Cannot replay saved params without a verified "
                        f"parameter layout: {exc}"
                    ) from exc
                saved_params_skipped = True
                saved_params_skip_reason = str(exc)
                params_arr = None
            if params_arr is None:
                if runner.optimize_design and runner.design_bundle is not None:
                    runner.apply_morphology(
                        runner.initial_morphology_parameters,
                        generate_mesh=False,
                    )
                sim.reset()
            else:
                action, cage = runner.unpack_params(params_arr)
                if runner.optimize_design and runner.design_bundle is not None and cage is not None:
                    init_cage = runner.initial_morphology_parameters
                    replay_param_diagnostics.update(
                        {
                            "params_len": int(params_arr.size),
                            "ndof_cage": int(runner.ndof_cage),
                            "cage_delta_norm": float(np.linalg.norm(cage - init_cage)),
                            "cage_delta_max": float(np.max(np.abs(cage - init_cage))) if cage.size else 0.0,
                        }
                    )
                if runner.optimize_design and runner.design_bundle is not None and cage is not None:
                    if bool(context.get("render", True)):
                        design_params, meshes = (
                            runner.parameterize_morphology_numpy(
                                cage,
                                generate_mesh=True,
                            )
                        )
                        render_xml = write_render_xml_with_meshes(
                            xml_path,
                            runner,
                            meshes,
                            Path(context.get("replay_dir", replay_dir)),
                        )
                        sim = redmax.Simulation(str(render_xml), False)
                        sim.set_design_params(design_params)
                    else:
                        runner.apply_morphology(
                            cage,
                            generate_mesh=False,
                        )
                sim.reset()
                u_all = runner.controls_from_action(action)
                remaining = max(0, step)
                ctrl_idx = 0
                while remaining > 0 and ctrl_idx < runner.num_ctrl_steps:
                    u_i = u_all[ctrl_idx * runner.ndof_u : (ctrl_idx + 1) * runner.ndof_u]
                    sim.set_u(u_i)
                    n = min(runner.sub_steps, remaining)
                    sim.forward(n)
                    remaining -= n
                    ctrl_idx += 1
        elif step > 0:
            sim.forward(step)
    elif step > 0:
        sim.forward(step)
    if not bool(context.get("render", True)):
        return {
            "ok": True,
            "step": step,
            "phase": phase,
            "rendered": False,
            "saved_params_skipped": saved_params_skipped,
            "saved_params_skip_reason": saved_params_skip_reason,
            **replay_param_diagnostics,
        }
    camera_pos = context.get("camera_pos")
    if camera_pos is not None:
        sim.viewer_options.camera_pos = np.asarray(camera_pos, dtype=np.float64)
    camera_lookat = context.get("camera_lookat")
    if camera_lookat is not None:
        sim.viewer_options.camera_lookat = np.asarray(
            camera_lookat,
            dtype=np.float64,
        )
    sim.viewer_options.speed = 0.2
    SimRenderer.replay(sim, record=False, record_path=None)
    return {
        "ok": True,
        "step": step,
        "phase": phase,
        "rendered": True,
        "saved_params_skipped": saved_params_skipped,
        "saved_params_skip_reason": saved_params_skip_reason,
        **replay_param_diagnostics,
    }


def write_render_xml_with_meshes(xml_path: str, runner: Any, meshes: list[Any], replay_dir: Path) -> Path:
    """Bake current render meshes into an XML to avoid RedMax viewer mesh-update crashes."""
    _ = replay_dir
    xml_parent = Path(xml_path).resolve().parent
    render_dir = xml_parent / "render_meshes" / Path(xml_path).stem
    render_dir.mkdir(parents=True, exist_ok=True)
    root = ET.parse(xml_path).getroot()
    body_by_name = {
        body.attrib.get("name"): body
        for body in root.iter("body")
        if body.attrib.get("name")
    }
    mesh_iter = iter(meshes)
    for idx, render_rec in enumerate(runner.design_bundle.spec.render_records):
        if render_rec.source_record is None:
            continue
        try:
            mesh = next(mesh_iter)
        except StopIteration:
            raise ValueError("Not enough generated rendering meshes for design render records")
        body = body_by_name.get(render_rec.body_name)
        if body is None or mesh.V.shape[1] == 0:
            continue
        mesh_path = render_dir / f"{idx:03d}_{render_rec.body_name}.obj"
        write_obj(mesh_path, mesh.V, mesh.F)
        body.set("mesh", os.path.relpath(mesh_path, xml_parent))
    try:
        next(mesh_iter)
        raise ValueError("Generated more rendering meshes than design render records")
    except StopIteration:
        pass
    render_xml = xml_parent / f"{Path(xml_path).stem}_render.xml"
    ET.ElementTree(root).write(str(render_xml), encoding="utf-8", xml_declaration=True)
    try:
        from bilevel.parameterization.rendering import write_render_mesh_mapping

        write_render_mesh_mapping(render_xml.with_suffix(".meshmap.json"), runner, meshes, xml_path)
    except Exception:
        pass
    return render_xml


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i in range(vertices.shape[1]):
            f.write(f"v {vertices[0, i]:.12g} {vertices[1, i]:.12g} {vertices[2, i]:.12g}\n")
        if faces is not None and faces.size:
            for i in range(faces.shape[1]):
                a, b, c = (int(faces[0, i]) + 1, int(faces[1, i]) + 1, int(faces[2, i]) + 1)
                f.write(f"f {a} {b} {c}\n")
