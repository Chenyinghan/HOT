# run_main.py
import os
import sys
import argparse
import json
import runpy
import importlib.util
import traceback
import shutil
import tempfile
import time
import copy
import xml.etree.ElementTree as ET


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

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.join(_REPO_ROOT, "core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

import numpy as np
import torch
import redmax_py as redmax
from bilevel.visualization.shape_overlay import (
    SHAPE_OVERLAY_RELATIVE_OFFSET,
    TOOL_COLORS,
    apply_tool_palette as _apply_tool_palette,
    compact_submesh as _compact_shape_overlay_submesh,
    deformation_region as _deformation_region,
    outward_render_shell as _shared_outward_render_shell,
)


# Visualization only: expand the optimized red comparison shell by exactly
# 0.1% of each mesh bounding-box diagonal.  This small fixed separation avoids
# coplanar red/baseline z-fighting flicker; it never enters simulation or saved
# morphology parameters.
_SHAPE_OVERLAY_RELATIVE_OFFSET = SHAPE_OVERLAY_RELATIVE_OFFSET

try:
    _torch_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.set_num_threads(_torch_threads)
    torch.set_num_interop_threads(1)
except Exception:
    pass

import faulthandler
faulthandler.enable()
sys.stdout.reconfigure(line_buffering=True)

from bilevel.runner import CoOptRunner, initialize_action


def _make_canonical_numerical_task(task_name, config):
    """Load one maintained plugin, then expose its verified numerical task."""

    from tasks import load_task

    return load_task(task_name, config=config).numerical_task


_RUN_MAIN_CONFIG_OVERRIDES = {
    "num_steps": "num_steps",
    "sub_steps": "sub_steps",
    "max_iters": "low_level_maxiter",
    "maxls": "low_level_maxls",
    "grad_clip": "low_level_grad_clip",
    "step_scale": "low_level_step_scale",
    "lr": "low_level_lr",
    "seed": "low_level_seed",
    "action_init_mode": "low_level_action_init_mode",
    "action_init_scale": "low_level_action_init_scale",
    "action_init_smoothing_passes": (
        "low_level_action_init_smoothing_passes"
    ),
    "force_connectivity": "force_connectivity",
    "generic_design_protocol": "generic_design_protocol",
    "morphology_parameterization": "morphology_parameterization",
    "preserve_handle_mount": "preserve_handle_mount",
    "direct_planar_max_handle_step": "direct_planar_max_handle_step",
    "direct_planar_max_action_step": "direct_planar_max_action_step",
    "action_trust_radius_initial": "action_trust_radius_initial",
    "action_trust_radius_min": "action_trust_radius_min",
    "action_trust_radius_max": "action_trust_radius_max",
    "action_trust_norm_mode": "action_trust_norm_mode",
    "action_trust_temporal_basis_knots": (
        "action_trust_temporal_basis_knots"
    ),
    "action_trust_horizon_scaling": "action_trust_horizon_scaling",
    "action_trust_reference_ctrl_steps": (
        "action_trust_reference_ctrl_steps"
    ),
    "action_trust_monotone_fallback": (
        "action_trust_monotone_fallback"
    ),
    "action_trust_raw_gradient_fallback": (
        "action_trust_raw_gradient_fallback"
    ),
    "action_trust_opposite_direction_poll": (
        "action_trust_opposite_direction_poll"
    ),
    "action_trust_radius_restart": "action_trust_radius_restart",
    "action_trust_stage_reset_radius": (
        "action_trust_stage_reset_radius"
    ),
    "action_trust_failure_patience": (
        "action_trust_failure_patience"
    ),
    "action_trust_shrink_factor": "action_trust_shrink_factor",
    "action_trust_growth_factor": "action_trust_growth_factor",
    "action_trust_accept_ratio": "action_trust_accept_ratio",
    "action_trust_shrink_ratio": "action_trust_shrink_ratio",
    "action_trust_growth_ratio": "action_trust_growth_ratio",
    "action_trust_boundary_fraction": (
        "action_trust_boundary_fraction"
    ),
    "action_trust_force_preconditioner": (
        "action_trust_force_preconditioner"
    ),
    "action_trust_torque_preconditioner": (
        "action_trust_torque_preconditioner"
    ),
    "action_trust_knot_gradient_normalization_power": (
        "action_trust_knot_gradient_normalization_power"
    ),
    "action_trust_temporal_preconditioner_power": (
        "action_trust_temporal_preconditioner_power"
    ),
    "action_trust_diagnostic_event_limit": (
        "action_trust_diagnostic_event_limit"
    ),
    "direct_planar_armijo_c1": "direct_planar_armijo_c1",
    "direct_planar_action_step_scale": (
        "direct_planar_action_step_scale"
    ),
    "direct_planar_design_step_scale": (
        "direct_planar_design_step_scale"
    ),
    "direct_planar_design_block_filter": (
        "direct_planar_design_block_filter"
    ),
    "direct_planar_physical_metric": "direct_planar_physical_metric",
    "direct_planar_diagnostic_event_limit": (
        "direct_planar_diagnostic_event_limit"
    ),
    "morphology_expansion_radius_initial": (
        "morphology_expansion_radius_initial"
    ),
    "morphology_expansion_radius_growth": (
        "morphology_expansion_radius_growth"
    ),
    "morphology_expansion_radius_shrink": (
        "morphology_expansion_radius_shrink"
    ),
    "morphology_expansion_max_rms": "morphology_expansion_max_rms",
    "morphology_expansion_loss_budget": (
        "morphology_expansion_loss_budget"
    ),
    "morphology_expansion_final_loss_budget": (
        "morphology_expansion_final_loss_budget"
    ),
    "morphology_expansion_repair_min_steps": (
        "morphology_expansion_repair_min_steps"
    ),
    "morphology_expansion_repair_max_steps": (
        "morphology_expansion_repair_max_steps"
    ),
    "morphology_expansion_repair_attempt_multiplier": (
        "morphology_expansion_repair_attempt_multiplier"
    ),
    "morphology_expansion_coarse_iterations": (
        "morphology_expansion_coarse_iterations"
    ),
    "morphology_expansion_target_schedule": (
        "morphology_expansion_target_schedule"
    ),
    "morphology_expansion_target_tolerance": (
        "morphology_expansion_target_tolerance"
    ),
    "morphology_expansion_target_search_trials": (
        "morphology_expansion_target_search_trials"
    ),
    "contact_continuation_scales": "contact_continuation_scales",
    "contact_continuation_weights": "contact_continuation_weights",
    "design_collision_check": "design_collision_check",
    "design_collision_margin": "design_collision_margin",
    "design_collision_max_report": "design_collision_max_report",
    "design_collision_check_ground": "design_collision_check_ground",
    "coef_sweep_path": "coef_sweep_path",
    "coef_sweep_backtrack": "coef_sweep_backtrack",
    "coef_ball_progress": "coef_ball_progress",
    "coef_ball_goal": "coef_ball_goal",
    "coef_swing": "coef_swing",
    "coef_control": "coef_control",
    "terminal_goal_weight": "terminal_goal_weight",
    "settle_sweep_path_weight": "settle_sweep_path_weight",
    "terminal_sweep_path_weight": "terminal_sweep_path_weight",
    "sweep_backtrack_scale": "sweep_backtrack_scale",
    "settle_swing_weight": "settle_swing_weight",
    "terminal_swing_weight": "terminal_swing_weight",
    "stage_future_loss_mode": "stage_future_loss_mode",
    "stage_stop_motion": "stage_stop_motion",
}


def _explicit_run_main_config(args):
    """Translate only explicitly supplied CLI values into task config."""

    overrides = {}
    for argument_name, config_name in _RUN_MAIN_CONFIG_OVERRIDES.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            if argument_name in (
                "contact_continuation_scales",
                "contact_continuation_weights",
            ):
                value = tuple(
                    float(token)
                    for token in value.replace(";", ",").split(",")
                    if token.strip()
                )
            overrides[config_name] = value
    design_optim = getattr(args, "design_optim", None)
    if design_optim is not None:
        overrides["optimize_design"] = bool(design_optim)
    return overrides


def _load_run_main_task(task_name, args):
    """Load the canonical task with explicit CLI overrides only."""

    from tasks import load_task

    return load_task(
        task_name,
        config=_explicit_run_main_config(args),
    )


def _make_hammer_extract_nail_task(args):
    return _load_run_main_task(
        "hammer_extract_nail",
        args,
    ).numerical_task


def _make_sweep_task(args):
    return _load_run_main_task(
        "sweep_balls",
        args,
    ).numerical_task



def _make_scoop_task(args):
    return _load_run_main_task(
        "scoop_balls",
        args,
    ).numerical_task


def _make_torque_task(args):
    return _load_run_main_task(
        "torque_bolt",
        args,
    ).numerical_task


RUN_MAIN_TASKS = {
    "sweep_balls": {
        "factory": _make_sweep_task,
        "model_path": "tasks/sweep_balls/reference.xml",
        "preserve_handle_mount": True,
    },
    "hammer_extract_nail": {
        "factory": _make_hammer_extract_nail_task,
        "model_path": "tasks/hammer_extract_nail/reference.xml",
        "preserve_handle_mount": True,
    },
    "scoop_balls": {
        "factory": _make_scoop_task,
        "model_path": "tasks/scoop_balls/reference.xml",
        "preserve_handle_mount": True,
    },
    "torque_bolt": {
        "factory": _make_torque_task,
        "model_path": "tasks/torque_bolt/reference.xml",
        "preserve_handle_mount": True,
    },
}


class GenericDesignTaskAdapter:
    """
    Delegate task objective/control behavior to an existing task, but force
    design initialization through generic_design instead of task-local
    parameterization_spec.py / parameterization_spec_torch.py.
    """

    def __init__(
        self,
        task,
        *,
        freeze_finger_design: bool = True,
        force_connectivity: bool = False,
        generic_design_protocol: str = "connected_direct_planar_hexahedron",
    ):
        self._task = task
        self._freeze_finger_design = bool(freeze_finger_design)
        self._force_connectivity = bool(force_connectivity)
        self._generic_design_protocol = str(generic_design_protocol)

    def __getattr__(self, name):
        return getattr(self._task, name)

    def name(self) -> str:
        return self._task.name()

    def init_design(self, model_path: str, sim):
        from bilevel.parameterization import build_design_bundle

        configure = getattr(self._task, "_configure_variable_layout", None)
        if configure is not None:
            configure(model_path)

        bundle = build_design_bundle(
            model_path,
            sim,
            {
                "optimize_finger_design": not self._freeze_finger_design,
                "force_connectivity": self._force_connectivity,
                "generic_design_protocol": self._generic_design_protocol,
            },
        )
        self._task._design_bundle = bundle
        bundle.apply(sim, bundle.init_cage_params, generate_mesh=False)
        configure_contacts = getattr(self._task, "_configure_function_group_contacts", None)
        if configure_contacts is not None:
            configure_contacts(bundle)
        return bundle

    def bounds(self, ndof_u, num_ctrl_steps, ndof_cage, optimize_design):
        action_dim = ndof_u * num_ctrl_steps
        task_bounds = self._task.bounds(ndof_u, num_ctrl_steps, 0, False)
        if len(task_bounds) < action_dim:
            raise ValueError(
                f"task.bounds returned {len(task_bounds)} entries for "
                f"{action_dim} action parameters"
            )
        bounds = list(task_bounds[:action_dim])
        if optimize_design:
            try:
                from bilevel.parameterization import cage_bounds_for_bundle

                bounds += cage_bounds_for_bundle(
                    self._task._design_bundle,
                    ndof_cage,
                    optimize_finger_design=not self._freeze_finger_design,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to construct canonical Head-morphology bounds; "
                    "refusing the removed legacy finger-prefix fallback."
                ) from exc
        return bounds


def _default_model_path(repo_root: str, task_name: str) -> str:
    cfg = RUN_MAIN_TASKS[task_name]
    return os.path.join(repo_root, str(cfg["model_path"]))


def _apply_canonical_run_main_defaults(args, config):
    """Populate execution fields after explicit CLI overrides are merged."""

    defaults = {
        "num_steps": ("num_steps", None),
        "sub_steps": ("sub_steps", None),
        "max_iters": ("low_level_maxiter", 100),
        "seed": ("low_level_seed", 0),
        "action_init_mode": ("low_level_action_init_mode", "task"),
        "action_init_scale": ("low_level_action_init_scale", 1.0),
        "action_init_smoothing_passes": (
            "low_level_action_init_smoothing_passes",
            0,
        ),
        "force_connectivity": ("force_connectivity", False),
        "generic_design_protocol": (
            "generic_design_protocol",
            "connected_direct_planar_hexahedron",
        ),
        "morphology_parameterization": (
            "morphology_parameterization",
            None,
        ),
        "preserve_handle_mount": ("preserve_handle_mount", False),
    }
    for argument_name, (config_name, fallback) in defaults.items():
        value = config.get(config_name, fallback)
        if value is None and fallback is None and config_name in (
            "num_steps",
            "sub_steps",
        ):
            raise ValueError(
                f"canonical task config is missing required {config_name!r}"
            )
        setattr(args, argument_name, value)


def debug_dim_check(sim, runner):
    """
    Check whether the design parameter dimension produced by the task-side
    parameterization matches the dimension expected by RedMax simulation.

    This is intentionally optional because parameterization may be expensive
    and should not be forced in every normal run.
    """
    morphology = runner.initial_morphology_parameters
    if morphology is None:
        raise ValueError("dimension check requires morphology parameters")

    print("[dim-check] sim.ndof_p =", sim.ndof_p, flush=True)

    dp_np = runner.parameterize_morphology_numpy(
        morphology,
        generate_mesh=False,
    )
    dp_np = np.asarray(dp_np)
    print("[dim-check] numpy design_params.shape =", dp_np.shape, flush=True)

    morphology_t = torch.tensor(
        morphology,
        dtype=torch.double,
        requires_grad=True,
    )
    dp_t = runner.parameterize_morphology_torch(morphology_t)
    print("[dim-check] torch design_params.shape =", tuple(dp_t.shape), flush=True)

    assert dp_np.shape[0] == sim.ndof_p, (
        f"NP design params dim mismatch: dp_np={dp_np.shape[0]}, sim.ndof_p={sim.ndof_p}"
    )
    assert dp_t.numel() == sim.ndof_p, (
        f"Torch design params dim mismatch: dp_t={dp_t.numel()}, sim.ndof_p={sim.ndof_p}"
    )

    print("[dim-check] passed", flush=True)


def _write_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(vertices.shape[1]):
            f.write(f"v {vertices[0, i]:.12g} {vertices[1, i]:.12g} {vertices[2, i]:.12g}\n")
        if faces is not None and faces.size:
            for i in range(faces.shape[1]):
                a, b, c = int(faces[0, i]) + 1, int(faces[1, i]) + 1, int(faces[2, i]) + 1
                f.write(f"f {a} {b} {c}\n")


def _outward_render_shell(
    vertices: np.ndarray,
    faces: np.ndarray,
    relative_offset: float,
):
    """Return a visualization-only normal-offset shell.

    The offset is a fraction of the local mesh bounding-box diagonal.  It is
    deliberately applied only to the temporary baked render mesh: simulation,
    collision geometry, saved parameters, and reported deformation remain
    exact.  A global winding correction makes the averaged vertex normals point
    away from the mesh centroid.
    """

    return _shared_outward_render_shell(vertices, faces, relative_offset)


def _relocated_xml_attr(xml_dir: str, tmp_dir: str, value: str) -> str:
    if not value:
        return value
    full_path = value if os.path.isabs(value) else os.path.abspath(os.path.join(xml_dir, value))
    return os.path.relpath(full_path, tmp_dir)


def _contact_count(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        return int(first) if first else 0
    except Exception:
        return 0


def _write_reduced_contact_file(src_path: str, dst_path: str, cap: int) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if not lines:
        raise ValueError(f"empty contact file: {src_path}")
    n = int(lines[0].strip())
    pts = lines[1 : 1 + n]
    if cap <= 0 or n <= cap:
        selected = list(range(n))
    else:
        selected = np.linspace(0, n - 1, cap, dtype=int).tolist()
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(f"{len(selected)}\n")
        for idx in selected:
            f.write(pts[int(idx)] + "\n")


def _write_run_main_render_xml(
    model_path: str,
    runner: CoOptRunner,
    meshes: list,
    tmp_dir: str,
    *,
    baseline_meshes=None,
    contact_cap=None,
    shape_overlay=False,
    local_shape_overlay=False,
    shape_overlay_offset=_SHAPE_OVERLAY_RELATIVE_OFFSET,
) -> str:
    """
    Bake generated render meshes into a temporary XML.

    This mirrors the bilevel replay workaround, but keeps all transient files in
    repo-local tmp/ and fixes relative mesh/contact paths because the XML is
    moved out of its original task folder.
    """
    model_path = os.path.abspath(model_path)
    xml_dir = os.path.dirname(model_path)
    root = ET.parse(model_path).getroot()

    contact_dir = os.path.join(tmp_dir, "contacts")
    contact_cache: dict[str, str] = {}
    cap = 0 if contact_cap is None else int(contact_cap)
    for body in root.iter("body"):
        for attr in ("mesh", "contacts"):
            if attr in body.attrib:
                full_path = body.attrib[attr] if os.path.isabs(body.attrib[attr]) else os.path.abspath(os.path.join(xml_dir, body.attrib[attr]))
                if attr == "contacts" and cap > 0 and _contact_count(full_path) > cap:
                    if full_path not in contact_cache:
                        os.makedirs(contact_dir, exist_ok=True)
                        reduced_path = os.path.join(contact_dir, f"{len(contact_cache):03d}_{os.path.basename(full_path)}")
                        _write_reduced_contact_file(full_path, reduced_path, cap)
                        contact_cache[full_path] = reduced_path
                    body.set(attr, os.path.relpath(contact_cache[full_path], tmp_dir))
                else:
                    body.set(attr, _relocated_xml_attr(xml_dir, tmp_dir, body.attrib[attr]))

    body_by_name = {
        body.attrib.get("name"): body
        for body in root.iter("body")
        if body.attrib.get("name")
    }

    if shape_overlay and local_shape_overlay:
        raise ValueError(
            "Whole-leaf and local shape-overlay modes are mutually exclusive"
        )

    palette_colors = {}
    overlay_targets = {}
    overlay_regions = {}
    if shape_overlay or local_shape_overlay:
        if baseline_meshes is None:
            raise ValueError("Shape overlay requires baseline render meshes")
        palette_colors = _apply_tool_palette(root)
    if shape_overlay:
        overlay_targets = _shape_overlay_targets(
            runner,
            baseline_meshes,
            meshes,
        )
        print(
            "[render] whole-leaf shape overlay targets="
            + json.dumps(overlay_targets, sort_keys=True),
            flush=True,
        )
    if local_shape_overlay:
        overlay_regions = _shape_overlay_regions(
            runner,
            baseline_meshes,
            meshes,
        )
        overlay_summary = {
            body_name: {
                key: region[key]
                for key in (
                    "max_displacement",
                    "tolerance",
                    "changed_vertex_count",
                    "vertex_count",
                    "changed_face_count",
                    "face_count",
                )
            }
            for body_name, region in overlay_regions.items()
        }
        print(
            "[render] local shape overlay regions="
            + json.dumps(overlay_summary, sort_keys=True),
            flush=True,
        )

    overlay_offset = float(shape_overlay_offset)
    if not np.isfinite(overlay_offset) or not 0.0 <= overlay_offset <= 0.02:
        raise ValueError(
            "shape-overlay render offset must be a finite fraction in [0, 0.02]"
        )
    mesh_iter = iter(meshes)
    baseline_iter = iter(baseline_meshes or ())
    render_mesh_dir = os.path.join(tmp_dir, "render_meshes")
    os.makedirs(render_mesh_dir, exist_ok=True)
    overlay_meshes = {}
    for idx, render_rec in enumerate(runner.design_bundle.spec.render_records):
        if render_rec.source_record is None:
            continue
        try:
            mesh = next(mesh_iter)
        except StopIteration:
            raise ValueError("Not enough generated render meshes for design render records")
        baseline_mesh = None
        if baseline_meshes is not None:
            try:
                baseline_mesh = next(baseline_iter)
            except StopIteration:
                raise ValueError(
                    "Not enough baseline render meshes for design render records"
                )
        body = body_by_name.get(render_rec.body_name)
        if body is None or mesh.V.shape[1] == 0:
            continue
        mesh_path = os.path.join(render_mesh_dir, f"{idx:03d}_{render_rec.body_name}.obj")
        body_name = str(render_rec.body_name)
        if shape_overlay and body_name in overlay_targets:
            shell_vertices, absolute_offset = _outward_render_shell(
                mesh.V,
                mesh.F,
                overlay_offset,
            )
            _write_obj(mesh_path, shell_vertices, mesh.F)
            body.set("rgba", TOOL_COLORS["stage2_shape_red"])
            baseline_path = os.path.join(
                render_mesh_dir,
                f"{idx:03d}_{body_name}_stage1_baseline.obj",
            )
            _write_obj(baseline_path, baseline_mesh.V, baseline_mesh.F)
            overlay_meshes[body_name] = {
                "baseline_mesh": os.path.relpath(baseline_path, tmp_dir),
                "absolute_offset": float(absolute_offset),
            }
            print(
                "[render] whole-leaf comparison shell:"
                f" body={body_name}"
                f" relative_offset={overlay_offset:.6g}"
                f" absolute_offset={absolute_offset:.6g}",
                flush=True,
            )
        else:
            _write_obj(mesh_path, mesh.V, mesh.F)
        body.set("mesh", os.path.relpath(mesh_path, tmp_dir))

        region = overlay_regions.get(body_name)
        if region is not None:
            shell_vertices, absolute_offset = _outward_render_shell(
                mesh.V,
                mesh.F,
                overlay_offset,
            )
            red_vertices, red_faces = _compact_shape_overlay_submesh(
                shell_vertices,
                mesh.F,
                region["face_mask"],
            )
            baseline_vertices, baseline_faces_local = _compact_shape_overlay_submesh(
                region["baseline_vertices"],
                region["baseline_faces"],
                region["face_mask"],
            )
            red_path = os.path.join(
                render_mesh_dir,
                f"{idx:03d}_{render_rec.body_name}_deformation_red_shell.obj",
            )
            baseline_path = os.path.join(
                render_mesh_dir,
                f"{idx:03d}_{render_rec.body_name}_stage1_region.obj",
            )
            _write_obj(red_path, red_vertices, red_faces)
            _write_obj(baseline_path, baseline_vertices, baseline_faces_local)
            overlay_meshes[body_name] = {
                "red_mesh": os.path.relpath(red_path, tmp_dir),
                "baseline_mesh": os.path.relpath(baseline_path, tmp_dir),
                "absolute_offset": float(absolute_offset),
            }
            print(
                "[render] local comparison shell:"
                f" body={render_rec.body_name}"
                f" vertices={region['changed_vertex_count']}/{region['vertex_count']}"
                f" faces={region['changed_face_count']}/{region['face_count']}"
                f" relative_offset={overlay_offset:.6g}"
                f" absolute_offset={absolute_offset:.6g}",
                flush=True,
            )

    try:
        next(mesh_iter)
        raise ValueError("Generated more render meshes than design render records")
    except StopIteration:
        pass
    if baseline_meshes is not None:
        try:
            next(baseline_iter)
            raise ValueError(
                "Generated fewer render meshes than baseline render meshes"
            )
        except StopIteration:
            pass

    if shape_overlay:
        _add_shape_overlay_to_render_xml(
            root,
            runner,
            overlay_meshes,
            palette_colors,
        )
    if local_shape_overlay:
        _add_local_shape_overlay_to_render_xml(
            root,
            runner,
            overlay_meshes,
            palette_colors,
        )

    render_xml = os.path.join(tmp_dir, "run_main_render.xml")
    ET.ElementTree(root).write(render_xml, encoding="utf-8", xml_declaration=True)
    try:
        from bilevel.parameterization.rendering import write_render_mesh_mapping

        write_render_mesh_mapping(
            os.path.splitext(render_xml)[0] + ".meshmap.json",
            runner,
            meshes,
            model_path,
        )
    except Exception:
        pass
    return render_xml


def _shape_overlay_regions(
    runner: CoOptRunner,
    baseline_meshes: list,
    final_meshes: list,
) -> dict:
    """Return local deformation regions for every changed design mesh."""

    render_records = [
        rec
        for rec in runner.design_bundle.spec.render_records
        if rec.source_record is not None
    ]
    if (
        len(baseline_meshes) != len(render_records)
        or len(final_meshes) != len(render_records)
    ):
        raise ValueError(
            "Shape-overlay mesh count does not match design render records: "
            f"baseline={len(baseline_meshes)} final={len(final_meshes)} "
            f"records={len(render_records)}"
        )

    regions = {}
    for render_rec, baseline_mesh, final_mesh in zip(
        render_records,
        baseline_meshes,
        final_meshes,
    ):
        baseline_v = np.asarray(baseline_mesh.V, dtype=np.float64)
        final_v = np.asarray(final_mesh.V, dtype=np.float64)
        baseline_f = np.asarray(baseline_mesh.F, dtype=np.int64)
        final_f = np.asarray(final_mesh.F, dtype=np.int64)
        if baseline_f.shape != final_f.shape or not np.array_equal(baseline_f, final_f):
            raise ValueError(
                "Shape-overlay baseline/final topology mismatch for "
                f"{render_rec.body_name}"
            )
        region = _deformation_region(baseline_v, final_v, final_f)
        if not region["changed_face_count"]:
            continue
        region["baseline_vertices"] = baseline_v
        region["baseline_faces"] = baseline_f
        regions[str(render_rec.body_name)] = region
    return regions


def _shape_overlay_targets(
    runner: CoOptRunner,
    baseline_meshes: list,
    final_meshes: list,
) -> dict:
    render_records = [
        rec
        for rec in runner.design_bundle.spec.render_records
        if rec.source_record is not None
    ]
    if (
        len(baseline_meshes) != len(render_records)
        or len(final_meshes) != len(render_records)
    ):
        raise ValueError(
            "Shape-overlay mesh count does not match design render records: "
            f"baseline={len(baseline_meshes)} final={len(final_meshes)} "
            f"records={len(render_records)}"
        )

    leaf_node_ids = {
        int(node_id)
        for rec in runner.design_bundle.spec.marker_records
        for node_id in tuple(getattr(rec, "function_group_leaves", ()) or ())
    }
    targets = {}
    for render_rec, baseline_mesh, final_mesh in zip(
        render_records,
        baseline_meshes,
        final_meshes,
    ):
        source = render_rec.source_record
        if leaf_node_ids and getattr(source, "node_id", None) not in leaf_node_ids:
            continue
        baseline_v = np.asarray(baseline_mesh.V, dtype=np.float64)
        final_v = np.asarray(final_mesh.V, dtype=np.float64)
        if baseline_v.shape != final_v.shape or baseline_v.ndim != 2:
            continue
        if baseline_v.shape[1] == 0:
            continue
        displacement = np.linalg.norm(final_v - baseline_v, axis=0)
        max_displacement = float(np.max(displacement))
        extent = np.ptp(baseline_v, axis=1)
        scale = max(float(np.linalg.norm(extent)), 1.0)
        if max_displacement > max(1e-9, 1e-8 * scale):
            targets[str(render_rec.body_name)] = max_displacement
    return targets


def _add_shape_overlay_to_render_xml(
    root: ET.Element,
    runner: CoOptRunner,
    overlay_meshes: dict,
    palette_colors: dict,
) -> None:
    """Add full Stage 1 ghosts beneath full red Stage 2 leaf meshes.

    This preserves the historical replay composition while replacing its
    universal green baseline with the standard asset-specific tool palette.
    """

    _add_shape_overlay_baseline_ghosts(
        root,
        runner,
        overlay_meshes,
        palette_colors,
    )


def _add_shape_overlay_baseline_ghosts(
    root: ET.Element,
    runner: CoOptRunner,
    overlay_meshes: dict,
    palette_colors: dict,
) -> None:
    """Clone zero-mass Stage 1 function subtrees for comparison rendering."""

    target_body_names = set(overlay_meshes)

    parent_by_element = {
        child: parent
        for parent in root.iter()
        for child in list(parent)
    }
    function_roots = [
        link
        for link in root.iter("link")
        if str(link.attrib.get("start_function_group", "")).strip().lower()
        in {"1", "true", "yes", "on"}
    ]

    def prepare_ghost_link(link: ET.Element, suffix: str) -> None:
        for child in list(link.findall("link")):
            if child.attrib.get("function_group_leaves") is not None:
                link.remove(child)
            else:
                prepare_ghost_link(child, suffix)
        link.set("name", f"overlay_baseline_{suffix}_{link.attrib.get('name', 'link')}")
        link.set("design_params", "0")
        joint = link.find("joint")
        if joint is not None:
            joint.set(
                "name",
                f"overlay_baseline_{suffix}_{joint.attrib.get('name', 'joint')}",
            )
        body = link.find("body")
        if body is not None:
            source_body_name = body.attrib.get("name", "")
            body.set(
                "name",
                f"overlay_baseline_{suffix}_{source_body_name or 'body'}",
            )
            if source_body_name in target_body_names:
                body.set(
                    "rgba",
                    palette_colors.get(
                        source_body_name,
                        TOOL_COLORS["head_deep_teal"],
                    ),
                )
                body.set(
                    "mesh",
                    overlay_meshes[source_body_name]["baseline_mesh"],
                )
            else:
                body.set("rgba", "0 0 0 0")
            body.set("mass", "0")
            body.set("inertia", "0 0 0")
            body.set("collision", "false")
            body.set("ground_contact", "false")
            body.attrib.pop("contacts", None)

    for index, function_root in enumerate(function_roots):
        subtree_body_names = {
            body.attrib.get("name", "")
            for body in function_root.iter("body")
        }
        if target_body_names.isdisjoint(subtree_body_names):
            continue
        parent = parent_by_element.get(function_root)
        if parent is None:
            continue
        ghost = copy.deepcopy(function_root)
        prepare_ghost_link(ghost, str(index))
        siblings = list(parent)
        parent.insert(siblings.index(function_root), ghost)


def _add_local_shape_overlay_to_render_xml(
    root: ET.Element,
    runner: CoOptRunner,
    overlay_meshes: dict,
    palette_colors: dict,
) -> None:
    """Add the retained vertex-local analysis overlay as a separate mode."""

    _add_shape_overlay_baseline_ghosts(
        root,
        runner,
        overlay_meshes,
        palette_colors,
    )
    target_body_names = set(overlay_meshes)

    link_by_body_name = {}
    for link in root.iter("link"):
        if link.attrib.get("name", "").startswith("overlay_baseline_"):
            continue
        body = link.find("body")
        if body is not None and body.attrib.get("name") in target_body_names:
            link_by_body_name[body.attrib["name"]] = link

    for body_name, paths in overlay_meshes.items():
        source_link = link_by_body_name.get(body_name)
        if source_link is None:
            continue
        source_body = source_link.find("body")
        overlay_link = ET.Element(
            "link",
            {
                "name": f"overlay_stage2_red_{body_name}",
                "design_params": "0",
            },
        )
        ET.SubElement(
            overlay_link,
            "joint",
            {
                "name": f"overlay_stage2_red_joint_{body_name}",
                "type": "fixed",
                "pos": "0 0 0",
                "quat": "1 0 0 0",
            },
        )
        body_attributes = {
            "name": f"overlay_stage2_red_body_{body_name}",
            "type": "abstract",
            "pos": source_body.attrib.get("pos", "0 0 0"),
            "quat": source_body.attrib.get("quat", "1 0 0 0"),
            "rgba": TOOL_COLORS["stage2_shape_red"],
            "mesh": paths["red_mesh"],
            "mass": "0",
            "inertia": "0 0 0",
            "collision": "false",
            "ground_contact": "false",
        }
        ET.SubElement(overlay_link, "body", body_attributes)
        source_link.append(overlay_link)


def _visualize_final_with_baked_meshes(runner: CoOptRunner, params: np.ndarray, repo_root: str) -> bool:
    if not runner.visualize or not runner.optimize_design or runner.design_bundle is None:
        return False

    from bilevel.runner import SimRenderer

    tmp_root = os.path.join(repo_root, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="run_main_render_", dir=tmp_root)
    try:
        action, cage = runner.unpack_params(params)
        if cage is None:
            return False

        _, baseline_meshes = runner.parameterize_morphology_numpy(
            runner.design_bundle.init_cage_params,
            generate_mesh=True,
        )
        design_params, meshes = runner.parameterize_morphology_numpy(
            cage,
            generate_mesh=True,
        )
        render_xml = _write_run_main_render_xml(
            runner.model_path,
            runner,
            meshes,
            tmp_dir,
            baseline_meshes=baseline_meshes,
            contact_cap=getattr(runner.args, "visualize_contact_cap", None),
            shape_overlay=bool(
                getattr(runner.args, "visualize_shape_overlay", False)
            ),
            local_shape_overlay=bool(
                getattr(runner.args, "visualize_local_shape_overlay", False)
            ),
            shape_overlay_offset=_SHAPE_OVERLAY_RELATIVE_OFFSET,
        )

        sim = redmax.Simulation(render_xml, False)
        sim.set_design_params(np.asarray(design_params, dtype=np.float64))
        sim.reset()

        u_all = runner.controls_from_action(action)
        u_knots = np.asarray(u_all, dtype=np.float64).reshape(
            runner.num_ctrl_steps,
            runner.ndof_u,
        )
        q_initial = np.asarray(sim.get_q(), dtype=np.float64).copy()
        print(
            "[render] loaded action:"
            f" latent_norm={float(np.linalg.norm(action)):.9g}"
            f" control_norm={float(np.linalg.norm(u_knots)):.9g}"
            f" nonzero_knots={int(np.count_nonzero(np.linalg.norm(u_knots, axis=1) > 1e-12))}"
            f" control_abs_max={np.max(np.abs(u_knots), axis=0).tolist()}",
            flush=True,
        )
        replay_step = getattr(runner.args, "visualize_step", None)
        remaining = runner.num_steps if replay_step is None else max(0, min(int(replay_step), runner.num_steps))
        total_replay = int(remaining)
        ctrl_idx = 0
        t0 = time.time()
        print(
            f"[render] baked final replay: xml={render_xml} step={total_replay} "
            f"ctrl_steps={runner.num_ctrl_steps}",
            flush=True,
        )
        if bool(getattr(runner.args, "visualize_shape_overlay", False)):
            print(
                "[render] shape overlay: full Stage 1 functional leaves keep "
                "their standard asset colors beneath full opaque-red Stage 2 "
                "leaves; final meshes use the historical 0.1% render-only shell",
                flush=True,
            )
        if bool(getattr(runner.args, "visualize_local_shape_overlay", False)):
            print(
                "[render] local shape overlay: moved-vertex triangles only; "
                "retained as a separate analysis mode",
                flush=True,
            )
        while remaining > 0 and ctrl_idx < runner.num_ctrl_steps:
            u_i = u_all[ctrl_idx * runner.ndof_u : (ctrl_idx + 1) * runner.ndof_u]
            sim.set_u(u_i)
            n = min(runner.sub_steps, remaining)
            s0 = time.time()
            sim.forward(n, verbose=runner.args.verbose)
            remaining -= n
            ctrl_idx += 1
            print(
                f"[render] replay ctrl {ctrl_idx}/{runner.num_ctrl_steps} "
                f"advanced={total_replay - remaining}/{total_replay} "
                f"dt={time.time() - s0:.3f}s elapsed={time.time() - t0:.3f}s",
                flush=True,
            )

        q_final = np.asarray(sim.get_q(), dtype=np.float64)
        q_delta = q_final - q_initial
        print(
            "[render] replay state:"
            f" q_delta_norm={float(np.linalg.norm(q_delta)):.9g}"
            f" q_delta_max={float(np.max(np.abs(q_delta)) if q_delta.size else 0.0):.9g}",
            flush=True,
        )
        if np.linalg.norm(u_knots) > 1e-12 and np.linalg.norm(q_delta) <= 1e-12:
            print(
                "[WARN] Replay loaded nonzero controls but produced no generalized-coordinate motion.",
                flush=True,
            )

        if runner.optimize_design and runner.design_bundle is not None:
            print("cage params = ", cage)
        print_info = runner.task.print_info
        print_info("Press [Esc] to continue")
        camera_pos = getattr(runner.args, "camera_pos", None)
        camera_lookat = getattr(runner.args, "camera_lookat", None)
        camera_up = getattr(runner.args, "camera_up", None)
        if camera_pos is not None:
            sim.viewer_options.camera_pos = np.asarray(camera_pos, dtype=np.float64)
        if camera_lookat is not None:
            sim.viewer_options.camera_lookat = np.asarray(
                camera_lookat, dtype=np.float64
            )
        if camera_up is not None:
            sim.viewer_options.camera_up = np.asarray(camera_up, dtype=np.float64)
        sim.viewer_options.speed = float(getattr(runner.args, "replay_speed", 0.2))
        record_base = getattr(runner.args, "record_file_name", None) or "record"
        SimRenderer.replay(
            sim,
            record=getattr(runner.args, "record", False),
            record_path=record_base + "_optimized.gif",
        )
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)






def build_arg_parser(repo_root):

    parser = argparse.ArgumentParser("Co-optimization runner", epilog="Replay a search result: python run_main.py replay --best-run-json PATH (use replay --help for options).")

    # Generic options
    parser.add_argument("--record-file-name", type=str, default="record")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="./results/tmp/")
    parser.add_argument("--load-dir", type=str, default=None)
    parser.add_argument(
        "--initial-action",
        type=str,
        default=None,
        help=(
            "Initialize a new optimization from the optimized action stored in "
            "a rollout directory or action artifact. Unlike --load-dir, this "
            "does not enter replay mode and does not import saved morphology."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sweep_balls",
        choices=sorted(RUN_MAIN_TASKS.keys()),
        help="Task objective/control logic to run",
    )
    parser.add_argument(
        "--model-xml",
        type=str,
        default=None,
        help="Path to the RedMax XML model. Defaults to the selected task's XML.",
    )

    # Visualization
    parser.add_argument("--visualize", dest="visualize", action="store_true", help="Enable interactive viewer")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Disable interactive viewer")
    parser.set_defaults(visualize=False)
    parser.add_argument("--visualize-every", type=int, default=None, help="Auto-visualize every N optimizer callbacks")
    shape_overlay_group = parser.add_mutually_exclusive_group()
    shape_overlay_group.add_argument(
        "--visualize-shape-overlay",
        action="store_true",
        help=(
            "Overlay full Stage 1 functional leaves in standard asset colors "
            "beneath full warm-red Stage 2 leaves (historical replay style)."
        ),
    )
    shape_overlay_group.add_argument(
        "--visualize-local-shape-overlay",
        action="store_true",
        help=(
            "Analysis mode: mark only triangles incident to Stage 1 -> Stage 2 "
            "moved vertices."
        ),
    )
    parser.add_argument(
        "--visualize-step",
        type=int,
        default=1000,
        help="Simulation step to replay before opening final baked visualization",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0.2,
        help="RedMax viewer playback speed multiplier.",
    )
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
        "--camera-up",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Override the RedMax replay camera up direction.",
    )
    parser.add_argument(
        "--visualize-contact-cap",
        type=int,
        default=None,
        help="Visualization-only cap on contact points per body in the temporary baked XML.",
    )

    # Optimization
    design_group = parser.add_mutually_exclusive_group()
    design_group.add_argument(
        "--design-optim",
        dest="design_optim",
        action="store_true",
        help="Enable morphology optimization.",
    )
    design_group.add_argument(
        "--no-design-optim",
        dest="design_optim",
        action="store_false",
        help="Run action-only optimization.",
    )
    parser.set_defaults(design_optim=None)
    parser.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help=(
            "Global optimizer outer-iteration budget. For target-shell Stage2, "
            "primary and generic fallback share this single budget."
        ),
    )
    parser.add_argument("--maxls", type=int, default=None)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Optional elementwise gradient clip; zero keeps the existing unclipped behavior.",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=None,
        help=(
            "Configuration-only step scale; not used by the active optimizers."
        ),
    )
    parser.add_argument(
        "--action-init-mode",
        choices=("task", "zero", "random"),
        default=None,
        help=(
            "Action initialization source. 'random' uses the requested seed for every seed, "
            "including seed 0; 'task' preserves task-specific initialization semantics."
        ),
    )
    parser.add_argument(
        "--action-init-scale",
        type=float,
        default=None,
        help="Multiply the task-provided action initialization by this nonnegative factor.",
    )
    parser.add_argument(
        "--action-init-smoothing-passes",
        type=int,
        default=None,
        help="Apply this many [1, 2, 1]/4 temporal smoothing passes to the initial action trajectory.",
    )
    parser.add_argument("--lr", type=float, default=None)
    from bilevel.lower.optimizers import OPTIMIZER_REGISTRY

    parser.add_argument(
        "--optimizer",
        dest="optimizer_strategy",
        choices=tuple(OPTIMIZER_REGISTRY.names()),
        default=None,
        help=(
            "Explicitly override the active canonical optimizer mode. "
            "By default the selected task's optimizer policy is used."
        ),
    )
    parser.add_argument(
        "--design-source",
        type=str,
        choices=("generic",),
        default="generic",
        help="Use the canonical shared generic-design parameterization.",
    )
    parser.add_argument(
        "--force-connectivity",
        dest="force_connectivity",
        action="store_true",
        help="When using generic design, bind cage handles on dock-connected tool faces.",
    )
    parser.add_argument(
        "--no-force-connectivity",
        dest="force_connectivity",
        action="store_false",
        help="Explicitly disable dock-connected cage-handle binding.",
    )
    parser.set_defaults(force_connectivity=None)
    parser.add_argument(
        "--generic-design-protocol",
        choices=("connected_direct_planar_hexahedron",),
        default=None,
        help="Generic design protocol to use when --design-source=generic.",
    )
    parser.add_argument(
        "--morphology-parameterization",
        choices=("unified_connected_head_morphology",),
        default=None,
        help=(
            "Explicit optimizer-facing morphology layout. The default keeps "
            "the task's established layout."
        ),
    )
    parser.add_argument(
        "--preserve-handle-mount",
        dest="preserve_handle_mount",
        action="store_true",
        help="Use the exact fixed Handle-to-Head mount constraints.",
    )
    parser.add_argument(
        "--no-preserve-handle-mount",
        dest="preserve_handle_mount",
        action="store_false",
        help="Disable the Handle-mount runner for compatibility experiments.",
    )
    parser.set_defaults(preserve_handle_mount=None)
    parser.add_argument(
        "--direct-planar-max-handle-step",
        type=float,
        default=None,
        help="Maximum normalized corner-handle displacement for one direct-planar line-search trial.",
    )
    parser.add_argument(
        "--direct-planar-max-action-step",
        type=float,
        default=None,
        help=(
            "Maximum actuator-utilization L2 displacement per control knot for "
            "one line-search trial. The default comes from the selected "
            "canonical task configuration."
        ),
    )
    parser.add_argument(
        "--action-trust-radius-initial",
        type=float,
        default=None,
        help=(
            "Initial total action-trajectory utilization trust radius. "
            "Defaults to the mount-preserving action limit when available."
        ),
    )
    parser.add_argument(
        "--action-trust-radius-min",
        type=float,
        default=None,
        help="Minimum adaptive action trust radius.",
    )
    parser.add_argument(
        "--action-trust-radius-max",
        type=float,
        default=None,
        help="Maximum adaptive action trust radius.",
    )
    parser.add_argument(
        "--action-trust-norm-mode",
        choices=("trajectory_l2", "rms_per_knot"),
        default=None,
        help="Metric used to measure action-trajectory trust steps.",
    )
    parser.add_argument(
        "--action-trust-temporal-basis-knots",
        type=int,
        default=None,
        help=(
            "Project action directions onto this many piecewise-linear "
            "temporal knots; zero disables projection."
        ),
    )
    parser.add_argument(
        "--action-trust-horizon-scaling",
        choices=("none", "impulse_invariant"),
        default=None,
        help=(
            "Scale action trust radii for horizons longer than the configured "
            "reference so one update has comparable integrated impulse."
        ),
    )
    parser.add_argument(
        "--action-trust-reference-ctrl-steps",
        type=int,
        default=None,
        help=(
            "Reference control-horizon length used by action trust scaling."
        ),
    )
    parser.add_argument(
        "--action-trust-monotone-fallback",
        dest="action_trust_monotone_fallback",
        action="store_true",
        help=(
            "Accept finite strict objective decreases when contact makes the "
            "Armijo prediction unreliable; poor agreement still shrinks the "
            "trust radius."
        ),
    )
    parser.add_argument(
        "--no-action-trust-monotone-fallback",
        dest="action_trust_monotone_fallback",
        action="store_false",
        help="Require the configured trust agreement ratio for acceptance.",
    )
    parser.set_defaults(action_trust_monotone_fallback=None)
    parser.add_argument(
        "--action-trust-raw-gradient-fallback",
        dest="action_trust_raw_gradient_fallback",
        action="store_true",
        help=(
            "Retry a rejected temporal-basis direction with the unprojected "
            "per-knot action gradient."
        ),
    )
    parser.add_argument(
        "--no-action-trust-raw-gradient-fallback",
        dest="action_trust_raw_gradient_fallback",
        action="store_false",
        help="Do not retry rejected temporal-basis action directions.",
    )
    parser.set_defaults(action_trust_raw_gradient_fallback=None)
    parser.add_argument(
        "--action-trust-opposite-direction-poll",
        dest="action_trust_opposite_direction_poll",
        action="store_true",
        help=(
            "Poll the opposite action direction by measured loss when "
            "contact invalidates both analytical descent directions."
        ),
    )
    parser.add_argument(
        "--no-action-trust-opposite-direction-poll",
        dest="action_trust_opposite_direction_poll",
        action="store_false",
        help="Disable derivative-free opposite-direction polling.",
    )
    parser.set_defaults(action_trust_opposite_direction_poll=None)
    parser.add_argument(
        "--action-trust-radius-restart",
        dest="action_trust_radius_restart",
        action="store_true",
        help=(
            "When contact noise traps the trust radius at its floor, retry the "
            "same descent direction from the configured stage-reset radius."
        ),
    )
    parser.add_argument(
        "--no-action-trust-radius-restart",
        dest="action_trust_radius_restart",
        action="store_false",
        help="Disable nonlocal trust-radius restarts.",
    )
    parser.set_defaults(action_trust_radius_restart=None)
    parser.add_argument(
        "--action-trust-stage-reset-radius",
        type=float,
        default=None,
        help=(
            "Minimum radius restored when entering a new contact "
            "continuation stage; zero preserves the previous radius."
        ),
    )
    parser.add_argument(
        "--action-trust-failure-patience",
        type=int,
        default=None,
        help="Consecutive failed searches allowed before ending a stage.",
    )
    parser.add_argument(
        "--action-trust-knot-gradient-normalization-power",
        type=float,
        default=None,
        help=(
            "Power in [0,1] used to normalize gradient magnitudes across "
            "control knots; zero disables normalization."
        ),
    )
    parser.add_argument(
        "--action-trust-shrink-factor",
        type=float,
        default=None,
        help="Radius multiplier after poor model agreement.",
    )
    parser.add_argument(
        "--action-trust-growth-factor",
        type=float,
        default=None,
        help="Radius multiplier after strong boundary-step agreement.",
    )
    parser.add_argument(
        "--action-trust-accept-ratio",
        type=float,
        default=None,
        help="Minimum actual-to-predicted reduction ratio for acceptance.",
    )
    parser.add_argument(
        "--action-trust-shrink-ratio",
        type=float,
        default=None,
        help="Agreement ratio below which the radius shrinks.",
    )
    parser.add_argument(
        "--action-trust-growth-ratio",
        type=float,
        default=None,
        help="Agreement ratio above which a boundary step grows the radius.",
    )
    parser.add_argument(
        "--action-trust-boundary-fraction",
        type=float,
        default=None,
        help="Fraction of the radius that identifies a boundary step.",
    )
    parser.add_argument(
        "--action-trust-force-preconditioner",
        type=float,
        default=None,
        help="Positive diagonal preconditioner for force gradients.",
    )
    parser.add_argument(
        "--action-trust-torque-preconditioner",
        type=float,
        default=None,
        help="Positive diagonal preconditioner for torque gradients.",
    )
    parser.add_argument(
        "--action-trust-temporal-preconditioner-power",
        type=float,
        default=None,
        help=(
            "Downweight early action knots by remaining_horizon**(-power)."
        ),
    )
    parser.add_argument(
        "--action-trust-diagnostic-event-limit",
        type=int,
        default=None,
        help=(
            "Maximum recent trust-region events retained in diagnostics; "
            "zero disables event retention."
        ),
    )
    parser.add_argument(
        "--contact-continuation-scales",
        type=str,
        default=None,
        help="Comma-separated physical contact scales used by block-coordinate optimization.",
    )
    parser.add_argument(
        "--contact-continuation-weights",
        type=str,
        default=None,
        help="Comma-separated iteration weights corresponding to the contact continuation scales.",
    )
    parser.add_argument(
        "--direct-planar-armijo-c1",
        type=float,
        default=None,
        help="Armijo sufficient-decrease coefficient for action updates.",
    )
    parser.add_argument(
        "--direct-planar-action-step-scale",
        type=float,
        default=None,
        help="Scale applied to the action direction before trust-radius normalization.",
    )
    parser.add_argument(
        "--direct-planar-design-step-scale",
        type=float,
        default=None,
        help="Preconditioner applied to direct-planar morphology blocks before retraction and step capping.",
    )
    parser.add_argument(
        "--direct-planar-design-block-filter",
        type=str,
        default=None,
        help=(
            "Comma-separated direct-planar design block filter. Use 'all' for the control, "
            "'function_leaf' for terminal function leaves, or substrings such as 'sweep'."
        ),
    )
    physical_metric_group = parser.add_mutually_exclusive_group()
    physical_metric_group.add_argument(
        "--direct-planar-physical-metric",
        dest="direct_planar_physical_metric",
        action="store_true",
        help=(
            "Normalize feasible morphology steps by induced physical "
            "cage-vertex displacement."
        ),
    )
    physical_metric_group.add_argument(
        "--no-direct-planar-physical-metric",
        dest="direct_planar_physical_metric",
        action="store_false",
        help="Use the legacy parameter-space morphology metric.",
    )
    parser.set_defaults(direct_planar_physical_metric=None)
    parser.add_argument(
        "--direct-planar-diagnostic-event-limit",
        type=int,
        default=None,
        help="Maximum recent morphology Armijo events retained in diagnostics.",
    )
    parser.add_argument(
        "--morphology-expansion-radius-initial",
        type=float,
        default=None,
        help=(
            "Initial normalized RMS morphology proposal radius; 0.01 means "
            "one percent of each link reference length."
        ),
    )
    parser.add_argument(
        "--morphology-expansion-radius-growth",
        type=float,
        default=None,
        help="Trust-radius multiplier after an accepted repaired shape proposal.",
    )
    parser.add_argument(
        "--morphology-expansion-radius-shrink",
        type=float,
        default=None,
        help="Trust-radius multiplier after a rejected shape proposal.",
    )
    parser.add_argument(
        "--morphology-expansion-max-rms",
        type=float,
        default=None,
        help="Maximum accumulated normalized RMS deformation from Stage 1.",
    )
    parser.add_argument(
        "--morphology-expansion-loss-budget",
        type=float,
        default=None,
        help=(
            "Allowed temporary relative loss increase during morphology "
            "search; target-shell defaults to 0.02."
        ),
    )
    parser.add_argument(
        "--morphology-expansion-final-loss-budget",
        type=float,
        default=None,
        help=(
            "Allowed relative loss increase in the committed Stage-2 result; "
            "defaults to zero."
        ),
    )
    parser.add_argument(
        "--morphology-expansion-repair-min-steps",
        type=int,
        default=None,
        help="Minimum accepted action-repair steps before early success exit.",
    )
    parser.add_argument(
        "--morphology-expansion-repair-max-steps",
        type=int,
        default=None,
        help="Maximum accepted action-repair steps per shape proposal.",
    )
    parser.add_argument(
        "--morphology-expansion-repair-attempt-multiplier",
        type=int,
        default=None,
        help=(
            "Maximum repair attempts as a multiplier of accepted-step budget; "
            "the legacy default is 3 and fair-budget experiments may use 1."
        ),
    )
    parser.add_argument(
        "--morphology-expansion-coarse-iterations",
        type=int,
        default=None,
        help=(
            "Initial iterations using generic low-dimensional axis-scaling "
            "gradient modes before full cage gradients."
        ),
    )
    parser.add_argument(
        "--morphology-expansion-target-schedule",
        type=str,
        default=None,
        help=(
            "Comma-separated cumulative normalized RMS shells for the opt-in "
            "target-shell Stage-2 strategy; default: 0.01,0.02,0.04,0.06,0.08,0.10."
        ),
    )
    parser.add_argument(
        "--morphology-expansion-target-tolerance",
        type=float,
        default=None,
        help=(
            "Normalized RMS tolerance used when locating and bracketing a "
            "target shell; default 0.0005 (0.05 percent of reference length)."
        ),
    )
    parser.add_argument(
        "--morphology-expansion-target-search-trials",
        type=int,
        default=None,
        help=(
            "Maximum geometry-only scalar searches used to locate each target "
            "shell before action repair; default 24."
        ),
    )
    parser.add_argument(
        "--design-collision-check",
        dest="design_collision_check",
        action="store_true",
        help="Reject direct-planar morphology trials that induce initial penetration.",
    )
    parser.add_argument(
        "--no-design-collision-check",
        dest="design_collision_check",
        action="store_false",
        help="Disable direct-planar morphology penetration rejection.",
    )
    parser.set_defaults(design_collision_check=None)
    parser.add_argument(
        "--design-collision-margin",
        type=float,
        default=None,
        help="Penetration tolerance for direct-planar design collision rejection.",
    )
    parser.add_argument(
        "--design-collision-max-report",
        type=int,
        default=None,
        help="Maximum number of design collision pairs to include in diagnostics.",
    )
    parser.add_argument(
        "--design-collision-ground-check",
        dest="design_collision_check_ground",
        action="store_true",
        help="Also reject direct-planar morphology trials for tool-ground penetration.",
    )
    parser.add_argument(
        "--no-design-collision-ground-check",
        dest="design_collision_check_ground",
        action="store_false",
        help="Do not reject direct-planar morphology trials for tool-ground penetration.",
    )
    parser.set_defaults(design_collision_check_ground=None)

    # Task rollout configuration
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--sub-steps", type=int, default=None)
    parser.add_argument("--coef-sweep-path", type=float, default=None)
    parser.add_argument("--coef-sweep-backtrack", type=float, default=None)
    parser.add_argument("--coef-ball-progress", type=float, default=None)
    parser.add_argument("--coef-ball-goal", type=float, default=None)
    parser.add_argument("--coef-swing", type=float, default=None)
    parser.add_argument(
        "--coef-control",
        type=float,
        default=None,
        help="Control-magnitude loss weight; defaults to 1.0 for the freeform task.",
    )
    parser.add_argument("--terminal-goal-weight", type=float, default=None)
    parser.add_argument(
        "--settle-sweep-path-weight", type=float, default=None
    )
    parser.add_argument(
        "--terminal-sweep-path-weight", type=float, default=None
    )
    parser.add_argument("--sweep-backtrack-scale", type=float, default=None)
    parser.add_argument("--settle-swing-weight", type=float, default=None)
    parser.add_argument("--terminal-swing-weight", type=float, default=None)
    parser.add_argument(
        "--stage-future-loss-mode",
        choices=("truncate", "full"),
        default=None,
        help=(
            "After a staged optimizer gate fails, either exclude losses after "
            "the last completed stage or evaluate the frozen tail normally."
        ),
    )
    stage_stop_group = parser.add_mutually_exclusive_group()
    stage_stop_group.add_argument(
        "--stage-stop-motion",
        dest="stage_stop_motion",
        action="store_true",
        help="Freeze controlled tool coordinates after a staged gate failure.",
    )
    stage_stop_group.add_argument(
        "--no-stage-stop-motion",
        dest="stage_stop_motion",
        action="store_false",
        help="Keep simulating tool motion after a staged gate failure.",
    )
    parser.set_defaults(stage_stop_motion=None)

    # Debugging
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test-derivatives", default=False, action="store_true")
    parser.add_argument("--debug-dim-check", action="store_true")

    return parser


def main(argv=None, replay_callback=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "replay":
        from bilevel.replay import main as replay_main
        return replay_main(argv[1:])

    repo_root = os.path.dirname(os.path.abspath(__file__))

    parser = build_arg_parser(repo_root)
    args = parser.parse_args(argv)
    if replay_callback is not None and args.load_dir is None:
        parser.error("replay_callback requires --load-dir")


    redmax_module_path = os.path.abspath(getattr(redmax, "__file__", ""))
    if replay_callback is None:
        print("[info] redmax_module:", redmax_module_path, flush=True)
    if not hasattr(redmax.Simulation, "get_solver_diagnostics"):
        raise RuntimeError(
            "Loaded RedMax extension is stale and lacks get_solver_diagnostics: "
            f"{redmax_module_path}. Rebuild with "
            "'(cd core && python setup.py build_ext --inplace)' and ensure core/ is on PYTHONPATH."
        )

    visualize = bool(args.visualize)

    # Important safety guard:
    # RedMax viewer / OpenGL may crash in headless SSH environments.
    if visualize and (
        os.environ.get("DISPLAY") is None
        and os.environ.get("WAYLAND_DISPLAY") is None
    ):
        print("[WARN] No DISPLAY/WAYLAND_DISPLAY detected; forcing --no-visualize", flush=True)
        visualize = False

    verbose = bool(args.verbose or visualize)
    constructor_verbose = False
    play_mode = args.load_dir is not None

    if replay_callback is None:
        os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pick task and XML from command line
    # ------------------------------------------------------------------
    task_plugin = _load_run_main_task(args.task, args)
    task_config = task_plugin.config
    _apply_canonical_run_main_defaults(args, task_config)
    task = task_plugin.numerical_task

    model_path = args.model_xml if args.model_xml is not None else _default_model_path(repo_root, args.task)
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    optimize_design_flag = bool(
        task_config.get("optimize_design", False)
    )
    active_morphology_parameterization = (
        args.morphology_parameterization
        if optimize_design_flag
        else None
    )

    if replay_callback is None:
        print("[info] cwd:", os.getcwd(), flush=True)
        print("[info] repo_root:", repo_root, flush=True)
        print("[info] model_path:", model_path, flush=True)
    print("[info] task:", args.task, flush=True)
    print("[info] model_path exists:", os.path.exists(model_path), flush=True)
    print("[info] design_source:", args.design_source, flush=True)
    print("[info] force_connectivity:", bool(args.force_connectivity), flush=True)
    print("[info] generic_design_protocol:", args.generic_design_protocol, flush=True)
    print(
        "[info] morphology_parameterization:",
        active_morphology_parameterization,
        flush=True,
    )
    print("[info] visualize:", visualize, flush=True)
    print("[info] verbose:", verbose, flush=True)
    print("[info] optimize_design:", optimize_design_flag, flush=True)
    print("[info] play_mode:", play_mode, flush=True)
    print("[info] num_steps:", args.num_steps, flush=True)
    print("[info] sub_steps:", args.sub_steps, flush=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model XML not found: {model_path}")

    # Build simulation
    
    if verbose:
        print("[WARN] RedMax constructor verbose is disabled; Python-side diagnostics remain enabled.", flush=True)
    sim = redmax.Simulation(model_path, constructor_verbose)

    if args.verbose:
        print("[DBG] after Simulation ctor, about to baseline reset", flush=True)
    sim.reset()
    if args.verbose:
        print("[DBG] baseline reset OK", flush=True)
        print("[DBG] sim.ndof_p =", sim.ndof_p, flush=True)
        print("[DBG] sim.ndof_u =", sim.ndof_u, flush=True)

    # Baseline forward probe before runner construction.
    # This tests whether the raw XML simulation itself can reset + forward.
    if args.verbose:
        print("[DBG] baseline reset+forward probe BEFORE runner", flush=True)
    u = np.zeros(sim.ndof_u, dtype=np.float64)
    sim.set_u(u)
    sim.forward(1)
    if args.verbose:
        print("[DBG] baseline forward probe OK", flush=True)

    if args.verbose:
        sim.print_ctrl_info()
        sim.print_design_params_info()

    if visualize:
        sim.viewer_options.camera_pos = np.asarray(
            args.camera_pos or [2.5, -4.0, 1.8], dtype=np.float64
        )
        if args.camera_lookat is not None:
            sim.viewer_options.camera_lookat = np.asarray(args.camera_lookat, dtype=np.float64)
        if args.camera_up is not None:
            sim.viewer_options.camera_up = np.asarray(args.camera_up, dtype=np.float64)

    # Construct runner
    
    # Important:
    # Do NOT call task.init_design(model_path, sim) manually here.
    # CoOptRunner should own design initialization. Calling it twice can
    # create duplicated design state, inconsistent ndof_p, stale pointers,
    # or native-level crashes.
    preserve_handle_mount = bool(args.preserve_handle_mount)
    Runner = CoOptRunner
    if preserve_handle_mount:
        from bilevel.parameterization.mount import MountPreservingCoOptRunner

        Runner = MountPreservingCoOptRunner
    print(
        "[info] preserve_handle_mount:",
        preserve_handle_mount,
        flush=True,
    )
    print("[info] runner:", Runner.__name__, flush=True)
    from bilevel.runtime import runner_args_for_task
    from bilevel.lower.runtime_args import configure_mount_runner_args

    runtime_context = {}
    if args.optimizer_strategy is not None:
        runtime_context["optimizer_strategy"] = args.optimizer_strategy
    runner_args = configure_mount_runner_args(
        runner_args_for_task(
            task_plugin,
            runtime_context,
            os.path.abspath(args.save_dir),
        ),
        task_config,
        runtime_context,
    )
    runner_args.verbose = verbose
    runner_args.record = bool(args.record)
    runner_args.record_file_name = args.record_file_name
    runner_args.rollout_dir = args.save_dir if replay_callback is None else None
    runner_args.visualize_shape_overlay = bool(
        args.visualize_shape_overlay
    )
    runner_args.visualize_local_shape_overlay = bool(
        args.visualize_local_shape_overlay
    )
    runner_args.visualize_step = args.visualize_step
    runner_args.visualize_contact_cap = args.visualize_contact_cap
    runner_args.replay_speed = float(args.replay_speed)
    runner_args.stage2_input_action_source = args.initial_action
    runner_args.stage2_input_model_source = model_path

    runner = Runner(
        sim,
        task,
        args=runner_args,
        model_path=model_path,
        visualize=visualize,
        optimize_design=optimize_design_flag,
        visualize_every_n=args.visualize_every,
        morphology_parameterization=active_morphology_parameterization,
    )

    if args.verbose:
        print("[DBG] runner constructed", flush=True)
    print("[info] action_scale:", runner._action_scale.tolist(), flush=True)
    ctrl_abs_max = getattr(task, "_freeform_ctrl_abs_max", None)
    if ctrl_abs_max is not None:
        print(
            "[info] physical_wrench_limit:",
            (runner._action_scale * float(ctrl_abs_max)).tolist(),
            flush=True,
        )

    # Optional dimension check after runner has constructed the design bundle.
    if args.debug_dim_check and optimize_design_flag and runner.design_bundle is not None:
        debug_dim_check(sim, runner)

    # Initialize optimization parameters
    
    action0 = initialize_action(
        task,
        sim.ndof_u,
        runner.num_ctrl_steps,
        args.seed,
        mode=args.action_init_mode,
        scale=args.action_init_scale,
        smoothing_passes=args.action_init_smoothing_passes,
    )

    cage0 = (
        runner.initial_morphology_parameters
        if optimize_design_flag and runner.design_bundle is not None
        else None
    )

    params0 = runner.pack_params(action0, cage0)
    if args.initial_action is not None:
        if play_mode:
            raise ValueError("--initial-action cannot be combined with --load-dir")
        from bilevel.lower.action_artifacts import load_action_artifact

        action_dim = int(runner.ndof_u * runner.num_ctrl_steps)
        artifact = load_action_artifact(
            args.initial_action,
            expected_dim=action_dim,
        )
        params0[:action_dim] = artifact.action
        params0 = runner.convert_action_parameterization(
            params0,
            artifact.action_parameterization,
        )
        # Stage coordinators may warm-fill genuinely zero/random future knots.
        # A loaded trajectory already contains intentional future controls and
        # must never be mistaken for a generated zero initialization.
        runner.args.action_init_mode = "artifact"
        print(
            "[info] initialized action from:",
            str(artifact.source),
            "source_parameterization=",
            artifact.action_parameterization,
            "target_parameterization=",
            runner.action_parameterization,
            "action_norm=",
            float(np.linalg.norm(params0[:action_dim])),
            flush=True,
        )
    if not play_mode:
        runner.save_initial(args.save_dir, params0)

    # Optimize or load
    
    if not play_mode:
        params = runner.optimize(params0)
        runner.save(args.save_dir, params)
    else:
        params = runner.load(args.load_dir)

    if replay_callback is not None:
        return replay_callback(runner, params)

    # Final visualization / playback.
    #
    # For generated BASS XMLs, RedMax can hang/crash if the viewer receives
    # deformed meshes through set_rendering_mesh on the original simulation.
    # Bake the final render meshes into a temporary repo-local XML instead.
    if not _visualize_final_with_baked_meshes(runner, params, repo_root):
        runner.visualize_final(params)

    if args.test_derivatives:
        runner.fd_test(params)


if __name__ == "__main__":
    raise SystemExit(main())
