"""Shared simulation and action/shape optimization runner."""
import os
import time
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import torch

from bilevel.visualization.renderer import SimRenderer
from bilevel.parameterization.bundle import DesignBundle


def print_info(*message):
    print('\033[96m', *message, '\033[0m')


from typing import Optional


def initialize_action(
    task,
    ndof_u: int,
    num_ctrl_steps: int,
    seed: int,
    *,
    mode: str = "task",
    scale: float = 1.0,
    smoothing_passes: int = 0,
) -> np.ndarray:
    mode = str(mode).strip().lower()
    scale = float(scale)
    smoothing_passes = int(smoothing_passes)
    if mode not in ("task", "zero", "random"):
        raise ValueError(f"Unknown action initialization mode: {mode!r}")
    if scale < 0.0:
        raise ValueError(f"action initialization scale must be nonnegative, got {scale}")
    if smoothing_passes < 0:
        raise ValueError(
            f"action initialization smoothing passes must be nonnegative, got {smoothing_passes}"
        )

    shape = (int(num_ctrl_steps), int(ndof_u))
    if mode == "task":
        action = np.asarray(
            task.init_action(ndof_u, num_ctrl_steps, seed=seed), dtype=np.float64
        ).reshape(shape)
    elif mode == "zero":
        action = np.zeros(shape, dtype=np.float64)
    else:
        rng = np.random.RandomState(int(seed))
        action = rng.uniform(-0.5, 0.5, size=shape)

    action *= scale
    for _ in range(smoothing_passes):
        padded = np.pad(action, ((1, 1), (0, 0)), mode="edge")
        action = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    return action.reshape(-1)


class BaseTask:
    def name(self) -> str:
        return self.__class__.__name__

    def num_steps(self) -> int:
        raise NotImplementedError

    def sub_steps(self) -> int:
        raise NotImplementedError

    def objective_weights(self) -> dict:
        return {}

    def optimization_rollout_control_steps(
        self, num_ctrl_steps: int
    ) -> int:
        """Return the causal rollout prefix needed by the active objective.

        The default preserves the complete task horizon. Staged tasks may
        shorten optimization rollouts when all active losses and gates lie in
        a strict prefix; final/full evaluations must still return the complete
        horizon.
        """

        return int(num_ctrl_steps)

    def init_task(self, sim):
        pass

    def configure_model(self, model_path: str, sim):
        """Configure XML-dependent task metadata for every optimization mode."""
        pass

    def init_design(self, model_path: str, sim):
        return None

    def init_action(self, ndof_u: int, num_ctrl_steps: int, seed: int) -> np.ndarray:
        if seed == 0:
            return np.zeros(ndof_u * num_ctrl_steps)
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.5, 0.5, size=(ndof_u * num_ctrl_steps,))

    def action_scale(self, ndof_u: int) -> np.ndarray:
        return np.ones(ndof_u, dtype=np.float64)

    def action_parameterization(self) -> str:
        """Name the stored action-coordinate semantics used by this task."""
        return "tanh_legacy_v1"

    def action_to_control(self, action: np.ndarray, ndof_u: int) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.size % int(ndof_u) != 0:
            raise ValueError(
                f"action size {action.size} is not divisible by ndof_u={ndof_u}"
            )
        scale = np.tile(self.action_scale(ndof_u), action.size // int(ndof_u))
        return np.tanh(action) * scale

    def action_to_utilization(self, action: np.ndarray, ndof_u: int) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.size % int(ndof_u) != 0:
            raise ValueError(
                f"action size {action.size} is not divisible by ndof_u={ndof_u}"
            )
        return np.tanh(action)

    def action_control_jacobian_diag(
        self, action: np.ndarray, ndof_u: int
    ) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.size % int(ndof_u) != 0:
            raise ValueError(
                f"action size {action.size} is not divisible by ndof_u={ndof_u}"
            )
        scale = np.tile(self.action_scale(ndof_u), action.size // int(ndof_u))
        return scale * (1.0 - np.tanh(action) ** 2)

    def bounds(self, ndof_u: int, num_ctrl_steps: int, ndof_cage: int, optimize_design: bool):
        b = [(-1.0, 1.0)] * (ndof_u * num_ctrl_steps)
        if optimize_design:
            b += [(0.5, 3.0)] * ndof_cage
        return b

    def compute_terms(self, i: int, num_ctrl_steps: int, u_i: np.ndarray, variables: np.ndarray, q: np.ndarray):
        raise NotImplementedError

    def compute_design_terms(
        self,
        i: int,
        num_ctrl_steps: int,
        u_i: np.ndarray,
        variables: np.ndarray,
        q: np.ndarray,
        design_params: np.ndarray,
        design_context,
    ):
        return {}

    def prepare_design_evaluation(self, design_params: np.ndarray, design_context) -> None:
        pass

    def contact_metric_requests(self):
        """Return body-pair requests, optionally followed by a metric name."""
        return ()

    def compute_contact_terms(
        self,
        i: int,
        num_ctrl_steps: int,
        u_i: np.ndarray,
        variables: np.ndarray,
        q: np.ndarray,
        contact_metrics,
    ):
        return {}

    def contact_metric_objective_grads(
        self,
        i: int,
        num_ctrl_steps: int,
        u_i: np.ndarray,
        variables: np.ndarray,
        q: np.ndarray,
        contact_metrics,
        coef: dict,
    ):
        """Return weighted d(objective)/d(metric) keyed by (body1, body2, metric)."""
        return {}

    def write_terminal_grads(
        self,
        i: int,
        num_ctrl_steps: int,
        u_i: np.ndarray,
        variables: np.ndarray,
        q: np.ndarray,
        ndof_u: int,
        ndof_var: int,
        ndof_r: int,
        sub_steps: int,
        coef: dict,
        df_du: np.ndarray,
        df_dvar: np.ndarray,
        df_dq: np.ndarray,
    ):
        raise NotImplementedError

    def write_design_grads(
        self,
        i: int,
        num_ctrl_steps: int,
        u_i: np.ndarray,
        variables: np.ndarray,
        q: np.ndarray,
        design_params: np.ndarray,
        design_context,
        ndof_u: int,
        ndof_var: int,
        ndof_r: int,
        sub_steps: int,
        coef: dict,
        df_du: np.ndarray,
        df_dvar: np.ndarray,
        df_dq: np.ndarray,
        df_dp: np.ndarray,
    ):
        pass

    def augment_parameter_gradient(
        self,
        runner,
        params: np.ndarray,
        grad: np.ndarray,
        *,
        compute_action_grad: bool = True,
        compute_design_grad: bool = True,
    ) -> np.ndarray:
        _ = (compute_action_grad, compute_design_grad)
        return grad
    
    def print_info(self, *args):
        print(*args)

    def render(self, sim, *, record=False, record_path=None):
        pass









def _design_bundle_output_dim(bundle: DesignBundle) -> Optional[int]:
    try:
        params = bundle.design_np.parameterize(bundle.init_cage_params, generate_mesh=False)
        return int(np.asarray(params).shape[0])
    except Exception:
        return None


class CoOptRunner:
    
    def __init__(
        self,
        sim,
        task: BaseTask,
        *,
        args,
        model_path: str,
        visualize: bool,
        optimize_design: bool,
        visualize_every_n: int = None,
        morphology_parameterization: Optional[str] = None,
    ):
        self.sim = sim
        self.task = task
        self.args = args
        self.model_path = model_path
        self.visualize = visualize
        self.optimize_design = optimize_design
        # optional: automatically visualize every N callback logs
        self.visualize_every_n = visualize_every_n

        self.ndof_u = sim.ndof_u
        self.ndof_r = sim.ndof_r
        self.ndof_var = sim.ndof_var
        self.ndof_p = sim.ndof_p

        self.num_steps = task.num_steps()
        self.sub_steps = task.sub_steps()
        assert self.num_steps % self.sub_steps == 0
        self.num_ctrl_steps = self.num_steps // self.sub_steps

        task.configure_model(model_path, sim)
        self.design_bundle = task.init_design(model_path, sim) if optimize_design else None
        self.morphology_runtime = None
        self.joint_parameter_layout = None
        self.morphology_parameterization_id = None
        if self.design_bundle is not None:
            self.design_bundle.model_path = model_path
            output_dim = _design_bundle_output_dim(self.design_bundle)
            if output_dim != self.ndof_p:
                raise ValueError(
                    "design parameterizer output dim "
                    f"{output_dim} does not match sim.ndof_p {self.ndof_p}. "
                    "Refusing to use DirectDesignBundle fallback on the normal path."
                )
        requested_morphology = str(
            morphology_parameterization or ""
        ).strip()
        if requested_morphology:
            from bilevel.parameterization import (
                UNIFIED_PARAMETERIZATION_ID,
                MorphologyRuntimeBridge,
                UnifiedConnectedHeadMorphology,
            )

            if requested_morphology != UNIFIED_PARAMETERIZATION_ID:
                raise ValueError(
                    "unsupported morphology parameterization "
                    f"{requested_morphology!r}"
                )
            if not optimize_design or self.design_bundle is None:
                raise ValueError(
                    "unified morphology requires optimize_design=True"
                )
            parameterization = UnifiedConnectedHeadMorphology(
                self.design_bundle
            )
            self.morphology_runtime = MorphologyRuntimeBridge(
                parameterization
            )
            self.morphology_parameterization_id = (
                self.morphology_runtime.parameterization_id
            )
            self.joint_parameter_layout = (
                self.morphology_runtime.joint_layout(
                    self.ndof_u * self.num_ctrl_steps
                )
            )
            self.ndof_morphology = self.morphology_runtime.morphology_dim
        else:
            self.ndof_morphology = (
                self.design_bundle.ndof_cage
                if self.design_bundle is not None
                else 0
            )
        self.ndof_cage = self.ndof_morphology

        self.f_log = []
        self.num_sim = 0
        self._design_collision_rejections = 0
        self._design_collision_last_report = None
        self._design_collision_initial_report = None
        self._design_collision_final_report = None
        self._action_optimizer_diagnostics = None
        self._optimizer_metadata = None
        self._staged_stop_policy = None
        self._staged_motion_snapshot = None
        self._last_forward_diagnostics = {}
        # internal counter for auto-visualization
        self._auto_vis_counter = 0
        action_scale = np.asarray(task.action_scale(self.ndof_u), dtype=np.float64)
        if action_scale.shape != (self.ndof_u,):
            raise ValueError(
                f"task.action_scale must return shape ({self.ndof_u},), got {action_scale.shape}"
            )
        self._action_scale = action_scale
        self.action_parameterization = str(task.action_parameterization())

        task.init_task(sim)
        self._design_fd_indices_cache = None

    def configure_staged_replay_stop(self, policy) -> None:
        """Restore a persisted final-stage stop policy for deterministic replay."""

        if not policy or not bool(policy.get("enabled", False)):
            self._staged_stop_policy = None
            self._staged_motion_snapshot = None
            return
        required = ("stop_knot", "future_loss_mode", "motion_dofs")
        missing = [key for key in required if key not in policy]
        if missing:
            raise ValueError(
                "stage-stop replay policy is missing: " + ", ".join(missing)
            )
        stop_knot = int(policy["stop_knot"])
        if stop_knot < 0 or stop_knot > self.num_ctrl_steps:
            raise ValueError(
                f"stage-stop replay knot {stop_knot} is outside "
                f"[0, {self.num_ctrl_steps}]"
            )
        self._staged_stop_policy = {
            **dict(policy),
            "enabled": True,
            "stop_knot": stop_knot,
            "future_loss_mode": str(policy["future_loss_mode"]),
            "motion_dofs": [int(value) for value in policy["motion_dofs"]],
        }
        self._staged_motion_snapshot = None

    def _reset_staged_motion_stop_runtime(self, sim=None):
        self._staged_motion_snapshot = None
        policy = self._staged_stop_policy
        if not policy:
            return
        target_sim = self.sim if sim is None else sim
        if not hasattr(target_sim, "set_state"):
            raise RuntimeError(
                "stage-stop motion requires RedMax Simulation.set_state; "
                "rebuild core/redmax_py after updating python_interface.cpp"
            )

    def _staged_loss_included(self, control_step: int) -> bool:
        policy = self._staged_stop_policy
        if not policy:
            return True
        if str(policy["future_loss_mode"]) == "full":
            return True
        return int(control_step) < int(policy["stop_knot"])

    def _freeze_staged_tool_state(self, *, capture: bool, sim=None) -> None:
        policy = self._staged_stop_policy
        if not policy:
            return
        target_sim = self.sim if sim is None else sim
        indices = np.asarray(policy["motion_dofs"], dtype=np.int64)
        q = np.asarray(target_sim.get_q(), dtype=np.float64).copy()
        qdot = np.asarray(target_sim.get_qdot(), dtype=np.float64).copy()
        if (
            np.any(indices < 0)
            or np.any(indices >= q.size)
            or qdot.shape != q.shape
        ):
            raise ValueError(
                "stage-stop motion DOFs do not match the RedMax state"
            )
        if capture or self._staged_motion_snapshot is None:
            self._staged_motion_snapshot = q[indices].copy()
        q[indices] = self._staged_motion_snapshot
        qdot[indices] = 0.0
        target_sim.set_state(q, qdot)

    def advance_control_step(
        self,
        control_step: int,
        control: np.ndarray,
        *,
        backward_flag: bool,
        verbose: bool,
        sim=None,
        num_sub_steps: Optional[int] = None,
    ) -> None:
        """Advance one knot, enforcing a post-termination static tool replay."""

        target_sim = self.sim if sim is None else sim
        steps = self.sub_steps if num_sub_steps is None else int(num_sub_steps)
        if steps < 0 or steps > self.sub_steps:
            raise ValueError(
                f"num_sub_steps must be in [0, {self.sub_steps}], got {steps}"
            )
        target_sim.set_u(control)
        policy = self._staged_stop_policy
        if not policy or int(control_step) < int(policy["stop_knot"]):
            target_sim.forward(steps, verbose=verbose)
            if (
                policy
                and int(control_step) + 1 == int(policy["stop_knot"])
                and steps == self.sub_steps
            ):
                if backward_flag:
                    raise RuntimeError(
                        "stage-stop motion is non-differentiable and may "
                        "only be used after optimization terminates"
                    )
                self._freeze_staged_tool_state(capture=True, sim=target_sim)
            return

        if backward_flag:
            raise RuntimeError(
                "stage-stop motion is non-differentiable and may only be "
                "used after optimization terminates"
            )
        for _ in range(steps):
            self._freeze_staged_tool_state(capture=False, sim=target_sim)
            target_sim.forward(1, verbose=verbose)
            self._freeze_staged_tool_state(capture=False, sim=target_sim)

    @property
    def action_parameter_dim(self) -> int:
        return int(self.ndof_u * self.num_ctrl_steps)

    @property
    def morphology_slice(self) -> slice:
        if self.joint_parameter_layout is not None:
            return self.joint_parameter_layout.morphology_slice
        start = self.action_parameter_dim
        return slice(start, start + self.ndof_morphology)

    @property
    def initial_morphology_parameters(self) -> Optional[np.ndarray]:
        if not self.optimize_design or self.design_bundle is None:
            return None
        if self.morphology_runtime is not None:
            return self.morphology_runtime.initial_morphology
        return np.asarray(
            self.design_bundle.init_cage_params,
            dtype=np.float64,
        ).copy()

    @property
    def morphology_diagnostic_context(self):
        if self.morphology_runtime is not None:
            return self.morphology_runtime.diagnostic_context
        return self.design_bundle

    def apply_morphology(
        self,
        morphology: np.ndarray,
        *,
        generate_mesh: bool = False,
    ):
        if self.morphology_runtime is not None:
            return self.morphology_runtime.apply(
                self.sim,
                morphology,
                generate_mesh=generate_mesh,
            )
        if self.design_bundle is None:
            raise ValueError("runner has no morphology parameterization")
        return self.design_bundle.apply(
            self.sim,
            morphology,
            generate_mesh=generate_mesh,
        )

    def parameterize_morphology_numpy(
        self,
        morphology: np.ndarray,
        *,
        generate_mesh: bool = False,
    ):
        if self.morphology_runtime is not None:
            return self.morphology_runtime.parameterize_numpy(
                morphology,
                generate_mesh=generate_mesh,
            )
        if self.design_bundle is None:
            raise ValueError("runner has no morphology parameterization")
        return self.design_bundle.design_np.parameterize(
            morphology,
            generate_mesh=generate_mesh,
        )

    def parameterize_morphology_torch(
        self,
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        if self.morphology_runtime is not None:
            return self.morphology_runtime.parameterize_torch(morphology)
        if self.design_bundle is None:
            raise ValueError("runner has no morphology parameterization")
        return self.design_bundle.design_torch.parameterize(morphology)

    def morphology_connection_diagnostics(
        self,
        morphology: np.ndarray,
    ):
        if self.morphology_runtime is not None:
            return self.morphology_runtime.connection_diagnostics(
                morphology
            )
        if self.design_bundle is None:
            return None
        diagnostic_fn = getattr(
            getattr(self.design_bundle, "design_np", None),
            "connection_diagnostics",
            None,
        )
        return None if diagnostic_fn is None else diagnostic_fn(morphology)

    def controls_from_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_parameterization == "redmax_native_v1":
            controls = np.asarray(action, dtype=np.float64)
        else:
            controls = np.asarray(
                self.task.action_to_control(action, self.ndof_u),
                dtype=np.float64,
            )
        if controls.shape != np.asarray(action).shape:
            raise ValueError(
                "task.action_to_control must preserve the flattened action shape, "
                f"got {controls.shape} for {np.asarray(action).shape}"
            )
        if not np.all(np.isfinite(controls)):
            raise ValueError("task.action_to_control produced non-finite controls")
        return controls

    def action_utilization(self, action: np.ndarray) -> np.ndarray:
        if self.action_parameterization == "redmax_native_v1":
            utilization = np.asarray(action, dtype=np.float64)
        else:
            utilization = np.asarray(
                self.task.action_to_utilization(action, self.ndof_u),
                dtype=np.float64,
            )
        if utilization.shape != np.asarray(action).shape:
            raise ValueError(
                "task.action_to_utilization must preserve the flattened action shape"
            )
        return utilization

    def action_control_jacobian_diag(self, action: np.ndarray) -> np.ndarray:
        if self.action_parameterization == "redmax_native_v1":
            jac = np.ones_like(np.asarray(action, dtype=np.float64))
        else:
            jac = np.asarray(
                self.task.action_control_jacobian_diag(action, self.ndof_u),
                dtype=np.float64,
            )
        if jac.shape != np.asarray(action).shape:
            raise ValueError(
                "task.action_control_jacobian_diag must preserve the flattened "
                "action shape"
            )
        return jac

    def convert_action_parameterization(
        self, params: np.ndarray, source_parameterization: Optional[str]
    ) -> np.ndarray:
        """Convert saved action coordinates into this runner's current protocol."""
        params = np.asarray(params, dtype=np.float64).copy()
        source = str(source_parameterization or "tanh_legacy_v1")
        target = self.action_parameterization
        if source == target:
            return params
        action_dim = self.ndof_u * self.num_ctrl_steps
        action = params[:action_dim]
        if (
            source == "tanh_legacy_v1"
            and target in ("bounded_linear_v1", "redmax_native_v1")
        ):
            params[:action_dim] = np.tanh(action)
            return params
        if (
            source in ("bounded_linear_v1", "redmax_native_v1")
            and target == "tanh_legacy_v1"
        ):
            eps = np.finfo(np.float64).eps
            params[:action_dim] = np.arctanh(np.clip(action, -1.0 + eps, 1.0 - eps))
            return params
        if {
            source,
            target,
        } == {"bounded_linear_v1", "redmax_native_v1"}:
            return params
        raise ValueError(
            f"Unsupported action parameterization conversion: {source!r} -> {target!r}"
        )

    def parameter_artifact_metadata(self):
        """Return the complete, versioned layout contract for saved params."""

        from bilevel.parameterization import metadata_for_runner

        return metadata_for_runner(self)

    def normalize_loaded_params(
        self,
        params: np.ndarray,
        metadata: Optional[dict] = None,
        *,
        legacy_action_parameterization: Optional[str] = None,
    ) -> np.ndarray:
        """Validate and safely migrate a saved vector into this runner."""

        from bilevel.parameterization import normalize_parameters_for_runner

        return normalize_parameters_for_runner(
            self,
            params,
            metadata,
            legacy_action_parameterization=legacy_action_parameterization,
        )

    def _bounds(self):
        action_dim = self.ndof_u * self.num_ctrl_steps
        task_bounds = self.task.bounds(
            self.ndof_u,
            self.num_ctrl_steps,
            0,
            False,
        )
        if len(task_bounds) < action_dim:
            raise ValueError(
                f"task.bounds returned {len(task_bounds)} entries for "
                f"{action_dim} action parameters"
            )
        action_bounds = list(task_bounds[:action_dim])
        morphology_runtime = getattr(
            self,
            "morphology_runtime",
            None,
        )
        if morphology_runtime is not None:
            return action_bounds + list(morphology_runtime.bounds())
        if self.optimize_design and getattr(self.design_bundle, "direct_design_params", False):
            bounds = action_bounds
            bounds += [(None, None)] * self.ndof_cage
            return bounds
        bounds = self.task.bounds(
            self.ndof_u,
            self.num_ctrl_steps,
            self.ndof_cage,
            self.optimize_design and self.design_bundle is not None,
        )
        if list(bounds[:action_dim]) != action_bounds:
            raise ValueError(
                "task.bounds changes action bounds when design optimization is enabled"
            )
        return bounds

    def _design_fd_indices(self):
        if self._design_fd_indices_cache is not None:
            return self._design_fd_indices_cache
        if not self.optimize_design or self.design_bundle is None or self.ndof_cage <= 0:
            self._design_fd_indices_cache = []
            return self._design_fd_indices_cache
        bounds = self._bounds()
        action_dim = self.ndof_u * self.num_ctrl_steps
        indices = []
        for local_idx, bound in enumerate(bounds[action_dim : action_dim + self.ndof_cage]):
            lo, hi = bound
            if lo is not None and hi is not None and abs(float(hi) - float(lo)) <= 1e-12:
                continue
            indices.append(local_idx)
        self._design_fd_indices_cache = indices
        return indices

    def _fill_design_fd_grad(self, params: np.ndarray, grad: np.ndarray, base_f: float) -> None:
        """Fallback for protocols whose shape dependence is not exposed by RedMax df_dp."""
        if not bool(getattr(self.args, "design_fd_grad", False)):
            return
        if not self.optimize_design or self.design_bundle is None or self.ndof_cage <= 0:
            return
        local_indices = self._design_fd_indices()
        if not local_indices:
            return

        action_dim = self.ndof_u * self.num_ctrl_steps
        cage_slice = slice(action_dim, action_dim + self.ndof_cage)
        analytic = grad[cage_slice]
        trigger_tol = float(getattr(self.args, "design_fd_trigger_tol", 1e-12) or 0.0)
        if np.all(np.isfinite(analytic)) and float(np.linalg.norm(analytic)) > trigger_tol:
            return

        bounds = self._bounds()
        step_base = float(getattr(self.args, "design_fd_step", 1e-4) or 1e-4)
        for local_idx in local_indices:
            param_idx = action_dim + local_idx
            lo, hi = bounds[param_idx]
            x0 = float(params[param_idx])
            step = step_base * max(1.0, abs(x0))
            plus = x0 + step
            minus = x0 - step
            if hi is not None:
                plus = min(plus, float(hi))
            if lo is not None:
                minus = max(minus, float(lo))

            if plus > x0 and minus < x0:
                p_plus = np.array(params, copy=True)
                p_minus = np.array(params, copy=True)
                p_plus[param_idx] = plus
                p_minus[param_idx] = minus
                f_plus, _ = self.forward(p_plus, backward_flag=False)
                f_minus, _ = self.forward(p_minus, backward_flag=False)
                denom = plus - minus
                if denom > 0.0:
                    grad[param_idx] = (float(f_plus) - float(f_minus)) / denom
            elif plus > x0:
                p_plus = np.array(params, copy=True)
                p_plus[param_idx] = plus
                f_plus, _ = self.forward(p_plus, backward_flag=False)
                grad[param_idx] = (float(f_plus) - float(base_f)) / (plus - x0)
            elif minus < x0:
                p_minus = np.array(params, copy=True)
                p_minus[param_idx] = minus
                f_minus, _ = self.forward(p_minus, backward_flag=False)
                grad[param_idx] = (float(base_f) - float(f_minus)) / (x0 - minus)

    def _release_design_torch_graph_refs(self) -> None:
        if self.morphology_runtime is not None:
            self.morphology_runtime.release_torch_graph_refs()
            return
        design_torch = getattr(getattr(self, "design_bundle", None), "design_torch", None)
        if design_torch is None:
            return
        for cage in getattr(design_torch, "tool_cages", []) or []:
            for attr in ("params", "vertices"):
                value = getattr(cage, attr, None)
                if isinstance(value, torch.Tensor):
                    setattr(cage, attr, value.detach())

    def pack_params(self, action: np.ndarray, cage_params: Optional[np.ndarray]):
        if self.joint_parameter_layout is not None:
            return self.joint_parameter_layout.pack(action, cage_params)
        if self.optimize_design and self.design_bundle is not None:
            return np.concatenate([action, cage_params], axis=0)
        return np.array(action, copy=True)

    def unpack_params(self, params: np.ndarray):
        if self.joint_parameter_layout is not None:
            return self.joint_parameter_layout.unpack(params)
        action = params[: self.ndof_u * self.num_ctrl_steps]
        cage = None
        if self.optimize_design and self.design_bundle is not None:
            cage = params[-self.ndof_cage:]
        return action, cage

    @staticmethod
    def _normalize_contact_metric_requests(requests):
        filters = []
        seen = set()
        for request in requests or ():
            if not isinstance(request, (tuple, list)) or len(request) not in (2, 3):
                raise ValueError(
                    "Contact metric requests must be (body1, body2) or "
                    "(body1, body2, metric) tuples"
                )
            body1 = str(request[0])
            body2 = str(request[1])
            if not body1 or not body2:
                raise ValueError("Contact metric body names must be non-empty")
            pair = (body1, body2)
            if pair not in seen:
                seen.add(pair)
                filters.append(pair)
        return filters

    @staticmethod
    def _contact_name_matches(pattern: str, value: str) -> bool:
        return pattern == "*" or pattern == value

    @classmethod
    def _matching_contact_metrics(cls, contact_metrics, body1: str, body2: str):
        matches = []
        for metric in contact_metrics:
            direct = (
                cls._contact_name_matches(body1, str(metric.body1))
                and cls._contact_name_matches(body2, str(metric.body2))
            )
            reverse = (
                cls._contact_name_matches(body1, str(metric.body2))
                and cls._contact_name_matches(body2, str(metric.body1))
            )
            if direct or reverse:
                matches.append(metric)
        return matches

    def _contact_metrics_for_step(self, filters, *, derivatives: bool):
        if not filters:
            return ()
        getter = getattr(self.sim, "get_contact_pair_metrics", None)
        if getter is None:
            raise RuntimeError(
                "This task requests contact metrics, but the loaded redmax_py "
                "extension does not expose get_contact_pair_metrics(). Rebuild core/."
            )
        return tuple(getter(filters, derivatives))

    def _accumulate_contact_metric_grads(
        self,
        *,
        metric_grads,
        contact_metrics,
        control_step: int,
        df_dq: np.ndarray,
        df_dp,
        design_gradient_active: bool,
    ) -> None:
        if not metric_grads:
            return
        q_stop = (control_step + 1) * self.sub_steps * self.ndof_r
        q_start = q_stop - self.ndof_r
        for key, scale in metric_grads.items():
            if not isinstance(key, (tuple, list)) or len(key) != 3:
                raise ValueError(
                    "Contact metric gradients must be keyed by "
                    "(body1, body2, metric_name)"
                )
            body1, body2, metric_name = map(str, key)
            scale = float(scale)
            if not np.isfinite(scale):
                raise ValueError(
                    f"Non-finite contact metric gradient for {tuple(key)!r}: {scale}"
                )
            if metric_name == "activation":
                q_attr = "dactivation_dq"
                p_attr = "dactivation_dp"
            elif metric_name == "elastic_normal_force":
                q_attr = "delastic_force_dq"
                p_attr = "delastic_force_dp"
            else:
                raise ValueError(
                    f"Unsupported differentiable contact metric {metric_name!r}; "
                    "expected 'activation' or 'elastic_normal_force'"
                )

            matches = self._matching_contact_metrics(
                contact_metrics, body1, body2
            )
            if not matches:
                raise ValueError(
                    f"No physical RedMax contact pair matches ({body1!r}, {body2!r})"
                )
            for metric in matches:
                dq = np.asarray(getattr(metric, q_attr), dtype=np.float64).reshape(-1)
                if dq.shape != (self.ndof_r,):
                    raise ValueError(
                        f"Contact metric derivative {q_attr} has shape {dq.shape}, "
                        f"expected {(self.ndof_r,)}"
                    )
                df_dq[q_start:q_stop] += scale * dq
                if design_gradient_active:
                    dp = np.asarray(
                        getattr(metric, p_attr), dtype=np.float64
                    ).reshape(-1)
                    if dp.shape != (self.ndof_p,):
                        raise ValueError(
                            f"Contact metric derivative {p_attr} has shape {dp.shape}, "
                            f"expected {(self.ndof_p,)}"
                        )
                    df_dp += scale * dp

    def forward(
        self,
        params: np.ndarray,
        backward_flag: bool = False,
        *,
        backward_action_flag: bool = True,
        backward_design_flag: bool = True,
    ):
        self.num_sim += 1

        action_gradient_active = bool(backward_flag and backward_action_flag)
        design_gradient_active = bool(
            backward_flag
            and backward_design_flag
            and self.optimize_design
            and self.design_bundle is not None
        )

        action, cage_params = self.unpack_params(params)
        u_all = self.controls_from_action(action)
        rollout_steps_fn = getattr(
            self.task, "optimization_rollout_control_steps", None
        )
        active_ctrl_steps = int(
            rollout_steps_fn(self.num_ctrl_steps)
            if callable(rollout_steps_fn)
            else self.num_ctrl_steps
        )
        if not 1 <= active_ctrl_steps <= self.num_ctrl_steps:
            raise ValueError(
                "optimization_rollout_control_steps(num_ctrl_steps) must "
                "return a value in "
                f"[1, {self.num_ctrl_steps}], got {active_ctrl_steps}"
            )
        active_num_steps = active_ctrl_steps * self.sub_steps

        # apply design if needed
        design_params = None
        design_context = self.morphology_diagnostic_context
        if self.optimize_design and self.design_bundle is not None:
            design_params, _ = self.apply_morphology(
                cage_params,
                generate_mesh=False,
            )
            self.task.prepare_design_evaluation(design_params, design_context)

        # reset sim
        self.sim.reset(
            backward_flag=backward_flag,
            backward_design_params_flag=design_gradient_active,
        )
        self._reset_staged_motion_stop_runtime()

        coef = self.task.objective_weights()
        contact_filters = self._normalize_contact_metric_requests(
            self.task.contact_metric_requests()
        )

        # accumulators
        terms_sum = {k: 0.0 for k in coef.keys()}
        stage_losses = {}
        f_total = 0.0

        if backward_flag:
            df_du = np.zeros(self.ndof_u * active_num_steps)
            df_dq = np.zeros(self.ndof_r * active_num_steps)
            df_dvar = np.zeros(self.ndof_var * active_num_steps)
            if self.optimize_design and self.design_bundle is not None:
                df_dp = np.zeros(self.ndof_p)

        for i in range(active_ctrl_steps):
            u_i = u_all[i * self.ndof_u : (i + 1) * self.ndof_u]
            self.advance_control_step(
                i,
                u_i,
                backward_flag=backward_flag,
                verbose=self.args.verbose,
            )

            variables = self.sim.get_variables()
            q = self.sim.get_q()
            contact_metrics = self._contact_metrics_for_step(
                contact_filters,
                derivatives=backward_flag,
            )

            terms_i = self.task.compute_terms(i, self.num_ctrl_steps, u_i, variables, q)
            contact_terms_i = self.task.compute_contact_terms(
                i,
                self.num_ctrl_steps,
                u_i,
                variables,
                q,
                contact_metrics,
            )
            if contact_terms_i:
                terms_i = dict(terms_i)
                for k, v in contact_terms_i.items():
                    terms_i[k] = terms_i.get(k, 0.0) + float(v)
            if design_params is not None:
                design_terms_i = self.task.compute_design_terms(
                    i,
                    self.num_ctrl_steps,
                    u_i,
                    variables,
                    q,
                    design_params,
                    design_context,
                )
                if design_terms_i:
                    terms_i = dict(terms_i)
                    for k, v in design_terms_i.items():
                        terms_i[k] = terms_i.get(k, 0.0) + float(v)

            stage_fn = getattr(self.task, "loss_stage_for_step", None)
            stage_name = (
                str(stage_fn(i, self.num_ctrl_steps))
                if callable(stage_fn)
                else "full"
            )
            stage_record = stage_losses.setdefault(
                stage_name,
                {
                    "raw_weighted_loss": 0.0,
                    "included_weighted_loss": 0.0,
                    "raw_terms": {},
                    "included_terms": {},
                    "steps": 0,
                    "included_steps": 0,
                },
            )
            stage_record["steps"] += 1
            include_loss = self._staged_loss_included(i)
            if include_loss:
                stage_record["included_steps"] += 1
            for k, v in terms_i.items():
                value = float(v)
                weight = float(coef.get(k, 1.0))
                stage_record["raw_terms"][k] = (
                    stage_record["raw_terms"].get(k, 0.0) + value
                )
                stage_record["raw_weighted_loss"] += weight * value
                if not include_loss:
                    continue
                terms_sum[k] = terms_sum.get(k, 0.0) + value
                stage_record["included_terms"][k] = (
                    stage_record["included_terms"].get(k, 0.0) + value
                )
                stage_record["included_weighted_loss"] += weight * value
                f_total += weight * value

            if backward_flag:
                self.task.write_terminal_grads(
                    i=i,
                    num_ctrl_steps=self.num_ctrl_steps,
                    u_i=u_i,
                    variables=variables,
                    q=q,
                    ndof_u=self.ndof_u,
                    ndof_var=self.ndof_var,
                    ndof_r=self.ndof_r,
                    sub_steps=self.sub_steps,
                    coef=coef,
                    df_du=df_du,
                    df_dvar=df_dvar,
                    df_dq=df_dq,
                )
                if design_params is not None:
                    self.task.write_design_grads(
                        i=i,
                        num_ctrl_steps=self.num_ctrl_steps,
                        u_i=u_i,
                        variables=variables,
                        q=q,
                        design_params=design_params,
                        design_context=design_context,
                        ndof_u=self.ndof_u,
                        ndof_var=self.ndof_var,
                        ndof_r=self.ndof_r,
                        sub_steps=self.sub_steps,
                        coef=coef,
                        df_du=df_du,
                        df_dvar=df_dvar,
                        df_dq=df_dq,
                        df_dp=df_dp,
                    )
                contact_metric_grads = self.task.contact_metric_objective_grads(
                    i,
                    self.num_ctrl_steps,
                    u_i,
                    variables,
                    q,
                    contact_metrics,
                    coef,
                )
                self._accumulate_contact_metric_grads(
                    metric_grads=contact_metric_grads,
                    contact_metrics=contact_metrics,
                    control_step=i,
                    df_dq=df_dq,
                    df_dp=df_dp if design_gradient_active else None,
                    design_gradient_active=design_gradient_active,
                )

        if backward_flag:
            self.sim.backward_info.set_flags(
                False,
                False,
                design_gradient_active,
                action_gradient_active,
            )
            self.sim.backward_info.df_du = df_du
            self.sim.backward_info.df_dq = df_dq
            self.sim.backward_info.df_dvar = df_dvar
            if self.optimize_design and self.design_bundle is not None:
                self.sim.backward_info.df_dp = df_dp

        self._last_forward_diagnostics = {
            "rollout_control_steps": int(active_ctrl_steps),
            "rollout_simulation_steps": int(active_num_steps),
            "per_stage_loss": stage_losses,
            "stage_stop": (
                None
                if self._staged_stop_policy is None
                else {
                    **dict(self._staged_stop_policy),
                    "motion_snapshot": (
                        None
                        if self._staged_motion_snapshot is None
                        else self._staged_motion_snapshot.tolist()
                    ),
                }
            ),
        }
        return f_total, terms_sum

    def loss_and_grad(
        self,
        params: np.ndarray,
        *,
        compute_action_grad: bool = True,
        compute_design_grad: bool = True,
    ):
        if not compute_action_grad and not compute_design_grad:
            raise ValueError("At least one parameter-gradient block must be enabled")
        with torch.no_grad():
            f, _ = self.forward(
                params,
                backward_flag=True,
                backward_action_flag=compute_action_grad,
                backward_design_flag=compute_design_grad,
            )
            self.sim.backward()

        grad = np.zeros_like(params, dtype=np.float64)

        # Control gradient: sum over sub-steps, then apply the task's
        # action-to-control Jacobian (linear for bounded freeform actions).
        action, cage = self.unpack_params(params)
        action_param_count = self.num_ctrl_steps * self.ndof_u
        active_ctrl_steps = int(
            self._last_forward_diagnostics["rollout_control_steps"]
        )
        active_num_steps = int(
            self._last_forward_diagnostics["rollout_simulation_steps"]
        )
        if compute_action_grad:
            df_du_full = np.copy(self.sim.backward_results.df_du)
            expected_du = active_num_steps * self.ndof_u
            if df_du_full.size != expected_du:
                raise ValueError(
                    "RedMax returned an action gradient with size "
                    f"{df_du_full.size}; expected {expected_du} for "
                    f"{active_ctrl_steps} active control steps"
                )
            g_u = np.sum(
                df_du_full.reshape(
                    active_ctrl_steps, self.sub_steps, self.ndof_u
                ),
                axis=1,
            ).reshape(-1)
            active_action_dim = active_ctrl_steps * self.ndof_u
            grad[:active_action_dim] = (
                g_u
                * self.action_control_jacobian_diag(action)[
                    :active_action_dim
                ]
            )

        # design grad: df_dp -> cage_params via torch parameterization
        if compute_design_grad and self.optimize_design and self.design_bundle is not None:
            try:
                df_dp = torch.tensor(np.copy(self.sim.backward_results.df_dp), dtype=torch.double)
                cage_t = torch.tensor(cage, dtype=torch.double, requires_grad=True)
                dp_t = self.parameterize_morphology_torch(cage_t)
                dp_t.backward(df_dp)
                grad[self.morphology_slice] = (
                    cage_t.grad.detach().cpu().numpy()
                )
                self._fill_design_fd_grad(params, grad, f)
            finally:
                self._release_design_torch_graph_refs()

        grad = np.asarray(
            self.task.augment_parameter_gradient(
                self,
                params,
                grad,
                compute_action_grad=compute_action_grad,
                compute_design_grad=compute_design_grad,
            ),
            dtype=np.float64,
        )
        if grad.shape != params.shape:
            raise ValueError(
                f"Task gradient augmentation returned shape {grad.shape}, expected {params.shape}"
            )

        if not compute_action_grad:
            grad[:action_param_count] = 0.0
        if not compute_design_grad and self.optimize_design and self.design_bundle is not None:
            grad[action_param_count:] = 0.0

        grad_clip = getattr(self.args, "grad_clip", None)
        if grad_clip is not None and float(grad_clip) > 0.0:
            np.clip(grad, -float(grad_clip), float(grad_clip), out=grad)

        return f, grad

    def _loss_and_grad_for_block(self, params: np.ndarray, block_name: str):
        if block_name == "action":
            return self.loss_and_grad(
                params,
                compute_action_grad=True,
                compute_design_grad=False,
            )
        if block_name == "design":
            return self.loss_and_grad(
                params,
                compute_action_grad=False,
                compute_design_grad=True,
            )
        raise ValueError(f"Unknown optimization block {block_name!r}")

    def _apply_replay_camera(self) -> None:
        camera_pos = getattr(self.args, "camera_pos", None)
        camera_lookat = getattr(self.args, "camera_lookat", None)
        camera_up = getattr(self.args, "camera_up", None)
        if camera_pos is not None:
            self.sim.viewer_options.camera_pos = np.asarray(
                camera_pos, dtype=np.float64
            )
        if camera_lookat is not None:
            self.sim.viewer_options.camera_lookat = np.asarray(
                camera_lookat, dtype=np.float64
            )
        if camera_up is not None:
            self.sim.viewer_options.camera_up = np.asarray(
                camera_up, dtype=np.float64
            )

    def callback(self, params: np.ndarray, render: bool = False, record: bool = False, record_path: Optional[str] = None, log: bool = True):
        f, info = self.forward(params, backward_flag=False)
        self.num_sim -= 1

        print_info("iteration ", len(self.f_log), ", num_sim = ", self.num_sim, ", Objective = ", f, info)
        if log:
            self.f_log.append(np.array([self.num_sim, f], dtype=np.float64))

        # auto-visualize every N logged callbacks if configured
        self._maybe_auto_visualize(params, record=record, record_path=record_path)

        if render:
            if self.optimize_design and self.design_bundle is not None:
                _, cage = self.unpack_params(params)
                self.apply_morphology(cage, generate_mesh=True)

            self._apply_replay_camera()
            self.sim.viewer_options.speed = 0.2
            SimRenderer.replay(self.sim, record=record, record_path=record_path)

    def _direct_planar_abs_blocks(self):
        if self.morphology_runtime is not None:
            context = self.morphology_runtime.diagnostic_context
            records = list(
                getattr(context.specification, "tool_records", ()) or ()
            )
            record_index_by_node = {
                int(record.node_id): index
                for index, record in enumerate(records)
                if getattr(record, "node_id", None) is not None
            }
            out = []
            morphology_start = int(self.morphology_slice.start)
            for block in self.morphology_runtime.active_blocks:
                block_out = {
                    "tool_index": record_index_by_node.get(
                        int(block.node_id),
                        -1,
                    ),
                    "node_id": int(block.node_id),
                    "start": int(block.start),
                    "stop": int(block.stop),
                    "abs_start": morphology_start + int(block.start),
                    "abs_stop": morphology_start + int(block.stop),
                    "deformable": True,
                    "link_name": block.link_name,
                    "body_name": block.body_name,
                    "face_mask": list(block.connected_face_mask),
                    "frozen_parameter_indices": list(
                        block.frozen_parameter_indices
                    ),
                    "direct_handle_mount": bool(
                        block.direct_handle_mount
                    ),
                    "reference_length": float(block.reference_length),
                }
                block_out["design_enabled"] = (
                    self._direct_planar_block_filter_allows(block_out)
                )
                out.append(block_out)
            return out
        bundle = getattr(self, "design_bundle", None)
        if (
            bundle is None
            or getattr(bundle, "generic_design_protocol", None)
            != "connected_direct_planar_hexahedron"
        ):
            return []
        action_dim = self.ndof_u * self.num_ctrl_steps
        out = []
        for block in getattr(bundle, "direct_planar_blocks", []) or []:
            block_out = {
                **block,
                "abs_start": action_dim + int(block["start"]),
                "abs_stop": action_dim + int(block["stop"]),
            }
            block_out["design_enabled"] = self._direct_planar_block_filter_allows(block_out)
            out.append(block_out)
        return out

    def _direct_planar_filter_tokens(self):
        raw = str(getattr(self.args, "direct_planar_design_block_filter", "all") or "all").strip()
        if raw.lower() in {"", "all", "*"}:
            return ()
        return tuple(tok.strip().lower() for tok in raw.replace(";", ",").split(",") if tok.strip())

    def _direct_planar_function_leaf_names(self):
        fg = getattr(self.task, "_function_group_contacts", None)
        records = list(getattr(fg, "leaf_records", []) or [])
        names: set[str] = set()
        for rec in records:
            for attr in ("link_name", "body_name"):
                value = str(getattr(rec, attr, "") or "").lower()
                if value:
                    names.add(value)
        return names

    def _direct_planar_block_filter_allows(self, block: dict) -> bool:
        if not bool(block.get("deformable", False)):
            return False
        tokens = self._direct_planar_filter_tokens()
        if not tokens:
            return True
        link_name = str(block.get("link_name", "") or "").lower()
        body_name = str(block.get("body_name", "") or "").lower()
        haystack = f"{link_name} {body_name} {block.get('tool_index', '')}".lower()
        if any(tok in {"function_leaf", "function_leaves", "terminal_leaf", "terminal_leaves"} for tok in tokens):
            leaf_names = self._direct_planar_function_leaf_names()
            if link_name in leaf_names or body_name in leaf_names:
                return True
        return any(tok in haystack for tok in tokens if tok not in {"function_leaf", "function_leaves", "terminal_leaf", "terminal_leaves"})

    def _direct_planar_geometry(self):
        from bilevel.parameterization import (
            geometry as geom_pkg,
        )

        return geom_pkg

    def _design_collision_enabled(self) -> bool:
        if self.design_bundle is None:
            return False
        protocol = getattr(self.design_bundle, "generic_design_protocol", None)
        if protocol != "connected_direct_planar_hexahedron":
            return False
        raw = getattr(self.args, "design_collision_check", None)
        if raw is None:
            return True
        return bool(raw)

    def _morphology_collision_policy(self):
        from bilevel.parameterization import MorphologyCollisionPolicy

        return MorphologyCollisionPolicy(
            enabled=self._design_collision_enabled(),
            margin=float(
                getattr(
                    self.args,
                    "design_collision_margin",
                    1e-4,
                )
                or 1e-4
            ),
            max_report=int(
                getattr(
                    self.args,
                    "design_collision_max_report",
                    8,
                )
                or 8
            ),
            check_ground=bool(
                getattr(
                    self.args,
                    "design_collision_check_ground",
                    False,
                )
            ),
        )

    def _design_collision_report_for_params(self, params: np.ndarray):
        if not self._design_collision_enabled():
            return None
        if self.morphology_runtime is not None:
            _, morphology = self.unpack_params(params)
            return self.morphology_runtime.collision_report(
                morphology,
                policy=self._morphology_collision_policy(),
                xml_path=self.model_path,
            )
        from bilevel.parameterization.collision import check_design_collision

        action_dim = self.ndof_u * self.num_ctrl_steps
        cage = np.asarray(params[action_dim : action_dim + self.ndof_cage], dtype=np.float64)
        return check_design_collision(
            cage,
            self.design_bundle,
            xml_path=self.model_path,
            margin=float(getattr(self.args, "design_collision_margin", 1e-4) or 1e-4),
            max_report=int(getattr(self.args, "design_collision_max_report", 8) or 8),
            check_ground=bool(getattr(self.args, "design_collision_check_ground", False)),
        )

    def _direct_planar_collision_ok(self, params: np.ndarray, trial: np.ndarray) -> bool:
        if not self._design_collision_enabled():
            return True
        if self.morphology_runtime is not None:
            _, previous = self.unpack_params(params)
            _, candidate = self.unpack_params(trial)
            decision = (
                self.morphology_runtime.validate_collision_transition(
                    previous,
                    candidate,
                    policy=self._morphology_collision_policy(),
                    xml_path=self.model_path,
                )
            )
            self._design_collision_last_report = (
                decision.report.to_dict()
                if decision.report is not None
                else None
            )
            if not decision.accepted:
                self._design_collision_rejections += 1
            return bool(decision.accepted)
        action_dim = self.ndof_u * self.num_ctrl_steps
        if np.allclose(params[action_dim:], trial[action_dim:], rtol=0.0, atol=1e-14):
            return True
        report = self._design_collision_report_for_params(trial)
        self._design_collision_last_report = report.to_dict() if report is not None else None
        if report is not None and not report.ok:
            self._design_collision_rejections += 1
            return False
        return True

    def _direct_planar_project_vector(self, at_params: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if self.morphology_runtime is not None:
            at_values = self.joint_parameter_layout.validate(at_params)
            direction = self.joint_parameter_layout.validate(vector)
            projected = np.array(direction, copy=True, dtype=np.float64)
            morphology = at_values[self.morphology_slice]
            morphology_direction = np.array(
                direction[self.morphology_slice],
                copy=True,
                dtype=np.float64,
            )
            for block in self._direct_planar_abs_blocks():
                if bool(block.get("design_enabled", True)):
                    continue
                local_start = (
                    int(block["abs_start"])
                    - int(self.morphology_slice.start)
                )
                local_stop = (
                    int(block["abs_stop"])
                    - int(self.morphology_slice.start)
                )
                morphology_direction[local_start:local_stop] = 0.0
            projected[self.morphology_slice] = (
                self.morphology_runtime.project_tangent(
                    morphology,
                    morphology_direction,
                )
            )
            return projected
        geom_pkg = self._direct_planar_geometry()

        projected = np.array(vector, copy=True, dtype=np.float64)
        action_dim = self.ndof_u * self.num_ctrl_steps
        if projected.shape[0] > action_dim:
            projected[action_dim:] = 0.0
        for block in self._direct_planar_abs_blocks():
            s = int(block["abs_start"])
            e = int(block["abs_stop"])
            if not bool(block.get("deformable", False)) or not bool(block.get("design_enabled", True)):
                projected[s:e] = 0.0
                continue
            if getattr(geom_pkg, "GENERIC_DESIGN_PROTOCOL", None) == "connected_direct_planar_hexahedron":
                projected[s:e] = geom_pkg.project_tangent(at_params[s:e], vector[s:e], face_mask=block.get("face_mask"))
            else:
                projected[s:e] = geom_pkg.project_tangent(at_params[s:e], vector[s:e])
        return projected

    def _direct_planar_retract_params(self, params: np.ndarray, step: np.ndarray):
        if self.morphology_runtime is not None:
            current = self.joint_parameter_layout.validate(params)
            direction = self.joint_parameter_layout.validate(step)
            trial = current + direction
            bounds = self._bounds()
            for index in range(self.action_parameter_dim):
                low, high = bounds[index]
                if low is not None:
                    trial[index] = max(trial[index], float(low))
                if high is not None:
                    trial[index] = min(trial[index], float(high))
            morphology_step = np.array(
                direction[self.morphology_slice],
                copy=True,
                dtype=np.float64,
            )
            for block in self._direct_planar_abs_blocks():
                if bool(block.get("design_enabled", True)):
                    continue
                local_start = (
                    int(block["abs_start"])
                    - int(self.morphology_slice.start)
                )
                local_stop = (
                    int(block["abs_stop"])
                    - int(self.morphology_slice.start)
                )
                morphology_step[local_start:local_stop] = 0.0
            result = self.morphology_runtime.retract(
                current[self.morphology_slice],
                morphology_step,
                max_shape_displacement=float(
                    getattr(
                        self.args,
                        "mount_preserving_max_shape_displacement",
                        0.125,
                    )
                    or 0.125
                ),
                mount_face_tolerance=float(
                    getattr(
                        self.args,
                        "mount_face_tolerance",
                        1e-7,
                    )
                ),
            )
            if not result.ok:
                return params, False
            trial[self.morphology_slice] = result.morphology
            if not self._direct_planar_collision_ok(current, trial):
                return params, False
            return trial, True
        geom_pkg = self._direct_planar_geometry()

        trial = np.asarray(params, dtype=np.float64) + np.asarray(step, dtype=np.float64)
        if not np.all(np.isfinite(trial)):
            return params, False

        # Clip action bounds; enforce geometry feasibility through retraction.
        bounds = self._bounds()
        action_dim = self.ndof_u * self.num_ctrl_steps
        for i in range(min(action_dim, len(bounds))):
            lo, hi = bounds[i]
            if lo is not None:
                trial[i] = max(trial[i], float(lo))
            if hi is not None:
                trial[i] = min(trial[i], float(hi))

        if self.ndof_cage > 0:
            cage_start = action_dim
            trial[cage_start:] = params[cage_start:]

        for block in self._direct_planar_abs_blocks():
            s = int(block["abs_start"])
            e = int(block["abs_stop"])
            if not bool(block.get("deformable", False)) or not bool(block.get("design_enabled", True)):
                trial[s:e] = params[s:e]
                continue
            if getattr(geom_pkg, "GENERIC_DESIGN_PROTOCOL", None) == "connected_direct_planar_hexahedron":
                result = geom_pkg.retract(params[s:e], step[s:e], face_mask=block.get("face_mask"))
            else:
                result = geom_pkg.retract(params[s:e], step[s:e])
            if not result.ok:
                return params, False
            try:
                if getattr(geom_pkg, "GENERIC_DESIGN_PROTOCOL", None) == "connected_direct_planar_hexahedron":
                    geom_pkg.validate_hexahedron(result.q, face_mask=block.get("face_mask"))
                else:
                    geom_pkg.validate_hexahedron(result.q)
            except ValueError:
                return params, False
            trial[s:e] = result.q
        if not self._direct_planar_collision_ok(params, trial):
            return params, False
        return trial, True

    def _direct_planar_scale_direction(self, params: np.ndarray, direction: np.ndarray) -> np.ndarray:
        from bilevel.parameterization.geometry import (
            max_vertex_displacement,
        )

        max_disp_allowed = float(getattr(self.args, "direct_planar_max_handle_step", 0.05) or 0.05)
        if max_disp_allowed <= 0.0:
            return direction
        max_disp = 0.0
        blocks = []
        for block in self._direct_planar_abs_blocks():
            if not bool(block.get("deformable", False)) or not bool(block.get("design_enabled", True)):
                continue
            s = int(block["abs_start"])
            e = int(block["abs_stop"])
            blocks.append((s, e))
            displacement = max_vertex_displacement(
                params[s:e],
                params[s:e] + direction[s:e],
            )
            if bool(
                getattr(
                    self.args,
                    "direct_planar_physical_metric",
                    False,
                )
            ):
                displacement *= float(
                    block.get("reference_length", 1.0) or 1.0
                )
            max_disp = max(max_disp, displacement)
        if max_disp > max_disp_allowed:
            scaled = np.array(direction, copy=True, dtype=np.float64)
            cage_scale = max_disp_allowed / max(max_disp, 1e-12)
            for s, e in blocks:
                scaled[s:e] *= cage_scale
            return scaled
        return direction

    def _direct_planar_scale_action_direction(
        self, params: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        max_step = float(getattr(self.args, "direct_planar_max_action_step", 0.05) or 0.05)
        if max_step <= 0.0 or self.ndof_u <= 0:
            return direction
        action_dim = self.ndof_u * self.num_ctrl_steps
        action = np.asarray(params[:action_dim], dtype=np.float64).reshape(
            self.num_ctrl_steps, self.ndof_u
        )
        action_direction = np.asarray(
            direction[:action_dim], dtype=np.float64
        ).reshape(self.num_ctrl_steps, self.ndof_u)

        def max_utilization_step(scale):
            current = self.action_utilization(action.reshape(-1)).reshape(
                self.num_ctrl_steps, self.ndof_u
            )
            candidate = self.action_utilization(
                (action + scale * action_direction).reshape(-1)
            ).reshape(self.num_ctrl_steps, self.ndof_u)
            delta = candidate - current
            norms = np.linalg.norm(delta, axis=1)
            return float(np.max(norms)) if norms.size else 0.0

        if max_utilization_step(1.0) <= max_step:
            return direction
        lo, hi = 0.0, 1.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if max_utilization_step(mid) <= max_step:
                lo = mid
            else:
                hi = mid
        scaled = np.array(direction, copy=True, dtype=np.float64)
        scaled[:action_dim] *= lo
        return scaled

    def _action_trust_config(self) -> dict:
        """Resolve and validate the shared action trust-region policy."""

        args = self.args
        legacy_initial = getattr(
            args, "mount_preserving_max_action_step", None
        )
        if legacy_initial is None:
            legacy_initial = getattr(
                args, "direct_planar_max_action_step", 0.05
            )
        initial = getattr(args, "action_trust_radius_initial", None)
        if initial is None:
            initial = legacy_initial
        maximum = getattr(args, "action_trust_radius_max", None)
        if maximum is None:
            legacy_maximum = getattr(
                args, "direct_planar_max_action_step", None
            )
            if legacy_maximum is None:
                legacy_maximum = 0.05
            maximum = max(
                float(initial),
                float(legacy_maximum),
            )
        minimum = getattr(args, "action_trust_radius_min", 1e-8)
        config = {
            "initial_radius": float(initial),
            "minimum_radius": float(minimum),
            "maximum_radius": float(maximum),
            "norm_mode": str(
                getattr(
                    args,
                    "action_trust_norm_mode",
                    "trajectory_l2",
                )
            ).strip().lower(),
            "temporal_basis_knots": int(
                getattr(args, "action_trust_temporal_basis_knots", 0)
                or 0
            ),
            "monotone_fallback": bool(
                getattr(
                    args,
                    "action_trust_monotone_fallback",
                    False,
                )
            ),
            "raw_gradient_fallback": bool(
                getattr(
                    args,
                    "action_trust_raw_gradient_fallback",
                    False,
                )
            ),
            "opposite_direction_poll": bool(
                getattr(
                    args,
                    "action_trust_opposite_direction_poll",
                    False,
                )
            ),
            "radius_restart": bool(
                getattr(args, "action_trust_radius_restart", False)
            ),
            "stage_reset_radius": float(
                getattr(
                    args,
                    "action_trust_stage_reset_radius",
                    0.0,
                )
                or 0.0
            ),
            "failure_patience": int(
                getattr(args, "action_trust_failure_patience", 1)
                or 1
            ),
            "shrink_factor": float(
                getattr(args, "action_trust_shrink_factor", 0.25)
            ),
            "growth_factor": float(
                getattr(args, "action_trust_growth_factor", 2.0)
            ),
            "accept_ratio": float(
                getattr(args, "action_trust_accept_ratio", 0.1)
            ),
            "shrink_ratio": float(
                getattr(args, "action_trust_shrink_ratio", 0.25)
            ),
            "growth_ratio": float(
                getattr(args, "action_trust_growth_ratio", 0.75)
            ),
            "boundary_fraction": float(
                getattr(args, "action_trust_boundary_fraction", 0.8)
            ),
            "force_preconditioner": float(
                getattr(args, "action_trust_force_preconditioner", 1.0)
            ),
            "torque_preconditioner": float(
                getattr(args, "action_trust_torque_preconditioner", 1.0)
            ),
            "knot_gradient_normalization_power": float(
                getattr(
                    args,
                    "action_trust_knot_gradient_normalization_power",
                    0.0,
                )
            ),
            "temporal_preconditioner_power": float(
                getattr(
                    args,
                    "action_trust_temporal_preconditioner_power",
                    0.0,
                )
            ),
            "diagnostic_event_limit": int(
                getattr(
                    args,
                    "action_trust_diagnostic_event_limit",
                    32,
                )
            ),
        }
        horizon_scaling = str(
            getattr(
                args,
                "action_trust_horizon_scaling",
                "none",
            )
        ).strip().lower()
        reference_ctrl_steps = int(
            getattr(
                args,
                "action_trust_reference_ctrl_steps",
                0,
            )
            or 0
        )
        if horizon_scaling not in ("none", "impulse_invariant"):
            raise ValueError(
                "action_trust_horizon_scaling must be 'none' or "
                "'impulse_invariant'"
            )
        if reference_ctrl_steps < 0:
            raise ValueError(
                "action_trust_reference_ctrl_steps must be nonnegative"
            )

        configured_radii = {
            name: float(config[name])
            for name in (
                "minimum_radius",
                "initial_radius",
                "maximum_radius",
                "stage_reset_radius",
            )
        }
        configured_basis_knots = int(config["temporal_basis_knots"])
        actual_ctrl_steps = max(
            1,
            int(
                getattr(
                    self,
                    "num_ctrl_steps",
                    reference_ctrl_steps or 1,
                )
            ),
        )
        horizon_scale = 1.0
        if (
            horizon_scaling == "impulse_invariant"
            and reference_ctrl_steps > 0
            and actual_ctrl_steps > reference_ctrl_steps
        ):
            # Keep the integrated first-order action change fixed. Under an
            # RMS-per-knot norm this is inverse-linear in horizon length;
            # under trajectory L2 it is inverse-square-root.
            ctrl_ratio = (
                float(reference_ctrl_steps) / float(actual_ctrl_steps)
            )
            horizon_scale = (
                ctrl_ratio
                if config["norm_mode"] == "rms_per_knot"
                else np.sqrt(ctrl_ratio)
            )
            for name in configured_radii:
                config[name] *= horizon_scale

            if (
                configured_basis_knots >= 2
                and reference_ctrl_steps >= 2
            ):
                intervals = int(
                    round(
                        float(configured_basis_knots - 1)
                        * float(actual_ctrl_steps - 1)
                        / float(reference_ctrl_steps - 1)
                    )
                )
                config["temporal_basis_knots"] = min(
                    actual_ctrl_steps,
                    max(2, intervals + 1),
                )
        config.update(
            {
                "horizon_scaling": horizon_scaling,
                "reference_ctrl_steps": reference_ctrl_steps,
                "actual_ctrl_steps": actual_ctrl_steps,
                "horizon_scale": float(horizon_scale),
                "configured_radii": configured_radii,
                "configured_temporal_basis_knots": (
                    configured_basis_knots
                ),
            }
        )
        if not (
            np.isfinite(config["minimum_radius"])
            and np.isfinite(config["initial_radius"])
            and np.isfinite(config["maximum_radius"])
            and 0.0 < config["minimum_radius"]
            <= config["initial_radius"]
            <= config["maximum_radius"]
        ):
            raise ValueError(
                "Action trust radii must satisfy "
                "0 < minimum <= initial <= maximum"
            )
        if config["norm_mode"] not in (
            "trajectory_l2",
            "rms_per_knot",
        ):
            raise ValueError(
                "action_trust_norm_mode must be 'trajectory_l2' or "
                "'rms_per_knot'"
            )
        if (
            config["temporal_basis_knots"] < 0
            or config["temporal_basis_knots"] == 1
        ):
            raise ValueError(
                "action_trust_temporal_basis_knots must be zero or at "
                "least two"
            )
        if (
            not np.isfinite(config["stage_reset_radius"])
            or config["stage_reset_radius"] < 0.0
            or config["stage_reset_radius"] > config["maximum_radius"]
        ):
            raise ValueError(
                "action_trust_stage_reset_radius must lie in "
                "[0, maximum_radius]"
            )
        if config["failure_patience"] < 1:
            raise ValueError(
                "action_trust_failure_patience must be positive"
            )
        if not (
            0.0 < config["shrink_factor"] < 1.0
            and config["growth_factor"] > 1.0
        ):
            raise ValueError(
                "Action trust shrink/growth factors must satisfy "
                "0 < shrink < 1 < growth"
            )
        if not (
            0.0 <= config["accept_ratio"]
            <= config["shrink_ratio"]
            < config["growth_ratio"]
            <= 1.0
        ):
            raise ValueError(
                "Action trust ratios must satisfy "
                "0 <= accept <= shrink < growth <= 1"
            )
        if not 0.0 < config["boundary_fraction"] <= 1.0:
            raise ValueError(
                "action_trust_boundary_fraction must be in (0, 1]"
            )
        for name in ("force_preconditioner", "torque_preconditioner"):
            if not np.isfinite(config[name]) or config[name] <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )
        if not (
            np.isfinite(config["knot_gradient_normalization_power"])
            and 0.0 <= config["knot_gradient_normalization_power"] <= 1.0
        ):
            raise ValueError(
                "action_trust_knot_gradient_normalization_power must lie "
                "in [0, 1]"
            )
        if (
            not np.isfinite(config["temporal_preconditioner_power"])
            or config["temporal_preconditioner_power"] < 0.0
        ):
            raise ValueError(
                "action_trust_temporal_preconditioner_power must be "
                "finite and nonnegative"
            )
        if config["diagnostic_event_limit"] < 0:
            raise ValueError(
                "action_trust_diagnostic_event_limit must be nonnegative"
            )
        return config

    def _reset_action_trust_region(self) -> dict:
        config = self._action_trust_config()
        self._action_trust_state = {
            "radius": float(config["initial_radius"]),
            "accepted_steps": 0,
            "rejected_trials": 0,
            "trial_exceptions": 0,
            "raw_gradient_fallback_attempts": 0,
            "raw_gradient_fallback_accepted": 0,
            "opposite_direction_poll_attempts": 0,
            "opposite_direction_poll_accepted": 0,
            "radius_restart_attempts": 0,
            "radius_restart_accepted": 0,
            "radius_restart_blocked": False,
            "event_count": 0,
            "events_dropped": 0,
            "events": [],
            "config": config,
        }
        self._last_action_trust_event = None
        return self._action_trust_state

    def _ensure_action_trust_region(self) -> dict:
        state = getattr(self, "_action_trust_state", None)
        if not isinstance(state, dict):
            state = self._reset_action_trust_region()
        return state

    def _prepare_action_trust_stage(self, stage_index: int) -> dict:
        """Recover a usable radius when contact continuation changes scale."""

        state = self._ensure_action_trust_region()
        radius_before = float(state["radius"])
        reset_radius = float(state["config"]["stage_reset_radius"])
        if int(stage_index) > 0 and reset_radius > 0.0:
            state["radius"] = max(radius_before, reset_radius)
        state["radius_restart_blocked"] = False
        return {
            "trust_radius_before_reset": radius_before,
            "trust_radius_start": float(state["radius"]),
        }

    def _action_trust_layout(self, params: np.ndarray):
        ndof_u = max(1, int(getattr(self, "ndof_u", 1)))
        num_ctrl_steps = max(
            1,
            int(
                getattr(
                    self,
                    "num_ctrl_steps",
                    max(1, np.asarray(params).size // ndof_u),
                )
            ),
        )
        action_dim = ndof_u * num_ctrl_steps
        if action_dim > np.asarray(params).size:
            raise ValueError(
                f"Action layout requires {action_dim} parameters, "
                f"got {np.asarray(params).size}"
            )
        return ndof_u, num_ctrl_steps, action_dim

    def _action_trust_utilization(
        self, action: np.ndarray
    ) -> np.ndarray:
        fn = getattr(self, "action_utilization", None)
        if callable(fn):
            return np.asarray(fn(action), dtype=np.float64)
        return np.asarray(action, dtype=np.float64)

    def _action_trust_step_metrics(
        self, params: np.ndarray, trial: np.ndarray
    ) -> dict:
        ndof_u, num_ctrl_steps, action_dim = (
            self._action_trust_layout(params)
        )
        current_u = self._action_trust_utilization(
            np.asarray(params[:action_dim], dtype=np.float64)
        ).reshape(num_ctrl_steps, ndof_u)
        trial_u = self._action_trust_utilization(
            np.asarray(trial[:action_dim], dtype=np.float64)
        ).reshape(num_ctrl_steps, ndof_u)
        delta = trial_u - current_u
        knot_norms = np.linalg.norm(delta, axis=1)
        trajectory_l2 = float(np.linalg.norm(delta))
        state = self._ensure_action_trust_region()
        if state["config"]["norm_mode"] == "rms_per_knot":
            trajectory_norm = trajectory_l2 / np.sqrt(
                float(num_ctrl_steps)
            )
        else:
            trajectory_norm = trajectory_l2
        force_stop = min(3, ndof_u)
        torque_start = min(3, ndof_u)
        torque_stop = min(6, ndof_u)
        return {
            "trajectory_utilization_norm": float(trajectory_norm),
            "trajectory_utilization_l2": trajectory_l2,
            "max_knot_utilization_norm": (
                float(np.max(knot_norms)) if knot_norms.size else 0.0
            ),
            "force_utilization_norm": float(
                np.linalg.norm(delta[:, :force_stop])
            ),
            "torque_utilization_norm": float(
                np.linalg.norm(delta[:, torque_start:torque_stop])
            ),
        }

    @staticmethod
    def _action_trust_project_temporal_basis(
        values: np.ndarray,
        basis_knots: int,
    ) -> np.ndarray:
        """Project a knot trajectory onto a piecewise-linear time basis."""

        values = np.asarray(values, dtype=np.float64)
        num_steps = int(values.shape[0])
        if basis_knots <= 0 or basis_knots >= num_steps:
            return np.array(values, copy=True)

        coordinates = np.linspace(
            0.0,
            float(basis_knots - 1),
            num_steps,
        )
        left = np.floor(coordinates).astype(np.int64)
        right = np.minimum(left + 1, basis_knots - 1)
        weight_right = coordinates - left
        basis = np.zeros((num_steps, basis_knots), dtype=np.float64)
        rows = np.arange(num_steps)
        basis[rows, left] += 1.0 - weight_right
        basis[rows, right] += weight_right
        gram = basis.T.dot(basis)
        coefficients = np.linalg.solve(
            gram,
            basis.T.dot(values),
        )
        return basis.dot(coefficients)

    def _scale_action_direction_to_trust_radius(
        self,
        params: np.ndarray,
        direction: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Cap total trajectory utilization displacement at ``radius``."""

        _, _, action_dim = self._action_trust_layout(params)
        direction = np.asarray(direction, dtype=np.float64)
        if radius <= 0.0 or not np.any(direction[:action_dim]):
            return np.zeros_like(direction)

        def step_norm(scale: float) -> float:
            trial = np.asarray(params, dtype=np.float64).copy()
            trial[:action_dim] += scale * direction[:action_dim]
            return self._action_trust_step_metrics(
                params, trial
            )["trajectory_utilization_norm"]

        if step_norm(1.0) <= radius:
            return np.array(direction, copy=True, dtype=np.float64)
        lo, hi = 0.0, 1.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if step_norm(mid) <= radius:
                lo = mid
            else:
                hi = mid
        scaled = np.array(direction, copy=True, dtype=np.float64)
        scaled[:action_dim] *= lo
        return scaled

    def _action_trust_direction(
        self,
        params: np.ndarray,
        gradient: np.ndarray,
        *,
        temporal_basis_knots=None,
    ) -> np.ndarray:
        """Build a positive-diagonally preconditioned descent direction."""

        state = self._ensure_action_trust_region()
        config = state["config"]
        basis_knots = (
            config["temporal_basis_knots"]
            if temporal_basis_knots is None
            else int(temporal_basis_knots)
        )
        if basis_knots < 0 or basis_knots == 1:
            raise ValueError(
                "temporal_basis_knots must be zero or at least two"
            )
        ndof_u, num_ctrl_steps, action_dim = (
            self._action_trust_layout(params)
        )
        direction = np.zeros_like(params, dtype=np.float64)
        action_gradient = np.asarray(
            gradient[:action_dim], dtype=np.float64
        ).reshape(num_ctrl_steps, ndof_u).copy()
        trainable_mask = getattr(
            self,
            "_action_optimizer_trainable_mask",
            None,
        )
        action_trainable = np.ones(
            (num_ctrl_steps, ndof_u),
            dtype=bool,
        )
        if trainable_mask is not None:
            trainable_mask = np.asarray(
                trainable_mask,
                dtype=bool,
            ).reshape(-1)
            if trainable_mask.shape != direction.shape:
                raise ValueError(
                    "action optimizer trainable mask shape "
                    f"{trainable_mask.shape} does not match "
                    f"parameter shape {direction.shape}"
                )
            action_trainable = trainable_mask[:action_dim].reshape(
                num_ctrl_steps,
                ndof_u,
            )
            action_gradient[~action_trainable] = 0.0

        remaining = np.arange(
            num_ctrl_steps, 0, -1, dtype=np.float64
        )
        temporal = remaining ** (
            -config["temporal_preconditioner_power"]
        )
        axis = np.ones(ndof_u, dtype=np.float64)
        axis[: min(3, ndof_u)] *= config["force_preconditioner"]
        if ndof_u > 3:
            axis[3 : min(6, ndof_u)] *= config[
                "torque_preconditioner"
            ]

        preconditioned = (
            action_gradient
            * temporal[:, None]
            * axis[None, :]
        )
        normalization_power = config[
            "knot_gradient_normalization_power"
        ]
        if normalization_power > 0.0:
            knot_norms = np.linalg.norm(preconditioned, axis=1)
            max_norm = float(np.max(knot_norms))
            if np.isfinite(max_norm) and max_norm > 0.0:
                floor = max(1e-12, 1e-3 * max_norm)
                preconditioned *= (
                    np.maximum(knot_norms, floor)
                    ** (-normalization_power)
                )[:, None]
        preconditioned = self._action_trust_project_temporal_basis(
            preconditioned,
            basis_knots,
        )
        preconditioned[~action_trainable] = 0.0
        if float(np.sum(action_gradient * preconditioned)) <= 0.0:
            preconditioned = (
                self._action_trust_project_temporal_basis(
                    action_gradient,
                    basis_knots,
                )
                * axis[None, :]
            )
            preconditioned[~action_trainable] = 0.0

        action_step_scale = float(
            getattr(
                self.args,
                "direct_planar_action_step_scale",
                1.0,
            )
            or 1.0
        )
        if action_step_scale <= 0.0:
            action_step_scale = 1.0
        direction[:action_dim] = (
            -preconditioned * action_step_scale
        ).reshape(-1)
        direction = self._project_action_direction_at_bounds(
            params, direction
        )
        if trainable_mask is not None:
            direction[~trainable_mask] = 0.0
        return self._scale_action_direction_to_trust_radius(
            params,
            direction,
            float(state["radius"]),
        )

    def _bounded_action_trial(
        self, params: np.ndarray, step: np.ndarray
    ):
        """Apply an action-only step without touching morphology."""

        _, _, action_dim = self._action_trust_layout(params)
        trial = np.asarray(params, dtype=np.float64).copy()
        trial[:action_dim] += np.asarray(
            step[:action_dim], dtype=np.float64
        )
        bounds = self._bounds()
        for idx in range(min(action_dim, len(bounds))):
            low, high = bounds[idx]
            if low is not None:
                trial[idx] = max(trial[idx], float(low))
            if high is not None:
                trial[idx] = min(trial[idx], float(high))
        return trial

    def _action_trust_search(
        self,
        params: np.ndarray,
        f: float,
        direction: np.ndarray,
        gradient: np.ndarray,
        *,
        c1: float,
        maxls: int,
    ):
        """Search one action step and adapt the persistent trust radius."""

        state = self._ensure_action_trust_region()
        config = state["config"]
        start_radius = float(state["radius"])
        base_direction = np.asarray(direction, dtype=np.float64).copy()
        base_trial = self._bounded_action_trial(params, base_direction)
        base_norm = self._action_trust_step_metrics(
            params, base_trial
        )["trajectory_utilization_norm"]
        trial_history = []
        accepted_params = None
        accepted_f = None
        accepted_info = None
        accepted = False
        returned_alpha = 0.0

        for trial_index in range(1, int(maxls) + 1):
            radius_used = float(state["radius"])
            trial_direction = (
                self._scale_action_direction_to_trust_radius(
                    params,
                    base_direction,
                    radius_used,
                )
            )
            trial = self._bounded_action_trial(
                params, trial_direction
            )
            feasible_step = trial - params
            metrics = self._action_trust_step_metrics(params, trial)
            predicted_reduction = -float(
                np.dot(gradient, feasible_step)
            )
            f_trial = float("inf")
            info_trial = {}
            error = None
            if (
                np.isfinite(predicted_reduction)
                and predicted_reduction > 0.0
                and metrics["trajectory_utilization_norm"] > 0.0
            ):
                try:
                    f_trial, info_trial = self.forward(
                        trial, backward_flag=False
                    )
                    f_trial = float(f_trial)
                except Exception as exc:
                    error = repr(exc)
                    state["trial_exceptions"] += 1

            actual_reduction = (
                float(f) - f_trial
                if np.isfinite(f_trial)
                else float("-inf")
            )
            agreement_ratio = (
                actual_reduction / predicted_reduction
                if (
                    np.isfinite(actual_reduction)
                    and np.isfinite(predicted_reduction)
                    and predicted_reduction > 0.0
                )
                else float("-inf")
            )
            armijo_ok = (
                np.isfinite(f_trial)
                and actual_reduction
                >= float(c1) * predicted_reduction
            )
            accepted_by_agreement = bool(
                armijo_ok
                and agreement_ratio >= config["accept_ratio"]
            )
            accepted_by_monotone_fallback = bool(
                config["monotone_fallback"]
                and np.isfinite(f_trial)
                and actual_reduction > 0.0
            )
            accepted = bool(
                accepted_by_agreement
                or accepted_by_monotone_fallback
            )
            on_boundary = (
                metrics["trajectory_utilization_norm"]
                >= config["boundary_fraction"] * radius_used
            )

            if (
                not np.isfinite(agreement_ratio)
                or agreement_ratio < config["shrink_ratio"]
            ):
                next_radius = max(
                    config["minimum_radius"],
                    radius_used * config["shrink_factor"],
                )
            elif (
                accepted
                and agreement_ratio >= config["growth_ratio"]
                and on_boundary
            ):
                next_radius = min(
                    config["maximum_radius"],
                    radius_used * config["growth_factor"],
                )
            else:
                next_radius = radius_used
            if not accepted and next_radius >= radius_used:
                next_radius = max(
                    config["minimum_radius"],
                    radius_used * config["shrink_factor"],
                )

            def diagnostic_value(value):
                return float(value) if np.isfinite(value) else None

            trial_record = {
                "trial": int(trial_index),
                "radius": radius_used,
                "next_radius": float(next_radius),
                "predicted_reduction": diagnostic_value(
                    predicted_reduction
                ),
                "actual_reduction": diagnostic_value(actual_reduction),
                "agreement_ratio": diagnostic_value(agreement_ratio),
                "armijo_ok": bool(armijo_ok),
                "accepted": bool(accepted),
                "accepted_by_monotone_fallback": bool(
                    accepted_by_monotone_fallback
                    and not accepted_by_agreement
                ),
                **metrics,
            }
            if error is not None:
                trial_record["error"] = error
            trial_history.append(trial_record)
            state["radius"] = float(next_radius)

            if accepted:
                accepted_params = trial
                accepted_f = f_trial
                accepted_info = info_trial
                state["accepted_steps"] += 1
                returned_alpha = (
                    metrics["trajectory_utilization_norm"] / base_norm
                    if base_norm > 0.0
                    else 1.0
                )
                break
            state["rejected_trials"] += 1
            if state["radius"] <= config["minimum_radius"]:
                break

        event = {
            "accepted": bool(accepted),
            "start_radius": start_radius,
            "final_radius": float(state["radius"]),
            "alpha": float(returned_alpha),
            "trials": len(trial_history),
        }
        if not accepted or len(trial_history) > 1:
            event["trial_history"] = trial_history
        if trial_history:
            event.update(
                {
                    key: trial_history[-1][key]
                    for key in (
                        "radius",
                        "next_radius",
                        "predicted_reduction",
                        "actual_reduction",
                        "agreement_ratio",
                        "armijo_ok",
                        "accepted_by_monotone_fallback",
                        "trajectory_utilization_norm",
                        "trajectory_utilization_l2",
                        "max_knot_utilization_norm",
                        "force_utilization_norm",
                        "torque_utilization_norm",
                    )
                }
            )
        state["event_count"] += 1
        event_limit = int(config["diagnostic_event_limit"])
        if event_limit > 0:
            state["events"].append(event)
            if len(state["events"]) > event_limit:
                del state["events"][:-event_limit]
                state["events_dropped"] += 1
        else:
            state["events_dropped"] += 1
        self._last_action_trust_event = event
        return (
            accepted,
            accepted_params,
            accepted_f,
            accepted_info,
            float(returned_alpha),
            len(trial_history),
        )

    def _action_trust_monotone_poll(
        self,
        params: np.ndarray,
        f: float,
        direction: np.ndarray,
        *,
        maxls: int,
    ):
        """Poll one direction using measured reduction instead of a gradient."""

        state = self._ensure_action_trust_region()
        config = state["config"]
        start_radius = float(state["radius"])
        trial_history = []
        accepted_params = None
        accepted_f = None
        accepted_info = None
        returned_alpha = 0.0
        base_trial = self._bounded_action_trial(params, direction)
        base_norm = self._action_trust_step_metrics(
            params, base_trial
        )["trajectory_utilization_norm"]

        for trial_index in range(1, int(maxls) + 1):
            radius_used = float(state["radius"])
            trial_direction = self._scale_action_direction_to_trust_radius(
                params,
                direction,
                radius_used,
            )
            trial = self._bounded_action_trial(params, trial_direction)
            metrics = self._action_trust_step_metrics(params, trial)
            f_trial = float("inf")
            info_trial = {}
            error = None
            if metrics["trajectory_utilization_norm"] > 0.0:
                try:
                    f_trial, info_trial = self.forward(
                        trial, backward_flag=False
                    )
                    f_trial = float(f_trial)
                except Exception as exc:
                    error = repr(exc)
                    state["trial_exceptions"] += 1
            actual_reduction = (
                float(f) - f_trial
                if np.isfinite(f_trial)
                else float("-inf")
            )
            accepted = bool(
                np.isfinite(actual_reduction)
                and actual_reduction > 0.0
            )
            next_radius = (
                radius_used
                if accepted
                else max(
                    config["minimum_radius"],
                    radius_used * config["shrink_factor"],
                )
            )
            record = {
                "trial": int(trial_index),
                "radius": radius_used,
                "next_radius": float(next_radius),
                "actual_reduction": (
                    float(actual_reduction)
                    if np.isfinite(actual_reduction)
                    else None
                ),
                "accepted": accepted,
                **metrics,
            }
            if error is not None:
                record["error"] = error
            trial_history.append(record)
            state["radius"] = float(next_radius)
            if accepted:
                accepted_params = trial
                accepted_f = f_trial
                accepted_info = info_trial
                state["accepted_steps"] += 1
                returned_alpha = (
                    metrics["trajectory_utilization_norm"] / base_norm
                    if base_norm > 0.0
                    else 1.0
                )
                break
            state["rejected_trials"] += 1
            if state["radius"] <= config["minimum_radius"]:
                break

        accepted = accepted_params is not None
        event = {
            "accepted": bool(accepted),
            "direction_variant": "opposite_direction_poll",
            "start_radius": start_radius,
            "final_radius": float(state["radius"]),
            "alpha": float(returned_alpha),
            "trials": len(trial_history),
            "trial_history": trial_history,
        }
        if trial_history:
            event.update(trial_history[-1])
        state["event_count"] += 1
        event_limit = int(config["diagnostic_event_limit"])
        if event_limit > 0:
            state["events"].append(event)
            if len(state["events"]) > event_limit:
                del state["events"][:-event_limit]
                state["events_dropped"] += 1
        else:
            state["events_dropped"] += 1
        self._last_action_trust_event = event
        return (
            accepted,
            accepted_params,
            accepted_f,
            accepted_info,
            float(returned_alpha),
            len(trial_history),
        )

    def _action_trust_search_with_fallback(
        self,
        params: np.ndarray,
        f: float,
        direction: np.ndarray,
        gradient: np.ndarray,
        *,
        c1: float,
        maxls: int,
    ):
        """Try configured measured-loss fallbacks after a rejected step."""

        state = self._ensure_action_trust_region()
        config = state["config"]
        start_radius = float(state["radius"])
        restart_radius = min(
            config["maximum_radius"],
            max(
                config["initial_radius"],
                config["stage_reset_radius"],
            ),
        )
        primary = self._action_trust_search(
            params,
            f,
            direction,
            gradient,
            c1=c1,
            maxls=maxls,
        )
        if primary[0]:
            state["radius_restart_blocked"] = False
            return primary
        primary_event = dict(self._last_action_trust_event or {})
        primary_final_radius = float(state["radius"])

        if (
            not config["raw_gradient_fallback"]
            or config["temporal_basis_knots"] <= 0
        ):
            if not config["opposite_direction_poll"]:
                return primary

            poll_direction = -np.asarray(direction, dtype=np.float64)
            state["radius"] = start_radius
            if config["radius_restart"] and start_radius < restart_radius:
                state["radius"] = restart_radius
                poll_direction = -self._action_trust_direction(
                    params,
                    gradient,
                )
            state["opposite_direction_poll_attempts"] += 1
            poll = self._action_trust_monotone_poll(
                params,
                f,
                poll_direction,
                maxls=maxls,
            )
            total_trials = int(primary[-1]) + int(poll[-1])
            if poll[0]:
                state["radius_restart_blocked"] = False
                state["opposite_direction_poll_accepted"] += 1
                return (
                    poll[0],
                    poll[1],
                    poll[2],
                    poll[3],
                    poll[4],
                    total_trials,
                )

            state["radius"] = min(
                primary_final_radius,
                float(state["radius"]),
            )
            final_event = dict(self._last_action_trust_event or {})
            final_event.update(
                {
                    "primary_trials": int(primary[-1]),
                    "primary_final_radius": primary_final_radius,
                    "final_radius": float(state["radius"]),
                }
            )
            if state["events"]:
                state["events"][-1].update(final_event)
            self._last_action_trust_event = final_event
            return (
                False,
                None,
                None,
                None,
                0.0,
                total_trials,
            )

        state["radius"] = start_radius
        raw_direction = self._action_trust_direction(
            params,
            gradient,
            temporal_basis_knots=0,
        )
        raw_descent = float(np.dot(gradient, raw_direction))
        if (
            not np.isfinite(raw_descent)
            or raw_descent >= 0.0
            or float(np.linalg.norm(raw_direction)) < 1e-12
            or np.allclose(
                raw_direction,
                direction,
                rtol=1e-10,
                atol=1e-14,
            )
        ):
            state["radius"] = primary_final_radius
            self._last_action_trust_event = primary_event
            if not config["opposite_direction_poll"]:
                return primary

            poll_direction = -np.asarray(direction, dtype=np.float64)
            state["radius"] = start_radius
            if config["radius_restart"] and start_radius < restart_radius:
                state["radius"] = restart_radius
                poll_direction = -self._action_trust_direction(
                    params,
                    gradient,
                )
            state["opposite_direction_poll_attempts"] += 1
            poll = self._action_trust_monotone_poll(
                params,
                f,
                poll_direction,
                maxls=maxls,
            )
            total_trials = int(primary[-1]) + int(poll[-1])
            if poll[0]:
                state["radius_restart_blocked"] = False
                state["opposite_direction_poll_accepted"] += 1
                return (
                    poll[0],
                    poll[1],
                    poll[2],
                    poll[3],
                    poll[4],
                    total_trials,
                )
            state["radius"] = min(
                primary_final_radius,
                float(state["radius"]),
            )
            final_event = dict(self._last_action_trust_event or {})
            final_event.update(
                {
                    "primary_trials": int(primary[-1]),
                    "primary_final_radius": primary_final_radius,
                    "final_radius": float(state["radius"]),
                }
            )
            if state["events"]:
                state["events"][-1].update(final_event)
            self._last_action_trust_event = final_event
            return (
                False,
                None,
                None,
                None,
                0.0,
                total_trials,
            )

        state["raw_gradient_fallback_attempts"] += 1
        fallback = self._action_trust_search(
            params,
            f,
            raw_direction,
            gradient,
            c1=c1,
            maxls=maxls,
        )
        fallback_event = dict(self._last_action_trust_event or {})
        fallback_event.update(
            {
                "direction_variant": "raw_gradient_fallback",
                "primary_trials": int(primary[-1]),
                "primary_final_radius": primary_final_radius,
            }
        )
        total_trials = int(primary[-1]) + int(fallback[-1])
        if fallback[0]:
            state["radius_restart_blocked"] = False
            state["raw_gradient_fallback_accepted"] += 1
            if state["events"]:
                state["events"][-1].update(fallback_event)
            self._last_action_trust_event = fallback_event
            return (
                fallback[0],
                fallback[1],
                fallback[2],
                fallback[3],
                fallback[4],
                total_trials,
            )

        failed_radii = [
            primary_final_radius,
            float(state["radius"]),
        ]
        if config["opposite_direction_poll"]:
            poll_direction = -raw_direction
            state["radius"] = start_radius
            if config["radius_restart"] and start_radius < restart_radius:
                state["radius"] = restart_radius
                poll_direction = -self._action_trust_direction(
                    params,
                    gradient,
                    temporal_basis_knots=0,
                )
            state["opposite_direction_poll_attempts"] += 1
            poll = self._action_trust_monotone_poll(
                params,
                f,
                poll_direction,
                maxls=maxls,
            )
            total_trials += int(poll[-1])
            if poll[0]:
                state["radius_restart_blocked"] = False
                state["opposite_direction_poll_accepted"] += 1
                return (
                    poll[0],
                    poll[1],
                    poll[2],
                    poll[3],
                    poll[4],
                    total_trials,
                )
            failed_radii.append(float(state["radius"]))

        at_radius_floor = (
            min(failed_radii)
            <= config["minimum_radius"] * (1.0 + 1e-12)
        )
        if (
            config["radius_restart"]
            and at_radius_floor
            and start_radius < restart_radius
            and not state["radius_restart_blocked"]
        ):
            state["radius_restart_attempts"] += 1
            state["radius_restart_blocked"] = True
            state["radius"] = restart_radius
            restart_direction = self._action_trust_direction(
                params,
                gradient,
            )
            restarted = self._action_trust_search(
                params,
                f,
                restart_direction,
                gradient,
                c1=c1,
                maxls=maxls,
            )
            total_trials += int(restarted[-1])
            restart_event = dict(self._last_action_trust_event or {})
            restart_event["direction_variant"] = "radius_restart"
            restart_event["restart_radius"] = float(restart_radius)
            if state["events"]:
                state["events"][-1].update(restart_event)
            self._last_action_trust_event = restart_event
            if restarted[0]:
                state["radius_restart_accepted"] += 1
                state["radius_restart_blocked"] = False
                return (
                    restarted[0],
                    restarted[1],
                    restarted[2],
                    restarted[3],
                    restarted[4],
                    total_trials,
                )
            failed_radii.append(float(state["radius"]))

        state["radius"] = min(failed_radii)
        final_event = dict(self._last_action_trust_event or fallback_event)
        final_event["final_radius"] = float(state["radius"])
        if state["events"]:
            state["events"][-1].update(final_event)
        self._last_action_trust_event = final_event
        return (
            False,
            None,
            None,
            None,
            0.0,
            total_trials,
        )

    def _action_trust_diagnostics(self) -> dict:
        state = self._ensure_action_trust_region()
        return {
            "radius": float(state["radius"]),
            "accepted_steps": int(state["accepted_steps"]),
            "rejected_trials": int(state["rejected_trials"]),
            "trial_exceptions": int(state["trial_exceptions"]),
            "raw_gradient_fallback_attempts": int(
                state["raw_gradient_fallback_attempts"]
            ),
            "raw_gradient_fallback_accepted": int(
                state["raw_gradient_fallback_accepted"]
            ),
            "opposite_direction_poll_attempts": int(
                state["opposite_direction_poll_attempts"]
            ),
            "opposite_direction_poll_accepted": int(
                state["opposite_direction_poll_accepted"]
            ),
            "radius_restart_attempts": int(
                state["radius_restart_attempts"]
            ),
            "radius_restart_accepted": int(
                state["radius_restart_accepted"]
            ),
            "event_count": int(state["event_count"]),
            "events_dropped": int(state["events_dropped"]),
            "config": dict(state["config"]),
            "events": list(state["events"]),
        }

    def _project_action_direction_at_bounds(
        self, params: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Remove action components that point out of an active box bound."""
        projected = np.asarray(direction, dtype=np.float64).copy()
        action_dim = self.ndof_u * self.num_ctrl_steps
        bounds = self._bounds()
        tol = 1e-12
        for idx in range(min(action_dim, len(bounds))):
            lo, hi = bounds[idx]
            value = float(params[idx])
            if lo is not None and value <= float(lo) + tol and projected[idx] < 0.0:
                projected[idx] = 0.0
            if hi is not None and value >= float(hi) - tol and projected[idx] > 0.0:
                projected[idx] = 0.0
        return projected

    def _contact_continuation_budgets(self, maxiter: int):
        raw_scales = getattr(self.args, "contact_continuation_scales", None)
        scales_from_args = raw_scales is not None
        if raw_scales is None:
            schedule = getattr(self.task, "contact_continuation_scales", None)
            raw_scales = schedule() if callable(schedule) else (1.0,)
        if isinstance(raw_scales, str):
            scales = tuple(float(v) for v in raw_scales.split(",") if v.strip())
        else:
            scales = tuple(float(v) for v in (raw_scales or (1.0,)))
        if not scales or any((not np.isfinite(v) or v < 0.0) for v in scales):
            raise ValueError(f"Invalid contact continuation schedule: {raw_scales!r}")
        if scales[-1] != 1.0:
            scales = scales + (1.0,)

        raw_weights = getattr(self.args, "contact_continuation_weights", None)
        if raw_weights is None and not scales_from_args:
            schedule = getattr(self.task, "contact_continuation_weights", None)
            raw_weights = schedule() if callable(schedule) else None
        if raw_weights is None:
            weights = (1.0,) * len(scales)
        elif isinstance(raw_weights, str):
            weights = tuple(float(v) for v in raw_weights.split(",") if v.strip())
        else:
            weights = tuple(float(v) for v in raw_weights)
        if len(weights) != len(scales) or any(
            not np.isfinite(value) or value <= 0.0 for value in weights
        ):
            raise ValueError(
                "Contact continuation weights must be positive and match the "
                f"{len(scales)} scales: {raw_weights!r}"
            )
        if maxiter <= 0:
            return ()
        if maxiter < len(scales):
            return ((1.0, maxiter),)
        normalized = np.asarray(weights, dtype=np.float64)
        normalized /= float(np.sum(normalized))
        exact = normalized * maxiter
        budgets = np.maximum(1, np.floor(exact).astype(np.int64))
        while int(np.sum(budgets)) > maxiter:
            candidates = np.flatnonzero(budgets > 1)
            idx = min(
                candidates,
                key=lambda item: (exact[item] - budgets[item], -int(item)),
            )
            budgets[int(idx)] -= 1
        remainder = maxiter - int(np.sum(budgets))
        if remainder > 0:
            order = np.argsort(-(exact - budgets), kind="stable")
            for idx in order[:remainder]:
                budgets[int(idx)] += 1
        return tuple(
            (scale, int(budget)) for scale, budget in zip(scales, budgets)
        )

    def _direct_planar_delta_diagnostics(self, params0: np.ndarray, params: np.ndarray) -> dict:
        action_dim = self.ndof_u * self.num_ctrl_steps
        cage0 = np.asarray(params0[action_dim:], dtype=np.float64)
        cage = np.asarray(params[action_dim:], dtype=np.float64)
        diag = {
            "action_delta_norm": float(np.linalg.norm(params[:action_dim] - params0[:action_dim])),
            "cage_delta_norm": float(np.linalg.norm(cage - cage0)),
            "cage_delta_max": float(np.max(np.abs(cage - cage0))) if cage.size else 0.0,
            "blocks": [],
            "design_collision_rejections": int(getattr(self, "_design_collision_rejections", 0)),
        }
        if self._design_collision_initial_report is not None:
            diag["design_collision_initial"] = self._design_collision_initial_report
        if self._design_collision_final_report is not None:
            diag["design_collision_final"] = self._design_collision_final_report
        if self._design_collision_last_report is not None:
            diag["design_collision_last_rejection"] = self._design_collision_last_report
        if not cage.size:
            return diag
        try:
            from bilevel.parameterization.geometry import (
                max_vertex_displacement,
            )
        except Exception:
            max_vertex_displacement = None
        design_params0 = design_params = None
        try:
            cage0_full = np.asarray(params0[action_dim : action_dim + self.ndof_cage], dtype=np.float64)
            cage_full = np.asarray(params[action_dim : action_dim + self.ndof_cage], dtype=np.float64)
            design_params0 = np.asarray(
                self.parameterize_morphology_numpy(
                    cage0_full,
                    generate_mesh=False,
                ),
                dtype=np.float64,
            )
            design_params = np.asarray(
                self.parameterize_morphology_numpy(
                    cage_full,
                    generate_mesh=False,
                ),
                dtype=np.float64,
            )
        except Exception:
            design_params0 = design_params = None
        for block in self._direct_planar_abs_blocks():
            s = int(block["abs_start"])
            e = int(block["abs_stop"])
            delta = np.asarray(params[s:e] - params0[s:e], dtype=np.float64)
            block_diag = {
                "link_name": str(block.get("link_name", "")),
                "body_name": str(block.get("body_name", "")),
                "deformable": bool(block.get("deformable", False)),
                "design_enabled": bool(block.get("design_enabled", True)),
                "delta_norm": float(np.linalg.norm(delta)),
                "delta_max": float(np.max(np.abs(delta))) if delta.size else 0.0,
            }
            if block.get("face_mask") is not None:
                mask = [bool(v) for v in block.get("face_mask")]
                block_diag["face_mask"] = mask
                block_diag["num_constrained_faces"] = int(sum(mask))
            if max_vertex_displacement is not None and delta.size:
                try:
                    block_diag["max_vertex_displacement"] = float(max_vertex_displacement(params0[s:e], params[s:e]))
                except Exception:
                    pass
            if design_params0 is not None and design_params is not None:
                try:
                    tool_index = int(block.get("tool_index", -1))
                    rec = self.design_bundle.spec.tool_records[tool_index]
                    if rec.p1_slice is not None:
                        t0 = design_params0[rec.p1_slice][9:12]
                        t1 = design_params[rec.p1_slice][9:12]
                        block_diag["joint_translation_delta"] = (t1 - t0).tolist()
                        block_diag["joint_translation_delta_norm"] = float(np.linalg.norm(t1 - t0))
                    if rec.p3_slice is not None and rec.p3_slice.stop > rec.p3_slice.start:
                        p30 = design_params0[rec.p3_slice].reshape(-1, 3)
                        p31 = design_params[rec.p3_slice].reshape(-1, 3)
                        span0 = np.ptp(p30, axis=0) if p30.shape[0] else np.zeros(3)
                        span1 = np.ptp(p31, axis=0) if p31.shape[0] else np.zeros(3)
                        block_diag["contact_span_initial"] = span0.tolist()
                        block_diag["contact_span_final"] = span1.tolist()
                        block_diag["contact_span_delta"] = (span1 - span0).tolist()
                except Exception:
                    pass
            diag["blocks"].append(block_diag)
        return diag

    def _maybe_auto_visualize(self, params: np.ndarray, *, record: bool = False, record_path: Optional[str] = None) -> None:
        if self.visualize_every_n is None or self.visualize_every_n <= 0:
            return
        self._auto_vis_counter += 1
        if (self._auto_vis_counter % self.visualize_every_n) != 0:
            return
        if self.optimize_design and self.design_bundle is not None:
            _, cage = self.unpack_params(params)
            self.apply_morphology(cage, generate_mesh=True)

        self._apply_replay_camera()
        self.sim.viewer_options.speed = 0.2
        SimRenderer.replay(self.sim, record=record, record_path=record_path)

    def _direct_planar_log_iteration(self, params: np.ndarray, f: float, info: dict) -> None:
        print_info("iteration ", len(self.f_log), ", num_sim = ", self.num_sim, ", Objective = ", f, info)
        self.f_log.append(np.array([self.num_sim, f], dtype=np.float64))
        self._maybe_auto_visualize(
            params,
            record=getattr(self.args, "record", False),
            record_path=getattr(self.args, "record_file_name", None),
        )

    def _direct_planar_block_direction(self, params: np.ndarray, grad: np.ndarray, block: str):
        g = self._direct_planar_project_vector(params, grad)
        direction = np.zeros_like(params, dtype=np.float64)
        action_dim = self.ndof_u * self.num_ctrl_steps

        if block == "action":
            direction = self._action_trust_direction(params, g)
            descent = float(
                np.dot(g[:action_dim], direction[:action_dim])
            )
        elif block == "design":
            design_step_scale = float(getattr(self.args, "direct_planar_design_step_scale", 1.0) or 1.0)
            if design_step_scale <= 0.0:
                design_step_scale = 1.0
            for dp_block in self._direct_planar_abs_blocks():
                if not bool(dp_block.get("deformable", False)) or not bool(dp_block.get("design_enabled", True)):
                    continue
                s = int(dp_block["abs_start"])
                e = int(dp_block["abs_stop"])
                direction[s:e] = -g[s:e] * design_step_scale
            direction = self._direct_planar_scale_direction(params, direction)
            # Sum only trainable design blocks.  Legacy layouts may contain
            # fixed zero prefixes that are absent from the unified shadow
            # layout; including those zeros changes BLAS reduction order by
            # one floating-point bit despite identical active vectors.
            descent = float(
                math.fsum(
                    float(np.dot(g[s:e], direction[s:e]))
                    for dp_block in self._direct_planar_abs_blocks()
                    if bool(dp_block.get("deformable", False))
                    and bool(dp_block.get("design_enabled", True))
                    for s, e in (
                        (
                            int(dp_block["abs_start"]),
                            int(dp_block["abs_stop"]),
                        ),
                    )
                )
            )
        else:
            raise ValueError(f"Unknown direct-planar block {block!r}")

        return g, direction, descent

    def _direct_planar_armijo_search(
        self,
        params: np.ndarray,
        f: float,
        direction: np.ndarray,
        gradient: np.ndarray,
        *,
        c1: float,
        maxls: int,
    ):
        accepted = False
        alpha = 1.0
        accepted_params = None
        accepted_f = None
        accepted_info = None
        trials = 0
        trial_history = []
        baseline_contact = self._directional_contact_signature()
        for trials in range(1, int(maxls) + 1):
            trial, ok = self._direct_planar_retract_params(params, alpha * direction)
            trial_record = {
                "trial": int(trials),
                "alpha": float(alpha),
                "retraction_accepted": bool(ok),
            }
            if ok:
                feasible_step = trial - params
                trial_descent = float(np.dot(gradient, feasible_step))
                trial_record.update(
                    {
                        "step_norm": float(
                            np.linalg.norm(feasible_step)
                        ),
                        "predicted_reduction": float(-trial_descent),
                        "physical_displacement": (
                            self._direct_planar_physical_step_diagnostics(
                                params,
                                trial,
                            )
                        ),
                    }
                )
                if not np.isfinite(trial_descent) or trial_descent >= 0.0:
                    trial_record["status"] = "non_descent_after_retraction"
                    trial_history.append(trial_record)
                    alpha *= 0.5
                    continue
                try:
                    f_trial, info_trial = self.forward(trial, backward_flag=False)
                    trial_record["actual_reduction"] = float(
                        f - f_trial
                    )
                    contact = self._directional_contact_signature()
                    trial_record["contact_signature"] = contact
                    trial_record["contact_set_changed"] = (
                        contact != baseline_contact
                    )
                    trial_record["stage_acceptance"] = (
                        self._directional_stage_acceptance()
                    )
                    trial_record["solver"] = (
                        dict(self.sim.get_solver_diagnostics())
                        if hasattr(self.sim, "get_solver_diagnostics")
                        else {}
                    )
                except Exception as exc:
                    f_trial, info_trial = float("inf"), {}
                    trial_record["error"] = repr(exc)
                    trial_record["status"] = "forward_failed"
                armijo_bound = float(f + c1 * trial_descent)
                trial_record["objective"] = float(f_trial)
                trial_record["armijo_bound"] = armijo_bound
                if np.isfinite(f_trial) and f_trial <= f + c1 * trial_descent:
                    accepted = True
                    accepted_params = trial
                    accepted_f = float(f_trial)
                    accepted_info = info_trial
                    trial_record["status"] = "accepted"
                    trial_history.append(trial_record)
                    break
                trial_record.setdefault("status", "insufficient_decrease")
            else:
                trial_record["status"] = "retraction_rejected"
            trial_history.append(trial_record)
            alpha *= 0.5
        event = {
            "accepted": bool(accepted),
            "trials": trial_history,
            "baseline_contact_signature": baseline_contact,
            "physical_metric": bool(
                getattr(
                    self.args,
                    "direct_planar_physical_metric",
                    False,
                )
            ),
        }
        self._last_direct_planar_armijo_event = event
        events = getattr(self, "_direct_planar_armijo_events", None)
        if events is None:
            events = []
            self._direct_planar_armijo_events = events
        events.append(event)
        limit = max(
            0,
            int(
                getattr(
                    self.args,
                    "direct_planar_diagnostic_event_limit",
                    32,
                )
                or 0
            ),
        )
        if limit == 0:
            events.clear()
        elif len(events) > limit:
            del events[:-limit]
        return accepted, accepted_params, accepted_f, accepted_info, alpha, trials

    def _direct_planar_physical_step_diagnostics(
        self,
        params: np.ndarray,
        trial: np.ndarray,
    ) -> dict:
        from bilevel.parameterization.geometry import (
            max_vertex_displacement,
        )

        displacements = []
        blocks = []
        for block in self._direct_planar_abs_blocks():
            if (
                not bool(block.get("deformable", False))
                or not bool(block.get("design_enabled", True))
            ):
                continue
            start = int(block["abs_start"])
            stop = int(block["abs_stop"])
            reference_length = float(
                block.get("reference_length", 1.0) or 1.0
            )
            value = float(
                max_vertex_displacement(
                    params[start:stop],
                    trial[start:stop],
                )
                * reference_length
            )
            displacements.append(value)
            blocks.append(
                {
                    "node_id": block.get("node_id"),
                    "link_name": str(block.get("link_name", "")),
                    "max_vertex_displacement": value,
                }
            )
        return {
            "max_vertex_displacement": (
                max(displacements) if displacements else 0.0
            ),
            "rms_block_displacement": (
                float(np.sqrt(np.mean(np.square(displacements))))
                if displacements
                else 0.0
            ),
            "blocks": blocks,
        }

    def _directional_contact_signature(self) -> dict:
        terminal = getattr(self.task, "_terminal_cache", None)
        if not isinstance(terminal, dict):
            return {}
        signature = {}
        for key, value in terminal.items():
            if "contact" not in str(key).lower():
                continue
            if isinstance(value, (bool, int, float, str)) or value is None:
                signature[str(key)] = value
        return signature

    def _directional_stage_acceptance(self):
        stage = str(getattr(self.task, "_optimization_stage", "full"))
        acceptance_fn = getattr(
            self.task,
            "optimization_stage_acceptance",
            None,
        )
        if stage == "full" or not callable(acceptance_fn):
            return None
        try:
            verdict = acceptance_fn(stage)
            if isinstance(verdict, dict):
                return dict(verdict)
            return {"accepted": bool(verdict), "stage": stage}
        except Exception as exc:
            return {"accepted": False, "stage": stage, "error": repr(exc)}

    def _morphology_expansion_config(self) -> dict:
        """Resolve the task-independent Stage-2 morphology policy."""

        def configured(name, default):
            value = getattr(self.args, name, None)
            return default if value is None else value

        config = {
            "initial_radius": float(
                configured(
                    "morphology_expansion_radius_initial",
                    0.01,
                )
            ),
            "growth": float(
                configured(
                    "morphology_expansion_radius_growth",
                    1.5,
                )
            ),
            "shrink": float(
                configured(
                    "morphology_expansion_radius_shrink",
                    0.5,
                )
            ),
            "max_rms": float(
                configured(
                    "morphology_expansion_max_rms",
                    0.125,
                )
            ),
            "loss_budget": float(
                configured(
                    "morphology_expansion_loss_budget",
                    0.0,
                )
            ),
            "final_loss_budget": float(
                configured(
                    "morphology_expansion_final_loss_budget",
                    0.0,
                )
            ),
            "repair_min_steps": int(
                configured(
                    "morphology_expansion_repair_min_steps",
                    3,
                )
            ),
            "repair_max_steps": int(
                configured(
                    "morphology_expansion_repair_max_steps",
                    10,
                )
            ),
            "repair_attempt_multiplier": int(
                configured(
                    "morphology_expansion_repair_attempt_multiplier",
                    3,
                )
            ),
            "coarse_iterations": int(
                configured(
                    "morphology_expansion_coarse_iterations",
                    3,
                )
            ),
            "minimum_radius": 1e-5,
            "minimum_expansion": 1e-6,
        }
        if not np.isfinite(config["initial_radius"]) or config["initial_radius"] <= 0.0:
            raise ValueError("morphology expansion initial radius must be positive")
        if not np.isfinite(config["growth"]) or config["growth"] <= 1.0:
            raise ValueError("morphology expansion growth must be greater than one")
        if not np.isfinite(config["shrink"]) or not 0.0 < config["shrink"] < 1.0:
            raise ValueError("morphology expansion shrink must lie in (0, 1)")
        if not np.isfinite(config["max_rms"]) or config["max_rms"] <= 0.0:
            raise ValueError("morphology expansion max RMS must be positive")
        if not np.isfinite(config["loss_budget"]) or config["loss_budget"] < 0.0:
            raise ValueError("morphology expansion loss budget must be nonnegative")
        if (
            not np.isfinite(config["final_loss_budget"])
            or config["final_loss_budget"] < 0.0
        ):
            raise ValueError(
                "morphology expansion final loss budget must be nonnegative"
            )
        if config["repair_min_steps"] < 0:
            raise ValueError("morphology repair minimum steps must be nonnegative")
        if config["repair_max_steps"] < config["repair_min_steps"]:
            raise ValueError(
                "morphology repair maximum steps must be at least the minimum"
            )
        if config["repair_attempt_multiplier"] < 1:
            raise ValueError(
                "morphology repair attempt multiplier must be at least one"
            )
        if config["coarse_iterations"] < 0:
            raise ValueError("morphology coarse iterations must be nonnegative")
        # Keep boundary search bounded: every trial may itself consume several
        # full-contact action-repair rollouts.  The existing coarse-iteration
        # option supplies the shared trial budget, with two trials as the
        # minimum needed to grow or shrink once.
        config["boundary_max_trials"] = max(
            2,
            config["coarse_iterations"],
        )
        return config

    def _morphology_target_shell_config(self) -> dict:
        """Resolve settings used by the default Stage-2 target-shell strategy."""

        config = dict(CoOptRunner._morphology_expansion_config(self))
        # Target-shell is the canonical Stage-2 policy.  Intermediate search
        # gates permit 2% relative slack so a non-convex continuation path is
        # not cut off.  Final selection is a separate hard gate and defaults
        # to zero degradation from the Stage-1 objective.  Explicit CLI/task
        # settings still take precedence.
        if getattr(
            self.args,
            "morphology_expansion_loss_budget",
            None,
        ) is None:
            config["loss_budget"] = 0.02
        raw_schedule = getattr(
            self.args,
            "morphology_expansion_target_schedule",
            None,
        )
        if raw_schedule is None:
            raw_schedule = "0.01,0.02,0.04,0.06,0.08,0.10"
        if isinstance(raw_schedule, str):
            tokens = [
                token.strip()
                for token in raw_schedule.split(",")
                if token.strip()
            ]
            try:
                schedule = tuple(float(token) for token in tokens)
            except ValueError as exc:
                raise ValueError(
                    "morphology target schedule must contain finite numbers"
                ) from exc
        else:
            schedule = tuple(float(value) for value in raw_schedule)
        if not schedule:
            raise ValueError("morphology target schedule must not be empty")
        if any(not np.isfinite(value) or value <= 0.0 for value in schedule):
            raise ValueError(
                "morphology target schedule values must be finite and positive"
            )
        if any(right <= left for left, right in zip(schedule, schedule[1:])):
            raise ValueError(
                "morphology target schedule must be strictly increasing"
            )
        if schedule[-1] > config["max_rms"] + 1e-12:
            raise ValueError(
                "morphology target schedule exceeds the configured maximum RMS"
            )
        tolerance_value = getattr(
            self.args,
            "morphology_expansion_target_tolerance",
            None,
        )
        search_trials_value = getattr(
            self.args,
            "morphology_expansion_target_search_trials",
            None,
        )
        tolerance = float(
            0.0005 if tolerance_value is None else tolerance_value
        )
        search_trials = int(
            24 if search_trials_value is None else search_trials_value
        )
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "morphology target tolerance must be finite and positive"
            )
        if tolerance >= schedule[0]:
            raise ValueError(
                "morphology target tolerance must be smaller than the first shell"
            )
        if search_trials < 4:
            raise ValueError(
                "morphology target search trials must be at least four"
            )
        config.update(
            {
                "target_schedule": schedule,
                "target_tolerance": tolerance,
                "target_search_trials": search_trials,
                "direction_policy": "deterministic_loss_radial_rotation",
            }
        )
        return config

    def _direct_planar_normalized_deformation(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
    ) -> dict:
        """Measure task- and topology-independent physical cage motion.

        Direct-planar cage coordinates are already divided by each block's
        reference length.  Measuring the seven movable vertices in that local
        frame therefore gives one dimensionless physical metric shared by all
        canonical tasks, independent of world-unit scale or tool topology.
        """

        from bilevel.parameterization.geometry import (
            vertices_from_q,
        )

        reference = np.asarray(reference, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64)
        squared = []
        block_reports = []
        for block in self._direct_planar_abs_blocks():
            if not bool(block.get("deformable", False)) or not bool(
                block.get("design_enabled", True)
            ):
                continue
            start = int(block["abs_start"])
            stop = int(block["abs_stop"])
            if stop - start != 18:
                continue
            initial_vertices = vertices_from_q(reference[start:stop])
            final_vertices = vertices_from_q(candidate[start:stop])
            vertex_delta = final_vertices[:, 1:] - initial_vertices[:, 1:]
            vertex_norms = np.linalg.norm(vertex_delta, axis=0)
            squared.extend(float(value * value) for value in vertex_norms)
            block_reports.append(
                {
                    "node_id": block.get("node_id"),
                    "link_name": str(block.get("link_name", "")),
                    "reference_length": float(
                        block.get("reference_length", 1.0) or 1.0
                    ),
                    "normalized_rms": float(
                        np.sqrt(np.mean(np.square(vertex_norms)))
                    ),
                    "normalized_max": float(np.max(vertex_norms)),
                }
            )
        return {
            "normalized_rms": (
                float(np.sqrt(np.mean(squared))) if squared else 0.0
            ),
            "normalized_max": (
                float(np.sqrt(max(squared))) if squared else 0.0
            ),
            "blocks": block_reports,
        }

    def _morphology_direction_normalized_rms(
        self,
        params: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        report = self._direct_planar_normalized_deformation(
            params,
            np.asarray(params, dtype=np.float64)
            + np.asarray(direction, dtype=np.float64),
        )
        return float(report["normalized_rms"])

    def _normalize_morphology_direction(
        self,
        params: np.ndarray,
        direction: np.ndarray,
        radius: float,
    ) -> Optional[np.ndarray]:
        projected = self._direct_planar_project_vector(params, direction)
        action_dim = self.ndof_u * self.num_ctrl_steps
        projected[:action_dim] = 0.0
        current_rms = self._morphology_direction_normalized_rms(
            params,
            projected,
        )
        if not np.isfinite(current_rms) or current_rms <= 1e-12:
            return None
        return projected * (float(radius) / current_rms)

    def _morphology_loss_feasible_direction(
        self,
        *,
        reference: np.ndarray,
        incumbent: np.ndarray,
        loss_gradient: np.ndarray,
        radius: float,
    ):
        """Maximize physical deformation in the local non-increasing-loss cone.

        At the Stage-1 origin the deformation gradient is zero, so the first
        direction is the ordinary differentiable task-loss descent direction.
        Thereafter the radial deformation gradient is projected onto the
        half-space whose first-order task-loss change is non-positive.  This
        uses no node, link, topology, axis, or task semantics.
        """

        incumbent = np.asarray(incumbent, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        action_dim = self.ndof_u * self.num_ctrl_steps

        projected_loss = self._direct_planar_project_vector(
            incumbent,
            np.asarray(loss_gradient, dtype=np.float64),
        )
        projected_loss[:action_dim] = 0.0
        loss_norm_sq = float(np.dot(projected_loss, projected_loss))
        outward = self._direct_planar_project_vector(
            incumbent,
            incumbent - reference,
        )
        outward[:action_dim] = 0.0
        outward_norm = float(np.linalg.norm(outward))
        if not np.isfinite(loss_norm_sq):
            return None, {
                "strategy": "nonfinite_loss_gradient",
                "predicted_loss_change": None,
                "radial_alignment": None,
            }
        if loss_norm_sq <= 1e-18:
            if outward_norm <= 1e-12:
                return None, {
                    "strategy": "zero_loss_and_deformation_gradient",
                    "predicted_loss_change": None,
                    "radial_alignment": None,
                }
            raw = outward.copy()
            strategy = "deformation_gradient_zero_loss_gradient"
        elif outward_norm <= 1e-12:
            raw = -projected_loss
            strategy = "task_descent_seed"
        else:
            raw = outward.copy()
            predicted = float(np.dot(projected_loss, raw))
            if predicted > 0.0:
                raw -= (predicted / loss_norm_sq) * projected_loss
                strategy = "deformation_gradient_loss_projected"
            else:
                strategy = "deformation_gradient_feasible"

        direction = self._normalize_morphology_direction(
            incumbent,
            raw,
            radius,
        )
        if direction is None:
            return None, {
                "strategy": strategy,
                "predicted_loss_change": None,
                "radial_alignment": None,
            }
        predicted_loss_change = float(np.dot(projected_loss, direction))
        radial_alignment = float(np.dot(outward, direction))
        if outward_norm > 1e-12 and radial_alignment <= 1e-14:
            return None, {
                "strategy": "no_locally_feasible_expansion",
                "predicted_loss_change": predicted_loss_change,
                "radial_alignment": radial_alignment,
            }
        return direction, {
            "strategy": strategy,
            "predicted_loss_change": predicted_loss_change,
            "radial_alignment": radial_alignment,
        }

    @staticmethod
    def _morphology_variant_config(config: dict, variant: str) -> dict:
        """Apply one task- and structure-independent experiment policy."""

        variant = str(variant).strip().lower()
        if variant not in {
            "baseline",
            "action_slack",
            "homotopy",
            "multidirection",
        }:
            raise ValueError(f"unknown success-constrained variant: {variant}")
        resolved = dict(config)
        resolved["variant"] = variant
        if variant == "action_slack":
            # Every candidate receives the complete action-repair budget before
            # the common hard gate decides whether its larger shape is safe.
            resolved["repair_min_steps"] = resolved["repair_max_steps"]
        elif variant == "homotopy":
            # The same target-independent radius is reached through three
            # continuation-sized outer increments.  This reduces abrupt
            # contact-set changes without inspecting tool topology or axes.
            resolved["homotopy_subdivisions"] = 3
            resolved["initial_radius"] /= resolved[
                "homotopy_subdivisions"
            ]
        elif variant == "multidirection":
            resolved["direction_policy"] = (
                "deterministic_loss_radial_rotation"
            )
        return resolved

    def _morphology_direction_candidates(
        self,
        *,
        reference: np.ndarray,
        incumbent: np.ndarray,
        loss_gradient: np.ndarray,
        radius: float,
        variant: str,
    ):
        """Return generic differentiable candidate directions.

        The multidirection variant uses only the task-loss gradient and the
        gradient of accumulated physical deformation.  It contains no names,
        graph roles, topology tests, axes, or task-specific target directions.
        """

        primary, primary_report = self._morphology_loss_feasible_direction(
            reference=reference,
            incumbent=incumbent,
            loss_gradient=loss_gradient,
            radius=radius,
        )
        if primary is None:
            return [], primary_report
        candidates = [(primary, dict(primary_report))]
        if variant != "multidirection":
            return candidates, primary_report

        incumbent = np.asarray(incumbent, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        action_dim = self.ndof_u * self.num_ctrl_steps
        projected_loss = self._direct_planar_project_vector(
            incumbent,
            np.asarray(loss_gradient, dtype=np.float64),
        )
        projected_loss[:action_dim] = 0.0
        outward = self._direct_planar_project_vector(
            incumbent,
            incumbent - reference,
        )
        outward[:action_dim] = 0.0
        outward_norm = float(np.linalg.norm(outward))
        raw_candidates = [
            ("task_loss_descent", -projected_loss),
        ]
        if outward_norm > 1e-12:
            raw_candidates.append(("pure_deformation_gradient", outward))
            primary_norm = float(np.linalg.norm(primary))
            loss_norm = float(np.linalg.norm(projected_loss))
            if primary_norm > 1e-12 and loss_norm > 1e-12:
                raw_candidates.append(
                    (
                        "balanced_loss_deformation",
                        primary / primary_norm
                        - projected_loss / loss_norm,
                    )
                )
        for label, raw in raw_candidates:
            direction = self._normalize_morphology_direction(
                incumbent,
                raw,
                radius,
            )
            if direction is None:
                continue
            radial_alignment = float(np.dot(outward, direction))
            if outward_norm > 1e-12 and radial_alignment <= 1e-14:
                continue
            direction_norm = float(np.linalg.norm(direction))
            duplicate = any(
                abs(float(np.dot(existing, direction)))
                >= 0.9999
                * max(1e-30, float(np.linalg.norm(existing)) * direction_norm)
                for existing, _ in candidates
            )
            if duplicate:
                continue
            candidates.append(
                (
                    direction,
                    {
                        "strategy": label,
                        "predicted_loss_change": float(
                            np.dot(projected_loss, direction)
                        ),
                        "radial_alignment": radial_alignment,
                    },
                )
            )
        return candidates, primary_report

    def _morphology_axis_fallback_directions(
        self,
        *,
        incumbent: np.ndarray,
        loss_gradient: np.ndarray,
        radius: float,
        existing_directions=(),
    ):
        """Return generic projected per-block axis-scale directions.

        The differentiable loss/radial directions remain the primary search
        policy.  These deterministic directions are a geometry-only fallback
        used only after a complete primary Stage-2 pass accepts no morphology.
        They use only the shared 18-coordinate connected-hexahedron
        representation; no task, asset, topology, or world-axis semantics
        enter the ordering.
        """

        incumbent = np.asarray(incumbent, dtype=np.float64)
        action_dim = self.ndof_u * self.num_ctrl_steps
        projected_loss = self._direct_planar_project_vector(
            incumbent,
            np.asarray(loss_gradient, dtype=np.float64),
        )
        projected_loss[:action_dim] = 0.0
        axis_indices = (
            ("x", (0, 1, 3, 6, 9, 12, 15)),
            ("y", (2, 4, 7, 10, 13, 16)),
            ("z", (5, 8, 11, 14, 17)),
        )
        blocks = sorted(
            (
                block
                for block in self._direct_planar_abs_blocks()
                if bool(block.get("deformable", False))
                and bool(block.get("design_enabled", True))
                and int(block["abs_stop"]) - int(block["abs_start"]) == 18
            ),
            key=lambda block: (
                int(
                    block.get("node_id", -1)
                    if block.get("node_id") is not None
                    else -1
                ),
                str(block.get("link_name", "")),
            ),
        )
        known = [
            np.asarray(direction, dtype=np.float64)
            for direction in existing_directions
        ]
        candidates = []

        def duplicates_known(direction: np.ndarray) -> bool:
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-12:
                return True
            for other in known:
                other_norm = float(np.linalg.norm(other))
                if other_norm <= 1e-12:
                    continue
                cosine = float(np.dot(other, direction)) / (
                    other_norm * direction_norm
                )
                # Opposite signs are intentionally distinct: expansion and
                # contraction can have different collision feasibility.
                if cosine >= 0.9999:
                    return True
            return False

        for block in blocks:
            start = int(block["abs_start"])
            stop = int(block["abs_stop"])
            for axis_order, (axis_name, indices) in enumerate(axis_indices):
                raw = np.zeros_like(incumbent, dtype=np.float64)
                local = raw[start:stop]
                local_values = incumbent[start:stop]
                local[list(indices)] = local_values[list(indices)]
                projected_axis = self._direct_planar_project_vector(
                    incumbent,
                    raw,
                )
                projected_axis[:action_dim] = 0.0
                for sign in (1, -1):
                    direction = self._normalize_morphology_direction(
                        incumbent,
                        float(sign) * projected_axis,
                        radius,
                    )
                    if direction is None or duplicates_known(direction):
                        continue
                    known.append(direction)
                    predicted_loss_change = float(
                        np.dot(projected_loss, direction)
                    )
                    candidates.append(
                        (
                            direction,
                            {
                                "strategy": "projected_axis_scale_fallback",
                                "predicted_loss_change": predicted_loss_change,
                                "radial_alignment": None,
                                "node_id": (
                                    int(block["node_id"])
                                    if block.get("node_id") is not None
                                    else None
                                ),
                                "link_name": str(block.get("link_name", "")),
                                "axis": axis_name,
                                "axis_order": int(axis_order),
                                "sign": int(sign),
                            },
                        )
                    )

        candidates.sort(
            key=lambda item: (
                item[1]["predicted_loss_change"]
                if np.isfinite(item[1]["predicted_loss_change"])
                else float("inf"),
                item[1]["node_id"]
                if item[1]["node_id"] is not None
                else -1,
                item[1]["axis_order"],
                -item[1]["sign"],
            )
        )
        return candidates

    def _morphology_proposal_for_target_shell(
        self,
        *,
        reference: np.ndarray,
        incumbent: np.ndarray,
        unit_direction: np.ndarray,
        target_rms: float,
        config: dict,
    ) -> dict:
        """Locate a retracted proposal on or just outside one RMS shell.

        This is a geometry-only scalar bracket.  It neither evaluates task
        success nor weakens the later action-repair/full-replay hard gate.
        The search is agnostic to task, topology, part names, and direction
        semantics; inward and outward vertex motion contribute equally.
        """

        reference = np.asarray(reference, dtype=np.float64)
        incumbent = np.asarray(incumbent, dtype=np.float64)
        unit_direction = np.asarray(unit_direction, dtype=np.float64)
        current = self._direct_planar_normalized_deformation(
            reference,
            incumbent,
        )
        current_rms = float(current["normalized_rms"])
        target_rms = float(target_rms)
        max_trials = int(config["target_search_trials"])
        max_step = max(
            4.0 * float(config["max_rms"]),
            2.0 * max(float(config["initial_radius"]), target_rms),
        )
        probe_radius = max(
            float(config["minimum_radius"]),
            target_rms - current_rms,
        )
        low = {
            "radius": 0.0,
            "params": incumbent.copy(),
            "deformation": current,
        }
        high = None
        invalid_upper = None
        attempts = []

        def probe(radius):
            proposal, retraction_ok = self._direct_planar_retract_params(
                incumbent,
                unit_direction * float(radius),
            )
            event = {
                "radius": float(radius),
                "retraction_ok": bool(retraction_ok),
            }
            if not retraction_ok:
                return None, event
            deformation = self._direct_planar_normalized_deformation(
                reference,
                proposal,
            )
            event["normalized_rms"] = float(deformation["normalized_rms"])
            return {
                "radius": float(radius),
                "params": np.asarray(proposal, dtype=np.float64).copy(),
                "deformation": deformation,
            }, event

        # First grow a scalar bracket.  The normalized unit direction makes
        # the first guess close to the requested cumulative RMS difference,
        # while doubling handles projection/retraction curvature generically.
        while len(attempts) < max_trials and probe_radius <= max_step + 1e-15:
            candidate, event = probe(probe_radius)
            attempts.append(event)
            if candidate is None:
                invalid_upper = float(probe_radius)
                break
            candidate_rms = float(candidate["deformation"]["normalized_rms"])
            if candidate_rms >= target_rms:
                high = candidate
                break
            if candidate_rms <= float(low["deformation"]["normalized_rms"]) + 1e-12:
                invalid_upper = float(probe_radius)
                break
            low = candidate
            probe_radius *= 2.0

        # Refine either a valid above-target bracket or the interval ending at
        # the first invalid retraction.  `high` always remains a valid proposal
        # on/outside the requested shell.
        upper_radius = (
            float(high["radius"])
            if high is not None
            else invalid_upper
        )
        while (
            len(attempts) < max_trials
            and upper_radius is not None
            and upper_radius - float(low["radius"])
            > float(config["minimum_radius"])
        ):
            mid = 0.5 * (float(low["radius"]) + float(upper_radius))
            candidate, event = probe(mid)
            attempts.append(event)
            if candidate is None:
                upper_radius = mid
                continue
            candidate_rms = float(candidate["deformation"]["normalized_rms"])
            if candidate_rms >= target_rms:
                high = candidate
                upper_radius = mid
            else:
                low = candidate

        if high is None:
            return {
                "reached": False,
                "reason": "target_shell_unreachable_along_direction",
                "target_rms": target_rms,
                "best_rms": float(low["deformation"]["normalized_rms"]),
                "search_trials": attempts,
            }
        return {
            "reached": True,
            "target_rms": target_rms,
            "radius": float(high["radius"]),
            "params": high["params"],
            "deformation": high["deformation"],
            "search_trials": attempts,
        }

    @staticmethod
    def _morphology_require_target_shell(gate: dict, target_rms: float) -> dict:
        """Add an explicit cumulative-RMS lower bound to an existing gate."""

        resolved = dict(gate)
        achieved_rms = float(resolved.get("deformation_rms", float("nan")))
        target_ok = bool(
            resolved.get("accepted", False)
            and np.isfinite(achieved_rms)
            and achieved_rms + 1e-12 >= float(target_rms)
        )
        resolved["target_shell_ok"] = target_ok
        resolved["target_rms"] = float(target_rms)
        if resolved.get("accepted", False) and not target_ok:
            resolved["accepted"] = False
            reasons = list(resolved.get("reasons", []))
            if "target_shell_not_reached" not in reasons:
                reasons.append("target_shell_not_reached")
            resolved["reasons"] = reasons
        return resolved

    @staticmethod
    def _morphology_expansion_filter_decision(
        *,
        task_success: bool,
        geometry_ok: bool,
        objective: float,
        loss_limit: float,
        deformation_rms: float,
        incumbent_rms: float,
        minimum_expansion: float,
        max_rms: float,
    ) -> dict:
        finite_objective = bool(np.isfinite(objective))
        numerical_loss_tolerance = 1e-9 * max(1.0, abs(float(loss_limit)))
        loss_ok = bool(
            finite_objective
            and objective <= loss_limit + numerical_loss_tolerance
        )
        expansion_ok = bool(
            np.isfinite(deformation_rms)
            and deformation_rms
            >= incumbent_rms + minimum_expansion
            and deformation_rms <= max_rms + 1e-12
        )
        accepted = bool(
            task_success and geometry_ok and loss_ok and expansion_ok
        )
        reasons = []
        if not task_success:
            reasons.append("task_success_failed")
        if not geometry_ok:
            reasons.append("geometry_failed")
        if not loss_ok:
            reasons.append("loss_budget_exceeded")
        if not expansion_ok:
            reasons.append("morphology_did_not_expand")
        return {
            "accepted": accepted,
            "reasons": reasons,
            "task_success": bool(task_success),
            "geometry_ok": bool(geometry_ok),
            "loss_ok": loss_ok,
            "expansion_ok": expansion_ok,
            "objective": float(objective),
            "loss_limit": float(loss_limit),
            "deformation_rms": float(deformation_rms),
            "incumbent_rms": float(incumbent_rms),
        }

    def _morphology_full_replay_gate(
        self,
        params: np.ndarray,
        *,
        reference: np.ndarray,
        baseline_objective: float,
        incumbent_rms: float,
        config: dict,
        require_expansion: bool,
    ) -> dict:
        """Run the authoritative full-contact task and geometry checks."""

        self.sim.set_contact_scale(1.0)
        collision = self._design_collision_report_for_params(params)
        # This optimizer advertises geometry as a hard gate.  An explicitly
        # disabled or unavailable collision audit is therefore not equivalent
        # to a passing audit.
        collision_ok = bool(collision is not None and collision.ok)
        report = {
            "collision": (
                None if collision is None else collision.to_dict()
            ),
            "geometry_ok": collision_ok,
        }
        try:
            objective, _ = self.forward(params, backward_flag=False)
            objective = float(objective)
        except Exception as exc:
            report.update(
                {
                    "accepted": False,
                    "task_success": False,
                    "objective": float("inf"),
                    "error": repr(exc),
                    "reasons": ["full_contact_forward_failed"],
                }
            )
            return report

        diagnostics_fn = getattr(self.task, "rollout_diagnostics", None)
        if not callable(diagnostics_fn):
            report.update(
                {
                    "accepted": False,
                    "task_success": False,
                    "objective": objective,
                    "reasons": ["missing_rollout_diagnostics"],
                }
            )
            return report
        try:
            diagnostics = dict(diagnostics_fn(self, params) or {})
        except Exception as exc:
            report.update(
                {
                    "accepted": False,
                    "task_success": False,
                    "objective": objective,
                    "error": repr(exc),
                    "reasons": ["full_replay_diagnostics_failed"],
                }
            )
            return report
        task_success = bool(diagnostics.get("task_success", False))
        deformation = self._direct_planar_normalized_deformation(
            reference,
            params,
        )
        loss_limit = float(
            baseline_objective
            + config["loss_budget"]
            * max(1.0, abs(float(baseline_objective)))
        )
        decision = self._morphology_expansion_filter_decision(
            task_success=task_success,
            geometry_ok=collision_ok,
            objective=objective,
            loss_limit=loss_limit,
            deformation_rms=float(deformation["normalized_rms"]),
            incumbent_rms=(
                float(incumbent_rms)
                if require_expansion
                else float(deformation["normalized_rms"])
            ),
            minimum_expansion=(
                float(config["minimum_expansion"])
                if require_expansion
                else 0.0
            ),
            max_rms=float(config["max_rms"]),
        )
        if not require_expansion:
            decision["expansion_ok"] = True
            decision["accepted"] = bool(
                task_success and collision_ok and decision["loss_ok"]
            )
            decision["reasons"] = [
                reason
                for reason in decision["reasons"]
                if reason != "morphology_did_not_expand"
            ]
        diagnostic_summary_keys = (
            "task_success",
            "success",
            "task_feasible",
            "raw_terminal_task_success",
            "symmetric_geometry_ok",
            "severe_geometry_overlap",
            "penetration_warning",
            "soft_penetration_ok",
        )
        diagnostic_summary = {
            key: diagnostics[key]
            for key in diagnostic_summary_keys
            if key in diagnostics
        }
        report.update(
            {
                **decision,
                "deformation": deformation,
                "task_diagnostics": diagnostic_summary,
            }
        )
        return report

    def _repair_action_for_morphology(
        self,
        proposal: np.ndarray,
        *,
        reference: np.ndarray,
        baseline_objective: float,
        incumbent_rms: float,
        config: dict,
    ):
        """Hold shape fixed while action trust-region steps recover success."""

        params = np.asarray(proposal, dtype=np.float64).copy()
        accepted_steps = 0
        attempts = 0
        maxls = max(1, int(getattr(self.args, "maxls", 12)))
        max_attempts = max(
            1,
            int(config.get("repair_attempt_multiplier", 3))
            * max(1, config["repair_max_steps"]),
        )
        c1 = float(
            getattr(self.args, "direct_planar_armijo_c1", 1e-4) or 1e-4
        )
        self._reset_action_trust_region()
        last_gate = None
        events = []
        while (
            attempts < max_attempts
            and accepted_steps < config["repair_max_steps"]
        ):
            attempts += 1
            try:
                objective, gradient = self._loss_and_grad_for_block(
                    params,
                    "action",
                )
            except Exception as exc:
                events.append(
                    {
                        "attempt": attempts,
                        "accepted": False,
                        "error": repr(exc),
                    }
                )
                break
            projected, direction, descent = self._direct_planar_block_direction(
                params,
                gradient,
                "action",
            )
            if (
                not np.isfinite(descent)
                or descent >= 0.0
                or float(np.linalg.norm(direction)) < 1e-12
            ):
                events.append(
                    {
                        "attempt": attempts,
                        "accepted": False,
                        "reason": "no_action_descent_direction",
                    }
                )
                break
            result = self._action_trust_search_with_fallback(
                params,
                float(objective),
                direction,
                projected,
                c1=c1,
                maxls=maxls,
            )
            accepted, repaired, repaired_f, _, alpha, trials = result
            events.append(
                {
                    "attempt": attempts,
                    "accepted": bool(accepted),
                    "objective_before": float(objective),
                    "objective_after": (
                        None if repaired_f is None else float(repaired_f)
                    ),
                    "alpha": float(alpha),
                    "trials": int(trials),
                }
            )
            if not accepted or repaired is None:
                continue
            params = np.asarray(repaired, dtype=np.float64)
            accepted_steps += 1
            if accepted_steps < config["repair_min_steps"]:
                continue
            last_gate = self._morphology_full_replay_gate(
                params,
                reference=reference,
                baseline_objective=baseline_objective,
                incumbent_rms=incumbent_rms,
                config=config,
                require_expansion=True,
            )
            if last_gate.get("accepted", False):
                break
        if last_gate is None or not last_gate.get("accepted", False):
            last_gate = self._morphology_full_replay_gate(
                params,
                reference=reference,
                baseline_objective=baseline_objective,
                incumbent_rms=incumbent_rms,
                config=config,
                require_expansion=True,
            )
        return params, last_gate, {
            "attempts": int(attempts),
            "accepted_steps": int(accepted_steps),
            "events": events,
        }


    def _optimize_success_constrained_target_shell(
        self,
        params0: np.ndarray,
        *,
        strategy_name: str = "success_constrained_target_shell",
    ) -> np.ndarray:
        """Run primary and optional axis-fallback search under one budget.

        ``optimize_maxiter`` is a global Stage-2 outer-iteration budget.  The
        generic axis fallback is started only when the primary pass commits no
        final-eligible morphology, and it receives only the iterations left by
        that pass.  Search-only candidates may use the configured temporary
        loss budget, but do not suppress fallback when none meet the final
        loss budget.
        """

        global_iteration_budget = max(
            0,
            int(getattr(self.args, "optimize_maxiter", 100)),
        )

        primary_result = (
            CoOptRunner._optimize_success_constrained_target_shell_pass(
                self,
                params0,
                strategy_name=strategy_name,
                direction_policy="primary",
                iteration_budget=global_iteration_budget,
            )
        )
        primary_diagnostics = self._action_optimizer_diagnostics
        primary_iterations = int(
            primary_diagnostics.get("iterations_attempted", 0)
            if isinstance(primary_diagnostics, dict)
            else 0
        )
        remaining_iteration_budget = max(
            0,
            global_iteration_budget - primary_iterations,
        )
        should_run_fallback = bool(
            remaining_iteration_budget > 0
            and isinstance(primary_diagnostics, dict)
            and primary_diagnostics.get("status") == "done"
            and float(
                primary_diagnostics.get("normalized_deformation_rms", 0.0)
            )
            <= float(
                primary_diagnostics.get("config", {}).get(
                    "minimum_expansion",
                    1e-6,
                )
            )
        )
        if not should_run_fallback:
            if isinstance(primary_diagnostics, dict):
                primary_diagnostics.update(
                    {
                        "iteration_budget_scope": (
                            "global_across_primary_and_fallback"
                        ),
                        "global_iteration_budget": int(
                            global_iteration_budget
                        ),
                        "primary_iteration_budget": int(
                            global_iteration_budget
                        ),
                        "primary_iterations_attempted": int(
                            primary_iterations
                        ),
                        "fallback_iteration_budget": 0,
                        "fallback_iterations_attempted": 0,
                        "total_iterations_attempted": int(
                            primary_iterations
                        ),
                    }
                )
            return primary_result

        print(
            "[optimizer] target-shell pass committed no "
            "final-eligible morphology; starting generic axis fallback with "
            f"remaining global budget={remaining_iteration_budget}",
            flush=True,
        )
        fallback_result = (
            CoOptRunner._optimize_success_constrained_target_shell_pass(
                self,
                params0,
                strategy_name=strategy_name,
                direction_policy="axis_fallback",
                iteration_budget=remaining_iteration_budget,
            )
        )
        fallback_diagnostics = self._action_optimizer_diagnostics
        if isinstance(fallback_diagnostics, dict):
            fallback_iterations = int(
                fallback_diagnostics.get("iterations_attempted", 0)
            )
            fallback_diagnostics["fallback_triggered"] = True
            fallback_diagnostics["fallback_trigger_reason"] = (
                "primary_pass_zero_final_eligible_morphology"
            )
            fallback_diagnostics["primary_pass"] = primary_diagnostics
            fallback_diagnostics.update(
                {
                    "iteration_budget_scope": (
                        "global_across_primary_and_fallback"
                    ),
                    "global_iteration_budget": int(
                        global_iteration_budget
                    ),
                    "primary_iteration_budget": int(
                        global_iteration_budget
                    ),
                    "primary_iterations_attempted": int(
                        primary_iterations
                    ),
                    "fallback_iteration_budget": int(
                        remaining_iteration_budget
                    ),
                    "fallback_iterations_attempted": int(
                        fallback_iterations
                    ),
                    "total_iterations_attempted": int(
                        primary_iterations + fallback_iterations
                    ),
                }
            )
        return fallback_result

    def _optimize_success_constrained_target_shell_pass(
        self,
        params0: np.ndarray,
        *,
        strategy_name: str = "success_constrained_target_shell",
        direction_policy: str,
        iteration_budget: Optional[int] = None,
    ) -> np.ndarray:
        """Pursue explicit cumulative RMS shells without weakening safety.

        This optimizer is intentionally isolated from the established
        success-constrained variants.  Each requested shell is a lower bound
        on true vertex-motion RMS, not a volume-growth direction: contraction,
        thickening, bending, and mixed deformation are all permitted.  A shape
        is committed only after the existing action repair and authoritative
        full-contact hard gate pass.  Failed shells are bisected against the
        last committed result, which remains the rollback point.
        """

        config = self._morphology_target_shell_config()
        reference = np.asarray(params0, dtype=np.float64).copy()
        self.sim.set_contact_scale(1.0)
        stage_setter = getattr(self.task, "set_optimization_stage", None)
        if callable(stage_setter):
            stage_setter("full")
        self.callback(reference, render=False, log=True)
        baseline_objective = float(self.f_log[-1][1])
        baseline_gate = self._morphology_full_replay_gate(
            reference,
            reference=reference,
            baseline_objective=baseline_objective,
            incumbent_rms=0.0,
            config=config,
            require_expansion=False,
        )
        self._design_collision_initial_report = baseline_gate.get("collision")
        if not baseline_gate.get("accepted", False):
            self._action_optimizer_diagnostics = {
                "optimizer": strategy_name,
                "status": "baseline_rejected",
                "termination_reason": "stage1_full_replay_gate_failed",
                "config": config,
                "baseline_gate": baseline_gate,
                "rolled_back_to_stage1": True,
                "accepted_proposals": 0,
                "completed_target_shells": 0,
                "proposal_events": [],
            }
            print(
                "[optimizer] target-shell morphology refused to start: "
                "Stage-1 baseline failed the authoritative full replay gate",
                flush=True,
            )
            return reference

        incumbent = reference.copy()
        incumbent_objective = float(baseline_gate["objective"])
        incumbent_rms = 0.0
        final_loss_budget = float(config.get("final_loss_budget", 0.0))
        final_loss_limit = float(
            baseline_objective
            + final_loss_budget
            * max(1.0, abs(float(baseline_objective)))
        )
        final_loss_tolerance = 1e-9 * max(1.0, abs(final_loss_limit))
        best_final_params = reference.copy()
        best_final_objective = float(baseline_gate["objective"])
        best_final_rms = 0.0
        best_final_relative_improvement = 0.0
        best_final_selection_score = 0.0
        best_final_selection_key = (
            best_final_selection_score,
            -best_final_objective,
            best_final_rms,
        )
        best_final_iteration = None
        maximum_final_eligible_rms = 0.0
        final_eligible_proposals = 0
        schedule = tuple(float(value) for value in config["target_schedule"])
        target_tolerance = float(config["target_tolerance"])
        configured_maxiter = max(
            0,
            int(getattr(self.args, "optimize_maxiter", 100)),
        )
        maxiter = (
            configured_maxiter
            if iteration_budget is None
            else max(0, min(int(iteration_budget), configured_maxiter))
        )
        accepted_proposals = 0
        completed_shells = 0
        failed_upper = None
        highest_attempted_target = 0.0
        events = []
        termination_reason = "budget_exhausted"

        for iteration in range(maxiter):
            while (
                completed_shells < len(schedule)
                and incumbent_rms
                + 1e-12
                >= schedule[completed_shells]
            ):
                completed_shells += 1
                failed_upper = None
            if completed_shells >= len(schedule):
                termination_reason = "target_deformation_reached"
                break

            requested_target = float(schedule[completed_shells])
            if failed_upper is not None:
                if failed_upper - incumbent_rms <= target_tolerance:
                    termination_reason = "target_shell_bracket_exhausted"
                    break
                attempted_target = 0.5 * (
                    float(failed_upper) + float(incumbent_rms)
                )
            else:
                attempted_target = requested_target
            if attempted_target - incumbent_rms <= config["minimum_expansion"]:
                termination_reason = "target_shell_bracket_exhausted"
                break
            highest_attempted_target = max(
                highest_attempted_target,
                attempted_target,
            )

            event = {
                "iteration": int(iteration),
                "requested_target_rms": requested_target,
                "attempted_target_rms": float(attempted_target),
                "incumbent_rms_before": float(incumbent_rms),
                "accepted": False,
                "direction_trials": [],
            }
            try:
                _, design_gradient = self._loss_and_grad_for_block(
                    incumbent,
                    "design",
                )
            except Exception as exc:
                event.update(
                    {
                        "reason": "design_gradient_failed",
                        "error": repr(exc),
                    }
                )
                events.append(event)
                failed_upper = (
                    attempted_target
                    if failed_upper is None
                    else min(float(failed_upper), attempted_target)
                )
                continue

            if direction_policy == "primary":
                direction_candidates, direction_failure = (
                    self._morphology_direction_candidates(
                        reference=reference,
                        incumbent=incumbent,
                        loss_gradient=design_gradient,
                        radius=float(config["initial_radius"]),
                        variant="multidirection",
                    )
                )
            elif direction_policy == "axis_fallback":
                direction_candidates = (
                    self._morphology_axis_fallback_directions(
                        incumbent=incumbent,
                        loss_gradient=design_gradient,
                        radius=float(config["initial_radius"]),
                    )
                )
                direction_failure = {
                    "strategy": "projected_axis_scale_fallback_unavailable"
                }
            else:
                raise ValueError(
                    "Unknown target-shell direction policy: "
                    f"{direction_policy!r}"
                )
            if not direction_candidates:
                event["reason"] = direction_failure["strategy"]
                events.append(event)
                failed_upper = (
                    attempted_target
                    if failed_upper is None
                    else min(float(failed_upper), attempted_target)
                )
                continue

            accepted_candidate = None
            trial_count = min(
                len(direction_candidates),
                int(config["boundary_max_trials"]),
            )
            start_index = iteration % len(direction_candidates)
            for offset in range(trial_count):
                direction_index = (
                    start_index + offset
                ) % len(direction_candidates)
                direction, direction_report = direction_candidates[
                    direction_index
                ]
                direction_trial = {
                    "direction_index": int(direction_index),
                    "direction": direction_report,
                    "accepted": False,
                }
                direction_radius = self._morphology_direction_normalized_rms(
                    incumbent,
                    direction,
                )
                if not np.isfinite(direction_radius) or direction_radius <= 1e-12:
                    direction_trial["reason"] = "zero_normalized_direction"
                    event["direction_trials"].append(direction_trial)
                    continue
                shell_proposal = self._morphology_proposal_for_target_shell(
                    reference=reference,
                    incumbent=incumbent,
                    unit_direction=(direction / direction_radius),
                    target_rms=attempted_target,
                    config=config,
                )
                direction_trial["shell_search"] = {
                    key: value
                    for key, value in shell_proposal.items()
                    if key != "params"
                }
                if not shell_proposal.get("reached", False):
                    direction_trial["reason"] = shell_proposal["reason"]
                    event["direction_trials"].append(direction_trial)
                    continue

                repaired, gate, repair = self._repair_action_for_morphology(
                    shell_proposal["params"],
                    reference=reference,
                    baseline_objective=baseline_objective,
                    incumbent_rms=incumbent_rms,
                    config=config,
                )
                gate = self._morphology_require_target_shell(
                    gate,
                    attempted_target,
                )
                shell_ok = bool(gate["target_shell_ok"])
                direction_trial.update(
                    {
                        "repair": repair,
                        "gate": gate,
                        "accepted": shell_ok,
                    }
                )
                event["direction_trials"].append(direction_trial)
                if shell_ok:
                    accepted_candidate = {
                        "params": np.asarray(repaired, dtype=np.float64).copy(),
                        "gate": gate,
                        "repair": repair,
                        "direction": direction_report,
                        "direction_index": int(direction_index),
                        "proposal_radius": float(shell_proposal["radius"]),
                    }
                    break

            if accepted_candidate is None:
                failed_upper = (
                    attempted_target
                    if failed_upper is None
                    else min(float(failed_upper), attempted_target)
                )
                event["reason"] = "target_shell_failed_hard_gate"
                events.append(event)
                rejection_reasons = sorted(
                    {
                        str(reason)
                        for trial in event["direction_trials"]
                        for reason in trial.get("gate", {}).get(
                            "reasons", [trial.get("reason", "unknown")]
                        )
                        if reason
                    }
                )
                print(
                    "[optimizer] target shell rejected",
                    json.dumps(
                        {
                            "iteration": int(iteration),
                            "requested_target_rms": requested_target,
                            "attempted_target_rms": attempted_target,
                            "incumbent_rms": incumbent_rms,
                            "gate_reasons": rejection_reasons,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            incumbent = accepted_candidate["params"]
            gate = accepted_candidate["gate"]
            repair = accepted_candidate["repair"]
            incumbent_objective = float(gate["objective"])
            incumbent_rms = float(gate["deformation_rms"])
            accepted_proposals += 1
            final_eligible = bool(
                np.isfinite(incumbent_objective)
                and incumbent_objective
                <= final_loss_limit + final_loss_tolerance
            )
            if final_eligible:
                final_eligible_proposals += 1
                maximum_final_eligible_rms = max(
                    maximum_final_eligible_rms,
                    incumbent_rms,
                )
                relative_improvement = float(
                    (baseline_objective - incumbent_objective)
                    / max(1.0, abs(baseline_objective))
                )
                selection_score = float(
                    incumbent_rms * max(0.0, relative_improvement)
                )
                # Select a balanced deformation/performance point instead of
                # sacrificing an arbitrarily large objective improvement for
                # the last increment of RMS.  Both factors are dimensionless.
                # Exact score ties prefer lower objective, then larger RMS.
                selection_key = (
                    selection_score,
                    -incumbent_objective,
                    incumbent_rms,
                )
                if selection_key > best_final_selection_key:
                    best_final_params = incumbent.copy()
                    best_final_objective = incumbent_objective
                    best_final_rms = incumbent_rms
                    best_final_relative_improvement = relative_improvement
                    best_final_selection_score = selection_score
                    best_final_selection_key = selection_key
                    best_final_iteration = int(iteration)
            # A successful intermediate shell changes both the shape and its
            # repaired action, so a failure bound measured from the previous
            # incumbent is no longer valid.  Retry the original requested
            # shell from this new successful continuation point.
            failed_upper = None
            event.update(
                {
                    "accepted": True,
                    "accepted_deformation_rms": incumbent_rms,
                    "accepted_objective": incumbent_objective,
                    "final_loss_eligible": final_eligible,
                    "final_relative_improvement": (
                        relative_improvement if final_eligible else None
                    ),
                    "final_selection_score": (
                        selection_score if final_eligible else None
                    ),
                    "accepted_direction_index": accepted_candidate[
                        "direction_index"
                    ],
                    "accepted_proposal_radius": accepted_candidate[
                        "proposal_radius"
                    ],
                }
            )
            events.append(event)
            self._direct_planar_log_iteration(
                incumbent,
                incumbent_objective,
                {
                    "optimizer": strategy_name,
                    "requested_target_rms": requested_target,
                    "attempted_target_rms": attempted_target,
                    "normalized_deformation_rms": incumbent_rms,
                    "direction_strategy": accepted_candidate["direction"][
                        "strategy"
                    ],
                    "action_repair_accepted_steps": repair["accepted_steps"],
                    "task_success": True,
                },
            )
            self._save_optimizer_checkpoint(
                incumbent,
                optimizer=strategy_name,
                objective=incumbent_objective,
                accepted_steps=accepted_proposals,
                iterations_attempted=iteration + 1,
                contact_scale=1.0,
                stages=[],
                event="accepted_target_shell_after_action_repair",
            )

        # Preserve the last search incumbent for diagnostics, then commit the
        # balanced deformation/performance candidate that also satisfies the
        # final loss budget.  This permits a temporary 2% degradation along
        # the search path without allowing any such degradation in the saved
        # result.
        search_incumbent_objective = float(incumbent_objective)
        search_incumbent_rms = float(incumbent_rms)
        search_termination_reason = termination_reason
        selected_search_incumbent = bool(
            best_final_iteration is not None
            and abs(best_final_rms - search_incumbent_rms)
            <= float(config["minimum_expansion"])
            and np.array_equal(best_final_params, incumbent)
        )
        incumbent = best_final_params.copy()
        incumbent_objective = float(best_final_objective)
        incumbent_rms = float(best_final_rms)

        # Count shells achieved by the candidate selected under the final
        # policy, rather than by a temporary search-only incumbent.
        completed_shells = 0
        while (
            completed_shells < len(schedule)
            and incumbent_rms + 1e-12 >= schedule[completed_shells]
        ):
            completed_shells += 1
            failed_upper = None
        if completed_shells >= len(schedule):
            termination_reason = "target_deformation_reached"

        final_config = dict(config)
        final_config["loss_budget"] = final_loss_budget
        final_gate = self._morphology_full_replay_gate(
            incumbent,
            reference=reference,
            baseline_objective=baseline_objective,
            incumbent_rms=incumbent_rms,
            config=final_config,
            require_expansion=False,
        )
        rolled_back = not bool(final_gate.get("accepted", False))
        if rolled_back:
            incumbent = reference.copy()
            incumbent_objective = float(baseline_gate["objective"])
            incumbent_rms = 0.0
            completed_shells = 0
            termination_reason = "final_gate_failed_stage1_rollback"
        final_collision = self._design_collision_report_for_params(incumbent)
        self._design_collision_final_report = (
            None if final_collision is None else final_collision.to_dict()
        )
        delta_diag = self._direct_planar_delta_diagnostics(reference, incumbent)
        self._action_optimizer_diagnostics = {
            "optimizer": strategy_name,
            "status": "rolled_back" if rolled_back else "done",
            "termination_reason": termination_reason,
            "config": config,
            "baseline_objective": float(baseline_objective),
            "final_objective": float(incumbent_objective),
            "improvement": float(baseline_objective - incumbent_objective),
            "accepted_proposals": int(accepted_proposals),
            "final_eligible_proposals": int(final_eligible_proposals),
            "pass_iteration_budget": int(maxiter),
            "iterations_attempted": int(len(events)),
            "completed_target_shells": int(completed_shells),
            "requested_final_target_rms": float(schedule[-1]),
            "highest_attempted_target_rms": float(highest_attempted_target),
            "normalized_deformation_rms": float(incumbent_rms),
            "maximum_feasible_rms": float(maximum_final_eligible_rms),
            "maximum_search_feasible_rms": float(search_incumbent_rms),
            "search_final_objective": float(search_incumbent_objective),
            "search_termination_reason": search_termination_reason,
            "final_loss_limit": float(final_loss_limit),
            "final_selection": {
                "policy": (
                    "deformation_times_relative_improvement_within_"
                    "final_loss_budget"
                ),
                "score": float(best_final_selection_score),
                "relative_improvement": float(
                    best_final_relative_improvement
                ),
                "tie_break": "lower_objective_then_larger_rms",
                "selected_search_incumbent": selected_search_incumbent,
                "selected_stage1_baseline": bool(
                    best_final_iteration is None
                ),
                "proposal_iteration": best_final_iteration,
            },
            "baseline_gate": baseline_gate,
            "final_gate": final_gate,
            "rolled_back_to_stage1": bool(rolled_back),
            "proposal_events": events,
            "direct_planar_delta": delta_diag,
            "trust_region": self._action_trust_diagnostics(),
        }
        print(
            f"[optimizer] {strategy_name} status =",
            self._action_optimizer_diagnostics["status"],
            "completed_shells =",
            int(completed_shells),
            "normalized_rms =",
            float(incumbent_rms),
            "target_rms =",
            float(schedule[-1]),
            "objective =",
            float(incumbent_objective),
            flush=True,
        )
        return incumbent


    def _optimize_action_trust_region(self, params0: np.ndarray):
        self.sim.set_contact_scale(1.0)
        initial_callback_failures = 0
        initial_callback_attempts = max(
            1,
            int(
                getattr(
                    self.args,
                    "action_trust_failure_patience",
                    1,
                )
                or 1
            ),
        )
        initial_callback_error = None
        for _ in range(initial_callback_attempts):
            try:
                self.callback(params0, render=False, log=True)
                initial_callback_error = None
                break
            except Exception as exc:
                initial_callback_failures += 1
                initial_callback_error = exc
        if initial_callback_error is not None:
            raise initial_callback_error
        initial_f = float(self.f_log[-1][1]) if self.f_log else float("inf")
        maxiter = max(0, int(getattr(self.args, "optimize_maxiter", 100)))
        maxls = max(1, int(getattr(self.args, "maxls", 20)))
        c1 = float(getattr(self.args, "direct_planar_armijo_c1", 1e-4) or 1e-4)
        params = np.asarray(params0, dtype=np.float64).copy()
        self._reset_action_trust_region()
        accepted_total = 0
        nit = 0
        t0 = time.time()
        stage_records = []
        optimizer_status = "done"
        stop_optimization = False
        previous_stage_objective = None
        full_contact_incumbent_params = params.copy()
        full_contact_incumbent_f = float(initial_f)
        full_contact_incumbent_source = "initial"
        full_contact_restarts = 0

        def evaluate_loss_and_grad_with_retries(stage_record):
            """Retry transient contact-gradient failures within policy."""

            config = self._action_trust_state["config"]
            attempts = max(1, int(config["failure_patience"]))
            last_error = None
            for attempt in range(attempts):
                try:
                    return self.loss_and_grad(params)
                except Exception as exc:
                    last_error = exc
                    stage_record["gradient_failures"] += 1
                    stage_record["last_error"] = repr(exc)
                    stage_record["gradient_retry_attempts"] = int(
                        stage_record.get("gradient_retry_attempts", 0) + 1
                    )
                    if attempt + 1 < attempts:
                        self._action_trust_state["radius"] = max(
                            float(config["minimum_radius"]),
                            float(self._action_trust_state["radius"])
                            * float(config["shrink_factor"]),
                        )
            raise last_error

        try:
            for stage_index, (contact_scale, stage_iters) in enumerate(
                self._contact_continuation_budgets(maxiter)
            ):
                self.sim.set_contact_scale(contact_scale)
                stage_radius = self._prepare_action_trust_stage(
                    stage_index
                )
                stage_record = {
                    "stage_index": int(stage_index),
                    "contact_scale": float(contact_scale),
                    "iteration_budget": int(stage_iters),
                    "iterations_attempted": 0,
                    "accepted_steps": 0,
                    "line_search_failures": 0,
                    "trial_exceptions": 0,
                    "gradient_failures": 0,
                    "termination_reason": "budget_exhausted",
                    "consecutive_line_search_failures": 0,
                    **stage_radius,
                }
                try:
                    f, grad = evaluate_loss_and_grad_with_retries(
                        stage_record
                    )
                except Exception as exc:
                    optimizer_status = "recovered"
                    stage_record["termination_reason"] = "gradient_evaluation_failed"
                    stage_record["last_error"] = repr(exc)
                    stage_records.append(stage_record)
                    break
                is_full_contact = bool(
                    np.isclose(
                        float(contact_scale),
                        1.0,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
                if is_full_contact:
                    stage_record["pre_restart_objective"] = float(f)
                    if (
                        np.isfinite(full_contact_incumbent_f)
                        and (
                            not np.isfinite(f)
                            or full_contact_incumbent_f
                            < float(f) - 1e-12
                        )
                    ):
                        params = full_contact_incumbent_params.copy()
                        f, grad = evaluate_loss_and_grad_with_retries(
                            stage_record
                        )
                        full_contact_restarts += 1
                        stage_record["full_contact_incumbent_restart"] = True
                        stage_record[
                            "full_contact_incumbent_objective"
                        ] = float(full_contact_incumbent_f)
                    else:
                        stage_record["full_contact_incumbent_restart"] = False
                stage_record["start_objective"] = float(f)
                if previous_stage_objective is not None:
                    stage_record["objective_jump_from_previous_scale"] = (
                        float(f) - float(previous_stage_objective)
                    )
                print(
                    "[optimizer] contact_stage_baseline =",
                    json.dumps(stage_record, sort_keys=True),
                    flush=True,
                )
                self._save_optimizer_checkpoint(
                    params,
                    optimizer="action_trust_region",
                    objective=f,
                    accepted_steps=accepted_total,
                    iterations_attempted=nit,
                    contact_scale=contact_scale,
                    stages=stage_records + [stage_record],
                    event="stage_baseline",
                )
                consecutive_failures = 0
                for _ in range(stage_iters):
                    nit += 1
                    stage_record["iterations_attempted"] += 1
                    direction = self._action_trust_direction(
                        params, np.asarray(grad, dtype=np.float64)
                    )
                    descent = float(np.dot(grad, direction))
                    if not np.isfinite(descent) or descent >= 0.0 or np.linalg.norm(direction) < 1e-12:
                        stage_record["termination_reason"] = "non_descent_direction"
                        break

                    exceptions_before = int(
                        self._action_trust_state["trial_exceptions"]
                    )
                    (
                        accepted,
                        accepted_params,
                        accepted_f,
                        accepted_info,
                        alpha,
                        trials,
                    ) = self._action_trust_search_with_fallback(
                        params,
                        f,
                        direction,
                        grad,
                        c1=c1,
                        maxls=maxls,
                    )
                    trust_event = dict(
                        self._last_action_trust_event or {}
                    )
                    stage_record["trial_exceptions"] += (
                        int(self._action_trust_state["trial_exceptions"])
                        - exceptions_before
                    )
                    if accepted:
                        consecutive_failures = 0
                        params = accepted_params
                        f = float(accepted_f)
                        accepted_total += 1
                        stage_record["accepted_steps"] += 1
                        if (
                            is_full_contact
                            and np.isfinite(f)
                            and (
                                not np.isfinite(full_contact_incumbent_f)
                                or f < full_contact_incumbent_f - 1e-12
                            )
                        ):
                            full_contact_incumbent_params = params.copy()
                            full_contact_incumbent_f = float(f)
                            full_contact_incumbent_source = "accepted_step"
                        log_info = dict(accepted_info or {})
                        log_info.update(
                            {
                                "optimizer": "action_trust_region",
                                "alpha": float(alpha),
                                "line_search_trials": int(trials),
                                "contact_scale": float(contact_scale),
                                "trust_radius": trust_event.get("radius"),
                                "trust_radius_next": trust_event.get(
                                    "next_radius"
                                ),
                                "trust_agreement_ratio": trust_event.get(
                                    "agreement_ratio"
                                ),
                                "trajectory_utilization_step": (
                                    trust_event.get(
                                        "trajectory_utilization_norm"
                                    )
                                ),
                            }
                        )
                        self._direct_planar_log_iteration(
                            params, f, log_info
                        )
                        self._save_optimizer_checkpoint(
                            params,
                            optimizer="action_trust_region",
                            objective=f,
                            accepted_steps=accepted_total,
                            iterations_attempted=nit,
                            contact_scale=contact_scale,
                            stages=stage_records + [stage_record],
                            event="accepted_step",
                        )
                        try:
                            _, grad = evaluate_loss_and_grad_with_retries(
                                stage_record
                            )
                        except Exception as exc:
                            optimizer_status = "recovered"
                            stop_optimization = True
                            stage_record["termination_reason"] = "gradient_evaluation_failed"
                            stage_record["last_error"] = repr(exc)
                    if stop_optimization:
                        break
                    if not accepted:
                        consecutive_failures += 1
                        stage_record["line_search_failures"] += 1
                        stage_record[
                            "consecutive_line_search_failures"
                        ] = consecutive_failures
                        if (
                            consecutive_failures
                            >= self._action_trust_state["config"][
                                "failure_patience"
                            ]
                        ):
                            stage_record[
                                "termination_reason"
                            ] = "line_search_failed"
                            break
                        try:
                            _, grad = evaluate_loss_and_grad_with_retries(
                                stage_record
                            )
                        except Exception as exc:
                            optimizer_status = "recovered"
                            stop_optimization = True
                            stage_record[
                                "termination_reason"
                            ] = "gradient_evaluation_failed"
                            stage_record["last_error"] = repr(exc)
                            break
                stage_record["end_objective"] = float(f)
                stage_record["trust_radius_end"] = float(
                    self._action_trust_state["radius"]
                )
                previous_stage_objective = float(f)
                stage_records.append(stage_record)
                self._save_optimizer_checkpoint(
                    params,
                    optimizer="action_trust_region",
                    objective=f,
                    accepted_steps=accepted_total,
                    iterations_attempted=nit,
                    contact_scale=contact_scale,
                    stages=stage_records,
                    event="stage_end",
                )
                if stop_optimization:
                    break
        finally:
            self.sim.set_contact_scale(1.0)

        final_selection = {
            "selected": "current",
            "discarded_objective": None,
        }
        try:
            final_f, _ = self.forward(params, backward_flag=False)
        except Exception as exc:
            optimizer_status = "recovered"
            final_f = float("inf")
            if stage_records:
                stage_records[-1]["final_evaluation_error"] = repr(exc)
            self._save_optimizer_checkpoint(
                params,
                optimizer="action_trust_region",
                objective=final_f,
                accepted_steps=accepted_total,
                iterations_attempted=nit,
                contact_scale=1.0,
                stages=stage_records,
                event="final_evaluation_failed",
                error=repr(exc),
            )
        if (
            np.isfinite(full_contact_incumbent_f)
            and (
                not np.isfinite(final_f)
                or full_contact_incumbent_f < float(final_f) - 1e-12
            )
        ):
            current_params = params.copy()
            current_final_f = float(final_f)
            incumbent_params = full_contact_incumbent_params.copy()
            try:
                incumbent_final_f, _ = self.forward(
                    incumbent_params,
                    backward_flag=False,
                )
            except Exception as exc:
                optimizer_status = "recovered"
                final_selection = {
                    "selected": "current",
                    "discarded_objective": float(
                        full_contact_incumbent_f
                    ),
                    "incumbent_evaluation_error": repr(exc),
                }
                params = current_params
                final_f = current_final_f
                if np.isfinite(final_f):
                    try:
                        # The failed incumbent replay may have replaced task
                        # endpoint caches. Restore caches for the parameters
                        # that are actually returned whenever possible.
                        final_f, _ = self.forward(
                            params,
                            backward_flag=False,
                        )
                    except Exception as restore_exc:
                        final_selection[
                            "current_cache_restore_error"
                        ] = repr(restore_exc)
                self._save_optimizer_checkpoint(
                    params,
                    optimizer="action_trust_region",
                    objective=final_f,
                    accepted_steps=accepted_total,
                    iterations_attempted=nit,
                    contact_scale=1.0,
                    stages=stage_records,
                    event="full_contact_incumbent_evaluation_failed",
                    error=repr(exc),
                )
            else:
                final_selection = {
                    "selected": "full_contact_incumbent",
                    "discarded_objective": current_final_f,
                }
                params = incumbent_params
                final_f = float(incumbent_final_f)
                self._save_optimizer_checkpoint(
                    params,
                    optimizer="action_trust_region",
                    objective=final_f,
                    accepted_steps=accepted_total,
                    iterations_attempted=nit,
                    contact_scale=1.0,
                    stages=stage_records,
                    event="full_contact_incumbent_selected",
                )
        if optimizer_status != "done":
            self._direct_planar_log_iteration(
                params,
                final_f,
                {
                    "optimizer": "action_trust_region",
                    "contact_scale": 1.0,
                    "recovered_final_evaluation": True,
                },
            )
        t1 = time.time()
        solver_diagnostics = (
            dict(self.sim.get_solver_diagnostics())
            if hasattr(self.sim, "get_solver_diagnostics")
            else {}
        )
        termination_reason = (
            stage_records[-1]["termination_reason"]
            if stage_records
            else "zero_iteration_budget"
        )
        self._action_optimizer_diagnostics = {
            "status": optimizer_status,
            "termination_reason": termination_reason,
            "iterations_attempted": int(nit),
            "accepted_steps": int(accepted_total),
            "initial_callback_failures": int(initial_callback_failures),
            "param_delta": float(np.linalg.norm(params - params0)),
            "improvement": initial_f - float(final_f),
            "final_objective": float(final_f),
            "stages": stage_records,
            "solver": solver_diagnostics,
            "trust_region": self._action_trust_diagnostics(),
            "full_contact_incumbent": {
                "objective": float(full_contact_incumbent_f),
                "source": full_contact_incumbent_source,
                "stage_restarts": int(full_contact_restarts),
                **final_selection,
            },
        }
        print("time = ", t1 - t0)
        print(
            "[optimizer] action_trust_region status =", optimizer_status,
            "nit =", int(nit),
            "accepted_steps =", int(accepted_total),
            "param_delta =", float(np.linalg.norm(params - params0)),
            "improvement =", initial_f - float(final_f),
            flush=True,
        )
        print(
            "[optimizer] action_trust_region_diagnostics =",
            json.dumps(self._action_optimizer_diagnostics, sort_keys=True),
            flush=True,
        )
        return params


    def optimize(self, params0: np.ndarray):
        """Dispatch one resolved strategy through the canonical registry."""

        from bilevel.lower.optimizers import (
            OPTIMIZER_REGISTRY,
            OptimizationMode,
            optimizer_selection_from_runner,
        )

        has_morphology = (
            self.optimize_design
            and self.design_bundle is not None
            and self.ndof_cage > 0
        )
        explicit_strategy = str(
            getattr(self.args, "optimizer_strategy", "") or ""
        ).strip()
        if has_morphology and explicit_strategy == "shape_only":
            mode = OptimizationMode.SHAPE_ONLY
        else:
            mode = (
                OptimizationMode.CO_REFINEMENT
                if has_morphology
                else OptimizationMode.ACTION_ONLY
            )
        design_protocol = (
            getattr(self.design_bundle, "generic_design_protocol", None)
            if has_morphology
            else None
        )
        strategy_name = optimizer_selection_from_runner(
            self.args,
            mode=mode,
            design_protocol=design_protocol,
        )
        self._optimizer_metadata = {
            "mode": mode.value,
            "strategy": strategy_name,
        }
        print(
            "[optimizer] selected",
            json.dumps(self._optimizer_metadata, sort_keys=True),
            flush=True,
        )
        return OPTIMIZER_REGISTRY.optimize(
            strategy_name,
            self,
            params0,
            mode=mode,
            design_protocol=design_protocol,
        )

    @staticmethod
    def _atomic_save_numpy(path: str, value: np.ndarray):
        tmp_path = f"{path}.tmp-{os.getpid()}"
        try:
            with open(tmp_path, "wb") as fp:
                np.save(fp, value)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _atomic_save_json(path: str, value: dict):
        tmp_path = f"{path}.tmp-{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(value, fp, indent=2, sort_keys=True)
                fp.flush()
                os.fsync(fp.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _save_optimizer_checkpoint(
        self,
        params: np.ndarray,
        *,
        optimizer: str,
        objective: float,
        accepted_steps: int,
        iterations_attempted: int,
        contact_scale: float,
        stages: list,
        event: str,
        error: Optional[str] = None,
    ):
        save_dir = getattr(self.args, "rollout_dir", None)
        if not save_dir:
            return
        try:
            params = np.asarray(params, dtype=np.float64)
            metadata = self.parameter_artifact_metadata()
            if (
                params.ndim != 1
                or params.size != metadata.total_dim
                or not np.all(np.isfinite(params))
            ):
                raise ValueError(
                    "optimizer checkpoint received invalid parameters"
                )
            os.makedirs(save_dir, exist_ok=True)
            self._atomic_save_numpy(
                os.path.join(save_dir, "params_checkpoint.npy"),
                params,
            )
            self._atomic_save_json(
                os.path.join(save_dir, "params_checkpoint_meta.json"),
                metadata.to_dict(),
            )
            flog = (
                np.asarray(self.f_log)
                if self.f_log
                else np.zeros((0, 2), dtype=np.float64)
            )
            self._atomic_save_numpy(
                os.path.join(save_dir, "logs_checkpoint.npy"),
                flog,
            )
            checkpoint = {
                "optimizer": str(optimizer),
                "event": str(event),
                "objective": float(objective),
                "accepted_steps": int(accepted_steps),
                "iterations_attempted": int(iterations_attempted),
                "contact_scale": float(contact_scale),
                "stages": stages,
                "updated_unix_time": float(time.time()),
            }
            if error is not None:
                checkpoint["error"] = str(error)
            self._atomic_save_json(
                os.path.join(save_dir, "optimizer_checkpoint.json"),
                checkpoint,
            )
        except Exception as exc:
            print_info(
                "[WARN] failed to write optimizer checkpoint:",
                repr(exc),
            )

    def save(self, save_dir: str, params: np.ndarray):
        metadata = self.parameter_artifact_metadata()
        params = np.asarray(params, dtype=np.float64)
        if params.ndim != 1 or params.size != metadata.total_dim:
            raise ValueError(
                f"cannot save parameter vector with shape {params.shape}; "
                f"expected ({metadata.total_dim},)"
            )
        if not np.all(np.isfinite(params)):
            raise ValueError("cannot save non-finite parameter vector")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "params.npy"), "wb") as fp:
            np.save(fp, params)
        with open(
            os.path.join(save_dir, "params_meta.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(
                metadata.to_dict(),
                fp,
                indent=2,
                sort_keys=True,
            )
        flog = np.array(self.f_log) if len(self.f_log) > 0 else np.zeros((0, 2))
        with open(os.path.join(save_dir, "logs.npy"), "wb") as fp:
            np.save(fp, flog)
        diag_fn = getattr(self.task, "rollout_diagnostics", None)
        rejected = getattr(
            self,
            "_staged_optimizer_rejected_checkpoint",
            None,
        )
        if rejected is not None:
            rejected_params = np.asarray(
                rejected.get("params"),
                dtype=np.float64,
            )
            stage_name = str(rejected.get("stage", "unknown"))
            safe_stage = "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in stage_name
            )
            rejected_dir = os.path.join(
                save_dir,
                f"rejected_stage_{safe_stage}",
            )
            if (
                rejected_params.ndim == 1
                and rejected_params.size == metadata.total_dim
                and np.all(np.isfinite(rejected_params))
            ):
                os.makedirs(rejected_dir, exist_ok=True)
                with open(
                    os.path.join(rejected_dir, "params.npy"),
                    "wb",
                ) as fp:
                    np.save(fp, rejected_params)
                with open(
                    os.path.join(rejected_dir, "params_meta.json"),
                    "w",
                    encoding="utf-8",
                ) as fp:
                    json.dump(
                        metadata.to_dict(),
                        fp,
                        indent=2,
                        sort_keys=True,
                    )
                with open(
                    os.path.join(rejected_dir, "logs.npy"),
                    "wb",
                ) as fp:
                    np.save(fp, flog)
                rejected_manifest = {
                    "stage": stage_name,
                    "rolled_back": True,
                    "acceptance": dict(
                        rejected.get("acceptance", {})
                    ),
                    "params_path": os.path.join(
                        rejected_dir,
                        "params.npy",
                    ),
                }
                if diag_fn is not None:
                    try:
                        rejected_diagnostics = diag_fn(
                            self,
                            rejected_params,
                        )
                        rejected_diagnostics["rejected_stage"] = stage_name
                        rejected_diagnostics["stage_acceptance"] = dict(
                            rejected.get("acceptance", {})
                        )
                        with open(
                            os.path.join(
                                rejected_dir,
                                "diagnostics.json",
                            ),
                            "w",
                            encoding="utf-8",
                        ) as fp:
                            json.dump(
                                rejected_diagnostics,
                                fp,
                                indent=2,
                            )
                    except Exception as exc:
                        print_info(
                            "[WARN] failed to write rejected-stage "
                            "rollout diagnostics:",
                            repr(exc),
                        )
                with open(
                    os.path.join(rejected_dir, "manifest.json"),
                    "w",
                    encoding="utf-8",
                ) as fp:
                    json.dump(
                        rejected_manifest,
                        fp,
                        indent=2,
                    )
                if self._action_optimizer_diagnostics is not None:
                    self._action_optimizer_diagnostics[
                        "rejected_checkpoint"
                    ] = rejected_manifest
            else:
                print_info(
                    "[WARN] skipped invalid rejected-stage parameter "
                    f"vector with shape {rejected_params.shape}"
                )
        if diag_fn is not None:
            try:
                diagnostics = diag_fn(self, params)
                optimizer_diagnostics = dict(
                    self._optimizer_metadata or {}
                )
                if self._action_optimizer_diagnostics is not None:
                    optimizer_diagnostics["details"] = (
                        self._action_optimizer_diagnostics
                    )
                if optimizer_diagnostics:
                    diagnostics["optimizer"] = optimizer_diagnostics
                with open(os.path.join(save_dir, "diagnostics.json"), "w") as fp:
                    json.dump(diagnostics, fp, indent=2)
            except Exception as exc:
                print_info("[WARN] failed to write rollout diagnostics:", repr(exc))

    def save_initial(self, save_dir: str, params: np.ndarray):
        """Save the reproducible pre-optimization state beside the final state."""
        metadata = self.parameter_artifact_metadata()
        params = np.asarray(params, dtype=np.float64)
        if params.ndim != 1 or params.size != metadata.total_dim:
            raise ValueError(
                "cannot save initial parameter vector with shape "
                f"{params.shape}; expected ({metadata.total_dim},)"
            )
        if not np.all(np.isfinite(params)):
            raise ValueError("cannot save non-finite initial parameter vector")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "params_initial.npy"), "wb") as fp:
            np.save(fp, params)
        with open(
            os.path.join(save_dir, "params_initial_meta.json"),
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(
                metadata.to_dict(),
                fp,
                indent=2,
                sort_keys=True,
            )

    def load(self, load_dir: str):
        params_name = "params.npy"
        logs_name = "logs.npy"
        meta_name = "params_meta.json"
        if not os.path.exists(os.path.join(load_dir, params_name)):
            checkpoint_path = os.path.join(
                load_dir, "params_checkpoint.npy"
            )
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"no final or checkpoint parameters in {load_dir}"
                )
            params_name = "params_checkpoint.npy"
            logs_name = "logs_checkpoint.npy"
            meta_name = "params_checkpoint_meta.json"
            print_info(
                "[WARN] final parameters are missing; loading the latest "
                "accepted optimizer checkpoint"
            )
        with open(os.path.join(load_dir, params_name), "rb") as fp:
            params = np.load(fp)
        logs_path = os.path.join(load_dir, logs_name)
        if os.path.exists(logs_path):
            with open(logs_path, "rb") as fp:
                self.f_log = list(np.load(fp))
        else:
            self.f_log = []
        meta_path = os.path.join(load_dir, meta_name)
        metadata = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fp:
                metadata = json.load(fp)
        return self.normalize_loaded_params(params, metadata)

    def visualize_final(self, params: np.ndarray):
        if self.visualize:
            if self.optimize_design and self.design_bundle is not None:
                print("cage params = ", params[-self.ndof_cage:])
            print_info("Press [Esc] to continue")
            record_base = getattr(self.args, "record_file_name", None) or "record"
            self.callback(params, render=True, record=self.args.record, record_path=record_base + "_optimized.gif", log=False)

    def fd_test(self, params: np.ndarray, num_checks: int = 8):
        f, grad = self.loss_and_grad(params)
        n_params = len(params)
        eps = 1e-5
        for _ in range(num_checks):
            df_fd = np.zeros(n_params)
            for i in range(n_params):
                p2 = params.copy()
                p2[i] += eps
                f2, _ = self.forward(p2, backward_flag=False)
                df_fd[i] = (f2 - f) / eps
            abs_error = np.linalg.norm(df_fd - grad)
            rel_error = abs_error / (np.linalg.norm(grad) + 1e-7)
            print("eps = ", eps)
            print("df_dparam : error = {:10.6e}, rel_error = {:10.6e}".format(abs_error, rel_error))
            df_fd_n = df_fd / (np.linalg.norm(df_fd) + 1e-12)
            g_n = grad / (np.linalg.norm(grad) + 1e-12)
            print("dot product: ", np.dot(df_fd_n, g_n))
            eps /= 10.0
