"""Maintained Stage 1 action and Stage 2 shape/action optimizers."""
from __future__ import annotations
from typing import FrozenSet, Mapping
import numpy as np
from .base import OptimizationMode, OptimizerStrategy
from .config import ACTION_TRUST_REGION, STAGED_ACTION_TRUST_REGION, SUCCESS_CONSTRAINED_TARGET_SHELL
from .registry import register_optimizer
_DIRECT_PLANAR_PROTOCOL = "connected_direct_planar_hexahedron"


class SuccessConstrainedTargetShell(OptimizerStrategy):
    """Pursue explicit cumulative-deformation shells behind hard gates."""

    name = SUCCESS_CONSTRAINED_TARGET_SHELL
    supported_modes = frozenset({OptimizationMode.CO_REFINEMENT})
    required_protocol = _DIRECT_PLANAR_PROTOCOL

    def optimize(self, runner, params0: np.ndarray) -> np.ndarray:
        return runner._optimize_success_constrained_target_shell(
            params0,
            strategy_name=self.name,
        )

class ActionTrustRegion(OptimizerStrategy):
    name = ACTION_TRUST_REGION
    supported_modes: FrozenSet[OptimizationMode] = frozenset(
        {OptimizationMode.ACTION_ONLY}
    )

    def optimize(self, runner, params0: np.ndarray) -> np.ndarray:
        return runner._optimize_action_trust_region(params0)

class StagedActionTrustRegion(OptimizerStrategy):
    """Apply one task-declared causal curriculum through the shared runner."""

    name = STAGED_ACTION_TRUST_REGION
    supported_modes: FrozenSet[OptimizationMode] = frozenset(
        {OptimizationMode.ACTION_ONLY}
    )

    def optimize(self, runner, params0: np.ndarray) -> np.ndarray:
        return self._optimize_task_stages(
            runner,
            params0,
            solve_stage=runner._optimize_action_trust_region,
            optimizer_name=STAGED_ACTION_TRUST_REGION,
            allow_morphology=False,
        )

    def _optimize_task_stages(
        self,
        runner,
        params0: np.ndarray,
        *,
        solve_stage,
        optimizer_name: str,
        allow_morphology: bool,
    ) -> np.ndarray:
        schedule_builder = getattr(
            runner.task,
            "optimization_stage_schedule",
            None,
        )
        stage_setter = getattr(runner.task, "set_optimization_stage", None)
        if not callable(schedule_builder) or not callable(stage_setter):
            raise TypeError(
                "staged_action_trust_region requires the task to implement "
                "optimization_stage_schedule(maxiter) and "
                "set_optimization_stage(stage)"
            )

        total_budget = max(
            0,
            int(getattr(runner.args, "optimize_maxiter", 100)),
        )
        if total_budget == 0:
            stage_setter("full")
            return solve_stage(params0)

        schedule = tuple(schedule_builder(total_budget))
        if not schedule:
            raise ValueError(
                "staged_action_trust_region received an empty stage schedule"
            )
        if sum(int(budget) for _, budget in schedule) != total_budget:
            raise ValueError(
                "staged action budgets must sum to optimize_maxiter"
            )

        params = np.asarray(params0, dtype=np.float64).copy()
        if hasattr(runner, "ndof_u") and hasattr(
            runner,
            "num_ctrl_steps",
        ):
            action_dim = int(runner.ndof_u * runner.num_ctrl_steps)
        elif allow_morphology:
            raise ValueError(
                "staged block-coordinate optimization requires ndof_u and "
                "num_ctrl_steps"
            )
        else:
            action_dim = len(params)
        if len(params) < action_dim:
            raise ValueError(
                "task-stage optimization received fewer parameters than "
                "the action trajectory requires"
            )
        if not allow_morphology and len(params) != action_dim:
            raise ValueError(
                "staged action optimization requires an action-only "
                "parameter vector"
            )
        runner._staged_optimizer_rejected_checkpoint = None
        runner._staged_stop_policy = None
        original_budget = total_budget
        stage_records = []
        completed_stages = []
        failed_stage = None
        stage_acceptance = getattr(
            runner.task,
            "optimization_stage_acceptance",
            None,
        )
        action_window = getattr(
            runner.task,
            "optimization_stage_action_window",
            None,
        )
        mask_attribute = "_action_optimizer_trainable_mask"
        had_original_mask = hasattr(runner, mask_attribute)
        original_mask = getattr(runner, mask_attribute, None)
        try:
            for stage_name, stage_budget in schedule:
                stage_setter(str(stage_name))
                runner.args.optimize_maxiter = int(stage_budget)
                stage_start = params.copy()
                start = None
                end = None
                if callable(action_window):
                    start, end = action_window(
                        str(stage_name),
                        int(runner.num_ctrl_steps),
                    )
                    start = int(start)
                    end = int(end)
                    if not 0 <= start < end <= int(
                        runner.num_ctrl_steps
                    ):
                        raise ValueError(
                            "optimization_stage_action_window(stage, "
                            "num_ctrl_steps) returned an invalid range "
                            f"{(start, end)!r}"
                        )
                    init_mode = str(
                        getattr(
                            runner.args,
                            "action_init_mode",
                            "task",
                        )
                    ).strip().lower()
                    if start > 0 and init_mode in ("zero", "random"):
                        held = params[
                            (start - 1) * runner.ndof_u :
                            start * runner.ndof_u
                        ].copy()
                        params = params.copy()
                        params[
                            start * runner.ndof_u :
                            end * runner.ndof_u
                        ] = np.tile(held, end - start)
                    trainable = np.zeros(len(params), dtype=bool)
                    trainable[
                        start * runner.ndof_u : end * runner.ndof_u
                    ] = True
                    if allow_morphology:
                        trainable[action_dim:] = True
                    setattr(runner, mask_attribute, trainable)
                warm_start_delta = float(
                    np.linalg.norm(params - stage_start)
                )
                radius_overrides = getattr(
                    runner.args,
                    "action_trust_radius_initial_by_stage",
                    None,
                )
                radius_override = None
                if isinstance(radius_overrides, Mapping):
                    raw_radius_override = radius_overrides.get(
                        str(stage_name)
                    )
                    if raw_radius_override is not None:
                        radius_override = float(raw_radius_override)
                        radius_max = float(
                            getattr(
                                runner.args,
                                "action_trust_radius_max",
                                radius_override,
                            )
                        )
                        if (
                            not np.isfinite(radius_override)
                            or radius_override <= 0.0
                            or radius_override > radius_max
                        ):
                            raise ValueError(
                                "action_trust_radius_initial_by_stage values "
                                "must be finite, positive, and no larger than "
                                "action_trust_radius_max"
                            )
                default_radius_initial = getattr(
                    runner.args, "action_trust_radius_initial", None
                )
                try:
                    if radius_override is not None:
                        runner.args.action_trust_radius_initial = (
                            radius_override
                        )
                    params = solve_stage(params)
                finally:
                    runner.args.action_trust_radius_initial = (
                        default_radius_initial
                    )
                future_hold_delta = 0.0
                future_hold_applied = False
                if (
                    callable(action_window)
                    and end < int(runner.num_ctrl_steps)
                    and init_mode in ("zero", "random")
                ):
                    future_start = int(end * runner.ndof_u)
                    held = params[
                        (end - 1) * runner.ndof_u :
                        end * runner.ndof_u
                    ].copy()
                    held_future = np.tile(
                        held,
                        int(runner.num_ctrl_steps) - end,
                    )
                    future_hold_delta = float(
                        np.linalg.norm(
                            params[
                                future_start:action_dim
                            ] - held_future
                        )
                    )
                    params = params.copy()
                    params[future_start:action_dim] = held_future
                    future_hold_applied = True
                diagnostics = dict(
                    getattr(runner, "_action_optimizer_diagnostics", {}) or {}
                )
                record = {
                    "name": str(stage_name),
                    "iteration_budget": int(stage_budget),
                    "action_window": (
                        None
                        if start is None or end is None
                        else [int(start), int(end)]
                    ),
                    "param_delta": float(
                        np.linalg.norm(params - stage_start)
                    ),
                    "diagnostics": diagnostics,
                    "warm_start_delta": warm_start_delta,
                    "future_hold_applied": future_hold_applied,
                    "future_hold_start_knot": (
                        int(end) if future_hold_applied else None
                    ),
                    "future_hold_delta": future_hold_delta,
                    "action_trust_radius_initial_override": radius_override,
                }
                if callable(stage_acceptance):
                    cached_acceptance = getattr(
                        runner.task,
                        "optimization_stage_acceptance_uses_cached_rollout",
                        None,
                    )
                    use_cached_acceptance = bool(
                        callable(cached_acceptance)
                        and cached_acceptance()
                    )
                    stage_forward_error = None
                    if use_cached_acceptance:
                        stage_objective = float(
                            diagnostics.get("final_objective", np.nan)
                        )
                        stage_terms = {}
                    else:
                        retries_fn = getattr(
                            runner.task,
                            "optimization_stage_acceptance_forward_retries",
                            None,
                        )
                        forward_attempts = max(
                            1,
                            int(
                                retries_fn()
                                if callable(retries_fn)
                                else 1
                            ),
                        )
                        forward_errors = []
                        for _ in range(forward_attempts):
                            try:
                                stage_objective, stage_terms = runner.forward(
                                    params,
                                    backward_flag=False,
                                )
                            except Exception as exc:
                                forward_errors.append(repr(exc))
                                continue
                            break
                        else:
                            stage_objective = float("inf")
                            stage_terms = {}
                            stage_forward_error = forward_errors[-1]
                        record["acceptance_forward_attempts"] = int(
                            len(forward_errors)
                            + (0 if stage_forward_error is not None else 1)
                        )
                        if forward_errors:
                            record["acceptance_forward_errors"] = list(
                                forward_errors
                            )
                    verdict = stage_acceptance(str(stage_name))
                    if isinstance(verdict, Mapping):
                        acceptance = dict(verdict)
                    else:
                        acceptance = {"accepted": bool(verdict)}
                    if "accepted" not in acceptance:
                        raise ValueError(
                            "optimization_stage_acceptance(stage) must "
                            "return a bool or a mapping containing 'accepted'"
                        )
                    acceptance["accepted"] = bool(
                        acceptance["accepted"]
                    )
                    record["objective_after_stage"] = float(
                        stage_objective
                    )
                    record["loss_terms_after_stage"] = {
                        key: float(value)
                        for key, value in stage_terms.items()
                    }
                    record["acceptance_rollout_source"] = (
                        "optimizer_cache"
                        if use_cached_acceptance
                        else "explicit_forward"
                    )
                    if stage_forward_error is not None:
                        acceptance = {
                            "accepted": False,
                            "stage": str(stage_name),
                            "reason": "stage_acceptance_forward_failed",
                            "exception": stage_forward_error,
                        }
                        record["acceptance_forward_error"] = (
                            stage_forward_error
                        )
                    record["per_stage_loss_after_stage"] = dict(
                        getattr(
                            runner,
                            "_last_forward_diagnostics",
                            {},
                        ).get("per_stage_loss", {})
                    )
                    record["acceptance"] = acceptance
                    if not acceptance["accepted"]:
                        failed_stage = str(stage_name)
                        # Retain the best trajectory found for the failed
                        # stage. It is still useful partial-task evidence for
                        # BASS even though the physical gate was not crossed.
                        record["rolled_back"] = False
                        record["gate_failed_retained"] = True
                    else:
                        completed_stages.append(str(stage_name))
                else:
                    completed_stages.append(str(stage_name))
                stage_records.append(record)
                if failed_stage is not None:
                    break
        finally:
            runner.args.optimize_maxiter = original_budget
            stage_setter("full")
            if had_original_mask:
                setattr(runner, mask_attribute, original_mask)
            elif hasattr(runner, mask_attribute):
                delattr(runner, mask_attribute)

        stop_policy = None
        if failed_stage is not None:
            stop_enabled = getattr(
                runner.task,
                "optimization_stage_stop_motion",
                None,
            )
            motion_dofs = getattr(
                runner.task,
                "optimization_stage_motion_dofs",
                None,
            )
            loss_mode = getattr(
                runner.task,
                "optimization_stage_future_loss_mode",
                None,
            )
            if (
                callable(stop_enabled)
                and bool(stop_enabled())
                and callable(motion_dofs)
                and callable(loss_mode)
                and callable(action_window)
            ):
                last_completed_stage = (
                    completed_stages[-1] if completed_stages else None
                )
                # The failed stage has already been simulated and optimized.
                # Freeze at its boundary, not at the previous completed
                # boundary. Otherwise a first-stage failure truncates every
                # loss term and collapses all candidate scores to zero.
                _, stop_knot = action_window(
                    failed_stage,
                    int(runner.num_ctrl_steps),
                )
                stop_policy = {
                    "enabled": True,
                    "reason": "stage_acceptance_failed",
                    "failed_stage": failed_stage,
                    "last_completed_stage": last_completed_stage,
                    "stop_knot": int(stop_knot),
                    "future_loss_mode": str(loss_mode()),
                    "motion_dofs": [
                        int(value) for value in motion_dofs()
                    ],
                }
                runner._staged_stop_policy = stop_policy

        final_forward_error = None
        try:
            final_objective, _ = runner.forward(
                params,
                backward_flag=False,
            )
            final_forward_diagnostics = dict(
                getattr(runner, "_last_forward_diagnostics", {}) or {}
            )
        except Exception as exc:
            final_forward_error = repr(exc)
            final_objective = float(
                next(
                    (
                        record.get("objective_after_stage")
                        for record in reversed(stage_records)
                        if np.isfinite(
                            float(
                                record.get(
                                    "objective_after_stage",
                                    np.inf,
                                )
                            )
                        )
                    ),
                    np.inf,
                )
            )
            final_forward_diagnostics = dict(
                getattr(runner, "_last_forward_diagnostics", {}) or {}
            )
        runner._action_optimizer_diagnostics = {
            "optimizer": optimizer_name,
            "task_stage_curriculum": True,
            "morphology_trainable": bool(allow_morphology),
            "status": (
                "recovered"
                if any(
                    record["diagnostics"].get("status") == "recovered"
                    for record in stage_records
                )
                else "done"
            ),
            "termination_reason": (
                "stage_acceptance_failed"
                if failed_stage is not None
                else "curriculum_complete"
            ),
            "failed_stage": failed_stage,
            "completed_stages": list(completed_stages),
            "last_completed_stage": (
                completed_stages[-1] if completed_stages else None
            ),
            "stop_policy": stop_policy,
            "iterations_attempted": int(
                sum(
                    record["diagnostics"].get("iterations_attempted", 0)
                    for record in stage_records
                )
            ),
            "accepted_steps": int(
                sum(
                    record["diagnostics"].get("accepted_steps", 0)
                    for record in stage_records
                )
            ),
            "committed_accepted_steps": int(
                sum(
                    record["diagnostics"].get("accepted_steps", 0)
                    for record in stage_records
                    if not record.get("rolled_back", False)
                )
            ),
            "rolled_back_accepted_steps": int(
                sum(
                    record["diagnostics"].get("accepted_steps", 0)
                    for record in stage_records
                    if record.get("rolled_back", False)
                )
            ),
            "param_delta": float(np.linalg.norm(params - params0)),
            "final_objective": float(final_objective),
            "final_per_stage_loss": final_forward_diagnostics.get(
                "per_stage_loss",
                {},
            ),
            "final_stage_stop": final_forward_diagnostics.get(
                "stage_stop"
            ),
            "final_forward_error": final_forward_error,
            "curriculum_stages": stage_records,
        }
        return params

for strategy in (ActionTrustRegion(), StagedActionTrustRegion(), SuccessConstrainedTargetShell()):
    register_optimizer(strategy)
