"""Shared construction of canonical lower-level runner arguments."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .optimizers import optimizer_policy_from_config


def runtime_value(
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    """Resolve one runtime value as context, then merged config, then default."""

    value = context.get(name)
    return config.get(name, default) if value is None else value


def build_common_runner_args(
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    rollout_dir: Path,
    *,
    default_maxiter: int,
) -> SimpleNamespace:
    """Build the runner fields shared by every canonical task."""

    return SimpleNamespace(
        verbose=bool(config.get("verbose", False)),
        record=False,
        rollout_dir=str(Path(rollout_dir)),
        record_file_name=str(Path(rollout_dir) / "replay"),
        action_init_mode=str(
            runtime_value(
                config,
                context,
                "low_level_action_init_mode",
                "task",
            )
        ).strip().lower(),
        optimize_maxiter=int(
            runtime_value(
                config,
                context,
                "low_level_maxiter",
                default_maxiter,
            )
        ),
        grad_clip=float(
            runtime_value(config, context, "low_level_grad_clip", 0.0) or 0.0
        ),
        step_scale=float(
            runtime_value(config, context, "low_level_step_scale", 1.0) or 1.0
        ),
        lr=float(runtime_value(config, context, "low_level_lr", 0.02) or 0.02),
    )


def build_runner_args(
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    rollout_dir: Path,
) -> SimpleNamespace:
    """Build all task-independent runner options from one canonical config.

    Optional fields are attached only when explicitly configured or overridden,
    preserving the runner's established defaults for tasks that never supplied
    them.
    """

    args = build_common_runner_args(
        config,
        context,
        rollout_dir,
        default_maxiter=int(config.get("low_level_maxiter", 100)),
    )
    args.optimizer_policy = optimizer_policy_from_config(config)
    optimizer_override = context.get("optimizer_strategy")
    if optimizer_override is not None:
        args.optimizer_strategy = str(optimizer_override)
    option_types = {
        "low_level_maxls": ("maxls", int),
        "action_optimizer": ("action_optimizer", str),
        "direct_planar_optimizer": ("direct_planar_optimizer", str),
        "direct_planar_armijo_c1": (
            "direct_planar_armijo_c1",
            float,
        ),
        "direct_planar_max_handle_step": (
            "direct_planar_max_handle_step",
            float,
        ),
        "direct_planar_max_action_step": (
            "direct_planar_max_action_step",
            float,
        ),
        "action_trust_radius_initial": (
            "action_trust_radius_initial",
            float,
        ),
        "action_trust_radius_initial_by_stage": (
            "action_trust_radius_initial_by_stage",
            dict,
        ),
        "action_trust_radius_min": (
            "action_trust_radius_min",
            float,
        ),
        "action_trust_radius_max": (
            "action_trust_radius_max",
            float,
        ),
        "action_trust_norm_mode": (
            "action_trust_norm_mode",
            str,
        ),
        "action_trust_temporal_basis_knots": (
            "action_trust_temporal_basis_knots",
            int,
        ),
        "action_trust_horizon_scaling": (
            "action_trust_horizon_scaling",
            str,
        ),
        "action_trust_reference_ctrl_steps": (
            "action_trust_reference_ctrl_steps",
            int,
        ),
        "action_trust_monotone_fallback": (
            "action_trust_monotone_fallback",
            bool,
        ),
        "action_trust_raw_gradient_fallback": (
            "action_trust_raw_gradient_fallback",
            bool,
        ),
        "action_trust_opposite_direction_poll": (
            "action_trust_opposite_direction_poll",
            bool,
        ),
        "action_trust_radius_restart": (
            "action_trust_radius_restart",
            bool,
        ),
        "action_trust_stage_reset_radius": (
            "action_trust_stage_reset_radius",
            float,
        ),
        "action_trust_failure_patience": (
            "action_trust_failure_patience",
            int,
        ),
        "action_trust_shrink_factor": (
            "action_trust_shrink_factor",
            float,
        ),
        "action_trust_growth_factor": (
            "action_trust_growth_factor",
            float,
        ),
        "action_trust_accept_ratio": (
            "action_trust_accept_ratio",
            float,
        ),
        "action_trust_shrink_ratio": (
            "action_trust_shrink_ratio",
            float,
        ),
        "action_trust_growth_ratio": (
            "action_trust_growth_ratio",
            float,
        ),
        "action_trust_boundary_fraction": (
            "action_trust_boundary_fraction",
            float,
        ),
        "action_trust_force_preconditioner": (
            "action_trust_force_preconditioner",
            float,
        ),
        "action_trust_torque_preconditioner": (
            "action_trust_torque_preconditioner",
            float,
        ),
        "action_trust_knot_gradient_normalization_power": (
            "action_trust_knot_gradient_normalization_power",
            float,
        ),
        "action_trust_temporal_preconditioner_power": (
            "action_trust_temporal_preconditioner_power",
            float,
        ),
        "action_trust_diagnostic_event_limit": (
            "action_trust_diagnostic_event_limit",
            int,
        ),
        "direct_planar_design_step_scale": (
            "direct_planar_design_step_scale",
            float,
        ),
        "direct_planar_action_step_scale": (
            "direct_planar_action_step_scale",
            float,
        ),
        "direct_planar_design_block_filter": (
            "direct_planar_design_block_filter",
            str,
        ),
        "direct_planar_physical_metric": (
            "direct_planar_physical_metric",
            bool,
        ),
        "direct_planar_diagnostic_event_limit": (
            "direct_planar_diagnostic_event_limit",
            int,
        ),
        "morphology_expansion_radius_initial": (
            "morphology_expansion_radius_initial",
            float,
        ),
        "morphology_expansion_radius_growth": (
            "morphology_expansion_radius_growth",
            float,
        ),
        "morphology_expansion_radius_shrink": (
            "morphology_expansion_radius_shrink",
            float,
        ),
        "morphology_expansion_max_rms": (
            "morphology_expansion_max_rms",
            float,
        ),
        "morphology_expansion_loss_budget": (
            "morphology_expansion_loss_budget",
            float,
        ),
        "morphology_expansion_final_loss_budget": (
            "morphology_expansion_final_loss_budget",
            float,
        ),
        "morphology_expansion_repair_min_steps": (
            "morphology_expansion_repair_min_steps",
            int,
        ),
        "morphology_expansion_repair_max_steps": (
            "morphology_expansion_repair_max_steps",
            int,
        ),
        "morphology_expansion_repair_attempt_multiplier": (
            "morphology_expansion_repair_attempt_multiplier",
            int,
        ),
        "morphology_expansion_coarse_iterations": (
            "morphology_expansion_coarse_iterations",
            int,
        ),
        "morphology_expansion_target_schedule": (
            "morphology_expansion_target_schedule",
            str,
        ),
        "morphology_expansion_target_tolerance": (
            "morphology_expansion_target_tolerance",
            float,
        ),
        "morphology_expansion_target_search_trials": (
            "morphology_expansion_target_search_trials",
            int,
        ),
        "design_collision_check": ("design_collision_check", bool),
        "design_collision_margin": ("design_collision_margin", float),
        "design_collision_max_report": (
            "design_collision_max_report",
            int,
        ),
        "design_collision_check_ground": (
            "design_collision_check_ground",
            bool,
        ),
        "design_fd_grad": ("design_fd_grad", bool),
        "design_fd_step": ("design_fd_step", float),
        "design_fd_trigger_tol": ("design_fd_trigger_tol", float),
        "stage2_robust_config": ("stage2_robust_config", str),
        "stage2_robust_calibration_count": (
            "stage2_robust_calibration_count",
            int,
        ),
        "stage2_robust_train_count": (
            "stage2_robust_train_count",
            int,
        ),
        "stage2_robust_heldout_count": (
            "stage2_robust_heldout_count",
            int,
        ),
        "stage2_robust_seed": ("stage2_robust_seed", int),
        "stage2_robust_risk_cvar": (
            "stage2_robust_risk_cvar",
            float,
        ),
        "stage2_min_deformation_rms": (
            "stage2_min_deformation_rms",
            float,
        ),
        "stage2_strong_deformation_rms": (
            "stage2_strong_deformation_rms",
            float,
        ),
        "stage2_min_robust_improvement": (
            "stage2_min_robust_improvement",
            float,
        ),
        "stage2_outer_iterations": ("stage2_outer_iterations", int),
        "stage2_robust_severity": ("stage2_robust_severity", float),
    }
    passthrough_options = {
        "contact_continuation_scales",
        "contact_continuation_weights",
    }
    for config_name, (argument_name, converter) in option_types.items():
        value = context.get(config_name)
        if value is None:
            value = config.get(config_name)
        if value is not None:
            setattr(args, argument_name, converter(value))
    for name in passthrough_options:
        value = context.get(name)
        if value is None:
            value = config.get(name)
        if value is not None:
            setattr(args, name, value)
    return args


def configure_mount_runner_args(
    args: Any,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Any:
    """Attach fixed-root constraint tolerances to shared runner arguments."""

    if not bool(config.get("preserve_handle_mount", False)):
        return args
    max_action_step = config.get(
        "mount_preserving_max_action_step",
        config.get("handle_head_max_action_step", 1e-4),
    )
    values = {
        "mount_preserving_max_shape_displacement": 0.125,
        "mount_translation_tolerance": 1e-7,
        "mount_rotation_tolerance": 1e-7,
        "mount_face_tolerance": 1e-7,
        "mount_preserving_max_action_step": max_action_step,
    }
    for name, default in values.items():
        value = context.get(name)
        if value is None:
            value = config.get(name, default)
        setattr(args, name, float(value))
    return args


__all__ = [
    "build_common_runner_args",
    "build_runner_args",
    "configure_mount_runner_args",
    "runtime_value",
]
