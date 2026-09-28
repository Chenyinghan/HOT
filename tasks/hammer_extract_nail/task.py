"""Canonical Handle-root hammer_extract_nail task plugin."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from bilevel import HandleSpec, MorphologySpec, RuntimeSpec, SearchSpec, TaskSpec
from bilevel.parameterization import UNIFIED_PARAMETERIZATION_ID
from bilevel.lower.optimizers import optimizer_contract_from_config
from tasks.base import TaskBase

from .objective import MISSION_NAME, TaskDynamics
from .task_progress import hammer_extract_nail_bass_reward


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / "config.json"


def _numbers(text: str, expected: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in str(text).split())
    if len(values) != expected:
        raise ValueError(
            f"expected {expected} numeric values, got {text!r}"
        )
    return values


class TaskDefinition(TaskBase):
    """Native two-function, five-stage hammer_extract_nail task contract."""

    config_path = CONFIG_PATH

    def _configure(self) -> None:
        task_kwargs = {
            key: self.config[key]
            for key in (
                "num_steps",
                "sub_steps",
                "coef_goal",
                "coef_tool_pose",
                "coef_impact",
                "coef_contact",
                "coef_control",
                "control_smooth_weight",
                "force_connectivity",
                "generic_design_protocol",
                "task_phase",
                "approach_end_fraction",
                "hammer_end_fraction",
                "transfer_end_fraction",
                "engage_end_fraction",
                "target_down_depth",
                "hammer_completion_depth",
                "target_up_lift",
                "goal_tolerance",
                "success_hold_knots",
                "premature_lift_tolerance",
                "approach_acceptance_distance",
                "transfer_approach_distance",
                "extraction_engage_distance",
                "engage_gate_tolerance",
                "extraction_approach_clearance",
                "max_contact_penetration",
                "geometry_audit_enabled",
                "geometry_audit_containment_fraction",
                "geometry_audit_normalized_depth",
                "geometry_audit_min_sustained_seconds",
                "hammer_target_speed",
                "hammer_speed_scale",
                "hammer_roll_pose_weight",
                "curriculum_stage_weights",
                "stage_stop_motion",
                "stage_future_loss_mode",
                "action_scale_x",
                "action_scale_y",
                "action_scale_z",
                "action_scale_roll",
                "roll_joint_name",
            )
            if key in self.config
        }
        self.numerical_task = TaskDynamics(**task_kwargs)
        self.numerical_task._task_config = dict(self.config)
        self.numerical_task._canonical_task_name = MISSION_NAME

        xml = dict(self.payload.get("xml", {}))
        if str(xml.get("root_aux_joint_type", "")).lower() != "revolute":
            raise ValueError(
                "hammer_extract_nail requires one public revolute root auxiliary "
                "joint"
            )
        if str(xml.get("root_aux_joint_name", "")) != str(
            self.config.get("roll_joint_name", "")
        ):
            raise ValueError(
                "task_config.roll_joint_name and "
                "xml.root_aux_joint_name must remain identical"
            )
        if _numbers(xml.get("root_aux_joint_axis", ""), 3) != (
            1.0,
            0.0,
            0.0,
        ):
            raise ValueError(
                "hammer_extract_nail roll must use the universal Handle's local "
                "long axis"
            )
        functions = tuple(
            str(entry["function_name"])
            for entry in self.payload.get("functions", ())
        )
        bass = dict(self.payload.get("bass", {}))
        selector = self.config.get(
            "asset_selector",
            self.payload.get("asset_selector", {}),
        )
        if not isinstance(selector, Mapping):
            raise TypeError("asset_selector must be a mapping")
        allowed_heads = tuple(str(value) for value in selector.get("ids", ()))
        mount_face = int(self.payload["root_blocked_face"])

        self._task_spec = TaskSpec(
            name=MISSION_NAME,
            functions=functions,
            scene_path=Path(str(self.config["scene_xml"])),
            horizon=int(self.config["num_steps"]),
            substeps=int(self.config["sub_steps"]),
            objective_weights=self.numerical_task.objective_weights(),
        )
        self._search_spec = SearchSpec(
            allowed_head_asset_ids=allowed_heads,
            max_head_links=int(bass["max_head_links"]),
            function_count=int(self.config["function_count"]),
            function_count_margin=int(
                self.payload.get("function_count_margin", 0)
            ),
            bass=bass,
        )
        self._handle_spec = HandleSpec(
            root_asset_id=str(self.payload["root_asset_id"]),
            mount_face=mount_face,
            open_faces=tuple(face for face in range(6) if face != mount_face),
            position=_numbers(xml.get("root_joint_pos", "0 0 0"), 3),
            orientation=_numbers(
                xml.get("root_asset_quat", "1 0 0 0"),
                4,
            ),
            actuator_type="world_xyz_plus_searched_handle_rotation",
            actuator_bounds=_numbers(
                xml.get("root_motor_ctrl_range", "-5e4 5e4"),
                2,
            ),
            actuator_gains={
                "P": float(xml.get("root_motor_P", 0.0)),
                "D": float(xml.get("root_motor_D", 0.0)),
            },
        )
        self._morphology_spec = MorphologySpec(
            parameterization_id=UNIFIED_PARAMETERIZATION_ID,
            constraints={
                "force_connectivity": bool(
                    self.config.get("force_connectivity", True)
                ),
                "preserve_handle_mount": bool(
                    self.config.get("preserve_handle_mount", True)
                ),
            },
            optimizer=optimizer_contract_from_config(
                self.config,
                default_maxiter=100,
            ),
            collision_policy={
                "enabled": bool(
                    self.config.get("design_collision_check", True)
                ),
                "margin": float(
                    self.config.get("design_collision_margin", 1e-4)
                ),
                "check_ground": bool(
                    self.config.get(
                        "design_collision_check_ground",
                        False,
                    )
                ),
            },
        )
        timeout = self.payload.get(
            "low_level_timeout",
            self.config.get("low_level_timeout"),
        )
        output_dir = Path(
            str(
                self.payload.get(
                    "output_dir",
                    "results/hammer_extract_nail_bilevel",
                )
            )
        )
        cache_dir = Path(
            str(
                self.payload.get(
                    "cache_dir",
                    "tasks/hammer_extract_nail/cache_search",
                )
            )
        )
        replay_dir = Path(
            str(self.payload.get("replay_dir", output_dir / "replay"))
        )
        self._runtime_spec = RuntimeSpec(
            timeout_seconds=None if timeout is None else float(timeout),
            numeric_threads=int(
                self.config.get("low_level_numeric_threads", 1)
            ),
            output_dir=output_dir,
            cache_dir=cache_dir,
            replay_dir=replay_dir,
            seed=int(self.config.get("low_level_seed", 0)),
        )


    def bass_evaluation(
        self,
        *,
        score: float,
        run_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        reward_config = dict(
            self.payload.get("bass", {}).get("task_progress_reward", {})
        )
        reward_config.update(
            dict(self.config.get("task_progress_reward", {}) or {})
        )
        if not reward_config.get("enabled", False):
            return None
        if not math.isfinite(float(score)):
            return {
                "bass_reward": 0.0,
                "task_success": False,
                "task_milestone": 0,
                "task_stage_count": 5,
                "task_progress": 0.0,
                "task_feasible": False,
                "loss_quality": 0.0,
                "raw_process_reward": 0.0,
                "normalized_failure_reward": 0.0,
                "failure_ceiling_enabled": bool(
                    reward_config.get("failure_ceiling_enabled", False)
                ),
                "failure_ceiling": float(
                    reward_config.get("failure_ceiling", 0.45)
                ),
                "final_scalar_reward": 0.0,
                "task_progress_gates": {},
                "task_progress_status": "evaluation_failed",
            }
        diagnostics = dict(run_result.get("diagnostics", {}) or {})
        required_fields = {
            "hammer_completion_depth",
            "maximum_engage_gate_distance",
            "min_extract_target_dist",
            "min_hammer_target_dist",
            "nail_down_depth",
            "nail_down_goal_met",
            "nail_up_lift",
            "target_up_lift",
            "task_success",
            "transfer_approach_distance",
        }
        missing = sorted(required_fields - set(diagnostics))
        terminal = diagnostics.get("terminal")
        if not isinstance(terminal, Mapping):
            missing.append("terminal")
        else:
            missing.extend(
                f"terminal.{field}"
                for field in (
                    "approach_distance",
                    "engagement_distance",
                    "hammer_contact_met",
                )
                if field not in terminal
            )
        if missing:
            raise ValueError(
                "hammer_extract_nail BASS reward requires complete rollout "
                f"diagnostics; missing={missing}"
            )
        return hammer_extract_nail_bass_reward(
            score=score,
            diagnostics=diagnostics,
            config=reward_config,
        )


__all__ = ["TaskDefinition"]
