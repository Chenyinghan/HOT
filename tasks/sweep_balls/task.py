"""Canonical Handle-root sweep_balls task plugin."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from bilevel import HandleSpec, MorphologySpec, RuntimeSpec, SearchSpec, TaskSpec
from bilevel.parameterization import UNIFIED_PARAMETERIZATION_ID
from bilevel.lower.optimizers import optimizer_contract_from_config
from tasks.base import TaskBase

from .objective import MISSION_NAME, TaskDynamics
from .task_progress import sweep_balls_bass_reward


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
    """Native task contract; all numerical semantics remain in ``objective``."""

    config_path = CONFIG_PATH

    def _configure(self) -> None:
        task_kwargs = {
            key: self.config[key]
            for key in (
                "num_steps",
                "sub_steps",
                "coef_sweep_ball_contact",
                "coef_sweep_backtrack",
                "coef_ball_cohesion",
                "coef_ball_goal",
                "coef_swing",
                "coef_control",
                "control_smooth_weight",
                "optimization_prefix_fractions",
                "sweep_axis",
                "sweep_height_offset",
                "sweep_vertical_scale",
                "sweep_contact_scale",
                "sweep_contact_quartic_weight",
                "sweep_contact_lateral_scale",
                "sweep_contact_min_gap",
                "sweep_contact_max_gap",
                "sweep_contact_lateral_slack",
                "sweep_contact_height_slack",
                "target_container_depth",
                "target_container_half_width",
                "ball_radius",
                "success_safety_margin",
                "success_hold_knots",
                "terminal_loss_knots",
                "terminal_goal_weight",
                "sweep_backtrack_scale",
                "ball_goal_lateral_half_width",
                "ball_cohesion_scale",
                "swing_free_deg",
                "swing_scale_deg",
                "terminal_swing_weight",
                "action_scale_x",
                "action_scale_y",
                "action_scale_z",
                "action_scale_swing",
                "swing_joint_name",
                "force_connectivity",
                "generic_design_protocol",
            )
            if key in self.config
        }
        self.numerical_task = TaskDynamics(**task_kwargs)
        # Stage2-DR clones canonical tasks per compiled scenario. Retain the
        # complete declarative config without changing Task's public API.
        self.numerical_task._task_config = dict(self.config)
        self.numerical_task._canonical_task_name = MISSION_NAME
        if (
            str(self.config.get("action_parameterization", ""))
            != self.numerical_task.action_parameterization()
        ):
            raise ValueError(
                "target_container config and numerical task must use the same "
                "action parameterization"
            )

        xml = dict(self.payload.get("xml", {}))
        if str(xml.get("root_joint_type", "")).lower() != "translational":
            raise ValueError(
                "target_container requires one public xyz translational root joint"
            )
        if str(xml.get("root_aux_joint_type", "")).lower() != "revolute":
            raise ValueError(
                "target_container requires one public revolute root auxiliary joint"
            )
        if str(xml.get("root_aux_joint_name", "")) != str(
            self.config.get("swing_joint_name", "")
        ):
            raise ValueError(
                "task_config.swing_joint_name and "
                "xml.root_aux_joint_name must remain identical"
            )
        if _numbers(xml.get("root_aux_joint_axis", ""), 3) != (
            0.0,
            1.0,
            0.0,
        ):
            raise ValueError(
                "target_container forward/backward swing must use world +Y axis"
            )

        functions = tuple(
            str(entry["function_name"])
            for entry in self.payload.get("functions", ())
        )
        if functions != ("sweep",):
            raise ValueError(
                "target_container exposes exactly one public function: sweep"
            )
        if int(self.config.get("function_count", 0)) != len(functions):
            raise ValueError(
                "target_container function_count must match the public functions"
            )
        bass = dict(self.payload.get("bass", {}))
        root_rotation_options = tuple(
            str(value).strip().lower()
            for value in bass.get("root_rotation_options", ())
        )
        if root_rotation_options != ("roll", "pitch", "yaw"):
            raise ValueError(
                "target_container BASS must select exactly one root rotation from "
                "roll, pitch, yaw"
            )
        if str(bass.get("reward_mode", "")) != "bounded_task":
            raise ValueError("target_container BASS must use bounded_task reward")
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
            actuator_type="world_xyz_plus_selected_axis_rotation",
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
                        True,
                    )
                ),
            },
        )
        timeout = self.payload.get("low_level_timeout")
        replay_dir = self.payload.get(
            "replay_dir",
            self.payload["cache_dir"],
        )
        self._runtime_spec = RuntimeSpec(
            timeout_seconds=None if timeout is None else float(timeout),
            numeric_threads=int(
                self.config.get("low_level_numeric_threads", 1)
            ),
            output_dir=Path(str(self.payload["output_dir"])),
            cache_dir=Path(str(self.payload["cache_dir"])),
            replay_dir=Path(str(replay_dir)),
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
        if not reward_config.get("enabled", False):
            return None
        if not math.isfinite(float(score)):
            return {
                "bass_reward": 0.0,
                "task_success": False,
                "task_milestone": 0,
                "task_stage_count": 3,
                "task_progress": 0.0,
                "loss_quality": 0.0,
                "task_progress_gates": {},
                "task_progress_status": "evaluation_failed",
            }
        diagnostics = dict(run_result.get("diagnostics", {}) or {})
        required_fields = {
            "mean_ball_progress_ratio",
            "safe_ball_fraction",
            "success_hold_fraction",
            "task_success",
        }
        missing = sorted(required_fields - set(diagnostics))
        if missing:
            raise ValueError(
                "target_container BASS reward requires complete rollout diagnostics; "
                f"missing={missing}"
            )
        return sweep_balls_bass_reward(
            score=score,
            diagnostics=diagnostics,
            config=reward_config,
        )


__all__ = ["TaskDefinition"]
