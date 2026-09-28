"""Canonical Handle-root torque task plugin."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET

from bilevel import HandleSpec, MorphologySpec, RuntimeSpec, SearchSpec, TaskSpec
from bilevel.parameterization import UNIFIED_PARAMETERIZATION_ID
from bilevel.lower.optimizers import optimizer_contract_from_config
from tasks.base import TaskBase

from .objective import MISSION_NAME, TaskDynamics
from .task_progress import torque_bass_reward


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
    """Single-function Align -> Engage -> Turn torque contract."""

    config_path = CONFIG_PATH

    def _configure(self) -> None:

        task_kwargs = {
            key: self.config[key]
            for key in (
                "num_steps",
                "sub_steps",
                "coef_tool_pose",
                "coef_turn_goal",
                "coef_control",
                "control_smooth_weight",
                "align_end_fraction",
                "engage_end_fraction",
                "pre_engage_distance",
                "radial_pose_scale",
                "axial_pose_scale",
                "engage_ramp_fraction",
                "nail_target_deg",
                "turn_handle_weight",
                "turn_sync_weight",
                "turn_terminal_weight",
                "turn_curriculum_segments",
                "success_nail_deg",
                "align_tolerance",
                "engage_tolerance",
                "success_pose_tolerance",
                "geometry_audit_enabled",
                "geometry_audit_containment_fraction",
                "geometry_audit_normalized_depth",
                "geometry_audit_min_sustained_seconds",
                "geometry_audit_operated_body_names",
                "curriculum_stage_weights",
                "stage_stop_motion",
                "stage_future_loss_mode",
                "action_scale_x",
                "action_scale_y",
                "action_scale_z",
                "action_scale_roll",
                "roll_joint_name",
                "force_connectivity",
                "generic_design_protocol",
            )
            if key in self.config
        }
        self.numerical_task = TaskDynamics(**task_kwargs)
        self.numerical_task._task_config = dict(self.config)
        self.numerical_task._canonical_task_name = MISSION_NAME

        xml = dict(self.payload.get("xml", {}))
        if str(xml.get("root_joint_type", "")).lower() != "translational":
            raise ValueError(
                "torque requires one public xyz translational root joint"
            )
        if str(xml.get("root_aux_joint_type", "")).lower() != "revolute":
            raise ValueError(
                "torque requires one public revolute root auxiliary joint"
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
                "torque roll must use the screw and Handle long axis +X"
            )

        functions = tuple(
            str(entry["function_name"])
            for entry in self.payload.get("functions", ())
        )
        if functions != ("torque",):
            raise ValueError(
                "torque exposes exactly one public function: torque"
            )
        if int(self.config.get("function_count", 0)) != len(functions):
            raise ValueError(
                "torque function_count must match the public functions"
            )

        bass = dict(self.payload.get("bass", {}))
        root_rotation_options = tuple(
            str(value).strip().lower()
            for value in bass.get("root_rotation_options", ())
        )
        if root_rotation_options != ("roll", "pitch", "yaw"):
            raise ValueError(
                "torque BASS must select root rotation from roll, pitch, yaw"
            )
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
                default_maxiter=80,
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
        timeout = self.payload.get(
            "low_level_timeout",
            self.config.get("low_level_timeout"),
        )
        output_dir = Path(
            str(self.payload.get("output_dir", "results/torque_bilevel"))
        )
        cache_dir = Path(
            str(self.payload.get("cache_dir", "tasks/torque_bolt/cache_search"))
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


    def validate_candidate(self, candidate: Any) -> dict:
        path = getattr(candidate, "model_path", candidate)
        try:
            root = ET.parse(Path(path)).getroot()
        except Exception as exc:
            return {
                "ok": False,
                "errors": [f"cannot parse candidate XML: {exc}"],
            }

        marker = root.find(".//link[@name='torque_endeffector']")
        errors = []
        if marker is None:
            errors.append("missing torque_endeffector")
        else:
            group_root = str(
                marker.attrib.get("function_group_root", "")
            ).strip()
            leaves_text = str(
                marker.attrib.get("function_group_leaves", "")
            ).strip()
            leaves = tuple(
                value.strip()
                for value in leaves_text.split(",")
                if value.strip()
            )
            if not group_root:
                errors.append(
                    "torque_endeffector has no function_group_root"
                )
            if not leaves:
                errors.append(
                    "torque_endeffector has no terminal leaves"
                )
            if len(set(leaves)) != len(leaves):
                errors.append(
                    "torque_endeffector has duplicate terminal leaves"
                )
            declared = marker.attrib.get("function_group_leaf_count")
            if declared is not None:
                try:
                    declared_count = int(declared)
                except ValueError:
                    errors.append(
                        "torque_endeffector has invalid leaf count"
                    )
                else:
                    if declared_count != len(leaves):
                        errors.append(
                            "torque_endeffector leaf metadata disagrees"
                        )
        return {"ok": not errors, "errors": errors}


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
                "task_stage_count": 3,
                "task_progress": 0.0,
                "task_feasible": False,
                "loss_quality": 0.0,
                "task_progress_gates": {},
                "task_progress_status": "evaluation_failed",
            }
        diagnostics = dict(run_result.get("diagnostics", {}) or {})
        required_fields = {
            "align_distance",
            "align_tolerance",
            "engage_distance",
            "engage_tolerance",
            "final_pose_distance",
            "nail_turn_deg",
            "success_nail_deg",
            "success_pose_tolerance",
            "task_success",
        }
        missing = sorted(required_fields - set(diagnostics))
        if missing:
            raise ValueError(
                "torque BASS reward requires complete rollout diagnostics; "
                f"missing={missing}"
            )
        return torque_bass_reward(
            score=score,
            diagnostics=diagnostics,
            config=reward_config,
        )


__all__ = ["TaskDefinition"]
