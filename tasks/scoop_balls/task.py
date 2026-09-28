"""Canonical searched Handle-root Scoop task plugin."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET

import numpy as np

from bilevel import HandleSpec, MorphologySpec, RuntimeSpec, SearchSpec, TaskSpec
from bilevel.parameterization import UNIFIED_PARAMETERIZATION_ID
from bilevel.lower.optimizers import optimizer_contract_from_config
from tasks.base import TaskBase

from .objective import MISSION_NAME, TaskDynamics
from .task_progress import scoop_bass_reward


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / "config.json"


def _numbers(text: str, expected: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in str(text).split())
    if len(values) != expected:
        raise ValueError(f"expected {expected} numeric values, got {text!r}")
    return values


class TaskDefinition(TaskBase):
    """Canonical Scoop interface for searched connected Heads."""

    config_path = CONFIG_PATH

    def _configure(self) -> None:
        self.numerical_task = TaskDynamics(
            num_steps=self.config["num_steps"],
            sub_steps=self.config["sub_steps"],
            task_config=self.config,
        )
        self.numerical_task._canonical_task_name = MISSION_NAME

        xml = dict(self.payload.get("xml", {}))
        if str(xml.get("root_joint_type", "")).lower() != "translational":
            raise ValueError(
                "scoop requires one public xyz translational root joint"
            )
        if str(xml.get("root_aux_joint_type", "")).lower() != "revolute":
            raise ValueError(
                "scoop requires one public revolute root auxiliary joint"
            )
        if str(xml.get("root_aux_joint_name", "")) != (
            "freeform_pitch_joint"
        ):
            raise ValueError(
                "the Scoop reference must retain freeform_pitch_joint"
            )
        if _numbers(xml.get("root_aux_joint_axis", ""), 3) != (
            0.0,
            1.0,
            0.0,
        ):
            raise ValueError(
                "the Scoop reference pitch joint must use the +Y axis"
            )
        if tuple(float(value) for value in self.config["root_joint_pos"]) != (
            _numbers(xml["root_joint_pos"], 3)
        ):
            raise ValueError(
                "task_config.root_joint_pos and xml.root_joint_pos "
                "must remain identical"
            )
        functions = tuple(
            str(entry["function_name"])
            for entry in self.payload.get("functions", ())
        )
        reference_head = dict(self.payload.get("reference_head", {}))
        bass = dict(self.payload.get("bass", {}))
        root_rotation_options = tuple(
            str(value).strip().lower()
            for value in bass.get("root_rotation_options", ())
        )
        if root_rotation_options != ("roll", "pitch", "yaw"):
            raise ValueError(
                "Scoop BASS must select root rotation from roll, pitch, yaw"
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
            position=_numbers(xml["root_joint_pos"], 3),
            orientation=_numbers(xml["root_asset_quat"], 4),
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
                "fixed_topology": False,
                "fixed_handle": True,
                "max_head_links": int(bass["max_head_links"]),
                "reference_head_links": int(reference_head["link_count"]),
                "preserve_handle_mount": bool(
                    self.config.get("preserve_handle_mount", True)
                ),
            },
            optimizer=optimizer_contract_from_config(
                self.config,
                default_maxiter=20,
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
        self._runtime_spec = RuntimeSpec(
            timeout_seconds=None if timeout is None else float(timeout),
            numeric_threads=int(
                self.config.get("low_level_numeric_threads", 1)
            ),
            output_dir=Path(str(self.payload["output_dir"])),
            cache_dir=Path(str(self.payload["cache_dir"])),
            replay_dir=Path(str(self.payload["replay_dir"])),
            seed=int(self.config.get("low_level_seed", 0)),
        )


    def seed_action(self, context: Any) -> Any:
        # Scoop is intentionally seed-free: every caller receives the same
        # neutral sequence and the staged objective must learn all motion.
        return np.zeros(
            int(context.ndof_u) * int(context.num_ctrl_steps),
            dtype=np.float64,
        )


    def validate_candidate(self, candidate: Any) -> dict:
        path = Path(getattr(candidate, "model_path", candidate))
        try:
            root = ET.parse(path).getroot()
        except Exception as exc:
            return {
                "ok": False,
                "errors": [f"cannot parse candidate XML: {exc}"],
            }
        handles = [
            link
            for link in root.findall(".//link")
            if (
                link.attrib.get("asset_role") == "fixed_root"
                and link.attrib.get("asset_id")
                == self.handle_spec().root_asset_id
            )
        ]
        allowed_head_ids = set(self.search_spec().allowed_head_asset_ids)
        all_heads = [
            link
            for link in root.findall(".//link")
            if link.attrib.get("asset_role") == "head"
        ]
        heads = [
            link
            for link in all_heads
            if link.attrib.get("asset_id") in allowed_head_ids
        ]
        errors = []
        if len(handles) != 1:
            errors.append(
                "searched Scoop requires exactly one fixed Handle declared "
                "by HandleSpec"
            )
        invalid_head_ids = sorted(
            {
                str(link.attrib.get("asset_id", ""))
                for link in all_heads
                if link.attrib.get("asset_id") not in allowed_head_ids
            }
        )
        if invalid_head_ids:
            errors.append(
                "searched Scoop contains Head assets outside its SearchSpec: "
                + ",".join(invalid_head_ids)
            )
        if not heads:
            errors.append("searched Scoop requires at least one legal Head")
        if len(heads) > self.search_spec().max_head_links:
            errors.append(
                "searched Scoop exceeds max_head_links: "
                f"{len(heads)} > {self.search_spec().max_head_links}"
            )
        mounted_heads = [
            link
            for link in heads
            if (
                link.attrib.get("welded_interface") == "true"
                and link.attrib.get("welded_parent_kind") == "handle"
            )
        ]
        if len(mounted_heads) != 1:
            errors.append(
                "searched Scoop requires exactly one Head welded to the Handle"
            )
        markers = root.findall(".//link[@name='scoop_endeffector']")
        if len(markers) != 1:
            errors.append(
                "searched Scoop requires exactly one scoop_endeffector, "
                f"found {len(markers)}"
            )
        else:
            marker = markers[0]
            leaves = tuple(
                value.strip()
                for value in marker.attrib.get(
                    "function_group_leaves", ""
                ).split(",")
                if value.strip()
            )
            head_ids = {
                str(head.attrib.get("node_id", "")) for head in heads
            }
            if not leaves:
                errors.append(
                    "scoop_endeffector has no function-group leaves"
                )
            if len(set(leaves)) != len(leaves):
                errors.append(
                    "scoop_endeffector has duplicate function-group leaves"
                )
            missing = [leaf for leaf in leaves if leaf not in head_ids]
            if missing:
                errors.append(
                    "scoop_endeffector references missing tool leaves: "
                    + ",".join(missing)
                )
            declared = marker.attrib.get("function_group_leaf_count")
            try:
                declared_count = int(declared) if declared is not None else None
            except ValueError:
                errors.append(
                    "scoop_endeffector has invalid function_group_leaf_count"
                )
            else:
                if declared_count is not None and declared_count != len(leaves):
                    errors.append(
                        "scoop_endeffector leaf metadata disagrees"
                    )
        variable_markers = root.findall(
            "./variable/endeffector[@joint='scoop_endeffector']"
        )
        if len(variable_markers) != 1:
            errors.append(
                "searched Scoop requires exactly one registered "
                "scoop_endeffector variable, "
                f"found {len(variable_markers)}"
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
                "raw_terminal_task_success": False,
                "task_milestone": 0,
                "task_stage_count": 3,
                "task_progress": 0.0,
                "task_feasible": False,
                "loss_quality": 0.0,
                "task_progress_gates": {},
                "task_progress_status": "evaluation_failed",
            }
        diagnostics = dict(run_result.get("diagnostics", {}) or {})
        missing = sorted(
            {
                "required_ball_count",
                "stage_gates",
                "lift_carry_success",
                "task_success",
            }
            - set(diagnostics)
        )
        if missing:
            raise ValueError(
                "scoop BASS reward requires complete rollout diagnostics; "
                f"missing={missing}"
            )
        return scoop_bass_reward(
            score=score,
            diagnostics=diagnostics,
            config=reward_config,
        )


__all__ = ["TaskDefinition"]
