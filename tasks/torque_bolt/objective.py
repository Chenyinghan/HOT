from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from bilevel.lower.geometry_audit import SymmetricOverlapAudit
from bilevel.runner import BaseTask, print_info as _print_info
from tasks.objective import TaskObjective


MISSION_NAME = "torque_bolt"


class TaskDynamics(BaseTask):
    """Three-stage, single-function torque objective.

    The searched Head exposes one ``torque`` function marker.  Its number
    of terminal leaves is deliberately unconstrained.  The action space is
    world xyz translation plus one BASS-selected Handle-centred rotation axis.
    No contact metric appears in the objective: physical contact remains part
    of the RedMax scene and successful screw rotation is the coupling proof.
    """

    has_finger_design = False
    optimize_design = False

    def __init__(
        self,
        *,
        num_steps=1200,
        sub_steps=30,
        coef_tool_pose=20.0,
        coef_turn_goal=100.0,
        coef_control=0.05,
        control_smooth_weight=0.25,
        align_end_fraction=0.30,
        engage_end_fraction=0.55,
        pre_engage_distance=1.3,
        radial_pose_scale=0.20,
        axial_pose_scale=1.3,
        engage_ramp_fraction=0.60,
        nail_target_deg=120.0,
        turn_handle_weight=1.0,
        turn_sync_weight=0.25,
        turn_terminal_weight=3.0,
        turn_curriculum_segments=3,
        success_nail_deg=110.0,
        align_tolerance=0.20,
        engage_tolerance=0.20,
        success_pose_tolerance=0.30,
        geometry_audit_enabled=True,
        geometry_audit_containment_fraction=0.2,
        geometry_audit_normalized_depth=1.0,
        geometry_audit_min_sustained_seconds=0.2,
        geometry_audit_operated_body_names=("nail_head", "nail_rod_body"),
        curriculum_stage_weights=(0.5, 3.0, 6.5),
        stage_stop_motion=True,
        stage_future_loss_mode="truncate",
        action_scale_x=5.0,
        action_scale_y=5.0,
        action_scale_z=5.0,
        action_scale_roll=3.0,
        roll_joint_name="freeform_roll_joint",
        force_connectivity=True,
        generic_design_protocol="connected_direct_planar_hexahedron",
    ):
        self._num_steps = int(num_steps)
        self._sub_steps = int(sub_steps)
        if self._num_steps <= 0 or self._sub_steps <= 0:
            raise ValueError("torque num_steps and sub_steps must be positive")

        self._coef = {
            "tool_pose": float(coef_tool_pose),
            "turn_goal": float(coef_turn_goal),
            "control": float(coef_control),
        }
        if any(
            not np.isfinite(value) or value < 0.0
            for value in self._coef.values()
        ):
            raise ValueError(
                "torque objective coefficients must be finite and nonnegative"
            )

        self._control_smooth_weight = float(control_smooth_weight)
        if (
            not np.isfinite(self._control_smooth_weight)
            or self._control_smooth_weight < 0.0
        ):
            raise ValueError(
                "torque control_smooth_weight must be finite and nonnegative"
            )

        self._phase_fractions = (
            float(align_end_fraction),
            float(engage_end_fraction),
        )
        if not (
            0.0
            < self._phase_fractions[0]
            < self._phase_fractions[1]
            < 1.0
        ):
            raise ValueError(
                "torque phase fractions must satisfy "
                "0 < align_end < engage_end < 1"
            )

        self._pre_engage_distance = float(pre_engage_distance)
        self._radial_pose_scale = float(radial_pose_scale)
        self._axial_pose_scale = float(axial_pose_scale)
        self._engage_ramp_fraction = float(engage_ramp_fraction)
        if (
            not np.isfinite(self._pre_engage_distance)
            or self._pre_engage_distance <= 0.0
            or not np.isfinite(self._radial_pose_scale)
            or self._radial_pose_scale <= 0.0
            or not np.isfinite(self._axial_pose_scale)
            or self._axial_pose_scale <= 0.0
            or not 0.0 < self._engage_ramp_fraction <= 1.0
        ):
            raise ValueError(
                "torque pre-engage distance and pose scales must be "
                "positive, and engage_ramp_fraction must be in (0, 1]"
            )

        self._nail_target = np.radians(float(nail_target_deg))
        self._success_nail = np.radians(float(success_nail_deg))
        if not 0.0 < self._success_nail <= self._nail_target < 2.0 * np.pi:
            raise ValueError(
                "torque angles must satisfy "
                "0 < success_nail_deg <= nail_target_deg < 360"
            )

        self._turn_handle_weight = float(turn_handle_weight)
        self._turn_sync_weight = float(turn_sync_weight)
        self._turn_terminal_weight = float(turn_terminal_weight)
        turn_weights = np.asarray(
            [
                self._turn_handle_weight,
                self._turn_sync_weight,
                self._turn_terminal_weight,
            ],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(turn_weights))
            or np.any(turn_weights < 0.0)
        ):
            raise ValueError(
                "torque turn weights must be finite and nonnegative"
            )

        try:
            turn_curriculum_segments_value = float(
                turn_curriculum_segments
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "torque turn_curriculum_segments must be a positive integer"
            ) from exc
        if (
            not np.isfinite(turn_curriculum_segments_value)
            or turn_curriculum_segments_value <= 0.0
            or not turn_curriculum_segments_value.is_integer()
        ):
            raise ValueError(
                "torque turn_curriculum_segments must be a positive integer"
            )
        self._configured_turn_curriculum_segments = int(
            turn_curriculum_segments_value
        )
        default_ctrl_steps = self._num_steps // self._sub_steps
        _, default_engage_end = self._phase_boundaries(default_ctrl_steps)
        available_turn_knots = default_ctrl_steps - default_engage_end
        if available_turn_knots <= 0:
            raise ValueError(
                "torque rollout must contain at least one Turn control knot"
            )
        self._turn_curriculum_segments = min(
            self._configured_turn_curriculum_segments,
            available_turn_knots,
        )

        self._align_tolerance = float(align_tolerance)
        self._engage_tolerance = float(engage_tolerance)
        self._success_pose_tolerance = float(success_pose_tolerance)
        pose_tolerances = np.asarray(
            [
                self._align_tolerance,
                self._engage_tolerance,
                self._success_pose_tolerance,
            ],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(pose_tolerances))
            or np.any(pose_tolerances <= 0.0)
        ):
            raise ValueError(
                "torque pose tolerances must be finite and positive"
            )

        self._geometry_audit_enabled = bool(geometry_audit_enabled)
        self._geometry_audit_containment_fraction = float(
            geometry_audit_containment_fraction
        )
        self._geometry_audit_normalized_depth = float(
            geometry_audit_normalized_depth
        )
        self._geometry_audit_min_sustained_seconds = float(
            geometry_audit_min_sustained_seconds
        )
        self._geometry_audit_operated_body_names = tuple(
            str(name).strip()
            for name in geometry_audit_operated_body_names
            if str(name).strip()
        )
        if not 0.0 < self._geometry_audit_containment_fraction <= 1.0:
            raise ValueError(
                "geometry_audit_containment_fraction must lie in (0, 1]"
            )
        if self._geometry_audit_normalized_depth <= 0.0:
            raise ValueError(
                "geometry_audit_normalized_depth must be positive"
            )
        if self._geometry_audit_min_sustained_seconds <= 0.0:
            raise ValueError(
                "geometry_audit_min_sustained_seconds must be positive"
            )
        if not self._geometry_audit_operated_body_names:
            raise ValueError(
                "geometry_audit_operated_body_names must be nonempty"
            )

        self._turn_optimization_stages = tuple(
            f"turn_{index}"
            for index in range(1, self._turn_curriculum_segments + 1)
        )
        self._optimization_stages = (
            "align",
            "engage",
            *self._turn_optimization_stages,
        )
        stage_weights = np.asarray(
            curriculum_stage_weights,
            dtype=np.float64,
        ).reshape(-1)
        if (
            stage_weights.shape != (3,)
            or not np.all(np.isfinite(stage_weights))
            or np.any(stage_weights <= 0.0)
        ):
            raise ValueError(
                "torque curriculum_stage_weights must contain three "
                "positive finite physical-phase values"
            )
        self._curriculum_stage_weights = np.concatenate(
            [
                stage_weights[:2],
                np.full(
                    self._turn_curriculum_segments,
                    stage_weights[2]
                    / float(self._turn_curriculum_segments),
                    dtype=np.float64,
                ),
            ]
        )
        self._optimization_stage = "full"
        self._stage_stop_motion = bool(stage_stop_motion)
        self._stage_future_loss_mode = str(
            stage_future_loss_mode
        ).strip().lower()
        if self._stage_future_loss_mode not in ("truncate", "full"):
            raise ValueError(
                "torque stage_future_loss_mode must be 'truncate' or 'full'"
            )

        self._canonical_action_scale = np.asarray(
            [
                action_scale_x,
                action_scale_y,
                action_scale_z,
                action_scale_roll,
            ],
            dtype=np.float64,
        )
        if (
            self._canonical_action_scale.shape != (4,)
            or not np.all(np.isfinite(self._canonical_action_scale))
            or np.any(self._canonical_action_scale <= 0.0)
        ):
            raise ValueError(
                "torque four-DOF action scales must be finite and positive"
            )

        self._roll_joint_name = str(roll_joint_name).strip()
        if not self._roll_joint_name:
            raise ValueError("torque roll_joint_name must be nonempty")

        self._force_connectivity = bool(force_connectivity)
        self._generic_design_protocol = str(generic_design_protocol)
        self._design_bundle = None

        self._var_nail_center = 0
        self._var_nail_axis = 3
        self._var_tool = 6
        self._required_var_len = 9

        self._q_nail = None
        self._q_root_translation = None
        self._q_root_roll = None
        self._root_rotation_mode = None

        self._reset_rollout_cache()

    def num_steps(self) -> int:
        return self._num_steps

    def sub_steps(self) -> int:
        return self._sub_steps

    def objective_weights(self) -> dict:
        return dict(self._coef)

    def action_scale(self, ndof_u: int) -> np.ndarray:
        if int(ndof_u) != 4:
            raise ValueError(
                "torque requires xyz translation plus one selected-axis "
                f"rotation control (4 controls), got {ndof_u}"
            )
        return self._canonical_action_scale.copy()

    def init_action(
        self,
        ndof_u: int,
        num_ctrl_steps: int,
        seed: int,
    ) -> np.ndarray:
        _ = seed
        self.action_scale(ndof_u)
        return np.zeros(
            int(ndof_u) * int(num_ctrl_steps),
            dtype=np.float64,
        )

    def init_task(self, sim) -> None:
        if int(sim.ndof_u) != 4:
            raise ValueError(
                "torque requires xyz translation plus one selected-axis "
                f"rotation motor (4 controls), got {sim.ndof_u}"
            )

    @staticmethod
    def _joint_dof(joint_type: str) -> int:
        return {
            "fixed": 0,
            "revolute": 1,
            "prismatic": 1,
            "planar": 2,
            "translational": 3,
            "spherical": 3,
            "spherical-euler": 3,
            "spherical-exp": 3,
            "free2d": 3,
            "free3d": 6,
            "free3d-euler": 6,
            "free3d-exp": 6,
        }.get(str(joint_type).lower(), 0)

    def _configure_q_layout(self, root: ET.Element) -> None:
        q_cursor = 0
        nail_idx = None
        root_translation = None
        root_roll = None
        root_rotation_mode = None
        rotation_joint_modes = {
            self._roll_joint_name: "roll",
            "freeform_roll_joint": "roll",
            "freeform_pitch_joint": "pitch",
            "freeform_yaw_joint": "yaw",
        }
        canonical_axes = {
            "roll": np.asarray([1.0, 0.0, 0.0]),
            "pitch": np.asarray([0.0, 1.0, 0.0]),
            "yaw": np.asarray([0.0, 0.0, 1.0]),
        }
        for joint in root.iter("joint"):
            name = joint.attrib.get("name", "")
            joint_type = joint.attrib.get("type", "").lower()
            dof = self._joint_dof(joint_type)
            if name == "nail_joint":
                if joint_type != "revolute" or dof != 1:
                    raise ValueError(
                        "torque nail_joint must be one-DOF revolute"
                    )
                nail_idx = q_cursor
            elif name == "freeform_root_joint":
                if joint_type != "translational" or dof != 3:
                    raise ValueError(
                        "torque freeform_root_joint must be translational"
                    )
                root_translation = slice(q_cursor, q_cursor + 3)
            elif name in rotation_joint_modes:
                if joint_type != "revolute" or dof != 1:
                    raise ValueError(
                        f"torque selected rotation joint {name} must be revolute"
                    )
                axis = np.fromstring(
                    joint.attrib.get("axis", ""),
                    sep=" ",
                    dtype=np.float64,
                )
                if (
                    axis.shape != (3,)
                    or float(np.linalg.norm(axis)) <= 1e-12
                ):
                    raise ValueError(
                        f"torque selected rotation joint {name} has invalid axis"
                    )
                axis /= float(np.linalg.norm(axis))
                expected_mode = rotation_joint_modes[name]
                expected_axis = canonical_axes.get(expected_mode)
                if expected_axis is None or not np.allclose(
                    axis, expected_axis, rtol=0.0, atol=1e-9
                ):
                    raise ValueError(
                        "torque selected rotation joint name and axis disagree"
                    )
                root_roll = q_cursor
                root_rotation_mode = expected_mode
            q_cursor += dof

        if nail_idx is None or root_translation is None or root_roll is None:
            raise ValueError(
                "torque XML must contain nail_joint, translational "
                "freeform_root_joint, and one selected rotation joint"
            )
        self._q_nail = int(nail_idx)
        self._q_root_translation = root_translation
        self._q_root_roll = int(root_roll)
        self._root_rotation_mode = str(root_rotation_mode)

    def _configure_variable_layout(self, model_path: str) -> None:
        root = ET.parse(model_path).getroot()
        self._configure_q_layout(root)
        variable = root.find("variable")
        if variable is None:
            raise ValueError("torque XML must contain a variable section")

        nail_center = None
        nail_axis = None
        tool = None
        entries = list(variable.findall("endeffector"))
        for idx, elem in enumerate(entries):
            joint = elem.attrib.get("joint", "")
            pos = np.fromstring(
                elem.attrib.get("pos", "0 0 0"),
                sep=" ",
                dtype=np.float64,
            )
            if pos.shape != (3,):
                raise ValueError(
                    f"torque variable entry {idx} has invalid position"
                )
            if joint == "nail_joint":
                if float(np.linalg.norm(pos)) <= 1e-12:
                    nail_center = idx
                else:
                    nail_axis = idx
            elif joint == "torque_endeffector":
                tool = idx

        if None in (nail_center, nail_axis, tool):
            raise ValueError(
                "torque XML variables must expose nail center, nail axis "
                "point, and torque_endeffector"
            )

        self._var_nail_center = 3 * int(nail_center)
        self._var_nail_axis = 3 * int(nail_axis)
        self._var_tool = 3 * int(tool)
        self._required_var_len = 3 * (
            max(int(nail_center), int(nail_axis), int(tool)) + 1
        )

    def configure_model(self, model_path: str, sim) -> None:
        _ = sim
        self._configure_variable_layout(model_path)

    def init_design(self, model_path: str, sim):
        from bilevel.parameterization import build_design_bundle

        self._configure_variable_layout(model_path)
        bundle = build_design_bundle(
            model_path,
            sim,
            {
                "optimize_finger_design": False,
                "force_connectivity": self._force_connectivity,
                "generic_design_protocol": self._generic_design_protocol,
            },
        )
        self._design_bundle = bundle
        bundle.apply(
            sim,
            bundle.init_cage_params,
            generate_mesh=False,
        )
        return bundle

    def bounds(
        self,
        ndof_u,
        num_ctrl_steps,
        ndof_cage,
        optimize_design,
    ):
        bounds = [(-1.0, 1.0)] * (
            int(ndof_u) * int(num_ctrl_steps)
        )
        if optimize_design:
            from bilevel.parameterization import cage_bounds_for_bundle

            bounds += cage_bounds_for_bundle(
                self._design_bundle,
                ndof_cage,
                optimize_finger_design=False,
            )
        return bounds

    def _phase_boundaries(self, num_ctrl_steps: int) -> tuple[int, int]:
        count = max(1, int(num_ctrl_steps))
        if count < 3:
            return 1, min(2, count)
        align_end = int(
            np.clip(
                round(count * self._phase_fractions[0]),
                1,
                count - 2,
            )
        )
        engage_end = int(
            np.clip(
                round(count * self._phase_fractions[1]),
                align_end + 1,
                count - 1,
            )
        )
        return align_end, engage_end

    def set_optimization_stage(self, stage) -> None:
        normalized = str(stage).strip().lower()
        valid = set(self._optimization_stages) | {"full"}
        if normalized not in valid:
            raise ValueError(
                f"unsupported torque optimization stage {stage!r}; "
                f"expected one of {sorted(valid)}"
            )
        self._optimization_stage = normalized

    def _is_turn_optimization_stage(self, stage=None) -> bool:
        normalized = str(
            self._optimization_stage if stage is None else stage
        ).strip().lower()
        return normalized in self._turn_optimization_stages

    def _turn_segment_windows(
        self,
        num_ctrl_steps: int,
    ) -> tuple[tuple[int, int], ...]:
        _, engage_end = self._phase_boundaries(num_ctrl_steps)
        turn_steps = int(num_ctrl_steps) - engage_end
        if turn_steps < self._turn_curriculum_segments:
            raise ValueError(
                "torque rollout has fewer Turn knots than curriculum segments"
            )
        base, remainder = divmod(
            turn_steps,
            self._turn_curriculum_segments,
        )
        windows = []
        start = int(engage_end)
        for index in range(self._turn_curriculum_segments):
            width = base + (1 if index < remainder else 0)
            end = start + width
            windows.append((start, end))
            start = end
        return tuple(windows)

    def _turn_stage_index(self, stage) -> int:
        normalized = str(stage).strip().lower()
        if normalized not in self._turn_optimization_stages:
            raise ValueError(
                f"unsupported torque Turn curriculum stage {stage!r}"
            )
        return self._turn_optimization_stages.index(normalized) + 1

    def optimization_stage_stop_motion(self):
        """Enable a static tool replay after a stage fails acceptance."""

        return self._stage_stop_motion

    def optimization_stage_future_loss_mode(self):
        """Return how post-failure losses contribute to final scoring."""

        return self._stage_future_loss_mode

    def optimization_stage_motion_dofs(self):
        """Return controlled tool coordinates frozen after stage failure."""

        indices = []
        if self._q_root_translation is not None:
            indices.extend(
                range(
                    int(self._q_root_translation.start),
                    int(self._q_root_translation.stop),
                )
            )
        if self._q_root_roll is not None:
            indices.append(int(self._q_root_roll))
        if not indices:
            raise RuntimeError(
                "torque could not resolve controlled tool motion DOFs"
            )
        return tuple(indices)

    def loss_stage_for_step(self, i, num_ctrl_steps):
        """Label each loss knot for per-stage optimization diagnostics."""

        align_end, engage_end = self._phase_boundaries(num_ctrl_steps)
        if int(i) < align_end:
            return "align"
        if int(i) < engage_end:
            return "engage"
        return "turn"

    def optimization_stage_schedule(self, maxiter):
        budget = max(0, int(maxiter))
        if budget == 0:
            return ()
        allocations = np.zeros(
            len(self._optimization_stages),
            dtype=np.int64,
        )
        if budget < len(allocations):
            allocations[:budget] = 1
        else:
            allocations[:] = 1
            remaining = budget - len(allocations)
            exact = (
                self._curriculum_stage_weights
                / float(np.sum(self._curriculum_stage_weights))
                * remaining
            )
            additions = np.floor(exact).astype(np.int64)
            allocations += additions
            remainder = remaining - int(np.sum(additions))
            if remainder:
                order = np.argsort(
                    -(exact - additions),
                    kind="stable",
                )
                allocations[order[:remainder]] += 1
        return tuple(
            (stage, int(stage_budget))
            for stage, stage_budget in zip(
                self._optimization_stages,
                allocations,
            )
            if stage_budget > 0
        )

    def optimization_stage_action_window(
        self,
        stage,
        num_ctrl_steps,
    ):
        normalized = str(stage).strip().lower()
        align_end, engage_end = self._phase_boundaries(num_ctrl_steps)
        windows = {
            "align": (0, align_end),
            "engage": (align_end, engage_end),
        }
        windows.update(
            dict(
                zip(
                    self._turn_optimization_stages,
                    self._turn_segment_windows(num_ctrl_steps),
                )
            )
        )
        if normalized not in windows:
            raise ValueError(
                f"unsupported torque optimization stage {stage!r}"
            )
        return windows[normalized]

    def _optimization_stage_end(self, num_ctrl_steps: int) -> int:
        if self._optimization_stage == "full":
            return int(num_ctrl_steps)
        return int(
            self.optimization_stage_action_window(
                self._optimization_stage,
                num_ctrl_steps,
            )[1]
        )

    def _objective_active(self, i: int, num_ctrl_steps: int) -> bool:
        return int(i) < self._optimization_stage_end(num_ctrl_steps)

    def _read_task_points(self, variables):
        values = np.asarray(variables, dtype=np.float64)
        if len(values) < self._required_var_len:
            raise ValueError(
                "torque expects nail center, nail axis point, and one "
                f"function marker ({self._required_var_len} values); "
                f"got {len(values)}"
            )
        return (
            values[
                self._var_nail_center :
                self._var_nail_center + 3
            ],
            values[
                self._var_nail_axis :
                self._var_nail_axis + 3
            ],
            values[self._var_tool : self._var_tool + 3],
        )

    def _geometry(self, variables):
        p_nail, p_axis, p_tool = self._read_task_points(variables)
        axis_vector = p_axis - p_nail
        axis_length = float(np.linalg.norm(axis_vector))
        if axis_length <= 1e-12:
            raise ValueError("torque nail axis marker is degenerate")
        axis = axis_vector / axis_length
        return p_nail, p_axis, p_tool, axis, axis_length

    def _pose_target(
        self,
        i: int,
        num_ctrl_steps: int,
    ) -> tuple[float, float]:
        align_end, engage_end = self._phase_boundaries(num_ctrl_steps)
        if i == align_end - 1:
            return self._pre_engage_distance, float(align_end)
        if align_end <= i < engage_end:
            engage_steps = engage_end - align_end
            ramp_steps = max(
                1,
                int(round(
                    engage_steps * self._engage_ramp_fraction
                )),
            )
            progress = min(
                1.0,
                float(i - align_end + 1) / float(ramp_steps),
            )
            return (
                self._pre_engage_distance * (1.0 - progress),
                1.0,
            )
        if i >= engage_end:
            return 0.0, 1.0
        return 0.0, 0.0

    def _pose_components(
        self,
        variables,
        offset: float,
    ):
        p_nail, p_axis, p_tool, axis, axis_length = (
            self._geometry(variables)
        )
        displacement = p_tool - p_nail
        axial_position = float(np.dot(displacement, axis))
        axial_error = axial_position + float(offset)
        radial_error = displacement - axial_position * axis
        residual = radial_error + axial_error * axis
        return (
            p_nail,
            p_axis,
            p_tool,
            axis,
            axis_length,
            displacement,
            axial_position,
            axial_error,
            radial_error,
            residual,
        )

    def _reset_rollout_cache(self) -> None:
        self._prev_control = None
        self._prev_control_by_step = {}
        self._engage_nail_angle = None
        self._engage_handle_angle = None
        self._align_distance = None
        self._align_axial_error = None
        self._align_radial_distance = None
        self._engage_distance = None
        self._engage_axial_error = None
        self._engage_radial_distance = None
        self._final_pose_distance = None
        self._final_axial_error = None
        self._final_radial_distance = None
        self._terminal_cache = {}
        self._stage_terminal_cache = {}

    def _turn_components(self, q, target_angle):
        if (
            self._engage_nail_angle is None
            or self._engage_handle_angle is None
        ):
            raise RuntimeError(
                "torque turn baseline was not captured at Engage boundary"
            )
        nail_delta = (
            float(q[self._q_nail])
            - float(self._engage_nail_angle)
        )
        handle_delta = (
            float(q[self._q_root_roll])
            - float(self._engage_handle_angle)
        )
        nail_error = nail_delta - float(target_angle)
        handle_error = handle_delta - float(target_angle)
        sync_error = handle_delta - nail_delta
        return (
            nail_delta,
            handle_delta,
            nail_error,
            handle_error,
            sync_error,
        )

    def compute_terms(
        self,
        i,
        num_ctrl_steps,
        u_i,
        variables,
        q,
    ):
        if i == 0:
            self._reset_rollout_cache()

        align_end, engage_end = self._phase_boundaries(
            num_ctrl_steps
        )
        objective_active = self._objective_active(
            i,
            num_ctrl_steps,
        )
        offset, pose_weight = self._pose_target(
            i,
            num_ctrl_steps,
        )
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            axial_error,
            radial_error,
            residual,
        ) = self._pose_components(variables, offset)
        tool_pose = (
            float(pose_weight)
            * (
                float(np.dot(radial_error, radial_error))
                / (self._radial_pose_scale ** 2)
                + float(axial_error ** 2)
                / (self._axial_pose_scale ** 2)
            )
            if objective_active
            else 0.0
        )

        if i == align_end - 1:
            self._align_distance = float(np.linalg.norm(residual))
            self._align_axial_error = float(abs(axial_error))
            self._align_radial_distance = float(
                np.linalg.norm(radial_error)
            )
        if i == engage_end - 1:
            self._engage_distance = float(np.linalg.norm(residual))
            self._engage_axial_error = float(abs(axial_error))
            self._engage_radial_distance = float(
                np.linalg.norm(radial_error)
            )
            self._engage_nail_angle = float(q[self._q_nail])
            self._engage_handle_angle = float(q[self._q_root_roll])
        if i == int(num_ctrl_steps) - 1:
            self._final_pose_distance = float(np.linalg.norm(residual))
            self._final_axial_error = float(abs(axial_error))
            self._final_radial_distance = float(
                np.linalg.norm(radial_error)
            )

        turn_goal = 0.0
        turn_objective_enabled = bool(
            self._optimization_stage == "full"
            or self._is_turn_optimization_stage()
        )
        if (
            objective_active
            and i >= engage_end
            and turn_objective_enabled
        ):
            turn_steps = int(num_ctrl_steps) - engage_end
            turn_progress = float(i - engage_end + 1) / float(turn_steps)
            target_angle = turn_progress * self._nail_target
            (
                nail_delta,
                handle_delta,
                nail_error,
                handle_error,
                sync_error,
            ) = self._turn_components(q, target_angle)
            target2 = self._nail_target ** 2
            turn_weight = 1.0 / float(turn_steps)
            if i == self._optimization_stage_end(num_ctrl_steps) - 1:
                turn_weight += self._turn_terminal_weight
            turn_goal = float(
                turn_weight
                *
                (
                    nail_error ** 2
                    + self._turn_handle_weight * handle_error ** 2
                    + self._turn_sync_weight * sync_error ** 2
                )
                / target2
            )
            if (
                self._is_turn_optimization_stage()
                and i == self._optimization_stage_end(num_ctrl_steps) - 1
            ):
                self._stage_terminal_cache = {
                    "stage": self._optimization_stage,
                    "nail_turn_rad": nail_delta,
                    "handle_turn_rad": handle_delta,
                    "nail_turn_deg": float(np.degrees(nail_delta)),
                    "handle_turn_deg": float(np.degrees(handle_delta)),
                    "pose_distance": float(np.linalg.norm(residual)),
                    "axial_error": float(abs(axial_error)),
                    "radial_distance": float(
                        np.linalg.norm(radial_error)
                    ),
                }
        if (
            i == int(num_ctrl_steps) - 1
            and (
                self._optimization_stage == "full"
                or self._optimization_stage
                == self._turn_optimization_stages[-1]
            )
        ):
            self._terminal_cache = {
                "nail_turn_rad": nail_delta,
                "handle_turn_rad": handle_delta,
                "nail_turn_deg": float(np.degrees(nail_delta)),
                "handle_turn_deg": float(
                    np.degrees(handle_delta)
                ),
                "align_distance": self._align_distance,
                "align_axial_error": self._align_axial_error,
                "align_radial_distance": self._align_radial_distance,
                "engage_distance": self._engage_distance,
                "engage_axial_error": self._engage_axial_error,
                "engage_radial_distance": (
                    self._engage_radial_distance
                ),
                "final_pose_distance": self._final_pose_distance,
                "final_axial_error": self._final_axial_error,
                "final_radial_distance": self._final_radial_distance,
                "turn_sync_error_deg": float(
                    np.degrees(sync_error)
                ),
                "task_success": bool(
                    nail_delta >= self._success_nail
                    and self._final_pose_distance
                    <= self._success_pose_tolerance
                ),
                "optimization_stage": self._optimization_stage,
            }

        u_i = np.asarray(u_i, dtype=np.float64)
        if objective_active and len(u_i):
            scale = self.action_scale(len(u_i))
            control = float(np.mean((u_i / scale) ** 2))
            if self._prev_control is not None:
                delta = (u_i - self._prev_control) / scale
                control += self._control_smooth_weight * float(
                    np.mean(delta ** 2)
                )
        else:
            control = 0.0
        self._prev_control_by_step[int(i)] = (
            None
            if self._prev_control is None
            else np.asarray(
                self._prev_control,
                dtype=np.float64,
            ).copy()
        )
        self._prev_control = u_i.copy()

        return {
            "tool_pose": float(tool_pose),
            "turn_goal": float(turn_goal),
            "control": float(control),
        }

    def write_terminal_grads(
        self,
        i,
        num_ctrl_steps,
        u_i,
        variables,
        q,
        ndof_u,
        ndof_var,
        ndof_r,
        sub_steps,
        coef,
        df_du,
        df_dvar,
        df_dq,
    ):
        u_i = np.asarray(u_i, dtype=np.float64)
        objective_active = self._objective_active(
            i,
            num_ctrl_steps,
        )
        if objective_active and ndof_u:
            scale = self.action_scale(ndof_u)
            u_base = int(i) * int(sub_steps) * int(ndof_u)
            df_du[u_base : u_base + ndof_u] += (
                float(coef["control"])
                * 2.0
                * u_i
                / (float(ndof_u) * scale ** 2)
            )
            previous = self._prev_control_by_step.get(int(i))
            if previous is not None:
                delta = u_i - previous
                smooth_scale = (
                    float(coef["control"])
                    * self._control_smooth_weight
                    * 2.0
                    / float(ndof_u)
                    / (scale ** 2)
                )
                df_du[u_base : u_base + ndof_u] += (
                    smooth_scale * delta
                )
                previous_base = (
                    (int(i) - 1)
                    * int(sub_steps)
                    * int(ndof_u)
                )
                if previous_base >= 0:
                    df_du[
                        previous_base :
                        previous_base + ndof_u
                    ] -= smooth_scale * delta

        offset, pose_weight = self._pose_target(
            i,
            num_ctrl_steps,
        )
        (
            _,
            _,
            _,
            axis,
            axis_length,
            displacement,
            axial_position,
            axial_error,
            radial_error,
            residual,
        ) = self._pose_components(variables, offset)
        last_step = (int(i) + 1) * int(sub_steps) - 1
        var_base = last_step * int(ndof_var)
        if objective_active and pose_weight > 0.0:
            weighted_pose = (
                float(coef["tool_pose"]) * float(pose_weight)
            )
            displacement_grad = 2.0 * weighted_pose * (
                radial_error / (self._radial_pose_scale ** 2)
                + axial_error
                * axis
                / (self._axial_pose_scale ** 2)
            )
            axis_objective_grad = (
                2.0
                * weighted_pose
                * (
                    axial_error / (self._axial_pose_scale ** 2)
                    - axial_position / (self._radial_pose_scale ** 2)
                )
                * displacement
            )
            projection_jacobian = (
                np.eye(3, dtype=np.float64)
                - np.outer(axis, axis)
            ) / axis_length
            axis_vector_grad = projection_jacobian @ axis_objective_grad
            tool_base = var_base + self._var_tool
            nail_base = var_base + self._var_nail_center
            axis_base = var_base + self._var_nail_axis
            df_dvar[tool_base : tool_base + 3] += displacement_grad
            df_dvar[nail_base : nail_base + 3] -= (
                displacement_grad + axis_vector_grad
            )
            df_dvar[axis_base : axis_base + 3] += axis_vector_grad

        if (
            objective_active
            and int(i) >= self._phase_boundaries(num_ctrl_steps)[1]
            and (
                self._optimization_stage == "full"
                or self._is_turn_optimization_stage()
            )
        ):
            _, engage_end = self._phase_boundaries(num_ctrl_steps)
            turn_steps = int(num_ctrl_steps) - engage_end
            turn_progress = (
                float(int(i) - engage_end + 1) / float(turn_steps)
            )
            target_angle = turn_progress * self._nail_target
            (
                _,
                _,
                nail_error,
                handle_error,
                sync_error,
            ) = self._turn_components(q, target_angle)
            turn_weight = 1.0 / float(turn_steps)
            if (
                int(i)
                == self._optimization_stage_end(num_ctrl_steps) - 1
            ):
                turn_weight += self._turn_terminal_weight
            common = (
                float(coef["turn_goal"])
                * 2.0
                * turn_weight
                / (self._nail_target ** 2)
            )
            nail_grad = common * (
                nail_error
                - self._turn_sync_weight * sync_error
            )
            handle_grad = common * (
                self._turn_handle_weight * handle_error
                + self._turn_sync_weight * sync_error
            )
            q_final_base = last_step * int(ndof_r)
            df_dq[q_final_base + self._q_nail] += nail_grad
            df_dq[
                q_final_base + self._q_root_roll
            ] += handle_grad

            boundary_step = int(engage_end) * int(sub_steps) - 1
            q_boundary_base = boundary_step * int(ndof_r)
            df_dq[q_boundary_base + self._q_nail] -= nail_grad
            df_dq[
                q_boundary_base + self._q_root_roll
            ] -= handle_grad

    def optimization_stage_acceptance(self, stage):
        normalized = str(stage).strip().lower()
        if normalized == "align":
            distance = (
                float("inf")
                if self._align_distance is None
                else float(self._align_distance)
            )
            return {
                "accepted": distance <= self._align_tolerance,
                "stage": normalized,
                "align_distance": distance,
                "align_axial_error": self._align_axial_error,
                "align_radial_distance": self._align_radial_distance,
                "required_distance": self._align_tolerance,
            }
        if normalized == "engage":
            distance = (
                float("inf")
                if self._engage_distance is None
                else float(self._engage_distance)
            )
            return {
                "accepted": distance <= self._engage_tolerance,
                "stage": normalized,
                "engage_distance": distance,
                "engage_axial_error": self._engage_axial_error,
                "engage_radial_distance": (
                    self._engage_radial_distance
                ),
                "required_distance": self._engage_tolerance,
            }
        if normalized in self._turn_optimization_stages:
            terminal = dict(self._stage_terminal_cache)
            segment_index = self._turn_stage_index(normalized)
            is_final_segment = bool(
                segment_index == self._turn_curriculum_segments
            )
            required_angle = (
                self._success_nail if is_final_segment else None
            )
            nail_turn = float(terminal.get("nail_turn_rad", 0.0))
            pose_distance = float(
                terminal.get("pose_distance", float("inf"))
            )
            pose_accepted = bool(
                pose_distance <= self._success_pose_tolerance
            )
            return {
                "accepted": bool(
                    pose_accepted
                    and (
                        not is_final_segment
                        or nail_turn >= self._success_nail
                    )
                ),
                "stage": normalized,
                "turn_curriculum_segment": int(segment_index),
                "turn_curriculum_segments": int(
                    self._turn_curriculum_segments
                ),
                "final_turn_segment": is_final_segment,
                "nail_turn_deg": float(
                    terminal.get("nail_turn_deg", 0.0)
                ),
                "required_nail_turn_deg": (
                    None
                    if required_angle is None
                    else float(np.degrees(required_angle))
                ),
                "pose_distance": pose_distance,
                "required_pose_distance": (
                    self._success_pose_tolerance
                ),
            }
        raise ValueError(
            f"unsupported torque optimization stage {stage!r}"
        )

    def rollout_diagnostics(self, runner, params):
        final_forward_diagnostics = dict(
            getattr(runner, "_last_forward_diagnostics", {}) or {}
        )
        optimizer_diagnostics = dict(
            getattr(runner, "_action_optimizer_diagnostics", {}) or {}
        )
        action, cage = runner.unpack_params(params)
        design_params = None
        if (
            runner.optimize_design
            and runner.design_bundle is not None
            and cage is not None
        ):
            design_params, _ = runner.apply_morphology(
                cage,
                generate_mesh=False,
            )

        controls = runner.controls_from_action(action)
        runner.sim.reset()
        runner._reset_staged_motion_stop_runtime()
        self._reset_rollout_cache()
        geometry_audit = None
        if self._geometry_audit_enabled:
            model_root = ET.parse(runner.model_path).getroot()
            tool_body_names = {
                str(body.attrib["name"])
                for body in model_root.findall(".//body[@name]")
                if str(body.attrib.get("asset_role", "")).strip().lower()
                == "head"
            }
            if not tool_body_names:
                raise ValueError(
                    "torque geometry audit found no asset_role='head' bodies"
                )
            geometry_audit = SymmetricOverlapAudit(
                runner.model_path,
                tool_body_names=tool_body_names,
                operated_body_names=set(
                    self._geometry_audit_operated_body_names
                ),
                bundle=(
                    runner.design_bundle
                    if runner.optimize_design and design_params is not None
                    else None
                ),
                design_params=design_params,
                containment_fraction_limit=(
                    self._geometry_audit_containment_fraction
                ),
                normalized_depth_limit=(
                    self._geometry_audit_normalized_depth
                ),
                sample_interval_seconds=(
                    float(runner.sub_steps) * 1.0e-3
                ),
                min_sustained_seconds=(
                    self._geometry_audit_min_sustained_seconds
                ),
            )
        trace = []
        align_end, engage_end = self._phase_boundaries(
            runner.num_ctrl_steps
        )
        for i in range(runner.num_ctrl_steps):
            u_i = controls[
                i * runner.ndof_u :
                (i + 1) * runner.ndof_u
            ]
            runner.advance_control_step(
                i,
                u_i,
                backward_flag=False,
                verbose=False,
            )
            variables = runner.sim.get_variables()
            q = np.asarray(runner.sim.get_q(), dtype=np.float64)
            if geometry_audit is not None:
                geometry_audit.observe(q, step=i)
            terms = self.compute_terms(
                i,
                runner.num_ctrl_steps,
                u_i,
                variables,
                q,
            )
            (
                _,
                _,
                _,
                _,
                _,
                _,
                seat_axial_position,
                _,
                seat_radial_error,
                seat_residual,
            ) = self._pose_components(variables, 0.0)
            trace.append(
                {
                    "control_step": int(i),
                    "phase": (
                        "align"
                        if i < align_end
                        else (
                            "engage"
                            if i < engage_end
                            else "turn"
                        )
                    ),
                    "seat_distance": float(
                        np.linalg.norm(seat_residual)
                    ),
                    "seat_axial_error": float(
                        abs(seat_axial_position)
                    ),
                    "seat_radial_distance": float(
                        np.linalg.norm(seat_radial_error)
                    ),
                    "nail_angle_deg": float(
                        np.degrees(q[self._q_nail])
                    ),
                    "handle_angle_deg": float(
                        np.degrees(q[self._q_root_roll])
                    ),
                    "tool_pose_term": float(terms["tool_pose"]),
                    "turn_goal_term": float(terms["turn_goal"]),
                }
            )

        final_q = np.asarray(runner.sim.get_q(), dtype=np.float64)
        nail_turn_rad = (
            0.0
            if self._engage_nail_angle is None
            else float(final_q[self._q_nail])
            - float(self._engage_nail_angle)
        )
        handle_turn_rad = (
            0.0
            if self._engage_handle_angle is None
            else float(final_q[self._q_root_roll])
            - float(self._engage_handle_angle)
        )
        terminal = dict(self._terminal_cache)
        geometry_summary = (
            geometry_audit.summary()
            if geometry_audit is not None
            else {
                "enabled": False,
                "ok": True,
                "severe_overlap": False,
            }
        )
        symmetric_geometry_ok = bool(geometry_summary.get("ok", True))
        raw_task_success = bool(
            nail_turn_rad >= self._success_nail
            and self._final_pose_distance is not None
            and self._final_pose_distance
            <= self._success_pose_tolerance
        )
        task_success = bool(raw_task_success and symmetric_geometry_ok)
        terminal.update(
            {
                "nail_turn_rad": nail_turn_rad,
                "handle_turn_rad": handle_turn_rad,
                "nail_turn_deg": float(np.degrees(nail_turn_rad)),
                "handle_turn_deg": float(np.degrees(handle_turn_rad)),
                "turn_sync_error_deg": float(
                    np.degrees(handle_turn_rad - nail_turn_rad)
                ),
                "align_distance": self._align_distance,
                "align_axial_error": self._align_axial_error,
                "align_radial_distance": self._align_radial_distance,
                "engage_distance": self._engage_distance,
                "engage_axial_error": self._engage_axial_error,
                "engage_radial_distance": self._engage_radial_distance,
                "final_pose_distance": self._final_pose_distance,
                "final_axial_error": self._final_axial_error,
                "final_radial_distance": self._final_radial_distance,
                "raw_task_success_before_geometry_audit": raw_task_success,
                "symmetric_geometry_audit": geometry_summary,
                "symmetric_geometry_ok": symmetric_geometry_ok,
                "task_success": task_success,
                "success": task_success,
                "nail_target_deg": float(
                    np.degrees(self._nail_target)
                ),
                "success_nail_deg": float(
                    np.degrees(self._success_nail)
                ),
                "success_pose_tolerance": (
                    self._success_pose_tolerance
                ),
                "align_tolerance": self._align_tolerance,
                "engage_tolerance": self._engage_tolerance,
                "root_rotation_mode": self._root_rotation_mode,
                "stage_stop": final_forward_diagnostics.get("stage_stop"),
                "per_stage_loss": final_forward_diagnostics.get(
                    "per_stage_loss", {}
                ),
                "optimizer": optimizer_diagnostics,
                "phase_boundary_steps": {
                    "align_end": int(align_end),
                    "engage_end": int(engage_end),
                    "turn_end": int(runner.num_ctrl_steps),
                },
                "turn_curriculum": {
                    "configured_segments": int(
                        self._configured_turn_curriculum_segments
                    ),
                    "segments": int(self._turn_curriculum_segments),
                    "windows": [
                        [int(start), int(end)]
                        for start, end in self._turn_segment_windows(
                            runner.num_ctrl_steps
                        )
                    ],
                    "required_nail_turn_deg": [
                        *(
                            [None]
                            * (self._turn_curriculum_segments - 1)
                        ),
                        float(np.degrees(self._success_nail)),
                    ],
                },
                "control_trace": trace,
            }
        )
        return terminal

    def print_info(self, *args):
        _print_info(*args)




__all__ = ["MISSION_NAME", "TaskDynamics", "TaskObjective"]
