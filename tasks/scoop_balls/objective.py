"""Generic continuous pickup objective for the canonical Scoop task.

The task observes only public scene variables:

* one ``scoop_endeffector`` function marker,
* the source marker,
* the transported spheres, and
* the four Handle controls (world xyz plus one selected-axis rotation).

No loss or milestone inspects Head geometry or an authored action trajectory.
An authoritative carry gate checks co-motion inside the current moving
function-group cuboid. ``reference.xml`` is merely
one legal compiled candidate and follows exactly the same code path as every
searched candidate.

The three causal stages are always present: Approach, Sweep, and Lift.  The
public staged optimizer only shortens the active
rollout and action window; it never changes the task definition.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bilevel.runner import BaseTask
from tasks.objective import TaskObjective


MISSION_NAME = "scoop_balls"


class TaskDynamics(BaseTask):
    """Four-control, zero-initialized payload pickup task."""

    has_finger_design = False
    optimize_design = False

    _STAGES = ("approach", "sweep", "lift")
    def __init__(self, *, num_steps, sub_steps, task_config):
        self._task_config = dict(task_config)
        self._num_steps = int(num_steps)
        self._sub_steps = int(sub_steps)
        if self._num_steps <= 0 or self._sub_steps <= 0:
            raise ValueError("scoop num_steps and sub_steps must be positive")

        def config_float(name):
            if name not in self._task_config:
                raise ValueError(
                    "scoop task_config is missing required field {!r}".format(
                        name
                    )
                )
            value = float(self._task_config[name])
            if not np.isfinite(value):
                raise ValueError("scoop {!r} must be finite".format(name))
            return value

        self._phase_fractions = tuple(
            config_float(name)
            for name in ("approach_end_fraction", "sweep_end_fraction")
        )
        if not (
            0.0
            < self._phase_fractions[0]
            < self._phase_fractions[1]
            < 1.0
        ):
            raise ValueError(
                "scoop phase fractions must be strictly increasing in (0, 1)"
            )

        self._optimization_stage = "full"

        raw_stage_weights = np.asarray(
            self._task_config.get(
                "curriculum_stage_weights", np.ones(len(self._STAGES))
            ),
            dtype=np.float64,
        )
        if raw_stage_weights.shape != (len(self._STAGES),):
            raise ValueError(
                "scoop curriculum_stage_weights must have three entries"
            )
        if np.any(~np.isfinite(raw_stage_weights)) or np.any(
            raw_stage_weights <= 0.0
        ):
            raise ValueError(
                "scoop curriculum_stage_weights must be finite and positive"
            )
        self._curriculum_stage_weights = raw_stage_weights

        self._coef = {
            "goal": config_float("coef_goal"),
            "payload": config_float("coef_payload"),
            "control": config_float("coef_control"),
        }
        if any(value < 0.0 for value in self._coef.values()):
            raise ValueError("scoop objective coefficients must be nonnegative")

        self._control_smooth_weight = config_float(
            "control_smooth_weight"
        )
        self._phase_reference_weight = config_float(
            "phase_reference_weight"
        )
        self._phase_endpoint_weight = config_float(
            "phase_endpoint_weight"
        )
        self._action_scale = np.asarray(
            [
                config_float("action_scale_x"),
                config_float("action_scale_y"),
                config_float("action_scale_z"),
                config_float("action_scale_pitch"),
            ],
            dtype=np.float64,
        )
        self._action_bound = config_float("action_bound")
        if (
            self._control_smooth_weight < 0.0
            or self._phase_reference_weight <= 0.0
            or self._phase_endpoint_weight <= 0.0
            or np.any(self._action_scale <= 0.0)
            or self._action_bound <= 0.0
        ):
            raise ValueError(
                "scoop action scales/bound must be positive and smoothing "
                "must be nonnegative"
            )

        self._approach_clearance_radii = config_float(
            "approach_clearance_ball_radii"
        )
        self._function_below_ball_bottom_target_clearance_radii = config_float(
            "function_below_ball_bottom_target_clearance_ball_radii"
        )
        self._function_height_tolerance_radii = config_float(
            "function_height_tolerance_ball_radii"
        )
        self._approach_position_tolerance_radii = config_float(
            "approach_position_tolerance_ball_radii"
        )
        self._approach_pitch = np.radians(
            config_float("approach_pitch_deg")
        )
        self._approach_angle_tolerance = np.radians(
            config_float("approach_angle_tolerance_deg")
        )
        self._approach_ball_motion_tolerance_radii = config_float(
            "approach_ball_motion_tolerance_ball_radii"
        )
        self._approach_ball_motion_weight = config_float(
            "approach_ball_motion_weight"
        )
        if min(
            self._approach_clearance_radii,
            self._function_below_ball_bottom_target_clearance_radii,
            self._function_height_tolerance_radii,
            self._approach_position_tolerance_radii,
            self._approach_angle_tolerance,
            self._approach_ball_motion_tolerance_radii,
            self._approach_ball_motion_weight,
        ) <= 0.0:
            raise ValueError("scoop Approach scales and tolerances must be positive")

        self._sweep_forward_offset_radii = config_float(
            "sweep_forward_offset_ball_radii"
        )
        self._sweep_position_tolerance_radii = config_float(
            "sweep_position_tolerance_ball_radii"
        )
        self._sweep_gate_position_tolerance_radii = config_float(
            "sweep_gate_position_tolerance_ball_radii"
        )
        self._sweep_pitch = np.radians(config_float("sweep_pitch_deg"))
        self._sweep_angle_tolerance = np.radians(
            config_float("sweep_angle_tolerance_deg")
        )
        self._sweep_gate_angle_tolerance = np.radians(
            config_float("sweep_gate_angle_tolerance_deg")
        )
        self._sweep_angle_weight = config_float("sweep_angle_weight")
        self._sweep_capture_distance_radii = config_float(
            "sweep_capture_xy_distance_ball_radii"
        )
        self._sweep_capture_loss_distance_radii = config_float(
            "sweep_capture_loss_xy_distance_ball_radii"
        )
        self._sweep_diagnostic_window_fraction = config_float(
            "sweep_diagnostic_window_fraction"
        )
        self._sweep_diagnostic_position_motion_tolerance_radii = config_float(
            "sweep_diagnostic_position_motion_tolerance_ball_radii"
        )
        self._sweep_diagnostic_pitch_motion_tolerance = np.radians(
            config_float("sweep_diagnostic_pitch_motion_tolerance_deg")
        )
        if min(
            self._sweep_forward_offset_radii,
            self._sweep_position_tolerance_radii,
            self._sweep_gate_position_tolerance_radii,
            self._sweep_angle_tolerance,
            self._sweep_gate_angle_tolerance,
            self._sweep_angle_weight,
            self._sweep_capture_distance_radii,
            self._sweep_capture_loss_distance_radii,
            self._sweep_diagnostic_window_fraction,
            self._sweep_diagnostic_position_motion_tolerance_radii,
            self._sweep_diagnostic_pitch_motion_tolerance,
        ) <= 0.0:
            raise ValueError("scoop Sweep scales and tolerances must be positive")
        if self._sweep_diagnostic_window_fraction > 1.0:
            raise ValueError(
                "scoop sweep_diagnostic_window_fraction must not exceed one"
            )
        self._carry_pitch = np.radians(config_float("carry_pitch_deg"))
        self._lift_height_above_source_support_radii = config_float(
            "lift_height_above_source_support_ball_radii"
        )
        self._lift_height_gate_radii = config_float(
            "lift_height_gate_ball_radii"
        )
        self._carry_angle_loss_scale = np.radians(
            config_float("carry_angle_loss_scale_deg")
        )
        self._lift_height_loss_scale_radii = config_float(
            "lift_height_loss_scale_ball_radii"
        )
        self._lift_height_weight = config_float("lift_height_weight")
        self._lift_horizontal_position_weight = config_float(
            "lift_horizontal_position_weight"
        )
        self._lift_horizontal_position_tolerance_radii = config_float(
            "lift_horizontal_position_tolerance_ball_radii"
        )
        self._lift_payload_clearance_radii = config_float(
            "lift_payload_clearance_ball_radii"
        )
        self._lift_angle_weight = config_float("lift_angle_weight")
        self._lift_horizontal_retention_weight = config_float(
            "lift_horizontal_retention_weight"
        )
        self._lift_payload_clearance_weight = config_float(
            "lift_payload_clearance_weight"
        )
        self._carry_retention_tolerance_radii = config_float(
            "carry_retention_tolerance_ball_radii"
        )
        self._lift_carry_min_fraction = config_float("lift_carry_min_fraction")
        if min(
            self._lift_height_above_source_support_radii,
            self._lift_height_gate_radii,
            self._carry_angle_loss_scale,
            self._lift_height_loss_scale_radii,
            self._lift_height_weight,
            self._lift_horizontal_position_weight,
            self._lift_horizontal_position_tolerance_radii,
            self._lift_payload_clearance_radii,
            self._lift_angle_weight,
            self._lift_horizontal_retention_weight,
            self._lift_payload_clearance_weight,
            self._carry_retention_tolerance_radii,
        ) <= 0.0:
            raise ValueError("scoop Lift scales and tolerances must be positive")
        if (
            self._lift_height_gate_radii
            > self._lift_height_above_source_support_radii
        ):
            raise ValueError(
                "scoop Lift height gate must not exceed the loss target"
            )
        if not 0.0 < self._lift_carry_min_fraction <= 1.0:
            raise ValueError(
                "scoop lift_carry_min_fraction must be in (0, 1]"
            )
        if self._carry_pitch <= 0.0:
            raise ValueError("scoop carry_pitch_deg must be positive")
        self._required_ball_count = int(
            self._task_config["required_ball_count"]
        )
        if self._required_ball_count <= 0:
            raise ValueError("scoop required_ball_count must be positive")
        # Default public variable order.  Generated candidates are always
        # rediscovered by marker name in configure_model.
        self._var_boxA_base = 0
        self._var_boxB_base = 3
        self._var_ball_bases = (6, 9)
        self._var_scoop_base = 12
        self._required_var_len = 15
        self._q_root_translation = slice(0, 3)
        # The reference selects pitch. Search candidates may select any one
        # canonical Handle rotation while retaining the same scalar control
        # and objective schedule.
        self._q_pitch = 3
        self._root_rotation_mode = "pitch"
        self._root_rotation_axis = np.asarray(
            [0.0, 1.0, 0.0], dtype=np.float64
        )
        self._root_rotation_joint_name = "freeform_pitch_joint"
        self._ball_radius = 0.5
        self._source_support_half_thickness = None
        self._payload_contact_pairs = ()

        self._initial_scoop = None
        self._initial_box_a = None
        self._initial_box_b = None
        self._initial_balls = None
        self._stage_endpoint_cache = {}
        self._stage_endpoint_points = {}
        self._stage_endpoint_balls = {}
        self._stage_boundary_pitches = {}
        self._lift_start_position = None
        self._carry_relative_anchor = None
        self._prev_control_for_terms = None
        self._prev_term_step = None
        self._control_prev_by_step = {}
        self._control_smooth_weight_by_step = {}
        self._prev_phase_marker = None
        self._prev_phase_pitch = None
        self._prev_phase_step = None
        self._prev_phase_stage = None
        self._phase_motion_by_step = {}
        self._lift_horizontal_drift_by_step = {}
        self._lift_motion_by_step = {}

    def num_steps(self):
        return self._num_steps

    def sub_steps(self):
        return self._sub_steps

    def objective_weights(self):
        return dict(self._coef)

    def action_scale(self, ndof_u):
        if int(ndof_u) == 4:
            return self._action_scale.copy()
        return np.ones(int(ndof_u), dtype=np.float64)

    def action_to_control(self, action, ndof_u):
        action = np.asarray(action, dtype=np.float64)
        count = action.size // int(ndof_u)
        scales = np.tile(self.action_scale(ndof_u), count)
        return np.tanh(action) * scales

    def action_control_jacobian_diag(self, action, ndof_u):
        action = np.asarray(action, dtype=np.float64)
        count = action.size // int(ndof_u)
        scales = np.tile(self.action_scale(ndof_u), count)
        return scales * (1.0 - np.tanh(action) ** 2)

    @staticmethod
    def _joint_dof(joint_type):
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

    @staticmethod
    def _normalized_axis(joint):
        axis = np.fromstring(
            joint.attrib.get("axis", ""),
            sep=" ",
            dtype=np.float64,
        )
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or float(np.linalg.norm(axis)) <= 1.0e-12
        ):
            raise ValueError(
                "scoop selected rotation joint {!r} has invalid axis".format(
                    joint.attrib.get("name", "")
                )
            )
        return axis / float(np.linalg.norm(axis))

    def _configure_q_layout(self, root):
        q_cursor = 0
        root_translation = None
        root_rotation = None
        root_rotation_mode = None
        root_rotation_axis = None
        root_rotation_joint_name = None
        root_rotation_count = 0
        rotation_joint_modes = {
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
            if name == "freeform_root_joint":
                if joint_type != "translational" or dof != 3:
                    raise ValueError(
                        "scoop freeform_root_joint must be translational"
                    )
                root_translation = slice(q_cursor, q_cursor + 3)
            elif name in rotation_joint_modes:
                if joint_type != "revolute" or dof != 1:
                    raise ValueError(
                        "scoop selected rotation joint {} must be "
                        "revolute".format(name)
                    )
                mode = rotation_joint_modes[name]
                axis = self._normalized_axis(joint)
                if not np.allclose(
                    axis,
                    canonical_axes[mode],
                    rtol=0.0,
                    atol=1.0e-9,
                ):
                    raise ValueError(
                        "scoop selected rotation joint name and axis disagree"
                    )
                root_rotation = q_cursor
                root_rotation_mode = mode
                root_rotation_axis = axis
                root_rotation_joint_name = name
                root_rotation_count += 1
            q_cursor += dof

        if root_translation is None or root_rotation is None:
            raise ValueError(
                "scoop XML must contain translational freeform_root_joint "
                "and one canonical rotation joint"
            )
        if root_rotation_count != 1:
            raise ValueError(
                "scoop XML must contain exactly one root rotation DOF "
                "chosen from roll, pitch, or yaw"
            )
        self._q_root_translation = root_translation
        # Retain the established internal name while it now identifies the
        # selected scalar rotation coordinate for every search branch.
        self._q_pitch = int(root_rotation)
        self._root_rotation_mode = str(root_rotation_mode)
        self._root_rotation_axis = np.asarray(
            root_rotation_axis, dtype=np.float64
        )
        self._root_rotation_joint_name = str(root_rotation_joint_name)

    @staticmethod
    def _require_position_motor(root, joint_name):
        motors = [
            motor
            for motor in root.iter("motor")
            if motor.attrib.get("joint") == joint_name
        ]
        if len(motors) != 1:
            raise ValueError(
                "scoop expects exactly one motor for {!r}".format(joint_name)
            )
        if motors[0].attrib.get("ctrl", "").lower() != "position":
            raise ValueError(
                "scoop motor {!r} must use position control".format(
                    joint_name
                )
            )

    def _configure_variable_layout(self, model_path, *, configure_q=True):
        root = ET.parse(model_path).getroot()
        if configure_q:
            self._configure_q_layout(root)
        variable = root.find("variable")
        if variable is None:
            raise ValueError(
                "scoop model is missing the canonical variable section"
            )

        scoops = []
        box_as = []
        box_bs = []
        balls = []
        for index, marker in enumerate(variable.findall("endeffector")):
            joint = marker.attrib.get("joint", "").lower()
            if joint == "scoop_endeffector":
                scoops.append(index)
            elif "boxa" in joint:
                box_as.append(index)
            elif "boxb" in joint:
                box_bs.append(index)
            elif "ball" in joint:
                balls.append(index)
        if (
            len(scoops) != 1
            or len(box_as) != 1
            or len(box_bs) != 1
            or len(balls) != self._required_ball_count
        ):
            raise ValueError(
                "scoop requires one public function marker, one source, one "
                "target, and exactly {} payload variables".format(
                    self._required_ball_count
                )
            )

        self._var_scoop_base = 3 * scoops[0]
        self._var_boxA_base = 3 * box_as[0]
        self._var_boxB_base = 3 * box_bs[0]
        self._var_ball_bases = tuple(3 * index for index in balls)
        self._required_var_len = 3 * (
            max([scoops[0], box_as[0], box_bs[0]] + balls) + 1
        )

        ball_joint_names = {
            marker.attrib.get("joint", "")
            for marker in variable.findall("endeffector")
            if "ball" in marker.attrib.get("joint", "").lower()
        }
        radii = []
        for link in root.iter("link"):
            joint = link.find("joint")
            body = link.find("body")
            if (
                joint is not None
                and body is not None
                and joint.attrib.get("name") in ball_joint_names
                and body.attrib.get("type") == "sphere"
            ):
                radii.append(float(body.attrib["radius"]))
        if len(radii) != len(balls) or min(radii) <= 0.0:
            raise ValueError(
                "scoop requires one physical sphere radius per payload"
            )
        if not np.allclose(radii, radii[0], rtol=1.0e-9, atol=1.0e-12):
            raise ValueError("scoop currently requires equal-radius payloads")
        self._ball_radius = float(radii[0])

        source_joint_names = {
            marker.attrib.get("joint", "")
            for marker in variable.findall("endeffector")
            if "boxa" in marker.attrib.get("joint", "").lower()
        }
        half_thicknesses = []
        for link in root.iter("link"):
            joint = link.find("joint")
            body = link.find("body")
            if (
                joint is not None
                and body is not None
                and joint.attrib.get("name") in source_joint_names
                and body.attrib.get("type") == "cuboid"
            ):
                size = np.asarray(
                    [float(value) for value in body.attrib["size"].split()],
                    dtype=np.float64,
                )
                if size.shape != (3,) or np.any(size <= 0.0):
                    raise ValueError("scoop source support size is invalid")
                half_thicknesses.append(0.5 * float(size[2]))
        if len(half_thicknesses) != 1:
            raise ValueError(
                "scoop requires one cuboid source support registered by boxA"
            )
        self._source_support_half_thickness = half_thicknesses[0]

    @staticmethod
    def _xml_contact_pairs(root):
        pairs = set()
        contact = root.find("contact")
        if contact is None:
            return pairs
        for element in contact:
            body1 = element.attrib.get(
                "body1", element.attrib.get("general_body")
            )
            body2 = element.attrib.get(
                "body2", element.attrib.get("primitive_body")
            )
            if body1 and body2:
                pairs.add(tuple(sorted((str(body1), str(body2)))))
        return pairs

    @staticmethod
    def _function_contact_bodies(root):
        marker = root.find(".//link[@name='scoop_endeffector']")
        if marker is None:
            return ()
        root_node_id = marker.attrib.get("function_group_root", "").strip()
        leaf_node_ids = tuple(
            value.strip()
            for value in marker.attrib.get("function_group_leaves", "").split(",")
            if value.strip()
        )
        if not root_node_id:
            return ()
        parent_by_element = {
            child: parent
            for parent in root.iter()
            for child in parent
        }
        links_by_node = {
            str(link.attrib["node_id"]): link
            for link in root.findall(".//link[@node_id]")
        }
        root_link = links_by_node.get(root_node_id)
        if root_link is None:
            return ()
        group_links = {root_link}
        for leaf_node_id in leaf_node_ids or (root_node_id,):
            link = links_by_node.get(leaf_node_id)
            while link is not None and link is not root_link:
                group_links.add(link)
                parent = parent_by_element.get(link)
                while parent is not None and parent.tag != "link":
                    parent = parent_by_element.get(parent)
                link = parent
            if link is not root_link:
                return ()
            group_links.add(root_link)
        bodies = []
        for link in root.findall(".//link[@node_id]"):
            if link not in group_links:
                continue
            body = link.find("body")
            body_name = None if body is None else body.attrib.get("name")
            if body_name and body_name not in bodies:
                bodies.append(str(body_name))
        return tuple(bodies)

    def _configure_semantic_contacts(self, model_path):
        root = ET.parse(model_path).getroot()
        variable = root.find("variable")
        if variable is None:
            raise ValueError(
                "scoop model is missing the canonical variable section"
            )
        payload_joints = tuple(
            marker.attrib.get("joint", "")
            for marker in variable.findall("endeffector")
            if "ball" in marker.attrib.get("joint", "").lower()
        )
        bodies_by_joint = {}
        for link in root.iter("link"):
            joint = link.find("joint")
            body = link.find("body")
            if joint is None or body is None:
                continue
            joint_name = joint.attrib.get("name", "")
            body_name = body.attrib.get("name", "")
            if joint_name in payload_joints and body_name:
                bodies_by_joint[joint_name] = str(body_name)
        payload_bodies = tuple(
            bodies_by_joint[joint_name]
            for joint_name in payload_joints
            if joint_name in bodies_by_joint
        )
        function_bodies = self._function_contact_bodies(root)
        physical_pairs = self._xml_contact_pairs(root)
        if (
            len(payload_bodies) != self._required_ball_count
            or not function_bodies
        ):
            raise ValueError(
                "scoop requires physical payload bodies and a public "
                "function-group contact set"
            )
        payload_pairs = []
        for payload_body in payload_bodies:
            pairs = tuple(
                (function_body, payload_body)
                for function_body in function_bodies
                if tuple(sorted((function_body, payload_body)))
                in physical_pairs
            )
            if len(pairs) != len(function_bodies):
                raise ValueError(
                    "scoop compiler must emit every function-group leaf "
                    "contact against every payload"
                )
            payload_pairs.append(pairs)
        self._payload_contact_pairs = tuple(payload_pairs)

    def configure_model(self, model_path, sim):
        _ = sim
        root = ET.parse(model_path).getroot()
        self._configure_q_layout(root)
        self._configure_variable_layout(model_path, configure_q=False)
        self._require_position_motor(root, "freeform_root_joint")
        self._require_position_motor(root, self._root_rotation_joint_name)
        self._configure_semantic_contacts(model_path)

    def _read_task_points(self, variables):
        values = np.asarray(variables, dtype=np.float64)
        if len(values) < self._required_var_len:
            raise ValueError(
                "scoop expects at least {} variables, got {}".format(
                    self._required_var_len, len(values)
                )
            )

        def point(base):
            return values[base : base + 3].copy()

        return (
            point(self._var_scoop_base),
            point(self._var_boxA_base),
            point(self._var_boxB_base),
            [point(base) for base in self._var_ball_bases],
        )

    def init_task(self, sim):
        if int(sim.ndof_u) != 4:
            raise ValueError(
                "scoop requires xyz translation plus one selected-axis "
                "Handle rotation control"
            )
        sim.set_q_init(np.zeros(sim.ndof_r, dtype=np.float64))
        sim.reset()
        (
            self._initial_scoop,
            self._initial_box_a,
            self._initial_box_b,
            initial_balls,
        ) = self._read_task_points(sim.get_variables())
        self._initial_balls = np.asarray(initial_balls, dtype=np.float64)

    def _require_initial_geometry(self):
        if (
            self._initial_scoop is None
            or self._initial_box_a is None
            or self._initial_box_b is None
            or self._initial_balls is None
            or self._source_support_half_thickness is None
        ):
            raise RuntimeError(
                "scoop configure_model/init_task must run before loss evaluation"
            )

    def _source_support_plane_z(self):
        self._require_initial_geometry()
        return float(
            self._initial_box_a[2] + self._source_support_half_thickness
        )

    def _source_rest_center_z(self):
        return self._source_support_plane_z() + self._ball_radius

    def _ball_cluster(self):
        self._require_initial_geometry()
        return np.mean(self._initial_balls, axis=0)

    def _approach_target(self):
        """Return a scene/object-scale target independent of Head geometry."""

        cluster = self._ball_cluster()
        target_xy = (
            cluster[:2]
            - self._sweep_direction()
            * self._approach_clearance_radii
            * self._ball_radius
        )
        return np.asarray(
            [
                target_xy[0],
                target_xy[1],
                self._source_support_plane_z()
                - self._function_below_ball_bottom_target_clearance_radii
                * self._ball_radius,
            ],
            dtype=np.float64,
        )

    def _sweep_direction(self):
        """Return the payload-normal direction away from the initial tool."""

        self._require_initial_geometry()
        payload_xy = self._initial_balls[:, :2]
        centered_xy = payload_xy - np.mean(payload_xy, axis=0)
        covariance = centered_xy.T @ centered_xy
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        lateral = eigenvectors[:, int(np.argmax(eigenvalues))]
        direction = np.asarray([lateral[1], -lateral[0]], dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-12:
            direction = self._initial_box_b[:2] - self._initial_box_a[:2]
            direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-12:
            direction = np.asarray([1.0, 0.0], dtype=np.float64)
            direction_norm = 1.0
        direction = direction / direction_norm
        tool_side = self._initial_scoop[:2] - self._ball_cluster()[:2]
        if float(np.dot(direction, tool_side)) > 0.0:
            direction = -direction
        return direction

    def _sweep_target_xy(self):
        """Return one object-scaled horizontal exit beyond the payload."""

        cluster = self._ball_cluster()
        return (
            cluster[:2]
            + self._sweep_direction()
            * self._sweep_forward_offset_radii
            * self._ball_radius
        )

    def _lift_target(self):
        if self._lift_start_position is None:
            raise RuntimeError("scoop Lift requires the Sweep boundary")
        target = np.asarray(
            self._lift_start_position, dtype=np.float64
        ).copy()
        target[2] = (
            self._source_support_plane_z()
            + self._lift_height_above_source_support_radii
            * self._ball_radius
        )
        return target

    def _lift_direction(self):
        direction = self._lift_target() - np.asarray(
            self._lift_start_position, dtype=np.float64
        )
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-12:
            raise RuntimeError("scoop Lift direction is undefined")
        return direction / length

    def _lift_horizontal_position_metrics(self, scoop):
        """Keep the public marker over its accepted Sweep boundary in xy."""

        if self._lift_start_position is None:
            raise RuntimeError("scoop Lift requires the Sweep boundary")
        delta = (
            np.asarray(scoop, dtype=np.float64)[:2]
            - np.asarray(self._lift_start_position, dtype=np.float64)[:2]
        )
        return {
            "delta": delta,
            "distance": float(np.linalg.norm(delta)),
            "tolerance": float(
                self._lift_horizontal_position_tolerance_radii
                * self._ball_radius
            ),
        }

    def _lift_payload_clearance(self, balls):
        required = self._lift_payload_clearance_radii * self._ball_radius
        clearance = (
            np.asarray(balls, dtype=np.float64)[:, 2]
            - self._source_rest_center_z()
        )
        return clearance, np.maximum(required - clearance, 0.0), required

    def _lift_horizontal_retention_metrics(self, scoop, balls):
        """Keep the Sweep-boundary payload/function relation in world xy."""

        if self._carry_relative_anchor is None:
            raise RuntimeError("scoop Lift requires the Sweep payload boundary")
        relative_xy = (
            np.asarray(balls, dtype=np.float64)[:, :2]
            - np.asarray(scoop, dtype=np.float64)[None, :2]
        )
        drift_xy = (
            relative_xy
            - np.asarray(self._carry_relative_anchor, dtype=np.float64)[:, :2]
        )
        tolerance = (
            self._carry_retention_tolerance_radii * self._ball_radius
        )
        return {
            "drift": drift_xy,
            "tolerance": float(tolerance),
            "loss": float(np.sum(drift_xy * drift_xy)) / tolerance ** 2,
        }

    def _lift_payload_clearance_metrics(self, pose, balls):
        clearance, _, required = self._lift_payload_clearance(balls)
        target = float(pose["progress"]) * required
        deficit = np.maximum(target - clearance, 0.0)
        return {
            "deficit": deficit,
            "required": float(required),
            "loss": float(np.sum(deficit * deficit)) / required ** 2,
        }

    def _phase_boundaries(self, num_ctrl_steps):
        count = max(1, int(num_ctrl_steps))
        if count < len(self._STAGES):
            return tuple(
                min(index + 1, count)
                for index in range(len(self._STAGES) - 1)
            )
        ends = []
        previous = 0
        remaining_after = len(self._STAGES) - 1
        for fraction in self._phase_fractions:
            end = int(
                np.clip(
                    round(count * fraction),
                    previous + 1,
                    count - remaining_after,
                )
            )
            ends.append(end)
            previous = end
            remaining_after -= 1
        return tuple(ends)

    def _phase_ranges(self, num_ctrl_steps):
        approach, sweep = self._phase_boundaries(num_ctrl_steps)
        count = int(num_ctrl_steps)
        return {
            "approach": (0, approach),
            "sweep": (approach, sweep),
            "lift": (sweep, count),
        }

    def _stage_for_step(self, i, num_ctrl_steps):
        for name, (start, end) in self._phase_ranges(
            num_ctrl_steps
        ).items():
            if start <= int(i) < end:
                return name
        return "lift"

    def _smooth_phase_progress(self, i, stage_start, stage_end):
        """Return one normalized ease-in/ease-out phase coordinate."""

        steps = max(1, int(stage_end) - int(stage_start))
        s = float(np.clip(
            (int(i) - int(stage_start) + 1) / float(steps),
            0.0,
            1.0,
        ))
        return s * s * (3.0 - 2.0 * s)

    def _nominal_stage_start(self, stage):
        if stage == "approach":
            return np.asarray(self._initial_scoop, dtype=np.float64), 0.0
        if stage == "sweep":
            return self._approach_target(), self._approach_pitch
        if stage == "lift":
            position = np.asarray(
                self._lift_start_position
                if self._lift_start_position is not None
                else [
                    *self._sweep_target_xy(),
                    self._approach_target()[2],
                ],
                dtype=np.float64,
            )
            return position, self._sweep_pitch
        raise ValueError("unsupported scoop stage {!r}".format(stage))

    def _phase_start_pose(self, stage):
        index = self._STAGES.index(stage)
        if index > 0:
            previous = self._STAGES[index - 1]
            if previous in self._stage_endpoint_points:
                return (
                    np.asarray(
                        self._stage_endpoint_points[previous], dtype=np.float64
                    ),
                    float(
                        self._stage_boundary_pitches.get(
                            previous, self._nominal_stage_start(stage)[1]
                        )
                    ),
                )
        return self._nominal_stage_start(stage)

    def _phase_start_balls(self, stage):
        index = self._STAGES.index(stage)
        if index > 0:
            previous = self._STAGES[index - 1]
            if previous in self._stage_endpoint_balls:
                return np.asarray(
                    self._stage_endpoint_balls[previous], dtype=np.float64
                )
        return np.asarray(self._initial_balls, dtype=np.float64)

    def _phase_goal_pose(self, stage, start_position):
        if stage == "approach":
            return self._approach_target(), self._approach_pitch
        if stage == "sweep":
            return np.asarray(
                [*self._sweep_target_xy(), self._approach_target()[2]],
                dtype=np.float64,
            ), self._sweep_pitch
        if stage == "lift":
            target = np.asarray(start_position, dtype=np.float64).copy()
            target[2] = (
                self._source_support_plane_z()
                + self._lift_height_above_source_support_radii
                * self._ball_radius
            )
            return target, self._carry_pitch
        raise ValueError("unsupported scoop stage {!r}".format(stage))

    def _phase_pose_metrics(
        self,
        stage,
        i,
        num_ctrl_steps,
        scoop,
        pitch,
    ):
        stage_start, stage_end = self._phase_ranges(num_ctrl_steps)[stage]
        start_position, start_pitch = self._phase_start_pose(stage)
        goal_position, goal_pitch = self._phase_goal_pose(
            stage, start_position
        )
        # Every stage tracks the same public function-group marker.  In
        # particular, Lift keeps its xy coordinates and changes only height;
        # it never switches the trajectory to the Handle root.
        measured_position = np.asarray(scoop, dtype=np.float64)
        reference_end = stage_end
        if stage == "lift":
            # Rise immediately and linearly through the complete Lift phase,
            # reaching the target only at the final control knot. Acquire the
            # general carry attitude with a monotone ease-out over the same
            # interval, placing stabilizing pitch ahead of the rise without a
            # hand-authored path or a separate hold phase.
            steps = max(1, stage_end - stage_start)
            phase = float(np.clip(
                (int(i) - stage_start + 1) / float(steps),
                0.0,
                1.0,
            ))
            motion_phase = phase
            position_progress = motion_phase
            angle_progress = motion_phase * (2.0 - motion_phase)
        else:
            position_progress = self._smooth_phase_progress(
                i, stage_start, reference_end
            )
            angle_progress = position_progress
        reference_position = (
            start_position
            + position_progress * (goal_position - start_position)
        )
        reference_pitch = start_pitch + angle_progress * (
            goal_pitch - start_pitch
        )

        if stage == "approach":
            tolerance = np.asarray(
                [
                    self._approach_position_tolerance_radii * self._ball_radius,
                    self._approach_position_tolerance_radii * self._ball_radius,
                    self._function_height_tolerance_radii * self._ball_radius,
                ]
            )
            position_weight = np.ones(3)
            angle_tolerance = self._approach_angle_tolerance
            angle_weight = 1.0
        elif stage == "sweep":
            tolerance = np.asarray(
                [
                    self._sweep_position_tolerance_radii * self._ball_radius,
                    self._sweep_position_tolerance_radii * self._ball_radius,
                    self._function_height_tolerance_radii * self._ball_radius,
                ]
            )
            position_weight = np.ones(3)
            angle_tolerance = self._sweep_angle_tolerance
            angle_weight = self._sweep_angle_weight
        elif stage == "lift":
            tolerance = np.asarray(
                [
                    self._lift_horizontal_position_tolerance_radii,
                    self._lift_horizontal_position_tolerance_radii,
                    self._lift_height_loss_scale_radii,
                ]
            ) * self._ball_radius
            position_weight = np.asarray(
                [
                    self._lift_horizontal_position_weight,
                    self._lift_horizontal_position_weight,
                    self._lift_height_weight,
                ]
            )
            # Lift acquires the general carry attitude while continuing to
            # track the same public function-group marker.
            angle_tolerance = self._carry_angle_loss_scale
            angle_weight = self._lift_angle_weight
        else:
            raise ValueError("unsupported scoop stage {!r}".format(stage))

        position_delta = measured_position - reference_position
        angle_delta = float(pitch - reference_pitch)
        if stage == "lift":
            # Lift defines lower bounds, so overshoot has no opposing gradient.
            position_delta = position_delta.copy()
            position_delta[2] = min(0.0, position_delta[2])
            angle_delta = min(0.0, angle_delta)
        return {
            "start_position": np.asarray(start_position, dtype=np.float64),
            "measured_position": measured_position.copy(),
            "reference_position": reference_position,
            "position_delta": position_delta,
            "position_tolerance": tolerance,
            "position_weight": position_weight,
            "angle_delta": angle_delta,
            "angle_tolerance": float(angle_tolerance),
            "angle_weight": float(angle_weight),
            "goal_position": goal_position,
            "goal_pitch": float(goal_pitch),
            "progress": float(position_progress),
            "angle_progress": float(angle_progress),
        }

    def _phase_loss_scale(
        self,
        stage_steps,
        endpoint,
        running_weight=1.0,
    ):
        scale = float(running_weight) / float(stage_steps)
        if endpoint:
            scale += self._phase_endpoint_weight
        return scale

    def _phase_pose_loss(
        self,
        metrics,
        stage_steps,
        endpoint,
    ):
        # Running tracking governs the complete motion; the shared terminal
        # term makes every phase finish at its final reference state.
        scale = self._phase_loss_scale(
            stage_steps,
            endpoint,
            self._phase_reference_weight,
        )
        position_loss = float(np.sum(
            metrics["position_weight"]
            * (metrics["position_delta"] / metrics["position_tolerance"]) ** 2
        ))
        angle_loss = (
            metrics["angle_weight"]
            * metrics["angle_delta"] ** 2
            / metrics["angle_tolerance"] ** 2
        )
        return scale * (position_loss + angle_loss)

    def _record_phase_motion(self, stage, i, scoop, pitch):
        """Record terminal motion for physical gates without adding a loss."""

        sequential = (
            self._prev_phase_step == int(i) - 1
            and self._prev_phase_stage == stage
        )
        record = None
        if sequential:
            record = {
                "position_error": (
                    np.asarray(scoop, dtype=np.float64)
                    - self._prev_phase_marker
                ),
                "pitch_error": float(pitch - self._prev_phase_pitch),
            }
        self._phase_motion_by_step[int(i)] = record
        self._prev_phase_marker = np.asarray(scoop, dtype=np.float64).copy()
        self._prev_phase_pitch = float(pitch)
        self._prev_phase_step = int(i)
        self._prev_phase_stage = stage

    def _lift_motion_metrics(self, i, scoop):
        """Measure candidate-independent Lift motion between public states."""

        sequential = self._prev_phase_step == int(i) - 1
        valid_boundary = (
            self._prev_phase_stage in ("sweep", "lift")
            and self._prev_phase_marker is not None
        )
        if not sequential or not valid_boundary:
            return {
                "previous_step": None,
                "delta": np.zeros(3, dtype=np.float64),
                "lateral_motion": 0.0,
                "sweep_backtrack": 0.0,
                "vertical_backtrack": 0.0,
            }

        delta = (
            np.asarray(scoop, dtype=np.float64)
            - np.asarray(self._prev_phase_marker, dtype=np.float64)
        )
        lateral_motion = float(np.linalg.norm(delta[:2]))
        sweep_direction = self._sweep_direction()
        sweep_progress = float(np.dot(delta[:2], sweep_direction))
        sweep_backtrack = max(0.0, -sweep_progress)
        vertical_backtrack = max(0.0, -float(delta[2]))
        return {
            "previous_step": int(i) - 1,
            "delta": delta,
            "lateral_motion": lateral_motion,
            "sweep_backtrack": sweep_backtrack,
            "vertical_backtrack": vertical_backtrack,
        }

    def _sweep_terminal_motion_diagnostics(self, stage_start, stage_end):
        stage_steps = max(1, int(stage_end) - int(stage_start))
        window = max(
            1,
            int(np.ceil(
                self._sweep_diagnostic_window_fraction * float(stage_steps)
            )),
        )
        records = [
            self._phase_motion_by_step.get(step)
            for step in range(max(int(stage_start) + 1, int(stage_end) - window), int(stage_end))
        ]
        records = [record for record in records if record is not None]
        if not records:
            return {
                "position_motion_max": float("inf"),
                "position_motion_tolerance": float(
                    self._sweep_diagnostic_position_motion_tolerance_radii
                    * self._ball_radius
                ),
                "pitch_motion_max": float("inf"),
                "pitch_motion_tolerance": float(
                    self._sweep_diagnostic_pitch_motion_tolerance
                ),
            }
        return {
            "position_motion_max": float(max(
                np.linalg.norm(record["position_error"])
                for record in records
            )),
            "position_motion_tolerance": float(
                self._sweep_diagnostic_position_motion_tolerance_radii
                * self._ball_radius
            ),
            "pitch_motion_max": float(max(
                abs(record["pitch_error"]) for record in records
            )),
            "pitch_motion_tolerance": float(
                self._sweep_diagnostic_pitch_motion_tolerance
            ),
        }

    def set_optimization_stage(self, stage):
        stage = str(stage).strip().lower()
        if stage not in set(self._STAGES) | {"full"}:
            raise ValueError("unsupported scoop stage {!r}".format(stage))
        self._optimization_stage = stage

    def optimization_stage_schedule(self, maxiter):
        budget = max(0, int(maxiter))
        if budget == 0:
            return ()
        count = len(self._STAGES)
        allocations = np.zeros(count, dtype=np.int64)
        if budget < count:
            allocations[:budget] = 1
        else:
            weights = self._curriculum_stage_weights
            exact = weights / float(np.sum(weights)) * budget
            allocations = np.floor(exact).astype(np.int64)
            remainder = budget - int(np.sum(allocations))
            if remainder:
                order = np.argsort(-(exact - allocations), kind="stable")
                allocations[order[:remainder]] += 1
        return tuple(
            (stage, int(stage_budget))
            for stage, stage_budget in zip(
                self._STAGES, allocations
            )
            if stage_budget > 0
        )

    def optimization_stage_action_window(self, stage, num_ctrl_steps):
        stage = str(stage).strip().lower()
        if stage not in self._STAGES:
            raise ValueError("unsupported scoop stage {!r}".format(stage))
        start, end = self._phase_ranges(num_ctrl_steps)[stage]
        # A stage may prepare a small, candidate-independent support region
        # after its physical boundary, but a later stage never reaches back
        # into an already accepted physical prefix. This is the causal
        # frozen-prefix convention: smooth outgoing handoffs without allowing
        # Sweep to rewrite Approach or Lift to rewrite Sweep.
        count = int(num_ctrl_steps)
        basis_knots = int(
            self._task_config.get(
                "action_trust_temporal_basis_knots", 0
            )
            or 0
        )
        stage_index = self._STAGES.index(stage)
        if 2 <= basis_knots < count and stage_index + 1 < len(self._STAGES):
            support = (
                int(np.ceil((count - 1) / float(basis_knots - 1)))
                + 1
            )
            end = min(count, end + support)
        return start, end

    def optimization_rollout_control_steps(self, num_ctrl_steps):
        if self._optimization_stage == "full":
            return int(num_ctrl_steps)
        # Simulate every knot in the causal outgoing-support window so the
        # prepared handoff is physically validated before acceptance.
        return self.optimization_stage_action_window(
            self._optimization_stage,
            num_ctrl_steps,
        )[1]

    def loss_stage_for_step(self, i, num_ctrl_steps):
        return self._stage_for_step(i, num_ctrl_steps)

    def contact_continuation_scales(self):
        return (1.0,)

    def contact_continuation_weights(self):
        return (1.0,)

    @staticmethod
    def optimization_stage_acceptance_forward_retries():
        return 3

    def _loss_active(self, stage):
        if self._optimization_stage == "full":
            return True
        active_index = self._STAGES.index(self._optimization_stage)
        stage_index = self._STAGES.index(stage)
        # Keep the complete causal prefix active while the shared optimizer
        # adjusts an overlapping handoff window. Earlier losses have no
        # gradient through later controls, but their rollout state supplies
        # the actual public boundary for every downstream stage.
        return stage_index <= active_index

    def _approach_metrics(self, scoop, balls, pitch):
        target = self._approach_target()
        position_tolerance = (
            self._approach_position_tolerance_radii * self._ball_radius
        )
        ball_motion_tolerance = (
            self._approach_ball_motion_tolerance_radii * self._ball_radius
        )
        position_delta = np.asarray(scoop, dtype=np.float64) - target
        height_tolerance = (
            self._function_height_tolerance_radii * self._ball_radius
        )
        height_error = float(abs(position_delta[2]))
        balls = np.asarray(balls, dtype=np.float64)
        below = self._below_ball_bottom_metrics(scoop, balls)
        ball_xy_delta = (
            balls[:, :2] - self._initial_balls[:, :2]
        )
        angle_delta = float(pitch - self._approach_pitch)
        ball_motion = np.linalg.norm(ball_xy_delta, axis=1)
        ball_motion_excess = np.maximum(
            ball_motion - ball_motion_tolerance,
            0.0,
        )
        return {
            "target": target,
            "position_delta": position_delta,
            "position_distance": float(np.linalg.norm(position_delta)),
            "horizontal_distance": float(
                np.linalg.norm(position_delta[:2])
            ),
            "position_tolerance": float(position_tolerance),
            "height_error": height_error,
            "height_tolerance": float(height_tolerance),
            "below": below,
            "angle_delta": angle_delta,
            "angle_tolerance": float(self._approach_angle_tolerance),
            "ball_xy_delta": ball_xy_delta,
            "ball_motion": ball_motion,
            "ball_motion_excess": ball_motion_excess,
            "ball_motion_max": float(np.max(ball_motion)),
            "ball_motion_tolerance": float(ball_motion_tolerance),
        }

    def _sweep_metrics(self, scoop, balls, pitch):
        target_xy = self._sweep_target_xy()
        position_tolerance = (
            self._sweep_position_tolerance_radii * self._ball_radius
        )
        height_target = float(self._approach_target()[2])
        height_tolerance = (
            self._function_height_tolerance_radii * self._ball_radius
        )
        capture_tolerance = (
            self._sweep_capture_distance_radii * self._ball_radius
        )
        loss_capture_tolerance = (
            self._sweep_capture_loss_distance_radii * self._ball_radius
        )
        scoop = np.asarray(scoop, dtype=np.float64)
        balls = np.asarray(balls, dtype=np.float64)
        position_delta_xy = scoop[:2] - target_xy
        height_delta = float(scoop[2] - height_target)
        height_error = float(abs(height_delta))
        below = self._below_ball_bottom_metrics(scoop, balls)
        relative_xy = balls[:, :2] - scoop[None, :2]
        distance = np.linalg.norm(relative_xy, axis=1)
        distance_excess = np.maximum(
            distance - loss_capture_tolerance, 0.0
        )
        angle_delta = float(pitch - self._sweep_pitch)
        captured = (
            (distance <= capture_tolerance)
            & below["satisfied"]
        )
        return {
            "target_xy": target_xy,
            "position_delta_xy": position_delta_xy,
            "position_distance_xy": float(np.linalg.norm(position_delta_xy)),
            "position_tolerance": float(position_tolerance),
            "height_target": height_target,
            "height_delta": height_delta,
            "height_tolerance": float(height_tolerance),
            "height_error": height_error,
            "below": below,
            "angle_delta": angle_delta,
            "angle_tolerance": float(self._sweep_angle_tolerance),
            "relative_xy": relative_xy,
            "distance": distance,
            "distance_excess": distance_excess,
            "capture_tolerance": float(capture_tolerance),
            "loss_capture_tolerance": float(loss_capture_tolerance),
            "captured": captured,
            "captured_count": int(np.count_nonzero(captured)),
        }

    @staticmethod
    def _capture_loss(metrics):
        return float(
            np.sum(
                metrics["distance_excess"] ** 2
                / metrics["loss_capture_tolerance"] ** 2
                + metrics["below"]["violation"] ** 2
                / metrics["height_tolerance"] ** 2
            )
        )

    def _below_ball_bottom_metrics(self, scoop, balls):
        """Measure the public function point below every payload bottom."""

        marker_z = float(np.asarray(scoop, dtype=np.float64)[2])
        balls = np.asarray(balls, dtype=np.float64)
        bottoms = balls[:, 2] - self._ball_radius
        clearance = bottoms - marker_z
        loss_clearance = (
            self._function_below_ball_bottom_target_clearance_radii
            * self._ball_radius
        )
        violation = np.maximum(loss_clearance - clearance, 0.0)
        return {
            "ball_bottom_z": bottoms,
            "clearance": clearance,
            "minimum_clearance": float(np.min(clearance)),
            "required_clearance": 0.0,
            "loss_clearance": float(loss_clearance),
            "violation": violation,
            "satisfied": clearance >= 0.0,
        }

    def _carry_retention_metrics(
        self,
        scoop,
        balls,
        *,
        anchor=None,
    ):
        if anchor is None:
            anchor = self._carry_relative_anchor
        if anchor is None:
            raise RuntimeError("scoop carry requires a stage boundary")
        scoop = np.asarray(scoop, dtype=np.float64)
        balls = np.asarray(balls, dtype=np.float64)
        relative = balls - scoop[None, :]
        drift = relative - np.asarray(anchor, dtype=np.float64)
        drift_distance = np.linalg.norm(drift, axis=1)
        tolerance = (
            self._carry_retention_tolerance_radii * self._ball_radius
        )
        return {
            "relative": relative,
            "drift": drift,
            "gradient_drift": drift,
            "distance": drift_distance,
            "tolerance": float(tolerance),
            "gate_tolerance": float(tolerance),
            "retained_count": int(np.count_nonzero(
                drift_distance <= tolerance
            )),
        }

    def _carry_metrics(
        self,
        scoop,
        balls,
        pitch,
        target,
        target_pitch,
        position_tolerance,
        angle_tolerance,
        retention_anchor=None,
    ):
        position_delta = np.asarray(scoop, dtype=np.float64) - np.asarray(
            target, dtype=np.float64
        )
        return {
            "target": np.asarray(target, dtype=np.float64),
            "position_delta": position_delta,
            "position_distance": float(np.linalg.norm(position_delta)),
            "position_tolerance": float(position_tolerance),
            "angle_delta": float(pitch - target_pitch),
            "angle_tolerance": float(angle_tolerance),
            "retention": self._carry_retention_metrics(
                scoop,
                balls,
                anchor=retention_anchor,
            ),
        }

    def compute_terms(self, i, num_ctrl_steps, u_i, variables, q):
        terms = {"goal": 0.0, "payload": 0.0, "control": 0.0}
        if int(i) == 0:
            self._stage_endpoint_cache = {}
            self._stage_endpoint_points = {}
            self._stage_endpoint_balls = {}
            self._stage_boundary_pitches = {}
            self._lift_start_position = None
            self._carry_relative_anchor = None
            self._prev_control_for_terms = None
            self._prev_term_step = None
            self._control_prev_by_step = {}
            self._control_smooth_weight_by_step = {}
            self._prev_phase_marker = None
            self._prev_phase_pitch = None
            self._prev_phase_step = None
            self._prev_phase_stage = None
            self._phase_motion_by_step = {}
            self._lift_horizontal_drift_by_step = {}
            self._lift_motion_by_step = {}

        stage = self._stage_for_step(i, num_ctrl_steps)
        stage_start, stage_end = self._phase_ranges(num_ctrl_steps)[stage]
        scoop, _, _, balls = self._read_task_points(variables)
        q = np.asarray(q, dtype=np.float64)
        pitch = float(q[self._q_pitch])
        endpoint = int(i) == stage_end - 1
        if endpoint:
            self._stage_endpoint_points[stage] = np.asarray(
                scoop, dtype=np.float64
            ).copy()
            self._stage_endpoint_balls[stage] = np.asarray(
                balls, dtype=np.float64
            ).copy()
            self._stage_boundary_pitches[stage] = pitch
        if stage == "sweep" and endpoint:
            self._lift_start_position = np.asarray(
                scoop, dtype=np.float64
            ).copy()
            self._carry_relative_anchor = (
                np.asarray(balls, dtype=np.float64)
                - np.asarray(scoop, dtype=np.float64)[None, :]
            )
        pose = self._phase_pose_metrics(
            stage,
            i,
            num_ctrl_steps,
            scoop,
            pitch,
        )
        stage_steps = max(1, stage_end - stage_start)
        if self._loss_active(stage):
            terms["goal"] = self._phase_pose_loss(
                pose,
                stage_steps,
                endpoint,
            )
        if stage == "lift":
            self._lift_motion_by_step[int(i)] = self._lift_motion_metrics(
                i, scoop
            )
        self._record_phase_motion(stage, i, scoop, pitch)
        if stage == "lift":
            horizontal_position = self._lift_horizontal_position_metrics(
                scoop
            )
            self._lift_horizontal_drift_by_step[int(i)] = (
                horizontal_position["distance"]
            )

        if self._loss_active(stage) and stage == "approach":
            metrics = self._approach_metrics(scoop, balls, pitch)
            payload_scale = self._phase_loss_scale(stage_steps, endpoint)
            payload_loss = (
                self._approach_ball_motion_weight
                * float(np.sum(metrics["ball_motion_excess"] ** 2))
                / metrics["ball_motion_tolerance"] ** 2
            )
            terms["payload"] = payload_scale * payload_loss
            terms["goal"] += (
                self._phase_loss_scale(
                    stage_steps,
                    endpoint,
                    self._phase_reference_weight,
                )
            ) * (
                float(np.sum(metrics["below"]["violation"] ** 2))
                / metrics["height_tolerance"] ** 2
            )
            if endpoint:
                self._stage_endpoint_cache[stage] = {
                    "horizontal_target_distance": metrics[
                        "horizontal_distance"
                    ],
                    "height_error": float(
                        abs(metrics["position_delta"][2])
                    ),
                    "height_tolerance": metrics["height_tolerance"],
                    "function_below_ball_bottom_min": metrics["below"][
                        "minimum_clearance"
                    ],
                    "function_below_ball_bottom_required": metrics["below"][
                        "required_clearance"
                    ],
                    "position_tolerance": metrics["position_tolerance"],
                    "angle_error_rad": float(abs(metrics["angle_delta"])),
                    "angle_tolerance_rad": metrics["angle_tolerance"],
                    "root_rotation_mode": self._root_rotation_mode,
                    "rotation_deg": float(np.degrees(pitch)),
                    # Compatibility field for existing Scoop reports. For a
                    # searched roll/yaw candidate, rotation_deg is canonical.
                    "pitch_deg": float(np.degrees(pitch)),
                    "source_ball_xy_motion_max": metrics[
                        "ball_motion_max"
                    ],
                    "source_ball_xy_motion_tolerance": metrics[
                        "ball_motion_tolerance"
                    ],
                }

        if self._loss_active(stage) and stage == "sweep":
            metrics = self._sweep_metrics(scoop, balls, pitch)
            payload_scale = self._phase_loss_scale(stage_steps, endpoint)
            terms["payload"] = (
                self._capture_loss(metrics) * payload_scale
            )
            if endpoint:
                terminal_motion = self._sweep_terminal_motion_diagnostics(
                    stage_start, stage_end
                )
                snapshot = self._stage_endpoint_cache.setdefault(
                    "sweep", {}
                )
                snapshot.update(
                    {
                        "horizontal_target_distance": metrics[
                            "position_distance_xy"
                        ],
                        "position_tolerance": (
                            self._sweep_gate_position_tolerance_radii
                            * self._ball_radius
                        ),
                        "height_error": float(abs(metrics["height_delta"])),
                        "height_tolerance": metrics["height_tolerance"],
                        "function_below_ball_bottom": metrics["below"][
                            "clearance"
                        ].tolist(),
                        "function_below_ball_bottom_required": metrics[
                            "below"
                        ]["required_clearance"],
                        "function_below_ball_bottom_tolerance": metrics[
                            "height_tolerance"
                        ],
                        "angle_error_rad": float(abs(metrics["angle_delta"])),
                        "angle_tolerance_rad": (
                            self._sweep_gate_angle_tolerance
                        ),
                        "payload_xy_distance": metrics["distance"].tolist(),
                        "capture_xy_distance_tolerance": metrics[
                            "capture_tolerance"
                        ],
                        "captured_count": metrics["captured_count"],
                        "terminal_motion_max": terminal_motion[
                            "position_motion_max"
                        ],
                        "terminal_motion_tolerance": terminal_motion[
                            "position_motion_tolerance"
                        ],
                        "terminal_pitch_motion_max_rad": terminal_motion[
                            "pitch_motion_max"
                        ],
                        "terminal_pitch_motion_tolerance_rad": terminal_motion[
                            "pitch_motion_tolerance"
                        ],
                    }
                )

        if self._loss_active(stage) and stage == "lift":
            height_tolerance = (
                self._lift_height_loss_scale_radii * self._ball_radius
            )
            metrics = self._carry_metrics(
                scoop,
                balls,
                pitch,
                self._lift_target(),
                self._carry_pitch,
                height_tolerance,
                self._carry_angle_loss_scale,
                retention_anchor=self._carry_relative_anchor,
            )
            payload_clearance, _, required_clearance = (
                self._lift_payload_clearance(balls)
            )
            position_delta = pose["position_delta"]
            horizontal_retention = self._lift_horizontal_retention_metrics(
                scoop, balls
            )
            payload_clearance_metrics = self._lift_payload_clearance_metrics(
                pose, balls
            )
            horizontal_position = self._lift_horizontal_position_metrics(
                scoop
            )
            terms["payload"] = (
                (
                    self._lift_horizontal_retention_weight
                    * horizontal_retention["loss"]
                    + self._lift_payload_clearance_weight
                    * payload_clearance_metrics["loss"]
                )
                * self._phase_loss_scale(stage_steps, endpoint)
            )
            if endpoint:
                self._stage_endpoint_cache[stage] = {
                    "height_above_source_support": float(
                        scoop[2] - self._source_support_plane_z()
                    ),
                    "height_target": float(
                        self._lift_height_above_source_support_radii
                        * self._ball_radius
                    ),
                    "height_required": float(
                        self._lift_height_gate_radii
                        * self._ball_radius
                    ),
                    "height_target_shortfall": float(
                        max(0.0, -position_delta[2])
                    ),
                    "height_shortfall": float(max(
                        0.0,
                        self._lift_height_gate_radii * self._ball_radius
                        - (scoop[2] - self._source_support_plane_z()),
                    )),
                    "height_loss_scale": float(height_tolerance),
                    "horizontal_position_drift": horizontal_position[
                        "distance"
                    ],
                    "horizontal_position_drift_max": float(max(
                        (
                            self._lift_horizontal_drift_by_step.get(
                                step, float("inf")
                            )
                            for step in range(stage_start, stage_end)
                        ),
                        default=float("inf"),
                    )),
                    "horizontal_position_tolerance": horizontal_position[
                        "tolerance"
                    ],
                    "horizontal_step_motion_max": float(max(
                        (
                            record["lateral_motion"]
                            for record in self._lift_motion_by_step.values()
                        ),
                        default=float("inf"),
                    )),
                    "vertical_backtrack_total": float(sum(
                        record["vertical_backtrack"]
                        for record in self._lift_motion_by_step.values()
                    )),
                    "sweep_backtrack_total": float(sum(
                        record["sweep_backtrack"]
                        for record in self._lift_motion_by_step.values()
                    )),
                    "pitch_shortfall_rad": float(max(0.0, -pose["angle_delta"])),
                    "pitch_rad": float(pitch),
                    "pitch_required_rad": float(self._carry_pitch),
                    "payload_relative_drift": metrics["retention"][
                        "distance"
                    ].tolist(),
                    "payload_clearance": payload_clearance.tolist(),
                    "payload_clearance_required": float(required_clearance),
                    "retention_distance_tolerance": metrics["retention"][
                        "gate_tolerance"
                    ],
                    "retained_count": metrics["retention"][
                        "retained_count"
                    ],
                }

        if len(u_i):
            control = np.asarray(u_i, dtype=np.float64)
            sequential = self._prev_term_step == int(i) - 1
            previous = self._prev_control_for_terms if sequential else None
            if self._loss_active(stage):
                self._control_prev_by_step[int(i)] = (
                    None
                    if previous is None
                    else np.asarray(previous, dtype=np.float64).copy()
                )
                smooth_weight = self._control_smooth_weight
                self._control_smooth_weight_by_step[int(i)] = float(
                    smooth_weight
                )
                if previous is not None:
                    scale = self.action_scale(len(control))
                    delta = (control - previous) / scale
                    terms["control"] = smooth_weight * float(
                        np.mean(delta * delta)
                    )
            self._prev_control_for_terms = control.copy()
            self._prev_term_step = int(i)
        return terms

    @staticmethod
    def _metric_for_pair(contact_metrics, pair):
        body1, body2 = pair
        for metric in contact_metrics:
            if {str(metric.body1), str(metric.body2)} == {
                str(body1), str(body2)
            }:
                return metric
        raise ValueError(
            "Missing requested physical contact metric for {!r}".format(pair)
        )

    def contact_metric_requests(self):
        pairs = tuple(
            pair
            for payload_pairs in self._payload_contact_pairs
            for pair in payload_pairs
        )
        return tuple(
            (body1, body2, "activation")
            for body1, body2 in dict.fromkeys(pairs)
        )

    def _payload_contact_metrics(
        self,
        contact_metrics,
        payload_centers=None,
        support_direction=None,
    ):
        activations = []
        touching = []
        supported = []
        maximizing_pairs = []
        centers = (
            None
            if payload_centers is None
            else np.asarray(payload_centers, dtype=np.float64)
        )
        direction = np.asarray(
            [0.0, 0.0, 1.0]
            if support_direction is None
            else support_direction,
            dtype=np.float64,
        )
        direction_norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or direction_norm <= 1.0e-12:
            raise ValueError(
                "scoop support direction must be a nonzero 3-vector"
            )
        direction = direction / direction_norm
        for payload_index, payload_pairs in enumerate(
            self._payload_contact_pairs
        ):
            metrics = tuple(
                self._metric_for_pair(contact_metrics, pair)
                for pair in payload_pairs
            )
            values = np.asarray(
                [float(metric.activation) for metric in metrics],
                dtype=np.float64,
            )
            maximum = float(np.max(values))
            activations.append(maximum)
            touching.append(
                any(
                    bool(metric.geometrically_touching)
                    for metric in metrics
                )
            )
            support = False
            if centers is not None:
                center = centers[payload_index]
                for metric in metrics:
                    if not bool(metric.geometrically_touching):
                        continue
                    for position, normal in zip(
                        metric.world_positions, metric.normals
                    ):
                        position = np.asarray(position, dtype=np.float64)
                        normal = np.asarray(normal, dtype=np.float64)
                        if (
                            float(np.dot(center - position, direction)) > 0.0
                            and float(np.dot(normal, direction)) < 0.0
                        ):
                            support = True
                            break
                    if support:
                        break
            supported.append(support)
            maximizing_pairs.append(
                tuple(
                    pair
                    for pair, value in zip(payload_pairs, values)
                    if np.isclose(
                        value, maximum, rtol=0.0, atol=1.0e-12
                    )
                )
            )
        return {
            "activation": np.asarray(activations, dtype=np.float64),
            "touching": np.asarray(touching, dtype=bool),
            "supported": np.asarray(supported, dtype=bool),
            "maximizing_pairs": tuple(maximizing_pairs),
        }

    def _lift_carry_gate(
        self, inside_samples, scoop_samples, ball_samples,
        lift_start_scoop, lift_start_balls, direction,
    ):
        """Require in-range payload ascent whenever the scoop is rising."""
        inside = np.asarray(inside_samples, dtype=bool)
        scoops = np.asarray(scoop_samples, dtype=np.float64)
        balls = np.asarray(ball_samples, dtype=np.float64)
        count = len(scoops)
        expected_balls = self._required_ball_count
        if (
            inside.shape != (count, expected_balls)
            or scoops.shape != (count, 3)
            or balls.shape != (count, expected_balls, 3)
        ):
            raise ValueError("scoop Lift carry samples have inconsistent shapes")
        if count == 0:
            return {
                "accepted": False,
                "sample_count": 0,
                "ascending_sample_count": 0,
                "inside_fraction": [],
                "carry_fraction": [],
                "inside_at_end": [False] * expected_balls,
                "min_carry_fraction": self._lift_carry_min_fraction,
            }
        direction = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or norm <= 1.0e-12:
            raise ValueError("scoop Lift direction must be a nonzero 3-vector")
        direction = direction / norm
        start_scoop = np.asarray(lift_start_scoop, dtype=np.float64).reshape(3)
        start_balls = np.asarray(lift_start_balls, dtype=np.float64).reshape(
            expected_balls, 3
        )
        scoop_step = np.diff(
            np.vstack((start_scoop, scoops)), axis=0
        ) @ direction
        payload_step = np.diff(
            np.concatenate((start_balls[None, :, :], balls), axis=0), axis=0
        ) @ direction
        ascending = scoop_step > 0.0
        following = payload_step > 0.0
        carried = inside & following & ascending[:, None]
        ascending_count = int(np.count_nonzero(ascending))
        inside_fraction = np.mean(inside, axis=0)
        carry_fraction = (
            np.sum(carried, axis=0) / float(ascending_count)
            if ascending_count
            else np.zeros(expected_balls, dtype=np.float64)
        )
        inside_at_end = inside[-1]
        accepted = bool(
            ascending_count
            and np.all(inside_fraction >= self._lift_carry_min_fraction)
            and np.all(carry_fraction >= self._lift_carry_min_fraction)
            and np.all(inside_at_end)
        )
        return {
            "accepted": accepted,
            "sample_count": count,
            "ascending_sample_count": ascending_count,
            "inside_fraction": inside_fraction.tolist(),
            "carry_fraction": carry_fraction.tolist(),
            "inside_at_end": inside_at_end.tolist(),
            "min_carry_fraction": self._lift_carry_min_fraction,
        }

    def compute_contact_terms(
        self,
        i,
        num_ctrl_steps,
        u_i,
        variables,
        q,
        contact_metrics,
    ):
        _ = (u_i, q)
        stage = self._stage_for_step(i, num_ctrl_steps)
        if (
            stage not in ("sweep", "lift")
            or not self._loss_active(stage)
            or not self._payload_contact_pairs
        ):
            return {}
        _, stage_end = self._phase_ranges(num_ctrl_steps)[stage]
        endpoint = int(i) == stage_end - 1
        _, _, _, payload_centers = self._read_task_points(variables)
        metrics = self._payload_contact_metrics(
            contact_metrics,
            payload_centers,
            self._lift_direction()
            if stage == "lift" or endpoint
            else None,
        )
        if endpoint:
            snapshot = self._stage_endpoint_cache.setdefault(stage, {})
            snapshot.update(
                {
                    "payload_contact_activation": metrics[
                        "activation"
                    ].tolist(),
                    "payload_contact_touching": metrics[
                        "touching"
                    ].tolist(),
                    "payload_contact_count": int(
                        np.count_nonzero(metrics["touching"])
                    ),
                    "payload_supported": metrics[
                        "supported"
                    ].tolist(),
                    "payload_supported_count": int(
                        np.count_nonzero(metrics["supported"])
                    ),
                }
            )
        return {}

    def contact_metric_objective_grads(
        self,
        i,
        num_ctrl_steps,
        u_i,
        variables,
        q,
        contact_metrics,
        coef,
    ):
        _ = (
            i,
            num_ctrl_steps,
            u_i,
            variables,
            q,
            contact_metrics,
            coef,
        )
        return {}

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
        stage = self._stage_for_step(i, num_ctrl_steps)
        if not self._loss_active(stage):
            return
        stage_start, stage_end = self._phase_ranges(num_ctrl_steps)[stage]
        scoop, _, _, balls = self._read_task_points(variables)
        q = np.asarray(q, dtype=np.float64)
        pitch = float(q[self._q_pitch])
        time_index = (int(i) + 1) * int(sub_steps) - 1
        var_base = time_index * int(ndof_var)
        q_base = time_index * int(ndof_r)

        endpoint = int(i) == stage_end - 1
        stage_steps = max(1, stage_end - stage_start)
        pose = self._phase_pose_metrics(
            stage,
            i,
            num_ctrl_steps,
            scoop,
            pitch,
        )
        marker_start = var_base + self._var_scoop_base
        pose_scale = (
            float(coef["goal"])
            * 2.0
            * self._phase_loss_scale(
                stage_steps,
                endpoint,
                self._phase_reference_weight,
            )
        )
        position_gradient = (
            pose_scale
            * pose["position_weight"]
            * pose["position_delta"]
            / pose["position_tolerance"] ** 2
        )
        df_dvar[marker_start : marker_start + 3] += position_gradient
        df_dq[q_base + self._q_pitch] += (
            pose_scale
            * pose["angle_weight"]
            * pose["angle_delta"]
            / pose["angle_tolerance"] ** 2
        )

        if stage == "approach":
            metrics = self._approach_metrics(scoop, balls, pitch)
            below_scale = (
                float(coef["goal"])
                * 2.0
                * self._phase_loss_scale(
                    stage_steps,
                    endpoint,
                    self._phase_reference_weight,
                )
                / metrics["height_tolerance"] ** 2
            )
            for ball_base, violation in zip(
                self._var_ball_bases,
                metrics["below"]["violation"],
            ):
                if violation <= 0.0:
                    continue
                df_dvar[marker_start + 2] += below_scale * violation
                df_dvar[var_base + ball_base + 2] -= (
                    below_scale * violation
                )

            payload_scale = (
                float(coef["payload"])
                * self._approach_ball_motion_weight
                * 2.0
                * self._phase_loss_scale(stage_steps, endpoint)
            )
            for ball_base, delta, motion, excess in zip(
                self._var_ball_bases,
                metrics["ball_xy_delta"],
                metrics["ball_motion"],
                metrics["ball_motion_excess"],
            ):
                start = var_base + ball_base
                if excess > 0.0 and motion > 0.0:
                    df_dvar[start : start + 2] += (
                        payload_scale
                        * excess
                        * delta
                        / motion
                        / metrics["ball_motion_tolerance"] ** 2
                    )

        if stage == "sweep":
            metrics = self._sweep_metrics(scoop, balls, pitch)
            capture_scale = self._phase_loss_scale(stage_steps, endpoint)
            payload_scale = (
                float(coef["payload"])
                * 2.0
                * capture_scale
            )
            marker_gradient = np.zeros(2, dtype=np.float64)
            for ball_base, relative, distance, distance_excess in zip(
                self._var_ball_bases,
                metrics["relative_xy"],
                metrics["distance"],
                metrics["distance_excess"],
            ):
                ball_gradient = np.zeros(2, dtype=np.float64)
                if distance_excess > 0.0 and distance > 0.0:
                    distance_gradient = (
                        payload_scale
                        * distance_excess
                        * relative
                        / distance
                        / metrics["loss_capture_tolerance"] ** 2
                    )
                    ball_gradient += distance_gradient
                    marker_gradient -= distance_gradient
                start = var_base + ball_base
                df_dvar[start : start + 2] += ball_gradient
            df_dvar[marker_start : marker_start + 2] += marker_gradient
            below_scale = (
                payload_scale / metrics["height_tolerance"] ** 2
            )
            for ball_base, violation in zip(
                self._var_ball_bases, metrics["below"]["violation"]
            ):
                if violation <= 0.0:
                    continue
                df_dvar[marker_start + 2] += below_scale * violation
                df_dvar[var_base + ball_base + 2] -= below_scale * violation

        if stage == "lift":
            retention = self._lift_horizontal_retention_metrics(scoop, balls)
            retention_scale = (
                float(coef["payload"])
                * 2.0
                * self._lift_horizontal_retention_weight
                * self._phase_loss_scale(stage_steps, endpoint)
                / retention["tolerance"] ** 2
            )
            marker_gradient = np.zeros(2, dtype=np.float64)
            for ball_base, drift in zip(
                self._var_ball_bases, retention["drift"]
            ):
                gradient = retention_scale * drift
                start = var_base + ball_base
                df_dvar[start : start + 2] += gradient
                marker_gradient -= gradient
            df_dvar[marker_start : marker_start + 2] += marker_gradient

            clearance = self._lift_payload_clearance_metrics(pose, balls)
            clearance_scale = (
                float(coef["payload"])
                * 2.0
                * self._lift_payload_clearance_weight
                * self._phase_loss_scale(stage_steps, endpoint)
                / clearance["required"] ** 2
            )
            for ball_base, deficit in zip(
                self._var_ball_bases, clearance["deficit"]
            ):
                if deficit <= 0.0:
                    continue
                df_dvar[var_base + ball_base + 2] -= (
                    clearance_scale * deficit
                )

        if int(ndof_u) > 0 and len(u_i):
            control = np.asarray(u_i, dtype=np.float64)
            u_base = int(i) * int(sub_steps) * int(ndof_u)
            previous = self._control_prev_by_step.get(int(i))
            if previous is not None:
                scale = self.action_scale(ndof_u)
                delta = control - previous
                smooth_weight = self._control_smooth_weight_by_step.get(
                    int(i), self._control_smooth_weight
                )
                smooth_scale = (
                    float(coef["control"])
                    * smooth_weight
                    * 2.0
                    / float(ndof_u)
                    / scale ** 2
                )
                df_du[u_base : u_base + ndof_u] += smooth_scale * delta
                previous_u_base = (
                    (int(i) - 1) * int(sub_steps) * int(ndof_u)
                )
                if previous_u_base >= 0:
                    df_du[
                        previous_u_base : previous_u_base + ndof_u
                    ] -= smooth_scale * delta

    def optimization_stage_acceptance(self, stage):
        stage = str(stage).strip().lower()
        if stage not in self._STAGES:
            return {"accepted": False, "stage": stage, "reason": "unknown_stage"}
        snapshot = dict(self._stage_endpoint_cache.get(stage, {}))
        if not snapshot:
            return {
                "accepted": False,
                "stage": stage,
                "reason": "missing_stage_endpoint",
            }
        if stage == "approach":
            accepted = (
                snapshot["horizontal_target_distance"]
                <= snapshot["position_tolerance"]
                and snapshot["height_error"]
                <= snapshot["height_tolerance"]
                and snapshot["function_below_ball_bottom_min"]
                >= snapshot["function_below_ball_bottom_required"]
                and snapshot["angle_error_rad"]
                <= snapshot["angle_tolerance_rad"]
                and snapshot["source_ball_xy_motion_max"]
                <= snapshot["source_ball_xy_motion_tolerance"]
            )
        elif stage == "sweep":
            accepted = (
                snapshot["horizontal_target_distance"]
                <= snapshot["position_tolerance"]
                and snapshot["height_error"]
                <= snapshot["height_tolerance"]
                and min(snapshot["function_below_ball_bottom"])
                >= -snapshot["function_below_ball_bottom_tolerance"]
                and snapshot["angle_error_rad"]
                <= snapshot["angle_tolerance_rad"]
                and max(snapshot["payload_xy_distance"])
                <= snapshot["capture_xy_distance_tolerance"]
                and snapshot["captured_count"]
                >= self._required_ball_count
            )
        elif stage == "lift":
            accepted = (
                snapshot["height_above_source_support"]
                >= snapshot["height_required"]
                and snapshot["horizontal_position_drift"]
                <= snapshot["horizontal_position_tolerance"]
                and min(snapshot["payload_clearance"])
                >= snapshot["payload_clearance_required"]
                and snapshot["pitch_rad"]
                >= snapshot["pitch_required_rad"]
                and snapshot["retained_count"]
                >= self._required_ball_count
            )
        else:
            raise AssertionError("unreachable Scoop stage")
        return {"accepted": bool(accepted), "stage": stage, **snapshot}

    def init_design(self, model_path, sim):
        from bilevel.parameterization import build_design_bundle

        self._configure_variable_layout(model_path)
        config = {
            "optimize_finger_design": False,
            "generic_design_protocol": "connected_direct_planar_hexahedron",
        }
        config.update(self._task_config)
        bundle = build_design_bundle(model_path, sim, config)
        self._design_bundle = bundle
        self._spec = bundle.spec
        bundle.apply(sim, bundle.init_cage_params, generate_mesh=False)
        return bundle

    def bounds(self, ndof_u, num_ctrl_steps, ndof_cage, optimize_design):
        action_bounds = [
            (-self._action_bound, self._action_bound)
        ] * (int(ndof_u) * int(num_ctrl_steps))
        if not optimize_design:
            return action_bounds
        from bilevel.parameterization import cage_bounds_for_bundle

        bundle = getattr(self, "_design_bundle", None)
        if bundle is None:
            raise RuntimeError("init_design must run before morphology bounds")
        margin = float(self._task_config["morphology_cage_bound"])
        return action_bounds + cage_bounds_for_bundle(
            bundle,
            ndof_cage,
            optimize_finger_design=False,
            handle_margin=margin,
        )

    def rollout_diagnostics(self, runner, params):
        """Replay the public equations and per-physics-step Lift carry."""
        import redmax_py

        replay_task = type(self)(
            num_steps=self._num_steps,
            sub_steps=self._sub_steps,
            task_config=self._task_config,
        )
        replay_sim = redmax_py.Simulation(str(runner.model_path), False)
        replay_runner = type(runner)(
            replay_sim,
            replay_task,
            args=runner.args,
            model_path=str(runner.model_path),
            visualize=False,
            optimize_design=bool(runner.optimize_design),
            morphology_parameterization=(
                runner.morphology_parameterization_id
                if runner.optimize_design
                else None
            ),
        )
        action, morphology = replay_runner.unpack_params(params)
        if (
            replay_runner.optimize_design
            and replay_runner.design_bundle is not None
            and morphology is not None
        ):
            replay_runner.apply_morphology(morphology, generate_mesh=False)
        replay_runner.sim.reset()
        replay_runner._reset_staged_motion_stop_runtime()
        replay_task.set_optimization_stage("full")
        controls = replay_runner.controls_from_action(action)
        ranges = replay_task._phase_ranges(replay_runner.num_ctrl_steps)
        endpoints = {name: end - 1 for name, (_, end) in ranges.items()}
        endpoint_names = {endpoint: name for name, endpoint in endpoints.items()}
        contact_filters = replay_runner._normalize_contact_metric_requests(
            replay_task.contact_metric_requests()
        )
        snapshots = {}
        final_balls = None
        final_pitch = 0.0
        lift_inside_samples = []
        lift_scoop_samples = []
        lift_ball_samples = []
        lift_start_scoop = None
        lift_start_balls = None
        lift_direction = None
        lift_start = ranges["lift"][0]
        from .retention import (
            current_cuboid_gate, function_group_cuboid, terminal_cuboid_gate,
        )
        function_cuboid = function_group_cuboid(replay_task, replay_runner)

        for index in range(replay_runner.num_ctrl_steps):
            control = controls[
                index * replay_runner.ndof_u
                : (index + 1) * replay_runner.ndof_u
            ]
            contact_metrics = None
            for _ in range(replay_runner.sub_steps):
                replay_runner.advance_control_step(
                    index,
                    control,
                    backward_flag=False,
                    verbose=False,
                    num_sub_steps=1,
                )
                if index >= lift_start:
                    variables = replay_runner.sim.get_variables()
                    _, _, _, balls = replay_task._read_task_points(variables)
                    range_gate = current_cuboid_gate(
                        replay_task, replay_runner, balls, function_cuboid
                    )
                    lift_inside_samples.append(range_gate["payload_inside"])
                    lift_scoop_samples.append(np.asarray(
                        replay_task._read_task_points(variables)[0],
                        dtype=np.float64,
                    ))
                    lift_ball_samples.append(np.asarray(balls, dtype=np.float64))
            variables = replay_runner.sim.get_variables()
            q = np.asarray(replay_runner.sim.get_q(), dtype=np.float64)
            if contact_metrics is None:
                contact_metrics = replay_runner._contact_metrics_for_step(
                    contact_filters, derivatives=False
                )
            replay_task.compute_terms(
                index,
                replay_runner.num_ctrl_steps,
                control,
                variables,
                q,
            )
            replay_task.compute_contact_terms(
                index,
                replay_runner.num_ctrl_steps,
                control,
                variables,
                q,
                contact_metrics,
            )
            scoop, _, _, balls = replay_task._read_task_points(variables)
            pitch = float(q[replay_task._q_pitch])
            final_balls = np.asarray(balls, dtype=np.float64)
            final_pitch = pitch
            if index == lift_start - 1:
                lift_start_scoop = np.asarray(scoop, dtype=np.float64).copy()
                lift_start_balls = np.asarray(balls, dtype=np.float64).copy()
                lift_direction = replay_task._lift_direction()
            if index in endpoint_names:
                stage = endpoint_names[index]
                gate = replay_task.optimization_stage_acceptance(stage)
                snapshots[stage] = {
                    "control_step": int(index),
                    "scoop_endeffector": np.asarray(
                        scoop, dtype=np.float64
                    ).tolist(),
                    "root_translation": np.asarray(
                        q[replay_task._q_root_translation], dtype=np.float64
                    ).tolist(),
                    "root_rotation_mode": (
                        replay_task._root_rotation_mode
                    ),
                    "rotation_deg": float(np.degrees(pitch)),
                    "pitch_deg": float(np.degrees(pitch)),
                    "balls": final_balls.tolist(),
                    **gate,
                }

        stage_gates = {
            stage: replay_task.optimization_stage_acceptance(stage)
            for stage in replay_task._STAGES
        }
        retention_gate = terminal_cuboid_gate(
            replay_task, replay_runner, final_balls
        )
        lift_carry_gate = replay_task._lift_carry_gate(
            lift_inside_samples, lift_scoop_samples, lift_ball_samples,
            lift_start_scoop, lift_start_balls, lift_direction,
        )
        task_success = bool(
            lift_carry_gate["accepted"]
            and stage_gates
            and all(gate["accepted"] for gate in stage_gates.values())
        )
        return {
            "stages": list(replay_task._STAGES),
            "phase_snapshots": snapshots,
            "stage_gates": stage_gates,
            "required_ball_count": int(replay_task._required_ball_count),
            "root_rotation_mode": replay_task._root_rotation_mode,
            "final_rotation_deg": float(np.degrees(final_pitch)),
            "final_pitch_deg": float(np.degrees(final_pitch)),
            "final_balls": (
                [] if final_balls is None else final_balls.tolist()
            ),
            "lift_gate_success": bool(
                stage_gates.get("lift", {}).get("accepted", False)
            ),
            "lift_carry_gate": lift_carry_gate,
            "lift_carry_success": bool(lift_carry_gate["accepted"]),
            "terminal_retention_gate": retention_gate,
            "terminal_retention_success": bool(retention_gate["accepted"]),
            "task_success": task_success,
        }







__all__ = ["MISSION_NAME", "TaskDynamics", "TaskObjective"]
