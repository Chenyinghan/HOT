"""Numerical objective for the canonical sweep task.

The task has one searched function, ``sweep``.  Its public Handle actuation is
the same compositional joint representation used by the other canonical
tasks: world xyz translation followed by one Handle-centred revolute joint.
For this scene the revolute axis is world +Y, so it produces the useful
forward/backward sweep swing rather than Handle-axis roll.

The loss is state-only: it uses relative working-face/object geometry,
one-directional physical progress, object cohesion, and distance to the
radius-aware success set.  It contains no action trajectory, absolute marker
trajectory, time-indexed object path, Head topology, or leaf-specific rule.
RedMax contact remains the only mechanism that can move the balls.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from bilevel.runner import BaseTask, print_info as _print_info
from tasks.objective import TaskObjective


MISSION_NAME = "sweep_balls"


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class _SweepMonotonicityTracker:
    """Rollout diagnostic for how one-directional a sweep trajectory is.

    This only reports; it never contributes to the loss or to the success
    condition.  Success remains "every ball sits inside the radius-aware
    target_container safe region for the whole hold window"; physical monotonicity is
    shaped by the ``sweep_backtrack`` loss term.
    """

    def __init__(self, sweep_axis: np.ndarray) -> None:
        self._axis = np.asarray(sweep_axis, dtype=np.float64)
        self._previous = {}
        self._first = {}
        self._last = {}
        self._max_retreat = {}
        self._total_retreat = {}

    def _update(self, name: str, progress: float) -> None:
        previous = self._previous.get(name)
        if previous is None:
            self._first[name] = float(progress)
        else:
            retreat = max(0.0, float(previous) - float(progress))
            self._max_retreat[name] = max(
                self._max_retreat.get(name, 0.0),
                retreat,
            )
            self._total_retreat[name] = (
                self._total_retreat.get(name, 0.0) + retreat
            )
        self._previous[name] = float(progress)
        self._last[name] = float(progress)

    def observe(self, p_sweep, control) -> None:
        self._update(
            "sweep",
            float(np.dot(np.asarray(p_sweep, dtype=np.float64), self._axis)),
        )
        translation = np.asarray(control, dtype=np.float64)[:3]
        self._update(
            "control",
            float(np.dot(translation, self._axis)),
        )

    def summary(self) -> dict:
        report = {"sweep_axis": self._axis.tolist()}
        for name in ("sweep", "control"):
            first = self._first.get(name)
            last = self._last.get(name)
            report[f"{name}_max_step_retreat"] = float(
                self._max_retreat.get(name, 0.0)
            )
            report[f"{name}_total_retreat"] = float(
                self._total_retreat.get(name, 0.0)
            )
            report[f"{name}_net_progress"] = (
                0.0
                if first is None or last is None
                else float(last - first)
            )
        return report


class TaskDynamics(BaseTask):
    """Causal state-only sweep objective with a terminal success hold."""

    has_finger_design = False
    optimize_design = False

    def __init__(
        self,
        *,
        num_steps=1800,
        sub_steps=30,
        coef_sweep_ball_contact=10000.0,
        coef_sweep_backtrack=1000.0,
        coef_ball_cohesion=200.0,
        coef_ball_goal=7000.0,
        coef_swing=5.0,
        coef_control=0.1,
        control_smooth_weight=0.25,
        optimization_prefix_fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
        sweep_axis=(-1.0, 0.0, 0.0),
        sweep_height_offset=-0.6,
        sweep_vertical_scale=0.5,
        sweep_contact_scale=0.6,
        sweep_contact_quartic_weight=0.25,
        sweep_contact_lateral_scale=0.25,
        sweep_contact_min_gap=0.55,
        sweep_contact_max_gap=1.0,
        sweep_contact_lateral_slack=0.25,
        sweep_contact_height_slack=0.1,
        target_container_depth=4.0,
        target_container_half_width=3.0,
        ball_radius=0.6,
        success_safety_margin=0.1,
        success_hold_knots=1,
        terminal_loss_knots=5,
        terminal_goal_weight=0.25,
        sweep_backtrack_scale=0.25,
        ball_goal_lateral_half_width=1.0,
        ball_cohesion_scale=1.2,
        swing_free_deg=5.0,
        swing_scale_deg=25.0,
        terminal_swing_weight=20.0,
        action_scale_x=24.0,
        action_scale_y=4.0,
        action_scale_z=4.0,
        action_scale_swing=0.7853981633974483,
        swing_joint_name="freeform_swing_joint",
        force_connectivity=True,
        generic_design_protocol="connected_direct_planar_hexahedron",
    ):
        self._num_steps = int(num_steps)
        self._sub_steps = int(sub_steps)
        if self._num_steps <= 0 or self._sub_steps <= 0:
            raise ValueError(
                "target_container num_steps and sub_steps must be positive"
            )
        self._coef = {
            "sweep_ball_contact": _finite_nonnegative(
                coef_sweep_ball_contact,
                "coef_sweep_ball_contact",
            ),
            "sweep_backtrack": _finite_nonnegative(
                coef_sweep_backtrack, "coef_sweep_backtrack"
            ),
            "ball_cohesion": _finite_nonnegative(
                coef_ball_cohesion, "coef_ball_cohesion"
            ),
            "ball_goal": _finite_nonnegative(
                coef_ball_goal, "coef_ball_goal"
            ),
            "swing": _finite_nonnegative(coef_swing, "coef_swing"),
            "control": _finite_nonnegative(
                coef_control, "coef_control"
            ),
        }
        self._control_smooth_weight = _finite_nonnegative(
            control_smooth_weight,
            "control_smooth_weight",
        )
        prefix_fractions = tuple(
            float(value) for value in optimization_prefix_fractions
        )
        if (
            not prefix_fractions
            or not np.all(np.isfinite(prefix_fractions))
            or any(not 0.0 < value <= 1.0 for value in prefix_fractions)
            or any(
                second <= first
                for first, second in zip(
                    prefix_fractions,
                    prefix_fractions[1:],
                )
            )
            or prefix_fractions[-1] != 1.0
        ):
            raise ValueError(
                "optimization_prefix_fractions must be strictly increasing "
                "in (0, 1] and end at 1"
            )
        self._optimization_prefix_fractions = prefix_fractions
        self._optimization_stage_names = tuple(
            f"prefix_{index + 1}"
            for index in range(len(prefix_fractions) - 1)
        ) + ("full",)
        self._optimization_stage = "full"

        axis = np.asarray(sweep_axis, dtype=np.float64).reshape(-1)
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or abs(float(axis[2])) > 1e-12
            or float(np.linalg.norm(axis[:2])) <= 1e-12
        ):
            raise ValueError(
                "sweep_axis must be a finite nonzero horizontal 3-vector"
            )
        self._sweep_axis = axis / float(np.linalg.norm(axis))
        self._lateral_axis = np.asarray(
            [
                -self._sweep_axis[1],
                self._sweep_axis[0],
                0.0,
            ],
            dtype=np.float64,
        )
        self._vertical_axis = np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )

        self._sweep_height_offset = float(sweep_height_offset)
        if not np.isfinite(self._sweep_height_offset):
            raise ValueError("sweep_height_offset must be finite")
        self._sweep_vertical_scale = _finite_positive(
            sweep_vertical_scale,
            "sweep_vertical_scale",
        )
        self._sweep_contact_scale = _finite_positive(
            sweep_contact_scale,
            "sweep_contact_scale",
        )
        self._sweep_contact_quartic_weight = _finite_nonnegative(
            sweep_contact_quartic_weight,
            "sweep_contact_quartic_weight",
        )
        self._sweep_contact_lateral_scale = _finite_positive(
            sweep_contact_lateral_scale,
            "sweep_contact_lateral_scale",
        )
        self._sweep_contact_min_gap = _finite_nonnegative(
            sweep_contact_min_gap,
            "sweep_contact_min_gap",
        )
        self._sweep_contact_max_gap = _finite_positive(
            sweep_contact_max_gap,
            "sweep_contact_max_gap",
        )
        self._sweep_contact_lateral_slack = _finite_nonnegative(
            sweep_contact_lateral_slack,
            "sweep_contact_lateral_slack",
        )
        self._sweep_contact_height_slack = _finite_nonnegative(
            sweep_contact_height_slack,
            "sweep_contact_height_slack",
        )

        self._target_container_depth = _finite_positive(
            target_container_depth,
            "target_container_depth",
        )
        self._target_container_half_width = _finite_positive(
            target_container_half_width,
            "target_container_half_width",
        )
        self._ball_radius = _finite_positive(
            ball_radius,
            "ball_radius",
        )
        if self._sweep_contact_min_gap >= self._sweep_contact_max_gap:
            raise ValueError(
                "sweep_contact_min_gap must be below sweep_contact_max_gap"
            )
        self._success_safety_margin = _finite_nonnegative(
            success_safety_margin,
            "success_safety_margin",
        )
        self._goal_margin = (
            self._ball_radius + self._success_safety_margin
        )
        if (
            2.0 * self._goal_margin >= self._target_container_depth
            or self._goal_margin >= self._target_container_half_width
        ):
            raise ValueError(
                "ball radius plus safety margin must leave a nonempty "
                "target_container success region"
            )
        self._success_hold_knots = int(success_hold_knots)
        if self._success_hold_knots <= 0:
            raise ValueError("success_hold_knots must be positive")
        self._terminal_loss_knots = int(terminal_loss_knots)
        if self._terminal_loss_knots <= 0:
            raise ValueError("terminal_loss_knots must be positive")
        self._terminal_goal_weight = _finite_nonnegative(
            terminal_goal_weight,
            "terminal_goal_weight",
        )
        self._sweep_backtrack_scale = _finite_positive(
            sweep_backtrack_scale,
            "sweep_backtrack_scale",
        )
        self._ball_goal_lateral_half_width = _finite_positive(
            ball_goal_lateral_half_width,
            "ball_goal_lateral_half_width",
        )
        if self._ball_goal_lateral_half_width > (
            self._target_container_half_width - self._goal_margin
        ):
            raise ValueError(
                "ball_goal_lateral_half_width must lie inside the "
                "radius-aware target_container safe width"
            )
        self._ball_cohesion_scale = _finite_positive(
            ball_cohesion_scale,
            "ball_cohesion_scale",
        )

        free_angle = np.radians(float(swing_free_deg))
        swing_scale = np.radians(float(swing_scale_deg))
        if (
            not np.isfinite(free_angle)
            or free_angle < 0.0
            or not np.isfinite(swing_scale)
            or swing_scale <= 0.0
        ):
            raise ValueError(
                "swing_free_deg must be finite and nonnegative and "
                "swing_scale_deg must be finite and positive"
            )
        self._swing_free_angle = float(free_angle)
        self._swing_scale = float(swing_scale)
        self._terminal_swing_weight = _finite_nonnegative(
            terminal_swing_weight,
            "terminal_swing_weight",
        )

        self._canonical_action_scale = np.asarray(
            [
                action_scale_x,
                action_scale_y,
                action_scale_z,
                action_scale_swing,
            ],
            dtype=np.float64,
        )
        if (
            self._canonical_action_scale.shape != (4,)
            or not np.all(np.isfinite(self._canonical_action_scale))
            or np.any(self._canonical_action_scale <= 0.0)
        ):
            raise ValueError(
                "target_container four-DOF action scales must be finite and positive"
            )
        self._swing_joint_name = str(swing_joint_name).strip()
        if not self._swing_joint_name:
            raise ValueError("swing_joint_name must be nonempty")
        self._force_connectivity = bool(force_connectivity)
        self._generic_design_protocol = str(generic_design_protocol)
        self._design_bundle = None

        self._q_root_translation = None
        self._q_root_swing = None
        self._root_rotation_mode = "pitch"
        self._root_rotation_axis = np.asarray(
            [0.0, 1.0, 0.0], dtype=np.float64
        )
        self._root_rotation_joint_name = self._swing_joint_name
        self._var_sweep_base = None
        self._var_target_container_base = None
        self._var_ball_bases = ()
        self._required_var_len = 0
        self._reset_rollout_cache()

    def num_steps(self) -> int:
        return self._num_steps

    def sub_steps(self) -> int:
        return self._sub_steps

    def objective_weights(self) -> dict:
        return dict(self._coef)

    def set_optimization_stage(self, stage: str) -> None:
        stage = str(stage).strip().lower()
        if stage not in self._optimization_stage_names:
            raise ValueError(
                f"unsupported sweep optimization stage {stage!r}"
            )
        self._optimization_stage = stage

    def optimization_stage_schedule(self, maxiter: int):
        """Split one public budget over causal prefixes of one loss."""
        budget = max(0, int(maxiter))
        if budget == 0:
            return ()
        count = len(self._optimization_stage_names)
        allocations = np.zeros(count, dtype=np.int64)
        allocations[: min(count, budget)] = 1
        remaining = max(0, budget - count)
        if remaining:
            weights = np.sqrt(
                np.asarray(
                    self._optimization_prefix_fractions,
                    dtype=np.float64,
                )
            )
            exact = remaining * weights / float(np.sum(weights))
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
            (name, int(stage_budget))
            for name, stage_budget in zip(
                self._optimization_stage_names,
                allocations,
            )
            if stage_budget > 0
        )

    def _optimization_prefix_end(self, num_ctrl_steps: int) -> int:
        if self._optimization_stage == "full":
            fraction = 1.0
        else:
            index = self._optimization_stage_names.index(
                self._optimization_stage
            )
            fraction = self._optimization_prefix_fractions[index]
        return min(
            int(num_ctrl_steps),
            max(1, int(np.ceil(fraction * int(num_ctrl_steps)))),
        )

    def _optimization_prefix_start(self, num_ctrl_steps: int) -> int:
        index = self._optimization_stage_names.index(
            self._optimization_stage
        )
        if index < 2:
            return 0
        previous_fraction = self._optimization_prefix_fractions[index - 1]
        return min(
            int(num_ctrl_steps) - 1,
            max(
                1,
                int(np.ceil(previous_fraction * int(num_ctrl_steps))),
            ),
        )

    def optimization_rollout_control_steps(
        self,
        num_ctrl_steps: int,
    ) -> int:
        return self._optimization_prefix_end(num_ctrl_steps)

    def optimization_stage_action_window(
        self,
        stage: str,
        num_ctrl_steps: int,
    ) -> tuple[int, int]:
        self.set_optimization_stage(stage)
        return (
            self._optimization_prefix_start(num_ctrl_steps),
            self._optimization_prefix_end(num_ctrl_steps),
        )

    def action_scale(self, ndof_u: int) -> np.ndarray:
        if int(ndof_u) != 4:
            raise ValueError(
                "target_container requires xyz translation plus one forward-swing "
                f"rotation control (4 controls), got {ndof_u}"
            )
        return self._canonical_action_scale.copy()

    def action_parameterization(self) -> str:
        return "tanh_legacy_v1"

    def init_action(
        self,
        ndof_u: int,
        num_ctrl_steps: int,
        seed: int,
    ) -> np.ndarray:
        _ = seed
        self.action_scale(ndof_u)
        if int(num_ctrl_steps) < 0:
            raise ValueError("num_ctrl_steps must be nonnegative")
        return np.zeros(
            int(num_ctrl_steps) * int(ndof_u),
            dtype=np.float64,
        )

    def init_task(self, sim) -> None:
        if int(sim.ndof_u) != 4:
            raise ValueError(
                "target_container requires xyz translation plus one selected-axis "
                f"motor (4 controls), got {sim.ndof_u}"
            )

    @staticmethod
    def _joint_dof(joint_type: str) -> int:
        dof = {
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
            "free3d-exp-decoupled": 6,
        }.get(str(joint_type).lower())
        if dof is None:
            raise ValueError(f"unknown RedMax joint type {joint_type!r}")
        return int(dof)

    @staticmethod
    def _normalized_axis(joint: ET.Element) -> np.ndarray:
        axis = np.fromstring(
            joint.attrib.get("axis", ""),
            sep=" ",
            dtype=np.float64,
        )
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or float(np.linalg.norm(axis)) <= 1e-12
        ):
            raise ValueError(
                f"joint {joint.attrib.get('name', '')!r} has invalid axis"
            )
        return axis / float(np.linalg.norm(axis))

    def _configure_q_layout(self, root: ET.Element) -> None:
        q_cursor = 0
        root_translation = None
        root_swing = None
        root_rotation_mode = None
        root_rotation_axis = None
        root_rotation_joint_name = None
        root_rotation_count = 0
        rotation_joint_modes = {
            self._swing_joint_name: "pitch",
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
            joint_type = joint.attrib.get("type", "").lower()
            dof = self._joint_dof(joint_type)
            name = joint.attrib.get("name", "")
            if name == "freeform_root_joint":
                if joint_type != "translational" or dof != 3:
                    raise ValueError(
                        "target_container freeform_root_joint must be translational"
                    )
                root_translation = slice(q_cursor, q_cursor + 3)
            elif name in rotation_joint_modes:
                if joint_type != "revolute" or dof != 1:
                    raise ValueError(
                        f"target_container selected rotation joint {name} must be revolute"
                    )
                mode = rotation_joint_modes[name]
                axis = self._normalized_axis(joint)
                if not np.allclose(
                    axis, canonical_axes[mode], rtol=0.0, atol=1e-9
                ):
                    raise ValueError(
                        "target_container selected rotation joint name and axis disagree"
                    )
                root_swing = q_cursor
                root_rotation_mode = mode
                root_rotation_axis = axis
                root_rotation_joint_name = name
                root_rotation_count += 1
            q_cursor += dof

        if root_translation is None or root_swing is None:
            raise ValueError(
                "target_container XML must contain a translational "
                "freeform_root_joint and one canonical rotation joint"
            )
        if root_rotation_count != 1:
            raise ValueError(
                "target_container XML must contain exactly one root rotation DOF "
                "chosen from roll, pitch, or yaw"
            )
        self._q_root_translation = root_translation
        self._q_root_swing = int(root_swing)
        self._root_rotation_mode = str(root_rotation_mode)
        self._root_rotation_axis = np.asarray(
            root_rotation_axis, dtype=np.float64
        )
        self._root_rotation_joint_name = str(root_rotation_joint_name)

    @staticmethod
    def _require_position_motor(
        root: ET.Element,
        joint_name: str,
    ) -> None:
        motors = [
            motor
            for motor in root.iter("motor")
            if motor.attrib.get("joint") == joint_name
        ]
        if len(motors) != 1:
            raise ValueError(
                f"target_container expects exactly one motor for {joint_name!r}"
            )
        if motors[0].attrib.get("ctrl", "").lower() != "position":
            raise ValueError(
                f"target_container motor {joint_name!r} must use position control"
            )

    def _configure_variable_layout(self, root: ET.Element) -> None:
        variable = root.find("variable")
        if variable is None:
            raise ValueError("target_container XML must contain a variable section")

        sweep_idx = None
        target_container_idx = None
        ball_indices = []
        for idx, elem in enumerate(variable.findall("endeffector")):
            joint_name = elem.attrib.get("joint", "")
            if joint_name == "sweep_endeffector":
                sweep_idx = idx
            elif joint_name == "target_container_base_joint":
                target_container_idx = idx
            elif (
                joint_name.startswith("ball")
                and joint_name.endswith("_joint")
            ):
                ball_indices.append(idx)

        if sweep_idx is None or target_container_idx is None or not ball_indices:
            raise ValueError(
                "target_container XML variables must expose sweep_endeffector, "
                "target_container_base_joint, and at least one ball joint"
            )
        self._var_sweep_base = 3 * int(sweep_idx)
        self._var_target_container_base = 3 * int(target_container_idx)
        self._var_ball_bases = tuple(
            3 * int(index) for index in ball_indices
        )
        self._required_var_len = 3 * (
            max([sweep_idx, target_container_idx, *ball_indices]) + 1
        )

    def configure_model(self, model_path: str, sim) -> None:
        _ = sim
        root = ET.parse(model_path).getroot()
        self._configure_q_layout(root)
        self._configure_variable_layout(root)
        self._require_position_motor(root, "freeform_root_joint")
        self._require_position_motor(root, self._root_rotation_joint_name)

    def init_design(self, model_path: str, sim):
        from bilevel.parameterization import build_design_bundle

        self.configure_model(model_path, sim)
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
        self.action_scale(ndof_u)
        bounds = [(-1.0, 1.0)] * (
            int(ndof_u) * int(num_ctrl_steps)
        )
        if optimize_design:
            from bilevel.parameterization import cage_bounds_for_bundle

            bounds += cage_bounds_for_bundle(
                self._design_bundle,
                int(ndof_cage),
                optimize_finger_design=False,
            )
        return bounds

    def _read_task_points(self, variables):
        values = np.asarray(variables, dtype=np.float64)
        if (
            self._var_sweep_base is None
            or self._var_target_container_base is None
            or not self._var_ball_bases
        ):
            raise RuntimeError(
                "target_container model must be configured before evaluating loss"
            )
        if len(values) < self._required_var_len:
            raise ValueError(
                "target_container expects sweep, target_container, and ball variables "
                f"({self._required_var_len} values), got {len(values)}"
            )
        return (
            values[
                self._var_sweep_base : self._var_sweep_base + 3
            ],
            values[
                self._var_target_container_base :
                self._var_target_container_base + 3
            ],
            [
                values[base : base + 3]
                for base in self._var_ball_bases
            ],
        )

    def _sweep_alpha(self, i: int, num_ctrl_steps: int) -> float:
        if int(num_ctrl_steps) <= 0:
            raise ValueError("num_ctrl_steps must be positive")
        return float(int(i) + 1) / float(num_ctrl_steps)

    @staticmethod
    def _running_weight(num_ctrl_steps: int) -> float:
        if int(num_ctrl_steps) <= 0:
            raise ValueError("num_ctrl_steps must be positive")
        return 1.0 / float(num_ctrl_steps)

    def _swing_weight(
        self,
        i: int,
        num_ctrl_steps: int,
    ) -> float:
        weight = self._running_weight(num_ctrl_steps)
        hold_steps = min(
            self._terminal_loss_knots,
            int(num_ctrl_steps),
        )
        if int(i) >= int(num_ctrl_steps) - hold_steps:
            weight += self._terminal_swing_weight / float(hold_steps)
        return float(weight)

    def loss_stage_for_step(
        self,
        i: int,
        num_ctrl_steps: int,
    ) -> str:
        return (
            "terminal_hold"
            if int(i) >= int(num_ctrl_steps) - min(
                self._terminal_loss_knots,
                int(num_ctrl_steps),
            )
            else "running"
        )

    def _rear_ball_index(self, balls: np.ndarray) -> int:
        return int(np.argmin(balls @ self._sweep_axis))

    def _sweep_ball_contact_term_and_grads(
        self,
        p_sweep: np.ndarray,
        p_balls: list[np.ndarray],
        running_weight: float,
    ) -> tuple[
        float,
        np.ndarray,
        list[np.ndarray],
        float,
        float,
        float,
    ]:
        """Keep the public working marker in a relative contact band."""
        sweep = np.asarray(p_sweep, dtype=np.float64)
        balls = np.asarray(p_balls, dtype=np.float64)
        rear_idx = self._rear_ball_index(balls)
        p_rear = np.asarray(p_balls[rear_idx], dtype=np.float64)
        signed_gap = float(
            np.dot(p_rear - sweep, self._sweep_axis)
        )
        longitudinal_residual = self._interval_residual(
            signed_gap,
            self._sweep_contact_min_gap,
            self._sweep_contact_max_gap,
        )
        ball_centroid = np.mean(balls, axis=0)
        lateral_offset = float(
            np.dot(sweep - ball_centroid, self._lateral_axis)
        )
        lateral_residual = self._interval_residual(
            lateral_offset,
            -self._sweep_contact_lateral_slack,
            self._sweep_contact_lateral_slack,
        )
        height_offset = float(
            sweep[2] - ball_centroid[2] - self._sweep_height_offset
        )
        height_residual = self._interval_residual(
            height_offset,
            -self._sweep_contact_height_slack,
            self._sweep_contact_height_slack,
        )

        longitudinal_normalized = (
            longitudinal_residual / self._sweep_contact_scale
        )
        longitudinal_common = float(running_weight) * (
            2.0
            * longitudinal_residual
            / (self._sweep_contact_scale ** 2)
            + 4.0
            * self._sweep_contact_quartic_weight
            * longitudinal_residual ** 3
            / (self._sweep_contact_scale ** 4)
        )
        lateral_common = (
            2.0
            * float(running_weight)
            * lateral_residual
            / (self._sweep_contact_lateral_scale ** 2)
        )
        vertical_common = (
            2.0
            * float(running_weight)
            * height_residual
            / (self._sweep_vertical_scale ** 2)
        )
        rear_grad = longitudinal_common * self._sweep_axis
        sweep_grad = -rear_grad
        sweep_grad += lateral_common * self._lateral_axis
        sweep_grad += vertical_common * self._vertical_axis
        ball_grads = [
            np.zeros(3, dtype=np.float64) for _ in p_balls
        ]
        ball_grads[rear_idx] = rear_grad
        centroid_grad = (
            -lateral_common * self._lateral_axis
            - vertical_common * self._vertical_axis
        ) / float(len(p_balls))
        for index in range(len(ball_grads)):
            ball_grads[index] += centroid_grad
        value = float(running_weight) * (
            longitudinal_normalized ** 2
            + self._sweep_contact_quartic_weight
            * longitudinal_normalized ** 4
            + (
                lateral_residual / self._sweep_contact_lateral_scale
            ) ** 2
            + (height_residual / self._sweep_vertical_scale) ** 2
        )
        return (
            float(value),
            sweep_grad,
            ball_grads,
            signed_gap,
            lateral_offset,
            height_offset,
        )

    def _safe_region_components(
        self,
        p_ball: np.ndarray,
        p_target_container: np.ndarray,
    ) -> tuple[float, float]:
        relative = (
            np.asarray(p_ball, dtype=np.float64)
            - np.asarray(p_target_container, dtype=np.float64)
        )
        depth_coordinate = float(
            np.dot(relative, -self._sweep_axis)
        )
        lateral_coordinate = float(
            np.dot(relative, self._lateral_axis)
        )
        return depth_coordinate, lateral_coordinate

    @staticmethod
    def _interval_residual(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        if value < lower:
            return float(value - lower)
        if value > upper:
            return float(value - upper)
        return 0.0

    def _ball_goal_term_and_grads(
        self,
        p_target_container: np.ndarray,
        p_balls: list[np.ndarray],
        goal_weight: float,
    ) -> tuple[float, list[np.ndarray], np.ndarray]:
        lateral_limit = self._ball_goal_lateral_half_width
        ball_count = float(len(p_balls))
        value = 0.0
        ball_grads = []
        target_container_grad = np.zeros(3, dtype=np.float64)
        for p_ball in p_balls:
            depth, lateral = self._safe_region_components(
                p_ball,
                p_target_container,
            )
            depth_residual = self._interval_residual(
                depth,
                self._goal_margin,
                self._target_container_depth - self._goal_margin,
            )
            lateral_residual = self._interval_residual(
                lateral,
                -lateral_limit,
                lateral_limit,
            )
            value += (
                (depth_residual / self._target_container_depth) ** 2
                + (
                    lateral_residual
                    / self._ball_goal_lateral_half_width
                )
                ** 2
            )
            grad = (
                2.0
                * float(goal_weight)
                / ball_count
                * (
                    depth_residual
                    * (-self._sweep_axis)
                    / (self._target_container_depth ** 2)
                    + lateral_residual
                    * self._lateral_axis
                    / (self._ball_goal_lateral_half_width ** 2)
                )
            )
            ball_grads.append(grad)
            target_container_grad -= grad
        return (
            float(goal_weight) * value / ball_count,
            ball_grads,
            target_container_grad,
        )

    def _ball_cohesion_term_and_grads(
        self,
        p_balls: list[np.ndarray],
        running_weight: float,
    ) -> tuple[float, list[np.ndarray]]:
        """Penalize separation beyond physical ball-to-ball contact."""
        points = [
            np.asarray(point, dtype=np.float64) for point in p_balls
        ]
        gradients = [np.zeros(3, dtype=np.float64) for _ in points]
        pair_count = len(points) * (len(points) - 1) // 2
        if pair_count == 0:
            return 0.0, gradients
        contact_distance = 2.0 * self._ball_radius
        value = 0.0
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                delta = points[first] - points[second]
                distance = float(np.linalg.norm(delta))
                excess = max(0.0, distance - contact_distance)
                if excess == 0.0 or distance <= 1.0e-12:
                    continue
                value += (excess / self._ball_cohesion_scale) ** 2
                common = (
                    2.0
                    * float(running_weight)
                    * excess
                    / (
                        float(pair_count)
                        * self._ball_cohesion_scale ** 2
                        * distance
                    )
                )
                pair_gradient = common * delta
                gradients[first] += pair_gradient
                gradients[second] -= pair_gradient
        return (
            float(running_weight) * value / float(pair_count),
            gradients,
        )

    def _sweep_backtrack_term_and_grads(
        self,
        p_sweep: np.ndarray,
        previous_sweep,
        running_weight: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        zero = np.zeros(3, dtype=np.float64)
        if previous_sweep is None:
            return 0.0, zero.copy(), zero.copy()
        current_progress = float(
            np.dot(
                np.asarray(p_sweep, dtype=np.float64)
                - np.asarray(previous_sweep, dtype=np.float64),
                self._sweep_axis,
            )
        )
        backtrack = max(0.0, -current_progress)
        if backtrack == 0.0:
            return 0.0, zero.copy(), zero.copy()
        common = (
            2.0
            * float(running_weight)
            * backtrack
            / (self._sweep_backtrack_scale ** 2)
        )
        current_grad = -common * self._sweep_axis
        reference_grad = common * self._sweep_axis
        return (
            float(running_weight)
            * (backtrack / self._sweep_backtrack_scale) ** 2,
            current_grad,
            reference_grad,
        )

    def _swing_term_and_grad(
        self,
        theta: float,
        running_weight: float,
    ) -> tuple[float, float]:
        excess = max(0.0, abs(float(theta)) - self._swing_free_angle)
        if excess == 0.0:
            return 0.0, 0.0
        sign = 1.0 if theta > 0.0 else -1.0
        return (
            float(running_weight)
            * (excess / self._swing_scale) ** 2,
            float(running_weight)
            * 2.0
            * excess
            * sign
            / (self._swing_scale ** 2),
        )

    def _ball_is_safe(
        self,
        p_ball: np.ndarray,
        p_target_container: np.ndarray,
    ) -> bool:
        depth, lateral = self._safe_region_components(
            p_ball,
            p_target_container,
        )
        return bool(
            self._goal_margin <= depth
            <= self._target_container_depth - self._goal_margin
            and abs(lateral)
            <= self._target_container_half_width - self._goal_margin
        )

    def _ball_safety_diagnostics(
        self,
        p_ball: np.ndarray,
        p_target_container: np.ndarray,
    ) -> dict:
        """Report signed margins to the existing success region.

        This is diagnostics-only: positive margins are inside the current
        radius-aware box, zero lies on its boundary, and negative values are
        outside. It intentionally reuses the public success geometry without
        changing the loss, gates, or physical rollout.
        """

        depth, lateral = self._safe_region_components(
            p_ball,
            p_target_container,
        )
        depth_lower = self._goal_margin
        depth_upper = self._target_container_depth - self._goal_margin
        lateral_limit = self._target_container_half_width - self._goal_margin
        depth_margin = min(
            depth - depth_lower,
            depth_upper - depth,
        )
        lateral_margin = lateral_limit - abs(lateral)
        minimum_margin = min(depth_margin, lateral_margin)
        return {
            "depth": float(depth),
            "lateral": float(lateral),
            "depth_margin": float(depth_margin),
            "lateral_margin": float(lateral_margin),
            "minimum_margin": float(minimum_margin),
            "is_safe": bool(minimum_margin >= 0.0),
        }

    def _reset_rollout_cache(self) -> None:
        self._previous_control = None
        self._previous_control_by_step = {}
        self._previous_sweep = None
        self._safe_history = []
        self._terminal_cache = {}
        self._step_cache = {}

    def compute_terms(
        self,
        i,
        num_ctrl_steps,
        u_i,
        variables,
        q,
    ):
        if int(i) == 0:
            self._reset_rollout_cache()

        p_sweep, p_target_container, p_balls = self._read_task_points(
            variables
        )
        p_sweep = np.asarray(p_sweep, dtype=np.float64)
        p_target_container = np.asarray(p_target_container, dtype=np.float64)
        p_balls = [
            np.asarray(point, dtype=np.float64)
            for point in p_balls
        ]
        running_weight = self._running_weight(num_ctrl_steps)
        (
            sweep_ball_contact,
            sweep_ball_contact_sweep_grad,
            sweep_ball_contact_ball_grads,
            sweep_ball_signed_gap,
            sweep_ball_lateral_offset,
            sweep_ball_height_offset,
        ) = self._sweep_ball_contact_term_and_grads(
            p_sweep,
            p_balls,
            running_weight,
        )
        (
            sweep_backtrack,
            sweep_backtrack_grad,
            sweep_backtrack_reference_grad,
        ) = self._sweep_backtrack_term_and_grads(
            p_sweep,
            self._previous_sweep,
            running_weight,
        )
        sweep_backtrack_reference_step = (
            None if self._previous_sweep is None else int(i) - 1
        )
        self._previous_sweep = p_sweep.copy()
        ball_cohesion, cohesion_grads = (
            self._ball_cohesion_term_and_grads(
                p_balls,
                running_weight,
            )
        )
        goal_weight = running_weight
        hold_steps = min(
            self._terminal_loss_knots,
            int(num_ctrl_steps),
        )
        if int(i) >= int(num_ctrl_steps) - hold_steps:
            goal_weight += (
                self._terminal_goal_weight / float(hold_steps)
            )
        ball_goal, goal_grads, target_container_goal_grad = (
            self._ball_goal_term_and_grads(
                p_target_container,
                p_balls,
                goal_weight,
            )
        )

        if (
            q is None
            or self._q_root_swing is None
            or len(q) <= self._q_root_swing
        ):
            raise ValueError(
                "target_container loss requires the configured swing joint state"
        )
        theta = float(q[self._q_root_swing])
        swing_weight = self._swing_weight(i, num_ctrl_steps)
        swing, swing_grad = self._swing_term_and_grad(
            theta,
            swing_weight,
        )

        control_vector = np.asarray(u_i, dtype=np.float64)
        scale = self.action_scale(len(control_vector))
        control = running_weight * float(
            np.mean((control_vector / scale) ** 2)
        )
        previous = self._previous_control
        if previous is not None:
            delta = (control_vector - previous) / scale
            control += (
                running_weight
                * self._control_smooth_weight
                * float(np.mean(delta ** 2))
            )
        self._previous_control_by_step[int(i)] = (
            None if previous is None else previous.copy()
        )
        self._previous_control = control_vector.copy()

        safe_flags = tuple(
            self._ball_is_safe(point, p_target_container)
            for point in p_balls
        )
        self._safe_history.append(safe_flags)
        hold_count = min(
            self._success_hold_knots,
            int(num_ctrl_steps),
        )
        if int(i) == int(num_ctrl_steps) - 1:
            trailing_safe_knots = 0
            for historical_flags in reversed(self._safe_history):
                if not all(historical_flags):
                    break
                trailing_safe_knots += 1
            held_safe = (
                len(self._safe_history) >= hold_count
                and all(
                    all(flags)
                    for flags in self._safe_history[-hold_count:]
                )
            )
            self._terminal_cache = {
                "task_success": bool(held_safe),
                "success": bool(held_safe),
                "ball_in_target_container": [
                    bool(value) for value in safe_flags
                ],
                "success_hold_knots": int(hold_count),
                "success_hold_achieved_knots": int(trailing_safe_knots),
                "success_hold_fraction": min(
                    1.0,
                    float(trailing_safe_knots) / float(hold_count),
                ),
                "swing_angle_deg": float(np.degrees(theta)),
                "root_rotation_mode": self._root_rotation_mode,
                "sweep_end": p_sweep.tolist(),
                "target_container_end": p_target_container.tolist(),
                "ball_end": [point.tolist() for point in p_balls],
            }

        self._step_cache[int(i)] = {
            "running_weight": float(running_weight),
            "swing_weight": float(swing_weight),
            "goal_weight": float(goal_weight),
            "sweep_ball_contact_sweep_grad": (
                sweep_ball_contact_sweep_grad
            ),
            "sweep_ball_contact_ball_grads": (
                sweep_ball_contact_ball_grads
            ),
            "sweep_ball_signed_gap": float(sweep_ball_signed_gap),
            "sweep_ball_lateral_offset": float(
                sweep_ball_lateral_offset
            ),
            "sweep_ball_height_offset": float(
                sweep_ball_height_offset
            ),
            "sweep_backtrack_grad": sweep_backtrack_grad,
            "sweep_backtrack_reference_grad": (
                sweep_backtrack_reference_grad
            ),
            "sweep_backtrack_reference_step": (
                sweep_backtrack_reference_step
            ),
            "cohesion_grads": cohesion_grads,
            "goal_grads": goal_grads,
            "target_container_goal_grad": target_container_goal_grad,
            "swing_grad": float(swing_grad),
        }

        return {
            "sweep_ball_contact": float(sweep_ball_contact),
            "sweep_backtrack": float(sweep_backtrack),
            "ball_cohesion": float(ball_cohesion),
            "ball_goal": float(ball_goal),
            "swing": float(swing),
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
        _ = q
        if int(i) not in self._step_cache:
            raise RuntimeError(
                "target_container gradients require compute_terms for the same step"
            )
        cache = self._step_cache[int(i)]
        last_step = (int(i) + 1) * int(sub_steps) - 1
        var_base = last_step * int(ndof_var)

        sweep_ball_contact_sweep_grad = (
            float(coef["sweep_ball_contact"])
            * cache["sweep_ball_contact_sweep_grad"]
        )
        sweep_backtrack_grad = (
            float(coef["sweep_backtrack"])
            * cache["sweep_backtrack_grad"]
        )
        sweep_base = var_base + int(self._var_sweep_base)
        df_dvar[sweep_base : sweep_base + 3] += (
            sweep_ball_contact_sweep_grad
            + sweep_backtrack_grad
        )
        reference_step = cache["sweep_backtrack_reference_step"]
        if reference_step is not None:
            previous_var_base = (
                (int(reference_step) + 1) * int(sub_steps) - 1
            ) * int(ndof_var)
            previous_sweep_base = (
                previous_var_base + int(self._var_sweep_base)
            )
            df_dvar[
                previous_sweep_base : previous_sweep_base + 3
            ] += (
                float(coef["sweep_backtrack"])
                * cache["sweep_backtrack_reference_grad"]
            )

        for base, cohesion_grad, goal_grad, contact_grad in zip(
            self._var_ball_bases,
            cache["cohesion_grads"],
            cache["goal_grads"],
            cache["sweep_ball_contact_ball_grads"],
        ):
            weighted = (
                float(coef["ball_cohesion"]) * cohesion_grad
                + float(coef["ball_goal"]) * goal_grad
                + float(coef["sweep_ball_contact"]) * contact_grad
            )
            ball_base = var_base + int(base)
            df_dvar[ball_base : ball_base + 3] += weighted
        target_container_base = var_base + int(self._var_target_container_base)
        df_dvar[
            target_container_base : target_container_base + 3
        ] += (
            float(coef["ball_goal"])
            * cache["target_container_goal_grad"]
        )
        q_base = last_step * int(ndof_r)
        df_dq[q_base + int(self._q_root_swing)] += (
            float(coef["swing"]) * cache["swing_grad"]
        )

        control_vector = np.asarray(u_i, dtype=np.float64)
        scale = self.action_scale(ndof_u)
        running_weight = float(cache["running_weight"])
        control_base = (
            int(i) * int(sub_steps) * int(ndof_u)
        )
        df_du[
            control_base : control_base + int(ndof_u)
        ] += (
            float(coef["control"])
            * running_weight
            * 2.0
            * control_vector
            / (float(ndof_u) * scale ** 2)
        )
        previous = self._previous_control_by_step.get(int(i))
        if previous is not None:
            previous_base = (
                (int(i) - 1)
                * int(sub_steps)
                * int(ndof_u)
            )
            delta = control_vector - previous
            smooth_gradient = (
                float(coef["control"])
                * running_weight
                * self._control_smooth_weight
                * 2.0
                * delta
                / (float(ndof_u) * scale ** 2)
            )
            df_du[
                control_base : control_base + int(ndof_u)
            ] += smooth_gradient
            df_du[
                previous_base : previous_base + int(ndof_u)
            ] -= smooth_gradient

    def rollout_diagnostics(self, runner, params):
        action, morphology = runner.unpack_params(params)
        if (
            runner.optimize_design
            and runner.design_bundle is not None
            and morphology is not None
        ):
            runner.apply_morphology(
                morphology,
                generate_mesh=False,
            )

        controls = runner.controls_from_action(action)
        runner.sim.reset()
        self._reset_rollout_cache()
        trace = []
        ball_start = None
        previous_sweep = None
        previous_balls = None
        monotonicity = _SweepMonotonicityTracker(self._sweep_axis)
        for i in range(runner.num_ctrl_steps):
            u_i = controls[
                i * runner.ndof_u :
                (i + 1) * runner.ndof_u
            ]
            runner.sim.set_u(u_i)
            runner.sim.forward(runner.sub_steps, verbose=False)
            variables = runner.sim.get_variables()
            q = np.asarray(runner.sim.get_q(), dtype=np.float64)
            p_sweep, p_target_container, p_balls = self._read_task_points(
                variables
            )
            p_sweep = np.asarray(p_sweep, dtype=np.float64)
            p_target_container = np.asarray(p_target_container, dtype=np.float64)
            p_balls = [
                np.asarray(point, dtype=np.float64)
                for point in p_balls
            ]
            if ball_start is None:
                ball_start = [
                    np.asarray(point).tolist()
                    for point in p_balls
                ]
            ball_safety = [
                self._ball_safety_diagnostics(point, p_target_container)
                for point in p_balls
            ]
            rear_ball_index = self._rear_ball_index(
                np.asarray(p_balls, dtype=np.float64)
            )
            sweep_rear_signed_gap = float(
                np.dot(
                    p_balls[rear_ball_index] - p_sweep,
                    self._sweep_axis,
                )
            )
            sweep_step_displacement = (
                None
                if previous_sweep is None
                else float(np.linalg.norm(p_sweep - previous_sweep))
            )
            ball_step_displacements = (
                [None] * len(p_balls)
                if previous_balls is None
                else [
                    float(np.linalg.norm(point - previous))
                    for point, previous in zip(p_balls, previous_balls)
                ]
            )
            monotonicity.observe(p_sweep, u_i)
            terms = self.compute_terms(
                i,
                runner.num_ctrl_steps,
                u_i,
                variables,
                q,
            )
            trace.append(
                {
                    "control_step": int(i),
                    "phase": self.loss_stage_for_step(
                        i,
                        runner.num_ctrl_steps,
                    ),
                    "sweep_alpha": float(
                        self._sweep_alpha(
                            i,
                            runner.num_ctrl_steps,
                        )
                    ),
                    "sweep": p_sweep.tolist(),
                    "balls": [
                        point.tolist()
                        for point in p_balls
                    ],
                    "ball_safety": ball_safety,
                    "all_balls_safe": bool(
                        all(entry["is_safe"] for entry in ball_safety)
                    ),
                    "minimum_ball_safe_margin": float(
                        min(
                            entry["minimum_margin"]
                            for entry in ball_safety
                        )
                    ),
                    "rear_ball_index": int(rear_ball_index),
                    "sweep_rear_signed_gap": sweep_rear_signed_gap,
                    "sweep_step_displacement": sweep_step_displacement,
                    "ball_step_displacements": ball_step_displacements,
                    "swing_angle_deg": float(
                        np.degrees(q[self._q_root_swing])
                    ),
                    "terms": {
                        name: float(value)
                        for name, value in terms.items()
                    },
                }
            )
            previous_sweep = p_sweep.copy()
            previous_balls = [point.copy() for point in p_balls]

        result = dict(self._terminal_cache)
        ball_end = list(result.get("ball_end", ()) or ())
        starts = list(ball_start or ())
        target_container_end = np.asarray(
            result.get("target_container_end", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        target = (
            target_container_end
            - 0.5 * self._target_container_depth * self._sweep_axis
        )
        progress_ratios = []
        for start, end in zip(starts, ball_end):
            start_point = np.asarray(start, dtype=np.float64)
            displacement = np.asarray(end, dtype=np.float64) - start_point
            signed_progress = float(np.dot(displacement, self._sweep_axis))
            required_progress = float(
                np.dot(target - start_point, self._sweep_axis)
            )
            progress_ratios.append(
                min(
                    1.0,
                    max(
                        0.0,
                        signed_progress / max(required_progress, 1.0e-12),
                    ),
                )
            )
        safe_flags = tuple(
            bool(value)
            for value in result.get("ball_in_target_container", ())
        )
        all_safe_steps = [
            int(entry["control_step"])
            for entry in trace
            if entry["all_balls_safe"]
        ]
        trailing_all_safe_knots = 0
        for entry in reversed(trace):
            if not entry["all_balls_safe"]:
                break
            trailing_all_safe_knots += 1
        terminal_window_size = min(10, len(trace))
        terminal_window = (
            trace[-terminal_window_size:]
            if terminal_window_size
            else []
        )
        terminal_window_minimum_safe_margin = (
            float(
                min(
                    entry["minimum_ball_safe_margin"]
                    for entry in terminal_window
                )
            )
            if terminal_window
            else None
        )
        terminal_ball_step_displacements = (
            list(trace[-1]["ball_step_displacements"])
            if trace
            else []
        )
        terminal_window_ball_displacements = [
            float(value)
            for entry in terminal_window
            for value in entry["ball_step_displacements"]
            if value is not None
        ]
        result.update(
            {
                "ball_start": ball_start or [],
                "ball_progress_ratios": progress_ratios,
                "mean_ball_progress_ratio": (
                    float(np.mean(progress_ratios))
                    if progress_ratios
                    else 0.0
                ),
                "safe_ball_fraction": (
                    float(sum(safe_flags)) / float(len(safe_flags))
                    if safe_flags
                    else 0.0
                ),
                "success_hold_fraction": float(
                    result.get("success_hold_fraction", 0.0)
                ),
                "root_rotation_mode": self._root_rotation_mode,
                "phase_boundary_steps": {
                    "terminal_hold_start": int(
                        runner.num_ctrl_steps
                        - min(
                            self._terminal_loss_knots,
                            runner.num_ctrl_steps,
                        )
                    ),
                    "terminal_hold_end": int(runner.num_ctrl_steps),
                },
                "control_trace": trace,
                "rollout_robustness": {
                    "first_all_safe_knot": (
                        all_safe_steps[0] if all_safe_steps else None
                    ),
                    "all_safe_knot_count": int(len(all_safe_steps)),
                    "trailing_all_safe_knots": int(
                        trailing_all_safe_knots
                    ),
                    "terminal_minimum_safe_margin": (
                        float(trace[-1]["minimum_ball_safe_margin"])
                        if trace
                        else None
                    ),
                    "terminal_window_knots": int(terminal_window_size),
                    "terminal_window_minimum_safe_margin": (
                        terminal_window_minimum_safe_margin
                    ),
                    "terminal_ball_step_displacements": (
                        terminal_ball_step_displacements
                    ),
                    "terminal_window_max_ball_step_displacement": (
                        float(max(terminal_window_ball_displacements))
                        if terminal_window_ball_displacements
                        else None
                    ),
                    "terminal_sweep_rear_signed_gap": (
                        float(trace[-1]["sweep_rear_signed_gap"])
                        if trace
                        else None
                    ),
                    "configured_success_hold_knots": int(
                        self._success_hold_knots
                    ),
                    "configured_terminal_loss_knots": int(
                        self._terminal_loss_knots
                    ),
                },
                "sweep_monotonicity": monotonicity.summary(),
                "action_parameterization": (
                    runner.action_parameterization
                ),
                "action_control_max_abs": (
                    np.max(
                        np.abs(
                            controls.reshape(
                                runner.num_ctrl_steps,
                                runner.ndof_u,
                            )
                        ),
                        axis=0,
                    ).tolist()
                    if runner.num_ctrl_steps
                    else [0.0] * runner.ndof_u
                ),
            }
        )
        if (
            runner.optimize_design
            and runner.design_bundle is not None
            and morphology is not None
        ):
            result["design_connectivity"] = (
                runner.morphology_connection_diagnostics(
                    morphology
                )
            )
        return result

    def print_info(self, *args):
        _print_info(*args)




__all__ = ["MISSION_NAME", "TaskDynamics", "TaskObjective"]
