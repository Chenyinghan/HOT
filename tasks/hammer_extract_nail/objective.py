from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from bilevel.lower.geometry_audit import SymmetricOverlapAudit
from bilevel.lower.success_policy import stage_aware_penetration_success
from bilevel.runner import BaseTask
from tasks.objective import TaskObjective


MISSION_NAME = "hammer_extract_nail"


class TaskDynamics(BaseTask):
    has_finger_design = False
    optimize_design = False

    def __init__(
        self,
        *,
        num_steps=6000,
        sub_steps=100,
        coef_goal=200.0,
        coef_tool_pose=5.0,
        coef_impact=2.0,
        coef_contact=100.0,
        coef_control=0.1,
        control_smooth_weight=0.25,
        force_connectivity=True,
        generic_design_protocol="connected_direct_planar_hexahedron",
        optimize_finger_design=False,
        task_phase="full",
        approach_end_fraction=0.20,
        hammer_end_fraction=0.35,
        transfer_end_fraction=0.55,
        engage_end_fraction=0.75,
        target_down_depth=2.0,
        hammer_completion_depth=1.6,
        target_up_lift=1.8,
        goal_tolerance=0.01,
        success_hold_knots=1,
        premature_lift_tolerance=0.09,
        approach_acceptance_distance=2.25,
        transfer_approach_distance=0.30,
        extraction_engage_distance=0.50,
        engage_gate_tolerance=0.10,
        extraction_approach_clearance=2.0,
        max_contact_penetration=0.15,
        geometry_audit_enabled=True,
        geometry_audit_containment_fraction=0.2,
        geometry_audit_normalized_depth=1.0,
        geometry_audit_min_sustained_seconds=0.2,
        hammer_target_speed=8.0,
        hammer_speed_scale=8.0,
        hammer_roll_pose_weight=1.0,
        engage_align_grad_clip=100.0,
        curriculum_stage_weights=(1.0, 3.0, 4.0, 3.6, 3.0, 3.4),
        stage_stop_motion=True,
        stage_future_loss_mode="truncate",
        action_scale_x=30.0,
        action_scale_y=15.0,
        action_scale_z=30.0,
        action_scale_roll=2.062511003414438,
        roll_joint_name="freeform_roll_joint",
    ):
        self._num_steps = int(num_steps)
        self._sub_steps = int(sub_steps)
        self._force_connectivity = bool(force_connectivity)
        self._generic_design_protocol = str(generic_design_protocol)
        self._optimize_finger_design = bool(optimize_finger_design)
        self._task_phase = str(task_phase).lower()
        if self._task_phase not in ("full", "hammer"):
            raise ValueError(f"Unsupported hammer_extract_nail task_phase '{task_phase}'. Expected 'full' or 'hammer'.")
        phase_fractions = (
            float(approach_end_fraction),
            float(hammer_end_fraction),
            float(transfer_end_fraction),
            float(engage_end_fraction),
        )
        if not (
            0.0
            < phase_fractions[0]
            < phase_fractions[1]
            < phase_fractions[2]
            < phase_fractions[3]
            < 1.0
        ):
            raise ValueError(
                "hammer_extract_nail phase fractions must satisfy "
                "0 < approach_end < hammer_end < transfer_end < engage_end < 1"
            )
        self._phase_fractions = phase_fractions

        self._coef = {
            "goal": float(coef_goal),
            "tool_pose": float(coef_tool_pose),
            "impact": float(coef_impact),
            "contact": float(coef_contact),
            "control": float(coef_control),
        }
        if any(value < 0.0 for value in self._coef.values()):
            raise ValueError("hammer_extract_nail objective coefficients must be nonnegative")

        self._pos_scale = 10.0
        self._control_smooth_weight = float(control_smooth_weight)
        if self._control_smooth_weight < 0.0:
            raise ValueError("control_smooth_weight must be nonnegative")
        self._target_down_depth = float(target_down_depth)
        self._hammer_completion_depth = float(
            hammer_completion_depth
        )
        self._target_up_lift = float(target_up_lift)
        if self._target_down_depth <= 0.0 or self._target_up_lift <= 0.0:
            raise ValueError("hammer_extract_nail progress targets must be positive")
        if not (
            0.0
            < self._hammer_completion_depth
            <= self._target_down_depth
        ):
            raise ValueError(
                "hammer_completion_depth must be positive and no greater "
                "than target_down_depth"
            )
        self._goal_tolerance = float(goal_tolerance)
        self._success_hold_knots = int(success_hold_knots)
        if self._success_hold_knots <= 0:
            raise ValueError("success_hold_knots must be positive")
        self._premature_lift_tolerance = float(premature_lift_tolerance)
        self._approach_acceptance_distance = float(
            approach_acceptance_distance
        )
        self._transfer_approach_distance = float(
            transfer_approach_distance
        )
        self._extraction_engage_distance = float(extraction_engage_distance)
        self._engage_gate_tolerance = float(engage_gate_tolerance)
        self._extraction_approach_clearance = float(extraction_approach_clearance)
        if min(
            self._goal_tolerance,
            self._premature_lift_tolerance,
            self._approach_acceptance_distance,
            self._transfer_approach_distance,
            self._extraction_engage_distance,
            self._engage_gate_tolerance,
            self._extraction_approach_clearance,
        ) < 0.0:
            raise ValueError("hammer_extract_nail extraction tolerances must be nonnegative")
        self._hammer_approach_height = 1.2
        self._hammer_press_depth = 1.2
        self._hammer_target_speed = float(hammer_target_speed)
        self._hammer_speed_scale = float(hammer_speed_scale)
        if self._hammer_target_speed <= 0.0 or self._hammer_speed_scale <= 0.0:
            raise ValueError("hammer speed target and scale must be positive")
        self._hammer_roll_pose_weight = float(
            hammer_roll_pose_weight
        )
        if (
            not np.isfinite(self._hammer_roll_pose_weight)
            or self._hammer_roll_pose_weight < 0.0
        ):
            raise ValueError(
                "hammer_roll_pose_weight must be finite and nonnegative"
            )
        self._hammer_roll_scale = float(np.pi / 2.0)
        self._hammer_roll_target = None
        self._root_joint_position = None
        self._engage_align_grad_clip = float(engage_align_grad_clip)
        if (
            not np.isfinite(self._engage_align_grad_clip)
            or self._engage_align_grad_clip <= 0.0
        ):
            raise ValueError(
                "engage_align_grad_clip must be finite and positive"
            )
        self._hammer_strike_ramp = (0.25, 0.65)
        self._hammer_seed_strike_start = 0.28
        self._hammer_seed_strike_target = 7.5
        # Historical handcrafted action-seed geometry.  Keep this value
        # separate from the learned stage targets so the comparison baseline
        # remains bitwise frozen.
        self._extract_under_cap_depth = 1.4
        self._extract_pull_height = self._target_up_lift
        self._extract_seed_pull_rotation = 0.0
        # The blue-nail variable is the cap center.  Its fixed joint is 3.2
        # units above the shaft center.  Seed-free Engage first places the
        # automatically generated extract marker at that shaft center, then
        # slides it to 0.95 below the cap before Pull begins.  This remains a
        # state-relative task waypoint, not an action trajectory.
        self._extract_shaft_center_depth = 3.2
        self._extract_precontact_depth = 0.95
        # Acceptance tolerance and optimization signal have different jobs.
        # Relaxing the former must not weaken the latter.
        self._engage_pose_scale = 0.30
        self._canonical_action_scale_4d = np.asarray(
            [
                action_scale_x,
                action_scale_y,
                action_scale_z,
                action_scale_roll,
            ],
            dtype=np.float64,
        )
        if (
            self._canonical_action_scale_4d.shape != (4,)
            or not np.all(np.isfinite(self._canonical_action_scale_4d))
            or np.any(self._canonical_action_scale_4d <= 0.0)
        ):
            raise ValueError(
                "hammer_extract_nail four-DOF action scales must be finite and "
                "positive"
            )
        self._roll_joint_name = str(roll_joint_name).strip()
        if not self._roll_joint_name:
            raise ValueError("hammer_extract_nail roll_joint_name must be nonempty")
        self._action_scale = np.array([6e5, 6e5, 1.5e5, 1.5e5, 1.5e5, 1.5e5], dtype=np.float64)
        self._optimization_stages = (
            "approach",
            "hammer",
            "transfer",
            "engage_align",
            "engage_contact",
            "pull",
        )
        stage_weights = np.asarray(
            curriculum_stage_weights,
            dtype=np.float64,
        ).reshape(-1)
        if (
            stage_weights.shape != (len(self._optimization_stages),)
            or not np.all(np.isfinite(stage_weights))
            or np.any(stage_weights <= 0.0)
        ):
            raise ValueError(
                "curriculum_stage_weights must contain six positive "
                "finite values"
            )
        self._curriculum_stage_weights = stage_weights
        self._stage_stop_motion = bool(stage_stop_motion)
        self._stage_future_loss_mode = str(
            stage_future_loss_mode
        ).strip().lower()
        if self._stage_future_loss_mode not in ("truncate", "full"):
            raise ValueError(
                "stage_future_loss_mode must be 'truncate' or 'full'"
            )
        self._optimization_stage = "full"
        self._hammer_contact_pairs = ()
        self._extract_contact_pairs = ()
        self._hammer_wrong_contact_pairs = ()
        self._extract_wrong_contact_pairs = ()
        self._hammer_operated_object_pairs = ()
        self._extract_operated_object_pairs = ()
        # RedMax's smoothed contact model applies real force before hard
        # geometric overlap.  This threshold rejects numerical tails while
        # accepting sustained physical contact without requiring penetration.
        self._contact_activation_threshold = 0.01
        # Encourage a firmer under-cap engagement than the minimum
        # success threshold.  The one-sided loss saturates here, while the
        # independent penetration gate below prevents "more engagement" from
        # becoming permission to tunnel through the cap.
        self._extract_contact_activation_target = 0.03
        self._max_contact_penetration = float(max_contact_penetration)
        if (
            not np.isfinite(self._max_contact_penetration)
            or self._max_contact_penetration <= 0.0
        ):
            raise ValueError(
                "max_contact_penetration must be finite and positive"
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

        self._var_hammer = slice(0, 3)
        self._var_nail_down = slice(3, 6)
        self._var_extract = slice(6, 9)
        self._var_nail_up = slice(9, 12)
        self._var_hammer_base = 0
        self._var_nail_down_base = 3
        self._var_extract_base = 6
        self._var_nail_up_base = 9
        self._required_var_len = 12
        self._q_nail_down = None
        self._q_nail_up = None
        self._q_freeform_root = None
        self._q_root_translation = None
        self._q_root_roll = None
        self._root_rotation_mode = "roll"
        self._root_rotation_axis = np.asarray(
            [1.0, 0.0, 0.0], dtype=np.float64
        )
        self._seed_hammer = np.array([16.9, 1.8, -1.05], dtype=np.float64)
        self._seed_nail_down = np.array([18.5, 2.8, -1.6], dtype=np.float64)
        self._seed_extract = np.array([16.9, -3.9, 0.0], dtype=np.float64)
        self._seed_nail_up = np.array([18.5, -7.0, -1.6], dtype=np.float64)

        self._reset_rollout_cache()

    def num_steps(self) -> int:
        return self._num_steps

    def sub_steps(self) -> int:
        return self._sub_steps

    def objective_weights(self) -> dict:
        return dict(self._coef)

    def init_task(self, sim):
        if int(sim.ndof_u) != 4:
            raise ValueError(
                "hammer_extract_nail requires xyz translation plus one "
                f"Handle-centred rotation motor (4 controls), got {sim.ndof_u}"
            )
        try:
            points = self._read_task_points(sim.get_variables())
            (
                self._seed_hammer,
                self._seed_nail_down,
                self._seed_extract,
                self._seed_nail_up,
            ) = tuple(np.asarray(point, dtype=np.float64).copy() for point in points)
            if self._root_joint_position is not None:
                axis = np.asarray(
                    self._root_rotation_axis, dtype=np.float64
                )
                axis /= max(float(np.linalg.norm(axis)), 1e-12)
                radial = self._seed_hammer - self._root_joint_position
                radial = radial - axis * float(np.dot(axis, radial))
                desired = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
                desired = desired - axis * float(np.dot(axis, desired))
                radial_norm = float(np.linalg.norm(radial))
                desired_norm = float(np.linalg.norm(desired))
                self._hammer_roll_target = None
                if radial_norm > 1e-9 and desired_norm > 1e-9:
                    radial /= radial_norm
                    desired /= desired_norm
                    self._hammer_roll_target = float(
                        np.arctan2(
                            float(np.dot(axis, np.cross(radial, desired))),
                            float(np.dot(radial, desired)),
                        )
                    )
        except Exception as exc:
            print(
                f"[WARN] Could not read initial hammer_extract_nail points; using XML defaults: {exc}",
                flush=True,
            )

    def action_scale(self, ndof_u: int) -> np.ndarray:
        if ndof_u == 4:
            return self._canonical_action_scale_4d.copy()
        if ndof_u in (3, 6):
            return np.array(
                [30.0, 15.0, 30.0, 1.5, 1.5, 1.5],
                dtype=np.float64,
            )[:ndof_u]
        if ndof_u <= len(self._action_scale):
            return self._action_scale[:ndof_u].copy()
        out = np.ones(ndof_u, dtype=np.float64)
        out[: len(self._action_scale)] = self._action_scale
        return out

    def init_action(self, ndof_u: int, num_ctrl_steps: int, seed: int) -> np.ndarray:
        if ndof_u == 4:
            translation = self.init_action(
                3,
                num_ctrl_steps,
                seed,
            ).reshape(num_ctrl_steps, 3)
            action = np.zeros(
                (num_ctrl_steps, 4),
                dtype=np.float64,
            )
            action[:, :3] = translation
            return action.reshape(-1)

        if seed == 0:
            action = np.zeros(ndof_u * num_ctrl_steps, dtype=np.float64)
        else:
            rng = np.random.RandomState(seed)
            action = rng.uniform(-0.15, 0.15, size=(ndof_u * num_ctrl_steps,))

        if ndof_u in (3, 6):
            action_scale = self.action_scale(ndof_u)

            def position_action(value, axis):
                ratio = np.clip(float(value) / float(action_scale[axis]), -0.95, 0.95)
                return float(np.arctanh(ratio))

            # Keep the historical handcrafted seed bitwise frozen.  The new
            # five-stage schedule belongs to the objective, not to this
            # comparison baseline.
            count = max(1, int(num_ctrl_steps))
            hammer_end = int(np.clip(round(count * 0.25), 1, count - 3))
            transfer_end = int(
                np.clip(round(count * 0.50), hammer_end + 1, count - 2)
            )
            engage_end = int(
                np.clip(round(count * 0.75), transfer_end + 1, count - 1)
            )
            hammer_exit_local = None
            for i in range(hammer_end):
                hammer_progress = self._phase_progress(i, 0, hammer_end)
                press_alpha = self._ramp_from_progress(
                    hammer_progress,
                    0.25,
                    0.75,
                )
                hammer_target = self._seed_nail_down.copy()
                hammer_target[2] += (
                    (1.0 - press_alpha) * self._hammer_approach_height
                    - press_alpha * self._hammer_press_depth
                )
                # Local freeform coordinates inherit the original 180-degree
                # x rotation, so local y/z have the opposite world direction.
                local_target = (hammer_target - self._seed_hammer) * np.array(
                    [1.0, -1.0, -1.0], dtype=np.float64
                )
                strike_alpha = self._ramp_from_progress(
                    hammer_progress,
                    self._hammer_seed_strike_start,
                    0.85,
                )
                local_target[2] = (
                    (1.0 - strike_alpha) * local_target[2]
                    + strike_alpha * self._hammer_seed_strike_target
                )
                hammer_exit_local = local_target.copy()
                for axis in range(3):
                    action[i * ndof_u + axis] = position_action(local_target[axis], axis)
            if not self._hammer_only():
                frame_sign = np.array([1.0, -1.0, -1.0], dtype=np.float64)
                approach_local = (
                    self._legacy_extract_approach_target(self._seed_nail_up)
                    - self._seed_extract
                ) * frame_sign
                engage_local = (
                    self._legacy_extract_interaction_target(self._seed_nail_up)
                    - self._seed_extract
                ) * frame_sign
                if hammer_exit_local is None:
                    hammer_exit_local = np.zeros(3, dtype=np.float64)

                for i in range(hammer_end, transfer_end):
                    alpha = self._phase_progress(
                        i,
                        hammer_end,
                        max(1, transfer_end - hammer_end),
                    )
                    local_target = (1.0 - alpha) * hammer_exit_local + alpha * approach_local
                    for axis in range(3):
                        action[i * ndof_u + axis] = position_action(local_target[axis], axis)

                for i in range(transfer_end, engage_end):
                    alpha = self._phase_progress(
                        i,
                        transfer_end,
                        max(1, engage_end - transfer_end),
                    )
                    local_target = (1.0 - alpha) * approach_local + alpha * engage_local
                    for axis in range(3):
                        action[i * ndof_u + axis] = position_action(local_target[axis], axis)

                for i in range(engage_end, num_ctrl_steps):
                    pull_alpha = self._phase_progress(
                        i,
                        engage_end,
                        max(1, num_ctrl_steps - engage_end),
                    )
                    extract_target = self._seed_nail_up.copy()
                    extract_target[2] += (
                        -self._extract_under_cap_depth
                        + pull_alpha * self._extract_pull_height
                    )
                    local_target = (extract_target - self._seed_extract) * frame_sign
                    for axis in range(3):
                        action[i * ndof_u + axis] = position_action(local_target[axis], axis)
                    if ndof_u == 6 and self._extract_seed_pull_rotation != 0.0:
                        action[i * ndof_u + 3] = position_action(
                            self._extract_seed_pull_rotation * pull_alpha,
                            3,
                        )
        elif ndof_u >= 2:
            count = max(1, int(num_ctrl_steps))
            hammer_end = int(np.clip(round(count * 0.25), 1, count - 3))
            transfer_end = int(
                np.clip(round(count * 0.50), hammer_end + 1, count - 2)
            )
            engage_end = int(
                np.clip(round(count * 0.75), transfer_end + 1, count - 1)
            )
            for i in range(hammer_end):
                progress = self._phase_progress(i, 0, hammer_end)
                action[i * ndof_u + 1] = 1.4 if progress > 0.15 else 0.6
            if not self._hammer_only():
                for i in range(hammer_end, transfer_end):
                    action[i * ndof_u + 1] = 0.25
                for i in range(transfer_end, engage_end):
                    action[i * ndof_u + 1] = 0.55
                for i in range(engage_end, num_ctrl_steps):
                    action[i * ndof_u + 1] = -0.9
        return action

    def generic_design_bounds_kwargs(self) -> dict:
        return {
            "finger_bounds": (0.85, 1.35),
            "tool_bounds": (-0.45, 0.75),
        }

    def augment_parameter_gradient(
        self,
        runner,
        params,
        grad,
        *,
        compute_action_grad=True,
        compute_design_grad=True,
    ):
        """Condition only the contact-sensitive shaft-alignment gradient."""

        _ = (params, compute_design_grad)
        if (
            not compute_action_grad
            or self._optimization_stage != "engage_align"
        ):
            return grad
        conditioned = np.asarray(grad, dtype=np.float64).copy()
        action_dim = int(runner.ndof_u * runner.num_ctrl_steps)
        np.clip(
            conditioned[:action_dim],
            -self._engage_align_grad_clip,
            self._engage_align_grad_clip,
            out=conditioned[:action_dim],
        )
        return conditioned

    def _hammer_only(self):
        return self._task_phase == "hammer"

    def set_optimization_stage(self, stage):
        stage = str(stage).strip().lower()
        valid = set(self._optimization_stages) | {"engage", "full"}
        if stage not in valid:
            raise ValueError(
                f"Unsupported hammer_extract_nail optimization stage {stage!r}; "
                f"expected one of {sorted(valid)}"
            )
        self._optimization_stage = stage

    def optimization_stage_stop_motion(self):
        """Enable a non-differentiable stopped replay after stage failure."""

        return self._stage_stop_motion

    def optimization_stage_future_loss_mode(self):
        """Return how stopped-replay tail losses contribute to the score."""

        return self._stage_future_loss_mode

    def optimization_stage_motion_dofs(self):
        """Return the controlled tool coordinates frozen after stage failure."""

        indices = []
        if self._q_root_translation is not None:
            indices.extend(
                range(
                    int(self._q_root_translation),
                    int(self._q_root_translation) + 3,
                )
            )
        if self._q_root_roll is not None:
            indices.append(int(self._q_root_roll))
        if not indices:
            raise RuntimeError(
                "hammer_extract_nail could not resolve controlled tool motion DOFs"
            )
        return tuple(indices)

    def optimization_stage_schedule(self, maxiter):
        """Allocate one public optimizer budget across the six task stages."""

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
                order = np.argsort(-(exact - additions), kind="stable")
                allocations[order[:remainder]] += 1
        return tuple(
            (stage, int(stage_budget))
            for stage, stage_budget in zip(
                self._optimization_stages,
                allocations,
            )
            if stage_budget > 0
        )

    def optimization_stage_action_window(self, stage, num_ctrl_steps):
        """Train only the current causal action segment."""

        stage = str(stage).strip().lower()
        approach_end, hammer_end, transfer_end, engage_end = (
            self._phase_boundaries(num_ctrl_steps)
        )
        windows = {
            "approach": (0, approach_end),
            # Hammer may refine its prerequisite approach.  Once Hammer is
            # accepted, every later window starts after hammer_end and the
            # complete approach+strike prefix is frozen.
            "hammer": (0, hammer_end),
            "transfer": (hammer_end, transfer_end),
            "engage_align": (
                transfer_end,
                self._engage_midpoint_end(num_ctrl_steps),
            ),
            "engage_contact": (
                self._engage_midpoint_end(num_ctrl_steps),
                engage_end,
            ),
            "engage": (transfer_end, engage_end),
            "pull": (engage_end, int(num_ctrl_steps)),
        }
        if stage not in windows:
            raise ValueError(
                f"Unsupported hammer_extract_nail optimization stage {stage!r}"
            )
        return windows[stage]

    def loss_stage_for_step(self, i, num_ctrl_steps):
        """Assign each control knot to one physical curriculum stage."""

        approach_end, hammer_end, transfer_end, engage_end = (
            self._phase_boundaries(num_ctrl_steps)
        )
        engage_midpoint = self._engage_midpoint_end(num_ctrl_steps)
        i = int(i)
        if i < approach_end:
            return "approach"
        if i < hammer_end:
            return "hammer"
        if i < transfer_end:
            return "transfer"
        if i < engage_midpoint:
            return "engage_align"
        if i < engage_end:
            return "engage_contact"
        return "pull"

    def _reset_rollout_cache(self):
        self._nail_down0_z = None
        self._nail_up0_z = None
        self._nail_down0_q = None
        self._nail_up0_q = None
        self._nail_up0_position = None
        self._prev_hammer_for_terms = None
        self._prev_control_for_terms = None
        self._prev_term_step = None
        self._hammer_prev_by_step = {}
        self._control_prev_by_step = {}
        self._approach_distance_at_end = None
        self._hammer_boundary_down_depth = None
        self._transfer_distance_at_end = None
        self._max_pre_pull_up = 0.0
        self._max_pull_up = 0.0
        self._current_up_goal_hold_knots = 0
        self._max_up_goal_hold_knots = 0
        self._last_up_goal_hold_step = None
        self._shaft_alignment_distance_at_mid = None
        self._engagement_distance_at_end = None
        self._min_engagement_distance = float("inf")
        self._max_hammer_contact_activation = 0.0
        self._max_extract_contact_activation = 0.0
        self._extract_contact_at_engage_end = False
        self._extract_physical_contact_at_engage_end = False
        self._extract_undercap_contact_at_engage_end = False
        self._extract_contact_activation_at_engage_end = 0.0
        self._max_extract_engage_penetration = 0.0
        self._extract_contact_during_pull = False
        self._extract_undercap_contact_during_pull = False
        self._max_extract_pull_penetration = 0.0
        self._max_head_nail_down_penetration = 0.0
        self._max_head_nail_down_penetration_pair = None
        self._max_head_nail_up_engage_penetration = 0.0
        self._max_head_nail_up_engage_penetration_pair = None
        self._max_head_nail_up_pull_penetration = 0.0
        self._max_head_nail_up_pull_penetration_pair = None
        self._terminal_cache = {}

    def _refresh_terminal_contact_outcomes(self):
        """Refresh contact-dependent terminal gates after contact sampling."""

        if not self._terminal_cache:
            return
        semantic_contact_configured = bool(
            self._hammer_contact_pairs
        ) and (
            self._hammer_only() or bool(self._extract_contact_pairs)
        )
        engagement_distance = float(
            self._terminal_cache.get("engagement_distance", float("inf"))
        )
        hammer_contact_met = bool(
            not semantic_contact_configured
            or self._max_hammer_contact_activation
            >= self._contact_activation_threshold
        )
        extract_contact_met = bool(
            self._hammer_only()
            or not semantic_contact_configured
            or (
                self._extract_undercap_contact_during_pull
                and self._max_head_nail_up_pull_penetration
                <= self._max_contact_penetration
            )
        )
        engagement_goal_met = bool(
            self._hammer_only()
            or (
                engagement_distance
                <= (
                    self._extraction_engage_distance
                    + self._engage_gate_tolerance
                )
                and self._max_head_nail_up_engage_penetration
                <= self._max_contact_penetration
            )
        )
        self._terminal_cache.update(
            {
                "engagement_goal_met": engagement_goal_met,
                "hammer_contact_activation": float(
                    self._max_hammer_contact_activation
                ),
                "extract_contact_activation": float(
                    self._max_extract_contact_activation
                ),
                "extract_contact_at_engage_end": bool(
                    self._extract_contact_at_engage_end
                ),
                "extract_physical_contact_at_engage_end": bool(
                    self._extract_physical_contact_at_engage_end
                ),
                "extract_undercap_contact_at_engage_end": bool(
                    self._extract_undercap_contact_at_engage_end
                ),
                "extract_contact_activation_at_engage_end": float(
                    self._extract_contact_activation_at_engage_end
                ),
                "max_extract_engage_penetration": float(
                    self._max_extract_engage_penetration
                ),
                "extract_contact_during_pull": bool(
                    self._extract_contact_during_pull
                ),
                "extract_undercap_contact_during_pull": bool(
                    self._extract_undercap_contact_during_pull
                ),
                "max_extract_pull_penetration": float(
                    self._max_extract_pull_penetration
                ),
                "max_head_nail_down_penetration": float(
                    self._max_head_nail_down_penetration
                ),
                "max_head_nail_down_penetration_pair": (
                    None
                    if self._max_head_nail_down_penetration_pair is None
                    else list(self._max_head_nail_down_penetration_pair)
                ),
                "max_head_nail_up_engage_penetration": float(
                    self._max_head_nail_up_engage_penetration
                ),
                "max_head_nail_up_engage_penetration_pair": (
                    None
                    if self._max_head_nail_up_engage_penetration_pair is None
                    else list(self._max_head_nail_up_engage_penetration_pair)
                ),
                "max_head_nail_up_pull_penetration": float(
                    self._max_head_nail_up_pull_penetration
                ),
                "max_head_nail_up_pull_penetration_pair": (
                    None
                    if self._max_head_nail_up_pull_penetration_pair is None
                    else list(self._max_head_nail_up_pull_penetration_pair)
                ),
                "hammer_contact_met": hammer_contact_met,
                "extract_contact_met": extract_contact_met,
            }
        )
        self._terminal_cache["task_success"] = bool(
            self._terminal_cache.get("down_goal_met", False)
            and engagement_goal_met
            and self._terminal_cache.get("premature_lift_ok", False)
            and self._terminal_cache.get("up_goal_met", False)
            and hammer_contact_met
            and self._max_head_nail_down_penetration
            <= self._max_contact_penetration
            and self._max_head_nail_up_pull_penetration
            <= self._max_contact_penetration
        )

    def _joint_ndof(self, joint_type):
        joint_type = str(joint_type).lower()
        if joint_type == "fixed":
            return 0
        if joint_type in ("revolute", "prismatic"):
            return 1
        if joint_type == "planar":
            return 2
        if joint_type in ("translational", "spherical", "spherical-euler", "spherical-exp", "free2d"):
            return 3
        if joint_type in ("free3d", "free3d-euler", "free3d-exp"):
            return 6
        return None

    def _configure_q_layout(self, root):
        self._q_nail_down = None
        self._q_nail_up = None
        self._q_freeform_root = None
        self._q_root_translation = None
        self._q_root_roll = None
        self._root_joint_position = None
        self._root_rotation_mode = "roll"
        self._root_rotation_axis = np.asarray(
            [1.0, 0.0, 0.0], dtype=np.float64
        )
        root_rotation = np.eye(3, dtype=np.float64)

        q_base = 0
        for elem in root.iter("joint"):
            name = elem.attrib.get("name", "")
            joint_type = elem.attrib.get("type", "").lower()
            ndof = self._joint_ndof(joint_type)
            if ndof is None:
                print(
                    f"[WARN] Unknown joint type '{joint_type}' while parsing hammer_extract_nail q layout; "
                    "falling back to world-z nail progress.",
                    flush=True,
                )
                self._q_nail_down = None
                self._q_nail_up = None
                return

            if name == "nail_down_joint" and joint_type == "prismatic":
                self._q_nail_down = q_base
            elif name == "nail_up_joint" and joint_type == "prismatic":
                self._q_nail_up = q_base
            elif (
                name == "freeform_root_joint"
                and joint_type == "translational"
            ):
                self._q_root_translation = q_base
                values = np.fromstring(
                    elem.attrib.get("pos", ""),
                    sep=" ",
                    dtype=np.float64,
                )
                if values.shape == (3,):
                    self._root_joint_position = values
                quat = np.fromstring(
                    elem.attrib.get("quat", "1 0 0 0"),
                    sep=" ",
                    dtype=np.float64,
                )
                if quat.shape == (4,):
                    w, x, y, z = quat
                    root_rotation = np.asarray(
                        [
                            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                        ],
                        dtype=np.float64,
                    )
            elif name == "freeform_root_joint" and joint_type == "free3d-exp":
                self._q_freeform_root = q_base
            elif (
                name in {
                    self._roll_joint_name,
                    "freeform_roll_joint",
                    "freeform_pitch_joint",
                    "freeform_yaw_joint",
                }
                and joint_type == "revolute"
            ):
                self._q_root_roll = q_base
                axis = np.fromstring(
                    elem.attrib.get("axis", "1 0 0"),
                    sep=" ",
                    dtype=np.float64,
                )
                if axis.shape == (3,) and float(np.linalg.norm(axis)) > 1e-12:
                    axis = axis / float(np.linalg.norm(axis))
                    self._root_rotation_axis = root_rotation @ axis
                    axis_index = int(np.argmax(np.abs(axis)))
                    self._root_rotation_mode = ("roll", "pitch", "yaw")[axis_index]
            q_base += ndof

        if self._q_nail_down is None or self._q_nail_up is None:
            print(
                "[WARN] Could not find prismatic nail joints in hammer_extract_nail XML; "
                "falling back to world-z nail progress.",
                flush=True,
            )

    def _has_q_progress(self, q):
        if self._q_nail_down is None or self._q_nail_up is None or q is None:
            return False
        return len(q) > max(self._q_nail_down, self._q_nail_up)

    def _has_q_freeform_root(self, q):
        if self._q_freeform_root is None or q is None:
            return False
        return len(q) >= self._q_freeform_root + 6

    def _read_nail_progress(self, q, p_nail_down, p_nail_up):
        if self._has_q_progress(q):
            q_down = float(q[self._q_nail_down])
            q_up = float(q[self._q_nail_up])
            if self._nail_down0_q is None:
                self._nail_down0_q = q_down
            if self._nail_up0_q is None:
                self._nail_up0_q = q_up
            return (
                float(q_down - self._nail_down0_q),
                float(q_up - self._nail_up0_q),
                True,
            )

        if self._nail_down0_z is None:
            self._nail_down0_z = float(p_nail_down[2])
        if self._nail_up0_z is None:
            self._nail_up0_z = float(p_nail_up[2])
        return (
            float(self._nail_down0_z - p_nail_down[2]),
            float(p_nail_up[2] - self._nail_up0_z),
            False,
        )

    def _configure_variable_layout(self, model_path: str):
        root = ET.parse(model_path).getroot()
        self._configure_q_layout(root)

        variable = root.find("variable")
        if variable is None:
            return

        entries = list(variable.findall("endeffector"))
        hammer_idx = None
        nail_down_idx = None
        extract_idx = None
        nail_up_idx = None
        for idx, elem in enumerate(entries):
            joint = elem.attrib.get("joint", "").lower()
            if "nail_down" in joint:
                nail_down_idx = idx
            elif "nail_up" in joint:
                nail_up_idx = idx
            elif "hammer" in joint:
                hammer_idx = idx
            elif "extract" in joint:
                extract_idx = idx

        if None in (hammer_idx, nail_down_idx, extract_idx, nail_up_idx):
            print(
                "[WARN] Could not infer full hammer_extract_nail variable layout from XML; "
                "using default layout.",
                flush=True,
            )
            return

        self._var_hammer_base = 3 * hammer_idx
        self._var_nail_down_base = 3 * nail_down_idx
        self._var_extract_base = 3 * extract_idx
        self._var_nail_up_base = 3 * nail_up_idx
        self._var_hammer = slice(self._var_hammer_base, self._var_hammer_base + 3)
        self._var_nail_down = slice(self._var_nail_down_base, self._var_nail_down_base + 3)
        self._var_extract = slice(self._var_extract_base, self._var_extract_base + 3)
        self._var_nail_up = slice(self._var_nail_up_base, self._var_nail_up_base + 3)
        self._required_var_len = 3 * (max(hammer_idx, nail_down_idx, extract_idx, nail_up_idx) + 1)

    @staticmethod
    def _xml_contact_pairs(root):
        pairs = set()
        contact = root.find("contact")
        if contact is None:
            return pairs
        for elem in contact:
            body1 = elem.attrib.get(
                "body1",
                elem.attrib.get("general_body"),
            )
            body2 = elem.attrib.get(
                "body2",
                elem.attrib.get("primitive_body"),
            )
            if body1 and body2:
                pairs.add(tuple(sorted((str(body1), str(body2)))))
        return pairs

    @staticmethod
    def _function_contact_bodies(root, function_name):
        marker = root.find(f".//link[@name='{function_name}_endeffector']")
        if marker is None:
            return ()
        node_ids = [
            value.strip()
            for value in marker.attrib.get(
                "function_group_leaves",
                marker.attrib.get("function_group_root", ""),
            ).split(",")
            if value.strip()
        ]
        by_node = {
            str(link.attrib["node_id"]): link
            for link in root.findall(".//link[@node_id]")
        }
        bodies = []
        for node_id in node_ids:
            link = by_node.get(node_id)
            body = None if link is None else link.find("body")
            body_name = None if body is None else body.attrib.get("name")
            if body_name and body_name not in bodies:
                bodies.append(str(body_name))
        return tuple(bodies)

    def _configure_semantic_contacts(self, model_path):
        root = ET.parse(model_path).getroot()
        physical_pairs = self._xml_contact_pairs(root)

        def supported_pairs(function_name, nail_body):
            return tuple(
                (body_name, nail_body)
                for body_name in self._function_contact_bodies(
                    root,
                    function_name,
                )
                if tuple(sorted((body_name, nail_body))) in physical_pairs
            )

        self._hammer_contact_pairs = supported_pairs(
            "hammer",
            "nail_down_cap",
        )
        self._extract_contact_pairs = supported_pairs(
            "extract",
            "nail_up_cap",
        )
        tool_bodies = tuple(
            str(body.attrib["name"])
            for body in root.findall(".//body[@name]")
            if body.attrib.get("name", "").startswith("body_tool_")
            and not body.attrib.get("name", "").endswith("_endeffector")
        )

        def wrong_pairs(nail_body, correct_pairs):
            correct_bodies = {
                body_name for body_name, _ in correct_pairs
            }
            return tuple(
                (body_name, nail_body)
                for body_name in tool_bodies
                if body_name not in correct_bodies
                and tuple(sorted((body_name, nail_body))) in physical_pairs
            )

        self._hammer_wrong_contact_pairs = wrong_pairs(
            "nail_down_cap",
            self._hammer_contact_pairs,
        )
        self._extract_wrong_contact_pairs = wrong_pairs(
            "nail_up_cap",
            self._extract_contact_pairs,
        )

        def operated_object_pairs(nail_bodies):
            expected = tuple(
                (body_name, nail_body)
                for body_name in tool_bodies
                for nail_body in nail_bodies
            )
            missing = tuple(
                pair
                for pair in expected
                if tuple(sorted(pair)) not in physical_pairs
            )
            if missing:
                raise ValueError(
                    "hammer_extract_nail XML must expose physical contact pairs "
                    "from every Head body to the complete operated nail; "
                    f"missing {missing!r}"
                )
            return expected

        # Penetration safety is deliberately broader than semantic contact:
        # every searched Head body is checked against both the shaft and cap
        # of the nail operated in that phase.  Function-group leaves still
        # define the intended hammer/extract contact and end-effector semantics.
        self._hammer_operated_object_pairs = operated_object_pairs(
            ("nail_down", "nail_down_cap")
        )
        self._extract_operated_object_pairs = operated_object_pairs(
            ("nail_up", "nail_up_cap")
        )
        if not self._hammer_contact_pairs or (
            not self._hammer_only() and not self._extract_contact_pairs
        ):
            raise ValueError(
                "hammer_extract_nail XML must expose physical contact pairs from "
                "the hammer/extract function-group leaves to their nail caps"
            )

    def _read_task_points(self, variables):
        if len(variables) < self._required_var_len:
            raise ValueError(
                "hammer_extract_nail task expects XML variables containing hammer, nail_down, "
                f"extract, and nail_up ({self._required_var_len} values), got {len(variables)}."
            )
        return (
            variables[self._var_hammer],
            variables[self._var_nail_down],
            variables[self._var_extract],
            variables[self._var_nail_up],
        )

    def configure_model(self, model_path: str, sim):
        """Configure state/variable indices for every runner mode.

        CoOptRunner calls this hook for both action-only validation and
        design optimization.  Keeping layout discovery here ensures that the
        two modes evaluate the same nail-progress and root-orientation terms.
        """

        self._configure_variable_layout(model_path)
        self._configure_semantic_contacts(model_path)
        if sim is not None and int(sim.ndof_u) == 4:
            if self._q_root_roll is None:
                raise ValueError(
                    "hammer_extract_nail four-control XML is missing the configured "
                    f"root rotation joint {self._roll_joint_name!r}"
                )

    def init_design(self, model_path: str, sim):
        from bilevel.parameterization import build_design_bundle

        self._configure_variable_layout(model_path)
        bundle = build_design_bundle(
            model_path,
            sim,
            {
                "optimize_finger_design": self._optimize_finger_design,
                "force_connectivity": self._force_connectivity,
                "generic_design_protocol": self._generic_design_protocol,
            },
        )
        self._design_bundle = bundle
        bundle.apply(sim, bundle.init_cage_params, generate_mesh=False)
        return bundle

    def bounds(self, ndof_u, num_ctrl_steps, ndof_cage, optimize_design):
        bounds = [(-1.0, 1.0)] * (ndof_u * num_ctrl_steps)
        if optimize_design:
            from bilevel.parameterization import cage_bounds_for_bundle

            bounds += cage_bounds_for_bundle(
                getattr(self, "_design_bundle", None),
                ndof_cage,
                optimize_finger_design=self._optimize_finger_design,
                **self.generic_design_bounds_kwargs(),
            )
        return bounds

    def _phase_boundaries(self, num_ctrl_steps):
        """Return exclusive approach/hammer/transfer/engage boundaries.

        The schedule is expressed as fractions of the rollout so changing the
        physical horizon does not silently collapse any task phase.  No
        morphology information is used here.
        """

        count = max(1, int(num_ctrl_steps))
        if self._hammer_only():
            if count < 2:
                return count, count, count, count
            approach_end = int(np.clip(round(count * 0.6), 1, count - 1))
            return approach_end, count, count, count
        if count < 5:
            return 1, min(2, count), min(3, count), min(4, count)
        approach_end = int(
            np.clip(round(count * self._phase_fractions[0]), 1, count - 4)
        )
        hammer_end = int(
            np.clip(
                round(count * self._phase_fractions[1]),
                approach_end + 1,
                count - 3,
            )
        )
        transfer_end = int(
            np.clip(
                round(count * self._phase_fractions[2]),
                hammer_end + 1,
                count - 2,
            )
        )
        engage_end = int(
            np.clip(
                round(count * self._phase_fractions[3]),
                transfer_end + 1,
                count - 1,
            )
        )
        return approach_end, hammer_end, transfer_end, engage_end

    def _optimization_stage_end(self, num_ctrl_steps):
        approach_end, hammer_end, transfer_end, engage_end = (
            self._phase_boundaries(num_ctrl_steps)
        )
        return {
            "approach": approach_end,
            "hammer": hammer_end,
            "transfer": transfer_end,
            "engage_align": self._engage_midpoint_end(num_ctrl_steps),
            "engage_contact": engage_end,
            "engage": engage_end,
            "pull": int(num_ctrl_steps),
            "full": int(num_ctrl_steps),
        }[self._optimization_stage]

    def _step_objective_active(self, i, num_ctrl_steps):
        return int(i) < self._optimization_stage_end(num_ctrl_steps)

    def _phase_split(self, num_ctrl_steps):
        """Backward-compatible alias for the end of the hammer phase."""

        return self._phase_boundaries(num_ctrl_steps)[1]

    def _phase_progress(self, i, start, length):
        if length <= 1:
            return 1.0
        return float(np.clip((i - start) / float(length - 1), 0.0, 1.0))

    def _ramp_from_progress(self, progress, start, end):
        if progress <= start:
            return 0.0
        if progress >= end:
            return 1.0
        return (progress - start) / max(end - start, 1e-9)

    def _approach_progress(self, i, num_ctrl_steps):
        approach_end, _, _, _ = self._phase_boundaries(num_ctrl_steps)
        return self._phase_progress(i, 0, approach_end)

    def _hammer_progress(self, i, num_ctrl_steps):
        approach_end, hammer_end, _, _ = self._phase_boundaries(num_ctrl_steps)
        return self._phase_progress(
            i,
            approach_end,
            max(1, hammer_end - approach_end),
        )

    def _transfer_progress(self, i, num_ctrl_steps):
        _, hammer_end, transfer_end, _ = self._phase_boundaries(num_ctrl_steps)
        return self._phase_progress(
            i,
            hammer_end,
            max(1, transfer_end - hammer_end),
        )

    def _engage_progress(self, i, num_ctrl_steps):
        _, _, transfer_end, engage_end = self._phase_boundaries(num_ctrl_steps)
        return self._phase_progress(
            i,
            transfer_end,
            max(1, engage_end - transfer_end),
        )

    def _engage_midpoint_end(self, num_ctrl_steps):
        """Exclusive end of the shaft-alignment half of Engage."""

        _, _, transfer_end, engage_end = self._phase_boundaries(
            num_ctrl_steps
        )
        return min(
            engage_end - 1,
            transfer_end + max(1, (engage_end - transfer_end) // 2),
        )

    def _pull_progress(self, i, num_ctrl_steps):
        _, _, _, engage_end = self._phase_boundaries(num_ctrl_steps)
        return self._phase_progress(
            i,
            engage_end,
            max(1, num_ctrl_steps - engage_end),
        )

    def _pull_lift_progress(self, i, num_ctrl_steps):
        """Pull starts already engaged, so lift spans the full phase."""

        return self._pull_progress(i, num_ctrl_steps)

    def _hammer_press_alpha(self, i, num_ctrl_steps):
        progress = self._hammer_progress(i, num_ctrl_steps)
        return self._ramp_from_progress(
            progress,
            self._hammer_strike_ramp[0],
            self._hammer_strike_ramp[1],
        )

    def _impact_active(self, i, num_ctrl_steps):
        approach_end, hammer_end, _, _ = self._phase_boundaries(num_ctrl_steps)
        if i < approach_end or i >= hammer_end:
            return False
        alpha = self._hammer_press_alpha(i, num_ctrl_steps)
        return 0.0 < alpha < 1.0

    def _semantic_contact_pairs_for_step(self, i, num_ctrl_steps):
        if not self._step_objective_active(i, num_ctrl_steps):
            return (), None
        approach_end, hammer_end, transfer_end, _ = self._phase_boundaries(
            num_ctrl_steps
        )
        if (
            approach_end <= i < hammer_end
            and self._hammer_press_alpha(i, num_ctrl_steps) >= 0.5
        ):
            return self._hammer_contact_pairs, "hammer"
        if not self._hammer_only() and i >= transfer_end:
            return self._extract_contact_pairs, "extract"
        return (), None

    def _forbidden_function_contacts_for_step(self, i, num_ctrl_steps):
        if not self._step_objective_active(i, num_ctrl_steps):
            return ()
        allowed_pairs, function_name = self._semantic_contact_pairs_for_step(
            i,
            num_ctrl_steps,
        )
        allowed = set(allowed_pairs)
        pairs = []
        if function_name != "hammer":
            pairs.extend(self._hammer_contact_pairs)
        if not self._hammer_only() and function_name != "extract":
            pairs.extend(self._extract_contact_pairs)
        return tuple(pair for pair in pairs if pair not in allowed)

    @staticmethod
    def _metric_for_pair(contact_metrics, pair):
        body1, body2 = pair
        for metric in contact_metrics:
            metric_pair = {str(metric.body1), str(metric.body2)}
            if metric_pair == {str(body1), str(body2)}:
                return metric
        raise ValueError(
            f"Missing requested physical contact metric for {pair!r}"
        )

    def contact_metric_requests(self):
        pairs = (
            tuple(self._hammer_contact_pairs)
            + tuple(self._extract_contact_pairs)
            + tuple(self._hammer_wrong_contact_pairs)
            + tuple(self._extract_wrong_contact_pairs)
            + tuple(self._hammer_operated_object_pairs)
            + tuple(self._extract_operated_object_pairs)
        )
        return tuple(
            (body1, body2, "activation")
            for body1, body2 in dict.fromkeys(pairs)
        )

    def _max_pair_penetration(self, contact_metrics, pairs):
        maximum = 0.0
        maximum_pair = None
        for pair in pairs:
            metric = self._metric_for_pair(contact_metrics, pair)
            penetration = float(
                getattr(metric, "max_penetration", 0.0)
            )
            if penetration > maximum:
                maximum = penetration
                maximum_pair = tuple(pair)
        return maximum, maximum_pair

    @staticmethod
    def _update_penetration_maximum(
        current_value,
        current_pair,
        candidate_value,
        candidate_pair,
    ):
        if candidate_value > current_value:
            return float(candidate_value), candidate_pair
        return float(current_value), current_pair

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
        _, hammer_end, transfer_end, engage_end = (
            self._phase_boundaries(num_ctrl_steps)
        )
        if i < hammer_end and self._hammer_operated_object_pairs:
            value, pair = self._max_pair_penetration(
                contact_metrics,
                self._hammer_operated_object_pairs,
            )
            (
                self._max_head_nail_down_penetration,
                self._max_head_nail_down_penetration_pair,
            ) = self._update_penetration_maximum(
                self._max_head_nail_down_penetration,
                self._max_head_nail_down_penetration_pair,
                value,
                pair,
            )
        if (
            not self._hammer_only()
            and i >= transfer_end
            and self._extract_operated_object_pairs
        ):
            value, pair = self._max_pair_penetration(
                contact_metrics,
                self._extract_operated_object_pairs,
            )
            if i < engage_end:
                (
                    self._max_head_nail_up_engage_penetration,
                    self._max_head_nail_up_engage_penetration_pair,
                ) = self._update_penetration_maximum(
                    self._max_head_nail_up_engage_penetration,
                    self._max_head_nail_up_engage_penetration_pair,
                    value,
                    pair,
                )
            else:
                (
                    self._max_head_nail_up_pull_penetration,
                    self._max_head_nail_up_pull_penetration_pair,
                ) = self._update_penetration_maximum(
                    self._max_head_nail_up_pull_penetration,
                    self._max_head_nail_up_pull_penetration_pair,
                    value,
                    pair,
                )
        pairs, function_name = self._semantic_contact_pairs_for_step(
            i,
            num_ctrl_steps,
        )
        correct_metrics = tuple(
            self._metric_for_pair(contact_metrics, pair)
            for pair in pairs
        )
        correct_activations = np.asarray(
            [
                float(metric.activation)
                for metric in correct_metrics
            ],
            dtype=np.float64,
        )
        if correct_activations.size:
            activation = float(np.mean(correct_activations))
        else:
            activation = 0.0
        if function_name == "hammer" and correct_activations.size:
            self._max_hammer_contact_activation = max(
                self._max_hammer_contact_activation,
                activation,
            )
        elif function_name == "extract" and correct_activations.size:
            self._max_extract_contact_activation = max(
                self._max_extract_contact_activation,
                activation,
            )
            try:
                _, _, p_extract, p_nail_up = self._read_task_points(
                    variables
                )
                extract_marker_z = float(p_extract[2])
                cap_center_z = float(p_nail_up[2])
            except Exception:
                extract_marker_z = float("nan")
                cap_center_z = float("nan")
            if i == engage_end - 1:
                self._extract_contact_at_engage_end = any(
                    bool(metric.geometrically_touching)
                    for metric in correct_metrics
                )
                self._extract_contact_activation_at_engage_end = activation
            for metric in correct_metrics:
                touching = bool(
                    getattr(metric, "geometrically_touching", False)
                )
                activation_i = float(
                    getattr(metric, "activation", 0.0)
                )
                physically_active = bool(
                    touching
                    or activation_i
                    >= self._contact_activation_threshold
                )
                if not physically_active:
                    continue
                penetration = float(
                    getattr(metric, "max_penetration", 0.0)
                )
                positions = tuple(
                    getattr(metric, "world_positions", ())
                )
                if positions:
                    undercap = bool(
                        np.isfinite(cap_center_z)
                        and any(
                            np.asarray(
                                position,
                                dtype=np.float64,
                            ).reshape(-1)[2]
                            <= cap_center_z
                            for position in positions
                        )
                    )
                else:
                    undercap = bool(
                        np.isfinite(cap_center_z)
                        and np.isfinite(extract_marker_z)
                        and extract_marker_z <= cap_center_z
                    )
                if i == engage_end - 1:
                    self._extract_physical_contact_at_engage_end = True
                    self._max_extract_engage_penetration = max(
                        self._max_extract_engage_penetration,
                        penetration,
                    )
                    if undercap:
                        self._extract_undercap_contact_at_engage_end = True
                if i >= engage_end:
                    self._extract_contact_during_pull = True
                    self._max_extract_pull_penetration = max(
                        self._max_extract_pull_penetration,
                        penetration,
                    )
                    if undercap:
                        self._extract_undercap_contact_during_pull = True

        # Extract extraction is intentionally geometry- and displacement-led.
        # Smooth contact activation remains useful replay diagnostics, but it
        # must not steer Transfer/Engage/Pull: small changes in RedMax's
        # smoothing tail otherwise overwhelm the marker-position objective
        # before the extract is geometrically engaged.  Hammer keeps its contact
        # objective because impact is the physical goal of that phase.
        _, hammer_end, _, _ = self._phase_boundaries(num_ctrl_steps)
        if i >= hammer_end:
            if i == int(num_ctrl_steps) - 1:
                self._refresh_terminal_contact_outcomes()
            return {"contact": 0.0}

        wrong_pairs = ()
        forbidden_pairs = ()
        if self._step_objective_active(i, num_ctrl_steps):
            wrong_pairs = (
                tuple(self._hammer_wrong_contact_pairs)
                + tuple(self._extract_wrong_contact_pairs)
            )
            forbidden_pairs = self._forbidden_function_contacts_for_step(
                i,
                num_ctrl_steps,
            )
        wrong_activations = np.asarray(
            [
                float(
                    self._metric_for_pair(
                        contact_metrics,
                        pair,
                    ).activation
                )
                for pair in wrong_pairs
            ],
            dtype=np.float64,
        )
        forbidden_activations = np.asarray(
            [
                float(
                    self._metric_for_pair(
                        contact_metrics,
                        pair,
                    ).activation
                )
                for pair in forbidden_pairs
            ],
            dtype=np.float64,
        )
        # Hammer keeps its established contact target.  For the extract, the
        # upward under-cap Engage sub-block and Pull use a one-sided minimum.
        # Reaching that modest force removes the reward, so there is no
        # incentive to increase penetration indefinitely.
        _, _, _, engage_end = self._phase_boundaries(num_ctrl_steps)
        midpoint_end = self._engage_midpoint_end(num_ctrl_steps)
        extract_contact_active = bool(
            function_name == "extract"
            and i >= midpoint_end
        )
        reward_correct_contact = bool(
            function_name == "hammer" or extract_contact_active
        )
        contact_target = (
            self._extract_contact_activation_target
            if extract_contact_active
            else 1.0
        )
        contact_errors = np.maximum(
            contact_target - correct_activations,
            0.0,
        )
        missing_contact = (
            0.0
            if not correct_activations.size
            or not reward_correct_contact
            else float(np.mean(contact_errors ** 2))
        )
        wrong_contact = float(
            np.sum(wrong_activations ** 2)
            + np.sum(forbidden_activations ** 2)
        )
        if i == int(num_ctrl_steps) - 1:
            self._refresh_terminal_contact_outcomes()
        return {"contact": missing_contact + wrong_contact}

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
        _ = (u_i, variables, q)
        pairs, function_name = self._semantic_contact_pairs_for_step(
            i,
            num_ctrl_steps,
        )
        _, hammer_end, _, _ = self._phase_boundaries(num_ctrl_steps)
        if i >= hammer_end:
            return {}
        midpoint_end = self._engage_midpoint_end(num_ctrl_steps)
        extract_contact_active = bool(
            function_name == "extract"
            and i >= midpoint_end
        )
        reward_correct_contact = bool(
            function_name == "hammer" or extract_contact_active
        )
        contact_target = (
            self._extract_contact_activation_target
            if extract_contact_active
            else 1.0
        )
        grads = {}
        if pairs and reward_correct_contact:
            scale = float(coef["contact"]) / float(len(pairs))
            for body1, body2 in pairs:
                activation = float(
                    self._metric_for_pair(
                        contact_metrics,
                        (body1, body2),
                    ).activation
                )
                if activation < contact_target:
                    grads[(body1, body2, "activation")] = (
                        2.0
                        * scale
                        * (activation - contact_target)
                    )
        if self._step_objective_active(i, num_ctrl_steps):
            penalized_pairs = (
                tuple(self._hammer_wrong_contact_pairs)
                + tuple(self._extract_wrong_contact_pairs)
                + self._forbidden_function_contacts_for_step(
                    i,
                    num_ctrl_steps,
                )
            )
            for body1, body2 in dict.fromkeys(penalized_pairs):
                activation = float(
                    self._metric_for_pair(
                        contact_metrics,
                        (body1, body2),
                    ).activation
                )
                if activation != 0.0:
                    grads[(body1, body2, "activation")] = (
                        2.0 * float(coef["contact"]) * activation
                    )
        return grads

    def _scheduled_nail_targets(self, i, num_ctrl_steps):
        approach_end, hammer_end, _, engage_end = self._phase_boundaries(
            num_ctrl_steps
        )
        if i < approach_end:
            down_target = 0.0
        elif i < hammer_end:
            down_target = (
                self._hammer_press_alpha(i, num_ctrl_steps)
                * self._target_down_depth
            )
        else:
            down_target = self._target_down_depth

        if self._hammer_only() or i < engage_end:
            up_target = 0.0
        else:
            up_target = (
                self._pull_lift_progress(i, num_ctrl_steps)
                * self._target_up_lift
            )
        return float(down_target), float(up_target)

    def _scheduled_goal_components(self, i, num_ctrl_steps):
        """Return targets and the two physical progress terms active now.

        Hammer progress is taught only during Hammer (plus one final
        preservation check). Repeating the same red-nail reward throughout
        Transfer/Engage/Pull let later action blocks profit from an earlier
        collision by the wrong tool body.  The blue nail is held down before
        Pull, then judged at the terminal knot only.  Dense tracking of an
        artificial lift timetable rejected physically successful pulls merely
        for lifting faster than that timetable.
        """

        _, hammer_end, _, engage_end = self._phase_boundaries(num_ctrl_steps)
        down_target, up_target = self._scheduled_nail_targets(
            i,
            num_ctrl_steps,
        )
        down_active = bool(
            i < hammer_end or i == int(num_ctrl_steps) - 1
        )
        if i < engage_end:
            up_weight = 1.0
        elif i == int(num_ctrl_steps) - 1:
            # Preserve the integrated strength of the former dense Pull goal
            # without prescribing its intermediate timing.
            up_weight = float(max(1, int(num_ctrl_steps) - engage_end))
        else:
            up_weight = 0.0
        return down_target, up_target, down_active, up_weight

    def _hammer_target(self, p_nail_down, i, num_ctrl_steps):
        approach_end, _, _, _ = self._phase_boundaries(num_ctrl_steps)
        if i < approach_end:
            progress = self._approach_progress(i, num_ctrl_steps)
            lift_alpha = min(progress / 0.5, 1.0)
            align_alpha = max((progress - 0.5) / 0.5, 0.0)
            target = np.asarray(self._seed_hammer, dtype=np.float64).copy()
            target[2] = (
                (1.0 - lift_alpha) * self._seed_hammer[2]
                + lift_alpha
                * (
                    float(p_nail_down[2])
                    + self._hammer_approach_height
                )
            )
            target[:2] = (
                (1.0 - align_alpha) * self._seed_hammer[:2]
                + align_alpha
                * np.asarray(p_nail_down[:2], dtype=np.float64)
            )
            return target
        alpha = self._hammer_press_alpha(i, num_ctrl_steps)
        target = np.asarray(p_nail_down, dtype=np.float64).copy()
        target[2] += (1.0 - alpha) * self._hammer_approach_height - alpha * self._hammer_press_depth
        return target

    def _hammer_target_nail_scale(self, i, num_ctrl_steps):
        approach_end, _, _, _ = self._phase_boundaries(num_ctrl_steps)
        if i >= approach_end:
            return np.ones(3, dtype=np.float64)
        progress = self._approach_progress(i, num_ctrl_steps)
        lift_alpha = min(progress / 0.5, 1.0)
        align_alpha = max((progress - 0.5) / 0.5, 0.0)
        return np.asarray(
            [align_alpha, align_alpha, lift_alpha],
            dtype=np.float64,
        )

    def _legacy_extract_interaction_target(self, p_nail_up):
        """Historical target used only by the frozen handcrafted seed."""

        target = np.asarray(p_nail_up, dtype=np.float64).copy()
        target[2] -= self._extract_under_cap_depth
        return target

    def _legacy_extract_approach_target(self, p_nail_up):
        target = self._legacy_extract_interaction_target(p_nail_up)
        return self._offset_extract_approach_target(target)

    def _extract_interaction_target(self, p_nail_up):
        """Place the generated extract marker at the blue nail shaft center."""

        target = np.asarray(p_nail_up, dtype=np.float64).copy()
        target[2] -= self._extract_shaft_center_depth
        return target

    def _extract_undercap_target(self, p_nail_up):
        """Keep the generated extract marker immediately below the moving cap."""

        target = np.asarray(p_nail_up, dtype=np.float64).copy()
        target[2] -= self._extract_precontact_depth
        return target

    def _offset_extract_approach_target(self, target):
        target = np.asarray(target, dtype=np.float64).copy()
        direction = (
            np.asarray(self._seed_extract[:2], dtype=np.float64)
            - np.asarray(self._seed_nail_up[:2], dtype=np.float64)
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            direction = np.array([-1.0, 0.0], dtype=np.float64)
        else:
            direction /= norm
        target[:2] += self._extraction_approach_clearance * direction
        return target

    def _extract_approach_target(self, p_nail_up):
        return self._offset_extract_approach_target(
            self._extract_interaction_target(p_nail_up)
        )

    def _extract_target(self, p_nail_up, i, num_ctrl_steps):
        """Scheduled target for the single extraction-region end-effector."""

        _, hammer_end, transfer_end, engage_end = self._phase_boundaries(
            num_ctrl_steps
        )
        if i < hammer_end:
            return np.asarray(self._seed_extract, dtype=np.float64).copy()
        approach = self._extract_approach_target(p_nail_up)
        if i < transfer_end:
            return approach
        shaft_center = self._extract_interaction_target(p_nail_up)
        if i < engage_end:
            midpoint_end = self._engage_midpoint_end(num_ctrl_steps)
            if i < midpoint_end:
                return shaft_center
            undercap = self._extract_undercap_target(p_nail_up)
            alpha = self._phase_progress(
                i,
                midpoint_end,
                max(1, engage_end - midpoint_end),
            )
            return (1.0 - alpha) * shaft_center + alpha * undercap
        # Pull begins already engaged.  Following one moving relative point
        # does not prescribe an upward action: the terminal nail goal is the
        # only signal that asks the coupled extract+nail system to rise.
        return self._extract_undercap_target(p_nail_up)

    def _extract_target_nail_scale(self, i, num_ctrl_steps):
        _, hammer_end, _, engage_end = self._phase_boundaries(
            num_ctrl_steps
        )
        if i < hammer_end:
            return 0.0
        return 1.0

    def _tool_pose_weight(self, i, num_ctrl_steps):
        """Preserve integrated pose strength for endpoint-only phases.

        Transfer and Engage use one endpoint pose term instead of a dense
        target at every control knot. Weighting that endpoint by the phase
        length keeps the pose/control tradeoff on the same scale as the dense
        formulation without adding another objective term.
        """

        _, hammer_end, transfer_end, engage_end = self._phase_boundaries(
            num_ctrl_steps
        )
        if self._hammer_only() or i < hammer_end:
            return 1.0
        if i < transfer_end:
            return (
                float(transfer_end - hammer_end)
                if i == transfer_end - 1
                else 0.0
            )
        if i < engage_end:
            midpoint_end = self._engage_midpoint_end(num_ctrl_steps)
            first_weight = float(midpoint_end - transfer_end)
            second_weight = float(engage_end - midpoint_end)
            if i == midpoint_end - 1:
                return first_weight
            if i == engage_end - 1:
                return second_weight
            return 0.0
        return 1.0

    def _tool_pose_active(self, i, num_ctrl_steps):
        """Return whether the pose term is active at this control knot."""

        return self._tool_pose_weight(i, num_ctrl_steps) > 0.0

    def _tool_pose_scale(self, i, num_ctrl_steps):
        """Return the characteristic distance for the active pose target.

        Engage first moves from the side approach point to the shaft-center
        point, then slides upward to the cap underside. Pull tracks that same
        moving relative point.  The guidance scale is deliberately separate
        from the acceptance gate so relaxing endpoint tolerance cannot weaken
        either motion against control regularization.
        """

        _, hammer_end, transfer_end, engage_end = self._phase_boundaries(
            num_ctrl_steps
        )
        if hammer_end <= i < transfer_end:
            return max(self._extraction_approach_clearance, 1e-9)
        if transfer_end <= i < engage_end:
            return self._engage_pose_scale
        if i >= engage_end:
            # Pull is relative tracking of one interaction point against the
            # moving cap.  Preserve strong lateral guidance while using a
            # moderate vertical scale: the earlier full-motion
            # scale let the marker lag almost half a unit, while applying the
            # 0.3 gate vertically made contact line search excessively stiff.
            lateral_scale = self._engage_pose_scale
            return np.asarray(
                [lateral_scale, lateral_scale, 1.0],
                dtype=np.float64,
            )
        return self._pos_scale

    def _tool_axis_diff(self, p_hammer, p_extract):
        """Diagnostic-only rigid tool-axis change.

        Canonical Extract action controls are translational, so orientation is
        intentionally not part of the action objective.
        """

        initial_axis = (
            np.asarray(self._seed_hammer, dtype=np.float64)
            - np.asarray(self._seed_extract, dtype=np.float64)
        )
        return (
            np.asarray(p_hammer, dtype=np.float64)
            - np.asarray(p_extract, dtype=np.float64)
            - initial_axis
        )

    def _root_rotation(self, q):
        if (
            self._q_root_roll is not None
            and q is not None
            and len(q) > self._q_root_roll
        ):
            return np.asarray(
                [float(q[self._q_root_roll])],
                dtype=np.float64,
            )
        if not self._has_q_freeform_root(q):
            return None
        base = self._q_freeform_root + 3
        return np.asarray(q[base : base + 3], dtype=np.float64)

    def _hammer_roll_error(self, q):
        if (
            self._hammer_roll_target is None
            or self._q_root_roll is None
            or q is None
            or len(q) <= self._q_root_roll
        ):
            return None
        error = (
            float(q[self._q_root_roll])
            - float(self._hammer_roll_target)
        )
        return float((error + np.pi) % (2.0 * np.pi) - np.pi)

    def compute_terms(self, i, num_ctrl_steps, u_i, variables, q):
        p_hammer, p_nail_down, p_extract, p_nail_up = self._read_task_points(variables)
        p_hammer = np.asarray(p_hammer, dtype=np.float64)
        p_nail_down = np.asarray(p_nail_down, dtype=np.float64)
        p_extract = np.asarray(p_extract, dtype=np.float64)
        p_nail_up = np.asarray(p_nail_up, dtype=np.float64)

        if i == 0:
            self._reset_rollout_cache()
        if self._nail_down0_z is None:
            self._nail_down0_z = float(p_nail_down[2])
        if self._nail_up0_z is None:
            self._nail_up0_z = float(p_nail_up[2])
        down_progress, up_progress, using_q_progress = self._read_nail_progress(q, p_nail_down, p_nail_up)
        if self._nail_up0_position is None:
            self._nail_up0_position = p_nail_up.copy()
            self._nail_up0_position[2] -= up_progress

        approach_end, hammer_end, transfer_end, engage_end = self._phase_boundaries(
            num_ctrl_steps
        )
        pose_scale2 = self._tool_pose_scale(i, num_ctrl_steps) ** 2

        if self._hammer_only() or i < hammer_end:
            p_target = self._hammer_target(p_nail_down, i, num_ctrl_steps)
            pose_error2 = (p_hammer - p_target) ** 2
            tool_pose = float(
                np.sum(pose_error2) / pose_scale2
                if np.ndim(pose_scale2) == 0
                else np.sum(pose_error2 / pose_scale2)
            )
            roll_error = self._hammer_roll_error(q)
            if roll_error is not None:
                tool_pose += (
                    self._hammer_roll_pose_weight
                    * (roll_error / self._hammer_roll_scale) ** 2
                )
        else:
            p_target = self._extract_target(p_nail_up, i, num_ctrl_steps)
            pose_error2 = (p_extract - p_target) ** 2
            tool_pose = float(
                np.sum(pose_error2) / pose_scale2
                if np.ndim(pose_scale2) == 0
                else np.sum(pose_error2 / pose_scale2)
            )
        tool_pose *= self._tool_pose_weight(i, num_ctrl_steps)

        (
            down_target,
            up_target,
            down_active,
            up_active,
        ) = self._scheduled_goal_components(
            i,
            num_ctrl_steps,
        )
        goal = float(
            float(down_active)
            * ((down_progress - down_target) / self._target_down_depth) ** 2
            + float(up_active)
            * ((up_progress - up_target) / self._target_up_lift) ** 2
        )
        objective_active = self._step_objective_active(
            i,
            num_ctrl_steps,
        )
        if not objective_active:
            tool_pose = 0.0
            goal = 0.0

        sequential = self._prev_term_step == i - 1
        prev_hammer = self._prev_hammer_for_terms if sequential else None
        prev_control = self._prev_control_for_terms if sequential else None
        self._hammer_prev_by_step[i] = (
            None
            if prev_hammer is None
            else np.asarray(prev_hammer, dtype=np.float64).copy()
        )
        self._control_prev_by_step[i] = (
            None
            if prev_control is None
            else np.asarray(prev_control, dtype=np.float64).copy()
        )

        impact = 0.0
        if (
            objective_active
            and prev_hammer is not None
            and self._impact_active(i, num_ctrl_steps)
        ):
            dt = float(self._sub_steps) * 1e-3
            velocity = (p_hammer - prev_hammer) / dt
            down_speed = -float(velocity[2])
            scale2 = self._hammer_speed_scale ** 2
            impact = float(
                ((down_speed - self._hammer_target_speed) ** 2)
                / scale2
                + np.dot(velocity[:2], velocity[:2]) / scale2
            )

        if len(u_i) > 0:
            u_i = np.asarray(u_i, dtype=np.float64)
            u_scale = self.action_scale(len(u_i))
            control = float(np.mean((u_i / u_scale) ** 2))
            if prev_control is not None:
                control_delta = (u_i - prev_control) / u_scale
                control += self._control_smooth_weight * float(
                    np.mean(control_delta ** 2)
                )
        else:
            control = 0.0

        self._prev_hammer_for_terms = p_hammer.copy()
        self._prev_control_for_terms = np.asarray(
            u_i,
            dtype=np.float64,
        ).copy()
        self._prev_term_step = int(i)

        interaction_distance = float(
            np.linalg.norm(p_extract - self._extract_interaction_target(p_nail_up))
        )
        undercap_distance = float(
            np.linalg.norm(p_extract - self._extract_undercap_target(p_nail_up))
        )
        # "Premature" means lifting before the extraction tool starts
        # engaging the nail. Once Engage begins, small nail motion caused by
        # real shaft/cap interaction is legitimate extraction progress.
        if i < transfer_end:
            self._max_pre_pull_up = max(self._max_pre_pull_up, float(up_progress))
        if transfer_end <= i < engage_end:
            self._min_engagement_distance = min(
                self._min_engagement_distance,
                interaction_distance,
            )
        if not self._hammer_only() and i >= engage_end:
            self._max_pull_up = max(
                self._max_pull_up,
                float(up_progress),
            )
            sequential_hold_sample = (
                self._last_up_goal_hold_step is not None
                and int(i) == self._last_up_goal_hold_step + 1
            )
            if (
                up_progress
                >= self._target_up_lift - self._goal_tolerance
            ):
                self._current_up_goal_hold_knots = (
                    self._current_up_goal_hold_knots + 1
                    if sequential_hold_sample
                    else 1
                )
                self._max_up_goal_hold_knots = max(
                    self._max_up_goal_hold_knots,
                    self._current_up_goal_hold_knots,
                )
            else:
                self._current_up_goal_hold_knots = 0
            self._last_up_goal_hold_step = int(i)

        approach_goal_step = approach_end - 1
        if i == approach_goal_step:
            self._approach_distance_at_end = float(
                np.linalg.norm(
                    p_hammer
                    - self._hammer_target(
                        p_nail_down,
                        i,
                        num_ctrl_steps,
                    )
                )
            )
        hammer_goal_step = hammer_end - 1
        if i == hammer_goal_step:
            self._hammer_boundary_down_depth = float(down_progress)
        transfer_goal_step = transfer_end - 1
        if not self._hammer_only() and i == transfer_goal_step:
            self._transfer_distance_at_end = float(
                np.linalg.norm(
                    p_extract - self._extract_approach_target(p_nail_up)
                )
            )
        engage_midpoint_goal_step = (
            self._engage_midpoint_end(num_ctrl_steps) - 1
        )
        if (
            not self._hammer_only()
            and i == engage_midpoint_goal_step
        ):
            self._shaft_alignment_distance_at_mid = interaction_distance
        engage_goal_step = engage_end - 1
        if not self._hammer_only() and i == engage_goal_step:
            self._engagement_distance_at_end = undercap_distance

        if i == num_ctrl_steps - 1:
            hammer_boundary_depth = (
                float(down_progress)
                if self._hammer_boundary_down_depth is None
                else float(self._hammer_boundary_down_depth)
            )
            engagement_distance = (
                undercap_distance
                if self._engagement_distance_at_end is None
                else float(self._engagement_distance_at_end)
            )
            down_goal_met = bool(
                hammer_boundary_depth
                >= self._hammer_completion_depth - self._goal_tolerance
                and down_progress
                >= self._hammer_completion_depth - self._goal_tolerance
            )
            premature_lift_ok = bool(
                self._hammer_only()
                or self._max_pre_pull_up <= self._premature_lift_tolerance
            )
            up_goal_met = bool(
                self._hammer_only()
                or self._max_up_goal_hold_knots
                >= self._success_hold_knots
            )
            semantic_contact_configured = bool(
                self._hammer_contact_pairs
            ) and (
                self._hammer_only() or bool(self._extract_contact_pairs)
            )
            hammer_contact_met = bool(
                not semantic_contact_configured
                or self._max_hammer_contact_activation
                >= self._contact_activation_threshold
            )
            extract_contact_met = bool(
                self._hammer_only()
                or not semantic_contact_configured
                or (
                    self._extract_undercap_contact_during_pull
                    and self._max_head_nail_up_pull_penetration
                    <= self._max_contact_penetration
                )
            )
            engagement_goal_met = bool(
                self._hammer_only()
                or (
                    engagement_distance
                    <= (
                        self._extraction_engage_distance
                        + self._engage_gate_tolerance
                    )
                    and self._max_head_nail_up_engage_penetration
                    <= self._max_contact_penetration
                )
            )
            self._terminal_cache = {
                "nail_down_depth": down_progress,
                "nail_up_lift": up_progress,
                "max_pull_nail_up_lift": float(self._max_pull_up),
                "success_hold_knots": int(self._success_hold_knots),
                "success_hold_achieved_knots": int(
                    self._max_up_goal_hold_knots
                ),
                "success_hold_fraction": min(
                    1.0,
                    float(self._max_up_goal_hold_knots)
                    / float(self._success_hold_knots),
                ),
                "approach_distance": self._approach_distance_at_end,
                "hammer_boundary_down_depth": hammer_boundary_depth,
                "transfer_distance": self._transfer_distance_at_end,
                "max_pre_pull_up": float(self._max_pre_pull_up),
                "shaft_alignment_distance": (
                    self._shaft_alignment_distance_at_mid
                ),
                "engagement_distance": engagement_distance,
                "down_goal_met": down_goal_met,
                "engagement_goal_met": engagement_goal_met,
                "premature_lift_ok": premature_lift_ok,
                "up_goal_met": up_goal_met,
                "hammer_contact_activation": float(
                    self._max_hammer_contact_activation
                ),
                "extract_contact_activation": float(
                    self._max_extract_contact_activation
                ),
                "extract_contact_at_engage_end": bool(
                    self._extract_contact_at_engage_end
                ),
                "extract_physical_contact_at_engage_end": bool(
                    self._extract_physical_contact_at_engage_end
                ),
                "extract_undercap_contact_at_engage_end": bool(
                    self._extract_undercap_contact_at_engage_end
                ),
                "extract_contact_activation_at_engage_end": float(
                    self._extract_contact_activation_at_engage_end
                ),
                "max_extract_engage_penetration": float(
                    self._max_extract_engage_penetration
                ),
                "extract_contact_during_pull": bool(
                    self._extract_contact_during_pull
                ),
                "extract_undercap_contact_during_pull": bool(
                    self._extract_undercap_contact_during_pull
                ),
                "max_extract_pull_penetration": float(
                    self._max_extract_pull_penetration
                ),
                "max_head_nail_down_penetration": float(
                    self._max_head_nail_down_penetration
                ),
                "max_head_nail_down_penetration_pair": (
                    None
                    if self._max_head_nail_down_penetration_pair is None
                    else list(self._max_head_nail_down_penetration_pair)
                ),
                "max_head_nail_up_engage_penetration": float(
                    self._max_head_nail_up_engage_penetration
                ),
                "max_head_nail_up_engage_penetration_pair": (
                    None
                    if self._max_head_nail_up_engage_penetration_pair is None
                    else list(self._max_head_nail_up_engage_penetration_pair)
                ),
                "max_head_nail_up_pull_penetration": float(
                    self._max_head_nail_up_pull_penetration
                ),
                "max_head_nail_up_pull_penetration_pair": (
                    None
                    if self._max_head_nail_up_pull_penetration_pair is None
                    else list(self._max_head_nail_up_pull_penetration_pair)
                ),
                "hammer_contact_met": hammer_contact_met,
                "extract_contact_met": extract_contact_met,
                "task_success": bool(
                    down_goal_met
                    and engagement_goal_met
                    and premature_lift_ok
                    and up_goal_met
                    and hammer_contact_met
                    and self._max_head_nail_down_penetration
                    <= self._max_contact_penetration
                    and self._max_head_nail_up_pull_penetration
                    <= self._max_contact_penetration
                ),
                "nail_progress_source": "q" if using_q_progress else "world_z",
                "task_phase": self._task_phase,
                "optimization_stage": self._optimization_stage,
            }
            self._refresh_terminal_contact_outcomes()

        return {
            "goal": float(goal),
            "tool_pose": float(tool_pose),
            "impact": float(impact),
            "contact": 0.0,
            "control": float(control),
        }

    def optimization_stage_acceptance(self, stage):
        """Return simple physical gates for the public staged optimizer."""

        stage = str(stage).strip().lower()
        terminal = dict(self._terminal_cache)
        common = {
            "accepted": True,
            "stage": stage,
        }
        if stage == "approach":
            raw_distance = terminal.get("approach_distance")
            distance = (
                float("inf")
                if raw_distance is None
                else float(raw_distance)
            )
            common.update(
                {
                    "accepted": (
                        np.isfinite(distance)
                        and distance
                        <= self._approach_acceptance_distance
                    ),
                    "approach_distance": distance,
                    "required_distance": (
                        self._approach_acceptance_distance
                    ),
                }
            )
            return common
        if stage == "hammer":
            depth = float(
                terminal.get("hammer_boundary_down_depth", 0.0)
            )
            final_depth = float(
                terminal.get("nail_down_depth", 0.0)
            )
            penetration = float(
                terminal.get("max_head_nail_down_penetration", 0.0)
            )
            common.update(
                {
                    "accepted": (
                        depth
                        >= self._hammer_completion_depth
                        - self._goal_tolerance
                        and final_depth
                        >= self._hammer_completion_depth
                        - self._goal_tolerance
                        and penetration <= self._max_contact_penetration
                    ),
                    "nail_down_depth": depth,
                    "final_nail_down_depth": final_depth,
                    "required_nail_down_depth": (
                        self._hammer_completion_depth
                    ),
                    "optimization_target_nail_down_depth": (
                        self._target_down_depth
                    ),
                    "goal_tolerance": self._goal_tolerance,
                    "max_head_nail_down_penetration": penetration,
                    "max_head_nail_down_penetration_pair": terminal.get(
                        "max_head_nail_down_penetration_pair"
                    ),
                    "max_allowed_penetration": (
                        self._max_contact_penetration
                    ),
                }
            )
            return common
        if stage == "transfer":
            distance = terminal.get("transfer_distance")
            distance = (
                float("inf")
                if distance is None
                else float(distance)
            )
            common.update(
                {
                    "accepted": (
                        bool(terminal.get("down_goal_met", False))
                        and distance <= self._transfer_approach_distance
                    ),
                    "hammer_preserved": bool(
                        terminal.get("down_goal_met", False)
                    ),
                    "approach_point_distance": distance,
                    "required_distance": (
                        self._transfer_approach_distance
                    ),
                }
            )
            return common
        if stage == "engage_align":
            distance = terminal.get("shaft_alignment_distance")
            distance = (
                float("inf")
                if distance is None
                else float(distance)
            )
            maximum_distance = (
                self._extraction_engage_distance
                + self._engage_gate_tolerance
            )
            common.update(
                {
                    "accepted": (
                        bool(terminal.get("down_goal_met", False))
                        and distance <= maximum_distance
                    ),
                    "hammer_preserved": bool(
                        terminal.get("down_goal_met", False)
                    ),
                    "shaft_alignment_distance": distance,
                    "required_distance": self._extraction_engage_distance,
                    "gate_tolerance": self._engage_gate_tolerance,
                    "maximum_accepted_distance": maximum_distance,
                }
            )
            return common
        if stage in ("engage_contact", "engage"):
            distance = float(
                terminal.get("engagement_distance", float("inf"))
            )
            physical_contact = bool(
                terminal.get(
                    "extract_physical_contact_at_engage_end",
                    False,
                )
            )
            undercap_contact = bool(
                terminal.get(
                    "extract_undercap_contact_at_engage_end",
                    False,
                )
            )
            penetration = float(
                terminal.get(
                    "max_head_nail_up_engage_penetration",
                    terminal.get("max_extract_engage_penetration", 0.0),
                )
            )
            maximum_distance = (
                self._extraction_engage_distance
                + self._engage_gate_tolerance
            )
            common.update(
                {
                    "accepted": (
                        bool(terminal.get("down_goal_met", False))
                        and distance <= maximum_distance
                        and penetration <= self._max_contact_penetration
                    ),
                    "hammer_preserved": bool(
                        terminal.get("down_goal_met", False)
                    ),
                    "shaft_alignment_distance": terminal.get(
                        "shaft_alignment_distance"
                    ),
                    "undercap_point_distance": distance,
                    "required_distance": self._extraction_engage_distance,
                    "gate_tolerance": self._engage_gate_tolerance,
                    "maximum_accepted_distance": maximum_distance,
                    "extract_physical_contact_at_engage_end": (
                        physical_contact
                    ),
                    "extract_undercap_contact_at_engage_end": (
                        undercap_contact
                    ),
                    "extract_contact_activation": float(
                        terminal.get("extract_contact_activation", 0.0)
                    ),
                    "extract_contact_activation_at_engage_end": float(
                        terminal.get(
                            "extract_contact_activation_at_engage_end",
                            0.0,
                        )
                    ),
                    "contact_activation_target": (
                        self._extract_contact_activation_target
                    ),
                    "max_extract_engage_penetration": float(
                        terminal.get("max_extract_engage_penetration", 0.0)
                    ),
                    "max_head_nail_up_engage_penetration": penetration,
                    "max_head_nail_up_engage_penetration_pair": (
                        terminal.get(
                            "max_head_nail_up_engage_penetration_pair"
                        )
                    ),
                    "max_allowed_penetration": (
                        self._max_contact_penetration
                    ),
                }
            )
            return common
        if stage == "pull":
            common.update(
                {
                    "accepted": bool(
                        terminal.get("task_success", False)
                    ),
                    "nail_up_lift": float(
                        terminal.get("nail_up_lift", 0.0)
                    ),
                    "required_nail_up_lift": self._target_up_lift,
                    "goal_tolerance": self._goal_tolerance,
                    "max_allowed_penetration": (
                        self._max_contact_penetration
                    ),
                    "extract_contact_met": bool(
                        terminal.get("extract_contact_met", False)
                    ),
                    "extract_contact_during_pull": bool(
                        terminal.get("extract_contact_during_pull", False)
                    ),
                    "extract_undercap_contact_during_pull": bool(
                        terminal.get(
                            "extract_undercap_contact_during_pull",
                            False,
                        )
                    ),
                    "max_extract_pull_penetration": float(
                        terminal.get("max_extract_pull_penetration", 0.0)
                    ),
                    "max_head_nail_up_pull_penetration": float(
                        terminal.get(
                            "max_head_nail_up_pull_penetration",
                            terminal.get(
                                "max_extract_pull_penetration",
                                0.0,
                            ),
                        )
                    ),
                    "max_head_nail_up_pull_penetration_pair": (
                        terminal.get(
                            "max_head_nail_up_pull_penetration_pair"
                        )
                    ),
                }
            )
            return common
        raise ValueError(
            f"Unsupported hammer_extract_nail optimization stage {stage!r}"
        )

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
        u_base = i * sub_steps * ndof_u
        if ndof_u > 0:
            u_i = np.asarray(u_i, dtype=np.float64)
            u_scale = self.action_scale(ndof_u)
            df_du[u_base : u_base + ndof_u] += (
                coef["control"] * 2.0 * u_i / (float(ndof_u) * (u_scale ** 2))
            )
            prev_control = self._control_prev_by_step.get(i)
            if prev_control is not None:
                delta = u_i - prev_control
                smooth_scale = (
                    coef["control"]
                    * self._control_smooth_weight
                    * 2.0
                    / float(ndof_u)
                    / (u_scale ** 2)
                )
                df_du[u_base : u_base + ndof_u] += smooth_scale * delta
                prev_u_base = (i - 1) * sub_steps * ndof_u
                if prev_u_base >= 0:
                    df_du[
                        prev_u_base : prev_u_base + ndof_u
                    ] -= smooth_scale * delta

        t_last = (i + 1) * sub_steps - 1
        base = t_last * ndof_var

        p_hammer, p_nail_down, p_extract, p_nail_up = self._read_task_points(variables)
        p_hammer = np.asarray(p_hammer, dtype=np.float64)
        p_nail_down = np.asarray(p_nail_down, dtype=np.float64)
        p_extract = np.asarray(p_extract, dtype=np.float64)
        p_nail_up = np.asarray(p_nail_up, dtype=np.float64)
        down_progress, up_progress, using_q_progress = self._read_nail_progress(q, p_nail_down, p_nail_up)

        _, hammer_end, _, _ = self._phase_boundaries(num_ctrl_steps)
        objective_scale = float(
            self._step_objective_active(i, num_ctrl_steps)
        )
        pose_scale2 = self._tool_pose_scale(i, num_ctrl_steps) ** 2
        hammer_base = base + self._var_hammer_base
        nail_down_base = base + self._var_nail_down_base
        extract_base = base + self._var_extract_base
        nail_up_base = base + self._var_nail_up_base

        if self._hammer_only() or i < hammer_end:
            p_target = self._hammer_target(p_nail_down, i, num_ctrl_steps)
            diff = p_hammer - p_target
            scale = (
                objective_scale
                * self._tool_pose_weight(i, num_ctrl_steps)
                * coef["tool_pose"]
                * 2.0
                / pose_scale2
            )
            df_dvar[hammer_base : hammer_base + 3] += scale * diff
            df_dvar[nail_down_base : nail_down_base + 3] -= (
                self._hammer_target_nail_scale(i, num_ctrl_steps)
                * scale
                * diff
            )
            roll_error = self._hammer_roll_error(q)
            if roll_error is not None:
                q_base = t_last * ndof_r + self._q_root_roll
                df_dq[q_base] += (
                    objective_scale
                    * self._tool_pose_weight(i, num_ctrl_steps)
                    * coef["tool_pose"]
                    * self._hammer_roll_pose_weight
                    * 2.0
                    * roll_error
                    / (self._hammer_roll_scale ** 2)
                )
        else:
            p_target = self._extract_target(p_nail_up, i, num_ctrl_steps)
            diff = p_extract - p_target
            scale = (
                objective_scale
                * self._tool_pose_weight(i, num_ctrl_steps)
                * coef["tool_pose"]
                * 2.0
                / pose_scale2
            )
            df_dvar[extract_base : extract_base + 3] += scale * diff
            df_dvar[nail_up_base : nail_up_base + 3] -= (
                self._extract_target_nail_scale(i, num_ctrl_steps) * scale * diff
            )

        (
            down_target,
            up_target,
            down_active,
            up_active,
        ) = self._scheduled_goal_components(
            i,
            num_ctrl_steps,
        )
        down_grad = (
            objective_scale
            * float(down_active)
            * coef["goal"]
            * 2.0
            * (down_progress - down_target)
            / (self._target_down_depth ** 2)
        )
        up_grad = (
            objective_scale
            * float(up_active)
            * coef["goal"]
            * 2.0
            * (up_progress - up_target)
            / (self._target_up_lift ** 2)
        )
        if using_q_progress:
            df_dq[t_last * ndof_r + self._q_nail_down] += down_grad
            df_dq[t_last * ndof_r + self._q_nail_up] += up_grad
        else:
            df_dvar[nail_down_base + 2] -= down_grad
            df_dvar[nail_up_base + 2] += up_grad

        prev_hammer = self._hammer_prev_by_step.get(i)
        if (
            objective_scale > 0.0
            and prev_hammer is not None
            and self._impact_active(i, num_ctrl_steps)
        ):
            dt = float(self._sub_steps) * 1e-3
            velocity = (p_hammer - prev_hammer) / dt
            down_speed = -float(velocity[2])
            scale2 = self._hammer_speed_scale ** 2
            impact_grad = np.zeros(3, dtype=np.float64)
            impact_grad[:2] = (
                2.0 * velocity[:2] / (scale2 * dt)
            )
            impact_grad[2] = (
                -2.0
                * (down_speed - self._hammer_target_speed)
                / (scale2 * dt)
            )
            impact_grad *= coef["impact"]
            df_dvar[hammer_base : hammer_base + 3] += impact_grad
            prev_t_last = i * sub_steps - 1
            if prev_t_last >= 0:
                prev_base = (
                    prev_t_last * ndof_var
                    + self._var_hammer_base
                )
                df_dvar[prev_base : prev_base + 3] -= impact_grad

    def rollout_diagnostics(self, runner, params):
        final_forward_diagnostics = dict(
            getattr(runner, "_last_forward_diagnostics", {}) or {}
        )
        optimizer_diagnostics = dict(
            getattr(runner, "_action_optimizer_diagnostics", {}) or {}
        )
        action, cage = runner.unpack_params(params)
        design_params = None
        if runner.optimize_design and runner.design_bundle is not None and cage is not None:
            design_params, _ = runner.apply_morphology(
                cage,
                generate_mesh=False,
            )

        self._reset_rollout_cache()
        runner.sim.reset()
        runner._reset_staged_motion_stop_runtime()

        u_all = runner.controls_from_action(action)
        terms_sum = {k: 0.0 for k in self.objective_weights().keys()}
        nail_down_start = None
        nail_down_end = None
        nail_up_start = None
        nail_up_end = None
        min_hammer_target_dist = float("inf")
        min_extract_target_dist = float("inf")
        hammer_target_dist_end = None
        extract_target_dist_end = None
        max_tool_axis_error = 0.0
        tool_axis_error_end = None
        max_root_rotation = 0.0
        root_rotation_end = None

        approach_end, hammer_end, transfer_end, engage_end = self._phase_boundaries(
            runner.num_ctrl_steps
        )
        phase_boundary_steps = {
            "approach_end": int(approach_end),
            "hammer_end": int(hammer_end),
            "transfer_end": int(transfer_end),
            "engage_end": int(engage_end),
            "rollout_end": int(runner.num_ctrl_steps),
        }
        phase_boundary_seconds = {
            key: float(step * runner.sub_steps * 1e-3)
            for key, step in phase_boundary_steps.items()
        }
        contact_filters = [
            (body1, body2)
            for body1, body2, _ in self.contact_metric_requests()
        ]
        dense_contact_tracking = bool(
            contact_filters
            and hasattr(runner.sim, "set_contact_tracking")
            and hasattr(runner.sim, "get_contact_summary")
        )
        if dense_contact_tracking:
            # This is a final/replay-only audit.  RedMax accumulates the
            # accepted state after every 1 ms physics substep, so transient
            # penetration between 100 ms control knots is not hidden.
            runner.sim.set_contact_tracking(True, contact_filters)
        geometry_audit = None
        if self._geometry_audit_enabled:
            tool_body_names = {
                str(body_name)
                for body_name, _ in (
                    tuple(self._hammer_operated_object_pairs)
                    + tuple(self._extract_operated_object_pairs)
                )
            }
            operated_body_names = {
                str(body_name)
                for _, body_name in (
                    tuple(self._hammer_operated_object_pairs)
                    + tuple(self._extract_operated_object_pairs)
                )
            }
            geometry_audit = SymmetricOverlapAudit(
                runner.model_path,
                tool_body_names=tool_body_names,
                operated_body_names=operated_body_names,
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
        for i in range(runner.num_ctrl_steps):
            u_i = u_all[i * runner.ndof_u : (i + 1) * runner.ndof_u]
            runner.advance_control_step(
                i,
                u_i,
                backward_flag=False,
                verbose=False,
            )

            variables = runner.sim.get_variables()
            q = runner.sim.get_q()
            if geometry_audit is not None:
                geometry_audit.observe(q, step=i)
            p_hammer, p_nail_down, p_extract, p_nail_up = self._read_task_points(variables)
            p_hammer = np.asarray(p_hammer, dtype=np.float64)
            p_nail_down = np.asarray(p_nail_down, dtype=np.float64)
            p_extract = np.asarray(p_extract, dtype=np.float64)
            p_nail_up = np.asarray(p_nail_up, dtype=np.float64)

            if nail_down_start is None:
                nail_down_start = p_nail_down.copy()
                nail_up_start = p_nail_up.copy()

            hammer_dist = float(np.linalg.norm(p_hammer - self._hammer_target(p_nail_down, i, runner.num_ctrl_steps)))
            extract_dist = float(np.linalg.norm(p_extract - self._extract_target(p_nail_up, i, runner.num_ctrl_steps)))
            axis_error = float(np.linalg.norm(self._tool_axis_diff(p_hammer, p_extract)))
            max_tool_axis_error = max(max_tool_axis_error, axis_error)
            tool_axis_error_end = axis_error
            root_rotation = self._root_rotation(q)
            if root_rotation is not None:
                root_rotation_norm = float(np.linalg.norm(root_rotation))
                max_root_rotation = max(max_root_rotation, root_rotation_norm)
                root_rotation_end = root_rotation_norm
            if self._hammer_only() or i < hammer_end:
                min_hammer_target_dist = min(min_hammer_target_dist, hammer_dist)
                hammer_target_dist_end = hammer_dist
            elif not self._hammer_only():
                min_extract_target_dist = min(min_extract_target_dist, extract_dist)
                extract_target_dist_end = extract_dist

            terms_i = self.compute_terms(i, runner.num_ctrl_steps, u_i, variables, q)
            if contact_filters:
                contact_metrics = runner.sim.get_contact_pair_metrics(
                    contact_filters,
                    False,
                )
                contact_terms = self.compute_contact_terms(
                    i,
                    runner.num_ctrl_steps,
                    u_i,
                    variables,
                    q,
                    contact_metrics,
                )
                for name, value in contact_terms.items():
                    terms_i[name] = terms_i.get(name, 0.0) + float(value)
            if runner._staged_loss_included(i):
                for k, v in terms_i.items():
                    terms_sum[k] = terms_sum.get(k, 0.0) + float(v)

            nail_down_end = p_nail_down.copy()
            nail_up_end = p_nail_up.copy()

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
        dense_summaries = (
            tuple(runner.sim.get_contact_summary())
            if dense_contact_tracking
            else ()
        )
        if dense_contact_tracking:
            runner.sim.set_contact_tracking(False, [])

        dense_down_penetration = 0.0
        dense_down_pair = None
        dense_up_penetration = 0.0
        dense_up_pair = None
        if dense_summaries and self._hammer_operated_object_pairs:
            dense_down_penetration, dense_down_pair = (
                self._max_pair_penetration(
                    dense_summaries,
                    self._hammer_operated_object_pairs,
                )
            )
        if dense_summaries and self._extract_operated_object_pairs:
            dense_up_penetration, dense_up_pair = (
                self._max_pair_penetration(
                    dense_summaries,
                    self._extract_operated_object_pairs,
                )
            )
        dense_penetration_ok = bool(
            dense_down_penetration <= self._max_contact_penetration
            and (
                self._hammer_only()
                or dense_up_penetration <= self._max_contact_penetration
            )
        )
        terminal.update(
            {
                "dense_contact_tracking": dense_contact_tracking,
                "max_head_nail_down_penetration_1ms": float(
                    dense_down_penetration
                ),
                "max_head_nail_down_penetration_1ms_pair": (
                    None if dense_down_pair is None else list(dense_down_pair)
                ),
                "max_head_nail_up_penetration_1ms": float(
                    dense_up_penetration
                ),
                "max_head_nail_up_penetration_1ms_pair": (
                    None if dense_up_pair is None else list(dense_up_pair)
                ),
                "operated_object_penetration_ok": dense_penetration_ok,
                "symmetric_geometry_audit": geometry_summary,
                "symmetric_geometry_ok": bool(
                    geometry_summary.get("ok", True)
                ),
            }
        )
        terminal["hammer_contact_activation"] = float(
            self._max_hammer_contact_activation
        )
        terminal["extract_contact_activation"] = float(
            self._max_extract_contact_activation
        )
        terminal["hammer_contact_met"] = bool(
            self._max_hammer_contact_activation
            >= self._contact_activation_threshold
        )
        terminal["extract_contact_met"] = bool(
            self._hammer_only()
            or (
                self._extract_undercap_contact_during_pull
                and self._max_head_nail_up_pull_penetration
                <= self._max_contact_penetration
            )
        )
        terminal["extract_contact_at_engage_end"] = bool(
            self._extract_contact_at_engage_end
        )
        terminal["extract_physical_contact_at_engage_end"] = bool(
            self._extract_physical_contact_at_engage_end
        )
        terminal["extract_undercap_contact_at_engage_end"] = bool(
            self._extract_undercap_contact_at_engage_end
        )
        terminal["extract_contact_activation_at_engage_end"] = float(
            self._extract_contact_activation_at_engage_end
        )
        terminal["max_extract_engage_penetration"] = float(
            self._max_extract_engage_penetration
        )
        terminal["extract_contact_during_pull"] = bool(
            self._extract_contact_during_pull
        )
        terminal["extract_undercap_contact_during_pull"] = bool(
            self._extract_undercap_contact_during_pull
        )
        terminal["max_extract_pull_penetration"] = float(
            self._max_extract_pull_penetration
        )
        terminal["max_head_nail_down_penetration"] = float(
            self._max_head_nail_down_penetration
        )
        terminal["max_head_nail_down_penetration_pair"] = (
            None
            if self._max_head_nail_down_penetration_pair is None
            else list(self._max_head_nail_down_penetration_pair)
        )
        terminal["max_head_nail_up_engage_penetration"] = float(
            self._max_head_nail_up_engage_penetration
        )
        terminal["max_head_nail_up_engage_penetration_pair"] = (
            None
            if self._max_head_nail_up_engage_penetration_pair is None
            else list(self._max_head_nail_up_engage_penetration_pair)
        )
        terminal["max_head_nail_up_pull_penetration"] = float(
            self._max_head_nail_up_pull_penetration
        )
        terminal["max_head_nail_up_pull_penetration_pair"] = (
            None
            if self._max_head_nail_up_pull_penetration_pair is None
            else list(self._max_head_nail_up_pull_penetration_pair)
        )
        control_knot_penetration_ok = bool(
            self._max_head_nail_down_penetration
            <= self._max_contact_penetration
            and (
                self._hammer_only()
                or (
                    self._max_head_nail_up_engage_penetration
                    <= self._max_contact_penetration
                    and self._max_head_nail_up_pull_penetration
                    <= self._max_contact_penetration
                )
            )
        )
        physical_success_without_soft_penetration = bool(
            terminal.get("down_goal_met", False)
            and (
                self._hammer_only()
                or float(
                    terminal.get("engagement_distance", float("inf"))
                )
                <= (
                    self._extraction_engage_distance
                    + self._engage_gate_tolerance
                )
            )
            and terminal.get("premature_lift_ok", False)
            and terminal.get("up_goal_met", False)
            and terminal["hammer_contact_met"]
        )
        success_policy = stage_aware_penetration_success(
            physical_success=physical_success_without_soft_penetration,
            optimize_design=bool(runner.optimize_design),
            soft_penetration_ok=bool(
                control_knot_penetration_ok and dense_penetration_ok
            ),
            severe_geometry_ok=terminal["symmetric_geometry_ok"],
        )
        terminal.update(
            {
                "control_knot_penetration_ok": (
                    control_knot_penetration_ok
                ),
                **success_policy,
            }
        )
        down_depth = terminal.get("nail_down_depth")
        up_lift = terminal.get("nail_up_lift")
        if nail_down_start is not None and nail_down_end is not None:
            if down_depth is None:
                down_depth = float(nail_down_start[2] - nail_down_end[2])
        if nail_up_start is not None and nail_up_end is not None:
            if up_lift is None:
                up_lift = float(nail_up_end[2] - nail_up_start[2])
        if down_depth is None:
            down_depth = 0.0
        if up_lift is None:
            up_lift = 0.0

        design_diag = None
        if runner.optimize_design and runner.design_bundle is not None and cage is not None:
            design_diag = runner.morphology_connection_diagnostics(cage)

        min_hammer = None if not np.isfinite(min_hammer_target_dist) else float(min_hammer_target_dist)
        min_extract = None if not np.isfinite(min_extract_target_dist) else float(min_extract_target_dist)

        return {
            "nail_down_start": [] if nail_down_start is None else nail_down_start.tolist(),
            "nail_down_end": [] if nail_down_end is None else nail_down_end.tolist(),
            "nail_down_depth": float(down_depth),
            "nail_down_goal_met": bool(
                terminal.get(
                    "down_goal_met",
                    down_depth
                    >= self._hammer_completion_depth
                    - self._goal_tolerance,
                )
            ),
            "nail_up_start": [] if nail_up_start is None else nail_up_start.tolist(),
            "nail_up_end": [] if nail_up_end is None else nail_up_end.tolist(),
            "nail_up_lift": float(up_lift),
            "max_pull_nail_up_lift": float(
                terminal.get("max_pull_nail_up_lift", up_lift)
            ),
            "nail_up_goal_met": bool(
                terminal.get(
                    "up_goal_met",
                    up_lift
                    >= self._target_up_lift - self._goal_tolerance,
                )
            ),
            "nail_progress_source": terminal.get("nail_progress_source", "world_z"),
            "target_down_depth": float(self._target_down_depth),
            "hammer_completion_depth": float(
                self._hammer_completion_depth
            ),
            "target_up_lift": float(self._target_up_lift),
            "goal_tolerance": float(self._goal_tolerance),
            "success_hold_knots": int(self._success_hold_knots),
            "success_hold_achieved_knots": int(
                terminal.get("success_hold_achieved_knots", 0)
            ),
            "success_hold_fraction": float(
                terminal.get("success_hold_fraction", 0.0)
            ),
            "premature_lift_tolerance": float(self._premature_lift_tolerance),
            "contact_activation_threshold": float(
                self._contact_activation_threshold
            ),
            "extract_contact_activation_target": float(
                self._extract_contact_activation_target
            ),
            "max_contact_penetration": float(
                self._max_contact_penetration
            ),
            "transfer_approach_distance": float(
                self._transfer_approach_distance
            ),
            "extraction_engage_distance": float(self._extraction_engage_distance),
            "engage_gate_tolerance": float(self._engage_gate_tolerance),
            "maximum_engage_gate_distance": float(
                self._extraction_engage_distance
                + self._engage_gate_tolerance
            ),
            "phase_boundary_steps": phase_boundary_steps,
            "phase_boundary_seconds_at_1ms": phase_boundary_seconds,
            "min_hammer_target_dist": min_hammer,
            "min_extract_target_dist": min_extract,
            "hammer_target_dist_end": None if hammer_target_dist_end is None else float(hammer_target_dist_end),
            "extract_target_dist_end": None if extract_target_dist_end is None else float(extract_target_dist_end),
            "max_tool_axis_error": float(max_tool_axis_error),
            "tool_axis_error_end": None if tool_axis_error_end is None else float(tool_axis_error_end),
            "max_root_rotation": float(max_root_rotation),
            "root_rotation_end": None if root_rotation_end is None else float(root_rotation_end),
            "terms_sum": {k: float(v) for k, v in terms_sum.items()},
            "per_stage_loss": final_forward_diagnostics.get(
                "per_stage_loss",
                {},
            ),
            "stage_stop": final_forward_diagnostics.get("stage_stop"),
            "optimizer_stage_termination": {
                "termination_reason": optimizer_diagnostics.get(
                    "termination_reason"
                ),
                "completed_stages": optimizer_diagnostics.get(
                    "completed_stages",
                    [],
                ),
                "last_completed_stage": optimizer_diagnostics.get(
                    "last_completed_stage"
                ),
                "failed_stage": optimizer_diagnostics.get("failed_stage"),
                "stop_policy": optimizer_diagnostics.get("stop_policy"),
            },
            "terminal": terminal,
            "task_success": bool(
                terminal.get(
                    "task_success",
                    terminal.get("down_goal_met", False)
                    and terminal.get("up_goal_met", False)
                    and terminal.get("hammer_contact_met", False)
                    and terminal.get("extract_contact_met", False),
                )
            ),
            "design_connectivity": design_diag,
            "task_phase": self._task_phase,
        }

    def print_info(self, *args):
        print(*args, flush=True)







__all__ = ["MISSION_NAME", "TaskDynamics", "TaskObjective"]
