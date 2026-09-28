"""Configuration types for BASS skeleton search."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .actions import ROOT_ROTATION_MODES


@dataclass
class BASSConfig:
    """Canonical grammar, evaluator budget and BASS posterior configuration."""

    threads: int = 4
    max_head_links: int | None = None
    max_depth: int = 6
    iteration_budget: int = 0
    search_time_budget: float | None = None
    calibration_budget: int = 0
    asset_volume_coeff: float = 0.0
    seed: int = 0
    max_actions_per_expansion: int = 0
    rollout_policy: str = "uniform"
    rollout_end_prob_by_depth: list[float] | None = None
    rollout_addlink_prob_by_depth: list[float] | None = None
    rollout_branching_penalty_by_depth: list[float] | None = None
    parallelization_mode: str = "independent"
    target_function_count: int | None = None
    function_count_margin: int = 0
    function_group_depth_delta: int | None = 0
    root_asset_id: str = "root/universal_handle"
    root_blocked_face: int | None = 2
    root_rotation_options: tuple[str, ...] = ()
    eval_workers: int | None = None
    reward_mode: str = "negative_cost"
    partial_state_transpositions: bool = False
    structural_factorization: bool = False
    scheduler_v2: bool = False
    scheduler_health_log_interval: float = 60.0
    scheduler_supply_critical_fraction: float = 0.7
    scheduler_supply_critical_seconds: float = 300.0
    scheduler_supply_recent_new_seconds: float = 600.0
    scheduler_supply_rate_window_seconds: float = 300.0
    scheduler_ready_low_watermark_fraction: float = 0.5
    scheduler_ready_high_watermark_fraction: float = 1.0
    scheduler_occupancy_warning_fraction: float = 0.8
    scheduler_occupancy_recovery_fraction: float = 0.9
    scheduler_occupancy_warning_seconds: float = 30.0
    scheduler_occupancy_recovery_seconds: float = 5.0
    acquisition: str | None = None
    milestone_thresholds: tuple[tuple[float, ...], ...] = ()
    continuation_prior_means: tuple[float, ...] = ()
    prior_strengths: tuple[float, ...] = (2.0,)
    calibration_artifact: str | None = None
    calibration_bins_per_stage: tuple[int, ...] = ()
    calibration_smoothing: float = 0.5
    calibration_threshold_sample: str = "interior"
    physical_dedup_mode: str = "off"
    physical_signature_eps: float = 1e-8
    function_semantic_labels: tuple[str, ...] = ()
    structural_dag_path: str | None = None
    structural_dag_mmap: bool = True
    diagnostics_jsonl: str | None = None
    log_progress: bool = False

    @property
    def max_total_links(self) -> int:
        """Return the internal root-inclusive link limit."""

        if self.max_head_links is not None:
            return int(self.max_head_links) + 1
        return int(self.max_depth)

    def validate(self) -> None:
        """Validate hyperparameter ranges.

        Raises:
            ValueError: If any field is invalid.
        """
        if self.threads < 1:
            raise ValueError("threads must be >= 1")
        if self.eval_workers is not None and self.eval_workers < 1:
            raise ValueError("eval_workers must be >= 1 when provided")
        if self.reward_mode not in {"negative_cost", "bounded_task"}:
            raise ValueError(
                "reward_mode must be one of {'negative_cost', 'bounded_task'}"
            )
        if self.scheduler_health_log_interval < 0.0:
            raise ValueError("scheduler_health_log_interval must be >= 0")
        if not 0.0 <= self.scheduler_supply_critical_fraction <= 1.0:
            raise ValueError(
                "scheduler_supply_critical_fraction must lie in [0, 1]"
            )
        if self.scheduler_supply_critical_seconds < 0.0:
            raise ValueError("scheduler_supply_critical_seconds must be >= 0")
        if self.scheduler_supply_recent_new_seconds <= 0.0:
            raise ValueError(
                "scheduler_supply_recent_new_seconds must be > 0"
            )
        if self.scheduler_supply_rate_window_seconds <= 0.0:
            raise ValueError(
                "scheduler_supply_rate_window_seconds must be > 0"
            )
        if not (
            0.0
            <= self.scheduler_ready_low_watermark_fraction
            <= self.scheduler_ready_high_watermark_fraction
        ):
            raise ValueError(
                "scheduler READY watermarks must satisfy 0 <= low <= high"
            )
        if self.scheduler_ready_high_watermark_fraction <= 0.0:
            raise ValueError(
                "scheduler_ready_high_watermark_fraction must be > 0"
            )
        if not (
            0.0
            <= self.scheduler_occupancy_warning_fraction
            <= self.scheduler_occupancy_recovery_fraction
            <= 1.0
        ):
            raise ValueError(
                "scheduler occupancy fractions must satisfy "
                "0 <= warning <= recovery <= 1"
            )
        if self.scheduler_occupancy_warning_seconds <= 0.0:
            raise ValueError("scheduler_occupancy_warning_seconds must be > 0")
        if self.scheduler_occupancy_recovery_seconds <= 0.0:
            raise ValueError("scheduler_occupancy_recovery_seconds must be > 0")
        if self.acquisition not in {None, "uniform_random", "bass_n1", "bass_n2"}:
            raise ValueError("acquisition must be bass_n1, bass_n2, or uniform_random")
        self.milestone_thresholds = tuple(
            tuple(float(threshold) for threshold in row)
            for row in self.milestone_thresholds
        )
        self.continuation_prior_means = tuple(
            float(value) for value in self.continuation_prior_means
        )
        self.prior_strengths = tuple(
            float(value) for value in self.prior_strengths
        )
        self.calibration_bins_per_stage = tuple(
            int(value) for value in self.calibration_bins_per_stage
        )
        bass_modes = {"bass_n1", "bass_n2"}
        if self.acquisition in bass_modes:
            from .bayesian_lookahead import MilestoneSchema, milestone_stopping_priors

            integrated_calibration = not self.calibration_artifact
            if integrated_calibration:
                if self.calibration_budget <= 0:
                    raise ValueError(
                        "fresh BASS requires either a calibration artifact "
                        "or positive calibration_budget"
                    )
                if any(
                    value < 0 for value in self.calibration_bins_per_stage
                ):
                    raise ValueError(
                        "integrated BASS calibration bin counts must be "
                        "non-negative"
                    )
                if self.milestone_thresholds or self.continuation_prior_means:
                    raise ValueError(
                        "integrated BASS calibration cannot also receive frozen "
                        "thresholds or continuation priors"
                    )
                if len(self.prior_strengths) != 1:
                    raise ValueError(
                        "integrated BASS calibration requires one shared prior strength"
                    )
                if (
                    not math.isfinite(self.calibration_smoothing)
                    or self.calibration_smoothing <= 0.0
                ):
                    raise ValueError("BASS calibration smoothing must be finite and > 0")
                if self.calibration_threshold_sample not in {"all", "interior"}:
                    raise ValueError(
                        "BASS calibration threshold sample must be 'all' or 'interior'"
                    )
            else:
                schema = MilestoneSchema(self.milestone_thresholds)
                milestone_stopping_priors(
                    self.continuation_prior_means,
                    self.prior_strengths,
                    level_count=schema.level_count,
                )
            if self.reward_mode != "bounded_task":
                raise ValueError(
                    "BASS requires reward_mode='bounded_task'"
                )
            if self.parallelization_mode != "shared_tree":
                raise ValueError(
                    "BASS requires parallelization_mode='shared_tree'"
                )
            if not self.partial_state_transpositions:
                raise ValueError(
                    "BASS requires partial_state_transpositions=True"
                )
            if self.physical_dedup_mode != "enforce":
                raise ValueError(
                    "BASS requires physical_dedup_mode='enforce'"
                )
            if not self.scheduler_v2:
                raise ValueError("BASS requires scheduler_v2=True")
            if self.structural_dag_path is None:
                raise ValueError("BASS currently requires --structural-dag-path")
            if self.calibration_artifact and self.calibration_budget != 0:
                raise ValueError(
                    "BASS with a frozen artifact requires "
                    "calibration_budget=0"
                )
        if self.physical_dedup_mode not in {"off", "observe", "enforce"}:
            raise ValueError(
                "physical_dedup_mode must be one of {'off', 'observe', 'enforce'}"
            )
        if (
            self.physical_signature_eps <= 0.0
            or not math.isfinite(self.physical_signature_eps)
        ):
            raise ValueError("physical_signature_eps must be positive and finite")
        semantic_labels = tuple(
            str(label).strip() for label in self.function_semantic_labels
        )
        if any(not label for label in semantic_labels):
            raise ValueError("function_semantic_labels must be non-empty strings")
        if len(set(semantic_labels)) != len(semantic_labels):
            raise ValueError("function_semantic_labels must not contain duplicates")
        if (
            semantic_labels
            and self.target_function_count is not None
            and len(semantic_labels)
            < self.target_function_count + self.function_count_margin
        ):
            raise ValueError(
                "function_semantic_labels must cover the maximum allowed function count"
            )
        self.function_semantic_labels = semantic_labels
        if self.structural_dag_path is not None:
            normalized_structural_dag_path = str(self.structural_dag_path).strip()
            if not normalized_structural_dag_path:
                raise ValueError("structural_dag_path must be non-empty when provided")
            self.structural_dag_path = normalized_structural_dag_path
        if self.max_head_links is not None and self.max_head_links < 1:
            raise ValueError("max_head_links must be >= 1 when provided")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.iteration_budget < 0:
            raise ValueError("iteration_budget must be >= 0; use 0 for unbounded search")
        if self.search_time_budget is not None and self.search_time_budget < 0.0:
            raise ValueError("search_time_budget must be >= 0 when provided")
        if self.calibration_budget < 0:
            raise ValueError("calibration_budget must be >= 0")
        if self.max_actions_per_expansion < 0:
            raise ValueError("max_actions_per_expansion must be >= 0; use 0 for uncapped expansion")
        if self.rollout_policy not in {"uniform", "depth_biased", "end_biased"}:
            raise ValueError(
                "rollout_policy must be one of "
                "{'uniform', 'depth_biased', 'end_biased'}"
            )
        if self.parallelization_mode not in {"independent", "shared_tree"}:
            raise ValueError("parallelization_mode must be one of {'independent', 'shared_tree'}")
        if self.calibration_budget and self.parallelization_mode != "shared_tree":
            raise ValueError(
                "calibration_budget requires parallelization_mode='shared_tree'"
            )
        if self.target_function_count is not None and self.target_function_count < 0:
            raise ValueError("target_function_count must be >= 0 when provided")
        if self.function_count_margin < 0:
            raise ValueError("function_count_margin must be >= 0")
        if self.function_group_depth_delta is not None and self.function_group_depth_delta < 0:
            raise ValueError("function_group_depth_delta must be >= 0 when provided")
        blocked_face = self.root_blocked_face
        if blocked_face is not None and not (0 <= blocked_face <= 5):
            raise ValueError("root_blocked_face must be in [0, 5] or None")
        if not self.root_asset_id:
            raise ValueError("root_asset_id must be a non-empty string")
        normalized_rotation_options = tuple(
            str(value).strip().lower()
            for value in self.root_rotation_options
        )
        if len(set(normalized_rotation_options)) != len(normalized_rotation_options):
            raise ValueError("root_rotation_options must not contain duplicates")
        invalid_rotation_options = set(normalized_rotation_options) - set(
            ROOT_ROTATION_MODES
        )
        if invalid_rotation_options:
            raise ValueError(
                "root_rotation_options contains unsupported modes: "
                f"{sorted(invalid_rotation_options)}"
            )
        self.root_rotation_options = normalized_rotation_options



        self._validate_optional_float_sequence(
            self.rollout_end_prob_by_depth,
            "rollout_end_prob_by_depth",
            min_value=0.0,
            max_value=1.0,
        )
        self._validate_optional_float_sequence(
            self.rollout_addlink_prob_by_depth,
            "rollout_addlink_prob_by_depth",
            min_value=0.0,
            max_value=1.0,
        )
        self._validate_optional_float_sequence(
            self.rollout_branching_penalty_by_depth,
            "rollout_branching_penalty_by_depth",
            min_value=0.0,
            max_value=1.0,
        )

    @staticmethod
    def _validate_optional_float_sequence(
        values: list[float] | None,
        name: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """Validate optional sequences of numeric hyperparameters."""
        if values is None:
            return
        if not values:
            raise ValueError(f"{name} must be non-empty when provided")
        for value in values:
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must contain numeric values")
            numeric_value = float(value)
            if min_value is not None and numeric_value < min_value:
                raise ValueError(f"{name} values must be >= {min_value}")
            if max_value is not None and numeric_value > max_value:
                raise ValueError(f"{name} values must be <= {max_value}")
