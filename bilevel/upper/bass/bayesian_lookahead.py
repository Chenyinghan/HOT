"""BASS milestone posteriors and Bayesian lookahead allocation.

Milestone counts update independent Beta stopping probabilities. Their
continuation probabilities define each partial state's task potential; one-
and two-query lookahead allocate evaluations across the current frontier.
Scalar task reward is separate from this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Sequence, Tuple, Union

import numpy as np


CALIBRATION_FORMAT = "hot-bass-calibration-v1"


def load_calibration_artifact(
    path: Union[str, Path],
    *,
    expected_task_name: str,
) -> Dict[str, Any]:
    """Load and validate one task-bound frozen BASS calibration artifact."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ValueError(
            "BASS requires an existing calibration artifact: {}".format(
                artifact_path
            )
        )
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            "cannot load BASS calibration artifact {}: {}".format(
                artifact_path, exc
            )
        ) from exc
    if payload.get("format") != CALIBRATION_FORMAT:
        raise ValueError(
            "BASS calibration artifact must use format {!r}; recalibrate it"
            .format(CALIBRATION_FORMAT)
        )
    task_name = str(payload.get("task_name", "")).strip()
    if task_name != str(expected_task_name).strip():
        raise ValueError(
            "BASS calibration task mismatch: artifact={!r}, search={!r}".format(
                task_name, expected_task_name
            )
        )
    required = (
        "progress_thresholds",
        "continuation_prior_means",
        "prior_strengths",
        "source_accepted_ordinal_rows",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(
            "BASS calibration artifact is missing: {}".format(", ".join(missing))
        )
    schema = MilestoneSchema(payload["progress_thresholds"])
    milestone_stopping_priors(
        payload["continuation_prior_means"],
        payload["prior_strengths"],
        level_count=schema.level_count,
    )
    if int(payload.get("stage_count", -1)) != schema.stage_count:
        raise ValueError("BASS calibration stage_count disagrees with its schema")
    if int(payload.get("ordinal_level_count", -1)) != schema.level_count:
        raise ValueError(
            "BASS calibration ordinal_level_count disagrees with its schema"
        )
    zero_ablation = payload.get("prior_source") == "uninformative_beta_1_1"
    if zero_ablation:
        if (int(payload["source_accepted_ordinal_rows"]) != 0
                or any(float(x) != 0.5 for x in payload["continuation_prior_means"])
                or any(float(x) != 2.0 for x in payload["prior_strengths"])):
            raise ValueError("invalid zero-calibration ablation prior")
    elif int(payload["source_accepted_ordinal_rows"]) <= 0:
        raise ValueError("BASS calibration contains no accepted ordinal outcomes")
    expected_mean = math.prod(float(value) for value in payload["continuation_prior_means"])
    if not math.isclose(
        float(payload.get("prior_success_mean", float("nan"))),
        expected_mean,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("BASS calibration prior_success_mean is inconsistent")
    return payload


def uninformative_prior_from_milestones(path, *, task_name):
    """Reuse calibrated milestone locations with independent Beta(1, 1) priors.

    No calibration outcomes are replayed or counted as evaluations in this run.
    The source calibration supplies only the frozen milestone locations.
    """
    source = Path(path)
    payload = load_calibration_artifact(source, expected_task_name=task_name)
    levels = MilestoneSchema(payload["progress_thresholds"]).level_count
    payload.update(prior_source="uninformative_beta_1_1",
                   milestone_schema_source=str(source),
                   source_accepted_ordinal_rows=0,
                   continuation_prior_means=[0.5] * levels,
                   prior_strengths=[2.0], prior_success_mean=0.5 ** levels)
    return payload


def _as_tuple_rows(
    rows: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in rows)


@dataclass(frozen=True)
class MilestoneSchema:
    """Fixed lexicographic ordinalization of milestone and stage progress."""

    progress_thresholds: Tuple[Tuple[float, ...], ...]
    offsets: Tuple[int, ...] = field(init=False)
    level_count: int = field(init=False)

    def __init__(self, progress_thresholds: Sequence[Sequence[float]]) -> None:
        thresholds = _as_tuple_rows(progress_thresholds)
        if not thresholds:
            raise ValueError("BASS requires thresholds for at least one task stage")
        offsets = []
        level_count = 0
        for stage, row in enumerate(thresholds):
            previous = 0.0
            for index, threshold in enumerate(row):
                if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
                    raise ValueError(
                        "BASS progress thresholds must be finite and lie in (0, 1)"
                    )
                if index and threshold <= previous:
                    raise ValueError(
                        f"BASS thresholds for stage {stage} must be strictly increasing"
                    )
                previous = threshold
            offsets.append(level_count)
            level_count += len(row) + 1
        object.__setattr__(self, "progress_thresholds", thresholds)
        object.__setattr__(self, "offsets", tuple(offsets))
        object.__setattr__(self, "level_count", int(level_count))

    @property
    def stage_count(self) -> int:
        return len(self.progress_thresholds)

    @property
    def success_category(self) -> int:
        return self.level_count

    def encode(self, evidence: Any, task_success: bool) -> int:
        """Encode one staged result as ``C in {0, ..., L}``.

        Infeasible-but-valid evaluations map to the lowest ordinal category.
        Invalid evaluations are filtered by the caller and provide no evidence.
        """

        if bool(task_success):
            return self.success_category
        if evidence is None:
            raise ValueError("BASS failure observations require StageEvidence")
        if int(evidence.stage_count) != self.stage_count:
            raise ValueError(
                "BASS stage-count mismatch: evaluator reported {} but schema has {}"
                .format(int(evidence.stage_count), self.stage_count)
            )
        if not bool(evidence.feasible):
            return 0
        milestone = int(evidence.milestone)
        if not 0 <= milestone <= self.stage_count:
            raise ValueError(
                "BASS failed outcomes require milestone in [0, stage_count]"
            )
        progress = float(evidence.progress)
        if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError("BASS task progress must be finite and lie in [0, 1]")
        if milestone == self.stage_count:
            # A separate terminal success filter may reject a trajectory after
            # every milestone passed. Keep it a failure without adding a stage.
            return self.success_category - 1
        bin_index = sum(
            progress >= threshold
            for threshold in self.progress_thresholds[milestone]
        )
        return self.offsets[milestone] + bin_index

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "bass_stage_count": self.stage_count,
            "bass_level_count": self.level_count,
            "bass_success_category": self.success_category,
            "milestone_thresholds": [
                list(row) for row in self.progress_thresholds
            ],
            "bass_stage_offsets": list(self.offsets),
        }


def _broadcast(values: Sequence[float], count: int, name: str) -> Tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if len(normalized) == 1:
        return normalized * count
    if len(normalized) != count:
        raise ValueError(f"{name} must contain one value or exactly {count} values")
    return normalized


def milestone_stopping_priors(
    continuation_means: Sequence[float],
    prior_strengths: Sequence[float],
    *,
    level_count: int,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return stopping-alpha and survival-beta parameters for all hazards."""

    means = _broadcast(
        continuation_means,
        level_count,
        "continuation_prior_means",
    )
    strengths = _broadcast(
        prior_strengths,
        level_count,
        "prior_strengths",
    )
    stop_alpha = []
    survive_beta = []
    for mean, strength in zip(means, strengths):
        if not math.isfinite(mean) or not 0.0 < mean < 1.0:
            raise ValueError("BASS continuation prior means must lie in (0, 1)")
        if not math.isfinite(strength) or strength <= 0.0:
            raise ValueError("BASS prior strengths must be finite and > 0")
        stop_alpha.append(strength * (1.0 - mean))
        survive_beta.append(strength * mean)
    return tuple(stop_alpha), tuple(survive_beta)


@dataclass
class MilestonePosterior:
    """Independent conjugate Beta stopping hazards for one canonical state."""

    prior_stop_alpha: Tuple[float, ...]
    prior_survive_beta: Tuple[float, ...]
    stop_counts: list[int] = field(init=False)
    survival_counts: list[int] = field(init=False)
    pending: int = 0
    observed_terminal_ids: set[Hashable] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self.prior_stop_alpha = tuple(float(value) for value in self.prior_stop_alpha)
        self.prior_survive_beta = tuple(
            float(value) for value in self.prior_survive_beta
        )
        if not self.prior_stop_alpha or (
            len(self.prior_stop_alpha) != len(self.prior_survive_beta)
        ):
            raise ValueError("BASS hazard prior arrays must have equal non-zero length")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in self.prior_stop_alpha + self.prior_survive_beta
        ):
            raise ValueError("BASS Beta hazard parameters must be finite and > 0")
        self.stop_counts = [0] * self.level_count
        self.survival_counts = [0] * self.level_count

    @property
    def level_count(self) -> int:
        return len(self.prior_stop_alpha)

    @property
    def observation_count(self) -> int:
        return len(self.observed_terminal_ids)

    def stop_alpha(self, level: int) -> float:
        return self.prior_stop_alpha[level] + self.stop_counts[level]

    def survive_beta(self, level: int) -> float:
        return self.prior_survive_beta[level] + self.survival_counts[level]

    def continuation_mean(self, level: int) -> float:
        alpha = self.stop_alpha(level)
        beta = self.survive_beta(level)
        return beta / (alpha + beta)

    def log_success_mean(self) -> float:
        return sum(
            math.log(self.survive_beta(level))
            - math.log(self.stop_alpha(level) + self.survive_beta(level))
            for level in range(self.level_count)
        )

    def success_mean(self) -> float:
        return math.exp(self.log_success_mean())

    def predictive_probabilities(self) -> Tuple[float, ...]:
        """Return posterior-predictive probabilities for categories 0..L."""

        probabilities = []
        reach = 1.0
        for level in range(self.level_count):
            continuation = self.continuation_mean(level)
            probabilities.append(reach * (1.0 - continuation))
            reach *= continuation
        probabilities.append(reach)
        return tuple(probabilities)

    def observe(self, category: int, terminal_id: Hashable) -> bool:
        """Apply one ordinal terminal at most once to this canonical state."""

        category = int(category)
        if not 0 <= category <= self.level_count:
            raise ValueError("BASS category must lie in [0, L]")
        if terminal_id in self.observed_terminal_ids:
            return False
        self.observed_terminal_ids.add(terminal_id)
        for level in range(min(category, self.level_count)):
            self.survival_counts[level] += 1
        if category < self.level_count:
            self.stop_counts[category] += 1
        return True

    def reserve(self, count: int = 1) -> None:
        count = int(count)
        if count < 0:
            raise ValueError("BASS pending reservation must be non-negative")
        self.pending += count

    def release(self, count: int = 1) -> None:
        count = int(count)
        if count < 0 or count > self.pending:
            raise RuntimeError("BASS pending reservation underflow")
        self.pending -= count

    @staticmethod
    def _rising_ratio(beta: float, total: float, order: int) -> float:
        ratio = 1.0
        for offset in range(order):
            ratio *= (beta + offset) / (total + offset)
        return ratio

    def success_moment(self, order: int) -> float:
        """Return exact ``E[p**order]`` for the product of Beta variables."""

        order = int(order)
        if order < 0:
            raise ValueError("BASS moment order must be non-negative")
        if order == 0:
            return 1.0
        moment = 1.0
        for level in range(self.level_count):
            alpha = self.stop_alpha(level)
            beta = self.survive_beta(level)
            moment *= self._rising_ratio(beta, alpha + beta, order)
        return moment

    @staticmethod
    def _moment_matched_beta(mean: float, second: float) -> Tuple[float, float]:
        variance = max(0.0, second - mean * mean)
        maximum = mean * (1.0 - mean)
        if variance <= 1.0e-18 or maximum <= 1.0e-18:
            concentration = 1.0e15
        else:
            concentration = max(1.0e-12, maximum / variance - 1.0)
        return mean * concentration, (1.0 - mean) * concentration

    def effective_beta(self) -> Tuple[float, float]:
        return self._moment_matched_beta(
            self.success_moment(1),
            self.success_moment(2),
        )

    def parallel_index(self, additional_pending: int = 0) -> float:
        """Moment-matched pending-worker index used only for batch scheduling."""

        pending = self.pending + int(additional_pending)
        if pending < 0:
            raise ValueError("BASS pending count must be non-negative")
        if pending == 0:
            return self.success_mean()
        alpha, beta = self.effective_beta()
        return alpha / (alpha + beta + pending)

    def _hypothetical_parameters(
        self,
        category: int,
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        category = int(category)
        if not 0 <= category <= self.level_count:
            raise ValueError("BASS category must lie in [0, L]")
        alphas = [self.stop_alpha(level) for level in range(self.level_count)]
        betas = [self.survive_beta(level) for level in range(self.level_count)]
        for level in range(min(category, self.level_count)):
            betas[level] += 1.0
        if category < self.level_count:
            alphas[category] += 1.0
        return tuple(alphas), tuple(betas)

    @staticmethod
    def _success_moment_from_parameters(
        alphas: Sequence[float],
        betas: Sequence[float],
        order: int,
    ) -> float:
        moment = 1.0
        for alpha, beta in zip(alphas, betas):
            moment *= MilestonePosterior._rising_ratio(
                float(beta),
                float(alpha) + float(beta),
                int(order),
            )
        return moment

    def hypothetical_success_mean(self, category: int) -> float:
        category = int(category)
        if not 0 <= category <= self.level_count:
            raise ValueError("BASS category must lie in [0, L]")
        if category == self.level_count:
            alphas, betas = self._hypothetical_parameters(category)
            return self._success_moment_from_parameters(alphas, betas, 1)
        return self.hypothetical_failure_success_means()[category]

    def hypothetical_failure_success_means(self) -> Tuple[float, ...]:
        """Return all failure-conditioned means in one O(L) prefix pass."""

        baseline = self.success_mean()
        survived_prefix_ratio = 1.0
        means = []
        for level in range(self.level_count):
            alpha = self.stop_alpha(level)
            beta = self.survive_beta(level)
            total = alpha + beta
            stop_ratio = total / (total + 1.0)
            means.append(baseline * survived_prefix_ratio * stop_ratio)
            survived_prefix_ratio *= (
                (beta + 1.0) * total
                / (beta * (total + 1.0))
            )
        return tuple(means)

    def hypothetical_parallel_index(self, category: int) -> float:
        alphas, betas = self._hypothetical_parameters(category)
        mean = self._success_moment_from_parameters(alphas, betas, 1)
        if self.pending == 0:
            return mean
        second = self._success_moment_from_parameters(alphas, betas, 2)
        alpha, beta = self._moment_matched_beta(mean, second)
        return alpha / (alpha + beta + self.pending)

    def two_step_value(self, best_other_mean: float) -> float:
        """Exact independent-arm two-query active-search Bellman value."""

        best_other_mean = max(0.0, float(best_other_mean))
        probabilities = self.predictive_probabilities()
        value = self.success_mean()
        hypothetical = self.hypothetical_failure_success_means()
        for category, probability in enumerate(probabilities[:-1]):
            value += probability * max(
                best_other_mean,
                hypothetical[category],
            )
        return value

    def two_step_parallel_index(self, best_other_mean: float) -> float:
        """Two-step value with moment-matched pending-worker diversification."""

        exact = self.two_step_value(best_other_mean)
        mean = self.success_mean()
        if self.pending == 0 or mean <= 0.0:
            return exact
        return exact * self.parallel_index() / mean

    def diagnostics(self) -> Dict[str, Any]:
        alpha, beta = self.effective_beta()
        probabilities = self.predictive_probabilities()
        return {
            "bass_observations": self.observation_count,
            "bass_pending": self.pending,
            "bass_success_mean": self.success_mean(),
            "bass_log_success_mean": self.log_success_mean(),
            "bass_parallel_index": self.parallel_index(),
            "bass_effective_beta_alpha": alpha,
            "bass_effective_beta_beta": beta,
            "bass_predictive_probabilities": list(probabilities),
            "bass_stop_counts": list(self.stop_counts),
            "bass_survival_counts": list(self.survival_counts),
            "bass_hazard_stop_alpha": [
                self.stop_alpha(level) for level in range(self.level_count)
            ],
            "bass_hazard_survive_beta": [
                self.survive_beta(level) for level in range(self.level_count)
            ],
        }


class MilestonePosteriorRegistry:
    """Canonical-state registry for shared BASS posteriors."""

    def __init__(
        self,
        *,
        schema: MilestoneSchema,
        continuation_means: Sequence[float],
        prior_strengths: Sequence[float],
    ) -> None:
        self.schema = schema
        self.prior_stop_alpha, self.prior_survive_beta = milestone_stopping_priors(
            continuation_means,
            prior_strengths,
            level_count=schema.level_count,
        )
        self._posteriors: Dict[Hashable, MilestonePosterior] = {}

    def posterior(self, canonical_key: Hashable) -> MilestonePosterior:
        posterior = self._posteriors.get(canonical_key)
        if posterior is None:
            posterior = MilestonePosterior(
                self.prior_stop_alpha,
                self.prior_survive_beta,
            )
            self._posteriors[canonical_key] = posterior
        return posterior

    @property
    def canonical_state_count(self) -> int:
        return len(self._posteriors)

    def two_step_parallel_scores(
        self,
        canonical_keys: Sequence[Hashable],
    ) -> np.ndarray:
        """Vectorize exact two-step scores over the sparse active frontier."""

        keys = list(canonical_keys)
        count = len(keys)
        if not count:
            return np.empty(0, dtype=np.float64)
        posteriors = [self.posterior(key) for key in keys]
        levels = self.schema.level_count
        stop_alpha = np.fromiter(
            (
                posterior.stop_alpha(level)
                for posterior in posteriors
                for level in range(levels)
            ),
            dtype=np.float64,
            count=count * levels,
        ).reshape(count, levels)
        survive_beta = np.fromiter(
            (
                posterior.survive_beta(level)
                for posterior in posteriors
                for level in range(levels)
            ),
            dtype=np.float64,
            count=count * levels,
        ).reshape(count, levels)
        pending = np.fromiter(
            (posterior.pending for posterior in posteriors),
            dtype=np.float64,
            count=count,
        )

        total = stop_alpha + survive_beta
        continuation = survive_beta / total
        mean = np.prod(continuation, axis=1)
        second = np.prod(
            survive_beta * (survive_beta + 1.0)
            / (total * (total + 1.0)),
            axis=1,
        )
        variance = np.maximum(0.0, second - mean * mean)
        maximum_variance = mean * (1.0 - mean)
        with np.errstate(divide="ignore", invalid="ignore"):
            concentration = maximum_variance / variance - 1.0
        concentration = np.where(
            (variance <= 1.0e-18) | (maximum_variance <= 1.0e-18),
            1.0e15,
            np.maximum(1.0e-12, concentration),
        )
        parallel_mean = mean * concentration / (concentration + pending)

        top_value = float(np.max(parallel_mean))
        top_candidates = np.flatnonzero(parallel_mean == top_value)
        top_index = min(top_candidates, key=lambda index: keys[int(index)])
        if count == 1:
            second_value = 0.0
        else:
            mask = np.ones(count, dtype=np.bool_)
            mask[int(top_index)] = False
            second_value = float(np.max(parallel_mean[mask]))
        best_other = np.full(count, top_value, dtype=np.float64)
        best_other[int(top_index)] = second_value

        reach = np.concatenate(
            (
                np.ones((count, 1), dtype=np.float64),
                np.cumprod(continuation[:, :-1], axis=1),
            ),
            axis=1,
        )
        stop_probability = reach * (1.0 - continuation)
        survival_ratio = (
            (survive_beta + 1.0) * total
            / (survive_beta * (total + 1.0))
        )
        survived_prefix_ratio = np.concatenate(
            (
                np.ones((count, 1), dtype=np.float64),
                np.cumprod(survival_ratio[:, :-1], axis=1),
            ),
            axis=1,
        )
        hypothetical_mean = (
            mean[:, None]
            * survived_prefix_ratio
            * total
            / (total + 1.0)
        )
        scores = mean + np.sum(
            stop_probability
            * np.maximum(best_other[:, None], hypothetical_mean),
            axis=1,
        )
        # Pending probes alter only the batch-scheduling approximation.
        return np.where(
            (pending > 0.0) & (mean > 0.0),
            scores * parallel_mean / mean,
            scores,
        )

    def diagnostics(self) -> Dict[str, Any]:
        continuation_means = [
            beta / (alpha + beta)
            for alpha, beta in zip(
                self.prior_stop_alpha,
                self.prior_survive_beta,
            )
        ]
        strengths = [
            alpha + beta
            for alpha, beta in zip(
                self.prior_stop_alpha,
                self.prior_survive_beta,
            )
        ]
        return {
            **self.schema.diagnostics(),
            "bass_canonical_state_count": self.canonical_state_count,
            "continuation_prior_means": continuation_means,
            "prior_strengths": strengths,
            "bass_prior_success_mean": math.prod(continuation_means),
        }
