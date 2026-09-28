#!/usr/bin/env python3
"""Freeze BASS thresholds and continuation priors from prior eval CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile from an empty sample")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _parse_bins(value: str, stage_count: int) -> List[int]:
    bins = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(bins) != stage_count or any(count < 0 for count in bins):
        raise argparse.ArgumentTypeError(
            "--bins-per-stage must provide one non-negative count per stage"
        )
    return bins


def calibrate(
    csv_path: Path,
    *,
    task_name: str,
    stage_count: int,
    bins_per_stage: Sequence[int],
    prior_strength: float,
    smoothing: float,
    threshold_sample: str = "all",
) -> dict:
    task_name = str(task_name).strip()
    if not task_name:
        raise ValueError("task_name must be non-empty")
    progress: Dict[int, List[float]] = {
        stage: [] for stage in range(stage_count)
    }
    accepted = 0
    total_rows = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            if row.get("status") != "ok":
                continue
            try:
                milestone = int(float(row["task_milestone"]))
                value = float(row["task_progress"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= milestone <= stage_count or not math.isfinite(value):
                continue
            if str(row.get("task_success", "")).lower() == "true":
                accepted += 1
                continue
            if milestone >= stage_count:
                continue
            progress[milestone].append(min(1.0, max(0.0, value)))
            accepted += 1

    thresholds = []
    threshold_sample_counts = []
    for stage, bin_count in enumerate(bins_per_stage):
        values = progress[stage]
        if threshold_sample == "all":
            threshold_values = values
        elif threshold_sample == "interior":
            # Boundary atoms describe a useful ordinal category but cannot
            # define an interior threshold. Leave them below/above cuts fitted
            # from informative progress values.
            threshold_values = [value for value in values if 0.0 < value < 1.0]
        else:
            raise ValueError("threshold_sample must be 'all' or 'interior'")
        threshold_sample_counts.append(len(threshold_values))
        if bin_count and not threshold_values:
            raise ValueError(
                f"stage {stage} requests {bin_count} bins but has no interior "
                "progress calibration data"
            )
        row = (
            [
                _quantile(threshold_values, index / float(bin_count + 1))
                for index in range(1, bin_count + 1)
            ]
            if bin_count
            else []
        )
        if any(not 0.0 < threshold < 1.0 for threshold in row) or any(
            current <= previous for previous, current in zip(row, row[1:])
        ):
            raise ValueError(
                f"stage {stage} produced degenerate progress thresholds {row}; "
                "reduce --bins-per-stage"
            )
        thresholds.append(row)

    offsets = []
    level_count = 0
    for row in thresholds:
        offsets.append(level_count)
        level_count += len(row) + 1
    category_counts = [0] * (level_count + 1)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            if str(row.get("task_success", "")).lower() == "true":
                category_counts[level_count] += 1
                continue
            try:
                milestone = int(float(row["task_milestone"]))
                value = float(row["task_progress"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= milestone <= stage_count or not math.isfinite(value):
                continue
            category = (
                level_count - 1
                if milestone == stage_count
                else offsets[milestone] + sum(
                    value >= threshold for threshold in thresholds[milestone]
                )
            )
            category_counts[category] += 1

    continuation_means = []
    at_risk_counts = []
    survival_counts = []
    for level in range(level_count):
        at_risk = sum(category_counts[level:])
        survived = sum(category_counts[level + 1 :])
        at_risk_counts.append(at_risk)
        survival_counts.append(survived)
        continuation_means.append(
            (survived + smoothing) / (at_risk + 2.0 * smoothing)
            if at_risk
            else 0.5
        )

    return {
        "format": "hot-bass-calibration-v1",
        "task_name": task_name,
        "source_eval_csv": str(csv_path),
        "source_total_rows": total_rows,
        "source_accepted_ordinal_rows": accepted,
        "stage_count": stage_count,
        "bins_per_stage": list(bins_per_stage),
        "progress_thresholds": thresholds,
        "stage_sample_counts": [len(progress[stage]) for stage in range(stage_count)],
        "threshold_sample_counts": threshold_sample_counts,
        "threshold_sampling": "{}_progress_quantiles".format(threshold_sample),
        "ordinal_level_count": level_count,
        "category_counts": category_counts,
        "hazard_at_risk_counts": at_risk_counts,
        "hazard_survival_counts": survival_counts,
        "continuation_prior_means": continuation_means,
        "prior_strengths": [float(prior_strength)],
        "prior_success_mean": math.prod(continuation_means),
        "smoothing": {
            "method": "symmetric_beta_posterior_mean",
            "alpha": float(smoothing),
            "beta": float(smoothing),
        },
        "usage": "Calibration data sets frozen priors only; do not replay it as current-run posterior evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--stage-count", type=int, required=True)
    parser.add_argument("--bins-per-stage", required=True)
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument(
        "--threshold-sample",
        choices=("all", "interior"),
        default="all",
        help="Fit progress quantiles from all failures or interior progress only.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage_count < 1:
        parser.error("--stage-count must be >= 1")
    if args.prior_strength <= 0.0 or args.smoothing <= 0.0:
        parser.error("prior strength and smoothing must be > 0")
    bins = _parse_bins(args.bins_per_stage, args.stage_count)
    result = calibrate(
        args.eval_csv,
        task_name=args.task_name,
        stage_count=args.stage_count,
        bins_per_stage=bins,
        prior_strength=args.prior_strength,
        smoothing=args.smoothing,
        threshold_sample=args.threshold_sample,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
