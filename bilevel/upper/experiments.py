#!/usr/bin/env python3
"""Resolve and run the paper's calibration, lookahead, or uniform ablation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks import CANONICAL_TASK_NAMES
from bilevel.upper.bass.bayesian_lookahead import uninformative_prior_from_milestones


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--task", choices=CANONICAL_TASK_NAMES, required=True)
    p.add_argument("--dag", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration-budget", type=int, choices=(9800, 980, 0), default=9800)
    p.add_argument("--lookahead", type=int, choices=(1, 2), default=2)
    p.add_argument("--uniform-random", action="store_true")
    p.add_argument("--milestone-schema", type=Path,
                   help="Full-calibration artifact supplying fixed thresholds for zero calibration")
    p.add_argument("--max-evaluations", type=int, default=40000)
    p.add_argument("--workers", type=int, default=140)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents:
        p.error("output must be repository-local")
    if args.workers < 1 or args.max_evaluations < 1:
        p.error("workers and max-evaluations must be positive")
    if not args.uniform_random and args.max_evaluations < args.calibration_budget:
        p.error("evaluation budget must cover calibration")
    if not args.uniform_random and args.calibration_budget == 0 and args.milestone_schema is None:
        p.error("zero calibration requires --milestone-schema; physical gates are never removed")
    output.mkdir(parents=True, exist_ok=False)
    payload = json.loads((ROOT / "tasks" / args.task / "config.json").read_text())
    search = payload.setdefault("bass", {})
    search.update(structural_dag_path=str(args.dag.resolve()), threads=1,
                  eval_workers=args.workers, iteration_budget=args.max_evaluations,
                  seed=args.seed, acquisition=("uniform_random" if args.uniform_random else
                                              "bass_n%d" % args.lookahead),
                  calibration_budget=0 if args.uniform_random else args.calibration_budget,
                  partial_state_transpositions=True, physical_dedup_mode="enforce",
                  scheduler_v2=True, parallelization_mode="shared_tree", reward_mode="bounded_task")
    if args.calibration_budget == 0 and not args.uniform_random:
        prior = uninformative_prior_from_milestones(args.milestone_schema, task_name=args.task)
        prior_path = output / "uninformative_prior.json"
        prior_path.write_text(json.dumps(prior, indent=2) + "\n")
        search.update(calibration_artifact=str(prior_path),
                      milestone_thresholds=prior["progress_thresholds"],
                      continuation_prior_means=prior["continuation_prior_means"],
                      prior_strengths=prior["prior_strengths"])
    payload.update(cache_dir=str(output / "cache"), output_dir=str(output / "output"),
                   replay_dir=str(output / "replay"), best_xml_out=str(output / "best.xml"),
                   best_run_json=str(output / "best_run.json"))
    config = output / "config.json"
    config.write_text(json.dumps(payload, indent=2) + "\n")
    command = [sys.executable, str(ROOT / "run_bilevel_search.py"), "--task-json", str(config)]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(json.dumps(command))
    if not args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
