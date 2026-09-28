"""Uniform sampling without replacement over physical terminals and root modes."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import random
import time

from .DAG.graph import (StaticVirtualRootProduct, StaticRuntimeOutcome,
                          StaticSearchResult, _coerce_runtime_outcome)


def run_uniform_search(tree, config, evaluator):
    runtime = StaticVirtualRootProduct(tree, config)
    remaining = [tree.physical_count] * len(runtime.runtimes)
    rng = random.Random(config.seed)
    budget = int(config.iteration_budget) * int(config.threads)
    if budget <= 0:
        raise ValueError("Uniform ablation requires a finite evaluation budget")
    budget = min(budget, sum(remaining))
    workers = int(config.eval_workers or config.threads)
    deadline = (time.monotonic() + config.search_time_budget
                if config.search_time_budget else float("inf"))
    submitted = completed = valid = 0
    best = best_probe = None
    first_success_position = None
    inflight = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while inflight or submitted < budget:
            while len(inflight) < workers and submitted < budget and time.monotonic() < deadline:
                draw = rng.randrange(sum(remaining))
                index = 0
                while draw >= remaining[index]:
                    draw -= remaining[index]
                    index += 1
                probe = runtime.runtimes[index].select_warmup_probe()
                if probe is None:
                    raise RuntimeError("Uniform terminal accounting disagrees with DAG state")
                probe = runtime._tag(probe, runtime.rotation_modes[index])
                runtime.reserve_probe(probe)
                remaining[index] -= 1
                submitted += 1
                future = pool.submit(evaluator, runtime.sequence_for_probe(probe))
                inflight[future] = (probe, submitted)
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for future in done:
                probe, position = inflight.pop(future)
                try:
                    outcome = _coerce_runtime_outcome(future.result())
                except Exception:
                    outcome = StaticRuntimeOutcome(float("inf"), 0.0, False, None, None)
                runtime.complete_probe(probe, outcome)
                completed += 1
                valid += int(outcome.valid)
                if outcome.valid and outcome.task_success:
                    first_success_position = min(position, first_success_position or position)
                key = lambda value: (bool(value.task_success), value.reward, -value.score)
                if outcome.valid and (best is None or key(outcome) > key(best)):
                    best, best_probe = outcome, probe
    sequence = [] if best_probe is None else runtime.sequence_for_probe(best_probe)
    # Function counts are optional result metadata; the canonical evaluator
    # records the full grammar sequence independently.
    return StaticSearchResult(sequence, float("inf") if best is None else best.score,
                              float("-inf") if best is None else best.reward,
                              submitted, completed, valid, None,
                              {**runtime.diagnostics(), "algorithm": "uniform_random_terminal",
                               "structural_dag_enabled": True,
                               "structural_dag_root_resolved": runtime.root_resolved,
                               "physical_terminal_count": runtime.evaluation_count,
                               "first_success_sample_position": first_success_position,
                               "calibration_evaluations": 0})
