"""BASS search over a canonical physical-state DAG."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple
from .actions import Action
from .config import BASSConfig
from .io_assets import AssetSpec
AABB = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


@dataclass(frozen=True)
class BASSEvaluation:
    """Task-aware evaluation while retaining the raw low-level objective."""

    score: float
    reward: float
    valid: bool = True
    task_milestone: int | None = None
    task_stage_count: int | None = None
    task_progress: float = 0.0
    task_success: bool | None = None
    task_feasible: bool = True

@dataclass
class SearchResult:
    """Output of one or multi-threaded BASS run.

    Attributes:
        best_sequence: Best complete sequence found.
        best_score: Objective of best sequence (lower is better).
        total_iterations: Number of tree iterations performed.
        completed_candidates: Number of complete skeletons evaluated.
        worker_summaries: Per-worker debug summary dictionaries.
        valid_completed_candidates: Completed candidates satisfying optional
            function-count constraints.
        rejected_count_mismatch: Completed candidates rejected by function-count
            constraints.
        best_function_count: Function-group/end-effector count for best_sequence.
        constraint_diagnostics: Search-level diagnostics for configured count
            target/margin.
    """

    best_sequence: List[Action]
    best_score: float
    total_iterations: int
    completed_candidates: int
    best_reward: float = float("-inf")
    worker_summaries: List[dict] = field(default_factory=list)
    valid_completed_candidates: int = 0
    rejected_count_mismatch: int = 0
    best_function_count: Optional[int] = None
    constraint_diagnostics: dict = field(default_factory=dict)

def run_bass(
    assets: Sequence[AssetSpec],
    config: BASSConfig,
    evaluator: Optional[Callable[[list[Action]], float]] = None,
    initial_forbidden_boxes: Optional[List[AABB]] = None,
    initial_root_box: Optional[AABB] = None,
) -> SearchResult:
    """Run upper-level BASS skeleton search with parallel workers.

    Args:
        assets: Available link primitive assets.
        config: BASS hyperparameters.
        evaluator: Optional objective function. If omitted, this function
            delegates to evaluate_sequence(), which raises NotImplementedError
            until lower-level optimization is integrated.
        initial_forbidden_boxes: Optional fixed occupancy AABBs in the synthetic
            root coordinate frame. Candidate links may not overlap them. This is
            intended for external geometry such as a finger template around the
            attachment tip.
        initial_root_box: Optional AABB for the synthetic root/attachment parent.
            Candidate links may overlap this box only when it is their direct
            parent, matching the existing parent-child collision convention.

    Returns:
        Aggregated SearchResult across all workers.
    """
    if config.acquisition not in {"bass_n1", "bass_n2", "uniform_random"}:
        raise ValueError("Select BASS N=1/N=2 or the uniform_random ablation")
    config.validate()
    if not assets:
        raise ValueError("assets must not be empty")
    if not any(asset.asset_id == config.root_asset_id for asset in assets):
        raise ValueError(
            f"root asset id {config.root_asset_id!r} was not found in loaded assets"
        )

    if evaluator is None:
        raise ValueError("BASS requires a physical evaluator")
    score_fn = evaluator
    forbidden = list(initial_forbidden_boxes or [])
    if config.structural_dag_path is not None:
        from .DAG.graph import StructuralDAG, run_structural_dag_search

        static_startup_started = time.perf_counter()
        tree = StructuralDAG.load(
            config.structural_dag_path,
            mmap=bool(config.structural_dag_mmap),
        )
        static_loaded_at = time.perf_counter()
        tree.validate_runtime()
        static_validated_at = time.perf_counter()
        tree.assert_compatible(
            assets,
            config,
            initial_root_box=initial_root_box,
            initial_forbidden_boxes=forbidden,
        )
        static_compatible_at = time.perf_counter()
        print(
            "[bass] static physical DAG loaded "
            f"path={tree.root} nodes={tree.node_count} "
            f"edges={tree.edge_count_total} terminals={tree.terminal_alias_count} "
            f"physical={tree.physical_count} "
            f"load_seconds={static_loaded_at - static_startup_started:.6f} "
            f"runtime_validate_seconds={static_validated_at - static_loaded_at:.6f} "
            f"compatibility_seconds={static_compatible_at - static_validated_at:.6f} "
            f"preprocess_seconds={static_compatible_at - static_loaded_at:.6f}",
            flush=True,
        )
        if config.acquisition == "uniform_random":
            from .uniform import run_uniform_search
            static_result = run_uniform_search(tree, config, score_fn)
        else:
            static_result = run_structural_dag_search(tree, config, score_fn)
        return SearchResult(
            best_sequence=static_result.best_sequence,
            best_score=static_result.best_score,
            best_reward=static_result.best_reward,
            total_iterations=static_result.total_iterations,
            completed_candidates=static_result.completed_candidates,
            valid_completed_candidates=static_result.valid_completed_candidates,
            rejected_count_mismatch=0,
            best_function_count=static_result.best_function_count,
            constraint_diagnostics=static_result.diagnostics,
            worker_summaries=[],
        )
    raise ValueError("BASS requires a canonical DAG; build it before searching")
