"""Upper-level BASS skeleton search package.

This package implements constructive BASS over sequence grammar actions:
optional `SelectRootRotation(mode)`, then
`AddLink(asset_id, p, d, f, q, start_function_group)` and `End`.
"""

from .actions import Action, AddLink, End, SelectRootRotation
from .config import BASSConfig
from .bayesian_lookahead import (
    MilestonePosterior,
    MilestonePosteriorRegistry,
    MilestoneSchema,
    milestone_stopping_priors,
)
from .io_assets import (
    AssetSpec,
    load_assets,
    resolve_asset_id,
    select_assets,
)
from .search import (
    SearchResult,
    run_bass,
)
from .DAG.canonical import (
    PACKED_SIGNATURE_VERSION,
    PHYSICAL_SIGNATURE_EPS,
    canonical_cuboid,
    pack_physical_signature,
    packed_signature_digest,
    physical_function_signature,
    physical_partial_state_signature,
)
from .DAG.graph import (
    StructuralDAG,
    StructuralDAGBuildResult,
    build_structural_dag,
)

__all__ = [
    "Action",
    "AddLink",
    "End",
    "SelectRootRotation",
    "BASSConfig",
    "MilestonePosterior",
    "MilestonePosteriorRegistry",
    "MilestoneSchema",
    "milestone_stopping_priors",
    "AssetSpec",
    "SearchResult",
    "load_assets",
    "resolve_asset_id",
    "select_assets",
    "run_bass",
    "PHYSICAL_SIGNATURE_EPS",
    "PACKED_SIGNATURE_VERSION",
    "canonical_cuboid",
    "pack_physical_signature",
    "packed_signature_digest",
    "physical_function_signature",
    "physical_partial_state_signature",
    "StructuralDAG",
    "StructuralDAGBuildResult",
    "build_structural_dag",
]
