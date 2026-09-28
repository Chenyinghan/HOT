"""Canonical Handle-root bi-level framework interfaces."""

from .contracts import (
    HandleSpec,
    MorphologySpec,
    RuntimeSpec,
    SearchSpec,
    TaskSpec,
)
from .parameterization import (
    PARAMETER_ARTIFACT_SCHEMA,
    PARAMETER_ARTIFACT_SCHEMA_VERSION,
    UNIFIED_PARAMETERIZATION_ID,
    JointParameterArtifactMetadata,
    MorphologyBlock,
    MorphologyCollisionDecision,
    MorphologyCollisionPolicy,
    MorphologyLayout,
    MorphologyParameterization,
    MorphologyRetractionResult,
    RedMaxParameterSlice,
    UnifiedConnectedHeadMorphology,
)
from .results import BilevelResult, EvaluationStatus
from .lower.optimizers import (
    OPTIMIZER_REGISTRY,
    OptimizationMode,
    OptimizerPolicy,
    OptimizerRegistry,
    OptimizerStrategy,
    register_optimizer,
)
from .lower.evaluation import (
    EVALUATION_IDENTITY_SCHEMA,
    EVALUATION_IDENTITY_SCHEMA_VERSION,
    EVALUATION_RESULT_SCHEMA,
    EVALUATION_RESULT_SCHEMA_VERSION,
    EvaluationIdentity,
    build_evaluation_identity,
)

__all__ = [
    "BilevelResult",
    "EvaluationStatus",
    "EvaluationIdentity",
    "EVALUATION_IDENTITY_SCHEMA",
    "EVALUATION_IDENTITY_SCHEMA_VERSION",
    "EVALUATION_RESULT_SCHEMA",
    "EVALUATION_RESULT_SCHEMA_VERSION",
    "HandleSpec",
    "JointParameterArtifactMetadata",
    "MorphologyBlock",
    "MorphologyCollisionDecision",
    "MorphologyCollisionPolicy",
    "MorphologyLayout",
    "MorphologyParameterization",
    "MorphologyRetractionResult",
    "MorphologySpec",
    "OPTIMIZER_REGISTRY",
    "OptimizationMode",
    "OptimizerPolicy",
    "OptimizerRegistry",
    "OptimizerStrategy",
    "PARAMETER_ARTIFACT_SCHEMA",
    "PARAMETER_ARTIFACT_SCHEMA_VERSION",
    "RedMaxParameterSlice",
    "RuntimeSpec",
    "SearchSpec",
    "TaskSpec",
    "UNIFIED_PARAMETERIZATION_ID",
    "UnifiedConnectedHeadMorphology",
    "build_evaluation_identity",
    "register_optimizer",
]
