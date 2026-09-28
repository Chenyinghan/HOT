"""The unified connected-Head parameterization and its geometric protocol."""
from .function_group import (
    FunctionGroupContactSet,
    build_function_group_contact_set,
    point_squared_distance,
)
from .collision import (
    CollisionReport,
    check_design_collision,
    check_design_params_collision,
)
from .design import (
    build_connected_head_topology,
    build_connected_direct_planar_hex_design_bundle,
    connected_direct_planar_hex_bounds_for_bundle,
)
from .topology import HeadConnection, HeadTopology, HeadTopologyBlock
from .geometry import (
    DIRECT_PARAM_DIM,
    GENERIC_DESIGN_PROTOCOL,
    constraint_jacobian,
    constraints as hexahedron_constraints,
    normalize_face_mask,
    project_tangent,
    retract,
    validate_hexahedron,
    vertices_from_q,
)
from .base import (
    UNIFIED_PARAMETERIZATION_ID,
    MorphologyBlock,
    MorphologyCollisionDecision,
    MorphologyCollisionPolicy,
    MorphologyLayout,
    MorphologyParameterization,
    MorphologyRetractionResult,
    RedMaxParameterSlice,
)
from .connected_head import UnifiedConnectedHeadMorphology
from .artifacts import (
    CONNECTED_DIRECT_PARAMETERIZATION_ID,
    PARAMETER_ARTIFACT_SCHEMA,
    PARAMETER_ARTIFACT_SCHEMA_VERSION,
    JointParameterArtifactMetadata,
    legacy_morphology_layout_manifest,
    metadata_for_runner,
    normalize_parameters_for_runner,
    parse_parameter_artifact_metadata,
    read_parameter_artifact_metadata,
    stable_layout_fingerprint,
    validate_parameter_vector,
)
from .runtime import (
    JointParameterLayout,
    MorphologyRuntimeBridge,
    MorphologyRuntimeContext,
)


CANONICAL_DESIGN_PROTOCOL = "connected_direct_planar_hexahedron"


def _protocol_from_config(task_config):
    task_config = task_config or {}
    protocol = str(
        task_config.get(
            "generic_design_protocol",
            task_config.get("design_protocol", CANONICAL_DESIGN_PROTOCOL),
        )
    )
    if protocol != CANONICAL_DESIGN_PROTOCOL:
        raise ValueError(
            f"Unsupported generic_design_protocol {protocol!r}; the canonical "
            f"framework only supports {CANONICAL_DESIGN_PROTOCOL!r}"
        )
    return protocol


def build_design_bundle(xml_path, sim, task_config=None):
    _protocol_from_config(task_config)
    from .design import (
        build_connected_direct_planar_hex_design_bundle,
    )

    return build_connected_direct_planar_hex_design_bundle(
        xml_path,
        sim,
        task_config,
    )


def cage_bounds_for_bundle(bundle, ndof_cage, **kwargs):
    protocol = getattr(bundle, "generic_design_protocol", None)
    if protocol != CANONICAL_DESIGN_PROTOCOL:
        raise ValueError(
            f"Expected canonical design bundle {CANONICAL_DESIGN_PROTOCOL!r}, "
            f"got {protocol!r}"
        )
    from .design import (
        connected_direct_planar_hex_bounds_for_bundle,
    )

    return connected_direct_planar_hex_bounds_for_bundle(
        bundle,
        ndof_cage,
        **kwargs,
    )


__all__ = [
    "CANONICAL_DESIGN_PROTOCOL",
    "CONNECTED_DIRECT_PARAMETERIZATION_ID",
    "CollisionReport",
    "DIRECT_PARAM_DIM",
    "FunctionGroupContactSet",
    "GENERIC_DESIGN_PROTOCOL",
    "HeadConnection",
    "HeadTopology",
    "HeadTopologyBlock",
    "JointParameterArtifactMetadata",
    "JointParameterLayout",
    "MorphologyBlock",
    "MorphologyCollisionDecision",
    "MorphologyCollisionPolicy",
    "MorphologyLayout",
    "MorphologyParameterization",
    "MorphologyRetractionResult",
    "MorphologyRuntimeBridge",
    "MorphologyRuntimeContext",
    "PARAMETER_ARTIFACT_SCHEMA",
    "PARAMETER_ARTIFACT_SCHEMA_VERSION",
    "RedMaxParameterSlice",
    "UNIFIED_PARAMETERIZATION_ID",
    "UnifiedConnectedHeadMorphology",
    "build_connected_direct_planar_hex_design_bundle",
    "build_connected_head_topology",
    "build_design_bundle",
    "build_function_group_contact_set",
    "cage_bounds_for_bundle",
    "check_design_collision",
    "check_design_params_collision",
    "connected_direct_planar_hex_bounds_for_bundle",
    "constraint_jacobian",
    "hexahedron_constraints",
    "legacy_morphology_layout_manifest",
    "metadata_for_runner",
    "normalize_face_mask",
    "normalize_parameters_for_runner",
    "parse_parameter_artifact_metadata",
    "point_squared_distance",
    "project_tangent",
    "read_parameter_artifact_metadata",
    "retract",
    "stable_layout_fingerprint",
    "validate_hexahedron",
    "validate_parameter_vector",
    "vertices_from_q",
]
