"""Physical-function terminal identity for rigid generated tools.

Physical-function equivalence
-----------------------------
Two completed designs are equivalent exactly when their generated rigid-body
multisets occupy the same world-space geometry and the same physical terminal
bodies carry the same ordered task-function semantics.

The key deliberately ignores construction sequence and topology, parent/child
relationships, dock and face identities, facing tokens, asset IDs, local body
frames, and XML names/order. The selected root rotation mode is retained
because it changes actuator dynamics. Multiplicity is retained. Different
rigid-body decompositions therefore remain distinct.

The current grammar emits rigid cuboids with uniform physical parameters. A
body is identified by its eight quantized world-space corners, not by an AABB
or rotation matrix. If asset-specific mass/material parameters become part of
the search space, they must be added to ``physical_parameters_key`` below.
"""

from __future__ import annotations

import hashlib
import struct
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from numbers import Real
from typing import Any, Iterable, Sequence, Tuple

from ..io_assets import AssetSpec
from ..state import (
    LinkContext,
    RealizedBody,
    SearchState,
    _physical_open_port_key,
)


PHYSICAL_SIGNATURE_EPS = 1e-8
PACKED_SIGNATURE_VERSION = "canonical-tuple-v1"

QuantizedPoint = Tuple[int, int, int]
PhysicalBodyKey = Tuple[Any, ...]
PhysicalFunctionSignature = Tuple[Any, ...]
PhysicalFunctionDescriptor = Tuple[Any, ...]
PhysicalPartialStateSignature = Tuple[Any, ...]


@dataclass(frozen=True)
class PhysicalPartialSignatureContext:
    """Immutable search-wide inputs reused by partial-state canonicalization."""

    asset_by_id: dict[str, AssetSpec]
    forbidden_key: Tuple[Any, ...]


def prepare_physical_partial_signature_context(
    assets: Sequence[AssetSpec],
    forbidden_boxes: Sequence[Any],
    *,
    eps: float = PHYSICAL_SIGNATURE_EPS,
) -> PhysicalPartialSignatureContext:
    """Precompute canonical inputs that do not change across transitions."""

    return PhysicalPartialSignatureContext(
        asset_by_id={asset.asset_id: asset for asset in assets},
        forbidden_key=tuple(
            sorted(_quantized_aabb(box, eps) for box in forbidden_boxes)
        ),
    )


def _encode_unsigned_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("unsigned varint cannot encode a negative value")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _encode_canonical_value(value: Any, output: bytearray) -> None:
    """Encode canonical tuples without pickle or Python object identity.

    The format is deliberately small and self-delimiting. It preserves exact
    Python value equality for the primitive types used by physical signatures;
    hashes of this byte stream are diagnostic/indexing aids only.
    """

    if value is None:
        output.append(0)
    elif value is False:
        output.append(1)
    elif value is True:
        output.append(2)
    elif isinstance(value, int):
        output.append(3)
        zigzag = 2 * value if value >= 0 else -2 * value - 1
        output.extend(_encode_unsigned_varint(zigzag))
    elif isinstance(value, str):
        output.append(4)
        payload = value.encode("utf-8")
        output.extend(_encode_unsigned_varint(len(payload)))
        output.extend(payload)
    elif isinstance(value, tuple):
        output.extend(
            _pack_canonical_tuple_cached(value)
            if _canonical_tuple_cache_safe(value)
            else _pack_canonical_tuple_uncached(value)
        )
    elif isinstance(value, float):
        output.append(6)
        output.extend(struct.pack("<d", value))
    elif isinstance(value, bytes):
        output.append(7)
        output.extend(_encode_unsigned_varint(len(value)))
        output.extend(value)
    else:
        raise TypeError(
            "unsupported canonical signature value type: {}".format(
                type(value).__name__
            )
        )


@lru_cache(maxsize=32768)
def _pack_canonical_tuple_cached(value: Tuple[Any, ...]) -> bytes:
    """Memoize exact encodings of immutable signature subtrees."""

    return _pack_canonical_tuple_uncached(value)


def _pack_canonical_tuple_uncached(value: Tuple[Any, ...]) -> bytes:
    """Encode a tuple while preserving Python type distinctions exactly."""

    output = bytearray((5,))
    output.extend(_encode_unsigned_varint(len(value)))
    for item in value:
        _encode_canonical_value(item, output)
    return bytes(output)


def _canonical_tuple_cache_safe(value: Tuple[Any, ...]) -> bool:
    """Exclude values whose Python equality conflates encoded types."""

    for item in value:
        if isinstance(item, tuple):
            if not _canonical_tuple_cache_safe(item):
                return False
        elif type(item) not in {int, str, bytes, type(None)}:
            return False
    return True


def pack_physical_signature(signature: Tuple[Any, ...]) -> bytes:
    """Return a deterministic exact byte representation of a physical key."""

    output = bytearray()
    _encode_canonical_value(signature, output)
    return bytes(output)


def packed_signature_digest(packed_signature: bytes) -> bytes:
    """Return a sort/index digest; complete bytes remain the equality key."""

    return hashlib.sha256(packed_signature).digest()


def _matvec(
    rotation: Tuple[float, ...],
    point: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    if len(rotation) != 9:
        raise ValueError("cuboid rotation must contain exactly nine values")
    return (
        rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2],
        rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2],
        rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2],
    )


def cuboid_world_vertices(
    center: Sequence[float],
    rotation: Sequence[float],
    half_extents: Sequence[float],
) -> Tuple[Tuple[float, float, float], ...]:
    """Return the eight world-space vertices of one oriented cuboid."""

    center_t = tuple(float(value) for value in center)
    rotation_t = tuple(float(value) for value in rotation)
    half_t = tuple(float(value) for value in half_extents)
    if len(center_t) != 3 or len(half_t) != 3:
        raise ValueError("cuboid center and half_extents must contain three values")
    if any(value <= 0.0 or not math.isfinite(value) for value in half_t):
        raise ValueError("cuboid half_extents must be positive and finite")
    if not all(math.isfinite(value) for value in (*center_t, *rotation_t)):
        raise ValueError("cuboid pose must be finite")

    vertices = []
    for signs in product((-1.0, 1.0), repeat=3):
        local = tuple(signs[index] * half_t[index] for index in range(3))
        offset = _matvec(rotation_t, local)
        vertices.append(
            tuple(center_t[index] + offset[index] for index in range(3))
        )
    return tuple(vertices)


def _quantize_point(point: Iterable[float], eps: float) -> QuantizedPoint:
    return tuple(int(round(float(value) / eps)) for value in point)  # type: ignore[return-value]


@lru_cache(maxsize=8192)
def _canonical_cuboid_cached(
    center: Tuple[float, ...],
    rotation: Tuple[float, ...],
    half_extents: Tuple[float, ...],
    eps: float,
) -> PhysicalBodyKey:
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError("physical signature epsilon must be positive and finite")
    geometry_key = tuple(
        sorted(
            _quantize_point(vertex, eps)
            for vertex in cuboid_world_vertices(center, rotation, half_extents)
        )
    )
    physical_parameters_key: Tuple[Any, ...] = ()
    return ("rigid-cuboid-v1", geometry_key, physical_parameters_key)


def canonical_cuboid(
    center: Sequence[float],
    rotation: Sequence[float],
    half_extents: Sequence[float],
    eps: float = PHYSICAL_SIGNATURE_EPS,
) -> PhysicalBodyKey:
    """Return a symmetry-free, exactly hashable physical cuboid key."""

    return _canonical_cuboid_cached(
        tuple(float(value) for value in center),
        tuple(float(value) for value in rotation),
        tuple(float(value) for value in half_extents),
        float(eps),
    )


def _quantized_aabb(box: Any, eps: float) -> Tuple[Any, ...]:
    minimum, maximum = box
    return (
        _quantize_point(minimum, eps),
        _quantize_point(maximum, eps),
    )


def _context_body_key(context: LinkContext, eps: float) -> PhysicalBodyKey:
    return canonical_cuboid(
        context.center,
        context.rotation,
        context.half_extents,
        eps,
    )


def physical_partial_state_signature(
    state: SearchState,
    assets: Sequence[AssetSpec],
    *,
    eps: float = PHYSICAL_SIGNATURE_EPS,
    signature_context: PhysicalPartialSignatureContext | None = None,
) -> PhysicalPartialStateSignature:
    """Return a continuation-safe physical key for an incomplete state.

    The key removes construction sequence, topology, link indices, local face
    IDs, and dock IDs. It retains the ordered DFS cursor stack because closing
    that stack defines future control flow. Each cursor is represented by its
    world geometry and physical open attachment affordances, not by its asset
    or local frame. Function roots and memberships are mapped to stable
    semantic ordinals rather than construction-time link IDs.

    Completed states use :func:`physical_function_signature` directly so one
    physical-function terminal becomes exactly one DAG node.
    """

    if state.is_complete:
        return (
            "physical-partial-state-v1-terminal",
            physical_function_signature(state, assets, eps=eps),
        )
    if not state.stack:
        raise ValueError("incomplete physical partial state has an empty stack")

    asset_by_id = (
        signature_context.asset_by_id
        if signature_context is not None
        else {asset.asset_id: asset for asset in assets}
    )
    missing = sorted(
        {
            context.asset_id
            for context in state.stack
            if context.asset_id is not None and context.asset_id not in asset_by_id
        }
        | {
            body.asset_id
            for body in state.realized_bodies
            if body.asset_id not in asset_by_id
        }
    )
    if missing:
        raise ValueError(
            "partial state references unknown assets: " + ", ".join(missing)
        )

    body_key_by_index = {
        body.link_index: canonical_cuboid(
            body.center,
            body.rotation,
            body.half_extents,
            eps,
        )
        for body in state.realized_bodies
    }
    # The fixed root is not part of realized_bodies but is physically relevant
    # to collision checks and future attachment opportunities.
    for context in state.stack:
        body_key_by_index.setdefault(
            context.link_index,
            _context_body_key(context, eps),
        )

    group_ordinal = {
        int(root_index): ordinal
        for ordinal, root_index in enumerate(state.function_group_roots)
    }

    body_records = []
    for body in state.realized_bodies:
        ordinal = (
            -1
            if body.function_group_root is None
            else group_ordinal[int(body.function_group_root)]
        )
        body_records.append(
            (
                body_key_by_index[body.link_index],
                int(ordinal),
                bool(state.child_counts.get(body.link_index, 0) == 0),
            )
        )

    stack_records = []
    for context in state.stack:
        asset = (
            None
            if context.asset_id is None
            else asset_by_id[context.asset_id]
        )
        active_ordinal = (
            -1
            if context.active_function_group_root is None
            else group_ordinal[int(context.active_function_group_root)]
        )
        root_ordinal = group_ordinal.get(int(context.link_index), -1)
        stack_records.append(
            (
                body_key_by_index[context.link_index],
                int(context.depth),
                int(active_ordinal),
                int(root_ordinal),
                bool(state.child_counts.get(context.link_index, 0) == 0),
                _physical_open_port_key(
                    context,
                    asset,
                    context.occupied_face_slots,
                ),
            )
        )

    function_records = []
    for ordinal, root_index in enumerate(state.function_group_roots):
        depths = state.function_group_leaf_depths.get(int(root_index))
        function_records.append(
            (
                _semantic_label(state, ordinal),
                body_key_by_index[int(root_index)],
                None if depths is None else (int(depths[0]), int(depths[1])),
            )
        )

    return (
        "physical-partial-state-v1",
        ("root-rotation-mode", state.root_rotation_mode),
        tuple(state.root_rotation_options),
        int(state.link_count),
        tuple(sorted(body_records)),
        tuple(stack_records),
        tuple(function_records),
        tuple(state.function_semantic_labels),
        int(state.uncovered_terminal_leaves),
        (
            signature_context.forbidden_key
            if signature_context is not None
            else tuple(
                sorted(_quantized_aabb(box, eps) for box in state.forbidden_boxes)
            )
        ),
    )


def _float_cuboid_descriptor(body: RealizedBody) -> Tuple[Any, ...]:
    return (
        "rigid-cuboid-v1",
        tuple(
            sorted(
                tuple(float(value) for value in vertex)
                for vertex in cuboid_world_vertices(
                    body.center,
                    body.rotation,
                    body.half_extents,
                )
            )
        ),
        (),
    )


def _semantic_label(state: SearchState, ordinal: int) -> str:
    if ordinal < len(state.function_semantic_labels):
        return state.function_semantic_labels[ordinal]
    return f"function:{ordinal}"


def _physical_components(
    state: SearchState,
    assets: Sequence[AssetSpec] | None,
    eps: float,
    *,
    quantized: bool,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[Any, ...]]:
    if not state.is_complete:
        raise ValueError("physical-function signature requires a completed state")

    if assets is not None:
        known_assets = {asset.asset_id for asset in assets}
        missing = sorted(
            {body.asset_id for body in state.realized_bodies} - known_assets
        )
        if missing:
            raise ValueError(
                "realized body references unknown assets: " + ", ".join(missing)
            )

    body_key_by_index = {}
    for body in state.realized_bodies:
        body_key_by_index[body.link_index] = (
            canonical_cuboid(
                body.center,
                body.rotation,
                body.half_extents,
                eps,
            )
            if quantized
            else _float_cuboid_descriptor(body)
        )

    scene_key = tuple(sorted(body_key_by_index.values()))
    functions = []
    for ordinal, group_root in enumerate(state.function_group_roots):
        leaves = []
        for body in state.realized_bodies:
            if body.function_group_root != group_root:
                continue
            if state.child_counts.get(body.link_index, 0) != 0:
                continue
            leaves.append(body_key_by_index[body.link_index])
        functions.append(
            (
                _semantic_label(state, ordinal),
                tuple(sorted(leaves)),
            )
        )
    mechanism_key = (
        "root-rotation-mode",
        state.root_rotation_mode,
    )
    return mechanism_key, scene_key, tuple(functions)


def physical_function_signature(
    state: SearchState,
    assets: Sequence[AssetSpec] | None = None,
    *,
    eps: float = PHYSICAL_SIGNATURE_EPS,
) -> PhysicalFunctionSignature:
    """Return the versioned physical rigid-scene plus function-semantic key."""

    mechanism_key, scene_key, function_key = _physical_components(
        state,
        assets,
        eps,
        quantized=True,
    )
    return (
        "physical-function-v2",
        mechanism_key,
        scene_key,
        function_key,
    )


def physical_function_descriptor(
    state: SearchState,
    assets: Sequence[AssetSpec] | None = None,
) -> PhysicalFunctionDescriptor:
    """Return an unquantized descriptor for debug collision verification."""

    mechanism_key, scene_key, function_key = _physical_components(
        state,
        assets,
        PHYSICAL_SIGNATURE_EPS,
        quantized=False,
    )
    return (
        "physical-function-descriptor-v2",
        mechanism_key,
        scene_key,
        function_key,
    )


def assert_physical_descriptors_close(
    left: PhysicalFunctionDescriptor,
    right: PhysicalFunctionDescriptor,
    *,
    atol: float,
) -> None:
    """Raise if equal quantized keys hide materially different geometry."""

    if atol < 0.0 or not math.isfinite(atol):
        raise ValueError("descriptor comparison tolerance must be finite and non-negative")

    def compare(a: Any, b: Any, path: str) -> None:
        if isinstance(a, Real) and isinstance(b, Real):
            if abs(float(a) - float(b)) > atol:
                raise AssertionError(
                    f"physical signature collision at {path}: {a} != {b}"
                )
            return
        if type(a) is not type(b) or not isinstance(a, tuple):
            if a != b:
                raise AssertionError(
                    f"physical signature collision at {path}: {a!r} != {b!r}"
                )
            return
        if len(a) != len(b):
            raise AssertionError(
                f"physical signature collision at {path}: length {len(a)} != {len(b)}"
            )
        for index, (left_item, right_item) in enumerate(zip(a, b)):
            compare(left_item, right_item, f"{path}[{index}]")

    compare(left, right, "descriptor")


def physical_signature_digest(signature: PhysicalFunctionSignature) -> str:
    """Return a compact stable diagnostic identifier for a physical key."""

    return hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()
