"""State transitions and grammar validity for constructive skeleton search."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import random
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from bilevel.upper.bass.connection_geometry import (
    EDGE_CORNER_CONNECTION,
    compatible_orientations,
    connection_family,
    edge_port_frame,
    resolve_edge_corner_pose,
)

from .actions import (
    Action,
    AddLink,
    End,
    PreparedAttachment,
    ROOT_ROTATION_MODES,
    SelectRootRotation,
)
from .io_assets import AssetSpec


FACE_COUNT = 6
ROOT_HALF_EXTENTS = (0.5, 0.5, 0.5)
IDENTITY_ROTATION = (
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
)
_ASSET_OUT_PORT_KEY_CACHE: Dict[int, Tuple[AssetSpec, Tuple]] = {}
_ASSET_OUT_PORT_KEY_CACHE_LOCK = threading.Lock()


def _face_axis(face: int) -> int:
    """Return axis index (x=0, y=1, z=2) corresponding to a face index."""
    return {0: 2, 1: 0, 2: 1, 3: 0, 4: 2, 5: 1}[face]


def _face_sign(face: int) -> float:
    """Return outward sign for a face index on its axis."""
    return {0: 1.0, 1: 1.0, 2: 1.0, 3: -1.0, 4: -1.0, 5: -1.0}[face]


def _tangent_axes(face: int) -> Tuple[int, int]:
    """Return tangent axis indices spanning a face plane."""
    axis = _face_axis(face)
    if axis == 0:
        return (1, 2)
    if axis == 1:
        return (0, 2)
    return (0, 1)


def _basis_from_face(face: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """Return local orthonormal face basis (u, v, n)."""
    n = [0.0, 0.0, 0.0]
    n[_face_axis(face)] = _face_sign(face)
    t0, t1 = _tangent_axes(face)
    u = [0.0, 0.0, 0.0]
    v = [0.0, 0.0, 0.0]
    u[t0] = 1.0
    v[t1] = 1.0
    # Keep the same right-handed convention as bilevel.upper.bass.assembly.
    cross = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    if sum(cross[i] * n[i] for i in range(3)) < 0.0:
        v = [-value for value in v]
    return (tuple(u), tuple(v), tuple(n))


def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Tuple[float, float, float]) -> float:
    return max(_dot(a, a) ** 0.5, 1e-12)


def _normalize(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    length = _norm(a)
    return (a[0] / length, a[1] / length, a[2] / length)


def _rodrigues(
    v: Tuple[float, float, float],
    axis: Tuple[float, float, float],
    angle: float,
) -> Tuple[float, float, float]:
    """Rotate vector v around axis by angle."""
    import math

    axis = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    axv = _cross(axis, v)
    adv = _dot(axis, v)
    return (
        v[0] * c + axv[0] * s + axis[0] * adv * (1.0 - c),
        v[1] * c + axv[1] * s + axis[1] * adv * (1.0 - c),
        v[2] * c + axv[2] * s + axis[2] * adv * (1.0 - c),
    )


def _matvec_from_columns(
    c0: Tuple[float, float, float],
    c1: Tuple[float, float, float],
    c2: Tuple[float, float, float],
    v: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    return (
        c0[0] * v[0] + c1[0] * v[1] + c2[0] * v[2],
        c0[1] * v[0] + c1[1] * v[1] + c2[1] * v[2],
        c0[2] * v[0] + c1[2] * v[1] + c2[2] * v[2],
    )


@dataclass(frozen=True)
class LinkContext:
    """Per-link cursor context used for DFS-style construction.

    Attributes:
        asset_id: Asset id for this link. Legacy synthetic root is None.
        center: Approximate world-space center of this link cuboid.
        half_extents: Link-local cuboid half extents.
        rotation: Row-major link-local-to-root rotation matrix.
        link_index: Index in SearchState.placed_boxes for overlap bookkeeping.
        depth: Root-relative construction depth.
        blocked_face: Face whose complete out-dock set is reserved. Generated
            links rely on explicit catalog out-docks and block the ingress
            face only when that face has no outgoing ports.
        start_function_group: Whether this link roots one functional group.
        active_function_group_root: Nearest active function-group root index.
        occupied_face_slots: Face-indexed occupied parent dock ids.
    """

    asset_id: Optional[str]
    center: Tuple[float, float, float]
    half_extents: Tuple[float, float, float]
    rotation: Tuple[float, ...]
    link_index: int
    depth: int
    blocked_face: int | None
    start_function_group: bool = False
    active_function_group_root: int | None = None
    occupied_face_slots: Dict[int, frozenset[int]] = field(
        default_factory=lambda: {face: frozenset() for face in range(FACE_COUNT)}
    )


@dataclass(frozen=True)
class RealizedBody:
    """Exact generated rigid-body realization retained after DFS closure.

    ``link_index`` is an internal bookkeeping identity only. Physical terminal
    canonicalization deliberately excludes it, ``asset_id``, and construction
    topology from the final body key.
    """

    link_index: int
    asset_id: str
    center: Tuple[float, float, float]
    half_extents: Tuple[float, float, float]
    rotation: Tuple[float, ...]
    function_group_root: int | None


@dataclass(frozen=True)
class SearchState:
    """Immutable search state for one partial sequence.

    Attributes:
        sequence: Grammar actions executed so far.
        stack: DFS construction stack; top is current cursor link.
        placed_boxes: World-space AABB list for constructed links. When a
            root_box is supplied, the root/attachment parent is stored at index 0.
        forbidden_boxes: World-space AABB list for fixed external occupancy.
            These boxes are never pushed onto the construction stack; they are
            used to reject candidate links that collide with known geometry such
            as a finger template near the tool attachment site.
        link_count: Number of links currently in skeleton, including root.
        child_counts: Number of constructed children for each placed link index.
        root_link_index: Placed-box index for the fixed attachment root. This
            link is excluded from end-effector/function counting.
        function_group_roots: Placed-box indices that root functional groups.
        function_group_leaf_depths: Min/max terminal-leaf depths observed for
            each function-group root after DFS closure of leaves.
        realized_bodies: Exact generated rigid-body poses. The fixed Handle
            root is intentionally excluded because it is task-constant and its
            world pose is owned by the task scene rather than the grammar.
        function_semantic_labels: Ordered task semantics assigned to function
            groups. Group creation ordinal indexes this tuple.
        uncovered_terminal_leaves: Closed terminal generated leaves that are
            not contained in any selected function group.
        root_rotation_options: Allowed one-DOF root rotation modes. An empty
            tuple preserves the legacy grammar with no root decision.
        root_rotation_mode: Selected mode, or None before the required root
            decision.
        is_complete: True when End is applied at root.
    """

    sequence: List[Action]
    stack: List[LinkContext]
    placed_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    forbidden_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    link_count: int
    child_counts: Dict[int, int]
    root_link_index: int | None
    function_group_roots: Tuple[int, ...]
    function_group_leaf_depths: Dict[int, Tuple[int, int]]
    realized_bodies: Tuple[RealizedBody, ...]
    function_semantic_labels: Tuple[str, ...]
    uncovered_terminal_leaves: int
    root_rotation_options: Tuple[str, ...]
    root_rotation_mode: str | None
    is_complete: bool


def initial_state(
    forbidden_boxes: Optional[
        List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    ] = None,
    root_box: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
    root_asset: Optional[AssetSpec] = None,
    root_blocked_face: int | None = None,
    root_rotation_options: Tuple[str, ...] = (),
    function_semantic_labels: Tuple[str, ...] = (),
) -> SearchState:
    """Create initial state with a single root link cursor.

    Args:
        forbidden_boxes: Optional fixed AABBs in the same coordinate frame as
            the synthetic root. Candidate links may not overlap these boxes.
        root_box: Optional AABB for the synthetic root itself. When provided,
            the root cursor uses this box for face/dock placement and stores it
            in ``placed_boxes`` as the direct parent. This is useful when the
            synthetic root represents an existing attachment body such as a
            finger tip.
        root_asset: Optional asset metadata for the fixed tip/root link. When
            supplied, root dock availability comes from this asset instead of
            the legacy synthetic center-dock fallback.
        root_blocked_face: Optional parent-facing face on the fixed root that
            cannot be selected for child attachment.

    Returns:
        Initialized SearchState.
    """
    if root_blocked_face is not None and not (0 <= root_blocked_face < FACE_COUNT):
        raise ValueError(f"root_blocked_face must be in [0, 5], got {root_blocked_face}")
    normalized_rotation_options = tuple(
        str(value).strip().lower() for value in root_rotation_options
    )
    if len(set(normalized_rotation_options)) != len(normalized_rotation_options):
        raise ValueError("root_rotation_options must not contain duplicates")
    invalid_rotation_options = set(normalized_rotation_options) - set(ROOT_ROTATION_MODES)
    if invalid_rotation_options:
        raise ValueError(
            "root_rotation_options contains unsupported modes: "
            f"{sorted(invalid_rotation_options)}"
        )

    if root_box is None and root_asset is None:
        root_center = (0.0, 0.0, 0.0)
        root_half = ROOT_HALF_EXTENTS
        root_link_index = -1
        placed_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
        child_counts: Dict[int, int] = {}
        root_index: int | None = None
    else:
        if root_box is None:
            root_center = (0.0, 0.0, 0.0)
            root_half = root_asset.half_extents if root_asset is not None else ROOT_HALF_EXTENTS
            root_box = _aabb_from_center_half(root_center, root_half)
        mn, mx = root_box
        root_center = (
            0.5 * (float(mn[0]) + float(mx[0])),
            0.5 * (float(mn[1]) + float(mx[1])),
            0.5 * (float(mn[2]) + float(mx[2])),
        )
        root_half = (
            max(0.5 * (float(mx[0]) - float(mn[0])), 1e-9),
            max(0.5 * (float(mx[1]) - float(mn[1])), 1e-9),
            max(0.5 * (float(mx[2]) - float(mn[2])), 1e-9),
        )
        root_link_index = 0
        placed_boxes = [root_box]
        child_counts = {root_link_index: 0}
        root_index = root_link_index

    root = LinkContext(
        asset_id=root_asset.asset_id if root_asset is not None else None,
        center=root_center,
        half_extents=root_half,
        rotation=IDENTITY_ROTATION,
        link_index=root_link_index,
        depth=0,
        blocked_face=root_blocked_face,
    )
    return SearchState(
        sequence=[],
        stack=[root],
        placed_boxes=placed_boxes,
        forbidden_boxes=list(forbidden_boxes or []),
        link_count=1,
        child_counts=child_counts,
        root_link_index=root_index,
        function_group_roots=(),
        function_group_leaf_depths={},
        realized_bodies=(),
        function_semantic_labels=tuple(
            str(label) for label in function_semantic_labels
        ),
        uncovered_terminal_leaves=0,
        root_rotation_options=normalized_rotation_options,
        root_rotation_mode=None,
        is_complete=False,
    )


def _parent_face_docks(
    context: LinkContext,
    asset_by_id: Dict[str, AssetSpec],
    face: int,
) -> List[dict]:
    """Return available docks on a parent face for a link context."""
    if context.asset_id is None:
        return [{"id": 0, "barycentric": [0.25, 0.25, 0.25, 0.25]}]
    parent_asset = asset_by_id.get(context.asset_id)
    if parent_asset is None:
        return []
    return parent_asset.out_docks_for_face(face)


def _child_face_docks(child_asset: AssetSpec, face: int) -> List[dict]:
    """Return allowed child-side docks on a child face."""
    return child_asset.in_docks_for_face(face)


def _child_dock_by_id(child_asset: AssetSpec, face: int, dock_id: int) -> Optional[dict]:
    """Find one child-side dock configuration by id on a child face."""
    for dock in _child_face_docks(child_asset, face):
        if int(dock.get("id", -1)) == dock_id:
            return dock
    return None


def _child_dock_facing_slots(child_asset: AssetSpec, face: int) -> List[Tuple[int, int]]:
    """Return unique child-side (dock_id, facing) slots for a child face."""
    slots: set[Tuple[int, int]] = set()
    for dock in _child_face_docks(child_asset, face):
        dock_id = int(dock.get("id", 0))
        for facing in dock.get("facing_options", [0, 1, 2, 3]):
            slots.add((dock_id, int(facing)))
    return sorted(slots)


def _pair_facing_slots(
    parent_dock: dict,
    child_asset: AssetSpec,
    face: int,
) -> List[Tuple[int, int]]:
    """Return pair-dependent child-dock orientation slots."""

    slots: set[Tuple[int, int]] = set()
    for child_dock in _child_face_docks(child_asset, face):
        child_dock_id = int(child_dock.get("id", 0))
        for facing in compatible_orientations(parent_dock, child_dock):
            slots.add((child_dock_id, int(facing)))
    return sorted(slots)


def _dock_by_id(
    context: LinkContext,
    asset_by_id: Dict[str, AssetSpec],
    face: int,
    dock_id: int,
) -> Optional[dict]:
    """Find one dock configuration by id on a parent face."""
    for dock in _parent_face_docks(context, asset_by_id, face):
        if int(dock.get("id", -1)) == dock_id:
            return dock
    return None


def _face_vertices(context: LinkContext, face: int) -> List[Tuple[float, float, float]]:
    """Return four world-space vertices of a link-local face."""
    result = []
    for local in _local_face_vertices(context.half_extents, face):
        offset = _matvec(context.rotation, local)
        result.append(
            (
                context.center[0] + offset[0],
                context.center[1] + offset[1],
                context.center[2] + offset[2],
            )
        )
    return result


def _local_face_vertices(
    half_extents: Tuple[float, float, float],
    face: int,
) -> List[Tuple[float, float, float]]:
    """Return four local-space vertices of a cuboid face."""
    axis = _face_axis(face)
    sign = _face_sign(face)
    t0, t1 = _tangent_axes(face)
    base = [0.0, 0.0, 0.0]
    base[axis] = sign * half_extents[axis]

    def point(s0: float, s1: float) -> Tuple[float, float, float]:
        p = list(base)
        p[t0] += s0 * half_extents[t0]
        p[t1] += s1 * half_extents[t1]
        return (p[0], p[1], p[2])

    return [
        point(-1.0, -1.0),
        point(1.0, -1.0),
        point(1.0, 1.0),
        point(-1.0, 1.0),
    ]


def _matvec(
    matrix: Tuple[float, ...],
    vector: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Multiply a row-major 3x3 matrix by a 3-vector."""
    return (
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    )


def _matmul(
    left: Tuple[float, ...],
    right: Tuple[float, ...],
) -> Tuple[float, ...]:
    """Multiply two row-major 3x3 matrices."""
    return tuple(
        sum(left[3 * row + k] * right[3 * k + col] for k in range(3))
        for row in range(3)
        for col in range(3)
    )


def _basis_from_context_face(
    context: LinkContext,
    face: int,
) -> Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]:
    """Return one link-local face basis transformed into the root frame."""
    u, v, n = _basis_from_face(face)
    return (
        _matvec(context.rotation, u),
        _matvec(context.rotation, v),
        _matvec(context.rotation, n),
    )


def _barycentric_anchor(
    context: LinkContext,
    face: int,
    dock: dict,
) -> Tuple[float, float, float]:
    """Compute world anchor point from 4-value barycentric dock weights."""
    vertices = _face_vertices(context, face)
    bary = dock.get("barycentric", [0.25, 0.25, 0.25, 0.25])
    if not isinstance(bary, list) or len(bary) != 4:
        raise ValueError(f"Dock barycentric must be length-4, got {bary}")
    weights = [float(v) for v in bary]
    total = sum(weights)
    if total <= 0.0:
        raise ValueError(f"Dock barycentric must have positive sum, got {weights}")
    weights = [v / total for v in weights]
    return (
        sum(weights[i] * vertices[i][0] for i in range(4)),
        sum(weights[i] * vertices[i][1] for i in range(4)),
        sum(weights[i] * vertices[i][2] for i in range(4)),
    )


def _local_barycentric_anchor(
    half_extents: Tuple[float, float, float],
    face: int,
    dock: dict,
) -> Tuple[float, float, float]:
    """Compute local cuboid anchor point from 4-value barycentric dock weights."""
    vertices = _local_face_vertices(half_extents, face)
    bary = dock.get("barycentric", [0.25, 0.25, 0.25, 0.25])
    weights = [float(v) for v in bary]
    total = sum(weights)
    if total <= 0.0:
        raise ValueError(f"Dock barycentric must have positive sum, got {weights}")
    weights = [v / total for v in weights]
    return (
        sum(weights[i] * vertices[i][0] for i in range(4)),
        sum(weights[i] * vertices[i][1] for i in range(4)),
        sum(weights[i] * vertices[i][2] for i in range(4)),
    )


def _rotation_signature_from_attachment(
    parent_context: LinkContext,
    parent_face: int,
    child_face: int,
    facing: int,
) -> Tuple[float, ...]:
    """Return the child local-to-root rotation matrix for an attachment."""
    u_p, _, n_p = _basis_from_context_face(parent_context, parent_face)
    u_q, v_q, n_q = _basis_from_face(child_face)
    n_target = (-n_p[0], -n_p[1], -n_p[2])
    import math

    u_target = _rodrigues(u_p, n_target, (int(facing) % 4) * (math.pi / 2.0))
    dot_un = _dot(u_target, n_target)
    u_target = _normalize(
        (
            u_target[0] - n_target[0] * dot_un,
            u_target[1] - n_target[1] * dot_un,
            u_target[2] - n_target[2] * dot_un,
        )
    )
    v_target = _normalize(_cross(n_target, u_target))
    source_basis = (u_q, v_q, n_q)
    target_basis = (u_target, v_target, n_target)
    # R = [u_target v_target n_target] @ [u_q v_q n_q].T.
    return tuple(
        sum(target_basis[k][row] * source_basis[k][col] for k in range(3))
        for row in range(3)
        for col in range(3)
    )


def _child_center_from_attachment(
    parent_context: LinkContext,
    parent_face: int,
    parent_dock: dict,
    child_half_extents: Tuple[float, float, float],
    child_face: int,
    child_dock: dict,
    facing: int,
) -> Tuple[float, float, float]:
    """Compute approximate child center from face-to-face docking.

    This assumes connected faces remain parallel and uses child-face axis extent
    along the parent-face normal direction.
    """
    anchor = _barycentric_anchor(parent_context, parent_face, parent_dock)
    child_anchor = _local_barycentric_anchor(child_half_extents, child_face, child_dock)
    u_p, _, n_p = _basis_from_context_face(parent_context, parent_face)
    u_q, v_q, n_q = _basis_from_face(child_face)
    n_target = (-n_p[0], -n_p[1], -n_p[2])
    import math

    u_target = _rodrigues(u_p, n_target, (int(facing) % 4) * (math.pi / 2.0))
    dot_un = _dot(u_target, n_target)
    u_target = _normalize(
        (
            u_target[0] - n_target[0] * dot_un,
            u_target[1] - n_target[1] * dot_un,
            u_target[2] - n_target[2] * dot_un,
        )
    )
    v_target = _normalize(_cross(n_target, u_target))

    # R = [u_target v_target n_target] @ [u_q v_q n_q].T
    child_coords = (_dot(u_q, child_anchor), _dot(v_q, child_anchor), _dot(n_q, child_anchor))
    rotated_child_anchor = _matvec_from_columns(u_target, v_target, n_target, child_coords)
    return (
        anchor[0] - rotated_child_anchor[0],
        anchor[1] - rotated_child_anchor[1],
        anchor[2] - rotated_child_anchor[2],
    )


def _dock_pose_key(dock: dict) -> Tuple:
    """Return the immutable subset of a dock that affects rigid placement."""
    barycentric = dock.get("barycentric", [0.25, 0.25, 0.25, 0.25])
    return (
        tuple(float(value) for value in barycentric),
        bool(dock.get("edge", False)),
        str(dock.get("edge_profile", "")),
        dock.get("edge_tangent_axis"),
        dock.get("panel_normal_axis"),
    )


def _dock_from_pose_key(key: Tuple) -> dict:
    barycentric, edge, edge_profile, edge_tangent_axis, panel_normal_axis = key
    dock = {"barycentric": list(barycentric)}
    if edge:
        dock["edge"] = True
        dock["edge_profile"] = edge_profile
        dock["edge_tangent_axis"] = int(edge_tangent_axis)
        dock["panel_normal_axis"] = int(panel_normal_axis)
    return dock


def _connection_pose_uncached(
    parent_context: LinkContext,
    parent_face: int,
    parent_dock: dict,
    child_asset: AssetSpec,
    child_face: int,
    child_dock: dict,
    facing: int,
) -> Tuple[Tuple[float, float, float], Tuple[float, ...]]:
    """Compute a connection pose without consulting the local-pose cache."""

    if connection_family(parent_dock, child_dock) != EDGE_CORNER_CONNECTION:
        return (
            _child_center_from_attachment(
                parent_context,
                parent_face=parent_face,
                parent_dock=parent_dock,
                child_half_extents=child_asset.half_extents,
                child_face=child_face,
                child_dock=child_dock,
                facing=facing,
            ),
            _rotation_signature_from_attachment(
                parent_context,
                parent_face,
                child_face,
                facing,
            ),
        )

    parent_anchor = _barycentric_anchor(parent_context, parent_face, parent_dock)
    rotation, target_anchor = resolve_edge_corner_pose(
        parent_rotation=np.asarray(parent_context.rotation, dtype=np.float64).reshape(3, 3),
        parent_anchor=parent_anchor,
        parent_half_extents=parent_context.half_extents,
        parent_face=parent_face,
        parent_port=parent_dock,
        child_half_extents=child_asset.half_extents,
        child_face=child_face,
        child_port=child_dock,
        orientation=facing,
    )
    child_anchor = np.asarray(
        _local_barycentric_anchor(child_asset.half_extents, child_face, child_dock),
        dtype=np.float64,
    )
    child_center = target_anchor - rotation @ child_anchor
    return (
        tuple(float(value) for value in child_center),
        tuple(float(value) for value in rotation.reshape(-1)),
    )


@lru_cache(maxsize=8192)
def _local_connection_pose(
    parent_half_extents: Tuple[float, float, float],
    parent_face: int,
    parent_dock_key: Tuple,
    child_half_extents: Tuple[float, float, float],
    child_face: int,
    child_dock_key: Tuple,
    facing: int,
) -> Tuple[Tuple[float, float, float], Tuple[float, ...]]:
    """Return a dock transform in the parent frame.

    Asset dimensions and dock metadata are immutable across a search. Caching
    this local transform removes repeated basis, normalization, and Rodrigues
    work; each world placement then needs only rigid composition.
    """

    local_parent = LinkContext(
        asset_id=None,
        center=(0.0, 0.0, 0.0),
        half_extents=parent_half_extents,
        rotation=IDENTITY_ROTATION,
        link_index=-1,
        depth=0,
        blocked_face=None,
    )
    child_asset = AssetSpec(
        asset_id="__cached_child__",
        half_extents=child_half_extents,
    )
    return _connection_pose_uncached(
        local_parent,
        parent_face,
        _dock_from_pose_key(parent_dock_key),
        child_asset,
        child_face,
        _dock_from_pose_key(child_dock_key),
        facing,
    )


def _connection_pose_from_attachment(
    parent_context: LinkContext,
    parent_face: int,
    parent_dock: dict,
    child_asset: AssetSpec,
    child_face: int,
    child_dock: dict,
    facing: int,
) -> Tuple[Tuple[float, float, float], Tuple[float, ...]]:
    """Compose a cached parent-local dock transform into the world frame."""

    local_center, local_rotation = _local_connection_pose(
        tuple(parent_context.half_extents),
        int(parent_face),
        _dock_pose_key(parent_dock),
        tuple(child_asset.half_extents),
        int(child_face),
        _dock_pose_key(child_dock),
        int(facing),
    )
    world_offset = _matvec(parent_context.rotation, local_center)
    world_center = (
        parent_context.center[0] + world_offset[0],
        parent_context.center[1] + world_offset[1],
        parent_context.center[2] + world_offset[2],
    )
    return world_center, _matmul(parent_context.rotation, local_rotation)


def _aabb_from_center_half(
    center: Tuple[float, float, float],
    half_extents: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Convert center+half extents to axis-aligned min/max box."""
    return (
        (center[0] - half_extents[0], center[1] - half_extents[1], center[2] - half_extents[2]),
        (center[0] + half_extents[0], center[1] + half_extents[1], center[2] + half_extents[2]),
    )


def _oriented_aabb_from_center_half(
    center: Tuple[float, float, float],
    half_extents: Tuple[float, float, float],
    rotation: Tuple[float, ...],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return the exact AABB of a rotated cuboid."""
    world_half = tuple(
        sum(abs(rotation[3 * row + col]) * half_extents[col] for col in range(3))
        for row in range(3)
    )
    return _aabb_from_center_half(center, world_half)


def _aabb_overlap(
    a: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    b: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    eps: float = 1e-9,
) -> bool:
    """Return whether two AABBs overlap with non-zero volume."""
    a_min, a_max = a
    b_min, b_max = b
    for i in range(3):
        if a_max[i] <= b_min[i] + eps or b_max[i] <= a_min[i] + eps:
            return False
    return True


def _collides_with_any_box(
    candidate_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    *,
    placed_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    forbidden_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    parent_link_index: int,
) -> bool:
    """Return True if a candidate AABB overlaps existing non-parent geometry."""
    for box_index, box in enumerate(placed_boxes):
        if box_index == parent_link_index:
            continue
        if _aabb_overlap(candidate_box, box):
            return True
    for box in forbidden_boxes:
        if _aabb_overlap(candidate_box, box):
            return True
    return False


def _passes_dock_collision_filter(
    context: LinkContext,
    asset_by_id: Dict[str, AssetSpec],
    p: int,
    d: int,
    placed_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    forbidden_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    child_asset: AssetSpec,
) -> bool:
    """Apply cuboid overlap hard filter for a candidate attachment.

    Candidate is illegal if its initialization cuboid overlaps any existing
    non-parent cuboid in the partial state.
    """
    candidate_dock = _dock_by_id(context, asset_by_id, p, d)
    if candidate_dock is None:
        return False

    child_half = child_asset.half_extents
    conservative_half = max(child_half)
    anchor = _barycentric_anchor(context, p, candidate_dock)
    axis = _face_axis(p)
    sign = _face_sign(p)
    center = [anchor[0], anchor[1], anchor[2]]
    center[axis] += sign * conservative_half
    candidate_box = _aabb_from_center_half(
        (center[0], center[1], center[2]),
        (conservative_half, conservative_half, conservative_half),
    )

    if _collides_with_any_box(
        candidate_box,
        placed_boxes=placed_boxes,
        forbidden_boxes=forbidden_boxes,
        parent_link_index=context.link_index,
    ):
        return False

    return True


def _candidate_child_box(
    parent_context: LinkContext,
    parent_face: int,
    parent_dock: dict,
    child_asset: AssetSpec,
    child_face: int,
    child_dock: dict,
    facing: int,
) -> Tuple[
    Tuple[float, float, float],
    Tuple[Tuple[float, float, float], Tuple[float, float, float]],
]:
    """Return the dock-correct child center and oriented-cuboid AABB."""
    child_center, child_rotation = _connection_pose_from_attachment(
        parent_context,
        parent_face=parent_face,
        parent_dock=parent_dock,
        child_asset=child_asset,
        child_face=child_face,
        child_dock=child_dock,
        facing=facing,
    )
    child_box = _oriented_aabb_from_center_half(
        child_center,
        child_asset.half_extents,
        child_rotation,
    )
    return child_center, child_box


def _attachment_transform_signature(
    parent_context: LinkContext,
    parent_face: int,
    parent_dock: dict,
    child_asset: AssetSpec,
    child_face: int,
    child_dock: dict,
    facing: int,
) -> Tuple:
    """Return a rounded rigid-pose signature for duplicate-action pruning.

    Two actions are pruned as duplicates only when they place the same asset at
    the same rigid pose. This is intentionally geometry-agnostic: symmetric
    assets benefit, while asymmetric assets keep distinct orientations.
    """
    child_center, child_rotation = _connection_pose_from_attachment(
        parent_context,
        parent_face=parent_face,
        parent_dock=parent_dock,
        child_asset=child_asset,
        child_face=child_face,
        child_dock=child_dock,
        facing=facing,
    )
    values = (
        *child_center,
        *child_rotation,
    )
    return (child_asset.asset_id, tuple(round(float(v), 6) for v in values))


def _attachment_state_token(
    state: SearchState,
    parent: LinkContext,
) -> Tuple:
    """Identify the exact geometry against which an attachment was validated."""

    return (
        int(parent.link_index),
        tuple(parent.center),
        tuple(parent.rotation),
        tuple(state.placed_boxes),
        tuple(state.forbidden_boxes),
        tuple(
            (int(face), tuple(sorted(int(slot) for slot in slots)))
            for face, slots in sorted(parent.occupied_face_slots.items())
        ),
    )


def _quantized_vector(values: Tuple[float, float, float], eps: float = 1e-8) -> Tuple[int, ...]:
    return tuple(int(round(float(value) / eps)) for value in values)


def _physical_cuboid_key(
    center: Tuple[float, float, float],
    rotation: Tuple[float, ...],
    half_extents: Tuple[float, float, float],
) -> Tuple:
    vertices = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                offset = _matvec(
                    rotation,
                    (
                        sx * half_extents[0],
                        sy * half_extents[1],
                        sz * half_extents[2],
                    ),
                )
                vertices.append(
                    _quantized_vector(
                        (
                            center[0] + offset[0],
                            center[1] + offset[1],
                            center[2] + offset[2],
                        )
                    )
                )
    return tuple(sorted(vertices))


def _physical_open_port_key(
    context: LinkContext,
    asset: AssetSpec | None,
    occupied_face_slots: Dict[int, frozenset[int]],
) -> Tuple:
    """Describe available docks without local face or dock identities."""

    descriptors = []
    for face in range(FACE_COUNT):
        if context.blocked_face == face:
            continue
        occupied = occupied_face_slots.get(face, frozenset())
        docks = (
            asset.out_docks_for_face(face)
            if asset is not None
            else [{"id": 0, "barycentric": [0.25, 0.25, 0.25, 0.25]}]
        )
        for dock in docks:
            dock_id = int(dock.get("id", 0))
            if dock_id in occupied:
                continue
            anchor = _barycentric_anchor(context, face, dock)
            _, _, normal = _basis_from_context_face(context, face)
            descriptor = [
                _quantized_vector(anchor),
                _quantized_vector(normal),
                str(connection_family(dock, dock)),
                str(dock.get("edge_profile", "")),
            ]
            if bool(dock.get("edge", False)):
                seam, outward, panel_normal = edge_port_frame(face, dock)
                descriptor.extend(
                    [
                        _quantized_vector(_matvec(context.rotation, tuple(seam))),
                        _quantized_vector(_matvec(context.rotation, tuple(outward))),
                        _quantized_vector(_matvec(context.rotation, tuple(panel_normal))),
                    ]
                )
            descriptors.append(tuple(descriptor))
    return tuple(sorted(descriptors))


def _physical_port_descriptor(
    context: LinkContext,
    face: int,
    dock: dict,
) -> Tuple:
    """Describe one attachment affordance in the context coordinate frame."""

    anchor = _barycentric_anchor(context, face, dock)
    _, _, normal = _basis_from_context_face(context, face)
    descriptor = [
        _quantized_vector(anchor),
        _quantized_vector(normal),
        str(connection_family(dock, dock)),
        str(dock.get("edge_profile", "")),
    ]
    if bool(dock.get("edge", False)):
        seam, outward, panel_normal = edge_port_frame(face, dock)
        descriptor.extend(
            [
                _quantized_vector(_matvec(context.rotation, tuple(seam))),
                _quantized_vector(_matvec(context.rotation, tuple(outward))),
                _quantized_vector(_matvec(context.rotation, tuple(panel_normal))),
            ]
        )
    return tuple(descriptor)


def _asset_out_ports_key(asset: AssetSpec) -> Tuple:
    """Return hashable outgoing-port metadata needed by successor quotienting."""

    cache_key = id(asset)
    cached = _ASSET_OUT_PORT_KEY_CACHE.get(cache_key)
    if cached is not None and cached[0] is asset:
        return cached[1]
    result = tuple(
        (
            int(face),
            int(dock.get("id", 0)),
            _dock_pose_key(dock),
        )
        for face in range(FACE_COUNT)
        for dock in asset.out_docks_for_face(face)
    )
    with _ASSET_OUT_PORT_KEY_CACHE_LOCK:
        cached = _ASSET_OUT_PORT_KEY_CACHE.get(cache_key)
        if cached is not None and cached[0] is asset:
            return cached[1]
        if len(_ASSET_OUT_PORT_KEY_CACHE) >= 256:
            _ASSET_OUT_PORT_KEY_CACHE.pop(next(iter(_ASSET_OUT_PORT_KEY_CACHE)))
        _ASSET_OUT_PORT_KEY_CACHE[cache_key] = (asset, result)
    return result


@lru_cache(maxsize=32768)
def _local_physical_attachment_successor_key(
    parent_half_extents: Tuple[float, float, float],
    parent_face: int,
    parent_dock_key: Tuple,
    child_half_extents: Tuple[float, float, float],
    child_face: int,
    child_dock_id: int,
    child_dock_key: Tuple,
    facing: int,
    child_out_ports_key: Tuple,
) -> Tuple:
    """Canonicalize a successor in the parent frame and cache the result.

    All candidates compared by one ``valid_actions`` call share the same
    parent pose and existing open-port multiset. Therefore equivalence only
    needs the realized child, the consumed parent port, and the child's future
    physical ports. A rigid world transform cannot change equality between
    these parent-local descriptors.
    """

    parent_dock = _dock_from_pose_key(parent_dock_key)
    child_dock = _dock_from_pose_key(child_dock_key)
    child_center, child_rotation = _local_connection_pose(
        parent_half_extents,
        parent_face,
        parent_dock_key,
        child_half_extents,
        child_face,
        child_dock_key,
        facing,
    )
    local_parent = LinkContext(
        asset_id=None,
        center=(0.0, 0.0, 0.0),
        half_extents=parent_half_extents,
        rotation=IDENTITY_ROTATION,
        link_index=-1,
        depth=0,
        blocked_face=None,
    )
    out_docks_per_face = {face: [] for face in range(FACE_COUNT)}
    child_occupied = {face: frozenset() for face in range(FACE_COUNT)}
    for face, dock_id, dock_key in child_out_ports_key:
        dock = _dock_from_pose_key(dock_key)
        dock["id"] = dock_id
        out_docks_per_face[face].append(dock)
        if face == child_face and dock_id == child_dock_id:
            child_occupied[face] = frozenset({dock_id})
    child_asset = AssetSpec(
        asset_id="__cached_child_ports__",
        half_extents=child_half_extents,
        out_docks_per_face=out_docks_per_face,
    )
    child_context = LinkContext(
        asset_id=None,
        center=child_center,
        half_extents=child_half_extents,
        rotation=child_rotation,
        link_index=-1,
        depth=1,
        blocked_face=(
            None if out_docks_per_face.get(child_face) else child_face
        ),
        occupied_face_slots=child_occupied,
    )
    return (
        "physical-attachment-successor-v2",
        _physical_cuboid_key(child_center, child_rotation, child_half_extents),
        _physical_port_descriptor(local_parent, parent_face, parent_dock),
        _physical_open_port_key(child_context, child_asset, child_occupied),
    )


def _physical_attachment_successor_key(
    parent: LinkContext,
    *,
    parent_face: int,
    parent_dock: dict,
    child_asset: AssetSpec,
    child_face: int,
    child_dock: dict,
    facing: int,
    child_out_ports_key: Tuple | None = None,
) -> Tuple:
    return _local_physical_attachment_successor_key(
        tuple(parent.half_extents),
        int(parent_face),
        _dock_pose_key(parent_dock),
        tuple(child_asset.half_extents),
        int(child_face),
        int(child_dock.get("id", 0)),
        _dock_pose_key(child_dock),
        int(facing),
        (
            child_out_ports_key
            if child_out_ports_key is not None
            else _asset_out_ports_key(child_asset)
        ),
    )


def _prepare_attachment(
    state: SearchState,
    parent: LinkContext,
    *,
    parent_face: int,
    parent_dock: dict,
    child_asset: AssetSpec,
    child_face: int,
    child_dock: dict,
    facing: int,
    child_out_ports_key: Tuple | None = None,
    state_token: Tuple | None = None,
) -> PreparedAttachment:
    """Compute one candidate pose, AABB, and dedup key exactly once."""

    child_center, child_rotation = _connection_pose_from_attachment(
        parent,
        parent_face=parent_face,
        parent_dock=parent_dock,
        child_asset=child_asset,
        child_face=child_face,
        child_dock=child_dock,
        facing=facing,
    )
    child_box = _oriented_aabb_from_center_half(
        child_center,
        child_asset.half_extents,
        child_rotation,
    )
    return PreparedAttachment(
        state_token=(
            state_token
            if state_token is not None
            else _attachment_state_token(state, parent)
        ),
        child_center=child_center,
        child_rotation=child_rotation,
        child_box=child_box,
        physical_successor_key=_physical_attachment_successor_key(
            parent,
            parent_face=parent_face,
            parent_dock=parent_dock,
            child_asset=child_asset,
            child_face=child_face,
            child_dock=child_dock,
            facing=facing,
            child_out_ports_key=child_out_ports_key,
        ),
    )


def _passes_child_attachment_collision_filter(
    context: LinkContext,
    asset_by_id: Dict[str, AssetSpec],
    p: int,
    d: int,
    child_asset: AssetSpec,
    q: int,
    child_dock_id: int,
    f: int,
    placed_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    forbidden_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
) -> bool:
    """Return whether the fully specified dock-to-dock attachment is collision-free."""
    parent_dock = _dock_by_id(context, asset_by_id, p, d)
    child_dock = _child_dock_by_id(child_asset, q, child_dock_id)
    if parent_dock is None or child_dock is None:
        return False
    _, child_box = _candidate_child_box(
        context,
        parent_face=p,
        parent_dock=parent_dock,
        child_asset=child_asset,
        child_face=q,
        child_dock=child_dock,
        facing=f,
    )
    return not _collides_with_any_box(
        child_box,
        placed_boxes=placed_boxes,
        forbidden_boxes=forbidden_boxes,
        parent_link_index=context.link_index,
    )


def _available_parent_slots(
    context: LinkContext,
    asset_by_id: Dict[str, AssetSpec],
    placed_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    forbidden_boxes: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    child_asset: AssetSpec,
) -> List[Tuple[int, int]]:
    """Enumerate selectable parent-side (p, d) tuples for AddLink action.

    Args:
        context: Current link context.
        asset_by_id: Asset lookup by id.
        placed_boxes: Existing placed AABBs.
        forbidden_boxes: Fixed external AABBs that must not be overlapped.
        child_asset: Candidate child asset metadata.

    Returns:
        List of available parent-side (p, d) tuples.
    """
    result: List[Tuple[int, int]] = []
    for p in range(FACE_COUNT):
        if context.blocked_face == p:
            continue

        occupied_docks = context.occupied_face_slots.get(p, frozenset())
        for dock in _parent_face_docks(context, asset_by_id, p):
            dock_id = int(dock.get("id", 0))
            if dock_id in occupied_docks:
                continue
            # Collision depends on both parent and child dock offsets. It is
            # checked once the complete attachment is known in valid_actions()
            # and apply_action(); filtering here would reject legal edge pairs.
            result.append((p, dock_id))
    return result


def _asset_index(assets: List[AssetSpec]) -> Dict[str, AssetSpec]:
    """Build fast lookup table from asset id to spec."""
    return {asset.asset_id: asset for asset in assets}


def function_count(state: SearchState) -> int:
    """Return current selected function-group count.

    A function/end-effector is represented by a generated node annotated with
    ``start_function_group=True``. The end-effector is later computed from the
    terminal leaves in that node's subtree.
    """
    return len(state.function_group_roots)


def function_count_completion_feasible(
    state: SearchState,
    max_depth: int,
    target_function_count: int | None,
    function_count_margin: int,
) -> bool:
    """Return whether any continuation can still satisfy the count band.

    Every missing function group requires at least one remaining AddLink to
    carry ``start_function_group=True``. A terminal leaf outside a function
    group also requires a future group-starting descendant; once the upper
    count bound is reached, such a leaf is irrecoverable.
    """

    if target_function_count is None:
        return True
    lower = max(
        0,
        int(target_function_count) - int(function_count_margin),
    )
    upper = int(target_function_count) + int(function_count_margin)
    actual = function_count(state)
    remaining_links = max(0, int(max_depth) - int(state.link_count))
    if actual > upper or actual + remaining_links < lower:
        return False
    if (
        actual >= upper
        and _current_terminal_leaf(state)
        and state.stack[-1].active_function_group_root is None
    ):
        return False
    return True


def _projected_addlink_function_count_feasible(
    state: SearchState,
    *,
    start_function_group: bool,
    max_depth: int,
    target_function_count: int | None,
    function_count_margin: int,
) -> bool:
    """Check count feasibility after one AddLink without rebuilding geometry."""

    if target_function_count is None:
        return True
    lower = max(
        0,
        int(target_function_count) - int(function_count_margin),
    )
    upper = int(target_function_count) + int(function_count_margin)
    actual_after = function_count(state) + int(start_function_group)
    remaining_after = max(
        0,
        int(max_depth) - (int(state.link_count) + 1),
    )
    if actual_after > upper or actual_after + remaining_after < lower:
        return False
    child_group = (
        -1
        if start_function_group
        else state.stack[-1].active_function_group_root
    )
    if actual_after >= upper and child_group is None:
        return False
    return True


def within_function_count_margin(
    state: SearchState,
    target_function_count: int | None,
    function_count_margin: int,
) -> bool:
    """Return whether state's leaf count is within the requested target band."""
    if target_function_count is None:
        return True
    actual = function_count(state)
    lower = max(0, target_function_count - function_count_margin)
    upper = target_function_count + function_count_margin
    return lower <= actual <= upper


def function_group_constraints_satisfied(
    state: SearchState,
    function_group_depth_delta: int | None,
) -> bool:
    """Return whether completed function-group annotations are valid."""
    if state.uncovered_terminal_leaves != 0:
        return False
    for root_index in state.function_group_roots:
        depths = state.function_group_leaf_depths.get(root_index)
        if depths is None:
            return False
        if function_group_depth_delta is not None:
            min_depth, max_depth = depths
            if max_depth - min_depth > int(function_group_depth_delta):
                return False
    return True


def _current_terminal_leaf(state: SearchState) -> bool:
    """Return whether the current DFS cursor is a generated terminal leaf."""
    if not state.stack:
        return False
    current = state.stack[-1]
    return (
        current.link_index >= 0
        and current.link_index != state.root_link_index
        and state.child_counts.get(current.link_index, 0) == 0
    )


def _end_action_allowed(
    state: SearchState,
    *,
    target_function_count: int | None,
    function_count_margin: int,
    function_group_depth_delta: int | None,
) -> bool:
    """Return whether closing the current DFS cursor can lead to a valid tree."""
    if not state.stack:
        return False

    current = state.stack[-1]
    if len(state.stack) == 1 and target_function_count is not None:
        lower = max(0, int(target_function_count) - int(function_count_margin))
        upper = int(target_function_count) + int(function_count_margin)
        actual = function_count(state)
        if actual < lower or actual > upper:
            return False

    if not _current_terminal_leaf(state):
        return True

    group_root = current.active_function_group_root
    if group_root is None:
        return False
    if function_group_depth_delta is None:
        return True
    old = state.function_group_leaf_depths.get(group_root)
    min_depth, max_depth = (current.depth, current.depth) if old is None else old
    min_depth = min(min_depth, current.depth)
    max_depth = max(max_depth, current.depth)
    return max_depth - min_depth <= int(function_group_depth_delta)


def apply_action(
    state: SearchState,
    action: Action,
    assets: List[AssetSpec],
) -> SearchState:
    """Apply one grammar action and return the next immutable state.

    Args:
        state: Current partial construction state.
        action: Action token to apply.
        assets: Available assets. Used to validate dock/facing legality.

    Returns:
        Next state after applying action.

    Raises:
        ValueError: If action violates grammar rules.
    """
    if state.is_complete:
        raise ValueError("Cannot apply actions to a completed sequence.")

    if isinstance(action, SelectRootRotation):
        if not state.root_rotation_options:
            raise ValueError("Root rotation selection is not enabled for this search.")
        if state.sequence or state.root_rotation_mode is not None:
            raise ValueError("Root rotation must be selected exactly once as the first action.")
        if action.mode not in state.root_rotation_options:
            raise ValueError(
                f"Root rotation mode {action.mode!r} is not enabled; "
                f"expected one of {state.root_rotation_options}."
            )
        return replace(
            state,
            sequence=[action],
            root_rotation_mode=action.mode,
        )

    if state.root_rotation_options and state.root_rotation_mode is None:
        raise ValueError("SelectRootRotation must be the first grammar action.")

    if isinstance(action, AddLink):
        if not state.stack:
            raise ValueError("Invalid state: empty stack before AddLink.")
        parent = state.stack[-1]

        asset_by_id = _asset_index(assets)
        child_asset = asset_by_id.get(action.asset_id)
        if child_asset is None:
            raise ValueError(f"Invalid AddLink: unknown asset_id={action.asset_id}")
        if action.start_function_group and parent.active_function_group_root is not None:
            raise ValueError("Invalid AddLink: nested function-group roots are not allowed.")

        available_slots = _available_parent_slots(
            parent,
            asset_by_id,
            placed_boxes=state.placed_boxes,
            forbidden_boxes=state.forbidden_boxes,
            child_asset=child_asset,
        )
        if (action.p, action.d) not in available_slots:
            raise ValueError(
                "Invalid AddLink: "
                f"(p={action.p}, d={action.d}) is not available on current link."
            )
        updated_face_slots = dict(parent.occupied_face_slots)
        occupied_on_face = set(updated_face_slots.get(action.p, frozenset()))
        occupied_on_face.add(action.d)
        updated_face_slots[action.p] = frozenset(occupied_on_face)

        parent_dock = _dock_by_id(parent, asset_by_id, action.p, action.d)
        if parent_dock is None:
            raise ValueError(
                "Invalid AddLink: "
                f"dock d={action.d} not found on face p={action.p} for current parent."
            )

        child_dock = _child_dock_by_id(child_asset, action.q, action.child_dock_id)
        if child_dock is None:
            raise ValueError(
                "Invalid AddLink: "
                f"child_dock_id={action.child_dock_id} not found on face q={action.q} "
                f"for child asset {action.asset_id}."
            )
        if action.f not in compatible_orientations(parent_dock, child_dock):
            raise ValueError(
                "Invalid AddLink: "
                f"orientation f={action.f} is not available for parent/child "
                f"dock pair ({action.p}, {action.d}) -> "
                f"({action.q}, {action.child_dock_id})."
            )

        prepared = action.prepared_attachment
        prepared_matches = (
            prepared is not None
            and prepared.state_token == _attachment_state_token(state, parent)
        )
        if not prepared_matches:
            prepared = _prepare_attachment(
                state,
                parent,
                parent_face=action.p,
                parent_dock=parent_dock,
                child_asset=child_asset,
                child_face=action.q,
                child_dock=child_dock,
                facing=action.f,
            )
            if _collides_with_any_box(
                prepared.child_box,
                placed_boxes=state.placed_boxes,
                forbidden_boxes=state.forbidden_boxes,
                parent_link_index=parent.link_index,
            ):
                raise ValueError(
                    "Invalid AddLink: candidate child cuboid overlaps existing component."
                )
        child_center = prepared.child_center
        child_rotation = prepared.child_rotation
        child_box = prepared.child_box

        new_placed = list(state.placed_boxes)
        new_placed.append(child_box)
        child_index = len(new_placed) - 1
        new_child_counts = dict(state.child_counts)
        if parent.link_index >= 0:
            new_child_counts[parent.link_index] = new_child_counts.get(parent.link_index, 0) + 1
        new_child_counts[child_index] = 0
        new_function_group_roots = state.function_group_roots
        if action.start_function_group:
            new_function_group_roots = (*new_function_group_roots, child_index)

        new_parent = replace(
            parent,
            occupied_face_slots=updated_face_slots,
        )
        child_active_function_group_root = (
            child_index if action.start_function_group else parent.active_function_group_root
        )
        child_ingress_out_docks = child_asset.out_docks_for_face(action.q)
        child_occupied_face_slots = {
            face: frozenset() for face in range(FACE_COUNT)
        }
        if any(
            int(dock.get("id", -1)) == action.child_dock_id
            for dock in child_ingress_out_docks
        ):
            child_occupied_face_slots[action.q] = frozenset(
                {action.child_dock_id}
            )
        new_child = LinkContext(
            asset_id=action.asset_id,
            center=child_center,
            half_extents=child_asset.half_extents,
            rotation=child_rotation,
            link_index=child_index,
            depth=parent.depth + 1,
            blocked_face=(
                None if child_ingress_out_docks else action.q
            ),
            start_function_group=action.start_function_group,
            active_function_group_root=child_active_function_group_root,
            occupied_face_slots=child_occupied_face_slots,
        )
        new_realized_bodies = (
            *state.realized_bodies,
            RealizedBody(
                link_index=child_index,
                asset_id=action.asset_id,
                center=child_center,
                half_extents=child_asset.half_extents,
                rotation=child_rotation,
                function_group_root=child_active_function_group_root,
            ),
        )
        sequence_action = AddLink(
            asset_id=action.asset_id,
            p=action.p,
            d=action.d,
            f=action.f,
            q=action.q,
            child_dock_id=action.child_dock_id,
            start_function_group=action.start_function_group,
        )

        new_stack = list(state.stack)
        new_stack[-1] = new_parent
        new_stack.append(new_child)
        return SearchState(
            # Prepared geometry is node-local scratch data. Keeping it out of
            # completed sequences avoids bloating evaluator IPC and artifacts.
            sequence=[*state.sequence, sequence_action],
            stack=new_stack,
            placed_boxes=new_placed,
            forbidden_boxes=state.forbidden_boxes,
            link_count=state.link_count + 1,
            child_counts=new_child_counts,
            root_link_index=state.root_link_index,
            function_group_roots=new_function_group_roots,
            function_group_leaf_depths=state.function_group_leaf_depths,
            realized_bodies=new_realized_bodies,
            function_semantic_labels=state.function_semantic_labels,
            uncovered_terminal_leaves=state.uncovered_terminal_leaves,
            root_rotation_options=state.root_rotation_options,
            root_rotation_mode=state.root_rotation_mode,
            is_complete=False,
        )

    if isinstance(action, End):
        if not state.stack:
            raise ValueError("Invalid state: empty stack before End.")
        current = state.stack[-1]
        new_function_group_leaf_depths = dict(state.function_group_leaf_depths)
        new_uncovered_terminal_leaves = state.uncovered_terminal_leaves
        if _current_terminal_leaf(state):
            group_root = current.active_function_group_root
            if group_root is None:
                new_uncovered_terminal_leaves += 1
            else:
                old_depths = new_function_group_leaf_depths.get(group_root)
                if old_depths is None:
                    new_function_group_leaf_depths[group_root] = (current.depth, current.depth)
                else:
                    new_function_group_leaf_depths[group_root] = (
                        min(old_depths[0], current.depth),
                        max(old_depths[1], current.depth),
                    )
        if len(state.stack) == 1:
            return SearchState(
                sequence=[*state.sequence, action],
                stack=[],
                placed_boxes=state.placed_boxes,
                forbidden_boxes=state.forbidden_boxes,
                link_count=state.link_count,
                child_counts=state.child_counts,
                root_link_index=state.root_link_index,
                function_group_roots=state.function_group_roots,
                function_group_leaf_depths=new_function_group_leaf_depths,
                realized_bodies=state.realized_bodies,
                function_semantic_labels=state.function_semantic_labels,
                uncovered_terminal_leaves=new_uncovered_terminal_leaves,
                root_rotation_options=state.root_rotation_options,
                root_rotation_mode=state.root_rotation_mode,
                is_complete=True,
            )

        return SearchState(
            sequence=[*state.sequence, action],
            stack=state.stack[:-1],
            placed_boxes=state.placed_boxes,
            forbidden_boxes=state.forbidden_boxes,
            link_count=state.link_count,
            child_counts=state.child_counts,
            root_link_index=state.root_link_index,
            function_group_roots=state.function_group_roots,
            function_group_leaf_depths=new_function_group_leaf_depths,
            realized_bodies=state.realized_bodies,
            function_semantic_labels=state.function_semantic_labels,
            uncovered_terminal_leaves=new_uncovered_terminal_leaves,
            root_rotation_options=state.root_rotation_options,
            root_rotation_mode=state.root_rotation_mode,
            is_complete=False,
        )

    raise ValueError(f"Unsupported action type: {type(action)}")


def valid_actions(
    state: SearchState,
    assets: List[AssetSpec],
    max_depth: int,
    max_actions: int | None = None,
    target_function_count: int | None = None,
    function_count_margin: int = 0,
    function_group_depth_delta: int | None = None,
    rng: random.Random | None = None,
) -> List[Action]:
    """Generate valid grammar actions from current state.

    Args:
        state: Current partial state.
        assets: Available asset candidates for AddLink.
        max_depth: Maximum number of links in final skeleton.
        max_actions: Optional cap on number of returned actions.
        target_function_count: Optional target number of function groups.
        function_count_margin: Allowed absolute discrepancy from the target.
        function_group_depth_delta: Maximum allowed leaf-depth spread inside
            one selected function group.

    Returns:
        List of valid action objects.
    """
    if state.is_complete or not state.stack:
        return []

    if state.root_rotation_options and state.root_rotation_mode is None:
        return [
            SelectRootRotation(mode)
            for mode in state.root_rotation_options
        ]
    if not function_count_completion_feasible(
        state,
        max_depth,
        target_function_count,
        function_count_margin,
    ):
        return []

    actions: List[Action] = []
    if _end_action_allowed(
        state,
        target_function_count=target_function_count,
        function_count_margin=function_count_margin,
        function_group_depth_delta=function_group_depth_delta,
    ):
        actions.append(End())
    can_add = state.link_count < max_depth
    if can_add:
        asset_by_id = _asset_index(assets)
        max_function_count = (
            target_function_count + function_count_margin
            if target_function_count is not None
            else None
        )
        grouped_actions: List[List[Action]] = []
        seen_physical_successors = set()
        attachment_state_token = _attachment_state_token(state, state.stack[-1])
        asset_order = list(assets)
        if rng is not None:
            rng.shuffle(asset_order)
        for asset in asset_order:
            if not asset.searchable:
                continue
            asset_actions: List[Action] = []
            child_out_ports_key = _asset_out_ports_key(asset)
            available_slots = _available_parent_slots(
                state.stack[-1],
                asset_by_id,
                placed_boxes=state.placed_boxes,
                forbidden_boxes=state.forbidden_boxes,
                child_asset=asset,
            )
            for p, d in available_slots:
                parent_dock = _dock_by_id(state.stack[-1], asset_by_id, p, d)
                if parent_dock is None:
                    continue
                for q in range(FACE_COUNT):
                    child_slots = _pair_facing_slots(parent_dock, asset, q)
                    if not child_slots:
                        continue
                    for child_dock_id, f in child_slots:
                        child_dock = _child_dock_by_id(asset, q, child_dock_id)
                        if parent_dock is None or child_dock is None:
                            continue
                        prepared = _prepare_attachment(
                            state,
                            state.stack[-1],
                            parent_face=p,
                            parent_dock=parent_dock,
                            child_asset=asset,
                            child_face=q,
                            child_dock=child_dock,
                            facing=f,
                            child_out_ports_key=child_out_ports_key,
                            state_token=attachment_state_token,
                        )
                        signature = prepared.physical_successor_key
                        if signature in seen_physical_successors:
                            continue
                        seen_physical_successors.add(signature)
                        if _collides_with_any_box(
                            prepared.child_box,
                            placed_boxes=state.placed_boxes,
                            forbidden_boxes=state.forbidden_boxes,
                            parent_link_index=state.stack[-1].link_index,
                        ):
                            continue
                        start_options = [False]
                        can_start_group = state.stack[-1].active_function_group_root is None
                        if can_start_group and (
                            max_function_count is None
                            or function_count(state) + 1 <= max_function_count
                        ):
                            start_options.append(True)
                        for start_function_group in start_options:
                            if not _projected_addlink_function_count_feasible(
                                state,
                                start_function_group=start_function_group,
                                max_depth=max_depth,
                                target_function_count=target_function_count,
                                function_count_margin=function_count_margin,
                            ):
                                continue
                            asset_actions.append(
                                AddLink(
                                    asset_id=asset.asset_id,
                                    p=p,
                                    d=d,
                                    f=f,
                                    q=q,
                                    child_dock_id=child_dock_id,
                                    start_function_group=start_function_group,
                                    prepared_attachment=prepared,
                                )
                            )
            if asset_actions:
                grouped_actions.append(asset_actions)

        # Interleave by asset before applying max_actions. Without this,
        # high-dock assets earlier in assets.json can exhaust the cap and starve
        # later primitives from ever being expanded.
        max_group_len = max((len(group) for group in grouped_actions), default=0)
        for action_idx in range(max_group_len):
            for group in grouped_actions:
                if action_idx >= len(group):
                    continue
                actions.append(group[action_idx])
                if max_actions is not None and len(actions) >= max_actions:
                    return actions
    return actions


def is_terminal(state: SearchState, max_depth: int) -> bool:
    """Return whether no further progress is possible or needed.

    Args:
        state: Search state.
        max_depth: Maximum number of links.

    Returns:
        True if state is complete, invalid, or saturated.
    """
    if state.is_complete:
        return True
    if not state.stack:
        return True
    if state.link_count > max_depth:
        return True
    return False
