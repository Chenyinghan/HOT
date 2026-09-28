"""Shared port-pair connection geometry.

Standard surface connections intentionally remain owned by the existing BASS
and XML paths.  This module only resolves the semantic panel-edge corner case,
where opposing cuboid face normals cannot produce a full right-angle seam.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


STANDARD_CONNECTION = "standard_face"
EDGE_CORNER_CONNECTION = "edge_corner_butt"
EDGE_CORNER_ORIENTATIONS = (0, 1, 2, 3)

_FACE_AXIS = {0: 2, 1: 0, 2: 1, 3: 0, 4: 2, 5: 1}
_FACE_SIGN = {0: 1.0, 1: 1.0, 2: 1.0, 3: -1.0, 4: -1.0, 5: -1.0}


def is_edge_port(port: Mapping[str, Any] | None) -> bool:
    return bool(port is not None and port.get("edge", False))


def connection_family(
    parent_port: Mapping[str, Any] | None,
    child_port: Mapping[str, Any] | None,
) -> str:
    """Resolve the mating family after both physical ports are selected."""

    if not (is_edge_port(parent_port) and is_edge_port(child_port)):
        return STANDARD_CONNECTION
    parent_profile = str(parent_port.get("edge_profile", ""))
    child_profile = str(child_port.get("edge_profile", ""))
    if not parent_profile or parent_profile != child_profile:
        return STANDARD_CONNECTION
    return EDGE_CORNER_CONNECTION


def compatible_orientations(
    parent_port: Mapping[str, Any],
    child_port: Mapping[str, Any],
) -> tuple[int, ...]:
    """Return the orientation language selected by a concrete port pair."""

    if connection_family(parent_port, child_port) == EDGE_CORNER_CONNECTION:
        return EDGE_CORNER_ORIENTATIONS
    values = child_port.get("facing_options", (0, 1, 2, 3))
    return tuple(sorted({int(value) for value in values}))


def _axis_vector(axis: int) -> np.ndarray:
    if axis not in (0, 1, 2):
        raise ValueError(f"edge semantic axis must lie in [0, 2], got {axis}")
    out = np.zeros(3, dtype=np.float64)
    out[int(axis)] = 1.0
    return out


def edge_port_frame(
    face: int,
    port: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the right-handed semantic frame ``(seam, outward, normal)``.

    ``normal`` is the positive panel-thickness direction.  ``seam`` is made
    deterministic from catalog metadata, then flipped when necessary so that
    ``cross(normal, seam)`` agrees with the port's outward face normal.
    """

    if not is_edge_port(port):
        raise ValueError("edge_port_frame requires an edge port")
    face = int(face)
    outward = _FACE_SIGN[face] * _axis_vector(_FACE_AXIS[face])
    seam = _axis_vector(int(port["edge_tangent_axis"]))
    normal = _axis_vector(int(port["panel_normal_axis"]))
    if abs(float(np.dot(seam, normal))) > 1e-12:
        raise ValueError("edge tangent and panel normal axes must differ")
    if abs(float(np.dot(outward, normal))) > 1e-12:
        raise ValueError("panel normal cannot be normal to its edge-port face")
    if float(np.dot(np.cross(normal, seam), outward)) < 0.0:
        seam = -seam
    return seam, outward, normal


def resolve_edge_corner_pose(
    *,
    parent_rotation: Sequence[float] | np.ndarray,
    parent_anchor: Sequence[float] | np.ndarray,
    parent_half_extents: Sequence[float] | np.ndarray,
    parent_face: int,
    parent_port: Mapping[str, Any],
    child_half_extents: Sequence[float] | np.ndarray,
    child_face: int,
    child_port: Mapping[str, Any],
    orientation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a collision-free, outer-flush panel corner in parent space.

    The returned rotation maps child-local coordinates into the same frame as
    ``parent_anchor``.  The returned target is the transformed child edge
    anchor.  Raw midsurface edge centers deliberately differ by the two
    half-thickness offsets; the physical broad-edge contact seam coincides.
    """

    if connection_family(parent_port, child_port) != EDGE_CORNER_CONNECTION:
        raise ValueError("resolve_edge_corner_pose requires compatible edge ports")
    orientation = int(orientation)
    if orientation not in EDGE_CORNER_ORIENTATIONS:
        raise ValueError(
            f"edge-corner orientation must be one of {EDGE_CORNER_ORIENTATIONS}, "
            f"got {orientation}"
        )

    parent_R = np.asarray(parent_rotation, dtype=np.float64).reshape(3, 3)
    parent_anchor = np.asarray(parent_anchor, dtype=np.float64).reshape(3)
    parent_half = np.asarray(parent_half_extents, dtype=np.float64).reshape(3)
    child_half = np.asarray(child_half_extents, dtype=np.float64).reshape(3)

    parent_s_local, parent_b_local, parent_n_local = edge_port_frame(
        parent_face,
        parent_port,
    )
    child_s, child_b, child_n = edge_port_frame(child_face, child_port)
    parent_s = parent_R @ parent_s_local
    parent_b = parent_R @ parent_b_local
    parent_n = parent_R @ parent_n_local

    # Bit 0 selects the side of the parent panel; bit 1 reverses the seam
    # tangent and therefore independently selects which child broad face is
    # flush with the parent edge.  Four states are required for multi-wall
    # corners: coupling fold and flush directions makes sibling wall
    # thicknesses overlap even though each parent-child butt is individually
    # valid.
    fold_side = 1.0 if (orientation & 1) == 0 else -1.0
    seam_sign = 1.0 if (orientation & 2) == 0 else -1.0
    target_s = seam_sign * parent_s
    target_b = -fold_side * parent_n
    target_n = np.cross(target_s, target_b)
    target_n /= max(float(np.linalg.norm(target_n)), 1e-12)

    source_frame = np.column_stack([child_s, child_b, child_n])
    target_frame = np.column_stack([target_s, target_b, target_n])
    rotation = target_frame @ source_frame.T

    parent_normal_axis = int(parent_port["panel_normal_axis"])
    child_normal_axis = int(child_port["panel_normal_axis"])
    parent_half_thickness = float(parent_half[parent_normal_axis])
    child_half_thickness = float(child_half[child_normal_axis])
    flush_sign = float(np.dot(target_n, parent_b))
    if abs(abs(flush_sign) - 1.0) > 1e-8:
        raise ValueError("edge-corner semantic frames do not form an orthogonal butt joint")
    target_anchor = (
        parent_anchor
        + fold_side * parent_half_thickness * parent_n
        - flush_sign * child_half_thickness * parent_b
    )
    return rotation, target_anchor
