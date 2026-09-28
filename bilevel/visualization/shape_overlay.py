"""Shared palette and geometry rules for Stage 1 -> Stage 2 overlays.

This module is deliberately renderer-independent so the canonical replay and
the recording-preview pipeline use exactly the same asset colors, displacement
threshold, triangle mask, submesh extraction, and 0.1% anti-z-fighting shell.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


SHAPE_OVERLAY_RELATIVE_OFFSET = 0.001

TOOL_COLORS = {
    "handle_brown": "0.5451 0.3686 0.2353 1",
    "head_deep_blue_gray": "0.2039 0.2863 0.3686 1",
    "head_blue_green": "0.1647 0.4980 0.5176 1",
    "head_deep_teal": "0.1216 0.3725 0.4078 1",
    "stage2_shape_red": "0.95 0.10 0.03 1",
}

# Color is keyed by asset identity, not node number.  Repeated instances of an
# asset therefore remain visually identical within and across all tasks.
TOOL_ASSET_COLORS = {
    "root/universal_handle": TOOL_COLORS["handle_brown"],
    "primitive/panel": TOOL_COLORS["head_deep_blue_gray"],
    "primitive/cube": TOOL_COLORS["head_deep_blue_gray"],
    "primitive/small_cuboid": TOOL_COLORS["head_blue_green"],
}


def apply_tool_palette(root) -> Dict[str, str]:
    """Apply the shared tool-only palette and return colors by body name.

    Bodies without tool asset metadata are deliberately untouched, which keeps
    every task's wall, nail, dust, target, and other scene objects unchanged.
    """

    assigned: Dict[str, str] = {}
    for body in root.iter("body"):
        asset_id = str(body.attrib.get("asset_id", ""))
        asset_role = str(body.attrib.get("asset_role", ""))
        if asset_id in TOOL_ASSET_COLORS:
            rgba = TOOL_ASSET_COLORS[asset_id]
        elif asset_role == "fixed_root":
            rgba = TOOL_COLORS["handle_brown"]
        elif asset_role == "head":
            rgba = TOOL_COLORS["head_deep_teal"]
        else:
            continue
        body.set("rgba", rgba)
        body_name = body.attrib.get("name")
        if body_name:
            assigned[str(body_name)] = rgba
    return assigned


def deformation_region(
    baseline_vertices: np.ndarray,
    final_vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, object]:
    """Return a face-local deformation region derived from vertex motion.

    A vertex is changed when its Stage 1 -> Stage 2 displacement exceeds the
    same numerical-noise tolerance used by the original whole-body overlay.
    A triangle enters the local overlay when at least one of its vertices is
    changed; this is the smallest triangle-level mask that cannot omit a moved
    vertex in RedMax's per-object material renderer.
    """

    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    final = np.asarray(final_vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if baseline.shape != final.shape or baseline.ndim != 2 or baseline.shape[0] != 3:
        raise ValueError(
            "shape-overlay baseline/final vertices must have matching 3xN shape"
        )
    if triangles.ndim != 2 or triangles.shape[0] != 3:
        raise ValueError("shape-overlay faces must have 3xF shape")
    if triangles.size and (
        int(np.min(triangles)) < 0 or int(np.max(triangles)) >= baseline.shape[1]
    ):
        raise ValueError("shape-overlay faces reference an invalid vertex")

    displacement = np.linalg.norm(final - baseline, axis=0)
    diagonal = float(np.linalg.norm(np.ptp(baseline, axis=1))) if baseline.shape[1] else 0.0
    scale = max(diagonal, 1.0)
    tolerance = max(1e-9, 1e-8 * scale)
    vertex_mask = displacement > tolerance
    if triangles.size:
        face_mask = np.any(vertex_mask[triangles], axis=0)
    else:
        face_mask = np.zeros(0, dtype=bool)

    return {
        "displacement": displacement,
        "vertex_mask": vertex_mask,
        "face_mask": face_mask,
        "tolerance": float(tolerance),
        "max_displacement": (
            float(np.max(displacement)) if displacement.size else 0.0
        ),
        "changed_vertex_count": int(np.count_nonzero(vertex_mask)),
        "vertex_count": int(vertex_mask.size),
        "changed_face_count": int(np.count_nonzero(face_mask)),
        "face_count": int(face_mask.size),
    }


def outward_render_shell(
    vertices: np.ndarray,
    faces: np.ndarray,
    relative_offset: float = SHAPE_OVERLAY_RELATIVE_OFFSET,
) -> Tuple[np.ndarray, float]:
    """Return the original visualization-only normal-offset shell.

    The offset is measured against the complete local mesh, before selecting
    the local deformation faces. This preserves the original exact 0.1% scale
    and its averaged full-mesh vertex normals.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if relative_offset <= 0.0 or vertices.ndim != 2 or vertices.shape[0] != 3:
        return np.array(vertices, copy=True), 0.0
    if faces.ndim != 2 or faces.shape[0] != 3 or not faces.size:
        return np.array(vertices, copy=True), 0.0
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=1)))
    absolute_offset = float(relative_offset) * diagonal
    if not np.isfinite(absolute_offset) or absolute_offset <= 0.0:
        return np.array(vertices, copy=True), 0.0

    normals = np.zeros_like(vertices, dtype=np.float64)
    for triangle in faces.T:
        i0, i1, i2 = (int(value) for value in triangle)
        if min(i0, i1, i2) < 0 or max(i0, i1, i2) >= vertices.shape[1]:
            continue
        face_normal = np.cross(
            vertices[:, i1] - vertices[:, i0],
            vertices[:, i2] - vertices[:, i0],
        )
        normals[:, i0] += face_normal
        normals[:, i1] += face_normal
        normals[:, i2] += face_normal
    lengths = np.linalg.norm(normals, axis=0)
    centroid = np.mean(vertices, axis=1, keepdims=True)
    radial = vertices - centroid
    radial_lengths = np.linalg.norm(radial, axis=0)
    valid = lengths > 1e-14
    normals[:, valid] /= lengths[valid]
    fallback = ~valid & (radial_lengths > 1e-14)
    normals[:, fallback] = radial[:, fallback] / radial_lengths[fallback]
    if float(np.sum(normals * radial)) < 0.0:
        normals *= -1.0
    return vertices + absolute_offset * normals, absolute_offset


def compact_submesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract selected triangles and remap them to a compact vertex array."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    if faces.ndim != 2 or faces.shape[0] != 3 or mask.size != faces.shape[1]:
        raise ValueError("shape-overlay face mask does not match 3xF faces")
    selected = faces[:, mask]
    if not selected.size:
        return np.zeros((3, 0), dtype=np.float64), np.zeros((3, 0), dtype=np.int64)
    used = np.unique(selected)
    remap = np.full(vertices.shape[1], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    return vertices[:, used], remap[selected]
