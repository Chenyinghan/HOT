"""Constraint primitives for connected Head morphology.

These functions mirror the currently verified mount-preserving Runner while
remaining independent of optimizer and task classes.
"""

from __future__ import annotations

import numpy as np

from bilevel.parameterization.geometry import (
    DIRECT_PARAM_DIM,
    RetractionResult,
    constraint_jacobian,
    constraints,
    max_vertex_displacement,
    validate_hexahedron,
    vertices_from_q,
)
from bilevel.parameterization.interpolation import FACE_CORNERS


_FACE_ID_TO_DOCK = {
    0: (2, 1),
    1: (0, 1),
    2: (1, 1),
    3: (0, -1),
    4: (2, -1),
    5: (1, -1),
}
_VERTEX_PARAMETER_INDICES = {
    0: (),
    1: (0,),
    2: (1, 2),
    3: (3, 4, 5),
    4: (6, 7, 8),
    5: (9, 10, 11),
    6: (12, 13, 14),
    7: (15, 16, 17),
}


def mount_face_parameter_indices(face_id: int | None) -> tuple[int, ...]:
    """Return the direct coordinates controlling an XML mount face."""

    dock = _FACE_ID_TO_DOCK.get(
        None if face_id is None else int(face_id)
    )
    if dock is None:
        return tuple(range(DIRECT_PARAM_DIM))
    indices: set[int] = set()
    for vertex in FACE_CORNERS[dock]:
        indices.update(_VERTEX_PARAMETER_INDICES[vertex])
    return tuple(sorted(indices))


def _selection_jacobian(
    indices: tuple[int, ...],
    width: int,
) -> np.ndarray:
    matrix = np.zeros((len(indices), width), dtype=np.float64)
    for row, index in enumerate(indices):
        matrix[row, index] = 1.0
    return matrix


def project_block_tangent(
    q: np.ndarray,
    vector: np.ndarray,
    *,
    face_mask,
    frozen_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Project one Head direction onto planar and Handle-mount constraints."""

    q_values = np.asarray(q, dtype=np.float64).reshape(DIRECT_PARAM_DIM)
    direction = np.asarray(
        vector,
        dtype=np.float64,
    ).reshape(DIRECT_PARAM_DIM)
    jacobians = [constraint_jacobian(q_values, face_mask=face_mask)]
    if frozen_indices:
        jacobians.append(
            _selection_jacobian(frozen_indices, DIRECT_PARAM_DIM)
        )
    jacobian = np.vstack(
        [value for value in jacobians if value.shape[0]]
    )
    if not jacobian.shape[0]:
        return direction
    gram = jacobian @ jacobian.T
    damping = (
        1e-10
        * float(np.trace(gram))
        / max(float(jacobian.shape[0]), 1.0)
        + 1e-12
    )
    correction = jacobian.T @ np.linalg.solve(
        gram + damping * np.eye(jacobian.shape[0]),
        jacobian @ direction,
    )
    return direction - correction


def _residual(
    q: np.ndarray,
    *,
    face_mask,
    frozen_indices: tuple[int, ...],
    baseline: np.ndarray,
) -> np.ndarray:
    planar = constraints(q, face_mask=face_mask)
    fixed = (
        q[list(frozen_indices)] - baseline[list(frozen_indices)]
        if frozen_indices
        else np.zeros(0, dtype=np.float64)
    )
    return np.concatenate([planar, fixed])


def _norm_inf(values: np.ndarray) -> float:
    return (
        0.0
        if not values.size
        else float(np.linalg.norm(values, ord=np.inf))
    )


def retract_block(
    q: np.ndarray,
    step: np.ndarray,
    *,
    face_mask,
    frozen_indices: tuple[int, ...] = (),
    baseline: np.ndarray,
) -> RetractionResult:
    """Retract one Head using the verified mount-preserving Newton solve."""

    original = np.asarray(q, dtype=np.float64).reshape(DIRECT_PARAM_DIM)
    baseline_values = np.asarray(
        baseline,
        dtype=np.float64,
    ).reshape(DIRECT_PARAM_DIM)
    z = original + np.asarray(
        step,
        dtype=np.float64,
    ).reshape(DIRECT_PARAM_DIM)
    if not np.all(np.isfinite(z)):
        return RetractionResult(
            q=original.copy(),
            ok=False,
            iterations=0,
            residual_inf=float("inf"),
        )
    select = (
        _selection_jacobian(frozen_indices, DIRECT_PARAM_DIM)
        if frozen_indices
        else np.zeros((0, DIRECT_PARAM_DIM), dtype=np.float64)
    )
    residual_norm = float("inf")
    iterations = 0
    for iterations in range(1, 13):
        residual = _residual(
            z,
            face_mask=face_mask,
            frozen_indices=frozen_indices,
            baseline=baseline_values,
        )
        residual_norm = _norm_inf(residual)
        if residual_norm <= 1e-10:
            break
        jacobian = np.vstack(
            [
                constraint_jacobian(z, face_mask=face_mask),
                select,
            ]
        )
        gram = jacobian @ jacobian.T
        damping = (
            1e-10
            * float(np.trace(gram))
            / max(float(jacobian.shape[0]), 1.0)
            + 1e-12
        )
        delta = -jacobian.T @ np.linalg.solve(
            gram + damping * np.eye(jacobian.shape[0]),
            residual,
        )
        accepted = False
        beta = 1.0
        for _ in range(12):
            trial = z + beta * delta
            trial_residual = _residual(
                trial,
                face_mask=face_mask,
                frozen_indices=frozen_indices,
                baseline=baseline_values,
            )
            trial_norm = _norm_inf(trial_residual)
            if np.all(np.isfinite(trial)) and trial_norm < residual_norm:
                z = trial
                residual_norm = trial_norm
                accepted = True
                break
            beta *= 0.5
        if not accepted:
            return RetractionResult(
                q=original.copy(),
                ok=False,
                iterations=iterations,
                residual_inf=residual_norm,
            )
    if frozen_indices:
        z[list(frozen_indices)] = baseline_values[list(frozen_indices)]
    try:
        validate_hexahedron(z, face_mask=face_mask)
    except ValueError:
        return RetractionResult(
            q=original.copy(),
            ok=False,
            iterations=iterations,
            residual_inf=residual_norm,
        )
    final_residual = _residual(
        z,
        face_mask=face_mask,
        frozen_indices=frozen_indices,
        baseline=baseline_values,
    )
    return RetractionResult(
        q=z,
        ok=True,
        iterations=iterations,
        residual_inf=_norm_inf(final_residual),
    )


def validate_block_geometry(
    q: np.ndarray,
    *,
    baseline: np.ndarray,
    face_mask,
    frozen_indices: tuple[int, ...] = (),
    mount_face_id: int | None = None,
    reference_length: float = 1.0,
    mount_face_tolerance: float = 1e-7,
    max_shape_displacement: float = 0.125,
) -> None:
    """Validate one Head's shape, displacement, and fixed Handle boundary."""

    values = np.asarray(q, dtype=np.float64).reshape(DIRECT_PARAM_DIM)
    baseline_values = np.asarray(
        baseline,
        dtype=np.float64,
    ).reshape(DIRECT_PARAM_DIM)
    validate_hexahedron(values, face_mask=face_mask)
    displacement = max_vertex_displacement(baseline_values, values)
    if displacement > float(max_shape_displacement) + 1e-10:
        raise ValueError(
            "Head shape displacement exceeds limit: "
            f"{displacement:.3e} > {float(max_shape_displacement):.3e}"
        )
    if not frozen_indices:
        return
    dock = _FACE_ID_TO_DOCK.get(
        None if mount_face_id is None else int(mount_face_id)
    )
    if dock is None:
        raise ValueError(
            f"invalid direct Handle mount face {mount_face_id!r}"
        )
    corners = FACE_CORNERS[dock]
    vertices = vertices_from_q(values)
    baseline_vertices = vertices_from_q(baseline_values)
    face_error = float(reference_length) * float(
        np.max(
            np.linalg.norm(
                (
                    vertices[:, corners]
                    - baseline_vertices[:, corners]
                ).T,
                axis=1,
            )
        )
    )
    if face_error > float(mount_face_tolerance):
        raise ValueError(
            "direct Handle mount face moved: "
            f"{face_error:.3e} > {float(mount_face_tolerance):.3e}"
        )
