from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bilevel.parameterization.interpolation import FACE_CORNERS


GENERIC_DESIGN_PROTOCOL = "connected_direct_planar_hexahedron"
DIRECT_PARAM_DIM = 18

# Vertex order: O, A, B, C, D, E, F, G.
EDGE_INDICES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (2, 4),
    (2, 6),
    (3, 5),
    (3, 6),
    (4, 7),
    (5, 7),
    (6, 7),
)

FACE_ORDER = (
    (2, -1),
    (1, -1),
    (0, -1),
    (0, 1),
    (1, 1),
    (2, 1),
)

FACE_CONSTRAINT_TETS = (
    (0, 1, 2, 4),
    (0, 1, 3, 5),
    (0, 2, 3, 6),
    (1, 4, 5, 7),
    (2, 4, 6, 7),
    (3, 5, 6, 7),
)

DOCK_TO_FACE_INDEX = {dock: idx for idx, dock in enumerate(FACE_ORDER)}

# Three-point Gauss-Legendre quadrature mapped from [-1, 1] to [0, 1].
# It integrates the polynomial volume moments of a trilinear hexahedron
# without selecting min/max vertices, so the result stays differentiable at
# symmetric boxes.
GAUSS_POINTS = (
    0.5 * (1.0 - np.sqrt(3.0 / 5.0)),
    0.5,
    0.5 * (1.0 + np.sqrt(3.0 / 5.0)),
)
GAUSS_WEIGHTS = (5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0)


@dataclass(frozen=True)
class RetractionResult:
    q: np.ndarray
    ok: bool
    iterations: int
    residual_inf: float


def reference_length(extents: np.ndarray, mode: str = "max") -> float:
    dims = np.maximum(np.asarray(extents, dtype=np.float64).reshape(3), 1e-12)
    if mode == "geom":
        return float(np.prod(dims) ** (1.0 / 3.0))
    return float(np.max(dims))


def baseline_q_from_extents(extents: np.ndarray, *, ref_length: float | None = None) -> np.ndarray:
    dims = np.maximum(np.asarray(extents, dtype=np.float64).reshape(3), 1e-12)
    ref = float(ref_length) if ref_length is not None else reference_length(dims)
    x, y, z = dims / max(ref, 1e-12)
    return np.asarray(
        [
            x,
            0.0,
            y,
            0.0,
            0.0,
            z,
            x,
            y,
            0.0,
            x,
            0.0,
            z,
            0.0,
            y,
            z,
            x,
            y,
            z,
        ],
        dtype=np.float64,
    )


def vertices_from_q(q: np.ndarray) -> np.ndarray:
    p = np.asarray(q, dtype=np.float64).reshape(DIRECT_PARAM_DIM)
    ax, bx, by = p[:3]
    C = p[3:6]
    D = p[6:9]
    E = p[9:12]
    F = p[12:15]
    G = p[15:18]
    O = np.zeros(3, dtype=np.float64)
    A = np.asarray([ax, 0.0, 0.0], dtype=np.float64)
    B = np.asarray([bx, by, 0.0], dtype=np.float64)
    return np.stack([O, A, B, C, D, E, F, G], axis=1)


def normalize_face_mask(face_mask: np.ndarray | list[bool] | tuple[bool, ...] | None) -> np.ndarray:
    if face_mask is None:
        return np.ones(len(FACE_ORDER), dtype=bool)
    mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != len(FACE_ORDER):
        raise ValueError(f"face_mask must have length {len(FACE_ORDER)}, got {mask.shape[0]}")
    return mask


def constraints(q: np.ndarray, face_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None) -> np.ndarray:
    V = vertices_from_q(q)
    mask = normalize_face_mask(face_mask)

    def det4(i: int, j: int, k: int, l: int) -> float:
        P = V[:, i]
        return float(np.linalg.det(np.column_stack([V[:, j] - P, V[:, k] - P, V[:, l] - P])))

    values = [det4(*tet) for idx, tet in enumerate(FACE_CONSTRAINT_TETS) if mask[idx]]
    return np.asarray(values, dtype=np.float64)


def _constraint_jacobian_fd(
    q: np.ndarray,
    eps: float = 1e-6,
    face_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
) -> np.ndarray:
    q0 = np.asarray(q, dtype=np.float64).reshape(DIRECT_PARAM_DIM)
    c0 = constraints(q0, face_mask=face_mask)
    J = np.zeros((c0.shape[0], DIRECT_PARAM_DIM), dtype=np.float64)
    if c0.shape[0] == 0:
        return J
    for i in range(DIRECT_PARAM_DIM):
        step = eps * max(1.0, abs(float(q0[i])))
        qp = q0.copy()
        qm = q0.copy()
        qp[i] += step
        qm[i] -= step
        J[:, i] = (constraints(qp, face_mask=face_mask) - constraints(qm, face_mask=face_mask)) / (2.0 * step)
    return J


def constraint_jacobian(
    q: np.ndarray,
    eps: float = 1e-6,
    face_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
) -> np.ndarray:
    try:
        import torch

        from .geometry_torch import constraints_torch

        mask = normalize_face_mask(face_mask)
        if not bool(np.any(mask)):
            return np.zeros((0, DIRECT_PARAM_DIM), dtype=np.float64)
        q_t = torch.tensor(np.asarray(q, dtype=np.float64).reshape(DIRECT_PARAM_DIM), dtype=torch.double, requires_grad=True)
        c_t = constraints_torch(q_t, face_mask=mask)
        rows = []
        for i in range(int(c_t.numel())):
            grad_i = torch.autograd.grad(c_t[i], q_t, retain_graph=True)[0]
            rows.append(grad_i.detach().cpu().numpy())
        return np.stack(rows, axis=0).astype(np.float64)
    except Exception:
        return _constraint_jacobian_fd(q, eps=eps, face_mask=face_mask)


def project_tangent(
    q: np.ndarray,
    v: np.ndarray,
    *,
    face_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
    lambda_rel: float = 1e-10,
    lambda_abs: float = 1e-12,
) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64).reshape(DIRECT_PARAM_DIM)
    vec = np.asarray(v, dtype=np.float64).reshape(DIRECT_PARAM_DIM)
    J = constraint_jacobian(q_arr, face_mask=face_mask)
    if J.shape[0] == 0:
        return vec
    JJt = J @ J.T
    damping = float(lambda_rel) * float(np.trace(JJt)) / max(float(J.shape[0]), 1.0) + float(lambda_abs)
    correction = J.T @ np.linalg.solve(JJt + damping * np.eye(J.shape[0], dtype=np.float64), J @ vec)
    return vec - correction


def _constraint_norm_inf(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return 0.0 if arr.shape[0] == 0 else float(np.linalg.norm(arr, ord=np.inf))


def retract(
    q: np.ndarray,
    xi: np.ndarray,
    *,
    face_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
    tol: float = 1e-10,
    max_iter: int = 8,
    lambda_rel: float = 1e-10,
    lambda_abs: float = 1e-12,
) -> RetractionResult:
    z = np.asarray(q, dtype=np.float64).reshape(DIRECT_PARAM_DIM) + np.asarray(xi, dtype=np.float64).reshape(DIRECT_PARAM_DIM)
    if not np.all(np.isfinite(z)):
        return RetractionResult(q=z, ok=False, iterations=0, residual_inf=float("inf"))
    last_norm = _constraint_norm_inf(constraints(z, face_mask=face_mask))
    if last_norm < tol:
        return RetractionResult(q=z, ok=True, iterations=0, residual_inf=last_norm)
    for it in range(1, max_iter + 1):
        c = constraints(z, face_mask=face_mask)
        J = constraint_jacobian(z, face_mask=face_mask)
        if J.shape[0] == 0:
            return RetractionResult(q=z, ok=True, iterations=it, residual_inf=0.0)
        JJt = J @ J.T
        damping = float(lambda_rel) * float(np.trace(JJt)) / max(float(J.shape[0]), 1.0) + float(lambda_abs)
        delta = -J.T @ np.linalg.solve(JJt + damping * np.eye(J.shape[0], dtype=np.float64), c)
        beta = 1.0
        accepted = False
        for _ in range(10):
            trial = z + beta * delta
            norm = _constraint_norm_inf(constraints(trial, face_mask=face_mask))
            if np.all(np.isfinite(trial)) and norm < last_norm:
                z = trial
                last_norm = norm
                accepted = True
                break
            beta *= 0.5
        if not accepted:
            break
        if last_norm < tol:
            return RetractionResult(q=z, ok=True, iterations=it, residual_inf=last_norm)
    return RetractionResult(q=z, ok=last_norm < tol, iterations=max_iter, residual_inf=last_norm)


def max_vertex_displacement(q0: np.ndarray, q1: np.ndarray) -> float:
    return float(np.max(np.linalg.norm((vertices_from_q(q1) - vertices_from_q(q0)).T, axis=1)))


def trilinear_jacobian_det(vertices: np.ndarray, uvw: np.ndarray) -> float:
    V = np.asarray(vertices, dtype=np.float64).reshape(3, 8)
    u, v, w = [float(x) for x in np.asarray(uvw, dtype=np.float64).reshape(3)]
    dN_du = np.asarray(
        [
            -(1 - v) * (1 - w),
            (1 - v) * (1 - w),
            -v * (1 - w),
            -(1 - v) * w,
            v * (1 - w),
            (1 - v) * w,
            -v * w,
            v * w,
        ],
        dtype=np.float64,
    )
    dN_dv = np.asarray(
        [
            -(1 - u) * (1 - w),
            -u * (1 - w),
            (1 - u) * (1 - w),
            -(1 - u) * w,
            u * (1 - w),
            -u * w,
            (1 - u) * w,
            u * w,
        ],
        dtype=np.float64,
    )
    dN_dw = np.asarray(
        [
            -(1 - u) * (1 - v),
            -u * (1 - v),
            -(1 - u) * v,
            (1 - u) * (1 - v),
            -u * v,
            u * (1 - v),
            (1 - u) * v,
            u * v,
        ],
        dtype=np.float64,
    )
    J = np.column_stack([V @ dN_du, V @ dN_dv, V @ dN_dw])
    return float(np.linalg.det(J))


def _trilinear_sample(
    vertices: np.ndarray,
    u: float,
    v: float,
    w: float,
) -> tuple[np.ndarray, np.ndarray]:
    V = np.asarray(vertices, dtype=np.float64).reshape(3, 8)
    N = np.asarray(
        [
            (1 - u) * (1 - v) * (1 - w),
            u * (1 - v) * (1 - w),
            (1 - u) * v * (1 - w),
            (1 - u) * (1 - v) * w,
            u * v * (1 - w),
            u * (1 - v) * w,
            (1 - u) * v * w,
            u * v * w,
        ],
        dtype=np.float64,
    )
    dN_du = np.asarray(
        [
            -(1 - v) * (1 - w),
            (1 - v) * (1 - w),
            -v * (1 - w),
            -(1 - v) * w,
            v * (1 - w),
            (1 - v) * w,
            -v * w,
            v * w,
        ],
        dtype=np.float64,
    )
    dN_dv = np.asarray(
        [
            -(1 - u) * (1 - w),
            -u * (1 - w),
            (1 - u) * (1 - w),
            -(1 - u) * w,
            u * (1 - w),
            -u * w,
            (1 - u) * w,
            u * w,
        ],
        dtype=np.float64,
    )
    dN_dw = np.asarray(
        [
            -(1 - u) * (1 - v),
            -u * (1 - v),
            -(1 - u) * v,
            (1 - u) * (1 - v),
            -u * v,
            u * (1 - v),
            (1 - u) * v,
            u * v,
        ],
        dtype=np.float64,
    )
    point = V @ N
    jacobian = np.column_stack(
        [V @ dN_du, V @ dN_dv, V @ dN_dw]
    )
    return point, jacobian


def hexahedron_mass_properties(
    vertices: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return volume and centroidal diagonal inertia factors per unit mass.

    The three factors are E[y^2 + z^2], E[x^2 + z^2], and E[x^2 + y^2]
    about the volume centroid. Fixed Gauss quadrature makes this a smooth
    function of all eight vertices.
    """

    volume = 0.0
    first_moment = np.zeros(3, dtype=np.float64)
    second_moment = np.zeros(3, dtype=np.float64)
    for u, wu in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
        for v, wv in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
            for w, ww in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
                point, jacobian = _trilinear_sample(vertices, u, v, w)
                weighted_volume = float(wu * wv * ww) * float(
                    np.linalg.det(jacobian)
                )
                volume += weighted_volume
                first_moment += weighted_volume * point
                second_moment += weighted_volume * point * point
    if not np.isfinite(volume) or volume <= 1e-12:
        raise ValueError(
            f"invalid hexahedron volume for mass properties: {volume}"
        )
    centroid = first_moment / volume
    variance = np.maximum(
        second_moment / volume - centroid * centroid,
        1e-15,
    )
    inertia_factors = np.asarray(
        [
            variance[1] + variance[2],
            variance[0] + variance[2],
            variance[0] + variance[1],
        ],
        dtype=np.float64,
    )
    return float(volume), inertia_factors


def hexahedron_centroid(vertices: np.ndarray) -> np.ndarray:
    """Return the differentiable volume centroid of a trilinear hexahedron."""

    volume = 0.0
    first_moment = np.zeros(3, dtype=np.float64)
    for u, wu in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
        for v, wv in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
            for w, ww in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
                point, jacobian = _trilinear_sample(vertices, u, v, w)
                weighted_volume = float(wu * wv * ww) * float(
                    np.linalg.det(jacobian)
                )
                volume += weighted_volume
                first_moment += weighted_volume * point
    if not np.isfinite(volume) or volume <= 1e-12:
        raise ValueError(
            f"invalid hexahedron volume for centroid: {volume}"
        )
    return first_moment / volume


def hexahedron_surface_area(vertices: np.ndarray) -> float:
    """Integrate the six bilinear face areas without bounding-box extrema."""

    V = np.asarray(vertices, dtype=np.float64).reshape(3, 8)
    area = 0.0
    for corners in FACE_CORNERS.values():
        p00, p10, p11, p01 = [V[:, index] for index in corners]
        for s, ws in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
            for t, wt in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
                tangent_s = (
                    -(1 - t) * p00
                    + (1 - t) * p10
                    + t * p11
                    - t * p01
                )
                tangent_t = (
                    -(1 - s) * p00
                    - s * p10
                    + s * p11
                    + (1 - s) * p01
                )
                area += float(ws * wt) * float(
                    np.linalg.norm(np.cross(tangent_s, tangent_t))
                )
    if not np.isfinite(area) or area <= 1e-12:
        raise ValueError(
            f"invalid hexahedron surface area for contact scale: {area}"
        )
    return float(area)


def validate_hexahedron(
    q: np.ndarray,
    *,
    face_mask: np.ndarray | list[bool] | tuple[bool, ...] | None = None,
    constraint_tol: float = 1e-8,
    edge_min: float = 1e-5,
    area_min: float = 1e-8,
    jac_min: float = 1e-8,
) -> None:
    c = constraints(q, face_mask=face_mask)
    c_norm = _constraint_norm_inf(c)
    if c_norm > constraint_tol:
        raise ValueError(f"connected direct planar constraints violated: ||c||_inf={c_norm:.3e}")
    V = vertices_from_q(q)
    for i, j in EDGE_INDICES:
        if float(np.linalg.norm(V[:, i] - V[:, j])) <= edge_min:
            raise ValueError(f"collapsed direct planar edge {(i, j)}")
    for dock, corners in FACE_CORNERS.items():
        pts = V[:, list(corners)]
        area = float(np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])))
        if area <= area_min:
            raise ValueError(f"collapsed direct planar face {dock}: area={area:.3e}")
    samples = (
        (0.5, 0.5, 0.5),
        (0.1, 0.1, 0.1),
        (0.9, 0.1, 0.1),
        (0.1, 0.9, 0.1),
        (0.1, 0.1, 0.9),
        (0.9, 0.9, 0.9),
    )
    signs = [trilinear_jacobian_det(V, sample) for sample in samples]
    if min(signs) <= jac_min:
        raise ValueError(f"invalid direct planar trilinear orientation: min det={min(signs):.3e}")
