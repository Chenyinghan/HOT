from __future__ import annotations

import torch

from bilevel.parameterization.interpolation import FACE_CORNERS

from .geometry import (
    DIRECT_PARAM_DIM,
    FACE_CONSTRAINT_TETS,
    GAUSS_POINTS,
    GAUSS_WEIGHTS,
    normalize_face_mask,
)


def vertices_from_q_torch(q: torch.Tensor) -> torch.Tensor:
    p = q.reshape(DIRECT_PARAM_DIM).to(dtype=torch.double)
    zero = torch.zeros((), dtype=torch.double, device=p.device)
    O = torch.stack([zero, zero, zero])
    A = torch.stack([p[0], zero, zero])
    B = torch.stack([p[1], p[2], zero])
    C = p[3:6]
    D = p[6:9]
    E = p[9:12]
    F = p[12:15]
    G = p[15:18]
    return torch.stack([O, A, B, C, D, E, F, G], dim=1)


def constraints_torch(q: torch.Tensor, face_mask=None) -> torch.Tensor:
    V = vertices_from_q_torch(q)
    mask = normalize_face_mask(face_mask)

    def det4(i: int, j: int, k: int, l: int) -> torch.Tensor:
        P = V[:, i]
        return torch.dot(V[:, j] - P, torch.cross(V[:, k] - P, V[:, l] - P, dim=0))

    values = [det4(*tet) for idx, tet in enumerate(FACE_CONSTRAINT_TETS) if bool(mask[idx])]
    if not values:
        return torch.zeros((0,), dtype=torch.double, device=q.device)
    return torch.stack(values)


def _trilinear_sample_torch(
    vertices: torch.Tensor,
    u: float,
    v: float,
    w: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    V = vertices.reshape(3, 8).to(dtype=torch.double)
    N = torch.tensor(
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
        dtype=torch.double,
        device=V.device,
    )
    dN_du = torch.tensor(
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
        dtype=torch.double,
        device=V.device,
    )
    dN_dv = torch.tensor(
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
        dtype=torch.double,
        device=V.device,
    )
    dN_dw = torch.tensor(
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
        dtype=torch.double,
        device=V.device,
    )
    point = V @ N
    jacobian = torch.stack(
        [V @ dN_du, V @ dN_dv, V @ dN_dw],
        dim=1,
    )
    return point, jacobian


def hexahedron_mass_properties_torch(
    vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    V = vertices.reshape(3, 8).to(dtype=torch.double)
    volume = torch.zeros((), dtype=torch.double, device=V.device)
    first_moment = torch.zeros(3, dtype=torch.double, device=V.device)
    second_moment = torch.zeros(3, dtype=torch.double, device=V.device)
    for u, wu in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
        for v, wv in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
            for w, ww in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
                point, jacobian = _trilinear_sample_torch(V, u, v, w)
                weighted_volume = float(wu * wv * ww) * torch.det(jacobian)
                volume = volume + weighted_volume
                first_moment = first_moment + weighted_volume * point
                second_moment = (
                    second_moment + weighted_volume * point * point
                )
    centroid = first_moment / torch.clamp(volume, min=1e-12)
    variance = torch.clamp(
        second_moment / torch.clamp(volume, min=1e-12)
        - centroid * centroid,
        min=1e-15,
    )
    inertia_factors = torch.stack(
        [
            variance[1] + variance[2],
            variance[0] + variance[2],
            variance[0] + variance[1],
        ]
    )
    return volume, inertia_factors


def hexahedron_centroid_torch(vertices: torch.Tensor) -> torch.Tensor:
    """Return the differentiable volume centroid of a trilinear hexahedron."""

    V = vertices.reshape(3, 8).to(dtype=torch.double)
    volume = torch.zeros((), dtype=torch.double, device=V.device)
    first_moment = torch.zeros(
        3,
        dtype=torch.double,
        device=V.device,
    )
    for u, wu in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
        for v, wv in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
            for w, ww in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
                point, jacobian = _trilinear_sample_torch(V, u, v, w)
                weighted_volume = (
                    float(wu * wv * ww) * torch.det(jacobian)
                )
                volume = volume + weighted_volume
                first_moment = (
                    first_moment + weighted_volume * point
                )
    return first_moment / torch.clamp(volume, min=1e-12)


def hexahedron_surface_area_torch(vertices: torch.Tensor) -> torch.Tensor:
    V = vertices.reshape(3, 8).to(dtype=torch.double)
    area = torch.zeros((), dtype=torch.double, device=V.device)
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
                area = area + float(ws * wt) * torch.linalg.norm(
                    torch.cross(tangent_s, tangent_t, dim=0)
                )
    return area
