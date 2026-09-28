from __future__ import annotations

import torch

from .interpolation import FACE_CORNERS


def trilinear_weights_torch(uvw: torch.Tensor) -> torch.Tensor:
    if uvw.numel() == 0:
        return torch.zeros((0, 8), dtype=torch.double, device=uvw.device)
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    one = torch.ones_like(u)
    return torch.stack(
        [
            (one - u) * (one - v) * (one - w),
            u * (one - v) * (one - w),
            (one - u) * v * (one - w),
            (one - u) * (one - v) * w,
            u * v * (one - w),
            u * (one - v) * w,
            (one - u) * v * w,
            u * v * w,
        ],
        dim=1,
    )


def map_uvw_to_vertices_torch(uvw: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    if uvw.numel() == 0:
        return torch.zeros((0, 3), dtype=torch.double, device=vertices.device)
    return trilinear_weights_torch(uvw.to(dtype=torch.double, device=vertices.device)) @ vertices.T


def face_point_torch(
    vertices: torch.Tensor,
    dock: tuple[int, int],
    barycentric: torch.Tensor | None = None,
) -> torch.Tensor:
    corners = list(FACE_CORNERS[dock])
    if barycentric is None:
        weights = torch.ones(4, dtype=torch.double, device=vertices.device) * 0.25
    else:
        weights = barycentric.to(dtype=torch.double, device=vertices.device).reshape(4)
        weights = weights / torch.clamp(torch.sum(weights), min=1e-12)
    return weights @ vertices[:, corners].T


def face_normal_torch(vertices: torch.Tensor, dock: tuple[int, int]) -> torch.Tensor:
    corners = list(FACE_CORNERS[dock])
    pts = vertices[:, corners]
    n = torch.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0], dim=0)
    norm = torch.norm(n)
    axis, sign = dock
    ref = torch.zeros(3, dtype=torch.double, device=vertices.device)
    ref[int(axis)] = float(sign)
    if bool((norm.detach().cpu() < 1e-12)):
        return ref
    n = n / torch.clamp(norm, min=1e-12)
    centroid = torch.mean(vertices, dim=1)
    face_center = torch.mean(pts, dim=1)
    return n if bool((torch.dot(n, face_center - centroid).detach().cpu() >= 0.0)) else -n


def face_point_at_uv_torch(vertices: torch.Tensor, dock: tuple[int, int], uv) -> torch.Tensor:
    corners = list(FACE_CORNERS[dock])
    pts = vertices[:, corners]
    u = float(max(0.0, min(1.0, float(uv[0]))))
    v = float(max(0.0, min(1.0, float(uv[1]))))
    p00, p10, p11, p01 = [pts[:, i] for i in range(4)]
    return (1.0 - u) * (1.0 - v) * p00 + u * (1.0 - v) * p10 + u * v * p11 + (1.0 - u) * v * p01


def face_frame_torch(vertices: torch.Tensor, dock: tuple[int, int]) -> torch.Tensor:
    corners = list(FACE_CORNERS[dock])
    pts = vertices[:, corners]
    n = face_normal_torch(vertices, dock)
    u = pts[:, 1] - pts[:, 0]
    u = u - n * torch.dot(n, u)
    norm_u = torch.norm(u)
    if bool((norm_u.detach().cpu() < 1e-12)):
        axis, _ = dock
        fallback = torch.zeros(3, dtype=torch.double, device=vertices.device)
        fallback[(int(axis) + 1) % 3] = 1.0
        u = fallback - n * torch.dot(n, fallback)
        norm_u = torch.norm(u)
    u = u / torch.clamp(norm_u, min=1e-12)
    v = torch.cross(n, u, dim=0)
    v = v / torch.clamp(torch.norm(v), min=1e-12)
    return torch.stack([u, v, n], dim=1)
