from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VERTEX_NAMES = ("O", "A", "B", "C", "D", "E", "F", "G")

# Vertex order: O, A, B, C, D, E, F, G.
FACE_CORNERS: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (2, -1): (0, 1, 4, 2),  # O, A, D, B
    (1, -1): (0, 1, 5, 3),  # O, A, E, C
    (0, -1): (0, 2, 6, 3),  # O, B, F, C
    (0, 1): (1, 4, 7, 5),   # A, D, G, E
    (1, 1): (2, 4, 7, 6),   # B, D, G, F
    (2, 1): (3, 5, 7, 6),   # C, E, G, F
}

TRILINEAR_VERTEX_SIGNS = np.asarray(
    [
        [0.0, 0.0, 0.0],  # O
        [1.0, 0.0, 0.0],  # A
        [0.0, 1.0, 0.0],  # B
        [0.0, 0.0, 1.0],  # C
        [1.0, 1.0, 0.0],  # D
        [1.0, 0.0, 1.0],  # E
        [0.0, 1.0, 1.0],  # F
        [1.0, 1.0, 1.0],  # G
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HexFrame:
    origin: np.ndarray
    extents: np.ndarray

    @property
    def safe_extents(self) -> np.ndarray:
        return np.maximum(self.extents, 1e-12)


def baseline_frame_from_vertices(vertices: np.ndarray) -> HexFrame:
    if vertices.shape[0] != 3:
        raise ValueError(f"Expected vertices with shape (3, n), got {vertices.shape}")
    if vertices.shape[1] == 0:
        origin = -0.5 * np.ones(3, dtype=np.float64)
        extents = np.ones(3, dtype=np.float64)
    else:
        mn = np.min(vertices, axis=1)
        mx = np.max(vertices, axis=1)
        origin = mn.astype(np.float64)
        extents = np.maximum(mx - mn, 1e-9).astype(np.float64)
    return HexFrame(origin=origin, extents=extents)


def trilinear_weights(uvw: np.ndarray) -> np.ndarray:
    uvw = np.asarray(uvw, dtype=np.float64)
    if uvw.size == 0:
        return np.zeros((0, 8), dtype=np.float64)
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    return np.stack(
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
        axis=1,
    )


def points_to_uvw(points: np.ndarray, frame: HexFrame, *, clip: bool = True) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    uvw = (np.asarray(points, dtype=np.float64) - frame.origin.reshape(1, 3)) / frame.safe_extents.reshape(1, 3)
    return np.clip(uvw, 0.0, 1.0) if clip else uvw


def map_uvw_to_vertices(uvw: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    if uvw.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return trilinear_weights(uvw) @ vertices.T


def face_point(vertices: np.ndarray, dock: tuple[int, int], barycentric: np.ndarray | None = None) -> np.ndarray:
    corners = FACE_CORNERS[dock]
    weights = np.ones(4, dtype=np.float64) * 0.25 if barycentric is None else np.asarray(barycentric, dtype=np.float64)
    weights = weights / (np.sum(weights) + 1e-12)
    return weights @ vertices[:, list(corners)].T


def face_normal(vertices: np.ndarray, dock: tuple[int, int]) -> np.ndarray:
    corners = FACE_CORNERS[dock]
    pts = vertices[:, list(corners)]
    n = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        axis, sign = dock
        out = np.zeros(3, dtype=np.float64)
        out[int(axis)] = float(sign)
        return out
    n = n / norm
    centroid = vertices.mean(axis=1)
    face_center = pts.mean(axis=1)
    return n if float(np.dot(n, face_center - centroid)) >= 0.0 else -n


def face_point_at_uv(vertices: np.ndarray, dock: tuple[int, int], uv: np.ndarray) -> np.ndarray:
    corners = FACE_CORNERS[dock]
    pts = vertices[:, list(corners)]
    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    p00, p10, p11, p01 = [pts[:, i] for i in range(4)]
    return (1.0 - u) * (1.0 - v) * p00 + u * (1.0 - v) * p10 + u * v * p11 + (1.0 - u) * v * p01


def face_frame(vertices: np.ndarray, dock: tuple[int, int]) -> np.ndarray:
    corners = FACE_CORNERS[dock]
    pts = vertices[:, list(corners)]
    n = face_normal(vertices, dock)
    u = pts[:, 1] - pts[:, 0]
    u = u - n * float(np.dot(n, u))
    norm_u = float(np.linalg.norm(u))
    if norm_u < 1e-12:
        axis, _ = dock
        fallback = np.zeros(3, dtype=np.float64)
        fallback[(int(axis) + 1) % 3] = 1.0
        u = fallback - n * float(np.dot(n, fallback))
        norm_u = float(np.linalg.norm(u))
    u = u / max(norm_u, 1e-12)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return np.column_stack([u, v, n])

FACE_ID_TO_DOCK: dict[int, tuple[int, int]] = {
    0: (2, 1),
    1: (0, 1),
    2: (1, 1),
    3: (0, -1),
    4: (2, -1),
    5: (1, -1),
}


def _dock_from_face_id(face_id: int | None) -> tuple[int, int] | None:
    if face_id is None:
        return None
    return FACE_ID_TO_DOCK.get(int(face_id))
