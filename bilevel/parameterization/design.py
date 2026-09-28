from __future__ import annotations

from typing import Any

import numpy as np
import torch

from bilevel.upper.bass.connection_geometry import EDGE_CORNER_CONNECTION
from bilevel.parameterization.bundle import DesignBundle

from bilevel.parameterization.scene import (
    SceneSpec,
    LinkRecord,
    _Mesh,
    _flatten_E_np,
    _is_deformable_tool_record,
    _read_contact_ids,
    _read_mesh,
    _read_points,
)
from bilevel.parameterization.interpolation import (
    FACE_CORNERS,
    _dock_from_face_id,
    baseline_frame_from_vertices,
    face_frame,
    face_normal,
    face_point,
    face_point_at_uv,
    points_to_uvw,
    trilinear_weights,
)
from bilevel.parameterization.interpolation_torch import (
    face_frame_torch,
    face_normal_torch,
    face_point_torch,
    face_point_at_uv_torch,
    trilinear_weights_torch,
)

from .geometry import (
    DIRECT_PARAM_DIM,
    DOCK_TO_FACE_INDEX,
    FACE_ORDER,
    GENERIC_DESIGN_PROTOCOL,
    baseline_q_from_extents,
    hexahedron_centroid,
    hexahedron_mass_properties,
    hexahedron_surface_area,
    normalize_face_mask,
    reference_length,
    validate_hexahedron,
    vertices_from_q,
)
from .geometry_torch import (
    hexahedron_centroid_torch,
    hexahedron_mass_properties_torch,
    hexahedron_surface_area_torch,
    vertices_from_q_torch,
)
from .topology import (
    HeadConnection,
    HeadTopology,
    HeadTopologyBlock,
)


_OPPOSITE_FACE_ID = {0: 4, 1: 3, 2: 5, 3: 1, 4: 0, 5: 2}
_AXIS_DOCKS = {
    0: ((0, -1), (0, 1)),
    1: ((1, -1), (1, 1)),
    2: ((2, -1), (2, 1)),
}


def _edge_axis_metadata(rec: LinkRecord, prefix: str) -> tuple[int, int]:
    attrs = getattr(rec, "attrs", {})
    return (
        int(attrs[f"planar_{prefix}_edge_tangent_axis"]),
        int(attrs[f"planar_{prefix}_panel_normal_axis"]),
    )


def _normalize_np(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _edge_frame_np(
    rec: LinkRecord,
    cage: "DirectPlanarToolHexNP",
    *,
    face: int,
    tangent_axis: int,
    normal_axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    neg_normal, pos_normal = _AXIS_DOCKS[int(normal_axis)]
    neg_tangent, pos_tangent = _AXIS_DOCKS[int(tangent_axis)]
    normal_delta = cage.dock_point(pos_normal) - cage.dock_point(neg_normal)
    tangent_delta = cage.dock_point(pos_tangent) - cage.dock_point(neg_tangent)
    normal = _normalize_np(normal_delta)
    tangent = tangent_delta - normal * float(np.dot(normal, tangent_delta))
    tangent = _normalize_np(tangent)
    outward = cage.dock_normal(_dock_from_face_id(face))
    semantic_outward = _normalize_np(np.cross(normal, tangent))
    if float(np.dot(semantic_outward, outward)) < 0.0:
        tangent = -tangent
        semantic_outward = -semantic_outward
    body_R = rec.body_E[:3, :3]
    return (
        body_R @ tangent,
        body_R @ semantic_outward,
        body_R @ normal,
        0.5 * float(np.linalg.norm(normal_delta)),
    )


def _normalize_torch(value: torch.Tensor) -> torch.Tensor:
    return value / torch.clamp(torch.linalg.norm(value), min=1e-12)


def _edge_frame_torch(
    rec: LinkRecord,
    cage: "DirectPlanarToolHexTorch",
    *,
    face: int,
    tangent_axis: int,
    normal_axis: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    neg_normal, pos_normal = _AXIS_DOCKS[int(normal_axis)]
    neg_tangent, pos_tangent = _AXIS_DOCKS[int(tangent_axis)]
    normal_delta = cage.dock_point(pos_normal) - cage.dock_point(neg_normal)
    tangent_delta = cage.dock_point(pos_tangent) - cage.dock_point(neg_tangent)
    normal = _normalize_torch(normal_delta)
    tangent = tangent_delta - normal * torch.dot(normal, tangent_delta)
    tangent = _normalize_torch(tangent)
    outward = cage.dock_normal(_dock_from_face_id(face))
    semantic_outward = _normalize_torch(torch.cross(normal, tangent, dim=0))
    if bool((torch.dot(semantic_outward, outward).detach().cpu() < 0.0)):
        tangent = -tangent
        semantic_outward = -semantic_outward
    body_R = torch.as_tensor(
        rec.body_E[:3, :3],
        dtype=torch.double,
        device=cage.vertices.device,
    )
    return (
        body_R @ tangent,
        body_R @ semantic_outward,
        body_R @ normal,
        0.5 * torch.linalg.norm(normal_delta),
    )


def _terminal_dock(rec: LinkRecord) -> tuple[int, int] | None:
    face = getattr(rec, "planar_endeffector_face", None)
    if face is None:
        child_face = getattr(rec, "planar_child_face", None)
        face = None if child_face is None else _OPPOSITE_FACE_ID.get(int(child_face))
    return _dock_from_face_id(face)


def _marker_leaf_records(spec: DirectPlanarHexSpec, marker: LinkRecord) -> list[LinkRecord]:
    node_to_tool = {
        int(rec.node_id): rec
        for rec in spec.tool_records
        if rec.node_id is not None
    }
    return [
        node_to_tool[node_id]
        for node_id in marker.function_group_leaves
        if node_id in node_to_tool
    ]


def _transform_from_flat_np(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(12)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = values[:9].reshape(3, 3)
    transform[:3, 3] = values[9:12]
    return transform


def _assemble_torch_overrides(
    baseline: torch.Tensor,
    overrides: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    if not overrides:
        return baseline
    parts = []
    cursor = 0
    for (start, stop), value in sorted(overrides.items()):
        if start < cursor:
            raise RuntimeError("overlapping design parameter override slices")
        if start > cursor:
            parts.append(baseline[cursor:start])
        parts.append(value)
        cursor = stop
    if cursor < baseline.numel():
        parts.append(baseline[cursor:])
    return torch.cat(parts)


class DirectPlanarHexSpec(SceneSpec):
    def ndof_cage(self) -> int:
        return DIRECT_PARAM_DIM * len(self.tool_records)

    @property
    def init_cage_params(self) -> np.ndarray:
        out = np.zeros(self.ndof_cage(), dtype=np.float64)
        cursor = 0
        for rec in self.tool_records:
            cage = DirectPlanarToolHexNP(rec)
            out[cursor : cursor + DIRECT_PARAM_DIM] = cage.params0
            cursor += DIRECT_PARAM_DIM
        return out


class DirectPlanarToolHexNP:
    def __init__(self, rec: LinkRecord):
        self.rec = rec
        self.face_mask = np.ones(len(FACE_ORDER), dtype=bool)
        self.V0, self.F = _read_mesh(rec.mesh_path)
        self.frame = baseline_frame_from_vertices(self.V0)
        self.ref_length = reference_length(self.frame.extents)
        self.params0 = baseline_q_from_extents(self.frame.extents, ref_length=self.ref_length)
        self.params = self.params0.copy()
        self.base_vertices = self._vertices_from_params(self.params0)
        self.vertices = self.base_vertices.copy()
        (
            self.base_volume,
            self.base_inertia_factors,
        ) = hexahedron_mass_properties(self.base_vertices)
        self.base_surface_area = hexahedron_surface_area(self.base_vertices)
        self.vertex_uvw = points_to_uvw(self.V0.T, self.frame) if self.V0.shape[1] else np.zeros((0, 3), dtype=np.float64)
        self.vertex_weights = trilinear_weights(self.vertex_uvw)
        self.contact_points = _read_points(rec.contact_path).T
        self.contact_ids = _read_contact_ids(rec.contact_path)
        if self.contact_points.shape[0] > 0 and self.contact_ids.shape[0] != self.contact_points.shape[0]:
            raise ValueError(
                f"Stale contact ids for {rec.contact_path}: got {self.contact_ids.shape[0]} ids for "
                f"{self.contact_points.shape[0]} contact points. Regenerate contacts and contact_id.npy together."
            )
        self.contact_uvw = points_to_uvw(self.contact_points, self.frame)
        self.contact_weights = trilinear_weights(self.contact_uvw)

    def reset(self) -> None:
        self.params = self.params0.copy()
        self.vertices = self.base_vertices.copy()

    def apply_params(self, params: np.ndarray) -> None:
        raw = np.asarray(params, dtype=np.float64).reshape(DIRECT_PARAM_DIM)
        if _is_deformable_tool_record(self.rec):
            validate_hexahedron(raw, face_mask=self.face_mask)
            self.params = raw
        else:
            self.params = self.params0.copy()
        self.vertices = self._vertices_from_params(self.params)

    def set_face_mask(self, face_mask: np.ndarray | list[bool] | tuple[bool, ...]) -> None:
        self.face_mask = normalize_face_mask(face_mask)

    def _vertices_from_params(self, params: np.ndarray) -> np.ndarray:
        return vertices_from_q(params) * self.ref_length + self.frame.origin.reshape(3, 1)

    def dock_from_vector(self, v: np.ndarray) -> tuple[int, int] | None:
        if np.linalg.norm(v) < 1e-12:
            return None
        axis = int(np.argmax(np.abs(v)))
        return axis, 1 if v[axis] >= 0.0 else -1

    def face_points(self, dock: tuple[int, int], *, initial: bool = False) -> np.ndarray:
        vertices = self.base_vertices if initial else self.vertices
        return vertices[:, list(FACE_CORNERS[dock])]

    def dock_point(
        self,
        dock: tuple[int, int],
        *,
        initial: bool = False,
        barycentric: np.ndarray | None = None,
    ) -> np.ndarray:
        vertices = self.base_vertices if initial else self.vertices
        return face_point(vertices, dock, barycentric)

    def dock_normal(self, dock: tuple[int, int]) -> np.ndarray:
        return face_normal(self.vertices, dock)

    def _face_point_at_uv(self, dock: tuple[int, int], uv: np.ndarray) -> np.ndarray:
        return face_point_at_uv(self.vertices, dock, uv)

    def bind_face_to_parent(self, *args, **kwargs) -> None:
        _ = (args, kwargs)

    def mesh_vertices(self) -> np.ndarray:
        if self.vertex_weights.shape[0] == 0:
            return self.V0
        return (self.vertex_weights @ self.vertices.T).T

    def contacts(self) -> np.ndarray:
        if self.contact_weights.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return self.contact_weights @ self.vertices.T

    def centroid(self) -> np.ndarray:
        return hexahedron_centroid(self.vertices)

    def extents(self, *, initial: bool = False) -> np.ndarray:
        vertices = self.base_vertices if initial else self.vertices
        return np.maximum(vertices.max(axis=1) - vertices.min(axis=1), 1e-9)

    def inertia(self, baseline_p4: np.ndarray | None = None) -> np.ndarray:
        volume, inertia_factors = hexahedron_mass_properties(self.vertices)
        if baseline_p4 is not None:
            base = np.asarray(baseline_p4, dtype=np.float64).reshape(4)
            volume_ratio = float(volume / self.base_volume)
            axis_scale = inertia_factors / np.maximum(
                self.base_inertia_factors,
                1e-15,
            )
            return np.concatenate([[base[0] * volume_ratio], base[1:] * volume_ratio * axis_scale])
        mass = float(volume)
        return np.asarray(
            [
                mass,
                mass * inertia_factors[0],
                mass * inertia_factors[1],
                mass * inertia_factors[2],
            ],
            dtype=np.float64,
        )

    def contact_scale(self, baseline_scale: float | None = None) -> float:
        scale = float(
            hexahedron_surface_area(self.vertices) / self.base_surface_area
        )
        return scale if baseline_scale is None else float(baseline_scale) * scale


class DirectPlanarToolHexTorch:
    def __init__(self, rec: LinkRecord):
        self.rec = rec
        self.face_mask = np.ones(len(FACE_ORDER), dtype=bool)
        V0, _ = _read_mesh(rec.mesh_path)
        self.V0 = torch.tensor(V0, dtype=torch.double)
        self.frame = baseline_frame_from_vertices(V0)
        self.origin = torch.tensor(self.frame.origin, dtype=torch.double)
        self.ref_length = float(reference_length(self.frame.extents))
        self.params0 = torch.tensor(
            baseline_q_from_extents(self.frame.extents, ref_length=self.ref_length),
            dtype=torch.double,
        )
        self.params = self.params0.clone()
        self.base_vertices = self._vertices_from_params(self.params0)
        self.vertices = self.base_vertices.clone()
        (
            self.base_volume,
            self.base_inertia_factors,
        ) = hexahedron_mass_properties_torch(self.base_vertices)
        self.base_surface_area = hexahedron_surface_area_torch(
            self.base_vertices
        )
        uvw = points_to_uvw(V0.T, self.frame) if V0.shape[1] else np.zeros((0, 3), dtype=np.float64)
        self.vertex_uvw = torch.tensor(uvw, dtype=torch.double)
        self.vertex_weights = trilinear_weights_torch(self.vertex_uvw)
        contact_points = _read_points(rec.contact_path).T
        contact_ids = _read_contact_ids(rec.contact_path)
        if contact_points.shape[0] > 0 and contact_ids.shape[0] != contact_points.shape[0]:
            raise ValueError(
                f"Stale contact ids for {rec.contact_path}: got {contact_ids.shape[0]} ids for "
                f"{contact_points.shape[0]} contact points. Regenerate contacts and contact_id.npy together."
            )
        self.contact_points = torch.tensor(contact_points, dtype=torch.double)
        self.contact_uvw = torch.tensor(points_to_uvw(contact_points, self.frame), dtype=torch.double)
        self.contact_weights = trilinear_weights_torch(self.contact_uvw)

    def reset(self) -> None:
        self.params = self.params0.clone()
        self.vertices = self.base_vertices.clone()

    def apply_params(self, params: torch.Tensor) -> None:
        if _is_deformable_tool_record(self.rec):
            self.params = params.reshape(DIRECT_PARAM_DIM).to(dtype=torch.double)
        else:
            self.params = self.params0.to(dtype=torch.double, device=params.device)
        self.vertices = self._vertices_from_params(self.params)

    def set_face_mask(self, face_mask: np.ndarray | list[bool] | tuple[bool, ...]) -> None:
        self.face_mask = normalize_face_mask(face_mask)

    def _vertices_from_params(self, params: torch.Tensor) -> torch.Tensor:
        return vertices_from_q_torch(params) * self.ref_length + self.origin.to(device=params.device).reshape(3, 1)

    def dock_from_vector(self, v: torch.Tensor) -> tuple[int, int] | None:
        v_np = v.detach().cpu().numpy()
        if np.linalg.norm(v_np) < 1e-12:
            return None
        axis = int(np.argmax(np.abs(v_np)))
        return axis, 1 if v_np[axis] >= 0.0 else -1

    def face_points(self, dock: tuple[int, int], *, initial: bool = False) -> torch.Tensor:
        vertices = self.base_vertices if initial else self.vertices
        return vertices[:, list(FACE_CORNERS[dock])]

    def dock_point(
        self,
        dock: tuple[int, int],
        *,
        initial: bool = False,
        barycentric: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        vertices = self.base_vertices if initial else self.vertices
        if barycentric is not None and not isinstance(barycentric, torch.Tensor):
            barycentric = torch.tensor(barycentric, dtype=torch.double, device=vertices.device)
        return face_point_torch(vertices, dock, barycentric)

    def dock_normal(self, dock: tuple[int, int]) -> torch.Tensor:
        return face_normal_torch(self.vertices, dock)

    def _face_point_at_uv(self, dock: tuple[int, int], uv: np.ndarray) -> torch.Tensor:
        return face_point_at_uv_torch(self.vertices, dock, uv)

    def bind_face_to_parent(self, *args, **kwargs) -> None:
        _ = (args, kwargs)

    def mesh_vertices(self) -> torch.Tensor:
        if self.vertex_weights.numel() == 0:
            return self.V0
        return (self.vertex_weights.to(dtype=torch.double, device=self.vertices.device) @ self.vertices.T).T

    def contacts(self) -> torch.Tensor:
        if self.contact_weights.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.double, device=self.vertices.device)
        return self.contact_weights.to(dtype=torch.double, device=self.vertices.device) @ self.vertices.T

    def centroid(self) -> torch.Tensor:
        return hexahedron_centroid_torch(self.vertices)

    def extents(self, *, initial: bool = False) -> torch.Tensor:
        vertices = self.base_vertices if initial else self.vertices
        return torch.clamp(torch.max(vertices, dim=1).values - torch.min(vertices, dim=1).values, min=1e-9)

    def inertia(self, baseline_p4: torch.Tensor | np.ndarray | None = None) -> torch.Tensor:
        volume, inertia_factors = hexahedron_mass_properties_torch(
            self.vertices
        )
        if baseline_p4 is not None:
            base = torch.as_tensor(
                baseline_p4,
                dtype=torch.double,
                device=volume.device,
            ).reshape(4)
            base_volume = self.base_volume.to(
                dtype=torch.double,
                device=volume.device,
            )
            base_factors = self.base_inertia_factors.to(
                dtype=torch.double,
                device=volume.device,
            )
            volume_ratio = volume / torch.clamp(base_volume, min=1e-12)
            axis_scale = inertia_factors / torch.clamp(
                base_factors,
                min=1e-15,
            )
            return torch.cat([base[:1] * volume_ratio, base[1:] * volume_ratio * axis_scale])
        mass = volume
        return torch.stack(
            [
                mass,
                mass * inertia_factors[0],
                mass * inertia_factors[1],
                mass * inertia_factors[2],
            ]
        )

    def contact_scale(self, baseline_scale: torch.Tensor | np.ndarray | float | None = None) -> torch.Tensor:
        area = hexahedron_surface_area_torch(self.vertices)
        base_area = self.base_surface_area.to(
            dtype=torch.double,
            device=area.device,
        )
        scale = area / torch.clamp(base_area, min=1e-12)
        if baseline_scale is None:
            return scale
        base = torch.as_tensor(baseline_scale, dtype=torch.double, device=scale.device).reshape(())
        return base * scale


def _infer_connection_docks_for_topology(
    spec: DirectPlanarHexSpec,
    tool_cages: list[DirectPlanarToolHexNP],
    tool_index: dict[int, int],
    child_index: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rec = spec.tool_records[child_index]
    parent = spec.records[rec.parent] if rec.parent is not None else None
    if parent is None or id(parent) not in tool_index:
        return None
    cage = tool_cages[child_index]
    parent_cage = tool_cages[tool_index[id(parent)]]
    rotation = rec.joint_E[:3, :3]
    translation = rec.joint_E[:3, 3]
    scale = max(
        float(np.linalg.norm(parent_cage.extents(initial=True))),
        float(np.linalg.norm(cage.extents(initial=True))),
        1.0,
    )
    best = None
    for parent_dock in FACE_CORNERS:
        parent_anchor = (
            parent.body_E[:3, :3]
            @ parent_cage.dock_point(parent_dock, initial=True)
            + parent.body_E[:3, 3]
        )
        for child_dock in FACE_CORNERS:
            child_anchor = (
                rec.body_E[:3, :3]
                @ cage.dock_point(child_dock, initial=True)
                + rec.body_E[:3, 3]
            )
            residual = float(
                np.linalg.norm(
                    parent_anchor
                    - (translation + rotation @ child_anchor)
                )
            )
            if best is None or residual < best[0]:
                best = (residual, parent_dock, child_dock)
    tolerance = max(1e-4, 1e-4 * scale)
    if best is None or best[0] > tolerance:
        return None
    return best[1], best[2]


def build_connected_head_topology(
    spec: DirectPlanarHexSpec,
    *,
    tool_cages: list[DirectPlanarToolHexNP] | None = None,
) -> HeadTopology:
    """Resolve all static Head connectivity once using NumPy baselines."""

    cages = (
        list(tool_cages)
        if tool_cages is not None
        else [DirectPlanarToolHexNP(rec) for rec in spec.tool_records]
    )
    if len(cages) != len(spec.tool_records):
        raise ValueError(
            "Head topology cage/record count mismatch: "
            f"{len(cages)} != {len(spec.tool_records)}"
        )
    tool_index = {
        id(record): index
        for index, record in enumerate(spec.tool_records)
    }
    face_masks = [
        np.zeros(len(FACE_ORDER), dtype=bool)
        for _ in spec.tool_records
    ]
    resolved = []
    for child_index, record in enumerate(spec.tool_records):
        parent = (
            spec.records[record.parent]
            if record.parent is not None
            else None
        )
        parent_tool_index = (
            tool_index[id(parent)]
            if parent is not None and id(parent) in tool_index
            else None
        )
        child_dock = _dock_from_face_id(record.planar_child_face)
        parent_dock = _dock_from_face_id(record.planar_parent_face)
        explicit = (
            child_dock is not None
            and record.planar_parent_face is not None
        )
        inferred = False
        if (
            parent_tool_index is not None
            and (child_dock is None or parent_dock is None)
        ):
            inferred_docks = _infer_connection_docks_for_topology(
                spec,
                cages,
                tool_index,
                child_index,
            )
            if inferred_docks is not None:
                parent_dock, child_dock = inferred_docks
                explicit = False
                inferred = True
        if parent is not None and child_dock is not None:
            face_index = DOCK_TO_FACE_INDEX.get(child_dock)
            if face_index is not None:
                face_masks[child_index][face_index] = True
        if parent_tool_index is not None and parent_dock is not None:
            face_index = DOCK_TO_FACE_INDEX.get(parent_dock)
            if face_index is not None:
                face_masks[parent_tool_index][face_index] = True
        resolved.append(
            (
                parent_tool_index,
                parent_dock,
                child_dock,
                explicit,
                inferred,
            )
        )
    for marker in spec.marker_records:
        parent = (
            spec.records[marker.parent]
            if marker.parent is not None
            else None
        )
        if parent is None or id(parent) not in tool_index:
            continue
        dock = _dock_from_face_id(marker.planar_parent_face)
        face_index = DOCK_TO_FACE_INDEX.get(dock)
        if face_index is not None:
            face_masks[tool_index[id(parent)]][face_index] = True

    occupied: list[set[tuple[int, int]]] = [
        set() for _ in spec.tool_records
    ]
    connections = []
    for child_index, record in enumerate(spec.tool_records):
        parent = (
            spec.records[record.parent]
            if record.parent is not None
            else None
        )
        translation = record.joint_E[:3, 3]
        rotation = record.joint_E[:3, :3]
        edge_corner = (
            str(getattr(record, "attrs", {}).get("planar_connection_family", ""))
            == EDGE_CORNER_CONNECTION
        )
        resolved_parent_index, resolved_parent_dock, resolved_child_dock, _, _ = resolved[
            child_index
        ]
        child_dock = (
            resolved_child_dock
            if edge_corner
            else cages[child_index].dock_from_vector(rotation.T @ (-translation))
        )
        if child_dock is not None:
            occupied[child_index].add(child_dock)
        if parent is None or id(parent) not in tool_index:
            continue
        parent_index = tool_index[id(parent)]
        parent_dock = (
            resolved_parent_dock
            if edge_corner and resolved_parent_index == parent_index
            else cages[parent_index].dock_from_vector(translation)
        )
        if parent_dock is not None:
            occupied[parent_index].add(parent_dock)
        if parent_dock is None or child_dock is None:
            continue
        connections.append(
            HeadConnection(
                parent_index=parent_index,
                child_index=child_index,
                parent_dock=parent_dock,
                child_dock=child_dock,
                rotation=tuple(
                    float(value) for value in rotation.reshape(-1)
                ),
            )
        )

    blocks = []
    for index, record in enumerate(spec.tool_records):
        (
            parent_tool_index,
            parent_dock,
            child_dock,
            explicit,
            inferred,
        ) = resolved[index]
        attrs = getattr(record, "attrs", {})
        blocks.append(
            HeadTopologyBlock(
                tool_index=index,
                node_id=(
                    None
                    if record.node_id is None
                    else int(record.node_id)
                ),
                connected_face_mask=tuple(
                    bool(value) for value in face_masks[index]
                ),
                occupied_docks=tuple(sorted(occupied[index])),
                parent_tool_index=parent_tool_index,
                parent_dock=parent_dock,
                child_dock=child_dock,
                explicit_connection=explicit,
                inferred_connection=inferred,
                direct_handle_mount=(
                    str(attrs.get("welded_interface", "")).lower()
                    == "true"
                    and str(attrs.get("welded_parent_kind", ""))
                    == "handle"
                ),
                parent_face=record.planar_parent_face,
                child_face=record.planar_child_face,
            )
        )
    return HeadTopology(
        blocks=tuple(blocks),
        connections=tuple(connections),
    )


def _connections_numpy(topology: HeadTopology) -> list[dict[str, Any]]:
    return [
        {
            "parent": connection.parent_index,
            "child": connection.child_index,
            "parent_dock": connection.parent_dock,
            "child_dock": connection.child_dock,
            "rotation": np.asarray(
                connection.rotation,
                dtype=np.float64,
            ).reshape(3, 3),
        }
        for connection in topology.connections
    ]


def _connections_torch(topology: HeadTopology) -> list[dict[str, Any]]:
    return [
        {
            "parent": connection.parent_index,
            "child": connection.child_index,
            "parent_dock": connection.parent_dock,
            "child_dock": connection.child_dock,
            "rotation": torch.tensor(
                connection.rotation,
                dtype=torch.double,
            ).reshape(3, 3),
        }
        for connection in topology.connections
    ]


class DirectPlanarHexDesignNP:
    def _build_occupied_docks(self) -> list[set[tuple[int, int]]]:
        occupied = [set() for _ in self.spec.tool_records]
        for i, rec in enumerate(self.spec.tool_records):
            parent = self.spec.records[rec.parent] if rec.parent is not None else None
            t = rec.joint_E[:3, 3]
            child_local = rec.joint_E[:3, :3].T @ (-t)
            dock = self.tool_cages[i].dock_from_vector(child_local)
            if dock is not None:
                occupied[i].add(dock)
            if parent is not None and id(parent) in self.tool_index:
                pidx = self.tool_index[id(parent)]
                pdock = self.tool_cages[pidx].dock_from_vector(t)
                if pdock is not None:
                    occupied[pidx].add(pdock)
        return occupied


    def _build_connections(self) -> list[dict[str, Any]]:
        connections: list[dict[str, Any]] = []
        for i, rec in enumerate(self.spec.tool_records):
            parent = self.spec.records[rec.parent] if rec.parent is not None else None
            if parent is None or id(parent) not in self.tool_index:
                continue
            pidx = self.tool_index[id(parent)]
            R = rec.joint_E[:3, :3]
            t = rec.joint_E[:3, 3]
            parent_dock = self.tool_cages[pidx].dock_from_vector(t)
            child_dock = self.tool_cages[i].dock_from_vector(R.T @ (-t))
            if parent_dock is None or child_dock is None:
                continue
            connections.append(
                {
                    "parent": pidx,
                    "child": i,
                    "parent_dock": parent_dock,
                    "child_dock": child_dock,
                    "rotation": R,
                }
            )
        return connections


    def _meshes_np(self, tool_params: np.ndarray) -> list[_Mesh]:
        meshes = []
        for render_rec in self.spec.render_records:
            rec = render_rec.source_record
            if rec is None:
                continue
            V, F = _read_mesh(render_rec.mesh_path)
            if rec.domain == "tool":
                i = self.tool_index.get(id(rec), -1)
                if 0 <= i < len(self.tool_cages):
                    V = self.tool_cages[i].mesh_vertices()
            meshes.append(_Mesh(V, F))
        return meshes


    generic_design_protocol = GENERIC_DESIGN_PROTOCOL

    def __init__(
        self,
        spec: DirectPlanarHexSpec,
        *,
        optimize_finger_design: bool = False,
        force_connectivity: bool = False,
        head_topology: HeadTopology | None = None,
    ):
        self.spec = spec
        self.optimize_finger_design = bool(optimize_finger_design)
        self.force_connectivity = bool(force_connectivity)
        self.finger_design = None
        if self.optimize_finger_design:
            raise ValueError(
                "finger morphology has been removed; use canonical "
                "Handle-root Head morphology"
            )
        self.tool_cages = [DirectPlanarToolHexNP(rec) for rec in spec.tool_records]
        self.tool_index = {id(rec): idx for idx, rec in enumerate(spec.tool_records)}
        self.head_topology = head_topology
        self._inferred_connection_docks: dict[int, tuple[tuple[int, int], tuple[int, int]] | None] = {}
        if head_topology is None:
            self.face_masks = self._build_connected_face_masks_np()
            self.occupied_docks = self._build_occupied_docks()
            self.connections = self._build_connections()
        else:
            if len(head_topology.blocks) != len(spec.tool_records):
                raise ValueError(
                    "Head topology block count does not match design spec"
                )
            self.face_masks = [
                np.asarray(
                    block.connected_face_mask,
                    dtype=bool,
                )
                for block in head_topology.blocks
            ]
            self.occupied_docks = [
                set(block.occupied_docks)
                for block in head_topology.blocks
            ]
            self.connections = _connections_numpy(head_topology)
        for cage, face_mask in zip(self.tool_cages, self.face_masks):
            cage.set_face_mask(face_mask)

        # RedMax's XML loader is the authority for the undeformed model.
        # Reconstructing that same model from mesh/contact assets introduces
        # small round-off differences (primarily in p1/p3), which can change a
        # contact-sensitive rollout before optimization takes its first step.
        # Cache the reconstructed origin so every parameterized model can be
        # expressed as an exact XML baseline plus a differentiable deformation
        # relative to that origin.
        self._xml_design_baseline = np.asarray(
            self.spec.baseline,
            dtype=np.float64,
        ).copy()
        self._reconstructed_design_origin = (
            self._parameterize_heads_unanchored(self.spec.init_cage_params)
        )

    def _parameterize_heads_unanchored(
        self,
        head_params: np.ndarray,
    ) -> np.ndarray:
        out = np.array(self.spec.baseline, copy=True)
        self._apply_tools_np(out, head_params)
        self._apply_markers_np(out, head_params)
        return out

    def _anchor_to_xml_baseline(
        self,
        design_params: np.ndarray,
    ) -> np.ndarray:
        return self._xml_design_baseline + (
            np.asarray(design_params, dtype=np.float64)
            - self._reconstructed_design_origin
        )

    def parameterize(
        self,
        cage_params: np.ndarray,
        generate_mesh: bool = False,
    ):
        values = np.asarray(cage_params, dtype=np.float64).reshape(-1)
        out = self._anchor_to_xml_baseline(
            self._parameterize_heads_unanchored(values)
        )
        if generate_mesh:
            return out, self._meshes_np(values)
        return out

    def parameterize_heads(
        self,
        head_params: np.ndarray,
        *,
        generate_mesh: bool = False,
    ):
        """Map only per-Head coordinates, with no legacy finger prefix."""

        values = np.asarray(head_params, dtype=np.float64).reshape(-1)
        expected = DIRECT_PARAM_DIM * len(self.spec.tool_records)
        if values.shape != (expected,):
            raise ValueError(
                f"Head morphology shape {values.shape} != expected "
                f"({expected},)"
            )
        out = self._anchor_to_xml_baseline(
            self._parameterize_heads_unanchored(values)
        )
        if generate_mesh:
            return out, self._meshes_np(values)
        return out

    @staticmethod
    def _mark_dock_face(mask: np.ndarray, dock: tuple[int, int] | None) -> None:
        if dock is None:
            return
        face_idx = DOCK_TO_FACE_INDEX.get(dock)
        if face_idx is not None:
            mask[face_idx] = True

    def _build_connected_face_masks_np(self) -> list[np.ndarray]:
        masks = [np.zeros(len(FACE_ORDER), dtype=bool) for _ in self.spec.tool_records]
        for i, rec in enumerate(self.spec.tool_records):
            parent = self.spec.records[rec.parent] if rec.parent is not None else None
            cdock = _dock_from_face_id(rec.planar_child_face)
            pdock = _dock_from_face_id(rec.planar_parent_face) if parent is not None and id(parent) in self.tool_index else None
            if parent is not None and id(parent) in self.tool_index and (cdock is None or pdock is None):
                inferred = self._infer_connection_docks_np(rec, self.tool_cages[i], parent)
                if inferred is not None:
                    pdock, cdock = inferred
            if parent is not None:
                self._mark_dock_face(masks[i], cdock)
            if parent is not None and id(parent) in self.tool_index and pdock is not None:
                self._mark_dock_face(masks[self.tool_index[id(parent)]], pdock)
        for marker in self.spec.marker_records:
            parent = self.spec.records[marker.parent] if marker.parent is not None else None
            if parent is not None and id(parent) in self.tool_index:
                self._mark_dock_face(masks[self.tool_index[id(parent)]], _dock_from_face_id(marker.planar_parent_face))
        return masks

    @staticmethod
    def _body_point_np(rec: LinkRecord, point: np.ndarray) -> np.ndarray:
        """Map a body-local mesh/cage point into the link frame."""
        return rec.body_E[:3, :3] @ np.asarray(point, dtype=np.float64) + rec.body_E[:3, 3]

    @staticmethod
    def _body_frame_np(rec: LinkRecord, frame: np.ndarray) -> np.ndarray:
        return rec.body_E[:3, :3] @ np.asarray(frame, dtype=np.float64)

    def _infer_connection_docks_np(
        self,
        rec: LinkRecord,
        cage: DirectPlanarToolHexNP,
        parent: LinkRecord | None,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        cached = self._inferred_connection_docks.get(id(rec), "missing")
        if cached != "missing":
            return cached
        if parent is None or id(parent) not in self.tool_index:
            self._inferred_connection_docks[id(rec)] = None
            return None

        parent_cage = self.tool_cages[self.tool_index[id(parent)]]
        R = rec.joint_E[:3, :3]
        t = rec.joint_E[:3, 3]
        scale = max(
            float(np.linalg.norm(parent_cage.extents(initial=True))),
            float(np.linalg.norm(cage.extents(initial=True))),
            1.0,
        )
        best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
        for pdock in FACE_CORNERS:
            parent_anchor = self._body_point_np(parent, parent_cage.dock_point(pdock, initial=True))
            for cdock in FACE_CORNERS:
                child_anchor = self._body_point_np(rec, cage.dock_point(cdock, initial=True))
                residual = float(np.linalg.norm(parent_anchor - (t + R @ child_anchor)))
                if best is None or residual < best[0]:
                    best = (residual, pdock, cdock)

        # Only infer legacy metadata when the baseline joint already expresses a
        # clear face-to-face dock. Larger residuals usually mean the child was
        # intentionally offset from the parent.
        tol = max(1e-4, 1e-4 * scale)
        if best is not None and best[0] <= tol:
            inferred: tuple[tuple[int, int], tuple[int, int]] | None = (best[1], best[2])
        else:
            inferred = None
        self._inferred_connection_docks[id(rec)] = inferred
        return inferred

    def _aligned_connection_rotation_np(
        self,
        parent_cage: DirectPlanarToolHexNP,
        parent_dock: tuple[int, int],
        child_cage: DirectPlanarToolHexNP,
        child_dock: tuple[int, int],
        baseline_R: np.ndarray,
    ) -> np.ndarray:
        parent_rec = parent_cage.rec
        child_rec = child_cage.rec
        parent_frame = self._body_frame_np(parent_rec, face_frame(parent_cage.vertices, parent_dock))
        child_frame = self._body_frame_np(child_rec, face_frame(child_cage.vertices, child_dock))
        n = parent_frame[:, 2]
        tangent_choices = (
            parent_frame[:, 0],
            parent_frame[:, 1],
            -parent_frame[:, 0],
            -parent_frame[:, 1],
        )
        best_R = None
        best_score = float("inf")
        for u in tangent_choices:
            target_n = -n
            target_v = np.cross(target_n, u)
            target_v = target_v / max(float(np.linalg.norm(target_v)), 1e-12)
            target = np.column_stack([u, target_v, target_n])
            R = target @ child_frame.T
            score = float(np.linalg.norm(R - baseline_R))
            if score < best_score:
                best_score = score
                best_R = R
        return np.asarray(best_R, dtype=np.float64)

    def _edge_corner_transform_np(
        self,
        rec: LinkRecord,
        cage: DirectPlanarToolHexNP,
        parent: LinkRecord,
        parent_cage: DirectPlanarToolHexNP,
        pdock: tuple[int, int],
        cdock: tuple[int, int],
        parent_barycentric: np.ndarray | None,
        child_barycentric: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        parent_tangent_axis, parent_normal_axis = _edge_axis_metadata(
            rec,
            "parent",
        )
        child_tangent_axis, child_normal_axis = _edge_axis_metadata(
            rec,
            "child",
        )
        parent_s, parent_b, parent_n, parent_half_thickness = _edge_frame_np(
            parent,
            parent_cage,
            face=int(rec.planar_parent_face),
            tangent_axis=parent_tangent_axis,
            normal_axis=parent_normal_axis,
        )
        child_s, child_b, child_n, child_half_thickness = _edge_frame_np(
            rec,
            cage,
            face=int(rec.planar_child_face),
            tangent_axis=child_tangent_axis,
            normal_axis=child_normal_axis,
        )
        orientation = int(rec.planar_facing or 0)
        fold_side = 1.0 if (orientation & 1) == 0 else -1.0
        seam_sign = 1.0 if (orientation & 2) == 0 else -1.0
        target_s = seam_sign * parent_s
        target_b = -fold_side * parent_n
        target_n = _normalize_np(np.cross(target_s, target_b))
        R = (
            np.column_stack([target_s, target_b, target_n])
            @ np.column_stack([child_s, child_b, child_n]).T
        )
        flush_sign = float(np.dot(target_n, parent_b))
        parent_anchor = self._body_point_np(
            parent,
            parent_cage.dock_point(pdock, barycentric=parent_barycentric),
        )
        child_anchor = self._body_point_np(
            rec,
            cage.dock_point(cdock, barycentric=child_barycentric),
        )
        target_anchor = (
            parent_anchor
            + fold_side * parent_half_thickness * parent_n
            - flush_sign * child_half_thickness * parent_b
        )
        return R, target_anchor - R @ child_anchor

    def _connection_transform_np(
        self,
        rec: LinkRecord,
        cage: DirectPlanarToolHexNP,
        parent: LinkRecord | None,
        t: np.ndarray,
        baseline_R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        topology_block = (
            None
            if self.head_topology is None
            else self.head_topology.block(self.tool_index[id(rec)])
        )
        cdock = (
            _dock_from_face_id(rec.planar_child_face)
            if topology_block is None
            else topology_block.child_dock
        )
        child_barycentric = rec.planar_child_barycentric
        pdock = (
            None
            if topology_block is None
            else topology_block.parent_dock
        )
        parent_barycentric = rec.planar_parent_barycentric
        explicit_connection = (
            cdock is not None and rec.planar_parent_face is not None
            if topology_block is None
            else topology_block.explicit_connection
        )
        if parent is not None and id(parent) in self.tool_index:
            if topology_block is None:
                pdock = _dock_from_face_id(rec.planar_parent_face)
            if (
                topology_block is None
                and (cdock is None or pdock is None)
            ):
                inferred = self._infer_connection_docks_np(rec, cage, parent)
                if inferred is not None:
                    pdock, cdock = inferred
                    child_barycentric = None
                    parent_barycentric = None
                    explicit_connection = False
            elif (
                topology_block is not None
                and topology_block.inferred_connection
            ):
                child_barycentric = None
                parent_barycentric = None
        if cdock is None:
            return baseline_R, t
        if parent is None or id(parent) not in self.tool_index:
            delta = self._body_point_np(rec, cage.dock_point(cdock, barycentric=child_barycentric)) - self._body_point_np(
                rec,
                cage.dock_point(cdock, initial=True, barycentric=child_barycentric),
            )
            return baseline_R, t - baseline_R @ delta

        if pdock is None:
            return baseline_R, t
        parent_cage = self.tool_cages[self.tool_index[id(parent)]]
        if (
            explicit_connection
            and str(getattr(rec, "attrs", {}).get("planar_connection_family", ""))
            == EDGE_CORNER_CONNECTION
        ):
            return self._edge_corner_transform_np(
                rec,
                cage,
                parent,
                parent_cage,
                pdock,
                cdock,
                parent_barycentric,
                child_barycentric,
            )
        R = (
            self._aligned_connection_rotation_np(parent_cage, pdock, cage, cdock, baseline_R)
            if explicit_connection
            else baseline_R
        )
        parent_anchor = self._body_point_np(parent, parent_cage.dock_point(pdock, barycentric=parent_barycentric))
        child_anchor = self._body_point_np(rec, cage.dock_point(cdock, barycentric=child_barycentric))
        return R, parent_anchor - R @ child_anchor

    def _apply_tools_np(self, out: np.ndarray, params: np.ndarray) -> None:
        cursor = 0
        for i, cage in enumerate(self.tool_cages):
            values = params[cursor : cursor + DIRECT_PARAM_DIM]
            if values.shape[0] != DIRECT_PARAM_DIM:
                values = cage.params0
            cage.apply_params(values)
            cursor += DIRECT_PARAM_DIM
        for i, rec in enumerate(self.spec.tool_records):
            cage = self.tool_cages[i]
            if rec.p1_slice is not None:
                E = np.array(rec.joint_E, copy=True)
                parent = self.spec.records[rec.parent] if rec.parent is not None else None
                t = rec.joint_E[:3, 3]
                R, translated = self._connection_transform_np(rec, cage, parent, t, rec.joint_E[:3, :3])
                E[:3, :3] = R
                E[:3, 3] = translated
                out[rec.p1_slice] = _flatten_E_np(E)
            if rec.p2_slice is not None:
                out[rec.p2_slice] = _flatten_E_np(rec.body_E)
            if rec.p3_slice is not None:
                pts = cage.contacts()
                if pts.shape[0]:
                    out[rec.p3_slice] = pts.reshape(-1)
            if rec.p4_slice is not None:
                out[rec.p4_slice] = cage.inertia(self.spec.baseline[rec.p4_slice])
            if rec.p6_slice is not None:
                out[rec.p6_slice] = cage.contact_scale(float(self.spec.baseline[rec.p6_slice][0]))

    def _record_transform_np(
        self,
        rec: LinkRecord,
        design_params: np.ndarray,
        cache: dict[int, np.ndarray],
    ) -> np.ndarray:
        key = id(rec)
        if key in cache:
            return cache[key]
        parent = np.eye(4, dtype=np.float64)
        if rec.parent is not None:
            parent = self._record_transform_np(self.spec.records[rec.parent], design_params, cache)
        joint = rec.joint_E
        if rec.p1_slice is not None:
            joint = _transform_from_flat_np(design_params[rec.p1_slice])
        body = rec.body_E
        if rec.p2_slice is not None:
            body = _transform_from_flat_np(design_params[rec.p2_slice])
        transform = parent @ joint @ body
        cache[key] = transform
        return transform

    def _function_marker_translation_np(
        self,
        marker: LinkRecord,
        design_params: np.ndarray,
    ) -> np.ndarray | None:
        leaves = _marker_leaf_records(self.spec, marker)
        if not leaves or marker.parent is None:
            return None
        cache: dict[int, np.ndarray] = {}
        points = []
        use_leaf_centers = len(leaves) > 1
        for leaf in leaves:
            if id(leaf) not in self.tool_index:
                continue
            cage = self.tool_cages[self.tool_index[id(leaf)]]
            if use_leaf_centers:
                local = cage.centroid()
            else:
                dock = _terminal_dock(leaf)
                if dock is None:
                    continue
                barycentric = getattr(
                    leaf,
                    "planar_endeffector_barycentric",
                    None,
                )
                local = cage.dock_point(
                    dock,
                    barycentric=barycentric,
                )
            leaf_transform = self._record_transform_np(leaf, design_params, cache)
            points.append(
                leaf_transform[:3, :3] @ local
                + leaf_transform[:3, 3]
            )
        if len(points) != len(leaves):
            return None
        parent = self.spec.records[marker.parent]
        parent_transform = self._record_transform_np(parent, design_params, cache)
        center = np.mean(np.asarray(points, dtype=np.float64), axis=0)
        return parent_transform[:3, :3].T @ (
            center - parent_transform[:3, 3]
        )

    def _apply_markers_np(self, out: np.ndarray, tool_params: np.ndarray) -> None:
        _ = tool_params
        for rec in self.spec.marker_records:
            if rec.p1_slice is None:
                continue
            E = np.array(rec.joint_E, copy=True)
            parent = self.spec.records[rec.parent] if rec.parent is not None else None
            function_translation = self._function_marker_translation_np(rec, out)
            pdock = _dock_from_face_id(rec.planar_parent_face)
            if function_translation is not None:
                E[:3, 3] = function_translation
            elif parent is not None and id(parent) in self.tool_index and pdock is not None:
                cage = self.tool_cages[self.tool_index[id(parent)]]
                E[:3, 3] = self._body_point_np(parent, cage.dock_point(pdock, barycentric=rec.planar_parent_barycentric))
            out[rec.p1_slice] = _flatten_E_np(E)


class DirectPlanarHexDesignTorch:
    def _build_occupied_docks(self) -> list[set[tuple[int, int]]]:
        occupied = [set() for _ in self.spec.tool_records]
        for i, rec in enumerate(self.spec.tool_records):
            parent = self.spec.records[rec.parent] if rec.parent is not None else None
            E = torch.tensor(rec.joint_E, dtype=torch.double)
            t = E[:3, 3]
            child_local = E[:3, :3].T @ (-t)
            dock = self.tool_cages[i].dock_from_vector(child_local)
            if dock is not None:
                occupied[i].add(dock)
            if parent is not None and id(parent) in self.tool_index:
                pidx = self.tool_index[id(parent)]
                pdock = self.tool_cages[pidx].dock_from_vector(t)
                if pdock is not None:
                    occupied[pidx].add(pdock)
        return occupied


    def _build_connections(self) -> list[dict[str, Any]]:
        connections: list[dict[str, Any]] = []
        for i, rec in enumerate(self.spec.tool_records):
            parent = self.spec.records[rec.parent] if rec.parent is not None else None
            if parent is None or id(parent) not in self.tool_index:
                continue
            pidx = self.tool_index[id(parent)]
            E = torch.tensor(rec.joint_E, dtype=torch.double)
            t = E[:3, 3]
            R = E[:3, :3]
            parent_dock = self.tool_cages[pidx].dock_from_vector(t)
            child_dock = self.tool_cages[i].dock_from_vector(R.T @ (-t))
            if parent_dock is None or child_dock is None:
                continue
            connections.append(
                {
                    "parent": pidx,
                    "child": i,
                    "parent_dock": parent_dock,
                    "child_dock": child_dock,
                    "rotation": R,
                }
            )
        return connections


    generic_design_protocol = GENERIC_DESIGN_PROTOCOL

    def __init__(
        self,
        spec: DirectPlanarHexSpec,
        *,
        optimize_finger_design: bool = False,
        force_connectivity: bool = False,
        head_topology: HeadTopology | None = None,
    ):
        self.spec = spec
        self.optimize_finger_design = bool(optimize_finger_design)
        self.force_connectivity = bool(force_connectivity)
        self.finger_design = None
        if self.optimize_finger_design:
            raise ValueError(
                "finger morphology has been removed; use canonical "
                "Handle-root Head morphology"
            )
        self.tool_cages = [DirectPlanarToolHexTorch(rec) for rec in spec.tool_records]
        self.tool_index = {id(rec): idx for idx, rec in enumerate(spec.tool_records)}
        self.head_topology = head_topology
        self._current_joint_transforms: dict[int, torch.Tensor] = {}
        self._inferred_connection_docks: dict[int, tuple[tuple[int, int], tuple[int, int]] | None] = {}
        if head_topology is None:
            self.face_masks = self._build_connected_face_masks_torch()
            self.occupied_docks = self._build_occupied_docks()
            self.connections = self._build_connections()
        else:
            if len(head_topology.blocks) != len(spec.tool_records):
                raise ValueError(
                    "Head topology block count does not match design spec"
                )
            self.face_masks = [
                np.asarray(
                    block.connected_face_mask,
                    dtype=bool,
                )
                for block in head_topology.blocks
            ]
            self.occupied_docks = [
                set(block.occupied_docks)
                for block in head_topology.blocks
            ]
            self.connections = _connections_torch(head_topology)
        for cage, face_mask in zip(self.tool_cages, self.face_masks):
            cage.set_face_mask(face_mask)

        initial = torch.as_tensor(
            self.spec.init_cage_params,
            dtype=torch.double,
        )
        self._reconstructed_design_origin = (
            self._parameterize_heads_unanchored(initial)
            .detach()
            .cpu()
            .numpy()
        )

    def _parameterize_heads_unanchored(
        self,
        head_params: torch.Tensor,
    ) -> torch.Tensor:
        values = head_params.reshape(-1).to(dtype=torch.double)
        baseline = torch.as_tensor(
            self.spec.baseline,
            dtype=torch.double,
            device=values.device,
        )
        overrides = self._tool_overrides(values)
        overrides.update(self._marker_overrides(values))
        return _assemble_torch_overrides(baseline, overrides)

    def _anchor_to_xml_baseline(
        self,
        design_params: torch.Tensor,
    ) -> torch.Tensor:
        baseline = torch.as_tensor(
            self.spec.baseline,
            dtype=torch.double,
            device=design_params.device,
        )
        reconstructed_origin = torch.as_tensor(
            self._reconstructed_design_origin,
            dtype=torch.double,
            device=design_params.device,
        )
        return baseline + (design_params - reconstructed_origin)

    def parameterize(
        self,
        cage_params: torch.Tensor,
        generate_mesh: bool = False,
    ) -> torch.Tensor:
        if generate_mesh:
            raise NotImplementedError("Torch mesh generation is not used")
        return self._anchor_to_xml_baseline(
            self._parameterize_heads_unanchored(cage_params)
        )

    def parameterize_heads(self, head_params: torch.Tensor) -> torch.Tensor:
        """Torch derivative path for active Head coordinates only."""

        values = head_params.reshape(-1).to(dtype=torch.double)
        expected = DIRECT_PARAM_DIM * len(self.spec.tool_records)
        if values.numel() != expected:
            raise ValueError(
                f"Head morphology dim {values.numel()} != expected {expected}"
            )
        return self._anchor_to_xml_baseline(
            self._parameterize_heads_unanchored(values)
        )

    @staticmethod
    def _mark_dock_face(mask: np.ndarray, dock: tuple[int, int] | None) -> None:
        if dock is None:
            return
        face_idx = DOCK_TO_FACE_INDEX.get(dock)
        if face_idx is not None:
            mask[face_idx] = True

    def _build_connected_face_masks_torch(self) -> list[np.ndarray]:
        masks = [np.zeros(len(FACE_ORDER), dtype=bool) for _ in self.spec.tool_records]
        for i, rec in enumerate(self.spec.tool_records):
            parent = self.spec.records[rec.parent] if rec.parent is not None else None
            cdock = _dock_from_face_id(rec.planar_child_face)
            pdock = _dock_from_face_id(rec.planar_parent_face) if parent is not None and id(parent) in self.tool_index else None
            if parent is not None and id(parent) in self.tool_index and (cdock is None or pdock is None):
                inferred = self._infer_connection_docks_torch(rec, self.tool_cages[i], parent)
                if inferred is not None:
                    pdock, cdock = inferred
            if parent is not None:
                self._mark_dock_face(masks[i], cdock)
            if parent is not None and id(parent) in self.tool_index and pdock is not None:
                self._mark_dock_face(masks[self.tool_index[id(parent)]], pdock)
        for marker in self.spec.marker_records:
            parent = self.spec.records[marker.parent] if marker.parent is not None else None
            if parent is not None and id(parent) in self.tool_index:
                self._mark_dock_face(masks[self.tool_index[id(parent)]], _dock_from_face_id(marker.planar_parent_face))
        return masks

    @staticmethod
    def _body_point_torch(rec: LinkRecord, point: torch.Tensor) -> torch.Tensor:
        E = torch.tensor(rec.body_E, dtype=torch.double, device=point.device)
        return E[:3, :3] @ point.to(dtype=torch.double) + E[:3, 3]

    @staticmethod
    def _body_frame_torch(rec: LinkRecord, frame: torch.Tensor) -> torch.Tensor:
        R = torch.tensor(rec.body_E[:3, :3], dtype=torch.double, device=frame.device)
        return R @ frame.to(dtype=torch.double)

    def _infer_connection_docks_torch(
        self,
        rec: LinkRecord,
        cage: DirectPlanarToolHexTorch,
        parent: LinkRecord | None,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        cached = self._inferred_connection_docks.get(id(rec), "missing")
        if cached != "missing":
            return cached
        if parent is None or id(parent) not in self.tool_index:
            self._inferred_connection_docks[id(rec)] = None
            return None

        parent_cage = self.tool_cages[self.tool_index[id(parent)]]
        R = rec.joint_E[:3, :3]
        t = rec.joint_E[:3, 3]
        parent_ext = parent_cage.extents(initial=True).detach().cpu().numpy()
        child_ext = cage.extents(initial=True).detach().cpu().numpy()
        scale = max(float(np.linalg.norm(parent_ext)), float(np.linalg.norm(child_ext)), 1.0)
        best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
        for pdock in FACE_CORNERS:
            parent_anchor = (
                parent.body_E[:3, :3]
                @ parent_cage.dock_point(pdock, initial=True).detach().cpu().numpy()
                + parent.body_E[:3, 3]
            )
            for cdock in FACE_CORNERS:
                child_anchor = rec.body_E[:3, :3] @ cage.dock_point(cdock, initial=True).detach().cpu().numpy() + rec.body_E[:3, 3]
                residual = float(np.linalg.norm(parent_anchor - (t + R @ child_anchor)))
                if best is None or residual < best[0]:
                    best = (residual, pdock, cdock)
        tol = max(1e-4, 1e-4 * scale)
        if best is not None and best[0] <= tol:
            inferred: tuple[tuple[int, int], tuple[int, int]] | None = (best[1], best[2])
        else:
            inferred = None
        self._inferred_connection_docks[id(rec)] = inferred
        return inferred

    def _aligned_connection_rotation_torch(
        self,
        parent_cage: DirectPlanarToolHexTorch,
        parent_dock: tuple[int, int],
        child_cage: DirectPlanarToolHexTorch,
        child_dock: tuple[int, int],
        baseline_R: torch.Tensor,
    ) -> torch.Tensor:
        parent_frame = self._body_frame_torch(parent_cage.rec, face_frame_torch(parent_cage.vertices, parent_dock))
        child_frame = self._body_frame_torch(child_cage.rec, face_frame_torch(child_cage.vertices, child_dock))
        n = parent_frame[:, 2]
        tangent_choices = (
            parent_frame[:, 0],
            parent_frame[:, 1],
            -parent_frame[:, 0],
            -parent_frame[:, 1],
        )
        candidates = []
        scores = []
        for u in tangent_choices:
            target_n = -n
            target_v = torch.cross(target_n, u, dim=0)
            target_v = target_v / torch.clamp(torch.norm(target_v), min=1e-12)
            target = torch.stack([u, target_v, target_n], dim=1)
            R = target @ child_frame.T
            candidates.append(R)
            scores.append(float(torch.norm((R - baseline_R).detach()).cpu()))
        return candidates[int(np.argmin(scores))]

    def _edge_corner_transform_torch(
        self,
        rec: LinkRecord,
        cage: DirectPlanarToolHexTorch,
        parent: LinkRecord,
        parent_cage: DirectPlanarToolHexTorch,
        pdock: tuple[int, int],
        cdock: tuple[int, int],
        parent_barycentric: np.ndarray | None,
        child_barycentric: np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parent_tangent_axis, parent_normal_axis = _edge_axis_metadata(
            rec,
            "parent",
        )
        child_tangent_axis, child_normal_axis = _edge_axis_metadata(
            rec,
            "child",
        )
        parent_s, parent_b, parent_n, parent_half_thickness = _edge_frame_torch(
            parent,
            parent_cage,
            face=int(rec.planar_parent_face),
            tangent_axis=parent_tangent_axis,
            normal_axis=parent_normal_axis,
        )
        child_s, child_b, child_n, child_half_thickness = _edge_frame_torch(
            rec,
            cage,
            face=int(rec.planar_child_face),
            tangent_axis=child_tangent_axis,
            normal_axis=child_normal_axis,
        )
        orientation = int(rec.planar_facing or 0)
        fold_side = 1.0 if (orientation & 1) == 0 else -1.0
        seam_sign = 1.0 if (orientation & 2) == 0 else -1.0
        target_s = seam_sign * parent_s
        target_b = -fold_side * parent_n
        target_n = _normalize_torch(torch.cross(target_s, target_b, dim=0))
        R = (
            torch.stack([target_s, target_b, target_n], dim=1)
            @ torch.stack([child_s, child_b, child_n], dim=1).T
        )
        flush_sign = torch.dot(target_n, parent_b)
        parent_anchor = self._body_point_torch(
            parent,
            parent_cage.dock_point(pdock, barycentric=parent_barycentric),
        )
        child_anchor = self._body_point_torch(
            rec,
            cage.dock_point(cdock, barycentric=child_barycentric),
        )
        target_anchor = (
            parent_anchor
            + fold_side * parent_half_thickness * parent_n
            - flush_sign * child_half_thickness * parent_b
        )
        return R, target_anchor - R @ child_anchor

    def _connection_transform_torch(
        self,
        rec: LinkRecord,
        cage: DirectPlanarToolHexTorch,
        parent: LinkRecord | None,
        t: torch.Tensor,
        baseline_R: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topology_block = (
            None
            if self.head_topology is None
            else self.head_topology.block(self.tool_index[id(rec)])
        )
        cdock = (
            _dock_from_face_id(rec.planar_child_face)
            if topology_block is None
            else topology_block.child_dock
        )
        child_barycentric = rec.planar_child_barycentric
        pdock = (
            None
            if topology_block is None
            else topology_block.parent_dock
        )
        parent_barycentric = rec.planar_parent_barycentric
        explicit_connection = (
            cdock is not None and rec.planar_parent_face is not None
            if topology_block is None
            else topology_block.explicit_connection
        )
        if parent is not None and id(parent) in self.tool_index:
            if topology_block is None:
                pdock = _dock_from_face_id(rec.planar_parent_face)
            if (
                topology_block is None
                and (cdock is None or pdock is None)
            ):
                inferred = self._infer_connection_docks_torch(rec, cage, parent)
                if inferred is not None:
                    pdock, cdock = inferred
                    child_barycentric = None
                    parent_barycentric = None
                    explicit_connection = False
            elif (
                topology_block is not None
                and topology_block.inferred_connection
            ):
                child_barycentric = None
                parent_barycentric = None
        if cdock is None:
            return baseline_R, t
        if parent is None or id(parent) not in self.tool_index:
            current_anchor = self._body_point_torch(rec, cage.dock_point(cdock, barycentric=child_barycentric))
            initial_anchor = self._body_point_torch(
                rec,
                cage.dock_point(cdock, initial=True, barycentric=child_barycentric),
            )
            delta = current_anchor - initial_anchor
            return baseline_R, t - baseline_R @ delta

        if pdock is None:
            return baseline_R, t
        parent_cage = self.tool_cages[self.tool_index[id(parent)]]
        if (
            explicit_connection
            and str(getattr(rec, "attrs", {}).get("planar_connection_family", ""))
            == EDGE_CORNER_CONNECTION
        ):
            return self._edge_corner_transform_torch(
                rec,
                cage,
                parent,
                parent_cage,
                pdock,
                cdock,
                parent_barycentric,
                child_barycentric,
            )
        R = (
            self._aligned_connection_rotation_torch(parent_cage, pdock, cage, cdock, baseline_R)
            if explicit_connection
            else baseline_R
        )
        parent_anchor = self._body_point_torch(parent, parent_cage.dock_point(pdock, barycentric=parent_barycentric))
        child_anchor = self._body_point_torch(rec, cage.dock_point(cdock, barycentric=child_barycentric))
        return R, parent_anchor - R @ child_anchor

    def _tool_overrides(self, params: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
        overrides: dict[tuple[int, int], torch.Tensor] = {}
        self._current_joint_transforms = {}
        if params.numel() == 0:
            return overrides
        cursor = 0
        for i, cage in enumerate(self.tool_cages):
            values = params[cursor : cursor + DIRECT_PARAM_DIM]
            if values.numel() != DIRECT_PARAM_DIM:
                values = cage.params0.to(dtype=torch.double, device=params.device)
            cage.apply_params(values)
            cursor += DIRECT_PARAM_DIM
        for i, rec in enumerate(self.spec.tool_records):
            cage = self.tool_cages[i]
            if rec.p1_slice is not None:
                E = torch.tensor(rec.joint_E, dtype=torch.double, device=params.device)
                t = E[:3, 3]
                R = E[:3, :3]
                parent = self.spec.records[rec.parent] if rec.parent is not None else None
                rotated, translated = self._connection_transform_torch(rec, cage, parent, t, R)
                E = torch.cat(
                    [
                        torch.cat([rotated, translated.reshape(3, 1)], dim=1),
                        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.double, device=params.device),
                    ],
                    dim=0,
                )
                self._current_joint_transforms[id(rec)] = E
                overrides[(rec.p1_slice.start, rec.p1_slice.stop)] = torch.cat(
                    [rotated.reshape(-1), translated]
                )
            else:
                self._current_joint_transforms[id(rec)] = torch.tensor(
                    rec.joint_E,
                    dtype=torch.double,
                    device=params.device,
                )
            if rec.p2_slice is not None:
                overrides[(rec.p2_slice.start, rec.p2_slice.stop)] = torch.tensor(
                    _flatten_E_np(rec.body_E),
                    dtype=torch.double,
                    device=params.device,
                )
            if rec.p3_slice is not None:
                pts = cage.contacts()
                if pts.numel():
                    overrides[(rec.p3_slice.start, rec.p3_slice.stop)] = pts.reshape(-1)
            if rec.p4_slice is not None:
                overrides[(rec.p4_slice.start, rec.p4_slice.stop)] = cage.inertia(self.spec.baseline[rec.p4_slice])
            if rec.p6_slice is not None:
                overrides[(rec.p6_slice.start, rec.p6_slice.stop)] = cage.contact_scale(
                    self.spec.baseline[rec.p6_slice][0]
                ).reshape(1)
        return overrides

    def _record_transform_torch(
        self,
        rec: LinkRecord,
        device: torch.device,
        cache: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        key = id(rec)
        if key in cache:
            return cache[key]
        parent = torch.eye(4, dtype=torch.double, device=device)
        if rec.parent is not None:
            parent = self._record_transform_torch(self.spec.records[rec.parent], device, cache)
        joint = self._current_joint_transforms.get(key)
        if joint is None:
            joint = torch.tensor(rec.joint_E, dtype=torch.double, device=device)
        body = torch.tensor(rec.body_E, dtype=torch.double, device=device)
        transform = parent @ joint @ body
        cache[key] = transform
        return transform

    def _function_marker_translation_torch(
        self,
        marker: LinkRecord,
        device: torch.device,
    ) -> torch.Tensor | None:
        leaves = _marker_leaf_records(self.spec, marker)
        if not leaves or marker.parent is None:
            return None
        cache: dict[int, torch.Tensor] = {}
        points = []
        use_leaf_centers = len(leaves) > 1
        for leaf in leaves:
            if id(leaf) not in self.tool_index:
                continue
            cage = self.tool_cages[self.tool_index[id(leaf)]]
            if use_leaf_centers:
                local = cage.centroid()
            else:
                dock = _terminal_dock(leaf)
                if dock is None:
                    continue
                barycentric = getattr(
                    leaf,
                    "planar_endeffector_barycentric",
                    None,
                )
                local = cage.dock_point(
                    dock,
                    barycentric=barycentric,
                )
            leaf_transform = self._record_transform_torch(leaf, device, cache)
            points.append(
                leaf_transform[:3, :3] @ local
                + leaf_transform[:3, 3]
            )
        if len(points) != len(leaves):
            return None
        parent = self.spec.records[marker.parent]
        parent_transform = self._record_transform_torch(parent, device, cache)
        center = torch.stack(points, dim=0).mean(dim=0)
        return parent_transform[:3, :3].T @ (
            center - parent_transform[:3, 3]
        )

    def _marker_overrides(self, tool_params: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
        overrides: dict[tuple[int, int], torch.Tensor] = {}
        for rec in self.spec.marker_records:
            if rec.p1_slice is None:
                continue
            flat = _flatten_E_np(rec.joint_E)
            rot = torch.tensor(flat[:9], dtype=torch.double, device=tool_params.device)
            translation = torch.tensor(rec.joint_E[:3, 3], dtype=torch.double, device=tool_params.device)
            parent = self.spec.records[rec.parent] if rec.parent is not None else None
            function_translation = self._function_marker_translation_torch(rec, tool_params.device)
            pdock = _dock_from_face_id(rec.planar_parent_face)
            if function_translation is not None:
                translation = function_translation
            elif parent is not None and id(parent) in self.tool_index and pdock is not None:
                cage = self.tool_cages[self.tool_index[id(parent)]]
                translation = self._body_point_torch(parent, cage.dock_point(pdock, barycentric=rec.planar_parent_barycentric))
            overrides[(rec.p1_slice.start, rec.p1_slice.stop)] = torch.cat([rot, translation])
        return overrides


def connected_direct_planar_hex_bounds_for_bundle(
    bundle: DesignBundle,
    ndof_cage: int,
    *,
    handle_margin: float = 2.0,
    **_: Any,
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    records = list(getattr(getattr(bundle, "spec", None), "tool_records", []) or [])
    cages = list(getattr(getattr(bundle, "design_np", None), "tool_cages", []) or [])
    for rec, cage in zip(records, cages):
        base = np.asarray(getattr(cage, "params0", np.zeros(DIRECT_PARAM_DIM)), dtype=np.float64)
        if not _is_deformable_tool_record(rec):
            bounds.extend([(float(v), float(v)) for v in base])
        else:
            bounds.extend((float(v - handle_margin), float(v + handle_margin)) for v in base)
    if len(bounds) < int(ndof_cage):
        bounds.extend([(-handle_margin, handle_margin)] * (int(ndof_cage) - len(bounds)))
    return bounds[: int(ndof_cage)]


def build_connected_direct_planar_hex_design_bundle(xml_path: str, sim: Any, task_config: dict[str, Any] | None = None) -> DesignBundle:
    task_config = task_config or {}
    optimize_finger_design = bool(task_config.get("optimize_finger_design", False))
    force_connectivity = bool(task_config.get("force_connectivity", False))
    spec = DirectPlanarHexSpec(xml_path, sim)
    design_np = DirectPlanarHexDesignNP(
        spec,
        optimize_finger_design=optimize_finger_design,
        force_connectivity=force_connectivity,
    )
    design_torch = DirectPlanarHexDesignTorch(
        spec,
        optimize_finger_design=optimize_finger_design,
        force_connectivity=force_connectivity,
    )
    bundle = DesignBundle(spec, design_np, design_torch, spec.init_cage_params)
    bundle.generic_design_protocol = GENERIC_DESIGN_PROTOCOL
    bundle.direct_planar_blocks = []
    cursor = 0
    for idx, rec in enumerate(spec.tool_records):
        face_mask = (
            np.asarray(design_np.face_masks[idx], dtype=bool).tolist()
            if idx < len(getattr(design_np, "face_masks", []))
            else [True] * len(FACE_ORDER)
        )
        bundle.direct_planar_blocks.append(
            {
                "tool_index": idx,
                "start": cursor,
                "stop": cursor + DIRECT_PARAM_DIM,
                "deformable": bool(_is_deformable_tool_record(rec)),
                "link_name": rec.link_name,
                "body_name": rec.body_name,
                "face_mask": face_mask,
                "face_order": list(FACE_ORDER),
            }
        )
        cursor += DIRECT_PARAM_DIM
    np_params = np.asarray(design_np.parameterize(bundle.init_cage_params, generate_mesh=False))
    torch_params = design_torch.parameterize(torch.tensor(bundle.init_cage_params, dtype=torch.double))
    if np_params.shape[0] != int(sim.ndof_p):
        raise ValueError(f"NumPy design output dim {np_params.shape[0]} != sim.ndof_p {sim.ndof_p}")
    if torch_params.numel() != int(sim.ndof_p):
        raise ValueError(f"Torch design output dim {torch_params.numel()} != sim.ndof_p {sim.ndof_p}")
    return bundle
