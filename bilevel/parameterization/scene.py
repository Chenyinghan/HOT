"""Scene records, resource loading and RedMax parameter slices."""
from __future__ import annotations
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


FINGER_ROLES = {
    "palm": "palm",
    "knuckle_parent": "k",
    "knuckle_child": "k",
    "joint_parent": "j",
    "joint_child": "j",
    "phalanx": "p",
    "tip": "t",
}


def _parse_vec(text: str | None, n: int, default: Iterable[float]) -> np.ndarray:
    if not text:
        return np.asarray(list(default), dtype=np.float64)
    values = [float(v) for v in text.split()]
    if len(values) != n:
        raise ValueError(f"Expected {n} floats, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _parse_optional_int(text: str | None) -> int | None:
    if text is None or text == "":
        return None
    return int(text)


def _parse_optional_barycentric(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    weights = _parse_vec(text, 4, (0.25, 0.25, 0.25, 0.25))
    total = float(np.sum(weights))
    if abs(total) <= 1e-12:
        return np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    return weights / total


def _parse_int_list(text: str | None) -> tuple[int, ...]:
    if not text:
        return ()
    out = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return tuple(out)


def _parse_bool(text: str | None) -> bool:
    return str(text or "").strip().lower() in {"1", "true", "yes", "y"}


def _infer_node_id(link: ET.Element, body: ET.Element | None) -> int | None:
    for text in (
        link.attrib.get("node_id"),
        body.attrib.get("node_id") if body is not None else None,
    ):
        if text not in (None, ""):
            return int(text)
    for text in (link.attrib.get("name", ""), body.attrib.get("name", "") if body is not None else ""):
        if not text.startswith(("link_tool_", "body_tool_")):
            continue
        match = re.search(r"_(\d+)$", text)
        if match:
            return int(match.group(1))
    return None


def _quat_wxyz_to_R(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _flatten_E_np(E: np.ndarray) -> np.ndarray:
    out = np.zeros(12, dtype=np.float64)
    out[:9] = E[:3, :3].reshape(-1)
    out[9:12] = E[:3, 3]
    return out


def _flatten_E_torch(E: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(12, dtype=torch.double)
    out[:9] = E[:3, :3].reshape(-1)
    out[9:12] = E[:3, 3]
    return out


def _E_from_joint(joint: ET.Element | None) -> np.ndarray:
    E = np.eye(4, dtype=np.float64)
    if joint is None:
        return E
    E[:3, :3] = _quat_wxyz_to_R(_parse_vec(joint.attrib.get("quat"), 4, (1, 0, 0, 0)))
    E[:3, 3] = _parse_vec(joint.attrib.get("pos"), 3, (0, 0, 0))
    return E


def _E_from_body(body: ET.Element | None) -> np.ndarray:
    E = np.eye(4, dtype=np.float64)
    if body is None:
        return E
    E[:3, :3] = _quat_wxyz_to_R(_parse_vec(body.attrib.get("quat"), 4, (1, 0, 0, 0)))
    E[:3, 3] = _parse_vec(body.attrib.get("pos"), 3, (0, 0, 0))
    return E


def _role_from_mesh(mesh: str | None) -> str | None:
    if not mesh:
        return None
    base = os.path.splitext(os.path.basename(mesh).lower())[0]
    for token, role in FINGER_ROLES.items():
        if token == base or token in base:
            return role
    return None


def _is_finger_mesh(mesh: str | None) -> bool:
    if not mesh:
        return False
    lowered = mesh.lower()
    return "/finger/" in lowered or "/finger_new/" in lowered or _role_from_mesh(mesh) is not None


def _is_marker_link(link: ET.Element, body: ET.Element | None) -> bool:
    link_name = link.attrib.get("name", "").lower()
    body_name = body.attrib.get("name", "").lower() if body is not None else ""
    return "endeffector" in link_name or "endeffector" in body_name


def _mask(link: ET.Element) -> int:
    return int(link.attrib.get("design_params", "0") or 0)


def _has(mask: int, bit: int) -> bool:
    return bool(mask & (1 << bit))


def _is_deformable_tool_record(rec: "LinkRecord") -> bool:
    # Only full-design Head records deform; connector masks stay frozen.
    return int(getattr(rec, "mask", 0)) == 47


def _resolve_path(path_text: str | None, xml_path: Path) -> Path | None:
    if not path_text:
        return None
    raw = Path(path_text)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([xml_path.parent / raw, REPO_ROOT / raw])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _count_contacts(path_text: str | None, xml_path: Path) -> int:
    path = _resolve_path(path_text, xml_path)
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    return int(first) if first else 0


def _read_points(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((3, 0), dtype=np.float64)
    with path.open("r", encoding="utf-8") as f:
        n = int(f.readline().strip())
        pts = np.zeros((3, n), dtype=np.float64)
        for i in range(n):
            pts[:, i] = [float(v) for v in f.readline().split()[:3]]
    return pts


def _sibling_path(path: Path | None, folder: str, suffix: str) -> Path | None:
    if path is None:
        return None
    root = path.parent.parent
    return root / folder / f"{path.stem}{suffix}"


def _read_mesh(path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        # Analytic cuboids and massless carriers have no mesh resource.
        return np.zeros((3, 0), dtype=np.float64), np.zeros((3, 0), dtype=np.int64)
    if not path.exists():
        raise FileNotFoundError(f"Missing mesh asset: {path}")
    try:
        if path.suffix.lower() == ".obj":
            # PyVista can also invoke its NumPy bridge while importing OBJ
            # point data (normals/texture coordinates), before .points access.
            import vtk

            reader = vtk.vtkOBJReader()
            reader.SetFileName(str(path))
            reader.Update()
            triangulator = vtk.vtkTriangleFilter()
            triangulator.SetInputData(reader.GetOutput())
            triangulator.PassVertsOff()
            triangulator.PassLinesOff()
            triangulator.Update()
            mesh = triangulator.GetOutput()
        else:
            import pyvista as pv

            mesh = pv.read(str(path)).triangulate()
        # VTK 9.0's NumPy bridge references the removed numpy.bool alias.
        # Native point/cell access preserves VTK's triangulated ordering without
        # depending on that bridge or installing process-global NumPy aliases.
        vertices = np.asarray(
            [mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())],
            dtype=np.float64,
        ).reshape(-1, 3).T
        triangles = []
        for i in range(mesh.GetNumberOfCells()):
            cell = mesh.GetCell(i)
            if cell.GetNumberOfPoints() != 3:
                raise ValueError("triangulated mesh contains a non-triangle cell")
            triangles.append([cell.GetPointId(j) for j in range(3)])
        faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3).T
        if vertices.shape[1] == 0 or faces.shape[1] == 0:
            raise ValueError("mesh has no vertices or triangles")
        if not np.isfinite(vertices).all():
            raise ValueError("mesh contains non-finite vertices")
        return vertices, faces
    except Exception as exc:
        raise ValueError(f"Cannot read mesh asset {path}: {exc}") from exc


def _contact_id_path(contact_path: Path | None) -> Path | None:
    if contact_path is None:
        return None
    return contact_path.with_name(f"{contact_path.stem}_id.npy")


def _read_contact_ids(contact_path: Path | None) -> np.ndarray:
    path = _contact_id_path(contact_path)
    if path is None or not path.exists():
        return np.zeros((0,), dtype=np.int64)
    with open(path, "rb") as stream:
        return np.load(stream)


@dataclass
class LinkRecord:
    link_name: str
    joint_name: str
    body_name: str
    domain: str
    role: str | None
    mask: int
    parent: int | None
    joint_E: np.ndarray
    body_E: np.ndarray
    mesh_path: Path | None
    contact_path: Path | None
    contact_n: int
    p1_slice: slice | None = None
    p2_slice: slice | None = None
    p3_slice: slice | None = None
    p4_slice: slice | None = None
    p5_slice: slice | None = None
    p6_slice: slice | None = None
    abstract_index: int | None = None
    planar_parent_face: int | None = None
    planar_child_face: int | None = None
    planar_dock_id: int | None = None
    planar_child_dock_id: int | None = None
    planar_facing: int | None = None
    planar_parent_barycentric: np.ndarray | None = None
    planar_child_barycentric: np.ndarray | None = None
    planar_endeffector_face: int | None = None
    planar_endeffector_barycentric: np.ndarray | None = None
    node_id: int | None = None
    start_function_group: bool = False
    function_group_root: int | None = None
    function_group_leaves: tuple[int, ...] = ()
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class RenderRecord:
    body_name: str
    mesh_path: Path | None
    source_record: LinkRecord | None = None


class SceneSpec:
    def __init__(self, xml_path: str | os.PathLike[str], sim: Any):
        self.xml_path = Path(xml_path).resolve()
        self.baseline = np.asarray(sim.get_design_params(), dtype=np.float64)
        self.ndof_p = int(sim.ndof_p)
        self.records: list[LinkRecord] = []
        self.finger_records: list[LinkRecord] = []
        self.tool_records: list[LinkRecord] = []
        self.marker_records: list[LinkRecord] = []
        self.abstract_records: list[LinkRecord] = []
        self.render_records: list[RenderRecord] = []
        self._parse()
        self._assign_slices()
        if self.ndof_p != self.total_ndof:
            raise ValueError(
                f"SceneSpec parsed ndof_p={self.total_ndof}, but RedMax sim.ndof_p={self.ndof_p}"
            )
        self.finger_spec = None


    def _parse(self) -> None:
        root = ET.parse(self.xml_path).getroot()
        for robot in root.findall("robot"):
            link = robot.find("link")
            if link is not None:
                self._visit(link, None)

    def _visit(self, link: ET.Element, parent: int | None) -> None:
        body = link.find("body")
        joint = link.find("joint")
        mesh = body.attrib.get("mesh") if body is not None else None
        role = _role_from_mesh(mesh)
        domain = "finger" if _is_finger_mesh(mesh) else "tool"
        if body is None or _mask(link) == 0:
            domain = "other"
        elif _is_marker_link(link, body):
            domain = "marker"
        if body is not None and body.attrib.get("type") == "abstract":
            self.render_records.append(
                RenderRecord(
                    body_name=body.attrib.get("name", ""),
                    mesh_path=_resolve_path(mesh, self.xml_path),
                    source_record=None,
                )
            )
        record = LinkRecord(
            link_name=link.attrib.get("name", ""),
            joint_name=joint.attrib.get("name", "") if joint is not None else "",
            body_name=body.attrib.get("name", "") if body is not None else "",
            domain=domain,
            role=role,
            mask=_mask(link),
            parent=parent,
            joint_E=_E_from_joint(joint),
            body_E=_E_from_body(body),
            mesh_path=_resolve_path(mesh, self.xml_path),
            contact_path=_resolve_path(body.attrib.get("contacts"), self.xml_path) if body is not None else None,
            contact_n=_count_contacts(body.attrib.get("contacts"), self.xml_path) if body is not None else 0,
            planar_parent_face=_parse_optional_int(link.attrib.get("planar_parent_face")),
            planar_child_face=_parse_optional_int(link.attrib.get("planar_child_face")),
            planar_dock_id=_parse_optional_int(link.attrib.get("planar_dock_id")),
            planar_child_dock_id=_parse_optional_int(link.attrib.get("planar_child_dock_id")),
            planar_facing=_parse_optional_int(link.attrib.get("planar_facing")),
            planar_parent_barycentric=_parse_optional_barycentric(link.attrib.get("planar_parent_barycentric")),
            planar_child_barycentric=_parse_optional_barycentric(link.attrib.get("planar_child_barycentric")),
            planar_endeffector_face=_parse_optional_int(link.attrib.get("planar_endeffector_face")),
            planar_endeffector_barycentric=_parse_optional_barycentric(
                link.attrib.get("planar_endeffector_barycentric")
            ),
            node_id=_infer_node_id(link, body),
            start_function_group=_parse_bool(link.attrib.get("start_function_group")),
            function_group_root=_parse_optional_int(link.attrib.get("function_group_root")),
            function_group_leaves=_parse_int_list(link.attrib.get("function_group_leaves")),
            attrs=dict(link.attrib),
        )
        cur_parent = parent
        if body is not None and record.mask:
            self.records.append(record)
            cur_parent = len(self.records) - 1
            if body.attrib.get("type") == "abstract":
                record.abstract_index = len(self.abstract_records)
                self.abstract_records.append(record)
                if self.render_records:
                    self.render_records[-1].source_record = record
            if record.domain == "finger":
                self.finger_records.append(record)
            elif record.domain == "tool":
                self.tool_records.append(record)
            elif record.domain == "marker":
                self.marker_records.append(record)
        for child in link.findall("link"):
            self._visit(child, cur_parent)

    def _assign_slices(self) -> None:
        p1 = p2 = p3 = p4 = p5 = p6 = 0
        for rec in self.records:
            if _has(rec.mask, 0):
                rec.p1_slice = slice(p1, p1 + 12)
                p1 += 12
            if _has(rec.mask, 4):
                rec.p5_slice = slice(p5, p5 + 2)
                p5 += 2
        for rec in self.records:
            if _has(rec.mask, 1):
                rec.p2_slice = slice(p2, p2 + 12)
                p2 += 12
            if _has(rec.mask, 2):
                n = rec.contact_n * 3
                rec.p3_slice = slice(p3, p3 + n)
                p3 += n
            if _has(rec.mask, 3):
                rec.p4_slice = slice(p4, p4 + 4)
                p4 += 4
            if _has(rec.mask, 5):
                rec.p6_slice = slice(p6, p6 + 1)
                p6 += 1
        self.ndof_p1, self.ndof_p2, self.ndof_p3 = p1, p2, p3
        self.ndof_p4, self.ndof_p5, self.ndof_p6 = p4, p5, p6
        self.total_ndof = p1 + p2 + p3 + p4 + p5 + p6
        for rec in self.records:
            if rec.p2_slice is not None:
                rec.p2_slice = slice(self.ndof_p1 + rec.p2_slice.start, self.ndof_p1 + rec.p2_slice.stop)
            if rec.p3_slice is not None:
                off = self.ndof_p1 + self.ndof_p2
                rec.p3_slice = slice(off + rec.p3_slice.start, off + rec.p3_slice.stop)
            if rec.p4_slice is not None:
                off = self.ndof_p1 + self.ndof_p2 + self.ndof_p3
                rec.p4_slice = slice(off + rec.p4_slice.start, off + rec.p4_slice.stop)
            if rec.p5_slice is not None:
                off = self.ndof_p1 + self.ndof_p2 + self.ndof_p3 + self.ndof_p4
                rec.p5_slice = slice(off + rec.p5_slice.start, off + rec.p5_slice.stop)
            if rec.p6_slice is not None:
                off = self.ndof_p1 + self.ndof_p2 + self.ndof_p3 + self.ndof_p4 + self.ndof_p5
                rec.p6_slice = slice(off + rec.p6_slice.start, off + rec.p6_slice.stop)


class _Mesh:
    def __init__(self, V: np.ndarray, F: np.ndarray | None = None):
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.F = (
            np.ascontiguousarray(F, dtype=np.int32)
            if F is not None
            else np.zeros((3, 0), dtype=np.int32)
        )
