from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bilevel.parameterization.scene import (
    LinkRecord,
    _E_from_body,
    _E_from_joint,
    _read_mesh,
    _resolve_path,
)


@dataclass
class BodyGeometry:
    name: str
    domain: str
    vertices: np.ndarray
    hull_vertices: np.ndarray | None = None
    radius: float | None = None
    center: np.ndarray | None = None

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        if self.vertices.size == 0:
            c = np.zeros(3, dtype=np.float64) if self.center is None else self.center
            return c.copy(), c.copy()
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)


@dataclass
class CollisionReport:
    ok: bool
    overlaps: list[dict[str, Any]]
    ground_violations: list[dict[str, Any]]
    checked_pairs: int
    margin: float

    def first_issue(self) -> dict[str, Any] | None:
        if self.overlaps:
            return self.overlaps[0]
        if self.ground_violations:
            return self.ground_violations[0]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "overlaps": self.overlaps,
            "ground_violations": self.ground_violations,
            "checked_pairs": int(self.checked_pairs),
            "margin": float(self.margin),
        }


def _E_from_flat(flat: np.ndarray) -> np.ndarray:
    E = np.eye(4, dtype=np.float64)
    values = np.asarray(flat, dtype=np.float64).reshape(12)
    E[:3, :3] = values[:9].reshape(3, 3)
    E[:3, 3] = values[9:12]
    return E


def _transform_points(E: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if pts.shape[0] == 3:
        pts = pts.T
    return pts @ E[:3, :3].T + E[:3, 3]


def _sphere_vertices(center: np.ndarray, radius: float) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r = float(radius)
    return np.asarray(
        list(itertools.product((-r, r), repeat=3)),
        dtype=np.float64,
    ) + c.reshape(1, 3)


def _xml_body_geometries(xml_path: str | Path) -> dict[str, BodyGeometry]:
    xml_path = Path(xml_path).resolve()
    root = ET.parse(xml_path).getroot()
    bodies: dict[str, BodyGeometry] = {}

    def visit(link: ET.Element, parent_E: np.ndarray) -> None:
        joint_E = _E_from_joint(link.find("joint"))
        link_E = parent_E @ joint_E
        body = link.find("body")
        if body is not None:
            body_E = link_E @ _E_from_body(body)
            name = body.attrib.get("name", "")
            body_type = body.attrib.get("type", "")
            if name:
                if body_type == "abstract":
                    V, _ = _read_mesh(_resolve_path(body.attrib.get("mesh"), xml_path))
                    vertices = _transform_points(body_E, V)
                    bodies[name] = BodyGeometry(name=name, domain="other", vertices=vertices, hull_vertices=vertices)
                elif body_type == "sphere":
                    center = body_E[:3, 3].copy()
                    radius = float(body.attrib.get("radius", 0.0) or 0.0)
                    bodies[name] = BodyGeometry(
                        name=name,
                        domain="other",
                        vertices=_sphere_vertices(center, radius),
                        radius=radius,
                        center=center,
                    )
        for child in link.findall("link"):
            visit(child, link_E)

    for robot in root.findall("robot"):
        for link in robot.findall("link"):
            visit(link, np.eye(4, dtype=np.float64))
    return bodies


def _xml_link_transforms(xml_path: str | Path) -> dict[str, np.ndarray]:
    root = ET.parse(Path(xml_path).resolve()).getroot()
    transforms: dict[str, np.ndarray] = {}

    def visit(link: ET.Element, parent_E: np.ndarray) -> None:
        link_E = parent_E @ _E_from_joint(link.find("joint"))
        body = link.find("body")
        if body is not None and body.attrib.get("name"):
            transforms[body.attrib["name"]] = link_E
        for child in link.findall("link"):
            visit(child, link_E)

    for robot in root.findall("robot"):
        for link in robot.findall("link"):
            visit(link, np.eye(4, dtype=np.float64))
    return transforms


def _record_transforms(
    spec: Any,
    design_params: np.ndarray,
    nominal_link_transforms: dict[str, np.ndarray] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Resolve design records in the complete XML ancestor frame.

    Fixed roots and other design_params=0 links are intentionally absent from
    the reduced design-record tree. Walking the XML preserves those static
    transforms for every task instead of requiring a Handle-specific patch.
    """

    design_params = np.asarray(design_params, dtype=np.float64)
    xml_path = getattr(spec, "xml_path", None)
    if xml_path is None:
        nominal_link_transforms = nominal_link_transforms or {}
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        link_world_by_idx: dict[int, np.ndarray] = {}
        for idx, record in enumerate(spec.records):
            parent_E = (
                link_world_by_idx.get(
                    int(record.parent),
                    np.eye(4, dtype=np.float64),
                )
                if record.parent is not None
                else np.eye(4, dtype=np.float64)
            )
            nominal_link_E = nominal_link_transforms.get(record.body_name)
            if nominal_link_E is not None:
                nominal_parent_E = np.eye(4, dtype=np.float64)
                if record.parent is not None:
                    parent_record = spec.records[int(record.parent)]
                    nominal_parent_E = nominal_link_transforms.get(
                        parent_record.body_name,
                        nominal_parent_E,
                    )
                nominal_relative_E = (
                    np.linalg.inv(nominal_parent_E) @ nominal_link_E
                )
                parent_E = (
                    parent_E
                    @ nominal_relative_E
                    @ np.linalg.inv(record.joint_E)
                )
            joint_E = (
                _E_from_flat(design_params[record.p1_slice])
                if record.p1_slice is not None
                else record.joint_E
            )
            link_E = parent_E @ joint_E
            body_E = (
                _E_from_flat(design_params[record.p2_slice])
                if record.p2_slice is not None
                else record.body_E
            )
            link_world_by_idx[idx] = link_E
            if record.body_name:
                out[record.body_name] = (link_E, body_E)
        return out
    records_by_name = {
        str(record.link_name): record
        for record in spec.records
        if str(record.link_name)
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def visit(link: ET.Element, parent_E: np.ndarray) -> None:
        record = records_by_name.get(str(link.attrib.get("name", "")))
        if record is None:
            link_E = parent_E @ _E_from_joint(link.find("joint"))
        else:
            joint_E = (
                _E_from_flat(design_params[record.p1_slice])
                if record.p1_slice is not None
                else record.joint_E
            )
            link_E = parent_E @ joint_E
            body_E = (
                _E_from_flat(design_params[record.p2_slice])
                if record.p2_slice is not None
                else record.body_E
            )
            if record.body_name:
                out[record.body_name] = (link_E, body_E)
        for child in link.findall("link"):
            visit(child, link_E)

    root = ET.parse(str(xml_path)).getroot()
    identity = np.eye(4, dtype=np.float64)
    for robot in root.findall("robot"):
        for link in robot.findall("link"):
            visit(link, identity)
    return out


def _tool_hull_vertices(bundle: Any, rec: LinkRecord, body_E: np.ndarray) -> np.ndarray | None:
    design_np = getattr(bundle, "design_np", None)
    tool_index = getattr(design_np, "tool_index", {})
    cages = getattr(design_np, "tool_cages", [])
    idx = tool_index.get(id(rec), None)
    if idx is None or idx < 0 or idx >= len(cages):
        return None
    cage = cages[idx]
    vertices = getattr(cage, "vertices", None)
    if vertices is None:
        return None
    return _transform_points(body_E, np.asarray(vertices, dtype=np.float64))


def _body_geometries(bundle: Any, design_params: np.ndarray) -> dict[str, BodyGeometry]:
    xml_path = getattr(bundle, "model_path", None) or getattr(getattr(bundle, "spec", None), "xml_path", None)
    bodies = _xml_body_geometries(xml_path) if xml_path is not None else {}
    spec = bundle.spec
    nominal_link_transforms = _xml_link_transforms(xml_path) if xml_path is not None else {}
    transforms = _record_transforms(spec, design_params, nominal_link_transforms)
    for rec in spec.records:
        if not rec.body_name or rec.body_name not in transforms:
            continue
        link_E, body_local_E = transforms[rec.body_name]
        body_E = link_E @ body_local_E
        if rec.domain == "tool":
            hull = _tool_hull_vertices(bundle, rec, body_E)
            if hull is not None and hull.size:
                vertices = hull
            else:
                V, _ = _read_mesh(rec.mesh_path)
                vertices = _transform_points(body_E, V)
                hull = vertices
        else:
            V, _ = _read_mesh(rec.mesh_path)
            vertices = _transform_points(body_E, V)
            hull = vertices
        bodies[rec.body_name] = BodyGeometry(
            name=rec.body_name,
            domain=rec.domain,
            vertices=vertices,
            hull_vertices=hull,
        )
    return bodies


def _design_contact_pairs(xml_path: str | Path, tool_body_names: set[str]) -> tuple[list[tuple[str, str, str]], set[str]]:
    root = ET.parse(str(xml_path)).getroot()
    contact = root.find("contact")
    pairs: list[tuple[str, str, str]] = []
    ground_bodies: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    seen_body_pairs: set[tuple[str, str]] = set()
    for a, b in itertools.combinations(sorted(tool_body_names), 2):
        key = ("tool_tool_structural", a, b)
        if key not in seen:
            seen.add(key)
            seen_body_pairs.add((a, b))
            pairs.append((a, b, "tool_tool_structural"))
    if contact is None:
        return pairs, ground_bodies
    for elem in contact:
        if elem.tag in {"general_contact", "collision_constraint"}:
            a = elem.attrib.get("body1")
            b = elem.attrib.get("body2")
            kind = elem.tag
        elif elem.tag == "general_primitive_contact":
            a = elem.attrib.get("general_body")
            b = elem.attrib.get("primitive_body")
            kind = "general_primitive_contact"
        elif elem.tag == "ground_contact":
            body = elem.attrib.get("body")
            if body in tool_body_names:
                ground_bodies.add(body)
            continue
        else:
            continue
        if not a or not b:
            continue
        if a not in tool_body_names and b not in tool_body_names:
            continue
        body_pair = (a, b) if a <= b else (b, a)
        if body_pair in seen_body_pairs:
            continue
        key = (kind, a, b) if a <= b else (kind, b, a)
        if key in seen:
            continue
        seen.add(key)
        seen_body_pairs.add(body_pair)
        pairs.append((a, b, kind))
    return pairs, ground_bodies


def _ground_plane(xml_path: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the XML ground plane as normalized ``(normal, position)``."""
    root = ET.parse(str(xml_path)).getroot()
    elem = root.find("ground")
    if elem is None:
        elem = root.find(".//ground")
    if elem is None:
        return None

    def vector(name: str, default: str) -> np.ndarray:
        values = np.fromstring(elem.attrib.get(name, default), sep=" ", dtype=np.float64)
        if values.size != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"Ground {name!r} must contain three finite values")
        return values

    normal = vector("normal", "0 0 1")
    position = vector("pos", "0 0 0")
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("Ground normal must be nonzero")
    return normal / norm, position


def _ground_violations(
    bodies: dict[str, BodyGeometry],
    tool_body_names: set[str],
    ground_plane: tuple[np.ndarray, np.ndarray] | None,
    margin: float,
) -> list[dict[str, Any]]:
    if ground_plane is None:
        return []
    ground_normal, ground_position = ground_plane
    violations: list[dict[str, Any]] = []
    for body_name in sorted(tool_body_names):
        body = bodies.get(body_name)
        if body is None or body.vertices.size == 0:
            continue
        signed_clearances = (body.vertices - ground_position.reshape(1, 3)) @ ground_normal
        min_clearance = float(np.min(signed_clearances))
        if min_clearance < -float(margin):
            violations.append(
                {
                    "body": body_name,
                    "min_signed_clearance": min_clearance,
                    "penetration": float(-min_clearance),
                    "ground_normal": [float(v) for v in ground_normal],
                    "ground_position": [float(v) for v in ground_position],
                }
            )
    return violations


def _aabb_overlap(a: BodyGeometry, b: BodyGeometry, margin: float) -> tuple[bool, np.ndarray]:
    lo_a, hi_a = a.aabb
    lo_b, hi_b = b.aabb
    overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
    return bool(np.all(overlap > margin)), overlap


def _unique_axes_from_hull(vertices: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    axes: list[np.ndarray] = []
    edges: set[tuple[int, int]] = set()
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(vertices)
        for eq in hull.equations:
            n = np.asarray(eq[:3], dtype=np.float64)
            norm = float(np.linalg.norm(n))
            if norm > 1e-12:
                axes.append(n / norm)
        for simplex in hull.simplices:
            ids = [int(v) for v in simplex]
            for i, j in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
                edges.add(tuple(sorted((i, j))))
    except Exception:
        return [], []
    return axes, sorted(edges)


def _sat_penetration(a_vertices: np.ndarray, b_vertices: np.ndarray, margin: float) -> tuple[bool, float]:
    a = np.asarray(a_vertices, dtype=np.float64)
    b = np.asarray(b_vertices, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 4 or b.shape[0] < 4:
        return True, float("inf")
    axes_a, edges_a = _unique_axes_from_hull(a)
    axes_b, edges_b = _unique_axes_from_hull(b)
    axes = axes_a + axes_b
    for ia, ja in edges_a:
        ea = a[ja] - a[ia]
        for ib, jb in edges_b:
            eb = b[jb] - b[ib]
            axis = np.cross(ea, eb)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-10:
                axes.append(axis / norm)
    if not axes:
        return True, float("inf")
    min_overlap = float("inf")
    for axis in axes:
        pa = a @ axis
        pb = b @ axis
        overlap = min(float(np.max(pa)), float(np.max(pb))) - max(float(np.min(pa)), float(np.min(pb)))
        if overlap <= margin:
            return False, overlap
        min_overlap = min(min_overlap, overlap)
    return True, min_overlap


def _pair_collision(a: BodyGeometry, b: BodyGeometry, margin: float) -> tuple[bool, np.ndarray, float, str]:
    aabb_hit, overlap = _aabb_overlap(a, b, margin)
    if not aabb_hit:
        return False, overlap, 0.0, "aabb"
    if a.hull_vertices is not None and b.hull_vertices is not None:
        hit, depth = _sat_penetration(a.hull_vertices, b.hull_vertices, margin)
        return hit, overlap, float(depth), "sat"
    return True, overlap, float(np.min(overlap)), "aabb"


def check_design_collision(
    cage_params: np.ndarray,
    bundle: Any,
    *,
    xml_path: str | Path | None = None,
    margin: float = 1e-4,
    max_report: int = 8,
    check_ground: bool = True,
) -> CollisionReport:
    xml_path = xml_path or getattr(bundle, "model_path", None) or getattr(getattr(bundle, "spec", None), "xml_path", None)
    if xml_path is None:
        return CollisionReport(ok=True, overlaps=[], ground_violations=[], checked_pairs=0, margin=float(margin))

    design_params = np.asarray(bundle.design_np.parameterize(np.asarray(cage_params, dtype=np.float64), generate_mesh=False), dtype=np.float64)
    return check_design_params_collision(
        design_params,
        bundle,
        xml_path=xml_path,
        margin=margin,
        max_report=max_report,
        check_ground=check_ground,
    )


def check_design_params_collision(
    design_params: np.ndarray,
    bundle: Any,
    *,
    xml_path: str | Path | None = None,
    margin: float = 1e-4,
    max_report: int = 8,
    check_ground: bool = True,
) -> CollisionReport:
    """Check collision from an already evaluated morphology forward pass."""

    xml_path = (
        xml_path
        or getattr(bundle, "model_path", None)
        or getattr(getattr(bundle, "spec", None), "xml_path", None)
    )
    if xml_path is None:
        return CollisionReport(
            ok=True,
            overlaps=[],
            ground_violations=[],
            checked_pairs=0,
            margin=float(margin),
        )
    design_params = np.asarray(design_params, dtype=np.float64)
    bodies = _body_geometries(bundle, design_params)
    tool_body_names = {rec.body_name for rec in bundle.spec.tool_records if rec.body_name}
    pairs, ground_contact_bodies = _design_contact_pairs(xml_path, tool_body_names)

    overlaps: list[dict[str, Any]] = []
    checked = 0
    for body_a, body_b, kind in pairs:
        a = bodies.get(body_a)
        b = bodies.get(body_b)
        if a is None or b is None or a.vertices.size == 0 or b.vertices.size == 0:
            continue
        checked += 1
        hit, overlap, depth, method = _pair_collision(a, b, float(margin))
        if not hit:
            continue
        overlaps.append(
            {
                "body1": body_a,
                "body2": body_b,
                "kind": kind,
                "method": method,
                "overlap": [float(v) for v in overlap],
                "penetration": float(depth),
                "domain1": a.domain,
                "domain2": b.domain,
            }
        )

    ground_violations: list[dict[str, Any]] = []
    if check_ground:
        ground_plane = _ground_plane(xml_path)
        if ground_plane is None and ground_contact_bodies:
            raise ValueError("XML declares ground_contact entries but has no ground plane")
        ground_violations = _ground_violations(bodies, tool_body_names, ground_plane, float(margin))

    overlaps.sort(key=lambda item: float(item.get("penetration", 0.0)), reverse=True)
    ground_violations.sort(key=lambda item: float(item.get("penetration", 0.0)), reverse=True)
    overlaps = overlaps[: int(max_report)]
    ground_violations = ground_violations[: int(max_report)]
    return CollisionReport(
        ok=not overlaps and not ground_violations,
        overlaps=overlaps,
        ground_violations=ground_violations,
        checked_pairs=checked,
        margin=float(margin),
    )
