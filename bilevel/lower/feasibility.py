"""Reject invalid fixed-root morphology before lower-level optimization."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from bilevel.parameterization.scene import _E_from_body, _E_from_joint, _read_mesh, _resolve_path


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Mesh vertices must be a matrix")
    if values.shape[0] == 3:
        values = values.T
    return values @ transform[:3, :3].T + transform[:3, 3]


def _vector(element: ET.Element, name: str, default: str) -> np.ndarray:
    value = np.fromstring(element.attrib.get(name, default), sep=" ", dtype=np.float64)
    if value.size != 3 or not np.all(np.isfinite(value)):
        raise ValueError(f"{element.tag}.{name} must contain three finite values")
    return value


def _read_vertices(mesh_path: Path) -> np.ndarray:
    """Read mesh vertices even when the optional pyvista dependency is absent."""

    vertices, _ = _read_mesh(mesh_path)
    local = np.asarray(vertices, dtype=np.float64)
    if local.size:
        return local.T if local.shape[0] == 3 else local
    obj_vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            values = line.split()
            if len(values) >= 4:
                obj_vertices.append(
                    [float(values[1]), float(values[2]), float(values[3])]
                )
    if not obj_vertices:
        raise ValueError(f"Mesh contains no readable vertices: {mesh_path}")
    return np.asarray(obj_vertices, dtype=np.float64)


def check_initial_feasibility(
    xml_path: str | Path,
    *,
    ground_tolerance: float = 1e-6,
    sphere_tolerance: float = 1e-6,
) -> dict:
    """Check initial Head vertices against ground and task spheres.

    This deliberately walks every fixed XML ancestor, so the result is in the
    same world frame as the rendered scene.  Sphere tests are exact for the
    current cube-only Head library: the sphere center is transformed into each
    rigid body's local frame and tested against that mesh's local AABB.
    """

    path = Path(xml_path).resolve()
    root = ET.parse(path).getroot()
    ground = root.find("ground")
    ground_plane = None
    if ground is not None:
        normal = _vector(ground, "normal", "0 0 1")
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError("Ground normal must be nonzero")
        ground_plane = (normal / norm, _vector(ground, "pos", "0 0 0"))

    heads: list[dict] = []
    spheres: list[dict] = []

    def visit(link: ET.Element, parent_E: np.ndarray) -> None:
        link_E = parent_E @ _E_from_joint(link.find("joint"))
        body = link.find("body")
        if body is not None:
            body_E = link_E @ _E_from_body(body)
            body_name = str(body.attrib.get("name", ""))
            is_head = (
                link.attrib.get("asset_role") == "head"
                or body.attrib.get("asset_role") == "head"
                or str(link.attrib.get("name", "")).startswith("link_tool_")
            )
            if is_head:
                mesh_path = _resolve_path(body.attrib.get("mesh"), path)
                local = _read_vertices(mesh_path)
                heads.append(
                    {
                        "link": str(link.attrib.get("name", "")),
                        "body": body_name,
                        "transform": body_E,
                        "local_vertices": local,
                        "world_vertices": _transform_points(body_E, local),
                    }
                )
            elif body.attrib.get("type") == "sphere":
                spheres.append(
                    {
                        "body": body_name,
                        "center": body_E[:3, 3].copy(),
                        "radius": float(body.attrib.get("radius", "0") or 0.0),
                    }
                )
        for child in link.findall("link"):
            visit(child, link_E)

    identity = np.eye(4, dtype=np.float64)
    for robot in root.findall("robot"):
        for link in robot.findall("link"):
            visit(link, identity)

    ground_violations = []
    minimum_ground_clearance = float("inf")
    if ground_plane is not None:
        normal, position = ground_plane
        for head in heads:
            clearances = (head["world_vertices"] - position.reshape(1, 3)) @ normal
            clearance = float(np.min(clearances))
            minimum_ground_clearance = min(minimum_ground_clearance, clearance)
            if clearance < -abs(float(ground_tolerance)):
                ground_violations.append(
                    {
                        "link": head["link"],
                        "body": head["body"],
                        "min_signed_clearance": clearance,
                        "penetration": -clearance,
                    }
                )

    sphere_violations = []
    minimum_sphere_clearance = float("inf")
    for head in heads:
        transform = head["transform"]
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        lo = np.min(head["local_vertices"], axis=0)
        hi = np.max(head["local_vertices"], axis=0)
        for sphere in spheres:
            local_center = rotation.T @ (sphere["center"] - translation)
            closest = np.minimum(np.maximum(local_center, lo), hi)
            distance = float(np.linalg.norm(local_center - closest))
            clearance = distance - float(sphere["radius"])
            minimum_sphere_clearance = min(minimum_sphere_clearance, clearance)
            if clearance < -abs(float(sphere_tolerance)):
                sphere_violations.append(
                    {
                        "head_link": head["link"],
                        "head_body": head["body"],
                        "sphere_body": sphere["body"],
                        "signed_clearance": clearance,
                        "penetration": -clearance,
                    }
                )

    return {
        "ok": not ground_violations and not sphere_violations,
        "head_count": len(heads),
        "sphere_count": len(spheres),
        "minimum_ground_clearance": (
            None if not np.isfinite(minimum_ground_clearance) else minimum_ground_clearance
        ),
        "minimum_sphere_clearance": (
            None if not np.isfinite(minimum_sphere_clearance) else minimum_sphere_clearance
        ),
        "ground_violations": ground_violations,
        "sphere_violations": sphere_violations,
    }


__all__ = ["check_initial_feasibility"]
