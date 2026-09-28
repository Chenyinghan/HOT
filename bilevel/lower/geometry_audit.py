"""Replay-only symmetric overlap audits for rigid collision bodies.

The physical solver is deliberately not changed here.  This module provides
an independent geometry validity signal for task success and search reward.
It checks both A-inside-B and B-inside-A, which catches containment cases that
one-way surface contact sampling can miss.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from bilevel.parameterization.scene import _E_from_body, _E_from_joint, _read_mesh, _resolve_path


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise ValueError("revolute/prismatic joint axis must be nonzero")
    x, y, z = axis / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _joint_ndof(joint_type: str) -> int:
    return {
        "fixed": 0,
        "revolute": 1,
        "prismatic": 1,
        "planar": 2,
        "translational": 3,
        "spherical": 3,
        "spherical-euler": 3,
        "spherical-exp": 3,
        "free2d": 3,
        "free3d": 6,
        "free3d-euler": 6,
        "free3d-exp": 6,
        "free3d-exp-decoupled": 6,
    }.get(str(joint_type).lower(), -1)


def _dynamic_joint_transform(joint: ET.Element | None, values: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if joint is None:
        return transform
    joint_type = str(joint.attrib.get("type", "fixed")).lower()
    if joint_type == "fixed":
        return transform
    if joint_type == "translational":
        transform[:3, 3] = values[:3]
        return transform
    if joint_type in {"prismatic", "revolute"}:
        axis = np.fromstring(joint.attrib.get("axis", ""), sep=" ", dtype=np.float64)
        if axis.shape != (3,):
            raise ValueError(f"joint {joint.attrib.get('name', '')!r} needs a 3D axis")
        if joint_type == "prismatic":
            transform[:3, 3] = axis / np.linalg.norm(axis) * float(values[0])
        else:
            transform[:3, :3] = _axis_angle(axis, float(values[0]))
        return transform
    raise NotImplementedError(
        f"runtime geometry audit does not yet support moving joint type {joint_type!r}"
    )


class _XmlKinematics:
    def __init__(self, xml_path: str | Path) -> None:
        self.xml_path = Path(xml_path).resolve()
        self.root = ET.parse(str(self.xml_path)).getroot()
        self._q_slices: dict[int, slice] = {}
        cursor = 0
        for joint in self.root.iter("joint"):
            ndof = _joint_ndof(joint.attrib.get("type", "fixed"))
            if ndof < 0:
                raise ValueError(
                    f"unknown joint type {joint.attrib.get('type')!r} in {self.xml_path}"
                )
            self._q_slices[id(joint)] = slice(cursor, cursor + ndof)
            cursor += ndof
        self.ndof = cursor

    def body_transforms(self, q: Sequence[float]) -> dict[str, np.ndarray]:
        q_array = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_array.size < self.ndof:
            raise ValueError(
                f"geometry audit q has {q_array.size} values, expected at least {self.ndof}"
            )
        out: dict[str, np.ndarray] = {}

        def visit(link: ET.Element, parent: np.ndarray) -> None:
            joint = link.find("joint")
            static = _E_from_joint(joint)
            q_slice = self._q_slices.get(id(joint), slice(0, 0))
            dynamic = _dynamic_joint_transform(joint, q_array[q_slice])
            link_transform = parent @ static @ dynamic
            for body in link.findall("body"):
                name = str(body.attrib.get("name", ""))
                if name:
                    out[name] = link_transform @ _E_from_body(body)
            for child in link.findall("link"):
                visit(child, link_transform)

        for robot in self.root.findall("robot"):
            for link in robot.findall("link"):
                visit(link, np.eye(4, dtype=np.float64))
        return out


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return values @ transform[:3, :3].T + transform[:3, 3]


def _cuboid_surface(size: np.ndarray, resolution: int = 7) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * np.asarray(size, dtype=np.float64).reshape(3)
    hull = np.asarray(
        [[sx * half[0], sy * half[1], sz * half[2]]
         for sx in (-1.0, 1.0)
         for sy in (-1.0, 1.0)
         for sz in (-1.0, 1.0)],
        dtype=np.float64,
    )
    grid = np.linspace(-1.0, 1.0, max(2, int(resolution)))
    samples: list[list[float]] = []
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        for sign in (-1.0, 1.0):
            for a in grid:
                for b in grid:
                    point = np.zeros(3, dtype=np.float64)
                    point[axis] = sign * half[axis]
                    point[other[0]] = a * half[other[0]]
                    point[other[1]] = b * half[other[1]]
                    samples.append(point.tolist())
    return hull, np.unique(np.asarray(samples, dtype=np.float64), axis=0)


def _mesh_surface(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim == 2 and points.shape[0] == 3:
        points = points.T
    points = points.reshape(-1, 3)
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        hull_points = points[np.unique(hull.simplices)]
        centroids = np.asarray(
            [np.mean(points[simplex], axis=0) for simplex in hull.simplices],
            dtype=np.float64,
        )
        edges = []
        for simplex in hull.simplices:
            ids = [int(value) for value in simplex]
            edges.extend(
                0.5 * (points[a] + points[b])
                for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0]))
            )
        surface = np.unique(
            np.concatenate((hull_points, centroids, np.asarray(edges)), axis=0),
            axis=0,
        )
        return hull_points, surface
    except Exception:
        return points, points


@dataclass(frozen=True)
class _ConvexBody:
    name: str
    q0_transform: np.ndarray
    hull_points_q0: np.ndarray
    surface_points_q0: np.ndarray
    equations_q0: np.ndarray
    min_extent: float


def _convex_body(
    name: str,
    q0_transform: np.ndarray,
    hull_points_q0: np.ndarray,
    surface_points_q0: np.ndarray,
) -> _ConvexBody:
    from scipy.spatial import ConvexHull

    hull_points = np.asarray(hull_points_q0, dtype=np.float64).reshape(-1, 3)
    surface_points = np.asarray(surface_points_q0, dtype=np.float64).reshape(-1, 3)
    hull = ConvexHull(hull_points)
    widths = []
    for equation in hull.equations:
        normal = np.asarray(equation[:3], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm > 1.0e-12:
            widths.append(float(np.ptp(hull_points @ (normal / norm))))
    positive = np.asarray(
        [width for width in widths if width > 1.0e-9],
        dtype=np.float64,
    )
    min_extent = float(np.min(positive)) if positive.size else 1.0
    return _ConvexBody(
        name=name,
        q0_transform=np.asarray(q0_transform, dtype=np.float64),
        hull_points_q0=hull_points,
        surface_points_q0=surface_points,
        equations_q0=np.asarray(hull.equations, dtype=np.float64),
        min_extent=min_extent,
    )


def _xml_convex_bodies(
    xml_path: Path,
    kinematics: _XmlKinematics,
    names: set[str],
) -> dict[str, _ConvexBody]:
    q0_transforms = kinematics.body_transforms(np.zeros(kinematics.ndof))
    root = kinematics.root
    bodies: dict[str, _ConvexBody] = {}
    for body in root.findall(".//body[@name]"):
        name = str(body.attrib.get("name", ""))
        if name not in names or name not in q0_transforms:
            continue
        transform = q0_transforms[name]
        body_type = str(body.attrib.get("type", "")).lower()
        if body_type == "cuboid":
            size = np.fromstring(body.attrib.get("size", ""), sep=" ", dtype=np.float64)
            if size.shape != (3,):
                continue
            hull_local, surface_local = _cuboid_surface(size)
        elif body_type == "abstract":
            vertices, _ = _read_mesh(_resolve_path(body.attrib.get("mesh"), xml_path))
            hull_local, surface_local = _mesh_surface(vertices)
        else:
            continue
        bodies[name] = _convex_body(
            name,
            transform,
            _transform_points(transform, hull_local),
            _transform_points(transform, surface_local),
        )
    return bodies


def _replace_tool_geometry(
    bodies: dict[str, _ConvexBody],
    *,
    bundle: Any,
    design_params: np.ndarray,
    tool_names: set[str],
) -> None:
    # Reuse the common morphology geometry resolver so optimized joint/body
    # transforms and cage vertices are represented exactly as in validation.
    from bilevel.parameterization.collision import _body_geometries

    resolved = _body_geometries(bundle, np.asarray(design_params, dtype=np.float64))
    for name in tool_names:
        geometry = resolved.get(name)
        previous = bodies.get(name)
        if geometry is None or previous is None:
            continue
        hull, surface = _mesh_surface(geometry.vertices)
        bodies[name] = _convex_body(
            name,
            previous.q0_transform,
            hull,
            surface,
        )


def _inside_measure(points: np.ndarray, target: _ConvexBody) -> tuple[float, float]:
    values = points @ target.equations_q0[:, :3].T + target.equations_q0[:, 3]
    signed = np.max(values, axis=1)
    depths = np.maximum(-signed, 0.0)
    return float(np.mean(depths > 1.0e-6)), float(np.max(depths, initial=0.0))


class SymmetricOverlapAudit:
    """Accumulate bidirectional convex-containment metrics during a replay."""

    def __init__(
        self,
        xml_path: str | Path,
        *,
        tool_body_names: Iterable[str],
        operated_body_names: Iterable[str],
        bundle: Any = None,
        design_params: np.ndarray | None = None,
        containment_fraction_limit: float = 0.2,
        normalized_depth_limit: float = 1.0,
        sample_interval_seconds: float = 0.1,
        min_sustained_seconds: float = 0.2,
    ) -> None:
        self.xml_path = Path(xml_path).resolve()
        self.tool_names = set(map(str, tool_body_names))
        self.operated_names = set(map(str, operated_body_names))
        if not 0.0 < float(containment_fraction_limit) <= 1.0:
            raise ValueError("containment_fraction_limit must lie in (0, 1]")
        if float(normalized_depth_limit) <= 0.0:
            raise ValueError("normalized_depth_limit must be positive")
        if float(sample_interval_seconds) <= 0.0:
            raise ValueError("sample_interval_seconds must be positive")
        if float(min_sustained_seconds) <= 0.0:
            raise ValueError("min_sustained_seconds must be positive")
        self.fraction_limit = float(containment_fraction_limit)
        self.depth_limit = float(normalized_depth_limit)
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.min_sustained_seconds = float(min_sustained_seconds)
        self.min_consecutive = max(
            1,
            int(
                math.ceil(
                    self.min_sustained_seconds
                    / self.sample_interval_seconds
                    - 1.0e-12
                )
            ),
        )
        self.kinematics = _XmlKinematics(self.xml_path)
        requested = self.tool_names | self.operated_names
        self.bodies = _xml_convex_bodies(self.xml_path, self.kinematics, requested)
        if bundle is not None and design_params is not None:
            _replace_tool_geometry(
                self.bodies,
                bundle=bundle,
                design_params=np.asarray(design_params, dtype=np.float64),
                tool_names=self.tool_names,
            )
        missing = sorted(requested - set(self.bodies))
        if missing:
            raise ValueError(f"geometry audit could not resolve convex bodies: {missing}")
        self._q0_transforms = {
            name: body.q0_transform for name, body in self.bodies.items()
        }
        self._max_fraction = 0.0
        self._max_normalized_depth = 0.0
        self._max_severity = 0.0
        self._worst: dict[str, Any] | None = None
        self._current_run = 0
        self._longest_run = 0
        self._samples = 0
        self._severe_samples = 0

    def observe(self, q: Sequence[float], *, step: int | None = None) -> None:
        transforms = self.kinematics.body_transforms(q)
        deltas = {
            name: transforms[name] @ np.linalg.inv(self._q0_transforms[name])
            for name in self.bodies
        }
        sample_severe = False
        for tool_name in sorted(self.tool_names):
            tool = self.bodies[tool_name]
            for object_name in sorted(self.operated_names):
                operated = self.bodies[object_name]
                # Map each source surface into the target's q=0 frame.  This
                # avoids rebuilding convex hull equations at every replay step.
                object_world = _transform_points(deltas[object_name], operated.surface_points_q0)
                object_in_tool_frame = _transform_points(np.linalg.inv(deltas[tool_name]), object_world)
                object_fraction, object_depth = _inside_measure(object_in_tool_frame, tool)

                tool_world = _transform_points(deltas[tool_name], tool.surface_points_q0)
                tool_in_object_frame = _transform_points(np.linalg.inv(deltas[object_name]), tool_world)
                tool_fraction, tool_depth = _inside_measure(tool_in_object_frame, operated)

                directional = (
                    ("operated_in_tool", object_fraction, object_depth, operated.min_extent),
                    ("tool_in_operated", tool_fraction, tool_depth, tool.min_extent),
                )
                for direction, fraction, depth, source_extent in directional:
                    normalized_depth = float(depth / max(source_extent, 1.0e-9))
                    severity = float(fraction * normalized_depth)
                    severe = bool(
                        fraction >= self.fraction_limit
                        and normalized_depth >= self.depth_limit
                    )
                    sample_severe = sample_severe or severe
                    self._max_fraction = max(self._max_fraction, fraction)
                    self._max_normalized_depth = max(
                        self._max_normalized_depth, normalized_depth
                    )
                    if severity > self._max_severity:
                        self._max_severity = severity
                        self._worst = {
                            "tool_body": tool_name,
                            "operated_body": object_name,
                            "direction": direction,
                            "containment_fraction": fraction,
                            "penetration_depth": depth,
                            "normalized_depth": normalized_depth,
                            "severity": severity,
                            "step": None if step is None else int(step),
                        }
        self._samples += 1
        if sample_severe:
            self._severe_samples += 1
            self._current_run += 1
            self._longest_run = max(self._longest_run, self._current_run)
        else:
            self._current_run = 0

    def summary(self) -> dict[str, Any]:
        severe = bool(self._longest_run >= self.min_consecutive)
        return {
            "enabled": True,
            "ok": not severe,
            "severe_overlap": severe,
            "sample_count": int(self._samples),
            "severe_sample_count": int(self._severe_samples),
            "longest_consecutive_severe_samples": int(self._longest_run),
            "longest_sustained_severe_seconds": float(
                self._longest_run * self.sample_interval_seconds
            ),
            "required_consecutive_samples": int(self.min_consecutive),
            "sample_interval_seconds": float(self.sample_interval_seconds),
            "min_sustained_seconds": float(self.min_sustained_seconds),
            "containment_fraction_limit": float(self.fraction_limit),
            "normalized_depth_limit": float(self.depth_limit),
            "max_containment_fraction": float(self._max_fraction),
            "max_normalized_depth": float(self._max_normalized_depth),
            "max_severity": float(self._max_severity),
            "worst_pair": self._worst,
            "method": "symmetric_convex_surface_containment",
        }


__all__ = ["SymmetricOverlapAudit"]
