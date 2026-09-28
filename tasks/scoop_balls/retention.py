"""Sphere range checks in the current function-group cuboid.

The envelope follows the function-group root body's axes and encloses all of
its deformed cages. It is independent of reference names, panel count and
asset dimensions. It supplies the Lift trajectory carry gate and detailed endpoint diagnostics.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np


def sphere_cuboid_gate(balls, radius, frame, lower, upper, required):
    """Require complete spheres inside the closed cuboid (1e-6 roundoff)."""
    balls = np.asarray(balls, dtype=np.float64).reshape(-1, 3)
    frame = np.asarray(frame, dtype=np.float64).reshape(4, 4)
    lower = np.asarray(lower, dtype=np.float64).reshape(3)
    upper = np.asarray(upper, dtype=np.float64).reshape(3)
    if (not np.all(np.isfinite(np.concatenate((frame.ravel(), lower, upper))))
            or np.any(upper <= lower) or not np.isfinite(radius) or radius <= 0
            or required <= 0):
        raise ValueError('Scoop containment requires finite, nondegenerate geometry')
    if not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-8):
        raise ValueError('Scoop containment frame must be rigid')
    local = (balls - frame[:3, 3]) @ frame[:3, :3]
    margins = np.minimum(local - lower - radius, upper - local - radius)
    inside = np.all(np.isfinite(margins) & (margins >= -1e-6), axis=1)
    return {
        'accepted': bool(np.count_nonzero(inside) >= required),
        'evaluation': 'terminal_full_sphere_in_function_group_cuboid',
        'required_ball_count': int(required),
        'contained_count': int(np.count_nonzero(inside)),
        'payload_inside': inside.tolist(),
        'payload_local_positions': local.tolist(),
        'payload_boundary_margin': np.min(margins, axis=1).tolist(),
        'ball_radius': float(radius),
        'cuboid_lower': lower.tolist(),
        'cuboid_upper': upper.tolist(),
        'cuboid_frame_world': frame.tolist(),
        'numerical_tolerance': 1e-6,
    }


def function_group_cuboid(task, runner):
    """Build the candidate's applied public-function envelope once."""
    from bilevel.parameterization.collision import (
        _record_transforms, _transform_points, _xml_link_transforms,
    )

    bundle = runner.design_bundle
    if bundle is None:
        # Action-only runners keep the compiled XML geometry without a design
        # bundle. Construct its cage description for inspection only; do not
        # apply it or change the simulated state.
        from bilevel.parameterization import build_design_bundle
        bundle = build_design_bundle(runner.model_path, runner.sim, task._task_config)
    root = ET.parse(runner.model_path).getroot()
    names = task._function_contact_bodies(root)
    if not names:
        raise ValueError('Scoop containment requires a public function group')
    design = np.asarray(runner.sim.get_design_params(), dtype=np.float64)
    transforms = _record_transforms(bundle.spec, design)
    records = {rec.body_name: rec for rec in bundle.spec.records}
    # XML traversal returns the function-group root before its descendants.
    root_link, root_body = transforms[names[0]]
    nominal_frame = root_link @ root_body
    points = []
    for name in names:
        rec = records[name]
        link_frame, body_frame = transforms[name]
        index = bundle.design_np.tool_index.get(id(rec))
        if index is None:
            raise ValueError('Scoop containment requires current deformed cages')
        cage = bundle.design_np.tool_cages[index]
        vertices = np.asarray(cage.vertices, dtype=np.float64).T
        if rec.p3_slice is not None:
            # Recover the eight cage vertices from the *applied* LBS contacts.
            # Unified morphology can use a different forward object from the
            # legacy bundle; its cached cage must never decide containment.
            contacts = design[rec.p3_slice].reshape(-1, 3)
            weights = np.asarray(cage.contact_weights, dtype=np.float64)
            if weights.shape != (len(contacts), 8):
                raise ValueError('Scoop containment found stale contact weights')
            vertices, _, rank, _ = np.linalg.lstsq(weights, contacts, rcond=None)
            if rank != 8 or not np.allclose(weights @ vertices, contacts, rtol=0, atol=1e-6):
                raise ValueError('Scoop containment cannot recover the applied LBS cage')
        points.append(_transform_points(link_frame @ body_frame, vertices))
    local = (np.concatenate(points) - nominal_frame[:3, 3]) @ nominal_frame[:3, :3]

    # The supported action API is a translational root, one selected revolute
    # axis, and a rigid connected Head. Preserve the full XML ancestor frames.
    nominal_links = _xml_link_transforms(runner.model_path)
    by_joint = {link.find('joint').get('name'): link
                for link in root.iter('link') if link.find('joint') is not None}
    translation_link = by_joint['freeform_root_joint']
    rotation_link = by_joint[task._root_rotation_joint_name]
    for link in rotation_link.iter('link'):
        joint = link.find('joint')
        if link is not rotation_link and joint is not None and joint.get('type') != 'fixed':
            raise ValueError('Scoop containment requires a rigid connected Head')
    translation_frame = nominal_links[translation_link.find('body').get('name')]
    rotation_frame = nominal_links[rotation_link.find('body').get('name')]
    axis = np.fromstring(rotation_link.find('joint').get('axis'), sep=' ')
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 0:
        raise ValueError('Scoop containment requires a finite rotation axis')
    axis = rotation_frame[:3, :3] @ (axis / np.linalg.norm(axis))
    return {
        "nominal_frame": nominal_frame,
        "lower": local.min(axis=0),
        "upper": local.max(axis=0),
        "translation_basis": translation_frame[:3, :3],
        "rotation_pivot": rotation_frame[:3, 3],
        "rotation_axis": axis,
        "function_body_count": len(names),
    }


def current_function_group_frame(task, runner, cuboid):
    """Return the public-function envelope frame at the current Handle pose."""
    q = np.asarray(runner.sim.get_q(), dtype=np.float64)
    angle = float(q[task._q_pitch])
    x, y, z = cuboid["rotation_axis"]
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    nominal_frame = cuboid["nominal_frame"]
    displacement = cuboid["translation_basis"] @ q[task._q_root_translation]
    pivot = cuboid["rotation_pivot"]
    frame = np.eye(4)
    frame[:3, :3] = rotation @ nominal_frame[:3, :3]
    frame[:3, 3] = pivot + displacement + rotation @ (nominal_frame[:3, 3] - pivot)
    return frame


def current_cuboid_gate(task, runner, balls, cuboid):
    """Test complete payload spheres against the current public envelope."""
    result = sphere_cuboid_gate(
        balls, task._ball_radius,
        current_function_group_frame(task, runner, cuboid),
        cuboid["lower"], cuboid["upper"], task._required_ball_count,
    )
    result["function_body_count"] = cuboid["function_body_count"]
    return result


def terminal_cuboid_gate(task, runner, balls):
    """Use applied design cages/transforms and the actual terminal Handle q."""
    return current_cuboid_gate(
        task, runner, balls, function_group_cuboid(task, runner)
    )
