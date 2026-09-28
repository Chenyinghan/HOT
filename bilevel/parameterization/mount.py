"""A CoOptRunner that preserves the fixed Handle-to-Head boundary.

The original connected-direct-planar parameterizer remains responsible for
Head-to-Head connections, allowing their interfaces to co-deform while staying
connected.  Only Head links attached directly to the fixed Handle declare
``welded_interface=true``; this runner freezes that child's four mount-face
vertices at their baseline positions.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from bilevel.runner import CoOptRunner


_FACE_ID_TO_DOCK = {
    0: (2, 1),
    1: (0, 1),
    2: (1, 1),
    3: (0, -1),
    4: (2, -1),
    5: (1, -1),
}
_FACE_CORNERS = {
    (2, -1): (0, 1, 4, 2),
    (1, -1): (0, 1, 5, 3),
    (0, -1): (0, 2, 6, 3),
    (0, 1): (1, 4, 7, 5),
    (1, 1): (2, 4, 7, 6),
    (2, 1): (3, 5, 7, 6),
}
_VERTEX_PARAMETER_INDICES = {
    0: (),
    1: (0,),
    2: (1, 2),
    3: (3, 4, 5),
    4: (6, 7, 8),
    5: (9, 10, 11),
    6: (12, 13, 14),
    7: (15, 16, 17),
}


def _face_parameter_indices(face_id) -> tuple[int, ...]:
    dock = _FACE_ID_TO_DOCK.get(None if face_id is None else int(face_id))
    if dock is None:
        return tuple(range(18))
    values = set()
    for vertex in _FACE_CORNERS[dock]:
        values.update(_VERTEX_PARAMETER_INDICES[vertex])
    return tuple(sorted(values))


class MountPreservingCoOptRunner(CoOptRunner):
    """Direct-planar runner with exact attachment-face equality constraints."""

    def __init__(self, *args, **kwargs):
        task = args[1] if len(args) >= 2 else kwargs.get("task")
        if task is not None:
            task._handle_head_model_path = kwargs.get("model_path")
        super().__init__(*args, **kwargs)
        self._mount_frozen_by_block: dict[int, tuple[int, ...]] = {}
        self._mount_baseline_by_block: dict[int, np.ndarray] = {}
        self._welded_connection_indices: tuple[int, ...] = ()
        if not self.optimize_design or self.design_bundle is None:
            return
        if getattr(self.design_bundle, "generic_design_protocol", None) != "connected_direct_planar_hexahedron":
            raise ValueError(
                "Mount-preserving shape optimization requires "
                "generic_design_protocol='connected_direct_planar_hexahedron'"
            )
        if self.morphology_runtime is not None:
            records = list(
                self.morphology_runtime.diagnostic_context
                .specification.tool_records
            )
            direct_mount_nodes = {
                int(block.node_id)
                for block in self.morphology_runtime.active_blocks
                if block.direct_handle_mount
            }
            self._welded_connection_indices = tuple(
                index
                for index, record in enumerate(records)
                if (
                    getattr(record, "node_id", None) is not None
                    and int(record.node_id) in direct_mount_nodes
                )
            )
            if not self._welded_connection_indices:
                raise ValueError(
                    "Mount-preserving optimization requires at least one "
                    "direct Handle-mount Head"
                )
            return
        self._configure_mount_constraints()

    def _configure_mount_constraints(self) -> None:
        records = list(self.design_bundle.spec.tool_records)
        frozen: dict[int, set[int]] = {idx: set() for idx in range(len(records))}
        welded: list[int] = []

        for child_idx, record in enumerate(records):
            if str(getattr(record, "attrs", {}).get("welded_interface", "")).lower() != "true":
                continue
            parent = self.design_bundle.spec.records[record.parent] if record.parent is not None else None
            parent_kind = str(getattr(record, "attrs", {}).get("welded_parent_kind", ""))
            # A design_params=0 fixed Handle is intentionally absent from the
            # generic design record table. Its XML transform cannot change,
            # so the only valid strict weld has no deformable-record parent.
            if parent_kind != "handle" or parent is not None:
                raise ValueError(
                    f"Only direct Handle-to-Head links may be welded: {record.link_name!r}"
                )
            if record.planar_child_face is None or record.planar_parent_face is None:
                raise ValueError(f"Welded Head link {record.link_name!r} has incomplete face metadata")
            welded.append(child_idx)
            if int(getattr(record, "mask", 0)) == 47:
                frozen[child_idx].update(_face_parameter_indices(record.planar_child_face))

        if not welded:
            raise ValueError("Mount-preserving optimization requires at least one welded_interface Head link")
        self._welded_connection_indices = tuple(welded)

        for block in self.design_bundle.direct_planar_blocks:
            idx = int(block["tool_index"])
            # Keep the connected-direct-planar face mask exactly as built by
            # the original parameterizer.  In particular, Head-to-Head faces
            # remain connected but are not frozen to their baseline geometry.
            values = tuple(sorted(frozen.get(idx, set())))
            self._mount_frozen_by_block[idx] = values
            start = int(block["start"])
            stop = int(block["stop"])
            self._mount_baseline_by_block[idx] = np.asarray(
                self.design_bundle.init_cage_params[start:stop], dtype=np.float64
            ).copy()

    @staticmethod
    def _selection_jacobian(indices: tuple[int, ...], width: int) -> np.ndarray:
        matrix = np.zeros((len(indices), width), dtype=np.float64)
        for row, index in enumerate(indices):
            matrix[row, index] = 1.0
        return matrix

    def _project_q(self, q: np.ndarray, vector: np.ndarray, face_mask, frozen: tuple[int, ...]) -> np.ndarray:
        from bilevel.parameterization.geometry import constraint_jacobian

        jacobians = [constraint_jacobian(q, face_mask=face_mask)]
        if frozen:
            jacobians.append(self._selection_jacobian(frozen, q.size))
        J = np.vstack([value for value in jacobians if value.shape[0]])
        if not J.shape[0]:
            return np.asarray(vector, dtype=np.float64)
        JJt = J @ J.T
        damping = 1e-10 * float(np.trace(JJt)) / max(float(J.shape[0]), 1.0) + 1e-12
        correction = J.T @ np.linalg.solve(JJt + damping * np.eye(J.shape[0]), J @ vector)
        return np.asarray(vector, dtype=np.float64) - correction

    def _direct_planar_project_vector(self, at_params: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if self.morphology_runtime is not None:
            return super()._direct_planar_project_vector(
                at_params,
                vector,
            )
        projected = np.array(vector, copy=True, dtype=np.float64)
        action_dim = self.ndof_u * self.num_ctrl_steps
        if projected.shape[0] > action_dim:
            projected[action_dim:] = 0.0
        for block in self._direct_planar_abs_blocks():
            start = int(block["abs_start"])
            stop = int(block["abs_stop"])
            if not bool(block.get("deformable", False)) or not bool(block.get("design_enabled", True)):
                projected[start:stop] = 0.0
                continue
            index = int(block["tool_index"])
            projected[start:stop] = self._project_q(
                np.asarray(at_params[start:stop], dtype=np.float64),
                np.asarray(vector[start:stop], dtype=np.float64),
                block.get("face_mask"),
                self._mount_frozen_by_block.get(index, ()),
            )
        return projected

    def _retract_q(self, q: np.ndarray, step: np.ndarray, face_mask, frozen: tuple[int, ...], baseline: np.ndarray):
        from bilevel.parameterization.geometry import (
            constraint_jacobian,
            constraints,
            validate_hexahedron,
        )

        z = np.asarray(q, dtype=np.float64) + np.asarray(step, dtype=np.float64)
        select = self._selection_jacobian(frozen, z.size) if frozen else np.zeros((0, z.size))
        for _ in range(12):
            planar = constraints(z, face_mask=face_mask)
            fixed = z[list(frozen)] - baseline[list(frozen)] if frozen else np.zeros(0)
            residual = np.concatenate([planar, fixed])
            if not residual.size or float(np.linalg.norm(residual, ord=np.inf)) <= 1e-10:
                break
            J = np.vstack([constraint_jacobian(z, face_mask=face_mask), select])
            JJt = J @ J.T
            damping = 1e-10 * float(np.trace(JJt)) / max(float(J.shape[0]), 1.0) + 1e-12
            delta = -J.T @ np.linalg.solve(JJt + damping * np.eye(J.shape[0]), residual)
            old_norm = float(np.linalg.norm(residual, ord=np.inf))
            accepted = False
            beta = 1.0
            for _ in range(12):
                trial = z + beta * delta
                trial_residual = np.concatenate(
                    [
                        constraints(trial, face_mask=face_mask),
                        trial[list(frozen)] - baseline[list(frozen)] if frozen else np.zeros(0),
                    ]
                )
                if np.all(np.isfinite(trial)) and float(np.linalg.norm(trial_residual, ord=np.inf)) < old_norm:
                    z = trial
                    accepted = True
                    break
                beta *= 0.5
            if not accepted:
                return np.asarray(q, dtype=np.float64), False
        if frozen:
            z[list(frozen)] = baseline[list(frozen)]
        try:
            validate_hexahedron(z, face_mask=face_mask)
        except ValueError:
            return np.asarray(q, dtype=np.float64), False
        return z, True

    def _direct_planar_retract_params(self, params: np.ndarray, step: np.ndarray):
        if self.morphology_runtime is not None:
            trial, ok = super()._direct_planar_retract_params(
                params,
                step,
            )
            if not ok:
                return params, False
            report = self.mount_integrity_report(trial)
            return (trial, True) if report["ok"] else (params, False)
        from bilevel.parameterization.geometry import max_vertex_displacement

        trial = np.asarray(params, dtype=np.float64) + np.asarray(step, dtype=np.float64)
        if not np.all(np.isfinite(trial)):
            return params, False
        action_dim = self.ndof_u * self.num_ctrl_steps
        bounds = self._bounds()
        for index in range(min(action_dim, len(bounds))):
            low, high = bounds[index]
            if low is not None:
                trial[index] = max(trial[index], float(low))
            if high is not None:
                trial[index] = min(trial[index], float(high))
        if self.ndof_cage:
            trial[action_dim:] = params[action_dim:]

        max_total_displacement = float(
            getattr(self.args, "mount_preserving_max_shape_displacement", 0.125) or 0.125
        )
        for block in self._direct_planar_abs_blocks():
            start = int(block["abs_start"])
            stop = int(block["abs_stop"])
            if not bool(block.get("deformable", False)) or not bool(block.get("design_enabled", True)):
                trial[start:stop] = params[start:stop]
                continue
            index = int(block["tool_index"])
            q_new, ok = self._retract_q(
                np.asarray(params[start:stop], dtype=np.float64),
                np.asarray(step[start:stop], dtype=np.float64),
                block.get("face_mask"),
                self._mount_frozen_by_block.get(index, ()),
                self._mount_baseline_by_block[index],
            )
            if not ok:
                return params, False
            if max_vertex_displacement(self._mount_baseline_by_block[index], q_new) > max_total_displacement:
                return params, False
            trial[start:stop] = q_new
        if not self._direct_planar_collision_ok(params, trial):
            return params, False
        report = self.mount_integrity_report(trial)
        return (trial, True) if report["ok"] else (params, False)

    def mount_integrity_report(self, params: np.ndarray) -> dict:
        if not self.optimize_design or self.design_bundle is None:
            return {
                "ok": True,
                "max_translation_error": 0.0,
                "max_rotation_error": 0.0,
                "max_face_error": 0.0,
                "max_shape_displacement": 0.0,
                "connections": [],
            }
        if self.morphology_runtime is not None:
            return self._unified_mount_integrity_report(params)
        from bilevel.parameterization.geometry import (
            max_vertex_displacement,
            vertices_from_q,
        )

        action_dim = self.ndof_u * self.num_ctrl_steps
        cage = np.asarray(params[action_dim : action_dim + self.ndof_cage], dtype=np.float64)
        baseline_cage = np.asarray(self.design_bundle.init_cage_params, dtype=np.float64)
        design = np.asarray(self.design_bundle.design_np.parameterize(cage, generate_mesh=False), dtype=np.float64)
        baseline_design = np.asarray(
            self.design_bundle.design_np.parameterize(baseline_cage, generate_mesh=False), dtype=np.float64
        )
        connections = []
        max_translation = max_rotation = max_face = max_shape = 0.0
        blocks = {int(block["tool_index"]): block for block in self.design_bundle.direct_planar_blocks}
        welded_indices = set(self._welded_connection_indices)
        for index, record in enumerate(self.design_bundle.spec.tool_records):
            block = blocks[index]
            start = int(block["start"])
            stop = int(block["stop"])
            q = cage[start:stop]
            q0 = baseline_cage[start:stop]
            shape_delta = max_vertex_displacement(q0, q)
            max_shape = max(max_shape, float(shape_delta))
            frozen = self._mount_frozen_by_block.get(index, ())
            face_error = 0.0
            if index in welded_indices:
                dock = _FACE_ID_TO_DOCK[int(record.planar_child_face)]
                corners = _FACE_CORNERS[dock]
                vertices = vertices_from_q(q)
                baseline_vertices = vertices_from_q(q0)
                cage_obj = self.design_bundle.design_np.tool_cages[index]
                face_error = float(
                    getattr(cage_obj, "ref_length", 1.0)
                    * np.max(
                        np.linalg.norm(
                            (vertices[:, corners] - baseline_vertices[:, corners]).T,
                            axis=1,
                        )
                    )
                )
            max_face = max(max_face, face_error)
            translation = rotation = 0.0
            if index in welded_indices and record.parent is not None and record.p1_slice is not None:
                before = baseline_design[record.p1_slice]
                after = design[record.p1_slice]
                rotation = float(np.linalg.norm(after[:9] - before[:9]))
                translation = float(np.linalg.norm(after[9:12] - before[9:12]))
                max_translation = max(max_translation, translation)
                max_rotation = max(max_rotation, rotation)
            if index in welded_indices:
                connections.append(
                    {
                        "link_name": record.link_name,
                        "socket_id": record.attrs.get("welded_socket_id"),
                        "parent_kind": record.attrs.get("welded_parent_kind"),
                        "parent_id": record.attrs.get("welded_parent_id"),
                        "parent_face": int(record.planar_parent_face),
                        "child_face": int(record.planar_child_face),
                        "translation_error": translation,
                        "rotation_error": rotation,
                        "face_vertex_error": face_error,
                        "face_error": face_error,
                    }
                )
        translation_tol = float(getattr(self.args, "mount_translation_tolerance", 1e-7))
        rotation_tol = float(getattr(self.args, "mount_rotation_tolerance", 1e-7))
        face_tol = float(getattr(self.args, "mount_face_tolerance", 1e-7))
        return {
            "ok": bool(
                max_translation <= translation_tol
                and max_rotation <= rotation_tol
                and max_face <= face_tol
                and max_shape <= float(getattr(self.args, "mount_preserving_max_shape_displacement", 0.125)) + 1e-10
            ),
            "max_translation_error": max_translation,
            "max_rotation_error": max_rotation,
            "max_face_error": max_face,
            "max_shape_displacement": max_shape,
            "connections": connections,
        }

    def _unified_mount_integrity_report(
        self,
        params: np.ndarray,
    ) -> dict:
        from bilevel.parameterization.geometry import (
            max_vertex_displacement,
            vertices_from_q,
        )

        _, morphology = self.unpack_params(params)
        baseline = self.initial_morphology_parameters
        design = np.asarray(
            self.parameterize_morphology_numpy(
                morphology,
                generate_mesh=False,
            ),
            dtype=np.float64,
        )
        baseline_design = np.asarray(
            self.parameterize_morphology_numpy(
                baseline,
                generate_mesh=False,
            ),
            dtype=np.float64,
        )
        specification = (
            self.morphology_runtime.diagnostic_context.specification
        )
        records_by_node = {
            int(record.node_id): record
            for record in specification.tool_records
            if getattr(record, "node_id", None) is not None
        }
        connections = []
        max_translation = max_rotation = max_face = max_shape = 0.0
        for block in self.morphology_runtime.active_blocks:
            start = int(block.start)
            stop = int(block.stop)
            current_q = morphology[start:stop]
            baseline_q = baseline[start:stop]
            shape_delta = max_vertex_displacement(
                baseline_q,
                current_q,
            )
            max_shape = max(max_shape, float(shape_delta))
            if not block.direct_handle_mount:
                continue
            record = records_by_node[int(block.node_id)]
            dock = _FACE_ID_TO_DOCK[int(block.child_face)]
            corners = _FACE_CORNERS[dock]
            vertices = vertices_from_q(current_q)
            baseline_vertices = vertices_from_q(baseline_q)
            face_error = float(
                block.reference_length
                * np.max(
                    np.linalg.norm(
                        (
                            vertices[:, corners]
                            - baseline_vertices[:, corners]
                        ).T,
                        axis=1,
                    )
                )
            )
            max_face = max(max_face, face_error)
            translation = rotation = 0.0
            if record.parent is not None and record.p1_slice is not None:
                before = baseline_design[record.p1_slice]
                after = design[record.p1_slice]
                rotation = float(
                    np.linalg.norm(after[:9] - before[:9])
                )
                translation = float(
                    np.linalg.norm(after[9:12] - before[9:12])
                )
                max_translation = max(max_translation, translation)
                max_rotation = max(max_rotation, rotation)
            connections.append(
                {
                    "link_name": record.link_name,
                    "socket_id": record.attrs.get(
                        "welded_socket_id"
                    ),
                    "parent_kind": record.attrs.get(
                        "welded_parent_kind"
                    ),
                    "parent_id": record.attrs.get(
                        "welded_parent_id"
                    ),
                    "parent_face": int(block.parent_face),
                    "child_face": int(block.child_face),
                    "translation_error": translation,
                    "rotation_error": rotation,
                    "face_vertex_error": face_error,
                    "face_error": face_error,
                }
            )
        translation_tol = float(
            getattr(self.args, "mount_translation_tolerance", 1e-7)
        )
        rotation_tol = float(
            getattr(self.args, "mount_rotation_tolerance", 1e-7)
        )
        face_tol = float(
            getattr(self.args, "mount_face_tolerance", 1e-7)
        )
        max_shape_allowed = float(
            getattr(
                self.args,
                "mount_preserving_max_shape_displacement",
                0.125,
            )
        )
        return {
            "ok": bool(
                max_translation <= translation_tol
                and max_rotation <= rotation_tol
                and max_face <= face_tol
                and max_shape <= max_shape_allowed + 1e-10
            ),
            "max_translation_error": max_translation,
            "max_rotation_error": max_rotation,
            "max_face_error": max_face,
            "max_shape_displacement": max_shape,
            "connections": connections,
        }

    def save(self, save_dir: str, params: np.ndarray):
        """Preserve the legacy artifacts and add an explicit weld audit."""

        super().save(save_dir, params)
        report = self.mount_integrity_report(params)
        if not report.get("ok", False):
            raise ValueError(f"Refusing to save a result with a broken welded interface: {report}")
        with open(os.path.join(save_dir, "welded_mount_report.json"), "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
