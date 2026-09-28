"""Active Head coordinates over connected direct-planar hexahedron geometry.

Frozen blocks are excluded from optimization; the full geometric vector remains
an internal assembly format and an explicit saved-artifact migration format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from bilevel.parameterization.collision import check_design_params_collision
from bilevel.parameterization.design import (
    DirectPlanarHexDesignNP,
    DirectPlanarHexDesignTorch,
    build_connected_head_topology,
)
from bilevel.parameterization.topology import (
    HeadTopology,
)

from .base import (
    UNIFIED_PARAMETERIZATION_ID,
    MorphologyBlock,
    MorphologyCollisionDecision,
    MorphologyCollisionPolicy,
    MorphologyLayout,
    MorphologyParameterization,
    MorphologyRetractionResult,
    redmax_slices_from_record,
)
from .constraints import (
    mount_face_parameter_indices,
    project_block_tangent,
    retract_block,
    validate_block_geometry,
)


_HEAD_BLOCK_PARAMETERIZATION = "connected_hexahedron"
@dataclass(frozen=True)
class _LegacyBlockBinding:
    source_start: int
    source_stop: int
    target_start: int | None
    target_stop: int | None

    @property
    def active(self) -> bool:
        return self.target_start is not None


def _compatibility_asset_id(record: Any) -> str:
    explicit = str(getattr(record, "attrs", {}).get("asset_id", "")).strip()
    if explicit:
        return explicit
    mesh_path = getattr(record, "mesh_path", None)
    if mesh_path is None:
        raise ValueError(
            f"Head record {getattr(record, 'link_name', '')!r} has no asset_id "
            "or mesh compatibility identifier"
        )
    return f"legacy_asset/{Path(mesh_path).stem}"


def _parent_node_id(spec: Any, record: Any) -> int | None:
    parent_index = getattr(record, "parent", None)
    if parent_index is None:
        return None
    parent = spec.records[int(parent_index)]
    node_id = getattr(parent, "node_id", None)
    return None if node_id is None else int(node_id)


def _is_direct_handle_mount(spec: Any, record: Any) -> bool:
    attrs = getattr(record, "attrs", {})
    welded = str(attrs.get("welded_interface", "")).lower() == "true"
    if not welded:
        return False
    parent = (
        spec.records[record.parent]
        if record.parent is not None
        else None
    )
    if str(attrs.get("welded_parent_kind", "")) != "handle" or parent is not None:
        raise ValueError(
            "only a direct fixed Handle-to-Head link may declare "
            f"welded_interface: {record.link_name!r}"
        )
    if record.planar_child_face is None or record.planar_parent_face is None:
        raise ValueError(
            f"welded Head {record.link_name!r} has incomplete face metadata"
        )
    return True


class UnifiedConnectedHeadMorphology(MorphologyParameterization):
    """Active-Head view over the current connected-direct DesignBundle."""


    def __init__(self, legacy_bundle: Any):
        protocol = str(
            getattr(legacy_bundle, "generic_design_protocol", "")
        )
        if protocol != "connected_direct_planar_hexahedron":
            raise ValueError(
                "connected Head parameterization requires connected_direct_planar_hexahedron, "
                f"got {protocol!r}"
            )
        initial_legacy = np.asarray(
            legacy_bundle.init_cage_params,
            dtype=np.float64,
        ).reshape(-1)
        records = list(legacy_bundle.spec.tool_records)
        source_blocks = list(legacy_bundle.direct_planar_blocks)
        if len(records) != len(source_blocks):
            raise ValueError(
                "legacy tool record/block count mismatch: "
                f"{len(records)} != {len(source_blocks)}"
            )
        legacy_design_np = getattr(legacy_bundle, "design_np", None)
        legacy_tool_cages = list(
            getattr(legacy_design_np, "tool_cages", ()) or ()
        )
        head_topology = (
            build_connected_head_topology(
                legacy_bundle.spec,
                tool_cages=legacy_tool_cages,
            )
            if len(legacy_tool_cages) == len(records)
            else None
        )

        active_blocks = []
        bindings = []
        active_initial = []
        constraint_reference_lengths = {}
        target_cursor = 0
        source_cursor = 0
        for record_index, (record, source) in enumerate(
            zip(records, source_blocks)
        ):
            source_start = int(source["start"])
            source_stop = int(source["stop"])
            if source_start != source_cursor:
                raise ValueError(
                    "Head morphology blocks must be contiguous: "
                    f"expected {source_cursor}, got {source_start}"
                )
            source_dim = source_stop - source_start
            if source_dim <= 0:
                raise ValueError("legacy morphology block must be non-empty")
            if bool(source.get("deformable", False)):
                direct_handle_mount = _is_direct_handle_mount(
                    legacy_bundle.spec,
                    record,
                )
                topology_block = (
                    None
                    if head_topology is None
                    else head_topology.block(record_index)
                )
                if (
                    topology_block is not None
                    and topology_block.direct_handle_mount
                    != direct_handle_mount
                ):
                    raise ValueError(
                        "Head topology Handle-mount metadata disagrees "
                        f"for {record.link_name!r}"
                    )
                source_face_mask = tuple(
                    bool(value)
                    for value in source.get("face_mask", ())
                )
                connected_face_mask = (
                    source_face_mask
                    if topology_block is None
                    else topology_block.connected_face_mask
                )
                if (
                    topology_block is not None
                    and source_face_mask != connected_face_mask
                ):
                    raise ValueError(
                        "Head topology face mask disagrees with legacy "
                        f"oracle for {record.link_name!r}"
                    )
                frozen_parameter_indices = (
                    mount_face_parameter_indices(
                        record.planar_child_face
                    )
                    if direct_handle_mount and int(record.mask) == 47
                    else ()
                )
                node_id = getattr(record, "node_id", None)
                if node_id is None:
                    raise ValueError(
                        f"active Head {record.link_name!r} has no stable node_id"
                    )
                tool_cages = getattr(
                    legacy_design_np,
                    "tool_cages",
                    (),
                )
                reference_length = (
                    float(
                        getattr(
                            tool_cages[record_index],
                            "ref_length",
                            1.0,
                        )
                    )
                    if record_index < len(tool_cages)
                    else 1.0
                )
                target_start = target_cursor
                target_stop = target_start + source_dim
                active_blocks.append(
                    MorphologyBlock(
                        node_id=int(node_id),
                        asset_id=_compatibility_asset_id(record),
                        parameterization=_HEAD_BLOCK_PARAMETERIZATION,
                        start=target_start,
                        stop=target_stop,
                        redmax_design_mask=int(record.mask),
                        redmax_slices=redmax_slices_from_record(record),
                        link_name=str(record.link_name),
                        body_name=str(record.body_name),
                        parent_node_id=_parent_node_id(
                            legacy_bundle.spec,
                            record,
                        ),
                        parent_face=getattr(
                            record,
                            "planar_parent_face",
                            None,
                        ),
                        child_face=getattr(
                            record,
                            "planar_child_face",
                            None,
                        ),
                        endeffector_face=getattr(
                            record,
                            "planar_endeffector_face",
                            None,
                        ),
                        connected_face_mask=connected_face_mask,
                        direct_handle_mount=direct_handle_mount,
                        frozen_parameter_indices=frozen_parameter_indices,
                        reference_length=reference_length,
                    )
                )
                constraint_reference_lengths[int(node_id)] = reference_length
                active_initial.append(
                    initial_legacy[source_start:source_stop]
                )
                bindings.append(
                    _LegacyBlockBinding(
                        source_start=source_start,
                        source_stop=source_stop,
                        target_start=target_start,
                        target_stop=target_stop,
                    )
                )
                target_cursor = target_stop
            else:
                bindings.append(
                    _LegacyBlockBinding(
                        source_start=source_start,
                        source_stop=source_stop,
                        target_start=None,
                        target_stop=None,
                    )
                )
            source_cursor = source_stop
        if source_cursor != initial_legacy.shape[0]:
            raise ValueError(
                f"legacy Head blocks stop at {source_cursor}, but morphology "
                f"dimension is {initial_legacy.shape[0]}"
            )

        redmax_dim = int(getattr(legacy_bundle.spec, "ndof_p"))
        self._legacy_bundle = legacy_bundle
        self._legacy_initial = initial_legacy.copy()
        self._bindings = tuple(bindings)
        self._head_topology = head_topology
        self._constraint_reference_lengths = (
            constraint_reference_lengths
        )
        self._forward_np: DirectPlanarHexDesignNP | None = None
        self._forward_torch: DirectPlanarHexDesignTorch | None = None
        self._force_connectivity = bool(
            getattr(legacy_bundle.design_np, "force_connectivity", False)
        ) if hasattr(legacy_bundle, "design_np") else False
        self._optimize_finger_design = bool(
            getattr(legacy_bundle.design_np, "optimize_finger_design", False)
        ) if hasattr(legacy_bundle, "design_np") else False
        if self._optimize_finger_design:
            raise ValueError(
                "unified Head morphology cannot include finger optimization"
            )
        self._layout = MorphologyLayout(
            parameterization_id=UNIFIED_PARAMETERIZATION_ID,
            blocks=tuple(active_blocks),
            morphology_dim=target_cursor,
            redmax_parameter_dim=redmax_dim,
        )
        self._initial = (
            np.concatenate(active_initial).astype(np.float64, copy=False)
            if active_initial
            else np.zeros(0, dtype=np.float64)
        )

    @property
    def layout(self) -> MorphologyLayout:
        return self._layout

    @property
    def initial_parameters(self) -> np.ndarray:
        return self._initial.copy()

    def bounds(
        self,
        *,
        head_parameter_margin: float = 2.0,
    ) -> tuple[tuple[float, float], ...]:
        margin = float(head_parameter_margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError(
                "head_parameter_margin must be finite and nonnegative"
            )
        initial = self.initial_parameters
        return tuple(
            (float(value - margin), float(value + margin))
            for value in initial
        )

    @property
    def legacy_parameter_dim(self) -> int:
        return int(self._legacy_initial.shape[0])

    @property
    def independent_forward_initialized(self) -> bool:
        return self._forward_np is not None and self._forward_torch is not None

    @property
    def head_topology(self) -> HeadTopology | None:
        return self._head_topology

    @property
    def specification(self) -> Any:
        return self._legacy_bundle.spec

    def _ensure_forward(self) -> None:
        if self.independent_forward_initialized:
            return
        spec = self._legacy_bundle.spec
        self._forward_np = DirectPlanarHexDesignNP(
            spec,
            optimize_finger_design=False,
            force_connectivity=self._force_connectivity,
            head_topology=self._head_topology,
        )
        self._forward_torch = DirectPlanarHexDesignTorch(
            spec,
            optimize_finger_design=False,
            force_connectivity=self._force_connectivity,
            head_topology=self._head_topology,
        )

    def expand_legacy_numpy(self, morphology: np.ndarray) -> np.ndarray:
        values = self.validate_parameters(morphology)
        legacy = self._legacy_initial.copy()
        for binding in self._bindings:
            if not binding.active:
                continue
            legacy[binding.source_start : binding.source_stop] = values[
                int(binding.target_start) : int(binding.target_stop)
            ]
        return legacy

    def expand_legacy_torch(
        self,
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        values = morphology.reshape(-1).to(dtype=torch.double)
        if values.numel() != self.layout.morphology_dim:
            raise ValueError(
                f"morphology dim {values.numel()} != "
                f"{self.layout.morphology_dim}"
            )
        parts = []
        source_cursor = 0
        for binding in self._bindings:
            if binding.source_start > source_cursor:
                parts.append(
                    torch.as_tensor(
                        self._legacy_initial[
                            source_cursor : binding.source_start
                        ],
                        dtype=torch.double,
                        device=values.device,
                    )
                )
            if binding.active:
                parts.append(
                    values[
                        int(binding.target_start) : int(binding.target_stop)
                    ]
                )
            else:
                parts.append(
                    torch.as_tensor(
                        self._legacy_initial[
                            binding.source_start : binding.source_stop
                        ],
                        dtype=torch.double,
                        device=values.device,
                    )
                )
            source_cursor = binding.source_stop
        if source_cursor < self.legacy_parameter_dim:
            parts.append(
                torch.as_tensor(
                    self._legacy_initial[source_cursor:],
                    dtype=torch.double,
                    device=values.device,
                )
            )
        return torch.cat(parts)

    def expand_head_numpy(self, morphology: np.ndarray) -> np.ndarray:
        values = self.validate_parameters(morphology)
        head = self._legacy_initial.copy()
        for binding in self._bindings:
            if not binding.active:
                continue
            head[
                binding.source_start : binding.source_stop
            ] = values[
                int(binding.target_start) : int(binding.target_stop)
            ]
        return head

    def expand_head_torch(self, morphology: torch.Tensor) -> torch.Tensor:
        values = morphology.reshape(-1).to(dtype=torch.double)
        if values.numel() != self.layout.morphology_dim:
            raise ValueError(
                f"morphology dim {values.numel()} != "
                f"{self.layout.morphology_dim}"
            )
        parts = []
        source_cursor = 0
        for binding in self._bindings:
            if binding.source_start > source_cursor:
                parts.append(
                    torch.as_tensor(
                        self._legacy_initial[
                            source_cursor : binding.source_start
                        ],
                        dtype=torch.double,
                        device=values.device,
                    )
                )
            if binding.active:
                parts.append(
                    values[
                        int(binding.target_start) : int(binding.target_stop)
                    ]
                )
            else:
                parts.append(
                    torch.as_tensor(
                        self._legacy_initial[
                            binding.source_start : binding.source_stop
                        ],
                        dtype=torch.double,
                        device=values.device,
                    )
                )
            source_cursor = binding.source_stop
        if source_cursor < self.legacy_parameter_dim:
            parts.append(
                torch.as_tensor(
                    self._legacy_initial[source_cursor:],
                    dtype=torch.double,
                    device=values.device,
                )
            )
        if not parts:
            return torch.empty(0, dtype=torch.double, device=values.device)
        return torch.cat(parts)

    def compress_legacy_numpy(self, legacy: np.ndarray) -> np.ndarray:
        values = np.asarray(legacy, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(
                f"legacy morphology must be one-dimensional, got "
                f"{values.shape}"
            )
        if values.shape != self._legacy_initial.shape:
            raise ValueError(
                f"legacy morphology shape {values.shape} != "
                f"{self._legacy_initial.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("legacy morphology contains non-finite values")
        active = []
        for binding in self._bindings:
            if binding.active:
                active.append(
                    values[binding.source_start : binding.source_stop]
                )
        return (
            np.concatenate(active).astype(np.float64, copy=False)
            if active
            else np.zeros(0, dtype=np.float64)
        )

    def migrate_legacy_numpy(self, legacy: np.ndarray) -> np.ndarray:
        """Remove only coordinates proven inactive by the current layout."""

        values = np.asarray(legacy, dtype=np.float64)
        active = self.compress_legacy_numpy(values)
        reconstructed = self.expand_legacy_numpy(active)
        if not np.array_equal(reconstructed, values):
            changed = np.flatnonzero(reconstructed != values)
            preview = ", ".join(str(int(index)) for index in changed[:8])
            raise ValueError(
                "legacy morphology changes fixed or inactive coordinates "
                f"at indices [{preview}]; refusing lossy migration"
            )
        return active

    def parameterize_numpy(
        self,
        morphology: np.ndarray,
        *,
        generate_mesh: bool = False,
    ):
        self._ensure_forward()
        return self._forward_np.parameterize_heads(
            self.expand_head_numpy(morphology),
            generate_mesh=generate_mesh,
        )

    def parameterize_torch(self, morphology: torch.Tensor) -> torch.Tensor:
        self._ensure_forward()
        return self._forward_torch.parameterize_heads(
            self.expand_head_torch(morphology)
        )

    def release_torch_graph_refs(self) -> None:
        forward = self._forward_torch
        if forward is None:
            return
        for cage in getattr(forward, "tool_cages", ()) or ():
            for attr in ("params", "vertices"):
                value = getattr(cage, attr, None)
                if isinstance(value, torch.Tensor):
                    setattr(cage, attr, value.detach())

    def connection_diagnostics(
        self,
        morphology: np.ndarray,
    ):
        values = self.validate_parameters(morphology)
        self._ensure_forward()
        diagnostic_fn = getattr(
            self._forward_np,
            "connection_diagnostics",
            None,
        )
        if diagnostic_fn is None:
            return None
        return diagnostic_fn(self.expand_head_numpy(values))

    def project_tangent(
        self,
        morphology: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        values = self.validate_parameters(morphology)
        direction = np.asarray(vector, dtype=np.float64).reshape(-1)
        if direction.shape != values.shape:
            raise ValueError(
                f"morphology vector shape {direction.shape} != "
                f"{values.shape}"
            )
        if not np.all(np.isfinite(direction)):
            raise ValueError("morphology vector contains non-finite values")
        projected = np.empty_like(direction)
        for block in self.layout.blocks:
            start = int(block.start)
            stop = int(block.stop)
            projected[start:stop] = project_block_tangent(
                values[start:stop],
                direction[start:stop],
                face_mask=block.connected_face_mask,
                frozen_indices=block.frozen_parameter_indices,
            )
        return projected

    def validate_geometry(
        self,
        morphology: np.ndarray,
        *,
        max_shape_displacement: float = 0.125,
        mount_face_tolerance: float = 1e-7,
    ) -> None:
        values = self.validate_parameters(morphology)
        if (
            not np.isfinite(max_shape_displacement)
            or max_shape_displacement < 0.0
        ):
            raise ValueError(
                "max_shape_displacement must be finite and nonnegative"
            )
        if (
            not np.isfinite(mount_face_tolerance)
            or mount_face_tolerance < 0.0
        ):
            raise ValueError(
                "mount_face_tolerance must be finite and nonnegative"
            )
        for block in self.layout.blocks:
            start = int(block.start)
            stop = int(block.stop)
            validate_block_geometry(
                values[start:stop],
                baseline=self._initial[start:stop],
                face_mask=block.connected_face_mask,
                frozen_indices=block.frozen_parameter_indices,
                mount_face_id=(
                    block.child_face
                    if block.direct_handle_mount
                    else None
                ),
                reference_length=self._constraint_reference_lengths[
                    block.node_id
                ],
                mount_face_tolerance=mount_face_tolerance,
                max_shape_displacement=max_shape_displacement,
            )

    def retract(
        self,
        morphology: np.ndarray,
        step: np.ndarray,
        *,
        max_shape_displacement: float = 0.125,
        mount_face_tolerance: float = 1e-7,
    ) -> MorphologyRetractionResult:
        values = self.validate_parameters(morphology)
        delta = np.asarray(step, dtype=np.float64).reshape(-1)
        if delta.shape != values.shape:
            raise ValueError(
                f"morphology step shape {delta.shape} != {values.shape}"
            )
        if not np.all(np.isfinite(delta)):
            return MorphologyRetractionResult(
                morphology=values.copy(),
                ok=False,
                reason="non-finite morphology step",
            )
        trial = values.copy()
        for block in self.layout.blocks:
            start = int(block.start)
            stop = int(block.stop)
            result = retract_block(
                values[start:stop],
                delta[start:stop],
                face_mask=block.connected_face_mask,
                frozen_indices=block.frozen_parameter_indices,
                baseline=self._initial[start:stop],
            )
            if not result.ok:
                return MorphologyRetractionResult(
                    morphology=values.copy(),
                    ok=False,
                    reason=f"Head node {block.node_id} retraction failed",
                )
            trial[start:stop] = result.q
        try:
            self.validate_geometry(
                trial,
                max_shape_displacement=max_shape_displacement,
                mount_face_tolerance=mount_face_tolerance,
            )
        except ValueError as exc:
            return MorphologyRetractionResult(
                morphology=values.copy(),
                ok=False,
                reason=str(exc),
            )
        return MorphologyRetractionResult(
            morphology=trial,
            ok=True,
        )

    def collision_report(
        self,
        morphology: np.ndarray,
        *,
        policy: MorphologyCollisionPolicy,
        xml_path: str | Path | None = None,
    ):
        values = self.validate_parameters(morphology)
        if not isinstance(policy, MorphologyCollisionPolicy):
            raise TypeError(
                "policy must be a MorphologyCollisionPolicy"
            )
        if not policy.enabled:
            return None
        self._ensure_forward()
        design_params = self.parameterize_numpy(
            values,
            generate_mesh=False,
        )
        collision_bundle = SimpleNamespace(
            spec=self._legacy_bundle.spec,
            design_np=self._forward_np,
            model_path=(
                xml_path
                or getattr(self._legacy_bundle, "model_path", None)
                or getattr(self._legacy_bundle.spec, "xml_path", None)
            ),
        )
        return check_design_params_collision(
            design_params,
            collision_bundle,
            xml_path=xml_path,
            margin=policy.margin,
            max_report=policy.max_report,
            check_ground=policy.check_ground,
        )

    def validate_collision_transition(
        self,
        previous: np.ndarray,
        candidate: np.ndarray,
        *,
        policy: MorphologyCollisionPolicy,
        xml_path: str | Path | None = None,
    ) -> MorphologyCollisionDecision:
        previous_values = self.validate_parameters(previous)
        candidate_values = self.validate_parameters(candidate)
        if not isinstance(policy, MorphologyCollisionPolicy):
            raise TypeError(
                "policy must be a MorphologyCollisionPolicy"
            )
        if not policy.enabled:
            return MorphologyCollisionDecision(
                accepted=True,
                report=None,
                skipped_unchanged=False,
            )
        if np.allclose(
            previous_values,
            candidate_values,
            rtol=0.0,
            atol=1e-14,
        ):
            return MorphologyCollisionDecision(
                accepted=True,
                report=None,
                skipped_unchanged=True,
            )
        report = self.collision_report(
            candidate_values,
            policy=policy,
            xml_path=xml_path,
        )
        return MorphologyCollisionDecision(
            accepted=bool(report is None or report.ok),
            report=report,
            skipped_unchanged=False,
        )

    def apply(
        self,
        simulation: Any,
        morphology: np.ndarray,
        *,
        generate_mesh: bool = False,
    ):
        result = self.parameterize_numpy(
            morphology,
            generate_mesh=generate_mesh,
        )
        if generate_mesh:
            design_params, meshes = result
        else:
            design_params = result
            meshes = None
        if len(design_params) != int(simulation.ndof_p):
            raise ValueError(
                f"RedMax parameter dim {len(design_params)} != "
                f"simulation.ndof_p {simulation.ndof_p}"
            )
        simulation.set_design_params(design_params)
        if generate_mesh:
            vertices, faces = self._legacy_bundle._complete_render_mesh(meshes)
            simulation.set_rendering_mesh(vertices, faces)
        return design_params, meshes
