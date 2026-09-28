from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _record_label(rec) -> str:
    return " ".join(
        str(getattr(rec, attr, "") or "").lower()
        for attr in ("link_name", "joint_name", "body_name")
    )


def _record_index(spec, rec) -> int | None:
    for idx, candidate in enumerate(getattr(spec, "records", []) or []):
        if candidate is rec:
            return idx
    return None


def _tool_children(spec) -> dict[int, list[int]]:
    tool_ids = {id(rec) for rec in getattr(spec, "tool_records", []) or []}
    children: dict[int, list[int]] = {}
    for idx, rec in enumerate(getattr(spec, "records", []) or []):
        parent = getattr(rec, "parent", None)
        if parent is None:
            continue
        parent_rec = spec.records[parent]
        if id(rec) in tool_ids and id(parent_rec) in tool_ids:
            children.setdefault(parent, []).append(idx)
    return children


def _terminal_tool_descendants(spec, root_rec) -> list:
    root_idx = _record_index(spec, root_rec)
    if root_idx is None:
        return [root_rec]
    children = _tool_children(spec)
    out = []
    stack = [root_idx]
    while stack:
        idx = stack.pop()
        child_indices = list(children.get(idx, []))
        if not child_indices:
            rec = spec.records[idx]
            if getattr(rec, "domain", None) == "tool":
                out.append(rec)
            continue
        stack.extend(child_indices)
    return out or [root_rec]


def _match_record(records: Iterable, *, joint_name: str | None, name_hint: str | None):
    joint_name_l = (joint_name or "").lower()
    name_hint_l = (name_hint or "").lower()
    for rec in records:
        if joint_name_l and str(getattr(rec, "joint_name", "")).lower() == joint_name_l:
            return rec
    if name_hint_l:
        for rec in records:
            if name_hint_l in _record_label(rec):
                return rec
    return None


def _softmin_squared(contacts: np.ndarray, target: np.ndarray, *, tau: float, axes: tuple[int, ...]):
    if contacts.shape[0] == 0:
        return 0.0, np.zeros((0,), dtype=np.float64), np.zeros((0, len(axes)), dtype=np.float64)
    tau = max(float(tau), 1e-9)
    target_axes = np.asarray(target, dtype=np.float64)[list(axes)]
    diff = contacts[:, list(axes)] - target_axes.reshape(1, len(axes))
    sq = np.sum(diff * diff, axis=1)
    logits = -sq / tau
    max_logit = float(np.max(logits))
    shifted = logits - max_logit
    weights = np.exp(shifted)
    weights /= max(float(np.sum(weights)), 1e-12)
    logsumexp = float(np.log(max(float(np.sum(np.exp(shifted))), 1e-12)) + max_logit)
    value = float(-tau * (logsumexp - np.log(float(contacts.shape[0]))))
    grad = 2.0 * weights.reshape(-1, 1) * diff
    return value, weights, grad


@dataclass
class FunctionGroupContactSet:
    name: str
    spec: object
    anchor_record: object | None
    leaf_records: list
    contact_records: list
    contact_slices: list[slice]
    baseline_contacts: np.ndarray

    @property
    def contact_count(self) -> int:
        return int(sum((sl.stop - sl.start) // 3 for sl in self.contact_slices))

    @property
    def baseline_center(self) -> np.ndarray:
        if self.baseline_contacts.shape[0] == 0:
            return np.zeros(3, dtype=np.float64)
        return np.mean(self.baseline_contacts, axis=0)

    def local_contacts(self, design_params: np.ndarray) -> np.ndarray:
        chunks = []
        for sl in self.contact_slices:
            if sl is None or sl.stop <= sl.start:
                continue
            chunk = np.asarray(design_params[sl], dtype=np.float64).reshape(-1, 3)
            if chunk.shape[0]:
                chunks.append(chunk)
        if not chunks:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _transform_np(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(12)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = values[:9].reshape(3, 3)
        transform[:3, 3] = values[9:12]
        return transform

    def _record_transform_np(self, rec, design_params: np.ndarray, cache: dict[int, np.ndarray]) -> np.ndarray:
        key = id(rec)
        if key in cache:
            return cache[key]
        parent = np.eye(4, dtype=np.float64)
        if rec.parent is not None:
            parent = self._record_transform_np(self.spec.records[rec.parent], design_params, cache)
        joint = rec.joint_E if rec.p1_slice is None else self._transform_np(design_params[rec.p1_slice])
        body = rec.body_E if rec.p2_slice is None else self._transform_np(design_params[rec.p2_slice])
        transform = parent @ joint @ body
        cache[key] = transform
        return transform

    def root_frame_contact_offsets(self, design_params: np.ndarray) -> np.ndarray:
        design_params = np.asarray(design_params, dtype=np.float64)
        if self.anchor_record is None:
            return self.local_contacts(design_params) - self.baseline_center.reshape(1, 3)

        cache: dict[int, np.ndarray] = {}
        anchor = self.anchor_record
        parent = np.eye(4, dtype=np.float64)
        if anchor.parent is not None:
            parent = self._record_transform_np(self.spec.records[anchor.parent], design_params, cache)
        anchor_joint = anchor.joint_E if anchor.p1_slice is None else self._transform_np(design_params[anchor.p1_slice])
        anchor_pos = (parent @ anchor_joint)[:3, 3]

        chunks = []
        for rec, sl in zip(self.contact_records, self.contact_slices):
            local = design_params[sl].reshape(-1, 3)
            if local.shape[0] == 0:
                continue
            transform = self._record_transform_np(rec, design_params, cache)
            root_points = local @ transform[:3, :3].T + transform[:3, 3]
            chunks.append(root_points - anchor_pos.reshape(1, 3))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float64)

    def root_frame_contact_offsets_torch(self, design_params):
        import torch

        def transform(values):
            values = values.reshape(12)
            row0 = torch.cat([values[0:3], values[9:10]])
            row1 = torch.cat([values[3:6], values[10:11]])
            row2 = torch.cat([values[6:9], values[11:12]])
            row3 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=values.dtype, device=values.device)
            return torch.stack([row0, row1, row2, row3])

        def constant_transform(value):
            return torch.tensor(value, dtype=design_params.dtype, device=design_params.device)

        cache = {}

        def record_transform(rec):
            key = id(rec)
            if key in cache:
                return cache[key]
            parent = torch.eye(4, dtype=design_params.dtype, device=design_params.device)
            if rec.parent is not None:
                parent = record_transform(self.spec.records[rec.parent])
            joint = constant_transform(rec.joint_E) if rec.p1_slice is None else transform(design_params[rec.p1_slice])
            body = constant_transform(rec.body_E) if rec.p2_slice is None else transform(design_params[rec.p2_slice])
            value = parent @ joint @ body
            cache[key] = value
            return value

        if self.anchor_record is None:
            contacts = [design_params[sl].reshape(-1, 3) for sl in self.contact_slices]
            if not contacts:
                return torch.zeros((0, 3), dtype=design_params.dtype, device=design_params.device)
            center = torch.tensor(self.baseline_center, dtype=design_params.dtype, device=design_params.device)
            return torch.cat(contacts, dim=0) - center.reshape(1, 3)

        anchor = self.anchor_record
        parent = torch.eye(4, dtype=design_params.dtype, device=design_params.device)
        if anchor.parent is not None:
            parent = record_transform(self.spec.records[anchor.parent])
        anchor_joint = constant_transform(anchor.joint_E) if anchor.p1_slice is None else transform(design_params[anchor.p1_slice])
        anchor_pos = (parent @ anchor_joint)[:3, 3]

        chunks = []
        for rec, sl in zip(self.contact_records, self.contact_slices):
            local = design_params[sl].reshape(-1, 3)
            if local.numel() == 0:
                continue
            rec_transform = record_transform(rec)
            root_points = local @ rec_transform[:3, :3].T + rec_transform[:3, 3]
            chunks.append(root_points - anchor_pos.reshape(1, 3))
        if not chunks:
            return torch.zeros((0, 3), dtype=design_params.dtype, device=design_params.device)
        return torch.cat(chunks, dim=0)

    def world_contacts_from_anchor(self, design_params: np.ndarray, anchor_world: np.ndarray) -> np.ndarray:
        contacts = self.local_contacts(design_params)
        if contacts.shape[0] == 0:
            return contacts
        anchor = np.asarray(anchor_world, dtype=np.float64).reshape(3)
        return anchor.reshape(1, 3) + contacts - self.baseline_center.reshape(1, 3)

    def accumulate_contact_grad(self, df_dp: np.ndarray, grad_contacts: np.ndarray) -> None:
        grad_contacts = np.asarray(grad_contacts, dtype=np.float64).reshape(-1, 3)
        cursor = 0
        for sl in self.contact_slices:
            n = (sl.stop - sl.start) // 3
            if n <= 0:
                continue
            df_dp[sl] += grad_contacts[cursor : cursor + n].reshape(-1)
            cursor += n

    def softmin_squared_distance(
        self,
        design_params: np.ndarray,
        anchor_world: np.ndarray,
        target_world: np.ndarray,
        *,
        tau: float,
        axes: tuple[int, ...] = (0, 1),
    ) -> dict:
        contacts_world = self.world_contacts_from_anchor(design_params, anchor_world)
        value, weights, grad_axes = _softmin_squared(
            contacts_world,
            np.asarray(target_world, dtype=np.float64),
            tau=tau,
            axes=axes,
        )
        grad_contacts = np.zeros_like(contacts_world)
        if grad_contacts.shape[0]:
            grad_contacts[:, list(axes)] = grad_axes
        grad_target = -np.sum(grad_contacts, axis=0) if grad_contacts.shape[0] else np.zeros(3, dtype=np.float64)
        grad_anchor = np.sum(grad_contacts, axis=0) if grad_contacts.shape[0] else np.zeros(3, dtype=np.float64)
        nearest_index = int(np.argmax(weights)) if weights.shape[0] else -1
        return {
            "value": value,
            "weights": weights,
            "contacts_world": contacts_world,
            "grad_contacts": grad_contacts,
            "grad_anchor": grad_anchor,
            "grad_target": grad_target,
            "nearest_index": nearest_index,
        }


def build_function_group_contact_set(
    bundle,
    *,
    variable_joint: str | None = None,
    name_hint: str | None = None,
) -> FunctionGroupContactSet | None:
    spec = getattr(bundle, "spec", None)
    if spec is None:
        return None

    markers = list(getattr(spec, "marker_records", []) or [])
    tools = list(getattr(spec, "tool_records", []) or [])
    marker = _match_record(markers, joint_name=variable_joint, name_hint=name_hint)
    node_to_tool = {
        int(getattr(rec, "node_id")): rec
        for rec in tools
        if getattr(rec, "node_id", None) is not None
    }

    leaf_records = []
    anchor_record = marker
    if marker is not None:
        leaf_ids = tuple(getattr(marker, "function_group_leaves", ()) or ())
        leaf_records = [node_to_tool[node_id] for node_id in leaf_ids if node_id in node_to_tool]
        if not leaf_records:
            parent_idx = getattr(marker, "parent", None)
            if parent_idx is not None and id(spec.records[parent_idx]) in {id(rec) for rec in tools}:
                leaf_records = _terminal_tool_descendants(spec, spec.records[parent_idx])
    else:
        root = _match_record(tools, joint_name=variable_joint, name_hint=name_hint)
        if root is None:
            return None
        anchor_record = root
        leaf_records = _terminal_tool_descendants(spec, root)

    contact_records = [
        rec
        for rec in leaf_records
        if getattr(rec, "p3_slice", None) is not None and rec.p3_slice.stop > rec.p3_slice.start
    ]
    contact_slices = [rec.p3_slice for rec in contact_records]
    if not contact_slices:
        return None

    baseline = np.asarray(getattr(spec, "baseline", np.zeros(0)), dtype=np.float64)
    chunks = [baseline[sl].reshape(-1, 3) for sl in contact_slices]
    baseline_contacts = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float64)
    return FunctionGroupContactSet(
        name=str(name_hint or variable_joint or "function_group"),
        spec=spec,
        anchor_record=anchor_record,
        leaf_records=list(leaf_records),
        contact_records=contact_records,
        contact_slices=contact_slices,
        baseline_contacts=baseline_contacts,
    )


def point_squared_distance(point: np.ndarray, target: np.ndarray, *, axes: tuple[int, ...] = (0, 1, 2)) -> dict:
    point = np.asarray(point, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = point[list(axes)] - target[list(axes)]
    grad_point = np.zeros(3, dtype=np.float64)
    grad_point[list(axes)] = 2.0 * diff
    return {
        "value": float(np.sum(diff * diff)),
        "grad_point": grad_point,
        "grad_target": -grad_point,
    }
