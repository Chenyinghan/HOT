"""Compile BASS grammar sequences into validated trees and RedMax scenes.

``compile_scene`` owns the complete sequence -> tree -> XML pipeline.
Tree codecs remain available for persisted search results and inspection.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np

from bilevel.upper.bass.actions import (
    ROOT_ROTATION_MODES,
    Action,
    AddLink,
    End,
    SelectRootRotation,
    action_from_dict,
)
from bilevel.upper.bass.connection_geometry import (
    EDGE_CORNER_CONNECTION,
    connection_family,
    resolve_edge_corner_pose,
)


@dataclass(frozen=True)
class SkeletonNode:
    """One node in a skeleton tree.

    Attributes:
        node_id: Stable integer node identifier.
        asset_id: Primitive identifier for this node. The synthetic root uses
            ``None``.
        parent_id: Parent node id. The root uses ``None``.
        children_ids: Ordered child node ids. The order is semantically
            significant because it defines the deterministic DFS serialization
            back into a sequence.
        depth: Root-relative depth, where the synthetic root has depth 0.
        start_function_group: Whether this node roots one functional group.
    """

    node_id: int
    asset_id: str | None
    parent_id: int | None
    children_ids: list[int] = field(default_factory=list)
    depth: int = 0
    start_function_group: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to a JSON-compatible dictionary."""
        return {
            "node_id": self.node_id,
            "asset_id": self.asset_id,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "depth": self.depth,
            "start_function_group": self.start_function_group,
        }


@dataclass(frozen=True)
class Attachment:
    """Labeled parent-child attachment edge.

    Attributes:
        from_node_id: Source node id.
        to_node_id: Destination node id.
        parent_face: Face index on the parent.
        dock_id: Dock index on the parent face.
        facing: In-plane facing index.
        child_face: Face index on the child attached back to the parent.
        child_dock_id: Dock index on the child face.
    """

    from_node_id: int
    to_node_id: int
    parent_face: int
    dock_id: int
    facing: int
    child_face: int
    child_dock_id: int = 0

    def to_dict(self) -> dict[str, int]:
        """Serialize attachment to a JSON-compatible dictionary."""
        return {
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "parent_face": self.parent_face,
            "dock_id": self.dock_id,
            "facing": self.facing,
            "child_face": self.child_face,
            "child_dock_id": self.child_dock_id,
        }


@dataclass(frozen=True)
class SkeletonTree:
    """Ordered rooted tree representation of a skeleton.

    Attributes:
        nodes: Mapping from node id to node object.
        attachments: Ordered list of parent-child edges. The order follows node
            construction order in the source sequence.
        root_id: Synthetic root node id.
        source_sequence: Optional serialized source actions used for debugging
            or round-trip assertions. This is not required for reconstruction.
        root_rotation_mode: Optional searched Handle-centred roll/pitch/yaw
            decision that precedes structural construction.
    """

    nodes: dict[int, SkeletonNode]
    attachments: list[Attachment]
    root_id: int
    source_sequence: list[dict[str, Any]] | None = None
    root_rotation_mode: str | None = None

    def get_node(self, node_id: int) -> SkeletonNode:
        """Return one node by id.

        Raises:
            KeyError: If the node id is not present.
        """
        return self.nodes[node_id]

    def root(self) -> SkeletonNode:
        """Return the root node."""
        return self.get_node(self.root_id)

    def attachment_map(self) -> dict[tuple[int, int], Attachment]:
        """Return fast lookup mapping from edge endpoints to attachment."""
        return {(edge.from_node_id, edge.to_node_id): edge for edge in self.attachments}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tree to a JSON-compatible dictionary."""
        payload = {
            "root_id": self.root_id,
            "nodes": {str(node_id): node.to_dict() for node_id, node in self.nodes.items()},
            "attachments": [edge.to_dict() for edge in self.attachments],
            "source_sequence": self.source_sequence,
        }
        if self.root_rotation_mode is not None:
            payload["root_rotation_mode"] = self.root_rotation_mode
        return payload


def validate_tree(tree: SkeletonTree) -> None:
    """Validate structural invariants of a ``SkeletonTree``.

    Raises:
        ValueError: If any invariant is violated.
    """
    if (
        tree.root_rotation_mode is not None
        and tree.root_rotation_mode not in ROOT_ROTATION_MODES
    ):
        raise ValueError(
            "root_rotation_mode must be one of "
            f"{ROOT_ROTATION_MODES}, got {tree.root_rotation_mode!r}"
        )
    if tree.root_id not in tree.nodes:
        raise ValueError(f"Root node {tree.root_id} is missing from tree.nodes")

    root = tree.nodes[tree.root_id]
    if root.parent_id is not None:
        raise ValueError("Root node must not have a parent_id")
    if root.depth != 0:
        raise ValueError("Root node depth must be 0")

    seen_ids = set()
    for node_id, node in tree.nodes.items():
        if node_id != node.node_id:
            raise ValueError(f"Node mapping key {node_id} does not match node.node_id {node.node_id}")
        if node_id in seen_ids:
            raise ValueError(f"Duplicate node id detected: {node_id}")
        seen_ids.add(node_id)

        if node.parent_id is None:
            if node_id != tree.root_id:
                raise ValueError(f"Only root may have parent_id=None, got node {node_id}")
        else:
            if node.parent_id not in tree.nodes:
                raise ValueError(f"Node {node_id} references missing parent {node.parent_id}")
            parent = tree.nodes[node.parent_id]
            if node_id not in parent.children_ids:
                raise ValueError(
                    f"Node {node_id} claims parent {node.parent_id}, but parent is missing the child"
                )
            if node.depth != parent.depth + 1:
                raise ValueError(
                    f"Node {node_id} depth {node.depth} must equal parent depth {parent.depth} + 1"
                )

        if len(node.children_ids) != len(set(node.children_ids)):
            raise ValueError(f"Node {node_id} contains duplicate child ids in children_ids")
        for child_id in node.children_ids:
            if child_id not in tree.nodes:
                raise ValueError(f"Node {node_id} references missing child {child_id}")
            child = tree.nodes[child_id]
            if child.parent_id != node_id:
                raise ValueError(
                    f"Child {child_id} of node {node_id} points back to parent {child.parent_id}"
                )

    attachment_pairs: set[tuple[int, int]] = set()
    for edge in tree.attachments:
        pair = (edge.from_node_id, edge.to_node_id)
        if pair in attachment_pairs:
            raise ValueError(f"Duplicate attachment for edge {pair}")
        attachment_pairs.add(pair)

        if edge.from_node_id not in tree.nodes:
            raise ValueError(f"Attachment references missing source node {edge.from_node_id}")
        if edge.to_node_id not in tree.nodes:
            raise ValueError(f"Attachment references missing destination node {edge.to_node_id}")

        child = tree.nodes[edge.to_node_id]
        parent = tree.nodes[edge.from_node_id]
        if child.parent_id != edge.from_node_id:
            raise ValueError(
                f"Attachment source mismatch for node {edge.to_node_id}: "
                f"node parent={child.parent_id}, edge source={edge.from_node_id}"
            )
        if edge.to_node_id not in parent.children_ids:
            raise ValueError(
                f"Attachment destination {edge.to_node_id} not present in source "
                f"{edge.from_node_id} children_ids"
            )

    expected_pairs = {
        (node.parent_id, node.node_id)
        for node in tree.nodes.values()
        if node.parent_id is not None
    }
    if attachment_pairs != expected_pairs:
        missing = expected_pairs - attachment_pairs
        extra = attachment_pairs - expected_pairs
        raise ValueError(f"Attachment set mismatch. Missing={missing}, extra={extra}")

    visited: set[int] = set()
    queue = deque([tree.root_id])
    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            raise ValueError(f"Cycle or repeated reachability detected at node {node_id}")
        visited.add(node_id)
        queue.extend(tree.nodes[node_id].children_ids)

    if visited != set(tree.nodes):
        unreachable = set(tree.nodes) - visited
        raise ValueError(f"Tree contains unreachable nodes: {sorted(unreachable)}")


def sequence_to_tree(
    sequence: Sequence[Action],
    *,
    keep_source_sequence: bool = True,
) -> SkeletonTree:
    """Convert a sequence grammar into an ordered rooted skeleton tree.

    Args:
        sequence: Action sequence produced by the upper-level BASS grammar.
        keep_source_sequence: If True, keep a serialized copy of the sequence
            inside the returned tree for debugging and round-trip inspection.

    Returns:
        A validated ``SkeletonTree``.

    Raises:
        ValueError: If the sequence is structurally invalid.
    """
    root_id = 0
    nodes: dict[int, SkeletonNode] = {
        root_id: SkeletonNode(
            node_id=root_id,
            asset_id=None,
            parent_id=None,
            children_ids=[],
            depth=0,
        )
    }
    attachments: list[Attachment] = []
    stack: list[int] = [root_id]
    next_node_id = 1
    completed = False
    root_rotation_mode: str | None = None

    for index, action in enumerate(sequence):
        if completed:
            raise ValueError(
                f"Action sequence continues after root completion at position {index}: {action!r}"
            )

        if isinstance(action, SelectRootRotation):
            if index != 0 or root_rotation_mode is not None:
                raise ValueError(
                    "SelectRootRotation must appear exactly once at position 0"
                )
            root_rotation_mode = action.mode
            continue

        if isinstance(action, AddLink):
            if not stack:
                raise ValueError(f"Cannot AddLink with an empty construction stack at {index}")

            parent_id = stack[-1]
            parent = nodes[parent_id]
            child_id = next_node_id
            next_node_id += 1

            child = SkeletonNode(
                node_id=child_id,
                asset_id=action.asset_id,
                parent_id=parent_id,
                children_ids=[],
                depth=parent.depth + 1,
                start_function_group=action.start_function_group,
            )
            nodes[child_id] = child

            updated_children = [*parent.children_ids, child_id]
            nodes[parent_id] = SkeletonNode(
                node_id=parent.node_id,
                asset_id=parent.asset_id,
                parent_id=parent.parent_id,
                children_ids=updated_children,
                depth=parent.depth,
                start_function_group=parent.start_function_group,
            )

            attachments.append(
                Attachment(
                    from_node_id=parent_id,
                    to_node_id=child_id,
                    parent_face=action.p,
                    dock_id=action.d,
                    facing=action.f,
                    child_face=action.q,
                    child_dock_id=action.child_dock_id,
                )
            )
            stack.append(child_id)
            continue

        if isinstance(action, End):
            if not stack:
                raise ValueError(f"Encountered End with an empty construction stack at {index}")
            if len(stack) == 1:
                stack.pop()
                completed = True
            else:
                stack.pop()
            continue

        raise ValueError(f"Unsupported action type at position {index}: {type(action)!r}")

    if not completed:
        raise ValueError("Sequence ended before closing the root with End().")
    if stack:
        raise ValueError("Internal error: construction stack must be empty after completion.")

    source_sequence = [action.to_dict() for action in sequence] if keep_source_sequence else None
    tree = SkeletonTree(
        nodes=nodes,
        attachments=attachments,
        root_id=root_id,
        source_sequence=source_sequence,
        root_rotation_mode=root_rotation_mode,
    )
    validate_tree(tree)
    return tree


def tree_to_sequence(tree: SkeletonTree) -> list[Action]:
    """Serialize an ordered rooted skeleton tree back into a DFS sequence.

    Args:
        tree: Tree representation with ordered children and labeled edges.

    Returns:
        A sequence of ``AddLink`` and ``End`` actions.
    """
    validate_tree(tree)
    attachment_map = tree.attachment_map()
    sequence: list[Action] = []
    if tree.root_rotation_mode is not None:
        sequence.append(SelectRootRotation(tree.root_rotation_mode))

    def visit(node_id: int) -> None:
        node = tree.get_node(node_id)
        for child_id in node.children_ids:
            child = tree.get_node(child_id)
            edge = attachment_map[(node_id, child_id)]
            sequence.append(
                AddLink(
                    asset_id=child.asset_id or "",
                    p=edge.parent_face,
                    d=edge.dock_id,
                    f=edge.facing,
                    q=edge.child_face,
                    child_dock_id=edge.child_dock_id,
                    start_function_group=child.start_function_group,
                )
            )
            visit(child_id)
            sequence.append(End())

    visit(tree.root_id)
    sequence.append(End())
    return sequence


def tree_to_dict(tree: SkeletonTree) -> dict[str, Any]:
    """Serialize a tree into a plain dictionary."""
    validate_tree(tree)
    return tree.to_dict()


def tree_from_dict(payload: dict[str, Any]) -> SkeletonTree:
    """Deserialize a ``SkeletonTree`` from a plain dictionary.

    Args:
        payload: Serialized tree payload.

    Returns:
        Parsed and validated ``SkeletonTree``.
    """
    nodes_raw = payload.get("nodes")
    if not isinstance(nodes_raw, dict):
        raise ValueError("Serialized tree must contain a 'nodes' mapping")

    attachments_raw = payload.get("attachments")
    if not isinstance(attachments_raw, list):
        raise ValueError("Serialized tree must contain an 'attachments' list")

    root_id = int(payload.get("root_id"))
    root_rotation_mode = payload.get("root_rotation_mode")
    if root_rotation_mode is not None:
        root_rotation_mode = str(root_rotation_mode).strip().lower()
    source_sequence = payload.get("source_sequence")
    if source_sequence is not None and not isinstance(source_sequence, list):
        raise ValueError("'source_sequence' must be a list when provided")

    nodes: dict[int, SkeletonNode] = {}
    for raw_key, raw_node in nodes_raw.items():
        if not isinstance(raw_node, dict):
            raise ValueError(f"Node entry {raw_key!r} must be an object")
        node = SkeletonNode(
            node_id=int(raw_node["node_id"]),
            asset_id=raw_node.get("asset_id"),
            parent_id=None if raw_node.get("parent_id") is None else int(raw_node["parent_id"]),
            children_ids=[int(value) for value in raw_node.get("children_ids", [])],
            depth=int(raw_node["depth"]),
            start_function_group=bool(raw_node.get("start_function_group", False)),
        )
        nodes[int(raw_key)] = node

    attachments: list[Attachment] = []
    for raw_edge in attachments_raw:
        if not isinstance(raw_edge, dict):
            raise ValueError("Attachment entries must be objects")
        attachments.append(
            Attachment(
                from_node_id=int(raw_edge["from_node_id"]),
                to_node_id=int(raw_edge["to_node_id"]),
                parent_face=int(raw_edge["parent_face"]),
                dock_id=int(raw_edge["dock_id"]),
                facing=int(raw_edge["facing"]),
                child_face=int(raw_edge["child_face"]),
                child_dock_id=int(raw_edge.get("child_dock_id", 0)),
            )
        )

    tree = SkeletonTree(
        nodes=nodes,
        attachments=attachments,
        root_id=root_id,
        source_sequence=source_sequence,
        root_rotation_mode=root_rotation_mode,
    )
    validate_tree(tree)
    return tree


def actions_from_bass_result_payload(payload: dict[str, Any]) -> list[Action]:
    """Parse ``best_sequence`` actions from an BASS result payload.

    Args:
        payload: Dictionary produced by ``bilevel.search``.

    Returns:
        List of parsed ``Action`` objects.

    Raises:
        ValueError: If the payload does not contain a valid ``best_sequence``.
    """
    raw_sequence = payload.get("best_sequence")
    if not isinstance(raw_sequence, list):
        raise ValueError("BASS result payload must contain a list field named 'best_sequence'")

    actions: list[Action] = []
    for index, raw_action in enumerate(raw_sequence):
        if not isinstance(raw_action, dict):
            raise ValueError(f"best_sequence[{index}] must be an action object")
        actions.append(action_from_dict(raw_action))
    return actions


def load_actions_from_bass_result_json(path: str | Path) -> list[Action]:
    """Load ``best_sequence`` from an BASS result JSON file as actions."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("BASS result JSON must contain an object at the top level")
    return actions_from_bass_result_payload(payload)


def bass_result_payload_to_tree(
    payload: dict[str, Any],
    *,
    keep_source_sequence: bool = True,
) -> SkeletonTree:
    """Convert an BASS result payload directly into a ``SkeletonTree``."""
    actions = actions_from_bass_result_payload(payload)
    return sequence_to_tree(actions, keep_source_sequence=keep_source_sequence)


def bass_result_json_to_tree(
    path: str | Path,
    *,
    keep_source_sequence: bool = True,
) -> SkeletonTree:
    """Read an BASS result JSON file and convert its best sequence into a tree."""
    actions = load_actions_from_bass_result_json(path)
    return sequence_to_tree(actions, keep_source_sequence=keep_source_sequence)


def actions_to_dicts(actions: Sequence[Action]) -> list[dict[str, Any]]:
    """Serialize actions for debugging or JSON output."""
    return [action.to_dict() for action in actions]


def convert_bass_json_to_tree(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert an BASS result JSON file into a tree JSON payload.

    Args:
        input_path: Path to an BASS result JSON containing ``best_sequence``.
        output_path: Optional path to write the resulting tree payload. When
            omitted, the payload is returned without writing a file.

    Returns:
        JSON-compatible payload containing the source sequence, tree, and
        round-trip sequence.
    """
    input_file = Path(input_path)
    actions = load_actions_from_bass_result_json(input_file)
    tree = bass_result_json_to_tree(input_file)
    tree_payload = tree_to_dict(tree)
    roundtrip_actions = tree_to_sequence(tree)

    action_dicts = actions_to_dicts(actions)
    roundtrip_dicts = actions_to_dicts(roundtrip_actions)
    roundtrip_ok = action_dicts == roundtrip_dicts

    output_payload = {
        "input": str(input_file),
        "roundtrip_ok": roundtrip_ok,
        "source_sequence": action_dicts,
        "tree": tree_payload,
        "roundtrip_sequence": roundtrip_dicts,
    }

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    return output_payload


def flatten_E(E: np.ndarray) -> np.ndarray:
    """Flatten a 4x4 transform matrix to 12-vector consistent with engine ordering.
    First 9 entries are rotation matrix (row-major), last 3 are translation.
    """
    R = E[0:3, 0:3]
    t = E[0:3, 3]
    flat = np.zeros(12, dtype=float)
    flat[0:9] = R.reshape(9)
    flat[9:12] = t
    return flat


def compose_E_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    E = np.eye(4, dtype=float)
    E[0:3, 0:3] = R
    E[0:3, 3] = t
    return E


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion in (w,x,y,z) ordering.
    Uses the standard stable algorithm.
    """
    # ensure shape
    assert R.shape == (3, 3)
    m = R
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2,1] - m[1,2]) / S
        y = (m[0,2] - m[2,0]) / S
        z = (m[1,0] - m[0,1]) / S
    else:
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / S
            x = 0.25 * S
            y = (m[0,1] + m[1,0]) / S
            z = (m[0,2] + m[2,0]) / S
        elif m[1,1] > m[2,2]:
            S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / S
            x = (m[0,1] + m[1,0]) / S
            y = 0.25 * S
            z = (m[1,2] + m[2,1]) / S
        else:
            S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / S
            x = (m[0,2] + m[2,0]) / S
            y = (m[1,2] + m[2,1]) / S
            z = 0.25 * S
    return np.array([w, x, y, z], dtype=float)


PALETTE = [
    (0.85, 0.15, 0.15),  # red
    (0.15, 0.85, 0.15),  # green
    (0.15, 0.15, 0.85),  # blue
    (0.85, 0.65, 0.15),  # orange
    (0.85, 0.15, 0.85),  # magenta
    (0.15, 0.85, 0.85),  # cyan
]


def get_color(index: int, alpha: float = 1.0):
    c = PALETTE[index % len(PALETTE)]
    return (c[0], c[1], c[2], alpha)


TOOL_DESIGN_PARAMS_DEFAULT = 47


ENDEFFECTOR_SIZE_DEFAULT = '0.1 0.1 0.1'


HANDLE_ROOT_JOINT_NAME = 'freeform_root_joint'


ROOT_ROTATION_AXES = {
    'roll': '1 0 0',
    'pitch': '0 1 0',
    'yaw': '0 0 1',
}


def _pretty_xml(elem):
    raw = ET.tostring(elem, 'utf-8')
    parsed = minidom.parseString(raw)
    # Use minidom to pretty print, then remove spurious blank lines
    pretty = parsed.toprettyxml(indent='  ')
    lines = [line for line in pretty.splitlines() if line.strip()]
    return '\n'.join(lines) + '\n'


@dataclass
class GeneratedRegistry:
    joints: Set[str] = field(default_factory=set)
    bodies: Set[str] = field(default_factory=set)
    contact_bodies: Set[str] = field(default_factory=set)
    tool_bodies: Set[str] = field(default_factory=set)
    tool_connected_body_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    marker_bodies: Set[str] = field(default_factory=set)
    marker_joints: List[str] = field(default_factory=list)


def _fmt_vec(values) -> str:
    out = []
    for value in values:
        number = float(value)
        rounded = round(number)
        if abs(number - rounded) < 1e-9:
            out.append(str(int(rounded)))
        else:
            out.append(f'{number:.12g}')
    return ' '.join(out)


def _normalized_barycentric(values) -> List[float]:
    if not isinstance(values, list) or len(values) != 4:
        values = [0.25, 0.25, 0.25, 0.25]
    weights = np.asarray([float(x) for x in values], dtype=float)
    total = float(np.sum(weights))
    if abs(total) <= 1e-12:
        weights = np.asarray([0.25, 0.25, 0.25, 0.25], dtype=float)
    else:
        weights = weights / total
    return [float(x) for x in weights]


def _parse_vec3(text: Any, default: Optional[np.ndarray] = None) -> np.ndarray:
    if default is None:
        default = np.zeros(3, dtype=float)
    if not isinstance(text, str):
        return np.asarray(default, dtype=float)
    parts = text.split()
    if len(parts) != 3:
        return np.asarray(default, dtype=float)
    return np.asarray([float(v) for v in parts], dtype=float)


def _quat_wxyz_to_matrix(text: Any) -> np.ndarray:
    if not isinstance(text, str):
        return np.eye(3, dtype=float)
    parts = text.split()
    if len(parts) != 4:
        return np.eye(3, dtype=float)
    w, x, y, z = [float(v) for v in parts]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return np.eye(3, dtype=float)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _node_local_transform(node: Dict[str, Any]) -> np.ndarray:
    E = np.eye(4, dtype=float)
    E[:3, :3] = _quat_wxyz_to_matrix(node.get('quat', '1 0 0 0'))
    E[:3, 3] = _parse_vec3(node.get('pos', '0 0 0'))
    return E


def _asset_base(asset_id: Optional[str]) -> str:
    if not asset_id:
        return 'root'
    return asset_id.split('/')[-1].replace('-', '_')


def _load_raw_assets_map(assets_json: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not assets_json or not os.path.exists(assets_json):
        return {}
    with open(assets_json, 'r', encoding='utf-8') as f:
        payload = __import__('json').load(f)
    records = payload.get('assets', payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or not item.get('id'):
            continue
        normalized = dict(item)
        resources = normalized.get('resources', {})
        if isinstance(resources, dict):
            for name in ('mesh', 'cage', 'contacts', 'contact_ids', 'weights'):
                resource = resources.get(name)
                if isinstance(resource, dict) and resource.get('path'):
                    normalized[name] = str(resource['path'])
        names = [str(normalized['id'])]
        names.extend(str(value) for value in normalized.get('aliases', []))
        for name in names:
            if name in result:
                raise ValueError(f'Duplicate asset ID or alias in catalog: {name!r}')
            result[name] = normalized
    return result


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[3])


def _resolve_asset_paths(asset_id: Optional[str], raw_assets: Dict[str, Dict[str, Any]], repo_root: str) -> Dict[str, str]:
    if not asset_id:
        return {}
    entry = raw_assets.get(asset_id)
    if entry is None:
        raise ValueError(
            f"Tree asset_id {asset_id!r} is not defined in the assets metadata. "
            "Do not emit XML with guessed mesh paths; regenerate the tree with this assets JSON."
        )
    out: Dict[str, str] = {}
    for key in ('mesh', 'contacts'):
        value = entry.get(key)
        if value:
            out[key] = value if not os.path.isabs(value) else os.path.relpath(value, repo_root)
    family = entry.get('category') or asset_id.split('/', 1)[0]
    primitive = asset_id.split('/', 1)[-1]
    if 'mesh' not in out:
        inferred = os.path.join('assets', family, 'meshes', primitive + '.obj')
        if os.path.exists(os.path.join(repo_root, inferred)):
            out['mesh'] = inferred
    if 'contacts' not in out:
        inferred = os.path.join('assets', family, 'contacts', primitive + '.txt')
        if os.path.exists(os.path.join(repo_root, inferred)):
            out['contacts'] = inferred
    if 'mesh' not in out:
        raise ValueError(f"Tree asset_id {asset_id!r} does not resolve to an existing mesh.")
    return out


def _resolve_existing_resource_path(path_value: str, base_dirs: List[str]) -> Optional[str]:
    if os.path.isabs(path_value):
        return path_value if os.path.exists(path_value) else None
    for base_dir in base_dirs:
        candidate = os.path.normpath(os.path.join(base_dir, path_value))
        if os.path.exists(candidate):
            return candidate
    return None


def _rewrite_resource_paths_for_output(
    root: ET.Element,
    *,
    out_path: str,
    scene_xml: str,
) -> None:
    output_dir = os.path.dirname(os.path.abspath(out_path)) or os.getcwd()
    base_dirs = [
        output_dir,
        os.path.dirname(os.path.abspath(scene_xml)),
        _repo_root(),
    ]
    for body in root.iter('body'):
        for attr in ('mesh', 'contacts'):
            path_value = body.attrib.get(attr)
            if not path_value:
                continue
            resolved = _resolve_existing_resource_path(path_value, base_dirs)
            if resolved is None:
                raise ValueError(f"Generated XML references missing {attr} path: {path_value}")
            body.set(attr, os.path.relpath(resolved, output_dir))


def _ensure_section(root: ET.Element, tag: str) -> ET.Element:
    elem = root.find(tag)
    if elem is None:
        elem = ET.SubElement(root, tag)
    return elem


def _collect_body_names(root: ET.Element) -> Set[str]:
    return {body.attrib['name'] for body in root.iter('body') if body.attrib.get('name')}


def _collect_joint_names(root: ET.Element) -> Set[str]:
    return {joint.attrib['name'] for joint in root.iter('joint') if joint.attrib.get('name')}


def _contact_key(elem: ET.Element) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    if elem.tag == 'general_primitive_contact':
        bodies = sorted(
            (
                elem.attrib.get('general_body', ''),
                elem.attrib.get('primitive_body', ''),
            )
        )
        # A generated reverse declaration is still the same physical body
        # pair.  Preserve the task-authored orientation and parameters instead
        # of appending a second runtime contact for that pair.
        return elem.tag, (('body1', bodies[0]), ('body2', bodies[1]))
    ref_keys = ('body', 'general_body', 'primitive_body', 'body1', 'body2')
    return elem.tag, tuple(sorted((k, elem.attrib.get(k, '')) for k in ref_keys if k in elem.attrib))


def _contact_candidate_bodies(root: ET.Element, *, exclude: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    ground_bodies: Set[str] = set()
    general_bodies: Set[str] = set()
    primitive_bodies: Set[str] = set()
    for body in root.iter('body'):
        name = body.attrib.get('name')
        if not name or name in exclude:
            continue
        if body.attrib.get('collision', 'true').strip().lower() in {'0', 'false', 'no'}:
            continue
        body_type = body.attrib.get('type', 'abstract')
        has_contacts = bool(body.attrib.get('contacts'))
        if body_type == 'abstract':
            if has_contacts:
                ground_bodies.add(name)
                general_bodies.add(name)
        else:
            ground_bodies.add(name)
            primitive_bodies.add(name)
    return ground_bodies, general_bodies, primitive_bodies


def _body_type_by_name(root: ET.Element) -> Dict[str, str]:
    return {
        body.attrib['name']: body.attrib.get('type', 'abstract')
        for body in root.iter('body')
        if body.attrib.get('name')
    }


def _tool_contact_excluded_bodies(root: ET.Element) -> Set[str]:
    """Return scene bodies that opt out of generated tool contacts.

    This is a pair-specific scene policy: the body remains collision-enabled
    for explicit object contacts, but generated tools neither receive a
    runtime contact force nor a preflight collision constraint against it.
    """

    disabled = {'0', 'false', 'no'}
    return {
        body.attrib['name']
        for body in root.iter('body')
        if body.attrib.get('name')
        and body.attrib.get('tool_contact', 'true').strip().lower() in disabled
    }


def _body_contact_topology(root: ET.Element) -> Tuple[Dict[str, int], Set[str], Set[str]]:
    """Return rigid-component ids, fixed-world bodies, and no-ground bodies."""
    component_by_body: Dict[str, int] = {}
    fixed_world_bodies: Set[str] = set()
    no_ground_bodies: Set[str] = set()
    next_component = 0

    def visit_link(
        link: ET.Element,
        parent_component: Optional[int],
        parent_fixed_world: bool,
    ) -> None:
        nonlocal next_component
        joint = link.find('joint')
        joint_type = joint.attrib.get('type', 'fixed') if joint is not None else 'fixed'
        if parent_component is not None and joint_type == 'fixed':
            component = parent_component
            fixed_world = parent_fixed_world
        else:
            component = next_component
            next_component += 1
            fixed_world = parent_component is None and joint_type == 'fixed'

        for body in link.findall('body'):
            name = body.attrib.get('name')
            if not name:
                continue
            component_by_body[name] = component
            if fixed_world:
                fixed_world_bodies.add(name)
            if body.attrib.get('ground_contact', 'true').strip().lower() in {'0', 'false', 'no'}:
                no_ground_bodies.add(name)
        for child in link.findall('link'):
            visit_link(child, component, fixed_world)

    for robot in root.findall('robot'):
        for link in robot.findall('link'):
            visit_link(link, None, False)
    return component_by_body, fixed_world_bodies, no_ground_bodies


def _same_rigid_component(body1: str, body2: str, component_by_body: Dict[str, int]) -> bool:
    component1 = component_by_body.get(body1)
    component2 = component_by_body.get(body2)
    return component1 is not None and component1 == component2


def _normalize_contact_declarations(
    contact: ET.Element,
    *,
    component_by_body: Dict[str, int],
    fixed_world_bodies: Set[str],
    no_ground_bodies: Set[str],
    collision_bodies: Set[str],
) -> None:
    """Remove invalid forces and make Python-only constraints explicit."""
    for elem in list(contact):
        if elem.tag == 'general_contact':
            body1 = elem.attrib.get('body1')
            body2 = elem.attrib.get('body2')
            contact.remove(elem)
            if not body1 or not body2 or _same_rigid_component(body1, body2, component_by_body):
                continue
            contact.append(
                ET.Element(
                    'collision_constraint',
                    {
                        'body1': body1,
                        'body2': body2,
                        'enforcement': 'preflight_design',
                    },
                )
            )
            continue
        if elem.tag == 'ground_contact':
            body = elem.attrib.get('body')
            if body in fixed_world_bodies or body in no_ground_bodies or body not in collision_bodies:
                contact.remove(elem)
            continue
        if elem.tag == 'general_primitive_contact':
            general_body = elem.attrib.get('general_body')
            primitive_body = elem.attrib.get('primitive_body')
            if (
                not general_body
                or not primitive_body
                or general_body not in collision_bodies
                or primitive_body not in collision_bodies
                or _same_rigid_component(general_body, primitive_body, component_by_body)
            ):
                contact.remove(elem)
            continue
        if elem.tag == 'sphere_sphere_contact':
            body1 = elem.attrib.get('body1')
            body2 = elem.attrib.get('body2')
            if (
                not body1
                or not body2
                or body1 not in collision_bodies
                or body2 not in collision_bodies
                or _same_rigid_component(body1, body2, component_by_body)
            ):
                contact.remove(elem)


def _face_axis(face: int) -> int:
    return {0: 2, 1: 0, 2: 1, 3: 0, 4: 2, 5: 1}[face]


def _face_sign(face: int) -> float:
    return {0: 1.0, 1: 1.0, 2: 1.0, 3: -1.0, 4: -1.0, 5: -1.0}[face]


def _opposite_face(face: int) -> int:
    return {0: 4, 1: 3, 2: 5, 3: 1, 4: 0, 5: 2}[face]


def _tangent_axes(face: int):
    axis = _face_axis(face)
    if axis == 0:
        return (1, 2)
    if axis == 1:
        return (0, 2)
    return (0, 1)


def _basis_from_face(face: int):
    n = np.zeros(3)
    n[_face_axis(face)] = _face_sign(face)
    t0, t1 = _tangent_axes(face)
    u = np.zeros(3)
    v = np.zeros(3)
    u[t0] = 1.0
    v[t1] = 1.0
    # ensure right-handed basis: cross(u, v) == n
    if np.dot(np.cross(u, v), n) < 0:
        v = -v
    return u, v, n


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def _face_vertices_local(half_extents, face: int):
    axis = _face_axis(face)
    sign = _face_sign(face)
    t0, t1 = _tangent_axes(face)
    h = np.asarray(half_extents, dtype=float)
    base = np.zeros(3)
    base[axis] = sign * h[axis]

    def point(s0, s1):
        p = base.copy()
        p[t0] += s0 * h[t0]
        p[t1] += s1 * h[t1]
        return p

    return [
        point(-1.0, -1.0),
        point(1.0, -1.0),
        point(1.0, 1.0),
        point(-1.0, 1.0),
    ]


def _anchor_local_from_dock(half_extents, face: int, dock: Dict[str, Any]) -> np.ndarray:
    verts = _face_vertices_local(half_extents, face)
    bary = dock.get('barycentric', [0.25, 0.25, 0.25, 0.25])
    if not isinstance(bary, list) or len(bary) != 4:
        bary = [0.25, 0.25, 0.25, 0.25]
    w = np.asarray([float(x) for x in bary], dtype=float)
    w = w / (np.sum(w) + 1e-12)
    out = np.zeros(3)
    for i in range(4):
        out += w[i] * verts[i]
    return out


def _tree_payload_to_simple_tree(
    payload: Dict[str, Any],
    *,
    assets_json: Optional[str] = None,
    repo_root: Optional[str] = None,
    root_asset_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert canonical SkeletonTree JSON into the dict format used by the XML emitter."""
    repo_root = repo_root or _repo_root()
    raw_assets = _load_raw_assets_map(assets_json)
    nodes = payload.get('nodes', {})
    if not isinstance(nodes, dict):
        raise ValueError("Tree payload must contain a 'nodes' mapping")
    attachments = payload.get('attachments', [])
    root_id = int(payload.get('root_id', 0))

    try:
        from bilevel.upper.bass.io_assets import load_assets
        specs = load_assets(assets_json) if assets_json else []
        spec_by_id = {}
        for spec in specs:
            names = [
                spec.asset_id,
                spec.canonical_id or spec.asset_id,
                *(spec.aliases or []),
            ]
            for name in names:
                existing = spec_by_id.get(name)
                if existing is not None and existing.asset_id != spec.asset_id:
                    raise ValueError(
                        f"Duplicate asset ID or alias in catalog: {name!r}"
                    )
                spec_by_id[name] = spec
    except Exception:
        spec_by_id = {}

    attach_by_child = {int(a['to_node_id']): a for a in attachments if 'to_node_id' in a}
    geom_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def load_geom(asset_id: Optional[str]) -> Dict[str, np.ndarray]:
        key = asset_id or '__root__'
        if key in geom_cache:
            return geom_cache[key]
        if asset_id is None and root_asset_id is not None:
            geom = dict(load_geom(root_asset_id))
            geom_cache[key] = geom
            return geom
        spec = spec_by_id.get(asset_id) if asset_id else None
        pts = None
        cage_path = getattr(spec, 'cage_path', None) if spec is not None else None
        if cage_path:
            abs_path = cage_path if os.path.isabs(cage_path) else os.path.join(repo_root, cage_path)
            if os.path.exists(abs_path):
                try:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    count = int(lines[0])
                    vals = [[float(x) for x in ln.split()[:3]] for ln in lines[1:1 + count]]
                    pts = np.asarray(vals, dtype=float) if vals else None
                except Exception:
                    pts = None
        if pts is None:
            half = np.asarray(getattr(spec, 'half_extents', (0.5, 0.5, 0.5)), dtype=float)
            mn, mx = -half, half
        else:
            mn, mx = pts.min(axis=0), pts.max(axis=0)
        geom = {'min': mn, 'max': mx, 'center': 0.5 * (mn + mx), 'half': 0.5 * (mx - mn)}
        geom_cache[key] = geom
        return geom

    def face_vertices(geom: Dict[str, np.ndarray], face: int):
        mn, mx = geom['min'], geom['max']
        x0, y0, z0 = mn
        x1, y1, z1 = mx
        if face == 0:
            return [np.array([x0, y0, z1]), np.array([x1, y0, z1]), np.array([x1, y1, z1]), np.array([x0, y1, z1])]
        if face == 4:
            return [np.array([x0, y0, z0]), np.array([x1, y0, z0]), np.array([x1, y1, z0]), np.array([x0, y1, z0])]
        if face == 1:
            return [np.array([x1, y0, z0]), np.array([x1, y1, z0]), np.array([x1, y1, z1]), np.array([x1, y0, z1])]
        if face == 3:
            return [np.array([x0, y0, z0]), np.array([x0, y1, z0]), np.array([x0, y1, z1]), np.array([x0, y0, z1])]
        if face == 2:
            return [np.array([x0, y1, z0]), np.array([x1, y1, z0]), np.array([x1, y1, z1]), np.array([x0, y1, z1])]
        return [np.array([x0, y0, z0]), np.array([x1, y0, z0]), np.array([x1, y0, z1]), np.array([x0, y0, z1])]

    def anchor_on_face(geom: Dict[str, np.ndarray], face: int, dock: Dict[str, Any]) -> np.ndarray:
        verts = face_vertices(geom, face)
        bary = dock.get('barycentric', [0.25, 0.25, 0.25, 0.25])
        if not isinstance(bary, list) or len(bary) != 4:
            bary = [0.25, 0.25, 0.25, 0.25]
        weights = np.asarray([float(x) for x in bary], dtype=float)
        weights = weights / (np.sum(weights) + 1e-12)
        return sum(weights[i] * verts[i] for i in range(4))

    simple_nodes: Dict[int, Dict[str, Any]] = {}
    for raw_id, node in nodes.items():
        node_id = int(raw_id)
        if node_id == root_id:
            continue
        asset_id = node.get('asset_id')
        base = _asset_base(asset_id)
        paths = _resolve_asset_paths(asset_id, raw_assets, repo_root)
        asset_record = raw_assets.get(asset_id, {})
        canonical_asset_id = str(asset_record.get('id') or asset_id)
        simple_nodes[node_id] = {
            'node_id': node_id,
            'asset_id': canonical_asset_id,
            'start_function_group': bool(node.get('start_function_group', False)),
            'name': f'{base}_{node_id}',
            'body': {
                'type': 'abstract',
                'mesh': paths.get('mesh'),
                'contacts': paths.get('contacts'),
                'mass': 1.0,
                'inertia': [1.0, 1.0, 1.0],
            },
            'children': [],
        }

    bfs_order = []
    queue = list(nodes.get(str(root_id), {}).get('children_ids', []))
    while queue:
        node_id = int(queue.pop(0))
        bfs_order.append(node_id)
        queue.extend(nodes.get(str(node_id), {}).get('children_ids', []))
    for raw_id in nodes:
        node_id = int(raw_id)
        if node_id != root_id and node_id not in bfs_order:
            bfs_order.append(node_id)

    for node_id in bfs_order:
        node = nodes.get(str(node_id), {})
        if node_id not in simple_nodes:
            continue
        attach = attach_by_child.get(node_id)
        if attach is None:
            simple_nodes[node_id]['pos'] = '0 0 0'
            simple_nodes[node_id]['quat'] = '1 0 0 0'
            simple_nodes[node_id]['body']['pos'] = '0 0 0'
            simple_nodes[node_id]['body']['quat'] = '1 0 0 0'
            continue

        parent_id = int(node.get('parent_id') if node.get('parent_id') is not None else root_id)
        parent_asset_id = nodes.get(str(parent_id), {}).get('asset_id')
        parent_lookup_asset_id = root_asset_id if parent_id == root_id else parent_asset_id
        parent_geom = load_geom(None if parent_id == root_id else parent_asset_id)
        child_geom = load_geom(node.get('asset_id'))

        parent_face = int(attach.get('parent_face', 0))
        child_face = int(attach.get('child_face', 0))
        dock_id = int(attach.get('dock_id', 0))
        child_dock_id = int(attach.get('child_dock_id', 0))
        facing = int(attach.get('facing', 0))

        parent_spec = spec_by_id.get(parent_lookup_asset_id)
        docks = parent_spec.out_docks_for_face(parent_face) if parent_spec is not None else []
        dock = next((d for d in docks if int(d.get('id', -1)) == dock_id), None)
        if dock is None:
            dock = {'id': dock_id, 'barycentric': [0.25, 0.25, 0.25, 0.25]}
        child_spec = spec_by_id.get(node.get('asset_id'))
        child_docks = child_spec.in_docks_for_face(child_face) if child_spec is not None else []
        child_dock = next((d for d in child_docks if int(d.get('id', -1)) == child_dock_id), None)
        if child_dock is None:
            child_dock = {'id': child_dock_id, 'barycentric': [0.25, 0.25, 0.25, 0.25]}
        parent_barycentric = _normalized_barycentric(dock.get('barycentric'))
        child_barycentric = _normalized_barycentric(child_dock.get('barycentric'))

        anchor_parent_body = anchor_on_face(parent_geom, parent_face, dock)
        anchor_parent_joint = anchor_parent_body if parent_id == root_id else anchor_parent_body - parent_geom['center']

        child_dock_anchor = anchor_on_face(child_geom, child_face, child_dock) - child_geom['center']
        family = connection_family(dock, child_dock)
        if family == EDGE_CORNER_CONNECTION:
            R, target_anchor = resolve_edge_corner_pose(
                parent_rotation=np.eye(3, dtype=float),
                parent_anchor=anchor_parent_joint,
                parent_half_extents=parent_geom['half'],
                parent_face=parent_face,
                parent_port=dock,
                child_half_extents=child_geom['half'],
                child_face=child_face,
                child_port=child_dock,
                orientation=facing,
            )
            joint_pos = target_anchor - (R @ child_dock_anchor)
        else:
            # Keep the established face-to-face transform unchanged for every
            # pair that is not a compatible edge-edge connection.
            u_p, _, n_p = _basis_from_face(parent_face)
            u_q, v_q, n_q = _basis_from_face(child_face)
            n_target = -n_p
            angle = (facing % 4) * (math.pi / 2.0)
            u_target = _rodrigues(u_p, n_target, angle)
            u_target = u_target - n_target * np.dot(u_target, n_target)
            u_target /= (np.linalg.norm(u_target) + 1e-12)
            v_target = np.cross(n_target, u_target)
            v_target /= (np.linalg.norm(v_target) + 1e-12)
            R = np.column_stack([u_target, v_target, n_target]) @ np.column_stack([u_q, v_q, n_q]).T
            joint_pos = anchor_parent_joint - (R @ child_dock_anchor)
        quat = rotation_matrix_to_quaternion(R)
        body_pos = -child_geom['center']

        simple_nodes[node_id]['pos'] = _fmt_vec(joint_pos)
        simple_nodes[node_id]['quat'] = _fmt_vec(quat)
        simple_nodes[node_id]['body']['pos'] = _fmt_vec(body_pos)
        simple_nodes[node_id]['body']['quat'] = '1 0 0 0'
        terminal_face = _opposite_face(child_face)
        terminal_barycentric = [0.25, 0.25, 0.25, 0.25]
        terminal_anchor = anchor_on_face(
            child_geom,
            terminal_face,
            {'id': 0, 'barycentric': terminal_barycentric},
        ) - child_geom['center']
        simple_nodes[node_id]['endeffector_pos'] = _fmt_vec(terminal_anchor)
        simple_nodes[node_id]['endeffector_face'] = terminal_face
        simple_nodes[node_id]['endeffector_barycentric'] = terminal_barycentric
        simple_nodes[node_id]['planar_connection'] = {
            'parent_face': parent_face,
            'child_face': child_face,
            'dock_id': dock_id,
            'child_dock_id': child_dock_id,
            'facing': facing,
            'parent_barycentric': parent_barycentric,
            'child_barycentric': child_barycentric,
            'family': family,
            'parent_edge_tangent_axis': dock.get('edge_tangent_axis'),
            'parent_panel_normal_axis': dock.get('panel_normal_axis'),
            'child_edge_tangent_axis': child_dock.get('edge_tangent_axis'),
            'child_panel_normal_axis': child_dock.get('panel_normal_axis'),
        }

    for raw_id, node in nodes.items():
        node_id = int(raw_id)
        if node_id == root_id or node_id not in simple_nodes:
            continue
        for child_id in node.get('children_ids', []):
            child_id = int(child_id)
            if child_id in simple_nodes:
                simple_nodes[node_id]['children'].append(simple_nodes[child_id])

    root_children = [int(v) for v in nodes.get(str(root_id), {}).get('children_ids', [])]
    if len(root_children) == 1:
        return simple_nodes[root_children[0]]
    return {
        'name': 'synthetic_root',
        'body': {'type': 'abstract'},
        'children': [simple_nodes[cid] for cid in root_children if cid in simple_nodes],
    }


def _make_tool_link(
    node: Dict[str, Any],
    *,
    attach_index: int,
    design_params: int,
    registry: GeneratedRegistry,
    link_by_node_id: Optional[Dict[int, ET.Element]] = None,
    parent_body_name: Optional[str] = None,
    color_idx: int = 0,
) -> ET.Element:
    node_id = int(node.get('node_id', color_idx))
    base = _asset_base(node.get('asset_id') or node.get('name'))
    link_name = f'link_tool_{base}_{attach_index}_{node_id}'
    joint_name = f'joint_tool_{base}_{attach_index}_{node_id}'
    body_name = f'body_tool_{base}_{attach_index}_{node_id}'

    asset_id = str(node.get('asset_id') or '')
    link_attrib = {
        'name': link_name,
        'design_params': str(int(design_params)),
        'node_id': str(node_id),
        'asset_role': 'head',
    }
    if asset_id:
        link_attrib['asset_id'] = asset_id
    if bool(node.get('start_function_group', False)):
        link_attrib['start_function_group'] = 'true'
    planar_connection = node.get('planar_connection')
    if isinstance(planar_connection, dict):
        for key in ('parent_face', 'child_face', 'dock_id', 'child_dock_id', 'facing'):
            if key in planar_connection:
                link_attrib[f'planar_{key}'] = str(int(planar_connection[key]))
        if 'parent_barycentric' in planar_connection:
            link_attrib['planar_parent_barycentric'] = _fmt_vec(planar_connection['parent_barycentric'])
        if 'child_barycentric' in planar_connection:
            link_attrib['planar_child_barycentric'] = _fmt_vec(planar_connection['child_barycentric'])
        if planar_connection.get('family'):
            link_attrib['planar_connection_family'] = str(planar_connection['family'])
        for key in (
            'parent_edge_tangent_axis',
            'parent_panel_normal_axis',
            'child_edge_tangent_axis',
            'child_panel_normal_axis',
        ):
            if planar_connection.get(key) is not None:
                link_attrib[f'planar_{key}'] = str(int(planar_connection[key]))
    if node.get('endeffector_face') is not None:
        link_attrib['planar_endeffector_face'] = str(int(node['endeffector_face']))
        link_attrib['planar_endeffector_barycentric'] = _fmt_vec(
            node.get('endeffector_barycentric', [0.25, 0.25, 0.25, 0.25])
        )

    link = ET.Element('link', link_attrib)
    if link_by_node_id is not None:
        link_by_node_id[node_id] = link
    ET.SubElement(
        link,
        'joint',
        {
            'name': joint_name,
            'type': node.get('joint', {}).get('type', 'fixed'),
            'pos': node.get('pos', '0 0 0'),
            'quat': node.get('quat', '1 0 0 0'),
        },
    )

    body_info = node.get('body', {})
    body_attrib = {
        'name': body_name,
        'type': body_info.get('type', 'abstract'),
        'pos': body_info.get('pos', '0 0 0'),
        'quat': body_info.get('quat', '1 0 0 0'),
        'rgba': ' '.join(map(str, node.get('rgba', get_color(color_idx)))),
        'asset_role': 'head',
    }
    if asset_id:
        body_attrib['asset_id'] = asset_id
    if body_attrib['type'] == 'abstract':
        if body_info.get('mesh'):
            body_attrib['mesh'] = body_info['mesh']
        if body_info.get('contacts'):
            body_attrib['contacts'] = body_info['contacts']
        body_attrib['mass'] = str(body_info.get('mass', 1.0))
        body_attrib['inertia'] = ' '.join(map(str, body_info.get('inertia', [1.0, 1.0, 1.0])))
    else:
        if body_info.get('size') is not None:
            body_attrib['size'] = ' '.join(map(str, body_info['size']))
        if body_info.get('density') is not None:
            body_attrib['density'] = str(body_info['density'])
    ET.SubElement(link, 'body', body_attrib)

    registry.joints.add(joint_name)
    registry.bodies.add(body_name)
    registry.tool_bodies.add(body_name)
    if parent_body_name and parent_body_name in registry.tool_bodies:
        registry.tool_connected_body_pairs.add(tuple(sorted((parent_body_name, body_name))))
    if body_info.get('contacts') or body_attrib['type'] != 'abstract':
        registry.contact_bodies.add(body_name)

    for idx, child in enumerate(node.get('children', [])):
        link.append(
            _make_tool_link(
                child,
                attach_index=attach_index,
                design_params=design_params,
                registry=registry,
                link_by_node_id=link_by_node_id,
                parent_body_name=body_name,
                color_idx=color_idx + idx + 1,
            )
        )
    return link


def _make_fixed_root_link(
    *,
    attach_index: int,
    root_asset_id: str,
    raw_assets: Dict[str, Dict[str, Any]],
    repo_root: str,
    design_params: int,
    pos: str,
    quat: str,
    registry: GeneratedRegistry,
    parent_body_name: Optional[str] = None,
) -> ET.Element:
    """Create the immutable Handle asset below the actuation carrier."""
    base = _asset_base(root_asset_id)
    paths = _resolve_asset_paths(root_asset_id, raw_assets, repo_root)
    asset_record = raw_assets.get(root_asset_id, {})
    asset_role = str(asset_record.get('role') or 'root')
    link_name = f'tip_{base}_{attach_index}'
    joint_name = f'joint_tip_{base}_{attach_index}'
    body_name = f'body_tip_{base}_{attach_index}'

    link = ET.Element(
        'link',
        {
            'name': link_name,
            'design_params': str(int(design_params)),
            'asset_id': str(asset_record.get('id') or root_asset_id),
            'asset_role': asset_role,
        },
    )
    ET.SubElement(
        link,
        'joint',
        {
            'name': joint_name,
            'type': 'fixed',
            'pos': pos,
            'quat': quat,
        },
    )
    ET.SubElement(
        link,
        'body',
        {
            'name': body_name,
            'type': 'abstract',
            'pos': '0 0 0',
            'quat': '1 0 0 0',
            'rgba': '0.8 0.8 0.2 1',
            'mesh': paths.get('mesh'),
            'contacts': paths.get('contacts'),
            'mass': '1.0',
            'inertia': '1.0 1.0 1.0',
            'asset_id': str(asset_record.get('id') or root_asset_id),
            'asset_role': asset_role,
        },
    )

    registry.joints.add(joint_name)
    registry.bodies.add(body_name)
    registry.tool_bodies.add(body_name)
    registry.contact_bodies.add(body_name)
    return link


def _weld_direct_fixed_root_children(root_link: ET.Element) -> int:
    """Mark only fixed-root-to-Head morphology boundaries as welded."""

    if root_link.attrib.get('asset_role') != 'fixed_root':
        return 0
    welded = 0
    for child in root_link.findall('link'):
        if child.attrib.get('asset_role') != 'head':
            continue
        if 'planar_parent_face' not in child.attrib or 'planar_child_face' not in child.attrib:
            raise ValueError(
                "Direct fixed-root Head lacks planar attachment metadata: "
                f"{child.attrib.get('name')!r}"
            )
        child.attrib.update(
            {
                'welded_interface': 'true',
                'welded_socket_id': (
                    f"face{child.attrib['planar_parent_face']}:"
                    f"dock{child.attrib.get('planar_dock_id', '0')}"
                ),
                'welded_parent_kind': 'handle',
                'welded_parent_id': root_link.attrib.get('name', ''),
            }
        )
        welded += 1
    return welded


def _remove_contacts_for_bodies(root: ET.Element, body_names: Set[str]) -> None:
    """Remove generated task contacts for non-contact fixed-root geometry."""

    contact = root.find('contact')
    if contact is None or not body_names:
        return
    reference_keys = {
        'body',
        'body1',
        'body2',
        'general_body',
        'primitive_body',
    }
    for element in list(contact):
        referenced = {
            value
            for key, value in element.attrib.items()
            if key in reference_keys
        }
        if referenced.intersection(body_names):
            contact.remove(element)


def _finalize_fixed_asset_root(root: ET.Element, root_links: List[ET.Element]) -> None:
    """Apply native weld/contact/dynamics semantics to fixed Handle roots."""

    fixed_roots = [
        link
        for link in root_links
        if link.attrib.get('asset_role') == 'fixed_root'
    ]
    if not fixed_roots:
        return
    welded = sum(_weld_direct_fixed_root_children(link) for link in fixed_roots)
    if welded == 0:
        raise ValueError("Fixed asset root has no directly attached Head")
    fixed_body_names = {
        body.attrib['name']
        for link in fixed_roots
        for body in link.findall('body')
        if body.attrib.get('name')
    }
    _remove_contacts_for_bodies(root, fixed_body_names)

    from bilevel.lower.proxy_dynamics import apply_proxy_dynamics

    apply_proxy_dynamics(root)


def _make_handle_root_robot(
    *,
    joint_type: str,
    joint_pos: str,
    joint_quat: str,
    joint_damping: Optional[str],
    root_body_size: str,
    auxiliary_joint_type: Optional[str],
    auxiliary_joint_name: str,
    auxiliary_joint_pos: str,
    auxiliary_joint_quat: str,
    auxiliary_joint_axis: str,
    auxiliary_joint_axis1: str,
    auxiliary_joint_damping: Optional[str],
    auxiliary_joint_lim: Optional[str],
    auxiliary_joint_lim_stiffness: Optional[str],
    registry: GeneratedRegistry,
) -> Tuple[ET.Element, ET.Element, str, Optional[str]]:
    """Create the non-design carrier that actuates the fixed Handle root."""
    if joint_type not in {
        'free3d',
        'free3d-euler',
        'free3d-exp',
        'free3d-exp-decoupled',
        'translational',
        'planar',
    }:
        raise ValueError(f"Unsupported freeform root_joint_type: {joint_type!r}")
    robot = ET.Element('robot')
    link = ET.SubElement(robot, 'link', {'name': 'freeform_root', 'design_params': '0'})
    joint_attrib = {
        'name': HANDLE_ROOT_JOINT_NAME,
        'type': joint_type,
        'pos': joint_pos,
        'quat': joint_quat,
    }
    if joint_damping is not None and str(joint_damping).strip():
        joint_attrib['damping'] = str(joint_damping)
    if joint_type == 'planar':
        joint_attrib.update({'axis0': '1 0 0', 'axis1': '0 1 0'})
    ET.SubElement(link, 'joint', joint_attrib)
    body_name = 'body_freeform_root'
    ET.SubElement(
        link,
        'body',
        {
            'name': body_name,
            'type': 'cuboid',
            'pos': '0 0 0',
            'quat': '1 0 0 0',
            'size': root_body_size,
            'density': '1e-9',
            'rgba': '0.7 0.7 0.7 0.15',
        },
    )
    registry.joints.add(HANDLE_ROOT_JOINT_NAME)
    registry.bodies.add(body_name)
    registry.marker_bodies.add(body_name)

    auxiliary_type = str(auxiliary_joint_type or '').strip().lower()
    if not auxiliary_type:
        return robot, link, HANDLE_ROOT_JOINT_NAME, None
    if auxiliary_type not in {'fixed', 'revolute', 'prismatic', 'planar'}:
        raise ValueError(
            "Unsupported root auxiliary joint type: "
            f"{auxiliary_joint_type!r}"
        )

    auxiliary_link = ET.SubElement(
        link,
        'link',
        {
            'name': 'root_auxiliary_carrier',
            'design_params': '0',
        },
    )
    auxiliary_attrib = {
        'name': str(auxiliary_joint_name),
        'type': auxiliary_type,
        'pos': str(auxiliary_joint_pos),
        'quat': str(auxiliary_joint_quat),
    }
    if auxiliary_type in {'revolute', 'prismatic'}:
        auxiliary_attrib['axis'] = str(auxiliary_joint_axis)
    elif auxiliary_type == 'planar':
        auxiliary_attrib['axis0'] = str(auxiliary_joint_axis)
        auxiliary_attrib['axis1'] = str(auxiliary_joint_axis1)
    if (
        auxiliary_joint_damping is not None
        and str(auxiliary_joint_damping).strip()
    ):
        auxiliary_attrib['damping'] = str(auxiliary_joint_damping)
    if (
        auxiliary_joint_lim is not None
        and str(auxiliary_joint_lim).strip()
    ):
        auxiliary_attrib['lim'] = str(auxiliary_joint_lim)
    if (
        auxiliary_joint_lim_stiffness is not None
        and str(auxiliary_joint_lim_stiffness).strip()
    ):
        auxiliary_attrib['lim_stiffness'] = str(
            auxiliary_joint_lim_stiffness
        )
    ET.SubElement(auxiliary_link, 'joint', auxiliary_attrib)

    auxiliary_body_name = 'body_root_auxiliary_carrier'
    ET.SubElement(
        auxiliary_link,
        'body',
        {
            'name': auxiliary_body_name,
            'type': 'cuboid',
            'pos': '0 0 0',
            'quat': '1 0 0 0',
            'size': root_body_size,
            'mass': '1e-4',
            'inertia': '1e-4 1e-4 1e-4',
            'rgba': '0 0 0 0',
            'collision': 'false',
            'ground_contact': 'false',
        },
    )
    registry.joints.add(str(auxiliary_joint_name))
    registry.bodies.add(auxiliary_body_name)
    auxiliary_actuator_joint = (
        None if auxiliary_type == 'fixed' else str(auxiliary_joint_name)
    )
    return (
        robot,
        auxiliary_link,
        HANDLE_ROOT_JOINT_NAME,
        auxiliary_actuator_joint,
    )


def _leaf_groups(simple_tree: Dict[str, Any]) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    groups: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []

    def visit(node: Dict[str, Any]):
        children = node.get('children', [])
        leaf_children = [child for child in children if not child.get('children')]
        if leaf_children:
            groups.append((node, leaf_children))
        for child in children:
            visit(child)

    if simple_tree.get('name') == 'synthetic_root':
        for child in simple_tree.get('children', []):
            visit(child)
    else:
        visit(simple_tree)
        if not simple_tree.get('children'):
            groups.append((simple_tree, [simple_tree]))
    return groups


def _leaf_functions(simple_tree: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    leaves: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    def visit(parent: Dict[str, Any], node: Dict[str, Any]):
        children = node.get('children', [])
        if not children:
            leaves.append((parent, node))
            return
        for child in children:
            visit(node, child)

    if simple_tree.get('name') == 'synthetic_root':
        for child in simple_tree.get('children', []):
            visit(simple_tree, child)
    else:
        visit(simple_tree, simple_tree)
    return leaves


def _terminal_leaves_under(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    leaves: List[Dict[str, Any]] = []

    def visit(cur: Dict[str, Any]) -> None:
        children = cur.get('children', [])
        if not children:
            leaves.append(cur)
            return
        for child in children:
            visit(child)

    visit(node)
    return leaves


def _function_groups(simple_tree: Dict[str, Any]) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Return selected function-group roots and their terminal leaves.

    If no node carries ``start_function_group=True``, fall back to the legacy
    one-terminal-leaf-per-function interpretation for old cached sequences.
    """
    groups: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    uncovered_leaves: List[Dict[str, Any]] = []

    def visit(node: Dict[str, Any], active_root: Optional[Dict[str, Any]]) -> None:
        starts_group = bool(node.get('start_function_group', False))
        if starts_group:
            if active_root is not None:
                raise ValueError("Nested start_function_group annotations are not allowed")
            active_root = node
            groups.append((node, _terminal_leaves_under(node)))

        children = node.get('children', [])
        if not children and active_root is None:
            uncovered_leaves.append(node)
            return
        for child in children:
            visit(child, active_root)

    if simple_tree.get('name') == 'synthetic_root':
        roots = simple_tree.get('children', [])
    else:
        roots = [simple_tree]
    for root in roots:
        visit(root, None)

    if not groups:
        return [(leaf_node, [leaf_node]) for _, leaf_node in _leaf_functions(simple_tree)]
    if uncovered_leaves:
        ids = [leaf.get('node_id', leaf.get('name', '?')) for leaf in uncovered_leaves]
        raise ValueError(f"Terminal leaves outside function groups: {ids}")
    return groups


def _average_leaf_endpoint_in_group_frame(
    group_root: Dict[str, Any],
    leaves: List[Dict[str, Any]],
) -> np.ndarray:
    """Return the public function marker in the group-root frame.

    A single-leaf function retains the terminal working-face point.  A
    multi-leaf function uses the equal-weight mean of the leaf geometry
    centers.  Generated tool-link frames are centered on their asset
    geometries, so the multi-leaf local point is the leaf-frame origin.
    """

    points: List[np.ndarray] = []
    wanted_ids = {id(leaf) for leaf in leaves}
    use_leaf_centers = len(leaves) > 1

    def visit(node: Dict[str, Any], E_group_node: np.ndarray) -> None:
        children = node.get('children', [])
        if not children and id(node) in wanted_ids:
            local = (
                np.zeros(3, dtype=float)
                if use_leaf_centers
                else _parse_vec3(node.get('endeffector_pos', '0 0 0'))
            )
            points.append(
                E_group_node[:3, :3] @ local
                + E_group_node[:3, 3]
            )
            return
        for child in children:
            visit(child, E_group_node @ _node_local_transform(child))

    visit(group_root, np.eye(4, dtype=float))
    if not points:
        raise ValueError(
            f"Function group root {group_root.get('node_id', group_root.get('name', '?'))!r} "
            "contains no terminal leaves"
        )
    return np.mean(np.stack(points, axis=0), axis=0)


def _normalize_function_entries(functions: Optional[List[Dict[str, Any]]], expected_count: int) -> List[Dict[str, str]]:
    if functions is None:
        raise ValueError(
            "XML end-effector generation requires one function entry per searched function. "
            "Provide task.json field: \"functions\": [{\"function_name\": \"...\"}, ...]."
        )
    if not isinstance(functions, list):
        raise ValueError("'functions' must be a list of objects containing function_name")
    if len(functions) != expected_count:
        raise ValueError(
            f"Expected {expected_count} function entries for generated end-effectors, got {len(functions)}"
        )

    seen: Set[str] = set()
    normalized: List[Dict[str, str]] = []
    for idx, item in enumerate(functions):
        if not isinstance(item, dict):
            raise ValueError(f"functions[{idx}] must be an object containing function_name")
        name = str(item.get('function_name', '')).strip()
        if not name:
            raise ValueError(f"functions[{idx}] must contain a non-empty function_name")
        safe = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in name)
        if not safe or safe[0].isdigit():
            safe = f'function_{safe}'
        if safe in seen:
            raise ValueError(f"Duplicate function_name after XML-safe normalization: {safe}")
        seen.add(safe)
        normalized.append({'function_name': safe})
    return normalized


def _add_endeffector_markers(
    parent_link: ET.Element,
    simple_tree: Dict[str, Any],
    *,
    attach_index: int,
    registry: GeneratedRegistry,
    functions: Optional[List[Dict[str, Any]]],
    link_by_node_id: Optional[Dict[int, ET.Element]] = None,
) -> None:
    function_groups = _function_groups(simple_tree)
    function_entries = _normalize_function_entries(functions, len(function_groups))
    for group_idx, (group_root, leaf_nodes) in enumerate(function_groups):
        function_name = function_entries[group_idx]['function_name']
        target_parent = parent_link
        group_id = group_root.get('node_id')
        if link_by_node_id is not None and group_id is not None:
            target_parent = link_by_node_id.get(int(group_id), target_parent)
        center = _average_leaf_endpoint_in_group_frame(group_root, leaf_nodes)
        if len(leaf_nodes) == 1 and link_by_node_id is not None:
            leaf_id = leaf_nodes[0].get('node_id')
            if leaf_id is not None and int(leaf_id) in link_by_node_id:
                target_parent = link_by_node_id[int(leaf_id)]
                center = _parse_vec3(leaf_nodes[0].get('endeffector_pos', '0 0 0'))
        marker_name = f'{function_name}_endeffector'
        if attach_index != 0:
            marker_name = f'{function_name}_{attach_index}_endeffector'
        joint_name = marker_name
        leaf_ids = [
            str(leaf.get('node_id'))
            for leaf in leaf_nodes
            if leaf.get('node_id') is not None
        ]
        link_attrib = {
            'name': marker_name,
            'design_params': '1',
            'function_group_root': '' if group_id is None else str(int(group_id)),
            'function_group_leaves': ','.join(leaf_ids),
            'function_group_leaf_count': str(len(leaf_nodes)),
        }
        if (
            len(leaf_nodes) == 1
            and leaf_nodes[0].get('endeffector_face') is not None
        ):
            leaf_node = leaf_nodes[0]
            link_attrib['planar_parent_face'] = str(int(leaf_node['endeffector_face']))
            link_attrib['planar_parent_barycentric'] = _fmt_vec(
                leaf_node.get('endeffector_barycentric', [0.25, 0.25, 0.25, 0.25])
            )
        link = ET.SubElement(target_parent, 'link', link_attrib)
        ET.SubElement(link, 'joint', {'name': joint_name, 'type': 'fixed', 'pos': _fmt_vec(center), 'quat': '1 0 0 0'})
        ET.SubElement(
            link,
            'body',
            {
                'name': f'body_{marker_name}',
                'type': 'cuboid',
                'pos': '0 0 0',
                'quat': '1 0 0 0',
                'size': ENDEFFECTOR_SIZE_DEFAULT,
                'mass': '0.0001',
                'inertia': '0.0001 0.0001 0.0001',
                'rgba': '0 0 0 0',
                'collision': 'false',
                'ground_contact': 'false',
            },
        )
        registry.joints.add(joint_name)
        body_name = f'body_{marker_name}'
        registry.bodies.add(body_name)
        registry.marker_bodies.add(body_name)
        registry.marker_joints.append(joint_name)


def _append_contact_if_missing(contact: ET.Element, existing: Set[Tuple[str, Tuple[Tuple[str, str], ...]]], elem: ET.Element) -> None:
    key = _contact_key(elem)
    if key in existing:
        return
    contact.append(elem)
    existing.add(key)


def _add_generated_contacts(
    root: ET.Element,
    registry: GeneratedRegistry,
    contact_model: Optional[Dict[str, Any]] = None,
) -> None:
    contact = _ensure_section(root, 'contact')
    ground_bodies, general_bodies, primitive_bodies = _contact_candidate_bodies(root, exclude=registry.marker_bodies)
    tool_contact_excluded = _tool_contact_excluded_bodies(root)
    collision_bodies = ground_bodies | general_bodies | primitive_bodies
    component_by_body, fixed_world_bodies, no_ground_bodies = _body_contact_topology(root)
    _normalize_contact_declarations(
        contact,
        component_by_body=component_by_body,
        fixed_world_bodies=fixed_world_bodies,
        no_ground_bodies=no_ground_bodies,
        collision_bodies=collision_bodies,
    )
    contact_reference_keys = {
        'body', 'body1', 'body2', 'general_body', 'primitive_body'
    }
    for elem in list(contact):
        referenced = {
            value
            for key, value in elem.attrib.items()
            if key in contact_reference_keys
        }
        if (
            referenced.intersection(registry.tool_bodies)
            and referenced.intersection(tool_contact_excluded)
        ):
            contact.remove(elem)
    ground_bodies -= fixed_world_bodies
    ground_bodies -= no_ground_bodies
    body_types = _body_type_by_name(root)

    # Generated contacts use the selected task's common contact-model contract.
    # RedMax is sensitive to these values, so conversion must not invent
    # task-specific stiffness outside that contract.
    model = dict(contact_model or {})

    tool_ground_contact_raw = model.get("tool_ground_contact_enabled", True)
    if isinstance(tool_ground_contact_raw, bool):
        tool_ground_contact_enabled = tool_ground_contact_raw
    elif isinstance(tool_ground_contact_raw, (int, float)):
        if tool_ground_contact_raw not in (0, 1):
            raise ValueError(
                "tool_ground_contact_enabled must be a boolean"
            )
        tool_ground_contact_enabled = bool(tool_ground_contact_raw)
    else:
        normalized = str(tool_ground_contact_raw).strip().lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(
                "tool_ground_contact_enabled must be a boolean"
            )
        tool_ground_contact_enabled = normalized in {"true", "1", "yes"}
    if not tool_ground_contact_enabled:
        ground_bodies.difference_update(registry.tool_bodies)
        for elem in list(contact.findall("ground_contact")):
            if elem.attrib.get("body") in registry.tool_bodies:
                contact.remove(elem)

    tool_scene_contact_raw = model.get("tool_scene_contact_enabled", True)
    if isinstance(tool_scene_contact_raw, bool):
        tool_scene_contact_enabled = tool_scene_contact_raw
    elif isinstance(tool_scene_contact_raw, (int, float)):
        if tool_scene_contact_raw not in (0, 1):
            raise ValueError(
                "tool_scene_contact_enabled must be a boolean"
            )
        tool_scene_contact_enabled = bool(tool_scene_contact_raw)
    else:
        normalized = str(tool_scene_contact_raw).strip().lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(
                "tool_scene_contact_enabled must be a boolean"
            )
        tool_scene_contact_enabled = normalized in {"true", "1", "yes"}

    def model_value(name: str, default: str) -> str:
        return default if name not in model else str(model[name])

    smooth = {
        'smoothing': model_value('smoothing_distance', '0.02'),
        'velocity_smoothing': model_value('slip_velocity', '0.05'),
    }
    abstract_ground_contact = {'kn': '1e4', 'kt': '1e3', 'mu': '1.2', 'damping': '1e2'}
    primitive_ground_contact = {'kn': '1e6', 'kt': '1e5', 'mu': '1.2', 'damping': '1e3'}
    # Deformed generated tools can expose broad or sharp contact patches. Keep
    # tool-ground contact compliant and low-friction so the optimizer does not
    # get stuck on tangential impulses when a candidate lightly penetrates.
    tool_ground_contact = {'kn': '5e4', 'kt': '1e3', 'mu': '0.35', 'damping': '5e2'}
    body_ball_contact = {
        'kn': model_value('normal_stiffness', '1e5'),
        'kt': model_value('tangential_stiffness', '1e4'),
        'mu': model_value('friction', '2.0'),
        'damping': model_value('damping', '5e2'),
        **smooth,
    }
    tool_scene_contact = {
        'kn': model_value('tool_scene_normal_stiffness', '2e5'),
        'kt': model_value('tool_scene_tangential_stiffness', '1e4'),
        'mu': model_value('tool_scene_friction', '0.8'),
        'damping': model_value('tool_scene_damping', '3e3'),
        **smooth,
    }
    primitive_ball_bodies = {
        body
        for body in primitive_bodies
        if body_types.get(body) == 'sphere' or 'ball' in body.lower()
    }
    primitive_scene_bodies = primitive_bodies - primitive_ball_bodies
    if not tool_scene_contact_enabled:
        for elem in list(contact.findall("general_primitive_contact")):
            if (
                elem.attrib.get("general_body") in registry.tool_bodies
                and elem.attrib.get("primitive_body")
                in primitive_scene_bodies
            ):
                contact.remove(elem)

    existing = {_contact_key(elem) for elem in list(contact)}

    for body_name in sorted(ground_bodies):
        is_primitive_body = body_name in primitive_bodies
        is_tool_body = body_name in registry.tool_bodies
        contact_params = (
            primitive_ground_contact
            if is_primitive_body
            else (tool_ground_contact if is_tool_body else abstract_ground_contact)
        )
        elem = ET.Element(
            'ground_contact',
            {'body': body_name, **contact_params},
        )
        _append_contact_if_missing(contact, existing, elem)

    for general_body in sorted(general_bodies):
        for primitive_body in primitive_ball_bodies:
            if _same_rigid_component(general_body, primitive_body, component_by_body):
                continue
            elem = ET.Element(
                'general_primitive_contact',
                {'general_body': general_body, 'primitive_body': primitive_body, **body_ball_contact},
            )
            _append_contact_if_missing(contact, existing, elem)
    for scene_body in sorted(primitive_scene_bodies):
        for ball_body in sorted(primitive_ball_bodies):
            if _same_rigid_component(scene_body, ball_body, component_by_body):
                continue
            elem = ET.Element(
                'general_primitive_contact',
                {'general_body': scene_body, 'primitive_body': ball_body, **body_ball_contact},
            )
            _append_contact_if_missing(contact, existing, elem)

    sorted_primitive_bodies = sorted(primitive_ball_bodies)
    for idx, body1 in enumerate(sorted_primitive_bodies):
        for body2 in sorted_primitive_bodies[idx + 1:]:
            elem = ET.Element(
                'sphere_sphere_contact',
                {
                    'body1': body1,
                    'body2': body2,
                    'kn': body_ball_contact['kn'],
                    'damping': body_ball_contact['damping'],
                    'smoothing': body_ball_contact['smoothing'],
                    'velocity_smoothing': body_ball_contact['velocity_smoothing'],
                },
            )
            _append_contact_if_missing(contact, existing, elem)

    scene_general_bodies = (
        general_bodies
        - registry.tool_bodies
        - tool_contact_excluded
    )
    abstract_pairs: Set[Tuple[str, str]] = set()

    sorted_scene_bodies = sorted(scene_general_bodies)
    sorted_tool_bodies = sorted(registry.tool_bodies & general_bodies)
    for idx, body1 in enumerate(sorted_tool_bodies):
        for body2 in sorted_tool_bodies[idx + 1:]:
            if tuple(sorted((body1, body2))) in registry.tool_connected_body_pairs:
                continue
            abstract_pairs.add((body1, body2))
    for tool_body in sorted_tool_bodies:
        for scene_body in sorted_scene_bodies:
            if _same_rigid_component(tool_body, scene_body, component_by_body):
                continue
            abstract_pairs.add(tuple(sorted((tool_body, scene_body))))
        for scene_body in sorted(
            primitive_scene_bodies - tool_contact_excluded
        ):
            if _same_rigid_component(tool_body, scene_body, component_by_body):
                continue
            if not tool_scene_contact_enabled:
                continue
            elem = ET.Element(
                'general_primitive_contact',
                {'general_body': tool_body, 'primitive_body': scene_body, **tool_scene_contact},
            )
            _append_contact_if_missing(contact, existing, elem)

    for body1, body2 in sorted(abstract_pairs):
        if _same_rigid_component(body1, body2, component_by_body):
            continue
        is_tool_pair = body1 in registry.tool_bodies and body2 in registry.tool_bodies
        is_tool_scene_pair = (
            (body1 in registry.tool_bodies and body2 in scene_general_bodies)
            or (body2 in registry.tool_bodies and body1 in scene_general_bodies)
        )
        elem = ET.Element(
            'collision_constraint',
            {
                'body1': body1,
                'body2': body2,
                'enforcement': 'preflight_design',
                'kind': (
                    'tool_tool'
                    if is_tool_pair
                    else ('tool_scene' if is_tool_scene_pair else 'abstract')
                ),
            },
        )
        _append_contact_if_missing(contact, existing, elem)


def _add_generated_variables(
    root: ET.Element,
    registry: GeneratedRegistry,
    *,
    marker_radius: str = '0.2',
) -> None:
    variable = _ensure_section(root, 'variable')
    existing = {
        (elem.attrib.get('joint'), elem.attrib.get('pos', '0 0 0'))
        for elem in variable.findall('endeffector')
    }
    for joint_name in registry.marker_joints:
        key = (joint_name, '0 0 0')
        if key in existing:
            continue
        ET.SubElement(
            variable,
            'endeffector',
            {
                'joint': joint_name,
                'pos': '0 0 0',
                'radius': str(marker_radius),
            },
        )
        existing.add(key)


def _validate_generated_references(root: ET.Element) -> None:
    bodies = _collect_body_names(root)
    joints = _collect_joint_names(root)
    if len(bodies) != len(list(body.attrib.get('name') for body in root.iter('body') if body.attrib.get('name'))):
        raise ValueError('Duplicate body names found after XML merge')
    if len(joints) != len(list(joint.attrib.get('name') for joint in root.iter('joint') if joint.attrib.get('name'))):
        raise ValueError('Duplicate joint names found after XML merge')

    for motor in root.findall('./actuator/motor'):
        joint = motor.attrib.get('joint')
        if joint and joint not in joints:
            raise ValueError(f'Actuator references missing joint: {joint}')
    for ee in root.findall('./variable/endeffector'):
        joint = ee.attrib.get('joint')
        if joint and joint not in joints:
            raise ValueError(f'Variable references missing joint: {joint}')
    for elem in root.findall('./contact/*'):
        for attr in ('body', 'general_body', 'primitive_body', 'body1', 'body2'):
            value = elem.attrib.get(attr)
            if value and value not in bodies:
                raise ValueError(f'Contact references missing body: {value}')


def _payload_to_simple_tree_auto(
    payload: Dict[str, Any],
    assets_json: Optional[str],
    *,
    root_asset_id: Optional[str] = None,
) -> Dict[str, Any]:
    if 'tree' in payload and isinstance(payload['tree'], dict):
        payload = payload['tree']
    if 'nodes' in payload:
        return _tree_payload_to_simple_tree(
            payload,
            assets_json=assets_json,
            root_asset_id=root_asset_id,
        )
    return payload


def convert_tree_to_scene_etree(
    tree: Dict[str, Any],
    *,
    scene_xml: str,
    assets_json: Optional[str] = None,
    root_asset_id: str,
    model_name: Optional[str] = None,
    tool_design_params: int = TOOL_DESIGN_PARAMS_DEFAULT,
    root_design_params: int = 0,
    root_asset_pos: str = '0 0 0',
    root_asset_quat: str = '1 0 0 0',
    functions: Optional[List[Dict[str, Any]]] = None,
    root_joint_type: str = 'fixed',
    root_joint_pos: str = '0 0 0',
    root_joint_quat: str = '1 0 0 0',
    root_joint_damping: Optional[str] = None,
    root_aux_joint_type: Optional[str] = None,
    root_aux_joint_name: str = 'freeform_aux_joint',
    root_aux_joint_pos: str = '0 0 0',
    root_aux_joint_quat: str = '1 0 0 0',
    root_aux_joint_axis: str = '0 1 0',
    root_aux_joint_axis1: str = '0 0 1',
    root_aux_joint_damping: Optional[str] = None,
    root_aux_joint_lim: Optional[str] = None,
    root_aux_joint_lim_stiffness: Optional[str] = None,
    generated_robot_placement: str = 'after_scene',
    root_body_size: str = '0.05 0.05 0.05',
    root_motor_ctrl: str = 'force',
    root_motor_ctrl_range: str = '-6e5 6e5',
    root_motor_P: str = '2e4',
    root_motor_D: str = '2e3',
    root_aux_motor_ctrl: str = 'position',
    root_aux_motor_ctrl_range: str = '-5e4 5e4',
    root_aux_motor_P: str = '2e4',
    root_aux_motor_D: str = '2e3',
    generated_endeffector_radius: str = '0.2',
    contact_model: Optional[Dict[str, Any]] = None,
) -> ET.Element:
    """Merge a searched Head tree under one fixed Handle root."""
    if not str(root_asset_id).strip():
        raise ValueError("root_asset_id must identify the fixed Handle asset")

    tree_payload = tree.get('tree', tree) if isinstance(tree, dict) else tree
    root_rotation_mode = (
        tree_payload.get('root_rotation_mode')
        if isinstance(tree_payload, dict)
        else None
    )
    if root_rotation_mode is not None:
        root_rotation_mode = str(root_rotation_mode).strip().lower()
        if root_rotation_mode not in ROOT_ROTATION_AXES:
            raise ValueError(
                "Unsupported searched root rotation mode: "
                f"{root_rotation_mode!r}"
            )
        root_aux_joint_type = 'revolute'
        root_aux_joint_name = f'freeform_{root_rotation_mode}_joint'
        root_aux_joint_axis = ROOT_ROTATION_AXES[root_rotation_mode]

    # The generated XML follows the canonical task/config naming, scene-merge,
    # contact, actuator, and end-effector contracts.
    simple_tree = _payload_to_simple_tree_auto(
        tree,
        assets_json,
        root_asset_id=str(root_asset_id),
    )
    raw_assets = _load_raw_assets_map(assets_json)
    repo_root = _repo_root()
    root = ET.parse(scene_xml).getroot()
    if model_name:
        root.set('model', model_name)

    registry = GeneratedRegistry()
    roots_to_attach = simple_tree.get('children', []) if simple_tree.get('name') == 'synthetic_root' else [simple_tree]
    root_joint = root_joint_type
    if root_joint == 'fixed':
        root_joint = 'free3d-euler'
    (
        handle_robot,
        attachment_root,
        root_actuator_joint,
        auxiliary_actuator_joint,
    ) = _make_handle_root_robot(
        joint_type=root_joint,
        joint_pos=root_joint_pos,
        joint_quat=root_joint_quat,
        joint_damping=root_joint_damping,
        root_body_size=root_body_size,
        auxiliary_joint_type=root_aux_joint_type,
        auxiliary_joint_name=root_aux_joint_name,
        auxiliary_joint_pos=root_aux_joint_pos,
        auxiliary_joint_quat=root_aux_joint_quat,
        auxiliary_joint_axis=root_aux_joint_axis,
        auxiliary_joint_axis1=root_aux_joint_axis1,
        auxiliary_joint_damping=root_aux_joint_damping,
        auxiliary_joint_lim=root_aux_joint_lim,
        auxiliary_joint_lim_stiffness=root_aux_joint_lim_stiffness,
        registry=registry,
    )
    if root_rotation_mode is not None:
        auxiliary_carrier = handle_robot.find(
            ".//link[@name='root_auxiliary_carrier']"
        )
        if auxiliary_carrier is None:
            raise RuntimeError(
                "searched root rotation did not create an auxiliary carrier"
            )
        auxiliary_carrier.set('root_rotation_mode', root_rotation_mode)

    generated_root_links: List[ET.Element] = []
    for attach_index, tip in enumerate([attachment_root]):
        link_by_node_id: Dict[int, ET.Element] = {}
        root_link = _make_fixed_root_link(
            attach_index=attach_index,
            root_asset_id=str(root_asset_id),
            raw_assets=raw_assets,
            repo_root=repo_root,
            design_params=int(root_design_params),
            pos=str(root_asset_pos),
            quat=str(root_asset_quat),
            registry=registry,
            parent_body_name=(
                tip.find('body').attrib.get('name')
                if tip.find('body') is not None
                else None
            ),
        )
        tip.append(root_link)
        generated_root_links.append(root_link)
        attachment_parent = root_link
        for root_idx, tool_root in enumerate(roots_to_attach):
            parent_body = attachment_parent.find('body')
            parent_body_name = parent_body.attrib.get('name') if parent_body is not None else None
            attachment_parent.append(
                _make_tool_link(
                    tool_root,
                    attach_index=attach_index,
                    design_params=tool_design_params,
                    registry=registry,
                    link_by_node_id=link_by_node_id,
                    parent_body_name=parent_body_name,
                    color_idx=root_idx,
                )
            )
        _add_endeffector_markers(
            attachment_parent,
            simple_tree,
            attach_index=attach_index,
            registry=registry,
            functions=functions,
            link_by_node_id=link_by_node_id,
        )

    placement = str(generated_robot_placement).strip().lower()
    if placement not in {'after_scene', 'before_scene'}:
        raise ValueError(
            "generated_robot_placement must be 'after_scene' or "
            f"'before_scene', got {generated_robot_placement!r}"
        )
    robot_indices = [
        index
        for index, child in enumerate(list(root))
        if child.tag == 'robot'
    ]
    if placement == 'before_scene':
        root.insert(robot_indices[0] if robot_indices else 0, handle_robot)
    elif robot_indices:
        root.insert(robot_indices[-1] + 1, handle_robot)
    else:
        root.insert(0, handle_robot)

    actuator = _ensure_section(root, 'actuator')
    ET.SubElement(
        actuator,
        'motor',
        {
            'joint': root_actuator_joint or HANDLE_ROOT_JOINT_NAME,
            'ctrl': root_motor_ctrl,
            'ctrl_range': root_motor_ctrl_range,
            **(
                {'P': root_motor_P, 'D': root_motor_D}
                if root_motor_ctrl == 'position'
                else {}
            ),
        },
    )
    if auxiliary_actuator_joint is not None:
        ET.SubElement(
            actuator,
            'motor',
            {
                'joint': auxiliary_actuator_joint,
                'ctrl': root_aux_motor_ctrl,
                'ctrl_range': root_aux_motor_ctrl_range,
                **(
                    {
                        'P': root_aux_motor_P,
                        'D': root_aux_motor_D,
                    }
                    if root_aux_motor_ctrl == 'position'
                    else {}
                ),
            },
        )

    _add_generated_contacts(root, registry, contact_model=contact_model)
    _add_generated_variables(
        root,
        registry,
        marker_radius=generated_endeffector_radius,
    )
    _finalize_fixed_asset_root(root, generated_root_links)
    _validate_generated_references(root)
    return root


def convert_tree_to_scene_xml(
    tree: Dict[str, Any],
    out_path: str,
    *,
    scene_xml: str,
    assets_json: Optional[str] = None,
    root_asset_id: str,
    model_name: Optional[str] = None,
    tool_design_params: int = TOOL_DESIGN_PARAMS_DEFAULT,
    root_design_params: int = 0,
    root_asset_pos: str = '0 0 0',
    root_asset_quat: str = '1 0 0 0',
    functions: Optional[List[Dict[str, Any]]] = None,
    root_joint_type: str = 'fixed',
    root_joint_pos: str = '0 0 0',
    root_joint_quat: str = '1 0 0 0',
    root_joint_damping: Optional[str] = None,
    root_aux_joint_type: Optional[str] = None,
    root_aux_joint_name: str = 'freeform_aux_joint',
    root_aux_joint_pos: str = '0 0 0',
    root_aux_joint_quat: str = '1 0 0 0',
    root_aux_joint_axis: str = '0 1 0',
    root_aux_joint_axis1: str = '0 0 1',
    root_aux_joint_damping: Optional[str] = None,
    root_aux_joint_lim: Optional[str] = None,
    root_aux_joint_lim_stiffness: Optional[str] = None,
    generated_robot_placement: str = 'after_scene',
    root_body_size: str = '0.05 0.05 0.05',
    root_motor_ctrl: str = 'force',
    root_motor_ctrl_range: str = '-6e5 6e5',
    root_motor_P: str = '2e4',
    root_motor_D: str = '2e3',
    root_aux_motor_ctrl: str = 'position',
    root_aux_motor_ctrl_range: str = '-5e4 5e4',
    root_aux_motor_P: str = '2e4',
    root_aux_motor_D: str = '2e3',
    generated_endeffector_radius: str = '0.2',
    contact_model: Optional[Dict[str, Any]] = None,
) -> str:
    et = convert_tree_to_scene_etree(
        tree,
        scene_xml=scene_xml,
        assets_json=assets_json,
        root_asset_id=root_asset_id,
        model_name=model_name,
        tool_design_params=tool_design_params,
        root_design_params=root_design_params,
        root_asset_pos=root_asset_pos,
        root_asset_quat=root_asset_quat,
        functions=functions,
        root_joint_type=root_joint_type,
        root_joint_pos=root_joint_pos,
        root_joint_quat=root_joint_quat,
        root_joint_damping=root_joint_damping,
        root_aux_joint_type=root_aux_joint_type,
        root_aux_joint_name=root_aux_joint_name,
        root_aux_joint_pos=root_aux_joint_pos,
        root_aux_joint_quat=root_aux_joint_quat,
        root_aux_joint_axis=root_aux_joint_axis,
        root_aux_joint_axis1=root_aux_joint_axis1,
        root_aux_joint_damping=root_aux_joint_damping,
        root_aux_joint_lim=root_aux_joint_lim,
        root_aux_joint_lim_stiffness=root_aux_joint_lim_stiffness,
        generated_robot_placement=generated_robot_placement,
        root_body_size=root_body_size,
        root_motor_ctrl=root_motor_ctrl,
        root_motor_ctrl_range=root_motor_ctrl_range,
        root_motor_P=root_motor_P,
        root_motor_D=root_motor_D,
        root_aux_motor_ctrl=root_aux_motor_ctrl,
        root_aux_motor_ctrl_range=root_aux_motor_ctrl_range,
        root_aux_motor_P=root_aux_motor_P,
        root_aux_motor_D=root_aux_motor_D,
        generated_endeffector_radius=generated_endeffector_radius,
        contact_model=contact_model,
    )
    _rewrite_resource_paths_for_output(
        et,
        out_path=out_path,
        scene_xml=scene_xml,
    )
    _validate_generated_references(et)
    xml_str = _pretty_xml(et)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    return out_path



def compile_scene(sequence: Sequence[Action], out_path: str, *, scene_xml: str,
                  root_asset_id: str, assets_json: Optional[str] = None,
                  **scene_options) -> str:
    """Validate grammar, assemble its ordered tree, and write a complete scene.

    Root rotation, function groups, docking, physics and resource rebasing use
    the same compiler as saved tree payloads. Invalid grammar fails before XML
    is written. ``scene_options`` are the keyword options of
    :func:`convert_tree_to_scene_xml`.
    """
    tree = sequence_to_tree(sequence)
    return convert_tree_to_scene_xml(
        tree_to_dict(tree), out_path, scene_xml=scene_xml,
        root_asset_id=root_asset_id, assets_json=assets_json, **scene_options,
    )
