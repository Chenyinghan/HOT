"""Immutable connectivity metadata for connected Head morphology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


Dock = Tuple[int, int]


@dataclass(frozen=True)
class HeadConnection:
    """One resolved deformable Head-to-Head connection."""

    parent_index: int
    child_index: int
    parent_dock: Dock
    child_dock: Dock
    rotation: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.parent_index < 0 or self.child_index < 0:
            raise ValueError("Head connection indices must be nonnegative")
        if len(self.rotation) != 9:
            raise ValueError("Head connection rotation must contain 9 values")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "parent_index": int(self.parent_index),
            "child_index": int(self.child_index),
            "parent_dock": list(self.parent_dock),
            "child_dock": list(self.child_dock),
            "rotation": list(self.rotation),
        }


@dataclass(frozen=True)
class HeadTopologyBlock:
    """Resolved topology metadata for one Head record."""

    tool_index: int
    node_id: int | None
    connected_face_mask: tuple[bool, ...]
    occupied_docks: tuple[Dock, ...]
    parent_tool_index: int | None
    parent_dock: Dock | None
    child_dock: Dock | None
    explicit_connection: bool
    inferred_connection: bool
    direct_handle_mount: bool
    parent_face: int | None
    child_face: int | None

    def __post_init__(self) -> None:
        if self.tool_index < 0:
            raise ValueError("Head topology tool_index must be nonnegative")
        if len(self.connected_face_mask) != 6:
            raise ValueError(
                "Head topology connected_face_mask must contain six faces"
            )
        if self.explicit_connection and self.inferred_connection:
            raise ValueError(
                "a Head connection cannot be both explicit and inferred"
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "tool_index": int(self.tool_index),
            "node_id": self.node_id,
            "connected_face_mask": list(self.connected_face_mask),
            "occupied_docks": [
                list(dock) for dock in self.occupied_docks
            ],
            "parent_tool_index": self.parent_tool_index,
            "parent_dock": (
                None if self.parent_dock is None else list(self.parent_dock)
            ),
            "child_dock": (
                None if self.child_dock is None else list(self.child_dock)
            ),
            "explicit_connection": bool(self.explicit_connection),
            "inferred_connection": bool(self.inferred_connection),
            "direct_handle_mount": bool(self.direct_handle_mount),
            "parent_face": self.parent_face,
            "child_face": self.child_face,
        }


@dataclass(frozen=True)
class HeadTopology:
    """Single topology source consumed by NumPy, Torch, and constraints."""

    blocks: tuple[HeadTopologyBlock, ...]
    connections: tuple[HeadConnection, ...]

    def __post_init__(self) -> None:
        for index, block in enumerate(self.blocks):
            if block.tool_index != index:
                raise ValueError(
                    "Head topology blocks must be ordered by tool_index"
                )
        count = len(self.blocks)
        for connection in self.connections:
            if (
                connection.parent_index >= count
                or connection.child_index >= count
            ):
                raise ValueError(
                    "Head connection index lies outside topology blocks"
                )

    def block(self, tool_index: int) -> HeadTopologyBlock:
        return self.blocks[int(tool_index)]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "blocks": [block.to_manifest() for block in self.blocks],
            "connections": [
                connection.to_manifest()
                for connection in self.connections
            ],
        }
