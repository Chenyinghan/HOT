"""Action grammar definitions for constructive skeleton search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


FACE_INDEX_MIN = 0
FACE_INDEX_MAX = 5
FACE_LABELS = ["+z", "+x", "+y", "-x", "-z", "-y"]
FACING_INDEX_MIN = 0
FACING_INDEX_MAX = 3
ROOT_ROTATION_MODES = ("roll", "pitch", "yaw")


def validate_face_index(value: int, name: str) -> None:
    """Validate a face index.

    Args:
        value: Integer face index.
        name: Field name for error messages.

    Raises:
        ValueError: If index is outside [0, 5].
    """
    if not (FACE_INDEX_MIN <= value <= FACE_INDEX_MAX):
        raise ValueError(f"{name} must be in [0, 5], got {value}.")


def validate_non_negative_index(value: int, name: str) -> None:
    """Validate that an index-like integer is non-negative.

    Args:
        value: Integer index.
        name: Field name for error messages.

    Raises:
        ValueError: If index is negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def validate_facing_index(value: int, name: str) -> None:
    """Validate a facing index.

    Args:
        value: Integer facing index.
        name: Field name for error messages.

    Raises:
        ValueError: If index is outside [0, 3].
    """
    if not (FACING_INDEX_MIN <= value <= FACING_INDEX_MAX):
        raise ValueError(f"{name} must be in [0, 3], got {value}.")


@dataclass(frozen=True)
class Action:
    """Base action type.

    Subclasses implement concrete grammar tokens.
    """

    kind: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize action to a JSON-compatible dictionary."""
        return {"kind": self.kind}


@dataclass(frozen=True)
class SelectRootRotation(Action):
    """Select the Handle-centred one-DOF rotation before constructing links."""

    mode: str

    def __init__(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in ROOT_ROTATION_MODES:
            raise ValueError(
                "root rotation mode must be one of "
                f"{ROOT_ROTATION_MODES}, got {mode!r}"
            )
        object.__setattr__(self, "kind", "SelectRootRotation")
        object.__setattr__(self, "mode", normalized)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "mode": self.mode}


@dataclass(frozen=True)
class AddLink(Action):
    """Add a new link at current cursor.

    Args:
        asset_id: Link primitive identifier.
        p: Face index on current link where child is attached.
        d: Dock index on parent face.
        f: Facing index in connection plane.
        q: Face index on new link that attaches to parent.
        child_dock_id: Dock index on child face q.
        start_function_group: Whether the new link roots one functional group.
    """

    asset_id: str
    p: int
    d: int
    f: int
    q: int
    child_dock_id: int
    start_function_group: bool
    prepared_attachment: Optional["PreparedAttachment"] = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    def __init__(
        self,
        asset_id: str,
        p: int,
        d: int,
        f: int,
        q: int,
        child_dock_id: int = 0,
        start_function_group: bool = False,
        prepared_attachment: Optional["PreparedAttachment"] = None,
    ) -> None:
        object.__setattr__(self, "kind", "AddLink")
        object.__setattr__(self, "asset_id", asset_id)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "f", f)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "child_dock_id", child_dock_id)
        object.__setattr__(self, "start_function_group", bool(start_function_group))
        object.__setattr__(self, "prepared_attachment", prepared_attachment)
        if not asset_id:
            raise ValueError("asset_id must be a non-empty string.")
        validate_face_index(p, "p")
        validate_non_negative_index(d, "d")
        validate_facing_index(f, "f")
        validate_face_index(q, "q")
        validate_non_negative_index(child_dock_id, "child_dock_id")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize AddLink action."""
        return {
            "kind": self.kind,
            "asset_id": self.asset_id,
            "p": self.p,
            "d": self.d,
            "f": self.f,
            "q": self.q,
            "child_dock_id": self.child_dock_id,
            "start_function_group": self.start_function_group,
        }


@dataclass(frozen=True)
class PreparedAttachment:
    """Validated geometry cached on an in-memory ``AddLink`` candidate.

    This payload is deliberately excluded from action equality, hashing, and
    serialization. ``state_token`` prevents reuse after the action is detached
    from the exact partial state for which collision validity was checked.
    """

    state_token: Tuple[Any, ...]
    child_center: Tuple[float, float, float]
    child_rotation: Tuple[float, ...]
    child_box: Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]
    physical_successor_key: Tuple[Any, ...]


@dataclass(frozen=True)
class End(Action):
    """End current link expansion and backtrack to parent."""

    def __init__(self) -> None:
        object.__setattr__(self, "kind", "End")


@dataclass(frozen=True)
class Grow(Action):
    """Virtual BASS decision that exposes geometric AddLink realizations.

    Grow is not part of the constructive grammar and must never be serialized
    into a completed skeleton. It exists only as a structural tree layer.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "kind", "Grow")


def action_from_dict(payload: Dict[str, Any]) -> Action:
    """Deserialize an action dictionary.

    Args:
        payload: Mapping with action fields.

    Returns:
        Parsed Action object.

    Raises:
        ValueError: If payload cannot be parsed into a known action.
    """
    kind = payload.get("kind")
    if kind == "SelectRootRotation":
        return SelectRootRotation(str(payload.get("mode", "")))
    if kind == "AddLink":
        start_function_group = payload.get(
            "start_function_group",
            payload.get("StartFunctionGroup", False),
        )
        if isinstance(start_function_group, str):
            start_function_group = start_function_group.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
        return AddLink(
            asset_id=str(payload.get("asset_id", "")),
            p=int(payload.get("p")),
            d=int(payload.get("d", 0)),
            f=int(payload.get("f", 0)),
            q=int(payload.get("q")),
            child_dock_id=int(payload.get("child_dock_id", payload.get("c", 0))),
            start_function_group=bool(start_function_group),
        )
    if kind == "End":
        return End()
    if kind == "Grow":
        raise ValueError("Grow is an internal BASS decision, not a grammar action")
    raise ValueError(f"Unsupported action kind: {kind}")
