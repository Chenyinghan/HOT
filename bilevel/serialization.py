"""Shared JSON normalization for search and subprocess result payloads."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    """Convert framework values to the established JSON-compatible shape."""

    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


__all__ = ["jsonable"]
