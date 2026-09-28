"""Asset metadata loading utilities for BASS search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


FACE_COUNT = 6
DEFAULT_FACING_OPTIONS = [0, 1, 2, 3]
DEFAULT_BARYCENTRIC4 = [0.25, 0.25, 0.25, 0.25]


def _default_in_docks_per_face() -> Dict[int, List[dict[str, Any]]]:
    """Return legacy-compatible child-side dock configuration for each face."""
    return {
        face: [
            {
                "id": 0,
                "barycentric": list(DEFAULT_BARYCENTRIC4),
                "facing_options": list(DEFAULT_FACING_OPTIONS),
            }
        ]
        for face in range(FACE_COUNT)
    }


def _default_out_docks_per_face() -> Dict[int, List[dict[str, Any]]]:
    """Return default parent-side dock configuration for each face."""
    return {
        face: [
            {
                "id": 0,
                "barycentric": list(DEFAULT_BARYCENTRIC4),
            }
        ]
        for face in range(FACE_COUNT)
    }


@dataclass(frozen=True)
class AssetSpec:
    """Canonical asset metadata used during action generation.

    Attributes:
        asset_id: Unique asset identifier.
        category: Optional grouping label.
        tags: Optional free-form labels.
        enabled: If False, asset is ignored by the search.
        out_docks_per_face: Map from parent face index to available parent-side
            dock configs. A dock config contains:
            - id: non-negative integer dock id
            - barycentric: optional four barycentric weights for face vertices
        in_docks_per_face: Map from child face index to available child-side
            dock configs. A dock config contains:
            - id: non-negative integer dock id
            - barycentric: optional four barycentric weights for face vertices
            - facing_options: optional list of facing indices in [0, 3]
        cage_path: Optional path to cage-handle txt file.
        half_extents: Approximate half extents of initialization cuboid.
    """

    asset_id: str
    canonical_id: Optional[str] = None
    aliases: Optional[List[str]] = None
    role: Optional[str] = None
    searchable: bool = True
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    enabled: bool = True
    out_docks_per_face: Optional[Dict[int, List[dict[str, Any]]]] = None
    in_docks_per_face: Optional[Dict[int, List[dict[str, Any]]]] = None
    cage_path: Optional[str] = None
    mesh_path: Optional[str] = None
    contacts_path: Optional[str] = None
    contact_ids_path: Optional[str] = None
    weights_path: Optional[str] = None
    half_extents: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    def resolved_out_docks_per_face(self) -> Dict[int, List[dict[str, Any]]]:
        """Return parent-side dock map with defaults applied for missing metadata."""
        if self.out_docks_per_face is None:
            return _default_out_docks_per_face()
        return self.out_docks_per_face

    def resolved_in_docks_per_face(self) -> Dict[int, List[dict[str, Any]]]:
        """Return child-side dock map with defaults applied for missing metadata."""
        if self.in_docks_per_face is None:
            return _default_in_docks_per_face()
        return self.in_docks_per_face

    def docks_for_face(self, face: int) -> List[dict[str, Any]]:
        """Return available parent-side docks for one face with fallback defaults."""
        return self.out_docks_for_face(face)

    def out_docks_for_face(self, face: int) -> List[dict[str, Any]]:
        """Return available parent-side docks for one face."""
        return self.resolved_out_docks_per_face().get(face, [])

    def in_docks_for_face(self, face: int) -> List[dict[str, Any]]:
        """Return available child-side docks for one face."""
        return self.resolved_in_docks_per_face().get(face, [])


def _deduplicate_assets(assets: Iterable[AssetSpec]) -> List[AssetSpec]:
    """Deduplicate assets while preserving insertion order.

    Args:
        assets: Iterable of normalized assets.

    Returns:
        Deduplicated list of assets.

    Raises:
        ValueError: If duplicate asset ids are found.
    """
    seen: Set[str] = set()
    result: List[AssetSpec] = []
    for asset in assets:
        if asset.asset_id in seen:
            raise ValueError(f"Duplicate asset id found: {asset.asset_id}")
        seen.add(asset.asset_id)
        result.append(asset)
    return result


def _from_txt(path: Path) -> List[AssetSpec]:
    """Load assets from a TXT file containing one id per line."""
    assets: List[AssetSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        assets.append(AssetSpec(asset_id=token))
    return _deduplicate_assets(assets)


def _normalize_record(record: Dict[str, Any]) -> AssetSpec:
    """Normalize one JSON/YAML asset record.

    Args:
        record: Raw mapping for one asset entry.

    Returns:
        Normalized AssetSpec.
    """
    asset_id = str(record.get("id", "")).strip()
    if not asset_id:
        raise ValueError("Each asset record must include non-empty 'id'.")
    canonical_id = str(record.get("canonical_id", asset_id)).strip()
    aliases_raw = record.get("aliases", [])
    aliases = [str(item) for item in aliases_raw]
    role_raw = record.get("role")
    role = str(role_raw) if role_raw is not None else None
    searchable = bool(record.get("searchable", record.get("enabled", True)))
    category_raw = record.get("category")
    category = str(category_raw) if category_raw is not None else None
    tags_raw = record.get("tags")
    tags: Optional[List[str]]
    if tags_raw is None:
        tags = None
    else:
        tags = [str(item) for item in tags_raw]
    enabled = bool(record.get("enabled", True))
    legacy_docks_raw = record.get("docks")
    out_docks_raw = record.get("out_docks")
    in_docks_raw = record.get("in_docks")
    out_docks_per_face = _normalize_docks(
        out_docks_raw if out_docks_raw is not None else legacy_docks_raw,
        allow_facing=False,
        field_name="out_docks" if out_docks_raw is not None else "docks",
    )
    in_docks_per_face = _normalize_docks(
        in_docks_raw if in_docks_raw is not None else legacy_docks_raw,
        allow_facing=True,
        field_name="in_docks" if in_docks_raw is not None else "docks",
    )
    resources = record.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError(f"asset {asset_id!r} resources must be an object")

    def resource_path(name: str, legacy_name: Optional[str] = None) -> Optional[str]:
        raw = resources.get(name)
        if isinstance(raw, dict):
            value = raw.get("path")
            return None if value is None else str(value)
        value = record.get(legacy_name or name)
        return None if value is None else str(value)

    cage_path = resource_path("cage") or _resolve_cage_path(
        record,
        asset_id,
        category,
    )
    mesh_path = resource_path("mesh")
    contacts_path = resource_path("contacts")
    contact_ids_path = resource_path("contact_ids")
    weights_path = resource_path("weights")
    half_extents = _infer_half_extents_from_cage(cage_path)
    return AssetSpec(
        asset_id=asset_id,
        canonical_id=canonical_id,
        aliases=aliases,
        role=role,
        searchable=searchable,
        category=category,
        tags=tags,
        enabled=enabled,
        out_docks_per_face=out_docks_per_face,
        in_docks_per_face=in_docks_per_face,
        cage_path=cage_path,
        mesh_path=mesh_path,
        contacts_path=contacts_path,
        contact_ids_path=contact_ids_path,
        weights_path=weights_path,
        half_extents=half_extents,
    )


def _normalize_docks(
    value: Any,
    *,
    allow_facing: bool,
    field_name: str,
) -> Optional[Dict[int, List[dict[str, Any]]]]:
    """Normalize per-face docks metadata.

    Expected format:
    {
            "0": [{"id": 0, "barycentric": [0.25,0.25,0.25,0.25], "facing_options": [0,1,2,3]}],
      ...
    }
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be a mapping from face index to dock list")

    result: Dict[int, List[dict[str, Any]]] = {}
    for raw_face, raw_docks in value.items():
        face = int(raw_face)
        if not (0 <= face < FACE_COUNT):
            raise ValueError(f"{field_name} face index must be in [0, 5], got {face}")
        if not isinstance(raw_docks, list):
            raise ValueError(f"{field_name}[{face}] must be a list")

        seen_dock_ids: Set[int] = set()
        normalized_face_docks: List[dict[str, Any]] = []
        for item in raw_docks:
            if not isinstance(item, dict):
                raise ValueError(f"{field_name}[{face}] entries must be objects")
            dock_id = int(item.get("id", 0))
            if dock_id < 0:
                raise ValueError(f"{field_name}[{face}].id must be >= 0, got {dock_id}")
            if dock_id in seen_dock_ids:
                raise ValueError(f"duplicate dock id {dock_id} on face {face}")
            seen_dock_ids.add(dock_id)

            barycentric = item.get("barycentric", list(DEFAULT_BARYCENTRIC4))
            if (
                not isinstance(barycentric, list)
                or len(barycentric) != 4
                or not all(isinstance(v, (int, float)) for v in barycentric)
            ):
                raise ValueError(
                    f"{field_name}[{face}].barycentric must be a four-number list, got {barycentric}"
                )
            barycentric_values = [float(v) for v in barycentric]
            bary_sum = sum(barycentric_values)
            if bary_sum <= 0.0:
                raise ValueError(
                    f"{field_name}[{face}].barycentric must have positive sum, got {barycentric_values}"
                )
            barycentric_values = [v / bary_sum for v in barycentric_values]

            normalized = {
                "id": dock_id,
                "barycentric": barycentric_values,
            }
            edge = bool(item.get("edge", False))
            if edge:
                edge_tangent_axis = int(item.get("edge_tangent_axis", -1))
                panel_normal_axis = int(item.get("panel_normal_axis", -1))
                if edge_tangent_axis not in (0, 1, 2):
                    raise ValueError(
                        f"{field_name}[{face}].edge_tangent_axis must lie in [0, 2]"
                    )
                if panel_normal_axis not in (0, 1, 2):
                    raise ValueError(
                        f"{field_name}[{face}].panel_normal_axis must lie in [0, 2]"
                    )
                if edge_tangent_axis == panel_normal_axis:
                    raise ValueError(
                        f"{field_name}[{face}] edge tangent and panel normal axes must differ"
                    )
                edge_profile = str(item.get("edge_profile", "")).strip()
                if not edge_profile:
                    raise ValueError(
                        f"{field_name}[{face}].edge_profile must be non-empty for edge ports"
                    )
                normalized.update(
                    {
                        "edge": True,
                        "edge_tangent_axis": edge_tangent_axis,
                        "panel_normal_axis": panel_normal_axis,
                        "edge_profile": edge_profile,
                    }
                )
            if allow_facing:
                facing_options = item.get("facing_options", list(DEFAULT_FACING_OPTIONS))
                if not isinstance(facing_options, list) or not facing_options:
                    raise ValueError(f"{field_name}[{face}].facing_options must be a non-empty list")
                normalized_facing = [int(v) for v in facing_options]
                if any(v < 0 or v > 3 for v in normalized_facing):
                    raise ValueError(
                        f"{field_name}[{face}].facing_options must contain values in [0, 3], got {normalized_facing}"
                    )
                normalized["facing_options"] = sorted(set(normalized_facing))

            normalized_face_docks.append(normalized)

        result[face] = normalized_face_docks

    return result


def _resolve_cage_path(record: Dict[str, Any], asset_id: str, category: Optional[str]) -> Optional[str]:
    """Resolve cage txt path for an asset, preferring explicit metadata."""
    explicit = record.get("cage")
    if explicit is not None:
        path = str(explicit)
        return path if path else None

    if "/" in asset_id:
        family, primitive = asset_id.split("/", 1)
        inferred = f"assets/{family}/cages/{primitive}.txt"
        if Path(inferred).exists():
            return inferred

    if category:
        inferred = f"assets/{category}/cages/{asset_id}.txt"
        if Path(inferred).exists():
            return inferred

    return None


def _infer_half_extents_from_cage(cage_path: Optional[str]) -> Tuple[float, float, float]:
    """Infer axis-aligned half extents from cage handle txt file.

    Falls back to a conservative unit cuboid when cage data is unavailable.
    """
    if not cage_path:
        return (0.5, 0.5, 0.5)

    file_path = Path(cage_path)
    if not file_path.exists():
        return (0.5, 0.5, 0.5)

    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        return (0.5, 0.5, 0.5)

    points: List[Tuple[float, float, float]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue

    if not points:
        return (0.5, 0.5, 0.5)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (
        max((max(xs) - min(xs)) * 0.5, 1e-6),
        max((max(ys) - min(ys)) * 0.5, 1e-6),
        max((max(zs) - min(zs)) * 0.5, 1e-6),
    )


def _from_json(path: Path) -> List[AssetSpec]:
    """Load assets from JSON file.

    Supports either:
    - list[dict]: [{"id": "..."}, ...]
    - dict with key "assets": {"assets": [{"id": "..."}, ...]}
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and payload.get("catalog_id")
    ):
        from assets.library import load_asset_catalog

        catalog = load_asset_catalog(path)
        records = []
        for asset in catalog.assets:
            records.append(
                {
                    "id": asset.asset_id,
                    "canonical_id": asset.asset_id,
                    "aliases": list(asset.aliases),
                    "role": asset.role,
                    "searchable": asset.searchable,
                    "enabled": True,
                    "tags": list(asset.tags),
                    "resources": {
                        name: {
                            "path": str(resource.path),
                        }
                        for name, resource in asset.resources.items()
                    },
                    "out_docks": dict(asset.out_docks),
                    "in_docks": dict(asset.in_docks),
                }
            )
    elif isinstance(payload, dict):
        records = payload.get("assets", [])
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("JSON must be a list or dict containing 'assets'.")
    assets = [_normalize_record(dict(record)) for record in records]
    return _deduplicate_assets(assets)


def resolve_asset_id(
    assets: Iterable[AssetSpec],
    asset_id_or_alias: str,
) -> str:
    """Resolve one canonical ID without assuming catalog contents."""

    requested = str(asset_id_or_alias)
    for asset in assets:
        names = {
            asset.asset_id,
            asset.canonical_id or asset.asset_id,
            *(asset.aliases or []),
        }
        if requested in names:
            return asset.asset_id
    raise KeyError(f"unknown asset ID or alias {requested!r}")


def select_assets(
    assets: Iterable[AssetSpec],
    selector: Optional[Dict[str, Any]] = None,
    *,
    required_ids: Iterable[str] = (),
) -> List[AssetSpec]:
    """Apply the stable task selector while retaining required root assets."""

    values = list(assets)
    data = dict(selector or {})
    requested_ids = {
        resolve_asset_id(values, value)
        for value in data.get("ids", [])
    }
    required = {
        resolve_asset_id(values, value)
        for value in required_ids
    }
    roles = {str(value) for value in data.get("roles", [])}
    all_tags = {str(value) for value in data.get("all_tags", [])}
    any_tags = {str(value) for value in data.get("any_tags", [])}
    searchable_only = bool(data.get("searchable_only", True))
    selected = []
    for asset in values:
        if asset.asset_id in required:
            selected.append(asset)
            continue
        tags = set(asset.tags or [])
        if searchable_only and not asset.searchable:
            continue
        if requested_ids and asset.asset_id not in requested_ids:
            continue
        if roles and asset.role not in roles:
            continue
        if all_tags and not all_tags.issubset(tags):
            continue
        if any_tags and not any_tags.intersection(tags):
            continue
        selected.append(asset)
    if not selected:
        raise ValueError("asset selector matched no assets")
    return selected


def _from_yaml(path: Path) -> List[AssetSpec]:
    """Load assets from YAML file.

    Raises:
        ImportError: If PyYAML is unavailable.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML assets. Install with: pip install pyyaml"
        ) from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("assets", [])
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("YAML must be a list or dict containing 'assets'.")
    assets = [_normalize_record(dict(record)) for record in records]
    return _deduplicate_assets(assets)


def load_assets(path: str) -> List[AssetSpec]:
    """Load and normalize enabled assets for BASS.

    Args:
        path: Path to .json/.yaml/.yml/.txt assets file.

    Returns:
        List of enabled assets.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If format is unsupported or data invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Asset file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        parsed = _from_json(file_path)
    elif suffix in {".yaml", ".yml"}:
        parsed = _from_yaml(file_path)
    elif suffix == ".txt":
        parsed = _from_txt(file_path)
    else:
        raise ValueError(f"Unsupported asset file extension: {suffix}")

    enabled_assets = [asset for asset in parsed if asset.enabled]
    if not enabled_assets:
        raise ValueError("No enabled assets were found in the provided file.")
    return enabled_assets
