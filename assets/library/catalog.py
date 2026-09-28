"""Stable catalog and selector interface for all current and future assets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


ASSET_CATALOG_SCHEMA_VERSION = 1
_REQUIRED_RESOURCES = (
    "mesh",
    "cage",
    "contacts",
    "contact_ids",
    "weights",
)


def _nonempty(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _strings(values: Sequence[Any], field_name: str) -> Tuple[str, ...]:
    result = tuple(_nonempty(value, field_name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class ResourceRecord:
    """One file owned by an asset record."""

    path: Path

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        field_name: str,
    ) -> "ResourceRecord":
        path = Path(_nonempty(value.get("path", ""), f"{field_name}.path"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"{field_name}.path must be repository-relative and may not "
                "escape the repository"
            )
        return cls(path=path)


@dataclass(frozen=True)
class AssetRecord:
    """Canonical identity, roles, resources, and docking metadata."""

    asset_id: str
    aliases: Tuple[str, ...]
    role: str
    searchable: bool
    tags: Tuple[str, ...]
    resources: Mapping[str, ResourceRecord]
    out_docks: Mapping[str, Any]
    in_docks: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AssetRecord":
        asset_id = _nonempty(value.get("id", ""), "asset.id")
        aliases = _strings(value.get("aliases", ()), f"{asset_id}.aliases")
        tags = _strings(value.get("tags", ()), f"{asset_id}.tags")
        role = _nonempty(value.get("role", ""), f"{asset_id}.role")
        raw_resources = value.get("resources")
        if not isinstance(raw_resources, Mapping):
            raise ValueError(f"{asset_id}.resources must be an object")
        resources = {
            _nonempty(name, f"{asset_id}.resources key"):
            ResourceRecord.from_mapping(
                record,
                field_name=f"{asset_id}.resources.{name}",
            )
            for name, record in raw_resources.items()
        }
        missing = sorted(set(_REQUIRED_RESOURCES) - set(resources))
        if missing:
            raise ValueError(
                f"{asset_id}.resources is missing required entries: {missing}"
            )
        return cls(
            asset_id=asset_id,
            aliases=aliases,
            role=role,
            searchable=bool(value.get("searchable", False)),
            tags=tags,
            resources=resources,
            out_docks=dict(value.get("out_docks", {})),
            in_docks=dict(value.get("in_docks", {})),
            metadata=dict(value.get("metadata", {})),
        )

    def resource(self, name: str) -> ResourceRecord:
        try:
            return self.resources[str(name)]
        except KeyError as exc:
            raise KeyError(
                f"asset {self.asset_id!r} has no resource {name!r}"
            ) from exc


@dataclass(frozen=True)
class AssetSelector:
    """Task-level, asset-count-independent catalog filter.

    Non-empty filter groups are combined with AND. ``ids`` accepts canonical
    IDs or declared aliases. ``all_tags`` and ``any_tags`` make future catalog
    changes data-only as long as role/tag semantics remain stable.
    """

    ids: Tuple[str, ...] = ()
    roles: Tuple[str, ...] = ()
    all_tags: Tuple[str, ...] = ()
    any_tags: Tuple[str, ...] = ()
    searchable_only: bool = True

    @classmethod
    def from_mapping(
        cls,
        value: Optional[Mapping[str, Any]],
    ) -> "AssetSelector":
        data = dict(value or {})
        return cls(
            ids=_strings(data.get("ids", ()), "selector.ids"),
            roles=_strings(data.get("roles", ()), "selector.roles"),
            all_tags=_strings(
                data.get("all_tags", ()),
                "selector.all_tags",
            ),
            any_tags=_strings(
                data.get("any_tags", ()),
                "selector.any_tags",
            ),
            searchable_only=bool(data.get("searchable_only", True)),
        )


@dataclass(frozen=True)
class AssetCatalog:
    """Loaded immutable view of one catalog version."""

    catalog_id: str
    catalog_version: int
    path: Path
    assets: Tuple[AssetRecord, ...]
    _by_id: Mapping[str, AssetRecord]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        path: Path,
    ) -> "AssetCatalog":
        schema_version = int(value.get("schema_version", -1))
        if schema_version != ASSET_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                "unsupported asset catalog schema_version "
                f"{schema_version}; expected {ASSET_CATALOG_SCHEMA_VERSION}"
            )
        catalog_id = _nonempty(value.get("catalog_id", ""), "catalog_id")
        catalog_version = int(value.get("catalog_version", 0))
        if catalog_version <= 0:
            raise ValueError("catalog_version must be positive")
        raw_assets = value.get("assets")
        if not isinstance(raw_assets, list) or not raw_assets:
            raise ValueError("catalog assets must be a non-empty list")
        assets = tuple(AssetRecord.from_mapping(item) for item in raw_assets)
        by_id: Dict[str, AssetRecord] = {}
        for asset in assets:
            for name in (asset.asset_id, *asset.aliases):
                if name in by_id:
                    raise ValueError(
                        f"asset ID or alias {name!r} is declared more than once"
                    )
                by_id[name] = asset
        return cls(
            catalog_id=catalog_id,
            catalog_version=catalog_version,
            path=Path(path),
            assets=assets,
            _by_id=by_id,
        )

    def resolve(self, asset_id_or_alias: str) -> AssetRecord:
        name = _nonempty(asset_id_or_alias, "asset ID")
        try:
            return self._by_id[name]
        except KeyError as exc:
            raise KeyError(f"unknown asset ID or alias {name!r}") from exc

    def canonical_id(self, asset_id_or_alias: str) -> str:
        return self.resolve(asset_id_or_alias).asset_id

    def select(
        self,
        selector: Optional[AssetSelector] = None,
    ) -> Tuple[AssetRecord, ...]:
        selected = selector or AssetSelector()
        requested = (
            {self.canonical_id(value) for value in selected.ids}
            if selected.ids
            else set()
        )
        roles = set(selected.roles)
        all_tags = set(selected.all_tags)
        any_tags = set(selected.any_tags)
        result = []
        for asset in self.assets:
            tags = set(asset.tags)
            if selected.searchable_only and not asset.searchable:
                continue
            if requested and asset.asset_id not in requested:
                continue
            if roles and asset.role not in roles:
                continue
            if all_tags and not all_tags.issubset(tags):
                continue
            if any_tags and not any_tags.intersection(tags):
                continue
            result.append(asset)
        return tuple(result)

    def resource_path(self, asset: AssetRecord, name: str) -> Path:
        repository_root = self.path.resolve().parents[2]
        resolved = (repository_root / asset.resource(name).path).resolve()
        try:
            resolved.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(
                f"asset resource escapes repository: {resolved}"
            ) from exc
        return resolved

    def verify_integrity(self) -> Mapping[str, Any]:
        """Verify files and mesh/contact/cage/weight consistency."""

        import numpy as np

        reports = []
        for asset in self.assets:
            paths = {
                name: self.resource_path(asset, name)
                for name in asset.resources
            }
            for name in asset.resources:
                path = paths[name]
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{asset.asset_id}.{name} does not exist: {path}"
                    )

            mesh_vertices = _obj_vertex_count(paths["mesh"])
            cage_handles = _point_file_count(
                paths["cage"],
                count_header=True,
            )
            contact_count = _point_file_count(
                paths["contacts"],
                count_header=True,
            )
            contact_ids = np.load(str(paths["contact_ids"]))
            weights = np.load(str(paths["weights"]))
            if contact_ids.ndim != 1 or contact_ids.shape[0] != contact_count:
                raise ValueError(
                    f"{asset.asset_id} contact/contact-ID dimensions differ: "
                    f"{contact_count} vs {contact_ids.shape}"
                )
            if contact_ids.dtype.kind not in {"i", "u"}:
                raise ValueError(
                    f"{asset.asset_id} contact IDs must be integers"
                )
            if contact_ids.size and (
                int(contact_ids.min()) < 0
                or int(contact_ids.max()) >= mesh_vertices
            ):
                raise ValueError(
                    f"{asset.asset_id} contact IDs exceed mesh vertices"
                )
            # The established LBS representation stores four homogeneous
            # coefficient rows per cage handle.
            expected_weights = (4 * cage_handles, mesh_vertices)
            if weights.shape != expected_weights:
                raise ValueError(
                    f"{asset.asset_id} weights shape {weights.shape} does not "
                    f"match cage/mesh dimensions {expected_weights}"
                )
            contacts = np.loadtxt(paths["contacts"], skiprows=1).reshape(-1, 3)
            vertices = np.asarray([
                [float(v) for v in line.split()[1:4]]
                for line in paths["mesh"].read_text().splitlines() if line.startswith("v ")
            ])
            if not np.isfinite(weights).all() or not np.isfinite(contacts).all() or not np.isfinite(vertices).all():
                raise ValueError(f"{asset.asset_id} has non-finite asset data")
            if not np.allclose(contacts, vertices[contact_ids], rtol=0, atol=5e-7):
                raise ValueError(f"{asset.asset_id} contact coordinates disagree with mesh vertex IDs")
            reports.append(
                {
                    "id": asset.asset_id,
                    "mesh_vertices": mesh_vertices,
                    "cage_handles": cage_handles,
                    "contacts": contact_count,
                    "weights_shape": list(weights.shape),
                }
            )
        return {
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "assets": reports,
        }


def _obj_vertex_count(path: Path) -> int:
    count = sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    )
    if count <= 0:
        raise ValueError(f"mesh has no vertices: {path}")
    return count


def _point_file_count(path: Path, *, count_header: bool) -> int:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if count_header:
        try:
            expected = int(lines[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid cage header: {path}") from exc
        actual = len(lines) - 1
        if actual != expected:
            raise ValueError(
                f"cage header/body count differs for {path}: "
                f"{expected} vs {actual}"
            )
        return actual
    if not lines:
        raise ValueError(f"point file is empty: {path}")
    if any(len(line.split()) < 3 for line in lines):
        raise ValueError(f"point file has a non-3D row: {path}")
    return len(lines)


def load_asset_catalog(path: Path) -> AssetCatalog:
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("asset catalog root must be an object")
    return AssetCatalog.from_mapping(payload, path=catalog_path)
