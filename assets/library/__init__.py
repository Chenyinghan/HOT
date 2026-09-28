"""Versioned, task-independent asset catalog."""

from .catalog import (
    ASSET_CATALOG_SCHEMA_VERSION,
    AssetCatalog,
    AssetRecord,
    AssetSelector,
    ResourceRecord,
    load_asset_catalog,
)

__all__ = [
    "ASSET_CATALOG_SCHEMA_VERSION",
    "AssetCatalog",
    "AssetRecord",
    "AssetSelector",
    "ResourceRecord",
    "load_asset_catalog",
]
