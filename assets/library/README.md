# Shared asset library

This directory is the canonical source for shared Handle and Head geometry. Tasks select assets through `catalog.json`.

The catalog interface has its own version. Asset IDs can evolve while the loader, selector, BASS, XML compiler and task interfaces use the same catalog contract.

## Stable contracts

- `catalog.schema.json` defines catalog version 1.
- `catalog.py` loads IDs and aliases, applies task selectors, and verifies resource files and array dimensions.
- Every asset declares a role, searchability, tags, resources, and docks.
- Tasks may select assets by canonical IDs, roles, or tags.
- Canonical resources require mesh, cage, contacts, contact IDs, and weights.
- Binary `.npy` files store the weight matrices.
- `python -m assets.generate` adds or refreshes cuboids in the catalog.

## Current canonical IDs

- `root/universal_handle`
- `primitive/cube`
- `primitive/panel`
- `primitive/small_cuboid`

These are the current canonical IDs. The catalog can grow as Head assets are added.

## Universal Handle

The shared Handle is a closed 16-sided cylinder with:

- length `8.0`;
- radius `0.45`;
- local long axis `+Y/-Y`;
- center/grip frame at `(0, 0, 0)`;
- end interfaces at `(0, 4, 0)` and `(0, -4, 0)`.

The checked-in Handle has its physical specification in the catalog entry’s `metadata.specification`.

For deformable Head assets, use the unified [cuboid generator](../generate.py).

Task-owned scene meshes and contact samples live in `tasks/<task>/scene/`. Scoop and Sweep use spheres with different tessellations; both Sweep balls share one task-local mesh. Scoop's box visuals reuse the shared cube through XML scaling.
