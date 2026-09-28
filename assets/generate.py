"""Generate a complete cuboid asset for the default shape–action pipeline."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np

from assets.library.catalog import AssetCatalog
from bilevel.parameterization.interpolation import (
    FACE_CORNERS, TRILINEAR_VERTEX_SIGNS, trilinear_weights,
)

LIBRARY = Path(__file__).resolve().parent / "library"
# Public docking convention: +Z, +X, +Y, -X, -Z, -Y.
FACES = ((2, 1, 0, 1), (0, 1, 1, 2), (1, 1, 0, 2),
         (0, -1, 1, 2), (2, -1, 0, 1), (1, -1, 0, 2))
# Docking files use lexicographic corners, while the deformation uses O..G.
CORNER_ORDER = (0, 3, 2, 6, 1, 5, 4, 7)
MAX_VERTICES = 1_000_000


def surface_area(dimensions):
    x, y, z = map(float, dimensions)
    return float(2 * (x*y + x*z + y*z))


def axis_samples(dimensions, density):
    """Choose a symmetric lattice close to the requested unique points/area."""
    dimensions = np.asarray(dimensions, dtype=float)
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError("dimensions must be three finite positive lengths")
    if np.any(dimensions < 1e-9):
        raise ValueError("dimensions must be at least 1e-9 model units for the default parameterization")
    if not math.isfinite(density) or density <= 0:
        raise ValueError("contact density must be finite and positive")
    area = surface_area(dimensions)
    target = density * area
    if not math.isfinite(target) or target <= 0 or target > MAX_VERTICES:
        raise ValueError(f"requested sampling exceeds {MAX_VERTICES:,} surface points")
    spacing = 1 / math.sqrt(density)
    estimates = dimensions / spacing + 1
    if not np.all(np.isfinite(estimates)) or np.max(estimates) > MAX_VERTICES:
        raise ValueError("requested axis sampling exceeds the mesh size limit")
    choices = [range(max(2, int(n)-3), max(3, int(n)+5)) for n in estimates]
    best = None
    for counts in itertools.product(*choices):
        if any(dimensions[i] == dimensions[j] and counts[i] != counts[j]
               for i in range(3) for j in range(i+1, 3)):
            continue
        n = np.asarray(counts, dtype=np.int64)
        count = 2 * (n[0]*n[1] + n[0]*n[2] + n[1]*n[2]) - 4*sum(n) + 8
        error = abs(count-target) / target
        spacing_error = float(np.mean(np.abs(np.log(dimensions / (n-1) / spacing))))
        candidate = (error + .08*spacing_error, error, spacing_error, counts)
        if best is None or candidate < best:
            best = candidate
    counts = best[-1]
    mesh_count = 2 * sum(counts[u]*counts[v] for u, v in ((0, 1), (0, 2), (1, 2)))
    if mesh_count > MAX_VERTICES:
        raise ValueError(f"sampling requires more than {MAX_VERTICES:,} mesh vertices")
    return counts


def cuboid_mesh(dimensions, counts):
    """Face-local mesh vertices, outward triangles, and unique contact indices."""
    axes = [np.linspace(-length/2, length/2, n) for length, n in zip(dimensions, counts)]
    vertices, triangles, contact_ids, seen = [], [], [], set()
    for fixed, side, u, v in FACES:
        ids = np.empty((counts[u], counts[v]), dtype=np.int64)
        for i, j in itertools.product(range(counts[u]), range(counts[v])):
            key = [0, 0, 0]
            key[fixed] = counts[fixed]-1 if side > 0 else 0
            key[u], key[v] = i, j
            key = tuple(key)
            index = len(vertices)
            ids[i, j] = index
            vertices.append([axes[k][key[k]] for k in range(3)])
            if key not in seen:
                seen.add(key)
                contact_ids.append(index)
        normal = np.zeros(3); normal[fixed] = side
        for i, j in itertools.product(range(counts[u]-1), range(counts[v]-1)):
            a, b, c, d = ids[i, j], ids[i+1, j], ids[i, j+1], ids[i+1, j+1]
            for triangle in ((a, b, d), (a, d, c)):
                p, q, r = (np.asarray(vertices[k]) for k in triangle)
                if np.dot(np.cross(q-p, r-p), normal) < 0:
                    triangle = (triangle[0], triangle[2], triangle[1])
                triangles.append(triangle)
    return np.asarray(vertices), np.asarray(triangles), np.asarray(contact_ids, dtype=np.int64)


def _points(path, points):
    np.savetxt(path, points, fmt="%.17g", header=str(len(points)), comments="")


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_bundle(directory, name, dimensions, density, counts):
    vertices, triangles, ids = cuboid_mesh(dimensions, counts)
    uvw = (vertices + dimensions/2) / dimensions
    weights = trilinear_weights(uvw)
    corners = (TRILINEAR_VERTEX_SIGNS - .5) * dimensions
    # Six docking centers are geometric interfaces, not extra shape variables.
    centers = np.zeros((6, 3))
    for i, (axis, side, _, _) in enumerate(FACES):
        centers[i, axis] = side * dimensions[axis]/2
    cage = np.vstack((corners[list(CORNER_ORDER)], centers))
    # Catalog storage convention: four homogeneous rows per docking handle.
    # Only the eight corners carry deformation weights; there is no MVC path.
    storage_weights = np.zeros((len(vertices), 14))
    storage_weights[:, :8] = weights[:, CORNER_ORDER]
    homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
    lbs = (storage_weights[:, :, None] * homogeneous[:, None, :]).reshape(len(vertices), 56).T
    paths = {"mesh": f"meshes/{name}.obj", "cage": f"cages/{name}.txt",
             "contacts": f"contacts/{name}.txt", "contact_ids": f"contacts/{name}_id.npy",
             "weights": f"weights/{name}.npy",
             "deformation": f"deformation/{name}.json"}
    for relative in paths.values():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
    with (directory / paths["mesh"]).open("w") as handle:
        handle.write("# Cuboid generated by python -m assets.generate\n")
        np.savetxt(handle, vertices, fmt="v %.17g %.17g %.17g")
        np.savetxt(handle, triangles+1, fmt="f %d %d %d")
    _points(directory / paths["cage"], cage)
    _points(directory / paths["contacts"], vertices[ids])
    np.save(directory / paths["contact_ids"], ids)
    np.save(directory / paths["weights"], lbs)
    _json(directory / paths["deformation"], {
        "format": "planar_hex_asset_v1", "baseline_vertices": corners.tolist(),
        "faces": {f"{axis}:{side}": list(indices) for (axis, side), indices in FACE_CORNERS.items()},
        "vertex_count": len(vertices), "contact_count": len(ids),
        "mesh_vertex_uvw": uvw.tolist(), "mesh_vertex_weights": weights.tolist(),
        "contact_uvw": uvw[ids].tolist(), "contact_weights": weights[ids].tolist(),
    })
    np.testing.assert_allclose(weights @ corners, vertices, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=1e-12, atol=1e-12)
    specification = {"generator_version": 1,
                "geometry": {"shape": "cuboid", "dimensions": dimensions.tolist(),
                             "axis_samples": list(counts), "vertex_count": len(vertices),
                             "triangle_count": len(triangles), "surface_area": surface_area(dimensions)},
                "contacts": {"count": len(ids), "requested_density": density,
                             "actual_density": len(ids)/surface_area(dimensions)}}
    return specification, paths


def generate_cuboid(name, dimensions, contact_density, *, output=LIBRARY, force=False):
    """Write a checked asset and return its catalog entry, preserving docking policy."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("name must contain only letters, digits, underscores and hyphens")
    dimensions = np.asarray(dimensions, dtype=float)
    density = float(contact_density)
    counts = axis_samples(dimensions, density)
    output = Path(output).resolve()
    catalog_path = output / "catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {
        "schema_version": 1, "catalog_id": "hot.shared_assets", "catalog_version": 1, "assets": []}
    matches = [a for a in catalog["assets"] if Path(a["resources"]["mesh"]["path"]).stem == name]
    if len(matches) > 1:
        raise ValueError("multiple catalog entries share this mesh; resolve their ownership first")
    existing = matches[0] if matches else None
    if existing and existing["role"] != "head_primitive":
        raise ValueError("cuboid generation cannot replace a fixed root asset")
    owned = [f"{folder}/{name}{suffix}" for folder, suffix in (
        ("meshes", ".obj"), ("cages", ".txt"), ("contacts", ".txt"),
        ("contacts", "_id.npy"), ("weights", ".npy"),
        ("deformation", ".json"))]
    obsolete_weights = output / "weights" / f"{name}.txt"
    if not force and (existing or obsolete_weights.exists() or any((output / p).exists() for p in owned)):
        raise FileExistsError(f"asset {name!r} already exists; use --force to regenerate it")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".generate-", dir=output) as temp:
        stage = Path(temp)
        specification, paths = _write_bundle(stage, name, dimensions, density, counts)
        entry = copy.deepcopy(existing) if existing else {
            "id": f"primitive/{name}", "aliases": [], "role": "head_primitive",
            "searchable": True, "tags": ["head", "primitive", "cuboid"],
            "in_docks": {"0": [{"id": 0, "barycentric": [.25]*4, "facing_options": [0, 1, 2, 3]}]},
            "out_docks": {str(i): [{"id": 0, "barycentric": [.25]*4}] for i in range(6)},
            "metadata": {}}
        entry.setdefault("metadata", {})["specification"] = specification
        prefix = output.relative_to(output.parent.parent)
        resources = {key: {"path": str(prefix / relative)}
                     for key, relative in paths.items()}
        entry.setdefault("resources", {}).pop("weights_text", None)
        entry["resources"].update(resources)
        # Validate staged files using the same catalog checks as search.
        probe = copy.deepcopy(entry)
        stage_prefix = stage.relative_to(stage.parent.parent)
        probe["resources"] = {key: {**resource, "path": str(stage_prefix / paths[key])}
                              for key, resource in resources.items()}
        AssetCatalog.from_mapping({**catalog, "assets": [probe]}, path=stage/"catalog.json").verify_integrity()
        if existing:
            catalog["assets"][catalog["assets"].index(existing)] = entry
        else:
            catalog["assets"].append(entry)
        if catalog_path.exists():
            catalog["catalog_version"] += 1
        AssetCatalog.from_mapping(catalog, path=catalog_path)
        _json(stage / "catalog.json", catalog)
        # Publish only after every generated artifact has passed validation.
        for relative in owned + ["catalog.json"]:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / relative).replace(destination)
        obsolete_weights.unlink(missing_ok=True)
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="asset resource name (registered as primitive/NAME)")
    parser.add_argument("--dimensions", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"),
                        help="full cuboid side lengths in model units, centered at the origin")
    parser.add_argument("--contact-density", type=float, required=True,
                        help="target unique contact points per square model unit")
    parser.add_argument("--output", type=Path, default=LIBRARY, help="asset library directory")
    parser.add_argument("--force", action="store_true", help="replace an existing cuboid and refresh its catalog hashes")
    args = parser.parse_args(argv)
    try:
        result = generate_cuboid(args.name, args.dimensions, args.contact_density,
                                 output=args.output, force=args.force)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    contacts = result["metadata"]["specification"]["contacts"]
    print(f"Generated {result['id']} in {args.output}")
    print(f"Contacts: {contacts['count']}; requested density={contacts['requested_density']:.6g}; "
          f"actual density={contacts['actual_density']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
