"""Rendering artifact helpers shared by the canonical morphology pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_render_mesh_mapping(
    mapping_path: str | Path,
    runner: Any,
    meshes: list[Any],
    xml_path: str | Path,
) -> None:
    """Write the generated-mesh-to-RedMax-body mapping used by replays."""
    bundle = getattr(runner, "design_bundle", None)
    if bundle is None:
        return
    entries = []
    mesh_iter = iter(meshes or [])
    for idx, render_rec in enumerate(
        getattr(bundle.spec, "render_records", []) or []
    ):
        if render_rec.source_record is None:
            continue
        try:
            mesh = next(mesh_iter)
        except StopIteration:
            break
        rec = render_rec.source_record
        entries.append(
            {
                "render_index": int(idx),
                "body_name": render_rec.body_name,
                "source_link_name": rec.link_name,
                "source_body_name": rec.body_name,
                "mask": int(getattr(rec, "mask", 0)),
                "mesh_vertices": int(np.asarray(mesh.V).shape[1]),
                "mesh_faces": (
                    int(np.asarray(mesh.F).shape[1])
                    if getattr(mesh, "F", None) is not None
                    else 0
                ),
            }
        )
    path = Path(mapping_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "xml_path": str(xml_path),
                "generic_design_protocol": getattr(
                    bundle, "generic_design_protocol", None
                ),
                "entries": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
