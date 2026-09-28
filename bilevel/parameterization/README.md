# Connected Head parameterization

This package implements the single supported optimization parameterization, `unified_connected_head_morphology`, using the geometric protocol `connected_direct_planar_hexahedron`.

Each deformable Head uses the direct 18-coordinate hexahedron representation. Mesh vertices and contact points follow the cage through interpolation weights, while connectivity constraints keep attached faces together. Connectors such as `tip_cube` keep fixed geometry, and Handle mounts keep their attachment constraints.

| Module | Responsibility |
| --- | --- |
| `base.py`, `connected_head.py` | Optimizer coordinates, layout, projection and retraction |
| `geometry.py`, `geometry_torch.py` | Direct hexahedron geometry and derivatives |
| `interpolation.py`, `interpolation_torch.py` | Cage interpolation and face frames |
| `topology.py`, `constraints.py` | Head connections and constraint assembly |
| `scene.py` | XML records, asset loading and RedMax parameter slices |
| `design.py`, `bundle.py` | NumPy/Torch geometry-to-simulator mapping and application |
| `runtime.py`, `mount.py` | Optimizer integration and mount-preserving refinement |
| `artifacts.py` | Saved-vector metadata, validation and layout conversion |
| `collision.py`, `function_group.py`, `rendering.py` | Collision checks, functional contacts and rendering |

Use `build_design_bundle` to construct geometry from an XML scene and `UnifiedConnectedHeadMorphology` for the active optimizer layout. The runtime validates the geometry protocol and asset consistency.

The `legacy_*` conversion helpers read saved parameter vectors and map them into the full internal geometry layout. Optimization uses `unified_connected_head_morphology`.
