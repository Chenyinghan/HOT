"""Apply geometry and assemble render meshes for a parameterized scene."""
import os
from typing import Optional
import xml.etree.ElementTree as ET
import numpy as np

class DesignBundle:
    def __init__(self, spec, design_np, design_torch, init_cage_params: np.ndarray):
        self.spec = spec
        self.design_np = design_np
        self.design_torch = design_torch
        self.init_cage_params = init_cage_params.astype(np.float64)
        self.ndof_cage = int(init_cage_params.shape[0])
        self.model_path = None
        self._static_render_mesh_cache = {}

    def _load_static_render_mesh(self, xml_dir: str, mesh_path: str):
        from pathlib import Path
        from bilevel.parameterization.scene import _read_mesh

        full_path = mesh_path
        if not os.path.isabs(full_path):
            full_path = os.path.normpath(os.path.join(xml_dir, mesh_path))
        if full_path not in self._static_render_mesh_cache:
            vertices, faces = _read_mesh(Path(full_path))
            V = np.ascontiguousarray(vertices, dtype=np.float64)
            F = np.ascontiguousarray(faces, dtype=np.int32)
            self._static_render_mesh_cache[full_path] = (V, F)
        V, F = self._static_render_mesh_cache[full_path]
        return np.copy(V), np.copy(F)

    @staticmethod
    def _xml_render_vector(
        text: Optional[str],
        *,
        size: int,
        default,
        name: str,
    ) -> np.ndarray:
        if text in (None, ""):
            return np.asarray(default, dtype=np.float64)
        values = np.asarray(
            [float(token) for token in str(text).split()],
            dtype=np.float64,
        )
        if values.shape != (int(size),):
            raise ValueError(
                f"XML render {name} must contain {size} values, "
                f"got {text!r}"
            )
        return values

    @staticmethod
    def _xml_render_quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
        values = np.asarray(quaternion, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(values))
        if norm <= 0.0:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = values / norm
        return np.asarray(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _transform_static_render_mesh(
        cls,
        vertices: np.ndarray,
        body: ET.Element,
        default_body: Optional[ET.Element],
    ) -> np.ndarray:
        """Match BodyAbstract's XML mesh-to-body rendering transform."""

        scale_text = body.attrib.get("scale")
        if scale_text in (None, "") and default_body is not None:
            scale_text = default_body.attrib.get("scale")
        scale = cls._xml_render_vector(
            scale_text,
            size=3,
            default=(1.0, 1.0, 1.0),
            name="scale",
        )
        transformed = np.asarray(vertices, dtype=np.float64) * scale.reshape(
            3, 1
        )

        # RedMax only applies this extra frame for a nested <visual>; the
        # body's own pos/quat belongs to the articulated body transform and
        # must not be baked into its local rendering vertices.
        visual = body.find("visual")
        if body.attrib.get("mesh") or visual is None:
            return np.ascontiguousarray(transformed, dtype=np.float64)

        visual_position = cls._xml_render_vector(
            visual.attrib.get("pos"),
            size=3,
            default=(0.0, 0.0, 0.0),
            name="visual pos",
        )
        visual_quaternion = cls._xml_render_vector(
            visual.attrib.get("quat"),
            size=4,
            default=(1.0, 0.0, 0.0, 0.0),
            name="visual quat",
        )
        rotation = cls._xml_render_quaternion_matrix(visual_quaternion)
        transformed = (
            rotation @ transformed
        ) + visual_position.reshape(3, 1)
        return np.ascontiguousarray(transformed, dtype=np.float64)

    @staticmethod
    def _render_mesh_arrays(mesh):
        V = np.ascontiguousarray(mesh.V, dtype=np.float64)
        F = np.ascontiguousarray(mesh.F, dtype=np.int32)
        if V.ndim != 2 or V.shape[0] != 3:
            raise ValueError(f"Render vertices must have shape (3, n), got {V.shape}")
        if F.ndim != 2 or F.shape[0] != 3:
            raise ValueError(f"Render faces must have shape (3, n), got {F.shape}")
        if F.size and (int(F.min()) < 0 or int(F.max()) >= V.shape[1]):
            raise ValueError(
                f"Render face indices out of range: min={int(F.min())} max={int(F.max())} vertices={V.shape[1]}"
            )
        return V, F

    def _complete_render_mesh(self, meshes):
        """
        RedMax expects one vertex array for every abstract body in XML order.
        The parameterizer returns meshes only for design-param bodies; static
        abstract bodies must keep their original OBJ vertices.
        """
        if not self.model_path:
            return [m.V for m in meshes], [m.F for m in meshes]

        xml_dir = os.path.dirname(os.path.abspath(self.model_path))
        root = ET.parse(self.model_path).getroot()
        default_body = root.find("./default/body")

        Vs = []
        Fs = []
        mapping = []
        mesh_idx = 0
        for link in root.findall(".//link"):
            body = link.find("body")
            if body is None or body.attrib.get("type") != "abstract":
                continue

            dp = link.attrib.get("design_params", "")
            if dp not in ("", "0"):
                if mesh_idx >= len(meshes):
                    raise ValueError(
                        "Not enough generated rendering meshes: "
                        f"need at least {mesh_idx + 1}, got {len(meshes)}"
                    )
                V, F = self._render_mesh_arrays(meshes[mesh_idx])
                Vs.append(V)
                Fs.append(F)
                mapping.append(
                    {
                        "render_index": int(mesh_idx),
                        "body_name": body.attrib.get("name", ""),
                        "mesh_vertices": int(V.shape[1]),
                        "mesh_faces": int(F.shape[1]),
                    }
                )
                mesh_idx += 1
            else:
                visual = body.find("visual")
                mesh_path = body.attrib.get("mesh", "")
                if not mesh_path and visual is not None:
                    mesh_path = visual.attrib.get("mesh", "")
                if not mesh_path:
                    raise ValueError(f"Abstract body '{body.attrib.get('name', '')}' has no mesh path")
                V, F = self._load_static_render_mesh(xml_dir, mesh_path)
                V = self._transform_static_render_mesh(
                    V,
                    body,
                    default_body,
                )
                Vs.append(V)
                Fs.append(F)

        if mesh_idx != len(meshes):
            raise ValueError(
                "Generated rendering mesh count does not match XML design abstract bodies: "
                f"used {mesh_idx}, got {len(meshes)}"
            )

        self.last_render_mesh_mapping = mapping
        return Vs, Fs

    def apply(self, sim, cage_params: np.ndarray, generate_mesh: bool):

        if generate_mesh:
            design_params, meshes = self.design_np.parameterize(cage_params, True)
            if len(design_params) != sim.ndof_p:
                raise ValueError(
                    f"design_params length {len(design_params)} does not match sim.ndof_p {sim.ndof_p}"
            )
            sim.set_design_params(design_params)
            Vs, Fs = self._complete_render_mesh(meshes)
            sim.set_rendering_mesh(Vs, Fs)
            return design_params, meshes
        else:
            design_params = self.design_np.parameterize(cage_params)
            if len(design_params) != sim.ndof_p:
                raise ValueError(
                    f"design_params length {len(design_params)} does not match sim.ndof_p {sim.ndof_p}"
                )
            sim.set_design_params(design_params)
            return design_params, None
