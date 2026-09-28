"""Normalized inertial proxy for fixed-Handle structure evaluation.

The generated geometry still defines rendering, welds, and task contacts, but
it does not give the Handle artificial physical weight.  The invisible root
carrier, generated Handle, and functional markers retain only epsilon inertia
because exact zero-mass bodies can make the native dynamics system singular.
Each contact-bearing Head body carries a stable positive mass, so adding Head
parts naturally makes the searched tool heavier.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET


PROXY_POLICY = "massless_carrier_physical_head_v4"
ROOT_PROXY_BODY = "body_freeform_root"
ROOT_PROXY_MASS = 1e-4
ROOT_PROXY_INERTIA = (1e-4, 1e-4, 1e-4)
GEOMETRY_EPSILON_MASS = 1e-4
GEOMETRY_EPSILON_INERTIA = (1e-4, 1e-4, 1e-4)
HEAD_CONTACT_MASS = 0.2
HEAD_CONTACT_INERTIA = (0.2, 0.2, 0.2)

_SCAFFOLD_PREFIXES = (
    "body_tip_universal_handle_",
    "body_fixed_scaffold_",
)
_HEAD_PREFIXES = (
    "body_tool_",
    "body_searched_head_",
)


def _fmt(value: float) -> str:
    return f"{float(value):.12g}"


def _fmt_vec(values) -> str:
    return " ".join(_fmt(value) for value in values)


def _set_inertia(body: ET.Element, mass: float, inertia) -> None:
    body.attrib.pop("density", None)
    body.set("mass", _fmt(mass))
    body.set("inertia", _fmt_vec(inertia))


def _is_marker(name: str) -> bool:
    return name.startswith("body_") and name.endswith("_endeffector")


def _asset_role(body: ET.Element) -> str:
    role = str(body.attrib.get("asset_role", "")).strip()
    if role:
        return role
    name = str(body.attrib.get("name", ""))
    if name.startswith(_SCAFFOLD_PREFIXES):
        return "fixed_root"
    if name.startswith(_HEAD_PREFIXES):
        return "head"
    return ""


def _vector(text: str | None, default) -> tuple[float, ...]:
    values = tuple(float(value) for value in str(text or "").split())
    return values if values else tuple(float(value) for value in default)


def _quat_normalized(quat) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in quat)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-15:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in values)


def _quat_multiply(left, right) -> tuple[float, float, float, float]:
    aw, ax, ay, az = _quat_normalized(left)
    bw, bx, by, bz = _quat_normalized(right)
    return _quat_normalized(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _quat_rotate(quat, point) -> tuple[float, float, float]:
    w, x, y, z = _quat_normalized(quat)
    px, py, pz = (float(value) for value in point)
    # Rotation matrix for a normalized wxyz quaternion.
    return (
        (1.0 - 2.0 * (y * y + z * z)) * px
        + 2.0 * (x * y - z * w) * py
        + 2.0 * (x * z + y * w) * pz,
        2.0 * (x * y + z * w) * px
        + (1.0 - 2.0 * (x * x + z * z)) * py
        + 2.0 * (y * z - x * w) * pz,
        2.0 * (x * z - y * w) * px
        + 2.0 * (y * z + x * w) * py
        + (1.0 - 2.0 * (x * x + y * y)) * pz,
    )


def _compose(parent_pos, parent_quat, local_pos, local_quat):
    offset = _quat_rotate(parent_quat, local_pos)
    return (
        tuple(float(parent_pos[i]) + offset[i] for i in range(3)),
        _quat_multiply(parent_quat, local_quat),
    )


def _handle_center_in_root(root: ET.Element) -> tuple[float, float, float]:
    """Return the Handle body center in the freeform-root link frame."""

    root_proxy = root.find(f".//body[@name='{ROOT_PROXY_BODY}']")
    if root_proxy is None:
        raise ValueError(f"Proxy dynamics requires {ROOT_PROXY_BODY!r}")
    freeform_link = next(
        (
            link
            for link in root.iter("link")
            if root_proxy in list(link.findall("body"))
        ),
        None,
    )
    if freeform_link is None:
        raise ValueError("Root proxy body is not attached to a link")

    found = None

    def visit(link: ET.Element, parent_pos, parent_quat, include_joint: bool) -> None:
        nonlocal found
        if found is not None:
            return
        link_pos = parent_pos
        link_quat = parent_quat
        if include_joint:
            joint = link.find("joint")
            if joint is not None:
                link_pos, link_quat = _compose(
                    parent_pos,
                    parent_quat,
                    _vector(joint.attrib.get("pos"), (0.0, 0.0, 0.0)),
                    _vector(joint.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)),
                )
        for body in link.findall("body"):
            if _asset_role(body) == "fixed_root":
                body_pos, _ = _compose(
                    link_pos,
                    link_quat,
                    _vector(body.attrib.get("pos"), (0.0, 0.0, 0.0)),
                    _vector(body.attrib.get("quat"), (1.0, 0.0, 0.0, 0.0)),
                )
                found = body_pos
                return
        for child in link.findall("link"):
            visit(child, link_pos, link_quat, True)

    visit(freeform_link, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), False)
    if found is None:
        raise ValueError("Proxy dynamics found no Handle body to anchor the root inertia")
    return found


def apply_proxy_dynamics(root: ET.Element) -> dict:
    """Apply the normalized proxy policy and return an auditable report."""

    root_proxy = root.find(f".//body[@name='{ROOT_PROXY_BODY}']")
    if root_proxy is None:
        raise ValueError(f"Proxy dynamics requires {ROOT_PROXY_BODY!r}")
    _set_inertia(root_proxy, ROOT_PROXY_MASS, ROOT_PROXY_INERTIA)
    root_proxy.set("pos", _fmt_vec(_handle_center_in_root(root)))
    # The inertia is isotropic, so proxy orientation is deliberately irrelevant.
    root_proxy.set("quat", "1 0 0 0")
    root_proxy.set("collision", "false")
    root_proxy.set("ground_contact", "false")
    root_proxy.set("rgba", "0 0 0 0")

    scaffold = []
    heads = []
    markers = []
    for body in root.iter("body"):
        name = str(body.attrib.get("name", ""))
        if not name or name == ROOT_PROXY_BODY:
            continue
        role = _asset_role(body)
        if role == "fixed_root":
            _set_inertia(body, GEOMETRY_EPSILON_MASS, GEOMETRY_EPSILON_INERTIA)
            scaffold.append(name)
            body.set("collision", "false")
            body.set("ground_contact", "false")
        elif role == "head":
            _set_inertia(body, HEAD_CONTACT_MASS, HEAD_CONTACT_INERTIA)
            heads.append(name)
        elif _is_marker(name):
            _set_inertia(body, GEOMETRY_EPSILON_MASS, GEOMETRY_EPSILON_INERTIA)
            body.set("collision", "false")
            body.set("ground_contact", "false")
            markers.append(name)

    if not scaffold:
        raise ValueError("Proxy dynamics found no generated Handle/scaffold body")
    if not heads:
        raise ValueError("Proxy dynamics found no searched Head body")
    return {
        "policy": PROXY_POLICY,
        "root_proxy_body": ROOT_PROXY_BODY,
        "root_carrier_mass": ROOT_PROXY_MASS,
        "root_carrier_inertia": list(ROOT_PROXY_INERTIA),
        "geometry_epsilon_mass": GEOMETRY_EPSILON_MASS,
        "geometry_epsilon_inertia": list(GEOMETRY_EPSILON_INERTIA),
        "head_contact_mass": HEAD_CONTACT_MASS,
        "head_contact_inertia": list(HEAD_CONTACT_INERTIA),
        "scaffold_bodies": scaffold,
        "head_bodies": heads,
        "marker_bodies": markers,
    }


def audit_proxy_dynamics(root: ET.Element) -> dict:
    """Check an XML without mutating it."""

    failures = []
    root_proxy = root.find(f".//body[@name='{ROOT_PROXY_BODY}']")
    if root_proxy is None:
        failures.append("missing_root_proxy")
    else:
        if root_proxy.attrib.get("mass") != _fmt(ROOT_PROXY_MASS):
            failures.append("root_proxy_mass_mismatch")
        if root_proxy.attrib.get("inertia") != _fmt_vec(ROOT_PROXY_INERTIA):
            failures.append("root_proxy_inertia_mismatch")
        if root_proxy.attrib.get("collision") != "false":
            failures.append("root_proxy_collision_enabled")
        if root_proxy.attrib.get("ground_contact") != "false":
            failures.append("root_proxy_ground_contact_enabled")
        if "density" in root_proxy.attrib:
            failures.append("root_proxy_density_not_removed")
        try:
            expected_anchor = _handle_center_in_root(root)
            actual_anchor = _vector(root_proxy.attrib.get("pos"), (0.0, 0.0, 0.0))
            if len(actual_anchor) != 3 or any(
                abs(actual_anchor[i] - expected_anchor[i]) > 1e-8 for i in range(3)
            ):
                failures.append("root_proxy_not_at_handle_center")
        except ValueError as exc:
            failures.append(f"root_proxy_anchor_invalid:{exc}")

    geometry = []
    scaffold = []
    heads = []
    for body in root.iter("body"):
        name = str(body.attrib.get("name", ""))
        role = _asset_role(body)
        if (
            role in {"fixed_root", "head"}
            or _is_marker(name)
        ):
            geometry.append(name)
            if role == "fixed_root":
                scaffold.append(name)
                if body.attrib.get("collision") != "false":
                    failures.append(f"handle_collision_enabled:{name}")
                if body.attrib.get("ground_contact") != "false":
                    failures.append(f"handle_ground_contact_enabled:{name}")
            elif role == "head":
                heads.append(name)
            expected_mass = (
                HEAD_CONTACT_MASS
                if role == "head"
                else GEOMETRY_EPSILON_MASS
            )
            expected_inertia = (
                HEAD_CONTACT_INERTIA
                if role == "head"
                else GEOMETRY_EPSILON_INERTIA
            )
            if body.attrib.get("mass") != _fmt(expected_mass):
                failures.append(f"geometry_mass_mismatch:{name}")
            if body.attrib.get("inertia") != _fmt_vec(expected_inertia):
                failures.append(f"geometry_inertia_mismatch:{name}")
            if "density" in body.attrib:
                failures.append(f"geometry_density_not_removed:{name}")
    if not scaffold:
        failures.append("missing_proxy_scaffold")
    if not heads:
        failures.append("missing_proxy_head")
    return {
        "ok": not failures,
        "policy": PROXY_POLICY,
        "failures": failures,
        "geometry_bodies": geometry,
        "scaffold_bodies": scaffold,
        "head_bodies": heads,
    }


__all__ = [
    "GEOMETRY_EPSILON_INERTIA",
    "GEOMETRY_EPSILON_MASS",
    "HEAD_CONTACT_INERTIA",
    "HEAD_CONTACT_MASS",
    "PROXY_POLICY",
    "ROOT_PROXY_BODY",
    "ROOT_PROXY_INERTIA",
    "ROOT_PROXY_MASS",
    "apply_proxy_dynamics",
    "audit_proxy_dynamics",
]
