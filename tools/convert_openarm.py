#!/usr/bin/env python3
"""
Convert OpenArm XACRO description files to a standalone URDF for the VILMA 3D viewer.
Generates the bimanual configuration (left arm + right arm + both grippers).
Works WITHOUT ROS by either:
  A) Monkeypatching xacro's package finder, OR
  B) Manually building the URDF from YAML configs (fallback).
"""

import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from math import pi

import yaml

# --------------------------------------------------------------------------- #
#                               PATHS                                          #
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
PKG_DIR = ROOT / "openarm_description-main" / "openarm_description-main"
OUT_DIR = ROOT / "data" / "Common" / "robot_models" / "openarm"

YAML_DIR = PKG_DIR / "config"
MESH_SRC = PKG_DIR / "meshes"

# --------------------------------------------------------------------------- #
#                           YAML LOADERS                                       #
# --------------------------------------------------------------------------- #

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

arm_kinematics      = load_yaml(YAML_DIR / "arm/v10/kinematics.yaml")
arm_kinematics_link = load_yaml(YAML_DIR / "arm/v10/kinematics_link.yaml")
arm_kinematics_off  = load_yaml(YAML_DIR / "arm/v10/kinematics_offset.yaml")
arm_joint_limits    = load_yaml(YAML_DIR / "arm/v10/joint_limits.yaml")
arm_inertials       = load_yaml(YAML_DIR / "arm/v10/inertials.yaml")
arm_control_gains   = load_yaml(YAML_DIR / "arm/v10/control_gains.yaml")

body_kinematics      = load_yaml(YAML_DIR / "body/v10/kinematics.yaml")
body_kinematics_link = load_yaml(YAML_DIR / "body/v10/kinematics_link.yaml")
body_inertials       = load_yaml(YAML_DIR / "body/v10/inertials.yaml")

hand_kinematics_link = load_yaml(YAML_DIR / "hand/openarm_hand/kinematics_link.yaml")
hand_inertials       = load_yaml(YAML_DIR / "hand/openarm_hand/inertials.yaml")

# --------------------------------------------------------------------------- #
#                      METHOD A – TRY XACRO                                    #
# --------------------------------------------------------------------------- #

def try_xacro() -> str | None:
    """Attempt to process the xacro file by monkey-patching $(find ...) resolution."""
    try:
        import xacro  # noqa: F811

        # Monkey-patch $(find openarm_description) → local path
        pkg_path_posix = PKG_DIR.as_posix()

        _orig_process = xacro.process_file

        def _patched_process(input_file_name, **kwargs):
            # Temporarily replace $(find openarm_description) in every included file
            return _orig_process(input_file_name, **kwargs)

        # Patch the substitution_args resolve function
        if hasattr(xacro, "substitution_args"):
            sa = xacro.substitution_args
            if hasattr(sa, "resolve_args"):
                _orig_resolve = sa.resolve_args

                def _patched_resolve(args_str, context=None, resolve_anon=True, filename=None):
                    args_str = args_str.replace(
                        "$(find openarm_description)", pkg_path_posix
                    )
                    return _orig_resolve(
                        args_str, context=context,
                        resolve_anon=resolve_anon, filename=filename
                    )

                sa.resolve_args = _patched_resolve

        # Also try patching at the ament level
        try:
            import ament_index_python.packages as ap

            _orig_get = ap.get_package_share_directory

            def _patched_get(package_name):
                if package_name == "openarm_description":
                    return pkg_path_posix
                return _orig_get(package_name)

            ap.get_package_share_directory = _patched_get
        except ImportError:
            pass

        xacro_file = str(PKG_DIR / "urdf" / "robot" / "v10.urdf.xacro")
        doc = xacro.process_file(
            xacro_file,
            mappings={
                "bimanual": "true",
                "hand": "true",
                "ros2_control": "false",
                "no_prefix": "false",
            },
        )
        urdf_str = doc.toprettyxml(indent="  ")
        # Replace package:// paths with relative paths
        urdf_str = urdf_str.replace("package://openarm_description/", "")
        return urdf_str

    except Exception as e:
        print(f"[INFO] xacro approach failed ({e}), falling back to manual URDF generation.")
        return None


# --------------------------------------------------------------------------- #
#                  METHOD B – MANUAL URDF GENERATION                           #
# --------------------------------------------------------------------------- #

def _fmt(v) -> str:
    """Format float without trailing zeros. Handles strings from YAML."""
    return f"{float(v):.10g}"


def _origin(x=0, y=0, z=0, roll=0, pitch=0, yaw=0) -> ET.Element:
    el = ET.Element("origin")
    el.set("xyz", f"{_fmt(x)} {_fmt(y)} {_fmt(z)}")
    el.set("rpy", f"{_fmt(roll)} {_fmt(pitch)} {_fmt(yaw)}")
    return el


def _inertial_from_arm(name: str, reflect: int = 1) -> ET.Element | None:
    if name not in arm_inertials:
        return None
    I = arm_inertials[name]
    inertial = ET.SubElement(ET.Element("_"), "inertial")
    o = I["origin"]
    inertial.append(_origin(o["x"], reflect * o["y"], o["z"], o.get("roll", 0), o.get("pitch", 0), o.get("yaw", 0)))
    mass = ET.SubElement(inertial, "mass")
    mass.set("value", _fmt(I["mass"]))
    if "inertia" in I:
        inrt = I["inertia"]
        ie = ET.SubElement(inertial, "inertia")
        for k in ("xx", "xy", "xz", "yy", "yz", "zz"):
            ie.set("i" + k, _fmt(inrt.get(k, 0)))
    return inertial


def _inertial_from_body(name: str) -> ET.Element | None:
    if name not in body_inertials:
        return None
    I = body_inertials[name]
    inertial = ET.SubElement(ET.Element("_"), "inertial")
    o = I["origin"]
    # body inertials uses xyz/rpy strings
    if isinstance(o.get("xyz"), str):
        xyz_vals = list(map(float, o["xyz"].split()))
        rpy_vals = list(map(float, o.get("rpy", "0 0 0").split()))
    else:
        xyz_vals = [o.get("x", 0), o.get("y", 0), o.get("z", 0)]
        rpy_vals = [o.get("roll", 0), o.get("pitch", 0), o.get("yaw", 0)]
    inertial.append(_origin(*xyz_vals, *rpy_vals))
    mass = ET.SubElement(inertial, "mass")
    mass.set("value", _fmt(I["mass"]))
    if "inertia" in I:
        inrt = I["inertia"]
        ie = ET.SubElement(inertial, "inertia")
        for k in ("xx", "xy", "xz", "yy", "yz", "zz"):
            ie.set("i" + k, _fmt(inrt.get(k, 0)))
    return inertial


def _inertial_from_hand(name: str) -> ET.Element | None:
    if name not in hand_inertials:
        return None
    I = hand_inertials[name]
    inertial = ET.SubElement(ET.Element("_"), "inertial")
    o = I["origin"]
    inertial.append(_origin(o["x"], o["y"], o["z"], o.get("roll", 0), o.get("pitch", 0), o.get("yaw", 0)))
    mass = ET.SubElement(inertial, "mass")
    mass.set("value", _fmt(I["mass"]))
    if "inertia" in I:
        inrt = I["inertia"]
        ie = ET.SubElement(inertial, "inertia")
        for k in ("xx", "xy", "xz", "yy", "yz", "zz"):
            ie.set("i" + k, _fmt(inrt.get(k, 0)))
    return inertial


def _arm_link(robot: ET.Element, link_name: str, prefix: str, reflect: int):
    """Create an arm link element (link0–link7).
    Uses lightweight collision STL meshes for visuals to keep size small."""
    full_name = f"{prefix}{link_name}"
    link = ET.SubElement(robot, "link")
    link.set("name", full_name)

    # Visual — use collision _symp.stl (tiny) instead of visual .dae (huge)
    if link_name in arm_kinematics_link:
        kl = arm_kinematics_link[link_name]["kinematic"]
        visual = ET.SubElement(link, "visual")
        visual.set("name", f"{full_name}_visual")
        visual.append(_origin(kl["x"], kl["y"], kl["z"], kl.get("roll", 0), kl.get("pitch", 0), kl.get("yaw", 0)))
        geom = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geom, "mesh")
        mesh.set("filename", f"meshes/arm/v10/collision/{link_name}_symp.stl")
        mesh.set("scale", f"0.001 {0.001 * reflect} 0.001")


def _arm_joint(robot: ET.Element, joint_num: int, prefix: str, reflect: int,
               parent_link: str, child_link: str):
    """Create an arm revolute joint."""
    jname = f"joint{joint_num}"
    joint = ET.SubElement(robot, "joint")
    joint.set("name", f"{prefix}{jname}")
    joint.set("type", "revolute")

    parent = ET.SubElement(joint, "parent")
    parent.set("link", parent_link)
    child = ET.SubElement(joint, "child")
    child.set("link", child_link)

    # Origin from kinematics
    k = arm_kinematics[jname]["kinematic"]
    # Apply kinematic offset for joint2 (bimanual offset)
    if jname in arm_kinematics_off:
        off = arm_kinematics_off[jname]["kinematic_offset"]
        ox = k["x"] + off["x"]
        oy = k["y"] + off["y"]
        oz = k["z"] + off["z"]
        oroll = reflect * (k.get("roll", 0) + off.get("roll", 0))
        opitch = reflect * (k.get("pitch", 0) + off.get("pitch", 0))
        oyaw = reflect * (k.get("yaw", 0) + off.get("yaw", 0))
    else:
        ox, oy, oz = k["x"], k["y"], k["z"]
        oroll = reflect * k.get("roll", 0)
        opitch = reflect * k.get("pitch", 0)
        oyaw = reflect * k.get("yaw", 0)

    joint.append(_origin(ox, oy, oz, oroll, opitch, oyaw))

    # Axis
    axes = {
        1: (0, 0, 1),
        2: (-1, 0, 0),
        3: (0, 0, 1),
        4: (0, 1, 0),
        5: (0, 0, 1),
        6: (1, 0, 0),
        7: (0, reflect, 0),
    }
    ax = axes.get(joint_num, (0, 0, 1))
    axis_el = ET.SubElement(joint, "axis")
    axis_el.set("xyz", f"{ax[0]} {ax[1]} {ax[2]}")

    # Limits
    lim_cfg = arm_joint_limits[jname]["limit"]
    raw_lower = lim_cfg["lower"] * reflect
    raw_upper = lim_cfg["upper"] * reflect

    # Apply bimanual offsets
    if joint_num == 1:
        arm_type = "left" if "left" in prefix else "right"
        if arm_type == "left":
            raw_lower += -2.094396
            raw_upper += -2.094396
    if joint_num == 2:
        arm_type = "left" if "left" in prefix else "right"
        if arm_type == "right":
            raw_lower += pi / 2
            raw_upper += pi / 2
        elif arm_type == "left":
            raw_lower += -pi / 2
            raw_upper += -pi / 2

    lower = min(raw_lower, raw_upper)
    upper = max(raw_lower, raw_upper)

    limit = ET.SubElement(joint, "limit")
    limit.set("lower", _fmt(lower))
    limit.set("upper", _fmt(upper))
    limit.set("effort", _fmt(lim_cfg["effort"]))
    limit.set("velocity", _fmt(lim_cfg["velocity"]))


def _build_arm(robot: ET.Element, arm_prefix: str, connected_to: str,
               base_xyz: str, base_rpy: str):
    """Build a complete 7-DOF arm (links + joints)."""
    prefix = f"openarm_{arm_prefix}"  # e.g. "openarm_left_"
    reflect = 1 if arm_prefix == "right_" else -1

    # link0
    _arm_link(robot, "link0", prefix, reflect)

    # Fixed joint connecting body to arm base
    base_joint = ET.SubElement(robot, "joint")
    base_joint.set("name", f"{prefix}base_joint")
    base_joint.set("type", "fixed")
    pj = ET.SubElement(base_joint, "parent")
    pj.set("link", connected_to)
    cj = ET.SubElement(base_joint, "child")
    cj.set("link", f"{prefix}link0")
    xyz_vals = list(map(float, base_xyz.split()))
    rpy_vals = list(map(float, base_rpy.split()))
    base_joint.append(_origin(*xyz_vals, *rpy_vals))

    # links 1-7 and joints 1-7
    for i in range(1, 8):
        _arm_link(robot, f"link{i}", prefix, reflect)
        _arm_joint(
            robot, i, prefix, reflect,
            f"{prefix}link{i - 1}", f"{prefix}link{i}"
        )


def _build_hand(robot: ET.Element, arm_prefix: str, connected_to: str):
    """Build the openarm_hand (hand link + 2 finger links + joints)."""
    ee_prefix = f"openarm_{arm_prefix}"  # e.g. "openarm_left_"

    # --- Hand link ---
    hand_link = ET.SubElement(robot, "link")
    hand_link.set("name", f"{ee_prefix}hand")
    if "hand" in hand_kinematics_link:
        kl = hand_kinematics_link["hand"]["kinematic"]
        visual = ET.SubElement(hand_link, "visual")
        visual.set("name", f"{ee_prefix}hand_visual")
        visual.append(_origin(kl["x"], kl["y"], kl["z"], kl.get("roll", 0), kl.get("pitch", 0), kl.get("yaw", 0)))
        geom = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geom, "mesh")
        mesh.set("filename", "meshes/ee/openarm_hand/collision/hand.stl")
        mesh.set("scale", "0.001 0.001 0.001")

    # --- Hand fixed joint ---
    hand_joint = ET.SubElement(robot, "joint")
    hand_joint.set("name", f"{arm_prefix}openarm_hand_joint")
    hand_joint.set("type", "fixed")
    pj = ET.SubElement(hand_joint, "parent")
    pj.set("link", connected_to)
    cj = ET.SubElement(hand_joint, "child")
    cj.set("link", f"{ee_prefix}hand")
    hand_joint.append(_origin(0, 0, 0.1001, 0, 0, 0))  # xyz_ee default

    # --- Hand TCP ---
    tcp_link = ET.SubElement(robot, "link")
    tcp_link.set("name", f"{ee_prefix}hand_tcp")
    tcp_joint = ET.SubElement(robot, "joint")
    tcp_joint.set("name", f"{ee_prefix}hand_tcp_joint")
    tcp_joint.set("type", "fixed")
    pj = ET.SubElement(tcp_joint, "parent")
    pj.set("link", f"{ee_prefix}hand")
    cj = ET.SubElement(tcp_joint, "child")
    cj.set("link", f"{ee_prefix}hand_tcp")
    tcp_joint.append(_origin(0, 0, 0.08, 0, 0, 0))  # tcp_xyz default

    # --- Left finger ---
    lf_link = ET.SubElement(robot, "link")
    lf_link.set("name", f"{ee_prefix}left_finger")
    if "left_finger" in hand_kinematics_link:
        kl = hand_kinematics_link["left_finger"]["kinematic"]
        visual = ET.SubElement(lf_link, "visual")
        visual.set("name", f"{ee_prefix}left_finger_visual")
        visual.append(_origin(kl["x"], kl["y"], kl["z"], kl.get("roll", 0), kl.get("pitch", 0), kl.get("yaw", 0)))
        geom = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geom, "mesh")
        mesh.set("filename", "meshes/ee/openarm_hand/collision/finger.stl")
        mesh.set("scale", "0.001 0.001 0.001")

    lf_joint = ET.SubElement(robot, "joint")
    lf_joint.set("name", f"{ee_prefix}finger_joint2")
    lf_joint.set("type", "prismatic")
    pj = ET.SubElement(lf_joint, "parent")
    pj.set("link", f"{ee_prefix}hand")
    cj = ET.SubElement(lf_joint, "child")
    cj.set("link", f"{ee_prefix}left_finger")
    lf_joint.append(_origin(0, 0.006, 0.015, 0, 0, 0))
    axis = ET.SubElement(lf_joint, "axis")
    axis.set("xyz", "0 1 0")
    limit = ET.SubElement(lf_joint, "limit")
    limit.set("effort", "333")
    limit.set("lower", "0.0")
    limit.set("upper", "0.044")
    limit.set("velocity", "10.0")
    mimic = ET.SubElement(lf_joint, "mimic")
    mimic.set("joint", f"{ee_prefix}finger_joint1")

    # --- Right finger ---
    rf_link = ET.SubElement(robot, "link")
    rf_link.set("name", f"{ee_prefix}right_finger")
    if "right_finger" in hand_kinematics_link:
        kl = hand_kinematics_link["right_finger"]["kinematic"]
        visual = ET.SubElement(rf_link, "visual")
        visual.set("name", f"{ee_prefix}right_finger_visual")
        visual.append(_origin(kl["x"], kl["y"], kl["z"], kl.get("roll", 0), kl.get("pitch", 0), kl.get("yaw", 0)))
        geom = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geom, "mesh")
        mesh.set("filename", "meshes/ee/openarm_hand/collision/finger.stl")
        mesh.set("scale", "0.001 -0.001 0.001")

    rf_joint = ET.SubElement(robot, "joint")
    rf_joint.set("name", f"{ee_prefix}finger_joint1")
    rf_joint.set("type", "prismatic")
    pj = ET.SubElement(rf_joint, "parent")
    pj.set("link", f"{ee_prefix}hand")
    cj = ET.SubElement(rf_joint, "child")
    cj.set("link", f"{ee_prefix}right_finger")
    rf_joint.append(_origin(0, -0.006, 0.015, 0, 0, 0))
    axis = ET.SubElement(rf_joint, "axis")
    axis.set("xyz", "0 -1 0")
    limit = ET.SubElement(rf_joint, "limit")
    limit.set("effort", "333")
    limit.set("lower", "0.0")
    limit.set("upper", "0.044")
    limit.set("velocity", "10.0")


def build_urdf_manual() -> str:
    """Build the full bimanual OpenArm URDF from YAML configs."""
    robot = ET.Element("robot")
    robot.set("name", "openarm")
    robot.set("xmlns:xacro", "http://www.ros.org/wiki/xacro")

    # --- World link ---
    world_link = ET.SubElement(robot, "link")
    world_link.set("name", "world")

    # --- Body ---
    body_prefix = "openarm_body_"
    body_link = ET.SubElement(robot, "link")
    body_link.set("name", f"{body_prefix}link0")
    if "body_link0" in body_kinematics_link:
        kl = body_kinematics_link["body_link0"]["kinematic"]
        visual = ET.SubElement(body_link, "visual")
        visual.set("name", f"{body_prefix}body_link0_visual")
        visual.append(_origin(kl["x"], kl["y"], kl["z"], kl.get("roll", 0), kl.get("pitch", 0), kl.get("yaw", 0)))
        geom = ET.SubElement(visual, "geometry")
        mesh = ET.SubElement(geom, "mesh")
        mesh.set("filename", "meshes/body/v10/collision/body_link0_symp.stl")
        mesh.set("scale", "0.001 0.001 0.001")

    # Body ↔ World fixed joint
    body_joint = ET.SubElement(robot, "joint")
    body_joint.set("name", f"{body_prefix}world_joint")
    body_joint.set("type", "fixed")
    pj = ET.SubElement(body_joint, "parent")
    pj.set("link", "world")
    cj = ET.SubElement(body_joint, "child")
    cj.set("link", f"{body_prefix}link0")
    body_joint.append(_origin(0, 0, 0, 0, 0, 0))

    # --- Left arm ---
    _build_arm(robot, "left_", f"{body_prefix}link0",
               "0.0 0.031 0.698", "-1.5708 0 0")

    # --- Right arm ---
    _build_arm(robot, "right_", f"{body_prefix}link0",
               "0.0 -0.031 0.698", "1.5708 0 0")

    # --- Left hand ---
    _build_hand(robot, "left_", "openarm_left_link7")

    # --- Right hand ---
    _build_hand(robot, "right_", "openarm_right_link7")

    # Pretty-print
    ET.indent(robot, space="  ")
    xml_str = ET.tostring(robot, encoding="unicode", xml_declaration=False)
    return '<?xml version="1.0" ?>\n' + xml_str + "\n"


# --------------------------------------------------------------------------- #
#                          CONFIG / ANIMATIONS                                 #
# --------------------------------------------------------------------------- #

def build_config() -> dict:
    """Build config.json for the VILMA 3D viewer."""
    # Home pose must be within joint limits AND produce a natural arm posture.
    # Found via FK search: arm in front, to the side, below shoulder.
    # EE ≈ (0.37, 0.12, 0.37) — a natural "ready" position.
    home_pose = {
        "openarm_left_joint1": -1.7277,   # ~-99° — arm pointing down/forward
        "openarm_left_joint2": 0.1,       # ~6° — slight shoulder offset
        "openarm_left_joint3": 0.0,
        "openarm_left_joint4": -1.32,     # ~-76° — elbow bent
        "openarm_left_joint5": 0.0,
        "openarm_left_joint6": 0.0,
        "openarm_left_joint7": 0.0,
        "openarm_right_joint1": 1.7277,   # mirrored
        "openarm_right_joint2": -0.1,     # mirrored
        "openarm_right_joint3": 0.0,
        "openarm_right_joint4": 1.32,     # mirrored
        "openarm_right_joint5": 0.0,
        "openarm_right_joint6": 0.0,
        "openarm_right_joint7": 0.0,
        "openarm_left_finger_joint1": 0.0,
        "openarm_left_finger_joint2": 0.0,
        "openarm_right_finger_joint1": 0.0,
        "openarm_right_finger_joint2": 0.0,
    }

    link_colors = {
        "world": "0x444444",
        "openarm_body_link0": "0x888888",
        # Left arm
        "openarm_left_link0": "0x333333",
        "openarm_left_link1": "0x4488FF",
        "openarm_left_link2": "0x4488FF",
        "openarm_left_link3": "0x4488FF",
        "openarm_left_link4": "0x4488FF",
        "openarm_left_link5": "0x4488FF",
        "openarm_left_link6": "0x4488FF",
        "openarm_left_link7": "0x4488FF",
        "openarm_left_hand": "0x22CCFF",
        "openarm_left_left_finger": "0x22CCFF",
        "openarm_left_right_finger": "0x22CCFF",
        # Right arm
        "openarm_right_link0": "0x333333",
        "openarm_right_link1": "0xFF8844",
        "openarm_right_link2": "0xFF8844",
        "openarm_right_link3": "0xFF8844",
        "openarm_right_link4": "0xFF8844",
        "openarm_right_link5": "0xFF8844",
        "openarm_right_link6": "0xFF8844",
        "openarm_right_link7": "0xFF8844",
        "openarm_right_hand": "0xFFCC22",
        "openarm_right_left_finger": "0xFFCC22",
        "openarm_right_right_finger": "0xFFCC22",
    }

    return {
        "home_pose": home_pose,
        "link_colors": link_colors,
        "rotation": [270, 0, 0],
        "position": [0, 0, 0],
        "scale": [1, 1, 1],
        "dmp_scale": [1.0, 1.0, 1.0],
        "dmp_offset": [0.3, 0.0, 0.35],
        "dmp_rotation_z": 0.0,
    }


def build_animations() -> dict:
    return {"animations": {}}


# --------------------------------------------------------------------------- #
#                               MAIN                                           #
# --------------------------------------------------------------------------- #

def main():
    print("=" * 60)
    print("  OpenArm → VILMA URDF Converter (bimanual + grippers)")
    print("=" * 60)

    # 1. Generate URDF
    print("\n[1/4] Generating URDF ...")
    urdf_str = try_xacro()
    if urdf_str is None:
        print("       Using manual URDF generation from YAML configs.")
        urdf_str = build_urdf_manual()
    print(f"       URDF size: {len(urdf_str):,} bytes")

    # 2. Create output directory
    print(f"\n[2/4] Creating output directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Write URDF
    urdf_path = OUT_DIR / "openarm.urdf"
    urdf_path.write_text(urdf_str, encoding="utf-8")
    print(f"       Written: {urdf_path}")

    # 4. Copy meshes
    print("\n[3/4] Copying meshes ...")
    dst_meshes = OUT_DIR / "meshes"
    if dst_meshes.exists():
        shutil.rmtree(dst_meshes)
    shutil.copytree(MESH_SRC, dst_meshes)
    # Count mesh files
    mesh_count = sum(1 for _ in dst_meshes.rglob("*") if _.is_file())
    print(f"       Copied {mesh_count} mesh files.")

    # 5. Write config.json
    print("\n[4/4] Writing config.json and animations.json ...")
    config_path = OUT_DIR / "config.json"
    config_path.write_text(json.dumps(build_config(), indent=2), encoding="utf-8")
    print(f"       Written: {config_path}")

    anim_path = OUT_DIR / "animations.json"
    anim_path.write_text(json.dumps(build_animations(), indent=2), encoding="utf-8")
    print(f"       Written: {anim_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  DONE! OpenArm model ready at:")
    print(f"    {OUT_DIR}")
    print()
    print("  Contents:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            print(f"    {f.name}  ({f.stat().st_size:,} bytes)")
        else:
            sub_count = sum(1 for _ in f.rglob("*") if _.is_file())
            print(f"    {f.name}/  ({sub_count} files)")
    print("=" * 60)


if __name__ == "__main__":
    main()
