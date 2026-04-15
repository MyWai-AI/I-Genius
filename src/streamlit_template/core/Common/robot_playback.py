# vilma_core/robot_playback.py
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from functools import lru_cache
import numpy as np
import csv
import xml.etree.ElementTree as ET
import shutil
import re

# NumPy 2.0 compatibility: provide deprecated aliases if missing
try:  # pragma: no cover
    _ = np.float  # type: ignore[attr-defined]
except AttributeError:  # NumPy >= 2.0
    np.float = float      # type: ignore[attr-defined]
    np.int = int          # type: ignore[attr-defined]
    np.complex = complex  # type: ignore[attr-defined]
    np.bool = bool        # type: ignore[attr-defined]
    np.object = object    # type: ignore[attr-defined]
    np.str = str          # type: ignore[attr-defined]

# Utilities

def _ensure_array(a, shape_last: Optional[int] = None):
    arr = np.asarray(a, dtype=float)
    if shape_last is not None and arr.shape[-1] != shape_last:
        raise ValueError(f"Expected last dim {shape_last}, got {arr.shape}")
    return arr


def _sanitize_cartesian_path(path: np.ndarray) -> np.ndarray:
    """Drop non-finite points and require at least one valid 3D point."""
    arr = _ensure_array(path, shape_last=3)
    if arr.ndim != 2:
        arr = np.reshape(arr, (-1, 3))
    finite_mask = np.all(np.isfinite(arr), axis=1)
    cleaned = arr[finite_mask]
    if cleaned.shape[0] == 0:
        raise ValueError("Cartesian path has no finite points after sanitization.")
    if cleaned.shape[0] < arr.shape[0]:
        dropped = int(arr.shape[0] - cleaned.shape[0])
        print(f"[WARN] Dropped {dropped} non-finite Cartesian points before IK.")
    return cleaned


def _sanitize_initial_position(initial: np.ndarray, num_links: int) -> np.ndarray:
    """Ensure IK seed has correct size and finite values."""
    q = np.asarray(initial, dtype=float).reshape(-1)
    if q.shape[0] != num_links:
        print(f"[WARN] IK seed length {q.shape[0]} != chain links {num_links}; using zeros.")
        q = np.zeros(num_links, dtype=float)
    if not np.all(np.isfinite(q)):
        print("[WARN] IK seed contains non-finite values; replacing with zeros.")
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    return q


def _normalize_joint_name(name: str) -> str:
    """Normalize joint names so uploaded robot configs can match URDF variants."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _joint_name_aliases(name: str) -> List[str]:
    """Generate stable aliases for flexible joint-name matching."""
    norm = _normalize_joint_name(name)
    if not norm:
        return []

    aliases = {norm}

    joint_pos = norm.rfind("joint")
    if joint_pos >= 0:
        aliases.add(norm[joint_pos:])

    match = re.search(r"(\d+)$", norm)
    if match:
        aliases.add(f"joint{match.group(1)}")

    return [alias for alias in aliases if alias]


def _clip_seed_to_chain_bounds(seed: np.ndarray, chain) -> np.ndarray:
    """Clip a seed to the chain bounds and zero inactive/fixed joints."""
    q = _sanitize_initial_position(seed, len(chain.links)).astype(float, copy=True)

    active_mask = getattr(chain, "active_links_mask", None)
    if active_mask is not None and len(active_mask) == len(q):
        q[np.logical_not(np.asarray(active_mask, dtype=bool))] = 0.0

    for i, link in enumerate(chain.links):
        bounds = getattr(link, "bounds", None)
        if not bounds or len(bounds) != 2:
            continue

        lower, upper = bounds
        lower = float(lower) if lower is not None else -np.inf
        upper = float(upper) if upper is not None else np.inf
        if lower > upper:
            lower, upper = upper, lower

        if np.isfinite(lower):
            q[i] = max(q[i], lower)
        if np.isfinite(upper):
            q[i] = min(q[i], upper)

    if not np.all(np.isfinite(q)):
        print("[WARN] IK seed became non-finite after clipping; replacing invalid entries with 0.0.")
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    return q


def _seed_from_chain_bounds(chain) -> np.ndarray:
    """Create a neutral seed from joint bounds when no good pose is available."""
    q = np.zeros(len(chain.links), dtype=float)
    for i, link in enumerate(chain.links):
        bounds = getattr(link, "bounds", None)
        if not bounds or len(bounds) != 2:
            continue

        lower, upper = bounds
        lower = float(lower) if lower is not None else -np.inf
        upper = float(upper) if upper is not None else np.inf
        if lower > upper:
            lower, upper = upper, lower

        if np.isfinite(lower) and np.isfinite(upper):
            q[i] = 0.5 * (lower + upper)
        elif np.isfinite(lower):
            q[i] = max(lower, 0.0)
        elif np.isfinite(upper):
            q[i] = min(upper, 0.0)

    return _clip_seed_to_chain_bounds(q, chain)


def _solve_ik_with_fallbacks(
    chain,
    target_position: np.ndarray,
    preferred_seed: np.ndarray,
    home_seed: Optional[np.ndarray] = None,
    target_orientation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Try multiple seed/orientation combinations before giving up on a point."""
    attempts: List[str] = []
    candidates: List[Tuple[str, np.ndarray]] = []

    for seed_name, seed in [
        ("previous", preferred_seed),
        ("home", home_seed),
        ("midpoint", _seed_from_chain_bounds(chain)),
        ("zero", np.zeros(len(chain.links), dtype=float)),
    ]:
        if seed is None:
            continue

        clipped = _clip_seed_to_chain_bounds(seed, chain)
        if any(np.allclose(clipped, prev, atol=1e-6, rtol=0.0) for _, prev in candidates):
            continue
        candidates.append((seed_name, clipped))

    solve_modes: List[Tuple[str, Dict[str, Any]]] = []
    if target_orientation is not None:
        solve_modes.append((
            "all",
            {"target_orientation": target_orientation, "orientation_mode": "all"},
        ))
    solve_modes.append(("position", {}))

    last_error: Optional[Exception] = None
    for mode_name, extra_kwargs in solve_modes:
        for seed_name, seed in candidates:
            try:
                q = chain.inverse_kinematics(
                    target_position=target_position,
                    initial_position=seed,
                    **extra_kwargs,
                )
                q = _clip_seed_to_chain_bounds(q, chain)
                if not np.all(np.isfinite(q)):
                    raise ValueError("IK returned non-finite joint values.")
                return q, {"mode": mode_name, "seed": seed_name}
            except Exception as e:
                last_error = e
                attempts.append(f"{mode_name}/{seed_name}: {type(e).__name__}: {e}")

    target_list = np.asarray(target_position, dtype=float).round(6).tolist()
    attempt_summary = "; ".join(attempts[:6])
    if len(attempts) > 6:
        attempt_summary += f"; ... {len(attempts) - 6} more attempts"
    raise ValueError(
        f"IK failed for target {target_list} after trying fallback seeds and modes. "
        f"Last error: {type(last_error).__name__ if last_error else 'UnknownError'}: "
        f"{last_error if last_error else 'unknown'}. Attempts: {attempt_summary}"
    )


# IK / Path mapping

def _sanitize_urdf_file(src_path: Path) -> Path:
    """
    Create a sanitized copy of the URDF by resolving all mesh filenames to absolute paths.
    This ensures that external libraries like urdfpy and ikpy can find meshes regardless
    of where they are.
    """
    try:
        tree = ET.parse(str(src_path))
        root = tree.getroot()
        changed = False

        # 1. Resolve mesh filenames to absolute paths
        for mesh in root.iter('mesh'):
            filename = mesh.get('filename')
            if filename:
                abs_path = _resolve_mesh_path(filename, src_path)
                if abs_path.exists():
                    mesh.set('filename', abs_path.resolve().as_posix())
                    changed = True

        # 2. Strip problematic elements that crash urdfpy or are irrelevant to visualization/IK
        # Generic approach: Remove transmission, gazebo, and controller-specific tags.
        # urdfpy is known to crash on these if they don't match its internal schema exactly.
        tags_to_strip = ['transmission', 'gazebo', 'ros_control']
        for tag in tags_to_strip:
            # We use list(root.findall(...)) because we are modifying the tree during iteration
            for elem in list(root.iter(tag)):
                # Handle nested matches or multiple instances
                # Many parsers might have trouble if we remove 'gazebo' while it has children
                # ET handle this fine by removing the whole subtree.
                parent = None
                # We need to find the parent to remove the element in ET
                # In modern Python 3.9+ we could use root.findall('.//' + tag + '/..')
                # For compatibility, we'll do a simple parent map or iterative removal
                pass

        # Robust removal using parent map
        parent_map = {c: p for p in root.iter() for c in p}
        for tag in tags_to_strip:
            for elem in list(root.iter(tag)):
                if elem in parent_map:
                    parent_map[elem].remove(elem)
                    changed = True

        out_path = src_path.parent / f"_sanitized_{src_path.name}"
        sanitized_xml = ET.tostring(root, encoding='unicode')
        
        if not out_path.exists() or out_path.read_text(encoding="utf-8", errors="ignore") != sanitized_xml:
            out_path.write_text(sanitized_xml, encoding="utf-8")
        return out_path
    except Exception as e:
        print(f"[WARN] URDF Sanitization failed for {src_path}: {e}")
        return src_path


def _detect_root_link(urdf_path: Path) -> Optional[str]:
    """Return the name of the root link in a URDF (the link that is never a child of any joint).

    Falls back to ``"base_link"`` so that existing robots that rely on the ikpy
    default keep working.
    """
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()

        all_links = {l.get("name") for l in root.findall("link") if l.get("name")}
        child_links = set()
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                child_links.add(child.get("link"))

        root_links = all_links - child_links
        if root_links:
            # Prefer "base_link" if present (backward compat), otherwise pick the single root
            if "base_link" in root_links:
                return "base_link"
            return sorted(root_links)[0]
    except Exception as e:
        print(f"[WARN] _detect_root_link failed: {e}")

    return "base_link"  # fall back to ikpy default


# @lru_cache(maxsize=4) # Cache removed to ensure fresh loading during debug
def load_chain(urdf_path: str):
    """Load ikpy kinematic chain from URDF."""
    try:
        from ikpy.chain import Chain
    except Exception as e:
        raise ImportError("ikpy not installed. Run: pip install ikpy") from e

    p = Path(urdf_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"URDF not found: {p}")

    # Sanitize package:// URIs for parsers that don't support the scheme
    p_sanitized = _sanitize_urdf_file(p)

    print(f"[DEBUG] load_chain loading: {p_sanitized}")

    # Auto-detect the root link of the URDF so we don't rely on the ikpy
    # default of "base_link" (which does not exist in every robot model).
    root_link_name = _detect_root_link(p_sanitized)
    base_elems = [root_link_name] if root_link_name else None
    print(f"[DEBUG] Detected root link: {root_link_name}")

    # Load chain first -> ALL joints active by default
    chain = Chain.from_urdf_file(str(p_sanitized), base_elements=base_elems, active_links_mask=None)

    # Use ElementTree to robustly find fixed joints (avoiding urdfpy dependency issues)
    # The log shows chain.links names are likely JOINT names (e.g., 'base_joint', 'jaco_joint_1')
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(p_sanitized))
        root = tree.getroot()
        
        fixed_joints = set()
        for joint in root.findall(".//joint"):
            j_name = joint.get("name")
            j_type = joint.get("type")
            if j_type == "fixed":
                fixed_joints.add(j_name)
        
        print(f"[DEBUG] Found fixed joints in URDF: {fixed_joints}")

        mask = []
        for lk in chain.links:
            # Logic: If the link name matches a fixed joint name -> Inactive
            # Also, commonly the first link is "Base link" (link name) which has no joint -> Inactive
            
            is_active = True
            
            # Check by name
            if lk.name in fixed_joints:
                is_active = False
            
            # Heuristic: 'Base' in name often implies static base
            if "Base" in lk.name and lk.name not in fixed_joints: 
                 # Be careful not to disable "Base Rotation Joint" if it were active
                 # But usually index 0 is static
                 pass

            # Explicit check for the known problematic joint
            if lk.name == "base_joint":
                is_active = False

            mask.append(is_active)
        
        # Force the very first link to be inactive (usually the phantom base link)
        if len(mask) > 0:
            mask[0] = False

        print(f"[DEBUG] Generated Mask: {mask}")
        print(f"[DEBUG] Chain Link Names: {[l.name for l in chain.links]}")

        # Re-load chain with the calculated mask
        chain = Chain.from_urdf_file(str(p_sanitized), base_elements=base_elems, active_links_mask=mask)
        return chain

    except Exception as e:
        print(f"[ERROR] Failed to mask fixed joints: {e}")
        return chain

def get_configured_joint_names(urdf_path: str) -> List[str]:
    """
    Returns the list of JOINT names corresponding to the loaded IK chain links.
    The viewer expects joint names (to drive rotation), but ikpy chain.links usually 
    exposes the link name. We must map Link -> Parent Joint.
    """
    chain = load_chain(urdf_path)
    
    # 1. Parse URDF to build ChildLink -> ParentJoint map
    # We use ElementTree for robustness (urdfpy might fail/be stripped)
    try:
        p_sanitized = _sanitize_urdf_file(Path(urdf_path))
        tree = ET.parse(str(p_sanitized))
        root = tree.getroot()
        
        link_to_joint = {}
        for joint in root.findall(".//joint"):
            j_name = joint.get("name")
            child = joint.find("child")
            if child is not None:
                c_link = child.get("link")
                link_to_joint[c_link] = j_name
                
        # 2. Iterate chain links and find corresponding joint
        joint_names = []
        for i, link in enumerate(chain.links):
            # link.name is the name of the link provided in the chain
            # Note: The first link is usually the base link, which might not have a parent joint 
            # (or connects to world).
            
            j_name = link_to_joint.get(link.name)
            if j_name:
                joint_names.append(j_name)
            elif link.name in link_to_joint.values():
                 # Fallback: The chain link name IS the joint name (common in ikpy)
                 joint_names.append(link.name)
            else:
                # If no parent joint found, it might be the base link.
                # However, the viewer usually ignores the base link joint anyway if it's fixed.
                # Use a placeholder or the link name itself to be safe?
                # Using link name as fallback might be confusing but better than crashing.
                # Common pattern: Base link is just a frame.
                joint_names.append(link.name + "_fixed_base_phantom")
                
        return joint_names

    except Exception as e:
        print(f"[ERROR] Failed to map links to joints: {e}")
        # Fallback: return link names and hope for the best (or matches default robot)
        return [l.name for l in chain.links]


def dmp_to_cartesian(
    dmp_npy: str,
    scale_xy_m: Tuple[float, float] = (0.4, 0.4),
    offset_xy_m: Tuple[float, float] = (0.2, 0.0),
    z_height_m: float = 0.25,
    flip_y: bool = True
) -> Dict[str, Any]:
    """Map normalized DMP (x,y) to world (X,Y,Z) in meters; Z fixed for now."""
    Yg = np.load(str(dmp_npy))  # (N,2)
    Yg = _ensure_array(Yg, shape_last=2)

    xs = offset_xy_m[0] + Yg[:, 0] * scale_xy_m[0]
    ys_raw = Yg[:, 1]
    ys = offset_xy_m[1] + (1.0 - ys_raw if flip_y else ys_raw) * scale_xy_m[1]
    zs = np.full_like(xs, float(z_height_m))
    return {"cartesian_path": np.stack([xs, ys, zs], axis=1)}


def dmp_xyz_to_cartesian(
    dmp_npy: str,
    scale_xyz=(0.5, 0.5, 0.4),
    offset_xyz=(0.3, 0.0, 0.15),
    flip_y=False,
    flip_z=False,
    rotate_z=0.0,  # Degrees
    add_arch=False,
    arch_height=0.2,
    target_frames=100,
    arm_reach=0.0,
):
    """Map XYZ DMP → robot-base Cartesian coordinates (meters).

    Pipeline:
        1. Zero-centre (subtract first point)
        2. Flip axes (camera → robot convention)
        3. Auto-scale so trajectory fits ``arm_reach * 0.6`` (when *arm_reach > 0*)
           ``scale_xyz`` then acts as a **post-multiplier** for fine-tuning.
           When *arm_reach ≤ 0*, ``scale_xyz`` is applied directly (legacy).
        4. Rotate around Z
        5. Translate by ``offset_xyz`` (defines trajectory start in robot frame)
        6. (Optional) add synthetic arch
        7. Smooth Z via Savitzky–Golay
        8. Resample to ``target_frames``

    Parameters
    ----------
    flip_z : bool
        Negate the Z component *after* centring.  Use for overhead cameras
        where decreasing camera-Z (depth) means the hand is lifting **up** in
        robot space.
    arm_reach : float
        Approximate reach of the robot arm (metres).  When > 0, enables
        auto-scaling: the trajectory is uniformly shrunk so that no axis
        exceeds ``arm_reach * 0.6`` (leaving margin for offset placement).
        ``scale_xyz`` is then applied on top as fractional multipliers
        (1.0 = fill workspace, 0.5 = half).
    """
    Yg = np.load(dmp_npy)          # (N,3)
    Yg = _ensure_array(Yg, shape_last=3)
    if Yg.ndim != 2 or Yg.shape[0] == 0:
        raise ValueError(f"Invalid XYZ DMP shape: {Yg.shape}")

    # Repair non-finite samples so downstream IK doesn't receive NaN/Inf.
    if not np.all(np.isfinite(Yg)):
        Yg = Yg.copy()
        for j in range(3):
            col = Yg[:, j]
            finite = np.isfinite(col)
            if not np.any(finite):
                # Entire column invalid: keep motion neutral on this axis.
                col[:] = 0.0
                print(f"[WARN] DMP axis {j} had no finite values; filled with 0.0.")
            elif not np.all(finite):
                idx = np.arange(len(col))
                col[~finite] = np.interp(idx[~finite], idx[finite], col[finite])
                print(f"[WARN] DMP axis {j} had non-finite values; interpolated missing samples.")
            Yg[:, j] = col

    # 1. Zero-centre relative to the first point
    origin = Yg[0].copy()
    centered = Yg - origin  # (N, 3) — starts at (0,0,0)

    # 2. Axis flips (camera → robot convention)
    if flip_y:
        centered[:, 1] = -centered[:, 1]
    if flip_z:
        centered[:, 2] = -centered[:, 2]

    # 3. Scaling
    if arm_reach > 0:
        # Auto-scale: compute max absolute displacement per axis
        max_disp = np.max(np.abs(centered), axis=0)  # (3,)
        max_disp = np.where(max_disp < 1e-6, 1e-6, max_disp)  # avoid /0

        workspace = arm_reach * 0.6  # use 60 % of reach as safe zone
        # Uniform scale factor to preserve shape proportions
        auto_s = workspace / np.max(max_disp)
        auto_s = min(auto_s, 5.0)  # cap: don't inflate truly tiny motions

        # Apply: auto-scale * user multiplier
        xs = centered[:, 0] * auto_s * scale_xyz[0]
        ys = centered[:, 1] * auto_s * scale_xyz[1]
        zs = centered[:, 2] * auto_s * scale_xyz[2]

        print(f"[DEBUG] Auto-scale: max_disp={max_disp}, "
              f"workspace={workspace:.3f}, auto_s={auto_s:.4f}, "
              f"user_scale={scale_xyz}")
    else:
        # Legacy: scale_xyz applied directly
        xs = centered[:, 0] * scale_xyz[0]
        ys = centered[:, 1] * scale_xyz[1]
        zs = centered[:, 2] * scale_xyz[2]

    # 4. Rotate around Z
    if rotate_z != 0.0:
        rad = np.deg2rad(rotate_z)
        c, s = np.cos(rad), np.sin(rad)
        xs_rot = xs * c - ys * s
        ys_rot = xs * s + ys * c
        xs, ys = xs_rot, ys_rot

    # 5. Translate (offset defines START position in robot frame)
    x = offset_xyz[0] + xs
    y = offset_xyz[1] + ys
    z = offset_xyz[2] + zs

    # 6. Synthetic arch (Z-lift)
    if add_arch:
        N = len(z)
        t = np.linspace(0, 1, N)
        z_lift = 4 * arch_height * t * (1 - t)
        z += z_lift

    # 7. Smooth Z to avoid vertical jitter
    try:
        from scipy.signal import savgol_filter
        if len(z) >= 11:
            z = savgol_filter(z, window_length=11, polyorder=3)
    except Exception:
        pass

    path = np.stack([x, y, z], axis=1)

    # 8. Resample to target_frames
    if target_frames is None:
        target_frames = 100

    if len(path) != target_frames:
        try:
            from scipy.interpolate import interp1d
            N = len(path)
            t_old = np.linspace(0, 1, N)
            t_new = np.linspace(0, 1, target_frames)
            f = interp1d(t_old, path, axis=0, kind='linear')
            path = f(t_new)
            print(f"[DEBUG] Resampled trajectory from {N} to {target_frames} points.")
        except ImportError:
            print("[WARN] scipy.interpolate not found, skipping upsampling.")

    return {
        "cartesian_path": path
    }



def get_robot_home_position(urdf_path: str) -> np.ndarray:
    """
    Return a reasonable home/ready position for the robot.
    This puts the robot in a comfortable starting pose.
    """
    chain = load_chain(str(Path(urdf_path).resolve()))
    num_joints = len(chain.links)
    
    q_home = np.zeros(num_joints)
    
    # Try to load home pose from config.json (sibling of URDF)
    config_path = Path(urdf_path).parent / "config.json"
    if config_path.exists():
        try:
            import json
            with config_path.open("r") as f:
                config = json.load(f)
            
            home_pose = config.get("home_pose", {})

            chain_joint_names = get_configured_joint_names(urdf_path)
            home_pose_lookup: Dict[str, float] = {}
            for cfg_name, cfg_value in home_pose.items():
                try:
                    numeric_value = float(cfg_value)
                except Exception:
                    continue
                for alias in _joint_name_aliases(cfg_name):
                    home_pose_lookup.setdefault(alias, numeric_value)

            matched = 0
            for i, j_name in enumerate(chain_joint_names):
                if i < num_joints:
                    for alias in _joint_name_aliases(j_name):
                        if alias in home_pose_lookup:
                            q_home[i] = home_pose_lookup[alias]
                            matched += 1
                            break

            print(f"[DEBUG] Loaded Home Pose from config ({matched} matches): {q_home}")
            return q_home
            
        except Exception as e:
            print(f"[WARN] Failed to load home pose from config: {e}")

    # Fallback to zeros (Safe Neutral)
    print("[DEBUG] Using default zero home pose.")
    return q_home

def compute_ik_trajectory(
    urdf_path: str,
    cartesian_path: np.ndarray,
    target_frame: Optional[int] = None,
    initial_position: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Solve IK for each 3D point; orientation unconstrained; returns (N,J) joints."""
    cartesian_path = _sanitize_cartesian_path(cartesian_path)
    chain = load_chain(str(Path(urdf_path).resolve()))
    
    # Use home position if no initial position provided
    if initial_position is None:
        initial_position = get_robot_home_position(urdf_path)
    initial_position = _clip_seed_to_chain_bounds(initial_position, chain)

    # === DEBUG: Print diagnostic info ===
    print("\n" + "="*60)
    print("[DEBUG] compute_ik_trajectory - Diagnostic Info")
    print("="*60)
    print(f"  URDF path: {urdf_path}")
    print(f"  Cartesian path shape: {cartesian_path.shape}")
    print(f"  Cartesian path dtype: {cartesian_path.dtype}")
    print(f"  Cartesian path range:")
    print(f"    X: min={cartesian_path[:, 0].min():.4f}, max={cartesian_path[:, 0].max():.4f}")
    print(f"    Y: min={cartesian_path[:, 1].min():.4f}, max={cartesian_path[:, 1].max():.4f}")
    print(f"    Z: min={cartesian_path[:, 2].min():.4f}, max={cartesian_path[:, 2].max():.4f}")
    print(f"  Contains NaN: {np.any(np.isnan(cartesian_path))}")
    print(f"  Contains Inf: {np.any(np.isinf(cartesian_path))}")
    print(f"  First point: {cartesian_path[0]}")
    print(f"  Last point:  {cartesian_path[-1]}")
    print(f"  Initial position: {initial_position}")
    print(f"  Initial position contains NaN: {np.any(np.isnan(initial_position))}")
    print(f"  Initial position contains Inf: {np.any(np.isinf(initial_position))}")
    print(f"  Chain links: {len(chain.links)}")
    print(f"  Chain link names: {[lk.name for lk in chain.links]}")
    print("-" * 60)
    print("  Full Cartesian Path (Points):")
    for i, pt in enumerate(cartesian_path):
        print(f"    Pt[{i}]: {pt}")
    print("="*60 + "\n")

    # Define a target orientation (Gripper pointing down)
    # Revert to Rot X 180 (Standard Down). Identity pointed Up.
    img_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    qs: List[np.ndarray] = []
    
    # Force "Elbow Up" configuration by biasing the seed
    # Current solution flips J3 to -120 deg (Underarm). We want Overarm.
    # Set J3 (index 2 usually) to positive.
    # We assume standard Comau chain: Base, J1, J2, J3... 
    # J3 is index 2? Chain link names were: Base Link, J1, J2...
    # So J3 corresponds to index 3 in q? (0=Base, 1=J1, 2=J2, 3=J3).
    # Wait, q has shape 9. 0=Base. 1=J1. 2=J2. 3=J3.
    # So we want q[3] > 0.
    
    q_home = np.array(initial_position, dtype=float, copy=True)
    q_prev = np.array(q_home, dtype=float, copy=True) if initial_position is not None else _seed_from_chain_bounds(chain)
    # Bias removed: Rely on initial_position (Home Pose from config) to seed the IK.
    # if len(q_prev) > 4: ...
    
    print(f"[DEBUG] Using seed from Home Pose: {q_prev}")
    
    for idx, p in enumerate(cartesian_path):
        # === DEBUG: Check for NaN/Inf before IK ===
        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            print(f"[DEBUG] ERROR: Point {idx} contains NaN/Inf: {p}")
        if np.any(np.isnan(q_prev)) or np.any(np.isinf(q_prev)):
            print(f"[DEBUG] ERROR: q_prev at point {idx} contains NaN/Inf: {q_prev}")
        
        try:
            # 1. Standard solve with continuity, but fall back if the initial residual is non-finite.
            q_cont, solve_meta = _solve_ik_with_fallbacks(
                chain=chain,
                target_position=p,
                preferred_seed=q_prev,
                home_seed=q_home,
                target_orientation=img_down,
            )
            if solve_meta["mode"] != "all" or solve_meta["seed"] != "previous":
                print(
                    f"[WARN] Frame {idx}: IK fallback used "
                    f"{solve_meta['mode']} mode with {solve_meta['seed']} seed."
                )
            
            # 2. Check for drift from Home Pose (Preferred Style)
            # If solution drifts too far from Home (e.g. Flipped Elbow), re-try seeding from Home.
            dist_drift = np.linalg.norm(q_cont - q_home)
            
            q_final = q_cont
            
            # Threshold: > 1.5 rad deviation suggests potential flip
            if dist_drift > 1.5:
                # Try re-solving from Home Pose
                q_style, style_meta = _solve_ik_with_fallbacks(
                    chain=chain,
                    target_position=p,
                    preferred_seed=q_home,
                    home_seed=q_home,
                    target_orientation=img_down,
                )
                if style_meta["mode"] != "all":
                    print(
                        f"[WARN] Frame {idx}: style IK fallback used "
                        f"{style_meta['mode']} mode with {style_meta['seed']} seed."
                    )
                
                # Check if style solution is valid (via Forward Kinematics error)
                fk_cont = chain.forward_kinematics(q_cont)[:3, 3]
                fk_style = chain.forward_kinematics(q_style)[:3, 3]
                
                err_cont = np.linalg.norm(fk_cont - p)
                err_style = np.linalg.norm(fk_style - p)
                
                # If style solution is accurate enough (comparable to continuity sol), prefer it
                # We allow a tiny bit more error if it buys us better style, but mostly we want valid IK.
                if err_style < max(err_cont * 2.0, 0.01): 
                    # And purely better style?
                    dist_style = np.linalg.norm(q_style - q_home)
                    if dist_style < dist_drift:
                        q_final = q_style
                        # print(f"[DEBUG] Frame {idx}: Corrected pose drift ({dist_drift:.2f} -> {dist_style:.2f})")
            
            q = q_final
            
            # === DEBUG: Check IK result ===
            if np.any(np.isnan(q)) or np.any(np.isinf(q)):
                print(f"[DEBUG] WARNING: IK result at point {idx} contains NaN/Inf: {q}")
            
            qs.append(q)
            q_prev = q
            
        except Exception as e:
            print(f"[DEBUG] IK FAILED at point {idx}")
            print(f"  Target position: {p}")
            print(f"  Initial position (q_prev): {q_prev}")
            print(f"  Error: {type(e).__name__}: {e}")
            raise ValueError(
                f"IK failed at frame {idx} for target {np.asarray(p, dtype=float).round(6).tolist()}. "
                f"Check robot config home_pose and DMP scaling/offset settings. Root cause: {e}"
            ) from e

    Q = np.vstack(qs)

    # ------------------------------------------------------------------
    # Post-processing: smooth large joint-space jumps (elbow flips etc.)
    # ------------------------------------------------------------------
    try:
        from scipy.signal import savgol_filter

        JUMP_THRESH_RAD = np.deg2rad(8)  # per-joint, per-frame threshold (lowered from 12°)
        MAX_VEL_RAD = np.deg2rad(60)     # max joint velocity: 60 °/frame

        # --- Pass 1: Savitzky-Golay smoothing (wider window, applied twice) ---
        diffs = np.abs(np.diff(Q, axis=0))          # (N-1, J)
        needs_smooth = np.any(diffs > JUMP_THRESH_RAD, axis=0)  # per-joint mask

        if np.any(needs_smooth) and Q.shape[0] >= 11:
            # First pass: wide window
            win1 = min(41, Q.shape[0] if Q.shape[0] % 2 == 1 else Q.shape[0] - 1)
            for j in np.where(needs_smooth)[0]:
                Q[:, j] = savgol_filter(Q[:, j], window_length=win1, polyorder=3)
            # Second pass: narrower window to clean up residual bumps
            win2 = min(21, Q.shape[0] if Q.shape[0] % 2 == 1 else Q.shape[0] - 1)
            for j in np.where(needs_smooth)[0]:
                Q[:, j] = savgol_filter(Q[:, j], window_length=win2, polyorder=3)
            print(f"[DEBUG] IK smoothing applied to joints {list(np.where(needs_smooth)[0])} "
                  f"(2-pass, win={win1}/{win2})")

        # --- Pass 2: Velocity clamping (cap max deg/frame change) ---
        clamped_any = False
        for iteration in range(3):  # iterate to propagate clamping
            diffs = np.diff(Q, axis=0)
            exceeded = np.abs(diffs) > MAX_VEL_RAD
            if not np.any(exceeded):
                break
            clamped_any = True
            for i in range(diffs.shape[0]):
                for j in range(diffs.shape[1]):
                    if abs(diffs[i, j]) > MAX_VEL_RAD:
                        sign = 1.0 if diffs[i, j] > 0 else -1.0
                        Q[i + 1, j] = Q[i, j] + sign * MAX_VEL_RAD

        if clamped_any:
            # Final light smoothing pass over clamped trajectory
            win3 = min(11, Q.shape[0] if Q.shape[0] % 2 == 1 else Q.shape[0] - 1)
            if Q.shape[0] >= win3:
                for j in range(Q.shape[1]):
                    Q[:, j] = savgol_filter(Q[:, j], window_length=win3, polyorder=3)
            max_delta = np.rad2deg(np.abs(np.diff(Q, axis=0)).max())
            print(f"[DEBUG] Velocity clamping applied (max {np.rad2deg(MAX_VEL_RAD):.0f}°/frame). "
                  f"Post-clamp max delta: {max_delta:.2f}°")

    except ImportError:
        pass
    
    print("-" * 60)
    print("  Joint Trajectory Analysis (Degrees):")
    link_names = [lk.name for lk in chain.links]
    q_deg = np.rad2deg(Q)
    
    col_idx = 0
    for i, link in enumerate(chain.links):
        if col_idx < Q.shape[1]:
             mn, mx = q_deg[:, col_idx].min(), q_deg[:, col_idx].max()
             print(f"    Joint {col_idx} ({link.name}): Min={mn:.1f}°, Max={mx:.1f}° | Delta={mx-mn:.1f}°")
             col_idx += 1
             
    # Frame-to-frame quality
    diffs_deg = np.rad2deg(np.abs(np.diff(Q, axis=0)))
    max_per_frame = diffs_deg.max(axis=1)
    print(f"  Frame-to-frame: mean={max_per_frame.mean():.2f}° max={max_per_frame.max():.2f}° "
          f"p95={np.percentile(max_per_frame, 95):.2f}°")
        
    print(f"[DEBUG] IK trajectory complete: shape={Q.shape}")
    return {"q_traj": Q, "dofs": Q.shape[1], "links": len(chain.links)}


# ---------------------------------------------------------------------------
# Gripper trajectory generation
# ---------------------------------------------------------------------------

def build_gripper_trajectory(
    cart_path: np.ndarray,
    urdf_path: str,
    grasp_idx: Optional[int] = None,
    release_idx: Optional[int] = None,
    open_val: float = 0.044,
    closed_val: float = 0.0,
) -> List[Optional[Dict[str, float]]]:
    """Create per-frame gripper joint configs for the trajectory.

    The gripper starts **open**, closes at the *grasp point*, stays closed
    until the *release point*, then opens again.

    Parameters
    ----------
    cart_path : (N, 3) array
        Cartesian end-effector path.
    urdf_path : str
        URDF path — used to discover finger joint names automatically.
    grasp_idx : int, optional
        Frame index of the grasp event.  If *None* it is auto-detected as
        the frame with the lowest Z coordinate (closest to the table).
    release_idx : int, optional
        Frame index of the release event.  If *None* the gripper stays
        closed from grasp until the end of the trajectory.
    open_val : float
        Joint value for "fingers open" (max prismatic travel = 0.044 m).
    closed_val : float
        Joint value for "fingers closed".

    Returns
    -------
    list[dict | None]
        One dict per frame mapping finger-joint names → values.
    """
    cart_path = _ensure_array(cart_path, shape_last=3)
    N = len(cart_path)

    # Auto-detect grasp index: lowest Z point
    if grasp_idx is None:
        grasp_idx = int(np.argmin(cart_path[:, 2]))

    # Discover finger joint names from the URDF
    finger_joints: List[str] = []
    try:
        tree = ET.parse(str(Path(urdf_path).resolve()))
        root = tree.getroot()
        for joint in root.findall("joint"):
            jname = joint.get("name", "")
            jtype = joint.get("type", "")
            if "finger" in jname.lower() and jtype == "prismatic":
                finger_joints.append(jname)
    except Exception:
        pass

    # Fallback: use known OpenArm finger joint names
    if not finger_joints:
        finger_joints = [
            "openarm_left_finger_joint1",
            "openarm_left_finger_joint2",
        ]

    print(f"[DEBUG] Gripper trajectory: grasp_idx={grasp_idx}/{N}, "
          f"release_idx={release_idx}/{N}, "
          f"finger_joints={finger_joints}, open={open_val}, closed={closed_val}")

    trajectory: List[Optional[Dict[str, float]]] = []
    for i in range(N):
        if i < grasp_idx:
            val = open_val          # approaching — gripper open
        elif release_idx is not None and i >= release_idx:
            val = open_val          # released — gripper open again
        else:
            val = closed_val        # grasping / moving — gripper closed
        trajectory.append({jn: val for jn in finger_joints})

    return trajectory


def augment_traj_points_with_gripper(
    traj_points: List[Dict[str, Any]],
    cart_path: np.ndarray,
    urdf_path: str,
    grasp_idx: Optional[int] = None,
    release_idx: Optional[int] = None,
    open_val: float = 0.044,
    closed_val: float = 0.0,
) -> List[Dict[str, Any]]:
    """Add gripper finger joint values to sync-viewer trajectory points.

    Each ``traj_points`` item has ``{"time": float, "q": {name: val, ...}}``.
    This function detects the grasp event (or uses the provided index) and
    injects finger joint entries so that the 3D viewer opens/closes the
    gripper at the right moment, then re-opens at the release point.

    Parameters
    ----------
    traj_points : list[dict]
        Trajectory points as expected by the sync_viewer component.
    cart_path : (N, 3) array
        Cartesian path corresponding to the same frames.
    urdf_path : str
        Path to the URDF (used to discover finger joint names).
    grasp_idx, release_idx, open_val, closed_val :
        Same semantics as ``build_gripper_trajectory``.

    Returns
    -------
    list[dict]
        The *same* list, mutated in-place for convenience.
    """
    if not traj_points:
        return traj_points

    gripper_traj = build_gripper_trajectory(
        cart_path, urdf_path,
        grasp_idx=grasp_idx, release_idx=release_idx,
        open_val=open_val, closed_val=closed_val,
    )

    for i, pt in enumerate(traj_points):
        if i < len(gripper_traj) and gripper_traj[i]:
            pt["q"].update(gripper_traj[i])

    return traj_points


def joints_to_csv(q_traj: np.ndarray, out_csv: str, time_s: Optional[np.ndarray] = None) -> str:
    """Save joint trajectory to CSV with columns: TIME, q0..qJ-1"""
    q_traj = _ensure_array(q_traj)
    N, J = q_traj.shape
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if time_s is None:
        time_s = np.linspace(0, N/30.0, N)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["TIME"] + [f"q{j}" for j in range(J)])
        for i in range(N):
            w.writerow([f"{float(time_s[i]):.6f}"] + [f"{float(v):.6f}" for v in q_traj[i]])
    return str(out_path.resolve())

def joint_positions_for_frame(urdf_path: str, q: np.ndarray) -> np.ndarray:
    """
    Link polyline for quick wireframe viewer.
    Uses ikpy.forward_kinematics(full_kinematics=True).
    """
    q = _ensure_array(q)
    chain = load_chain(str(Path(urdf_path).resolve()))
    T_list = chain.forward_kinematics(q, full_kinematics=True)
    if isinstance(T_list, np.ndarray) and T_list.ndim == 3 and T_list.shape[1:] == (4, 4):
        transforms = [T_list[i] for i in range(T_list.shape[0])]
    else:
        transforms = list(T_list)
    pts = []
    if len(transforms) == 0 or not np.allclose(transforms[0], np.eye(4), atol=1e-9):
        pts.append(np.array([0., 0., 0.]))
    for T in transforms:
        pts.append(T[:3, 3])
    return np.vstack(pts)

# Mesh rendering (urdfpy + trimesh)

def _resolve_mesh_path(filename: str, urdf_path: Path) -> Path:
    """
    Robustly resolve mesh paths:
    - trims whitespace, normalizes slashes
    - detects 'package://' even if embedded in longer strings
    - tries <urdf_dir>/<pkg>/tail, <urdf_dir>/tail
    - auto-maps suffix variants: _support/_config/_description
    - scans sibling folders next to the URDF
    """
    f = str(filename).strip().replace("\\", "/")

    # detect package:// anywhere in the string
    idx = f.lower().find("package://")
    if idx >= 0:
        f2 = f[idx + len("package://"):]
        parts = f2.split("/", 1)
        pkg = parts[0] if parts else ""
        tail = parts[1] if len(parts) > 1 else ""

        candidates: List[Path] = []

        # 1) Standard: <urdf_dir>/<pkg>/tail
        candidates.append(urdf_path.parent / pkg / tail)

        # 2) Try suffix variants (_support, _config, _description, or none)
        variants = ["_support", "_config", "_description", ""]
        for suf in variants:
            if pkg.endswith(suf) and suf != "":
                base = pkg[: -len(suf)]
                for alt in variants:
                    alt_pkg = base + alt
                    if alt_pkg != pkg:
                        candidates.append(urdf_path.parent / alt_pkg / tail)

        # 3) Just tail next to URDF
        candidates.append(urdf_path.parent / tail)

        # 4) Scan sibling folders: <urdf_dir>/<any_dir>/tail
        try:
            for d in urdf_path.parent.iterdir():
                if d.is_dir():
                    candidates.append(d / tail)
        except Exception:
            pass

        # Return the first that exists
        for c in candidates:
            if c.exists():
                return c.resolve()

        # Fallback to the most reasonable default
        return (urdf_path.parent / pkg / tail).resolve()

    # No package:// — resolve relative to URDF dir
    p = Path(f)
    if not p.is_absolute():
        p = urdf_path.parent / p
    return p.resolve()

def _pose_to_matrix(pose) -> np.ndarray:
    """Convert urdfpy Pose/array/None to 4x4 matrix."""
    if pose is None:
        return np.eye(4)
    try:
        M = np.array(pose.matrix)
        if M.shape == (4, 4):
            return M
    except Exception:
        pass
    try:
        M = np.array(pose)
        if M.shape == (4, 4):
            return M
    except Exception:
        pass
    try:
        xyz = np.array(getattr(pose, "xyz", [0, 0, 0]), dtype=float)
        rpy = np.array(getattr(pose, "rpy", [0, 0, 0]), dtype=float)
        cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
        cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
        cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
        Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        R = Rz @ Ry @ Rx
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = xyz
        return T
    except Exception:
        return np.eye(4)

@lru_cache(maxsize=2)
def _load_visuals(urdf_path: str) -> Dict[str, Any]:
    """
    Load link visual meshes via urdfpy + trimesh.
    Returns:
      {
        'urdf_file': Path,
        'link_meshes': { link_name: [ { 'verts': (N,3), 'faces': (M,3), 'T_vis': (4,4), 'rgba': (r,g,b,a) }, ... ] },
        'chain_link_names': [ ... ]
      }
    """
    try:
        from urdfpy import URDF
    except Exception as e:
        raise ImportError(f"urdfpy import failed: {e}") from e
    try:
        import trimesh
    except Exception as e:
        raise ImportError(f"trimesh import failed: {e}. Tip: pip install trimesh pycollada lxml") from e

    urdf_p = Path(urdf_path).resolve()
    if not urdf_p.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_p}")

    # Sanitize package:// URIs to local relative paths before loading
    urdf_p_sanitized = _sanitize_urdf_file(urdf_p)

    model = URDF.load(str(urdf_p_sanitized))
    link_meshes: Dict[str, List[Dict[str, Any]]] = {}

    for link in model.links:
        if not link.visuals:
            continue
        for vis in link.visuals:
            if vis.geometry is None or vis.geometry.mesh is None:
                continue
            mesh_geom = vis.geometry.mesh
            mesh_file = _resolve_mesh_path(mesh_geom.filename, urdf_p_sanitized)
            scale = np.array(mesh_geom.scale if mesh_geom.scale is not None else [1,1,1], dtype=float)
            T_vis = _pose_to_matrix(vis.origin)

            try:
                tm = trimesh.load(str(mesh_file), force='mesh', skip_materials=True)
            except Exception:
                continue
            if hasattr(tm, 'geometry'):
                if len(tm.geometry) == 0:
                    continue
                tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))
            if tm.is_empty:
                continue

            verts = np.asarray(tm.vertices, dtype=float)
            faces = np.asarray(tm.faces, dtype=np.int64)
            verts = verts * scale.reshape(1, 3)

            # Assign distinct colors per link for better visualization
            rgba = (0.7, 0.7, 0.75, 1.0)
            try:
                if vis.material and vis.material.color and vis.material.color.rgba is not None:
                    # Override with URDF color if explicitly set
                    rgba = tuple(vis.material.color.rgba)
            except Exception:
                pass

            link_meshes.setdefault(link.name, []).append({
                "verts": verts, "faces": faces,
                "T_vis": T_vis, "rgba": rgba
            })

    chain = load_chain(str(urdf_p_sanitized))
    chain_link_names = [lk.name for lk in chain.links]
    return {"urdf_file": urdf_p_sanitized, "link_meshes": link_meshes, "chain_link_names": chain_link_names}

def _norm_name(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    # Replace any non-alnum characters with underscore
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    return "".join(out)

def compute_robot_workspace_bounds(urdf_path: str, num_samples: int = 50) -> Dict[str, Any]:
    """
    Compute the maximum workspace bounds of the robot by sampling joint configurations.
    Returns min/max bounds for X, Y, Z axes.
    """
    chain = load_chain(str(Path(urdf_path).resolve()))
    
    # Get joint limits from the chain
    all_positions = []
    
    # Sample multiple random configurations within joint limits
    for _ in range(num_samples):
        q = np.zeros(len(chain.links))
        for i, link in enumerate(chain.links):
            if hasattr(link, 'bounds') and link.bounds is not None:
                lower, upper = link.bounds
                q[i] = np.random.uniform(lower, upper)
        
        # Get all link positions for this configuration
        T_list = chain.forward_kinematics(q, full_kinematics=True)
        if isinstance(T_list, np.ndarray) and T_list.ndim == 3:
            transforms = [T_list[i] for i in range(T_list.shape[0])]
        else:
            transforms = list(T_list)
        
        for T in transforms:
            all_positions.append(T[:3, 3])
    
    all_positions = np.array(all_positions)
    
    # Add margin for visualization (20%)
    margin = 0.2
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    return {
        "x_range": [x_min - margin * x_range, x_max + margin * x_range],
        "y_range": [y_min - margin * y_range, y_max + margin * y_range],
        "z_range": [max(0, z_min - margin * z_range), z_max + margin * z_range],
    }

def meshes_for_pose(urdf_path: str, q: np.ndarray, extra_joint_cfg: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """
    Build per-link transformed mesh (verts/faces/rgba) for the given joint pose q.
    Returns items ready for Plotly Mesh3d.

    Parameters
    ----------
    extra_joint_cfg : dict, optional
        Additional joint-name → value pairs (e.g. gripper finger joints)
        that are NOT part of the ikpy kinematic chain but should be driven
        in the FK computation (urdfpy).
    """
    q = _ensure_array(q)
    data = _load_visuals(urdf_path)
    link_meshes = data["link_meshes"]

    # Prefer URDF-based FK so link names align 1:1 with visuals
    urdf_file = Path(data.get("urdf_file", urdf_path)).resolve()
    try:
        from urdfpy import URDF
    except Exception as e:
        # Fallback to IKPY path if urdfpy unavailable
        URDF = None  # type: ignore

    name_to_T_link: Dict[str, np.ndarray] = {}
    if URDF is not None:
        try:
            model = URDF.load(str(urdf_file))
            # Map q -> joint name config in model order discovered via ikpy chain
            chain = load_chain(urdf_path)
            chain_names = [lk.name for lk in chain.links]
            cfg: Dict[str, float] = {}
            
            # Fix: map 1:1 using index, do not skip indices in q
            for i, nm in enumerate(chain_names):
                if i < len(q):
                    # Only add to config if it matches a known joint in the URDF model
                    if nm in getattr(model, "joint_map", {}):
                         cfg[nm] = float(q[i])

            # Merge extra joints (e.g. gripper finger joints)
            if extra_joint_cfg:
                for jname, jval in extra_joint_cfg.items():
                    if jname in getattr(model, "joint_map", {}):
                        cfg[jname] = float(jval)
            
            # Compute FK per-link
            try:
                tf_map = model.link_fk(cfg=cfg)
            except TypeError:
                # Older urdfpy signature
                tf_map = model.link_fk(cfg)
            # Normalize to name -> T
            for k, T in tf_map.items():
                name = getattr(k, "name", k)
                name_to_T_link[str(name)] = np.array(T)
        except Exception:
            name_to_T_link = {}

    if not name_to_T_link:
        # Fallback to IKPY FK (may not align names for some URDFs)
        chain = load_chain(urdf_path)
        T_list = chain.forward_kinematics(q, full_kinematics=True)
        if isinstance(T_list, np.ndarray) and T_list.ndim == 3 and T_list.shape[1:] == (4, 4):
            transforms = [T_list[i] for i in range(T_list.shape[0])]
        else:
            transforms = list(T_list)
        chain_names = [lk.name for lk in chain.links]
        name_to_T_link = {chain_names[i]: transforms[i] for i in range(len(chain_names))}

    # Build traces
    traces: List[Dict[str, Any]] = []
    
    # DEBUG: Print first time only (or just print, it's fine for now) to see link names
    # DEBUG: Print first time only (or just print, it's fine for now) to see link names
    # print(f"[DEBUG] Available mesh links: {list(link_meshes.keys())}")
    
    for link_name, visuals in link_meshes.items():
        # Find transform by exact name or normalized mapping
        T_link = name_to_T_link.get(link_name)
        if T_link is None:
            # Try normalized match
            key = _norm_name(link_name)
            for nm, T in name_to_T_link.items():
                if _norm_name(nm) == key or key in _norm_name(nm) or _norm_name(nm) in key:
                    T_link = T
                    break
        if T_link is None:
            continue

        R_link, t_link = T_link[:3, :3], T_link[:3, 3]
        for vis in visuals:
            T = vis["T_vis"]
            R = R_link @ T[:3, :3]
            t = (R_link @ T[:3, 3]) + t_link

            V = vis["verts"]
            Vw = (V @ R.T) + t
            
            # FIX FRONT-FACING ORIENTATION 
            # FOR THE FRONT VIEW
            
            # R_face_front = np.array([
            #     [-1,  0,  0],
            #     [ 0, -1,  0],
            #     [ 0,  0,  1]
            # ])
            # Vw = Vw @ R_face_front.T

            # FIX FRONT-FACING + slight rotation


            F = vis["faces"]
            r, g, b, a = vis["rgba"]

            traces.append({
                "name": link_name,
                "x": Vw[:, 0], "y": Vw[:, 1], "z": Vw[:, 2],
                "i": F[:, 0],  "j": F[:, 1],  "k": F[:, 2],
                "rgba": (r, g, b, a)
            })
    return traces

# Optional helpers (debug)

def list_chain_links(urdf_path: str) -> List[str]:
    """Return ikpy chain link names (in order)."""
    chain = load_chain(urdf_path)
    return [lk.name for lk in chain.links]


import streamlit as st

@st.cache_resource(show_spinner=False)
def cached_meshes_for_pose(
    urdf_path: str,
    q_tuple: tuple,
    extra_joint_cfg_tuple: Optional[tuple] = None,
):
    """
    Cached version of meshes_for_pose.
    Prevents recomputing robot meshes on every Streamlit rerun.

    Parameters
    ----------
    extra_joint_cfg_tuple : tuple of (key, value) pairs, optional
        Hashable representation of extra joint config (e.g. gripper).
        Pass ``tuple(sorted(extra_dict.items()))`` from the caller.
    """
    import numpy as np
    q = np.array(q_tuple)
    extra = dict(extra_joint_cfg_tuple) if extra_joint_cfg_tuple else None
    return meshes_for_pose(urdf_path, q, extra_joint_cfg=extra)










#Backup


# vilma_core/robot_playback.py
# from pathlib import Path
# from typing import Dict, Any, Optional, Tuple, List
# from functools import lru_cache
# import numpy as np
# import csv

# # NumPy 2.0 compatibility: provide deprecated aliases if missing
# try:  # pragma: no cover
#     _ = np.float  # type: ignore[attr-defined]
# except AttributeError:  # NumPy >= 2.0
#     np.float = float      # type: ignore[attr-defined]
#     np.int = int          # type: ignore[attr-defined]
#     np.complex = complex  # type: ignore[attr-defined]
#     np.bool = bool        # type: ignore[attr-defined]
#     np.object = object    # type: ignore[attr-defined]
#     np.str = str          # type: ignore[attr-defined]

# # Utilities

# def _ensure_array(a, shape_last: Optional[int] = None):
#     arr = np.asarray(a, dtype=float)
#     if shape_last is not None and arr.shape[-1] != shape_last:
#         raise ValueError(f"Expected last dim {shape_last}, got {arr.shape}")
#     return arr


# # IK / Path mapping

# def _sanitize_urdf_file(src_path: Path) -> Path:
#     """
#     Create a sanitized copy of the URDF if it contains ROS package:// URIs by
#     stripping the "package://" scheme from mesh filenames, making them relative.
#     Returns the path to the sanitized (or original) file. Idempotent.
#     """
#     try:
#         text = src_path.read_text(encoding="utf-8", errors="ignore")
#     except Exception:
#         return src_path

#     if "package://" not in text:
#         return src_path

#     # Strip the scheme only; keep the rest of the path relative to the URDF dir
#     sanitized = text.replace("package://", "")

#     out_path = src_path.parent / f"_sanitized_{src_path.name}"
#     try:
#         # Only write if content differs or file missing
#         if not out_path.exists() or out_path.read_text(encoding="utf-8", errors="ignore") != sanitized:
#             out_path.write_text(sanitized, encoding="utf-8")
#         return out_path
#     except Exception:
#         # Fallback to original if we can't write
#         return src_path

# @lru_cache(maxsize=4)
# def load_chain(urdf_path: str):
#     """Load ikpy kinematic chain from URDF (cached)."""
#     try:
#         from ikpy.chain import Chain
#     except Exception as e:
#         raise ImportError("ikpy not installed. Run: pip install ikpy") from e

#     p = Path(urdf_path).resolve()
#     if not p.exists():
#         raise FileNotFoundError(f"URDF not found: {p}")

#     # Sanitize package:// URIs for parsers that don't support the scheme
#     p_sanitized = _sanitize_urdf_file(p)

#     chain = Chain.from_urdf_file(str(p_sanitized), base_elements=None, active_links_mask=None)
#     return chain

# def dmp_to_cartesian(
#     dmp_npy: str,
#     scale_xy_m: Tuple[float, float] = (0.4, 0.4),
#     offset_xy_m: Tuple[float, float] = (0.2, 0.0),
#     z_height_m: float = 0.25,
#     flip_y: bool = True
# ) -> Dict[str, Any]:
#     """Map normalized DMP (x,y) to world (X,Y,Z) in meters; Z fixed for now."""
#     Yg = np.load(str(dmp_npy))  # (N,2)
#     Yg = _ensure_array(Yg, shape_last=2)

#     xs = offset_xy_m[0] + Yg[:, 0] * scale_xy_m[0]
#     ys_raw = Yg[:, 1]
#     ys = offset_xy_m[1] + (1.0 - ys_raw if flip_y else ys_raw) * scale_xy_m[1]
#     zs = np.full_like(xs, float(z_height_m))
#     return {"cartesian_path": np.stack([xs, ys, zs], axis=1)}


# def dmp_xyz_to_cartesian(
#     dmp_npy: str,
#     scale_xyz=(0.5, 0.5, 0.4),
#     offset_xyz=(0.3, 0.0, 0.15),
#     flip_y=True,
# ):
#     """
#     Map XYZ DMP → world Cartesian coordinates (meters)
#     """
#     Yg = np.load(dmp_npy)          # (N,3)
#     Yg = _ensure_array(Yg, shape_last=3)

#     x = offset_xyz[0] + Yg[:, 0] * scale_xyz[0]

#     y_raw = Yg[:, 1]
#     y = offset_xyz[1] + (1.0 - y_raw if flip_y else y_raw) * scale_xyz[1]

#     z = offset_xyz[2] + Yg[:, 2] * scale_xyz[2]

#     # Smooth Z to avoid vertical jitter
#     try:
#         from scipy.signal import savgol_filter
#         z = savgol_filter(z, window_length=11, polyorder=3)
#     except Exception:
#         pass


#     return {
#         "cartesian_path": np.stack([x, y, z], axis=1)
#     }

# def compute_ik_trajectory(
#     urdf_path: str,
#     cartesian_path: np.ndarray,
#     target_frame: Optional[int] = None,
#     initial_position: Optional[np.ndarray] = None
# ) -> Dict[str, Any]:
#     """Solve IK for each 3D point; orientation unconstrained; returns (N,J) joints."""
#     cartesian_path = _ensure_array(cartesian_path, shape_last=3)
#     chain = load_chain(str(Path(urdf_path).resolve()))

#     qs: List[np.ndarray] = []
#     q_prev = initial_position
#     for p in cartesian_path:
#         q = chain.inverse_kinematics(
#             target_position=p,
#             target_orientation=None,
#             initial_position=q_prev,
#         )
#         qs.append(q)
#         q_prev = q

#     Q = np.vstack(qs)
#     return {"q_traj": Q, "dofs": Q.shape[1], "links": len(chain.links)}

# def joints_to_csv(q_traj: np.ndarray, out_csv: str, time_s: Optional[np.ndarray] = None) -> str:
#     """Save joint trajectory to CSV with columns: TIME, q0..qJ-1"""
#     q_traj = _ensure_array(q_traj)
#     N, J = q_traj.shape
#     out_path = Path(out_csv)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     if time_s is None:
#         time_s = np.linspace(0, N/30.0, N)
#     with out_path.open("w", newline="", encoding="utf-8") as f:
#         w = csv.writer(f)
#         w.writerow(["TIME"] + [f"q{j}" for j in range(J)])
#         for i in range(N):
#             w.writerow([f"{float(time_s[i]):.6f}"] + [f"{float(v):.6f}" for v in q_traj[i]])
#     return str(out_path.resolve())

# def joint_positions_for_frame(urdf_path: str, q: np.ndarray) -> np.ndarray:
#     """
#     Link polyline for quick wireframe viewer.
#     Uses ikpy.forward_kinematics(full_kinematics=True).
#     """
#     q = _ensure_array(q)
#     chain = load_chain(str(Path(urdf_path).resolve()))
#     T_list = chain.forward_kinematics(q, full_kinematics=True)
#     if isinstance(T_list, np.ndarray) and T_list.ndim == 3 and T_list.shape[1:] == (4, 4):
#         transforms = [T_list[i] for i in range(T_list.shape[0])]
#     else:
#         transforms = list(T_list)
#     pts = []
#     if len(transforms) == 0 or not np.allclose(transforms[0], np.eye(4), atol=1e-9):
#         pts.append(np.array([0., 0., 0.]))
#     for T in transforms:
#         pts.append(T[:3, 3])
#     return np.vstack(pts)

# # Mesh rendering (urdfpy + trimesh)

# def _resolve_mesh_path(filename: str, urdf_path: Path) -> Path:
#     """
#     Robustly resolve mesh paths:
#     - trims whitespace, normalizes slashes
#     - detects 'package://' even if embedded in longer strings
#     - tries <urdf_dir>/<pkg>/tail, <urdf_dir>/tail
#     - auto-maps suffix variants: _support/_config/_description
#     - scans sibling folders next to the URDF
#     """
#     f = str(filename).strip().replace("\\", "/")

#     # detect package:// anywhere in the string
#     idx = f.lower().find("package://")
#     if idx >= 0:
#         f2 = f[idx + len("package://"):]
#         parts = f2.split("/", 1)
#         pkg = parts[0] if parts else ""
#         tail = parts[1] if len(parts) > 1 else ""

#         candidates: List[Path] = []

#         # 1) Standard: <urdf_dir>/<pkg>/tail
#         candidates.append(urdf_path.parent / pkg / tail)

#         # 2) Try suffix variants (_support, _config, _description, or none)
#         variants = ["_support", "_config", "_description", ""]
#         for suf in variants:
#             if pkg.endswith(suf) and suf != "":
#                 base = pkg[: -len(suf)]
#                 for alt in variants:
#                     alt_pkg = base + alt
#                     if alt_pkg != pkg:
#                         candidates.append(urdf_path.parent / alt_pkg / tail)

#         # 3) Just tail next to URDF
#         candidates.append(urdf_path.parent / tail)

#         # 4) Scan sibling folders: <urdf_dir>/<any_dir>/tail
#         try:
#             for d in urdf_path.parent.iterdir():
#                 if d.is_dir():
#                     candidates.append(d / tail)
#         except Exception:
#             pass

#         # Return the first that exists
#         for c in candidates:
#             if c.exists():
#                 return c.resolve()

#         # Fallback to the most reasonable default
#         return (urdf_path.parent / pkg / tail).resolve()

#     # No package:// — resolve relative to URDF dir
#     p = Path(f)
#     if not p.is_absolute():
#         p = urdf_path.parent / p
#     return p.resolve()

# def _pose_to_matrix(pose) -> np.ndarray:
#     """Convert urdfpy Pose/array/None to 4x4 matrix."""
#     if pose is None:
#         return np.eye(4)
#     try:
#         M = np.array(pose.matrix)
#         if M.shape == (4, 4):
#             return M
#     except Exception:
#         pass
#     try:
#         M = np.array(pose)
#         if M.shape == (4, 4):
#             return M
#     except Exception:
#         pass
#     try:
#         xyz = np.array(getattr(pose, "xyz", [0, 0, 0]), dtype=float)
#         rpy = np.array(getattr(pose, "rpy", [0, 0, 0]), dtype=float)
#         cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
#         cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
#         cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
#         Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
#         Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
#         Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
#         R = Rz @ Ry @ Rx
#         T = np.eye(4); T[:3,:3] = R; T[:3,3] = xyz
#         return T
#     except Exception:
#         return np.eye(4)

# @lru_cache(maxsize=2)
# def _load_visuals(urdf_path: str) -> Dict[str, Any]:
#     """
#     Load link visual meshes via urdfpy + trimesh.
#     Returns:
#       {
#         'urdf_file': Path,
#         'link_meshes': { link_name: [ { 'verts': (N,3), 'faces': (M,3), 'T_vis': (4,4), 'rgba': (r,g,b,a) }, ... ] },
#         'chain_link_names': [ ... ]
#       }
#     """
#     try:
#         from urdfpy import URDF
#     except Exception as e:
#         raise ImportError(f"urdfpy import failed: {e}") from e
#     try:
#         import trimesh
#     except Exception as e:
#         raise ImportError(f"trimesh import failed: {e}. Tip: pip install trimesh pycollada lxml") from e

#     urdf_p = Path(urdf_path).resolve()
#     if not urdf_p.exists():
#         raise FileNotFoundError(f"URDF not found: {urdf_p}")

#     # Sanitize package:// URIs to local relative paths before loading
#     urdf_p_sanitized = _sanitize_urdf_file(urdf_p)

#     model = URDF.load(str(urdf_p_sanitized))
#     link_meshes: Dict[str, List[Dict[str, Any]]] = {}

#     for link in model.links:
#         if not link.visuals:
#             continue
#         for vis in link.visuals:
#             if vis.geometry is None or vis.geometry.mesh is None:
#                 continue
#             mesh_geom = vis.geometry.mesh
#             mesh_file = _resolve_mesh_path(mesh_geom.filename, urdf_p_sanitized)
#             scale = np.array(mesh_geom.scale if mesh_geom.scale is not None else [1,1,1], dtype=float)
#             T_vis = _pose_to_matrix(vis.origin)

#             try:
#                 tm = trimesh.load(str(mesh_file), force='mesh', skip_materials=True)
#             except Exception:
#                 continue
#             if hasattr(tm, 'geometry'):
#                 if len(tm.geometry) == 0:
#                     continue
#                 tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))
#             if tm.is_empty:
#                 continue

#             verts = np.asarray(tm.vertices, dtype=float)
#             faces = np.asarray(tm.faces, dtype=np.int64)
#             verts = verts * scale.reshape(1, 3)

#             rgba = (0.7, 0.7, 0.75, 1.0)
#             try:
#                 if vis.material and vis.material.color and vis.material.color.rgba is not None:
#                     rgba = tuple(vis.material.color.rgba)
#             except Exception:
#                 pass

#             link_meshes.setdefault(link.name, []).append({
#                 "verts": verts, "faces": faces,
#                 "T_vis": T_vis, "rgba": rgba
#             })

#     chain = load_chain(str(urdf_p_sanitized))
#     chain_link_names = [lk.name for lk in chain.links]
#     return {"urdf_file": urdf_p_sanitized, "link_meshes": link_meshes, "chain_link_names": chain_link_names}

# def _norm_name(name: str) -> str:
#     s = name.strip().lower().replace(" ", "_")
#     # Replace any non-alnum characters with underscore
#     out = []
#     for ch in s:
#         out.append(ch if (ch.isalnum() or ch == "_") else "_")
#     return "".join(out)

# def meshes_for_pose(urdf_path: str, q: np.ndarray) -> List[Dict[str, Any]]:
#     """
#     Build per-link transformed mesh (verts/faces/rgba) for the given joint pose q.
#     Returns items ready for Plotly Mesh3d.
#     """
#     q = _ensure_array(q)
#     data = _load_visuals(urdf_path)
#     link_meshes = data["link_meshes"]

#     # Prefer URDF-based FK so link names align 1:1 with visuals
#     urdf_file = Path(data.get("urdf_file", urdf_path)).resolve()
#     try:
#         from urdfpy import URDF
#     except Exception as e:
#         # Fallback to IKPY path if urdfpy unavailable
#         URDF = None  # type: ignore

#     name_to_T_link: Dict[str, np.ndarray] = {}
#     if URDF is not None:
#         try:
#             model = URDF.load(str(urdf_file))
#             # Map q -> joint name config in model order discovered via ikpy chain
#             chain = load_chain(urdf_path)
#             chain_names = [lk.name for lk in chain.links]
#             cfg: Dict[str, float] = {}
#             qi = 0
#             for nm in chain_names:
#                 j = getattr(model, "joint_map", {}).get(nm)
#                 if j is None:
#                     continue
#                 if getattr(j, "joint_type", "fixed") == "fixed":
#                     continue
#                 if qi < q.shape[0]:
#                     cfg[nm] = float(q[qi])
#                     qi += 1
#             # Compute FK per-link
#             try:
#                 tf_map = model.link_fk(cfg=cfg)
#             except TypeError:
#                 # Older urdfpy signature
#                 tf_map = model.link_fk(cfg)
#             # Normalize to name -> T
#             for k, T in tf_map.items():
#                 name = getattr(k, "name", k)
#                 name_to_T_link[str(name)] = np.array(T)
#         except Exception:
#             name_to_T_link = {}

#     if not name_to_T_link:
#         # Fallback to IKPY FK (may not align names for some URDFs)
#         chain = load_chain(urdf_path)
#         T_list = chain.forward_kinematics(q, full_kinematics=True)
#         if isinstance(T_list, np.ndarray) and T_list.ndim == 3 and T_list.shape[1:] == (4, 4):
#             transforms = [T_list[i] for i in range(T_list.shape[0])]
#         else:
#             transforms = list(T_list)
#         chain_names = [lk.name for lk in chain.links]
#         name_to_T_link = {chain_names[i]: transforms[i] for i in range(len(chain_names))}

#     # Build traces
#     traces: List[Dict[str, Any]] = []
#     for link_name, visuals in link_meshes.items():
#         # Find transform by exact name or normalized mapping
#         T_link = name_to_T_link.get(link_name)
#         if T_link is None:
#             # Try normalized match
#             key = _norm_name(link_name)
#             for nm, T in name_to_T_link.items():
#                 if _norm_name(nm) == key or key in _norm_name(nm) or _norm_name(nm) in key:
#                     T_link = T
#                     break
#         if T_link is None:
#             continue

#         R_link, t_link = T_link[:3, :3], T_link[:3, 3]
#         for vis in visuals:
#             T = vis["T_vis"]
#             R = R_link @ T[:3, :3]
#             t = (R_link @ T[:3, 3]) + t_link

#             V = vis["verts"]
#             Vw = (V @ R.T) + t
            
#             # FIX FRONT-FACING ORIENTATION 
#             # FOR THE FRONT VIEW
            
#             # R_face_front = np.array([
#             #     [-1,  0,  0],
#             #     [ 0, -1,  0],
#             #     [ 0,  0,  1]
#             # ])
#             # Vw = Vw @ R_face_front.T

#             # FIX FRONT-FACING + slight rotation


#             F = vis["faces"]
#             r, g, b, a = vis["rgba"]

#             traces.append({
#                 "name": link_name,
#                 "x": Vw[:, 0], "y": Vw[:, 1], "z": Vw[:, 2],
#                 "i": F[:, 0],  "j": F[:, 1],  "k": F[:, 2],
#                 "rgba": (r, g, b, a)
#             })
#     return traces

# # Optional helpers (debug)

# def list_chain_links(urdf_path: str) -> List[str]:
#     """Return ikpy chain link names (in order)."""
#     chain = load_chain(urdf_path)
#     return [lk.name for lk in chain.links]


# import streamlit as st

# @st.cache_resource(show_spinner=False)
# def cached_meshes_for_pose(
#     urdf_path: str,
#     q_tuple: tuple,
# ):
#     """
#     Cached version of meshes_for_pose.
#     Prevents recomputing robot meshes on every Streamlit rerun.
#     """
#     import numpy as np
#     q = np.array(q_tuple)
#     return meshes_for_pose(urdf_path, q)
