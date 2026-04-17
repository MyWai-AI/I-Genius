"""
ZIP Upload Handler for SVO Data
================================

Handles .zip uploads that contain pre-extracted SVO pipeline data.
The ZIP should mirror the data/SVO folder structure (or a subset of it).

Minimum required contents for the pipeline to work:
  - camera/<session_id>.npy   (camera intrinsics dict: fx, fy, cx, cy, width, height)
  - frames/<session_id>/*.png  (RGB frames)

Recommended (for full 3D pipeline):
  - depth_meters/<session_id>/*.npy  (depth arrays matching each RGB frame)

Optional:
  - depth_color/<session_id>/*.png   (colorized depth heatmaps)
  - videos/<session_id>/*.mp4        (pre-built video for playback)
  - segmentation/<session_id>/...    (segmentation masks)

Example ZIP structure:
    my_svo_data.zip
    ├── camera/
    │   └── my_session.npy
    ├── frames/
    │   └── my_session/
    │       ├── frame_000000.png
    │       ├── frame_000001.png
    │       └── ...
    ├── depth_meters/
    │   └── my_session/
    │       ├── frame_000000.npy
    │       └── ...
    └── depth_color/    (optional)
        └── my_session/
            └── ...
"""
import os
import zipfile
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Known SVO sub-folders that we expect in the ZIP
SVO_KNOWN_SUBDIRS = {
    "camera", "frames", "depth_meters", "depth_color",
    "depth_raw", "videos", "segmentation",
    "hands", "objects", "dmp", "plots",
}


class ZipValidationResult:
    """Result of validating a ZIP file for SVO pipeline compatibility."""

    def __init__(self):
        self.is_valid: bool = False
        self.session_id: Optional[str] = None
        self.has_camera: bool = False
        self.has_frames: bool = False
        self.has_depth_meters: bool = False
        self.has_depth_color: bool = False
        self.has_videos: bool = False
        self.frame_count: int = 0
        self.depth_count: int = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.root_prefix: str = ""  # e.g. "", "SVO/", "my_data/"

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.session_id:
            lines.append(f"Session ID: **{self.session_id}**")
        lines.append(f"Camera intrinsics: {'✅' if self.has_camera else '❌ MISSING (required)'}")
        lines.append(f"RGB frames: {'✅' if self.has_frames else '❌ MISSING (required)'} ({self.frame_count} files)")
        lines.append(f"Depth meters: {'✅' if self.has_depth_meters else '⚠️ Missing'} ({self.depth_count} files)")
        lines.append(f"Depth color: {'✅' if self.has_depth_color else '—'}")
        lines.append(f"Videos: {'✅' if self.has_videos else '—'}")
        if self.errors:
            lines.append("\n**Errors:**")
            for e in self.errors:
                lines.append(f"  ❌ {e}")
        if self.warnings:
            lines.append("\n**Warnings:**")
            for w in self.warnings:
                lines.append(f"  ⚠️ {w}")
        return "\n".join(lines)


def validate_svo_zip(zip_file) -> ZipValidationResult:
    """
    Validate a ZIP file to check if it contains valid SVO pipeline data.

    Parameters
    ----------
    zip_file : file-like or str/Path
        A file-like object (e.g. Streamlit UploadedFile) or path to a ZIP file.

    Returns
    -------
    ZipValidationResult
        Validation result with details about what was found.
    """
    result = ZipValidationResult()

    try:
        zf = zipfile.ZipFile(zip_file, "r")
    except (zipfile.BadZipFile, Exception) as e:
        result.errors.append(f"Invalid ZIP file: {e}")
        return result

    names = zf.namelist()
    if not names:
        result.errors.append("ZIP file is empty.")
        zf.close()
        return result

    # --- Detect root prefix ---
    # The ZIP might have files directly like camera/xxx.npy or under a root folder like SVO/camera/xxx.npy
    root_prefix = _detect_root_prefix(names)
    result.root_prefix = root_prefix

    # --- Detect session ID ---
    # Look in frames/<session_id>/ or camera/<session_id>.npy
    session_id = _detect_session_id(names, root_prefix)
    result.session_id = session_id

    if not session_id:
        result.errors.append(
            "Could not detect session ID. The ZIP must contain "
            "frames/<session_id>/ folder or camera/<session_id>.npy file."
        )
        zf.close()
        return result

    # --- Check camera intrinsics ---
    camera_patterns = [
        f"{root_prefix}camera/{session_id}.npy",
        f"{root_prefix}camera/{session_id}.npz",
        f"{root_prefix}camera/camera_intrinsics.npy",
        f"{root_prefix}camera/camera_intrinsics.npz",
        # Nested: camera/<session_id>/camera_intrinsics.npz (common export pattern)
        f"{root_prefix}camera/{session_id}/camera_intrinsics.npy",
        f"{root_prefix}camera/{session_id}/camera_intrinsics.npz",
    ]
    for pat in camera_patterns:
        if pat in names:
            result.has_camera = True
            break

    if not result.has_camera:
        result.errors.append(
            f"Camera intrinsics not found. Expected one of: "
            f"camera/{session_id}.npy, camera/{session_id}.npz, "
            f"camera/camera_intrinsics.npy, camera/camera_intrinsics.npz"
        )

    # --- Check RGB frames ---
    frames_prefix = f"{root_prefix}frames/{session_id}/"
    frame_files = [
        n for n in names
        if n.startswith(frames_prefix)
        and not n.endswith("/")
        and any(n.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".npy"))
    ]
    result.frame_count = len(frame_files)
    result.has_frames = result.frame_count > 0

    if not result.has_frames:
        result.errors.append(
            f"No RGB frames found. Expected image files in frames/{session_id}/"
        )

    # --- Check depth_meters ---
    depth_prefix = f"{root_prefix}depth_meters/{session_id}/"
    depth_files = [
        n for n in names
        if n.startswith(depth_prefix)
        and not n.endswith("/")
        and n.lower().endswith(".npy")
    ]
    result.depth_count = len(depth_files)
    result.has_depth_meters = result.depth_count > 0

    if not result.has_depth_meters:
        result.warnings.append(
            f"No depth data found in depth_meters/{session_id}/. "
            "3D hand tracking will not work without depth data."
        )

    # --- Check depth_color (optional) ---
    dc_prefix = f"{root_prefix}depth_color/{session_id}/"
    dc_files = [n for n in names if n.startswith(dc_prefix) and not n.endswith("/")]
    result.has_depth_color = len(dc_files) > 0

    # --- Check videos (optional) ---
    vid_prefix = f"{root_prefix}videos/{session_id}/"
    vid_files = [
        n for n in names
        if n.startswith(vid_prefix) and n.lower().endswith(".mp4")
    ]
    result.has_videos = len(vid_files) > 0

    # --- Final validity ---
    result.is_valid = result.has_camera and result.has_frames

    if result.is_valid and not result.has_depth_meters:
        result.warnings.append(
            "ZIP is valid but depth data is missing. "
            "Only 2D pipeline steps will work correctly."
        )

    zf.close()
    return result


def extract_svo_zip(
    zip_file,
    target_root: str = "data/SVO",
    session_id: Optional[str] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Extract a validated SVO ZIP file into the target data directory.

    Parameters
    ----------
    zip_file : file-like or str/Path
        The ZIP file to extract.
    target_root : str
        Root directory to extract into (default: data/SVO).
    session_id : str, optional
        Override session ID. If None, auto-detected from ZIP contents.

    Returns
    -------
    tuple of (success: bool, message: str, session_id: str or None)
    """
    try:
        zf = zipfile.ZipFile(zip_file, "r")
    except Exception as e:
        return False, f"Invalid ZIP file: {e}", None

    names = zf.namelist()
    root_prefix = _detect_root_prefix(names)
    detected_sid = _detect_session_id(names, root_prefix)

    if session_id is None:
        session_id = detected_sid

    if not session_id:
        zf.close()
        return False, "Could not determine session ID from ZIP contents.", None

    target = Path(target_root)
    target.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    skipped_count = 0

    for member in names:
        # Skip directories
        if member.endswith("/"):
            continue

        # Strip root prefix to get relative path
        if root_prefix and member.startswith(root_prefix):
            rel_path = member[len(root_prefix):]
        else:
            rel_path = member

        # Destination path
        dest_path = target / rel_path

        # Skip if file already exists and same size
        if dest_path.exists():
            info = zf.getinfo(member)
            if dest_path.stat().st_size == info.file_size:
                skipped_count += 1
                continue

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract
        with zf.open(member) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())
        extracted_count += 1

    zf.close()

    msg = f"Extracted {extracted_count} files to {target_root}/"
    if skipped_count > 0:
        msg += f" ({skipped_count} already existed, skipped)"

    # --- Normalise nested intrinsics path ---
    # Pattern: camera/<session_id>/camera_intrinsics.npz  →  camera/<session_id>.npz
    # This allows the existing .npz → .npy conversion in local_page.py to pick it up.
    nested_intr = target / "camera" / session_id / "camera_intrinsics.npz"
    nested_intr_npy = target / "camera" / session_id / "camera_intrinsics.npy"
    flat_intr_npz = target / "camera" / f"{session_id}.npz"
    flat_intr_npy = target / "camera" / f"{session_id}.npy"

    if not flat_intr_npy.exists() and not flat_intr_npz.exists():
        if nested_intr.exists():
            import shutil
            shutil.copy2(str(nested_intr), str(flat_intr_npz))
            logger.info(f"Copied nested intrinsics to {flat_intr_npz}")
        elif nested_intr_npy.exists():
            import shutil
            shutil.copy2(str(nested_intr_npy), str(flat_intr_npy))
            logger.info(f"Copied nested intrinsics to {flat_intr_npy}")

    return True, msg, session_id


def get_zip_session_hash(zip_file_name: str) -> str:
    """Generate a deterministic session ID hash from the ZIP filename."""
    name_clean = zip_file_name.rsplit(".", 1)[0]
    name_clean = "".join(c for c in name_clean if c.isalnum() or c in ("_", "-"))
    return name_clean


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_root_prefix(names: List[str]) -> str:
    """
    Detect if files in the ZIP are under a common root folder.

    For example:
      - "camera/session.npy" → root_prefix = ""
      - "SVO/camera/session.npy" → root_prefix = "SVO/"
      - "my_data/frames/session/" → root_prefix = "my_data/"
    """
    # Check if any top-level entry matches a known SVO subdir
    top_level = set()
    for n in names:
        parts = n.split("/")
        if parts[0]:
            top_level.add(parts[0])

    # If any top-level directory is a known SVO subdir, no prefix
    if top_level & SVO_KNOWN_SUBDIRS:
        return ""

    # Otherwise, check if all files share a common first directory
    # and under that we find known subdirs
    first_dirs = set()
    for n in names:
        parts = n.split("/")
        if len(parts) >= 2 and parts[0]:
            first_dirs.add(parts[0])

    if len(first_dirs) == 1:
        candidate = first_dirs.pop()
        # Check if known subdirs appear under this candidate
        second_level = set()
        prefix = candidate + "/"
        for n in names:
            if n.startswith(prefix):
                rest = n[len(prefix):]
                sub = rest.split("/")[0]
                if sub:
                    second_level.add(sub)
        if second_level & SVO_KNOWN_SUBDIRS:
            return prefix

    return ""


def _detect_session_id(names: List[str], root_prefix: str) -> Optional[str]:
    """
    Detect the session ID from ZIP contents.

    Strategy:
    1. Look at frames/<session_id>/ directories
    2. Look at camera/<session_id>.npy files
    """
    session_ids = set()

    # From frames directory
    frames_prefix = f"{root_prefix}frames/"
    for n in names:
        if n.startswith(frames_prefix):
            rest = n[len(frames_prefix):]
            parts = rest.split("/")
            if len(parts) >= 2 and parts[0]:
                session_ids.add(parts[0])

    if len(session_ids) == 1:
        return session_ids.pop()

    # From camera directory
    camera_prefix = f"{root_prefix}camera/"
    for n in names:
        if n.startswith(camera_prefix) and not n.endswith("/"):
            fname = n[len(camera_prefix):]
            if "/" not in fname:  # Direct file in camera/
                base = fname.rsplit(".", 1)[0]
                if base not in ("camera_intrinsics",):
                    session_ids.add(base)

    if len(session_ids) == 1:
        return session_ids.pop()

    # Multiple session IDs found — pick the first from frames
    if session_ids:
        return sorted(session_ids)[0]

    return None
