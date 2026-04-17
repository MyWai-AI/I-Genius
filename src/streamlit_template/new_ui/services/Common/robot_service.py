
import os
import shutil
import zipfile
import hashlib
import json
import base64
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)

def resolve_robot_path(base_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the primary URDF file within a directory.
    scans recursively for .urdf files.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None
        
    urdfs = list(base_dir.rglob("*.urdf"))
    
    # Filter out sanitized urdfs if they exist alongside originals
    # valid_urdfs = [u for u in urdfs if "_sanitized" not in u.name]
    # But for now, we just pick the first one found or prefer one without underscore?
    # Actually, simpler is just taking the first one.
    
    if urdfs:
        return urdfs[0]
    return None

def hydrate_urdf_with_meshes(urdf_path: Union[str, Path], base_dir: Optional[Path] = None) -> Optional[str]:
    """
    Reads a URDF file, finds all <mesh filename="..."> tags, 
    reads the referenced files, encodes them as Base64, 
    and returns the modified URDF string with Data URIs.
    """
    urdf_path = Path(urdf_path)
    if base_dir is None:
        base_dir = urdf_path.parent
        
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        for mesh in root.iter('mesh'):
            filename = mesh.get('filename')
            if filename:
                # Resolve path logic
                real_path = None
                
                # 1. Handle package:// syntax
                if filename.startswith('package://'):
                     # Heuristic: Strip package prefix and look in base_dir
                     subpath = filename.replace('package://', '')
                     parts = subpath.split('/')
                     
                     candidates = [
                        base_dir / subpath,
                        urdf_path.parent / subpath,
                     ]
                     
                     if len(parts) > 1:
                        # Try stripping package name folder
                        suffix_no_pkg = "/".join(parts[1:])
                        candidates.append(base_dir / suffix_no_pkg)
                        # Try just filename
                        candidates.append(base_dir / parts[-1])
                        
                     # Recursive search as fallback
                     try:
                         found = list(base_dir.rglob(Path(filename).name))
                         candidates.extend(found)
                     except: pass
                     
                     for c in candidates:
                        if c.exists():
                            real_path = c
                            break
                            
                else:
                    # 2. Relative path
                    clean_filename = filename.lstrip("/").lstrip("\\")
                    candidates = [
                        (urdf_path.parent / clean_filename).resolve(),
                        (urdf_path.parent / filename).resolve(),
                        base_dir / Path(clean_filename).name
                    ]
                    # Recursive search
                    try:
                         found = list(base_dir.rglob(Path(clean_filename).name))
                         candidates.extend(found)
                    except: pass
                    
                    for c in candidates:
                        if c.exists():
                            real_path = c
                            break
                
                if real_path and real_path.exists():
                    with open(real_path, "rb") as f:
                        b64_data = base64.b64encode(f.read()).decode("utf-8")
                        
                        ext = real_path.suffix.lower()
                        mime = "model/stl" 
                        if ext == ".obj": mime = "model/obj"
                        elif ext == ".dae": mime = "model/vnd.collada+xml"
                        
                        mesh.set('filename', f"data:{mime};base64,{b64_data}")
                else:
                    logger.warning(f"Could not find mesh file for {filename}")

        out_xml = ET.tostring(root, encoding='unicode')
        return out_xml

    except Exception as e:
        logger.error(f"Failed to hydrate URDF: {e}")
        return None

def prepare_robot_viewer_data(
    robot_path: Union[str, Path], 
    is_zip: bool = False
) -> Dict[str, Any]:
    """
    Prepares the viewer_data dictionary for the SyncViewer.
    
    Args:
        robot_path: Path to the .zip file or .urdf file/directory.
        is_zip: If True, treats robot_path as a zip file to extract.
    
    Returns:
        Dict containing the 'models' list for the viewer.
    """
    robot_path = Path(robot_path)
    work_dir = robot_path.parent
    
    # 1. Handle Zip Extraction
    if is_zip and robot_path.suffix.lower() == ".zip":
        # Create temp dir for extraction based on hash
        file_hash = hashlib.md5(str(robot_path).encode()).hexdigest()[:8]
        extract_dir = work_dir / f"extracted_{file_hash}"
        
        if not extract_dir.exists():
            try:
                with zipfile.ZipFile(robot_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except Exception as e:
                logger.error(f"Failed to extract zip: {e}")
                return {}
        
        work_dir = extract_dir
    else:
        # Check if it's a directory or a direct URDF file
        if robot_path.is_file() and robot_path.suffix.lower() == ".urdf":
            work_dir = robot_path.parent
        elif robot_path.is_dir():
            work_dir = robot_path

    # 2. Find URDF
    urdf_path = resolve_robot_path(work_dir)
    if not urdf_path:
        logger.error(f"No URDF found in {work_dir}")
        return {}

    # 3. Load Config
    config_path = urdf_path.parent / "config.json"
    robot_config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                robot_config = json.load(f)
        except Exception:
            pass

    # 4. Hydrate URDF
    hydrated_xml = hydrate_urdf_with_meshes(urdf_path, base_dir=work_dir)
    if hydrated_xml:
        urdf_b64 = base64.b64encode(hydrated_xml.encode('utf-8')).decode("utf-8")
        urdf_uri = f"data:text/xml;base64,{urdf_b64}"
    else:
        # Fallback to direct read
        with open(urdf_path, "rb") as f:
            urdf_b64 = base64.b64encode(f.read()).decode("utf-8")
        urdf_uri = f"data:text/xml;base64,{urdf_b64}"

    # 5. Construct Data
    home_pose = robot_config.get("home_pose", {})
    link_colors = robot_config.get("link_colors", {})
    scale = robot_config.get("scale", [1, 1, 1])
    rotation = robot_config.get("rotation", [-90, 0, 0])

    return {
        "models": [
            {
                "entityName": "RobotPreview",
                "loadType": "URDF",
                "path": urdf_uri,
                "rotation": rotation,
                "position": [0,0,0],
                "scale": scale,
                "homePose": home_pose,
                "linkColors": link_colors,
                "animations": {},
                "trajectory": []
            }
        ]
    }
