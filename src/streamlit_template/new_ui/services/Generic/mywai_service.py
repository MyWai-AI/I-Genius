# mywai_service.py — MYWAI Platform Service

"""
MYWAI API Service Module

Uses mywai_python_integration_kit exclusively for API calls.

Key API functions from the kit:
- login(UserLoginCredentials) -> (token, expiry_datetime) - uses SSO
- initialize_apis(endpoint, auth_token) - must be called after login
- get_equipments() -> Equipments with .items list
- get_facts_by_equipment_id(equipment_id) -> Facts object
"""

import logging
from typing import Tuple, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Default endpoint
DEFAULT_ENDPOINT = "https://igenius.platform.myw.ai/api"

# Import mywai_python_integration_kit
from mywai_python_integration_kit.apis.services.equipment_apis import (
    get_equipments,
    get_facts_by_equipment_id,
    get_equipment_by_id,
    get_equipment_with_sensor,
)
from mywai_python_integration_kit.apis import initialize_apis
from mywai_python_integration_kit.apis.services.users_api import login as kit_login
from mywai_python_integration_kit.data.mywai_schemas.base_apis import UserLoginCredentials

MYWAI_KIT_AVAILABLE = True
logger.info("mywai_python_integration_kit loaded successfully")


def login_mywai(email: str, password: str, endpoint: str = None) -> Tuple[bool, str, Optional[dict]]:
    """
    Login to MYWAI platform using mywai_python_integration_kit.
    
    The kit uses SSO authentication at sso.platform.myw.ai.
    After successful login, initialize_apis() is automatically called.
    
    Args:
        email: User email
        password: User password
        endpoint: API endpoint (optional, default: https://igenius.platform.myw.ai/api)
        
    Returns:
        Tuple of (success, token_or_error, user_data)
    """
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    
    try:
        logger.info(f"Logging in via mywai_python_integration_kit SSO for {email}")
        
        # Login using the kit (uses SSO at sso.platform.myw.ai)
        user_credential = UserLoginCredentials(email=email, password=password)
        auth_token, expiry_date = kit_login(user_credential)
        
        if auth_token:
            # Initialize APIs with the token
            logger.info(f"Initializing APIs with endpoint: {endpoint}")
            initialize_apis(endpoint=endpoint, auth_token=auth_token)
            
            logger.info(f"Login successful for {email}, token expires: {expiry_date}")
            return True, auth_token, {
                "email": email, 
                "token": auth_token, 
                "expiry": str(expiry_date) if expiry_date else None
            }
        else:
            return False, "Login failed: No token returned", None
            
    except ValueError as e:
        # Invalid token/credentials - log as warning without full traceback
        logger.warning(f"Login validation failed for {email}: {e}")
        return False, f"Login failed: {str(e)}", None
        
    except Exception as e:
        logger.exception("Login error")
        return False, f"Login error: {str(e)}", None


def get_equipments_list(token: str = None, endpoint: str = None) -> Tuple[bool, Any]:
    """
    Get all equipments from MYWAI platform using mywai_python_integration_kit.
    
    Returns:
        Tuple of (success, Equipments.items list or error message)
    """
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    
    try:
        logger.info("Getting equipments via mywai_python_integration_kit")
        
        # Re-initialize if token provided
        if token:
            initialize_apis(endpoint=endpoint, auth_token=token)
        
        # Get equipments using the kit - returns Equipments object
        result = get_equipments()
        
        if result and hasattr(result, 'items'):
            logger.info(f"Found {len(result.items)} equipments")
            return True, result.items  # Return the list of Equipment objects
        elif result:
            return True, result
        else:
            return False, "No equipments returned"
            
    except Exception as e:
        logger.exception("Error getting equipments")
        return False, f"Error: {str(e)}"


def get_facts_by_equipment(equipment_id: int, token: str = None, endpoint: str = None) -> Tuple[bool, Any]:
    """
    Get facts/measures for a specific equipment using mywai_python_integration_kit.
    
    Returns:
        Tuple of (success, Facts object or error message)
    """
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    
    try:
        logger.info(f"Getting facts for equipment {equipment_id} via kit")
        
        if token:
            initialize_apis(endpoint=endpoint, auth_token=token)
        
        # Get facts using the kit - returns Facts object
        result = get_facts_by_equipment_id(equipment_id)
        
        if result:
            return True, result
        else:
            return False, "No facts returned"
            
    except Exception as e:
        logger.exception("Error getting facts")
        return False, f"Error: {str(e)}"


def extract_videos_from_facts(facts_obj: Any) -> list:
    """
    Parse a Facts object and extract video files.
    
    Args:
        facts_obj: The Facts object return from get_facts_by_equipment_id
        
    Returns:
        List of dictionaries containing video metadata:
        {
            "name": str,
            "container": str,
            "file_path": str,
            "timestamp": datetime,
            "fact_id": int
        }
    """
    videos = []
    
    # Check if we have a valid Facts object with a list of facts
    if not hasattr(facts_obj, "facts") or not facts_obj.facts:
        return []
        
    # Iterate through each Fact
    for fact in facts_obj.facts:
        if not hasattr(fact, "fact_items") or not fact.fact_items:
            continue
            
        # Iterate through items in the Fact
        for item in fact.fact_items:
            # Check for video type (file_type usually 4 for VIDEO in MYWAI)
            # We can also check explicit file extensions if needed
            is_video = False
            
            # Check file_type attribute if available (assuming 4 is VIDEO based on common MYWAI schema)
            if hasattr(item, "file_type") and item.file_type == 4:
                is_video = True
            # Fallback: check file extension
            elif hasattr(item, "file_path") and item.file_path:
                ext = item.file_path.lower().split('.')[-1]
                if ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
                    is_video = True
            
            if is_video:
                videos.append({
                    "name": item.file_path.split('/')[-1] if item.file_path else f"Video {item.id}",
                    "container": getattr(item, "container_name", None) or "mywai-video-container", # Fallback name
                    "file_path": item.file_path,
                    "timestamp": fact.timestamp,
                    "fact_id": fact.id,
                    "measure_name": getattr(item, "measure_name", "Unknown Measure")
                })
                
    # Sort by timestamp descending
    videos.sort(key=lambda x: x["timestamp"], reverse=True)
    return videos


def process_facts_for_display(facts_obj: Any) -> list:
    """
    Parse a Facts object and prepare for display.
    Groups facts by Session ID (extracted from SVO files) or by Fact ID for BAG/Video.
    """
    import re
    from collections import defaultdict
    
    processed_tasks = {}
    
    if not hasattr(facts_obj, "facts") or not facts_obj.facts:
        return []

    # Regex to catch SVO files like 639076238549565273_frame_000147_OpenArm001.jpg
    # and 639076237149074951_camera_intrinsics_OpenArm001.npz
    svo_pattern = re.compile(r'(?:.*frame_\d+|.*camera_intrinsics)_(.+)\.(jpg|npy|npz)$', re.IGNORECASE)
        
    for fact in facts_obj.facts:
        if not hasattr(fact, "fact_items") or not fact.fact_items:
            print(f"DEBUG: Fact {getattr(fact, 'id', 'Unknown')} has no items.")
            continue
            
        # We might have multiple sessions uploaded in one fact, or one session across multiple facts.
        # We group items into tasks. A task is either a full SVO session, a BAG file, or a Video.
        
        for item in fact.fact_items:
            path = getattr(item, "file_path", "")
            if not path:
                continue
            
            filename = path.split('/')[-1]
            ext = filename.lower().split('.')[-1]
            print(f"DEBUG: FactItem vars: {vars(item)}")
            
            # Check SVO pattern
            svo_match = svo_pattern.match(filename)
            
            # Helper to parse mywai timestamp strings to datetime for UI
            def _parse_ts(ts):
                if isinstance(ts, str):
                    from datetime import datetime
                    try:
                        # Sometimes trailing Z, sometimes microseconds
                        ts_clean = ts.replace("Z", "")
                        if "." in ts_clean:
                            ts_clean = ts_clean[:26] # Try to cap microsecond precision
                        return datetime.fromisoformat(ts_clean)
                    except ValueError:
                        return datetime.now()
                return ts or datetime.now()

            if svo_match:
                session_id = svo_match.group(1)
                task_key = f"svo_{session_id}"
                
                if task_key not in processed_tasks:
                    print(f"DEBUG: Creating new SVO task for session {session_id}")
                    processed_tasks[task_key] = {
                        "fact_id": fact.id,
                        "timestamp": _parse_ts(fact.timestamp),
                        "name": session_id,
                        "task_type": "svo",
                        "session_id": session_id,
                        "items": [],
                        "measure_name": getattr(item, "measure_name", "Unknown Measure")
                    }
                
                processed_tasks[task_key]["items"].append({
                    "name": filename,
                    "fact_id": fact.id,
                    "file_path": path,
                    "label": "RGB" if ext == "jpg" else ("Depth" if ext == "npy" else "Camera"),
                    "ext": ext
                })
                
            elif ext == "bag":
                task_key = f"bag_{fact.id}_{filename}"
                processed_tasks[task_key] = {
                    "fact_id": fact.id,
                    "timestamp": _parse_ts(fact.timestamp),
                    "name": f"{filename}",
                    "task_type": "bag",
                    "items": [{
                        "name": filename,
                        "fact_id": fact.id,
                        "file_path": path,
                        "label": filename
                    }],
                    "measure_name": getattr(item, "measure_name", "Unknown Measure")
                }
                
            elif ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
                task_key = f"video_{fact.id}_{filename}"
                
                label = "Video"
                if "rgb" in filename.lower():
                    label = "RGB"
                elif "depth" in filename.lower() or "rgbd" in filename.lower():
                    label = "RGBD"
                else:
                    label = filename

                processed_tasks[task_key] = {
                    "fact_id": fact.id,
                    "timestamp": _parse_ts(fact.timestamp),
                    "name": f"Video - {filename}",
                    "task_type": "video",
                    "items": [{
                        "name": filename,
                        "fact_id": fact.id,
                        "file_path": path,
                        "label": label,
                        "ext": ext
                    }],
                    "measure_name": getattr(item, "measure_name", "Unknown Measure")
                }
                
    # Format the dictionary back to a list
    result_list = []
    for k, v in processed_tasks.items():
        # Alias items to videos for UI compatibility 
        # (Though we'll modify UI to handle these types directly)
        v["videos"] = sorted(v["items"], key=lambda x: x["label"])
        del v["items"]
        result_list.append(v)
            
    # Sort tasks by timestamp descending
    result_list.sort(key=lambda x: x["timestamp"], reverse=True)
    return result_list


def download_blobs_concurrently(items: list, token: str, endpoint: str, max_workers: int = 15) -> tuple[bool, str]:
    """
    Concurrently downloads a list of items using ThreadPoolExecutor.
    Each item must be a dictionary with at least:
        - file_path: The filename on MYWAI (e.g. 639076239122460629_frame_000208_OpenArm001.jpg)
        - fact_id:   The GUID of the parent Fact
        - local_path: Local output path to save the file
    
    Uses POST /Fact/downloadItemFile with body:
        {"blobFolder":"generic","fileName":"...","factId":"...","measureId":0}
    
    Returns: (success_bool, message)
    """
    import concurrent.futures
    import requests

    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    headers = {"accept": "*/*", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    download_url = f"{endpoint}/Fact/downloadItemFile"

    def _download_single(item):
        from pathlib import Path
        import base64
        
        f_name = item["file_path"].split("/")[-1]  # just the filename
        fact_id = item["fact_id"]
        l_path = Path(item["local_path"])
        
        payload = {
            "blobFolder": "generic",
            "fileName": f_name,
            "factId": fact_id,
            "measureId": 0
        }
        
        try:
            r = requests.post(download_url, headers=headers, json=payload, verify=False, timeout=120)
            if r.status_code != 200:
                return False, f"Failed {r.status_code} for {f_name}"
            
            # Response is JSON: {"rawBytes": "<base64>", "contentType": "...", ...}
            resp_json = r.json()
            raw_b64 = resp_json.get("rawBytes")
            if not raw_b64:
                return False, f"No rawBytes in response for {f_name}"
            
            binary_data = base64.b64decode(raw_b64)
                
            l_path.parent.mkdir(parents=True, exist_ok=True)
            with open(l_path, "wb") as f:
                f.write(binary_data)
            
            return True, str(l_path)
        except Exception as e:
            return False, f"Error on {f_name}: {e}"

    success_count = 0
    fail_count = 0
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_single, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            ok, msg = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                errors.append(msg)

    if fail_count == 0:
        return True, f"Successfully downloaded all {success_count} files."
    else:
        return False, f"Downloaded {success_count} files, but {fail_count} failed. First error: {errors[0] if errors else 'Unknown'}"


def download_blob(
    container: str,
    file_path: str,
    output_path: str,
    token: str = None,
    endpoint: str = None,
) -> Tuple[bool, str]:
    """
    Download a generic file from MYWAI storage.
    
    Uses direct HTTP since the kit doesn't have a simple public download helper.
    """
    import requests
    import urllib.parse
    from pathlib import Path
    
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    
    # Try the new Storage endpoint first
    blob_path = urllib.parse.quote(f"{container}/{file_path}", safe="")
    url = f"{endpoint}/Storage/getBlob/{blob_path}"
    
    headers = {"accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        r = requests.get(url, headers=headers, verify=False, timeout=120, stream=True)
        
        # Fallback to older Fact Generic blob if Storage route 404s
        if r.status_code == 404:
             fallback_url = f"{endpoint}/Fact/getGenericFile/{container}/{file_path}"
             r = requests.get(fallback_url, headers=headers, verify=False, timeout=120, stream=True)

        if r.status_code != 200:
            return False, f"Failed {r.status_code}: {r.text[:300]}"
        
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        
        return True, f"Downloaded → {out}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def download_blob_directory(
    directory_path: str,
    local_destination: str,
    token: str = None,
    endpoint: str = None
) -> Tuple[bool, str]:
    """
    Download all files and subdirectories from a MYWAI blob Storage directory.
    Uses GET /Storage/getPathContent/{directory}
    """
    import requests
    import urllib.parse
    from pathlib import Path
    
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    # ensure directory ends with slash to identify as directory for the blob storage API, OR URL-encode
    enc_dir = urllib.parse.quote(directory_path, safe="")
    url = f"{endpoint}/Storage/getPathContent/{enc_dir}"
    
    headers = {"accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    try:
        logger.info(f"[DEBUG DL] Attempting to list directory: {url}")
        r = requests.get(url, headers=headers, verify=False, timeout=60)
        if r.status_code != 200:
            logger.error(f"[DEBUG DL] Failed to list {directory_path}! URL: {url} | Status: {r.status_code} | Response: {r.text[:200]}")
            return False, f"Failed to list directory {r.status_code}: {r.text[:200]}"
            
        file_paths = r.json() # assuming it returns a list of blob paths
        logger.info(f"[DEBUG DL] Successfully listed directory {directory_path}, found {len(file_paths)} items.")
        if not file_paths:
            return True, "Directory is empty or not found."
            
        successes = 0
        local_dir = Path(local_destination)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        container = directory_path.split("/")[0] if "/" in directory_path else directory_path
        
        for full_blob_path in file_paths:
             # full_blob_path ex: "equipmentmodels/eq/123/openarm.urdf"
             # Remove the directory_path prefix so we can reconstruct the relative layout
             relative_path = full_blob_path
             if relative_path.startswith(directory_path):
                 relative_path = relative_path[len(directory_path):].lstrip("/")
                 
             # if the blob path already includes container, we don't want to double it
             # the download_blob expects `container` and `file_path`.
             parts = full_blob_path.split("/")
             blob_container = parts[0]
             blob_file_path = "/".join(parts[1:])
             
             out_path = local_dir / relative_path
             logger.info(f"[DEBUG DL] Downloading file {blob_file_path} to {out_path}")
             ok, _ = download_blob(blob_container, blob_file_path, str(out_path), token, endpoint)
             if ok:
                 successes += 1
                 
        return True, f"Downloaded {successes}/{len(file_paths)} files to {local_dir}"
    except Exception as e:
        logger.error(f"[DEBUG DL] Error listing/downloading: {e}")
        return False, f"Error listing/downloading directory: {e}"


def upload_blob(
    container: str,
    file_path: str,
    local_path: str,
    token: str = None,
    endpoint: str = None,
) -> Tuple[bool, str]:
    """
    Upload a local file to MYWAI blob storage.

    Uses the kit's blob.upload_blob which talks to the Swagger-registered
    POST /Storage/uploadBlob route. The kit's ApiObjectsContainer must already
    be initialised (happens automatically after login_mywai).

    Args:
        container: Blob container name (e.g. "equipmentmodels").
        file_path: Target path inside the container (e.g. "eq/123/robot/animations.json").
        local_path: Absolute local path to the file to upload.
        token: Bearer token (only used for fallback raw HTTP if kit fails).
        endpoint: API endpoint base URL.

    Returns:
        (success, message)
    """
    from pathlib import Path

    if not Path(local_path).exists():
        return False, f"Local file not found: {local_path}"

    # Clean up duplicated container logic
    pass_filepath = file_path
    if container and pass_filepath.startswith(container + "/"):
        pass_filepath = pass_filepath[len(container)+1:]

    # --- Try kit route first (ApiObjectsContainer already initialised at login) ---
    try:
        logger.info(f"[DEBUG UPLOAD] Attempting Kit Route: {container}/{pass_filepath} from {local_path}")
        from mywai_python_integration_kit.apis.services.blob import save_blob
        save_blob(
            blob_path=f"{container}/{pass_filepath}",
            local_file_path=local_path,
        )
        logger.info(f"[DEBUG UPLOAD] Kit upload successful.")
        return True, f"Uploaded → {container}/{pass_filepath}"
    except Exception as kit_err:
        logger.warning(f"[DEBUG UPLOAD] Kit upload failed ({kit_err}), trying raw HTTP fallback…")

    # --- Fallback: raw HTTP POST ---
    import requests
    import urllib.parse
    endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
    file_name = Path(local_path).name
    path_to_save = str(Path(pass_filepath).parent).replace("\\", "/")
    
    url = f"{endpoint}/Storage/uploadBlob?fileName={file_name}&pathToSave={urllib.parse.quote(container + '/' + path_to_save, safe='')}"
    headers = {"accept": "*/*"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        logger.info(f"[DEBUG UPLOAD] Calling raw HTTP POST: {url}")
        with open(local_path, "rb") as fh:
            data = fh.read()
            r = requests.post(url, headers=headers, data=data, verify=False, timeout=60)
        
        logger.info(f"[DEBUG UPLOAD] Primary raw HTTP Status: {r.status_code} Body: {r.text[:200]}")
        if r.status_code in (200, 202):
            return True, f"Uploaded → {container}/{file_path}"
            
        # Fallback to older Fact Generic blob
        fallback_url = f"{endpoint}/Fact/uploadGenericFile/{container}/{path_to_save}"
        logger.info(f"[DEBUG UPLOAD] Primary failed, attempting fallback URL: {fallback_url}")
        with open(local_path, "rb") as fh:
            r_fb = requests.post(
                fallback_url,
                headers=headers,
                files={"file": (file_name, fh, "application/json")},
                verify=False,
                timeout=60,
            )
        logger.info(f"[DEBUG UPLOAD] Fallback HTTP Status: {r_fb.status_code} Body: {r_fb.text[:200]}")
        if r_fb.status_code in (200, 202):
            return True, f"Uploaded → {container}/{file_path}"
        return False, f"Upload failed {r.status_code}/{r_fb.status_code}"
    except Exception as e:
        return False, f"Upload error: {str(e)}"


def download_equipment_model_folder(
    equipment_id: int,
    model_blob_path: str,
    target_local_dir: str = None,
    container: str = "equipmentmodels",
    token: str = None,
    endpoint: str = None,
) -> Tuple[bool, str, str]:
    """
    Download a full robot model folder from MYWAI blob storage.

    For a .zip blob   → download + extract; returns extracted dir.
    For a .urdf blob  → calls download_blob_directory on the parent folder
                        to recursively fetch the URDF, meshes, and config.json
                        from the same server directory.

    Args:
        equipment_id: Numeric equipment ID.
        model_blob_path: Server-side blob path, e.g. "equipment/123/robot.zip"
                         or "equipment/123/openarm.urdf".
        target_local_dir: Exact local directory to place the downloaded files. 
                          Defaults to data/Common/robot_models/downloads/eq_{equipment_id}/
        container: Blob container name on the server.
        token: Bearer token.
        endpoint: API endpoint base URL.

    Returns:
        (success, message, local_dir_path)
    """
    import zipfile
    from pathlib import Path

    local_dir_path = target_local_dir if target_local_dir else f"data/Common/robot_models/downloads/eq_{equipment_id}"
    local_dir = Path(local_dir_path)
    local_dir.mkdir(parents=True, exist_ok=True)

    blob_filename = model_blob_path.split("/")[-1]
    # Server-side base directory (everything before the filename)
    blob_dir = "/".join(model_blob_path.split("/")[:-1])  # e.g. "equipment/123"

    # ------------------------------------------------------------------ ZIP --
    if blob_filename.lower().endswith(".zip"):
        zip_local = local_dir / blob_filename
        if not zip_local.exists():
            ok, msg = download_blob(container, model_blob_path, str(zip_local), token, endpoint)
            if not ok:
                return False, f"Failed to download zip: {msg}", ""

        # Extract
        extract_dir = local_dir / "extracted"
        if not extract_dir.exists():
            try:
                with zipfile.ZipFile(zip_local, "r") as zf:
                    zf.extractall(extract_dir)
            except Exception as e:
                return False, f"Failed to extract zip: {e}", ""

        return True, f"Extracted to {extract_dir}", str(extract_dir)

    # ----------------------------------------------------------------- URDF --
    if blob_filename.lower().endswith(".urdf"):
        import xml.etree.ElementTree as ET
        import time
        import posixpath
        
        urdf_local = local_dir / blob_filename

        # Clean container / file path separation for download_blob
        pass_container = container
        pass_filepath_base = blob_dir
        if pass_container and pass_filepath_base.startswith(pass_container + "/"):
             pass_filepath_base = pass_filepath_base[len(pass_container)+1:]

        # 1. Download main URDF
        urdf_server_path = f"{pass_filepath_base}/{blob_filename}" if pass_filepath_base else blob_filename
        ok, msg = download_blob(pass_container, urdf_server_path, str(urdf_local), token, endpoint)
        if not ok:
             return False, f"Failed to download URDF: {msg}", ""

        # 2. Download Companion Files silently
        for comp in ["config.json", "animations.json"]:
             comp_server_path = f"{pass_filepath_base}/{comp}" if pass_filepath_base else comp
             download_blob(pass_container, comp_server_path, str(local_dir / comp), token, endpoint)

        # 3. Parse URDF to find mesh references
        try:
             tree = ET.parse(urdf_local)
             root = tree.getroot()
             mesh_refs = []
             for mesh in root.iter("mesh"):
                 fn = mesh.get("filename", "")
                 if fn:
                     mesh_refs.append(fn)
        except Exception as e:
             logger.warning(f"Could not parse URDF for mesh refs: {e}")
             mesh_refs = []

        # 4. Mesh Path Resolution & Downloading
        for mesh_ref in mesh_refs:
             raw_path = mesh_ref

             if raw_path.startswith("package://") or raw_path.startswith("/"):
                 # Remove "package://" or leading "/"
                 raw_path = raw_path.replace("package://", "").lstrip("/")
                 # Remove the first folder name (assumed package name)
                 parts = raw_path.split("/")
                 if len(parts) > 1:
                     clean_path = "/".join(parts[1:])
                 else:
                     clean_path = raw_path
             else:
                 # Standard relative path, resolve normally 
                 clean_path = posixpath.normpath(raw_path)

             # Prepend Base Folder
             local_mesh = local_dir / clean_path
             local_mesh.parent.mkdir(parents=True, exist_ok=True)

             if local_mesh.exists():
                 continue

             mesh_server_path = f"{pass_filepath_base}/{clean_path}" if pass_filepath_base else clean_path
             
             # 5. Download Mesh with Retry Mechanism
             retries = [0.2, 0.4, 0.6]
             for attempt, delay in enumerate([0] + retries):
                 if attempt > 0:
                     time.sleep(delay)
                 
                 ok, _ = download_blob(pass_container, mesh_server_path, str(local_mesh), token, endpoint)
                 if ok:
                     break
             else:
                 logger.warning(f"Could not download mesh {mesh_server_path} after retries")

        return True, f"Robot folder ready at {local_dir}", str(local_dir)

    return False, f"Unsupported model format: {blob_filename}", ""


def is_kit_available() -> bool:
    """Check if mywai_python_integration_kit is available."""
    return MYWAI_KIT_AVAILABLE


# Re-export kit functions for direct use
kit_get_equipments = get_equipments
kit_get_facts_by_equipment_id = get_facts_by_equipment_id
kit_get_equipment_by_id = get_equipment_by_id
kit_get_equipment_with_sensor = get_equipment_with_sensor
kit_initialize_apis = initialize_apis
