# ZED SVO ZIP Workflow

This tutorial explains how to export a ZED `.svo` or `.svo2` recording into the RGB-D ZIP structure expected by the local SVO pipeline.

The goal is to produce one upload-ready ZIP that contains synchronized RGB frames, metric depth maps, camera intrinsics, and an optional preview MP4.

## When To Use This

Use this workflow when you need accurate pick-and-place trajectories from a ZED camera. Plain MP4 files do not contain metric depth, so they are useful for quick visual demos but are not the recommended path for precise robot-space motion.

The local app upload path is:

```text
Local Workspace -> Select Video / BAG / SVO / Data ZIP -> Run Pipeline
```

## Required ZIP Structure

The ZIP must mirror the `data/SVO/` folder structure with at least the **camera intrinsics** and **RGB frames**:

```text
my_svo_data.zip
├── camera/
│   └── <session_id>.npy        # REQUIRED: Camera intrinsics dictionary
├── frames/
│   └── <session_id>/           # REQUIRED: RGB frames
│       ├── frame_000000.png
│       ├── frame_000001.png
│       ├── frame_000002.png
│       └── ...
├── depth_meters/               # RECOMMENDED: Depth arrays for 3D tracking
│   └── <session_id>/
│       ├── frame_000000.npy
│       ├── frame_000001.npy
│       └── ...
├── depth_color/                # OPTIONAL: Colorized depth heatmaps
│   └── <session_id>/
│       ├── frame_000000.png
│       └── ...
└── videos/                     # OPTIONAL: Pre-built video for playback
    └── <session_id>/
        └── video.mp4
```

> **Note:** The `<session_id>` is a unique identifier for the recording session. It is auto-detected from the ZIP contents. All sub-folders must use the **same** session ID.

> **Note:** The ZIP can optionally have a single root folder (e.g., `SVO/camera/...`). The system will detect and strip this prefix automatically.

> **Frames can be `.npy` instead of `.png`**: If RGB frames are stored as `.npy` arrays, they will be automatically converted to PNG on upload.

---

## Camera Intrinsics Format

The `camera/<session_id>.npy` file must contain a dictionary (saved with `np.save()`) with these keys:

```python
{
    "fx": 525.0,      # Focal length X (pixels)
    "fy": 525.0,      # Focal length Y (pixels)
    "cx": 320.0,      # Principal point X (pixels)
    "cy": 240.0,      # Principal point Y (pixels)
    "width": 1280,    # Image width (pixels)
    "height": 720     # Image height (pixels)
}
```

Alternatively, a `.npz` file with the same keys is also supported.

---

## Depth Data Format

Each file in `depth_meters/<session_id>/` should be a NumPy array (`.npy`) with shape `(H, W)` containing depth values **in meters** (float32).

The filename must match the corresponding RGB frame (e.g., `frame_000000.npy` ↔ `frame_000000.png`).

---

## Minimum vs Full Data

| Data | Required? | Purpose |
|:-----|:---------:|:--------|
| `camera/<id>.npy` | **Yes** | Camera intrinsics for 3D projection |
| `frames/<id>/*.png` | **Yes** | RGB frames for hand/object detection |
| `depth_meters/<id>/*.npy` | Recommended | Depth for 3D hand tracking |
| `depth_color/<id>/*.png` | Optional | Visual depth heatmap display |
| `videos/<id>/*.mp4` | Optional | Pre-built playback video |

Without depth data, the pipeline will still run but **3D hand tracking will not work** (only 2D detection).

---

## How To Upload

### Local Workspace
1. Go to the **Local Workspace** in the I-Genius app.
2. Use the **"Select Video / BAG / SVO / Data ZIP"** uploader to select your ZIP file.
3. The system validates the ZIP contents and shows a summary.
4. If valid, data is extracted to `data/SVO/` and a preview is shown.
5. Click **"Run Pipeline"** to start the SVO pipeline.

## Google Colab ZED Export Cells

Use these cells when extracting a `.svo` or `.svo2` file in Google Colab. They write the exact ZIP-ready folder structure expected by the local SVO pipeline.

### Cell 1: Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Cell 2: Install ZED SDK and Python Dependencies

```bash
!apt-get update -y
!apt-get install -y --no-install-recommends zstd wget less udev zip

!wget -q -O ZED_SDK_Linux.run https://download.stereolabs.com/zedsdk/4.2/cu12/ubuntu22
!chmod +x ZED_SDK_Linux.run
!./ZED_SDK_Linux.run -- silent skip_tools skip_od_module

!python3 /usr/local/zed/get_python_api.py
!pip uninstall -y numpy opencv-python opencv-python-headless opencv-contrib-python
!pip install numpy==1.26.4 opencv-python==4.11.0.86
```

If Colab asks to restart the runtime after the NumPy install, restart it and run Cell 1 again before continuing.

### Cell 3: Verify Imports

```python
import numpy as np
import cv2
import pyzed.sl as sl

print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("ZED loaded")
assert np.__version__ == "1.26.4"
```

### Cell 4: Extract SVO Into ZIP-Ready Folders

Only edit `svo_path`, `export_root`, and optionally `session_id_override`.

```python
import os
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

# ================== EDIT ONLY THESE ==================
svo_path = "/content/drive/MyDrive/SVO Extracter/Svo Files/HD2K_Without_objects.svo2"
export_root = "/content/drive/MyDrive/SVO Extracter/Svo Files Extracted/igenius_zip_ready"
session_id_override = ""  # Leave empty to use the SVO filename without .svo/.svo2
reset_export_root = True
# ====================================================

svo_path = Path(svo_path)
export_root = Path(export_root)

if not svo_path.is_file():
    raise RuntimeError(f"SVO file not found: {svo_path}")

raw_session_id = session_id_override.strip() or svo_path.stem
session_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw_session_id)
print("Session ID:", session_id)

if reset_export_root and export_root.exists():
    shutil.rmtree(export_root)

frames_dir = export_root / "frames" / session_id
depth_dir = export_root / "depth_meters" / session_id
depth_color_dir = export_root / "depth_color" / session_id
camera_dir = export_root / "camera"
video_dir = export_root / "videos" / session_id

for directory in [frames_dir, depth_dir, depth_color_dir, camera_dir, video_dir]:
    directory.mkdir(parents=True, exist_ok=True)

init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file(str(svo_path))
init_parameters.svo_real_time_mode = False
init_parameters.coordinate_units = sl.UNIT.METER

zed = sl.Camera()
status = zed.open(init_parameters)
if status != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError(f"Error opening SVO: {status}")

cam_info = zed.get_camera_information()
calib = cam_info.camera_configuration.calibration_parameters.left_cam
resolution = cam_info.camera_configuration.resolution

intrinsics_data = {
    "width": int(resolution.width),
    "height": int(resolution.height),
    "fx": float(calib.fx),
    "fy": float(calib.fy),
    "cx": float(calib.cx),
    "cy": float(calib.cy),
    "distortion": np.array(list(calib.disto), dtype=np.float32),
}

np.savez(camera_dir / f"{session_id}.npz", **intrinsics_data)
np.save(camera_dir / f"{session_id}.npy", intrinsics_data)

image = sl.Mat()
depth = sl.Mat()
runtime = sl.RuntimeParameters()

frame_index = 0
timestamps_ns = []

while True:
    err = zed.grab(runtime)

    if err == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        img_data = image.get_data()

        if img_data.ndim == 3 and img_data.shape[2] == 4:
            bgr = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        else:
            bgr = img_data

        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_m = depth.get_data().astype(np.float32)

        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        timestamps_ns.append(timestamp.get_nanoseconds())

        frame_name = f"frame_{frame_index:06d}"
        rgb_path = frames_dir / f"{frame_name}.png"
        depth_path = depth_dir / f"{frame_name}.npy"

        rgb_ok = cv2.imwrite(str(rgb_path), bgr)
        if not rgb_ok or not rgb_path.exists():
            raise RuntimeError(f"Failed to write RGB frame: {rgb_path}")

        np.save(str(depth_path), depth_m)
        if not depth_path.exists():
            raise RuntimeError(f"Failed to write depth frame: {depth_path}")

        valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
        if valid.size:
            d_min, d_max = np.percentile(valid, [2, 98])
            depth_vis = np.clip(depth_m, d_min, d_max)
            depth_vis = np.nan_to_num(depth_vis, nan=d_max, posinf=d_max, neginf=d_min)
            depth_vis = ((depth_vis - d_min) / max(d_max - d_min, 1e-6) * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(depth_color_dir / f"{frame_name}.png"), depth_vis)

        if frame_index % 100 == 0:
            print(f"Processed {frame_index}")

        frame_index += 1

    elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
        print(f"Finished. Total frames: {frame_index}")
        break
    else:
        raise RuntimeError(f"Grab error: {err}")

np.save(export_root / "timestamps.npy", np.array(timestamps_ns, dtype=np.int64))

metadata = {
    "session_id": session_id,
    "source_svo": str(svo_path),
    "frame_count": frame_index,
    "depth_units": "meters",
    "rgb_pattern": f"frames/{session_id}/frame_000000.png",
    "depth_pattern": f"depth_meters/{session_id}/frame_000000.npy",
}
with open(export_root / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

zed.close()

rgb_count = len(list(frames_dir.glob("frame_*.png")))
depth_count = len(list(depth_dir.glob("frame_*.npy")))
print("RGB frames written:", rgb_count)
print("Depth frames written:", depth_count)

assert frame_index > 0
assert rgb_count == frame_index
assert depth_count == frame_index
```

### Cell 5: Reconstruct Preview Video and Create ZIP

```python
import os
import zipfile
from pathlib import Path

import cv2

fps = 15.0
zip_path = export_root.parent / f"{session_id}_svo_data.zip"
video_path = video_dir / f"{session_id}.mp4"

frame_files = sorted(frames_dir.glob("frame_*.png"))
depth_files = sorted(depth_dir.glob("frame_*.npy"))
if not frame_files:
    raise RuntimeError(f"No frames found in {frames_dir}")
if len(frame_files) != len(depth_files):
    raise RuntimeError(f"RGB/depth count mismatch: {len(frame_files)} RGB vs {len(depth_files)} depth")

first = cv2.imread(str(frame_files[0]))
h, w = first.shape[:2]

writer = None
for fourcc_str in ["avc1", "H264", "mp4v"]:
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    if writer.isOpened():
        break
    writer.release()
    writer = None

if writer is None:
    raise RuntimeError("Could not open MP4 writer")

for frame_file in frame_files:
    frame = cv2.imread(str(frame_file))
    if frame is not None:
        writer.write(frame)
writer.release()

if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(export_root.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(export_root))

print(f"Created ZIP: {zip_path}")
```

### Cell 6: Validate ZIP Before Upload

```python
import zipfile

with zipfile.ZipFile(zip_path, "r") as zf:
    names = set(zf.namelist())

required = [
    f"camera/{session_id}.npy",
    f"camera/{session_id}.npz",
    f"frames/{session_id}/frame_000000.png",
    f"depth_meters/{session_id}/frame_000000.npy",
    f"videos/{session_id}/{session_id}.mp4",
]

missing = [name for name in required if name not in names]
frame_count = len([name for name in names if name.startswith(f"frames/{session_id}/frame_") and name.endswith(".png")])
depth_count = len([name for name in names if name.startswith(f"depth_meters/{session_id}/frame_") and name.endswith(".npy")])

print("ZIP:", zip_path)
print("Session:", session_id)
print("RGB frames:", frame_count)
print("Depth frames:", depth_count)
print("Missing required files:", missing)

assert not missing
assert frame_count > 0
assert frame_count == depth_count
```

The ZIP printed by Cell 6 is the file to upload in the local app.

Expected ZIP contents:

```text
<session_id>_svo_data.zip
├── camera/
│   ├── <session_id>.npy
│   └── <session_id>.npz
├── frames/
│   └── <session_id>/
│       ├── frame_000000.png
│       └── ...
├── depth_meters/
│   └── <session_id>/
│       ├── frame_000000.npy
│       └── ...
├── depth_color/
│   └── <session_id>/
│       ├── frame_000000.png
│       └── ...
├── videos/
│   └── <session_id>/
│       └── <session_id>.mp4
├── metadata.json
└── timestamps.npy
```

## Naming Rules That Matter

- Use the same `<session_id>` under `frames/`, `depth_meters/`, `depth_color/`, and `videos/`.
- Use six-digit frame names: `frame_000000`, `frame_000001`, ...
- RGB and depth filenames must share the same frame number: `frame_000000.png` and `frame_000000.npy`.
- Do not use `depth_000000.npy`; the SVO pipeline maps depth files by the `frame_` number.
- Store ZED depth in meters, not millimeters.

## Creating the ZIP From Existing Extracted Data

If you already have correctly named data under `data/SVO/`, you can create an SVO data ZIP like this:

```python
import zipfile
import os

session_id = "my_recording_01"
output_zip = f"{session_id}_svo_data.zip"

with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    # Camera intrinsics
    zf.write(f"data/SVO/camera/{session_id}.npy", f"camera/{session_id}.npy")

    # RGB frames
    frames_dir = f"data/SVO/frames/{session_id}"
    for fname in sorted(os.listdir(frames_dir)):
        zf.write(os.path.join(frames_dir, fname), f"frames/{session_id}/{fname}")

    # Depth (recommended)
    depth_dir = f"data/SVO/depth_meters/{session_id}"
    if os.path.exists(depth_dir):
        for fname in sorted(os.listdir(depth_dir)):
            zf.write(os.path.join(depth_dir, fname), f"depth_meters/{session_id}/{fname}")

    # Preview video (optional)
    video_dir = f"data/SVO/videos/{session_id}"
    if os.path.exists(video_dir):
        for fname in sorted(os.listdir(video_dir)):
            zf.write(os.path.join(video_dir, fname), f"videos/{session_id}/{fname}")

print(f"Created {output_zip}")
```

## Troubleshooting

### The app says RGB frames are missing

The ZIP must contain RGB image files under:

```text
frames/<session_id>/frame_000000.png
```

Depth visualizations under `depth_color/` do not replace RGB frames. Re-run the Colab exporter and make sure Cell 6 reports `RGB frames: <N>` with `N > 0`.

### RGB and depth counts are different

This usually means the export folder still contains stale files from an older run. In Cell 4, keep:

```python
reset_export_root = True
```

The exporter deletes the old export folder before writing new frames, so RGB, depth, and depth-color outputs are generated from the same SVO pass.

### File names look random or are not sequential

The expected output from the updated cells is sequential:

```text
frame_000000.png
frame_000001.png
frame_000002.png
```

If you see random names or mismatched counts, the ZIP was created from mixed folders or an older extraction script. Use the cells in this tutorial and upload only the ZIP printed by Cell 6.

### The session ID contains `.svo2`

The updated exporter derives `session_id` from `Path(svo_path).stem`, so `.svo` and `.svo2` are removed automatically. If needed, set `session_id_override` manually in Cell 4.

### The ZIP is large

Metric depth arrays are saved as float NumPy files, so large ZIPs are normal. For public examples, use a short sample SVO recording.
