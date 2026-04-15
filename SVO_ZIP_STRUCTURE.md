# SVO Data ZIP Upload Structure

To skip live SVO extraction (which requires the ZED SDK), you can upload a `.zip` file containing **pre-extracted SVO data**. This is currently supported for the **SVO pipeline** only.

## 📂 Required ZIP Structure

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

## 📷 Camera Intrinsics Format

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

## 📊 Depth Data Format

Each file in `depth_meters/<session_id>/` should be a NumPy array (`.npy`) with shape `(H, W)` containing depth values **in meters** (float32).

The filename must match the corresponding RGB frame (e.g., `frame_000000.npy` ↔ `frame_000000.png`).

---

## ⚠️ Minimum vs Full Data

| Data | Required? | Purpose |
|:-----|:---------:|:--------|
| `camera/<id>.npy` | **Yes** | Camera intrinsics for 3D projection |
| `frames/<id>/*.png` | **Yes** | RGB frames for hand/object detection |
| `depth_meters/<id>/*.npy` | Recommended | Depth for 3D hand tracking |
| `depth_color/<id>/*.png` | Optional | Visual depth heatmap display |
| `videos/<id>/*.mp4` | Optional | Pre-built playback video |

Without depth data, the pipeline will still run but **3D hand tracking will not work** (only 2D detection).

---

## 🛠️ How to Upload

### Local Page
1. Go to the **Local Page** in the VILMA UI.
2. Use the **"Or Upload SVO Data (.zip)"** uploader to select your ZIP file.
3. The system validates the ZIP contents and shows a summary.
4. If valid, data is extracted to `data/SVO/` and a preview is shown.
5. Click **"Run Pipeline"** to start the SVO pipeline.

### MYWAI Page
1. On the **Playback View** for a task, expand  **"📦 Upload SVO Data ZIP (Optional)"**.
2. Upload your ZIP file.
3. After extraction, click **"Run Pipeline"** to start the SVO pipeline.

> **Tip:** If a `.zip` file is downloaded from MYWAI blob storage, it is automatically detected and processed as SVO data if it matches the expected structure.

---

## 🔧 Creating the ZIP

Example Python script to create an SVO data ZIP from extracted data:

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

print(f"Created {output_zip}")
```
