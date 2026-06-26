# Quick Demo — Hello World

Run I-Genius end-to-end **without any physical hardware or ZED camera**.  
Two demo files are provided in `data/demo/`:

| File | What it gives you |
|------|-------------------|
| `HelloWorld.zip` | Full pipeline — RGB frames + depth maps + camera intrinsics. Enables 3D hand tracking, object detection, DMP, and robot trajectory. **Recommended.** |
| `HelloWorld.mp4` | Quick start — plain video only, no depth. Runs frames and object detection but skips 3D steps. |

---

## 1. Install

> Requires Python 3.10 – 3.12 and `uv` (`pip install uv`).

```bash
git clone https://github.com/your-org/I-Genius.git
cd I-Genius

python3.12 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

uv sync
```

> **Docker alternative:**
> ```bash
> docker compose up --build
> # open http://localhost:9002
> ```

---

## 2. Launch the app

```bash
# Linux / macOS
source .venv/bin/activate
uv run streamlit run src/streamlit_template/new_ui/pages/Common/landing_page.py \
    --server.port 8504 --server.address 0.0.0.0

# Windows
.venv\Scripts\activate
uv run streamlit run src\streamlit_template\new_ui\pages\Common\landing_page.py --server.port 8504
```

Open **http://localhost:8504**.

**Expected:** I-Genius landing page with *Local Workspace* and *FIWARE Integration* cards.

---

## 3. Open Local Workspace

Click **Open Local Workspace**.

**Expected:** Three file uploaders and a **Run Pipeline** button appear at the top of the page:

- **Select Video / BAG / SVO / Data ZIP** — main input
- **Select Metadata (Optional)** — JSON trajectory metadata
- **Select Robot Zip (Optional)** — custom URDF for robot playback

---

## 4. Upload a demo file

Choose one of the two paths:

---

### Path A — Full pipeline with depth (recommended)

Upload **`data/demo/HelloWorld.zip`** in the first uploader.

**Expected:**
1. Spinner: *Validating SVO data ZIP* → *Extracting SVO data* → *Reconstructing video from 20 frames*
2. Page reruns — video player appears on the left showing the demo sequence
3. The 🎞️ Frames, ℹ️ Details, 🤖 Robot, 🔍 AI Tools, and 📐 Calibration tabs appear on the right

> If the video shows "No video with supported format", re-upload the same ZIP.  
> The app detects the incompatible codec, rebuilds as H.264, and reloads automatically.

---

### Path B — Quick start with MP4

Upload **`data/demo/HelloWorld.mp4`** in the first uploader.

**Expected:** Frames are extracted immediately. No depth data — steps 1–3 of the pipeline run but steps 4 (DMP) and 5 (Robot) are limited without 3D depth.

---

## 5. Run the pipeline

Click **Run Pipeline**.

The pipeline runs 5 sequential steps:

| # | Step | What it does | Expected output |
|---|------|--------------|-----------------|
| 1 | Hands | MediaPipe hand landmark extraction | 3D hand trajectory in viewer |
| 2 | Objects | YOLOv8 detection + BoT-SORT tracking | Bounding-box overlay on base frame |
| 3 | Segments | Motion segmentation from hand trajectory | Segment boundaries |
| 4 | DMP | Dynamic Movement Primitive fitting | DMP trajectory + plots in `data/SVO/dmp/` |
| 5 | Robot | IK-solved joint trajectory | Animated URDF robot in the 🤖 Robot tab |

> Steps 4–5 require a robot URDF. Upload one via *Select Robot Zip* before running, or see `tutorials/custom_robot_zip.md`.  
> Without a robot, steps 1–3 still fully validate the pipeline.

---

## 6. Select an object with HelloWorld.pt

Switch to the **🔍 AI Tools** tab.

The base frame is detected automatically on load using the default YOLO model.  
To use the custom demo model:

1. In the **Model** dropdown, select **`HelloWorld.pt`** — this is a custom YOLOv8 model trained on the demo scene objects, stored in `data/Common/ai_model/object/HelloWorld.pt`
2. Click **Detect on Frame 0** to re-run detection with this model
3. The annotated base frame updates — bounding boxes appear around detected objects
4. In the **Select object to track** dropdown, pick the target object (e.g. the item being grasped)

**Expected:** The selected bounding box is shown in the left image and highlighted in the dropdown. The bbox is now locked for downstream trajectory computation.

---

## 7. Explore the results

| Tab | What to look for |
|-----|-----------------|
| 🎞️ Frames | Scrub through the 20 frames with the timeline slider |
| ℹ️ Details | Session ID, frame count, camera intrinsics, pipeline step statuses |
| 🤖 Robot | Interactive 3D URDF viewer with animated trajectory (if step 5 ran) |
| 🔍 AI Tools | Object detection on frame 0; select target object |
| 📐 Calibration | Upload `cam2base_calibration.json` for real-robot coordinate mapping |

---

## 8. Generate a new trajectory (Skill Reuse)

After a successful pipeline run, click **Action** (popover in the 🤖 Robot tab) → **Generate New Trajectory**.

This opens the **Skill Reuse** page:

1. The base trajectory is loaded from `data/SVO/dmp/`
2. Click **Detect on Frame 0** to find objects in the new scene
3. Select a new target object
4. Set the release point
5. Click **Compute Skill Reuse Trajectory** → generates an offset-shifted trajectory
6. Optionally push to robot via Vulcanexus/CycloneDDS

---

## 9. Verify files written

After a successful ZIP run the following are populated:

```
data/SVO/
├── frames/<session>/         # 20 RGB frames (.jpg)
├── depth_meters/<session>/   # 20 depth maps (.npy)  — ZIP path only
├── camera/<session>.npy      # Camera intrinsics
├── hands/<session>/          # Hand landmark JSONs
├── objects/<session>_1/      # Object tracking outputs
├── seg/<session>_1/          # Segmentation boundaries
├── dmp/<session>_1/          # DMP trajectory + plots
└── videos/<session>/         # Reconstructed H.264 MP4
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| *No video with supported format* | Re-upload the ZIP — codec rebuilt as H.264 automatically |
| *ModuleNotFoundError: ultralytics* | Run `uv sync` again |
| *Step 2 stays pending / YOLO download* | First run downloads `yolov8n.pt` (~6 MB); wait and re-run |
| *HelloWorld.pt not in dropdown* | Ensure `data/Common/ai_model/object/HelloWorld.pt` exists |
| *Robot step "URDF not found"* | Upload a robot ZIP via *Select Robot Zip*; see `tutorials/custom_robot_zip.md` |
| *Blank page after upload* | Hard-refresh (`Ctrl+Shift+R`) to clear Streamlit widget cache |

---

## Next steps

- [`tutorials/svo_zip_workflow.md`](svo_zip_workflow.md) — prepare your own SVO ZIP from a ZED recording
- [`tutorials/custom_robot_zip.md`](custom_robot_zip.md) — package a custom URDF for robot playback
- [`scripts/vulcanexus/`](../scripts/vulcanexus/) — push trajectories to a real robot over LAN
