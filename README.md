# I-Genius

I-Genius is a local-first visual imitation learning toolkit for robot manipulation demonstrations. The current reusable open-source workflow focuses on ZED `.svo` / `.svo2` recordings exported into an RGB-D ZIP, then processed locally into trajectories, DMP outputs, and robot playback.

## Current Scope

This repository is organized around:

- local Streamlit UI for uploading demonstration data
- ZED/SVO RGB-D ZIP uploads for accurate pick-and-place workflows
- hand, object, trajectory, DMP, and robot playback pipeline steps
- reusable documentation for preparing data outside the app
- local runtime folders that are ignored by git

This repository should not contain:

- private platform credentials
- cloud integration screenshots
- private package feed files
- generated videos, extracted frames, depth maps, or DMP runtime outputs
- one-off notebook outputs or temporary experiment data

## Quick Start

Create the Python environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip uv
uv sync
```

Run the local app:

```bash
source .venv/bin/activate
uv run streamlit run src/streamlit_template/new_ui/pages/Common/landing_page.py --server.port 8504 --server.address 0.0.0.0
```

Open:

```text
http://localhost:8504
```

Docker is also available:

```bash
docker compose up --build
```

Open:

```text
http://localhost:9002
```

## Recommended Workflow

For precise pick-and-place data, use the ZED/SVO ZIP workflow:

1. Record a demonstration with a ZED camera as `.svo` or `.svo2`.
2. Use the Colab cells in [tutorials/svo_zip_workflow.md](tutorials/svo_zip_workflow.md) to export RGB frames, metric depth, camera intrinsics, and a preview video.
3. Upload the generated `<session_id>_svo_data.zip` in the local app.
4. Confirm that the ZIP validator shows camera intrinsics, RGB frames, and depth meters.
5. Run the SVO pipeline.
6. Review the hand, object, trajectory, DMP, and robot playback steps.

Expected upload structure:

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
└── videos/
    └── <session_id>/
        └── <session_id>.mp4
```

The detailed export and validation tutorial is here:

- [tutorials/svo_zip_workflow.md](tutorials/svo_zip_workflow.md)
- [tutorials/custom_robot_zip.md](tutorials/custom_robot_zip.md)

## Project Structure

- `src/streamlit_template/` - Streamlit UI and pipeline logic
- `src/streamlit_template/core/SVO/` - SVO processing steps
- `src/streamlit_template/new_ui/services/SVO/` - SVO upload and helper services
- `data/SVO/` - local SVO runtime folders with `.gitkeep` placeholders
- `data/Common/` - shared models, robot files, and required app assets
- `tutorials/` - user-facing setup and data-preparation guides
- `docker-compose.yml` - local container entry point
- `pyproject.toml` - Python project dependencies

## Runtime Data Policy

Runtime outputs belong under `data/SVO/`, `data/Generic/`, or `data/BAG/`, but they should not be committed. The repository tracks only folder placeholders and source assets needed to run the app.

To clean generated runtime files while keeping placeholders, run:

```bash
find data/SVO data/Generic data/BAG -type f ! -name .gitkeep -delete
find data/SVO data/Generic data/BAG -type d -empty -delete
```

Use this before preparing a public commit if you have been testing uploads locally.

## Notes

- ZED depth exported by the tutorial is stored in meters.
- RGB and depth frame names must match exactly, for example `frame_000000.png` and `frame_000000.npy`.
- MP4 input can be useful for quick demos, but it does not contain metric depth and is not the recommended path for precise robot-space pick-and-place.
