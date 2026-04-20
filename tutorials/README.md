# Tutorials

This folder contains the reusable user-facing guides for I-Genius. Keep tutorial material here so the repository root stays clean and easy to scan.

## Start Here

1. Follow the setup steps in the root [README.md](../README.md).
2. Prepare a ZED/SVO RGB-D ZIP with [svo_zip_workflow.md](svo_zip_workflow.md).
3. Upload the ZIP in the local app and run the SVO pipeline.

## Guides

- [svo_zip_workflow.md](svo_zip_workflow.md) - export a ZED `.svo` or `.svo2` recording into the ZIP structure accepted by the local SVO pipeline.
- [custom_robot_zip.md](custom_robot_zip.md) - package a reusable custom robot URDF, config, and meshes for upload in the local app.

## Documentation Rules

- Keep step-by-step user guides in this folder.
- Keep the root `README.md` short and focused on setup plus the recommended workflow.
- Do not add generated outputs, experiment logs, screenshots from private platforms, or one-off notebook transcripts.
