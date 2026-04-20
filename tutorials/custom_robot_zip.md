# Custom Robot ZIP Workflow

This tutorial explains how to package a custom robot model for upload in the local I-Genius app.

Use this when you want the robot playback step to show your own URDF, colors, home pose, and DMP alignment settings instead of the default robot.

## Expected ZIP Structure

The ZIP should contain a URDF, optional robot configuration, and the required mesh files:

```text
my_robot.zip
├── robot.urdf
├── config.json
├── animations.json
└── meshes/
    ├── base_link.stl
    ├── link_1.stl
    └── ...
```

Notes:

- `robot.urdf` is required.
- `config.json` is optional but strongly recommended.
- `animations.json` is optional if you want named robot animations.
- all mesh paths in the URDF should resolve relative to the extracted ZIP contents.

## URDF Rules

- Use relative mesh paths such as `meshes/link_1.stl` whenever possible.
- `package://...` paths are also supported, but relative paths are easier to keep portable in a ZIP upload.
- Include every referenced mesh file inside the ZIP.

## Recommended `config.json`

Example:

```json
{
  "home_pose": {
    "Joint_1": 0.0,
    "Joint_2": -0.52,
    "Joint_3": 1.05,
    "Joint_4": 0.0,
    "Joint_5": -1.57,
    "Joint_6": 0.0
  },
  "link_colors": {
    "base_link": "0x333333",
    "Link_1": "0xFFA500",
    "Link_2": "0x00FF00",
    "Link_6": "#FF00FF"
  },
  "scale": [1, 1, 1],
  "rotation": [-90, 0, 0],
  "position": [0, 0, 0],
  "dmp_scale": [0.5, 0.5, 0.5],
  "dmp_offset": [0.4, 0.0, 0.2],
  "dmp_rotation_z": 90.0
}
```

## Config Fields

| Field | Purpose |
| :--- | :--- |
| `home_pose` | Default joint values used for the home position |
| `link_colors` | Per-link viewer colors |
| `scale` | Visual scale in the viewer |
| `rotation` | Visual rotation used to align the robot model |
| `position` | Visual position offset |
| `dmp_scale` | Scaling used when mapping the learned path into robot space |
| `dmp_offset` | Offset applied to the learned path |
| `dmp_rotation_z` | Rotation around the robot base Z axis |
| `animations` | Optional named trajectories |

## Upload In The App

1. Open the local app.
2. Go to `Local Workspace`.
3. Use the `Select Robot Zip (Optional)` uploader.
4. Upload your robot ZIP.
5. Run the pipeline and inspect the `Robot` step.

## Troubleshooting

### The robot does not render

Check that:

- `robot.urdf` exists in the ZIP
- every referenced mesh file is included
- mesh paths in the URDF match the extracted ZIP layout

### Colors or home pose are missing

Check that `config.json` is valid JSON and that the joint and link names match the URDF exactly.

### The DMP path is not aligned with the robot

Tune:

- `dmp_scale`
- `dmp_offset`
- `dmp_rotation_z`

These are the main alignment controls for matching the detected demonstration path to your robot workspace.
