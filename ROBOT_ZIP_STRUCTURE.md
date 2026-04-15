# Robot Zip Upload Structure

To upload a custom robot to VILMA, you must create a `.zip` file containing your robot's URDF, configuration, and mesh files.

## 📂 Zip File Structure

The zip file **must** have the following flat structure (or a single root folder containing these):

```text
my_robot.zip
├── robot.urdf          # REQUIRED: The URDF definition file.
├── config.json         # OPTIONAL: Pose, colors, Dmp settings, and animations.
├── animation.json      # OPTIONAL: Separate file for animations (alternative to config.json).
└── meshes/             # REQUIRED: Folder containing all STL/OBJ/DAE files.
    ├── base_link.stl
    ├── link_1.stl
    └── ...
```

> **Note:** Your URDF must reference meshes using relative paths (e.g., `filename="meshes/link_1.stl"`). `package://` paths are also supported and resolved automatically.

---

## ⚙️ Config.json Schema

The `config.json` file is crucial for defining how the robot looks and behaves in the VILMA pipeline.

### Full Example
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

### Fields Description

| Field | Type | Description |
| :--- | :--- | :--- |
| **`home_pose`** | Object | Key-value pairs matching your URDF **Joint Names** to their initial angles (in radians). Used for the "Home" position. |
| **`link_colors`** | Object | Key-value pairs matching your URDF **Link Names** to Hex colors (e.g., `0xFF0000` or `#FF0000`). |
| **`scale`** | [x, y, z] | Visual scaling of the robot model in the viewer (default: `[1, 1, 1]`). |
| **`rotation`** | [x, y, z] | Visual rotation in degrees to align the robot (default: `[-90, 0, 0]` is common for ROS-to-Three.js). |
| **`dmp_scale`** | [x, y, z] | **Critical:** Scaling factors for mapping the detected hand trajectory to the robot's workspace. |
| **`dmp_offset`** | [x, y, z] | **Critical:** Offset (in meters) to shift the trajectory into the robot's feasible workspace. |
| **`dmp_rotation_z`** | Number | Rotation (in degrees) around Z-axis to align the trajectory with the robot base. |
| **`animations`** | Object | (Optional) Dictionary of named trajectories, e.g., `{"Run": [{"time":0, "q":{...}}]}`. Can also be in `animation.json`. |

---

## 🛠️ How to Upload
1. Go to the **Local Page** in the VILMA UI.
2. In the "Select Robot Zip" uploader, drop your `.zip` file.
3. The system will extract it and override the default robot.
4. Check the **"Robot"** tab to preview your model (Home Pose and Colors should appear).
5. Run the **Pipeline**; Step 4 will now use your custom robot and DMP settings.
