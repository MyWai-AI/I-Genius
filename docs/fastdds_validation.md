# Fast DDS Validation - 2026-07-13

## Validated Architecture

```text
I-GENIUS / VILMA
-> Vulcanexus ROS 2 publisher
-> /learned_trajectory
-> geometry_msgs/msg/PoseArray
-> Fast DDS through rmw_fastrtps_cpp
-> separate ROS 2 Humble edge machine on the same LAN
```

## Transport Contract

| Item | Value |
|---|---|
| `ROS_DOMAIN_ID` | `42` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` |
| Trajectory topic | `/learned_trajectory` |
| Trajectory type | `geometry_msgs/msg/PoseArray` |

## Observed Result

A PoseArray published from the server-side VILMA / Vulcanexus path was received by a separate ROS 2 Humble edge machine on the same LAN using the contract above.

The restored Skill Reuse Push-to-Robot path publishes the selected dense Cartesian `.npy` trajectory. For the previously inspected screwdriver example, the selected artifact was:

```text
data/SVO/skill_reuse/HD720_SN25577940_13-07-18_3/skill_reuse_screwdriver.npy
```

That `.npy` file has shape `(229, 3)`, so the restored path publishes 229 Cartesian poses for that example.