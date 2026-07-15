# ARISE Vulcanexus Migration Checklist

This checklist records the current reusable-module alignment target for ARISE submission.

## Architecture Rules

- ROS 2 / Vulcanexus / Fast DDS is the official reusable middleware interface.
- The reusable module exposes standard ROS 2 / DDS interfaces.
- DDS discovery and network routing are deployment-specific.
- Valid DDS-native deployment mechanisms include native DDS discovery, Fast DDS Discovery Server, DDS Router, and Fast DDS WAN TCP configuration where supported by repository helpers.
- FIWARE / Orion-LD is a separate northbound execution metadata and status layer.
- FIWARE does not carry raw robot trajectories.
- This repository does not implement the DDS-NGSI-LD Enabler.

## Canonical Trajectory Contract

| Setting | Value |
|---|---|
| `ROS_DOMAIN_ID` | `42` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` |
| Topic | `/learned_trajectory` |
| Message | `geometry_msgs/msg/PoseArray` |

## Repository Evidence

Application:

- `Dockerfile`
- `docker-compose.yml`
- service `vilma-agent`

Publisher:

- `scripts/vulcanexus/docker_publish_traj.sh`
- `scripts/vulcanexus/traj_pose_array_pub.py`

Edge receiver:

- `scripts/vulcanexus/run_edge_receiver.sh`
- `scripts/vulcanexus/edge_receive_posearray.py`

Discovery Server:

- `scripts/vulcanexus/docker_run_fastdds_discovery_server.sh`
- default UDP port `14520`

DDS Router:

- `scripts/vulcanexus/ddsrouter_cloud.template.yaml`
- `scripts/vulcanexus/ddsrouter_edge.template.yaml`
- `scripts/vulcanexus/render_ddsrouter_wan_config.sh`
- `scripts/vulcanexus/docker_run_ddsrouter.sh`
- default WAN TCP port `45678`

FIWARE:

- `docker-compose.fiware.yml`
- `src/streamlit_template/new_ui/services/Common/fiware_service.py`
- `src/streamlit_template/new_ui/pages/Common/fiware_page.py`

## Submission Checks

- Base Compose validates and exposes only `vilma-agent`.
- GPU Compose overlay validates and still exposes only `vilma-agent`.
- FIWARE Compose validates separately.
- Current documentation does not claim middleware services are started by base Compose.
- Push-to-Robot UI describes the Vulcanexus ROS 2 / Fast DDS interface.
- Edge Operator workflow exposes DDS / ROS 2 receiver operation without active bridge lifecycle controls.
- Fast DDS Discovery Server default port is `14520`.
- DDS Router WAN TCP port remains `45678`.
- DDS Router is described as a DDS-native option, not as a proven production default.
- Historical deployment evidence that does not describe the current reusable architecture is not part of the current submitted documentation set.

## Non-Blocking Evidence Gaps

- Cross-machine DDS Router runtime validation must be executed and recorded before DDS Router can be called the validated production path.
- Physical robot execution depends on robot SDK installation, safety checks, and deployment-specific operator procedure.
- DDS network routing must be selected and validated for each target topology.
