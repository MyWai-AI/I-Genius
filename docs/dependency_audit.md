# Dependency Audit

This audit describes the current repository state for application deployment, ROS 2 / Vulcanexus / Fast DDS trajectory transport, DDS Router helpers, and FIWARE.

## Host Prerequisites

| Area | Requirement | Used by |
|---|---|---|
| OS | Ubuntu 22.04 or compatible Linux | development and deployment host |
| Python | Python 3.10 or newer | local Streamlit app and tooling |
| Package manager | `uv` | Python dependency sync |
| System libraries | `build-essential`, `git`, `curl`, `ffmpeg`, `libgl1`, `libglib2.0-0` | Python wheels and media processing |
| Docker | Docker Engine | application container and optional FIWARE Compose |
| Docker Compose | Docker Compose plugin | `vilma-agent` and FIWARE Compose validation |
| ROS 2 / Vulcanexus | Humble-compatible ROS 2 environment | trajectory publisher, receiver, executor, and DDS helpers |
| Fast DDS | `rmw_fastrtps_cpp` | canonical ROS 2 middleware implementation |

## Python Dependencies

The Python environment is defined by `pyproject.toml`.

Notable runtime dependencies include:

- Streamlit UI: `streamlit`, `streamlit-post-message`
- Computer vision and trajectory work: `numpy`, `opencv-python-headless`, `mediapipe`, `ultralytics`, `transformers`, `timm`, `scipy`, `scikit-learn`, `plotly`
- Robot/model tooling: `ikpy`, `urdfpy`, `trimesh`, `pycollada`, `shapely`, `networkx`
- Communication helpers: `flask`, `requests`, `python-dotenv`
- Optional RealSense support through the declared extra

## Containers

Current tracked Compose files define:

| Compose file | Services | Role |
|---|---|---|
| `docker-compose.yml` | `vilma-agent` | Streamlit application, `9002:8504` |
| `docker-compose.gpu.yml` | overlay for `vilma-agent` | NVIDIA GPU visibility |
| `docker-compose.fiware.yml` | `fiware-mongo`, `fiware-orion-ld` | optional northbound metadata/status layer |

The base application Compose file does not define middleware services.

## ROS 2 / Robot Communication

The reusable trajectory path is:

```text
VILMA UI
-> data/_runtime/vulcanexus/last_cartesian_push.csv
-> scripts/vulcanexus/docker_publish_traj.sh
-> Vulcanexus ROS 2 publisher
-> /learned_trajectory
-> Fast DDS
-> Edge ROS 2 receiver
-> robot-specific backend
```

Canonical defaults:

| Setting | Value |
|---|---|
| Topic | `/learned_trajectory` |
| Message | `geometry_msgs/msg/PoseArray` |
| Status topic | `/trajectory_status` optional |
| `ROS_DOMAIN_ID` | `42` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` |
| CSV path | `data/_runtime/vulcanexus/last_cartesian_push.csv` |

## DDS Helper Scripts

Publisher and receiver:

- `scripts/vulcanexus/docker_publish_traj.sh`
- `scripts/vulcanexus/traj_pose_array_pub.py`
- `scripts/vulcanexus/run_edge_receiver.sh`
- `scripts/vulcanexus/edge_receive_posearray.py`

Discovery Server:

- `scripts/vulcanexus/docker_run_fastdds_discovery_server.sh`
- default UDP port `14520`
- `ROS_DISCOVERY_SERVER=<server-ip>:14520`

DDS Router:

- `scripts/vulcanexus/ddsrouter_cloud.template.yaml`
- `scripts/vulcanexus/ddsrouter_edge.template.yaml`
- `scripts/vulcanexus/render_ddsrouter_wan_config.sh`
- `scripts/vulcanexus/docker_run_ddsrouter.sh`
- default WAN TCP port `45678`

Fast DDS WAN TCP profiles:

- `scripts/vulcanexus/fastdds_wan_tcp_server.template.xml`
- `scripts/vulcanexus/fastdds_wan_tcp_client.template.xml`
- `scripts/vulcanexus/render_fastdds_wan_tcp_profile.sh`
- `scripts/vulcanexus/docker_run_fastdds_discovery_server_xml.sh`

DDS Router is a DDS-native option for routed server-to-edge deployments. The repository contains helper support, but no completed cross-machine runtime validation is claimed here.

## Environment Variables

Application:

- `DEBUG_MODE`
- `LOCAL_HOST_RUN_ENV`
- `DATA_MAX_AGE_HOURS`
- `DATA_CLEANUP_INTERVAL_MINUTES`

ROS 2 / DDS:

- `ROS_DOMAIN_ID`
- `RMW_IMPLEMENTATION`
- `ROS_LOCALHOST_ONLY`
- `ROS_DISCOVERY_SERVER`
- `FASTDDS_UDP_ADDRESS`
- `FASTDDS_UDP_PORT`
- `VULCANEXUS_CONTAINER`
- `VULCANEXUS_SUB_CONTAINER`
- `HOST_WORKSPACE`
- `CONTAINER_WORKSPACE`

DDS Router:

- `CLOUD_PUBLIC_HOST`
- `CLOUD_LISTEN_IP`
- `ROS_DOMAIN_ID_VALUE`
- `WAN_PORT`
- `DDSROUTER_CONTAINER`
- `HOST_CONFIG_PATH`

FIWARE:

- `IGENIUS_FIWARE_BROKER_URL`
- `IGENIUS_FIWARE_CONTEXT_URL`

## FIWARE

FIWARE / Orion-LD is optional and separate from the trajectory transport. It stores execution metadata and lifecycle/status information through NGSI-LD REST. Raw trajectory delivery remains ROS 2 / DDS, and this repository does not implement the DDS-NGSI-LD Enabler.

## Intentionally Manual

- Supplying any private credentials or tokens
- Installing camera vendor SDKs such as ZED SDK when native depth support is needed
- Installing/configuring the robot SDK and validating safety limits
- Choosing the DDS discovery or routing strategy for the target network topology
- Opening deployment-specific firewall/security-group ports such as `14520`, `45678`, `9002`, `1026`, and `5050`
