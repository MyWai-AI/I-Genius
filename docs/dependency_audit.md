# Dependency Audit

This audit covers a fresh Ubuntu 22.04 checkout and keeps the existing
Push-to-Robot, Vulcanexus, Fast DDS, and Zenoh behavior intact.

## Host Prerequisites

| Area | Requirement | Used by |
|---|---|---|
| OS | Ubuntu 22.04 Jammy | validated host target |
| Python | Python 3.10.14 or newer, `python3-venv`, `python3-pip` | local `uv sync`, Streamlit app |
| Package manager | `uv` | Python dependency sync |
| Build/system libs | `build-essential`, `git`, `curl`, `ca-certificates`, `gnupg`, `ffmpeg`, `libgl1`, `libglib2.0-0`, `iproute2` | Python wheels, OpenCV/media, diagnostics |
| Docker | Docker Engine | app container, Vulcanexus container, Zenoh bridges |
| Docker Compose | Docker Compose plugin (`docker compose`) | existing `docker-compose.yml` app deployment |
| ROS 2 on host | Optional on the server; required on Edge if running `run_edge_receiver.sh` outside Docker | Edge receiver and ROS debug commands |
| Fast DDS | `rmw_fastrtps_cpp` and optional Fast DDS Discovery Server | current ROS 2 transport defaults |

## Python Dependencies

The Python environment is defined by `pyproject.toml` and `uv.lock`.
Notable runtime dependencies include:

- Streamlit UI: `streamlit`, `streamlit-post-message`, Streamlit helper components.
- Computer vision and trajectory work: `numpy`, `opencv-python-headless`, `mediapipe`, `ultralytics`, `transformers`, `timm`, `scipy`, `scikit-learn`, `plotly`.
- Robot/model tooling: `ikpy`, `urdfpy`, `trimesh`, `pycollada`, `shapely`, `networkx`.
- Communication helpers: `cyclonedds`, `flask`, `requests`, `python-dotenv`.
- MyWai private package: `mywai-python-integration-kit==0.5.15+build.11277`.
- Optional RealSense: `pyrealsense2` via `.[realsense]`.

`mywai-python-integration-kit` comes from the private Azure DevOps feed, so
`MYWAI_ARTIFACTS_MAIL` and `MYWAI_ARTIFACTS_TOKEN` are required before `uv sync`
can complete.

## Required Containers

| Container | Image | Role | Notes |
|---|---|---|---|
| `vilma-agent` | built from local `Dockerfile` | Streamlit deployment container | existing `docker-compose.yml`, port `9002:8504` |
| `vulcanexus_humble` | `eprosima/vulcanexus:humble-desktop` | ROS 2/Vulcanexus publisher runtime | expected by `scripts/vulcanexus/docker_publish_traj.sh` |
| `zenoh_cloud` | `eclipse/zenoh-bridge-dds:latest` | server-side DDS bridge listener | tested command: `-d 42 -l tcp/0.0.0.0:11811` |
| `zenoh_edge` | `eclipse/zenoh-bridge-dds:latest` | Edge-side DDS bridge connector | tested endpoint: `tcp/sestrilevante.platform.myw.ai:11811` |
| `fiware-mongo`, `fiware-orion-ld` | pinned in `docker-compose.fiware.yml` | optional FIWARE integration | not required for Push-to-Robot |
| `edge_receiver` | `osrf/ros:humble-ros-base` | optional generic Edge receiver | Compose `edge` profile |
| `fastdds_discovery_server` | `eprosima/vulcanexus:humble-desktop` | optional Fast DDS Discovery Server | Compose `fastdds` profile |

`bootstrap.sh` now creates or starts the existing `vulcanexus_humble`,
`zenoh_cloud`, and `zenoh_edge` containers without replacing manually-created
containers of the same name.

`docker-compose.yml` now also defines the fresh-machine server stack directly:
`vilma-agent`, `vulcanexus_humble`, and `zenoh_cloud`.

## ROS 2 / Robot Communication

The tested Push-to-Robot path is:

```text
VILMA UI
-> data/_runtime/vulcanexus/last_cartesian_push.csv
-> scripts/vulcanexus/docker_publish_traj.sh
-> vulcanexus_humble
-> ROS 2 /learned_trajectory
-> zenoh_cloud
-> zenoh_edge
-> Edge ROS 2 /learned_trajectory
-> robot receiver/backend
```

Current defaults:

| Setting | Value |
|---|---|
| Topic | `/learned_trajectory` |
| Message | `geometry_msgs/msg/PoseArray` |
| Status topic | `/trajectory_status` optional |
| `ROS_DOMAIN_ID` | `42` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` |
| CSV path | `data/_runtime/vulcanexus/last_cartesian_push.csv` |

The robot relay path is separate: `robot_relay/relay_server.py` is a Flask
service for the robot PC, listens on port `5050`, and needs the Fairino Python
SDK before real robot motion lines are enabled.

## Environment Variables

Required for dependency installation:

- `MYWAI_ARTIFACTS_MAIL`
- `MYWAI_ARTIFACTS_TOKEN`

Core app variables:

- `DEBUG_MODE`
- `END_POINT`
- `LOCAL_HOST_RUN_ENV`
- `MYWAI_USER`
- `MYWAI_PASSWORD`
- `MYWAI_API_ENDPOINT`
- `MYWAI_ENDPOINT`
- `MYWAI_TOKEN_PLATFORM`
- `DATA_ROOT`
- `DATA_MAX_AGE_HOURS`
- `DATA_CLEANUP_INTERVAL_MINUTES`

ROS/Vulcanexus/Zenoh variables:

- `ROS_DOMAIN_ID`
- `RMW_IMPLEMENTATION`
- `ROS_LOCALHOST_ONLY`
- `ROS_DISCOVERY_SERVER`
- `VILMA_VULCANEXUS_DISCOVERY_SERVER`
- `VILMA_VULCANEXUS_REPEAT`
- `VILMA_VULCANEXUS_RATE_HZ`
- `VILMA_VULCANEXUS_STATUS_WAIT_SEC`
- `HOST_WORKSPACE`
- `CONTAINER_WORKSPACE`
- `VULCANEXUS_CONTAINER`
- `VULCANEXUS_SUB_CONTAINER`
- `VULCANEXUS_IMAGE`
- `ZENOH_BRIDGE_DDS_IMAGE`
- `ZENOH_CLOUD_CONTAINER`
- `ZENOH_EDGE_CONTAINER`
- `ZENOH_CLOUD_LISTEN`
- `ZENOH_EDGE_CONNECT`

Optional/feature-specific variables:

- `HF_TOKEN`
- `VILMA_FIWARE_BROKER_URL`
- `VILMA_OBJECT_CONF`
- `VILMA_ANCHOR_CLASS`
- `ROBOT_IP`
- `ROBOT_RELAY_HOST`
- `ROBOT_RELAY_PORT`

## Manual Setup Found Before Automation

- Install Docker Engine and Docker Compose plugin.
- Enable and start Docker.
- Install Python, `uv`, and system libraries for OpenCV/media processing.
- Create `.env` and provide private MyWai feed credentials.
- Create `.netrc` for the Azure DevOps Python feed.
- Run `uv sync`.
- Manually create/start `vulcanexus_humble`.
- Manually create/start `zenoh_cloud` on the server.
- Manually create/start `zenoh_edge` on the Edge machine.
- Set ROS environment variables consistently on server and Edge.
- Install ROS 2 Humble on the Edge machine when running host-side receivers.
- Start the Edge receiver or robot backend.

## Automated Now

- `.env` creation from `.env.example`.
- Filling `HOST_WORKSPACE` for the current checkout.
- Runtime directory creation.
- Ubuntu base package installation.
- Docker Engine and Compose plugin installation.
- Docker service enable/start.
- `uv` installation.
- uv-managed Python 3.10.14 installation when the host `python3` is older than the pyproject requirement.
- `.netrc` creation when MyWai feed credentials are present.
- `uv sync` when credentials are present.
- Optional ROS 2 Humble host install with `--install-ros2`.
- Idempotent startup of `vulcanexus_humble`, `zenoh_cloud`, and `zenoh_edge`.
- Compose-managed startup of the default server stack with `docker compose up -d`.

## Still Intentionally Manual

- Supplying private credentials and tokens.
- Installing camera vendor SDKs such as ZED SDK when native depth support is needed.
- Installing/configuring the Fairino Python SDK and confirming robot safety limits.
- Choosing the production Zenoh edge endpoint for a specific network.
- Opening firewall/security-group ports such as `11811`, `8504`, `8505`, and `5050`.
