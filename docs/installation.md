# Installation

This repository has two separate runtime layers:

- the I-Genius/VILMA application, run locally or with Docker Compose
- ROS 2 / Vulcanexus / Fast DDS helpers under `scripts/vulcanexus/`, run only when publishing or receiving robot trajectories

The application Dockerfile does not install Vulcanexus. The base Compose file starts only the application service named `vilma-agent`.

## Prerequisites

- Ubuntu 22.04 or a compatible Linux host
- Python 3.10 or newer for local development
- `uv` for Python dependency management
- Docker Engine and the Docker Compose plugin for container deployment
- ROS 2 Humble or Vulcanexus on hosts that run the ROS 2 publisher or edge receiver directly
- NVIDIA Container Toolkit only when using `docker-compose.gpu.yml`

## Environment File

Create a local `.env` from the tracked template:

```bash
cp .env.template .env
```

The canonical ROS 2 / DDS defaults are:

```env
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ROS_LOCALHOST_ONLY=0
```

Set `ROS_DISCOVERY_SERVER=<server-ip>:14520` only for deployments that use a Fast DDS Discovery Server.

## Local Application

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip uv
uv sync
uv run streamlit run src/streamlit_template/new_ui/pages/Common/landing_page.py \
  --server.port 8504 \
  --server.address 0.0.0.0
```

Open:

```text
http://localhost:8504
```

## Docker Application

Start the application service:

```bash
docker compose up -d --build
```

The service is:

```text
vilma-agent
```

The app is available at:

```text
http://localhost:9002
```

To validate the service set:

```bash
docker compose config --services
```

Expected output:

```text
vilma-agent
```

## GPU Overlay

Use the GPU overlay only on hosts with NVIDIA Container Toolkit installed:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The overlay augments `vilma-agent` with GPU visibility. It does not add middleware services.

## ROS 2 / Vulcanexus Helpers

The reusable module exposes standard ROS 2 / DDS interfaces. DDS discovery and network routing are deployment-specific and are configured according to the target network topology.

Trajectory contract:

| Setting | Value |
|---|---|
| `ROS_DOMAIN_ID` | `42` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` |
| Topic | `/learned_trajectory` |
| Message | `geometry_msgs/msg/PoseArray` |

Publisher helper:

```bash
bash scripts/vulcanexus/docker_publish_traj.sh
```

Edge receiver helper:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
bash scripts/vulcanexus/run_edge_receiver.sh
```

Fast DDS Discovery Server helper:

```bash
FASTDDS_UDP_ADDRESS=<server-ip> \
FASTDDS_UDP_PORT=14520 \
bash scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

DDS Router is a DDS-native option for routed server-to-edge deployments:

```bash
CLOUD_PUBLIC_HOST=<public-host-or-ip> bash scripts/vulcanexus/render_ddsrouter_wan_config.sh cloud
CLOUD_PUBLIC_HOST=<public-host-or-ip> bash scripts/vulcanexus/render_ddsrouter_wan_config.sh edge
bash scripts/vulcanexus/docker_run_ddsrouter.sh
```

## FIWARE

FIWARE / Orion-LD is optional and separate from robot trajectory delivery:

```bash
docker compose -f docker-compose.fiware.yml up -d
```

Orion-LD listens on `http://localhost:1026`. FIWARE stores execution metadata and lifecycle/status information. Raw robot trajectory delivery remains on ROS 2 / DDS, and this repository does not implement the DDS-NGSI-LD Enabler.
