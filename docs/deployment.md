# Deployment

I-Genius/VILMA separates the application from robot middleware deployment.

- Application: `Dockerfile`, `docker-compose.yml`, service `vilma-agent`
- ROS 2 / Vulcanexus / Fast DDS helpers: `scripts/vulcanexus/`
- FIWARE / Orion-LD: optional northbound metadata/status layer in `docker-compose.fiware.yml`

The reusable module exposes standard ROS 2 / DDS interfaces. DDS discovery and network routing are deployment-specific and are configured according to the target network topology.

## Application Deployment

Start the app:

```bash
docker compose up -d --build
```

The current base Compose service set is:

```text
vilma-agent
```

The app maps host port `9002` to Streamlit port `8504`:

```text
http://localhost:9002
```

Useful commands:

```bash
docker compose config --quiet
docker compose config --services
docker compose logs -f vilma-agent
docker compose stop vilma-agent
```

## GPU Deployment

On GPU hosts with NVIDIA Container Toolkit:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Expected service set:

```text
vilma-agent
```

## ROS 2 / DDS Trajectory Contract

Vulcanexus / ROS 2 / Fast DDS carries robot trajectory communication.

| Setting | Value |
|---|---|
| `ROS_DOMAIN_ID` | `42` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` |
| Topic | `/learned_trajectory` |
| Message | `geometry_msgs/msg/PoseArray` |
| Optional status topic | `/trajectory_status` |

The application generates a Cartesian trajectory artifact. The publisher helper converts that artifact to a `geometry_msgs/msg/PoseArray` and publishes it on `/learned_trajectory`.

Publisher-side helper files:

- `scripts/vulcanexus/docker_publish_traj.sh`
- `scripts/vulcanexus/traj_pose_array_pub.py`

Edge-side helper files:

- `scripts/vulcanexus/run_edge_receiver.sh`
- `scripts/vulcanexus/edge_receive_posearray.py`
- `scripts/vulcanexus/run_edge_executor.sh`
- `scripts/vulcanexus/traj_pose_array_executor.py`

## DDS Discovery Options

DDS network deployment is deployment-specific. Valid DDS-native options include:

- native DDS discovery on a reachable LAN
- Fast DDS Discovery Server
- DDS Router for routed server-to-edge deployments
- Fast DDS WAN TCP profiles where supported by the helper scripts

### Native DDS Discovery

Use matching ROS settings on publisher and receiver:

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
```

Then run the receiver on the edge host:

```bash
bash scripts/vulcanexus/run_edge_receiver.sh
```

### Fast DDS Discovery Server

Start the Discovery Server in the Vulcanexus container:

```bash
FASTDDS_UDP_ADDRESS=<server-ip> \
FASTDDS_UDP_PORT=14520 \
bash scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

Set this on publisher and receiver shells:

```bash
export ROS_DISCOVERY_SERVER=<server-ip>:14520
```

### DDS Router

DDS Router is a DDS-native option for routed server-to-edge deployments. The repository provides templates and helpers:

- `scripts/vulcanexus/ddsrouter_cloud.template.yaml`
- `scripts/vulcanexus/ddsrouter_edge.template.yaml`
- `scripts/vulcanexus/render_ddsrouter_wan_config.sh`
- `scripts/vulcanexus/docker_run_ddsrouter.sh`

Default DDS Router WAN TCP port:

```text
45678
```

Render cloud and edge configs:

```bash
CLOUD_PUBLIC_HOST=<public-host-or-ip> bash scripts/vulcanexus/render_ddsrouter_wan_config.sh cloud
CLOUD_PUBLIC_HOST=<public-host-or-ip> bash scripts/vulcanexus/render_ddsrouter_wan_config.sh edge
```

Run a rendered config inside the configured Vulcanexus container:

```bash
HOST_CONFIG_PATH=/tmp/ddsrouter_cloud.yaml bash scripts/vulcanexus/docker_run_ddsrouter.sh
```

Do not treat DDS Router as the validated production default unless a deployment has actually executed and recorded a cross-machine runtime test.

## FIWARE / Orion-LD

FIWARE is separate from robot trajectory transport:

```bash
docker compose -f docker-compose.fiware.yml up -d
```

FIWARE / Orion-LD stores execution metadata and lifecycle/status information. It does not carry raw robot trajectories. Raw trajectory delivery remains ROS 2 / DDS. This repository does not implement the DDS-NGSI-LD Enabler.

## Ports

| Port | Role |
|---|---|
| `8504` | Streamlit process inside the app |
| `9002 -> 8504` | Docker-published application |
| `14520` | Fast DDS Discovery Server default UDP port |
| `45678` | DDS Router WAN TCP default |
| `1026` | FIWARE Orion-LD |
| `5050` | Robot relay where applicable |
