# Deployment

This repository now supports a fresh-machine setup without changing the
application runtime path or the existing Push-to-Robot implementation.

## Server Deployment

On the VILMA server:

```bash
git clone <repo>
cd vilma-agent
cp .env.example .env
```

Edit `.env` and set at least:

```env
MYWAI_ARTIFACTS_MAIL=you@example.com
MYWAI_ARTIFACTS_TOKEN=<token>
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ROS_LOCALHOST_ONLY=0
ZENOH_CLOUD_LISTEN=tcp/0.0.0.0:11811
```

Run:

```bash
./install_dependencies.sh --skip-uv-sync
docker compose --env-file .env up -d --build
```

This creates or starts:

- `vilma-agent`
- `vulcanexus_humble`
- `zenoh_cloud`

The containers use `restart: unless-stopped`, so they restart when Docker starts
after a reboot.

## Application Deployment

To restart only the existing Compose app after it has been built:

```bash
docker compose --env-file .env up -d vilma-agent
```

The app is available on host port `9002` by default.

Logs:

```bash
docker compose logs -f vilma-agent
```

Stop only the app:

```bash
docker compose stop vilma-agent
```

## Edge Deployment

On the Edge machine:

```bash
git clone <repo>
cd vilma-agent
cp .env.example .env
```

Set the Edge bridge endpoint in `.env`:

```env
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ROS_LOCALHOST_ONLY=0
ZENOH_EDGE_CONNECT=tcp/sestrilevante.platform.myw.ai:11811
```

Then run:

```bash
./install_dependencies.sh --skip-uv-sync
docker compose --env-file .env --profile edge up -d zenoh_edge edge_receiver
```

Alternatively, run the reusable Edge receiver directly on a ROS 2 host:

```bash
source /opt/ros/humble/setup.bash
bash ./scripts/vulcanexus/run_edge_receiver.sh
```

For a systemd deployment, adapt the existing examples:

- `scripts/vulcanexus/edge_receiver.service.example`
- `scripts/vulcanexus/edge_executor.service.example`

## Push-to-Robot Runtime Defaults

| Setting | Value |
|---|---|
| ROS domain | `42` |
| RMW | `rmw_fastrtps_cpp` |
| Topic | `/learned_trajectory` |
| Message | `geometry_msgs/msg/PoseArray` |
| Server bridge | `zenoh_cloud -d 42 -l tcp/0.0.0.0:11811` |
| Edge bridge | `zenoh_edge -d 42 -e <server-or-relay-endpoint>` |

## Compose Profiles

| Command | Starts |
|---|---|
| `docker compose up -d` | server stack: `vilma-agent`, `vulcanexus_humble`, `zenoh_cloud` |
| `docker compose --profile edge up -d` | server stack plus `zenoh_edge`, `edge_receiver` |
| `docker compose --profile fastdds up -d fastdds_discovery_server` | optional Fast DDS Discovery Server |

## Optional Fast DDS Discovery Server

The Zenoh path above is the tested default. For the same-LAN Fast DDS Discovery
Server workflow, keep using the existing helper:

```bash
FASTDDS_UDP_ADDRESS=<server-ip> \
docker compose --env-file .env --profile fastdds up -d fastdds_discovery_server
```

Then set this on publisher and subscriber shells:

```bash
export ROS_DISCOVERY_SERVER=<server-ip>:14520
```

## Optional FIWARE

FIWARE remains separate from Push-to-Robot:

```bash
docker compose -f docker-compose.fiware.yml up -d
```

Orion-LD listens on `http://localhost:1026`.

## Ports

| Port | Service |
|---|---|
| `8504` | Streamlit local process |
| `8505` | MJPEG stream when live stream is active |
| `9002` | Docker-published Streamlit app |
| `11811` | Zenoh bridge listener |
| `5050` | Robot relay Flask service |
| `1026` | Optional FIWARE Orion-LD |
