# Deployment Architecture

This document captures the smallest deployment-layer change needed to make a
fresh machine reproducible while preserving the existing Push-to-Robot path.

## 1. Current Deployment Architecture

The current application path is:

```text
VILMA UI
-> data/_runtime/vulcanexus/last_cartesian_push.csv
-> scripts/vulcanexus/docker_publish_traj.sh
-> Docker container: vulcanexus_humble
-> ROS 2 topic: /learned_trajectory
-> Zenoh cloud bridge
-> Zenoh edge bridge
-> Edge ROS 2 subscriber / robot backend
```

Important current behavior:

- Push-to-Robot is implemented in application code and reuses
  `scripts/vulcanexus/docker_publish_traj.sh`.
- That script expects a Docker container named `vulcanexus_humble` unless
  `VULCANEXUS_CONTAINER` overrides it.
- The script publishes `geometry_msgs/msg/PoseArray` on `/learned_trajectory`.
- The tested ROS settings are `ROS_DOMAIN_ID=42`,
  `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, and `ROS_LOCALHOST_ONLY=0`.
- Zenoh bridge containers were previously created manually as `zenoh_cloud` and
  `zenoh_edge`.
- The Fast DDS Discovery Server workflow is optional and already wrapped by
  `scripts/vulcanexus/docker_run_fastdds_discovery_server.sh`.

## 2. Dependencies Currently External

Before this deployment update, a fresh host still needed manual work for:

- Docker Engine and Docker Compose plugin.
- Private MyWai feed credentials for app builds and local `uv sync`.
- Vulcanexus Humble runtime container.
- Zenoh DDS bridge containers.
- Optional Fast DDS Discovery Server startup.
- Edge-side ROS 2 Humble when running host-side receivers.
- Robot-side Fairino SDK and safety-specific robot backend configuration.

## 3. Proposed Deployment Architecture

The proposed default server deployment is Compose-managed:

```text
docker compose up -d
  -> vilma-agent
  -> vulcanexus_humble
  -> zenoh_cloud
```

Optional profiles are available for non-default roles:

```text
docker compose --profile edge up -d
  -> zenoh_edge
  -> edge_receiver

docker compose --profile fastdds up -d fastdds_discovery_server
  -> Fast DDS Discovery Server
```

This keeps the existing names, images, host networking, topic names, scripts,
and runtime defaults. No Push-to-Robot Python code is changed.

## 4. Required Docker Compose Changes

`docker-compose.yml` now defines:

- `vilma-agent`: existing application service.
- `vulcanexus_humble`: `eprosima/vulcanexus:humble-desktop`, host networking,
  repo mounted at `/workspace/vilma-agent`, command `sleep infinity`.
- `zenoh_cloud`: `eclipse/zenoh-bridge-dds:latest`, host networking,
  command `-d 42 -l tcp/0.0.0.0:11811`.
- `zenoh_edge`: optional `edge` profile, command
  `-d 42 -e ${ZENOH_EDGE_CONNECT}`.
- `edge_receiver`: optional `edge` profile running the existing
  `scripts/vulcanexus/run_edge_receiver.sh`.
- `fastdds_discovery_server`: optional `fastdds` profile using the Vulcanexus
  image and `fastdds discovery`.

The `vilma-agent` service also mounts:

- `./data:/mywai/data` so generated Push-to-Robot CSV files are visible to the
  Vulcanexus bind mount.
- `/var/run/docker.sock:/var/run/docker.sock` so the existing
  `docker_publish_traj.sh` can control `vulcanexus_humble` from inside the app
  container.

## 5. Required Dockerfile / Startup Services

The only Dockerfile change is installing the Docker CLI package in the
application image. The app container uses the host Docker daemon through the
mounted socket; it does not run Docker-in-Docker.

No new application startup service is required. The existing `supervisord.conf`
still starts Streamlit and data cleanup.

## 6. Fresh Ubuntu 22.04 Steps

Server:

```bash
sudo apt update
sudo apt install -y git

git clone <repo>
cd vilma-agent
cp .env.example .env
```

Edit `.env` and set:

```env
MYWAI_ARTIFACTS_MAIL=<email>
MYWAI_ARTIFACTS_TOKEN=<token>
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ROS_LOCALHOST_ONLY=0
```

Install host prerequisites, then start the server stack:

```bash
./install_dependencies.sh --skip-uv-sync
docker compose --env-file .env up -d --build
```

The app is exposed on `http://<server>:9002`.

Edge machine:

```bash
sudo apt update
sudo apt install -y git

git clone <repo>
cd vilma-agent
cp .env.example .env
```

Set `ZENOH_EDGE_CONNECT` in `.env`, then run:

```bash
./install_dependencies.sh --skip-uv-sync
docker compose --env-file .env --profile edge up -d zenoh_edge edge_receiver
```

If the robot-specific Fairino backend is used, install and configure the
Fairino SDK on the robot PC as before.

## 7. Validation Plan

1. Confirm server containers:

   ```bash
   docker compose ps
   docker ps --filter name=vulcanexus_humble
   docker ps --filter name=zenoh_cloud
   ```

2. Confirm the app container can control the Vulcanexus container through the
   existing Docker-based publish path:

   ```bash
   docker exec vilma-agent docker ps --filter name=vulcanexus_humble
   ```

3. Confirm the shared CSV path:

   ```bash
   mkdir -p data/_runtime/vulcanexus
   printf 'x,y,z\n0.1,0.2,0.3\n' > data/_runtime/vulcanexus/last_cartesian_push.csv
   docker exec vilma-agent bash -lc \
     'CSV_RELATIVE_PATH=data/_runtime/vulcanexus/last_cartesian_push.csv REPEAT=3 RATE_HZ=1 ./scripts/vulcanexus/docker_publish_traj.sh'
   ```

4. Confirm Edge bridge/receiver:

   ```bash
   docker compose --profile edge ps
   docker logs zenoh_edge --tail 50
   docker logs vilma_edge_receiver --tail 50
   ```

5. Validate through the UI:

   - Open the VILMA UI.
   - Build or reuse a Cartesian trajectory.
   - Open Push to Robot.
   - Keep domain `42`, topic `/learned_trajectory`, repeat `10000`, rate `1`.
   - Click Push to Robot.
   - Confirm the same success message as the current workflow.
   - On Edge, confirm receipt of `/learned_trajectory` or generated receiver
     artifacts.

6. Optional same-LAN Fast DDS Discovery Server validation:

   ```bash
   FASTDDS_UDP_ADDRESS=<server-ip> \
   docker compose --env-file .env --profile fastdds up -d fastdds_discovery_server
   ```

   Then set `ROS_DISCOVERY_SERVER=<server-ip>:14520` on publisher/subscriber
   shells and repeat the publish test.
