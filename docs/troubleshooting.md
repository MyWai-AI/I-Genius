# Troubleshooting

## `uv sync` Was Skipped

Cause: `MYWAI_ARTIFACTS_MAIL` and `MYWAI_ARTIFACTS_TOKEN` are missing.

Fix:

```bash
[ -f .env ] || cp .env.example .env
# edit .env and fill MYWAI_ARTIFACTS_MAIL / MYWAI_ARTIFACTS_TOKEN
./install_dependencies.sh --skip-system
```

## Docker Permission Denied

The setup scripts use `sudo docker` when the current user cannot access Docker.
For future shells without sudo:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Then verify:

```bash
docker ps
```

## Container Already Exists

`bootstrap.sh` preserves existing manual containers. If a container named
`vulcanexus_humble`, `zenoh_cloud`, or `zenoh_edge` already exists, setup starts
it instead of recreating it.

Check status:

```bash
docker ps -a --filter name=vulcanexus_humble
docker ps -a --filter name=zenoh_cloud
docker ps -a --filter name=zenoh_edge
```

If `docker compose up -d` reports that one of these container names is already
in use, the host has a pre-existing manually-created container. Either keep
using `./setup.sh --start-app`, which reuses existing containers, or migrate the
container to Compose during a maintenance window:

```bash
docker stop vulcanexus_humble zenoh_cloud
# remove only after confirming these are the old manually-created containers
docker rm vulcanexus_humble zenoh_cloud
docker compose --env-file .env up -d --build
```

## Push to Robot Fails: No Subscriber

Check that the server containers are running:

```bash
docker ps --filter name=vulcanexus_humble
docker ps --filter name=zenoh_cloud
docker logs zenoh_cloud --tail 50
```

If the app is running in the `vilma-agent` container, confirm it can access the
host Docker daemon:

```bash
docker exec vilma-agent docker ps --filter name=vulcanexus_humble
```

If this fails, check that `/var/run/docker.sock` is mounted by Compose and that
the image was rebuilt after the Docker CLI was added:

```bash
docker compose --env-file .env up -d --build vilma-agent
```

Check matching ROS settings on both machines:

```bash
echo "$ROS_DOMAIN_ID"
echo "$RMW_IMPLEMENTATION"
echo "$ROS_LOCALHOST_ONLY"
```

Expected values:

```text
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ROS_LOCALHOST_ONLY=0
```

## Edge Does Not Receive `/learned_trajectory`

On Edge:

```bash
docker ps --filter name=zenoh_edge
docker logs zenoh_edge --tail 50
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
ros2 topic list
```

If the Edge machine uses a different relay/server endpoint, update
`ZENOH_EDGE_CONNECT` in `.env` and rerun:

```bash
./bootstrap.sh --role edge --containers-only
```

If the old `zenoh_edge` container was created with the wrong endpoint, remove
and recreate only after confirming it is safe:

```bash
docker rm -f zenoh_edge
./bootstrap.sh --role edge --containers-only
```

## `HOST_WORKSPACE` Points to the Wrong Directory

This can happen if the checkout was moved after setup.

Fix:

```bash
./bootstrap.sh --env-only
```

Then inspect `.env` and confirm:

```env
HOST_WORKSPACE=/absolute/path/to/vilma-agent
```

## ROS 2 Setup File Missing on Edge

Install ROS 2 Humble host dependencies:

```bash
./install_dependencies.sh --skip-uv-sync --install-ros2
```

Then verify:

```bash
test -f /opt/ros/humble/setup.bash && echo ok
```

## Docker Image Pull Fails

Check network/proxy access to Docker Hub:

```bash
docker pull eprosima/vulcanexus:humble-desktop
docker pull eclipse/zenoh-bridge-dds:latest
```

If the host uses an HTTP proxy, configure Docker daemon proxy settings and rerun
`./bootstrap.sh --containers-only`.

## Port Conflicts

Common ports:

```bash
ss -ltnp | grep -E ':(8504|8505|9002|11811|5050|1026)\b'
```

Resolve the conflicting service or change the relevant app/bridge port in
`.env`.

## Robot Relay Works in Stub Mode Only

`robot_relay/relay_server.py` intentionally keeps Fairino SDK calls commented
until the robot PC is configured. Install the Fairino Python SDK v2 on the robot
PC, confirm safety parameters, then enable the SDK lines in that file for that
robot deployment.

## Useful Reset Commands

Start existing infrastructure:

```bash
./bootstrap.sh --role server --containers-only
```

View logs:

```bash
docker logs vulcanexus_humble --tail 50
docker logs zenoh_cloud --tail 50
docker logs zenoh_edge --tail 50
```

Manual publish smoke test:

```bash
CSV_RELATIVE_PATH=data/_runtime/vulcanexus/last_cartesian_push.csv \
REPEAT=10 \
RATE_HZ=1 \
ROS_DOMAIN_ID=42 \
RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
ROS_LOCALHOST_ONLY=0 \
./scripts/vulcanexus/docker_publish_traj.sh
```
