# Vulcanexus Cross-Machine Trajectory Test

This document defines the reusable cross-machine validation procedure for `/learned_trajectory`.

The test contract is:

| Setting | Value |
|---|---|
| `ROS_DOMAIN_ID` | `42` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` |
| Topic | `/learned_trajectory` |
| Message | `geometry_msgs/msg/PoseArray` |

## Same-LAN Native DDS

On the edge machine:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
bash scripts/vulcanexus/run_edge_receiver.sh
```

On the publisher machine:

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
bash scripts/vulcanexus/docker_publish_traj.sh
```

Pass criteria:

- the publisher sends `geometry_msgs/msg/PoseArray` on `/learned_trajectory`
- the edge receiver writes `latest_received_trajectory.csv`
- the edge receiver writes `latest_received_metadata.json`

## Fast DDS Discovery Server

On the Discovery Server host:

```bash
FASTDDS_UDP_ADDRESS=<server-ip> \
FASTDDS_UDP_PORT=14520 \
bash scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

On publisher and edge receiver shells:

```bash
export ROS_DISCOVERY_SERVER=<server-ip>:14520
```

Repeat the same publish/receive test.

## DDS Router

DDS Router is a DDS-native option for routed server-to-edge deployments.

Render configs:

```bash
CLOUD_PUBLIC_HOST=<public-host-or-ip> bash scripts/vulcanexus/render_ddsrouter_wan_config.sh cloud
CLOUD_PUBLIC_HOST=<public-host-or-ip> bash scripts/vulcanexus/render_ddsrouter_wan_config.sh edge
```

Default WAN TCP port:

```text
45678
```

Run each rendered config in the selected Vulcanexus environment:

```bash
HOST_CONFIG_PATH=/tmp/ddsrouter_cloud.yaml bash scripts/vulcanexus/docker_run_ddsrouter.sh
HOST_CONFIG_PATH=/tmp/ddsrouter_edge.yaml bash scripts/vulcanexus/docker_run_ddsrouter.sh
```

Pass criteria are the same as the native DDS test. Do not claim DDS Router is the validated production default until this test is actually executed across the intended machines and the result is recorded.

## FIWARE Separation Check

FIWARE / Orion-LD is not part of raw trajectory delivery. If FIWARE is enabled, verify only metadata/status updates through:

```bash
docker compose -f docker-compose.fiware.yml config --quiet
docker compose -f docker-compose.fiware.yml up -d
```

Raw trajectory delivery remains on ROS 2 / DDS.
