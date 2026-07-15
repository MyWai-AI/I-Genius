# Troubleshooting

## Base Compose Does Not Match the Documentation

Validate the actual app service set:

```bash
docker compose config --quiet
docker compose config --services
```

Expected service:

```text
vilma-agent
```

The base Compose file does not start middleware helpers. Start ROS 2 / Vulcanexus / Fast DDS tooling with the scripts under `scripts/vulcanexus/`.

## GPU Compose Fails

Validate the overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --services
```

Expected service:

```text
vilma-agent
```

If the container starts but PyTorch or Ultralytics cannot see the GPU, check that the NVIDIA Container Toolkit is installed on the host.

## App Port Is Not Reachable

The Compose app maps host port `9002` to Streamlit port `8504`.

Check:

```bash
docker compose ps
docker compose logs --tail 100 vilma-agent
ss -ltnp | grep -E ':(8504|9002)\b'
```

## Push to Robot Fails: No Subscriber

Confirm the canonical ROS 2 / DDS settings on publisher and receiver:

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

Confirm the receiver is listening for the trajectory topic:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
ros2 topic list
```

Expected trajectory contract:

```text
/learned_trajectory
geometry_msgs/msg/PoseArray
```

## Fast DDS Discovery Server

The helper default is `14520`.

Start:

```bash
FASTDDS_UDP_ADDRESS=<server-ip> \
FASTDDS_UDP_PORT=14520 \
bash scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

On publisher and receiver:

```bash
export ROS_DISCOVERY_SERVER=<server-ip>:14520
```

If discovery still fails, check firewall rules and confirm both machines can reach the selected server IP and port.

## DDS Router

DDS Router is available as a DDS-native option for routed server-to-edge deployments.

Render configs:

```bash
CLOUD_PUBLIC_HOST=<public-host-or-ip> bash scripts/vulcanexus/render_ddsrouter_wan_config.sh cloud
CLOUD_PUBLIC_HOST=<public-host-or-ip> bash scripts/vulcanexus/render_ddsrouter_wan_config.sh edge
```

Run:

```bash
HOST_CONFIG_PATH=/tmp/ddsrouter_cloud.yaml bash scripts/vulcanexus/docker_run_ddsrouter.sh
```

Default WAN TCP port:

```text
45678
```

Do not change this port unless the deployment topology requires it and both rendered configs are updated together.

## Edge Receiver Does Not Write Artifacts

Run the receiver directly and watch the logs:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
bash scripts/vulcanexus/run_edge_receiver.sh
```

The receiver writes:

```text
edge_receiver_output/latest_received_trajectory.csv
edge_receiver_output/latest_received_metadata.json
```

If a Discovery Server is used, also set:

```bash
export ROS_DISCOVERY_SERVER=<server-ip>:14520
```

## FIWARE / Orion-LD

FIWARE is optional and separate from trajectory delivery:

```bash
docker compose -f docker-compose.fiware.yml config --quiet
docker compose -f docker-compose.fiware.yml up -d
```

Orion-LD listens on `http://localhost:1026`. It stores execution metadata and lifecycle/status information. It does not carry raw robot trajectories.

## Port Conflicts

Common ports:

```bash
ss -ltnp | grep -E ':(8504|9002|14520|45678|5050|1026)\b'
```

Resolve conflicts according to the deployment-specific network plan.

## Robot Execution

Physical robot motion is outside the reusable DDS contract. Confirm the robot SDK, safety limits, workspace bounds, and operator procedure on the robot PC before enabling any physical execution path.
