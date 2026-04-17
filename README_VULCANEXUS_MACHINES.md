# Vulcanexus LAN Integration

This guide describes the proven same-LAN trajectory delivery flow for this repo.

It is the validated path for sending `geometry_msgs/msg/PoseArray` on
`/learned_trajectory` from VILMA to another ROS 2 machine using Vulcanexus and a
Fast DDS Discovery Server.

## Architecture

The intended split is:

- `VILMA UI`: authoring and orchestration
- `Vulcanexus LAN`: southbound transport adapter
- `edge receiver`: reusable subscriber or artifact intake
- `robot backend`: robot-specific execution layer

## Assumptions

- both machines are on the same LAN
- the publisher host runs Docker and this repo
- the subscriber host has ROS 2 Humble
- both sides use `rmw_fastrtps_cpp`
- the tested Discovery Server port is `14520`

Replace these placeholders in the commands below:

- `<repo-root>`: local checkout of this repo
- `<server-ip>`: LAN IP of the Discovery Server host

The known-good example used during validation was:

- `<server-ip>` = `192.168.0.10`
- `ROS_DISCOVERY_SERVER=192.168.0.10:14520`

## Common ROS Settings

Use these on both sides:

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DISCOVERY_SERVER=<server-ip>:14520
```

## Publisher Host

Recreate the Vulcanexus container:

```bash
cd <repo-root>
docker rm -f vulcanexus_humble || true
docker run -d --name vulcanexus_humble --net=host -v <repo-root>:/workspace/vilma-agent eprosima/vulcanexus:humble-desktop sleep infinity
```

Start the Discovery Server and keep that terminal open:

```bash
cd <repo-root>
FASTDDS_FORCE_RESTART=1 FASTDDS_UDP_ADDRESS=<server-ip> FASTDDS_UDP_PORT=14520 ./scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

## Subscriber Host

If this repo is available on the subscriber machine, prefer the reusable edge
receiver:

```bash
cd <repo-root>
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DISCOVERY_SERVER=<server-ip>:14520
OUTPUT_DIR="$PWD/edge_receiver_output" ARTIFACT_PREFIX=latest_received ./scripts/vulcanexus/run_edge_receiver.sh
```

If the repo is not available there, use this validation subscriber:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DISCOVERY_SERVER=<server-ip>:14520
python3 - <<'PY'
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray

class TrajectorySubscriber(Node):
    def __init__(self):
        super().__init__('traj_sub_peer')
        self.create_subscription(PoseArray, '/learned_trajectory', self.callback, 10)
        self.get_logger().info('Waiting for /learned_trajectory')

    def callback(self, msg):
        self.get_logger().info(
            f"Received PoseArray with {len(msg.poses)} poses; frame_id={msg.header.frame_id!r}"
        )

rclpy.init()
node = TrajectorySubscriber()
rclpy.spin(node)
PY
```

## Publish From the CLI

Use the helper script from the publisher host:

```bash
cd <repo-root>
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DISCOVERY_SERVER=<server-ip>:14520
REPEAT=40 RATE_HZ=2 WAIT_FOR_SUBSCRIBER_SEC=15 ./scripts/vulcanexus/docker_publish_traj.sh
```

## Publish From the UI

Start the new UI on the publisher host:

```bash
cd <repo-root>
source .venv/bin/activate
export ROS_DOMAIN_ID=42
export ROS_DISCOVERY_SERVER=<server-ip>:14520
python -m streamlit run src/streamlit_template/new_ui/pages/Common/landing_page.py --server.address 0.0.0.0 --server.port 8505
```

In the robot step:

1. Open `Action`
2. Choose `Vulcanexus LAN (PoseArray)`
3. Keep:
   - `Topic`: `/learned_trajectory`
   - `Domain ID`: `42`
   - `Discovery Server`: `<server-ip>:14520`
4. Click `Push to Robot`

## Success Criteria

The flow is working when:

- the UI reports a successful publish
- the subscriber prints `Received PoseArray ...`
- if the reusable edge receiver is used, it also writes artifacts and can
  publish `/trajectory_status`

## Troubleshooting

- Recreate `vulcanexus_humble` with `--net=host` and the repo bind mount before
  a fresh test.
- Source `/opt/vulcanexus/humble/setup.bash` only inside the Vulcanexus
  container.
- Source `/opt/ros/humble/setup.bash` on plain ROS 2 subscriber hosts.
- If no trajectory arrives, first smoke-test Discovery Server with
  `demo_nodes_cpp talker/listener`.
- This README is for same-LAN use. For different networks, use a VPN-based path
  rather than raw public DDS exposure.

## Related Files

- [scripts/vulcanexus/README.md](/home/vvijaykumar/vilma-agent/scripts/vulcanexus/README.md)
- [docs/vulcanexus_cross_machine_test.md](/home/vvijaykumar/vilma-agent/docs/vulcanexus_cross_machine_test.md)
- [docs/edge_receiver_contract.md](/home/vvijaykumar/vilma-agent/docs/edge_receiver_contract.md)
