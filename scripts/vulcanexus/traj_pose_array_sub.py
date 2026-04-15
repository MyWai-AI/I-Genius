#!/usr/bin/env python3

import argparse

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscribe to a geometry_msgs/PoseArray topic and print messages."
    )
    parser.add_argument("--topic", default="/learned_trajectory")
    parser.add_argument("--node-name", default="traj_pose_array_sub")
    return parser.parse_args()


class TrajectorySubscriber(Node):
    def __init__(self, topic: str, node_name: str) -> None:
        super().__init__(node_name)
        self.subscription = self.create_subscription(
            PoseArray, topic, self._callback, 10
        )
        self.get_logger().info(f"Waiting for PoseArray messages on {topic}")

    def _callback(self, msg: PoseArray) -> None:
        self.get_logger().info(
            f"Received PoseArray with {len(msg.poses)} poses; "
            f"frame_id={msg.header.frame_id!r}"
        )
        print(msg)


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = TrajectorySubscriber(args.topic, args.node_name)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
