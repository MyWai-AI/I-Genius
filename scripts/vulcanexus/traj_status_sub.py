#!/usr/bin/env python3

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a std_msgs/String status message and print it."
    )
    parser.add_argument("--topic", default="/trajectory_status")
    parser.add_argument("--node-name", default="traj_status_sub")
    parser.add_argument("--timeout-sec", type=float, default=8.0)
    return parser.parse_args()


class StatusSubscriber(Node):
    def __init__(self, topic: str, node_name: str) -> None:
        super().__init__(node_name)
        self.message = None
        self.subscription = self.create_subscription(
            String, topic, self._callback, 10
        )
        self.get_logger().info(f"Waiting for status messages on {topic}")

    def _callback(self, msg: String) -> None:
        self.message = msg.data
        self.get_logger().info(f"Received status on topic: {msg.data}")


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = StatusSubscriber(args.topic, args.node_name)
    deadline = time.monotonic() + max(args.timeout_sec, 0.0)
    try:
        while time.monotonic() < deadline and node.message is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.message is None:
            node.get_logger().warning(
                f"No status received on {args.topic} after {args.timeout_sec:.1f}s."
            )
            return 1
        print(node.message)
        try:
            json.loads(node.message)
        except json.JSONDecodeError:
            pass
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
