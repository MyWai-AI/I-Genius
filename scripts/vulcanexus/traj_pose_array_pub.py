#!/usr/bin/env python3

import argparse
import csv
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


RELIABILITY_BY_NAME = {
    "reliable": ReliabilityPolicy.RELIABLE,
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
}

DURABILITY_BY_NAME = {
    "volatile": DurabilityPolicy.VOLATILE,
    "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a geometry_msgs/PoseArray built from a CSV trajectory."
    )
    parser.add_argument("--csv-path", required=True, help="CSV file with x,y,z columns.")
    parser.add_argument("--topic", default="/learned_trajectory")
    parser.add_argument("--frame-id", default="camera_frame")
    parser.add_argument("--node-name", default="traj_pose_array_pub")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to publish.")
    parser.add_argument("--rate-hz", type=float, default=2.0, help="Loop rate when repeat > 1.")
    parser.add_argument(
        "--wait-for-subscriber-sec",
        type=float,
        default=10.0,
        help="Wait this long for at least one subscriber before publishing.",
    )
    parser.add_argument("--qos-depth", type=int, default=10)
    parser.add_argument(
        "--qos-reliability",
        choices=sorted(RELIABILITY_BY_NAME),
        default="reliable",
    )
    parser.add_argument(
        "--qos-durability",
        choices=sorted(DURABILITY_BY_NAME),
        default="volatile",
    )
    return parser.parse_args()


def build_pose_array(csv_path: Path, frame_id: str) -> PoseArray:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    msg = PoseArray()
    msg.header.frame_id = frame_id

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"x", "y", "z"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain x,y,z columns; found {reader.fieldnames!r}"
            )

        for row in reader:
            pose = Pose()
            pose.position.x = float(row["x"])
            pose.position.y = float(row["y"])
            pose.position.z = float(row["z"])
            pose.orientation.w = 1.0
            msg.poses.append(pose)

    if not msg.poses:
        raise ValueError(f"CSV contains no trajectory points: {csv_path}")

    return msg


def wait_for_subscriber(node: Node, topic: str, timeout_sec: float) -> int:
    deadline = time.monotonic() + max(timeout_sec, 0.0)

    while time.monotonic() < deadline:
        publishers = node.count_subscribers(topic)
        if publishers > 0:
            return publishers
        rclpy.spin_once(node, timeout_sec=0.1)

    return node.count_subscribers(topic)


def main() -> int:
    args = parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be > 0")

    csv_path = Path(args.csv_path).expanduser()
    pose_array = build_pose_array(csv_path, args.frame_id)

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=args.qos_depth,
        reliability=RELIABILITY_BY_NAME[args.qos_reliability],
        durability=DURABILITY_BY_NAME[args.qos_durability],
    )

    rclpy.init()
    node = Node(args.node_name)
    publisher = node.create_publisher(PoseArray, args.topic, qos)

    try:
        subscriber_count = wait_for_subscriber(
            node, args.topic, args.wait_for_subscriber_sec
        )
        if subscriber_count == 0:
            node.get_logger().warn(
                f"No subscribers discovered on {args.topic} after "
                f"{args.wait_for_subscriber_sec:.1f}s; publishing anyway."
            )
        else:
            node.get_logger().info(
                f"Discovered {subscriber_count} subscriber(s) on {args.topic}."
            )

        period_sec = 1.0 / args.rate_hz
        for index in range(args.repeat):
            pose_array.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(pose_array)
            node.get_logger().info(
                f"Published {index + 1}/{args.repeat} PoseArray with "
                f"{len(pose_array.poses)} poses from {csv_path}."
            )
            rclpy.spin_once(node, timeout_sec=0.05)
            if index + 1 < args.repeat:
                time.sleep(period_sec)

        rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
