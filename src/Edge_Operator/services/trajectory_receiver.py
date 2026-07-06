#!/usr/bin/env python3

import csv
import json
from pathlib import Path

import rclpy
import yaml

from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.yaml"
DEFAULT_TRAJECTORY_CSV = ROOT / "data" / "trajectories" / "received_trajectory.csv"


def load_trajectory_csv_path():
    if not SETTINGS_PATH.exists():
        return DEFAULT_TRAJECTORY_CSV

    settings = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    configured = settings.get("files", {}).get("trajectory_csv")
    if not configured:
        return DEFAULT_TRAJECTORY_CSV

    path = Path(configured)
    if not path.is_absolute():
        path = ROOT / path
    return path


class TrajectoryReceiver(Node):

    def __init__(self):
        super().__init__("trajectory_receiver")
        self.trajectory_csv = load_trajectory_csv_path()
        self.latest_metadata = None

        self.pose_subscription = self.create_subscription(
            PoseArray,
            "/learned_trajectory",
            self.callback,
            10,
        )
        self.metadata_subscription = self.create_subscription(
            String,
            "/learned_trajectory_metadata",
            self.metadata_callback,
            10,
        )

        self.get_logger().info(
            "Waiting for /learned_trajectory ..."
        )
        self.get_logger().info(
            "Waiting for /learned_trajectory_metadata ..."
        )

    def metadata_callback(self, msg):
        try:
            metadata = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(
                f"Ignoring invalid trajectory metadata JSON: {exc}"
            )
            return

        rows = metadata.get("rows")
        if not isinstance(rows, list):
            self.get_logger().warning(
                "Ignoring trajectory metadata without a rows list"
            )
            return

        self.latest_metadata = metadata
        self.get_logger().info(
            f"Received trajectory metadata with {len(rows)} rows"
        )

    def callback(self, msg):

        count = len(msg.poses)

        self.get_logger().info(
            f"Received trajectory with {count} poses"
        )

        filename = self.trajectory_csv
        filename.parent.mkdir(parents=True, exist_ok=True)

        metadata_rows = []
        if self.latest_metadata is not None:
            rows = self.latest_metadata.get("rows", [])
            if len(rows) == count:
                metadata_rows = rows
            else:
                self.get_logger().warning(
                    "Ignoring trajectory metadata because row count "
                    f"({len(rows)}) does not match pose count ({count})"
                )

        with filename.open("w", newline="", encoding="utf-8") as f:

            writer = csv.writer(
                f,
                quoting=csv.QUOTE_ALL,
            )

            writer.writerow(
                [
                    "x_mm",
                    "y_mm",
                    "z_mm",
                    "event",
                    "source_index",
                    "label",
                ]
            )

            if metadata_rows:
                for row in metadata_rows:
                    writer.writerow(
                        [
                            row.get("x_mm", ""),
                            row.get("y_mm", ""),
                            row.get("z_mm", ""),
                            row.get("event", ""),
                            row.get("source_index", ""),
                            row.get("label", ""),
                        ]
                    )
            else:
                for pose in msg.poses:

                    writer.writerow(
                        [
                            pose.position.x * 1000.0,
                            pose.position.y * 1000.0,
                            pose.position.z * 1000.0,
                            "",
                            "",
                            "",
                        ]
                    )

        self.get_logger().info(
            f"Saved {count} poses to {filename}"
        )


def main():

    rclpy.init()

    node = TrajectoryReceiver()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
