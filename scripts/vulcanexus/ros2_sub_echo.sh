#!/usr/bin/env bash

set -euo pipefail

SETUP_FILE="${ROS_SETUP_FILE:-/opt/ros/humble/setup.bash}"
TOPIC="${1:-/learned_trajectory}"
MSG_TYPE="${2:-geometry_msgs/msg/PoseArray}"

if [ ! -f "$SETUP_FILE" ]; then
  echo "ROS setup file not found: $SETUP_FILE" >&2
  exit 1
fi

source "$SETUP_FILE"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

echo "ROS setup: $SETUP_FILE"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
if [ -n "${ROS_DISCOVERY_SERVER:-}" ]; then
  echo "ROS_DISCOVERY_SERVER=$ROS_DISCOVERY_SERVER"
fi

shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

ros2 topic echo "$TOPIC" "$MSG_TYPE" "$@"
