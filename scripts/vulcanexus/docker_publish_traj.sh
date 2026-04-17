#!/usr/bin/env bash

set -euo pipefail

CONTAINER="${VULCANEXUS_CONTAINER:-vulcanexus_humble}"
HOST_WORKSPACE="${HOST_WORKSPACE:-/home/vvijaykumar/vilma-agent}"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspace/vilma-agent}"
CSV_RELATIVE_PATH="${CSV_RELATIVE_PATH:-data/BAG Data/dmp/skill_reuse_traj.csv}"
TOPIC="${TOPIC:-/learned_trajectory}"
FRAME_ID="${FRAME_ID:-camera_frame}"
REPEAT="${REPEAT:-1}"
RATE_HZ="${RATE_HZ:-2.0}"
WAIT_FOR_SUBSCRIBER_SEC="${WAIT_FOR_SUBSCRIBER_SEC:-15}"
QOS_RELIABILITY="${QOS_RELIABILITY:-reliable}"
QOS_DURABILITY="${QOS_DURABILITY:-volatile}"

HOST_CSV_PATH="$HOST_WORKSPACE/$CSV_RELATIVE_PATH"
CONTAINER_CSV_PATH="$CONTAINER_WORKSPACE/$CSV_RELATIVE_PATH"
CONTAINER_SCRIPT_PATH="$CONTAINER_WORKSPACE/scripts/vulcanexus/traj_pose_array_pub.py"
FALLBACK_CONTAINER_DIR="/tmp/vilma-agent-vulcanexus-pub"

if [ ! -f "$HOST_CSV_PATH" ]; then
  echo "Host CSV file not found: $HOST_CSV_PATH" >&2
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
  echo "Starting container $CONTAINER"
  docker start "$CONTAINER" >/dev/null
fi

if docker exec -i "$CONTAINER" test -f "$CONTAINER_SCRIPT_PATH" \
  && docker exec -i "$CONTAINER" test -f "$CONTAINER_CSV_PATH"; then
  EFFECTIVE_CONTAINER_SCRIPT_PATH="$CONTAINER_SCRIPT_PATH"
  EFFECTIVE_CONTAINER_CSV_PATH="$CONTAINER_CSV_PATH"
else
  EFFECTIVE_CONTAINER_DIR="$FALLBACK_CONTAINER_DIR"
  EFFECTIVE_CONTAINER_SCRIPT_PATH="$EFFECTIVE_CONTAINER_DIR/traj_pose_array_pub.py"
  EFFECTIVE_CONTAINER_CSV_PATH="$EFFECTIVE_CONTAINER_DIR/$(basename "$HOST_CSV_PATH")"

  echo "Container does not have the workspace mounted at $CONTAINER_WORKSPACE."
  echo "Copying the publisher script and CSV into $CONTAINER:$EFFECTIVE_CONTAINER_DIR"

  docker exec -i "$CONTAINER" mkdir -p "$EFFECTIVE_CONTAINER_DIR"
  docker cp "$HOST_WORKSPACE/scripts/vulcanexus/traj_pose_array_pub.py" \
    "$CONTAINER:$EFFECTIVE_CONTAINER_SCRIPT_PATH"
  docker cp "$HOST_CSV_PATH" "$CONTAINER:$EFFECTIVE_CONTAINER_CSV_PATH"
fi

echo "Container: $CONTAINER"
echo "CSV: $EFFECTIVE_CONTAINER_CSV_PATH"
echo "Topic: $TOPIC"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}"
if [ -n "${ROS_DISCOVERY_SERVER:-}" ]; then
  echo "ROS_DISCOVERY_SERVER=$ROS_DISCOVERY_SERVER"
fi

docker exec -i "$CONTAINER" bash -lc "
source /opt/vulcanexus/humble/setup.bash
export ROS_DOMAIN_ID='${ROS_DOMAIN_ID:-42}'
export RMW_IMPLEMENTATION='${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}'
export ROS_LOCALHOST_ONLY='${ROS_LOCALHOST_ONLY:-0}'
if [ -n '${ROS_DISCOVERY_SERVER:-}' ]; then
  export ROS_DISCOVERY_SERVER='${ROS_DISCOVERY_SERVER:-}'
fi
python3 '$EFFECTIVE_CONTAINER_SCRIPT_PATH' \
  --csv-path '$EFFECTIVE_CONTAINER_CSV_PATH' \
  --topic '$TOPIC' \
  --frame-id '$FRAME_ID' \
  --repeat '$REPEAT' \
  --rate-hz '$RATE_HZ' \
  --wait-for-subscriber-sec '$WAIT_FOR_SUBSCRIBER_SEC' \
  --qos-reliability '$QOS_RELIABILITY' \
  --qos-durability '$QOS_DURABILITY'
"
