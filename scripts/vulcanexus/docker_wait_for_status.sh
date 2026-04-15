#!/usr/bin/env bash

set -euo pipefail

CONTAINER="${VULCANEXUS_SUB_CONTAINER:-vulcanexus_humble}"
HOST_WORKSPACE="${HOST_WORKSPACE:-/home/vvijaykumar/vilma-agent}"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspace/vilma-agent}"
TOPIC="${TOPIC:-/trajectory_status}"
SETUP_FILE="${ROS_SETUP_FILE:-/opt/vulcanexus/humble/setup.bash}"
NODE_NAME="${NODE_NAME:-traj_status_sub}"
TIMEOUT_SEC="${TIMEOUT_SEC:-8}"

CONTAINER_SCRIPT_PATH="$CONTAINER_WORKSPACE/scripts/vulcanexus/traj_status_sub.py"
FALLBACK_CONTAINER_DIR="/tmp/vilma-agent-vulcanexus-status"

if [ ! -f "$HOST_WORKSPACE/scripts/vulcanexus/traj_status_sub.py" ]; then
  echo "Host status subscriber script not found: $HOST_WORKSPACE/scripts/vulcanexus/traj_status_sub.py" >&2
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
  echo "Starting container $CONTAINER"
  docker start "$CONTAINER" >/dev/null
fi

if docker exec -i "$CONTAINER" test -f "$CONTAINER_SCRIPT_PATH"; then
  EFFECTIVE_CONTAINER_SCRIPT_PATH="$CONTAINER_SCRIPT_PATH"
else
  EFFECTIVE_CONTAINER_DIR="$FALLBACK_CONTAINER_DIR"
  EFFECTIVE_CONTAINER_SCRIPT_PATH="$EFFECTIVE_CONTAINER_DIR/traj_status_sub.py"

  echo "Container does not have the workspace mounted at $CONTAINER_WORKSPACE."
  echo "Copying the status subscriber script into $CONTAINER:$EFFECTIVE_CONTAINER_DIR"

  docker exec -i "$CONTAINER" mkdir -p "$EFFECTIVE_CONTAINER_DIR"
  docker cp "$HOST_WORKSPACE/scripts/vulcanexus/traj_status_sub.py" \
    "$CONTAINER:$EFFECTIVE_CONTAINER_SCRIPT_PATH"
fi

docker exec -i "$CONTAINER" bash -lc "
source '$SETUP_FILE'
export ROS_DOMAIN_ID='${ROS_DOMAIN_ID:-42}'
export RMW_IMPLEMENTATION='${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}'
export ROS_LOCALHOST_ONLY='${ROS_LOCALHOST_ONLY:-0}'
if [ -n '${ROS_DISCOVERY_SERVER:-}' ]; then
  export ROS_DISCOVERY_SERVER='${ROS_DISCOVERY_SERVER:-}'
fi
python3 '$EFFECTIVE_CONTAINER_SCRIPT_PATH' \
  --topic '$TOPIC' \
  --node-name '$NODE_NAME' \
  --timeout-sec '$TIMEOUT_SEC'
"
