#!/usr/bin/env bash

set -euo pipefail

CONTAINER="${VULCANEXUS_CONTAINER:-vulcanexus_humble}"
HOST_XML_PATH="${HOST_XML_PATH:-/tmp/fastdds_server_configuration.xml}"
CONTAINER_XML_PATH="${CONTAINER_XML_PATH:-/workspace/vilma-agent/.tmp/fastdds_server_configuration.xml}"
ROS_SETUP_FILE="${ROS_SETUP_FILE:-/opt/vulcanexus/humble/setup.bash}"

if [ ! -f "$HOST_XML_PATH" ]; then
  echo "Discovery Server XML not found on host: $HOST_XML_PATH" >&2
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]; then
  echo "Starting container $CONTAINER"
  docker start "$CONTAINER" >/dev/null
fi

CONTAINER_XML_DIR="$(dirname "$CONTAINER_XML_PATH")"

docker exec -i "$CONTAINER" mkdir -p "$CONTAINER_XML_DIR"
docker cp "$HOST_XML_PATH" "$CONTAINER:$CONTAINER_XML_PATH"

echo "Container: $CONTAINER"
echo "Discovery Server XML: $CONTAINER_XML_PATH"
echo "ROS setup: $ROS_SETUP_FILE"

docker exec -it "$CONTAINER" bash -lc "
source '$ROS_SETUP_FILE'
fastdds discovery -x '$CONTAINER_XML_PATH'
"
