#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"

if [ -z "$MODE" ]; then
  echo "Usage: $0 <cloud|edge>" >&2
  exit 1
fi

case "$MODE" in
  cloud)
    TEMPLATE_PATH="/home/vvijaykumar/vilma-agent/scripts/vulcanexus/ddsrouter_cloud.template.yaml"
    OUTPUT_PATH="${OUTPUT_PATH:-/tmp/ddsrouter_cloud.yaml}"
    ;;
  edge)
    TEMPLATE_PATH="/home/vvijaykumar/vilma-agent/scripts/vulcanexus/ddsrouter_edge.template.yaml"
    OUTPUT_PATH="${OUTPUT_PATH:-/tmp/ddsrouter_edge.yaml}"
    ;;
  *)
    echo "Invalid mode: $MODE" >&2
    echo "Usage: $0 <cloud|edge>" >&2
    exit 1
    ;;
esac

CLOUD_PUBLIC_HOST="${CLOUD_PUBLIC_HOST:-}"
USER_CLOUD_LISTEN_IP="${CLOUD_LISTEN_IP:-}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID_VALUE:-42}"
WAN_PORT="${WAN_PORT:-45678}"

if [ -z "$CLOUD_PUBLIC_HOST" ]; then
  echo "Set CLOUD_PUBLIC_HOST to the cloud public DNS name or IP." >&2
  exit 1
fi

if [ ! -f "$TEMPLATE_PATH" ]; then
  echo "Template not found: $TEMPLATE_PATH" >&2
  exit 1
fi

if printf '%s' "$CLOUD_PUBLIC_HOST" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
  RESOLVED_CLOUD_PUBLIC_IP="$CLOUD_PUBLIC_HOST"
else
  RESOLVED_CLOUD_PUBLIC_IP="$(getent ahostsv4 "$CLOUD_PUBLIC_HOST" | awk 'NR==1 {print $1}')"
  if [ -z "$RESOLVED_CLOUD_PUBLIC_IP" ]; then
    echo "Could not resolve CLOUD_PUBLIC_HOST to an IPv4 address: $CLOUD_PUBLIC_HOST" >&2
    exit 1
  fi
fi

if [ -n "$USER_CLOUD_LISTEN_IP" ]; then
  CLOUD_LISTEN_IP="$USER_CLOUD_LISTEN_IP"
else
  if [ "$MODE" = "cloud" ]; then
    # Bind the cloud router on all local interfaces by default. The public
    # address is still reflected via CLOUD_PUBLIC_HOST / external-port.
    CLOUD_LISTEN_IP="0.0.0.0"
  else
    CLOUD_LISTEN_IP="$RESOLVED_CLOUD_PUBLIC_IP"
  fi
fi

sed \
  -e "s/CHANGE_ME_CLOUD_PUBLIC_IP/$RESOLVED_CLOUD_PUBLIC_IP/g" \
  -e "s/CHANGE_ME_CLOUD_LISTEN_IP/$CLOUD_LISTEN_IP/g" \
  -e "s/domain: 42/domain: $ROS_DOMAIN_ID_VALUE/g" \
  -e "s/port: 45678/port: $WAN_PORT/g" \
  -e "s/external-port: 45678/external-port: $WAN_PORT/g" \
  "$TEMPLATE_PATH" > "$OUTPUT_PATH"

echo "Wrote $MODE DDS Router config to $OUTPUT_PATH"
echo "CLOUD_PUBLIC_HOST=$CLOUD_PUBLIC_HOST"
echo "RESOLVED_CLOUD_PUBLIC_IP=$RESOLVED_CLOUD_PUBLIC_IP"
echo "CLOUD_LISTEN_IP=$CLOUD_LISTEN_IP"
echo "ROS_DOMAIN_ID_VALUE=$ROS_DOMAIN_ID_VALUE"
echo "WAN_PORT=$WAN_PORT"
