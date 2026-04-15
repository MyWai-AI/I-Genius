#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"
shift || true

SERVER_GUID_PREFIX_DEFAULT="44.53.00.5f.45.50.52.4f.53.49.4d.41"

usage() {
  cat <<'EOF' >&2
Usage:
  render_fastdds_wan_tcp_profile.sh server [options]
  render_fastdds_wan_tcp_profile.sh client [options]

Server options:
  --public-host <dns-or-ip>       Public IP/DNS visible to the client
  --listen-port <port>            TCP listening port for the Discovery Server
  --output <path>                 Output XML path
  --profile-name <name>           Optional participant profile name
  --transport-id <id>             Optional TCP transport id
  --server-guid-prefix <prefix>   Optional server GUID prefix

Client options:
  --client-public-host <dns-or-ip>  Public IP/DNS visible to the peer
  --listen-port <port>              TCP listening port for this client
  --server-host <dns-or-ip>         Public IP/DNS of the Discovery Server
  --server-port <port>              TCP port of the Discovery Server
  --output <path>                   Output XML path
  --profile-name <name>             Optional participant profile name
  --transport-id <id>               Optional TCP transport id
  --server-guid-prefix <prefix>     Optional server GUID prefix
EOF
  exit 1
}

resolve_ipv4() {
  local host="$1"
  if printf '%s' "$host" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
    printf '%s\n' "$host"
    return 0
  fi

  local resolved
  resolved="$(getent ahostsv4 "$host" | awk 'NR==1 {print $1}')"
  if [ -z "$resolved" ]; then
    echo "Could not resolve host to IPv4: $host" >&2
    exit 1
  fi
  printf '%s\n' "$resolved"
}

PUBLIC_HOST=""
CLIENT_PUBLIC_HOST=""
SERVER_HOST=""
LISTEN_PORT=""
SERVER_PORT=""
OUTPUT_PATH=""
PROFILE_NAME=""
TRANSPORT_ID=""
SERVER_GUID_PREFIX="$SERVER_GUID_PREFIX_DEFAULT"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --public-host)
      PUBLIC_HOST="${2:-}"
      shift 2
      ;;
    --client-public-host)
      CLIENT_PUBLIC_HOST="${2:-}"
      shift 2
      ;;
    --server-host)
      SERVER_HOST="${2:-}"
      shift 2
      ;;
    --listen-port)
      LISTEN_PORT="${2:-}"
      shift 2
      ;;
    --server-port)
      SERVER_PORT="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:-}"
      shift 2
      ;;
    --profile-name)
      PROFILE_NAME="${2:-}"
      shift 2
      ;;
    --transport-id)
      TRANSPORT_ID="${2:-}"
      shift 2
      ;;
    --server-guid-prefix)
      SERVER_GUID_PREFIX="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$MODE" in
  server)
    [ -n "$PUBLIC_HOST" ] || usage
    [ -n "$LISTEN_PORT" ] || usage
    OUTPUT_PATH="${OUTPUT_PATH:-/tmp/fastdds_server_configuration.xml}"
    PROFILE_NAME="${PROFILE_NAME:-TCP_discovery_server_profile}"
    TRANSPORT_ID="${TRANSPORT_ID:-TCP_ds_transport}"
    TEMPLATE_PATH="$SCRIPT_DIR/fastdds_wan_tcp_server.template.xml"
    WAN_ADDR="$(resolve_ipv4 "$PUBLIC_HOST")"

    sed \
      -e "s/CHANGE_ME_TRANSPORT_ID/$TRANSPORT_ID/g" \
      -e "s/CHANGE_ME_LISTEN_PORT/$LISTEN_PORT/g" \
      -e "s/CHANGE_ME_WAN_ADDR/$WAN_ADDR/g" \
      -e "s/CHANGE_ME_PROFILE_NAME/$PROFILE_NAME/g" \
      -e "s/CHANGE_ME_SERVER_GUID_PREFIX/$SERVER_GUID_PREFIX/g" \
      "$TEMPLATE_PATH" > "$OUTPUT_PATH"

    echo "Wrote server XML profile to $OUTPUT_PATH"
    echo "WAN_ADDR=$WAN_ADDR"
    echo "LISTEN_PORT=$LISTEN_PORT"
    echo "PROFILE_NAME=$PROFILE_NAME"
    ;;

  client)
    [ -n "$CLIENT_PUBLIC_HOST" ] || usage
    [ -n "$LISTEN_PORT" ] || usage
    [ -n "$SERVER_HOST" ] || usage
    [ -n "$SERVER_PORT" ] || usage
    OUTPUT_PATH="${OUTPUT_PATH:-/tmp/fastdds_client_configuration.xml}"
    PROFILE_NAME="${PROFILE_NAME:-TCP_client_profile}"
    TRANSPORT_ID="${TRANSPORT_ID:-TCP_client_transport}"
    TEMPLATE_PATH="$SCRIPT_DIR/fastdds_wan_tcp_client.template.xml"
    CLIENT_WAN_ADDR="$(resolve_ipv4 "$CLIENT_PUBLIC_HOST")"
    SERVER_WAN_ADDR="$(resolve_ipv4 "$SERVER_HOST")"

    sed \
      -e "s/CHANGE_ME_TRANSPORT_ID/$TRANSPORT_ID/g" \
      -e "s/CHANGE_ME_LISTEN_PORT/$LISTEN_PORT/g" \
      -e "s/CHANGE_ME_CLIENT_WAN_ADDR/$CLIENT_WAN_ADDR/g" \
      -e "s/CHANGE_ME_PROFILE_NAME/$PROFILE_NAME/g" \
      -e "s/CHANGE_ME_SERVER_GUID_PREFIX/$SERVER_GUID_PREFIX/g" \
      -e "s/CHANGE_ME_SERVER_WAN_ADDR/$SERVER_WAN_ADDR/g" \
      -e "s/CHANGE_ME_SERVER_PORT/$SERVER_PORT/g" \
      "$TEMPLATE_PATH" > "$OUTPUT_PATH"

    echo "Wrote client XML profile to $OUTPUT_PATH"
    echo "CLIENT_WAN_ADDR=$CLIENT_WAN_ADDR"
    echo "LISTEN_PORT=$LISTEN_PORT"
    echo "SERVER_WAN_ADDR=$SERVER_WAN_ADDR"
    echo "SERVER_PORT=$SERVER_PORT"
    echo "PROFILE_NAME=$PROFILE_NAME"
    ;;

  *)
    usage
    ;;
esac
