# Installation

Use this on a fresh Ubuntu 22.04 developer/server machine:

```bash
git clone <repo>
cd vilma-agent
cp .env.example .env
# edit .env and set MYWAI_ARTIFACTS_MAIL / MYWAI_ARTIFACTS_TOKEN
./install_dependencies.sh --skip-uv-sync
docker compose --env-file .env up -d --build
```

This prepares host dependencies and starts:

- `vilma-agent`
- `vulcanexus_humble`
- `zenoh_cloud`

The app is available on `http://localhost:9002`.

For the repo-managed one-command path, use:

```bash
./setup.sh --start-app
```

That preserves the manual-container-compatible bootstrap flow and starts the
existing app Compose service.

## Private Package Credentials

Python dependency installation requires the private MyWai package feed. Add
these values to `.env` or pass them in the environment before running setup:

```bash
MYWAI_ARTIFACTS_MAIL=you@example.com \
MYWAI_ARTIFACTS_TOKEN=<token> \
./setup.sh
```

If setup already ran without credentials, add them to `.env` and run:

```bash
./install_dependencies.sh --skip-system
```

## Start the App Locally

For a local non-container app run after `uv sync` succeeds:

```bash
source .venv/bin/activate
streamlit run src/streamlit_template/new_ui/pages/Common/landing_page.py \
  --server.port 8504 \
  --server.address 0.0.0.0
```

Open `http://localhost:8504`.

## Start the Existing Docker App

The deployment Compose workflow starts the server stack:

```bash
docker compose --env-file .env up -d --build
```

To restart only the app service:

```bash
docker compose --env-file .env up -d vilma-agent
```

The container maps host port `9002` to Streamlit port `8504`.

## Edge Machine Setup

On an Edge/robot-side Ubuntu 22.04 machine:

```bash
git clone <repo>
cd vilma-agent
./install_dependencies.sh --skip-uv-sync
docker compose --env-file .env --profile edge up -d zenoh_edge edge_receiver
```

This starts `zenoh_edge` and the generic `edge_receiver` container. To run the
receiver directly on a ROS 2 host instead:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
bash ./scripts/vulcanexus/run_edge_receiver.sh
```

## Setup Script Roles

`setup.sh` remains available for compatibility with hosts that already have
manually-created containers.

| Role | Starts |
|---|---|
| `server` | `vulcanexus_humble`, `zenoh_cloud` |
| `edge` | `zenoh_edge` |
| `all` | all three infrastructure containers |
| `none` | no infrastructure containers |

Examples:

```bash
./setup.sh --role server
./setup.sh --role edge --install-ros2
./setup.sh --role all
./setup.sh --role none --skip-infra
```

## Verify

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

Expected server containers:

```text
vilma-agent
vulcanexus_humble
zenoh_cloud
```

Expected Edge container:

```text
zenoh_edge
vilma_edge_receiver
```
