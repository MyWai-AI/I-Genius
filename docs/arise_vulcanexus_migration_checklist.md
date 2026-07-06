# ARISE Vulcanexus Migration Checklist

This file keeps the current work organized across the two local repositories:

- `~/vilma-agent`: experiment and validation workspace
- `~/vilma-agent-clean`: clean upstream base for the final push branch

The rule is simple:

- prove behavior in `vilma-agent`
- port only the proven subset into `vilma-agent-clean`

## Current Proven Runtime Milestones

These items are already validated in `vilma-agent`:

1. VILMA generates a final robot-stage Cartesian trajectory.
2. The VILMA `Push to Robot` action can publish through Vulcanexus.
3. The edge VM can subscribe to `/learned_trajectory`.
4. Manual publish from `data/_runtime/vulcanexus/last_cartesian_push.csv` works.
5. The local Streamlit app is the correct runtime for the new publish path.
6. The reusable split for the Comau lab setup is now defined as:
   - generic edge receiver
   - robot-specific backend

These items are in progress:

1. Edge status feedback on `/trajectory_status`
2. Generic edge receiver deployment for the Ubuntu 20.04 + Noetic + Humble-container case
3. Backend handoff from receiver artifact to robot-specific execution

## Repo Roles

### `~/vilma-agent`

Use this repo for:

- experiments
- runtime debugging
- DDS/Vulcanexus validation
- temporary scripts and notes
- rapid UI changes

This repo is allowed to stay messy while the flow is being proven.

### `~/vilma-agent-clean`

Use this repo for:

- the final feature branch
- clean, reviewable commits
- mentor-facing architecture alignment
- the branch intended to be pushed

Do not try to make `vilma-agent` itself the pushable repo.

## Python Version Alignment

Keep both repos on the same Python version:

- `Python 3.12.9`

Use this interpreter when creating the clean repo venv:

```bash
/home/vvijaykumar/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/bin/python3.12
```

The project install flow is still:

- `uv sync`

Note: `vilma-agent-clean` currently still needs the native Cyclone DDS dependency resolved before `uv sync` fully succeeds.

## Ready-To-Port Files

These are the first files to move into `vilma-agent-clean` once the current milestone is stable:

- `src/streamlit_template/new_ui/services/Common/robot_action_service.py`
- `src/streamlit_template/new_ui/pages/SVO/svo_pipeline_page.py`
- `src/streamlit_template/new_ui/pages/BAG/bag_pipeline_page.py`
- `src/streamlit_template/new_ui/pages/Generic/pipeline_page.py`
- `scripts/vulcanexus/docker_publish_traj.sh`
- `scripts/vulcanexus/docker_run_fastdds_discovery_server.sh`
- `scripts/vulcanexus/docker_subscribe_traj.sh`
- `scripts/vulcanexus/docker_wait_for_status.sh`
- `scripts/vulcanexus/edge_receive_posearray.py`
- `scripts/vulcanexus/run_edge_receiver.sh`
- `scripts/vulcanexus/comau_backend_example.sh`
- `scripts/vulcanexus/run_edge_executor.sh`
- `scripts/vulcanexus/traj_pose_array_pub.py`
- `scripts/vulcanexus/traj_pose_array_sub.py`
- `scripts/vulcanexus/traj_pose_array_executor.py`
- `scripts/vulcanexus/traj_status_sub.py`
- `scripts/vulcanexus/README.md`
- `README_VULCANEXUS_MACHINES.md`
- `docs/vulcanexus_cross_machine_test.md`

## Review Before Porting

These files have local changes or additions in `vilma-agent`, but they should be reviewed deliberately before moving them into the clean repo:

- `README.md`
- `src/streamlit_template/core/Common/robot_playback.py`
- `src/streamlit_template/core/SVO/99_run_full_pipeline.py`
- `src/streamlit_template/new_ui/components/Common/frame_viewer.py`
- `src/streamlit_template/new_ui/services/Generic/pipeline_service.py`
- `src/streamlit_template/core/SVO/14_plot_robot_execution.py`
- `src/streamlit_template/core/SVO/06to merge_compute_object_distance_offsets.py`

These are likely mixed with unrelated experimental work and should not be ported blindly.

## Do Not Port

Do not port these into `vilma-agent-clean` unless there is a very explicit reason:

- notebooks
- local PDFs
- raw experiment data under `data/`
- local uploads/downloads
- temporary runtime exports under `data/_runtime/`
- `bag_alignment_check/`
- one-off debug scripts in the repo root

## Docker Compose Direction

Do not restructure Compose until the runtime loop is complete.

The target service split should be:

1. `vilma-ui`
2. `trajectory-push-adapter`
3. `edge-receiver`
4. robot-specific backend or documented external consumer
4. later: `northbound-api` / FIWARE adapter

The current local success path is still:

- local Streamlit from `vilma-agent`
- Vulcanexus helper scripts on `server1`
- edge ROS 2 / execution side

## Mentor Alignment Target

The final pushed branch should make these boundaries explicit:

1. `VILMA` is the frontend/orchestrator
2. `Vulcanexus` is the southbound transport adapter
3. `Edge executor` is the robot-side adapter
4. `FIWARE` or HTTP context/API belongs on the northbound side
5. the reusable asset is the execution-ready trajectory plus the transport/execution adapters

## Next Steps In Order

1. Finish edge executor dry-run
2. Verify `/trajectory_status` feedback
3. Freeze the current runtime behavior
4. Define the final Compose/service split
5. Port the proven subset into `vilma-agent-clean`
6. Create the final feature branch there
7. Clean docs and architecture explanation
8. Only then prepare mentor-facing diagrams/slides
