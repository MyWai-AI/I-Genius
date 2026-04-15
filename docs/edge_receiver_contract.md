# Edge Receiver Contract

This document defines the reusable boundary between:

- `vilma-agent` as the trajectory producer
- Vulcanexus / ROS 2 as the transport layer
- robot-side execution backends such as the Ubuntu 20.04 + ROS1 Noetic Comau stack

## Purpose

The goal is to make the edge side reusable across multiple deployments.

`vilma-agent` should not know how to execute on a specific robot. It should only:

1. produce an execution-ready Cartesian trajectory
2. publish it through the transport layer

The edge side is then responsible for:

1. receiving the trajectory
2. persisting a normalized artifact
3. optionally invoking a robot-specific backend

## Transport Contract

Input topic:

- `/learned_trajectory`
- `geometry_msgs/msg/PoseArray`

Status topic:

- `/trajectory_status`
- `std_msgs/msg/String`
- JSON payload describing states such as `received`, `persisted`, `completed`, `backend_completed`, `failed`

## Artifact Contract

The generic edge receiver writes two files:

- `<prefix>_trajectory.csv`
- `<prefix>_metadata.json`

### CSV format

Columns:

- `x`
- `y`
- `z`

The receiver preserves the incoming Cartesian points as-is.

### Metadata format

Fields:

- `created_at`
- `frame_id`
- `units`
- `point_count`
- `source_topic`
- `status_topic`
- `artifact_prefix`
- `first_point`
- `last_point`

## Deployment Pattern For Ubuntu 20.04 + ROS1 Noetic + ROS2 Humble Container

Recommended split:

1. `vilma-agent` publishes the final trajectory from the cloud/server side
2. a ROS 2 Humble container on the robot machine runs the generic edge receiver
3. the receiver writes the normalized artifact into a shared host folder
4. a host-side Comau backend consumes that artifact
5. the existing Noetic Comau tools validate/remap/execute it

This keeps:

- ROS 2 communication in the Humble container
- ROS 1 Comau execution on the Ubuntu 20.04 host
- robot-specific logic outside `vilma-agent`

## Backend Hook Contract

The generic edge receiver can optionally invoke a backend executable after persisting the artifact.

If configured, the backend receives these environment variables:

- `TRAJECTORY_CSV`
- `TRAJECTORY_METADATA_JSON`
- `TRAJECTORY_FRAME_ID`
- `TRAJECTORY_UNITS`
- `TRAJECTORY_POINT_COUNT`
- `TRAJECTORY_ARTIFACT_PREFIX`

This allows different robot backends to be swapped in without changing the producer side.

## Specific Comau Recommendation

For the current Comau setup:

- keep validation/remap and execution in the Vivek-side / Comau-side tools
- do not move Comau execution logic into `vilma-agent`
- treat the Comau adapter as one robot-specific backend behind the generic edge receiver

## Scripts Added In This Repo

- `scripts/vulcanexus/edge_receive_posearray.py`
- `scripts/vulcanexus/run_edge_receiver.sh`
- `scripts/vulcanexus/comau_backend_example.sh`

These are additive and do not replace the existing dry-run executor flow.
