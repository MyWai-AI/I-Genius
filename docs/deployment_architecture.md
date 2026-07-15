# Deployment Architecture

The reusable module architecture is ROS 2 / Vulcanexus / Fast DDS at the robot communication boundary, with deployment-specific DDS discovery or routing.

```text
I-Genius / VILMA
-> Cartesian trajectory artifact
-> Vulcanexus ROS 2 publisher
-> /learned_trajectory
-> geometry_msgs/msg/PoseArray
-> Fast DDS through rmw_fastrtps_cpp
-> DDS network deployment
-> Edge ROS 2 receiver
-> Robot-specific backend
```

The reusable module exposes standard ROS 2 / DDS interfaces. DDS discovery and network routing are deployment-specific and are configured according to the target network topology.

## Application Boundary

The application layer is:

- `Dockerfile`
- `docker-compose.yml`
- Compose service `vilma-agent`

The base Compose file exposes only:

```text
vilma-agent
```

It does not start Vulcanexus, an edge receiver, a Fast DDS Discovery Server, or DDS Router.

## ROS 2 / DDS Contract

| Setting | Value |
|---|---|
| `ROS_DOMAIN_ID` | `42` |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` |
| `ROS_LOCALHOST_ONLY` | `0` |
| Topic | `/learned_trajectory` |
| Message | `geometry_msgs/msg/PoseArray` |

Publisher helpers:

- `scripts/vulcanexus/docker_publish_traj.sh`
- `scripts/vulcanexus/traj_pose_array_pub.py`

Edge receiver helpers:

- `scripts/vulcanexus/run_edge_receiver.sh`
- `scripts/vulcanexus/edge_receive_posearray.py`

Edge executor helpers:

- `scripts/vulcanexus/run_edge_executor.sh`
- `scripts/vulcanexus/traj_pose_array_executor.py`

## DDS Network Deployment Options

Valid DDS-native deployment mechanisms include:

- native DDS discovery
- Fast DDS Discovery Server
- DDS Router
- Fast DDS WAN TCP configuration where supported by repository helper scripts

Fast DDS Discovery Server helper:

- `scripts/vulcanexus/docker_run_fastdds_discovery_server.sh`
- default UDP port `14520`

DDS Router helper set:

- `scripts/vulcanexus/ddsrouter_cloud.template.yaml`
- `scripts/vulcanexus/ddsrouter_edge.template.yaml`
- `scripts/vulcanexus/render_ddsrouter_wan_config.sh`
- `scripts/vulcanexus/docker_run_ddsrouter.sh`
- default WAN TCP port `45678`

DDS Router is a DDS-native option for routed server-to-edge deployments. The repository contains templates and helper scripts, but this document does not claim a completed cross-machine runtime validation.

Fast DDS WAN TCP helper set:

- `scripts/vulcanexus/fastdds_wan_tcp_server.template.xml`
- `scripts/vulcanexus/fastdds_wan_tcp_client.template.xml`
- `scripts/vulcanexus/render_fastdds_wan_tcp_profile.sh`
- `scripts/vulcanexus/docker_run_fastdds_discovery_server_xml.sh`

## FIWARE Boundary

FIWARE is a separate northbound execution metadata and status layer:

```text
Execution metadata / status
-> NGSI-LD REST
-> Orion-LD / FIWARE
```

Vulcanexus / ROS 2 / Fast DDS carries robot trajectory communication. FIWARE / Orion-LD stores execution metadata and lifecycle/status information. FIWARE does not carry raw trajectory delivery, and this repository does not implement the DDS-NGSI-LD Enabler.

## Review Rule

Do not add middleware services to `docker-compose.yml` merely to match older documentation. Middleware lifecycle is deployment-specific and remains handled through the helper scripts unless a tested service split is deliberately introduced later.
