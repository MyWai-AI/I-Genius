# Vulcanexus Cross-Machine README

This guide shows how to publish a trajectory from one machine and subscribe to it on another machine using ROS 2 + Vulcanexus + Fast DDS.

It covers:

- same network, direct ROS 2 discovery
- same network, but more reliable discovery using Fast DDS Discovery Server
- different servers or different networks using DDS Router

The reusable scripts are in this folder:

- `traj_pose_array_pub.py`
- `docker_publish_traj.sh`
- `ros2_sub_echo.sh`
- `docker_run_fastdds_discovery_server.sh`
- `show_network_info.sh`
- `ddsrouter_cloud.template.yaml`
- `ddsrouter_edge.template.yaml`
- `edge_receive_posearray.py`
- `run_edge_receiver.sh`
- `comau_backend_example.sh`

## Reusable edge pattern

For deployments where the robot machine is different from the VILMA server, the
recommended pattern is:

```text
vilma-agent
-> Vulcanexus / ROS 2 transport
-> generic edge receiver
-> normalized trajectory artifact
-> robot-specific backend
```

This matters for mixed systems such as:

- Ubuntu 20.04 host with ROS 1 Noetic robot drivers
- ROS 2 Humble Docker container used only for communication

In that setup:

- `edge_receive_posearray.py` is the reusable receiver
- the backend remains robot-specific
- `vilma-agent` stays robot-agnostic

## 0. Can this server talk to any machine in the world?

Short answer: no, not as it is currently configured.

Your current Discovery Server is listening on:

```text
192.168.0.10:11811
```

That `192.168.0.10` address is a private LAN address. It is normally reachable only by:

- machines on the same local network
- machines connected through a VPN into that network
- machines that can reach it through explicit router or firewall rules

It is not a public Internet address.

So:

- another machine on the same LAN or routed private network can use it
- a random machine on the public Internet cannot directly use `192.168.0.10`

Also, your Fast DDS output showed:

```text
Security: NO
```

So even if you make it reachable from outside, it is not a good idea to expose it openly to the Internet without additional network protection.

For machines in different sites or over the Internet, the practical options are:

- use a VPN between the sites, then use Discovery Server or DDS Router over the VPN IPs
- use DDS Router with a reachable public IP and the required firewall or port-forwarding rules
- use the official TCP-over-WAN Discovery Server approach with explicit transport configuration

## 0.1 What the other machine must have installed

The answer depends on the role of the other machine.

### Case A. The other machine is only a subscriber or publisher

It does not need to run Discovery Server.

It needs:

- ROS 2 Humble or Vulcanexus Humble
- Fast DDS RMW support
- the message definitions for the topic type
- matching environment settings

For your current topic, the message type is:

```text
geometry_msgs/msg/PoseArray
```

So the other machine must be able to use `geometry_msgs`.

### Case B. The other machine will host Discovery Server

It needs:

- Vulcanexus, or
- Fast DDS CLI tools with `fastdds discovery`

### Case C. The other machine will be the WAN relay side

It needs:

- Vulcanexus or DDS Router installed
- the DDS Router YAML configuration

## 0.2 Recommended minimum install on another Ubuntu 22.04 machine

If the other machine only needs to subscribe to `/learned_trajectory`, the simplest supported setup is standard ROS 2 Humble with Fast DDS.

Install:

```bash
sudo apt update
sudo apt install -y ros-humble-ros-base ros-humble-rmw-fastrtps-cpp
```

If `geometry_msgs` is not already available, also install:

```bash
sudo apt install -y ros-humble-geometry-msgs
```

Then in each new terminal:

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

If you want the other machine to use Vulcanexus instead of standard ROS 2, install Vulcanexus and source:

```bash
source /opt/vulcanexus/humble/setup.bash
```

## 0.3 Minimum compatibility rules

For this communication to work, the two machines should match on these items:

- same `ROS_DOMAIN_ID`
- same RMW implementation, here `rmw_fastrtps_cpp`
- `ROS_LOCALHOST_ONLY=0`
- same topic name, here `/learned_trajectory`
- same topic type, here `geometry_msgs/msg/PoseArray`

Recommended:

- use the same ROS 2 distribution on both sides
- for your case, use Humble on both sides

This is the simplest and least error-prone setup.

## Important shell note

This command is wrong:

```bash
FASTDDS_UDP_ADDRESS=<DISCOVERY_SERVER_IP> FASTDDS_UDP_PORT=11811 ./scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

The angle brackets are not literal text in bash. Bash treats `<DISCOVERY_SERVER_IP>` like input redirection, so it looks for a file named `DISCOVERY_SERVER_IP`.

Use a real IP address instead, for example:

```bash
FASTDDS_UDP_ADDRESS=192.168.10.25 FASTDDS_UDP_PORT=11811 ./scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

Or first put the IP in a variable:

```bash
DISCOVERY_SERVER_IP=192.168.10.25
FASTDDS_UDP_ADDRESS="$DISCOVERY_SERVER_IP" FASTDDS_UDP_PORT=11811 ./scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

## 1. Decide which machine is which

You need two machines:

- publisher machine
  - this is where `vulcanexus_humble` runs and where the trajectory CSV already exists
- subscriber machine
  - this is where you will run `ros2 topic echo`

If you want to use Discovery Server, choose one machine to host it:

- discovery server host
  - usually the machine that both other machines can reach by IP

In your case, a likely first setup is:

- `server1.localdomain`
  - publisher
  - discovery server host
- `vivek.vm.server1.localdomain`
  - subscriber

That is only a suggestion. Either machine can be the subscriber or discovery server host as long as the other machine can reach it.

## 2. Find the correct IP address on each machine

Run this on each machine:

```bash
cd /home/vvijaykumar/vilma-agent
./scripts/vulcanexus/show_network_info.sh
```

This prints:

- hostname
- primary IPv4 address
- all IPv4 addresses
- default route

If you want the shortest single command, use:

```bash
ip route get 1.1.1.1 | awk '/src/ {for (i = 1; i <= NF; ++i) if ($i == "src") {print $(i+1); exit}}'
```

Useful extra checks:

```bash
hostname -I
ip -brief -4 addr show up
```

Important:

- use the IP from the actual machine that will run ROS 2 or Discovery Server
- do not assume the VNC URL IP is the same as the VM's ROS IP
- `http://192.168.10.250:6081` may only be the VNC gateway or web console address

## 3. Test basic reachability between machines

From machine A, test machine B:

```bash
ping -c 3 <OTHER_MACHINE_IP>
ssh vvijaykumar@<OTHER_MACHINE_IP>
```

If DNS names work, these are also fine:

```bash
ssh vvijaykumar@server1.localdomain
ssh vvijaykumar@vivek.vm.server1.localdomain
```

If SSH by hostname works but IP does not, or the other way around, note that for ROS 2 the IP path matters more than the hostname.

## 4. Shared ROS 2 settings

Use the same ROS 2 settings on both machines:

```bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
```

These must match on both sides.

If the subscriber machine uses Vulcanexus instead of standard ROS 2, set:

```bash
export ROS_SETUP_FILE=/opt/vulcanexus/humble/setup.bash
```

The subscriber helper script will then source Vulcanexus instead of `/opt/ros/humble/setup.bash`.

## 5. Option A: same network, direct ROS 2 discovery

Try this first if both machines are on the same LAN and multicast is allowed.

### Subscriber machine

```bash
cd /home/vvijaykumar/vilma-agent
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_SETUP_FILE=/opt/ros/humble/setup.bash
./scripts/vulcanexus/ros2_sub_echo.sh /learned_trajectory geometry_msgs/msg/PoseArray
```

Keep that terminal open.

### Publisher machine

```bash
cd /home/vvijaykumar/vilma-agent
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
REPEAT=40 RATE_HZ=2 WAIT_FOR_SUBSCRIBER_SEC=15 ./scripts/vulcanexus/docker_publish_traj.sh
```

Why loop publish instead of one-shot?

- a one-shot publish can be missed if discovery is not finished yet
- looping for 20 seconds gives the subscriber time to appear

Expected result:

- publisher terminal shows `Published ... PoseArray ...`
- subscriber terminal prints a `geometry_msgs/msg/PoseArray`

If this works, you are done.

If this does not work, move to Option B.

## 6. Option B: same network, recommended for VMs, use Fast DDS Discovery Server

This is usually the best choice for:

- VMs
- Wi-Fi
- bridged or NAT virtual networks
- cloud overlays
- cases where direct discovery does not find peers reliably

### Step B1. Choose the Discovery Server host

Assume `server1.localdomain` will host it.

Find its IP on `server1.localdomain`:

```bash
cd /home/vvijaykumar/vilma-agent
./scripts/vulcanexus/show_network_info.sh
```

Assume the script prints:

```text
Primary IPv4 used for outbound traffic:
192.168.10.25
```

Then use:

```bash
DISCOVERY_SERVER_IP=192.168.10.25
```

### Step B2. Start Discovery Server on the chosen host

On `server1.localdomain`:

```bash
cd /home/vvijaykumar/vilma-agent
DISCOVERY_SERVER_IP="$(ip route get 1.1.1.1 | awk '/src/ {for (i = 1; i <= NF; ++i) if ($i == "src") {print $(i+1); exit}}')"
echo "$DISCOVERY_SERVER_IP"
FASTDDS_UDP_ADDRESS="$DISCOVERY_SERVER_IP" FASTDDS_UDP_PORT=11811 ./scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

Leave that terminal running.

If you see an error about the listening port already being allocated, it usually means Discovery Server is already running. Reuse it, or restart it with:

```bash
FASTDDS_FORCE_RESTART=1 FASTDDS_UDP_ADDRESS="$DISCOVERY_SERVER_IP" FASTDDS_UDP_PORT=11811 ./scripts/vulcanexus/docker_run_fastdds_discovery_server.sh
```

### Step B3. Start the subscriber on the other machine

On `vivek.vm.server1.localdomain`:

```bash
cd /home/vvijaykumar/vilma-agent
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_SETUP_FILE=/opt/ros/humble/setup.bash
export ROS_DISCOVERY_SERVER=192.168.10.25:11811
./scripts/vulcanexus/ros2_sub_echo.sh /learned_trajectory geometry_msgs/msg/PoseArray
```

Replace `192.168.10.25` with the actual Discovery Server IP.

### Step B4. Publish from the machine with the CSV and Vulcanexus container

On the publisher machine:

```bash
cd /home/vvijaykumar/vilma-agent
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
export ROS_DISCOVERY_SERVER=192.168.10.25:11811
REPEAT=40 RATE_HZ=2 WAIT_FOR_SUBSCRIBER_SEC=15 ./scripts/vulcanexus/docker_publish_traj.sh
```

Replace `192.168.10.25` with the actual Discovery Server IP.

Expected result:

- Discovery Server terminal stays running and prints no fatal errors
- publisher prints publish logs
- subscriber prints the PoseArray

## 7. Option C: different servers or different networks, use DDS Router

Use this when:

- the machines are on different networks
- they are separated by NAT
- you cannot rely on multicast or shared LAN discovery

### Important WAN note

For machines on different Internet networks, do not treat a plain UDP Discovery Server test like:

```bash
export ROS_DISCOVERY_SERVER=<public-dns-or-ip>:11811
```

as the final solution by itself.

Two common issues appear in that setup:

- `nc -zvu <host> 11811` only proves you can send a UDP packet to that port; it does not prove DDS discovery and data exchange will complete end to end
- `ros2 topic echo` and other ROS 2 CLI tools can fail to see topics under Discovery Server v2 unless the daemon or CLI is configured as a Super Client

For real WAN deployments, prefer one of these:

- DDS Router between the two networks
- TCP over WAN with Discovery Server using XML transport configuration
- a VPN between the two sites, then reuse the same-network Discovery Server setup

### Step C1. Open or forward the cloud TCP port

Open TCP port `45678` on the cloud side, or port-forward it from the public router to the cloud host running DDS Router.

If you already have a public DNS name such as `cloud.example.com`, that name should resolve to the public address that forwards to the cloud host.

Important:

- in the DDS Router v4 YAML used by `eprosima/vulcanexus:humble-desktop`, the WAN `ip:` field must be a literal IPv4 address
- the helper script `render_ddsrouter_wan_config.sh` resolves a hostname such as `cloud.example.com` to the current IPv4 automatically before writing the YAML

### Step C2. Start the cloud Vulcanexus container

On the cloud host:

```bash
cd /home/vvijaykumar/vilma-agent
docker rm -f vulcanexus_humble || true
docker run -d --name vulcanexus_humble --network host -v /home/vvijaykumar/vilma-agent:/workspace/vilma-agent eprosima/vulcanexus:humble-desktop sleep infinity
```

### Step C3. Generate the cloud DDS Router config

On the cloud host:

```bash
cd /home/vvijaykumar/vilma-agent
CLOUD_PUBLIC_HOST=cloud.example.com WAN_PORT=45678 ROS_DOMAIN_ID_VALUE=42 ./scripts/vulcanexus/render_ddsrouter_wan_config.sh cloud
cat /tmp/ddsrouter_cloud.yaml
```

Replace `cloud.example.com` with your real public DNS name or public IP if different.

### Step C4. Start DDS Router on the cloud side

On the cloud host:

```bash
cd /home/vvijaykumar/vilma-agent
HOST_CONFIG_PATH=/tmp/ddsrouter_cloud.yaml DDSROUTER_CONTAINER=vulcanexus_humble ./scripts/vulcanexus/docker_run_ddsrouter.sh
```

Leave this terminal open.

### Step C5. Start a DDS Router container on the edge VM

On the edge VM:

```bash
cd "$HOME/vilma-agent"
docker rm -f vulcanexus_router_edge || true
docker run -d --name vulcanexus_router_edge --network host -v "$HOME/vilma-agent:/workspace/vilma-agent" eprosima/vulcanexus:humble-desktop sleep infinity
```

### Step C6. Generate the edge DDS Router config

On the edge VM:

```bash
cd "$HOME/vilma-agent"
CLOUD_PUBLIC_HOST=cloud.example.com WAN_PORT=45678 ROS_DOMAIN_ID_VALUE=42 ./scripts/vulcanexus/render_ddsrouter_wan_config.sh edge
cat /tmp/ddsrouter_edge.yaml
```

### Step C7. Start DDS Router on the edge VM

On the edge VM:

```bash
cd "$HOME/vilma-agent"
HOST_CONFIG_PATH=/tmp/ddsrouter_edge.yaml DDSROUTER_CONTAINER=vulcanexus_router_edge ./scripts/vulcanexus/docker_run_ddsrouter.sh
```

Leave this terminal open.

### Step C8. Run a local subscriber on the edge VM

On the edge VM, in a new terminal:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
unset ROS_DISCOVERY_SERVER
ros2 topic echo /test_topic std_msgs/msg/String
```

For the trajectory topic instead:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
unset ROS_DISCOVERY_SERVER
python3 "$HOME/vilma-agent/scripts/vulcanexus/traj_pose_array_sub.py"
```

### Step C9. Publish on the cloud side

For a small smoke test:

```bash
docker exec -it vulcanexus_humble bash -lc "
source /opt/vulcanexus/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
unset ROS_DISCOVERY_SERVER
ros2 topic pub /test_topic std_msgs/msg/String '{data: fastdds_router}' -r 1
"
```

For the real trajectory:

```bash
cd /home/vvijaykumar/vilma-agent
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
unset ROS_DISCOVERY_SERVER
REPEAT=40 RATE_HZ=2 WAIT_FOR_SUBSCRIBER_SEC=15 ./scripts/vulcanexus/docker_publish_traj.sh
```

### Step C10. What success looks like

- the cloud DDS Router stays running without fatal errors
- the edge DDS Router stays running without fatal errors
- the edge subscriber receives `/test_topic` or `/learned_trajectory`
- neither the publisher nor the subscriber uses `ROS_DISCOVERY_SERVER` directly in this DDS Router mode

## 8. Quick troubleshooting checklist

Run these on both sides:

```bash
echo "$ROS_DOMAIN_ID"
echo "$RMW_IMPLEMENTATION"
echo "$ROS_LOCALHOST_ONLY"
echo "${ROS_DISCOVERY_SERVER:-<unset>}"
```

Check the topic locally:

```bash
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic info /learned_trajectory
```

If Discovery Server is used, confirm both sides point to the same value:

```bash
echo "$ROS_DISCOVERY_SERVER"
```

If direct discovery fails but Discovery Server works, the issue is probably multicast visibility on the VM network.

If DDS Router is used, confirm:

- the cloud IP in both YAML files is correct
- TCP port `45678` is open
- both local participants use `domain: 42`

## 9. Most practical recommendation

For your case, I recommend this order:

1. Try direct same-network mode once.
2. If no data appears, switch to Discovery Server.
3. If the machines are on different routed networks, use DDS Router.

For two VMs on a server, Discovery Server is usually the sweet spot.
