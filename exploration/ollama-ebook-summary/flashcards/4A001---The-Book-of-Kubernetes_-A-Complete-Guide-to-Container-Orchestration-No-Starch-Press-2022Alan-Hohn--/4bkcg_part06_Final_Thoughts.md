# Flashcards: 4A001---The-Book-of-Kubernetes_-A-Complete-Guide-to-Container-Orchestration-No-Starch-Press-2022Alan-Hohn--_processed (Part 6)

**Starting Chapter:** Final Thoughts

---

---
#### Identifying Network Interfaces
To control network interfaces, you first need to identify them using `ip addr`. This command lists all available network interfaces on your system.

:p How do we identify which network interface to control?
??x
We use the `ip addr` command to list all network interfaces and their details. This helps in identifying the specific interface that needs bandwidth limiting or other network controls.
```bash
root@host01:~# ip addr
```
x??

---
#### Applying Bandwidth Limits with Token Bucket Filter (TBF)
Token Bucket Filter (TBF) is used to limit bandwidth by controlling the token bucket refill rate and size. When a packet arrives, it consumes a token. If there are no tokens available, the packet is queued until tokens become available.

:p How do we apply bandwidth limits using TBF?
??x
To apply bandwidth limits using TBF, you would use the `tc` command with specific parameters to set up the token bucket filter. For example:
```bash
root@host01:~# tc qdisc add dev eth0 root tbf rate 100mbit burst 1540 latency 5ms
```
This command sets a maximum bandwidth limit of 100 Mbps for the `eth0` interface. The `burst` and `latency` parameters control how much data can be buffered before packets start being dropped or queued.

x??

---
#### Observing Bandwidth Limit in Action
After applying the TBF, you can test if the bandwidth is indeed limited by running a network performance test like `iperf3`.

:p What command did we run to verify that the bandwidth limit was applied?
??x
We ran the `iperf3` command to measure the actual network throughput after setting the TBF. The output showed that the transfer rate was significantly lower than expected, confirming the bandwidth limit.
```bash
root@host01:~# iperf3 -c 192.168.61.12
```
x??

---
#### Understanding the Iperf3 Output
The `iperf3` output provides insights into how well the TBF settings are working, including transfer rates and packet retransmissions.

:p What did we observe in the iperf3 output?
??x
In the `iperf3` output, we observed that the transfer rate was limited to 95.7 Mbps (100 Mbps is our limit), confirming that the TBF settings are effective.
```text
[  5] local 192.168.61.11 port 49048 connected to 192.168.61.12 port 5201
...
[ ID] Interval           Transfer     Bitrate         Retr
[  5]   0.00-10.00  sec   114 MBytes  95.7 Mbits/sec    0
```
x??

---
#### Isolating Processes for Precise Control
While the TBF limits bandwidth for all processes on a system, to control bandwidth usage more precisely, we need to isolate specific processes from other processes.

:p Why is it important to isolate processes?
??x
Isolating processes helps in managing and limiting resources such as CPU, memory, and network bandwidth more effectively. This ensures that no single process impacts the performance of others on the system.
x??

---
#### Introducing Control Groups (cgroups)
Control Groups (cgroups) allow us to manage resource limits for groups of processes.

:p What are control groups used for?
??x
Control groups help in managing resources like CPU, memory, and network bandwidth. They ensure that processes share these resources fairly and do not cause performance issues for other processes.
```bash
root@host01:~# cgroup -l
```
x??

---
#### Network Namespaces for Isolation
Network namespaces allow each container to have its own set of network interfaces, IP addresses, and ports.

:p What is the purpose of Linux network namespaces?
??x
The purpose of Linux network namespaces is to isolate processes so that they can operate as if they were on a separate network. This means each container has its own isolated view of the network stack, including separate IP addresses and ports.
```bash
root@host01:~# ip netns
```
x??

---
#### Traffic Flow Through Network Namespaces
Traffic from different network namespaces flows through the system to allow communication between containers and other parts of the network.

:p How does traffic flow between containers using network namespaces?
??x
Traffic from one container (network namespace) can reach another container or the external network by routing through the host machine. The `ip` command with appropriate parameters can be used to set up such routing.
```bash
root@host01:~# ip route add 192.168.61.0/24 dev eth0 table cont1
root@host01:~# ip rule add from 10.0.0.1 table cont1
```
x??

---

#### Network Namespaces and Container Networking Overview
Network namespaces are a critical concept for understanding container networking. They provide each container with its own network stack, including IP addresses, routing tables, and network devices. Containers within the same host share the physical network but communicate via virtual interfaces managed by the operating system.
Background context: Traditional networking involves multiple levels of abstraction (e.g., IP, MAC, etc.). With containers, we introduce an additional layer of complexity to maintain isolation and facilitate communication across different hosts.

:p What is a key concept for understanding container networking?
??x
Network namespaces enable each container to have its own independent network stack. This allows processes in different containers to appear as if they are on separate machines and ensures isolation.
x??

---

#### Network Devices per Container
Each container gets its own virtual network devices, including an IP address that is isolated from the host’s network. These devices allow for proper routing of traffic within the container environment.

:p How does each container get a separate network device?
??x
Each container has its own set of virtual network devices (e.g., eth0) with unique IP addresses assigned by the container runtime. This ensures that processes inside different containers can communicate independently, and also allows them to use any desired port without conflicts.
```sh
# Example command to create a container with a separate network device
crictl runp <container_image> --network-hostname=nginx1
```
x??

---

#### Container Networking on the Same Host
Containers running on the same host can communicate through a bridge device, which acts as a virtual switch. This allows containers to connect and route traffic between them.

:p How do containers on the same host communicate?
??x
Containers running on the same host use a bridge device (e.g., veth pair) for communication. The bridge device connects each container's virtual network interface to a common physical network, enabling them to communicate as if they were separate machines.
```sh
# Example of creating a bridge and connecting containers
crictl runp <container_image> --network-hostname=nginx1
crictl runp <container_image> --network-hostname=nginx2
```
x??

---

#### Network Isolation in Containers
Network isolation is crucial for maintaining reliability. Processes within the same container cannot affect each other due to their isolated network stacks, and this helps prevent conflicts over port usage.

:p Why is network isolation important in containers?
??x
Network isolation ensures that processes running in different containers do not interfere with each other’s network operations. This prevents issues like two processes trying to bind to the same port on the host interface.
```sh
# Example of setting up NGINX servers with isolated network interfaces
source /opt/nginx.sh
```
x??

---

#### Address Translation for Container Networking
Address translation (NAT) is used in container networking to enable containers to connect to external networks while keeping their internal addresses private. This mechanism hides the complexity and details of internal networking.

:p What role does address translation play in container networking?
??x
Address translation, or NAT, allows containers to communicate with external systems using the host’s network interface while keeping their own IP addresses private. The container runtime manages this mapping automatically.
```sh
# Example command for setting up NAT
iptables -t nat -A POSTROUTING -s 10.85.0.0/16 ! -o docker0 -j MASQUERADE
```
x??

---

#### Running Examples of Containers

:p How do you run multiple NGINX servers in containers?
??x
To run multiple NGINX servers, you can use a script like `nginx.sh` to launch the containers with distinct network interfaces and IP addresses.
```sh
source /opt/nginx.sh
crictl ps  # Verify that both NGINX instances are running
```
x??

---

#### Exploring Network Devices in Containers

:p How do you verify that each container has a separate network device?
??x
You can use the `ip addr` command inside a container to check its assigned IP address and network interfaces.
```sh
# Example commands to view network devices inside a container
crictl exec $N1C_ID ip addr
```
x??

---

#### Understanding Network Isolation in Kubernetes

:p How does network isolation work with containers running on different hosts?
??x
Kubernetes uses overlay networks (like CNI plugins) to manage communication between containers across different nodes. Each node has its own set of virtual network devices, ensuring that containers can communicate without exposing their internal networking details.
```sh
# Example command for deploying a pod with Kubernetes
kubectl run nginx --image=nginx:latest
```
x??

---

#### Addressing Multiple Topics

:p How do multiple network concepts differ in this context?
??x
The key differences include:
- **Network Namespaces**: Ensure isolation and separate network stacks.
- **Bridging**: Allows communication between containers on the same host via virtual switches.
- **Address Translation (NAT)**: Enables secure external access for internal networks.
Each concept is crucial for building reliable, scalable, and isolated containerized environments.
x??

---

#### Network Namespaces Overview
Network namespaces are a feature of Linux that allow for the isolation and management of network interfaces, routing tables, IP addresses, and other networking parameters. Each namespace can have its own set of these components, making it possible to create isolated network environments within the same host.

CRI-O uses network namespaces to provide containerized applications with their own virtual view of the network stack, ensuring that each application operates in a separate, isolated environment.
:p What is the purpose of network namespaces in CRI-O?
??x
Network namespaces enable isolation and management of network interfaces, routing tables, IP addresses, and other networking parameters for different containers or pods. In CRI-O, this allows each container to have its own virtual view of the network stack without interfering with others.
x??

---
#### Listing Network Namespaces
The `lsns -t net` command lists all the network namespaces currently in use on a system.

```bash
root@host01:/opt# lsns -t net
         NS TYPE NPROCS   PID USER    NETNSID NSFS                   COMMAND 4026531992 net     114     1 root unassigned                        /sbin/init 4026532196 net       4  5801 root          0 /run/netns/ab8be6e6... /pause
```
:p How can you list all network namespaces on a system?
??x
You can use the `lsns -t net` command to list all network namespaces currently in use on a system.
x??

---
#### Inspecting Network Namespaces with ip netns
The `ip netns list` command is used to display a list of network namespaces.

```bash
root@host01:/opt# ip netns list 7c185da0-04e2-4321-b2eb-da18ceb5fcf6 (id: 2) d26ca6c6-d524-4ae2-b9b7-5489c3db92ce (id: 1) 38bbb724-3420-46f0-bb50-9a150a9f0889 (id: 0)
```
:p How can you list network namespaces using `ip netns`?
??x
You can use the `ip netns list` command to display a list of network namespaces on your system.
x??

---
#### Extracting Network Namespace Path Using crictl and jq
The `crictl inspectp <pod_id>` command combined with `jq` can be used to extract specific information from the container metadata.

```bash
root@host01:/opt# NETNS_PATH=$(crictl inspectp$ B1P_ID | jq -r '.info.runtimeSpec.linux.namespaces[]|select(.type=="network").path')
root@host01:/opt# echo $NETNS_PATH /var/run/netns/7c185da0-04e2-4321-b2eb-da18ceb5fcf6
```
:p How can you extract the network namespace path for a container using `crictl` and `jq`?
??x
You can use `crictl inspectp <pod_id>` to get detailed information about a pod, then pipe that output to `jq` to filter and extract the specific network namespace path.

```bash
NETNS_PATH=$(crictl inspectp$ B1P_ID | jq -r '.info.runtimeSpec.linux.namespaces[]|select(.type=="network").path')
echo $NETNS_PATH
```
x??

---
#### Understanding Network Namespace Isolation with Examples
In the provided text, CRI-O is used to create and manage containers in a way that isolates network configurations. Each container has its own network namespace, allowing multiple processes (like NGINX servers) to bind to port 80 without conflict.

```bash
root@host01:/opt# ip netns list
7c185da0-04e2-4321-b2eb-da18ceb5fcf6 (id: 2)
d26ca6c6-d524-4ae2-b9b7-5489c3db92ce (id: 1)
38bbb724-3420-46f0-bb50-9a150a9f0889 (id: 0)
```
:p How does network namespace isolation work in CRI-O?
??x
Network namespace isolation in CRI-O works by creating a separate namespace for each container or pod, allowing them to have unique networking configurations. This means that even though multiple processes might try to bind to the same port (like port 80), they can do so without conflicting because they operate in different network namespaces.

For example:
```bash
NETNS=$(basename $(crictl inspectp $ B1P_ID | jq -r '.info.runtimeSpec.linux.namespaces[]|select(.type=="network").path'))
```
This command extracts the path to the network namespace of a specific container, allowing you to manage its network configuration independently.
x??

---

#### Extracting Network Namespace Information Using `crictl` and `jq`
We need to extract specific information about network namespaces using tools like `crictl` and `jq`. This process helps in managing container networking more effectively, especially when debugging or scripting. The `jq` tool is used for parsing JSON output from `crictl inspectp` and `crictl inspect`, making the data easier to handle.

:p How do we extract the network namespace information for a container using `crictl`?
??x
To extract the network namespace, you first use `crictl inspectp <pod_id>` or `crictl inspect <container_id>` depending on whether you're dealing with pods or containers. You then parse this output to find the relevant network namespace information.

Example command:
```bash
crictl inspectp <pod_id> | jq '.info.network_settings.network_ns'
```

This command returns a JSON object containing details of the network namespace, which can be further processed using `jq`.

??x
The answer is that you use `crictl inspect` or `crictl inspectp` to gather information and then use `jq` to parse this output for specific fields like the network namespace.

```bash
# Example command
crictl inspectp <pod_id> | jq '.info.network_settings.network_ns'
```

x??

---

#### Using `ps --pid` with Process ID from `ip netns pids`
To understand which processes are using a particular network namespace, we can combine commands to list the processes.

:p How do you determine which processes belong to a specific network namespace?
??x
You use the combination of `ip netns pids <namespace_id>` to get the PIDs of processes in that namespace and then pass these PIDs to `ps --pid` for detailed process information. This way, you can see exactly what commands are running within that namespace.

Example command:
```bash
ps --pid $(ip netns pids$ NETNS)
```

This command retrieves all the processes associated with a specific network namespace by first getting their PIDs and then listing them with `ps`.

??x
The answer is to use `ip netns pids <namespace_id>` to get the process IDs (PIDs) of processes in that namespace, then pass these PIDs to `ps --pid` for detailed information about each process.

```bash
# Example command
ps --pid $(ip netns pids$ NETNS)
```

x??

---

#### Running Commands within a Network Namespace Using `ip netns exec`
The `ip netns exec <namespace_id> <command>` allows you to run commands inside the context of a specific network namespace, providing an environment where networking commands can be executed as if from inside the container.

:p How do you execute a command within a network namespace?
??x
You use the `ip netns exec` command followed by the namespace ID and the command you wish to run. This allows you to execute commands like `ip addr show` or any other networking-related command in the context of that specific network namespace.

Example command:
```bash
ip netns exec $NETNS ip addr
```

This command runs the `ip addr` command inside the specified network namespace, giving you a view of the network interfaces and their configurations as if from within the container.

??x
The answer is to use `ip netns exec <namespace_id> <command>` to run commands within the context of that specific network namespace. For example:
```bash
# Example command
ip netns exec $NETNS ip addr
```

This command runs `ip addr` inside the specified network namespace, providing a view of the network interfaces and their configurations.

x??

---

#### Understanding Network Namespace in CRI-O
CRI-O creates and manages network devices and places them into the appropriate network namespaces. This is crucial for container networking as it isolates each container's network configuration from others.

:p How does CRI-O manage network namespaces?
??x
CRI-O uses Kubernetes' container runtime interface (CRI) to create and manage network namespaces for containers. It ensures that each container has its own isolated network stack, which can be configured independently of other containers.

Example steps:
1. `crictl inspect` is used to get detailed information about a container.
2. Using `jq`, specific fields like the network namespace are extracted from this JSON output.
3. `ip netns exec` and `ps --pid` commands help in verifying which processes use these namespaces and what devices are present.

CRI-O configures these networks as part of its runtime operations, ensuring proper isolation and management of containerized applications.

??x
The answer is that CRI-O manages network namespaces by leveraging Kubernetes' CRI to create and configure isolated network stacks for containers. This involves inspecting containers with `crictl`, parsing the output using `jq`, and then verifying processes and devices within those namespaces using `ip netns exec` and `ps --pid`.

```bash
# Example steps
crictl inspect <container_id> | jq '.info.network_settings.network_ns'
ps --pid $(ip netns pids$ NETNS)
```

x??

---

#### Creating Network Namespaces
Background context: Network namespaces allow for isolation of network stacks, making it possible to run different networking environments within a single host. Each namespace has its own set of devices and routing tables.

:p What command is used to create a new network namespace?
??x
The `ip netns add` command is used to create a new network namespace.
```bash
root@host01:/opt# ip netns add myns
```
x??

---
#### Listing Network Namespaces
Background context: After creating a network namespace, it can be listed and managed using the `ip netns list` command. This helps in verifying that the namespace has been created successfully.

:p What command lists all existing network namespaces?
??x
The `ip netns list` command is used to list all existing network namespaces.
```bash
root@host01:/opt# ip netns list
myns 7c185da0-04e2-4321-b2eb-da18ceb5fcf6 (id: 2)
d26ca6c6-d524-4ae2-b9b7-5489c3db92ce (id: 1)
38bbb724-3420-46f0-bb50-9a150a9f0889 (id: 0)
```
x??

---
#### Upgrading the Network Stack
Background context: The basic network stack in a new namespace includes only a loopback interface. To make it more useful, we need to bring up this interface and ensure it has an IP address.

:p How do you bring up the loopback interface in a network namespace?
??x
You use the `ip netns exec` command along with `ip link set dev lo up` to bring up the loopback interface.
```bash
root@host01:/opt# ip netns exec myns ip link set dev lo up
```
x??

---
#### Creating a veth Pair
Background context: A virtual Ethernet (veth) pair is created to establish communication between the host and the network namespace. Each end of the veth is a separate device, allowing for bidirectional traffic.

:p How do you create a veth pair connecting the host and the network namespace?
??x
The `ip link add` command with the type `veth` creates a veth pair, one side in the host and one in the specified network namespace.
```bash
root@host01:/opt# ip link add myveth-host type veth \                    peer myveth-myns netns myns
```
x??

---
#### Assigning IP Addresses to veth Pair
Background context: After creating a veth pair, each end must be configured with an IP address and brought up to enable communication.

:p How do you assign an IP address to the network namespace side of a veth pair?
??x
You use the `ip addr add` command followed by bringing up the interface using `ip link set dev`.
```bash
root@host01:/opt# ip netns exec myns ip addr add 10.85.0.254/16 \                    dev myveth-myns
```
x??

---
#### Bringing Up veth Pair Interfaces
Background context: Both sides of the veth pair need to be brought up to enable communication between the host and the network namespace.

:p How do you bring up a veth interface on the host?
??x
The `ip link set dev` command is used to bring up the veth interface.
```bash
root@host01:/opt# ip link set dev myveth-host up
```
x??

---

