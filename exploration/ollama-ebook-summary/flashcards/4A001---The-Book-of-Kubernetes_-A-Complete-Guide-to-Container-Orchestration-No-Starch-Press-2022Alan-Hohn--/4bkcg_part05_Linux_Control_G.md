# Flashcards: 4A001---The-Book-of-Kubernetes_-A-Complete-Guide-to-Container-Orchestration-No-Starch-Press-2022Alan-Hohn--_processed (Part 5)

**Starting Chapter:** Linux Control Groups

---

---
#### Process Priority and CPU Usage in Kubernetes
Process priority is a relative measurement. Even low-priority processes can use all available CPU resources if no higher-priority processes are running.
:p What does process priority mean in the context of CPU usage?
??x
In the context of CPU usage, process priority is a way to determine which processes should be given more attention by the kernel when scheduling CPU time. However, it's relative and doesn't guarantee specific CPU allocation; lower-priority processes can still use significant CPU if no higher-priority processes are running.
x??
---
#### Limitations of Process Priority in Kubernetes
Process priority is not suitable for Kubernetes because:
1. Containers need to be allocated to any host with enough resources, regardless of their relative priority.
2. A single Kubernetes cluster may support multiple applications or teams, making it impractical to know every container's priority.
3. Without knowing the CPU usage, Kubernetes cannot efficiently schedule containers on hosts.
:p Why is process priority not sufficient for managing CPU utilization in a Kubernetes environment?
??x
Process priority alone is insufficient because it doesn't provide the necessary guarantees about resource allocation across different hosts and multiple applications or teams using a single cluster. It's difficult to predict how much CPU each container will use, leading to potential imbalance where multiple containers might compete for resources.
x??
---
#### Linux Control Groups (cgroups)
Control groups (cgroups) manage process resource utilization by limiting access to system resources such as CPU, memory, and block devices. Each resource type can have a hierarchy of cgroups associated with it.
:p What is the purpose of control groups (cgroups)?
??x
The purpose of control groups is to provide finer-grained control over how processes use system resources like CPU time, memory, and I/O operations, ensuring that no single process monopolizes critical resources at the expense of others.
x??
---
#### Example of cgroup Directory Structure
Each resource type has its own directory in the cgroups filesystem. For example:
```
/cpu  - for CPU scheduling limits
/memory  - for memory usage limits
/devices - for device access control
```
:p What is the structure of the cgroups filesystem?
??x
The cgroups filesystem organizes resources into directories, each representing a different type of resource that can be controlled. For example, `cpu` controls CPU time, `memory` controls memory usage, and `devices` controls device access.
x??
---
#### Creating and Configuring Cgroups
Cgroups are configured through the specific filesystem interface provided by Linux. The creation and configuration process involve setting limits on resources for processes grouped within a cgroup.
:p How do you create and configure cgroups?
??x
To create and configure cgroups, you typically use shell commands to set resource limits:
```sh
# Create a new cgroup named 'my-cgroup'
mkdir /sys/fs/cgroup/cpu/my-cgroup

# Set CPU limit for the group
echo 1000 > /sys/fs/cgroup/cpu/my-cgroup/cpu.cfs_period_us
echo 900 > /sys/fs/cgroup/cpu/my-cgroup/cpu.cfs_quota_us
```
x??
---

---

#### CPU Control Groups Overview
This section explains how to manage CPU usage using control groups (cgroups) on a Linux system. The `/sys/fs/cgroup/cpu` directory provides insights into how processes are organized and controlled within different cgroups.

:p What is the significance of examining directories like `/sys/fs/cgroup/cpu`?
??x
The significance lies in understanding the hierarchical structure of cgroups and identifying which process belongs to which cgroup, facilitating effective resource management. Each subdirectory represents a separate CPU control group with its own set of configuration files.
```bash
root@host01:/sys/fs/cgroup/cpu# cd /sys/fs/cgroup/cpu
root@host01:/sys/fs/cgroup/cpu# ls -F
cgroup.clone_children  cpuacct.stat               cpuacct.usage       init.scope/      cgroup.sane_behavior   cpu.cfs_period_us    cpu.shares         cpu.stat          system.slice/
cpuacct.usage_all      cpuacct.usage_cpuacct.usage              notify_on_release  cpuacct.usage_percpu  cpuacct.usage_sys     cpuacct.usage_user  tasks             user.slice/
```
x??

---

#### Identifying the CPU Control Group of a Process
This part explains how to determine which cgroup a specific process is assigned to, using tools like `pgrep` and `grep`.

:p How can you find out which cgroup a running process belongs to?
??x
You can use the `pgrep` command along with `grep` to identify the cgroup of a process. The example shows how to locate the specific directory within `/sys/fs/cgroup/cpu` that contains information about a particular process.

```bash
root@host01:/sys/fs/cgroup/cpu# grep -R $(pgrep stress-ng-cpu) system.slice/runc-050c.../cgroup.procs:5964
root@host01:/sys/fs/cgroup/cpu# grep -R $(pgrep stress-ng-cpu) system.slice/runc-050c.../tasks:5964
```
These commands search the cgroup hierarchy for the process ID (PID) of the `stress-ng` command, indicating that it is part of a specific cgroup under `system.slice`.

x??

---

#### Understanding CPU Control Group Configuration Files
This section details what configuration files are present in the `/sys/fs/cgroup/cpu` directories and their functions.

:p What are the key configuration files found in the root directory of a CPU control group, and what do they represent?
??x
The key configuration files include:
- `cpu.shares`: Determines the relative share of CPU resources for this cgroup.
- `cpu.cfs_period_us`: Defines the length of a period in microseconds.
- `cpu.cfs_quota_us`: Specifies the maximum amount of CPU time (in microseconds) allowed during each period.

```bash
root@host01:/sys/fs/cgroup/cpu# ls -F system.slice/runc-${CUL_ID}.scope/
cgroup.clone_children  cpu.uclamp.max        cpuacct.usage_percpu_sys cgroup.procs           cpu.uclamp.min        cpuacct.usage_percpu_user
cpu.cfs_period_us      cpuacct.stat          cpuacct.usage_sys         cpuacct.usage         cpuacct.usage_all     notify_on_release
cpu.cfs_quota_us       cpuacct.usage         cpuacct.usage_user        cpuacct.usage_cpu    cpuacct.usage_percpu  tasks
```
These files control the distribution and limits of CPU resources within the cgroup.

x??

---

#### Practical Example: Limiting CPU Usage with `stress-ng`
This example demonstrates how to limit the CPU usage of a running process using cgroups.

:p How can you use the `stress-ng` command in conjunction with cgroups to limit CPU usage?
??x
By moving a process into a specific cgroup and configuring its limits, you can effectively control the amount of CPU time it uses. The example shows how to verify the current CPU usage and then set limitations.

```bash
root@host01:/sys/fs/cgroup/cpu# top -b -n 1 -p $(pgrep -d , stress)
```
This command checks the current CPU usage before any changes are made.

Next, you can limit the CPU usage by modifying files like `cpu.cfs_period_us` and `cpu.cfs_quota_us` in the cgroup directory of the process. For instance:

```bash
root@host01:/sys/fs/cgroup/cpu/system.slice/runc-${CUL_ID}.scope# echo 5000 > cpu.cfs_period_us
root@host01:/sys/fs/cgroup/cpu/system.slice/runc-${CUL_ID}.scope# echo 4000 > cpu.cfs_quota_us
```
This configuration limits the process to use only 80% of the CPU time during each period.

x??

---

#### Setting Absolute Quota on CPU Usage Using cgroup
Background context: In this scenario, we are managing a container's CPU usage by setting an absolute quota. The system uses cgroups to limit resource utilization. We can adjust `cpu.cfs_quota_us` and `cpu.cfs_period_us` to control the amount of CPU time allowed per period.

The formula for calculating the percentage of CPU is:
$$\text{Percentage of CPU} = \left( \frac{\text{quota}}{\text{period}} \right) \times 100$$:p How can we set a quota on the stress container to limit its CPU usage?
??x
To set a quota, we first need to know the period defined by `cpu.cfs_period_us`. In this case, it is set to 100,000 μs (or 0.1 seconds). By updating `cpu.cfs_quota_us`, we can limit the amount of CPU time the container can use.

For example, setting `cpu.cfs_quota_us` to 50,000 allows the processes in this cgroup 50,000 μs out of every 100,000 μs, which results in approximately 50% of a CPU.
```bash
echo "50000" > cpu.cfs_quota_us
```
x??

---

#### Verifying the CPU Usage Limitation
Background context: After setting the quota, we need to verify that the stress container is indeed limited by the configured CPU usage.

:p How can we confirm that the stress container's CPU usage has been limited?
??x
We can use `top` or other monitoring tools to check the CPU usage of the processes. For example:
```bash
top -b -n 1 -p$(pgrep -d , stress)
```
This command will show the current CPU usage percentage for the `stress` container.

In our case, we see that the stress container is using approximately 50% of one CPU.
x??

---

#### Stopping the Stress Container
Background context: After verifying the settings and observing the impact, it's important to stop the container to avoid any ongoing resource consumption.

:p How do you stop a container managed by CRI-O?
??x
To stop a container using `crictl`, you need to first stop the pod and then remove the sandbox. The commands are as follows:
```bash
crictl stopp <PUL_ID>
crictl rmp <PUL_ID>
```
These commands will stop and remove the specified container, ensuring it no longer consumes resources.
x??

---

#### Configuring CPU Limits in crictl YAML Files
Background context: For better management, we can configure CPU limits directly in the `crictl` YAML files to enforce them by CRI-O. This eliminates the need for manual intervention each time.

:p How do you specify CPU quotas and periods in the crictl YAML configuration?
??x
In the container configuration file (e.g., `co-clim.yaml`), you can set `cpu_period` and `cpu_quota` to define the limits:
```yaml
linux:
  resources:
    cpu_period: 100000
    cpu_quota: 10000
```
- `cpu_period`: This corresponds to `cpu.cfs_period_us`, defining the length of each period in microseconds.
- `cpu_quota`: This corresponds to `cpu.cfs_quota_us`, setting the amount of CPU time allowed per period.

These settings limit the stress container to 10% of a CPU on average.
x??

---

#### Launching a Container with Specified CPU Limits
Background context: After configuring, we can launch the container and verify that it adheres to the defined limits.

:p How do you launch a container with specified CPU limits using crictl?
??x
You can launch the container by specifying the pod and container configurations in `crictl`:
```bash
PCL_ID=$(crictl runp po-clim.yaml)
CCL_ID=$(crictl create$ PCL_ID co-clim.yaml po-clim.yaml)
crictl start $CCL_ID
```
This command sequence creates the pod, attaches the container to it, and starts the container.

After starting the container, you can confirm its CPU usage using `top` or other monitoring tools.
x??

---

---
#### CPU Usage Monitoring
Background context: The provided text discusses monitoring CPU usage using `top`, a utility that provides real-time information about processes running on a system. The example shows how to interpret the output of `top` to determine which processes are using CPU resources and how much.
:p How does `top` help in monitoring CPU usage?
??x
`top` is a powerful tool for monitoring system performance by displaying a dynamic view of active processes, including their CPU usage percentage. The example shows that while one process (stress-ng) has 10.0% CPU usage and another (also stress-ng) has 0.0%, the overall CPU usage reported by `top` is 10.3% user space CPU time.

```shell
Tasks:   4 total,   2 running,   2 sleeping,   0 stopped,   0 zombie
percentCpu(s): 10.3 us, 0.0 sy, 0.0 ni, 89.7 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
```
This output indicates that the system is using 10.3% of its CPU in user space (us).

x??
---

#### Cgroup Configuration for CPU Quota
Background context: The text explains how Kubernetes' CRI-O runtime manages and enforces CPU quotas through cgroups, which are a Linux kernel feature allowing control over resources such as CPU time allocation.
:p How does CRI-O manage CPU usage in containers using cgroups?
??x
CRI-O creates a hierarchical structure of cgroups to allocate specific CPU resources to containers. In the example, CRI-O sets up a `pod.slice` hierarchy and configures the container's cgroup with a maximum CPU quota.

```shell
root@host01:/opt# cd /sys/fs/cgroup/cpu/pod.slice
root@host01:...pod.slice# cat crio-$CCL_ID.scope/cpu.cfs_quota_us 10000
```
This command shows that the container's cgroup (`crio-$CCL_ID.scope`) has a CPU quota set to `10000` microseconds, which limits how much CPU time the process can use.

x??
---

#### Memory Limits in Linux
Background context: The provided text discusses memory management and limitations on processes. It explains that setting a virtual memory limit ensures that a process cannot exceed a certain amount of memory, including both RAM and swap space.
:p How does `ulimit` help manage memory usage in Linux processes?
??x
The `ulimit` command allows setting resource limits for processes running from the current shell session. The example sets a maximum virtual memory limit to 256 MiB.

```shell
root@host01:~# ulimit -v 262144
```
This command restricts each process started from this shell session to use up to 262144 bytes (256 MiB) of virtual memory. The limit prevents processes from consuming excessive swap space.

To verify the limit, attempting to allocate more than the allowed amount results in a failure:

```shell
root@host01:~# cat /dev/zero | head -c 500m | tail
tail: memory exhausted
```
The `cat /dev/zero` command reads from `/dev/zero`, and `head -c 500m` attempts to keep the first 500 MiB in memory. However, when it tries to allocate more space with `tail`, the limit is hit, causing an error.

x??
---

#### Unix Process Memory Limits
Unix process memory limits allow us to control how much memory a single process can use. However, they are not sufficient for container environments due to their limitations:
- They apply only to individual processes or entire users, which does not fit the structure of containers that have multiple related processes.
- Applying limits per user is ineffective in container orchestration systems like Kubernetes because all containers run under the same user.

:p What are the main reasons Unix process memory limits cannot be used effectively for containers?
??x
Unix process memory limits are insufficient for container environments because they:
1. Apply only to individual processes or entire users, not groups of related processes.
2. Ineffective in a container orchestration environment like Kubernetes where all containers belong to the same user.

This makes it challenging to manage resources across multiple processes within a single container and between different containers.
x??

---

#### Cgroups for Memory Limits
Cgroups (Control Groups) are used to enforce memory limits effectively in container environments. They allow setting resource limits at the cgroup level, ensuring that all processes within a container adhere to those limits.

:p How does using cgroups help with managing memory limits in containers?
??x
Using cgroups helps manage memory limits by allowing you to set resource constraints at the cgroup level. This ensures that all processes within a container are constrained by these limits, rather than applying limits individually or at the user level.
```yaml
# Example YAML configuration for setting memory limit with cgroups
---
metadata:
  name: stress2
  namespace: crio linux:
    cgroup_parent: pod.slice
    security_context:
      namespace_options:
        network: 2

---
metadata:
  name: stress2
  image: docker.io/bookofkubernetes/stress:stable
  args:
    - "--vm"
    - "1"
    - "--vm-bytes"
    - "512M"
    - "-v"
  linux:
    resources:
      memory_limit_in_bytes: 268435456  # 256MiB
      cpu_period: 100000
      cpu_quota: 10000
```
x??

---

#### Applying Memory Limits with Cgroups in Kubernetes
When using Kubernetes, you can configure memory limits for containers within a Pod by specifying the `memory_limit_in_bytes` parameter. This ensures that the container does not exceed its allocated memory.

:p How do you apply a memory limit to a container in Kubernetes using cgroups?
??x
To apply a memory limit to a container in Kubernetes using cgroups, you need to specify the `memory_limit_in_bytes` parameter in the container's configuration. Here’s an example:
```yaml
# YAML configuration for setting a memory limit with cgroups in Kubernetes
---
metadata:
  name: stress2
  namespace: crio linux:
    cgroup_parent: pod.slice
    security_context:
      namespace_options:
        network: 2

---
metadata:
  name: stress2
  image: docker.io/bookofkubernetes/stress:stable
  args:
    - "--vm"
    - "1"
    - "--vm-bytes"
    - "512M"
    - "-v"
  linux:
    resources:
      memory_limit_in_bytes: 268435456  # 256MiB
      cpu_period: 100000
      cpu_quota: 10000
```
This configuration ensures that the `stress` container is limited to 256 MiB of memory and 10% CPU (10,000 microseconds per period with a quota of 10,000 microseconds).
x??

---

#### Running Containers with Memory Limits in CRI-O
To run containers with memory limits using CRI-O, you need to define the Pod and Container configurations as shown. This ensures that the container adheres to the specified resource limits.

:p How do you create a Pod and Container configuration for running a container with memory limits in CRI-O?
??x
To create a Pod and Container configuration for running a container with memory limits in CRI-O, follow these steps:

1. Define the Pod configuration:
```yaml
# pod-mlim.yaml
---
metadata:
  name: stress2
  namespace: crio linux:
    cgroup_parent: pod.slice
    security_context:
      namespace_options:
        network: 2
```

2. Define the Container configuration within the Pod:
```yaml
# co-mlim.yaml
---
metadata:
  name: stress2
  image: docker.io/bookofkubernetes/stress:stable
  args:
    - "--vm"
    - "1"
    - "--vm-bytes"
    - "512M"
    - "-v"
  linux:
    resources:
      memory_limit_in_bytes: 268435456  # 256MiB
      cpu_period: 100000
      cpu_quota: 10000
```

3. Run the container using `crictl` commands:
```sh
root@host01:~# cd /opt
root@host01:/opt# PML_ID=$(crictl runp po-mlim.yaml)
root@host01:/opt# CML_ID=$(crictl create$ PML_ID co-mlim.yaml po-mlim.yaml)
root@host01:/opt# crictl start $CML_ID
```

This ensures that the `stress` container runs with a memory limit of 256 MiB and a CPU quota of 10,000 microseconds.
x??

---

#### OOM Killer in Container Environments
The context is about understanding why containers might be killed by the Out of Memory (OOM) killer, especially in a scenario where memory limits are exceeded. The OOM killer intervenes when the system runs out of memory, killing processes to prevent the entire system from crashing.
:p Why does the OOM killer intervene with stress-ng-vm?
??x
The OOM killer intervenes because the system is running low on memory and needs to free up resources by terminating some processes. In this case, `stress-ng-vm` repeatedly hits its memory limit set by the container runtime (e.g., CRI-O), causing it to be terminated with a `SIGKILL` signal.
```shell
stress-ng: debug: [11] stress-ng-vm: child died: signal 9 'SIGKILL' (instance 0)
```
x??

---

#### OOM Killer vs. Memory Limits
To understand the difference, we need to know that regular memory limits cause allocation failures rather than process termination by the OOM killer.
:p How does the OOM killer differ from setting a simple memory limit?
??x
The OOM killer is triggered when the system is critically low on memory and there are no other processes willing to release their memory. In contrast, setting a simple memory limit causes any attempt to exceed that limit to fail immediately without involving the OOM killer.
```shell
# Example of a process hitting a memory limit
stress-ng: info:  [6] dispatching hogs: 1 vm ...
stress-ng: debug: [11] stress-ng-vm: started [11] (instance 0)
stress-ng: debug: [11] stress-ng-vm using method 'all'
```
x??

---

#### OOM Killer in Containerized Environments
The context explains that container environments use the OOM killer to manage memory usage, especially when scaling and reliability are key concerns.
:p Why is the OOM killer used in containerized environments?
??x
The OOM killer is used because containers are designed to be lightweight and scalable. If a process exceeds its allocated memory, it's better to kill the process quickly rather than let it hang, which could impact other services or even crash the system.
```shell
# Example of OOM killer action
[  696.651056] oom_reaper: reaped process 8756 (stress-ng-vm)...
```
x??

---

#### Memory Management in Containers
The text discusses how containers manage memory and the risks associated with setting strict limits.
:p What is a potential downside of using strict memory limits without OOM killer intervention?
??x
A potential downside is that processes may hang or become unresponsive if they exceed their allocated memory, potentially leading to system instability. The alternative—using the OOM killer—is generally considered safer as it ensures processes are terminated quickly and cleanly.
```shell
stress-ng: debug: [11] stress-ng-vm: child died: signal 9 'SIGKILL' (instance 0)
```
x??

---

#### Turning Off OOM Killer for a Container
The context explains that while the default is to use the OOM killer, it's possible to disable this feature. However, disabling can lead to processes being paused rather than terminated.
:p What happens if you turn off the OOM killer for a container?
??x
Turning off the OOM killer means that instead of terminating the process with `SIGKILL`, the process will be paused until memory is freed by other processes in the same cgroup. This can lead to unproductive processes, as they are not killed but also not running effectively.
```shell
# Example of a process being paused
stress-ng: debug: [11] stress-ng-vm: assuming killed by OOM killer, restarting again...
```
x??

---

#### Managing the Stress Container
The context provides instructions on how to handle a continuously failing `stress` container.
:p How should we address the continuously failing `stress` container?
??x
To stop the continuously failing `stress` container, you can use commands like `kill` or `stop` in your container management tool. For example:
```shell
# Example of stopping a container
root@host01:/opt# docker stop $CML_ID
```
This stops the container without necessarily addressing the root cause (e.g., excessive memory usage).
x??

---

#### Stopping and Removing Containers and Pods
Stopping and removing containers and pods is crucial for managing resources effectively. The commands `crictl stop` and `crictl rm` are used to manage these resources, ensuring that unused processes do not consume unnecessary CPU or memory.

:p What command is used to stop a container?
??x
The `crictl stop` command is used to stop a container by its ID.
```bash
crictl stop $CML_ID
```
x??

---

#### Stopping and Removing Pods
Similar to stopping containers, pods can also be stopped and removed using the `crictl` tool. This ensures that resources are freed up for other processes.

:p What command is used to remove a pod?
??x
The `crictl rm` command is used to remove a pod by its ID.
```bash
crictl rm $PML_ID
```
x??

---

#### Proper Network Management
Proper network management involves controlling both ingress and egress traffic. However, Linux kernel's cgroup capabilities are limited when it comes to managing ingress traffic at the host level.

:p Why can't we control ingress traffic using cgroups?
??x
Ingress traffic cannot be controlled by cgroups directly because the system has no control over incoming data; it only manages outgoing (egress) traffic. For precise network management, configuration at a switch or router is necessary.
x??

---

#### Network Bandwidth Control with `tc`
To limit egress bandwidth, tools like `tc` (Traffic Control) can be used to apply rate limits and ensure that processes do not saturate the available bandwidth.

:p How is the `tc qdisc add dev` command used to set a quota for outgoing traffic?
??x
The `tc qdisc add dev` command sets up a traffic control discipline on the specified network interface (`$IFACE`). Here, it adds a TBF (Token Bucket Filter) with a rate limit of 100mbit and burst size of 256kbit.

```bash
root@host01:~# IFACE=$(ip -o addr | grep 192.168.61.11 | awk '{print$2}')
root@host01:~# tc qdisc add dev $IFACE root tbf rate 100mbit \
    burst 256kbit latency 400ms
```
This command ensures that outgoing traffic does not exceed the specified rate, providing a more reliable network environment.

x??

---

#### Measuring Network Performance with `iperf3`
`iperf3` is used to test network performance before and after applying bandwidth limits. This helps in validating whether the configured limits are effective.

:p How can we measure egress bandwidth using `iperf3`?
??x
To measure egress bandwidth, you can use the `iperf3` command from a client machine (`host01`) to a server running on another machine (`host02`).

```bash
root@host01:~# iperf3 -c 192.168.61.12
Connecting to host 192.168.61.12, port 5201

[ ID] Interval           Transfer     Bitrate         Retr
[  5]   0.00-10.00  sec  2.18 GBytes  1.87 Gbits/sec  13184             sender
[  5]   0.00-10.00  sec  2.18 GBytes  1.87 Gbits/sec                  receiver
```
This output shows the initial bandwidth performance before any limits are applied.

x??

---

