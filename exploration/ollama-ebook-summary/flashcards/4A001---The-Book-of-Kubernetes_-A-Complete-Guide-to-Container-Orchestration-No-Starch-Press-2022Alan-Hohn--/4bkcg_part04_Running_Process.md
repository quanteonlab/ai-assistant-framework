# Flashcards: 4A001---The-Book-of-Kubernetes_-A-Complete-Guide-to-Container-Orchestration-No-Starch-Press-2022Alan-Hohn--_processed (Part 4)

**Starting Chapter:** Running Processes in Namespaces Directly

---

#### crictl inspect and Extracting PID
The `crictl inspect` command provides a wealth of information about containers, which can be accessed via JSON output. The `jq` tool is used to extract specific fields like the PID from this output.

:p How do you use `crictl` and `jq` to get the PID of a container?
??x
To use `crictl inspect` with `jq`, first run:
```bash
crictl inspect $CONTAINER_ID | jq '.info.id'
```
This command inspects the specified container ID and extracts the PID from the JSON output. The `.info.id` path in `jq` accesses the specific field containing the PID.

x??

---

#### Understanding CRI-O Namespaces
CRI-O (Container Runtime Interface - Open Source) handles namespaces differently compared to other container runtimes. For a given container, CRI-O typically creates only 4 types of namespaces: UTS, IPC, MNT, and PID. The lack of a network namespace suggests that the host's network is shared.

:p List the types of namespaces created by CRI-O for a typical container.
??x
CRI-O generally creates four types of namespaces for containers:

- **UTS**: Names the container.
- **IPC**: Manages inter-process communication mechanisms.
- **MNT**: Handles mount points and filesystems.
- **PID**: Manages processes within the namespace.

These namespaces are often associated with the `pause` command, which is a placeholder process used by CRI-O to manage containers. Additional MNT namespaces might exist if multiple containers share a pod.

x??

---

#### Running Processes Directly in Namespaces
To run processes directly in namespaces, you can use the `unshare` command. This allows you to create new namespaces and run a program isolated from others in the system. The `-p` flag creates a PID namespace, while `--mount-proc` ensures `/proc` is correctly remounted.

:p How do you use `unshare` to run a process in a PID namespace?
??x
To run a process in a PID namespace using `unshare`, execute the following command:

```bash
unshare -f -p --mount-proc -- /bin/sh -c /bin/bash
```

This command does the following:
- `-f`: Create a new filesystem namespace.
- `-p`: Create a new PID namespace.
- `--mount-proc`: Ensure `/proc` is remounted correctly to show the correct process information.
- `-- /bin/sh -c /bin/bash`: Run a shell that starts another shell.

:p What happens when you run this command?
??x
When you run the `unshare` command, it creates a new PID namespace and runs `/bin/sh`, which then executes `/bin/bash`. This process cannot see other processes outside its namespace. The resulting output of `ps -ef` will only show the commands run within that specific namespace.

```bash
root           1       0  0 22:21 pts/0    00:00:00 /bin/sh -c /bin/bash
root           2       1  0 22:21 pts/0    00:00:00 /bin/bash
```

x??

---

#### Identifying Namespaces in Linux
The `lsns` command lists all namespaces on a system. Each namespace entry includes its type, number of processes, PID, and the command associated with it.

:p How do you identify a specific process's namespace using `lsns`?
??x
To find the namespace related to a specific process, use:

```bash
ls -l /proc/<PID>/ns/pid
```

This will show which PID namespace a process is in. For example:

```bash
root@host01:~# ls -l /proc/12345/ns/pid
lrwxrwxrwx 1 root root 0 Mar 6 18:00 /proc/12345/ns/pid -> 'pid:[4026532190]'
```

:p How do you list all namespaces and find the one related to a process?
??x
To list all namespaces, use:

```bash
lsns
```

Then, filter or search for the namespace associated with your specific process. For example, if the PID is 12345, look for entries in `lsns` that match this PID.

For instance, running `lsns` might show:

```bash
root@host01:~# lsns
         NS TYPE   NPROCS   PID USER            COMMAND ...
9876      mnt    2       12345 root /bin/bash ...
```

This shows that the MNT namespace for process 12345 is `mnt:[9876]`.

x??

---

#### Process Management and Signaling in Containers
Background context: In container environments, processes need to be managed properly. This involves understanding how signals are passed between a parent process (like `sh`) and its children processes. Proper handling of signals is crucial to avoid issues like zombie processes when containers are terminated.
:p How does the shell (`sh`) handle signal passing in a container environment?
??x
In the given context, `sh` handles signal passing by ensuring that it correctly forwards signals to its child processes. When a kill signal is sent to `sh`, it propagates this signal to all of its children, causing them to terminate properly.
```bash
# Example command to simulate sending a kill signal from outside the namespace
kill -9 12111
```
x??

---

#### Signaling and Process Termination
Background context: The example demonstrates how `sh` handles termination signals by passing them on to its children processes. This is important for managing the lifecycle of processes within containers, ensuring they are terminated gracefully.
:p What happens when a kill signal is sent to `sh` from outside the namespace?
??x
When a kill signal (e.g., `-9`) is sent to `sh`, it forwards this signal to its child processes. As a result, all related processes terminate correctly, preventing issues like zombie processes. In the given example, when the command `kill -9 12111` is executed from outside the namespace, `sh` receives and forwards this signal to its children.
```bash
# Example command to simulate sending a kill signal from outside the namespace
kill -9 12111
```
x??

---

#### Namespace and Process Isolation in Containers
Background context: The text explains that containers create an isolated environment for processes, similar to traditional process isolation techniques. However, this isolation is achieved through Linux namespaces rather than full virtualization.
:p What are the key differences between container isolation and virtual machine (VM) environments?
??x
Container isolation, facilitated by Linux namespaces, isolates processes at a higher level compared to VMs. Unlike VMs, containers share the same kernel but provide isolated views of resources like network devices, file systems, and process IDs. This means that while each container has its own virtualized environment for these resources, they still share the underlying hardware and operating system.
x??

---

#### Resource Sharing in Containers
Background context: The text emphasizes that processes within a container share the same CPU, memory, and network resources. While namespaces provide isolation at the process level, resource limits need to be set to prevent one process from hogging all available resources.
:p How do containers handle shared resources like CPU, memory, and network?
??x
Processes in containers share the same underlying hardware resources (CPU, memory, and network). This sharing can lead to issues if a single process consumes too many resources. However, namespaces alone cannot address resource overutilization; therefore, resource limiting mechanisms must be implemented.
```bash
# Example command for setting resource limits using cgroups
echo 1024 > /sys/fs/cgroup/cpu/cpu.shares
```
x??

---

#### Container Runtimes and Namespaces
Background context: The example demonstrates the use of `containerd` and CRI-O, which employ namespaces to isolate containers from each other. This isolation prevents processes in one container from interfering with those in another.
:p How do `containerd` and CRI-O ensure process isolation between different containers?
??x
`containerd` and CRI-O use Linux namespaces to provide a separate view of system resources for each container. By utilizing namespaces, they isolate processes such as network devices, file systems, and user IDs. This ensures that processes running in one container cannot interfere with those in another.
```bash
# Example command to list namespace information
nsenter --mount=/proc/1/mounts --uts=hostname --ipc=ipcs --net=iproute2 --pid=1 -- /bin/bash
```
x??

---

#### Process Interference and Resource Management
Background context: The text points out that while namespaces provide isolation, they do not prevent processes from consuming too many resources. This can affect other processes within the container.
:p How does resource consumption by one process impact others in a shared namespace?
??x
If a single process consumes excessive CPU, memory, or network resources, it can starve other processes running in the same container of these resources. Namespaces provide isolation but do not enforce strict limits on resource usage. To manage this, resource management tools must be used to ensure fair distribution and prevent overconsumption.
x??

---

#### Process Isolation and Resource Management
Background context: In Chapter 2, we discussed process isolation to prevent processes from affecting each other. However, without resource limits, a process could still consume too many resources (CPU, memory, network), impacting others.

:p What is the importance of limiting CPU, memory, and network resources for processes?
??x
Limiting these resources ensures that no single process consumes all available resources, thereby preventing starvation or degradation in performance for other processes. This is crucial for resource management in a containerized environment like Kubernetes.
x??

---

#### Completely Fair Scheduler (CFS)
Background context: The Linux kernel uses the Completely Fair Scheduler (CFS) to manage CPU scheduling among processes.

:p How does CFS ensure fairness?
??x
CFS ensures fairness by dynamically allocating CPU time based on the priority of each process. It creates a prioritized list, giving higher priority to more important processes and ensuring that all processes get an opportunity to run.
x??

---

#### Real-Time vs Non-Real-Time Policies
Background context: The scheduler supports different policies, categorized into real-time and non-real-time. Real-time processes have critical deadlines, while non-real-time processes are less urgent.

:p What distinguishes real-time from non-real-time policies?
??x
Real-time policies prioritize processes that must complete tasks before a deadline (e.g., data collection from hardware). Non-real-time policies handle tasks with no strict deadlines but still need fair scheduling.
x??

---

#### Scheduling Policy and Priority
Background context: CFS schedules processes based on both the policy and priority within that policy.

:p How does CFS determine which process to run next?
??x
CFS determines the next process by considering both the scheduling policy (real-time vs. non-real-time) and the priority of each process under that policy. Real-time processes are always given higher priority than non-real-time ones.
x??

---

#### CPU Limits in Kubernetes
Background context: Applying CPU limits ensures that a container does not consume more resources than allocated, providing certainty for resource allocation.

:p How can we apply CPU limits to containers?
??x
To apply CPU limits, you define the maximum amount of CPU (in cores or millicores) a container can use. For example, `cpu-limit: 500m` means the container cannot consume more than 500 millicores.
x??

---

#### Real-Time Processes Example
Background context: Real-time processes have critical deadlines and are given higher priority.

:p What is an example of a real-time process?
??x
An example of a real-time process could be a system collecting data from an embedded hardware device. The process must read the data before the hardware buffer overflows, making it crucial to prioritize this process.
x??

---

#### Non-Real-Time Processes Example
Background context: Non-real-time processes do not have strict deadlines but still require fair scheduling.

:p What is an example of a non-real-time process?
??x
An example of a non-real-time process could be a web server handling requests. While timely response is important, it does not have the same critical deadline as real-time tasks.
x??

---

#### Running Examples from Repository
Background context: The book provides examples in a GitHub repository for practical learning.

:p How do I run an example from the provided repository?
??x
To run an example, first clone the repository:
```bash
git clone https://github.com/book-of-kubernetes/examples.git
```
Then navigate to the appropriate directory and follow any specific instructions provided.
x??

---

#### Linux `ps` Command and Process Scheduling Policies
The `ps` command provides information about processes running on a system. It can be used with different options to display various attributes of each process, such as the scheduling class (CLS), real-time priority (RTPRIO), nice level (NI), and the command name (comm).

:p What is the output of the `ps -e -o pid,class,rtprio,ni,comm` command telling us about processes running on a system?
??x
The `ps -e -o pid,class,rtprio,ni,comm` command outputs a list of all processes running on the system along with their process ID (PID), scheduling class (CLS), real-time priority (RTPRIO), nice level (NI), and command name (comm).

For example:
```
root@host01:~# ps -e -o pid,class,rtprio,ni,comm
 PID CLS RTPRIO NI COMMAND
 1   TS    -     0 systemd ...
 6   TS    -    -20 kworker/0:0H-kblockd ...
 11  FF    99   - migration/0
 12  FF    50   - idle_inject/0
 85  FF    99   - watchdogd ...
 484 RR    99   - multipathd ...
 7967 TS    -     0 ps
```

This output indicates that some processes are time-sharing (TS), which means they follow the default scheduling policy. Processes with a real-time policy (FF) have higher priority over non-real-time policies. Real-time processes like `watchdogd` and `multipathd` require high-priority execution to ensure system stability.

The nice level (-20 to 19) affects non-real-time processes, where -20 is the highest priority.
x??

---
#### Scheduling Policies in Linux
Linux supports multiple scheduling policies for different types of tasks. The most common are time-sharing (TS), first-in-first-out (FIFO or FF), and round-robin (RR).

:p What does the `ps` command output tell us about a process's scheduling class?
??x
The `ps -e -o pid,class,rtprio,ni,comm` command outputs the scheduling class of each process. The class can be:
- TS: Time-sharing policy for normal processes.
- FF: First-in-first-out (FIFO) or real-time priority.

For example:
```
PID CLS RTPRIO NI COMMAND
1   TS    -     0 systemd ...
6   TS    -    -20 kworker/0:0H-kblockd ...
11  FF    99   - migration/0
12  FF    50   - idle_inject/0
85  FF    99   - watchdogd ...
484 RR    99   - multipathd ...
7967 TS    -     0 ps
```

Here, `systemd` and other non-real-time processes are listed as `TS`, while `watchdogd` is a real-time process with the `FF` class.
x??

---
#### Setting Process Priorities in Linux
Linux provides mechanisms to set and control process priorities. The priority can be adjusted using tools like `renice` or directly through the scheduling policy.

:p What are the two numeric fields that indicate how processes are prioritized within their respective policies?
??x
The `ps -e -o pid,class,rtprio,ni,comm` command provides two numeric fields to indicate process priorities:
- **RTPRIO**: Real-time priority (applies only to real-time processes).
- **NI**: Nice level (applies only to non-real-time processes).

For example:
```
PID CLS RTPRIO NI COMMAND
1   TS    -     0 systemd ...
6   TS    -    -20 kworker/0:0H-kblockd ...
11  FF    99   - migration/0
12  FF    50   - idle_inject/0
85  FF    99   - watchdogd ...
484 RR    99   - multipathd ...
7967 TS    -     0 ps
```

The `NI` field ranges from -20 (highest priority) to 19 (lowest priority), while the `RTPRIO` field indicates real-time priorities.
x??

---
#### Running a Containerized Process with CRI-O
CRI-O is a container runtime for Kubernetes clusters. It allows running and managing containers on Linux systems.

:p How can you start a container using CRI-O, and what are the steps involved?
??x
To start a container using CRI-O, you need to define Pod and Container YAML files, pull the container image, and then run the container through `crictl`.

1. **Pull the Image:**
   ```sh
   root@host01:/opt# crictl pull docker.io/bookofkubernetes/stress:stable
   ```

2. **Create a Pod and Container using YAML Files:**

   - Pod YAML (`po-nolim.yaml`):
     ```yaml
     ---
     metadata:
       name: stress
       namespace: crio
     linux:
       security_context:
         namespace_options:
           network: 2
     ```

   - Container YAML (`co-nolim.yaml`):
     ```yaml
     ---
     metadata:
       name: stress
     image:
       image: docker.io/bookofkubernetes/stress:stable
     args:
       - "--cpu"
       - "1"
       - "-v"
     ```

3. **Run the Pod and Container:**
   ```sh
   root@host01:/opt# cd /opt
   root@host01:/opt# PUL_ID=$(crictl runp po-nolim.yaml)
   root@host01:/opt# CUL_ID=$(crictl create $PUL_ID co-nolim.yaml po-nolim.yaml)
   root@host01:/opt# crictl start $CUL_ID
   ```

4. **Verify the Container is Running:**
   ```sh
   root@host01:/opt# crictl ps
   ```

These steps ensure that a containerized process, specifically `stress`, runs in a defined environment using CRI-O.
x??

---

#### Checking Container Status Using `crictl ps`
Background context: The command `crictl ps` is used to check the status of containers running on a container runtime, ensuring that the desired container is running as expected. This is particularly useful when setting up and verifying containerized applications.

:p What does the `crictl ps` command do?
??x
The `crictl ps` command lists all the running containers along with their details such as state, image, and resource usage. It helps in confirming that a specific container or application is running correctly.
```bash
root@host01:/opt# crictl ps
```
x??

---

#### Using `top` Command to Monitor CPU Usage
Background context: The `top` command provides real-time monitoring of system processes and resources, including CPU usage. It helps in identifying which processes are consuming the most resources, which is crucial for performance optimization.

:p How can you use the `top` command to check the current CPU usage?
??x
You can use the `top` command with specific options to monitor the CPU usage of a particular process. For example, running `top -b -n 1 -p <PID>` will give detailed information about the specified process in batch mode (output goes to stdout), for one iteration, and targeting a specific PID.
```bash
root@host01:/opt# top -b -n 1 -p $(pgrep -d , stress)
```
x??

---

#### Changing Process Priority with `renice`
Background context: The `renice` command is used to change the priority of running processes, which affects how they are scheduled by the operating system. This can be useful for adjusting resource allocation based on specific needs or priorities.

:p How do you use the `renice` command to change process priority?
??x
The `renice` command changes the priority (nice value) of a running process. The `-n` option is used to specify the new nice value, and the `-p` option specifies the process ID (PID). The old and new priorities are displayed.
```bash
root@host01:/opt# renice -n 19 -p $(pgrep -d ' ' stress)
```
x??

---

#### Understanding Nice Values
Background context: In Unix-like operating systems, the `nice` value is used to determine a process's priority. Lower values (e.g., 0) mean higher priority, while higher values (e.g., 19) mean lower priority. The default nice value for processes created by users is 20.

:p What does a nice value of 19 signify in the context of process scheduling?
??x
A nice value of 19 signifies that the process has a lower priority compared to others. Processes with a higher nice value are given less CPU time, allowing more urgent or critical processes to run first.
```bash
root@host01:/opt# top -b -n 1 -p $(pgrep -d , stress)
```
x??

---

#### Observing Process Changes After Renice
Background context: After changing the priority of a process using `renice`, you can observe changes in its scheduling and resource allocation by monitoring it again with tools like `top`.

:p How do you verify that the priority change has been applied to the stress processes?
??x
After running the `renice` command, you can verify the change by checking the output of the `top` command again. The nice value in the `NI` column should reflect the new setting.
```bash
root@host01:/opt# top -b -n 1 -p $(pgrep -d , stress)
```
x??

---

