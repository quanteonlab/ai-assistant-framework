# Flashcards: 4A001---The-Book-of-Kubernetes_-A-Complete-Guide-to-Container-Orchestration-No-Starch-Press-2022Alan-Hohn--_processed (Part 3)

**Starting Chapter:** 2 PROCESS ISOLATION

---

---
#### Service and Traffic Distribution
Kubernetes Services provide an IP address and routing to one or more endpoints. In this case, a Service routes traffic across multiple Pods that are running as endpoints.

:p What is a Kubernetes Service and how does it manage traffic distribution?
??x
A Kubernetes Service provides a stable network identity (IP address) for a set of Pods and manages the routing of traffic to those Pods. It abstracts the underlying Pod IP addresses from clients, allowing you to change the backend Pods without affecting the external access points.

In this specific example, the Service is configured with:
- **IP Address**: 10.43.231.177 (This is a placeholder for the actual Service IP)
- **Port Configuration**: 
  - Service listens on Port 80/TCP
  - Targets are set to Port 5000/TCP
- **Endpoints**:
  - 10.42.0.10:5000
  - 10.42.0.11:5000
  - 10.42.0.14:5000 (with two more unlisted)

This setup indicates that the Service is load-balancing traffic across five Pods.

??x
---

---
#### Kubernetes and Microservices Architecture
Kubernetes enables the deployment of applications as a set of containers, providing scalability and self-healing capabilities through its orchestration features. Modern applications are often designed using microservices architecture to achieve better scalability and reliability by deploying components independently.

:p How does Kubernetes facilitate modern application development?
??x
Kubernetes allows developers to focus on building applications while managing the deployment, scaling, and monitoring of containerized services. It provides a platform for dynamically allocating resources based on demand, ensuring that applications remain highly available even under varying load conditions.

Key features include:
- **Self-healing**: Kubernetes can automatically restart failed containers.
- **Scalability**: Applications can scale up or down based on resource utilization metrics.
- **Deployment and Rollouts**: Simplifies rolling out new versions of applications without downtime.

For example, you might use the following command to deploy a simple application with three replicas:
```bash
kubectl run myapp --image=nginx --replicas=3
```
This would start an Nginx container replicated across three instances managed by Kubernetes.

??x
---

---
#### Process Isolation Using Linux Namespaces
Linux namespaces provide a mechanism for isolating processes and resources from the host environment. This isolation can be used to create a distinct system within the same physical or virtual machine, which is useful in containerization.

:p What are Linux namespaces and how do they enable process isolation?
??x
Linux namespaces allow a process to see its own view of certain system resources. For example:
- **Mount Namespace**: Each namespace has its own mount table, meaning you can have separate file systems within the same machine.
- **PID Namespace**: Processes in different PID namespaces are completely isolated from each other; they do not share any processes.

Creating a simple process with a PID namespace using the following pseudocode:
```java
// Pseudocode for creating a new PID namespace and running a process inside it
public class CreateNamespace {
    public static void main(String[] args) {
        // Create a new namespace
        ProcessBuilder pb = new ProcessBuilder("sh", "-c", "unshare --fork --pid bash -c 'echo $$'");
        try {
            Process p = pb.start();
            int pid = p.waitFor(); // Wait for the process to finish and retrieve its PID
            System.out.println("PID inside namespace: " + pid);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
```
This code creates a new PID namespace and runs a shell command inside it, printing out the PID of the shell process. The PID is isolated from the host's process list.

??x
---

#### Process Isolation Overview
Background context explaining the need for process isolation. Processes share a single computer, but they must be isolated to prevent interference and ensure fair resource usage. This includes managing CPU, memory, storage, and network resources.

:p What is the primary reason for needing process isolation in computing systems?
??x
The primary reason for needing process isolation is to allow multiple programs to run on the same hardware without interfering with each other. This prevents issues such as one program consuming too much resources, overwriting files, extracting secrets, or causing another program to misbehave.

---
#### Filesystem Permissions and Process Isolation
Background context explaining how filesystem permissions control visibility and actions of processes. Linux uses owner and group permissions for read, write, execute capabilities.

:p How do filesystem permissions contribute to process isolation?
??x
Filesystem permissions limit what files a process can access or modify. By setting appropriate ownership and permissions (e.g., -rw-r-----), the system ensures that one process cannot easily interfere with another's data. For example:
```bash
-prompt# ls -l /var/log/auth.log
```
This command shows file permissions where only the owner ('syslog') can write to `/var/log/auth.log`.

---
#### Chroot for Process Isolation
Background context on `chroot` and its role in isolating processes by limiting their view of the filesystem.

:p What is `chroot` used for in process isolation?
??x
`chroot` changes a process's root directory, effectively isolating it from other parts of the file system. This limits what files are accessible to the process, ensuring that sensitive data or critical system components are not exposed. The example given involves setting up:
```bash
mkdir /tmp/newroot
cp --parents /bin/bash /bin/ls /tmp/newroot
# ... copy necessary libraries
chroot /tmp/newroot /bin/bash
```
This command sequence sets up a new root directory for the process, making only the specified files and directories accessible.

---
#### Containers and Process Isolation
Background context on how containers use namespaces to provide isolation. Namespaces create separate views of system resources like processes, users, filesystems, and network interfaces.

:p How do namespaces enable process isolation in container runtimes?
??x
Namespaces allow containers to have their own isolated view of the operating system's resources. This includes:
- **Process Namespace**: Ensures that each container sees only its own processes.
- **User Namespace**: Maps users and groups differently between host and container, isolating identities.

For example, using `lxc` (a simple container runtime), you can create a namespace for processes as follows:
```bash
lxc launch ubuntu:20.04 my-container
```
This command creates an isolated environment where the process inside is not affected by other containers or the host system.

---
#### Directly Creating Namespaces with Linux Commands
Background context on using `unshare` and other Linux commands to create namespaces directly without container runtimes.

:p How can you use `unshare` to create a namespace for processes?
??x
`unshare` is a command that creates a new process namespace. You can run an application within this namespace, ensuring it has its own view of the system:
```bash
sudo unshare -p bash
```
This command starts a new bash shell in a separate process namespace, where any processes spawned will have their own isolated environment.

---
#### Virtual Machines vs Containers for Isolation
Background context comparing VMs and containers based on isolation and overhead.

:p What are the key differences between virtual machines (VMs) and containers regarding isolation?
??x
Virtual machines provide full hardware abstraction, which offers better isolation but comes with higher overhead. Containers share the host kernel and resources, providing lighter-weight isolation with less overhead. The key differences include:
- **VMs**: Run their own operating system, requiring a complete OS installation.
- **Containers**: Use shared kernel, reducing startup time and resource usage.

For example, comparing VM start times vs container startups:
```bash
# VM: Slow due to full OS boot
sudo virt-install --name vm1 --memory 2048 --vcpus=1 --os-type linux

# Container: Fast as it shares the host kernel
sudo podman run -it fedora bash
```
The container startup is much faster because it leverages existing resources.

#### Process Isolation via Containers

Containers provide isolation for processes by separating them from other parts of the system while sharing the kernel and physical hardware. This allows processes to operate as if they were on a separate, isolated machine.

:p What is process isolation in containers?
??x
Process isolation in containers refers to the ability to run multiple applications or services in their own environment with controlled access to resources such as CPU, memory, storage, and network. Each container appears to be running its own operating system instance but shares the kernel of a single host OS for efficiency.
x??

---

#### Kinds of Isolation in Containers

Containers offer several kinds of isolation that work together to ensure processes operate independently:

- Mounted filesystems: Allows files and directories from the host or other containers to be mounted inside a container.
- Hostname and domain name: Each container can have its own hostname and domain name for networking purposes.
- Interprocess communication (IPC): Containers provide isolated IPC mechanisms so that processes within different containers cannot directly communicate with each other.
- Process identifiers (PIDs): Each container has its own PID namespace, making it appear to the processes inside that they are running in a separate system.
- Network devices: Containers have their own network stack and can be configured with specific IP addresses and networks.

:p How many kinds of isolation does a container provide?
??x
A container provides five kinds of isolation:
1. Mounted filesystems
2. Hostname and domain name
3. Interprocess communication (IPC)
4. Process identifiers (PIDs)
5. Network devices
These isolations work together to ensure that processes in different containers appear as if they are running on separate systems.
x??

---

#### Container Platforms and Container Runtimes

Container platforms like Docker provide a higher-level abstraction for managing containerized applications, including storage, networking, and security. Under the hood, Docker uses a container runtime such as containerd to manage processes within containers.

:p What is a container platform?
??x
A container platform is an environment that provides tools and services for developing, deploying, and running containerized applications. It includes functionalities like container storage, networking, and security. For example, Docker is a popular container platform.
x??

---

#### Containerd as the Container Runtime

Containerd is a container runtime used by modern versions of Docker to manage processes in containers at a low level. It provides essential functionality for running and managing containers.

:p What is containerd?
??x
Containerd is a container runtime that manages and runs containers. It works under the hood when using Docker, providing low-level functionalities such as starting, stopping, and restarting containers.
x??

---

#### Installing containerd

To use containerd directly, we need to install it on the system. This involves setting up HTTP/S support for Apt, adding the official Docker package registry, and then installing containerd.

:p How do you install containerd?
??x
To install containerd:

1. Update Apt:
   ```
   root@host01:~# apt update
   ```

2. Install Apt transport for HTTPS:
   ```
   root@host01:~# apt -y install apt-transport-https
   ```

3. Add the Docker package registry and install containerd.io:
   ```shell
   root@host01:~# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   root@host01:~# echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu focal stable" > /etc/apt/sources.list.d/docker.list
   root@host01:~# apt update && apt install -y containerd.io
   ```

4. Verify the installation:
   ```
   root@host01:~# ctr images ls
   REF TYPE DIGEST SIZE PLATFORMS LABELS
   The final command ensures that the package is installed correctly, and the service is running.
   ```
x??

---

#### Using `ctr` Command

The `ctr` command is used to interact with containerd directly. It allows you to list images, inspect containers, manage resources, etc.

:p How do you use the `ctr` command?
??x
The `ctr` command can be used to interact with containerd directly for various operations such as listing images or inspecting containers:

```shell
root@host01:~# ctr images ls
REF TYPE DIGEST SIZE PLATFORMS LABELS
```

This command lists the images available in your container runtime.
x??

---

#### Container Runtimes Overview
Background context: Container runtimes are low-level libraries that form the foundation of containerized applications. They handle running and managing containers but do not provide user-facing tools for direct interaction, as these tasks are typically handled by higher-level orchestration platforms like Docker or Kubernetes.

:p What is a container runtime?
??x
A container runtime is a library responsible for creating and managing containers at a low level. It provides the underlying infrastructure for running containers without providing high-level user interfaces, which are usually managed by tools such as Docker or Kubernetes.
x??

---
#### Pulling Images with Containerd
Background context: To run a container using `containerd`, you need to first pull an image from a registry. Unlike Docker, `containerd` requires specifying the full path of the image including the registry hostname and tag.

:p How do we download an image using `containerd`?
??x
To download an image using `containerd`, you use the `ctr image pull` command followed by the full path to the image. For example, to pull the BusyBox image:
```shell
root@host01:~# ctr image pull docker.io/library/busybox:latest
```
This command retrieves and stores the specified Docker image locally.
x??

---
#### Running a Container with `ctr`
Background context: Once an image is downloaded using `containerd`, you can run a container from that image. The `ctr` tool provides various options for running containers, including managing terminal input/output.

:p How do we run a container using the `ctr` command?
??x
To run a container using the `ctr` command, you use the following syntax:
```shell
root@host01:~# ctr run -t --rm <IMAGE_REF> COMMAND
```
- `-t` creates a TTY for the container.
- `--rm` tells `containerd` to delete the container when the main process stops.
For example, to run BusyBox:
```shell
root@host01:~# ctr run -t --rm docker.io/library/busybox:latest v1
```
This command starts a shell in the container and provides an interactive session.
x??

---
#### Understanding Container Isolation
Background context: Containers provide isolated environments for running applications. This isolation includes separate network stacks and process spaces, ensuring that each container operates independently.

:p What does the `ps -ef` command show inside a container?
??x
The `ps -ef` command inside a container shows information about the processes running within it. For example:
```shell
root@host01:~# ps -ef
PID   USER     TIME  COMMAND
1 root      0:00 sh
8 root      0:00 ps -ef
```
This output indicates that the container has a single process (the shell), and any other commands run inside it will also be listed under this process tree.
x??

---
#### Container Networking in `containerd`
Background context: By default, Docker provides an additional network interface attached to a bridge for containers. This allows communication between containers and access to external networks via Network Address Translation (NAT). However, with lower-level runtimes like `containerd`, you have more control over networking.

:p Why does the container lack a bridge interface?
??x
The container lacks a bridge interface because `containerd` is a low-level container runtime that focuses on managing images and running containers. It does not automatically configure network bridges or handle NAT, which are typically managed by higher-level orchestration platforms like Docker.
x??

---

#### Linux Namespaces Overview
Background context explaining the role of namespaces in containerization. Linux namespaces provide a way to isolate processes from each other and from the host system, ensuring that a process sees only the resources it is supposed to access. Different types of namespaces (mnt, uts, ipc, pid, net) are used for various levels of isolation.

:p What are Linux namespaces and how do they contribute to container isolation?
??x
Linux namespaces are a key feature in the Linux kernel that allow processes to have their own isolated view of system resources such as filesystems, process IDs, network interfaces, etc. This isolation ensures that processes running in different containers cannot interfere with each other or with the host system.
```bash
root@host01:~# lsns | grep 18088
4026532180 mnt         1 18088 root            sh
4026532181 uts         1 18088 root            sh
4026532182 ipc         1 18088 root            sh
4026532183 pid         1 18088 root            sh
4026532185 net         1 18088 root            sh
```
x??

---

#### Types of Namespaces
The text mentions five types of namespaces: mnt, uts, ipc, pid, and net. These are used to isolate various aspects of a process's environment.

:p What are the different types of Linux namespaces mentioned in the text?
??x
The five types of Linux namespaces mentioned in the text are:
- **mnt**: Mount points, isolating filesystems.
- **uts**: Unix time sharing, including hostname and network domain.
- **ipc**: Interprocess communication (e.g., shared memory).
- **pid**: Process identifiers and list of running processes.
- **net**: Network resources (interfaces, routing table, firewall).

These namespaces provide comprehensive isolation for containerized applications.
```bash
root@host01:~# lsns | grep 18088
4026532180 mnt         1 18088 root            sh
4026532181 uts         1 18088 root            sh
4026532182 ipc         1 18088 root            sh
4026532183 pid         1 18088 root            sh
4026532185 net         1 18088 root            sh
```
x??

---

#### Using `ps` to Verify PID
The text describes how to use the `ps` command along with `grep` to verify the process ID (PID) of a container's shell process.

:p How can you use `ps` and `grep` commands together to find the correct PID for a running container?
??x
To verify the process ID (PID) of a container's shell process, you can use the following steps:
1. List running containers using `ctr task ls`.
2. Use `ps -ef | grep <PID> | grep -v grep` to identify the specific process.

Here’s an example command sequence:

```bash
root@host01:~# ctr task ls
TASK    PID      STATUS     v1      18088    RUNNING

root@host01:~# ps -ef | grep 18088 | grep -v grep
root       18088   18067  0 18:46 pts/0    00:00:00 sh
```

This command sequence helps to confirm the PID of the container’s shell process.
```bash
root@host01:~# ps -ef | grep 18067 | grep -v grep
root       18067       1  0 18:46 ?        00:00:00 /usr/bin/containerd-shim-runc-v2 -namespace default -id v1 -address /run/containerd/containerd.sock
```
x??

---

#### Using `lsns` to List Namespaces
The text explains how to use the `lsns` command to list the namespaces associated with a container.

:p How can you use the `lsns` command to find out which namespaces are used by a specific container?
??x
To list the namespaces that are being used by a specific container, you can run the following command:

```bash
root@host01:~# lsns | grep 18088
4026532180 mnt         1 18088 root            sh
4026532181 uts         1 18088 root            sh
4026532182 ipc         1 18088 root            sh
4026532183 pid         1 18088 root            sh
4026532185 net         1 18088 root            sh
```

This command lists the namespaces and their types associated with the specified PID, indicating that `containerd` uses these namespaces to fully isolate the container.
x??

---

#### Exiting a Container
The text describes how to exit from within a running container.

:p How do you exit from within a running container?
??x
To exit from within a running container, simply run the `exit` command:

```bash
/ # exit
```

This command will close the shell session inside the container and return you to the host system's terminal.
x??

---

#### CRI-O and Container Runtimes Overview
Background context explaining the role of container runtimes like CRI-O. It discusses how CRI-O is used by tools such as Podman, Buildah, and Skopeo on Red Hat 8 systems.

:p What are some key tools that use CRI-O for managing containers?
??x
CRI-O is primarily used with tools like Podman, Buildah, and Skopeo to manage containerized applications. These tools facilitate the deployment, management, and execution of containers within a system.
x??

---
#### Setting Up Repositories for CRI-O Installation
Background context on how repositories are set up for CRI-O installation using `apt` on Debian-based systems.

:p How do you configure repositories for installing CRI-O?
??x
To configure repositories for CRI-O, you need to add files to `/etc/apt/sources.list.d/`. The following commands illustrate the process:

```sh
root@host01:~# echo "deb $REPO/$ OS/ /" > /etc/apt/sources.list.d/kubic.list
root@host01:~# echo "deb $REPO:/cri-o:/$ VERSION/$ OS/ /" \
    > /etc/apt/sources.list.d/kubic.cri-o.list
```
These commands add the necessary repository URLs for CRI-O, which are specific to the version and distribution.

x??

---
#### Installing CRI-O Using `apt`
Background on using package managers like `apt` to install software packages.

:p How do you install CRI-O and its dependencies?
??x
You can install CRI-O and its dependencies by running:

```sh
root@host01:~# apt update && apt install -y cri-o cri-o-runc
```
This command updates the package index and installs CRI-O along with `cri-o-runc`, a required runtime for CRI-O.

x??

---
#### Starting CRI-O Service
Background on managing services using systemd.

:p How do you start and enable the CRI-O service?
??x
To ensure that CRI-O starts at boot and is running, you can use:

```sh
root@host01:~# systemctl enable crio && systemctl start crio
```
These commands enable the `crio` service to start automatically on system boot and initiate it immediately.

x??

---
#### Installing crictl for Testing CRI-O
Background context explaining why `crictl` is necessary for testing container runtimes.

:p Why do you need `crictl` when using CRI-O?
??x
`crictl` is a command-line tool designed to interact with any container runtime that supports the Container Runtime Interface (CRI). It is essential because, unlike `containerd`, CRI-O does not ship with its own command-line tools. Therefore, installing `crictl` allows for testing and managing containers using CRI-O.

x??

---
#### Configuring crictl to Connect to CRI-O
Background on configuring `crictl` to connect to the CRI-O runtime.

:p How do you configure `crictl` to connect to CRI-O?
??x
To configure `crictl`, you need to edit or create a configuration file (`/etc/crictl.yaml`) with the following content:

```yaml
runtime-endpoint: unix:///var/run/crio/crio.sock
image-endpoint: unix:///var/run/crio/crio.sock
```
This configuration specifies that `crictl` should connect to CRI-O’s Unix socket for both runtime and image operations.

x??

---
#### Defining a Pod with crictl
Background on creating Pods using Kubernetes-like syntax.

:p How do you define a Pod using `crictl`?
??x
To define a Pod, you can use the following YAML file (`pod.yaml`):

```yaml
---
metadata:
  name: busybox
  namespace: crio
linux:
  security_context:
    namespace_options:
      network: 2
```
This configuration defines a simple Pod with a single container named `busybox`.

x??

---

#### CRI-O Container Network Configuration
CRI-O uses a Container Network Interface (CNI) plugin to manage network namespaces for containers. However, if you set `network: 2` in your configuration, CRI-O will use the host's network namespace instead of creating a separate one.
:p How does setting `network: 2` affect container networking in CRI-O?
??x
Setting `network: 2` tells CRI-O to use the host's network namespace for the container. This means that the container will share the same network interfaces and configurations as the host, without creating a separate network namespace.
```yaml
# Example of pod.yaml with network set to 2
metadata:
  name: my-pod
spec:
  runtime_config: "*"
  pod_runtime: "crio"
  containers:
    - metadata:
        name: busybox
      image: docker.io/library/busybox:latest
      args:
        - "/bin/sleep"
        - "36000"
      network: 2
```
x??

---
#### Pulling Docker Images for CRI-O Containers
Before running a container, you need to ensure that the required Docker image is pulled and available locally. This can be done using `crictl pull`.
:p How do you pull a Docker image into your system before creating a container with CRI-O?
??x
To pull a Docker image, use the `crictl pull` command followed by the full image name including the repository and tag.
```sh
# Example of pulling busybox:latest
root@host01:~# crictl pull docker.io/library/busybox:latest
Image is up to date for docker.io/library/busybox@sha256:...
```
x??

---
#### Creating and Starting a CRI-O Container
After ensuring the image is pulled, you can create and start a container using `crictl`. This involves defining the pod and container configurations.
:p How do you create and start a CRI-O container for BusyBox?
??x
To create and start a CRI-O container for BusyBox:
1. Pull the image.
2. Run the pod with `crictl runp`.
3. Create the container within that pod using `crictl create`.
4. Start the container with `crictl start`.

Example commands:
```sh
# Step 1: Pull the image
root@host01:~# crictl pull docker.io/library/busybox:latest

# Step 2: Create and run the pod
root@host01:~# POD_ID=$(crictl runp pod.yaml)

# Step 3: Create the container
root@host01:~# CONTAINER_ID=$(crictl create$ POD_ID container.yaml pod.yaml)

# Step 4: Start the container
root@host01:~# crictl start $CONTAINER_ID
```
x??

---
#### Inspecting and Executing Commands in CRI-O Containers
After creating a container, you can inspect its state and execute commands within it.
:p How do you inspect a CRI-O container and run an exec command?
??x
To inspect the state of a CRI-O container:
```sh
root@host01:~# crictl inspect <CONTAINER_ID>
```

To execute a command inside the container, use `crictl exec` with `-ti` to attach a terminal.
```sh
root@host01:~# crictl exec -ti <CONTAINER_ID> /bin/sh

# Example of commands executed within the container
/container # ip a
...
/container # ps -ef
```
x??

---
#### Understanding CRI-O Process Isolation with Linux Namespaces
CRI-O uses Linux namespaces for process isolation, even when network namespace is shared. This can be seen by examining the processes inside the container.
:p How do you verify that CRI-O containers are using Linux namespaces?
??x
To verify the use of Linux namespaces in a CRI-O container, inspect the running processes and check their PID:
```sh
root@host01:~# PID=$(crictl inspect$ CONTAINER_ID | jq '.info.pid')
root@host01:/opt# ps -ef | grep $PID | grep -v grep
```

This command shows that the container's process is isolated, but shares network with the host.
```sh
root       23894       1  0 20:15 ?        00:00:00 /usr/bin/conmon ...
root       23906   23894  0 20:15 ?        00:00:00 /bin/sleep 36000
```
x??

---

