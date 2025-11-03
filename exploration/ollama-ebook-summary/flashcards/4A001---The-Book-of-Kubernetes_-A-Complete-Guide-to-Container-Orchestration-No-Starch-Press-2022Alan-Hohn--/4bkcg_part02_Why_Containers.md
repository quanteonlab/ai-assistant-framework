# Flashcards: 4A001---The-Book-of-Kubernetes_-A-Complete-Guide-to-Container-Orchestration-No-Starch-Press-2022Alan-Hohn--_processed (Part 2)

**Starting Chapter:** Why Containers

---

---
#### Resilience and Microservices
In a scenario where an application can handle hardware or software failures without affecting end-users, it is considered resilient. However, resilience from one instance to another might not be enough if only separate instances are running.

:p What does resilience mean in the context of microservices?
??x
Resilience means that the application should be able to handle a failure (either hardware or software) such that end-users remain unaware. If multiple unrelated instances are running, the failure of one instance might not be hidden from users.
x??

---
#### Modern Application Architecture
Modern application architecture using microservices is appealing due to its flexibility and scalability. However, building such applications involves significant trade-offs related to complexity in managing individual services.

:p Why does modern application architecture with microservices sound appealing?
??x
Modern application architecture with microservices offers the advantage of independent deployment and scaling capabilities for each service, enhancing overall system resilience and allowing teams to work independently on different components. However, this comes at a cost: it increases the complexity in packaging, deploying, configuring, and maintaining multiple services.
x??

---
#### Trade-offs in Microservices Architecture
Engineering trade-offs are crucial when designing modern applications with microservices. While microservices offer independence and flexibility, they introduce complex problems such as deployment, configuration, and dependency management.

:p What are some significant trade-offs when using microservices?
??x
When using microservices, the complexity arises from managing numerous small pieces. This includes packaging, deploying, configuring, and maintaining each service independently. Additionally, ensuring multiple instances of a service can communicate and scale efficiently while handling failures introduces substantial challenges.
x??

---
#### Containers in Microservices Architecture
Containers are used to manage the complexities introduced by microservices architecture. They help with isolation, versioning, fast startup, and low overhead, making it easier to deploy and maintain microservices.

:p Why do we need containers for microservices?
??x
Containers provide a solution to the challenges of managing individual microservices. They ensure that each service is isolated from others, can be easily packaged and deployed, and started quickly with minimal resource overhead. Containers also help manage dependencies and facilitate updates.
x??

---
#### Requirements for Containers
To address the needs of deploying microservices, containers must bundle applications, uniquely identify versions, isolate services, start quickly, and minimize resource usage.

:p What are the requirements for a single microservice when using containers?
??x
The requirements include:
- **Packaging**: Bundling the application with dependencies.
- **Versioning**: Identifying unique versions for updates.
- **Isolation**: Ensuring services do not interfere with each other.
- **Fast startup**: Quickly starting new instances.
- **Low overhead**: Minimizing resource usage.

Containers meet these requirements by providing isolation, low overhead, and fast startup. Each container runs from a container image that includes the application and its dependencies.
x??

---
#### Orchestration for Multiple Microservices
Orchestration is necessary to manage multiple microservices working together. It involves clustering services across servers to ensure processing, memory, and storage are effectively utilized.

:p What does orchestration involve in managing multiple microservices?
??x
Orchestration involves providing a way to cluster containers (running microservices) across multiple servers to distribute processing, memory, and storage resources efficiently. This ensures that the system can handle failures and scale appropriately.
x??

---

#### Discovery: How Microservices Find Each Other
Background context explaining how microservices discover and communicate with each other. In a dynamic environment, containers might move around, so a discovery mechanism is needed to ensure that services can find one another reliably.

:p What is the purpose of a discovery service in a microservice architecture?
??x
A discovery service helps microservices locate and communicate with each other by maintaining up-to-date information about where instances of various services are running. This is crucial because containers might be deployed or moved to different servers dynamically.
x??

---
#### Configuration: Decoupling Configurations from Code
Background context explaining the importance of decoupling configurations from code in microservices, allowing for easier reconfiguration and deployment without changing the application logic.

:p Why is configuration separation important?
??x
Configuration separation allows developers to change runtime settings without modifying the source code. This makes it easier to maintain and update applications by keeping configuration details separate from the business logic.
x??

---
#### Access Control: Managing Container Authorization
Background context explaining how access control ensures only authorized containers are allowed to run, maintaining security within a microservice architecture.

:p What is the role of access control in container orchestration?
??x
Access control manages authorization for creating and running containers. It ensures that only properly authorized containers can be executed, helping prevent unauthorized or malicious activities.
x??

---
#### Load Balancing: Distributing Requests Among Containers
Background context explaining how load balancing distributes incoming requests across multiple instances of a service to avoid overloading any single instance.

:p What is the purpose of load balancing in microservices?
??x
Load balancing aims to distribute incoming requests evenly among all available instances of a microservice. This not only helps in managing traffic but also ensures that no single container bears an excessive workload, leading to better performance and reliability.
x??

---
#### Monitoring: Detecting Failed Microservice Instances
Background context explaining the importance of monitoring for identifying failed or unhealthy microservices to ensure load balancing works effectively.

:p Why is monitoring crucial for a healthy microservice architecture?
??x
Monitoring is essential for detecting failures in microservices. Without proper monitoring, traffic might be directed to failing instances, which can degrade the overall performance and availability of the application.
x??

---
#### Resilience: Automatic Recovery from Failures
Background context explaining how resilience mechanisms help in automatically recovering from failures within a microservice architecture.

:p What is the purpose of resilience in container orchestration?
??x
Resilience ensures that the system can automatically recover from failures. This prevents cascading effects where a single failure can bring down an entire application, making the service more robust and reliable.
x??

---
#### Container Orchestration: Running Containers Dynamically
Background context explaining how container orchestration environments like Kubernetes manage containers across multiple servers.

:p What is the role of container orchestration in managing microservices?
??x
Container orchestration tools like Kubernetes allow treating multiple servers as a single set of resources to run containers. They dynamically allocate containers based on availability, provide distributed communication and storage, and help manage the overall lifecycle of containers.
x??

---
#### Running Containers with Docker: Basic Commands
Background context explaining how to use basic Docker commands to run containers, illustrating the concept of containerized applications.

:p How do you create and run a container using Docker?
??x
To create and run a container in Docker, use the `docker run` command followed by the name or path of the image. For example:
```bash
docker run <image_name>
```
This command starts a new container based on the specified image.
x??

---

---

#### Docker Container Setup and Execution
Background context: The text describes setting up a Rocky Linux container using Docker. It explains how to download, start, and interact with the container, as well as listing key differences between the host system and the container environment.

:p What command was used to download and start the Rocky Linux container?
??x
The `docker run -ti rockylinux:8` command was used to download and start a Rocky Linux 8 container with an interactive terminal.
x??

---

#### Container Environment vs. Host System Differences
Background context: The text highlights several differences between the container environment and the host system, such as hostname, filesystem contents, package manager, process list, network devices, and kernel version.

:p What is the difference in hostname when running commands inside a container compared to the host?
??x
In a container, the hostname typically starts with `18f20e2d7e49` (though it will vary), while on the host system, the hostname would be something else like `host01`.
x??

---

#### Network Interface in Container
Background context: The text describes how Docker allocates a virtual network interface to containers. Each container gets its own IP address and MAC address within the Docker network.

:p What is the network interface description for the container's loopback device?
??x
The loopback device in the container has the following description:
```
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 ... link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
```
x??

---

#### Running Processes in a Container
Background context: The text lists the running processes when inside a container. Typically, there are very few processes running due to the minimal nature of containers.

:p What is the process ID (PID) and command for the default shell session when inside the container?
??x
The default shell session has PID 1 and runs `/bin/bash` as the command.
```bash
UID          PID    PPID  C STIME TTY          TIME CMD
root         1       0  0 13:30 pts/0    00:00:00 /bin/bash
```
x??

---

#### Package Manager in Container
Background context: The text mentions the use of `yum` as the package manager inside the container, which is different from the host system's package manager.

:p Which package manager was used to install packages within the container?
??x
The package manager used to install packages within the container was `yum`.
x??

---

#### Network Device Descriptions in Container
Background context: The text details how Docker assigns virtual network interfaces to containers, including their MAC addresses and IP addresses.

:p What is the description of the `eth0` network interface inside the container?
??x
The `eth0` network interface inside the container has the following description:
```
18: eth0@if19: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 ... link/ether 02:42:ac:11:00:02 brd ff:ff:ff:ff:ff:ff link-netnsid 0
```
x??

---

#### Hostname and Kernel Version in Container
Background context: The text explains that despite running a Rocky Linux container, the `uname -v` command still shows an Ubuntu kernel version.

:p What command was used to check the kernel version inside the container?
??x
The command used to check the kernel version inside the container is `uname -v`.
x??

---

#### Conclusion: Container vs. Host System Differences
Background context: The text summarizes various differences between a container environment and the host system, such as hostname, package manager, running processes, network interfaces, and kernel version.

:p What are some key differences observed when comparing the container to the host system?
??x
Key differences include:
- Different hostname (e.g., `18f20e2d7e49` in the container vs. `host01` on the host)
- Use of different package managers (`yum` in the container vs. not specified for the host)
- Limited number of running processes (usually just `/bin/bash`)
- Virtual network interfaces and IP addresses specific to the container
- The same Ubuntu kernel version reported by `uname -v`
x??

---

#### Container Image and Volume Mounts
Docker containers are often misunderstood as virtual machines due to their isolated nature. When we pull a container image, it is similar to downloading an OS image for a virtual machine. We can run a container from this image with specific configurations like volume mounts and environment variables.

:p How does pulling and running a Docker container differ from traditional VM operations?
??x
When we "pull" a Docker container, it resembles downloading an OS image for a virtual machine. However, when we "run" the container, we can mount host volumes into the container to share files between the host and guest environments. Additionally, we can set environment variables that are accessible within the container.

```bash
# Pulling the Alpine Linux container image
root@host01:~# docker pull alpine:3

# Running a container with volume mounts and an environment variable
root@host01:~# docker run -ti -v /:/host -e hello=world alpine:3
```
x??

---

#### Container Processes and Environment Variables
Containers appear to have their own operating system, including processes, filesystems, and networking. However, unlike virtual machines, containers share the host's kernel, meaning they cannot install separate kernel modules or device drivers.

:p Can a container run an SSH server by default?
??x
No, by default, a container does not include an SSH server because it runs in a lightweight environment that shares the host’s kernel. Containers typically do not have system services running unless explicitly started within the container.

```bash
# Running an Alpine Linux container with an environment variable set
root@host01:~# docker run -ti -v /:/host -e hello=world alpine:3

# Accessing the environment variable inside the container
/ # echo $hello
world
```
x??

---

#### Docker Daemon and Port Forwarding
Docker allows us to manage containers in a way that blends concepts from both virtual machines and regular processes. The `-d` flag starts a container in daemon mode, meaning it runs in the background like a regular process.

:p How does port forwarding work with Docker?
??x
Port forwarding allows traffic from one network interface to be forwarded to another within the same host or between different hosts. In this context, Docker maps a host port to a container port using the `-p` flag.

```bash
# Running NGINX in a detached (background) mode and forwarding port 8080 to 80
root@host01:~# docker run -d -p 8080:80 nginx

# Checking running containers
root@host01:~# docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                    NAMES
e9c5e8702037        nginx               "nginx"             2 seconds ago       Up 1 second          0.0.0.0:8080->80/tcp      funny_montalcini

# Connecting to the running service using curl
root@host01:~# curl http://localhost:8080/
```
x??

---

#### Containers vs Virtual Machines
Containers provide an isolated environment that resembles a separate system but do not have their own kernel. They are lightweight and share resources with the host, such as the file system and network stack.

:p Why can't we SSH into a container by default?
??x
By default, containers do not include services like SSH because they run in a lightweight environment shared with the host's kernel. To enable SSH or other system services within a container, they must be explicitly started inside the container using appropriate Docker commands and configurations.

```bash
# Running an Alpine Linux container without an SSH server by default
root@host01:~# docker run -ti alpine:3
/ # echo $hello  # This will not work as the environment variable is set in the parent command

# Starting a service like SSH within the container would require additional steps, e.g.,
root@host01:~# docker run -ti --rm -v /etc/ssh:/etc/ssh alpine sh -c 'apk add openssh && ssh-keygen -A'
```
x??

---

#### Running NGINX in a Container
Background context: The text explains how to run an NGINX server within a container using a single command. This approach leverages the benefits of containers without the overhead of virtual machines, ensuring that NGINX runs as a regular process and does not conflict with other applications installed on the host system.
:p How can you verify that NGINX is running in a container?
??x
To check if NGINX is running in a container, use the `ps` command to list processes:
```sh
root@host01:~# ps -ef | grep nginx | grep -v grep
```
This command filters out the `grep` process and checks for any instances of `nginx`. If NGINX is running as expected, you should see output similar to this:
```
root     35729 35703 0 14:17 ?        00:00:00 nginx: master process /usr/sbin/nginx -g daemon on;master_process on;
systemd+ 35796 35729 0 14:17 ?        00:00:00 nginx: worker process
```
x??

---
#### Kubernetes and Application Deployment
Background context: The text introduces the concept of deploying applications using Kubernetes, a container orchestration framework that provides load balancing, resilience, and automated scaling. It discusses setting up a single-node K3s cluster and deploying an example "to-do" application with both Node.js and PostgreSQL components.
:p What command is used to check the status of nodes in a Kubernetes cluster?
??x
To check the status of nodes in a Kubernetes cluster, use the `kubectl` command with the `get nodes` option:
```sh
root@host01:~# k3s kubectl get nodes
```
This will provide information about each node, such as its name (`NAME`), status (`STATUS`), roles (`ROLES`), age (`AGE`), and version (`VERSION`). The output might look like this:
```
NAME     STATUS   ROLES             AGE   VERSION
host01   Ready    control-plane...  2d    v1...
```
x??

---
#### Understanding Pods in Kubernetes
Background context: In Kubernetes, a Pod is the smallest deployable unit and can contain one or more containers. The text provides an example of deploying multiple Node.js instances alongside a PostgreSQL database.
:p What command lists all running pods in a Kubernetes cluster?
??x
To list all running pods in a Kubernetes cluster, use the `kubectl` command with the `get pods` option:
```sh
root@host01:~# k3s kubectl get pods
```
This will output information about each pod, including its name (`NAME`), readiness status (`READY`), current state (`STATUS`), restart count (`RESTARTS`), and age (`AGE`). For example:
```
NAME                       READY   STATUS    RESTARTS   AGE
todo-db-7df8b44d65-744mt   1/1     Running   0          2d
todo-655ff549f8-l4dxt      1/1     Running   0          2d
todo-655ff549f8-gc7b6      1/1     Running   1          2d
todo-655ff549f8-qq8ff      1/1     Running   1          2d
```
x??

---
#### Kubernetes Scaling and Recovery Mechanism
Background context: The text highlights how to scale applications in a Kubernetes cluster, emphasizing the automated recovery mechanism when containers are destroyed.
:p How can you manually increase the number of pods running for a specific deployment?
??x
To manually increase the number of pods running for a specific deployment in Kubernetes, use the `scale` command with the `--replicas` option:
```sh
root@host01:~# k3s kubectl scale --replicas=5 deployment todo
```
After scaling, you can verify that new pods have been created by listing all pods again using the following command:
```sh
root@host01:~# k3s kubectl get pods
```
This will show an updated list with more pods running for the `todo` deployment.
x??

---
#### Kubernetes Service and Load Balancing
Background context: The text explains how services in Kubernetes route traffic to different pods, ensuring load balancing across instances. It mentions that this functionality is crucial when accessing applications deployed on Kubernetes.
:p How can you describe the service associated with a running application?
??x
To describe the service associated with a running application, use the `kubectl` command with the `describe service` option:
```sh
root@host01:~# k3s kubectl describe service todo
```
This will provide detailed information about the service, including its type, IP address, endpoints, and more. This is essential for understanding how traffic is routed to different pods.
x??

---

