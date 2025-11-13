# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 23)

**Starting Chapter:** Unreliable Networks

---

#### Fault Handling in Distributed Systems
Background context: The reliability of a distributed system is often less than its individual components due to the possibility of faults. These faults can range from network interruptions, component failures, or software bugs. It's crucial for developers and operators to design systems that can handle such faults gracefully.

:p How should you approach handling faults in a distributed system?
??x
Faults must be considered part of the normal operation of a system, not an exception. Developers need to anticipate potential issues and design fault-tolerant mechanisms into their software. Testing environments should simulate these faults to ensure the system behaves as expected.
x??

---

#### Building Reliable Systems from Unreliable Components
Background context: A reliable system can be constructed by layering protocols or algorithms that handle failures at a higher level, even if underlying components are unreliable. Examples include error-correcting codes and TCP on top of IP.

:p How does building a more reliable system work when starting with less reliable components?
??x
By adding layers of protocols or software mechanisms that handle the unreliability of lower levels. For instance, TCP handles packet loss and reordering by ensuring packets are retransmitted if they're lost. The higher-level system can mask some low-level faults, making it easier to reason about failures.

```java
// Pseudocode for a simple TCP-like mechanism
class ReliableTransport {
    void sendRequest(Request req) {
        // Send the request over an unreliable channel
        sendUnreliableChannel(req);
        
        // Wait for response or timeout
        Response resp = waitForResponse(req.id, timeout);
        
        // Process the received response
        processResponse(resp);
    }
    
    void sendUnreliableChannel(Request req) {
        // Code to send over an unreliable channel
    }
    
    Response waitForResponse(long id, long timeout) {
        // Wait for a response within the timeout period
    }
    
    void processResponse(Response resp) {
        // Handle received response
    }
}
```
x??

---

#### Unreliable Networks in Distributed Systems
Background context: In distributed systems, networks are often asynchronous packet networks where messages may be lost, delayed, duplicated, or out of order. These characteristics introduce challenges for reliable communication.

:p What are the common issues with unreliable networks?
??x
Common issues include:
1. Request loss due to network failures.
2. Queued requests due to network congestion.
3. Node failure (crash or power down).
4. Temporary unavailability of nodes due to resource-intensive operations like garbage collection.
5. Lost responses on the network.
6. Delayed responses due to network overload.

```java
// Pseudocode for handling request and response in an unreliable network
class UnreliableNetwork {
    void sendRequest(Request req) throws NetworkException {
        // Send the request, which may be lost or delayed
        if (randomEvent()) { throw new NetworkException("Request lost"); }
        
        // Handle potential retransmissions
        while (!receivedResponse(req.id)) {
            try {
                Thread.sleep(randomDelay());
            } catch (InterruptedException e) {}
        }
    }
    
    boolean receivedResponse(long id) {
        // Check if a response has been received for the request
    }
}
```
x??

---

#### Process Pauses in Distributed Systems
Background context: In distributed systems, nodes can experience pauses due to various reasons such as garbage collection. These pauses can affect the responsiveness of requests.

:p How do process pauses impact distributed systems?
??x
Process pauses can significantly affect the behavior of distributed systems. For example, during a long garbage collection pause, a node may be unresponsive for an extended period. This can cause delays in processing and handling requests from other nodes, potentially leading to timeouts or failures.

```java
// Pseudocode for handling process pauses
class Node {
    void handleRequest(Request req) throws ProcessPauseException {
        try {
            // Simulate a long garbage collection pause
            Thread.sleep(randomLongTime());
            
            // Process the request normally after the pause
            process(req);
        } catch (InterruptedException e) {}
        
        if (processSucceeded()) {
            return Response.SUCCESS;
        } else {
            throw new ProcessPauseException("Process paused during request handling");
        }
    }
    
    boolean processSucceeded() {
        // Logic to determine if processing was successful
    }
}
```
x??

---

#### Network Faults and Reliability Challenges
Network faults can be surprisingly common, even in controlled environments like data centers. Studies have shown that network issues are frequent, with medium-sized data centers experiencing about 12 network faults per month on average. Components such as switches and load balancers fail at high rates. Redundant networking gear does not fully mitigate these failures due to human errors.

:p What is the frequency of network faults in a medium-sized data center?
??x
On average, a medium-sized data center experiences about 12 network faults per month.
x??

---

#### Network Faults and Human Errors
Network issues are often caused by human errors, such as misconfigurations. These mistakes can lead to outages even in well-managed environments.

:p What is a common cause of network failures?
??x
Human errors, particularly misconfigurations, are a major cause of network outages.
x??

---

#### Network Faults and Public Cloud Services
Public cloud services like EC2 are known for frequent transient network glitches. Even private data centers can experience issues, such as software upgrades causing topology reconfigurations.

:p How common are network glitches in public cloud services?
??x
Network glitches are frequent in public cloud services like EC2.
x??

---

#### Network Partitions and Faults
A network partition occurs when one part of the network is cut off from the rest due to a fault. This can lead to deadlocks or data deletion if not handled properly.

:p What is a network partition?
??x
A network partition happens when part of the network is isolated from the rest, often due to a fault.
x??

---

#### Handling Network Faults in Software
Software must be designed to handle network faults because failures can occur. This includes defining and testing error handling mechanisms to prevent deadlocks or data loss.

:p Why do software systems need to handle network faults?
??x
Software needs to handle network faults to ensure reliable operation, as network failures can lead to serious issues like deadlocks or data deletion.
x??

---

#### Network Interface Failures
Network interfaces can sometimes fail in unexpected ways. For example, an interface might drop all inbound packets while still sending outbound packets successfully.

:p How can a network interface behave unexpectedly?
??x
A network interface can behave unpredictably by dropping all inbound packets but still sending outbound packets successfully.
x??

---

#### Concept of Network Faults in Practice
Network faults are common and can occur even in well-managed environments. Redundant networking gear does not fully mitigate these issues due to human errors.

:p What factors contribute to network faults?
??x
Network faults are influenced by factors such as human errors, hardware failures, software upgrades, and external events like undersea cable damage.
x??

---

#### Handling Network Faults

Network faults can cause unexpected behavior in software, even if network reliability is generally good. It might be sufficient to show an error message and hope for a quick resolution rather than attempting complex fault tolerance.

:p How should you handle network faults when your network is normally reliable?
??x
When the network is typically reliable, showing an error message to users while the issue is resolved can be a valid approach. However, ensure that the system can recover from these issues and test its response by deliberately triggering faults (e.g., Chaos Monkey).

```java
public void handleNetworkFault() {
    try {
        // Attempt network operation
    } catch (IOException e) {
        System.err.println("An error occurred: " + e.getMessage());
        // Provide feedback to users or log the issue for further investigation.
    }
}
```
x??

---

#### Detecting Faulty Nodes

Automatic detection of faulty nodes is crucial in distributed systems. This involves mechanisms like load balancers stopping requests from dead nodes and leaders failing over to followers.

:p How can a load balancer detect that a node is dead?
??x
A load balancer can detect a dead node by ensuring the process on the node is listening on the correct port. If not, it considers the node unresponsive.

```java
public boolean checkNodeAlive(String ipAddress, int port) {
    try (Socket socket = new Socket(ipAddress, port)) {
        // If connection succeeds, the node is likely alive.
        return true;
    } catch (IOException e) {
        // Node not responding or process crashed.
        return false;
    }
}
```
x??

---

#### Detecting Faults in Distributed Systems

Fault detection can be challenging due to network unreliability. Specific cases include unreachable nodes, failed processes, and hardware-level failures.

:p How does the operating system help detect a node that has crashed?
??x
The operating system helps by closing or refusing TCP connections if no process is listening on the destination port after receiving a request.

```java
public boolean checkProcessAlive(String ipAddress, int port) {
    try (Socket socket = new Socket(ipAddress, port)) {
        // If connection fails with RST or FIN, the node is likely dead.
        return false;
    } catch (IOException e) {
        // Node might still be running but handling request.
        return true;
    }
}
```
x??

---

#### Detecting Hardware-Level Failures

Hardware-level failures can be detected by querying network switches. This requires access to management interfaces, which may not always be available.

:p How can you detect hardware-level link failures?
??x
Querying network switches for link failure status is one method. However, this option is unavailable if you are using the internet or a shared datacenter without switch access.

```java
public boolean checkHardwareFailure(SwitchManager switchManager) {
    try {
        return switchManager.isLinkDown("remoteIP");
    } catch (NoAccessException e) {
        // Unable to query due to network issues.
        return false;
    }
}
```
x??

---

#### Handling Node Outages

In distributed databases, promoting a follower to leader status after the current leader fails is essential for maintaining system availability.

:p What happens when a leader node in a distributed database fails?
??x
When a leader node fails, one of its followers must be promoted to become the new leader. This ensures continuous operation and data integrity.

```java
public void promoteFollowerToLeader(Follower follower) {
    // Logic for promoting follower.
    System.out.println(follower.getName() + " is now the new leader.");
}
```
x??

---

#### Ensuring Positive Response for Requests

In distributed systems, waiting for a response is necessary to confirm successful request handling. Simply receiving a TCP acknowledgment or an error message may not be sufficient.

:p How can you ensure that a network request was successfully processed?
??x
You need a positive response from the application itself to guarantee that a request was handled correctly. This involves retrying requests, waiting for timeouts, and considering nodes dead if no response is received within the timeout period.

```java
public boolean waitForRequestCompletion(String request) {
    int retries = 3;
    while (retries > 0) {
        try {
            // Send request.
            return true; // Assume success based on application response.
        } catch (IOException e) {
            Thread.sleep(1000); // Wait before retrying.
            retries--;
        }
    }
    return false; // Timeout and consider node dead.
}
```
x??

---

#### Timeout and Unbounded Delays
Background context: When using timeouts to detect faults, there's a trade-off between the risk of declaring a node dead too early (prematurely) versus waiting too long before detecting a fault. A short timeout can quickly identify issues but may incorrectly declare nodes as dead due to temporary slowdowns or load spikes. Conversely, longer timeouts reduce the risk of false positives but increase wait times and potential user frustration.

A formula is not directly applicable here, but we can discuss the impact on system performance:
- Long timeouts mean users might experience delays.
- Short timeouts decrease detection time but can cause unnecessary node failures.

:p What are the trade-offs between using short and long timeouts in detecting faults?
??x
Short timeouts reduce the risk of false negatives by quickly identifying dead nodes, which is crucial for maintaining system reliability. However, they carry a higher risk of declaring nodes dead due to temporary issues like load spikes or network congestion, potentially causing double actions (e.g., sending emails twice) and increasing the overall load on the system.

On the other hand, long timeouts ensure that only genuine failures are detected but at the cost of increased wait times for users and potential error messages. This can be particularly problematic in high-load systems where nodes might just be temporarily slow rather than dead.
x??

---

#### Unreliable Networks
Background context: In many real-world systems, networks do not provide guarantees on maximum delays or server response times. Asynchronous networks may experience unbounded delays due to network congestion and queueing.

Formula: $\text{Timeout} = 2d + r $-$ d$: Maximum delay for packets.
- $r$: Time taken by a non-failed node to handle a request.

However, in practice, both these guarantees are rarely available. Therefore, the system must account for potential spikes in round-trip times which can throw off the timing calculations.

:p How would you determine an appropriate timeout value in systems with unreliable networks?
??x
Given that most real-world networks do not provide guaranteed maximum delays or response times, it is challenging to set a fixed timeout. Instead, you need to consider the worst-case scenario where round-trip times can spike due to network congestion and server load.

A practical approach would be to monitor historical data on average delays and add a buffer time that accounts for potential spikes. For example:
```java
public int calculateTimeout(int avgDelay, int bufferTime) {
    return 2 * avgDelay + bufferTime;
}
```
This method dynamically adjusts the timeout based on observed network behavior, ensuring it remains robust against transient issues without being overly conservative.

x??

---

#### Network Congestion and Queueing
Background context: Network congestion occurs when multiple nodes try to send packets to the same destination simultaneously. The switch must queue these packets before sending them one by one. If the queue fills up, packets are dropped and need to be resent, even if the network is functioning fine.

:p How does network congestion affect packet delivery?
??x
Network congestion affects packet delivery by causing delays and potential packet loss. When a node sends multiple packets to a congested switch or link, the switch queues these packets. If the incoming data rate exceeds the outgoing capacity of the switch, the queue can fill up, leading to packet drops.

To handle this in pseudocode:
```java
public class NetworkSwitch {
    private Queue<Packet> queue = new LinkedList<>();

    public void send(Packet packet) {
        if (queue.size() >= MAX_QUEUE_SIZE) {
            // Drop packet and log error
            System.out.println("Queue full, dropping packet: " + packet);
            return;
        }
        queue.add(packet);
        processQueue();
    }

    private void processQueue() {
        while (!queue.isEmpty()) {
            Packet packet = queue.remove();
            sendPacketToDestination(packet);
        }
    }

    private void sendPacketToDestination(Packet packet) {
        // Simulate sending packet to destination
        System.out.println("Sending packet: " + packet);
    }
}
```
In this example, the switch enforces a maximum queue size and drops packets when full. The `processQueue` method ensures that packets are sent in order.

x??

---

#### Packet Queuing on Destination Machine
When a packet reaches the destination machine, it may be queued by the operating system if all CPU cores are busy. This queueing can cause delays depending on the load of the machine.

:p What happens to network packets when all CPU cores are busy at the destination?
??x
If all CPU cores are busy at the destination machine, incoming network packets are queued by the operating system until resources become available for processing. The length of this queuing delay can vary significantly based on the overall load and availability of the CPU.

```java
// Pseudocode to simulate packet handling in an overloaded system
public void handlePacket(Packet p) {
    if (allCoresBusy()) { // Assume a function that checks core utilization
        queue.add(p); // Packet is added to a waiting queue
    } else {
        processPacket(p); // Immediate processing of the packet
    }
}
```
x??

---

#### Virtual Machine Queuing in Virtualized Environments
In virtualized environments, running operating systems can be paused for brief periods while another VM uses CPU cores. This pausing increases network delays as the VM is temporarily unable to consume incoming data.

:p How does virtualization affect network performance?
??x
Virtualization introduces additional queuing points where network traffic may experience delays. When a virtual machine (VM) is paused, it cannot process incoming packets immediately, leading to buffer queues managed by the virtual machine monitor (VMM). These queues contribute to increased variability in network delays.

```java
// Pseudocode for VMM managing VM pause and resume states
public void manageVirtualMachine(VirtualMachine vm) {
    if (needToPauseVM(vm)) { // Assume a function that decides when to pause the VM
        pauseVM(vm); // Pause the VM, potentially queuing incoming packets
        resumePreviousVM(); // Resume another VM on the same core
    } else {
        continueVM(vm); // Continue processing for the VM
    }
}
```
x??

---

#### TCP Flow Control and Congestion Avoidance
TCP performs flow control by limiting its own rate of sending data to avoid overwhelming network links or receiving nodes. This mechanism causes additional queueing at both the sender and receiver ends.

:p What is TCP's approach to managing network traffic?
??x
TCP uses flow control mechanisms like congestion avoidance to manage network traffic efficiently. By adjusting the rate of packet transmission based on observed network conditions, it prevents overloading of network links or receiving nodes. This process introduces additional queuing at both the sender and receiver.

```java
// Pseudocode for TCP flow control mechanism
public void sendPacket(Packet p) {
    if (shouldReduceSpeed()) { // Assume a function that checks congestion state
        slowStart(); // Reduce sending speed to avoid congestion
    } else {
        increaseSendRate(); // Increase sending rate if no congestion detected
    }
}
```
x??

---

#### Switch Queueing Delays in Networks
When multiple machines send traffic to the same destination, switch queues can fill up. This occurs because all ports trying to send packets to the same destination compete for limited bandwidth.

:p How does network switching contribute to delays?
??x
Switches manage data flow between devices and can create queuing delays when there is a high volume of traffic destined for the same port. If multiple machines attempt to send packets simultaneously towards the same switch port, they may all be queued until the switch has capacity to handle their transmission.

```java
// Pseudocode for switch queue management
public void sendPacket(SwitchPort source, SwitchPort destination) {
    if (destinationQueueFull(destination)) { // Check if the destination queue is full
        addToSwitchQueue(source, destination); // Add packet to the switch's waiting queue
        waitUntilTransmitPossible(); // Wait until the destination port can transmit
    } else {
        transmitPacketImmediately(source, destination); // Send packet directly without queuing
    }
}
```
x??

---

#### TCP Retransmission Mechanism
TCP considers a packet lost if it is not acknowledged within a timeout period. Lost packets are automatically retransmitted, adding to the variability of network delays even when the application does not see these delays.

:p How does TCP handle packet loss?
??x
TCP uses a retransmission mechanism where it waits for an acknowledgment before considering a packet lost. If a packet is not acknowledged within the configured timeout period (calculated based on observed round-trip times), TCP assumes the packet was lost and resends it. This process introduces delays as the application must wait for timeouts to expire.

```java
// Pseudocode for TCP retransmission logic
public void sendPacket(Packet p) {
    transmit(p); // Send packet over the network
    if (!ackReceived(p)) { // Check if acknowledgment was received
        timeoutStart(p); // Start a timer for the packet's expected arrival
        while (!ackReceived(p) && !timeoutExpired()) { // Wait for acknowledgment or timeout
            doNothing(); // Just wait, no action needed here
        }
        if (timeoutExpired() && !ackReceived(p)) {
            retransmitPacket(p); // Resend the packet if still not acknowledged
        }
    }
}
```
x??

---

#### UDP vs. TCP for Latency-Sensitive Applications
UDP is chosen over TCP in latency-sensitive applications like videoconferencing and VoIP because it avoids flow control and retransmission, reducing variability but potentially causing delays due to switch queues and scheduling.

:p What are the trade-offs between using UDP and TCP?
??x
The choice between UDP and TCP depends on the application's requirements. UDP is suitable for real-time applications like videoconferencing or VoIP because it avoids flow control and retransmission, reducing variability in packet delivery times. However, this comes at the cost of potential data loss since UDP does not guarantee packet delivery. TCP ensures reliable delivery but introduces more variable delays due to its congestion avoidance mechanisms.

```java
// Pseudocode for choosing between UDP and TCP based on application type
public Protocol chooseProtocol(Application app) {
    if (app.isRealTime()) { // Assume a function that checks if the application is real-time
        return UDP; // Use UDP for real-time applications
    } else {
        return TCP; // Use TCP for non-real-time applications
    }
}
```
x??

---

#### Network Variability and Noisy Neighbors
Background context: In public clouds and multi-tenant datacenters, shared resources can lead to network delays that are highly variable. This variability is often exacerbated by "noisy neighbors," where other customers use significant amounts of resources, affecting your application's performance.

:p How do noisy neighbors impact network reliability in a multi-tenant environment?
??x
Noisy neighbors can significantly increase network latency and packet loss, as the shared network resources are heavily utilized. This variability can make it challenging to predict and reliably service requests, especially in applications that require low-latency responses.
x??

---

#### Phi Accrual Failure Detector
Background context: The phi accrual failure detector is a mechanism used to detect failures based on observed response times rather than constant timeouts. It helps balance the trade-off between failure detection delay and risk of premature timeouts.

:p What is the phi accrual failure detector, and how does it work?
??x
The Phi Accrual Failure Detector is a technique that measures network round-trip times over an extended period across many machines to determine the expected variability of delays. It then uses this information to automatically adjust timeouts based on observed response time distributions.

Example pseudocode for phi accrual failure detector:
```java
public class PhiAccrualFailureDetector {
    private double phi;
    
    public void update(double responseTime) {
        // Update phi with the new response time
        // phi is a measure of the expected variability in delays
        phi = calculatePhi(responseTime);
    }
    
    public boolean isFailed() {
        // Check if the system has failed based on phi value
        return phi > failureThreshold;
    }
}

// Pseudocode for calculating phi (simplified)
public double calculatePhi(double responseTime) {
    // Implement logic to update phi based on observed response times
    // This involves statistical analysis of past and current response times
    // Example: phi = mean(responseTimes) + stdDev(responseTimes);
}
```
x??

---

#### Synchronous vs. Asynchronous Networks
Background context: Traditional fixed-line telephone networks provide reliable, low-latency connections due to the use of circuit switching. In contrast, datacenter networks and the internet use packet switching, which can suffer from unbounded delays and queueing.

:p What is the key difference between circuit-switched and packet-switched networks in terms of reliability?
??x
The key difference lies in how bandwidth is allocated and used:
- Circuit-switched networks allocate a fixed amount of bandwidth for each call, ensuring constant latency and reliable transmission.
- Packet-switched networks (like Ethernet and IP) dynamically allocate resources among multiple users, leading to variable delays but potentially higher utilization.

Example: In a circuit-switched network like ISDN, once a call is established, it gets guaranteed bandwidth. In contrast, in packet-switched networks, packets compete for bandwidth, which can lead to delays.
x??

---

#### Datacenter Network Utilization and Bursty Traffic
Background context: Datacenter networks are optimized for bursty traffic, dynamically allocating resources among multiple users. Circuit switching would be less efficient for these scenarios due to the need for frequent re-allocation of bandwidth.

:p Why do datacenter networks use packet switching instead of circuit switching?
??x
Datacenter networks use packet switching because it is better suited for handling bursty traffic:
- Packet switching dynamically allocates network resources, maximizing utilization.
- It allows multiple users to share bandwidth efficiently without the overhead of setting up fixed circuits.
- Circuits would be underutilized during periods of low demand and over-subscribed during spikes in demand.

Example: A web server might receive sporadic requests for data. With packet switching, the network can dynamically adjust resource allocation based on current demand.
x??

---

#### Quality of Service (QoS) and Admission Control
Background context: QoS and admission control mechanisms allow for more controlled resource sharing in packet-switched networks. These techniques can emulate circuit switching or provide statistically bounded delays.

:p How do QoS and admission control work to optimize network performance?
??x
Quality of Service (QoS) and admission control manage network resources by prioritizing and scheduling packets, and controlling the rate at which senders push data into the network:
- QoS involves prioritization and scheduling of packets based on application requirements.
- Admission control limits the number of users or the amount of traffic that can be admitted to ensure stable performance.

Example: InfiniBand uses end-to-end flow control to reduce queueing in networks, although it can still suffer from link congestion. By carefully managing QoS and admission control, statistically bounded delay can be achieved.
x??

---

#### Latency Guarantees vs. Utilization
Background context: Static resource partitioning (dedicated hardware) provides latency guarantees but reduces utilization due to fixed allocations. Dynamic resource sharing maximizes utilization but introduces variable delays.

:p What is the trade-off between latency guarantees and network utilization?
??x
The trade-off involves balancing:
- **Latency Guarantees**: Fixed resource allocation ensures low, predictable latency but may underutilize resources.
- **Utilization**: Dynamic resource sharing optimizes bandwidth usage but introduces variability in delay.

Example: In a datacenter, dedicating resources to a specific application provides guaranteed performance but may leave other applications with insufficient resources. Conversely, dynamically allocating resources among multiple applications maximizes overall throughput at the cost of increased latency variance.
x??

---

