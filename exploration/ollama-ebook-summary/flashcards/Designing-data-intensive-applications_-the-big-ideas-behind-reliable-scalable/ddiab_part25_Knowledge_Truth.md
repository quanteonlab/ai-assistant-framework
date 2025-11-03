# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 25)

**Starting Chapter:** Knowledge Truth and Lies. The Truth Is Defined by the Majority

---

#### Uncertainty in Distributed Systems
Background context: In distributed systems, nodes cannot be sure about the state or behavior of other nodes. They can only make guesses based on messages received through an unreliable network with variable delays. Partial failures and unreliable clocks further complicate the situation.

:p How does the uncertainty in distributed systems affect node behavior?
??x
In distributed systems, nodes must make decisions based on partial and potentially unreliable information from their peers. Because of the unreliable nature of message passing and potential partial failures, a node cannot be certain about another node’s state or availability. For instance, if a node fails to receive a response within a timeout period, it might incorrectly assume that the other node is dead.

```java
// Pseudocode for handling timeouts in distributed systems
public void handleMessage(Node sender, Message msg) {
    if (System.currentTimeMillis() - lastReceived[msg.source] > timeout) {
        markNodeAsDead(msg.source);
    }
}

private void markNodeAsDead(int nodeId) {
    // Update local state to reflect that the node is assumed dead
}
```
x??

---

#### Asymmetric Faults in Distributed Systems
Background context: An asymmetric fault occurs when a node can receive messages but not send them, leading other nodes to mistakenly declare it as faulty.

:p How does an asymmetric fault manifest in a distributed system?
??x
An asymmetric fault happens when a node is able to receive all incoming messages but cannot send any outgoing ones. This situation can lead to other nodes wrongly declaring the node dead or malfunctioning because they do not receive acknowledgments from it. The node might be fully functional and receiving requests, but without sending responses, it appears non-responsive.

```java
// Pseudocode for detecting an asymmetric fault
public class Node {
    private boolean isFaulty = false;

    public void receiveMessage(Message msg) {
        // Process the incoming message
        if (!sendAck(msg)) {  // sendAck() returns false due to faulty network or local failure
            markNodeAsDead();
        }
    }

    private boolean sendAck(Message msg) {
        // Simulate sending an acknowledgment, which might fail
        return Math.random() < 0.5;  // Randomly decide if the send is successful
    }

    private void markNodeAsDead() {
        isFaulty = true;
        notifyOtherNodes();
    }
}
```
x??

---

#### Majority Decisions in Distributed Systems
Background context: In distributed systems, decisions often rely on majority consensus. If a node does not hear from others within a timeout period or if it notices discrepancies, it may take actions based on the majority view.

:p How can a node determine the state of another node when faced with network delays and unresponsive nodes?
??x
A node can determine the state of another node by sending messages and waiting for responses. If no response is received within a timeout period or if the node notices that its messages are not being acknowledged, it may infer that there might be an issue but cannot be certain unless a majority of other nodes agree on the state.

```java
// Pseudocode for determining node state based on majority consensus
public class Node {
    private Map<Node, Boolean> receivedResponses;

    public void requestState(Node target) {
        sendRequest(target);
        waitForResponse(target);

        if (isTimeout()) {
            markNodeAsDead(target);
        }
    }

    private void sendRequest(Node target) {
        // Send a request to the target node
        receivedResponses.put(target, false);
    }

    private void waitForResponse(Node target) {
        // Simulate waiting for response from the target node
        if (!receivedResponses.get(target)) {
            markNodeAsDead(target);
        }
    }

    private boolean isTimeout() {
        // Check if the timeout period has expired
        return true;  // Simplified check, in reality, it would be more complex
    }

    private void markNodeAsDead(Node target) {
        // Update local state to reflect that the node is assumed dead
    }
}
```
x??

---

#### Long Garbage Collection Pauses
Background context: In distributed systems, a node might experience long pauses during garbage collection. This can affect its ability to respond to messages in a timely manner.

:p How can a node handle long garbage collection pauses while maintaining system reliability?
??x
During long garbage collection (GC) pauses, nodes may not be able to process or send messages in a timely fashion. To maintain system reliability, nodes should implement strategies such as queuing incoming requests and attempting to resume processing as soon as the pause ends.

```java
// Pseudocode for handling long GC pauses
public class Node {
    private Queue<Message> messageQueue;

    public void handleRequest(Message msg) {
        if (isGCInProgress()) {  // Simulate checking if a garbage collection is in progress
            queueMessage(msg);
        } else {
            processMessageImmediately(msg);
        }
    }

    private void queueMessage(Message msg) {
        // Add the message to the queue for processing after GC finishes
        messageQueue.add(msg);
    }

    private void processMessageImmediately(Message msg) {
        // Process the message as soon as possible
        handleMessage(msg);
    }

    private boolean isGCInProgress() {
        return System.currentTimeMillis() - lastGCEnd > gcPauseThreshold;
    }
}
```
x??

---

#### GC Paused Nodes and Quorum Decisions
Background context: This concept explains how garbage collection (GC) pauses can affect a node's operation within a distributed system, leading to scenarios where nodes may incorrectly declare each other as dead. It emphasizes the importance of quorums in making decisions about the state of nodes.

:p What is the impact of GC on a node in a distributed system?
??x
During a garbage collection (GC) pause, all threads of a node are preempted and paused, preventing any request processing or response sending. This can lead to other nodes waiting for an extended period before concluding that the node has failed and removing it from service.
```java
// Pseudocode example showing GC pause effect
public void handleRequest() {
    try {
        // Simulate request handling
        processRequest();
    } catch (ThreadInterruptionException e) {
        System.out.println("GC paused, unable to process request.");
    }
}
```
x??

---

#### Quorum Voting Mechanism
Background context: The use of quorums in distributed systems ensures that decisions are made based on the agreement of a majority of nodes. This prevents single-node failures from causing system-wide issues.

:p How does a quorum mechanism help in decision-making within a distributed system?
??x
A quorum mechanism helps by requiring a minimum number of votes (a majority) from several nodes to make a decision, thereby reducing reliance on any single node. For example, with five nodes, at least three must agree for a decision to be valid.
```java
// Pseudocode example of a simple quorum voting system
public boolean makeDecision() {
    int votes = 0;
    // Simulate voting process
    if (vote(true)) votes++;
    if (vote(false)) votes++;
    return votes > 2; // Return true if more than half voted yes
}
```
x??

---

#### Handling Node Failures and Split Brain
Background context: In distributed systems, split brain occurs when nodes diverge into two separate groups that think they are the primary node. This can lead to data corruption or service failures.

:p What is split brain in a distributed system?
??x
Split brain happens when two or more nodes believe they should be the leader (primary) for a resource, leading them to make conflicting decisions and potentially causing data corruption or service outages.
```java
// Pseudocode example of handling potential split brain scenario
public void electLeader() {
    if (checkMajorityConsensus(true)) {
        // Leader elected
    } else {
        // Handle failed leader election process
    }
}
```
x??

---

#### Importance of Quorums in Consensus Algorithms
Background context: Quorums are crucial in ensuring that decisions made by distributed systems are consistent and reliable, even when some nodes fail. This is particularly important for consensus algorithms where agreement among multiple nodes is necessary.

:p Why are quorums essential in the implementation of consensus algorithms?
??x
Quorums ensure consistency and reliability in a distributed system by requiring a majority vote from several nodes to make decisions. This prevents single-node failures from causing incorrect state changes or service disruptions, maintaining the integrity of the system.
```java
// Pseudocode example of quorum-based decision making
public boolean consensus(String decision) {
    int requiredMajority = (nodes.size() / 2) + 1;
    int votesForDecision = 0;
    for (Node node : nodes) {
        if (node.decide(decision)) {
            votesForDecision++;
        }
    }
    return votesForDecision >= requiredMajority;
}
```
x??

---

#### Distributed System Reliability Through Redundancy
Background context: In a distributed system, relying on a single node can lead to failure and downtime. Therefore, implementing redundancy through quorums helps ensure that the system remains operational even when some nodes fail.

:p How does redundancy improve the reliability of a distributed system?
??x
Redundancy improves reliability by ensuring that decisions are based on multiple nodes rather than just one. Quorum-based systems can handle node failures gracefully because they require agreement from a majority of nodes before making decisions, reducing the risk of incorrect state changes or service disruptions.
```java
// Pseudocode example of handling node failure and redundancy
public void ensureRedundancy() {
    List<Node> aliveNodes = getAliveNodes();
    if (aliveNodes.size() >= requiredMajority) {
        // Continue operation with quorum support
    } else {
        // Handle failure scenario
    }
}
```
x??

---

#### Distributed Lock Implementation Bug
Background context: In a distributed system, ensuring exclusive access to a resource (like a file) by a single client at a time is crucial. An incorrect implementation of locking can lead to data corruption when a lease expires but the client continues to believe it has valid access.
:p What issue arises due to an incorrect implementation of a distributed lock?
??x
The issue is that if a client holding a lease pauses for too long, its lease might expire while another client acquires the lease. When the original client resumes, it mistakenly believes it still holds the valid lease and tries to write to the file, leading to data corruption.
x??

---

#### Fencing Tokens Concept
Background context: To prevent such issues in distributed systems where a resource (like storage) is accessed under lock, fencing tokens are used. These tokens ensure that writes occur only in the order of increasing numbers, thus preventing overlapping writes from different clients.
:p What mechanism can be used to protect against a client acting on an expired lease?
??x
A mechanism called fencing tokens can be used. Every time a lock or lease is granted by the server, it returns a token that increases each time a new lock is acquired. Clients must include this token with their write requests. If a paused client resumes and attempts to write without a valid newer token, the storage service rejects the request.
x??

---

#### Implementing Fencing Tokens in Practice
Background context: In practice, ZooKeeper can be used as a lock service that returns fencing tokens like transaction IDs (zxid) or node versions (cversion). These are guaranteed to be monotonically increasing and suitable for fence checks.
:p How do fencing tokens work with ZooKeeper?
??x
Fencing tokens in ZooKeeper, such as transaction IDs (zxid) or node versions (cversion), are used. Every time a lock is granted, these values increase. When a client wants to write, it includes the current token. The storage service checks if this token is newer than any previously processed writes. If not, the request is rejected.
x??

---

#### Overcoming Lack of Explicit Fencing Tokens
Background context: For systems that do not natively support fencing tokens, alternative methods can be employed. For example, in file storage services, including a timestamp or a unique identifier as part of the filename can serve as a workaround for fence checks.
:p How can you handle resources without explicit support for fencing tokens?
??x
For resources that lack explicit support for fencing tokens, you can include some form of fencing token in the filename. For instance, append a unique identifier (like a timestamp) to each file name used by clients. This way, when writing, you ensure that only newer versions are accepted.
x??

---

#### Byzantine Faults
Background context explaining the concept. In distributed systems, nodes may be unreliable and sometimes malicious. A Byzantine fault is a situation where a node might send arbitrary faulty or corrupted responses. This can happen due to various reasons including network delays, outdated state, etc.

If a node wants to subvert the system's guarantees, it could send messages with a fake fencing token. However, in this book, we assume nodes are unreliable but honest: they may be slow or never respond (due to faults), and their state might be outdated due to GC pauses or network delays. But if a node does respond, it is telling the truth according to its current knowledge.

:p What are Byzantine faults?
??x
Byzantine faults refer to situations in distributed systems where nodes may send arbitrary faulty or corrupted responses. These can occur due to unreliability (like network delays) or malicious intent.
x??

---

#### Fencing Tokens
Background context explaining the concept. Fencing tokens can help detect and block a node that is inadvertently acting in error, such as when it hasn’t yet found out that its lease has expired.

:p What is the purpose of fencing tokens?
??x
Fencing tokens are used to detect and block nodes that are accidentally misbehaving (e.g., due to a lease expiration). They help protect services from abusive clients.
x??

---

#### Byzantine Generals Problem
Background context explaining the concept. The Byzantine Generals Problem is a generalization of the Two Generals Problem, where generals need to agree on a battle plan despite unreliable communication. In the Byzantine version, some traitors might send false messages to confuse others.

:p What is the Byzantine Generals Problem?
??x
The Byzantine Generals Problem is about reaching consensus among nodes in a distributed system where some nodes may be malicious and send fake or untrue messages.
x??

---

#### Byzantine Fault-Tolerant Systems
Background context explaining the concept. A system is Byzantine fault-tolerant if it can operate correctly even when some nodes are malfunctioning or being attacked by malicious entities.

:p What makes a system Byzantine fault-tolerant?
??x
A system is considered Byzantine fault-tolerant if it continues to function correctly despite some nodes malfunctioning or being compromised. This concern is relevant in specific scenarios like aerospace environments where failures can be catastrophic.
x??

---

#### Practical Considerations in Distributed Systems
Background context explaining the concept. In many server-side data systems, assuming Byzantine faults is impractical due to low memory corruption rates and controlled hardware.

:p Why are Byzantine fault-tolerant solutions often impractical for most server-side data systems?
??x
Byzantine fault-tolerant solutions are impractical in most server-side data systems because of the practical costs involved. Memory corruption rates are low, and nodes are usually controlled by a single organization, making them more reliable.
x??

---

#### Web Application Security
Background context explaining the concept. Web applications need to handle arbitrary and malicious client behavior since clients like web browsers can be under end-user control.

:p Why do web applications need input validation?
??x
Web applications require input validation because users (end-users) can send arbitrary and potentially malicious inputs through web browsers. This helps prevent issues like SQL injection and cross-site scripting.
x??

---

#### Peer-to-Peer Networks and Byzantine Fault Tolerance
Background context explaining the concept. In peer-to-peer networks, there is no central authority to rely on, making Byzantine fault tolerance more relevant.

:p Why are Byzantine fault-tolerant protocols important in peer-to-peer networks?
??x
Byzantine fault-tolerant protocols are crucial in peer-to-peer networks because these systems lack a central authority. This means nodes must be able to agree on data without trusting each other completely.
x??

---

#### Supermajority Requirement for Byzantine Fault Tolerance
Background context explaining the concept. Most Byzantine fault-tolerant algorithms require more than two-thirds of nodes to function correctly.

:p What is the supermajority requirement in Byzantine fault tolerance?
??x
Byzantine fault-tolerant protocols generally require a supermajority (more than two-thirds) of nodes to be functioning correctly. This means that even if some nodes are faulty, the majority can still reach agreement.
x??

---

#### Weak Forms of Lying
Background context explaining the concept. In distributed systems, even though nodes are generally assumed to be honest, it's important to implement mechanisms that guard against weak forms of "lying." These can include invalid messages due to hardware issues, software bugs, and misconfiguration. Such protection mechanisms provide a simple layer of reliability without full Byzantine fault tolerance.

:p What is an example of a mechanism to protect against weak forms of lying in distributed systems?
??x
One example is the use of checksums in application-level protocols to detect corrupted network packets due to hardware issues or bugs. Additionally, sanitizing user inputs and performing sanity checks can prevent denial of service attacks.
??x

---

#### NTP Client Configuration for Robust Time Synchronization
Background context explaining the concept. Network Time Protocol (NTP) clients can enhance their robustness by contacting multiple server addresses during synchronization. By checking that a majority of servers agree on some time range, misconfigured or incorrect time reports are detected and excluded.

:p How does using multiple NTP servers improve the robustness of time synchronization?
??x
Using multiple NTP servers improves robustness because it allows the client to estimate errors and exclude any outliers. For instance, an NTP client can contact all available server addresses, calculate error estimates, and ensure that a majority of servers agree on a reasonable time range.
??x

---

#### System Model: Synchronous Model
Background context explaining the concept. The synchronous model in distributed systems assumes bounded network delay, process pauses, and clock errors, meaning you know these values will never exceed some fixed upper bound.

:p What is the synchronous system model used for?
??x
The synchronous system model is used to design algorithms that can tolerate various faults by assuming certain constraints on timing. It helps in writing robust algorithms that do not depend heavily on unpredictable hardware and software configurations.
??x

---

#### System Model: Partially Synchronous Model
Background context explaining the concept. The partially synchronous model acknowledges that systems behave like synchronous ones most of the time, but occasionally exceed the bounds for network delay, process pauses, and clock drift.

:p How does the partially synchronous system model differ from the synchronous one?
??x
The partially synchronous system model differs because it allows for occasional breaches in timing assumptions. While the system behaves well most of the time, network delays, process pauses, and clock errors can become arbitrarily large when these rare events occur.
??x

---

These flashcards cover key concepts in distributed systems from the provided text, including mechanisms to protect against weak forms of lying, NTP client configuration for robust time synchronization, and system models used in algorithm design.

---
#### Asynchronous Model
The asynchronous model is a very restrictive approach where an algorithm cannot make any timing assumptions and does not have access to a clock. This means it must handle situations without knowing when operations will complete or what the current state of other nodes might be.

In this model, besides timing issues, node failures are also a concern. The three most common system models for nodes include:
- **Crash-stop faults**: Nodes can fail by crashing and do not come back.
- **Crash-recovery faults**: Nodes can crash at any moment but have stable storage that is preserved across crashes.
- **Byzantine (arbitrary) faults**: Nodes can behave in arbitrary ways, potentially deceiving other nodes.

:p What are the three common node failure models in distributed systems?
??x
The three common node failure models include:
1. Crash-stop faults: Nodes stop responding and do not come back.
2. Crash-recovery faults: Nodes may crash but eventually recover with preserved stable storage.
3. Byzantine (arbitrary) faults: Nodes can behave arbitrarily, including lying or trying to deceive other nodes.

x??

---
#### Correctness of Distributed Algorithms
To define the correctness of a distributed algorithm, one must specify its properties. For example:
- **Uniqueness**: Ensure no two requests for a fencing token return the same value.
- **Monotonic sequence**: Tokens should be returned in a strictly increasing order based on request completion times.
- **Availability**: Requests should eventually receive a response if they do not crash.

:p How does an algorithm's correctness relate to its properties in distributed systems?
??x
An algorithm is considered correct if it satisfies its defined properties under all possible scenarios specified by the system model. For instance, in generating fencing tokens for locks:
- Uniqueness ensures that no two requests return the same token.
- Monotonic sequence ensures tokens are returned in a strictly increasing order based on request completion times.
- Availability ensures that non-crashing nodes receive responses eventually.

x??

---
#### Safety vs. Liveness Properties
Safety properties ensure "nothing bad happens," while liveness properties guarantee "something good eventually happens." Key safety and liveness examples include:
- **Uniqueness** and **monotonic sequence**: These are safety properties because they ensure specific behaviors do not lead to harmful outcomes.
- **Availability**: This is a liveness property since it ensures some positive outcome (response) will eventually occur.

:p What distinguishes safety from liveness properties in distributed systems?
??x
Safety properties guarantee that "nothing bad happens," meaning the algorithm does not produce incorrect or harmful outputs. Liveness properties ensure that "something good eventually happens," such as a node receiving a response. For example:
- Uniqueness and monotonic sequence are safety properties because they prevent collisions or ordering issues.
- Availability is a liveness property, ensuring nodes get responses over time.

x??

---

#### Safety Properties Violation
Background context: In distributed systems, safety properties define behaviors that should never occur. If a safety property is violated, we can pinpoint exactly when and where it happened. The violation cannot be undone; once it occurs, the damage is done.

:p What happens if a safety property is violated in a distributed system?
??x
If a safety property is violated, we can identify a specific point in time when it was broken. For example, if the uniqueness property was violated, we can determine which operation resulted in duplicate fencing tokens being returned. Once this violation occurs, it cannot be undone; the damage is permanent.

This means that even after fixing the system, any subsequent state might reflect the incorrect outcome caused by the safety violation.
x??

---

#### Liveness Properties
Background context: Liveness properties ensure that the system eventually reaches a desired state or behavior. Unlike safety properties, liveness violations can be resolved over time if certain conditions are met. For example, a request may not receive a response immediately, but it might in the future.

:p What distinguishes liveness from safety properties?
??x
Liveness properties allow for situations where the property might temporarily fail to hold, but there is always hope that it can be satisfied eventually. In contrast, safety properties must never be violated under any circumstances.

For example:
- Safety: A distributed algorithm must ensure no duplicate transactions occur.
- Liveness: A request should receive a response from a majority of nodes if the network recovers and most nodes are still operational.

Code Example (Pseudocode):
```pseudocode
if(network_recovered() && majority_nodes_operational()) {
    send_response()
}
```
x??

---

#### Partially Synchronous Model
Background context: The partially synchronous model is a system model that assumes the network will eventually return to a synchronous state after a period of disruption. This means any network interruption lasts only for a finite duration and then gets repaired.

:p What does the definition of the partially synchronous model require?
??x
The partially synchronous model requires that the system returns to a synchronous state eventually, meaning any period of network interruption lasts only for a finite duration and is then repaired.

This implies:
- Networks may experience outages or disruptions.
- After these disruptions, normal operation should resume within a finite time frame.
- This model helps in designing algorithms that can handle temporary network issues while ensuring eventual correctness.

Code Example (Pseudocode):
```pseudocode
if(network_interrupted()) {
    wait_until_network_recovered()
}
```
x??

---

#### System Models and Reality Mismatch
Background context: While system models help in reasoning about the correctness of distributed algorithms, they often simplify real-world complexities. Assumptions such as data surviving crashes or nodes remembering stored data can break down in practice.

:p What are some common issues with system model assumptions?
??x
Common issues with system model assumptions include:

- Data Corruption: Data on disk might get corrupted due to hardware errors, firmware bugs, or misconfigurations.
- Hardware Failures: A server might fail to recognize its hard drives upon reboot, even if the drives are correctly attached.
- Node Amnesia: Nodes may forget data they previously stored, breaking quorum conditions.

These issues highlight the need for more realistic system models but also make reasoning about distributed algorithms harder because assumptions that simplify analysis can no longer be relied upon.

Code Example (Pseudocode):
```pseudocode
if(is_disk_corrupted() || is_firmware_bug()) {
    handle_data_recovery()
}
```
x??

---

#### Quorum Algorithms and Node Amnesia
Background context: Quorum algorithms rely on nodes remembering the data they store. If a node suffers from amnesia, it can break the quorum condition, leading to incorrect algorithm behavior.

:p How does node amnesia affect the correctness of quorum algorithms?
??x
Node amnesia affects the correctness of quorum algorithms because these algorithms depend on nodes accurately recalling and verifying the stored data. If a node forgets previously stored data, this breaks the quorum conditions, thus breaking the algorithm's correctness.

For example:
- A node claiming to have stored data might actually not have it anymore.
- This can lead to incorrect read or write operations violating the algorithm’s intended behavior.

Code Example (Pseudocode):
```pseudocode
if(node_forgets_data()) {
    invalidate_quorum_conditions()
}
```
x??

