# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 30)

**Starting Chapter:** Fault-Tolerant Consensus

---

---
#### Scalability and Coordination Challenges in Application Servers
Background context: The text discusses the challenges associated with adding or removing application servers, especially when a coordinator is part of an application server. This changes the nature of deployment as the logs become critical for recovery after a crash, making the application stateful.
:p What are the implications of having a coordinator within an application server?
??x
Having a coordinator within an application server means that its logs are crucial for recovering in-doubt transactions after a crash. Unlike stateless systems where failures can be handled by restarting processes, this setup requires the coordinator's logs to ensure transactional consistency. This makes the system more complex and less fault-tolerant as it relies heavily on the coordinator.
x??
---

---
#### Limitations of XA Transactions
Background context: The text highlights that XA transactions have limitations due to their need to be compatible with a wide range of data systems, making them a lowest common denominator. These include not being able to detect deadlocks across different systems and compatibility issues with Serializable Snapshot Isolation (SSI).
:p What are the key limitations of XA transactions?
??x
Key limitations of XA transactions include:
- Inability to detect deadlocks across different systems.
- Lack of support for SSI, requiring a standardized protocol for conflict identification between systems.
- Necessity to be a lowest common denominator due to compatibility with various data systems.
x??
---

---
#### Impact on Fault-Tolerance in Distributed Transactions
Background context: The text discusses the challenges that distributed transactions pose to fault-tolerant systems. Specifically, it mentions that if any part of the system fails, all participants must respond for the transaction to commit successfully, which can amplify failures.
:p How do distributed transactions affect fault tolerance?
??x
Distributed transactions can amplify failures because the success of a transaction depends on all participants responding successfully. If any participant fails before responding, the transaction will fail even if others have completed their part. This runs counter to building fault-tolerant systems where resilience is crucial.
x??
---

---
#### Consensus Algorithms: Uniform Agreement
Background context: The text introduces the concept of consensus in distributed systems, formalizing it as a mechanism for multiple nodes to agree on a value. It specifies that uniform agreement means no two nodes decide differently, and every non-crashing node eventually decides some value.
:p What is uniform agreement in consensus algorithms?
??x
Uniform agreement in consensus algorithms ensures that no two non-failing nodes decide on different values. This property guarantees consistency among the participating nodes by preventing conflicts over proposed values.
x??
---

---
#### Validity in Consensus Algorithms
Background context: The text details another property of consensus algorithms called validity, which states that if a node decides a value v, then v must have been proposed by some node.
:p What does the validity property guarantee in consensus algorithms?
??x
The validity property guarantees that once a value v is decided upon by any non-failing node, it was indeed proposed by at least one of the nodes. This ensures that decisions are based on proposals made within the system, maintaining consistency and reliability.
x??
---

---
#### Uniform Agreement and Integrity Properties
Background context: The uniform agreement and integrity properties are fundamental to consensus mechanisms, ensuring that all nodes agree on a single outcome and once a decision is made, it cannot be changed. This forms the core idea of consensus where everyone decides on the same outcome.

:p What do the uniform agreement and integrity properties ensure in a consensus mechanism?
??x
These properties ensure that all nodes in a system decide on the same outcome (uniform agreement) and that decisions once made cannot be changed (integrity). This means that if one node agrees to commit to a transaction, no other node can later vote against it.
x??

---
#### Validity Property
Background context: The validity property ensures that trivial solutions such as always deciding null are ruled out. For example, an algorithm should not decide null regardless of the input.

:p What is the purpose of the validity property in consensus mechanisms?
??x
The validity property ensures that a consensus mechanism does more than just decide on a default value (like null). It mandates that decisions made by the system must be meaningful and relevant to the proposed action, thus ruling out trivial solutions.
x??

---
#### Hardcoding a Dictator Node
Background context: If you do not care about fault tolerance, achieving agreement, integrity, and validity is straightforward—hardcoding one node as a dictator can make all decisions. However, this approach fails if that node fails.

:p How does hardcoding one node as a dictator affect the system's reliability?
??x
Hardcoding one node as a dictator makes the system highly unreliable because if that single node fails, no decisions can be made by the entire system. This is particularly evident in distributed systems where nodes must remain operational for consensus to proceed.
x??

---
#### Two-Phase Commit (2PC)
Background context: The two-phase commit protocol (2PC) is a common method used in distributed transactions. However, it fails the termination property if the coordinator node crashes because in-doubt participants cannot decide whether to commit or abort.

:p What issue does 2PC face when the coordinator node fails?
??x
When the coordinator node fails, 2PC faces the problem of in-doubt participants who cannot determine whether to commit or abort the transaction. This failure can lead to deadlocks where transactions remain in an undecided state.
x??

---
#### Termination Property
Background context: The termination property ensures that a consensus algorithm must make progress and not sit idle indefinitely. In other words, it guarantees that decisions will be made even if some nodes fail.

:p What does the termination property ensure in a consensus mechanism?
??x
The termination property ensures that a consensus algorithm must eventually reach a decision despite failures among nodes. This means that no matter what happens (even with node crashes), the system should not get stuck indefinitely and must make progress.
x??

---
#### System Model for Consensus
Background context: The system model for consensus assumes nodes crash by suddenly disappearing, never to return. This assumption is made to ensure algorithms can tolerate failures without getting stuck.

:p What assumption does the consensus algorithm make about node crashes?
??x
The consensus algorithm assumes that when a node crashes, it will disappear and not come back online. This means any solution relying on recovering from such nodes will fail to satisfy the termination property.
x??

---
#### Majority Requirement for Termination
Background context: It has been proven that no consensus algorithm can guarantee termination if more than half of the nodes are faulty or unreachable. At least a majority of correctly functioning nodes is required.

:p What is the minimum requirement for ensuring termination in consensus algorithms?
??x
To ensure termination, consensus algorithms require at least a majority of nodes to be functioning correctly. This means that fewer than half of the nodes can fail without causing the system to get stuck indefinitely.
x??

---
#### Safety vs Liveness Properties
Background context: Safety properties (agreement, integrity, and validity) guarantee correctness in decisions, while liveness properties like termination ensure that progress is made.

:p How do safety and liveness properties differ?
??x
Safety properties ensure that the system does not make incorrect or invalid decisions. Integrity ensures once a decision is made, it cannot be changed. Validity ensures decisions are not trivial. On the other hand, liveness properties like termination guarantee that the system will eventually reach a decision.
x??

---
#### Byzantine Faults
Background context: Most consensus algorithms assume nodes do not exhibit Byzantine faults—where nodes send contradictory messages or fail to follow the protocol correctly.

:p What is the assumption about node behavior in most consensus algorithms?
??x
Most consensus algorithms assume that nodes do not have Byzantine faults, meaning they strictly adhere to the protocol and do not send conflicting messages. If a node behaves erratically (Byzantine fault), it can break the safety properties of the protocol.
x??

---

#### Consensus and Byzantine Faults
Background context: Consensus algorithms are designed to ensure that a group of nodes agrees on a single value. In the presence of Byzantine faults, consensus is possible if fewer than one-third of the nodes are faulty.

:p What makes consensus robust against Byzantine faults?
??x
Consensus can be made robust against Byzantine faults as long as fewer than one-third of the nodes are faulty. This is based on theoretical results in distributed systems.
x??

---
#### Fault-Tolerant Consensus Algorithms
Background context: Several well-known algorithms like Viewstamped Replication (VSR), Paxos, Raft, and Zab are used for fault-tolerant consensus.

:p Which are the best-known fault-tolerant consensus algorithms?
??x
The best-known fault-tolerant consensus algorithms include Viewstamped Replication (VSR), Paxos, Raft, and Zab.
x??

---
#### Total Order Broadcast
Background context: Many consensus algorithms can be seen as a form of total order broadcast, where messages are delivered in the same order to all nodes. This is equivalent to performing several rounds of consensus.

:p How does total order broadcast relate to consensus?
??x
Total order broadcast involves delivering messages exactly once, in the same order, to all nodes. This can be seen as repeated rounds of consensus, with each round deciding on one message for delivery.
x??

---
#### Viewstamped Replication (VSR), Raft, and Zab
Background context: These algorithms implement total order broadcast directly, making them more efficient than doing repeated rounds of one-value-at-a-time consensus.

:p What do VSR, Raft, and Zab have in common?
??x
VSR, Raft, and Zab are similar in that they all implement total order broadcast. They use the agreement property to ensure messages are delivered in the same order, integrity to prevent duplication, validity to ensure messages aren't corrupted, and termination to avoid message loss.
x??

---
#### Multi-Paxos Optimization
Background context: Paxos can be optimized by performing multiple rounds of consensus decisions for a sequence of values.

:p What is Multi-Paxos?
??x
Multi-Paxos is an optimization of the Paxos algorithm that involves deciding on a sequence of values through repeated rounds of consensus, where each round decides on one message to be delivered in the total order.
x??

---
#### Single-Leader Replication and Consensus
Background context: Single-leader replication, discussed in Chapter 5, can be seen as a form of total order broadcast if the leader is chosen by humans.

:p Why wasn’t consensus needed in Chapter 5 when discussing single-leader replication?
??x
In Chapter 5, consensus was not explicitly discussed because single-leader replication relies on manual selection and configuration of leaders. This results in a "consensus algorithm" where only one node accepts writes and applies them to followers in the same order, ensuring consistency.
x??

---

#### Epoch Numbering and Quorums
Epoch numbering is a mechanism used to ensure uniqueness of leaders within consensus protocols. Each epoch has an incremented number, and nodes rely on these numbers to resolve leader conflicts. The protocol guarantees that only one leader exists per epoch, which helps prevent split brain scenarios where multiple nodes believe they are the leader.

A quorum system ensures that decisions are made with agreement from a majority of the nodes. In distributed systems, this is crucial for maintaining consistency and preventing conflicting states.

:p What is an epoch number in consensus protocols?
??x
An epoch number or ballot number (in Paxos), view number (in Viewstamped Replication), and term number (in Raft) is used to ensure that the leader is unique within a specific timeframe. Each time a potential leader is considered dead, nodes initiate a vote to elect a new leader, incrementing the epoch number each time.
x??

---

#### Leader Election with Epoch Numbers
Epoch numbers play a critical role in ensuring consistency by providing a mechanism for resolving conflicts between leaders across different epochs. A higher epoch number always prevails when there is a conflict.

:p How does epoch numbering help resolve conflicts between leaders?
??x
Epoch numbering helps resolve conflicts by assigning each leader election a unique, monotonically increasing number. When two nodes claim leadership simultaneously and their proposed values are in conflict, the node with the higher epoch number is chosen as the valid leader.
x??

---

#### Role of Quorums in Decision Making
Quorums ensure that decisions made by leaders are accepted by a majority of nodes. A quorum typically consists of more than half of all nodes in the system.

:p What is a quorum in the context of consensus protocols?
??x
A quorum is a subset of nodes whose agreement is required to make a decision valid and final in distributed systems. For every decision, the leader sends its proposal to the other nodes and waits for a quorum of them to agree.
x??

---

#### How Leaders Ensure Their Leadership Status
Leaders must verify their leadership status by collecting votes from a quorum of nodes. This process ensures that no higher epoch number has taken over.

:p How do leaders check if they are still valid leaders?
??x
Leaders check their validity by requesting votes from a quorum of nodes, ensuring none of the other nodes have a higher epoch number that might take leadership. If no conflicting leader is found in this vote, the current leader can be confident it holds the leadership.
x??

---

#### Overlapping Quorums for Consistency
To ensure consistency, both the election and proposal quorums must overlap. This ensures that any decision made by a leader has been validated during its leadership election.

:p Why are overlapping quorums important in consensus protocols?
??x
Overlapping quorums are crucial because they ensure that decisions made by leaders have already been validated during their leadership elections. This prevents conflicts and ensures that the current leader can conclude it still holds the leadership if no higher epoch number is detected.
x??

---

#### Example of Overlapping Quorums in Code

:p How does overlapping quorum logic work in a simple pseudocode example?
??x
In a simple scenario, a leader sends its decision to a set of nodes and waits for a majority response. The same set of nodes that participated in the leader election must also validate the proposal.

```java
public class LeaderElection {
    private Set<Node> quorum; // Nodes involved in leadership election

    public boolean propose(int value) {
        // Send proposal to all nodes
        Map<Node, Boolean> responses = new HashMap<>();
        for (Node node : quorum) {
            responses.put(node, sendProposal(node, value));
        }

        // Check if a quorum of nodes responded positively
        int affirmativeVotes = 0;
        for (Boolean response : responses.values()) {
            if (response) {
                affirmativeVotes++;
            }
        }

        return affirmativeVotes > quorum.size() / 2; // Majority rule
    }

    private boolean sendProposal(Node node, int value) {
        // Simulate sending the proposal and getting a response
        // In reality, this would involve network communication
        // Return true if the node accepted the proposal
        return Math.random() < 0.5; // Simplified logic for illustration
    }
}
```
x??

---

#### Two-Phase Commit (2PC) vs. Consensus Algorithms

Background context: The passage discusses the differences between two-phase commit and consensus algorithms, particularly focusing on their voting processes and fault tolerance mechanisms.

:p How do 2PC and consensus algorithms differ in their voting process?
??x
In two-phase commit (2PC), there is no elected coordinator; instead, each participant decides independently whether to commit or rollback based on the messages received from other participants. Consensus algorithms elect a leader (or coordinator) that gathers votes from a majority of nodes before deciding the proposed value.

Consensus algorithms require only a majority for decision-making, whereas 2PC requires all participants ("yes" vote from every participant) to ensure agreement.

??x
The answer is about the differences in how these two processes handle voting and decision-making. The leader election process in consensus algorithms simplifies the decision process by requiring only a majority, while 2PC needs unanimous consent.
```java
// Pseudocode for a simple 2PC scenario
public class TwoPhaseCommit {
    public void startTransaction() { /* initiate transaction */ }
    
    public void proposeValue(boolean value) { /* propose value to all participants */ }
    
    public boolean decideValue() {
        // Gather votes from all participants
        if (majorityVotesFor(value)) {
            return true; // Commit the transaction
        } else {
            return false; // Rollback the transaction
        }
    }

    private boolean majorityVotesFor(boolean value) { /* implementation */ }
}
```
x??

---

#### Majority Requirement in Consensus Algorithms

Background context: The passage explains that consensus algorithms require a strict majority to operate, which means at least three nodes are needed for one failure tolerance, and five nodes for two failures.

:p How many nodes are required for fault tolerance in a consensus algorithm?
??x
For fault tolerance in consensus algorithms, you need a minimum of three nodes to tolerate one failure (as the remaining two out of three form a majority). To tolerate two failures, you would need at least five nodes (with the remaining three forming a majority).

??x
The answer is based on the requirement for a strict majority. In a system with \( N \) nodes, the minimum number of nodes required to tolerate \( f \) failures is \( 2f + 1 \). For one failure tolerance:
- Nodes = 3 (since \( 2*1 + 1 = 3 \))
For two failure tolerance:
- Nodes = 5 (since \( 2*2 + 1 = 5 \))

Example for three nodes and one failure tolerance:
```java
public class ConsensusAlgorithm {
    private int totalNodes; // Total number of nodes in the system
    private int failedNodes; // Number of failed nodes

    public ConsensusAlgorithm(int totalNodes) { this.totalNodes = totalNodes; }

    public boolean tolerateFailure() {
        if (totalNodes >= 3 && failedNodes <= 1) return true;
        return false;
    }
}
```
x??

---

#### Recovery Process in Consensus Algorithms

Background context: The passage mentions that consensus algorithms include a recovery process to ensure nodes can get into a consistent state after a new leader is elected, maintaining safety properties.

:p What is the role of the recovery process in consensus algorithms?
??x
The recovery process in consensus algorithms ensures that nodes can be brought back into a consistent and correct state following the election of a new leader. This helps maintain the safety properties (agreement, integrity, and validity) of the system even after failures.

??x
Recovery processes typically involve steps such as re-execution of transactions, applying logs from backups, or using quorums to ensure that all nodes agree on the state. The recovery process is crucial for ensuring fault tolerance and consistency in distributed systems.
```java
public class RecoveryProcess {
    private List<Node> nodes; // List of all nodes

    public void recover() {
        // Reinitialize state based on a majority of log entries or backups
        nodes.forEach(node -> node.initializeState());
        
        // Ensure agreement on the new leader and state
        for (Node node : nodes) {
            if (!node.isLeaderElected()) continue;
            // Apply state changes from logs
            applyLogs(node);
        }
    }

    private void applyLogs(Node node) {
        // Logic to apply logs from backups or quorums
    }
}
```
x??

---

#### Synchronous vs. Asynchronous Replication

Background context: The passage contrasts synchronous and asynchronous replication, explaining that databases often use asynchronous replication for better performance despite the risk of losing committed data during failover.

:p What are the main differences between synchronous and asynchronous replication?
??x
Synchronous replication ensures that a write operation is acknowledged only after it has been successfully written to multiple replicas. This approach guarantees no data loss but can introduce latency and reduce overall system throughput, as writes must wait for acknowledgment from all replicas before being confirmed.

Asynchronous replication, on the other hand, allows write operations to return immediately after being written to one or more replicas. While this improves performance by reducing latency, it comes with a risk of losing committed data during failover if not managed properly.

??x
The main difference lies in how they handle write acknowledgments and data consistency:
- Synchronous replication: Write returns only after confirmation from all replicas.
- Asynchronous replication: Write acknowledges immediately but may lose data on failure.

Example code for synchronous replication (simplified):
```java
public class SynchronousReplication {
    private List<Node> nodes; // List of node replicas

    public void writeData(byte[] data) {
        int requiredReplicas = nodes.size() / 2 + 1;
        int committedReplicas = 0;

        for (Node node : nodes) {
            if (node.write(data)) {
                committedReplicas++;
                if (committedReplicas >= requiredReplicas) break;
            }
        }

        // Write returns only after required replicas acknowledge
    }
}
```
x??

---

#### Limitations of Consensus Algorithms

Background context: The passage outlines the limitations of consensus algorithms, including their synchronous nature and the requirement for a majority to operate. It also discusses issues with dynamic membership and network unreliability.

:p What are some key limitations of consensus algorithms?
??x
Key limitations of consensus algorithms include:
- **Synchronous Nature**: They often require a strict majority to operate, making them inherently synchronous.
- **Minimum Node Requirement**: To tolerate \( f \) failures, at least \( 2f + 1 \) nodes are required.
- **Dynamic Membership Challenges**: Adding or removing nodes dynamically is difficult due to the fixed set of participants.
- **Network Unreliability Sensitivity**: They can be sensitive to network issues and may experience frequent leader elections during transient failures.

??x
The limitations highlight the trade-offs between fault tolerance, performance, and complexity in consensus algorithms. These challenges make them suitable for certain critical applications but not necessarily for all distributed systems.

Example code for leader election (simplified):
```java
public class LeaderElection {
    private Node currentLeader;
    private Map<Node, Long> heartbeatTimers;

    public void startElection() {
        if (!heartbeatTimers.containsKey(currentLeader)) return; // No active leader

        long currentTime = System.currentTimeMillis();
        for (Map.Entry<Node, Long> entry : heartbeatTimers.entrySet()) {
            if (currentTime - entry.getValue() > ELECTION_TIMEOUT) {
                electNewLeader(entry.getKey());
                break;
            }
        }
    }

    private void electNewLeader(Node node) {
        // Logic to elect a new leader
        currentLeader = node;
    }
}
```
x??

---

#### Distributed Transactions and Consensus

Background context: The passage concludes by noting that consensus algorithms are critical for implementing distributed transactions, but their strict majority requirement can limit their use in less fault-tolerant systems.

:p How do consensus algorithms support distributed transactions?
??x
Consensus algorithms provide the necessary safety properties (agreement, integrity, and validity) required for distributed transactions. They enable total order broadcast, which is essential for implementing linearizable atomic operations in a fault-tolerant manner.

By ensuring that all nodes agree on the same sequence of events, consensus algorithms make it possible to achieve strong consistency across multiple nodes.

??x
Consensus algorithms support distributed transactions through mechanisms like total order broadcast. This ensures that all replicas process operations in the same order and maintain the correct state.

Example for implementing linearizable storage using total order broadcast:
```java
public class LinearizableStorage {
    private ConsensusAlgorithm consensus; // Using a consensus algorithm

    public void writeData(String key, byte[] data) {
        // Propose write operation to consensus
        consensus.proposeWriteOperation(key, data);
        
        // Await confirmation from consensus
        if (consensus.decideValue()) {
            // Apply update locally
            applyLocalUpdate(key, data);
        }
    }

    private void applyLocalUpdate(String key, byte[] data) { /* Implementation */ }
}
```
x??

---

#### Linearizable Atomic Operations
Background context explaining the concept. Linearizable atomic operations allow for a sequence of operations to appear as if they were executed atomically and linearly, even when distributed across different nodes. This is crucial for ensuring that the order of operations is consistent, which prevents conflicts and ensures correctness in a distributed environment.

:p What are linearizable atomic operations used for?
??x
Linearizable atomic operations are used to ensure that multiple operations on shared data appear to have been executed sequentially, as if they were performed by a single-threaded execution. This property is essential for building systems where the order of operations must be strictly enforced, such as distributed locks and leader election mechanisms.
```java
// Pseudocode for implementing an atomic compare-and-set operation in ZooKeeper
public class CompareAndSet {
    private int currentValue;
    private int expectedValue;
    private int newValue;

    public boolean compareAndSet(int expectedValue, int newValue) {
        if (this.currentValue == expectedValue) {
            this.currentValue = newValue;
            return true;
        }
        return false;
    }
}
```
x??

---

#### Distributed Locks and Ephemeral Nodes
Background context explaining the concept. Distributed locks are used to coordinate access to shared resources in a distributed system, ensuring that only one node can modify or read certain data at any given time. Ephemeral nodes in ZooKeeper help manage these locks by automatically releasing them when a session times out.

:p How do ephemeral nodes work in ZooKeeper?
??x
Ephemeral nodes in ZooKeeper are special types of nodes that are automatically deleted when the client's session expires or times out. This mechanism helps in managing distributed locks, as any lock held by an expired session will be released, preventing deadlocks and ensuring proper resource management.

Example:
- A node registers a lock on a specific key.
- If the node's session expires (due to network issues or crashes), the ephemeral node representing the lock is automatically removed from ZooKeeper.
```java
// Pseudocode for creating an ephemeral node in ZooKeeper
public class EphemeralNode {
    public Node createEphemeral(String path, byte[] data) {
        // Code to create an ephemeral node with a session-specific identifier
        return new Node(path, data);
    }
}
```
x??

---

#### Total Ordering of Operations
Background context explaining the concept. Total ordering of operations ensures that all operations are processed in a strict order, which is crucial for maintaining consistency and preventing conflicts in distributed systems. ZooKeeper achieves this by assigning each operation a unique transaction ID (zxid) that monotonically increases over time.

:p How does ZooKeeper ensure total ordering of operations?
??x
ZooKeeper ensures total ordering of operations by maintaining a strict sequence in which all operations are processed. Each operation is assigned a unique transaction ID (zxid), which is a monotonically increasing number. This zxid guarantees that the order of operations can be tracked and enforced, preventing conflicts and ensuring consistency.

Example:
- Client A performs an operation with zxid 10.
- Client B performs another operation with zxid 11.
- These operations are processed in the exact sequence they were initiated, regardless of network delays or node failures.

```java
// Pseudocode for assigning transaction IDs (zxids) in ZooKeeper
public class TransactionID {
    private int nextZxid;

    public synchronized int getNextTransactionID() {
        return ++nextZxid;
    }
}
```
x??

---

#### Failure Detection and Heartbeats
Background context explaining the concept. Failure detection is a critical aspect of distributed systems, as it ensures that nodes can quickly identify when another node has failed or become unresponsive. ZooKeeper uses heartbeats to maintain long-lived sessions between clients and servers, allowing for automatic session timeouts if necessary.

:p How does ZooKeeper handle failure detection?
??x
ZooKeeper handles failure detection by maintaining a heartbeat mechanism between clients and the server nodes. Clients keep their sessions active by sending periodic heartbeats to the ZooKeeper servers. If no heartbeats are received within the configured timeout period, the session is considered dead, and any ephemeral nodes held by that client will be automatically deleted.

Example:
- Client A sends a heartbeat every 2 seconds.
- Server receives the heartbeat and updates the last seen timestamp for Client A.
- After 5 seconds of no heartbeats, the server declares the session to be dead and removes all associated ephemeral nodes.

```java
// Pseudocode for managing sessions in ZooKeeper
public class SessionManager {
    private Map<String, Long> lastSeenTimestamps;

    public void heartbeat(String sessionId) {
        lastSeenTimestamps.put(sessionId, System.currentTimeMillis());
    }

    public boolean checkSessionTimeout(String sessionId) {
        if (System.currentTimeMillis() - lastSeenTimestamps.get(sessionId) > timeout) {
            return true;
        }
        return false;
    }
}
```
x??

---

#### Change Notifications
Background context explaining the concept. Change notifications allow clients to be notified of changes in ZooKeeper data, enabling them to react to updates without constantly polling for new information. This feature is particularly useful for maintaining state consistency and providing real-time feedback.

:p How do change notifications work in ZooKeeper?
??x
Change notifications in ZooKeeper work by allowing a client to watch specific nodes or paths for modifications. When the watched node changes, ZooKeeper sends a notification to the client, informing it of the update without requiring continuous polling.

Example:
- Client A watches path /znode1.
- Node /znode1 is updated by another process.
- ZooKeeper sends a notification to Client A indicating that the data has changed.

```java
// Pseudocode for implementing change notifications in ZooKeeper
public class ChangeNotification {
    public void watchNode(String path, Watcher watcher) {
        // Code to register the client as an observer of changes on the specified path
    }

    public void onChange(Watcher.Event.KeeperEvent event) {
        if (event.getType() == Watcher.Event.KeeperEvent.EventType.NodeDataChanged) {
            System.out.println("Node data changed: " + event.getPath());
        }
    }
}
```
x??

---

#### Rebalancing Partitions
Background context explaining the concept. In distributed systems, rebalancing partitions is a common task to ensure that resources are evenly distributed among nodes as new nodes join or existing nodes fail. ZooKeeper can facilitate this process by providing atomic operations and notifications for coordinated partitioning.

:p How does ZooKeeper help in rebalancing partitions?
??x
ZooKeeper helps in rebalancing partitions through the use of atomic operations, ephemeral nodes, and change notifications. By leveraging these features, applications can automatically redistribute work among nodes without manual intervention, ensuring that resources are balanced efficiently as new or failing nodes join or leave the cluster.

Example:
- Node A fails.
- ZooKeeper detects the failure and triggers a rebalancing operation.
- Other nodes take over the failed node's workload using atomic operations and notifications to coordinate the transition seamlessly.

```java
// Pseudocode for rebalancing partitions in ZooKeeper
public class PartitionRebalancer {
    public void handleNodeFailure(String failedNode) {
        // Code to notify other nodes about the failure and trigger partition redistribution
    }

    public void redistributePartitions() {
        // Use atomic operations to move data from failed node to healthy nodes
    }
}
```
x??

---

#### Service Discovery
Background context explaining the concept. Service discovery is a mechanism used in distributed systems to dynamically locate services based on their names or attributes, often without knowing their actual IP addresses beforehand. While traditional DNS can be used for this purpose, consensus-based systems like ZooKeeper provide more reliable and robust solutions.

:p How does service discovery work with ZooKeeper?
??x
Service discovery with ZooKeeper works by registering the network endpoints of services as ephemeral nodes when they start up. Other services can then query ZooKeeper to find out which IP addresses correspond to specific services, making it easier to establish connections dynamically without hardcoding addresses.

Example:
- Service X starts and registers its endpoint at /serviceX.
- Another service Y queries ZooKeeper for the endpoints of service X by watching path /serviceX.

```java
// Pseudocode for service discovery in ZooKeeper
public class ServiceDiscovery {
    public void registerService(String serviceName, String ipAddress) {
        // Code to create an ephemeral node representing the service's endpoint
    }

    public List<String> discoverServices(String serviceName) {
        // Code to watch and return a list of endpoints for the specified service name
        return new ArrayList<>();
    }
}
```
x??

---

#### Membership Services
Background context explaining the concept. Membership services are crucial for maintaining information about which nodes are part of a distributed system, including their current state and roles. ZooKeeper has been used extensively in this role due to its ability to manage node membership, failure detection, and coordination.

:p What is the significance of membership services in distributed systems?
??x
Membership services are significant in distributed systems because they provide critical information about which nodes are part of a cluster, their current state, and roles. This information is essential for maintaining system integrity, ensuring that only authorized nodes can participate, and facilitating failover mechanisms.

Example:
- ZooKeeper maintains a list of active members in the form of ephemeral nodes.
- When a node fails or joins, this change is tracked by ZooKeeper and used to update membership information.

```java
// Pseudocode for managing membership in ZooKeeper
public class MembershipManager {
    private Set<String> activeMembers;

    public void addMember(String nodeId) {
        // Code to add a new member to the active list of nodes
    }

    public void removeMember(String nodeId) {
        // Code to remove a failed or leaving node from the active list
    }
}
```
x??

#### Membership Service and Consensus
Background context: The membership service is crucial for determining which nodes are active members of a cluster. Due to unbounded network delays, it's challenging to reliably detect node failures. However, combining failure detection with consensus allows nodes to agree on the current state of their membership.

:p What is the purpose of coupling failure detection with consensus in a distributed system?
??x
To enable nodes to collectively decide which members are considered alive or dead, despite network delays and unreliable communication.
x??

---

#### Linearizability
Background context: Linearizability is a consistency model where operations appear as if they were executed one after another on a single copy of the data. This makes replicated data seem atomic, much like a variable in a single-threaded program.

:p What is linearizability and why is it useful?
??x
Linearizability ensures that operations on replicated data are consistent and behave atomically as if there was only a single copy of the data. It simplifies understanding and debugging because it abstracts away the complexity of multiple replicas.
x??

---

#### Causality
Background context: Unlike linearizability, which orders all operations in one timeline, causality allows for concurrent operations by providing an ordering based on cause and effect.

:p What does causality provide that linearizability doesn't?
??x
Causality offers a weaker consistency model where some things can be concurrent. It provides a version history like a branching timeline with merging branches, reducing coordination overhead compared to linearizability.
x??

---

#### Consensus Problems
Background context: Achieving consensus is about making all nodes agree on what was decided and ensuring that decisions are irrevocable. Various problems, including ensuring unique usernames in concurrent registration scenarios, can be reduced to the problem of consensus.

:p What does achieving consensus solve in a distributed system?
??x
Achieving consensus ensures that all nodes in a distributed system agree on a decision and make it irrevocable, which is crucial for operations like leader election or ensuring uniqueness across multiple nodes.
x??

---

#### Linearizable Compare-and-Set Registers
Background context: A linearizable compare-and-set (CAS) register atomically decides whether to set its value based on the current value. This operation needs to be consistent and atomic.

:p What does a linearizable compare-and-set register do?
??x
A linearizable compare-and-set register atomically checks if the current value matches a given parameter, and if so, sets it to a new value, ensuring consistency as if operating on a single copy of data.
x??

---

#### Timestamp Ordering is Not Sufficient
Background context: Using timestamps alone may not suffice for certain operations, such as ensuring unique usernames during concurrent registration. The problem arises because nodes might have different views of the current state.

:p Why can't timestamp ordering alone ensure uniqueness in concurrent registrations?
??x
Timestamps do not inherently prevent concurrent modifications. If two nodes independently decide to register the same username at nearly the same time, their timestamps might be similar or identical, leading to potential conflicts without additional coordination mechanisms.
x??

--- 

#### Causality and Timestamps
Background context: While timestamps provide a basic ordering of events, causality can handle more complex scenarios where operations are concurrent but still need to respect cause and effect.

:p How does causality help in resolving concurrency issues that timestamps alone cannot?
??x
Causality provides a way to order events based on their causal relationships, allowing for concurrent operations while respecting the overall flow of events. This is useful when timestamps might not provide enough information about the sequence due to clock skew or other delays.
x??

--- 

#### Summary of Consistency Models
Background context: The text explores different consistency models like linearizability and causality, highlighting their strengths and weaknesses.

:p What are some key differences between linearizable and causal consistency?
??x
Linearizability requires operations to be atomic and appear in a single, totally ordered timeline, while causality allows for concurrent operations but ensures events are ordered based on cause and effect. Linearizability is easier to understand but less flexible with network issues, whereas causality is more robust and better handles concurrency.
x??

#### Atomic Transaction Commit
Background context: A database must decide whether to commit or abort a distributed transaction. This decision is crucial for ensuring data consistency and integrity.

:p What decision does a database need to make regarding atomic transactions?
??x
The database needs to determine whether to commit or abort a distributed transaction.
x??

---

#### Total Order Broadcast
Background context: The messaging system must decide on the order in which to deliver messages. Ensuring a total order of message delivery is essential for maintaining consistency and order among nodes.

:p What does the messaging system need to ensure when delivering messages?
??x
The messaging system needs to ensure that messages are delivered in a specific, ordered sequence.
x??

---

#### Locks and Leases
Background context: When several clients race to acquire a lock or lease, only one can succeed. The system must decide which client gets the lock.

:p How does the system determine which client successfully acquires a lock?
??x
The system decides which client is granted the lock based on predefined rules or mechanisms.
x??

---

#### Membership/Coordination Service
Background context: Given failure detectors, such as timeouts, the system needs to decide which nodes are alive and which should be considered dead due to session timeouts.

:p What does the membership service need to determine?
??x
The membership service needs to identify which nodes are currently alive and which have timed out.
x??

---

#### Uniqueness Constraint
Background context: Concurrent transactions may try to create records with the same key. The system must decide which transaction should succeed and which should fail due to a uniqueness constraint.

:p How does the system handle concurrent transactions trying to insert conflicting records?
??x
The system decides which transaction succeeds by ensuring no two records have the same key, enforcing constraints.
x??

---

#### Single-Leader Database
Background context: In a single-leader database, all decision-making power is vested in one node (the leader). This setup provides linearizability and consistency but introduces failover challenges.

:p What does a single-leader database rely on for making decisions?
??x
A single-leader database relies on the leader node to make critical decisions such as transaction commits, message ordering, lock acquisition, leadership, and uniqueness constraints.
x??

---

#### Consensus in Single-Leader Database
Background context: A single-leader approach can handle decision-making but faces issues if the leader fails or becomes unreachable. Three approaches are discussed for handling this situation.

:p What are the three ways to handle a failed leader in a single-leader database?
??x
1. Wait for the leader to recover.
2. Manually fail over by choosing a new leader.
3. Use an algorithm to automatically choose a new leader.
x??

---

#### Fault-Tolerant Consensus Algorithms
Background context: Even with a leader, consensus algorithms are still required for maintaining leadership and handling leadership changes. Tools like ZooKeeper can provide outsourced consensus, failure detection, and membership services.

:p Why is consensus still necessary in single-leader databases?
??x
Consensus is still necessary because it ensures that the system can handle leader failures or network interruptions by automatically selecting a new leader.
x??

---

#### ZooKeeper Usage for Fault-Tolerance
Background context explaining when and why ZooKeeper is used. ZooKeeper is a service that helps applications coordinate with each other reliably, often used to manage distributed systems where consensus is required.

If your application needs fault-tolerant coordination among nodes, especially in a distributed system, using ZooKeeper can be very beneficial. It provides features like leader election, configuration management, and centralized logging which help achieve high availability and consistency.

:p When should you use ZooKeeper for your application?
??x
ZooKeeper is advisable when your application requires fault-tolerant coordination among nodes, especially in a distributed system where consensus is needed.
x??

---

#### Leaderless Systems vs. Global Consensus
Explanation of systems that do not require global consensus and how they handle conflicts.

In leaderless replication or multi-leader replication systems, the absence of a single leader can simplify conflict resolution but may also lead to non-linearizable data updates. These systems rely on local decision-making and might use techniques like vector clocks or distributed transactions to manage conflicts without requiring full consensus.

:p Can every system benefit from using global consensus?
??x
Not every system benefits from using global consensus. Leaderless and multi-leader replication systems often do not require it because they can handle conflicts through local decision-making, avoiding the need for linearizable updates across all nodes.
x??

---

#### Theoretical Foundations of Distributed Systems
Explanation on how theoretical papers inform practical work in distributed systems.

Theoretical research provides foundational knowledge about what is achievable and what isn't in distributed systems. These studies often explore edge cases and limitations that real-world implementations must consider, making them invaluable for designing robust distributed applications.

:p Why are theoretical papers important in the field of distributed systems?
??x
Theoretical papers are crucial because they help us understand the limits and possibilities within distributed systems. They inform practical work by delineating what is theoretically possible and what isn't, guiding the design of reliable and efficient distributed systems.
x??

---

#### Part II Summary - Replication, Partitioning, Transactions, Failure Models, Consistency
Summary of the topics covered in Part II of the book.

In Part II, the book covers a comprehensive range of topics including replication strategies, partitioning techniques, transaction management, failure models, and consistency models. These concepts form the theoretical foundation needed to build reliable distributed systems.

:p What are the main topics covered in Part II?
??x
Part II covers replication (Chapter 5), partitioning (Chapter 6), transactions (Chapter 7), distributed system failure models (Chapter 8), and finally consistency and consensus (Chapter 9).
x??

---

#### Practical Building Blocks for Distributed Systems
Explanation of how to build powerful applications using heterogeneous building blocks.

After establishing a strong theoretical foundation, the next step is to apply this knowledge practically. This involves understanding how to integrate various components or "building blocks" into a cohesive system that can handle distributed tasks efficiently and reliably.

:p What does Part III focus on?
??x
Part III focuses on practical systems by discussing how to build powerful applications from heterogeneous building blocks.
x??

---

#### References for Further Reading
List of key references provided in the text, highlighting their importance.

The book references several key papers that are essential for understanding distributed systems. These include articles on eventual consistency, distributed transaction management, and the theoretical foundations of consensus algorithms. Exploring these resources can provide deeper insights into various aspects of distributed system design and operation.

:p What additional reading is recommended?
??x
Additional reading is highly recommended to gain a deeper understanding of key concepts in distributed systems. References such as "Eventual Consistency Today" by Peter Bailis and Ali Ghodsi, "Consistency, Availability, and Convergence" by Prince Mahajan et al., and papers on linearizability are particularly valuable.
x??

---

