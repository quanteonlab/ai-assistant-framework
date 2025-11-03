# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 27)

**Starting Chapter:** Relying on Linearizability

---

#### Linearizability and its Importance in Distributed Systems

In distributed systems, ensuring that operations appear to users as though they were executed sequentially on a single machine is crucial. This property is known as linearizability.

Background context: In distributed databases like ZooKeeper and etcd, writes are linearizable by default, meaning the sequence of operations appears as if they happened one after another on a single server. However, reads can be stale because replicas might serve different versions due to eventual consistency models.

:p What circumstances make linearizability particularly useful in distributed systems?
??x
Linearizability is crucial for scenarios where strict order of operations must be maintained and conflicts avoided. Examples include:
- **Locking and leader election**: Ensures that only one node acts as the leader at any time, preventing split-brain scenarios.
- **Uniqueness constraints**: Guarantees that data like usernames or database records are unique across all nodes.
- **Cross-channel timing dependencies**: Ensures that actions in different parts of the system (e.g., user input and background processing) occur in a consistent order.

x??

---

#### Example of Uniqueness Constraints

Unique constraints, such as ensuring no two users have the same username or preventing negative account balances, require linearizable storage to maintain consistency across all nodes.

Background context: In distributed databases, enforcing unique constraints involves operations that must be treated atomically and consistently. This often requires linearizability to ensure that changes are applied in a strict order and observed uniformly by all nodes.

:p How can you enforce uniqueness constraints using linearizable storage?
??x
To enforce uniqueness constraints like ensuring no two users have the same username, you can use an atomic compare-and-set operation. When a user tries to register with a username, the system checks if the username is already taken and sets it only if not.

```java
public boolean createUser(String username) {
    // Check if the username exists
    boolean usernameExists = checkUsername(username);
    
    // Use a linearizable storage service (like ZooKeeper or etcd)
    StorageService storage = new StorageService();
    
    // Attempt to set the username atomically
    return storage.compareAndSet(username, "user_id");
}
```

x??

---

#### Linearizability in Distributed Locking

Distributed systems often use distributed locks for leader election and coordination tasks. These locks must be linearizable to ensure that all nodes agree on which node holds the lock.

Background context: A distributed lock allows multiple nodes to coordinate access to shared resources by ensuring only one node can acquire the lock at a time. This requires linearizability to prevent conflicts and ensure correctness.

:p Why is linearizability important for implementing distributed locks?
??x
Linearizability is essential because it ensures that all nodes in a distributed system agree on which node holds the lock. Without this guarantee, multiple nodes might attempt to access the same resource simultaneously, leading to inconsistent states or data corruption.

```java
public class DistributedLock {
    private final String lockKey;
    private volatile boolean locked = false;

    public DistributedLock(String lockKey) {
        this.lockKey = lockKey;
    }

    public void acquire() throws InterruptedException {
        StorageService storage = new StorageService();
        
        // Attempt to acquire the lock atomically
        while (!storage.compareAndSet(lockKey, "locked")) {
            Thread.sleep(10); // Avoid busy waiting by sleeping briefly
        }
    }

    public boolean release() {
        StorageService storage = new StorageService();
        
        // Release the lock if it is held by this node
        return storage.compareAndSet(lockKey, "");
    }
}
```

x??

---

#### Ensuring Consistency in Distributed Systems

In distributed systems with eventual consistency, reads may be stale. However, linearizable operations are necessary for ensuring that critical operations like constraints and leader election behave correctly.

Background context: While eventual consistency models allow for some degree of staleness in read operations, linearizable writes ensure that critical state changes are applied consistently across all nodes. This is essential for maintaining the integrity and correctness of the system.

:p In what scenarios would you need to rely on linearizability?
??x
Linearizability is crucial in scenarios where strict order of operations must be maintained:
- **Locking and leader election**: Ensuring that only one node acts as the leader.
- **Uniqueness constraints**: Preventing conflicts like duplicate usernames or negative account balances.

```java
public class ConstraintEnforcer {
    private final StorageService storage;

    public ConstraintEnforcer(StorageService storage) {
        this.storage = storage;
    }

    public void enforceUniqueConstraint(String key, String value) {
        // Check if the value is already in use
        boolean exists = storage.exists(key, value);
        
        // Use a linearizable operation to set the constraint
        if (!exists && storage.set(key, value)) {
            System.out.println("Constraint enforced successfully.");
        } else {
            throw new ConstraintViolationException("Unique constraint violated");
        }
    }
}
```

x??

---

#### Linearizability and Race Conditions

Background context: The concept of linearizability is crucial for ensuring that a system behaves as though it has only one copy of data, with all operations being atomic. This ensures consistency, particularly when dealing with multiple nodes or communication channels.

:p What is the risk when file storage service is not linearizable in a web server and image resizer setup?

??x
The risk involves race conditions where the message queue (step 3 and step 4) might be faster than internal replication inside the storage service. When the resizer fetches the image (step 5), it could see an old version or nothing, leading to inconsistent full-size and resized images in file storage.

For example:
- Suppose a new image is uploaded.
- The message queue processes this event quickly.
- Before the image storage service updates its copy, another process fetches the image from the old state of the storage.

In such cases, if the resizer processes an old version of the image, it results in inconsistent data. This inconsistency arises because there are two communication channels: file storage and message queue, without a recency guarantee provided by linearizability.

??x
The solution involves ensuring that either the write is acknowledged on both channels or implementing mechanisms to ensure one channel does not proceed before the other. However, this adds complexity.
x??

---

#### Single-Leader Replication (Potentially Linearizable)

Background context: In a single-leader replication setup, writes are handled by a leader node, while followers maintain backup copies of data. Reads can be performed from the leader or synchronous followers.

:p How can reads from a leader or synchronously updated followers potentially lead to linearizability?

??x
Reads from the leader or synchronously updated followers have the potential to be linearizable because they operate on a single copy of the data, and all operations are atomic. This means that if you read from the current state of the leader, it behaves as though there is only one copy of the data.

For instance:
- If a write operation completes successfully in the leader node, subsequent reads will reflect this change immediately or after replication to followers.
```java
// Pseudocode for a single-leader system
class Leader {
    private Map<String, String> data = new ConcurrentHashMap<>();

    public void write(String key, String value) {
        // Write operation is atomic and committed in the leader node first
        data.put(key, value);
    }

    public String read(String key) {
        // Read directly from the leader node or a follower that has caught up
        return data.get(key);
    }
}
```
x??

---

#### Multi-Leader Replication (Not Linearizable)

Background context: In multi-leader replication, writes can occur concurrently on multiple nodes. These writes are asynchronously replicated to other nodes, leading to potential conflicts and inconsistent data.

:p Why is multi-leader replication generally not linearizable?

??x
Multi-leader replication typically results in non-linearizability because it allows concurrent write operations across different nodes. Since these nodes do not have a single authoritative copy of the data, they can produce conflicting writes that require resolution. This lack of coordination among leaders leads to inconsistent state.

For example:
- Node A and Node B both attempt to update the same key simultaneously.
- Both updates might succeed independently without conflict detection.
```java
// Pseudocode for multi-leader replication scenario
class MultiLeader {
    private Map<String, String> data = new ConcurrentHashMap<>();

    public void write(String key, String value) {
        // Write operation is asynchronous and may not be applied immediately on all nodes
        data.put(key, value);
    }

    public String read(String key) {
        // Reads might return conflicting values if writes were not propagated correctly
        return data.get(key);
    }
}
```
x??

---

#### Consensus Algorithms (Linearizable)

Background context: Some consensus algorithms can implement linearizable storage safely due to their design, which prevents split brain and stale replicas. These algorithms ensure that all nodes agree on the state of the system.

:p How do consensus algorithms like ZooKeeper and etcd provide linearizability?

??x
Consensus algorithms such as those used by ZooKeeper and etcd can provide linearizable storage because they include mechanisms to prevent split brain and stale replicas. These algorithms ensure that operations are atomic, consistent, isolated, and durable (ACID properties), effectively mimicking a single-node system across multiple nodes.

For example:
- In ZooKeeper's implementation of consensus, every operation is proposed by a leader and accepted by the majority of followers.
```java
// Pseudocode for ZooKeeper consensus process
class ZKConsensus {
    private Map<String, String> data = new ConcurrentHashMap<>();

    public void write(String key, String value) {
        // Leader proposes a change, and followers agree on it
        leader.propose(key, value);
        data.put(key, value);
    }

    public String read(String key) {
        // Reads reflect the current state agreed upon by all nodes
        return data.get(key);
    }
}
```
x??

---

#### Linearizability and Quorums

Background context: Even with strict quorum reads and writes in a Dynamo-style model, linearizability can still be violated due to variable network delays. The quorum conditions might not guarantee the order of operations if nodes experience different network latencies.

:p What does Figure 9-6 illustrate about linearizable executions?

??x
Figure 9-6 illustrates a scenario where a system with strict quorum reads and writes is not necessarily linearizable due to variable network delays. Even though the conditions for quorums are met (w + r > n), operations can still execute in a non-linearizable manner if requests are processed out of order.

For example:
- Node A reads from two nodes, sees the new value 1.
- Concurrently, Node B reads from different nodes and gets back old values 0.
```java
// Pseudocode for demonstrating non-linearizable behavior with quorums
class DynamoModel {
    private Map<String, String> data = new ConcurrentHashMap<>();

    public void write(String key, String value) {
        // Leader proposes a change, and followers agree on it
        leader.propose(key, value);
        data.put(key, value);
    }

    public String read(String key) {
        // Reads might return conflicting values due to variable network delays
        if (checkQuorum()) {
            return data.get(key);
        } else {
            return "Not linearizable";
        }
    }

    private boolean checkQuorum() {
        // Implementation for checking quorum conditions
        return true; // Simplified example
    }
}
```
x??

---
#### Dynamo-Style Quorum Linearizability
In systems like Dynamo, which use a leaderless approach with multiple replicas, it is possible to achieve linearizability but at the cost of reduced performance. This can be done by performing read repair synchronously and ensuring that writers also read the latest state from a quorum before writing.
:p What is required for achieving linearizability in systems like Dynamo?
??x
To achieve linearizability in Dynamo-style systems, both readers and writers need to perform additional steps:
- **Readers** must perform read repair synchronously before returning results. This involves checking the consistency of data across a quorum of nodes.
- **Writers** must read the latest state from a quorum of nodes before sending their writes. This ensures that writes are based on the most recent information available.

For example, in Cassandra, synchronous read repair is applied during quorum reads:
```java
// Pseudocode for Cassandra's quorum read and write process
public class CassandraNode {
    public void performQuorumRead(String key) {
        // Read from a quorum of nodes, check consistency, then return results.
    }
    
    public void performQuorumWrite(String key, String value) {
        // Read the latest state from a quorum of nodes before writing.
        // Perform write and ensure it is consistent with the quorum.
    }
}
```
x??

---
#### Riak vs. Cassandra on Linearizability
Riak does not perform synchronous read repair due to performance penalties, while Cassandra waits for read repair to complete during quorum reads but loses linearizability in scenarios of multiple concurrent writes using last-write-wins conflict resolution. This means that only linearizable read and write operations can be implemented this way; a compare-and-set operation cannot because it requires a consensus algorithm.
:p How do Riak and Cassandra handle linearizability differently?
??x
- **Riak**: Does not perform synchronous read repair, which avoids performance penalties but sacrifices linearizability in certain scenarios.
- **Cassandra**: Waits for read repair to complete during quorum reads. This ensures consistency but can lead to loss of linearizability if multiple concurrent writes occur and follow the last-write-wins policy.

For example:
```java
// Pseudocode illustrating Cassandra's approach
public class CassandraNode {
    public void performQuorumRead(String key) {
        // Perform read repair synchronously.
        // Ensure data consistency before returning results.
    }
    
    public void performWrite(String key, String value) {
        // Read the latest state from a quorum of nodes.
        // Apply write operation and ensure it is consistent with the quorum.
    }
}
```
x??

---
#### Multi-Leader vs. Single-Leader Replication
Multi-leader replication allows each datacenter to continue operating normally during network interruptions, as writes are queued up and exchanged when connectivity is restored. In contrast, single-leader replication requires all read and write requests to be sent synchronously over the network to the leader in case of a network interruption between datacenters.
:p What are the differences in handling network interruptions between multi-leader and single-leader replication?
??x
- **Multi-Leader Replication**: Each datacenter can operate independently. Writes from one datacenter are asynchronously replicated to others, allowing operations to continue even if network connectivity is interrupted.

```java
// Pseudocode for Multi-Leader Replication
public class MultiLeaderDatacenter {
    public void handleWrite(String key, String value) {
        // Write locally and queue the write operation.
        // Exchange queued writes when network connection is restored.
    }
    
    public void handleRead(String key) {
        // Read from local storage or replica if available.
    }
}
```

- **Single-Leader Replication**: The leader must be in one datacenter. Reads and writes need to be directed synchronously to the leader, causing a halt in operations for followers when network connectivity is lost.

```java
// Pseudocode for Single-Leader Replication
public class SingleLeaderDatacenter {
    public void handleWrite(String key, String value) {
        // Send write request to leader.
        // Ensure leader processes and returns confirmation.
    }
    
    public void handleRead(String key) {
        // Send read request to leader.
        // Await response from leader before returning results.
    }
}
```
x??

---

#### Linearizability and Network Interruptions
Background context: The provided text discusses how linearizable reads and writes can be problematic when network interruptions occur. Specifically, if an application requires linearizability, a network partition can cause some replicas to become unavailable as they cannot contact the leader or other replicas.

:p What is the issue with linearizable databases in the presence of network partitions?
??x
Network partitions can lead to unavailability for some replicas that are disconnected from others. If a replica cannot communicate with the leader or other replicas, it must wait until the network problem is resolved or return an error, making it unavailable during this time.
x??

---

#### Multi-Leader Replication and Availability
Background context: The text explains how if an application does not require linearizability, it can be designed in a way that each replica processes requests independently even when disconnected from other replicas. This approach ensures availability in the face of network problems.

:p How can applications remain available during network partitions without sacrificing linearizability?
??x
Applications can use multi-leader replication where each replica can process requests independently. In this setup, even if a replica is disconnected from others due to a network partition, it can still handle requests locally and maintain availability.
x??

---

#### The CAP Theorem
Background context: The CAP theorem discusses the trade-offs between consistency, availability, and partition tolerance in distributed systems. Eric Brewer proposed this theorem in 2000, but the concept has roots in earlier database design principles.

:p What does the CAP theorem state?
??x
The CAP theorem states that in a distributed system, you can have at most two out of the three guarantees: Consistency (C), Availability (A), and Partition Tolerance (P). You cannot simultaneously achieve all three.
x??

---

#### Network Faults vs. Partitions
Background context: The text differentiates between network faults and partitions. Network faults are broader, including various types of failures, whereas a partition is specifically a network issue where the system can be split into multiple isolated parts.

:p How does the CAP theorem define a partition?
??x
In the context of the CAP theorem, a partition refers to a situation in a distributed system where communication between nodes is disrupted. This disruption can lead to data silos and unavailability, making it impossible for some nodes to communicate with others.
x??

---

#### Sharding (Data Partitioning)
Background context: The text mentions sharding as a method of deliberately breaking down large datasets into smaller parts to manage them more efficiently. Sharding is distinct from network partitions in that it is a deliberate design choice.

:p What is sharding, and how does it differ from network partitions?
??x
Sharding involves dividing a large dataset into smaller chunks or shards. Each shard can be managed by a separate node, which simplifies data management and scaling. Network partitions are a type of fault where nodes become isolated due to network issues, whereas sharding is a proactive method of managing data.
x??

---

#### Impact on Database Design
Background context: The CAP theorem influenced the design space for distributed databases, encouraging engineers to consider multi-leader replication and other non-linearizable models that ensure availability even during partitions.

:p How did the CAP theorem influence database design?
??x
The CAP theorem prompted database designers to explore a wider range of distributed systems that prioritize availability over linearizability in the presence of network faults. This led to the development of NoSQL databases, which often implement multi-leader replication and other strategies to ensure high availability.
x??

---

#### CAP Theorem Overview
Background context: The CAP theorem states that a distributed system can only provide two of the following three guarantees at most: Consistency (all nodes see the same data at the same time), Availability (every request receives a response about whether it was successful or not), and Partition Tolerance (the system continues to operate despite arbitrary message loss or failure). The theorem highlights that in partitioned networks, achieving all three guarantees simultaneously is impossible.
:p What does the CAP theorem state?
??x
The CAP theorem asserts that in distributed systems, you can achieve at most two out of the three guarantees: Consistency, Availability, and Partition Tolerance. This means when a network fault occurs, you have to choose between either ensuring consistency or availability. 
x??

---

#### Linearizability vs. Performance Trade-offs
Background context: Linearizability is a strong form of consistency that requires operations to appear as if they occur atomically in some total order. However, achieving linearizability comes with significant performance overhead due to the need for coordination across nodes.
:p Why do systems often not provide linearizable guarantees?
??x
Systems avoid providing linearizable guarantees primarily because it incurs high performance costs. Linearizability requires that all operations appear as if they were executed atomically and in a global order, which necessitates additional synchronization mechanisms. These mechanisms can lead to increased latency and reduced throughput.
x??

---

#### Multi-core Memory Consistency Models
Background context: Modern CPUs use caches to improve performance by allowing local writes to be faster. However, this introduces inconsistencies between cores due to asynchronous memory updates. Linearizability is often impractical in such systems because of these inherent delays.
:p Why is linearizability not feasible on multi-core systems?
??x
Linearizability is not practical on multi-core systems due to the nature of modern CPUs using caches and store buffers. Writes by one core may appear out of order or not visible to other cores until synchronization mechanisms are used, which can significantly degrade performance.
x??

---

#### Network Delays and Linearizability
Background context: The CAP theorem focuses on network partitions but doesn't fully address all types of network delays. Linearizable systems require responses to be immediate, which is often impractical due to variable network delays.
:p Why do linearizable systems struggle with network delays?
??x
Linearizable systems struggle with network delays because the guarantee requires that every operation appears as if it were executed atomically in a global order. This means there must be no delay between operations, but in reality, networks often have variable and unbounded delays which cannot be fully guaranteed.
x??

---

#### Consistency Models and Performance Trade-offs
Background context: While linearizability is a strong guarantee of consistency, many distributed systems opt for weaker consistency models to improve performance. These models sacrifice some level of consistency to achieve lower latency.
:p Why do some distributed systems prefer weaker consistency models?
??x
Distributed systems prefer weaker consistency models because they can offer better performance by reducing the overhead associated with maintaining linearizability. Weaker models allow operations to be less strictly ordered, enabling faster response times which are crucial for latency-sensitive applications.
x??

---

#### Theoretical Limits of Linearizability
Background context: Atiya and Welch’s theorem proves that achieving linearizable consistency requires a response time proportional to network delay uncertainty. This makes linearizability impractical in networks with highly variable delays.
:p According to Atiya and Welch, what is the minimum response time for linearizable reads and writes?
??x
According to Atiya and Welch's theorem, the response time for linearizable reads and writes must be at least proportional to the uncertainty of network delay. In networks with highly variable delays, this means that the response time will inevitably be high.
x??

---

#### Avoiding Linearizability Without Sacrificing Correctness
Background context: Even though linearizability is theoretically important, many systems choose to avoid it for performance reasons. This doesn't mean correctness is compromised; alternative approaches can maintain consistency guarantees without the overhead of linearizability.
:p How can distributed systems achieve high availability and partition tolerance while avoiding linearizability?
??x
Distributed systems can achieve high availability and partition tolerance by using weaker consistency models that still ensure correctness but with lower performance overhead. This involves designing algorithms where operations appear to be executed in some order, but not necessarily the strict global order required for linearizability.
x??

---

