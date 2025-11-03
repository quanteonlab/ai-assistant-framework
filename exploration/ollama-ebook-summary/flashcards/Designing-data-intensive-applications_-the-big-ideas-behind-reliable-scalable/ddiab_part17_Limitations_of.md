# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 17)

**Starting Chapter:** Limitations of Quorum Consistency

---

---
#### Quorum Consistency Overview
Quorum consistency ensures data reliability by requiring a minimum number of nodes to agree on write and read operations. If fewer than required w or r nodes are available, writes or reads will return an error due to unavailability. Nodes can be unavailable for various reasons such as hardware failures, network issues, or operational errors.
:p What is quorum consistency used for?
??x
Quorum consistency ensures that the data written and read operations are handled by a sufficient number of nodes to maintain reliability. It helps in handling node failures by ensuring overlapping sets of nodes for writes and reads.
x??

---
#### Setting w and r Values
In quorum consistency, choosing appropriate values for w (write quorum) and r (read quorum) is crucial. Often, both are set as a majority (more than n/2) to tolerate up to n/2 node failures while ensuring that the read operation will return an updated value.
:p How do you choose w and r in a distributed system?
??x
To choose w and r effectively, one must consider the number of replicas (n). Typically, both are set as more than n/2. For example, if there are 5 nodes, setting both w and r to 3 ensures that even with up to two node failures, writes and reads will still succeed.
x??

---
#### Quorum Conditions
Quorum conditions ensure overlap between the sets of nodes used for write and read operations. This guarantees that at least one node in the read set has the latest value after a write operation.
:p What does it mean for w and r to satisfy the quorum condition?
??x
The quorum condition is satisfied if the sum of w (write quorum) and r (read quorum) is greater than n (total number of replicas). This ensures that there will be an overlap between the nodes used for writes and reads, guaranteeing that at least one node in the read set has the latest value.
x??

---
#### Flexibility with Quorum Assignments
Quorum assignments can vary beyond majorities. While a majority is common to tolerate up to n/2 failures, other configurations are possible where w + r ≤ n. This configuration allows lower latency and higher availability but increases the risk of reading stale values.
:p How does setting w + r ≤ n affect read and write operations?
??x
Setting w + r ≤ n means that reads and writes still go to all n nodes, but fewer successful responses are needed for an operation to succeed. This configuration reduces latency and improves availability by tolerating more network disruptions. However, it increases the risk of stale data being returned.
x??

---
#### Edge Cases with Quorum Consistency
Edge cases can arise where stale values are read even when w + r > n. These include scenarios like sloppy quorums (where writes may go to different nodes than reads), concurrent writes leading to ambiguity, and network interruptions affecting the quorum condition.
:p What are common edge cases in quorum consistency?
??x
Common edge cases in quorum consistency include:
- Sloppy quorums where writes might not overlap with reads.
- Concurrent write scenarios that can lead to ambiguity about which value is latest.
- Network disruptions leading to a drop below the required w or r nodes for successful read and write operations.
x??

---
#### Leaderless Replication
In leaderless replication, there’s no single node responsible for coordinating writes. This increases flexibility but complicates monitoring and maintaining quorum conditions due to the lack of a fixed order in which writes are applied.
:p What is a key challenge with leaderless replication?
??x
A key challenge with leaderless replication is ensuring consistent application of writes without a central coordinator, making it harder to monitor and maintain quorum conditions. Without a leader, the order in which writes are applied can vary across nodes, complicating the monitoring process.
x??

---
#### Monitoring Staleness
Monitoring staleness involves tracking whether databases return up-to-date results. Even if an application can tolerate some staleness, understanding replication health is crucial to prevent significant delays or failures.
:p How do you monitor staleness in distributed systems?
??x
Monitoring staleness typically involves checking the replication lag using metrics exposed by the database. For leader-based replication, this is straightforward as writes are applied in order and each node has a position in the replication log. However, in leaderless systems, monitoring becomes more complex due to varying write application orders.
x??

---

#### Read Repair and Staleness
Background context explaining read repair, including how it handles stale values. Note that anti-entropy is not used; values can become very old if infrequently read.

:p What is read repair, and why might a value be returned as stale?
??x
Read repair is a technique where a database automatically updates replicas of data to ensure consistency. If a replica returns a stale value due to infrequent reads, the system may update other replicas to reflect the current value, thus preventing staleness in future requests. However, without anti-entropy mechanisms, values can become very old if they are only read rarely.

```java
// Pseudocode for Read Repair Logic
public void readRepair(DataValue requestedData) {
    // Find all stale replicas and update them with the latest data value.
    List<Replica> staleReplicas = findStaleReplicas(requestedData);
    for (Replica replica : staleReplicas) {
        replica.updateWith(requestedData.latestVersion());
    }
}
```
x??

---

#### Eventual Consistency
Background context explaining eventual consistency and its importance in quantifying the term "eventually."

:p How does eventual consistency handle the guarantee of data being consistent over time?
??x
Eventual consistency is a design principle where changes to data eventually propagate across all replicas, but there are no guarantees on how long it takes for this to happen. This means that data may be inconsistent in some replicas temporarily. Quantifying "eventually" is crucial for operability and can impact system behavior.

```java
// Pseudocode for Eventual Consistency Logic
public void applyEventualConsistency(DataUpdate update) {
    // Propagate the update across all nodes, but with no guarantee on timeliness.
    List<Node> nodes = getAllNodes();
    for (Node node : nodes) {
        node.updateWith(update);
    }
}
```
x??

---

#### Sloppy Quorums and Hinted Handoff
Background context explaining how quorums work and the challenges of network interruptions, leading to the need for sloppy quorums and hinted handoffs.

:p How do sloppy quorums and hinted handoff help in maintaining write availability during network disruptions?
??x
Sloppy quorums allow writes even if fewer than w or r nodes are reachable by accepting writes from any w nodes. Hinted handoff is used when a client loses connectivity to its primary replicas, temporarily writing to other reachable nodes until the network stabilizes and can route requests back to the primary nodes.

```java
// Pseudocode for Sloppy Quorum Write Logic
public void writeSloppily(DataValue data) {
    List<Node> reachableNodes = findReachableNodes();
    if (reachableNodes.size() >= w) {
        for (Node node : reachableNodes) {
            node.write(data);
        }
    } else {
        // Hint the handoff to a different set of nodes.
        hintedHandoff(data, reachableNodes.get(0));
    }
}

// Pseudocode for Hinted Handoff Logic
public void hintedHandoff(DataValue data, Node targetNode) {
    targetNode.writeWithHint(data);
}
```
x??

---

---
#### Sloppy Quorum Concept
Background context: In Dynamo-style databases, a sloppy quorum is an optional feature that ensures durability but not necessarily immediate read consistency. This means data is stored on `w` nodes somewhere, and there's no guarantee a read from `r` nodes will see the latest version until hinted handoff completes.
:p What does a sloppy quorum ensure in Dynamo-style databases?
??x
A sloppy quorum ensures that data is written to at least `w` nodes but doesn't guarantee immediate read consistency. It provides durability and eventual consistency, where reads may not see the latest updates until hinted handoff has completed.
x??

---
#### Multi-Datacenter Operation in Leaderless Replication
Background context: Multi-datacenter operation in leaderless replication allows for distributed writes and reads across multiple datacenters without a single leader. This model is suitable for handling network interruptions, latency spikes, and conflicting concurrent writes by ensuring nodes can operate independently within their local clusters.
:p How does Cassandra and Voldemort handle multi-datacenter support?
??x
Cassandra and Voldemort implement multi-datacenter support in the leaderless model where:
- The number of replicas `n` includes nodes from all datacenters.
- Each write is sent to all replicas, but clients wait for acknowledgment only from a quorum within their local datacenter.
- Writes to other datacenters are often configured asynchronously, reducing latency impact on local operations.

For example:
```java
// Pseudocode in Java
void handleWrite(DistributedDatabase db, String key, Object value) {
    List<DatacenterNode> nodes = db.getAllNodes();
    for (DatacenterNode node : nodes) {
        node.sendWrite(key, value);
    }
}
```
x??

---
#### Detecting Concurrent Writes in Dynamo-Style Databases
Background context: In Dynamo-style databases, multiple clients can concurrently write to the same key. This leads to conflicts which must be resolved by ensuring nodes have a well-defined ordering of writes or by using conflict resolution mechanisms.
:p How do nodes handle concurrent writes in a three-node datastore?
??x
In a three-node datastore with concurrent writes from two clients A and B, the nodes may receive requests in different orders due to network delays. For instance:
- Node 1 receives write from A but not B.
- Node 2 first receives write from A then B.
- Node 3 first receives write from B then A.

Without a well-defined ordering or conflict resolution mechanism, nodes would become permanently inconsistent. To handle this, Dynamo-style databases often use techniques like hinted handoff and read repair to resolve conflicts.
x??

---

#### Last Write Wins (LWW) Conflict Resolution

In distributed systems, achieving eventual consistency can be challenging due to concurrent writes. One approach is to use Last Write Wins (LWW), where a replica only stores the most recent value and older values are overwritten or discarded.

To determine which write is more "recent," timestamps are often attached to each write request. The write with the highest timestamp is considered the most recent, while others are discarded. This method ensures eventual convergence but sacrifices data durability in cases of concurrent writes.

LWW is implemented as follows:
1. Each write operation includes a timestamp.
2. If multiple writes occur concurrently and have different timestamps, the one with the higher timestamp "wins."
3. Only the latest winning write survives; all others are discarded.

:p How does LWW achieve eventual convergence?
??x
LWW achieves eventual convergence by ensuring that among concurrent writes, only the most recent (highest-timestamped) write is stored across replicas. Any earlier writes are discarded or overwritten, leading to a consistent state over time.
x??

---
#### Handling Concurrent Writes in Cassandra

Cassandra uses LWW as its default mechanism for conflict resolution between concurrent writes. Each write operation is assigned a unique timestamp.

:p What makes LWW suitable for certain use cases like caching?
??x
LWW can be acceptable in scenarios where losing data is tolerable, such as in caching systems. Since the system doesn't strictly require all updates to persist, it can discard older updates without significant impact on overall functionality.
x??

---
#### Determining Concurrency and Timestamps

To decide whether two operations are concurrent, we use timestamps to order writes. For example, if operation A inserts a value, and operation B increments that same value after the insertion, they are not considered concurrent because B's write depends on the state set by A.

:p How do you determine if two operations are concurrent?
??x
Two operations are determined to be concurrent if neither can be said to have happened before the other. In practice, this is often resolved using timestamps where each write operation has a unique timestamp. If multiple writes occur simultaneously and share a timestamp or have overlapping times, their exact order is undefined.
x??

---
#### Leaderless Replication

In leaderless replication systems like Cassandra and Riak, there's no central coordinator to manage the sequence of operations. Each node processes writes independently.

:p How do LWW implementations handle concurrent writes in a distributed system?
??x
LWW implementations handle concurrent writes by attaching timestamps to each write request. The write with the highest timestamp "wins" and is stored; all others are discarded or overwritten, ensuring eventual consistency across the replicas.
x??

---
#### Example of Concurrent Writes

Consider two nodes processing write requests independently. Node A inserts a value into a key, and Node B increments that same key after A's insertion.

:p How would LWW resolve the conflict between these two concurrent writes?
??x
In this scenario, assuming Node B's timestamp is higher than or equal to Node A's, according to LWW, Node B's increment operation "wins" and overwrites the value inserted by Node A. The older write (Node A) is discarded.
x??

---

#### Causal Dependency and Concurrency
Background context: The text discusses how operations can be ordered based on their causal relationship. If one operation depends on another, it is considered to have happened after the first. Conversely, if two operations are unaware of each other, they are said to be concurrent.
:p What does the term "causal dependency" mean in the context of database operations?
??x
Causal dependency refers to a situation where an operation B builds upon or depends on another operation A, implying that B cannot occur until after A has completed. This relationship can be visualized as one operation happening before another due to its direct influence.
x??

---

#### Concurrent Operations
Background context: In distributed systems, operations are considered concurrent if they do not have any causal dependency and are unaware of each other's existence during their execution. Exact timing is less important than the awareness of these operations.
:p How can you determine if two database operations are concurrent?
??x
To determine if two operations are concurrent, check whether neither operation has knowledge or dependence on the other. In a distributed system, this means that both operations should be unaware of each other's existence at the point of execution.
x??

---

#### Happens-Before Relationship
Background context: The concept involves determining if one operation happened before another based on whether the second depends on the first for correctness. This is crucial for understanding concurrency in databases and resolving conflicts.
:p How does the happens-before relationship help define concurrency?
??x
The happens-before relationship helps define concurrency by establishing a temporal order between operations where one must occur before the other to maintain correct system behavior. If operation B happens after A, then B depends on A; otherwise, they are concurrent if neither knows about the other.
x??

---

#### Concurrency and Time in Distributed Systems
Background context: In distributed systems, exact timing of events is often challenging due to clock differences and network delays. The concept introduces a more abstract notion of concurrency based on awareness rather than absolute time.
:p Why does it matter that operations are unaware of each other's occurrence in concurrent operations?
??x
It matters because being unaware means the system treats both operations as happening independently, without one influencing the outcome of the other. This abstraction simplifies conflict resolution and ensures that operations can be processed in a way that preserves consistency.
x??

---

#### Capturing the Happens-Before Relationship Algorithm
Background context: The text outlines the need for an algorithm to determine if two operations are concurrent or if there is a happens-before relationship, focusing on a single replica scenario first before generalizing it to multiple replicas.
:p How would you design a simple algorithm to detect whether two operations are concurrent?
??x
To detect concurrency, we can use a timestamp-based approach where each operation records its occurrence time. If one operation has an earlier recorded time than the other, the latter is considered to happen after the former. If both times overlap or there's no clear order, they are concurrent.
```java
public class Operation {
    private long startTime;
    private long endTime;

    public Operation(long startTime, long endTime) {
        this.startTime = startTime;
        this.endTime = endTime;
    }

    public boolean happensBefore(Operation other) {
        return (this.endTime < other.startTime);
    }
}
```
x??

---

#### Leaderless Replication in Distributed Databases
Background context: The text explains how leaderless replication works, where a database has multiple replicas that operate independently. Determining the correct sequence of operations is crucial to maintain data consistency.
:p What is the significance of having only one replica for simplicity?
??x
Having only one replica simplifies the algorithm by eliminating the need for a central coordinator or leader. This makes it easier to understand and implement the happens-before relationship without complicating the system with leader election processes.
x??

---

#### Example Scenario: Concurrent Shopping Cart Operations
Background context: The text uses an example of two clients adding items to the same shopping cart concurrently, which helps illustrate the concepts of concurrent operations and the need for conflict resolution in a distributed database setting.
:p How does this scenario help explain the concept of concurrent operations?
??x
This scenario demonstrates that even though both clients add items simultaneously (concurrently), they are unaware of each other's actions. Therefore, their operations can be considered concurrent, and any conflicts must be resolved to maintain data integrity in a distributed system.
x??

---

#### Concurrent Writes and Versioning
Background context: In distributed systems, particularly those involving leaderless replication like a shopping cart example, concurrent writes can lead to complex versioning scenarios. This is managed through version numbers that ensure causality and consistency without requiring a central leader.

:p What are the key aspects of managing concurrent writes in a leaderless replication system?
??x
The key aspects include maintaining version numbers for each write operation, ensuring clients read before writing, merging values from previous reads, and handling concurrency based on version numbers. The server must overwrite older versions but keep newer concurrent ones.
```java
// Pseudocode for handling a write request with a specific version number
public void handleWrite(String key, List<String> valueList, int version) {
    if (version < getCurrentVersion(key)) {
        // Overwrite old versions and keep concurrent versions
        overwriteOldVersions(valueList, version);
        keepConcurrentVersions();
    } else {
        // Handle as a concurrent write
        storeNewValue(valueList);
    }
}

private void overwriteOldVersions(List<String> valueList, int version) {
    for (int i = 0; i < versions.size(); i++) {
        if (versions.get(i).version <= version) {
            values.remove(versions.get(i));
        }
    }
}

private void keepConcurrentVersions() {
    // Keep values with higher version numbers
}
```
x??

---

#### Causal Dependencies in Replication
Background context: In the example, clients concurrently modify a shopping cart, leading to multiple writes and maintaining causal dependencies through version numbers. Each write operation depends on previous reads, ensuring that no data is lost and all operations are recorded.

:p How does the server handle concurrent writes in terms of causality?
??x
The server handles concurrent writes by using version numbers to determine which writes overwrite older versions and which are concurrent. It increments a version number with each write and uses this to manage writes from different clients that might be happening concurrently.
```java
// Pseudocode for managing concurrent writes based on version numbers
public void handleWrite(String key, List<String> valueList, int version) {
    if (version < getLatestVersion(key)) {
        // Overwrite old versions and keep concurrent versions
        overwriteOldVersions(valueList, version);
        keepConcurrentVersions();
    } else {
        // Handle as a concurrent write
        storeNewValue(valueList);
    }
}

private void overwriteOldVersions(List<String> valueList, int version) {
    for (int i = 0; i < versions.size(); i++) {
        if (versions.get(i).version <= version) {
            values.remove(versions.get(i));
        }
    }
}

private void keepConcurrentVersions() {
    // Keep values with higher version numbers
}
```
x??

---

#### Version Numbering System
Background context: Each write operation to a key increments its version number, and the server keeps track of these versions. This helps in managing concurrent writes by ensuring that newer versions overwrite older ones but keep any newer concurrent versions.

:p How does the server increment version numbers for each write?
??x
The server increments version numbers for each write operation to ensure causality and consistency. For a key, every time it is written, the version number is incremented. This allows the server to track which writes are newer and should overwrite older ones while keeping concurrent writes.

```java
// Pseudocode for incrementing version numbers
public int getLatestVersion(String key) {
    // Return the latest version of the given key
    return versions.get(key);
}

public void updateKey(String key, List<String> values) {
    int newVersion = getLatestVersion(key) + 1; // Increment by one for each write
    storeNewVersion(newVersion); // Store the new version number
    storeValues(values); // Store the new values with the updated version
}

private void storeNewVersion(int version) {
    versions.put(currentKey, version);
}
```
x??

---

#### Client Reading Before Writing
Background context: Clients must read a key before writing to ensure they have the latest state. This helps in managing concurrent writes by ensuring that reads include all non-overwritten values and allow clients to merge these values correctly before writing.

:p Why do clients need to read before writing?
??x
Clients need to read before writing because it ensures they have the most up-to-date state of the key being written. By reading, clients can gather all existing values (even if some are concurrent), merge them with their intended changes, and then send a new write request that includes these merged values along with the correct version number.

```java
// Pseudocode for client read before write
public void performWrite(String key, List<String> valueList) {
    int currentVersion = getLatestVersion(key); // Read to get the latest version

    // Merge received values and new data
    List<String> mergedValues = mergeReceivedValues(currentVersion, valueList);

    // Send the updated list with the correct version number
    sendWriteRequest(mergedValues, currentVersion);
}

private List<String> mergeReceivedValues(int version, List<String> newValue) {
    if (version >= getLatestVersion(key)) {
        return newValue; // No need to merge further
    }
    // Merge received values and new data
}
```
x??

---

#### Overwriting vs. Concurrent Versions
Background context: When handling writes in a leaderless replication system, the server must distinguish between overwriting old versions and keeping concurrent ones. This is achieved by comparing version numbers and deciding whether to overwrite or keep based on causality.

:p How does the server decide whether to overwrite or keep values during a write?
??x
The server decides whether to overwrite or keep values during a write by comparing the received version number with the current latest version of the key. If the received version is less than the latest, it overwrites older versions but keeps concurrent ones. Otherwise, it treats the write as concurrent and stores it.

```java
// Pseudocode for deciding whether to overwrite or keep values
public void handleWrite(String key, List<String> valueList, int version) {
    if (version < getLatestVersion(key)) {
        // Overwrite old versions and keep concurrent versions
        overwriteOldVersions(valueList, version);
        keepConcurrentVersions();
    } else {
        // Handle as a concurrent write
        storeNewValue(valueList);
    }
}

private void overwriteOldVersions(List<String> valueList, int version) {
    for (int i = 0; i < versions.size(); i++) {
        if (versions.get(i).version <= version) {
            values.remove(versions.get(i));
        }
    }
}

private void keepConcurrentVersions() {
    // Keep values with higher version numbers
}
```
x??

---

#### Sibling Values and Merging
Sibling values occur when concurrent writes happen to a single key. In such cases, Riak calls them siblings and requires clients to merge these values to avoid data loss.

:p What are sibling values in the context of distributed databases?
??x
Sibling values refer to multiple versions of the same value that result from concurrent writes to a single key. These values must be merged by the client to ensure no data is lost.
x??

---

#### Tombstone Markers
Tombstone markers indicate deletions when merging siblings. They are used to handle removals in distributed databases where items might have been deleted only on one replica.

:p How do tombstones help in managing deletions during sibling merges?
??x
Tombstones mark the deletion of an item, even if it was removed only from a single replica. When merging siblings, these markers ensure that the item is not re-added to the database, preserving data integrity.
x??

---

#### Version Vectors
Version vectors are used in multi-replica systems without a leader to track dependencies between operations and manage concurrent writes.

:p What is a version vector?
??x
A version vector is a collection of version numbers from all replicas that helps distinguish between overwrites and concurrent writes. It ensures safe reads and writes across multiple replicas.
x??

---

#### Causal Context in Riak
Riak uses causal context to encode version vectors for sending with read operations. This ensures that clients can correctly handle the merging of siblings.

:p What is causal context?
??x
Causal context, used by Riak, is a string representation of version vectors sent along with read operations. It helps in distinguishing between overwrites and concurrent writes when writing back to the database.
x??

---

#### Dotted Version Vector
The dotted version vector is an interesting variant used in Riak 2.0 for managing concurrent writes without a leader.

:p What is the dotted version vector?
??x
The dotted version vector is a specific implementation of version vectors used in Riak 2.0 to manage concurrent writes. It allows distinguishing between overwrites and concurrent writes, ensuring safe operations across multiple replicas.
x??

---

#### Merging Siblings with Union Approach
In scenarios where items can be added but not removed, using the union approach to merge siblings is a reasonable strategy.

:p How does the union approach work in merging siblings?
??x
The union approach involves combining all unique elements from sibling values. For example, if one cart has [milk, flour, eggs] and another has [eggs, milk, ham], the result would be [milk, flour, eggs, bacon, ham], removing duplicates.
x??

---

#### Handling Removals in Siblings
When items can be removed from a shopping cart, simply taking the union of siblings might not give the correct result. A marker must indicate removal to avoid re-adding deleted items.

:p How should siblings be handled when deletions are involved?
??x
To handle deletions correctly, use tombstone markers that indicate an item has been removed. When merging siblings, these markers prevent re-added deleted items, ensuring data integrity.
x??

---

#### Automatic Conflict Resolution with CRDTs
CRDTs (Conflict-free Replicated Data Types) can automatically merge siblings in sensible ways, including preserving deletions.

:p What are CRDTs and how do they help?
??x
CRDTs are a family of data structures designed to manage concurrent writes without the need for external coordination. They can automatically merge siblings while preserving operations like additions and deletions.
x??

---

#### High Availability and Replication
Replication is used to keep a system running even when one or more machines, including entire data centers, fail. This ensures continuous operation of applications.

:p What is the purpose of using replication for high availability?
??x
The purpose of using replication for high availability is to maintain the system's uptime by ensuring that there are multiple copies of the data stored on different nodes. If one node fails, another node can take over without disrupting service.
x??

---

#### Disconnected Operation and Replication
Replication allows applications to continue functioning during network interruptions when data is kept synchronized across multiple machines.

:p How does replication enable disconnected operation?
??x
Replication enables disconnected operation by maintaining a synchronized state of the data on different nodes. This way, if there's a network interruption, each node can operate independently until connectivity is restored.
x??

---

#### Latency and Replication
Geographic placement of data close to users reduces latency, allowing faster interaction with the system.

:p What role does replication play in reducing latency?
??x
Replication plays a crucial role in reducing latency by storing copies of data geographically close to the users. This minimizes the distance data travels between the user and the server, thus reducing response times.
x??

---

#### Scalability and Replication
Multiple replicas can handle higher read volumes compared to a single machine, improving overall system throughput.

:p How does replication enhance scalability?
??x
Replication enhances scalability by distributing reads across multiple replicas. This allows more concurrent users to access data simultaneously without overloading any single node.
x??

---

#### Single-Leader Replication
A single leader manages all writes and sends change events to followers, which may serve read requests but can have stale data.

:p What is the primary advantage of single-leader replication?
??x
The primary advantage of single-leader replication is its simplicity. It requires minimal conflict resolution since only one node (the leader) handles write operations. Additionally, it's straightforward to implement and understand.
x??

---

#### Multi-Leader Replication
Multiple nodes can accept writes, and they communicate data change events among themselves, allowing for more robust systems but complicating fault tolerance.

:p What is a key challenge in multi-leader replication?
??x
A key challenge in multi-leader replication is managing conflicts that arise when multiple leaders attempt to write to the same piece of data simultaneously. This requires complex conflict resolution mechanisms.
x??

---

#### Leaderless Replication
Writes are distributed among several nodes, and reads can be performed from any node to detect and correct stale data.

:p What is a main benefit of leaderless replication?
??x
A main benefit of leaderless replication is its robustness in the presence of faulty nodes or network interruptions. Since no single node is responsible for all writes, the system can continue operating even if some nodes fail.
x??

---

#### Synchronous vs Asynchronous Replication
Synchronous replication ensures immediate acknowledgment before committing data changes, while asynchronous replication may delay this process.

:p What is a consequence of using asynchronous replication?
??x
A consequence of using asynchronous replication is that it can lead to data loss or inconsistency if the leader fails and an asynchronously updated follower is promoted without proper synchronization.
x??

---

#### Consistency Models in Replication
Consistency models like Read-after-write, Monotonic reads, and Consistent prefix reads help ensure predictable behavior under replication lag.

:p What does Read-after-write consistency guarantee?
??x
Read-after-write consistency guarantees that a user will always see the data they just wrote. This ensures that operations are idempotent and maintain causality.
x??

---

---
#### Concurrency Issues in Multi-Leader and Leaderless Replication
Background context: In multi-leader and leaderless replication, multiple nodes can accept write operations simultaneously. This setup increases throughput but introduces complexities such as concurrent writes leading to potential conflicts.

:p What are the key challenges introduced by multi-leader and leaderless replication regarding concurrency?
??x
The key challenges include managing concurrent writes where different leaders may attempt to update the same data at the same time, potentially causing conflicts. The system must have mechanisms to detect these conflicts and resolve them, such as merge strategies or conflict resolution algorithms.

Example of a conflict detection mechanism in pseudocode:
```pseudocode
function handleWriteOperation(operationA, operationB) {
    if (operationA.timeStamp < operationB.timeStamp) {
        return operationA happened before operationB;
    } else if (operationA.timeStamp > operationB.timeStamp) {
        return operationB happened before operationA;
    } else {
        // Conflict detected
        resolveConflict(operationA, operationB);
    }
}

function resolveConflict(op1, op2) {
    // Implement conflict resolution logic here
}
```
x??

---
#### Conflict Resolution through Merging Updates
Background context: When concurrent writes occur in distributed systems, merging the updates is a common approach to resolve conflicts. This involves combining changes made by multiple operations into a single, consistent state.

:p How does merge-based conflict resolution work in distributed databases?
??x
Merge-based conflict resolution works by applying all concurrent write operations and then merging their results into one coherent version of the data. The system ensures that no conflicting updates are applied simultaneously to avoid overwriting valid changes.

Example of merge logic for two concurrent updates:
```pseudocode
function mergeUpdates(update1, update2) {
    let result = {};
    
    // Apply both updates to a temporary object
    result = applyUpdate(result, update1);
    result = applyUpdate(result, update2);
    
    return result;
}

function applyUpdate(state, update) {
    for (let key in update) {
        if (!state.hasOwnProperty(key)) {
            state[key] = update[key];
        } else {
            // Handle specific logic to merge or resolve conflicts
            if (update[key] === "value1" && state[key] === "value2") {
                state[key] = "mergedValue";
            }
        }
    }
    return state;
}
```
x??

---
#### References and Further Reading
Background context: The text references various papers, articles, and books to support the discussion on distributed databases and replication techniques. These sources provide deeper insights into specific technologies and concepts.

:p What is the significance of referencing multiple academic papers and technical documents?
??x
Referencing multiple academic papers and technical documents provides a comprehensive understanding of the state-of-the-art in distributed systems, highlighting different approaches and their trade-offs. It helps readers to explore deeper into particular techniques or systems that are mentioned.

For example, references like [1] "Notes on Distributed Databases" by IBM Research provide foundational knowledge about distributed databases, while papers such as [8] "Chain Replication for Supporting High Throughput and Availability" discuss advanced replication strategies used in modern cloud services.

Example of referencing a paper:
Reference: [8] Brad Calder, Ju Wang, Aaron Ogus, et al.: “ Windows Azure Storage: A Highly Available Cloud Storage Service with Strong Consistency ,” at 23rd ACM Symposium on Operating Systems Principles (SOSP), October 2011. This paper discusses the architecture and consistency models used in distributed cloud storage systems.

x??

---

---
#### HBase Master/Master Replication Issue (HBASE-7709)
Background context: The issue discussed in [43] by Lars Hofhansl highlights a potential infinite loop problem that could occur in a master/master replication setup within HBase. This is crucial for understanding the complexities and challenges associated with maintaining consistency across multiple masters.

:p What does HBASE-7709 refer to?
??x
HBASE-7709 refers to an issue where there might be an infinite loop possible in the master/master replication architecture of HBase. This highlights a critical problem that needs addressing for ensuring reliable and consistent data replication between multiple masters.
x??

---
#### Weighted Voting for Replicated Data (Gifford, 1979)
Background context: David K. Gifford's work on weighted voting for replicated data [44] introduced an approach to handle the challenges of distributed systems where nodes have different weights or priorities in decision-making processes. This method ensures that more important nodes can influence decisions more significantly.

:p What is weighted voting used for?
??x
Weighted voting is used to ensure that critical operations are handled with greater priority by assigning higher importance (weight) to certain nodes within a replicated system. This approach helps maintain the overall stability and performance of the distributed system.
x??

---
#### Flexible Paxos (Howard, Malkhi, Spiegelman, 2016)
Background context: The paper [45] by Heidi Howard, Dahlia Malkhi, and Alexander Spiegelman discusses Flexible Paxos, which revisits the concept of quorum intersection. This method aims to optimize the consensus process in distributed systems, making it more flexible and adaptable to different network conditions.

:p What is Flexible Paxos?
??x
Flexible Paxos is a method for optimizing the consensus process in distributed systems by redefining how quorums are used. It allows for more dynamic and efficient decision-making across nodes, improving overall system performance.
x??

---
#### Re: Absolute Consistency (Blomstedt, 2012)
Background context: In his email [46], Joseph Blomstedt discusses the challenges of achieving absolute consistency in distributed systems like Riak. The discussion revolves around balancing consistency with other aspects such as availability and partition tolerance.

:p What did Joseph Blomstedt discuss about consistency?
??x
Joseph Blomstedt discussed the difficulties in achieving absolute consistency in distributed systems, specifically highlighting that maintaining high levels of consistency can come at the cost of reducing availability and potentially causing partitions. He emphasized the need for a balanced approach to ensure reliability.
x??

---
#### Eventual Consistency (Bailis et al., 2014)
Background context: The paper [48] by Peter Bailis, Shivaram Venkataraman, Michael J. Franklin, and others provides a detailed analysis of eventual consistency using the PBS tool. It quantifies how eventual consistency performs under different conditions and helps in understanding its behavior.

:p What is PBS used for?
??x
PBS (Providing Broadcast Semantics) is used to measure and analyze the performance of systems achieving eventual consistency. The tool helps quantify the trade-offs between consistency, availability, and partition tolerance.
x??

---
#### Modern Hinted Handoff (Ellis, 2012)
Background context: Jonathan Ellis's article [49] on modern hinted handoff discusses an improvement in Cassandra for handling replica placement failures by automatically redirecting read/write requests. This mechanism ensures data remains accessible even if a node is down.

:p What is modern hinted handoff?
??x
Modern hinted handoff is a feature in distributed databases like Cassandra that automatically redirects read/write requests to other nodes when the primary node fails or becomes unavailable, ensuring data remains accessible without manual intervention.
x??

---
#### Project Voldemort Wiki (2013)
Background context: The [50] link points to an older version of the Project Voldemort wiki, which provided documentation and insights into the distributed database system. It was a valuable resource for understanding Voldemort's architecture and usage.

:p What does the Project Voldemort Wiki document?
??x
The Project Voldemort Wiki documented the architecture and usage of the Voldemort distributed database system, providing essential information for developers and administrators.
x??

---
#### Apache Cassandra 2.0 Documentation (2014)
Background context: The [51] link leads to the official documentation for Apache Cassandra 2.0, which detailed its features, performance optimizations, and best practices for deployment and management.

:p What does the Apache Cassandra 2.0 Documentation cover?
??x
The Apache Cassandra 2.0 Documentation covered various aspects of the distributed database system, including its architecture, features, performance tuning, and guidelines for deploying and managing the software.
x??

---
#### Riak Multi-Datacenter Replication (Basho Technologies, 2014)
Background context: The [52] whitepaper from Basho Technologies detailed how Riak supports multi-datacenter replication, ensuring data consistency across different geographical locations. This is crucial for applications requiring high availability and disaster recovery.

:p What does the Riak Enterprise documentation cover?
??x
The Riak Enterprise Multi-Datacenter Replication documentation covered strategies and methodologies to ensure consistent data distribution across multiple data centers, including best practices and implementation details.
x??

---
#### Why Cassandra Doesn’t Need Vector Clocks (Ellis, 2013)
Background context: In his article [53], Jonathan Ellis argues that Cassandra does not require vector clocks for achieving consistency. He explains how other mechanisms can effectively handle causality in distributed systems.

:p What does Jonathan Ellis say about Cassandra?
??x
Jonathan Ellis states that Cassandra does not need vector clocks to achieve consistent behavior, as alternative methods suffice for handling causality and ensuring data integrity.
x??

---
#### Time, Clocks, and the Ordering of Events (Lamport, 1978)
Background context: Leslie Lamport's seminal paper [54] introduced fundamental concepts about time and ordering of events in distributed systems. This work laid the groundwork for understanding how to manage causal relationships between events.

:p What is Leslie Lamport’s contribution?
??x
Leslie Lamport contributed by introducing foundational ideas on time, clocks, and the ordering of events in distributed systems, which are critical for managing causality and consistency.
x??

---
#### Riak 2.0: Data Types (Jacobson, 2014)
Background context: In his blog post [55], Joel Jacobson describes data types introduced in Riak 2.0, which aimed to enhance flexibility and data handling capabilities within the distributed database system.

:p What did Joel Jacobson discuss about Riak 2.0?
??x
Joel Jacobson discussed new data types implemented in Riak 2.0, designed to improve flexibility and data management in the distributed database.
x??

---
#### Detection of Mutual Inconsistency (Stott Parker Jr., Popek, Rudisin, 1983)
Background context: The paper [56] by Stott Parker Jr., Gerald J. Popek, Gerard Rudisin, et al., introduced methods for detecting mutual inconsistency in distributed systems. This is important for maintaining the integrity of replicated data.

:p What did Stott Parker Jr. and co-authors discuss?
??x
Stott Parker Jr. and his colleagues discussed techniques for detecting mutual inconsistency in distributed systems to ensure that all nodes maintain a coherent state.
x??

---
#### Dotted Version Vectors (Preguiça, Baquero, Almeida, 2010)
Background context: [57] by Nuno Preguiça, Carlos Baquero, Paulo Sérgio Almeida, et al., introduces dotted version vectors as a logical clock mechanism for optimistic replication. This method helps manage versioning and causality in distributed systems.

:p What is Dotted Version Vectors?
??x
Dotted Version Vectors are used as a logical clock mechanism for managing versioning and causality in optimistic replication scenarios, ensuring that replicas can correctly handle concurrent updates.
x??

---
#### Vector Clocks Revisited (Bailis et al., 2015)
Background context: Russell Brown's blog post [58] revisits vector clocks to discuss their limitations and how they are used. The discussion provides insights into the practical challenges of implementing vector clocks in distributed systems.

:p What does Russell Brown’s blog cover?
??x
Russell Brown’s blog covers the reevaluation of vector clocks, discussing their limitations and usage in distributed systems, providing context for why alternative methods like dotted version vectors might be preferred.
x??

---
#### Version Vectors Are Not Vector Clocks (Baquero, 2011)
Background context: Carlos Baquero's blog post [59] clarifies the distinction between version vectors and vector clocks. This is important for understanding different approaches to managing causality in distributed systems.

:p What does Carlos Baquero’s blog explain?
??x
Carlos Baquero’s blog explains that version vectors are not equivalent to vector clocks, highlighting key differences and their implications for distributed system design.
x??

---
#### Detecting Causal Relationships (Schwarz, Mattern, 1994)
Background context: [58] by Reinhard Schwarz and Friedemann Mattern discusses methods for detecting causal relationships in distributed computations. This is essential for ensuring that nodes can correctly order events based on their dependencies.

:p What did Schwarz and Mattern discuss?
??x
Reinhard Schwarz and Friedemann Mattern discussed techniques for detecting causal relationships in distributed computations, which are vital for managing the ordering of events across different nodes.
x??

---

