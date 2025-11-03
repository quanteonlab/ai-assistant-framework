# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 16)

**Starting Chapter:** Handling Write Conflicts

---

#### Collaborative Editing and Database Replication
Background context: Collaborative editing is often compared to offline editing use cases. In this model, changes are made locally on a client's device and then asynchronously replicated to a server or other users' clients. The goal is to ensure that multiple users can edit the same document without conflicts.
:p What key concept does collaborative editing resemble in database replication?
??x
Collaborative editing resembles offline editing use cases where local modifications are made before being synchronized with a central server or distributed across multiple replicas. This process involves ensuring consistency and handling potential conflicts when changes overlap.
x??

---

#### Single-Leader Replication vs. Multi-Leader Replication
Background context: In single-leader replication, a single node acts as the primary source of truth for writes, while in multi-leader replication, there is no such restriction, allowing multiple nodes to accept write operations independently.
:p What are the main differences between single-leader and multi-leader replication?
??x
In single-leader replication, one node (leader) is responsible for handling all write operations. Other nodes act as replicas that only replicate data from the leader. In contrast, in multi-leader replication, multiple nodes can handle write operations independently, leading to faster collaboration but increased complexity due to potential conflicts.
x??

---

#### Write Conflicts and Conflict Resolution
Background context: Write conflicts occur when multiple users attempt to modify the same record at the same time without proper coordination. Handling these conflicts requires mechanisms such as locking or conflict resolution algorithms.
:p What is a write conflict, and how can it be handled?
??x
A write conflict occurs when two or more users try to modify the same piece of data simultaneously. To handle this, you can use locking (blocking subsequent writes until current ones are committed) or implement conflict resolution logic that merges changes after they are detected.
```java
// Pseudocode for a simple conflict resolution
public class ConflictResolver {
    public String resolveConflict(String change1, String change2) {
        // Logic to merge changes
        return "MergedChange";
    }
}
```
x??

---

#### Synchronous vs. Asynchronous Conflict Detection
Background context: In single-leader systems, conflicts are detected immediately (synchronously), whereas in multi-leader setups, conflicts may only be detected later when data is replicated.
:p What are the differences between synchronous and asynchronous conflict detection?
??x
In a single-leader system, the second write transaction is either blocked until the first one completes or aborted if it cannot wait. In contrast, in a multi-leader setup, both writes can succeed initially, and conflicts are only detected asynchronously later. Asynchronous detection means that resolving conflicts may be more difficult as users might have moved on.
x??

---

#### Conflict Avoidance Strategies
Background context: One strategy to avoid write conflicts is by ensuring all writes for a record go through the same leader node. This approach can simplify conflict resolution but limits flexibility in routing requests.
:p How can you avoid write conflicts?
??x
You can avoid write conflicts by ensuring that all writes for a particular record always pass through the same leader node. For example, in a user-editable application, you could route all requests from a single user to a specific datacenter and use its leader for all read/write operations.
```java
// Pseudocode for routing based on user ID
public String determineLeaderForUser(String userId) {
    // Logic to map user IDs to leaders
    return "Datacenter1-Leader";
}
```
x??

---

#### Conflict Resolution Mechanisms

Conflict resolution is necessary when a single-leader database transitions to a multi-leader configuration where writes can be applied out of order on different leaders. This leads to potential inconsistencies among replicas.

:p What are some common methods for resolving conflicts in a multi-leader database?
??x
There are several methods to resolve conflicts:
1. **Last Write Wins (LWW)**: Assign each write a unique identifier such as a timestamp, and the last write is considered the winner.
2. **Replica Priority**: Assign each replica a unique ID, and writes from higher-numbered replicas take precedence over lower-numbered ones.
3. **Value Merging**: Merge conflicting values into a single value (e.g., concatenate them).
4. **Explicit Conflict Resolution Logic**: Use application code to resolve conflicts at read time or write time.

:p How does the Last Write Wins (LWW) strategy work?
??x
In LWW, each write operation is assigned a unique identifier such as a timestamp. When conflicting writes are detected, the one with the highest identifier wins and other conflicting writes are discarded.

```java
// Pseudocode for handling Last Write Wins (LWW)
class Database {
    public void handleWrite(long timestamp, String key, String value) {
        if (!conflictDetected(timestamp, key)) {
            storeWrite(timestamp, key, value);
        } else {
            // Discard conflicting writes and keep the latest one
            discardConflictingWrites(key);
            storeWrite(timestamp, key, value);
        }
    }

    private boolean conflictDetected(long timestamp, String key) {
        // Check if a write with the same key has a higher timestamp
        return false;
    }

    private void storeWrite(long timestamp, String key, String value) {
        // Store the write in the database
    }

    private void discardConflictingWrites(String key) {
        // Discard any conflicting writes for the given key
    }
}
```
x??

---

#### Replica Priority

Replica priority is a strategy where each replica has a unique ID, and writes from higher-numbered replicas are prioritized over those from lower-numbered ones. This method also leads to data loss if a write originates from a lower-priority replica after a higher-priority one.

:p How does the Replica Priority mechanism work?
??x
In Replica Priority, each replica is assigned a unique ID. When conflicting writes occur, the system prioritizes writes that originate from replicas with higher IDs over those from lower IDs. This approach also leads to data loss if a write originates after another with a higher-priority replica.

```java
// Pseudocode for handling Replica Priority
class Database {
    public void handleWrite(long replicaId, String key, String value) {
        if (!conflictDetected(replicaId, key)) {
            storeWrite(replicaId, key, value);
        } else {
            // Discard conflicting writes and keep the highest-priority one
            discardConflictingWrites(key);
            storeWrite(replicaId, key, value);
        }
    }

    private boolean conflictDetected(long replicaId, String key) {
        // Check if a write with the same key has a higher replica ID
        return false;
    }

    private void storeWrite(long replicaId, String key, String value) {
        // Store the write in the database
    }

    private void discardConflictingWrites(String key) {
        // Discard any conflicting writes for the given key
    }
}
```
x??

---

#### Value Merging

Value merging is a strategy where conflicting values are combined to produce a single, merged value. This could involve concatenating or ordering the conflicting values.

:p How does Value Merging work?
??x
In Value Merging, conflicting writes for the same key are combined into a single value using some predefined logic (e.g., alphabetical order and concatenation). For instance, if "B" is written first and "C" later, merging them could result in "B/C".

```java
// Pseudocode for handling Value Merging
class Database {
    public void handleWrite(String key, String value) {
        if (!conflictDetected(key)) {
            storeWrite(key, value);
        } else {
            // Merge conflicting writes
            mergeConflictingWrites(key, value);
            storeWrite(key, mergedValue);
        }
    }

    private boolean conflictDetected(String key) {
        // Check if a write with the same key is already present
        return false;
    }

    private void storeWrite(String key, String value) {
        // Store the write in the database
    }

    private String mergeConflictingWrites(String key, String newWrite) {
        // Logic to merge conflicting writes
        // For example: concatenate or order them
        if (conflictsDetected(key)) {
            return "B/C"; // Example merged value
        }
        return null;
    }
}
```
x??

---

#### Explicit Conflict Resolution

Explicit conflict resolution involves recording conflicts in a data structure and resolving them at read time using application logic. This can involve user prompts or automatic resolution.

:p How does Explicit Conflict Resolution work?
??x
In Explicit Conflict Resolution, conflicting writes are recorded with all their information. When the data is read, multiple versions of the data are returned to the application, which then resolves conflicts either by prompting a user or automatically.

```java
// Pseudocode for handling Explicit Conflict Resolution on Write
class Database {
    public void handleWrite(String key, String value) {
        if (!conflictDetected(key)) {
            storeWrite(key, value);
        } else {
            // Record conflicting writes
            recordConflict(key, value);
            storeWrite(key, value); // Store the new write
        }
    }

    private boolean conflictDetected(String key) {
        // Check if a write with the same key is already present
        return false;
    }

    private void storeWrite(String key, String value) {
        // Store the write in the database
    }

    private void recordConflict(String key, String value) {
        // Record the conflicting write in an explicit data structure
    }
}

// Pseudocode for handling Explicit Conflict Resolution on Read
class DatabaseReader {
    public String readData(String key) {
        if (conflictDetected(key)) {
            return handleConflicts(key);
        } else {
            // Normal read operation
            return fetchValue(key);
        }
    }

    private boolean conflictDetected(String key) {
        // Check if there are conflicting writes for the given key
        return false;
    }

    private String fetchValue(String key) {
        // Fetch normal value from database
        return "B";
    }

    private String handleConflicts(String key) {
        // Resolve conflicts and return a resolved version
        // This could involve prompting a user or automatic resolution logic
        return "B/C"; // Example merged value
    }
}
```
x??

---

#### Conflict Resolution in Distributed Databases

Conflict resolution is a critical aspect of distributed databases, particularly in systems where data can be modified concurrently. In CouchDB, for instance, conflicts are typically resolved on an individual document level rather than at the transaction level. This means that each write operation within a transaction is considered separately when resolving conflicts.

:p How does CouchDB handle conflict resolution?
??x
CouchDB resolves conflicts at the level of individual documents, not entire transactions. Each write operation within a transaction is treated independently for conflict resolution purposes.
x??

---
#### Automatic Conflict Resolution Challenges

Automatic conflict resolution can become complex and error-prone. An example from Amazon illustrates this issue: for some time, the conflict resolution logic on shopping carts would preserve items added but not those removed, leading to unexpected behavior where items might reappear in customer carts.

:p What was the issue with Amazon's conflict resolution logic?
??x
The issue was that the conflict resolution logic preserved items added to the cart but did not preserve items removed from it. This caused customers to sometimes see previously removed items reappearing in their carts.
x??

---
#### Conflict-Free Replicated Datatypes (CRDTs)

CRDTs are a family of data structures designed for concurrent editing by multiple users, which automatically resolve conflicts in sensible ways. These datatypes can be implemented in databases like Riak 2.0.

:p What are CRDTs and what do they do?
??x
CRDTs are data structures that allow for concurrent modification by multiple users while automatically resolving conflicts. They enable consistent updates without manual conflict resolution steps.
x??

---
#### Mergeable Persistent Data Structures

Mergeable persistent data structures track history explicitly, similar to Git, using a three-way merge function. This approach helps in resolving conflicts more effectively.

:p How do mergeable persistent data structures resolve conflicts?
??x
Mergeable persistent data structures use explicit tracking of history and a three-way merge function to resolve conflicts. This method ensures that changes from different versions can be combined without manual intervention.
x??

---
#### Operational Transformation

Operational transformation is the conflict resolution algorithm behind collaborative editing tools like Etherpad and Google Docs. It’s specifically designed for concurrent editing of ordered lists, such as text documents.

:p What is operational transformation used for?
??x
Operational transformation is used to resolve conflicts in real-time collaborative editing applications. It ensures that changes made by different users can be merged coherently without losing consistency.
x??

---
#### Example of a Conflict

In the example provided, two writes concurrently modified the same field in the same record, setting it to two different values. This clearly indicates a conflict.

:p What is an example of a clear conflict?
??x
A clear example of a conflict is when two concurrent writes modify the same field in the same record but set it to different values.
x??

---
#### Multi-Leader Replication

Multi-leader replication involves synchronizing data across multiple leaders. This can introduce complexity due to potential conflicts that need to be resolved.

:p What is multi-leader replication?
??x
Multi-leader replication refers to a scenario where data is synchronized across multiple leaders in a distributed system, which may lead to complex conflict resolution scenarios.
x??

---
#### Conflict Resolution in Distributed Systems

Conflict resolution can become complicated, especially as the number of concurrent modifications increases. Automated solutions like CRDTs, mergeable persistent data structures, and operational transformation are being explored.

:p What automated methods are used for conflict resolution?
??x
Automated methods include Conflict-Free Replicated Datatypes (CRDTs), Mergeable Persistent Data Structures, and Operational Transformation. These approaches aim to handle concurrent modifications more effectively.
x??

---

#### Multi-Leader Replication Overview

In distributed systems, multi-leader replication refers to a scenario where there are multiple nodes (leaders) that can accept write operations. This is different from a star schema, which describes data model structures rather than communication topologies.

:p What is the main difference between multi-leader replication and a star schema?
??x
Multi-leader replication involves setting up write paths among multiple leaders to ensure writes are propagated appropriately, while a star schema defines how data models are structured for analytics. The key distinction lies in their primary focus: one on communication topology, the other on data model structure.
x??

---

#### Conflict Resolution Between Leaders

When two users make bookings on different leaders, conflicts can arise as there is no immediate way to resolve them without manual intervention or a more complex system.

:p How does conflict resolution work between different leaders in multi-leader replication?
??x
Conflict resolution in multi-leader replication requires careful handling. When booking conflicts occur due to writes being made on different leaders, the system may need to detect and reconcile these conflicts later. This often involves additional mechanisms or manual steps to ensure consistency across all nodes.

In a distributed system like this, automatic conflict detection and resolution can be challenging, as it might require coordination between multiple nodes to determine which write operation should take precedence.
x??

---

#### Replication Topologies

Replication topologies describe how writes are propagated from one node to another in multi-leader scenarios. Examples include circular, star, and all-to-all topologies.

:p What are some examples of replication topologies used in multi-leader systems?
??x
Some common replication topologies used in multi-leader systems are:
- **Circular Topology**: Each leader receives writes from one node and forwards them to another.
- **Star Topology**: A single root node forwards writes to all other nodes.
- **All-to-all Topology**: Every leader sends its writes to every other leader.

These topologies differ in their complexity and fault tolerance, with densely connected topologies like all-to-all offering better resilience against failures but potentially higher latency due to additional routing paths.
x??

---

#### Identifying and Ignoring Replicated Data

To prevent infinite replication loops, nodes are given unique identifiers. Writes are tagged with these identifiers, and a node will ignore changes that it has already processed.

:p How does the system avoid processing the same write multiple times in multi-leader replication?
??x
The system avoids processing the same write multiple times by tagging each write with the identifiers of all the nodes it has passed through. When a node receives a data change, it checks if this identifier is present. If it is, the node ignores the write because it knows that the message has already been processed.

For example:
```java
public class ReplicationLog {
    private List<String> processedNodes = new ArrayList<>();

    public void processWrite(String write) {
        String nodeId = getNodeId();  // Assume this method returns a unique identifier for the node
        if (processedNodes.contains(nodeId)) {
            return; // Ignore write, already processed
        }
        
        processedNodes.add(nodeId);
        // Process the write normally
    }
}
```
x??

---

#### Handling Replication Message Order

In circular and star topologies, a single node failure can disrupt communication. In more densely connected topologies like all-to-all, messages may arrive out of order due to network conditions.

:p What issues might arise with replication message order in multi-leader systems?
??x
Replication message order issues can arise from the following:
- **Topology Disruptions**: A single node failure in circular or star topologies can disrupt communication between other nodes.
- **Network Congestion**: In all-to-all topologies, faster network links may cause some messages to "overtake" others, leading to out-of-order delivery.

To handle these issues, systems often need additional logic for causality checks and ensuring writes are processed in the correct order.

For instance, a system might use timestamps or sequence numbers to detect and resolve out-of-order writes:
```java
public class CausalConsistencyChecker {
    private Map<String, Long> writeTimestamps = new HashMap<>();

    public boolean isCausallyValid(String writeId, long timestamp) {
        if (writeTimestamps.containsKey(writeId)) {
            return writeTimestamps.get(writeId) < timestamp;
        }
        return true; // Assume valid
    }

    public void recordWrite(String writeId, long timestamp) {
        writeTimestamps.put(writeId, timestamp);
    }
}
```
x??

---

#### Fault Tolerance in Replication

Densely connected topologies like all-to-all provide better fault tolerance because messages can travel along different paths. However, they can introduce issues with message order and network conditions.

:p What is the trade-off between fault tolerance and message order in multi-leader replication?
??x
The trade-off involves balancing fault tolerance and message order:
- **Fault Tolerance**: Densely connected topologies like all-to-all provide better resilience against single points of failure because messages can travel through multiple paths.
- **Message Order**: However, such topologies may struggle with network conditions causing out-of-order delivery. Nodes must implement mechanisms to ensure causality checks and maintain the correct order of operations.

To achieve both fault tolerance and ordered writes, systems often use a combination of topology design and additional consistency mechanisms:
```java
public class ReplicationManager {
    private List<ReplicaNode> nodes = new ArrayList<>();
    
    public void replicateWrite(WriteRequest request) {
        for (ReplicaNode node : nodes) {
            node.forwardWrite(request);
        }
    }

    public boolean validateWriteOrder(String writeId, long timestamp) {
        // Check against timestamps and causality
        return isCausallyValid(writeId, timestamp);
    }
}
```
x??

---

#### Leaderless Replication Overview
Background context explaining leaderless replication. In this model, there is no single node acting as a leader to order writes and ensure consistency across replicas. Instead, clients can send write requests directly to any replica or through a coordinator that does not enforce an ordering of writes.

:p What is the main characteristic of leaderless replication?
??x
In leaderless replication, there is no central leader node responsible for enforcing the order in which writes are processed; instead, multiple replicas can accept writes independently. This design eliminates the need for failovers but introduces challenges in maintaining consistent data across replicas.
x??

---
#### Failover Handling in Leader-based vs. Leaderless Replication
Explaining how failure handling works differently between leader-based and leaderless replication configurations.

:p How does a system handle writes when a replica is down in a leader-based configuration?
??x
In a leader-based configuration, if one of the replicas goes down, writes may need to be redirected or the system might perform a failover to a new leader. This process ensures that writes continue to be ordered and processed correctly.
x??

---
#### Quorum Writes and Reads
Explanation on quorum requirements for write operations in Dynamo-style databases.

:p What is the minimum number of replicas needed to confirm a successful write operation in Dynamo-style databases?
??x
In Dynamo-style databases, a minimum of w nodes must confirm a write operation. This ensures that at least two out of three replicas (for n=3) have received and stored the write.
x??

---
#### Read Repair Process
Explanation on how read repair works to keep data consistent across all replicas.

:p How does read repair ensure up-to-date values in leaderless replication?
??x
Read repair involves a client detecting stale responses during a read operation. If a client reads different versions from multiple nodes, it writes the most recent version back to any node with an outdated value.
Example pseudocode:
```python
def perform_read_repair(replicas):
    latest_value = max(replicas.values(), key=lambda x: x.version)
    for replica in replicas:
        if replica.value.version < latest_value.version:
            replica.write(latest_value)
```
x??

---
#### Anti-Entropy Process
Explanation on the anti-entropy process used to synchronize data between replicas.

:p What is the purpose of the anti-entropy process in Dynamo-style databases?
??x
The anti-entropy process ensures that all replicas eventually have the latest data by constantly checking for differences and synchronizing missing data. It does not enforce a specific order, leading to potentially delayed updates.
x??

---
#### Quorum Read Operation
Explanation on how quorum reads work to ensure up-to-date values.

:p How are quorum reads configured in Dynamo-style databases?
??x
Quorum reads involve querying at least r nodes for each read operation. If w + r > n, the system ensures that at least one of the r nodes queried has the latest write.
Example configuration: 
```python
n = 3  # Number of replicas
w = 2  # Write quorum
r = 2  # Read quorum
```
x??

---
#### Tolerating Node Unavailability
Explanation on how node unavailability can be tolerated in leaderless replication.

:p How many unavailable nodes can the system tolerate with n=5, w=3, r=3?
??x
With n=5, w=3, and r=3, the system can tolerate up to two unavailable nodes. As long as at least one of the r nodes queried during a read or write operation has seen the most recent successful write, reads and writes will continue to return up-to-date values.
x??

---

