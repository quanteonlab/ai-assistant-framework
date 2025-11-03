# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 14)

**Starting Chapter:** Handling Node Outages

---

#### Leader and Follower Configuration
Background context: In a distributed database system, leader-based replication is often used where one node (the leader) handles all write operations and other nodes (followers) handle read operations. This configuration can be either synchronous or asynchronous.

:p What are the characteristics of leader-based replication?
??x
In leader-based replication, writes are handled by the leader, which replicates changes to followers. If configured asynchronously, the leader may continue processing new writes even if some replicas have not yet received them. This means that writes might not be durable unless they reach all followers.
x??

---
#### Semi-Synchronous Replication
Background context: Semi-synchronous replication is a type of leader-based replication where the leader waits for at least one follower to confirm receipt of a write before returning confirmation to the client.

:p What distinguishes semi-synchronous replication from completely asynchronous replication?
??x
In semi-synchronous replication, writes are not lost as long as at least one replica confirms receiving them. This provides better durability compared to fully asynchronous systems but may introduce latency due to waiting for confirmations.
x??

---
#### Setting Up New Followers
Background context: When adding a new follower, it's crucial that the new node has an up-to-date copy of the leader’s data.

:p How can you ensure a new follower has an accurate copy of the leader’s data?
??x
Take a consistent snapshot of the leader’s database (without locking) and copy it to the new follower. Then, have the new follower catch up by requesting all subsequent changes from the leader, using the replication log.
x??

---
#### Handling Node Outages
Background context: Nodes in a distributed system can fail due to hardware issues or maintenance.

:p What is failover in the context of leader-based replication?
??x
Failover refers to the process where a new leader is chosen and clients are reconfigured to send writes to this new leader after the current leader fails. This ensures the system remains available.
x??

---
#### Catch-Up Recovery for Follower Failure
Background context: When a follower node crashes, it can recover by replaying logs of received data changes.

:p How does a failed follower recover from an outage?
??x
A failed follower recovers by connecting to the leader and requesting all the data changes that occurred since its last snapshot. It then applies these changes to catch up with the current state.
x??

---
#### Leader Failure: Failover Process
Background context: If the leader fails, a new leader needs to be elected to maintain system availability.

:p What steps are involved in handling a leader failure?
??x
1. Determine that the leader has failed by using timeout mechanisms.
2. Elect a new leader through an election process or appointing it manually.
3. Reconfigure clients to send writes to the new leader and ensure all replicas follow the new leader.
x??

---
#### Challenges with Failover in Asynchronous Replication
Background context: In asynchronous replication, if a new leader is chosen after some writes from the old leader have not yet been replicated, these writes might be lost or cause conflicts.

:p What are potential issues during failover for systems using asynchronous replication?
??x
The main issue is that unreplicated writes from the old leader may be lost if the former leader rejoins. New leaders might receive conflicting writes in the meantime. The common solution is to discard unreplicated writes, which can violate clients' durability expectations.
x??

---
#### Example of Database Replication Setup
Background context: Different databases have different methods for setting up and managing followers.

:p How does one set up a follower in a database system like MySQL?
??x
1. Take a consistent snapshot (e.g., using `innobackupex`).
2. Copy the snapshot to the new follower.
3. Connect the follower to the leader and request all data changes since the snapshot.
4. The follower processes these changes, catching up with the leader’s state.
x??

---

#### STONITH and Leader Detection
Background context: STONITH stands for Shoot The Other Node In The Head, which is a method of fencing to ensure only one node is active. This approach helps prevent split brain scenarios where two nodes believe they are the leader. If not carefully designed, automatic failover mechanisms can result in both nodes being shut down.
:p What is STONITH and how does it help in distributed systems?
??x
STONITH is a mechanism used to ensure that only one node remains active in a distributed system by forcibly shutting down an unresponsive or rogue node. This helps prevent split brain scenarios where multiple nodes might think they are the leader, leading to data inconsistencies.
For example, if Node A and Node B both believe they are leaders due to network issues, STONITH can be used to shut down one of them, ensuring only one node continues operation.

```java
// Pseudocode for a simple STONITH implementation
public class STONITHHandler {
    private Map<String, Boolean> nodesStatus;

    public void detectAndFence(String nodeId) {
        if (nodesStatus.get(nodeId).isUnresponsive()) {
            // Perform fencing to shut down the node
            shutdownNode(nodeId);
        }
    }

    private void shutdownNode(String nodeId) {
        // Logic to forcefully shut down the node
        System.out.println("Shutting down " + nodeId + " due to unresponsiveness.");
    }
}
```
x??

---

#### Split Brain Scenario
Background context: A split brain scenario occurs when two nodes in a distributed system both believe they are the leader. Without proper conflict resolution, this can lead to data loss or corruption.
:p What is a split brain scenario and why is it dangerous?
??x
A split brain scenario happens when two nodes in a distributed system erroneously think they are the leaders simultaneously due to network partitioning or other issues. This situation is dangerous because both nodes might accept writes, leading to inconsistencies and data loss if there's no mechanism to resolve conflicts.

For example:
- Node A and Node B are part of a cluster.
- Due to a network glitch, communication between them is lost.
- Both nodes continue running assuming they are the leader.
- If both nodes accept writes without coordination, this can lead to conflicting states that cannot be resolved, causing data corruption.

x??

---

#### Timeout for Leader Detection
Background context: Determining when a node should be declared dead involves setting an appropriate timeout. A longer timeout means slower recovery in case of failure but shorter timeouts may trigger unnecessary failovers.
:p What factors are considered when determining the right timeout for leader detection?
??x
Factors to consider include:
- **Failure Recovery Time**: Longer timeouts allow more time for nodes to recover from temporary issues like network latency or load spikes.
- **Unnecessary Failovers**: Shorter timeouts might cause failovers due to temporary conditions, leading to suboptimal performance.

For example, a system might choose a timeout based on expected network conditions and node responsiveness:
```java
public class LeaderTimeoutConfig {
    private int timeoutMs;

    public void configure(int networkLatency) {
        // Adjust the timeout based on expected network latency
        if (networkLatency < 100) {
            timeoutMs = 500; // Shorter timeout for low-latency networks
        } else {
            timeoutMs = 2000; // Longer timeout for high-latency networks
        }
    }

    public int getTimeout() {
        return timeoutMs;
    }
}
```
x??

---

#### Statement-Based Replication
Background context: Statement-based replication logs every write request executed by the leader and sends these statements to followers. While simple, it can lead to issues with non-deterministic functions, autoincrement columns, and side effects.
:p What is statement-based replication and what are its limitations?
??x
Statement-based replication involves logging all SQL statements (INSERT, UPDATE, DELETE) executed by the leader and sending them to followers. Each follower then parses and executes these statements as if they were received from a client.

Limitations include:
- Non-deterministic functions like `NOW()` or `RAND()` can produce different values on each replica.
- Autoincrement columns may require specific order of execution across replicas.
- Side effects, such as triggers or stored procedures, might not be deterministic between replicas.

For example, handling non-deterministic functions by replacing them with fixed return values:
```java
public class StatementReplicator {
    private Map<String, String> functionReplacements;

    public void logStatement(String statement) {
        // Replace non-deterministic functions with fixed values
        if (statement.contains("NOW()")) {
            statement = statement.replace("NOW()", "'2023-10-01'");
        }
        // Add more replacements as needed
    }

    public String getReplacementFunction(String funcName) {
        return functionReplacements.getOrDefault(funcName, funcName);
    }
}
```
x??

---

#### Write-Ahead Log (WAL) Shipping
Background context: WAL shipping logs all writes to an append-only log that followers can process to build replicas. This method is used in PostgreSQL and Oracle but can be coupled closely with the storage engine.
:p What is WAL shipping and its main advantages?
??x
Write-ahead logging involves maintaining a write-ahead log (WAL) where every database write is first appended before being applied to the main data files. Followers read this log to build replicas, ensuring data consistency.

Advantages include:
- Replication can be decoupled from the storage engine's internals.
- Allows backward compatibility between leader and follower versions.

For example, a simple WAL shipping mechanism in PostgreSQL:
```java
public class WALShippingHandler {
    private List<WriteOperation> logEntries;

    public void appendLogEntry(WriteOperation entry) {
        // Append new write operation to the log
        logEntries.add(entry);
    }

    public void processLogOnFollower() {
        for (WriteOperation entry : logEntries) {
            entry.applyToDatabase(); // Apply operations on the follower
        }
    }

    static class WriteOperation {
        private String sql;
        
        public void applyToDatabase() {
            // Logic to apply the write operation
        }
    }
}
```
x??

---

#### Logical (Row-Based) Log Replication
Background context: Logical log replication logs writes at a row level, decoupling from the storage engine's internals. This allows more flexibility in running different versions of database software on leader and follower nodes.
:p What is logical (row-based) log replication and why is it useful?
??x
Logical log replication involves logging write operations at the row-level rather than at the statement level, allowing decoupling from the storage engine's internal representations. This method is used in MySQL with its binary logs.

Benefits include:
- Flexibility to run different versions of database software on leader and follower nodes.
- Easier parsing by external applications for data transfer or auditing purposes.

For example, a logical log entry for row-based replication:
```java
public class LogicalLogEntry {
    private String tableName;
    private int opType; // INSERT, DELETE, UPDATE
    private Map<String, Object> values;

    public void applyToDatabase() {
        if (opType == INSERT) {
            // Insert new row
        } else if (opType == DELETE) {
            // Delete existing row by primary key
        } else if (opType == UPDATE) {
            // Update existing row with new values
        }
    }

    public void logInsert(String tableName, Map<String, Object> values) {
        this.tableName = tableName;
        this.values = values;
        opType = INSERT;
    }

    public void logDelete(String tableName, int primaryKey) {
        this.tableName = tableName;
        this.opType = DELETE;
    }

    public void logUpdate(String tableName, Map<String, Object> beforeValues, Map<String, Object> afterValues) {
        this.tableName = tableName;
        this.values = afterValues; // Use new values
        opType = UPDATE;
    }
}
```
x??

#### Change Data Capture (CDC)
Background context: Change data capture is a technique used to track and record changes made to data in a database, which can be used for offline analysis or building custom indexes. This technique allows developers to react to and process these changes as they happen.

:p What is change data capture?
??x
Change data capture (CDC) is a method that tracks and records the changes made to data in a database system. These changes are then available for various purposes such as offline analysis or building custom indexes, enabling real-time processing of data modifications.
x??

---

#### Trigger-based Replication Overview
Background context: In some cases, the default replication mechanisms provided by databases may not suffice due to specific requirements like replicating only certain subsets of data, handling different types of database systems, or implementing conflict resolution logic. Trigger-based replication allows more flexibility by using application code to handle these scenarios.

:p What is trigger-based replication?
??x
Trigger-based replication involves the use of triggers and stored procedures within the application layer to replicate data changes from one system to another. Triggers are pieces of custom application code that automatically execute when a data change (write transaction) occurs in the database, logging these changes for external processes to read and apply necessary logic.

:p How does trigger-based replication work?
??x
Trigger-based replication works by having triggers register custom application code that executes on data changes. These triggers can log changes into separate tables, which are then read by an external process where any required application logic is applied before replicating the data change to another system.

Example:
```java
// Pseudocode for a trigger in SQL

CREATE TRIGGER log_changes
AFTER INSERT ON customers
FOR EACH ROW
BEGIN
    INSERT INTO change_log (customer_id, action, timestamp)
    VALUES (NEW.customer_id, 'INSERT', NOW());
END;
```
This example shows how a `log_changes` trigger logs insert operations into the `change_log` table.

x??

---

#### Databus for Oracle and Bucardo for Postgres
Background context: For more flexible replication scenarios, tools like Databus for Oracle and Bucardo for PostgreSQL use triggers and stored procedures to replicate data changes. These tools operate at the application layer, providing greater flexibility but with increased overhead.

:p How do Databus for Oracle and Bucardo for Postgres work?
??x
Databus for Oracle and Bucardo for PostgreSQL work by using triggers within the database to capture changes made to the data. The triggers log these changes into a separate table from which an external process can read and apply necessary application logic, replicating the changes to another system.

:p What are some benefits of using Databus or Bucardo?
??x
Benefits include increased flexibility in handling different types of database systems, replicating only specific subsets of data, and implementing custom conflict resolution logic. However, these methods come with greater overhead and a higher risk of bugs compared to built-in replication mechanisms.

x??

---

#### Problems with Replication Lag: Read-Your-Writes Consistency
Background context: Asynchronous replication can lead to issues where reads from replicas may not reflect the latest writes, causing apparent inconsistencies in applications. Ensuring read-after-write consistency (also known as read-your-writes) is crucial for maintaining user trust and application reliability.

:p What is read-after-write consistency?
??x
Read-after-write consistency ensures that if a user reloads the page after submitting data, they will always see any updates they submitted themselves. This guarantee provides assurance to users that their input has been saved correctly, even when reads are served from replicas.

:p How can we implement read-after-write consistency in a system with leader-based replication?
??x
Implementing read-after-write consistency involves techniques such as reading data the user may have modified from the leader and other data from followers. Criteria for deciding whether to read from the leader or follower include:
- Always reading a user's own profile from the leader.
- Tracking the last update time and making all reads from the leader within one minute of the last update.
- Monitoring replication lag on followers and preventing queries on any follower more than one minute behind the leader.

Example code snippet in Java:
```java
public class ReadConsistencyService {
    public boolean shouldReadFromLeader(String userId) {
        // Check if user's profile has been updated recently
        long lastUpdateTimestamp = getLastUpdateTimestamp(userId);
        long currentTime = System.currentTimeMillis();
        
        return (currentTime - lastUpdateTimestamp) < 60000;
    }
}
```
This example checks the timestamp of the most recent update for a user and ensures that reads are always from the leader if the update was within one minute.

x??

---

