# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 26)

**Starting Chapter:** Summary

---

#### Network Packet Loss and Delay
In distributed systems, network packet loss and arbitrary delays are common issues. These problems can occur during message transmission over a network. The reliability of communication between nodes cannot be guaranteed due to these uncertainties.

:p What is an example of a problem that can arise from network packet loss and delay in a distributed system?
??x
An example of a problem that can arise is when a node sends a message but does not receive the reply, making it uncertain whether the message was successfully delivered. This ambiguity can lead to incorrect assumptions about the state or behavior of other nodes.
x??

---

#### Clock Synchronization Issues
Even with Network Time Protocol (NTP) setup, clock synchronization between nodes in a distributed system can be problematic. Nodes may experience significant time discrepancies, unexpected jumps, or have unreliable measures of their own clock errors.

:p How does NTP help in maintaining clock synchronization among nodes?
??x
NTP helps to synchronize the clocks across different nodes by periodically adjusting them to match an accurate external reference time source. However, despite its efforts, issues such as significant time discrepancies, sudden jumps, and inaccurate error intervals can still occur.
x??

---

#### Partial Failures in Distributed Systems
Partial failures, where a process may pause for a substantial amount of time or be declared dead by other nodes before coming back to life, are critical challenges in distributed systems. These partial failures can manifest due to various reasons such as garbage collection pauses.

:p What is the impact of partial failures on processes in a distributed system?
??x
Partial failures can lead to unpredictable behavior where a process might pause unexpectedly, be incorrectly flagged as dead by other nodes, and then resume execution without realizing it was paused. This unpredictability complicates fault tolerance and reliable operation within the system.
x??

---

#### Detecting Node Failures Using Timeouts
To handle partial failures, distributed systems often rely on timeouts to determine if a remote node is still available. However, this approach can lead to false positives or negatives due to network variability.

:p How does timeout-based failure detection work in distributed systems?
??x
Timeout-based failure detection works by setting a time limit for receiving a reply from another node. If the reply is not received within the timeout period, it's assumed that the node has failed. However, this method can incorrectly suspect nodes of crashing due to network delays or degraded states.
x??

---

#### Handling Degraded Node States
Degraded states, where a node functions at reduced capacity but continues to operate, pose additional challenges in distributed systems. Examples include network interfaces dropping to lower throughput rates unexpectedly.

:p What is an example of a scenario where a node might be considered "limping"?
??x
An example of a scenario is when a Gigabit network interface card suddenly drops its throughput to 1 Kb/s due to a driver bug, allowing the node to continue functioning but at a much reduced capacity.
x??

---

#### Consistency and Fault Tolerance in Distributed Systems

Background context explaining the concept. In distributed systems, ensuring that all nodes agree on a decision (consensus) or maintain a consistent state is crucial. However, achieving these goals can be challenging due to network delays, packet loss, and node failures.

If applicable, add code examples with explanations.
:p What are the key challenges in building fault-tolerant distributed systems?
??x
The key challenges include network faults (lost, reordered, duplicated packets), approximate clocks, and nodes that can pause or crash at any time. These issues make it difficult to ensure consistent behavior across all nodes.

For example, consider a scenario where multiple nodes are trying to elect a leader in the presence of network delays:
```java
public class Election {
    private Node currentLeader;
    private Set<Node> nodes = new HashSet<>();

    public void addNode(Node node) { this.nodes.add(node); }
    
    public void startElection() {
        for (Node node : nodes) {
            if (!node.isAlive()) continue; // Skip crashed nodes
            if (node == currentLeader) continue; // Avoid self-election

            // Logic to elect a new leader based on the consensus algorithm
        }
    }
}
```
x??

---

#### Linearizability

Background context explaining the concept. Linearizability is one of the strongest consistency models in distributed systems, where every operation appears instantaneous and atomic. This means that if two operations appear to happen at different times from a client's perspective, they can be reordered without affecting the behavior.

:p What is linearizability?
??x
Linearizability ensures that all operations on a replicated data store appear to have happened instantaneously and atomically in some order. Operations are sequenced as if they executed one after another, even though in reality, distributed systems may handle them asynchronously.

For example:
```java
public class LinearizableOperation {
    private volatile int value;

    public void write(int newValue) { // Write operation
        synchronized (this) {
            value = newValue;
            notifyAll(); // Notify all waiting readers and writers
        }
    }

    public int read() { // Read operation
        synchronized (this) {
            while (value == -1) wait(); // Wait until the value is set
            return value;
        }
    }
}
```
x??

---

#### Event Ordering in Distributed Systems

Background context explaining the concept. Ensuring that events are ordered correctly can be crucial for maintaining consistency and avoiding race conditions in distributed systems. Causality ensures that if event A caused event B, then A must happen before B.

:p What is causality?
??x
Causality is a fundamental concept in distributed systems where an event A causes another event B if there exists a causal relationship between them. This means that the occurrence of A logically precedes and influences B.

For example:
```java
public class EventOrdering {
    private Map<String, List<Event>> events = new HashMap<>();

    public void addEvent(String key, Event event) {
        if (!events.containsKey(key)) events.put(key, new ArrayList<>());
        events.get(key).add(event);
    }

    public boolean isCausal(Event a, Event b) {
        for (List<Event> list : events.values()) {
            if (list.contains(a) && list.contains(b)) {
                int indexA = list.indexOf(a);
                int indexB = list.indexOf(b);
                return indexA < indexB; // Check causality based on order
            }
        }
        return false;
    }
}
```
x??

---

#### Distributed Transactions and Consensus

Background context explaining the concept. Achieving consensus in a distributed system where nodes may fail or experience delays is challenging but essential for reliable distributed operations. Distributed transactions often require solving the consensus problem, ensuring that all nodes agree on the outcome of an operation.

:p What are the challenges in achieving consensus in distributed systems?
??x
The main challenges include network delays, node failures (crashes), packet loss, and approximate clocks. These issues make it difficult to ensure that all nodes can reach agreement on a decision without any of them failing or misbehaving.

For example:
```java
public class Consensus {
    private Set<Node> nodes = new HashSet<>();
    
    public void addNode(Node node) { this.nodes.add(node); }
    
    public boolean achieveConsensus(int value) {
        int requiredVotes = (nodes.size() / 2) + 1;
        Map<Node, Boolean> votes = new HashMap<>();

        for (Node node : nodes) {
            // Send request to each node
            if (node.sendRequest(value)) {
                votes.put(node, true); // Node agreed with the value
            } else {
                votes.put(node, false); // Node failed or disagreed
            }
        }

        return votes.values().stream().allMatch(vote -> vote); // Check if all nodes agreed
    }
}
```
x??

---

#### Linearizability Overview
Background context: In an eventually consistent database, different replicas might return different answers to the same question at the same time. To avoid this confusion and ensure all clients see a single copy of data with up-to-date values, linearizability is introduced. It makes systems appear as if there were only one replica.
:p What is linearizability?
??x
Linearizability ensures that operations on the database are atomic and consistent, making it seem like there is only one copy of the data. This means every read operation will return the most recent value written by any client. It provides a recency guarantee where all clients reading from the database must see the latest value.
---
#### Example of Non-linearizable System
Background context: The example given involves two users, Alice and Bob, checking a sports website for the final score of the 2014 FIFA World Cup. Alice sees the result first, but when Bob refreshes his page slightly later, he gets an outdated version due to replication lag.
:p What happens in this scenario that violates linearizability?
??x
In this scenario, Bob's request returns a stale result, showing that the game is still ongoing even after the final score has been announced. This contradicts Alice’s immediate update and breaks the recency guarantee required by linearizability, as Bob should have seen an up-to-date result.
---
#### Making a System Linearizable
Background context: To achieve linearizability, all read operations must reflect the most recent write operation that has completed. This means that after a client successfully writes to the database, subsequent reads from any client must return that value immediately without delay or stale data.
:p How does linearizability ensure consistency in distributed systems?
??x
Linearizability ensures consistency by treating each atomic operation as if it were performed sequentially on a single copy of the data. This means every read operation retrieves the latest value written, regardless of which replica is being queried. The system must guarantee that the recency property holds for all operations.
---
#### Register in Linearizable Systems
Background context: In distributed systems literature, a register (like key x) refers to a single piece of data that can be read from and written to by different clients concurrently. This concept is analogous to variables in programming languages but operates across multiple nodes in a network.
:p What does the term "register" refer to in linearizable databases?
??x
In linearizable databases, a register (like key x) represents a piece of data that can be accessed and modified by multiple clients simultaneously. It acts as if there were only one copy of the data being manipulated, ensuring all operations are atomic and consistent. This concept is crucial for understanding how linearizability maintains consistency across distributed systems.
---
#### Linearizable System Example
Background context: Figure 9-2 illustrates a scenario where three clients (C1, C2, and C3) concurrently read and write to the same key x in a linearizable database. Each client's operations are treated as if they were performed sequentially on a single copy of the data.
:p How do concurrent reads and writes behave in a linearizable system?
??x
In a linearizable system, even though multiple clients can read and write concurrently, each operation is treated as if it were executed one at a time on a single replica. This means that after C1 writes to key x, any subsequent read by C2 or C3 will reflect this latest value immediately. The system ensures a recency guarantee for all operations.
---
#### Implementing Linearizability
Background context: Achieving linearizability involves careful handling of concurrent requests to ensure that the order of operations is preserved as if they were executed sequentially on a single node. This often requires sophisticated algorithms and coordination mechanisms.
:p What techniques are used to implement linearizability?
??x
Implementing linearizability typically involves using synchronization primitives, such as locks or semaphores, to manage concurrent access to shared data. Additionally, ensuring that writes are replicated before reads can see the updated value is crucial. Techniques like two-phase commit or more advanced consensus algorithms (e.g., Raft) might be used to coordinate these operations.
---

---
#### Global Clock Assumption

Background context: The diagram assumes a global clock for analysis purposes, even though real systems typically lack accurate clocks. This assumption helps in analyzing distributed algorithms without actual access to an accurate global clock.

:p What is the global clock assumption used for in this context?
??x
The global clock assumption simplifies the analysis of distributed algorithms by pretending that all clients and servers have a synchronized view of time, which aids in understanding linearizability. However, real-world systems do not have such precise clocks; instead, they rely on approximations like quartz oscillators and Network Time Protocol (NTP).

```java
// Example of using Quartz for timing
import org.quartz.*;
import org.quartz.impl.StdSchedulerFactory;

public class ClockExample {
    public static void main(String[] args) throws Exception {
        Scheduler scheduler = StdSchedulerFactory.getDefaultScheduler();
        JobDetail job = newJob(MyJob.class)
                .withIdentity("job1", "group1")
                .build();

        Trigger trigger = newTrigger()
                .withIdentity("trigger1", "group1")
                .startNow()
                .withSchedule(simpleSchedule()
                        .withIntervalInSeconds(5)
                        .repeatForever())
                .build();

        scheduler.scheduleJob(job, trigger);
        scheduler.start();
    }
}
```
x??

---
#### Regular Register

Background context: A regular register is a data structure where read operations may return either the old or new value if they are concurrent with a write operation. This behavior is common in distributed systems and affects how clients perceive the state of shared variables.

:p What is a regular register, and what behavior does it exhibit?
??x
A regular register is a type of memory location in a distributed system where reads may return either the old or new value if they are concurrent with a write. This means that during a write operation, any read requests that overlap might see the previous state (old value) or the updated state (new value).

```java
// Pseudocode for a regular register implementation
class RegularRegister {
    private int value;

    public void write(int newValue) {
        // Simulate writing to the register
        value = newValue;
    }

    public int read() {
        // Read operation might return old or new value based on concurrency with writes
        return value; // Simplified logic for illustration
    }
}
```
x??

---
#### Linearizability

Background context: The example discusses a scenario where reads and writes to a register can cause inconsistencies due to concurrent operations. To achieve linearizability, additional constraints are needed to ensure that all read operations see a consistent state.

:p What is the issue with regular registers in terms of concurrency?
??x
The issue with regular registers is that when reads and writes overlap, they can return inconsistent values, leading to potential race conditions where readers might observe outdated or new data unexpectedly. This inconsistency violates the expectation of linearizability, which requires operations to appear as if they are executed sequentially.

```java
// Pseudocode for read/write operations in a register
class RegularRegister {
    private int value;

    public void write(int newValue) {
        value = newValue;
    }

    public int read() {
        // Read might return old or new value based on when it overlaps with the write operation
        return value; // Simplified logic for illustration
    }
}
```
x??

---
#### Linearizability Constraint

Background context: The text explains that to achieve linearizability, additional constraints are needed. Specifically, reads concurrent with writes must not see inconsistent values but instead must see a consistent state.

:p What constraint is added to ensure linearizability in the example?
??x
To ensure linearizability, an additional constraint is added such that all read operations that overlap with write operations must return either the old value or the new value consistently. This prevents readers from seeing inconsistent data during concurrent writes and ensures a "single copy of the data" model.

```java
// Pseudocode for enforcing linearizability in reads/writes
class LinearizableRegister {
    private int value;

    public void write(int newValue) {
        // Write operation updates the register value
        value = newValue;
    }

    public synchronized int read() {
        // Ensure that reads see a consistent state by avoiding concurrent modifications
        return value; // Simplified logic for illustration
    }
}
```
x??

---

#### Linearizability Concept
Linearizability is a property of concurrent systems where each operation appears to have executed atomically at some point between its start and end. This means that if one client's read returns the new value, all subsequent reads must also return the new value until it is overwritten.

:p What does linearizability ensure in concurrent systems?
??x
Linearizability ensures that operations appear to execute in a sequential order as if they were executed one after another, even when multiple clients are performing them concurrently. This property guarantees that once a read operation returns a new value, all subsequent reads will return the same updated value until it is overwritten.
x??

---
#### Read and Write Operations
In linearizable systems, writes to a shared variable must be visible to all subsequent reads after they have occurred. The timing of these operations is crucial for maintaining consistency.

:p How does linearizability affect read and write operations in concurrent systems?
??x
Linearizability ensures that once a write operation sets a new value to a register, all subsequent reads (whether by the same client or another) must return the new value immediately after the write has completed. This means there cannot be any intermediate states where a read returns an old value before the write is fully executed.
x??

---
#### Atomic Compare-and-Swap Operation
The `cas(x, vold, vnew)` operation allows for atomic comparison and update of a register's value based on its current state.

:p What does the `cas` operation do?
??x
The `cas` (Compare-and-Swap) operation checks if the current value of the variable $x $ is equal to$v_{old}$. If true, it atomically sets $ x$to $ v_{new}$. Otherwise, the operation leaves the register unchanged and returns an error. This ensures that updates are done atomically without interfering with other concurrent operations.
x??

---
#### Timing Diagrams in Linearizable Systems
Timing diagrams help visualize when reads and writes take effect within a linearizable system. They illustrate how operations must be ordered to maintain consistency.

:p How does a timing diagram illustrate the behavior of linearizable systems?
??x
A timing diagram illustrates the sequence of events that occur during read and write operations in a linearizable system. It shows how each operation (read or write) appears to have taken effect atomically at some point in time, ensuring that subsequent reads return the latest value written by previous operations.

Example code for visualizing this with markers:
```java
public class TimingDiagram {
    public static void main(String[] args) {
        // Simulate timing of read and write operations
        System.out.println("Read A starts");
        Thread.sleep(100);  // Simulating time delay
        System.out.println("Write B starts");
        Thread.sleep(50);   // Simulating time delay
        System.out.println("Write C starts");
        Thread.sleep(30);   // Simulating time delay
        System.out.println("Read A ends with value 1");
    }
}
```
x??

---
#### Non-Linearizable Behavior Example
The example in the text shows a scenario where the final read by client B is not linearizable because it returns an outdated value.

:p Why is the final read by client B not considered linearizable?
??x
Client B's final read is not linearizable because it returned the old value 1, which was written after the current write operation. In a linearizable system, all subsequent reads must return the new value immediately after the write has completed, ensuring consistency and visibility of updates.

In this case, even though A wrote 1 first, B's read did not see that change until D had already set x to 0, then A set it back to 1. The database processed these operations in a different order than the requests were sent, violating the linearizable property.
x??

---

#### Atomic Compare-and-Swap (CAS)
Atomic compare-and-swap operations are used to ensure that a value is not concurrently changed by another client between reading and writing. This operation checks if the current value of a variable matches an expected value; if it does, it writes a new value. Otherwise, it fails without changing the state.

:p How can atomic compare-and-swap (CAS) help in managing concurrent updates?
??x
Atomic compare-and-swap helps manage concurrent updates by ensuring that only one operation can change the state of a variable at any given time. If multiple clients try to update the same value simultaneously, the CAS operation will succeed for only one client and fail for others. For example:
- Client B attempts to set `x = 2` using CAS when it reads `x` as `0`.
- Client C also attempts to set `x = 4` after reading `x` as `1`.
- If the database processes C’s request first, C will succeed (because the value of `x` was `1`), but B’s CAS operation will fail because the value changed from `0` to `2`.

```java
// Pseudocode for CAS operation in a concurrent environment
public boolean cas(Object xExpected, Object xNew) {
    if (this.value == xExpected) {
        this.value = xNew;
        return true; // Operation succeeded
    } else {
        return false; // Operation failed due to concurrency
    }
}
```
x??

---

#### Non-Linearizable Reads
Non-linearizable reads occur when a read operation returns a value that is not the latest written by any transaction. This can happen if another transaction writes a different value between the time of the first and second read.

:p How does a non-linearizable read violate the linearizability property?
??x
A non-linearizable read violates the linearizability property because it returns an outdated value that could be newer according to some other client. For instance, if Client A reads `x` as `4`, and then Client B attempts to read the same variable but gets a value of `2`, this is not allowed under linearizability. This is similar to the example in Figure 9-1 where Alice (Client B) cannot read an older value than Bob (Client A).

```java
// Example scenario in Java:
class NonLinearizableExample {
    private int x = 0;

    public void readAndWrite() {
        // Client C reads and writes 'x'
        x = 4; // written by client C

        // Client B tries to read 'x'
        System.out.println("Client B reads: " + x); // May print 2, violating linearizability
    }
}
```
x??

---

#### Serializability vs. Linearizability
Serializability and linearizability are two different consistency models in database systems.

:p What is the difference between serializability and linearizability?
??x
Serializability ensures that transactions behave as if they were executed one after another, even though they may overlap in time. It groups operations into transactions and guarantees that no transaction can see intermediate states of other transactions. Linearizability focuses on individual reads and writes to a register (a single object) ensuring that each read or write operation appears to be instantaneous.

For example, serializable snapshot isolation (SSI) is not linearizable because it provides consistent snapshots for reads but may return older values than the latest written by another transaction.

```java
// Example of Serializability in Java using two-phase locking:
class SerializableExample {
    private int x = 0;

    public void transactionOperation() {
        lock(x); // Acquire locks before any operation
        try {
            x = 4; // Write operation within a transaction
            // Simulate other operations
        } finally {
            unlock(x); // Release locks after transaction completion
        }
    }
}
```
x??

---

#### Linearizability in Practice
Linearizability is tested by recording the timings of all requests and responses to ensure they can be arranged into a valid sequential order.

:p How can one test if a system’s behavior is linearizable?
??x
Testing for linearizability involves recording the sequence of operations (requests and responses) and checking whether these operations can be reordered such that each operation appears as though it has completed instantaneously before any subsequent operations. This is computationally expensive but necessary to ensure that all reads and writes are consistent with a sequential order.

```java
// Example of logging request timings in Java:
class LinearizabilityTest {
    private List<Request> requests = new ArrayList<>();

    public void logRequest(Request r) {
        // Log the start time of each request
        long startTime = System.currentTimeMillis();
        requests.add(new Request(r, startTime));
    }

    public boolean isLinearizable() {
        Collections.sort(requests, Comparator.comparingLong(r -> r.startTime)); // Reorder based on actual timings

        for (int i = 1; i < requests.size(); i++) {
            if (!requests.get(i).canFollow(requests.get(i - 1))) { // Check order validity
                return false;
            }
        }
        return true;
    }
}

class Request {
    private Operation operation;
    private long startTime;

    public Request(Operation o, long start) {
        this.operation = o;
        this.startTime = start;
    }

    public boolean canFollow(Request previous) {
        // Check if current request logically follows the previous one based on timing
        return this.startTime > previous.startTime;
    }
}
```
x??

---

