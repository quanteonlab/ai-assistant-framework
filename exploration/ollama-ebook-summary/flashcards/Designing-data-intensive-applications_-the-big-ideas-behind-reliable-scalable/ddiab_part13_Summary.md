# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 13)

**Starting Chapter:** Summary

---

---
#### Akka's Default Serialization
Akka by default uses Java’s built-in serialization, which does not provide forward or backward compatibility. This can lead to issues during rolling upgrades as different versions of the application might need to communicate.
:p How does Akka handle data serialization by default?
??x
By default, Akka uses Java’s built-in serialization mechanism, which lacks the ability for backward and forward compatibility. This means that if different versions of a service are running, older code might not be able to read newer data or vice versa.
```java
// Example of how Java's serialization works in Akka
ActorRef sender = system.actorOf(Props.create(MyActor.class));
sender.tell("message", null);
```
x??
---

#### Protocol Buffers for Forward/Backward Compatibility
Protocol Buffers can be used as an alternative to Java’s built-in serialization, providing the ability to perform rolling upgrades. This is because Protocol Buffers support schema evolution.
:p Can you explain how using Protocol Buffers helps in rolling upgrades with Akka?
??x
Using Protocol Buffers for data encoding allows for better compatibility between different versions of a service running on different nodes. The protocol buffers schema can evolve over time, meaning that new code can read old data and vice versa without issues.
```java
// Example of defining a message in Protocol Buffers
message User {
  required string name = 1;
  optional int32 id = 2;
}
```
x??
---

#### Orleans Data Encoding Format
Orleans, by default, uses a custom encoding format that does not support rolling upgrades. To achieve rolling upgrades, one would need to set up a new cluster and migrate traffic over.
:p What is the default behavior of Orleans regarding data encoding and rolling upgrades?
??x
By default, Orleans employs a custom encoding format for data which does not enable smooth rolling upgrades. For implementing rolling upgrades, a new cluster must be established, traffic needs to be migrated from the old cluster to the new one, and then the old cluster can be shut down.
```java
// Example of setting up Orleans
var manager = new GrainManager();
manager.SetConnectionString(connectionString);
```
x??
---

#### Erlang OTP Schema Evolution Challenges
In Erlang OTP, making changes to record schemas is challenging. Rolling upgrades are possible but require careful planning. The introduction of an experimental map data type might simplify this process in the future.
:p What difficulties does Erlang OTP face when it comes to evolving its record schemas for rolling upgrades?
??x
Erlang OTP makes changing record schemas difficult, which complicates implementing rolling upgrades. While rolling upgrades are feasible, they need careful planning. An experimental new map data type, introduced in Erlang R17 (2014), might simplify schema evolution.
```erlang
% Example of a record in Erlang
-record(user, {name :: string(), id :: integer()}).
```
x??
---

#### Data Encoding Formats and Compatibility
Several encoding formats are discussed, each with their own compatibility properties. Programming language-specific encodings often lack backward/forward compatibility, while schema-driven binary formats like Thrift and Protocol Buffers provide clear definitions for these.
:p What are the key differences between programming language-specific encodings and schema-driven binary formats in terms of backward/forward compatibility?
??x
Programming language-specific encodings are tied to a single programming language and often fail to offer forward or backward compatibility. In contrast, schema-driven binary formats like Thrift and Protocol Buffers allow for compact and efficient encoding with well-defined forward and backward compatibility semantics.
```java
// Example of using Protocol Buffers in Java
public class User {
  public static final MessageLite.BUILDER = User.getDefaultInstance().toBuilder();
}
```
x??
---

#### Modes of Dataflow in Distributed Systems
Data flows through various mechanisms like databases, RPCs, and asynchronous message passing. Each scenario requires different handling for encoding and decoding data.
:p In what ways do the modes of dataflow (databases, RPC, asynchronous messaging) affect how data is encoded and decoded?
??x
In databases, data is encoded by a process writing to it and decoded by one reading from it. For RPCs and REST APIs, clients encode requests, servers decode them, generate responses, and then the client decodes the response. Asynchronous message passing involves nodes sending messages that are encoded by senders and decoded by recipients.
```java
// Example of asynchronous messaging in Java with Akka
ActorRef sender = system.actorOf(Props.create(MyActor.class));
sender.tell("message", null);
```
x??
---

#### Nonuniform Memory Access (NUMA)
Background context explaining NUMA. The architecture of large machines often involves non-uniform memory access, where different CPUs have closer and faster access to certain parts of memory compared to others. This can impact performance if not properly managed.

:p What is NUMA and how does it affect a system's performance?
??x
NUMA stands for Nonuniform Memory Access. In systems with multiple CPUs, some banks of memory are physically located closer to one CPU than another. As a result, the access time to these memories can vary significantly. This non-uniformity can lead to inefficiencies if not managed correctly, because a task running on a certain CPU might be more efficient if it accesses nearby memory.

For example, in a NUMA architecture, you would want to ensure that tasks are scheduled and data is allocated so that each CPU primarily accesses the memory banks closest to it. This requires careful partitioning and workload balancing.
??x
The question about this concept was: What is NUMA and how does it affect a system's performance?
x??

---

#### Network Attached Storage (NAS) or Storage Area Network (SAN)
Background context explaining NAS and SAN, including their use cases for scaling to higher loads. NAS allows file-level access over a network, while SAN provides block-level storage via a dedicated network.

:p How do NAS and SAN work in the context of scaling a data system?
??x
Network Attached Storage (NAS) and Storage Area Network (SAN) are two common ways to scale storage for data systems.

- **Network Attached Storage (NAS)**: Provides file-level access over a network. It is easier to set up than SAN but may have performance limitations because it relies on the network for all operations.
  
- **Storage Area Network (SAN)**: Offers block-level access through a dedicated high-speed network. It allows multiple servers to access shared storage, providing better performance and scalability compared to NAS.

Both NAS and SAN can be used to scale data systems by offloading storage from individual machines onto centralized storage solutions, which can handle higher loads more efficiently.
??x
The question about this concept was: How do NAS and SAN work in the context of scaling a data system?
x??

---

#### Shared-Memory Architecture
Background context on shared-memory architecture, including its limitations such as cost growth and scalability issues. It is commonly used when needing to scale within a single machine.

:p What is shared-memory architecture, and what are its main drawbacks?
??x
Shared-memory architecture involves combining multiple CPUs, RAM chips, and disks under one operating system to create a system where any CPU can access any part of the memory or disk via a fast interconnect. While this approach allows all components to be treated as a single machine, it comes with several limitations:

1. **Cost Growth**: The cost does not grow linearly; doubling the number of CPUs and resources typically costs significantly more than double.
2. **Scalability Limits**: Due to bottlenecks, even a machine twice the size may not handle twice the load.

While high-end machines offer hot-swappable components for some fault tolerance, they are still limited in terms of geographic location since all components must reside within the same physical space.

```java
public class SharedMemoryExample {
    private static final int NUM_CPUS = 4; // Example with 4 CPUs

    public void process() {
        // Accessing shared memory
        for (int i = 0; i < NUM_CPUS; i++) {
            // Process data using shared resources
        }
    }
}
```
The example shows a simple method where each CPU processes data using shared resources. However, it highlights the complexity in managing such a system.
??x
The question about this concept was: What is shared-memory architecture, and what are its main drawbacks?
x??

---

#### Shared-Disk Architecture
Background context on shared-disk architectures, which use multiple machines with independent CPUs and RAM but share data across an array of disks connected via a fast network. This approach can handle some data warehousing workloads but has limitations due to contention and locking overhead.

:p What is the main difference between shared-memory and shared-disk architectures?
??x
The main difference between shared-memory and shared-disk architectures lies in how they manage resources:

- **Shared-Memory Architecture**: All CPUs, RAM, and disks are connected under one operating system. Any CPU can access any part of memory or disk via a fast interconnect. However, the cost grows faster than linearly, and scalability is limited due to bottlenecks.

- **Shared-Disk Architecture**: Multiple machines with independent CPUs and RAM share data on an array of disks that are connected via a fast network. This approach allows for higher storage capacity but introduces challenges like contention and overhead from locking mechanisms, limiting its scalability.

In summary, shared-memory architecture is better suited for applications requiring high performance within a single machine, while shared-disk architectures are more suitable for distributed environments with shared data access.
??x
The question about this concept was: What is the main difference between shared-memory and shared-disk architectures?
x??

---

#### Shared-Nothing Architecture
Background context on shared-nothing architectures, which involve distributing tasks across multiple nodes that operate independently. This approach is popular due to its flexibility in price/performance ratio and ability to handle multi-region deployments.

:p How does a shared-nothing architecture differ from shared-memory or shared-disk architectures?
??x
A shared-nothing architecture involves each node running the database software independently, using its own CPUs, RAM, and disks. Coordination between nodes is done at the software level over a conventional network. This approach offers several advantages:

1. **Cost Efficiency**: No special hardware is required, allowing you to use machines with the best price/performance ratio.
2. **Scalability**: You can distribute data across multiple geographic regions, reducing latency and increasing availability.
3. **Flexibility**: Suitable for small companies as well as large-scale deployments.

However, it also introduces additional complexity for applications due to distributed coordination and may limit the expressive power of the data models you can use.

```java
public class SharedNothingExample {
    private static final int NUM_NODES = 4; // Example with 4 nodes

    public void distributeData() {
        // Assign tasks to different nodes
        for (int i = 0; i < NUM_NODES; i++) {
            Node node = getNode(i);
            node.processData();
        }
    }

    private Node getNode(int id) {
        // Logic to get the appropriate node based on ID
        return new Node(id);
    }
}
```
The example illustrates a basic method for distributing tasks across nodes, demonstrating how coordination can be handled in such an architecture.
??x
The question about this concept was: How does a shared-nothing architecture differ from shared-memory or shared-disk architectures?
x??

---

#### Replication vs. Partitioning
Background context on replication and partitioning as common ways to distribute data across multiple nodes, often used together for redundancy and performance improvements.

:p What are the two main methods of distributing data in distributed systems: replication and partitioning? How do they differ?
??x
In distributed systems, data can be distributed using two primary methods:

1. **Replication**: Keeping a copy of the same data on several different nodes potentially in different locations. Replication provides redundancy, ensuring that if some nodes are unavailable, the data can still be served from the remaining nodes. It also helps improve performance by allowing reads to be distributed.

2. **Partitioning (Sharding)**: Splitting a large database into smaller subsets called partitions, assigning each partition to a different node. This approach scales well for read-heavy workloads and allows for efficient load balancing.

These methods are often used together. For example, in Figure II-1, the database is split into two partitions, with each partition replicated across multiple nodes.

```java
public class ReplicationPartitioningExample {
    private static final int NUM_PARTITIONS = 2;
    private static final int REPLICA_COUNT_PER_PARTITION = 3;

    public void distributeData() {
        for (int i = 0; i < NUM_PARTITIONS; i++) {
            // Assign partitions to nodes
            for (int j = 0; j < REPLICA_COUNT_PER_PARTITION; j++) {
                Node node = getNode(i * REPLICA_COUNT_PER_PARTITION + j);
                node.storePartition(i);
            }
        }
    }

    private Node getNode(int id) {
        // Logic to get the appropriate node based on ID
        return new Node(id);
    }
}
```
The example demonstrates how data can be both replicated and partitioned across nodes, illustrating a common approach in distributed systems.
??x
The question about this concept was: What are the two main methods of distributing data in distributed systems: replication and partitioning? How do they differ?
x??

---

#### Replication Basics
Replication involves keeping a copy of the same data on multiple machines connected via a network. This is done to reduce latency, increase availability, and scale out read throughput.

:p What are the primary reasons for using replication?
??x
The main reasons for using replication include:
- Keeping data geographically close to users to reduce latency.
- Ensuring system availability even if some parts fail.
- Scaling out by allowing more machines to serve read queries, thereby increasing read throughput.
x??

---

#### Single-Leader Replication
In this approach, one replica is designated as the leader. All writes must be sent to the leader, which then replicates them to other followers.

:p How does single-leader replication work?
??x
Single-leader replication works as follows:
1. One of the replicas is chosen as the leader.
2. Clients send write requests to this leader.
3. The leader writes the new data locally and sends a log of changes (replication stream) to other followers.
4. Followers update their local copies based on these logs.

Example flow in pseudocode:
```plaintext
if (isLeader()) {
    // Process write request
    applyChangesToLocalStorage();
    sendReplicationLogToFollower();
} else if (isFollower()) {
    // Wait for leader to push changes, then apply locally.
}
```
x??

---

#### Synchronous vs. Asynchronous Replication
Synchronous replication waits until the follower confirms receipt of a write before reporting success to the client, ensuring up-to-date and consistent data. Asynchronous replication sends the message without waiting.

:p What is the difference between synchronous and asynchronous replication?
??x
The key difference lies in how they handle writes:
- Synchronous: The leader waits for confirmation from followers that they received the write request before reporting success to the client.
- Asynchronous: The leader sends the write request immediately, regardless of whether followers confirm receipt.

Example flow in pseudocode (synchronous):
```plaintext
if (isLeader()) {
    // Process write request
    applyChangesToLocalStorage();
    waitForFollowerConfirmation();  // Wait for confirmation from all followers
    sendSuccessToClient();         // Report success to client after confirmation
} else if (isFollower()) {
    // Receive and process changes from leader
}
```

Example flow in pseudocode (asynchronous):
```plaintext
if (isLeader()) {
    // Process write request
    applyChangesToLocalStorage();
    sendReplicationLogToFollower();  // Send without waiting for confirmation
} else if (isFollower()) {
    // Receive and process changes from leader
}
```
x??

---

#### Trade-offs in Replication
There are trade-offs to consider when setting up replication, such as choosing between synchronous or asynchronous methods, handling failed replicas, and ensuring data consistency.

:p What are some important considerations when implementing replication?
??x
Important considerations include:
- **Synchronous vs. Asynchronous**: Choosing based on latency requirements and failure tolerance.
- **Handling Failed Replicas**: Strategies for dealing with unreplied followers to ensure data availability.
- **Data Consistency**: Ensuring that writes are processed in the correct order across all replicas.

Example configuration decision in Java pseudocode:
```java
if (replicationMode == SYNCHRONOUS) {
    // Implement logic to wait for follower confirmations before reporting success
} else if (replicationMode == ASYNCHRONOUS) {
    // Send messages without waiting for responses
}
```
x??

---

