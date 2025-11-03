# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 18)

**Starting Chapter:** Partitioning by Hash of Key

---

#### Partitioning Overview
Background context explaining partitioning. Grace Hopper's quote highlights the importance of breaking down databases to avoid limitations and support future needs. The main reason for partitioning is scalability, allowing data to be distributed across many disks and processors.

:p What is partitioning in database management?
??x
Partitioning involves breaking a large database into smaller ones to enhance performance and scalability. Each piece of data belongs to exactly one partition, which can be placed on different nodes in a shared-nothing cluster.
x??

---

#### Sharding Terminology
Explanation about the term "sharding" and how it is used in various databases.

:p What are some terms associated with partitioning in different databases?
??x
In MongoDB, Elassearch, and SolrCloud, partitioning is called sharding. In HBase, it's known as a region; in Bigtable, it’s a tablet; in Cassandra and Riak, it’s a vnode; and in Couchbase, it’s a vBucket.
x??

---

#### Partitioned Databases
Explanation about the main reason for wanting to partition data and how it can be achieved.

:p Why is partitioning data important?
??x
Partitioning is crucial for scalability. By breaking large datasets into smaller partitions, each node can handle queries independently, improving query throughput and allowing more nodes to parallelize complex queries.
x??

---

#### Shared-Nothing Clusters
Explanation about shared-nothing clusters and their relevance to partitioning.

:p What is a shared-nothing cluster?
??x
A shared-nothing cluster means that no two nodes share any resources, such as memory or disks. This allows each node to operate independently, making it easier to distribute data across multiple nodes for better performance.
x??

---

#### Indexing and Partitioning Interaction
Explanation about how indexing interacts with partitioning.

:p How does indexing interact with partitioning?
??x
Indexing can significantly improve query performance within a partition but may require additional indexes if queries need to span partitions. The choice of index depends on the query patterns and the data distribution.
x??

---

#### Rebalancing Partitions
Explanation about rebalancing when adding or removing nodes.

:p What is rebalancing in the context of partitioning?
??x
Rebalancing involves redistributing data among partitions when new nodes are added or existing ones are removed. This ensures that the load is evenly distributed across all nodes.
x??

---

#### Request Routing and Query Execution
Explanation about how requests are routed to the right partitions.

:p How do databases route requests to the right partitions?
??x
Databases use routing mechanisms, often based on partition keys, to direct requests to the appropriate partitions. This ensures that queries are executed efficiently by accessing only relevant data.
x??

---

#### Partitioning and Replication Combined
Explanation about combining partitioning with replication for fault tolerance.

:p How does combining partitioning with replication work?
??x
Combining partitioning with replication stores copies of each partition on multiple nodes, ensuring fault tolerance. Even though each record belongs to one partition, it may be stored on several nodes for redundancy.
x??

---

#### Leader-Follower Replication Model
Background context explaining the concept of leader-follower replication model. Each partition's leader is assigned to one node, and its followers are assigned to other nodes. The leader handles all write operations, while followers replicate these writes from the leader. Read operations can be handled by either leaders or followers.
If applicable, add code examples with explanations:
```java
public class LeaderFollowerReplication {
    // Assume a simple in-memory data structure for demonstration
    private Map<String, String> dataStore;
    
    public void setLeader(String partitionKey, String leaderNode) {
        // Assign the node as the leader for the specified partition
    }
    
    public void addFollower(String partitionKey, String followerNode) {
        // Add a node as a follower to the specified partition's leader
    }
    
    public void replicateWrite(String partitionKey, String key, String value) {
        // Leader replicates the write operation to all followers
    }
}
```
:p What is the leader-follower replication model in the context of partitioning?
??x
The leader-follower replication model ensures that each partition has a single leader node responsible for handling writes and some follower nodes responsible for replicating these writes. This setup helps in distributing read operations across multiple nodes, improving overall system performance.
```java
public class LeaderFollowerReplication {
    // code here
}
```
x??

---

#### Partitioning of Key-Value Data
Background context explaining how partitioning is used to spread data and query load evenly across nodes. The goal is to distribute the data so that each node takes a fair share, allowing multiple nodes to handle increased loads.
If applicable, add code examples with explanations:
```java
public class KeyValuePartitioner {
    private Map<String, String> partitions;
    
    public void assignKeyRange(String keyStart, String keyEnd, String partitionNode) {
        // Assign a range of keys from keyStart to keyEnd to the specified node
    }
}
```
:p How do you decide which records to store on which nodes in a key-value data model?
??x
Deciding how to assign records (keys) to nodes is crucial for efficient query handling. By assigning ranges of keys, we can ensure that each node handles a fair share of the data and queries. This approach helps distribute the load evenly across all nodes.
```java
public class KeyValuePartitioner {
    // code here
}
```
x??

---

#### Partitioning by Key Range
Background context explaining key range partitioning as an example of how to assign ranges of keys (from some minimum to some maximum) to each partition. This method allows for efficient querying based on key ranges.
If applicable, add code examples with explanations:
```java
public class KeyRangePartitioner {
    private Map<String, String> keyRanges;
    
    public void setKeyRange(String rangeStart, String rangeEnd, String node) {
        // Assign the specified key range to the given node
    }
}
```
:p What is key range partitioning and how does it work?
??x
Key range partitioning involves assigning a continuous range of keys (from some minimum value to some maximum value) to each partition. This method allows for efficient querying as you can quickly determine which partition contains a given key based on the assigned ranges.
```java
public class KeyRangePartitioner {
    // code here
}
```
x??

---

#### Skew and Hot Spots in Partitioning
Background context explaining skew and hot spots, terms used to describe unfair data distribution that can make partitioning less effective. A hot spot occurs when a single partition receives disproportionately high load.
If applicable, add code examples with explanations:
```java
public class PartitionSkewDetector {
    private Map<String, Long> partitionLoad;
    
    public void recordPartitionLoad(String partitionKey, long load) {
        // Record the load on each partition
    }
}
```
:p What are skew and hot spots in the context of partitioning?
??x
Skew and hot spots describe situations where data or query loads are unevenly distributed across partitions. Skew occurs when some partitions have more data or queries than others, making partitioning less effective. A hot spot is a specific case where one partition receives disproportionately high load.
```java
public class PartitionSkewDetector {
    // code here
}
```
x??

---

#### Key Range Partitioning Strategy

Key range partitioning is a strategy used to distribute data evenly across partitions. The idea is to define boundaries for partitions based on keys, typically using ranges of time or other continuous values.

This method can be useful when dealing with time-series data like sensor readings, where you want to fetch all measurements within a specific time frame easily.

However, it has a downside: if the key used for partitioning skews towards certain values (like today's timestamp in the example), writes may end up being concentrated on one or few partitions, leading to hotspots and imbalanced load distribution.

:p How can key range partitioning lead to performance issues?
??x
Key range partitioning can cause performance issues when the keys are not evenly distributed. For instance, if a system is storing sensor data with timestamps as the primary key, all writes might end up in one partition (e.g., today's measurements), leading to an overloaded partition and underutilized others.

This uneven distribution of load can result in hotspots and suboptimal resource utilization.
x??

---

#### Hash-Based Partitioning Strategy

Hash-based partitioning uses a hash function to distribute keys more evenly across partitions. This approach helps avoid the issues caused by key skew, ensuring that writes are more uniformly distributed among all partitions.

A good hash function should take skewed data and make it appear uniformly random. Commonly used hash functions for this purpose include MD5 or Fowler–Noll–Vo (FNV).

In partitioning, each partition is assigned a range of hashes, not keys. Any key whose hashed value falls within that range will be stored in the corresponding partition.

:p How does using a hash function help distribute data more evenly?
??x
Using a hash function helps distribute data more evenly by ensuring that skewed input data gets spread out randomly across partitions. For example, even if two keys are very similar or identical, their hashed values might differ significantly, leading to better load distribution.

In practice, this is achieved by assigning each partition a range of hashes, and any key falling within that range is stored in the respective partition.
x??

---

#### Sensor Database Example

Consider an application storing data from a network of sensors, where keys are timestamps (year-month-day-hour-minute-second). Range scans on these timestamps can be useful for fetching all readings within a specific time frame.

However, using just the timestamp as the key can lead to hotspots, as all writes might go to the partition corresponding to today's measurements.

:p How can sensor data storage lead to uneven load distribution?
??x
Sensor data storage using timestamps as keys can lead to uneven load distribution because all write operations (sensor readings) tend to cluster in a single partition for the current day. This results in hotspots where one partition handles most of the writes, while others are underutilized.

To address this, you could prefix each timestamp with the sensor name, so partitions first by sensor and then by time. This spreads the write load more evenly across partitions.
x??

---

#### Hash Function Implementation

When implementing hash-based partitioning in a distributed system like Cassandra or MongoDB, you use a robust hash function that can handle similar keys differently.

For instance, both Cassandra and MongoDB use MD5 for their hash functions. The basic idea is to map each key to a unique range of partitions based on its hashed value.

:p What is an example of a hash function used in distributed databases?
??x
An example of a hash function used in distributed databases is MD5. It takes a string input and returns a 128-bit hash value, which can be mapped to a partition ID. Here’s a simple pseudocode for how this might work:

```pseudocode
function getPartitionId(key):
    hashValue = md5(key)
    return hashValue % numberOfPartitions
```

This function ensures that keys are distributed across partitions in a uniform manner, reducing the risk of hotspots.
x??

---

---
#### Partitioning by Hash of Key
Background context explaining the concept. This technique is good at distributing keys fairly among partitions, using evenly spaced or pseudorandom partition boundaries (consistent hashing). Consistent hashing avoids central control or distributed consensus.
:p What is consistent hashing?
??x
Consistent hashing is a method for distributing load across an internet-wide system of caches like a content delivery network (CDN) by randomly choosing partition boundaries. It aims to minimize reassignment of keys when nodes are added or removed, reducing the need for central control or distributed consensus.
??x

---
#### Consistent Hashing
Background context explaining the concept. Consistent hashing was defined by Karger et al. and is used in systems like CDNs. It uses randomly chosen partition boundaries to avoid needing central control or distributed consensus.
:p How does consistent hashing work?
??x
Consistent hashing works by mapping keys to a circular ring, where each key maps to a point on the ring. Nodes are also placed on this ring, and when a new node is added or an existing one removed, only the nodes that are close to the removed/addition points need reassignment.
```java
public class ConsistentHashing {
    private static final int RING_SIZE = 2^32;

    public int hash(String key) {
        return key.hashCode() % RING_SIZE;
    }

    // Logic for placing nodes and rebalancing
}
```
x??

---
#### Key-Range Partitioning vs. Hash Partitioning
Background context explaining the concept. Key-range partitioning maintains adjacency of keys, allowing efficient range queries. However, hash partitioning loses this property as keys are distributed across partitions.
:p Why does key-range partitioning support efficient range queries?
??x
Key-range partitioning supports efficient range queries because it keeps related data in contiguous ranges within a single partition. This allows for direct access to the relevant data without scanning all partitions.
??x

---
#### Cassandra's Compound Primary Key with Hash Partitioning
Background context explaining the concept. In Cassandra, a compound primary key can be declared with multiple columns where only the first part is hashed for partitioning, while other parts are used as an index for sorting data in SSTables (sorted string tables).
:p How does Cassandra handle hash partitioning?
??x
In Cassandra, only the first column of a compound primary key is hashed to determine the partition. The remaining columns act as a concatenated index, allowing efficient querying and sorting within partitions.
```java
public class CompoundKeyExample {
    @PrimaryKey("partitionKey", "sortKey1", "sortKey2")
    public class Row {
        // Column definitions here
    }
}
```
x??

---
#### Range Queries with Hash Partitioning
Background context explaining the concept. Range queries in hash partitioned systems like MongoDB require sending requests to all partitions if based on key hashing, as keys are scattered across partitions.
:p How does MongoDB handle range queries?
??x
MongoDB requires sending range queries to all partitions when using hash-based sharding mode because keys are hashed and scattered across different partitions. This can lead to inefficient performance due to the need to scan multiple partitions for a single query.
??x

---
#### Partitioning Strategies Summary
Background context explaining the concept. Different systems like Cassandra, Riak, Couchbase, and Voldemort handle partitioning differently: Cassandra uses compound primary keys with hash partitioning, while others either do not support range queries or use consistent hashing which is less effective for databases.
:p What are some key differences in partitioning strategies between different NoSQL databases?
??x
Key differences include:
- **Cassandra**: Uses a compound primary key where only the first part is hashed for partitioning, with other parts acting as an index. It achieves a balance between range queries and efficient sorting within partitions.
- **Riak, Couchbase, Voldemort**: Do not support range queries on the primary key, making them less flexible for certain types of queries.
??x

---

#### Partitioning Secondary Indexes by Term
In a distributed database system, secondary indexes can be partitioned to improve read performance. Instead of each partition having its own local index, a global index that covers data across all partitions is often created. However, storing this global index on one node would create a bottleneck and reduce the benefits of partitioning.
:p What is the primary reason for partitioning a secondary index by term?
??x
The main reason for partitioning a secondary index by term is to improve read performance while avoiding bottlenecks that could occur if the entire index were stored on a single node. By partitioning, each term can be efficiently queried without the need to scatter gather across all partitions.
x??

---
#### Global Index Partitioning Strategy
The global index for term-partitioned secondary indexes is divided into different partitions based on terms (e.g., colors starting with 'a' to 'r' in one partition and those starting with 's' to 'z' in another). This allows queries to be targeted to specific partitions rather than scanning all of them.
:p How can a global index be partitioned for term-partitioned secondary indexes?
??x
A global index for term-partitioned secondary indexes can be partitioned based on the terms that define the data. For example, colors starting with 'a' to 'r' could be in one partition and those starting with 's' to 'z' in another. This allows queries like "all red cars" to be targeted directly to the appropriate partitions without scanning all of them.
x??

---
#### Partitioning by Term vs Hash
Term-partitioned indexes can be partitioned either by the term itself or using a hash of the term. Partitioning by the term is useful for range scans, while hashing provides more even distribution of load.
:p What are the two methods to partition a term-partitioned secondary index?
??x
There are two main methods to partition a term-partitioned secondary index: 
1. By the term itself, which can be useful for range scans (e.g., on numeric properties).
2. Using a hash of the term, which provides more even distribution of load.
Each method has its own advantages and use cases depending on the specific query patterns.
x??

---
#### Global Index Advantages
A global secondary index overcomes some limitations of local indexes by allowing efficient reads, as clients only need to request data from the partition containing the desired term. However, writes are slower and more complicated because they may affect multiple partitions.
:p What is a key advantage of using a global (term-partitioned) secondary index?
??x
A key advantage of using a global (term-partitioned) secondary index is that it can make reads more efficient. Rather than doing scatter/gather operations across all partitions, clients only need to request data from the partition containing the desired term.
x??

---
#### Global Index Drawbacks
While global indexes provide benefits for read efficiency, they come with drawbacks, such as slower and more complicated writes because a single write operation may affect multiple partitions. Additionally, updates to the index are often asynchronous.
:p What is one of the main disadvantages of using a global (term-partitioned) secondary index?
??x
One of the main disadvantages of using a global (term-partitioned) secondary index is that writes are slower and more complicated because a single write operation may affect multiple partitions. Additionally, updates to the global index are often asynchronous, meaning changes made through a write might not immediately reflect in the index.
x??

---
#### Implementation Considerations
Implementing term-partitioned secondary indexes requires careful consideration of how terms map to partitions, as well as handling distributed transactions across affected partitions for consistency. In practice, updates to these indexes can be asynchronous due to limitations in some database systems.
:p What challenges are associated with implementing a global (term-partitioned) secondary index?
??x
Challenges associated with implementing a global (term-partitioned) secondary index include:
1. Handling distributed transactions across partitions for consistency, which is not always supported by all databases.
2. Asynchronous updates to the index, where changes made through writes may not immediately reflect in the index due to propagation delays.
These challenges require careful design and consideration of the database system's capabilities.
x??

---
#### Examples of Global Term-Partitioned Indexes
Riak’s search feature and Oracle data warehouse both support global term-partitioned indexes. These can be particularly useful for full-text indexing scenarios, where terms are words in documents.
:p What systems support global term-partitioned indexes?
??x
Systems that support global term-partitioned indexes include:
1. Riak's search feature, which allows for efficient searching across partitioned data.
2. Oracle data warehouse, which offers the option to choose between local and global indexing.
These features are particularly useful in full-text indexing scenarios where terms are words within documents.
x??

---

#### Hash Mod N Partitioning Issue
Background context: When using hash partitioning, sometimes a simple approach like `hash(key) % n` (where `n` is the number of nodes) is tempting because it seems straightforward. However, this method can cause excessive data movement during rebalancing when the number of nodes changes.

:p What problem does the mod N approach have during cluster expansion or contraction?
??x
The mod N approach causes frequent reassignment of keys to different partitions as the number of nodes changes, leading to unnecessary and costly data migrations. For example, if you initially use `hash(key) % 10` with 10 nodes, a key might be on node 6. When expanding to 11 nodes, that key would need to move to node 3, and when further expanding to 12 nodes, it would move again to node 0.

To illustrate:
```java
public class HashModExample {
    public int getPartition(int hash) {
        return hash % 10; // Example with 10 nodes
    }
}
```
x??

---

#### Fixed Number of Partitions Strategy
Background context: To mitigate the frequent reassignment issues, a better approach is to use a fixed number of partitions that exceed the number of nodes. This ensures that when adding or removing nodes, only some partitions are reassigned.

:p How does this strategy work in practice?
??x
This strategy involves creating more partitions than there are nodes and assigning multiple partitions to each node from the start. For instance, if a cluster has 10 nodes, you might initially create 1,000 partitions with about 100 per node.

When adding or removing nodes, only some of these partitions are reassigned. Specifically:
- Adding a node: The new node takes over a few partitions from existing nodes.
- Removing a node: Partitions are reassigned back to the remaining nodes.

The key is that the number and assignment of keys to partitions remain constant; only their distribution among nodes changes. This minimizes data movement and network load during rebalancing.

```java
public class PartitionAssignment {
    private final int totalPartitions = 1000;
    private final int nodes = 10;

    public void reassignPartitions(int newNodes) {
        // Logic to evenly distribute partitions among existing and new nodes.
    }
}
```
x??

---

#### Partitioning by Key Range

Background context: In databases that use key range partitioning, data is divided into partitions based on key ranges. This method ensures balanced load distribution across nodes and handles variable dataset sizes effectively.

If applicable, add code examples with explanations:
:p How does key range partitioning work in HBase?
??x
In HBase, data is partitioned into regions based on key ranges. Each region corresponds to a specific key range, and these regions are mapped to different servers (nodes) for load balancing. When the size of a region grows beyond a certain threshold, it is split into two smaller regions.

```java
// Pseudocode for region splitting in HBase
public class RegionSplitter {
    public void splitRegionIfNecessary(byte[] startKey, byte[] endKey) {
        // Check if the region should be split based on its size
        if (regionSizeIsTooLarge(startKey, endKey)) {
            // Split the region into two new regions with appropriate keys
            byte[] midPoint = calculateMidpoint(startKey, endKey);
            byte[] newStartKey = Arrays.copyOf(midPoint, midPoint.length - 1);
            byte[] newEndKey = midPoint;
            splitRegion(newStartKey, newEndKey);
        }
    }

    private boolean regionSizeIsTooLarge(byte[] startKey, byte[] endKey) {
        // Check if the size of the current region exceeds a configurable threshold
        return (endKey.length - startKey.length) > MAX_REGION_SIZE;
    }

    private byte[] calculateMidpoint(byte[] startKey, byte[] endKey) {
        // Calculate the midpoint between start and end keys for splitting
        int midPointIndex = (startKey.length + endKey.length) / 2;
        return Arrays.copyOf(startKey, midPointIndex);
    }
}
```
x??

---

#### Dynamic Partitioning

Background context: Dynamic partitioning is a method used in databases like HBase and RethinkDB to adapt the number of partitions based on the total data volume. This approach ensures that overheads remain low when there is little data but scales efficiently as the dataset grows.

:p How does dynamic partitioning work in HBase?
??x
In HBase, dynamic partitioning works by splitting a large partition into two smaller partitions once it exceeds a certain size threshold. Conversely, if a partition shrinks significantly due to deletions or other operations, it can be merged with an adjacent partition.

```java
// Pseudocode for dynamic partitioning in HBase
public class DynamicPartitioner {
    public void splitPartitionIfNecessary(byte[] startKey, byte[] endKey) {
        // Check if the current partition size exceeds a configurable threshold
        if (partitionSizeIsTooLarge(startKey, endKey)) {
            // Calculate the midpoint key for splitting
            byte[] midPoint = calculateMidpoint(startKey, endKey);
            splitPartition(midPoint);
        }
    }

    private boolean partitionSizeIsTooLarge(byte[] startKey, byte[] endKey) {
        // Check if the size of the current partition exceeds a configurable threshold
        return (endKey.length - startKey.length) > MAX_PARTITION_SIZE;
    }

    private byte[] calculateMidpoint(byte[] startKey, byte[] endKey) {
        // Calculate the midpoint between start and end keys for splitting
        int midPointIndex = (startKey.length + endKey.length) / 2;
        return Arrays.copyOf(startKey, midPointIndex);
    }
}
```
x??

---

#### Pre-splitting

Background context: Pre-splitting is a technique used in databases that support dynamic partitioning. It involves setting up an initial set of partitions before the database starts receiving any data. This helps to avoid a situation where all writes initially have to be processed by one node.

:p What is pre-splitting, and why is it necessary?
??x
Pre-splitting is a technique used in databases like HBase and MongoDB to initialize an appropriate number of partitions based on expected key ranges before the database starts receiving any data. This avoids bottlenecks where all writes initially have to be processed by one node.

For example, if you know that your keys will range from 0 to 1 billion, you might pre-split into 10 initial partitions:

```java
// Pseudocode for pre-splitting in HBase
public class PreSplitter {
    public void initializePartitions(int numInitialPartitions) {
        byte[][] splitPoints = new byte[numInitialPartitions - 1][];
        int keyRangePerPartition = (int) Math.ceil((double) MAX_KEY / numInitialPartitions);
        
        for (int i = 0; i < numInitialPartitions - 1; i++) {
            // Calculate the start and end keys for each partition
            byte[] startKey = calculateStartKey(i * keyRangePerPartition, (i + 1) * keyRangePerPartition - 1);
            splitPoints[i] = startKey;
        }
        
        createInitialPartitions(splitPoints);
    }

    private byte[] calculateStartKey(int startValue, int endValue) {
        // Convert the numerical values to byte array keys
        return ByteBuffer.allocate(8).putLong(startValue).array();
    }
}
```
x??

---

#### Rebalancing Partitions

Background context: When a partition grows too large or shrinks due to data deletions, it needs to be rebalanced across nodes. This ensures that the load is evenly distributed and prevents any single node from becoming overloaded.

:p How does HBase handle the transfer of partitions between nodes?
??x
HBase handles the transfer of partitions between nodes by using its underlying distributed filesystem (HDFS). When a partition grows too large, it is split into two smaller partitions. One of these halves can then be transferred to another node in the cluster.

```java
// Pseudocode for partition rebalancing in HBase
public class PartitionRebalancer {
    public void transferPartitionToNewNode(byte[] startKey, byte[] endKey, Node newNode) {
        // Determine which half of the split partition should be moved
        byte[] midPoint = calculateMidpoint(startKey, endKey);
        boolean moveToNewNode = (new Random().nextBoolean()) ? true : false;
        
        if (moveToNewNode) {
            // Transfer one half to the new node via HDFS
            transferHalfToHDFS(midPoint, newNode);
        }
    }

    private byte[] calculateMidpoint(byte[] startKey, byte[] endKey) {
        // Calculate the midpoint between start and end keys for splitting
        int midPointIndex = (startKey.length + endKey.length) / 2;
        return Arrays.copyOf(startKey, midPointIndex);
    }

    private void transferHalfToHDFS(byte[] midPoint, Node newNode) {
        // Transfer half of the split partition to the new node via HDFS
        // This involves copying files from old node to new node's HDFS directory
    }
}
```
x??

---

#### Partitioning Strategies and Their Impact on Data Distribution

Background context: The document discusses various strategies for partitioning data in distributed databases, focusing on how these strategies impact the distribution of partitions across nodes. Different methods such as range-based, hash-based, and consistent hashing are explained.

:p How do you describe the range-based partitioning strategy?
??x
Range-based partitioning involves dividing the dataset into ranges, where each node manages a specific range. The size of each partition is proportional to the size of the dataset, making the number of partitions independent of the number of nodes.
x??

---

#### Hash-Based Partitioning

Background context: This section discusses hash-based partitioning, which uses hashing functions to distribute data evenly across nodes.

:p What is the key characteristic of hash-based partitioning?
??x
Hash-based partitioning uses a hash function to map keys to partitions. Each node handles a subset of partitions determined by its hash range.
x??

---

#### Consistent Hashing

Background context: Consistent hashing is mentioned as an approach used in Cassandra and Ketama, where the number of partitions per node is fixed.

:p Explain consistent hashing in the context of data distribution.
??x
Consistent hashing ensures that keys are evenly distributed across nodes by using a hash ring. When new nodes join or existing ones leave, only a few keys need to be moved, minimizing disruption and load imbalance.
x??

---

#### Rebalancing Strategies

Background context: The document explains how rebalancing can be automated or manual, with fully automatic and fully manual approaches described.

:p What is the main advantage of automatic rebalancing?
??x
Automatic rebalancing automates partition redistribution without human intervention, reducing operational overhead. However, it may lead to unpredictable performance changes.
x??

---

#### Request Routing Mechanisms

Background context: The passage discusses different methods for routing client requests to the correct node based on partitioning and load balancing strategies.

:p How does a round-robin load balancer route requests?
??x
A round-robin load balancer routes requests by cycling through nodes in a predefined order. If any node owns the requested partition, it handles the request; otherwise, it forwards the request to another node.
x??

---

#### Service Discovery

Background context: Service discovery involves dynamically determining which node or service should handle client requests as partitions are reassigned.

:p What is the purpose of service discovery in distributed systems?
??x
Service discovery ensures that clients can correctly route their requests to the appropriate nodes even when partition assignments change. This is crucial for maintaining availability and consistency.
x??

---

#### Using ZooKeeper for Metadata Management

Background context: The text mentions using a coordination service like ZooKeeper to manage partitioning metadata, ensuring all nodes stay up-to-date with changes.

:p What role does ZooKeeper play in managing partitions?
??x
ZooKeeper acts as a central coordination service that maintains the authoritative mapping of partitions to nodes. Nodes register themselves and subscribe to updates from ZooKeeper to ensure they have the latest routing information.
x??

---

