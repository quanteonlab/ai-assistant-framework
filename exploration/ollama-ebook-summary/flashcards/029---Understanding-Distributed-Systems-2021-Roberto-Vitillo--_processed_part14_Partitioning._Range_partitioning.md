# Flashcards: 029---Understanding-Distributed-Systems-2021-Roberto-Vitillo--_processed (Part 14)

**Starting Chapter:** Partitioning. Range partitioning

---

---
#### Sharding Strategies Overview
Sharding is a technique used to distribute data across multiple nodes. When a dataset outgrows a single node, it needs to be partitioned into smaller datasets that can reside on different nodes. This ensures scalability and improves performance by distributing the load.

In this context, we will discuss how keys are mapped to partitions in a sharded key-value store.
:p What is sharding and why is it necessary?
??x
Sharding is a technique used when a dataset outgrows a single node, necessitating the distribution of data across multiple nodes. This approach helps in scaling applications by distributing the load and improving performance.

```java
public class ShardManager {
    private Map<String, Node> keyToNodeMap;

    public ShardManager() {
        // Initialize map to store key-to-node mappings
        this.keyToNodeMap = new HashMap<>();
    }

    public void addShard(String key, Node node) {
        keyToNodeMap.put(key, node);
    }

    public Node getNodeForKey(String key) {
        return keyToNodeMap.get(key);
    }
}
```
x??
---

#### Mapping Keys to Partitions
The mapping between keys and partitions is typically maintained in a strongly-consistent configuration store like etcd or Zookeeper. This ensures that the system has up-to-date information on how keys are distributed across nodes.

:p How is the key-to-node mapping managed in a sharded environment?
??x
In a sharded environment, the key-to-node mapping is often stored in a strongly-consistent configuration store such as etcd or Zookeeper. This ensures that all nodes have consistent and up-to-date information about which node is responsible for each key.

```java
public class ShardMappingService {
    private ConsistentConfigStore configStore;

    public ShardMappingService(ConsistentConfigStore configStore) {
        this.configStore = configStore;
    }

    public void updateShardMapping(String key, Node node) {
        configStore.putKeyToNodeMapping(key, node);
    }

    public Node getNodeForKey(String key) {
        return configStore.getKeyToNodeMapping(key);
    }
}
```
x??
---

#### Gateway Service for Routing
A gateway service can be used to route requests based on the mapping of keys to partitions and partitions to nodes. This service is crucial for handling read and write operations efficiently.

:p What role does a gateway service play in sharding?
??x
A gateway service acts as an intermediary between client requests and the shards (partitions) managed by different nodes. It routes requests based on the mapping of keys to partitions and ensures that each request goes to the correct node, optimizing performance and reliability.

```java
public class ShardGateway {
    private Map<String, Node> keyToNodeMap;

    public ShardGateway(Map<String, Node> keyToNodeMap) {
        this.keyToNodeMap = keyToNodeMap;
    }

    public void handleRequest(String key) {
        Node node = keyToNodeMap.get(key);
        if (node != null) {
            // Route request to the appropriate node
            node.handleRequest(key);
        } else {
            throw new IllegalArgumentException("Key not found in any shard");
        }
    }
}
```
x??
---

#### Range Partitioning
Range partitioning splits data into partitions based on a continuous range of keys. This is useful for scanning ranges efficiently but can lead to imbalanced partitions if the key distribution is uneven.

:p How does range partitioning work, and what are its potential issues?
??x
Range partitioning works by splitting the key space into contiguous ranges, where each partition holds a specific range of values. For example, if you partition data by date, one partition might hold all records from January 1 to March 31, another from April 1 to June 30, and so on.

However, this approach can lead to unbalanced partitions if the key distribution is uneven. Additionally, some access patterns, such as writing current-day data, may cause hotspots in a single partition, degrading performance.

```java
// Example of range partitioning by date
public class RangePartition {
    public static void main(String[] args) {
        // Assuming we have a list of dates and want to create partitions based on these dates.
        List<Date> dates = Arrays.asList(new Date(), new Date(), ...);
        
        int numPartitions = 4; // Define the number of partitions
        long stepSize = (dates.size() - 1) / numPartitions; // Calculate the size of each partition range
        
        for(int i = 0; i < dates.size(); i += stepSize) {
            Date start = dates.get(i);
            Date end = i + stepSize < dates.size() ? dates.get(i + stepSize) : null;
            
            System.out.println("Partition from " + start + " to " + (end != null ? end : "null"));
        }
    }
}
```
x??

---

#### Hash Partitioning
Hash partitioning uses a hash function to distribute keys uniformly across partitions, which can help balance the load. However, it doesn't guarantee an even distribution of access patterns.

:p How does hash partitioning work, and what are its potential issues?
??x
Hash partitioning assigns each key to a specific partition using a hash function. The idea is to map a potentially non-uniformly distributed key space into a uniformly distributed hash space. For example, simple modular hashing can be implemented as `hash(key) mod N`, where `N` is the number of partitions.

However, this approach doesn't eliminate hotspots if access patterns are not uniform. If a single key is accessed more frequently than others, all bets are off, and the partition containing that hotkey needs to be split further or the key itself can be split into multiple sub-keys.

```java
// Example of hash partitioning using modular hashing
public class HashPartition {
    public static void main(String[] args) {
        int numPartitions = 4; // Define the number of partitions
        String key = "exampleKey"; // The key to be hashed
        
        int hashValue = (key.hashCode() % numPartitions); // Simple modular hashing
        System.out.println("Partition: " + hashValue);
    }
}
```
x??

---

#### Stable Hashing and Ring Hashing
Stable hashing, such as ring hashing or consistent hashing, ensures that only a small subset of keys need to be rehashed when a new partition is added. This reduces the overhead of data shuffling.

:p What are stable hashing strategies like ring hashing and consistent hashing, and how do they work?
??x
Stable hashing strategies like ring hashing and consistent hashing ensure that the number of keys that need to be reassigned remains minimal when adding a new partition. For example, in consistent hashing, both partitions and keys are randomly distributed on a circle, and each key is assigned to the next partition in clockwise order.

When a new partition is added, only the keys mapped to it need to be reassigned. This minimizes the overhead of data shuffling.

```java
// Example of consistent hashing (simplified)
public class ConsistentHashing {
    public static void main(String[] args) {
        int numPartitions = 5; // Define the number of partitions
        String key1 = "key1"; // The first key to be hashed and assigned a partition
        String key2 = "key2"; // Another example key
        
        // Assume we have a circle with partition identifiers and keys distributed randomly.
        
        int hashKey1 = (key1.hashCode() % numPartitions); // Hash the first key
        int hashKey2 = (key2.hashCode() % numPartitions); // Hash the second key
        
        System.out.println("Partition for " + key1 + ": " + hashKey1);
        System.out.println("Partition for " + key2 + ": " + hashKey2);
    }
}
```
x??

---

#### Loss of Sort Order in Hash Partitioning
While hash partitioning loses the sort order over partitions, data within an individual partition can still be sorted based on a secondary key.

:p What is the main drawback of hash partitioning compared to range partitioning?
??x
The main drawback of hash partitioning compared to range partitioning is that it loses the natural sort order over partitions. Range partitioning maintains a continuous and ordered sequence within each partition, which can be useful for efficient scans and queries.

However, with hash partitioning, the data within an individual partition cannot maintain any specific order; sorting must be done based on secondary keys if needed.

```java
// Example of maintaining sort order in range partitioning vs. hash partitioning
public class SortOrder {
    public static void main(String[] args) {
        // Range Partitioning
        List<String> data = Arrays.asList("a", "b", "c", "d");
        int numPartitions = 2;
        
        for(int i = 0; i < data.size(); i += (data.size() / numPartitions)) {
            List<String> partition = data.subList(i, Math.min(data.size(), i + (data.size() / numPartitions)));
            System.out.println("Partition: " + partition);
        }
        
        // Hash Partitioning
        int hashData[] = new int[data.size()];
        for(int i = 0; i < data.size(); i++) {
            hashData[i] = Math.abs(data.get(i).hashCode() % numPartitions);
        }
        
        for (int i = 0; i < numPartitions; i++) {
            List<String> partition = new ArrayList<>();
            for(int j = 0; j < data.size(); j++) {
                if(hashData[j] == i) {
                    partition.add(data.get(j));
                }
            }
            System.out.println("Partition: " + partition);
        }
    }
}
```
x??

---

#### Rebalancing

Rebalancing is a process that occurs when the number of requests to the data store becomes too large, or the dataset's size becomes too large. To handle this, the number of nodes serving partitions needs to be increased. Conversely, if the dataset’s size keeps shrinking, the number of nodes can be reduced to minimize costs.

Rebalancing must be implemented in a way that minimizes disruption to the data store, ensuring it continues to serve requests efficiently. The amount of data transferred during rebalancing needs to be minimized to avoid impacting performance and availability.

:p What is rebalancing?
??x
Rebalancing refers to the process of adjusting the number of nodes serving partitions in a distributed system when the load or dataset size changes significantly. It aims to maintain an optimal distribution of work across nodes to ensure efficient data processing while minimizing disruption.
x??

---

#### Static Partitioning

In static partitioning, the idea is to create more partitions than necessary during the initialization phase and assign multiple partitions per node. When a new node joins, some existing partitions are moved to the new node, maintaining an always-balanced state.

However, this approach has limitations because the number of partitions is set at the beginning and can't be easily changed afterward. Having too many or too few partitions can lead to performance overhead or scalability issues respectively.

:p What is static partitioning?
??x
Static partitioning involves creating more partitions than necessary during system initialization and assigning multiple partitions per node. When a new node joins, existing partitions are redistributed to maintain balance. The main drawback is that the number of partitions cannot be easily changed after initialization.
x??

---

#### Dynamic Partitioning

Dynamic partitioning is an alternative where partitions are created on-demand rather than upfront. It starts with a single partition and splits it when it grows too large or becomes 'too hot'. Conversely, if adjacent partitions become small enough, they can be merged.

:p What is dynamic partitioning?
??x
Dynamic partitioning involves creating partitions as needed rather than upfront. It begins with a single partition that is split into smaller ones when necessary due to growth or high load (becoming "too hot"). Similarly, adjacent small partitions can be combined if they shrink in size.
x??

---

#### Practical Considerations

Introducing partitions adds complexity to the system even though it might seem simple on the surface. Ensuring balanced partitioning is crucial as a single hot partition can become a bottleneck and limit scalability.

Each partition operates independently, requiring atomic updates across multiple partitions for transactions.

:p What are practical considerations in introducing partitions?
??x
Practical considerations include added complexity due to partition management and the risk of partition imbalance leading to bottlenecks. Each partition being independent necessitates atomic transaction handling across multiple partitions.

Code Example:
```java
public class PartitionManager {
    public void updateMultiplePartitions(Transaction tx) {
        // Logic to ensure updates are atomic across multiple partitions
    }
}
```
x??

---

