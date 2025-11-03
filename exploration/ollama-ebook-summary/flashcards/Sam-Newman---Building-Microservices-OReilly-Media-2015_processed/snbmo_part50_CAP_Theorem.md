# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 50)

**Starting Chapter:** CAP Theorem

---

---
#### CAP Theorem Overview
The CAP theorem states that in a distributed system, we can only achieve two of three desirable properties: consistency, availability, and partition tolerance. Since Eric Brewer published his original conjecture, it has been mathematically proven.

:p What does the CAP theorem state about distributed systems?
??x
The CAP theorem indicates that in any distributed system, you can only ensure at most two out of the following three qualities: Consistency (C), Availability (A), and Partition Tolerance (P). This means that if a network partition occurs, it is impossible to guarantee all three properties simultaneously.
x??

---
#### Network Partition Scenario
In the scenario described, we have an inventory service deployed across two separate data centers. Each data center has its own database node, and these nodes try to synchronize their data.

:p What happens when the network link between the two data centers fails?
??x
When the network link between the two data centers fails, synchronization stops. Writes made to one database may not propagate to the other database. This leads to a situation where the databases become out of sync.
x??

---
#### Consistency and Availability in Failures
In the scenario with failed network links, reads and writes are done via the local database node, but replication is used to synchronize data between nodes.

:p What trade-offs might occur due to a network partition in this setup?
??x
Due to the network partition, you may have to trade off consistency and availability. For example:
- Consistency: Writes made to one database may not be reflected immediately in the other database.
- Availability: Since writes are only acknowledged by the local node, they might appear successful even if replication is delayed or fails.

To handle this, most databases support queuing techniques to ensure eventual consistency after network recovery.
x??

---
#### Queuing Technique for Recovery
Queuing mechanisms can be used to ensure that data can still be written and recovered from during a network failure. These queues allow writes to continue despite partition issues.

:p How do queuing techniques help in maintaining availability during network partitions?
??x
Queuing techniques provide a way to maintain availability by allowing writes to succeed even when replication is delayed or failed. Once the network recovers, these queued writes can be replayed to ensure eventual consistency.
x??

---
#### Handling Network Partitions
Handling network partitions involves ensuring that the system can continue operating despite communication failures between parts of the system.

:p What are some strategies for maintaining availability during a network partition?
??x
Strategies include:
- Using queuing mechanisms to buffer writes until network recovery.
- Implementing optimistic replication where data is written locally and then propagated later.

These techniques help ensure that the system remains available, even when partitions occur.
x??

---
#### Example of Multiprimary Replication
Multiprimary replication involves setting up multiple primary nodes in different locations. This allows for continued operation despite network failures.

:p How does multiprimary replication affect data consistency and availability?
??x
Multiprimary replication can improve both availability and partition tolerance by allowing writes to any of the primary nodes, but it may reduce strict consistency guarantees as conflicts might arise if not managed properly.
x??

---

---
#### AP System
In a distributed system, we may face partitions where network failures occur between database nodes. If we choose to keep availability and partition tolerance (AP), changes made on one node may not be immediately reflected on another due to the partition. This results in potentially stale data being served.
:p What is an AP system?
??x
An AP system, or an Availability-Partition Tolerance system, prioritizes availability and partition tolerance over consistency. In such a scenario, if there's a network failure between database nodes (DC1 and DC2), the node that doesn't receive updates will continue to serve potentially stale data until synchronization occurs.
```java
// Pseudocode for handling AP in a distributed system
public void handleAPPartition() {
    // Check if the current node can communicate with another node
    boolean canCommunicate = checkNetworkStatus();
    
    if (!canCommunicate) {
        // If communication is lost, accept stale data to ensure availability and partition tolerance
        serveStaleData();
    } else {
        // If communication is restored, synchronize data across nodes
        synchronizeData();
    }
}
```
x??

---
#### Eventually Consistent Systems
To maintain high availability in the face of network partitions, systems often adopt eventual consistency. This means that after a series of updates and potential delays in replication, all nodes will eventually reflect the same state. However, during the delay period, users might experience stale data.
:p What is an eventually consistent system?
??x
An eventually consistent system accepts reduced immediate consistency to maintain high availability (A) and partition tolerance (P). The key idea here is that updates will propagate through the network over time, but not instantaneously. Users may temporarily encounter old or inconsistent data until synchronization occurs.
```java
// Pseudocode for handling eventual consistency
public void updateDataEventuallyConsistent(String data) {
    // Update local copy of data immediately
    localDatabase.update(data);
    
    // Schedule an asynchronous replication task to other nodes
    scheduleReplicationTask();
}
```
x??

---
#### CP System
To ensure strong consistency across all nodes, the system must sacrifice availability (A) and may experience downtime during network partitions. This mode ensures that once a node is updated, all replicas will eventually reflect those updates.
:p What is a CP system?
??x
A CP (Consistency-Partition Tolerance) system prioritizes consistency over availability and partition tolerance. In the presence of a network failure, the system may temporarily shut down or limit its functionality to ensure that once an update is made, all nodes will eventually reflect it. This approach sacrifices availability during the partition period.
```java
// Pseudocode for handling CP in a distributed system
public void handleCPPartition() {
    // Check if the current node can communicate with another node
    boolean canCommunicate = checkNetworkStatus();
    
    if (!canCommunicate) {
        // If communication is lost, block all write requests until synchronization is possible
        rejectWriteRequests();
    } else {
        // If communication is restored, process and replicate writes to other nodes
        processAndReplicateWrites();
    }
}
```
x??

---
#### Transactional Reads in Distributed Systems
Consistency across multiple nodes in distributed systems requires coordination through transactions. However, initiating a transactional read involves locking mechanisms that can cause significant delays and potential deadlocks.
:p What are the challenges with transactional reads in distributed systems?
??x
Transactional reads in distributed systems pose significant challenges due to the need for cross-node coordination via locks. Ensuring that a read operation does not conflict with an ongoing write operation across multiple nodes is complex, especially given network partitions or failures. This can lead to performance issues and potential deadlocks.
```java
// Pseudocode for initiating a transactional read
public String performTransactionalRead(String key) {
    // Attempt to lock the remote node while reading from local node
    if (tryLockRemoteNode(key)) {
        try {
            return localDatabase.read(key);
        } finally {
            unlockRemoteNode(key); // Ensure the lock is released even on failure
        }
    } else {
        throw new LockAcquisitionException("Could not acquire lock for transactional read");
    }
}
```
x??

---

#### Distributed Transactions and CAP Theorem
Background context: This section discusses the challenges of implementing distributed transactions, focusing on the CAP theorem. CAP stands for Consistency, Availability, and Partition Tolerance. The core issue is ensuring consistency across multiple nodes when a network partition occurs. In practice, it's highly advised not to invent your own distributed consistent data store but rather use existing solutions.
:p What does CAP stand for in the context of distributed systems?
??x
CAP stands for Consistency, Availability, and Partition Tolerance. These are three critical properties that a distributed system can have at most two out of these three at any given time according to the CAP theorem.
```java
// Example pseudocode to illustrate the concept
public class Node {
    public void handleRequest() {
        // Handle request logic here
    }
}
```
x??

---

#### Importance of Using Existing Solutions for Distributed Transactions
Background context: The text emphasizes that inventing your own distributed consistent data store is highly risky and should be avoided. Instead, it's better to use existing solutions or aim for eventually consistent AP systems.
:p Why does the author strongly advise against inventing a CP (Consistency, Partition Tolerance) system?
??x
The author advises against inventing a CP system because building such a system from scratch is extremely challenging and risky. It requires deep understanding and expertise, often involving reading numerous research papers, obtaining a PhD, and still being prone to errors. Using existing solutions like Consul for a strongly consistent key/value store can significantly reduce these risks.
```java
// Example pseudocode for using an existing solution
public class DataStoreClient {
    public void useExistingSolution() {
        // Code to interact with an existing data store
    }
}
```
x??

---

#### Trade-Off Between AP and CP Systems
Background context: The text discusses the trade-offs between AP (Availability, Partition Tolerance) and CP (Consistency, Partition Tolerance) systems. It highlights that there is no one-size-fits-all solution and that understanding these trade-offs is crucial for making informed decisions.
:p What does the CAP theorem suggest about system design?
??x
The CAP theorem suggests that in a distributed system, you can have at most two out of the three properties: Consistency, Availability, and Partition Tolerance. This means that depending on your requirements, you might need to sacrifice one property for the others.
```java
// Example pseudocode illustrating trade-offs
public class SystemDesign {
    public void chooseSystemType() {
        // Logic to decide between AP or CP based on business needs
    }
}
```
x??

---

#### Microservices and CAP Theorem
Background context: The text concludes by discussing how individual microservices within a system can have different requirements for consistency. It emphasizes that while the entire system might be designed as an eventually consistent AP system, certain critical services may need to ensure consistency.
:p How can a system design accommodate both AP and CP needs?
??x
A system can accommodate both AP and CP needs by designing individual microservices based on their specific requirements. For example, non-critical services like a catalog might be designed as an eventually consistent AP system, while critical services like inventory might need to ensure consistency.
```java
// Example pseudocode for service design
public class Microservice {
    public void designService() {
        // Logic to determine if the service needs to be CP or AP
    }
}
```
x??

---

#### Understanding the Trade-Offs in Practice
Background context: The text underscores that understanding the trade-offs between AP and CP is essential. It advises that without knowing the specific business impact, it's difficult to make an informed decision about which system type to use.
:p How can businesses understand the trade-offs better?
??x
Businesses can understand the trade-offs better by thoroughly analyzing their specific requirements and the potential impacts of each design choice on their operations. This includes understanding how stale data or inconsistent records might affect different parts of their business processes.
```java
// Example pseudocode for decision-making
public class BusinessAnalysis {
    public void analyzeRequirements() {
        // Code to analyze business impact and make informed decisions
    }
}
```
x??

---

#### Trade-Offs in CAP Theorem
Background context explaining the concept. The CAP theorem states that a distributed system can exhibit at most two of the following properties: Consistency, Availability, and Partition Tolerance. However, real-world systems often allow for more nuanced trade-offs within individual service capabilities.
:p What is the main idea discussed regarding CAP in this text?
??x
The text explains that while the CAP theorem sets a mathematical limit on what can be achieved in distributed systems—stating that you can only have at most two of Consistency, Availability, and Partition Tolerance—it highlights that many systems allow for more flexible trade-offs within their services. For instance, in Apache Cassandra, different consistency levels (strict, quorum, or single node) are available based on the specific needs.
x??

---

#### Different Consistency Levels in Cassandra
Background context explaining the concept. Cassandra offers various consistency levels to balance between strict consistency and availability in a distributed environment.
:p How does Cassandra allow for nuanced trade-offs between consistency and availability?
??x
Cassandra allows for different types of reads that enable varying degrees of consistency:

- **Strict Consistency**: Blocks until all replicas have responded with the same value, ensuring eventual consistency but potentially leading to long blocking times if one replica is unavailable.
```java
// Pseudocode for a strict consistency read in Cassandra
ConsistencyLevel ALL = ConsistencyLevel.ONE;
session.execute("SELECT * FROM inventory WHERE album_id = ? ", albumId, ALL);
```

- **Quorum Read**: Blocks until a quorum of replicas have responded, providing a balance between performance and consistency.
```java
// Pseudocode for a quorum read in Cassandra
ConsistencyLevel QUORUM = ConsistencyLevel.ONE;
session.execute("SELECT * FROM inventory WHERE album_id = ? ", albumId, QUORUM);
```

- **Single Node Read**: Does not block until a quorum of replicas have responded; returns the first available node's response.
```java
// Pseudocode for a single node read in Cassandra
ConsistencyLevel ONE = ConsistencyLevel.ONE;
session.execute("SELECT * FROM inventory WHERE album_id = ? ", albumId, ONE);
```
x??

---

#### Real-World vs. Electronic World Consistency
Background context explaining the concept. The text contrasts how electronic systems handle consistency with the unpredictability of real-world events.
:p How do real-world scenarios challenge the consistency assumptions in electronic systems?
??x
Real-world scenarios often introduce inconsistencies that electronic systems cannot account for because they are designed to operate within controlled environments where stopping the world and ensuring perfect consistency is feasible. For example, physical inventory counts can be affected by unforeseen events such as damage or theft, whereas an electronic system might only decrement a count when a sale transaction is successfully processed.

To handle such inconsistencies, some systems opt for AP (Availability and Partition Tolerance) strategies where occasional discrepancies are acceptable in exchange for better availability and scalability. For instance:
```java
// Pseudocode demonstrating handling of inconsistent states in a real-world inventory system
if (physicalInventoryCount < electronicInventoryCount) {
    notifyUser("Item out of stock, please contact support.");
}
```
x??

---

#### Service Discovery in Microservices
Background context explaining the concept. As microservices architectures scale, managing where services are running and how to locate them becomes increasingly important for monitoring and usage.
:p What is service discovery in a microservices environment?
??x
Service discovery involves mechanisms that help distributed systems find and communicate with other services. In a microservices architecture, this is critical because services can be deployed dynamically across multiple instances or even different environments.

Service discovery tools like Consul, Eureka, or Kubernetes Service Discovery are commonly used to manage the locations of running services so that consuming microservices can reliably connect to them.
```java
// Pseudocode for a simple service discovery in Kubernetes
Map<String, String> labels = new HashMap<>();
labels.put("app", "inventory");
DiscoveryClient discoveryClient = new DiscoveryClient();
List<ServiceInstance> instances = discoveryClient.getInstances("inventory-service");
ServiceInstance instance = instances.get(0);
String uri = instance.getUri().toString();
```
x??

---

#### DNS Service Discovery
Background context explaining the concept. DNS allows associating a name with an IP address, facilitating service discovery by using names instead of IPs. It is particularly useful for managing different environments (e.g., dev, uat, prod) through templates or separate domain name servers.

:p What are some advantages and disadvantages of using DNS for service discovery?
??x
DNS offers several advantages:
- Well-understood and widely supported.
- Easy to use with most technology stacks.

However, it also has notable downsides:
- Updating DNS entries can be challenging in dynamic environments.
- TTL (Time To Live) values mean clients might hold onto old IP addresses for a while.
- Caching at various levels (network devices, JVMs, etc.) can further complicate service updates.

DNS is better suited for static setups with single nodes or where load balancers are used to manage instances dynamically.
x??

---
#### DNS and Load Balancer Integration
Background context explaining the concept. Integrating DNS with a load balancer allows dynamic management of services by updating only the load balancer's configuration, which in turn updates the traffic distribution.

:p How can using a load balancer improve service discovery when compared to directly resolving IP addresses?
??x
Using a load balancer improves service discovery in several ways:
- Simplifies managing multiple instances behind a single endpoint.
- Allows for easier scaling and deployment of new instances without changing DNS records frequently.
- Provides redundancy and fault tolerance.

Code Example: Configuration for a hypothetical load balancer (e.g., Nginx) to route traffic to different backend services.
```nginx
http {
    upstream accounts_backend {
        server 192.0.2.1;
        server 192.0.2.2;
    }

    server {
        listen 80;

        location /accounts {
            proxy_pass http://accounts_backend;
        }
    }
}
```
x??

---
#### DNS Round-Robining
Background context explaining the concept. DNS round-robin involves distributing traffic across multiple servers by having a single DNS entry point to all backend services, which can be problematic if one of the backends becomes unavailable.

:p What are the main issues with using DNS round-robin for service discovery?
??x
The main issues with using DNS round-robin include:
- Clients may route requests to unhealthy or sick hosts.
- Lack of visibility into individual host health, making it difficult to reroute traffic away from a failing instance.
- Increased complexity in managing and troubleshooting network configurations.

Code Example: An example of how DNS round-robin might be implemented incorrectly (not recommended).
```dns
accounts.musiccorp.com.     IN A 192.0.2.1
accounts.musiccorp.com.     IN A 192.0.2.2
```
x??

---
#### Conclusion on DNS for Service Discovery
Background context explaining the concept. While DNS is a reliable and well-understood method, it may not be the best fit for highly dynamic environments with frequently changing instances.

:p When might DNS still be a good choice for service discovery?
??x
DNS can still be a good choice when:
- The environment has static setups with no frequent deployments.
- There are few services that do not require high availability or load balancing.
- An existing infrastructure already supports DNS and the overhead of maintaining it is manageable.

However, in environments where instances are frequently deployed and destroyed, more advanced solutions like Consul may be necessary to ensure service discovery remains effective and up-to-date.
x??

---

