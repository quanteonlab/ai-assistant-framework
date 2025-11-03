# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 39)

**Starting Chapter:** Summary

---

#### Stream-table join (stream enrichment)
Stream-table join is a technique where data from a stream is enriched by joining it with data stored in a table. This can be particularly useful when processing real-time or event-driven data, as it allows for dynamic and flexible data augmentation.

:p What is the purpose of using "stream-table join" in stream processing?
??x
The purpose of using "stream-table join" (or stream enrichment) is to enrich incoming streaming data with static or semi-static information from a table. This can enhance real-time analytics, enable complex event processing, and provide more context-rich events for decision-making processes.

For example, if you are processing transaction streams in financial applications, joining these transactions with customer profiles stored in a database would allow you to personalize the transaction messages with user-specific details.
??x

---

#### Local State Replication
Replicating state locally within stream processors can prevent data loss during recovery from failures. This approach ensures that when a task fails and is reassigned, it can resume processing where it left off without missing any events.

:p How does local state replication help in recovering from failures?
??x
Local state replication helps by keeping the state of operations local to each stream processor instance. When a failure occurs and the task is reassigned to a new instance, that new task can read the replicated state to resume processing. This prevents data loss because the state can be restored without needing external storage or coordination.

For example, in Apache Flink, operators maintain their states locally and periodically capture snapshots of these states, which are then stored durably. During recovery, the new task reads from the latest snapshot.
??x

---

#### Periodic State Snapshots
Periodic state snapshots involve capturing a consistent view of the operator's state at regular intervals and storing it in durable storage.

:p What is periodic state snapshotting used for?
??x
Periodic state snapshotting is used to ensure that during recovery from failures, the stream processor can resume processing from the latest known consistent state. This approach helps maintain data integrity by reducing the risk of partial or duplicate processing.

For instance, Apache Flink periodically takes snapshots of operator states and writes them to a durable storage like HDFS.
??x

---

#### Log Compaction
Log compaction is a mechanism where older log entries are discarded if newer ones with the same key overwrite them. This helps in managing large volumes of log data efficiently.

:p How does log compaction work?
??x
Log compaction works by retaining only the latest log entry for each unique key, effectively compacting the log data. When old log entries can be safely discarded because they have been superseded by newer ones, storage space is conserved, and processing overhead is reduced.

For example, in Kafka Streams, state changes are logged to a dedicated topic with log compaction enabled. This ensures that only the latest updates are retained.
??x

---

#### Message Brokers Comparison
Message brokers like AMQP/JMS-style message brokers and log-based message brokers serve different purposes and have distinct characteristics regarding how messages are handled.

:p What is an example use case for an AMQP/JMS-style message broker?
??x
An example use case for an AMQP/JMS-style message broker includes task queues where the exact order of message processing is not crucial, and there's no need to revisit processed messages. This type of broker assigns each message individually to a consumer, which acknowledges the message upon successful processing. Once acknowledged, the message is deleted from the queue.

```java
// Pseudocode for AMQP/JMS-style message broker interaction
public class TaskQueue {
    public void sendMessage(String message) {
        // Send the message to a specific consumer
    }

    public boolean acknowledgeMessage(String messageId) {
        // Acknowledge the receipt and processing of the message by the consumer
        return true;  // Return true if acknowledged successfully
    }
}
```
??x

---

#### Log-based Message Broker with Checkpointing
Log-based message brokers retain messages on disk, allowing for replay or checking progress through offsets. This is useful in scenarios where historical data needs to be accessed.

:p How does a log-based message broker ensure parallel processing?
??x
A log-based message broker ensures parallel processing by partitioning the stream of messages across multiple consumer nodes. Each consumer tracks its progress by checkpointing the offset of the last message it has processed. This allows for fault tolerance and scalability, as consumers can resume from their last known position without needing to start over.

For example, in Kafka Streams:
```java
// Pseudocode for a log-based message broker with checkpointing
public class LogBasedMessageBroker {
    public void assignPartitionToConsumer(String partition) {
        // Assign the given partition to the consumer
    }

    public boolean checkpointOffset(long offset) {
        // Checkpoint the current processing position of the consumer
        return true;  // Return true if checkpoint successful
    }
}
```
??x

---

#### Log-Based Approach Overview
Log-based approaches are similar to database replication logs and log-structured storage engines. They are particularly useful for stream processing systems that consume input streams and generate derived state or output streams. Streams can originate from various sources such as user activity events, periodic sensor readings, data feeds (e.g., financial market data), and even database changes.
:p What is the key similarity between the log-based approach and other storage methods?
??x
The log-based approach shares similarities with replication logs in databases and log-structured storage engines. It is especially useful for stream processing systems that need to derive state or output streams from input events.
x??

---

#### Representing Databases as Streams
Representing databases as streams allows for keeping derived data systems continually up-to-date by consuming the changelog of database changes. This can involve implicit change data capture or explicit event sourcing.
:p How does representing a database as a stream benefit derived data systems?
??x
Representing a database as a stream helps keep derived data systems (like search indexes, caches, and analytics) continuously updated. By consuming the changelog that captures all database changes, these derived systems can be kept in sync with the latest state of the data.
x??

---

#### Stream Processing Techniques
Stream processing involves several techniques such as complex event processing, windowed aggregations, and materialized views. These techniques help in searching for patterns, computing aggregations over time windows, and maintaining up-to-date views on derived data.
:p List three purposes of stream processing mentioned in the text.
??x
The three purposes of stream processing mentioned are:
1. Searching for event patterns (complex event processing)
2. Computing windowed aggregations (stream analytics)
3. Keeping derived data systems up to date (materialized views)
x??

---

#### Time Reasoning in Stream Processors
Stream processors must handle time reasoning, distinguishing between processing time and event timestamps. They also need to deal with straggler events that arrive after the window of interest is considered complete.
:p What are the two types of times that stream processors must distinguish?
??x
Stream processors must distinguish between:
1. Processing time: The time at which a message is processed by the system.
2. Event timestamps: The actual timestamp associated with the event, as recorded in the input data.
x??

---

#### Stream Joins
Stream joins can be categorized into three types based on their inputs: stream-stream joins (with self-joins), stream-table joins, and table-table joins. Each type involves different strategies for joining streams or tables to produce derived outputs.
:p Name the three types of joins mentioned in the text.
??x
The three types of joins mentioned are:
1. Stream-stream joins
2. Stream-table joins
3. Table-table joins
x??

---

#### Fault Tolerance and Exactly-Once Semantics
Techniques for achieving fault tolerance and exactly-once semantics involve methods to ensure that messages are processed reliably, even in the presence of failures or retries.
:p What is the goal of ensuring exactly-once semantics in stream processing?
??x
The goal of ensuring exactly-once semantics in stream processing is to guarantee that each message is processed exactly once, preventing duplicates and ensuring data integrity despite potential failures or retries.
x??

---

#### Future of Data Systems
In this final chapter, the discussion shifts towards envisioning how data systems and applications should be designed and built for the future. The aim is to explore ideas that could fundamentally improve reliability, scalability, and maintainability of applications.

The context drawn from St. Thomas Aquinas' philosophy highlights the importance of having a higher purpose or end goal beyond just preservation or maintenance. This principle extends to modern software development, where merely ensuring the survival of an application (e.g., keeping it running indefinitely) is not sufficient; instead, the focus should be on creating applications that are robust, correct, and capable of evolving over time.

The objective here is to start a productive discussion about potential improvements in designing future data systems. This includes approaches like fault-tolerance algorithms for reliability, partitioning strategies for scalability, and mechanisms for evolution and abstraction to enhance maintainability.
:p What does the author suggest as the primary goal for future applications?
??x
The author suggests that the primary goal for future applications should be to create robust, correct, evolvable, and ultimately beneficial systems. This means moving beyond just maintaining an application by ensuring it is reliable, scalable, and easy to maintain over time.
x??

---
#### Reliability in Future Data Systems
Reliability is a crucial aspect of modern data systems. The discussion here revolves around fault-tolerance algorithms that can help ensure the robustness of applications.

Fault-tolerance refers to the ability of a system to continue operating correctly even if some (but not all) of its components fail. Key techniques include replication, redundancy, and load balancing.
:p What is fault-tolerance in the context of future data systems?
??x
Fault-tolerance in the context of future data systems refers to the capability of an application or system to continue functioning correctly despite failures of some of its components. This involves strategies like replication (redundant copies of data and processes), redundancy, and load balancing.
x??

---
#### Scalability in Future Data Systems
Scalability is another critical aspect discussed for future applications. Partitioning strategies are mentioned as a way to improve scalability.

Partitioning involves breaking down large datasets or tasks into smaller chunks that can be processed independently. This not only improves performance but also enhances the ability of an application to handle increased load.
:p What does partitioning mean in the context of improving scalability?
??x
Partitioning means dividing large datasets or complex tasks into smaller, manageable parts that can be processed independently. This approach helps improve performance and scalability by allowing different components of a system to operate more efficiently and manage larger workloads without becoming overwhelmed.
x??

---
#### Maintainability in Future Data Systems
Maintainability is discussed as another important aspect for future applications. The chapter emphasizes the need for mechanisms that facilitate evolution and abstraction.

Mechanisms such as modular design, version control, and automation tools can significantly enhance maintainability by making it easier to modify and update systems over time.
:p What are some key mechanisms mentioned in improving maintainability?
??x
Key mechanisms mentioned in improving maintainability include:
- Modular design: Breaking down the system into manageable, reusable components.
- Version control: Tracking changes and managing different versions of code or configurations.
- Automation tools: Tools that help with repetitive tasks, reducing human error and increasing efficiency.

These mechanisms facilitate easier modifications and updates to systems over time.
x??

---
#### Evolution in Future Data Systems
Evolution involves the ability of an application or system to adapt and improve over time. The chapter suggests that this is essential for creating applications that are not just functional but also beneficial to humanity.

The idea here is to design applications that can evolve their functionalities, performance, and even purpose as technology advances.
:p What does evolution mean in the context of future data systems?
??x
Evolution in the context of future data systems means designing applications with the ability to adapt and improve over time. This involves creating systems that can be updated, optimized, and expanded upon as new technologies and needs arise, ensuring they remain relevant and beneficial.
x??

---

#### Data Integration Challenges
In this book, various solutions to data storage and retrieval problems have been discussed. For instance, different storage engines (log-structured storage, B-trees, column-oriented storage) and replication strategies (single-leader, multi-leader, leaderless) are mentioned in Chapters 3 and 5 respectively. Each solution has its pros, cons, and trade-offs.
:p What is the main challenge when choosing a solution for data integration?
??x
The primary challenge lies in mapping the appropriate software tool to specific usage circumstances. Given that no single piece of software can fit all possible scenarios due to varying use cases, it often requires combining different tools to meet diverse requirements.
For example, integrating an OLTP database with a full-text search index is common. While some databases include basic full-text indexing features (e.g., PostgreSQL), more complex searches may necessitate specialized information retrieval tools.
??x
---

#### Combining Specialized Tools for Data Integration
Data integration becomes increasingly challenging as the number of different representations of data increases. To handle diverse use cases, it is often necessary to integrate various types of software tools, such as databases and search indexes, with other systems like analytics platforms, caching layers, machine learning models, or notification systems.
:p How can one approach integrating an OLTP database with a full-text search index?
??x
To integrate an OLTP database with a full-text search index, you would typically need to maintain two separate data stores. The OLTP database can handle transactional operations efficiently, while the full-text search index provides fast keyword queries.
Here is an example of how this could be achieved using a simple Java-based integration layer:
```java
public class DataIntegrationLayer {
    private Database db;
    private SearchIndex search;

    public DataIntegrationLayer(Database db, SearchIndex search) {
        this.db = db;
        this.search = search;
    }

    // Method to insert data into both the database and the search index
    public void addData(String data) {
        db.insert(data);
        search.indexDocument(data);
    }

    // Method to perform a full-text search query
    public List<String> searchDocuments(String keyword) {
        return search.query(keyword);
    }
}
```
This integration layer ensures that updates in the OLTP database are reflected in the search index, maintaining consistency across both systems.
??x
---

#### Data Integration and Obscure Features
Background context: Software engineers often make statements about the necessity or lack thereof for certain technologies based on their personal experience. However, the range of applications and requirements can be vast, leading to subjective judgments. For example, one might consider a feature obscure if it is rarely needed in a specific domain but could be crucial in another.
:p What does this passage suggest about making broad statements regarding technology necessity?
??x
The passage suggests that making blanket statements like "99 percent of people only need X" or "don't need X" can be misleading because the range of different applications and requirements for data is extremely diverse. What one person might consider an unnecessary or obscure feature could be a central requirement for someone else.
x??

---

#### Dataflows Across an Organization
Background context: When dealing with data across multiple storage systems, understanding dataflows becomes crucial. Ensuring that data writes are managed correctly is essential to maintain consistency and avoid conflicts between different storage systems.
:p How can ensuring the correct order of data writes help in maintaining consistency?
??x
Ensuring the correct order of data writes helps maintain consistency by preventing conflicts between different storage systems. For example, if changes are first written to a system-of-record database and then applied to a search index in the same order, the search index will always be derived from the system-of-record, ensuring consistency (assuming no bugs in the software).

Example code:
```java
// Pseudocode for applying changes in the correct order
public void applyChanges(int orderId) {
    // Step 1: Write to the database first
    database.write(orderId, newOrderDetails);
    
    // Step 2: Apply changes to the search index next
    searchIndex.updateFromDatabase(orderId, newOrderDetails);
}
```
x??

---

#### Change Data Capture (CDC)
Background context: Change Data Capture is a technique used to capture and apply changes made to a database in real-time. This ensures that all derived systems are consistent with the source of truth.
:p What is change data capture (CDC) and why is it important?
??x
Change Data Capture (CDC) is a method for capturing and applying changes from a system-of-record database to other storage systems, such as search indices, in real-time. It ensures that all derived data systems are consistent with the source of truth.

Example:
```java
// Pseudocode for CDC
public void applyChanges(int orderId) {
    // Step 1: Capture and store changes in a change log
    changelogService.logChange(orderId, newOrderDetails);
    
    // Step 2: Apply these changes to the search index in the same order
    searchIndex.applyFromChangelog(orderId, newOrderDetails);
}
```
x??

---

#### Handling Concurrent Writes Without CDC
Background context: If data is written directly to multiple systems without using change data capture (CDC), conflicts can arise due to different processing orders.
:p What issue arises from allowing direct writes to both the database and search index by applications?
??x
Allowing direct writes to both the database and search index by applications introduces the problem of conflicting writes being processed in a different order. This leads to inconsistencies between the database and the search index.

Example:
```java
// Pseudocode for concurrent writes without CDC
public void applyConcurrentWrites() {
    // Client 1 writes to database then search index
    client1.writeToDatabase(orderId, newOrderDetails);
    client1.writeToSearchIndex(orderId, newOrderDetails);
    
    // Client 2 writes to search index then database
    client2.writeToSearchIndex(orderId, newOrderDetails);
    client2.writeToDatabase(orderId, newOrderDetails);
}
```
In this scenario, the two storage systems may make contradictory decisions and become permanently inconsistent with each other.

x??

---

#### Total Order Replication Approach
Background context: To ensure consistent data across multiple storage systems, a total order replication approach can be used. This involves deciding on an ordering for all writes through a single system.
:p How does the state machine replication approach help in ensuring consistency?
??x
The state machine replication approach ensures consistency by funneling all user input through a single system that decides the order of all writes. This makes it easier to derive other representations of data by processing the writes in the same order.

Example:
```java
// Pseudocode for total order replication
public void applyTotalOrderWrites() {
    // Single system decides on write ordering
    coordinatorService.processWrite(orderId, newOrderDetails);
    
    // Apply changes to both database and search index based on ordered writes
    database.applyWrite(orderId, newOrderDetails);
    searchIndex.applyFromCoordinator(orderId, newOrderDetails);
}
```
x??

---

#### Event Sourcing for Derived Data Systems
Background context: Event sourcing involves storing all state changes as a sequence of events. This can make it easier to update derived data systems deterministically and idempotently.
:p How does event sourcing help in updating derived data systems?
??x
Event sourcing helps in updating derived data systems by making the process deterministic and idempotent, which simplifies fault recovery. By storing all state changes as a sequence of events, it becomes easier to apply these changes consistently across different storage systems.

Example:
```java
// Pseudocode for event sourcing updates
public void updateDerivedSystem(int orderId) {
    // Apply events in the order they were recorded
    derivedData.applyEvents(orderId);
}
```
x??

---

#### Distributed Transactions vs. Derived Data Systems
Background context: Both distributed transactions and derived data systems aim to keep different data systems consistent, but they do so by different means. Distributed transactions enforce consistency through coordination, while derived data systems rely on a total order of writes.
:p How does the approach of using derived data systems compare to that of distributed transactions?
??x
The approach of using derived data systems compares to distributed transactions in terms of achieving similar goals (consistency across different storage systems) but via different means. Distributed transactions enforce consistency through coordination, while derived data systems rely on a total order of writes.

Example:
```java
// Pseudocode for distributed transaction
public void performDistributedTransaction() {
    // Begin transaction
    TransactionManager.begin();
    
    try {
        database.write(orderId, newOrderDetails);
        searchIndex.updateFromDatabase(orderId, newOrderDetails);
        
        // Commit transaction
        TransactionManager.commit();
    } catch (Exception e) {
        // Rollback transaction if something goes wrong
        TransactionManager.rollback();
    }
}
```

Example for derived data systems:
```java
// Pseudocode for total order replication
public void applyTotalOrderWrites() {
    coordinatorService.processWrite(orderId, newOrderDetails);
    
    database.applyWrite(orderId, newOrderDetails);
    searchIndex.applyFromCoordinator(orderId, newOrderDetails);
}
```
Both approaches aim to ensure consistency but use different mechanisms.

x??

---

---
#### Distributed Transactions and Ordering Mechanisms
Distributed transactions decide on an ordering of writes by using locks for mutual exclusion (see “Two-Phase Locking (2PL)” on page 257), while CDC and event sourcing use a log for ordering. Distributed transactions use atomic commit to ensure that changes take effect exactly once, while log-based systems are often based on deterministic retry and idempotence.
:p What mechanism do distributed transactions use for deciding the order of writes?
??x
Distributed transactions use locks for mutual exclusion (Two-Phase Locking) to decide the order of writes. This ensures that only one transaction can access a resource at a time, maintaining consistency across the system.
x??
---

---
#### Atomic Commit and Consistency Guarantees
Distributed transactions use atomic commit to ensure changes take effect exactly once, whereas log-based systems often rely on deterministic retry and idempotence for consistency. Transaction systems typically provide linearizability (see “Linearizability” on page 324), offering useful guarantees such as reading your own writes.
:p What does atomic commit in distributed transactions guarantee?
??x
Atomic commit in distributed transactions guarantees that changes take effect exactly once, ensuring a consistent state across the system. This means either all operations succeed or none do, maintaining integrity and consistency.
x??
---

---
#### Linearizability and Timing Guarantees
Transaction systems usually provide linearizability (see “Linearizability” on page 324), which implies useful guarantees such as reading your own writes (see “Reading Your Own Writes” on page 162). On the other hand, derived data systems are often updated asynchronously and do not offer the same timing guarantees.
:p What is linearizability?
??x
Linearizability is a consistency model in distributed systems where all operations appear to be executed atomically and sequentially. This means that any sequence of operations can be reordered without changing the outcome, ensuring strong consistency. It provides useful guarantees like reading your own writes, meaning that once you have written something, you will always see it.
x??
---

---
#### Limitations of Total Ordering
With systems that are small enough, constructing a totally ordered event log is entirely feasible (as demonstrated by databases with single-leader replication). However, as systems scale toward bigger and more complex workloads, limitations begin to emerge. These include the need for leader nodes in each datacenter to handle network delays efficiently.
:p What are the challenges of maintaining total ordering across multiple geographically distributed datacenters?
??x
Maintaining total ordering across multiple geographically distributed datacenters is challenging due to network latency and throughput constraints. To address this, separate leaders are typically used in each datacenter, but this introduces ambiguity in the order of events originating from different datacenters.
x??
---

---
#### Total Order Broadcast and Consensus Algorithms
Deciding on a total order of events is known as total order broadcast (equivalent to consensus). Most consensus algorithms assume sufficient throughput for a single node. Designing scalable consensus algorithms that work well in geographically distributed settings remains an open research problem.
:p What does total order broadcast refer to?
??x
Total order broadcast refers to the process of deciding on a total order of events, which is equivalent to solving the consensus problem. This ensures that all nodes agree on the sequence of events despite network delays and partitioning.
x??
---

---
#### Microservices and Event Ordering
When applications are deployed as microservices, each service and its durable state are deployed independently with no shared state between services. This can lead to undefined order of events originating from different services, making total ordering more difficult.
:p How does deploying applications as microservices affect event ordering?
??x
Deploying applications as microservices leads to independent units of deployment where services have their own state and do not share it. Consequently, events originating in different services lack a defined order, complicating the process of total ordering.
x??
---

---
#### Offline Client Operations and Event Ordering
Some applications maintain client-side state that is updated immediately on user input without waiting for server confirmation and can continue to work offline. This often leads to clients and servers seeing events in different orders.
:p How do client-side updates affect event ordering?
??x
Client-side updates occur independently of server confirmation, allowing immediate changes. However, this can result in clients and servers seeing events in different orders, making it difficult to maintain a consistent total order across the system.
x??
---

