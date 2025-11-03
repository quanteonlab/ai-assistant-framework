# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 36)

**Starting Chapter:** Partitioned Logs

---

#### Redelivery and Message Ordering
Background context explaining how load balancing, redelivery, and consumer crashes affect message ordering. The JMS and AMQP standards aim to preserve message order, but combining these features with load balancing can result in reordered messages.

:p How does load balancing combined with redelivery affect the processing order of messages?
??x
When a consumer, such as Consumer 2, crashes while processing a message (e.g., m3), the message is redelivered to another consumer that was not originally handling it. For example, if Consumer 1 was processing m4 at the same time and m3 was then redelivered, Consumer 1 would process messages in the order of m4, m3, m5. This results in a reordering of m3 and m4 compared to their original sending order.

```java
// Pseudocode for handling message delivery with load balancing and redelivery
public class MessageProcessor {
    void processMessage(Message msg) {
        try {
            // Process the message
        } catch (Exception e) {
            // Redeliver the message if a crash occurs
            broker.redeliverMessage(msg);
        }
    }
}
```
x??

---

#### Separate Queues Per Consumer
Background context explaining that using separate queues for each consumer can avoid message reordering issues. This approach ensures messages are processed in order without interference from other consumers.

:p How does using separate queues per consumer prevent message reordering?
??x
Using separate queues for each consumer means that each consumer processes its own queue independently, avoiding the interference and redelivery issues caused by load balancing. Since each consumer only receives messages intended for it, there is no risk of message reordering or loss.

```java
// Pseudocode to configure a separate queue per consumer
public class ConsumerConfigurator {
    void configureSeparateQueues() {
        for (Consumer consumer : consumers) {
            broker.createQueueFor(consumer);
        }
    }
}
```
x??

---

#### Message Reordering and Dependencies
Background context explaining the potential issues with message reordering, particularly when messages have causal dependencies. This is important in stream processing where the order of operations matters.

:p Why can message reordering be problematic?
??x
Message reordering can be problematic because it violates the expected sequence of operations, especially when messages are causally dependent on each other. For instance, in a financial application, processing a payment request before its corresponding account balance update could lead to incorrect state changes. The JMS and AMQP standards aim to preserve message order, but combining load balancing with redelivery can disrupt this order.

```java
// Pseudocode for handling causally dependent messages
public class FinancialProcessor {
    void processPaymentRequest(PaymentRequest request) {
        // Update the account balance
        AccountService.updateBalance(request.accountId, -request.amount);
        
        // Send confirmation message
        Message confirmMsg = MessageFactory.createConfirmationMessage();
        broker.sendMessage(confirmMsg); // Ensures order with respect to request
    }
}
```
x??

---

#### Partitioned Logs and Durable Storage
Background context explaining the difference between durable storage (like databases) and transient messaging, where messages are typically deleted after processing. This section discusses log-based message brokers as a potential hybrid solution.

:p What is the main difference between database storage and transient messaging in terms of persistence?
??x
The main difference lies in how data persists. Databases and filesystems store information permanently until explicitly deleted, making derived data creation more predictable and repeatable. In contrast, traditional message brokers delete messages after delivery to consumers, as they operate under a transient messaging model where the focus is on low-latency notifications rather than durable storage.

```java
// Pseudocode for log-based message broker setup
public class LogBasedBroker {
    void start() {
        // Initialize persistent logs
        File[] logs = createLogs();
        
        // Start appending and reading from logs
        new Thread(() -> appendMessages(logs)).start();
        new Thread(() -> readMessages(logs)).start();
    }
    
    private void appendMessages(File[] logs) {
        while (true) {
            Message msg = producer.sendMessage();
            for (File log : logs) {
                writeLog(log, msg);
            }
        }
    }

    private void readMessages(File[] logs) {
        while (true) {
            File lastLog = getLatestLog(logs);
            if (!lastLog.exists()) continue;
            
            Message msg = readNextMessage(lastLog);
            consumer.receiveAndProcess(msg);
        }
    }
}
```
x??

---

#### Log-Based Message Brokers
Background context explaining the concept of using logs for message storage, which can combine the benefits of durable storage with low-latency messaging.

:p How does a log-based message broker work?
??x
A log-based message broker stores messages in an append-only sequence on disk (a log). Producers send messages by appending them to the end of the log, and consumers read these logs sequentially or wait for notifications when new messages are appended. The Unix tool `tail -f` works similarly by watching a file for data being appended.

```java
// Pseudocode for using logs in message broker
public class LogBasedBroker {
    void processMessage() {
        while (true) {
            Message msg = tailLog();
            if (msg != null) {
                consumer.receiveAndProcess(msg);
            } else {
                waitUntilNewData(); // Wait until new data is appended
            }
        }
    }

    private Message tailLog() {
        for (File log : logs) {
            if (!log.exists()) continue;
            
            Message msg = readNextMessage(log);
            if (msg != null) return msg;
        }
        return null;
    }
}
```
x??

---
#### Partitioning Log Files for Scalability
Background context: To handle higher throughput, log files can be partitioned across different machines. Each partition acts as a separate append-only file where messages are stored sequentially with increasing offsets.

:p How does partitioning help in handling higher throughput?
??x
Partitioning helps by distributing the load across multiple machines, allowing for concurrent reads and writes to different partitions. This approach increases overall system throughput.
```java
// Pseudocode for adding a message to a partitioned log
void appendMessage(String topic, String partitionId, Message message) {
    // Get or create the partition file
    File partitionFile = getPartitionFile(topic, partitionId);
    
    // Append the message with an increasing offset
    long nextOffset = getNextAvailableOffset(partitionFile);
    write(message, nextOffset, partitionFile);
}
```
x??
---

#### Defining Topics as Groups of Partitions
Background context: A topic is a group of partitions that handle messages of the same type. This abstraction allows for efficient processing and scaling.

:p How does defining topics as groups of partitions benefit message processing?
??x
Defining topics helps in organizing similar types of messages together, making it easier to manage and process them. Each partition within a topic can be independently managed and scaled.
```java
// Pseudocode for assigning partitions to consumers
void assignPartitions(List<Partition> partitions, ConsumerGroup group) {
    // Assigning partitions based on the consumer's availability or load balancing strategy
    List<PartitionAssignment> assignments = getAssignments(group, partitions);
    
    // Notify each consumer about its assigned partitions
    notifyConsumers(assignments);
}
```
x??
---

#### Load Balancing Across Consumers
Background context: In a log-based message broker, consumers can be assigned entire partitions to process. This approach ensures that processing is done in a sequential manner within the partition, which simplifies offset management.

:p How does assigning entire partitions to consumers facilitate load balancing?
??x
Assigning entire partitions to consumers allows for coarse-grained load balancing where each consumer handles all messages in its assigned partition. This reduces complexity compared to individually assigning messages, as it ensures that processing is done sequentially within a partition.
```java
// Pseudocode for sequential message consumption from a partition
void processPartition(Partition partition) {
    long offset = 0;
    
    while (offset < getNextAvailableOffset(partition.file)) {
        Message message = readMessage(offset, partition.file);
        
        // Process the message
        process(message);
        
        // Move to next offset
        offset++;
    }
}
```
x??
---

#### Handling Head-of-Line Blocking
Background context: In a single-threaded processing model for partitions, slow messages can hold up the processing of subsequent messages within the same partition. This is known as head-of-line blocking.

:p What is head-of-line blocking in the context of message processing?
??x
Head-of-line blocking occurs when a slow-consuming process holds up the processing of other messages in the same partition because the broker processes messages sequentially within a partition.
```java
// Pseudocode for handling head-of-line blocking
void consumePartitionMessage(Partition partition, Message message) {
    // Check if the current message is slow to process
    if (isSlowToProcess(message)) {
        // Buffer or pause processing until the slow message is completed
        buffer(message);
    } else {
        // Process the message immediately
        process(message);
    }
}
```
x??
---

#### JMS/AMQP vs Log-Based Approach

Background context: The passage discusses when to use a JMS/AMQP style of message broker versus a log-based approach for handling messages. Key factors include the expense of processing messages, the importance of message ordering, and high throughput scenarios.

:p What are the key situations where a JMS/AMQP style of message broker is preferable?

??x
A JMS/AMQP style of message broker is more suitable when messages may be expensive to process, and you want to parallelize processing on a per-message basis. Additionally, if message ordering is not critical, this approach is preferred over the log-based method.

Example scenario:
```java
// Scenario where JMS/AMQP is used for parallel processing
public class MessageProcessor {
    public void processMessage(String message) {
        // Code to process each message in parallel
    }
}
```
x??

---

#### Consumer Offsets and Log-Based Systems

Background context: The passage explains how consumer offsets work in log-based systems. It highlights that sequential consumption of partitions makes it easy to track which messages have been processed, reducing the need for tracking acknowledgments for every single message.

:p How do consumer offsets help in log-based systems?

??x
Consumer offsets simplify the tracking of processed messages by allowing the system to know that all messages with an offset less than a consumer's current offset have already been processed. The broker does not need to track individual message acknowledgments, only periodic recording of consumer offsets. This reduces bookkeeping overhead and increases throughput.

Example logic:
```java
// Example function to check if a message is processed
public boolean isMessageProcessed(int offset) {
    return this.currentOffset > offset;
}
```
x??

---

#### Disk Space Usage in Log-Based Systems

Background context: The passage discusses the management of disk space in log-based systems, noting that logs are divided into segments and old segments can be deleted or moved to archive storage. It explains how this allows for a buffer of messages before older ones start getting overwritten.

:p How does the log-based system manage disk space?

??x
The log is segmented, and periodically old segments are deleted or moved to archive storage. This creates a bounded-size buffer that discards old messages when full. The buffer size is limited by available disk space, allowing for buffering of several days' worth of messages.

Example calculation:
```java
// Back-of-the-envelope calculation for disk space usage
public long calculateDiskSpace(int driveCapacityGB, int writeThroughputMBPS) {
    return (driveCapacityGB * 1024L * 1024 / writeThroughputMBPS);
}
```
x??

---

#### Handling Slow Consumers

Background context: The passage explains how log-based systems handle slow consumers by effectively dropping old messages that go beyond the size of the buffer. It also mentions monitoring consumer offsets to alert operators when a significant backlog occurs.

:p What happens if a consumer falls behind in processing?

??x
If a consumer falls so far behind that it requires older messages than those retained on disk, it will not be able to read these messages due to the bounded-size buffer. The system effectively drops old messages as they go beyond the buffer's capacity. Operators can monitor how far the consumer is behind and alert when significant delays occur.

Example monitoring logic:
```java
// Example function to check if a consumer is significantly behind
public boolean isConsumerBehind(long currentOffset, long requiredOffset) {
    return (currentOffset - requiredOffset) > BUFFER_SIZE;
}
```
x??

---

#### Log-Based Message Brokers vs. Traditional Message Brokers
Log-based message brokers store messages in a persistent log, allowing consumers to replay old messages without affecting other consumers or services. This is different from traditional message brokers where shutting down a consumer can disrupt service due to accumulated messages in queues.

:p What is the key difference between log-based and traditional message brokers when it comes to handling consumer shutdowns?
??x
Log-based message brokers allow a single consumer to stop consuming without disrupting other consumers or services. Traditional message brokers may leave lingering messages that continue to take up resources, potentially affecting active consumers.
```java
// Example of how offsets work in log-based messaging
public void consumeMessages() {
    // Consumer starts from offset X and processes messages sequentially
    for (Message msg : log.getMessageStream(offset)) {
        process(msg);
        moveOffsetForward(); // Update the offset to the next message
    }
}
```
x??

---

#### Event Streams and Databases
In a database context, events can be seen as records of data changes that need to be processed. Replication logs in databases are streams of write events, ensuring replicas end up in the same state through deterministic processing.

:p How do event streams relate to database replication?
??x
Event streams describe data changes in a database. In replication, these events are processed by followers, which apply them to their own copies of the database to maintain consistency across all replicas.
```java
// Pseudocode for applying write events from a leader to a follower
public void applyWriteEvents(List<WriteEvent> events) {
    for (WriteEvent event : events) {
        apply(event); // Apply each write event to the local database copy
    }
}
```
x??

---

#### Synchronization Across Heterogeneous Systems
In complex applications, different systems store data in optimized representations. Ensuring these systems stay synchronized can be challenging due to race conditions and inconsistent updates when using dual writes.

:p What is a common challenge with maintaining consistency across heterogeneous systems?
??x
Race conditions and inconsistencies arise when multiple clients concurrently update the same piece of data. Dual writes may result in different systems holding outdated or conflicting versions of the same data.
```java
// Pseudocode for handling concurrent updates to ensure consistency
public void handleConcurrentWrites() {
    // Ensure both database and search index are updated atomically
    if (updateDatabase(newValue)) {
        updateSearchIndex(newValue);
    }
}
```
x??

---

#### Atomic Commit Problem in Synchronization
Ensuring that changes across multiple systems succeed or fail together is challenging. The atomic commit problem, often solved using Two-Phase Commit (2PC), can be expensive and complex.

:p What is the atomic commit problem in the context of data synchronization?
??x
The atomic commit problem arises when updates need to be applied consistently across multiple systems. Ensuring that all systems either succeed or fail together requires coordination mechanisms like 2PC, which can be resource-intensive.
```java
// Pseudocode for a simplified Two-Phase Commit (2PC)
public void twoPhaseCommit() {
    if (prepare()) { // Prepare the transaction on both sides
        commit(); // Commit on both sides
    } else {
        abort(); // Abort on both sides
    }
}
```
x??

---

#### Change Data Capture (CDC) Overview
Change data capture involves observing and extracting all data changes made to a database so they can be replicated to other systems. Historically, databases have treated their replication logs as internal implementation details, making it difficult for clients to leverage them directly.

:p What is the main challenge with traditional database replication logs?
??x
The primary challenge is that databases traditionally consider their replication logs as internal and not exposed as a public API. Clients are expected to interact with the database through its data model and query language rather than parsing the log to extract changes.
x??

---

#### Use of CDC in Stream Processing
CDC allows for continuous application of changes to other systems, ensuring consistency across different storage technologies like search indices, caches, or data warehouses.

:p How can change data capture be used to maintain consistency between a database and another system?
??x
Change data capture can be used by continuously applying the changes from one database (the source) to another system (the target). For instance, capturing changes in a database and applying them in real-time to a search index ensures that both systems remain consistent. This process can be implemented using log-based message brokers or through triggers within the database.
x??

---

#### Implementation of Change Data Capture
Implementing CDC often involves setting up mechanisms like log consumers (derived data systems) that observe changes from the source database and apply them to other systems.

:p What are some common methods for implementing change data capture?
??x
Common methods include using database triggers, parsing replication logs, or leveraging specialized tools. For example, LinkedIn's Databus, Facebook’s Wormhole, and Yahoo’s Sherpa use these techniques at scale. PostgreSQL's Bottled Water uses an API to decode write-ahead logs, while Maxwell and Debezium parse MySQL binlogs.
x??

---

#### Asynchronous Nature of CDC
CDC systems are typically asynchronous, meaning the source database does not wait for changes to be applied to consumers before committing.

:p What is a key characteristic of change data capture implementations?
??x
A key characteristic is its asynchronous nature. The system of record commits changes without waiting for acknowledgment from downstream consumers. This approach provides operational flexibility but introduces challenges related to replication lag.
x??

---

#### Initial Snapshot in CDC
Initial snapshot involves taking a full copy of the database state before applying logs, ensuring complete data consistency.

:p What role does an initial snapshot play in change data capture?
??x
An initial snapshot is crucial for obtaining a consistent starting point. Without it, only incremental changes may miss out on older or outdated data. For example, building a new full-text index requires the entire database copy to ensure all data is included.
x??

---

#### Log Compaction in CDC
Log compaction helps manage storage by keeping only recent updates and removing duplicates.

:p What technique can be used to reduce log size while maintaining data integrity in change data capture?
??x
Log compaction can help reduce storage requirements. It periodically identifies duplicate records, retains the most recent update for each key, and discards older versions. For instance, a tombstone (a null value) might indicate deletions during log compaction.
x??

---

#### API Support for Change Streams
Modern databases are increasingly supporting change streams as a first-class interface.

:p How do modern databases support change data capture?
??x
Modern databases like RethinkDB, Firebase, CouchDB, and Meteor provide built-in mechanisms for change streams. For example, RethinkDB allows subscribing to notifications when query results change, while MongoDB's oplog is used by VoltDB for continuous data export.
x??

---

