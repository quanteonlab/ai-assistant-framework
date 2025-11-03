# Flashcards: 029---Understanding-Distributed-Systems-2021-Roberto-Vitillo--_processed (Part 13)

**Starting Chapter:** Guarantees

---

#### Broadcast Messaging
Broadcast messaging allows a producer to write a message to a publish-subscribe channel, which is then broadcasted to all consumers. This pattern is used for notifying a group of processes about a specific event. We have previously encountered this when discussing log-based transactions.

:p What is the key feature of broadcast messaging?
??x
The key feature of broadcast messaging is that it allows a single producer to send messages to multiple consumers without needing to know their addresses, number, or availability.
x??

---

#### Message Channel Implementation
Message channels are implemented using messaging services like AWS SQS or Kafka. These services act as buffers for messages, decoupling producers from consumers.

:p What role does the messaging service play in message channel implementation?
??x
The messaging service acts as a buffer for messages and decouples producers from consumers by abstracting away their addresses and availability.
x??

---

#### Tradeoffs and Guarantees
Different message brokers offer different tradeoffs such as delivery guarantees, message durability, latency, supported standards, and support for competing consumers. Some brokers do not guarantee strict order of messages due to the distributed nature of the implementation.

:p Why might a broker like AWS SQS standard queues not provide strong ordering guarantees?
??x
A broker like AWS SQS standard queues does not provide strong ordering guarantees because ensuring message order across multiple nodes is challenging and requires coordination. The distributed nature of the system makes it difficult to guarantee order, leading to simpler implementations that do not offer strict ordering.
x??

---

#### Partitioning for Ordering
To ensure message order, some brokers partition a channel into sub-channels where each sub-channel can be handled by a single process. This ensures that messages within a sub-channel are processed in the order they were sent.

:p How does partitioning help with message ordering?
??x
Partitioning helps with message ordering by dividing the channel into smaller sub-channels, each managed by a single process. Since only one consumer reads from each sub-channel, it becomes easier to guarantee that messages within the same sub-channel are processed in order.
x??

---

#### Drawbacks of Partitioning
While partitioning can help with order guarantees, it also introduces challenges such as potential hotspots where specific partitions may become overloaded.

:p What is a drawback of implementing message channels through partitioning?
??x
A drawback of implementing message channels through partitioning is that specific partitions might become much hotter than others. This can lead to situations where single consumers reading from these partitions struggle to keep up with the load, potentially degrading performance.
x??

---

#### Competing Consumers Pattern
The competing consumers pattern involves using leader election to allow only one consumer process to read from a sub-channel, ensuring message order is preserved.

:p How does the competing consumers pattern ensure message order?
??x
The competing consumers pattern ensures message order by allowing only one consumer process to read from each sub-channel. This is achieved through mechanisms like leader election, which ensures that only one process can consume messages from a partition at any given time.
x??

---

#### Broker Limits and Performance
Brokers have various limits such as the maximum supported size of messages and other constraints. These limitations affect the overall performance and scalability of message processing.

:p What factors might limit the performance of a broker?
??x
Factors that can limit the performance of a broker include the maximum supported size of messages, message durability guarantees, latency requirements, and support for different messaging standards. These limits impact how efficiently messages are processed and delivered.
x??

---

---
#### Channels and Message Delivery
Channels are point-to-point and support an arbitrary number of producers and consumers. Messages are delivered to consumers at least once, meaning a message may be processed more than once or not at all if there's a failure.

While a consumer is processing a message, the message remains persisted in the channel but other consumers cannot read it for the duration of the visibility timeout. The visibility timeout guarantees that if a consumer crashes while processing the message, the message will become visible to other consumers again when the timeout triggers. When the consumer is done processing the message, it deletes it from the channel, preventing it from being received by any other consumer in the future.

This guarantee is very similar to what cloud services such as Amazon’s SQS and Azure Storage Queues offer.
:p What are the key characteristics of a message in a channel regarding visibility timeout?
??x
The key characteristics include that while a consumer processes a message, it remains visible only to that specific consumer. Once the consumer finishes processing (or within the visibility timeout), the message becomes invisible again to other consumers. If the consumer crashes during this time, the message will be made visible again when the timeout triggers.
??x
This ensures that messages are not lost due to a crash but also prevents multiple consumers from attempting to process the same message simultaneously.
```java
// Example code for setting visibility timeout in pseudocode
void setVisibilityTimeout(Channel channel, Message msg) {
    // Logic to set visibility timeout on the message so it remains invisible
    // to other consumers while being processed by one consumer
}
```
x??
---

#### Exactly-once Processing Risk
A consumer must delete a message from the channel once it's done processing it, ensuring no duplication or loss. If a consumer deletes a message before processing it and crashes afterward, the message could be lost. Conversely, if a consumer only deletes a message after processing it and crashes, the same message might get reprocessed later.

Because of these risks, there is no such thing as exactly-once message delivery in practical implementations.
:p What are the potential risks when implementing exactly-once processing?
??x
The main risks include:
- Deleting a message before processing it: If a crash occurs after deletion but before processing, the message will be lost forever.
- Deleting a message only after processing it: A crash after processing and before deletion can cause the same message to be processed again.
```java
// Pseudocode for handling message processing with idempotence
class MessageProcessor {
    public void processMessage(Message msg) {
        // Process logic here
        markMessageAsProcessed(msg); // Marking as processed without deleting immediately
    }

    private void markMessageAsProcessed(Message msg) {
        // Logic to mark the message as processed (e.g., update a database)
    }
}
```
x??
---

#### Idempotent Messages for Simulating Exactly-once Delivery
To simulate exactly-once processing, consumers can require messages to be idempotent. This means that even if a message is delivered multiple times, executing it again will not change the state beyond what was achieved on the first execution.

:p How does requiring idempotence in messages help simulate exactly-once processing?
??x
Requiring messages to be idempotent allows for reprocessing without causing any side effects. If a message is processed more than once due to delivery failures, it will have no additional impact on the state since each operation behaves identically.

```java
// Example of an idempotent process method in pseudocode
class IdempotentProcess {
    public void executeMessage(Message msg) {
        // Process logic that ensures no side effects if called multiple times
    }
}
```
x??
---

#### Maximum Retry Limit
Background context: When a message processing consistently fails, it is necessary to limit the number of times a specific message can be read from the channel. This prevents the message from being picked up repeatedly and potentially causing data loss.
:p How does limiting the maximum number of retries ensure robustness in message processing?
??x
Limiting the maximum number of retries ensures that messages do not get stuck indefinitely due to processing failures. By setting a cap on how many times a message can be delivered, consumers are forced to handle them within a reasonable limit. Once this limit is reached, the message is moved to a dead-letter channel, where it can be inspected for debugging or resolved by human intervention.

```java
public class MessageProcessor {
    private int retryCounter = 0;

    public void processMessage(String message) {
        try {
            // Process the message logic here.
        } catch (Exception e) {
            if (++retryCounter < MAX_RETRIES) {
                throw new RuntimeException("Failed to process message, retrying...", e);
            }
            // Move to dead-letter channel after max retries
            moveToDeadLetterChannel(message);
        }
    }

    private void moveToDeadLetterChannel(String message) {
        // Logic for moving the message to a dead-letter channel.
    }
}
```
x??

---

#### Backlog Management
Background context: A messaging system introduces bi-modal behavior—no backlog and expected operation, or backlog buildup leading to degraded performance. Managing backlogs is crucial as they can consume significant resources and delay processing of healthy messages.
:p What are the reasons for a backlog in a messaging system?
??x
Backlogs in a messaging system can occur due to several reasons:
1. An increase in the number of producers or their throughput, leading to more messages than consumers can handle.
2. Consumers becoming slower at processing individual messages, reducing their deletion rate from the channel.
3. Consumers failing to process some messages, causing them to be repeatedly picked up by other consumers and eventually end up in the dead-letter channel.

These issues can create a negative feedback loop where healthy messages are delayed, and consumer resources are wasted on retrying failed messages.
x??

---

#### Fault Isolation
Background context: Problematic producers that emit "poisonous" messages can degrade the entire system. It is essential to isolate such producers before they affect other components of the system.
:p How can consumers handle problematic producers?
??x
Consumers can handle problematic producers by treating messages from specific users differently based on their consistency in failure. For instance, if messages from a particular user fail repeatedly, consumers can route these messages to an alternate low-priority channel instead of processing them directly.

```java
public class Consumer {
    public void processMessage(String message) {
        // Check the user ID or some other identifier.
        if (isNoisyUser(message)) {
            moveToAlternateChannel(message);
        } else {
            processRegularMessage(message);
        }
    }

    private boolean isNoisyUser(String message) {
        // Logic to identify noisy users based on message content.
        return true;
    }

    private void moveToAlternateChannel(String message) {
        // Route the message to a low-priority channel.
    }

    private void processRegularMessage(String message) {
        // Regular processing logic for normal messages.
    }
}
```
x??

---

#### Reference Plus Blob
Background context: Transmitting large binary objects (blobs) can be challenging due to size limitations in messaging channels. A common solution is to use object storage services and send URLs of the blobs via messages.
:p How does using a queue plus blob pattern work?
??x
The "queue plus blob" pattern involves uploading large binary objects to an object storage service like AWS S3 or Azure Blob Storage, then sending the URL of these blobs through the message broker. This approach allows for the transfer of large files while adhering to size limitations imposed by messaging systems.

```java
public class MessageProducer {
    public void sendMessageWithBlob(String blobUrl) throws Exception {
        // Send the blob URL via the message broker.
        String message = "URL: " + blobUrl;
        sendMessage(message);

        // Upload the actual file to object storage.
        uploadFileToStorage(blobUrl);
    }

    private void sendMessage(String message) {
        // Code for sending a message through the message broker.
    }

    private void uploadFileToStorage(String url) {
        // Code for uploading a file to an object storage service.
    }
}
```
x??

---

