# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 18)

**Starting Chapter:** APIs

---

#### Time-Series Databases Overview
Time-series databases are specialized for handling data generated at regular intervals or events, often used in IoT and log-based applications. They handle high write volumes and support fast read/write operations through memory buffering.

:p What is a time-series database?
??x
A time-series database is designed to store and manage large volumes of timestamped data from sensors, logs, or other event-based systems. These databases optimize for high write throughput and fast access times by using in-memory buffers and efficient storage mechanisms.
x??

---

#### Measurement vs. Event-Based Data
Measurement data are regularly generated, like sensor readings (temperature), while event-based data is irregular, such as motion detection.

:p How do measurement and event-based data differ?
??x
Measurement data typically comes from sensors that generate regular updates at fixed intervals or based on specific conditions, whereas event-based data is created sporadically when a particular event occurs. For example, temperature readings are continuous measurements, while a motion sensor triggers an event only when it detects movement.
x??

---

#### Schema Characteristics of Time-Series Databases
Time-series databases typically store a timestamp and a few key fields, with the data ordered by timestamps.

:p What is typical schema for time-series databases?
??x
The schema in time-series databases usually includes a timestamp and a small set of fields. The data is ordered based on the timestamp, making it ideal for operational analytics but less suitable for business intelligence (BI) use cases.
x??

---

#### Time-Series Databases Use Cases
Time-series databases are used in various applications like IoT, event logs, ad tech, and fintech where real-time data processing and storage are critical.

:p In which scenarios would you typically use a time-series database?
??x
You would typically use a time-series database for applications involving high-frequency data collection from devices (IoT), monitoring systems (ad tech, fintech), and log management. These databases excel in handling write-heavy workloads with fast read/write access.
x??

---

#### REST API Paradigm
REST stands for representational state transfer, defined by Roy Fielding in his PhD dissertation. It uses HTTP verbs like GET and PUT to make interactions stateless.

:p What is REST?
??x
REST (Representational State Transfer) is an architectural style for designing networked applications that defines principles and constraints for creating scalable web services. It leverages the HTTP protocol to manage statelessness, where each interaction operates independently without maintaining session-specific state.
x??

---

#### Stateless vs. Stateful Interactions in REST
In contrast to traditional sessions with associated state variables, REST interactions are stateless; they operate on a global system state rather than per-session state.

:p What is the key feature of stateless interactions in REST?
??x
The key feature of stateless interactions in REST is that each request from any client contains all the information needed by the server to understand and complete the request. There is no session or state maintained on the server, meaning each interaction operates independently.
x??

---

#### Client Libraries for REST APIs
Client libraries help developers interact with REST APIs more easily, handling tasks like authentication and mapping methods.

:p What role do client libraries play in interacting with REST APIs?
??x
Client libraries simplify interactions with REST APIs by abstracting away details such as authentication mechanisms and HTTP method mappings. They provide a higher-level API that makes it easier to work with the underlying implementation.
x??

---

#### GraphQL vs. REST
GraphQL allows for more flexible queries over multiple data models, whereas REST restricts query results to a specific model.

:p How does GraphQL differ from REST?
??x
GraphQL provides a query language for fetching data in a structured manner that can retrieve multiple data models in a single request, offering more flexibility compared to REST, which typically follows a predefined schema and returns only the specified data.
x??

---

#### Data Ingestion Pipelines with APIs
Setting up data ingestion pipelines from REST or GraphQL APIs involves using client libraries and managing synchronization tasks.

:p What tools can help set up data ingestion pipelines?
??x
Various tools, including client libraries for specific languages (like Python) and services like SaaS connectors or open-source libraries, simplify the setup of data ingestion pipelines. These tools handle tasks such as authentication, mapping API methods to classes, and managing data synchronization.
x??

---

#### Webhooks
Background context: Webhooks are a simple event-based data-transmission pattern where, when specific events happen in the source system, this triggers a call to an HTTP endpoint hosted by the data consumer. This is often referred to as "reverse APIs" because the connection goes from the source system to the data sink, rather than the typical API model which moves in the opposite direction.
:p What are webhooks and how do they differ from traditional APIs?
??x
Webhooks enable a source system to send data to a consumer through an HTTP endpoint when specific events occur. They allow for real-time updates and notifications as opposed to periodic polling or checks that traditional APIs might perform.

They differ from typical APIs in that:
- Webhooks are triggered by events, while APIs are usually called explicitly.
- Webhooks can be bidirectional (though often used unidirectionally), while most APIs are request-response.
- Webhooks send data directly when an event is fired, whereas APIs typically return a response after receiving a request.

Example of a webhook implementation:
```java
public class WebhookListener {
    public void handleEvent(String eventData) {
        // Code to process the incoming event and possibly trigger downstream processes
    }
}
```
x??

---
#### Reverse APIs
Background context: Reverse APIs, as mentioned in the text, refer to webhooks which are a type of API connection where data is sent from the source system to the data consumer. This contrasts with traditional client-server APIs where the client initiates requests and the server responds.
:p What do reverse APIs (webhooks) do?
??x
Reverse APIs (webhooks) allow data consumers to receive notifications or updates directly when specific events occur in the source system. They are "push" mechanisms, sending data to the consumer rather than requiring the consumer to make periodic requests.

Example of a webhook being used:
```java
public class DataConsumer {
    public void registerWebhook(EventHandler handler) {
        // Code to register the event handler to receive notifications when events occur
    }
}
```
x??

---
#### RPC and gRPC
Background context: Remote Procedure Call (RPC) is a programming paradigm in distributed computing, allowing you to run procedures on remote systems as if they were local. gRPC is an open-source framework for building efficient, modern, and high-performance services that use the HTTP/2 protocol.
:p What is gRPC?
??x
gRPC is a high-performance, open-source framework developed by Google for building robust distributed systems. It uses the Protocol Buffers data format to serialize messages and relies on HTTP/2 for transport.

Key features of gRPC include:
- Bidirectional streaming over a single TCP connection.
- Efficient use of CPU, power, battery life, and bandwidth through optimized protocols.
- Common client libraries available in multiple languages.

Example of a simple gRPC service definition:
```proto
syntax = "proto3";

package calculator;

service Calculator {
    rpc Multiply (Request) returns (Response);
}

message Request {
    int64 a = 1;
    int64 b = 2;
}

message Response {
    int64 result = 1;
}
```
x??

---
#### Message Queues and Event Streams
Background context: Message queues are used to ingest data at high velocity and volume, often in the context of event-driven architectures. They allow for decoupling between systems by queuing messages until a receiver is ready to process them.
:p What role do message queues play?
??x
Message queues act as intermediaries that temporarily store messages from producers (e.g., source systems) before they are processed by consumers (e.g., backend services). This helps in managing asynchronous and reliable data flows, ensuring that even if a consumer is not available when a producer sends a message, the message will be held until it can be processed.

Example of using a message queue:
```java
public class Producer {
    public void sendMessage(String message) {
        // Code to send the message to the queue
    }
}

public class Consumer {
    public void receiveMessage() {
        // Code to receive and process messages from the queue
    }
}
```
x??

---
#### Data Sharing in Multitenant Systems
Background context: In multitenant systems, data can be shared among multiple tenants while maintaining security policies. This is particularly useful in cloud environments where different organizations or teams need access to common datasets.
:p How does data sharing work in a multitenant system?
??x
Data sharing in multitenant systems allows for the secure and controlled exchange of data between different tenants (users, departments, or organizations) within a shared environment. Fine-grained permission systems ensure that only authorized users have access to specific parts of the data.

Example of setting up permissions:
```java
public class DataSharingManager {
    public void grantAccess(String tenantId, String dataSet, Set<String> roles) {
        // Code to set permissions for the specified tenant on a dataset with given roles
    }
}
```
x??

---

#### Data Sharing and Data Mesh

Background context: Data sharing can streamline data pipelines within an organization by allowing units to manage their data selectively, while still maintaining control over compute and query costs. This approach facilitates decentralized data management patterns such as data mesh.

:p What is the primary benefit of data sharing in organizations?
??x
The main benefit of data sharing is that it allows different units within an organization to manage their own data independently, share relevant information with other departments when necessary, and facilitate a more flexible and scalable architecture. This approach can lead to better utilization of resources and improved efficiency.
x??

---

#### Data Mesh

Background context: Data mesh involves decentralized data management where individual business domains have ownership over specific datasets. It emphasizes the use of common components to enable efficient data exchange and expertise sharing.

:p What is data mesh?
??x
Data mesh is a distributed data architecture that decentralizes data management by giving different business units or domains control over their own data assets. Each domain owns its data, and there are clear interfaces for data consumption and production.
x??

---

#### Third-Party Data Sources

Background context: Companies increasingly want to make their data available to customers and users, creating a "flywheel" effect where more data leads to better application integration and further data collection.

:p Why do companies make their data available?
??x
Companies make their data available because it encourages user adoption and usage. When users can integrate third-party data into their applications, they gain access to richer datasets, which in turn drives greater data volume. This cycle of increased data leads to more sophisticated applications, creating a flywheel effect that benefits both the company providing the data and its customers.
x??

---

#### APIs for Data Access

Background context: APIs are commonly used for direct third-party data access, enabling deep integration capabilities.

:p What is an API?
??x
An API (Application Programming Interface) is a set of rules and protocols for building software applications. It allows different software systems to communicate with each other. APIs often provide the means to pull and push data between systems, facilitating seamless integration.
x??

---

#### Message Queues

Background context: Message queues are used in event-driven architectures as mechanisms to asynchronously send small pieces of data (messages) between discrete systems.

:p What is a message queue?
??x
A message queue is an asynchronous communication mechanism that allows sending and receiving small messages between different systems using a publish-subscribe model. Data is published to the queue, and subscribers receive it based on their subscription criteria.
x??

---

#### Event-Streaming Platforms

Background context: Event-driven architectures use event-streaming platforms for near real-time analytics and data processing.

:p What are event-streaming platforms?
??x
Event-streaming platforms handle continuous streams of events in near real-time. They are crucial in event-driven architectures, where events can trigger work within applications and feed into real-time analytics.
x??

---

#### Practical Details of Message Queues

Background context: Message queues facilitate asynchronous communication by publishing messages that are consumed by subscribers.

:p How does a message queue work?
??x
A message queue works using the publish-subscribe model. Messages are published to the queue, and subscribers receive them based on their subscriptions. The subscriber acknowledges receipt of the message, which removes it from the queue.
```java
public class MessageQueueExample {
    public void sendMessage(String message) {
        // Publish the message to the queue
    }
    
    public void subscribe() {
        // Subscribe to the queue and listen for messages
    }
}
```
x??

---

#### Message Queues and Decoupling
Background context explaining that message queues allow applications and systems to be decoupled from each other, useful in microservices architectures. Message queues buffer messages for load spikes and ensure durability through replication.

:p What are the primary benefits of using message queues?
??x
The primary benefits include decoupling applications and systems, handling transient load spikes, ensuring message delivery durability through a distributed architecture with replication. These features make message queues essential in microservices architectures.
x??

---

#### Message Ordering and Delivery
Background context explaining that the order of messages can significantly impact downstream subscribers. Order guarantees vary depending on the technology used.

:p How does exactly once delivery work in message queues?
??x
Exactly once delivery means a message is sent only once, and after the subscriber acknowledges receipt, the message disappears and won’t be delivered again. This ensures no duplicates but requires idempotent processing of messages.
x??

---

#### Delivery Frequency and Idempotency
Background context explaining that messages can be sent exactly once or at least once, with idempotent systems ensuring consistent outcomes regardless of delivery frequency.

:p What is the significance of idempotency in message handling?
??x
Idempotency ensures that processing a message multiple times has the same outcome as processing it once. This is crucial for managing scenarios where messages might be delivered more than once, such as retries or failures before acknowledgments.
x??

---

#### Scalability and Message Queues
Background context explaining how popular message queues are horizontally scalable, buffering messages to handle load spikes and ensuring durability.

:p How do horizontally scalable message queues contribute to system resilience?
??x
Horizontally scalable message queues can run across multiple servers, allowing them to dynamically scale up or down. They buffer messages when systems fall behind and durably store messages for resilience against failure.
x??

---

#### Event-Streaming Platforms
Background context explaining that event-streaming platforms continue the functionality of message queues but with a focus on passing streams of data rather than just routing messages.

:p What differentiates event-streaming platforms from traditional message queues?
??x
Event-streaming platforms primarily differ in their handling of continuous data streams, focusing on processing and analyzing these streams in real-time. While traditional message queues route messages with specific delivery guarantees, event-streaming platforms handle the flow of events more continuously and often in a streaming fashion.
x??

---

#### Event-Streaming Platform Overview

In an event-streaming platform, data is ingested and processed as a continuous stream of events. These platforms are designed to handle real-time or near-real-time data processing, where messages are retained for a certain period, allowing for message replay from past points in time.

:p What is the primary function of an event-streaming platform?
??x
An event-streaming platform ingests and processes data as a stream of events. It retains these events for a specified duration, enabling replay functionality.
x??

---

#### Topics in Event-Streaming Platforms

Topics in an event-streaming platform are used to group related events. Producers send events to topics, which can be consumed by multiple subscribers.

:p What is the role of a topic in an event-streaming platform?
??x
A topic serves as a collection of related events. Producers stream events to these topics, and multiple consumers can subscribe to them for processing.
x??

---

#### Event Structure

An event consists of three main components: a key, value, and timestamp. These elements provide context and timing information about the event.

:p What are the three main components of an event in an event-streaming platform?
??x
The three main components of an event are:
- Key: Identifies the specific event.
- Value: Contains the data or details of the event.
- Timestamp: Indicates when the event occurred.
x??

---

#### Stream Partitions

Stream partitions divide a stream into multiple streams to enable parallel processing and increase throughput. Messages with the same partition key are guaranteed to be processed together.

:p What is the purpose of stream partitions in an event-streaming platform?
??x
The purpose of stream partitions is to enhance parallelism and improve the throughput of data processing. By dividing the stream, messages with the same partition key are always processed together, ensuring consistency.
x??

---

#### Example Event

An example event for an ecommerce order might look like this:
```json
{
   "Key":"Order # 12345",
   "Value":"SKU 123, purchase price of $100",
   "Timestamp":"2023-01-02 06:01:00"
}
```

:p What is an example event for an ecommerce order in an event-streaming platform?
??x
An example event for an ecommerce order might look like this:
```json
{
   "Key":"Order # 12345",
   "Value":"SKU 123, purchase price of $100",
   "Timestamp":"2023-01-02 06:01:00"
}
```
This event contains:
- Key: Identifies the specific order.
- Value: Details about the items purchased and their total cost.
- Timestamp: The time when the order was placed.
x??

---

#### Subscribers in an Event-Driven System

Subscribers, such as fulfillment and marketing, consume events from topics. Fulfillment uses events to trigger processes, while marketing performs real-time analytics or trains ML models.

:p How do subscribers use events in an event-driven system?
??x
In an event-driven system, subscribers consume events from topics for different purposes:
- Fulfillment: Uses events to trigger and manage the fulfillment process.
- Marketing: Performs real-time analytics on events or trains/uses ML models based on them.
x??

---

#### Order Processing Example

An order-processing system generates events that are published to a topic named `web orders`. Two subscribers—fulfillment and marketing—pull events from this topic.

:p How does the order-processing system work with an event-streaming platform?
??x
The order-processing system works by:
1. Generating events for each order.
2. Publishing these events to a topic called `web orders`.
3. Subscribers, such as fulfillment and marketing, consume these events for different purposes (fulfillment: process orders; marketing: analyze data or train ML models).
x??

---

