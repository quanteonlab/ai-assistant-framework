# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 34)

**Starting Chapter:** Materialized Views Federation and Query Virtualization

---

#### Apache Spark vs Hadoop MapReduce
Background context: The passage compares Apache Spark and Hadoop MapReduce, highlighting their differences in handling data processing tasks. While both frameworks support distributed computing, they differ in how they manage memory, disk usage, and overall performance.

:p What is the main difference between Apache Spark and Hadoop MapReduce?
??x
Apache Spark offers better performance due to its ability to cache data in memory, whereas Hadoop MapReduce relies heavily on disk operations. This means that Spark can process data much faster by keeping frequently accessed data in RAM, which reduces I/O overhead.
??x

---

#### In-Memory Processing and Disk Usage
Background context: The text emphasizes the importance of in-memory processing over traditional disk-based storage methods used in Hadoop MapReduce.

:p How does Apache Spark handle memory management differently from Hadoop MapReduce?
??x
Apache Spark leverages in-memory caching to store data in RAM, which speeds up processing times significantly. It allows for efficient data manipulation and reduces the need for frequent reads/writes to disk.
??x

---

#### Materialized Views
Background context: The passage introduces materialized views as a technique that precomputes query results to improve performance.

:p What is a materialized view?
??x
A materialized view is a database object that stores the result of a query. Unlike regular views, it precomputes and caches the data, allowing for faster retrieval when queried.
??x

---

#### Composable Materialized Views
Background context: The text discusses the limitations of traditional materialized views and introduces composable materialized views as an advanced technique.

:p What is the difference between a traditional materialized view and a composable one?
??x
A traditional materialized view cannot select from another materialized view, while a composable one can. This allows for more complex transformations and chaining of precomputed results.
??x

---

#### Federation Queries
Background context: The passage explains how federated queries enable OLAP databases to query external data sources.

:p What are federation queries?
??x
Federation queries allow an OLAP database to select from multiple external data sources, such as object storage or relational databases. They combine results from different sources into a unified result set.
??x

---

#### Data Virtualization
Background context: The text describes how data virtualization abstracts away the underlying data storage and provides a unified interface for querying.

:p What is data virtualization?
??x
Data virtualization refers to a system that does not store data internally but instead processes queries against external sources. This allows for flexible querying of various data sources without needing to move or replicate data.
??x

#### Query Pushdown
Background context explaining query pushdown. This technique involves moving as much work as possible to the source databases, thereby offloading computation from virtualization layers and potentially reducing data transfer over the network.

Engineers often use this approach with data virtualization tools like Trino to improve performance by leveraging the native capabilities of underlying systems. Filtering predicates are pushed down into queries on source databases whenever feasible.

:p What is query pushdown in the context of database management?
??x
Query pushdown refers to the practice of moving as much work, particularly filtering operations, down to the source databases where it can be executed more efficiently.
x??

---
#### Data Virtualization
Background context explaining data virtualization. It involves abstracting away barriers between different data sources and presenting them in a unified view without physically consolidating the data.

Data virtualization is useful for organizations dealing with data scattered across various systems, but should be used judiciously to avoid overloading production databases with analytical workloads.

:p How does data virtualization help manage data from multiple sources?
??x
Data virtualization helps by creating a unified view of data stored in disparate sources without physically moving or consolidating the data. This abstraction allows different parts of an organization to access and use data as if it were stored in one place.
x??

---
#### Streaming Transformations vs Queries
Background context explaining the difference between streaming transformations and queries. Both run dynamically, but while queries present a current view of data, transformations aim to prepare data for downstream consumption.

:p What is the key difference between streaming transformations and streaming queries?
??x
The key difference lies in their purpose: streaming queries are used to provide real-time views of data, whereas streaming transformations focus on enriching or modifying incoming streams to prepare them for further processing.
x??

---
#### Streaming DAGs (Directed Acyclic Graph)
Background context explaining the concept of a streaming DAG. This idea is about dynamically combining and transforming multiple streams in real time.

A simple example involves merging website clickstream data with IoT data, then preprocessing each stream into a standard format before enriching them to provide a unified view.

:p What is a streaming DAG?
??x
A streaming DAG (Directed Acyclic Graph) represents the dynamic combination and transformation of multiple streams in real-time. It allows for complex operations on incoming streams, such as merging, splitting, and enriching data.
x??

---
#### Micro-batch vs True Streaming
Background context explaining the difference between micro-batch and true streaming approaches. Micro-batching involves breaking down long-running processes into smaller batches to improve performance.

:p What is the main difference between micro-batch and true streaming?
??x
The main difference lies in their approach to processing: micro-batching breaks down large tasks into smaller, more manageable chunks (micro-batches), whereas true streaming processes data events one at a time with minimal batch intervals.
x??

---
#### Code Example for Streaming DAG
Background context explaining how Pulsar simplifies the creation of streaming DAGs.

:p Provide an example in pseudocode for defining a simple streaming DAG using Pulsar.
??x
```java
public class SimpleStreamingDAG {
    // Define topics and transformations
    Topic clickstreamTopic = createTopic("clickstream");
    Topic iotTopic = createTopic("iot");

    // Process clickstream data
    Stream<String> clickstreamStream = consume(clickstreamTopic);

    // Preprocess clickstream data
    clickstreamStream.map(event -> enrichClickstreamEvent(event));

    // Process IoT data
    Stream<DeviceEvent> iotStream = consume(iotTopic);

    // Enrich IoT events with metadata
    iotStream.joinOnMetadata(deviceId -> enrichIoTEventWithMetadata(deviceId));

    // Combine enriched streams
    clickstreamStream.merge(iotStream);
}
```
The code demonstrates defining topics, consuming and processing streams, and combining them to create a streaming DAG.
x??

---

#### Window Frequency and Latency Considerations
Window frequency refers to how often a batch of data is processed, while latency denotes the delay between when an event occurs and when it is included in the analysis. For Black Friday sales metrics, micro-batches are suitable if updates occur every few minutes. However, for critical operations like DDoS detection, true streaming with lower latency may be necessary.

:p What frequency should be used for processing Black Friday sales metrics?
??x
Micro-batch processing with a batch frequency that matches the update interval (e.g., every few minutes) is appropriate for Black Friday sales metrics. This ensures timely aggregation of data without excessive resource consumption.
x??

---

#### True Streaming vs Micro-Batch Processing
True streaming processes events as soon as they arrive, whereas micro-batch processing collects events in batches before processing them. The choice depends on the requirements: true streaming offers lower latency but may consume more resources.

:p When would you use true streaming over micro-batch processing?
??x
True streaming should be used when real-time insights are critical, such as detecting DDoS attacks or financial market anomalies. Micro-batch processing is preferable for periodic updates where slight delays are acceptable and resource efficiency is a concern.
x??

---

#### Domain Expertise and Real-World Testing
Domain expertise and real-world testing are crucial in choosing the right data processing strategy. Vendors often provide benchmarks that may not accurately reflect real-world performance.

:p Why is domain expertise important when deciding between true streaming and micro-batch processing?
??x
Domain expertise ensures that the chosen data processing strategy aligns with specific business needs, whereas real-world testing validates whether a solution performs as expected in actual operational conditions. Vendors might overstate their technology's capabilities through cherry-picked benchmarks.
x??

---

#### Data Engineer Responsibilities
Data engineers are involved in designing, building, and maintaining systems that query and transform data. They also implement data models within these systems.

:p What is the role of a data engineer during transformations?
??x
The data engineer designs, builds, and maintains systems for querying and transforming data while implementing data models. This involves ensuring the integrity and reliability of the data processing pipeline.
x??

---

#### Upstream Stakeholders and Data Sources
Upstream stakeholders include those who control business definitions and logic as well as engineers managing the source systems generating data.

:p Who are the upstream stakeholders when dealing with transformations?
??x
Upstream stakeholders are the business owners defining logic and controls, along with the engineers responsible for the systems that generate the raw data. Engaging with both groups ensures a comprehensive understanding of data sources and requirements.
x??

---

