# Flashcards: 2B005---Streaming-Systems_processed (Part 27)

**Starting Chapter:** Spark

---

#### Spark's Origins and Early Success
Background context: Apache Spark was developed around 2009 at UC Berkeley's AMPLab. It gained fame due to its ability to perform most calculations in memory, significantly improving performance over traditional Hadoop jobs by leveraging Resilient Distributed Datasets (RDDs). RDDs capture the lineage of data and allow for efficient recomputation after failures.

:p What was Spark’s initial contribution that made it famous?
??x
Spark's initial contribution was its ability to perform most calculations in memory, significantly improving performance over traditional Hadoop jobs. This was achieved using Resilient Distributed Datasets (RDDs), which capture the lineage of data and allow for efficient recomputation after failures.
x??

---

#### Spark Streaming Introduction
Background context: Tathagata Das, a graduate student at UC Berkeley’s AMPLab, realized that Spark's fast batch processing engine could be repurposed to handle streaming data. This led to the development of Spark Streaming in 2013.

:p How did Spark Streaming come into existence?
??x
Spark Streaming came into existence when Tathagata Das, a graduate student at UC Berkeley’s AMPLab, realized that Spark's fast batch processing engine could be repurposed to handle streaming data. This insight led to the development of Spark Streaming.
x??

---

#### Processing-Time Windowing in Spark Streaming
Background context: Spark Streaming initially supported only processing-time windowing, which was a significant limitation for use cases requiring event time or handling late data.

:p What was the main limitation of the original version of Spark Streaming?
??x
The main limitation of the original version of Spark Streaming (1.x variants) was that it provided support only for processing-time windowing. This meant that any use case that cared about event time, needed to deal with late data, or required out-of-order data could not be handled out of the box without additional user-implemented code.
x??

---

#### Microbatch vs True Streaming Debate
Background context: Spark Streaming uses a microbatch approach, which has been criticized for being less flexible than true streaming engines. However, its performance in terms of latency and throughput is still quite good.

:p What is the primary criticism against Spark Streaming’s microbatch architecture?
??x
The primary criticism against Spark Streaming's microbatch architecture is that it processes data at a global level, which limits flexibility compared to true streaming engines. Critics argue that this approach cannot achieve both low per-key latency and high overall throughput simultaneously.
x??

---

#### Spark Streaming's Impact on Stream Processing
Background context: Spark Streaming provided strong consistency semantics for in-order data or event-time-agnostic computations, making it a significant milestone in stream processing.

:p What was the key contribution of Spark Streaming to the field of stream processing?
??x
The key contribution of Spark Streaming was that it offered strong consistency semantics for in-order data or event-time-agnostic computations. This made it the first publicly available large-scale stream processing engine with correctness guarantees akin to batch systems.
x??

---

#### Current State of Spark and Spark Streaming
Background context: As of today, Spark 2.x variants are expanding on Spark Streaming's semantic capabilities while addressing some of its limitations through a new true streaming architecture.

:p What is the current direction of development for Spark?
??x
The current direction of development for Spark includes expanding on Spark Streaming’s semantic capabilities in Spark 2.x variants. These newer versions incorporate many parts of the model described in this book and attempt to simplify complex pieces. Additionally, there are efforts to develop a new true streaming architecture to address microbatch criticisms.
x??

---

#### MillWheel Overview
Background context explaining the concept of MillWheel and its initial focus. MillWheel was Google’s original, general-purpose stream processing architecture founded by Paul Nordstrom around when Google opened its Seattle office in 2008.

:p What is MillWheel?
??x
MillWheel is a stream processing architecture developed at Google that originally aimed for low-latency data processing with weak consistency but later shifted to support strong consistency and robust out-of-order processing due to customer needs. It’s well known for providing exactly-once guarantees, persistent state, watermarks, and persistent timers.
x??

---

#### Exactly-Once Guarantees
Background context explaining the importance of exactly-once guarantees in stream processing pipelines. These guarantees ensure that each message is processed only once, which is crucial for correctness.

:p What are exactly-once guarantees?
??x
Exactly-once guarantees ensure that each message in a stream is processed precisely one time, preventing both duplication and omission. This is critical to maintaining the integrity of long-running pipelines executing on unreliable hardware.
x??

---

#### Persistent State
Background context explaining persistent state's role in maintaining correctness across pipeline executions. Persistent state helps maintain consistency even when hardware failures occur.

:p What is persistent state?
??x
Persistent state refers to the ability to store and recover data consistently, ensuring that the state of a stream processing pipeline remains intact despite hardware failures or restarts. This feature provides the foundation for maintaining long-term correctness.
x??

---

#### Watermarks
Background context explaining watermarks' role in reasoning about out-of-order input data. Watermarks help track progress and completeness of input streams.

:p What are watermarks?
??x
Watermarks are used to track the known progress or completeness of inputs being provided to a stream processing system, especially useful for handling out-of-order data. They allow the system to determine when it has seen enough data to make accurate decisions about the state of an input.
x??

---

#### Persistent Timers
Background context explaining persistent timers' role in linking watermarks with pipeline business logic. Persistent timers help manage time-based operations crucial for anomaly detection and other use cases.

:p What are persistent timers?
??x
Persistent timers enable the tracking of time across multiple processing cycles, which is essential for managing state that depends on elapsed time. They provide a link between watermarks and the pipeline’s business logic, ensuring accurate timing even when data arrives out-of-order.
x??

---

#### True Streaming Use Cases
Background context explaining the difference between true streaming use cases and materialized view semantics. True streaming use cases require continuous processing and immediate responses, while materialized views are suitable for periodic updates.

:p What are true streaming use cases?
??x
True streaming use cases involve scenarios where results need to be processed and consumed in real-time, such as anomaly detection or generating live analytics. These use cases require continuous, record-by-record processing and immediate response times rather than batch updates.
x??

---

#### Zeitgeist Pipeline Example
Background context explaining the specific needs of the Zeitgeist pipeline for anomaly detection. The pipeline required a way to identify anomalies without polling an output table.

:p What was the challenge faced by the Zeitgeist pipeline?
??x
The Zeitgeist pipeline faced the challenge of identifying anomalies in search query traffic, particularly for anomalous dips (decreases in query traffic). It needed a mechanism that could accurately detect these anomalies based on the completeness of input data without relying on processing-time delays.
x??

---

#### Watermarks and Input Completeness
Background context explaining how watermarks track input completeness. Watermarks help in dealing with out-of-order data by providing a metric for reasoning about the progress of inputs.

:p How do watermarks work?
??x
Watermarks work by tracking the known progress or completeness of inputs, allowing the system to determine when it has seen enough data to make accurate decisions. For simple sources, perfect watermarks can be computed; for complex sources, heuristics are used.
x??

---

#### MillWheel Paper and Contributions
Background context explaining the focus of the "MillWheel: Fault-Tolerant Stream Processing at Internet Scale" paper. The paper highlights challenges in providing correctness in a system like MillWheel.

:p What does the MillWheel paper focus on?
??x
The MillWheel paper focuses on the difficulties of providing correctness in systems like MillWheel, particularly emphasizing consistency guarantees and watermarks as key areas of focus.
x??

---

#### MillWheel and Streaming Flume Integration
MillWheel, a streaming backend for Flume, was integrated into Flume to form Streaming Flume. This integration occurred shortly after the publication of the MillWheel paper. The primary goal was to provide robust, low-latency processing of out-of-order data.
:p How did MillWheel improve the capabilities of Flume?
??x
MillWheel improved Flume's capabilities by providing mechanisms for exactly-once processing, persistent state management, watermarks, and persistent timers. These features enabled reliable handling of streaming data even on unreliable commodity hardware, moving closer to the true promise of stream processing.
x??

---

#### Windmill Successor to MillWheel
Windmill is being developed as a successor to MillWheel within Google. It aims to incorporate all the best ideas from MillWheel and add new features like better scheduling and dispatch mechanisms, as well as a cleaner separation of user and system code.
:p What are some key differences between Windmill and MillWheel?
??x
Windmill represents an entirely new implementation that builds upon the foundational concepts introduced by MillWheel but includes enhancements such as improved scheduling, better dispatching, and clearer separation between user and system code. It is designed to be more robust and efficient while continuing to support exactly-once processing and other essential features.
x??

---

#### Kafka as a Streaming Transport Layer
Kafka is not a data processing framework but serves as a persistent streaming transport layer, implemented as partitioned logs. It was originally developed at LinkedIn by Neha Narkhede and Jay Kreps.
:p What makes Kafka unique among the systems discussed in this chapter?
??x
Kafka's uniqueness lies in its role as a persistent streaming transport rather than a data processing framework. It offers features such as durability, replayability, elastic isolation between producers and consumers, and a conceptual link to stream and table theory, making it foundational for many stream processing installations.
x??

---

#### Durability and Replayability in Kafka
Kafka introduced the concept of durability and replayability from database systems into streaming data processing. This change provided robustness and reliability that was lacking in earlier ephemeral queuing systems like RabbitMQ or plain-old TCP sockets.
:p How did Kafka change the landscape of stream processing?
??x
Kafka changed the landscape by introducing durable logs, enabling input sources to be replayed for tasks such as backfills, prototyping, development, and regression testing. This feature helped in achieving end-to-end exactly-once guarantees in many stream processing engines.
x??

---

#### Streams and Tables Theory Popularized by Kafka
Streams and tables theory, first discussed extensively in Chapter 6, is crucial for understanding data processing concepts. Kafka played a significant role in popularizing this concept, linking it to databases and enabling developers to think about data in terms of streams and tables.
:p What was the significance of Kafka's contribution to stream and table theory?
??x
Kafka significantly contributed by making the streams and tables theory more accessible and relevant for streaming systems. This theory is foundational for both users and developers, providing a conceptual link between batch processing and streaming data, thereby enhancing the reliability and robustness of stream processing systems.
x??

---

#### "I ❤ Logs" by Jay Kreps
"I ❤ Logs" is a resource that delves into the foundations of Kafka's design. It is recommended reading for those interested in learning more about the concepts underlying Kafka and other stream processing systems.
:p What is the purpose of the book "I ❤ Logs"?
??x
The purpose of the book "I ❤ Logs" by Jay Kreps is to provide a deep dive into the foundational concepts behind Kafka, focusing on logs and their role in building reliable streaming systems. It serves as an excellent resource for understanding the design principles and practical applications of these concepts.
x??

---

#### Windmill vs. MillWheel
Windmill represents a ground-up rewrite of MillWheel within Google. It incorporates all the best ideas from its predecessor while adding features such as better scheduling, dispatch mechanisms, and improved separation of user and system code.
:p What are the key differences between Windmill and MillWheel?
??x
Key differences include that Windmill is an entirely new implementation built upon MillWheel's foundation but includes enhancements like better scheduling, dispatching, and a clearer separation between user and system code. It aims to improve robustness and efficiency while continuing to support essential features.
x??

---

