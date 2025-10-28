# Flashcards: 2B005---Streaming-Systems_processed (Part 29)

**Starting Chapter:** Summary

---

#### MapReduce
MapReduce introduced a simple set of abstractions for data processing on top of a robust and scalable execution engine, allowing data engineers to focus on business logic rather than handling distributed system complexities. This system was crucial in scaling up data processing tasks effectively.

:p What are the key features that made MapReduce significant?
??x
Key features included providing a simple abstraction layer (map and reduce functions) for distributed computing, enabling easy scalability through a fault-tolerant framework designed to work on commodity hardware. This allowed developers to handle large-scale datasets without worrying about low-level system details.
x??

---

#### Hadoop
Hadoop extended the MapReduce model by creating an open-source platform that expanded its capabilities beyond just processing data. It fostered a vibrant ecosystem where various tools and frameworks could be built upon it, leading to innovation and growth.

:p What was one of the main contributions of Hadoop?
??x
One of Hadoop's primary contributions was establishing a thriving open-source ecosystem around MapReduce. This openness allowed for the development of additional tools and frameworks that extended beyond simple data processing tasks, fostering innovation in big data technologies.
x??

---

#### Flume
Flume improved upon pipeline management by introducing logical pipelines and an intelligent optimizer. It made it easier to write clean, maintainable pipelines without sacrificing performance through manual optimizations.

:p How did Flume enhance the processing of data?
??x
Flume enhanced data processing by coupling high-level logical pipeline operations with a smart optimizer. This allowed for cleaner and more maintainable code while still achieving optimal performance, surpassing the limitations of map-reduce-only frameworks.
x??

---

#### Storm
Storm was designed to provide low-latency stream processing with weak consistency guarantees. It popularized the Lambda architecture by allowing real-time data processing alongside batch systems.

:p What unique feature did Storm introduce?
??x
Storm introduced a focus on providing low-latency stream processing while sacrificing some correctness for performance. This approach, known as the "Lambda Architecture," allowed for both real-time and eventually consistent results.
x??

---

#### Spark
Spark improved upon MapReduce by offering strongly consistent batch processing that could be used for continuous data processing. It demonstrated it was possible to have both low-latency and correct results.

:p How did Spark address latency and correctness in stream processing?
??x
Spark addressed these challenges by using a model where repeated runs of a strongly consistent batch engine provided low-latency, correct results, particularly suitable for in-order datasets.
x??

---

#### MillWheel
MillWheel focused on out-of-order data processing with strong consistency. It introduced concepts like watermarks and timers to handle time-based operations effectively.

:p What problem did MillWheel solve?
??x
MillWheel solved the challenge of robustly processing out-of-order data by integrating strong consistency and exactly-once semantics, utilizing tools like watermarks and timers.
x??

---

#### Kafka
Kafka revolutionized streaming systems by applying a durable log concept to stream transports. It brought back replayability and helped popularize the ideas of stream and table theory.

:p What was a key innovation introduced by Kafka?
??x
A key innovation was Kafka's application of a durable log (persistent data storage) to streaming transports, providing robustness and replayability features that were lacking in ephemeral systems like RabbitMQ and TCP sockets.
x??

---

#### Cloud Dataflow
Cloud Dataflow provided a unified model for batch plus streaming processing by combining concepts from MillWheel with the pipeline optimization capabilities of Flume. It balanced correctness, latency, and cost effectively.

:p How did Cloud Dataflow integrate different technologies?
??x
Cloud Dataflow integrated out-of-order stream processing (from MillWheel) with logically optimized pipelines (from Flume), offering a unified model for both batch and streaming data processing that could balance various trade-offs.
x??

---

#### Flink
Flink was an open-source innovator in stream processing, combining out-of-order capabilities with features like distributed snapshots and savepoints. It raised the bar for open-source stream processing.

:p What made Flink significant?
??x
Flink's significance lay in its rapid adoption of out-of-order processing and innovative features such as distributed snapshots and savepoints, significantly advancing open-source stream processing technology.
x??

---

#### Beam
Beam provided a portable abstraction layer that incorporated ideas from various leading systems. It aimed to make stream processing more accessible by aligning with SQL-like declarative logic.

:p What was the primary goal of Beam?
??x
The primary goal of Beam was to create a portable abstraction layer that combined the best ideas from across the industry, making stream processing easier and more flexible through a SQL-like programming model.
x??

---

