# Flashcards: 2B005---Streaming-Systems_processed (Part 4)

**Starting Chapter:** Batch Foundations What and Where. What Transformations

---

#### Batch Processing: What Results Are Calculated?

Background context explaining batch processing and transformations. In classic batch processing, the focus is on computing results based on a fixed dataset that has been collected over a period of time.

:p What are the "What" results calculated in batch processing?
??x
In batch processing, the primary goal is to compute aggregates or transformations over a static dataset. These calculations are performed offline and typically result in summarized data that can be used for reporting, analysis, or decision-making purposes. For instance, if you have a set of user scores reported by users' phones, you might want to calculate the total score for each team.

Example code snippet illustrating batch processing logic:
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());
```

x??

---

#### Batch Processing: Where in Event Time Are Results Calculated?

Background context on event time and processing time. In batch processing, the data is typically processed after it has been fully collected (infinite event-time window), so all events are observed before any results are produced.

:p Where do we calculate results for a batch process?
??x
In batch processing, results are calculated once all of the input events have been observed and processed. This means that the entire dataset is available at one point in time, allowing us to perform aggregate operations on it without worrying about ongoing data ingestion or real-time constraints.

Example code snippet showing batch processing:
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());
```

x??

---

#### Windowing in Batch Processing

Background context on windowing. Windowing is the process of slicing up a data stream into manageable chunks or windows to allow for time-based analysis and processing.

:p What does "Where" mean when discussing batch processing?
??x
When discussing "Where" in the context of batch processing, we are referring to where in event time the results are calculated. In batch processing, the entire dataset is available at a single point in time after it has been fully collected, so all events are considered for the window before any calculations are performed.

Example code snippet demonstrating how windows might be applied:
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey())
                                               .apply(Window.into(FixedWindows.of(Duration.standardMinutes(1))));
```

x??

---

#### Time-Lapse Diagrams

Background context on time-lapse diagrams. These are visual representations of how data evolves over both event and processing times in a pipeline, helping to illustrate the flow and transformations step by step.

:p How do we visualize the execution of a batch process?
??x
Time-lapse diagrams provide a visual representation of the execution of a batch process over time, showing inputs and outputs in both event time (on the x-axis) and processing time (on the y-axis). Inputs are represented as circles that darken as they are observed by the pipeline. State is shown as rectangles (gray for state, blue for output), with aggregate values at the top.

Example of a time-lapse diagram:
```
Figure 2-3. Classic batch processing
Inputs: Circles representing scores (e.g., "5", "9"), darken as they are observed.
State and output: Rectangles showing accumulated results over event time, with outputs materialized once all inputs are seen.
```

x??

---

#### Key/Value PCollection in Apache Beam

Background context on PCollections and PTransforms. In the Apache Beam model, data is processed using transformations applied to collections of key-value pairs.

:p What are PCollections and PTransforms used for?
??x
PCollections represent datasets across which parallel transformations can be performed, while PTransforms are operations that transform one or more input PCollections into a new output PCollection. In the context of batch processing with Apache Beam, these concepts allow for efficient data processing pipelines.

Example code snippet demonstrating usage:
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());
```

x??

---

#### Summation Pipeline Example

Background context on a simple pipeline for summing scores. This example uses a PCollection of key-value pairs to accumulate and output the total score per team.

:p How would you set up a basic batch processing pipeline?
??x
To set up a basic batch processing pipeline, you typically start by reading in raw data from an input source, parsing it into key-value pairs, and then applying transformations to compute aggregate results. For example:
```java
PCollection<String> raw = IO.read(...); // Read in the raw data
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn()); // Parse scores into key-value pairs
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey()); // Sum up scores per team
```

x??

---

#### Processing Time vs. Event Time

Background context on the difference between processing time and event time. Processing time refers to when data is observed by the pipeline, while event time represents when events occurred in reality.

:p What are event time and processing time?
??x
Event time refers to the actual occurrence of an event (e.g., a user score), whereas processing time refers to when the pipeline observes that event. In batch processing, these times align as all data is collected before any computations are performed.

Example code snippet illustrating the difference:
```java
PCollection<String> raw = IO.read(...);
// Raw input is read at some point in time.
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
// Parsing and transformation happen after the data has been read (processing time).
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());
// Final aggregation is performed once all inputs are observed.
```

x??

---

#### Handling Unbounded Data in Batch Processing

Background context on handling unbounded data sources. Classic batch processing may not be sufficient for unbounded datasets as it requires waiting until the end of all data.

:p How do we handle unbounded data in batch processing?
??x
Classic batch processing is not suitable for unbounded data sources because you cannot wait indefinitely for new data to arrive. To handle such cases, techniques like windowing can be applied to process data within defined windows or intervals, even if the data stream continues.

Example code snippet using windowing:
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey())
                                               .apply(Window.into(FixedWindows.of(Duration.standardMinutes(1))));
// Process data in 1-minute windows.
```

x??

#### Windowing Strategies Overview
Background context: The text introduces different windowing strategies used in data processing, such as fixed windows, sliding windows, and session windows. These techniques are essential for breaking down streaming data into manageable chunks to process it effectively.

:p What are some common windowing strategies mentioned?
??x
Common windowing strategies include fixed windows, sliding windows, and session windows. Fixed windows apply a uniform duration across all data points, while sliding windows allow overlapping time intervals, and session windows group events based on idle periods between them.
x??

---

#### Example of Fixed Windows in Beam
Background context: The text demonstrates how to implement a windowing strategy using Apache Beam by applying a `Window.into` transform for fixed windows. This example specifically highlights the use of fixed two-minute windows.

:p How can you apply fixed windows with a two-minute duration in Apache Beam?
??x
You can apply fixed windows with a two-minute duration using the `FixedWindows.of(TWO_MINUTES)` method within the `Window.into` transform.
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .apply(Sum.integersPerKey());
```
This code snippet illustrates how to window the input data into fixed two-minute intervals and then sum the integers per key within each window.
x??

---

#### Batch Execution of Windowed Pipeline
Background context: The text mentions that while batch processing can be thought of as a subset of streaming, it uses batch engines for simplicity in understanding the mechanics before moving to streaming. The example provided demonstrates executing the windowed pipeline on a batch engine.

:p How does Beam handle both batch and streaming pipelines?
??x
Apache Beam provides a unified model where semantically, batch processing is considered a special case of streaming with no out-of-order or late data. This means that the same code can be used for both batch and streaming scenarios.
x??

---

#### Visualizing Windowing Strategies
Background context: The text refers to Figure 2-4 which visually demonstrates aligned and unaligned windows across different keys, helping to understand how these strategies differ.

:p What are aligned and unaligned windows in the context of windowing strategies?
??x
Aligned windows apply the same windowing strategy across all data points, ensuring consistency. Unaligned windows can vary based on specific conditions, such as idle periods for session windows or overlapping intervals for sliding windows.
x??

---

#### Streaming vs Batch Execution
Background context: The text indicates that while the example provided is executed on a batch engine for clarity, switching to a streaming engine would be the next step.

:p What does Beam's unified model imply for batch and streaming processing?
??x
Beam’s unified model implies that batch processing can be seen as a special case of streaming where there are no out-of-order events or late data. The same pipeline code can be used for both scenarios, making it easier to transition between them.
x??

---

#### Windowed Summation on a Batch Engine
Windowing is a technique where data is grouped into windows based on time or value. In this context, windowed summation involves accumulating inputs over a fixed time interval (e.g., two minutes) before producing output.
:p What does windowing involve in the context of batch engines?
??x
Windowing involves grouping input data into fixed-sized windows, typically based on event-time intervals like two minutes. The system accumulates data within these windows until they are consumed entirely and then produces a single output for each window.
x??

---
#### Triggers in Streaming Systems
Triggers determine when to materialize results during the processing of unbounded streams. They help manage the timing of result generation, balancing latency with correctness.
:p What is the primary function of triggers in streaming systems?
??x
The primary function of triggers is to define when output for a window should be generated during the processing of an unbounded stream. Triggers can generate updates periodically or wait until they believe input for a window is complete before producing results.
x??

---
#### Repeated Update Triggers
Repeated update triggers provide periodic updates to windows as their contents evolve. These are common in streaming systems, offering simple and useful semantics for continuous aggregation tasks.
:p What are repeated update triggers used for?
??x
Repeated update triggers are used to provide regular, incremental updates to the results of windowed aggregations. They generate panes of output periodically based on new input data, allowing for near-real-time processing with eventual consistency.
x??

---
#### Completeness Triggers
Completeness triggers wait until they believe that the input for a window is complete before producing a pane of output. This approach mirrors batch processing semantics but applied to individual windows in a streaming context.
:p What does a completeness trigger do?
??x
A completeness trigger waits until it determines that the input for a specific window is believed to be complete, then generates an output pane based on this data. This ensures results are only produced when all expected events have arrived within the window, similar to batch processing but scoped per-window.
x??

---
#### Per-Record Triggering
Per-record triggering fires after each new record is processed, generating a separate pane for each input record in a window. It works well with systems where updates can be frequently polled.
:p What does per-record triggering do?
??x
Per-record triggering generates a pane of output for every new record as it arrives within the window. This approach is useful when results need to be available immediately, but may result in frequent, small updates that are "chatty" and resource-intensive.
x??

---
#### Aligned Delays in Triggers
Aligned delays schedule updates based on fixed processing-time intervals that align across all keys and windows. This approach provides predictable updates at regular intervals.
:p What is an aligned delay trigger?
??x
An aligned delay trigger schedules updates at fixed, periodic intervals, such as every two minutes. This ensures consistent, predictable updates across all windows, similar to microbatch systems like Spark Streaming.
Example code:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .triggering(Repeatedly(AlignedDelay(TWO_MINUTES)))
    .apply(Sum.integersPerKey());
```
x??

---
#### Unaligned Delays in Triggers
Unaligned delays schedule updates based on the observed data within each window, potentially leading to more even load distribution over time. This approach reduces burstiness.
:p What is an unaligned delay trigger?
??x
An unaligned delay trigger schedules updates relative to the data observed within a given window, which can lead to a more evenly distributed workload over time. Unlike aligned delays, these updates are not fixed and may vary per window.
Example code:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .triggering(Repeatedly(UnalignedDelay(TWO_MINUTES)))
    .apply(Sum.integersPerKey());
```
x??

---
#### Watermarks in Streaming Systems
Watermarks are used to track the progress of event-time data and provide a mechanism for handling late data. They help maintain correctness by ensuring that only complete windows produce outputs.
:p What role do watermarks play in streaming systems?
??x
Watermarks track the latest known timestamp of unprocessed events, helping to manage the completeness of data within windows. They are crucial for dealing with late data and maintaining correct results by delaying output until all expected events have arrived.
x??

---

