# Flashcards: 2B005---Streaming-Systems_processed (Part 16)

**Starting Chapter:** Reconciling with Batch Processing. What Transformations

---

#### Batch Processing and Stream/Tables Theory
Background context explaining how batch processing fits into stream/table theory. Discuss the basic pattern of tables becoming streams, and then being processed until a grouping operation is hit, which turns them back into tables.

:p How does batch processing fit into stream/table theory?
??x
Batch processing can be seen as a special case of stream processing where the input data is read in its entirety (forming a table), transformed through a series of nongrouping operations, and then grouped to produce final results. The key difference lies in how the data flows: in batch processing, it's a one-time transformation of static data.

```java
// Example of reading a file into a PCollection (table) for batch processing
PCollection<String> raw = IO.readFromFile("input.txt");
```
x??

---

#### Bounded vs. Unbounded Data
Explanation on how streams relate to bounded and unbounded data, emphasizing that in the context of stream/table theory, streams are simply the in-motion form of data.

:p How do streams relate to bounded/unbounded data?
??x
Streams can represent both bounded and unbounded data. In batch processing, which is a subset of stream processing, the input is typically finite (bounded). However, from the perspective of stream/table theory, it's easier to see that both types of data can be processed as streams.

```java
// Example of reading a file for bounded data
PCollection<String> raw = IO.readFromFile("input.txt");
// Example of consuming events in real-time for unbounded data
PCollection<KV<Team, Integer>> input = KafkaConsumer.getEvents();
```
x??

---

#### Transformations: What and How
Explanation on the types of transformations (nongrouping and grouping) within stream/table theory. Discuss how these operations relate to building models, counting sums, filtering spam, etc.

:p What are the two main types of what transforms in stream/table theory?
??x
In stream/table theory, there are two main types of what transformations: nongrouping and grouping. Nongrouping transformations accept a stream of records and produce a new transformed stream (e.g., filters, exploders). Grouping transformations group the streams into tables by some key or rule (e.g., joins, aggregations).

```java
// Example of a nongrouping transformation: filtering spam messages
PCollection<String> nonSpam = raw.apply(new FilterFn());

// Example of a grouping transformation: summing team scores per team
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());
```
x??

---

#### Streams and Tables in Classic Batch Processing
Explanation on the classic batch processing pipeline, including event-time/processing-time visualization.

:p How does a simple summation pipeline look in a streams and tables view?
??x
A simple summation pipeline reads data, parses it into individual team member scores, and then sums those scores per team. In a streams and tables view, this process can be represented as:

```java
PCollection<String> raw = IO.readFromFile("input.txt");
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());
```

This pipeline sums team scores, but the streams and tables view emphasizes that grouping operations create tables where final results can be stored.

```java
// Example of a summation pipeline
PCollection<String> raw = IO.readFromFile("input.txt");
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());
```
x??

---

#### Grouping and Ungrouping Operations
Explanation on the nature of grouping operations and their inverse "ungrouping" in stream processing.

:p What is the "ungrouping" inverse of a grouping operation?
??x
Grouping operations in stream/table theory group records together, transforming streams into tables. The ungrouping inverse would put these grouped records back into motion as separate elements within a stream. However, this concept is more theoretical and less commonly used directly.

```java
// Example of a grouping operation (creating a table)
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());

// Theoretical example of ungrouping: This is not typically done in practice.
PCollection<Integer> ungrouped = totals.apply(UngroupFn());
```
x??

---

#### Stream Processors as Databases
Explanation on the idea that anywhere you have a grouping operation, it creates a table with potentially useful data.

:p Why can we read results directly from grouped operations?
??x
Grouping operations in stream processing create tables where final results are stored. If these final results don't need further transformation downstream, they can be read directly from the resulting table. This approach saves resources and storage space by eliminating redundant data storage and additional sink stages.

```java
// Example of reading results directly from a grouped operation
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());
KV<Team, Integer> result = totals.peek();
```

This is particularly useful in scenarios where the values are your final results and don't require further processing.

```java
// Example of serving data directly from state tables (hypothetical)
PCollection<KV<Team, Integer>> results = StateTable.readFrom("team_scores");
```
x??

---

#### Window Assignment
Background context explaining the concept. Window assignment involves placing a record into one or more windows, which effectively combines the window definition with the user-assigned key for that record to create an implicit composite key used at grouping time. This process is crucial for stream-to-table conversion because it drives how data are grouped and aggregated.

If applicable, add code examples with explanations:
```java
// Example of applying a windowing transform
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .apply(Sum.integersPerKey());
```
:p What is the primary purpose of window assignment in stream processing?
??x
The primary purpose of window assignment is to place a record into one or more windows, thereby combining the window definition with the user-assigned key for that record. This process results in an implicit composite key used at grouping time, which drives how data are grouped and aggregated.
x??

---

#### Window Merging
Background context explaining the concept. The effect of window merging is more complex than simple window assignment but still straightforward when considering logical operations. In a stream processing system, when grouping records into windows that can merge, the system must account for all possible merges involving the same key.

:p What is the impact of window merging on data grouping?
??x
Window merging impacts data grouping by requiring the system to consider all windows sharing the same key and potentially merging with new incoming data. This means that when a new element arrives, the system needs to determine which existing windows can merge with it and handle these merges atomically.

For example, in a batch engine, window merging might result in multiple mutations to the table:
1. Delete unmerged windows.
2. Insert merged windows.

This process ensures strong consistency for correctness guarantees.
x??

---

#### Streams and Tables View
Background context explaining the concept. The streams and tables view shows how data processing operates on both streaming and batch engines, where grouping by key and window is a core operation. In this view, each group of records in a stream forms a window, which is then processed to create a table.

:p How does the streams and tables view differ from an event-time/processing-time view?
??x
The streams and tables view differs from the event-time/processing-time view by focusing on how data are grouped into windows and transformed into tables. In the streams and tables view, each group of records in a stream forms a window, which is then processed to create a table. This view highlights the grouping operations that occur during stream processing and their impact on table creation.

For instance, if we have two windows (A and B) for a key, the system will first group all data by the key and then merge any overlapping or contiguous windows A and B into a single window.
x??

---

#### Hierarchical Key in Window Merging
Background context explaining the concept. In window merging, the system treats the user-assigned key and the window as part of a hierarchical composite key to handle complex grouping operations.

:p How does the system treat keys and windows during window merging?
??x
During window merging, the system treats the user-assigned key and the window as part of a hierarchical composite key. This allows the system to first group data by the root of the hierarchy (the user-assigned key) and then proceed with grouping by window within that key.

For example:
1. Grouping by key: User-assigned key
2. Merging windows: Window as a child component of the user-assigned key

This hierarchical treatment enables the system to handle complex merging operations efficiently.
x??

---

#### Atomicity/Parallelization in Window Merging
Background context explaining the concept. The atomicity and parallelization units are defined based on keys rather than key+window, ensuring strong consistency for correctness guarantees.

:p Why do systems that support window merging typically define the unit of atomicity as key?
??x
Systems that support window merging typically define the unit of atomicity as the key rather than key+window to ensure strong consistency. This is because the merging operation must inspect all existing windows for a given key, determine which can merge with new incoming data, and then commit these merges atomically.

By treating keys as the atomic units, systems can manage the complexity of window merging more efficiently while maintaining correctness guarantees.
x??

---

#### Window Merging Semantics
Background context explaining the concept. Detailed semantics of how window merging affects table mutations and changelogs over time.

:p How does window merging affect the changelog that dictates a table's contents?
??x
Window merging affects the changelog by modifying it to reflect the merged state of windows. For non-merging windows, each new element being grouped results in a single mutation (adding the element to its key+window group). However, with merging windows, grouping a new element can result in one or more existing windows merging with the new window.

The system must inspect all existing windows for a given key, determine which can merge with the new window, and then atomically commit deletes for unmerged windows while inserting the merged window into the table. This ensures that the changelog accurately reflects the current state of the data.
x??

---

#### Triggers in Data Processing

Triggers are mechanisms used to dictate when the contents of a window will be materialized, particularly useful with unbounded data. Watermarks provide signals for input completeness, which can trigger the materialization process.

:p What is the role of triggers in handling unbounded data streams?
??x
Triggers play a crucial role in determining when to send grouped (windowed) data downstream as a stream. This is essential for processing unbounded data because it helps manage and convert the state from tables back into streams at appropriate moments, ensuring that all relevant data has been received before aggregation or further processing.

For example, using a trigger like `AfterCount(1)` in Apache Beam means emitting results every time a new record arrives. This continuous materialization can be useful but may not always align with business needs for more controlled batch-like operations.

```java
PCollection<KV<Team, Integer>> totals = input.apply(
    Window.into(FixedWindows.of(TWO_MINUTES))
          .triggering(Repeatedly(AfterCount(1)))
);
```
x??

---

#### Per-Record Trigger Example

Per-record triggers emit results for every new record that arrives in the stream. This can be useful when you want to immediately process each incoming piece of data.

:p How does a per-record trigger work in practice?
??x
A per-record trigger processes and emits results as soon as it receives a new record, effectively making the aggregation or processing result available on-the-fly. This is particularly useful for real-time applications where immediate feedback is required.

Example: In Apache Beam, using `Repeatedly(AfterCount(1))` as the triggering condition means that a new window will be triggered every time a single record arrives in the stream.

```java
PCollection<KV<Team, Integer>> totals = input.apply(
    Window.into(FixedWindows.of(TWO_MINUTES))
          .triggering(Repeatedly(AfterCount(1)))
);
```
x??

---

#### Watermark-based Trigger Example

Watermark-based triggers emit results when the watermark passes a certain threshold, indicating that no more data will arrive for the current window.

:p How do watermark-based triggers work?
??x
Watermark-based triggers are used to ensure that all late-arriving data have been processed before emitting results. This is particularly useful in batch-like operations where you want to wait until an approximate end of input (EoI) signal is received, such as a watermark passing the current window's end.

Example: In Apache Beam, using `AfterWatermark.pastEndOfWindow()` means that results will be emitted when the watermark passes the end of the window, ensuring that no late data can affect the result.

```java
PCollection<KV<Team, Integer>> totals = input.apply(
    Window.into(FixedWindows.of(TWO_MINUTES))
          .triggering(AfterWatermark.pastEndOfWindow())
);
```
x??

---

#### Grouping and Triggers

Grouping in data processing is about aggregating streams into tables (or stateful windows), while triggers determine when these aggregated results should be sent back to the stream as a new, ungrouped result.

:p How do grouping and triggers complement each other in data processing?
??x
Grouping and triggers complement each other by first transforming a stream into a table through windowing operations, then using triggers to decide when to emit results from these tables as streams. Grouping helps manage state and perform aggregations over windows of time or conditions, while triggers ensure that the processed data is sent back to the stream at appropriate times.

Example: In Apache Beam, grouping data into windows and applying a trigger like `AfterWatermark.pastEndOfWindow()` ensures that results are only emitted when the watermark indicates no more late data will arrive.

```java
PCollection<KV<Team, Integer>> totals = input.apply(
    Window.into(FixedWindows.of(TWO_MINUTES))
          .triggering(AfterWatermark.pastEndOfWindow())
);
```
x??

---

#### Watermark Completeness Trigger
Background context: In Apache Beam, a watermark is used to determine when enough data has arrived for processing. A watermark completeness trigger ensures that windows are processed once the watermark passes their end boundaries.

:p What is a watermark completeness trigger and how does it work in Apache Beam?
??x
A watermark completeness trigger is designed to materialize windows when the watermark reaches the end of each window, enabling progressive emission of results from an unbounded input stream. This allows for real-time processing while ensuring that all relevant data has been processed.

For example, consider a scenario where you are analyzing user activity in a two-minute sliding window:

```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
                 .triggering(AfterWatermark()))
    .apply(Sum.integersPerKey());
```

Here, `AfterWatermark()` triggers the windows to be processed as soon as the watermark passes their end.

x??

---

#### Early/Fire/On-Time/Late Trigger
Background context: In Apache Beam, more complex triggering mechanisms can be defined using early firings and late firings. This allows for handling early data (data that arrives before the watermark), on-time data, and late data (data that arrives after the watermark).

:p How does an early/on-time/late trigger work in Apache Beam?
??x
An early/on-time/late trigger in Apache Beam is a more comprehensive mechanism that allows you to specify conditions under which windows should be processed. It includes mechanisms for handling early data, on-time data, and late data:

```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
                 .triggering(
                     AfterWatermark()
                         .withEarlyFirings(AlignedDelay(ONE_MINUTE))
                         .withLateFirings(AfterCount(1))))
    .apply(Sum.integersPerKey());
```

- `AfterWatermark()` ensures that windows are processed when the watermark passes.
- `WithEarlyFirings(AlignedDelay(ONE_MINUTE))` allows early data to be processed with a delay of one minute.
- `WithLateFirings(AfterCount(1))` processes late data once it arrives.

This setup enables a more nuanced handling of different types of data, ensuring that the system can emit results in a timely manner while still processing all relevant data.

x??

---

#### Batch Processing Triggers
Background context: In traditional batch processing, triggers are used to signal when the input is complete. This ensures that the entire pipeline waits for the final state before emitting the result.

:p How do batch processing triggers work?
??x
In Apache Beam, classic batch processing scenarios use a trigger that fires when all data in the input source is considered complete. This means that once the initial MapRead stage of a MapReduce job has all its data available, it can start processing. For example:

```java
PCollection<String> raw = IO.read(...);
// Map and Reduce stages would follow, but not shown here.
```

For table-to-stream conversions in the middle of the pipeline (like the ReduceRead stage), a similar trigger mechanism is used, but it waits for all data to be written to the shuffle before proceeding.

This ensures that batch pipelines wait until they have processed all input data before emitting their final results.

x??

---

#### Ungrouping Effect
Background context: Triggers in Apache Beam can cause an ungrouping effect on state tables. As windows are materialized, the state is separated into individual window streams, reflecting the processing of each window independently.

:p What is the ungrouping effect caused by triggers in Apache Beam?
??x
The ungrouping effect occurs when a trigger causes windows to be processed and their results to be emitted separately from other data. For instance, if you have a stream with windows defined over two minutes:

```java
PCollection<String> raw = IO.read(...);
// Apply windowing and triggers here.
```

With an `AfterWatermark()` trigger, as the watermark passes each window's end boundary, that window's results are separated from others. This is evident in the streams and tables view of the pipeline.

For example, consider the following setup:

```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
                 .triggering(AfterWatermark()))
    .apply(Sum.integersPerKey());
```

As the watermark passes each window's end boundary, the state table ungroups into separate streams for each window.

x??

---

---
#### Trigger Guarantees and Asynchronicity
In most batch processing systems, triggers are not finely granular due to their lock-step nature. Triggers like `AfterWatermark` or `AfterCount(N)` do not provide precise timing but rather a lower bound of when data might be processed.

:p What is the nature of trigger guarantees in batch and streaming systems?
??x
In most existing batch processing systems, triggers are less granular due to their lock-step read-process-group-write-repeat sequence. For instance, `AfterWatermark` does not guarantee that it will fire exactly at the end of a window but rather after some time has passed since the watermark was last updated. Similarly, `AfterCount(N)` only guarantees that at least N elements have been processed.

```java
// Example of an AfterWatermark trigger in Java
public class WatermarkTrigger {
    public boolean shouldProcess(long currentTimestamp) {
        // Check if the watermark has passed the end of the window
        return currentTimestamp >= watermark;
    }
}
```
x??

---
#### Trigger Names and Asynchronous Behavior
The naming conventions of triggers like `AfterWatermark` or `AfterCount(N)` are designed to accommodate both batch and streaming systems. These names reflect the natural asynchronicity and nondeterminism inherent in triggering mechanisms.

:p Why were specific trigger names chosen?
??x
Specific trigger names such as `AfterWatermark` and `AfterCount(N)` were chosen not just for compatibility with batch systems but also to reflect their behavior within a streaming context. For example, `AfterWatermark` does not guarantee firing exactly at the end of a window but rather after some time has passed since the watermark was last updated. Similarly, `AfterCount(N)` only guarantees that at least N elements have been processed.

```java
// Example of an AfterCount trigger in Java
public class CountTrigger {
    private int count = 0;
    public boolean shouldProcess(int element) {
        // Increment count and check if it reaches the threshold
        count++;
        return count >= threshold;
    }
}
```
x??

---
#### Blending Batch and Streaming Systems
The main difference between batch and streaming systems is their ability to handle incremental data processing. However, this difference is more of a latency/throughput trade-off rather than a fundamental semantic distinction.

:p What is the main difference between batch and streaming systems?
??x
The primary difference between batch and streaming systems lies in their handling of incremental data processing. Batch systems are designed for large, bounded datasets with lower latency but higher throughput, while streaming systems handle unbounded streams with potentially higher latencies but better real-time processing capabilities. This trade-off is highlighted by the ability to trigger tables incrementally in streaming systems, which is not as straightforward in batch systems.

```java
// Example of a simple stream processing pipeline in Java
public class StreamProcessor {
    public void processStream(Stream<String> data) {
        // Process each element and emit results
        data.map(element -> transformElement(element))
             .filter(result -> isRelevant(result))
             .forEach(result -> sendToStorage(result));
    }
}
```
x??

---
#### Efficiency Differences Between Batch and Streaming Systems
Batch and streaming systems differ primarily in efficiency (batch has higher throughput at the cost of latency) and natural handling of unbounded data streams.

:p What are the main differences between batch and streaming systems?
??x
The main difference between batch and streaming systems lies in their efficiency and how they handle data. Batch processing is optimized for high-throughput with potentially higher latencies, suitable for large, bounded datasets. Streaming systems, on the other hand, are designed to process unbounded streams of data more efficiently, often at the cost of some latency but with real-time capabilities.

```java
// Example comparison in Java
public class BatchVsStreaming {
    public void batchProcessing() {
        // Process entire dataset before returning results
    }

    public void streamProcessing() {
        // Continuously process incoming data and emit results immediately
    }
}
```
x??

---

