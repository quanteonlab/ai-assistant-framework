# Flashcards: 2B005---Streaming-Systems_processed (Part 17)

**Starting Chapter:** A Holistic View of Streams and Tables in the Beam Model

---

#### Latency vs. Throughput Trade-off
Background context: The efficiency delta between systems often comes from balancing latency with throughput. Larger bundle sizes can enhance throughput at the cost of increased latency, and more efficient shuffle implementations can improve performance without significant trade-offs.

:p How does larger bundle size affect system efficiency?
??x
Larger bundle sizes increase overall throughput by allowing more data to be processed in parallel, but they also introduce higher latency because processing each batch takes longer. This is a common trade-off between maximizing the amount of work done at once and minimizing the time taken for individual operations.

```java
// Example of setting a large bundle size
PipelineOptions options = PipelineOptionsFactory.create();
options.setNumShards(100); // Large number of shards can increase throughput but may introduce higher latency.
```
x??

---

#### Apache Beam's Unification Approach
Background context: Apache Beam aims to unify batch and streaming processing by providing a general data processing model that can handle both efficiently. It achieves this through transparently tuning implementation details such as bundle sizes, shuffle implementations, etc., under the covers.

:p How does Apache Beam aim to unify batch and streaming?
??x
Apache Beam unifies batch and streaming by offering a unified programming model that can seamlessly process both types of data without the need for separate batch and stream processing engines. This is achieved through adaptive tuning of internal parameters like bundle sizes, shuffle implementations, etc., which automatically adjust based on the specific use case to balance latency, throughput, and cost.

```java
// Example of configuring Beam pipeline options
PipelineOptions options = PipelineOptionsFactory.create();
options.setStreaming(true); // Configuring for streaming processing.
```
x??

---

#### Accumulation Modes in Apache Beam
Background context: Accumulation modes define how results are refined over time as new data arrives. In Beam, there are three accumulation modes: discarding, accumulating, and accumulating and retracting.

:p What are the three accumulation modes in Apache Beam?
??x
The three accumulation modes in Apache Beam are:
1. **Discarding Mode**: The system either throws away the previous value or keeps a copy to compute the delta.
2. **Accumulating Mode**: No additional work is needed; the current value for the window is emitted directly.
3. **Accumulating and Retracting Mode**: Keeps copies of all previously triggered (but not yet retracted) values, necessary for correctly reverting effects.

```java
// Example configuration in Beam code
Window<PCollection<String>> windowed = pcoll.apply(Window.into(FixedWindows.of(Duration.standardMinutes(1))));
```
x??

---

#### Streams and Tables in Beam Model
Background context: The Beam model integrates concepts from streams and tables to handle both batch and streaming data. Understanding how these integrate helps in designing pipelines that can efficiently process any type of data.

:p How does the Beam model represent streams and tables?
??x
In the Beam model, streams and tables are unified into a single processing framework where:
- Streams represent continuous data flows.
- Tables provide stateful transformations with windowing and accumulation mechanisms.
This integration allows for handling both batch and streaming data by leveraging the best of both worlds: bounded data processed like traditional batches and unbounded data handled with stream-like semantics.

```java
// Example of a Beam pipeline using both streams and tables
Pipeline p = Pipeline.create(options);
PCollection<String> pcoll = p.apply("ReadFromSource", TextIO.read().from("path/to/data"));
pcoll.apply(Window.into(FixedWindows.of(Duration.standardMinutes(1))));
```
x??

---

---
#### Logical versus Physical Operations
Logical operations are user-defined, while physical operations are those executed by the underlying engine. The optimizer converts logical operations into a sequence of primitive, supported operations.

:p How does the optimization process convert logical operations to physical ones?
??x
The optimization process translates high-level, abstract operations (like Parse) into concrete, low-level operations that can be directly executed by the engine. For instance, if you use `ParseFn`, it might need to be broken down into multiple primitive steps such as splitting strings and converting them to integers.

```java
// Pseudocode for a simplified Parse operation
public class ParseFn extends DoFn<String, KV<Team, Integer>> {
    public void processElement(BatchModeProcessContext<String> c) {
        String raw = c.element();
        // Split the string into team name and score
        String[] parts = raw.split(",");
        Team team = parseTeam(parts[0]);
        int score = parseInt(parts[1]);
        c.output(KV.of(team, score));
    }
}
```
x??

---
#### Physical Stages and Fusion
Physical stages are groups of operations that can be executed together to reduce overheads like serialization and network communication.

:p What is the benefit of fusing physical operations into a single stage?
??x
Fusing physical operations reduces unnecessary overheads such as serialization, network communication, and deserialization. By combining multiple logical phases into one physical stage, the pipeline can operate more efficiently, leading to better performance.

For example, if you have two consecutive operations that process raw data, they might be fused into a single stage to reduce intermediate processing steps:

```java
// Pseudocode for fusion of ReadFromSource and Parse in a single stage
public void optimizePlan(PCollection<String> raw) {
    // Fused into one physical stage
    return raw.apply(new CoProcessor<>(ReadFromSource.class, ParseFn.class));
}
```
x??

---
#### Keys, Values, Windows, and Partitioning
These determine how data is grouped for processing. The optimizer annotates each PCollection to reflect these characteristics.

:p What do keys, values, windows, and partitioning represent in the context of a pipeline?
??x
Keys, values, windows, and partitioning are critical components that define how data is processed within a pipeline:

- **Key**: Defines groups of records (e.g., `KV<Team, Integer>`).
- **Value**: The actual data within each group.
- **Window**: Time-based grouping to process elements in chunks over time.
- **Partitioning**: How the data is distributed across parallel processing units.

For example:
```java
// Applying windowing and partitioning
PCollection<KV<Team, Integer>> input = raw.apply(
    Window.into(FixedWindows.of(TWO_MINUTES))
           .triggering(AfterWatermark()
                       .withEarlyFirings(AlignedDelay(ONE_MINUTE))
                       .withLateFirings(AfterCount(1)))
).apply(Sum.integersPerKey());
```
This code groups elements by `Team` and processes them in fixed 2-minute windows, triggering sums every minute for early firings.

x??

---
#### Logical Phases of a Team Score Summation Pipeline
The pipeline breaks down into logical phases with specific intermediate PCollection types that represent data transformations and operations.

:p What are the main differences between Figures 6-13 and 6-14 in terms of logical versus physical operations?
??x
In Figure 6-13, we see a high-level, logical view of the pipeline. In contrast, Figure 6-14 shows how this is translated into physical stages with primitive operations:

- **Logical Operations**: High-level abstractions (e.g., `Parse`, `Sum`).
- **Physical Stages**: Primitive operations executed by the engine.

For example:
```java
// Logical and Physical Phases
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .apply(Sum.integersPerKey());
```

x??

---

---
#### Parse Operation
The `Parse` operation extracts a key (team ID) and value (user score) from raw strings. It is a nongrouping operation, meaning that the stream it consumes remains a stream on the other side.

:p What does the `Parse` operation do in the context of processing data streams?
??x
The `Parse` operation takes raw input strings and extracts two pieces of information: a key (team ID) and a value (user score). This process is straightforward and ensures that the output remains as a stream, not altering its structure.

```java
public class ParseExample {
    public static void main(String[] args) {
        String input = "Team123 85";
        String[] parts = input.split(" ");
        String teamId = parts[0];
        int score = Integer.parseInt(parts[1]);
        System.out.println("Team ID: " + teamId + ", Score: " + score);
    }
}
```
x??

---
#### Window+Trigger Operation
The `Window+Trigger` operation involves two main steps:
1. **Window Assignment**: Each element is assigned to a set of windows, done through the `AssignWindows` operation.
2. **Triggering**: This happens after grouping and converts the table created by grouping back into a stream.

:p What are the two primary operations that make up the `Window+Trigger` process?
??x
The `Window+Trigger` process consists of:
1. **Window Assignment**: This step assigns each element to specific windows using the `AssignWindows` operation, which is nongrouping and simply annotates elements with their respective window(s), maintaining the stream structure.
2. **Triggering**: After grouping, this step converts the resulting table back into a stream, typically through its own dedicated operation following `GroupMergeAndCombine`.

```java
public class WindowExample {
    public void assignWindows() {
        // Logic to assign windows to elements
    }

    public void triggerEvents() {
        // Logic to convert grouped tables back into streams
    }
}
```
x??

---
#### Sum Operation
Summation in Beam is a composite operation that involves partitioning and aggregation. Key steps are:
1. **Partitioning**: Redirects elements with the same keys to the same physical machine, often referred to as shuffling.
2. **Grouping**: Merges windows and groups by key and window.
3. **Combining**: Aggregates individual elements incrementally.

:p What does the `Sum` operation involve in detail?
??x
The `Sum` operation is a composite process involving:
1. **Partitioning**: This nongrouping operation redirects elements to physical machines based on keys, ensuring that elements with the same key end up together.
2. **Grouping**: Involves window merging and grouping by key and window.
3. **Combining**: Aggregates individual elements incrementally using a `CombineFn` in Beam.

```java
public class SumExample {
    public void partitionElements() {
        // Logic to redirect elements based on keys
    }

    public void groupAndMergeWindows() {
        // Logic to merge windows and group by key and window
    }

    public void combineElements() {
        // Incremental aggregation logic using CombineFn
    }
}
```
x??

---
#### WriteToSink Operation
The `WriteToSink` operation writes the stream produced by triggering (a table or a stream) into an output data sink. It involves grouping if writing to a table, and no additional grouping if writing to a stream.

:p What is the role of the `WriteToSink` operation in the process?
??x
The `WriteToSink` operation writes the stream produced by triggering (either a table or a stream) into an output data sink. It requires:
- **Grouping**: If writing to a table, grouping may be necessary.
- **No Grouping**: No additional grouping is required if writing to a stream.

```java
public class WriteToSinkExample {
    public void writeData() {
        // Logic to write the stream into an output sink
    }
}
```
x??

---

---
#### Data Processing Pipelines
Data processing pipelines consist of tables, streams, and operations upon them. Tables are data at rest, while streams represent data in motion over time.

:p What is a table in the context of data processing?
??x
A table represents data that is "at rest" or accumulated over time. It acts as a container for storing data that can be observed and analyzed, but does not change its form within the table until explicitly modified by operations.
x??

---
#### Streams vs Tables
Streams are data in motion, encoding a discretized view of how tables evolve over time.

:p What is a stream?
??x
A stream consists of data points flowing continuously or intermittently. It captures the evolution of a table over time and represents the data as it moves through processing stages.
x??

---
#### Nongrouping Operations (stream → stream)
Nongrouping operations apply to streams, altering their content while keeping them in motion.

:p What are nongrouping operations?
??x
Nongrouping operations modify individual elements within a stream without changing the structure or grouping of the data. They preserve the cardinality and continuity of the stream.
x??

---
#### Grouping Operations (stream → table)
Grouping operations bring data from streams to rest, forming tables that evolve over time.

:p What are grouping operations?
??x
Grouping operations aggregate data within a stream by key or window, effectively "bringing" them to rest in a new tabular form. These operations can include windowing which incorporates event-time dimensions.
x??

---
#### Windowing Operations (stream → table)
Windowing operations incorporate the dimension of event time into grouping.

:p What is windowing?
??x
Windowing is a technique that segments data streams based on time intervals, allowing for analysis within those specific time windows. It captures data points that fall within the same time frame to perform aggregations.
x??

---
#### Merging Windows (stream → table)
Merging windows dynamically combine over time, reshaping in response to observed data.

:p How do merging windows work?
??x
Merging windows continuously update their state as new data arrives. They dynamically adjust to include or exclude data based on the current window's boundaries and the event-time conditions specified.
x??

---
#### Ungrouping Operations (table → stream)
Ungrouping operations trigger data within tables into motion, forming streams.

:p What are ungrouping operations?
??x
Ungrouping operations take data that is in a table form, ungroup them, and move them to a stream. This process captures the evolution of the table over time.
x??

---
#### Watermarks (table → stream)
Watermarks provide a notion of input completeness relative to event time.

:p What are watermarks used for?
??x
Watermarks are used in stream processing to indicate when all events with an earlier timestamp have been received. They help manage input data completeness and ensure that processing does not lag behind the event-time window.
x??

---
#### Accumulation Mode (table → stream)
The accumulation mode of triggering determines the nature of the resulting stream.

:p What role does the accumulation mode play?
??x
The accumulation mode dictates whether the stream contains deltas or values, and if it provides retraction for previous deltas. This setting is crucial in defining how data changes are tracked and managed over time.
x??

---
#### Table to Table Operations (table → table)
There are no operations that consume a table and yield a table.

:p Why can't there be direct table-to-table operations?
??x
Direct modifications between tables without converting to streams first are not possible because the state of data must change from rest to motion for processing. All changes to a table require conversion to stream, modification, and then back to a table.
x??

---

#### Streams and Tables Relativity

Streams and tables are two sides of the same coin, with streams representing data in motion and tables representing static data. The aggregation of a stream over time yields a table, while observing changes to a table over time produces a stream.

:p How do we define streams and tables relative to each other?
??x
Streams can be transformed into tables through aggregation operations, where the stream represents updates over time that are collected into a table. Conversely, tables can be viewed as a snapshot of data at rest, which can be observed for changes in real-time to produce a stream.

For example:
```java
// Example code to transform a stream into a table using Apache Beam
PCollection<KV<String, Integer>> input = ...; // Stream of updates
PCollection<KV<String, Integer>> aggregatedTable = input
    .groupByKey() // Grouping operation
    .apply("Summarize", Sum.<Integer>create()); // Aggregation to sum values by key
```
x??

---
#### Nongrouping Operations

Nongrouping operations process elements of a stream independently, maintaining the stream's inherent motion.

:p What characterizes nongrouping operations in data processing pipelines?
??x
Nongrouping operations operate on individual elements within a stream without aggregating them into groups. These operations are element-wise and do not alter the fundamental nature of the stream as data in motion.

For example:
```java
// Example code for an nongrouping operation using Apache Beam
PCollection<String> input = ...; // Stream of strings
PCollection<String> output = input.apply("Transform", ParDo.of(new DoFn<String, String>() {
    @ProcessElement
    public void processElement(@Element String element, OutputReceiver<String> out) {
        out.output(element.toUpperCase());
    }
}));
```
x??

---
#### Grouping Operations

Grouping operations are crucial as they turn streams into tables by aggregating elements based on a key.

:p What is the role of grouping operations in data processing pipelines?
??x
Grouping operations are essential because they take a stream and aggregate it over keys, transforming it into a table. This operation pauses the flow of data to group related records together, effectively bringing the stream to rest as a table where further aggregation or analysis can be performed.

For example:
```java
// Example code for grouping using Apache Beam
PCollection<KV<String, Integer>> input = ...; // Stream with key-value pairs
PCollection<KV<String, Integer>> aggregatedTable = input
    .groupByKey() // Grouping by keys
    .apply("Summarize", Sum.<Integer>create()); // Aggregation to sum values by key
```
x??

---
#### Ungrouping Operations

Ungrouping operations, or triggering operations, are used to revert the transformation from a table back into a stream, allowing for further processing.

:p What is the purpose of ungrouping (triggering) operations in data processing pipelines?
??x
Ungrouping operations, also known as triggering operations, reverse the effect of grouping by turning tables back into streams. These operations are necessary when you need to process aggregated data in a way that respects time or other constraints, essentially reactivating the flow of data.

For example:
```java
// Example code for ungrouping using Apache Beam (hypothetical)
PCollection<KV<String, Integer>> table = ...; // Aggregated table
PCollection<KV<String, Integer>> stream = table.apply("Trigger", Trigger.<String, Integer>eventTime());
```
x??

---
#### Batch vs. Streaming in Streams-and-Tables Perspective

The streams-and-tables perspective reveals that batch and streaming are fundamentally the same conceptually, differing only in how data is managed over time.

:p How do batch and streaming processes align from a streams-and-tables perspective?
??x
From a streams-and-tables perspective, both batch and streaming processes treat data as either static tables or dynamic streams. The key difference lies in the nature of the processing pipeline—whether it reads from a static table or a continuous stream—and how data flows through the system over time.

For example:
```java
// Example code for batch job using Apache Beam (static input)
PCollection<KV<String, Integer>> inputFile = ...; // Static dataset
PCollection<KV<String, Integer>> outputTable = inputFile
    .groupByKey() // Grouping operation
    .apply("Summarize", Sum.<Integer>create()); // Aggregation to sum values by key

// Example code for streaming job using Apache Beam (continuous input)
PInputDataStream<String> inputStream = ...; // Continuous stream of data
PCollection<KV<String, Integer>> outputTable = inputStream
    .apply("Transform", ParDo.of(new DoFn<String, KV<String, Integer>>() {
        @ProcessElement
        public void processElement(@Element String element, OutputReceiver<KV<String, Integer>> out) {
            // Transform and group by key
        }
    }))
    .groupByKey()
    .apply("Summarize", Sum.<Integer>create());
```
x??

---

