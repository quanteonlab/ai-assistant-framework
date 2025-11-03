# Flashcards: 2B005---Streaming-Systems_processed (Part 18)

**Starting Chapter:** Motivation

---

#### Motivation for Persistent State
Background context: The text explains that people write books to capture ephemeral ideas so they can be revisited. Similarly, persistent state is required in data processing pipelines to store and revisit data over time.

:p What motivates the need for persistent state in data processing pipelines?
??x
Persistent state is necessary because it allows us to capture and revisit data over time, which is crucial for operations that require historical context or long-term analysis. Without persistent state, transient computations would be lost, making it impossible to review past data or build upon previous results.
x??

---
#### Implicit State in Pipelines
Background context: The text mentions that two forms of implicit state are often found within pipelines but do not meet the criteria for robust storage.

:p What are the two forms of implicit state typically found in pipelines?
??x
The two forms of implicit state are:
1. Data stored on local disk, which is unreliable and can be lost.
2. Data stored in memory, which is volatile and prone to loss during system crashes or restarts.
x??

---
#### Advertising Conversion Attribution Use Case
Background context: The text uses the advertising conversion attribution as an example of a use case that struggles with implicit state.

:p How does the advertising conversion attribution use case illustrate the limitations of implicit state?
??x
The advertising conversion attribution use case illustrates the limitations of implicit state because it requires tracking events over time and correlating them to specific ad campaigns. Implicit state, stored in memory or local disk, cannot reliably capture this historical data, making it difficult to accurately attribute conversions to different ads.

This is problematic because:
- Data must be retained across multiple sessions.
- Historical context is crucial for accurate attribution.
x??

---
#### Explicit State Management
Background context: The text explains the need for an explicit form of persistent state management that can handle robust storage requirements.

:p What are the key features of a general, explicit form of persistent state management?
??x
A general, explicit form of persistent state management should have these key features:
- Robust and durable storage.
- Ability to handle large volumes of data.
- Support for distributed computing environments.
- Consistency across multiple nodes or machines.

These features ensure that the state can be reliably stored and accessed even in the face of failures or scaling requirements.
x??

---
#### Apache Beam State API
Background context: The text introduces a concrete manifestation of one such state API, as found in Apache Beam, to manage persistent state explicitly.

:p What is an example of a state API for managing persistent state, as mentioned in the text?
??x
Apache Beam provides a state API that allows developers to explicitly manage persistent state. This API supports operations like reading from and writing to state, handling state changes during processing, and ensuring consistent behavior across distributed systems.

Example code in Pseudocode:
```java
public class AdConversionAttribution {
    private StateSpec<Long> impressionsCount;

    public void processElement(PaneElement<String> element) {
        // Read the count of impressions for this ad
        long count = state.impressionsCount.read().get();

        // Increment the count and write back to state
        count += 1;
        state.impressionsCount.write(count);
    }
}
```
This example shows how to read from and write to a state object in Apache Beam, ensuring that the state is managed explicitly and robustly.
x??

---

#### Inevitability of Failure
Background context explaining why long-running pipelines need some form of persistent state. This is critical for unbounded data processing, where pipelines are expected to run indefinitely but face interruptions due to various reasons like machine failures or code changes.

:p Why is persistent state necessary in the case of long-running pipelines processing unbounded input data?
??x
Persistent state is necessary because long-running pipelines must be able to resume from their last known checkpoint after encountering an interruption. This ensures that they can continue processing without losing progress due to unforeseen interruptions such as machine failures, planned maintenance, or misconfigured commands.

```java
public class ExamplePipeline {
    public void processUnboundedData() {
        // Logic for handling unbounded data with persistent state
        while (true) {
            try {
                // Process the next batch of data
                handleBatchOfData(getNextBatch());
            } catch (Exception e) {
                log.error("Pipeline interrupted, saving checkpoint and retrying.");
                saveCheckpoint();
                continue;
            }
        }
    }

    private void handleBatchOfData(List<DataRecord> records) {
        // Process the records here
    }

    private List<DataRecord> getNextBatch() {
        // Fetch the next batch of data from the source
        return dataSource.getRecords();
    }

    private void saveCheckpoint() {
        // Save the current state to a persistent storage like a database or file system
    }
}
```
x??

---

#### Bounded vs. Unbounded Data Processing
Background context explaining how bounded and unbounded datasets are historically processed differently, with batch systems often assuming reprocessing is possible and streaming systems built for infinite data.

:p Why do batch pipelines use persistent state, if at all?
??x
Batch pipelines typically do not require as much persistent state because they can assume that input data can be reprocessed in their entirety upon failure. However, some level of checkpointing might still be used to minimize the cost of recomputations and ensure correctness.

```java
public class ExampleBatchPipeline {
    public void processBoundedData() {
        // Logic for handling bounded data without needing persistent state
        try (BufferedReader reader = new BufferedReader(new FileReader("input.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                processDataLine(line);
            }
        } catch (Exception e) {
            log.error("Pipeline interrupted, but can be restarted from the beginning.");
        }
    }

    private void processDataLine(String line) {
        // Process each data line here
    }
}
```
x??

---

#### Correctness and Efficiency Through Checkpointing
Background context explaining how persistent state helps maintain correctness in light of ephemeral inputs and minimizes redundant work.

:p How does checkpointing help with minimizing duplicated work and data persisted during failures?
??x
Checkpointing helps by recording the intermediate state of a pipeline, allowing it to resume from a known point after a failure without needing to recompute all previous steps. This reduces both the computational overhead and storage requirements needed for recovery.

```java
public class ExampleCheckpointedPipeline {
    public void processStreamWithCheckpoints() {
        try (KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props)) {
            Properties props = new Properties();
            // Initialize properties with broker addresses etc.
            while (true) {
                consumer.poll(Duration.ofMillis(100)); // Consume data
                try {
                    // Process the batch of records
                    processBatch(consumer);
                } catch (Exception e) {
                    log.error("Pipeline interrupted, saving checkpoint and retrying.");
                    saveCheckpoint();
                    continue;
                }
            }
        }
    }

    private void processBatch(KafkaConsumer<String, String> consumer) {
        // Logic to process each batch of records
    }

    private void saveCheckpoint() {
        // Save the current state to a persistent storage like a Kafka offset or file system
    }
}
```
x??

---

#### Intermediate Data and Checkpointing
Background context explaining how checkpointing intermediate data can significantly reduce the amount of data stored during failures, making recovery more efficient.

:p How does checkpointing partial results help in reducing redundant work?
??x
Checkpointing partial results helps by storing only essential parts of the ongoing computation. This means that upon failure, only these critical pieces need to be reprocessed, greatly reducing both the computational and storage overheads compared to recomputing everything from scratch.

```java
public class ExampleMeanCalculation {
    public double calculateMean() {
        int count = 0;
        long sum = 0;
        try (BufferedReader reader = new BufferedReader(new FileReader("input.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (count % CHECKPOINT_INTERVAL == 0 && count > 0) {
                    saveCheckpoint(count, sum);
                }
                long value = Long.parseLong(line); // Simplified for example
                sum += value;
                ++count;
            }
        } catch (Exception e) {
            log.error("Pipeline interrupted, saving checkpoint and retrying.");
            saveCheckpoint(count, sum);
        }
        return sum / count;
    }

    private void saveCheckpoint(int count, long sum) {
        // Save the current state to a persistent storage
        log.info("Checkpoint saved at count: {}, sum: {}", count, sum);
    }
}
```
x??

---

#### Persistent State Management for Unbounded Data
Persistent state is crucial for data processing pipelines handling unbounded inputs. It helps maintain correctness and efficient fault tolerance, balancing between storing all data (for strong consistency) and minimizing storage to enhance efficiency.

:p What is persistent state management important for in the context of unbounded data processing?
??x
Persistent state management ensures that pipelines can process infinite or very large datasets while maintaining correctness and providing strong consistency guarantees. It helps in managing the amount of data stored by only keeping necessary states, thus reducing overall storage needs.
x??

---
#### Raw Grouping in Apache Beam Pipelines
In Apache Beam, raw grouping aggregates all incoming elements for a given key into an `Iterable`. This method is straightforward but can lead to high memory usage when dealing with large datasets.

:p How does the `GroupByKey` operation work in Apache Beam?
??x
The `GroupByKey` operation in Apache Beam groups all elements based on their keys. It appends new elements arriving under the same key to existing groups, producing a table at rest where each group is represented by an `Iterable`. This method ensures that all related data for a given key are stored together.

For example:
```java
PCollection<String> raw = ...;
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Iterable<Integer>>> groupedScores = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .apply(GroupByKey.create());
```
x??

---
#### Summing Values Using Raw Grouping
To sum values in groups using raw grouping, one can apply a transformation to convert the `Iterable` of integers into an `IntStream`, then perform the summation.

:p How does the pipeline sum up scores for each team after applying `GroupByKey`?
??x
After applying `GroupByKey`, we have a `PCollection<KV<Team, Iterable<Integer>>>`. To sum these values, we use a transformation that maps over each element and converts the `Iterable<Integer>` to an `IntStream`, then sums it up.

Example code:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(MapElements.via((KV<String, Iterable<Integer>> kv) ->
        StreamSupport.intStream(kv.getValue().spliterator(), false).sum()));
```
This pipeline step processes each group (team), converts the collection of scores to a stream, and sums them up.

x??

---
#### Early, On-Time, and Late Triggering
Triggers in Apache Beam pipelines control when pane emissions occur. In this context, triggers handle early and late firings based on watermarks or other conditions.

:p What does the `GroupByKey` operation do in terms of data grouping?
??x
The `GroupByKey` operation groups elements by their keys into an `Iterable`. This means that all values associated with a particular key are collected together. The output is a table where each group is represented as a list or iterable collection.

Example:
```java
PCollection<KV<Team, Iterable<Integer>>> groupedScores = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .apply(GroupByKey.create());
```
x??

---
#### Triggering Early and Late Firing in Pipelines
Apache Beam pipelines use triggers to determine when to emit pane results based on certain conditions like watermarks.

:p How are early and late firings configured using the `GroupByKey` operation?
??x
Early and late firings can be configured with triggers applied after a `GroupByKey`. For instance, we configure an early firing after one minute and a late firing after receiving one more element.

Example:
```java
PCollection<KV<Team, Integer>> input = ...;
PCollection<KV<Team, Iterable<Integer>>> groupedScores = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .apply(Triggering
        .afterWatermark()
        .withEarlyFirings(AlignedDelay(ONE_MINUTE))
        .withLateFirings(AfterCount(1)));
```
x??

---

