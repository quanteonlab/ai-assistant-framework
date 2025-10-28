# Flashcards: 2B005---Streaming-Systems_processed (Part 13)

**Starting Chapter:** 5. Exactly-Once and Side Effects. Why Exactly Once Matters

---

---
#### Exactly-Once Processing
Background context explaining exactly-once processing. This concept is crucial for ensuring that records are processed only once, which prevents data loss or duplication issues.

:p What does "exactly-once" processing mean?
??x
Exactly-once processing means ensuring that every record in a data stream is processed exactly one time without any duplicates or omissions. This ensures accurate and consistent results in data processing pipelines.
x??

---
#### Motivating Example: Google Cloud Dataflow
Background context about how specific systems, like Google Cloud Dataflow, handle exactly-once processing.

:p How does Google Cloud Dataflow ensure exactly-once processing?
??x
Google Cloud Dataflow uses a combination of techniques to ensure exactly-once processing. It employs idempotent processing where each record can be processed multiple times but yields the same result every time. This is achieved through unique identifiers for records and ensuring that operations are designed to handle retries without producing duplicate results.

```java
// Pseudocode example of an idempotent operation in Dataflow
public class IdempotentProcessor {
    private final Map<String, String> processedRecords = new ConcurrentHashMap<>();

    public void processRecord(String recordId, String record) {
        if (!processedRecords.containsKey(recordId)) {
            // Process the record
            System.out.println("Processing: " + record);
            // Mark as processed to avoid duplicates
            processedRecords.put(recordId, record);
        }
    }
}
```
x??

---
#### Challenges of Exactly-Once Processing in Other Systems
Background context on challenges faced by other systems when implementing exactly-once processing.

:p What are the difficulties faced by other general-purpose streaming systems in achieving exactly-once processing?
??x
Other general-purpose streaming systems often face challenges because they provide only at-least-once guarantees. This means that records might be processed multiple times, leading to inaccurate results due to duplicated data. These systems typically perform aggregations in memory and can lose these aggregations if machines crash, making their results unreliable.

```java
// Example of an aggregation issue in a stream processing system
public class AggregationExample {
    private final Map<String, Integer> counts = new ConcurrentHashMap<>();

    public void process(String key) {
        int count = counts.getOrDefault(key, 0);
        // Process the record and update the count
        System.out.println("Count for " + key + ": " + (count + 1));
        counts.put(key, count + 1);
    }
}
```
The system relies on in-memory state, which can be lost if a machine crashes, leading to potential data loss.

x??

---
#### Lambda Architecture
Background context explaining the Lambda architecture and its limitations.

:p What is the Lambda Architecture?
??x
The Lambda Architecture is a strategy for processing large-scale data that combines both batch processing (for full accuracy) and stream processing (for low-latency results). It consists of two pipelines: one for real-time streaming, which processes data quickly but inaccurately due to potential failures; and another for batch processing, which runs later to produce the correct answer. However, this architecture is complex and has limitations such as inaccuracy, inconsistency, complexity, and unpredictability.

x??

---

#### Lambda Architecture Limitations

Background context explaining the limitations of the Lambda architecture, especially regarding latency and accuracy. The Lambda architecture is designed for handling large-scale data processing but doesn't inherently provide low-latency correct results.

:p What are some business use cases that require low-latency correct results?
??x
Low-latency use cases often include real-time monitoring systems, financial trading platforms, or any scenario where immediate feedback and decision-making based on the most current data is crucial. The Lambda architecture, while scalable, may not meet these requirements due to its batch-processing nature.

```java
// Example of a simple Lambda function that processes data with high latency
public void processRecord(String record) {
    // Simulated processing with a delay
    try {
        Thread.sleep(1000); // Simulate 1 second of processing time
    } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
    }
    System.out.println("Processed record: " + record);
}
```
x??

---

#### Exactly-Once Processing in Beam

Background context on exactly-once processing, which ensures that records are processed exactly once and avoids data loss. This is crucial for reliable data processing.

:p What does the term "exactly-once" mean in the context of Beam and data processing?
??x
Exactly-once processing means that each record is processed exactly one time, ensuring both accuracy (no duplicates) and completeness (no data loss). In Beam, this feature helps users count on accurate results while avoiding risks of data loss.

```java
// Example configuration for a Beam pipeline to ensure at-least-once processing
Pipeline p = Pipeline.create(options);
PCollection<String> lines = p.apply(TextIO.read().from("input.txt"));
lines.apply(ParDo.of(new DoFn<String, String>() {
    @ProcessElement
    public void processElement(@Element String line, OutputReceiver<String> out) throws IOException {
        // Process the element here, ensuring it is processed exactly once.
        out.output(line);
    }
}));
```
x??

---

#### Accuracy vs. Completeness

Background context on how Beam pipelines handle late data and completeness. The accuracy of a pipeline ensures no records are dropped or duplicated, while completeness addresses whether all relevant data is processed.

:p How does Beam handle late arriving data in terms of accuracy?
??x
Beam allows users to configure a latency window during which late data can still be processed accurately. Any data arriving after this window is explicitly dropped, contributing to completeness but not affecting the accuracy of on-time records.

```java
// Example configuration for processing with a 5-minute grace period for late data
Pipeline p = Pipeline.create(options);
PCollection<String> lines = p.apply(TextIO.read().from("input.txt"));
lines.apply(Window.into(FixedWindows.of(Duration.standardMinutes(5))));
```
x??

---

#### Side Effects in Beam

Background context on custom code execution within a Beam pipeline and the challenges of ensuring it runs exactly once. Custom side effects can lead to issues if not managed properly.

:p How does Beam handle custom side effects during record processing?
??x
Beam does not guarantee that custom code is run only once per record, even for streaming or batch pipelines. Users must manage these side effects manually to ensure they do not cause data duplication or loss.

```java
// Example of a DoFn with potential side effects
public class ProcessRecord extends DoFn<String, String> {
    @ProcessElement
    public void processElement(@Element String record) throws IOException {
        // Custom processing that may have side effects
        System.out.println("Processing: " + record);
        // Additional code that might not be idempotent
    }
}
```
x??

---

#### At-Least-Once Processing
Background context explaining at-least-once processing. This is necessary to guarantee that a record will be processed even if a worker fails during the execution of a pipeline. However, it can lead to non-idempotent side effects being executed more than once for a given record.
:p What is at-least-once processing in Apache Beam?
??x
At-least-once processing ensures that each record in a Dataflow pipeline will be processed at least once even if a worker fails. However, because the same record might run through multiple workers or get re-executed due to failures, non-idempotent side effects (such as writing to an external service) can be executed more than once for a given record.
```java
// Example of a non-idempotent side effect in a pipeline
p.apply("ReadRecords", Read.from(MySource.class))
   .apply("ProcessRecords", new DoFn<Record, Void>() {
       @ProcessElement
       public void process(ProcessContext c) {
           // Non-idempotent operation
           updateExternalService(c.element());
       }
   });
```
x??

---

#### Exactly-Once Processing
Background context explaining exactly-once processing. This is a more stringent requirement where each record should be processed exactly once, ensuring idempotent side effects are handled correctly.
:p What does exactly-once processing mean in Apache Beam?
??x
Exactly-once processing means that each record in a Dataflow pipeline will be processed exactly once. It ensures that non-idempotent operations (such as writing to an external service) are executed only once, even if there are worker failures and re-executions.
```java
// Example of ensuring exactly-once processing with idempotent operations
p.apply("ReadRecords", Read.from(MySource.class))
   .apply("ProcessAndWriteRecords", new DoFn<Record, Void>() {
       @ProcessElement
       public void process(ProcessContext c) {
           // Idempotent operation
           writeToExternalServiceIdempotently(c.element());
       }
   });
```
x??

---

#### Pipeline Example: Simple Streaming Aggregation
Background context explaining the pipeline example provided. The example shows how Dataflow pipelines compute windowed aggregations over event data.
:p What does the provided pipeline example do?
??x
The provided pipeline example reads streaming events, computes per-user counts of events in 1-minute windows, and then aggregates these counts to calculate a global count for each minute. It writes both sets of results (per-user and global counts) to unspecified sinks.
```java
Pipeline p = Pipeline.create(options);
PCollection<...> perUserCounts = 
    p.apply(ReadFromUnboundedSource.read())
      .apply(new KeyByUser())
      .Window.<..>into(FixedWindows.of(Duration.standardMinutes(1)))
      .apply(Count.perKey());
perUserCounts.apply(new ProcessPerUserCountsAndWriteToSink());
perUserCounts.apply(Values.create()).apply(Count.globally())
  .apply(new ProcessGlobalCountAndWriteToSink());
p.run();
```
x??

---

#### Shuffle in Dataflow
Background context explaining the shuffle process. Shuffles ensure that all records with the same key are processed on the same machine, which is crucial for accurate data processing.
:p What is a shuffle in Apache Beam Dataflow?
??x
A shuffle in Apache Beam Dataflow refers to the process where records with the same key are grouped and moved between workers so they can be processed together. This is necessary for operations like `GroupByKey`, ensuring that all records sharing the same key end up on the same machine.
```java
// Example of a shuffle operation
p.apply("KeyByUser", KeyByUser())
   .apply("WindowIntoFixedWindows", Window.<..>into(FixedWindows.of(Duration.standardMinutes(1))))
   .apply("CountPerKey", Count.perKey());
```
x??

---

#### End-to-End Exactly Once Processing for Sources and Sinks
Background context explaining end-to-end exactly once processing. This is the capability provided by Dataflow to handle interactions with external sources and sinks in a way that ensures data integrity.
:p What does "end-to-end exactly once" mean in Apache Beam?
??x
End-to-end exactly once processing means that Dataflow guarantees that every record read from an external source will be processed exactly once, and every record written to an external sink will be committed only after successful processing. This is crucial for maintaining data integrity when reading from or writing to the outside world.
```java
// Example of a pipeline with end-to-end exactly once processing
p.apply("ReadFromSource", Read.from(MyExternalSource.class))
   .apply("ProcessRecords", new DoFn<Record, Void>() {
       @ProcessElement
       public void process(ProcessContext c) {
           // Process the record and write to an external sink
           writeToExternalSink(c.element());
       }
   })
   .apply("WriteToSink", Write.to(MyExternalSink.class));
```
x??

---

