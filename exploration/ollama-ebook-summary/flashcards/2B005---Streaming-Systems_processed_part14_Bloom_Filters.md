# Flashcards: 2B005---Streaming-Systems_processed (Part 14)

**Starting Chapter:** Bloom Filters

---

#### Shuffle Mechanism in Dataflow
Dataflow uses RPCs (Remote Procedure Calls) for its shuffle process. RPC failures can occur due to network interruptions, timeouts, or server crashes. To ensure that every record is shuffled exactly once, Dataflow implements upstream backup: the sender retries failed RPCs until it receives positive acknowledgment of receipt.
:p What mechanism does Dataflow use to ensure records are not lost during shuffling?
??x
Dataflow ensures records are not lost by employing upstream backup. This means the sender retries failed RPCs until a positive acknowledgment is received. Even if the sender crashes, retrying continues, ensuring each record is delivered at least once.
```java
public class Sender {
    public void sendRecord(byte[] data) {
        while (!isAcknowledged(data)) {
            try {
                network.sendRPC(data);
            } catch (Exception e) {
                // Retry logic here
            }
        }
    }

    private boolean isAcknowledged(byte[] data) {
        // Check if the data has been acknowledged
    }
}
```
x??

---

#### Duplicate Detection in Shuffle
To handle retries that might introduce duplicates, Dataflow tags every message with a unique identifier. Each receiver maintains a catalog of identifiers it has processed. Upon receiving a record, its identifier is checked against this catalog; if found, the record is dropped as a duplicate.
:p How does Dataflow detect and remove duplicates in shuffles?
??x
Dataflow uses unique identifiers to tag every message sent during shuffling. Each receiver maintains a catalog of identifiers that have already been seen and processed. When receiving a new record, its identifier is checked against this catalog. If the identifier exists, indicating a duplicate, the record is dropped.
```java
public class Receiver {
    private final Set<Long> processedIdentifiers = new HashSet<>();

    public void processRecord(byte[] data, long id) {
        if (processedIdentifiers.contains(id)) {
            // Drop as duplicate
            return;
        }
        // Process the record
        processedIdentifiers.add(id);
    }
}
```
x??

---

#### Determinism in ParDo Operations
In Dataflow's Beam model, user code can produce nondeterministic outputs. A `ParDo` might execute twice on the same input due to retries and generate different outputs each time. Ensuring only one output commits into the pipeline is challenging because deterministic IDs cannot be guaranteed with nondeterministic behavior.
:p How does Dataflow handle nondeterminism in ParDo operations?
??x
Dataflow faces challenges with nondeterminism in `ParDo` operations where a single input record can produce different outputs upon retry. Determining which output should commit into the pipeline is difficult because nondeterministic logic makes it hard to ensure both outputs have the same deterministic ID.
```java
public class ParDoExample {
    public PCollection<String> process(PCollection<String> input) {
        return input.apply("ParDo", ParDo.of(new DoFn<String, String>() {
            @ProcessElement
            public void processElement(@Element String element, OutputReceiver<String> receiver) {
                try {
                    // Nondeterministic operation that might produce different results on retry
                    String output = someOperation(element);
                    receiver.output(output);
                } catch (Exception e) {
                    // Retry logic here
                }
            }
        }));
    }

    private String someOperation(String element) throws Exception {
        // Some nondeterministic operation
    }
}
```
x??

---

#### Addressing Determinism in Shuffle and ParDo
Ensuring determinism is crucial, especially with retries. Dataflow needs to manage both the shuffle process and `ParDo` operations carefully. For shuffles, using unique identifiers helps detect duplicates. For `ParDo`, ensuring only one output commits involves careful handling of nondeterministic behavior.
:p What steps does Dataflow take to address determinism in its operations?
??x
Dataflow addresses determinism by:
1. **Shuffle Process**: Using unique identifiers for messages and maintaining a catalog on the receiver side to detect duplicates.
2. **ParDo Operations**: Handling nondeterministic outputs carefully, ensuring only one deterministic output commits into the pipeline despite potential retries.

This involves managing unique IDs and using robust retry mechanisms.
```java
public class DataflowExample {
    public void handleShuffle() {
        // Unique identifier handling for shuffle
    }

    public void handleParDo(PCollection<String> input) {
        return input.apply("ParDo", ParDo.of(new DoFn<String, String>() {
            @ProcessElement
            public void processElement(@Element String element, OutputReceiver<String> receiver) {
                try {
                    // Process the element deterministically
                    String output = someDeterministicOperation(element);
                    receiver.output(output);
                } catch (Exception e) {
                    // Retry logic here
                }
            }
        }));
    }

    private String someDeterministicOperation(String element) {
        // Deterministic operation for ParDo
    }
}
```
x??

---
#### Deterministic vs. Nondeterministic Processing
Background context: Many data processing pipelines require transforms that are inherently nondeterministic due to external factors like current time, random numbers, or external data sources such as Cloud Bigtable. This can lead to issues where pipeline outputs vary between runs.
:p What is the difference between deterministic and nondeterministic processing in data pipelines?
??x
Nondeterministic processing occurs when a task's outcome may change due to external factors, such as current time or data from external systems that might change during retries. Deterministic processing ensures consistent results regardless of how many times it is run.
For example:
```java
// Nondeterministic code: Lookup in Cloud Bigtable
String lookupData = bigtableClient.lookup(rowKey);
```
x??

---
#### Checkpointing Mechanism in Dataflow
Background context: To address nondeterministic processing, Apache Dataflow uses checkpointing to make the process effectively deterministic. Each output from a transform is stored along with its unique ID before being delivered to the next stage.
:p What is the role of checkpointing in ensuring determinism in data pipelines?
??x
Checkpointing helps ensure that if a pipeline needs to be retried, it can replay the exact state it was in when the checkpoint was taken. This prevents re-executing user code unnecessarily and ensures consistent results across retries.
Example:
```java
// Checkpointing example
DataflowPipelineOptions options = PipelineOptionsFactory.create();
options.setCheckpointLocation("gs://my-checkpoints");
Pipeline p = Pipeline.create(options);
p.apply("ReadFromSource", Read.from(...))
   .apply("ProcessWithTransform", ParDo.of(new DoFn<...>(...)
         .withOutputTags((...) -> PCollectionTag.TEMP_TAG)))
   .getOptions().setCheckpointingType(CheckpointOptions.BEFORE_MAJOR_COMBINER);
```
x??

---
#### Exactly-once Shuffle Delivery
Background context: To implement exactly-once shuffle delivery, Dataflow uses a consistent store to prevent duplicates and reduce I/O. This ensures that records are processed only once, even if they arrive out of order or late.
:p What is the goal of exactly-once shuffle delivery in Apache Dataflow?
??x
The goal is to ensure that each record is processed exactly once, regardless of when it arrives or how many times the pipeline retries due to failures. This prevents duplicate processing and ensures data integrity.
Example:
```java
// Exactly-once shuffle example
PTransform<PInput, POutput> ptransform = ...;
p.apply(ptransform).setCoder(...);
```
x??

---
#### Graph Optimization in Dataflow
Background context: To reduce the overhead of checkpointing and I/O, Dataflow employs graph optimization techniques like fusion. Fusion combines multiple logical steps into a single execution stage to minimize data transfer and state usage.
:p What is the concept of "fusion" in Apache Dataflow?
??x
Fusion is an optimization technique where the Dataflow service merges multiple logical steps into a single physical step. This reduces the amount of data transfer and minimizes state usage, thereby improving performance.
Example:
```java
// Fusion example
PTransform<PInput, POutput> fusedStep = p.apply("FusedStep", ParDo.of(new DoFn<...>(...)
      .withOutputTags((...) -> PCollectionTag.TEMP_TAG)))
      .apply(Fusion.disable());
```
x??

---
#### Bloom Filters for I/O Reduction
Background context: To further reduce the I/O overhead, Dataflow uses Bloom filters. These probabilistic data structures help determine whether a record has already been processed without checking the full catalog of IDs.
:p What are Bloom filters and how do they help in reducing I/O?
??x
Bloom filters are used to reduce I/O by providing a space-efficient way to test whether an element is a member of a set. They allow checking if a record has already been seen, avoiding unnecessary checks against the full catalog of IDs.
Example:
```java
// Using Bloom filter for deduplication
PTransform<PInput, POutput> deduplicate = p.apply("Deduplicate", GroupByKey.create())
      .apply(BloomFilter.<Key, Value>create(1000));
```
x??

#### Bloom Filters for Duplicate Detection
Background context: Bloom filters are compact data structures that allow quick set-membership checks. They can return false positives but never false negatives, making them perfect for scenarios where avoiding unnecessary lookups is crucial. The primary use case here is detecting duplicate records in a data pipeline.
:p What are Bloom filters and how do they work?
??x
Bloom filters are probabilistic data structures used to test whether an element is a member of a set. They can determine if something might be in the set (with a certain probability) or for sure not in the set. The core idea is that false positives are allowed, but false negatives are not.
```java
// Pseudo-code for adding elements to a Bloom filter
public class BloomFilter {
    private BitSet bitSet;
    private int[] hashFunctions;

    public void add(int element) {
        // Hash the element using multiple functions and set bits in the bitSet
        for (int i : hashFunctions) {
            bitSet.set(hash(i, element));
        }
    }

    public boolean mightContain(int element) {
        for (int i : hashFunctions) {
            if (!bitSet.get(hash(i, element))) {
                return false;
            }
        }
        // If all bits are set, it's a potential match
        return true;
    }

    private int hash(int index, int element) {
        // Hash function implementation
    }
}
```
x??

---

#### Use of Bloom Filters in Dataflow Pipeline
Background context: In the described data pipeline, Bloom filters help reduce unnecessary lookups by quickly identifying non-duplicate records. The use of Bloom filters here is optimized for exactly-once processing, where most incoming records are not duplicates.
:p How do Bloom filters improve performance in a healthy pipeline?
??x
Bloom filters improve performance by allowing quick checks to determine if a record has already been processed. If the filter indicates that a record might be a duplicate (with low false positives), the more expensive lookup from stable storage can be skipped. Only when the filter returns true is the secondary check performed.
```java
// Pseudo-code for BloomFilter usage in Dataflow pipeline
public class PipelineBloomFilter {
    private Map<Interval, BloomFilter> bloomFilters;

    public void handleRecord(Record record) {
        // Get the appropriate Bloom filter based on the timestamp
        BloomFilter currentFilter = getBloomFilter(record.getTime());
        
        // Check if the record might be a duplicate
        if (!currentFilter.mightContain(record.getId())) {
            // No need for expensive lookup; record is not a duplicate
            return;
        }

        // Perform the secondary catalog lookup to verify the record's status
        catalogLookup(record);
    }
}
```
x??

---

#### Time-Based Bloom Filter Implementation
Background context: To mitigate the issue of false positives over time, Dataflow creates separate Bloom filters for every 10-minute range. This approach ensures that filters do not saturate and can be garbage-collected over time.
:p How does Dataflow manage Bloom filter saturation?
??x
Dataflow manages Bloom filter saturation by creating a new filter for each 10-minute interval and using the system timestamp to determine which filter to query. Filters are garbage collected over time, preventing them from becoming overly large and reducing false positives.
```java
// Pseudo-code for managing time-based Bloom filters in Dataflow
public class DataflowBloomFilters {
    private Map<Long, BloomFilter> bloomFilters;

    public void initialize() {
        // Load existing Bloom filters or create new ones based on current timestamp
        long currentTime = System.currentTimeMillis();
        Long nearestTimestamp = currentTime / 600_000 * 600_000; // Nearest 10-minute interval

        if (!bloomFilters.containsKey(nearestTimestamp)) {
            bloomFilters.put(nearestTimestamp, new BloomFilter());
        }

        currentBloomFilter = bloomFilters.get(nearestTimestamp);
    }

    public void handleRecord(Record record) {
        long currentTime = System.currentTimeMillis();
        Long nearestTimestamp = currentTime / 600_000 * 600_000; // Nearest 10-minute interval

        if (!bloomFilters.containsKey(nearestTimestamp)) {
            bloomFilters.put(nearestTimestamp, new BloomFilter());
        }

        currentBloomFilter = bloomFilters.get(nearestTimestamp);
        handleRecordWithBloomFilter(record);
    }
}
```
x??

---

#### Duplicate Detection in the Pipeline
Background context: The pipeline uses a combination of Bloom filters and catalog lookups to ensure exactly-once processing. Records are first checked against a Bloom filter, and only those that might be duplicates trigger a secondary lookup.
:p How does the pipeline handle duplicate detection for incoming records?
??x
The pipeline handles duplicate detection by using Bloom filters to quickly identify non-duplicate records. If a record passes the Bloom filter check (i.e., it is not flagged as potentially a duplicate), no further action is needed. Otherwise, a catalog lookup is performed to confirm whether the record is indeed a duplicate.
```java
// Pseudo-code for handling duplicates in Dataflow pipeline
public class DuplicateHandler {
    private Map<Long, BloomFilter> bloomFilters;

    public void handleRecord(Record record) {
        long currentTime = System.currentTimeMillis();
        Long nearestTimestamp = currentTime / 600_000 * 600_000; // Nearest 10-minute interval

        if (!bloomFilters.containsKey(nearestTimestamp)) {
            bloomFilters.put(nearestTimestamp, new BloomFilter());
        }

        currentBloomFilter = bloomFilters.get(nearestTimestamp);
        
        if (currentBloomFilter.mightContain(record.getId())) {
            catalogLookup(record); // Perform secondary lookup
        } else {
            processRecord(record); // No need for further processing
        }
    }
}
```
x??

---

#### Garbage Collection Strategy for Record IDs

Background context: The Dataflow framework stores a catalog of unique record IDs seen by each worker to ensure state consistency and uniqueness. To manage storage efficiently, it uses garbage collection (GC) to remove old identifiers that have been acknowledged.

:p How does Dataflow handle the garbage collection of record IDs?
??x
Dataflow implements garbage collection based on system timestamps rather than sequence numbers. It calculates a watermark using these timestamps to identify records that can be safely removed. This approach leverages the processing-time watermark discussed in Chapter 3, which helps determine when old records have been acknowledged and thus can be cleaned up.

```java
// Pseudocode for calculating garbage collection watermark
public long calculateGarbageCollectionWatermark() {
    return System.currentTimeMillis() - gcTimeout;
}
```
x??

---

#### Handling Network Remnants

Background context: Network remnants occur when old messages get stuck inside the network and later resurface, potentially leading to processing of duplicate records. Dataflow manages this by ensuring that record deliveries are acknowledged before advancing the low watermark for garbage collection.

:p How does Dataflow handle records with old timestamps due to network remnants?
??x
When a record arrives with an old timestamp after the low watermark has already advanced, it is considered a duplicate and ignored because it has already been successfully processed. This is managed by ensuring that the low watermark only advances once all deliveries up to that point have been acknowledged.

```java
// Pseudocode for handling network remnants
public boolean shouldIgnoreRecord(long timestamp) {
    return garbageCollectionWatermark > timestamp;
}
```
x??

---

#### Deterministic Sources in Dataflow

Background context: Deterministic sources ensure the same order and unique identification of records every time they are read. This is crucial for guaranteeing exactly-once processing without duplicates.

:p What makes a source deterministic in Dataflow?
??x
A source is considered deterministic if it consistently generates records with unique identifiers that do not change upon re-reading. Examples include reading from files, where the byte location and filename uniquely identify each record, or using Apache Kafka topics, which maintain a fixed order within partitions.

```java
// Example of handling file-based deterministic source in Java
public class FileBasedSource {
    public List<String> readRecords(String filePath) throws IOException {
        // Read records from file ensuring unique identifiers
        return Files.readAllLines(Paths.get(filePath));
    }
}
```
x??

---

#### Exactly-Once Processing for Sources

Background context: Dataflow needs to ensure that every record produced by a source is processed exactly once, especially in cases where sources might fail and need retries. Deterministic sources simplify this process.

:p How does Dataflow handle exactly-once processing for deterministic sources?
??x
For deterministic sources like file readers or Apache Kafka partitions, Dataflow can automatically manage exactly-once processing because the order and uniqueness of records are guaranteed. The service can generate unique IDs based on filenames and byte locations for files, ensuring no duplicates.

```java
// Example of handling deterministic source in Java
public class DeterministicSource {
    public List<String> readRecords(String fileName) throws IOException {
        // Ensure each record has a unique ID
        return Files.readAllLines(Paths.get(fileName)).stream()
                .map(record -> generateUniqueID(record, fileName))
                .collect(Collectors.toList());
    }
    
    private String generateUniqueID(String record, String fileName) {
        // Generate unique ID based on record content and file name
        return UUID.randomUUID().toString();
    }
}
```
x??

---

These flashcards cover the key concepts in the provided text, ensuring a clear understanding of garbage collection strategies, handling network remnants, deterministic sources, and exactly-once processing.

#### Nondeterministic Sources and Pub/Sub
Background context explaining how Dataflow handles nondeterministic sources like Google Cloud Pub/Sub. Multiple subscribers can pull from a Pub/Sub topic, but which subscribers receive a given message is unpredictable. When processing fails, Pub/Sub will redeliver messages to different workers in potentially a different order.

:p How does Dataflow handle messages from Google Cloud Pub/Sub?
??x
Dataflow handles messages from Google Cloud Pub/Sub by considering them nondeterministic because multiple subscribers can pull the same message and the delivery might be to different workers with potential reordering. To manage this, sources that are nondeterministic (like Pub/Sub) must provide unique record IDs so Dataflow can filter out duplicate records during retries.

```java
public class PubSubSource extends UnboundedReader {
    @Override
    public RecordId getCurrentRecordId() {
        // Generate a unique ID for each message.
        return new RecordId(message.getUniqueMessageID());
    }
}
```
x??

---

#### Exactly Once Semantics in Sinks
Background context explaining the challenge of ensuring exactly once delivery when writing data to external sinks. Dataflow does not guarantee exactly-once application of side effects, so custom logic is needed to ensure outputs are delivered only once.

:p How can a sink ensure that its output is delivered exactly once?
??x
To ensure exactly once semantics in a sink, built-in sinks provided by the Beam SDK should be used whenever possible. These sinks are designed to handle duplicates correctly even if executed multiple times. For custom sinks, idempotent operations should be implemented so that they can be replayed without changing the output. However, if components within the side-effect operation (like a DoFn in windowed aggregation) are nondeterministic and might change on replay, additional logic is needed to ensure deterministic behavior.

```java
public class IdempotentSink extends PColl.writeFile {
    @Override
    public void write(String fileName, String content) throws IOException {
        // Ensure the operation is idempotent.
        Files.write(Paths.get(fileName), content.getBytes(), StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
    }
}
```
x??

---

#### Handling Nondeterminism in Sinks
Background context explaining that even with idempotent operations, nondeterminism can still occur due to changes in logical record sets during replay. Dataflow guarantees only one version of a DoFn’s output per shuffle boundary.

:p How does nondeterminism affect sinks and side-effect operations?
??x
Nondeterminism affects sinks and side-effect operations because it can change the set of records processed between different executions. For instance, in windowed aggregations, late elements might cause the window to fire again with a different logical record set. Idempotency alone is insufficient because different logical record sets are sent each time. To handle this, additional logic must be implemented to ensure deterministic behavior, such as maintaining consistent state or using versioning mechanisms.

```java
public class DeterministicWindowSink extends PColl.writeFile {
    private int version = 0;

    @Override
    public void write(String fileName, String content) throws IOException {
        // Maintain a version number to ensure idempotency across different invocations.
        Files.write(Paths.get(fileName + "?v=" + version), content.getBytes(), StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
    }
}
```
x??

---

#### Shuffle Boundaries in Dataflow
Background context explaining that Dataflow guarantees only one version of a DoFn’s output per shuffle boundary. This ensures that even if a worker crashes and restarts, the same data will not be processed again after the shuffle.

:p How does Dataflow ensure exactly once semantics with shuffle boundaries?
??x
Dataflow ensures exactly once semantics within the context of shuffle boundaries by guaranteeing that only one version of a DoFn’s output can pass through each shuffle boundary. This means that if a worker crashes before committing its window processing, when it restarts, the same window will not be processed again because Dataflow has already ensured that the output is deterministic up to the last committed state.

```java
public class ShuffleBoundaryExample {
    @ProcessElement
    public void process(ProcessContext context) throws IOException {
        // Process elements and ensure they are committed before a shuffle boundary.
        if (context.shuffleBoundaryPassed()) {
            // Commit the processed data after ensuring it is deterministic.
            Files.write(Paths.get("output"), context.element().getBytes());
        }
    }
}
```
x??

---

#### Reshuffle Transform in Dataflow Pipelines
Reshuffle is a built-in transform used to ensure deterministic record processing, particularly when dealing with side effects. It ensures that a `DoFn` always receives the same input records during retries, preventing the generation of different results from the same inputs.
If you have a pipeline pattern as follows:
```java
.apply(Window.<...>into(FixedWindows.of(Duration.standardMinutes(1))))
.apply(GroupByKey.<...>.create())
.apply(new PrepareOutputData())
.apply(Reshuffle.<...>.of())
.apply(WriteToSideEffect());
```
:p How does the Reshuffle transform ensure deterministic processing in a Dataflow pipeline?
??x
The `Reshuffle` transform ensures that the `DoFn` (such as `PrepareOutputData`) always receives the same input records during retries. This is crucial because, without it, if the entire process were run after a failure, both `PrepareOutputData` and subsequent steps might produce different results due to state changes or other side effects.
```java
// Example of adding Reshuffle in between two transforms
pipeline
    .apply(Window.<...>into(FixedWindows.of(Duration.standardMinutes(1))))
    .apply(GroupByKey.<...>.create())
    .apply(new PrepareOutputData())
    .apply(Reshuffle.<...>.of()) // Ensures deterministic input for the next transform
    .apply(WriteToSideEffect());
```
x??

---

#### Use of GroupByKey with Reshuffle
`GroupByKey` is another way to ensure deterministic processing, but it may be replaced by a special annotation in future versions. Currently, `Reshuffle` can achieve stable input into a `DoFn`, ensuring that the same records are processed during retries.
:p How does adding `GroupByKey` and `Reshuffle` together help maintain deterministic processing?
??x
Adding `GroupByKey` before `Reshuffle` ensures that the same records are processed by a `DoFn` during retries. The combination of these transforms guarantees that the side-effect operations, such as writing to a database or file, will receive consistent input.
```java
// Example pipeline with GroupByKey and Reshuffle
pipeline
    .apply(Window.<...>into(FixedWindows.of(Duration.standardMinutes(1))))
    .apply(GroupByKey.<...>.create())
    .apply(new PrepareOutputData()) // May produce different results on retries
    .apply(Reshuffle.<...>.of()) // Ensures the same input for subsequent steps
    .apply(WriteToSideEffect()); // Needs to be idempotent or handle duplicates
```
x??

---

#### Example of Cloud Pub/Sub as a Source in Dataflow
Cloud Pub/Sub is designed for distributed use, enabling multiple publishers and subscribers. It ensures that records are delivered reliably but may be redelivered if not acknowledged.
:p How does the nature of Cloud Pub/Sub make it challenging to use directly with Dataflow pipelines?
??x
The challenge lies in ensuring deterministic processing when using Cloud Pub/Sub as a source. Since Pub/Sub delivers messages multiple times and acknowledges them based on subscription, it can lead to non-deterministic behavior if not handled properly.
To handle this, you might need to buffer the incoming data, process it deterministically (e.g., using `Reshuffle` or `GroupByKey`), and then write the side effects in a way that is idempotent. This ensures that even if messages are redelivered, they do not cause duplicate writes.
```java
// Example of handling Cloud Pub/Sub with Dataflow
pipeline
    .apply(PubsubIO.readStrings()
        .fromTopic("your-topic")
        .withSubscription("your-subscription"))
    .apply(Window.<...>into(FixedWindows.of(Duration.standardMinutes(1))))
    .apply(GroupByKey.<...>.create())
    .apply(new PrepareOutputData()) // Ensures deterministic processing
    .apply(Reshuffle.<...>.of())
    .apply(WriteToSideEffect()); // Needs to be idempotent
```
x??

---

#### Pub/Sub and Record Duplication Handling
Background context: In Apache Beam Dataflow, handling Pub/Sub messages involves dealing with potential duplicates due to retries. The system uses a message ID provided by Pub/Sub for deduplication but may still face issues if the publishing service introduces unique IDs during retries.
:p How does Dataflow handle duplicates in Pub/Sub messages?
??x
Dataflow handles duplicates in Pub/Sub messages by allowing users to provide their own record IDs as custom attributes. If the same ID is used upon retry, Dataflow can detect these duplicates and avoid processing them multiple times. This ensures that the pipeline operates correctly even when records are delivered more than once.
??? 
---

#### File Sink Example: Windowed File Writes
Background context: Beam’s file sinks can be used to continuously output records to files in a windowed manner. The `TextIO` transform is an example of how this works, allowing for the creation of multiple files per window interval.
:p What does the following snippet do?
```java
.apply(Window.<..> into(FixedWindows.of(Duration.standardMinutes(1))))
   .apply(TextIO.writeStrings().to(new MyNamePolicy()).withWindowedWrites());
```
??x
This snippet applies a `FixedWindows` transformation to group records by one-minute intervals and then writes the results to files. The `TextIO.writeStrings()` method is used with a custom `MyNamePolicy` to determine filenames, ensuring that each window gets written to a separate file.
??? 
---

#### Consistent Streaming File Sink Implementation
Background context: To ensure exactly-once processing for file sinks in Apache Beam, the system uses a combination of temporary and final files. This approach ensures that any non-idempotent operations are performed only once during the finalize step.
:p What is the role of `WriteTempFile` in the streaming file sink implementation?
??x
The `WriteTempFile` transform writes records to temporary files based on their shard and window information. It creates multiple temporary files, but only one will pass through to the finalize stage after all retries are completed. This ensures that each unique record is processed exactly once.
??? 
---

#### Finalize Stage in File Sink Implementation
Background context: After writing temporary files, a finalize transform atomically renames these files to their final locations. This step uses idempotent operations to ensure consistency and avoid duplicate writes.
:p What does the `Finalize` transform do in the file sink implementation?
??x
The `Finalize` transform ensures that each temporary file is renamed to its final location atomically. This operation is idempotent, meaning it can be safely run multiple times without affecting the outcome. Once a file is finalized, any further attempts at renaming will have no effect.
??? 
---

#### Google BigQuery Sink Example
Background context: Beam provides a BigQuery sink that leverages BigQuery's streaming insert API for low-latency data insertion. The streaming insert API allows tagging inserts with unique IDs to filter out duplicates.
:p How does the BigQuery sink handle exactly-once processing?
??x
The BigQuery sink handles exactly-once processing by using a unique ID per insert and leveraging BigQuery’s ability to filter out duplicate records. This ensures that each record is processed only once, maintaining consistency in data ingestion.
??? 
---

#### Duplicate Inserts Handling in BigQuery Sink
Background context: When using a BigQuery sink in Dataflow, it is crucial to handle duplicate inserts effectively. To ensure that only unique records are inserted into BigQuery, a unique identifier (ID) must be generated for each record before attempting an insert operation. This process involves two steps: generating statistically unique IDs and ensuring these IDs remain stable across retries.

:p What is the purpose of using `UUID.randomUUID()` in this context?
??x
The purpose of using `UUID.randomUUID()` is to generate a universally unique identifier (UUID) that can be used as a unique ID for each record. This ensures that each record processed by Dataflow gets a distinct and statistically unique identifier, which helps in filtering out duplicate records during BigQuery insert operations.

```java
String uniqueId = UUID.randomUUID().toString();
```
x??

---

#### Reshuffling for Stability Across Retries
Background context: After generating the unique identifiers for records, it is essential to reshuffle the data. This step ensures that the applied identifiers remain stable and do not change across different Dataflow retries. By doing so, any duplicate attempts to insert into BigQuery will use the same UUID.

:p What does the `Reshuffle` transform do in this scenario?
??x
The `Reshuffle` transform ensures that the data is re-partitioned randomly after the unique identifiers are generated. This randomization guarantees that even if a Dataflow job is retried, the UUIDs assigned to each record will remain the same, thereby maintaining consistency across retries.

```java
// Reshuffle the data so that the applied identifiers are stable and will not change.
.data.apply(Reshuffle.of());
```
x??

---

#### Inserting into BigQuery with Unique IDs
Background context: Once the unique IDs have been assigned to each record and the data has been reshuffled, the next step is to insert these records into BigQuery. The unique IDs are used to filter out duplicate inserts.

:p How does the `ParDo` transform handle inserting into BigQuery?
??x
The `ParDo` transform processes each element with a custom DoFn that extracts the record and its associated unique ID, then inserts it into BigQuery. This transformation ensures that only records with unique IDs are inserted, effectively filtering out duplicates.

```java
.apply(ParDo.of(new DoFn<RecordWithId, Void>() {
    @ProcessElement
    public void processElement(ProcessContext context) {
        insertIntoBigQuery(context.element().record(), context.element().id());
    }
}));
```
x??

---

#### Non-Idempotent and Idempotent Steps in Dataflow Pipeline
Background context: In the provided pseudocode, the pipeline is split into two steps: a non-idempotent step for generating unique identifiers and an idempotent step for inserting records into BigQuery. The non-idempotent step generates UUIDs, while the idempotent step ensures that inserts can be retried without causing duplicates.

:p What are the characteristics of a non-idempotent step in this Dataflow pipeline?
??x
A non-idempotent step in this Dataflow pipeline is responsible for generating unique identifiers. The key characteristic here is that it produces different results each time it is executed, even if the same input data is processed multiple times. This is achieved through the use of `UUID.randomUUID()`, which generates a new UUID every time.

```java
String uniqueId = UUID.randomUUID().toString();
```
x??

---

#### Idempotent Step for BigQuery Insertion
Background context: The idempotent step in the pipeline ensures that once records are inserted into BigQuery, subsequent attempts to insert them will not result in duplicate entries. This is achieved by leveraging the stable UUIDs generated during the non-idempotent step.

:p What makes a step idempotent in this Dataflow scenario?
??x
An idempotent step in this Dataflow scenario ensures that any number of retries or resubmissions do not change the outcome beyond the initial successful execution. In the provided code, the insertion into BigQuery is idempotent because it uses the same UUIDs generated during the non-idempotent step. If an insert attempt fails and is retried, it will still use the same UUID, ensuring that no duplicates are created.

```java
.insertIntoBigQuery(context.element().record(), context.element().id());
```
x??

