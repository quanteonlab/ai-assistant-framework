# Flashcards: 2B005---Streaming-Systems_processed (Part 28)

**Starting Chapter:** Cloud Dataflow

---

#### Unified Batch Plus Streaming Programming Model
Background context explaining the concept. This model was developed to provide a single, nearly seamless experience for both batch and streaming data processing, addressing the need for flexibility and efficiency in handling various use cases. The model is built on the commonalities between batch and streaming processing by treating them as minor variations of streams and tables.

:p What is the unified batch plus streaming programming model?
??x
The unified batch plus streaming programming model combines both batch and streaming data processing under a single framework, leveraging shared underlying mechanics while providing flexible configurations for different use cases. This approach simplifies the handling of unbounded, out-of-order datasets by offering mechanisms like windowing, watermarks, and custom accumulation modes.

```java
// Example: Applying transformations with unified model
PCollection<KV<String, Integer>> input = ...; // Input PCollection

input.apply("TransformWindow",
    Window.<KV<String, Integer>>into(SlidingWindows.of(Duration.standardMinutes(5)).every(Duration.standardSeconds(2)))
        .triggering(AfterWatermark.pastEndOfWindow())
        .discardingFiredPanes());

input.apply("Accumulate",
    GroupBy.key()
        .updating((key, value, accumulator) -> accumulator.add(value))
        .withoutDuplicates());
```
x??

---

#### Event-Time Windows
Background context explaining the concept. Event-time windows are crucial for processing out-of-order data by grouping events based on their event times rather than arrival times. This approach enables powerful analytic constructs like sessions and provides a way to handle late-arriving data.

:p What is an event-time window, and how does it differ from a processing time window?
??x
An event-time window groups elements based on the timestamp associated with each element (event-time). Unlike processing time windows, which group elements based on when they arrive in the pipeline, event-time windows are more suitable for applications where data might be delayed or out-of-order.

```java
// Example: Applying an unaligned event-time window
PCollection<KV<String, Integer>> input = ...; // Input PCollection

input.apply("SessionWindow",
    Window.into(Sessions.withGapDuration(Duration.standardMinutes(2))));
```
x??

---

#### Custom Windowing Support
Background context explaining the concept. Custom windowing support allows users to define their own window strategies based on specific business requirements, rather than relying on predefined window sizes.

:p What is custom windowing support, and why is it important?
??x
Custom windowing support enables the definition of custom window strategies tailored to specific use cases, providing greater flexibility in data processing. This feature is crucial for applications with unique timing or grouping requirements that don't fit standard window sizes.

```java
// Example: Defining a custom window strategy
PCollection<KV<String, Integer>> input = ...; // Input PCollection

input.apply("CustomWindow",
    Window.into(FixedWindows.of(Duration.standardMinutes(5)))
        .withAllowedLateness(Duration.ZERO)
        .triggering(AfterWatermark.pastEndOfWindow())
        .accumulatingFiredPanes());
```
x??

---

#### Flexible Triggering and Accumulation Modes
Background context explaining the concept. Flexible triggering and accumulation modes allow users to shape the flow of data through pipelines, tailoring the processing to meet specific correctness, latency, or cost requirements.

:p What are flexible triggering and accumulation modes?
??x
Flexible triggering and accumulation modes enable fine-grained control over how data flows through a pipeline. These features allow users to design processing logic that balances factors like correctness, latency, and cost according to their application's needs.

```java
// Example: Customizing triggering and accumulation
PCollection<KV<String, Integer>> input = ...; // Input PCollection

input.apply("CustomTrigger",
    Triggering.of(new Trigger() {
        @Override
        public PCollection<KV<String, Integer>> trigger(PTransform.Context context) {
            return input;
        }
    }));
```
x??

---

#### Watermarks for Reasoning About Input Completeness
Background context explaining the concept. Watermarks are used to reason about the completeness of input data, particularly in streaming applications where data might be delayed or missing.

:p What is a watermark, and how does it help with incomplete data?
??x
A watermark is a mechanism that represents the latest known timestamp beyond which no late data should arrive for the current window. Watermarks are crucial in handling late-arriving data, ensuring that processing only occurs when enough data has arrived to make meaningful decisions.

```java
// Example: Using watermarks for completeness reasoning
PCollection<KV<String, Integer>> input = ...; // Input PCollection

input.apply("AssignWatermark",
    Apply.<KV<String, Integer>, KV<String, Integer>>of(
        new ParDoFn() {
            @Override
            public void processElement(KV<String, Integer> element) {
                // Assign a watermark based on the timestamp of the input
                context.outputWithTimestamp(element.getKey(), element.getValue());
            }
        }))
    .apply("AssignWatermark",
        SetWatermark.of(Duration.ZERO));
```
x??

---

#### Logical Abstraction of Underlying Execution Environment
Background context explaining the concept. Logical abstraction allows users to work with a simplified, higher-level API that abstracts away the complexities of the underlying execution environment (batch, micro-batch, or streaming).

:p What is logical abstraction in the context of Cloud Dataflow?
??x
Logical abstraction provides a clean, high-level programming interface that hides the details of the underlying execution environment. This abstraction allows users to write code without worrying about whether they are processing batch data, micro-batch data, or streaming data.

```java
// Example: Logical abstraction in action
PCollection<String> input = ...; // Input PCollection

input.apply("Transform",
    ParDo.of(new DoFn<String, String>() {
        @ProcessElement
        public void processElement(ProcessContext c) {
            c.output(c.element() + " transformed");
        }
    }));
```
x??

#### Dataflow/Beam Programming Model
Background context: Flink adopted the Dataflow/Beam programming model, which enhanced its semantic capability and placed it at a competitive advantage in the streaming world. This model supports both batch and streaming data processing.

:p What is the significance of the Dataflow/Beam programming model for Flink?
??x
The adoption of the Dataflow/Beam programming model significantly improved Flink's ability to handle both batch and streaming data, making it more semantically capable compared to other systems at the time. This model allowed Flink to leverage a unified programming approach that simplified development and increased flexibility.

```java
// Example code using Beam in Java
Pipeline p = Pipeline.create(options);
p.apply("Read from source", TextIO.read().from("input.txt"))
   .apply("Process elements", ParDo.of(new DoFn<String, String>() {
       @Override
       public void processElement(@Input(required = true) String element,
                                  OutputReceiver<String> out) throws Exception {
           // Process the data and emit results
       }
   }))
   .apply("Write to sink", TextIO.write().to("output.txt"));
```
x??

---

#### Chandy-Lamport Snapshots
Background context: Flink introduced an efficient snapshotting implementation derived from research by Chandy and Lamport, which provided strong consistency guarantees necessary for correctness.

:p What is the primary purpose of using Chandy-Lamport snapshots in Flink?
??x
The primary purpose of using Chandy-Lamport snapshots in Flink is to ensure strong consistency and provide exactly-once semantics. These snapshots help in checkpointing the system state periodically, ensuring that data processing can be resumed from a known good state if failures occur.

```java
// Example code for snapshotting in Java
public class FlinkCheckpoint {
    private final StreamExecutionEnvironment env;
    
    public FlinkCheckpoint(StreamExecutionEnvironment env) {
        this.env = env;
    }
    
    public void setupSnapshot(SnapshotContext context) throws Exception {
        // Logic to save state before checkpointing
    }
}
```
x??

---

#### Barrier Propagation Mechanism
Background context: The barrier propagation mechanism in Flink acts as an alignment mechanism between distributed workers, ensuring that checkpoints are triggered when all data from upstream producers has been processed.

:p How does the barrier propagation mechanism work in Flink?
??x
The barrier propagation mechanism works by sending periodic barriers through the system. When a consumer receives a barrier on all its input channels (i.e., from all of its upstream producers), it can then checkpoint its current state for all active keys. This ensures that processing can be resumed accurately after any failure.

```java
// Pseudocode example of barrier propagation in Flink
public class DataProcessor {
    public void processElement(Data data) {
        // Process the data
        if (shouldSendBarrier()) {
            sendBarrier();
        }
    }

    private boolean shouldSendBarrier() {
        // Logic to determine when a barrier should be sent
        return true; // Simplified example
    }

    private void sendBarrier() {
        // Send the barrier through the network
    }
}
```
x??

---

#### Exactly-Once Semantics
Background context: Flink's implementation of exactly-once semantics allowed it to achieve greater accuracy in data processing compared to other systems, especially when paired with event-time processing.

:p What does "exactly-once" semantics mean for Flink?
??x
Exactly-once semantics means that each piece of data is processed only once, even if there are failures and recoveries. This ensures that the system can handle both duplicate and missing events correctly, providing reliable and consistent results.

```java
// Example code in Java to enable exactly-once processing
public class FlinkJob {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
        env.enableCheckpointing(5000); // Checkpoint every 5 seconds

        DataStream<String> source = env.addSource(new MySourceFunction());
        DataStream<String> processed = source.keyBy(key -> key)
                                             .process(new FlinkProcessor());
        
        processed.addSink(new FlinkSinkFunction());
    }
}
```
x??

---

#### Savepoints for Pipeline Restart
Background context: Savepoints in Flink allow the entire pipeline to be restarted from any point in the past, providing a feature that extends beyond the durability of Kafka’s replay mechanism.

:p What are savepoints and how do they work in Flink?
??x
Savepoints in Flink enable the entire streaming pipeline to be restarted from a specific checkpoint, allowing graceful evolution over time. This is achieved by periodically saving the state of the system at global checkpoints, which can then be used to resume processing even if parts of the pipeline have been modified or expanded.

```java
// Example code for savepoints in Flink
public class SavepointManager {
    public void createSavepoint(StreamExecutionEnvironment env) throws Exception {
        String savepointPath = "/path/to/savepoint";
        env.saveSnapshot(savepointPath);
    }
}
```
x??

---

#### Performance of Flink Compared to Storm
Background context: Jamie Grier's article "Extending the Yahoo. Streaming Benchmark" demonstrated that Flink could achieve greater accuracy and higher throughput compared to Storm, even when running at a much lower cost.

:p How did Flink outperform Storm in the Yahoo. Streaming Benchmark?
??x
Flink outperformed Storm by providing exactly-once semantics with better accuracy, achieving 7.5 times the throughput of Storm without exactly-once semantics. By tuning checkpointing frequencies and removing network bottlenecks, Flink could achieve almost 40 times the throughput of Storm.

```java
// Pseudocode example to compare performance between Flink and Storm
public class PerformanceBenchmark {
    public void runBenchmark() {
        // Code to run Flink job
        long flinkThroughput = measureFlinkThroughput();
        
        // Code to run Storm job
        long stormThroughput = measureStormThroughput();
        
        System.out.println("Flink Throughput: " + flinkThroughput);
        System.out.println("Storm Throughput: " + stormThroughput);
    }
    
    private long measureFlinkThroughput() {
        // Logic to measure Flink throughput
        return 1000; // Simplified example
    }

    private long measureStormThroughput() {
        // Logic to measure Storm throughput
        return 250; // Simplified example
    }
}
```
x??

---

#### Flink’s Rapid Rise to Stream Processing Juggernaut
Flink has become a dominant player in stream processing due to its adoption of existing ideas, innovations, and quick iteration cycles. It was the first to bring practical streaming SQL APIs to market for large-scale, distributed stream processing engines.

:p What are the three primary characteristics that contributed to Flink's rise as a stream processing engine?
??x
The three primary characteristics contributing to Flink’s success include:
1. **Incorporating Best Existing Ideas**: Being an open source adopter of the Dataflow/Beam Model.
2. **Bringing Own Innovations**: Introducing features like strong consistency via snapshots and savepoints, and streaming SQL.
3. **Quick Iteration and Repeatable Innovation**: Continuous improvements and frequent updates.

These factors combined with Flink’s open-source nature have consistently raised the bar in the streaming processing industry.
x??

---

#### Apache Beam Overview
Apache Beam is a programming model for both batch and stream processing that aims to act as a lingua franca for programmatic data processing. It includes components like unified programming models, SDKs, DSLs, and runners.

:p What are the key components of Apache Beam?
??x
Apache Beam consists of several key components:
1. **Unified Programming Model**: Inherited from Cloud Dataflow, this model is independent of any language implementations or runtime systems.
2. **SDKs (Software Development Kits)**: Implementing the programming model in idiomatic ways for languages like Java, Python, and Go.
3. **DSLs (Domain Specific Languages)**: Specialized interfaces built on top of SDKs to handle specific domains.
4. **Runners**: Execution engines that translate Beam pipelines into physical executions.

These components work together to provide a flexible and portable framework for data processing.
x??

---

#### Beam’s Core Vision
Beam positions itself as an abstraction layer, aiming to bring value through full cross-language portability with runners. It focuses on including only the best ideas from across the community and encourages innovation in both Beam and its runners.

:p How does Apache Beam position itself relative to its runners?
??x
Apache Beam positions itself as a high-level programming model that abstracts away the differences between various execution engines (runners). Its core vision is to be a lingua franca for expressing programmatic data processing pipelines, ensuring it never settles for being just an intersection or union of features found in its runners. Instead, it aims to incorporate only the best ideas from across the community.

This approach ensures that Beam can innovate while maintaining portability and value:
```java
// Example: Using a Java SDK with Flink runner
Pipeline p = Pipeline.create(options);
p.apply("ReadFromSource", Read.from(new MySource()))
   .apply("ProcessData", ParDo.of(new DoFn<byte[], String, String>() {
       @ProcessElement
       public void processElement(ProcessContext c) {
           // Processing logic here
       }
   }))
   .apply("WriteToSink", Write.to(new MySink()));
```

The key is to ensure that Beam never limits itself to the lowest common denominator but instead aims for innovation in both its model and the runners.
x??

---

#### Powerful and Modular I/O in Apache Beam
Beam introduces APIs like SplittableDoFn, which supports scalable and composable sources. This feature allows for dynamic work rebalancing and improved performance.

:p What is an example of a powerful and modular I/O connector in Apache Beam?
??x
An example of a powerful and modular I/O connector in Apache Beam is the `SplittableDoFn` API, which enables writing scalable and composable sources. This feature supports dynamic work rebalancing and other innovative functionalities.

Here’s an example of how to use SplittableDoFn:
```java
@Setup
public void setup() {
    // Setup logic here
}

@Teardown
public void teardown() {
    // Teardown logic here
}

@Override
public void processElement(ProcessContext c) throws Exception {
    PCollection<T> input = c.getInputs().get("input");
    PCollection<KV<Long, T>> splitInput = input.apply(Splitter.splitDoFn(input));

    for (KV<Long, T> record : splitInput) {
        // Process each split
    }
}
```

This example shows how `SplittableDoFn` can be used to handle large data sources by splitting them into manageable chunks.
x??

---

#### Innovation in Beam and Runners
Beam encourages innovation both within its own model and among runners. It aims to support new features that may not be initially available in all runners but will eventually become standard as their value is recognized.

:p How does Beam encourage innovation?
??x
Beam encourages innovation by allowing both the Beam framework and its runners to introduce new runtime features independently of each other. These innovations are expected to spread over time, ensuring that the entire ecosystem benefits from advancements.

For example, Flink’s state snapshotting mechanism (savepoints) is an innovative feature that has not been universally adopted yet but will likely be integrated into Beam as more runners recognize its value:
```java
// Example of saving a checkpoint in Flink
env.checkpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
env.getCheckpointConfig().enableExternalizedCheckpoints(ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);
```

This example illustrates how Beam and its runners can drive innovation, eventually leading to broader adoption of these features.
x??

---

