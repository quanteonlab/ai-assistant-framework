# Flashcards: 2B005---Streaming-Systems_processed (Part 26)

**Starting Chapter:** Hadoop

---

#### MapReduce Overview
MapReduce introduced a simple and scalable programming model for processing large datasets. It abstracts away complex distributed system details, allowing developers to focus on writing map and reduce functions.

:p What does MapReduce provide that makes it suitable for massive-scale data processing?
??x
MapReduce provides a straightforward API where users can define mapping and reducing functions. This abstraction allows the underlying system to handle the complexities of distributing tasks across multiple nodes and managing distributed storage, making it easier to process big data.

```java
// Pseudocode for MapReduce job
public void runJob() {
    // Step 1: Map phase - Process input records into key-value pairs.
    List<Output> outputs = mapFunction(inputRecords);

    // Step 2: Shuffle and sort - Combine similar keys from the map phase, then sort them.
    Map<Key, List<Value>> groupedOutputs = shuffleAndSort(outputs);

    // Step 3: Reduce phase - Process the grouped key-value pairs to produce final results.
    Map<Key, Value> reducedResults = reduceFunction(groupedOutputs);
}
```
x??

---

#### Hadoop Introduction
In 2005, Doug Cutting and Mike Cafarella developed Hadoop based on the MapReduce model. Their goal was to create a distributed web crawler using an existing version of Google’s distributed filesystem (later named HDFS).

:p What triggered the development of Hadoop?
??x
Hadoop was developed in 2005 when Doug Cutting and Mike Cafarella needed a distributed system for their Nutch webcrawler project. They had already created HDFS, and adding MapReduce functionality seemed like a natural progression.

```java
// Pseudocode for basic Hadoop setup
public void setupHadoop() {
    // Step 1: Initialize HDFS file system.
    FileSystem fs = FileSystem.get(conf);
    
    // Step 2: Create or open a file in HDFS.
    FSDataOutputStream out = fs.create(new Path("/example.out"));
    
    // Step 3: Write data to the file.
    out.writeBytes("Hello, Hadoop!");
    
    // Step 4: Close the output stream and finalize operations.
    out.close();
}
```
x??

---

#### Open Sourcing Hadoop
Hadoop was open sourced in 2006 by Yahoo!, giving it a significant boost. The source code for both HDFS and MapReduce was made available under an Apache license.

:p Why did Yahoo! decide to open source Hadoop?
??x
Yahoo! decided to open source Hadoop because they believed in the power of community-driven development and wanted to accelerate innovation by making Hadoop’s source code publicly accessible. This move helped establish Hadoop as a robust, widely-used framework for big data processing.

```java
// Pseudocode for opening Hadoop
public void openHadoop() {
    // Step 1: Set up the Hadoop configuration.
    Configuration conf = new Configuration();
    
    // Step 2: Initialize the FileSystem and start using it.
    FileSystem fs = FileSystem.get(conf);
    
    // Step 3: Perform operations on files in the HDFS.
    Path[] paths = fs.listFiles(new Path("/"));
    for (Path path : paths) {
        System.out.println(path.getName());
    }
}
```
x??

---

#### Growth of the Hadoop Ecosystem
With Yahoo!’s support, Hadoop attracted significant attention and contributed to the growth of an ecosystem of open-source data processing tools. The community-driven development model further propelled its adoption.

:p How did the ecosystem surrounding Hadoop grow?
??x
The open sourcing of Hadoop led to a rapid expansion of its ecosystem as developers from around the world contributed additional tools, libraries, and enhancements. This collaborative approach fostered innovation and made Hadoop an integral part of many big data solutions.

```java
// Pseudocode for adding a tool to Hadoop
public void addTool() {
    // Step 1: Define a custom MapReduce job.
    JobConf conf = new JobConf();
    
    // Step 2: Add the path to your custom Hadoop tool jar file.
    FileInputFormat.addInputPath(conf, new Path("/input"));
    FileOutputFormat.setOutputPath(conf, new Path("/output"));
    
    // Step 3: Run the job with the added tool.
    JobClient.runJob(conf);
}
```
x??

---

#### Open Source Ecosystem Around Hadoop
Background context: The open source ecosystem around Hadoop played a crucial role in shaping the industry. By creating an open community, engineers could improve and extend ideas from early papers like GFS and MapReduce, leading to the development of numerous useful tools such as Pig, Hive, HBase, Crunch, etc.
:p What was the significance of the open source ecosystem around Hadoop?
??x
The open source ecosystem allowed for a diverse range of innovations and improvements by leveraging collective effort. This openness fostered a thriving community where engineers could collaborate on enhancing existing frameworks and creating new tools. Tools like Pig and Hive provided higher-level abstractions over MapReduce, making data processing more accessible to developers.
x??

---

#### Flume Java: Introduction
Background context: Flume was developed at Google Seattle in 2007 to address shortcomings of MapReduce by providing a composable API for describing data processing pipelines. It aimed to optimize pipelines and reduce the need for bespoke orchestration systems, thereby making development more efficient.
:p What motivated the creation of Flume Java?
??x
Flume was created to overcome the limitations of MapReduce’s rigid structure, which included the need for multiple independent jobs to process a pipeline, leading to inefficiencies and manual optimizations that increased complexity. By providing a high-level API, Flume aimed to simplify and optimize data processing pipelines.
x??

---

#### Composable High-Level Pipelines in Flume
Background context: Flume introduced PCollection and PTransform concepts similar to Beam for defining high-level pipelines. These pipelines could be optimized automatically by the framework, reducing manual intervention required for performance tuning.
:p What did Flume provide to simplify pipeline definitions?
??x
Flume provided a composable API based on PCollection and PTransform that allowed engineers to define data processing pipelines more intuitively. This approach enabled automatic optimization of these pipelines, reducing the need for manual optimizations that often resulted in obfuscated code.
x??

---

#### Optimization Techniques in Flume
Background context: Flume offered various automatic optimizations like fusion and combiner lifting to enhance pipeline efficiency. Fusion combined logically independent stages into a single physical operation, while combiner lifting performed partial aggregation before shuffling data, reducing network costs.
:p What is the concept of fusion in Flume?
??x
Fusion in Flume combines two logically independent stages (either sequentially or in parallel) into a single physical operation to eliminate serialization/deserialization and network costs. This optimization can significantly improve performance by reducing the overhead associated with inter-stage data transfer.
```java
// Pseudocode for fusion example
PCollection<KV<String, Integer>> input = ...;
PCollection<KV<String, Integer>> filtered = input.filter(...);
PCollection<KV<String, Integer>> enriched = input.map(...);
// Fusion combines filter and map operations into a single physical operation
```
x??

---

#### Combiner Lifting in Flume
Background context: Combiner lifting is an optimization that applies partial aggregation on the sender side of group-by-key operations before completing aggregation on the consumer side, reducing network shuffling costs for hot keys.
:p What is combiner lifting in Flume?
??x
Combiner lifting in Flume involves performing partial aggregation at the source of a group-by-key operation to reduce the amount of data shuffled over the network. This optimization is particularly useful for hot keys where it can significantly decrease network traffic and improve overall performance.
```java
// Pseudocode for combiner lifting example
PCollection<KV<String, Integer>> input = ...;
PCollection<KV<String, Integer>> aggregated = input.apply(GroupByKey.create())
                                      .apply(Combine.globally(SumIntegerCombiner.create()));
```
x??

---

#### Dynamic Work Rebalancing in Flume
Background context: The dynamic work rebalancing feature, or liquid sharding, automatically redistributes extra work from straggler tasks to idle workers, improving resource utilization and job completion times.
:p What is the concept of dynamic work rebalancing (liquid sharding) in Flume?
??x
Dynamic work rebalancing, colloquially known as liquid sharding, adjusts the distribution of workload over time by moving tasks from slower or straggling workers to idle ones. This ensures that resources are optimally utilized and job completion times are minimized.
```java
// Pseudocode for dynamic work rebalancing example
ExecutorService executor = Executors.newFixedThreadPool(numWorkers);
for (Task task : tasks) {
    Future future = executor.submit(task);
    // Periodically check and reassign tasks to balance load dynamically
}
```
x??

---

#### Streaming Semantics in Flume
Background context: As part of its evolution, Flume was extended to support streaming semantics, enabling execution on stream processing systems like MillWheel. This extension allowed for continuous data processing alongside batch processing.
:p What did Flume extend to support?
??x
Flume was extended to support streaming semantics, allowing pipelines to be executed not just in a batch mode but also on stream processing systems like MillWheel. This capability enabled both continuous and batch processing within the same framework, enhancing its versatility for diverse use cases.
```java
// Pseudocode for streaming pipeline example
PCollection<KV<String, Integer>> input = ...;
PCollection<KV<String, Integer>> processed = input.apply(MapElements.via(new Function<String, Integer>() {
    public Integer apply(String element) {
        // Process the stream data in real-time
        return element.length();
    }
}));
```
x??

---
#### Flume Introduction
Flume was one of the early systems to introduce high-level pipelines for automatic optimization, allowing developers to create complex data pipelines without manual orchestration or optimization. The primary benefit was keeping the code logical and clear while enabling much larger and more intricate pipeline designs.

:p What introduced high-level pipelines that allowed for automatic optimization in Flume?
??x
Flume introduced high-level pipelines through a framework that automatically optimized clearly written, logical pipelines. This made it possible to create much larger and more complex data processing workflows without needing extensive manual intervention.
x??

---
#### Apache Storm Overview
Apache Storm was developed by Nathan Marz at BackType as a solution for real-time data processing. It aimed to simplify the development of distributed systems by handling most of the dirty work under the hood, allowing developers to focus on data processing logic rather than system management.

:p What problem did Apache Storm solve?
??x
Apache Storm solved the problem of complex and error-prone distributed systems used in real-time data processing. By abstracting away many low-level details, Storm allowed developers to write simpler, more maintainable code for processing streams of data.
x??

---
#### Storm's Approach to Latency vs Consistency Trade-off
Storm offered at-most once or at-least once semantics with per-record processing and no integrated persistent state to achieve lower latency. However, this approach came at the cost of strong consistency guarantees.

:p What trade-off did Apache Storm make for achieving lower latency?
??x
Apache Storm traded strong consistency guarantees for lower latency by using at-most once or at-least once semantics and avoiding integrated persistent state. This meant that while results could be delivered faster, they might not always be exactly correct.
x??

---
#### Lambda Architecture Overview
The Lambda Architecture was a solution introduced to address the limitations of Storm's weak consistency model. It combined real-time data processing from Storm with batch processing using Hadoop to provide both low-latency and eventually consistent results.

:p What is the Lambda Architecture?
??x
The Lambda Architecture was designed to handle both real-time and historical data processing requirements by combining a stream-processing layer (using Storm) for low-latency, inexact results with a batch-processing layer (using Hadoop) for high-latency, exact results. These layers were then merged to provide an eventually consistent view of the processed data.
x??

---
#### Heron Introduction
Twitter developed Heron as an internal project to address performance and maintainability issues faced by Storm while maintaining API compatibility. It was later open-sourced under its own independent foundation.

:p What is Heron?
??x
Heron is a stream processing system developed internally by Twitter to improve upon the limitations of Storm, particularly focusing on better performance and maintainability. After its development, it was open-sourced and operates independently from Apache.
x??

---

