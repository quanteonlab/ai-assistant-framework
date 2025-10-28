# Flashcards: 2B005---Streaming-Systems_processed (Part 1)

**Starting Chapter:** Navigating This Book

---

---
#### Streaming 101
Background context: The first chapter introduces the basics of stream processing, establishing essential terminology and discussing the capabilities of streaming systems. It distinguishes between two important domains of time (processing time and event time) and looks at common data processing patterns.

:p What are the key topics covered in Chapter 1, "Streaming 101"?
??x
Chapter 1 covers the foundational concepts of stream processing, including establishing terminology, discussing the capabilities of streaming systems, distinguishing between processing time and event time, and examining common data processing patterns. The chapter sets up the fundamental understanding necessary for more advanced topics.

```java
// Example code snippet to illustrate processing time vs event time
public class TimeProcessingExample {
    public static void main(String[] args) {
        // ProcessingTimeSource simulates a source that uses processing time
        Timer timer = new Timer();
        
        // EventTimeSource simulates a source that uses event time
        TimestampEvent event = new TimestampEvent(System.currentTimeMillis());
    }
}
```
x??

---
#### What, Where, When, and How of Data Processing
Background context: This chapter delves into the core concepts of robust stream processing over out-of-order data. It analyzes these concepts within the context of a concrete running example and uses animated diagrams to highlight different dimensions of time.

:p What is the main focus of Chapter 2, "The What, Where, When, and How of Data Processing"?
??x
Chapter 2 focuses on providing detailed analysis of core concepts in robust stream processing over out-of-order data. It examines these concepts using a specific running example and animated diagrams to illustrate various aspects of time.

```java
// Example code snippet illustrating out-of-order processing with timestamps
public class OutOfOrderProcessingExample {
    public static void main(String[] args) {
        TimestampedEvent event1 = new TimestampedEvent(1L, "event1");
        TimestampedEvent event2 = new TimestampedEvent(3L, "event2");
        TimestampedEvent event3 = new TimestampedEvent(2L, "event3");

        // Process events in order
        processEvents(new ArrayList<>(Arrays.asList(event1, event2, event3)));
    }

    private static void processEvents(List<TimestampedEvent> events) {
        for (TimestampedEvent event : events) {
            System.out.println("Processing event at timestamp: " + event.getTimestamp());
        }
    }
}
```
x??

---
#### Watermarks
Background context: Chapter 3, "Watermarks," authored by Slava, provides a deep survey of temporal progress metrics. It explains how these are created and propagated through pipelines and concludes with detailed examinations of two real-world watermark implementations.

:p What does Chapter 3, "Watermarks" cover?
??x
Chapter 3 covers the creation and propagation of temporal progress metrics (watermarks) in stream processing systems. The chapter explores how watermarks are used to handle out-of-order data and ensure that all events within a certain time frame have been processed.

```java
// Example code snippet illustrating watermark generation
public class WatermarkExample {
    public static void main(String[] args) {
        long timestamp = 100L; // Event timestamp in milliseconds
        long maxTimestamp = 98L; // Maximum timestamp seen so far

        // Generate the next expected watermark
        long nextWatermark = generateNextWatermark(timestamp, maxTimestamp);
    }

    private static long generateNextWatermark(long currentTimestamp, long maxTimestamp) {
        return Math.max(maxTimestamp + 1000, System.currentTimeMillis());
    }
}
```
x??

---
#### Advanced Windowing
Background context: This chapter builds upon the concepts introduced in Chapter 2, diving into advanced windowing and triggering mechanisms. It covers topics such as processing-time windows, sessions, and continuation triggers.

:p What are the main topics covered in Chapter 4, "Advanced Windowing"?
??x
Chapter 4 delves into advanced windowing and triggering mechanisms used in stream processing. The chapter explores concepts like processing-time windows, session windows, and continuation triggers to handle complex data processing scenarios more effectively.

```java
// Example code snippet illustrating session window logic
public class SessionWindowExample {
    public static void main(String[] args) {
        long timestamp1 = 0L; // Event timestamp in milliseconds
        long timestamp2 = 1000L;
        long timestamp3 = 500L;

        List<TimestampedEvent> events = new ArrayList<>(Arrays.asList(
            new TimestampedEvent(timestamp1, "event1"),
            new TimestampedEvent(timestamp2, "event2"),
            new TimestampedEvent(timestamp3, "event3")));

        // Process session windows
        processSessionWindows(events);
    }

    private static void processSessionWindows(List<TimestampedEvent> events) {
        Set<Long> activeSessions = new HashSet<>();
        
        for (TimestampedEvent event : events) {
            long sessionTimeout = 1000L; // Session timeout in milliseconds

            if (event.getTimestamp() - (activeSessions.stream().min(Long::compare).orElse(0L)) > sessionTimeout) {
                activeSessions.clear();
            }
            
            System.out.println("Processing event with session: " + event.getTimestamp());
        }
    }
}
```
x??

---
#### Exactly-Once and Side Effects
Background context: Chapter 5, authored by Reuven, addresses the challenges of providing end-to-end exactly-once processing semantics. It examines three different approaches to implementing exactly-once processing using Apache Flink, Apache Spark, and Google Cloud Dataflow.

:p What is the main focus of Chapter 5, "Exactly-Once and Side Effects"?
??x
Chapter 5 focuses on the challenges involved in providing end-to-end exactly-once (or effectively-once) processing semantics. The chapter explores three different approaches to implementing exactly-once processing: Apache Flink, Apache Spark, and Google Cloud Dataflow.

```java
// Example code snippet illustrating a simple stateful transformation in Apache Beam using Java SDK
public class ExactlyOnceExample {
    public static void main(String[] args) {
        Pipeline p = Pipeline.create();
        
        PCollection<String> input = p.apply("ReadInput", TextIO.read().from("input.txt"));

        // Apply exactly-once processing logic
        PCollection<String> result = input.apply(ParDo.of(new DoFn<String, String>() {
            @ProcessElement
            public void processElement(@Element String element) throws IOException {
                // Exactly-once processing implementation
                System.out.println("Processing: " + element);
            }
        }));

        p.run().waitUntilFinish();
    }
}
```
x??

---
#### Streams and Tables
Background context: Part II of the book, starting with Chapter 6, "Streams and Tables," introduces the basic idea of streams and tables as a way to think about stream processing. This concept is popularized by members of the Apache Kafka community but has roots in database theory.

:p What is the main focus of Chapter 6, "Streams and Tables"?
??x
Chapter 6 focuses on introducing the basic idea of streams and tables as a conceptual framework for understanding stream processing. The chapter explores how this approach relates to both modern streaming systems and traditional databases, providing insights into how data can be processed and managed over time.

```java
// Example code snippet illustrating a simple transformation between streams and tables in Apache Beam
public class StreamsAndTablesExample {
    public static void main(String[] args) {
        Pipeline p = Pipeline.create();
        
        PCollection<KV<String, Integer>> streamInput = p.apply("ReadStream", TextIO.read().from("stream-input.txt"));

        // Transform the stream into a table-like structure
        PCollection<KV<String, Iterable<Integer>>> tableOutput = streamInput
            .apply(GroupByKey.by(KV::getKey))
            .apply(ParDo.of(new DoFn<KV<String, Iterable<Integer>>, KV<String, Integer>>() {
                @ProcessElement
                public void processElement(@Element KV<String, Iterable<Integer>> element) throws Exception {
                    int sum = 0;
                    for (Integer value : element.getValue()) {
                        sum += value;
                    }
                    
                    // Output the aggregated result as a stream record
                    output(element.getKey(), sum);
                }
            }));

        p.run().waitUntilFinish();
    }
}
```
x??

---

#### Time-varying Relations
Time-varying relations represent a critical concept in understanding streaming systems. They are essentially dynamic datasets that change over time, capturing temporal aspects of data. This idea is foundational to grasping how streaming processes work and contrasts with traditional batch processing methods.

:p What are time-varying relations and why are they important in the context of stream processing?
??x
Time-varying relations are a type of dataset that changes over time. They represent the dynamic nature of data streams, allowing for the analysis of data as it comes in, rather than processing static snapshots. This is crucial because streaming systems need to handle and process data continuously as new elements arrive.

For example, consider an application where you want to track user activity on a website. The dataset representing this activity changes every time a user logs in or out, visits different pages, etc. Time-varying relations enable the system to keep track of these ongoing events and perform computations based on them.
x??

---

#### Theory of Streams and Tables
The theory of streams and tables is pivotal for understanding how streaming systems operate relative to traditional batch processing. This theory provides a unified framework that encompasses both approaches, making it easier to transition between different data processing paradigms.

:p What is the theory of streams and tables?
??x
The theory of streams and tables offers a generalized view that integrates stream processing and batch processing into a single conceptual model. It aims to bridge the gap between these two paradigms by treating both as special cases within a broader framework, thereby facilitating a more unified approach to data processing.

This theory is foundational because it allows developers and researchers to better understand how streaming systems can leverage familiar tools from relational algebra (tables) while also handling time-varying nature of streams. For instance, a streaming join operation could be seen as an extension or variant of a traditional join, but adapted for the dynamic nature of data in streams.
x??

---

#### Persistent State in Streaming Pipelines
Persistent state is essential for managing and retaining information across iterations in streaming pipelines. It enables systems to maintain context and continuity, which is critical for tasks like aggregations, windowing, and maintaining historical views.

:p Why is persistent state important in streaming pipelines?
??x
Persistent state is crucial because it allows streaming pipelines to retain information across processing cycles, enabling functionalities such as aggregations, windowed computations, and maintaining historical data. Without persistent state, each element would be processed independently of its previous context, making tasks like tracking user sessions or calculating moving averages impossible.

For example, if you want to calculate the average value over a sliding window, you need to maintain a running sum of values and the count of elements within that window. Persistent state ensures that these counts are retained between processing iterations.
x??

---

#### Streaming SQL
Streaming SQL extends traditional relational algebra and SQL by incorporating streaming semantics into query languages. This extension is necessary because traditional SQL is designed for batch processing, which does not account for the continuous nature of data in streams.

:p How does Streaming SQL extend traditional SQL?
??x
Streaming SQL extends traditional SQL by introducing concepts like time windows, event-time semantics, and stateful operations that are essential for handling real-time data. These extensions allow SQL queries to operate on continuously arriving data while still maintaining the declarative power and ease of use familiar from batch processing.

For instance, a streaming SQL query might include a `WINDOW` clause to define how different events should be grouped or aggregated over time. This is in contrast to traditional SQL where such operations are not inherently supported without additional programming logic.

Example:
```sql
SELECT user_id, COUNT(*) OVER (PARTITION BY user_id ORDER BY event_time ROWS BETWEEN 10 PRECEDING AND CURRENT ROW) AS recent_activity
FROM events
```

This query calculates the number of recent activities for each user based on a window of the last 10 events.
x??

---

#### Streaming Joins
Streaming joins are complex operations that deal with joining datasets in real-time. They require careful handling of temporal validity windows to ensure accurate and consistent results, especially when dealing with out-of-order data.

:p What is the challenge with streaming joins?
??x
The primary challenge with streaming joins lies in their need to handle temporal validity windows and out-of-order events efficiently. Unlike batch joins where all data can be processed at once, streaming joins must account for incoming events that may arrive after related records have already been processed or are yet to come.

This necessitates maintaining state across multiple events and ensuring that the join results reflect the correct time context of the data. For example, in financial trading systems, it is crucial to correctly match buy and sell orders as they stream in, even if their timestamps do not strictly follow each other in a chronological order.
x??

---

#### Evolution of Large-Scale Data Processing
The evolution from MapReduce to modern streaming systems highlights key advancements that have made real-time data processing more efficient and scalable. This historical context is important for understanding the current state of distributed computing.

:p What does Chapter 10 cover?
??x
Chapter 10, "The Evolution of Large-Scale Data Processing," traces the lineage of data processing systems from their origins with MapReduce to the development of streaming architectures that support real-time analysis. The chapter examines significant contributions and transformations in this domain, providing insights into how modern distributed computing has been shaped.

Key points covered include:
- Early batch processing models
- Introduction and impact of MapReduce
- Emergence of stream processing frameworks like Apache Storm and Apache Flink
- Challenges faced by traditional systems and innovations that addressed them

Understanding these historical developments helps in appreciating the current landscape of data processing technologies.
x??

---

#### Online Index and Figure References
Background context explaining how figures are referenced online. Figures can be accessed via URLs like `http://www.streamingbook.net/fig/<FIGURE-NUMBER>`. These figures are LaTeX/Tikz drawings rendered to PDFs, then converted into animated GIFs.

:p What is the format for accessing specific figures in the book?
??x
The URL format to access a figure in the book is: `http://www.streamingbook.net/fig/<FIGURE-NUMBER>`. For example, Figure 2-5 would be at `http://www.streamingbook.net/fig/2-5`.

```java
// Example of how a URL might look for a figure
String url = "http://www.streamingbook.net/fig/2-5";
```
x??

---

#### Animated Figures Code and Source
Background context explaining the source code and its complexity. The book provides full source code on GitHub, but warns that it is extensive and poorly structured.

:p What does the code for animated figures consist of?
??x
The code for animated figures consists of approximately 14,000 lines of LaTeX/Tikz code. This code grew organically and was not intended to be read or used by others, making it difficult to navigate and understand.

```java
// Example repository URL
String githubUrl = "http://github.com/takidau/animations";
```
x??

---

#### Code Snippets and Implementations
Background context explaining the availability of code snippets for Beam Model concepts. The code is primarily provided as PTransform implementations with unit tests, aimed at understanding semantics.

:p Where can I find the implementation of Beam Model concepts?
??x
The implementation of Beam Model concepts can be found on GitHub at `http://github.com/takidau/streamingbook`. This includes PTransform implementations and accompanying unit tests for examples from chapters 2 and 4, as well as state and timers concepts in chapter 7.

```java
// Example repository URL
String beamCodeUrl = "http://github.com/takidau/streamingbook";
```
x??

---

#### Standalone Pipeline Implementation
Background context explaining the standalone pipeline implementation. It illustrates the difference between unit tests and real pipelines.

:p What is the purpose of `Example2_1.java`?
??x
`Example2_1.java` is a standalone version of Example 2-1's pipeline that can be run locally or using a distributed Beam runner. Its purpose is to illustrate the difference between running a unit test and executing a real pipeline.

```java
// Sample class name for the standalone implementation
public class Example2_1 {
    public static void main(String[] args) {
        // Code to execute the pipeline
    }
}
```
x??

---

#### State and Timers Implementation
Background context explaining the use of state and timers in Beam Model concepts. It provides an example of conversion attribution using these primitives.

:p What does `StateAndTimers.java` implement?
??x
`StateAndTimers.java` implements the conversion attribution example from Chapter 7, utilizing Beam’s state and timers primitives to manage state updates and timers for delayed processing.

```java
// Example class name for State and Timers implementation
public class StateAndTimers {
    public static void main(String[] args) {
        // Code using state and timers
    }
}
```
x??

---

#### Validity Windows Implementation
Background context explaining the temporal validity windows implementation. This is used to manage temporal windows in data processing.

:p What does `ValidityWindows.java` do?
??x
`ValidityWindows.java` provides a temporal validity windows implementation, which is crucial for managing how data is processed over time windows in streaming applications.

```java
// Example class name for Validity Windows implementation
public class ValidityWindows {
    public static void main(String[] args) {
        // Code to manage validity windows
    }
}
```
x??

---

#### Utilities Implementation
Background context explaining the utility methods provided. These are shared across different implementations and can be used for common tasks.

:p What is the purpose of `Utils.java`?
??x
`Utils.java` contains shared utility methods that can be reused across various pipeline implementations, providing a centralized location for commonly needed functions.

```java
// Example class name for Utilities implementation
public class Utils {
    public static void main(String[] args) {
        // Code to demonstrate usage of utility methods
    }
}
```
x??
---

