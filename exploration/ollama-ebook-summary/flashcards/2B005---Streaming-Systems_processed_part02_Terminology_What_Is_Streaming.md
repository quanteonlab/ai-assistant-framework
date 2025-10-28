# Flashcards: 2B005---Streaming-Systems_processed (Part 2)

**Starting Chapter:** Terminology What Is Streaming

---

#### Terminology: Streaming vs. Bounded/Unbounded Data

Background context explaining the concept of streaming and data cardinality. The term "streaming" is often misused, leading to misunderstandings about its true capabilities.

:p What does the author define as a "Streaming system"?

??x
The author defines a "Streaming system" as a type of data processing engine that is designed with infinite datasets in mind. This precise definition isolates the term from colloquial uses that might imply limitations such as approximate or speculative results, despite well-designed streaming systems being capable of producing correct and repeatable results.
x??

---

#### Terminology: Bounded vs. Unbounded Data

Explanation on the two main types of data cardinality.

:p What are bounded and unbounded datasets according to the author?

??x
According to the author, bounded datasets are finite in size, while unbounded datasets are infinite (at least theoretically). The unbounded nature of infinite datasets imposes additional burdens on data processing frameworks that consume them.
x??

---

#### Terminology: Cardinality

Explanation on cardinality and its importance.

:p What is cardinality according to the text?

??x
Cardinality of a dataset dictates its size, with the most salient aspect being whether it is finite or infinite. The author prefers to use these two terms:
- Bounded data: A type of dataset that is finite in size.
- Unbounded data: A type of dataset that is infinite (at least theoretically).
x??

---

#### Terminology: Constitution

Introduction on the concept and its relevance.

:p What does "constitution" refer to according to the text?

??x
The constitution of a dataset dictates its physical manifestation, defining how one can interact with it. The author mentions two primary constitutions in Chapter 6 but provides this brief introduction.
x??

---

#### Terminology: Infinite vs. Finite Datasets

Explanation on the implications of unbounded data.

:p How does the unbounded nature of infinite datasets affect data processing frameworks?

??x
The unbounded nature of infinite datasets imposes additional burdens on data processing frameworks that consume them. These systems must handle potentially endless streams of data, which can complicate data management and resource allocation.
x??

---

#### Terminology: Stream Processing vs. Batch Processing

Introduction to the difference between stream and batch processing.

:p What is the key distinction between streaming and batch processing according to the text?

??x
Streaming involves unbounded data processing, often with low latency requirements, while batch processing deals with finite datasets that are processed in batches. Streaming systems need to handle infinite or very large datasets, whereas batch systems typically deal with smaller, more defined sets of data.
x??

---

#### Terminology: Time Domains

Introduction to the two primary domains of time in data processing.

:p What are the two primary time domains relevant in data processing?

??x
The two primary time domains relevant in data processing are:
- Batch time: Where historical or static datasets are processed according to fixed, known intervals.
- Streaming time: A continuous and potentially unbounded stream of data arriving at irregular intervals.
x??

---

#### Terminology: Latency vs. Consistency

Explanation on the trade-offs between latency and consistency.

:p How do streaming systems balance latency and consistency?

??x
Streaming systems often aim for low-latency processing, which can come at the cost of slightly less consistent results compared to batch processing. However, they provide more timely insights into data changes, making them suitable for real-time applications.
x??

---

#### Terminology: Approximate vs. Speculative Results

Explanation on the limitations and capabilities of streaming systems.

:p What kind of results do well-designed streaming systems produce?

??x
Well-designed streaming systems can produce correct, consistent, repeatable results just as any existing batch engine. The term "streaming" is used to describe a specific type of data processing system designed for infinite datasets.
x??

---

#### Terminology: Modern Data Consumer Needs

Explanation on the capabilities needed in modern data processing.

:p What mindset should data processing system builders adopt according to the text?

??x
Data processing system builders need to adopt a frame of mind that addresses the needs of modern data consumers, which include ever-more timely insights into data and the ability to handle massive, unbounded datasets.
x??

---

#### Terminology: Streaming Execution

Explanation on the historical context of streaming execution.

:p How has "streaming" historically been used in processing?

??x
Historically, "streaming" was often described by how it was accomplished via low-latency, approximate, or speculative results. This colloquial use can mislead about the true capabilities of well-designed streaming systems.
x??

---

---
#### Streaming vs. Tables
Background context: The text discusses the relationship between streams and tables, emphasizing that SQL systems traditionally deal with tables while MapReduce lineage data processing systems handle streams. It also highlights the importance of understanding time-varying relations in these contexts.

:p What is the difference between a holistic view (tables) and an element-by-element view (streams)?
??x
The key distinction lies in how data is processed: 
- **Tables**: Represent a complete snapshot or historical dataset at a specific point in time, providing a holistic view of the data.
- **Streams**: Offer an element-by-element view showing how the dataset evolves over time. They are more suited for real-time processing and dynamic updates.

In terms of implementation:
```java
// Example: Processing a table (snapshot)
public class TableProcessor {
    public void processTable() {
        // Code to read from and write to a complete dataset snapshot
    }
}

// Example: Processing a stream (dynamic view)
public class StreamProcessor {
    public void processStream() {
        // Code to handle data as it arrives, making decisions or updates in real-time
    }
}
```
x??

---
#### Lambda Architecture
Background context: The text explains the Lambda Architecture, which combines batch and streaming systems to provide both low-latency (inaccurate) results from a stream and eventually correct results from a batch system.

:p What is the Lambda Architecture?
??x
The Lambda Architecture is a design pattern that involves running two pipelines in tandem:
- **Batch Pipeline**: Provides accurate but potentially slower results through a traditional batch processing approach.
- **Streaming Pipeline**: Offers low-latency, speculative or approximate results using a stream processing engine.

The aim is to leverage the strengths of both: real-time responsiveness from streaming and accuracy from batch. However, maintaining such a setup can be complex due to the need for managing two separate pipelines and merging their outputs.

Example of how the Lambda Architecture might be implemented:
```java
public class LambdaArchitect {
    private BatchPipeline batchPipeline;
    private StreamPipeline streamPipeline;

    public void runLambdaArchitecture() {
        // Run both pipelines in parallel and merge results at a later stage
        this.batchPipeline.run();
        this.streamPipeline.run();
        // Logic to merge and reconcile outputs from batch and stream pipelines
    }
}
```
x??

---
#### Kappa Architecture
Background context: Proposed by Jay Kreps, the Kappa Architecture suggests running a single pipeline using well-designed systems that can handle both bounded and unbounded data.

:p What is the Kappa Architecture?
??x
The Kappa Architecture advocates for a unified approach where a single system (ideally designed to be robust) handles both batch processing and stream processing. This avoids the complexity of maintaining dual pipelines as seen in the Lambda Architecture.

Example code illustrating a Kappa Architecture approach:
```java
public class KappaProcessor {
    public void processUnboundedData() {
        // Code logic that can handle both streaming and batch inputs effectively
    }
}
```
x??

---
#### Efficiency Differences Between Batch and Streaming
Background context: The text discusses the efficiency differences between batch and stream processing systems, attributing these differences largely to design choices in handling data bundles and shuffle transports.

:p What are the main reasons for efficiency differences between batch and streaming systems?
??x
The primary reasons for efficiency differences include:
- **Increased Bundling**: Batch systems often bundle data more efficiently, optimizing storage and processing.
- **Sophisticated Optimizations**: Modern batch systems use advanced techniques to achieve high throughput with fewer resources.

To bridge this gap, the text suggests that clever optimizations found in batch systems could be applied to streaming systems for better efficiency. An example of such an approach might involve:
```java
public class HybridProcessor {
    public void optimizeForEfficiency() {
        // Code that incorporates both batch and stream processing techniques
        // For instance, using efficient data structures or parallel processing strategies
    }
}
```
x??

---

#### Event Time vs Processing Time
In data processing, especially for unbounded and unordered data, it is crucial to understand the difference between event time and processing time. Event time represents when events actually occurred, while processing time refers to when they are observed by the system.

The key takeaway here is that in an ideal world, these two times would be equal, but due to various real-world factors like network issues, software limitations, and data characteristics, this is rarely the case. A typical scenario involves a lag or skew between event time and processing time.

:p Explain the difference between event time and processing time.
??x
Event time is the actual occurrence of events as they happen in the real world. Processing time refers to when these events are observed by the system. The skew between these two times can be significant due to various factors such as network delays, software contention, or data distribution.

For example:
- A user logs into a system at 10:00 AM (event time).
- The system receives this log in at 10:30 AM (processing time), showing a 30-minute lag.
??x
---

#### Time-Domain Mapping
The mapping between event time and processing time is visualized on an x-y axis where the x-axis represents event-time completeness, meaning the total amount of data that has been observed up to a given point in event time. The y-axis represents normal clock time as observed by the system.

In real-world systems, the actual progress often looks like the red line in Figure 1-1. This line shows how processing time and event time can be out of sync due to various factors.

:p Describe the x-y axis mapping in a data processing context.
??x
The x-axis represents event-time completeness, showing all events with an occurrence before X have been observed by the system. The y-axis represents the progress of processing time, which is measured as normal clock time from when the processing starts. For instance, if at 10:30 AM (y-axis), the system has seen all events up to 10:00 AM (x-axis), this point would be plotted.

For example:
- At 9:00 AM, event-time completeness is 85%.
- By 10:30 AM, processing time might have completed only 60% of the data due to delays or processing issues.

The red line in Figure 1-1 illustrates this mapping. The black dashed line with a slope of 1 represents an ideal scenario where event and processing times are synchronized.
??x
---

#### Lag/Skew Between Event Time and Processing Time
Lag refers to the vertical distance between the ideal and the actual processing-time line, indicating delay in processing events. Skew is the horizontal distance from the ideal line, showing how far behind in event time the pipeline currently is.

These skews can vary over time due to dynamic factors like network congestion or data throughput variability. Importantly, lag/skew cannot be ignored if you need correct analysis based on actual event times.

:p Define and explain lag and skew in data processing.
??x
Lag refers to the delay observed between when events occur and when they are processed by the system. It is represented as the vertical distance between the ideal line (event time = processing time) and the red line showing reality. Skew, on the other hand, is the horizontal distance from the ideal line, representing how far behind in event time the current state of the pipeline is.

For example:
- At 10:30 AM, if events occurring at 10:00 AM are not processed until 10:45 AM, there’s a 15-minute lag.
- If by 10:30 AM, only events up to 9:45 AM have been processed, the skew is 15 minutes.

Lag and skew can be identical if both delays are due to processing issues, or they may differ based on factors like data distribution or network delay.
??x
---

#### Windowing for Unbounded Data Processing
Windowing is a technique used in unbounded data processing to manage and analyze large volumes of data by dividing them into manageable chunks. This allows systems to process data in finite pieces along temporal boundaries, making it easier to handle delays and skews between event time and processing time.

Typically, windowing involves defining windows based on the current processing time rather than the actual event times. However, if correctness is needed (i.e., analyzing data as they occur), these windows must be defined using event time instead of processing time.

:p What is windowing in unbounded data processing?
??x
Windowing is a technique to handle large and potentially infinite datasets by dividing them into manageable temporal chunks. These windows can be defined in different ways, but for correctness, it's crucial to define them based on actual event times rather than the system’s current processing time.

For example:
- If you want to analyze user behavior over a specific hour using data up to 10:00 AM, you would create an event-time window from 9:00 AM to 10:00 AM.
- However, if the system is lagging and has only processed events up to 9:30 AM, using processing time (e.g., 9:00 AM - 10:00 AM) would misplace data.

In summary, windowing helps in managing unbounded datasets but requires careful consideration of event times for accurate analysis.
??x
---

