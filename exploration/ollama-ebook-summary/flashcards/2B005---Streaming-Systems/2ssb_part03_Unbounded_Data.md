# Flashcards: 2B005---Streaming-Systems_processed (Part 3)

**Starting Chapter:** Unbounded Data Streaming

---

#### Completeness Problem in Event-Time Windows
Background context explaining the concept. When dealing with unbounded data, especially using event-time windows, there's a challenge in determining when all relevant data for an event time X has been observed due to the lack of a predictable mapping between processing time and event time.
This issue arises because in unbounded datasets, new data can continuously arrive, making it difficult to ascertain that you've seen everything related to a particular event time. 
:p How does the completeness problem affect the use of event-time windows with unbounded data?
??x
The completeness problem means that without a clear mapping between processing time and event time, determining when all data relevant to an event time X has been processed is challenging. This is problematic because many data processing systems rely on assumptions about completeness to function correctly.
In practical terms, this can mean that a system might continue to process new incoming data for what it thinks is the same event period, leading to potential inaccuracies or delays in results.

```java
public class EventTimeWindowExample {
    public void handleEvent(Event e) {
        // Logic to handle events
        if (isEventTimeCompleted(e)) {
            // Process the window of events
            processEventsForTime(e.getTime());
        }
    }

    private boolean isEventTimeCompleted(Event e) {
        // Check if all relevant data for the event time has been seen.
        return false; // Placeholder logic, actual implementation depends on system design
    }
}
```
x??

---

#### Data Processing Patterns: Bounded vs. Unbounded

Background context explaining the concept. In bounded data processing, we have a finite dataset that is processed to extract structured and valuable information using batch or streaming engines. For unbounded data, the approach needs to accommodate continuous incoming data.
:p What distinguishes bounded from unbounded data processing?
??x
Bounded data processing involves working with a fixed amount of data, typically input once and then processed through various stages such as mapping, reducing, and aggregating. The result is usually a structured dataset that can be stored or further analyzed.

Unbounded data processing deals with streams of incoming data where the total volume is unknown and potentially infinite. This requires handling real-time updates, reprocessing old data, and dealing with uncertainties in the completeness of data.
x??

---

#### Fixed Windows for Unbounded Data

Background context explaining the concept. To process unbounded datasets using batch engines, a common approach is to break down the stream into fixed-size windows, treating each window as an independent bounded dataset that can be processed by the batch engine.

:p What is a typical method for processing unbounded data with batch engines?
??x
A typical method involves dividing the incoming data stream into fixed-sized windows and then running each of these windows through the batch engine. This approach ensures that the batch engine can handle the data in manageable chunks, treating each window as an independent, bounded dataset.
This method is known as "tumbling windows" because the windows do not overlap and slide a fixed amount at a time.

```java
public class FixedWindowProcessor {
    private int windowSize;

    public void processUnboundedData(Stream<Event> events) {
        // Create sliding windows of size 'windowSize'
        for (int i = 0; i < events.size(); i += windowSize) {
            List<Event> window = events.subList(i, Math.min(i + windowSize, events.size()));
            batchEngine.process(window); // Process the bounded data in a batch engine
        }
    }

    private boolean isWindowComplete(List<Event> window) {
        // Logic to check if the current window has all necessary data
        return true; // Placeholder logic
    }
}
```
x??

---

#### Time-Based Shuffle for Logs Processing
Background context explaining how logs can be processed using time-based shuffles. Events are written into directory and file hierarchies whose names encode the window they correspond to, making initial processing seem straightforward but with inherent challenges.

:p What is a time-based shuffle in the context of log processing?
??x
A time-based shuffle involves organizing events into appropriate windows based on their event times before processing them. This makes it appear initially that the data can be easily processed by placing each event in its corresponding window.
x??

---

#### Completeness Problem in Logs Processing
Background explaining the issue where some events might be delayed or not collected due to network partitions, global collection, and mobile device limitations.

:p What is a completeness problem in logs processing?
??x
A completeness problem arises when some events are delayed en route to logs, possibly due to network issues, the need for events to be transferred globally before processing, or delays from mobile devices. This means that you cannot always assume all relevant data will be available at once.
x??

---

#### Delayed Processing and Reprocessing in Logs Processing
Explanation of dealing with completeness problems by delaying processing until sure all events are collected or reprocessing the entire batch.

:p How do systems handle delayed events during logs processing?
??x
Systems often delay processing until they are confident that all relevant events have been collected. Alternatively, they may reprocess the entire batch for a given window whenever late data arrive.
x??

---

#### Sessions with Ad Hoc Fixed Windows and Batch Engines
Explanation of how sessions are typically defined and broken down when using batch engines.

:p How do ad hoc fixed windows affect session calculation?
??x
Sessions are periods of activity terminated by inactivity gaps. Using batch engines, these sessions might be split across batches, leading to inefficiencies. Increasing batch sizes reduces splits but increases latency. Alternatively, stitching logic can help, adding complexity.
x??

---

#### Streaming Systems for Unbounded Data
Explanation of why streaming systems are suitable for unbounded data and characteristics of such data.

:p Why are streaming systems better suited for unbounded data?
??x
Streaming systems are designed to handle unbounded data effectively. Unbounded data often have highly unordered event times, varying event-time skews, and other complexities that batch engines struggle with efficiently.
x??

---

#### Time-Agnostic Approaches
Explanation of the time-agnostic approach in dealing with unbounded data.

:p What is a time-agnostic approach to processing unbounded data?
??x
A time-agnostic approach involves designing systems where the timing of events does not significantly impact their processing. This can involve complex logic to handle unordered and skewed event times.
x??

---

#### Approximation Methods for Unbounded Data Processing
Explanation of approximation methods used in dealing with highly unordered and skewed event-time data.

:p What are some approximation methods for handling unbounded data?
??x
Approximation methods include techniques that trade off accuracy for performance, often involving sampling or heuristic approaches to manage the complexity of unbounded data.
x??

---

#### Windowing by Processing Time
Explanation of windowing strategies based on processing time rather than event time.

:p How does windowing by processing time differ from windowing by event time?
??x
Windowing by processing time involves organizing data around when they are processed, not when they occurred. This is useful in scenarios where the exact timing of events is less critical.
x??

---

#### Windowing by Event Time
Explanation of windowing strategies based on event time.

:p How does windowing by event time work?
??x
Windowing by event time involves organizing data around their occurrence times, allowing for more accurate analysis of temporal data. This approach requires handling issues like delayed events and data skew.
x??

---

#### Time-Agnostic Processing
Time-agnostic processing is used for cases where time is essentially irrelevant, meaning that all logic is data-driven. Since the processing depends solely on the arrival of new data, streaming engines do not need to support specific temporal operations other than basic data delivery.

Batch systems can also handle unbounded data sources by dividing them into smaller bounded datasets and processing each dataset independently.
:p What is time-agnostic processing?
??x
Time-agnostic processing refers to scenarios where the logic does not depend on the timestamp or sequence of events, but rather focuses on the data itself. In such cases, all relevant operations are driven by new incoming data.

Batch systems can process unbounded data sources by breaking them down into smaller, manageable datasets for independent processing.
x??

---

#### Filtering
Filtering is a basic form of time-agnostic processing where you examine each record as it arrives and drop or retain it based on specific criteria. This operation does not depend on the temporal order or event-time skew.

Here’s an example in pseudocode:
```java
for (each log entry in web traffic logs) {
    if (entry.domain == domainOfInterest) {
        // retain the record
    } else {
        drop the record;
    }
}
```
:p What is filtering in time-agnostic processing?
??x
Filtering involves examining each incoming data element and deciding whether to keep or discard it based on a specific criterion. This process does not depend on the order or timing of the events.

For example, when processing web traffic logs, you would check if an entry belongs to a specific domain; if so, retain it; otherwise, drop it.
x??

---

#### Inner Joins
An inner join in time-agnostic processing means joining two unbounded data sources based on matching elements from both. The logic only cares about the presence of corresponding elements, not their temporal sequence.

Here’s an example in pseudocode:
```java
Set<String> bufferA = new HashSet<>();
for (elementA : sourceA) {
    bufferA.add(elementA);
}

for (elementB : sourceB) {
    if (bufferA.contains(elementB)) {
        emit joined record;
    }
}
```
:p What is an inner join in time-agnostic processing?
??x
An inner join in time-agnostic processing involves joining two unbounded data sources by matching elements from both. The process buffers elements from one source and checks them against the other when corresponding elements are observed.

For instance, you buffer up elements from `sourceA` and then check each element of `sourceB` to see if it exists in the buffer; if so, emit a joined record.
x??

---

#### Approximation Algorithms
Approximation algorithms include methods like approximate Top-N, streaming k-means, etc., which provide solutions that are not exact but close enough for many practical purposes.

:p What is an approximation algorithm?
??x
An approximation algorithm provides solutions that are nearly correct but not necessarily precise. These algorithms are useful when exact answers are computationally expensive or infeasible.

For example, approximate Top-N can be used to find the top N elements from a stream of data without sorting all the elements.
x??

---

#### Approximation Algorithms

Approximation algorithms are designed to handle unbounded sources of input, providing outputs that closely resemble desired results. The main advantages include low overhead and suitability for processing vast or continuous data sets.

However, they have several drawbacks:
- Limited availability: Not many such algorithms exist.
- Complexity: Developing new ones can be challenging due to their intricate nature.
- Approximate nature: This limits their utility in precise scenarios.

These algorithms often incorporate some form of time-based decay and process elements as they arrive. The processing-time element is crucial for error bounds that depend on data order, which becomes irrelevant when dealing with unordered or varying event-time skew data.

:p What are the main disadvantages of approximation algorithms?
??x
The main disadvantages of approximation algorithms include limited availability, complexity in development, and their approximate nature limiting precise utility.
x??

---

#### Windowing Techniques

Windowing is a technique used to process unbounded or bounded data sources by dividing them into finite chunks for processing based on temporal boundaries. This approach helps manage large volumes of data and provides mechanisms for analyzing data over specific time intervals.

There are two primary types of windowing:
1. **Fixed Windows (Tumbling Windows):** These divide time into uniform segments that apply across the entire dataset, either uniformly or with phase shifts to different subsets.
2. **Sliding Windows:** These have a fixed length and period, leading to overlapping windows if the period is less than the length.

:p What are two main types of windowing techniques?
??x
Two main types of windowing techniques are Fixed Windows (Tumbling Windows) and Sliding Windows.
x??

---

#### Fixed Windows

Fixed or tumbling windows slice time into uniform segments that apply across the entire dataset. These can be applied uniformly (aligned windows) or with phase shifts for different subsets (unaligned windows).

:p What is an example of a fixed window in terms of its application?
??x
An example of a fixed window is dividing time into uniform segments and applying these segments consistently across the entire dataset, either without any shift or with a phase shift to different subsets.
x??

---

#### Sliding Windows

Sliding windows are a generalization of fixed windows. They are defined by a fixed length and period:
- If the period is less than the length, the windows overlap.
- If the period equals the length, you have fixed windows (tumbling).
- If the period is greater than the length, it results in sampling over subsets of data.

:p What happens if the period in sliding windows is greater than the length?
??x
If the period in sliding windows is greater than the length, it results in a weird sort of sampling window that looks only at subsets of the data over time.
x??

---

#### Example Code for Windowing

Here’s an example of how fixed and sliding windows might be implemented in pseudocode:

```java
// Pseudocode for Fixed Windows (Tumbling)
public class FixedWindow {
    private final int length; // Length of window
    
    public FixedWindow(int length) {
        this.length = length;
    }
    
    public void process(Event event) {
        // Process events within the current window
        if (event.getTime() >= start && event.getTime() < end) {
            processEvent(event);
        }
        
        // Update window boundaries for the next event
        updateWindowBoundaries();
    }
}

// Pseudocode for Sliding Windows
public class SlidingWindow extends FixedWindow {
    private final int period; // Period of overlap
    
    public SlidingWindow(int length, int period) {
        super(length);
        this.period = period;
    }
    
    public void process(Event event) {
        if (event.getTime() >= start && event.getTime() < end) {
            processEvent(event);
        }
        
        updateWindowBoundaries();
    }

    private void updateWindowBoundaries() {
        // Update the window boundaries based on the period
        this.start = this.end - length;
        this.end = this.start + length;
    }
}
```

:p How would you implement a sliding window in pseudocode?
??x
To implement a sliding window, you can extend a fixed window class and adjust its updateWindowBoundaries method to include an overlap period. Here’s an example:

```java
public class SlidingWindow extends FixedWindow {
    private final int period; // Period of overlap
    
    public SlidingWindow(int length, int period) {
        super(length);
        this.period = period;
    }
    
    public void process(Event event) {
        if (event.getTime() >= start && event.getTime() < end) {
            processEvent(event);
        }
        
        updateWindowBoundaries();
    }

    private void updateWindowBoundaries() {
        // Update the window boundaries based on the period
        this.start = this.end - length;
        this.end = this.start + length;
    }
}
```

This pseudocode shows how to manage overlapping windows in a sliding window implementation.
x??

---

#### Sliding Windows Overview
Background context explaining sliding windows, their typical alignment and use cases. Note that while they are drawn to suggest motion, all windows apply across the entire dataset.
:p What are sliding windows used for?
??x
Sliding windows are often used for analyzing data in a dynamic manner where elements are processed in a moving window fashion. They allow for real-time analysis by breaking down the data into manageable chunks that can be processed as they come in, providing insights at different points in time.
x??

---

#### Sessions in Windowing
Sessions refer to sequences of events terminated by gaps of inactivity greater than some timeout. They are used for analyzing user behavior over time, grouping temporally related events such as a sequence of videos watched consecutively.
:p What is the definition and use case for sessions?
??x
Sessions are defined as sequences of events that are grouped together until there is a gap in activity larger than a specified timeout period. This technique is commonly used to analyze user behavior over time, where each session represents a series of actions taken by a user during a single sitting or period.
x??

---

#### Processing-Time Windowing
Processing-time windowing buffers incoming data into windows based on the order they arrive until a certain amount of processing time has passed. For example, five-minute fixed windows buffer data for five minutes before sending it downstream.
:p What is processing-time windowing?
??x
Processing-time windowing involves buffering incoming data into windows according to the order in which the data arrives and processing them when a certain amount of processing time (e.g., 5 minutes) has elapsed. This method ensures that all data for a particular window are processed together, making it suitable for scenarios where real-time analysis is critical.
x??

---

#### Properties of Processing-Time Windowing
Processing-time windowing has several advantages: simplicity in implementation due to straightforward buffering logic and judging window completeness with perfect knowledge of input arrival times. However, it requires event-time ordered data for accurate results.
:p What are the key properties of processing-time windowing?
??x
Key properties include:
- **Simplicity**: Easy to implement as you just buffer incoming data until a window closes.
- **Judging Window Completeness**: The system can determine when a window is complete with perfect accuracy due to knowing all inputs have arrived.
- **Late Data Handling**: No need for dealing with "late" data since the system always has full knowledge of input arrival times.

However, it relies on event-time ordered data, which is often not available in real-world distributed systems.
x??

---

#### Disadvantage of Processing-Time Windowing
One major downside to processing-time windowing is that if the incoming data have associated event times, they must arrive in order for accurate results. This can be challenging with many real-world input sources.
:p What is a significant drawback of processing-time windowing?
??x
A significant drawback is that processing-time windows require the data to arrive in the correct event time order to accurately reflect when events happened. In practice, this is often difficult to achieve, especially with distributed and disparate input sources where event times might not be strictly ordered.
x??

---

#### Event-Time Windowing Basics
Background context: In scenarios where data arrives out of order due to network issues or other delays, processing the data strictly by event time rather than processing time is crucial. This method ensures that data is grouped and processed according to when it actually occurred, not just when it arrived.
:p What is event-time windowing used for?
??x
Event-time windowing is used in situations where data can arrive out of order due to network issues or other delays. It ensures that the processing is based on the actual occurrence times of events rather than the arrival times at a processing pipeline, providing more accurate and useful insights.
x??

---
#### Fixed-Size Event-Time Windows
Background context: Fixed-size event-time windows are used when we need to process data in finite chunks that reflect the times at which those events occurred. This method is particularly useful for scenarios where the exact timing of events is critical.
:p How do fixed-sized event-time windows work?
??x
Fixed-sized event-time windows work by dividing the stream of events into time-based segments, each representing a fixed duration (e.g., one hour). Each segment processes all the data that occurred within its specific timeframe. This ensures that even if data arrives out of order, it will still be grouped correctly based on when the events actually happened.
x??

---
#### Session Windows in Event-Time
Background context: Session windows are used to capture bursts of activity over a period where individual sessions may vary in duration but are generally short-lived and closely related. Event-time session windows allow for dynamic window sizes, making them ideal for tracking user behavior or other transient activities.
:p How do event-time session windows differ from fixed-size windows?
??x
Event-time session windows differ from fixed-size windows because they dynamically group data based on the actual occurrence times of events. This means that sessions can vary in length and are defined by periods of activity, rather than being constrained to a predefined duration. For example, a user’s session might end when there is no activity for a certain period, not necessarily at the start or end of an hour.
x??

---
#### Challenges with Event-Time Windows
Background context: While event-time windowing provides accurate processing based on actual event occurrence times, it comes with challenges such as increased buffering requirements and uncertainty in determining when a window is complete. These issues must be managed to ensure effective data processing.
:p What are the main drawbacks of using event-time windows?
??x
The main drawbacks of using event-time windows include:
1. **Buffering**: Data must often live longer than the actual window length, requiring more buffering and storage.
2. **Completeness Uncertainty**: There is no good way to know when all data for a given window has arrived, making it challenging to determine when results are ready.
x??

---
#### Buffering in Event-Time Windows
Background context: The extended lifetimes of event-time windows mean that more data must be buffered while waiting for the entire set. This can lead to increased storage needs but is often manageable due to the relatively low cost of persistent storage compared to other resources like CPU and network bandwidth.
:p How does buffering work in event-time windowing?
??x
Buffering in event-time windowing involves keeping more data available as windows extend their lifetimes. This ensures that all relevant data for a specific time period is processed together, even if it arrives late. While this increases storage requirements, modern data processing systems often have efficient persistent storage and caching layers to mitigate these costs.
x??

---
#### Completeness in Event-Time Windows
Background context: Determining when windows are complete can be challenging because there’s no reliable way to know if all relevant data for a window has been received. This uncertainty impacts the timing of result materialization, requiring heuristic methods or watermarks to estimate completion.
:p How is completeness determined in event-time windows?
??x
Completeness in event-time windows is often determined using heuristics like watermarks. A watermark represents an estimated point beyond which no new events will arrive for a particular window. When the current processing time exceeds the watermark, it can be assumed that the window is complete and results can be materialized.
x??

---

#### Streaming vs. Batch Processing
Streaming and batch processing are two distinct but related approaches to data processing, often categorized under "streaming." However, batch processing deals with fixed, bounded datasets that fit into memory, while streaming systems handle unbounded datasets where data keeps arriving over time.

:p What is the key difference between streaming and batch processing?
??x
Streaming processes unbounded datasets continuously, whereas batch processing handles fixed, bounded datasets. Streaming systems are designed to process incoming data as it arrives, making them suitable for real-time analysis, while batch systems typically operate on completed batches of data that fit into memory.

```java
public class BatchProcessingExample {
    public void processData(List<String> data) {
        // Process the entire dataset in one go
    }
}

public class StreamingProcessingExample {
    public void processEvent(Event event) {
        // Process each event as it arrives
    }
}
```
x??

---

#### Cardinality and Encoding of Large-Scale Datasets
Large-scale datasets can be categorized based on cardinality (bounded or unbounded) and encoding (tables vs. streams).

:p What are the two important dimensions when categorizing large-scale datasets?
??x
The two important dimensions for categorizing large-scale datasets are cardinality, which refers to whether the dataset is bounded or unbounded, and encoding, which differentiates between table-based and stream-based data.

```java
// Example of checking if a dataset is bounded or unbounded
public boolean isBoundedDataset(List<String> dataset) {
    return dataset.size() < 1000; // Threshold for small datasets
}

public boolean isUnboundedStream(Iterator<Event> events) {
    while (events.hasNext()) {
        events.next();
    }
    return true; // Stream-like behavior indicating unbounded data
}
```
x??

---

#### Correctness in Streaming Systems
Correctness is a critical aspect of streaming systems, especially when absolute accuracy is necessary. This involves materializing results for windows and refining them over time to handle window completeness.

:p What does correctness mean in the context of streaming systems?
??x
In streaming systems, correctness refers to ensuring that the processed data adheres to specific consistency guarantees, such as exactly-once processing, where each record is processed once and only once. This is crucial for applications like billing, where accuracy is paramount.

```java
public class CorrectnessGuaranteeExample {
    public void processRecords(List<Record> records) {
        // Ensure each record is processed exactly once using idempotent operations
    }
}
```
x??

---

#### Event Time vs. Processing Time
Event time and processing time are two crucial concepts in streaming systems, representing when an event actually occurred versus when the system processes it.

:p What is the difference between event time and processing time?
??x
Event time refers to the actual occurrence of events as they happen in the real world, while processing time represents when these events are processed by a system. Event time can be used for more accurate analysis, whereas processing time is simpler but less precise.

```java
public class TimeExample {
    public long getEventTime(Event event) {
        return event.getTimestamp();
    }

    public long getProcessingTime() {
        return System.currentTimeMillis();
    }
}
```
x??

---

#### Windowing in Streaming Systems
Windowing techniques are used to group events into time-based intervals for processing, either by processing time or event time. Common window types include fixed windows and sliding windows.

:p What is a window in the context of streaming systems?
??x
A window in streaming systems is a time-based interval used to group events together for processing. Windows can be fixed (with predefined start and end times) or sliding (where new events are added as they arrive, while older ones drop off).

```java
public class FixedWindowExample {
    public void processFixedWindow(List<Event> windowedEvents) {
        // Process the events within a fixed time interval
    }
}

public class SlidingWindowExample {
    public void processSlidingWindow(Stream<Event> stream) {
        // Process events in a sliding window, e.g., every 5 minutes
    }
}
```
x??

---

#### Microbatch Systems
Microbatch systems are a type of streaming system that uses repeated executions of batch processing engines to handle unbounded data. They combine the benefits of both batch and streaming approaches.

:p What is a microbatch system?
??x
A microbatch system processes unbounded data in small, bounded batches by repeatedly executing a batch processing engine. This approach allows for more deterministic and controllable processing compared to fully streamed systems but still maintains some real-time aspects.

```java
public class MicroBatchExample {
    public void processMicroBatch(Stream<Event> events) {
        // Process the current batch of events, then wait for the next one
    }
}
```
x??

---

#### Exactly-Once Processing Guarantee
Exactly-once processing is a consistency guarantee that ensures each record is processed once and only once. It's crucial in scenarios requiring high accuracy.

:p What does exactly-once processing mean?
??x
Exactly-once processing means ensuring that every record is processed exactly one time, without duplicates or omissions. This is important for applications like financial transactions where data integrity must be maintained.

```java
public class ExactlyOnceGuaranteeExample {
    public void processRecords(List<Record> records) {
        // Ensure each record is processed once using idempotent operations
    }
}
```
x??

---

#### Watermarks in Streaming Systems
Watermarks are used to represent the maximum known event time for a window. They help in dealing with late-arriving data and ensure that processing can proceed without getting stuck on incomplete windows.

:p What is a watermark in streaming systems?
??x
A watermark in streaming systems represents the latest known point at which all events within a window have arrived. It helps in managing state and ensuring that processing doesn't get stuck due to late-arriving data.

```java
public class WatermarkExample {
    public long getCurrentWatermark() {
        // Return the current watermark value, e.g., based on the latest known event time
    }
}
```
x??

---

#### Triggers
Background context explaining triggers. Triggers are mechanisms for declaring when the output for a window should be materialized relative to some external signal. They provide flexibility in choosing when outputs should be emitted, acting as flow control mechanisms and allowing you to declare when to take snapshots of results over time.

:p What is a trigger?
??x
A trigger is a mechanism that dictates when the output for a window should be materialized based on an external signal. It acts like a shutter release on a camera, enabling you to decide when to capture a snapshot of the results being computed.
x??

---

#### Watermarks
Background context explaining watermarks. A watermark is a notion of input completeness with respect to event times. The value of a watermark indicates that all inputs with an event time less than or equal to the watermark have been observed.

:p What is a watermark?
??x
A watermark represents a metric of progress when observing unbounded data sources. It states that "all input data with event times less than X have been observed." This helps in determining the completeness of input data relative to their event times.
x??

---

#### Accumulation
Background context explaining accumulation modes and their relevance in unbounded data processing. Different accumulation modes specify how multiple results for the same window relate to each other, impacting semantics and costs.

:p What is an accumulation mode?
??x
An accumulation mode defines the relationship between multiple results observed for the same window. It specifies whether results are independent, build upon previous results, or both.
x??

---
#### What Results Are Calculated?
Background context explaining that this question pertains to the types of transformations within a pipeline. Common examples include computing sums, building histograms, and training machine learning models.

:p What does "What results are calculated" mean in unbounded data processing?
??x
This question refers to the types of transformations performed on the data within the pipeline. It involves operations like computing sums, creating histograms, or training machine learning models.
x??

---
#### Where in Event Time Are Results Calculated?
Background context explaining that this question is answered by the use of event-time windowing. Common examples include fixed and sliding windows, as well as more complex types.

:p "Where in event time are results calculated" refers to what exactly?
??x
This question relates to where and when in the timeline (event time) the data processing occurs. It involves using event-time windowing techniques such as fixed, sliding, or session-based windows.
x??

---
#### When in Processing Time Are Results Materialized?
Background context explaining that this is answered by triggers and watermarks. This can involve repeated updates or a single output per window.

:p "When in processing time are results materialized" refers to what exactly?
??x
This question concerns the timing of result outputs during processing, which is determined by the use of triggers and optionally watermarks. It involves deciding when to emit results based on the processing timeline.
x??

---
#### How Do Refinements of Results Relate?
Background context explaining that this pertains to how subsequent results relate to previous ones within a window. Common accumulation modes include discarding, accumulating, and accumulating and retracting.

:p "How do refinements of results relate" refers to what exactly?
??x
This question deals with the relationship between subsequent results in relation to previous ones within the same window. It involves understanding how new data influences or refines previously calculated results through different accumulation modes.
x??

---

