# Flashcards: 2B005---Streaming-Systems_processed (Part 6)

**Starting Chapter:** How Accumulation

---

#### Discarding Accumulation Mode
Discarding accumulation mode is where every time a pane is materialized, any stored state is discarded. This means each successive pane is independent from any that came before it. This approach can be useful when the downstream consumer performs its own aggregation.

:p In discarding mode, how do subsequent panes relate to each other?
??x
In discarding mode, each pane operates independently and does not retain any state from previous panes. Each new pane starts with a clean slate.
```java
// Pseudo-code for discarding accumulation mode in Apache Beam:
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
        .triggering(
            AfterWatermark()
                .withEarlyFirings(AlignedDelay(ONE_MINUTE))
                .withLateFirings(AtCount(1)))
        .discardingFiredPanes())
    .apply(Sum.integersPerKey());
```
x??

---

#### Accumulating Accumulation Mode
Accumulating accumulation mode retains any stored state when a pane is materialized, and future inputs are accumulated into the existing state. This means each successive pane builds upon the previous ones.

:p In accumulating mode, how do subsequent panes build on each other?
??x
In accumulating mode, the state from previous panes is retained, so new values can be added to the existing aggregated result without starting over. Each new pane updates the accumulated state.
```java
// Pseudo-code for accumulating accumulation mode in Apache Beam:
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
        .triggering(
            AfterWatermark()
                .withEarlyFirings(AlignedDelay(ONE_MINUTE))
                .withLateFirings(AtCount(1)))
        .accumulatingAndRetractingFiredPanes())
    .apply(Sum.integersPerKey());
```
x??

---

#### Accumulating and Retracting Mode
Accumulating and retracting mode combines the benefits of both accumulating and discarding modes. It retains state for accumulation but also produces retractions when producing a new pane, which helps in cleaning up old values.

:p How does accumulating and retracting mode handle previous states?
??x
In accumulating and retracting mode, each new pane not only accumulates new data into the existing state but also retracts (removes) the previous state. This is useful for scenarios where data needs to be regrouped by a different dimension or when dynamic windows require replacing multiple old windows.
```java
// Pseudo-code for accumulating and retracting mode in Apache Beam:
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
        .triggering(
            AfterWatermark()
                .withEarlyFirings(AlignedDelay(ONE_MINUTE))
                .withLateFirings(AtCount(1)))
        .accumulatingAndRetractingFiredPanes())
    .apply(Sum.integersPerKey());
```
x??

---

#### Example of Discarding Accumulation Mode
Example 2-9 illustrates the discarding mode in action, where each pane only incorporates values from that specific pane and does not retain state from previous panes.

:p How does the output differ between accumulating and discarding modes?
??x
In discarding mode, the final value observed in a pane does not include data from previous panes. However, summing all independent panes will give the correct total. This contrasts with accumulating mode where each new pane builds upon the existing state, leading to potential double-counting if summed separately.
```java
// Output example for discarding and accumulating modes:
Table 2-1. Comparing accumulation modes using the second window from Figure 2-11

| Pane      | Discarding Mode | Accumulating Mode | Accumulating & Retracting Mode |
|-----------|-----------------|------------------|--------------------------------|
| Pane 1:   | inputs=[3]       | 3                | 3                              |
| Pane 2:   | inputs=[8, 1]    | 9 (3 + 8 + 1)    | 12 (3 + 8 + 1 - 3)              |
| Value of final normal pane | 9              | 12               | 12                             |
| Sum of all panes   | 12             | 15               | 12                             |
```
x??

---

---

#### Event Time vs Processing Time
Background context: Understanding event time versus processing time is crucial for robust stream processing. Event time refers to when events occur, while processing time refers to when they are observed by your data processing system.

:p What is the difference between event time and processing time?
??x
Event time is about when an event occurs in reality, whereas processing time is about when it's processed by your system. This distinction is important for ensuring correct results even if there's a delay or out-of-order events.
x??

---

#### Windowing
Background context: Windowing is used to manage unbounded data by slicing it along temporal boundaries. In the Beam Model, windowing is specifically defined within event time.

:p What does windowing do in stream processing?
??x
Windowing slices continuous data streams into manageable segments (windows) for processing. This allows for batch-like operations on a streaming system.
x??

---

#### Triggers
Background context: Triggers define when materialization of output makes sense for your particular use case, providing flexibility in how results are calculated and when they're produced.

:p What is the purpose of triggers in stream processing?
??x
Triggers specify when to emit or update results based on the current state of data within a window. They help control the timing of result materialization.
x??

---

#### Watermarks
Background context: Watermarks provide a notion of progress in event time, enabling reasoning about completeness and missing data in out-of-order processing systems.

:p What is the role of watermarks in stream processing?
??x
Watermarks track the latest known event timestamp, allowing the system to determine when enough data has arrived for a window. This helps manage late-arriving data.
x??

---

#### Accumulation Modes
Background context: Accumulation modes describe how results are refined as windows evolve and materialize multiple times.

:p What are the different accumulation modes in stream processing?
??x
The accumulation modes include discarding, accumulating, and accumulating and retracting. Each mode offers a trade-off between correctness, latency, and cost.
x??

---

#### Discarding Mode
Background context: In discarding mode, previous results are discarded when new data arrives.

:p How does the discarding mode work in stream processing?
??x
In discarding mode, each window is processed independently. As new windows come in, old results are discarded, making it cheaper but potentially less correct.
x??

---

#### Accumulating Mode
Background context: In accumulating mode, previous results accumulate to form a running total.

:p How does the accumulating mode work in stream processing?
??x
In accumulating mode, results from previous windows accumulate with new data. This provides more accurate results at the cost of increased storage and computation.
x??

---

#### Accumulating and Retracting Mode
Background context: In accumulating and retracting mode, both accumulation and retraction of results are performed as windows evolve.

:p How does the accumulating and retracting mode work in stream processing?
??x
In this mode, results from previous windows accumulate with new data, but older results are also retracted when they expire. This balances between correctness and cost.
x??

---

#### Fixed Windows in Event Time
Background context: Fixed windowing slices data into time-based intervals for processing.

:p What is fixed windowing in event time?
??x
Fixed windowing divides the stream into non-overlapping segments based on a predefined duration, ensuring each window has a distinct start and end.
x??

---

#### Heuristic Watermark Triggers
Background context: Heuristic watermarks are used to approximate the progress of data arrival.

:p How do heuristic watermark triggers work in stream processing?
??x
Heuristic watermark triggers estimate when enough data is likely available by looking at patterns or heuristics, rather than waiting for a perfect watermark.
x??

---

#### Early/On-Time/Late Trigger Mode
Background context: Trigger modes determine the timing of result materialization based on various conditions.

:p What are the different trigger modes in early/on-time/late?
??x
The trigger modes include early (materializing results before all data is known), on-time (materializing when expected), and late (waiting for more data).
x??

---

#### Summary of Concepts
Background context: This summary recaps the major concepts and questions addressed throughout the chapter, providing a framework for understanding stream processing.

:p What are the key concepts covered in this chapter?
??x
The key concepts include event time vs. processing time, windowing, triggers, watermarks, accumulation modes, and fixed windows in event time.
x??

---

#### Summary of Questions
Background context: The questions used to frame the exploration help in understanding the stream processing model.

:p What are the four questions framing our exploration?
??x
The questions are: what results are calculated (transformations), where in event time are results calculated (windowing), when in processing time are results materialized (triggers plus watermarks), and how do refinements of results relate (accumulation).
x??

---

#### Flexibility in Stream Processing Models
Background context: The Beam Model offers various output variations with minimal code changes, allowing for different trade-offs.

:p How can the Beam Model be used to achieve different outputs?
??x
The Beam Model uses different accumulation modes and trigger configurations to produce varying results, balancing correctness, latency, and cost.
x??

---

#### Watermarks in Detail
Background context: This section delves deeper into watermarks, providing more details on their implementation and usage.

:p What is the role of a heuristic watermark in stream processing?
??x
A heuristic watermark approximates when enough data has arrived for a window to be processed. It's less precise but can achieve faster results by dropping late-arriving data.
x??

---

#### Trade-offs in Accumulation Modes
Background context: Understanding the trade-offs between different accumulation modes is crucial for choosing the right approach.

:p What are the trade-offs of accumulating and retracting mode?
??x
Accumulating and retracting mode balances correctness with cost by both accumulating new data and retraction old data as windows evolve.
x??

---

#### Discarding Mode vs. Accumulating Mode
Background context: Comparing discarding and accumulating modes highlights their differences in handling windowed results.

:p How does the discarding mode differ from the accumulating mode?
??x
Discarding mode discards previous results, making it cheaper but potentially less correct. Accumulating mode retains old results, providing more accurate but costly results.
x??

---

#### Future Directions
Background context: The chapter concludes by hinting at future directions in stream processing and Beam Model.

:p What future directions are mentioned for the Beam Model?
??x
Future iterations of the Beam Model will simplify triggers, introduce sink triggers for accumulating modes, and support SQL with new features like beam 3.0.
x??
---

#### Watermark Introduction
Background context: In event-time processing, we need to determine when it is safe to consider an event-time window closed. This involves understanding where in event time data are processed and when results are materialized.

:p What is a watermark used for in stream processing?
??x
A watermark is used to indicate the progress of the pipeline relative to its unbounded input, allowing us to determine when it is safe to close an event-time window without missing any more data. It helps in ensuring correctness by tracking the oldest unprocessed message's timestamp.
x??

---
#### Naive Approaches Explained
Background context: Two naive approaches for solving the event-time windowing problem are considered but found wanting due to their lack of robustness.

:p Why is simply basing event-time windows on processing time not a good strategy?
??x
Basing event-time windows solely on processing time is problematic because processing and event times rarely coincide. Delays in data processing or transmission can lead to incorrect window assignments, making this approach unreliable for ensuring correctness.
x??

---
#### Rate-Based Approach Limitations
Background context: Considering the rate of messages processed by the pipeline also has limitations as it does not address completeness or guarantee that all messages for a specific time interval have been seen.

:p Why is relying on processing rate insufficient to determine when an event-time window can be closed?
??x
Reliance on processing rate alone is insufficient because rates can vary widely based on input variability, resource availability, and other factors. It does not provide information about whether all messages for a specific time interval have been processed, which is crucial for correctness.
x??

---
#### Event Timestamp Assumption
Background context: Each message in the stream has an associated logical event timestamp that represents when the event occurred.

:p What is the significance of each message having a logical event timestamp?
??x
The significance of each message having a logical event timestamp is that it allows us to track and manage progress in processing unbounded data. This helps in determining the watermark, which indicates where we are relative to the oldest unprocessed event.
x??

---
#### Watermark Calculation
Background context: The watermark is defined as the leftmost edge of the "in-flight" distribution of messages, representing the oldest unprocessed event timestamp.

:p How do you calculate a watermark from message timestamps?
??x
To calculate a watermark, we examine the distribution of event timestamps for active in-flight messages. The watermark is set at the leftmost edge of this distribution, which corresponds to the oldest unprocessed event timestamp.
```java
// Pseudocode to calculate watermark
public int getWatermark(List<Integer> timestamps) {
    if (timestamps.isEmpty()) return -1; // No messages ingested yet
    return Collections.min(timestamps); // Find and return the minimum timestamp
}
```
x??

---
#### Watermark Propagation
Background context: Watermarks propagate through a data processing pipeline, affecting how output timestamps are determined.

:p How do watermarks affect the processing of an event-time window?
??x
Watermarks affect the processing of an event-time window by ensuring that only messages with timestamps greater than or equal to the watermark can be considered for inclusion in the current window. This helps maintain correctness by excluding any outdated events.
x??

---
#### Output Timestamps and Watermarks
Background context: The presence of watermarks influences when results are materialized, ensuring completeness.

:p How do watermarks influence output timestamps?
??x
Watermarks influence output timestamps by effectively marking the boundary between processed and unprocessed data. When a watermark is advanced, it signifies that we have seen all messages up to that point in time, thus allowing us to close the window and produce results accordingly.
```java
// Pseudocode for materializing results based on watermarks
public void materializeResults(List<Integer> timestamps) {
    int latestWatermark = getLatestWatermark(timestamps);
    for (int timestamp : timestamps) {
        if (timestamp >= latestWatermark) {
            processAndOutput(timestamp); // Process and output valid messages
        }
    }
}
```
x??

---
#### Robust Progress Measure
Background context: Watermarks provide a more robust measure of progress than rate-based metrics.

:p Why is the watermark a better measure of progress compared to processing rates?
??x
Watermarks are a better measure of progress because they directly track the oldest unprocessed event, ensuring that we have seen all relevant data up to that point. In contrast, processing rates can vary widely and do not guarantee completeness or correctness.
x??

---

