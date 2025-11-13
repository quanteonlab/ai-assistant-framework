# Flashcards: 2B005---Streaming-Systems_processed (Part 5)

**Starting Chapter:** When Watermarks

---

#### Watermarks Overview
Background context explaining watermarks. Watermarks are temporal notions of input completeness in the event-time domain, used to measure progress and completeness relative to the event times of records being processed. They represent a function $F(P) \rightarrow E $, where $ P $ is processing time and $ E$ is event time.
:p What are watermarks?
??x
Watermarks help track the completeness of input data in terms of event time, allowing for proper handling of late data in stream processing systems. They serve as a function that maps processing time to event time, indicating the latest event time for which all inputs have been observed or will be seen.
```java
// Pseudo-code example:
class Watermark {
    public int F(P) { // Function mapping processing time to event time
        return E; // Event time up to which all data with event times less than E are considered complete
    }
}
```
x??

---

#### Perfect vs. Heuristic Watermarks
Explanation of the difference between perfect and heuristic watermarks, including their practical applications.
:p What types of watermarks exist?
??x
Perfect watermarks provide a strict guarantee that no more data with event times less than $E$ will be seen again, ideal for scenarios where complete input knowledge is available. Heuristic watermarks use available information to estimate completeness, making them useful when perfect knowledge is impractical or too expensive to calculate.
```java
// Pseudo-code example:
class PerfectWatermark {
    public int F(P) { // Function mapping processing time to event time with strict guarantees
        return E; 
    }
}

class HeuristicWatermark {
    public int F(P) { // Function mapping processing time to event time based on heuristics
        return estimatedE;
    }
}
```
x??

---

#### Watermarks and Completeness Triggers
Explanation of how watermarks enable the use of completeness triggers in stream processing.
:p How do watermarks support completeness triggers?
??x
Watermarks facilitate the implementation of completeness triggers by providing a measure of input completeness based on event time. This allows the system to determine when all expected data for a given window have been observed, ensuring that late data does not delay the completion of results.

```java
// Pseudo-code example:
PCollection<KV<Team, Integer>> totals = input 
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .triggering(AfterWatermark())
    .apply(Sum.integersPerKey());
```
x??

---

#### Example Pipeline with Watermarks
Explanation of an example pipeline that uses watermarks.
:p What does the provided example pipeline demonstrate?
??x
The example demonstrates a pipeline where `PCollection` elements are processed within fixed windows and use an after-watermark trigger to ensure completeness before producing results. This setup allows for accurate computation even when dealing with late arriving data.

```java
// Pseudo-code example:
PCollection<KV<Team, Integer>> totals = input 
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .triggering(AfterWatermark())
    .apply(Sum.integersPerKey());
```
x??

---

#### Watermarks and Different Implementations
Explanation of how the same dataset can use different watermark implementations.
:p How can watermarks vary in implementation?
??x
Watermarks can differ based on the specific algorithm used to calculate them. For scenarios with perfect input knowledge, a perfect watermark can be implemented, providing strict guarantees. However, for cases lacking this information or where it's computationally expensive, heuristic watermarks are more practical and provide an educated estimate.

```java
// Pseudo-code example:
class PerfectWatermark {
    public int F(P) { 
        return E; // Strict guarantee of completeness 
    }
}

class HeuristicWatermark {
    public int F(P) { 
        return estimatedE; // Estimate based on available data 
    }
}
```
x??

---

#### Watermarks and Their Importance in Event-Time Processing

Background context explaining the concept. In event-time processing, watermarks are used to capture the completeness of input data. They help in determining when an output for a window can be considered complete based on the time events could have arrived.

:p What is the role of watermarks in event-time processing?
??x
Watermarks serve as a marker indicating that all input events with timestamps before or equal to the watermark value are expected to have been received. This helps in determining when a window's output can be considered complete, thus enabling proper handling of late data and reasoning about missing data.

```java
// Pseudocode for updating watermarks
public void updateWatermark(long currentEventTime) {
    if (currentEventTime > lastKnownEarliestUnprocessedTimestamp) {
        lastKnownEarliestUnprocessedTimestamp = currentEventTime;
    }
}
```
x??

---

#### Perfect vs. Heuristic Watermarks

Background context explaining the concept. The text contrasts two types of watermarks: perfect and heuristic. A perfect watermark precisely marks when all events before a certain time have been received, whereas a heuristic watermark makes an educated guess about completeness based on observed patterns.

:p What is the difference between a perfect watermark and a heuristic watermark?
??x
A **perfect watermark** accurately captures the event-time completeness of the pipeline as it progresses. It ensures that once a watermark reaches a window's end, all possible input events for that window have been received.

In contrast, a **heuristic watermark** is an approximation based on observed patterns in the data stream. While it can provide early results and reduce latency, it may incorrectly trigger output before all relevant data has arrived, leading to potential correctness issues.

```java
// Example of heuristic watermark logic
public class HeuristicWatermark {
    private long lastProcessedTimestamp;
    
    public void update(long currentEventTime) {
        if (currentEventTime > lastProcessedTimestamp + threshold) {
            // Advance the watermark based on observed patterns
            lastProcessedTimestamp = currentEventTime - threshold;
        }
    }
}
```
x??

---

#### Latency Issues with Watermarks

Background context explaining the concept. The text discusses two main latency issues related to watermarks: they can be too slow or too fast, impacting the performance and correctness of event-time processing systems.

:p What are the latency issues associated with using watermarks in event-time processing?
??x
Watermarks can suffer from being **too slow** when known unprocessed data delays their advancement. This results in delayed output as the watermark waits for all possible input events before advancing.

Conversely, they can be **too fast**, causing late data to be incorrectly processed by triggering output prematurely. This issue is more common with heuristic watermarks, which make assumptions about completeness that may not always hold true.

```java
// Example of delayed watermark logic (slow)
public void processEvent(long eventTime) {
    if (eventTime > lastWatermark + knownDelayThreshold) {
        lastWatermark = eventTime;
        // Advance the watermark only when all events are expected to have arrived
    }
}
```

x??

---

#### Combining Repeated Updates and Watermarks

Background context explaining the concept. The text suggests that combining repeated update triggers for low latency with watermarks for completeness can provide a balanced approach to event-time processing.

:p How can we combine repeated updates and watermarks for better performance?
??x
To combine the benefits of both repeated updates and watermarks, one could implement a strategy where:

1. **Repeated Updates:** Continuously process new data as they arrive, providing early results.
2. **Watermark Advancement:** Use watermarks to ensure that output is only materialized when all relevant input events are likely complete.

This hybrid approach can provide the best of both worlds: low latency from repeated updates and correctness from watermark-based completeness checks.

```java
// Pseudocode for combining repeated updates and watermarks
public class HybridTrigger {
    private long lastWatermark;
    
    public void processEvent(long eventTime) {
        // Repeated update logic
        if (eventTime > lastProcessedTimestamp) {
            // Process the event immediately
            lastProcessedTimestamp = eventTime;
            emitResults();
        }
        
        // Watermark logic
        if (eventTime > lastWatermark + knownDelayThreshold) {
            lastWatermark = eventTime;
            // Ensure all events with timestamps <= watermark have been processed
            if (allEventsProcessed(lastWatermark)) {
                emitFinalResults();
            }
        }
    }
}
```
x??

---

#### Early, On-Time, and Late Panes via Early/On-Time/Late Trigger
Background context: Beam provides an extension called early/on-time/late trigger which combines repeated update triggering with completeness/watermark triggering. This mechanism partitions window panes into three categories based on when they are materialized.
- **Early Panes**: Result from a repeated update trigger that periodically fires up until the watermark passes the end of the window. These panes contain speculative results but allow observing the evolution over time as new data arrive.
- **On-Time Pane**: Result from the completeness/watermark trigger firing after the watermark passes the end of the window, providing an assertion that input for this window is now complete and it's safe to reason about missing data.
- **Late Panes**: Result from another repeated update trigger that fires periodically due to late data arriving after the watermark has passed the end of the window. Late panes compensate for watermarks being too fast.

:p What are early, on-time, and late panes in the context of Beam's early/on-time/late trigger?
??x
Early, on-time, and late panes are categorized based on how they are generated by triggers during a window's processing:
- **Early Panes**: These result from repeated updates triggered periodically until the watermark passes the end of the window. They contain speculative results but help in observing the evolving state over time.
- **On-Time Pane**: This is produced when the completeness/watermark trigger fires after the watermark passes the end of the window, indicating that the input for this window is now complete and it's safe to reason about missing data.
- **Late Panes**: These are generated by a repeated update trigger firing due to late data arriving after the watermark has passed the end of the window. They help in compensating for watermarks being too fast.

Code example:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
        .triggering(AfterWatermark()
            .withEarlyFirings(AlignedDelay(ONE_MINUTE))
            .withLateFirings(AfterCount(1))))
    .apply(Sum.integersPerKey());
```
x??

---

#### Aligned Delay and Per-Record Triggers for Early Firings
Background context: For the early firings, Beam supports two types of triggers:
- **Aligned Delay Trigger**: Periodically fires up until the watermark passes the end of the window.
- **Per-Record Trigger**: Fires based on individual records.

:p How are early firings handled in this scenario?
??x
Early firings use an aligned delay trigger to periodically fire updates until the watermark passes the end of the window. This helps in providing speculative results while allowing observation of evolving data over time, addressing the issue where watermarks might be too slow.
The code for setting up these triggers looks like:
```java
.apply(Window.into(FixedWindows.of(TWO_MINUTES))
    .triggering(AfterWatermark()
        .withEarlyFirings(AlignedDelay(ONE_MINUTE))))
```
x??

---

#### Per-Record Trigger for Late Firings
Background context: For the late firings, a per-record trigger is used to handle late data arriving after the watermark has passed. This ensures that there is minimal latency and unnecessary delays in processing late updates.
:p How are late firings handled in this scenario?
??x
Late firings use a per-record trigger to process any late data arriving after the watermark has passed. This approach minimizes latency and avoids unnecessary delays, ensuring efficient handling of late data.
The code for setting up these triggers looks like:
```java
.apply(Window.into(FixedWindows.of(TWO_MINUTES))
    .triggering(AfterWatermark()
        .withLateFirings(AfterCount(1))))
```
x??

---

#### Impact on Time-to-First-Output and Latency Reduction
Background context: The combination of early, on-time, and late triggers significantly reduces time-to-first-output and minimizes latency. Early updates provide speculative results for evolving data, while on-time firings ensure final, complete data is processed, and late firings handle any remaining late data.
:p How does the use of early/late triggers impact time-to-first-output and latency?
??x
The use of early/late triggers reduces time-to-first-output by providing speculative updates (early panes) for evolving data. This helps in observing how the window's state changes over time. On-time firings ensure that when the watermark passes, a complete pane is generated with final results. Late firings handle any late arriving data after the watermark has passed, reducing overall latency and ensuring efficient processing.
The impact can be seen by comparing scenarios where watermarks might be too slow or too fast:
- **Perfect Watermark Case**: Time-to-first-output is significantly reduced from almost seven minutes to three and a half minutes.
- **Heuristic Watermark Case**: Improved time-to-first-output and minimal latency between input completion and final output pane materialization.

Example code demonstrating the setup of early/late triggers:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
        .triggering(AfterWatermark()
            .withEarlyFirings(AlignedDelay(ONE_MINUTE))
            .withLateFirings(AfterCount(1))))
    .apply(Sum.integersPerKey());
```
x??

#### Allowed Lateness Concept
Background context: In out-of-order stream processing systems, late data is a significant challenge. The allowed lateness concept helps manage this by setting a limit on how late records can be and still be processed. This is critical for maintaining system efficiency and managing resources effectively.

:p What is allowed lateness in the context of stream processing?
??x
Allowed lateness allows us to set a horizon on how late any given record may be relative to the watermark for the system to bother processing it. Any data arriving after this horizon are simply dropped, preventing state from lingering indefinitely and saving resources.
```
allowedLateness = 1 minute; // Example setting in code
```
x??

---

#### Garbage Collection Concept
Background context: In long-lived out-of-order stream processing systems, persistent state for windows needs to be managed carefully. Persistent state can consume significant disk space or computational resources if not handled properly. Therefore, garbage collection policies are necessary.

:p What is the purpose of garbage collection in stream processing?
??x
The purpose of garbage collection is to ensure that system state does not linger indefinitely, which could lead to resource exhaustion (disk space) and inefficiency. By defining a horizon on allowed lateness, the system can decide when it's safe to discard old data.
```
// Pseudocode for garbage collection logic
if (watermark > windowEnd + allowedLateness) {
  // Discard state of the window
}
```
x??

---

#### Watermarks and Lateness Concept
Background context: Watermarks are used in stream processing to denote the progress of event-time. Low watermarks pessimistically attempt to capture the oldest unprocessed record, while high watermarks optimistically track the newest known record. Allowed lateness allows setting a horizon for late data.

:p What is the difference between low and high watermarks?
??x
Low watermarks are used to pessimistically track the event-time of the oldest unprocessed record in the system. They ensure that processing does not miss any potentially relevant data, even if it arrives late.
High watermarks optimistically track the event-time of the newest known record, allowing the system to drop older records beyond a specified lateness threshold.

Example code snippet for low watermark:
```java
// Example setting for low watermark logic
if (watermark > oldestUnprocessedRecord) {
  // Process data accordingly
}
```

Example code snippet for high watermark with allowed lateness:
```java
// Example setting for high watermark with allowed lateness
if (currentWatermark + allowedLateness > newestProcessedRecord) {
  // Decide to discard older records
}
```
x??

---

#### Early/Fire and Late Firings Concept
Background context: In stream processing, early firings and late firings are used to handle the timing of window triggers. Early firings allow partial results before all data is available, while late firings ensure that final results include any late data.

:p What do early firings and late firings refer to in stream processing?
??x
Early firings and late firings are strategies for handling window triggers. Early firings enable the system to emit intermediate results even when not all input data has arrived, which can improve responsiveness. Late firings ensure that final results include any late-arriving data.

Example code snippet with early and late firings:
```java
PCollection<KV<Team, Integer>> totals = input
  .apply(Window.into(FixedWindows.of(TWO_MINUTES))
    .triggering(
      AfterWatermark()
        .withEarlyFirings(AlignedDelay(ONE_MINUTE))
        .withLateFirings(AfterCount(1)))
    .withAllowedLateness(ONE_MINUTE))
  .apply(Sum.integersPerKey());
```
x??

---

#### Example of Early/Fire and Late Firings with Allowed Lateness
Background context: The provided example illustrates how early firings, late firings, and allowed lateness can be combined to process data effectively in a stream processing pipeline.

:p What does the following code snippet demonstrate?
```java
PCollection<KV<Team, Integer>> totals = input
  .apply(Window.into(FixedWindows.of(TWO_MINUTES))
    .triggering(
      AfterWatermark()
        .withEarlyFirings(AlignedDelay(ONE_MINUTE))
        .withLateFirings(AfterCount(1)))
    .withAllowedLateness(ONE_MINUTE))
  .apply(Sum.integersPerKey());
```
??x
The code snippet demonstrates a stream processing pipeline where windows of data are triggered for early firings (after one minute) and late firings (after the first late record). Additionally, it includes an allowed lateness of one minute, meaning that any data arriving within this period can still be processed.

This setup ensures timely partial results while accommodating potential late data.
x??

---

