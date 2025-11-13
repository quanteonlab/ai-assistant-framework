# Flashcards: 2B005---Streaming-Systems_processed (Part 8)

**Starting Chapter:** Understanding Watermark Propagation

---

#### Understanding Watermarks in Beam Pipeline
Background context explaining watermarks in Apache Beam. Watermarks are used to model timestamps and ensure that late data does not cause incorrect results. They help in tracking processing delays across stages of a pipeline, ensuring that computations only proceed when all data up to the current watermark has been processed.
:p What is the purpose of using watermarks in Apache Beam pipelines?
??x
Watermarks are used to track the progress and timestamping of elements as they flow through different stages of a pipeline. They help ensure that late-arriving data does not corrupt early results, maintaining correctness by delaying processing until all potentially relevant data has arrived.
x??

---
#### Per-output Buffer Watermark
Explanation of how per-output buffer watermarks provide better visibility into the system's behavior and allow for easier diagnosis of issues such as stuckness. Watermarks at this level of granularity help in understanding where exactly messages are getting stuck or delayed.
:p What does a per-output buffer watermark track?
??x
A per-output buffer watermark tracks the location of messages across various buffers in different stages of the pipeline, providing better visibility and helping diagnose issues like stuckness.
x??

---
#### Example: Gaming Score Calculation for User Engagement
Explanation of how to calculate user engagement levels by measuring session lengths. This example involves processing two independent datasets (Mobile Scores and Console Scores) in parallel using Beam pipelines.
:p How does this example measure user engagement?
??x
This example measures user engagement by first calculating per-user session lengths from gaming scores, assuming that the duration a user stays engaged with the game is a proxy for their enjoyment. It then calculates average session lengths within fixed time windows to provide insights into overall user behavior.
x??

---
#### Calculating Per-User Session Lengths
Explanation of the Beam pipeline steps involved in calculating per-user session lengths using windowing and custom transformations.
:p What are the steps in calculating per-user session lengths?
??x
The steps involve reading input data, applying windows to group by sessions with a specific gap duration (1 minute in this case), triggering processing at the watermark, discarding fired panes, and then computing session lengths via a custom transform that groups by user and calculates session length based on window size.
```java
PCollection<Double> mobileSessions = IO.read(new MobileInputSource())
    .apply(Window.into(Sessions.withGapDuration(Duration.standardMinutes(1)))
                .triggering(AtWatermark())
                .discardingFiredPanes())
    .apply(CalculateWindowLength());
```
x??

---
#### Flattening and Computing Average Session Lengths
Explanation of how to combine results from multiple stages into a single output for global session-length averages.
:p How is the combined dataset used to calculate average session lengths?
??x
The two datasets (Mobile Sessions and Console Sessions) are flattened into one PCollection. Then, these session lengths are re-windowed into fixed windows of time, and an average is calculated using the `Mean.globally()` transform.
```java
PCollection<Float> averageSessionLengths = PCollectionList.of(mobileSessions).and(consoleSessions)
    .apply(Flatten.pCollections())
    .apply(Window.into(FixedWindows.of(Duration.standardMinutes(2)))
                .triggering(AtWatermark()))
    .apply(Mean.globally());
```
x??

---
#### Watermark Propagation in Pipeline Stages
Explanation of how output watermarks relate to input watermarks and their propagation through pipeline stages.
:p How do the output and input watermarks interact in this pipeline?
??x
The output watermark for each stage is at least as old as its corresponding input watermark, reflecting processing time. The input watermark for a downstream stage is the minimum of the output watermarks from the upstream stages, ensuring that only relevant data up to the watermark is processed.
x??

---
#### Differentiating Between Output and Input Watermarks
Explanation of how watermarks propagate in different stages and their significance in pipeline execution.
:p What do the output and input watermarks represent?
??x
The output watermark represents the latest timestamp at which processing can be considered complete for a stage, while the input watermark indicates the earliest point in time from which data is considered relevant. The propagation of these watermarks ensures that late data does not affect early results.
x??

---
#### Applying Custom Transforms to Calculate Session Lengths
Explanation of custom `CalculateWindowLength` transform used to compute session lengths and its role in pipeline processing.
:p What is the role of the `CalculateWindowLength` transform?
??x
The `CalculateWindowLength` transform groups by user, treating the size of the current window as the value for each window. It computes the per-user session length based on this grouping, ensuring that the session lengths are calculated accurately from the input data.
```java
// PTransform logic to calculate session lengths
public class CalculateWindowLength extends PTransform<PCollection<KV<String, Iterable<Long>>>, PCollection<Double>> {
    @Override
    public PCollection<Double> apply(PCollection<KV<String, Iterable<Long>>> input) {
        return input.apply(GroupByKey.create())
                     .apply(MapElements.into(TypeDescriptors.doubles()).via(kv -> kv.getValue().size()));
    }
}
```
x??

---

#### Watermark Propagation and Output Timestamps Overview
Background context: In processing pipelines, output timestamps play a crucial role in determining how watermarks progress through stages. Understanding these nuances helps in optimizing the pipeline's performance and correctness.

:p What are the key aspects of watermark propagation and output timestamps in data processing pipelines?
??x
Watermark propagation involves ensuring that watermarks move forward as new non-late elements arrive, but not backward. Output timestamps dictate when a window's results are considered finalized, impacting how watermarks progress through different stages.

For example, consider the following pipeline steps:
- **Stage 1**: A window is applied with a specific timestamp combiner.
- **Stage 2**: The output of Stage 1 sets the watermark based on certain conditions.

The choice of output timestamps can significantly affect watermark progression and overall performance. 
x??

---

#### End of Window as Output Timestamp
Background context: One common approach to setting output timestamps is using the end of the window, which ensures that the results are representative of the entire window period.

:p How does using the end of the window as an output timestamp impact watermark progression?
??x
Using the end of the window as an output timestamp allows for smooth watermark progress because watermarks can advance once all elements in a window have been processed. This approach is safe and ensures that late arriving data do not cause backtracking of watermarks.

```java
// Example using Apache Beam SDK (pseudo-code)
PCollection<Double> sessions = 
    PCollection.of(...).apply(
        Window.into(Sessions.withGapDuration(Duration.standardMinutes(1)))
               .triggering(AtWatermark())
               .withTimestampCombiner(EARLIEST)
               .discardingFiredPanes()
    ).apply(CalculateWindowLength());
```
x??

---

#### Timestamp of First Nonlate Element as Output Timestamp
Background context: Another approach is to use the timestamp of the first non-late element, which can make watermarks more conservative.

:p How does using the timestamp of the first nonlate element affect watermark progression?
??x
Using the timestamp of the first nonlate element ensures that watermarks are set based on early elements, making them more conservative. This approach minimizes false positives but may delay watermark progress due to potential late elements.

```java
// Example using Apache Beam SDK (pseudo-code)
PCollection<Double> sessions = 
    PCollection.of(...).apply(
        Window.into(Sessions.withGapDuration(Duration.standardMinutes(1)))
               .triggering(AtWatermark())
               .withTimestampCombiner(EARLIEST)
               .discardingFiredPanes()
    ).apply(CalculateWindowLength());
```
x??

---

#### Timestamp of a Specific Element as Output Timestamp
Background context: Sometimes, using the timestamp of a specific element that has processed results can be useful for certain use cases.

:p How does using the timestamp of a specific element impact watermark progression?
??x
Using the timestamp of a specific element (e.g., query or click in join operations) provides flexibility but must ensure that the element is not late. This approach can lead to more precise watermarks, though it requires careful validation of non-lateness.

```java
// Example using Apache Beam SDK (pseudo-code)
PCollection<Double> sessions = 
    PCollection.of(...).apply(
        Window.into(Sessions.withGapDuration(Duration.standardMinutes(1)))
               .triggering(AtWatermark())
               .withTimestampCombiner(EARLIEST)
               .discardingFiredPanes()
    ).apply(CalculateWindowLength());
```
x??

---

#### Comparison of Watermarks and Results
Background context: Different choices for output timestamps can affect watermark delays and the results produced.

:p How does choosing different output timestamps impact the overall pipeline?
??x
Choosing different output timestamps impacts watermark delays and the results. Using the end of the window as an output timestamp allows smoother watermark progression, whereas using the first nonlate element makes watermarks more conservative but may delay progress.

Example diagrams show:
- **Figure 3-7**: Watermarks at the end of session windows.
- **Figure 3-8**: Watermarks at the beginning of session windows (i.e., earliest timestamp).

In both figures, watermark delays and resulting average session lengths differ significantly due to these choices.
x??

---

#### Semantic Differences
Background context: The choice of output timestamps has semantic differences that affect how data is processed and watermarks are managed.

:p How do the semantic differences between different output timestamps manifest in a pipeline?
??x
Semantic differences arise from how watermarks progress and when windows are considered complete. Using the end of the window makes watermark progression smoother but may delay final results, while using the first nonlate element ensures early conservative watermarking but can be delayed by late elements.

For example:
- **End of Window**: Ensures all data is processed before finalizing.
- **First Nonlate Element**: More conservative and potentially earlier, but delays watermark progress.

These differences are crucial for balancing accuracy and latency in pipelines.
x??

---

#### Session Timestamp Assignment Impact on Fixed Windows

Background context: When session timestamps are assigned to match the earliest non-late element within a session, individual sessions may end up in different fixed window buckets during subsequent calculations. This difference is not inherently right or wrong but critical for understanding and making choices based on specific use cases.

:p How does assigning session timestamps affect fixed windows?
??x
Assigning session timestamps to the earliest non-late element can cause sessions to be distributed across different fixed window buckets, leading to varying results in downstream processing. This assignment ensures correctness but may introduce delays due to watermark holding back mechanisms.
x??

---

#### Handling Sliding Windows with Overlapping

Background context: In sliding windows, a naive approach of setting output timestamps as the earliest element can lead to delayed emissions. This is because watermarks are held back based on this timestamp.

:p What issue arises when using overlapping sliding windows?
??x
Using the earliest element timestamp for output in overlapping sliding windows can cause delays. As elements move through multiple windows, later windows wait for earlier ones, even if they have completed their processing logic.
x??

---

#### Naive Approach to Sliding Windows

Background context: A naive approach to setting the output timestamp as the minimum event time often results in delayed emissions and watermark holding back issues.

:p Why might the naive approach of using the earliest element timestamp cause problems?
??x
The naive approach can lead to unnecessary delays because watermarks are held back based on the earliest event time. This means that even if a window has completed its processing, it may not be emitted downstream until all earlier windows have been processed and their watermarks have passed.
x??

---

#### Overlapping Windows with Special Logic

Background context: Beam provides special logic to ensure overlapping windows emit results in a timely manner by setting the output timestamp for each window $N+1 $ greater than the end of window$N$.

:p How does Beam handle overlapping windows to avoid delays?
??x
Beam uses special logic where the output timestamp for each window $N+1 $ is set to be greater than the end time of window$N$. This ensures that completed windows can be emitted promptly, avoiding unnecessary delays.
x??

---

#### Percentile Watermarks

Background context: Traditional watermarks track the minimum event time, but percentile watermarks use any percentile of the distribution. This allows for faster and smoother watermark advancement by discarding outliers.

:p What is a benefit of using percentile watermarks over traditional minimum event time tracking?
??x
Using percentile watermarks can provide faster and smoother watermark advancement by ignoring outliers in the long tail of the distribution. This means that the system can be more responsive to most events, even if some late events are still pending.
x??

---

#### Example Scenario with Percentile Watermark

Background context: In a scenario where event times form a compact distribution (Figure 3-9), percentile watermarks like the 90th percentile can approximate the 100th percentile closely. However, in cases with outliers (Figure 3-10), the 90th percentile significantly leads the 100th percentile.

:p How does a compact event time distribution affect percentile watermarking?
??x
In a compact distribution of event times, the 90th percentile watermark can be very close to the 100th percentile, meaning that most events are well within this watermark. This allows for efficient processing and smooth watermark advancement.
x??

---

#### Outlier Impact on Percentile Watermarks

Background context: In scenarios with outliers (Figure 3-10), percentile watermarks like the 90th can be significantly ahead of the 100th percentile, allowing the system to process most events more quickly.

:p How do outliers affect percentile watermarking?
??x
Outliers in event times can cause the 90th percentile watermark to be much earlier than the 100th percentile. By discarding these outliers, the system can advance its watermark more smoothly and efficiently, focusing on processing the bulk of events.
x??

---

