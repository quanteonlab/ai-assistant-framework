# Flashcards: 2B005---Streaming-Systems_processed (Part 22)

**Starting Chapter:** Stream and Table Selection

---

---
#### SCAN-AND-STREAM Trigger Definition
Background context: The `SCAN-AND-STREAM` trigger is a mechanism for converting tables into unbounded streams that capture the evolution of data over time. Unlike the traditional `SCAN` trigger, which processes the table at a point in time and stops, `SCAN-AND-STREAM` continues to process all subsequent modifications (inserts, deletes, updates) as they occur.

:p What is the primary difference between the `SCAN-AND-STREAM` trigger and the traditional `SCAN` trigger?
??x
The `SCAN-AND-STREAM` trigger processes a table continuously by emitting both the initial state of the table at a point in time and all subsequent modifications, creating an unbounded stream. In contrast, the `SCAN` trigger only emits the full contents of the table at one specific moment in time and then stops.

```java
// Pseudocode for SCAN-AND-STREAM Trigger
public void processTable(Table table) {
    // Emit initial state of the table
    emit(table);
    
    // Continuously listen to changes
    subscribeToChanges(table, (change) -> {
        switch(change.type) {
            case INSERT: 
                // Emit new row(s)
                emit(newRows(change.rows));
                break;
            case DELETE: 
                // Emit deleted row(s)
                emit(deletedRows(change.rows));
                break;
            case UPDATE: 
                // Emit updated (undo/redo) rows
                emit(undoneRows(change.undoneRows), doneRows(change.doneRows));
        }
    });
}
```
x??

---
#### Materialized Views and `SCAN-AND-STREAM` Trigger
Background context: The materialized views in the system use an implicit `SCAN-AND-STREAM` trigger instead of a traditional `SCAN` trigger to continuously update their state based on the modifications to the underlying input tables.

:p How does the `SCAN-AND-STREAM` trigger differ from the `SCAN` trigger when dealing with materialized views?
??x
The `SCAN-AND-STREAM` trigger is used implicitly for materialized views, allowing them to capture the continuous changes in the input tables over time. In contrast, a traditional `SCAN` trigger processes the table at a point in time and does not provide updates as new data arrives.

```java
// Pseudocode for Materialized View with SCAN-AND-STREAM Trigger
public void updateMaterializedView(Table inputTable) {
    // Use SCAN-AND-STREAM to process changes
    subscribeToChanges(inputTable, (change) -> {
        switch(change.type) {
            case INSERT: 
                // Update view with new data
                updateView(insertRows(change.rows));
                break;
            case DELETE: 
                // Remove old data from the view
                removeDataFromView(deleteRows(change.rows));
                break;
            case UPDATE: 
                // Handle updates by first deleting then inserting
                handleUpdate(change.undoneRows, change.doneRows);
        }
    });
}
```
x??

---
#### Default Selection of Streams and Tables
Background context: The choice of whether the output is a stream or a table depends on the input types. If all inputs are tables, the output should be a table; if any input is a stream, the output must be a stream.

:p What are the default rules for determining whether an output is a stream or a table?
??x
The default rules state that if all inputs to a query are tables, the output will be a table. If any of the inputs are streams, the output will be a stream. This ensures compatibility with traditional SQL behavior while allowing flexibility in handling streaming data.

```java
// Pseudocode for Default Stream/Table Selection
public OutputType determineOutputType(List<InputType> inputs) {
    if (inputs.stream().allMatch(input -> input instanceof Table)) {
        return TABLE;
    } else {
        return STREAM;
    }
}
```
x??

---

#### Event-Time vs Processing-Time Timestamps
Event-time timestamps capture the time at which an event occurred, whereas processing-time timestamps are supplied by the system and represent when records arrive. In SQL, these can be represented as additional columns.

:p What is the difference between event-time and processing-time timestamps?
??x
Event-time timestamps indicate the exact moment an event took place, while processing-time timestamps record when a record arrives in the system for processing.
For example:
- EventTime: When a user submits a score (e.g., 12:00:26)
- ProcTime: When the record is ingested into the database (e.g., 12:05:19)

```sql
-- SQL Example of a table with both event-time and processing-time columns
CREATE TABLE Scores (
    Name VARCHAR(255),
    Team VARCHAR(255),
    Score INT,
    EventTime TIMESTAMP,
    ProcTime TIMESTAMP
);
```
x??

---

#### Ordering Data by Event-Time and Processing-Time
Data can be ordered in SQL tables using `ORDER BY` clauses, either by event-time or processing-time. This is useful for understanding the sequence of events versus the order in which data is processed.

:p How can you order a table by event-time and by processing-time in SQL?
??x
In SQL, ordering by event-time and processing-time can be done as follows:

- **Event-Time Order:**
```sql
SELECT * FROM Scores ORDER BY EventTime;
```

- **Processing-Time Order:**
```sql
SELECT * FROM Scores ORDER BY ProcTime;
```

These commands sort the records based on the specified timestamps, allowing you to see how events are sequenced versus when they were processed.
x??

---

#### Temporal Validation Range (TVR) in SQL
Temporal validation range (TVR) is a concept that splits data into ranges based on event-time and processing-time. It helps in understanding the temporal context of each record.

:p What is a Temporal Validation Range (TVR) and how does it work?
??x
A Temporal Validation Range (TVR) splits records based on their event-time, showing the time intervals during which an event could have occurred. For example:

```sql
-- SQL Example to display TVR for Scores table
SELECT [EventTime1, EventTime2), Score, EventTime, ProcTime 
FROM (
    VALUES 
        ('[12:00:00, 12:05:19)', 5, '12:00:26', '12:05:19'),
        ...
) AS T(ScoreRange, Score, EventTime, ProcTime);
```

Each row represents a range of possible event times for the given score. This helps in validating whether an event could have occurred within the specified time frame.
x??

---

#### Unbounded vs Bounded Streams
Streams can be either unbounded (continuously receiving data) or bounded (finite datasets). Unbounded streams are commonly used for real-time processing, whereas bounded streams might represent historical data.

:p What is the difference between a bounded and an unbounded stream in SQL?
??x
- **Unbounded Stream:** Continuously receives new data over time. For example:
```sql
SELECT * FROM UNBOUNDED_STREAM;
```

- **Bounded Stream:** Contains a finite dataset that does not change after its initial load. For example:
```sql
SELECT * FROM BOUNDED_STREAM;
```
The unbounded stream is used for real-time processing, while the bounded stream is typically used for historical data analysis.
x??

---

#### Example of Unbounded Stream with Processing-Time
In SQL, handling an unbounded stream involves continuously updating the dataset as new records arrive. This is useful for applications that require up-to-date information.

:p How would you represent an unbounded stream in SQL with processing-time?
??x
To represent an unbounded stream with processing-time in SQL, you can use a `WINDOW` clause to define the current state of the data:

```sql
SELECT Score, EventTime, ProcTime 
FROM (
    SELECT *, ROW_NUMBER() OVER (ORDER BY ProcTime DESC) as RowNum 
    FROM UNBOUNDED_STREAM
) AS T 
WHERE RowNum = 1;
```

This query keeps track of the latest record based on processing-time and filters it to get the most recent data.
x??

---

#### Example of Bounded Stream with Processing-Time
Bounded streams are typically used for historical data. They have a fixed set of records that do not change over time.

:p How would you represent a bounded stream in SQL with processing-time?
??x
To represent a bounded stream in SQL, you can load the dataset once and treat it as a static dataset:

```sql
SELECT Score, EventTime, ProcTime 
FROM BOUNDED_STREAM;
```

This query simply selects all records from the bounded stream without any additional processing.
x??

--- 

These flashcards cover key concepts related to event-time and processing-time timestamps, ordering data in SQL, temporal validation ranges (TVR), and handling streams in a SQL context. Each card provides background information, relevant code examples, and prompts for understanding each concept deeply.

---
#### Raw Input Records vs. Transformed Relations
Raw input records are unprocessed data that have not undergone any transformations or processing steps within a pipeline. This stage focuses on reading and parsing the raw data without altering its structure.

:p How does the initial phase of a pipeline differ from the transformed relations in terms of processing?
??x
The initial phase processes raw data, which might involve reading from a file or database but not applying any transformations such as filtering or summing. Once data is parsed into a more structured form, it can be manipulated and analyzed.

For example:
```java
PCollection<String> raw = p.apply("ReadFromSource", TextIO.read().from("input.txt"));
```
Here, `raw` contains the initial unprocessed records from a source file.
x??
---

#### Summation Pipeline for Batch Processing
This concept involves performing aggregation operations on a dataset using batch processing techniques. The pipeline starts with raw input data and applies transformations to aggregate values per key.

:p What is the purpose of the summation pipeline in batch processing, as described here?
??x
The summation pipeline aims to calculate the total scores or counts for each team from the entire dataset at once. It involves reading raw records, parsing them into a structured format (e.g., `KV<Team, Integer>`), and then summing up these values.

For example:
```java
PCollection<KV<Team, Integer>> totals = input.apply(Sum.integersPerKey());
```
Here, the pipeline reads raw data, parses it, and then sums the integers per key (team).
x??
---

#### Streams and Tables View of Classic Batch Processing
The streams and tables view in this context refers to how a batch processing pipeline visualizes the flow and results. In bounded datasets, the result is rendered as a static table showing the final aggregated values.

:p How does the streams and tables view represent classic batch processing pipelines?
??x
In classic batch processing, the streams and tables view shows a single answer after all input data have been processed. This view typically includes metrics like total score, maximum event time, and maximum processing time.

For example:
```
------------------------------------------
| Total | MAX(EventTime) | MAX(ProcTime) |
------------------------------------------
|    48 |       12:07:46 |      12:09:00 |
------------------------------------------
00:00 / 00:00
```
This table shows the final aggregated results, with `END-OF-STREAM` markers indicating completion.
x??
---

#### Introduction to Windowing in SQL
Windowing allows for grouping data into time-based windows. This modification of key-based grouping is crucial for handling temporal operations and stream processing.

:p What does windowing do in the context of SQL and stream processing?
??x
Windowing in SQL divides data into logical groups based on time intervals, allowing for more complex aggregation operations over a sliding or tumbling time window. It enables processing data chunks within defined time frames to handle real-time streams effectively.

For example:
```sql
SELECT team, SUM(score)
FROM events
GROUP BY team, SESSION WINDOW session_end = TIMESTAMPADD(MINUTE, 10, event_time);
```
This SQL query sums scores for each team over a 10-minute window.
x??
---

#### Example of Windowed Beam Pipeline
The example provided demonstrates how to apply windows in the Beam framework. Fixed windows group data into fixed-size intervals, which is useful for batch processing with streaming data.

:p What does the following Beam pipeline code do and why use it?
??x
This pipeline applies a `FixedWindows` transformation to divide data into 2-minute windows before summing integers per key. This setup is typical in stream processing scenarios where data needs to be aggregated over time intervals.

For example:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES)))
    .apply(Sum.integersPerKey());
```
This code segments the data into 2-minute windows and then sums up scores for each team within those windows.
x??
---

#### Windowing in SQL for Batch Processing

Background context: The passage explains how windowing can be implemented using both implicit and explicit methods within SQL. Implicit windowing involves adding a unique feature of the window, such as an end timestamp, to the GROUP BY clause. Explicit windowing uses built-in operations like those supported by Apache Calcite.

:p How does implicit windowing work in SQL?
??x
Implicit windowing works by including some unique feature of the window, such as the end timestamp, into the GROUP BY statement. This allows SQL to group and aggregate data over a defined window period without explicitly defining the window boundaries within the query itself.
```sql
SELECT Total, Window, MAX(ProcTime)
FROM (
  SELECT SUM(Total) OVER (ORDER BY ProcTime ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS Total,
         TO_TIMESTAMP(TimestampField - INTERVAL '2 MINUTE') as WindowStart,
         TO_TIMESTAMP(TimestampField) as WindowEnd
  FROM DataStream
)
GROUP BY Window, MAX(ProcTime);
```
x??

---

#### Explicit Windowing in SQL

Background context: The passage highlights that explicit windowing uses built-in operations to define the windows directly. This can make the expression of more complex groupings easier and more intuitive.

:p How does explicit windowing differ from implicit windowing?
??x
Explicit windowing involves using specific windowing operations provided by the SQL system, such as those in Apache Calcite, to define the windows explicitly within the query. This approach makes it easier to express complex groupings like session windows compared to the ad hoc method.
```sql
SELECT Total, Window, MAX(ProcTime)
FROM (
  SELECT SUM(Total) OVER (PARTITION BY SessionID ORDER BY ProcTime ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS Total,
         TO_TIMESTAMP(TimestampField - INTERVAL '2 MINUTE') as WindowStart,
         TO_TIMESTAMP(TimestampField) as WindowEnd
  FROM DataStream
)
GROUP BY Window, MAX(ProcTime);
```
x??

---

#### Why Use Explicit Windowing in SQL

Background context: The passage discusses the advantages of using explicit windowing constructs over implicit ones. It mentions that explicit windowing simplifies the process by handling the window computation math and allows for more complex groupings like sessions.

:p What are the two main reasons to use explicit windowing constructs in SQL?
??x
The two main reasons are:
1. Explicit windowing takes care of the window-computation math, making it easier to consistently get things right when specifying basic parameters like width and slide directly.
2. It allows for more concise expression of complex dynamic groupings such as sessions.
```sql
-- Example using Apache Calcite's window function
SELECT Total, Window, MAX(ProcTime)
FROM (
  SELECT SUM(Total) OVER (PARTITION BY SessionID ORDER BY ProcTime ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS Total,
         TO_TIMESTAMP(TimestampField - INTERVAL '2 MINUTE') as WindowStart,
         TO_TIMESTAMP(TimestampField) as WindowEnd
  FROM DataStream
)
GROUP BY Window, MAX(ProcTime);
```
x??

---

#### Triggers and Watermarks for Stream Processing

Background context: The passage revisits the Beam Model's concept of triggers but suggests a different default setting for SQL. It proposes triggering on every element rather than using a single watermark trigger.

:p What is the suggested default for triggering in SQL?
??x
The suggested default for triggering in SQL is to trigger on every element, drawing inspiration from materialized views. This differs from the Beam Model's default of using a single watermark trigger.
```sql
-- Example triggering on every element in SQL
SELECT Total, Window, MAX(ProcTime)
FROM (
  SELECT SUM(Total) OVER (PARTITION BY SessionID ORDER BY ProcTime ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS Total,
         TO_TIMESTAMP(TimestampField - INTERVAL '2 MINUTE') as WindowStart,
         TO_TIMESTAMP(TimestampField) as WindowEnd
  FROM DataStream
)
GROUP BY Window, MAX(ProcTime);
```
x??

---

---
#### Per-Record Triggers
In the context of processing streaming data, per-record triggers ensure that a new output is generated for every new input record. This approach aligns with how materialized views operate and provides full fidelity by not losing any information during the conversion process.

:p What are the primary benefits of using per-record triggers in stream processing?
??x
The primary benefits include simplicity and fidelity. The semantics of per-record updates are straightforward, making it easier to understand how data is processed. Additionally, per-record triggering preserves all information, ensuring a full-fidelity representation of the data flow.

For instance, consider the following pseudo-code snippet that demonstrates using `AfterCount(1)` in Apache Beam:
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
        .triggering(Repeatedly(AfterCount(1))))
    .apply(Sum.integersPerKey());
```

This code sets up a pipeline where every new record triggers an update to the aggregated score.
x??
---

#### Cost Consideration of Per-Record Triggers
While per-record triggers offer clarity and simplicity, they come with a cost. Grouping operations can reduce data cardinality, which is often used as an optimization technique in stream processing pipelines.

:p What are the main disadvantages of using per-record triggers?
??x
The main disadvantage is increased computational cost due to the application of triggers after grouping operations. Since grouping reduces the number of records, applying triggers on aggregated results downstream can be less costly than on individual records. However, this comes at the expense of potentially more complex and resource-intensive processing.

For example, if we have a stream of team scores and want to compute total scores over fixed windows:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
        .triggering(Repeatedly(AfterCount(1))))
    .apply(Sum.integersPerKey());
```
Here, the `AfterCount(1)` trigger ensures that a sum is computed for each new record, which can be costly if not optimized.
x??
---

#### Streams and Tables Rendering
When using per-record triggers, the behavior of the stream and table rendering in a streaming engine changes. The aggregate values remain at rest while ungrouped streams continue to flow.

:p How does the use of per-record triggers affect the rendering of aggregated data in a stream processing system?
??x
With per-record triggers, the aggregate values do not move; they remain stable as new records are processed. Meanwhile, ungrouped streams containing individual records continue to flow through the system. This can be visualized using diagrams where grouped and ungrouped data paths diverge.

For instance, consider a pipeline that calculates total scores:
```java
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(FixedWindows.of(TWO_MINUTES))
        .triggering(Repeatedly(AfterCount(1))))
    .apply(Sum.integersPerKey());
```
This setup ensures that each new record triggers an update to the total score.
x??
---

#### SQL Representation of Per-Record Triggers
In SQL, per-record triggers can be represented as a continuous stream of values. This is similar to how change data capture systems work.

:p What does the SQL representation of a per-record trigger look like in practice?
??x
The SQL representation using per-record triggers would involve continuously updating records as new inputs arrive. Each update reflects changes made by each new input, providing a detailed view of ongoing transformations.

For example, if we have a stream representing team scores over time:
```
------------------------------------------------
| Total | Window               | MAX(ProcTime)     |
------------------------------------------------
| 5     | [12:00:00, 12:02:00) | 12:05:19         |
| 7     | [12:02:00, 12:04:00) | 12:05:39         |
| 10    | [12:02:00, 12:04:00) | 12:06:13         |
| 4     | [12:04:00, 12:06:00) | 12:06:39         |
| 18    | [12:02:00, 12:04:00) | 12:07:06         |
| 3     | [12:06:00, 12:08:00) | 12:07:19         |
| 14    | [12:00:00, 12:02:00) | 12:08:19         |
| 11    | [12:06:00, 12:08:00) | 12:08:39         |
| 12    | [12:06:00, 12:08:00) | 12:09:00         |
------------------------------------------------
```
This continuous stream reflects how each new record updates the total score over time.
x??
---

#### Watermark Triggers in Beam Pipeline
In data processing pipelines, especially for large-scale applications like mobile apps, it's crucial to manage the cost and efficiency of processing updates. One common challenge is handling downstream updates in real-time based on upstream user scores. This can be optimized using watermark triggers in Apache Beam.

Watermark triggers help in managing late data by providing a mechanism to process elements once their timestamps are past a certain threshold (watermark). If an element's timestamp is after the current watermark, it will be processed immediately; otherwise, it might be buffered or dropped depending on the trigger settings.
:p What does a watermark trigger do in Apache Beam?
??x
A watermark trigger processes elements based on their timestamps relative to a watermark. Elements with timestamps past the watermark are processed immediately, while those not yet past the watermark are deferred and may either buffer until the watermark is surpassed or be dropped if they are considered too late.
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply( new ParseFn() );
PCollection<KV<Team, Integer>> totals = input
   .apply(Window.into(FixedWindows.of(TWO_MINUTES))
                 .triggering(AfterWatermark()))
   .apply(Sum.integersPerKey());
```
x??

---

#### Single Output per Window with Watermark Triggers
Using watermark triggers in Beam pipelines can result in exactly one output per window, enhancing efficiency and reducing overhead. This is particularly useful when processing continuous streams of data where each window's sum or aggregation needs to be calculated precisely once.

The following example demonstrates how to use a watermark trigger to achieve single outputs per window.
:p How does using `AfterWatermark()` affect the pipeline output?
??x
Using `AfterWatermark()` in Beam ensures that the PCollection processes elements only when the watermark has surpassed their timestamps. This results in exactly one output per window, as the system waits for all possible late data to arrive before triggering the final state of the window.

Here's an example:
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
   .apply(Window.into(FixedWindows.of(TWO_MINUTES))
                 .triggering(AfterWatermark())
   .apply(Sum.integersPerKey());
```
x??

---

#### Handling Late Data with Watermark Triggers
Late data is a common issue in real-time stream processing, where elements arrive after the expected timestamp. To handle late data effectively, Beam supports watermark triggers that can process these late elements by triggering an output when they are received.

This example shows how to use `AfterWatermark().withLateFirings(AfterCount(1))` to handle late records.
:p How does `AfterWatermark().withLateFirings(AfterCount(1))` work in Beam?
??x
`AfterWatermark().withLateFirings(AfterCount(1))` allows the pipeline to emit results not only when the watermark passes but also once for any late elements that arrive. This ensures that every record, even those that come late, is processed.

Example:
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
   .apply(Window.into(FixedWindows.of(TWO_MINUTES))
                 .triggering(AfterWatermark().withLateFirings(AfterCount(1)))
   .apply(Sum.integersPerKey());
```
x??

---

#### Emission of Results at Specific Timestamps
Understanding when trigger firings occur is crucial for managing the flow of data in pipelines. The `EmitTime` column in the example indicates the exact timestamp when results are emitted, which helps in tracking the timing and ensuring that late data is processed correctly.

This example illustrates how to capture `EmitTime` values.
:p How does `EmitTime` help in understanding trigger firings?
??x
`EmitTime` captures the precise moment when a window's aggregation result is emitted. This timestamp helps in tracking the timing of processing and ensuring that late data triggers the emission at the correct time.

Example:
```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
   .apply(Window.into(FixedWindows.of(TWO_MINUTES))
                 .triggering(AfterWatermark())
   .apply(Sum.integersPerKey());
```
Here, `EmitTime` would be the timestamp of each row's emission.
x??

#### EmitTiming and EmitIndex System Columns
Background context: When processing data in a windowed manner, understanding the timing of each row relative to the watermark and identifying the pane index can be crucial for accurate results. The `Sys.EmitTiming` system column provides information about when each emitted value is relative to the watermark, while `Sys.EmitIndex` helps identify the sequence of emissions within a specific window.

:p What are the roles of `Sys.EmitTiming` and `Sys.EmitIndex` in windowed processing?
??x
The roles of `Sys.EmitTiming` and `Sys.EmitIndex` in windowed processing are to provide detailed insights into the timing and sequence of data within windows. `Sys.EmitTiming` indicates whether a value is emitted on time or late relative to the watermark, which helps ensure correct results even with delayed data. On the other hand, `Sys.EmitIndex` tracks the order of emissions for each window, useful for managing state across multiple rows.

```java
// Example code snippet demonstrating the use of Sys.EmitTiming and Sys.EmitIndex
public class WindowProcessing {
    public PCollection<KV<Team, Integer>> processInput(PCollection<String> raw) {
        return raw.apply("Parse", ParDo.of(new ParseFn()))
                   .apply(Window.into(FixedWindows.of(TWO_MINUTES))
                              .triggering(Repeatedly(UnalignedDelay(ONE_MINUTE))))
                   .apply(Sum.integersPerKey());
    }
}
```
x??

---

#### Repeated Delay Triggers
Background context: In situations where you need to process data with a delay relative to the most recent incoming record, using repeated delay triggers can help distribute processing load more evenly. This is particularly useful in scenarios where data arrives sporadically or in bursts.

:p How does the repeated delay trigger work?
??x
The repeated delay trigger works by triggering a window one minute after any new data for it arrives, but with additional delays between subsequent triggers to avoid excessive processing load. This method ensures that each window is processed not too soon after receiving its first input, and also provides time for handling late-arriving data.

```java
// Example code snippet using repeated delay triggering
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> totals = raw.apply(
    "Parse", ParDo.of(new ParseFn())
).apply(Window.into(FixedWindows.of(TWO_MINUTES))
           .triggering(Repeatedly(UnalignedDelay(ONE_MINUTE))))
  .apply(Sum.integersPerKey());
```

This approach helps in balancing the load and managing costs effectively by delaying triggers, which allows some rows to be elided if they are not significantly different from previously processed ones.

x??

---

#### Windowed Summation with Repeated One-Minute Delay Triggering
Background context: In scenarios where timely aggregation of data is necessary but late-arriving data should still contribute to the final result, using a repeated delay trigger can provide an effective solution. This method processes windows one minute after receiving their initial data and then continues to update based on new incoming data.

:p How does windowed summation with repeated one-minute delay triggering work?
??x
Windowed summation with repeated one-minute delay triggering works by initially processing each window after a minute of inactivity (or the arrival of any new data) and then continuing to process it as new data arrives. This approach ensures that each window gets timely attention while also accommodating late-arriving data.

```java
// Example code snippet for windowed summation with repeated delay triggering
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> totals = raw.apply(
    "Parse", ParDo.of(new ParseFn())
).apply(Window.into(FixedWindows.of(TWO_MINUTES))
           .triggering(Repeatedly(UnalignedDelay(ONE_MINUTE))))
  .apply(Sum.integersPerKey());
```

This method helps in achieving a good balance between timeliness and accuracy, as it ensures that each window is processed with some delay to handle late data appropriately.

x??

---

---
#### Data-Driven Triggers in SQL
Background context explaining how data-driven triggers could fit naturally within an `EMIT WHEN` clause, with examples like triggering on a condition such as `Score > 10`. However, it is explained that this would essentially behave similarly to a `HAVING` clause and can be achieved by simply using the predicate at the end of the query. The question arises whether explicit data-driven triggers are necessary.
:p How does an `EMIT WHEN Score > 10` statement in SQL work?
??x
An `EMIT WHEN Score > 10` would trigger on every record, check if the condition is met for each row, and then propagate downstream only those rows where the condition holds true. This is effectively similar to using a `HAVING` clause at the end of the query.
```sql
SELECT * 
FROM scores 
EMIT WHEN Score > 10;
```
x??

---
#### Accumulation Mode in SQL
Background context on the default use of accumulation mode, where later revisions build upon previous ones. This approach is used for simplicity and matches earlier examples but has significant drawbacks when dealing with sequences of grouping operations due to over-counting issues.
:p What is the main drawback of using accumulation mode in SQL queries?
??x
The main drawback of accumulation mode is that it can lead to over-counting issues, especially in scenarios involving multiple serial grouping operations. This happens because a single row can be included multiple times in an aggregation if its revisions are not properly handled.
```sql
-- Example: Incorrect handling of row revisions leading to over-counting
SELECT SUM(score) 
FROM scores 
GROUP BY user_id;
```
x??

---
#### Retraction Mode as the Default
Explanation on why retraction mode is preferred by default, especially in systems that allow for complex queries with multiple grouping operations. This approach avoids including rows multiple times and ensures accurate aggregation.
:p Why is retraction mode considered better than accumulation mode by default?
??x
Retraction mode is preferred because it prevents over-counting issues in queries involving multiple grouping operations. By retracting old values, the system ensures that each row is only included once in the final aggregation result, leading to more accurate computations.
```sql
-- Example of retraction mode ensuring accuracy
SELECT SUM(score) 
FROM scores 
GROUP BY user_id;
```
x??

---

---
#### Retractions in SQL Pipelines
Retractions are a feature that allows a pipeline to explicitly remove elements that were previously emitted, making it easier to maintain idempotent and exactly-once semantics when writing incremental sessions or updates to storage systems like HBase. In scenarios where session windows evolve over time, retraction is crucial because the new session might replace an older one.

:p What are retractions in SQL pipelines?
??x
Retractions are a feature that allows a pipeline to explicitly remove elements (sessions) that were previously emitted, ensuring idempotent and exactly-once semantics when writing incremental updates or sessions to storage systems like HBase. This is particularly useful when new session windows overlap with old ones, requiring the replacement of previous session data.
x??

---
#### Session Windows Without Retractions
In a Beam pipeline without retractions, aggregation results are accumulated but do not remove older sessions that have been replaced by newer ones. This leads to potential overwrites in storage systems like HBase and complicates maintaining exactly-once semantics.

:p What happens if a session window is aggregated without using retractions?
??x
Without retractions, the Beam pipeline accumulates session windows but does not remove older sessions when they are replaced by new ones. As a result, the output stream will contain overlapping sessions that overwrite previous data in storage systems like HBase, making it difficult to maintain exactly-once semantics.

For example:
```sql
00:00 / 00:00 -------------------------------------------
| Total | Window               | EmitTime |
-------------------------------------------
| 5     | [12:00:26, 12:01:26) | 12:05:19 |
| 7     | [12:02:26, 12:03:26) | 12:05:39 |
| 3     | [12:03:39, 12:04:39) | 12:06:13 |
| 7     | [12:03:39, 12:05:19) | 12:06:46 |
```
x??

---
#### Session Windows With Retractions
When using retractions in a Beam pipeline, the system explicitly notifies about sessions that were added and those that were removed. This approach simplifies maintaining exactly-once semantics by providing clear changes to be applied.

:p How do retractions help in managing session windows?
??x
Retractions help in managing session windows by explicitly notifying the system about which sessions are being added or removed. This means the pipeline can maintain exactly-once semantics more effectively, as it knows precisely what changes need to be made to the storage systems like HBase.

For example:
```sql
00:00 / 00:00 -------------------------------------------
| Total | Window               | EmitTime |
-------------------------------------------
| 5     | [12:00:26, 12:01:26) | 12:05:19 |
| 7     | [12:02:26, 12:03:26) | 12:05:39 |
| 3     | [12:03:39, 12:04:39) | 12:06:13 |
| -7    | [12:03:39, 12:05:19) | 12:06:46 | (retraction)
| 3     | [12:06:39, 12:07:39) | 12:07:19 |
```
x??

---
#### Beam Code for Non-Retracting Pipeline
The provided Beam code shows a session window pipeline that accumulates totals but does not handle retractions. This setup can lead to issues when writing incremental sessions to key/value stores.

:p What is the Beam code snippet showing in Example 8-7?
??x
Example 8-7 shows a Beam code snippet for a session window pipeline that uses per-record triggering and accumulation but does not include any handling for retractions. This setup accumulates totals over time but does not remove old sessions when new ones are added, potentially causing issues with storage systems like HBase.

```java
PCollection<String> raw = IO.read(...);
PCollection<KV<Team, Integer>> input = raw.apply(new ParseFn());
PCollection<KV<Team, Integer>> totals = input
    .apply(Window.into(Sessions.withGapDuration(ONE_MINUTE))
        .triggering(Repeatedly(AfterCount(1)))
        .accumulatingFiredPanes())
    .apply(Sum.integersPerKey());
```
x??

---
#### SQL Output with Retractions
The provided SQL output shows how sessions are managed when retractions are used. Each session is either added or removed, making it clear what changes need to be applied.

:p What does the SQL output in Figure 8-13 show?
??x
Figure 8-13 demonstrates how sessions are managed with retractions enabled. The output shows both additions and removals of sessions explicitly, which helps in maintaining exactly-once semantics when writing incremental updates or sessions to storage systems like HBase.

For example:
```sql
00:00 / 00:00 -------------------------------------------
| Total | Window               | EmitTime |
-------------------------------------------
| 5     | [12:00:26, 12:01:26) | 12:05:19 |
| 7     | [12:02:26, 12:03:26) | 12:05:39 |
| -7    | [12:03:39, 12:05:19) | 12:06:46 | (retraction)
| 3     | [12:06:39, 12:07:39) | 12:07:19 |
```
x??

---

#### Handling Sessions and Retractions without Retractions Enabled

Background context: In scenarios where you need to process streams of incremental sessions, such as handling user activity or session-based data, managing superseded sessions can be complex. Without retractions enabled, each new session must overwrite previous ones through a series of read-modify-write operations, which are costly and non-idempotent.

:p How would processing sessions without enabling retractions work?
??x
Without enabling retractions, handling new sessions involves a significant cost because you need to compare and possibly delete old sessions before adding the new one. This process is not idempotent and can lead to loss of exactly-once semantics.

For example:
```java
// Pseudocode for processing a session without retractions
public void processSession(Session newSession) {
    // Read existing sessions from storage
    List<Session> existingSessions = readExistingSessions();
    
    // Compare each existing session with the new one to identify overlaps
    for (Session existing : existingSessions) {
        if (newSession.overlaps(existing)) {
            // Issue delete commands for obsolete sessions
            deleteSession(existing);
        }
    }

    // Write the new session after all deletions are completed
    writeSession(newSession);
}
```
x??

---

#### Handling Sessions and Retractions with Retractions Enabled

Background context: Enabling retractions simplifies managing superseded sessions by automatically generating retraction rows when a new session replaces an old one. This allows for more efficient updates to the state, ensuring that only the latest sessions are stored.

:p How does enabling retractions simplify the handling of superseded sessions?
??x
Enabling retractions makes it easier to manage superseded sessions because each new session automatically generates retraction rows for any previous sessions it replaces. This means you can simply write new sessions as they arrive and delete old ones using retraction rows.

For example:
```java
// Pseudocode for processing a session with retractions enabled
public void processSession(Session newSession) {
    // Write the new session to storage (redo row)
    writeSession(newSession);

    // Generate and apply retraction rows for any obsolete sessions
    List<Session> existingSessions = readExistingSessions();
    for (Session existing : existingSessions) {
        if (newSession.overlaps(existing)) {
            // Apply a retraction row for the obsolete session
            applyRetraction(existing);
        }
    }
}
```
x??

---

#### Discarding Mode vs. Accumulating and Retracting Mode

Background context: For specific use cases, discarding mode can be valuable as it allows simpler pipelines to partially aggregate high-volume input data and write them into a storage system that supports aggregation. However, outside of these narrow use cases, enabling discarding mode is confusing and error-prone.

:p How does incorporating retractions in SQL help with accumulating and retracting mode?
??x
Incorporating retractions in SQL helps by allowing for both accumulating and retracting modes naturally. Retractions ensure that only the latest sessions are stored, making it easier to manage state updates without complex read-modify-write operations.

For example:
```sql
-- SQL Query Example with Retractions
SELECT 
    SUM(Total) AS Total,
    Window
FROM (
    SELECT 
        5 AS Total,
        '[12:00:26, 12:01:26)' AS Window,
        '12:05:19' AS EmitTime
    UNION ALL
    SELECT 
        7 AS Total,
        '[12:02:26, 12:03:26)' AS Window,
        '12:05:39' AS EmitTime
    -- Additional rows...
) t
GROUP BY Window
```
x??

---

#### Discarding Mode in Specific Use Cases

Background context: Discarding mode can be useful for simple pipelines where high-volume data is aggregated using a single grouping operation, and the results are written to a storage system that supports aggregation. However, outside of these specific use cases, discarding mode is not recommended due to its complexity and error-prone nature.

:p Why might discarding mode be confusing and error-prone in general pipelines?
??x
Discarding mode can be confusing and error-prone because it requires careful management of state updates without retractions. In complex pipelines, ensuring that only the latest state is retained while correctly managing deletions can lead to bugs or incorrect results.

For example:
```java
// Pseudocode for a pipeline with discarding mode
public void processSession(Session newSession) {
    // Write the new session (ignoring any existing sessions)
    writeSession(newSession);

    // No need for retraction rows, but state updates must be carefully managed
}
```
x??

---

---
#### Streaming vs. Nonstreaming Data Processing: Key Difference
Background context explaining that streaming data processing differs from nonstreaming (point-in-time) data processing by adding a temporal dimension to the data's evolution over time. This leads to the concept of Time-Variable Relations (TVRs), which represent how relations change over time.
:p What is the key difference between streaming and nonstreaming data processing?
??x
The key difference lies in the added temporal dimension, where streaming data processes the evolution of a relation as it changes over time. This contrasts with nonstreaming or snapshot-based processing, which captures static points-in-time views of relations.

For example:
- Nonstreaming: A relation R(t) at time t is a fixed point-in-time view.
- Streaming: A TVR (Time-Variable Relation) represents the sequence of snapshots over time.

In SQL and relational algebra, this means that traditional operators like selection, projection, and join can still be applied in a streaming context without losing their closure property. This ensures that all SQL constructs remain functional even when dealing with evolving data.
x??

---
#### Time-Variable Relations (TVRs)
Background context explaining the concept of TVRs as sequences of snapshot relations over time, which capture the evolution of a relation dynamically. This maintains the closure properties of relational algebra and allows for robust stream processing within the same framework.

:p What are Time-Variable Relations (TVRs)?
??x
Time-Variable Relations (TVRs) represent how a relation changes over time as a sequence of snapshots at different points in time. They maintain the integrity of relational operations by ensuring that all standard SQL constructs and operators can be applied consistently, thus preserving the closure property.

For example:
```java
// Pseudocode for creating a TVR from a stream of data
TVR myRelation = Stream.of(data).collect(TVR::new);
```
x??

---
#### Beam Model vs. Classic SQL Model: Biases
Background context explaining that both the Beam model and classic SQL model have inherent biases, with Beam being stream-oriented and SQL being table-oriented. This distinction affects how data is processed and managed within each framework.

:p What are the key differences between the Beam model and the classic SQL model?
??x
The Beam model focuses on a stream-oriented approach, meaning it processes data in a continuous, event-driven manner. In contrast, the classic SQL model is table-oriented, handling data in discrete, snapshot-based views.

This difference impacts processing:
- Beam: Suitable for real-time analytics where data flows continuously.
- SQL: Best suited for batch processing and historical data analysis with static snapshots.

For example:
```java
// Pseudocode to illustrate stream vs. table processing
BeamPipeline pipeline = Pipeline.create();
pipeline.apply("ReadFromStream", Read.from(...))
       .apply("ProcessAsStream", ProcessAsStream())
       .apply("WriteToTable", Write.into(...));

SqlQuery query = "SELECT * FROM database WHERE timestamp > '2023-10-01'"; // Table-oriented
```
x??

---
#### Language Extensions for Stream Processing in SQL
Background context explaining the necessity of adding language extensions to support robust stream processing within SQL. This includes keywords like `TABLE`, `STREAM`, and `TVR` for specifying the type of data rendering.

:p What are the hypothetical language extensions needed for robust stream processing in SQL?
??x
The necessary language extensions include:
- **Table (`TABLE`)**: Specifies that a relation should be processed as a static, snapshot view.
- **Stream (`STREAM`)**: Indicates that a relation is to be processed in real-time with temporal considerations.
- **TVR (`Time-Variable Relation`)**: Represents the evolution of relations over time and allows for dynamic query execution.

For example:
```sql
SELECT * FROM customers TABLE; -- Table-oriented query
SELECT * FROM sales STREAM WHERE timestamp > '2023-10-01'; -- Stream-oriented query with a filter
```
x??

---
#### Windowing in SQL
Background context explaining the value of windowing operators for expressing complex, dynamic groupings like sessions. While some simple windows can be declared declaratively using existing SQL constructs, explicit windowing operators encapsulate the necessary computations and improve query expressiveness.

:p What are windowing operators in SQL?
??x
Windowing operators allow you to define and apply data windows over a set of rows in a table or stream. These operators help in performing complex aggregations and operations on subsets of data that change dynamically over time, such as session-based analysis.

For example:
```sql
-- Using an existing window function like ROW_NUMBER() OVER (PARTITION BY ...)
SELECT customer_id, timestamp, amount,
       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY timestamp) as row_num
FROM transactions;

-- Explicit windowing operator for sessions
SELECT * FROM sales WINDOW session AS (
    SELECT customer_id, SUM(amount)
    FROM sales
    GROUP BY customer_id, session_start(timestamp, 1 HOUR)
);
```
x??

---
#### Watermarks in Stream Processing Systems
Background context explaining watermarks as a system-level feature used to manage the completeness of data and generate streams with authoritative versions. This is particularly useful for use cases where direct stream consumption is necessary.

:p What are watermarks in stream processing systems?
??x
Watermarks are used to track the progress of data in streaming applications, ensuring that all data up to a certain point is considered complete before it's processed further. They help manage the completeness of input data and generate authoritative versions of rows only after the system believes the input for those rows is fully available.

For example:
```java
// Pseudocode for watermark handling in a stream processing system
WatermarkManager manager = new WatermarkManager();
Stream<Row> output = input.apply("AssignTimestampsAndWatersmarks", new AssignerWithPeriodicWatermarks<>(
    new CustomAssigner(), new TimeSerializable(5000)));
```
x??

#### Watermark Triggers
Watermark triggers are used to yield a single output per window when the inputs to that window are believed to be complete. This is particularly useful for handling late-arriving data where the system needs to assert with confidence that no more updates will come.
:p What does a watermark trigger do in stream processing?
??x
A watermark trigger processes a window of data and generates one output per window when it believes all relevant data has arrived. It helps manage late events by asserting that no further updates are expected for the current window, allowing the system to finalize its state before moving on.
```sql
-- Example SQL using watermark triggers
SELECT * FROM stream_data WINDOW TUMBLING (SIZE 1 HOUR) WATERSHED BY event_timestamp;
```
x??

---

#### Repeated Delay Triggers
Repeated delay triggers are used for providing periodic updates. This type of trigger is useful when the application needs to receive regular, continuous data streams rather than single-shot or event-based outputs.
:p What does a repeated delay trigger do in stream processing?
??x
A repeated delay trigger generates output periodically, even if no new input events have arrived since the last output. It is ideal for scenarios where continuous monitoring and updates are required, such as real-time analytics or periodic data aggregation.
```java
// Pseudocode example of a repeated delay trigger implementation
public class RepeatedDelayTrigger {
    private long delayTime;
    
    public void onEvent(Event event) {
        // Process the event
        process(event);
        
        // Schedule next output after a delay
        scheduleNextOutput();
    }
    
    private void scheduleNextOutput() {
        // Schedule next output in delay time
        Timer.schedule(this, delayTime);
    }
}
```
x??

---

#### System Columns for Time-Varying Relations
System columns provide metadata that are useful when consuming time-varying relations (TVRs) as streams. These system columns include processing time, emit timing, emit index, and undo status.
:p What are some important system columns used in stream processing?
??x
Important system columns used in stream processing include:
- `Sys.MTime`: The processing time at which a row was last modified.
- `Sys.EmitTiming`: The timing of the row emit relative to the watermark (early, on-time, late).
- `Sys.EmitIndex`: The zero-based index of the emit version for this row.
- `Sys.Undo`: Indicates whether the row is a normal row or a retraction.

These columns help in managing data flow and handling retractions effectively.
```sql
-- Example SQL query with system columns
SELECT Sys.MTime, Sys.EmitTiming, Sys.EmitIndex, Sys.Undo FROM stream_data;
```
x??

---

#### Valid Relation Concept
A valid relation for an SQL query is one where the attributes used in the query exist within the relation. If a relation does not contain the required attribute, it is considered invalid and will result in a query execution error.
:p What defines a "valid relation" in SQL queries?
??x
A valid relation in SQL queries must contain all attributes referenced by the query. For example, for the query `SELECT x FROM y`, the relation `y` must have an attribute named `x`. If it does not, the query is invalid and will yield a query execution error.
```java
// Example code to check if a relation is valid
public boolean isValidRelation(Map<String, List<Object>> attributes) {
    String attributeNeeded = "x"; // Example attribute from SELECT statement
    return attributes.containsKey(attributeNeeded);
}
```
x??

---

#### Default Triggers in Stream Processing
If unspecified, the default trigger in stream processing is per-record triggering. This provides straightforward and natural semantics matching those of materialized views. Other useful triggers include watermark triggers for complete window outputs and repeated delay triggers for periodic updates.
:p What are the different types of triggers used in stream processing?
??x
Triggers in stream processing can be categorized as follows:
1. **Per-Record Triggering**: The default type, providing straightforward semantics similar to materialized views.
2. **Watermark Triggers**: Yield a single output per window when the inputs to that window are believed to be complete.
3. **Repeated Delay Triggers**: Provide periodic updates rather than event-based outputs.

These triggers help in managing data flow and ensuring correct processing of streams.
```sql
-- Example SQL with different types of triggers
SELECT * FROM stream_data PER_RECORD;
SELECT * FROM stream_data WINDOW TUMBLING (SIZE 1 HOUR) WATERSHED BY event_timestamp;
SELECT * FROM stream_data REPEAT_DELAY(INTERVAL 5 SECONDS);
```
x??

---

#### Streaming vs Table Views
Background context: The text discusses how streams and tables can be viewed differently, even though they are fundamentally time-varying relations. Tables represent a snapshot of data at a specific point in time, while streams represent continuous changes over time.

:p How do streams and tables differ despite both being time-varying relations?
??x
Streams represent continuous updates over time, whereas tables capture a snapshot of the state at a particular moment. This difference impacts how operations are performed on them; for example, handling insertions, deletions, and updates in streams requires different mechanisms compared to updating static tables.

For instance:
- **Tables** can be thought of as fixed collections of data points.
- **Streams** need to handle changes dynamically and maintain state over time.

In pseudocode, handling a stream might look like this:

```java
for (DataRecord record : incomingStream) {
    // Process the record based on current state
}
```

x??

---

#### Triggers in Pipelines
Background context: The text mentions that triggers should be specified at the outputs of a pipeline and automatically propagated throughout. This approach aims to simplify the handling of stream semantics.

:p How can triggers be used in pipelines?
??x
Triggers in pipelines are defined at the output, indicating when changes need to be processed or aggregated. When a trigger is activated (e.g., after a certain number of records have been received), it propagates upstream to affect how intermediate transformations handle their data. This ensures that the pipeline correctly handles the state and timing of input streams.

For example:
- **Trigger:** "Process records every 50 events."
- **Pseudocode:**
```java
Pipeline p = Pipeline.create();
PCollection<Record> records = p.apply("ReadRecords", TextIO.read().from(pattern));
records.apply(new ProcessWindowFn<>(50, new DefaultAggregatorFn<>()));
```

x??

---

#### MATERIALIZED Views vs Bounded Queries
Background context: The text compares the use of materialized views and bounded queries. Materialized views are designed to incrementally update based on a stream of changes, while bounded queries process data from the start to the end.

:p What is the difference between using MATERIALIZED views and re-executing bounded queries?
??x
MATERIALIZED views are optimized to handle incremental updates in response to streaming data, ensuring that only new or updated data is processed. Bounded queries, on the other hand, process the entire dataset from start to finish each time they are executed.

For example:
- **Materialized View:**
```sql
CREATE MATERIALIZED VIEW sales AS SELECT product_id, SUM(amount) as total_sales FROM transactions GROUP BY product_id;
```
This view updates incrementally based on new or updated transaction records.
- **Bounded Query (e.g., SQL):**
```sql
SELECT product_id, SUM(amount) as total_sales FROM transactions WHERE date >= '2023-01-01' AND date <= '2023-12-31';
```
This query processes the entire dataset for a specific time range.

x??

---

#### Incremental vs Full Processing
Background context: The text highlights that both streams and tables can be processed incrementally or in full, depending on the nature of the data and the processing requirements.

:p How do you decide between incremental and full processing?
??x
Incremental processing is suitable for streaming applications where changes are continuous and need to be handled dynamically. Full processing might be more appropriate when dealing with bounded datasets that can be reprocessed from start to finish, such as historical data or periodic snapshots of current state.

For example:
- **Incremental Processing:**
```java
Pipeline p = Pipeline.create();
PCollection<Record> records = p.apply("ReadRecords", TextIO.read().from(pattern));
records.apply(new ProcessWindowFn<>(50, new DefaultAggregatorFn()));
```
This processes data in windows of 50 records and aggregates them incrementally.

- **Full Processing:**
```java
Pipeline p = Pipeline.create();
PCollection<Record> records = p.apply("ReadRecords", TextIO.read().from(pattern));
records.apply(new ProcessWindowFn<>(TimeDomain.FULL, new FinalAggregatorFn()));
```
This processes the entire dataset at once for a final aggregation.

x??

---

#### SQL's Table Bias and Batch Processing
Background context: SQL was designed primarily for batch processing, where data is processed in large chunks, often once a day or less. This bias towards tables in SQL can sometimes make it challenging to handle real-time data streams efficiently.

:p How does the table bias of SQL affect its handling of real-time data?
??x
The table bias of SQL can limit its flexibility when dealing with real-time data because traditional SQL operations are optimized for batch processing, making it harder to manage streaming data in a continuous and dynamic manner.
x??

---
#### Capturing Event Time for Real-Time Data
Background context: In some cases, using the current processing time as event time for records can be beneficial. This is particularly useful when logging events directly into tables like TVRs (Time-Varying Relations), where the timestamp of ingestion serves as a natural event time.

:p Why might you use the current processing time as event time in real-time data systems?
??x
Using the current processing time as event time can be advantageous because it reflects the actual moment when an event occurred or was ingested, which is crucial for maintaining the temporal accuracy of records in real-time systems.
x??

---
#### Use of Retractions and Event Time
Background context: Retractions are used to handle changes to data over time. However, they should not be mandatory but rather an option that can be enabled based on specific use cases.

:p In what scenarios might retractions be unnecessary or undesirable?
??x
Retractions may be unnecessary in scenarios where the system can detect that they are not required as an optimization. For example, when writing results into external storage systems that support per-key updates, there is no need to reinsert old data that has already been overwritten.
x??

---
#### Windowing Constructs and Sessions
Background context: Windowing constructs allow processing of data within specific time windows, such as sessions in real-time data processing. The definition of "index" becomes complex when dealing with merging windows like sessions.

:p How do you define the index for merged session windows?
??x
To define the index for merged session windows, take the maximum of all previous session indices being merged and increment by one. This ensures that each session has a unique identifier even after merging.
x??

---
#### Handling Sessions in State Tables
Background context: In systems where sessions need to be managed, writing or deleting sessions at their emit time is crucial. Reading from state tables should also consider the output watermark to ensure coherence.

:p What are the steps to ensure a globally coherent view of all sessions at any given time?
??x
To ensure a globally coherent view of all sessions:
1. Write or delete each session at its emit time.
2. Read only from the HBase table at a timestamp that is less than the output watermark from your pipeline to synchronize reads with multiple, independent writes and deletes when sessions merge.

Alternatively, serving sessions directly from state tables can simplify this process.
x??

---
#### Calcite Support for Windowing Constructs
Background context: The Calcite library supports windowing constructs as described in certain chapters. This support enables more complex data processing involving time-based windows and aggregations.

:p What does the Calcite library provide in terms of windowing?
??x
The Calcite library provides support for windowing constructs, allowing more sophisticated time-based operations like session windows or tumbling windows to be defined and processed.
x??

---

