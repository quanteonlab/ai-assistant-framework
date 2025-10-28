# Flashcards: 2B005---Streaming-Systems_processed (Part 21)

**Starting Chapter:** Streams and Tables

---

#### Time-Varying Relations Overview
Background context explaining time-varying relations. Time-varying relations capture the full history of a relation over time, where each snapshot at a specific point in time is represented as a classic relation (or table).
:p What are time-varying relations?
??x
Time-varying relations represent the evolution of data over time by capturing snapshots of the relation at different points in time. These snapshots form a sequence that shows how the data changes with each event.
x??

---

#### Time-Varying Relations vs Classic Relations
Explanation on how time-varying relations are composed of classic relations and how they behave independently for each snapshot.
:p How do time-varying relations relate to classic relations?
??x
Time-varying relations are sequences of classic relations, where each classic relation represents the state of the relation at a particular point in time. Each individual classic relation can be treated as a static representation of the data at that specific moment, making it possible to apply queries independently on each snapshot.
x??

---

#### Time-Varying Relations and Stream Processing
Explanation on how time-varying relations tie into stream processing by representing historical data over time.
:p How do time-varying relations relate to stream processing?
??x
Time-varying relations provide a way to represent the history of changes in data over time, which is crucial for stream processing applications. By capturing snapshots at different points in time, these relations allow us to analyze and process data streams effectively, maintaining historical context.
x??

---

#### Observing Time-Varying Relations as Tables
Explanation on how observing time-varying relations as tables yields specific snapshots at given times.
:p How can we observe a time-varying relation as a table?
??x
To observe a time-varying relation as a table, we effectively query the relation for its state at a particular point in time. This snapshot represents a classic relation that captures all data up to and including the specified time.
Example: Observing the grouped time-varying relation at 12:07:
```plaintext
-------------------------
| Name  | Total | Time  |
-------------------------
| Julie | 12    | 12:07 |
| Frank | 3     | 12:03 |
-------------------------
```
x??

---

#### SQL Support for Time-Varying Relations
Explanation on how the SQL 2011 standard supports time-varying relations through temporal tables and AS OF SYSTEM TIME.
:p How does SQL support time-varying relations?
??x
The SQL 2011 standard introduces "temporal tables," which store a versioned history of the table over time. This is essentially a form of time-varying relation. The AS OF SYSTEM TIME construct allows for querying and receiving snapshots of these temporal tables at specific points in time, making it possible to view historical data as if it were current.
Example: Querying the relation at 12:03:
```sql
SELECT * FROM temporal_table AS OF '2023-10-15T12:03';
```
x??

---

#### Practical Example of Time-Varying Relations in SQL
Explanation on how to use SQL queries with time-varying relations.
:p How can we query a time-varying relation using SQL?
??x
To query a time-varying relation, you can use the AS OF SYSTEM TIME clause to specify the point in time for which you want the snapshot. For example:
```sql
SELECT * FROM grouped_time_varying_relation AS OF '2023-10-15T12:03';
```
This query would return the state of the relation as it existed at 12:03, providing a snapshot that can be used for analysis.
x??

---

#### Time-Varying Relation (TVR) Snapshots
Background context explaining that TVRs capture a snapshot of relations at specific points in time. The example provided shows how data changes over time with different snapshots.

:p What are the characteristics of a time-varying relation (TVR)?
??x
A time-varying relation captures the state of a table at various points in time, showing how it evolves over a period. Each snapshot represents the table's state at a particular moment.
For example:
```plaintext
|       [-inf, 12:01)       | [12:01, 12:03)      |
| Name  | Total | Time  |   | Name  | Total | Time  |
|-------|-------|-------|   |-------|-------|-------|
| Julie | 7     | 12:01 |   |       |       |       |
| Frank | 3     | 12:01 |   |       |       |       |

|       [12:03, 12:07)      |    [12:07, now)      |
| Name  | Total | Time  |   | Name  | Total | Time  |
|-------|-------|-------|   |-------|-------|-------|
| Julie | 8     | 12:03 |   | Julie | 12    | 12:07 |
| Frank | 3     | 12:03 |   | Frank | 3     | 12:03 |
```
x??

---

#### Stream of Changes
Background context explaining that streams capture the sequence of changes in a time-varying relation, focusing on event-by-event updates.

:p How does a stream differ from a traditional TVR snapshot?
??x
A stream captures the evolution of a time-varying relation by tracking the sequence of changes rather than holistically capturing snapshots. This means that each row in the stream reflects an update or change to the original data.
For example, starting at 12:01:
```plaintext
| Name  | Total | Time  |
-------------------------
| Julie | 7     | 12:01 |

| Name  | Total | Time  | Undo   |
-------------------------
| Julie | 7     | 12:01 |        |
```
x??

---

#### Stream Rendering Example
Background context explaining the introduction of new keywords and columns for stream rendering, including `STREAM` keyword and `Sys.Undo` column.

:p How would a stream rendering look at 12:01 according to the provided example?
??x
At 12:01, the stream rendering of the time-varying relation would look like this:
```plaintext
| Name  | Total | Time  |
-------------------------
| Julie | 7     | 12:01 |

| Name  | Total | Time  | Undo   |
-------------------------
| Julie | 7     | 12:01 |        |
```
x??

---

#### Stream Keyword and Undo Column
Background context explaining the purpose of `STREAM` keyword and `Sys.Undo` column in stream rendering.

:p What are the roles of the `STREAM` keyword and `Sys.Undo` column?
??x
The `STREAM` keyword indicates that a query should return an event-by-event stream capturing the evolution of the time-varying relation over time. The `Sys.Undo` column is used to identify rows that represent retractions (deletions or changes).
For example, in the stream rendering at 12:03:
```plaintext
| Name  | Total | Time  |
-------------------------
| Julie | 8     | 12:03 |

| Name  | Total | Time  | Undo   |
-------------------------
| Julie | 7     | 12:01 |        |
| Julie | 8     | 12:03 | 12:01  |
```
x??

---

#### Event Sequence in Streams
Background context explaining how streams capture the sequence of changes rather than snapshots.

:p How does a stream capture the evolution of a time-varying relation?
??x
Streams capture the sequence of changes that result in the snapshots of a time-varying relation. Each event in the stream represents an update, insertion, or deletion.
For example:
```plaintext
| Name  | Total | Time  |
-------------------------
| Julie | 7     | 12:01 |

| Name  | Total | Time  | Undo   |
-------------------------
| Julie | 7     | 12:01 |        |
| Julie | 8     | 12:03 | 12:01  |
```
This shows that at `12:03`, the total for Julie was updated from 7 to 8, and the previous state (at `12:01`) is recorded in the `Undo` column.
x??

---

#### Table vs. Stream Rendering Differences
Background context explaining how tables and streams differ, particularly focusing on their rendering at different times. Mention that table rendering provides a complete snapshot while stream rendering represents an ongoing process.

:p How do the table and stream renderings differ when displaying data at 12:01?
??x
The difference lies in the completeness of the displayed data and how they handle ongoing changes. At 12:01, the table version shows all the final data points, ending with a line of dashes to signify completion. In contrast, the stream rendering is incomplete, indicated by an ellipsis-like line of periods that suggests more data could be added in the future.

The stream rendering captures the ongoing nature of data processing and change over time. For instance, at 12:03, three new rows appear for the stream query, reflecting changes due to new inputs or updates.
x??

---

#### Undo Column in Stream Rendering
Explanation about how the `Sys.Undo` column allows capturing the retraction of previously reported values during data aggregation and processing.

:p How does the `Sys.Undo` column function in a stream rendering?
??x
The `Sys.Undo` column is crucial for tracking changes to aggregated values. It helps distinguish between normal rows and rows that represent a retraction of previously reported values. For example, if Julie's total score was initially reported as 7 but later updated to 8 due to new data, the `Sys.Undo` column would mark this change, indicating that the previous value (7) was incorrect.

This allows for accurate tracking and updating of aggregated totals over time.
x??

---

#### Stream Rendering and OLTP Tables
Explanation about how stream rendering can be seen as a sequence of insertions and deletions to materialize an aggregation in an OLTP environment. 

:p How does the `STREAM` rendering relate to Online Transaction Processing (OLTP) tables?
??x
The `STREAM` rendering can be viewed as capturing a sequence of operations that are akin to the operations performed in an OLTP database: insertions, deletions, and updates. Specifically, it reflects how data is aggregated over time and how this aggregation changes with new inputs.

In essence, if you were to materialize the stream representation in an OLTP system, it would simulate a series of `INSERT` and `DELETE` operations that update the state of the relation as new data arrives. This mirrors the dynamic nature of OLTP tables, which are constantly being updated through transactions.
x??

---

#### Handling Retractions in Stream Rendering
Explanation on whether to include or exclude retractions from stream rendering depending on needs.

:p When should you include or exclude retractions in a `STREAM` query?
??x
Whether to include or exclude retractions depends on the specific requirements of your application. If you are interested in tracking all changes, including the retraction and subsequent updates of values, then you would keep the `Sys.Undo` column.

However, if you only care about the current state of the data without concern for historical corrections, you can omit these retractions. This simplifies the output by removing the need to handle and display retracted rows.
x??

---

#### Stream vs Table-Based Time-Varying Relation (TVR) Overview
Background context: The text discusses the differences between stream and table-based representations of time-varying relations. It explains how a stream captures changes over time, while a table captures snapshots at specific points in time.

:p How do streams and tables differ when representing time-varying relations?
??x
Streams capture individual changes to the relation over time, whereas tables provide a snapshot of the entire relation at a specific point in time. The stream rendering is more concise as it captures only the delta of changes between each snapshot, while the sequence-of-tables TVR rendering provides clarity by showing the evolution of the relation over time.

```java
// Example of a Stream processing logic (pseudo-code)
public class StreamProcessor {
    public void process(Stream<String> input) {
        input.map(line -> parseLine(line))
             .filter(event -> event.getTime().isAfter(lastSnapshotTime))
             .forEach(event -> updateTable(event));
    }
}
```
x??

---

#### Value of the STREAM Rendering
Background context: The text highlights that the stream rendering is more concise and captures only the changes between each snapshot in time, making it easier to understand the evolution of data.

:p What is a key advantage of using the STREAM rendering for representing time-varying relations?
??x
The key advantage of the stream rendering is its conciseness. It captures only the delta of changes between each point-in-time relation snapshot, making it easier to understand how and when changes occur in the data.

```java
// Example of a simple Stream rendering logic (pseudo-code)
public class StreamRenderer {
    public void render(Stream<ChangeEvent> stream) {
        stream.forEach(event -> System.out.println(event.toString()));
    }
}
```
x??

---

#### Value of the Sequence-of-Tables TVR Rendering
Background context: The text also emphasizes that the sequence-of-tables TVR rendering provides clarity by showing how the relation evolves over time, maintaining a natural relationship to classic relations and defining relational semantics within the context of streaming.

:p What is a key advantage of using the sequence-of-tables TVR rendering?
??x
A key advantage of using the sequence-of-tables TVR rendering is its clarity. It captures the evolution of the relation over time in a format that highlights its natural relationship to classic relations, providing a simple and clear definition of relational semantics within the context of streaming.

```java
// Example of a Sequence-of-Tables rendering logic (pseudo-code)
public class TableRenderer {
    public void render(List<TVRSnapshot> snapshots) {
        for (TVRSnapshot snapshot : snapshots) {
            System.out.println(snapshot.toString());
        }
    }
}
```
x??

---

#### Stream/Table Duality
Background context: The text explains that streams and tables are essentially two different physical manifestations of the same concept, representing a complete time-varying relation from both sides. However, in practice, they can be lossy due to resource constraints.

:p What is meant by the "stream/table duality" in the context of time-varying relations?
??x
The stream/table duality refers to the idea that streams and tables are two different physical manifestations of the same concept. They both represent a complete time-varying relation but from different perspectives: streams focus on individual changes over time, while tables capture snapshots at specific points in time. However, in practice, due to resource constraints, they can be lossy, meaning they may not fully encode all historical information.

```java
// Example of a Stream/Table dual rendering logic (pseudo-code)
public class DualRenderer {
    public void render(Stream<ChangeEvent> stream, List<TVRSnapshot> tables) {
        // Process and compare both representations
    }
}
```
x??

---

#### Trade-offs in Stream/Table Representations
Background context: The text discusses the trade-offs between full-fidelity streams and tables versus lossy versions. Full-fidelity is impractical for large data sources, leading to partial-fidelity representations that sacrifice some information but offer benefits like reduced resource costs.

:p What are the common scenarios where stream and table manifestations of a time-varying relation might be lossy?
??x
Common scenarios where stream and table manifestations of a time-varying relation might be lossy include encoding only the most recent version, compressing history to specific point-in-time snapshots, and garbage-collecting older versions. For streams, they often encode only a limited duration of the evolution, typically a relatively recent portion of the history.

```java
// Example of Stream and Table Lossiness Handling (pseudo-code)
public class DataEncoder {
    public void encode(Stream<ChangeEvent> stream) {
        // Handle lossy encoding by retaining only recent changes
    }

    public void encode(List<TVRSnapshot> tables) {
        // Handle lossy encoding by retaining snapshots up to a certain threshold
    }
}
```
x??

---

---
#### Stream Bias in Beam Model
Background context explaining the inherent stream bias in the Beam Model. The Beam Model's approach to processing data is fundamentally based on streams, even for batch operations.

Beam operates using `PCollections`, which are always streams. This means that every transformation or operation in a Beam pipeline is applied to streams and results in new streams. Even operations like grouping, which conceptually should result in tables, are treated as stream transformations under the hood.

:p What is the inherent bias of the Beam Model regarding data processing?
??x
The inherent bias of the Beam Model is towards streams. In Beam, `PCollections` represent streams, and all transformations operate on these streams, even for batch operations which need to be converted into a stream-based model internally.

This bias requires special handling when dealing with tables. For example:
- **Sources**: Consume tables and trigger them in specific ways (e.g., every update, batch updates, or snapshots).
- **Sinks**: Write tables by grouping streams in certain predefined manners.
- **Grouping/Ungrouping Operations**: Provide flexibility but always operate on streams.

Code Example to illustrate the transformation of a table into a stream:
```java
PCollection<KV<String, Integer>> groupedData = input.apply(GroupByKey.create());
```
x??

---
#### Stream vs. Table in Beam Model
Background context explaining how the Beam Model treats tables and their relationship with streams. Tables are treated specially as they need to be converted into streams for processing.

In Beam, operations like grouping (which typically result in table-like structures) must be handled differently because streams are the fundamental unit of data flow.

:p How does the Beam Model handle table operations?
??x
The Beam Model handles table operations by converting them into stream-based transformations. For example:
- Grouping operations (`GroupByKey`) operate on `PCollections` (streams), even if they conceptually create tables.
- When writing to a sink, users need to define how input streams are grouped.

This conversion is necessary because in Beam's internal model, all data flows as streams:

```java
// Example of grouping and then applying a trigger explicitly:
PCollection<KV<String, Integer>> groupedData = input.apply(GroupByKey.create());
groupedData.apply(Trigger.afterWatermark(Duration.ofSeconds(10)).discardingFiredPanes());
```

x??

---
#### Triggers in Beam Model
Background context explaining the concept of triggers and their role in managing data in the Beam Model. Triggers determine how and when updates to a dataset should be processed.

Triggers are used to control the timing of operations on `PCollections`, which are streams. Predeclaration or post-declaration of triggers is necessary because the Beam model operates on streams, not tables.

:p What are triggers in the context of Beam Model?
??x
Triggers in the Beam Model determine how and when updates to a dataset should be processed. They help manage the timing of operations on `PCollections`, which represent streams of data.

There are two ways to declare triggers:
1. **Predeclaration**: Triggers are specified before the table to which they apply.
2. **Post-declaration**: Triggers are specified after the table to which they apply.

For example, consider a scenario where you want to process updates every 10 seconds:

```java
// Predeclaring trigger for a stream:
PCollection<KV<String, Integer>> input = ...;
input.apply(GroupByKey.create())
      .apply(Trigger.afterWatermark(Duration.ofSeconds(10)).discardingFiredPanes());

// Post-declaring trigger:
PCollection<KV<String, Integer>> input = ...;
input.apply("Apply Trigger", ParDo.of(new DoFn<WindowedValue<KV<String, Integer>>, KV<String, Integer>>() {
  @ProcessElement
  public void process(ProcessContext c) {
    // Process the element based on a custom trigger logic.
  }
}))
```

x??

---

---
#### Beam Model's Stream-Centric Approach
The Beam model uses a stream-centric approach, where operations on data are defined over streams. In this context, windows and triggers play crucial roles to process and observe changes in data streams.

:p What is the primary characteristic of Beam's stream-centric approach?
??x
Beam’s primary characteristic lies in its focus on processing data streams through the definition of windows and triggers. These mechanisms allow for the segmentation of continuous data into manageable chunks for analysis, enabling a flexible and efficient way to handle real-time or batch processing.
```
// Pseudocode example:
pCollection.apply(Window.into(FixedWindows.of(Duration.standardMinutes(5))));
pCollection.apply(Trigger.afterProcessingTime(Duration.standardSeconds(10)));
```
x??
---

#### SQL Model's Table-Biased Approach
SQL, on the other hand, takes a table-biased approach where queries operate on tables to produce new tables. This model aligns closely with batch processing paradigms but can also handle real-time data through snapshots.

:p How does the SQL model differ from Beam in terms of handling data?
??x
The SQL model differs significantly by focusing on querying and transforming existing tables, producing new tables as results. This approach is well-suited for both batch and streaming scenarios using techniques like snapshot triggering.
```
// Example SQL Query:
SELECT team, SUM(score) as total
FROM UserScores
GROUP BY team;
```
x??
---

#### Snapshot Triggering in the SQL Model
In the context of SQL queries, a snapshot triggering mechanism is used to ensure that the data at query execution time is correctly captured and processed. This ensures consistency by taking a snapshot of the table's state before processing.

:p What does snapshot triggering accomplish in the SQL model?
??x
Snapshot triggering captures the current state of a table as it existed when the query was executed, ensuring consistent results without being affected by changes that occur after the query starts but before it completes.
```
// Pseudocode for Snapshot Trigger:
Stream<UserScores> snapshot = inputTable.trigger(SnapshotTrigger.create());
```
x??
---

#### Projection and Grouping in SQL Queries
SQL queries often involve multiple steps, including projection (selecting specific columns) and grouping (aggregating data by certain keys). These operations can be split into separate queries for better manageability.

:p How do projection and grouping fit into the SQL model's processing pipeline?
??x
Projection involves selecting specific columns from a table, while grouping aggregates rows based on common keys. Splitting these steps allows for more modular query design and easier management of complex transformations.
```sql
// Example Queries:
SELECT team, score 
INTO TeamAndScore 
FROM UserScores;

SELECT team, SUM(score) as total 
INTO TeamTotals 
FROM TeamAndScore 
GROUP BY team;
```
x??
---

#### Stream and Table Conversions in SQL
Background context: In classic SQL, streams are not first-class objects. Consequently, when a query involves operations that produce streams (like `SELECT`), these are often converted to tables due to the table-centric nature of SQL. This conversion can introduce implicit steps that may affect performance.
:p How does the conversion from stream to table and back again occur in the first query?
??x
In the first query, the `SELECT` operation produces a stream, but since SQL operates on tables, this stream is converted into a table using an implicit `SCAN` operation. The `GROUP BY` then groups these rows by their identity (using physical storage offsets), and after grouping, it might need to be scanned back into a stream for further operations like aggregation.
```sql
-- Pseudocode representing the first query's execution plan
SELECT team, score FROM input_table;
```
x??

---

#### Having Clause Impact on Table Bias
Background context: The addition of a `HAVING` clause changes how intermediate tables are handled. It introduces an implicit stream that must be filtered and then re-grouped back into a table.
:p What happens when a `HAVING` clause is added to the query?
??x
When a `HAVING` clause is added, it triggers an implicit conversion of the table (TeamTotals) into a stream. This stream is then filtered according to the rules in the `HAVING` clause and grouped back into a new table called LargeTeamTotals.
```sql
-- Pseudocode representing the second query's execution plan with HAVING
SELECT team, SUM(score) FROM TeamAndScore GROUP BY team HAVING SUM(score) > threshold;
```
x??

---

#### Input Tables in SQL
Background context: In classic SQL, input tables are always implicitly converted into a stream at query execution time and then grouped. This behavior is similar to batch processing systems like MapReduce.
:p How do input tables behave in SQL?
??x
Input tables in SQL are always implicitly converted into a stream when the query starts execution. This conversion yields a bounded stream containing a snapshot of the table at that moment, akin to how data is processed in batch systems such as MapReduce.
```java
// Pseudocode for processing input tables in SQL
StreamTable inputStream = new StreamTable(tableData);
```
x??

---

#### Output Tables in SQL
Background context: Output tables in SQL can be either direct results of a final grouping operation or implicit groupings applied to terminal streams. This behavior mirrors the handling of data in batch systems.
:p What characterizes output tables in SQL?
??x
Output tables in SQL are typically created by:
1. Final grouping operations that directly manifest as tables.
2. Implicit groupings on terminal streams when no final grouping is present, using a unique identifier for each row to group the stream back into a table.
This behavior closely aligns with how batch processing systems handle data at the end of their pipeline.
```java
// Pseudocode for creating an output table in SQL
Table outputTable = terminalStream.groupBy(team).apply("SUM", score);
```
x??

---

#### Grouping/Ungrouping Operations
Grouping and ungrouping operations in SQL provide flexibility primarily through grouping mechanisms. Unlike Beam, which offers a broader suite of streaming operations including join and cube functionalities, SQL supports only one type of implicit ungrouping operation: triggering an intermediate table after all contributing data has been incorporated.
:p What is the difference between grouping/ungrouping operations in Beam and SQL?
??x
Beam provides a more comprehensive suite of operations that include join, cube, etc., offering greater flexibility in both grouping and ungrouping. In contrast, SQL supports only one type of implicit ungrouping operation, which involves triggering an intermediate table once all contributing data has been processed.
```java
// Example: Beam's rich set of operations
PCollection<KV<String, Integer>> scores = ...;
PCollection<KV<String, Iterable<Integer>>> groupedScores = scores
    .apply(GroupByKey.create())
    .apply(MapElements.into(KV.class)
        .via(score -> KV.of(score.getKey(), Iterables.cycle(score.getValue()))));
```
x??

---

#### Materialized Views in SQL
Materialized views represent a specific type of stream processing where data is physically stored as tables and updated continuously based on changes to the source table(s). This feature adds streaming capabilities to SQL without significantly altering its core operations, including its inherent focus on tabular data.
:p How do materialized views add stream processing capabilities to SQL?
??x
Materialized views enable continuous, ongoing queries that process updates to a base table in real-time. They achieve this by keeping the view's data synchronized with the source tables' changes. This feature is akin to time-varying relations and provides a way to perform complex, continuously updating queries.
```sql
CREATE MATERIALIZED VIEW TeamAndScoreView AS 
SELECT team, score 
FROM UserScores;

CREATE MATERIALIZED VIEW LargeTeamTotalsView AS 
SELECT team, SUM(score) as total 
FROM TeamAndScoreView 
GROUP BY team 
HAVING SUM(score) > 100;
```
x??

---

#### Physical Execution Diagram of Materialized Views
The physical execution diagram for materialized views looks nearly identical to that of one-off queries. However, the key difference lies in the trigger mechanism used: a `SCAN-AND-STREAM` operation is substituted instead of a simple `SCAN`, ensuring continuous updates based on data changes.
:p How does the physical execution plan differ when using materialized views compared to regular SQL queries?
??x
The physical execution plan for materialized views remains similar to that of one-off queries, with the main difference being the trigger mechanism. Instead of a simple `SCAN` operation, a `SCAN-AND-STREAM` is used to ensure continuous updates based on data changes.
```java
// Pseudo-code for a SCAN-AND-STREAM operation
public class ScanAndStreamTrigger {
    public void processElement(Data data) {
        // Process each element from the source and update the materialized view
    }
}
```
x??

---

