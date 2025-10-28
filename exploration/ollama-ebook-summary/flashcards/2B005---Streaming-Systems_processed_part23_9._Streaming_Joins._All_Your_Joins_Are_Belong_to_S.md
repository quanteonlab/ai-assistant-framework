# Flashcards: 2B005---Streaming-Systems_processed (Part 23)

**Starting Chapter:** 9. Streaming Joins. All Your Joins Are Belong to Streaming

---

---
#### Streaming Joins Overview
Background context explaining that joins are grouping operations, and understanding them through a streaming perspective simplifies their complexity. Key points include:
- All joins can be considered as streaming joins.
- Tools for reasoning about time (windowing, watermarks, triggers) apply to streaming joins.

:p What is the primary concept introduced in this section?
??x
The primary concept introduced in this section is that all types of joins can be treated as streaming joins. This simplifies understanding by leveraging existing knowledge of grouping operations and applying it to streaming data.
x??

---
#### Joins as Grouping Operations
Explanation that joins are a specific type of grouping operation where data sharing some property (key) are collected into related groups.

:p How do joins differ from other grouping operations?
??x
Joins specifically group together data elements based on shared keys, creating related sets of data. This is distinct from simple grouping operations which may not necessarily rely on key matching.
x??

---
#### Streaming Grouping Operations
Explanation that streaming grouping operations consume streams and yield tables.

:p What happens during a streaming grouping operation?
??x
During a streaming grouping operation, the system processes incoming records in a stream and groups them based on shared keys. The result is a table where grouped data elements are presented.
x??

---
#### Applying Join Concepts to Streaming Data
Explanation that adding time dimensions complicates joins but can be managed using familiar tools like windowing and triggers.

:p How do streaming systems handle the complexity of join operations?
??x
Streaming systems handle join complexities by applying familiar techniques such as windowing, watermarks, and triggers. These tools allow for efficient processing of incoming data streams, making the concept of streaming joins more manageable.
x??

---
#### Example Datasets for Joins
Explanation that the datasets `Left` and `Right` are used to demonstrate various types of joins.

:p What datasets are introduced in this section?
??x
The datasets `Left` and `Right` are introduced. Each contains columns `Num`, `Id`, and `Time`. These datasets will be used throughout the text to illustrate different join operations.
x??

---
#### Key Columns in Datasets
Explanation of the key columns: `Num`, `Id`, and `Time`.

:p What are the column names and their purposes?
??x
The dataset columns include:
- `Num`: A single number identifier.
- `Id`: A unique identifier for each record, combining the table name prefix (`L` or `R`) with the `Num`.
- `Time`: The arrival time of the record in the system.

These columns are used to uniquely identify records and track their timing within the stream.
x??

---
#### Uniqueness of Join Keys
Explanation that initial datasets will have strictly unique join keys, introducing more complex cases later.

:p What is noted about the uniqueness of join keys initially?
??x
It is noted that the initial datasets will have strictly unique join keys. This simplifies the initial explanation but acknowledges that more complicated scenarios with duplicate keys will be introduced later.
x??

---

#### Full OUTER Join Concept
Background context explaining that a FULL OUTER join combines all rows from both datasets, including unmatched ones. The join predicate is an equality statement with at most one matching row on each side for equi joins.

:p What is a FULL OUTER join and how does it combine data from two tables?
??x
A FULL OUTER join returns the union of records from both input tables based on a specified condition (join key), including unmatched rows from either table. It combines all rows from both datasets, resulting in a complete list where matched rows have corresponding values, while unmatched rows will show `null` for the missing side.

Example SQL:
```sql
SELECT Left.Id as L, Right.Id as R 
FROM Left FULL OUTER JOIN Right ON Left.Num = Right.Num;
```

This can be visualized with a simple example in code to illustrate how matching and non-matching rows are combined.
x??

---

#### Streaming Joins Introduction
Background context on streaming joins over unbounded data and the myth that all such joins require windowing. Explains grouping operations and triggers used for observing join results as streams.

:p How can we observe an unwindowed join result as a stream without using windowing?
??x
We can observe an unwindowed join result as a stream by applying appropriate triggering mechanisms instead of waiting until all input is seen. Common options include:
- Trigger on every record (materialized view semantics)
- Periodic triggers based on processing time

For example, to observe the results after each new record arrives without waiting for all data:

```sql
SELECT STREAM L.Id as L, R.Id as R, CURRENT_TIMESTAMP as Time 
FROM Left FULL OUTER JOIN Right ON Left.Num = Right.Num;
```

x??

---

#### Time-Varying Relations (TVRs)
Background on time-varying relations and their importance in streaming contexts. Explains how to visualize changes over time using TVRs.

:p How do we represent a FULL OUTER join result as a time-varying relation?
??x
To represent a FULL OUTER join result as a time-varying relation (TVR), we need to capture the evolution of joined data over time, highlighting insertions and deletions. This can be illustrated by breaking down the TVR into snapshots with changes highlighted.

Example:
```sql
SELECT TVR L.Id as L, R.Id as R 
FROM Left FULL OUTER JOIN Right ON Left.Num = Right.Num;
```

This would show transitions between different states over time, such as new matches appearing and old ones disappearing. For instance:

```plaintext
-----------------------------------------------------------------------
-- |  [-inf, t1)  |  [t1, t2) |  [t2, t3)   |
| --------------- | ----------- | ----------  |
| L    | R    |     L    | R    |      L    | R    |
| ----- | ----- | -------: | ---: | -------: | ---: |
| null | R1   | -> L1  -> | null | -> L2  -> | null |
| L3   | null |           | L3   |           | L3   |
-----------------------------------------------------------------------
```

x??

---

#### Trigger Mechanisms
Explanation of different trigger mechanisms for streaming joins, including per-record, watermark-based, and periodic triggers.

:p What are the different types of triggers we can use for streaming join results?
??x
For streaming join results, we have several triggering mechanisms:
1. **Per-Record**: Triggers immediately after each new record arrives.
2. **Watermark-Based**: Waits until a certain temporal chunk is complete (e.g., within a watermark).
3. **Periodic Based on Processing Time**: Triggers at regular intervals of processing time.

Example for per-record trigger:
```sql
SELECT STREAM L.Id as L, R.Id as R 
FROM Left FULL OUTER JOIN Right ON Left.Num = Right.Num;
```

x??

---

#### Streaming TVR Example
Detailed example of how a streaming join result can be visualized over time with TVRs and deltas.

:p How do we visualize the evolution of a streaming join result using TVRs?
??x
We can visualize the evolution by breaking down the time-varying relation (TVR) into snapshots, highlighting changes between each state. For instance:

```plaintext
-----------------------------------------------------------------------
-- |  [-inf, t1)  |  [t1, t2) |  [t2, t3)   |
| --------------- | ----------- | ----------  |
| L    | R    |     L    | R    |      L    | R    |
| ----- | ----- | -------: | ---: | -------: | ---: |
| null | R1   | -> L1  -> | null | -> L2  -> | null |
| L3   | null |           | L3   |           | L3   |
-----------------------------------------------------------------------
```

This shows the state at different time intervals, illustrating insertions and deletions.

x??

---

#### Triggered Streams
Explanation of how to capture specific deltas in a streaming join result using triggered streams.

:p How do we capture specific changes (deltas) in a streaming join result?
??x
To capture specific changes, we can use the `CURRENT_TIMESTAMP` and `Sys.Undo` columns. For instance:

```sql
SELECT STREAM L.Id as L, R.Id as R, CURRENT_TIMESTAMP as Time, Sys.Undo as Undo 
FROM Left FULL OUTER JOIN Right ON Left.Num = Right.Num;
```

This allows us to see exactly when rows materialize and when updates lead to retraction of previous versions.

x??

---

#### Full Outer Join Concept
Background context explaining full outer joins and their relevance to streaming data. A full outer join combines all records from both tables, showing matches as well as non-matches with nulls for unmatched rows on one side or the other.
:p What is a full outer join in the context of streaming data?
??x
A full outer join in the context of streaming data combines all records from both input streams (Left and Right), showing matches as well as non-matches. Non-matching rows are represented with null values for the side where there is no match.
x??

---

#### Streaming Full Outer Join Example
Example provided to illustrate a full outer join in a streaming environment, showing how it captures the evolution of data over time.
:p How does the streaming full outer join evolve over time?
??x
The streaming full outer join evolves by continuously updating its state as new records arrive from both input streams. It maintains the entire relation over time, showing matches and non-matches with nulls where there are no corresponding entries in one of the streams.

For instance:
```sql
SELECT STREAM Left.Id as L,
       Right.Id   as R,
       Sys.EmitTime as Time,
       Sys.Undo   as Undo
FROM Left FULL OUTER JOIN Right
ON L.Num = R.Num;
```
This query shows how each record from both streams is matched or marked with a null, depending on whether there is a corresponding entry in the other stream.
x??

---

#### LEFT Outer Join Concept
Explanation of left outer join and its relation to full outer join. A left outer join includes all records from the left table and any matching records from the right table; unmatched rows have nulls for the right side.
:p How does a left outer join differ from a full outer join?
??x
A left outer join differs from a full outer join in that it includes all records from the left dataset (table or stream) and any matching records from the right. If there is no match, the result will include nulls for the columns from the right side.
x??

---

#### LEFT Outer Join Example
Illustration of a left outer join by filtering out unmatched rows from the full outer join.
:p How can you visualize a left outer join using a full outer join?
??x
You can visualize a left outer join by taking the full outer join and removing (filtering) all rows where there is no match on the right side. This means that only records with corresponding entries in the left stream will be included, while unmatched right-side entries are marked as null.

For example:
```sql
SELECT STREAM Left.Id as L,
       Right.Id   as R,
       Sys.EmitTime as Time,
       Sys.Undo   as Undo
FROM Left  LEFT OUTER JOIN Right
ON L.Num = R.Num;
```
This query filters out any rows where there is no match on the right, resulting in a stream that only includes left-side records and their corresponding (or null) right-side matches.
x??

---

#### LEFT Outer Join Stream Example
Detailed example showing the output of a left outer join over time with streaming data.
:p What does the output look like for a left outer join with streaming data?
??x
The output of a left outer join with streaming data will include all records from the left stream and matching records from the right. Unmatched right-side records are represented as nulls on the right side.

For example, during a given time period:
```sql
12:00> SELECT STREAM Left.Id as L,
           Right.Id   as R,
           Sys.EmitTime as Time,
           Sys.Undo   as Undo
FROM Left  LEFT OUTER JOIN Right
ON L.Num = R.Num;
```
This query might produce the following output:
```
| L    | R    | Time  | Undo |
|------|------|-------|------|
| null | R4   | 12:05 |      |
| L3   | null | 12:06 | undo |
| L2   | R2   | 12:06 |      |
```
Each row shows the evolving state of the join, with matches and unmatched right-side records represented as nulls.
x??

---

