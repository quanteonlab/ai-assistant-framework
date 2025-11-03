# Flashcards: 2B005---Streaming-Systems_processed (Part 24)

**Starting Chapter:** INNER

---

---
#### RIGHT OUTER JOIN Concept
Background context explaining the RIGHT OUTER JOIN. A RIGHT OUTER JOIN includes all rows from the right dataset, and corresponding matching rows from the left dataset. If no match is found on the left side, the result will include a row with NULL values for those columns.

:p What does a RIGHT OUTER JOIN do?
??x
A RIGHT OUTER JOIN returns all rows from the right table (Right) and the matched rows from the left table (Left). If there is no match in the left table, the result includes NULLs for the left-side columns.
Example:
```sql
SELECT   STREAM Left.Id as L,
         Right.Id   as R,
         Left.Id as L,
         Sys.EmitTime as Time,
         Right.Id as R,
         Sys.Undo   as Undo
FROM Left  RIGHT OUTER JOIN Right
ON L.Num = R.Num;
```
x??

---
#### INNER JOIN Concept
Background context explaining the INNER JOIN. An INNER JOIN returns only the rows that have matching values in both tables based on a specified column.

:p What is an INNER JOIN?
??x
An INNER JOIN produces a result set that contains only the rows where there is a match between the columns of two tables. Any rows without a corresponding match are excluded.
Example:
```sql
SELECT   STREAM Left.Id as L,
         Right.Id   as R,
         Left.Id as L,
         Sys.EmitTime as Time,
         Right.Id as R,
         Sys.Undo   as Undo
FROM Left  INNER JOIN Right
ON L.Num = R.Num;
```
x??

---
#### ANTI JOIN Concept
Background context explaining the ANTI JOIN. An ANTI JOIN returns all rows from the right table (Right) where there is no match in the left table (Left).

:p What does an ANTI JOIN do?
??x
An ANTI JOIN filters out all rows that have a corresponding match on the left side of the join, returning only those rows from the right dataset that do not match any row in the left dataset.
Example:
```sql
SELECT   STREAM Left.Id as L,
         Right.Id   as R,
         Left.Id as L,
         Sys.EmitTime as Time,
         Right.Id as R,
         Sys.Undo   as Undo
FROM Left  ANTI JOIN Right
ON L.Num = R.Num;
```
x??

---
#### Retractions in INNER JOIN Streams Concept
Background context explaining how retractions are handled in INNER JOIN streams. When a value is updated or deleted, it can affect the join results, leading to retractions and reinsertions.

:p How do retractions play a role in INNER JOIN streams?
??x
Retractions in INNER JOIN streams occur when data changes in either table involved in the join. If an existing row is updated or removed (retracted), this change propagates through both the stream and the resulting join, leading to retractions or reinsertions of rows.
Example:
```sql
SELECT   STREAM LeftV2.Id as L,
         Right.Id   as R,
         LeftV2.Id as L,
         Sys.EmitTime as Time,
         Right.Id as R,
         Sys.Undo   as Undo
FROM LeftV2  INNER JOIN Right
ON L.Num = R.Num;
```
If `Left` row with `Num` of 3 is updated from "L3" to "L3v2" at 12:07, this will result in a retraction of the old value and insertion of the new one.
x??

---

---
#### ANTI Join Explanation
Background context explaining how ANTI joins work. In an ANTI join, you match rows from one stream with those that do not match any row in another stream based on a given condition.

:p What happens during an ANTI join?
??x
During an ANTI join, for each row in the left stream (`Left`), it checks if there is a corresponding row in the right stream (`Right`) that matches the specified condition. If no such match exists, the row from the left stream is included in the result.

Example:
```sql
SELECT STREAM Left.Id as L,
       Right.Id   as R,
       Sys.EmitTime as Time,
       Sys.Undo   as Undo
FROM Left ANTI JOIN Right
ON L.Num = R.Num;
```

This query will return rows from `Left` where there is no corresponding row in `Right` that matches the condition (`L.Num = R.Num`). Retractions (undo) are common when a match appears later.

x??
---

#### INNER Join vs. ANTI Join
Background context explaining the difference between INNER and ANTI joins, including their expected outputs.

:p How do INNER and ANTI joins differ in output?
??x
INNER join returns all rows from both streams where there is a match based on the specified condition.
ANTI join returns only those rows from the left stream that have no corresponding matches in the right stream.

Example:
```sql
SELECT STREAM Left.Id as L,
       Right.Id   as R,
       Sys.EmitTime as Time,
       Sys.Undo   as Undo
FROM Left INNER JOIN Right
ON L.Num = R.Num;

SELECT STREAM Left.Id as L,
       Right.Id   as R,
       Sys.EmitTime as Time,
       Sys.Undo   as Undo
FROM Left ANTI JOIN Right
ON L.Num = R.Num;
```

The first query will return rows where `L.Num` matches `R.Num`, while the second will only include rows from `Left` where there is no match in `Right`.

x??
---

#### SEMI Join Explanation
Background context explaining how SEMI joins work. A SEMI join returns a row from the left stream for each matching row in the right stream, but drops the right side of the joined values.

:p What does a SEMI join do?
??x
A SEMI join checks if there is at least one match between rows in the two streams based on a condition. It returns only the rows from the left stream that have matches, discarding the right stream's data.

Example:
```sql
SELECT STREAM Left.Id as L,
       Right.Id   as R,
       Sys.EmitTime as Time,
       Sys.Undo   as Undo
FROM Left SEMI JOIN Right
ON L.Num = R.Num;
```

This query will return rows from `Left` where there is at least one corresponding row in `Right`.

x??
---

#### INNER and SEMI Joins in N:M Cardinality

Background context: This concept explains the differences between INNER and SEMI joins, particularly when dealing with many-to-many (N:M) cardinalities. INNER join returns rows where there is at least one match in both datasets, while SEMI join only filters the left dataset based on the existence of a matching row in the right dataset.

:p How do INNER and SEMI joins behave differently with N:M cardinality?
??x
INNER join will return all combinations of rows from the left and right tables where there is at least one match. In contrast, SEMI join ensures that only rows in the left table are returned if any matching row exists in the right table.

Explanation: For example, consider a situation where multiple rows on the right side can correspond to a single row on the left side. INNER join will return every combination of these matches, while SEMI join will simply ensure that each matching pair is recognized but does not repeat the results.

```java
public class Example {
    // Consider two tables Left and Right with N:M cardinality.
    // InnerJoin would produce all possible pairs where there's a match:
    // e.g., L1-R2, L1-R3, L2-R2, L3-R4A, L3-R4B
    // SemiJoin would only ensure that the left side row has at least one match.
}
```
x??

---

#### FULL OUTER Join with N:M Cardinality

Background context: A FULL OUTER join combines all records from both input tables and returns NULL on either side if no matches are found. This is useful when you want to see all rows from both sides, even those that do not have a matching row in the other table.

:p How does a FULL OUTER join behave with N:M cardinality relations?
??x
A FULL OUTER join will return all records from both the left and right tables. For any pair of rows where there is no match between them, NULL values will be returned on one or both sides.

Explanation: Using the example data provided, each row in `LeftNM` will be paired with every row in `RightNM` based on their `N_M` fields. If a match does not exist, NULL values will be filled in for that side of the join.

```java
public class Example {
    // Consider FULL OUTER JOIN between LeftNM and RightNM.
    // For example:
    // L5A -> R1 (L5A has no N_M 0:1 matches, thus R1 is null)
    // R4B -> L3 (R4B has no N_M 1:2 matches, thus L3 is null)
}
```
x??

---

#### SEMI Join in N:M Cardinality

Background context: A SEMI join filters the left dataset based on the existence of a match in the right dataset. It does not return any columns from the right side, only ensuring that there is at least one matching row.

:p What is the behavior of a SEMI join with an N:M cardinality?
??x
A SEMI join will ensure that each row on the left side has at least one match in the right side. However, it does not return any columns from the right dataset, only indicating the existence of a match.

Explanation: In a scenario where multiple rows can correspond to a single row (N:M), a SEMI join will filter the left table based on whether there is at least one matching row in the right table. This means that if `L5A` has two matches with `R4A` and `R4B`, only `L5A` will appear, but it won't return any columns from `RightNM`.

```java
public class Example {
    // SEMI join example:
    // If L5A -> {R4A, R4B}, then the output would be:
    // | L5A |
}
```
x??

---

#### INNER Join with N:M Cardinality

Background context: An INNER join returns only rows where there is a match in both datasets. It filters out any rows that do not have a corresponding row in the other table.

:p How does an INNER join behave in an N:M cardinality scenario?
??x
An INNER join will return all combinations of rows from the left and right tables where there is at least one match. This means that if a row on the left side has multiple matches on the right side, each match will be returned as a separate result.

Explanation: In the context of N:M cardinality, an INNER join will produce every possible pair of matching rows between the two datasets. If `L5A` has two matches with `R4A` and `R4B`, both combinations (`L5A-R4A` and `L5A-R4B`) will be returned.

```java
public class Example {
    // INNER join example:
    // If L5A -> {R4A, R4B}, the output would be:
    // | L5A - R4A |
    // | L5A - R4B |
}
```
x??

#### Filtered INNER Join vs. SEMI Join

Background context explaining the difference between a filtered INNER join and a SEMI join, including how they handle duplicate values.

:p What is the primary difference between a filtered INNER join and a SEMI join when dealing with multiplicative joins?
??x

The primary difference lies in handling multiple matches (M > 1) for a given predicate. In a **filtered INNER JOIN**, if there are multiple matches, all combinations of these matches will be included in the result set, leading to duplicate values. On the other hand, a **SEMI JOIN** does not include duplicates and only returns unique rows based on the matching predicate.

For example:
```sql
SELECT COALESCE(LeftNM.N_M, RightNM.N_M) as N_M,
       LeftNM.Id as L 
FROM LeftNM INNER JOIN RightNM 
ON LeftNM.N_M = RightNM.N_M;
```

This results in duplicates for each row with a multiplicity greater than 1:

| N_M | L   |
|-----|-----|
| 1:1 | L3  |
| 1:2 | L4  |
| 1:2 | L4  |
| 2:1 | L5A |
| 2:1 | L5B |

In contrast, a **SEMI JOIN**:

```sql
SELECT COALESCE(LeftNM.N_M, RightNM.N_M) as N_M,
       LeftNM.Id as L 
FROM LeftNM SEMI JOIN RightNM 
ON LeftNM.N_M = RightNM.N_M;
```

Only returns unique values for each match:
??x

The result includes only the first occurrence of a match:

| N_M | L   |
|-----|-----|
| 1:1 | L3  |
| 1:2 | L4  |
| 2:1 | L5A |

This is because the SEMI JOIN operation does not retain duplicates, ensuring that each row with multiple matches appears only once in the result set.
x??

---

#### STREAM Renderings of INNER Join and SEMI Join

Background context explaining how stream renderings provide insights into which rows are filtered out.

:p How do stream renderings help differentiate between INNER JOIN and SEMI JOIN results?
??x

Stream renderings show the intermediate steps and highlight how duplicate rows are handled differently in each join type. Specifically, they display all incoming rows and indicate when a row is filtered out due to duplicates.

For an INNER JOIN:
```sql
SELECT COALESCE(LeftNM.N_M, RightNM.N_M) as N_M,
       LeftNM.Id as L,
       Sys.EmitTime as Time,
       Sys.Undo as Undo 
FROM LeftNM INNER JOIN RightNM 
ON LeftNM.N_M = RightNM.N_M;
```

Example output:

| N_M | L   | Time  | Undo |
|-----|-----|-------|------|
| 1:2 | null | 12:03 |      |
| 1:2 | null | 12:04 |      |
| 1:2 | null | 12:05 | undo |
| 1:2 | L4   | 12:05 |      |

Here, duplicate rows (1:2) are filtered out after the first occurrence.

For a SEMI JOIN:
```sql
SELECT COALESCE(LeftNM.N_M, RightNM.N_M) as N_M,
       LeftNM.Id as L,
       Sys.EmitTime as Time,
       Sys.Undo as Undo 
FROM LeftNM SEMI JOIN RightNM 
ON LeftNM.N_M = RightNM.N_M;
```

Example output:

| N_M | L   | Time  | Undo |
|-----|-----|-------|------|
| 1:2 | L4   | 12:05 |      |

Only the first occurrence of a match is included, and any subsequent duplicates are discarded.

This stream rendering provides clarity on how the join operation processes multiple matches.
x??

---

#### Multiplicative Nature of Joins

Background context explaining the multiplicative nature of joins with examples involving different cardinalities (e.g., 2:2, 3:3).

:p Explain the multiplicative nature of joins in datasets with multiple rows matching the same predicate.
??x

The multiplicative nature of joins arises from the fact that if a row in one table matches multiple rows in another table, each combination of these matches is included in the result set. This can lead to an exponential increase in the number of resulting rows.

For example:
- A join with cardinality 2:2 (two matching pairs) results in four combinations.
- A join with cardinality 3:3 (three matching triples) results in nine combinations.

This is illustrated by examples like:

```sql
SELECT COALESCE(LeftNM.N_M, RightNM.N_M) as N_M,
       LeftNM.Id as L 
FROM LeftNM INNER JOIN RightNM 
ON LeftNM.N_M = RightNM.N_M;
```

With 2:2 cardinality:
- If `LeftNM` has two rows and `RightNM` has two matching rows for each, the result will have four combinations.

```sql
SELECT COALESCE(LeftNM.N_M, RightNM.N_M) as N_M,
       LeftNM.Id as L 
FROM LeftNM SEMI JOIN RightNM 
ON LeftNM.N_M = RightNM.N_M;
```

With 2:2 cardinality:
- The result will still include only unique values, but the stream rendering shows all incoming rows and filters out duplicates.
??x

This example helps illustrate how joins process multiple matches, with the inner join including all combinations (resulting in duplicates), while the SEMI join retains only the first occurrence of each match.

For a more concrete understanding:
```sql
CREATE TABLE LeftNM (
    N_M INT,
    Id VARCHAR(10)
);

INSERT INTO LeftNM VALUES 
('1:2', 'L4'), ('1:3', 'L5');

CREATE TABLE RightNM (
    N_M INT,
    Id VARCHAR(10)
);

INSERT INTO RightNM VALUES
('1:2', 'R6'), ('1:3', 'R7'), ('1:3', 'R8');
```

The inner join will result in:
```sql
SELECT COALESCE(LeftNM.N_M, RightNM.N_M) as N_M,
       LeftNM.Id as L 
FROM LeftNM INNER JOIN RightNM 
ON LeftNM.N_M = RightNM.N_M;
```
Result:

| N_M | L   |
|-----|-----|
| 1:2 | null |
| 1:2 | R6  |
| 1:3 | null |
| 1:3 | R7  |
| 1:3 | R8  |

The SEMI join will result in:
```sql
SELECT COALESCE(LeftNM.N_M, RightNM.N_M) as N_M,
       LeftNM.Id as L 
FROM LeftNM SEMI JOIN RightNM 
ON LeftNM.N_M = RightNM.N_M;
```
Result:

| N_M | L   |
|-----|-----|
| 1:2 | null |
| 1:3 | R7  |

In the SEMI join, only unique matches are returned.
x??

#### Streaming Joins and Their Functionality

Background context: The passage explains that streaming joins operate similarly to how we understand joins between streams and tables, but they capture the historical evolution of join operations over time. This is contrasted with join tables which only represent a snapshot at a particular point in time.

:p How do streaming joins differ from traditional join tables?
??x
Streaming joins maintain a history of the join operation as it evolves over time, whereas traditional join tables capture a single snapshot. The streaming approach allows for continuous processing and updating of results as new data arrives.
x??

---

#### Core Join Primitive: FULL OUTER Join

Background context: The core underlying join primitive is the FULL OUTER JOIN, which combines all joined and unjoined rows from both relations into a single result set.

:p What is the primary type of join operation that underlies other variants?
??x
The FULL OUTER JOIN is the fundamental join operation. It collects together all the joined and unjoined rows in a relation. Other types such as LEFT OUTER, RIGHT OUTER, INNER, ANTI, and SEMI are derived by adding additional filtering conditions on top of this full outer join.
x??

---

#### Variants of Join Operations

Background context: The passage lists several types of joins (LEFT OUTER, RIGHT OUTER, INNER, ANTI, and SEMI) which all add an extra layer of filtering after the FULL OUTER JOIN operation.

:p How do LEFT OUTER, RIGHT OUTER, INNER, ANTI, and SEMI join operations relate to the FULL OUTER JOIN?
??x
These join types are built on top of the FULL OUTER JOIN by applying additional filters. For example:
- A LEFT OUTER JOIN keeps all rows from the left relation and adds matching right side rows if they exist.
- A RIGHT OUTER JOIN does the same but for the right relation instead.
- An INNER JOIN only returns rows where there is a match in both relations.
- ANTI join returns only rows that do not have matches on the right-hand side, effectively returning the complement of an INNER JOIN.
- SEMI join returns only the left side row when a match occurs.

The FULL OUTER JOIN provides the base set of all possible joins and unjoins between the two tables, from which other types can be derived through filtering.
x??

---

#### Windowed Joins

Background context: Windowed joins extend the concept of streaming joins by partitioning time in meaningful ways. This allows for more precise control over how data is joined based on specific time intervals.

:p Why are windowed joins useful?
??x
Windowed joins are beneficial because they allow for time-partitioned processing, which can be crucial for:
- Partitioning events into logical periods (e.g., daily windows).
- Improving performance by limiting the range of data considered in a join.
- Handling outer joins where one side might never show up due to temporal validity constraints.

Windowing also helps with timing out unjoined rows when dealing with unbounded data sources, which is not possible without it for classic batch processing approaches.
x??

---

#### Temporal Validity Joins

Background context: The passage mentions that while many streaming systems support fixed windows, there's a need for more sophisticated ways of partitioning time in joins, particularly temporal validity joins. These are discussed as an advanced topic.

:p What is the significance of temporal validity joins?
??x
Temporal validity joins allow for more precise control over data freshness and relevance by defining specific periods during which data is considered valid. This feature is not natively supported by most streaming systems today but can be highly useful in scenarios where historical data needs to be joined with current or future data within certain time windows.

For example, if you want to join sales data from a specific quarter with inventory levels from the same period, temporal validity joins would ensure that only relevant and current data is used for each join operation.
x??

---

#### Timing Out Unjoined Rows in Joins

Background context: The passage explains how windowed joins can help time out unjoined rows by comparing watermarks to the end of the window. This mechanism ensures that after a certain point, any remaining unprocessed rows can be safely considered as timeouts.

:p How do watermarks and windows contribute to timing out unjoined rows?
??x
Watermarks in streaming systems provide a way to gauge the completeness of input sources over time. By applying these watermarks within windowed joins, you can define when it is safe to timeout unjoined rows. Specifically:
- When the watermark passes the end of the window, all remaining data in that window can be considered complete.
- At this point, any unprocessed rows can be safely timed out and their partial results materialized.

This approach is particularly useful for outer joins where one side might never appear due to temporal constraints, ensuring efficient processing without waiting indefinitely.
x??

---

---
#### Fixed Windows and Temporal Validity Joins

In temporal validity joins, time is introduced into the join criteria by using windowing techniques. This allows us to focus on data that falls within a specific time interval or "window." By adding these windows, we can control which rows are considered during the join process based on their timestamps.

When implementing fixed windows for joining tables, we use window functions such as `TUMBLE` to partition the dataset into intervals. The join then only considers pairs of records that fall within the same time interval or window.

:p What is a fixed window in the context of temporal validity joins?
??x
A fixed window in the context of temporal validity joins refers to dividing data into predefined, non-overlapping intervals based on timestamps. Each row is assigned to a specific time interval, and the join operation only considers pairs of records that fall within the same time interval.
```
// Example usage of TUMBLE function
SELECT 
    TABLE *, 
    TUMBLE(Time, INTERVAL '5' MINUTE) as Window 
FROM Left;
```
x??

---
#### Join Criteria with Time Windows

In temporal validity joins, the traditional equality join condition `Left.Num = Right.Num` is expanded to include time window conditions. Specifically, a row from Table A (Left) and a row from Table B (Right) are considered for joining only if they fall within the same specified time interval.

The join criteria now becomes:
```
Left.Num = Right.Num AND Left.Window = Right.Window
```

:p How does the traditional join condition change in temporal validity joins?
??x
In temporal validity joins, the traditional equality join condition `Left.Num = Right.Num` is extended to include a window condition. This means that for two records to be joined, not only must they match on some key (like `Num`), but also their time windows must overlap or coincide.

This expanded condition ensures that we are only joining data within the same temporal interval.
```
// Example join with window equality
FROM Left
FULL OUTER JOIN Right 
ON L.Num = R.Num AND TUMBLE(Left.Time, INTERVAL '5' MINUTE) = TUMBLE(Right.Time, INTERVAL '5' MINUTE);
```
x??

---
#### Windowed Join Example

In the provided example, two tables (Left and Right) are windowed into five-minute fixed intervals using the `TUMBLE` function. This results in each row being associated with a specific time interval.

The original join on `Num` is extended to include window equality to ensure that only rows within the same time interval are joined.

:p What happens when applying temporal validity joins to our example tables?
??x
When applying temporal validity joins, the join condition expands from simply matching keys (like `Num`) to also including a check for the time windows in which these records fall. In the provided example, this means that:

- The original join on `Left.Num = Right.Num` is extended to:
  ```
  Left.Num = Right.Num AND TUMBLE(Left.Time, INTERVAL '5' MINUTE) = TUMBLE(Right.Time, INTERVAL '5' MINUTE)
  ```

This ensures that only records from the same time interval are joined. For instance, `L2` and `R2` do not fall within the same five-minute window `[12:00, 12:05)`, so they are not joined.

```
// Example join with window equality
FROM Left
FULL OUTER JOIN Right 
ON L.Num = R.Num AND TUMBLE(Left.Time, INTERVAL '5' MINUTE) = TUMBLE(Right.Time, INTERVAL '5' MINUTE);
```
x??

---
#### Unwindowed vs. Windowed Joins

The provided tables illustrate the difference between unwindowed and windowed joins by showing how rows that do not fall within the same time interval are excluded from the join.

In the unwindowed join (left table), all records with matching `Num` values are joined, regardless of their timestamps:

```
| L1   | null |
|  L2   | R2    |
| L3   | R3   |
| null | R4   |
```

In the windowed join (right table), only rows that fall within the same five-minute interval are joined:

```
| L1   | null |
|  L2   | null |
| L3   | R3   |
| null | R4   |
```

:p How do unwindowed and windowed joins differ in terms of their output?
??x
Unwindowed joins (left table) consider all rows that match on the join key (`Num`), regardless of their timestamps. This means that records with different time intervals are still considered for joining.

In contrast, windowed joins (right table) only join rows that fall within the same specified time interval. This ensures that we focus on data relevant to a specific period, as defined by the `TUMBLE` function.

This results in a more granular and time-bound join operation:

```
// Example unwindowed join
SELECT 
    TABLE Left.Id as L, 
    Right.Id as R 
FROM Left 
FULL OUTER JOIN Right ON L.Num = R.Num;

// Example windowed join
SELECT 
    TABLE Left.Id as L, 
    Right.Id as R, 
    COALESCE(
        TUMBLE(Left.Time, INTERVAL '5' MINUTE),
        TUMBLE(Right.Time, INTERVAL '5' MINUTE)
    ) AS Window
FROM Left 
FULL OUTER JOIN Right 
ON L.Num = R.Num AND 
TUMBLE(Left.Time, INTERVAL '5' MINUTE) = TUMBLE(Right.Time, INTERVAL '5' MINUTE);
```
x??

#### Unwindowed vs Windowed FULL OUTER Join

Background context: This concept explains the difference between performing a `FULL OUTER JOIN` without any windowing and with windowing. Without windowing, all rows from both datasets are matched on the condition provided. With windowing, rows are matched only within the same time window.

:p How does a `FULL OUTER JOIN` differ when applied to unwindowed data compared to windowed data?
??x
In unwindowed `FULL OUTER JOIN`, all rows from both datasets (`Left` and `Right`) are considered for matching based on the join condition. If no match is found, NULLs are used to fill in the missing columns.

In windowed `FULL OUTER JOIN`, data is partitioned into time windows, and a row from one dataset is matched with rows from the other dataset only if they fall within the same time window. Any unmatched rows outside of their respective windows result in NULL values for that window.

Example:
Unwindowed:
```sql
SELECT 
  L.Id as L,
  R.Id as R,
  Sys.EmitTime as Time,
  COALESCE(
    LEFT.Time, RIGHT.Time
  ) AS Window,
  FROM Left
  FULL OUTER JOIN Right
  ON L.Num = R.Num;
```

Windowed:
```sql
SELECT 
  L.Id as L,
  R.Id as R,
  Sys.EmitTime as Time,
  TUMBLE(Left.Time, INTERVAL '5' MINUTE) AS Window
FROM Left
FULL OUTER JOIN Right
ON L.Num = R.Num AND
   TUMBLE(Left.Time, INTERVAL '5' MINUTE) = 
   TUMBLE(Right.Time, INTERVAL '5' MINUTE);
```
x??

---

#### Unwindowed FULL OUTER Join Example

Background context: An example of an unwindowed `FULL OUTER JOIN` is provided to illustrate how rows are matched without considering any time constraints.

:p How does the unwindowed `FULL OUTER JOIN` handle unmatched rows?
??x
In an unwindowed `FULL OUTER JOIN`, if a row in `Left` does not find a match in `Right`, it will have NULL values for columns from `Right`. Similarly, if a row in `Right` does not find a match in `Left`, it will have NULL values for columns from `Left`.

Example:
```sql
12:10> SELECT 
      L.Id as L,
      R.Id as R,
      Sys.EmitTime as Time,
      COALESCE(
        TUMBLE(LEFT.Time, INTERVAL '5' MINUTE), 
        TUMBLE(RIGHT.Time, INTERVAL '5' MINUTE)
      ) AS Window
FROM Left
FULL OUTER JOIN Right
ON L.Num = R.Num;
```

This query ensures that all rows from both datasets are considered for matching based on the `Num` column. Any unmatched rows will result in NULL values.

Output:
```plaintext
-----------------------------
| L    | R    | Time  | Window |
-----------------------------
| null | R2   | 12:01 |       |
| L1   | null | 12:02 | [12:00, 12:05)|
| L3   | null | 12:03 | [12:00, 12:05)|
| L3   | null | 12:04 | [12:00, 12:05)|
| undo | R4   | 12:05 |       |
|  l2  | r2   | 12:06 |       |
-----------------------------
```
x??

---

#### Windowed FULL OUTER Join Example

Background context: An example of a windowed `FULL OUTER JOIN` is provided to illustrate how rows are matched only within the same time window.

:p How does the windowed `FULL OUTER JOIN` handle unmatched rows?
??x
In a windowed `FULL OUTER JOIN`, data is partitioned into fixed intervals (time windows), and a row from one dataset can match with rows from another dataset only if they fall within the same time window. Any unmatched rows outside of their respective windows will result in NULL values for that window.

Example:
```sql
12:10> SELECT 
      L.Id as L,
      R.Id as R,
      Sys.EmitTime as Time,
      TUMBLE(LEFT.Time, INTERVAL '5' MINUTE) AS Window
FROM Left
FULL OUTER JOIN Right
ON L.Num = R.Num AND 
   TUMBLE(LEFT.Time, INTERVAL '5' MINUTE) = 
   TUMBLE(RIGHT.Time, INTERVAL '5' MINUTE);
```

This query ensures that rows are matched only if they fall within the same time window. Any unmatched rows will result in NULL values.

Output:
```plaintext
-----------------------------
| L    | R    | Time  | Window           |
-----------------------------
| null | r2   | 12:01 | [12:00, 12:05)   |
| l1   | null | 12:02 | [12:00, 12:05)   |
| l3   | null | 12:03 | [12:00, 12:05)   |
| undo | null | 12:04 | [12:00, 12:05)   |
| r3   | l3   | 12:04 | [12:00, 12:05)   |
| null | r4   | 12:05 | [12:05, 12:10)   |
| l2   | null | 12:06 | [12:05, 12:10)   |
-----------------------------
```
x??

---

#### Windowed LEFT OUTER Join Example

Background context: An example of a windowed `LEFT OUTER JOIN` is provided to illustrate how it differs from the unwindowed version.

:p How does the windowed `LEFT OUTER JOIN` handle unmatched rows?
??x
In a windowed `LEFT OUTER JOIN`, only rows from the left dataset (`Left`) are considered for matching with the right dataset (`Right`). If no match is found, columns from the right dataset will have NULL values. The join condition must be within the same time window.

Example:
```sql
12:10> SELECT 
      L.Id as L,
      R.Id as R,
      TUMBLE(LEFT.Time, INTERVAL '5' MINUTE) AS Window
FROM Left
LEFT OUTER JOIN Right
ON L.Num = R.Num AND 
   TUMBLE(Left.Time, INTERVAL '5' MINUTE) = 
   TUMBLE(Right.Time, INTERVAL '5' MINUTE);
```

This query ensures that rows from the `Left` dataset are matched with rows from the `Right` dataset only if they fall within the same time window. Any unmatched rows will result in NULL values for columns from the right dataset.

Output:
```plaintext
-------------------------
| L    | R    | Window   |
-------------------------
| l1   | null | [12:00, 12:05)|
| l3   | r3   | [12:00, 12:05)|
| l2   | null | [12:05, 12:10)|
-------------------------
```
x??

---

