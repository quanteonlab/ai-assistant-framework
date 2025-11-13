# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 32)

**Starting Chapter:** Improving Query Performance

---

#### SQL Compilation and Bytecode Conversion
Background context: The database engine compiles the SQL, parsing the code to check for proper semantics and ensuring that the database objects referenced exist and that the current user has the appropriate access. After validation, the SQL code is converted into bytecode, which expresses the steps that must be executed on the database in an efficient, machine-readable format.

:p What happens during the compilation and conversion of SQL code?
??x
During the compilation phase, the SQL statement is parsed to ensure it follows the correct syntax and semantics. It also checks if all referenced objects (tables, views, etc.) exist and if the current user has the necessary permissions to access them.

After validation, the SQL query is converted into bytecode. This bytecode contains instructions that are machine-readable and efficient for execution by the database engine.
??x
---
#### Query Optimizer Overview
Background context: The query optimizer analyzes the bytecode to determine how to execute the query efficiently. It reorders and refactors steps to use available resources optimally, aiming to minimize costs while maximizing performance.

:p What is the role of the query optimizer in executing SQL queries?
??x
The query optimizer plays a crucial role by analyzing the bytecode generated from the SQL query. Its main tasks include:

- **Reordering Steps**: It rearranges the steps involved in executing the query to ensure they are performed in the most efficient order.
- **Refactoring Queries**: It modifies the query execution plan to better utilize available resources, such as indexes and data scans.

The goal is to find the least expensive way to execute the query while ensuring that performance requirements are met. This process involves evaluating various factors like join types, index usage, and scan sizes.
??x
---
#### Optimizing Joins in Queries
Background context: Joins are a fundamental aspect of combining datasets in data engineering. The choice of join strategy can significantly impact query performance. Prejoining data or maintaining normalized schemas with prejoined tables for common use cases can improve efficiency.

:p What techniques can be used to optimize joins in SQL queries?
??x
Several techniques can be employed to optimize joins in SQL queries:

- **Prejoin Data**: If the same data is repeatedly joined, it makes sense to prejoin this data. This reduces redundant computations and speeds up query execution.
- **Schema Optimization**: Consider relaxing normalization conditions to widen tables or use newer data structures like arrays or structs to replace frequently joined entity relationships.

These changes can improve performance by reducing the computational load during query execution.
??x
---
#### Improving Query Performance through Materialized Views
Background context: Materialized views are precomputed results that store the outcome of a complex query. Using materialized views for common queries can significantly enhance performance since they bypass the need to compute the entire result set each time.

:p How do materialized views help in improving query performance?
??x
Materialized views improve query performance by storing precomputed results. Instead of running the full query each time, the database fetches data from the materialized view, which is already computed and optimized. This reduces the need for repetitive computations and speeds up response times.

Here’s an example of creating a materialized view in SQL:
```sql
CREATE MATERIALIZED VIEW sales_summary AS
SELECT product_id, SUM(quantity) as total_quantity
FROM sales
GROUP BY product_id;
```
This view can be refreshed periodically or on-demand to ensure its data remains current.
??x

#### Complex Join Logic and Performance Optimization
Background context: In complex join operations, especially when dealing with many-to-many relationships, the number of rows can significantly increase, leading to performance degradation. Databases may struggle to handle such large result sets efficiently. PostgreSQL allows creating indexes on computed fields, which can optimize certain types of queries.

:p What is an example where indexing a computed field might be useful in join conditions?
??x
Indexing a computed field like `lower()` in PostgreSQL can help when the query involves converting string fields to lowercase for comparison. For instance, if you have a `name` column and want to perform case-insensitive joins or filters:

```sql
CREATE INDEX idx_name_lower ON your_table(lower(name));
```
This index will allow PostgreSQL's optimizer to use it effectively in queries where the lower() function is applied.

x??

---

#### Row Explosion in Joins
Background context: Row explosion occurs when join keys have many-to-many matches, leading to a cross-join of all matching rows. This can result in an exponential increase in the number of output rows, consuming significant database resources and potentially causing queries to fail.

:p How does row explosion occur in joins?
??x
Row explosion happens due to repeated values in join keys from both tables involved in the join operation. For example, if a value in table A repeats five times and a value in table B repeats 10 times, the cross-join will generate $5 \times 10 = 50$ rows.

```sql
-- Example of a join that might lead to row explosion
SELECT * FROM tableA JOIN tableB ON (tableA.key = tableB.key);
```
If there are many such repeated values, it can drastically increase the output size and resource usage.

x??

---

#### Optimizing Joins with Predicate Reordering
Background context: Not all databases can reorder joins and predicates. However, reordering can significantly reduce computational resources required by a query, especially when early stages of the join generate a large number of rows.

:p Why is predicate reordering important for performance?
??x
Predicate reordering is crucial because it allows the database to apply filtering conditions as early in the query execution as possible, reducing the intermediate result set size. This can prevent row explosion from affecting later stages where more complex processing might occur.

For example, if a join generates a large number of rows and a predicate could filter out many of them, moving that predicate earlier in the execution plan can save resources:

```sql
-- Original query
SELECT * FROM tableA JOIN (SELECT * FROM tableB WHERE condition) AS filtered_tableB ON (tableA.key = filtered_tableB.key);

-- Optimized version with reordering
SELECT * FROM (SELECT * FROM tableA WHERE another_condition) AS filtered_tableA JOIN tableB ON (filtered_tableA.key = tableB.key);
```

In the optimized version, the `another_condition` is applied earlier, potentially reducing the number of rows before the join operation.

x??

---

#### Common Table Expressions (CTEs)
Background context: CTEs allow breaking down complex queries into more manageable parts. They are useful for readability and can sometimes provide performance benefits over temporary tables or nested subqueries.

:p What is the advantage of using CTEs in complex queries?
??x
Using CTEs makes complex queries easier to read and understand by breaking them down into smaller, reusable components. This not only improves maintainability but also can lead to better performance if the database can optimize these parts more effectively than nested subqueries or temporary tables.

Example of using a CTE:
```sql
WITH filtered_tableA AS (
    SELECT * FROM tableA WHERE condition1
),
filtered_tableB AS (
    SELECT * FROM tableB WHERE condition2
)
SELECT * FROM filtered_tableA JOIN filtered_tableB ON (filtered_tableA.key = filtered_tableB.key);
```

Here, `filtered_tableA` and `filtered_tableB` are CTEs that apply conditions early in the query process, making the main query simpler.

x??

---

#### Understanding Query Optimization with Explain Plans
Background context: The database's query optimizer plays a crucial role in determining how queries are executed. Explain plans provide insights into how the optimizer decides to execute a query, including resource usage and performance statistics at each stage.

:p How can you use EXPLAIN plans to understand your query’s performance?
??x
EXPLAIN plans help identify bottlenecks in query execution by showing the optimization steps taken by the database. They reveal details like used tables, indexes, cache usage, and resource consumption per step. Visual representations of these plans can be invaluable for troubleshooting and optimizing queries.

Example of using EXPLAIN:
```sql
-- Example explain plan command
EXPLAIN SELECT * FROM tableA JOIN tableB ON (tableA.key = tableB.key);
```
The output will show the execution path, helping you understand where the optimizer is spending most resources.

x??

---

#### Avoid Full Table Scans
Background context: When querying a database, it's essential to be mindful of the data being scanned. Full table scans are inefficient and can be costly in terms of performance and financial expense. Selecting only necessary columns and rows is crucial for optimal query performance.

:p How do you avoid full table scans?
??x
To avoid full table scans, you should:
1. Use predicates to filter out unnecessary rows.
2. Select only the required columns using `SELECT column1, column2`.
3. Consider clustering keys or partitions in column-oriented databases like Snowflake and BigQuery.

For example, if your query involves filtering by a specific value, use a predicate:

```sql
SELECT column1, column2 FROM table WHERE condition;
```

x??

---

#### Pruning Strategies for Column-Oriented Databases
Background context: In column-oriented databases, pruning is more effective when you can limit the columns that are accessed. This reduces the amount of data scanned and improves performance.

:p How does pruning work in column-oriented databases?
??x
In column-oriented databases like Snowflake and BigQuery:
1. Use `SELECT` statements to specify only the necessary columns.
2. For very large tables, define a cluster key to order the table's data for efficient access.
3. Partition tables into smaller segments to query specific partitions instead of the entire dataset.

For example:

```sql
-- Selecting specific columns and using a cluster key
SELECT column1, column2 FROM table WHERE condition;

-- Defining a cluster key (Snowflake-specific)
ALTER TABLE table SET CLUSTER BY (column);
```

x??

---

#### Pruning Strategies for Row-Oriented Databases
Background context: In row-oriented databases, pruning is often achieved through indexes. Indexes allow the database to quickly locate and retrieve specific rows without scanning the entire table.

:p How does pruning work in row-oriented databases?
??x
In row-oriented databases:
1. Create indexes on columns that are frequently used in `WHERE` clauses.
2. Avoid over-indexing, as too many indexes can degrade performance.
3. Use composite indexes to combine multiple columns for even better performance.

For example:

```sql
-- Creating an index (SQL Server)
CREATE INDEX idx_column ON table(column);
```

x??

---

#### Understanding Database Commits and Transactions
Background context: A database commit is a transaction that makes changes permanent within the database. Ensuring transactions are ACID-compliant helps maintain data consistency, especially in scenarios where multiple operations might be performed simultaneously.

:p What should you know about how your database handles commits?
??x
To understand how your database handles commits:
1. Determine if it supports ACID compliance.
2. Understand transaction isolation levels to avoid dirty reads.
3. Be aware of the impact of frequent small commits on storage and performance.

For example, checking for ACID compliance in PostgreSQL:

```sql
-- Check ACID compliance (PostgreSQL)
SELECT * FROM pg_database WHERE datname = 'your_db_name';
```

x??

---

#### Impact of Commits on Storage and Performance
Background context: Committing changes to a database can affect storage management. Large numbers of small commits might lead to cluttered file systems, requiring periodic vacuuming.

:p How do frequent small commits impact the database?
??x
Frequent small commits in databases like PostgreSQL can:
1. Create new files representing the state after each commit.
2. Retain old files for failure checkpoint references.
3. Lead to storage space issues and potential performance degradation if not managed properly.

For example, handling small commits:

```sql
-- Handling small commits (PostgreSQL)
BEGIN;
UPDATE table SET column = value WHERE condition;
COMMIT;
```

x??

---

#### PostgreSQL Row Locking Mechanism
Background context explaining how PostgreSQL handles row locking during read and write operations. The approach can degrade performance due to blocking reads and writes on certain rows, making it less suitable for large-scale analytics applications.

:p What is the main disadvantage of using PostgreSQL for large-scale analytics queries?
??x
The main disadvantage of using PostgreSQL for large-scale analytics queries is that it requires row locking, which blocks both reads and writes to specific rows. This can lead to performance degradation as multiple concurrent operations may wait for locks, thereby slowing down read and write processes.

```java
// Example illustrating how a read operation might block due to row locking
public class ReadOperation {
    public void performRead(String tableName) {
        // Code that acquires a lock on rows during the read operation
        try (Connection conn = DriverManager.getConnection(DB_URL)) {
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery("SELECT * FROM " + tableName);
            while (rs.next()) {
                // Process each row here
            }
        } catch (SQLException e) {
            // Handle exceptions
        }
    }
}
```
x??

---

#### BigQuery’s Commit Model
Background context explaining how Google BigQuery handles data commits and reads, ensuring a consistent snapshot for the duration of a query. It does not lock tables during read operations but allows only one write operation at a time.

:p How does Google BigQuery handle read queries to ensure consistency?
??x
Google BigQuery ensures consistency by reading from the latest committed snapshot of the table when a read query is issued. Regardless of how long the query runs, it will always use this snapshot and not see any subsequent changes. This approach avoids row locking during reads but enforces write concurrency by queuing multiple write operations.

```java
// Example illustrating BigQuery's commit model for reads and writes
public class BigQueryReadWriteExample {
    public void performRead() {
        // Code to initiate a read query using the latest committed snapshot
        try (Connection conn = DriverManager.getConnection(DB_URL)) {
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery("SELECT * FROM my_table");
            while (rs.next()) {
                // Process each row here
            }
        } catch (SQLException e) {
            // Handle exceptions
        }
    }

    public void performWrite() throws SQLException {
        // Code to queue write operations
        try (Connection conn = DriverManager.getConnection(DB_URL)) {
            Statement stmt = conn.createStatement();
            String insertQuery = "INSERT INTO my_table (column1, column2) VALUES ('value1', 'value2')";
            int rowsAffected = stmt.executeUpdate(insertQuery);
            if (rowsAffected == 0) {
                // Handle no rows affected
            }
        }
    }
}
```
x??

---

#### MongoDB’s Variable Consistency Model
Background context explaining how MongoDB offers configurable consistency options at the database and query levels, providing high write concurrency but risking data loss under heavy traffic.

:p What is a key feature of MongoDB's approach to consistency?
??x
A key feature of MongoDB's approach to consistency is its variable consistency model, which allows engineers to configure different levels of consistency both globally for the database and on an individual query basis. This flexibility enables high write concurrency but can lead to data loss if the system gets overwhelmed with traffic.

```java
// Example illustrating a read operation in MongoDB with configurable consistency
public class MongoDBReadExample {
    public void performRead(String collectionName) {
        // Code using MongoDB's API to set read preference and initiate a query
        MongoClient mongoClient = MongoClients.create("mongodb://localhost:27017");
        MongoDatabase database = mongoClient.getDatabase("mydatabase");
        MongoCollection<Document> collection = database.getCollection(collectionName);
        
        FindIterable<Document> iterable = collection.find();
        for (Document document : iterable) {
            // Process each document here
        }
    }
}
```
x??

---

#### Importance of Choosing the Right Technology
Background context emphasizing the importance of choosing appropriate technology and configuring it correctly to achieve success or avoid massive failure. This includes understanding commit and consistency models.

:p Why is it important to choose the right database technology for a project?
??x
Choosing the right database technology is crucial because different technologies have varying strengths and weaknesses. Technologies such as PostgreSQL, BigQuery, and MongoDB are each suited to specific types of workloads and configurations. Understanding these differences helps in selecting the best fit for a given application or workload. Proper configuration can significantly impact whether a project succeeds or fails.

```java
// Example comparing technology choices for different use cases
public class TechnologySelectionExample {
    public void selectTechnology(String requirement) {
        if (requirement.equals("Large-scale Analytics")) {
            System.out.println("Use BigQuery");
        } else if (requirement.equals("High Write Concurrency")) {
            System.out.println("Use MongoDB with appropriate consistency settings");
        } else if (requirement.equals("Consistent Read-Write Operations")) {
            System.out.println("Use PostgreSQL with careful configuration for concurrent access");
        }
    }
}
```
x??

---

#### Concept: Vacuuming Dead Records
Background context explaining the concept. Transactions can create new records while retaining old ones as pointers, leading to accumulated dead records over time. These records should be removed through a process called vacuuming to free up space and improve query performance.

Vacuuming is crucial in databases to manage storage efficiency and query optimization. In object storage-backed databases like BigQuery, Snowflake, and Databricks, old data retention impacts storage costs. Vacuum operations are managed differently based on the database type:
- **BigQuery** uses a fixed seven-day history window.
- **Snowflake** allows controlling table snapshots via "time-travel" intervals.
- **Databricks** retains data indefinitely until manually vacuumed.

Vacuuming can impact performance and available storage in Amazon Redshift, so users need to balance the benefits with potential drawbacks. Vacuum operations run automatically but may be manually initiated for tuning purposes.

:p What is vacuuming in database management?
??x
Vacuuming is a process that removes dead records from a database, freeing up space and improving query performance by ensuring that only relevant data remains.
x??

---
#### Concept: Impact of Vacuuming on Storage Costs
Background context explaining the concept. In databases backed by object storage (BigQuery, Snowflake, Databricks), old data retention is managed differently to control storage costs.

In **Snowflake**, users set a "time-travel" interval that determines how long table snapshots are retained before they are auto vacuumed.
In **Databricks**, data is generally retained indefinitely until manually vacuumed. This allows for better management of direct S3 storage costs as it provides flexibility in controlling the amount of data stored.

:p How does Snowflake manage old data retention?
??x
Snowflake manages old data retention through a "time-travel" interval, which determines how long table snapshots are retained before they are auto vacuumed.
x??

---
#### Concept: Vacuuming in Amazon Redshift
Background context explaining the concept. Amazon Redshift handles its cluster disks differently and vacuuming can impact performance and available storage.

Vacuum operations run automatically but users may sometimes want to run them manually for tuning purposes. In Redshift, engineers need to be aware of how vacuuming impacts their system's storage efficiency and query performance.

:p How does vacuuming impact Amazon Redshift?
??x
Vacuuming in Amazon Redshift can impact performance and available storage. While it runs automatically, users may manually initiate vacuum operations for tuning purposes.
x??

---
#### Concept: Vacuuming in Relational Databases (PostgreSQL and MySQL)
Background context explaining the concept. Relational databases like PostgreSQL and MySQL require more attention to vacuuming due to high transactional activity.

With large numbers of transactional operations, dead records can accumulate rapidly, necessitating regular vacuuming to maintain optimal performance and storage efficiency.

:p Why is vacuuming critical in relational databases?
??x
Vacuuming is critical in relational databases like PostgreSQL and MySQL because frequent transactions can lead to a rapid accumulation of dead records. Regular vacuuming helps maintain optimal performance and storage efficiency.
x??

---

#### Query Caching for Cost Efficiency

Background context: When running a query on a database that charges based on data retrieval, repeatedly executing the same query can lead to significant costs. To mitigate this, many cloud OLAP databases cache the results of frequently run queries.

:p What is the main benefit of leveraging cached query results?
??x
The main benefit of leveraging cached query results is reducing the cost associated with running the same query multiple times. Instead of executing a query that may have taken 40 seconds to retrieve and process data, subsequent runs can return cached results almost instantaneously.

```java
// Example of a simple caching mechanism in Java
public class QueryCache {
    private Map<String, Result> cache = new HashMap<>();

    public Result getCachedQueryResult(String query) {
        if (cache.containsKey(query)) {
            System.out.println("Returning result from cache.");
            return cache.get(query);
        } else {
            System.out.println("Executing cold query.");
            Result result = executeColdQuery(query); // Simulate executing a cold query
            cache.put(query, result);
            return result;
        }
    }

    private Result executeColdQuery(String query) {
        // Logic to run the query and get results
        return new Result();
    }
}
```
x??

---

#### Fast-Follower Approach for Streaming Data Queries

Background context: In scenarios where streaming data is involved, traditional batch processing methods are not sufficient. The fast-follower approach uses a separate database as a fast follower to a production database, enabling real-time analytics with minimal impact on the production system.

:p What is the primary advantage of using a fast-follower pattern for streaming queries?
??x
The primary advantage of using a fast-follower pattern for streaming queries is that it allows serving real-time analytics while minimizing the load and potential disruption to the production database. This approach ensures that the production workload remains unaffected by analytical workloads.

```java
// Pseudocode for implementing a basic fast-follower pattern
public class FastFollower {
    private Database productionDB;
    private Database analyticsDB;

    public void setupFastFollower() {
        // Set up continuous CDC to keep the analytics DB in sync with the production DB
        ContinuousCDC cdc = new ContinuousCDC(productionDB, analyticsDB);
        cdc.start();

        // Query the analytics database for real-time insights
        String query = "SELECT * FROM analytics_table";
        Result result = analyticsDB.executeQuery(query);
    }
}

class ContinuousCDC {
    private Database source;
    private Database target;

    public void start() {
        while (true) {
            Record record = source.fetchRecord(); // Simulate fetching a record from the production DB
            if (record != null) {
                target.insert(record); // Insert the record into the analytics DB to keep it in sync
            }
        }
    }
}
```
x??

---

#### Materialized Views for Query Caching

Background context: Materialized views provide an alternative form of query caching. A materialized view stores the result of a query as a physical table, allowing for faster retrieval and reducing the need to recompute results.

:p How do materialized views differ from simple in-memory caches?
??x
Materialized views store the result of a query as a physical table, which can be queried like any other database table. This differs from simple in-memory caches because materialized views persist their results on disk and can be queried directly, whereas in-memory caches are temporary and may not provide direct access to stored data.

```java
// Pseudocode for creating a materialized view
public class MaterializedViewManager {
    private Database db;

    public void createMaterializedView(String query) {
        // Create a physical table with the result of the given query
        String tableName = "materialized_view_" + System.currentTimeMillis();
        Table viewTable = db.createTable(tableName);
        
        // Execute the query and populate the materialized view table
        ResultSet resultSet = db.executeQuery(query);
        while (resultSet.hasNext()) {
            Row row = resultSet.next();
            viewTable.insert(row);
        }
    }
}
```
x??

---

#### Query Patterns for Streaming Data

Background context: When dealing with streaming data, traditional batch processing techniques are not suitable. Specialized query patterns like continuous CDC (Change Data Capture) enable real-time analytics by keeping an analytics database in sync with a production database.

:p What is the key feature of the fast-follower approach when querying streaming data?
??x
The key feature of the fast-follower approach when querying streaming data is that it allows serving real-time analytics with minimal impact on the production system. The analytics database acts as a replica of the production database, continuously synchronized through CDC to provide up-to-date information for queries.

```java
// Pseudocode for implementing continuous CDC in a fast-follower pattern
public class ContinuousCDC {
    private Database source;
    private Database target;

    public void start() {
        while (true) {
            Record record = source.fetchRecord(); // Simulate fetching a record from the production DB
            if (record != null) {
                target.insert(record); // Insert the record into the analytics DB to keep it in sync
            }
        }
    }
}
```
x??

---
#### CDC with a Fast-Follower Analytics Database
Background context: The fast-follower approach is used for computing trailing statistics on vast historical data with near real-time updates. However, it doesn’t fundamentally rethink batch query patterns; you are still running `SELECT` queries against the current table state and missing the opportunity to dynamically trigger events off changes in the stream.
:p What is the main limitation of using a fast-follower analytics database for trailing statistics?
??x
The main limitations include that it doesn't change how batch queries are executed, meaning you are still querying the current state of the table. It also misses the chance to automatically respond to data changes as they occur in real-time.
```java
// Example: Running a SELECT query on a fast-follower database
public void runBatchQuery(String sql) {
    Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/db", "user", "password");
    Statement stmt = connection.createStatement();
    ResultSet rs = stmt.executeQuery(sql);
    // Process the results
}
```
x??
---

#### Kappa Architecture Overview
Background context: The Kappa architecture, introduced in Chapter 3, treats all data as events and stores them as a stream rather than a table. This approach allows for longer retention periods (months or years) compared to purely real-time systems.
:p What is the key idea behind the Kappa architecture?
??x
The key idea of the Kappa architecture is to treat streaming storage as both a real-time transport layer and a database for retrieving and querying historical data. Data can be queried directly from this storage, allowing for longer retention periods compared to traditional batch processing systems.
```java
// Example: Using Kafka Streams API to process events in real-time
KStream<String, String> stream = builder.stream(topic);
stream.foreach((key, value) -> {
    // Process each event
});
```
x??
---

#### Streaming Windows in Kappa Architecture
Background context: In streaming systems, windows are used to process data in small batches based on dynamic triggers. Different types of windows include session, fixed-time, and sliding.
:p What is a window in the context of streaming processing?
??x
A window in streaming processing is a small batch of records processed together based on dynamic triggers. Windows can be session-based (processing data until an event ends), fixed-time (a predefined time interval), or sliding (processing overlapping data intervals).
```java
// Example: Using Apache Flink to define a sliding window
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
DataStream<MyEvent> stream = env.addSource(new MySourceFunction());
TimeWindow timeWindow = Time.seconds(10);
stream.timeWindow(timeWindow).reduce((event1, event2) -> {
    // Combine events in the window
    return new CombinedEvent(event1, event2);
});
```
x??
---

---
#### Session Windows
Session windows group events that occur close together, and filter out periods of inactivity when no events occur. A user session can be defined as any time interval with no activity gap exceeding a certain period (e.g., five minutes). The system processes these sessions dynamically by applying time conditions to user activity on web or desktop applications.
:p What is the definition of a session window?
??x
Session windows are used in both batch and streaming systems, where events are grouped based on their occurrence within close intervals. If there's an inactivity gap exceeding five minutes (or any other defined threshold), the current session ends, and statistics for that session are calculated.
For example, if a user is inactive for more than 5 minutes, their activity is considered as ending, and the system processes all events recorded during this session to calculate relevant metrics like page views or time spent on-site.

```java
// Pseudocode for detecting inactivity gaps in a streaming application
public void processEvent(Event event) {
    if (lastActivityTime == null || isMoreThanFiveMinutesInactive(event.time)) {
        closeCurrentSession();
        lastActivityTime = event.time;
    }
}
```
x??

---
#### Fixed-Time Windows (Tumbling Windows)
Fixed-time windows, also known as tumbling windows, feature fixed time periods that run on a fixed schedule and process all data since the previous window is closed. This type of window runs in intervals such as every 20 seconds or every hour.
:p What are fixed-time windows?
??x
Fixed-time windows (tumbling windows) break down the continuous stream of events into non-overlapping segments of predefined duration, processing each segment independently. For instance, a tumbling window with a 20-second interval will process all data from the start to the end of every 20 seconds.

```java
// Pseudocode for fixed-time window logic
public void processWindow(long startTime) {
    long endTime = startTime + windowSize; // e.g., 20 seconds
    List<Event> eventsInCurrentWindow = collectEventsFrom(startTime, endTime);
    calculateStatistics(eventsInCurrentWindow);
}
```
x??

---
#### Sliding Windows
Sliding windows bucket events into fixed time-length windows where separate windows might overlap. A new window is generated every few seconds (or minutes), allowing for more frequent processing compared to fixed-time windows.
:p What are sliding windows?
??x
Sliding windows divide the event stream into overlapping segments of a fixed duration, enabling continuous monitoring and analysis of recent events. For example, you can generate a 60-second window every 30 seconds, which means each new window overlaps with the previous one by half its length.

```java
// Pseudocode for sliding window logic
public void processWindow(long startTime) {
    long endTime = startTime + windowSize; // e.g., 60 seconds
    List<Event> eventsInCurrentWindow = collectEventsFrom(startTime, endTime);
    calculateStatistics(eventsInCurrentWindow);
}
```
x??

---
#### Handling Late-Arriving Data in Session Windows
Late-arriving data refers to situations where data points can arrive up to a certain time after the event they represent. For session windows, this can be handled by allowing late events that occur within a grace period (e.g., 5 minutes).
:p How do you handle late-arriving data in session windows?
??x
Late-arriving data is managed by defining a grace period during which late events are still considered valid for the current session. If an event arrives within this grace period, it's included in the current session window.

```java
// Pseudocode for handling late-arriving data
public void processEvent(Event event) {
    long currentTime = System.currentTimeMillis();
    if (isWithinGracePeriod(event.time, currentTime)) {
        // Include the event in the current session window
        addEventToSessionWindow(event);
    }
}
```
x??

---

#### Watermarks
Watermarks are thresholds used by a window to determine whether data is considered within the established time interval or late-arriving. If a piece of data arrives that is older than the watermark, it is treated as late-arriving data.

:p What is a watermark in stream processing?
??x
A watermark is a threshold in stream processing that helps manage out-of-order data and late-arriving events by defining an upper bound for data timestamps within a window. When data arrives with a timestamp older than the current watermark, it is considered late and may be discarded or processed differently depending on the system's configuration.
```java
// Pseudocode to demonstrate watermarking in a stream processing system
public class WatermarkExample {
    long watermark;

    public void processEvent(Event event) {
        if (event.timestamp > watermark) {
            // Process data within the current window
        } else {
            // Handle late-arriving data
        }
        updateWatermark(event.timestamp); // Update watermark based on the latest data processed
    }

    private void updateWatermark(long newTimestamp) {
        if (newTimestamp - watermark > someThreshold) {
            watermark = newTimestamp;
        }
    }
}
```
x??

---

#### Combining Streams with Other Data: Conventional Table Joins
Conventional table joins can be used to combine data from streams and tables. A stream can feed one or both of the two tables that are joined, allowing for dynamic updates based on incoming data.

:p How does a conventional table join work in the context of combining streams and static data?
??x
In the context of combining streams and static data, a conventional table join involves dynamically joining a stream with a traditional database (table). The stream can feed one or both tables, ensuring that real-time updates are reflected in the joined results.

For example:
- A stream of events is continuously fed into a database.
- During a join operation, each event from the stream is matched against existing data in the table(s).

This approach allows for dynamic enrichment and real-time analytics without requiring offline processing or materialized views.

```java
// Pseudocode to demonstrate conventional table joins with streams
public class TableJoinExample {
    public void processStreamEvent(Event event) {
        // Retrieve records from the database that match the stream event
        List<Record> matchingRecords = database.query("SELECT * FROM static_table WHERE id = " + event.id);
        
        // Join logic: merge stream data with retrieved records
        for (Record record : matchingRecords) {
            Result joinedResult = join(event, record);
            processJoinedResult(joinedResult);
        }
    }

    private Result join(Event event, Record record) {
        // Logic to combine event and record into a single result object
        return new Result(event, record);
    }

    private void processJoinedResult(Result result) {
        // Process the joined data
    }
}
```
x??

---

#### Enrichment in Streaming Systems
Enrichment involves joining a stream with other data sources to enhance the information contained within the stream. This is often done to provide more detailed or context-rich events.

:p What is enrichment in streaming systems?
??x
Enrichment in streaming systems refers to the process of combining data from a stream with additional, external data to provide enhanced and more valuable information. For example, an online retailer might receive product IDs and user IDs as part of an event stream but enrich this data by adding detailed product descriptions and user demographic information.

The enriched events can then be used for various purposes such as targeted marketing or personalized recommendations.

```java
// Pseudocode to demonstrate enrichment in a streaming system
public class EnrichmentExample {
    public void processEvent(Event event) {
        // Retrieve product details from an in-memory cache or database
        ProductDetails product = productCache.get(event.productId);
        
        // Retrieve user demographic data
        UserDemographicInfo user = userRepository.getUserDemographics(event.userId);
        
        // Combine the enriched information into a new event object
        EnrichedEvent enrichedEvent = enrichEventWithDetails(event, product, user);
        
        // Output the enriched event to another stream or storage
        outputEnrichedEvent(enrichedEvent);
    }

    private EnrichedEvent enrichEventWithDetails(Event event, ProductDetails product, UserDemographicInfo user) {
        return new EnrichedEvent(event, product.name, user.age, user.location);
    }

    private void outputEnrichedEvent(EnrichedEvent enrichedEvent) {
        // Output the enriched event to a new stream or storage
    }
}
```
x??

---

#### Stream-to-Stream Joining
Increasingly, streaming systems support direct joining of two streams. This allows for complex data transformations and analyses by combining events from different sources in real-time.

:p What is stream-to-stream joining?
??x
Stream-to-stream joining involves directly merging or combining data from two separate event streams to create a unified view or perform complex analysis on the combined data. This capability is particularly useful for scenarios where multiple types of events need to be analyzed together in real-time, such as integrating web event data with ad platform streaming data.

```java
// Pseudocode to demonstrate stream-to-stream joining
public class StreamToStreamJoinExample {
    public void joinStreams(Stream<WebEvent> webEvents, Stream<AdEvent> adEvents) {
        // Join the two streams on a common key (e.g., user ID)
        Stream<JoinedEvent> joinedStream = webEvents.join(adEvents, (webEvent, adEvent) -> webEvent.userId.equals(adEvent.userId),
            (webEvent, adEvent) -> new JoinedEvent(webEvent, adEvent));
        
        // Process the joined stream
        joinedStream.forEach(this::processJoinedEvent);
    }

    private void processJoinedEvent(JoinedEvent event) {
        // Perform analysis or actions based on the joined data
    }
}
```
x??

---

#### Streaming Data Joins Complexity
Background context explaining the concept. When streaming data is joined, different streams may arrive at varying latencies, leading to complications. For example, one stream might have a five-minute delay compared to another. Events like session close or offline events can be significantly delayed due to network conditions.
:p What challenges do streaming data joins face regarding latency and event timing?
??x
Streaming data joins face significant challenges because different streams may arrive at the join point with varying latencies. This means that one stream might lag behind others, leading to potential delays in processing related events. For instance, an ad platform's data might have a five-minute delay compared to other streams.
In addition, certain events can be delayed due to network conditions or other factors. A user's session close event could be delayed if the device is offline and only comes back online after the user regains mobile network access.

For example:
- Stream A: Ad Data (5-minute delay)
- Stream B: User Session Data

These delays mean that by the time a relevant event from Stream B arrives, it might not match with events in Stream A due to latency.
x??

---

#### Streaming Buffer Retention Interval
Background context explaining the concept. To manage these latencies and ensure timely processing of related events, streaming systems often use buffers. The buffer retention interval is configurable; setting a longer retention interval requires more storage but allows for joining more delayed events. Events in the buffer are eventually evicted after the retention period has passed.
:p What is a key factor in streaming joins that uses buffering to manage latency?
??x
A key factor in streaming joins that uses buffering to manage latency is the configurable buffer retention interval. This interval determines how long events remain in the buffer before being processed or removed. A longer retention interval allows for joining more delayed events but requires additional storage and resources.
For example, if an event from Stream B (a user session close) arrives after a five-minute delay due to network conditions, setting a longer buffer retention interval ensures that it can still be joined with corresponding events in Stream A (ad data).
x??

---

#### Data Modeling Importance
Background context explaining the concept. Data modeling is crucial for organizing and structuring data in a way that supports business needs. It involves choosing a coherent structure for data to ensure effective communication and workflow within an organization. Poor or absent data models can lead to redundant, mismatched, or incorrect data.
:p Why is data modeling important?
??x
Data modeling is important because it ensures that data is structured coherently to support the business logic and goals of an organization. A well-constructed data model captures how communication and work naturally flow within the organization, making data more useful for decision-making.

Poor or absent data models can lead to several issues:
1. **Redundant Data**: Duplicates and inconsistencies that complicate analysis.
2. **Mismatched Data**: Inaccurate data leading to incorrect business decisions.
3. **Incorrect Data**: Data that does not accurately reflect the real-world processes, causing confusion.

For example, a poorly modeled database might have redundant fields or inconsistent naming conventions, making it difficult for analysts and engineers to understand and work with the data effectively.
x??

---

#### Data Model Definition
Background context explaining the concept. A data model represents how data relates to the real world, reflecting the structured and standardized way data should be organized to best support organizational processes, definitions, workflows, and logic. It captures natural communication flows within an organization.
:p What is a data model?
??x
A data model is a representation of how data relates to the real world. It reflects how data must be structured and standardized to best represent an organization's processes, definitions, workflows, and business logic. A good data model should correlate with impactful business decisions by capturing how communication and work naturally flow within the organization.

For example:
- In an e-commerce system, a customer data model might include fields like `customer_id`, `name`, `email`, and `order_history`. This structure helps in efficiently managing and querying customer-related data to support sales and marketing strategies.
x??

---

#### Data Model in Streaming and ML
Background context explaining the concept. As the importance of data grows, so does the need for coherent business logic through data modeling. The rise of data management practices such as data governance and quality has highlighted the critical role that well-constructed data models play. New paradigms are needed to effectively handle streaming data and machine learning.
:p How do modern practices like data governance affect data modeling?
??x
Modern practices like data governance emphasize coherent business logic, which drives the need for well-structured data models. Data governance ensures that data is managed consistently across an organization, adhering to rules and standards.

Data models are crucial in this context because they help align data with business objectives. Poorly modeled data can lead to confusion, redundancy, and incorrect decisions. As data becomes more prominent in companies, there's a growing recognition that robust data modeling is essential for realizing value at higher levels of the Data Science Hierarchy of Needs.

For example:
- In a financial institution, a well-defined data model ensures that customer transactional data is consistent and accurate, supporting compliance and risk management strategies.
x??

#### Customer Definition and Modeling
Background context explaining how different departments may have varying definitions of a customer. For example, someone who has bought from you over the last 30 days might be considered an active customer, while those who haven't purchased in six months or a year could be considered dormant customers.
:p How can defining customers differently across departments impact downstream reports and churn models?
??x
Defining customers differently across departments can significantly impact how customer behavior is reported and analyzed. For instance, different definitions might affect the accuracy of churn models, which rely heavily on the time since the last purchase as a critical variable. A consistent definition ensures that all stakeholders are aligned on what constitutes an active or dormant customer.
```java
public class CustomerDefinition {
    public boolean isActiveCustomer(long lastPurchaseTime) {
        // Check if the customer has made a purchase within the past 30 days
        return (System.currentTimeMillis() - lastPurchaseTime) < 30 * 24 * 60 * 60 * 1000;
    }
}
```
x??

---

#### Batch Data Modeling Continuum
Background context explaining the three main types of data models: conceptual, logical, and physical. These models form a continuum from abstract business logic to concrete database implementation.
:p What are the three main stages in batch data modeling?
??x
The three main stages in batch data modeling are:

1. **Conceptual Model**: This model contains business logic and rules, describing the system's data, such as schemas, tables, and fields (names and types). It is often visualized using an entity-relationship diagram.
2. **Logical Model**: This model details how the conceptual model will be implemented in practice by adding significant detail. For example, it includes information on specific data types and mappings of primary and foreign keys.
3. **Physical Model**: This model defines how the logical model will be implemented in a database system. It includes specific databases, schemas, and tables with configuration details.

The continuum helps in moving from abstract to concrete implementation.
x??

---

#### Conceptual Data Models
Background context on conceptual data models, which focus on business logic and rules, describing the system's data such as schemas, tables, and fields (names and types). ER diagrams are standard tools for visualizing these relationships.
:p What tool is commonly used to visualize a conceptual model?
??x
A common tool used to visualize a conceptual model is an entity-relationship (ER) diagram. This diagram helps in encoding the connections among various entities such as customer ID, customer name, customer address, and customer orders.

For example:
```plaintext
+----------------+     +---------------+
| Customer       |     | Orders        |
+----------------+     +---------------+
| customer_id    |<--- | order_id      |
| customer_name  |     | product_id    |
| customer_addr  |     | order_date    |
+----------------+     | total_price   |
                       +---------------+
```
x??

---

#### Logical Data Models
Background context on logical data models, which add significant detail to the conceptual model. This includes specific data types and mappings of primary and foreign keys.
:p What additional information is added in a logical model compared to a conceptual model?
??x
In a logical model, additional information such as specific data types (e.g., `int`, `varchar`) for fields like customer ID, customer names, and addresses is added. Primary and foreign key mappings are also defined. For example:

```java
public class Customer {
    private int customerId;
    private String customerName;
    private String customerAddress;

    // Getters and setters

    public void setCustomerId(int customerId) {
        this.customerId = customerId;
    }

    public void setCustomerName(String customerName) {
        this.customerName = customerName;
    }

    public void setCustomerAddress(String customerAddress) {
        this.customerAddress = customerAddress;
    }
}
```
x??

---

#### Physical Data Models
Background context on physical data models, which define the concrete implementation in a database system. This includes specific databases, schemas, and tables with configuration details.
:p What does the physical model include?
??x
The physical model includes specific databases, schemas, and tables along with their configuration details. For example:

```java
public class DatabaseConfiguration {
    private String databaseName;
    private List<Table> tables;

    public DatabaseConfiguration(String dbName) {
        this.databaseName = dbName;
    }

    public void addTable(Table table) {
        this.tables.add(table);
    }
}

public class Table {
    private String tableName;
    private List<Field> fields;

    public Table(String name) {
        this.tableName = name;
    }

    public void addField(Field field) {
        this.fields.add(field);
    }
}

public class Field {
    private String fieldName;
    private FieldType type;

    public Field(String name, FieldType type) {
        this.fieldName = name;
        this.type = type;
    }
}
```
x??

---

#### Granularity in Data Modeling
Background context on the importance of granularity (resolution at which data is stored and queried). This can affect query performance and storage efficiency.
:p What does the term "granularity" mean in data modeling?
??x
In data modeling, the term "granularity" refers to the level of detail or resolution at which data is stored and queried. It affects how data is partitioned, indexed, and accessed, impacting both query performance and storage efficiency.

For example, storing customer orders at a very granular level (e.g., order lines with individual items) might allow for detailed analysis but can also increase the complexity of queries and storage requirements.
```java
public class OrderLine {
    private int orderId;
    private int productId;
    private double quantity;

    public void setOrderId(int orderId) {
        this.orderId = orderId;
    }

    public void setProductId(int productId) {
        this.productId = productId;
    }

    public void setQuantity(double quantity) {
        this.quantity = quantity;
    }
}
```
x??

---

#### Grain Level in Data Modeling
Background context explaining that grain level refers to the detail of data stored at a certain level. Typically, primary keys like customer ID or order ID are used with timestamps for more precision. This example discusses creating detailed and highly granular tables vs. aggregated summary tables.

:p What is the meaning of "grain" in data modeling?
??x
Grain refers to the level of detail at which data is stored, such as using primary keys like customer ID or order ID typically accompanied by a date or timestamp for increased fidelity.
x??

---
#### Coarse-Grained vs. Fine-Grained Data Aggregation
Background context explaining that coarse-grained data aggregates high-level summary statistics while fine-grained data provides detailed insights into the transactions and records.

:p What is the difference between coarse-grained and fine-grained data aggregation?
??x
Coarse-grained data aggregation involves summarizing data to provide higher-level summaries, whereas fine-grained data aggregation retains detailed information at a lower level of granularity. Fine-grained data allows for easier re-aggregation when more detailed reports are needed.
x??

---
#### Normalization in Database Design
Background context explaining that normalization is a database design principle aimed at reducing redundancy and ensuring referential integrity through strict control over table relationships.

:p What is normalization, and why is it important?
??x
Normalization is a practice in database design to enforce strict control over the relationships between tables and columns. It aims to remove data redundancy and ensure referential integrity. Normalization helps maintain consistency and reduces the complexity of application programs.
x??

---
#### Normal Forms
Background context explaining that normal forms are specific levels or stages of normalization, each with its own set of rules.

:p What are normal forms in database design?
??x
Normal forms are sequential stages of normalization in a database. Each form includes the conditions of all previous forms and introduces additional constraints to reduce data redundancy further.
x??

---
#### Codd's Normal Forms
Background context explaining that Codd introduced four main objectives and three normal forms for normalization.

:p What were Codd’s four main objectives for normalization?
??x
Codd outlined four main objectives:
1. To free the collection of relations from undesirable insertion, update, and deletion dependencies.
2. To reduce the need for restructuring the collection of relations as new types of data are introduced, increasing the lifespan of application programs.
3. To make the relational model more informative to users.
4. To make the collection of relations neutral to the query statistics, where these statistics are liable to change over time.
x??

---
#### Denormalized Data
Background context explaining that denormalization involves intentionally reintroducing redundancy in a database to improve performance.

:p What is denormalized data?
??x
Denormalized data refers to a database design where intentional redundancy is introduced to optimize performance. This is the opposite of normalization and can be useful when query performance is more critical than strict data integrity.
x??

---

---
#### Denormalized OrderDetail Table
In our initial table, `OrderDetail`, we see that it is a denormalized structure. The primary key is `OrderID` and there are nested objects within the `OrderItems` field which contain multiple product details (SKU, Price, Quantity, Name).
:p What does the term "denormalized" mean in this context?
??x
In this context, "denormalized" means that data is stored in a way that doesn't follow normalization rules. Specifically, the table contains nested objects within `OrderItems` field, which makes it difficult to manage and manipulate.
```java
// Pseudocode for adding an order item to the OrderDetail table
public void addOrderItem(Order order) {
    // Nested object is stored as a stringified JSON or similar structure
    String orderItemsJson = "[{ \"sku\": 1, \"price\": 50, \"quantity\" : 1, \"name\":\"Thingamajig\" }, { \"sku\": 2, \"price\": 25, \"quantity\" : 2, \"name\": \"Whatchamacallit\" }]";
    order.setOrderItems(orderItemsJson);
}
```
x??
---

---
#### Moving to First Normal Form (1NF)
To convert the `OrderDetail` table into first normal form (1NF), we need to ensure that each column contains a single value and remove any nested data. We break down the `OrderItems` field into four separate columns: Sku, Price, Quantity, and ProductName.
:p What is First Normal Form (1NF) in database normalization?
??x
First Normal Form (1NF) requires that all the values of each column in a table must be atomic (indivisible), meaning simple and non-repetitive. In other words, no sub-values or arrays can exist within any single cell. Each value should be unique and distinct.
```java
// Pseudocode for transforming OrderDetail to 1NF
public void transformTo1NF(OrderDetail orderDetail) {
    List<OrderItem> items = Json.parse(orderDetail.getOrderItems()); // Parse the nested object
    for (OrderItem item : items) {
        String[] values = { item.getSku(), item.getPrice(), item.getQuantity(), item.getName() };
        // Insert or update each value in separate columns
    }
}
```
x??
---

---
#### Creating a Unique Primary Key
In the transformed `OrderDetail` table, we see that the primary key is not unique because multiple rows share the same `OrderID`. To create a unique primary key, we add a `LineItemNumber` column which numbers each line item in an order.
:p What is meant by "unique primary key"?
??x
A unique primary key means that every row in the table must have a distinct identifier. This unique value helps to uniquely identify each record in the database table and no two rows can share the same primary key value. It ensures that each item is identifiable without any ambiguity.
```java
// Pseudocode for adding LineItemNumber
public void addLineItemNumbers(List<OrderDetail> orderDetails) {
    int lineNumber = 1;
    for (OrderDetail detail : orderDetails) {
        String[] items = detail.getOrderItems();
        for (String item : items) {
            // Parse the item and set the LineItemNumber
            detail.setLineItemNumber(lineNumber++);
        }
    }
}
```
x??
---

---
#### Ensuring Second Normal Form (2NF)
To move to second normal form (2NF), we need to ensure that no partial dependencies exist. In our case, the `OrderID` column partially determines the `CustomerName`, so we split the table into two: `Orders` and `OrderLineItem`.
:p What is a partial dependency?
??x
A partial dependency occurs when non-key columns in a composite key are determined by only part of the primary key. For example, if `CustomerName` can be determined just from `OrderID` (a subset of the primary key), then there is a partial dependency.
```java
// Pseudocode for creating Orders and OrderLineItem tables
public void createNormalizedTables(List<OrderDetail> orderDetails) {
    Map<String, Order> ordersMap = new HashMap<>();
    List<OrderLineItem> itemsList = new ArrayList<>();

    for (OrderDetail detail : orderDetails) {
        String orderId = detail.getOrderID();
        Order order = ordersMap.get(orderId);
        if (order == null) {
            // Create a new Order
            order = new Order(detail.getCustomerID(), detail.getCustomerName(), detail.getOrderDate());
            ordersMap.put(orderId, order);
        }

        List<String> items = detail.getOrderItems();
        for (String item : items) {
            // Parse the item and add it to OrderLineItem
            OrderLineItem itemObj = parseOrderItem(item);
            itemsList.add(itemObj);
        }
    }

    return new Tuple<>(ordersMap.values(), itemsList);
}
```
x??
---

---
#### Transitive Dependencies in 3NF
In our `OrderLineItem` table, the `Sku` determines the `ProductName`, creating a transitive dependency. To remove this, we split the `OrderLineItem` into two tables: `OrderLineItem` and `Skus`.
:p What is a transitive dependency?
??x
A transitive dependency occurs when a non-key field depends on another non-key field, both of which depend on some part or all of the primary key. In our example, `Sku` (a non-key) determines `ProductName`, and this determination indirectly involves other columns.
```java
// Pseudocode for breaking OrderLineItem into two tables
public void breakOrderLineItem(List<OrderLineItem> items) {
    Map<String, Sku> skusMap = new HashMap<>();
    List<OrderLineItem> orderItemsList = new ArrayList<>();

    for (OrderLineItem item : items) {
        String skuId = item.getSku();
        Sku sku = skusMap.get(skuId);
        if (sku == null) {
            // Create a new Sku
            sku = new Sku(skuId, parseProductName(item));
            skusMap.put(skuId, sku);
        }
        // Assign the Sku to OrderLineItem
        item.setSku(sku);
        orderItemsList.add(item);
    }

    return new Tuple<>(orderItemsList, skusMap.values());
}
```
x??
---

