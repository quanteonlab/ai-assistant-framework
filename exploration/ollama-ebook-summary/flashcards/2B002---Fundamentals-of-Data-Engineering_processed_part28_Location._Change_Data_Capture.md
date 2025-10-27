# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 28)

**Starting Chapter:** Location. Change Data Capture

---

---
#### Dead-Letter Queue Concept
A dead-letter queue is a mechanism used to segregate problematic events that cannot be processed by consumers. These events are stored separately from those successfully ingested, preventing them from blocking other messages and aiding in diagnosing ingestion errors.

:p What is the purpose of using a dead-letter queue?
??x
The primary purpose of using a dead-letter queue (DLQ) is to handle events that fail processing or cannot be consumed by the intended consumer. By isolating these "bad" events, DLQs prevent them from blocking other messages and provide engineers with a means to diagnose and fix issues that cause ingestion errors.

For example:
- If an event contains invalid data,
- A message fails due to a transient error,
- Or if a consumer is not able to process it for any reason (e.g., network issue, application crash),

These events are moved to the DLQ where they can be examined and addressed. This allows for the reprocessing of some messages after fixing underlying issues.

```java
// Pseudocode for moving problematic messages to DLQ
public void handleMessage(Message message) {
    try {
        // Process message
        process(message);
    } catch (Exception e) {
        // Move to dead-letter queue if processing fails
        moveFailedMessageToDLQ(message, e);
    }
}
```
x??

---
#### Pull vs. Push Subscriptions
Pull subscriptions involve consumers actively fetching messages from a topic or stream. In contrast, push subscriptions allow services to write messages directly to a listener on the consumer.

:p How do Kafka and Kinesis differ in terms of subscription methods?
??x
Kafka and Kinesis support only pull subscriptions. Consumers must explicitly request messages by pulling them from topics or streams. This means that consumers are responsible for fetching data, confirming receipt, and processing it.

Example code (Java):
```java
// Pull Subscription Example using Kafka
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "my-group");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Collections.singletonList("my-topic"));

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records)
        process(record.value());
}
```
x??

---
#### Location Considerations in Streaming Ingestion
Integrating streaming across multiple locations can enhance redundancy and improve latency by consuming data closer to where it is generated. However, this needs careful balancing against the cost of moving data between regions for analytics.

:p Why is location an important consideration in streaming ingestion?
??x
Location matters because streaming data ingested closer to its source typically offers better performance due to reduced network latency. However, there's a trade-off with the cost associated with replicating and transmitting data across different regions or availability zones.

For instance:
- In regions far from where the data is generated, there may be increased latency,
- Costs for moving large volumes of data between regions can also increase significantly.

Thus, it’s essential to carefully evaluate these factors when designing a streaming architecture. This balance helps in optimizing both performance and cost efficiency.

```java
// Pseudocode for evaluating location trade-offs
public class LocationEvaluation {
    private Map<String, Cost> costMap = new HashMap<>();

    public void evaluateLocation(String region) {
        double cost = calculateEgressCost(region);
        if (cost > THRESHOLD_COST) {
            System.out.println("Not optimal to replicate in " + region);
        } else {
            System.out.println("Replication is feasible in " + region);
        }
    }

    private double calculateEgressCost(String region) {
        // Implement logic to estimate cost based on egress data rates and regions
        return 0.1 * dataVolumeInMB; // Hypothetical cost calculation
    }
}
```
x??

---
#### Direct Database Connection Ingestion
Data can be ingested directly from databases by querying them using protocols like ODBC or JDBC. These connections allow applications to pull data over a network, translating commands into database-specific queries and back.

:p How does direct database connection work for ingestion?
??x
Direct database connection involves establishing a network connection between the application and the database to query and read data. Commonly used APIs include ODBC (Open Database Connectivity) or JDBC (Java Database Connectivity).

ODBC works by using a driver hosted on the client side that translates standard API commands into specific database commands, facilitating cross-database interaction.

```java
// Example of ODBC connection in Java
public class DatabaseIngestion {
    public void fetchAndProcessData() throws SQLException {
        Connection conn = DriverManager.getConnection("jdbc:odbc:myDataSource", "username", "password");
        Statement stmt = conn.createStatement();
        
        ResultSet rs = stmt.executeQuery("SELECT * FROM myTable");

        while (rs.next()) {
            // Process each row of data
            processRow(rs.getString("column1"), rs.getString("column2"));
        }
    }

    private void processRow(String col1, String col2) {
        // Logic to process the fetched data
    }
}
```
x??

---

---
#### Java Virtual Machine (JVM)
The JVM is a standard, portable virtual machine that supports running compiled code. It offers performance through just-in-time (JIT) compilation and is used extensively for cross-platform applications due to its portability across hardware architectures and operating systems.
:p What is the role of the Java Virtual Machine (JVM)?
??x
The Java Virtual Machine acts as a runtime environment that executes bytecode, ensuring code written in Java or compatible languages can run on any platform where a JVM implementation is available. It provides memory management, security, and performance optimizations through JIT compilation.
```java
public class Example {
    public static void main(String[] args) {
        // The JVM compiles this to machine code at runtime using JIT.
    }
}
```
x??

---
#### Just-In-Time (JIT) Compiler
The just-in-time compiler is a component of the Java Virtual Machine that compiles bytecode into native machine code at runtime, enhancing performance by executing optimized versions of frequently used methods directly on the underlying hardware.
:p What does the JIT compiler do in the context of JVM?
??x
The JIT compiler takes compiled Java bytecode and converts it into machine-specific instructions before execution. This process is performed dynamically based on method usage frequency to optimize performance. Here's a simplified example:
```java
public class Example {
    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        
        // Method that might be JIT-compiled during its frequent calls.
        for (int i = 0; i < 1000000; i++) {
            doSomething(i);
        }
        
        long end = System.currentTimeMillis();
        System.out.println("Time taken: " + (end - start) + " ms");
    }

    private static void doSomething(int i) {
        // Some method implementation.
    }
}
```
x??

---
#### JDBC vs. ODBC
JDBC (Java Database Connectivity) and ODBC (Open Database Connectivity) are standards for accessing databases from Java applications, providing portability across different operating systems and architectures. While ODBC requires separate binaries for each architecture/OS version, JDBC is compatible with any JVM language and framework.
:p How does JDBC compare to ODBC in terms of compatibility?
??x
JDBC offers better cross-platform compatibility because it works seamlessly with any JVM language (Java, Scala, Clojure, Kotlin) and data frameworks like Spark. In contrast, ODBC necessitates separate binaries for different operating systems and architectures, requiring more maintenance from the vendor.
```java
// Example JDBC connection setup
public class DatabaseConnection {
    public static Connection connectToDatabase() throws SQLException {
        String url = "jdbc:mysql://localhost:3306/mydatabase";
        String username = "root";
        String password = "password123";
        
        return DriverManager.getConnection(url, username, password);
    }
}
```
x??

---
#### Parallel Data Ingestion
Parallel data ingestion involves using multiple simultaneous connections to a database to pull data in parallel, which can significantly speed up the process. However, this approach increases load on the source database.
:p How does parallel data ingestion work?
??x
Parallel data ingestion allows reading and processing data from multiple sources concurrently, reducing overall data ingestion time. This is achieved by creating and managing multiple connections that operate simultaneously. While it speeds up the process, it can also increase the load on the source database.
```java
public class ParallelIngestion {
    public static void main(String[] args) throws SQLException {
        int numConnections = 5; // Number of parallel connections to create
        List<Connection> connections = new ArrayList<>();
        
        for (int i = 0; i < numConnections; i++) {
            connections.add(DatabaseConnection.connectToDatabase());
        }
        
        // Process data from each connection concurrently.
    }
}
```
x??

---
#### Change Data Capture (CDC)
Change Data Capture is the process of ingesting changes made to a database, such as updates or inserts. It's commonly used in analytics and data warehousing applications where real-time or near-real-time data processing is required.
:p What is change data capture (CDC)?
??x
Change Data Capture (CDC) captures modifications in a source database system for further analysis or processing. For example, it can be used to periodically or continuously ingest table changes from a PostgreSQL database for analytics purposes. This process helps maintain up-to-date data without manual intervention.
```java
public class CDCExample {
    public static void main(String[] args) throws SQLException {
        Connection sourceDB = DatabaseConnection.connectToDatabase();
        
        // SQL statement to capture changes in the database
        String sql = "SELECT * FROM table WHERE version_column > last_version";
        
        try (Statement stmt = sourceDB.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            while (rs.next()) {
                // Process each row of changed data.
            }
        }
    }
}
```
x??

---

---
#### Batch-oriented CDC Overview
Batch-oriented Change Data Capture (CDC) involves querying a database to find all records that have been updated since a specified time. The process is driven by an `updated_at` field which tracks when a record was last written or modified.

The key limitation of batch-oriented CDC is that it only captures the latest state of the row, missing any intermediate changes.
:p What are the limitations of batch-oriented CDC?
??x
Batch-oriented CDC can only capture the final state of rows since the specified timestamp. It does not provide information about all changes that occurred to a record during the period, leading to potential data loss or inaccuracies in the updates.

For example:
- A customer makes five transactions on an account over 24 hours.
- The batch query will only return the last updated balance, missing details of intermediate transactions.
??x
---

---
#### Insert-Only Schema for Batch CDC Mitigation
An insert-only schema can mitigate some of the limitations associated with batch-oriented CDC. In this approach, each update to a record is recorded as a new row in the table instead of updating an existing one.

This method preserves all historical data and provides more detailed information on changes.
:p How does an insert-only schema help in batch CDC?
??x
An insert-only schema helps by recording each change as a separate entry. This way, every transaction to the database results in a new row being added to the table, rather than modifying existing rows.

This ensures that all historical data is preserved and can be easily queried for detailed insights into changes over time.
??x
---

---
#### Continuous CDC Overview
Continuous Change Data Capture (CDC) continuously captures all changes to a database in real-time. It treats each write operation as an event and can support near-real-time data ingestion.

Continuous CDC tools like Debezium can read database logs and stream events to external systems such as Apache Kafka.
:p What is continuous CDC, and how does it differ from batch-oriented CDC?
??x
Continuous CDC continuously captures all changes to a database in real-time. Unlike batch-oriented CDC which captures changes at predefined intervals (e.g., every 24 hours), continuous CDC processes each write event immediately.

For example:
- A bank transaction occurs.
- Continuous CDC tools capture this transaction and stream it to an external system instantly.
??x
---

---
#### Log-based CDC with PostgreSQL
Log-based Change Data Capture (CDC) involves using the binary log of a database like PostgreSQL. This log records every change sequentially, allowing CDC tools to read these events.

For example, using Debezium, one can capture and process these logs to stream changes into Apache Kafka.
:p How does log-based CDC work with PostgreSQL?
??x
Log-based CDC works by leveraging the binary logs of a database like PostgreSQL. The binary log records every change made to the database sequentially. A tool like Debezium reads these logs and sends events to an external system, such as Apache Kafka.

For example:
```java
// Pseudocode for reading PostgreSQL binary logs using Debezium
DebeziumConnector dbz = new DebeziumConnector();
dbz.connect();
while (true) {
    Event event = dbz.nextEvent();
    if (event != null) {
        // Process the event and send it to Kafka or another system
    }
}
```
x??
---

---
#### Synchronous Database Replication
Synchronous database replication ensures that a secondary database is fully in sync with the primary database. This type of replication is tightly coupled and typically requires both databases to be of the same type (e.g., PostgreSQL to PostgreSQL).

Synchronous replication can offload read queries from the primary database by directing them to the replica, reducing the load on the primary.
:p What is synchronous database replication?
??x
Synchronous database replication ensures that a secondary database remains fully in sync with the primary. This means any write operation that succeeds on the primary will also succeed on the replica.

For example:
- A write query to PostgreSQL (primary) succeeds.
- The same change is immediately applied to another PostgreSQL instance acting as a read replica.

This guarantees consistency between the databases and allows for load balancing by redirecting read queries to the replica, reducing the burden on the primary database.
??x
---

#### CDC Replication and Asynchronous Considerations
CDC (Change Data Capture) replication involves capturing changes made to a database and replicating them to another database. This method is crucial for maintaining data consistency across different systems, especially important for analytical applications that need real-time or near-real-time data.

Asynchronous CDC allows the replica to be delayed slightly from the primary database, making it suitable for environments where this delay does not significantly impact operations. It provides a loosely coupled architecture pattern, which can direct events to various targets like object storage and streaming analytics processors simultaneously.

However, asynchronous CDC is not without its drawbacks. It consumes resources such as memory, disk bandwidth, storage, CPU time, and network bandwidth on the primary database. Therefore, engineers must conduct thorough tests before deploying CDC in production environments to avoid operational issues.

:p What are the key considerations for implementing asynchronous CDC replication?
??x
Key considerations include resource consumption on the primary database (memory, disk bandwidth, storage, CPU time, and network bandwidth), testing in a non-production environment to ensure stability, and understanding that slight delays may be acceptable due to its loosely coupled nature. Additionally, engineers must balance the need for real-time data with potential operational impacts.
x??

---

#### API Data Ingestion Challenges
APIs play a significant role as data sources, particularly in organizations with numerous external data sources such as SaaS platforms or partner companies. However, there is no standardized method for exchanging data over APIs, leading to challenges like reading documentation and writing maintenance-heavy code.

To address these challenges, three trends are emerging: 
1. API client libraries that simplify access by various programming languages.
2. Data connector platforms providing turnkey solutions with frameworks for custom connectors.
3. Data sharing through platforms like BigQuery, Snowflake, Redshift, or S3, which facilitate storing and processing data without the need for direct API access.

:p How can engineers reduce time spent on API connection code?
??x
Engineers can leverage existing managed services and API client libraries that provide standardized solutions, reducing the need to write custom connectors from scratch. Managed services often support building custom API connectors with standard formats or in serverless function frameworks like AWS Lambda, which handle scheduling and synchronization. This approach saves time and resources.
x??

---

#### Message Queues and Event-Streaming Platforms
Message queues and event-streaming platforms are essential for ingesting real-time data from web applications, mobile apps, IoT sensors, and smart devices. They allow events to be consumed at the individual event level or aggregated into ordered logs.

Messages handle events individually and transiently, acknowledging and removing them once processed. Streams, on the other hand, ingest events into an ordered log that persists for extended periods, allowing complex queries and transformations over time.

:p What is the main difference between messages and streams?
??x
The main difference lies in their handling of data:
- **Messages**: Handle individual events transiently; once consumed, they are acknowledged and removed from the queue.
- **Streams**: Ingest events into an ordered log that can persist for long periods, enabling complex queries, aggregations, and transformations over time.

Example Code (Pseudocode):
```pseudocode
if message:
    process_message(message)
else if stream:
    process_stream(stream)
```
x??

---

#### Managed Data Connectors Overview
Managed data connectors are services that provide pre-built connectors for various data sources, eliminating the need for custom development. These platforms offer turnkey solutions with frameworks for writing and managing custom connectors.

They support various ingestion methods like CDC (Change Data Capture), replication, truncate and reload, and allow setting permissions, credentials, and update frequencies to begin syncing data automatically. Vendors or cloud providers manage and monitor data syncs, alerting on failures with detailed error logs.

:p When should engineers consider using managed data connectors?
??x
Engineers should consider using managed data connectors when there are well-supported data sources that don't require custom development, as it saves time and resources. Managed services can provide standard API technical specifications or run connector code in serverless function frameworks while handling scheduling and synchronization details.

This approach is particularly valuable for high-value work where engineers can focus on more critical tasks.
x??

---

#### Managed Connector Platforms
Background context explaining the use of managed connector platforms. These tools are designed to handle data integration tasks efficiently, offering prebuilt and custom connectors that reduce the burden of creating and managing these components manually.

The creation and management of data connectors can be cumbersome due to their undifferentiated nature. Managed services often provide a more streamlined approach, allowing organizations to focus on their core business logic rather than low-level infrastructure details.

:p What are managed connector platforms in the context of data integration?
??x
Managed connector platforms are specialized tools that handle the creation and management of data connectors for organizations. They offer prebuilt and customizable options, reducing the workload typically associated with manual connector development and maintenance.
x??

---

#### Object Storage for Data Movement
Background context explaining object storage as a multitenant system in public clouds. It supports massive amounts of data storage, making it ideal for various use cases such as moving data to and from data lakes, between teams, or between organizations.

Object storage can provide short-term access via signed URLs, granting temporary permissions to users. This feature ensures secure file exchanges while maintaining compliance with security standards.

:p What is object storage used for in the context of data movement?
??x
Object storage is primarily used for moving and storing large volumes of data in public clouds. It supports use cases such as transferring data between teams or organizations, integrating with data lakes, and facilitating file exchanges through signed URLs.
x??

---

#### Electronic Data Interchange (EDI)
Background context explaining EDI as a method of data movement that can be archaic but still relevant for some systems due to limitations in modern infrastructure.

Engineers can enhance traditional EDI methods by automating processes. For example, setting up a cloud-based email server that automatically saves received files into company object storage upon receipt and triggers subsequent ingestion and processing workflows.

:p What is the role of automation in enhancing EDI?
??x
Automation plays a crucial role in enhancing traditional EDI methods by reducing manual intervention. By setting up a system where files are saved to object storage as soon as they arrive, it ensures that data can be processed efficiently without human involvement.
x??

---

#### Database and File Export Considerations
Background context explaining the challenges associated with exporting large datasets from transactional systems. Source system engineers must carefully plan when to run export queries to avoid affecting application performance.

Techniques such as breaking down exports into smaller parts or using read replicas can mitigate the load on the source database during frequent exports.

:p How should one approach file export in a transactional system?
??x
When exporting large datasets from transactional systems, engineers must plan carefully to minimize impact on application performance. Strategies include breaking down exports into smaller segments by key ranges or single partitions and using read replicas to reduce load, especially for frequent exports.
x??

---

