# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 34)

**Starting Chapter:** Events. Data Quality Monitoring

---

#### Events vs Logs
Background context explaining how events and logs differ. Events are structured data triggered by specific activities, often stored in SQL databases due to their structured nature. Logs, on the other hand, are semi-structured data that provide more flexibility regarding content, structure, and triggering mechanisms.

:p How do events and logs differ?
??x
Events are typically structured data emitted during runtime and are usually related to specific actions or activities, such as HTTP requests or API calls. They are well-suited for storage in SQL databases due to their structured nature. Logs, however, are semi-structured data that can be more flexible in terms of content and structure. They often include severity levels (Debug, Info, Warning, Error, Critical) and are stored using tools like ELK stack or cloud-based services.

```java
// Example of logging an error in Java
public class App {
    private static final Logger logger = LoggerFactory.getLogger(App.class);

    public void processPayment(String accountNumber, double amount) {
        try {
            // Process payment logic here
        } catch (Exception e) {
            logger.error("Error processing payment for account: {} with amount: {}", accountNumber, amount);
        }
    }
}
```
x??

---

#### Traces in Monitoring Systems
Background context on how traces provide more detailed insights into system behavior. They capture comprehensive details of processes as they move through multiple systems and are essential for troubleshooting complex architectures.

:p What is a trace in the monitoring layer?
??x
A trace represents a record of a process's steps as it progresses through one or more systems. It captures detailed information about business operations, application behaviors, and security incidents. Examples include tracing trade orders, payment flows, loan applications, etc.

```java
// Pseudocode for capturing an application trace
public class ApplicationTracer {
    private TraceId currentTraceId;

    public void startTrace(String traceId) {
        this.currentTraceId = traceId;
        System.out.println("Starting trace with ID: " + traceId);
    }

    public void logStep(String stepDescription) {
        if (currentTraceId != null) {
            System.out.println("Step " + currentTraceId + ": " + stepDescription);
        }
    }
}
```
x??

---

#### Unique Transaction Identifiers
Background context on the importance of unique identifiers for transaction tracing. These identifiers ensure all related activities can be linked together, providing a complete view of the transaction's lifecycle.

:p What is a trace identifier in the context of transaction tracing?
??x
A trace identifier, also known as a transaction ID or correlation ID, uniquely identifies and monitors the path of a particular transaction or request across complex systems. This ensures that all related activities can be linked together, providing transparency throughout the process. Examples include UTIs defined in ISO 23897 for financial transactions.

```java
// Pseudocode for generating a unique transaction identifier
public class TransactionIdGenerator {
    public static String generateUTI() {
        return UUID.randomUUID().toString();
    }
}
```
x??

---

#### Metrics, Events, Logs, and Traces
Background context on the different types of monitoring data: metrics, events, logs, and traces. Each serves a unique purpose in system monitoring.

:p What are the four main types of monitoring data discussed?
??x
The four main types of monitoring data are:
1. **Metrics**: Quantitative measures that summarize specific aspects or dimensions of data.
2. **Events**: Structured data triggered by specific activities, often stored in SQL databases due to their structured nature.
3. **Logs**: Semi-structured data with more flexibility regarding content and structure, typically classified into severity levels like Debug, Info, Warning, Error, Critical.
4. **Traces**: Detailed records of processes as they move through one or more systems.

```java
// Example of logging a metric in Java
public class MetricsLogger {
    private static final Logger logger = LoggerFactory.getLogger(MetricsLogger.class);

    public void logTransactionSuccess(String transactionId) {
        logger.info("Transaction {} processed successfully", transactionId);
    }
}
```
x??

---

#### Data Quality Monitoring
Background context on data quality monitoring, including the importance of defining and implementing DQMs. It discusses common techniques like using ratios for error detection and ensuring data timeliness.

:p What is the purpose of data quality monitoring?
??x
The purpose of data quality monitoring is to ensure that financial institutions' generated and used data consistently meets high-quality standards. This involves defining Data Quality Metrics (DQMs), which are quantitative measures used to assess and summarize the quality of specific aspects or dimensions of data.

Common techniques include:
- Using ratios such as error ratio, duplicate ratio, missing values ratio.
- Ensuring data timeliness by measuring the age of a data item against a reference date.

```java
// Example of calculating an error ratio in Java
public class DataQualityMonitor {
    private int totalRecords;
    private int erroneousRecords;

    public double calculateErrorRatio() {
        return (double) erroneousRecords / totalRecords;
    }
}
```
x??

#### Data Profiling Overview
Data profiling is a comprehensive process that scans an entire dataset to understand its data quality attributes. It helps identify issues such as missing values, inconsistencies, and duplicates, providing insights into the overall data health.

:p What is data profiling?
??x
Data profiling involves scanning a dataset to assess its quality attributes comprehensively, identifying issues like missing or inconsistent data.
x??

---

#### Data Quality Rules
Defining and testing data quality rules against one or more records can provide quicker insights than full data profiling. For example, setting an error ratio not exceeding 1 percent.

:p What is the purpose of defining data quality rules?
??x
The purpose is to quickly identify data issues by applying specific criteria to a subset of the dataset rather than conducting an exhaustive data profile.
x??

---

#### Data Quality Tools
Several commercial and open-source tools are available for data quality management, such as Ataccama ONE Data Quality Suite, Informatica Data Quality, Precisely Trillium, SAS Data Quality, Talend Data Fabric, Great Expectations, Soda Core, and DataCleaner.

:p What types of data quality tools exist?
??x
Commercial tools include Ataccama ONE Data Quality Suite, Informatica Data Quality, Precisely Trillium, SAS Data Quality, and Talend Data Fabric. Open-source options are Great Expectations, Soda Core, and DataCleaner.
x??

---

#### Continuous Data Quality Monitoring
Data quality monitoring is an ongoing process that needs to be regularly revisited and refined based on changes in the financial data landscape.

:p Why is continuous data quality monitoring important?
??x
Continuous data quality monitoring ensures the relevance and effectiveness of your data quality management practices, adapting to new data types, dimensions, and business requirements.
x??

---

#### Performance Monitoring Overview
Performance refers to how well a financial data infrastructure meets key operational criteria such as speed, latency, throughput, availability, scalability, and resource utilization.

:p What does performance monitoring focus on?
??x
Performance monitoring focuses on ensuring the financial data infrastructure can efficiently handle operations by meeting criteria like speed, latency, and resource utilization.
x??

---

#### Time to Insight (TTI)
TTI is the time taken from when data is generated until actionable insights are derived. A shorter TTI enables efficient decision-making.

:p What is Time to Insight (TTI)?
??x
Time to Insight (TTI) measures the time required to derive actionable insights from data, starting from its generation.
x??

---

#### Time to Market (TTM)
TTM refers to the duration it takes for a product to progress from ideation to market introduction.

:p What is Time to Market (TTM)?
??x
Time to Market (TTM) measures how quickly a product can be developed and brought to market after an idea has been conceptualized.
x??

---

#### Performance Metrics Overview
Various metrics are used to monitor data infrastructure performance, including software-agnostic metrics like RAM usage, CPU usage, storage usage, and execution time.

:p What types of performance metrics exist?
??x
Performance metrics include software-agnostic ones such as RAM usage, CPU usage, storage usage, and execution time. Specific software systems also have tailored metrics.
x??

---

#### Database Performance Metrics Example
For database management systems, key metrics like read/write operations, active connections, connection pool usage, query count, transaction runtime, and replication lag can be monitored.

:p What are some specific database performance metrics?
??x
Specific database performance metrics include read/write operations, active connections, connection pool usage, query count, transaction runtime, and replication lag.
x??

---

#### Granular Database Query Metrics
Granular metrics for database queries could include records read, blocks read, bytes scanned, and number of output rows.

:p What are some granular metrics for database queries?
??x
Granular metrics for database queries may include records read, blocks read, bytes scanned, and the final number of rows returned by a query.
x??

---

---
#### Scan Time
Background context: The time spent by the database scanning the data is a critical performance metric. It helps understand how long it takes for the database to access and process required data. This can be important when dealing with large datasets.

:p What does scan time measure in the context of database operations?
??x
Scan time measures the duration the database spends on scanning or reading through the data during query execution.
x??

---
#### Sort Time
Background context: The time spent by the database sorting the data is another performance metric. Sorting can be a resource-intensive process, especially when dealing with large datasets.

:p What does sort time measure in the context of database operations?
??x
Sort time measures the duration the database spends on sorting data during query execution.
x??

---
#### Aggregation Time
Background context: Aggregation time is the time spent by the database aggregating (summarizing) the data. This can include operations like sum, count, average, etc.

:p What does aggregation time measure in the context of database operations?
??x
Aggregation time measures the duration the database spends on summarizing or aggregating data during query execution.
x??

---
#### Peak Memory Usage
Background context: The maximum amount of memory used during query execution is a key performance metric. It helps identify if there are any memory bottlenecks that could affect query performance.

:p What does peak memory usage measure in the context of database operations?
??x
Peak memory usage measures the highest amount of memory (RAM) consumed by the database during the execution of a query.
x??

---
#### Query Plans
Background context: Monitoring query plans provides insights into how data is fetched and processed. It helps detect costly queries and unused DB objects, aiding in optimizing performance.

:p What are query plans used for?
??x
Query plans are executed steps that a data storage system follows to fetch data in response to a query. They help identify inefficient or unnecessary operations, enabling optimization.
x??

---
#### Trade Execution Latency
Background context: Trade execution latency is the time taken from when a trade order is placed until it is executed. It's critical for optimizing trading strategies and minimizing execution risks.

:p What does trade execution latency measure in financial markets?
??x
Trade execution latency measures the time from when a trade order is placed to when it is executed.
x??

---
#### Trade Execution Error Rate
Background context: The frequency of failed transactions or trade orders indicates operational inefficiencies. This metric helps in identifying and addressing issues in the trading process.

:p What does trade execution error rate measure?
??x
Trade execution error rate measures how frequently trade orders fail, indicating potential operational inefficiencies.
x??

---
#### Trade Settlement Time
Background context: The duration from when a trade is executed to when it is settled reflects the efficiency of the settlement process. This metric is crucial for ensuring timely and accurate settlements.

:p What does trade settlement time measure?
??x
Trade settlement time measures the duration from when a trade is executed until it is settled.
x??

---
#### Market Data Latency
Background context: The time it takes for financial market data to be received from the exchange or data provider to the trading system impacts the speed of decision-making. This metric ensures that traders have access to up-to-date information.

:p What does market data latency measure?
??x
Market data latency measures the time it takes for financial market data to be delivered from the exchange or data provider to the trading system.
x??

---
#### Algorithmic Trading Performance Metrics
Background context: Metrics such as algorithm execution time and success rate highlight the effectiveness and reliability of algorithmic trading strategies. These are essential for optimizing trading systems.

:p What does algorithmic trading performance include?
??x
Algorithmic trading performance includes metrics like algorithm execution time and success rate, which show how effectively algorithms execute trades.
x??

---
#### Risk Exposure Calculation Time
Background context: The time taken to compute and update risk exposure metrics is critical for managing financial risks in real time. Efficient calculation ensures timely adjustments.

:p What does risk exposure calculation time measure?
??x
Risk exposure calculation time measures the duration it takes to compute and update risk exposure metrics.
x??

---
#### Customer Transaction Processing Time
Background context: The duration from when a customer initiates a transaction (e.g., deposit, withdrawal) until its completion impacts customer satisfaction and operational efficiency.

:p What does customer transaction processing time measure?
??x
Customer transaction processing time measures the duration from when a customer initiates a transaction until it is completed.
x??

---
#### Network Metrics in Real-Time Financial Data Feeds
Background context: Key network metrics like packets per second (PPS), packet size distribution, round-trip time (RTT), and bandwidth utilization are essential for ensuring efficient data delivery.

:p What does PPS measure?
??x
Packets per second (PPS) measures the number of data packets transmitted and received per second.
x??

---
#### System Metrics in Real-Time Financial Data Feeds
Background context: CPU usage, memory utilization, I/O wait time, and interrupt time are critical system metrics that help monitor the health and performance of the infrastructure.

:p What does CPU usage measure?
??x
CPU usage measures the percentage of CPU capacity used by the system.
x??

---
#### Backpressure in Data Processing Systems
Background context: Backpressure occurs when a system receives more message requests than it can handle. Buffer or queue sizes monitor data waiting to be processed, indicating potential bottlenecks.

:p What does backpressure track?
??x
Backpressure tracks the system's ability to handle incoming data rates.
x??

---
#### Incident Response and Management Metrics
Background context: Metrics like Mean Time to Detect (MTTD) and Mean Time Before Failure (MTBF) help evaluate an organization’s effectiveness in preventing and managing system downtimes.

:p What does MTTD measure?
??x
Mean Time to Detect (MTTD) measures the expected time it takes to detect an issue or incident after it has occurred.
x??

---

#### MTTR, MTTF, and MTTA
Background context explaining Mean Time to Recovery (MTTR), Mean Time to Failure (MTTF), and Mean Time to Acknowledge (MTTA). These metrics are crucial for measuring system reliability and team responsiveness. Low values of these metrics indicate efficient issue resolution and alerting systems.
:p What is the meaning of MTTR?
??x
MTTR, or Mean Time to Recovery, is the expected time before a system recovers from a failure and becomes fully operational. A low MTTR indicates that the engineering team is effective at resolving issues quickly and efficiently.
x??

---
#### Metrics in Financial Market Infrastructures
Background context explaining the importance of metrics like MTTR, MTTA, etc., in financial market infrastructures due to their always-on nature and high-frequency/high-volume activities. Outages can lead to significant repercussions.
:p What are some key performance indicators (KPIs) for financial market infrastructures?
??x
Key KPIs include Mean Time to Recovery (MTTR), which measures how quickly the system recovers from a failure; Mean Time to Acknowledge (MTTA), which measures how fast issues are addressed once reported; and incident metrics like these are crucial due to the always-on nature of financial markets.
x??

---
#### Cost Monitoring
Background context explaining the importance of cost monitoring in cloud services, especially with serverless computing. Cloud services offer flexible on-demand pricing but can lead to unforeseen costs if not managed properly.
:p What is the significance of cost monitoring for cloud services?
??x
Cost monitoring is crucial because it helps ensure that expenditures remain within acceptable bounds and spot patterns of excessive usage. This is particularly important in cloud services due to their pay-as-you-go model, which can result in unexpected costs if not carefully managed.
x??

---
#### Serverless Computing Economics
Background context explaining the economics of serverless computing, including factors like execution time and memory allocation that significantly impact costs. Real-world examples are provided to illustrate potential cost variations.
:p How does increasing execution time or memory affect AWS Lambda costs?
??x
Increasing execution time or memory allocation can significantly increase AWS Lambda costs. For instance, a function executing for 60 seconds with 1,024 MB of RAM would incur higher costs compared to the same function running for 10 seconds with 128 MB of RAM.
To illustrate:
```plaintext
Scenario 1: 100 RPS with 10-second invocations (128 MB)
Cost per invocation = $0.0000000021 * 10,000 ms =$0.000021
Monthly cost = $0.000021 * 100 RPS * 100,000 seconds =$210

Scenario 2: 1-minute invocations (60 seconds)
Cost per invocation = $0.0000000021 * 60,000 ms =$0.000126
Monthly cost = $0.000126 * 100 RPS * 100,000 seconds =$1,260

Scenario 3: 1 GB memory (1,024 MB)
Cost per invocation = $0.0000000167 * 60,000 ms =$0.001002
Monthly cost = $0.001002 * 100 RPS * 100,000 seconds =$10,020
```
x??

---
#### FinOps and Cloud Cost Management
Background context explaining the concept of FinOps as a framework for managing cloud costs across engineering, finance, and business teams. It includes five pillars to optimize cloud usage.
:p What is FinOps?
??x
FinOps (Financial Operations) is an operational framework and cultural practice that maximizes the business value of cloud services by enabling timely data-driven decisions and creating financial accountability through collaboration between engineering, finance, and business teams.
The FinOps framework consists of five pillars:
1. Cross-functional collaboration among engineering, finance, and business teams to enable fast product delivery while ensuring financial and cost control.
2. Each team takes ownership of its cloud usage and costs.
3. A central team establishes and promotes FinOps best practices.
4. Teams take advantage of the cloud's on-demand and variable cost model while optimizing tradeoffs among speed, quality, and cost in their applications.
5. Decisions are driven by the business value, such as increased revenue, innovative product development, reduced fixed costs, and faster feature releases.
x??

---

