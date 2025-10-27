# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 31)

**Starting Chapter:** DataOps

---

#### Data Tokenization and Hashing
Background context: When handling sensitive data like customer emails, it is crucial to ensure that even if an email address is hashed (transformed into a fixed-size string of characters), it cannot be easily reversed or linked back to its original form. Common hashing functions can make this process difficult but not impossible without proper salting and other security measures.
:p How might a person with access to one of your customers' emails use a simple hash function, like MD5, to find the customer in your data?
??x
Hashing alone is insufficient for protecting privacy since it can be reverse-engineered or brute-forced. Using simple hashing functions without additional strategies such as salting makes re-identification of hashed data much easier.
For example:
Given a simple hash function `MD5`, if an attacker has the customer's email and knows the MD5 hash, they could potentially use rainbow tables (a precomputed table for reversing cryptographic hash functions) to find the original value. This highlights the importance of using strong hashing algorithms with proper salting techniques.
??x
To enhance security, always use a strong hashing algorithm combined with random salts unique to each piece of data. For example:
```java
public String hashWithSalt(String password, String salt) {
    return messageDigest.digest((salt + password).getBytes()).toString();
}
```
x??

---

#### Data Pipeline Monitoring and Reliability
Background context: Ensuring the reliability of data pipelines is critical for maintaining the integrity of downstream systems. Proper monitoring helps in detecting failures early and ensuring that data processing flows smoothly.
:p Why is monitoring crucial during the ingestion stage of a data pipeline?
??x
Monitoring is essential because it ensures the data pipeline functions as expected without downtime. Without proper monitoring, any issues may go unnoticed until they significantly impact operations or lead to stale data.
For example:
```java
public void monitorIngestionJob() {
    // Check job status and log errors if necessary
    if (jobStatus.isFailed()) {
        logger.error("Ingestion job failed: " + jobStatus.getErrorDetails());
    }
}
```
x??

---

#### Data Quality Tests
Background context: Data quality is paramount for effective decision-making. Poor data can lead to incorrect business decisions, which can be costly and damaging.
:p How can data regressions manifest differently in the data space compared to software systems?
??x
In contrast to software systems where regressions are typically binary (e.g., request failure rates), data regressions often present as subtle statistical distortions. This makes them harder to detect because they may not immediately trigger obvious alerts.
For example:
```java
public boolean checkDataRegression(double oldMetric, double newMetric) {
    // Implement a threshold-based approach to detect changes in metrics
    if (Math.abs(oldMetric - newMetric) > THRESHOLD) {
        return true;
    }
    return false;
}
```
x??

---

#### Importance of True Orchestration
In data engineering, ingestion is a critical step that often involves complex and interdependent processes. As data complexity grows, simple cron jobs may not suffice; true orchestration becomes necessary to manage and coordinate these tasks effectively.
:p What does "true orchestration" refer to in the context of data engineering?
??x
True orchestration refers to a system capable of scheduling complete task graphs rather than individual tasks. This means starting each ingestion task at the appropriate scheduled time, allowing downstream processing steps to begin as soon as ingestion tasks are completed.
x??

---
#### Challenges and Best Practices for Ingestion
Ingestion is an engineering-intensive stage that often involves custom plumbing with external systems. It requires data engineers to build custom solutions using frameworks like Kafka or Pulsar, or homegrown solutions. However, the use of managed tools like Fivetran, Matillion, and Airbyte can simplify this process.
:p What are some best practices for handling ingestion in a complex data graph?
??x
Key best practices include:
- Using managed tools like Fivetran, Matillion, and Airbyte to handle heavy lifting.
- Developing high software development competency where it matters.
- Implementing version control and code review processes.
- Writing decoupled code to avoid tight dependencies on source or destination systems.
Example of decoupling logic in pseudocode:
```pseudocode
function startIngestionTask(task) {
    if (task.isValid()) {
        task.execute();
        notifyDownstreamTasks(task);
    }
}
```
x??

---
#### Complexity and Evolution of Ingestion Processes
The complexity of data ingestion has grown, particularly with the shift towards streaming data pipelines. Organizations need to balance correctness, latency, and cost in their data processing strategies.
:p How is data ingestion changing due to the move from batch to streaming pipelines?
??x
Data ingestion is evolving as organizations move from batch processing to real-time or near-real-time streaming pipelines. This requires new tools and techniques to handle out-of-order data, ensuring correctness and managing latency efficiently. For example, using frameworks like Apache Flink can help manage these challenges by providing a unified approach to both batch and stream processing.
x??

---
#### Importance of Robust Ingestion for Data Applications
Ingestion is fundamental for enabling advanced data applications such as analytics and machine learning models. Without robust ingestion processes, even the most sophisticated analytical tools cannot function effectively.
:p Why is robust data ingestion crucial for successful data applications?
??x
Robust data ingestion is crucial because it ensures that clean, consistent, and secure data flows to its destination. This data is then available for further processing and analysis, which is essential for building accurate analytics and machine learning models. Without reliable ingestion, the quality of downstream processes and final insights can be significantly compromised.
x??

---
#### Current Technologies in Ingestion
Managed tools like Airbyte, Fivetran, and Matillion have simplified the complexity of data ingestion by providing pre-built connectors and services that automate many aspects of this process.
:p What are some managed tools used for simplifying data ingestion?
??x
Some managed tools used for simplifying data ingestion include:
- **Airbyte**: Provides open-source connector frameworks to integrate with various sources.
- **Fivetran**: Offers automated ETL (Extract, Transform, Load) services.
- **Matillion**: A cloud-based platform that streamlines the process of building and managing data pipelines.
Example of using Airbyte in pseudocode:
```pseudocode
function configureAirbyteIntegration(source, destination) {
    airbyteClient = new AirbyteClient();
    integration = airbyteClient.createIntegration(source, destination);
    integration.startSync();
}
```
x??

---

#### What Is a Query?
Background context explaining the concept. Queries allow you to retrieve and act on data, which is essential for data engineering, data science, and analysis. They involve CRUD operations: read (SELECT), create (INSERT), update (UPDATE), delete (DELETE).
:p What is a query in the context of data engineering?
??x
A query is a fundamental operation that allows you to retrieve and act on data. It involves various CRUD (Create, Read, Update, Delete) operations.
x??

---

#### Data Definition Language (DDL)
Background context explaining DDL. Data definition language (DDL) commands are used to create, modify, or delete database objects such as tables, schemas, databases, etc. Common SQL DDL expressions include `CREATE`, `DROP`, and `ALTER`.
:p What does DDL stand for in the context of database operations?
??x
Data Definition Language is a set of commands used to define the structure of a database. It includes operations like creating new objects, modifying existing ones, or deleting them.
x??

---

#### Data Manipulation Language (DML)
Background context explaining DML. Data manipulation language (DML) commands are used to insert, update, delete, and select data within these database objects. Common DML commands include `INSERT`, `UPDATE`, `DELETE`, and `SELECT`.
:p What does DML stand for in the context of database operations?
??x
Data Manipulation Language is a set of commands used to manipulate data within database objects. It includes actions like inserting, updating, deleting, or selecting records.
x??

---

#### Data Control Language (DCL)
Background context explaining DCL. Data control language (DCL) allows you to manage and control access to the database by using SQL commands such as `GRANT`, `DENY`, and `REVOKE`.
:p What does DCL stand for in the context of database operations?
??x
Data Control Language is a set of commands used to manage and control access to the database. It includes granting, denying, or revoking permissions.
x??

---

#### Transaction Control Language (TCL)
Background context explaining TCL. Transaction control language (TCL) supports commands that control the details of transactions. Common TCL commands include `COMMIT` and `ROLLBACK`.
:p What does TCL stand for in the context of database operations?
??x
Transaction Control Language is a set of commands used to manage the execution and state changes of transactions within a database. It includes committing or rolling back transactions.
x??

---

#### The Life of a Query
Background context explaining query execution flow. When you execute a SQL query, it involves multiple steps: parsing, planning, optimization, and execution. The process ensures that your request is handled efficiently by the database engine.
:p How does a query work in a typical SQL environment?
??x
A query's life cycle includes several stages: parsing (validating the syntax of the query), planning (deciding how to execute it), optimization (choosing the most efficient execution plan), and execution (running the chosen plan). These steps ensure that your request is processed efficiently.
x??

---

#### Query Example: CRUD Operations
Background context explaining CRUD operations. CRUD stands for Create, Read, Update, Delete. In SQL, these are common DML commands used to manipulate data in a database.
:p Provide an example of each CRUD operation using SQL syntax.
??x
Sure, here's an example of each CRUD operation:

- **Create (INSERT)**: Adding new records into a table.
  ```sql
  INSERT INTO employees (id, name, position) VALUES (103, 'John Doe', 'Software Engineer');
  ```

- **Read (SELECT)**: Retrieving specific records from a table with conditions.
  ```sql
  SELECT * FROM employees WHERE department = 'Engineering';
  ```

- **Update (UPDATE)**: Modifying existing records in the database.
  ```sql
  UPDATE employees SET salary = 80000 WHERE id = 103;
  ```

- **Delete (DELETE)**: Removing records from a table.
  ```sql
  DELETE FROM employees WHERE id = 104;
  ```
x??

---

#### Transaction Control Language Example
Background context explaining TCL. TCL commands manage the commit and rollback of transactions to ensure data integrity and consistency.
:p Provide an example of using TCL commands in SQL syntax.
??x
Here's an example of using TCL commands:

- **Commit (COMMIT)**: Committing a transaction after all changes have been made.
  ```sql
  COMMIT;
  ```

- **Rollback (ROLLBACK)**: Rolling back the transaction if any error occurs during execution.
  ```sql
  ROLLBACK;
  ```
x??

---

#### Data Access Control Example
Background context explaining DCL. DCL commands control access to data and help manage who can read, write, or delete data in a database.
:p Provide an example of using DCL commands in SQL syntax.
??x
Here's an example of using DCL commands:

- **Grant (GRANT)**: Granting SELECT permission to Sarah on the `data_science_db` database.
  ```sql
  GRANT SELECT ON data_science_db TO user_name Sarah;
  ```

- **Revoke (REVOKE)**: Revoking SELECT permission from Sarah on the `data_science_db` database.
  ```sql
  REVOKE SELECT ON data_science_db FROM user_name Sarah;
  ```
x??

---

