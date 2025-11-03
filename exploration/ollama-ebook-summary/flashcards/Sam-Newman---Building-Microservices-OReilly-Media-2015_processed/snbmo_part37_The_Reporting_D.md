# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 37)

**Starting Chapter:** The Reporting Database

---

#### Handling Order Placement Failures
Background context: When an order is placed but fails to be captured, it can be handled by either retrying the operation later or aborting and compensating for the failure. This approach aligns with eventual consistency principles, where a system may not always be in a consistent state immediately after operations, but will eventually reach one.
:p How do we handle order placement failures?
??x
We can queue up the failed order and attempt to insert it into the warehouse’s picking table later. Alternatively, if the operation cannot be completed due to constraints, we might need to abort the entire operation and revert any committed transactions. In both cases, ensuring that the system eventually reaches a consistent state is key.
??x
---

#### Retrying Failed Operations
Background context: For some operations, retrying them at a later time can resolve issues if they are transient or due to temporary network or resource constraints. This approach uses eventual consistency principles where the system corrects itself over time.
:p What is an example of handling failed order placement through retry?
??x
We could queue up the order in a log file or another queue and attempt to insert it into the warehouse’s picking table again at a later date. If the network error was temporary, retrying might successfully complete the operation.
??x
---

#### Compensating Transactions for Failed Operations
Background context: When an operation fails but has already committed part of its transaction, a compensating transaction is required to undo the partial changes and revert the system back to its initial state. This ensures that no data is lost or left in an inconsistent state.
:p How do we handle a failed order insertion using a compensating transaction?
??x
We would issue a DELETE statement to remove the order from the database, effectively rolling it back. Additionally, we need to report via the UI that the operation failed and inform the user. The logic for handling this could be part of the same service or potentially distributed across different services.
```java
// Pseudocode for compensating transaction
public void handleFailedOrderInsertion() {
    try {
        // Remove order from database
        deleteFromDatabase("DELETE FROM orders WHERE id = ?", orderId);
        
        // Report failure to UI
        reportFailureToUI(orderId, "Order insertion failed.");
    } catch (Exception e) {
        log.error("Compensating transaction failed: ", e);
        // Handle retry or cleanup if necessary
    }
}
```
x??

---

#### Eventual Consistency in Long-Lived Operations
Background context: For long-lived operations where immediate consistency is not critical, eventual consistency can be employed. This approach allows the system to be inconsistent for a short period and resolve any inconsistencies over time.
:p How does eventual consistency apply to order placement?
??x
Eventual consistency means that even if an operation like order placement fails initially, it will eventually succeed once network or resource issues are resolved. The system can queue failed operations and retry them later until they complete successfully.
??x
---

#### Distributed Transactions for Immediate Consistency
Background context: For critical operations where immediate consistency is required across multiple systems, distributed transactions using a transaction manager can be used to ensure that all involved transactions either commit or roll back together.
:p What is the two-phase commit algorithm?
??x
The two-phase commit (2PC) algorithm involves two phases:
1. Voting phase: All participating nodes in the transaction are asked if they are ready to commit.
2. Commit phase: If all nodes agree, the transaction commits; otherwise, it rolls back.

Here’s an example of how 2PC might be implemented:
```java
// Pseudocode for Two-Phase Commit
public void twoPhaseCommit(TransactionManager manager) {
    // Voting Phase
    if (manager.prepare()) {
        // Commit Phase
        manager.commit();
    } else {
        manager.rollback();
    }
}
```
x??

---

#### Distributed Transaction Voting Mechanism
Background context explaining how participants (cohorts) in a distributed transaction communicate with a central transaction manager to decide on committing or rolling back transactions. The process involves each cohort signaling whether it can proceed, and based on consensus, the transaction manager decides the fate of all participating parties.
:p What does each participant do during the voting phase of a distributed transaction?
??x
During the voting phase, each participant (cohort) in the distributed transaction communicates with the central transaction manager to indicate if its local transaction can go ahead. This is essentially a yes or no vote on whether it's ready for the transaction to be committed.
```java
public class Cohort {
    public String vote() {
        // Logic to check if the local transaction can proceed
        boolean canProceed = checkLocalTransactionConditions();
        
        return canProceed ? "yes" : "no";
    }
}
```
x??

---

#### Outcome of Voting and Transaction Manager's Decision
Background context explaining that based on the votes received from all participants, the transaction manager decides whether to commit or roll back the entire transaction. If any participant gives a "no" vote, the transaction is rolled back.
:p What happens if one cohort responds with a "no" during voting?
??x
If any of the cohorts respond with a "no," the transaction manager will decide to rollback all participants and not proceed with the commit. This ensures that no partial transactions occur and maintains consistency among all involved parties.
```java
public class TransactionManager {
    public void processVotingResults(Map<String, String> votes) {
        boolean hasNoVote = votes.values().contains("no");
        
        if (hasNoVote) {
            // Rollback all participants
            rollbackAllParticipants(votes);
        } else {
            // Commit all participants
            commitAllParticipants(votes);
        }
    }
}
```
x??

---

#### Vulnerability to Outages and Resource Locking
Background context discussing the risk of outages where a transaction manager or a participant failing to respond can cause pending transactions to never complete. Also, the coordination process involves locks which can lead to resource contention, making scaling systems more challenging.
:p What are the risks associated with the central transaction manager going down?
??x
If the transaction manager goes down, any ongoing transactions will remain in limbo and never complete. This is because all participants halt until they receive instructions from the transaction manager. Consequently, this can lead to incomplete or lost transactions, which is a significant risk.
```java
public class TransactionManager {
    public void handleTransactionManagerDown() {
        // Logic to handle the scenario where the transaction manager goes down
        System.out.println("Transaction manager has gone down. Pending transactions will not complete.");
        
        // Optionally, retry logic or fallback mechanisms can be implemented here
    }
}
```
x??

---

#### Compensation and Consistency in Distributed Systems
Background context discussing how distributed transactions add complexity but might inhibit scaling. The text also mentions the importance of considering eventual consistency when possible to simplify system design.
:p When is it preferable to use compensating retry logic over distributed transactions?
??x
It is preferable to use compensating retry logic, especially for operations that do not require strict transactional consistency across all components. This approach allows systems to be more scalable and easier to manage by leveraging eventual consistency, where state eventually converges despite individual failures.
```java
public class OrderProcessingSystem {
    public void processOrder() {
        // Attempt to process the order with retry logic if necessary
        try {
            performOrderOperations();
        } catch (Exception e) {
            handleCompensatingActions(e);
        }
    }

    private void performOrderOperations() {
        // Perform operations like placing an order in the database and sending a confirmation email
    }

    private void handleCompensatingActions(Exception e) {
        // Logic to undo partial changes if necessary, ensuring eventual consistency
    }
}
```
x??

---

#### Concrete Concepts for Complex Transactions
Background context on creating specific concepts (like "in-process-order") to manage complex transactions in distributed systems. This approach helps with monitoring and managing these more intricate operations.
:p How can creating a concrete concept like an "in-process-order" aid in managing complex transactions?
??x
Creating a concrete concept such as an "in-process-order" allows you to encapsulate all logic related to processing the order, making it easier to manage exceptions and ensure consistency. This abstraction provides a clear structure and a single point of reference for handling operations like compensating transactions.
```java
public class InProcessOrder {
    private boolean isCommitted = false;
    
    public void commit() {
        if (!isCommitted) {
            // Logic to fully commit the order
            System.out.println("Order has been fully committed.");
            isCommitted = true;
        } else {
            System.out.println("Order is already committed.");
        }
    }

    public void rollback() {
        if (isCommitted) {
            // Logic to undo partial changes
            System.out.println("Rolling back order due to an error.");
            isCommitted = false;
        }
    }
}
```
x??

#### Reporting Database Overview
Background context: In a monolithic service architecture, reporting typically involves combining data from various parts of an organization to generate useful insights. The standard approach is to use a single database that serves both the main application and the reporting system.

:p What are the challenges in maintaining a single database for both primary and reporting purposes?
??x
The main challenges include schema management and optimization conflicts:

- **Schema Management**: A change in the schema affects both the live services and reporting systems, making changes difficult to coordinate.
- **Optimization Conflicts**: Optimizing the database for one use case (e.g., read-heavy operations) may negatively impact the other use case (e.g., transactional write operations).

These challenges often result in a suboptimal schema that doesn't perform well for either purpose.

x??

---
#### Read Replication
Background context: In standard architectures, read replicas are used to offload reporting queries from the main database. This setup helps prevent high load on the primary database during query execution.

:p How does using a read replica benefit the reporting system?
??x
Using a read replica benefits the reporting system by:

- Reducing the load on the primary database.
- Allowing more efficient and faster querying for reports without impacting the main service's performance.

However, this setup still faces challenges such as schema management conflicts and limited optimization options. The schema must support both live operations and reporting queries, which can be challenging to balance.

x??

---
#### Schema Management Challenges
Background context: In a monolithic architecture, the database schema serves as an API for both the application services and the reporting system. This shared schema can lead to complications when making changes, as any change in the schema affects multiple systems.

:p Why is managing the schema challenging in a monolithic setup?
??x
Managing the schema is challenging because:

- **Interconnected Systems**: Changes to the schema must be carefully coordinated between all services and the reporting system.
- **Impact on Performance**: Schema changes can have significant performance implications for both live operations and reporting queries.

This interdependence often results in compromises that may not optimize either use case effectively.

x??

---
#### Data Optimization Conflicts
Background context: The limitations of a single database schema mean that optimizations for one part of the system (e.g., read-heavy operations) can negatively impact another part (e.g., write-heavy operations). This creates a trade-off where optimizing for reporting might degrade the performance of the main application.

:p How does the single database model limit data optimization?
??x
The single database model limits data optimization because:

- **Shared Schema**: The schema must cater to both read and write operations, often leading to compromises.
- **Performance Trade-offs**: Optimizations that improve one aspect (e.g., faster reads) may degrade another (e.g., slower writes).

This can result in a suboptimal database design where the schema is not ideal for either primary application use or reporting.

x??

---
#### Exploring New Database Technologies
Background context: The monolithic database model might not be optimal when dealing with diverse data models and storage needs. Different applications may benefit from different types of databases (e.g., graph, document, column-oriented).

:p What are the benefits of exploring new database technologies for reporting?
??x
Exploring new database technologies offers several benefits:

- **Improved Data Modeling**: Using a graph database like Neo4j or a document store like MongoDB can better fit certain data models.
- **Scalability and Performance**: Column-oriented databases like Cassandra can provide better performance for large volumes of data.

These technologies allow for more specialized optimizations that might not be possible with traditional relational databases, leading to improved efficiency and flexibility in handling diverse data needs.

x??

---

#### Data Retrieval via Service Calls
Data retrieval for reporting systems often involves fetching data from various sources using API calls. For simple reports, this can be straightforward, but complex scenarios require pulling large volumes of data across multiple systems, leading to performance issues and challenges with maintaining accuracy.

:p What is the main challenge when using API calls for data retrieval in a reporting system?
??x
The main challenge is that the APIs exposed by various microservices may not be designed for efficient reporting use cases. This can lead to inefficient data retrieval methods, such as making multiple API calls to gather all necessary data, which can be slow and resource-intensive.

For example, if you need customer data, a service might only allow fetching customers by ID or searching by fields, but not in bulk. You would have to make a separate call for each customer, which is inefficient.
x??

---

#### Batch Data Retrieval
To handle large volumes of data more efficiently, batch APIs can be used. These APIs are designed to retrieve multiple records at once rather than individual ones.

:p How does a batch API improve the efficiency of data retrieval?
??x
A batch API improves efficiency by allowing you to pass a list of IDs or other identifiers in one request, reducing the number of calls needed and thus improving performance. For example, instead of making separate requests for each customer ID, you can send a single request with all the required IDs.

Example pseudocode:
```java
// Pseudocode for using batch API
List<Long> customerIDs = getCustomerIDs();
BatchRequest batchRequest = new BatchRequest(customerIDs);
response = postToAPI(batchRequest);

if (response.status == 202) {
    // Wait until the request is completed
    while (!isRequestCompleted(response.location)) {}
    
    response = getFromAPI(response.location); // Check for status 201 to ensure data is ready
    if (response.status == 201) {
        exportData(response.data);
    }
}
```
x??

---

#### Reporting System Challenges with APIs
Reporting systems often face challenges when interacting with APIs due to the design of these APIs, which may not be optimized for reporting tasks.

:p What are some challenges in using existing APIs for a reporting system?
??x
Some challenges include:
1. **Inefficient API Design**: APIs might only support fetching data by ID or specific criteria rather than in bulk.
2. **Multiple Calls Required**: To gather comprehensive data, multiple calls may be necessary, leading to inefficiency and increased load on the service being queried.
3. **Cache Misses**: Reporting often involves accessing less frequently used data (the long tail), which can lead to cache misses despite having cache headers.

Example:
If you need customer data for a 24-month report, making individual API calls for each customer could result in many cache misses, slowing down the process.
x??

---

#### Batch Export Resource Endpoint
A batch export resource endpoint allows users to request large datasets by sending a POST request with a list of identifiers.

:p What is the purpose of a batch export resource endpoint?
??x
The purpose of a batch export resource endpoint is to allow efficient data retrieval for reporting purposes. Instead of making multiple API calls, this endpoint accepts a list of IDs or other criteria and processes them in one go, returning an HTTP 202 response indicating that the request has been accepted but not yet processed.

Once processing is complete, it returns an HTTP 201 Created status along with the data. This approach avoids frequent cache misses and reduces API load by batching requests into fewer calls.

Example:
```java
// Pseudocode for using batch export endpoint
BatchCustomerExportRequest request = new BatchCustomerExportRequest(customerIDs);
response = postToAPI(request);

if (response.status == 202) {
    // Poll the resource until it is ready
    while (!isRequestCompleted(response.location)) {}
    
    response = getFromAPI(response.location); // Check for status 201 to ensure data is ready
    
    if (response.status == 201) {
        fileLocation = response.data;
        // Fetch and process the CSV file from the shared location
    }
}
```
x??

---

#### Traditional Reporting Needs vs. Batch Export
While batch export can be useful, it might not always be the best solution for traditional reporting needs due to complexity and potential inefficiencies.

:p Why is a batch export resource endpoint less favored for traditional reporting systems?
??x
A batch export resource endpoint may introduce unnecessary complexity for simple reporting tasks and does not scale as well for traditional reporting. It is more suited for specific scenarios, such as data exports or bulk insertions, rather than routine reporting needs where simplicity and performance are crucial.

For example, a simpler solution might involve periodically fetching data into an SQL database using a scheduled job, which can be easier to integrate with third-party tools and maintain.
x??

---

---
#### Data Pump Mechanism
Background context: The text discusses an alternative approach to data retrieval for reporting, where a standalone program (data pump) pushes data from the source database directly into a central reporting database. This method reduces overhead associated with HTTP calls and simplifies integration management.

:p What is the main benefit of using a data pump in the context described?
??x
The primary benefit is reduced overhead compared to making numerous HTTP requests, as well as simplified management by having the same team handle both the service's internal schema and the reporting database. This approach minimizes coupling issues that typically arise when multiple systems integrate with a shared database.
x??

---
#### Implementation of Data Pump
Background context: The data pump is implemented using command-line tools or scripts triggered via Cron jobs, ensuring it runs periodically to update the central reporting database. It requires intimate knowledge of both source and target schemas to map one schema to another.

:p How can a simple command-line program be set up to trigger a data pump?
??x
A simple command-line program could use shell commands like `cron` or other scheduling tools to run at specified intervals. For example, using Bash:
```bash
# Example cron entry in crontab file
0 5 * * * /path/to/data-pump.sh >> /path/to/logfile.log 2>&1
```
This cron job runs the script `data-pump.sh` every day at 5 AM and logs output to a specified log file.
x??

---
#### Version Control for Data Pump
Background context: To ensure consistency and reduce risks, the data pump is version-controlled along with the service code. Builds of the data pump are created as an additional artifact in the service build process.

:p Why should the data pump be version-controlled alongside the service?
??x
Version-controlling the data pump ensures that any changes or updates to how data is extracted and transformed from the source database to the reporting schema are tracked and managed systematically. This practice helps maintain a history of changes, facilitating easier rollback if issues arise post-deployment.
x??

---
#### Materialized Views for Schema Segmentation
Background context: For relational databases, materialized views can be used to create separate schemas for each service in the reporting database. This approach allows exposing only necessary parts of the schema while maintaining performance.

:p How do materialized views help in managing a segmented schema?
??x
Materialized views allow creating aggregated or transformed views of data from various services within the reporting database, minimizing direct access to underlying schemas. This can be particularly useful for large-scale applications with multiple data sources needing distinct reporting mechanisms.
x??

---
#### Challenges and Mitigations
Background context: While data pumps offer benefits like reduced overhead and easier management, challenges such as schema coupling persist. Implementing techniques like materialized views can mitigate some of these issues but may introduce additional complexities.

:p What are the primary downsides of implementing a segmented schema approach?
??x
The main downsides include increased complexity in managing different schemas, potential performance degradation due to frequent updates, and the challenge of maintaining backward compatibility when altering underlying data structures.
x??

---

