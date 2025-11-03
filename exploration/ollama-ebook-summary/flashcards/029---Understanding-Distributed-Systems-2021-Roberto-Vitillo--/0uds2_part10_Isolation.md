# Flashcards: 029---Understanding-Distributed-Systems-2021-Roberto-Vitillo--_processed (Part 10)

**Starting Chapter:** Isolation

---

#### Idempotent Messages
Background context: In a workflow implementing asynchronous transactions, messages might appear twice at the receiving end. This can lead to duplication issues if actions are not designed to handle such cases. To address this, participants need to make their messages idempotent, meaning that applying the operation multiple times has the same effect as applying it once.
:p What is meant by an idempotent message in the context of asynchronous transactions?
??x
An idempotent message is one where repeating the action (like sending a message) does not change the outcome beyond the initial application. For example, if a message represents a payment transaction, sending the same message twice should result in only one payment being processed, not two.
??x
To ensure messages are idempotent, you might design your system so that each message contains enough information to handle retries without causing side effects.

```java
public class PaymentMessage {
    private String orderId;
    private BigDecimal amount;

    public PaymentMessage(String orderId, BigDecimal amount) {
        this.orderId = orderId;
        this.amount = amount;
    }

    // Methods to ensure idempotency like checking if the payment has already been processed
}
```
x??

---

#### Serverless Cloud Compute Services for Workflows
Background context: Implementing complex workflows can be challenging, especially when dealing with synchronous and asynchronous transactions. Serverless cloud compute services like AWS Step Functions or Azure Durable Functions simplify this by providing managed environments where you can define and run workflows without managing the underlying infrastructure.
:p How do serverless cloud compute services help in implementing workflows?
??x
Serverless cloud compute services automate much of the infrastructure management required to implement complex workflows. They allow developers to focus on defining the logic of the workflow using a visual editor or programming interfaces, abstracting away the details of scaling and managing servers. This makes it easier to build scalable and fault-tolerant applications without the overhead of traditional server management.
??x
For instance, AWS Step Functions use state machines to define workflows, allowing you to chain multiple tasks together with conditional logic.

```json
{
    "Comment": "A simple example of a state machine",
    "StartAt": "SendEmail",
    "States": {
        "SendEmail": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:<region>:<account-id>:function:<function-name>",
            "Next": "SendSNS"
        },
        "SendSNS": {
            "Type": "Task",
            "Resource": "arn:aws:sns:<region>:<account-id>:<topic-name>",
            "End": true
        }
    }
}
```
x??

---

#### Isolation in Asynchronous Transactions
Background context: The two-phase commit (2PC) protocol provides strong isolation guarantees, but it is blocking. To achieve non-blocking transaction processing with Sagas, we had to give up on these traditional ACID guarantees. However, it turns out that there are ways to work around the lack of isolation.
:p How can the lack of isolation in asynchronous transactions be mitigated?
??x
The lack of isolation in asynchronous transactions can be mitigated using techniques such as semantic locks. The idea is that any data modified by Sagas is marked with a "dirty" flag, which is only cleared at the end of the transaction when it completes. Another transaction attempting to access a dirty record would either fail and roll back its changes or block until the dirty flag is cleared.
??x
This approach can introduce deadlocks, requiring strategies to mitigate them. For example, you could implement a timeout mechanism where if a lock does not clear within a certain period, the transaction fails.

```java
public class LockManager {
    private Map<String, Boolean> locks;

    public boolean tryLock(String resource) {
        // Logic to acquire and release locks
        return locks.putIfAbsent(resource, true) == null;
    }

    public void unlock(String resource) {
        if (locks.remove(resource) != null) {
            // Handle successful unlock
        }
    }
}
```
x??

---

#### Semantic Locks for Isolation in Sagas
Background context: To work around the lack of isolation guarantees in asynchronous transactions, semantic locks can be used. Any data modified by Sagas is marked with a "dirty" flag that gets cleared only when the transaction completes. Transactions accessing dirty records may fail and rollback or block until the flag is cleared.
:p How do semantic locks work to provide some level of isolation?
??x
Semantic locks work by marking data as "dirty" whenever it's modified during a transaction. This means that any subsequent transactions attempting to modify the same data will detect the dirty state and handle it accordingly. If another transaction tries to access a dirty record, it can either fail and rollback its changes or block until the dirty flag is cleared.
??x
This approach ensures that modifications are processed atomically, even in an asynchronous setting.

```java
public class TransactionManager {
    private Map<String, Boolean> dirtyRecords;

    public void markRecordAsDirty(String recordId) {
        dirtyRecords.put(recordId, true);
    }

    public boolean isRecordDirty(String recordId) {
        return dirtyRecords.containsKey(recordId);
    }
}
```
x??

#### Functional Decomposition
Functional decomposition involves breaking an application into separate services, each with its own well-defined responsibility. This approach enhances modularity and maintainability.

Background context: When developing applications, it's common to decompose code into functions, classes, or modules. Extending this idea, an entire application can be broken down into smaller, independently deployable services.

:p What is the main goal of functional decomposition in distributed systems?
??x
The primary goal of functional decomposition is to improve modularity and maintainability by breaking down a large application into smaller, more manageable pieces that can be deployed and scaled independently.
x??

---

#### API Gateway for Service Communication
An API gateway acts as a proxy for the application, routing, composing, and translating requests.

Background context: After decomposing an application into services, external clients need to communicate with these services. An API gateway facilitates this communication by providing a single entry point.

:p How does an API gateway help in managing communication between external clients and decomposed services?
??x
An API gateway helps manage communication by acting as a central hub that routes, composes, and translates requests from external clients to the appropriate service. It provides a unified interface for interacting with multiple services.
x??

---

#### Read-Path vs Write-Path Decoupling
Decoupling an API’s read path from its write path allows using different technologies that fit specific use cases.

Background context: In many applications, reading and writing data can be handled differently due to performance or technology constraints. By separating these paths, the system can optimize each part independently.

:p Why is decoupling the read-path from the write-path beneficial?
??x
Decoupling the read-path from the write-path allows using different technologies optimized for specific tasks. For example, the read path might use caching and asynchronous processing to enhance performance, while the write path could handle transactions and consistency.
x??

---

#### Asynchronous Messaging Channels
Asynchronous messaging channels decouple producers and consumers by sending messages between parties.

Background context: In a distributed system, direct synchronous communication can be problematic due to availability issues. Asynchronous messaging provides a more reliable way to communicate.

:p How do asynchronous messaging channels help in a distributed system?
??x
Asynchronous messaging channels help by decoupling the producer and consumer of messages. This ensures that producers can send messages even if consumers are temporarily unavailable, improving overall reliability.
x??

---

#### Partitioning Strategies
Partitioning involves distributing data across multiple nodes to handle larger datasets.

Background context: As applications grow, a single node may not be able to store or process all the data. Partitioning strategies like range and hash partitioning can help distribute the load more effectively.

:p What is the purpose of partitioning in distributed systems?
??x
The purpose of partitioning is to distribute data across multiple nodes, enabling larger datasets to be managed and processed more efficiently.
x??

---

#### Load Balancing Across Nodes
Load balancing distributes incoming requests among multiple nodes to manage traffic effectively.

Background context: Load balancers can route requests to different servers based on various criteria like round-robin or least connections. This ensures no single server becomes a bottleneck.

:p How does load balancing contribute to managing traffic in distributed systems?
??x
Load balancing contributes by distributing incoming requests among multiple nodes, ensuring that no single node is overwhelmed and maintaining consistent performance across the system.
x??

---

#### Data Replication Across Nodes
Data replication involves creating copies of data on multiple nodes to ensure availability.

Background context: Replicating data can help in achieving high availability and fault tolerance. Different strategies like single-leader, multi-leader, or leaderless systems are available with varying trade-offs.

:p What is the main benefit of replicating data across nodes?
??x
The main benefit of replicating data across nodes is improved availability and fault tolerance. Replicas ensure that even if one node fails, another can take over without service interruption.
x??

---

#### Caching Strategies
Caching involves storing frequently accessed data in a faster-access storage layer to improve performance.

Background context: In-memory caches like Redis or in-process caches can significantly reduce the load on database servers by caching frequently accessed data. However, they come with challenges like cache invalidation and consistency.

:p What are the key benefits of using caching in distributed systems?
??x
The key benefits of using caching include improved performance through faster access to frequently used data, reduced load on backend databases, and enhanced scalability by offloading read operations.
x??

---

#### Monolithic Application Challenges
Background context: An application typically starts its life as a monolith, often a single-stateless web service that exposes a RESTful HTTP API and uses a relational database. It is composed of components or libraries implementing different business capabilities. As more feature teams contribute to the same codebase, the complexity increases over time.
:p What are some challenges faced by teams working on a monolithic application?
??x
The challenges include increased coupling between components as more features are added, leading to frequent stepping on each other’s toes and decreased productivity. The codebase becomes complex, making it difficult for anyone to fully understand every part of the system, which complicates implementing new features or fixing bugs.
??? 
---

#### Componentization of Monolithic Applications
Background context: Even if the backend is componentized into different libraries owned by different teams, changes still require redeploying the service. A change that introduces a bug like a memory leak can potentially affect the entire service. Rolling back a faulty build affects all teams' velocity.
:p How does componentization help in managing changes and bugs in a monolithic application?
??x
Componentization helps in managing changes by allowing different libraries to be updated independently, reducing the risk of breaking other components. However, it still requires redeploying the service whenever there is a change. A memory leak or similar bug can affect the entire service, emphasizing the need for careful testing and rollback strategies.
??? 
---

#### Microservices Architecture
Background context: To mitigate the growing pains of monolithic applications, splitting them into independently deployable services that communicate via APIs is one approach. This architectural style is called the microservice architecture. The term "micro" can be misleading as it does not imply small in size or functionality.
:p What is the key difference between a monolithic application and a microservices architecture?
??x
In a microservices architecture, independent services are deployed separately and communicate via APIs, creating boundaries that are harder to violate compared to components within the same process. This approach decouples services, reducing the impact of changes or bugs on other parts of the system.
??? 
---

#### Service-Oriented Architecture (SOA)
Background context: The term microservices is sometimes seen as misleading due to its implication of small size and functionality. A more appropriate name for this architecture could be service-oriented architecture (SOA), but it comes with its own set of baggage.
:p Why might "microservices" be considered a misleading term?
??x
The term "microservices" can be misleading because the services do not necessarily have to be small or lightweight; they are more about being independently deployable and loosely coupled. The true essence is in the service-oriented architecture, where services are designed to be self-contained and communicate via well-defined APIs.
??? 
---

