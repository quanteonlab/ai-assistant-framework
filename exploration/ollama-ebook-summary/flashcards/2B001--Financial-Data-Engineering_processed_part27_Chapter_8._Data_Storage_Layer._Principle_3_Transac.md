# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 27)

**Starting Chapter:** Chapter 8. Data Storage Layer. Principle 3 Transactional Guarantee

---

---
#### Business Requirements for Data Storage Systems
Business requirements significantly influence the design and implementation of a data storage system (DSS). These requirements can vary widely, impacting various aspects such as ease of access, schema flexibility, user scalability, speed of data access, querying capabilities, storage needs, data sharing, and data aggregation.

Background context: Understanding business requirements is crucial because they directly affect the performance, reliability, and cost-effectiveness of the DSS. For instance, a requirement for high-speed data access may necessitate specific technologies or optimizations that differ from those needed for simpler queries.

:p What are some examples of data-related business requirements?
??x
Examples include:
- Ease of data access: A user-friendly interface for accessing data.
- Schema flexibility: The ability to add new fields as features evolve.
- User scalability: Handling a large number of concurrent users without degradation in performance.
- Speed of data access: Critical for high-frequency trading scenarios.
- Querying capabilities: Support for complex analytical queries and filters.
- Storage needs: Managing large datasets, such as market transactions.
- Data sharing: Collaborating with external entities.
- Data aggregation: For regulatory reporting purposes.

x??
---

#### Principles of DSS Design
The design principles outlined in this section provide a framework for selecting and implementing an appropriate data storage system (DSS) that aligns with business requirements. These principles emphasize the importance of involving the business team early in the decision-making process to ensure that the technical solution meets real-world needs.

Background context: Financial applications, such as trading platforms, depend heavily on well-designed DSS. Making incorrect choices can lead to costly and complex data migration projects later on. Understanding these principles helps financial data engineers make informed decisions based on a combination of business requirements and technical feasibility.

:p What is the first principle mentioned for designing a Data Storage System?
??x
The first principle is: Business Requirements.
This principle emphasizes that data engineering should be driven by business needs, and it's essential to include feedback from the business team in your decision-making process. By understanding business expectations, you can make better design choices.

x??
---

#### Iterative and Collaborative DSS Design Approach
The design of a Data Storage System (DSS) involves an iterative and collaborative approach where technical implementation aligns with business requirements through continuous feedback and adjustments.

Background context: This approach allows for flexibility in meeting evolving needs while ensuring that the final solution is robust and meets both current and future demands. Collaboration between data engineers and the business team ensures that all stakeholders have a clear understanding of the design process and its outcomes.

:p How does an iterative and collaborative approach benefit DSS design?
??x
An iterative and collaborative approach benefits DSS design by ensuring:
- Continuous feedback: Regular input from the business team helps in refining requirements and designs.
- Flexibility: The ability to adjust plans as new information becomes available or as business needs change.
- Alignment: A clear understanding of how technical decisions impact business goals, leading to more effective solutions.

x??
---

#### Data Modeling Process Overview

Background context explaining the data modeling process and its importance. The process is broken down into three phases: conceptual, logical, and physical.

:p Describe the three-phase approach of data modeling?
??x
The three-phase approach to data modeling includes:
1. **Conceptual Phase**: This phase involves a communicative process where data engineers and stakeholders discuss all their data needs without considering specific storage technologies.
2. **Logical Phase**: In this phase, the conceptual model is mapped to structured constructs such as rows, columns, tables, and documents that can be implemented by a Data Storage System (DSS).
3. **Physical Phase**: This phase translates the logical model into DSS language, often known as Data Definition Language (DDL), which involves choosing specific storage technologies like relational databases or document stores.

Code examples to illustrate each phase might not directly apply here since it's more about process rather than implementation.
??x
The answer with detailed explanations:
1. **Conceptual Phase**: During this phase, data engineers and stakeholders meet to discuss initial requirements without considering the specifics of how the data will be stored or which DSS will be used.
2. **Logical Phase**: Here, the conceptual model is mapped to structured constructs that can be implemented by a specific DSS. For example, you determine whether the data should be stored in tables (relational) or documents (NoSQL).
3. **Physical Phase**: This phase involves translating the logical model into the syntax and structure used by the chosen DSS, such as SQL for relational databases.

---

#### ACID Properties and Financial Standards

Background context: The text discusses how financial standards like ISO 20022 can be modeled using data modeling techniques, similar to how derivatives are structured. It introduces the concept of transactional guarantees through the lens of Data Storage Systems (DSS) ensuring accurate data states.

:p What is an ACID property and what does it ensure in a DSS?
??x
ACID properties ensure that database transactions are atomic, consistent, isolated, and durable. In simpler terms:
- **Atomicity**: A transaction is all-or-nothing; either it fully completes or none of its actions take effect.
- **Consistency**: The transaction must maintain the integrity of data constraints at all times.
- **Isolation**: Concurrency control ensures that transactions are not affected by other concurrent transactions.

For example, in a bank account system:
```java
public class BankAccount {
    private double balance;

    public void deposit(double amount) {
        if (amount > 0) {
            this.balance += amount;
            // Log the transaction
        }
    }

    public boolean withdraw(double amount) {
        if (this.balance >= amount && amount > 0) {
            this.balance -= amount;
            // Log the transaction
            return true;
        }
        return false;
    }
}
```
x??

---

#### Transactional Guarantee

Background context: The text explains that a Data Storage System (DSS) must ensure data consistency and reliability, reflecting real-world transactions accurately. It uses examples like purchasing a book and a car to illustrate this concept.

:p What is the transactional guarantee in DSS?
??x
Transactional guarantees ensure that all instructions within a single transaction are either fully committed or not executed at all, maintaining the integrity of the data state. This means that if you have $10,000 and attempt to buy a book for $50 and a car for $10,000 in one transaction, your final balance must be exactly $9,950 or zero, not negative.

For example:
```java
public class TransactionManager {
    public boolean executeTransaction(BookingRequest request) {
        // Check balances before the transaction
        if (bankAccount.checkBalance(request.getBook().getCost() + request.getCar().getCost()) > 0) {
            bankAccount.withdraw(request.getBook().getCost());
            bankAccount.withdraw(request.getCar().getCost());
            return true;
        }
        return false;
    }
}
```
x??

---

#### Atomicity in DSS

Background context: Atomicity ensures that a transaction is an all-or-nothing operation. If any part of the transaction fails, none of it should be applied.

:p What does atomicity ensure in a Data Storage System (DSS)?
??x
Atomicity ensures that every transaction either fully completes or nothing at all happens, preserving data integrity. For example, if you buy both a book and a car with your bank account balance:

```java
public class Transaction {
    private BankAccount bankAccount;
    private Book book;
    private Car car;

    public boolean execute() {
        // Check available balance
        if (bankAccount.getBalance() >= book.getPrice() + car.getPrice()) {
            try {
                bankAccount.withdraw(book.getPrice());
                bankAccount.withdraw(car.getPrice());
                book.purchase();
                car.purchase();
                return true;
            } catch (Exception e) {
                // Revert all changes in case of failure
                bankAccount.deposit(book.getPrice());
                bankAccount.deposit(car.getPrice());
            }
        }
        return false;
    }
}
```
x??

---

#### Consistency in DSS

Background context: Consistency ensures that data constraints and structural integrity are preserved throughout transactions. The text provides an example where a transaction fails because the available balance is insufficient.

:p What does consistency ensure in a Data Storage System (DSS)?
??x
Consistency ensures that all operations within a transaction comply with predefined rules or constraints, maintaining data integrity. For instance, if you have $10,050 but try to spend more than what's available:

```java
public class BankAccount {
    private double balance;

    public void withdraw(double amount) throws InsufficientFundsException {
        if (this.balance >= amount && amount > 0) {
            this.balance -= amount;
        } else {
            throw new InsufficientFundsException();
        }
    }
}
```
x??

---

#### Non-negative Constraint for Account Balance
In financial applications, ensuring that an account balance remains non-negative is crucial to prevent inconsistencies. This constraint typically falls under the responsibility of DSS engineers who establish and implement data consistency checks, constraints, and validations based on business requirements both at the DSS level as well as within the applications interacting with it.
:p What is a key requirement for maintaining account balances in financial transactions?
??x
Maintaining non-negative constraints ensures that account balances do not go below zero, preventing inconsistencies. This responsibility often lies with DSS engineers who design and enforce these checks to ensure data integrity.
x??

---

#### Snapshot Isolation Strategy
Snapshot isolation (SI) is a database isolation level that provides a consistent view of the database at the start of each transaction. Each transaction sees a snapshot of the data as it was when the transaction began, including all committed changes up to that point in time. This ensures that concurrent transactions do not see uncommitted changes from other sessions.
:p What is Snapshot Isolation (SI)?
??x
Snapshot isolation allows each transaction to view a consistent version of the database at its start time, incorporating all committed changes. This technique helps maintain data integrity during concurrent operations without conflicting with ongoing transactions.
x??

---

#### Durability in Data Storage Systems
Durability ensures that once a transaction has been successfully committed, it will be recorded permanently and remain unaffected by any subsequent system failures such as power outages or crashes. One reliable method to achieve durability is Write-Ahead-Logging (WAL), which logs all changes before they are written to disk.
:p How does Write-Ahead-Logging (WAL) ensure data durability?
??x
Write-Ahead-Logging (WAL) ensures that all transactions' changes are first recorded in a log file. Only after the change is logged and acknowledged as complete, it is applied to the actual database. This method guarantees that even if a crash occurs before writing the changes directly to disk, they can be recovered from the log upon restart.
x??

---

#### ACID Properties for Financial DSS
ACID properties (Atomicity, Consistency, Isolation, Durability) are critical in financial data storage systems. These properties ensure that transactions either complete entirely or not at all, maintaining consistent and reliable state changes within the system.
:p What does ACID stand for in data storage contexts?
??x
ACID stands for Atomicity, Consistency, Isolation, and Durability. These properties guarantee that financial transaction processing maintains a consistent state by ensuring that transactions either complete entirely or not at all.
x??

---

#### BASE Properties (Basically Available, Soft State, Eventually Consistent)
The BASE model is an alternative to ACID where applications prioritize availability over strict consistency. BASE systems ensure data can be accessed from other nodes even if one node fails and allow developers to manage data consistency manually rather than relying on the DSS.
:p What are the key characteristics of the BASE properties?
??x
The BASE properties include:
- Basically Available: Ensures that the system remains available as much as possible, even under load or failure conditions.
- Soft State: Permits temporary inconsistencies and allows them to be resolved over time.
- Eventually Consistent: Guarantees eventual consistency but does not ensure immediate synchronization of data across all nodes. This model is useful in scenarios where high availability and scalability are more critical than strict consistency.
x??

---

#### Application of ACID vs BASE
ACID properties are essential for financial transactions requiring consistent and reliable state changes, whereas BASE models might be suitable for applications prioritizing speed and scale over immediate consistency, such as trading platforms analyzing large datasets.
:p In what scenarios would you prefer a BASE model over an ACID model?
??x
You might prefer a BASE model when:
- High availability and fast data processing are more critical than strict consistency.
- The application can tolerate some level of inconsistency for faster performance and scalability.
- Speed and handling large volumes of data in real-time analysis are prioritized over immediate updates, as seen in trading platforms.
x??

---

#### CAP Theorem
Background context explaining the CAP theorem. According to the CAP theorem, a distributed data storage system (DSS) can guarantee at most two out of the following three features: Consistency, Availability, and Partition Tolerance during a network partition.

In ACID transactions, consistency means that the database remains in a valid state after a transaction. In contrast, in CAP, consistency refers to all nodes maintaining a consistent view of the data.

If there is a network partition, you must decide whether to prioritize:
- Consistency: Ensure balance consistency but ATM services will not be available.
- Availability: Allow each ATM to perform operations, risking an inconsistent state until resolution.

:p What does the CAP theorem state about distributed data storage systems?
??x
The CAP theorem states that in a distributed system during a network partition, you can guarantee at most two out of three features: Consistency (all nodes have a consistent view), Availability (the system responds to all requests), and Partition Tolerance (continued operation despite the failure).

In ACID transactions, consistency ensures database integrity. In CAP, consistency means all nodes see the same data.

Example scenarios:
- Blocking operations until network partition is resolved for consistency.
- Allowing limited operations like balance inquiry during a partition for availability.

```java
public class ATMSystem {
    private Map<String, Double> accounts; // Simulated distributed storage

    public void processTransaction(String accountID, double amount) {
        if (networkPartitionDetected()) { 
            // Handle partition
            return;
        }
        performOperation(accountID, amount);
    }

    private void performOperation(String accountID, double amount) {
        // Logic to update account balance and ensure consistency
    }

    private boolean networkPartitionDetected() {
        // Check for network partition condition
        return true; // Simulated check
    }
}
```
x??

---

#### PACELC Theorem
Background context explaining the PACELC theorem. This is an extension of CAP, considering two scenarios:
1. During a network partition: Trade off between consistency and availability.
2. Without partitions: Trade off between latency and consistency.

If data replication exists, there might be intermediate states before full consistency or availability.

:p What does PACELC stand for and what are its main aspects?
??x
PACELC stands for Partition Tolerance, Availability, Consistency, Eventual, Latency. Its main aspects involve:
1. During a network partition: Trade off between consistency and availability.
2. Without partitions: Trade off between latency and consistency.

When data replication is in place, the system might experience intermediate states before reaching full consistency or availability.

```java
public class DistributedStorageSystem {
    private Map<String, String> storage; // Simulated distributed storage

    public void writeData(String key, String value) {
        if (isNetworkPartition()) { 
            // Handle partition with eventual consistency
            return;
        }
        ensureConsistency(key, value);
    }

    private boolean isNetworkPartition() {
        // Check for network partition condition
        return true; // Simulated check
    }

    private void ensureConsistency(String key, String value) {
        // Logic to update storage and ensure consistency eventually
    }
}
```
x??

---

#### Consistency Tradeoffs in DSS Design
Background context explaining the need for trade-offs in designing distributed data storage systems. Different levels of consistency can be provided depending on the specific requirements.

Example: A bank with two ATMs connected over a network requires that customer balances never drop below zero. Two ATMs communicate to ensure consistency, but during a network partition, decisions must be made between full availability and full consistency.

:p What are the trade-offs in designing distributed data storage systems?
??x
In designing distributed data storage systems (DSS), there are several trade-offs:
- Between Consistency: Ensuring all nodes have the same view of data.
- Availability: Ensuring the system responds to all requests at all times.
- Partition Tolerance: Ensuring the system continues operating during network partitions.

During a network partition, designers must decide whether to block operations for consistency or allow them for availability. The decision can significantly impact how the system behaves under failures.

Example:
```java
public class BankSystem {
    private Map<String, Double> accounts; // Simulated distributed storage

    public void performTransaction(String accountID, double amount) {
        if (isNetworkPartition()) { 
            // Handle partition with availability or consistency
            return;
        }
        ensureConsistency(accountID, amount);
    }

    private boolean isNetworkPartition() {
        // Check for network partition condition
        return true; // Simulated check
    }

    private void ensureConsistency(String accountID, double amount) {
        // Logic to update accounts and ensure consistency
    }
}
```
x??

---

#### Scalability in DSS Design
Background context explaining scalability. A system is considered scalable if it can handle varying or increasing amounts of workload effectively.

Scalability can be achieved through storage scalability (handling more data) and compute scalability (handling more read/write requests).

:p What are the two main aspects of scalability in distributed data storage systems?
??x
The two main aspects of scalability in distributed data storage systems are:
1. **Storage Scalability**: The system's ability to handle increasing volumes of data without hitting space limitations.
2. **Compute Scalability**: The system’s efficiency in handling varying amounts of read/write requests.

Examples include vertical scaling (replacing a single machine with a more powerful one) and horizontal scaling (adding new nodes to an existing cluster).

```java
public class StorageSystem {
    private List<String> data; // Simulated storage

    public void addData(String newData) {
        if (!isFull()) { 
            // Handle adding data
            data.add(newData);
        } else {
            // Handle capacity limit
        }
    }

    private boolean isFull() {
        // Check for full condition
        return true; // Simulated check
    }
}

public class ComputeSystem {
    private List<Runnable> tasks;

    public void addTask(Runnable task) {
        if (!isMaxCapacity()) { 
            // Handle adding tasks
            tasks.add(task);
        } else {
            // Handle capacity limit
        }
    }

    private boolean isMaxCapacity() {
        // Check for max capacity condition
        return true; // Simulated check
    }
}
```
x??

---

#### Database Stress Testing and Benchmarking
Background context explaining the importance of database stress testing. It simulates large data generation, queries, and concurrent requests to identify operational limits.

Examples include industry-accepted standards like TPC-C and TPC-E which simulate order-entry environments and brokerage firm scenarios respectively.

:p What is a common method for evaluating system performance in distributed data storage systems?
??x
A common method for evaluating system performance in distributed data storage systems is through **database stress testing**. This involves simulating large amounts of data, queries, and concurrent requests to identify the operational limits and potential system failures.

Examples include:
- TPC-C: Simulates an order-entry environment where users submit transactions against a database.
- TPC-E: Simulates a brokerage firm with various customer transactions related to trades, account inquiries, and market research.

These benchmarks measure performance in terms of transactions processed per minute or seconds.

```java
public class StressTestScenario {
    private List<String> operations; // Simulated operations

    public void runStressTest() {
        for (int i = 0; i < MAX_OPERATIONS; i++) { 
            simulateOperation();
        }
    }

    private void simulateOperation() {
        // Logic to execute a simulated operation
    }
}
```
x??

---

#### Data Encryption at Rest
Background context: Ensuring strong database security is crucial, and data encryption at rest protects sensitive information such as customer account details stored within databases. This method involves encoding data so that it can only be read or accessed if a specific decryption key is provided.

:p What is the primary purpose of data encryption at rest?
??x
The main purpose of data encryption at rest is to protect sensitive data, such as customer account information, from unauthorized access when stored in databases. This ensures that even if an attacker gains physical access to the storage medium, they cannot read or use the encrypted data without the decryption key.
```java
// Example: Encrypting a String using Java's Cipher class
import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;

public class EncryptionExample {
    public static void main(String[] args) throws Exception {
        byte[] keyBytes = "ThisIsASecretKey".getBytes();
        SecretKeySpec secretKey = new SecretKeySpec(keyBytes, "AES");
        
        String originalString = "CustomerAccountInfo";
        Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encryptedValue = cipher.doFinal(originalString.getBytes());
    }
}
```
x??

---
#### Data Encryption in Transit
Background context: Protecting data during transmission from one point to another (e.g., payment information) is vital. Using encryption at this stage ensures that even if the data is intercepted, it remains unreadable without a decryption key.

:p What does data encryption in transit ensure?
??x
Data encryption in transit ensures that sensitive information, such as payment details, remains secure during transmission over networks or the internet. This prevents interception and unauthorized access by ensuring that the data can only be decrypted with the appropriate key at its destination.
```java
// Example: Implementing SSL/TLS for secure HTTPS connections
import javax.net.ssl.HttpsURLConnection;
import java.io.OutputStream;

public class SecureConnectionExample {
    public static void main(String[] args) throws Exception {
        HttpsURLConnection conn = (HttpsURLConnection) new URL("https://example.com/api").openConnection();
        
        // Sending a request to the server with sensitive data
        String input = "SensitiveData";
        OutputStream os = conn.getOutputStream();
        os.write(input.getBytes());
    }
}
```
x??

---
#### Database Access Permissions and Roles
Background context: Proper management of database access permissions ensures that only authorized users have the necessary privileges. This is critical for maintaining data integrity, confidentiality, and availability.

:p What are database roles?
??x
Database roles represent predefined sets of privileges or permissions. They allow administrators to manage user access more efficiently by assigning appropriate roles rather than individual permissions, simplifying security management.
```java
// Example: Assigning a role in SQL
-- SQL command to create a new role and assign it certain privileges
CREATE ROLE finance_role;
GRANT SELECT, INSERT ON accounts TO finance_role;

-- Assigning the role to users
GRANT finance_role TO alice, bob;
```
x??

---
#### Separation of Database Servers from Other Application Servers
Background context: This practice minimizes the attack surface by segregating database servers from application servers. It helps prevent unauthorized access and ensures that sensitive data is not exposed through application vulnerabilities.

:p Why separate database servers from other application servers?
??x
Separating database servers from other application servers reduces the risk of security breaches, as it limits the potential points of entry for attackers. This segregation prevents direct access to sensitive data stored in databases, even if an attacker manages to compromise an application server.
```java
// Example: Network architecture separation using VLANs
-- Network configuration example where database and application networks are isolated
network 192.168.100.0/24 {
    description "Application Server Network";
}

network 192.168.200.0/24 {
    description "Database Server Network";
}
```
x??

---
#### Backups and Data Recovery Plans
Background context: Regular backups ensure that data can be restored in case of loss or corruption. Effective recovery plans are essential for minimizing downtime and ensuring business continuity.

:p What is the primary purpose of a backup plan?
??x
The primary purpose of a backup plan is to safeguard against data loss by creating copies of critical data at regular intervals. This allows for quick restoration if data is lost due to hardware failure, accidental deletion, or other issues.
```java
// Example: Creating backups using a command-line tool
# Using MySQL's built-in mysqldump for database backup
mysqldump -u username -p --databases db1 db2 > /path/to/backup/db_backup.sql

# Scheduling backups with cron in Unix-based systems
* * * * * root /usr/bin/mysqldump -u username -p --databases db1 db2 > /path/to/backup/db_backup.sql
```
x??

---
#### Real-time Security Information and Event Monitoring
Background context: Continuous monitoring of security-related events helps in detecting and responding to potential threats promptly. This practice involves setting up alerts and automated response mechanisms.

:p What is the benefit of real-time monitoring?
??x
The benefit of real-time monitoring is that it enables organizations to detect security incidents and respond immediately, reducing the impact of potential threats before they can cause significant damage.
```java
// Example: Implementing real-time monitoring with Splunk
# Configuring a Splunk search for suspicious login attempts
index=main (sourcetype=login_access | stats count by username | where count > 10)

# Setting up an alert in Splunk based on the above search
alert email "Suspicious Login Detected" to admin@example.com when index=main sourcetype=login_access count > 10
```
x??

---

