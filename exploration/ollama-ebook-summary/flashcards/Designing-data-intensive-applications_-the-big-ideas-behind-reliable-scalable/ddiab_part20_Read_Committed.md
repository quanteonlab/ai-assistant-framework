# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 20)

**Starting Chapter:** Read Committed

---

#### Weak Isolation Levels Overview
Background context: The passage discusses various levels of transaction isolation, particularly focusing on how weaker forms of isolation are used to balance between performance and correctness. It mentions that while serializable isolation ensures transactions run as if they were executed one after another, it comes at a cost in terms of performance. Therefore, databases often use weaker levels of isolation to improve concurrency.
:p What is the main issue with using serializable isolation?
??x
Serializable isolation guarantees that transactions behave as if they are executed sequentially, which can be very costly in terms of performance. The passage suggests this level of isolation is not practical for all systems due to its high overhead.
x??

---
#### Read Committed Isolation Level
Background context: Read committed ensures that a transaction only reads data that has been committed and does not overwrite uncommitted data. This helps prevent dirty reads and writes, making the database state more predictable.
:p What are two guarantees provided by read committed isolation?
??x
1. No dirty reads: A transaction cannot see uncommitted changes made by another transaction.
2. No dirty writes: A transaction will not overwrite uncommitted data in other transactions.
x??

---
#### Preventing Dirty Reads
Background context: In the read committed isolation level, a transaction must ensure that it does not perform a dirty read, where it sees partially updated data. This can lead to confusion and incorrect decisions by users or other transactions.
:p How does preventing dirty reads benefit database operations?
??x
Preventing dirty reads ensures that transactions see consistent and committed data, which helps avoid confusion and incorrect decision-making. It maintains the integrity of the transaction and prevents reading uncommitted changes, making it easier to reason about the state of the database.
x??

---
#### Preventing Dirty Writes
Background context: A dirty write occurs when a transaction overwrites an uncommitted value with another transaction's committed data. In read committed isolation, preventing dirty writes ensures that later transactions do not overwrite uncommitted values, which can lead to inconsistent states in the database.
:p What is a scenario where dirty writes might occur?
??x
Consider two concurrent transactions updating the same row in a database. If one transaction has written but not yet committed and another transaction tries to update the same row, this could result in a dirty write if the second transaction overwrites the uncommitted value from the first.
x??

---
#### Implementing Read Committed Isolation
Background context: Databases implementing read committed isolation typically use row-level locks to prevent dirty writes. A transaction must acquire a lock on an object before modifying it and hold that lock until the transaction is committed or aborted.
:p How do databases enforce read committed by default in many systems?
??x
Databases implement read committed by using row-level locks for write operations. When a transaction wants to modify a particular object, it acquires a lock and holds it until the transaction completes. Only one transaction can hold a lock on any given object at a time, preventing dirty writes.
x??

---
#### Example of Dirty Writes
Background context: The passage provides an example where two transactions update different parts of a database (e.g., updating a car listing and sending an invoice) but might still result in inconsistent states due to uncommitted changes.
:p How can the race condition between counter increments lead to incorrect outcomes?
??x
In a scenario with concurrent increment operations, if one transaction commits before another, the latter may overwrite the former's uncommitted value. This can cause inconsistencies, such as an invoice being sent to the wrong buyer in the car sales example provided.
x??

---

#### Read Committed Isolation
Background context explaining the concept. Read committed isolation ensures that transactions only see committed data, preventing dirty reads but not addressing all concurrency issues. It is used by IBM DB2 and Microsoft SQL Server with `read_committed_snapshot=off` configuration.

:p What is read committed isolation?
??x
Read committed isolation is a transaction isolation level where transactions can only read data that has been committed by other transactions. This means that uncommitted data modifications are not visible, which prevents dirty reads. However, it does not prevent non-repeatable reads and phantom reads.
x??

---
#### Non-Repeatable Read (Read Skew)
Background context explaining the concept. Non-repeatable read or read skew occurs when a transaction sees different versions of the same data during its execution.

:p What is non-repeatable read in the context of database transactions?
??x
Non-repeatable read, also known as read skew, happens when a transaction reads a value that has been updated by another transaction. This can lead to inconsistent results if the transaction needs to read the same data multiple times.
For example:
- Alice observes her $1,000 in two accounts at different moments and sees only $900 instead of the expected $1,000 after a transfer between the accounts.

x??

---
#### Read Skew Example
Background context explaining the concept. An illustration of how read skew can occur.

:p Explain the example given for non-repeatable reads or read skew.
??x
In the provided example:
- Alice has two bank accounts each with $500, totaling$1,000.
- A transaction transfers $100 from one account to the other.
- If Alice checks her balances at a time when the transfer is being processed, she might see an inconsistent state: one account with $400 and the other with$600.

x??

---
#### Snapshot Isolation
Background context explaining the concept. Snapshot isolation maintains consistency across transactions by remembering old values of data objects during write operations.

:p How does snapshot isolation prevent non-repeatable reads?
??x
Snapshot isolation prevents non-repeatable reads by maintaining a snapshot of the data before any write operation occurs. For every object written, the database keeps track of both the old committed value and the new value set by the transaction holding the write lock. Other transactions reading the same object during the ongoing transaction get the old value until the new value is fully committed.

x??

---
#### Repeatable Read vs. Snapshot Isolation
Background context explaining the concept. Repeatable read ensures that a transaction can read the same data multiple times, and it will see the same values throughout its duration.

:p What is repeatable read in database transactions?
??x
Repeatable read guarantees that once a transaction reads some data, that data cannot be modified by other transactions until the current transaction commits. This means that if Alice starts reading account balances at time T1 and continues to read them later (at time T2), she will see the same values as seen at T1.

x??

---
#### Concurrency Issues in Read Committed Isolation
Background context explaining the concept. Even with read committed isolation, there can still be concurrency issues such as non-repeatable reads and phantom reads.

:p What are some concurrency issues that read committed isolation does not fully address?
??x
Read committed isolation does not prevent non-repeatable reads (read skew) and phantom reads. A transaction might observe different values for the same data during its execution, leading to inconsistencies. For example:
- Non-repeatable reads: Alice sees different account balances at different times.
- Phantom reads: New rows appear or disappear in a table that matches the criteria of a query.

x??

---

#### Snapshot Isolation Overview
Background context: When running analytic queries or integrity checks, it is essential to ensure that transactions observe a consistent snapshot of the database. This prevents nonsensical results due to data being observed at different points in time by various transactions.

:p What is snapshot isolation?
??x
Snapshot isolation ensures each transaction reads from a consistent snapshot of the database—specifically, the state committed when the transaction started. Even if other transactions modify the data after it was read, the reading transaction sees only the old version.
x??

---

#### Consistent Snapshot Concept
Background context: With snapshot isolation, transactions see the state of the database as it existed at a specific point in time rather than the current state during execution.

:p How does snapshot isolation work?
??x
Each transaction starts with a consistent snapshot of the database's state. This means that all reads within the transaction are based on this snapshot, not the current state. Other transactions can modify the data, but these changes do not affect the currently executing transaction.
x??

---

#### Implementation of Snapshot Isolation
Background context: Implementing snapshot isolation involves managing multiple versions of database objects to ensure consistency across different transactions.

:p What is a key principle of implementing snapshot isolation?
??x
A key principle is that readers never block writers, and writers never block readers. This allows the database to handle long-running read queries on a consistent snapshot while processing writes normally.
x??

---

#### Multi-Version Concurrency Control (MVCC)
Background context: MVCC is used in snapshot isolation to maintain multiple versions of an object, allowing transactions to see different states.

:p How does MVCC support snapshot isolation?
??x
MVCC supports snapshot isolation by maintaining several committed versions of database objects. This allows transactions to read from a specific point in time without blocking other writes or reads.
x??

---

#### Example of MVCC in PostgreSQL
Background context: PostgreSQL uses MVCC for snapshot isolation, storing multiple versions of an object.

:p How is MVCC implemented in PostgreSQL?
??x
In PostgreSQL, MVCC is used by keeping several versions of an object. For example, a transaction can read the committed version as it existed at the start of the transaction. This allows long-running reads to operate on consistent snapshots without blocking writes.
Example code in pseudocode:
```pseudocode
// Pseudo-code for MVCC implementation in PostgreSQL
function getTransactionSnapshot(transaction_id) {
    return getCommittedVersionsBefore(transaction_id);
}

function readCommittedVersion(obj, snapshot) {
    return getObjectFromVersion(obj, snapshot.committed_version);
}
```
x??

---

#### Read Committed vs. Snapshot Isolation
Background context: While both ensure no dirty reads, they differ in how they manage snapshots.

:p How do read committed isolation and snapshot isolation differ?
??x
Read committed isolation ensures that a transaction sees the results of any committed transactions that occurred before it started. Snapshot isolation goes further by allowing each transaction to see the database state at a specific point in time, even if changes occur after the transaction starts.
x??

---

#### Vacuum Process for PostgreSQL
Background context: The vacuum process in PostgreSQL manages storage and prevents overflow issues.

:p What is the role of the vacuum process in PostgreSQL?
??x
The vacuum process in PostgreSQL performs cleanup to ensure that overflow does not affect data. It helps manage storage by removing dead tuples, which can prevent issues related to transaction IDs overflowing after approximately 4 billion transactions.
x??

---

#### Transaction IDs and Multi-Version Concurrency Control (MVCC)
Transaction IDs are assigned to each transaction started. Each write operation is tagged with the writer's transaction ID. This system allows for maintaining a snapshot of the database at any point in time, supporting transactions without blocking others.

:p How does the MVCC system ensure that transactions can read a consistent snapshot?
??x
The MVCC system ensures consistency by using transaction IDs to determine which versions of data are visible to each transaction. A transaction can only see changes made by earlier transactions and ignores writes from later transactions or aborted transactions. This way, multiple transactions can read the database simultaneously without interfering with each other.

```java
public class Transaction {
    private int txid; // Unique transaction ID
    
    public void write(int txid, Object data) {
        this.txid = txid;
        // Tagging data with current txid and storing in the MVCC system
    }
    
    public Object read() {
        // Return data based on visibility rules using current txid
        return data; 
    }
}
```
x??

---

#### Visibility Rules for Snapshot Isolation
Visibility rules are defined to ensure that a transaction sees a consistent snapshot of the database. These rules ignore writes from later transactions, aborted transactions, and any writes from transactions with IDs newer than the current transaction.

:p What visibility rules define how a transaction observes the database?
??x
Visibility rules dictate which versions of data a transaction can see based on the transaction ID (txid) at the start of each transaction. The rules are as follows:
1. Ignore all writes made by in-progress transactions with newer txids.
2. Ignore all writes made by aborted transactions.
3. Only consider writes from earlier txids, even if those transactions commit later.

```java
public class TransactionManager {
    public boolean isVisible(int currentTxid, int readTxid) {
        return readTxid <= currentTxid; // Check if the transaction is visible
    }
}
```
x??

---

#### Multi-Version Objects and Garbage Collection
In a multi-version system, objects are not deleted but marked for deletion. A garbage collection process removes these marked rows at an appropriate time to free up space.

:p How does the database handle deletions in a snapshot isolation environment?
??x
Deletions in a snapshot isolation environment do not actually remove data from the database; instead, they mark the row as deleted using a `deleted_by` field. The garbage collection process later identifies and removes these marked rows to free up space.

```java
public class Row {
    private int balance;
    private int created_by;
    private int deleted_by = -1; // Marked for deletion
    
    public void markForDeletion(int txid) {
        this.deleted_by = txid;
    }
    
    public boolean isMarkedForDeletion() {
        return deleted_by != -1;
    }
}
```
x??

---

#### Update as Delete and Create
An update operation in a snapshot isolation database translates into deleting the old version of the row and creating a new one. This ensures that multiple transactions can read different versions of the data simultaneously.

:p How is an update operation implemented in a multi-version system?
??x
In a multi-version system, an update operation does not modify the existing row but instead deletes the old row version and creates a new one with updated values. This approach allows concurrent reads to see consistent snapshots without interference from other writes.

```java
public class Account {
    public void updateBalance(int accountId, int newBalance) {
        Row oldRow = getRow(accountId); // Get the current balance
        oldRow.markForDeletion(currentTxid);
        
        Row newRow = new Row(newBalance);
        newRow.created_by = currentTxid; // Mark with current transaction ID
        
        addRow(newRow); // Add the new row to storage
    }
}
```
x??

---

#### Indexes in Multi-Version Databases
Indexes must handle versions of objects accurately. They point to all versions of an object, and index queries filter out non-visible versions. Old versions can be garbage collected when no longer visible.

:p How do indexes work with multi-version data?
??x
Indexes in a multi-version database store pointers to all versions of each object. During read operations, the index query filters out any versions that are not visible according to the current transaction's txid. Old versions marked for deletion can be removed by garbage collection when they are no longer needed.

```java
public class Index {
    private List<Integer> rowVersions; // Pointers to all object versions
    
    public void addRow(int version) {
        this.rowVersions.add(version);
    }
    
    public List<int> getVisibleRows() {
        return this.rowVersions.stream()
                               .filter(version -> !isObjectDeleted(version))
                               .collect(Collectors.toList());
    }
}
```
x??

---

#### Append-Only B-Trees
Background context explaining how append-only B-trees work, including that they are used in systems like CouchDB, Datomic, and LMDB. These systems use an append-only/copy-on-write variant of B-trees to manage updates without overwriting existing pages.

:p How do append-only B-trees handle updates?
??x
Append-only B-trees create a new copy of each modified page instead of overwriting the old one. Parent pages up to the root are updated to point to these new versions. This ensures that any page not affected by a write remains immutable.
For example, if updating node `A` in the B-tree, a new version of `A` is created and all parent nodes (including the root) are updated to reference this new version.

```java
// Pseudocode for modifying a node in an append-only B-tree
public void updateNode(Node oldNode, Node newNode) {
    // Create a copy of the new node with the changes
    Node copiedNewNode = copy(newNode);
    
    // Update parent pointers to point to the new node
    for (ParentNode parent : parentsOf(oldNode)) {
        parent.updateChildPointer(oldNode, copiedNewNode);
    }
    
    // Replace the old node in the tree with the copied new node
    replaceNodeInParents(oldNode, copiedNewNode);
}
```
x??

---

#### Snapshot Isolation Levels
Background context on snapshot isolation levels and how they are used to manage concurrent transactions. Explain that this is particularly useful for read-only transactions but can have naming confusion due to standards differences.

:p What does snapshot isolation provide in terms of concurrency control?
??x
Snapshot isolation provides a consistent view of the database state at the start of each transaction, even when reading data concurrently written by other transactions. This means that a transaction sees a single version of the data, as it existed at some point in time (the "snapshot").

For example, if two transactions `T1` and `T2` run concurrently:
- `T1` starts and reads data from a certain snapshot.
- `T2` writes to the same data while `T1` is still running.
- When `T1` commits, it sees its original snapshot without any changes made by `T2`.

```java
// Pseudocode for managing snapshots in snapshot isolation
class Transaction {
    private Snapshot currentSnapshot;
    
    public void start() {
        // Initialize the transaction with a new snapshot
        this.currentSnapshot = new Snapshot();
    }
    
    public Object readData(Object key) {
        return currentSnapshot.get(key);
    }
}
```
x??

---

#### Lost Updates Problem
Background context on lost updates, which occur when two transactions concurrently update the same data, leading to one transaction losing its changes.

:p What is a lost update?
??x
A lost update happens when two transactions T1 and T2 try to update the same data at the same time. If both transactions commit their writes, but one overwrites the changes made by the other, then the second write effectively "loses" or loses visibility of the first write.

For example:
- Transaction `T1` reads a value from memory.
- Transaction `T2` also reads and updates that value.
- Both `T1` and `T2` commit their writes independently.
- If `T1`'s changes are lost due to `T2` committing first, the original value written by `T1` is overwritten.

```java
// Pseudocode for detecting lost updates in concurrent transactions
public class TransactionManager {
    private Map<String, Object> dataStore = new HashMap<>();
    
    public void updateData(String key, Object value) {
        if (dataStore.containsKey(key)) { // Check to avoid overwriting other transaction's writes
            // Simulate a lost update scenario
            if (ThreadLocalRandom.current().nextBoolean()) {
                System.out.println("Lost Update Detected: Value overwritten");
            } else {
                dataStore.put(key, value);
            }
        } else {
            dataStore.put(key, value);
        }
    }
}
```
x??

---

#### Conflict Resolution in Text Editing
Background context on how editing a text document can be modeled as a stream of atomic mutations. Mention that this approach helps manage conflicts between concurrently writing transactions.

:p How can editing a text document be represented using atomic mutations?
??x
Editing a text document can be represented by breaking down the changes into a series of atomic operations (mutations). Each mutation represents an indivisible change, such as inserting or deleting a character. By representing edits this way, conflicts between concurrently writing transactions can be more easily managed.

For example:
- Transaction `T1` inserts "hello" at position 5.
- Transaction `T2` deletes the word "world" starting from position 10.
These operations are atomic and can be applied in any order without causing conflicts. The system can then apply these mutations sequentially to resolve them.

```java
// Pseudocode for applying atomic mutations on text document
public class TextDocument {
    private String content = "";
    
    public void applyMutations(List<Mutation> mutations) {
        for (Mutation mutation : mutations) {
            if (mutation instanceof Insertion) {
                // Apply insertion mutation
            } else if (mutation instanceof Deletion) {
                // Apply deletion mutation
            }
            // Continue applying remaining mutations
        }
    }
    
    public static class Mutation {
        // Define specific types of mutations here
    }
}
```
x??

#### Lost Update Problem
The lost update problem occurs when two transactions concurrently read a value from the database, modify it, and write back the modified value. If this happens, one of the modifications can be lost because the second transaction overwrites the first.

:p What is the lost update problem?
??x
The lost update problem is an issue that arises in concurrent operations where multiple transactions try to read a shared resource (like a database), modify it locally, and write back the modified value. If two or more transactions do this concurrently, one of their updates can be lost because the last transaction's update overwrites the earlier ones.

For example:
- Two users incrementing a counter.
- Two users updating an account balance.
```java
// Pseudocode for Lost Update Problem Example
public class Counter {
    private int value = 0;

    public void increment() {
        int current = this.value; // Read from database
        this.value = current + 1; // Modify locally
        updateValue(current + 1); // Write back to database
    }

    private void updateValue(int newValue) {
        // Code to write the new value back to the database
    }
}
```
x??

---

#### Atomic Write Operations
Atomic write operations are designed to eliminate the need for read-modify-write cycles by providing a single, atomic operation that can be performed directly on the database. This is particularly useful in scenarios where updating a counter or an account balance is required.

:p What are atomic write operations?
??x
Atomic write operations allow transactions to update data in one step without needing to perform intermediate steps like reading and writing separately. These operations ensure that either all parts of the operation succeed, or none do, maintaining consistency.

For example:
```sql
-- SQL Example for Atomic Write Operation
UPDATE counters SET value = value + 1 WHERE key = 'foo';
```
This SQL statement is atomic because it increments the counter in a single step. No intermediate steps are involved, reducing the risk of lost updates and ensuring data integrity.
x??

---

#### Explicit Locking Mechanism
Explicit locking involves application-level mechanisms where transactions explicitly lock objects that they intend to update. This ensures that only one transaction can read or write an object at a time, thereby preventing concurrent modifications.

:p What is explicit locking?
??x
Explicit locking is a technique where the application code directly controls access to shared resources by locking them during the update process. If another transaction tries to access the same resource while it's locked, that transaction will be forced to wait until the lock is released.

For example:
```java
// Pseudocode for Explicit Locking Example
public class Counter {
    private int value = 0;
    private final Object lock = new Object();

    public synchronized void increment() {
        synchronized (lock) { // Acquire the lock
            int current = this.value; // Read from database
            this.value = current + 1; // Modify locally
            updateValue(current + 1); // Write back to database
        }
    }

    private void updateValue(int newValue) {
        // Code to write the new value back to the database
    }
}
```
In this example, the `increment` method is synchronized, which acts as a lock. This ensures that only one thread can execute the `increment` method at any given time.
x??

---

#### Atomic Operations and Locking Mechanisms
Background context: In multiplayer games or complex applications, ensuring that concurrent operations do not lead to data inconsistencies is crucial. One approach involves using atomic operations and locks to manage these scenarios.

If a single operation (atomic) cannot cover all necessary checks, locks are employed to ensure exclusive access to the data during critical sections of code. For example, in a game where multiple players can move the same figure concurrently, you might need both logical checks and locking mechanisms to prevent conflicts.

:p What is an atomic operation?
??x
An atomic operation is a single, indivisible unit of work that is performed as a whole without interruption or interference from other processes. If it fails, no part of the operation should be committed, ensuring data integrity.
```sql
BEGIN TRANSACTION;
SELECT * FROM figures   WHERE name = 'robot' AND game_id = 222   FOR UPDATE; -- Locks the rows for update

-- Check if move is valid and then update position
UPDATE figures SET position = 'c4' WHERE id = 1234;

COMMIT;
```
x??

---

#### Lost Update Detection
Background context: When implementing concurrent transactions, a lost update can occur when two or more transactions modify the same data. To prevent this, some databases automatically detect and handle lost updates.

:p How does automatic detection of lost updates work?
??x
Automatic detection of lost updates involves monitoring changes made by transactions to ensure that no transaction overwrites another's modifications. If a conflict is detected, the database aborts the conflicting transaction, requiring it to retry its operations.
For example, in PostgreSQL or SQL Server with snapshot isolation:
```sql
-- Example in PostgreSQL with Repeatable Read isolation level
BEGIN;
SELECT * FROM figures WHERE name = 'robot' AND game_id = 222 FOR UPDATE; -- Locks rows for update

-- Check if move is valid and then update position
UPDATE figures SET position = 'c4' WHERE id = 1234;

COMMIT;
```
x??

---

#### Compare-and-Set Operation
Background context: In systems without transactions, a compare-and-set operation can be used to ensure that an update only occurs if the current value matches the expected value. This avoids lost updates by comparing and setting the new value conditionally.

:p What is a compare-and-set operation?
??x
A compare-and-set (CAS) operation allows updating a data item only if its current state matches a specific expected state. If the actual state differs, the update fails, and you must retry with updated expectations.
For example:
```sql
-- Attempt to update wiki page content in PostgreSQL
UPDATE wiki_pages SET content = 'new content' WHERE id = 1234 AND content = 'old content';
```
If another transaction has modified `content` since the read, this operation will fail and require retrying with an updated expected value.
x??

---

#### Weak Isolation Levels
Background context: Different isolation levels in databases can affect how transactions handle concurrent updates. Weak isolation levels (like Read Committed) may not enforce strict serializability, which means they are more prone to issues like dirty reads or non-repeatable reads.

:p What is the difference between strong and weak isolation levels?
??x
Strong isolation levels (Serializable, Repeatable Read in PostgreSQL/Oracle, Snapshot Isolation in SQL Server) ensure that transactions operate as if they were executed serially, preventing issues like dirty reads, non-repeatable reads, or phantom reads. Weak isolation levels (Read Committed) only enforce a basic order of execution and may allow these types of inconsistencies.
For example:
```sql
-- PostgreSQL Repeatable Read Isolation Level Example
BEGIN;
SELECT * FROM figures WHERE name = 'robot' AND game_id = 222 FOR UPDATE; -- Locks rows for update

-- Check if move is valid and then update position
UPDATE figures SET position = 'c4' WHERE id = 1234;

COMMIT;
```
x??

---

