# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 8)

**Starting Chapter:** SSTables and LSM-Trees

---

#### SSTables and Sorted String Tables
Background context: In database systems, especially for storage engines like those used in Cassandra or RocksDB, handling large volumes of data efficiently is crucial. Log-structured merge-trees (LSM-Trees) are a common approach to achieve this. Each log-structured storage segment initially stores key-value pairs in the order they were written, but values later in the log take precedence over earlier ones for the same key.

The introduction of SSTables (Sorted String Tables) modifies these segments by requiring keys to be sorted within each file. This allows for more efficient merging and reading operations, while still maintaining sequential write performance.

:p What is an SSTable?
??x
An SSTable is a segment file in which the sequence of key-value pairs is sorted by key. It retains all values written during some period but ensures that only the most recent value per key is kept across segments. This format enables efficient merging, index-free seeks, and compression.
x??

---
#### Merging SSTables
Background context: When dealing with multiple SSTable files, merging them to reduce fragmentation and improve performance is necessary. The merge process is designed to be simple and efficient, leveraging the sorted nature of each file.

:p How does the merge process work for SSTables?
??x
The merge process involves reading input files side by side and writing out the lowest key (according to the sort order) to the output file repeatedly until all files are exhausted. This results in a new merged segment file that is also sorted.
```java
// Pseudocode for merging SSTables
while (!files.isEmpty()) {
    KeyMinEntry minEntry = null;
    int index = -1;

    for (int i = 0; i < files.size(); ++i) {
        if (minEntry == null || compare(files.get(i).key, minEntry.key) < 0) {
            minEntry = files.get(i);
            index = i;
        }
    }

    output.append(minEntry.key, minEntry.value);
    if (!files.get(index).nextKey()) { // If no more keys in the file
        files.remove(index); // Remove it from the list of open files
    }
}
```
x??

---
#### Efficient Search with SSTables
Background context: With SSTables, searching for a specific key does not require an index if the file is sorted. Instead, you can leverage the sorted order to perform efficient range scans and locate keys without needing in-memory indices.

:p How can one efficiently search for a key in an SSTable?
??x
To find a key in an SSTable, you don't need an in-memory index of all keys. You use the sorted nature of the file: if you know the offsets of two surrounding keys and their values, you can jump to the offset of the lower key and scan sequentially until you find the target key or exhaust the range.
```java
// Pseudocode for binary search-like approach with SSTables
int start = 0;
int end = segmentFile.length - 1;

while (start < end) {
    int mid = (start + end) / 2;
    
    if (keyAt(mid).equals(targetKey)) {
        return valueAt(mid);
    } else if (keyAt(mid).compareTo(targetKey) < 0) {
        start = mid + 1;
    } else {
        end = mid - 1;
    }
}
return null; // If key not found
```
x??

---
#### Compaction and Memtables
Background context: To manage writes efficiently, the storage engine uses a combination of in-memory structures (memtables) and on-disk sorted files (SSTables). Memtables are used for active writes, while SSTables store historical data. Periodic compaction processes merge these to optimize disk usage.

:p What is the role of memtables in an LSM-Tree?
??x
Memtables act as an in-memory storage structure that receives incoming writes. They maintain a sorted order of key-value pairs. When they reach a certain size threshold, they are flushed to disk as SSTables, becoming part of the immutable on-disk storage.
```java
// Pseudocode for memtable management
class Memtable {
    private TreeMap<Key, Value> entries;

    public void put(Key key, Value value) {
        if (entries.size() > MAX_MEMTABLE_SIZE) flushToDisk();
        entries.put(key, value);
    }

    // Method to create an SSTable from the current state of memtable
    public void flushToDisk() {
        // Convert TreeMap to a byte[] and write it as an SSTable file
    }
}
```
x??

---

#### Log-Structured Merge Trees (LSM-Trees)
Background context explaining LSM-trees. These are storage engines that merge and compact sorted files, often used in databases like RocksDB and LevelDB for efficient write throughput while supporting range queries.

:p What is an LSM-tree?
??x
An LSM-tree is a log-structured storage engine that keeps a cascade of SSTables (Sorted String Tables) which are merged in the background. This structure allows high write throughput due to sequential disk writes and supports range queries efficiently by maintaining keys in sorted order.
```java
// Pseudocode for an insert operation in an LSM-tree
public void insert(byte[] key, byte[] value) {
    // 1. Append the key-value pair to the memtable (in-memory buffer)
    memtable.append(key, value);
    
    // 2. If the memtable exceeds a certain size, flush it into an SSTable
    if (memtable.size() > MAX_MEMTABLE_SIZE) {
        writeMemTableToSSTable(memtable);
    }
}
```
x??

---

#### Bloom Filters in LSM-Trees
Background context explaining how Bloom filters are used to optimize key existence checks in LSM-trees. Bloom filters provide a space-efficient way to test if an element is a member of a set without actually storing the elements.

:p How do Bloom filters work in LSM-trees?
??x
Bloom filters are used in LSM-trees to quickly check whether a key does not exist in the database, thereby avoiding unnecessary disk reads. A Bloom filter works by mapping keys to multiple bits in a bit array and setting those bits when the key is inserted.

```java
// Pseudocode for a Bloom filter insertion
public void add(byte[] key) {
    int[] hashes = computeHashes(key);
    for (int hash : hashes) {
        bitArray[hash] = 1;
    }
}

// Pseudocode for checking if a key may exist in the database
public boolean mightContain(byte[] key) {
    int[] hashes = computeHashes(key);
    for (int hash : hashes) {
        if (bitArray[hash] == 0) {
            return false; // Key definitely not present
        }
    }
    return true; // May or may not be present, but likely present
}
```
x??

---

#### Compaction Strategies in LSM-Trees
Background context explaining different compaction strategies used to manage SSTables and maintain efficient storage. Common strategies include size-tiered and leveled compaction.

:p What are the main types of compaction strategies used in LSM-trees?
??x
There are two common compaction strategies used in LSM-trees: size-tiered and leveled compaction.
- **Size-Tiered Compaction**: Newer and smaller SSTables are successively merged into older and larger ones. This helps in maintaining a balanced tree structure but may lead to more frequent merges.
- **Leveled Compaction**: Key ranges are split into smaller SSTables, with older data moved into separate levels. This allows compaction to proceed incrementally, using less disk space.

For example, in LevelDB (which uses leveled compaction):
```java
// Pseudocode for size-tiered compaction
public void compact() {
    List<SSTable> tables = getSSTables();
    if (tables.size() > 1) { // If there are multiple SSTables
        mergeOldestTables(tables);
    }
}

// Pseudocode for leveled compaction
public void compact() {
    int level = determineLevelForCompaction(); // Determine the appropriate level based on key ranges
    moveKeysToAppropriateLevel(level, keysToMove);
}
```
x??

---

#### B-Trees in Indexing Structures
Background context explaining why B-trees are widely used and how they differ from LSM-trees. B-trees keep key-value pairs sorted by key, enabling efficient lookups and range queries.

:p What is a B-tree?
??x
A B-tree is a self-balancing tree data structure that keeps key-value pairs sorted by key, allowing for efficient key-value lookups and range queries. Unlike LSM-trees, B-trees are more commonly used in relational databases and many nonrelational databases due to their well-established performance characteristics.

:p How does a B-tree perform an insertion?
??x
Insertion in a B-tree involves the following steps:
1. Start at the root node.
2. Compare the key with the nodes' keys, moving down to the appropriate child node.
3. If the leaf node has space (i.e., it is not full), insert the new key-value pair.
4. If the leaf node is full, split it and move the median key to the parent node.

```java
// Pseudocode for B-tree insertion
public void insert(byte[] key, byte[] value) {
    Node root = getRootNode();
    
    while (true) {
        if (root.isLeaf()) {
            // Leaf node, find appropriate position and insert
            int index = binarySearch(root.keys, key);
            root.values.add(index + 1, value);
            root.keys.add(index + 1, key);
            
            // If the leaf is full, split it
            if (root.isFull()) {
                splitLeafNode(root);
            }
        } else {
            // Non-leaf node, move down to appropriate child
            int index = binarySearch(root.keys, key);
            Node child = root.children.get(index);
            insert(child, key, value);
            
            // If the child is full, it needs to be split and moved up
            if (child.isFull()) {
                splitChild(root, child, index);
            }
        }
    }
}
```
x??

---

#### B-Tree Overview
B-trees break down databases into fixed-size blocks or pages, typically 4 KB in size. Each page can be identified using an address, allowing for page references that form a tree structure. The root page is the starting point for looking up keys.
:p What are the key features of B-tree design and how do they relate to hardware?
??x
B-trees use fixed-size pages, which correspond to disk blocks. Each page contains multiple keys and pointers to child pages. This structure forms a tree where each node (page) points to its children based on key ranges.
```java
public class BTreeNode {
    List<KeyPage> keys; // Keys in the node
    List<BTreeNode> children; // Child nodes corresponding to key ranges
}
```
x??

---

#### Branching Factor of B-Tree
The branching factor is the number of references to child pages within a single page. It determines how many sub-ranges each parent can split into.
:p What does the term "branching factor" mean in the context of B-trees?
??x
The branching factor indicates the maximum number of direct children (or pointers) that a node can have in a B-tree. A higher branching factor means more keys per page and fewer levels in the tree, which can improve performance.
```java
// Example calculation for branching factor
int branchingFactor = totalKeysPerPage / 2; // Assuming each key has two child references
```
x??

---

#### Insertion into B-Tree
Inserting a new key involves finding the appropriate leaf page and adding it there. If necessary, pages are split to accommodate more keys.
:p How is a new key inserted in a B-tree?
??x
To insert a new key, first find the correct leaf page where the key belongs based on its value. If the leaf page has enough space, add the key. Otherwise, split the page into two and update the parent node to account for the new key range.
```java
public void insertKey(int key) {
    BTreeNode leaf = findLeafNode(key); // Find appropriate leaf node
    if (leaf.hasSpaceForKey(key)) {
        leaf.addKey(key); // Add key directly
    } else {
        splitPage(leaf, key); // Split the page and update parent node
    }
}
```
x??

---

#### Deletion from B-Tree
Deleting a key involves adjusting the tree to maintain balance. If a page becomes too small after deletion, it may merge with another.
:p How does deleting a key in a B-tree work?
??x
Deletion is complex because it needs to keep the tree balanced. When removing a key, if the page containing the key has enough space, simply remove it. Otherwise, perform merges or splits as needed to maintain balance and ensure every non-leaf node still satisfies the B-tree properties.
```java
public void deleteKey(int key) {
    BTreeNode leaf = findLeafNode(key); // Find appropriate leaf node
    if (leaf.deleteKey(key)) { // If key is found, remove it
        // Rebalance tree if necessary
    }
}
```
x??

---

#### Key Lookup in B-Tree
Starting from the root page, keys and references guide you through the tree to find the desired value or location.
:p How does a key lookup work in a B-tree?
??x
Key lookup starts at the root node. The node contains keys that help determine which child node to traverse next based on the target key's value. This process continues until reaching a leaf node, where the exact location of the data can be found or inferred.
```java
public Value findValue(int key) {
    BTreeNode current = rootNode; // Start from root
    while (current != null && !current.isLeaf()) { // Traverse to appropriate child
        current = current.getChildNodeForKey(key);
    }
    if (current.isLeaf()) { // At a leaf node, retrieve value
        return current.getValueForKey(key);
    } else {
        throw new Error("Key not found"); // If not at a leaf
    }
}
```
x??

---

#### B-Tree Depth and Balance
B-trees ensure that the tree remains balanced with a depth of O(log n), where n is the number of keys. This maintains efficient search times.
:p What ensures that a B-tree remains balanced?
??x
A B-tree maintains balance through its structure, ensuring that each non-leaf node has between 2 and the branching factor (b) children, and every leaf is at the same depth. When operations like insertions or deletions cause imbalance, specific algorithms are applied to rebalance the tree.
```java
public void ensureBalance(BTreeNode node) {
    // Rebalance logic here - ensures each non-leaf has 2 to b children
}
```
x??

#### Overwriting Pages on Disk
Background context: In database systems, overwriting a page is a hardware operation where data is physically written to the disk. This process can vary significantly between different storage technologies such as magnetic hard drives and SSDs.

On a **magnetic hard drive**, overwriting involves moving the disk head to the correct track and waiting for the sector containing the old data to spin into position, then writing new data to that sector.
```java
// Pseudocode for hard disk operation
public void overwritePage(long pageAddress, byte[] newData) {
    // Move head to the correct track
    moveHeadTo(pageAddress);
    
    // Wait for the appropriate sector to pass under the read/write head
    while (currentSector != pageAddress % SECTOR_SIZE) {
        spinPlatter();
    }
    
    // Write new data to the disk
    writeData(newData);
}
```

On **SSDs**, overwriting is more complex due to block-level writes. An SSD must erase and rewrite an entire block, which typically contains multiple pages.
:p How does overwriting work on a magnetic hard drive?
??x
Overwriting on a hard drive involves positioning the disk head over the correct track, waiting for the sector containing the old data to align with the read/write head, and then writing new data to that sector. The process is synchronous and requires mechanical movement of the head.
x??

---
#### Overwriting Pages in B-Trees
Background context: In B-trees used in databases, overwriting a page can lead to inconsistencies if not managed properly due to potential crashes after partial writes.

When a **page split** occurs because an insertion causes a page to overflow, both pages need to be overwritten, along with their parent page to update references.
```java
// Pseudocode for page split and overwrite in B-tree
public void insertIntoBTree(Node root, byte[] newData) {
    Node newPage = splitPage(root);
    
    // Overwrite the two child pages and their parent reference
    overwritePages(newPage.getData(), newChildPage1.getData(), newChildPage2.getData());
}
```

The danger lies in partial writes; if a crash occurs after only some of the pages are written, it can result in an inconsistent state (e.g., orphaned pages).
:p What happens when a page split occurs during an insertion?
??x
When a page split occurs due to an insertion, both child pages and their parent reference need to be overwritten. This ensures that all pointers remain consistent after the operation. If only some of these writes are completed before a crash, the tree can become inconsistent.
x??

---
#### Write-Ahead Log (WAL)
Background context: A **write-ahead log** is used in B-tree implementations for crash recovery by logging every modification to an append-only file.

When the database restarts after a crash, it can use this log to restore the B-tree to a consistent state. This approach ensures that no data is lost and that all changes are durable.
```java
// Pseudocode for using WAL in B-tree operations
public void insertIntoBTreeWithWAL(Node root, byte[] newData) {
    // Log the operation to the write-ahead log before applying it
    logOperation(newData);
    
    Node newPage = splitPage(root);
    
    overwritePages(newPage.getData(), newChildPage1.getData(), newChildPage2.getData());
}
```
:p How does a write-ahead log help in B-tree operations?
??x
A write-ahead log helps by logging every modification before it is applied to the actual tree pages. This ensures that if a crash occurs, the database can recover by replaying the log and restoring the tree to a consistent state. It prevents data loss and ensures durability.
x??

---
#### Concurrency Control in B-Trees
Background context: Managing multiple threads accessing a B-tree simultaneously requires careful concurrency control to avoid inconsistent states.

Concurrency is typically handled using **latches** (lightweight locks) that protect the tree’s data structures, ensuring that no thread sees an inconsistent state of the tree.
```java
// Pseudocode for concurrency control in B-tree
public void insertIntoBTree(Node root, byte[] newData) {
    // Acquire a latch on the node to ensure exclusive access
    acquireLatch(root);
    
    try {
        Node newPage = splitPage(root);
        
        overwritePages(newPage.getData(), newChildPage1.getData(), newChildPage2.getData());
    } finally {
        // Release the latch once operations are complete
        releaseLatch(root);
    }
}
```
:p How is concurrency control achieved in B-tree implementations?
??x
Concurrency control in B-trees is managed using latches (lightweight locks) that ensure exclusive access to tree nodes. This prevents multiple threads from accessing and modifying the same node simultaneously, which could result in inconsistent states.
x??

---
#### Copy-on-Write Scheme
Background context: Some databases use a **copy-on-write scheme** for crash recovery, where modifications are written to different locations rather than overwriting existing pages.

This approach creates new versions of affected pages and updates parent pointers accordingly. It is useful for concurrent access since it does not interfere with incoming queries.
```java
// Pseudocode for copy-on-write in LMDB
public void insertIntoLMDB(byte[] newData) {
    // Allocate a new page for the data
    byte[] newPage = allocateNewPage();
    
    // Copy the data to the new location
    System.arraycopy(newData, 0, newPage, 0, newData.length);
    
    // Update parent pointers in existing pages to point to the new page
    updateParentPointers(newPage);
}
```
:p What is a copy-on-write scheme?
??x
A copy-on-write scheme involves writing modifications to different locations rather than overwriting existing pages. This method creates new versions of affected pages and updates parent pointers, ensuring consistency even when multiple threads are accessing the B-tree concurrently.
x??

---

#### B+ Tree Variant
Background context: This variant of B-tree is sometimes referred to as a B+ tree, although it's so common that it might not be distinguished from other B-tree variants. It often involves optimizations like abbreviated keys and additional pointers for efficient scanning.

:p What are the key characteristics of this B+ tree variant?
??x
This B+ tree variant optimizes storage by using abbreviated keys on interior nodes, which only need to provide boundary information between key ranges. Additionally, it includes extra pointers in leaf pages to facilitate sequential scans without jumping back to parent pages. These optimizations help maintain a higher branching factor and minimize the number of levels needed.

```java
// Pseudocode for creating a B+ tree node with abbreviated keys and pointers
public class BPlusTreeNode {
    private Key[] keys;
    private PageReference[] children; // Pointers to child nodes
    private PageReference siblingLeft, siblingRight; // Pointers to siblings

    public void addKey(Key key) {
        // Logic to insert the key in sorted order while maintaining boundary information
    }
}
```
x??

---

#### Sequential Layout of Leaf Pages
Background context: Many B-tree implementations try to arrange leaf pages sequentially on disk to improve read performance by reducing disk seeks. However, maintaining this sequential layout as the tree grows can be challenging.

:p How does maintaining a sequential layout on disk benefit B-trees?
??x
Maintaining a sequential layout on disk helps in minimizing disk seeks during range queries, which are common in many applications. By keeping leaf pages contiguous and in order, the database can read keys sequentially without needing to jump between non-adjacent locations on disk.

```java
// Pseudocode for maintaining sequential layout of B+ tree leaf nodes
public class BPlusTree {
    private Page[] leafPages; // Array of leaf node references

    public void insert(Key key) {
        int position = findInsertPosition(key);
        if (leafPages[position] == null) {
            createNewLeafNode(position, key);
        } else {
            addKeyToExistingLeaf(leafPages[position], key);
            if (isOverflowed(leafPages[position])) {
                splitLeafNode(leafPages[position]);
            }
        }
    }

    private int findInsertPosition(Key key) {
        // Logic to determine the correct position for inserting a new key
        return Arrays.binarySearch(leafPages, key.getKeyValue());
    }
}
```
x??

---

#### Write Amplification in LSM-Trees
Background context: Write amplification occurs when a single write operation results in multiple writes to disk over time. This is common in log-structured storage systems due to compaction and merging of SSTables.

:p What is write amplification, and why is it problematic for LSM-trees?
??x
Write amplification refers to the situation where a single write operation leads to multiple writes to disk over the lifetime of the database system. This occurs because LSM-trees rewrite data during compaction and merging processes, leading to inefficiencies and potential performance bottlenecks.

```java
// Example logic for handling write amplification in an LSM-tree
public class LSMTree {
    private List<SSTable> sstables;

    public void write(Key key, Value value) {
        // Write the new data to the current SSTable
        if (isSSTableFull()) {
            mergeAndCompactSSTables();
        }
        addDataToCurrentSSTable(key, value);
    }

    private void mergeAndCompactSSTables() {
        // Logic for merging and compacting multiple SSTables into fewer, larger files
        // This process can lead to write amplification as old data is rewritten
    }
}
```
x??

---

#### Comparison of B-Trees and LSM-Trees
Background context: While traditional B-trees are well-established in database systems, LSM-trees offer advantages in certain scenarios due to their performance characteristics.

:p What are the general performance differences between B-trees and LSM-trees?
??x
B-trees are typically faster for reads as they provide direct access to key-value pairs through a balanced tree structure. However, LSM-trees excel in write-heavy applications because they minimize the number of disk writes by rewriting large segments during compaction.

```java
// Pseudocode comparing read and write performance between B-tree and LSM-tree
public class StorageEngine {
    private BTree bTree;
    private LSMTree lsmTree;

    public void comparePerformance() {
        // Benchmark read operations on both storage engines
        long readTimeBTree = benchmarkRead(bTree);
        long readTimeLSMTree = benchmarkRead(lsmTree);

        // Benchmark write operations
        long writeTimeBTree = benchmarkWrite(bTree);
        long writeTimeLSMTree = benchmarkWrite(lsmTree);

        // Output the results for comparison
        System.out.println("Read Time (B-Tree): " + readTimeBTree);
        System.out.println("Read Time (LSM-Tree): " + readTimeLSMTree);
        System.out.println("Write Time (B-Tree): " + writeTimeBTree);
        System.out.println("Write Time (LSM-Tree): " + writeTimeLSMTree);
    }

    private long benchmarkRead(StorageEngine engine) {
        // Logic to measure read performance
        return 0; // Placeholder for actual measurement
    }

    private long benchmarkWrite(StorageEngine engine) {
        // Logic to measure write performance
        return 0; // Placeholder for actual measurement
    }
}
```
x??

---

#### Advantages of LSM-Trees
Background context: LSM-trees offer several advantages over traditional B-tree storage engines, particularly in terms of write performance and reduced fragmentation.

:p What are some key advantages of using LSM-trees?
??x
LSM-trees have lower write amplification because they rewrite data during compaction rather than writing to the tree directly. This reduces the number of writes required per piece of data. Additionally, LSM-trees can sustain higher write throughput due to sequential writes and reduced fragmentation.

```java
// Example logic for handling sequential writes in LSM-trees
public class LSMTree {
    private List<SSTable> sstables;

    public void write(Key key, Value value) {
        // Write the new data to a current SSTable
        if (isSSTableFull()) {
            mergeAndCompactSSTables();
        }
        addDataToCurrentSSTable(key, value);
    }

    private void mergeAndCompactSSTables() {
        // Logic for merging and compacting multiple SSTables into fewer, larger files
        // This helps in reducing fragmentation and write amplification
    }
}
```
x??

---

#### Downsides of LSM-Trees
Background context: While LSM-trees offer advantages, they also have downsides such as potential interference with concurrent reads and writes during compaction.

:p What are some disadvantages of using LSM-trees?
??x
One major downside of LSM-trees is that the compaction process can interfere with ongoing reads and writes. Despite storage engines attempting to perform compaction incrementally, the process can still impact performance. Additionally, while write amplification is reduced, the overhead of managing multiple SSTables and their merging can introduce complexity.

```java
// Example logic for handling concurrent operations during compaction in LSM-trees
public class LSMTree {
    private List<SSTable> sstables;

    public void write(Key key, Value value) {
        // Write the new data to a current SSTable or initiate compaction if necessary
        if (isSSTableFull()) {
            mergeAndCompactSSTables();
        }
        addDataToCurrentSSTable(key, value);
    }

    private void mergeAndCompactSSTables() {
        // Logic for merging and compacting multiple SSTables into fewer, larger files
        // This can impact concurrent read/write operations due to increased overhead
    }
}
```
x??

---

