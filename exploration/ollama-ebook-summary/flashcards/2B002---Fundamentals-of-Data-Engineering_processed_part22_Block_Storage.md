# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 22)

**Starting Chapter:** Block Storage

---

#### Block Storage Overview
Block storage provides fine control over storage size, scalability, and data durability beyond raw disks. Blocks are the smallest addressable units of data on a disk, often 4,096 bytes for current disks. Each block can contain metadata for error detection/correction.
:p What is block storage?
??x
Block storage refers to the type of raw storage provided by SSDs and magnetic disks in the cloud. It allows fine control over storage size, scalability, and data durability beyond what is offered by raw disks. The smallest addressable unit on a disk is called a block, typically 4,096 bytes.
x??

---

#### Transactional Database Systems
Transactional database systems access disks at a block level to optimize performance. For row-oriented databases, rows of data are written as continuous streams. SSDs have improved seek-time performance but still rely heavily on high random access performance provided by direct block storage devices.
:p How do transactional database systems typically handle disk access?
??x
Transactional database systems use block-level access for optimal performance. They write rows of data as continuous streams, which was originally the case. With SSDs, although seek time has improved, they still rely on high random access performance to ensure efficient operations when accessing individual blocks.
x??

---

#### RAID Overview
RAID stands for Redundant Array of Independent Disks and is used to improve data durability, enhance performance, and combine capacity from multiple drives. An array can be presented as a single block device with various encoding and parity schemes depending on the desired balance between performance and fault tolerance.
:p What does RAID stand for?
??x
RAID stands for Redundant Array of Independent Disks. It is used to improve data durability, enhance performance, and combine capacity from multiple drives by controlling these disks together.
x??

---

#### Storage Area Network (SAN)
Storage area networks provide virtualized block storage devices over a network, typically from a centralized storage pool. This allows fine-grained storage scaling and enhances performance, availability, and durability. SAN can be on-premises or cloud-based.
:p What is a Storage Area Network (SAN)?
??x
A Storage Area Network (SAN) provides virtualized block storage devices over a network, usually from a centralized storage pool. It enables fine-grained storage scaling, improved performance, enhanced availability, and durability compared to traditional storage solutions.
x??

---

#### Cloud Virtualized Block Storage - Amazon EBS Example
Amazon Elastic Block Store (EBS) is an example of cloud virtualized block storage that abstracts physical disks behind a virtual layer, allowing for finer control over storage. It offers different performance tiers with IOPS and throughput metrics. SSD-backed storage provides higher performance but costs more per gigabyte than magnetic disk-backed storage.
:p What is Amazon Elastic Block Store (EBS)?
??x
Amazon Elastic Block Store (EBS) is cloud virtualized block storage that abstracts physical disks behind a virtual layer, providing finer control over storage with different performance tiers. It offers IOPS and throughput metrics; SSD-backed storage provides higher performance but at a higher cost per gigabyte compared to magnetic disk-backed storage.
x??

---

#### Local Instance Volumes
Local instance volumes are physically attached to the host server running a VM and offer low latency and high IOPS, making them very cost-effective. However, they do not support advanced virtualization features like EBS such as snapshots or replication. The contents of these disks are lost when the VM is deleted.
:p What are local instance volumes?
??x
Local instance volumes are physically attached to the host server running a VM and offer low latency and high IOPS at very low cost. They do not support advanced virtualization features like EBS such as snapshots or replication, but their contents are lost when the VM is deleted.
x??

---

#### Local Disk Failure and Resilience Strategies

Background context: The text discusses the use of instance store volumes, which are locally attached storage options. These can be cost-effective but come with risks such as local disk failure, accidental VM shutdowns, or cloud outages.

:p What are the potential consequences of a local disk failure in an ephemeral Hadoop cluster?
??x
In the event of a local disk failure on instance store volumes, data stored on these disks will be lost. This can disrupt operations and potentially cause data loss if no backup mechanisms are in place. Accidental VM or cluster shutdowns or zonal/ regional cloud outages could also lead to similar issues.

To mitigate risks:
- Periodic checkpoint backups to S3 can prevent data loss.
- Use of resilient distributed datasets (RDDs) in Hadoop ensures that the data is replicated across multiple nodes, reducing the risk of local disk failure impacting operations.

```java
// Example code for periodic checkpointing using Apache Spark
public class CheckpointManager {
    public void saveCheckpoints() {
        // Assuming sparkContext and rdd are already initialized
        rdd.checkpoint(); // Saves an RDD in a resilient manner

        // Schedule periodic checkpoints
        long interval = 30 * 60 * 1000; // Every 30 minutes
        new Timer().schedule(new CheckpointTask(sparkContext), interval, interval);
    }

    class CheckpointTask extends TimerTask {
        private SparkContext sparkContext;

        public CheckpointTask(SparkContext sc) {
            this.sparkContext = sc;
        }

        @Override
        public void run() {
            // Perform checkpointing logic here
            for (RDD<?> rdd : sparkContext.getAllRDDs()) {
                rdd.checkpoint();
            }
        }
    }
}
```
x??

---

#### Object Storage Overview

Background context: The text introduces object storage as a key component in data storage systems, particularly relevant with the rise of big data and cloud computing. Examples include Amazon S3, Azure Blob Storage, and Google Cloud Storage (GCS).

:p What is object storage, and how does it differ from traditional file storage?
??x
Object storage is a specialized form of storage designed for storing immutable objects, which can be any type of data such as TXT files, CSVs, JSON, images, videos, or audio. Unlike local disk-based file systems where files can be modified in place, objects are stored once and cannot be changed directly; if changes need to be made, a new object must be created.

Key differences from traditional file storage:
- **Immutability**: Once written, objects cannot be altered.
- **No Random Writes or Appends**: All writes require rewriting the entire object.
- **Parallelism and Scalability**: Object stores support high-speed parallel stream reads and writes across many disks.

```java
// Pseudocode for writing an object to S3 using AWS SDK
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.model.PutObjectRequest;

public class ObjectUploader {
    private AmazonS3 s3Client;

    public ObjectUploader(AmazonS3 client) {
        this.s3Client = client;
    }

    public void uploadFile(String bucketName, String key, File file) throws IOException {
        PutObjectRequest request = new PutObjectRequest(bucketName, key, file);
        s3Client.putObject(request);
    }
}
```
x??

---

#### Characteristics of Object Stores

Background context: The text highlights the characteristics and use cases of object storage, emphasizing its role in big data and cloud environments.

:p What are the primary characteristics of an object store?
??x
Primary characteristics of an object store include:
- **Immutable Data**: Once written, objects cannot be modified; to change or append data, a new object must be created.
- **Parallelism**: Supports high-speed parallel stream reads and writes across multiple disks.
- **Scalability**: Read bandwidth can scale with the number of parallel requests, virtual machines employed for reading, and CPU cores.

These characteristics make object stores well-suited for scenarios where large volumes of data need to be stored and accessed efficiently in a scalable manner.

```java
// Pseudocode for reading an object from S3 using AWS SDK
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.model.GetObjectRequest;

public class ObjectDownloader {
    private AmazonS3 s3Client;

    public ObjectDownloader(AmazonS3 client) {
        this.s3Client = client;
    }

    public void downloadFile(String bucketName, String key, File file) throws IOException {
        GetObjectRequest request = new GetObjectRequest(bucketName, key);
        s3Client.getObject(request, new FileOutputStream(file));
    }
}
```
x??

---

#### Object Storage Ideal for High-Volume Web Traffic and Big Data Processing
Cloud object storage is designed to handle high-volume web traffic and support big data processing by providing scalable, durable, and highly available storage. These characteristics enable efficient data management and processing, especially when combined with ephemeral compute clusters.
:p What are the key benefits of using cloud object storage for web traffic and big data processing?
??x
The key benefits include:
- High durability: Data is stored across multiple availability zones, reducing the risk of loss.
- Scalability: Storage space can be increased as needed without physical limitations.
- Availability: Ensures that data remains accessible even during hardware failures or maintenance.
These features are particularly useful for organizations that need to process large volumes of data on-demand and scale their resources flexibly.

```java
// Example of creating a storage object in Java using a hypothetical ObjectStorage library
public class CloudStorageExample {
    private ObjectStorageClient client;

    public void initializeStorage() {
        client = new ObjectStorageClient("your-credentials");
        // Initialize with availability zones for high redundancy
        client.setAvailabilityZones(new String[]{"zone1", "zone2"});
    }

    public void uploadData(String filePath) {
        client.uploadFile(filePath, "bucket-name", "object-key");
    }
}
```
x??

---

#### Cloud Object Storage and Compute Separation
Cloud object storage allows for the separation of compute and storage, enabling engineers to process data using ephemeral clusters that can be scaled up or down as needed. This approach is ideal for organizations that require flexible and cost-effective big data processing capabilities.
:p How does cloud object storage support the separation of compute and storage?
??x
Cloud object storage supports the separation by:
- Providing durable, scalable, and highly available storage that can be accessed independently from compute resources.
- Allowing the use of ephemeral clusters for processing data, which can be scaled up or down based on demand.
This setup is particularly beneficial for organizations that want to process large datasets without investing in permanent hardware.

```java
// Example of processing data with an ephemeral cluster using a hypothetical EphemeralCluster library
public class DataProcessingExample {
    private EphemeralClusterManager manager;

    public void processData() {
        // Create and initialize the ephemeral cluster
        manager.createCluster("cluster-name", "storage-uri");
        
        // Upload data to object storage
        ObjectStorageClient client = new ObjectStorageClient("your-credentials");
        client.uploadFile("/path/to/data.csv", "data-bucket", "input-data");

        // Execute a data processing job on the ephemeral cluster
        manager.runJob("cluster-name", "/path/to/processing-script.py", "input-data");
    }
}
```
x??

---

#### Storage Classes and Tiers in Cloud Object Storage
Cloud storage vendors offer different classes of storage at varying costs, balancing durability and availability. These tiers are useful for organizations that need to optimize their storage costs while maintaining necessary levels of data protection.
:p What is the purpose of offering different storage classes in cloud object storage?
??x
The purpose of offering different storage classes is to:
- Provide flexibility in choosing between high durability and lower cost options based on specific needs.
- Allow organizations to balance storage performance, availability, and cost according to their priorities.

For example, a class with higher durability might be used for critical data, while a more cost-effective class could be used for non-critical or archived data.

```java
// Example of selecting a storage class in Java using a hypothetical StorageManager library
public class StorageClassSelectionExample {
    private StorageManager manager;

    public void selectStorageClass() {
        // Define the required level of durability and availability
        String className = "high-durability-class";
        
        // Initialize with the selected storage class
        manager.initialize("storage-uri", className);
        
        // Upload data to object storage using the selected class
        ObjectStorageClient client = new ObjectStorageClient(manager.getCredentials());
        client.uploadFile("/path/to/data.csv", "data-bucket", "input-data");
    }
}
```
x??

---

#### Performance of Object Stores for Data Engineering
Object stores provide excellent performance for large batch reads and writes, making them suitable for massive OLAP systems. However, they are not ideal for frequent small updates due to their inherent design.
:p How do object stores perform in data engineering applications?
??x
Object stores excel in:
- Large batch operations: High throughput for reading and writing large amounts of data, which is common in big data processing.
- Data access patterns: Well-suited for scenarios where data is updated infrequently or when updates involve a large volume of data.

However, they are not optimized for:
- Frequent small updates: Due to the nature of object storage, frequent small updates can lead to performance degradation and higher costs.

```java
// Example of batch processing with an object store in Java using a hypothetical BatchProcessor library
public class ObjectStoreBatchProcessingExample {
    private BatchProcessor processor;

    public void processBatchData() {
        // Initialize the batch processor with access credentials
        processor = new BatchProcessor("your-credentials");

        // Upload data to object storage
        ObjectStorageClient client = new ObjectStorageClient(processor.getCredentials());
        client.uploadFile("/path/to/batch-data.csv", "data-bucket", "batch-input");

        // Perform a batch processing job on the uploaded data
        processor.runBatchJob("cluster-name", "/path/to/processing-script.py", "batch-input");
    }
}
```
x??

---

#### Object Stores in Data Lakes
Object stores have become the standard storage for data lakes, supporting large-scale data ingestion and flexible querying. They are particularly effective due to their ability to handle unstructured data without constraints.
:p Why are object stores suitable for data lakes?
??x
Object stores are suitable for data lakes because:
- They can store any type of binary data (unstructured) efficiently.
- They support write-once, read-many (WORM) operations well.
- They provide high scalability and durability, which aligns with the needs of data lakes.

In recent years, tools like Apache Hudi and Delta Lake have enhanced their capabilities to manage updates more effectively.

```java
// Example of using an object store for a data lake in Java using a hypothetical DataLakeManager library
public class DataLakeExample {
    private DataLakeManager manager;

    public void setupDataLake() {
        // Initialize the data lake with necessary configurations
        manager = new DataLakeManager("storage-uri", "data-bucket");
        
        // Upload raw data to the object store
        ObjectStorageClient client = new ObjectStorageClient(manager.getCredentials());
        client.uploadFile("/path/to/raw-data.csv", "raw-data-bucket", "input-data");
    }
}
```
x??

---

#### Object Stores vs. File Stores
Background context: The text discusses how object stores differ from file stores, particularly focusing on their lack of a directory tree structure for managing objects.

:p What is a key difference between object stores and file stores when it comes to storing data?
??x
Object stores do not use a directory tree to find objects. Instead, they utilize top-level logical containers (buckets) and reference objects by unique keys within those buckets. This contrasts with traditional file stores where directories and files are organized in a hierarchical manner.

For example:
- In S3: `S3://oreilly-data-engineering-book/data-example.json` uses the bucket name `oreilly-data-engineering-book` and key `data-example.json`.
- The full path in object stores can mimic directory semantics, but it is not a true hierarchy. For instance, storing an object like `S3://oreilly-data-engineering-book/project-data/11/23/2021/data.txt`.

:p What operations are costly when using object stores for "directory"-level tasks?
??x
Operations that mimic directory-level actions in object stores can be expensive. For example, running a command like `aws ls S3://oreilly-data-engineering-book/project-data/11/` requires the object store to filter keys based on the key prefix (`project-data/11/`). If the bucket contains millions of objects, even if only a few are relevant under that directory path, this operation can take considerable time.

:p What is eventual consistency in object stores?
??x
Eventual consistency means that after a write operation, there might be a period during which reads could return stale data. The system guarantees that eventually, all reads will reflect the latest written state. This differs from strong consistency where every read immediately reflects the most recent write.

:p How can one achieve strong consistency in object stores?
??x
One approach to achieving strong consistency is by using a strongly consistent database alongside the object store. For example:

1. Write the object.
2. Write the returned metadata for the object version to the strongly consistent database (e.g., PostgreSQL).

Here's an illustrative pseudocode:
```pseudocode
function updateObject(objectStore, objectKey, newData) {
    // Step 1: Write the new data as a new version of the object
    let newObj = writeDataToS3(objectStore, objectKey, newData);
    
    // Step 2: Update metadata in PostgreSQL to mark this as the latest version
    updateMetadataInPostgres(newObj.versionId);
}
```
This ensures that any read operation will eventually see the most recent data written.

---
Note: This set of flashcards aims to cover key concepts from the provided text, focusing on differences between object stores and file stores, costs associated with directory-like operations, and methods for achieving strong consistency in object storage.

---
#### Object Versioning and Metadata Handling
Object versioning involves maintaining multiple versions of an object stored under a single key. The metadata (hash or timestamp) uniquely identifies each version, ensuring uniqueness when combined with the object key.

:p What is the process for fetching and reading an object in a versioned storage system?
??x
The process involves several steps to ensure that the latest and correct version of the object is retrieved:

1. Fetch the latest metadata from the strongly consistent database.
2. Query the object metadata using the object key. If the data matches the fetched metadata, read the object data.
3. If there's a mismatch in metadata, repeat step 2 until the latest version of the object is returned.

This method ensures strong consistency but can lead to increased latency due to potential staleness during rewrites.

```java
public class ObjectReader {
    public void fetchLatestObject(String key) {
        // Step 1: Fetch the latest metadata
        String latestMetadata = fetchLatestMetadataFromDatabase(key);

        while (true) {
            // Step 2: Query object metadata and read data if it matches
            String objectMetadata = queryObjectMetadata(key);
            byte[] objectData = readObjectData(key);

            if (objectMetadata.equals(latestMetadata)) {
                return objectData;
            } else {
                // Step 3: If mismatch, repeat step 2 until latest version is returned
                Thread.sleep(100); // Simulate retry mechanism with a delay
            }
        }
    }

    private String fetchLatestMetadataFromDatabase(String key) {
        // Implementation to fetch the latest metadata from database
        return "latest metadata";
    }

    private String queryObjectMetadata(String key) {
        // Implementation to query object metadata using key
        return "current metadata";
    }

    private byte[] readObjectData(String key) {
        // Implementation to read object data using key
        return new byte[1024];
    }
}
```
x??

---
#### Object Versioning and Garbage Collection
With versioning, the system retains multiple versions of an object under a single key. The garbage collector waits until all references are dereferenced before cleaning up old versions, thus ensuring that historical data remains available.

:p What happens to old object versions after they are no longer referenced?
??x
Old object versions remain in storage as long as there are still references pointing to them. The storage system's garbage collector only deallocates space when all references (pointers) to an old version have been removed, which can take time depending on the frequency of updates and data retention policies.

```java
public class GarbageCollector {
    public void cleanUpDereferencedData(String key, String versionId) {
        // Implementation to check if all references are dereferenced
        boolean allReferencesRemoved = checkAllReferencesRemoved(key, versionId);

        if (allReferencesRemoved) {
            // Deallocate space for the old version
            deallocateSpaceForKeyVersion(key, versionId);
        }
    }

    private boolean checkAllReferencesRemoved(String key, String versionId) {
        // Simulate checking all references
        return true; // Assume all references are removed for simplicity
    }

    private void deallocateSpaceForKeyVersion(String key, String versionId) {
        // Implementation to deallocate space and recycle disk capacity
        System.out.println("Deallocated space for " + key + " - " + versionId);
    }
}
```
x??

---
#### Object Versioning and Consistency Management
Using object versioning, clients can request specific versions of objects by providing both the object key and its version metadata. This method ensures that the same immutable data is always returned when using the specified pair.

:p How does object versioning handle consistency issues in object storage?
??x
Object versioning handles consistency issues by combining the object key with a unique version identifier (hash or timestamp). When a client requests an object, they specify both the key and the desired version. This ensures that the system always returns the same immutable data for this pair, even if there are multiple versions available.

```java
public class ConsistencyManager {
    public byte[] getObjectByVersion(String key, String version) throws VersionNotFoundException {
        // Fetch metadata to check if the requested version exists
        Metadata metadata = fetchMetadataForKeyVersion(key, version);

        if (metadata == null) {
            throw new VersionNotFoundException("Version " + version + " not found for key: " + key);
        }

        return readObjectData(metadata.getObjectPath());
    }

    private Metadata fetchMetadataForKeyVersion(String key, String version) {
        // Simulate fetching metadata
        if ("v1".equals(version)) {
            return new Metadata("/path/to/object/v1", 123456789L);
        } else {
            return null;
        }
    }

    private byte[] readObjectData(String path) {
        // Read and return the data from the specified path
        return "object data".getBytes();
    }
}
```
x??

---
#### Object Versioning and Storage Costs
While object versioning ensures strong consistency, it can incur significant storage costs due to the need to retain multiple versions of an object. The cost depends on factors such as data size and update frequency.

:p What are the potential storage costs associated with object versioning?
??x
The storage costs for object versioning depend significantly on the amount of data stored in each version and how often objects are updated. Each version is typically stored in full, rather than as incremental snapshots, which means that retaining multiple versions can greatly increase storage requirements.

```java
public class StorageCostCalculator {
    public long calculateStorageCost(long dataSize, int numberOfVersions) {
        // Assuming each object version requires the same amount of space
        return dataSize * numberOfVersions;
    }

    public void manageStorageCosts(String key, String version) throws VersionNotFoundException {
        // Calculate and log storage costs for requested version
        long cost = calculateStorageCost(1024L, 5); // Example: 1KB per object with 5 versions

        if (cost > 1024 * 1024 * 8) { // Threshold for high costs
            System.out.println("Warning: High storage cost of " + cost + " bytes for key: " + key);
        }
    }
}
```
x??

---

#### Cloud Storage Tiers and Lifecycle Policies
Cloud vendors provide different storage tiers to optimize costs based on access frequency. Each tier has its own trade-offs between storage cost and retrieval cost. Lifecycle policies can automatically move data to appropriate storage classes based on predefined rules, reducing management overhead.

:p What are cloud storage tiers?
??x
Cloud storage tiers refer to different levels of storage services offered by cloud vendors at varying costs based on access frequency and durability requirements. These tiers allow users to choose the most cost-effective option for their specific needs.
x??

---
#### S3 Standard-Infrequent Access (SIA)
This tier offers lower monthly storage costs compared to standard storage but comes with higher retrieval costs when data is accessed.

:p What does S3 Standard-Infrequent Access provide?
??x
S3 Standard-Infrequent Access provides a reduced-cost storage option for data that is infrequently accessed. It maintains high availability and durability, similar to the standard tier, but charges more for retrieving objects from this tier.
x??

---
#### Amazon S3 One Zone-Infrequent Access (One Zone-IA)
This tier further reduces costs by replicating data within a single AWS region, reducing availability slightly due to regional failure risks.

:p What is unique about Amazon S3 One Zone-Infrequent Access?
??x
Amazon S3 One Zone-Infrequent Access offers the lowest storage cost among the Infrequent Access tiers but sacrifices some availability. It replicates data in only one AWS zone, leading to a 0.5% decrease in projected availability compared to the standard tier.
x??

---
#### Amazon S3 Glacier and Its Tiers
S3 Glacier provides extremely low-cost long-term storage with high retrieval costs for accessing stored data. The Deep Archive tier offers even lower storage rates but requires longer data restoration times.

:p What is Amazon S3 Glacier?
??x
Amazon S3 Glacier is a highly durable, low-cost object storage service designed for secure, long-term archival storage and backup of data that does not require frequent access. It includes various tiers like Glacier and Glacier Deep Archive, which offer different trade-offs between storage cost and retrieval speed.
x??

---
#### S3 Glacier Deep Archive
This tier provides the lowest storage costs but requires 12-hour or longer wait times for data restoration.

:p What is Amazon S3 Glacier Deep Archive?
??x
Amazon S3 Glacier Deep Archive offers extremely low storage costs, starting at $1 per terabyte per month. However, it comes with a significant trade-off in retrieval speed; restoring data from this tier can take up to 12 hours.
x??

---
#### Object Store-Backed Filesystems (s3fs and Amazon S3 File Gateway)
Tools like s3fs and Amazon S3 File Gateway allow users to mount S3 buckets as local filesystems, making it easier to interact with cloud storage.

:p What are object store-backed filesystems?
??x
Object store-backed filesystems enable the use of S3 buckets as if they were traditional block or file storage. Tools such as s3fs and Amazon S3 File Gateway facilitate this by allowing users to mount S3 buckets on their local systems, providing a familiar interface for file operations while leveraging cloud storage capabilities.
x??

---
#### Characteristics of Object Store-Backed Filesystems
When using object store-backed filesystems, writes may be combined into new objects, affecting the performance and efficiency of high-speed transactional writing.

:p What considerations should users keep in mind when using s3fs or Amazon S3 File Gateway?
??x
Users of object store-backed filesystems like s3fs and Amazon S3 File Gateway need to consider how write operations are handled. These tools often combine changes into new objects, which can be efficient but may not support high-speed transactional writing effectively.
x??

---

#### Purpose and Use Case Identification
Background context: As a data engineer, you need to identify the purpose of storing the data. This involves understanding what the data will be used for within your organization.

:p What is the primary consideration when identifying the purpose of storing data as a data engineer?
??x
The primary consideration is determining the purpose and use case of the stored data, such as supporting data science, analytics, or reporting.
x??

---

#### Update Patterns
Background context: Understanding how data will be updated (bulk updates, streaming inserts, or upserts) is crucial for selecting the appropriate storage abstraction.

:p How do update patterns impact the choice of storage abstraction?
??x
Update patterns significantly affect the choice of storage abstraction. For example, bulk updates are more suitable for traditional data warehouses, while streaming inserts and upserts are better handled by systems designed to support real-time processing.
x??

---

#### Cost Considerations
Background context: Direct and indirect financial costs, as well as time to value and opportunity costs, play a critical role in selecting the right storage abstraction.

:p What factors should be considered when evaluating the cost of different storage abstractions?
??x
When evaluating the cost, consider direct and indirect financial costs, such as hardware, software licensing, and maintenance. Also, evaluate the time to value (how quickly you can derive insights) and opportunity costs (what could have been done with the resources spent on a particular system).
x??

---

#### Separation of Storage and Compute
Background context: The trend is moving towards separating storage and compute, though most systems still hybridize both. This separation affects purpose, speed, and cost.

:p How does the separation of storage and compute impact data engineering?
??x
The separation of storage and compute enhances flexibility and scalability. It allows for optimizing each layer independently—storage can focus on large-scale data storage while compute resources handle processing tasks efficiently.
x??

---

#### Data Warehouse Overview
Background context: A data warehouse is a standard OLAP (Online Analytical Processing) data architecture designed to store historical data and support business intelligence queries.

:p What does the term "data warehouse" refer to in data engineering?
??x
The term "data warehouse" refers to technology platforms that store historical data, an architecture for centralizing data, and an organizational pattern within a company aimed at supporting business intelligence queries.
x??

---

#### Storage Trends
Background context: Traditional data warehouses built on conventional transactional databases and MPP (Massively Parallel Processing) systems are evolving towards cloud-based solutions.

:p How have storage trends evolved from traditional to modern data warehousing?
??x
Storage trends have shifted from building data warehouses using conventional transactional databases and row-based MPP systems (e.g., Teradata, IBM Netezza) to utilizing columnar MPP systems (e.g., Vertica, Teradata Columnar), and eventually moving towards cloud data warehouses and platforms like Google BigQuery.
x??

---

#### Data Engineering Storage Abstractions
Background context: Various storage abstractions support different use cases such as data science, analytics, and reporting. These include data warehouse, data lake, data lakehouse, data platforms, and data catalogs.

:p What are the main types of storage abstractions that a data engineer should be familiar with?
??x
The main types of storage abstractions include data warehouses, data lakes, data lakehouses, data platforms, and data catalogs. These abstractions support various use cases such as data science, analytics, and reporting.
x??

---

#### Blurring Lines Between OLAP Databases and Data Lakes
Background context: The lines between OLAP databases and data lakes are becoming increasingly blurred due to the popularity of separating storage from compute.

:p How might the differences between OLAP databases and data lakes be minimized in the future?
??x
The differences between OLAP databases and data lakes may become minimal in the future as major cloud providers continue to evolve their offerings. This evolution involves both functional and technical similarities under the hood, potentially making it difficult to distinguish one from the other.
x??

---

