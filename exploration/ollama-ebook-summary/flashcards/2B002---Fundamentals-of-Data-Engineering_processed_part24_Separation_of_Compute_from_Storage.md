# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 24)

**Starting Chapter:** Separation of Compute from Storage

---

#### Data Catalog Use Cases
Organizational and technical use cases of data catalogs allow metadata to be easily available, enhancing query performance in a data lakehouse. Business users, analysts, scientists, and engineers can search for relevant data, improving cross-organizational communication and collaboration.
:p What are the organizational benefits of using data catalogs?
??x
Data catalogs facilitate easier access to data across various departments, enabling better collaboration among business users, analysts, scientists, and engineers. They streamline the process of finding relevant datasets, thus enhancing productivity and decision-making capabilities within an organization.
x??

---

#### Data Sharing
Allows organizations and individuals to share specific data with defined permissions between entities. This feature is crucial for internal collaboration (e.g., sharing sandbox data) or external partnerships (e.g., ad tech companies sharing advertising data). Cloud multitenant environments facilitate this but also introduce security challenges that need careful policy management.
:p What does data sharing enable in organizations?
??x
Data sharing enables the exchange of specific datasets between entities with defined permissions. This can be used for internal collaboration, such as data scientists sharing sandbox data with colleagues, or external partnerships where organizations share sensitive information like advertising metrics.
x??

---

#### Schema
The schema defines the expected structure and organization of data, including its form, file format, data types, and hierarchical relationships. Well-defined schemas make data more useful and easier to consume in future analyses.
:p What does a well-defined schema provide for data?
??x
A well-defined schema provides clear instructions on how to interpret and use data effectively. It defines the structure (structured, semistructured, or unstructured), file formats, expected data types, and relationships between different pieces of data, making it easier to manage and utilize in future analyses.
x??

---

#### Schema Patterns: Schema on Write
Traditional pattern where a table has an integrated schema, requiring all writes to conform to this schema. A schema metastore is necessary for enforcing these rules. This approach ensures data consistency but restricts flexibility.
:p What is the main characteristic of schema on write?
??x
The main characteristic of schema on write is that it enforces a predefined schema at the time of writing, ensuring all data conforms to this structure. While this helps maintain consistency and ease of use in future analyses, it limits the flexibility of accepting diverse or changing data types.
x??

---

#### Schema Patterns: Schema on Read
Dynamic creation of schemas when data is read. The reader must determine the schema upon accessing the data. This approach supports more flexible data ingestion but requires careful handling to ensure accurate interpretation.
:p What does schema on read allow?
??x
Schema on read allows for dynamic schema creation at runtime, enabling flexibility in accepting diverse or evolving data types without upfront constraints. While it enhances adaptability, it requires robust mechanisms to accurately interpret the data when accessed.
x??

---

#### Separation of Compute from Storage
A fundamental concept throughout this book, emphasizing that compute and storage should be decoupled for better scalability, performance, and efficiency. This separation allows different components to scale independently based on their specific needs.
:p What is the key advantage of separating compute from storage?
??x
The key advantage of separating compute from storage is improved flexibility and efficiency. Different components can scale independently based on their specific requirements, leading to better overall system performance and resource utilization.
x??

---

#### Colocation of Compute and Storage
Colocation is a method used to improve database performance by placing compute and storage resources close together. This proximity allows for faster, lower-latency disk reads and higher bandwidth compared to remote access.

For transactional databases:
- Data blocks are stored locally on the same machine or within a small network.
- This setup minimizes latency and maximizes I/O operations per second (IOPS).

For analytics query systems using HDFS and MapReduce:
- Data blocks are distributed across multiple nodes in a cluster.
- Map tasks are dispatched to data locations, reducing network overhead.
- Locality optimization is crucial for improving performance.

:p What is colocation of compute and storage?
??x
Colocation of compute and storage refers to the practice of placing computing resources and storage devices close together to minimize latency and improve I/O operations. This method enhances database performance by ensuring that data can be accessed rapidly without incurring significant network overhead.
x??

---

#### Separation of Compute and Storage
The shift toward separating compute from storage is driven by the need for ephemeral resource management, scalability, and improved data durability.

Ephemeral and scalable resources:
- In cloud environments, it’s cheaper to pay-as-you-go than to maintain dedicated hardware 24/7.
- Workloads can vary significantly; therefore, dynamically scaling resources is more efficient.
- Examples include web servers in online retail and periodic big data batch jobs.

Data durability and availability:
- Cloud object stores provide high uptime through redundancy across multiple zones or regions.
- S3, for instance, stores data across multiple zones to ensure high availability even if one zone fails due to a natural disaster.
- Multiple cloud regions can further mitigate the risk of misconfiguration by deploying updates regionally.

:p Why is there a shift toward separating compute and storage?
??x
There is a shift toward separating compute and storage in cloud environments because it allows for more efficient resource management. By dynamically scaling resources based on demand, organizations can reduce costs and improve performance. Cloud object stores provide high uptime and data durability through redundancy across multiple zones or regions.
x??

---

#### Hybrid Separation and Colocation
Hybrid separation and colocation combine the benefits of both approaches by using multi-tier caching and hybrid object storage.

Multitier caching:
- Utilizes object storage for long-term data retention and access.
- Local storage is used during queries and various stages of data pipelines to improve performance.
- Example: Using SSDs or NVMe drives locally, while storing less frequently accessed data on S3 or similar services.

Hybrid object storage:
- Tightly integrates compute with object storage to balance the benefits of both.
- Examples include Google Cloud Storage and Amazon EFS (Elastic File System).

:p How do we hybridize separation and colocation?
??x
We hybridize separation and colocation by using multitier caching, where local storage is used for performance-critical operations while data is stored in object storage for long-term retention. This approach combines the benefits of both methods to optimize resource usage and performance.

For example:
```java
public class DataPipeline {
    private S3Client s3;
    private LocalStorageCache cache;

    public void process() {
        String key = "important-data";
        
        // Check if data is in local cache
        if (cache.containsKey(key)) {
            byte[] data = cache.get(key);
            // Process data locally
            processData(data);
        } else {
            // Fetch from S3 if not in cache
            byte[] data = s3.getObject(key).getContent();
            cache.put(key, data);  // Store in local cache for future use
            processData(data);
        }
    }

    private void processData(byte[] data) {
        // Process the data as needed
    }
}
```
x??

---

#### AWS EMR with S3 and HDFS
AWS Elastic MapReduce (EMR) allows engineers to use temporary Hadoop Distributed File System (HDFS) clusters for processing large datasets. The system can reference both Amazon Simple Storage Service (S3) and local HDFS as filesystems. A common pattern is to:
1. Mount an HDFS cluster on SSD drives.
2. Pull data from S3.
3. Process the data in intermediate steps, storing results locally on HDFS for better performance.
4. Write full results back to S3 once processing is complete.
5. Delete the temporary HDFS cluster.

This approach leverages local storage (SSD) within the EMR cluster to speed up processing and reduce latency compared to direct operations from S3, which can be slower due to network overhead.

:p How does AWS EMR utilize S3 and HDFS in its data processing pipeline?
??x
AWS EMR uses temporary Hadoop Distributed File System (HDFS) clusters running on SSD drives. It leverages both S3 and local HDFS for efficient data storage and processing:
1. Data is initially pulled from S3 into the HDFS cluster.
2. Intermediate results are stored in HDFS to take advantage of faster local disk access.
3. Once all processing steps are complete, final results are written back to S3.
4. The temporary EMR cluster and associated HDFS resources are then deleted.

This method optimizes performance by reducing network latency between S3 and the compute nodes, thereby speeding up data processing tasks.

```java
// Pseudocode for a simplified EMR job flow
public void processEMRJob(String s3BucketName, String hdfsPath) {
    // Pull data from S3 into HDFS
    pullDataFromS3(s3BucketName, hdfsPath);

    // Process the data in intermediate steps
    processDataInHDFS(hdfsPath);

    // Write final results back to S3
    writeResultsToS3(s3BucketName);
}

public void pullDataFromS3(String s3BucketName, String hdfsPath) {
    // Code to copy files from S3 to HDFS
}

public void processDataInHDFS(String hdfsPath) {
    // Code to process data in HDFS using MapReduce or Spark
}

public void writeResultsToS3(String s3BucketName) {
    // Code to move processed results back to S3
}
```
x??

---
#### Apache Spark and HDFS
Apache Spark is often used for big data processing, where it relies on distributed filesystems like HDFS for temporary storage between steps. The main issue with using DRAM (Dynamic Random Access Memory) directly as memory is its high cost. Therefore, running Spark in the cloud allows users to rent large amounts of memory temporarily and release it when not needed.

:p How does Apache Spark use HDFS or other distributed filesystems?
??x
Apache Spark uses HDFS or another ephemeral distributed filesystem for temporary storage between processing steps. This approach helps manage the cost of DRAM, which is expensive compared to disk-based storage:
1. Data can be stored temporarily on a distributed filesystem.
2. Intermediate results are processed and stored in memory (DRAM) for faster access during the computation.
3. Final results are written back to persistent storage such as HDFS or another suitable filesystem.

This method balances performance with cost efficiency, leveraging cheaper disk-based storage while taking advantage of fast in-memory processing.

```java
// Pseudocode for a simplified Spark job flow
public void runSparkJob(String inputPath, String outputPath) {
    // Load data from distributed filesystem (e.g., HDFS)
    Dataset<Row> dataset = spark.read().format("csv").load(inputPath);

    // Process the data using Spark transformations and actions
    dataset.transformationsAndActions();

    // Write final results back to persistent storage (e.g., HDFS or S3)
    dataset.write().mode(SaveMode.Overwrite).parquet(outputPath);
}
```
x??

---
#### Apache Druid and SSDs
Apache Druid is designed for high performance by heavily relying on Solid State Drives (SSDs) for data storage. Given that SSDs are significantly more expensive than magnetic disks, Druid optimizes storage costs by keeping only one copy of the data in its cluster. This approach reduces "live" storage costs by a factor of three.

:p How does Apache Druid manage storage costs while maintaining high performance?
??x
Apache Druid maintains high performance and manages storage costs efficiently by:
1. Using SSDs for fast data access.
2. Keeping only one copy of the data in the cluster to reduce live storage costs (by a factor of three).
3. Utilizing an object store as its durability layer, ensuring data can be recovered from node failures or data corruption.

This approach balances performance with cost efficiency by leveraging expensive SSDs for fast read/write operations and using cheaper object stores for backup and recovery purposes.

```java
// Pseudocode for Apache Druid's data ingestion process
public void ingestDataIntoDruid(String dataSource, String jsonData) {
    // Process and serialize the input data into compressed columns
    byte[] serializedData = processDataAndSerialize(jsonData);

    // Write the serialized data to cluster SSDs and object storage
    writeDataToCluster(serializedData);
}

// Pseudocode for writing data to the cluster
private void writeDataToCluster(byte[] data) {
    // Write data to SSDs in the cluster
    writeToSSDs(data);

    // Store a copy of the data in an object store for durability and recovery
    storeInObjectStore(data);
}
```
x??

---
#### Hybrid Object Storage
Google's Colossus file storage system supports fine-grained control over data block location, allowing BigQuery to colocate customer tables in single locations. This results in ultra-high bandwidth queries by leveraging local network resources.

:p What is hybrid object storage and how does it work?
??x
Hybrid object storage combines the clean abstractions of object storage with some advantages of colocating compute and storage:
1. It allows for fine-grained control over data block location.
2. Colocating tables in a single location can significantly improve query performance by maximizing local network bandwidth.

Public cloud services like AWS S3 Select allow users to filter S3 data directly within the S3 cluster, optimizing performance and efficiency.

```java
// Pseudocode for hybrid object storage usage
public void processS3Data(String s3BucketName) {
    // Use S3 Select to filter data locally before returning it over the network
    String filteredData = filterDataLocally(s3BucketName);

    // Process the filtered data as needed
    processData(filteredData);
}

private String filterDataLocally(String s3BucketName) {
    // Code to query S3 for specific data and process it locally within the cluster
    return s3Client.filterData(s3BucketName, criteria);
}
```
x??

---

#### Zero-Copy Cloning
Background context: In cloud-based systems, zero-copy cloning is a technique where a new virtual copy of an object (e.g., a table) is created without physically copying underlying data. Instead, new pointers are created to raw data files, and any future changes will only be recorded in the new object.
This can significantly reduce storage costs and improve performance by avoiding unnecessary data duplication.
If applicable, add code examples with explanations:
```python
# Example of shallow copy (zero-copy cloning) in Python
old_table = [1, 2, 3]
new_table = old_table[:]  # Shallow copy, new_table points to the same memory as old_table

# Modifying old_table will not affect new_table since they share the same data
old_table.append(4)
print(new_table)  # Output: [1, 2, 3]
```
:p How does zero-copy cloning work in cloud-based systems?
??x
Zero-copy cloning works by creating a new virtual copy of an object (e.g., a table) without physically copying the underlying data. Instead, new pointers are created to the raw data files, ensuring that any future changes will only be recorded in the new object and not affect the old one.
This approach is efficient because it avoids unnecessary duplication of data, reducing storage costs and improving performance.
x??

---
#### Hot, Warm, Cold Data
Background context: Depending on how frequently data is accessed, it can be categorized into three persistence levels—hot, warm, and cold. Each tier has different hardware and cost implications:
- **Hot data**: Instant or frequent access requirements with expensive storage but cheap retrieval.
- **Warm data**: Accessed semi-regularly (e.g., monthly) with cheaper storage than hot but more expensive than cold.
- **Cold data**: Infrequently accessed, stored in durable and cheap hardware like HDDs or tapes, intended for long-term archival.

If applicable, add code examples with explanations:
```java
// Example of moving data based on access patterns
if (dataAccessFrequency == "FREQUENT") {
    moveToTier("HOT");
} else if (dataAccessFrequency == "MONTHLY") {
    moveToTier("WARM");
} else {
    moveToTier("COLD");
}
```
:p How do you categorize data into hot, warm, and cold tiers based on access patterns?
??x
Data is categorized into hot, warm, and cold tiers based on the frequency of its access:
- **Hot Data**: Frequently accessed with high performance requirements. Storing this data typically uses SSD or memory.
- **Warm Data**: Accessed less frequently (e.g., monthly) but still needed occasionally. Storage tiers like S3 Infrequently Accessed Tier or Google Cloud Nearline accommodate warm data.
- **Cold Data**: Infrequently accessed, stored in cheaper and durable hardware like HDDs or cloud-based archival systems for long-term archiving.

This categorization helps optimize storage costs by balancing performance needs with financial constraints.
x??

---
#### Data Storage Lifecycle
Background context: The data storage lifecycle involves understanding how often data is accessed (frequency) and how important it is to downstream users. This concept includes:
- **Access Frequency**: How frequently the data is queried or used.
- **Use Cases**: The intended purpose of the data.

If applicable, add code examples with explanations:
```python
# Example of evaluating access frequency for data retention policy
if (queryAccessFrequency >= "FREQUENT"):
    retainForever("hot")
elif (queryAccessFrequency == "MONTHLY"):
    moveToTier("warm")
else:
    moveToTier("cold")
```
:p What does the data storage lifecycle involve?
??x
The data storage lifecycle involves evaluating how often data is accessed and its importance to downstream users. It includes understanding access frequency, use cases, and retention policies.

Key aspects are:
- **Hot Data**: Frequently queried with high performance requirements.
- **Warm Data**: Accessed less frequently but still needed occasionally.
- **Cold Data**: Infrequently accessed, stored for long-term archiving.

This lifecycle helps in optimizing storage costs by ensuring data is stored appropriately according to its access patterns and value.
x??

---
#### Data Retention
Background context: With the rise of big data, there was a tendency to retain all possible data. However, this led to unwieldy and dirty datasets, causing regulatory issues. Modern data engineers must consider:
- **Data Value**: The worth of stored data based on its use case.
- **Impact on Downstream Users**: How missing data affects users.

If applicable, add code examples with explanations:
```python
# Example of implementing a data retention policy
if (dataValue == "CRITICAL" and queryAccessFrequency >= "FREQUENT"):
    retainForever("hot")
elif (dataValue == "MEDIUM" and queryAccessFrequency == "MONTHLY"):
    moveToTier("warm")
else:
    moveToTier("cold")
```
:p What is data retention, and why is it important?
??x
Data retention involves deciding what data to keep and for how long based on its value and use cases. It's crucial because retaining all data can lead to inefficiencies and regulatory issues.

Key points are:
- **Value of Data**: Determine if the data is critical or can be easily re-created.
- **Impact on Users**: Assess how missing data affects downstream processes.

By implementing proper retention policies, you ensure that only necessary data is stored, optimizing costs and maintaining data quality.
x??

---

#### Time Considerations for Data Value
Background context: The value of data to downstream users depends on its age. Newer data is generally more valuable and frequently accessed than older data due to relevance and timeliness.

Technical limitations may determine how long data can be stored in certain tiers, such as cache or memory. Hot storage needs a time-to-live (TTL) setting to prevent it from becoming full and impacting performance.
:p How does the age of data affect its value?
??x
The value of data increases with its recency because newer information is more relevant and valuable for decision-making processes. Older data may still be useful but has less immediate relevance.

For example, in a financial application, recent stock prices are crucial for trading decisions, whereas historical trends might also provide insights but not as urgently.
x??

---
#### Compliance Requirements
Background context: Certain regulations require you to keep specific data accessible for certain periods. These requirements include the need to hold data until it is no longer needed or mandated by compliance standards like HIPAA and PCI.

Compliance also involves having a process for searching, accessing, and deleting data in line with regulatory guidelines.
:p How do compliance requirements impact data storage?
??x
Compliance requirements dictate how long you must retain data and under what conditions. For instance, HIPAA requires healthcare data to be kept for at least six years after the last interaction or treatment.

You need a robust storage and archival system that ensures data can be searched and accessed as required by regulations while also allowing for timely deletion if necessary.
x??

---
#### Cost Management
Background context: Managing costs associated with storing large volumes of data is essential. Automatic data lifecycle management practices help in moving less frequently used data (cold storage) from expensive hot tiers to cheaper storage solutions.

The goal is to balance cost efficiency without compromising on data accessibility and compliance requirements.
:p How does automatic data lifecycle management impact storage costs?
??x
Automatic data lifecycle management allows you to move infrequently accessed data to colder, more economical storage tiers. This practice helps in reducing overall storage costs by ensuring that expensive hot storage only holds the most relevant or frequently used data.

For example:
```python
def manage_data_lifecycle(data, retention_period):
    current_time = datetime.now()
    for item in data:
        if (current_time - item['last_accessed']).days > retention_period:
            item['storage_tier'] = 'cold'
```
x??

---
#### Single-Tenant vs. Multitenant Storage
Background context: The choice between single-tenant and multitenant storage architectures affects how resources are allocated, shared, and accessed by different users or organizations.

Single-tenant storage provides dedicated resources for each tenant, ensuring isolation and security but at a higher cost. In contrast, multitenant storage shares these resources among multiple tenants, reducing costs but potentially compromising on performance if not managed correctly.
:p What is the key difference between single-tenant and multitenant storage?
??x
Single-tenant storage provides dedicated resources (compute, network, storage) for each tenant, ensuring isolation. Multitenant storage, however, shares these resources among multiple tenants, reducing costs but potentially impacting performance if not managed properly.

In a single-tenant architecture:
```python
class SingleTenantStorage:
    def __init__(self):
        self.tenant_database = {}

    def add_tenant(self, tenant_id, database):
        self.tenant_database[tenant_id] = database

    def get_data(self, tenant_id, query):
        return self.tenant_database[tenant_id].execute_query(query)
```
In a multitenant architecture:
```python
class MultiTenantStorage:
    def __init__(self):
        self.shared_database = {}

    def add_tenant(self, tenant_id, data):
        if tenant_id not in self.shared_database:
            self.shared_database[tenant_id] = data

    def get_data(self, tenant_id, query):
        return self.shared_database[tenant_id].execute_query(query)
```
x??

---
#### Working with Other Teams
Background context: Effective collaboration is critical between data engineers and other teams such as DevOps, security, and cloud architects. Defining clear roles and responsibilities helps in creating efficient workflows.

Data engineers need to understand the domain of their operations and ensure they have the necessary permissions and processes in place for deploying infrastructure.
:p How do data engineers interact with other IT infrastructure owners?
??x
Data engineers work closely with DevOps teams, security specialists, and cloud architects. Clear definitions of roles and responsibilities are essential for smooth collaboration.

For instance:
- Data engineers should know if they can deploy their infrastructure directly or need to coordinate with DevOps.
- Security requirements must be clearly defined and enforced across all data operations.
- Cloud architects provide insights into the best practices for storage architecture design, ensuring compliance and cost-effectiveness.

This interaction helps in creating a unified and efficient IT environment.
x??

---

