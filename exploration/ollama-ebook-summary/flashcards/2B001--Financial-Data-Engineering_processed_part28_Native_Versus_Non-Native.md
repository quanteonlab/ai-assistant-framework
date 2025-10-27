# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 28)

**Starting Chapter:** Native Versus Non-Native

---

---
#### Primary Versus Secondary Data Storage Systems
In data storage architecture, a primary Data Storage System (DSS) is designed to be secure and permanent, serving as the stronghold of your data. However, it may lack features necessary for advanced querying and filtering operations. In such cases, a secondary DSS reads from the primary database, providing enhanced capabilities like sophisticated search queries but with potentially reduced consistency.

:p How does a primary Data Storage System differ from a secondary one?
??x
A primary DSS is secure and permanent, acting as the main repository for data storage. It lacks advanced querying features compared to a secondary DSS, which reads from it and supports complex operations like sophisticated search queries but may not maintain the same level of data consistency.

```java
// Example pseudo-code to fetch data from primary and process with secondary system
public class DataHandling {
    PrimaryStorage primaryDB = new PrimaryStorage();
    SecondaryStorage secondaryDB = new SecondaryStorage();

    public void handleData() {
        String[] primaryData = primaryDB.fetchData();
        for (String data : primaryData) {
            // Process data using secondary storage features like Elasticsearch indexing
            secondaryDB.indexData(data);
        }
    }
}
```
x??

---
#### Operational Versus Analytical Processes
Operational processes are focused on day-to-day business operations, such as retrieving customer account details, updating balances, recording financial transactions, and executing payments. In contrast, analytical processes provide insights into business performance by analyzing historical data.

:p What distinguishes operational from analytical processes in a business context?
??x
Operational processes handle routine transactions and queries that are essential for daily business operations. Examples include customer account lookups, balance updates, transaction recording, and payment execution. Analytical processes, on the other hand, involve complex analysis of historical data to gain insights into business performance.

```java
// Example pseudo-code for operational process: Updating a customer's account balance
public class AccountService {
    private Database db;

    public void updateBalance(String accountId, double amount) {
        // Operational process: Update the balance in the primary database
        db.updateBalance(accountId, amount);
    }
}
```
x??

---
#### Conductor and Elasticsearch Usage at Netflix
Netflix uses Conductor for orchestrating workflows, storing primary data with Dynomite. For more advanced search capabilities, Elasticsearch is employed as a secondary indexing system.

:p How does Netflix use Conductor and Elasticsearch together?
??x
Netflix employs Conductor to manage orchestration engines, leveraging Dynomite for primary storage. To enable powerful search functionalities, they integrate Elasticsearch as a secondary system that indexes data from the primary database. This setup allows both efficient workflow management and advanced search capabilities.

```java
// Example pseudo-code showing integration of Conductor with Elasticsearch
public class WorkflowManager {
    private Conductor conductor;
    private ElasticsearchIndexer indexer;

    public void manageWorkflow(String workflowId) {
        // Use Conductor to execute the workflow
        String result = conductor.execute(workflowId);
        
        // Index relevant data using Elasticsearch for future search and discovery
        indexer.indexData(result);
    }
}
```
x??

---

#### OLTP vs. OLAP DSSs
Background context: In Data Storage Systems (DSS), operational processes are often handled by Online Transaction Processing (OLTP) systems, which prioritize reliability and transactional guarantees. Analytical processes, on the other hand, are managed by Online Analytical Processing (OLAP) systems, which focus on speed and advanced querying capabilities.

Typical scenarios involve using OLTP as a primary DSS for business transactions and an OLAP system to store a copy of the data for analytical purposes.
:p What distinguishes OLTP from OLAP in terms of their primary goals?
??x
OLTP systems prioritize reliability and transactional guarantees, whereas OLAP systems focus on speed and advanced querying capabilities. 
For example, if you have a trading desk that needs real-time updates and reliable transactions (OLTP), it might use an RDBMS like PostgreSQL for its primary database.
x??

---

#### Native vs. Non-Native DSSs
Background context: A Data Storage System (DSS) can be specialized for specific tasks or designed to handle multiple types of data, such as tabular, document, graph, and key-value data.

A native DSS is specifically designed and optimized for a particular task, like Neo4j for graph databases. A non-native DSS can handle the same type of data but was not primarily designed for it, such as PostgreSQL.
:p What should you choose when your application’s core functionality relies on an optimized DSS tailored for specific tasks?
??x
Choose a native DSS for applications where the core functionality relies on specialized processing. For example, if you need to perform complex graph analysis, use Neo4j, which is specifically designed and optimized for graph data.
x??

---

#### Multi-Model vs. Polyglot Persistence
Background context: Polyglot persistence uses different native Data Storage Systems (DSSs) for each function, while a multi-model DSS supports multiple Data Storage Models within a single integrated environment.

Polyglot persistence might use PostgreSQL for user accounts, Neo4j for social network data, MongoDB for content storage, and Redis for caching. A multi-model DSS could integrate these functionalities into one system.
:p What is the difference between polyglot persistence and multi-model approaches?
??x
Polyglot persistence involves using different native DSSs for each function, whereas a multi-model approach supports multiple Data Storage Models within a single integrated environment. 
For instance, in polyglot persistence, you might use PostgreSQL (RDBMS) for user accounts, Neo4j (graph database) for social network data, MongoDB (document store) for content storage, and Redis (in-memory data structure store) for caching.
x??

---

#### Example of Multi-Model Approach
Background context: A multi-model DSS can support different Data Storage Models within a single integrated environment. It may offer native or integrated support for different models like tabular, document, graph, and key-value.

Consider a scenario where you need to manage relational data, social network data, user-generated content, and frequently accessed cache data.
:p How might a multi-model approach be implemented in practice?
??x
A multi-model DSS could integrate PostgreSQL (RDBMS) for managing user accounts, Neo4j (graph database) for handling social network data and graph-based queries, MongoDB (document store) for storing and querying user-generated content and posts, and Redis (in-memory data structure store) for caching frequently accessed data.
x??

---

#### Polyglot Persistence Approach
Polyglot persistence is a strategy that allows an application to use multiple database technologies for different components of the application. This approach leverages the strengths of each data storage technology, providing better performance and flexibility.

The trade-off with this method is that it can add significant overhead due to the complexity of managing and integrating multiple databases. It requires careful planning to ensure consistency and interoperability between different storage solutions.

:p What are the key benefits of polyglot persistence?
??x
Polyglot persistence offers better performance by leveraging the strengths of each database technology, as well as flexibility in handling diverse data types. However, it also introduces complexity and potential overhead.
x??

---

#### Multi-Model Approach
Multi-model databases support multiple data models within a single storage system. This approach is simpler to manage compared to polyglot persistence since only one database instance needs to be managed.

However, while multi-model databases can handle various types of data, they might not provide the same level of performance or optimization for specific tasks as specialized databases.

:p What are the advantages and disadvantages of using a multi-model approach?
??x
Advantages: Multi-model databases offer simplicity in management since only one database instance is required. They support multiple data models like document, graph, and key-value storage.
Disadvantages: While they can handle various types of data, they might not provide the same level of performance as specialized databases for specific tasks.

Example:
```java
public class MultiModelDB {
    public void storeDocument(Document doc) { ... }
    public void storeGraph(Graph graph) { ... }
    public void storeKeyValue(String key, String value) { ... }
}
```
x??

---

#### Data Lake Model
A data lake is a centralized repository that stores raw, unprocessed data from various sources. It supports storing large amounts of structured and unstructured data in their original format.

Data lakes offer several advantages for financial institutions, including flexibility, simple data ingestion, data integration, archiving, analysis, insights, governance, and separation of storage and compute layers.

:p What are the key features of a data lake?
??x
Key features of a data lake include:
- Data variety and agility: Flexibility in storing any type of data without predefined schemas.
- Simple data ingestion: Cost-effective and easy to ingest large volumes of raw data.
- Data integration: Centralized storage for all types of data, facilitating compliance, data aggregation, risk management, customer analysis, and experimentation.
- Data archiving: Cost-effective solution for long-term data retention.
- Data analysis and insights: Access historical snapshots and perform ad hoc queries.
- Data governance: Implement access control, cataloging, auditing, reporting, and compliance practices.
- Separation of storage and compute: Scalable architecture to handle large volumes of data.

Example:
```java
public class DataLake {
    public void storeData(byte[] raw_data) { ... }
    public byte[] retrieveData(String query) { ... }
}
```
x??

---

#### Schema on Read vs. Schema on Write
Schema on read and schema on write are two different approaches to handling data schemas in data storage systems.

- **Schema on Write**: Data is modeled and structured before being stored, which can be more efficient for querying but less flexible.
- **Schema on Read**: The data schema is generated during the query process. This approach provides flexibility but may have higher overhead costs due to the dynamic nature of generating schemas at runtime.

:p What are the differences between schema on read and schema on write?
??x
Differences:
- **Schema on Write**: Data models and structures are defined before storage, making querying more efficient but less flexible.
- **Schema on Read**: Schemas are generated dynamically during query execution, providing greater flexibility but higher overhead due to dynamic processing.

Example:
```java
public class SchemaOnWrite {
    public void storeDataWithSchema(byte[] data) { ... }
}

public class SchemaOnRead {
    public byte[] readDataAndGenerateSchema(String query) { ... }
}
```
x??

---

#### Data Ingestion into a Data Lake
Data lakes offer simple and cost-effective data ingestion, especially when dealing with large volumes of raw data. This is because they do not require transformations or harmonization before storage.

:p Why is data ingestion into a data lake simple and cost-effective?
??x
Data ingestion into a data lake is simple and cost-effective because it does not necessitate any transformations or harmonizations prior to storage. The lack of predefined schemas allows for easy and cheap bulk loading of raw, unprocessed data from various sources.

Example:
```java
public class DataLakeIngestor {
    public void ingestData(byte[] raw_data) { ... }
}
```
x??

---

#### Data Lakes Overview

Background context: A data lake is a centralized repository for storing raw data, independent of any specific technology. It emphasizes scalability and fault tolerance rather than enforcing schema or structure on ingested data.

:p What are some characteristics of a data lake?
??x
A data lake typically includes:
- Centralized storage for raw data
- Scalability to handle large amounts of data
- Fault tolerance
- No strict schema enforcement

The architecture allows for diverse and unstructured data ingestion, making it flexible but requiring careful design.
x??

---

#### Technological Implementations - HDFS

Background context: Hadoop Distributed File System (HDFS) is a part of the Hadoop ecosystem used to build big data applications. It runs on commodity servers and provides open source solutions for distributed file storage.

:p What is HDFS, and what are its key features?
??x
HDFS is an open-source distributed file system that allows storing large amounts of data across multiple commodity servers in a fault-tolerant manner.
Key features include:
- Scalability: Can handle petabytes of data.
- Fault tolerance: Replicates data across nodes for reliability.

Example implementation in Java could involve initializing and writing to HDFS:
```java
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;

public class HDFSDemo {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        
        // Initialize connection
        Path path = new Path("hdfs://localhost:9000/testdir");
        if (!fs.exists(path)) {
            fs.mkdirs(path); // Create directory if it doesn't exist
        }
    }
}
```
x??

---

#### Cloud-Based Data Lakes

Background context: Cloud-based data lakes offer ease of use and scalability, making them a popular choice. Managed services like AWS S3 provide seamless integration and reduce maintenance.

:p What are some advantages of using cloud-based data lakes?
??x
Advantages include:
- Simplicity in setup and management.
- Seamless scaling capabilities.
- Reduced need for on-premises hardware and maintenance.

For example, creating an AWS S3 bucket can be done quickly through the console or API with minimal configuration steps.
x??

---

#### Data Modeling in Cloud-Based Data Lakes

Background context: In a cloud-based data lake like AWS S3, data is organized using buckets. These buckets serve as logical containers for different types of data.

:p How are buckets used to model data in a cloud-based data lake?
??x
Buckets in AWS S3 act as logical models where you can organize and categorize your data based on business requirements.
Common organization methods include:
- Landing zones for raw files.
- Staging zones for enriched files.
- Business zones for analytics and research data.
- Trusted zones for anonymized and analysis-ready data.

Example of organizing buckets in AWS S3:
```java
// Pseudocode to demonstrate bucket organization
public class DataLakeOrganization {
    public void organizeBuckets() {
        // Create different types of buckets
        String[] bucketNames = {"logs", "financial-vendors", "analytics"};
        
        for (String name : bucketNames) {
            s3.createBucket(name);
            s3.setBucketPermission(name, permissionSet);
        }
    }
}
```
x??

---

#### Data Governance Considerations

Background context: Building an enterprise-wide data lake requires considering multiple challenges and factors related to governance, such as data modeling and security.

:p What are some key factors to consider when designing a data lake?
??x
Key factors include:
- Data modeling strategies.
- Security and access controls.
- Compliance requirements.
- Quality of data.

For example, ensuring proper bucket organization and implementing user permissions can help manage data effectively.
x??

---

#### Landing Zone Structure
In the context of a data lake, the landing zone is where raw files are stored without any modification. This structure helps organize data by time and format to ensure clarity and ease of access.

:p What is the purpose of the landing zone in a data lake?
??x
The landing zone stores raw files as they arrive from various sources without any transformation or schema enforcement, ensuring that each file's original state is preserved for later processing. This approach facilitates easier validation and initial inspection of data before further transformations are applied.

Code Example to illustrate structure:
```java
public class LandingZone {
    public static String getLandingPath(String format, int year, int month) {
        return "/landing_bucket/" + format + "/" + year + "/" + month;
    }
}
```
x??

---

#### Staging and Business Zones
The staging zone is where data gets cleaned and aggregated on a daily basis. The business zone further splits the data based on specific business areas for more granular analysis.

:p What are the characteristics of the staging and business zones in a data lake?
??x
In the staging zone, data undergoes cleaning and aggregation processes to ensure it meets the required standards before moving into the business zone. This phase is crucial for error detection and data quality improvement. The business zone then organizes this cleaned data by specific business areas, facilitating targeted analysis.

Code Example:
```java
public class DataTransformation {
    public void cleanAndAggregateData(String sourcePath) {
        // Cleaning logic
        // Aggregation logic
        System.out.println("Staging completed.");
    }
    
    public void splitByBusinessArea(String inputPath, String outputPath) {
        // Business area splitting logic
        System.out.println("Split by business area completed.");
    }
}
```
x??

---

#### Trusted Zone for Anonymized Data
A trusted zone is a part of the data lake where anonymized and cleaned data can be stored safely. This ensures that sensitive information is protected while still allowing for analysis.

:p What is the role of the trusted zone in a data lake?
??x
The trusted zone serves as a secure storage area for data that has been anonymized to protect sensitive information. This allows users to perform analysis and generate insights without compromising personal or confidential data.

Code Example:
```java
public class AnonymizationService {
    public void anonymizeData(String sourcePath, String outputPath) {
        // Anonymization logic
        System.out.println("Anonymized data stored in: " + outputPath);
    }
}
```
x??

---

#### Data Governance Layer
Implementing a data governance layer on top of a data lake ensures that metadata, sources, transformations, and other critical aspects are managed effectively. This helps maintain the integrity and quality of the data.

:p Why is implementing a data governance layer essential for a data lake?
??x
Implementing a data governance layer is crucial because it manages metadata, tracks data sources, monitors transformations, and ensures that data is of high quality and consistent. Without proper governance, a data lake can become disorganized and difficult to manage, leading to issues like data inconsistency and accessibility problems.

Code Example:
```java
public class DataGovernance {
    public void registerMetadata(String filePath, String metadata) {
        // Registering metadata logic
    }
    
    public void validateDataSources(String dataSource) {
        // Validation logic for sources
    }
}
```
x??

---

#### Avoiding Mixed File Formats in Folders
To maintain clarity and ease of access, it is recommended to avoid storing different file formats within the same folder. This ensures that files can be easily managed and processed according to their type.

:p How should file formats be organized in a data lake?
??x
File formats should be organized separately within folders to avoid mixing them. For example, if ingestion layers load files in gzip, CSV, and TXT formats, these should have dedicated landing zones for each format to ensure clarity and ease of processing.

Code Example:
```java
public class FileFormatOrganization {
    public void organizeFiles(String[] fileTypes, int year, int month) {
        for (String fileType : fileTypes) {
            String path = "/landing_bucket/" + fileType + "/" + year + "/" + month;
            // Organize files to the appropriate folder
        }
    }
}
```
x??

---

#### Privacy Considerations in Data Lakes
Background context: When dealing with data lakes, privacy is a crucial concern. Sensitive user information such as personally identifiable information (PII) must be handled carefully to ensure compliance with privacy rules. This often involves anonymizing sensitive data and separating storage for sensitive versus non-sensitive data.
:p What are the primary concerns regarding privacy in data lakes?
??x
The main concerns include ensuring the protection of personally identifiable information (PII) through anonymization and separation of storage buckets for different types of data to comply with predefined privacy rules.
x??

---

#### Security Measures in Data Lakes
Background context: Security is vital for protecting data lakes from unauthorized access, malicious attacks, and data loss. Given the cloud-based nature of many data lakes, security measures need to account for potential risks such as cloud misconfiguration.
:p What are some key security measures that should be implemented for data lakes?
??x
Key security measures include encrypting data both at rest and in transit, implementing access controls, establishing data retention rules (like security locks), and ensuring compliance with predefined privacy rules. Additionally, securing against cloud misconfigurations is critical by properly setting up resources, services, or settings.
x??

---

#### Quality Controls in Data Lakes
Background context: The quality of ingested data should be checked to ensure reliability and accuracy, although not all types of data require such checks. For example, log data typically does not need quality controls, but business and financial data do.
:p Why are quality controls important for some data types but not others?
??x
Quality controls are crucial for ensuring the integrity and accuracy of ingested data, particularly for business and financial data where errors could be costly. Log data often does not require these checks because it is more about capturing events rather than deriving insights.
x??

---

#### Data Cataloging in Data Lakes
Background context: A data catalog provides users with functionalities to search for data, metadata, versions, ingestion history, and other attributes within the data lake. This helps in managing and utilizing the vast amount of stored data effectively.
:p What is a data catalog used for in a data lake?
??x
A data catalog serves as a valuable tool that allows users to search for data, metadata, versions, ingestion history, and other associated attributes. It aids in efficiently managing and accessing the large volume of data stored within the data lake.
x??

---

#### Data Governance Layers in Data Lakes
Background context: Implementing a governance layer is essential for maintaining control over privacy, security, quality, and cataloging aspects of the data. However, this can be complex and require significant investment depending on the complexity of the data lake.
:p What are the key elements to consider when implementing a data governance layer in a data lake?
??x
Key elements include focusing initially on privacy, security, quality control, and data cataloging. These elements help in maintaining compliance, protecting sensitive data, ensuring data accuracy, and providing easy access to stored information.
x??

---

#### Data Lakehouse Model
Background context: The data lakehouse model aims to introduce more structure into data lakes by combining a data lake with a data warehouse, along with related analytics and processing services. This approach helps in integrating raw data from the lake with structured queries using SQL-like languages.
:p What is the main idea behind the data lakehouse model?
??x
The main idea of the data lakehouse is to create an integrated data architecture that combines a data lake, data warehouse, and related analytics and processing services. This approach allows for storing raw data in the lake and querying it with SQL-like languages using tools like Amazon Athena or Amazon Redshift Spectrum.
x??

---

#### Financial Use Cases of Data Lakes
Background context: The financial industry has shown significant interest in building data lakes due to their potential benefits, such as improved decision-making processes and enhanced operational efficiency. Managed cloud storage solutions are often leveraged by financial institutions for these purposes.
:p Why have many financial institutions started investing in data lakes?
??x
Financial institutions have started investing in data lakes because of the potential benefits like better decision-making processes and increased operational efficiency. Leveraging managed cloud storage solutions allows them to build robust, scalable systems that can handle large volumes of diverse data types.
x??

---

#### Cloud Migration and Data Lakes
Background context: Many financial institutions are moving to cloud environments, and data lake solutions like AWS S3 are becoming popular due to their simplicity and flexibility. These platforms can store any type or size of data efficiently.

:p What is a key reason for financial institutions choosing data lakes such as AWS S3?
??x
A key reason for financial institutions opting for data lakes like AWS S3 is the ability to store, share, and archive diverse types of unstructured data easily. This flexibility supports regulatory compliance and innovation in data-driven products.
x??

---
#### Unstructured Data Processing
Background context: Financial institutions handle a large amount of unstructured data, including reports, prospectuses, client files, news data, logs, etc., which are challenging to manage using traditional structured database systems.

:p What type of data do financial institutions commonly process?
??x
Financial institutions commonly process unstructured data such as reports, prospectuses, client files, news data, and logs. These types of data require a flexible storage solution like data lakes.
x??

---
#### Data Aggregation for Compliance and Innovation
Background context: Data aggregation is crucial both as a regulatory requirement and to support innovation in financial institutions. By consolidating data in one location, these institutions can quickly respond to regulatory inquiries and enable the development of new data-driven products.

:p How does data aggregation benefit financial institutions?
??x
Data aggregation benefits financial institutions by enabling them to consolidate data from various sources into a single repository, facilitating quick responses to regulatory inquiries and supporting the development of data-driven products. This consolidation is essential for compliance and innovation.
x??

---
#### Compliance in Data Lakes
Background context: Financial institutions must adhere to strict compliance requirements related to data privacy, security, retention, and protection. Data lakes can provide features that meet these needs.

:p What AWS S3 feature ensures the integrity of archived data?
??x
The AWS S3 WORM (Write-Once-Read-Many) storage feature, such as S3 Glacier Vaults and S3 Object Lock, ensures the integrity of archived data in accordance with US SEC and FINRA rules. This feature prevents accidental or unauthorized modifications to stored objects.
x??

---
#### NASDAQ Data Lakehouse Example
Background context: NASDAQ implemented a data lake solution using AWS S3 to manage large-scale data processing for its financial services operations.

:p What did NASDAQ achieve by implementing AWS S3?
??x
NASDAQ achieved the decoupling of the compute layer from the storage layer, allowing independent scaling of each component. This setup enabled seamless management of large datasets and prevented contention issues during concurrent read and write operations.
x??

---
#### Decoupled Compute and Storage Layers
Background context: By implementing AWS S3, NASDAQ was able to separate the compute layer (e.g., data processing) from the storage layer, allowing for more efficient scaling of each component independently.

:p How did NASDAQ manage large datasets?
??x
NASDAQ managed large datasets by decoupling the compute layer from the storage layer using AWS S3. This approach allowed independent scaling of components and seamless management of concurrent read and write operations on large datasets.
x??

---

