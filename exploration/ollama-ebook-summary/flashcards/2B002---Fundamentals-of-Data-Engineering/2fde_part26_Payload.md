# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 26)

**Starting Chapter:** Payload

---

#### Asynchronous Ingestion
Asynchronous ingestion allows for individual events to be processed independently, much like a microservices architecture. Events can become available in storage immediately after being ingested, providing flexibility and handling spikes in event rates gracefully.
:p What is asynchronous ingestion?
??x
Asynchronous ingestion refers to the processing of data where each stage operates on individual events as they become available, without waiting for all previous stages to complete their operations. This method uses a buffer like Kinesis Data Stream to moderate the load and prevent overwhelming downstream processes during high event rates.
```java
// Example Pseudocode
public class EventProcessor {
    public void processEvent(Event event) {
        // Parse and enrich the event
        EnrichedEvent enrichedEvent = parseAndEnrich(event);
        
        // Forward to next stage or buffer if necessary
        buffer.add(enrichedEvent);
    }
}
```
x??

---

#### Individual Event Processing
In an asynchronous pipeline, individual events are processed in parallel across a Beam cluster as soon as they become available. This is beneficial for handling different stages of data processing without waiting for the previous stages to complete.
:p How does individual event processing work in an asynchronous pipeline?
??x
Individual event processing allows each stage of the pipeline to start processing data items as soon as they are ingested, utilizing parallelism across a distributed cluster managed by Apache Beam. This approach ensures that processes can handle varying rates of events efficiently and scale resources dynamically.
```java
// Example Pseudocode
public class BeamStage {
    public void processEvent(Event event) {
        // Parse the event data
        String parsedData = parse(event);
        
        // Enrich or modify the data as needed
        EnrichedEvent enrichedEvent = enrich(parsedData);
        
        // Send to next stage or buffer
        sendToNextStage(enrichedEvent);
    }
}
```
x??

---

#### Kinesis Data Stream Buffering
Kinesis Data Stream acts as a buffer that moderates the load by delaying event processing, ensuring that downstream systems are not overwhelmed during high event rates. It allows quick processing of events when the rate is low and handles any backlog.
:p What role does Kinesis Data Stream play in an asynchronous ingestion pipeline?
??x
Kinesis Data Stream serves as a buffer to moderate the flow of data through the pipeline. When event rates spike, it delays processing to prevent downstream systems from being overwhelmed. During periods of lower event rates, events move quickly through the pipeline.
```java
// Example Pseudocode
public class KinesisBuffer {
    public void addEvent(Event event) {
        // Add event to buffer
        buffer.add(event);
        
        // Process events in batches or as they become available
        processEvents();
    }
    
    private void processEvents() {
        for (Event e : buffer) {
            handleEvent(e);
        }
    }
}
```
x??

---

#### Serialization and Deserialization
Serialization encodes data from a source, preparing it for transmission and storage. Deserialization is the reverse process where data structures are reconstructed at the destination. Proper deserialization ensures that ingested data can be used effectively in downstream processes.
:p What is the importance of serialization and deserialization in data ingestion?
??x
Serialization and deserialization are crucial steps in moving data between different systems or stages within a pipeline. Serialization prepares the data for transmission and storage by encoding it into a format suitable for transfer, while deserialization reconstructs the original data structures at the destination.

```java
// Example Pseudocode
public class Serializer {
    public String serialize(Object data) {
        // Convert object to string representation
        return jsonParser.toJson(data);
    }
    
    public Object deserialize(String data) {
        // Parse JSON back into an object
        return jsonParser.parseObject(data, new TypeReference<Object>() {});
    }
}
```
x??

---

#### Throughput and Scalability
In theory, ingestion should not be a bottleneck. However, in practice, scaling to accommodate high data volumes is essential. Systems must be designed with scalability in mind to ensure they can handle varying data throughput.
:p Why is throughput and scalability important in data ingestion?
??x
Throughput and scalability are critical for handling increasing data volumes as requirements change. As the amount of ingested data grows, systems need to scale resources efficiently to maintain performance without bottlenecks.

```java
// Example Pseudocode
public class ScalableIngestionSystem {
    public void ingestData(List<Event> events) {
        // Scale resources based on event count
        if (events.size() > THRESHOLD) {
            scalingPolicy.scaleUp();
        } else {
            scalingPolicy.scaleDown();
        }
        
        // Process data in batches or individual events
        for (Event e : events) {
            process(e);
        }
    }
    
    private void process(Event event) {
        // Handle each event appropriately
    }
}
```
x??

---

---
#### Reliability and Durability in Data Ingestion
Background context: Ensuring that data pipelines can handle failures gracefully, maintain high uptime, and prevent data loss or corruption is crucial. This involves proper failover mechanisms for ingestion systems to ensure reliability. Data durability means ensuring that once ingested, the data remains available and not lost.
:p What does reliability in data ingestion mean?
??x
Reliability in data ingestion refers to the ability of the system to maintain high uptime and have proper failover mechanisms to handle failures without compromising the integrity of the data being processed.

In more technical terms, it ensures that:
- The system is available when needed (high uptime).
- Failures are handled gracefully with minimal downtime.
- Data is not lost or corrupted during processing. 
```java
public class ReliabilityManager {
    // Code to handle failover and recovery logic
}
```
x??

---
#### Direct and Indirect Costs of Reliability and Durability
Background context: While ensuring reliability and durability can prevent data loss, it also incurs costs both directly (increased cloud and labor expenses) and indirectly (team workload). These trade-offs need to be evaluated based on the criticality of the data and the potential impact of failures.
:p What are the direct and indirect costs associated with building a highly redundant system for reliability?
??x
Direct costs include:
- Increased cloud storage and processing expenses.
- Higher labor costs due to the need for a dedicated team to handle outages.

Indirect costs encompass:
- Continuous monitoring, maintenance, and operations overhead.
- Stress on engineering teams dealing with constant vigilance against potential failures.

Example of evaluating trade-offs:
```java
public class CostEvaluator {
    public double evaluateCosts(double cloudExpenses, int laborHours) {
        return cloudExpenses + (laborHours * hourlyRate);
    }
}
```
x??

---
#### Kind of Data in Ingestion Payloads
Background context: The kind of data ingested directly impacts how it is processed and stored downstream. Kind includes type (e.g., tabular, image) and format (CSV, Parquet). Understanding the nature of the data helps in choosing appropriate storage and processing techniques.
:p What does "kind" refer to when discussing ingestion payloads?
??x
"Kind" refers to the characteristics of the data being ingested, specifically its type and format. Type can be tabular, image, video, text, etc., while format defines how this data is represented in bytes (e.g., CSV, Parquet for tabular data; JPG, PNG for images).

Example code snippet:
```java
public class DataType {
    private String type;
    private String format;

    public void setKind(String type, String format) {
        this.type = type;
        this.format = format;
    }
}
```
x??

---
#### Shape of Data in Ingestion Payloads
Background context: The shape (dimensions) of the data is critical across various stages of the data engineering lifecycle. Understanding the dimensions helps in optimizing storage, processing, and analysis.
:p What does "shape" refer to when discussing ingestion payloads?
??x
"Shape" refers to the dimensional characteristics of the data being ingested, which are crucial for understanding how it should be stored, processed, or analyzed.

Example:
- A tabular dataset may have rows and columns (2D shape).
- An image might have width, height, and color channels (3D shape).

```java
public class DataShape {
    private int dimensions;

    public void setShape(int[] dimensions) {
        this.dimensions = Arrays.stream(dimensions).reduce((a,b) -> a * b).orElse(0);
    }
}
```
x??

---

#### Data Shapes in Various Formats
Background context: Different types of data have distinct shapes that are important for understanding and processing. These include tabular, semistructured JSON, unstructured text, images, uncompressed audio, and more. The size and schema (if applicable) are key factors to consider during the ingestion phase.

:p What are the different types of data mentioned in the context, and what is their typical shape or structure?
??x
The different types of data mentioned include:
- **Tabular Data**: Number of rows and columns.
- **Semistructured JSON**: Key-value pairs with nesting depth.
- **Unstructured Text**: Number of words, characters, or bytes in the text body.
- **Images**: Width, height, and RGB color depth (e.g., 8 bits per pixel).
- **Uncompressed Audio**: Number of channels, sample depth, sample rate, and length.

These shapes are crucial for understanding how to process and store data effectively. For example, when importing a CSV file into a database table with more columns than the table, an error might occur due to mismatched dimensions.
x??

---

#### Schema and Data Types
Background context: Understanding the schema and data types of different kinds of data is essential for effective data processing. While tabular and semistructured data have explicit schemas, unstructured data like text or images may not. Schemas describe the fields and their types within a dataset.

:p What are some examples of data that might have an explicit schema, and what do they typically include?
??x
Examples of data with explicit schemas include:
- **Tabular Data**: Number of rows and columns.
- **Semistructured JSON Data**: Key-value pairs and nesting depth.

These schemas help in defining the structure and types of data within fields. For instance, a tabular dataset might have a schema describing how many rows and columns it contains, while semistructured JSON would define key-value pairs and their nesting levels.
x??

---

#### Engineering Considerations for Ingestion Phase
Background context: The ingestion phase involves handling large datasets efficiently by managing factors like size, compression, and chunking. Compressing data into formats such as ZIP or TAR can reduce payload size, making transmission easier over a network.

:p What are some methods to manage the size of large payloads during the ingestion phase?
??x
Methods to manage the size of large payloads include:
- **Compression**: Using formats like ZIP and TAR to compress the data.
- **Chunking**: Splitting large files into smaller chunks, which can be transmitted individually over a network.

These techniques help in reducing the overall payload size, making it easier to handle during transmission. For example, splitting a large file into 1 MB chunks can make it more manageable for network transfer and reassembly at the destination.
x??

---

#### Schema and APIs
Background context: Understanding schemas is not limited to databases; APIs also present schema challenges. Some vendor APIs come with friendly reporting methods that prepare data for analytics, while others require manual handling.

:p What are some challenges in understanding the underlying schema when working with APIs?
??x
Challenges in understanding the underlying schema include:
- **Complex Object Structures**: Natural structures in object-oriented languages (like Java or Python) may map to messy database schemas.
- **Class Structure Familiarity**: Engineers need to be familiar with the class structure of application code to understand the data flow.

These challenges can make it difficult for engineers to comprehend and work with data effectively, especially when dealing with complex structures that don't align well with operational databases. Understanding these mappings is crucial for successful data integration.
x??

---

#### Data Ingestion Practices
Background context: When loading large files into cloud object storage or data warehouses, splitting the file into smaller chunks can improve network transmission efficiency and ease of handling.

:p How do engineers typically handle large payloads when ingesting them into a cloud environment?
??x
Typically, engineers handle large payloads by:
- **Splitting Files**: Dividing large files into smaller, manageable chunks.
- **Compression**: Compressing these chunks using formats like ZIP or TAR to reduce their size.

For example, if ingesting a 10 GB file, it might be split into 5 MB chunks and compressed before transmission. At the destination, these chunks are reassembled to form the complete dataset.
x??

---

#### Summary of Key Concepts
Background context: This section summarizes key concepts related to data shapes, schemas, payload management, and API handling to provide a comprehensive understanding.

:p What are the main takeaways from the provided text regarding data processing?
??x
Main takeaways include:
- **Data Shapes**: Understanding the shape or structure of different types of data (e.g., tabular, semistructured JSON).
- **Schemas and Data Types**: Recognizing that explicit schemas are crucial for structured data like tables and JSON.
- **Payload Management**: Techniques such as compression and chunking to handle large datasets efficiently.
- **API Schema Handling**: Challenges in understanding API schemas and the importance of familiarization with application code.

These concepts are essential for effective data processing, ingestion, and integration.
x??

---

#### Understanding Data Ingestion and Schema Changes
Background context: The passage discusses how data engineers need to understand application internals, particularly source schemas, during the ingestion process. It highlights that schema changes are frequent but often beyond the control of data engineers, who must implement strategies for detecting and handling these changes.

:p What is the primary responsibility of data engineers concerning source schemas?
??x
Data engineers are responsible for understanding the intricacies of source schemas to ensure smooth data ingestion processes. They need to be aware of potential schema changes that can disrupt downstream pipelines.
x??

---

#### Handling Schema Changes in Ingestion Pipelines
Background context: The text emphasizes the importance of detecting and handling schema changes, noting that while automation tools can help, human intervention is still necessary for informing stakeholders about critical changes.

:p How do data engineers deal with schema changes to maintain pipeline integrity?
??x
Data engineers use a combination of automated detection tools and manual strategies to handle schema changes. They monitor source systems for changes and implement scripts or workflows to update target tables automatically. However, they must also inform analysts and data scientists about any schema changes that violate existing assumptions.
x??

---

#### Importance of Communication in Schema Changes
Background context: Effective communication is crucial between those making schema changes and those impacted by these changes. While automation can help detect schema changes, informing relevant stakeholders about such changes remains essential.

:p Why is communication critical when dealing with schema changes?
??x
Communication is vital because it ensures that all parties involved are aware of the schema changes. Without proper communication, data integrity and consistency might be compromised, leading to potential issues in reports and models.
x??

---

#### Role of Schema Registries in Streaming Data
Background context: In streaming environments, schemas can evolve frequently between producers and consumers. Schema registries help maintain schema integrity by tracking versions and providing a consistent model for serialization and deserialization.

:p What is the role of schema registries in streaming data?
??x
Schema registries act as metadata repositories that track schema versions and histories, ensuring consistency in message serialization and deserialization across producers and consumers.
x??

---

#### Metadata in Data Ingestion
Background context: The passage explains how metadata is crucial for understanding data. Without proper metadata, the value of data can be significantly diminished.

:p What is the significance of metadata in data ingestion?
??x
Metadata provides critical information about the data, such as schema details and historical changes. It helps ensure that the data is correctly understood and used by analysts and data scientists.
x??

---

#### Detecting Schema Changes Using Automation Tools
Background context: The text suggests that automation tools can detect and handle schema changes effectively, but human oversight remains important to inform stakeholders of critical changes.

:p How do automation tools assist in detecting schema changes?
??x
Automation tools can monitor source systems for schema changes and update target tables accordingly. However, data engineers must still ensure that analysts and data scientists are informed about any changes that violate existing assumptions.
x??

---

#### Managing Metadata and Data Lake Challenges
Background context: The text highlights the importance of metadata management in maintaining the quality and usability of data lakes. It warns against potential issues if metadata is not properly managed.

:p Why is managing metadata critical for a data lake?
??x
Managing metadata is crucial because it ensures that data can be effectively understood, accessed, and utilized by various stakeholders. Without proper metadata, data lakes can become disorganized and less valuable.
x??

---

---
#### Push Versus Pull Patterns
Background context: The push versus pull strategy describes different methods of transferring data from a source system to a target. In a push strategy, the source actively sends data to the destination, whereas in a pull strategy, the destination retrieves the data directly from the source.
:p What is the difference between push and pull strategies in data transfer?
??x
In a push strategy, the source initiatively transmits data to the destination without waiting for any acknowledgment or request. In contrast, with a pull strategy, the target actively requests data from the source whenever needed.

For example:
```java
// Push Strategy Example (Source)
public void sendData() {
    // Code to send data
}

// Pull Strategy Example (Target)
public Data getData() {
    return source.getData();
}
```
x??

---
#### Polling for Changes in a Source System
Background context: Polling is another pattern that involves periodically checking the data source for any changes. When changes are detected, the destination pulls the updated data.
:p What is polling and how does it differ from pull strategy?
??x
Polling differs from a regular pull strategy by actively querying the data source at predefined intervals to check for new or changed data instead of passively waiting for data to be sent.

Example:
```java
public class Polling {
    private int interval;
    
    public void startPolling() {
        while (true) {
            // Check if there is any change in the data source
            if (isChangeDetected()) {
                fetchData();
            }
            try {
                Thread.sleep(interval);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    private boolean isChangeDetected() {
        // Logic to detect changes in the data source
        return false;
    }
    
    private void fetchData() {
        // Code to fetch updated data from the source
    }
}
```
x??

---
#### Batch Ingestion Considerations
Background context: Batch ingestion involves processing large sets of data, often at scheduled intervals. This can be done based on time or size criteria.
:p What are the two common types of batch ingestion patterns mentioned in the text?
??x
The two common types of batch ingestion patterns are:
1. Time-interval batch ingestion: Data is ingested at fixed time intervals (e.g., daily).
2. Size-based batch ingestion: Data is broken down into discrete blocks based on the size or number of events.

Example code for time-interval batch ingestion in Java:
```java
public class BatchIngestion {
    private int batchSize;
    
    public void startIngestion() {
        while (true) {
            // Ingest data once every day
            ingestData();
            
            try {
                Thread.sleep(batchSize * 1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    private void ingestData() {
        // Code to read and process data in batches
    }
}
```
x??

---

---
#### Full Snapshots vs. Differential Updates
Background context: Data engineers have to decide between capturing full snapshots of a source system or using differential updates (incremental updates). Full snapshots capture the entire current state, whereas differential updates only pull changes since the last read.

:p What is the difference between full snapshot and differential update methods?
??x
Full snapshot reads involve grabbing the entire current state of the source system on each update. Differential updates, on the other hand, allow engineers to pull only the updates and changes that have occurred since the last read from the source system. Differential updates are preferred for minimizing network traffic and target storage usage.

Code Example:
```java
// Full Snapshot Method
void fullSnapshot() {
    // Logic to fetch the entire current state of the source system.
}

// Differential Update Method
void differentialUpdate(long lastReadTimestamp) {
    // Logic to fetch only changes made after 'lastReadTimestamp'.
}
```
x??

---
#### File-Based Export and Ingestion
Background context: Data is often moved between databases using files. Data is serialized into files in an exchangeable format, which are then provided to an ingestion system. This method provides a push-based approach where the export processes run on the source system side.

:p What are the key advantages of file-based export and ingestion?
??x
Key advantages include:
- Security: Direct access to backend systems is often undesirable due to security reasons.
- Control: Source system engineers have complete control over what data gets exported and how it is preprocessed.
- Flexibility: Files can be provided to target systems in various ways, such as object storage, SFTP, EDI, or SCP.

Code Example:
```java
// Pseudocode for File-Based Export
public void exportDataToFile(String filePath) {
    // Logic to serialize and save data into a file at 'filePath'.
}

// Pseudocode for Ingestion from File
public void ingestDataFromFile(String filePath) {
    // Logic to read the file and process the data.
}
```
x??

---
#### ETL vs. ELT
Background context: Chapter 3 introduced ETL (Extract, Transform, Load) and ELT (Extract, Load, Transform), both common ingestion, storage, and transformation patterns in batch workloads.

:p What does the 'extract' part of ETL and ELT refer to?
??x
The extract part involves getting data from a source system. While it often implies pulling data, extraction can also be push-based. It may require reading metadata and schema changes.

Code Example:
```java
// Pseudocode for Extract in ETL
public void extractData() {
    // Logic to get data from the source system.
}

// Pseudocode for Extract in ELT
public void extractData() {
    // Logic to get data from the source system.
}
```
x??

---
#### Loading Data (ETL and ELT)
Background context: Once data is extracted, it can either be transformed before loading into a storage destination or loaded directly with transformation happening later.

:p What considerations should be kept in mind when loading data?
??x
When loading data:
- Type of system being loaded.
- Schema of the data.
- Performance impact of loading large volumes of data.

Code Example:
```java
// Pseudocode for Loading Data (ETL)
public void loadData() {
    // Logic to load transformed data into a storage destination.
}

// Pseudocode for Loading Data (ELT)
public void loadData() {
    // Logic to load raw or partially transformed data into storage, then transform later.
}
```
x??

---

