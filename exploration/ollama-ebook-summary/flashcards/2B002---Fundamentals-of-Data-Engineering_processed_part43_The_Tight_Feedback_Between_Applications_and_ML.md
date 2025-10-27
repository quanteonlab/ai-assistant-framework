# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 43)

**Starting Chapter:** The Tight Feedback Between Applications and ML

---

#### Modern Data Stack (MDS) Limitations
Background context: The modern data stack is praised for its powerful tools, cost-effectiveness, and empowerment of analysts. However, it has limitations when compared to next-generation real-time applications due to its cloud data warehouse-centric nature.

:p What are the key limitations of the Modern Data Stack?
??x
The key limitations include:

- It is essentially a repackaging of old data warehousing practices using modern technologies.
- It focuses on internal-facing analytics and data science, which may not meet the demands of real-time applications.
- Batch processing techniques limit its ability to handle continuous streams of data efficiently.

x??

---

#### Live Data Stack Evolution
Background context: The live data stack aims to move beyond traditional MDS by integrating real-time analytics and ML into applications. This evolution is driven by the need for automation and sophisticated real-time data processing in business-critical applications like TikTok, Uber, or Google.

:p What drives the shift toward a Live Data Stack?
??x
The shift toward a live data stack is driven by:

- Automation replacing repetitive analytical tasks.
- The need for real-time decision-making and actions based on events as they occur.
- The democratization of advanced streaming technologies previously exclusive to large tech companies.

x??

---

#### Streaming Pipelines and Real-Time Analytical Databases
Background context: Traditional MDS focuses on batch processing, while the live data stack embraces streaming pipelines and real-time analytical databases for continuous data flow. These tools enable subsecond queries and fast ingestion.

:p What are the two core technologies of the live data stack?
??x
The two core technologies of the live data stack are:

- Streaming Pipelines: Continuous stream-based processing.
- Real-Time Analytical Databases: Enabling fast ingestion and real-time query capabilities.

x??

---

#### ETL vs STL Transformation
Background context: The move from traditional ELT to modern STL transformations is driven by the need for continuous data streams. This shift impacts how data is extracted, transformed, and loaded into systems.

:p What does STL stand for in the context of data engineering?
??x
STL stands for Stream, Transform, and Load. It refers to a transformation approach where:

- Extraction: Continuous process.
- Transformation: Occurs as part of the streaming pipeline.
- Loading: Integrates real-time data into storage systems.

x??

---

#### Data Modeling in Real-Time Systems
Background context: Traditional batch-oriented modeling techniques are not suitable for real-time systems. New data-modeling approaches will be needed to handle dynamic and continuous streams of data.

:p Why is traditional data modeling less suited for real-time systems?
??x
Traditional data modeling techniques, such as those used in the MDS, are designed for batch processing and ad hoc queries. They struggle with:

- Continuous streaming ingestion.
- Real-time query requirements.
- Dynamic and evolving data definitions.

New approaches will focus on upstream definitions layers, metrics, lineage, and continuous evolution of data models throughout the lifecycle.

x??

---

#### Fusion of Application and Data Layers
Background context: The integration of application and data layers is a key aspect of the live data stack. This fusion aims to create seamless real-time decision-making within applications.

:p How will the application and data layers be integrated in the future?
??x
The application and data layers will be integrated by:

- Applications becoming part of the data stack.
- Real-time automation and decision-making powered by streaming pipelines and ML.
- Shortening the time between stages of the data engineering lifecycle through continuous updates.

x??

---

#### Tight Feedback Between Applications and ML
Background context: The text discusses the future trend of integrating machine learning (ML) more closely with applications, emphasizing that this integration is expected to become tighter as data volumes and velocities increase. This integration will lead to smarter applications capable of real-time adaptation based on data changes.
:p How does the tightening feedback loop between applications and ML impact application intelligence?
??x
The tightening feedback loop means that applications can now receive and process data in real time, allowing them to adapt and make decisions based on up-to-date information. This leads to more intelligent and responsive applications that provide better user experiences and increased business value.
```java
// Pseudocode for a simple application that uses ML for real-time decision-making
public class SmartApplication {
    private MachineLearningModel model;
    
    public void processData(Data data) {
        // Process the incoming data using the ML model
        Prediction prediction = model.predict(data);
        
        // Use the prediction to adapt the application's behavior in real time
        if (prediction.isActionNeeded()) {
            takeAction(prediction.getAction());
        }
    }
    
    private void takeAction(Action action) {
        // Code for taking a specific action based on the ML prediction
    }
}
```
x??

---

#### Dark Matter Data and Spreadsheets
Background context: The text highlights that despite BI tools, spreadsheets remain one of the most widely used data platforms. These spreadsheets are often used for complex analytics but do not fully integrate into sophisticated data systems or databases. The author suggests a new class of tools might emerge to combine spreadsheet-like interactivity with cloud OLAP backend power.
:p Why is the use of spreadsheets in data analysis considered "dark matter"?
??x
The term "dark matter" is used metaphorically to describe how much data analytics and processing happens within spreadsheets but remains hidden or disconnected from formal, managed data systems. This means a significant portion of business-critical information may not be fully integrated into enterprise data management practices.
```java
// Pseudocode for a basic spreadsheet operation (simplified)
public class Spreadsheet {
    private List<List<String>> data;
    
    public void addRow(List<String> newRow) {
        // Add a new row to the spreadsheet
        data.add(newRow);
    }
    
    public List<String> getData(int rowIndex) {
        // Return a specific row of data
        return data.get(rowIndex);
    }
}
```
x??

---

#### The Future of Data Engineering
Background context: This section discusses various trends and predictions in data engineering, including the rise of managed tooling, the live data stack, and the potential shift towards real-time processing over batch processing. It also emphasizes the importance of practical application and ongoing exploration.
:p What does the author suggest is critical for successful technology adoption?
??x
For successful technology adoption, it is crucial to focus on practical applications that improve user experience, create value, and define new types of applications. This involves not just creating tools but also using them effectively in real-world scenarios.

The author suggests continuously exploring new technologies, engaging with communities, reading the latest literature, participating in meetups, asking questions, and sharing expertise.
```java
// Example of a function that identifies trends or technologies to explore
public List<String> identifyTrends() {
    // Pseudocode for identifying interesting trends or technologies based on research
    String[] potentialTrends = {"Real-time data processing", "Cloud analytics", "AI-driven insights"};
    
    return Arrays.asList(potentialTrends);
}
```
x??

---

---
#### Columnar Serialization Overview
Columnar serialization is a data storage technique that organizes data by columns instead of rows, which can be advantageous for certain types of queries. This method allows reading only specific fields rather than entire rows, reducing the amount of data read and processed.

:p What are the main advantages of columnar serialization?
??x
The primary advantages include:
- Ability to read a subset of fields without scanning full rows.
- Efficient storage due to similar values being stored together, enabling compression techniques like tokenizing repeated values.
- Better suited for analytics where complex queries may need to scan only specific columns.

Code Example (Pseudocode):
```java
// Pseudocode example to illustrate columnar read operation
for each column of interest:
    read the column data;
    process the column data as needed;
```
x??

---
#### Schema Information in Parquet
Parquet stores schema information within its encoded data, making it self-describing. This is unlike CSV where schemas need to be separately defined or inferred.

:p How does Parquet handle schema information compared to CSV?
??x
Unlike CSV, which requires external metadata for schema definition, Parquet embeds schema details directly into the file format. This makes Parquet more portable and easier to use with various tools without needing additional configuration steps.

Code Example (Pseudocode):
```java
// Pseudocode example to read a parquet file
parquetFile = openParquetFile();
schemaInfo = getSchemaFromParquet(parquetFile);
processDataAccordingToSchema(schemaInfo, parquetFile);
```
x??

---
#### ORC Storage Format
ORC (Optimized Row Columnar) is another columnar storage format similar to Parquet. It was primarily used with Apache Hive but has seen less adoption in modern cloud ecosystems due to newer and more versatile formats like Parquet.

:p How does ORC differ from Parquet?
??x
ORC and Parquet are both columnar storage formats, but ORC has seen reduced usage compared to Parquet. While both support columnar storage, ORC is not as widely supported in modern cloud tools such as Snowflake and BigQuery, which prefer or have better integration with Parquet.

Code Example (Pseudocode):
```java
// Pseudocode example to check ORC file support
if (supportsORCImportAndExport(tool)) {
    importDataFromORC(file);
} else {
    // Handle lack of support for ORC files in the tool
}
```
x??

---
#### Apache Arrow Overview
Apache Arrow is a memory format that supports efficient data interchange and storage. It allows dense columnar packing, making it suitable for both in-memory processing and long-term storage.

:p What is the main purpose of Apache Arrow?
??x
The primary purpose of Apache Arrow is to provide an optimized data format for in-memory processing and efficient data exchange between systems. It supports complex structures like nested JSON while enabling high performance through dense columnar packing and direct memory access.

Code Example (Pseudocode):
```java
// Pseudocode example to use Apache Arrow for in-memory storage
arrowMemoryChunk = new ArrowMemoryChunk();
addDataToArrowMemoryChunk(arrowMemoryChunk, data);
queryDataFromArrowMemoryChunk(arrowMemoryChunk);
```
x??

---
#### Shredding Technique in Apache Arrow
The shredding technique used by Apache Arrow maps complex nested structures (like JSON) into separate columns. This makes it easier to process and query such data without the overhead of deserialization.

:p How does the shredding technique work in Apache Arrow?
??x
In Apache Arrow, the shredding technique converts complex nested data structures (e.g., JSON documents) into multiple columns. Each location within the schema is mapped to a separate column, allowing for efficient processing and querying of nested data without needing to deserialize the entire structure.

Code Example (Pseudocode):
```java
// Pseudocode example to apply shredding in Apache Arrow
for each field in the JSON document:
    createColumnForField(field);
    mapValuesFromDocumentToColumn(document, column);
```
x??

---

