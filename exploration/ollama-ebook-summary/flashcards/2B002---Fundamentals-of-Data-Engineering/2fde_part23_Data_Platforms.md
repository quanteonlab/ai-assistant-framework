# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 23)

**Starting Chapter:** Data Platforms

---

#### Data Lakes Revisited
Background context: James Dixon, in his blog post "Data Lakes Revisited," discusses how cloud data warehouses are often used to organize massive amounts of unprocessed raw data similar to a true data lake. Cloud data warehouses excel in handling large volumes of structured and semi-structured data but struggle with truly unstructured data such as images or videos.
:p What did James Dixon propose regarding the use of cloud data warehouses?
??x
James Dixon suggested that cloud data warehouses can be used to organize and manage massive amounts of raw, unprocessed data akin to a data lake. The key advantage is their ability to handle large volumes of structured and semi-structured data efficiently.
x??

---

#### Cloud Data Warehouses and True Data Lakes
Background context: Cloud data warehouses are designed for handling massive amounts of structured and semi-structured data but lack the capability to manage truly unstructured data such as images, video, or audio. A true data lake stores raw, unprocessed data extensively, typically leveraging Hadoop systems.
:p How do cloud data warehouses differ from true data lakes in terms of managing data?
??x
Cloud data warehouses are optimized for structured and semi-structured data but cannot handle truly unstructured data like images, videos, or audio. In contrast, true data lakes store raw, unprocessed data extensively and can manage a wide variety of formats including unstructured ones.
x??

---

#### Data Lake Evolution: Migration to Cloud Object Storage
Background context: The evolution of data lake storage has seen significant changes over the last five years, primarily focusing on migrating from Hadoop systems towards cloud object storage for long-term data retention. This migration offers cost benefits and scalability advantages.
:p What major development occurred in the evolution of data lakes regarding storage?
??x
A major development was the move away from Hadoop towards cloud object storage for long-term data retention in data lake environments, offering improved cost efficiency and scalability.
x??

---

#### Data Lakehouse Concept
Background context: The data lakehouse combines aspects of traditional data warehouses and raw unprocessed data lakes. It provides a balance by retaining the benefits of both architectures—raw data storage like a true data lake and robust table and schema support akin to a data warehouse.
:p How does the data lakehouse concept integrate elements from data lakes and data warehouses?
??x
The data lakehouse integrates the raw, unprocessed data storage of a true data lake with the robust table and schema support, along with features for managing incremental updates and deletes, found in traditional data warehouses.
x??

---

#### Delta Lake as an Open Source Storage Management System
Background context: Databricks promoted the concept of the data lakehouse through their open source storage management system called Delta Lake. It supports versioning, rollback, and advanced data management functionalities on object storage.
:p What is Delta Lake, and what does it offer?
??x
Delta Lake is an open-source storage management system that enhances raw data lakes by providing robust table and schema support, incremental updates, deletes, and version control capabilities, making it suitable for both structured and semi-structured data.
x??

---

#### Similarities with Commercial Data Platforms
Background context: The architecture of the data lakehouse is similar to commercial data platforms like BigQuery and Snowflake. These systems store data in object storage while providing advanced management features.
:p How do the architectures of data lakehouses compare with those of commercial data platforms?
??x
The architecture of data lakehouses mirrors that of commercial data platforms such as BigQuery and Snowflake, which both store data in object storage but offer advanced data management functionalities to ensure robustness and scalability.
x??

---

#### Data Lakehouse Architecture
A data lakehouse combines elements of both data lakes and data warehouses, aiming to balance the flexibility of a lake with the structured querying capabilities of a warehouse. This architecture supports automated metadata management and table history, ensuring seamless update and delete operations without exposing the complexity of underlying file and storage management.
:p What is a key characteristic of a data lakehouse?
??x
A data lakehouse combines the flexibility of a data lake with the structured querying capabilities of a data warehouse, providing both raw and structured data storage alongside automated metadata management and support for various tools to read data directly from object storage without managing underlying files.
x??

---
#### Interoperability in Data Lakehouses
Interoperability is a significant advantage of data lakehouses over proprietary tools. Storing data in open file formats allows easier exchange between different tools, reducing the overhead associated with reserializing data from proprietary formats to common ones like Parquet or ORC.
:p What does interoperability mean in the context of data lakehouses?
??x
Interoperability in data lakehouses means that various tools can easily connect to the metadata layer and read data directly from object storage, without having to manage or convert data stored in a proprietary format. This facilitates seamless integration between different analytical tools.
x??

---
#### Stream-to-Batch Storage Architecture
The stream-to-batch architecture involves writing streaming data to multiple consumers: real-time processing systems for generating statistics on the stream and batch storage consumers for long-term retention and batch queries. Tools like AWS Kinesis Firehose can generate S3 objects based on configurable triggers, while BigQuery automatically reserializes streaming data into columnar object storage.
:p How does a stream-to-batch architecture handle real-time and historical data?
??x
In a stream-to-batch architecture, streaming data is written to multiple consumers: one or more real-time processing systems for generating statistics on the fly, and at least one batch storage consumer for long-term retention and batch queries. AWS Kinesis Firehose can create S3 objects based on configurable triggers (like time or batch size), while BigQuery automatically converts streaming data into columnar object storage.
x??

---
#### Data Platforms
Data platforms are vendor-created ecosystems of interoperable tools with tight integration into the core data storage layer, aiming to simplify the work of data engineering and potentially generate significant vendor lock-in. These platforms often emphasize close integration with object storage for handling unstructured data use cases.
:p What is a key feature of data platforms?
??x
A key feature of data platforms is their ability to provide an ecosystem of interoperable tools that are tightly integrated into the core data storage layer, designed to simplify the work of data engineering and potentially create significant vendor lock-in. These platforms often integrate closely with object storage for handling unstructured data use cases.
x??

---
#### Data Engineering Storage Abstractions
Data engineering storage abstractions refer to managing data in a way that abstracts away the complexities of underlying file and storage management, allowing engineers to focus on higher-level operations such as metadata management and querying. This is especially relevant when dealing with large-scale data processing and storage needs.
:p What does data engineering storage abstraction aim to achieve?
??x
Data engineering storage abstractions aim to simplify the handling of large-scale data by abstracting away the complexities of underlying file and storage management, allowing engineers to focus on higher-level operations such as metadata management and querying. This is particularly useful in environments where data volume and variety are high.
x??

---

#### Data Catalog
A data catalog is a centralized metadata store for all data across an organization. It integrates with various systems and abstractions, working across operational and analytics data sources while providing lineage and presentation of data relationships.
:p What is a data catalog?
??x
A data catalog serves as a central repository for metadata related to the organization's datasets, enabling users to search, discover, and understand their data better. It supports integration with different data systems like data lakes, warehouses, and operational databases.
??x

---

#### Catalog Application Integration
Ideally, data applications are designed to integrate directly with catalog APIs to handle their metadata and updates. As catalogs become more prevalent in an organization, this ideal becomes more achievable.
:p How can data applications be integrated with a data catalog?
??x
Data applications can be integrated with a data catalog by leveraging its APIs. This integration allows the application to automatically manage metadata related to the datasets it uses. Here is a simplified example of how this might look in pseudocode:

```pseudocode
// Pseudocode for integrating an application with a data catalog API

function integrateWithCatalog(appName) {
    // Step 1: Initialize connection to the data catalog API
    let catalogAPI = new CatalogAPI()

    // Step 2: Register the application with the catalog
    catalogAPI.registerApplication(appName)

    // Step 3: Define metadata handling functions for the app
    function handleMetadataChanges() {
        // Function logic to update and retrieve metadata as needed
    }

    // Step 4: Use catalog API throughout the application's lifecycle
    while (true) {
        // Fetch data from various sources using catalog metadata
        let metadata = catalogAPI.getMetadataForDatasets()
        
        // Process the fetched metadata
        handleMetadataChanges(metadata)
    }
}
```
This example demonstrates how an application can continuously interact with a data catalog to manage and use its metadata.
??x

---

#### Automated Scanning
In practice, cataloging systems often rely on automated scanning layers that collect metadata from various sources such as data lakes, warehouses, and operational databases. These tools can infer relationships or sensitive data attributes automatically.
:p How do cataloging systems typically gather metadata?
??x
Cataloging systems use automated scanning layers to collect metadata from diverse data sources like data lakes, warehouses, and operational databases. This process can also involve inferring additional information such as key relationships or identifying sensitive data.

Here is a simplified example of an automated scanning function:

```pseudocode
// Pseudocode for an automated scanning mechanism

function scanAndCollectMetadata(source) {
    // Step 1: Connect to the data source (e.g., database, file system)
    let dataSource = connectToDataSource(source)

    // Step 2: Scan the data and collect metadata information
    let metadataInfo = dataSource.scanData()

    // Step 3: Infer relationships or sensitive data attributes if necessary
    if (metadataInfo.containsSensitiveData) {
        processSensitiveData(metadataInfo)
    }

    return metadataInfo
}

// Example usage of the scanning function

function collectAllMetadata() {
    let sources = [dataLake, dataWarehouse, operationalDB]
    
    for each source in sources {
        let metadata = scanAndCollectMetadata(source)
        
        // Store or use the metadata as needed
        storeMetadata(metadata)
    }
}
```
This pseudocode illustrates how a scanning function can be implemented to collect and process metadata from different data sources.
??x

---

#### Data Portal and Social Layer
Data catalogs often provide a web interface for users to search, view relationships between datasets, and enhance user interaction through features like Wiki functionality. This social layer allows users to collaborate by sharing information, requesting data, and posting updates.
:p What additional functionalities do data portals and social layers offer in data catalogs?
??x
Data portals in data catalogs provide a web interface where users can search for and view relationships between datasets. Social layers enhance user interaction through features like Wiki functionality. Users can share information, request data from others, and post updates as they become available.

Here is an example of how a social layer might be implemented:

```pseudocode
// Pseudocode for implementing a social layer in a data catalog

class User {
    function searchForData(query) {
        // Logic to search the catalog based on user's query
    }

    function shareInformation(info, datasetName) {
        // Logic to post information related to a specific dataset
    }

    function requestInformation(requestDetails, targetUser) {
        // Logic to send a request for data or information to another user
    }
}

class SocialLayer {
    function displayDataPortal() {
        // Display the web interface with search and view functionalities
    }

    function enableCollaboration(usersList) {
        // Enable collaboration features such as sharing and requesting information
    }
}
```
This pseudocode outlines how a social layer can be integrated into a data catalog, providing users with collaborative tools.
??x

