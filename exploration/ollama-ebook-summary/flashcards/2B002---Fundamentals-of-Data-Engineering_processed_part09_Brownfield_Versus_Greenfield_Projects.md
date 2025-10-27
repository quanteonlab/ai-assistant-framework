# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 9)

**Starting Chapter:** Brownfield Versus Greenfield Projects

---

#### Tight vs Loose Coupling in Data Architecture
Background context: The concepts of tight versus loose coupling originated from software development. These principles have been around for over 20 years, but are now being applied to data architecture. In traditional architectures, data is often monolithic and tightly coupled, which can limit flexibility and scalability.

:p What does the term "tight vs loose coupling" refer to in the context of data architecture?
??x
In the context of data architecture, tight coupling refers to systems where components are closely interconnected, making changes or updates difficult without affecting other parts. Loose coupling means that components are designed to interact through well-defined interfaces, allowing for more flexibility and easier maintenance.

For example:
- A monolithic data warehouse where all data processing is done in one place can be tightly coupled.
- A microservices architecture with separate sales-specific and inventory-specific data pipelines would be an example of loose coupling.

```java
// Example of tight coupling (bad)
public class DataProcessor {
    private DataWarehouse warehouse;
    
    public void process() {
        // Process data directly from the same shared resource
        warehouse.updateData();
    }
}

// Example of loose coupling (good)
public interface DataStore {
    void updateData();
}

class SalesDataProcessor {
    private final DataStore salesWarehouse;

    public SalesDataProcessor(DataStore salesWarehouse) {
        this.salesWarehouse = salesWarehouse;
    }

    public void processSalesData() {
        // Process data from a specialized source
        salesWarehouse.updateData();
    }
}
```
x??

---

#### Monolithic vs Microservices in Data Architecture
Background context: Moving towards microservices in the context of data involves breaking down large, monolithic systems into smaller, more manageable and independent services. This can lead to better scalability, maintainability, and flexibility.

:p What is a common approach for transitioning from a monolithic data architecture to a more flexible design?
??x
A common approach is to introduce domain-specific microservices, each handling specific aspects of the business logic. For example, instead of having one central data warehouse that processes all types of data, you could have separate data pipelines and warehouses for sales, inventory, and product domains.

For instance:
- A "SalesDataPipeline" connects directly to a "SalesDataWarehouse."
- An "InventoryDataPipeline" connects to an "InventoryDataWarehouse."

```java
// Example of microservices architecture in Java
public class SalesDataPipeline {
    private final SalesDataSource source;
    private final SalesDataSink sink;

    public SalesDataPipeline(SalesDataSource source, SalesDataSink sink) {
        this.source = source;
        this.sink = sink;
    }

    public void process() {
        // Process sales data and update the warehouse
        List<SalesRecord> records = source.getAllRecords();
        sink.updateDatabase(records);
    }
}
```
x??

---

#### Multitier Architecture in Data Design
Background context: A multitier architecture separates concerns into layers, often to manage complexity better. It addresses technical challenges like shared resources but does not inherently solve the problem of domain sharing across different teams.

:p What is a multitier architecture and why might it be used in data design?
??x
A multitier architecture in data design refers to separating concerns into distinct layers that communicate with each other through well-defined interfaces. This approach can help manage complexity, security, and scalability by isolating different aspects of the system.

For example:
- Presentation layer (GUI or API)
- Business logic layer (data processing)
- Data storage layer

```java
// Example of a simple multitier architecture in Java
public class DataLayer {
    private final PersistenceProvider provider;

    public DataLayer(PersistenceProvider provider) {
        this.provider = provider;
    }

    public List<SalesRecord> getAllRecords() {
        return provider.loadAllSalesRecords();
    }
}

public interface PersistenceProvider {
    List<SalesRecord> loadAllSalesRecords();
}
```
x??

---

#### Centralization vs Decentralization in Data Architecture
Background context: In data architecture, there are two primary approaches to managing and sharing data across different teams or domains—centralization (one team manages all the data) and decentralization (each domain prepares its own data).

:p What is centralization in data architecture?
??x
Centralization in data architecture refers to a model where one dedicated team collects and manages all data from various sources, ensuring consistency and control. This approach simplifies access and management but can create bottlenecks and single points of failure.

For example:
- A single "DataCentral" team gathers sales, inventory, and product data into a unified warehouse.
```java
// Example of centralization in Java
public class DataCentral {
    private final SalesDataSource salesSource;
    private final InventoryDataSource inventorySource;
    
    public DataCentral(SalesDataSource salesSource, InventoryDataSource inventorySource) {
        this.salesSource = salesSource;
        this.inventorySource = inventorySource;
    }

    public void reconcileData() {
        List<SalesRecord> salesRecords = salesSource.getAllRecords();
        List<InventoryRecord> inventoryRecords = inventorySource.getAllRecords();
        // Reconcile and update the central warehouse
        CentralWarehouse.update(salesRecords, inventoryRecords);
    }
}
```
x??

---

#### Data Mesh Architecture
Background context: A data mesh is a decentralized approach to managing data in an organization where each team prepares its own domain-specific datasets for consumption by other teams. This approach promotes self-service and autonomy.

:p What is the Data Mesh architecture?
??x
The Data Mesh architecture is a decentralized method of organizing data governance, where each software team is responsible for preparing its specific domain's data for use across the organization. It aims to promote self-service, autonomy, and better alignment with business needs by enabling teams to own their data processing and delivery.

In a data mesh:
- Each team owns its data and processes it according to defined standards.
- Teams are responsible for making their data accessible and usable by other parts of the organization.

For example:
- The SalesTeam prepares sales-specific datasets, which can be consumed by other teams like Marketing or Finance.

```java
// Example of a Data Mesh in Java
public interface DataProvider<T> {
    T getData();
}

class SalesDataProvider implements DataProvider<SalesRecord> {
    public List<SalesRecord> getData() {
        // Fetch and process sales data
        return fetchSalesDataFromDatabase();
    }
}

class MarketingTeam {
    private final DataProvider<SalesRecord> provider;

    public MarketingTeam(DataProvider<SalesRecord> provider) {
        this.provider = provider;
    }

    public void analyzeSales() {
        List<SalesRecord> records = provider.getData();
        // Analyze sales data
    }
}
```
x??

---

#### Multitenancy Considerations
Background context: In large organizations, multitenancy is a common practice where different departments or customers share resources. The main factors to consider are performance and security. Performance issues can arise due to resource contention (noisy neighbor problem), while security requires proper data isolation to prevent unauthorized access.
:p What are the primary factors to consider in multitenancy?
??x
The primary factors to consider in multitenancy are:
- **Performance**: Ensure consistent system performance for all tenants, avoiding noisy neighbor problems where one tenant's high usage degrades other tenants' performance.
- **Security**: Maintain data isolation and prevent unauthorized access from different tenants.

This involves strategies like using views for data isolation but ensuring they do not leak sensitive information. 
x??

---

#### Event-Driven Workflow
Background context: An event-driven workflow is a system where events (such as new orders or updates) are produced, routed, and consumed by various services asynchronously. This approach helps in managing the state of an event across multiple services, making the architecture more robust.
:p What does an event-driven workflow encompass?
??x
An event-driven workflow encompasses:
- **Event Production**: Creating events such as a new order or update to an existing order.
- **Routing**: Routing these events to appropriate consumers without tight coupling among producers and consumers.
- **Consumption**: Services that consume the events, processing them asynchronously.

This allows for loose coupling between services and better management of event states. 
x??

---

#### Event-Driven Architecture
Background context: An event-driven architecture (EDA) uses the principles of an event-driven workflow to distribute state changes across multiple services. This architecture is beneficial when services need to handle events independently, such as in distributed systems or during failures.
:p What is an event-driven architecture?
??x
An event-driven architecture (EDA) is a system design where:
- Events are produced and consumed by different services asynchronously.
- The state of the system is managed across multiple services.

This architecture helps in handling failures gracefully, managing state changes independently, and ensuring loose coupling between services. 
x??

---

#### Brownfield vs Greenfield Projects
Background context: Before designing a data architecture project, it's crucial to understand whether you are starting from scratch (greenfield) or working with an existing architecture that needs redesigning (brownfield). This distinction affects the initial planning and implementation approach.
:p What is the difference between brownfield and greenfield projects?
??x
- **Brownfield Projects**: Involves redesigning and improving an existing system. Initial planning must account for the current structure and functionalities.
- **Greenfield Projects**: Starting from a clean slate with no pre-existing infrastructure or codebase. The initial design can be more flexible, but it requires thorough planning to ensure all necessary features are included.

Understanding this distinction helps in setting realistic expectations and planning accordingly. 
x??

---

#### Brownfield Projects
Background context explaining brownfield projects, including their constraints and approach. Often involve refactoring an existing architecture with limited flexibility due to past choices.
:p What are the main characteristics of a brownfield project?
??x
Brownfield projects typically involve reorganizing and refactoring an existing system, constrained by decisions made in the present and past. The goal is to transition to new business and technical objectives while understanding and respecting the legacy architecture's complexities.
```
public class BrownfieldExample {
    public void refactorArchitecture() {
        // Logic to gradually change parts of the old architecture
    }
}
```
x??

---

#### Strangler Pattern
Explanation of the strangler pattern, a popular approach for brownfield projects. Involves incrementally replacing legacy components with new systems.
:p What is the strangler pattern in the context of brownfield projects?
??x
The strangler pattern is an approach where new systems are incrementally introduced to replace parts of an existing architecture over time. This allows gradual replacement while maintaining flexibility and reversibility.
```
public class StranglerExample {
    public void incrementallyReplaceLegacyComponents() {
        // Code to introduce new components that gradually replace old ones
    }
}
```
x??

---

#### Big-Bang Overhaul
Explanation of the big-bang approach, which involves a full rewrite or overhaul of an existing system.
:p What is the big-bang overhaul approach in brownfield projects?
??x
The big-bang overhaul involves a complete and immediate replacement of the old architecture. While popular due to its simplicity, it can be risky due to lack of planning and potential for irreversible decisions.
```
public class BigBangExample {
    public void rewriteArchitecture() {
        // Code for a full rewrite without incremental steps
    }
}
```
x??

---

#### Greenfield Projects
Explanation of greenfield projects, which start with a clean slate and no legacy constraints. Often easier but can lead to shiny object syndrome.
:p What are the main characteristics of greenfield projects?
??x
Greenfield projects allow for a fresh start without the constraints of existing architectures. While they offer opportunities to use modern tools and patterns, teams might fall into "shiny object syndrome" by overusing new technologies.
```
public class GreenfieldExample {
    public void pioneerNewArchitecture() {
        // Code for starting with a clean slate
    }
}
```
x??

---

#### Shiny Object Syndrome
Explanation of shiny object syndrome in the context of greenfield projects. Teams might get excited and focus on using the latest technologies without considering project requirements.
:p What is "shiny object syndrome"?
??x
Shiny object syndrome refers to a tendency for teams to prioritize the adoption of new, exciting technologies over fulfilling the actual requirements of the project. This can lead to distractions from core goals.
```
public class ShinyObjectExample {
    public void avoidShinyObjects() {
        // Code or logic to ensure focus on requirements first
    }
}
```
x??

---

#### Resume-Driven Development
Explanation of resume-driven development, where teams prioritize impressing stakeholders with new technologies over achieving project goals.
:p What is "resume-driven development"?
??x
Resume-driven development occurs when the primary goal becomes showcasing impressive technology stacks rather than meeting the actual objectives and requirements of the project. This can detract from effective project execution.
```
public class ResumeDrivenExample {
    public void prioritizeRequirements() {
        // Code to ensure focus on project goals over tech stack
    }
}
```
x??

---
#### Data Warehouse Definition and History
Data warehouses are central data hubs used for reporting and analysis. They typically store highly formatted and structured data optimized for analytics use cases.

In 1989, Bill Inmon introduced the concept of a data warehouse as "a subject-oriented, integrated, nonvolatile, and time-variant collection of data in support of management’s decisions." This definition still holds relevance today despite advancements in technology.

:p What is a data warehouse according to Bill Inmon's original definition?
??x
A data warehouse is defined by Inmon as "a subject-oriented, integrated, nonvolatile, and time-variant collection of data used for management decision support."

The key characteristics include:
- Subject-oriented: Data organized around business processes or topics.
- Integrated: A unified view of the organization’s data from various sources.
- Nonvolatile: Once loaded, data does not change but may be appended over time.
- Time-variant: Historical data is retained to provide a temporal perspective.

:x??
---
#### Types of Data Warehouse Architectures
There are two main types of data warehouse architectures: organizational and technical. Organizational architecture relates to business team structures and processes, while the technical architecture focuses on the infrastructure used for processing data.

:p What are the two main types of data warehouse architectures mentioned in the text?
??x
The two main types of data warehouse architectures are:
1. Organizational Data Warehouse Architecture: Relates to how a company's business teams structure their data and processes.
2. Technical Data Warehouse Architecture: Refers to the technical infrastructure used for processing data, such as MPP (Massively Parallel Processing) systems.

:x??
---
#### Organizational Data Warehouse Architecture
Organizational architecture organizes data according to specific business team structures and processes. Key characteristics include separating OLAP from production databases and centralizing data traditionally through ETL (Extract, Transform, Load).

:p What are the two main characteristics of organizational data warehouse architecture mentioned in the text?
??x
The two main characteristics of organizational data warehouse architecture are:
1. Separates online analytical processing (OLAP) from production databases (online transaction processing).
2. Centralizes and organizes data traditionally through ETL processes.

Explanation: The separation between OLAP and production databases is critical for improving analytics performance as the business grows, by directing load away from production systems. ETL processes involve extracting data from source systems, transforming it to clean and standardize it, and then loading it into a target database system.
:x??
---
#### Technical Data Warehouse Architecture
Technical architecture includes MPP (Massively Parallel Processing) systems that are optimized for high-performance aggregation and statistical calculations. More recent trends have seen the shift towards columnar storage to handle larger data sets.

:p What does technical data warehouse architecture typically include?
??x
Technical data warehouse architecture typically includes:
- MPP (Massively Parallel Processing) systems: These support SQL semantics similar to relational application databases but are optimized for scanning large amounts of data in parallel.
- Columnar storage: This has become increasingly popular, especially in cloud data warehouses, to facilitate handling larger datasets and complex queries.

:x??
---
#### Extract, Transform, Load (ETL)
ETL processes involve pulling data from source systems, cleaning and standardizing it through transformations, and then loading it into the target database system. The technical MPP architecture often supports this ETL process with powerful computational resources.

:p What does the ETL process consist of?
??x
The ETL process consists of three main steps:
1. Extract: Pulls data from source systems.
2. Transform: Cleans and standardizes data, organizing it according to business logic in a highly modeled form.
3. Load: Pushes data into the target database system.

:p What is the significance of the ETL process in data warehousing?
??x
The ETL process is significant because it ensures that raw data from various sources is transformed and cleaned before being loaded into the data warehouse, making it ready for analysis and reporting.

:x??
---
#### Extract, Load, Transform (ELT)
ELT is an alternative to traditional ETL where transformations are handled directly in the data warehouse. This approach leverages the computational power of cloud data warehouses for batch processing and analytics.

:p What is ELT, and how does it differ from traditional ETL?
??x
ELT stands for Extract, Load, Transform. Unlike traditional ETL, where transformations occur before loading data into the warehouse, in ELT, data gets moved directly to a staging area in raw form, and transformations are handled within the data warehouse.

:p What is an advantage of using ELT over traditional ETL?
??x
An advantage of using ELT over traditional ETL is that it allows leveraging the massive computational power of cloud data warehouses for batch processing and analytics, reducing the need to handle transformations externally before loading the data.
:x??
---
#### Cloud Data Warehouses
Cloud data warehouses represent a significant evolution from on-premises systems. They offer scalable, pay-as-you-go models, allowing companies to manage infrastructure more efficiently as their data grows in complexity.

:p What are cloud data warehouses and what advantages do they offer over traditional on-premises data warehouses?
??x
Cloud data warehouses are advanced data warehouse architectures that provide scalable, pay-as-you-go services. They offer several advantages:
- On-demand scalability: Companies can spin up clusters as needed.
- Cost-effectiveness: No upfront investment required for hardware and maintenance.
- Flexibility: Ability to easily scale computing resources based on demand.

:x??
---

#### Cloud Data Warehouses and Their Evolution
Background context: The text discusses how cloud data warehouses are evolving beyond traditional MPP (Massively Parallel Processing) systems to offer a more comprehensive platform for data processing, analytics, and reporting. It highlights the transition from traditional terms like "data warehouse" to a new term that encompasses broader capabilities.
:p What is the significance of cloud data warehouses in modern data management?
??x
Cloud data warehouses are evolving into a new data platform with much broader capabilities than those offered by a traditional MPP system. They serve as central repositories for organizational data, providing advanced analytics and reporting functionalities. This evolution addresses the limitations of traditional on-premises data warehouses and leverages cloud infrastructure to enhance scalability, performance, and cost-effectiveness.
x??

---

#### Data Marts
Background context: Data marts are a more refined subset of a warehouse designed to serve analytics and reporting for specific departments or suborganizations within an enterprise. They exist as part of the broader organizational data strategy but focus on delivering tailored data insights to particular users or teams.
:p What is the purpose of using data marts in data architecture?
??x
Data marts are used to make data more easily accessible to analysts and report developers, focusing on specific departments or suborganizations within an enterprise. They provide a stage of transformation beyond the initial ETL (Extract, Transform, Load) pipelines, which can significantly improve performance for complex queries that require joining and aggregating large amounts of raw data.
x??

---

#### Data Lake 1.0
Background context: The data lake is a storage system where all types of data—structured and unstructured—are stored in their native format without stringent schema constraints. It aims to democratize access to vast quantities of data for various analytics purposes, but it faces significant challenges related to management and governance.
:p What was the primary goal of data lake 1.0?
??x
The primary goal of data lake 1.0 was to create a central repository that could store all types of structured and unstructured data without strict schema constraints, making it accessible for various analytics purposes and democratizing access to this data within an organization.
x??

---

#### Challenges with Data Lake 1.0
Background context: While the initial vision of the data lake promised unlimited storage and processing power, it faced several challenges such as lack of schema management, data cataloging tools, and difficulty in processing complex queries without transforming the data first. Additionally, regulatory requirements like GDPR posed new challenges.
:p What were some of the main issues with implementing data lakes?
??x
Some of the main issues with implementing data lakes included a lack of schema management, poor data cataloging tools, and difficulties in processing complex queries without first transforming the data. These challenges led to terms such as "data swamp," "dark data," and WORN (Where Once Was a Repository) being coined, highlighting the unmanageable state of many early data lake projects.
x??

---

#### Data Marts Workflow
Background context: The text describes the general workflow for using ETL or ELT pipelines to populate data marts. These steps involve extracting, transforming, and loading data into a more refined subset of a warehouse designed for specific departments or suborganizations within an enterprise.
:p What is the general workflow for ETL/ELT plus data marts?
??x
The general workflow involves using ETL (Extract, Transform, Load) or ELT (Extract, Load, Transform) processes to populate data marts. This process extracts raw data from various sources, transforms it as needed, and loads it into a more refined subset of the warehouse that is specific to the needs of particular departments or suborganizations.
x??

---

#### Data Lakes and Their Challenges
Background context: The initial data lakes, while promising flexible storage of large volumes of data, faced significant challenges. These included difficulty in managing deletions or updates to rows (DML operations), high costs associated with managing Hadoop clusters, and a steep learning curve for users.
:p What were the main challenges faced by first-generation data lakes?
??x
The main challenges faced by first-generation data lakes included:
- Difficulty implementing DML operations due to the need to create new tables entirely.
- High costs due to the complexities of managing Hadoop clusters, which required large teams at high salaries.
- A steep learning curve and the complexity of using raw Apache code without vendor support.

x??

---
#### Data Lakehouse Concept
Background context: To address the limitations of first-generation data lakes, a new concept called "data lakehouse" was introduced. This combines the benefits of both data lakes and data warehouses, offering better control and management while retaining the flexibility of object storage.
:p How does a data lakehouse differ from traditional data lakes?
??x
A data lakehouse differs from traditional data lakes by supporting:
- Atomicity, consistency, isolation, and durability (ACID) transactions, allowing updates and deletions to rows.
- Incorporation of controls, data management, and data structures found in data warehouses.

Example code showing a basic transformation logic using pseudocode:
```pseudocode
function processRecords(records) {
    for each record in records {
        // Update or delete logic here
        if (record.meetsCondition()) {
            updateRecord(record);
        } else {
            deleteRecord(record);
        }
    }
}
```
x??

---
#### Cloud Data Warehouse Architecture
Background context: The evolution of cloud data warehouse architectures has made them increasingly similar to the structure of first-generation data lakes. These systems now support petabyte-scale queries, store various unstructured and semistructured data types, and integrate with advanced processing technologies.
:p How does a modern cloud data warehouse architecture differ from traditional on-premises solutions?
??x
A modern cloud data warehouse differs from traditional on-premises solutions by:
- Separating compute from storage.
- Supporting petabyte-scale queries.
- Storing various unstructured and semistructured data types.
- Integrating with advanced processing technologies like Spark or Beam.

Example code showing a query execution in a cloud data warehouse using pseudocode:
```pseudocode
function executeQuery(query) {
    // Connect to the cloud data warehouse
    connectToCloudWarehouse();
    
    // Execute the query and return results
    results = runQuery(query);
    
    // Process the results
    processResults(results);
}
```
x??

---
#### Convergence of Data Lakes and Warehouses
Background context: The trend towards convergence aims to fully realize the promise of data lakes by incorporating controls, management, and structure from data warehouses. This approach combines the flexibility of a data lake with the benefits of structured data management.
:p What is the primary goal of the convergence between data lakes and data warehouses?
??x
The primary goal of the convergence between data lakes and data warehouses is to:
- Combine the flexibility of storing large volumes of raw, unstructured data from a data lake.
- Integrate robust controls, data management, and structured data support as found in traditional data warehouses.

Example code showing how a cloud-based data lakehouse might be implemented using pseudocode:
```pseudocode
function implementDataLakehouse(sourceData) {
    // Store data in object storage (like an S3 bucket)
    storeInObjectStorage(sourceData);
    
    // Use Spark to process the data for transformation and analysis
    transformedData = applyTransformationsUsingSpark(sourceData);
    
    // Load processed data into a cloud data warehouse
    loadIntoCloudWarehouse(transformedData);
}
```
x??

---

---
#### Converged Data Platform
Background context: The text discusses how data lakes and data warehouses are converging into a single, unified architecture called a converged data platform. This platform combines the capabilities of both environments to offer users greater flexibility and seamless integration between structured and unstructured data.

Vendors like AWS, Azure, Google Cloud, Snowflake, and Databricks are leading in offering these platforms with tightly integrated tools for various data processing needs. Future data engineers will have more options to choose from based on vendor offerings, ecosystem support, and openness.
:p What is a converged data platform?
??x
A converged data platform combines the functionalities of traditional data lakes and warehouses into one unified architecture, providing both structured and unstructured data capabilities in an integrated manner. It aims to offer greater flexibility for data engineers while reducing complexity.
x??

---
#### Modern Data Stack
Background context: The modern data stack is a trendy analytics architecture that focuses on using cloud-based, modular tools to create a flexible and cost-effective data infrastructure. This approach contrasts with traditional monolithic toolsets by emphasizing simplicity, modularity, and ease of use.

The components include data pipelines, storage, transformation, data management/governance, monitoring, visualization, and exploration. The modern data stack is characterized by self-service analytics, agile data management, and the use of open source tools or simple proprietary tools with clear pricing structures.
:p What distinguishes the modern data stack from traditional architectures?
??x
The modern data stack stands out through its modular design, cloud-based components, and emphasis on simplicity. Unlike monolithic toolsets, it offers a combination of easy-to-use off-the-shelf tools that can be assembled in a plug-and-play manner to create a customized data architecture.
x??

---
#### Lambda Architecture
Background context: The Lambda architecture was developed as an early solution for handling both batch and streaming data processing needs. It is based on the idea of running separate systems for real-time (streaming) and historical (batch) data, while aggregating their outputs in a combined view.

In this architecture, the source system sends data to two destinations: a stream processor that provides low-latency responses, and a batch processor that generates precomputed views. The serving layer then combines these processed results for comprehensive analytics.
:p What is the Lambda architecture used for?
??x
The Lambda architecture is designed to handle both real-time and historical data processing requirements efficiently. It separates the data flow into two main paths: stream processing for low-latency queries and batch processing for aggregating and transforming data over time, then combines these results for comprehensive analytics.
x??

---

