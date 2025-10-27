# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 23)

**Starting Chapter:** Criterion 5 Single Versus Multi-Cloud

---

#### Hybrid Cloud Strategy
Background context explaining hybrid cloud strategy, including cost and operational complexity. A company can build a hybrid cloud using a combination of a VPC (Virtual Private Cloud) and public cloud services where the vendor maintains the underlying infrastructure, reducing maintenance costs. This approach allows companies to leverage the flexibility and security of both private and public clouds.

:p What are the main disadvantages of implementing a hybrid cloud strategy?
??x
The main disadvantages include increased cost due to building and maintaining both private and public cloud components, as well as additional operational complexity from coordinating and orchestrating two different environments. This coordination is necessary for seamless data flow and service integration between the private and public clouds.
x??

---

#### Multi-Cloud Strategy
Background context explaining multi-cloud strategy, where an organization uses services from multiple cloud providers to address specific business needs and mitigate vendor lock-in risks. While starting with a single cloud provider is often simpler due to similar offerings like virtual machines or managed databases, using multiple providers can offer more flexibility and potentially lower costs.

:p What are the main advantages of adopting a multi-cloud strategy?
??x
The main advantages include enhanced flexibility in adapting to changes introduced by different providers, better cost management through choosing services based on specific needs (e.g., data-intensive queries vs. time-consuming queries), and reduced risk of vendor lock-in as you can switch providers if needed. This approach also allows leveraging the best features from each provider.
x??

---

#### Cost Considerations in Multi-Cloud Strategy
Background context explaining how cloud pricing models differ among providers, affecting cost predictions and management. For example, Google’s BigQuery charges based on bytes fetched, while Snowflake charges based on query duration. Careful consideration of these factors is essential to avoid unexpected costs.

:p How can multi-cloud strategy be more economical for certain workloads?
??x
Multi-cloud can be more economical when the specific needs of a workload align with different pricing models offered by various cloud providers. For instance, if you have data-intensive but fast queries, Snowflake might offer lower costs due to its per-duration charging model; conversely, BigQuery could be cheaper for time-consuming queries that fetch less data.
x??

---

#### Vendor Lock-In Risk
Background context explaining the concept of vendor lock-in and its implications. This occurs when switching from one cloud provider to another is costly or impractical, potentially limiting a company’s flexibility and adaptability.

:p What is vendor lock-in, and why is it disadvantageous?
??x
Vendor lock-in refers to situations where the cost and effort required to switch from one cloud provider to another are so high that the client becomes essentially stuck with their current provider. This can be disadvantageous if the provider makes changes that increase costs or introduces downtime issues.
x??

---

#### Case Study: Wells Fargo Multi-Cloud Strategy
Background context on how Wells Fargo implemented a multi-cloud strategy by leveraging both Microsoft Azure and Google Cloud for different workloads, ensuring secure and trusted environments while maintaining flexibility.

:p How did Wells Fargo implement its multi-cloud strategy?
??x
Wells Fargo implemented its multi-cloud strategy by using Microsoft Azure as the primary foundation for most day-to-day data and analytical needs, empowering employee collaboration. Additionally, it leveraged Google Cloud for advanced data analytics, artificial intelligence, and personalized customer solutions.
x??

---

#### Single Versus Multi-Cloud Strategy
Background context explaining single versus multi-cloud strategies, highlighting that while a single provider might offer simpler management and similar services, using multiple providers can provide more flexibility and cost benefits.

:p What are the main differences between a single cloud provider and a multi-cloud strategy?
??x
A single cloud provider offers simplicity in management but may limit flexibility and choice. In contrast, a multi-cloud strategy allows for leveraging different providers' strengths to meet specific business needs, providing better cost management and mitigating vendor lock-in risks.
x??

---

#### Cloud Cost Benchmarking
Background context on the importance of carefully considering all costs involved in purchasing cloud services, including managed database services. Not all reported prices are reflective of total costs, as factors like scaling needs can significantly impact overall expenses.

:p Why is it important to consider all costs when choosing a cloud service?
??x
It's crucial to consider all costs associated with a cloud service because the single-unit on-demand price might not fully reflect the total cost. Factors such as predictable vs. unpredictable usage, data transfer rates, and scaling needs can significantly impact overall expenses.
x??

---

#### Cloud-Based Product Features
Background context explaining how some cloud-based products offer unique features that make them more appealing to certain clients. For instance, Snowflake’s data warehousing solution allows users to choose the underlying cloud platform independently.

:p How does Snowflake’s data warehousing model provide flexibility for its users?
??x
Snowflake provides flexibility by separating storage and computing, allowing data to be stored and managed independently of compute resources. Users can choose the cloud platform (Azure, AWS, or Google) on which to deploy the Snowflake service, ensuring that their data remains within their preferred infrastructure.
x??

---

#### Hybrid Multi-Cloud Architecture
Wells Fargo's digital strategy is built on a hybrid multi-cloud architecture that leverages both public cloud infrastructures and third-party-owned data centers. This approach also includes private cloud and traditional hosting services to ensure security, reliability, and flexibility. The hybrid model allows for better performance, enhanced security, cost optimization, scalability, and innovation.
:p What does Wells Fargo’s hybrid multi-cloud architecture involve?
??x
This architecture involves combining public clouds with third-party-owned data centers, using both private cloud and traditional hosting services to create a flexible digital foundation that meets business demands in terms of performance, security, cost, scalability, and innovation.
x??

---

#### Monolithic Architecture
A monolithic architecture organizes codebases into a single location where all components are tightly coupled. This means making changes typically requires redeploying the entire application rather than just specific parts. It provides ease of deployment, high cohesion, simpler development workflow, easier monitoring and testing, higher throughput, and simplified code reusability.
:p What is a monolithic architecture?
??x
A monolithic architecture organizes all components in a single location with tight coupling between them. Making changes usually requires redeploying the entire application. This architecture offers ease of deployment, high cohesion, simpler development workflow, easier monitoring and testing, higher throughput, and simplified code reusability.
x??

---

#### Modular Architecture
Modular or microservices architecture splits applications into smaller modules that can be developed, deployed, tested, and scaled independently. Each module is owned by a different team managing its resource and feature requirements separately. Microservices are small, autonomous applications working together to achieve common goals.
:p What is modular or microservices architecture?
??x
Modular or microservices architecture splits an application into smaller modules that can be developed, deployed, tested, and scaled independently. Each module is owned by a different team managing its resource and feature requirements separately. Microservices are small, autonomous applications working together to achieve common goals.
x??

---

#### Monolithic Architecture Advantages
Monolith architectures provide several advantages including ease of deployment, high cohesion and consistency, simpler development workflow, easier monitoring, testing (end-to-end), higher throughput and performance, and simplified code reusability. However, they can become very complex and hard to predict as the application scales.
:p What are some advantages of monolithic architecture?
??x
Monolith architectures offer several advantages: ease of deployment, high cohesion and consistency, simpler development workflow, easier monitoring, end-to-end testing, higher throughput and performance, and simplified code reusability. However, they can become very complex and hard to predict as the application scales.
x??

---

#### Monolithic Architecture Pitfalls
Monolith architectures can lead to a highly complex and hard-to-predict application codebase, making it challenging to scale, extend, understand, and debug. As the number of components and dependencies increases, understanding the impact of local changes on the system’s overall behavior becomes increasingly difficult.
:p What are some challenges associated with monolithic architecture?
??x
Monolith architectures can lead to a highly complex and hard-to-predict application codebase, making it challenging to scale, extend, understand, and debug. As the number of components and dependencies increases, understanding the impact of local changes on the system’s overall behavior becomes increasingly difficult.
x??

---

#### Amazon Prime Video Case Study
Amazon Prime Video decided to switch back to a monolithic architecture from a distributed microservices architecture in 2023. This move helped achieve higher scalability and resilience while reducing infrastructure costs by 90%.
:p What did Amazon Prime Video do with its architecture?
??x
In 2023, Amazon Prime Video switched back to a monolithic architecture from a distributed microservices architecture. This move helped achieve higher scalability and resilience while reducing infrastructure costs by 90%.
x??

---

#### Monolith to Microservices Refactoring
Refactoring a complex monolith codebase into microservices can be necessary when the application reaches its limits in terms of scalability and performance. Best practices exist for this process, including using software architecture evaluation techniques to assess quality and reliability requirements.
:p What is involved in refactoring a monolithic architecture to microservices?
??x
Refactoring a complex monolith codebase into microservices can be necessary when the application reaches its limits in terms of scalability and performance. Best practices include using software architecture evaluation techniques to assess quality and reliability requirements, tools such as the Modularity Maturity Index (MMI), fitness functions, and software metrics to determine which components need refactoring, replacement, or retention.
x??

---

#### Financial Data Engineering Lifecycle Overview
This section provides an overview of how financial data is managed through a structured lifecycle. It covers key phases such as ingestion, storage, transformation and delivery, and monitoring.
:p What does the financial data engineering lifecycle include?
??x
The financial data engineering lifecycle includes ingestion (collecting and processing raw data), storage (maintaining large volumes of historical and current data), transformation and delivery (cleaning, formatting, and delivering data to various stakeholders), and monitoring (ensuring data quality and system performance).
??x

---

#### Criteria for Evaluating Technological Alternatives
The chapter outlines six criteria that financial institutions can use to evaluate technological alternatives for the Financial Data Engineering Lifecycle (FDEL) stack. These criteria help ensure that chosen technologies meet specific needs.
:p What are the six criteria mentioned in the text?
??x
The six criteria include:
1. Performance: Ensuring data can be processed and delivered efficiently.
2. Scalability: Ability to handle increasing amounts of data without performance degradation.
3. Cost-effectiveness: Balancing cost with functionality and maintenance.
4. Security: Protecting sensitive financial data from unauthorized access.
5. Compliance: Adhering to regulatory requirements for data handling.
6. Integration: Ease of integration with existing systems and technologies.
??x

---

#### Ingestion Layer Overview
The ingestion layer focuses on collecting and processing raw financial data, preparing it for further use in the FDEL stack. This involves various sources like APIs, databases, and files.
:p What is the primary focus of the ingestion layer?
??x
The primary focus of the ingestion layer is to collect and process raw financial data from diverse sources such as APIs, databases, and file systems, ensuring it meets the necessary quality standards for further processing.
??x

---

#### Storage Layer Overview
This section covers how financial data is stored efficiently. It involves selecting appropriate storage technologies that can handle large volumes of historical and current data while maintaining performance.
:p What does the storage layer address?
??x
The storage layer addresses how to store large volumes of historical and current financial data in a way that ensures high performance, scalability, and reliability. It focuses on choosing the right storage technology to support the ingestion and subsequent layers of the FDEL stack.
??x

---

#### Transformation and Delivery Layer Overview
This layer involves cleaning, formatting, and delivering the processed data to various stakeholders such as analysts, traders, and regulatory bodies. This ensures that the data is in a usable format for downstream processes.
:p What does the transformation and delivery layer do?
??x
The transformation and delivery layer cleans, formats, and delivers the processed financial data to various stakeholders like analysts, traders, and regulatory bodies, ensuring it is in a usable and standardized format for further analysis or reporting purposes.
??x

---

#### Monitoring Layer Overview
The monitoring layer ensures that the FDEL stack operates smoothly by providing real-time insights into system performance and data quality. This helps in identifying issues early and maintaining high standards of service.
:p What does the monitoring layer ensure?
??x
The monitoring layer ensures that the Financial Data Engineering Lifecycle (FDEL) stack operates smoothly by providing real-time insights into system performance and data quality, helping to identify issues early and maintain high standards of service.
??x

---

#### Detailed Coverage in Subsequent Chapters
The next four chapters will delve deeper into each layer of the FDEL: ingestion, storage, transformation and delivery, and monitoring. Each chapter will provide more technical details on specific aspects of these layers.
:p What is covered in the subsequent four chapters?
??x
In the next four chapters, detailed coverage will be provided for each layer of the Financial Data Engineering Lifecycle (FDEL): 
- Chapter 7: Ingestion Layer
- Chapter 8: Storage Layer
- Chapter 9: Transformation and Delivery Layer
- Chapter 10: Monitoring Layer
Each chapter will offer more technical details on specific aspects of these layers.
??x

---

