# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 19)

**Starting Chapter:** Whom Youll Work With

---

#### Stream Partitioning
Stream partitioning is a technique used to distribute messages across different partitions based on a chosen partition key. This ensures that related messages are processed together, which can be crucial for maintaining data integrity and performance in distributed systems. The process involves dividing the message ID (or any other chosen key) by the number of partitions and taking the remainder as the partition identifier.

For example:
- If you have 3 partitions and a message with an ID that gives a remainder of 0 when divided by 3, it will be assigned to partition 0.
- Messages that give remainders of 1 and 2 upon division by 3 would be placed in partitions 1 and 2 respectively.

:p What is stream partitioning?
??x
Stream partitioning is a method for distributing messages across multiple partitions based on a chosen key, ensuring related data are processed together. This technique helps manage load distribution and maintain consistency in distributed systems.
x??

---

#### Partition Key Selection
Choosing an appropriate partition key is crucial to avoid hotspotting (where one or few partitions receive disproportionately more traffic). For instance, using a device ID as the partition key for IoT applications ensures that all messages from a particular device are processed by the same server. However, if the distribution of devices across regions is uneven, such as California, Texas, Florida, and New York having significantly more devices than others, the partitions associated with these states might become overloaded.

:p How does one avoid hotspotting in stream partitioning?
??x
To avoid hotspotting, carefully select a partition key that evenly distributes messages. For IoT applications, using device IDs can ensure related data are processed together. However, if geographical distribution is uneven, consider alternative keys to balance the load more effectively.
x??

---

#### Fault Tolerance and Resilience in Event-Streaming Platforms
Event-streaming platforms provide fault tolerance by storing streams across multiple nodes. If a node fails, another takes over without affecting accessibility or data integrity. This ensures that records are not lost and can be reliably ingested and processed even when failures occur.

:p What is the role of fault tolerance in event-streaming platforms?
??x
Fault tolerance in event-streaming platforms ensures that streams remain accessible even if nodes fail. Data ingestion and processing continue without interruption, maintaining reliability and availability.
x??

---

#### Working with Stakeholders in Source Systems
Understanding stakeholders involved in source systems is vital for successful data engineering projects. Typically, two categories of stakeholders are encountered: system stakeholders who build and maintain the source systems (e.g., software engineers), and data stakeholders who own the data (usually IT or data governance groups).

:p Who are the key stakeholders in source systems?
??x
Key stakeholders in source systems include system stakeholders who manage and develop the source systems, and data stakeholders who control access to the data. These roles can sometimes overlap.
x??

---

#### Data Contracts
A data contract is a written agreement between the owner of a source system and the team ingesting the data for use in a data pipeline. It specifies the data being extracted, the method (full or incremental), frequency, and contact details for both parties.

:p What is a data contract?
??x
A data contract is a formal agreement that outlines what data will be extracted from a source system, how often it will be updated, and provides contact information for both the source system owner and the data ingestion team.
x??

---

#### Feedback Loop with Stakeholders
Establishing a feedback loop between data engineers and stakeholders of source systems helps in understanding how data is consumed and used. This ensures that changes or issues in upstream sources are promptly addressed.

:p Why is a feedback loop important for data engineers?
??x
A feedback loop is crucial as it enables data engineers to be aware of any changes or issues in the upstream source data, ensuring they can adapt their systems accordingly.
x??

---

#### Understanding SLAs and SLOs
Background context: Service-Level Agreements (SLAs) are contracts between a provider and customer that define service expectations, including uptime, response times, and other performance metrics. A Service-Level Objective (SLO) measures how well these agreements are met. Establishing clear SLAs and SLOs ensures reliability and quality of data.

:p What is the difference between an SLA and an SLO?
??x
An SLA defines what you can expect from source systems, such as reliable availability and high-quality data. An SLO measures performance against these expectations, like 99 percent uptime for data sources.
x??

---

#### Verbal Setting of Expectations
Background context: When formal SLAs or SLOs seem too rigid, verbal agreements can still be effective in setting expectations with upstream providers about key requirements such as uptime and data quality.

:p How can a data engineer set informal expectations with source system owners?
??x
A data engineer should verbally communicate their needs regarding uptime, data quality, and other critical metrics to the stakeholders of source systems. This verbal agreement ensures that both parties understand each other's requirements.
x??

---

#### Impact of Undercurrents on Source Systems
Background context: Undercurrents refer to underlying factors or practices in source systems (e.g., security, architecture) that can significantly influence data engineering efforts. These undercurrents are often outside the direct control of a data engineer.

:p What does "undercurrents" mean in the context of source systems?
??x
Undercurrents in source systems refer to implicit assumptions about best practices, such as data security, architecture, and DevOps principles that affect how data is generated. These factors can impact data reliability and quality.
x??

---

#### Security Considerations for Source Systems
Background context: Ensuring the security of data within source systems is critical. This includes measures like encryption, secure access methods, and secure handling of credentials.

:p What are some key security considerations when accessing a source system?
??x
Key security considerations include:
- Data being securely encrypted both at rest and in transit.
- Accessing the source system via a virtual private network (VPN) or over a public internet.
- Storing passwords, tokens, and other credentials securely using tools like key managers or password managers.
- Verifying the legitimacy of the source system to prevent malicious data ingestion.

Example code for secure SSH key management:
```python
import getpass

def manage_ssh_keys():
    # Prompt user for password without echoing
    ssh_key_password = getpass.getpass("Enter your SSH key password: ")
    
    # Use a tool like ssh-agent or a key manager to securely store the key
    # Example using an assumed secure method:
    print(f"Your SSH key is now managed securely with {ssh_key_password}.")
```
x??

---

---

#### Data Governance
Understanding how data is managed in source systems is crucial for effective data engineering. It involves knowing who manages the data and whether it's governed reliably.
:p What are the primary concerns related to data governance?
??x
The primary concerns related to data governance include understanding who manages the data, ensuring reliable management practices, and making sure that data is organized and accessible in a clear manner.

---

#### Data Quality
Ensuring high-quality data in upstream systems requires collaboration with source system teams. Setting expectations on data quality involves working closely with these teams.
:p How should you ensure data quality and integrity in upstream systems?
??x
You should work with source system teams to set expectations on data and communication. This includes establishing clear criteria for data quality, such as accuracy, completeness, and consistency. Regularly reviewing the quality of incoming data can help maintain high standards.

---

#### Schema Management
Schema changes are common and must be anticipated by data engineers. Collaboration with source system teams is essential to stay informed about impending schema changes.
:p How should you manage schema changes in upstream systems?
??x
You should expect that upstream schemas will change over time. To manage this, collaborate with source system teams to be notified of upcoming schema changes. This allows for timely adjustments in your data ingestion and transformation processes.

---

#### Master Data Management
Master data management (MDM) practices control the creation of records across systems. Understanding MDM is important when dealing with complex datasets.
:p How does master data management impact upstream records?
??x
Master data management practices or systems ensure that records are created consistently across multiple systems. This impacts how you handle data in your pipelines, as you need to align with these practices to maintain data integrity and avoid duplication.

---

#### Privacy and Ethics
Accessing raw data may come with privacy and ethical considerations. Understanding the implications of source data is crucial for compliance.
:p What should you consider regarding privacy and ethics?
??x
You should understand whether access to raw data is available, how data will be obfuscated, and the retention policies for the data. Additionally, consider regulatory requirements and internal policies that may impact how you handle the data.

---

#### DataOps Considerations
DataOps involves ensuring operational excellence across the entire stack, from development to production. It requires clear communication between data engineering teams and source system stakeholders.
:p How can you ensure operational excellence in a DataOps environment?
??x
Ensure operational excellence by setting up clear communication chains with source system teams. Work towards incorporating DevOps practices into both your and their workflows to address errors quickly. This involves regular collaboration, automation setup, observability measures, and incident response planning.

---

#### Automation Impact
Automation impacts both the source systems and data workflows. Data engineers should consider decoupling these systems to ensure independent operation.
:p How does automation impact your work in DataOps?
??x
Automation in source systems can affect your data workflow automation. Consider decoupling these systems so they can operate independently, ensuring that issues in one system do not cascade into the other.

---

#### Observability and Monitoring
Observability is key to identifying issues with source systems proactively. Setting up monitoring for uptime and data conformance ensures that you are aware of any problems.
:p How should you monitor source systems?
??x
Set up monitoring for source system uptime and data quality. Use existing monitoring tools created by the teams owning these systems. Also, establish checks to ensure that data conforms to expected standards for downstream usage.

---

#### Incident Response Planning
Incident response plans are necessary to handle unexpected situations in data pipelines.
:p What should be included in an incident response plan?
??x
An incident response plan should include strategies for handling issues such as source systems going offline. It should outline steps like backfilling lost data once the system is back online and ensuring minimal disruption to reports.

---

#### Data Architecture Reliability
Understanding the reliability of upstream systems is crucial, given that all systems suffer from entropy over time.
:p How does understanding system reliability benefit your work?
??x
Understanding system reliability helps in designing robust data pipelines. It involves knowing how often a system fails and how long it takes to restore functionality. This knowledge aids in planning for potential disruptions.

---

#### Data Architecture Durability
Data loss due to hardware failures or network outages is inevitable. Understanding how source systems handle such issues ensures your managed data systems are resilient.
:p How should you account for durability concerns?
??x
Account for durability by understanding how the source system handles data loss from hardware failures or network outages. Develop plans for handling outages over extended periods and limiting the impact of these events.

---

#### Data Architecture Availability
Ensuring availability is critical when systems are expected to be up and running at specific times.
:p How can you guarantee system availability?
??x
Guaranteeing availability involves understanding who is responsible for the source system’s design and how breaking changes will be managed. Create Service Level Agreements (SLAs) with the source system team to set expectations on potential failures.

---

#### Orchestration Cadence and Frequency
Orchestration within data engineering workflows requires correct network access, authentication, and authorization.
:p What should you consider when orchestrating with source systems?
??x
Consider the cadence and frequency of data availability. Determine if data is available on a fixed schedule or can be accessed dynamically. Also, think about integrating application and data workloads into shared Kubernetes clusters for better management.

---

