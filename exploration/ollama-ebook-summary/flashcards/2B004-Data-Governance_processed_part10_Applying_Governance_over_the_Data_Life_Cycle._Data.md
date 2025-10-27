# Flashcards: 2B004-Data-Governance_processed (Part 10)

**Starting Chapter:** Applying Governance over the Data Life Cycle. Data Governance Framework

---

#### Ensure Applicable Data Policies Are Captured
Background context: This involves making sure that all necessary data policies are documented and followed. This is crucial for compliance and audits, as it provides a clear roadmap on how data should be managed throughout its lifecycle.

:p What does ensuring applicable data policies involve?
??x
Ensuring applicable data policies involve capturing all relevant data policies required by organizational or legal standards. These policies should cover aspects such as data privacy, security, retention, and disposal. By documenting these policies, organizations can ensure consistent application of rules governing how data is managed.

For example:
- GDPR for EU data protection.
- HIPAA for healthcare industry in the USA.

This documentation helps in conducting audits where compliance with legal requirements must be demonstrated.
x??

---

#### Define Roles and Responsibilities
Background context: Defining roles and responsibilities ensures that everyone understands their part in the governance process. It includes assigning tasks to teams or individuals who will manage metadata, ensure policy adherence, etc.

:p What is the importance of defining roles and responsibilities?
??x
Defining roles and responsibilities is crucial because it clarifies what each team or individual needs to do to support data governance. This includes:

- Metadata management: Who will create and maintain metadata about datasets.
- Policy enforcement: Who will ensure that all policies are followed.

An example could be:
- Data Analysts: Responsible for analyzing the data using tools and methods defined by the organization.
- Data Stewards: Ensuring that the data is managed according to policies and standards.
- IT Teams: Implementing technical solutions to support governance processes.

This helps in creating a clear roadmap, making the process more organized and easier to follow.
x??

---

#### Applying Governance Over the Data Life Cycle
Background context: Data governance needs to cover all stages of the data lifecycle from creation to destruction. It involves people, processes, and technology working together to manage data effectively.

:p What does applying governance over the data life cycle entail?
??x
Applying governance over the data life cycle means integrating people, processes, and technology at every stage of the data's existence:

- Creation: Ensuring data is captured accurately.
- Storage: Managing where and how data is stored securely.
- Consumption: Making sure data is used appropriately by stakeholders.
- Archiving: Storing data for future reference after its primary use.
- Destruction: Removing data that no longer has any value or legal requirements.

For example:
```python
def govern_data_lifecycle(data, stage):
    if stage == 'creation':
        validate_data_source(data)
    elif stage == 'storage':
        secure_storage(data)
    elif stage == 'consumption':
        provide_access_control(data)
    elif stage == 'archiving':
        store_for_future_use(data)
    elif stage == 'destruction':
        delete_obsolete_data(data)
```
x??

---

#### Data Governance Framework
Background context: A framework helps visualize the plan and ensures a comprehensive approach to data governance. It covers concepts from metadata management to data archiving.

:p What is a data governance framework?
??x
A data governance framework provides a structured way to manage data throughout its lifecycle, integrating people, processes, and technology. It includes:

- Metadata Management: Tracking data attributes and definitions.
- Policy Enforcement: Ensuring compliance with legal and organizational standards.
- Data Archiving: Storing data for future use after primary usage is completed.

A simple framework could be represented as:
```
+-----------------------------+
|   Creation                  |
|  - Validate source         |
+-----------------------------+
|   Storage                   |
|  - Secure storage          |
+-----------------------------+
|   Consumption               |
|  - Provide access control  |
+-----------------------------+
|   Archiving                 |
|  - Store for future use    |
+-----------------------------+
|   Destruction               |
|  - Delete obsolete data    |
+-----------------------------+
```
x??

---

#### Data Life Cycle Overview

Background context: The data life cycle refers to the stages a piece of data goes through from its creation or capture until it is archived or disposed of. Understanding this lifecycle helps organizations manage their data effectively and ensure compliance with regulatory requirements.

:p What are the key phases in the data life cycle mentioned in the text?
??x
The text mentions five main phases: Data Creation, Data Processing, Data Storage, Data Usage, and Data Archiving.
x??

---

#### Data Creation Phase

Background context: During the creation phase, data is initially captured or created. This involves collecting metadata and lineage information to trace where the data comes from and how it will be used downstream.

:p What should organizations aim to capture during the initial data creation phase?
??x
Organizations should aim to capture both metadata (describing the data) and lineage (tracing the origin and flow of the data). Additionally, processes such as classification and profiling can be employed if dealing with sensitive data assets.
x??

---

#### Data Processing Phase

Background context: The processing phase involves integrating, cleaning, scrubbing, or performing ETL (Extract-Transform-Load) on the data to prepare it for storage and analysis. Ensuring data integrity during this phase is crucial.

:p Why is data quality preservation important during the data processing phase?
??x
Data quality preservation is critical because it ensures that the data remains accurate and consistent throughout its lifecycle, which is essential for reliable downstream processes like analytics.
x??

---

#### Data Storage Phase

Background context: In the storage phase, data and metadata are stored for future analysis. Proper security measures such as encryption at rest and backup strategies should be implemented.

:p What are some key practices to ensure data security during the data storage phase?
??x
Key practices include encrypting data both in transit and at rest, backing up data to ensure redundancy, implementing automated data protection solutions (e.g., encryption, masking), and establishing a robust recovery plan.
x??

---

#### Data Usage Phase

Background context: The usage phase involves analyzing and consuming the data for insights. Business intelligence tools play a critical role here, as does ensuring that the right people and systems have access to the data.

:p What are some best practices for managing data usage?
??x
Best practices include using a data catalog to help users discover data assets through metadata, implementing privacy controls, managing access permissions, and ensuring compliance with regulatory or contractual constraints.
x??

---

#### Data Archiving Phase

Background context: In the archiving phase, data is removed from active use but stored for potential future reference. Data classification should guide how long data is retained before disposal.

:p What considerations are involved in the data archiving process?
??x
Considerations include classifying data to determine its retention period and disposal method based on its sensitivity, criticality, and organizational needs.
x??

---

#### Perimeter Security and Data Protection Beyond Perimeter Security
Perimeter security is not sufficient for protecting data. It should be complemented with additional measures to ensure data security at rest, such as encryption, data masking, and permanent deletion.

:p What are the limitations of perimeter security in protecting data?
??x
Perimeter security alone cannot prevent unauthorized access or breaches once data has been compromised inside the network. Additional protections like encryption, data masking, and data lifecycle management are necessary to ensure data remains secure even if accessed by unauthorized individuals.
x??

---

#### Data Destruction Process
Before destroying any data, it is crucial to confirm whether there are retention policies in place that would require data to be kept for a certain period.

:p What should be confirmed before purging any data?
??x
Before deleting any data, you must check if there are any polices or regulations mandating the retention of specific data. Data classification helps guide the appropriate retention and disposal methods.
x??

---

#### Compliance Policy Creation
Creating a compliance policy involves understanding state and federal regulations, industry standards, and governance policies to ensure proper handling and protection of data.

:p How often should IT stakeholders revisit guidelines for destroying data?
??x
IT stakeholders are advised to review guidelines for destroying data every 12–18 months to ensure ongoing compliance with changing rules.
x??

---

#### Data Life Cycle Framework
Data goes through multiple stages from ingestion, processing, storage, and eventually destruction. Each stage involves specific actions such as scanning, classifying, encrypting, and auditing.

:p Describe the different pieces data might go through in a cloud-data platform.
??x
In a cloud-data platform, data moves through several stages:
1. Ingestion: Data is scanned, classified, and tagged before processing.
2. Staging: Data goes into buckets (ingest, released, admin quarantine).
3. Scanning and classification for sensitive information like PII.
4. Redaction or anonymization of sensitive data.
5. Tagging with PII labels.
6. Quality checks on the data.
7. Capture data provenance for lineage.
8. Encryption in transit.
9. Storage in a warehouse or lake, encrypted at rest.
10. Metadata addition and cataloging.
11. Audit trails capturing security health.

Each step ensures proper handling and protection of data throughout its lifecycle.
x??

---

#### Case Study: Nike and Strava Data Usage
Nike launched a new running shoe claiming a 4% speed increase, but lacked sufficient evidence. The New York Times used Strava data to investigate the claim by analyzing real-life performance records.

:p How did the New York Times use Strava data to test Nike's claims?
??x
The New York Times collected half a million real-life performance records from Strava and analyzed them to determine if there was any correlation between using Nike's Vaporfly shoes and achieving a 4% speed increase. This involved looking for patterns in the existing data without the controlled experimental setup needed for definitive proof.
x??

---

#### Access Management Throughout Data Lifecycle
Proper access management ensures that only authorized individuals have access to specific data at each stage of its lifecycle.

:p Why is robust identity and access management (IAM) important throughout a data platform's lifecycle?
??x
Robust IAM solutions are crucial because they ensure the right people and services have the correct permissions to access appropriate data across all stages. This prevents unauthorized access, ensures compliance, and maintains security.
x??

---

#### Data Provenance and Lineage Capture
Capturing provenance information helps in tracking how data changes over time and understanding its origin.

:p What is data provenance and why is it important?
??x
Data provenance refers to the record of all prior states and transformations of a piece of digital information. It's important for maintaining integrity, traceability, and compliance with regulatory requirements.
x??

---

#### Backup and Recovery Processes
Backup and recovery processes are essential in case of disasters or data loss.

:p Why is it necessary to employ backup and recovery processes?
??x
Backup and recovery processes are vital because they ensure that data can be restored after a disaster, maintaining business continuity. They also help comply with regulatory requirements for data retention.
x??

---

#### Example Scenario: Cloud Data Ingestion Pipeline
A business wants to ingest and share sensitive data on a cloud platform, requiring multiple security measures like encryption, tagging, and auditing.

:p Describe the steps involved in ingesting and processing sensitive data in a cloud environment.
??x
Steps include:
1. Configuring an ingestion pipeline using batch or streaming services for raw data movement.
2. Scanning and classifying data to identify sensitive information (PII).
3. Redacting or anonymizing sensitive parts of the data, capturing metadata.
4. Tagging data with PII labels.
5. Checking data quality.
6. Capturing provenance information.
7. Encrypting data in transit.
8. Storing encrypted data at rest.
9. Implementing backup and recovery mechanisms.
10. Adding business metadata for cataloging and discovery.
11. Maintaining audit trails.

Each step ensures secure handling of sensitive data throughout its lifecycle.
x??

