# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 19)

**Starting Chapter:** Principle 7 Data Ownership

---

#### Global Banking Industry Savings

Background context: The global banking industry saves between $2–4 billion annually through client onboarding cost reductions, primarily due to the implementation of standards that promote consistency and reliability. Standards like ISO 21586 ensure uniformity in descriptions of banking products and services (BPoS).

:p How do standards contribute to savings in the global banking industry?
??x
Standards contribute to savings by promoting consistency across key aspects such as quality, compatibility, interoperability, comparability, and reliability. For instance, ISO 21586 standardizes the description of banking products or services (BPoS), enabling customers to understand and compare them effectively.

For example, using a standardized format for describing loan products ensures that all banks use consistent terms and descriptions, which simplifies client onboarding processes and reduces errors.
??x
---

#### Data Backups

Background context: Financial institutions face operational risks due to data loss, which can occur through various means such as accidental destruction or hardware failure. Data backups are a critical strategy for mitigating these risks.

:p What is the purpose of data backups in financial institutions?
??x
The primary purpose of data backups is to prevent data loss and ensure business continuity by creating copies of original data stored in different locations. These backups can be used to recover data in case of an accident, such as a hardware failure or data corruption.
??x

---

#### Data Archiving

Background context: Financial institutions often need to manage large volumes of historical data that are no longer actively used but still required for regulatory compliance, audits, and reviews. Data archiving is the process of moving such data from production systems to long-term storage.

:p What is the difference between data backups and data archiving?
??x
Data backups and data archiving serve different purposes:
- **Data Backups**: Created primarily as a safeguard against data loss due to accidents or disasters, allowing recovery of critical data. They are part of disaster recovery strategies.
- **Data Archiving**: Used for long-term storage of historical data that is no longer actively used but still needed for compliance and auditing purposes.

For example, a financial institution might archive transactional records beyond a certain period while keeping recent records in active use.
??x

---

#### Data Backup Lifecycle

Background context: Implementing an effective data backup strategy involves defining steps and elements such as backup timing, data types, number of backups, security measures, storage locations, recovery tests, retention times, and deletion policies.

:p What are the key components of a data backup lifecycle?
??x
Key components of a data backup lifecycle include:
- **Backup Timing**: Scheduled, on-demand, or event-driven.
- **Data Types**: Operational, analytical, client-related.
- **Number of Backups**: Determined based on storage capacity and recovery needs.
- **Security Measures**: Encryption to protect data during transit and at rest.
- **Storage Locations**: Multizones, geographies, and data centers.
- **Recovery Tests and Plans**: Regularly scheduled to ensure backups can be restored when needed.
- **Retention Time**: Period for which backups are stored before deletion.
- **Deletion**: Process of removing old backups that no longer meet retention requirements.

For example:
```java
public class DataBackupManager {
    public void scheduleBackups(BackupPolicy policy) {
        // Schedule backups based on the provided policy
    }
    
    public void testRecovery() {
        // Test recovery process to ensure data can be restored from backups
    }
}
```
x??

---

#### Data Archival Policy

Background context: A data archival policy manages the archiving of historical data that is no longer actively used but still required for compliance and auditing purposes. This involves creating a data retention policy, using appropriate software, and ensuring access to archived data.

:p What are the key elements of a data archival policy?
??x
Key elements of a data archival policy include:
- **Data Retention Policy**: Defines how long specific types of data must be retained.
- **Data Archival Software**: Tools used for moving and managing data in archiving processes.
- **Data Access and Discovery Functionalities**: Features that allow users to search, retrieve, and manage archived data.

For example:
```java
public class DataArchivalPolicy {
    private RetentionRules rules;
    
    public void enforceRetention(RetentionRules rules) {
        // Implement logic to enforce the retention policy based on specific rules
    }
}
```
x??

---

#### Data Aggregation in Financial Institutions
Background context: Financial institutions operate through diverse activities like lending, payments, investments, and insurance. Each activity is often overseen within distinct organizational silos, leading to data being maintained using separate systems. This fragmentation makes it challenging to generate an aggregated view of the institution’s operations and risks.

:p What are some challenges associated with data aggregation in financial institutions?
??x
The primary challenge lies in consolidating data across multiple business units, legal entities, and disparate data storage systems. This complexity often hinders the ability to create a comprehensive overview of an institution's activities and risk exposures.
```java
public class DataAggregationChallenge {
    public void printChallenges() {
        System.out.println("Fragmented silos prevent effective aggregation.");
        System.out.println("Inconsistent data formats across different systems.");
        System.out.println("Lack of standardized processes for data exchange.");
    }
}
```
x??

---

#### Basel Committee's Principles on Data Aggregation
Background context: The Basel Committee published 13 principles to guide the design and implementation of data aggregation capabilities in financial institutions. These principles aim to address the challenges of integrating disparate data systems and generating a unified view of an institution’s activities.

:p How many principles did the Basel Committee publish for designing and implementing data aggregation?
??x
The Basel Committee published 13 principles for designing and implementing data aggregation capabilities in financial institutions.
```java
public class BaselCommitteePrinciples {
    public int getNumberOfPrinciples() {
        return 13;
    }
}
```
x??

---

#### Data Lineage and Information Lifecycle
Background context: Financial data engineers often use the data lifecycle approach, also known as the information lifecycle. This framework helps understand how data evolves from creation to usage and archiving. The lifecycle includes five main phases: extraction, transformation, storage, usage, and archiving.

:p What are the five main phases of the data lifecycle?
??x
The five main phases of the data lifecycle are:
1. Extraction
2. Transformation
3. Storage
4. Usage
5. Archiving
```java
public class DataLifecyclePhases {
    public void printPhases() {
        System.out.println("Extraction - Data is collected from various sources.");
        System.out.println("Transformation - Raw data is cleaned and prepared for analysis.");
        System.out.println("Storage - Cleaned data is stored in a structured format.");
        System.out.println("Usage - Data is used for various analytical purposes.");
        System.out.println("Archiving - Old data that is no longer actively used is preserved.");
    }
}
```
x??

---

#### Data Lineage Framework
Data lineage is crucial for financial institutions to track and understand the transformations, movements, and origins of data throughout its lifecycle. This helps ensure data integrity and reduces risks associated with errors or misinterpretations.

:p What is data lineage?
??x
Data lineage refers to tracking a given data object through its entire lifecycle, including how it was generated, transformed, stored, delivered, and archived. It involves understanding the discrete steps in processing data from creation to consumption, ensuring transparency and accountability within financial systems.
??x

---

#### Lineage Graphs
Lineage graphs provide a visual representation of the history of data processing. They are useful for understanding complex data transformation processes but can become less performant as logic complexity increases.

:p What is a lineage graph?
??x
A lineage graph is a graphical visualization that shows the historical flow and transformations of data through different stages in the data pipeline. It helps users trace the origin and evolution of each piece of data, making it easier to manage complex data processes.
??x

---

#### Audit Trails for Financial Data
Audit trails are special implementations of data lineage designed specifically for financial transactions. They record every step of a transaction chronologically, which is essential for regulatory compliance and fraud detection.

:p What is an audit trail?
??x
An audit trail is a chronological record of all actions related to the generation, processing, or consumption of data, particularly useful in finance for tracking activities such as trades, accounting transactions, and other financial operations.
??x

---

#### Order Audit Trail System (OATS)
OATS is an automated system used by financial institutions to record orders, quotes, and other trade-related data. It helps in the efficient management and auditing of order lifecycle.

:p What does OATS do?
??x
The Order Audit Trail System (OATS) automates the recording of information on orders, quotes, and related trade data from all shares traded on the National Market System (NMS). This system aids in tracking an order's lifecycle from its reception through execution or cancellation.
??x

---

#### Data Catalogs for Financial Institutions
Data catalogs are essential tools that help financial institutions manage large volumes of diverse data. They provide searchable metadata and management tools to facilitate data discovery and documentation.

:p What is a data catalog?
??x
A data catalog is a set of metadata combined with search and management tools, enabling users to find and document data assets within an organization efficiently. It acts as a central inventory that allows for quick and easy access to relevant data.
??x

---

#### Implementation of Data Catalogs
Data catalogs can be implemented in various ways depending on the institution's needs and complexity. They range from simple databases storing metadata to full-fledged applications with advanced features.

:p How are data catalogs typically implemented?
??x
Data catalogs can be implemented using databases where metadata is stored directly, or as sophisticated applications offering features like a user interface (UI), search functionality, metadata management, user permissions, and API integration.
??x

---

#### Example of Data Catalog Implementation: CKAN
CKAN is an open-source Python-based library that provides a practical example for implementing data catalogs. It supports the storage and retrieval of metadata in a structured manner.

:p What is CKAN?
??x
CKAN is an open-source software that implements data catalogs by providing tools to store, search, manage, and document metadata in a structured format. It can be used as a central inventory system for financial institutions.
??x

---
#### Data Ownership: Legal Context
Data ownership can refer to the legal owner of a given data asset, particularly as concerns increase with data collection and cloud storage. This topic has gained prominence due to more frequent data collection practices and third-party storage solutions like cloud services.

:p What is the legal context of data ownership?
??x
The legal context refers to who holds the rights over a specific piece of data, especially when data is collected about people or organizations and stored in the cloud. Legal owners might have different rights depending on local laws, contracts, and agreements.
x??

---
#### Data Ownership: Organizational Context
In an organizational setting, data ownership involves designating individuals or teams to manage specific data assets, leveraging their domain expertise for better maintenance and governance.

:p How does data ownership function within a financial institution?
??x
Data ownership is assigned to individuals or teams who are experts in the relevant domain. They oversee tasks such as collection, cleaning, sharing, and management of specific data assets. This approach ensures that those with deep understanding can maintain and manage the data more effectively.

For example:
- A financial analyst team might be designated as owners for stock market data.
```java
// Pseudocode to assign data ownership roles
public class DataOwner {
    private String name;
    private String role;

    public DataOwner(String name, String role) {
        this.name = name;
        this.role = role;
    }

    // Method to check if the owner can manage a specific dataset
    public boolean canManageData(String dataAsset) {
        return this.role.equals("Financial Analyst") && dataAsset.startsWith("stock-");
    }
}
```
x??

---
#### Data Contracts: Definition and Purpose
Data contracts are agreements that outline how data should be generated, governed, and used. They help in defining the structure, semantics, and other critical aspects of data.

:p What is a data contract?
??x
A data contract is an agreement between data generators (e.g., software engineers) and consumers (e.g., business analysts or data scientists). It defines how the data should be structured, governed, and used to meet specific business requirements. This ensures that both parties agree on expectations, reducing issues related to data quality.

For example:
```java
// Pseudocode for defining a basic data contract
public class DataContract {
    private String id;
    private String title;
    private String description;
    private String owner;

    public DataContract(String id, String title, String description, String owner) {
        this.id = id;
        this.title = title;
        this.description = description;
        this.owner = owner;
    }

    // Method to check if the contract matches specific criteria
    public boolean meetsRequirements(Map<String, Object> requirements) {
        return this.title.equals("Daily Adjusted Stock Price Extraction") && 
               this.id.startsWith("stock-price-extraction");
    }
}
```
x??

---
#### Data Contracts: Real-World Example
An illustrative example of a data contract involves setting up daily price extraction for the top 100 US stocks, ensuring it meets specific criteria like no null prices and timely delivery.

:p Provide an example of a data contract specification.
??x
Here is an example of a basic data contract specification:

```json
{
    "dataContractSpecification": "0.0.1",
    "id": "stock-price-extraction",
    "info": {
        "title": "Daily Adjusted Stock Price Extraction",
        "version": "0.0.1",
        "description": "daily extraction of the adjusted stock price of the top 100 U.S stocks by market capitalization.",
        "owner": "Analytics Team",
        "contact": {
            "name": "John Smith (Analytics Team Lead)",
            "email": "john.smith@example.com"
        }
    }
}
```

This contract ensures that data extraction meets specific business requirements, such as timely delivery and data quality standards.
x??

---

#### Data Contract Overview
Data contracts are design patterns focused on automating data quality and governance across various systems. They emphasize consistency, reliability, and compliance of data usage.

:p What is a data contract?
??x
A data contract is a design pattern that ensures automated quality and governance in diverse data systems by specifying usage terms, SLAs, limitations, and cost structures.
x??

---

#### Service Level Agreement (SLA)
The service level agreement outlines the specific time frame for data availability. In this case, it states that the data must be available by 10:00 AM on each working day.

:p What is the SLA mentioned in the contract?
??x
The SLA specifies that the data should be available at 10:00 AM of each working day.
x??

---

#### Daily Record Count Limitation
The daily record count limitation restricts the number of records processed per day, indicating a constraint on the amount of data that can be handled.

:p What is the daily record count limit?
??x
The daily record count limit is 100 observations per day.
x??

---

#### Data Usage and Applications
The usage terms indicate that the data can be used for financial analysis, backtesting, and machine learning applications. This highlights the broad applicability of the dataset.

:p What are the permitted uses of this data?
??x
This data can be used for financial analysis, backtesting, and machine learning purposes.
x??

---

#### Data Limitations
The limitations mentioned include that the data is not suitable for intra-day financial time series analysis. Additionally, there are specific restrictions on identifiers and data processing limits.

:p What are the main limitations of the provided data?
??x
The main limitations are:
- Not suitable for intra-day financial time series analysis.
- May be missing some identifiers such as ISIN.
- Max data processing per day: 10 Gigabytes.
- Max instrument requests per day: 1000 instruments.

x??

---

#### Data Schema Description
The schema describes the structure of the data, including fields like `price_date`, `adjusted_price`, and `instrument_ticker`. It specifies that each record represents an observation for a single instrument on a specific date.

:p What does the data schema describe?
??x
The data schema defines the structure of each record in the dataset. Each record contains:
- `price_date`: The timestamp of the price observation.
- `adjusted_price`: The adjusted price value, with 4 decimal precision and a range from 0 to 1,000,000.
- `instrument_ticker`: The ticker identifier for the stock.

x??

---

#### Data Reconciliation
Data reconciliation involves aligning diverse sets of records across different systems to ensure consistent financial transactions and balances. This is crucial for minimizing errors and ensuring operational integrity.

:p What is data reconciliation?
??x
Data reconciliation is a process that involves aligning disparate records from various systems to create a unified view of financial transactions and balances, thereby reducing errors and discrepancies.
x??

---

#### Portfolio Reconciliation Process
Portfolio reconciliation compares records between two or more counterparties (e.g., management companies and custodians) for the same financial instrument. This helps in verifying holdings, transactions, and positions.

:p How is portfolio reconciliation typically performed?
??x
Portfolio reconciliation involves comparing records of holdings, transactions, and positions between different entities (such as a management company and a custodian). For example:
- Management Company A reports a $50 million exposure.
- Custodian B reports a $49.5 million exposure.

By aligning these records, discrepancies can be identified and resolved to ensure accurate financial reporting.

x??

---

#### Example of Portfolio Reconciliation
In the fund industry, multiple entities may hold different but related data for the same portfolio exposure. This example illustrates how reconciliation helps in identifying and resolving inconsistencies.

:p Provide an example of a portfolio reconciliation scenario.
??x
Suppose:
- Management Company A reports $50 million exposure in technology stocks for a specific mutual fund.
- Custodian B shows a $49.5 million exposure in the same sector.

Through portfolio reconciliation, these discrepancies can be identified and resolved to ensure accurate financial reporting and maintain customer trust.

x??

---

#### Portfolio Reconciliation Process
Portfolio reconciliation is a critical process used to ensure consistency and accuracy of financial data between different entities involved in managing or holding assets. This process involves comparing the data from Management Company A (MC) and Custodian B (CB) to identify and resolve discrepancies, thereby providing a unified view of the fund’s holdings.

:p What is portfolio reconciliation?
??x
Portfolio reconciliation is the process of comparing financial data between two or more parties involved in managing assets, such as MC and CB. This ensures that both parties have accurate and consistent records of the fund's holdings.
x??

---

#### Payment Process Discrepancies
In payment processes, multiple entities are often involved in storing ledger records, leading to discrepancies among banks, FinTech companies, and service providers like BaaS providers. These discrepancies can arise due to various factors including data duplication, incomplete or inaccurate record-keeping practices, system upgrades, and the complexity of reconciling multiple systems.

:p What causes discrepancies in payment processes?
??x
Discrepancies in payment processes can be caused by several factors:
1. **Data Duplication**: Multiple entries for the same transaction.
2. **Incomplete or Inaccurate Record-Keeping Practices**: Errors or omissions in recording transactions.
3. **System Upgrades**: Delays or errors during upgrades that affect ledger balances.
4. **Complexity of Reconciliation**: The challenge of aligning records from multiple systems.

For example, consider a transaction where both the bank and BaaS provider record the payment but with slight differences:
```java
// Pseudocode to simulate discrepancy in recording transactions
class Transaction {
    String account;
    double amount;
    boolean recorded;

    public void recordTransaction(String account, double amount) {
        this.account = account;
        this.amount = amount;
        this.recorded = false; // Initially not recorded
    }

    public void updateRecord() { 
        if (!recorded) {
            System.out.println("Recording transaction for " + account);
            recorded = true; 
        } else {
            System.out.println("Transaction already recorded.");
        }
    }
}

// Example usage
Transaction bankLedger = new Transaction();
bankLedger.recordTransaction("Account123", 500.0);

Transaction baaSProvider = new Transaction();
baasProvider.recordTransaction("Account123", 499.99); // Slight discrepancy

// Attempt to reconcile
if (!bankLedger.equals(baaSProvider)) {
    System.out.println("Discrepancy found: " + (bankLedger.amount - baaSProvider.amount));
}
```
x??

---

#### Data Security and Privacy in Financial Institutions
Financial institutions deal with highly sensitive data such as customer financial information, transactions, investment strategies, and credit card numbers. This makes them prime targets for cyberattacks. The industry is particularly vulnerable due to the complexity of interdependent systems, potential cascading effects from security breaches, and significant monetary impacts.

:p Why are financial institutions particularly vulnerable to security threats?
??x
Financial institutions are particularly vulnerable to security threats because:
1. **Highly Sensitive Data**: They handle sensitive data like customer financial information, transactions, investment strategies, and credit card numbers.
2. **Complex Interdependencies**: The interdependence of financial systems means a breach in one institution can cause a cascading effect across the entire system.
3. **Significant Monetary Impact**: Financial data is directly connected to client funds, making breaches costly.

For example, consider the impact of a security breach:
```java
class BankSecurityIncident {
    String bankName;
    int numberOfCustomersImpacted;
    double financialImpact;

    public void reportIncident(String bankName, int customers, double financialLoss) {
        this.bankName = bankName;
        this.numberOfCustomersImpacted = customers;
        this.financialImpact = financialLoss;
    }
}

// Example usage
BankSecurityIncident citiGroupBreach2011 = new BankSecurityIncident();
citiGroupBreach2011.reportIncident("Citi Group US", 360_000, 10_000_000.0);
```
x??

---

#### ISO 27001 for Data Security
ISO 27001 is a widely recognized standard that provides guidelines for developing an Information Security Management System (ISMS) to protect data from cyber threats. It covers stages such as assessing vulnerability risks, developing policies and procedures, training employees, and managing incidents.

:p What does ISO 27001 cover?
??x
ISO 27001 covers the following aspects of data security:
1. **Assessing Vulnerability Risks**: Identifying potential security threats.
2. **Developing Policies and Procedures**: Creating guidelines for secure operations.
3. **Training Employees**: Ensuring staff are knowledgeable about security practices.
4. **Managing Incidents**: Handling breaches and other security incidents.

For example, an organization might follow these steps:
```java
// Pseudocode to simulate ISO 27001 compliance process
class ISMS {
    String riskAssessment;
    String policiesAndProcedures;
    String employeeTraining;
    String incidentManagement;

    public void initiateProcess() {
        this.riskAssessment = "Identify and assess risks";
        this.policiesAndProcedures = "Develop security policies and procedures";
        this.employeeTraining = "Train employees on security practices";
        this.incidentManagement = "Implement an incident response plan";
    }

    // Example method to check compliance
    public boolean isCompliant() {
        return !riskAssessment.isEmpty() && 
               !policiesAndProcedures.isEmpty() &&
               !employeeTraining.isEmpty() &&
               !incidentManagement.isEmpty();
    }
}

// Example usage
ISMS myOrganization = new ISMS();
myOrganization.initiateProcess();

if (myOrganization.isCompliant()) {
    System.out.println("The organization is ISO 27001 compliant.");
} else {
    System.out.println("The organization needs to improve its compliance process.");
}
```
x??

---

---
#### ISO 27701 Overview
ISO 27701 builds on top of ISO 27001 to ensure that a system is in place for handling personally identifiable information (PII) according to data legislation and regulations. This standard guides organizations through the steps needed to protect PII, comply with data regulations, and be transparent about personal data handling.
:p What does ISO 27701 focus on?
??x
ISO 27701 focuses on ensuring that a system is established for managing personally identifiable information (PII) in compliance with relevant data legislation and regulations. It provides guidelines to protect PII, ensure regulatory compliance, and maintain transparency regarding how personal data is handled.
x??
---

---
#### Data Security vs Privacy
When designing for security, it's assumed that an adversary might launch a cyberattack against the organization. When designing for privacy, it is assumed that personal data may not be handled in accordance with legal requirements.
:p How do security and privacy design approaches differ?
??x
Security design focuses on protecting systems from potential cyber threats by assuming adversaries might launch attacks. Privacy design ensures compliance with data regulations and laws, focusing on proper handling of PII.

For example:
```java
public class SecurityDesign {
    public void preventCyberattacks() {
        // Implementing firewalls, encryption, access controls, etc.
    }
}

public class PrivacyDesign {
    public void handlePIICompliance() {
        // Ensuring data protection policies are followed and transparency is maintained.
    }
}
```
x??
---

---
#### Types of Cyberattacks on Financial Institutions
Financial institutions can face a variety of cyberattacks including malware, ransomware, spoofing, spam and phishing, DDoS attacks, corporate account takeovers, brute force attacks, and SQL injections.

:p List the types of cyberattacks against financial institutions.
??x
The types of cyberattacks against financial institutions include:
- Malware: malicious software that can enable hackers to access sensitive data.
- Ransomware: malware where data is held hostage in exchange for a ransom.
- Spoofing: impersonating and replicating a financial institution's website.
- Spam and phishing: email-based cybercrimes soliciting personal information.
- DDoS attack: flooding the server with internet requests.
- Corporate account takeover: gaining access to corporate accounts for fraudulent transactions.
- Brute force attacks: guessing user credentials through trial and error.
- SQL injection: manipulating backend databases.

Example:
```java
public class CyberattackTypes {
    public void detectMalware() {
        // Implementing antivirus software, firewalls, etc.
    }

    public void handleRansomware() {
        // Backup systems, secure data storage.
    }
}
```
x??
---

