# Flashcards: 2B004-Data-Governance_processed (Part 14)

**Starting Chapter:** Summary

---

#### Data Quality Overview
Data quality ensures that data is accurate, complete, and timely for a specific business use case. Different business tasks require different levels of these qualities.

:p What does "data quality" ensure?
??x
Data quality ensures that the data's accuracy, completeness, and timeliness are relevant to the business use case in mind.
x??

---

#### Importance of Data Quality
Data quality has real-life impacts. Poor data can lead to incorrect decisions and wasted resources.

:p How does poor data affect businesses?
??x
Poor data can result in incorrect decisions and inefficient processes, leading to financial losses and reputational damage.
x??

---

#### Techniques for Improving Data Quality
Several techniques exist to enhance data quality, including data cleaning, validation, normalization, and enrichment.

:p Name two techniques used to improve data quality.
??x
Two techniques used to improve data quality are data cleaning and validation. 
Data cleaning involves identifying and correcting or removing inaccurate records, while validation ensures that the data meets specific criteria before being stored in a database.
x??

---

#### Handling Data Quality Early
It's recommended to handle data quality issues early, as close to the source of the data as possible.

:p Why is it important to handle data quality early?
??x
Handling data quality early helps prevent errors from propagating through various processes and analytics workloads. It saves time and resources by addressing issues closer to their origin.
x??

---

#### Monitoring Data Products
Regularly monitoring the resultant products of your data sources ensures that they meet the current business needs.

:p What is the importance of monitoring data products?
??x
Monitoring data products is crucial because it helps maintain the accuracy, completeness, and timeliness of data. This process ensures that any changes or issues in the source data are identified early, preventing them from affecting downstream analytics.
x??

---

#### Revisiting Data Sources for New Tasks
When repurposing data for a different analytics workload, it's important to revisit the original sources to ensure they meet the new business task requirements.

:p Why should you revisit data sources when repurposing data?
??x
Revisiting data sources is essential because the same data may need to be adjusted or validated differently depending on its new use. This step helps ensure that the data quality remains high and relevant for the new analytics workload.
x??

---

#### Data Transformations
Data transformations are crucial steps in moving data between systems, often involving processes like extract-transform-load (ETL) or its modern variant, ELT. ETL involves extracting data from sources, transforming it into a desired format, and loading it into a target system. In contrast, ELT extracts the data first but loads it directly into the target system without extensive transformation, which is then handled within the data warehousing solution.

:p What are the main steps involved in data transformations (ETL or ELT)?
??x
The main steps involve extracting data from sources, transforming it to meet requirements, and loading it into a destination. ETL processes often include normalization and cleaning during the transformation phase, while ELT focuses on transformation within the target system after initial extraction.
```java
// Pseudocode for ETL process
public void performETL(String sourceSystem) {
    // Extract data from source system
    String extractedData = extractFrom(sourceSystem);
    
    // Validate and transform data
    String transformedData = validateAndTransform(extractedData);
    
    // Load transformed data into destination
    loadIntoDestination(transformedData);
}
```
x??

---

#### Data Validation During Extraction
Data validation is a critical step during the extraction process. It ensures that the retrieved values are as expected, verifying the completeness and accuracy of records against predefined criteria. This helps in maintaining the quality and reliability of data even before it undergoes further processing.

:p What is the purpose of performing data validation during the extraction phase?
??x
The purpose of performing data validation during the extraction phase is to ensure that the retrieved values match expected values, verifying completeness and accuracy. This step helps maintain data quality by identifying and correcting discrepancies early in the process.
```java
// Pseudocode for data validation
public boolean validateData(String extractedData) {
    // Check if records are complete and accurate
    return checkCompletenessAndAccuracy(extractedData);
}
```
x??

---

#### Data Lineage
Lineage, also known as provenance, is the recording of the path that data takes through various stages such as extraction, transformation, and loading. It provides a historical context for datasets, explaining their origins and transformations. This information helps in understanding why certain datasets exist and where they came from.

:p What does lineage record in the context of data transformations?
??x
Lineage records the path that data travels through different stages like extraction, transformation, and loading. It explains how datasets were created, transformed, imported, and used throughout their lifecycle, helping to answer questions about dataset origins and purposes.
```java
// Pseudocode for lineage recording
public void recordLineage(String source, String transformationDetails, String destination) {
    // Record the path data takes from extraction to loading
    log("Data extracted from: " + source);
    log("Transformed using: " + transformationDetails);
    log("Loaded into: " + destination);
}
```
x??

---

#### ETL vs. ELT
ETL and ELT are methods for moving and transforming data between systems. While ETL involves extracting data, transforming it, and then loading it, ELT extracts the data first and loads it directly into a target system without extensive transformation. The decision on which to use depends on the specific requirements and capabilities of the environment.

:p What is the difference between ETL and ELT in terms of data processing?
??x
ETL involves extracting data from sources, transforming it through validation and normalization, and then loading it into a destination system. In contrast, ELT extracts data first, loads it directly into the target system, and performs transformation within that system. The choice depends on whether extensive transformation is needed before loading or can be handled in the target environment.
```java
// Pseudocode for ETL vs. ELT decision
public void decideETLorELT(String requirement) {
    if (requirement.includes("heavy transformation")) {
        performETL();
    } else {
        performELT();
    }
}
```
x??

---

#### Importance of Context in Data Extraction
Maintaining the business context during data extraction is crucial because early normalization and cleaning processes may remove valuable information. It's important to consider what might be needed for future use cases when extracting data, as this can impact the trustworthiness and governance of the data.

:p Why is it important to keep the business context in mind during data extraction?
??x
It is essential to maintain the business context during data extraction because early normalization and cleaning processes can remove valuable information. Considering potential future needs when extracting data ensures that you do not inadvertently lose critical details, which could be necessary for new use cases. This helps maintain the trustworthiness and governance of the data.
```java
// Pseudocode for considering business context in data extraction
public void extractDataWithContext(String source) {
    // Extract data with awareness of future needs
    String extractedData = extractFrom(source, considerFutureNeeds());
    
    // Perform validation and transformation
    String validatedAndTransformedData = validateAndTransform(extractedData);
}
```
x??

---

#### Scorecard for Data Sources
A scorecard is a tool used to evaluate the information context of data sources. It describes what information is present and potentially lost during transformations, helping to ensure that necessary details are not discarded early in the process.

:p What is a scorecard and how does it help in data governance?
??x
A scorecard is a tool for evaluating the information context of data sources, describing what information is present and what might be lost during transformations. This helps in ensuring that critical details are not discarded too early, maintaining the integrity and usefulness of the data.
```java
// Pseudocode for creating a scorecard
public void createScorecard(String source) {
    // Evaluate information context of the source
    String evaluation = evaluateInformationContext(source);
    
    // Log or store the scorecard details
    log("Source: " + source + ", Evaluation: " + evaluation);
}
```
x??
---

#### Lineage Importance in Data Governance
Data moves around your data lake, intermingling and interacting with other sources to produce insights. Metadata about these data sources is at risk of getting lost as data travels. This can affect decisions on:
- Data quality: Assessing reliability (e.g., automatic vs. human-curated).
- Sensitivity: Handling different levels of sensitivity in data.
Lineage ensures that metadata follows the data, supporting governance by tracking its journey.

:p Why is lineage important for data governance?
??x
Lineage is crucial because it helps maintain information about the origin and nature of data as it processes through various stages. This information supports decisions on:
- Data quality: Ensuring reliability.
- Sensitivity: Protecting privacy.
- Authenticity: Preventing mixing of certain sources.

Tracking lineage allows organizations to make informed decisions about:
- Mixing different data sources.
- Access control based on data characteristics.
- Maintaining the accuracy and relevance of derived data products.
x??

---

#### Collecting Lineage Information
Ideally, a data warehouse or catalog should collect lineage information from origin to final data product. However, this is not always common. Alternative methods like API logs can help infer lineage:
- SQL job logs: Track table creation statements and predecessors.
- Procedural pipelines (R, Python): Monitor programmatically generated data.

:p How can we ideally collect lineage information?
??x
Ideally, the data warehouse or catalog should have a facility that starts from the data origin and ends with final products. This would involve:
- Tracking every transaction along the way.
- Using tools like job audit logs to create lineage graphs.

However, this is often not practical, so alternative methods are needed to infer or manually curate information where gaps exist.

Example of using API log for lineage collection:
```java
public class LogAnalyzer {
    public void analyzeLog(String logEntry) {
        // Parse and extract metadata from the log entry.
        // Example: Identify SQL statements and their predecessors.
    }
}
```
x??

---

#### Granularity in Lineage
Lineage granularity can range from table/file level to column/row level, with higher levels of detail offering more precise tracking. Table-level lineage tracks which tables contribute to a new table:
- "This table is a product of this process and that other table."

Column-level lineage provides detailed information on how specific columns are combined:
- "This table consists of the following columns from another table."
- Useful for tracking sensitive data like PII.

Row-level lineage focuses on transactions, while dataset-level lineage offers coarse information about sources.

:p What is column-level lineage?
??x
Column-level lineage tracks the origin and transformation of individual columns as they move through different stages. For example:
- "Table C consists of columns X and Y from Table A and columns Z and W from Table B."
This level of detail helps in tracking sensitive information such as PII across multiple transformations.

Example of column-level tracking in pseudocode:
```java
public class DataTracker {
    public void trackColumns(String sourceTableName, String targetTableName) {
        // Track which columns from the source table contribute to the target table.
        // Example: if columns 1 and 3 are marked as PII in Table A, they will be tracked in Table B.
    }
}
```
x??

---

#### Granular Access Controls
Granular access controls allow more detailed permissions at a column or file level. This is useful for tables containing both sensitive data and general analytics data.

Example use case: Telecommunications company retail store transactions:
- Logs contain item details, date, price.
- Also include customer names (PII).

Using labels-based granular access controls:
```java
public class AccessController {
    public boolean allowAccess(String user, String table, int[] columns) {
        // Check if the user has permission to access specific columns in a table.
        // Example: Allow access to columns 1-3 and 5-8 of Table A but deny column 4.
    }
}
```
:p How can we implement more granular access controls?
??x
Implementing more granular access controls involves allowing permissions at the column or file level. For example:
- "Allow access to columns 1–3 and 5–8 of this table but not column 4."
This is particularly useful for tables containing a mix of sensitive data and general analytics.

Example implementation in Java:
```java
public class AccessController {
    public boolean allowAccess(String user, String tableName, int[] allowedColumns) {
        // Check if the user has permission to access specific columns.
        // Example logic: Grant or deny based on column numbers.
    }
}
```
x??

---

