# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 9)

**Starting Chapter:** Data Warehousing

---

#### OLTP and OLAP Differences
Background context explaining the differences between Online Transaction Processing (OLTP) and Online Analytical Processing (OLAP). The text highlights that transaction processing involves quick reads and writes for small numbers of records, while analytics involve scanning large datasets to calculate aggregate statistics.

:p What are the main characteristics distinguishing OLTP from OLAP?
??x
OLTP systems are designed for fast, frequent transactions with low-latency requirements. They typically handle a small number of records per query fetched by key, and their writes are random-access and low-latency based on user input. In contrast, OLAP systems are geared towards historical data analysis over large datasets, performing aggregate operations like count, sum, or average.

OLTP vs. OLAP characteristics:
- **Main read pattern:** Small number of records per query in OLTP; aggregates over large numbers in OLAP.
- **Main write pattern:** Random-access, low-latency writes from user input in OLTP; bulk imports (ETL) or event streams for OLAP.
- **Primary users:** End-users/customers via web applications in OLTP; internal analysts for decision support in OLAP.
- **Data representation:** Latest state of data (current point in time) in OLTP; history of events over time in OLAP.
- **Dataset size:** Gigabytes to terabytes in OLTP; terabytes to petabytes in OLAP.

Example of OLTP vs. OLAP comparison:
```java
public class TransactionProcessing {
    // Example method for handling a transaction, e.g., making a sale or updating inventory
}

public class AnalyticalQuery {
    // Example method for an analytical query, e.g., calculating total revenue per store
}
```
x??

---

#### Data Warehousing Overview
Background context explaining the concept of data warehousing and its role in enterprise-level database management. The text notes that data warehouses are separate databases used by analysts to query historical data without impacting transaction processing systems.

:p What is a data warehouse, and why is it used?
??x
A data warehouse is a read-only copy of data from various OLTP (Online Transaction Processing) systems within an enterprise. It is designed for ad-hoc analytical queries that do not affect the performance of ongoing transactions. Data warehouses help analysts make informed decisions by providing historical data in an analysis-friendly schema.

Example ETL process:
```java
public class ETLProcess {
    public void extractData() {
        // Code to periodically or continuously fetch data from OLTP systems
    }

    public void transformData() {
        // Code to clean and format data for analytical queries
    }

    public void loadToWarehouse() {
        // Code to insert transformed data into the data warehouse
    }
}
```
x??

---

#### ETL Process Explanation
Background context explaining how data is extracted, transformed, and loaded into a data warehouse. The text describes the Extract–Transform–Load (ETL) process in detail.

:p What are the steps involved in the ETL process for a data warehouse?
??x
The ETL process involves three main steps: Extraction, Transformation, and Loading.
- **Extraction:** Data is extracted from OLTP systems either through periodic dumps or continuous updates.
- **Transformation:** The extracted data is cleaned, formatted into an analysis-friendly schema, and possibly aggregated before being loaded.
- **Loading:** The transformed data is then inserted into the data warehouse.

Example ETL process:
```java
public class ETLProcess {
    public void extractData() {
        // Code to fetch data from OLTP systems (dumps or updates)
    }

    public void transformData() {
        // Code to clean and format data for analytical queries
    }

    public void loadToWarehouse() {
        // Code to insert transformed data into the data warehouse
    }
}
```
x??

---

#### Indexing Algorithms for OLTP vs. OLAP
Background context explaining that indexing algorithms work well for OLTP but are not suitable for OLAP due to their different access patterns.

:p Why are indexing algorithms effective for OLTP but not as good for OLAP?
??x
Indexing algorithms designed for OLTP are optimized for fast, frequent reads and writes involving small numbers of records. These indexes allow quick lookups by key and support low-latency random-access operations typical in transaction processing systems.

In contrast, OLAP queries often require scanning large datasets to perform aggregate calculations like count or sum over many records. Traditional indexing algorithms are not efficient for such tasks, making them less effective for OLAP.

Example of an inefficient index for OLAP:
```java
public class InefficientIndex {
    public boolean containsKey(String key) {
        // Search through the entire dataset to check if the key exists
        return false; // Dummy implementation
    }
}
```
x??

---

#### OLTP Databases vs. Data Warehouses
Background context explaining how transaction processing (OLTP) databases and data warehouses differ, focusing on their data models, query patterns, and optimization goals.

Both OLTP databases and data warehouses provide SQL interfaces but are optimized for different types of queries:
- OLTP databases optimize for high-speed transactions with ACID properties.
- Data warehouses focus on complex analytical queries over large datasets.

Many database vendors specialize in either transaction processing or analytics workloads, though some like Microsoft SQL Server and SAP HANA offer both. However, they increasingly split into separate storage and query engines accessible through a common SQL interface.

:p How do OLTP databases and data warehouses differ?
??x
OLTP databases are optimized for high-speed transactions with ACID properties (atomicity, consistency, isolation, durability), whereas data warehouses focus on complex analytical queries over large datasets. Both provide SQL interfaces but have different internal optimizations.
x??

---

#### Star Schema in Data Warehouses
Background context explaining the star schema, a common data model used in data warehouses for analytic purposes.

Star schemas are used to represent data for analysis in a highly structured manner. The central table is called the fact table, containing facts (data points) and foreign key references to dimension tables that provide additional details about each fact.

:p What is a star schema?
??x
A star schema is a common data model used in data warehouses where a central fact table contains detailed information, with foreign keys linking to multiple dimension tables. This structure allows for efficient analysis.
x??

---

#### Fact Tables and Dimension Tables
Background context explaining the roles of fact tables and dimension tables within a star schema.

Fact tables store facts (data points) about events, while dimension tables provide additional details such as who, what, where, when, how, why of each event. This structure facilitates complex queries for analysis.

:p What are fact tables and dimension tables?
??x
- Fact tables store detailed information about events.
- Dimension tables provide additional context (who, what, where, when, how, why) of the events referenced in fact tables.
x??

---

#### Example Schema: Star Schema in a Data Warehouse
Background context showing an example schema for a data warehouse at a grocery retailer.

The central table is the `fact_sales` table, which contains sales records. Foreign key references link to dimension tables like `dim_product`, providing detailed product information.

:p What does the example schema show?
??x
The example schema shows a star schema used in a data warehouse for a grocery retailer. The `fact_sales` table captures individual sales events, while `dim_product` provides details about each product.
x??

---

#### Large Scale Data Warehouses
Background context explaining the scale and structure of large enterprise data warehouses.

Large enterprises like Apple, Walmart, or eBay may have tens of petabytes of transaction history in their data warehouses, mostly stored in fact tables. The schema is designed to support complex analytical queries efficiently.

:p What characteristics do large-scale data warehouses typically have?
??x
Large-scale data warehouses often contain tens of petabytes of transactional data, primarily in fact tables. They are optimized for handling complex analytical queries and provide detailed dimension information.
x??

---

#### Open Source SQL-on-Hadoop Projects
Background context on open-source projects aiming to compete with commercial data warehouse systems.

Projects like Apache Hive, Spark SQL, Cloudera Impala, Facebook Presto, Apache Tajo, and Apache Drill offer alternatives to traditional commercial data warehouses by leveraging Hadoop for big data processing.

:p What are some open-source SQL-on-Hadoop projects?
??x
Open-source SQL-on-Hadoop projects include Apache Hive, Spark SQL, Cloudera Impala, Facebook Presto, Apache Tajo, and Apache Drill. These systems aim to provide analytics capabilities over large datasets.
x??

---

#### Dremel and SQL-on-Hadoop Projects
Background context on how some open-source projects are inspired by Google's Dremel for efficient data processing.

Projects like Apache Drill draw inspiration from Google’s Dremel project for its ability to process semi-structured data efficiently. This approach is relevant in SQL-on-Hadoop environments.

:p How do some open-source projects relate to Google’s Dremel?
??x
Projects like Apache Drill are inspired by Google’s Dremel, which allows efficient processing of semi-structured data, making them suitable for large-scale analytics in Hadoop ecosystems.
x??

---

#### Star Schema
Background context: The star schema is a data warehouse design that organizes data around facts and dimensions. Fact tables store measurements, while dimension tables provide metadata about these facts. The name "star" comes from the visual representation of these tables in a database schema.

:p What is a star schema and how does it organize data?
??x
A star schema organizes data by centralizing fact tables surrounded by multiple dimension tables. Fact tables contain measurements or metrics, while dimension tables provide context through metadata.
Example:
```sql
-- Star Schema Example
CREATE TABLE fact_sales (
    sale_id INT,
    date_key INT,
    product_sk INT,
    quantity INT
);

CREATE TABLE dim_date (
    date_key INT PRIMARY KEY,
    year INT,
    month INT,
    day_of_week VARCHAR(10)
);

CREATE TABLE dim_product (
    product_sk INT PRIMARY KEY,
    category VARCHAR(50),
    brand_name VARCHAR(50)
);
```
x??

---

#### Snowflake Schema
Background context: The snowflake schema is a variation of the star schema where dimension tables are further broken down into subdimensions. This results in a more normalized structure but can be less intuitive for analysts.

:p What is a snowflake schema and how does it differ from a star schema?
??x
A snowflake schema extends the star schema by breaking dimension tables into smaller, related tables. This normalization reduces redundancy but makes the schema harder to navigate for non-technical users.
Example:
```sql
-- Snowflake Schema Example
CREATE TABLE dim_product (
    product_sk INT PRIMARY KEY,
    category VARCHAR(50),
    brand_name VARCHAR(50)
);

CREATE TABLE dim_brand (
    brand_id INT PRIMARY KEY,
    brand_name VARCHAR(50)
);
```
x??

---

#### Column-Oriented Storage
Background context: In column-oriented storage, data is stored in columns rather than rows. This approach optimizes the storage and querying of large datasets by reducing the amount of disk I/O needed.

:p How does column-oriented storage work?
??x
Column-oriented storage organizes data into columns instead of rows, allowing queries to read only the necessary columns. This reduces unnecessary processing and enhances query performance.
Example:
```sql
-- Column-Oriented Storage Example
CREATE TABLE fact_sales (
    sale_id INT,
    date_key INT,
    product_sk INT,
    quantity INT,
    price DECIMAL(10,2),
    // many more attributes...
) WITH (ORIENTATION = COLUMN);
```
x??

---

#### Comparison of Row-Oriented vs. Column-Oriented Storage
Background context: Traditional row-oriented storage stores all columns in a single table row, while column-oriented storage separates data into individual files per column.

:p What are the key differences between row-oriented and column-oriented storage?
??x
Row-oriented storage stores all attributes together for each record, making it efficient for transactional workloads. Column-oriented storage organizes data by column, allowing queries to access only relevant columns, optimizing performance.
Example:
```sql
-- Row-Oriented Storage Example
CREATE TABLE fact_sales (
    sale_id INT,
    date_key INT,
    product_sk INT,
    quantity INT,
    price DECIMAL(10,2),
    // many more attributes...
);
```
x??

---

#### Query Optimization in Column-Oriented Storage
Background context: Queries on wide tables can benefit significantly from column-oriented storage because only the necessary columns are read and processed.

:p How does a query like Example 3-1 benefit from column-oriented storage?
??x
The query in Example 3-1 accesses fewer than 5 out of over 100 columns, which is efficient with column-oriented storage. Only these required columns are read, reducing the amount of data that needs to be processed.
Example SQL:
```sql
-- Query from Example 3-1
SELECT   dim_date.weekday, dim_product.category,
         SUM(fact_sales.quantity) AS quantity_sold 
FROM fact_sales
JOIN dim_date     ON fact_sales.date_key = dim_date.date_key
JOIN dim_product  ON fact_sales.product_sk = dim_product.product_sk
WHERE dim_date.year = 2013 AND dim_product.category IN ('Fresh fruit', 'Candy')
GROUP BY   dim_date.weekday, dim_product.category;
```
x??

---

