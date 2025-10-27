# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 16)

**Starting Chapter:** Conclusion. Additional Resources

---

#### Airflow Alternatives and Orchestration Frameworks
In this context, Airflow is a popular open-source platform for orchestrating workflows. However, it has limitations that may prompt data engineers to explore alternatives like Prefect or Dagster. These frameworks aim to address issues by rethinking components of the Airflow architecture.
:p What are some key orchestration contenders as mentioned in the text?
??x
Prefect and Dagster are two notable alternatives to Airflow. They aim to solve some of the problems with Airflow by rethinking its component architecture, providing more robust features for schema management, lineage tracking, and cataloging.
```python
# Example of a simple Prefect flow
from prefect import Flow

def say_hello(name):
    print(f"Hello {name}!")

with Flow("ExampleFlow") as flow:
    # Define tasks here
    task1 = say_hello("World")
```
x??

---

#### Simplification and Abstraction in Data Engineering
The text emphasizes the importance of simplification and abstraction across the data stack. It encourages the use of prebuilt open-source solutions to avoid reinventing the wheel, as many common problems have already been solved.
:p Why is it important for a data engineer to strive for simplification and abstraction?
??x
Simplification and abstraction are crucial because they allow data engineers to focus on areas that provide a competitive advantage. By using prebuilt open-source solutions for tasks like database connections, the engineer can avoid spending resources on "undifferentiated heavy lifting." This approach enables the company to concentrate on developing unique algorithms or processes that truly differentiate it from competitors.
```python
# Example of abstracting database connection in Python
def get_data_from_db(query):
    # Assume this function uses a prebuilt library like psycopg2 for PostgreSQL
    conn = psycopg2.connect("dbname=example user=example")
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()

result = get_data_from_db("SELECT * FROM users")
```
x??

---

#### Choosing the Right Technologies
Selecting appropriate technologies is a complex task due to rapid technological advancements and evolving best practices. The text suggests considering factors like use case, cost, build versus buy, and modularization when making choices.
:p What are some key considerations for choosing technology as mentioned in the text?
??x
Key considerations include:
- **Use Case**: Understand how each technology fits the specific requirements of your project.
- **Cost**: Evaluate both direct and indirect costs associated with different technologies.
- **Build vs. Buy**: Decide whether developing custom solutions or using off-the-shelf products is more cost-effective.
- **Modularization**: Ensure that chosen technologies can be integrated seamlessly to form a cohesive data stack.
```python
# Example of modularizing a data pipeline in Python using Prefect and Dask
from prefect import task, Flow
from dask.distributed import Client

@task
def process_data(data):
    return [x * 2 for x in data]

with Flow("DataProcessingFlow") as flow:
    client = Client()
    processed_data = process_data(range(10))

flow.run()
```
x??

---

#### Cloud FinOps and Cost Management
The text introduces the concept of Cloud FinOps, which focuses on managing cloud costs effectively. It references several resources for understanding more about this topic.
:p What is Cloud FinOps?
??x
Cloud FinOps involves applying financial operations principles to manage cloud spending efficiently. This includes monitoring usage, optimizing resource allocation, and ensuring cost transparency within an organization.
```python
# Example of a simple AWS Cost Explorer report using Boto3
import boto3

client = boto3.client('ce')
response = client.get_cost_and_usage(
    TimePeriod={
        'Start': '2023-01-01',
        'End': '2023-01-31'
    },
    Granularity='MONTHLY',
    Metrics=['UnblendedCost']
)
print(response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount'])
```
x??

---

#### Modern Data Stack
The text discusses the concept of a modern data stack, which involves leveraging various technologies to build efficient and scalable data pipelines. It encourages continuous improvement and innovation in this area.
:p What is meant by "modern data stack"?
??x
A modern data stack refers to an integrated set of tools and platforms used for processing, storing, and analyzing large volumes of data. This includes using cloud-native services, advanced analytics tools, and automation frameworks like Prefect or Dagster. The goal is to create a flexible and scalable infrastructure that can support complex data engineering tasks.
```python
# Example of integrating modern data stack components in Python
from prefect import Flow
from dask.distributed import Client

@task
def preprocess_data(data):
    # Data preprocessing logic here
    return data

with Flow("ModernDataStackFlow") as flow:
    client = Client()
    preprocessed_data = preprocess_data(range(10))

flow.run()
```
x??

---

#### Data Generation in Source Systems Overview
This chapter discusses the first stage of the data engineering lifecycle, which involves understanding and managing source systems where raw data is generated. The focus is on identifying the types of source systems and their characteristics to ensure effective data collection for downstream use cases.

:p What are the primary objectives of this phase in data engineering?
??x
The primary objectives are to understand the origin of the data, its generation process, and its characteristics so that it can be effectively processed and used for various purposes. This involves identifying the types of source systems, their quirks, and how they generate data.
x??

---
#### Types of Source Systems
Popular operational source systems include databases, applications, and IoT devices. These generate data in real-time or at specific intervals.

:p What are some common types of source systems mentioned in this phase?
??x
Some common types of source systems include databases, applications, and Internet of Things (IoT) devices. These systems generate data either in real-time or at specific intervals.
x??

---
#### Characteristics of Source Systems
Source systems vary widely in terms of the type of data they generate—structured, semi-structured, or unstructured—and how frequently this data is updated.

:p What are the different types of data generated by source systems?
??x
Source systems can generate three main types of data: structured (e.g., relational databases), semi-structured (e.g., JSON, XML), and unstructured (e.g., text files, images). Each type requires specific handling and processing techniques.
x??

---
#### Real-Time Data Generation
Real-time data generation is common in applications and IoT devices where the data must be processed immediately or near-immediately.

:p How does real-time data generation differ from other types of data generation?
??x
Real-time data generation involves generating data as it happens, often requiring immediate processing. This contrasts with batch processing, where data is collected over a period before being processed.
x??

---
#### Batch Data Generation
Batch data generation occurs at specific intervals and can be used for periodic updates or large-scale data processes.

:p What is the typical scenario for batch data generation?
??x
Batch data generation is suitable for scenarios where data is collected periodically (e.g., daily, weekly) before being processed. This method is often used for large-scale data processing tasks.
x??

---
#### Data Quirks and Considerations
When working with source systems, it’s important to understand any quirks or peculiarities in the data generation process that might affect data quality.

:p Why is understanding the quirks of source systems crucial?
??x
Understanding quirks such as inconsistent data formats, missing values, or unexpected data types is crucial because these can significantly impact data quality and downstream processes. It helps in designing appropriate ETL (Extract, Transform, Load) strategies.
x??

---
#### Undercurrents of Data Engineering
The principles of data engineering apply throughout the lifecycle, including this initial phase where data is generated.

:p How do the undercurrents of data engineering play a role in the first stage?
??x
The undercurrents of data engineering, such as data quality management and efficient data storage strategies, play a critical role even at the source system level. Ensuring these principles are adhered to from the start can streamline processes later on.
x??

---

#### Analog Data Creation
Analog data creation occurs in the real world, such as vocal speech, sign language, writing on paper, or playing an instrument. This type of data is often transient; for instance, a verbal conversation's contents are usually lost after it ends.

:p What characterizes analog data creation?
??x
Analog data creation involves generating data through physical processes in the real world, such as spoken words, written notes, or musical performances. These forms of data are not stored or preserved in digital format and tend to be ephemeral.
x??

---

#### Digital Data Creation
Digital data is either created by converting analog data into a digital form or directly from a digital system. For example, a mobile app converts speech (analog) into text (digital), while an ecommerce platform records transactions as digital data.

:p What are the two main ways to create digital data?
??x
Digital data can be created in two primary ways:
1. By converting analog data: An example is how a mobile texting app transforms spoken words or written notes into digital text.
2. As the native product of a digital system: A transaction on an ecommerce platform creates digital records, such as charge details and order information stored in databases.
x??

---

#### Data Generation from Source Systems
Source systems produce data through various mechanisms. It is crucial to understand how these systems generate data to effectively manage data pipelines.

:p How do you capture analog data?
??x
Analog data is generated in the real world, such as by speaking, writing, or playing an instrument. For instance, a conversation's contents are lost after it ends, making this type of data transient.
x??

---

#### Data Generation from Source Systems (Continued)
Understanding how source systems generate data involves examining their operational patterns and quirks.

:p What is the significance of understanding the nature of data generated in source systems?
??x
Understanding the nature of data created in source systems is crucial because it helps data engineers design effective data pipelines, handle data inconsistencies, and ensure proper data ingestion. This knowledge enables better data management by recognizing unique characteristics or issues specific to each system.
x??

---

#### Files and Unstructured Data
Files are sequences of bytes typically stored on disks. Applications often write data to files which can contain various types of information like local parameters, events, logs, images, and audio.

:p What are examples of unstructured data?
??x
Examples of unstructured data include text documents, emails, social media posts, images, videos, audio recordings, etc. These forms of data do not have a predefined format or structure.
x??

---

#### RDBMS Source Systems
RDBMS (Relational Database Management System) source systems store and manage structured relational data. Understanding their operations is vital for effective data ingestion.

:p What are some key operations of an RDBMS?
??x
Key operations of an RDBMS include:
- Writing data: Inserting new records into the database.
- Committing changes: Saving transactions to ensure atomicity.
- Querying data: Retrieving information from the database using SQL commands.
These operations are essential for understanding how data is ingested and managed within an RDBMS.

```java
// Example of a simple insert operation in Java using JDBC
public void writeDataToDatabase(String sql) {
    try (Connection conn = DriverManager.getConnection(url, user, password);
         Statement stmt = conn.createStatement()) {
        int rowsAffected = stmt.executeUpdate(sql);
        System.out.println(rowsAffected + " row(s) inserted.");
    } catch (SQLException e) {
        e.printStackTrace();
    }
}
```
x??

