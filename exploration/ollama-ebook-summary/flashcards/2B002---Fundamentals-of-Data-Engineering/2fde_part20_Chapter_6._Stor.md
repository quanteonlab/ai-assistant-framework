# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 20)

**Starting Chapter:** Chapter 6. Storage

---

#### Networking Considerations
Understanding how your application will communicate over the network is crucial for accessing source systems. You need to consider whether you are working with HTTP(s), SSH, or a VPN.

:p What factors should be considered when designing networking access for a data engineering task?
??x
When designing networking access for a data engineering task, you should consider several key factors:
- The protocol used (HTTP/HTTPS, SSH, etc.)
- Authentication and authorization mechanisms
- Security considerations such as encryption and secure tunnels like a VPN

Consider the following code snippet to illustrate checking if an HTTPS request is successful:

```java
import java.net.HttpURLConnection;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class NetworkAccess {
    public boolean checkHttpsRequest(String url) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(url).openConnection();
        try {
            int responseCode = connection.getResponseCode();
            if(responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream()));
                String inputLine;
                while ((inputLine = in.readLine()) != null) {}
                return true;
            } else {
                return false;
            }
        } finally {
            connection.disconnect();
        }
    }
}
```
x??

---

#### Authentication and Authorization
Proper credentials management is essential to securely access source systems. Storing sensitive information like tokens or passwords should be handled carefully.

:p How can you manage authentication and authorization in a data engineering project?
??x
Managing authentication and authorization involves several steps:
- Use secure methods to store credentials such as environment variables, secrets managers, or encrypted files.
- Ensure that IAM roles are correctly configured to grant necessary permissions for specific tasks.
- Avoid hardcoding sensitive information directly into your application code.

Example of using environment variables in Java:

```java
public class AuthManager {
    private final String username;
    private final String password;

    public AuthManager(String envVarName) {
        this.username = System.getenv(envVarName + "_USERNAME");
        this.password = System.getenv(envVarName + "_PASSWORD");
    }
}
```
x??

---

#### Access Patterns
Understanding the data access patterns, such as using APIs or database drivers, is critical for efficient and secure data retrieval.

:p What are some key considerations when accessing source systems through APIs?
??x
Key considerations when accessing source systems through APIs include:
- Using REST/GraphQL requests appropriately.
- Handling response volumes and pagination to avoid performance issues.
- Implementing retry logic and timeouts to handle transient errors gracefully.

Example of handling retries in Java:

```java
public class RetryHandler {
    public void makeApiCall() {
        int attempts = 0;
        final int MAX_RETRIES = 5;
        
        while (attempts < MAX_RETRIES) {
            try {
                // Make API call here
                break;
            } catch (IOException e) {
                attempts++;
                if (attempts >= MAX_RETRIES) {
                    throw new RuntimeException("Failed to make API call after " + attempts + " retries", e);
                }
                // Wait before retrying
                Thread.sleep(1000);
            }
        }
    }
}
```
x??

---

#### Orchestration and Parallelization
Orchestrating tasks and managing parallel access are essential for efficient data engineering workflows.

:p How can you ensure that your code integrates with an orchestration framework?
??x
Integrating your code with an orchestration framework involves:
- Using standardized APIs provided by the framework.
- Configuring workflows to manage dependencies and task sequencing.
- Ensuring that tasks can be executed in parallel while managing concurrency issues.

Example of a simple workflow using Apache Airflow:

```java
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def my_task():
    # Task logic here
    pass

dag = DAG('my_dag', start_date=datetime(2023, 1, 1))

task1 = PythonOperator(
    task_id='task_1',
    python_callable=my_task,
    dag=dag
)
```
x??

---

#### Deployment Strategies
Deployment strategies are critical for rolling out changes to your data engineering codebase without disrupting operations.

:p How can you handle the deployment of source code changes in a production environment?
??x
Handling deployments involves:
- Using CI/CD pipelines to automate testing and release processes.
- Implementing rollback mechanisms if something goes wrong during deployment.
- Monitoring deployed systems for issues post-deployment.

Example of a simple CI/CD pipeline using Jenkins:

```groovy
pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building the application'
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying to production'
            }
        }
    }
}
```
x??

---

#### Data Storage Considerations
Storage is a fundamental aspect of data engineering, affecting every stage of the lifecycle from ingestion to serving.

:p Why is understanding storage crucial in data engineering?
??x
Understanding storage is crucial because:
- It influences how and where data is stored.
- Proper storage choices can optimize performance and reduce costs.
- Different use cases require different types of storage solutions.

Example of choosing a suitable storage solution based on use case:

```java
public class StorageSelector {
    public String chooseStorage(String dataFrequency, int volume) {
        if (dataFrequency.equals("real-time") && volume > 100000) {
            return "Kafka";
        } else if (dataFrequency.equals("daily") && volume < 5000) {
            return "SQLite";
        }
        // Add more conditions as needed
        return null;
    }
}
```
x??

---

---
#### Magnetic Disk Drive Technology
Background context explaining the technology behind magnetic disk drives, including how data is encoded and decoded using a read/write head. Discuss the physical components like spinning platters and ferromagnetic film.

:p What are the key components of a magnetic disk drive?
??x
The key components of a magnetic disk drive include spinning platters coated with a ferromagnetic film, and a read/write head that magnetizes the film during write operations and detects the magnetic field for reads. The read/write head moves across tracks on the platter to access data.
```java
// Pseudocode to illustrate movement and operation of a read/write head
class MagneticDisk {
    void writeData(byte[] data) {
        // Code to magnetize the film with binary data
    }
    
    byte[] readData() {
        // Code to detect magnetic field for bit-stream output
        return new byte[1024];
    }
}
```
x??

---
#### Performance Metrics of Magnetic Disk Drives
Background context explaining the performance metrics such as disk transfer speed, areal density, and linear density. Discuss how these metrics affect data access times.

:p What factors influence the performance of magnetic disk drives?
??x
The performance of magnetic disk drives is influenced by several key factors including:
- **Disk Transfer Speed**: The rate at which data can be read or written.
- **Areal Density**: The amount of gigabits stored per square inch, which scales with capacity.
- **Linear Density**: The number of bits per inch, which affects transfer speed.

These metrics are interrelated; as capacity increases (areal density), transfer speed does not scale proportionally due to linear density limitations. For example, doubling the areal density may only result in a 2x increase in transfer speed.
```java
// Pseudocode for calculating performance metrics
class DiskPerformance {
    int arealDensity = calculateArealDensity();
    int linearDensity = calculateLinearDensity();
    
    double transferSpeed() {
        // Transfer speed is not directly proportional to capacity due to density constraints
        return Math.sqrt(arealDensity) * linearDensity;
    }
}
```
x??

---
#### Seek Time and Rotational Latency
Background context explaining the concepts of seek time and rotational latency, which affect data access times in magnetic disk drives. Discuss how these metrics impact overall performance.

:p What are seek time and rotational latency?
??x
Seek time refers to the time it takes for the read/write heads to move from their current position to the desired location on the disk's platter. Rotational latency is the delay experienced as data must wait to pass under the read/write heads, since the disk rotates continuously.

Both seek time and rotational latency significantly impact overall performance; they can introduce substantial delays, especially for random access operations.
```java
// Pseudocode illustrating seek time and rotational latency
class SeekTimeLatency {
    int seekTime = 5000; // Time in milliseconds to move heads to a specific track
    
    double rotationalLatency(double rotationSpeed) {
        // Rotational speed is given in RPM (Revolutions Per Minute)
        return 60 / rotationSpeed;
    }
}
```
x??

---

#### Magnetic Drives vs. SSDs
Background context: The text compares magnetic drives (HDDs) and solid-state drives (SSDs) in terms of performance, cost, and usage scenarios. Key differences include latency, IOPS, transfer speeds, and application suitability.

:p What are the main differences between magnetic drives (HDDs) and SSDs?
??x
Magnetic drives (HDDs) offer lower latency, higher IOPS, and faster transfer rates compared to solid-state drives (SSDs), but at a much higher cost. HDDs are favored for large-scale data storage due to their lower cost per gigabyte, while SSDs excel in applications requiring high-speed random access.
x??

---
#### Latency and IOPS of Magnetic Drives
Background context: The text discusses the performance metrics of magnetic drives, including latency and IOPS.

:p What is the average latency for accessing a piece of data on a magnetic drive?
??x
The average latency for accessing a piece of data on a magnetic drive is over four milliseconds.
x??

---
#### Input/Output Operations per Second (IOPS)
Background context: The text mentions that input/output operations per second (IOPS) are crucial for transactional databases, with magnetic drives ranging from 50 to 500 IOPS.

:p What does IOPS measure in the context of storage systems?
??x
IOPS measures the number of read or write operations a drive can perform per second. In the context of storage systems, it is particularly important for transactional databases.
x??

---
#### Seek Time and Rotational Latency
Background context: The text explains how techniques like using higher rotational speed and limiting disk platter radius can reduce seek time and rotational latency in magnetic drives.

:p How can seek time be reduced in a magnetic drive?
??x
Seek time can be reduced by increasing the rotational speed of the magnetic drive, limiting the radius of the disk platter, or writing data into only a narrow band on the disk.
x??

---
#### Cloud Object Storage and Parallelism
Background context: The text highlights how cloud object storage distributes data across thousands of disks to achieve high transfer rates.

:p How does parallelism improve data transfer rates in cloud object storage?
??x
Parallelism improves data transfer rates in cloud object storage by reading from numerous disks simultaneously, which is limited primarily by network performance rather than disk transfer rate.
x??

---
#### Solid-State Drive (SSD) Characteristics
Background context: The text describes the features of SSDs, including their ability to deliver lower latency and higher IOPS compared to magnetic drives.

:p What are some key characteristics of solid-state drives (SSDs)?
??x
Key characteristics of SSDs include:
- Lower average latency (less than 0.1 ms)
- Higher IOPS (tens of thousands per second)
- Higher transfer speeds (many gigabytes per second)
- No physical rotating disk or magnetic head, leading to faster access times.
x??

---
#### Random Access Memory (RAM) vs. Magnetic Drives and SSDs
Background context: The text contrasts RAM with magnetic drives and SSDs in terms of performance and cost.

:p What are the main differences between RAM and magnetic drives/SSDs?
??x
The main differences include:
- RAM offers significantly higher transfer speeds and faster retrieval times.
- RAM is volatile, losing data when unpowered.
- RAM is more expensive per gigabyte but offers much higher bandwidth and IOPS compared to SSDs.
- RAM is limited in the amount of memory attached to an individual CPU and memory controller.
x??

---
#### DDR5 Memory Characteristics
Background context: The text explains the characteristics of DDR5 memory, a type of dynamic random-access memory (DRAM).

:p What are some key features of DDR5 memory?
??x
Key features of DDR5 memory include:
- Data retrieval latency on the order of 100 ns.
- Support for data retrieval at up to 100 GB/s bandwidth and millions of IOPS per CPU.
- Significantly higher cost compared to SSD storage, around $10/GB.
- Limited by the amount of RAM attached to an individual CPU and memory controller.
x??

---
#### Cache in CPUs
Background context: The text describes CPU cache as a type of memory that stores frequently accessed data for ultrafast retrieval during processing.

:p What is CPU cache?
??x
CPU cache is a small, fast memory located directly on the CPU die or in the same package. It stores frequently and recently accessed data to facilitate ultrafast retrieval during processing.
x??

---

