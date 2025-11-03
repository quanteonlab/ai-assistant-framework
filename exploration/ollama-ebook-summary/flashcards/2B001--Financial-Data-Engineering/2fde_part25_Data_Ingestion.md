# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 25)

**Starting Chapter:** Data Ingestion Technologies. Financial APIs

---

#### SWIFT Transfer Process
Background context: This section explains how SWIFT transfers work between banks. It covers both direct and indirect transfer processes, emphasizing the role of intermediary or correspondent banks when there is no direct relationship between two banking entities.

:p What are the steps involved in a SWIFT transfer if Bank 1 and Bank 2 do not have a corresponding banking relationship?
??x
In this scenario, the payment process involves one or more intermediary or correspondent banks. These banks act as intermediaries to facilitate the exchange of messages such as MT101 (Credit Advice) and MT103 (Customer Credit Transfer). The primary objective is to ensure that funds are transferred from the sender's bank (Bank 1) to the receiver's bank (Bank 2) through a series of transactions involving these intermediary banks.
x??

---

#### SWIFT GPI - Global Payments Innovation
Background context: SWIFT introduced SWIFT GPI to enhance cross-border payment experiences. This initiative focuses on increasing speed, traceability, and transparency in international payments.

:p How does SWIFT GPI address the industry's demands for greater speed, traceability, and transparency?
??x
SWIFT GPI addresses these demands by providing end-to-end tracking of payments from originator to beneficiary, ensuring faster settlement times and more detailed transaction information. This initiative allows customers to receive real-time updates on their payment status, reducing uncertainty and improving the overall customer experience.
x??

---

#### Data Ingestion Technologies Overview
Background context: Financial data infrastructure requires robust mechanisms for ingesting large volumes of data efficiently. APIs are a key technology used for integrating such capabilities.

:p What is an API (Application Programming Interface) in the context of financial data ingestion?
??x
An API allows one software component to interact with another by defining rules, protocols, and methods for interaction between two software types. In the context of financial data ingestion, APIs enable applications to exchange structured data securely and efficiently.
x??

---

#### REST APIs
Background context: REST APIs are a popular choice due to their simplicity and wide support across various programming languages.

:p What is a key characteristic of REST APIs that differentiates them from other API types?
??x
A key characteristic of REST APIs is the use of HTTP methods like GET, POST, and PUT. Specifically, the difference between PUT and POST lies in idempotency: making multiple identical PUT requests will always produce the same result, whereas POST can potentially lead to state changes with each request.
x??

---

#### Python API Implementation
Background context: Python is a common choice for data engineers due to its versatility and ease of use. Flask and FastAPI are popular frameworks for implementing web-based APIs in Python.

:p How does the PUT method ensure idempotency in REST APIs?
??x
The PUT method ensures idempotency by allowing multiple identical requests to be safely repeated without changing their effect beyond the initial application. In other words, making the same PUT request multiple times should produce the same result each time.
x??

---

#### Example of a Simple API Using Flask
Background context: This example demonstrates how to implement a simple RESTful API using the Flask framework in Python.

:p What is the purpose of defining routes and methods in a Flask application?
??x
Defining routes and methods in a Flask application allows you to map URLs to specific functions that handle HTTP requests. Each route can have associated methods (e.g., GET, POST) that determine how the server responds to incoming requests.
```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/api/payment', methods=['POST'])
def process_payment():
    # Process payment logic here
    return "Payment processed"

if __name__ == '__main__':
    app.run(debug=True)
```
x??

---

These flashcards cover key concepts and provide context, explanations, and examples where relevant.

#### OpenFIGI API for Financial Instrument Mapping
Background context explaining the concept: The OpenFIGI API is a powerful tool used to map financial instruments using their unique identifiers. It helps in resolving ambiguities and inconsistencies that can arise due to different naming conventions, ticker symbols, or exchange codes. This is particularly useful in finance where interoperability between different systems is crucial.

:p What does the provided Bash `curl` command do?
??x
The provided `curl` command makes a POST request to the OpenFIGI API to fetch mapping information for the financial instrument with the ticker symbol AAPL on the New York Stock Exchange (UN). The data being sent includes an identifier type (`TICKER`) and its corresponding value.

```bash
curl -X POST \
      -H "Content-Type: application/json" \
      -d '[{"idType": "TICKER", "idValue": "AAPL", "exchCode": "UN"}]' \
      https://api.openfigi.com/v2/mapping
```

x??

---

#### Payment Gateway APIs Overview
Background context explaining the concept: Payment gateway APIs facilitate electronic payments between merchants and financial institutions. They handle various types of payments such as credit cards, debit cards, digital wallets, and bank transfers. The core component of these systems is an API that manages different phases of the payment lifecycle.

:p What are some examples of well-known payment gateway providers?
??x
Some well-known payment gateway providers include Square, Stripe, PayPal, Authorize.Net, and Adyen. These platforms offer APIs to enable merchants to process payments securely and efficiently.

x??

---

#### Financial Data Vendor API Usage
Background context explaining the concept: Financial data vendors provide APIs that allow clients to programmatically retrieve and import financial data. This is essential for developing data-driven applications in finance. Examples include Bloomberg's SAPI, LSEG's Eikon Data API, and FactSet's Formula API.

:p How can you ensure smooth integration with a financial data vendor’s API?
??x
To ensure smooth integration with a financial data vendor’s API, it is crucial to familiarize yourself with any limitations such as single-request size limits (e.g., 100 prices per request), request rate limits (e.g., 1K requests/day), request timeout, and maximum concurrent requests. Always check the vendor's official documentation or web page for specific limitations.

x??

---

#### Example Code for API Request
Background context explaining the concept: The provided code examples illustrate how to make API requests in a programming language like Java or Bash.

:p How can you simulate making an API request using a simple example?
??x
Here is a simple example of making an HTTP POST request using Java with HttpClient. This example demonstrates sending data to a financial data vendor's API and handling the response.

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class ApiExample {
    public static void main(String[] args) throws Exception {
        HttpClient client = HttpClient.newHttpClient();
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("https://api.example.com/v2/mapping"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString("[{\"idType\": \"TICKER\", \"idValue\": \"AAPL\", \"exchCode\": \"UN\"}]"))
                .build();
        
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        System.out.println(response.body());
    }
}
```

x??

---

#### Open Finance Initiatives
Open finance initiatives seek to establish a digital ecosystem where financial data can be shared seamlessly between financial institutions and third-party service providers. This approach aims to foster collaboration among market participants, leading to the development of innovative financial products and services driven by financial data.

:p What are open finance initiatives?
??x
Open finance initiatives aim to create an ecosystem that allows for the seamless sharing of financial data between banks, other financial institutions, and third-party service providers. These initiatives promote collaboration to develop better financial products and services using data.
x??

---

#### Open Banking
A specific implementation of open finance, open banking involves traditional banking institutions collaborating with FinTech firms to offer innovative financial products by sharing data through ad hoc financial APIs.

:p What is the definition of open banking?
??x
Open banking refers to a framework that allows banks and other financial institutions to share customer data securely with third-party service providers, enabling the development of new financial services and products.
x??

---

#### Account Information Service Providers (AISPs)
AISPs are companies licensed under frameworks like PSD2 in Europe. They collect, aggregate, and facilitate access to a user's financial information across multiple institutions.

:p What is an AISP?
??x
An AISP is a company that has been granted a license by regulatory bodies such as PSD2 in the European Union. Their primary function is to collect, aggregate, and provide access to a customer’s financial data from various bank accounts.
x??

---

#### Payment Initiation Service Providers (PISPs)
PISPs enable direct payments from a consumer's bank account to online merchants through facilitated payment initiation.

:p What is a PISP?
??x
A PISP facilitates the process of initiating payments directly from a customer’s bank account to an online merchant. They play a role in ensuring secure and authorized transactions between banks and merchants.
x??

---

#### Financial APIs for Open Banking
Financial APIs are essential tools designed with business, user, and application requirements in mind. JPMorgan Chase categorizes financial APIs into data and service APIs.

:p What are financial APIs?
??x
Financial APIs are software interfaces that allow third-party developers to access financial services and data from banks or other financial institutions securely. They can be classified into data APIs for requesting financial information and service APIs for enabling various financial services.
x??

---

#### API Design Considerations in Open Banking
APIs should be designed with business, user, and application requirements in mind to facilitate collaboration and innovation.

:p How should APIs be designed?
??x
APIs should be designed considering the needs of the business, users, and applications. This ensures that they are not only functional but also reusable across different departments, channels, and product lines, fostering innovation and collaboration.
x??

---

#### Regulatory Frameworks for Open Banking
Regulatory frameworks such as PSD2 in Europe mandate banks to facilitate third-party access to payment data securely.

:p What regulatory framework supports open banking?
??x
The second Payment Services Directive (PSD2), adopted by the EU in 2015, mandates that banks must provide secure and facilitated access to payment data for third-party service providers.
x??

---

#### Platforms for Open Banking
Companies like Tink and Powens offer platforms that connect various financial institutions, facilitating open finance ecosystems.

:p What companies support open banking?
??x
Companies such as Tink, a Swedish company acquired by Visa, and Powens, a French company connecting over 1,800 institutions in Europe, provide platforms for open banking. These platforms enable seamless connections between financial institutions and third-party service providers.
x??

---

---
#### Service APIs Overview
Service APIs are used to create and trigger instances of services, such as initiating a payment or balance inquiry. They facilitate seamless data exchange and communication between applications through an API infrastructure.
:p What is the primary use case for service APIs?
??x
Service APIs are primarily used to initiate specific actions, such as payments or balance inquiries, by creating and triggering instances of these services. This enables applications to interact with backend systems in a controlled and secure manner.
x??

---
#### API Integration Strategy
API integration involves connecting different applications using their APIs to facilitate data exchange and communication, thereby enabling creativity and innovation within the infrastructure.
:p What is the goal of an API integration strategy?
??x
The goal of an API integration strategy is to create a seamless infrastructure for data exchange and communication between various applications, fostering creativity and innovation through interconnected systems.
x??

---
#### Performance Metrics for APIs
API performance is measured by its ability to handle large numbers of concurrent requests and the request response time. Common metrics include hits per second (HPS) and requests per second (RPS).
:p What are common API performance metrics?
??x
Common API performance metrics include hits per second (HPS) and requests per second (RPS). These metrics measure the ability of an API to handle a large number of concurrent requests in one second.
x??

---
#### Performance Optimization Techniques
Performance optimization techniques for APIs include load balancing, caching, rate limiting, and throttling. These methods help manage and control high volumes of traffic efficiently.
:p What are some common performance optimization techniques?
??x
Common performance optimization techniques for APIs include:
- Load Balancing: Distributes incoming network traffic across multiple servers to ensure no single server is overwhelmed.
- Caching: Stores frequently accessed data in temporary storage (cache) to reduce the number of requests to the backend and improve response times.
- Rate Limiting: Limits the rate at which clients can request resources from the API, preventing abuse or overwhelming the system.
- Throttling: Similar to rate limiting but often used for more fine-grained control over access.

Example pseudocode for rate limiting:
```python
def rate_limit(user_ip):
    if user_ip in rate_limited_ips:
        return False
    else:
        add_user_to_rate_limited_ips(user_ip)
        return True
```
x??

---
#### Security Elements of APIs
Security elements include authentication and authorization to control how and who can interact with the API. Tools like firewalls, OAuth 2.0, API keys, and API gateways are commonly used for securing APIs.
:p What are key security considerations when designing an API?
??x
Key security considerations when designing an API involve:
- Authentication: Ensuring that only authorized entities can access the API. Common methods include API keys, tokens, and OAuth 2.0.
- Authorization: Controlling what actions users or applications are allowed to perform once authenticated.

Example pseudocode for simple authentication using API keys:
```java
public boolean authenticate(String apiKey) {
    if (apiKey.equals(secretApiKey)) {
        return true;
    } else {
        return false;
    }
}
```
x??

---
#### SQL Injection Attack
An SQL injection attack allows a cybercriminal to exploit vulnerabilities in an application’s input validation mechanisms, injecting malicious inputs that alter the behavior of backend SQL queries.
:p What is SQL injection and how does it work?
??x
SQL injection (SQLi) is an attack where a cybercriminal exploits vulnerabilities in an application's input validation mechanisms by injecting malicious SQL statements. This can lead to unauthorized access to data or even complete system compromise.

Example:
A naive user ID validation mechanism could be vulnerable to SQLi if not properly sanitized.
```java
String userId = request.getParameter("user_id");
// Vulnerable query
String sqlQuery = "SELECT first_name, last_name, account_balance FROM user_accounts WHERE user_id = " + userId;
```
??x
In the above example, if `userId` is directly concatenated into the SQL query without sanitization, an attacker could inject malicious input like `"267 OR 1=1"`, leading to a SQL injection attack.

To prevent this, use prepared statements:
```java
String sqlQuery = "SELECT first_name, last_name, account_balance FROM user_accounts WHERE user_id = ?";
PreparedStatement statement = connection.prepareStatement(sqlQuery);
statement.setString(1, userId);
ResultSet rs = statement.executeQuery();
```
x??

---

#### SQL Injection Analysis
Background context: The provided text references a seminal paper by William G. Halfond, Jeremy Viegas, and Alessandro Orso, which classifies SQL injection attacks and countermeasures. Additionally, it mentions that an SQL condition like `1=1` always evaluates to TRUE, making the entire WHERE statement true regardless of user input.
:p Explain the impact of using `1=1` in a SQL query context?
??x
When `1=1` is used in a SQL query's WHERE clause, it bypasses any conditional logic because it will always evaluate to true. This means that any subsequent conditions or filters are ignored, leading to the full dataset being queried instead of just the intended records.
For example:
```sql
SELECT * FROM users WHERE 1=1 AND user_id = 'user_input'
```
If `user_input` is not properly sanitized and contains `1' OR '1'='1`, the query becomes:
```sql
SELECT * FROM users WHERE 1=1 AND user_id = '1' OR '1'='1'
```
This results in all records being returned, potentially leading to data breaches.
x??

---

#### Financial Data Feeds Overview
Background context: The text provides an overview of financial data feeds used in the financial industry. These feeds deliver real-time or historical financial data to traders and institutions via various sources such as stock exchanges, news providers, and market data vendors. Key characteristics include latency, throughput, and delivery guarantees.
:p What are some examples of financial data feeds mentioned in the text?
??x
Examples of financial data feeds include:
- S&P Global’s Xpressfeed: Offers access to over 200 datasets and allows customization.
- LSEG’s Real-Time – Ultra Direct: Provides high-performance, low-latency real-time market data.
- Bloomberg Market Data Feed (B-PIPE): A comprehensive financial data feed from Bloomberg.
- NYSE Trades Data Feed and NASDAQ Market Data Feeds: Streaming trading data directly from the exchanges with minimal latency.
- MT News Wires: Real-time news headlines and text.
x??

---

#### Data Ingestion Challenges
Background context: The text discusses challenges in managing large volumes of financial data, particularly when using multiple data feeds. Cloud technology helps mitigate issues like information overload by scaling storage and retrieval capabilities without manual infrastructure management.
:p What challenge does cloud technology help address in the context of financial data feeds?
??x
Cloud technology addresses the challenge of information overload, which occurs when a high volume of data from one or more feeds overwhelms existing infrastructure. By leveraging cloud services, financial institutions can scale their storage and retrieval capabilities dynamically without needing to manage physical hardware.
x??

---

#### Real-Time Data Feeds
Background context: The text explains that real-time data feeds provide continuous streams of current financial information essential for timely decision-making in trading and other financial activities. These feeds can be configured by users to specify the timing, location, and specific data points needed.
:p What are some examples of features offered by real-time financial data feeds?
??x
Real-time financial data feeds often offer features such as:
- Customizable data extraction: Allowing users to select which datasets they need.
- Adjustable delivery locations: Specifying where the data should be sent or stored.
For example, LSEG’s Real-Time – Ultra Direct allows for high-performance, low-latency delivery of market data. Bloomberg's B-PIPE also provides a wide range of configurable options for financial institutions.
x??

---

#### Historical Data Feeds
Background context: While real-time feeds are crucial in the financial industry, historical data feeds provide valuable insights into past performance and trends. These feeds can be used for analysis, backtesting strategies, and understanding market behavior over time.
:p How do historical data feeds differ from real-time data feeds?
??x
Historical data feeds differ from real-time data feeds in that they deliver pre-recorded financial data at a specific point in time rather than providing live updates. This makes them useful for analysis, backtesting trading strategies, and understanding market behavior retrospectively.
For example, while real-time feeds might provide the latest stock prices and trades as events happen, historical feeds would offer snapshots of these transactions at earlier points in time, allowing analysts to review past performance.
x??

---

#### Data Latency and Throughput
Background context: The text highlights that financial data feeds vary in terms of latency (the delay between a trigger event and the availability of data) and throughput (the rate at which data is delivered). Low latency is crucial for real-time trading, while high throughput ensures large volumes of data can be processed efficiently.
:p Explain the importance of low latency in financial data feeds.
??x
Low latency in financial data feeds is critical because it reduces the delay between when a market event occurs and when the data becomes available to traders. This is essential in fast-paced trading environments where quick reaction times can significantly impact profitability. For instance, a stock price change needs to be reflected immediately so that traders can make informed decisions.
x??

---

#### Data Feed Delivery Guarantees
Background context: The text mentions that financial data feeds offer delivery guarantees, ensuring reliability and integrity of the data being transmitted. These guarantees are important for maintaining trust in the data and enabling robust trading strategies.
:p What does a delivery guarantee ensure in the context of financial data feeds?
??x
A delivery guarantee ensures that the data delivered by financial data feeds is accurate, timely, and complete. This means that traders can rely on the integrity of the data they receive, which is crucial for making informed decisions and developing effective trading strategies.
For example, a high-quality market data feed would have guarantees around data accuracy, timeliness, and completeness to ensure that traders are not misled by erroneous or outdated information.
x??

---

#### Secure File Transfer Protocol (SFTP)
Background context: SFTP is a widely used protocol for secure file transfer, leveraging SSH to encrypt both data and commands. It offers security, reliability, and platform independence, making it suitable for bulk and large file transfers. However, it may not be the best option in high-speed and large-volume systems due to its slower performance compared to alternatives.
If applicable, add code examples with explanations:
```java
// Example of an SFTP setup using Java
import com.jcraft.jsch.ChannelSftp;
import com.jcraft.jsch.JSch;
import com.jcraft.jsch.Session;

public class SFTPExample {
    public void transferFile(String hostname, int port, String user, String password, String remotePath) throws Exception {
        JSch jsch = new JSch();
        Session session = jsch.getSession(user, hostname, port);
        session.setPassword(password);

        // Avoid asking for key confirmation
        java.util.Properties config = new java.util.Properties();
        config.put("StrictHostKeyChecking", "no");
        session.setConfig(config);
        session.connect();

        ChannelSftp channelSftp = (ChannelSftp) session.openChannel("sftp");
        channelSftp.connect();
        // Further code to transfer files
    }
}
```
:p What is SFTP and how does it work?
??x
SFTP stands for Secure File Transfer Protocol, which uses SSH (Secure Shell) protocol for secure file transfers. It encrypts both the data and commands sent between machines, ensuring security during the transfer process. This makes it suitable for transferring sensitive financial information.
x??

---

#### Managed File Transfer Solutions (MFT)
Background context: MFT solutions enhance SFTP with additional enterprise-level functionalities such as enhanced security, performance optimization, compliance management, and advanced reporting. These solutions are designed to simplify complex file transfer processes while maintaining high standards of security and efficiency.
If applicable, add code examples with explanations:
```java
// Example of using a managed file transfer solution (MFT)
import com.example.managed.file.transfer.MFTClient;

public class MFTExample {
    public void initiateFileTransfer(String sourcePath, String destinationPath) throws Exception {
        MFTClient mftClient = new MFTClient();
        // Configure the client with necessary details
        mftClient.configureSource(sourcePath);
        mftClient.configureDestination(destinationPath);

        // Start the file transfer process
        mftClient.transferFiles();
    }
}
```
:p What is Managed File Transfer (MFT) and how does it differ from SFTP?
??x
Managed File Transfer (MFT) solutions extend the functionality of standard SFTP by providing enhanced security, performance optimization, compliance management, and advanced reporting capabilities. They simplify complex file transfer processes while maintaining high standards.
x??

---

#### Cloud-Based Data Sharing and Access
Background context: Cloud-based data sharing and access offer a reliable and convenient method for exchanging data between entities. Users can leverage various cloud features when working with the data, such as user interfaces, querying capabilities, data management, search functions, and more. This approach also offers cost-saving benefits and seamless integration with other cloud services.
If applicable, add code examples with explanations:
```java
// Example of using Google Cloud for accessing data
import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;
import com.google.api.client.http.HttpRequest;

public class CloudDataAccess {
    public void downloadFileFromCloud(String bucketName, String fileName) throws Exception {
        Storage storage = StorageOptions.getDefaultInstance().getService();
        // Download the file from Google Cloud Storage
        HttpRequest req = storage.get(bucketName, fileName).executeMediaAsInputStream();
        // Further code to process or save the downloaded file
    }
}
```
:p How does cloud-based data sharing and access work?
??x
Cloud-based data sharing and access involves creating a storage bucket or database within a dedicated and isolated cloud environment. The provider uploads data, and the target user is authorized to access and manipulate it. Updates are continuously pushed to the storage location, providing immediate access for users.
x??

---

#### Case Study: FactSet Integration with AWS Redshift and Snowflake
Background context: FactSet integrates its financial datasets into popular cloud data warehouse services like AWS Redshift and Snowflake, offering a centralized platform for accessing and querying data. This integration saves clients the need to clean, model, and normalize data manually.
If applicable, add code examples with explanations:
```java
// Example of using AWS Redshift
import com.amazonaws.services.redshift.AmazonRedshift;
import com.amazonaws.services.redshift.AmazonRedshiftClientBuilder;

public class RedshiftExample {
    public void queryData(String query) throws Exception {
        AmazonRedshift redshift = AmazonRedshiftClientBuilder.defaultClient();
        // Execute the SQL query
        String result = redshift.executeStatement(query).getRows().toString();
        System.out.println("Query Result: " + result);
    }
}
```
:p How does FactSet's cloud-based data delivery work?
??x
FactSet integrates its financial datasets into popular cloud data warehouse services like AWS Redshift and Snowflake, making the data ready for querying. Clients can access this pre-populated data without needing to clean, model, or normalize it, simplifying workflow management.
x??

---

#### Financial Data Marketplaces in Cloud Computing
Background context: Financial data marketplaces are managed cloud solutions that allow financial data providers to distribute and share their data through a single cloud interface. This eliminates the need for providers to build and maintain infrastructure for storage, distribution, billing, and user management.
If applicable, add code examples with explanations:
```java
// Example of using AWS Data Exchange for Financial Services
import com.amazonaws.services.datasync.AWSDataSync;

public class DataExchangeExample {
    public void subscribeToDataset(String datasetArn) throws Exception {
        AWSDataSync dataSync = AWSDataSyncClientBuilder.defaultClient();
        // Subscribe to the dataset
        dataSync.subscribeToDataset(datasetArn);
    }
}
```
:p What are financial data marketplaces and how do they work?
??x
Financial data marketplaces are managed cloud solutions that allow financial data providers to distribute and share their data through a single cloud interface, eliminating the need for them to build and maintain infrastructure.
x??

---

