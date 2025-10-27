# Flashcards: 2B002---Fundamentals-of-Data-Engineering_processed (Part 8)

**Starting Chapter:** Principle 9 Embrace FinOps

---

#### Distributed System Design - Loose Coupling and Iterative Improvement
Background context: The passage describes a distributed system where many small teams independently work on different components of a large, complex system. These teams communicate via APIs and messages, enabling them to evolve their parts without worrying about others' internal details.

:p What is the principle behind each team's ability to rapidly improve its component independently?
??x
This principle leverages loose coupling, which means that changes in one part do not affect other parts directly. Teams can make iterative improvements to their components and publish new features or updates with minimal impact on the system as a whole.

The key is that teams are responsible for only their specific module's details but rely on published APIs to interact with others, ensuring flexibility and independent evolution.
x??

---

#### Reversible Decisions in Architecture
Background context: The text emphasizes making reversible decisions to keep architecture agile. A reversible decision or "two-way door" allows changes without irreparable impacts.

:p Why is aiming for two-way doors important?
??x
Aiming for two-way doors is crucial because it ensures flexibility and reversibility, allowing the system to adapt quickly to changing requirements or technologies. This approach minimizes risk and simplifies maintenance by providing options to revert changes if needed.
x??

---

#### Cloud-Native Architecture Principles
Background context: The passage mentions five principles of cloud-native architecture from Tom Grey’s article on Google Cloud blog.

:p What does the principle "Prioritize Security" entail for data engineers?
??x
The principle emphasizes that every data engineer must take responsibility for securing the systems they build and maintain. This involves focusing on two main ideas: zero-trust security and the shared responsibility model, ensuring robust security practices are implemented.
x??

---

#### Zero-Trust Security Model
Background context: The text outlines the importance of zero-trust security, a model that assumes no entity should be automatically trusted based on network location alone.

:p Explain what zero-trust security means in practical terms?
??x
Zero-trust security requires verification and validation for every access request—regardless of whether it comes from inside or outside the perimeter. This approach ensures that all entities must provide credentials to prove their identity before they can gain access, thereby enhancing security by reducing unauthorized access risks.

In practice, this means implementing multi-factor authentication (MFA), strict access controls, and continuous monitoring for unusual activities.
x??

---

#### Shared Responsibility Model
Background context: The text discusses the shared responsibility model in cloud security, where both the service provider and the customer share responsibilities to maintain a secure environment.

:p How does the shared responsibility model work between AWS and its customers?
??x
In the shared responsibility model, AWS is responsible for securing the infrastructure (cloud platform) and the operations that run on it. Customers are responsible for securing their applications, data, and configurations within the cloud environment. This division ensures a balanced approach to security where both parties have specific roles in maintaining overall system security.

For example:
- **AWS:** Manages encryption keys, secure network boundaries, and physical hardware.
- **Customer:** Manages application code, user management, and access controls.
x??

---

#### Hardened-Perimeter vs Zero-Trust Security Models
Background context: The traditional hardened-perimeter security model has been widely used but is now considered vulnerable to both insider and external threats. This model assumes that everything inside the perimeter is trusted, while anything outside is untrusted. However, as security breaches through human targets have increased, this assumption becomes less reliable.
:p What are the limitations of the traditional hardened-perimeter security model?
??x
The limitations include:
- Vulnerability to insider attacks due to trusted insiders potentially compromising internal assets.
- Difficulty in protecting against external threats that can exploit human weaknesses within an otherwise secure network.

Code examples and explanations would not be applicable here as this is a theoretical concept. However, you could illustrate with a simple pseudocode example showing how a traditional security check might be bypassed through social engineering:
```pseudocode
if (userIsInTrustedNetwork) {
    grantAccess();
} else if (canBypassAuthentication(user)) { // Example condition for insider threat
    grantAccess(); // This should not happen in modern security practices
}
```
x??

---

#### Zero-Trust Security Model
Background context: The zero-trust security model is an alternative approach that does not assume anything inside or outside a network to be inherently trusted. It enforces strict authentication and authorization for all entities, regardless of their location.
:p What is the core principle of the zero-trust security model?
??x
The core principle of the zero-trust security model is that no user, device, application, or service should be automatically trusted just because it's inside a network perimeter. Access must be explicitly granted based on least privilege and continuous verification.

Code examples:
```java
public class ZeroTrustAuthenticator {
    public boolean authenticateUser(String username, String password) {
        // Check credentials against the database or identity provider
        if (isValidUser(username, password)) {
            return true;
        } else {
            logFailedAuthenticationAttempt(username);
            return false;
        }
    }

    private boolean isValidUser(String username, String password) {
        // Implementation to validate user credentials
        return users.contains(username) && passwords.get(username).equals(password);
    }

    private void logFailedAuthenticationAttempt(String username) {
        // Log the failed attempt for monitoring and security purposes
    }
}
```
x??

---

#### Shared Responsibility Model in Cloud Providers
Background context: The shared responsibility model divides security responsibilities between cloud providers and their customers. AWS, as an example, is responsible for securing its infrastructure and services, while users are responsible for securing their applications and data.
:p What does the shared responsibility model entail in a cloud environment?
??x
In a cloud environment, the shared responsibility model entails that:
- Cloud providers like AWS are responsible for securing the underlying infrastructure running their services (cloud).
- Users are responsible for securing their own applications, data, and how they use the services provided by the cloud provider.

Code examples illustrating this concept might include setting up security policies in an AWS environment. Here’s a simple example of configuring a VPC with no external connectivity:
```java
// Pseudocode to set up a VPC with restricted access
public class VpcSetup {
    public void configureVpc(String vpcId) {
        // Set up VPC rules to allow only necessary traffic and restrict everything else
        if (isInternalTraffic(vpcId)) {
            grantAccess();
        } else {
            denyAccess();
        }
    }

    private boolean isInternalTraffic(String vpcId) {
        // Logic to determine if the traffic is internal or external
        return true; // Simplified example
    }

    public void grantAccess() {
        // Grant necessary access
    }

    public void denyAccess() {
        // Deny all other access
    }
}
```
x??

---

#### Data Engineers as Security Engineers
Background context: With the shift to cloud-native architectures and the erosion of traditional perimeter security, data engineers are increasingly responsible for designing and implementing robust security measures. This includes understanding the shared responsibility model and taking an active role in securing applications and data.
:p Why should data engineers consider themselves security engineers?
??x
Data engineers should consider themselves security engineers because:
- They are often responsible for the security of the applications and data they develop, even if not explicitly in a security role.
- Traditional perimeter-based security is less effective due to increased internal threats from human targets and external threats that can exploit connected devices.

Code examples might involve implementing encryption or access controls within an application developed by a data engineer:
```java
public class DataSecurityManager {
    public void encryptData(String data) {
        // Implementation of data encryption logic
        String encryptedData = encrypt(data);
        storeEncryptedData(encryptedData); // Store the encrypted data securely
    }

    private String encrypt(String data) {
        // Logic to encrypt the data
        return "ENCRYPTED_" + data;
    }

    private void storeEncryptedData(String encryptedData) {
        // Securely storing the encrypted data, e.g., in a database with access controls
    }
}
```
x??

---

#### Failure to Assume Implicit Responsibilities Can Lead to Dire Consequences
Numerous data breaches have resulted from misconfigurations, such as Amazon S3 buckets being set with public access. This highlights the importance of recognizing that those handling data must take ultimate responsibility for securing it.

:p Why is it crucial for data handlers to assume they are responsible for securing the data?
??x
It is crucial because failing to do so can result in significant security breaches, leading to potential loss of sensitive information and financial or reputational damage. Handling data requires a proactive approach to ensure that all necessary precautions are taken to protect it.

```java
public class DataSecurity {
    public void configureS3Bucket(String bucketName, String acl) throws Exception {
        if (!acl.equals("private")) {
            throw new Exception("Ensure S3 buckets are configured with private access.");
        }
    }
}
```
x??

---

#### FinOps: Evolving Cloud Financial Management Discipline
FinOps is a discipline that integrates engineering, finance, technology, and business teams to make data-driven spending decisions in cloud environments. This helps organizations achieve maximum business value by optimizing the cost structure of their data systems.

:p What does FinOps primarily aim to achieve?
??x
FinOps aims to optimize the financial aspects of cloud usage by enabling collaboration between different departments (engineering, finance, technology, and business) to make informed spending decisions that balance costs with performance and efficiency.

```java
public class FinOpsBudget {
    public double calculateOptimalSpending(double estimatedCost, int resourceUtilization) {
        return Math.max(estimatedCost * 0.95, resourceUtilization * 1.2);
    }
}
```
x??

---

#### Performance Engineering vs. Cost Optimization in Cloud Environments
In traditional on-premises data systems, performance engineering focused on maximizing performance with fixed resources and future needs. In the cloud era, FinOps requires engineers to consider cost structures, such as choosing between AWS spot instances or reserved capacity based on cost-effectiveness.

:p How does FinOps change the approach from traditional performance engineering?
??x
FinOps changes the focus from simply optimizing performance to also considering cost efficiency. Engineers must now balance performance needs with financial constraints, making decisions like whether to use AWS spot instances for distributed clusters or reserve capacity for larger jobs based on cost and performance trade-offs.

```java
public class CostEfficiency {
    public String decideInstanceType(int jobSize) {
        if (jobSize < 1000) return "spot instance";
        else return "reserved capacity";
    }
}
```
x??

---

#### Managing Spending with FinOps in Cloud Environments
FinOps involves monitoring spending continuously and adjusting strategies like switching between pay-as-you-go models and reserved capacities. This requires setting hard limits to prevent excessive spending, which can be catastrophic for small companies.

:p What are some key practices of FinOps in managing cloud spend?
??x
Key practices include ongoing budget monitoring, using pay-per-query or other flexible billing models, implementing cost controls such as requester-pays policies for S3 buckets, and setting up alerts to detect and address excessive spending. These practices help ensure that costs remain under control while maintaining the necessary performance levels.

```java
public class SpendingMonitor {
    public void setupSpendingAlert(double threshold) {
        // Implement alert logic based on cost thresholds
    }
}
```
x??

---

#### Adopting Hard Limits for Spending with FinOps
In response to potential spending spikes, companies can set hard limits that trigger a "graceful failure" mode. This is similar to how systems handle excessive traffic but applied to financial constraints.

:p How does the concept of "graceful failure" apply in FinOps?
??x
The concept of "graceful failure" applies by setting spending limits and allowing for predefined actions when those limits are exceeded, such as reducing resource usage or pausing non-essential operations. This prevents catastrophic failures due to excessive costs.

```java
public class GracefulFailure {
    public void handleSpendingSpike(double currentCost) {
        if (currentCost > THRESHOLD) {
            System.out.println("Spending spike detected, initiating graceful failure.");
            // Implement logic for reducing resources or pausing operations
        }
    }
}
```
x??

---

#### Monitoring and Addressing Excessive Data Access Spending
For public data sharing, monitoring requester-pays policies can prevent unexpected high costs. This ensures that data teams can manage access and spending effectively to avoid financial risks.

:p What is a key practice for managing public data sharing in FinOps?
??x
A key practice is setting up requester-pays policies or closely monitoring data access spending. By doing so, data teams can ensure that excessive access does not lead to uncontrolled costs, which could threaten the budget of small companies.

```java
public class DataAccessMonitor {
    public void monitorS3BucketSpending(String bucketName) throws Exception {
        // Logic to check and alert for high S3 spending
    }
}
```
x??

---

#### FinOps Foundation and Partner Tiers
Background context: The FinOps Foundation was established in 2019 to formalize practices related to financial operations in cloud computing. It now has 300 members and introduces new partner tiers for Cloud Service Providers (CSPs) and vendors.
:p What is the purpose of the FinOps Foundation?
??x
The purpose of the FinOps Foundation is to promote best practices, share knowledge, and collaborate among professionals involved in financial operations related to cloud computing. It aims to help organizations optimize their cloud spend and improve financial transparency.
x??

---

#### Domain and Services in Architecture
Background context: In software architecture, domains represent real-world subject areas, while services are sets of functionalities designed to accomplish specific tasks within those domains.
:p What is the difference between a domain and a service?
??x
A domain represents the real-world subject area for which you're architecting (e.g., sales or accounting). A service is a set of functionality whose goal is to perform a particular task, such as processing orders or invoicing. For example, in a sales domain, there could be multiple services like orders, invoicing, and products.
x??

---

#### Scalability and Elasticity
Background context: These are critical characteristics for designing data systems that can handle increased demand without manual intervention from engineers. Scalability allows increasing the system's capacity to manage higher loads, while elasticity enables automatic scaling based on current workload.
:p What is the difference between scalability and elasticity?
??x
Scalability refers to the ability of a system to increase its capacity to handle more load by adding resources (e.g., more machines). Elasticity extends this concept by allowing systems to dynamically scale up or down automatically as needed. For example, in a cloud environment, scaling down can save costs when demand decreases.
x??

---

#### Availability and Reliability
Background context: These are measures of how well a system performs its intended function over time. High availability ensures the system is operable most of the time, while reliability indicates the probability that the system will meet performance standards during a specified interval.
:p What are availability and reliability in data systems?
??x
Availability refers to the percentage of time an IT service or component is operational. Reliability measures the likelihood that a system meets its performance requirements over a given period. For instance, if a data processing pipeline needs to run 24/7, high availability ensures it remains functional throughout the day.
x??

---

#### Distributed Systems
Background context: Distributed systems enable higher overall scalability and reliability by distributing tasks across multiple machines or nodes. This approach allows for better handling of increased loads without relying solely on single-machine solutions.
:p What is a distributed system?
??x
A distributed system consists of multiple autonomous computers that communicate over a network to achieve a common goal, such as processing data or executing tasks. Each node in the system can handle part of the workload, and they coordinate their actions to ensure overall performance and reliability.
x??

---

#### Horizontal Scaling
Background context: Horizontal scaling involves adding more machines to distribute the load and meet resource requirements, ensuring that the system can handle increased demand efficiently without manual intervention.
:p What is horizontal scaling?
??x
Horizontal scaling involves increasing the number of machines or nodes in a distributed system to manage higher loads. Each new node can take on part of the workload, allowing the entire system to scale outwards rather than upwards (vertical scaling). For example:
```java
public class LoadBalancer {
    List<Node> nodes = new ArrayList<>();
    
    public void addNode(Node newNode) {
        nodes.add(newNode);
    }
    
    public Node chooseNode() {
        // Logic to select a node for the next task
        return nodes.get(random.nextInt(nodes.size()));
    }
}
```
This example shows how a load balancer might distribute tasks among multiple nodes in a horizontally scaled system.
x??

---

#### Tight Versus Loose Coupling: Tiers, Monoliths, and Microservices
Background context explaining the concept of coupling in data architecture. This involves understanding how dependencies within domains and services can be either tightly or loosely coupled, which affects the reliability and flexibility of the system.

If applicable, add code examples with explanations.
:p What is a key difference between tight and loose coupling in the context of data architecture?
??x
In tight coupling, every part of a domain and service is vitally dependent upon every other domain and service. In contrast, loosely coupled systems have decentralized domains and services that do not strictly depend on each other. Loose coupling allows for greater flexibility and reliability since changes or failures in one area are less likely to impact others.

For example:
- Tight Coupling: A change in the business logic layer requires recompiling and redeploying the entire application.
- Loose Coupling: A change in a specific service only affects that service, without impacting other services.
??x
The answer with detailed explanations.
```java
// Example of tight coupling
public class App {
    private Database db;
    private Logic logic;

    public App() {
        this.db = new Database();
        this.logic = new Logic(db);
    }
}

// Example of loose coupling
public class ServiceA {
    private IDatabaseService dbService;

    public ServiceA(IDatabaseService dbService) {
        this.dbService = dbService;
    }

    // Methods using dbService without direct dependency on specific implementations
}
```
x??

---

#### Single-Tier Architecture
Background context explaining the concept of a single-tier architecture, where the database and application are tightly coupled on a single server. This setup is good for prototyping but not suitable for production environments due to its vulnerabilities.

:p What is an example scenario where a single-tier architecture would be used?
??x
A single-tier architecture is often used in development environments or during the early stages of application development when simplicity and ease of deployment are more critical than robustness. For instance, it might be employed on a developer’s local machine for testing purposes.

:p What are some limitations of using a single-tier architecture in production?
??x
Single-tier architectures present significant limitations in production due to their tightly coupled nature:

- Single points of failure: If the server fails, all services fail.
- Resource contention: The database and application compete for limited resources (disk, CPU, memory), which can lead to performance issues or even system downtime.
- Inability to run complex queries against the production database without risking system availability.

:p How does a single-tier architecture handle redundancy?
??x
Even when redundancies are implemented, such as failover replicas, single-tier architectures still face limitations. For example, running analytics queries on the main application database could overwhelm it and cause the entire system to become unavailable.

:x
The answer with detailed explanations.
```java
// Example of a simple single-tier setup
public class MyApp {
    private Database db;
    private ApplicationLogic logic;

    public MyApp() {
        this.db = new Database();
        this.logic = new ApplicationLogic(db);
    }
}
```
x??

---

#### Multitier Architecture
Background context explaining the concept of multitier architecture, where layers are separated to achieve maximum reliability and flexibility. A common example is a three-tier architecture.

:p What are the benefits of using a multitier architecture?
??x
The primary benefit of a multitier architecture is improved reliability and flexibility:

- Separation of concerns: Each layer can be developed and maintained independently.
- Scalability: Resources can be distributed across layers, allowing for better performance management.
- Fault isolation: If one service fails, it does not necessarily affect the entire system.

:p What are the different tiers typically found in a multitier architecture?
??x
A typical multitier architecture consists of:

1. Data Tier: Handles data storage and retrieval.
2. Application Logic Tier: Contains business rules and application logic.
3. Presentation Tier: Manages user interfaces and interactions.

:p How does resource contention play into the design considerations for a multitier architecture in distributed systems?
??x
In a distributed system, designers need to consider whether nodes should handle requests independently (shared-nothing architecture) or share resources such as memory, disk, and CPU. Decisions here can significantly impact performance and reliability.

:p How does resource sharing affect data access and performance in a multitier architecture?
??x
Resource sharing can improve performance by allowing multiple nodes to handle requests concurrently, but it also increases the risk of contention for shared resources (e.g., database connections). In a shared-nothing architecture, each node handles its own requests without relying on other nodes. However, this approach may limit scalability.

:x
The answer with detailed explanations.
```java
// Example of a simple multitier setup
public class DataTier {
    // Data handling logic
}

public class ApplicationLogicTier {
    private DataTier dataTier;

    public ApplicationLogicTier(DataTier dataTier) {
        this.dataTier = dataTier;
    }

    // Business logic and application processing
}

public class PresentationTier {
    private ApplicationLogicTier appLogicTier;

    public PresentationTier(ApplicationLogicTier appLogicTier) {
        this.appLogicTier = appLogicTier;
    }

    // UI logic and user interaction handling
}
```
x??

---

#### Monoliths
Background context explaining the concept of monolithic architectures, where all components are tightly coupled within a single codebase. Discusses technical and domain coupling.

:p What is an example scenario where a monolith would be used?
??x
Monoliths are often used in small-scale applications or during the early stages of development when simplicity and rapid deployment are more critical than scalability or maintainability. They can simplify initial development by reducing complexity, as all components are tightly coupled within one codebase.

:p What is technical coupling in a monolith?
??x
Technical coupling refers to how different architectural tiers (e.g., data, application logic) are intertwined within the single codebase of a monolith. This tight integration makes it difficult to isolate or replace specific components without affecting others.

:p How does domain coupling differ from technical coupling in a monolith?
??x
Domain coupling refers to how different domains (business units, services) interact and depend on each other within a monolith. While technical coupling is about the architectural tiers, domain coupling involves the logical integration of various business functions.

:p What are some challenges when refactoring a monolith into microservices?
??x
Refactoring a monolith into microservices can be complex due to the tight dependencies between components and services:

- High initial effort: Extracting and defining boundaries for each service requires significant rework.
- Risk of unintended consequences: Improving one component might negatively impact others, leading to "whack-a-mole" scenarios.

:p How does a monolith evolve into a big ball of mud?
??x
A big ball of mud occurs when the complexity of a monolithic architecture grows without proper refactoring or modularization. Over time, the codebase becomes harder to understand and maintain, with many dependencies that are difficult to untangle.

:x
The answer with detailed explanations.
```java
// Example of a complex monolith
public class Monolith {
    private Data data;
    private Logic logic;

    public Monolith() {
        this.data = new Data();
        this.logic = new Logic(data);
    }

    // Complex interdependencies between components
}
```
x??

---

#### Microservices Architecture
Background context explaining the concept of microservices, which are separate and loosely coupled services that provide specific functions. Discusses the differences from monoliths.

:p How does a microservices architecture differ from a monolithic architecture?
??x
Microservices architecture differs fundamentally from monoliths by introducing decentralization and loose coupling:

- **Decentralized Services:** Each service has its own database, making it independent.
- **Loose Coupling:** Services operate independently; failure in one service does not affect others.

:p What is a common approach to converting an existing monolith into microservices?
??x
Converting a monolith into microservices typically involves the following steps:

1. Identify bounded contexts: Determine distinct business domains that can become individual services.
2. Extract components: Gradually refactor and separate parts of the monolith into standalone services.
3. Introduce service boundaries: Clearly define APIs for communication between services.

:p What are the benefits of a microservices architecture?
??x
Benefits of microservices include:

- **Scalability:** Services can scale independently based on demand.
- **Resilience:** Failure in one service does not bring down others.
- **Maintainability:** Smaller, more focused services are easier to develop and maintain.

:p What challenges might arise when implementing a microservices architecture?
??x
Challenges include:

- Complexity: Managing multiple services can be complex due to distributed nature.
- Deployment complexity: Each service needs its own deployment strategy.
- Dependency management: Services depend on each other, requiring careful API design.

:x
The answer with detailed explanations.
```java
// Example of a microservices-based architecture
public interface IDatabaseService {
    // Service-specific database operations
}

public class ServiceA implements IDatabaseService {
    private Database db;

    public ServiceA(Database db) {
        this.db = db;
    }

    @Override
    public void performAction() {
        // Specific logic for ServiceA
    }
}
```
x??

---

