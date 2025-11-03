# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 49)

**Starting Chapter:** Load Balancing

---

---
#### Scaling for Performance
In microservice architectures, scaling is primarily done to handle more load and reduce latency. Vertical scaling involves getting a bigger box with faster CPU and better I/O capabilities, which can improve processing speed and throughput but may be expensive and not always resource-efficient.

:p What does vertical scaling refer to in the context of scaling microservices?
??x
Vertical scaling refers to increasing the performance of a single server by upgrading its hardware resources such as CPU, memory, or storage. This approach aims to handle more load and reduce latency through enhanced computational power.
```java
// Example of applying vertical scaling using cloud services API
public class CloudInstanceResize {
    public void resizeInstance(int newCores, int newMemoryGB) throws Exception {
        // Code to resize the instance in a cloud environment with updated resources
        // Assume this is part of a larger orchestration framework
        // Example pseudo-code:
        // cloudAPI.resize(instanceId, newCores, newMemoryGB);
    }
}
```
x??
---

#### Splitting Workloads
To improve throughput and scaling, microservices can be split across multiple hosts. Initially, it might be tempting to run multiple services on a single host for cost efficiency or simplicity of management. However, as the application grows, this approach may lead to resource bottlenecks.

:p How does splitting workloads help in microservice architectures?
??x
Splitting workloads helps by distributing the load among multiple hosts, thereby improving throughput and scaling. By separating critical functionalities from non-critical ones, we can optimize each service for its specific tasks. For example, if an accounts service handles both customer financial record management and query generation, splitting these into two separate services can reduce the load on the core functionality while ensuring that reporting queries don't impact critical operations.
```java
// Example of splitting a microservice based on workload
public class AccountService {
    public void manageAccounts() {
        // Code for managing customer financial records
    }

    public void generateReports() {
        // Code for generating reports, potentially using more efficient query methods
    }
}

// Suggested refactoring
public class AccountsService {
    public void manageCustomerRecords() {
        // Refactored code focusing on critical operations
    }
}

public class ReportingService {
    public void generateFinancialReports() {
        // Specialized code for generating reports, possibly using distributed computing or caching
    }
}
```
x??
---

#### Spreading Your Risk
To ensure resilience in the system, it's important to avoid concentrating all services on a single host. This reduces the impact of any single point failure.

:p How can spreading risk improve the resilience of microservices?
??x
Spreading your risk means distributing the services across multiple hosts to minimize the impact of any single point failure. If one service fails, others remain operational, ensuring that the overall system remains functional. For example, if you have a financial accounts service and an independent reporting service, each can fail independently without causing significant disruptions.

```java
// Example of hosting microservices on different instances for resilience
public class ServiceDeployment {
    public void deployServices() {
        // Code to deploy accounts service on one instance
        deploy("AccountsService", "instance1");

        // Code to deploy reporting service on a separate instance
        deploy("ReportingService", "instance2");
    }

    private void deploy(String serviceName, String instanceId) {
        // Pseudo-code for deploying services
        System.out.println("Deploying " + serviceName + " on " + instanceId);
    }
}
```
x??
---

#### Virtualization and Host Distribution
Background context explaining how virtual hosts can impact service availability. Discuss the importance of distributing services across different physical boxes to reduce downtime risks.

:p How can virtualization impact the availability of services?
??x
Virtualization platforms can present a single point of failure if all services are running on virtual hosts hosted on the same physical box. If that physical box fails, multiple services could go down simultaneously. To mitigate this risk, it is crucial to distribute the virtual hosts across different physical boxes or data centers.

```java
// Example of distributing VMs across physical hosts
public class HostDistributor {
    public void distributeVMs(List<VirtualMachine> vms) {
        List<PhysicalHost> physicalHosts = getPhysicalHosts();
        for (VirtualMachine vm : vms) {
            int randomHostIndex = getRandomInt(physicalHosts.size());
            PhysicalHost host = physicalHosts.get(randomHostIndex);
            host.addVM(vm);
        }
    }
}
```
x??

---

#### SAN Failure Impact
Background context on the risks associated with Storage Area Networks (SANs). Explain how a failure of a large, expensive SAN can affect multiple virtual machines connected to it.

:p How does a SAN failure impact VMs?
??x
A SAN failure can be catastrophic as it may take down all connected VMs since they are typically mapped to a single SAN. This risk is significant because SANs are critical infrastructure designed not to fail but do occasionally go down, causing severe outages and data loss.

```java
// Example of handling SAN failures in a distributed system
public class SanFailHandler {
    public void handleSanFailure(List<VirtualMachine> vms) {
        // Notify all VMs connected to the failed SAN about their failure state.
        for (VirtualMachine vm : vms) {
            if (vm.isAttachedToFailedSan()) {
                vm.shutdownGracefully();
                System.out.println("Notifying " + vm.getName() + " of San Failure");
            }
        }
    }
}
```
x??

---

#### Data Center and Rack Placement
Background context on the importance of placing services in different racks or across multiple data centers to avoid a single point of failure.

:p Why is it important to distribute services across multiple racks?
??x
Distributing services across multiple racks within a data center can prevent a localized issue, such as a network outage or physical damage, from impacting all services. Similarly, spreading services across different data centers provides redundancy and protects against large-scale disasters like natural calamities.

```java
// Example of distributing services across racks
public class RackDistributor {
    public void distributeServices(List<Service> services) {
        List<Rack> racks = getRacks();
        for (Service service : services) {
            int randomRackIndex = getRandomInt(racks.size());
            Rack rack = racks.get(randomRackIndex);
            rack.addService(service);
        }
    }
}
```
x??

---

#### Service Level Agreements (SLAs)
Background context on the significance of SLAs in ensuring uptime and recovery time objectives for services. Explain how to plan accordingly if the service provider does not meet your required SLA.

:p What is an SLA, and why is it important?
??x
An SLA is a formal agreement that defines the level of service expected by a customer from a service provider. It typically includes uptime guarantees, recovery time objectives (RTOs), and other performance metrics. Understanding the SLA terms is crucial to ensure your services meet the required availability. If the SLA does not match your needs, you might need alternative solutions or plan for redundancy.

```java
// Example of checking against an SLA
public class SlADownTimeChecker {
    public boolean checkSLADowntime(double actualDowntime) {
        double agreedUptime = 99.95; // Example agreed uptime in percentage
        double maxAllowedDowntime = (100 - agreedUptime) / 100;
        return actualDowntime <= maxAllowedDowntime;
    }
}
```
x??

---

#### AWS Regions and Availability Zones
Background context on how AWS is structured with regions and availability zones to provide high availability. Explain the importance of distributing workloads across multiple availability zones.

:p How do AWS regions and availability zones contribute to service availability?
??x
AWS is divided into regions, each containing one or more availability zones (AZs). AZs are designed to be isolated from failures in other AZs within the same region, ensuring higher availability. Distributing services across multiple AZs minimizes the risk of a single point of failure.

```java
// Example of distributing workloads across AWS availability zones
public class AwsWorkloadDistributor {
    public void distributeWorkloads(List<Service> services) {
        List<AvailabilityZone> azs = getAzs();
        for (Service service : services) {
            int randomAzIndex = getRandomInt(azs.size());
            AvailabilityZone az = azs.get(randomAzIndex);
            az.addService(service);
        }
    }
}
```
x??

---

#### Disaster Recovery Planning
Background context on the importance of having a backup hosting platform to ensure resilience. Explain how multiple suppliers can provide additional layers of protection.

:p Why is disaster recovery planning important?
??x
Disaster recovery planning is crucial for minimizing downtime and ensuring service availability in case your primary provider fails to meet its SLA guarantees. Having an alternative hosting platform with a different supplier provides redundancy and helps mitigate the risk of catastrophic failures.

```java
// Example of setting up a disaster recovery plan
public class DisasterRecoveryPlanner {
    public void setupDisasterRecovery(List<Service> services, Supplier backupSupplier) {
        for (Service service : services) {
            BackupService backup = backupSupplier.getBackupFor(service);
            service.setBackup(backup);
        }
    }
}
```
x??

---

#### Load Balancers and Scaling

Background context explaining the concept. Load balancers are essential for distributing network traffic to multiple instances of a service, ensuring that no single instance is overwhelmed. They also manage the health check of these instances and can add or remove them as needed.

:p What is a load balancer?
??x
A load balancer distributes incoming network requests across multiple backend servers based on predefined algorithms. It ensures that no single server bears too much traffic, thus improving overall system performance and reliability.
x??

---
#### SSL Termination

Background context explaining the concept. SSL termination involves decrypting HTTPS traffic at the load balancer before it reaches the target instances. Historically, this was done to offload encryption/decryption tasks from the backend servers. However, modern systems often use this feature to simplify host configuration.

:p What is SSL termination?
??x
SSL termination means that a load balancer decrypts incoming HTTPS traffic and forwards it as HTTP to the backend server. This reduces the overhead on backend instances but can introduce security risks if not properly managed.
x??

---
#### VLAN for Improved Security

Background context explaining the concept. A Virtual Local Area Network (VLAN) is used to isolate network segments, ensuring that requests only enter specific parts of a network via routers. In this case, it helps in securing microservices by limiting external access.

:p How can VLANs improve security?
??x
Using VLANs improves security by isolating the internal communication of microservices from external access. All traffic between instances within the same VLAN is handled over HTTP, while only HTTPS is used for external requests. This reduces the risk of man-in-the-middle attacks.
x??

---
#### AWS Elastic Load Balancers (ELBs)

Background context explaining the concept. AWS provides a variety of load balancers, including Elastic Load Balancers (ELBs), which can handle SSL termination and be configured with security groups or VPCs to manage network traffic.

:p What are AWS ELBs?
??x
AWS Elastic Load Balancers (ELBs) distribute incoming traffic across multiple targets in an Auto Scaling group. They support features like SSL termination, where HTTPS traffic is decrypted at the load balancer before being forwarded as HTTP to the backend instances.
x??

---
#### Software vs. Hardware Load Balancers

Background context explaining the concept. While hardware load balancers offer robust performance and reliability, they can be difficult to automate. In contrast, software load balancers like mod_proxy provide more flexibility in configuration.

:p What is a key difference between software and hardware load balancers?
??x
The primary difference lies in automation and flexibility. Hardware load balancers are designed for high performance and reliability but may lack the ease of reconfiguration and automation compared to software load balancers, which can be easier to modify as needed.
x??

---
#### Single Points of Failure

Background context explaining the concept. A single point of failure occurs when a critical component in a system can bring down the entire system if it fails. In distributed systems, ensuring that no single component is a bottleneck or critical path is essential.

:p What is a single point of failure?
??x
A single point of failure refers to any element within a system whose failure would cause the entire system to fail. In microservices architectures, this could be a database running on a single host. Ensuring there are no such points is crucial for maintaining system resilience.
x??

---
#### Managing Load Balancer Configuration

Background context explaining the concept. Proper management of load balancer configuration ensures that changes can be automated and stored in version control, reducing the risk of human error.

:p Why should load balancer configurations be managed like service configurations?
??x
Load balancer configurations should be treated similarly to other service configurations—stored in version control systems and able to be automatically applied. This helps maintain consistency across environments and reduces manual configuration errors.
x??

---

#### Worker-Based Systems Load Balancing

Worker-based systems are an alternative to traditional load balancing mechanisms where a pool of worker instances process tasks from a shared backlog. These workers can be employed for batch work, asynchronous jobs, and handling peak loads through dynamic scaling.

This model is particularly useful for tasks such as image thumbnail processing, sending emails, or generating reports. The key benefit is improved resiliency since failing workers do not result in lost data; only delays in task completion are experienced.

If the work queue itself is resilient (e.g., using a message broker like Kafka), worker-based systems can scale both for increased throughput and better fault tolerance.
:p How does a worker-based system handle tasks compared to traditional load balancing?
??x
A worker-based system processes tasks from a shared backlog where multiple instances of workers collaborate. Unlike traditional load balancers that route requests to available servers, workers pull tasks from a queue and process them independently.

This approach is especially useful for batch work and asynchronous jobs, ensuring that the impact of a worker failing is minimal since tasks can be retried later.
x??

---
#### Scaling Architectures

As systems grow, initial architectures may become inadequate. According to Jeff Dean's principle, "design for ~10× growth, but plan to rewrite before ~100×," scaling requires rethinking and rewriting parts of the architecture.

This often involves refactoring monolithic applications into microservices, selecting more scalable data stores, or adopting new technologies like event-driven architectures. The key is preparing for potential radical changes as the system evolves.
:p Why might a simple monolithic application need to be redesigned when handling increased load?
??x
A simple monolithic application may struggle with increased load due to its single-threaded nature and shared resources, leading to bottlenecks under higher concurrency.

To handle more substantial growth, you might split the monolith into microservices, each responsible for a specific task. This modular approach allows better scaling and resilience compared to a single large application.
x??

---
#### Gilt Case Study

Gilt started with a simple Rails application that managed well for two years but faced significant load as its business grew rapidly.

At a certain point, the existing architecture became insufficient, necessitating a redesign. The company eventually split their monolithic application into microservices to better manage different aspects of their e-commerce system.
:p What happened at Gilt when they reached a specific scaling threshold?
??x
Gilt's initial Rails application performed well for two years but struggled with increased load as the business expanded.

To handle this, they had to redesign the application by breaking it into microservices. This allowed them to manage different parts of their e-commerce system more efficiently and scale each component independently.
x??

---

#### Rapid Experimentation and Scaling Considerations
Background context: When starting a new project, it is crucial to experiment rapidly without over-preparing for scale. Eric Ries' story illustrates that focusing on understanding demand can save time and resources compared to building for scale upfront.

:p What does the text suggest about how projects should be initiated in terms of scaling?
??x
The text suggests initiating projects by rapidly experimenting rather than preparing for massive scale at the outset. This approach allows teams to understand if there is any demand before investing heavily in scaling infrastructure. For instance, Eric Ries could have tested market interest by creating a dummy link and spent six months on a beach instead of building a product that no one used.

```java
public class RapidExperimentation {
    public static void main(String[] args) {
        // Simulate a quick test to check if there is any demand
        boolean hasDemand = true;  // Hypothetical check for market interest
        if (hasDemand) {
            System.out.println("Proceed with development.");
        } else {
            System.out.println("Explore other ideas or pivot.");
        }
    }
}
```
x??

---

#### Scaling Stateless Microservices
Background context: Stateless microservices are easier to scale compared to those that store data. The focus is on handling increased read and write operations without maintaining state across requests.

:p How does scaling stateless microservices differ from stateful services?
??x
Scaling stateless microservices involves managing the number of instances or replicas based on load, whereas stateful services require additional considerations for state management and data consistency. Stateless services can be more straightforward to scale horizontally by adding more instances, while stateful services might need specialized technologies like distributed databases.

```java
public class StatelessMicroserviceScaling {
    public static void main(String[] args) {
        // Example of scaling stateless microservices
        int currentLoad = 50;  // Current number of requests per second
        if (currentLoad > 100) {  // Threshold for scaling
            System.out.println("Spinning up additional instances.");
        } else {
            System.out.println("Current load is manageable with existing resources.");
        }
    }
}
```
x??

---

#### Availability vs. Durability of Data
Background context: It’s crucial to distinguish between the availability of a service and the durability of data stored within it. These are two independent aspects that require different strategies.

:p What does the text highlight about the separation of service availability and data durability?
??x
The text emphasizes that service availability refers to whether users can access the application, while data durability concerns the persistence of user data. For example, storing data in a resilient filesystem ensures data safety but may not guarantee service availability if the database fails. A standby replica ensures data safety but requires mechanisms to bring it back online.

```java
public class AvailabilityVersusDurability {
    public static void main(String[] args) {
        boolean isDatabaseDown = true;  // Hypothetical scenario
        String dataSafetyStrategy = "Resilient filesystem";  // Ensures data isn't lost
        if (isDatabaseDown) {
            System.out.println("Data safety strategy: " + dataSafetyStrategy);
            System.out.println("Service availability may be compromised.");
        }
    }
}
```
x??

---

#### Scaling for Reads in Databases
Background context: Many services are read-heavy, making efficient read scaling essential. Caching and read replicas are common strategies to manage high read loads.

:p How does the text suggest scaling reads in databases?
??x
The text suggests using caching and read replicas as effective strategies for managing high read loads. Caching stores frequently accessed data in memory to reduce database load, while read replicas distribute read operations across multiple copies of the database.

```java
public class ReadReplicaExample {
    public static void main(String[] args) {
        // Simulate reading from a primary and replica databases
        String primaryRead = "SELECT * FROM products";  // Query for read data
        String replicaRead = "SELECT * FROM products";  // Same query to another replica

        System.out.println("Primary database: " + primaryRead);
        System.out.println("Replica database: " + replicaRead);
    }
}
```
x??

---

#### Eventually Consistent Systems
Background context explaining the concept. Eventually consistent systems allow temporary inconsistencies to achieve scalability, but require handling these inconsistencies appropriately.
:p What is an eventually consistent system?
??x
An eventually consistent system allows for temporary data inconsistency across nodes or replicas to provide high availability and scalability. Developers must handle these inconsistencies when reading data from the system.
x??

---

#### Using Read Replicas for Scaling Reads
Background context explaining how read replicas can be used to scale reads, providing a mechanism for improving performance without significant changes.
:p What is a read replica in database scaling?
??x
A read replica is an additional database node that mirrors the primary database. It helps distribute read operations and thus improves read scalability without significantly impacting write operations.
x??

---

#### Sharding for Writes: Overview
Background context explaining how sharding can be used to scale writes by distributing data across multiple nodes, but with challenges in handling queries.
:p What is sharding?
??x
Sharding involves splitting the dataset into smaller parts and storing each part on a separate database node. This allows horizontal scaling of write operations by distributing the load across multiple nodes.
x??

---

#### Handling Queries Across Shards
Background context explaining the challenges of querying data across shards, including possible solutions like map/reduce jobs or caching.
:p How can you handle queries that span multiple shards?
??x
Queries spanning multiple shards can be handled using asynchronous mechanisms like map/reduce jobs in databases such as MongoDB. These jobs execute complex queries on each shard and then combine the results to provide a unified view.
x??

---

#### Adding Shards to an Existing Cluster
Background context explaining how adding more database nodes (shards) affects data distribution, with potential downtime issues addressed by some systems.
:p How can you add shards to an existing cluster?
??x
Adding shards to an existing cluster involves rebalancing the data across new and existing nodes. Some databases handle this asynchronously in the background, such as Cassandra, which can add shards without significant downtime.
x??

---

#### Resiliency and Data Replication with Sharding
Background context explaining how sharding can affect resiliency and the importance of replicating data to multiple nodes for reliability.
:p What is the impact of sharding on resiliency?
??x
Sharding can reduce resiliency if not managed properly. For example, if customer records A–M are always written to a single instance, that instance being unavailable would mean losing access to those records. Cassandra addresses this by replicating data across multiple nodes in a ring.
x??

---

#### Evaluating Database Technologies for Scaling Writes
Background context explaining the challenges of scaling write volume and the considerations when choosing database technology.
:p When might you need to change your database technology?
??x
You might need to change your database technology when you hit limits on scaling write volume. Buying a bigger box is often quick, but evaluating systems like Cassandra, MongoDB, or Riak can offer better long-term solutions for scalable writes.
x??

---

---
#### Shared Database Infrastructure
Background context explaining how traditional RDBMS separate databases and schemas, allowing multiple microservices to run on a single database. This setup reduces the number of machines needed but introduces significant risks due to potential single points of failure.

:p What is shared database infrastructure used for?
??x
Shared database infrastructure separates the concept of the database itself from the schema, enabling one running database to host multiple independent schemas, each serving a different microservice. This can reduce the need for multiple databases and servers but also increases the risk of a single point of failure if the database goes down.

x??
---

---
#### Command-Query Responsibility Segregation (CQRS)
Background context on CQRS, which separates commands for modifying state from queries used to retrieve data. It allows different models to handle scaling independently, potentially offering more flexibility in terms of how and where data is stored and accessed.

:p What does the CQRS pattern separate?
??x
The CQRS pattern separates commands (for modifying state) and queries (for retrieving data). These are processed by separate systems, allowing for different scaling strategies and storage models. For instance, commands can be event-sourced, while queries can use projections or a different type of store.

x??
---

---
#### Event Sourcing
Background context on how CQRS can implement event sourcing, where commands are stored as events, which can then be used to reconstruct the state of the domain objects. This approach allows for more flexible and scalable query models.

:p How does event sourcing work in CQRS?
??x
In CQRS with event sourcing, commands are recorded as events in a data store. These events can later be processed to update the model or to generate projections that represent the current state of the domain objects. This process allows for maintaining a history of changes and reconstructing the current state.

Example code:
```java
public class EventSourcingService {
    private final List<Event> eventList;

    public void applyCommand(Command command) {
        // Apply the command by creating an event
        Event event = new Event(command.getDetails());
        eventList.add(event);
        
        // Optionally, update state or projections based on the event
    }
}
```

x??
---

---
#### Caching
Background context explaining how caching can be used to store previous results of operations and serve them faster in subsequent requests. It is commonly used to reduce database round-trips and improve performance.

:p What is caching used for?
??x
Caching stores the result of expensive or frequent operations, so that future calls with the same parameters can retrieve the result from cache instead of recalculating it. This reduces the load on underlying systems like databases and improves response times.

Example code:
```java
public class CacheManager {
    private Map<String, String> cache = new HashMap<>();

    public String getCachedValue(String key) {
        return cache.getOrDefault(key, fetchFromDatabase(key));
    }

    private String fetchFromDatabase(String key) {
        // Simulate fetching from a database or other service
        return "value";
    }
}
```

x??
---

#### Client-Side Caching
Client-side caching involves storing cached results directly on the client's device, allowing it to decide when and if to fetch a fresh copy. This can significantly reduce network calls, leading to performance benefits. However, managing stale data invalidation and rolling out changes can be challenging.
:p What is client-side caching?
??x
Client-side caching refers to storing cached results directly on the client's device. The client determines whether to retrieve a fresh copy, which can drastically reduce network calls and alleviate load on downstream services. However, invalidating stale data and making changes to caching behavior across multiple consumers might be difficult.
x??

---

#### Proxy Caching
Proxy caching involves placing a proxy between the client and server. This setup can cache traffic from various services, providing a transparent way to add caching without modifying clients or servers directly. Reverse proxies like Squid or Varnish are common examples.
:p What is proxy caching?
??x
Proxy caching places a proxy between the client and server, handling caching for multiple services in an opaque manner. This can be simpler than changing existing systems, as it caches generic traffic that passes through it. However, adding additional network hops should be considered.
x??

---

#### Server-Side Caching
Server-side caching involves storing cached data on servers using tools like Redis or Memcache. It offloads the client from managing cache invalidation and reduces load on downstream services by keeping frequently accessed data closer to the source.
:p What is server-side caching?
??x
Server-side caching stores cached results on servers, often utilizing systems like Redis or Memcache. This approach keeps frequently accessed data close to the service, reducing load on downstream services and simplifying invalidation of stale data within a single service boundary.
x??

---

#### Mixed Caching Approaches
For public-facing websites, combining client-side, proxy caching, and server-side caching is common. Each method addresses different needs, such as reducing network calls, adding transparency through proxies, or handling cache invalidation locally.
:p How do you typically approach caching in a public-facing website?
??x
Typically, public-facing websites use a combination of client-side, proxy, and server-side caching. Client-side caching reduces network load, while proxy caching adds transparency by caching generic traffic. Server-side caching handles local invalidation and keeps frequently accessed data close to the service.
x??

---

#### No Caching Strategy
In some distributed systems, no caching strategy might be sufficient if the system can handle the current load without performance bottlenecks. The decision on whether or not to cache depends on factors such as required data freshness and current system capabilities.
:p In what scenarios might you avoid using a caching strategy?
??x
Avoiding a caching strategy in distributed systems is appropriate when the system can handle its current load without performance issues. This might be the case if the application doesn't experience high traffic or strict requirements for data freshness, making caching unnecessary and potentially complicating the architecture.
x??

---

#### Key Considerations for Caching
The choice of caching strategy depends on factors like expected load, data freshness needs, and current system capabilities. Each approach (client-side, proxy, server-side) has its pros and cons in terms of performance, management complexity, and invalidation mechanisms.
:p What key factors should you consider when choosing a caching strategy?
??x
Key considerations for choosing a caching strategy include expected load, required data freshness, and current system capabilities. Each approach—client-side, proxy, server-side—has distinct advantages (performance benefits, ease of implementation) and challenges (complexity in invalidation, additional network hops). Understanding these factors helps in selecting the most appropriate caching mechanism.
x??

---

---
#### Cache-Control Directives
Background context explaining how `cache-control` directives work. These headers are used by servers to instruct clients on whether and for how long a resource should be cached.

:p What are `cache-control` directives, and what do they control?
??x
`cache-control` directives allow server-side configuration of caching behavior. They inform the client about whether to cache a resource and for how long. This can significantly reduce the number of requests made to the server by serving stale content from the cache.

Example usage in an HTTP response:
```http
Cache-Control: max-age=3600, public
```
This tells clients that they should cache this resource for 1 hour (`max-age=3600`) and allows them to share (publish) the cached copy with other users.

x??
---

---
#### Expires Header
Background context explaining the `Expires` header. Unlike `cache-control`, which specifies a time-to-live, the `Expires` header sets an absolute date and time after which a resource should be considered stale.

:p How does the `Expires` header differ from `Cache-Control` directives in managing cache freshness?
??x
The `Expires` header provides a specific timestamp when a cached resource becomes outdated. Unlike `cache-control`, it gives clients a fixed date and time to fetch the latest version of the content.

Example usage:
```http
Expires: Thu, 01 Dec 2022 16:00:00 GMT
```
This sets an absolute expiration date for the resource. After this timestamp, clients should revalidate or fetch a fresh copy from the server.

x??
---

---
#### ETags and Conditional GETs
Background context explaining Entity Tags (ETags) and how they are used in conditional requests. ETags allow servers to identify unique versions of resources, enabling efficient updates without changing the URI.

:p What is an ETag, and when would you use it?
??x
An ETag is a unique identifier for a version of a resource. It helps determine if the cached content has been updated since the last request. Using ETags in conditional GET requests (via `If-None-Match` header) ensures that clients only receive new or updated resources.

Example usage:
```http
GET /customer-record HTTP/1.1
Host: example.com
If-None-Match: o5t6fkd2sa
```
This request checks if the ETag of the customer record matches `o5t6fkd2sa`. If it does, a `304 Not Modified` is returned; otherwise, a new resource and updated ETag are sent.

x??
---

#### Caching for Writes
Caching can be beneficial for writes, particularly through a technique called write-behind caching. This approach involves writing data to a local cache and flushing it to a downstream source later. It's useful when dealing with bursts of writes or situations where the same data might be written multiple times.
Write-behind caches can also buffer and batch writes, providing performance optimizations. If the downstream service is unavailable, buffered writes can be queued up and sent through when the service becomes available again.

:p What are the benefits of using write-behind caching?
??x
Using write-behind caching offers several benefits:
1. **Performance Optimization**: By buffering writes in a local cache before sending them to the downstream source, you reduce the load on the network and can batch multiple writes.
2. **Resilience Against Downtime**: If the downstream service is unavailable, previously buffered writes can still be sent through when it becomes available again.
3. **Handling Write Bursts**: It helps manage sudden spikes in write traffic by smoothing out these bursts.

This approach ensures that data updates are not lost even if the downstream service is temporarily down.

x??

---

#### Caching for Resilience
Client-side caching can enhance system resilience, especially when dealing with potential failures. When a downstream service is unavailable, cached but potentially stale data can be used to serve requests.
Additionally, techniques like periodic crawling of live sites and generating static versions can ensure that even if the main site goes down, a version of the website remains available.

:p How does client-side caching improve system resilience?
??x
Client-side caching improves system resilience by providing fallback options when the downstream service is unavailable:
1. **Stale Data Serves**: When the cache contains stale but cached data, it can be served to clients, ensuring that no requests are left unanswered.
2. **Static Site Generation**: Techniques such as periodic crawling of a live site and generating static versions can ensure that even if the main system fails, a version of the site remains available.

This approach ensures high availability with potentially stale but usable data, improving overall system reliability.

x??

---

#### Hiding the Origin
In scenarios where a cache failure leads to a large number of requests hitting the origin, it's important to protect the origin from overwhelming traffic. One effective strategy is to have the origin itself populate the cache asynchronously when needed.
This way, if there's a cache miss, the origin can be alerted and start repopulating the cache in the background.

:p How does hiding the origin behind caching help?
??x
Hiding the origin behind caching helps by protecting it from overwhelming traffic during cache failures:
1. **Asynchronous Repopulation**: When a cache miss occurs, the origin is notified to populate the cache asynchronously.
2. **Background Processing**: This allows for background processing of cache requests, preventing a sudden surge in load on the origin.

By ensuring that the origin only handles necessary requests and deferring others to background processes, the system can remain stable and responsive.

x??

---

#### Hiding the Origin from the Client and Populating Cache Asynchronously
Background context explaining how hiding the origin from the client can be a strategy to ensure system resilience. By failing requests fast, avoiding resource usage or latency increases, and ensuring cache failures do not cascade downstream, it gives the system a chance to recover.

:p How does hiding the origin from the client contribute to system resilience?
??x
Hiding the origin from the client helps in maintaining system availability by quickly failing requests that encounter issues. This prevents the failure from consuming resources or increasing latency, which could otherwise propagate and affect other parts of the system. If a cache fails, the request is served directly from the source, ensuring that downstream components are not burdened with failed cache requests.
x??

---

#### Failing Requests Fast
Background context on how failing requests quickly can help prevent resource usage and reduce latency, thereby maintaining overall system performance even when parts of the system fail.

:p Why should failures be handled by immediately failing requests?
??x
Failing requests fast helps in not consuming unnecessary resources or increasing latency. By ensuring that a failed request does not wait for additional processing, the system can quickly recover and handle subsequent requests more efficiently. This approach minimizes the impact of transient errors and ensures that other parts of the system remain responsive.
x??

---

#### Caching Considerations
Background context on the importance of keeping caching simple to avoid data staleness issues in complex architectures like microservices.

:p Why is it important to keep caching simple?
??x
Keeping caching simple reduces the complexity and potential for errors, making it easier to manage and maintain. In a microservice architecture where multiple services are involved, having too many caches can make it difficult to track and ensure data freshness. A simpler cache design allows for better control over how and when data is cached, reducing the risk of serving stale or incorrect data.
x??

---

#### Cache Poisoning
Background context on the risks associated with cache poisoning, particularly in scenarios where multiple layers of caching exist between a user and the source of fresh data.

:p What is cache poisoning, and why is it problematic?
??x
Cache poisoning occurs when cached data is altered to serve stale or incorrect data. This can be particularly problematic in architectures with multiple caches, such as CDNs, ISPs, and user browsers, where controlling these layers can be challenging. If a cache is not properly invalidated, stale data can persist indefinitely, leading to issues for users.
x??

---

#### Example of Cache Poisoning Scenario
Background context on an actual scenario where cache poisoning led to serving stale data due to misconfigured HTTP headers.

:p Describe the scenario where cache poisoning caused serving stale data?
??x
In a project using a strangler application to intercept calls to multiple legacy systems, a small subset of pages started serving stale data indefinitely. This happened because a bug introduced in the application logic resulted in not setting proper HTTP cache headers on some responses. The downstream application had an `Expires: Never` header that was not overridden by our application's headers. As a result, these pages remained cached indefinitely in Squid and user browsers until manually cleared or replaced.
x??

---

#### Clearing Cache to Resolve Cache Poisoning
Background context on the steps taken to resolve cache poisoning issues, including code fixes and manual cache clearing.

:p How was the cache poisoning issue resolved?
??x
To resolve the cache poisoning issue, the following steps were taken:
1. A fix was applied to the application's cache header insertion code.
2. A release was pushed to deploy the fix.
3. The relevant region of Squid cache was manually cleared to remove old data.

These actions helped mitigate the impact but did not fully resolve the problem since pages with `Expires: Never` headers were still cached in user browsers. The final solution involved changing the URLs of affected pages so that they would be refetched by users.
x??

---

#### Caching and Its Complexities
Caching is a powerful mechanism that can significantly improve performance by reducing access times to data. However, it requires careful management of the entire data path from source to destination to avoid potential issues such as stale or inconsistent data.

:p What are the key considerations for managing caching in distributed systems?
??x
The key considerations include ensuring data consistency across cache and database layers, handling cache invalidation policies effectively, and understanding how changes propagate through the system. You need to design your caching strategy based on the specific application requirements and data access patterns.
```java
public class CacheManager {
    public void invalidateCache(String key) {
        // Code to invalidate cache entry for a given key
    }
}
```
x??

---

#### Autoscaling Fundamentals
Autoscaling allows you to automatically adjust the number of active instances based on demand. This is particularly useful in managing cost and ensuring optimal performance.

:p What are some scenarios where autoscaling can be effectively used?
??x
Autoscaling can be effective for managing load variations, such as daily or seasonal trends, and responding quickly to sudden spikes in traffic. For example, scaling up instances during peak hours and downgrading them when demand decreases helps in maintaining optimal performance while controlling costs.
```java
public class Autoscaler {
    public void scaleUp(int desiredInstances) {
        // Code to launch additional instances
    }

    public void scaleDown() {
        // Code to terminate excess instances
    }
}
```
x??

---

#### Reactive vs. Predictive Scaling
Reactive scaling involves responding to changes in load or instance failures, while predictive scaling anticipates future needs based on historical data.

:p What are the advantages and disadvantages of reactive and predictive scaling?
??x
Reactive scaling allows for immediate response to unexpected events but can lead to overprovisioning during periods of low demand. Predictive scaling, using historical data, can prevent under-provisioning and reduce costs by maintaining optimal resource levels. However, it requires accurate forecasting and robust testing.
```java
public class LoadPredictor {
    public void predictLoad() {
        // Code to analyze historical load patterns
    }
}
```
x??

---

#### Autoscaling for Failures
Using autoscaling to handle instance failures ensures that the system remains available by automatically replacing failed instances.

:p How does AWS manage instance failures through autoscaling?
??x
AWS allows you to set rules such as "at least 5 instances in this group," ensuring that if one instance fails, a new one is automatically launched. This helps maintain system availability and reduces downtime due to individual instance failures.
```java
public class AwsAutoScaler {
    public void configureMinInstances(int minInstances) {
        // Code to set minimum instance requirement
    }
}
```
x??

---

#### Cost Management with Autoscaling
Autoscaling can help in cost management by allowing you to pay only for the computing resources used. However, careful observation and data analysis are essential.

:p What is a critical step when implementing autoscaling to ensure cost efficiency?
??x
A critical step is setting up comprehensive load tests to validate scaling rules. Without proper testing, it's difficult to predict how well your system will handle varying loads, leading to potential overprovisioning or underprovisioning.
```java
public class LoadTester {
    public void testScalability(int[] expectedLoads) {
        // Code to simulate different load scenarios and test scaling behavior
    }
}
```
x??

---

