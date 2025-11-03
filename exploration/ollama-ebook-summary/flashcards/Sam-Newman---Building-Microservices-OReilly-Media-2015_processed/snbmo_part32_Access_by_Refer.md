# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 32)

**Starting Chapter:** Access by Reference

---

#### Services as State Machines
Background context explaining that services are modeled around bounded contexts, where each service owns its logic and controls lifecycle events. This approach avoids anemic services and maintains cohesion by centralizing decision-making within the service itself.

:p What is a microservice's role in managing lifecycle events?
??x
A microservice should control all lifecycle events related to its domain concepts. For instance, when dealing with a customer, the customer service decides which changes are allowed based on its internal logic and rules.
x??

---
#### Reactive Extensions (Rx)
Background context explaining that reactive extensions allow for more flexible handling of asynchronous operations by observing outcomes instead of blocking calls. This is particularly useful in distributed systems where concurrent calls need to be managed efficiently.

:p How does Rx help manage multiple service calls in a microservice architecture?
??x
Reactive extensions enable you to handle multiple service calls without blocking, making it easier to manage concurrent requests and compose their results. For example, you can use `Observable` from RxJava to observe the response of a downstream service call and react accordingly.
x??

---
#### DRY Principle in Microservices
Background context explaining the Don't Repeat Yourself (DRY) principle and how it applies to microservice architectures by emphasizing avoiding redundant code that duplicates system behavior.

:p How does the DRY principle apply differently in a microservice world compared to traditional monolithic applications?
??x
In a microservice world, DRY means avoiding duplication of behavior and knowledge across services. This is important because each service should be self-contained and responsible for its domain logic. Redundant code can lead to inconsistencies and increased maintenance complexity.
x??

---

#### DRY Principle and Microservices
Background context explaining the importance of DRY (Don't Repeat Yourself) in software development. In microservice architectures, while applying DRY can lead to reusable code, it also poses risks such as introducing coupling between services. Code duplication is easier to manage compared to the complexity introduced by shared libraries.
:p What are the main concerns when applying DRY in a microservice architecture?
??x
Applying DRY in a microservice architecture can introduce unnecessary coupling if changes propagate across multiple services, especially when common code leaks outside service boundaries. This can lead to maintenance issues and potential bugs due to misaligned updates between services.
```java
// Example of a shared domain object
public class SharedDomainObject {
    private String id;
    private String name;

    public void updateName(String newName) {
        // Update logic here
    }
}
```
x??

---

#### Coupling in Microservices via Common Code
Context around the dangers of using common code across microservices. This can lead to tight coupling, especially if changes in shared libraries require updates in multiple services and message queues.
:p What are some potential issues with shared domain objects or common libraries in a microservice architecture?
??x
Shared domain objects or common libraries can cause tight coupling between services, as changes in one service necessitate updates across many others. This can lead to maintenance overhead, increased risk of bugs due to synchronization issues, and the need for draining message queues.
```java
// Example of shared domain object causing coupling issues
public class SharedDomainObject {
    private String id;
    private String name;

    public void updateName(String newName) {
        // Update logic here
    }
}
```
x??

---

#### Tailored Service Templates and Code Duplication
Explanation on the benefits and limitations of using a tailored service template. While it can help in avoiding code duplication, it can also limit technology choices if strictly enforced.
:p How does RealEstate.com.au ensure they avoid coupling when creating new services?
??x
RealEstate.com.au uses a tailored service template to bootstrap new service creation but avoids sharing this code across different services by copying the template for each new service. This approach helps in maintaining loose coupling and reducing the risk of breaking changes propagating through multiple services.
```java
// Example of using a tailored service template
public class ServiceTemplate {
    // Common configurations and setup logic here
}
```
x??

---

#### Client Libraries and API Design
Discussion on the importance of client libraries for ease of use but also the risks of tightly coupling server and client logic. AWS' approach with SDKs is highlighted as a good model.
:p Why can creating client libraries pose challenges in microservice architectures?
??x
Creating client libraries can introduce tight coupling between the server API and the client code, potentially leading to changes in one needing updates in the other. This can limit technology choices and make it harder to implement fixes or use alternative technologies.
```java
// Example of a client library with tightly coupled logic
public class ServiceClient {
    public void doSomething() {
        // Server-specific logic here
    }
}
```
x??

---

#### Amazon Web Services Model for Client Libraries
Explanation on AWS' approach where SDKs provide abstractions over underlying APIs, written by different teams to avoid coupling.
:p What is the key benefit of using community or third-party SDKs in microservices?
??x
Using community or third-party SDKs can help decouple client code from server logic, allowing for greater flexibility in technology choices and reducing the risk of tightly coupled updates. This approach leverages abstractions provided by trusted sources to ensure loose coupling.
```java
// Example of using an AWS SDK
public class AwsServiceClient {
    public void doSomething() {
        // Use AWS SDK methods here
    }
}
```
x??

#### Client Library Approach and Its Implications
Background context: The client library approach is often discussed as a way to manage service interactions, particularly within large-scale systems like Netflix. This method ensures reliability and scalability by handling aspects such as service discovery, failure modes, logging, etc. However, it also comes with the risk of increased coupling between client and server.
:p How does the client library approach benefit and potentially harm system design?
??x
The benefits include centralized management of common issues like service discovery and failure handling, which can significantly improve reliability and scalability. However, over time, this can lead to tight coupling between clients and servers, making it harder to make changes independently.

```java
// Example of a simplified client library method for service discovery
public class ClientLibrary {
    public String discoverService(String serviceName) {
        // Logic to discover the service endpoint
        return "http://service-endpoint";
    }
}
```
x??

---

#### Service Discovery and Failure Handling
Background context: When using microservices, services need to be able to find each other at runtime. Netflix's client libraries manage this dynamically through mechanisms like DNS or configuration management systems.
:p What is service discovery in the context of microservices?
??x
Service discovery refers to the process by which a microservice identifies and locates another microservice during runtime. This is crucial for dynamic, distributed systems where services can fail or scale up/down independently.

```java
// Pseudocode example of service discovery using DNS
public class ServiceDiscovery {
    public String getEndpoint(String serviceName) {
        // Query DNS for the endpoint of the service
        return "http://service-endpoint-123";
    }
}
```
x??

---

#### Client/Server Coupling and Upgrades
Background context: Ensuring that client upgrades can happen independently of server updates is critical. Netflix emphasizes this by allowing clients to decide when they upgrade, maintaining system reliability even if one component fails or changes.
:p How does Netflix manage the coupling between client and server?
??x
Netflix manages coupling through its client libraries, which handle service discovery, failure modes, logging, etc., allowing clients to independently upgrade without disrupting the entire system. This is achieved by separating concerns: handling underlying transport protocols from business logic.

```java
// Pseudocode example of client managing upgrades
public class ClientUpgradeManager {
    public void checkForUpdate() {
        // Logic to check for and apply updates
        if (shouldUpdate()) {
            updateClientLibrary();
        }
    }

    private boolean shouldUpdate() {
        // Check conditions to decide whether an upgrade is necessary
        return true;
    }
}
```
x??

---

#### Passing Domain Entity References
Background context: In a microservice architecture, it's important to ensure that the lifecycle of domain entities is managed within their respective services. This means when you request data, it should be from the source service where the entity's state is considered authoritative.
:p How do we handle references to domain entities in a microservice architecture?
??x
When retrieving a domain entity like `Customer`, treat the service hosting that entity as the single source of truth. Pass around a reference to the original resource instead of a cached memory of its previous state, ensuring you can always fetch the latest data.

```java
// Example method for fetching customer details from the Customer Service
public class CustomerServiceClient {
    public Customer getCustomer(String customerId) {
        // Fetch the current state of the customer from the service
        return client.request("GET", "/customer/" + customerId);
    }
}
```
x??

---

#### Avoiding Data Duplication and Maintaining Efficiency
Background context: In microservices, frequently accessing data can lead to inefficiencies. Balancing between having a cached memory of an entity and ensuring you fetch the latest state from the source service is crucial for maintaining system performance.
:p How do we balance between using a cached memory and fetching the latest state?
??x
You should use a cached memory of an entity but also maintain a reference to its original resource. This way, if the data needs to be refreshed or updated, you can fetch the new state from the source service.

```java
// Example method for handling customer references
public class CustomerHandler {
    private final String customerId;
    private final Customer cachedCustomer;

    public CustomerHandler(String customerId) {
        this.customerId = customerId;
        this.cachedCustomer = getCustomerFromCache();
    }

    public void updateCustomer() {
        // Check if the cached customer is stale and fetch from the service
        if (isStale(cachedCustomer)) {
            cachedCustomer = getCustomerFromService();
        }
    }

    private Customer getCustomerFromCache() {
        // Return a cached memory of the customer
        return new Customer("John Doe", "john.doe@example.com");
    }

    private boolean isStale(Customer cachedCustomer) {
        // Logic to determine if the cache needs to be refreshed
        return true;
    }

    private Customer getCustomerFromService() {
        // Fetch the latest state from the service
        return customerServiceClient.getCustomer(customerId);
    }
}
```
x??

#### Asynchronous Request Handling vs. Resource References
Background context explaining how asynchronous request handling and resource references can be used to manage data retrieval and processing efficiently. When an order is shipped, sending a request with customer details can work, but using URIs for Customer and Order resources allows the email service to fetch the latest information when needed.
:p How does using URIs for Customer and Order resources benefit asynchronous request handling?
??x
Using URIs for Customer and Order resources benefits asynchronous request handling by allowing the email service to look up the latest state of these entities when it is time to send the email. This approach ensures that the most current data is used, reducing the risk of sending outdated information.
```java
// Example of using URI in a request
public void sendShipmentNotification(String customerUri) {
    // The email service fetches the customer details from the URI
    Customer customer = customerService.getCustomer(customerUri);
    String emailBody = generateEmailBody(customer.getName(), customer.getOrderDetails());
    sendEmail(emailBody);
}
```
x??

---

#### Event-Based Collaboration and Resource State Management
Background context explaining event-based collaboration, where a "this happened" message is sent without the need to include all details. The importance of knowing what specifically happened when an event occurs is highlighted.
:p Why is it important to know "what happened" in event-based collaborations?
??x
Knowing "what happened" in event-based collaborations is crucial because it provides context and details about the state or change that occurred, which can be essential for making informed decisions. For instance, if a customer's resource changes, knowing what the customer looked like at the time of the event helps in understanding the specific circumstances leading to the change.
```java
// Example of an event handler receiving a change in a Customer resource
public void handleCustomerChangeEvent(CustomerChangeEvent event) {
    // Retrieve the state of the Customer at the time of the event
    Customer previousState = customerService.getCustomer(event.getCustomerId(), event.getTimeStamp());
    processCustomerChange(previousState, event);
}
```
x??

---

#### Load Management and Caching Strategies
Background context explaining how accessing resources by reference can lead to increased load on the resource service. The use of caching strategies can mitigate this issue, but it requires careful management of data freshness.
:p How can caching help in managing load when using references for resource access?
??x
Caching can help manage load when using references for resource access by storing frequently accessed data temporarily. By setting appropriate cache controls and invalidating cached data as needed, the system can reduce the number of direct requests to the resource service, thereby减轻负载。例如，HTTP 提供了多种缓存控制选项，如 `Cache-Control` 和 `ETag`，可以用来管理数据的时效性。
```java
// Example of using cache controls in HTTP headers
public void sendShipmentNotification(String customerUri) {
    String url = "http://example.com/api/customers/" + customerUri;
    // Using Cache-Control header to manage caching
    HttpHeaders headers = new HttpHeaders();
    headers.setCacheControl("max-age=3600");
    
    ResponseEntity<Customer> response = restTemplate.exchange(
        url, HttpMethod.GET, new HttpEntity<>(headers), Customer.class);
    
    if (response.getStatusCode() == HttpStatus.NOT_MODIFIED) {
        // Handle cached data
    } else {
        // Process the fetched customer details
        Customer customer = response.getBody();
        String emailBody = generateEmailBody(customer.getName(), customer.getOrderDetails());
        sendEmail(emailBody);
    }
}
```
x??

---

#### Data Freshness and Request Efficiency
Background context explaining the importance of knowing data freshness when passing around information in requests. Overloading requests with too much data can increase coupling, but not including enough might lead to sending outdated information.
:p How does ensuring data freshness impact request efficiency?
??x
Ensuring data freshness impacts request efficiency by reducing the amount of redundant data passed in each request and minimizing the risk of using outdated information. By providing clear indicators of when a resource was in a given state, services can make more informed decisions about whether to fetch new data or use cached information.
```java
// Example of including freshness information with a request
public void sendShipmentNotification(String customerUri) {
    String url = "http://example.com/api/customers/" + customerUri;
    // Adding ETag header for freshness check
    HttpHeaders headers = new HttpHeaders();
    headers.set("If-None-Match", "etag-value");
    
    ResponseEntity<Customer> response = restTemplate.exchange(
        url, HttpMethod.GET, new HttpEntity<>(headers), Customer.class);
    
    if (response.getStatusCode() == HttpStatus.NOT_MODIFIED) {
        // Handle cached data
    } else {
        // Process the fetched customer details
        Customer customer = response.getBody();
        String emailBody = generateEmailBody(customer.getName(), customer.getOrderDetails());
        sendEmail(emailBody);
    }
}
```
x??
---

---
#### Defer Breaking Changes for as Long as Possible
Background context: The objective is to reduce the impact of making breaking changes by deferring them. This can be achieved through careful choice of integration technologies and encouraging good behavior in clients to avoid tight coupling.

:p How does picking REST over database integration help in avoiding breaking changes?
??x
Picking REST helps because it reduces the likelihood of changes in internal implementation details affecting the service interface. For example, if you're using a database for integration, a change in schema might require immediate updates in all dependent services due to direct access to database fields.

For contrast:
```java
// Example of tight coupling with database (not recommended)
public class OrderService {
    private final CustomerRepository customerRepo;

    public void sendOrderShippedEmail(int customerId) {
        var customer = customerRepo.findById(customerId).orElseThrow();
        // Direct access and manipulation
        String firstName = customer.getFirstName();
        String lastName = customer.getLastName();
        
        // Email logic here...
    }
}
```

In REST, you can use a more flexible approach:
```java
// Example of using a more flexible approach with REST
public class OrderService {
    private final HttpClient httpClient;
    
    public void sendOrderShippedEmail(int customerId) {
        var customerResponse = httpClient.sendRequestToCustomerService(customerId);
        
        // Use XML parsing or JSON to extract necessary fields
        String firstName = customerResponse.getFirstname();
        String lastName = customerResponse.getLastname();
        
        // Email logic here...
    }
}
```
x??

---
#### Tolerant Reader Pattern
Background context: A tolerant reader is a pattern where the service implementation can evolve without breaking clients by ignoring changes that are not relevant to the client. This allows for more flexible and future-proof services.

:p How does the tolerant reader pattern help in avoiding breaking changes?
??x
The tolerant reader pattern helps because it allows the service to change its internal structure or fields without affecting the external interface, provided that irrelevant details can be ignored by clients.

For example:
```xml
<!-- Original response -->
<customer>
    <firstname> Sam </firstname>
    <lastname> Newman </lastname>
    <email> sam@magpiebrain.com </email>
    <telephoneNumber> 555-1234-5678 </telephoneNumber>
</customer>

<!-- Restructured response after removing telephoneNumber -->
<customer>
    <naming>
        <firstname> Sam </firstname>
        <lastname> Newman </lastname>
        <nickname> Magpiebrain </nickname>
        <fullname> Sam "Magpiebrain" Newman </fullname>
    </naming>
    <email> sam@magpiebrain.com </email>
</customer>

// Implementation in Java
public String getEmailFromCustomerResponse(String customerId) {
    // Example using XPath to extract email without needing to know the inner structure
    String xml = getCustomerXml(customerId);
    String email = XPathExpression.evaluate("//email", xml, null);
    return email;
}
```
x??

---
#### Postel’s Law (Robustness Principle)
Background context: Postel's Law states that "Be conservative in what you do, be liberal in what you accept from others." This principle encourages systems to be forgiving of errors and flexible with input, which is useful in scenarios where the service might change or unexpected variations occur.

:p How does Postel’s Law apply to handling changes in microservices?
??x
Postel's Law applies by encouraging services to be tolerant of unexpected or changed inputs while maintaining strictness about their own outputs. This means that a service should validate and handle inputs gracefully, allowing it to evolve without breaking other services that depend on it.

For example:
```java
public class EmailService {
    public void sendOrderShippedEmail(String customerId) {
        // Robust parsing of customer response
        String xml = getCustomerDetails(customerId);
        
        try {
            // Extract necessary fields using a tolerant approach
            String firstName = XPathExpression.evaluate("//firstname", xml, null);
            String lastName = XPathExpression.evaluate("//lastname", xml, null);
            
            // Proceed with email logic...
        } catch (XPathExpressionException e) {
            log.error("Failed to parse customer details.", e);
            throw new RuntimeException("Customer data missing or malformed.");
        }
    }
}
```
x??

---

#### Catch Breaking Changes Early
Background context: The importance of identifying breaking changes early is crucial to maintain service compatibility and user satisfaction. Consumer-driven contracts and running tests with supported client libraries are techniques to detect such issues before deployment.

:p What technique can help identify breaking changes early?
??x
Using consumer-driven contracts or running tests using each supported library against the latest service can help spot these problems early.
x??

---

#### Use Semantic Versioning
Background context: Semantic versioning is a specification that uses MAJOR.MINOR.PATCH to indicate the type of changes in software releases. This allows clients to predict whether their application will work with new versions of a service.

:p How does semantic versioning help in managing service compatibility?
??x
Semantic versioning helps by clearly defining what changes are backward incompatible, compatible, or bug fixes through MAJOR.MINOR.PATCH increments. Clients can easily determine if they need to update their code based on the version number.
x??

---

#### Coexist Different Endpoints
Background context: Coexisting different versions of endpoints allows for gradual migration without forcing all consumers to upgrade simultaneously. This approach is useful when breaking changes are necessary.

:p How can coexistence of different endpoint versions be managed?
??x
By deploying a new service that supports both the old and new interfaces, allowing time for consumers to migrate gradually. Once all consumers have migrated, the old endpoint can be removed.
x??

---

#### Expand and Contract Pattern
Background context: The expand and contract pattern is used when coexisting multiple versions of an API. It involves initially supporting both the old and new versions, then eventually retiring the old version.

:p How does the expand and contract pattern work in practice?
??x
Initially, support both old (e.g., V1) and new (e.g., V2 or V3) interfaces. Gradually phase out the old interface as consumers start using the new one until it can be completely removed.
x??

---

#### URI Versioning vs Request Header Versioning
Background context: Different methods for routing requests to different API versions include using version numbers in URIs and headers. Both approaches have their pros and cons.

:p How can versioning be implemented via URI?
??x
Using version numbers in the URI, such as `/v1/customer/` or `/v2/customer/`, makes it clear which version is being requested but may allow clients to hardcode templates.
x??

---

#### Protocol Buffer Versioning
Background context: In RPC systems, different versions of methods can be managed by placing them in different namespaces. This approach becomes complex when dealing with the same types across versions.

:p How can versioning be handled using protocol buffers?
??x
Using namespace prefixes like `v1.createCustomer` and `v2.createCustomer` to differentiate between method versions. However, this complexity increases when managing different types sent over the network.
x??

---

