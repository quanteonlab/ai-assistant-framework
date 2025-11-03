# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 29)

**Starting Chapter:** Turtles All the Way Down

---

#### Bounded Contexts and Microservices Alignment
Background context explaining that bounded contexts are crucial for creating well-defined, self-contained microservices. Properly aligning microservices with these bounded contexts ensures loose coupling and high cohesion within the system.

:p What is a bounded context, and why is it important in microservices architecture?
??x
A bounded context is a region of the domain where all like-minded business capabilities live together, providing clear boundaries for the scope of responsibilities. It helps in avoiding tight coupling by clearly defining what models should be shared and not shared between services.

This concept ensures that each service focuses on specific functionalities, making it easier to manage and scale individual components independently. Bounded contexts are essential because they allow teams to work on different parts of a system without interfering with others.
x??

---
#### Module and Service Boundaries
Background context explaining how modules can be used within a process boundary to keep related code together while reducing coupling between services.

:p How do you start organizing your new codebase into modules and services?
??x
Start by identifying the bounded contexts in your domain. These contexts should be modeled as modules, where shared models are exposed and hidden models remain private. This organization helps in creating a structure that aligns well with microservices.

When starting out, it's advisable to keep the system more monolithic; this allows you to get familiar with the domain before deciding on service boundaries. Overly aggressive decomposition can lead to high costs due to changing requirements or misaligned services.
x??

---
#### Microservices and Bounded Contexts
Background context explaining that microservices should cleanly align with bounded contexts, ensuring loose coupling and strong cohesion.

:p Why should microservices be aligned with bounded contexts?
??x
Microservices should be aligned with bounded contexts because it ensures that each service focuses on a specific set of functionalities. This alignment helps in achieving high cohesion within the services and loose coupling between them, making the system more maintainable and scalable.

Aligning microservices with bounded contexts also simplifies testing and deployment processes since each context can operate independently.
x??

---
#### Premature Decomposition
Background context explaining how ThoughtWorks experienced challenges when decomposing a system into microservices too quickly. It highlights that premature decomposition can lead to high costs due to changing requirements or misaligned services.

:p What is the risk of prematurely splitting a system into microservices?
??x
Prematurely decomposing a system into microservices can be costly, especially if you are new to the domain. When initial boundaries are not well thought out, they may need frequent changes leading to high costs and complex maintenance. It's often better to have an existing monolithic codebase that can be gradually refactored into microservices once the domain is more familiar.
x??

---
#### Business Capabilities
Background context explaining the focus should be on business capabilities rather than shared data when identifying bounded contexts.

:p What should you consider first when defining a bounded context?
??x
When defining a bounded context, start by considering what business capabilities that context provides to other parts of the domain. For example, the warehouse may provide the capability to get a current stock list or handle order fulfillment. This approach ensures that services are designed around their primary functions rather than shared data.

Focusing on business capabilities helps in creating more meaningful and cohesive microservices.
x??

---
#### Nested Bounded Contexts
Background context explaining that bounded contexts can contain further nested contexts, which may be better represented as a single service or split into separate services based on organizational structure.

:p How do you handle nested bounded contexts when designing microservices?
??x
Nested bounded contexts can exist within larger, coarser-grained contexts. It's important to first consider the larger context and then subdivide along these nested contexts for potential benefits. Whether to model the higher-level bounded context as a top-level service or split it into separate services depends on the organizational structure.

For example, you could decompose the warehouse into order fulfillment, inventory management, or goods receiving capabilities. If different teams manage these functionalities, they should be modeled as separate microservices. Otherwise, the nested approach might make sense for simplicity and easier testing.
x??

---
#### Turtles All the Way Down
Background context explaining that bounded contexts can have sub-contexts which may remain hidden to collaborating services.

:p How do you design microservices when there are nested bounded contexts?
??x
When designing microservices with nested bounded contexts, start by considering the larger coarser-grained contexts. You can either model these as top-level boundaries or split them into separate services based on organizational structure and testing needs. For example, a warehouse context might have sub-contexts like inventory management, order fulfillment, and goods receiving.

These nested contexts can be hidden from collaborating microservices, making the overall system appear to operate using business capabilities without revealing the underlying service architecture.
x??

---
#### Testing and Isolation
Background context explaining how testing can benefit from understanding the boundaries between services, especially when dealing with complex architectures like microservices.

:p How does understanding service boundaries help in testing?
??x
Understanding service boundaries helps in designing effective tests. You might decide to have end-to-end tests where all services within a bounded context are launched together, but for other collaborators, you can stub out these internal services. This approach provides a unit of isolation when considering larger-scoped tests.

For example, if testing a service that consumes the warehouse, you don't need to stub each service inside the warehouse context; instead, you might just stub the more coarse-grained API.
x??

---

#### Changes to System Functionality
Background context: The changes made to a system are often driven by business needs and involve modifying the functionality or capabilities exposed to customers. When the system is decomposed into bounded contexts, these changes can be isolated more effectively within one microservice boundary.

:p What does changing system functionality typically involve?
??x
Changing system functionality involves altering the behavior of the software in a way that better aligns with business requirements. This might include adding new features, modifying existing ones, or removing outdated functionalities to enhance user experience and meet evolving business goals.
x??

---

#### Bounded Contexts and Microservices
Background context: Bounded contexts represent different parts of an organization's domain and are used in Domain-Driven Design (DDD) to ensure that terms have a clear meaning within specific areas. When microservices align with bounded contexts, it helps maintain consistency in business terminology across the system.

:p How do bounded contexts affect communication between microservices?
??x
Bounded contexts ensure that each service uses consistent and well-defined terms and concepts from their respective domain areas. This alignment is crucial for clear and effective communication between microservices because it reduces ambiguity and ensures that services understand and interpret requests correctly based on shared business language.

For example, if a bounded context in the "Order Management" area defines a "Product", another service within the same context should use this definition without confusion.
x??

---

#### Technical Boundary Example
Background context: The example provided discusses issues arising from incorrectly modeling services. Initially, a monolithic system was split into two parts based on geographical and organizational lines rather than domain boundaries.

:p What were the main issues with the service decomposition in this case?
??x
The main issues included:
1. Frequent changes required in both front-end and back-end services.
2. Excessive use of RPC-style method calls, leading to brittleness.
3. Chatty interfaces resulting in performance issues necessitating elaborate batching mechanisms.

This decomposition did not align with the business domain but instead followed a technical seam (splitting repository layers).
x??

---

#### Onion Architecture
Background context: The term "onion architecture" refers to systems with many layers, which can make maintenance and changes difficult. The given example describes such an architecture that had issues due to its complexity.

:p What is the issue described as "onion architecture" in this text?
??x
The "onion architecture" described here refers to a system with multiple layers (like an onion), making it challenging to navigate and modify. In the case provided, the service was split into two parts, each with its own interfaces and logic, leading to high complexity and difficulty in managing changes.

Code example:
```java
public class OrderService {
    private final ProductRepository productRepo;
    
    public OrderService(ProductRepository productRepo) {
        this.productRepo = productRepo;
    }
    
    public void placeOrder(Order order) {
        Product product = productRepo.findById(order.getProductId());
        // ... more complex logic ...
    }
}
```

Explanation: This example shows a simple service that depends on another repository, which can add layers of complexity in larger systems.
x??

---

#### Vertical vs. Horizontal Service Boundaries
Background context: The text contrasts vertical decomposition (aligned with the business domain) against horizontal decomposition (splitting along technical seams).

:p What is the difference between vertical and horizontal service boundaries?
??x
Vertical service boundaries align services with specific parts of the business domain, ensuring that each microservice encapsulates a cohesive set of functionalities. Horizontal boundaries split services based on operational concerns like data storage or geographical distribution.

For example:
- Vertical: An "Order Management" service handles all aspects related to orders.
- Horizontal: A database service might be horizontally scaled across multiple machines but still falls under the same domain (e.g., order management).

Horizontal boundaries can be useful for performance optimizations, while vertical boundaries support clearer business logic and easier maintenance.
x??

---

---
#### Bounded Contexts and Microservices
Bounded contexts are a fundamental concept in Domain-Driven Design (DDD) that help identify domains within a problem space. They allow for clear boundaries between different parts of a system, ensuring loose coupling and high cohesion among microservices.

Background context:
In DDD, the term "bounded context" refers to the area where a specific domain model is valid and applicable. Each bounded context has its own vocabulary, rules, and constraints that must be respected by all parties within it.

:p What are bounded contexts in Domain-Driven Design?
??x
Bounded contexts are areas of focus within a larger problem space where a particular domain model is valid and follows specific rules and constraints. They help in identifying independent domains with distinct boundaries, ensuring microservices maintain their autonomy.
x??

---
#### Microservice Subdivision
The chapter hints at the potential need to further subdivide microservices into smaller components.

Background context:
Even after defining bounded contexts, there might be complex interactions within those contexts that require even finer-grained decomposition. This can help in reducing complexity and improving manageability of individual services.

:p Why is it important to consider subdividing microservices?
??x
Subdividing microservices can help address complex internal logic by breaking down the system into more manageable parts, enhancing modularity, and making it easier to scale and maintain.
x??

---
#### MusicCorp Example Domain
MusicCorp serves as an example domain throughout the book.

Background context:
Using a specific domain like MusicCorp helps illustrate various concepts related to microservices and DDD. It provides a concrete context for discussing principles such as bounded contexts and service boundaries.

:p What is the role of MusicCorp in the book?
??x
MusicCorp acts as an example domain used throughout the book to demonstrate key concepts related to microservices, bounded contexts, and Domain-Driven Design.
x??

---
#### Interface Technology for Microservices
Choosing the right technology for integrating microservices is crucial.

Background context:
There are several technologies available for microservice integration, such as SOAP, XML-RPC, REST, Protocol Buffers, etc. Each has its strengths and weaknesses, but some considerations include avoiding breaking changes and ensuring API agnosticism to accommodate future changes in technology stacks.

:p What factors should be considered when choosing an interface technology for microservices?
??x
When selecting an interface technology for microservices, consider factors such as minimizing breaking changes, ensuring backward compatibility, and keeping APIs technology-agnostic to support potential shifts in the tech stack.
x??

---
#### Avoiding Breaking Changes
Handling breaking changes is important to maintain service autonomy.

Background context:
Even small changes can impact consumers of a microservice if not managed correctly. Techniques like semantic versioning can help manage such changes without causing disruptions.

:p How can we handle breaking changes when making updates to a microservice?
??x
To handle breaking changes, use techniques like semantic versioning and ensure that any new features or changes are backward-compatible unless absolutely necessary. This helps maintain the autonomy of services and their consumers.
x??

---
#### Technology-Agnostic APIs
Ensuring API agnosticism is crucial for future-proofing.

Background context:
As technology evolves rapidly, APIs should be designed in a way that minimizes dependencies on specific technologies or frameworks. This allows services to adapt more easily when adopting new tools or changing stacks.

:p Why is it important to keep APIs technology-agnostic?
??x
Keeping APIs technology-agnostic is essential for future-proofing systems and allowing them to adapt to changes in the tech stack without significant disruptions. It promotes flexibility and maintainability.
x??

---

---
#### Microservices and API Technology-Agnosticism
In this context, microservices are favored for their flexibility. One of the core principles is to ensure that communication between services (microservices) uses APIs that do not dictate specific technology stacks, thus maintaining technological freedom.
:p Why is it important to keep APIs used in microservices technology-agnostic?
??x
It's essential because using a technology-specific API could limit our choice of technologies for implementing microservices. This means we should avoid integration methods or tools that force us into a particular tech stack. For instance, choosing an HTTP-based API is more flexible than one tied to a specific framework or library.
```java
// Example of a simple REST API method in Java
public class CustomerService {
    @GetMapping("/customers/{id}")
    public ResponseEntity<Customer> getCustomer(@PathVariable Long id) {
        // Logic to fetch customer from database
        return new ResponseEntity<>(customer, HttpStatus.OK);
    }
}
```
x??
---

#### Simplifying Service Consumption for Consumers
The objective here is to make services easy to use by consumers. This involves considering the cost of using a service and providing tools that can ease adoption without tying consumers too closely to our internal implementation.
:p How can we ensure that our microservices are simple for their consumers to use?
??x
We should aim to give consumers full freedom in their technology choices while making it easy for them. One approach is to provide client libraries, but this must be balanced against increased coupling between the service and its consumers. We want to avoid exposing internal implementation details that could cause disruptions if changes are made.
```java
// Example of a simple client library method in Java
public class CustomerClient {
    public void registerCustomer(Customer customer) {
        // Logic to send request to CustomerService
    }
}
```
x??
---

#### Hiding Internal Implementation Details
To prevent consumers from being bound to our internal implementation, we should avoid exposing details that could cause issues if changes are made. This helps reduce coupling and the cost of change.
:p Why is it important to hide internal implementation details in microservices?
??x
Hiding internal implementation details ensures that when we need to make changes inside a microservice, those changes do not impact consumers of the service. By doing so, we reduce technical debt within the service and maintain flexibility in our development processes.
```java
// Example of abstracting database access in Java
public class CustomerRepository {
    public void save(Customer customer) {
        // Abstracted logic to save customer without revealing internal details
    }
}
```
x??
---

#### Common Integration Options: The Shared Database Approach
The shared database is a common integration approach where services read and write data directly from the same database. While simple, it can lead to tight coupling between services.
:p How does the shared database method work in microservices architecture?
??x
In the shared database method, multiple services interact with the database to retrieve or modify data. This approach is straightforward but can result in high coupling because changes made to the schema or data access patterns by one service may affect others.
```java
// Example of a simple SQL operation in Java using JPA
public class CustomerService {
    @PersistenceContext
    private EntityManager entityManager;

    public void createCustomer(Customer customer) {
        entityManager.persist(customer);
    }
}
```
x??
---

#### Exposing Internal Implementation Details Through DB Integration
Background context explaining the concept. When services use direct database integration to access and modify data, they are essentially allowing external parties to view and bind to internal implementation details. This can lead to breaking changes if the schema or logic is altered. The data stored in the database becomes a large shared API that is brittle and prone to failures during updates.

If a change is required in how helpdesk manages customers, it must be carefully managed to avoid breaking other services tied to the same database structure.
:p What are the risks of allowing external parties to view and bind to internal implementation details through DB integration?
??x
The main risks include:
1. Breaking changes: Any schema modifications or logic updates can disrupt consumers that rely on the current structure.
2. Brittle API: The database becomes a large, shared API that is sensitive to any changes.
3. Regression testing: Significant effort is required to ensure no part of the schema breaks during updates.

For example, if you decide to add a new field or change data types in a customer table used by multiple services:
```java
// Old schema
public class Customer {
    private String name;
    private int age;
}

// New schema with additional fields
public class Customer {
    private String name;
    private int age;
    private boolean isActive;
}
```
x??

---
#### Tying Consumers to Specific Technology Choices
Background context explaining the concept. When services directly interact with a database, they are tied to specific technology choices such as relational databases or nonrelational ones. This tight coupling can cause problems if there is a need for a different storage solution in the future.

If your system starts using a relational database but later requires a NoSQL database due to scalability issues, consumers will be tightly coupled to their current implementation.
:p How does direct DB integration tie consumers to specific technology choices?
??x
Direct DB integration ties consumers to specific technology choices because:
1. Consumers are dependent on the database schema and structure.
2. Changing the storage solution (e.g., from a relational to a NoSQL database) can require significant changes in consumer code.

For example, if you initially use JPA for a relational database but later need to switch to MongoDB:
```java
// JPA Entity
@Entity
public class Customer {
    @Id
    private Long id;
    private String name;
}

// Potential change when moving to MongoDB (example)
MongoCollection<Document> customerCollection = db.getCollection("customers");
Document document = new Document("name", "John Doe").append("_id", 1L);
customerCollection.insertOne(document);
```
x??

---
#### Impact on Logic and Cohesion
Background context explaining the concept. Direct manipulation of database records by consumers leads to scattered logic, making it difficult to maintain a cohesive service.

If multiple services (e.g., warehouse, registration UI, call center) all need to edit customer information, changes must be propagated across these different points.
:p How does direct DB integration affect the behavior and cohesion of services?
??x
Direct DB integration affects behavior and cohesion because:
1. Logic for manipulating data is spread across multiple consumers (services).
2. Any bug or change in logic needs to be fixed in multiple places, leading to poor code organization.

For example, if you need to update a customer's address through different entry points:
```java
// Warehouse service
public void updateCustomerAddress(Long id, String newAddress) {
    // Database call to update address
}

// Registration UI service
public void updateCustomerAddress(Long id, String newAddress) {
    // Database call to update address
}
```
x??

---
#### Breaking Strong Cohesion and Loose Coupling
Background context explaining the concept. The core principles of microservices include strong cohesion (where a service has a single responsibility) and loose coupling (services interact minimally). Direct database integration undermines these principles by exposing internal representations.

Consumers are directly manipulating DB records, leading to tight coupling and poor cohesion.
:p How does direct DB integration break the core principles of good microservices?
??x
Direct DB integration breaks the core principles of good microservices by:
1. Breaking strong cohesion: Services have multiple responsibilities (data manipulation and business logic).
2. Reducing loose coupling: Consumers are tightly coupled to specific implementation details, making changes difficult.

For example, if a customer service needs to add new functionality but is tightly coupled to the database:
```java
// Original Customer Service
public void updateCustomer(Customer customer) {
    // Logic for updating customer in DB
}

// New feature requiring changes
public void changeCustomerStatus(Long id, boolean status) {
    // This requires changing multiple places where customer information is updated
}
```
x??

---

