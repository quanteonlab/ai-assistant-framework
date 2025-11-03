# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 6)

**Starting Chapter:** Downsides to REST Over HTTP

---

#### JSON vs XML
Background context explaining the differences and usage between JSON and XML. JSON is simpler and more lightweight, but lacks certain standards like hypermedia controls that are available in XML.

:p What is the difference between JSON and XML?
??x
JSON (JavaScript Object Notation) is a lightweight data-interchange format inspired by JavaScript object literal notation. It is easier to write and read for humans compared to XML. JSON uses key-value pairs enclosed in curly braces, making it simpler and more compact than XML. However, JSON does not natively support hypermedia controls like link elements that are easily handled in XML.

XML (eXtensible Markup Language) on the other hand, is a markup language designed for transporting and storing data. It uses tags to wrap around text to identify different types of information. XML supports more complex structures and has built-in mechanisms for handling hypermedia controls using `link` elements or other custom elements.

Example JSON:
```json
{
  "name": "John",
  "age": 30,
  "city": "New York"
}
```

Example XML with link element:
```xml
<root>
  <name>John</name>
  <age>30</age>
  <link rel="self" href="/john"/>
  <city>New York</city>
</root>
```
x??

---

#### HAL (Hypertext Application Language)
Background context explaining the need for a standard way to handle hypermedia controls in JSON, and how HAL addresses this.

:p What is HAL and why is it important?
??x
HAL (Hypertext Application Language) is a specification that aims to provide a consistent way of handling hypermedia controls within JSON responses. It addresses the lack of native support for link elements similar to XML's `<link>` tag in JSON. HAL provides a standard format for representing links and other metadata, which can help in creating more maintainable and flexible RESTful services.

Example HAL:
```json
{
  "_links": {
    "self": { "href": "/resource/1" },
    "collection": { "href": "/resources" }
  }
}
```
x??

---

#### XML Tool Support
Background context explaining the advantages of using XML, particularly in terms of tool support for handling payloads.

:p What are some advantages of using XML over JSON?
??x
XML has several advantages when compared to JSON:

1. **Tool Support**: XML is often better supported by existing tools and libraries.
2. **XPath and CSS Selectors**: XML supports XPath, a powerful query language that allows you to extract specific parts of the payload easily. Additionally, CSS selectors can be used as an alternative or complement to XPath, making it even easier for developers who are familiar with HTML.
3. **Standardization**: XML has been around longer and is more standardized, meaning there are more established tools and practices.

Example using XPATH in XML:
```xml
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <body>
        <h2>Users</h2>
        <table border="1">
          <tr bgcolor="#9acd32">
            <th>Name</th>
            <th>Email</th>
          </tr>
          <xsl:for-each select="users/user">
            <tr>
              <td><xsl:value-of select="name"/></td>
              <td><xsl:value-of select="email"/></td>
            </tr>
          </xsl:for-each>
        </table>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
```
x??

---

#### REST Frameworks and Overhead
Background context explaining the trade-offs when using REST frameworks, including potential coupling issues.

:p What are some downsides to using REST frameworks like Spring Boot?
??x
While REST frameworks can simplify development and reduce boilerplate code, they often come with a set of assumptions that can lead to tight coupling between the service implementation and its external interface. For example, some frameworks might promote directly exposing database representations as API endpoints without proper abstraction.

This approach can cause problems later on when you need to change the internal data structures or storage mechanisms, as it locks you into specific implementations too early in the development process.

Example of tight coupling:
```java
@RestController
public class UserResource {
    @Autowired
    private UserRepository userRepository;

    @GetMapping("/users")
    public List<User> getUsers() {
        return userRepository.findAll();
    }
}
```
In this example, the `UserRepository` directly interacts with a database, making changes to storage mechanisms harder.

A better approach would be to abstract data access and expose only necessary interfaces:
```java
@RestController
public class UserResource {
    @Autowired
    private UserService userService;

    @GetMapping("/users")
    public List<UserDto> getUsers() {
        return userService.getAllUsers();
    }
}
```
x??

---

#### HTTP Overhead
Background context explaining the potential overhead of using REST over HTTP, especially in scenarios requiring low latency.

:p Why might REST over HTTP be a poor choice for low-latency communications?
??x
REST over HTTP can introduce significant overhead due to its reliance on the full HTTP protocol stack. Each request incurs a round-trip time that includes:

1. **HTTP Handshake**: Establishing a connection and negotiating headers.
2. **Serialization/Deserialization Overhead**: Converting between different data formats (e.g., JSON, XML).
3. **Security Headers**: Including authentication and authorization checks.

These overheads can be substantial in scenarios where extremely low-latency or small message sizes are critical.

For example, consider a WebSocket scenario:
```java
@ServerEndpoint("/websocket")
public class WebSocketHandler {
    @OnOpen
    public void onOpen(Session session) {
        System.out.println("New client connected: " + session.getId());
    }

    @OnMessage
    public String handleMessage(String message) {
        // Process and return the message
        return "Echo: " + message;
    }
}
```
In this case, WebSockets provide a more efficient way to stream data directly between client and server without the overhead of HTTP.

x??

---

#### Message Broker Overview
Message brokers like RabbitMQ serve as intermediaries between producers and consumers of events. They handle subscriptions, ensuring that consumers are informed when an event arrives. Brokers can also manage state, such as tracking which messages a consumer has already seen.

:p What is the role of message brokers in asynchronous communication?
??x
Message brokers act as intermediaries for publishing and subscribing to events. They help ensure that producers can send events and consumers can be notified about those events efficiently. Additionally, they handle state management, keeping track of what messages have been consumed by each subscriber.

For example, consider a scenario where an order is placed in an e-commerce system:
- A message broker could be used for the order service to emit an event when an order is created.
- Other services (such as inventory or email notification) can subscribe to this event and take appropriate actions.

```java
// Pseudocode example of emitting an event using a message broker
public class OrderService {
    private MessageBroker messageBroker;

    public void createOrder(Order order) {
        // Emit the event through the message broker
        messageBroker.publish("order.created", order);
    }
}

// Example subscription in another service to consume events
public class InventoryService {
    private MessageBroker messageBroker;
    private Set<String> seenOrders = new HashSet<>();

    public void subscribeToOrderEvents() {
        // Subscribe to specific event types
        messageBroker.subscribe("order.created", this::handleOrderCreated);
    }

    private void handleOrderCreated(Order order) {
        if (!seenOrders.contains(order.getId())) {
            // Process the order only once
            seenOrders.add(order.getId());
            // Update inventory or perform other actions
        }
    }
}
```
x??

---

#### ATOM for Event Propagation
ATOM is a REST-compliant specification that defines semantics for publishing feeds of resources. It allows services to publish events and consumers to poll the feed for updates.

:p Can you describe how ATOM can be used for event propagation?
??x
ATOM can be used to propagate events by allowing services to publish updates as resources in a feed, which can then be consumed by other services that poll this feed. This approach leverages HTTP, making it easy to integrate with existing systems and benefiting from its scalability.

For instance, if the customer service changes (e.g., a new customer is added), the customer service could publish an event to a feed. Consumers could periodically check this feed for any changes using simple HTTP requests.

```java
// Pseudocode example of publishing events using ATOM
public class CustomerService {
    private HttpClient httpClient;
    private String atomFeedUrl;

    public void addCustomer(Customer customer) {
        // Create the Atom entry for the new customer
        String xmlEntry = generateAtomEntry(customer);
        
        // Post the XML to the feed URL
        Response response = httpClient.post(atomFeedUrl, xmlEntry);
        if (response.isSuccess()) {
            System.out.println("Customer added successfully.");
        } else {
            System.out.println("Failed to add customer: " + response.getErrorMessage());
        }
    }

    private String generateAtomEntry(Customer customer) {
        // Generate the Atom XML entry for the customer
        return "<entry><title>" + customer.getName() + "</title><content type='text'>" + customer.getDescription() + "</content></entry>";
    }
}

// Pseudocode example of consuming events from ATOM feed
public class ConsumerService {
    private HttpClient httpClient;
    private String atomFeedUrl;

    public void checkForNewCustomers() {
        // Fetch the Atom feed and parse for new entries
        Response response = httpClient.get(atomFeedUrl);
        if (response.isSuccess()) {
            String xmlContent = response.getContent();
            List<Customer> newCustomers = parseAtomEntries(xmlContent);
            processNewCustomers(newCustomers);
        } else {
            System.out.println("Failed to fetch the feed: " + response.getErrorMessage());
        }
    }

    private List<Customer> parseAtomEntries(String xmlContent) {
        // Parse the Atom XML content and extract customer entries
        // Return a list of new customers found in the feed
        return new ArrayList<>();
    }

    private void processNewCustomers(List<Customer> newCustomers) {
        for (Customer customer : newCustomers) {
            System.out.println("Processing new customer: " + customer.getName());
            // Perform any necessary actions with the new customer
        }
    }
}
```
x??

---

#### Competing Consumer Pattern in ATOM
The Competing Consumer pattern involves bringing up multiple worker instances to compete for messages. This approach is useful for scaling the number of workers to handle a list of independent jobs, but it introduces complexity if shared state needs to be managed.

:p How does the Competing Consumer pattern work with ATOM events?
??x
In the Competing Consumer pattern, you bring up multiple worker instances that can all compete for messages. This is useful when you need to scale the number of workers to handle a list of independent jobs efficiently. However, if two or more consumers see the same message, it could lead to redundant processing.

With ATOM, this means managing shared state among all the competing consumers to reduce the likelihood of duplicate work. For example, each consumer needs to keep track of which messages have already been processed and avoid reprocessing them.

```java
// Pseudocode example of implementing Competing Consumer with ATOM
public class WorkerService {
    private MessageBroker messageBroker;
    private Set<String> seenMessages = new HashSet<>();

    public void startConsuming() {
        // Subscribe to the event feed
        messageBroker.subscribe("event.feed", this::processEvent);
    }

    private void processEvent(Event event) {
        if (!seenMessages.contains(event.getMessageId())) {
            // Mark as processed before handling
            seenMessages.add(event.getMessageId());
            handleEvent(event);
        }
    }

    private void handleEvent(Event event) {
        // Process the event
        System.out.println("Handling event: " + event.getType() + ", ID: " + event.getMessageId());
        // Perform necessary actions for this event
    }
}

// Example of how workers are started and compete for events
public class Application {
    public static void main(String[] args) {
        List<WorkerService> workers = new ArrayList<>();
        
        for (int i = 0; i < numberOfWorkers; i++) {
            WorkerService worker = new WorkerService();
            // Start the worker to consume events
            worker.startConsuming();
            workers.add(worker);
        }
    }
}
```
x??

---

---
#### Asynchronous Architecture Complexity
Background context: Asynchronous architectures, such as event-driven systems, can offer decoupled and scalable solutions but introduce complexities that need careful management. These complexities include handling long-running requests, managing message queues, and ensuring reliable processing of messages.

:p What are some challenges associated with asynchronous architectures?
??x
Challenges in asynchronous architectures include:
- Managing long-running async requests: Determining what to do when a response comes back, especially if the original node is no longer available.
- Handling competing consumer patterns: Workers may crash or fail to process messages correctly, leading to potential deadlocks and retry issues.
- Ensuring reliable message processing: Implementing mechanisms like dead letter queues (DLQ) for failed messages.

Code examples:
```java
// Pseudocode for handling a long-running async request
public void handleRequestAsync(String request) {
    try {
        // Logic to initiate the async process
        Future<String> result = processAsync(request);
        
        // Logic to handle the response when it comes back
        String response = result.get();
        log.info("Received response: " + response);
    } catch (ExecutionException | InterruptedException e) {
        log.error("Error processing request", e);
    }
}
```
x??

---
#### Catastrophic Failover Example
Background context: The author provides a real-world example from 2006 involving a bank's pricing system, which demonstrates the challenges of asynchronous architectures. A bug caused workers to crash repeatedly due to unhandled messages in a transacted queue.

:p What is a catastrophic failover and how did it manifest in this case?
??x
A catastrophic failover refers to a situation where a message processing system fails repeatedly, leading to continuous retries that can cause more issues. In the example provided:
- A bug caused certain pricing requests to crash workers.
- Transacted queues led to locks timing out and messages being re-added to the queue.
- Other workers would then pick up these failed messages and crash again, creating a cycle.

The issue was resolved by fixing the bug, setting retry limits, and implementing a message hospital (dead letter queue) for failed messages:
```java
// Pseudocode for handling dead letter queues
public void handleFailedMessage(String message) {
    // Check if maximum retries have been reached
    if (!messageProcessor.reachedMaxRetries(message)) {
        messageHospital.enqueue(message);
        log.warn("Failed to process message, enqueuing in hospital: " + message);
    } else {
        log.error("Exceeded max retries for message: " + message);
    }
}
```
x??

---
#### Importance of Monitoring and Correlation IDs
Background context: The author emphasizes the importance of having good monitoring tools and using correlation IDs to trace requests across different processes. This is crucial in complex, asynchronous systems.

:p Why are monitoring tools and correlation IDs important in asynchronous architectures?
??x
Monitoring tools help track system performance, detect failures, and ensure that all components are functioning correctly. Correlation IDs allow tracing of individual requests through a system, even when they span multiple services or processes.

For example:
- Monitoring can help identify patterns in failures, resource usage, and latency.
- Correlation IDs can be used to log and trace the flow of messages across different components.

```java
// Example of using correlation IDs for logging
public void processRequest(String request) {
    String corrId = UUID.randomUUID().toString();
    log.info("Processing request [id: " + corrId + ", request: " + request + "]");
    
    try {
        // Process the request asynchronously
        processAsync(request, corrId);
    } catch (Exception e) {
        log.error("Failed to process request [id: " + corrId + "]", e);
    }
}
```
x??

---

