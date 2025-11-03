# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 31)

**Starting Chapter:** Downsides to REST Over HTTP

---

#### JSON vs. XML
Background context explaining the differences between JSON and XML, their popularity, and use cases. Highlight the advantages and disadvantages of both formats.

:p What are the main differences between JSON and XML in terms of simplicity and ease of consumption?

??x
JSON is a simpler format compared to XML, making it easier for clients to consume resources over HTTP. It is more popular due to its lightweight nature and ease of parsing with tools like JSONPATH. However, JSON lacks the hypermedia control definitions that XML provides, such as link elements, which are crucial for RESTful services.

XML, on the other hand, offers better tool support, especially in terms of data extraction using XPATH or CSS selectors. This makes it easier to extract specific parts of the payload when versioning APIs. However, JSON is generally considered more compact and easier to handle due to its simpler syntax.

Example:
```json
{
  "name": "John Doe",
  "age": 30,
  "email": "johndoe@example.com"
}
```

vs.

Example:
```xml
<user>
    <name>John Doe</name>
    <age>30</age>
    <email>johndoe@example.com</email>
</user>
```
x??

---

#### HAL and Hypermedia Controls
Background context explaining the importance of hypermedia controls in RESTful services, especially with JSON. Introduce HAL as a standard for adding such controls to JSON payloads.

:p What is HAL and how does it address the lack of hypermedia control in JSON?

??x
HAL (Hypertext Application Language) is a standard that attempts to add hypermedia controls to JSON payloads, similar to how XML defines link elements. This makes it easier to navigate resources within RESTful services by providing links directly within the JSON payload.

For example:
```json
{
  "_links": {
    "self": { "href": "/users/123" },
    "avatar": { "href": "/users/123/avatar.jpg" }
  }
}
```

HAL provides a structured way to include links and other metadata within JSON responses, enhancing the discoverability of resources. Tools like the web-based HAL browser can help in exploring these hypermedia controls.

x??

---

#### Using HTML as an API Format
Background context on using HTML for both UI and API purposes, highlighting its benefits and drawbacks compared to XML or JSON.

:p Can you explain why some interfaces use HTML instead of XML or JSON?

??x
Using HTML as a format for APIs can be attractive because it can serve both as a user interface (UI) and an application programming interface (API). This dual functionality allows developers to leverage existing HTML parsers and tools. However, this approach has pitfalls since the interactions between humans and computers differ significantly.

For instance:
```html
<!DOCTYPE html>
<html>
  <body>
    <div class="user-profile">
      <h1>John Doe</h1>
      <p><img src="/users/123/avatar.jpg" /></p>
      <a href="/users">Back to Users List</a>
    </div>
  </body>
</html>
```

While this can be convenient, it may lead to issues such as misinterpretation of HTML semantics in the context of an API. It is important to ensure that interactions are clearly defined and separated to avoid confusion.

x??

---

#### Frameworks and Short-Term Gain
Background on how popular frameworks for RESTful services can sometimes promote bad practices by prioritizing ease of implementation over long-term maintainability.

:p Why do some frameworks encourage the direct exposure of database objects in API responses?

??x
Some frameworks, such as Spring Boot, make it very easy to directly expose database objects via APIs. While this can speed up initial development, it often leads to tight coupling between the data storage and the service's interface. This approach can cause significant issues in the long term due to the difficulty of changing the underlying data representation without affecting the API.

For example:
```java
@RestController
@RequestMapping("/users")
public class UserController {
    @GetMapping("/{id}")
    public User getUserById(@PathVariable Long id) {
        return userRepository.findById(id).orElseThrow(() -> new ResourceNotFoundException("User not found"));
    }
}
```

This practice can lead to maintenance problems and make it hard to evolve the API independently of changes in data storage. A better approach is to delay implementing proper persistence until the interface has stabilized, allowing the design to be driven by consumer needs rather than technical implementation details.

x??

---

#### REST Over HTTP Drawbacks
Background on the challenges associated with creating client stubs for RESTful services compared to RPC-based systems. Discuss the overhead of HTTP and potential performance issues.

:p What are some downsides of using REST over HTTP in terms of client consumption?

??x
One major downside of REST over HTTP is the difficulty in generating client stubs, unlike with RPC-based systems where tools can easily generate clients from service definitions. While HTTP client libraries provide excellent support for making requests, hypermedia controls require custom implementation and cannot leverage well-established client frameworks.

For instance:
- Lack of built-in support for hypermedia controls makes it challenging to automatically generate a client.
- Potential need to manually implement client logic that can lead to RPC-like behavior or shared client-server code, which introduces coupling issues.

Example:
```java
// Manual Hypermedia Handling in REST Client
public class MyRestClient {
    private RestTemplate restTemplate;

    public User getUserById(Long id) {
        String url = "/users/" + id;
        ResponseEntity<User> response = restTemplate.getForEntity(url, User.class);
        if (response.hasBody()) {
            return response.getBody();
        }
        throw new ResourceNotFoundException("User not found");
    }
}
```

The overhead of HTTP also includes the initial request and response headers, which can add latency. For low-latency applications, more lightweight protocols like WebSockets or direct TCP-based communication might be preferred.

x??

---

#### Message Brokers for Event-Based Communication
Message brokers like RabbitMQ offer a scalable and resilient way to handle event-based, asynchronous communication between microservices. They allow producers to publish events and consumers to subscribe to these events via APIs.

:p What are message brokers, and how do they facilitate asynchronous communication in microservices?
??x
Message brokers act as intermediaries that enable producers (publishers) of events to communicate with subscribers without needing direct interaction. This is particularly useful for implementing event-driven architectures where services need to react to external or internal events asynchronously.

Code example:
```java
// Producer (Publisher)
RabbitTemplate rabbitTemplate = new RabbitTemplate(connectionFactory);
rabbitTemplate.convertAndSend("queueName", "eventPayload");

// Consumer (Subscriber)
@RabbitListener(queues = "queueName")
public void handleEvent(String event) {
    // Process the event
}
```
x??

---
#### ATOM as an Event Propagation Method
ATOM is a REST-compliant specification that can be used to propagate events. It defines semantics for publishing feeds of resources, allowing clients to poll these feeds for changes.

:p How does ATOM differ from message brokers in propagating events?
??x
ATOM allows services to publish events by creating and consuming feeds. While it reuses the existing HTTP infrastructure and associated libraries, it lacks some of the built-in features provided by message brokers such as handling duplicate messages, managing shared state among consumers, and ensuring reliable delivery.

Code example:
```java
// Publishing an event using ATOM
Feed feed = new Feed();
Entry entry = new Entry();
entry.setTitle("New Customer Created");
feed.add(entry);
webClient.post()
    .uri("/events")
    .header(HttpHeaders.CONTENT_TYPE, AtomMediaType.ATOM_XML_VALUE)
    .body(BodyInserters.fromValue(feed))
    .retrieve()
    .toBodilessEntity();

// Consuming the feed
ReactiveFeedsClient client = FeedsClient.create();
client.subscribe("/events", event -> {
    System.out.println("Received event: " + event.getTitle());
});
```
x??

---
#### Competing Consumer Pattern with ATOM
The Competing Consumer pattern is used to scale up the number of workers handling independent jobs by having multiple consumers compete for messages. However, managing shared state among these consumers can be complex.

:p What challenges does using the Competing Consumer pattern face when implementing ATOM-based event propagation?
??x
Using the Competing Consumer pattern with ATOM requires careful management of shared state to avoid duplicate processing. Unlike message brokers, where a standard queue automatically handles this issue, ATOM consumers must implement their own logic to track seen messages and coordinate among themselves.

Code example:
```java
// Pseudocode for Competing Consumers in ATOM
List<Consumer> consumers = new ArrayList<>();
for (int i = 0; i < numberOfConsumers; i++) {
    Consumer consumer = new Consumer();
    consumers.add(consumer);
}

while (!allMessagesProcessed()) {
    Entry entry = pollFeedEntries();
    List<Consumer> winners = selectWinningConsumers(entry);
    
    for (Consumer winner : winners) {
        processEntry(winner, entry);
    }
}
```
x??

---
#### Synchronous vs. Asynchronous Communication
The same considerations apply to the encoding of events as with requests and responses in synchronous communication. JSON is often a suitable choice for encoding asynchronous event data.

:p How does synchronous request/response differ from asynchronous event-based communication when it comes to encoding messages?
??x
In both synchronous and asynchronous communication, JSON can be used effectively for encoding messages. The primary difference lies in the handling of these messages: synchronous methods involve immediate acknowledgment and response, while asynchronous methods rely on event subscriptions and callbacks.

Code example:
```java
// Synchronous Request/Response Example
public String sendRequest(String request) {
    // Send request to server
    String response = sendAndReceive(request);
    return response;
}

// Asynchronous Event Handling Example
@RabbitListener(queues = "eventQueue")
public void handleEvent(String event) {
    System.out.println("Received event: " + event);
}
```
x??

---

---
#### Event-Driven Architecture Complexity
Event-driven architectures offer more decoupled and scalable systems, but they also introduce complexities related to managing messages. This includes challenges with long-running async requests and short-lived async tasks.

:p What are some of the complexities associated with event-driven architectures?
??x
Some key complexities include handling response management, ensuring proper message delivery in distributed systems, dealing with transient failures like node down scenarios, and implementing robust error recovery mechanisms. Additionally, monitoring and tracing across process boundaries become essential.

Code Example: Implementing a basic correlation ID for tracing requests.
```java
public class Request {
    private String correlationId;
    
    public Request(String correlationId) {
        this.correlationId = correlationId;
    }
    
    // Method to log the request with correlation ID
    public void logRequest() {
        System.out.println("Request logged with correlation ID: " + correlationId);
    }
}
```
x??

---
#### Competing Consumer Pattern and Catastrophic Failover
In an event-driven architecture, using a competing consumer pattern can lead to issues like catastrophic failover. This happens when messages are retried indefinitely without proper limits.

:p What is a catastrophic failover in the context of asynchronous architectures?
??x
A catastrophic failover occurs when a message processing system continuously retries messages that cause failures without any retry limit or error handling mechanism, leading to infinite loops and resource exhaustion. This can be exacerbated by bugs that cause workers to crash repeatedly on specific tasks.

Code Example: Implementing a simple retry mechanism with a maximum limit.
```java
public class Worker {
    private int maxRetries = 3;
    
    public void processMessage(String message) {
        for (int i = 0; i < maxRetries; i++) {
            try {
                // Process the message
                System.out.println("Processing message: " + message);
                break; // Successfully processed, exit loop
            } catch (Exception e) {
                if (i == maxRetries - 1) { // Last retry attempt
                    throw new RuntimeException("Failed to process message after " + maxRetries + " retries", e);
                }
                System.out.println("Message processing failed. Retrying...");
            }
        }
    }
}
```
x??

---
#### Dead Letter Queue (DLQ)
A dead letter queue is essential in event-driven systems for handling messages that fail multiple times due to unexpected issues.

:p What is a dead letter queue and why is it necessary?
??x
A dead letter queue (DLQ) is a separate queue where failed or problematic messages are moved after they have exceeded their retry limits. It helps in diagnosing and managing messages that cannot be processed correctly, providing a safety net for error-prone tasks.

Code Example: Redirecting messages to a DLQ.
```java
public class MessageQueueManager {
    private Map<String, Queue> queues = new HashMap<>();
    
    public void sendMessageToDLQ(String message) {
        String queueName = "deadLetterQueue";
        
        if (!queues.containsKey(queueName)) {
            queues.put(queueName, createQueue());
        }
        
        // Send the message to the DLQ
        send(message, queueName);
    }
    
    private Queue createQueue() {
        return new Queue(); // Assume this creates a new queue
    }
    
    private void send(String message, String queueName) {
        queues.get(queueName).send(message); // Simulate sending a message to the DLQ
    }
}
```
x??

---
#### Monitoring and Correlation IDs
Monitoring and using correlation IDs are crucial for tracing request paths in event-driven architectures. This ensures better visibility into the system's behavior.

:p Why is monitoring and correlation ID implementation important?
??x
Monitoring helps in tracking the performance, health, and flow of data through a distributed system. Correlation IDs allow developers to trace requests across different services or processes, making it easier to diagnose issues and understand the request lifecycle.

Code Example: Logging with correlation IDs for tracing.
```java
public class RequestLogger {
    private String correlationId;
    
    public RequestLogger(String correlationId) {
        this.correlationId = correlationId;
    }
    
    public void logRequest() {
        System.out.println("Request logged with correlation ID: " + correlationId);
    }
}
```
x??

---

