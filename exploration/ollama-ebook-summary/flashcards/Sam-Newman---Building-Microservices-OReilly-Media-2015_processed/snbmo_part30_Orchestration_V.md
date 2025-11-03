# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 30)

**Starting Chapter:** Orchestration Versus Choreography

---

#### Synchronous Communication
Synchronous communication involves making a call to a remote server which blocks until the operation completes. This means that the caller waits for the response before proceeding further.
:p What is synchronous communication?
??x
In synchronous communication, a client sends a request and waits for a response before moving on. The process is blocking, meaning the client cannot do anything else while waiting for the result. 
```java
public String makeSynchronousCall() {
    // Simulate making a call to a remote server
    Thread.sleep(1000);  // Blocking operation
    return "Response from Server";
}
```
x??

---

#### Asynchronous Communication
Asynchronous communication involves calling a service where the caller doesn’t wait for the operation to complete before returning, and may not even care whether or not the operation completes at all. It’s useful in scenarios where keeping a connection open is impractical due to network latency.
:p What is asynchronous communication?
??x
In asynchronous communication, the client sends a request without waiting for a response. The process is non-blocking; the client continues execution and may be notified later via a callback or event handler if necessary. 
```java
public void makeAsynchronousCall() {
    // Simulate making an asynchronous call to a remote server
    new Thread(() -> {
        try {
            Thread.sleep(1000);  // Simulating network delay
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("Operation completed");
    }).start();  // Non-blocking operation
}
```
x??

---

#### Request/Response Model
The request/response model involves a client initiating a request and waiting for the response. It can work with both synchronous and asynchronous communication.
:p What is the request/response model?
??x
In the request/response model, a client sends a request to a service and waits for a response before proceeding. This model aligns well with synchronous communication but also supports asynchronous calls where clients register callbacks or event listeners. 
```java
public String processRequest(String input) {
    // Simulate processing and returning a response
    Thread.sleep(1000);  // Blocking operation
    return "Processed: " + input;
}
```
x??

---

#### Event-Based Collaboration
Event-based collaboration involves the client emitting events, and other parties react to these events. The model is inherently asynchronous.
:p What is event-based collaboration?
??x
In event-based collaboration, a client emits an event (like "Customer created") which other services can subscribe to. These subscribers then handle the event based on their specific logic. This approach is highly decoupled and leverages asynchronous communication.
```java
public void emitEvent(String eventType) {
    // Simulate emitting an event
    System.out.println(eventType + " event emitted");
}
```
x??

---

#### Orchestration vs Choreography
Orchestration involves a central brain (like a service) guiding the process through request/response calls. Choreography involves services independently reacting to events they subscribe to.
:p What is orchestration?
??x
In orchestration, a central brain (often a specific service like Customer Service in our example) manages and coordinates multiple steps of a process using request/response calls. This approach can be simpler but can centralize too much logic into one place.
```java
public void handleCustomerCreation() {
    // Simulate calling other services via request/response
    loyaltyPointsBank.createRecord();
    emailService.sendWelcomeEmail();
    postalService.sendWelcomePack();
}
```
x??

---

#### Orchestration vs Choreography (continued)
Orchestration involves a central service guiding the process, while choreography has each component independently reacting to events. Choreography is more decoupled but requires additional monitoring for tracking processes.
:p What is choreography?
??x
In choreography, services independently react to events they subscribe to without relying on a centralized orchestrator. Each service understands its role and reacts accordingly. This approach leads to highly decoupled systems but requires extra effort for process monitoring.
```java
public void handleCustomerCreation() {
    // Simulate emitting an event
    emitEvent("Customer created");
}
```
x??

---

---
#### Remote Procedure Calls (RPC) Overview
Background context: RPC is a technique for making local calls that execute on remote services. It aims to make remote method calls appear as if they were local, simplifying development by abstracting network communications.

:p What are some key characteristics of RPC technologies?
??x
Some key characteristics include:
- The goal is to hide the complexity of remote calls behind a local call interface.
- They can be binary (like Java RMI, Thrift) or use XML for message formats (like SOAP).
- Some rely on an interface definition like WSDL, while others require tighter coupling.

These technologies often provide client and server stub generation tools which help in rapid development but come with downsides such as tight platform coupling.
x??

---
#### Platform Coupling in RPC
Background context: Many RPC mechanisms are tightly coupled to specific platforms or languages, limiting interoperability. For example, Java RMI ties both the client and server to the JVM.

:p How does platform coupling affect RPC implementations?
??x
Platform coupling can restrict which technologies can be used on the client and server sides. Using an interface like Java RMI means that both parties must use Java, which can limit flexibility but also provide tighter integration.

This tight coupling can expose internal technical implementation details, reducing interoperability between different platforms.
x??

---
#### Performance Considerations in RPC
Background context: Local calls are much faster than remote ones due to the overhead of marshalling and un-marshalling payloads and network latency. RPC implementations need to handle these differences carefully when designing APIs.

:p How do local and remote method calls differ, and why is this important for API design?
??x
Local method calls are fast because they occur in-process without any network communication. Remote procedure calls, however, involve significant overhead due to serialization/deserialization and network latency. This means that the performance characteristics of local methods cannot be assumed when designing remote interfaces.

For instance:
```java
// Local call example (fast)
public Customer findCustomer(String id) {
    // In-process call
}

// Remote call example (potentially slow)
public Customer findCustomer(String id) throws RemoteException {
    // Network communication and serialization/deserialization overhead
}
```
Developers must consider these differences when designing APIs to ensure efficient performance.
x??

---
#### Brittleness in RPC Implementations
Background context: Some popular RPC implementations, such as Java RMI, can introduce brittleness into the system due to their tight coupling and reliance on fixed method signatures.

:p What are the consequences of changing a remote API's methods in an RPC implementation?
??x
Changes to remote APIs can lead to significant deployment issues. For example, adding or removing fields in a class that is serialized over the network requires updating both client and server components simultaneously.

Consider:
```java
// Original method signature
public Customer findCustomer(String id) throws RemoteException;
```
Adding a new method:
```java
// New method signature
public Customer findCustomer(String emailAddress) throws RemoteException;
```
This change necessitates regenerating stubs, which can cause compatibility issues with existing clients.

To mitigate this:
- Design APIs carefully to avoid frequent changes.
- Use versioning strategies to manage backward and forward compatibility.
x??

---
#### Network Reliability in RPC
Background context: Networks are inherently unreliable and can fail in various ways. This unreliability must be considered when designing resilient systems using RPC.

:p How should developers handle network failures in their RPC-based applications?
??x
Developers should assume that networks are prone to failure, whether fast or slow, and can even corrupt data packets. Key considerations include:
- Implementing retry mechanisms.
- Handling timeouts gracefully.
- Designing stateless services where possible to reduce dependency on network reliability.

For example:
```java
public class NetworkResilientService {
    public Customer findCustomer(String id) throws Exception {
        try {
            return remoteFindCustomer(id);
        } catch (RemoteException e) {
            log.error("Network error, retrying...", e);
            return retryFindCustomer(id); // Retry logic
        }
    }

    private Customer remoteFindCustomer(String id) throws RemoteException {
        // Actual RPC call
    }

    private Customer retryFindCustomer(String id) throws Exception {
        // Retry implementation
    }
}
```
This ensures that the application can handle transient network issues.
x??

---

#### RPC Overview
Background context: Remote Procedure Call (RPC) is a method of communication between different processes or programs. It allows a program to call functions on another computer (or even the same computer, if they are separate processes) without knowing anything about where that function is implemented.

:p What does RPC stand for and what does it allow in programming?
??x
RPC stands for Remote Procedure Call, which allows a program to invoke a procedure on a different system over a network. It essentially abstracts away the complexity of remote communication.
x??

---

#### Shortcomings of RPC
Background context: While RPC can be beneficial for certain operations, some implementations and common practices lead to several issues such as tight coupling between client and server code, difficulty in upgrading the server interface without affecting clients, and potential pitfalls related to network latency and failure.

:p What are some common shortcomings or challenges with using RPC?
??x
Some common challenges include:
- Tight coupling of client and server interfaces.
- Difficulty in evolving the server without forcing synchronized upgrades from all clients.
- Potential for significant overhead due to network latency and reliability issues.
- Need for consistent release cycles between client and server versions.
x??

---

#### Protocol Buffers or Thrift
Background context: To mitigate some of the shortcomings of traditional RPC, modern mechanisms like Protocol Buffers (protobuf) and Apache Thrift are used. These tools allow defining interface descriptions in a language-neutral manner that can be used by different programming languages to generate corresponding client and server code.

:p What are Protocol Buffers and Thrift, and why are they useful?
??x
Protocol Buffers and Thrift are tools for serializing structured data between applications or systems written in different languages. They help avoid the need for lock-step releases of client and server code by generating language-specific implementations from a common interface definition.

For example:
- Protocol Buffers: Define message formats using `.proto` files, which can be used to generate corresponding code in various languages.
```java
// Example .proto file snippet
message Customer {
  int32 id = 1;
  string name = 2;
}
```

- Thrift: Similar functionality but with a different syntax and tooling. Both tools help in reducing the boilerplate code for network communication.

These tools allow more flexible interface evolution, making it easier to add or modify methods without breaking existing client implementations.
x??

---

#### REST Overview
Background context: Representational State Transfer (REST) is an architectural style inspired by the web, where resources are represented and manipulated using standard HTTP methods like GET, POST, PUT, DELETE. The focus on resource-based interactions makes it a good fit for microservices architectures.

:p What is REST, and how does it differ from RPC?
??x
REST is an architectural style focused on stateless communication based on the representation of resources. It differs from RPC in that:
- Resources are central: Clients interact with specific resources rather than calling procedures.
- Standard protocols: Uses HTTP for requests and responses (GET, POST, PUT, DELETE).
- Decoupling: The internal storage format is independent of the external representation.

Example REST interaction using HTTP methods:
```java
// Example of a simple GET request to fetch customer data
public Customer getCustomer(int id) {
    String url = "http://api.example.com/customers/" + id;
    // Use HTTP client library to send GET request and parse response
    return customerData; // Parsed JSON or XML into Customer object
}
```
x??

---

#### Resource Concept in REST
Background context: In REST, a resource is something that the service exposes for manipulation. Resources can be anything such as a Customer, Product, Order, etc. The key idea is to treat these entities uniformly and represent them with unique URIs (Uniform Resource Identifiers).

:p What is a resource in REST architecture?
??x
A resource in REST refers to any object or entity that the server exposes for manipulation through standard HTTP methods. For example, a customer record can be treated as a resource. Resources are uniquely identified by URIs and can have different representations (e.g., JSON, XML).

Example URI: `http://api.example.com/customers/12345`

Interactions with this resource might include:
- GET `/customers/12345` to fetch the customer details.
- PUT `/customers/12345` to update the customer information.
- DELETE `/customers/12345` to remove the customer record.

The representation of the resource (e.g., in JSON) is independent of its internal storage format.
x??

---

#### HTTP and REST
Background context: While REST can use various underlying protocols, it is most commonly implemented over HTTP due to its built-in support for CRUD operations through standard verbs like GET, POST, PUT, DELETE. These methods are well-defined by the HTTP specification.

:p How does HTTP fit into the REST architecture?
??x
HTTP fits into the REST architecture as a means of transmitting resource representations between client and server. The HTTP verbs (GET, POST, PUT, DELETE) map directly to basic operations on resources in REST:

- GET: Retrieve a representation of a resource.
- POST: Create or update a resource.
- PUT: Update an existing resource.
- DELETE: Remove a resource.

Example HTTP interaction for creating a new customer:
```java
// Example using HttpClient library
HttpClient httpClient = HttpClient.newHttpClient();
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("http://api.example.com/customers"))
    .header("Content-Type", "application/json")
    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
    .build();

HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
// Process the HTTP response
```
x??

---

#### GET vs POST Methods
Background context explaining the use of GET and POST methods. The GET method is idempotent, meaning it can be called any number of times without different results each time (e.g., retrieving a customer's details). The POST method is used to create new resources on the server.
:p What are the key differences between GET and POST methods?
??x
The GET method retrieves existing resources from the server in an idempotent way, meaning multiple identical requests will have no additional effect beyond a single request. On the other hand, the POST method creates new resources on the server and is not idempotent because sending the same request more than once can result in different outcomes.
???x
The answer with detailed explanations.
In essence, GET is used for fetching data without changing state (like customer details), while POST is used for creating or updating data (like a new customer record). Here’s a simple pseudocode to illustrate:

```pseudocode
// Example of using GET and POST methods in pseudo-code
function getCustomerDetails(id) {
    // Makes an idempotent request to retrieve customer details
    sendGETRequest("https://api.example.com/customers/" + id)
}

function createNewCustomer(customerData) {
    // Creates a new resource on the server by sending POST data
    sendPOSTRequest("https://api.example.com/customers", customerData)
}
```

This illustrates how GET and POST methods are used in different scenarios to achieve idempotent and non-idempotent operations respectively.
????

---

#### HTTP Ecosystem Benefits
Background context explaining the benefits of using HTTP, including caching proxies like Varnish, load balancers like mod_proxy, and various monitoring tools that support HTTP natively. The ecosystem provides numerous security controls such as basic auth and client certificates.
:p What are some key benefits of using HTTP for web services?
??x
The primary benefits include the ability to use existing infrastructure like caching proxies (Varnish) and load balancers (mod_proxy), which can help handle large volumes of traffic efficiently. Additionally, the ecosystem supports various security measures such as basic authentication and client certificates out of the box.
???x
The answer with detailed explanations.
Using HTTP for web services leverages a robust ecosystem that includes tools like Varnish for caching responses to reduce load on servers and mod_proxy for distributing requests across multiple servers, enhancing scalability. Security-wise, built-in features such as basic authentication (using headers) or client certificates provide a secure connection layer.

Here’s an example of using HTTP Basic Auth in Java:
```java
import java.net.HttpURLConnection;
import java.net.URL;

public class HttpClient {
    public static void makeRequest(String username, String password) throws Exception {
        URL url = new URL("https://api.example.com/resource");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET"); // Or "POST" for creating resources
        conn.setRequestProperty("Authorization", "Basic " + 
            new String(Base64.getEncoder().encode((username + ":" + password).getBytes())));
        int responseCode = conn.getResponseCode();
        if (responseCode == HttpURLConnection.HTTP_OK) {
            // Handle success
        } else {
            // Handle failure
        }
    }
}
```
This code demonstrates how to use HTTP Basic Auth by encoding the credentials and sending them in the request header.
????

---

#### Hypermedia as the Engine of Application State (HATEOAS)
Background context explaining HATEOAS, which is a principle that encourages clients to navigate resources through hyperlinks. The concept involves rich content with links to other pieces of content, similar to how web pages work today.
:p What does "Hypermedia as the Engine of Application State" (HATEOAS) mean?
??x
HATEOAS means that client applications should interact with servers by following links embedded within hypermedia representations. Instead of hardcoding specific URIs or endpoints in client code, HATEOAS allows clients to discover and navigate resources dynamically via hyperlinks.
???x
The answer with detailed explanations.
In essence, HATEOAS is a design principle that promotes state transitions and interactions through links present in the responses from the server. This approach minimizes coupling between the client and server by allowing the server to dictate which actions are available.

For example, consider an API response containing hyperlinks:
```json
{
    "id": 123,
    "name": "John Doe",
    "actions": [
        {
            "href": "/customers/123/orders",
            "rel": "orders"
        },
        {
            "href": "/customers/123/edit",
            "rel": "edit"
        }
    ]
}
```
The client can navigate to `/customers/123/orders` or `/customers/123/edit` based on the `actions` links provided in the JSON response, without needing to know these URIs beforehand.

This dynamic linking is a key feature of modern web applications and APIs like RESTful services.
????

---

#### Hypermedia Controls and Shopping Carts Analogy

Background context: The text discusses how hypermedia controls, similar to the familiar shopping cart on websites, enable users (both human and electronic) to interact with web services in a more intuitive way. These controls provide links or references that guide clients (like browsers or apps) to perform actions such as navigating to another page or executing an operation.
:p What is the analogy used to explain hypermedia controls?
??x
The analogy compares hypermedia controls to shopping carts on websites, where users can still recognize and interact with them even if their appearance changes over time. This comparison illustrates how clients need to understand implicit contracts, such as knowing that a cart contains items for purchase.
x??

---

#### Hypermedia Controls in MusicCorp Example

Background context: The text provides an example of hypermedia controls used on a MusicCorp album listing page. These controls help the client (e.g., a browser or app) perform actions like navigating to an artist's page or purchasing an album by understanding and following specific links with defined relations.
:p What are the two hypermedia controls mentioned in the MusicCorp example?
??x
The two hypermedia controls mentioned are:
1. A link with `rel="/artist"` that navigates to the artist’s page.
2. A link with `rel="/instantpurchase"` that allows instant purchase of the album.
x??

---

#### Decoupling Between Client and Server

Background context: The text explains how hypermedia controls provide a high level of decoupling between clients and servers by allowing changes in the underlying implementation without breaking client functionality. This is achieved through understanding and following specific links or references, similar to how humans recognize and interact with shopping carts.
:p How does hypermedia control help achieve decoupling?
??x
Hypermedia controls help achieve decoupling by abstracting the client from the detailed implementation of the server. Clients can navigate using defined relations (e.g., artist or instantpurchase) without needing to know specific URIs, making it easier for both parties to change their implementations over time.
```java
// Example pseudo-code for following a hypermedia control
public void followHyperlink(String rel, String href) {
    if ("artist".equals(rel)) {
        navigateToArtistPage(href);
    } else if ("instantpurchase".equals(rel)) {
        purchaseAlbum(href);
    }
}
```
x??

---

#### Progressive Discovery of the API

Background context: The text emphasizes that following links through hypermedia controls allows clients to progressively discover the available operations and endpoints, which is beneficial for implementing new clients. However, this approach can be "chatty," meaning it may involve more requests than a direct method.
:p What does progressive discovery mean in the context of API usage?
??x
Progressive discovery means that clients use hypermedia controls (links) to explore and interact with different parts of an API over multiple steps or requests. This allows for gradual learning of the available operations without needing all information upfront, which is useful for implementing new clients.
x??

---

#### Trade-offs in Using Hypermedia Controls

Background context: The text discusses both benefits and potential drawbacks of using hypermedia controls, such as increased flexibility but also "chattiness." It suggests starting with this approach and optimizing only if necessary due to performance concerns.
:p What is a trade-off mentioned for using hypermedia controls?
??x
A trade-off mentioned is the potential increase in "chattiness," meaning more requests or interactions required by clients to discover all available operations. While this can be less efficient, it provides greater flexibility and decoupling between client and server implementations.
```java
// Example pseudo-code for handling multiple API calls
public void exploreAPI() {
    followHyperlink("artist", "/artist/theBrakes");
    followHyperlink("instantpurchase", "/instantPurchase/1234");
}
```
x??

---

#### Initial Work Required

Background context: The text acknowledges that using hypermedia controls requires initial setup and understanding, but the benefits often come later. This is a common challenge in adopting new technologies or approaches.
:p Why might not everyone be sold on using hypermedia controls?
??x
Not everyone may be enthusiastic about using hypermedia controls because there is an initial upfront work required to understand and implement this approach. While the long-term benefits include better decoupling and flexibility, these advantages often take time to materialize.
x??

---

