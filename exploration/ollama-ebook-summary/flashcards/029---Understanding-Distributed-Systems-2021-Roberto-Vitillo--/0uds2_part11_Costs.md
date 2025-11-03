# Flashcards: 029---Understanding-Distributed-Systems-2021-Roberto-Vitillo--_processed (Part 11)

**Starting Chapter:** Costs

---

#### Benefits of Microservices
Breaking down the backend by business capabilities into a set of services allows each service to be developed and operated independently. This approach can significantly increase development speed for several reasons:
- Smaller teams are more effective as communication overhead grows quadratically with team size.
- Each team has its own release schedule, reducing cross-team communication time.
- Smaller codebases make it easier for developers to understand and ramp up new hires.
- Smaller codebases also improve developer productivity by not slowing down IDEs.

:p What are the benefits of breaking the backend into microservices?
??x
Breaking the backend into microservices allows each service to be developed and operated independently, which can increase development speed. This is due to several factors:
- Effective communication among smaller teams.
- Independent release schedules for each team.
- Smaller codebases that make it easier for developers to understand the system and onboard new hires.
- Improved developer productivity as IDEs are not slowed down by larger codebases.

??x
---
#### Costs of Microservices
Embracing microservices adds more moving parts to the overall system, which comes at a cost. The benefits must be amortized across many development teams for this approach to be worthwhile. Some key challenges include:

- Development Experience: Using different languages, libraries, and data stores in each microservice can lead to an unmanageable application.
  - Example: Developers may find it challenging to switch between teams if the software stack is completely different.
- Resource Provisioning: Supporting a large number of independent services requires simple and automated resource management.
  - Example: Teams should not have their own unique ways of provisioning resources, but rather rely on automation.
- Communication: Remote calls are expensive and come with issues like failures, asynchrony, and batching.
  - Example: Developers need to implement defense mechanisms to handle remote call failures.

:p What are the costs associated with fully embracing microservices?
??x
The costs associated with fully embracing microservices include:
- Increased complexity due to more moving parts in the system.
- Development experience challenges when using different languages, libraries, and data stores across multiple teams.
- Resource provisioning challenges that require standardization and automation for simplicity and consistency.
- Communication overhead and performance hits from remote calls.

??x
---
#### Standardization in Microservices
Standardization is needed to avoid an unmanageable application where each microservice uses a different language, library, or data store. This can be achieved by encouraging specific technologies while still allowing some freedom:
- Example: Providing a great development experience for teams that use recommended languages and technologies.

:p How does standardization play a role in microservices?
??x
Standardization is crucial to avoid an unmanageable application where each microservice uses different languages, libraries, or data stores. This can be achieved by encouraging the use of specific technologies while still providing some flexibility:
- Example: Teams can have their own development experience as long as they stick with the recommended portfolio of languages and technologies.

??x
---
#### Resource Management in Microservices
Resource management is crucial to support a large number of independent services. This requires simple resource provisioning and configuration, which should be handled through automation:
- Example: You don’t want every team to come up with their own way of provisioning resources; instead, use automation tools.

:p What are the key aspects of resource management in microservices?
??x
Key aspects of resource management in microservices include:
- Simple and automated resource provisioning for teams.
- Configuring provisioned resources once they have been set up.
- Using automation tools to manage resources consistently across all services.

??x
---
#### Continuous Integration, Delivery, and Deployment
Continuous integration ensures that code changes are merged into the main branch after an automated build and test suite has run. Once a change is merged, it should be automatically deployed to a production-like environment where additional tests ensure no dependencies or use cases are broken:
- Example: Individual microservices can be tested independently of each other, but testing their integration is much harder.

:p What does continuous integration involve in the context of microservices?
??x
Continuous integration involves merging code changes into the main branch after an automated build and test suite has run. Once merged, the code should be automatically deployed to a production-like environment where additional tests ensure no dependencies or use cases are broken:
- Example: While testing individual microservices is not more challenging than in monolithic applications, testing their integration can be much harder.

??x
---

#### Service Operations Challenges
Background context: In a microservices architecture, individual services are developed and maintained by separate teams. However, these teams usually handle both development and operations for their respective services, which can create conflicts between adding new features and maintaining the service.

:p What are some challenges faced when using microservices in terms of operations?
??x
The primary challenges include:
- Staffing each team with its own dedicated operations team is expensive.
- Teams that develop a service are typically on-call for it, creating a conflict of interest during sprint planning.
- Debugging system failures can be difficult as you cannot load the entire application locally to step through it with a debugger.

Code examples can illustrate these challenges:
```java
// Example of a simple logging mechanism in Java
public class ServiceLogger {
    public void logException(Exception e) {
        System.out.println("Error: " + e.getMessage());
        // More sophisticated logging would be needed for production systems
    }
}
```
x??

---

#### Eventual Consistency
Background context: When splitting an application into separate services, the data model no longer resides in a single datastore. Ensuring atomic updates and strong consistency across multiple databases is slow, expensive, and difficult.

:p What does eventual consistency mean in microservices architecture?
??x
Eventual consistency means that after some period of time, all operations against a distributed system have completed and the data has propagated to all nodes so that eventually, if you read from any node, you will see the same data. This is necessary because splitting an application into separate services means that updates can't be guaranteed to happen atomically across multiple databases.

Code examples can illustrate eventual consistency:
```java
// Pseudocode for implementing eventual consistency using a queue system
public class EventualConsistencyManager {
    private Queue<String> operationsQueue = new LinkedList<>();

    public void addOperation(String operation) {
        operationsQueue.add(operation);
    }

    public String getLatestOperation() {
        return operationsQueue.peek();
    }
}
```
x??

---

#### Practical Considerations for Microservices
Background context: While microservices offer many benefits, they also introduce significant complexity. Teams need to carefully consider when and how to split a monolithic application into services.

:p Why is it generally best to start with a monolith before splitting it?
??x
It's generally best to start with a monolith because:
- Splitting an application introduces additional complexity that can be overwhelming.
- Finding the right boundaries between services requires experience and time, which are better spent in a monolithic environment where they can be refined.

Code examples might not directly apply here, but this could illustrate the process of refactoring a monolith to microservices:
```java
// Example code showing potential initial approach (monolithic)
public class MonolithicApplication {
    public void doSomething() {
        // Business logic that spans multiple layers or services in a monolith
    }
}
```
x??

---

#### Moving Boundaries Within a Monolith
Background context: Initially, the boundaries between services are harder to define and move within a monolith. As the application matures, it becomes easier to identify the optimal service boundaries.

:p How does moving boundaries within a monolith help in microservices architecture?
??x
Moving boundaries within a monolith helps by:
- Allowing teams to refactor parts of the application incrementally.
- Making it easier to experiment with different service designs without fully committing to splitting the entire application.
- Reducing initial complexity and risk when first introducing microservices.

Code examples might include refactoring code in a monolithic setup:
```java
// Refactoring part of the monolith into a microservice
public class UserService {
    private UserRepository userRepository = new UserRepository();

    public User getUserById(int id) {
        return userRepository.findById(id);
    }
}
```
x??

---

#### API Gateway Overview
The internal APIs of a microservices application can be hidden by a public API that acts as a facade or proxy for the internal services. This public API is called the API gateway and it provides a transparent interface to its clients, who are unaware they are communicating through an intermediary.

:p What is the role of an API gateway in a microservices architecture?
??x
The API gateway serves as an intermediary layer between external clients and the internal microservices. Its main roles include routing requests to appropriate backend services, providing a unified public API that abstracts away the complexity of interacting with multiple services, and offering features like composition and translation.

```java
// Example of simple routing logic in pseudocode
public class ApiGateway {
    private Map<String, String> routingMap;

    public ApiGateway(Map<String, String> routingMap) {
        this.routingMap = routingMap;
    }

    public Response handleRequest(Request request) {
        String path = request.getPath();
        if (routingMap.containsKey(path)) {
            // Route to the appropriate backend service
            return routeToBackendService(routingMap.get(path), request);
        } else {
            return new Response("404 Not Found");
        }
    }

    private Response routeToBackendService(String internalPath, Request request) {
        // Logic to send request to the correct microservice and handle its response
        // For simplicity, assume a function call to a backend service with given path
        BackendService backend = new BackendService(internalPath);
        return backend.handleRequest(request);
    }
}
```
x??

---

#### Routing in API Gateway
The API gateway uses a routing map to direct incoming requests to the appropriate internal microservices. This mechanism ensures that even if the internal implementation changes, clients can still use familiar external APIs.

:p How does the API gateway handle request routing?
??x
The API gateway maintains a mapping between public (external) paths and private (internal) service paths. When an incoming request is received, it checks this map to determine which microservice should process the request. This allows for flexibility in internal implementation changes without affecting external clients.

```java
// Example of routing logic using a simple map
public class RoutingMap {
    private Map<String, String> pathMapping = new HashMap<>();

    public void addPathMapping(String externalPath, String internalPath) {
        pathMapping.put(externalPath, internalPath);
    }

    public String getInternalPathForExternalRequest(String externalPath) {
        return pathMapping.getOrDefault(externalPath, null);
    }
}
```
x??

---

#### Composition in API Gateway
In distributed systems, data often resides across multiple services. The API gateway can provide a higher-level API that aggregates and composes responses from these different sources into a single response to the client.

:p How does composition work in an API gateway?
??x
Composition allows the API gateway to query multiple microservices and merge their responses into a cohesive output for the client. This abstraction simplifies the client's interaction with multiple services, reducing the number of requests it needs to make to fetch necessary data.

```java
// Example of composition logic in pseudocode
public class ComposedAPI {
    private List<BackendService> services;

    public ComposedAPI(List<BackendService> services) {
        this.services = services;
    }

    public Response getCombinedData(Request request) {
        List<Response> responses = new ArrayList<>();
        for (BackendService service : services) {
            responses.add(service.handleRequest(request));
        }
        // Logic to combine the individual responses into a single response
        return combineResponses(responses);
    }

    private Response combineResponses(List<Response> responses) {
        // Logic to merge multiple responses into one
        // For simplicity, assume merging content of all responses into a single string
        StringBuilder combinedContent = new StringBuilder();
        for (Response r : responses) {
            combinedContent.append(r.getContent());
        }
        return new Response(combinedContent.toString());
    }
}
```
x??

---

#### Translation in API Gateway
API gateways can translate data between different formats or protocols, ensuring seamless communication across microservices with varying APIs.

:p What is the purpose of translation in an API gateway?
??x
Translation allows the API gateway to adapt and convert request and response formats as needed. This could involve converting between JSON and XML, handling different versions of an API, or even adapting the protocol used (e.g., HTTP vs. WebSocket).

```java
// Example of translation logic in pseudocode
public class Translator {
    public Request translateRequest(Request rawRequest) {
        // Logic to convert raw request into a standardized format
        return new StandardizedRequest(rawRequest);
    }

    public Response translateResponse(Response rawResponse) {
        // Logic to adapt the response according to client needs
        return new AdapatedResponse(rawResponse);
    }
}
```
x??

---

