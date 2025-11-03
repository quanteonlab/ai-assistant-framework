# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 48)

**Starting Chapter:** Circuit Breakers

---

#### Timeouts
Timeouts are crucial for managing downstream system failures. They determine how long a system should wait before considering a call to be failed and acting on it. If set too low, you risk losing calls that might have succeeded; if too high, your system can become slow or unresponsive.

:p What is the importance of setting timeouts correctly in a distributed system?
??x
Timeouts are essential because they balance between ensuring timely responses from downstream systems and avoiding unnecessary delays. Incorrect timeout settings can lead to either failing too early (resulting in lost requests) or failing too late (leading to prolonged processing times).

For example, consider a scenario where you have an HTTP request that may take 5-10 seconds under normal conditions but could hang indefinitely if the backend service is down.

```java
public class TimeoutExample {
    public static void makeRequest(String url, int timeout) throws Exception {
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpGet httpGet = new HttpGet(url);
            CloseableHttpResponse response = httpClient.execute(httpGet, new BasicHttpContext(), timeout);
            // Process the response
        } catch (IOException e) {
            throw new Exception("Request timed out or failed", e);
        }
    }
}
```
x??

---

#### Circuit Breakers
Circuit breakers act as a safeguard for downstream services that might be experiencing issues. They monitor service health and, upon detecting a fault threshold, stop sending requests to the problematic service, thereby preventing the main system from being overwhelmed by failed calls.

:p What is a circuit breaker in software architecture?
??x
A circuit breaker acts like an electrical one but for your application's services. When a downstream service fails repeatedly, instead of continuously retrying (which can overwhelm both the upstream and downstream systems), the circuit breaker trips, failing fast and sending errors back to the caller.

Here’s a simplified example in Java:

```java
public class CircuitBreakerExample {
    private final int maxFailures;
    private volatile boolean isBroken = false;
    private AtomicInteger failureCount = new AtomicInteger(0);

    public CircuitBreakerExample(int maxFailures) {
        this.maxFailures = maxFailures;
    }

    public void recordFailure() {
        if (isBroken()) {
            incrementFailureCount();
        }
    }

    public boolean isBroken() {
        return isBroken && failureCount.get() >= maxFailures;
    }

    private void incrementFailureCount() {
        if (!isBroken) {
            failureCount.incrementAndGet();
        } else if (failureCount.incrementAndGet() > 2 * maxFailures) {
            // Reset after a period of stability
            resetCircuitBreaker();
        }
    }

    private void resetCircuitBreaker() {
        isBroken = false;
        failureCount.set(0);
    }
}
```
x??

---

#### Handling Failures During Maintenance
During maintenance, circuit breakers can be manually blown to safely isolate a microservice or component. This allows for the safe shutdown and restart of services without affecting other parts of the system.

:p How can you use circuit breakers during routine maintenance?
??x
Circuit breakers are useful tools that help manage service health. During maintenance, they can be manually triggered to fail fast, ensuring that downstream services do not receive requests from the maintenance target, thus preventing unexpected failures in other components.

For example:

```java
public class MaintenanceMode {
    private final CircuitBreaker circuitBreaker;

    public MaintenanceMode(CircuitBreaker breaker) {
        this.circuitBreaker = breaker;
    }

    public void triggerMaintenance() {
        circuitBreaker.recordFailure();
    }

    public boolean isMaintenanceModeActive() {
        return circuitBreaker.isBroken();
    }
}
```
x??

---

#### Queuing and Retrying Requests
During the blown state of a circuit breaker, you have options such as queuing requests. This can be beneficial if retries are part of the business logic or if it’s an asynchronous job.

:p What is one way to handle requests during the blown state of a circuit breaker?
??x
One approach is to queue up failed requests and retry them later when the circuit breaker returns to the closed state. This is particularly useful for non-critical operations that can be retried without impacting the primary workflow.

For example, if you have an asynchronous job processing emails:

```java
public class RequestQueue {
    private final BlockingQueue<Runnable> queue = new LinkedBlockingQueue<>();

    public void submitRequest(Runnable task) throws InterruptedException {
        queue.put(task);
    }

    public boolean isQueueEmpty() {
        return queue.isEmpty();
    }
}

public class RetryService {
    private final RequestQueue queue;
    private final CircuitBreaker breaker;

    public RetryService(RequestQueue queue, CircuitBreaker breaker) {
        this.queue = queue;
        this.breaker = breaker;
    }

    public void processRequest(Runnable task) throws InterruptedException {
        if (breaker.isBroken()) {
            // Queue the request
            queue.submitRequest(task);
        } else {
            // Process normally
            task.run();
        }
    }
}
```
x??

#### Bulkheads in Software Architecture
Background context explaining the concept. The bulkhead pattern is introduced by Nygard as a way to isolate parts of a system from failure, similar to how bulkheads on ships protect sections from damage if one part is compromised.
:p What are bulkheads and how do they work in software architecture?
??x
Bulkheads in software architecture help prevent the failure of one part of the system from causing widespread issues. They can be implemented through various means such as using separate connection pools, microservices, or circuit breakers. For instance, using different connection pools for each downstream service ensures that if one pool gets exhausted, others remain unaffected.
```java
// Example of separating connection pools in Java
DataSource pool1 = new DataSource(); // Connection pool 1
DataSource pool2 = new DataSource(); // Connection pool 2

// Code to use separate pools for different services
if (serviceANeedsConnection()) {
    Connection conn = pool1.getConnection();
} else if (serviceBNeedsConnection()) {
    Connection conn = pool2.getConnection();
}
```
x??

---

#### Microservices and Separation of Concerns
Background context explaining the concept. By separating functionality into microservices, we can reduce the risk of one part of the system failing to impact others.
:p How does separation of concerns through microservices help in implementing bulkheads?
??x
Separating functionalities into separate microservices helps isolate failures within specific services, preventing them from cascading and affecting other parts of the system. This approach ensures that if a particular service fails or behaves slowly, it won't disrupt the entire application.
```java
// Example of separating services into different processes
public class UserService {
    public User getUserById(int id) { ... }
}

public class OrderService {
    public Order getOrderById(int id) { ... }
}
```
x??

---

#### Circuit Breakers for Bulkheads
Background context explaining the concept. Circuit breakers act as automatic mechanisms to isolate failing services, protecting both the consumer and the downstream service from further damage.
:p What is a circuit breaker and how does it function in implementing bulkheads?
??x
A circuit breaker acts as an automated mechanism that isolates a failing part of the system, preventing further calls to that service. It can also protect the downstream service by limiting the number of requests to avoid overwhelming it. When a service fails or behaves slowly, the circuit breaker trips and stops sending requests to it until it recovers.
```java
// Example of using Hystrix for implementing a circuit breaker in Java
HystrixCommand.Setter withGroupKey("serviceA")
    .andCommandKey("getUserById")
    .build()
.execute();
```
x??

---

#### Load Shedding and Bulkheads
Background context explaining the concept. Load shedding is a technique where bulkheads reject requests under certain conditions to prevent resource saturation.
:p What is load shedding, and how does it relate to bulkheads?
??x
Load shedding in bulkheads involves rejecting requests when resources are becoming saturated. This helps ensure that critical systems don't become overwhelmed and act as bottlenecks for other services. When a service is nearing its capacity, it can reject additional requests to prevent resource exhaustion.
```java
// Example of load shedding using Hystrix in Java
HystrixCommand.Setter withGroupKey("serviceA")
    .andCommandPropertiesDefaults(HystrixCommandProperties.Setter()
        .withCircuitBreakerRequestVolumeThreshold(10)
        .withCircuitBreakerErrorThresholdPercentage(50))
    .build();
```
x??

---

#### Service Isolation
Background context: The more one service depends on another being up, the more the health of one impacts the ability of the other to do its job. If we can use integration techniques that allow a downstream server to be offline, upstream services are less likely to be affected by outages, planned or unplanned.
This is particularly important for maintaining system stability and ensuring that failures in one part of the service do not cascade into others.

:p What are the benefits of increasing isolation between services?
??x
Increasing isolation between services reduces the need for coordination between teams, allowing them more autonomy to operate and evolve their services. It also minimizes the impact of outages in downstream services on upstream services.
---
#### Idempotency
Background context: In idempotent operations, the outcome doesn’t change after the first application, even if the operation is subsequently applied multiple times. This characteristic is very useful when recovering from errors by replaying messages that haven't been processed.

:p What does it mean for an operation to be idempotent?
??x
An idempotent operation ensures that applying the operation multiple times has the same effect as applying it once. For example, in Example 11-2, adding points to a customer's account is idempotent because adding the same amount of points again does not change the total.

:p How can we make a non-idempotent operation like adding points more idempotent?
??x
By providing additional information that uniquely identifies the operation. In Example 11-2, the credit call becomes idempotent by including a `reason` element with an order ID.
---
#### HTTP Idempotency and Service Design
Background context: Some HTTP verbs like GET and PUT are defined to be idempotent in the HTTP specification. However, for these operations to remain idempotent, the service must handle them in such a way that subsequent identical requests have no additional effect.

:p Why is it important for a service handling HTTP GET or PUT requests to ensure they are idempotent?
??x
Ensuring that services handle GET and PUT requests idempotently prevents unexpected side effects when these requests are called multiple times. For example, if a PUT request updates a resource, repeated calls should not result in duplicate updates.

:p How can a service implement an idempotent HTTP PUT method for updating customer information?
??x
To make the `PUT /customers/{id}` endpoint idempotent, you could hash the request data and use it as a unique identifier. Only update the database if the hashed data matches the existing record.
```java
public void updateCustomer(@PathVariable("id") String id, @RequestBody CustomerRequest customer) {
    String existingDataHash = customerRepository.getDataHashForId(id);
    String currentHash = hashRequest(customer);
    if (!existingDataHash.equals(currentHash)) {
        // Update the database with new data
        customerRepository.save(id, customer);
    }
}
```
---
#### Event Processing and Idempotency
Background context: In event-driven architectures, processing events in an idempotent manner ensures that even if messages are delivered multiple times due to delivery failures or retries, the system's state remains consistent.

:p How can you ensure that your service processes events idempotently?
??x
Ensure that each event is processed only once by maintaining a record of which events have been processed. If an event is received more than once, it should be ignored or handled appropriately to avoid duplicate processing.
```java
public void handleEvent(Event event) {
    if (eventProcessor.hasProcessed(event)) {
        // Ignore the event as it has already been processed
        return;
    }
    // Process the event and mark it as handled
    process(event);
    eventProcessor.markAsProcessed(event);
}
```
---
#### HTTP Verb Idempotency in Practice
Background context: Not all operations that are defined to be idempotent in HTTP (like GET) should always be treated as such. For example, a service might record the fact that a request was received even if it doesn't perform any state-changing operation.

:p Can you use an HTTP GET method for logging purposes without causing side effects?
??x
Yes, you can use an HTTP GET method to log the receipt of a request and collect metrics like response time. The key is to ensure that such calls are idempotent in terms of the underlying business logic, not the entire system state.
```java
@GetMapping("/log")
public ResponseEntity<?> logRequest() {
    long startTime = System.currentTimeMillis();
    // Log the request details
    logger.info("Received GET request at: {}", new Date(startTime));
    
    // Record response time for monitoring purposes
    long endTime = System.currentTimeMillis();
    double elapsedTime = (endTime - startTime) / 1000.0;
    metricsService.logResponseTime(elapsedTime);
    
    return ResponseEntity.ok().build(); // Non-idempotent in terms of system state, but idempotent in business logic
}
```
---
#### Service Autonomy and Isolation
Background context: Increased isolation between services allows teams to operate more autonomously. This is beneficial because it reduces the coordination needed between service owners, enabling them to evolve their services more freely.

:p How does increasing isolation between services impact team autonomy?
??x
Increasing isolation between services allows teams to operate with greater independence. With less need for coordinated changes and fewer dependencies on other services, teams can make decisions and implement changes without waiting for consensus or synchronization with other teams.
---

