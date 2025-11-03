# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 47)

**Starting Chapter:** How Much Is Too Much

---

#### Conway’s Law in Reverse
Conway’s law states that an organization's communication structure will influence its software architecture. The provided text suggests that a system design can also shape organizational structures, especially when the business context changes.

Background: The company was initially focused on print operations with a small online presence. Over time, the digital side of the business grew significantly while the physical print business declined. Despite this shift in focus, the original three-part system (input, core, output) persisted, leading to an organizational structure that aligned with these parts.

:p Can you explain how Conway’s Law in reverse can impact a company's organization?
??x
Conway’s Law in reverse implies that as a system evolves and changes over time, so too will the organizational structure that supports it. In the example provided, the original design of the three-part system (input, core, output) influenced the creation of an IT organizational structure with corresponding channels or divisions for each part. This structure persisted even as the company’s primary business operations shifted from print to digital. As a result, changing this organizational structure to better align with current business needs required rethinking both the system and its governance.

x??

---

#### Microservices and Developer Awareness
In a microservice environment, developers must consider more operational concerns such as network calls and service failures compared to monolithic systems where these issues are often abstracted away. This shift can be challenging for developers accustomed to simpler coding environments.

Background: Moving from a monolithic system to microservices requires developers to think about the broader implications of their code, particularly in terms of network boundaries and potential failure points. This is due to the distributed nature of microservices where components are loosely coupled and communicate via well-defined APIs or protocols.

:p How does moving from a monolithic system to microservices impact developer responsibilities?
??x
Moving from a monolithic system to microservices significantly expands the scope of developers' responsibilities. In a monolithic architecture, developers primarily focus on coding within their modules without needing deep understanding of other parts of the application and its deployment environments. However, in a microservice environment, developers must consider network calls across boundaries, handle failures robustly, and manage data consistency issues.

This transition necessitates that developers are more aware of operational concerns such as fault tolerance, security, and performance. They need to design services that can communicate effectively over the network while maintaining high availability and resilience.

x??

---

#### Organizational Structure in a Microservice Environment
Organizations using microservices might face challenges if their development teams have not been trained or accustomed to working within this environment. This shift can be particularly difficult for developers who were previously used to simpler coding paradigms in monolithic systems.

Background: The transition from monolithic architecture to microservices requires developers to adapt to new ways of thinking about software design, deployment, and maintenance. Developers must understand the implications of network calls and service failures more deeply than they would in a monolithic setup where such concerns are often abstracted away by the underlying infrastructure.

:p What challenges might arise when transitioning from monolithic systems to microservices?
??x
Transitioning from monolithic systems to microservices can present several challenges, primarily due to the shift in how developers think about their work. Developers accustomed to monolithic architectures may struggle with the increased complexity and operational concerns associated with microservices.

Key challenges include:
- **Understanding Network Boundaries:** Developers must write code that interacts effectively across network boundaries.
- **Handling Failures:** Robust error handling is critical, as individual services might fail independently of others.
- **Data Management:** Managing data consistency in a distributed system can be complex and requires careful design.

These changes require developers to develop new skills and potentially undergo training or reorientation to adapt to the microservices environment.

x??

---

#### Pushing Power into Development Teams
Background context: The passage discusses the challenges of granting more autonomy to development teams, especially when transitioning from a traditional to a microservices architecture. It highlights that developers accustomed to blaming others may struggle with full accountability.

:p What are some challenges associated with pushing power and increased autonomy into development teams?
??x
The key challenges include:
1. Developers who have traditionally thrown work "over the wall" might feel uncomfortable being fully accountable for their work.
2. There can be contractual barriers preventing developers from carrying support pagers for systems they support.
3. It's essential to consider the current staff's appetite for change and avoid overwhelming them too quickly.

Understanding these challenges is crucial before making significant changes, as it helps in crafting a more effective transition plan.

x??

---

#### Aligning Service Ownership with Colocated Teams
Background context: The text emphasizes aligning service ownership with colocated teams to ensure that the system design matches organizational structure. This alignment helps in reducing tension points and ensures smoother operations.

:p How does aligning service ownership with colocated teams help reduce tensions?
??x
Aligning service ownership with colocated teams can significantly reduce tensions by:
1. Ensuring that team members are responsible for services related to their bounded contexts.
2. Facilitating better communication and coordination among team members.
3. Aligning individual responsibilities with the overall organizational goals.

This approach helps in creating a more cohesive development environment where each team owns its specific set of services, leading to fewer conflicts and improved efficiency.

x??

---

#### Challenges at Scale
Background context: The passage highlights that microservices become increasingly complex as they grow from small examples to larger, more intricate systems. It discusses handling multiple service failures and managing hundreds of services, which introduces new challenges.

:p What are some challenges faced when microservice architectures grow beyond a few discrete services?
??x
Some key challenges include:
1. Managing the failure of multiple separate services.
2. Handling complexity as the number of services increases.
3. Ensuring effective communication and coordination across numerous services.
4. Dealing with the increased load on support teams.

These challenges require careful planning and robust strategies to maintain system reliability and performance.

x??

---

#### Failure Management in Microservices
Background context: The text mentions that understanding failure is crucial, as hardware failures (like hard disks) and software crashes can occur frequently in microservice architectures.

:p What are some common types of failures mentioned in the text?
??x
The common types of failures mentioned include:
1. Hardware failures such as hard disk failures.
2. Software crashes or bugs within services.

These failures highlight the need for robust failure management strategies, including redundancy, monitoring, and recovery mechanisms.

x??

---

#### Importance of People in Transition
Background context: The passage stresses that people are central to any organizational change, especially when transitioning to a microservices architecture. It emphasizes understanding staff's readiness for change and ensuring clear communication about responsibilities.

:p Why is it important to understand the current staff’s appetite for change before making significant changes?
??x
It is crucial to understand the staff’s appetite for change because:
1. Rapid or forced changes can lead to resistance and lower morale.
2. Tailoring the transition pace according to team readiness ensures smoother implementation.
3. Clear communication about responsibilities helps in building trust and commitment among team members.

By understanding these aspects, organizations can better manage transitions and enhance overall success rates.

x??

---

#### Embracing Failure in Distributed Systems
Background context: In distributed systems, network unreliability and hardware failures are common. Understanding that failure is inevitable at scale allows for better design practices. For example, using cheaper components can be more efficient if you plan for frequent replacement due to high likelihood of failure.
:p What does the text suggest about handling service failures?
??x
The text suggests embracing the possibility of service failures and planning for them. By expecting that services might fail, it becomes easier to handle planned outages and make different trade-offs in system design. For instance, instead of spending a lot on making a single node highly resilient, one can use cheaper components like bare motherboards with velcro-hard drives.
??x
This approach allows organizations to focus more on graceful recovery mechanisms rather than futile attempts to prevent failure altogether.

```java
public class ServiceFailureHandler {
    public void handleServiceFailure() {
        // Code for handling failures gracefully
        System.out.println("Handling service failure.");
        // Example: Gracefully stopping the service and restarting it from a backup.
        startBackupService();
    }

    private void startBackupService() {
        // Pseudocode for starting a backup service
        System.out.println("Starting backup service...");
    }
}
```
x??
---
#### Cross-Functional Requirements in Distributed Systems
Background context: Cross-functional requirements involve understanding durability, availability, throughput, and latency in distributed systems. The goal is to determine the appropriate level of resilience needed based on specific use cases.
:p What does the text imply about autoscaling systems for a reporting system that runs only twice a month?
??x
The text implies that an autoscaling system might be overkill for a reporting system that only needs to run twice a month, as being down for a day or two isn't a significant issue. Therefore, resources can be allocated more efficiently by understanding the specific requirements and not overengineering solutions.
??x
This approach allows organizations to allocate resources effectively based on real needs rather than hypothetical worst-case scenarios.

```java
public class ReportingSystem {
    public void scheduleReportingRun() {
        // Code for scheduling a reporting run that only occurs twice a month
        System.out.println("Scheduling monthly reports.");
        // Example: Running the report every 30 days without autoscaling.
        runReportOnScheduledDay();
    }

    private void runReportOnScheduledDay() {
        // Pseudocode for running the report on scheduled day
        System.out.println("Running scheduled reports...");
    }
}
```
x??
---

---
#### Blue/Green Deployments
Blue/green deployments are a strategy to minimize downtime during software updates. In this approach, two identical environments (blue and green) run concurrently, with one version handling live traffic while the other is prepared for deployment. After thorough testing, the traffic is switched from the blue environment to the green environment.

This method allows for rolling updates without interrupting service to end-users, as both versions of the application coexist during the transition period.
:p What are blue/green deployments used for?
??x
Blue/green deployments are a strategy to minimize downtime during software updates by maintaining two identical environments (blue and green) where one version handles live traffic while the other is prepared for deployment. This allows for testing new versions in parallel with the current ones, ensuring that any issues can be identified before fully switching over.
x??

---
#### Understanding User Needs
Understanding user needs involves recognizing how much failure a system can tolerate or how fast it should perform based on the users' requirements. These requirements are driven by the nature and criticality of the service being provided.

For example, an online ecommerce system might require frequent updates with minimal downtime to ensure smooth customer experiences, whereas a corporate intranet knowledge base may not need such rigorous measures.
:p How do user needs influence system design?
??x
User needs significantly impact system design by determining factors like tolerance for failure and required response times. For instance, an online ecommerce system might require high availability and quick response times to ensure smooth transactions, while a corporate intranet knowledge base may tolerate some downtime or slower performance since it serves internal users with less critical access requirements.
x??

---
#### Response Time/Latency
Response time and latency are crucial in measuring the speed at which operations within a system complete. It's important to establish targets for various percentiles of response times under different load conditions.

For example, you might set a target where 90% of responses should be completed within 2 seconds when handling 200 concurrent connections per second.
:p How do you measure and define acceptable response time?
??x
Acceptable response time is measured by setting targets for various percentiles of response times under different load conditions. For instance, a target might be defined as: "We expect the website to have a 90th-percentile response time of 2 seconds when handling 200 concurrent connections per second." This helps ensure that most users experience fast service while allowing some outliers.
x??

---
#### Availability
Availability refers to the percentage of time a system is operational and available for use. It's important to consider whether services are expected to be available 24/7 or if there can be acceptable periods of downtime.

Measuring availability often involves calculating uptime percentages, but it’s crucial to understand that these metrics might not directly reflect user experience.
:p How do you measure system availability?
??x
System availability is typically measured by the percentage of time a service is operational and available for use. This can be calculated as: `Availability = (Uptime / Total Time) * 100`. However, it's important to note that this metric might not directly reflect user experience since users expect consistent reliability rather than just uptime percentages.
x??

---
#### Durability of Data
Durability of data refers to how long and under what conditions data can be retained. This varies based on the specific requirements and nature of the data.

For instance, financial transaction records may need to be kept for many years, whereas user session logs might only need to be stored for a year or less.
:p How do you determine data durability?
??x
Data durability is determined by considering how long and under what conditions data should be retained. This varies based on specific requirements, such as keeping financial transaction records for multiple years while user session logs might only need to be stored for a year or less.
x??

---

#### Degrading Functionality in Microservices
Background context: In a microservice architecture, ensuring that functionality can degrade gracefully when individual services are down is crucial for building a resilient system. This involves understanding the impact of each service's outage and preparing alternative actions.

:p What does degrading functionality entail in a microservice architecture?
??x
Degrading functionality means designing your application to continue functioning with reduced but essential features even if one or more microservices fail. For example, if the shopping cart service is down, you might still show the product details page but hide the cart section.

```java
// Example pseudocode for degrading function in Java
public void displayProductDetails(Product product) {
    // Display product details normally
    displayDetails(product);
    
    // Check if the shopping cart service is available
    boolean isCartServiceAvailable = checkIfServiceIsUp("shopping-cart-service");
    
    if (!isCartServiceAvailable) {
        // If not, degrade by showing a placeholder instead of full cart functionality
        showPlaceholderForShoppingCart();
    }
}
```
x??

---

#### Architectural Safety Measures in Microservices
Background context: To prevent cascading failures and ensure that one microservice outage doesn't bring down the entire system, it's important to implement architectural safety measures. These include circuit breakers, timeouts, and bulkheads.

:p What are some common architectural safety measures used in a microservice architecture?
??x
Some common architectural safety measures include:

1. **Circuit Breaker**: A mechanism that allows you to stop making requests to a problematic service until it has recovered.
2. **Timeouts**: Setting appropriate timeouts for HTTP calls to prevent blocking the application when a downstream service is slow or unresponsive.
3. **Bulkheads (or Service Isolation)**: Using separate connection pools and limits to prevent one service from overloading others.

```java
// Example pseudocode for implementing circuit breaker in Java using Resilience4j
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;

public class ServiceClient {
    private final CircuitBreakerRegistry registry = CircuitBreakerRegistry.ofDefaults();
    private final CircuitBreaker circuitBreaker = registry.getCircuitBreaker("shopping-cart-service");

    public void processRequest() {
        try {
            if (circuitBreaker.isOpen()) {
                throw new RuntimeException("Circuit breaker is open, retry later");
            }
            
            // Perform the request
            performRequest();
        } catch (RuntimeException e) {
            circuitBreaker.executeWithTry(() -> performRequest());
        }
    }

    private void performRequest() {
        // Logic to make a downstream service call
    }
}
```
x??

---

#### Strangler Pattern in Microservices
Background context: The strangler pattern is used when you need to gradually replace an old system with a new one. It involves wrapping the old application's functionality while incrementally moving away from it.

:p What is the strangler pattern, and how does it work?
??x
The strangler pattern is a technique where a new application wraps around the legacy system, providing a layer of abstraction that gradually replaces the old functionalities over time. This allows for a smooth transition without requiring an immediate full-scale replacement.

```java
// Example pseudocode for wrapping functionality in Java
public class ClassifiedAdsSystem {
    private LegacyAdSystem legacyAdSystem;
    private NewAdSystem newAdSystem;

    public void displayAd(String adId) {
        try {
            // First, attempt to get the ad from the new system
            AdDetails adDetails = newAdSystem.getAd(adId);
            
            if (adDetails != null) {
                return adDetails;
            }
            
            // If the new system fails, fallback to the legacy system
            adDetails = legacyAdSystem.getAd(adId);
        } catch (Exception e) {
            log.error("Failed to get ad from both systems", e);
        }

        return adDetails; // Return whatever was found
    }
}
```
x??

---

#### Example of a Cascading Failure in Microservices
Background context: The text provides an example where a downstream service's slow response led to a cascading failure, causing the entire system to go down.

:p What caused the cascading failure described in the text?
??x
The cascading failure was caused by a slow-down in one of the older legacy ad systems that was still serving a significant portion of the traffic. The slow service exhausted the HTTP connection pool, leading to an accumulation of blocked threads and ultimately bringing down the system.

```java
// Pseudocode illustrating the issue with the HTTP connection pool
public class HttpConnectionPool {
    private BlockingQueue<Worker> workers;

    public void makeRequest(HttpRequest request) throws InterruptedException {
        Worker worker = workers.take(); // Wait until a worker is available

        try {
            // Make an HTTP request using the worker
            HttpResponse response = makeHttpRequest(request);
            
            if (response.isTimeout()) {
                // If the request timed out, put the worker back and wait for another one
                putWorkerBack(worker);
            }
        } catch (IOException e) {
            // Handle exceptions by marking the worker as failed and re-adding to the queue
            markAsFailedAndReaddToQueue(worker);
        }
    }

    private void makeHttpRequest(HttpRequest request) throws IOException, InterruptedException {
        // Simulate a slow request
        Thread.sleep(1000); // Sleep for 1 second to simulate slowness
    }
}
```
x??

---

#### Antifragile Organization Concept
Background context: Nassim Taleb introduced the concept of antifragility, which refers to systems that benefit from failure and disorder. Ariel Tseitlin applied this idea to organizations like Netflix, highlighting how they embrace and even incite failure through various mechanisms.
:p What does the concept of an antifragile organization involve?
??x
The concept involves creating organizational structures that not only tolerate but also thrive on failure and uncertainty. Organizations should be designed in such a way that they can benefit from disorder, learning from failures to become more robust over time.
??x

---

#### Game Days for Simulating Failures
Background context: Google regularly performs 'game days,' where it simulates server failures and has teams respond to these events. This helps ensure preparedness and resilience in their systems.
:p What is a game day, and why does Netflix adopt similar practices?
??x
A game day is an exercise where organizations simulate system failures and have various teams react to these scenarios. By doing so, they can test the resilience of their systems and improve response times. Netflix also uses similar methods, such as running the Chaos Monkey in production daily.
??x

---

#### The Chaos Monkey Tool
Background context: Netflix has developed a tool called the Chaos Monkey that runs in production environments daily to simulate machine failures. This ensures developers are prepared for unexpected issues.
:p What is the Chaos Monkey, and how does it work?
??x
The Chaos Monkey is a program developed by Netflix that randomly turns off machines during certain hours of the day in production environments. Its purpose is to test system resilience against unexpected failures.
```java
public class ChaosMonkey {
    public void run() {
        while (true) {
            Random random = new Random();
            int machineToTurnOff = random.nextInt(totalNumberOfMachines);
            turnOffMachine(machineToTurnOff);
        }
    }

    private void turnOffMachine(int machineNumber) {
        // Code to safely shut down the specified machine
    }
}
```
??x

---

#### Simian Army of Failure Bots
Background context: Netflix’s Simian Army includes various failure bots, such as Chaos Gorilla and Latency Monkey, designed to simulate different types of system failures.
:p What is the Simian Army in the context of Netflix?
??x
The Simian Army refers to a suite of tools developed by Netflix that simulate different types of failures. These include Chaos Gorilla (which takes out an entire availability center) and Latency Monkey (which simulates slow network connectivity between machines).
```java
public class SimianArmy {
    public void runChaosGorilla() {
        // Code to simulate failure in an entire AWS availability center
    }

    public void runLatencyMonkey() {
        // Code to simulate slow network connectivity
    }
}
```
??x

---

#### Embracing and Inciting Failure Through Software
Background context: By actively causing failures, Netflix ensures that their systems are robust enough to handle real-world issues. This approach is crucial for organizations dealing with highly distributed systems.
:p Why does Netflix embrace failure through software?
??x
Netflix embraces failure through software because it prepares the system for real-world issues by ensuring developers are constantly ready and able to handle unexpected failures. This proactive approach improves overall system resilience and reliability.
??x

---

#### Blameless Culture in Organizations
Background context: Netflix promotes a blameless culture where mistakes are seen as learning opportunities rather than punishment grounds. This fosters an environment where employees feel safe to report issues and improve the system.
:p How does Netflix promote a blameless culture?
??x
Netflix promotes a blameless culture by treating failures as learning opportunities and not assigning blame when things go wrong. This approach encourages team members to report issues openly, which in turn helps the organization continuously evolve and improve.
??x

---

#### Importance of Handling Failures in Distributed Systems
Background context: As systems become more distributed, they inherently face greater risks of failure. However, embracing these failures through robust testing can lead to more resilient systems that better support customer needs.
:p Why is it important for organizations to prepare for system failures?
??x
It is crucial for organizations to prepare for system failures because modern systems are increasingly distributed and prone to unpredictable issues. By simulating and handling these failures proactively, organizations can build more robust and reliable systems that can withstand real-world challenges.
??x

