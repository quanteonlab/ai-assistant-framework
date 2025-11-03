# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 22)

**Starting Chapter:** How Much Is Too Much

---

#### Conway’s Law in Reverse
Background context explaining how system design can influence organizational structure, and the anecdotal evidence provided by a client example. The original system was designed for a modest website with content sourced from third parties and later evolved as the business operations shifted towards its online presence.

:p How did the organization's structure evolve in response to changes in the company's primary business focus?
??x
In this case, as the print side of the business diminished and the digital presence grew, the original system design inadvertently laid the path for how the organization grew. The IT department was structured into three channels or divisions aligned with the input, core, and output parts of the business, each having separate delivery teams.

The organizational structure didn't predate the system but grew around it as the company's focus shifted towards its digital presence.
x??

---

#### Microservices and Developer Mindset
Background context explaining how microservices can change a developer’s mindset from thinking about code in isolation to considering network boundaries and operational concerns. It also discusses the challenges of moving developers from monolithic systems to microservices.

:p How might moving from a monolithic system to microservices affect a developer's role?
??x
Moving from a monolithic system, where developers may have used just one language and been oblivious to operational concerns, to a microservice environment requires a shift in mindset. Developers must now consider network boundaries, the implications of failure, and operational concerns more carefully.

For example, if a developer writes a function in a microservice that calls another service across a network boundary, they need to think about how failures or delays might impact their system.
x??

---

#### Organizational Structure Adaptation
Background context explaining the difficulty organizations face when trying to adapt their structure to fit new technologies or business models. The example provided discusses the challenges of aligning an organization’s structure with its evolving technology stack.

:p What is a common challenge in adapting organizational structures to align with technological advancements?
??x
A common challenge is that organizational structures often don’t change as quickly as technological and business requirements evolve. For instance, in the described scenario, even though the company shifted from a print-based to an online-focused operation, the existing organizational structure based on legacy systems did not naturally adapt to this new focus.

To overcome such challenges, significant changes in both technology and organization might be required simultaneously.
x??

---

#### People Problem
Background context citing Gerry Weinberg’s quote that it's always a people problem. The text discusses how microservices can create a more complex operational environment for developers.

:p How does the transition to microservices impact organizational culture according to the provided text?
??x
The transition to microservices is not just about technology but also involves significant cultural shifts within an organization. Developers must now be aware of network boundaries, failure modes, and other operational concerns that were previously irrelevant in monolithic systems.

This shift can be challenging for developers who are used to working in a simpler environment where they could focus solely on coding without worrying about such complexities.
x??

---

#### Empowering Development Teams
Background context: The text emphasizes the importance of empowering development teams by increasing their autonomy. This approach can introduce challenges, particularly for teams accustomed to blaming others and potentially facing contractual barriers that prevent them from carrying support pagers.

:p How does increased autonomy impact development teams?
??x
Empowering development teams with more autonomy can increase productivity and innovation but may also lead to accountability issues if team members are not used to full responsibility. Additionally, there might be contractual barriers that hinder the ability of developers to carry support pagers for the systems they maintain.

```java
// Example of a developer taking responsibility in a microservices environment
public class DeveloperResponsibility {
    public void takeOwnership(String system) {
        System.out.println("Developer is now fully accountable for " + system);
    }
}
```
x??

---

#### Understanding Staff Appitude to Change
Background context: The text stresses the importance of understanding how your staff feels about changes in the organization, especially when transitioning to microservices. It suggests that pushing change too fast can lead to resistance and failure.

:p How should you approach changing your current staff's practices?
??x
You should understand your staff’s appetite for change and avoid pushing them too quickly. Consider providing a transitional period where a separate team handles frontline support or deployment, allowing developers time to adjust to new practices. Recognize that some changes may require different types of people within the organization.

```java
// Example of phased transition
public class TransitionPlan {
    public void introduceMicroservices() {
        System.out.println("Introducing microservices with a phased approach.");
        System.out.println("Separate support team for initial period.");
    }
}
```
x??

---

#### Aligning Service Ownership to Colocated Teams
Background context: The text discusses aligning service ownership to colocated teams and bounded contexts within the organization. This alignment is crucial for avoiding tension points in microservice architectures.

:p Why is it important to align service ownership with colocated teams?
??x
Aligning service ownership with colocated teams ensures that teams are responsible for their specific parts of the system, which can improve communication and reduce complexity. Bounded contexts help define clear boundaries around services, ensuring that teams understand their responsibilities and avoid overlap.

```java
// Example of defining bounded context in a microservices architecture
public class BoundedContext {
    public void defineContext(String serviceName) {
        System.out.println("Defining bounded context for " + serviceName);
    }
}
```
x??

---

#### Challenges at Scale
Background context: The text highlights the challenges that arise as microservice architectures grow beyond simple examples. It mentions handling multiple service failures and managing a large number of services.

:p What are some challenges when dealing with complex microservice architectures?
??x
Challenges include handling the failure of multiple separate services, managing hundreds of services, and coping with the complexity introduced by more microservices than people can handle effectively. These issues require robust monitoring, testing, and management strategies to ensure system reliability and maintainability.

```java
// Example of error handling in a complex microservice architecture
public class ErrorHandling {
    public void handleFailures() {
        try {
            // Simulate service call
            String result = "Service Result";
            if (result == null) {
                throw new Exception("Service failed");
            }
        } catch (Exception e) {
            System.out.println("Handling failure: " + e.getMessage());
        }
    }
}
```
x??

---

#### Conway's Law and Service Design
Background context: The text introduces Conway’s law, which suggests that the structure of an organization influences the design of its systems. It emphasizes aligning service ownership with organizational boundaries to avoid tension points.

:p How does Conway’s law impact microservice architecture?
??x
Conway’s law states that the structure of a system mirrors the communication structure within an organization. Therefore, designing services should reflect the natural boundaries and team structures of the organization to ensure effective collaboration and reduce conflicts.

```java
// Example of applying Conway's Law in service design
public class ServiceDesign {
    public void alignWithOrganization() {
        System.out.println("Aligning microservices with organizational structure.");
    }
}
```
x??

---

#### Embracing Failure at Scale
Background context: In distributed systems, failure is an inevitable aspect that needs to be planned for and managed. Understanding this principle helps organizations make better trade-offs and design more resilient systems.

:p What are some reasons why embracing failure is important in large-scale systems?
??x
Embracing failure is crucial because it allows us to focus on making our system robust against failures rather than trying to prevent them entirely. By acknowledging that hardware can fail, we can adopt strategies like graceful degradation and planned outages, which are easier to handle than unexpected disruptions.

For example, consider a service that needs to be upgraded. Planning for the failure of an individual node means you can perform in-place upgrades more easily by simply bringing down one instance at a time without affecting the entire system's availability.

```java
// Pseudocode for graceful upgrade
public void upgradeService() {
    // Mark some nodes for upgrade
    List<Node> nodesToUpgrade = getNodesMarkedForUpgrade();

    for (Node node : nodesToUpgrade) {
        shutdownAndReplace(node);
    }
}
```
x??

---

#### Hard Drives and Reliability at Scale
Background context: As the number of hard drives increases in a system, the likelihood of failure also increases. This is due to statistical certainty rather than individual reliability.

:p Why do hard drives fail more often as the number of hard drives increases?
??x
Hard drives fail based on their overall usage and lifespan. Even if each drive has a low probability of failure, with a large number of drives, the combined probability of at least one failing becomes significant due to the law of large numbers.

For instance, if you have 100 drives with a 1% chance of failure per year, the probability that none will fail is (0.99)^100 ≈ 36.7%. Conversely, the probability that at least one drive fails is about 1 - 0.367 = 63.3%.

```java
// Pseudocode for calculating failure probabilities
public double calculateFailureProbability(int drives) {
    double individualFailureRate = 0.01; // Example: 1% per year
    return 1 - Math.pow(1 - individualFailureRate, drives);
}
```
x??

---

#### Google's Approach to Server Failures
Background context: Google uses a robust approach to handle server failures by planning for them at the hardware level. This involves using bare motherboards and velcro attachments for hard drives.

:p How does Google’s method of handling server failures demonstrate resilience?
??x
Google’s use of bare motherboards and velcro hard drive attachments shows a pragmatic approach to hardware reliability. By not screwing in hard drives, they can quickly replace them when needed, reducing downtime and maintenance complexity.

This method reduces the time and effort required for planned outages during upgrades or maintenance, making the process more efficient and less disruptive.

```java
// Pseudocode for Google's server rack setup
public void setupServerRack() {
    List<Motherboard> motherboards = getMotherboards();
    for (Motherboard motherboard : motherboards) {
        attachHardDriveByVelcro(motherboard);
    }
}

private void attachHardDriveByVelcro(Motherboard motherboard) {
    // Attach hard drive using velcro
}
```
x??

---

#### Trade-offs in Resilience and Cost
Background context: When building resilient systems, it's important to balance the cost of implementing high-reliability solutions against the potential benefits. Sometimes, simpler designs can be more effective if failure is accounted for.

:p How can assuming that hardware will fail help in making trade-off decisions?
??x
Assuming hardware will fail helps by allowing you to focus on designing systems that are resilient and can handle failures gracefully. This mindset can lead to cost savings because you don't need to invest heavily in highly redundant or fault-tolerant hardware.

For example, using cheaper components with simpler failure handling mechanisms (like velcro-attached hard drives) might be sufficient if the system is designed to tolerate individual node failures without significant impact on overall service availability.

```java
// Pseudocode for cost-effective design
public void designSystem(String systemType) {
    if (systemType.equals("reporting")) {
        useCheapComponents();
    } else {
        useExpensiveButReliableComponents();
    }
}

private void useCheapComponents() {
    // Implement using bare motherboards and velcro hard drives
}
```
x??

---

#### Cross-Functional Requirements in Distributed Systems
Background context: Understanding cross-functional requirements helps in designing systems that meet durability, availability, throughput, and latency goals. However, the level of redundancy and fault tolerance required can vary based on the specific application.

:p How does understanding cross-functional requirements help in system design?
??x
Understanding cross-functional requirements is essential for designing robust distributed systems. By knowing exactly what levels of data durability, service availability, throughput, and acceptable latency are needed, you can make informed decisions about trade-offs between cost, complexity, and reliability.

For example, an application that only runs twice a month might not require the same level of redundancy as one that needs to be available 24/7. Autoscaling systems can be effective but might be overkill for applications with low demand patterns.

```java
// Pseudocode for implementing cross-functional requirements
public void configureSystem(Configuration config) {
    if (config.getRequirementLevel() == "high") {
        setupHighAvailability();
    } else if (config.getRequirementLevel() == "medium") {
        setupMediumAvailability();
    } else {
        setupBasicAvailability();
    }
}

private void setupHighAvailability() {
    // Implement high availability strategies
}
```
x??

---

---
#### Blue/Green Deployments
Blue/green deployments are a strategy to minimize downtime when updating or maintaining software. In this approach, two identical environments (blue and green) are maintained, with one environment serving live traffic while the other is updated or maintained.

:p How does blue/green deployment work?
??x
In blue/green deployment, you maintain two versions of your application: a current version (blue) that serves live traffic and a new version (green). The traffic can be switched between these environments without affecting users. This allows for smooth updates and maintenance.
```
// Example pseudocode to switch traffic from blue to green
function switchTraffic(newVersion) {
    if (newVersion === 'green') {
        // Redirect all requests to the green environment
        // Start scaling up the green environment gradually
    } else {
        // Continue using the blue environment for traffic
    }
}
```
x??

---
#### User Requirements and Tolerances
Understanding user requirements is crucial for designing a system that meets their needs. You need to identify how much downtime or latency users can tolerate, which depends on the nature of the service (e.g., ecommerce vs. corporate intranet).

:p What factors influence how much failure or latency a system can handle?
??x
The tolerance for failures and latency varies depending on the type of service:
- Ecommerce systems might require minimal downtime to avoid lost sales.
- Corporate intranets may tolerate more downtime since they are not as critical.

To determine these requirements, you need to ask questions about user expectations and help users understand the trade-offs between different levels of service. For instance, if a 90th percentile response time of 2 seconds is required with 200 concurrent connections per second, this is a specific requirement that needs to be documented.
```
// Example code snippet to measure response times
public class PerformanceMonitor {
    public void checkResponseTime(int connectionsPerSecond) {
        // Logic to simulate and measure response time under load
        long responseTime = simulateLoad(connectionsPerSecond);
        if (responseTime > 2000) { // Assuming 2 seconds is the threshold
            System.out.println("Response time exceeded expected limits.");
        } else {
            System.out.println("Performance within acceptable range.");
        }
    }

    private long simulateLoad(int connectionsPerSecond) {
        // Simulate load and return response time in milliseconds
        return 1500; // Example value
    }
}
```
x??

---
#### Response Time/Latency
Response times are critical for user satisfaction, especially on services with high traffic. Measuring these at different levels of concurrency helps understand how the system performs under various loads.

:p How can you measure response time effectively?
??x
To measure response time effectively:
1. Use a representative set of operations to test.
2. Test with varying numbers of concurrent users or connections.
3. Set targets based on percentiles, such as 90th percentile response times.
4. Document the expected performance under different load conditions.

For example, you might define: "We expect the website to have a 90th percentile response time of 2 seconds when handling 200 concurrent connections per second."

```java
public class ResponseTimeMonitor {
    public void measureResponseTime(int users) {
        long[] responseTimes = simulateLoad(users);
        
        // Calculate and print the 90th percentile response time
        double[] sortedResponseTimes = Arrays.stream(responseTimes).sorted().toArray();
        int index = (int) Math.round(sortedResponseTimes.length * 0.9);
        System.out.println("90th Percentile Response Time: " + sortedResponseTimes[index] + " ms");
    }
    
    private long[] simulateLoad(int users) {
        // Simulate load and return an array of response times
        long[] responses = new long[users];
        for (int i = 0; i < users; i++) {
            responses[i] = (long) Math.random() * 2500 + 1000;
        }
        return responses;
    }
}
```
x??

---
#### Availability
Availability refers to the uptime of a service. It is important for critical services where downtime could have significant consequences.

:p How do you measure availability?
??x
Measuring availability involves determining how often a system is available to users and ensuring it meets expected reliability standards. While measuring periods of acceptable downtime can be useful from a historical reporting perspective, the primary focus should be on whether the service is accessible when needed.

For example:
- A 99.9% uptime guarantee means only 0.1% of time could be down.
- Historical metrics might show that your system has been available 99.85% of the time over a year.

```java
public class AvailabilityMonitor {
    public double calculateAvailability(long totalUptime, long totalDowntime) {
        return (totalUptime * 100.0 / (totalUptime + totalDowntime));
    }
    
    public void monitorAvailability() {
        // Simulate uptime and downtime
        long uptime = 8760; // Assume the service was up for a year
        long downtime = 24; // Example downtime in hours
        
        double availability = calculateAvailability(uptime, downtime);
        System.out.println("Current Availability: " + availability + "%");
    }
}
```
x??

---
#### Data Durability
Data durability involves how much data can be lost and for how long it needs to be retained. The retention period can vary based on the nature of the data (e.g., financial transactions vs. session logs).

:p How do you determine the durability requirements for different types of data?
??x
To determine data durability:
- Financial records often require long-term storage to comply with regulatory requirements.
- Session logs might only need short-term retention, such as a year or less.

For example, your system might specify:
- Keep financial transaction records for 10 years.
- Store session logs for one year for space efficiency.

```java
public class DataDurabilityPolicy {
    public void defineDataRetentionPolicies() {
        // Define policies based on data type
        Map<String, Integer> retentionPeriods = new HashMap<>();
        retentionPeriods.put("financial_transactions", 10); // Years
        retentionPeriods.put("session_logs", 1); // Year
        
        for (Map.Entry<String, Integer> entry : retentionPeriods.entrySet()) {
            System.out.println(entry.getKey() + " must be retained for: " + entry.getValue() + " years.");
        }
    }
}
```
x??

---

#### Degrading Functionality
Background context: The ability to safely degrade functionality is crucial when building a resilient system with microservices. Understanding which parts of the functionality are critical and knowing how to handle failures gracefully can significantly improve system resilience.

:p What is degrading functionality, and why is it important in a microservice architecture?
??x
Degrading functionality refers to designing your system so that if one or more microservices fail, you can still provide some level of service. This approach improves overall system resilience by ensuring that even if parts of the system are down, critical functionalities remain accessible. For example, if a shopping cart service is unavailable, you might still display product listings but hide or disable related UI elements.

In a monolithic application, system health is binary—either it's working or not. However, in microservice architectures, understanding and managing nuanced failure scenarios becomes essential.
x??

---

#### Architectural Safety Measures
Background context: To ensure that failures do not cause widespread issues, architectural safety measures such as bulkheads, timeouts, and circuit breakers are crucial. These measures help prevent a single point of failure from cascading into the entire system.

:p What is the purpose of implementing architectural safety measures in a microservice architecture?
??x
The purpose of implementing architectural safety measures is to protect the overall health of the system by preventing the impact of failures from spreading. This includes techniques like setting timeouts, using bulkheads (separate connection pools), and implementing circuit breakers to avoid sending requests to unhealthy services.

By standardizing these practices, you can ensure that your application remains stable even when parts of it fail.
x??

---

#### Strangler Application Example
Background context: The example describes a scenario where an online classified ads website was strangling older legacy applications by gradually replacing them with new microservices. This process can introduce vulnerabilities if not managed carefully.

:p What is the risk of failing to manage dependencies properly during the strangler application process?
??x
The risk of failing to manage dependencies properly during the strangler application process is that a single point of failure in an older, poorly maintained service can bring down the entire system. This is because modern services often depend on multiple legacy services, and if one of these dependencies fails, it can cause a cascade of failures throughout the system.

In the example provided, the slow response time of a downstream ad system, which handled only 5% of traffic, brought down the whole site due to its impact on request handling.
x??

---

#### Timeout Configuration
Background context: The example highlights the importance of properly configuring timeouts in HTTP connection pools. Misconfigured timeouts can lead to excessive resource consumption and potential cascading failures.

:p What is the issue with misconfigured timeouts in an HTTP connection pool?
??x
Misconfigured timeouts in an HTTP connection pool can cause a build-up of blocked threads when requests are slow or fail to complete within the timeout period. This can lead to excessive resource usage, such as connections and memory, which can exhaust system resources and cause the application to crash.

In the example, setting incorrect timeouts caused the system to peak at around 800 connections in just five minutes, bringing it down.
x??

---

#### Bulkheads
Background context: Bulkheads are a method of isolating parts of your application so that failures in one part do not affect others. This is particularly useful when dealing with external dependencies that may be unreliable.

:p What is the role of bulkheads in managing microservice architectures?
??x
The role of bulkheads in managing microservice architectures is to separate different connection pools or service boundaries, ensuring that a failure in one part of the system does not affect other parts. By isolating services, you can control resource consumption and prevent cascading failures.

For example, using separate HTTP connection pools for different downstream services helps limit the impact of a failing service on others.
x??

---

#### Circuit Breaker
Background context: A circuit breaker is a pattern used to handle faults by breaking the circuit when a certain threshold of failures is reached. This prevents further requests from being sent to unhealthy services, allowing them to recover.

:p How does a circuit breaker work in the context of microservice architectures?
??x
A circuit breaker works by monitoring the success rate of service calls. When a defined threshold of failures is reached, it trips and stops sending requests to that service. This allows the failing service time to recover without overwhelming the system with more failed requests.

In the example, implementing a circuit breaker would have prevented traffic from being sent to the unhealthy downstream service, allowing it to recover.
x??

---

#### Antifragile Organization Concept
Background context explaining that organizations like Netflix and Google have embraced a philosophy of building systems resilient to failure by regularly causing it. This is based on Nassim Taleb’s concept of antifragility, where systems benefit from disorder and failure.
:p What is the core idea behind an antifragile organization?
??x
The core idea is that organizations should be designed in such a way that they not only survive but thrive under conditions of uncertainty, stress, and disruption. This involves actively encouraging failure and building robust mechanisms to handle it. Organizations like Netflix and Google exemplify this by regularly causing system failures to ensure their systems are resilient.
x??

---
#### Game Days for Simulating Failures
Background context on how some organizations, including Google during my time there, simulate server failure through game days where teams practice reacting to such events. This is a form of preparing the organization for real-world scenarios.
:p What are game days in the context of simulating failures?
??x
Game days are exercises where organizations simulate system failures and have their teams react to them as if it were a real incident. The goal is to prepare the organization for potential real-world disruptions by providing practice under controlled conditions.
x??

---
#### DiRT (Disaster Recovery Test) Exercises
Background context on how Google conducts annual disaster recovery tests, including large-scale simulations like earthquakes, to test its resilience against significant disruptions.
:p What are DiRT exercises?
??x
DiRT stands for Disaster Recovery Tests. These are comprehensive exercises where Google simulates large-scale disasters such as earthquakes to test the robustness and readiness of its systems. The objective is to ensure that critical services can withstand major disruptions.
x??

---
#### Chaos Monkey: A Simian Army Tool
Background context on Netflix's use of tools like Chaos Monkey, which randomly turns off machines during certain hours, ensuring developers are prepared for failures in production environments.
:p What is the Chaos Monkey?
??x
Chaos Monkey is a tool developed by Netflix that periodically and unpredictably terminates random instances within its infrastructure. It runs daily to simulate failure scenarios in production, forcing developers to build resilient systems capable of handling unexpected outages.
x??

---
#### Latency Monkey: Simulating Network Delays
Background context on how the Latency Monkey simulates slow network connectivity between machines, adding another layer of complexity and unpredictability to test system resilience.
:p What is the Latency Monkey?
??x
Latency Monkey is a tool that simulates network delays by introducing artificial latency between different nodes in the system. It helps test whether applications can handle degraded network conditions without failing catastrophically.
x??

---
#### Open-Source Simian Army Tools
Background context on Netflix making its Simian Army tools, including Chaos Monkey and Latency Monkey, available under an open-source license for other organizations to use and improve upon.
:p Why did Netflix make the Simian Army tools open source?
??x
Netflix made the Simian Army tools, like Chaos Monkey and Latency Monkey, open source to encourage wider adoption and improvement. By sharing these tools, they hope that more organizations can benefit from building resilient systems through regular failure simulation exercises.
x??

---
#### Blameless Culture for Learning from Failures
Background context on Netflix's emphasis on a blameless culture where mistakes are seen as opportunities for learning rather than punishment. This approach empowers developers and fosters continuous improvement.
:p What is the blameless culture in the context of system failures?
??x
The blameless culture in the context of system failures refers to an organizational mindset that views errors and failures as valuable learning experiences. Instead of punishing individuals, this approach focuses on understanding what went wrong and improving systems to prevent similar issues in the future.
x??

---
#### Preparing for Distributed System Failures
Background context on how distributed systems, due to their nature, are inherently more vulnerable to failure compared to centralized ones. The importance of preparing for such failures is highlighted by companies like Netflix and Google.
:p Why do distributed systems need special preparation?
??x
Distributed systems are more vulnerable to failures because they rely on multiple independent nodes that can fail independently. Unlike centralized systems, where a single point of failure might be easier to manage, distributed systems require robust mechanisms to handle individual node failures gracefully. Preparing for these failures ensures better system reliability and customer satisfaction.
x??

---

