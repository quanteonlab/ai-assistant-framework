# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 43)

**Starting Chapter:** Consumer-Driven Tests to the Rescue

---

---
#### Flaky and Brittle Tests
Background context explaining the concept. As test scope increases, so too do the number of moving parts, which can introduce issues that do not relate to the functionality under test but rather to external factors like service availability or network conditions.

:p What are flaky tests in end-to-end testing?
??x
Flaky tests are those that sometimes fail and pass when re-run. They do not provide reliable information about the state of the system being tested because their failure is often due to external factors such as service outages or temporary network glitches rather than actual bugs in the code.

For example, consider a test verifying an order placement process involving multiple services:
```java
public class OrderPlacementTest {
    @Test
    public void shouldPlaceOrder() {
        // Code to place an order through multiple services
        // Test might fail due to one of the services being down or network issues.
    }
}
```
x??

---
#### Normalization of Deviance
Explanation of Diane Vaughan's concept and its relevance in software testing. It refers to a situation where, over time, deviant behavior (in this case, flaky tests) becomes accepted as normal within an organization.

:p What does normalization of deviance mean in the context of end-to-end testing?
??x
Normalization of deviance means that over time, organizations can become accustomed to accepting substandard or problematic behaviors (like consistently failing but unreliable tests) as being acceptable. This acceptance can lead to a loss of faith in the test suite and result in real issues going unnoticed.

For instance, if a flaky test fails frequently and everyone just re-runs it hoping it will pass next time, this could indicate normalization of deviance.
x??

---
#### Test Ownership for End-to-End Tests
Explanation that end-to-end tests covering specific services should ideally be written by the team responsible for those services.

:p Who is typically responsible for writing end-to-end tests?
??x
The team that owns a particular service should write and maintain its end-to-end tests. This ensures accountability and understanding of the system's behavior from both the developers and testers.

For example, if there are multiple teams involved in a project, and one team is responsible for the order placement service, then this team would be expected to write all related end-to-end tests.
x??

---

#### Test Ownership and Shared Codebase Approach

Background context: The passage discusses a common issue where end-to-end tests become a free-for-all, leading to an explosion of test cases without clear ownership. This can result in broken tests being ignored or becoming outdated due to lack of engagement from the original code developers.

:p How can teams ensure that the health and maintenance of their end-to-end test suite are managed effectively?
??x
To ensure effective management of the end-to-end test suite, it is essential to treat it as a shared codebase with joint ownership. Teams should be free to check in tests but must share responsibility for maintaining and improving the overall health of the suite. This approach helps align developers with their own testing responsibilities, reducing the likelihood of ignoring broken or obsolete tests.

```java
// Example of how teams can collaborate on test maintenance
public class TestCollaboration {
    public static void main(String[] args) {
        // Pseudocode for a shared responsibility mechanism
        List<String> teamMembers = Arrays.asList("TeamA", "TeamB", "TeamC");
        
        while (true) {
            int randomIndex = new Random().nextInt(teamMembers.size());
            String currentOwner = teamMembers.get(randomIndex);
            
            System.out.println("Current owner: " + currentOwner);
            // Code to run maintenance tasks and ensure tests are up-to-date
            // This could involve writing, reviewing, or fixing tests
            
            try {
                Thread.sleep(30 * 60 * 1000); // Sleep for half an hour before next round
            } catch (InterruptedException e) {
                System.out.println("Thread was interrupted, failed to complete operation");
            }
        }
    }
}
```
x??

---

#### Test Suite Performance and Flakiness

Background context: The text highlights the challenges of managing long-running test suites that can be flaky. This includes issues like slow execution times and frequent false positives or negatives.

:p How does the length and reliability of end-to-end tests affect team productivity?
??x
The length and reliability of end-to-end tests significantly impact team productivity. Long-running tests can consume large amounts of time, making them less effective for rapid feedback cycles. Flaky tests add to this problem by generating false positives or negatives, leading to unnecessary debugging sessions and decreased trust in the test suite.

```java
// Example of a simple timer class to measure test execution time
public class TestExecutionTimer {
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        
        // Simulate running a test that takes some time
        try {
            Thread.sleep(10 * 60 * 1000); // Sleep for 10 minutes
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return;
        }
        
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        System.out.println("Test took: " + executionTime / 60000 + " minutes");
    }
}
```
x??

---

#### Parallel Execution of Tests

Background context: The text suggests using parallel test execution tools, such as Selenium Grid, to mitigate the issues with slow and flaky tests. This approach can help distribute load across multiple machines or browsers.

:p How can parallel execution improve test suite performance?
??x
Parallel execution improves test suite performance by distributing the workload across multiple threads or machines. This reduces the overall time taken for the entire test suite to run, making it more efficient and faster.

```java
// Example of how to use a Selenium Grid to run tests in parallel
public class ParallelTestRunner {
    public static void main(String[] args) throws MalformedURLException {
        // Register with the remote hub
        DesiredCapabilities desiredCapabilities = new DesiredCapabilities();
        URL gridHubUrl = new URL("http://localhost:4444/wd/hub");
        
        WebDriver driver = new RemoteWebDriver(gridHubUrl, desiredCapabilities);
        
        try {
            // Simulate running a test in parallel on the remote node
            driver.get("https://example.com");
            System.out.println("Test passed on " + gridHubUrl);
            
        } finally {
            driver.quit();
        }
    }
}
```
x??

---

#### Avoiding Test Duplication and Redundancy

Background context: The text mentions the importance of curating end-to-end test suites to reduce overlap in test coverage. This is crucial for maintaining a manageable and efficient testing process.

:p How can teams ensure they are not duplicating tests or leaving gaps in their test coverage?
??x
To avoid duplicating tests or leaving gaps, teams should regularly review and refactor their test suite to ensure comprehensive but non-redundant coverage. This involves identifying overlapping tests that can be merged or refactored into more generalized versions.

```java
// Example of how to refactor redundant tests
public class TestRefactoring {
    public static void main(String[] args) {
        // Original tests for two similar functionalities
        System.out.println("Test 1: " + testFunctionalityA());
        System.out.println("Test 2: " + testFunctionalityB());
        
        // Refactored version to avoid redundancy
        System.out.println("Refactored Test: " + refactorTests());
    }
    
    private static boolean testFunctionalityA() {
        return true; // Simulate a passing test for functionality A
    }
    
    private static boolean testFunctionalityB() {
        return true; // Simulate a passing test for functionality B
    }
    
    private static boolean refactorTests() {
        return testFunctionalityA() && testFunctionalityB(); // Refactor into one generalized test
    }
}
```
x??

---

#### Risk and Test Curation
Background context: The text discusses the challenges of balancing test value versus burden, especially with large-scale test suites. It mentions that while removing tests can reduce running time, it increases risk if a bug is introduced.

:p How does the balance between test value and burden affect test management?
??x
The balance between test value and burden affects test management by requiring a careful assessment of risk. If a feature is covered in multiple tests, eliminating redundant ones can save significant time but may increase the risk of undetected bugs. Effective test curation requires a deep understanding of risks associated with each test.

```java
public class TestCurationExample {
    public boolean shouldRemoveTest(Test oldTest, Test newTest) {
        // Check if both tests cover the same feature
        boolean coversSameFeature = oldTest.coveredFeature().equals(newTest.coveredFeature());
        
        // If they do, determine if one is redundant based on factors like coverage and time taken
        return coversSameFeature && (oldTest.getTimeTaken() > newTest.getTimeTaken() || !newTest.isCritical());
    }
}
```
x??

---

#### Feedback Cycles in End-to-End Tests
Background context: The text highlights the issue of long feedback cycles with end-to-end tests, which can lead to a pile-up when integration breaks. This affects developer productivity and deployment timelines.

:p How do long feedback cycles impact deployments?
??x
Long feedback cycles impact deployments by increasing the time required to fix issues. When an end-to-end test suite is lengthy, fixing a break takes longer, reducing the frequency of successful tests. This can result in fewer services being deployable into production, leading to pile-ups where new changes accumulate while waiting for builds.

```java
public class FeedbackCycleExample {
    public void updateDeploymentStatus(int buildTime) {
        if (buildTime > 60 * 5) { // Build time of more than 5 minutes
            System.out.println("Pile-up warning: Long feedback cycle detected.");
        }
    }
}
```
x??

---

#### The Metaversion Concept
Background context: The text introduces the idea of metaversion, where multiple services are versioned together, leading to coupling and a return to monolithic practices. This concept challenges the core advantage of microservices.

:p What is the metaversion and why is it problematic?
??x
The metaversion refers to the practice of treating an entire system as one unit for deployment purposes by versioning all its services together. This approach can lead to coupling, where once-separate services become tightly integrated due to frequent simultaneous deployments. Over time, this can revert the benefits of microservices and result in a tangled mess that is harder to manage.

```java
public class MetaversionExample {
    public void deploySystemVersion(String systemVersion) {
        // Deploy all services with the same version
        for (Service service : services) {
            service.deploy(systemVersion);
        }
    }
}
```
x??

---

#### Coupling and Microservices
Background context: The text discusses how coupling can arise when services are deployed together due to frequent simultaneous changes, leading to a less flexible microservice architecture.

:p How does coupling affect the deployment of microservices?
??x
Coupling affects the deployment of microservices by causing services to become interdependent. When multiple services are deployed simultaneously and their versions are tied (metaversion), it reduces the independence that microservices aim to provide. This can make individual service deployments harder, as changes in one service may require redeployments in others, leading to a less modular architecture.

```java
public class MicroserviceCouplingExample {
    public void deployService(Service service) {
        // Check for coupling with other services before deploying
        if (isCoupled(service)) {
            System.out.println("Deployment blocked: Service is coupled.");
        } else {
            service.deploy();
        }
    }

    private boolean isCoupled(Service service) {
        // Logic to check if the service is coupled with others
        return true; // Simplified for example
    }
}
```
x??

---

---
#### Test Journeys Instead of Stories
In managing a system composed of many services, end-to-end tests can become unmanageably large and complex. This is particularly true when dealing with more than two or three services, leading to a "Cartesian-like explosion" of test scenarios. Over time, every new piece of functionality may result in a new end-to-end test, creating a bloated suite that provides poor feedback cycles due to overlaps.
:p Why are end-to-end tests problematic for systems with multiple services?
??x
End-to-end tests become overly complex and unmanageable as the number of services increases. They can lead to a large number of test scenarios (Cartesian explosion) and result in a bloated suite that offers poor feedback cycles due to overlaps in test coverage.
```java
// Example of a simple end-to-end test in Java using JUnit
@Test
public void testOrderCD() {
    // Setup
    OrderService orderService = new OrderService();
    
    // Action
    OrderResult result = orderService.placeOrder("Customer1", "CD123");
    
    // Assert
    assertEquals(OrderStatus.COMPLETED, result.getStatus());
}
```
x??

---
#### Consumer-Driven Tests (CDCs)
Consumer-driven tests are used to ensure that when a new service is deployed, it won't break its consumers. This method avoids the need for end-to-end testing by defining and enforcing consumer expectations on a service in code form as tests. These tests run as part of the CI build of the producer, ensuring no deployment if any contract is broken.
:p What are consumer-driven tests (CDCs) used for?
??x
Consumer-driven tests (CDCs) ensure that deploying new services does not break their consumers by defining and enforcing expectations in code form. They run during the CI process to prevent deployments that violate these contracts.
```java
// Example of a CDC test in Java using JUnit
@Test
public void testCustomerServiceForHelpdesk() {
    // Setup
    CustomerService customerService = mock(CustomerService.class);
    
    // Action & Assert
    when(customerService.getCustomerDetails("12345")).thenReturn(new Customer("John Doe", "Sales Rep"));
    assertEquals("John Doe", customerService.getCustomerDetails("12345").getName());
}
```
x??

---

#### Pact Overview
Pact is a consumer-driven testing tool that was originally developed for Ruby and has since expanded to include JVM and .NET ports. It allows consumers to define expectations about how services will behave, which are then verified by producers.

The process involves creating expectations on the consumer side using a domain-specific language (DSL), launching a local mock server, and running these expectations against it to generate a Pact specification file in JSON format.
:p What is Pact and how does it work?
??x
Pact works by allowing consumers to define what they expect from producers through contracts. These contracts are then verified by the producer's service. The key steps include:
1. Defining expectations using a Ruby DSL on the consumer side.
2. Running these expectations against a local mock server to create a Pact specification file (JSON).
3. Verifying these expectations on the producer side.

Example in pseudocode:
```pseudocode
// Consumer-side code to define expectations and run them against a mock service
define_expectations() {
    // Define interactions between consumer and producer using DSL
}

run_pact_specifications() {
    start_mock_server()
    run_expectations()
    generate_json_file()
}
```
x??

---

#### Pact Specification File
The Pact specification file is a formal JSON document that contains the expectations defined by the consumer. It serves as a contract between the producer and consumer, ensuring both services behave as expected.
:p What is the Pact specification file?
??x
The Pact specification file is a JSON-formatted document that encapsulates the interactions and responses expected from one service to another. This file acts as a formal agreement or contract, allowing isolated testing of the consumer against the producer's API.

Example in JSON format:
```json
{
    "consumer": {
        "name": "Payment Service"
    },
    "provider": {
        "name": "Account Service"
    },
    "interactions": [
        {
            "description": "Verify account balance",
            "request": {
                "method": "GET",
                "path": "/accounts/12345"
            },
            "response": {
                "status": 200,
                "body": {"balance": 500},
                "headers": {"Content-Type": "application/json"}
            }
        }
    ]
}
```
x??

---

#### Pact Broker
The Pact Broker is a tool that helps manage multiple versions of Pact specification files. It allows for storing and versioning these contracts, enabling tests to be run against different versions of the consumer or producer.
:p What is a Pact Broker?
??x
A Pact Broker is a service that stores multiple versions of Pact specification files. This enables teams to run consumer-driven contract tests against various versions of consumers or producers, facilitating better collaboration and version control in development environments.

Example setup with a Pact Broker:
```sh
# Adding the Pact CLI tool for interaction with the broker
pip install pact-jvm-provider-cucumber

# Running the provider app with Pact Broker
./target/pact-jvm-provider-cucumber-1.0.0.jar --broker-url http://localhost:8080
```
x??

---

#### Pacto vs Pact
Pacto is another Ruby tool for consumer-driven testing, which records interactions between client and server to generate expectations dynamically. In contrast, Pact regenerates expectations with every build.

Key differences include:
- **Dynamic vs Static Expectations**: Pacto generates static expectations based on existing interactions, whereas Pact regenerates expectations in the consumer codebase.
- **Usage Context**: Pact is more suitable for services still being developed or as part of a continuous integration pipeline.

Example comparison:
```pseudocode
// Using Pacto to record and generate expectations
generate_expectations() {
    start_server()
    record_interactions()
    generate_static_specifications()
}

// Using Pact to define and regenerate expectations dynamically
regenerate_expectations() {
    define_new_expectations()
    run_tests()
    update_pact_file()
}
```
x??

#### Agile Stories as Conversations
Agile stories are placeholders for conversations about what a service API should look like. CDCs (Continuous Dialogues and Contracts) act similarly, serving as codifications of discussions on evolving APIs. They trigger conversations when issues arise.
:p What is the purpose of using Continuous Dialogue and Contracts (CDCs) in agile development?
??x
CDCs are used to codify conversations about service APIs and their evolution. When these contracts break, they prompt new dialogues to refine or update the API design. This ensures that both consumer and producing services maintain good communication and trust.
x??

---

#### End-to-End Tests vs CDCs
End-to-end tests have significant disadvantages as systems grow more complex. Many organizations prefer CDCs (Continuous Dialogues and Contracts) over extensive end-to-end testing, especially in microservices environments. However, some still use limited end-to-end tests for semantic monitoring.
:p Should you rely on end-to-end tests or CDCs?
??x
End-to-end tests are useful during the learning phase of implementing microservices but can be phased out as you improve CDCs and production monitoring techniques. In some organizations with a low appetite for learning in production, they may continue to use extensive end-to-end testing, though this approach is generally seen as less efficient.
x??

---

#### Semantic Monitoring
Semantic monitoring uses existing journey tests to monitor the production system after deployment, instead of running new tests every time. This allows teams to focus on improving other aspects like CDCs and monitoring without compromising safety.
:p How does semantic monitoring work?
??x
Semantic monitoring leverages pre-existing end-to-end tests run in a production-like environment to detect issues before they impact users. These tests help ensure the system behaves as expected once deployed, acting more like continuous integration for the live environment.
x??

---

#### Smoke Tests and Blue/Green Deployments
Smoke tests are small sets of tests run against newly deployed software to confirm that basic functionality is working correctly in a new environment. Blue/green deployments involve running two versions of a service simultaneously, with traffic directed only to one version until it passes all tests.
:p What is the purpose of smoke tests?
??x
The purpose of smoke tests is to quickly verify that a newly deployed microservice functions correctly without full-scale production traffic. This helps catch environment-specific issues early in the deployment process.
```java
public class SmokeTestRunner {
    public void runSmokeTests() {
        // Test basic functionality like database connections, API calls, etc.
    }
}
```
x??

---

#### Blue/Green Deployments
Blue/green deployments involve running two versions of a service simultaneously. One version receives real production traffic while the other is tested in situ for issues before going live. This approach minimizes downtime and allows for easy rollbacks if needed.
:p How does blue/green deployment work?
??x
In blue/green deployments, you deploy both old (blue) and new (green) versions of a microservice side-by-side. Initially, only the old version receives traffic. Once tested, the traffic is switched to the new version, which then becomes live while the previous version remains available as a fallback.
```java
public class DeploymentManager {
    public void deployNewVersion(String serviceName) {
        // Deploy both blue and green versions
        // Test the green version in situ
        if (testsPass()) {
            redirectTraffic("green");
        } else {
            rollback();
        }
    }

    private boolean testsPass() {
        // Run in situ tests on the new version
        return true; // Placeholder for actual test logic
    }

    private void redirectTraffic(String version) {
        // Redirect traffic to the specified version
    }

    private void rollback() {
        // Roll back to the old version if issues are detected
    }
}
```
x??

---

#### Canary Releasing
Canary releasing is a technique where a small subset of users or traffic is directed to a new microservice version. If it passes, more users are gradually introduced until full-scale deployment. This allows for gradual and controlled rollouts.
:p What is canary releasing?
??x
Canary releasing involves deploying a new version of a microservice to a small group of users first. If the new version performs well, it is rolled out incrementally to more users before being fully deployed, allowing issues to be identified and addressed in a controlled manner.
```java
public class CanaryReleaseManager {
    public void canaryRelease(String serviceName) {
        // Deploy new version with a small percentage of traffic
        if (testsPassForCanary()) {
            graduallyIncreaseTraffic();
        } else {
            rollback();
        }
    }

    private boolean testsPassForCanary() {
        // Run initial tests on the canary group
        return true; // Placeholder for actual test logic
    }

    private void graduallyIncreaseTraffic() {
        // Gradually increase traffic to the new version
    }

    private void rollback() {
        // Roll back to the old version if issues are detected
    }
}
```
x??
---

