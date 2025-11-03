# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 18)

**Starting Chapter:** Consumer-Driven Tests to the Rescue

---

---
#### Flaky and Brittle Tests
Background context: As test scope increases, so too do the number of moving parts. These can introduce failures that are not related to the functionality under test. Temporary issues like network glitches or race conditions can cause tests to fail despite the core functionality being correct.

:p What is a flaky test?
??x
A test that sometimes fails and sometimes passes, providing little useful information about the system's state. These tests are often brittle because they may depend on external factors such as service availability, network conditions, or race conditions.
x??

---
#### Normalization of Deviance
Background context: Over time, if we accept flaky tests as normal, it can lead to a situation where broken functionality is tolerated, leading to potential issues in the application. This phenomenon is described by Diane Vaughan.

:p What does the normalization of deviance refer to?
??x
It refers to the human tendency to accept something as normal over time even if it's problematic initially. In testing terms, this means that flaky tests can become accepted as part of the standard, potentially leading to a disregard for actual issues in the system.
x??

---
#### Test Ownership and Maintenance
Background context: For end-to-end tests involving multiple services, determining who should write and maintain these tests is crucial. Initially, one might think each service's team owns their respective tests. However, when dealing with shared end-to-end tests, deciding on ownership becomes more complex.

:p Who writes the end-to-end tests in a multi-team environment?
??x
In a multi-team environment, the end-to-end tests should ideally be maintained by a cross-functional team that includes representatives from each service's owning team. Alternatively, these tests might be owned by a dedicated test or quality assurance (QA) team.
x??

---
#### Strategies to Mitigate Flaky Tests
Background context: To address flaky tests, various strategies can be employed such as rewriting the tests to avoid multi-threading issues, improving environment stability, or replacing flaky tests with smaller, more deterministic ones.

:p How should flaky tests be handled?
??x
Flaky tests should be identified and tracked down. If immediate fixes are not possible, they should be removed from the test suite. Efforts can include rewriting the tests to avoid multi-threading issues, making the underlying environment more stable, or replacing them with smaller, less problematic tests.
x??

---
#### Example of Flaky Test Handling
Background context: Martin Fowler advocates for removing flaky tests and finding ways to make them deterministic.

:p What advice does Martin Fowler give regarding flaky tests?
??x
Martin Fowler suggests tracking down flaky tests. If immediate fixes are not possible, these tests should be removed from the suite. Rewriting tests to avoid multi-threading issues, improving environment stability, or replacing flaky tests with smaller tests can help.
x??

---
#### Code Example for Flaky Test Identification
Background context: To identify and track down flaky tests, it's useful to log detailed information about test failures.

:p How can we log detailed failure information in a test?
??x
We can log detailed failure information by capturing logs during test execution. Here’s an example of how this might be done using Java:

```java
public class TestLogger {
    private static final Logger logger = LoggerFactory.getLogger(TestLogger.class);

    public void runTest() {
        try {
            // Test logic here
            if (/* test fails */) {
                logger.error("Test failed with stack trace: ", e);
                throw new RuntimeException("Test failed");
            }
        } catch (Exception e) {
            logger.error("Test caught an exception: ", e);
        }
    }
}
```
x??

---

---

#### Free-for-All Test Environment
Background context: The issue arises when all teams have unrestricted access to add tests without a clear understanding of the overall test suite health. This leads to an uncontrolled increase in test cases, potentially causing the "test snow cone" phenomenon where the number and complexity of tests overwhelm the system.

:p What are the key issues with having an uncontrolled environment for adding end-to-end tests?
??x
The main issues include:
1. **Explosion of Test Cases**: Teams might add redundant or irrelevant tests without a cohesive strategy.
2. **Health Ignored**: Lack of clear ownership means that test results may be ignored, and when they break, teams assume it's someone else’s problem.
3. **Increased Cycle Times**: Developers wait for dedicated testing teams to write end-to-end tests, increasing development cycles.

??x
---

#### Dedicated Team Ownership vs. Joint Ownership
Background context: A common organizational response is to centralize test writing in a dedicated team, but this can lead to decreased involvement from the original service developers, causing delays and reduced knowledge of the tests' specifics.

:p Why is joint ownership between teams and their respective test suites beneficial?
??x
Joint ownership ensures:
1. **Proximity**: Teams remain closely involved with the code they write, reducing distance and increasing accountability.
2. **Speed**: Faster response to failures as the developers can quickly diagnose issues without waiting for a separate team.
3. **Quality**: Higher quality tests as developers have more context on their service's functionality.

??x
---

#### Slowness of End-to-End Test Suites
Background context: The issue arises when end-to-end test suites are slow to run, often due to lack of optimization and comprehensive coverage, leading to delays in development cycles and reduced developer engagement.

:p Why is the slowness of end-to-end tests a significant problem?
??x
The main problems include:
1. **Delayed Feedback**: Developers may take hours or even days to identify broken functionality.
2. **Context Switches**: Frequent interruptions as developers need to shift focus from coding to debugging test failures.

??x
---

#### Parallel Execution and Test Suite Optimization
Background context: Running tests in parallel can mitigate the slowness issue, but it's not a substitute for actively optimizing the suite by removing redundant or flaky tests. Tools like Selenium Grid facilitate parallel execution.

:p How can we improve the performance of end-to-end test suites?
??x
Improvements include:
1. **Parallel Execution**: Using tools like Selenium Grid to run tests in parallel.
2. **Optimization**: Regularly reviewing and refining the suite to reduce redundancy and flakiness.
3. **Contextual Relevance**: Ensuring that every test is relevant and contributes meaningfully to coverage.

??x
---

#### Flakiness of End-to-End Tests
Background context: Flaky tests can significantly hamper development efficiency, as they may break for reasons unrelated to the actual code changes. This requires careful management and continuous optimization.

:p What challenges do flaky end-to-end tests present?
??x
Challenges include:
1. **False Positives/Negatives**: Tests that fail or pass intermittently due to external factors.
2. **Delayed Feedback**: Developers may not get accurate feedback on their code changes, leading to misdirection in debugging.

??x
---

#### Joint Ownership as the Best Balance
Background context: The ideal approach is a shared ownership model where multiple teams contribute to and are responsible for the end-to-end test suite. This balances centralization with decentralization, ensuring both accountability and flexibility.

:p How does joint ownership improve the management of end-to-end tests?
??x
Joint ownership improves:
1. **Shared Responsibility**: Multiple teams take responsibility, reducing the burden on any single team.
2. **Cohesive Strategy**: A shared understanding of test priorities and coverage ensures a more coherent suite.
3. **Developer Involvement**: Developers remain engaged with their code's tests, enhancing quality and responsiveness.

??x
---

#### Balancing Test Value and Burden
Background context: Discusses the trade-off between maintaining a large number of tests to ensure comprehensive coverage and the risk associated with removing or refactoring tests, which can lead to missed bugs.

:p How do you balance the value added by tests against the burden they entail?
??x
The question focuses on understanding how to manage the cost-benefit analysis in testing. Given that each test removed might prevent a bug from being caught but could also save time and effort, it's crucial to have a clear risk assessment framework.

In practice, identifying redundant or less valuable tests can improve efficiency without compromising too much on coverage. However, this is challenging due to the inherent difficulties humans have in assessing risk accurately.
x??

---
#### Long Feedback Cycles and Pile-Ups
Background context: Describes how long feedback cycles from end-to-end tests impact developer productivity and deployment processes, leading to pile-ups where multiple changes accumulate during test failures.

:p How does a long feedback cycle affect the deployment process?
??x
A long feedback cycle means that fixing issues in end-to-end tests takes significant time. This delays the availability of deployable services because only software that passes all tests can be deployed. Consequently, while waiting for tests to pass, other changes from upstream teams might pile up, increasing the complexity and risk of deployments.

For instance, if a seven-hour-long build is broken, developers cannot check in code until it's fixed, which can lead to delays in deploying new features.
x??

---
#### Metaversion and Microservices
Background context: Discusses the concept of "metaversion" where multiple services are versioned together, leading to coupled systems that lose the advantage of independent deployment.

:p What is the concept of metaversion?
??x
Metaversion refers to the practice of treating a system as a whole unit for versioning and releasing updates. This means changing and deploying all related microservices at once rather than independently.

The main issue with metaversion is that it can lead to services becoming tightly coupled, which was one of the key benefits of using microservices (i.e., being able to deploy changes in isolation).

For example:
```java
public class MicroserviceA {
    public void updateService() {
        // Code for updating Service A
    }
}

public class MicroserviceB {
    public void updateService() {
        // Code for updating Service B
    }
}
```
Metaversion would involve bundling these updates together, potentially leading to issues if one service fails during deployment.
x??

---

#### End-to-End Test Journeys vs. Stories
Background context: The passage discusses the challenges and limitations of end-to-end (E2E) tests when dealing with multiple services, highlighting the potential for test suite bloat and inefficiency.

:p What is a key issue with using E2E tests for multiple services?
??x
The key issue with using E2E tests for multiple services is that they can quickly become bloated, leading to an explosion in scenarios under test. This situation worsens if a new end-to-end test is added for every piece of functionality introduced, resulting in poor feedback cycles and overlapping test coverage.

Example: In a system with 10 services, adding one E2E test per service could result in 100 individual tests that might overlap significantly.
x??

---

#### Core Journeys vs. Individual Functionalities
Background context: The passage suggests focusing on core journeys rather than individual functionalities to manage the complexity of end-to-end tests.

:p How can you manage complex systems with end-to-end tests without making them bloated?
??x
By focusing on a small number of core journeys that cover high-value interactions, you can reduce the downsides of integration tests. These core journeys need to be mutually agreed upon and jointly owned by relevant teams. For example, in a music shop system, key journeys might include ordering a CD, returning a product, or creating a new customer.

Example: If each journey covers a critical interaction like "order a CD," you can ensure that the core functionality is tested without overwhelming the test suite with individual functional tests.
x??

---

#### Consumer-Driven Contracts (CDCs)
Background context: The passage introduces consumer-driven contracts as an alternative to traditional end-to-end testing, focusing on ensuring service interoperability.

:p What is the purpose of using consumer-driven contracts in software development?
??x
The purpose of using consumer-driven contracts (CDCs) is to ensure that when a new service is deployed to production, it won’t break its consumers. CDCs define the expectations of consumers on a service and capture these expectations as tests, which are then run against the producer.

Example: For a customer service with two consumers—helpdesk and web shop—the tests would validate how each consumer uses the service.
x??

---

#### Consumer-Driven Tests in Practice
Background context: The passage explains how to implement and benefit from consumer-driven tests, focusing on their integration into CI/CD pipelines.

:p How are consumer-driven tests run against a single producer?
??x
Consumer-driven tests (CDCs) are run against the producer service by itself with any downstream dependencies stubbed out. This allows these tests to be isolated and faster compared to end-to-end tests. For example, in our customer service scenario, both the helpdesk’s and web shop’s expectations would be tested separately but independently of each other.

Example: 
```java
// Pseudocode for a CDC test
public void testCustomerServiceWithHelpdesk() {
    // Stub out all dependencies except those relevant to the helpdesk
    // Run tests that validate how the customer service behaves as seen by the helpdesk
}

public void testCustomerServiceWithWebShop() {
    // Stub out all dependencies except those relevant to the web shop
    // Run tests that validate how the customer service behaves as seen by the web shop
}
```
x??

---

#### Integration into Test Pyramid
Background context: The passage describes where consumer-driven tests fit in the test pyramid and their unique focus.

:p How do consumer-driven tests sit in relation to other types of tests in a test pyramid?
??x
Consumer-driven tests (CDCs) sit at the same level as service-level tests in the test pyramid, but with a different focus. They are focused on validating how a consumer will use the service and can be run against the service independently of its downstream dependencies.

Example:
```java
// Pseudocode for placing CDCs in the test pyramid
public class ServiceTests {
    @Test
    public void testCustomerServiceWithHelpdesk() {
        // Test logic here
    }

    @Test
    public void testCustomerServiceWithWebShop() {
        // Test logic here
    }
}

// Higher-level integration tests would be located further up the pyramid, testing multiple services together.
```
x??

---

#### Overview of Pact
Pact is a consumer-driven testing tool that was originally developed for Ruby but now supports JVM and .NET. It allows consumers to define expectations about producer behavior, creating JSON-based pact files that are used to verify the producer's API. This approach ensures that both parties have a clear understanding of their interaction.
:p What does Pact enable in software development?
??x
Pact enables consumer-driven testing by allowing developers to define what they expect from an API and then verify these expectations against the actual implementation. This process helps ensure that changes in one component do not break the functionality expected by other components. The use of JSON specifications makes it language-agnostic, meaning different parts of a system can work with the same pact files.
??x
---

#### Pact Workflow Overview
In the Pact workflow, the consumer starts by defining expectations using a Ruby DSL. These expectations are then tested against a local mock server to create a pact file. On the producer side, this JSON pact file is used to verify that the API behaves as expected. This process requires both the consumer and producer to be in different builds.
:p How does Pact work between the consumer and producer?
??x
Pact works by having the consumer define expectations using a Ruby DSL, which are then tested against a mock server to generate a pact file. On the producer side, this JSON pact file is used to drive API calls and verify responses. The key steps are:
1. Consumer defines expectations.
2. Mock server tests these expectations.
3. Pact file is created.
4. Producer uses the pact file to ensure compliance.
```ruby
# Example of defining a contract in Ruby using Pact DSL
contract = Pact.new('consumer', 'provider') do
  upon_receiving("a request for data") do
    with_stubs(path: '/data', method: :get) do |stubbed_request|
      will_respond_with(
        status: 200,
        body: { "key" => "value" },
        headers: { 'Content-Type' => 'application/json' }
      )
    end
  end
end

contract.provide_pact to_file: 'pact.json'
```
x??

---

#### Pact Specification File
The pact file is a JSON specification that formalizes the interaction between consumer and producer. It can be hand-coded but using the language API like Pact’s Ruby DSL makes it easier, especially for testing purposes.
:p What is the purpose of a pact file in Pact?
??x
The purpose of a pact file in Pact is to document and validate the expected interactions between a consumer and a provider. This JSON-based specification ensures that both parties have clear expectations about API behavior, which can be verified at different stages of development.

Using the language API (like Pact’s Ruby DSL) makes it easier to generate these specifications dynamically during tests.
??x
---

#### Using Pact for Multi-Language Projects
Pact's ability to use a JSON-based pact file across multiple languages is particularly useful. For example, you can define expectations in Ruby but verify them against a Java implementation using the JVM port of Pact.
:p How does Pact support multi-language projects?
??x
Pact supports multi-language projects by allowing developers to create and validate API contracts using language-specific tools (like Pact’s Ruby DSL) while ensuring compatibility across different languages. The JSON-based pact files are portable, meaning they can be used with different programming languages without needing to rewrite the contract definitions.

For instance:
- Define expectations in a Ruby consumer.
- Use these expectations to generate a pact file.
- Verify this pact file against a Java producer using Pact’s JVM port.
```java
// Example of verifying a pact file in Java
PactBroker pactBroker = new PactBroker("http://localhost:1234");
Consumer consumer = new Consumer("ExampleConsumer", "1.0").hasPactsWith(pactBroker);
Provider provider = new Provider("ExampleProvider", "1.0");

consumer.given("some condition")
  .uponReceiving("a request for data")
  .withHeaders(Map.of("Accept", "application/json"))
  .willRespondWith()
    .status(200)
    .body("{\"key\":\"value\"}")
    .headers(Map.of("Content-Type", "application/json"));

provider.verifyPact(consumer);
```
x??

---

#### Pacto vs. Pact
Pacto is another Ruby tool for consumer-driven testing that records interactions to generate expectations. While it simplifies creating initial contracts, these are static and not regenerated with each build as in Pact.
:p How does Pacto differ from Pact?
??x
Pacto differs from Pact primarily in its approach to contract generation and verification:
- **Static Contracts:** Once generated, the expectations in Pacto remain static. They do not change with new builds or versions of the consumer.
- **Dynamic Regeneration:** With Pact, the consumer regenerates the expectations in every build, ensuring that they reflect the latest requirements.

Pacto is useful for quickly setting up initial contracts by recording interactions, while Pact ensures that these contracts are dynamic and continuously validated against evolving requirements.
??x
---

#### Agile Stories and Conversations
Agile stories are placeholders for conversations that define what a service API should look like. CDCs (Conversations Driven Contracts) codify these discussions and serve as triggers for evolving APIs when they break.
:p What is the purpose of agile stories?
??x
Agile stories act as placeholders for detailed discussions about what an API should do, ensuring that all stakeholders understand the requirements before development starts. They are a way to define user needs in a collaborative manner.
x??

---

#### CDCs and Communication
CDCs require good communication and trust between consumer and producing services. In intra-team or same-person scenarios, this is easier but in third-party or large-scale public API consumption, frequent communication and trust might be lacking.
:p What challenges can arise when implementing CDCs for third-party services?
??x
Implementing CDCs for third-party services can be challenging due to the lack of direct communication channels and mutual trust. This can lead to misalignments in API expectations and difficulties in maintaining consistent contract evolution.
x??

---

#### End-to-End Tests vs. CDCs
While end-to-end tests are valuable, they have significant drawbacks as the number of moving parts increases. Many organizations prefer CDCs for intra-team or internal APIs but use end-to-end tests with semantic monitoring to catch issues before production.
:p Why might an organization choose not to use end-to-end tests exclusively?
??x
An organization may opt against exclusive reliance on end-to-end tests because they introduce overhead, are slow, and do not scale well with increasing complexity. CDCs and improved monitoring often provide better insights into API contract integrity without the downsides of comprehensive end-to-end testing.
x??

---

#### Semantic Monitoring
Semantic monitoring involves using pre-existing end-to-end tests to monitor production systems after deployment. This approach helps catch issues in real-world scenarios that might not be covered by unit or integration tests.
:p What is semantic monitoring and how does it differ from traditional end-to-end testing?
??x
Semantic monitoring uses existing end-to-end tests to continuously monitor a live production system, providing insights into actual usage and performance. Unlike traditional end-to-end tests run before deployment, semantic monitoring focuses on the real-world usage patterns post-deployment.
x??

---

#### Blue/Green Deployments
Blue/green deployments involve deploying two versions of an application simultaneously, where one receives live traffic while the other is tested in situ. This technique helps detect and fix issues before they affect all users.
:p How does a blue/green deployment work?
??x
In a blue/green deployment, both production environments (blue and green) run identical code but receive different traffic. The new version (green) is tested thoroughly while live traffic continues to flow through the old version (blue). Once confirmed as stable, all traffic switches to the new version.
```java
public class BlueGreenDeployment {
    private Map<String, String> dnsEntries;
    private LoadBalancer lb;

    public void deployNewVersion(String newVersion) {
        // Update DNS entries and load balancer to direct traffic to blue environment
        updateTraffic(newVersion, "blue");

        // Perform testing on the green version
        testGreenEnvironment(newVersion);

        // Switch all traffic to the new green version if tests pass
        updateTraffic(newVersion, "green");
    }

    private void updateTraffic(String newVersion, String environment) {
        dnsEntries.put(environment, newVersion);
        lb.updateLoadBalancingConfiguration(dnsEntries);
    }
}
```
x??

---

#### Canary Releasing
Canary releasing is a deployment strategy where a small percentage of users are directed to the new version of an application. This approach helps detect issues without impacting all users and provides feedback for gradual rollouts.
:p What is canary releasing?
??x
Canary releasing involves deploying a new version of an application to a small, controlled group of users (typically 1% or less) before full-scale release. It allows teams to gather feedback and ensure the new version works as expected in a production-like environment.
x??

---

