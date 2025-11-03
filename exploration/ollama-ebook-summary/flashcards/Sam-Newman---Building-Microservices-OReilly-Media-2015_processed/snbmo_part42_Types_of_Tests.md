# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 42)

**Starting Chapter:** Types of Tests

---

#### Microservice Deployment Practices Recap
Background context: The provided summary covers best practices for deploying microservices, emphasizing independence, automation, and using tools like AWS. It highlights the importance of a single repository per service and CI builds.

:p What are the main points covered in the recap section?
??x
The key points include:
1. Maintaining the ability to release services independently.
2. Ensuring technology supports independent release.
3. Preferring one repository per microservice.
4. Having one CI build per microservice for separate deployment.
5. Moving towards single-service per host/container using technologies like LXC or Docker.
6. Emphasizing automation and using platforms like AWS for ease of management.
7. Understanding the impact on developers and providing tools for self-service deployments.

x??

---

#### Testing Microservices
Background context: The summary discusses the challenges in testing microservices due to their distributed nature and introduces a quadrant system for categorizing tests by Brian Marick.

:p What does the testing section cover?
??x
The section covers:
1. Challenges in effectively and efficiently testing microservices.
2. Types of tests using the Marick quadrant (technology-facing vs. user-facing).
3. The trend towards automating as much testing as possible.
4. Importance of automated testing for validating software quickly.

x??

---

#### Marick’s Testing Quadrant
Background context: Brian Marick's quadrant categorizes tests into technology-facing and user-facing categories, with subcategories for performance tests, unit tests, acceptance tests, and exploratory testing.

:p What is the Marick testing quadrant?
??x
The Marick testing quadrant divides tests into:
1. Technology-Facing (Performance Tests & Unit Tests)
2. User-Facing (Acceptance Tests & Exploratory Testing)

Example code for a simple performance test in Java:
```java
import org.junit.Test;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class PerformanceTest {
    @Test
    public void testPerformance() {
        long start = System.currentTimeMillis();
        // Code to execute and measure time
        long end = System.currentTimeMillis();
        assertThat("The method should not take longer than 100ms", (end - start) <= 100, is(true));
    }
}
```

x??

---

#### Technology-Facing Tests
Background context: Technology-facing tests are automated and aid developers in creating the system. They include performance tests and small-scoped unit tests.

:p What are technology-facing tests?
??x
Technology-facing tests are those that help developers create the system, such as:
- Performance tests (measure speed and resource usage)
- Small-scoped unit tests (test individual components)

Example code for a simple unit test in Java:
```java
import org.junit.Test;
import static org.junit.Assert.assertEquals;

public class MyServiceTest {
    @Test
    public void testFunctionality() {
        MyService service = new MyService();
        assertEquals(5, service.add(2, 3));
    }
}
```

x??

---

#### User-Facing Tests
Background context: User-facing tests are automated and help non-technical stakeholders understand how the system works. They include large-scoped end-to-end tests and manual testing like user testing.

:p What are user-facing tests?
??x
User-facing tests provide insights to non-technical stakeholders and include:
- Large-scoped, end-to-end tests (acceptance tests)
- Manual testing (like exploratory or user testing)

Example code for a simple acceptance test in Java using Selenium WebDriver (pseudo-code):
```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.testng.Assert;

public class MyWebPageTest {
    @Test
    public void testWebPage() {
        WebDriver driver = initializeWebDriver();
        try {
            WebElement element = driver.findElement(By.id("someId"));
            Assert.assertEquals(element.getText(), "Expected Text");
        } finally {
            driver.quit();
        }
    }
}
```

x??

---

#### Trade-Offs in Automated Testing
Background context: The summary mentions the need to balance speed of release with software quality through automated testing, highlighting that different types of tests have their roles and trade-offs.

:p What are the key considerations when deciding on types of automated tests?
??x
Key considerations include:
1. Balancing speed of release vs. software quality.
2. Deciding how much manual vs. automated testing is needed.
3. Choosing between performance, unit, acceptance, or exploratory tests based on system requirements.

Example of a trade-off scenario in Java:
```java
// Scenario: Opt for fewer but more robust end-to-end tests over many small unit tests
public class MySystem {
    @Test
    public void testEntireWorkflow() {
        // Simulate workflow steps
        boolean result = performComplexOperation();
        Assert.assertTrue(result);
    }
}
```

x??

---

#### Test Pyramid Overview
Background context explaining the concept. The Test Pyramid, as outlined by Mike Cohn in his book "Succeeding with Agile," helps us understand the types and proportions of automated tests needed for a software project. Traditionally, it splits automated tests into three layers: Unit, Service, and UI.
:p What is the Test Pyramid?
??x
The Test Pyramid is a model that illustrates the proportions of different types of automated tests in a software development process, emphasizing the importance of various test levels to ensure comprehensive coverage.
x??

---

#### Ambiguity in Definitions
Background context explaining the concept. The terms used in the Test Pyramid can be ambiguous and have varying interpretations among developers. "Service" is particularly open to interpretation, and definitions of unit tests vary widely.
:p What are some issues with using generic terms like "Service" in test categorization?
??x
Using generic terms like "Service" in test categorization can lead to confusion because these terms may mean different things to different people. For example, a "service" could refer to an internal service within the application or an external API, which complicates the categorization of tests.
x??

---

#### Preference for UI Tests
Background context explaining the concept. The author prefers using "end-to-end" tests for UI interactions rather than sticking with the term "UI" test to avoid ambiguity.
:p Why does the author prefer "end-to-end" over "UI" for tests?
??x
The author prefers "end-to-end" over "UI" because it better captures the essence of testing entire flows and user journeys, reducing confusion. "UI" can be misleading as it might imply only graphical interface interactions without considering other layers.
x??

---

#### Layer Descriptions
Background context explaining the concept. The Test Pyramid traditionally categorizes tests into three levels: Unit, Service, and UI. However, these terms are ambiguous and may need clarification.
:p What are the traditional layers of the Test Pyramid?
??x
The traditional layers of the Test Pyramid are:
- **Unit Tests**: Tests that focus on individual components or functions.
- **Service Tests**: Tests for integration between components (can vary in interpretation).
- **UI Tests**: Tests focusing on end-to-end user interactions with the application.

These layers aim to cover different aspects of the software, from low-level to high-level functionality.
x??

---

#### Example Scenario
Background context explaining the concept. An example scenario is provided involving a helpdesk application and a main website interacting with customer service, which in turn interacts with a loyalty points bank.
:p Describe the example scenario presented in the text?
??x
The example scenario involves a music shop system with:
- A helpdesk application
- A main website 
- Customer service for retrieving, reviewing, and editing customer details
- Interaction with a loyalty points bank where customers accrue points from purchases.

This setup is used to illustrate different testing scenarios across various layers of the Test Pyramid.
x??

---

#### Diving into Scenarios
Background context explaining the concept. The example scenario provides specific interactions that can be tested at different levels of the Test Pyramid.
:p What does the author suggest we should look at in this example?
??x
The author suggests looking at specific interactions within the example to understand how they would be tested at different layers of the Test Pyramid, such as unit tests for individual functions, service tests for interactions between components, and end-to-end tests for full user journeys.
x??

---

#### Unit Tests
Background context explaining the concept. Unit tests typically test a single function or method call, generated by techniques like TDD and property-based testing. These tests are technology-facing and help catch most of the bugs. They are very fast on modern hardware, allowing for running thousands in less than a minute.
If applicable, add code examples with explanations.
:p What is the purpose of unit tests?
??x
Unit tests provide very fast feedback about whether our functionality is good. They support refactoring by catching mistakes during restructuring of code without external dependencies or network connections.

```java
public class CustomerServiceTest {
    @Test
    public void testGetCustomerInfo() {
        // Arrange
        String customerId = "12345";
        Customer expectedCustomer = new Customer("John Doe", 30);
        
        // Act & Assert
        assertEquals(expectedCustomer, customerService.getCustomer(customerId));
    }
}
```
x??

---

#### Service Tests
Background context explaining the concept. Service tests bypass the user interface to test services directly in a monolithic application or individual services in a microservices system. They help improve isolation by stubbing out external collaborators.
If applicable, add code examples with explanations.
:p How do service tests differ from unit tests?
??x
Service tests are more comprehensive as they test individual services' capabilities independently, whereas unit tests focus on smaller functions. Service tests can be faster or slower depending on the complexity of interactions (e.g., real databases, network calls). They provide better isolation and fewer moving parts compared to end-to-end tests.

```java
public class UserServiceTest {
    @Test
    public void testUserService() {
        // Arrange
        String username = "testUser";
        User expectedUser = new User(username);
        
        // Stub external collaborators if needed
        
        // Act & Assert
        assertEquals(expectedUser, userService.getUserByUsername(username));
    }
}
```
x??

---

#### End-to-End Tests
Background context explaining the concept. End-to-end tests cover an entire system, often driving a GUI through a browser or mimicking user interactions. They provide high confidence that production code works but can be tricky to implement in microservices due to increased complexity.
If applicable, add code examples with explanations.
:p What are end-to-end tests useful for?
??x
End-to-end tests cover the entire system scope and simulate real-world usage scenarios, giving developers a sense of confidence that the application will work correctly in production. However, they can be slower and harder to debug when something goes wrong.

```java
public class CustomerServiceTest {
    @Test
    public void testCustomerUploadFile() {
        // Arrange
        WebDriver driver = new ChromeDriver();
        
        // Act & Assert
        driver.get("http://localhost:8080/upload");
        WebElement uploadButton = driver.findElement(By.id("upload-button"));
        File file = new File("path/to/file");
        uploadButton.sendKeys(file.getAbsolutePath());
        assertTrue(driver.getPageSource().contains("File uploaded successfully"));
    }
}
```
x??

---

#### Trade-Offs
Background context explaining the concept. As you move up the test pyramid, the scope of tests increases, along with confidence in functionality working correctly but at the cost of longer feedback cycles and more difficult debugging when a test fails.
If applicable, add code examples with explanations.
:p What are the trade-offs between different levels of tests?
??x
The trade-offs include:
- **Speed vs. Scope**: Unit tests are fast, while end-to-end tests take longer to run.
- **Debugging Complexity**: Smaller unit tests are easier to debug compared to complex end-to-end scenarios.
- **Confidence Levels**: Larger scope like end-to-end provides higher confidence but at the cost of slower feedback cycles.

```java
// Example of a test suite covering different levels
@RunWith(Parameterized.class)
public class TestSuite {
    @Parameterized.Parameters
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][]{
            { "Unit Test", 10, 1_000_000 },
            { "Service Test", 50, 300_000 },
            { "End-to-End Test", 100, 200_000 }
        });
    }

    private String testType;
    private int speed; // in milliseconds
    private int confidence;

    public TestSuite(String testType, int speed, int confidence) {
        this.testType = testType;
        this.speed = speed;
        this.confidence = confidence;
    }

    @Test
    public void test() {
        System.out.println("Running " + testType);
        assertEquals(this.speed > 0 ? "Fast" : "Slow", "Fast");
        assertTrue(confidence >= 90 ? true : false); // This is just an example condition for clarity.
    }
}
```
x??

---

---
#### Importance of Different Scope Tests
Background context explaining that different types of automated tests are used for various purposes, each with trade-offs. The pyramid model (unit tests at the bottom, service and end-to-end tests at the top) is often used to illustrate the need for these different levels of tests.
:p What is the main idea behind using tests of different scope?
??x
The main idea is that unit tests provide fast feedback on small changes, while broader-scoped tests like service and end-to-end tests ensure that the overall system works. However, each type of test has its trade-offs—unit tests are quick but less comprehensive, whereas broader tests cover more ground but take longer to run.
x??

---
#### Feedback Cycles in Testing
Explanation on how different types of tests affect feedback cycles. Smaller-scoped tests like unit tests provide quicker feedback and pinpoint the exact issue, while larger-scoped tests help ensure overall system functionality.
:p How do smaller-scoped tests improve our development process?
??x
Smaller-scoped tests, such as unit tests, allow developers to identify issues quickly by focusing on small code changes. This leads to faster feedback cycles and easier debugging because the failure is often localized to a single line of code or a small component.
x??

---
#### Trade-offs in Test Pyramid
Description that different types of automated tests have trade-offs, with an order of magnitude more tests recommended as you descend the pyramid (unit > service > end-to-end).
:p What rule of thumb is suggested for balancing test coverage across levels?
??x
A good rule of thumb is to have approximately an order of magnitude more unit tests than service or end-to-end tests. For example, if there are 4,000 unit tests, there might be 1,000 service tests and 60 end-to-end tests.
x??

---
#### Monolithic System Example
Example provided with specific numbers to illustrate the balance of tests in a system: 4,000 unit tests, 1,000 service tests, and 60 end-to-end tests. Mentioned that there was an imbalance where too many service and end-to-end tests were present.
:p What does the example demonstrate about test coverage in a monolithic system?
??x
The example demonstrates that while having a large number of unit tests is beneficial for quick feedback, the presence of 1,000 service tests and only 60 end-to-end tests suggests an imbalance. The service and end-to-end tests were identified as problematic, leading to efforts to replace them with more unit tests.
x??

---
#### Test Snow Cone
Description of a test snow cone or inverted pyramid where there are few small-scoped tests but many larger-scoped ones, resulting in slow feedback cycles and long build times.
:p What is the issue with an inverted pyramid (test snow cone) approach to testing?
??x
An inverted pyramid approach leads to very long feedback cycles because large, broad tests take longer to run. This can cause significant delays in identifying issues, especially when integrated into continuous integration builds, where slow test runs reduce build frequency and increase the time a broken system remains undetected.
x??

---
#### Implementing Service Tests
Explanation on how service tests aim to test functionality across multiple services while isolating dependencies using stubs. Need to deploy a binary artifact for the service under test and configure stub services for any downstream collaborators.
:p How do you implement service tests?
??x
Service tests require deploying a stubbed version of collaborating services to isolate the service under test. The steps include:
1. Deploying the binary artifact for the service.
2. Launching or ensuring that stub services are running.
3. Configuring the service under test to connect to these stub services.
4. Setting up the stubs to return expected responses.

Example code might look like this:
```java
// Pseudocode example
public class ServiceUnderTest {
    private final CustomerService customerService;
    
    public ServiceUnderTest(CustomerService customerService) {
        this.customerService = customerService;
    }
    
    public void testFunctionality() {
        // Test logic here, using the stubbed services to mock external dependencies.
    }
}

// Stub service configuration
public class StubCustomerService implements CustomerService {
    @Override
    public Points getPoints(Customer customer) {
        if (customer.getId().equals("expectedCustomerId")) {
            return 100;
        }
        return 0;
    }
}
```
x??

---
#### Mocking and Stubs
Explanation on the difference between mocking and stubbing. Stubbing involves creating a mock service that responds with canned responses to known requests.
:p What is the purpose of stubbing in testing?
??x
Stubbing is used to simulate or replace real dependencies during tests, allowing for isolated unit tests. By setting up predefined responses (canned responses), stubs enable developers to test specific behaviors without relying on external services, which might be slow or unreliable.
x??

---

#### Stub vs Mock Approach
Background context: The passage discusses the differences between using stubs and mocks in testing, explaining that while stubs are used to simulate a dependency without worrying about the number of calls, mocks ensure that expected methods are called. This is particularly relevant for service tests where interactions with external services need to be controlled.
:p What is the main difference between using a stub and a mock when testing?
??x
The key difference lies in their behavior regarding method calls:
- A **stub** returns predefined responses but doesn't care if it's called multiple times or not. It’s used for simulating dependencies without worrying about interaction details.
- A **mock**, on the other hand, ensures that specific methods are called and can fail tests if these expectations aren’t met.

For example, in a service test:
```java
// Stub Example (simulates a dependency)
public class CustomerServiceTest {
    @Test
    public void testBalance() {
        PointsBankStub pointsBank = new PointsBankStub();
        pointsBank.setReturnValueForCustomer123(15000);
        
        // Test logic using the stubbed points bank.
    }
}

// Mock Example (ensures a method is called)
public class CustomerServiceTest {
    @Test
    public void testBalance() throws Exception {
        PointsBankMock pointsBank = new PointsBankMock();
        pointsBank.expects().method("getCustomerBalance").withArgs(123);
        
        // Test logic using the mocked points bank.
    }
}
```
x??

---
#### Scaffolding Tests with Stubs and Mocks
Background context: The passage suggests that while stubs are used to simplify tests by providing predefined responses, mocks are more detailed and can ensure specific interactions happen. This distinction is crucial for maintaining test isolation and ensuring the correct behavior of dependent services.
:p Why might one choose a mock over a stub in service testing?
??x
One chooses a mock over a stub when they need to verify that certain methods on a dependency were called during the execution of a test. Stubs are used primarily to provide consistent responses, whereas mocks can enforce expected interactions and validate method calls.

Example scenario:
```java
// Mock Example (ensures specific interaction)
public class CustomerServiceTest {
    @Test
    public void testCreateCustomer() throws Exception {
        PointsBankMock pointsBank = new PointsBankMock();
        pointsBank.expects().method("initializeBalance").withArgs(123);
        
        // Logic to create a customer and expect the balance initialization.
    }
}
```
x??

---
#### Implementing Stubs Manually vs Using Tools
Background context: The text mentions various methods of creating stubs, from using web servers like Apache or Nginx to embedded Jetty containers. However, it also highlights the utility of tools like Mountebank for programmatically setting up stub endpoints.
:p What is Mountebank and how does it simplify test setup?
??x
Mountebank is a software appliance that can be programmed via HTTP requests to simulate various types of services (HTTP, TCP) as stubs or mocks. It simplifies the process by allowing users to define responses, set expectations, and dynamically manage these configurations without manually setting up servers.

Example usage:
```java
// Starting Mountebank with a specific port and protocol
public class TestSetup {
    public void startMountebank() throws Exception {
        ProcessBuilder pb = new ProcessBuilder("mountebank", "-p", "8080");
        pb.inheritIO();
        Process process = pb.start();
        
        // Configure Mountebank to stub a service.
        URL url = new URL("http://localhost:8080/stubs/points-bank");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("PUT");
        String response = "{ \"responses\": [{ \"delay\": 1, \"body\": {\"balance\": 15000} }] }";
        conn.setDoOutput(true);
        OutputStream os = conn.getOutputStream();
        os.write(response.getBytes());
        os.flush();
    }
}
```
x??

---
#### Deploying Service Tests with Dependency Simulations
Background context: The passage concludes by discussing the implications of using stubs and mocks in deployment scenarios. It emphasizes that while tests may pass when only the primary service is tested, changes could potentially break dependent services.
:p What potential risk does Mountebank testing face during deployment?
??x
Mountebank testing can introduce a risk where passing tests for a single service do not guarantee that all dependent services will work correctly in production. This means that while the customer service passes tests using the stubbed points bank, changes could potentially break other services like the helpdesk and web shop if they rely on the same behavior.

Example scenario:
```java
public class DeploymentTest {
    @Test
    public void testDeployment() throws Exception {
        // Start Mountebank to simulate the points bank.
        startMountebank();
        
        // Test customer service with stubbed points bank.
        CustomerService customerService = new CustomerService();
        assertTrue(customerService.verifyCustomerBalance(123, 15000));
        
        // Deploy the customer service and verify that helpdesk and web shop also work correctly.
    }
}
```
x??

---

#### End-to-End Tests Overview
Background context: In a microservice architecture, end-to-end tests are crucial to ensure that all services work together as expected. These tests drive functionality through user interfaces and cover a wide range of the system, providing high confidence but at the cost of speed and diagnostic complexity.
:p What is the purpose of end-to-end tests in a microservice system?
??x
End-to-end tests aim to verify the interaction between multiple services in a microservice architecture. They simulate real-world user interactions and provide an overview of how different components work together, ensuring that changes do not introduce bugs or break existing functionality.
```java
// Example pseudocode for running end-to-end tests
public void runEndToEndTests() {
    // Deploy all services together
    deployServices();
    
    // Run tests against the deployed services
    runUserInterfaceTests();
}
```
x??

---

#### Deployment Strategy for End-to-End Tests
Background context: To effectively manage end-to-end tests, a deployment strategy that integrates them with other pipelines is necessary. This ensures that changes in one service can trigger tests on dependent services without manual intervention.
:p How should end-to-end tests be integrated into the CI/CD pipeline?
??x
End-to-end tests should be part of a multi-pipeline model where each service's build triggers an integration test stage, which then runs end-to-end tests. This approach ensures that any change in one service can trigger tests on dependent services automatically.
```java
// Pseudocode for integrating end-to-end tests into CI/CD pipeline
public void integrateEndToEndTests() {
    // Define a listener for service builds
    buildListener = new BuildListener();
    
    // Register the listener to trigger integration tests
    buildListener.registerTrigger(new IntegrationTestTrigger());
    
    // Whenever a build is triggered, run end-to-end tests
    buildListener.onBuildTriggered(() -> runEndToEndTests());
}
```
x??

---

#### Fan-in Model for End-to-End Tests
Background context: A fan-in model allows multiple services to trigger end-to-end tests in a single stage. This approach ensures that changes are validated across all dependent components without duplicating effort or deploying unnecessary services.
:p What is the benefit of using a fan-in model for end-to-end tests?
??x
The fan-in model benefits from avoiding redundant deployments and tests by having multiple pipelines trigger the same end-to-end test stage. It ensures that every time a service changes, its related tests are run, but it also validates interactions with other services without additional manual steps.
```java
// Pseudocode for implementing fan-in model
public void implementFanInModel() {
    // Define an integration test pipeline
    IntegrationTestPipeline pipeline = new IntegrationTestPipeline();
    
    // Add multiple service build stages to the pipeline
    ServiceAStage stageA = new ServiceAStage();
    ServiceBStage stageB = new ServiceBStage();
    
    // Fan-in by adding both stages to the end-to-end tests stage
    EndToEndTestsStage endToEndStage = new EndToEndTestsStage();
    endToEndStage.addStage(stageA);
    endToEndStage.addStage(stageB);
}
```
x??

---

#### Handling Different Versions in End-to-End Tests
Background context: When deploying services, it's critical to determine which versions of dependent services should be used for end-to-end tests. This decision can impact the accuracy and reliability of the test results.
:p How do you handle different versions of services during end-to-end testing?
??x
To handle different versions of services, you can either use production versions or the latest staging versions, depending on the CI/CD pipeline's maturity. A common approach is to fan in multiple pipelines that run end-to-end tests against both stable and staging environments.
```java
// Pseudocode for handling different service versions
public void handleServiceVersions() {
    // Define a method to get the correct version based on deployment stage
    ServiceVersion getCorrectVersion(String stage) {
        if (stage.equals("production")) {
            return productionVersion;
        } else {
            return stagingVersion;
        }
    }
    
    // Use this method in end-to-end tests
    EndToEndTestsStage endToEndStage = new EndToEndTestsStage(getCorrectVersion);
}
```
x??

---

#### Avoiding Redundant Tests and Effort
Background context: Running the same tests across multiple services can lead to redundant effort. It's essential to design a pipeline that minimizes this redundancy while ensuring comprehensive coverage.
:p How can you avoid redundant tests in end-to-end testing pipelines?
??x
To avoid redundant tests, ensure that each service has its own local and integration tests before running end-to-end tests. This approach ensures that the end-to-end tests focus only on interactions between services rather than duplicating functionality.
```java
// Pseudocode for avoiding redundant tests
public void avoidRedundantTests() {
    // Define a pipeline stage for each service's local and integration tests
    ServiceATestStage stageA = new ServiceATestStage();
    ServiceBTestStage stageB = new ServiceBTestStage();
    
    // Fan in these stages to the end-to-end test stage
    EndToEndTestsStage endToEndStage = new EndToEndTestsStage();
    endToEndStage.addStage(stageA);
    endToEndStage.addStage(stageB);
}
```
x??

