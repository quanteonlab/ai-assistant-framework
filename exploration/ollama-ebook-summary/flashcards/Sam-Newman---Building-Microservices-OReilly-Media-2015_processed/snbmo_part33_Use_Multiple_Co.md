# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 33)

**Starting Chapter:** Use Multiple Concurrent Service Versions

---

#### Multiple Concurrent Service Versions

Background context: This versioning approach involves running different versions of a microservice simultaneously. It is particularly useful when there are legacy consumers that cannot be immediately updated to the latest service version, due to high costs or constraints with older devices and systems.

:p What are the main reasons for using multiple concurrent service versions?
??x
The main reasons include:
- Supporting legacy consumers that can't be quickly upgraded.
- Handling rare cases where it's too costly to change old consumer systems.
- Ensuring backward compatibility while allowing newer features in updated services. 

This approach is used sparingly by Netflix, especially when dealing with older or embedded devices that are tied to specific API versions.

---

#### Codebase Branching for Multiple Versions

Background context: When implementing multiple concurrent service versions, it often requires branching the codebase to maintain two different sets of services. This can lead to complexity and issues related to merging changes between branches.

:p What is a significant challenge when using codebase branching for maintaining multiple service versions?
??x
A significant challenge is that branching the codebase for each service version means having to manage two distinct sets of code, which complicates the development process and increases the risk of integration issues. It also requires careful management of feature branches and merges.

```java
public class ExampleBranching {
    private String branchVersion;

    public ExampleBranching(String branch) {
        this.branchVersion = branch;
    }

    // Code to handle logic differently based on branch version
}
```
x??

---

#### Middleware for Routing

Background context: In the multiple concurrent service versions approach, smarts are required to direct consumers to the correct microservice version. This often involves implementing routing logic in middleware or a bunch of `nginx` scripts.

:p What is a potential downside of implementing routing logic using middleware?
??x
A potential downside is that routing logic can become complex and hard to reason about. The behavior for directing traffic to specific service versions might end up scattered across multiple files or systems, leading to increased difficulty in understanding how the system handles different consumer versions.

```java
public class ServiceRouter {
    public void routeRequest(String consumerVersion) {
        if (consumerVersion.equals("oldVersion")) {
            // Route to old version of the service
        } else {
            // Route to new version of the service
        }
    }
}
```
x??

---

#### Persistent State Management

Background context: When managing multiple concurrent service versions, it is crucial to ensure that data created by one version of a service can be accessed and used by all other versions. This involves handling persistent state in a way that is consistent across different service versions.

:p What are the challenges related to managing persistent state with coexisting service versions?
??x
Challenges include:
- Ensuring that customers created by either version of the service can be stored and made visible to all services, regardless of which version was used to create them.
- Managing data consistency when different versions of a microservice need access to the same persistent state.

These challenges can introduce additional complexity in designing and maintaining the system.

x??

---

#### Blue/Green Deployments

Background context: Coexisting multiple service versions is particularly useful during blue/green deployments or canary releases. These deployment strategies allow for gradual rollouts, minimizing risks by keeping old and new services running concurrently for a short period.

:p How does a blue/green deployment work?
??x
In a blue/green deployment:
- Two identical environments (blue and green) run in parallel.
- Initially, all traffic is directed to the blue environment.
- When ready, the blue environment is shut down, and traffic is redirected to the green one.

This process allows for seamless switching between versions without downtime.

```java
public class BlueGreenDeployer {
    public void deployNewVersion(String newEnv) {
        if (newEnv.equals("green")) {
            // Switch traffic from blue to green
        } else {
            // Maintain current setup
        }
    }
}
```
x??

---

#### User Interface Evolution
Background context: The text discusses how user interfaces have evolved over time, starting from desktop applications using Swing and Motif to web-based interfaces with JavaScript for dynamic behavior. It then moves towards a more holistic digital approach where various constraints need to be considered.

:p What are the key stages of evolution in user interface design mentioned in the text?
??x
The key stages of evolution in user interface design include:
1. Early desktop applications using technologies like Swing and Motif.
2. Web-based interfaces where most logic is on the server side, rendering full pages to clients.
3. Introduction of JavaScript for more dynamic client-side interactions.
4. Current focus on digital holistically, considering various devices and constraints.

x??

---
#### Thin vs Fat UIs
Background context: The text contrasts thin and fat user interfaces, explaining that in early web applications, the server rendered the entire page with minimal interaction from the browser. With time, more dynamic behavior has been added to client-side interfaces using JavaScript.

:p How did the nature of user interfaces change from their initial stage to modern web applications?
??x
Initially, user interfaces were "thin" where most logic resided on the server side and rendered full pages to clients. Modern web applications have become "fat," integrating more dynamic behavior with JavaScript running in the browser.

Example code showing a simple fat UI using JavaScript:
```javascript
// Pseudocode for a simple interactive form that updates dynamically
function updateForm() {
    let input = document.getElementById('inputField').value;
    document.getElementById('outputField').innerText = 'You entered: ' + input;
}
```

x??

---
#### Desktop vs Web Applications
Background context: The text contrasts desktop applications, which were initially more "fat" with local file manipulation and server-side components, to web applications that started as "thin," where the server rendered full pages.

:p How did the nature of user interfaces change from traditional desktop applications to modern web applications?
??x
Desktop applications were traditionally "fat," characterized by rich client-side experiences involving local file manipulation and extensive use of technologies like Swing. Web applications initially emerged as "thin," with most logic on the server rendering full pages to clients. Over time, these evolved into more dynamic and interactive experiences using JavaScript.

Example code showing a simple form submission in an early web application:
```html
<!-- Early web application HTML -->
<form action="/submit" method="post">
    <input type="text" name="message">
    <button type="submit">Submit</button>
</form>
```

x??

---
#### Holistic Digital Approach
Background context: The text discusses the shift towards a more holistic digital approach where different interfaces (desktop, mobile, wearable, physical) are considered to deliver services.

:p What is the current trend in designing user interfaces according to the text?
??x
The current trend in designing user interfaces involves adopting a more holistic and integrated approach. Instead of treating web or mobile applications separately, organizations aim to provide seamless experiences across various devices like desktops, mobile devices, wearables, and physical stores.

Example code showing how microservices can be used to curate different experiences:
```java
// Pseudocode for a service that adapts based on the user's device type
public class UserInterfaceService {
    public String getUIExperience(String deviceType) {
        if ("mobile".equals(deviceType)) {
            return "Mobile UI";
        } else if ("web".equals(deviceType)) {
            return "Web UI";
        } else {
            return "Desktop UI";
        }
    }
}
```

x??

---
#### Constraints in User Interface Design
Background context: The text highlights the various constraints that need to be considered when designing user interfaces, including browser compatibility, screen resolution, and mobile-specific limitations such as network bandwidth and battery life.

:p What are some of the key constraints mentioned for modern web applications?
??x
Key constraints for modern web applications include:
- Browser compatibility and user agent variations.
- Screen resolution and aspect ratios.
- Network conditions and bandwidth limitations, especially on mobile devices.
- Battery management concerns that can affect user experience due to interactions draining battery life.

Example code showing a simple check for screen size in JavaScript:
```javascript
// Pseudocode for checking the screen size and adapting UI
function adaptUI() {
    if (window.innerWidth > 768) {
        // Large screen - apply desktop styles
    } else {
        // Small screen - apply mobile or tablet styles
    }
}
```

x??

---

#### Tablet Interaction Constraints
Background context: The nature of interactions changes significantly when using different devices like tablets. Traditional mouse-based interactions such as right-clicking are not straightforward on touchscreens, requiring adaptation for better user experience.

:p How do tablet interaction constraints differ from traditional desktop interactions?
??x
Tablet interactions are primarily based on touch and gestures rather than a mouse or keyboard. This means that common actions like right-clicks need to be emulated through long-presses or other multi-finger gestures. Designing for tablets involves considering how users will interact with the screen, including navigation, selection, and input methods.

```java
// Example pseudocode for handling tablet interactions
public void handleTabletInteraction(LongPressGesture gesture) {
    if (gesture.isRightClickEmulation()) {
        // Handle right-click emulation
    } else if (gesture.isSwipe()) {
        // Handle swipe gestures
    }
}
```
x??

---

#### Mobile Phone Interaction Design
Background context: On mobile phones, the design needs to cater to one-handed use and primarily thumb control. This involves optimizing the layout for single-hand operation, focusing on navigation and interaction with a limited set of controls.

:p How does designing for mobile phones differ from other devices?
??x
Designing for mobile phones requires considering the ergonomics of single-handed use. Key interactions should be accessible using only one hand, particularly with the thumb. This includes optimizing button placements, minimizing the number of steps required to complete tasks, and ensuring that critical actions can be performed without requiring extensive navigation.

```java
// Example pseudocode for mobile phone interface layout
public void designForMobileThumbControl() {
    // Place frequently used buttons within easy reach of the thumb
    Button拇指Button = new Button();
    ThumbPositionLayout.add(m指姆Button);
    
    // Ensure all critical actions are easily accessible with one thumb swipe or tap
}
```
x??

---

#### SMS Interface for Low-Bandwidth Areas
Background context: In regions where bandwidth is limited, using SMS as an interface can be a viable alternative. This method ensures that services remain accessible even when internet connectivity is poor.

:p How does SMS-based interaction benefit in low-bandwidth areas?
??x
SMS-based interactions are beneficial in low-bandwidth areas because they do not rely on high-speed data connections. Services can provide information and functionality via text messages, making them more reliable and cost-effective for users with limited internet access. This approach ensures that critical services remain accessible even during periods of poor network conditions.

```java
// Example pseudocode for SMS interaction
public void sendSMSForService(String message) {
    // Code to send an SMS with the provided message
    send("1234567890", message);
    
    // Handle incoming SMS responses
    handleIncomingSMS();
}
```
x??

---

#### API Composition for Web and Mobile Interfaces
Background context: When building user interfaces, especially for web and mobile applications, directly interacting with APIs can provide flexibility. This approach allows the UI to query or modify data through HTTP calls, leveraging existing service-to-service communication protocols.

:p What is the main advantage of using API composition in web and mobile interfaces?
??x
The main advantage of using API composition is that it leverages existing service-to-service communication protocols (HTTP/JSON/XML) to build user interfaces dynamically. This approach provides flexibility in how data is presented, allowing for tailored responses based on the device capabilities. It enables easy integration with back-end services and supports a wide range of devices.

```java
// Example pseudocode for API composition
public void retrieveCustomerRecord(String customerId) {
    // Make an HTTP GET request to retrieve customer record
    String url = "https://api.example.com/customers/" + customerId;
    HttpClient client = new HttpClient();
    String response = client.executeGetRequest(url);
    
    // Process the response and update UI components accordingly
    processResponse(response);
}
```
x??

---

#### UI Fragment Composition for Services
Background context: Instead of having the UI make API calls to retrieve all data, services can provide pre-composed UI fragments. These fragments are then combined with other UI elements to create a complete user interface.

:p What is the advantage of using UI fragment composition?
??x
The advantage of using UI fragment composition is that it allows for modular and reusable components. Services can provide specific parts of the UI, which can be easily integrated into larger interfaces. This approach reduces redundancy in code and ensures that each service focuses on its core functionality while still contributing to the overall user experience.

```java
// Example pseudocode for UI fragment composition
public void createUserInterface() {
    // Retrieve a recommendation widget from the recommendation service
    Widget recommendationWidget = getRecommendationService().getRecommendationWidget();
    
    // Combine this with other UI elements to form the complete interface
    InterfaceBuilder.add(recommendationWidget);
    InterfaceBuilder.add(otherUIElements);
}
```
x??
---

#### Coarse-Grained UI Fragments
Background context: This approach involves serving up coarser-grained parts of a user interface (UI) from server-side applications. These fragments are assembled to form a complete application or website, often managed by specific teams responsible for certain functionalities. The aim is to allow faster development cycles and easier management of different parts of the UI.
:p What is the main advantage of serving up coarse-grained UI fragments?
??x
The primary advantage is that the same team that makes changes to the services can also manage those changes in the corresponding UI components, facilitating quicker deployment and updates. This aligns well with team ownership models, where each team handles specific functionalities.
```java
// Example of a simple server-side template for assembling fragments
public class UIAssemblyService {
    public String assembleUI(String fragment) {
        // Logic to fetch and assemble UI fragments
        return "<html><body>" + fragment + "</body></html>";
    }
}
```
x??

---
#### Ensuring Consistent User Experience
Background context: When serving up coarser-grained UI components, maintaining a consistent user experience is crucial. Users expect seamless interactions across different parts of the interface. Techniques like living style guides can be employed to share assets and enforce consistency.
:p How do living style guides help in ensuring consistent user experiences?
??x
Living style guides provide a shared repository for HTML components, CSS, and images, which ensures that developers adhere to a unified design language. By centralizing these resources, teams can maintain a cohesive look and feel across different parts of the UI, reducing inconsistencies.
```java
// Example of a living style guide concept in code form
public class LivingStyleGuide {
    public void shareAssets(String componentName) {
        // Logic to fetch and provide assets for a specific component
        System.out.println("Fetching assets for " + componentName);
    }
}
```
x??

---
#### Hybrid Approach for Native Applications
Background context: While the coarse-grained UI approach works well in web environments, native applications may require a different strategy. A hybrid approach can involve using native apps to serve up HTML components, but this has shown downsides and is not ideal for all scenarios.
:p What are some disadvantages of using a hybrid approach with native applications?
??x
Some disadvantages include performance overhead due to the need to render HTML within native applications, potential security risks, and challenges in maintaining consistent user experiences across different platforms. Additionally, the complexity of handling UI components can increase development time and maintenance efforts.
```java
// Example of a hybrid approach code snippet
public class HybridAppRenderer {
    public void renderHTML(String htmlContent) {
        // Code to render HTML content within a native application context
        System.out.println("Rendering HTML: " + htmlContent);
    }
}
```
x??

---
#### Responsive Components for Different Devices
Background context: Even with web-only UIs, different devices may require varying treatments. Using responsive design can help adapt components to fit various screen sizes and orientations, ensuring a good user experience across devices.
:p How does responsive design benefit the development of web applications?
??x
Responsive design ensures that web applications adapt gracefully to different screen sizes and resolutions, providing an optimal user experience on desktops, tablets, and mobile phones. By using CSS media queries and flexible layouts, developers can ensure components scale appropriately, maintaining usability and aesthetics.
```css
/* Example of responsive design CSS */
@media (max-width: 600px) {
    .component {
        font-size: 14px;
    }
}
```
x??

---
#### Challenges with Cross-Cutting Interactions
Background context: In some cases, the capabilities offered by a service may not fit neatly into widgets or pages. For example, dynamic recommendations that need to be triggered on search queries might require more complex interactions than what coarse-grained fragments can provide.
:p What are scenarios where coarse-grained UI fragments fall short?
??x
Coarse-grained UI fragments may struggle with cross-cutting interactions, such as real-time recommendations or dynamic content updates. For instance, when a user performs a search, immediate feedback like type-ahead suggestions requiring fresh data might not be easily accommodated within the fragment-based approach.
```java
// Example of handling complex interaction
public class SearchService {
    public void handleSearch(String query) {
        // Logic to fetch and display dynamic recommendations
        System.out.println("Handling search: " + query);
    }
}
```
x??

#### Chatty Interfaces and Backend Services
Chatty interfaces refer to user interfaces that make many requests to backend services, which can lead to inefficiencies and performance issues. To address this problem, a common solution is to use an API gateway or server-side aggregation endpoint that handles multiple backend calls and aggregates content as needed for different devices.
:p What is the purpose of using an API gateway or server-side aggregation endpoint?
??x
The purpose is to centralize requests from frontend applications to backend services. This reduces the number of direct connections, improves performance by caching results, and allows for easier management of content variations across different devices.
```java
// Example of a simplified API Gateway handling two backend calls
public class ApiGateway {
    private final BackendService service1;
    private final BackendService service2;

    public ApiResponse handleRequest(Request request) {
        // Logic to aggregate responses from multiple services
        ApiResponse response1 = service1.handleRequest(request);
        ApiResponse response2 = service2.handleRequest(request);
        
        // Combine responses as needed for different devices
        return combineResponses(response1, response2);
    }
}
```
x??

---

#### Monolithic Gateway vs. Dedicated Backends for Frontend (BFFs)
A monolithic gateway handles calls to/from multiple user interfaces and services in one giant layer, which can lead to a loss of isolation between different UI components and limit the ability to release them independently.
:p What are the disadvantages of using a single monolithic gateway?
??x
The main disadvantages include:
- Loss of isolation: Different UIs may need different logic or data, but a shared gateway might not support this effectively.
- Complexity: As more services and UIs get added, the monolithic gateway can become complex and harder to maintain.
- Dependency issues: Changes in one part of the system can impact others if they share a single gateway.

A better approach is to use dedicated backends for frontend (BFFs), where each backend handles requests specific to a particular user interface. This allows teams focusing on different UIs to manage their own server-side components independently.
```java
// Example of BFF pattern with one backend per UI
public class HomeUIBackend {
    private final UserService userService;
    
    public ApiResponse handleHomeRequest(Request request) {
        // Specific logic for handling home UI requests
        return userService.handleHomeRequest(request);
    }
}
```
x??

---

#### Business Logic and API Authentication
When using BFFs, it's important to keep business logic specific to the services themselves and avoid moving this logic into the BFF layer. Additionally, an API authentication and authorization layer can sit between the BFF and UI to manage security concerns.
:p Why should business logic stay in the services themselves instead of being moved to BFFs?
??x
Business logic should remain in the services because:
- Services are responsible for their own data and operations; moving this logic into BFFs can lead to redundancy and inconsistencies.
- It promotes better separation of concerns, making it easier to maintain and scale individual components.
- Security and business rules should be defined where they make the most sense within the application architecture.

For example, if a service handles user authentication, this logic should reside in that service rather than being replicated in multiple BFFs.
```java
// Example of keeping business logic in services
public class UserService {
    public ApiResponse authenticateUser(Request request) {
        // Authentication logic
        return new ApiResponse(true); // Simplified example
    }
}
```
x??

---

