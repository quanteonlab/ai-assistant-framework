# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 34)

**Starting Chapter:** On Your Own Terms

---

#### Fragment-Based Assembly for Websites vs. Backend-for-Frontends for Mobile Applications
Organizations can adopt different approaches to building their applications based on specific needs and use cases, rather than using a one-size-fits-all solution. For instance, they might use fragment-based assembly for websites while opting for backend-for-frontends (BFF) for mobile applications.

:p How does the choice between fragment-based assembly and BFF differ in web development compared to mobile app development?
??x
The decision is driven by considerations such as performance, maintenance, and integration complexity. Fragment-based assembly allows for modular components that can be assembled differently depending on the context (e.g., desktop vs. mobile). BFFs, on the other hand, are specialized backends tailored specifically for front-end consumption, which can enhance performance in mobile apps.

In detail, fragment-based assembly works by breaking down the web application into reusable, self-contained pieces of UI and logic that can be dynamically assembled based on user interactions or device capabilities. This approach promotes reusability and flexibility across different platforms.

BFFs, however, are designed to serve a specific client (like a mobile app) and can optimize data retrieval, transformations, and presentation tailored for the mobile environment, reducing network latency and improving performance.
??x
```java
// Example of BFF setup in Java
public class MobileService {
    public Map<String, Object> getMobileData() {
        // Logic to fetch and transform data specifically for mobile app
        return transformedData;
    }
}
```
x??

---

#### Integrating with Third-Party Software (COTS or SaaS)
Organizations often need to integrate with third-party commercial off-the-shelf (COTS) software or software as a service (SaaS). The decision on whether to build custom solutions or purchase existing products depends on the uniqueness and strategic importance of those tools.

:p Why do organizations typically prefer buying COTS or SaaS over building custom solutions?
??x
Organizations often opt for commercial off-the-shelf software because it allows them to meet their needs more efficiently without reinventing the wheel. Key reasons include:

1. **Cost-effectiveness**: Building a custom solution can be prohibitively expensive compared to acquiring and integrating existing products.
2. **Time-to-market**: Custom development takes time, whereas buying an existing product often reduces this period significantly.
3. **Risk management**: Leveraging tried-and-tested solutions reduces the risk associated with developing in-house systems.

However, organizations should carefully evaluate whether a product is truly strategic or can be customized to fit their unique requirements before making a decision.
??x
```java
// Example of integrating with a third-party SaaS API in Java
public class ExternalServiceClient {
    private String apiKey;
    
    public ExternalServiceClient(String apiKey) {
        this.apiKey = apiKey;
    }
    
    public ResponseData fetchData() {
        // Logic to fetch data from the external service using the API key
        return response;
    }
}
```
x??

---

#### Lack of Control When Integrating with COTS Products
When integrating with commercial off-the-shelf products, organizations often face limitations in terms of control over technical decisions and customization options. This can lead to challenges such as dependency on proprietary protocols or environments.

:p What are the main challenges when integrating with third-party software from a technical standpoint?
??x
The primary challenges include:

1. **Vendor Lock-In**: Being tied to a vendor’s technology stack, which may not align with internal development practices.
2. **Limited Customization Options**: Proprietary systems often restrict how and where customizations can be made.
3. **Integration Complexity**: Dealing with various communication protocols (e.g., SOAP, REST) or direct database access, leading to tightly coupled systems.
4. **Maintenance Overhead**: Managing updates, compatibility issues, and support for third-party products.

For example, integrating a CMS that does not support continuous integration can lead to significant maintenance overhead as any upgrade might break custom features.
??x
```java
// Example of handling different protocols in Java
public class IntegrationManager {
    public void integrateServices(List<Service> services) throws IOException {
        for (Service service : services) {
            if ("SOAP".equals(service.getProtocol())) {
                // Handle SOAP integration logic
            } else if ("REST".equals(service.getProtocol())) {
                // Handle REST integration logic
            }
        }
    }
}
```
x??

---

#### Customization Challenges with COTS Products
Even when a third-party tool is customizable, the cost and effort required for customization can be high. Organizations must consider whether the benefits of customizing outweigh the costs.

:p When should organizations opt against complex customization of third-party tools?
??x
Organizations should avoid complex customization when:

1. **The Tool’s Core Capabilities Aren’t Unique**: If the tool provides capabilities that are broadly applicable and not tailored to specific business needs.
2. **Cost vs. Benefit Analysis**: Customization might be more expensive than developing a custom solution from scratch, especially if it doesn’t add significant value.
3. **Maintenance Overhead**: Continuous maintenance of customizations can become a burden.

For instance, using an off-the-shelf CMS that lacks modern features like continuous integration and robust APIs might make the effort to customize it not worth the cost, especially for non-strategic applications.
??x
```java
// Example of assessing customization feasibility in Java
public class CustomizationAssessment {
    public boolean isCustomizationFeasible(Product product) {
        // Logic to evaluate whether customizing the product makes sense based on criteria such as uniqueness and strategic importance
        return feasible;
    }
}
```
x??

---

#### Integration Spaghetti: The Challenge of Multiple Communication Protocols
The integration complexity can increase significantly when multiple systems use different communication protocols, leading to what is known as "integration spaghetti."

:p How does the use of multiple communication protocols (e.g., SOAP, REST) complicate integration?
??x
Using various communication protocols complicates integration due to:

1. **Protocol Differences**: Each protocol has its own set of rules and formats, making it difficult to standardize on a single approach.
2. **Tight Coupling**: Direct access to underlying data stores or proprietary APIs can lead to tightly coupled systems that are hard to maintain and scale.
3. **Maintenance Overhead**: Managing different protocols requires additional development effort and increases the overall complexity of the system.

For example, integrating a SOAP-based service with an XML-RPC service introduces challenges in handling different message formats and ensuring compatibility.
??x
```java
// Example of handling multiple communication protocols in Java
public class ProtocolAdapter {
    public void handleRequest(String protocol, String request) throws IOException {
        if ("SOAP".equals(protocol)) {
            // Handle SOAP request logic
        } else if ("XML-RPC".equals(protocol)) {
            // Handle XML-RPC request logic
        }
    }
}
```
x??

---

#### CMS as a Service
Background context: Customizing or integrating Content Management Systems (CMS) is often necessary for enterprise organizations to display dynamic content. However, many commercial CMSes are not ideal platforms for custom coding due to their limitations in page layout and template systems.

:p How can you customize the functionality of a CMS while maintaining control over your own technology stack?
??x
By fronting the CMS with your own service that handles the website's presentation and integration with other services. This approach allows you to use the CMS for content creation and retrieval while writing custom code in your own environment.

To illustrate, let’s consider an example where a static site needs dynamic content such as customer records or product offerings:

```java
public class WebsiteFacade {
    private final CMS cms;

    public WebsiteFacade(CMS cms) {
        this.cms = cms;
    }

    public String getDynamicContent(String contentId) {
        // Retrieve and format the dynamic content from the CMS
        return "Dynamic Content: " + cms.getContent(contentId);
    }
}
```

Here, `WebsiteFacade` acts as a service that interacts with the underlying CMS to provide dynamic content to the external world. This separation allows you to manage your custom logic independently of the CMS.

x??

---
#### Multirole CRM System
Background context: Customer Relationship Management (CRM) tools can become single points of failure and tangled dependencies, especially when they are used extensively in an organization. These tools often lack flexibility and control over their API interfaces, making them challenging to integrate with other systems.

:p How can you decouple the critical functionality of a CRM system while maintaining domain integrity?
??x
By creating façade services that model your business domain concepts, thereby abstracting away the complexities of the CRM tool. This approach helps in separating concerns and providing cleaner integration points for both internal and external systems.

For instance, if a CRM tool is used to manage projects but multiple other systems require project data:

```java
public class ProjectService {
    private final CRM crm;

    public ProjectService(CRM crm) {
        this.crm = crm;
    }

    public Project getProjectById(String projectId) {
        // Retrieve and format the project details from the CRM
        return new Project(crm.getProjectDetails(projectId));
    }
}
```

Here, `ProjectService` acts as a facade that interacts with the CRM to retrieve and present project information in a structured manner. This separation ensures that changes or issues in the CRM do not affect other parts of your system.

x??

---

#### Strangler Application Pattern
The Strangler Application Pattern is a useful technique when dealing with legacy systems or third-party commercial off-the-shelf (COTS) platforms that are not fully under your control. The goal is to gradually replace the existing system without causing significant disruptions. This pattern involves capturing and intercepting calls to the old system, allowing you to decide whether these calls should be directed to existing legacy code or new code you have written.
:p What is the Strangler Application Pattern used for?
??x
The Strangler Application Pattern is used to replace legacy or third-party systems by gradually replacing them with new functionality while maintaining compatibility. It allows you to introduce new services and phase out old ones incrementally, ensuring that critical operations can continue uninterrupted during the transition.
x??

---
#### Microservices Interception
In microservice architectures, when dealing with existing legacy systems, it is often more practical to use a series of microservices instead of a single monolithic application to intercept calls. This approach allows for finer control over which parts of the legacy system are being replaced and can be managed more efficiently.
:p How do you handle interception in a microservice architecture?
??x
In a microservice architecture, interception is handled by using multiple microservices to capture and redirect original calls from the legacy system. These microservices act as intermediaries between the external requests and the old system or new code. This can be achieved through the use of proxies that forward requests to either the legacy system or the new microservices based on predefined rules.
x??

---
#### Proxy Usage in Interception
When dealing with complex interception scenarios, especially when using microservices, a proxy may be necessary to capture and redirect original calls. Proxies act as intermediaries, enabling the gradual replacement of old functionality without disrupting existing operations.
:p What role does a proxy play in interception?
??x
A proxy plays a crucial role in interception by acting as an intermediary between external requests and either the legacy system or new microservices. It captures incoming calls and decides whether to route them to the legacy system or the newer services based on predefined conditions. This allows for a smooth transition without requiring all clients to be immediately updated.
x??

---
#### Monolithic Application Decomposition
Monolithic applications often grow large and complex over time, making it difficult to manage and evolve their design. The Strangler Application Pattern can also be applied to decompose these monoliths into smaller, more manageable microservices. This approach helps in gradually refactoring the application without causing significant disruption.
:p How do you decompose a monolithic application?
??x
Decomposing a monolithic application involves using the Strangler Application Pattern to capture and redirect calls from the old system to new microservices or legacy code as needed. By carefully planning which parts of the monolith are replaced first, you can gradually break down the application into smaller, more manageable services that are easier to maintain and evolve.
x??

---
#### Legacy System Decomposition
When working with older systems, the Strangler Application Pattern can also be applied to decompose these legacy applications. This involves capturing and redirecting calls from the old system to new microservices or legacy code as appropriate, allowing for a gradual transition without disrupting existing operations.
:p How does the Strangler Application Pattern apply to legacy systems?
??x
The Strangler Application Pattern applies to legacy systems by allowing you to capture and redirect calls to the old system. This enables you to replace functionality gradually while ensuring that critical operations continue uninterrupted. It involves using microservices or proxies to manage these redirections, making it easier to phase out older components over time.
x??

---

