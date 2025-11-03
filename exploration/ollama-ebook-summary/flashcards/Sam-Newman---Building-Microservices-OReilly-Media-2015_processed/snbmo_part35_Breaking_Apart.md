# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 35)

**Starting Chapter:** Breaking Apart MusicCorp

---

---
#### Importance of Cohesion and Coupling in Services
Background context explaining that monolithic services often lack cohesion and tight coupling, making them hard to maintain and evolve. Michael Feathers’ concept of a seam is introduced as a way to address this issue by isolating parts of code for easier modification.
:p What is the problem with monoliths in terms of cohesion and coupling?
??x
The main issues with monolithic services are that they tend to be poorly cohesive (mixed with unrelated code) and tightly coupled, where changes in one part can impact many other parts. This makes it difficult to make changes without significant ripple effects.
```java
// Example of a monolithic service function that mixes different functionalities
public class MonolithicService {
    public void processOrderAndRecommend() {
        // Code for processing order
        // Code for generating recommendations
    }
}
```
x??

---
#### Concept of Seams in Software Development
Background context explaining Michael Feathers’ definition of a seam, which is a portion of code that can be isolated and modified without impacting the rest of the system.
:p What is a seam in software development?
??x
A seam is a part of the codebase that can be treated independently. It allows developers to modify or replace functionality with minimal impact on other parts of the system, enabling easier maintenance and evolution.
```java
// Example of refactoring a method into a seam
public class Service {
    private void processCatalogData() { ... }
    
    public void processAllData() {
        // Logic that now calls processCatalogData as a separate entity
    }
}
```
x??

---
#### Identifying Bounded Contexts for Seams
Background context explaining how bounded contexts are crucial in identifying seams, representing cohesive and loosely coupled boundaries within an organization.
:p How do bounded contexts help identify seams?
??x
Bounded contexts represent specific areas of functionality that can be isolated. They help us identify seams because they define clear boundaries where changes in one area won't necessarily affect another. This makes it easier to refactor the codebase into smaller, more manageable services.
```java
// Example of identifying a bounded context (Catalog)
public class CatalogService {
    private void manageMetadata() { ... }
    
    public void serveItems() {
        // Logic for managing metadata and serving items
    }
}
```
x??

---
#### Breaking Apart the Monolithic Backend of MusicCorp
Background context explaining how to identify and separate bounded contexts in a large monolithic backend service. The example uses MusicCorp’s services as a case study.
:p How should we approach breaking apart a large monolithic backend service?
??x
Start by identifying high-level bounded contexts, such as Catalog, Finance Reporting, Warehouse Dispatching, and Recommendation. Then create packages for each context and move the relevant code into these new packages to isolate them from one another.
```java
// Example of creating package structures in Java
package com.musiccorp.catalog;

public class CatalogService {
    private void manageMetadata() { ... }
}

package com.musiccorp.finance;

public class FinanceService {
    private void handlePayments() { ... }
}
```
x??

---
#### Analyzing Dependencies and Refactoring Codebases
Background context explaining the importance of analyzing dependencies between packages to ensure accurate refactoring. Tools like Structure 101 are mentioned as aids in this process.
:p How can we use tools to analyze dependencies between packages?
??x
Tools like Structure 101 allow us to visualize the dependencies graphically, making it easier to understand and refactor our codebase. By analyzing these visualizations, we can identify misplaced or unnecessary dependencies, ensuring that each package operates independently as intended.
```java
// Example of using a tool to visualize dependencies
Structure101 tool = new Structure101();
tool.analyzeProjects("com.musiccorp.*");
tool.showDependencies();
```
x??

---

#### Pace of Change
Background context explaining the concept. If a part of the monolith is changing more frequently, separating it into its own service can help manage those changes more efficiently and independently.
:p Which driver could suggest that we start by splitting out the code related to inventory management?
??x
It suggests starting with the "Pace of Change" driver because if there are frequent updates or significant modifications planned for a specific part of the system, like managing inventory, separating this area can allow for more agile and independent development cycles. This separation helps in making changes faster without impacting other parts of the monolith.
x??

---

#### Team Structure
Background context explaining the concept. When teams are geographically dispersed or have specialized roles, it might make sense to separate their respective code bases into distinct services so that each team can take ownership and manage their specific areas more effectively.
:p How does separating the code for a particular region's team (e.g., Hawaii) support better management of change?
??x
Separating the code for the Hawaii team allows them to have full ownership over their specific area, enabling faster and more focused development cycles. This approach can reduce dependency on other teams and streamline communication and coordination within the team.
x??

---

#### Security
Background context explaining the concept. If there is a need to enforce stricter security measures in certain parts of an application, separating those areas into distinct services can enhance protection by isolating sensitive information or operations.
:p Why might it be advantageous to split out finance-related code for improved security?
??x
Splitting out the finance-related code as a separate service allows for enhanced security measures such as more robust monitoring, stronger data encryption during transit and at rest. This isolation helps in reducing the attack surface and ensuring that sensitive financial information is protected more effectively.
x??

---

#### Technology
Background context explaining the concept. Introducing new technologies or algorithms into specific parts of an application can be facilitated by separating those areas into independent services, which allows for testing and deployment without affecting other components.
:p How could splitting out recommendation code benefit the system?
??x
Splitting out the recommendation code into a separate service enables the team to experiment with new algorithms using tools like Clojure. This separation facilitates the introduction of alternative implementations for testing and evaluation, potentially leading to improved user experiences.
x??

---

#### Tangled Dependencies
Background context explaining the concept. Understanding and managing dependencies between different parts of an application is crucial when deciding which parts to separate into microservices. The goal is to minimize interdependencies to ensure smoother refactoring and development processes.
:p What approach can help in identifying the least entangled part of a monolith for separation?
??x
Using tools like package modeling or dependency graph visualization (e.g., SchemaSpy) can help identify the parts of the code that are least dependent on others. By focusing on these less entangled areas, it becomes easier to separate them into distinct services with minimal disruption.
x??

---

#### The Database
Background context explaining the concept. Databases often act as a central hub for data access and storage in monolithic applications. Separating database access code can provide better isolation and management of different parts of an application’s data model.
:p How does splitting out repository layers help in understanding database dependencies?
??x
Splitting out repository layers allows for clearer separation between the business logic and the database access code. By organizing these layers based on bounded contexts, it becomes easier to identify which parts of the database are used by specific sections of the application. Tools like SchemaSpy can then be used to visualize relationships between tables, helping to understand and manage dependencies.
x??

---

Each flashcard is designed to help you grasp key concepts related to splitting a monolith into microservices, with clear explanations and examples where appropriate.

#### Breaking Foreign Key Relationships
Background context: The example discusses a scenario where the finance code uses a foreign key relationship to reference data from the catalog code. This leads to concerns about data integrity and performance when the catalog and finance services are separated into their own rights.

:p What is the problem introduced by using a foreign key relationship between the ledger table (finance) and the line item table (catalog)?
??x
The problem arises because the finance code directly accesses the line item table, which should be part of the catalog service. This coupling introduces tight integration that needs to be managed when services are separated.

Example:
```java
// Pseudocode showing direct access in finance package
public class FinanceService {
    public void generateReport(int ledgerId) {
        LineItem lineItem = lineItemRepository.findById(ledger.getLineItemId());
        // Use lineItem data for report generation
    }
}
```
x??

---

#### Exposing Data via API Call
Background context: To decouple the finance code from the catalog code, an API call is introduced instead of direct database access. This change allows the finance service to request necessary information through a well-defined interface.

:p How can we refactor the finance code to avoid direct database access and instead use an API call?
??x
We should create an API in the catalog package that exposes the required data. The finance code will then make an API call to retrieve this information.

Example:
```java
// Pseudocode for API call in FinanceService
public class FinanceService {
    public void generateReport(int ledgerId) {
        String catalogData = catalogClient.getCatalogData(ledger.getLineItemId());
        // Use catalogData for report generation
    }
}
```
x??

---

#### Performance Considerations
Background context: The introduction of an API call might increase the number of database calls needed to generate reports. This is a trade-off between data integrity and performance.

:p Why do we need to consider performance when making changes in database relationships?
??x
We need to consider performance because multiple database calls can impact system speed, especially under load. However, sometimes sacrificing some performance for better separation of concerns or other benefits (like easier maintenance) is acceptable if the change does not significantly degrade overall performance.

Example:
```java
// Pseudocode comparing current and proposed systems
public class CurrentSystem {
    // Single database call to generate report
}

public class NewSystem {
    // Two API calls to generate report
}
```
x??

---

#### Removal of Foreign Key Constraint
Background context: With the introduction of services, foreign key constraints are no longer appropriate. Instead, each service must manage its own data consistency.

:p What happens when a foreign key constraint is removed from the database?
??x
When a foreign key constraint is removed, data integrity becomes managed by the application logic rather than enforced at the database level. This means that the services need to implement their own mechanisms for ensuring data consistency.

Example:
```java
// Pseudocode showing service-level validation
public class OrderService {
    public void processOrder(Order order) {
        if (catalogService.checkCatalogItemExists(order.getCatalogItemId())) {
            // Proceed with processing
        } else {
            throw new InvalidOrderException("Invalid catalog item ID");
        }
    }
}
```
x??

---

#### Managing Data Consistency Across Services
Background context: When foreign key relationships are removed, ensuring data consistency across services becomes a manual process. This may require implementing custom checks or triggers.

:p How can we ensure data consistency when removing foreign keys between services?
??x
We need to implement custom checks and possibly trigger actions to clean up related data in the event of inconsistencies. For example, if an order refers to a non-existent catalog item, the system should either prevent such orders or handle them gracefully by updating or deleting invalid references.

Example:
```java
// Pseudocode for custom consistency check
public class OrderService {
    public void processOrder(Order order) {
        try {
            if (catalogService.checkCatalogItemExists(order.getCatalogItemId())) {
                // Proceed with processing
            } else {
                throw new InvalidOrderException("Invalid catalog item ID");
            }
        } catch (InvalidOrderException e) {
            orderRepository.delete(order);  // Clean up invalid order
        }
    }
}
```
x??

---

