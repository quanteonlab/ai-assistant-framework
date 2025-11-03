# Flashcards: Sam-Newman---Building-Microservices-OReilly-Media-2015_processed (Part 36)

**Starting Chapter:** Example Shared Static Data

---

#### Managing Shared Static Data in Databases

Background context: The passage discusses various approaches to handling shared static data, such as country codes, within a system that involves multiple services. The primary challenge is ensuring consistency across these services when updating this static data.

:p How can we manage shared static data like country codes between different services?

??x
One approach is to store the shared static data in a database table and have each service read from its own copy of this table. However, this method can introduce challenges in maintaining consistency if updates are not synchronized across all services.

```java
// Pseudocode for fetching country codes from a database
public List<String> fetchCountryCodes() {
    Connection conn = null;
    Statement stmt = null;
    ResultSet rs = null;

    try {
        conn = DatabaseConnection.getConnection();
        stmt = conn.createStatement();
        rs = stmt.executeQuery("SELECT code FROM countries");
        while (rs.next()) {
            countryCodes.add(rs.getString("code"));
        }
    } catch (SQLException e) {
        // Handle exception
    } finally {
        closeResources(conn, stmt, rs);
    }

    return countryCodes;
}

private void closeResources(Connection conn, Statement stmt, ResultSet rs) {
    try {
        if (rs != null) rs.close();
        if (stmt != null) stmt.close();
        if (conn != null) conn.close();
    } catch (SQLException e) {
        // Handle exception
    }
}
```

x??

---

#### Duplicating Shared Static Data Tables

Background context: One option mentioned in the passage is duplicating the shared static data tables for each package or service. This approach can simplify updates but introduces a consistency challenge.

:p How does duplicating the table for each package or service help manage shared static data?

??x
Duplicating the table ensures that each service has its own copy of the shared static data, which can be easier to maintain and update independently. However, this can lead to inconsistencies if not all services are updated simultaneously.

```java
// Pseudocode for duplicating a table in SQL
public void duplicateTable(String originalTableName, String newTableName) {
    Connection conn = null;
    Statement stmt = null;

    try {
        conn = DatabaseConnection.getConnection();
        stmt = conn.createStatement();

        // Create a new table with the same schema as the original
        String createStatement = "CREATE TABLE " + newTableName + " AS SELECT * FROM " + originalTableName;
        stmt.executeUpdate(createStatement);
        
        System.out.println("Table duplicated successfully.");
    } catch (SQLException e) {
        e.printStackTrace();
    } finally {
        closeResources(conn, stmt);
    }
}

private void closeResources(Connection conn, Statement stmt) {
    try {
        if (stmt != null) stmt.close();
        if (conn != null) conn.close();
    } catch (SQLException e) {
        // Handle exception
    }
}
```

x??

---

#### Treating Shared Static Data as Code

Background context: Another option is to treat the shared static data like code, stored in configuration files or directly in the service's codebase. This method can simplify updates and maintain consistency more easily.

:p How does treating shared static data as code help manage it?

??x
Treating shared static data as code (e.g., using properties files or enums) allows for easier updates since configuration files are typically simpler to change than database tables. This approach also helps in maintaining a single source of truth within the application's codebase.

```java
// Pseudocode for reading country codes from a properties file
public List<String> fetchCountryCodes() {
    Properties prop = new Properties();
    InputStream input = null;

    try {
        // load a properties file
        input = new FileInputStream("config/country_codes.properties");
        prop.load(input);

        Enumeration<Object> keys = prop.propertyNames();
        while (keys.hasMoreElements()) {
            String key = (String) keys.nextElement();
            countryCodes.add(key);
        }
    } catch (IOException ex) {
        ex.printStackTrace();
    } finally {
        if (input != null) {
            try {
                input.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    return countryCodes;
}
```

x??

---

#### Pushing Static Data into a Separate Service

Background context: In some cases, the volume and complexity of static reference data may warrant creating a dedicated service to manage this data. This approach ensures consistency but can be overkill for simpler scenarios.

:p When should we consider pushing static data into a separate service?

??x
This approach is appropriate when the static reference data is highly complex or frequently updated, requiring specialized management and validation logic. However, it may be overkill if the data is simple (like country codes) and infrequently changes.

```java
// Pseudocode for an API to get country codes from a dedicated service
public class StaticDataServiceClient {
    private static final String SERVICE_URL = "http://static-data-service/country-codes";

    public List<String> fetchCountryCodes() throws IOException, InterruptedException {
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(SERVICE_URL))
                .build();

        HttpResponse<String> response = client.send(request, BodyHandlers.ofString());
        
        return Arrays.asList(response.body().split(","));
    }
}
```

x??

---

#### Shared Mutable Data Concept
Background context: In systems design, shared mutable data can lead to complex interdependencies and make it difficult to maintain and scale individual components. This is often seen when different parts of an application modify a common database table or object.

:p What are the challenges associated with shared mutable data in a multi-component system?
??x
The challenges include increased complexity due to concurrent modifications, race conditions, and inconsistent states across different modules. It can also lead to tightly coupled code, making it harder to make changes without affecting other parts of the system.
```java
public void updateCustomerPayment(int customerId, double amount) {
    // Code to update customer payment in a shared table
}
```
x??

---
#### Domain Concept Absence
Background context: The text highlights that sometimes domain concepts are not explicitly modeled in code but are represented implicitly in databases. This can lead to issues when trying to understand and separate the responsibilities of different components.

:p Why is recognizing domain concepts important in systems design?
??x
Recognizing domain concepts helps in defining clear boundaries between different parts of a system, making it easier to implement changes without affecting other components. It also aids in maintaining a clean architecture by ensuring that each component has well-defined responsibilities.
```java
public class CustomerService {
    // Methods to interact with customer data
}
```
x??

---
#### Bounded Context Introduction
Background context: A bounded context is a way of defining the scope and responsibility of a particular part of an application. It helps in creating clear, isolated units that can be developed independently while still maintaining coherence within their specific domain.

:p What does recognizing the bounded context of a customer mean?
??x
Recognizing the bounded context of a customer means identifying the specific domain or area where the customer data is relevant and defining the rules, responsibilities, and constraints associated with it. This helps in isolating concerns and ensuring that changes to one part do not inadvertently affect another.
```java
public class FinanceService {
    // Methods for managing payments
}
```
x??

---
#### Package Creation for Customer Service
Background context: To address issues related to shared mutable data, a new package is created to encapsulate the customer-related logic. This separation helps in isolating changes and improving modularity.

:p What is the benefit of creating a dedicated `Customer` package?
??x
The benefit is that it provides a clear boundary for managing customer data, reducing the risk of unintended side effects when making changes. It also improves code organization and makes it easier to understand the flow of information within the system.
```java
package com.example.customer;
public class Customer {
    // Attributes and methods related to customers
}
```
x??

---

#### Identifying Database Concerns Through Bounded Contexts
Background context explaining how database concerns can be identified through bounded contexts. This involves recognizing distinct areas of responsibility within an application that should correspond to separate database schemas.
:p How can you identify different database concerns in a system?
??x
By identifying and grouping the application code around bounded contexts, you can recognize different areas of responsibility that need to be managed separately. For instance, if your application handles both customer orders and warehouse inventory management, these two functionalities might have distinct data requirements and should be stored separately.
x??

---

#### Splitting Shared Tables Across Contexts
Background context explaining the necessity of splitting shared tables into separate contexts when dealing with different bounded contexts that require unique database schemas. This is a step towards separating concerns within a monolithic schema to ensure each context has its dedicated data storage.
:p What happens if you keep shared tables across different contexts?
??x
Keeping shared tables in a single schema can lead to conflating concerns, where multiple functionalities are mixed together, making it harder to manage and evolve the database structure. For example, maintaining both customer orders and warehouse inventory in one table might make sense initially but becomes problematic as the application scales and more complex business rules are introduced.
x??

---

#### Staging Service Separation
Background context explaining how separation of services should be staged before fully implementing it. This involves temporarily separating schemas while keeping the application code together to test changes without disrupting other parts of the system.
:p What is the advantage of staging a service separation?
??x
Staging a service separation allows you to test and validate schema changes in isolation from application logic, reducing risk and allowing for easier rollback if issues arise. By separating schemas but keeping application code together, you can ensure that any issues related to database changes do not impact existing functionality.
x??

---

#### Transactional Boundaries and Schema Splitting
Background context explaining the importance of maintaining transactional boundaries when splitting schemas. This involves ensuring that actions affecting multiple tables within a single operation are performed atomically to maintain data consistency.
:p What happens when you split schemas but operations span across them?
??x
When splitting schemas, if transactions need to span across different databases, it can lead to issues with transactional integrity and increased complexity in database interactions. For example, creating an order and updating the warehouse picking table might work together within a single transaction in a monolithic schema, but this atomicity is lost when each table resides in its own schema.
x??

---

#### Example of Transaction Spanning Boundaries
Background context explaining how transactions can span across boundaries after splitting schemas. This involves demonstrating a scenario where an operation that updates multiple tables needs to be performed as part of a single transaction.
:p How does the MusicCorp example illustrate transaction spanning?
??x
In the MusicCorp example, creating an order and updating the warehouse picking table should ideally happen atomically. However, when these are split into separate schemas, they need to be updated independently, potentially leading to inconsistent states if one update fails while the other succeeds.
```java
// Pseudocode for a potential solution
try {
    // Insert order record in customer schema
    Order order = new Order();
    orderRepository.save(order);

    // Insert picking record in warehouse schema
    PickingRecord record = new PickingRecord();
    pickingRepository.save(record);
} catch (Exception e) {
    // Rollback both operations if any fail
    orderRepository.rollbackSave(order.getId());
    pickingRepository.rollbackSave(record.getId());
}
```
x??

