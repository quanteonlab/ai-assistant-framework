# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 39)

**Starting Chapter:** Conceptual Model Business Requirements

---

#### Entities and Their Attributes
Background context: In the project, you are designing a bank account management system using PostgreSQL. The conceptual phase focuses on understanding business requirements and defining the high-level structure of the database system by identifying entities and their attributes.

:p List the seven types of entities required for the bank account management system.
??x
The seven types of entities needed are accounts, customers, loans, transactions, branches, employees, and cards. Each entity has specific attributes that will be stored in the database to manage various aspects of the banking operations.

For example:
- **Accounts**: IDs, customer ID, type (savings, checking), balance, etc.
- **Customers**: IDs, names, addresses, statuses, contact details.
- **Loans**: IDs, terms (amount, duration, interest rate, payment schedule, start and end dates), type (mortgage or personal loan).
- **Transactions**: IDs, account ID, employee ID (if initiated by an employee), timestamps, amounts.
- **Branches**: IDs, names, addresses, phone numbers.
- **Employees**: IDs, names, job titles, branch ID.
- **Cards**: Cardholder ID, associated accounts, card numbers, issuance and expiration dates.

x??

---

#### Relationships Between Entities
Background context: The relationships between entities are crucial for maintaining data integrity and facilitating efficient data retrieval. Understanding these relationships helps in defining how different pieces of information interlink within the database.

:p Describe the key relationships that need to be established.
??x
The following relationships must be established:
- Each account should be linked to a customer.
- Each loan should be linked to a customer.
- Each loan payment should be linked to a transaction and an account.
- Each employee should be affiliated with a branch.
- Each card should be associated with both a customer and an account.

These relationships ensure that data is correctly interconnected, supporting efficient management and retrieval of information. For example:
- A single customer can have multiple accounts.
- A loan can only belong to one customer but may involve transactions across different accounts.
- An employee works in one branch but may process transactions from any account or branch.

x??

---

#### Conceptual Model
Background context: The conceptual model focuses on understanding business requirements and defining the high-level structure of the database system. This involves identifying entities, their attributes, and relationships to form a clear picture of the data architecture.

:p What are the main components of the conceptual model for this project?
??x
The main components of the conceptual model for this project include:
- **Entities**: Accounts, customers, loans, transactions, branches, employees, and cards.
- **Attributes**: Detailed characteristics that describe each entity (e.g., account type, loan terms, customer contact information).
- **Relationships**: Interconnections between entities to ensure data integrity.

For instance, an `Account` entity has attributes such as ID, type, balance, and associated customer. The relationship between `Customer` and `Account` ensures that a single customer can have multiple accounts.

x??

---

#### Physical Data Modeling
Background context: After defining the conceptual model, the next step is to implement this design in a physical database schema. This involves mapping entities to tables and attributes to columns in a relational database management system (RDBMS).

:p How would you map an entity like `Customer` to a table in PostgreSQL?
??x
To map the `Customer` entity to a PostgreSQL table, you would define a table structure that includes all relevant attributes. Here’s an example of how this might look:

```sql
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    address VARCHAR(100),
    status ENUM('active', 'inactive') DEFAULT 'active',
    email VARCHAR(50),
    phone_number VARCHAR(20)
);
```

- `customer_id`: A unique identifier for each customer, generated automatically.
- `first_name` and `last_name`: Names of the customer.
- `address`: The address of the customer.
- `status`: Current status (active or inactive).
- `email` and `phone_number`: Contact information.

This table structure ensures that all necessary attributes are captured in a structured format, facilitating efficient data storage and retrieval.

x??

---

#### Multitenant Database Design
Background context: While not explicitly mentioned, designing for multitenancy is an important consideration. In this context, the database design might need to accommodate multiple banks or branches with similar structures but distinct sets of data.

:p How would you ensure that different branches can have their own customer and account data in PostgreSQL?
??x
To ensure that each branch has its own set of customers and accounts, you can use a multitenant approach. One common method is to include the `branch_id` as a foreign key in related tables. For example:

```sql
CREATE TABLE branches (
    branch_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address VARCHAR(100),
    phone_number VARCHAR(20)
);

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    branch_id INTEGER REFERENCES branches(branch_id) ON DELETE CASCADE,
    -- Other attributes
);
```

By adding `branch_id` to the `customers` table, you can associate each customer with a specific branch. This ensures that data is segregated by branch while maintaining relationships.

x??

---

#### Multiclient Support
Background context: For a more comprehensive design, supporting multiple clients (banks) might require additional tables or structures. Each client would have its own set of branches and associated entities.

:p How can you support multiple banks in the database schema?
??x
To support multiple banks, you can introduce an `organization` entity to manage different financial institutions. This would involve creating a hierarchy where each bank has its own branches, customers, and other related data.

Here’s how you might structure this:

```sql
CREATE TABLE organizations (
    organization_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE branches (
    branch_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address VARCHAR(100),
    phone_number VARCHAR(20),
    organization_id INTEGER REFERENCES organizations(organization_id) ON DELETE CASCADE
);

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    branch_id INTEGER REFERENCES branches(branch_id) ON DELETE CASCADE,
    -- Other attributes
);
```

- `organizations`: Manages different banks or financial institutions.
- `branches`: Belongs to an organization and contains data specific to each branch.

This approach ensures that data is organized by both bank (organization) and branch, providing a flexible structure for supporting multiple clients.

x??

---

#### Business Constraints for Database Implementation
Background context: The business team has outlined specific constraints to ensure high data quality, consistency, and integrity within a database implementation. These constraints cover various aspects such as minimum balance limitations, unique IDs, data redundancy minimization, null value restrictions, and uniqueness of certain fields.

:p What are the key business constraints mentioned in the text for database implementation?
??x
The key constraints include:
- Account balances must never go below a customer-specific minimum amount.
- All entity records (e.g., accounts, loans, transactions) must be identified with unique IDs.
- Data redundancy should be minimized.
- Null values are not permitted for certain fields.
- Specific fields, such as email addresses, must be unique across records.

These constraints help ensure data quality, consistency, and integrity in the database system. 
x??

---

#### Logical Model: Entity Relationship Diagram (ERD)
Background context: The logical model involves selecting a suitable data storage model after agreement on the conceptual model from stakeholders. The financial data engineering team has chosen the relational model due to its effectiveness in organizing entities into tables and ensuring data integrity through constraints. An ERD is used to visualize this structure.

:p What is an Entity Relationship Diagram (ERD) and why is it important for a logical model?
??x
An Entity Relationship Diagram (ERD) is a visual representation that models the database structure by illustrating entities, their attributes, and relationships among these entities. It helps in organizing data into tables, ensuring normalization, and maintaining data integrity.

The ERD is crucial because:
- It provides a clear visualization of how different entities are related.
- It supports the implementation of relational databases.
- It ensures proper normalization to minimize redundancy and improve data quality.

Here’s an example of how you might represent an account entity in an ERD:
```plaintext
Account (ID: Integer, CustomerName: String, Balance: Decimal)
```

This shows that each `Account` has a unique ID, customer name, and balance. 
x??

---

#### Relational Model Selection
Background context: The relational model was chosen because it effectively organizes various entities into distinct tables and ensures data integrity through database constraints. It supports the implementation of normalized data structures.

:p Why did the financial data engineering team choose the relational model for this system?
??x
The relational model was chosen due to its effectiveness in:
- Organizing data into distinct tables.
- Ensuring data integrity through constraints.
- Supporting a normalized data structure which minimizes redundancy and improves data quality.

The relational model is suitable because it aligns well with the business requirements, such as ensuring unique IDs for entities and maintaining minimum balance limits. 
x??

---

#### ERD Construction
Background context: An ERD is constructed using information from the conceptual phase and provides a visual representation of database structure. Various tools like Lucidchart, Creately, DBDiagram, QuickDBD, etc., can be used to create these diagrams.

:p What are some key elements of an Entity Relationship Diagram (ERD)?
??x
Key elements of an ERD include:
- **Entities**: Represented as tables or objects.
- **Attributes**: Fields or properties associated with entities.
- **Relationships**: Connections between entities, showing how they interact.

For example, in a bank account management system, you might have the following relationships:
```plaintext
Account (ID, CustomerName, Balance) - one-to-many relationship with Transactions
```

This indicates that each `Account` can have multiple transactions. 
x??

---

#### Normalization and Data Integrity
Background context: The relational model supports normalization to minimize data redundancy and ensure data integrity. This is crucial for maintaining high-quality data in the database.

:p How does normalization help in achieving better data quality?
??x
Normalization helps achieve better data quality by:
- Reducing or eliminating redundant data.
- Ensuring that each piece of data has only one place where it can be updated, thus reducing inconsistencies.

For instance, consider a scenario where an account balance is stored directly on the account table and also in transactions. Normalization would ensure this data is stored once in the appropriate related tables (like `Transactions`) to avoid redundancy.

Normalization levels:
1. 1NF: Ensures each column contains atomic values.
2. 2NF: Removes non-key columns that are not dependent on the primary key.
3. 3NF: Eliminates dependencies between non-key attributes and composite keys.

By following these steps, data quality is improved as updates can be made in a single place, ensuring consistency across the database. 
x??

---

#### PostgreSQL Database Setup and Testing
Background context: This section explains how to set up a PostgreSQL database environment for local testing using Docker containers. The goal is to create an isolated, reproducible setup that allows users to interact with the PostgreSQL database through pgAdmin.

:p How do you start the PostgreSQL database and pgAdmin UI locally?
??x
You start by navigating to the specified path in your terminal and executing the `docker compose up -d` command. This command runs two Docker containers: one for the PostgreSQL database instance and another for pgAdmin, which provides a user-friendly interface to interact with PostgreSQL.

```bash
# Navigate to the project directory
cd {path/to/book/repo}/FinancialDataEngineering/book/chapter_12/project_1

# Run the docker compose command
docker compose up -d
```

Once the containers are up and running, open your browser and visit `http://localhost:8080` to access the pgAdmin UI. Log in using the dummy credentials specified in the `.env` file.
x??

---

#### Creating Tables with DDL Statements
Background context: Data Definition Language (DDL) is used to define the structure of database objects, including creating and altering tables. PostgreSQL uses specific syntax for DDL commands such as `CREATE TABLE`.

:p What is a DDL statement commonly used for?
??x
A Data Definition Language (DDL) statement is commonly used to create or alter database objects in SQL. In this context, we use it to define the structure of our tables within the bank account management system.

For example, here’s how you would create an `Account` table:

```sql
-- PostgreSQL CREATE TABLE Account (
id SERIAL PRIMARY KEY,
customer_id INT NOT NULL REFERENCES Customer (id),
branch_id INT NOT NULL REFERENCES Branch(id),
type INT NOT NULL REFERENCES AccountType (id),
currency VARCHAR NOT NULL,
number VARCHAR NOT NULL UNIQUE,
balance DECIMAL(10,2) NOT NULL,
minimum_balance DECIMAL(10,2) NOT NULL DEFAULT 0,
date_opened DATE NOT NULL,
date_closed DATE,
status INT NOT NULL REFERENCES AccountStatusType (id),
CHECK (balance >= minimum_balance)
);
```

This statement creates a table with an auto-incrementing primary key (`id`), foreign keys to related tables like `Customer`, and constraints such as ensuring the balance is always greater than or equal to the minimum balance.
x??

---

#### Insert Operations Using DML
Background context: Data Manipulation Language (DML) statements are used for adding, updating, and deleting records in a database. The most common operation involves inserting new data into tables.

:p How do you insert a new customer record?
??x
To insert a new customer record, you use the `INSERT INTO` statement with all required fields:

```sql
-- PostgreSQL INSERT INTO Customer (
customer_type_id,
name,
country,
city,
street,
phone_number,
email,
status
) VALUES (
1, 'John Smith', 'US', 'New York', '123 Main St', '123-456-7890', 'john@example.com', 1
);
```

This command adds a new row to the `Customer` table with the specified values. Ensure all required fields are included, and refer to the foreign keys correctly.
x??

---

#### Using Transactions for Concurrency Control
Background context: In PostgreSQL, transactions are used to ensure that database operations are performed atomically, meaning they either complete entirely or not at all. Explicit transactions can be started using `BEGIN;` and ended with `COMMIT;`.

:p How do you record a payment transaction in the account table while updating the balance?
??x
To record a payment transaction and update the account balance within an explicit transaction, use the following SQL commands:

```sql
-- PostgreSQL BEGIN;
INSERT INTO transaction (
account_id,
type,
currency,
date,
amount
) VALUES (1, 4, 'USD', '2022-01-01 08:00:00', -40.00);
UPDATE account SET balance = balance - 40.00 WHERE id = 1;
COMMIT;
```

This transaction ensures that both the `INSERT` and `UPDATE` operations are executed atomically, maintaining data integrity by preventing partial updates.
x??

---

