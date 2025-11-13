# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 7)

**Starting Chapter:** The Foundation Datalog

---

#### CODASYL and Graph Databases Comparison
CODASYL’s network model and graph databases both handle many-to-many relationships but differ significantly. In CODASYL, a database schema dictated which record types could reference each other, whereas in graph databases, any vertex can connect to any other vertex, providing greater flexibility.
:p How do CODASYL and graph databases handle the relationship between vertices or records?
??x
In CODASYL, relationships were strictly defined by the schema, meaning that only specific record types could reference one another. In contrast, graph databases allow any node (vertex) to be connected to any other node without such restrictions.
x??

---

#### Access Paths and Vertex IDs in Graph Databases
Graph databases provide multiple ways to access a vertex beyond just traversal paths as seen in CODASYL. Vertices can be accessed by their unique ID or through indexes based on specific values.
:p How do graph databases offer more flexibility compared to CODASYL when accessing data?
??x
Graph databases offer greater flexibility because vertices can be directly referenced using their unique IDs, and they also support indexing which allows for faster querying of vertices based on specific property values. In contrast, CODASYL required traversing predefined access paths to reach a particular record.
x??

---

#### Record Ordering in CODASYL vs Graph Databases
In CODASYL, child records were an ordered set, meaning the database had to maintain this order and applications needed to handle the ordering when inserting new records. In graph databases, vertices and edges are unordered, simplifying insertion and query processes as sorting is only done during query execution.
:p How does the handling of record ordering differ between CODASYL and graph databases?
??x
In CODASYL, child records were ordered within their parent records, which meant that both the database schema and applications had to manage this order. In contrast, graph databases do not enforce any inherent ordering on vertices or edges, making insertion simpler and allowing for more flexible query execution.
x??

---

#### Query Languages in Graph Databases
Graph databases support high-level declarative languages like Cypher and SPARQL, but fundamentally build upon the older language Datalog, which is a subset of Prolog. Datalog uses rules to define new predicates based on existing data or other rules.
:p What query languages do graph databases support?
??x
Graph databases support multiple query languages such as Cypher, SPARQL, and Datalog (which is derived from Prolog). These languages allow for expressing complex queries in a declarative manner.
x??

---

#### Datalog Syntax Example
Datalog uses a syntax similar to Prolog but is based on the triple-store model. It defines data and rules using patterns like `predicate(arg1, arg2)`. For example, relationships can be defined as `within(Location, Via)` or properties as `name(Location, Name)`.
:p How does Datalog represent triples and define rules?
??x
Datalog represents triples by writing them in the form of predicates: `predicate(arg1, arg2)`. Rules are defined using patterns like `within(Location, Via)` to establish relationships between entities. For example:
```prolog
name(namerica, 'North America').
type(namerica, continent).
```
x??

---

#### Datalog Rule Application Process
Datalog applies rules by matching the right-hand side of the rule with existing data in the database. When a match is found, the left-hand side of the rule is added to the database. For instance, given `within_recursive(Location, Name) :- name(Location, Name)`, if `name(namerica, 'North America')` exists, it will generate `within_recursive(namerica, 'North America')`.
:p How does Datalog apply rules?
??x
Datalog applies rules by checking the right-hand side of a rule against existing data. If a match is found, the left-hand side is added to the database. For example:
```prolog
rule1: within_recursive(Location, Name) :- name(Location, Name).
```
Given `name(namerica, 'North America')` exists, this rule will generate:
```prolog
within_recursive(namerica, 'North America').
```
x??

---

#### Complex Queries in Datalog
Complex queries in Datalog are built up incrementally through rules. For example, to find if someone migrated from one place to another, a series of rules can be defined and applied sequentially.
:p How does Datalog build complex queries?
??x
Datalog builds complex queries by defining rules that refer to other rules or data in the database. For instance:
```prolog
rule1: within_recursive(Location, Name) :- name(Location, Name).
rule2: within_recursive(Location, Name) :- within(Location, Via), 
                                           within_recursive(Via, Name).

rule3: migrated(Name, BornIn, LivingIn) :-
       name(Person, Name),
       born_in(Person, BornLoc),
       within_recursive(BornLoc, BornIn),
       lives_in(Person, LivingLoc),
       within_recursive(LivingLoc, LivingIn).
```
These rules can be applied step by step to derive new information from the existing data.
x??

#### Data Models Overview
Data models are essential for organizing and storing data efficiently. Historically, hierarchical models were used but didn't handle many-to-many relationships well. Relational databases addressed this issue with tables and SQL, while more recent NoSQL databases offer alternatives like document and graph databases.

:p What is the main limitation of the hierarchical model in handling relationships?
??x
The hierarchical model struggled to represent many-to-many relationships effectively.
x??

---

#### Hierarchical Model Limitation
As mentioned, the hierarchical model was good for simple tree-like structures but didn’t handle complex relationships well. For example, a user could be friends with multiple users, and each of those users could have multiple friends.

:p How does the hierarchical model fail to represent many-to-many relationships?
??x
In a hierarchical model, data is structured in a tree or linear hierarchy where one node can only relate directly to its parent or child nodes. This structure makes it difficult to establish bidirectional or cross-node relationships.
x??

---

#### Relational Model Introduction
The relational model introduced tables and SQL (Structured Query Language) to better handle many-to-many relationships through the use of primary keys, foreign keys, and joins.

:p What is a key advantage of the relational model over hierarchical models?
??x
The relational model allows for more complex data relationships via primary and foreign keys, enabling efficient management of many-to-many relationships.
x??

---

#### NoSQL Databases Overview
NoSQL databases diverged into document and graph databases. Document databases store self-contained documents, while graph databases handle highly interconnected data.

:p What are the main differences between document and graph databases?
??x
Document databases store self-contained documents with a flexible schema, whereas graph databases focus on representing relationships between entities in a more complex manner.
x??

---

#### Graph Databases Use Case
Graph databases excel at handling applications where entities have many connections to other entities. For example, social networks or recommendation engines.

:p What type of application is best suited for graph databases?
??x
Applications that require modeling and querying highly interconnected data, such as social networks, recommendation systems, or complex network analysis.
x??

---

#### Query Languages Overview
Various query languages exist for different data models: SQL (relational), Cypher (graph), MongoDB's aggregation pipeline (document), etc.

:p What is the primary purpose of SQL in database management?
??x
SQL is used to manage and query relational databases, providing structured queries for inserting, updating, deleting, and retrieving data.
x??

---

#### Cypher Query Language
Cypher is a declarative graph query language. For example, to find all friends of a user named "Alice", you would use a pattern matching syntax.

:p What does Cypher allow users to do?
??x
Cypher allows users to write queries for graph databases by defining patterns and relationships between nodes.
x??

---

#### MongoDB Aggregation Pipeline
MongoDB's aggregation pipeline processes documents through a series of stages, similar to SQL joins but more flexible. It can be used to aggregate data in complex ways.

:p How does the MongoDB aggregation pipeline work?
??x
The MongoDB aggregation pipeline processes documents stage by stage, allowing for complex operations such as filtering, grouping, and projecting.
```javascript
db.collection.aggregate([
   { $match: { field: "value" } },
   { $group: { _id: "$ field", count: {$sum: 1 } } }
])
```
x??

---

#### CSS and XSL/XPath Parallels
CSS is used for styling HTML, while XSL (XSLT) can transform XML documents into other formats. XPath provides a way to navigate through elements in an XML document.

:p What are the primary uses of CSS?
??x
CSS is primarily used for specifying the presentation of HTML and XML documents, including layout, colors, fonts, and more.
x??

---

#### Genome Databases Specialization
Genome databases like GenBank handle sequence similarity searches. These searches involve comparing a long DNA string against a database to find similar sequences.

:p What kind of problem does genome data pose for traditional relational databases?
??x
Genome data is typically very large, and the structure (long strings) doesn't fit well into the tabular format of traditional relational databases.
x??

---

#### Big Data in Particle Physics
Particle physics has been using big data techniques for decades. Projects like the Large Hadron Collider now work with vast amounts of data.

:p What kind of technology is used to handle large-scale data analysis in particle physics?
??x
Large-scale data analysis in particle physics uses specialized software and infrastructure, often involving distributed computing frameworks and big data technologies.
x??

---

#### Key-Value Storage Implementation

Background context: The provided example implements a simple key-value store using Bash functions. This implementation is straightforward and highlights basic storage and retrieval operations.

:p What does `db_set` function do?
??x
The `db_set` function stores a key-value pair in the database. It appends to a file, where each line contains a comma-separated key-value pair.
```bash
db_set 123456 '{\"name\":\"London\",\"attractions\":[\"Big Ben\",\"London Eye\"]}'
```
x??

---
#### Lookup Performance

Background context: The `db_get` function in the example performs poorly for large databases as it searches through the entire file. This is inefficient, leading to a linear search time complexity of O(n).

:p Why does the `db_get` function have poor performance?
??x
The `db_get` function has poor performance because it scans the entire database file from beginning to end each time a key lookup is requested. This results in a linear time complexity, making it inefficient for large datasets.
x??

---
#### Log-Structured Storage Engines

Background context: Many databases use log-based storage engines where data is appended only and can help with high write throughput but complicate read operations.

:p What is the advantage of using a log in database storage?
??x
The primary advantage of using a log in database storage is that it allows for efficient append-only writes, which can provide high write throughput. However, this simplifies the data structure at the cost of making random reads more complex.
x??

---
#### Indexing for Efficient Lookups

Background context: To improve lookup performance, databases use indexes to maintain additional metadata that helps locate specific data efficiently.

:p What is an index in database terms?
??x
An index in database terms is a separate data structure derived from the primary data. It acts as a signpost and speeds up read queries by providing quick access to data.
x??

---
#### Trade-offs Between Indexing

Background context: Indexes improve read performance but can slow down writes due to additional overhead.

:p What are the trade-offs of using indexes?
??x
Indexes improve read performance by reducing lookup times. However, they introduce overhead during write operations as the index needs to be updated every time data is written, slowing down write performance.
x??

---
#### Selecting Appropriate Storage Engines

Background context: Different types of workloads require different storage engines; for instance, transactional workloads differ from analytics workloads.

:p Why should application developers care about storage engine selection?
??x
Application developers need to choose the appropriate storage engine based on their workload. Proper selection can optimize performance and meet specific requirements, such as high throughput or complex query handling.
x??

---
#### Log-Structured vs Page-Oriented Storage

Background context: Two main types of storage engines are log-structured and page-oriented, with B-trees being an example of the latter.

:p What is the difference between log-structured and page-oriented storage?
??x
Log-structured storage is append-only, ideal for high write throughput scenarios. Page-oriented storage, such as B-trees, supports both reads and writes more efficiently but can handle larger datasets.
x??

---

