# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 6)

**Starting Chapter:** Property Graphs

---

#### Graph Data Models Overview
Background context: In data modeling, many-to-many relationships are crucial. While document and relational models handle simpler cases well, more complex many-to-many connections may require a graph model. A graph consists of vertices (nodes) and edges (relationships). Typical examples include social networks, the web graph, and road networks.

:p What is a graph in data modeling?
??x
A graph in data modeling consists of two main types of objects: vertices (nodes or entities) and edges (relationships or arcs). These elements are used to represent complex relationships between data points. For example, in a social network, people are nodes, and connections between them are edges.
x??

---

#### Property Graph Model
Background context: The property graph model allows flexibility by using unique identifiers for vertices and edges, along with labels and properties. Vertices can connect to any other vertex through edges, and each edge has labels describing the type of relationship.

:p What are the main components of a vertex in a property graph?
??x
Vertices in a property graph have:
- A unique identifier (vertex_id)
- A set of outgoing edges
- A set of incoming edges
- A collection of properties (key-value pairs)

Example: In Facebook’s data model, vertices could represent people, locations, events, checkins, and comments.
x??

---

#### Edge Components in Property Graphs
Background context: Edges connect two vertices and have unique identifiers, labels to describe the relationship, and associated properties. Efficient traversal is enabled by indexes on tail_vertex and head_vertex.

:p What information does an edge contain in a property graph?
??x
An edge contains:
- A unique identifier (edge_id)
- The starting vertex (tail_vertex) as a reference to another vertex
- The ending vertex (head_vertex) as a reference to another vertex
- A label describing the relationship between vertices
- A collection of properties (key-value pairs)

Example: In a social network, an edge labeled "FRIEND" might connect two people.
x??

---

#### Storing Property Graphs in Relational Tables
Background context: Vertices and edges can be stored using relational tables with unique identifiers and indexes for efficient querying. Properties are typically stored as JSON to allow flexibility.

:p How would you represent a property graph in a PostgreSQL database?
??x
You could use the following schema:
```sql
CREATE TABLE vertices (
    vertex_id integer PRIMARY KEY,
    properties json
);

CREATE TABLE edges (
    edge_id integer PRIMARY KEY,
    tail_vertex integer REFERENCES vertices (vertex_id),
    head_vertex integer REFERENCES vertices (vertex_id),
    label text,
    properties json
);
```
Indexes on `tail_vertex` and `head_vertex` help in querying incoming and outgoing edges efficiently.

Example:
```sql
CREATE INDEX edges_tails ON edges (tail_vertex);
CREATE INDEX edges_heads ON edges (head_vertex);
```

This schema allows for flexible storage of vertices and edges while enabling efficient traversal.
x??

---

#### Vertex Properties in Property Graphs
Background context: Each vertex can have a collection of properties, stored as key-value pairs. This flexibility is useful for representing diverse data types within the same graph.

:p What are properties in the context of property graphs?
??x
Properties in property graphs are key-value pairs associated with vertices or edges, allowing additional metadata and attributes to be stored. For example, a vertex representing a person might have properties like `name`, `age`, and `location`.

Example: A vertex could look like this:
```json
{
    "name": "Lucy",
    "age": 30,
    "location": "Idaho"
}
```
x??

---

#### Edge Labels in Property Graphs
Background context: Edges have labels to describe the type of relationship between two vertices. This helps maintain a clean data model by categorizing relationships.

:p What is an edge label used for in property graphs?
??x
An edge label in property graphs is used to describe the kind of relationship between two vertices, enabling clear and organized data representation. For example, in a social network, edges could be labeled "FRIEND", "CO-WORKER", or "RELATIVE".

Example: An edge connecting Lucy and Alain with a label "MARRIED" indicates their marital relationship.
x??

---

#### Efficient Traversal Using Indexes
Background context: In property graphs, indexes on `tail_vertex` and `head_vertex` enable efficient traversal. This allows for quick access to incoming or outgoing edges from any vertex.

:p How do indexes on tail_vertex and head_vertex facilitate graph traversal?
??x
Indexes on `tail_vertex` and `head_vertex` allow for fast querying of incoming and outgoing edges from a specific vertex. For instance, if you want to find all the friends of a person (vertices with an edge labeled "FRIEND" pointing to them), you would query using the index.

Example:
```sql
SELECT * FROM edges WHERE head_vertex = <vertex_id> AND label = 'FRIEND';
```
x??

---

#### Graph Database Basics
Background context explaining graph databases, their uses, and how they differ from traditional relational databases. Graph databases use vertices (nodes) to represent entities and edges to represent relationships between them.

Graphs are good for evolvability: as you add features to your application, a graph can easily be extended to accommodate changes in your application’s data structures.
:p What is a graph database?
??x
A graph database is a type of NoSQL database that stores and queries data using vertices (nodes) and edges. Vertices represent entities, and edges represent relationships between these entities. The structure allows for flexible schema and efficient querying based on complex relationships.

For example:
```java
public class Vertex {
    String id;
    String label;

    public Vertex(String id, String label) {
        this.id = id;
        this.label = label;
    }
}

public class Edge {
    String fromVertexId;
    String toVertexId;
    String label;

    public Edge(String fromVertexId, String toVertexId, String label) {
        this.fromVertexId = fromVertexId;
        this.toVertexId = toVertexId;
        this.label = label;
    }
}
```
x??

---

#### Cypher Query Language
Background context explaining the Cypher query language for property graphs. It is used in Neo4j and designed to be declarative, making it easy to write complex queries.

Example 2-3 shows how to insert data into a graph database using Cypher.
:p What is Cypher?
??x
Cypher is a declarative query language specifically designed for working with property graphs. It was created for Neo4j and allows you to define the structure of your data, create relationships between nodes (vertices), and query those structures.

Example 2-3 demonstrates inserting a subset of the data into a graph database:
```cypher
CREATE   (NAmerica:Location {name: 'North America', type:'continent' }),
         (USA:Location      {name: 'United States', type:'country'   }),
         (Idaho:Location    {name: 'Idaho',         type:'state'    }),
         (Lucy:Person       {name: 'Lucy' })
```
x??

---

#### Querying Graph Databases
Background context on how to write and execute queries in a graph database. It involves understanding vertex, edge relationships, and patterns.

Example 2-4 shows how to find people who emigrated from the US to Europe using Cypher.
:p How do you query for people who emigrated from the US to Europe?
??x
To query for people who emigrated from the US to Europe, you can use a Cypher query that follows these steps:
1. Find all vertices representing people born in the USA and currently living in Europe.
2. Use pattern matching to find the relationships.

Example 2-4 shows the Cypher query:

```cypher
MATCH   (person) -[:BORN_IN]-> ()
         () -[:WITHIN*0..]-> (us:Location {name: 'United States'}),
     (person) -[:LIVES_IN]-> ()
         () -[:WITHIN*0..]-> (eu:Location {name: 'Europe'})
RETURN person.name
```

This query finds any vertex `person` who:
1. Has a `BORN_IN` edge to some location within the US.
2. Currently lives in Europe.

The pattern matching in Cypher is very flexible, allowing for variable-length paths and complex relationships.
x??

---

#### SQL vs. Graph Queries
Background context on why graph queries differ from SQL queries when working with relational data structures. Graph databases allow for more dynamic traversal of edges and vertices, whereas SQL requires predefined joins and fixed path lengths.

Example 2-5 shows the same query in SQL using recursive common table expressions.
:p How do you represent a variable-length path in Cypher?
??x
In Cypher, you can use the `*` operator to represent a variable-length path. For example:

```cypher
() -[:WITHIN*0..]-> (vertex)
```

This means "follow a WITHIN edge zero or more times." This flexibility allows for dynamic traversal of the graph.

For instance:
```cypher
MATCH   (person) -[:BORN_IN]-> ()
         () -[:WITHIN*0..]-> (us:Location {name: 'United States'}),
     (person) -[:LIVES_IN]-> ()
         () -[:WITHIN*0..]-> (eu:Location {name: 'Europe'})
RETURN person.name
```

This query finds people born in the US and currently living in Europe, even if they live in different levels of the location hierarchy.
x??

---

#### Recursive Common Table Expressions in SQL
Background context on recursive common table expressions (CTEs) for SQL. They allow for dynamic path traversal similar to graph queries.

Example 2-5 shows how to use CTEs to find people who emigrated from the US to Europe.
:p How do you use recursive common table expressions (CTEs) in SQL?
??x
Recursive common table expressions (CTEs) in SQL allow you to define a query that can call itself, enabling dynamic path traversal. This is particularly useful for graph-like data where paths between nodes are not fixed.

Example 2-5 demonstrates using CTEs in SQL:

```sql
WITH RECURSIVE   -- in_usa is the set of vertex IDs of all locations within the United States
in_usa(vertex_id ) AS (
    SELECT vertex_id FROM vertices WHERE properties ->>'name' = 'United States'
    UNION 
    SELECT edges.tail_vertex FROM edges JOIN in_usa ON edges.head_vertex = in_usa.vertex_id
    WHERE edges.label = 'within'
),   -- in_europe is the set of vertex IDs of all locations within Europe
in_europe (vertex_id ) AS (
    SELECT vertex_id FROM vertices WHERE properties ->>'name' = 'Europe'
    UNION 
    SELECT edges.tail_vertex FROM edges JOIN in_europe ON edges.head_vertex = in_europe.vertex_id
    WHERE edges.label = 'within'   ),   -- born_in_usa is the set of vertex IDs of all people born in the US
born_in_usa (vertex_id ) AS (
    SELECT edges.tail_vertex FROM edges JOIN in_usa ON edges.head_vertex = in_usa.vertex_id
    WHERE edges.label = 'born_in'
)
-- Now find people who emigrated from the US to Europe
SELECT person.name FROM born_in_usa
JOIN vertices v1 ON born_in_usa.vertex_id = v1.id
JOIN edges e2 ON v1.id = e2.head_vertex AND e2.label = 'lives_in'
JOIN vertices v2 ON e2.tail_vertex = v2.id
WHERE v2.properties ->>'name' = 'Europe';
```

This query finds people born in the US and currently living in Europe by defining recursive CTEs to traverse the location hierarchy.
x??

---

#### Property Graph Model Overview
The property graph model represents data using vertices (nodes) and edges (relationships). Each vertex can have properties, and each edge connects two vertices with a label indicating their relationship.

:p What is the primary characteristic of the property graph model?
??x
Vertices represent entities, while edges connect these entities and define relationships between them. Vertices can have multiple properties.
x??

---

#### Query to Find People Born in USA and Living in Europe
This query involves joining vertices based on specific edge labels to find people who were born in the USA and are currently living in Europe.

:p How would you write a SQL-like query to find people born in the USA and living in Europe?
??x
```sql
WITH born_in_usa AS (
    SELECT edges.tail_vertex AS vertex_id 
    FROM edges 
    JOIN usa ON edges.head_vertex = usa.vertex_id 
    WHERE edges.label = 'born_in'
), lives_in_europe AS (
    SELECT edges.tail_vertex AS vertex_id 
    FROM edges 
    JOIN europe ON edges.head_vertex = europe.vertex_id 
    WHERE edges.label = 'lives_in'
)
SELECT vertices.properties->>'name' 
FROM vertices 
JOIN born_in_usa ON vertices.vertex_id = born_in_usa.vertex_id
JOIN lives_in_europe ON vertices.vertex_id = lives_in_europe.vertex_id;
```
x??

---

#### Intersecting Sets of Vertices
This involves creating sets based on specific vertex properties and then intersecting those sets to find common elements.

:p How would you create a set of vertices in a property graph that represent the "United States"?
??x
To create a set of vertices representing the "United States," you can use a similar approach as finding people born there. However, since we're dealing with places rather than individuals, it might look like this:

```sql
WITH in_usa AS (
    -- code to find all vertices connected to "United States" by 'located_in' edges
)
-- Similarly for Europe
```
x??

---

#### Triple-Store Model Explanation
The triple-store model stores data as triples: (subject, predicate, object). This is similar to the property graph model but uses different terminology.

:p What are the key components of a triple in a triple-store?
??x
A triple consists of three parts:
1. **Subject**: A vertex or another entity.
2. **Predicate**: Describes the relationship between the subject and object, akin to an edge label.
3. **Object**: Can be either a value (like a string) or another entity.

Example: `("Jim", "likes", "bananas")` where `"Jim"` is the subject, `"likes"` is the predicate, and `"bananas"` is the object.
x??

---

#### Example of Triples in Turtle Format
Turtle format is used to represent triples in a readable manner. This example shows how data might be represented.

:p How would you write the triple (Jim, likes, bananas) in Turtle format?
??x
```turtle
@prefix ex: <http://example.org/> .
ex:Jim ex:likes "bananas" .
```
x??

---

#### Building Up Sets of Vertices Based on Properties
This involves traversing from vertices with specific properties to find connected entities.

:p How would you build up the set of vertices representing places in a property graph?
??x
To build up the set of vertices, starting from a place (e.g., "United States"), follow all incoming edges that indicate location (`located_in`), and add these vertices to the set. This is done recursively.

```sql
WITH in_usa AS (
    SELECT edges.tail_vertex 
    FROM edges 
    JOIN usa ON edges.head_vertex = usa.vertex_id 
    WHERE edges.label = 'located_in'
)
-- Similarly for Europe
```
x??

---

#### Intersecting Sets of People Based on Birth and Location
This involves finding people who meet multiple criteria, such as being born in the USA and currently living in Europe.

:p How would you intersect sets to find people based on birthplace and current location?
??x
```sql
WITH born_in_usa AS (
    SELECT edges.tail_vertex AS vertex_id 
    FROM edges 
    JOIN usa ON edges.head_vertex = usa.vertex_id 
    WHERE edges.label = 'born_in'
), lives_in_europe AS (
    SELECT edges.tail_vertex AS vertex_id 
    FROM edges 
    JOIN europe ON edges.head_vertex = europe.vertex_id 
    WHERE edges.label = 'lives_in'
)
SELECT vertices.properties->>'name' 
FROM vertices 
JOIN born_in_usa ON vertices.vertex_id = born_in_usa.vertex_id
JOIN lives_in_europe ON vertices.vertex_id = lives_in_europe.vertex_id;
```
x??

---

#### Turtle Triples Representation
Background context: The provided text explains how data can be represented using Turtle triples, a format for representing data in RDF (Resource Description Framework). This representation uses URI prefixes to define namespaces and "blank nodes" (`_:someName`) to represent vertices that do not have unique URIs.
:p What is the significance of blank nodes (`_:someName`) in Turtle triples?
??x
Blank nodes are used when a node does not have a unique identifier, as they exist only within the context of the current graph. They are typically used for representing entities that are not accessible by URIs or do not need to be uniquely identified.
For example, in the provided data:
```turtle
_:lucy     a       :Person;   :name "Lucy";          :bornIn _:idaho.
```
The blank node `_:lucy` and `_:idaho` are used to represent entities that do not have unique URIs but are referred to in the data. The `_:` prefix indicates that these are local identifiers for this specific data context.

---
#### More Concise Turtle Syntax
Background context: The example shows a more concise way of writing Turtle triples by grouping properties together using semicolons (`;`). This reduces repetition and makes the syntax more readable.
:p How does using semicolons in Turtle syntax reduce redundancy?
??x
Using semicolons allows multiple properties to be associated with the same subject without having to repeat the subject. For instance, instead of writing:
```turtle
_:lucy     a       :Person.
_:lucy     :name   "Lucy".
_:lucy     :bornIn _:idaho.
```
You can write:
```turtle
_:lucy     a :Person;   :name "Lucy";          :bornIn _:idaho.
```
This reduces the amount of repetition and makes the data more readable.

---
#### Triple Stores vs. Semantic Web
Background context: The text discusses the distinction between triple stores, which are databases that store and query RDF (Resource Description Framework) data using triples, and the semantic web. It explains that while they are closely linked in many people’s minds, they serve different purposes.
:p What is a key difference between triple stores and the semantic web?
??x
A key difference is that triple stores provide a mechanism for storing and querying RDF data, whereas the semantic web is an idea about publishing machine-readable data on the internet to allow automatic combination of information from different sources. While triple stores can be used as part of implementing the semantic web, they do not necessarily need to support all its features.

---
#### Resource Description Framework (RDF)
Background context: RDF is a standard for representing metadata and linking data between different resources. The provided text mentions that Turtle is a human-readable format for RDF.
:p What does RDF stand for and what is its purpose?
??x
RDF stands for "Resource Description Framework." Its primary purpose is to represent information about web resources in a way that can be processed by software, allowing data from different sources to be combined into a unified whole. This allows for the creation of a "web of data" where machines can understand and process data just as humans do with text.

---
#### Triple Stores in Datomic
Background context: The passage discusses Datomic, which is described as a triple store that does not claim any semantic web integration.
:p What is Datomic and how is it different from other triple stores?
??x
Datomic is a type of database management system designed to handle large-scale data with strong transactional guarantees. It uses a 5-tuple model rather than the standard RDF triples, providing versioning capabilities. Unlike many other triple stores that are often associated with semantic web technologies, Datomic does not explicitly claim to support or integrate directly with those concepts.

#### RDF and URI Usage
RDF (Resource Description Framework) is designed for internet-wide data exchange, leading to the use of URIs as subjects, predicates, and objects. This design ensures compatibility with other datasets by using fully qualified URLs even when not resolving to anything specific. The namespace can be defined once at the top.
:p What are the key characteristics of RDF in terms of URI usage?
??x
In RDF, URIs are used for all components (subject, predicate, and object) to ensure interoperability and avoid conflicts with other datasets that might use different meanings for the same words. A namespace is defined using a URL or non-resolvable URI like `urn:example:`.
```java
// Example of defining a prefix in RDF
PREFIX : <urn:example:>
```
x??

---

#### SPARQL Query Language
SPARQL is a query language for RDF data, specifically designed to work with triple-stores. It predates Cypher and shares similar syntax due to pattern matching influences.
:p What is SPARQL used for in the context of RDF data?
??x
SPARQL is used to query RDF datasets by constructing patterns that match triples within the graph structure. The language allows for complex queries involving multiple relationships.
```java
// Example SPARQL query to find people who have moved from US to Europe
PREFIX : <urn:example:> 
SELECT ?personName 
WHERE { 
  ?person :name ?personName . 
  ?person :bornIn / :within* / :name "United States" . 
  ?person :livesIn / :within* / :name "Europe" . 
}
```
x??

---

#### Cypher and SPARQL Similarities
Cypher, a graph query language used by Neo4j, borrows pattern matching concepts from SPARQL. This results in similar-looking syntax for certain queries.
:p How does Cypher compare to SPARQL in terms of query patterns?
??x
Cypher and SPARQL share similarities due to borrowing pattern matching logic from each other. For example, both use a pattern matching approach where nodes and relationships are matched based on their labels and properties.
```java
// Example Cypher query equivalent to the SPARQL one
MATCH (person)-[:BORN_IN]->()<-[:WITHIN*0..]-(location)
RETURN person.name AS personName
```
x??

---

#### Property Matching in RDF
In RDF, properties are matched using predicates. This allows for flexibility where a property can be represented as an edge in the graph.
:p How is property matching handled differently in RDF compared to traditional graph databases?
??x
In RDF, properties are treated like edges and are matched using predicates. For instance, a predicate like `:name` can represent both a property and a relationship, making it flexible for various data modeling needs.
```java
// Example of matching a property in SPARQL
?usa :name "United States".
```
x??

---

#### Conclusion
The examples provided illustrate how RDF and SPARQL handle data representation and querying. Understanding these concepts is crucial for working with graph-like data models.
:p What are the key takeaways from the provided text regarding RDF, SPARQL, and their applications?
??x
Key takeaways include:
- URIs in RDF ensure compatibility across datasets.
- SPARQL provides a powerful query language for RDF data.
- Cypher and SPARQL share similarities due to borrowing pattern matching concepts.
- Properties and relationships are represented using predicates in RDF.
These points highlight the importance of understanding these tools for effectively querying graph-like data models.
x??

---

