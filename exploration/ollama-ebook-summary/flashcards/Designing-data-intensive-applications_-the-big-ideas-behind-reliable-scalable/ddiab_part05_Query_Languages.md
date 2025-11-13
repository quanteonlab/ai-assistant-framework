# Flashcards: Designing-data-intensive-applications_-the-big-ideas-behind-reliable-scalable_processed (Part 5)

**Starting Chapter:** Query Languages for Data

---

#### Nonsimple Domains in Relational Databases
:p What is a nonsimple domain, and how does it relate to JSON documents?
??x
A nonsimple domain allows values in a row of a relational table to be more complex than just primitive datatypes like numbers or strings. Instead, these values can be nested relations (tables), supporting an arbitrarily nested tree structure similar to what is found in JSON documents. Codd's original description of the relational model included this concept.
x??

---

#### Support for JSON Documents in Relational Databases
:p How have major relational databases like PostgreSQL and MySQL supported JSON documents over time?
??x
PostgreSQL since version 9.3, MySQL since version 5.7, and IBM DB2 since version 10.5 have added significant support for JSON documents within their query languages. This support allows these databases to handle document-like data while still performing relational queries. As web APIs increasingly use JSON, it is likely that other relational database management systems will follow suit.
x??

---

#### Relational and Document Databases Convergence
:p How are relational and document databases becoming more similar over time?
??x
Relational and document databases are converging through features such as JSON support in relational databases and query language enhancements in document databases. This convergence allows applications to leverage the strengths of both models, using hybrid approaches that best fit their needs. Examples include PostgreSQL's and MySQL's JSON support and RethinkDB's ability to perform joins similar to those found in traditional relational databases.
x??

---

#### Query Languages: Imperative vs Declarative
:p What is the difference between an imperative query language and a declarative one?
??x
An imperative query language, like many common programming languages, specifies step-by-step instructions for the computer to follow. An example of this is shown in the provided code where animals are filtered to return only sharks:
```java
function getSharks() {
    var sharks = [];
    for (var i = 0; i < animals.length; i++) {
        if (animals[i].family === "Sharks") {
            sharks.push(animals[i]);
        }
    }
    return sharks;
}
```
In contrast, a declarative query language like SQL or relational algebra specifies the desired pattern of data without detailing how to achieve it. The example provided for the imperative approach can be expressed more concisely in a declarative manner:
```sql
sharks = σ family="Sharks" (animals)
```
The key difference is that the database's query optimizer decides on the most efficient way to execute a declarative query, while an imperative query must provide specific instructions.
x??

---

#### Benefits of Declarative Query Languages
:p Why are declarative query languages more attractive than imperative ones?
??x
Declarative query languages like SQL or relational algebra are more concise and easier to work with compared to imperative APIs. They also hide the internal implementation details of the database engine, allowing for performance improvements without requiring changes to queries. Additionally, they can facilitate parallel execution better because they specify only the desired results rather than the exact steps required to achieve them.
x??

---

#### Declarative vs. Imperative Approaches: CSS and XSL

In web development, declarative languages like CSS (Cascading Style Sheets) and XSL (eXtensible Stylesheet Language) allow developers to specify how elements should look without detailing every step required for the browser to apply these styles.

CSS uses selectors to target specific parts of a document. For example, `li.selected > p` targets all `<p>` tags that are direct children of an `<li>` with the class `selected`.

XSL is more complex and can be used for transforming XML documents into other formats, but it also uses XPath expressions, which are similar to CSS selectors.

In contrast, imperative approaches like JavaScript provide detailed steps on how elements should be manipulated. This often involves querying the DOM (Document Object Model) and modifying properties directly.

:p What is an example of a declarative approach in web development?
??x
A declarative approach uses predefined rules or patterns to specify how elements should look or behave, such as using CSS selectors.
```css
/* Example of a CSS selector */
li.selected > p {
    background-color: blue;
}
```
x??

---
#### Imperative Approach with JavaScript and DOM API

JavaScript can be used imperatively to manipulate the document. The following code snippet demonstrates how to change the background color of selected elements using the core DOM API.

:p What is an example of an imperative approach in web development?
??x
An imperative approach involves writing detailed steps or instructions for the browser to follow, such as querying the DOM and modifying properties directly.
```javascript
var liElements = document.getElementsByTagName("li");
for (var i = 0; i < liElements.length; i++) {
    if (liElements[i].className === "selected") {
        var children = liElements[i].childNodes;
        for (var j = 0; j < children.length; j++) {
            var child = children[j];
            if (child.nodeType === Node.ELEMENT_NODE && child.tagName === "P") {
                child.setAttribute("style", "background-color: blue");
            }
        }
    }
}
```
x??

---
#### Performance and Maintenance Considerations

Declarative approaches like CSS and XSL can be easier to maintain and update because changes are made globally without altering the application logic. Browsers also optimize these declarative styles automatically.

Imperative scripts, on the other hand, require more detailed coding which can lead to issues such as incorrect style removal when class attributes change or needing to rewrite code if new APIs become available.

:p Why might a declarative approach be easier to maintain than an imperative one?
??x
A declarative approach is easier to maintain because changes are made in a global context and do not require altering the application logic. Browsers can automatically detect and apply these changes, making it simpler to update styles or behaviors.
```css
/* Example of declarative style change */
li.selected > p {
    background-color: blue;
}
```
x??

---
#### Browser Compatibility with CSS and XSL

CSS and XPath expressions used in XSL are designed to be backward-compatible. This means that browser vendors can implement optimizations without breaking existing applications.

:p Why is backward compatibility important for declarative languages like CSS?
??x
Backward compatibility ensures that existing code works even as new versions of the language or tools are released, allowing developers to rely on their existing implementations while benefiting from improved performance and features.
```css
/* Example of a CSS selector */
li.selected > p {
    background-color: blue;
}
```
x??

---

#### MapReduce Programming Model
Background context explaining the MapReduce programming model. It is a framework for processing large data sets across many machines, popularized by Google. The Map function processes input data and generates intermediate key-value pairs, while the Reduce function aggregates these pairs to generate output.

:p What are the main components of the MapReduce programming model?
??x
The main components of the MapReduce programming model are:
- **Map Function**: Takes a set of inputs and generates a set of intermediate key-value pairs.
- **Shuffle Phase**: Shuffles the generated intermediate key-value pairs to the corresponding reducers based on their keys.
- **Reduce Function**: Aggregates the values for each unique key, producing the final output.

For example:
```java
// Pseudocode for Map and Reduce functions in MapReduce

public void map(String key, String value) {
    // Parse input (key, value)
    emit(key, value);
}

public void reduce(Key key, Iterator values) {
    while (values.hasNext()) {
        // Process the values associated with a given key
    }
}
```
x??

---

#### PostgreSQL Query Example for MapReduce
Background context explaining how a simple query can be written using SQL in PostgreSQL to generate monthly shark sightings reports.

:p How would you write a PostgreSQL query to count the number of shark sightings per month?
??x
To write a PostgreSQL query to count the number of shark sightings per month, use the `SELECT` statement with `date_trunc` and `GROUP BY` clauses:

```sql
SELECT date_trunc('month', observation_timestamp) AS observation_month,
       SUM(num_animals) AS total_animals
FROM observations
WHERE family = 'Sharks'
GROUP BY date_trunc('month', observation_timestamp);
```

This query works by:
1. Truncating the `observation_timestamp` to the beginning of each month using `date_trunc`.
2. Filtering records where the `family` is `'Sharks'`.
3. Grouping results by the truncated timestamp.
4. Summing up the number of animals seen in each group.

The result will provide a count of shark sightings per month:
```sql
observation_month | total_animals
------------------|--------------
2023-10-01        | 5            -- Example output
```
x??

---

#### MongoDB MapReduce Example
Background context explaining how the same query can be implemented in MongoDB using its `mapReduce` function.

:p How would you write a MongoDB mapReduce script to count the number of shark sightings per month?
??x
To write a MongoDB mapReduce script to count the number of shark sightings per month, use the following code:

```javascript
db.observations.mapReduce(
    function() {  // Map Function
        var year = this.observationTimestamp.getFullYear();
        var month = this.observationTimestamp.getMonth() + 1;
        emit(year + "-" + month, this.numAnimals);
    },
    function(key, values) {  // Reduce Function
        return Array.sum(values);
    },
    {
        query: { family: "Sharks" },  // Filter by 'Sharks' family
        out: "monthlySharkReport"     // Output collection name
    }
);
```

This script works as follows:
1. **Map Function**: Extracts the year and month from `observationTimestamp` and emits a key-value pair.
2. **Reduce Function**: Sums up the values for each unique key (month and year).
3. **Query Filter**: Filters documents where the family is `'Sharks'`.
4. **Output Collection**: Stores results in `monthlySharkReport`.

The output will be stored in the collection `monthlySharkReport`, providing a count of shark sightings per month.
x??

---

#### MongoDB Aggregation Pipeline
Background context explaining how the aggregation pipeline can be used to achieve similar functionality with a more declarative approach.

:p How would you write an aggregation pipeline query in MongoDB to count the number of shark sightings per month?
??x
To write an aggregation pipeline query in MongoDB to count the number of shark sightings per month, use the following code:

```javascript
db.observations.aggregate([
    { $match: { family: "Sharks" } },  // Filter by 'Sharks' family
    {
        $group: {
            _id: { year: { $year: "$ observationTimestamp" }, month: {$month: "$ observationTimestamp" } },
            totalAnimals: { $sum: "$ numAnimals" }
        }
    }
]);
```

This pipeline works as follows:
1. **$match Stage**: Filters documents where the family is `'Sharks'`.
2. **$group Stage**: Groups by year and month using `$ year` and `$ month`, respectively, and sums up `numAnimals`.

The result will provide a count of shark sightings per month in the aggregation pipeline output:
```javascript
{ _id: { year: 2023, month: 10 }, totalAnimals: 5 }
```
x??

---

