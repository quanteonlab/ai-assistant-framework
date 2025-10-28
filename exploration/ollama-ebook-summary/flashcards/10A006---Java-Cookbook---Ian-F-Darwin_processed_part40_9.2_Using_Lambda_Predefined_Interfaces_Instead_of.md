# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 40)

**Starting Chapter:** 9.2 Using Lambda Predefined Interfaces Instead of Your Own. Problem. Solution. Roll Your Own Functional Interface

---

#### Using Lambda Predefined Interfaces Instead of Your Own
In Java 8, predefined interfaces from `java.util.function` package can be used for lambda expressions instead of defining your own. These interfaces provide common functional methods that are widely applicable.

The `Predicate<T>` interface is one such example, which has only one method: `boolean test(T t)`.

:p What is the `Predicate<T>` interface used for in Java 8?
??x
`Predicate<T>` is an interface that represents a predicate (a boolean-valued function) of one argument. It checks whether some condition is met for a given input.
```java
// Example usage:
public class CameraSearch {
    public List<Camera> search(Predicate<Camera> tester) {
        // Implementation details here
    }
}
```
x??

---
#### Custom Functional Interface in Java 8
A functional interface is an interface that has exactly one abstract method. It can be used with lambda expressions to provide a concise way of passing behavior as a parameter.

The `@FunctionalInterface` annotation can be used to declare that an interface should be treated as a functional interface by the compiler, even if it's not annotated as such.

:p What is a custom functional interface in Java 8?
??x
A custom functional interface is an interface with exactly one abstract method that can be implemented using lambda expressions. It allows for flexible and concise code when dealing with functional programming concepts.
```java
// Example of a custom functional interface:
interface MyFunctionalInterface {
    int compute(int x);
}
```
x??

---
#### Implementing Custom Functional Interface Using Lambda
You can implement the abstract method of a custom functional interface using lambda expressions.

For example, `MyFunctionalInterface` has an abstract method `int compute(int x)`. This method can be implemented by a lambda expression that provides the desired behavior.

:p How would you use a lambda to implement a custom functional interface?
??x
You can use a lambda expression to provide the implementation for the abstract method of a custom functional interface. For example, if `MyFunctionalInterface` has an abstract method `int compute(int x)`, it can be implemented as follows:
```java
public class CustomImplementation {
    public static void main(String[] args) {
        MyFunctionalInterface myFunc = (x) -> x * x + 1;
        int result = myFunc.compute(5); // result will be 26
    }
}
```
x??

---
#### Default Methods in Functional Interfaces
Default methods allow you to add implementation to an interface, which can be overridden by implementing classes. This is useful for creating "mixins" or providing a base behavior that can be extended.

A default method has a method body and it can exist alongside other abstract methods in the same interface. Only one non-default method per functional interface may have a method body.

:p What are default methods in interfaces?
??x
Default methods in interfaces provide a way to add implementation to an interface without breaking existing implementations of that interface. This allows for a flexible way to extend or modify behavior while maintaining compatibility with older code.
```java
// Example of an interface with a default method:
public interface ThisIsStillFunctional {
    default int compute(int ix) { return ix * ix + 1; }
    int anotherMethod(int y);
}
```
x??

---

#### Introduction to Streams in Java 8
Streams are a new mechanism introduced with Java 8 that allow for processing data through a pipeline-like mechanism. This approach simplifies complex data transformations and aggregations using a fluent programming style.

:p What is the main feature of Streams in Java 8?
??x
Streams enable processing data via a pipeline where each step can perform operations like filtering, mapping, or reducing, with varying levels of parallelism. This makes complex data manipulations more concise and readable.
x??

---

#### Stream Producing Methods
These methods create a stream from a collection or array. They are the starting point for any stream processing.

:p What is an example of a stream producing method?
??x
The `Arrays.stream()` method is used to generate a stream from an array, as seen in Example 9-2.
```java
static Hero[] heroes = {
    new Hero("Grelber", 21),
    new Hero("Roderick", 12),
    new Hero("Francisco", 35),
    new Hero("Superman", 65),
    new Hero("Jumbletron", 22)
};

// Using Arrays.stream() to generate a stream from an array
Arrays.stream(heroes);
```
x??

---

#### Stream Passing Methods
These methods operate on a stream and return the reference to it, allowing for chaining of operations.

:p What are some examples of stream passing methods?
??x
Examples include `distinct()` (removes duplicate elements), `filter()` (filters elements based on a predicate), `limit()` (limits the number of elements processed), `map()` (maps each element using a provided operation), `peek()` (performs an action with each element and returns a reference to the stream for further processing), `sorted()` (sorts the elements in the stream), and `unordered()` (marks the stream as unordered).

```java
// Chaining operations like filter, map, and sorted
Arrays.stream(heroes)
    .filter(hero -> hero.age >= 21) // Filter adult heroes
    .map(Hero::name)                // Map to names
    .sorted()                       // Sort alphabetically
```
x??

---

#### Stream Terminating Methods
These methods conclude a streaming operation, producing the result of the processing.

:p What is an example of a stream terminating method?
??x
An example of a stream terminating method is `collect()`, which gathers elements into a collection. Another example could be `count()`, which returns the number of elements in the stream.
```java
// Terminating with collect to sum ages of adult heroes
long totalAge = Arrays.stream(heroes)
    .filter(hero -> hero.age >= 21) // Filter adult heroes
    .mapToInt(Hero::age)             // Convert age to int stream
    .sum();                          // Sum the ages

// Terminating with count to get the number of elements
long numAdultHeroes = Arrays.stream(heroes)
    .filter(hero -> hero.age >= 21) // Filter adult heroes
    .count();                        // Count the filtered heroes
```
x??

---

#### Lambda Expressions in Streams
Lambdas are used almost invariably for controlling stream operations like filtering, mapping, or sorting.

:p What is a lambda expression and how is it used with streams?
??x
A lambda expression is an anonymous function that can be passed as an argument to methods. In the context of streams, lambdas are often used to define conditions or transformations during operations such as `filter()`, `map()`, and `sorted()`.

Example:
```java
// Using a lambda with filter to find heroes older than 21
List<Hero> adultHeroes = Arrays.stream(heroes)
    .filter(hero -> hero.age >= 21) // Lambda expression filtering heroes
    .collect(Collectors.toList());   // Collecting the filtered heroes into a list

// Sorting names alphabetically using lambda
Arrays.stream(heroes)
    .sorted((hero1, hero2) -> hero1.name.compareTo(hero2.name)) // Lambda for sorting
    .forEach(System.out::println);                             // Printing sorted names
```
x??

---

---

#### Streams and Collectors Introduction
Streams are a powerful feature in Java for processing sequences of elements. They provide a more functional approach to handling collections by breaking down operations into smaller, reusable pieces.

:p What is the purpose of using streams in Java?
??x
The purpose of using streams in Java is to simplify and enhance the way you process data in collections such as lists or arrays. Streams allow for operations like filtering, mapping, reducing, and more, providing a functional programming style that can make your code cleaner and easier to reason about.

:p How does collecting with collectors simplify stream processing?
??x
Collecting with collectors simplifies stream processing by allowing you to efficiently summarize the content of a stream into a single result. Collectors are like folds in functional programming languages, combining elements in streams into a final result using a series of operations. This makes it easier to perform complex operations on collections without writing verbose code.

:p What is the difference between supplier(), accumulator(), and combiner() functions?
??x
The `supplier()` function creates an empty mutable container that will hold the results of your stream operations. The `accumulator()` function adds elements into this container, while the `combiner()` function merges multiple containers (from parallel streams) into a single container.

:p What is the optional final transform in the collector?
??x
The optional final transform in the collector, known as the `finisher()`, allows for an additional transformation on the result container after all elements have been accumulated. This can be used to perform operations like converting the container to a specific format or performing a custom operation.

---

#### Example of Using Streams and Collectors

:p How do you sum up years of experience from adult heroes using streams?
??x
You can use Java Streams along with collectors to filter, map, and reduce elements. Here's how:

```java
long adultYearsExperience = Arrays.stream(heroes)
                                 .filter(b -> b.age >= 18) 
                                 .mapToInt(Hero::getAge)
                                 .sum();
```

In this example:
- `Arrays.stream(heroes)` creates a stream from the array.
- `.filter(b -> b.age >= 18)` filters heroes who are adults.
- `.mapToInt(Hero::getAge)` maps each hero to their age as an integer.
- `.sum()` reduces the stream by summing up all ages.

:x?

---

#### Sorting Heroes by Name

:p How do you sort a list of heroes by name using streams?
??x
Sorting elements in Java Streams can be achieved using `sorted()`. Here's how:

```java
List<Object> sorted = Arrays.stream(heroes)
                            .sorted((h1, h2) -> h1.name.compareTo(h2.name))
                            .map(Hero::getName)
                            .collect(Collectors.toList());
```

In this example:
- `.sorted()` sorts the stream based on a custom comparator.
- `.map(Hero::getName)` maps each hero to their name as an object in the list.

:x?

---

#### Implementing Word Frequency Count

:p How do you implement word frequency count using Java Streams and Collectors?
??x
Implementing word frequency count involves breaking lines into words, counting occurrences of each word, and sorting them by frequency. Here's a simplified version:

```java
Map<String, Long> map = Files.lines(Path.of(args[0]))
                             .flatMap(s -> Stream.of(s.split("\\s+")))
                             .collect(Collectors.groupingBy(String::toLowerCase, Collectors.counting()));
```

In this example:
- `Files.lines(Path.of(args[0]))` reads lines from a file.
- `.flatMap(s -> Stream.of(s.split("\\s+")))` splits each line into words and flattens the stream.
- `.collect(Collectors.groupingBy(String::toLowerCase, Collectors.counting()))` groups words by their lowercase form and counts occurrences.

:p How do you print the 20 most frequent words sorted in descending order?
??x
To print the top 20 most frequent words:

```java
map.entrySet()
   .stream()
   .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
   .limit(20)
   .map(entry -> String.format("%-15d %s", entry.getValue(), entry.getKey()))
   .forEach(System.out::println);
```

In this example:
- `.entrySet()` gets a stream of entries.
- `.sorted(Map.Entry.<String, Long>comparingByValue().reversed())` sorts by the value (frequency) in reverse order.
- `.limit(20)` limits to the top 20 entries.
- `map(...)` formats and prints each entry.

:x?

---

---

