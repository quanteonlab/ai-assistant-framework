# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 32)

**Starting Chapter:** 7.7 Eschewing Duplicates with a Set. Problem. Solution. 7.8 Structuring Data in a Linked List

---

#### Set Interface Overview
Background context: The `Set` interface is a part of Java's Collection framework and provides an unordered collection with unique elements. It does not allow duplicate values, unlike the `List` which maintains order but allows duplicates.

:p What distinguishes a `Set` from a `List` in terms of storing data?
??x
A `Set` enforces uniqueness by ensuring that no two objects are equal as determined by their `equals()` method. In contrast, a `List` can contain duplicate elements and retains the order in which they are added.

The key difference is that a `Set` does not support index-based operations like `add(int, Object)` or `get(int)`, because knowing how many unique items there are might not match the number of total additions if duplicates were present. 

Code example:
```java
// Example usage of Set
Set<String> hashSet = new HashSet<>();
hashSet.add("One");
hashSet.add("Two");
hashSet.add("One"); // Duplicate, but only one "One" is added
hashSet.add("Three");

for (String s : hashSet) {
    System.out.println(s);
}
```
x??

---

#### Adding Elements to a Set
:p How do you add elements to a `Set` in Java?
??x
You can use the `add()` method of the `Set` interface to insert elements. The `add()` method returns `true` if the element was not already present and `false` otherwise, due to the uniqueness constraint.

Code example:
```java
Set<String> hashSet = new HashSet<>();
hashSet.add("One");
hashSet.add("Two");
hashSet.add("One"); // Duplicate, but only one "One" is added
hashSet.add("Three");

System.out.println(hashSet); // Output: [One, Two, Three]
```
x??

---

#### Checking for Duplicates in a Set
:p How can you check if an element already exists in a `Set`?
??x
You can use the `contains()` method to determine whether a set contains a particular element. This method checks the existence of elements based on their equality as determined by the `equals()` and `hashCode()` methods.

Code example:
```java
Set<String> hashSet = new HashSet<>();
hashSet.add("One");
hashSet.add("Two");
System.out.println(hashSet.contains("Three")); // Output: false

// Adding a duplicate
System.out.println(hashSet.contains("One")); // Output: true
```
x??

---

#### Set Operations and Methods
:p What are some common methods available in the `Set` interface?
??x
The `Set` interface provides several key methods, including:

- `add(E e)`: Adds an element to the set.
- `remove(Object o)`: Removes a specified element from this set if it is present.
- `contains(Object o)`: Returns true if this set contains the specified element.
- `size()`: Returns the number of elements in this set (the cardinality).
- `isEmpty()`: Returns true if this set contains no elements.

Code example:
```java
Set<String> hashSet = new HashSet<>();
hashSet.add("One");
hashSet.remove("Two"); // Removing an element that was not added, so nothing happens

System.out.println(hashSet.size()); // Output: 1
System.out.println(hashSet.isEmpty()); // Output: false
```
x??

---

#### Mutable Objects in a Set
:p What precautions should be taken when using mutable objects as elements in a `Set`?
??x
When using mutable objects as elements in a `Set`, care must be exercised because the behavior of the set is not specified if the value of an object is changed in a manner that affects `equals()` comparisons while the object is an element in the set. This can lead to unpredictable results.

For example, modifying any field of the object that contributes to its hash code or equality check (as determined by `equals()`) can cause issues with the set's internal operations.

Code example:
```java
class MutableObject {
    private String value;

    public MutableObject(String value) {
        this.value = value;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof MutableObject)) return false;
        return ((MutableObject) obj).value.equals(value);
    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }
}

Set<MutableObject> set = new HashSet<>();
set.add(new MutableObject("one"));
// Modifying the object while it's in the set can cause issues
((MutableObject)set.iterator().next()).value = "two";
```
x??

---

#### SortedSet Interface and TreeSet Implementation
Background context: The `SortedSet` interface is used for maintaining a sorted collection of unique elements. `TreeSet` is one of its common implementations, which uses a tree-based data structure to ensure that elements are stored in a natural order or according to a custom comparator.

:p What is the difference between `SortedSet` and `TreeSet`?
??x
The `SortedSet` interface provides an ordered collection where elements maintain their insertion order. `TreeSet` implements this interface by using a Red-Black tree, ensuring that elements are sorted based on natural ordering or a custom comparator.

```java
// Example of creating a TreeSet with Doubles
Set<Double> nums = Set.of(Math.PI, 22D / 7, Math.E);
```
x??

---

#### Using the `of` Method in Java 9
Background context: The `of` method was introduced in Java 9 as part of the `Set`, `List`, and other collection interfaces. It allows for creating immutable collections with a concise syntax.

:p How does the `of` method work?
??x
The `of` method is used to create an immutable set or list from static elements. These collections are type-safe, and their immutability ensures that once created, they cannot be modified.

```java
// Example of using Set.of to create a set with predefined values
Set<Double> nums = Set.of(Math.PI, 22D / 7, Math.E);
```
x??

---

#### Using `LinkedHashSet` for Ordered Collections
Background context: While `TreeSet` provides sorted collections, it does not maintain the order of insertion. If maintaining both uniqueness and ordering is required, `LinkedHashSet` can be used. It combines the properties of a `HashSet` with an ordered linked list.

:p What are the advantages of using `LinkedHashSet`?
??x
The main advantage of `LinkedHashSet` is that it maintains the elements in the order they were inserted while also ensuring uniqueness. This makes it useful when both ordering and uniqueness are required, but not necessarily sorting based on element values.

```java
// Example of using LinkedHashSet to maintain insertion order
Set<String> firstNames = new LinkedHashSet<>();
firstNames.add("Robin");
firstNames.add("Jaime");
firstNames.add("Joey");
```
x??

---

#### Implementing a Simple `LinkedList` in Java
Background context: A linked list is a linear data structure where each element (node) contains a reference to the next node. The `LinkedList` class in Java provides an implementation of this structure with various methods for manipulation.

:p How does the `add` method work in a linked list?
??x
The `add` method appends an element to the end of the linked list by creating a new node and linking it to the last node. If no nodes exist, it creates the first node as well.

```java
// Pseudocode for adding elements to a LinkedList
public void add(T o) {
    if (last == null) {  // If list is empty
        last = new TNode<>(o, null);
        first = last;  // Both are same in an empty list
    } else {
        last.next = new TNode<>(o, null);
        last = last.next;
    }
}
```
x??

---

#### Using `ListIterator` for Bidirectional Traversal
Background context: The `ListIterator` interface extends the `Iterator` and adds support for bidirectional traversal of elements in a list. This is useful when you need to iterate over elements in both forward and backward directions.

:p How does `ListIterator` work?
??x
The `ListIterator` provides methods to traverse elements in either direction (forward or backward) within the collection. It also supports insertion, replacement, and removal of elements during traversal.

```java
// Example of using ListIterator for bidirectional iteration
ListIterator<String> li = l.listIterator();
while (li.hasNext()) {
    System.out.println("Next element: " + li.next());
}
while (li.hasPrevious()) {
    System.out.println("Back to: " + li.previous());
}
```
x??

---

#### Clearing and Adding Elements in a Custom `LinkedList` Implementation
Background context: A custom implementation of a linked list can provide more control over the operations performed on it. This includes methods like `clear`, `add`, and `addAll`.

:p What is the purpose of the `clear` method?
??x
The `clear` method resets the linked list to its initial state by discarding all current nodes, effectively removing all elements.

```java
// Pseudocode for clearing a LinkedList
public void clear() {
    first = new TNode<>(null, null);
    last = first;
}
```
x??

---

#### Adding Elements at Specific Positions in a `LinkedList`
Background context: To add elements at specific positions within the linked list, you need to traverse the list until you reach the desired position and then insert the new node.

:p How does the `add(int i, T o)` method work?
??x
The `add` method inserts an element at a specified index by traversing from the head of the list until it reaches the insertion point. It then creates a new node with the given data and links it appropriately within the existing structure.

```java
// Pseudocode for adding elements at specific positions
public void add(int i, T o) {
    TNode<T> t = first;
    for (int j = 0; j < i && t != null; j++) {
        t = t.next;
    }
    if (t == null) throw new IndexOutOfBoundsException();
    final TNode<T> nn = new TNode<>(o, t.next);
    t.next = nn;
}
```
x??

---

#### Linked List Implementation Issues
Background context: The provided Java code snippet shows an incomplete implementation of a linked list. It includes methods for accessing elements and converting to arrays but does not use `java.util.LinkedList`. This example is intended to highlight common issues and differences from using the standard library's implementation.
:p What are the problems with directly implementing a linked list as shown in the code?
??x
The code directly implements basic operations of a linked list without leveraging the standard `java.util.LinkedList` class. Direct implementations often lead to issues such as:
- Potential memory leaks due to improper handling of nodes.
- Lack of built-in methods that would be provided by `LinkedList`, such as add, remove, and clear operations.
- Inefficient manipulation of the list, especially when trying to maintain order or handle concurrent access.

Using `java.util.LinkedList` ensures better performance, safety, and ease of use with its rich set of methods and thread-safety features.
x??

---

#### Using HashMap for Mapping
Background context: The text discusses using a `HashMap` to establish one-way mappings between different objects. It provides an example where company names are mapped to their addresses.
:p How does the `HashMap` class help in creating one-way mappings?
??x
The `HashMap` class helps create one-way mappings by providing key-value pairs, where keys and values can be of any object type. This is useful for scenarios like mapping user inputs (keys) to corresponding actions or data (values).

For example:
```java
Map<String, String> map = new HashMap<>();
map.put("Adobe", "Mountain View, CA");
```
Here, `"Adobe"` is the key and `"Mountain View, CA"` is the value. This structure allows for efficient lookups where you can retrieve the address by providing the company name as a key.
x??

---

#### Retrieval from HashMap
Background context: The text demonstrates retrieving values from a `HashMap` using different methods such as direct key lookup and iterating over all entries.
:p How can you retrieve a single value from a `HashMap` given its key?
??x
You can retrieve a single value from a `HashMap` by calling the `get()` method with the corresponding key. This method returns the value associated with the specified key, or null if no mapping for the key is present.

Example:
```java
String queryString = "O'Reilly";
String resultString = map.get(queryString);
System.out.println("They are located in: " + resultString);
```
This code retrieves the address associated with the company name `"O'Reilly"` and prints it.
x??

---

#### Iterating Over HashMap Entries
Background context: The text explains how to iterate over all entries in a `HashMap` using both traditional for-loops and lambda expressions.
:p How can you print all key-value pairs from a `HashMap`?
??x
You can print all key-value pairs from a `HashMap` by iterating over its entry set. This can be done with either a traditional for-loop or a stream-based approach using lambda expressions.

Example:
```java
// Using a for-each loop
for (String key : map.keySet()) {
    System.out.println("Key " + key + "; Value " + map.get(key));
}

// Using a forEach() method and lambda expression
map.entrySet().forEach(mE -> 
    System.out.println("Key: " + mE.getKey() + "; Value: " + mE.getValue()));
```
These methods ensure that all entries are printed, allowing for easy data retrieval or report generation.
x??

---

#### Concurrent Modification Handling
Background context: The text mentions the importance of handling concurrent modifications when iterating over a `HashMap`.
:p What happens if you modify a `HashMap` while iterating through it?
??x
If you attempt to modify a `HashMap` (such as adding, removing, or clearing elements) while iterating through it using an iterator, and the iteration itself is not controlled explicitly, a `ConcurrentModificationException` will be thrown.

To avoid this exception, you should use the explicit iterator methods provided by `HashMap`. For example:
```java
Iterator<String> it = map.keySet().iterator();
while (it.hasNext()) {
    // Perform some operation on key or value
}
```
This approach ensures that modifications are handled safely during iteration.
x??

---

