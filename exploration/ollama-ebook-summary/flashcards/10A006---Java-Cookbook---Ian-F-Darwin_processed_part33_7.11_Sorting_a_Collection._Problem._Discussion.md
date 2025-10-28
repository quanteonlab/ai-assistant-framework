# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 33)

**Starting Chapter:** 7.11 Sorting a Collection. Problem. Discussion

---

#### Properties of Java HashMap and Properties Classes
Background context explaining how `HashMap` and `Properties` classes are used to store properties, but do not guarantee any specific order. The output ordering can vary based on internal implementations.

:p How does the order of entries differ between `HashMap` and `Properties` in the provided example?
??x
The order of entries is not guaranteed by both `HashMap` and `Properties`. The order may vary each time you access or retrieve elements due to their underlying implementation details. For `HashMap`, it relies on hash codes for quick access but does not maintain any specific order. Similarly, `Properties` are stored as a key-value map in the file format, which also doesn't guarantee any particular sequence when iterating.

```java
import java.util.Properties;

public class PropsExample {
    public static void main(String[] args) {
        Properties props = new Properties();
        // Adding properties
        props.setProperty("Sony", "Japan");
        props.setProperty("Sun", "Mountain View, CA");
        props.setProperty("IBM", "White Plains, NY");
        
        // Outputting the properties
        for (String key : props.keySet()) {
            System.out.println(key + "=" + props.getProperty(key));
        }
    }
}
```
x??

---

#### FileProperties Class Constructor
Background context explaining that `FileProperties` is a custom class that extends or wraps around `Properties`. It includes a constructor to load properties from a file, which can throw an exception.

:p How does the `FileProperties` constructor work?
??x
The `FileProperties` constructor loads properties from a specified file when it is created. If the file cannot be read or any I/O error occurs during loading, it throws an `IOException`.

```java
import com.darwinsys.util.FileProperties;

public class FilePropsExample {
    public static void main(String[] args) {
        try {
            Properties p = new FileProperties("PropsDemo.out");
            // Use the properties as needed
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```
x??

---

#### Sorting Collections in Java
Background context explaining that sorting collections is necessary when default ordering is not sufficient. `Arrays.sort()` and `Collections.sort()` are commonly used methods for this purpose.

:p How can you sort an array of strings using the default order?
??x
You can use the static method `Arrays.sort(strings)` to sort an array of strings in lexicographical (dictionary) order by default.

```java
public class SortArray {
    public static void main(String[] args) {
        String[] strings = { "painful", "mainly", "gaining", "raindrops" };
        
        Arrays.sort(strings);
        
        for (int i = 0; i < strings.length; i++) {
            System.out.println(strings[i]);
        }
    }
}
```
x??

---

#### Custom Comparator for Substring Comparison
Background context explaining that sometimes default sorting might not meet the requirement, and a custom comparator can be used to sort based on specific criteria.

:p How can you create a custom comparator to ignore the first character of strings during sorting?
??x
To create a custom comparator that ignores the first character of strings, implement the `Comparator<String>` interface. Here's an example:

```java
import java.util.Comparator;

public class SubstringComparator implements Comparator<String> {
    @Override
    public int compare(String s1, String s2) {
        s1 = s1.substring(1);
        s2 = s2.substring(1);
        
        return s1.compareTo(s2); // or use: return s1.substring(1).compareTo(s2.substring(1));
    }
}
```

This comparator can be used with `Arrays.sort()`:

```java
public class SubstringComparatorDemo {
    public static void main(String[] args) {
        String[] strings = { "painful", "mainly", "gaining", "raindrops" };
        
        Arrays.sort(strings);
        dump(strings, "Using Default Sort");
        
        Arrays.sort(strings, new SubstringComparator());
        dump(strings, "Using SubstringComparator");

        System.out.println("Functional approach:");
        Arrays.stream(strings)
              .sorted(Comparator.comparing(s -> s.substring(1)))
              .forEach(System.out::println);
    }

    static void dump(String[] args, String title) {
        System.out.println(title);
        for (String s : args) {
            System.out.println(s);
        }
    }
}
```
x??

---

#### Natural Ordering and Comparable Interface
Background context explaining that the `Comparable` interface allows objects to be sorted based on their natural ordering, which should be consistent with the `equals()` method. The `compareTo()` method is used for comparison.
:p What does the `Comparable` interface enable in Java?
??x
The `Comparable` interface enables objects to define a natural ordering that can be used for sorting and comparing. This is achieved through the implementation of the `compareTo()` method, which compares two instances of the class.
??x
```java
// Example of a simple Comparable implementation
public class Person implements Comparable<Person> {
    private String name;
    private int age;

    @Override
    public int compareTo(Person other) {
        return this.age - other.age; // Sorting by age in ascending order
    }
}
```
x??

---

#### Consistency Between `equals()` and `compareTo()`
Background context on the importance of ensuring that the `compareTo()` method is consistent with the `equals()` method. This means if two objects are equal, they should compare as equal.
:p How does the documentation recommend implementing `Comparable` for natural ordering to be consistent with `equals()`?
??x
The documentation recommends that a class’s natural ordering (implemented via `Comparable`) must be consistent with its `equals()` method. Specifically, if two instances of a class are considered equal by `equals()`, their `compareTo()` should return zero.
??x
```java
// Example to ensure compareTo is consistent with equals()
public int compareTo(Person other) {
    if (this.equals(other)) { // Check using equals first
        return 0; // If equals, must return 0 for compareTo
    }
    // Further comparison logic
}
```
x??

---

#### Implementing `equals()` and `hashCode()`
Background context on the relationship between `equals()` and `hashCode()`. The documentation suggests that if a class implements `equals()`, it should also implement `hashCode()` to ensure consistency.
:p Why is implementing `hashCode()` recommended when using `equals()`?
??x
Implementing `hashCode()` is recommended when using `equals()` because of the general contract between these two methods. According to Java's API documentation, if two objects are equal (`equals()` returns true), they must have the same hash code (`hashCode()`). This ensures that objects can be correctly placed in hash-based collections like HashMap and HashSet.
??x
```java
// Example implementing hashCode() with equals()
@Override
public int hashCode() {
    int result = 17;
    result = 31 * result + name.hashCode();
    result = 31 * result + age;
    return result;
}
```
x??

---

#### Custom Comparison Logic in `Appt` Class
Background context on the complexity of comparison logic when dealing with multiple fields. In this example, the `Appt` class compares appointments based on date and time (if available), followed by text.
:p How does the `Appt` class handle complex comparisons?
??x
The `Appt` class handles complex comparisons by breaking down the comparison into smaller parts:
1. First, it compares dates. If the dates are different, it returns the result of comparing those dates.
2. If the dates are the same, it then checks times. Only if both times are non-null does it compare them.
3. If only one time is null or none are set, additional logic determines which appointment should come first (all-day appointments sort low).
4. Finally, if all else fails and dates/times are identical, it compares the text.
??x
```java
@Override
public int compareTo(Appt a2) {
    // Compare dates first
    int dateComp = date.compareTo(a2.date);
    if (dateComp != 0) return dateComp;

    // Dates are the same. Compare times next
    if (time != null && a2.time != null) {
        int timeComp = time.compareTo(a2.time);
        if (timeComp != 0) return timeComp;
    } else if (time == null && a2.time != null) {
        return -1; // All-day appts sort low
    } else if (a2.time == null && time != null) {
        return +1; // Non-all-day appt sorts high
    }

    // Dates and times are the same. Compare text
    return text.compareTo(a2.text);
}
```
x??

---

#### `hashCode()` Implementation for `Appt` Class
Background context on how `hashCode()` is implemented to ensure consistent behavior with `equals()`. The `hashCode()` method must be carefully crafted to handle nulls and use appropriate hash codes.
:p How does the `Appt` class implement its `hashCode()`?
??x
The `Appt` class implements its `hashCode()` by using a prime number multiplier (31) to combine the hash codes of non-null fields. It ensures that null values are handled appropriately, avoiding potential issues with null references.
??x
```java
@Override
public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((date == null) ? 0 : date.hashCode());
    result = prime * result + ((text == null) ? 0 : text.hashCode());
    result = prime * result + ((time == null) ? 0 : time.hashCode());
    return result;
}
```
x??

---

---
#### Sorting a Set and Printing
Background context: The text explains how to sort elements of a set using a TreeMap, which maintains keys (and values) in sorted order. This is useful when you need to process or display data in an ordered manner.

:p How do you sort the elements of a set and print them?
??x
To sort the elements of a set, you can use a `TreeMap` because it automatically sorts its keys. Here's how you can implement this:

```java
import java.util.Set;
import java.util.TreeMap;

public class SortSetExample {
    Set<String> theSet = new HashSet<>();
    
    public void printSortedSet() {
        // Convert set to a TreeMap, which will sort elements by key (natural ordering or custom comparator)
        TreeMap<String, String> sortedMap = new TreeMap<>(theSet);
        
        // Print the whole list in sorted order
        System.out.println("Sorted list:");
        sortedMap.forEach((name) -> System.out.println(name));
    }
}
```

The `TreeMap` constructor takes a set and sorts its elements. The `forEach` method is used to print each element.
x??
---

#### Using Hashtable or HashMap for Sorted Output
Background context: While `Hashtable` and `HashMap` do not maintain any order, you can convert them into a `TreeMap` using the TreeMap's constructor that accepts a map. This way, you can get sorted output based on natural ordering of keys (or custom comparator).

:p How can you sort elements from a `Hashtable` or `HashMap`?
??x
To sort elements from a `Hashtable` or `HashMap`, you can convert them into a `TreeMap`. Here’s how:

```java
import java.util.Map;
import java.util.TreeMap;

public class SortedMapExample {
    Map<String, String> unsortedHashMap = new HashMap<>();

    public void sortMap() {
        // Convert map to a TreeMap, which will automatically sort elements by key (or custom comparator)
        TreeMap<String, String> sortedMap = new TreeMap<>(unsortedHashMap);

        // Print the whole list in sorted order
        System.out.println("Sorted map:");
        for (String key : sortedMap.keySet()) {
            System.out.println(key + " -> " + sortedMap.get(key));
        }
    }
}
```

The `TreeMap` constructor takes a map and sorts its entries by their natural ordering or through a custom comparator.
x??
---

#### Checking for an Object in a Collection
Background context: The text discusses various methods to check if a collection contains a specific value. Different methods exist depending on the type of collection, such as `ArrayList`, `HashSet`, `HashMap`, etc.

:p How do you check whether a given collection contains a particular value?
??x
You can use different methods based on the type of collection:

- **For ArrayList, HashSet, LinkedList, Properties, Vector**:
  ```java
  boolean contains = collection.contains(value);
  ```

- **For HashMap and Hashtable**:
  ```java
  boolean containsKey = map.containsKey(key);
  boolean containsValue = map.containsValue(value);
  ```

- **For Stack**:
  ```java
  int index = stack.indexOf(object);
  ```

These methods perform a linear search if the collection is a `List` or `Set`, but they are fast for hashed collections like `HashSet` and `HashMap`.
x??
---

#### Example of Binary Search in Arrays
Background context: The text provides an example where arrays need to be sorted before using binary search. This ensures faster search times compared to linear searches, especially on large datasets.

:p How do you perform a binary search on a random array of integers?
??x
First, ensure the array is sorted. Then use `Arrays.binarySearch()`:

```java
import java.util.Arrays;

public class BinarySearchExample {
    public static void main(String[] args) {
        int[] data = new int[MAX];
        Random r = new Random();
        
        // Fill the array with random numbers (this step is not shown)
        
        // Sort the array to make binary search possible
        Arrays.sort(data);
        
        // Perform binary search
        int index = Arrays.binarySearch(data, NEEDLE);
        
        if (index >= 0) {
            System.out.println("Found at index: " + index);
        } else {
            System.out.println("Not found");
        }
    }
}
```

The `Arrays.sort()` method sorts the array in ascending order. The `Arrays.binarySearch()` method performs a binary search on the sorted array to find the specified value.
x??
---

#### Generating Random Integers within a Range
Background context: This concept involves generating random integers using Java's `Random` class and mapping these values to a specific range. The `nextFloat()` method is used, which returns a float value between 0 (inclusive) and 1 (exclusive), and it gets multiplied by the desired maximum value (`MAX`) to scale the output.

:p How does one generate random integers within a given range using Java's Random class?
??x
The process involves using `Random.nextFloat()` method, which generates values in the range [0.0, 1.0). By multiplying this result with the desired maximum value and casting it to an integer, you can create a uniform distribution of random integers within that range.

```java
import java.util.Random;

public class RandomGenerator {
    private static final int MAX = 100; // Example max value

    public void generateRandomIntegers(int count) {
        Random r = new Random();
        int[] haystack = new int[count];
        for (int i = 0; i < count; ++i) {
            haystack[i] = (int)(r.nextFloat() * MAX);
        }
    }
}
```
x??

---

#### Precondition for Binary Search
Background context: This topic explains the necessity of having a sorted array or collection before performing a binary search. The binary search algorithm works by repeatedly dividing the search interval in half, which requires that elements are already ordered.

:p What is the precondition for using the `Arrays.binarySearch` method?
??x
The data must be sorted prior to calling `Arrays.binarySearch`. Binary search operates on the principle of divide and conquer, where it splits the search space into halves until the target element is found or determined not to exist within the array.

```java
import java.util.Arrays;

public class BinarySearchExample {
    private static final int NEEDLE = 50; // Example needle value
    private static final int[] haystack = {1, 3, 7, 9, 22, 45, 63, 88}; // Sorted example array

    public boolean findNeedle() {
        Arrays.sort(haystack); // Ensure the data is sorted first
        int i = Arrays.binarySearch(haystack, NEEDLE);
        if (i >= 0) {
            System.out.println("Value " + NEEDLE + " occurs at haystack[" + i + "]");
            return true;
        } else {
            System.out.println("Value " + NEEDLE + " does not occur in haystack; nearest value is " + haystack[-(i+2)] + " (found at " + -(i+2) + ")");
            return false;
        }
    }
}
```
x??

---

#### Converting a Collection to an Array
Background context: This topic discusses the process of converting a `Collection` object into a Java language array using the `toArray()` method. The `toArray()` method can be called on a collection and optionally accepts an array as an argument, which affects the type of the returned array.

:p How do you convert a Collection to a Java array?
??x
You use the `toArray()` method available in all Collection implementations. By default, it returns an `Object[]`. You can also pass an existing array to this method to fill and return that array or allocate a new one if necessary based on the size of the collection.

```java
import java.util.ArrayList;
import java.util.Collections;

public class CollectionToArrayExample {
    public static void main(String[] args) {
        ArrayList<Integer> list = new ArrayList<>();
        // Add elements to the list...
        
        Integer[] array = list.toArray(new Integer[list.size()]); // Type safe conversion
        
        for (Integer element : array) {
            System.out.println(element);
        }
    }
}
```
x??

---

