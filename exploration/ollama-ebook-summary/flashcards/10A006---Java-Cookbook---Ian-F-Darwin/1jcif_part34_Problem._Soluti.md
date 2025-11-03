# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 34)

**Starting Chapter:** Problem. Solution. Discussion

---

#### ArrayList to Object Array Conversion

Background context: In Java, when converting a collection like `ArrayList` to an array of objects, you need to ensure that the array type is compatible with the elements stored in the list. Otherwise, an `ArrayStoreException` will be thrown.

Example code:
```java
List<String> list = new ArrayList<>();
list.add("Blobbo");
list.add("Cracked");
list.add("Dumbo");

// Convert a collection to Object[], which can store objects of any type.
Object[] ol = list.toArray();
System.out.println("Array of Object has length " + ol.length);

String[] sl = (String[]) list.toArray(new String[0]);
System.out.println("Array of String has length " + sl.length);
```

:p How does the `toArray()` method work in converting a collection to an array?
??x
The `toArray()` method converts a collection into an array. If no argument is provided, it returns an `Object[]` that can store objects of any type. When you cast this object array back to a specific type (like `String[]`), you ensure type safety.

For example:
```java
List<String> list = new ArrayList<>();
list.add("Blobbo");
list.add("Cracked");
list.add("Dumbo");

// Convert to Object[]
Object[] ol = list.toArray();
System.out.println(Arrays.toString(ol)); // [Blobbo, Cracked, Dumbo]

// Explicitly convert and cast to String[]
String[] sl = (String[]) list.toArray(new String[0]);
System.out.println(Arrays.toString(sl)); // [Blobbo, Cracked, Dumbo]
```
x??

---
#### Making Your Data Iterable

Background context: To make your data structure iterable in Java, you need to implement the `Iterable` interface. This involves providing an implementation of the `iterator()` method that returns an `Iterator`.

Example code:
```java
static class Demo implements Iterable<String> {
    String[] data = { "One", "Two", "Three" };

    class DemoIterator implements Iterator<String> {
        int i = 0;

        public boolean hasNext() {
            return i < data.length;
        }

        public String next() {
            return data[i++];
        }

        public void remove() {
            throw new UnsupportedOperationException("remove");
        }
    }

    public Iterator<String> iterator() {
        return new DemoIterator();
    }
}
```

:p How does the `Iterable` interface make your class iterable?
??x
The `Iterable` interface allows a class to be used in a "foreach" loop. By implementing this interface, you must provide an implementation of the `iterator()` method that returns an instance of the `Iterator`.

Example:
```java
public class IterableDemo {
    static class Demo implements Iterable<String> {
        String[] data = { "One", "Two", "Three" };

        class DemoIterator implements Iterator<String> {
            int i = 0;

            public boolean hasNext() {
                return i < data.length;
            }

            public String next() {
                return data[i++];
            }

            public void remove() {
                throw new UnsupportedOperationException("remove");
            }
        }

        public Iterator<String> iterator() {
            return new DemoIterator();
        }
    }

    public static void main(String[] args) {
        Demo demo = new Demo();
        for (String s : demo) {
            System.out.println(s);
        }
    }
}
```
x??

---
#### Iterator Interface

Background context: The `Iterator` interface is used to traverse elements in a collection. It has three methods: `hasNext()`, `next()`, and `remove()`.

Example code:
```java
public class IterableDemo {
    static class Demo implements Iterable<String> {
        String[] data = { "One", "Two", "Three" };

        class DemoIterator implements Iterator<String> {
            int i = 0;

            public boolean hasNext() {
                return i < data.length;
            }

            public String next() {
                return data[i++];
            }

            public void remove() {
                throw new UnsupportedOperationException("remove");
            }
        }

        public Iterator<String> iterator() {
            return new DemoIterator();
        }
    }
}
```

:p What are the methods of the `Iterator` interface?
??x
The `Iterator` interface has three main methods:
- `hasNext()` - returns true if there is a next element in the iteration.
- `next()` - returns the next element from the collection and advances the cursor position.
- `remove()` - removes the last element returned by the iterator. Note that this method can throw an `UnsupportedOperationException` as it is not mandatory to support this operation.

Example implementation:
```java
class DemoIterator implements Iterator<String> {
    int i = 0;

    public boolean hasNext() {
        return i < data.length;
    }

    public String next() {
        return data[i++];
    }

    public void remove() {
        throw new UnsupportedOperationException("remove");
    }
}
```
x??

---
#### Array Iterator Example

Background context: The `ArrayIterator` class is a simple implementation of the `Iterator` interface for an array. This can be used in "foreach" loops.

Example code:
```java
import com.darwinsys.util.ArrayIterator;

public class ArrayIteratorDemo {
    private final static String[] names = { "rose", "petunia", "tulip" };

    public static void main(String[] args) {
        ArrayIterator<String> arrayIterator = new ArrayIterator<>(names);
        System.out.println("Java 5, 6 way");
        for (String s : arrayIterator) {
            System.out.println(s);
        }
        System.out.println("Java 5, 6 ways");
        arrayIterator.forEach(s -> System.out.println(s));
        arrayIterator.forEach(System.out::println);
    }
}
```

:p How can you use an `ArrayIterator` in a foreach loop?
??x
You can use the `ArrayIterator` to traverse elements of an array using a "foreach" loop. The `forEach()` method allows you to pass a lambda expression or a method reference for processing each element.

Example usage:
```java
import com.darwinsys.util.ArrayIterator;

public class ArrayIteratorDemo {
    private final static String[] names = { "rose", "petunia", "tulip" };

    public static void main(String[] args) {
        ArrayIterator<String> arrayIterator = new ArrayIterator<>(names);
        System.out.println("Java 5, 6 way");
        for (String s : arrayIterator) {
            System.out.println(s);
        }
        System.out.println("Java 5, 6 ways");
        arrayIterator.forEach(s -> System.out.println(s));
        arrayIterator.forEach(System.out::println);
    }
}
```
x??

---

#### Java 8 Iterable.foreach
Background context: In Java 8, `foreach` was added to the `Iterator` interface as a default method. This allows for easier iteration over collections using lambda expressions without needing to manually implement an iterator.

:p How does `foreach` work in Java 8 with iterators?
??x
`foreach` is a method in the `Iterator` interface that allows you to iterate through elements of a collection or array, using a lambda expression. It simplifies the process by abstracting away manual iteration logic and leveraging lambda expressions for concise code.

Example usage:
```java
public class Example {
    public static void main(String[] args) {
        List<String> list = Arrays.asList("Java", "Scala", "Kotlin");
        
        // Using foreach with Iterator and lambda expression
        Iterator<String> iterator = list.iterator();
        while (iterator.hasNext()) {
            iterator.forEachRemaining(s -> System.out.println(s));
        }
    }
}
```
x??

---

#### Custom Stack Implementation: ToyStack
Background context: `ToyStack` is a simple class for stacking values of the primitive type int. It provides basic stack operations like push, pop, and peek.

:p What are the main methods provided by `ToyStack`?
??x
The main methods provided by `ToyStack` include:
- `push(int n)`: Adds an element onto the stack.
- `pop()`: Returns and removes the top element from the stack.
- `peek()`: Returns the top element without removing it.

Example of using `ToyStack`:
```java
public class ToyStack {
    protected int MAX_DEPTH = 10;
    protected int depth = 0;
    protected int[] stack = new int[MAX_DEPTH];

    public void push(int n) { 
        stack[depth++] = n; 
    }

    public int pop() { 
        return stack[--depth]; 
    }

    public int peek() { 
        return stack[depth - 1]; 
    }
}
```
x??

---

#### Generic Stack Interface: SimpleStack
Background context: `SimpleStack` is an interface that defines the basic operations of a stack, making it easier to implement different types of stacks. It uses generics to support any type of data.

:p What are the main methods defined in the `SimpleStack` interface?
??x
The main methods defined in the `SimpleStack` interface include:
- `empty()`: Returns true if the stack is empty.
- `push(T n)`: Adds an element onto the stack.
- `pop()`: Returns and removes the top element from the stack.
- `peek()`: Returns the top element without removing it.

Example usage of `SimpleStack` interface in a class:
```java
public class MyStack<T> implements SimpleStack<T> {
    private int depth = 0;
    public static final int DEFAULT_INITIAL = 10;

    @Override
    public boolean empty() { 
        return depth == 0; 
    }

    @Override
    public void push(T obj) {
        stack[depth++] = obj; 
    }

    @Override
    public T pop() {
        --depth;
        T tmp = stack[depth];
        stack[depth] = null;
        return tmp; 
    }

    @Override
    public T peek() { 
        if (depth == 0) { 
            return null; 
        } 
        return stack[depth - 1]; 
    }
}
```
x??

---

#### Using a Stack of Objects: MyStack
Background context: `MyStack` is an implementation class that adheres to the `SimpleStack` interface and supports generic types. It includes additional methods for error checking, like `hasRoom()`.

:p What additional method does `MyStack` provide beyond those in `SimpleStack`?
??x
`MyStack` provides an additional method called `hasRoom()`, which checks if there is enough space to add more elements without exceeding the stack's capacity. This method helps prevent errors when pushing too many elements into a fixed-size stack.

Example usage of `hasRoom()`:
```java
public boolean hasRoom() {
    return depth < stack.length;
}
```
x??

---

#### Multidimensional Arrays in Java
Background context: In Java, arrays can hold any reference type and are themselves a reference type. This means that you can have arrays of arrays, known as multidimensional arrays. The length of each array within a multidimensional array does not need to be the same, providing flexibility.
:p How do you allocate and initialize a two-dimensional array in Java?
??x
You can allocate and initialize a two-dimensional array using loops or initializers. For example:
```java
public class ArrayTwoDObjects {
    public static String[][] getArrayInfo () {
        String info[][] = new String[10][10];
        for (int i=0; i < info.length; i++) {
            for (int j = 0; j < info[i].length; j++) {
                info[i][j] = "String[" + i + "," + j + "]";
            }
        }
        return info;
    }

    public static void main(String[] args) {
        print("from getArrayInfo", getArrayInfo());
    }

    public static void print(String tag, String[][] array) {
        System.out.println("Array " + tag + " is " + array.length + " x " + array[0].length);
        for (int i = 0; i < 2; i++) { // Print selected elements
            for (int j = 0; j < 2; j++) {
                System.out.println("Array[" + i + "][" + j + "] = " + array[i][j]);
            }
        }
    }
}
```
x??

---
#### Accessing Elements in Multidimensional Arrays
Background context: Once a multidimensional array is created, you can access its elements using multiple indices. The first index refers to the row and subsequent indices refer to the columns.
:p How do you print selected elements from a two-dimensional array?
??x
You can selectively print elements by iterating through specific subscripts:
```java
public static void print(String tag, String[][] array) {
    System.out.println("Array " + tag + " is " + array.length + " x " + array[0].length);
    for (int i = 0; i < 2; i++) { // Print selected elements
        for (int j = 0; j < 2; j++) {
            System.out.println("Array[" + i + "][" + j + "] = " + array[i][j]);
        }
    }
}
```
x??

---
#### Multidimensional Array Lengths
Background context: Each sub-array within a multidimensional array can have different lengths, providing flexibility in how data is structured. The `length` attribute of each sub-array gives the number of elements it contains.
:p How does Java ensure that columns of a two-dimensional array do not all need to be the same length?
??x
Java ensures this by allowing each sub-array within a multidimensional array to have its own independent length. For example:
```java
public static String[][] getArrayInfo () {
    String info[][] = new String[10][10];
    for (int i=0; i < info.length; i++) {
        for (int j = 0; j < info[i].length; j++) {
            info[i][j] = "String[" + i + "," + j + "]";
        }
    }
    return info;
}
```
x??

---
#### Multidimensional Array Initialization
Background context: You can initialize a multidimensional array using loops or direct assignment. The initialization process involves assigning values to each element based on its indices.
:p How do you initialize and print a two-dimensional string array with specific values?
??x
You can initialize and print a two-dimensional string array as follows:
```java
public static String[][] getArrayInfo () {
    String info[][] = new String[10][10];
    for (int i=0; i < info.length; i++) {
        for (int j = 0; j < info[i].length; j++) {
            info[i][j] = "String[" + i + "," + j + "]";
        }
    }
    return info;
}

public static void main(String[] args) {
    print("from getArrayInfo", getArrayInfo());
}
```
x??

---
#### Array Length Attributes
Background context: In a two-dimensional array, the `length` attribute of each sub-array gives its size. This allows for flexible multidimensional data structures where rows and columns can have varying lengths.
:p How do you determine the length of each row in a two-dimensional array?
??x
You can use the `length` attribute of each sub-array to determine its size:
```java
public static void print(String tag, String[][] array) {
    System.out.println("Array " + tag + " is " + array.length + " x " + array[0].length);
}
```
x??

---

