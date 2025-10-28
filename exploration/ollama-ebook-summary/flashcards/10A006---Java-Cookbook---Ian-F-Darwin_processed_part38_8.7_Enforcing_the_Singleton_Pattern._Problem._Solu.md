# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 38)

**Starting Chapter:** 8.7 Enforcing the Singleton Pattern. Problem. Solution. Discussion

---

#### Using Optional to Handle Nullable Values
Background context: The `Optional` class is used to avoid null pointer exceptions and make it explicit when a value might be absent. It encapsulates an object that may or may not be present, providing utilities for dealing with optional values.

Code example:
```java
public class Item {
    private String name;
    private Optional<String> dueDate;

    public Item(String name, String dueDate) {
        this.name = name;
        this.dueDate = Optional.ofNullable(dueDate);
    }

    @Override
    public String toString() {
        return String.format("%s: %s", name, 
            dueDate.isPresent() ? "Item is due on " + dueDate.get() : 
            "Sorry, do not know when item is due");
    }
}
```
:p How can you use the `Optional` class to handle a nullable value?
??x
The `Optional` class helps manage cases where a variable might be null by wrapping it. If the value is present, methods like `.get()` can access its content safely; otherwise, it handles absence gracefully.

```java
// Example usage
String dueDate = "2023-12-31";
Item item = new Item("Holiday", dueDate);
System.out.println(item.toString()); // Output: Holiday: Item is due on 2023-12-31

dueDate = null;
item = new Item("Meeting", dueDate);
System.out.println(item.toString()); // Output: Meeting: Sorry, do not know when item is due
```
x??

---

#### Singleton Pattern Implementation Using Enum
Background context: The Singleton pattern ensures that a class has only one instance and provides a global point of access to it. This is useful for managing shared resources or controlling access to a single object.

Code example:
```java
public enum EnumSingleton {
    INSTANCE;

    public String demoMethod() {
        return "demo";
    }
}
```
:p How can you implement the Singleton pattern using an enum?
??x
Using an enum is one of the easiest ways to enforce the Singleton pattern. The `INSTANCE` keyword ensures that only one instance of the enum exists, and methods within it are protected by this singleton-ness.

```java
// Demonstrate the enum method:
System.out.println(EnumSingleton.INSTANCE.demoMethod());
```
x??

---

#### Singleton Pattern Implementation Using Private Constructor and Static Factory Method
Background context: Another common way to implement the Singleton pattern is through a private constructor and a static factory method. This approach ensures that no other instance can be created except by calling the provided method.

Code example:
```java
public class Singleton {
    // Static Initializer block runs before class becomes available for code, avoiding broken lazy initialization anti-pattern.
    private static Singleton instance = new Singleton();

    // Private Constructor prevents any other class from instantiating.
    private Singleton() {}

    // Static 'instance' method
    public static Singleton getInstance() {
        return instance;
    }

    // Other methods protected by singleton-ness would be here...
    public String demoMethod() {
        return "demo";
    }
}
```
:p How can you implement the Singleton pattern with a private constructor and a static factory method?
??x
Implementing the Singleton pattern using a private constructor and a static factory method ensures that only one instance of the class is created. The `getInstance()` method provides access to this single instance.

```java
// Demonstrate the singleton method:
System.out.println(Singleton.getInstance().demoMethod());
```
x??

---

---
#### Singleton Pattern Overview
Singletons ensure that a class has only one instance, and provide a global point of access to it. This pattern is useful when exactly one object is needed to coordinate actions across the system.

In Java, singleton instances are typically created using lazy initialization (not explicitly shown in the provided code snippet), where the object is not instantiated until its first use.
:p How does Java handle singleton instance creation?
??x
Java handles singleton instance creation through lazy loading. The Singleton instance is created only when `getInstance()` is called for the first time, and subsequent calls return the same instance.

Example of a simple Singleton class:
```java
public class Singleton {
    private static Singleton uniqueInstance;

    // Private constructor to prevent instantiation from outside
    private Singleton() {}

    public static Singleton getInstance() {
        if (uniqueInstance == null) {
            uniqueInstance = new Singleton();
        }
        return uniqueInstance;
    }

    public String demoMethod () {         return "demo";     }
}
```
x??

---
#### Why Not Use Lazy Evaluation for Singleton?
While Java lazily loads classes, the singleton instance is not deferred to the first use of its methods. The Singleton's constructor is called when `getInstance()` is first invoked.

Note that `Singleton.getInstance().demoMethod();` will call the constructor and return "demo" immediately.
:p Why isn't lazy evaluation necessary for singletons in Java?
??x
Java's class loading mechanism ensures that classes are loaded only once, but this doesn't defer the construction of singleton instances until their first use. The Singleton instance is created when `getInstance()` is called, not lazily.

Example demonstrating Singleton creation:
```java
public class Singleton {
    private static Singleton uniqueInstance;

    // Private constructor to prevent instantiation from outside
    private Singleton() {}

    public static Singleton getInstance() {
        if (uniqueInstance == null) {
            synchronized(Singleton.class) {  // Ensures thread safety
                if (uniqueInstance == null) {
                    uniqueInstance = new Singleton();
                }
            }
        }
        return uniqueInstance;
    }

    public String demoMethod () {         return "demo";     }
}
```
x??

---
#### Code-Based Singleton with Cloning Prevention
A code-based Singleton can be made final to prevent subclassing. Additionally, providing a `clone()` method that throws an exception prevents accidental cloning.

However, making the constructor private and the class final is sufficient.
:p Should a code-based Singleton provide a public final clone() method?
??x
No, a code-based Singleton does not need to provide a public final `clone()` method because preventing subclassing and cloning can be achieved by simply having the constructor as private and the class as final. Subclasses that attempt to override or call `clone()` will fail due to access restrictions.

Example of a final Singleton:
```java
public final class Singleton {
    private static Singleton uniqueInstance;

    // Private constructor to prevent instantiation from outside
    private Singleton() {}

    public static Singleton getInstance() {
        if (uniqueInstance == null) {
            synchronized(Singleton.class) {  // Ensures thread safety
                if (uniqueInstance == null) {
                    uniqueInstance = new Singleton();
                }
            }
        }
        return uniqueInstance;
    }

    public String demoMethod () {         return "demo";     }
}
```
x??

---
#### Custom Exceptions in Java
Creating application-specific exceptions is a good practice. You can subclass `Exception` or `RuntimeException` to provide more meaningful error handling.

Custom exception classes should be named descriptively and should have constructors that accept the necessary parameters.
:p How do you create an application-specific exception class?
??x
To create an application-specific exception class, you can extend either `Exception` (checked) or `RuntimeException` (unchecked). Here's how to create a custom unchecked exception:

```java
public class MyApplicationException extends RuntimeException {
    public MyApplicationException(String message) {
        super(message);
    }

    // Optionally add more constructors for different scenarios
}
```

This allows you to throw and catch exceptions that are specific to your application, making the code easier to understand.

Example usage:
```java
try {
    // Some operation that might fail
} catch (MyApplicationException e) {
    System.out.println("Caught a MyApplicationException: " + e.getMessage());
}
```
x??

---

---

#### Subclassing Throwable Directly

Background context: In theory, you could subclass `Throwable` directly. However, it is generally considered poor practice to do so.

:p What are the reasons why subclassing `Throwable` directly is discouraged?
??x
Subclassing `Throwable` directly is discouraged because `Throwable` is not intended for direct subclassing. Instead, it is more conventional and appropriate to subclass either `Exception` or `RuntimeException`. Subclasses of `Exception` (checked exceptions) require applications to handle them explicitly by catching the exception or throwing it upward via a method's `throws` clause.
x??

---

#### Extending Exception

Background context: Typically, you would extend `Exception` for checked exceptions. Checked exceptions are those that an application developer is required to catch or throw upward.

:p What constructors should be provided when extending `Exception`?
??x
When extending `Exception`, it is customary to provide at least the following constructors:

1. A no-argument constructor.
2. A one-string argument constructor.
3. A two-argument constructor—a string message and a `Throwable cause`.

Example code:
```java
public class ChessMoveException extends Exception {
    private static final long serialVersionUID = 802911736988179079L;

    public ChessMoveException() {
        super();
    }

    public ChessMoveException(String msg) {
        super(msg);
    }

    public ChessMoveException(String msg, Exception cause) {
        super(msg, cause);
    }
}
```
x??

---

#### Extending RuntimeException

Background context: `RuntimeException` is used for unchecked exceptions. These do not need to be declared in the method signature of methods that throw them.

:p What constructors should be provided when extending `RuntimeException`?
??x
When extending `RuntimeException`, you typically provide at least the following constructors:

1. A no-argument constructor.
2. A one-string argument constructor.

Example code:
```java
public class IllegalMoveException extends RuntimeException {
    public IllegalMoveException() {
        super();
    }

    public IllegalMoveException(String msg) {
        super(msg);
    }
}
```
x??

---

#### Stack Trace and Cause

Background context: If the code receiving an exception performs a stack trace operation on it, the cause will appear with a prefix such as "Root Cause is".

:p What happens when you perform a stack trace on a `ChessMoveException`?
??x
When you perform a stack trace on a `ChessMoveException`, if it has been created with a two-argument constructor that includes a `cause`, the `cause` will be displayed in the stack trace with a prefix such as "Root Cause is". This helps to understand the underlying reason for the exception.

Example code:
```java
public class Main {
    public static void main(String[] args) {
        try {
            throw new ChessMoveException("Illegal move detected", new RuntimeException("Internal error"));
        } catch (ChessMoveException e) {
            e.printStackTrace();
        }
    }
}
```
Output:
```
Root Cause is: java.lang.RuntimeException: Internal error
java.lang.RuntimeException: Internal error
    at oo.ChessMoveException.<init>(ChessMoveException.java:10)
    at oo.Main.main(Main.java:7)
```

x??

---

#### Using Predefined Exception Subclasses

Background context: The `javadoc` documentation lists many subclasses of `Exception`. You should check there first to see if a predefined exception subclass fits your needs.

:p How can you check for available predefined exception subclasses?
??x
You can use the Javadoc documentation or online resources like the official Java API documentation to explore and identify appropriate predefined exception subclasses. For example, in the case of `ChessMoveException`, you might find that there is a similar predefined subclass such as `IllegalArgumentException` or another suitable exception type.

Example:
```java
public class Main {
    public static void main(String[] args) {
        try {
            throw new IllegalArgumentException("Invalid move");
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        }
    }
}
```
Output:
```
java.lang.IllegalArgumentException: Invalid move
    at Main.main(Main.java:7)
```

x??

---

