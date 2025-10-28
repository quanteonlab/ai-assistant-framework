# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 37)

**Starting Chapter:** 8.4 PolymorphismAbstract Methods. Problem. Solution. Discussion

---

#### Adding Items to a Building Management System
Background context: The provided Java program demonstrates adding various items like lights and computer components to a building management system. This showcases polymorphism and abstract methods usage in Java.

:p How does the `BuildingManagement` class add items, and what is printed when you run this code?
??x
The `BuildingManagement` class uses an `things.add(thing)` method to add various objects like `RoomLights`, `EmergencyLight`, and `ComputerCPU`. When you run the program, it prints each item being added as follows:

```java
public void add(Light thing) {
    System.out.println("Adding " + thing);
    things.add(thing);
}
```

The output reflects that all items are being added to the management system.

x??

---
#### Polymorphism in Java
Background context: The text explains how polymorphism works with abstract methods and subclasses. It demonstrates using a `Shape` class as a parent, which has an abstract method `computeArea()`, allowing different subclasses like `Rectangle` and `Circle` to implement their own versions.

:p What is the purpose of making a method abstract in Java?
??x
The purpose of making a method abstract in Java is to ensure that any subclass implementing this class must provide its specific implementation. This is useful for ensuring that certain methods are implemented, but leaving the exact logic to be defined by each subclass.

Example:
```java
public abstract class Shape {
    protected int x, y;
    public abstract double computeArea();
}

public class Rectangle extends Shape {
    double width, height;
    public double computeArea() {
        return width * height;
    }
}
```

x??

---
#### Abstract Method in `Shape`
Background context: The `Shape` class has an abstract method `computeArea()` which must be implemented by any subclass. This is used to calculate the area of different shapes like a rectangle or circle.

:p How does Java ensure that subclasses implement the `computeArea()` method?
??x
Java ensures that subclasses implement the `computeArea()` method by declaring it as `abstract` in the parent class (`Shape`). When you attempt to create an instance of any subclass, the compiler enforces that the abstract method is implemented.

Example:
```java
public abstract class Shape {
    protected int x, y;
    public abstract double computeArea();
}

public class Rectangle extends Shape {
    double width, height;
    @Override
    public double computeArea() {
        return width * height;
    }
}
```

x??

---
#### Polymorphism in `ShapeDriver`
Background context: The `ShapeDriver` class iterates over a collection of `Shape` objects and calls the `computeArea()` method on each, demonstrating polymorphism. This allows for a flexible design where specific implementations are handled within subclasses.

:p How does the `totalAreas()` method work to calculate areas using polymorphism?
??x
The `totalAreas()` method works by iterating over all `Shape` objects in the collection and calling their `computeArea()` methods, which are overridden in respective subclasses. This leverages polymorphism to call the correct implementation based on the object's actual class.

Example:
```java
public double totalAreas() {
    double total = 0.0;
    for (Shape s : allShapes) {
        total += s.computeArea();
    }
    return total;
}
```

x??

---

#### Typesafe Enumerations in Java
Background context explaining that Java introduced a more type-safe and flexible way to handle enums compared to C. Java enums are implemented as classes, subclassed from `java.lang.Enum`. They offer better readability and safety than using `final int` constants or custom enum classes.
If applicable, add code examples with explanations:
```java
// Incorrect usage in C
enum { BLACK, RED, ORANGE } color;
enum { READ, UNREAD } state; /*ARGSUSED*/
int main(int argc, char *argv[]) {
    color = RED;
    color = READ; // This will compile but give bad results in C

    return 0;
}

// Correct usage in Java
public enum Color {
    BLACK, RED, ORANGE
}

public enum State {
    READ, UNREAD
}
```
:p What is a key difference between Java enums and C enums when handling values?
??x
Java enums are type-safe; you cannot accidentally use values other than those defined for the given enumeration. In contrast, C enums can be used in any integer context, leading to potential bugs.
In Java, using `enum` ensures that only valid enum constants can be assigned, making the code more reliable and maintainable.
```java
public enum Media {
    BOOK, MUSIC_CD, MUSIC_VINYL, MOVIE_VHS, MOVIE_DVD;
}
```
x??

---
#### Using Enums in Java for Polymorphism
Background context explaining that enums in Java can implement interfaces, have constructors, fields, and methods. They are classes that extend `java.lang.Enum`. This allows for polymorphic behavior.
:p How can you use enums to achieve polymorphism?
??x
Enums in Java can be used to achieve polymorphism by implementing interfaces or adding methods. For example:
```java
public enum Media {
    BOOK, MUSIC_CD, MUSIC_VINYL, MOVIE_VHS, MOVIE_DVD;

    // Method overriding
    public String toString() {
        return "Default: " + name();
    }
}

// Overriding in specific enums
public enum MediaFancy {
    BOOK {
        @Override
        public String toString() {
            return "Book";
        }
    },
    MUSIC_CD,
    MUSIC_VINYL,
    MOVIE_VHS,
    MOVIE_DVD;

    // Additional methods can be added to each enum constant
}
```
x??

---
#### Iterating Over Enums in Java
Background context explaining that enums are iterable, which means you can iterate over them using a for-each loop. This is useful for performing actions on all possible values of an enum.
:p How can you iterate over the elements of an enum in Java?
??x
You can iterate over the elements of an enum by calling the `values()` method, which returns an array of all enum constants. You can then use a for-each loop to iterate through them:
```java
public class Product {
    public static void main(String[] args) {
        Media[] data = {Media.BOOK, Media.MOVIE_DVD, Media.MUSIC_VINYL};
        for (Media mf : data) {
            System.out.println(mf);
        }
    }
}
```
x??

---
#### Enum Constants and Namespaces
Background context explaining that each enum type in Java has its own separate namespace. This means you don't have to prefix constants with a name, making the code cleaner.
:p Why is it beneficial to use namespaces for enums in Java?
??x
Using namespaces for enums in Java prevents naming conflicts. Each enum type lives in its own namespace, so you don't need to prefix each constant with some sort of class name. This makes the code more readable and maintainable.
For example:
```java
public enum Media {
    BOOK, MUSIC_CD, MUSIC_VINYL, MOVIE_VHS, MOVIE_DVD;
}
```
You can directly use `Media.BOOK` instead of a longer qualified name like `MusicBook.BOOK`.
x??

---
#### Enums and Switch Statements in Java
Background context explaining that enums in Java support the use of switch statements. This makes it easy to handle different cases based on an enum value.
:p How can you use switch statements with enums in Java?
??x
Enums in Java are compatible with switch statements, allowing for clean and readable code when handling different cases:
```java
public class Product {
    String title;
    String artist;
    Media media;

    public Product(String artist, String title, Media media) {
        this.title = title;
        this.artist = artist;
        this.media = media;
    }

    @Override
    public String toString() {
        switch (media) {
            case BOOK:
                return title + " is a book";
            case MUSIC_CD:
                return title + " is a CD";
            case MUSIC_VINYL:
                return title + " is a relic of the age of vinyl";
            case MOVIE_VHS:
                return title + " is on old video tape";
            case MOVIE_DVD:
                return title + " is on DVD";
            default:
                return title + ": Unknown media " + media;
        }
    }
}
```
x??

---

#### EnumList.java Class
Background context: The provided Java program `EnumList` demonstrates the usage of enums. Enums provide a way to define a set of named values, which can be used as a type and passed around like other variables.

:p What does the `State` enum in the `EnumList` class represent?
??x
The `State` enum represents different states that an object or system might have, such as ON, OFF, and UNKNOWN. Enums are useful for defining a fixed set of constants.
```java
enum State {
    ON,
    OFF,
    UNKNOWN
}
```
x??

---

#### Optional Class in Java 8
Background context: The `Optional` class was introduced in Java 8 to address issues related to null references, especially NullPointerExceptions (NPEs). It provides a way to handle the absence of a value gracefully without resorting to null checks.

:p What is the purpose of using the `Optional` class?
??x
The purpose of using the `Optional` class is to avoid NullPointerExceptions by providing a mechanism to represent an optional presence of a value. It helps in making code more robust and readable.
```java
Optional<String> opt = Optional.of("What a day.");
```
x??

---

#### Creating Optional Objects
Background context: The `Optional` class can be created using different factory methods, such as `empty()`, `of(T obj)`, and `ofNullable(T obj)`.

:p How do you create an empty `Optional` object?
??x
You can create an empty `Optional` object by calling the `Optional.empty()` method.
```java
Optional<String> optEmpty = Optional.empty();
```
x??

---

#### Checking for Present Value
Background context: One of the main operations on `Optional` is checking if it contains a value using methods like `isEmpty()` and `isPresent()`.

:p How do you check if an `Optional` object contains a present value?
??x
You can check if an `Optional` object contains a present value by calling the `isPresent()` method.
```java
if (opt.isPresent()) {
    System.out.println("Value is " + opt.get());
} else {
    System.out.println("Value is not present.");
}
```
x??

---

#### Using orElse Method
Background context: The `orElse(T)`, `orElseGet(Supplier<? extends T>)`, and `orElseThrow(Supplier<Throwable>)` methods in `Optional` provide a default value if the `Optional` is empty.

:p How do you use the `orElse()` method to provide a default value?
??x
You can use the `orElse(T)` method to provide a default value when an `Optional` object is empty.
```java
System.out.println("Value is " + opt.orElse("not present"));
```
x??

---

#### Passing Values into Methods Using Optional
Background context: Wrapping values in `Optional` objects before passing them to methods can help manage null values more gracefully.

:p How does wrapping a value in an `Optional` object affect method calls?
??x
Wrapping a value in an `Optional` object allows you to handle the absence of a value more gracefully. You can use methods like `orElse()`, `ifPresent(Consumer<? super T>)`, and others to manage null values without explicitly checking for null.
```java
List.of(
    new Item("Item 1", LocalDate.now().plusDays(7)),
    new Item("Item 2")
).forEach(System.out::println);
```
x??

---

#### `Item` Class in Example 8-5
Background context: The `Item` class is a static inner class used to demonstrate the usage of `Optional` with date values that might be missing.

:p How does the `Item` class handle `dueDate`?
??x
The `Item` class handles `dueDate` by wrapping it in an `Optional<LocalDate>` object, allowing it to represent a potentially missing due date.
```java
static class Item {
    String name;
    Optional<LocalDate> dueDate;

    Item(String name) { this(name, null); }
    Item(String name, LocalDate dueDate) { this.name = name; this.dueDate = (dueDate == null) ? Optional.empty() : Optional.of(dueDate); }
}
```
x??

---

#### Summary of Key Concepts
Background context: This section covers the use of enums and `Optional` in Java 8 to manage values that might be absent, providing a safer and more expressive way of handling null references.

:p What are some key takeaways from this text?
??x
Key takeaways include understanding how to use enums for fixed sets of constants, and how to handle potentially missing values using the `Optional` class. These techniques help prevent NullPointerExceptions and make code more robust and readable.
```java
// Example usage in a real-world context
List.of(
    new Item("Item 1", LocalDate.now().plusDays(7)),
    new Item("Item 2")
).forEach(System.out::println);
```
x??

---

