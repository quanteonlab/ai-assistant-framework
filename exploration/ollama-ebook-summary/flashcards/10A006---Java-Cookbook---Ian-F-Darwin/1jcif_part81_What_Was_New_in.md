# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 81)

**Starting Chapter:** What Was New in Java 11 September 2018

---

#### Local Variable Type Inference
Background context: The `var` keyword was introduced as part of Java 10 to enable local variable type inference, making the code more concise. However, it has specific usage rules and limitations.
:p What is the issue with using `var` in this scenario?
??x
The error occurs because `var` cannot be used for local variables without an initializer. In such cases, you need to explicitly specify the data type or provide a variable initialization.
```java
var var = 123; // This works fine
// var huh; // This would cause an error without an initializer
```
x??

---

#### LocalDateTime Example
Background context: `LocalDateTime` is part of Java's DateTime API introduced in Java 8, allowing for precise date and time manipulations. The example demonstrates how to obtain the current local date and time.
:p How does `java.time.LocalDateTime.now()` work?
??x
The method `LocalDateTime.now()` returns a `LocalDateTime` object representing the current date and time on this system with the zone id of the default time-zone, in the format "2019-08-31T20:47:36.440491".
```java
var z = java.time.LocalDateTime.now();
```
x??

---

#### HashMap Example
Background context: `HashMap` is a part of Java's Collections Framework used to store key-value pairs in an unordered structure, with fast access and modification.
:p How do you add a key-value pair to a `HashMap`?
??x
To add a key-value pair to a `HashMap`, you use the `put` method. The example shows how to add "Meh" as a key with the value 123.
```java
var map = new HashMap<String, Integer>();
map.put("Meh", 123);
```
x??

---

#### Unmodifiable List vs CopyOf Method
Background context: Java's `List` provides an unmodifiable list view that does not allow modifications directly. The `copyOf()` method was introduced to create a truly unmodifiable copy of the list.
:p What is the difference between `unmodifiableList()` and `copyOf()`?
??x
The `unmodifiableList()` method returns an unmodifiable view of the underlying list, which means changes in the original list are reflected. The `copyOf()` method creates a new collection containing the elements of the specified collection as its elements; this new collection is unmodifiable.
```java
List<String> original = List.of("A", "B");
// old: Unmodifiable view
List<String> modifiableView = Collections.unmodifiableList(original);

// New: Truly unmodifiable copy
List<String> trulyUnmodifiableCopy = List.copyOf(original);
```
x??

---

#### Java 10 Deprecations and Removals
Background context: With each new version of Java, certain features are deprecated or removed to improve the language. The text mentions that many old features were removed or deprecated in Java 10.
:p What does it mean when a feature is deprecated?
??x
A deprecation warning suggests that a particular method, class, or other construct should not be used because it is outdated and will likely be removed in future versions of the language. The `@Deprecated` annotation marks methods, classes, constructors, or fields as deprecated.
```java
@Deprecated
public void oldMethod() {
    // Old implementation
}
```
x??

---

#### Additional Resources Mentioned
Background context: The text references additional resources for further reading and understanding about Java's changes and deprecations in version 10.
:p What can developers refer to learn more about the deprecated features in Java 10?
??x
Developers can refer to the official documentation, community forums, articles, and blog posts such as Simon Ritter’s article "Java 10 Pitfalls for the Unwary" and the list of removed or deprecated features on DZone.
x??

---

#### Single-File Run-from-Source (JEP 330)
Background context: Java 11 introduced a feature that allows for easier compilation and running of single-source files, making it simpler to work with small programs or scripts directly from source. This is particularly useful when you want to quickly test out a new piece of code without the overhead of setting up a full project structure.

:p What is JEP 330 in Java 11?
??x
JEP 330, also known as "Single-File Run-from-Source," enables users to run and compile a single source file directly using the `java` command. This means you can type something like `java HelloWorld.java`, which will both compile and execute your program.
```java
// Example of running a simple Java file
class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```
x??

---

#### Java 12 Preview Changes
Background context: In Java 12, the JDK introduced the concept of "Preview Changes." These are features that have been added to the JDK but are not yet part of the official specification. The idea is to get early feedback from users and make improvements based on real-world usage before finalizing them for release.

:p What is a key feature in Java 12 related to preview changes?
??x
Java 12 introduced "Preview Changes," which refers to features that have been added to the JDK but are not yet part of the official specification. These features can be enabled and used by developers, who can provide feedback during their development process.
```java
// Example of using a preview feature in Java 12 (switch expressions)
switch (expression) {
    case "value" -> System.out.println("Matched value");
}
```
x??

---

#### Java 12 Language Changes: Switch Expressions with Values
Background context: One of the significant language changes in Java 12 is the introduction of switch expressions that can yield a value. This feature simplifies code by allowing you to write more concise and readable logic for handling cases.

:p What new feature was introduced in Java 12 regarding switch statements?
??x
Java 12 introduced switch expressions with values, which allow the `switch` statement to return a value directly. This makes the syntax cleaner and allows developers to avoid using traditional if-else constructs.
```java
// Example of a switch expression in Java 12
public String getGreeting(String name) {
    return switch (name) {
        case "Alice" -> "Hello, Alice!";
        case "Bob" -> "Greetings, Bob!";
        default -> "How do you do?";
    };
}
```
x??

---

#### Java 12 API Changes: Tee Collector
Background context: Another significant API change in Java 12 is the introduction of a `Tee Collector`, which allows input to be copied into multiple output streams. This can be useful for logging or writing to multiple destinations simultaneously.

:p What new collector was introduced in Java 12?
??x
Java 12 introduced the `Tee Collector`, which enables copying data from one stream to multiple other streams. This feature is particularly useful for scenarios where you need to log or process data in multiple ways.
```java
// Example of using Tee Collector in Java 12
Stream<String> originalStream = Stream.of("A", "B", "C");
List<String> collector1 = new ArrayList<>();
List<String> collector2 = new ArrayList<>();

originalStream.collect(Collectors.toCollection(ArrayList::new), collector1::add, collector2::add);
// collector1 and collector2 will now contain the elements of the original stream
```
x??

---

#### Java 12 API Changes: CompactNumberFormat
Background context: The `CompactNumberFormat` class was introduced in Java 12 as a replacement for the older `ScaledNumberFormat`. This new format provides more concise and human-readable representations of numbers, such as printing `2048` as `2K`.

:p What is the purpose of CompactNumberFormat in Java 12?
??x
The purpose of `CompactNumberFormat` in Java 12 is to provide a more readable and compact way of formatting numbers. It replaces the older `ScaledNumberFormat` and can format large numbers in a shorter, more understandable form.
```java
// Example of using CompactNumberFormat in Java 12
NumberFormat formatter = NumberFormat.getCompactNumberInstance(NumberFormat.Style.SHORT);
System.out.println(formatter.format(2048)); // Outputs "2K"
```
x??

---

#### Java 12 API Changes: String.indent()
Background context: The `String` class gained a new method, `indent()`, in Java 12. This method allows you to add spaces before each line of the string, which can be useful for text formatting.

:p What is the `indent()` method in Java 12?
??x
The `indent()` method in Java 12 adds a specified number of spaces at the beginning of every line in a `String`. This can be particularly useful for aligning or indenting multi-line strings.
```java
// Example of using String.indent() in Java 12
String text = "This\nis\na\ntest";
System.out.println(text.indent(4)); // Outputs: 
        "This" + (line break)
        "is"   + (line break)
        "a"    + (line break)
        "test"
```
x??

---

#### Java 12 GC Improvements
Background context: Garbage Collection improvements in Java 12 include the introduction of Shenandoah, a low-pause-time garbage collector. Additionally, there are enhancements to G1 GC to improve pause times.

:p What is an important GC improvement introduced in Java 12?
??x
One of the important garbage collection (GC) improvements in Java 12 is the introduction of the Shenandoah collector, which focuses on reducing pause times during garbage collection. Additionally, there are enhancements made to the G1 garbage collector to improve its performance and reduce pause times.
```java
// Example configuration for enabling Shenandoah GC
-XX:+UnlockExperimentalVMOptions -XX:+UseShenandoahGC
```
x??

---

#### Improved Garbage Collection and AppCDS in Java 13

Java 13 introduced further enhancements to garbage collection (GC) mechanisms, making them more efficient. Application Class Data Sharing (AppCDS), a feature that allows writing an archive of all classes used in an application, was also improved.

:p What is the significance of AppCDS in Java 13?
??x
AppCDS helps in reducing startup time for applications by providing a preloaded class data archive. When this archive is present, the JVM can load these classes more quickly during startup, leading to faster performance.

Example scenario: Consider an application that starts up slower than desired because of the overhead associated with loading and initializing all necessary classes. By using AppCDS, you can generate a binary file containing the metadata and code for commonly used classes. During the initial JVM launch, this archive can be loaded, reducing the time required to start the application.

??x
```java
// Example pseudo-code to demonstrate how AppCDS might be used

public class Main {
    public static void main(String[] args) {
        // The JVM command line might include an option to use a pre-built AppCDS archive:
        // java -Xshare:dump -XX:+UseAppCDS -XX:CDSArchiveName=myarchive.cds
        // During application startup, the JVM would then use this archive to speed up class loading.
    }
}
```
x??

---

#### Record Types in Java 14

Record types were introduced as a preview feature for Java 14. They provide a concise way to declare data classes with fields and a constructor.

:p How do record types simplify the creation of simple data structures?
??x
Record types allow you to create data classes concisely by automatically generating constructors, getters, and other methods that would otherwise need to be manually defined. This simplifies the implementation and maintenance of such classes.

Example: Consider a `Point` class with two fields—`x` and `y`. Using record types, you can declare it as follows:

```java
record Point(int x, int y) {
    // Automatically generated constructor, getters, equals(), hashCode(), toString()
}
```

:p How does the Java compiler handle the automatic generation of methods for a record type?
??x
The Java compiler automatically generates standard methods such as constructors, getters (`getX()`, `getY()`), an implementation of `equals()` and `hashCode()`, a custom `toString()` method, and other utility methods. This reduces the boilerplate code needed to create simple data classes.

Example:
```java
record Point(int x, int y) {
    // Automatically generated constructor: new Point(10, 20)
    // Getter for 'x': public int getX() { return x; }
    // Equals and HashCode: @Override public boolean equals(Object o) { ... }
    // toString(): @Override public String toString() { return "Point{" + "x=" + x + ", y=" + y + '}'; }
}
```
x??

---

#### Sealed Types in Java 14

Sealed types are a preview feature for Java 14, which allow class designers to control subclassing by enumerating all the allowed subclasses. This helps in preventing unwanted or malicious inheritance.

:p What is the purpose of sealed types in Java?
??x
The primary purpose of sealed types is to restrict how a certain type can be subclassed. By defining a sealed type and listing permitted subtypes, you ensure that only specific classes can extend your class. This feature enhances code safety and maintainability by preventing accidental or malicious extensions.

Example: Consider the `Person` class, which could have two allowed subclasses—`Customer` and `SalesRep`.

```java
public abstract sealed class Person permits Customer, SalesRep {
    // Common fields and methods for all persons
}

class Customer extends Person {
    // Specific implementation details for customers
}
```

:p How does the syntax for defining a sealed type work?
??x
The syntax for defining a sealed type involves using the `sealed` keyword followed by the class definition, which includes a list of permitted subclasses in the `permits` clause.

Example:
```java
public abstract sealed class Person permits Customer, SalesRep {
    // Common fields and methods for all persons

    public Person() {
        // Constructor logic
    }

    void commonMethod() {
        // Common method implementation
    }
}

class Customer extends Person {
    // Specific implementation details for customers

    @Override
    void commonMethod() {
        super.commonMethod();
        // Additional customer-specific behavior
    }
}
```
x??

---

#### Text Blocks in Java 14

Text blocks, also known as multiline text strings, are a preview feature that allow declaring multi-line strings with triple double-quotes. They simplify the syntax for handling long or formatted string literals.

:p How do text blocks differ from traditional multi-line string literals?
??x
Text blocks provide a more readable and concise way to write multi-line strings by using three double quotes (`"""`) instead of escaping newline characters in single or double-quoted strings. This makes it easier to read and maintain long or formatted string content.

Example:
```java
String long = """
This is a long text String.
It can span multiple lines without requiring escape sequences.
"""
```

:p What are the benefits of using text blocks over traditional multi-line strings?
??x
Using text blocks offers several advantages, such as better readability and ease of maintenance. They eliminate the need for escaping newline characters (`\n`) or other special characters that might be present in the string content.

Example:
```java
String normal = "This is a long text String.\n" +
                "It can span multiple lines without requiring escape sequences.";
// vs.
String block = """
This is a long text String.
It can span multiple lines without requiring escape sequences.
"""
```
x??

---

#### jpackage Tool in Java 14

The `jpackage` tool, also previewed for Java 14, generates self-installing applications on supported operating systems. It helps in creating installers that package an application along with necessary libraries and resources.

:p What is the main function of the `jpackage` tool?
??x
The `jpackage` tool simplifies the process of packaging a Java application into a complete installer for various operating systems (e.g., Windows, macOS, Linux). It bundles all required dependencies, ensuring that the final package can run independently on different platforms.

Example:
```bash
# Example command to create an installer for a Java application
jpackage --input build/dist \
         --output dist/installers \
         --name MyApp \
         --type app-image \
         --arch x64
```

:p How does `jpackage` ensure that the final package runs independently?
??x
The `jpackage` tool includes all necessary libraries and resources required by your Java application, creating a self-contained bundle. This ensures that when users install the package, they do not need to have specific Java runtime environments or additional dependencies installed on their systems.

Example:
```bash
# Example command details
--input build/dist : Specifies the directory containing the application's JAR and resources.
--output dist/installers : Sets the output directory where the installer will be placed.
--name MyApp : Names the application for the user interface and installer files.
--type app-image : Indicates that a single executable image is to be created.
--arch x64 : Specifies the architecture (e.g., 32-bit or 64-bit) of the target platform.
```
x??

---

#### JavaFX and Its Role in Desktop Development

JavaFX was developed as an alternative to applets for desktop applications. It offers a rich set of UI controls, animations, and rendering capabilities.

:p How did JavaFX replace applets in client-side technologies?
??x
JavaFX replaced applets by providing a more robust and capable framework for building desktop applications. Unlike applets, which were limited by the security restrictions imposed by web browsers, JavaFX could run as standalone applications with greater flexibility.

Example:
- **Applet limitations**: Restricted to web environments, required browser permissions, and had limited access to system resources.
- **JavaFX capabilities**: Allowed full-screen applications, provided rich UI controls, supported direct rendering on various platforms, and was not restricted by the same-origin policy.

:p What are some key features of JavaFX?
??x
Key features of JavaFX include:

1. **Rich UI Controls**: A wide range of graphical user interface components.
2. **Animation**: Built-in support for creating smooth animations and transitions.
3. **Rendering**: High-quality rendering capabilities, including 2D and 3D graphics.
4. **Performance**: Optimized performance through native code and hardware acceleration.

Example:
```java
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class ExampleJavaFXApp extends Application {
    @Override
    public void start(Stage primaryStage) {
        // Create a scene with a rectangle as the root node.
        Rectangle rect = new Rectangle(100, 50);
        rect.setFill(Color.RED);

        Scene scene = new Scene(rect, 300, 250);

        primaryStage.setTitle("JavaFX Example");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
```
x??

---

#### Description Differentiation

- **Concept Title**: Focuses on the specific improvements in garbage collection and class data sharing.
- **Record Types**: Covers the new feature of record types in Java 14, which simplifies data class creation.
- **Sealed Types**: Describes sealed types for restricted subclassing.
- **Text Blocks**: Explains the syntax and benefits of using text blocks for multi-line strings.
- **jpackage Tool**: Details how to use `jpackage` for creating self-installing Java applications.

