# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 11)

**Starting Chapter:** Problem. Solution. Discussion

---

#### Introduction to Java Modules System
Background context explaining the introduction and purpose of the Java Module System. The system was designed for building large applications out of many small pieces, addressing issues beyond what Maven or Gradle handle.

:p What is the primary goal of the Java Modules System?
??x
The primary goal of the Java Modules System (JPMS) is to manage visibility of classes between different chunks of application code developed by potentially untrusted developers. It enhances access control at a finer granularity than traditional public, private, protected, and default modifiers.
x??

---

#### Comparison with Existing Build Tools
Explanation on how the Java Module System interacts with existing build tools like Maven or Gradle.

:p How does the Java Module System interact with existing build tools?
??x
The Java Module System works alongside existing build tools such as Maven or Gradle. You still need to configure dependencies, download them, and package them using your chosen build tool. However, the Modules system ensures that the visibility of classes between modules is controlled.
x??

---

#### Packages in Java
Explanation on how packages enhance class organization.

:p What are the benefits of using packages in Java?
??x
Packages in Java provide a way to logically group related classes and interfaces into namespaces. This helps in organizing code, reducing naming conflicts, and controlling access at the package level. For example:
```java
package com.example;
public class ExampleClass {
    // class members
}
```
x??

---

#### Modules vs Maven Modules
Clarification on the difference between Java modules and Maven’s concept of modules.

:p What is the key difference between Java modules and Maven’s modules?
??x
Java modules refer to a mechanism in the Java platform that controls visibility and dependencies at compile-time and runtime. Maven modules, on the other hand, are part of project organization where a project can be broken down into smaller subprojects for better management.
x??

---

#### Reflection and JPMS Warnings
Explanation on why warnings related to reflection might appear.

:p Why do I see warnings about illegal reflective access?
??x
These warnings occur when the Java Platform Module System (JPMS) checks that no types are accessed in encapsulated packages within a module. These warnings will diminish over time as more libraries and applications get modularized.
Example warning:
```
Illegal reflective access by com.foo.Bar
(file:/Users/ian/.m2/repository/com/foo/1.3.1/foo-1.3.1.jar)
to field java.util.Properties.defaults
```
x??

---

#### Using Deprecated Classes and Java Modules
Background context explaining how using deprecated classes can lead to compilation issues, especially with newer versions of Java. The example provided deals with the `sun.misc.Unsafe` class which is a part of JDK but not officially supported or guaranteed to be available in future releases.

:p Why does the code need to use the `jdk.unsupported` module when working with deprecated classes like `sun.misc.Unsafe`?
??x
The code needs to use the `jdk.unsupported` module because the `sun.misc.Unsafe` class is considered an internal proprietary API and may be removed in a future Java release. To compile and run this code, you must explicitly require the `jdk.unsupported` module in your `module-info.java`.

```java
// Content of module-info.java
module javasrc.unsafe {
    requires jdk.unsupported;  // Other required modules can also be specified here.
}
```

x??

---

#### Building and Running a Module with Maven
Background context explaining how to use Maven for building Java projects that utilize the new Java Modules system. The example demonstrates compiling, packaging, and running a module using Maven commands.

:p How do you build and run a Java module project using Maven?
??x
To build and run a Java module project using Maven, follow these steps:

1. Ensure your `module-info.java` is correctly configured to include the required modules.
2. Run the following Maven command to clean, compile, package, and test the project:
```shell
mvn clean package
```
3. Use the Java runtime with the appropriate classpath (`-cp` or `-classpath`) to run the module.

Example commands:

```shell
$mvn clean package
# Result shows successful build.
```

```shell$ java -cp target/classes unsafe/LoadAverage 3.54 1.94 1.62
# Output should match system uptime command output, e.g., "3.54 1.94 1.62".
```

x??

---

#### Module System and Java API Modules
Background context explaining the new module system in Java introduced in Java 9, which allows finer control over dependencies and isolation of code.

:p How do you list all available modules in a Java application?
??x
You can list all available modules using the `java --list-modules` command. This command provides a list of all standard and JDK modules that are currently loaded or discoverable on your system.

Example command:

```shell
$ java --list-modules
# Output lists various modules, e.g.,
java.base
java.compiler
java.datatransfer
...
```

x??

---

#### Requiring Specific Modules in Your Application
Background context explaining how to explicitly require specific Java API modules in a module-info.java file.

:p What do you need to include in your `module-info.java` if you want to use the Java Desktop API for graphical applications?
??x
To include the Java Desktop API (which includes AWT and Swing) in your module, you must add the following line to your `module-info.java`:

```java
// Content of module-info.java
module javasrc.unsafe {
    requires java.desktop;  // Other required modules can also be specified here.
}
```

This ensures that when you run your application, it has access to AWT and Swing components.

x??

---

#### Warnings from Using Proprietary APIs
Background context explaining the warnings related to using unsupported or internal APIs like `sun.misc.Unsafe`.

:p Why do you get "internal proprietary API" warnings when running Java code that uses deprecated classes?
??x
You receive "internal proprietary API" warnings because the classes and methods being used, such as `sun.misc.Unsafe`, are considered implementation details of the JVM. They may change or be removed in future versions of Java.

These warnings indicate potential issues with your application's compatibility across different Java releases but do not affect compilation or runtime execution. However, it is advisable to avoid using these APIs unless absolutely necessary and understand their potential risks.

x??

---

#### Using Reflection API
Background context explaining the use of reflection to access internal classes and methods that are not directly accessible through public interfaces.

:p How can you obtain an `Unsafe` instance using Java Reflection?
??x
You can obtain an `Unsafe` instance by leveraging Java's Reflection API. Here is how you do it:

1. Use the `Field` class to get a reference to the `theUnsafe` field in the `Unsafe` class.
2. Set the field accessible and then use it to get the `Unsafe` instance.

Example code:
```java
public static void main(String[] args) throws Exception {
    Field f = Unsafe.class.getDeclaredField("theUnsafe");
    f.setAccessible(true);
    Unsafe unsafe = (Unsafe) f.get(null);
    
    int nelem = 3;
    double loadAvg[] = new double[nelem];
    unsafe.getLoadAverage(loadAvg, nelem);

    for (double d : loadAvg) {
        System.out.printf(" %4.2f ", d);
    }
    System.out.println();
}
```

x??

---

#### Substring Method in Java
Background context: In Java, strings are immutable. However, you can create new substrings from existing ones using the `substring` method of the `String` class. This method returns a view of the specified portion of this string's characters.

The substring methods are overloaded:
- One-argument form: `substring(int beginIndex)` - Returns a new string that is the substring of this string starting at index beginIndex and extending to the end.
- Two-argument form: `substring(int beginIndex, int endIndex)` - Returns a new string that is the substring of this string starting at the specified beginIndex and ending at the specified endIndex (exclusive).

If the beginning index is less than zero or greater than the length of this string, then IllegalArgumentException is thrown. If end index is negative or if `beginIndex` > `endIndex`, an empty string is returned.

:p How does Java's substring method work?
??x
Java's `substring` method in the `String` class allows you to create a new string that consists of a portion of another string. The `substring(int beginIndex)` form returns characters from the specified index to the end, while the two-argument form `substring(int beginIndex, int endIndex)` specifies both start and end indices (end being exclusive).

Here's an example demonstrating its usage:
```java
public class SubStringDemo {
    public static void main(String[] av) {
        String a = "Java is great.";
        System.out.println(a);
        
        // Create substring starting at index 5 to the end
        String b = a.substring(5); 
        System.out.println(b); // prints: is great.
        
        // Create substring from index 5 to 7 (exclusive)
        String c = a.substring(5, 7);
        System.out.println(c); // prints: is
        
        // Create substring starting at index 5 to the end using length
        String d = a.substring(5, a.length());
        System.out.println(d); // prints: is great.
    }
}
```
x??

---

#### Tokenizing Strings with StringTokenizer
Background context: `StringTokenizer` in Java can be used to break a string into tokens based on specified delimiters. It implements the Iterator interface and provides methods like `hasMoreTokens()` and `nextToken()`. 

The basic usage involves creating a `StringTokenizer` object by passing the string and delimiter(s) as parameters, then using its methods to iterate over the tokens.

:p How can you use StringTokenizer in Java?
??x
You can use the `StringTokenizer` class in Java to break a string into tokens based on specified delimiters. The constructor of `StringTokenizer` takes the input string and one or more delimiters as parameters, and it returns an iterator that provides the tokens.

Here's how you can split a string "Hello World of Java" into tokens:
```java
import java.util.StringTokenizer;

public class StrTokDemo {
    public static void main(String[] args) {
        StringTokenizer st = new StringTokenizer("Hello World of Java");
        while (st.hasMoreTokens()) {
            System.out.println("Token: " + st.nextToken());
        }
    }
}
```
This code outputs:
```
Token: Hello
Token: World
Token: of
Token: Java
```

For cases where you need to preserve null fields or handle consecutive delimiters, you can pass `true` as the third argument to the constructor, which treats each delimiter as a token.
x??

---

#### Using Regular Expressions for Tokenization
Background context: Java provides powerful tools like regular expressions (regex) to split strings based on complex patterns. The `split()` method of the `String` class can be used with regex to tokenize strings.

The syntax for using `split()` is:
```java
String[] tokens = some_input_string.split(regex, limit);
```
Where `regex` is a regular expression and `limit` is optional - if specified, it limits the number of substrings that are returned.

:p How can you use regular expressions to split a string in Java?
??x
You can use regular expressions with the `split()` method of the `String` class to tokenize strings based on complex patterns. The `split()` method takes two parameters: the regex pattern and an optional limit parameter which specifies the maximum number of substrings.

For example, splitting a string "Hello World of Java" into words using spaces as delimiters:
```java
String[] words = "Hello World of Java".split(" ");
```
This will return `["Hello", "World", "of", "Java"]`.

To split on multiple spaces or tabs, you can use the regex pattern `\s+`:
```java
String[] tokens = "Hello  World\t of   Java".split("\\s+");
```
This will output: `["Hello", "World", "of", "Java"]`.
x??

---

#### Handling Consecutive Delimiters in StringTokenizer
Background context: When using `StringTokenizer`, by default, consecutive delimiters are treated as a single delimiter and the resulting tokens do not include these consecutive delimiters. However, you can preserve them by passing `true` to the constructor.

:p How does `StringTokenizer` handle multiple consecutive delimiters?
??x
By default, `StringTokenizer` treats multiple consecutive delimiters as one and discards adjacent delimiters. This behavior ensures that tokens are properly separated based on specified delimiters.

However, if you want to preserve these extra delimiters as part of the tokens, you can pass `true` as the third argument to the `StringTokenizer` constructor:
```java
StringTokenizer st = new StringTokenizer("Hello, World|of|Java", ", |", true);
```
This will ensure that consecutive delimiters are included in the output.

Here's an example demonstrating this behavior:
```java
import java.util.StringTokenizer;

public class StrTokDemo3 {
    public static void main(String[] args) {
        StringTokenizer st = new StringTokenizer("Hello, World|of|Java", ", |", true);
        while (st.hasMoreElements()) {
            System.out.println("Token: " + st.nextElement());
        }
    }
}
```
Output:
```
Token: Hello
Token: , 
Token: Token: 
Token: World
Token: | 
Token: of
Token: | 
Token: Java
```

This output shows that both delimiters and empty tokens are preserved.
x??

---

#### Processing Strings with Missing Fields Using StringTokenizer
Background context: In scenarios where input strings might have missing fields (e.g., "FirstName|LastName||Company|PhoneNumber"), using `StringTokenizer` with the appropriate delimiter settings can help in preserving these null fields.

:p How does `StringTokenizer` handle missing fields in a string?
??x
`StringTokenizer` can handle missing fields by treating consecutive delimiters as tokens. When you pass `true` to its constructor, it ensures that every delimiter is treated as a token, even if they are consecutive.

For example, processing a line like "FirstName|LastName||Company|PhoneNumber" with commas and spaces as delimiters:
```java
StringTokenizer st = new StringTokenizer("FirstName|LastName||Company|PhoneNumber", "|, ", true);
```
This will output tokens that include the empty fields:
```java
Token: FirstName
Token: |
Token: LastName
Token: ||
Token: Company
Token: |
Token: PhoneNumber
```

Here's a full example:
```java
import java.util.StringTokenizer;

public class StrTokDemo4 {
    public final static int MAXFIELDS = 5;
    public final static String DELIM = "|";

    public static void main(String[] args) {
        StringTokenizer st = new StringTokenizer("FirstName|LastName||Company|PhoneNumber", DELIM + ", ", true);
        while (st.hasMoreTokens()) {
            System.out.println("Token: " + st.nextToken());
        }
    }
}
```
This outputs:
```
Token: FirstName
Token: |
Token: LastName
Token: ||
Token: Company
Token: |
Token: PhoneNumber
```

The `MAXFIELDS` and `DELIM` constants are used to specify the maximum number of fields and the delimiter, respectively. The `true` parameter in the constructor ensures that empty fields are also included as tokens.
x??

---

#### String Tokenizer vs Regular Expressions
Background context: In Java, `StringTokenizer` is a useful tool for breaking down strings into tokens based on delimiters. However, the provided example demonstrates that this method can become cumbersome and inflexible when dealing with varying numbers of null fields or complex input patterns.

:p What are the limitations of using StringTokenizer in scenarios involving variable numbers of null fields?
??x
The limitations include its rigidity in handling missing values, as it requires predefined delimiters. This makes it less flexible compared to regular expressions which can handle more dynamic and varied input patterns.
x??

---
#### Regular Expressions for Parsing Strings
Background context: Regular expressions offer a powerful alternative to `StringTokenizer` by providing greater flexibility and versatility when parsing strings based on complex rules.

:p How do regular expressions provide more flexibility than StringTokenizer?
??x
Regular expressions allow for pattern matching with significant flexibility, enabling the extraction of specific patterns like numbers or words from a string without predefined delimiters. This makes them suitable for handling dynamic input formats.
x??

---
#### Example Code Using Regular Expressions
Background context: The example provided uses regular expressions to extract all numbers from a given string. It demonstrates how to use `Pattern` and `Matcher` in Java to find matches based on a specified pattern.

:p Provide the code snippet that extracts all numbers from a String using regular expressions.
??x
```java
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RegexExample {
    public static void main(String[] args) {
        String inputString = "Course 101: Introduction to Java, Course 205: Advanced Topics";
        
        // Create a matcher using the pattern
        Matcher tokenizer = Pattern.compile("\\d+").matcher(inputString);
        
        while (tokenizer.find()) {
            String courseString = tokenizer.group(0); // Get the matched number string
            int courseNumber = Integer.parseInt(courseString); // Convert to integer
            System.out.println("Extracted number: " + courseNumber);
        }
    }
}
```
The `Pattern.compile("\\d+")` creates a pattern that matches one or more digits. The `Matcher.find()` method searches the input string for this pattern and returns true if it finds a match, allowing us to iterate over all matches using `group(0)`.
x??

---
#### Handling Null Fields with Regular Expressions
Background context: The example shows how regular expressions can handle null fields in strings by extracting meaningful data without needing predefined delimiters or handling missing values explicitly.

:p How does the provided code snippet manage null fields when processing input strings?
??x
The `Pattern.compile("\\d+")` pattern is used to extract numbers, which implicitly handles null fields since it only matches actual numeric patterns. This means that non-numeric content like delimiters or other characters are ignored, making the process flexible and adaptable to various string formats.
x??

---
#### Flexibility in User Input Handling
Background context: The example illustrates how regular expressions can be more flexible than `StringTokenizer` when dealing with user input. Regular expressions allow for pattern matching that can accommodate variable numbers of null fields or different delimiters.

:p Explain why using regular expressions is better suited for handling complex and varied user inputs.
??x
Using regular expressions is better suited because they offer a powerful pattern-matching capability that can handle various input formats dynamically. Unlike `StringTokenizer`, which requires predefined delimiters, regular expressions can match patterns based on specific criteria, making them more adaptable to different types of user inputs without explicit handling for null fields or varying delimiters.
x??

---

