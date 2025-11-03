# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 75)

**Starting Chapter:** 17.7 Performance Timing. Problem. Solution. Discussion

---

#### Profiling Tools and Performance Measurement
Background context explaining the importance of profiling tools and performance measurement. Discuss the evolution from Oracle JDK to VisualVM and Java Flight Recorder, highlighting their respective roles in performance analysis.

:p What are profilers, and why are they important for programmers?
??x
Profiler tools, also known as performance analyzers, help find bottlenecks in your program by showing both the number of times each method was called and the amount of time spent in each. They provide valuable insights into application performance and can be crucial for optimizing code.

Examples of profiling tools include VisualVM (open-sourced from Oracle JDK to VisualVM project) and Java Flight Recorder, which is part of the JDK and designed to collect detailed runtime data that can be analyzed by Java Mission Control.
x??

---

#### Measuring a Single Method Using Time
Background context explaining how measuring a single method's execution time can help in optimizing code. Discuss using `System.currentTimeMillis()` for simple timing.

:p How can you measure the efficiency of a specific operation in Java?
??x
You can measure the efficiency of a specific operation by saving the JVM’s accumulated time before and after dynamically loading a main program, then calculating the difference between those times. This method provides an approximate total time taken by the operation under test.

Here is a simple example:
```java
long startTime = System.currentTimeMillis();
// code to be timed
long endTime = System.currentTimeMillis();

System.out.println("Time taken: " + (endTime - startTime) + " milliseconds");
```
This approach helps in understanding how long a particular piece of code takes to execute, allowing for optimization.
x??

---

#### Theory A vs. Theory B: String Concatenation and println()
Background context discussing the theory that string concatenation might be inefficient compared to `println()`.

:p How do you test whether string concatenation or `println()` is more efficient?
??x
To test which operation (string concatenation or using `println()`) is more efficient, you can write a simple timing program. Here’s an example for testing string concatenation with `System.out.println`:

```java
public class StringPrintA {
    public static void main(String[] argv) {
        Object o = "Hello World";
        for (int i = 0; i < 100000; i++) {
            System.out.println("<p><b>" + o.toString() + "</b></p>");
        }
    }
}
```

To test `println()` without string concatenation:
```java
public class StringPrintB {
    public static void main(String[] argv) {
        Object o = "Hello World";
        for (int i = 0; i < 100000; i++) {
            System.out.print("<p><b>");
            System.out.print(o.toString());
            System.out.print("</b></p>");
            System.out.println();
        }
    }
}
```

By running these tests and measuring the time taken, you can determine which approach is more efficient.
x??

---

#### Garbage Collection (GC) in Java
Background context explaining garbage collection behavior in Java and its significance. Discuss key presentations from Sun/Oracle at JavaOne.

:p What is garbage collection, and why is it important for Java applications?
??x
Garbage collection (GC) is a process that automatically manages the memory allocation and deallocation of objects within an application. It helps to reduce memory leaks and ensures efficient use of system resources by reclaiming unused memory. In Java, GC is managed by the JVM.

The importance of understanding GC behavior is highlighted in presentations such as "Garbage Collection in the Java HotSpot Virtual Machine" from 2003 and "Garbage Collection-Friendly Programming" from 2007, both presented at JavaOne.

GC can significantly impact application performance. For example:
```java
// Example of a method that could cause memory leaks if not properly managed
public void keepReference(Object obj) {
    while (true) {
        ref = obj;
    }
}
```
Understanding how to write GC-friendly code can improve application efficiency.
x??

---

#### Building a Simple Time Command in Java
Background context explaining the use of `System.currentTimeMillis()` for timing. Discuss creating a simple time command in Java.

:p How can you build a simplified version of a 'time' command in Java?
??x
You can build a simplified version of a 'time' command in Java by using `System.currentTimeMillis()`. This method allows you to measure the execution time of any given class by dynamically loading it and measuring the start and end times.

Here’s an example:
```java
public class Time {
    public static void main(String[] argv) throws Exception {
        // Instantiate target class, from argv[0]
        Class<?> c = Class.forName(argv[0]);
        
        // Find its static main method (use our own argv as the signature)
        Class<?>[] classes = { argv.getClass() };
        Method main = c.getMethod("main", classes);
        
        // Make new argv array, dropping class name from front
        String nargv[] = new String[argv.length - 1];
        System.arraycopy(argv, 1, nargv, 0, nargv.length);

        long startTime = System.currentTimeMillis();
        main.invoke(null, (Object) nargv);
        long endTime = System.currentTimeMillis();

        System.out.println("Time taken: " + (endTime - startTime) + " milliseconds");
    }
}
```

This program dynamically loads a specified class and measures the time it takes to execute its `main` method.
x??

---

---
#### Timing Program Execution
Background context: The provided Java program measures the runtime of a given class's main method by calculating the difference between start and end times using `System.currentTimeMillis()`. This method is useful for benchmarking but has limitations, such as excluding certain initialization overheads.

:p How does this Java program measure the execution time of a main method?
??x
The program uses `System.currentTimeMillis()` to record the current time before and after running the main method. The difference between these times gives the runtime in milliseconds.
```java
long t0 = System.currentTimeMillis(); // Record start time
main.invoke(null, nargs);             // Run the main program
long t1 = System.currentTimeMillis(); // Record end time

// Calculate runtime
long runTime = t1 - t0;
System.err.println("runTime=" + Double.toString(runTime / 1000D));
```
x??
---

#### Excluding Initialization Overhead
Background context: Directly comparing operating system timing with the above program's timing can lead to discrepancies because the latter excludes certain initialization overheads that are present in OS-level timing.

:p Why might results from the operating system time command differ when compared to this Java program’s results?
??x
The operating system time includes all processes and their associated overhead, such as JVM startup time. In contrast, the provided program measures only the runtime of the main method after the JVM has started.
x??
---

#### Printing Class Information
Background context: The text discusses using reflection in Java to print class information, similar to how `javap` works. This involves getting a `Class` object and using its methods like `getFields()` and `getMethods()`.

:p How can you use reflection to print all the fields and methods of a given class?
??x
By obtaining a `Class` object and calling `getDeclaredFields()`, you can retrieve an array of `Field` objects representing the declared fields in that class. Similarly, using `getDeclaredMethods()` returns an array of `Method` objects.

Here is how you might implement it:
```java
protected void doClass(String className) {
    try {
        Class<?> c = Class.forName(className);
        
        // Print annotations if any
        final Annotation[] annotations = c.getAnnotations();
        for (Annotation a : annotations) {
            System.out.println(a);
        }

        // Print class name and opening brace
        System.out.println(c + " {");

        // Get declared fields
        Field fields[] = c.getDeclaredFields();
        for (Field f : fields) {
            System.out.println(f);
        }
    } catch (ClassNotFoundException e) {
        e.printStackTrace();
    }
}
```
x??
---

#### Enumerating Classes in a Package

Enumerating all classes within a package is challenging due to the dynamic nature of Java and the JVM. The `java.lang.Package` class does not provide direct support for listing all classes within a given package.

The following code demonstrates how to enumerate classes by scanning the CLASSPATH, which works only for local directories and JAR files but not dynamically loaded classes.

:p How can you list the contents of a package in Java?
??x
You can't directly list all classes in a package using standard reflection APIs. However, you can achieve this by scanning the CLASSPATH to find class files or JAR entries that belong to the given package. This approach is limited because it only works for local directories and JAR files.

The provided code iterates through URLs from the CLASSPATH, handling both `file` and `jar` protocols separately.

```java
public class ClassesInPackage {
    public static String[] getPackageContent(String packageName)
            throws IOException {
        final String packageAsDirName = packageName.replace(".", "/");
        final List<String> list = new ArrayList<>();
        final Enumeration<URL> urls = Thread.currentThread()
                .getContextClassLoader()
                .getResources(packageAsDirName);
        while (urls.hasMoreElements()) {
            URL url = urls.nextElement();
            // System.out.println("URL = " + url);
            String file = url.getFile();
            switch (url.getProtocol()) {
                case "file":
                    File dir = new File(file);
                    for (File f : dir.listFiles()) {
                        list.add(packageAsDirName + "/" + f.getName());
                    }
                    break;
                case "jar":
                    int colon = file.indexOf(':');
                    int bang = file.indexOf('.');
                    String jarFileName = file.substring(colon + 1, bang);
                    JarFile jarFile = new JarFile(jarFileName);
                    Enumeration<JarEntry> entries = jarFile.entries();
                    while (entries.hasMoreElements()) {
                        JarEntry e = entries.nextElement();
                        String jarEntryName = e.getName();
                        if (jarEntryName.endsWith("/") && 
                                jarEntryName.startsWith(packageAsDirName)) {
                            list.add(jarEntryName);
                        }
                    }
                    break;
                default:
                    throw new IllegalStateException(
                            "Dunno what to do with URL " + url);
            }
        }
        return list.toArray(new String[0]);
    }

    public static void main(String[] args) throws IOException {
        String[] names = getPackageContent("com.darwinsys.io");
        for (String name : names) {
            System.out.println(name);
        }
        System.out.println("Done");
    }
}
```
x??

---
#### Handling Package Names

The code provided handles package names by converting them to directory names using the `replace` method. This is necessary because file URLs are treated as directories.

:p How does the code handle package names?
??x
Package names are converted into directory paths by replacing the dot (`.`) with a forward slash (`/`). This transformation allows treating the package name as a directory structure, which can be used to locate class files in local directories or JAR entries in JAR files.

For example, if you have a package named `com.darwinsys.io`, the code converts it into `com/darwinsys/io`.

```java
final String packageAsDirName = packageName.replace(".", "/");
```
x??

---
#### URL Protocols Handling

The provided code handles two types of URLs: `file` and `jar`. For each type, it processes class files differently.

:p What are the protocols handled in the given code?
??x
The code handles both `file` and `jar` protocols. The handling logic is different for these protocols:

1. **File Protocol**:
   - The URL represents a directory on the file system.
   - The code converts this to a `File` object and lists all class files in that directory.

2. **Jar Protocol**:
   - The URL contains information about a JAR file and its contents.
   - The code extracts the JAR filename, opens it as a `JarFile`, and then iterates through its entries to find matching classes.

```java
case "file":
    // This is the easy case: "file" is
    // the full path to the classpath directory
    File dir = new File(file);
    for (File f : dir.listFiles()) {
        list.add(packageAsDirName + "/" + f.getName());
    }
    break;
case "jar":
    int colon = file.indexOf(':');
    int bang = file.indexOf('.');
    String jarFileName = file.substring(colon + 1, bang);
    JarFile jarFile = new JarFile(jarFileName);
    Enumeration<JarEntry> entries = jarFile.entries();
    while (entries.hasMoreElements()) {
        JarEntry e = entries.nextElement();
        String jarEntryName = e.getName();
        if (jarEntryName.endsWith("/") && 
                jarEntryName.startsWith(packageAsDirName)) {
            list.add(jarEntryName);
        }
    }
    break;
```
x??

---
#### Enumerating Classes in JARs

For `jar` URLs, the code extracts the JAR filename from the URL and uses it to open a `JarFile`. It then enumerates entries in the JAR file, checking if they are class files belonging to the specified package.

:p How does the code handle JAR URLs?
??x
The code handles JAR URLs by extracting the JAR filename from the protocol string. Once the JAR file is opened as a `JarFile`, it enumerates its entries and checks each entry's name to determine if it corresponds to a class file in the specified package.

Here’s a detailed breakdown:

1. **Extracting Jar Filename**:
   - The URL contains information about the JAR file.
   - The filename is extracted using string manipulation: `file.substring(colon + 1, bang)`.

2. **Opening JarFile and Enumerating Entries**:
   - A new `JarFile` object is created with the extracted JAR filename.
   - The entries in the JAR are enumerated using an `Enumeration<JarEntry>`.
   - For each entry, its name is checked to see if it ends with a slash (`/`) and starts with the package directory name.

3. **Adding Matching Entries**:
   - If an entry matches the criteria (class file in the specified package), its name is added to the list.

```java
int colon = file.indexOf(':');
int bang = file.indexOf('.');
String jarFileName = file.substring(colon + 1, bang);
JarFile jarFile = new JarFile(jarFileName);
Enumeration<JarEntry> entries = jarFile.entries();
while (entries.hasMoreElements()) {
    JarEntry e = entries.nextElement();
    String jarEntryName = e.getName();
    if (jarEntryName.endsWith("/") && 
            jarEntryName.startsWith(packageAsDirName)) {
        list.add(jarEntryName);
    }
}
```
x??

---

