# Flashcards: 10A006---Java-Cookbook---Ian-F-Darwin_processed (Part 68)

**Starting Chapter:** 15.8 Creating a Smaller Distribution with jlink. Problem. Solution. 15.9 Using JPMS to Create a Module

---

#### Modularizing Your Application for Smaller Distributions

Background context: In Java 9, `jlink` was introduced as a tool to create smaller, more efficient Java distributions by including only the necessary modules. This is particularly useful when distributing applications to end users where minimizing download size and runtime overhead are important.

:p What is `jlink` used for in Java 9?
??x
`jlink` is used to create a custom Java distribution that includes only the required modules and classes from the JDK, thus reducing the overall size of the application bundle. This process modularizes your application and removes unused parts.
x??

---

#### Using jdeps to Identify Required Modules

Background context: The `jdeps` tool helps identify which modules are being used by an application, which is crucial for creating a minimal Java distribution using `jlink`. By analyzing dependencies, you can ensure that only the necessary classes are included in your final package.

:p How does `jdeps` help in modularizing applications?
??x
The `jdeps` tool helps identify the modules and packages that an application depends on. This information is essential for creating a minimal Java distribution with `jlink`, as it allows you to know exactly which modules need to be included in your final package.
x??

---

#### Compiling and Packaging Your Application

Background context: Before using `jlink` to create a smaller distribution, your application needs to be compiled and packaged into its own module. This involves creating a `module-info.java` file that defines the dependencies of your application.

:p How do you compile and package your application with `javac`?
??x
You can use the `javac` command along with the `-d` option to specify the output directory for compiled classes, followed by packaging them into a JAR file. Here’s an example:
```sh
$ javac -d . src/*.java
$ jar cvf demo.jar module-info.class demo
```
This compiles your source files and packages the `module-info.class` along with other classes into a single JAR file named `demo.jar`.
x??

---

#### Running jlink to Create a Mini-Java Distribution

Background context: After identifying the required modules, you can use `jlink` to create a custom Java distribution. This includes only your application and the necessary modules from the JDK.

:p How do you run `jlink` to create a mini-Java distribution?
??x
To run `jlink`, you need to specify the module path, options for excluding header files and man pages, compressing output, stripping debug information, adding launchers, and specifying the output directory. Here’s an example:
```sh
$ jlink --module-path . \
        --no-header-files \
        --no-man-pages --compress=2 --strip-debug \
        --launcher rundemo=demo/demo.Hello \
        --add-modules demo --output mini-java
```
This command creates a custom Java distribution in the `mini-java` directory, including only your application and its required modules.
x??

---

#### Understanding the Output of jdeps

Background context: The output of `jdeps` provides information on which modules and classes are being used by your application. This is crucial for ensuring that all necessary dependencies are included when creating a minimal Java distribution.

:p What does the output of `jdeps` tell you?
??x
The output of `jdeps` tells you which modules and packages your application depends on, helping to identify exactly what needs to be included in the final package. For example:
```sh
$ jdeps --module-path . demo.jar demo  [file:///Users/ian/workspace/javasrc/jlink/./]
requires mandated java.base (@11.0.2)
demo -> java.base
    demo          -> java.io        java.base
    demo          -> java.lang      java.base
```
This output indicates that `java.base` is a required module, and `demo` uses classes from the `java.io` and `java.lang` packages.
x??

---

#### Module System Introduction
Background context: The Java 9 introduced the module system (JPMS) to manage dependencies and encapsulate libraries. A `module-info.java` file is used to declare a module's name, its dependencies on other modules, and the packages it exports.

:p What does a `module-info.java` file do?
??x
A `module-info.java` file provides metadata about your module, including the module's name, its dependencies, and which packages are exported. It allows Java to understand how different modules interact with each other at runtime.
```java
// Example of a basic module declaration
module foo {
    // Empty
}
```
x??

---

#### Module Dependencies
Background context: Modules can depend on other modules by listing them in the `requires` clause within the `module-info.java`. This helps in managing dependencies and avoids classpath conflicts.

:p How do you add a dependency to another module in a `module-info.java` file?
??x
You use the `requires` keyword followed by the name of the module that your current module depends on. For example, if your module needs to interact with JavaFX (which is part of the java.desktop module), you would write:
```java
module com.darwinsys.api {
    requires java.desktop;
}
```
x??

---

#### Exporting Packages
Background context: To expose certain packages as a public API that can be accessed by other modules, you use the `exports` keyword in your `module-info.java`. This helps in controlling which parts of your module are visible to other code.

:p How do you export a package in a `module-info.java` file?
??x
You use the `exports` keyword followed by the fully qualified name of the package. For example, if you want to make all packages under com.darwinsys.api available to other modules:
```java
module com.darwinsys.api {
    exports com.darwinsys.api;
}
```
x??

---

#### Using Automatic Modules
Background context: If a JAR file does not declare itself as a module in its `module-info.java`, the Java runtime can automatically generate a module for it. However, this is discouraged due to potential issues with dependency management.

:p What warning might you see when using automatic modules?
??x
You may receive a warning like:
```plaintext
[WARNING] *********************************************************************
[WARNING] * Required filename-based automodules detected. Please don't publish *
[WARNING] * this project to a public artifact repository.                      *
[WARNING] *********************************************************************
```
This is because the Java runtime automatically generates a module for JAR files without an explicit `module-info.java`, which might cause issues in a production environment.
x??

---

#### Providing Services
Background context: The `provides` keyword allows modules to declare services that can be used by other modules. This feature enables multiple implementations of an interface, each within its own module.

:p How do you provide a service using the `module-info.java` file?
??x
You use the `provides` keyword followed by the fully qualified name of the interface and then specify which implementation class provides that service. For example:
```java
module com.darwinsys.api {
    ...
    provides com.darwinsys.locks.LockManager with com.darwinsys.locks.LockManagerImpl;
}
```
This means that `LockManager` can be provided by the `LockManagerImpl` class in a different module.
x??

---

#### Opening Packages for Reflection
Background context: By default, packages exported through `exports` cannot be introspected using reflection. The `opens` keyword allows a module to expose its internal classes and interfaces to another module that needs to use them.

:p How do you open a package for reflection in the `module-info.java` file?
??x
You use the `opens` keyword followed by the fully qualified name of the package. For example:
```java
module com.darwinsys.api {
    ...
    opens com.darwinsys.model;
}
```
This allows another module to reflectively access classes and interfaces in the `com.darwinsys.model` package.
x??

---

#### Optional Interface Usage
Background context: The `Optional` interface is used to represent an optional value. It can be used to avoid null pointer exceptions and make your APIs more robust by indicating that a method may return no value at all.

:p What does the given code snippet demonstrate about using `Optional` with `ServiceLoader.load()`?
??x
The code snippet demonstrates how to use `Optional` to handle cases where an implementation of `LockManager` might not be found. By checking if `Optional<LockManager>` is present, it ensures that a null pointer exception does not occur when trying to access the value.

```java
Optional<LockManager> opt = ServiceLoader.load(LockManager.class).findFirst();
if (!opt.isPresent()) {
    throw new RuntimeException("Could not find implementation of LockManager");
}
LockManager mgr = opt.get();
```

Explanation: 
- `ServiceLoader.load(LockManager.class)` attempts to load the `LockManager` service provider.
- `.findFirst()` returns an `Optional<LockManager>` that may contain a value or be empty if no implementation is found.
- The `if (!opt.isPresent())` condition checks if there is a value (i.e., a LockManager instance) available. If not, it throws a runtime exception indicating that no suitable implementation was found.

If the `Optional<LockManager>` does contain an element, `.get()` method retrieves the actual `LockManager` object.
x??

---

#### ServiceLoader and Module Systems
Background context: The Java Platform Module System (JPMS) introduces a way to manage dependencies between modules in a more structured manner. Before JPMS, classes were loaded using class loaders; now with modules, services can be looked up using `ServiceLoader`.

:p What is the role of `ServiceLoader` in the provided code?
??x
The role of `ServiceLoader` in the provided code is to load implementations of the `LockManager` service provider. It scans for any available providers on the classpath and returns an `Optional<LockManager>` that can be checked to see if a suitable implementation was found.

```java
Optional<LockManager> opt = ServiceLoader.load(LockManager.class).findFirst();
if (!opt.isPresent()) {
    throw new RuntimeException("Could not find implementation of LockManager");
}
```

Explanation: 
- `ServiceLoader.load(LockManager.class)` queries the system for all available implementations that implement or extend `LockManager`.
- `.findFirst()` retrieves the first found provider, which is wrapped in an `Optional<LockManager>`.
- The `if (!opt.isPresent())` condition ensures that if no suitable implementation of `LockManager` was found, a runtime exception is thrown.
x??

---

#### Handling Unavailable Services with Optional
Background context: The Java 9 Modularity introduced the `Optional` class to help manage null values gracefully. This is especially useful in scenarios where services might not be available due to module dependencies or lack of implementation.

:p How does using `Optional` prevent potential runtime issues?
??x
Using `Optional` prevents potential runtime issues by avoiding null pointer exceptions when dealing with optional values. Instead of returning a null value, which can lead to unhandled exceptions later in the code, `Optional` provides methods like `isPresent()` and `get()`, allowing developers to handle the absence or presence of a value explicitly.

```java
Optional<LockManager> opt = ServiceLoader.load(LockManager.class).findFirst();
if (!opt.isPresent()) {
    throw new RuntimeException("Could not find implementation of LockManager");
}
```

Explanation: 
- By using `Optional`, you can check if a value is present with `.isPresent()`. If no value is found, the application throws an exception.
- This approach ensures that your code does not inadvertently assume a non-null value and can gracefully handle cases where a service might be missing.

Using `Optional` in this way enhances the robustness of your application by making it clear when certain components are optional or may fail to load.
x??

---

#### Migrating to JPMS
Background context: The Java Platform Module System (JPMS) was introduced in Java 9 to improve modularity and dependency management. Migration from traditional class loading to module systems involves changes in project structure, configuration, and possibly source code.

:p What is the main purpose of migrating to a modular system like JPMS?
??x
The main purpose of migrating to a modular system like JPMS is to enhance the modularity and manageability of Java applications. It provides better control over dependencies, clearer separation between modules, and improved performance due to optimized module loading.

Migration involves several steps such as:

1. **Organizing code into modules**: Grouping related classes and resources into modules.
2. **Defining module boundaries**: Clearly defining which packages are public and accessible from other modules.
3. **Updating dependencies**: Specifying dependencies between modules using the `module-info.java` file.

For example, a modular project might have multiple modules:

```java
// module-info.java in a LockManager module
module com.example.lockmanager {
    requires java.base;
    exports com.example.lockmanager.api; // Public API
}
```

Explanation: 
- By migrating to JPMS, developers can better organize their code into logical units (modules), making it easier to maintain and scale.
- The `module-info.java` file is crucial as it defines the module's identity, its dependencies, and public APIs. This helps in managing dependencies more explicitly and reducing classpath-related issues.

This migration process ensures that applications are built with a clearer structure, leading to more robust and modular software.
x??

---

