# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 37)

**Starting Chapter:** Singletons Impede Changeability and Testability

---

---
#### Singleton Pattern Overview
Singleton patterns are used when you want to ensure that a particular class has only one instance and provide a global point of access to it. This is often necessary for managing global resources or states within an application.

:p Why might developers use the Singleton pattern?
??x
The Singleton pattern is useful in scenarios where there needs to be exactly one instance of a class, such as when managing system-wide resources like databases, loggers, clocks, or configurations. The primary goal is to ensure that all parts of the codebase can access this single instance uniformly and consistently.

```java
// Example Singleton implementation in Java
public class Logger {
    private static final Logger INSTANCE = new Logger();

    // Private constructor to prevent instantiation from outside
    private Logger() {}

    public static Logger getInstance() {
        return INSTANCE;
    }

    // Methods for logging can be added here
}
```
x??

---
#### Challenges with Singleton
Singletons are often criticized because they create tight coupling between classes, making the code harder to test and maintain. Additionally, singletons can make it difficult to implement changes without affecting other parts of the system.

:p Why might developers dislike using Singleton?
??x
Developers dislike Singleton because it tends to introduce artificial dependencies and makes testing more challenging. It violates guidelines for design that emphasize flexibility and testability. The main issues stem from its nature as a global variable, which can lead to hidden side effects and difficulty in managing state changes.

```java
// Example of a potential issue with Singleton
public class DatabaseSingleton {
    private static final DatabaseSingleton instance = new DatabaseSingleton();
    private Connection connection;

    private DatabaseSingleton() {}

    public static DatabaseSingleton getInstance() {
        return instance;
    }

    // Other methods to manage the database can be added here

    public void connect() throws SQLException {
        if (connection == null) {
            connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/mydb", "user", "password");
        }
    }
}
```
x??

---
#### Singleton and Change
Singletons are sometimes necessary to represent global aspects of a program. However, it is crucial to design them in a way that they can be modified without impacting the rest of the system.

:p In what situations might developers use Singleton?
??x
Developers may use Singleton when there is a need for something to exist only once and should be accessible from multiple parts of the application. This includes scenarios like managing global state, resources, or services such as logging, configuration, or database connections.

```java
// Example of using Singleton for a global service in Java
public class ConfigurationService {
    private static final ConfigurationService INSTANCE = new ConfigurationService();
    private Properties properties;

    private ConfigurationService() {}

    public static ConfigurationService getInstance() {
        return INSTANCE;
    }

    // Methods to load and retrieve configuration values can be added here

    public String getProperty(String key) {
        if (properties == null) {
            properties = new Properties();
            try (InputStream input = new FileInputStream("config.properties")) {
                properties.load(input);
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        }
        return properties.getProperty(key);
    }
}
```
x??

---
#### Singleton and Testability
Singletons can make testing more difficult because they can hold state across different test cases, leading to potential side effects. To address this, it is important to design singletons in a way that allows for mocking or stubbing during tests.

:p How can developers ensure Singleton is testable?
??x
To ensure that a Singleton is testable, developers should encapsulate the Singleton's dependencies and use interfaces or abstract classes. This allows testers to provide mock implementations of these dependencies when needed. Additionally, Singletons should be designed with an awareness of their global nature; they should not rely on side effects that could affect other parts of the application.

```java
// Example of making a Singleton testable in Java
public class Logger {
    private static final Logger instance = new Logger();
    private final ILogWriter logWriter;

    public Logger(ILogWriter logWriter) {
        this.logWriter = logWriter;
    }

    // Method to get the singleton instance with an injected dependency
    public static Logger getInstance(ILogWriter logWriter) {
        if (instance == null) {
            instance = new Logger(logWriter);
        }
        return instance;
    }

    // Methods for logging can be added here

    public void writeLog(String message) {
        logWriter.write(message);
    }
}

// Interface for the logger writer
public interface ILogWriter {
    void write(String message);
}
```
x??

---

---
#### Mutable Global State Issues
Background context explaining the problems associated with mutable global state, especially in multithreaded environments. Mention how difficult it is to control access and ensure correctness.
:p What are some of the main issues with mutable global state?
??x
Mutable global state can cause several issues:
1. **Controlled Access**: It's challenging to manage who can modify or read from this state, especially in a multithreaded environment where race conditions and deadlocks can easily occur.
2. **Correctness Guarantee**: Ensuring the data remains consistent across all functions that interact with it is difficult, making debugging and maintaining code harder.
3. **Reasoning Difficulty**: It's hard to understand how changes in one part of the system will affect others because state mutations are often hidden within function implementations.

```java
public class GlobalStateExample {
    private static int globalCounter = 0;

    public void increment() {
        globalCounter++;
    }

    public int getCount() {
        return globalCounter;
    }
}
```
x??

---
#### Singleton Pattern and Global State
Explain how the Singleton pattern can be used to manage global state while mitigating some of its common issues. Highlight that Singletons enforce a single instance but allow for controlled access.
:p How does using a Singleton help in managing global state?
??x
Using a Singleton helps in managing global state by providing controlled access to the shared data, which reduces the risk of race conditions and ensures that only one instance of the class exists.

For example, consider a logger Singleton:
```cpp
class Logger {
private:
    static Logger* instance;
    // Private constructor and copy/move constructors/assignment operators

public:
    static Logger& getInstance() {
        if (instance == nullptr) {
            instance = new Logger();
        }
        return *instance;
    }

    void log(const std::string& message) {
        // Log the message
    }
};
```
Here, `Logger` can only be accessed via its `getInstance()` method, which ensures that there is a single global state. This approach helps in simplifying the design of applications where some states need to be shared globally but in a controlled manner.

x??

---
#### Unidirectional Data Flow with Singletons
Explain how restricting data flow within Singletons can help mitigate issues related to mutable global state.
:p What are the benefits of unidirectional data flow in Singleton implementations?
??x
Unidirectional data flow in Singleton implementations restricts access patterns, making it easier to manage and reason about the state. For instance:
- A logger Singleton should only allow writing (logging) but not reading (accessing logs).
- A configuration Singleton should allow reading of global settings but not modification.

Here's an example of a read-only Singleton:
```cpp
class ConfigSingleton {
private:
    static ConfigSingleton* instance;
    std::string configValue;

    // Private constructor and copy/move constructors/assignment operators

public:
    static ConfigSingleton& getInstance() {
        if (instance == nullptr) {
            instance = new ConfigSingleton();
        }
        return *instance;
    }

    const std::string& getConfigValue() const {
        return configValue;
    }
};

ConfigSingleton* ConfigSingleton::instance = nullptr;
```
x??

---
#### Singleton and Testability
Explain the impact of Singletons on testability and changeability in software design.
:p What are the consequences of using a Singleton for global state?
??x
Using a Singleton can make code harder to test and less changeable because functions that use Singletons depend on the represented global data. This dependency makes it difficult to isolate parts of the system for testing or changing behavior.

For example, consider the Database Singleton being used by multiple classes:
```cpp
class Widget {
public:
    void doSomething() {
        DatabaseSingleton::getInstance().read();
    }
};

class Gadget {
public:
    void doSomething() {
        DatabaseSingleton::getInstance().write();
    }
};
```
Here, `Widget` and `Gadget` both depend on the global state provided by `DatabaseSingleton`, making it challenging to test their behavior in isolation without setting up a real database or mocking it.

x??

---

#### Dependency Inversion Principle (DIP)
Background context: The Dependency Inversion Principle states that high-level modules should not depend on low-level modules, but both should depend on abstractions. This principle helps ensure that the architecture is flexible and maintainable by decoupling concrete implementations from their clients.
:p What does the Dependency Inversion Principle recommend in terms of dependencies?
??x
The Dependency Inversion Principle recommends that high-level modules (e.g., application logic) should not depend on low-level modules (e.g., database access). Instead, both should depend on abstractions. This ensures that changes in the implementation details do not affect the higher levels.
```java
// Example of a violation of DIP using Singleton pattern for database access
public class DatabaseSingleton {
    private static final DatabaseSingleton INSTANCE = new DatabaseSingleton();
    
    public static DatabaseSingleton getInstance() {
        return INSTANCE;
    }
    
    // Database methods
}

public class Widget {
    public void doSomething() {
        DatabaseSingleton.getInstance().executeQuery();  // Direct dependency on concrete implementation
    }
}
```
x??

---

#### Inverted Dependency Structure with Singleton Pattern
Background context: The text discusses the illusion of a proper dependency structure when using Singletons. Although it appears that dependencies should run from lower to higher levels, in reality, there is an inverted dependency where all dependencies point towards the concrete implementation.
:p What issue does the text highlight regarding the use of Singleton pattern for database access?
??x
The text highlights that even though the database seems to be a high-level abstraction (Singleton), it actually has strong and invisible dependencies from various parts of the code, making the system rigid. This violates the Dependency Inversion Principle because all dependencies point towards the lower level, indicating a lack of proper abstraction.
```java
// Example showing inverted dependency with Singleton pattern for database access
public class Widget {
    public void doSomething() {
        DatabaseSingleton.getInstance().executeQuery();  // Direct dependency on concrete implementation
    }
}
```
x??

---

#### Testability Issues with Singleton Pattern
Background context: The use of Singleton patterns for critical components like databases can severely impact testability. Tests become tightly coupled with the specific implementation, making it difficult to substitute or mock the dependencies.
:p How does using a Singleton pattern affect testing in software development?
??x
Using a Singleton pattern for database access makes tests less flexible and harder to write because all functions that depend on the Singleton are also dependent on the same instance. This means you cannot easily replace the Singleton with mocks, stubs, or fakes during testing.
```java
// Example of a test issue due to Singleton dependency
public class WidgetTest {
    public void testDoSomething() {
        DatabaseSingleton.getInstance().executeQuery();  // Test is tightly coupled with the Singleton instance
    }
}
```
x??

---

#### Replacing Concrete Implementation with Abstractions
Background context: The text emphasizes the importance of making database access a true implementation detail on the low level, allowing for easier replacement and testing. This involves creating abstractions that can be easily substituted.
:p Why is it important to replace concrete implementations like databases with abstractions in software architecture?
??x
It is crucial to replace concrete implementations like databases with abstractions because this allows for more flexible and maintainable code. Abstractions enable you to change the database implementation without affecting higher-level logic, making the system easier to test and modify.
```java
// Example of replacing Singleton with an interface abstraction
public interface Database {
    void executeQuery();
}

public class Widget {
    private final Database db;

    public Widget(Database db) {
        this.db = db;
    }

    public void doSomething() {
        db.executeQuery();  // Dependency on abstraction, not concrete implementation
    }
}
```
x??

---

#### Realizing Proper Architecture with Abstractions
Background context: The text suggests that for a proper architecture, dependencies should run from lower to higher levels. Implementing database access through abstractions can achieve this and improve the overall structure of the system.
:p How can we ensure that our software architecture adheres to the Dependency Inversion Principle?
??x
To ensure that your software architecture adheres to the Dependency Inversion Principle, you need to implement dependencies from lower to higher levels. This means using interfaces or abstract classes for database access and injecting them into dependent components rather than directly using concrete implementations like Singletons.
```java
// Example of realizing proper architecture with abstractions
public interface Database {
    void executeQuery();
}

public class Widget {
    private final Database db;

    public Widget(Database db) {
        this.db = db;
    }

    public void doSomething() {
        db.executeQuery();  // Dependency on abstraction, not concrete implementation
    }
}
```
x??

#### Decorator Pattern and Singleton
In C++, using `std::pmr::monotonic_buffer_resource`, we see a practical example of both the Decorator pattern and Singleton. The `std::pmr::monotonic_buffer_resource` serves as an abstraction over memory management, but its implementation is concrete.
:p How does the `std::pmr::monotonic_buffer_resource` exemplify both the Decorator and Singleton patterns?
??x
The `std::pmr::monotonic_buffer_resource` acts as a decorator by wrapping around raw memory to manage it more effectively. It also behaves like a singleton since the function `std::pmr::null_memory_resource()` always returns the same instance of an allocator, representing a single point of access.
```cpp
std::pmr::monotonic_buffer_resource buffer{raw.data(), raw.size(), std::pmr::null_memory_resource()};
```
x??

#### Strategy Pattern with Singleton
The `std::pmr::get_default_resource()` and `std::pmr::set_default_resource()` functions in C++ demonstrate how to change the global allocator at runtime. This is an application of the Strategy pattern, where different allocation strategies can be swapped out dynamically.
:p How does changing the default resource using `std::pmr::get_default_resource()` and `std::pmr::set_default_resource()` relate to the Strategy design pattern?
??x
Changing the default resource with these functions allows you to implement various allocation strategies. By setting a new allocator as the default, you can swap out different allocators at runtime, which is characteristic of the Strategy pattern.
```cpp
std::pmr::set_default_resource(std::pmr::null_memory_resource());
```
x??

#### Custom Allocator and Inversion of Control
Creating a custom allocator like `CustomAllocator` shows how to implement an abstraction that can be used globally while keeping specific implementations hidden. This inversion of control is crucial for modular design.
:p How does the `CustomAllocator` class illustrate the Strategy pattern in C++?
??x
The `CustomAllocator` class demonstrates the Strategy pattern by implementing a concrete allocator strategy. By inheriting from `std::pmr::memory_resource`, it can be set as the default resource, allowing different allocation strategies to be swapped out at runtime.
```cpp
class CustomAllocator : public std::pmr::memory_resource {
public:
    // Constructor and methods...
private:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(std::pmr::memory_resource const& other) const noexcept override;
};
```
x??

---

#### Dependency Inversion via `std::pmr::memory_resource` Abstraction
Background context: The example illustrates how to use the `std::pmr::memory_resource` abstraction from C++17 to achieve dependency inversion, which helps manage global state and initialization order issues. This pattern is applied in a database scenario where the `Database` class acts as a Singleton but becomes an implementation detail.
:p How does the text explain using `std::pmr::memory_resource` for dependency inversion?
??x
The text explains that by creating abstractions like `PersistenceInterface`, one can decouple higher-level logic from specific implementations, making it easier to swap out different database systems. The `get_persistence_interface()` function and `set_persistence_interface()` provide a way to set the global state without hardcoding dependencies on concrete implementations.
```cpp
class PersistenceInterface {
public:
    virtual ~PersistenceInterface() = default;
    bool read(/*some arguments*/) const { return do_read(/*...*/); }
    bool write(/*some arguments*/) { return do_write(/*...*/); }
private:
    virtual bool do_read(/*some arguments*/) const = 0;
    virtual bool do_write(/*some arguments*/) = 0;
};
PersistenceInterface* get_persistence_interface();
void set_persistence_interface(PersistenceInterface* persistence);
```
x??

---

#### Singleton Database Implementation with Lazy Initialization
Background context: The example shows how to implement the `Database` class as a Singleton that lazily initializes itself only if no other database is specified. This approach uses lazy initialization and local static variables for thread safety.
:p How does the text describe implementing a Singleton `Database` class?
??x
The `Database` class is implemented as a Singleton with lazy initialization. It checks if another persistence interface has been set using `set_persistence_interface()`. If not, it creates an instance of `Database` only once and uses local static variables to ensure thread safety.
```cpp
PersistenceInterface* get_persistence_interface() {
    // Local object, initialized by an 'Immediately Invoked Lambda Expression (IILE)'
    static bool init = [](){
        if(instance) {
            static Database db;
            instance = &db;
        }
        return true;  // or false, as the actual value does not matter.
    }();
    return instance;
}
```
x??

---

#### Strategy Design Pattern for Flexibility and Extension
Background context: The text introduces the `Strategy` design pattern to allow flexible database implementations. By introducing an abstract interface (`PersistenceInterface`) with a global access point, one can easily replace or extend the database implementation without affecting other parts of the code.
:p How does the example use the `Strategy` design pattern in this scenario?
??x
The `Strategy` design pattern is used by creating an abstract base class `PersistenceInterface` that defines common operations for different database implementations. This allows clients to switch between concrete implementations or test stubs easily, adhering to the Open-Closed Principle.
```cpp
class PersistenceInterface {
public:
    virtual ~PersistenceInterface() = default;
    bool read(/*some arguments*/) const { return do_read(/*...*/); }
    bool write(/*some arguments*/) { return do_write(/*...*/); }
private:
    virtual bool do_read(/*some arguments*/) const = 0;
    virtual bool do_write(/*some arguments*/) = 0;
};
```
x??

---

#### Thread Safety Concerns in Singleton Implementation
Background context: The example highlights potential thread safety issues with the current implementation of `get_persistence_interface()`. Even though it uses local static variables, there could be scenarios where setting a persistence interface before the first call to `get_persistence_interface()` leads to unexpected behavior.
:p What is the main thread safety issue in the provided Singleton implementation?
??x
The primary thread safety issue is that if `set_persistence_interface()` is called before the first call to `get_persistence_interface()`, the static local variable might not initialize correctly, leading either to an unused database instance or lost initialization.
```cpp
PersistenceInterface* get_persistence_interface() {
    // Local object, initialized by an 'Immediately Invoked Lambda Expression (IILE)'
    static bool init = [](){
        if(instance) {
            static Database db;
            instance = &db;
        }
        return true;  // or false, as the actual value does not matter.
    }();
    return instance;
}
```
x??

---

#### Non-Intrusive Solutions for Singleton Classes
Background context: If one cannot modify a given Singleton class, alternative non-intrusive design patterns like `Adapter` and `External Polymorphism` can be used to wrap or extend its functionality without changing the original class.
:p What are the two non-intrusive solutions mentioned for handling Singleton classes?
??x
The two non-intrusive solutions for handling Singleton classes are:
1. **Adapter**: Use it if an inheritance hierarchy is already in place, allowing wrapping of the Singleton to adapt its interface.
2. **External Polymorphism**: Use it when no inheritance hierarchy exists, enabling non-intrusive runtime polymorphism by using dynamic dispatch mechanisms.
```cpp
// Example of Adapter Pattern
class Adapter : public TargetInterface {
public:
    void doSomething() override { target->doOriginal(); }
private:
    std::unique_ptr<Source> target;
};
```
x??

---

