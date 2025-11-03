# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 38)

**Starting Chapter:** Moving Toward Local Dependency Injection

---

#### Global State Management
Background context: The text discusses managing global state, specifically through a `set_persistence_interface()` function. This function is used to set up persistence for various operations but poses limitations when called arbitrarily throughout testing.

:p Should the `set_persistence_interface()` function be allowed multiple calls?
??x
No, allowing it to be called multiple times would limit its use in testing scenarios where resetting the persistence system at the beginning of each test is necessary. This flexibility ensures that tests are isolated and do not rely on previous states.
x??

---

#### Local Dependency Injection via Constructor
Background context: To avoid global state issues, the text suggests using local dependency injection through constructors to pass dependencies directly to objects.

:p How does passing a persistence interface during Widget construction help?
??x
Passing a `PersistenceInterface` during Widget's construction ensures that each instance of `Widget` is properly initialized with its required dependencies. This makes the class more robust and easier to test, as dependencies are clearly defined.

```cpp
//---- <Widget.h> ----------------    
#include <PersistenceInterface.h>
class Widget {
public:
    Widget(PersistenceInterface* persistence)  // Dependency injection
        : persistence_(persistence)
    {}
    void doSomething(/*some arguments*/) {
        // ... persistence_->read(/*some arguments*/); 
        // ...
    }
private: 
    PersistenceInterface* persistence_{};
};
```
x??

---

#### Direct vs. Indirect Dependency Injection
Background context: The text explores whether to pass dependencies directly through member functions or as function arguments, weighing the pros and cons of each approach.

:p How does passing a `PersistenceInterface` via a member function's argument compare to using it in the constructor?
??x
Passing a `PersistenceInterface` via an argument in a member function (e.g., `doSomething(PersistenceInterface* persistence, /*some arguments*/)`) can make functions more flexible. However, this approach may become cumbersome when dealing with multiple dependencies and large call stacks.

```cpp
//---- <Widget.h> ----------------    
#include <PersistenceInterface.h>
class Widget {
public:
    void doSomething(/*some arguments*/) {
        // ... persistence_->read(/*some arguments*/); 
        // ...
    }
    void doSomething(PersistenceInterface* persistence, /*some arguments*/) {
        // ... persistence->read(/*some arguments*/);
        // ...
    }
private: 
    PersistenceInterface* persistence_{};
};
```
x??

---

#### Wrapper Function for Dependency Injection
Background context: The text proposes a compromise by introducing a wrapper function that simplifies dependency injection without requiring deep layering of dependencies.

:p What is the advantage of using a wrapper function to manage dependencies?
??x
Using a wrapper function allows local decision-making regarding dependencies while avoiding the complexity and unwieldiness of passing multiple dependencies through numerous function calls. The wrapper function acts as an intermediary, setting up the necessary context before invoking the actual logic that requires those dependencies.

```cpp
//---- <Widget.h> ----------------    
#include <PersistenceInterface.h>
class Widget {
public:
    void doSomething(/*some arguments*/) { 
        doSomething(get_persistence_interface(), /*some arguments*/); 
    }
    void doSomething(PersistenceInterface* persistence, /*some arguments*/) { 
        // ... persistence->read(/*some arguments*/);
        // ...
    }
private: 
    PersistenceInterface* get_persistence_interface() {
        // Code to return the current persistence interface
    }
};
```
x??

---

---
#### Singleton Pattern Overview
Background context: The Singleton pattern is a design pattern that ensures a class has only one instance and provides a global point of access to it. This makes it suitable for managing critical resources such as database connections or loggers.

The Singleton pattern is crucial for scenarios where you need controlled access to an object, often referred to as the "global state" in codebases. However, despite its benefits, it comes with several drawbacks, especially related to global state management and testability.

:p What are the key points about the Singleton pattern discussed in the text?
??x
The Singleton pattern is a design pattern that restricts the instantiation of a class to one "single" instance, thus controlling access to critical resources such as database connections or loggers. It aims to provide a single point of access while avoiding issues associated with global state.

Key points include:
1. **Single Instance Guarantee**: Ensures only one instance exists.
2. **Global Access Point**: Provides a way to access the singleton object from anywhere in the application.
3. **Disadvantages**:
   - **Global State Flaws**: Risks introducing tightly coupled code and making the system harder to test.
   - **Testability Issues**: Difficult to mock or replace with alternatives for testing purposes.

??x
The answer with detailed explanations.
```java
public class Singleton {
    private static Singleton instance;

    // Private constructor to prevent instantiation from outside
    private Singleton() {}

    // Public method to get the single instance of the class
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized(Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }

    // Example method in singleton class
    public void doSomething() {
        System.out.println("Singleton doing something");
    }
}
```
This code snippet demonstrates the classic implementation of a Singleton with synchronization to ensure thread safety and avoid multiple instances. The `getInstance()` method is used as a global access point.

```java
public class Client {
    public static void main(String[] args) {
        Singleton singleton = Singleton.getInstance();
        singleton.doSomething();
    }
}
```
The client code uses the Singleton instance to call methods, ensuring controlled access and leveraging its single instance nature.
x??

---
#### Singletons with Unidirectional Data Flow
Background context: The text suggests that when using Singleton patterns, it is better to use them in scenarios where there are few global aspects needed. Additionally, it recommends designing Singletons for change and testability by using unidirectional data flow.

Unidirectional data flow means ensuring that the Singleton object does not directly depend on other objects but rather depends on interfaces or abstractions, thus making the design more modular and easier to test.

:p How can you improve a Singleton's design for better modularity and testability?
??x
By designing Singletons with unidirectional data flow, you ensure that they do not tightly couple themselves with other classes. Instead, dependencies are passed through interfaces or abstractions, making the code more flexible and easier to test.

This approach involves:
1. **Dependency Injection**: Pass necessary dependencies as arguments rather than having them hardcoded in the Singleton.
2. **Strategy Pattern**: Use the Strategy pattern to define a family of algorithms, encapsulate each one, and make them interchangeable. This allows for different behaviors without modifying the client code.

Example using Dependency Injection:
```java
public interface Logger {
    void log(String message);
}

public class DatabaseLogger implements Logger {
    @Override
    public void log(String message) {
        // Log implementation
    }
}

public class SingletonWithStrategyPattern {
    private final Logger logger;

    public SingletonWithStrategyPattern(Logger logger) {
        this.logger = logger;
    }

    public void doSomething() {
        logger.log("Doing something");
    }
}
```
In this example, `SingletonWithStrategyPattern` depends on an injected `Logger`, allowing for different logging strategies without changing the Singleton's implementation.

x??

---
#### Strategy Design Pattern for Dependency Inversion
Background context: The text suggests using the Strategy design pattern to invert dependencies and enable dependency injection in Singletons. This helps manage changes and improve testability by making it easier to switch implementations at runtime.

The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. By applying this pattern, you can decouple the Singleton from concrete implementations, allowing for more flexible behavior.

:p How does the Strategy design pattern help in managing dependencies in Singletons?
??x
The Strategy design pattern helps manage dependencies in Singletons by allowing different behaviors to be implemented as separate classes (strategies) and then injected into the Singleton. This inversion of control makes the code more modular, testable, and flexible.

Steps for applying the Strategy pattern:
1. **Define Strategies**: Create interface or abstract class that defines the behavior.
2. **Implement Strategies**: Provide concrete implementations for each strategy.
3. **Inject Strategies**: Pass the chosen strategy to the Singleton via its constructor or setter method.

Example using the Strategy pattern:
```java
public interface Logger {
    void log(String message);
}

public class ConsoleLogger implements Logger {
    @Override
    public void log(String message) {
        System.out.println(message);
    }
}

public class DatabaseLogger implements Logger {
    @Override
    public void log(String message) {
        // Database logging implementation
    }
}

public class SingletonWithStrategyPattern {
    private final Logger logger;

    public SingletonWithStrategyPattern(Logger logger) {
        this.logger = logger;
    }

    public void doSomething() {
        logger.log("Doing something");
    }
}
```
In this example, the `SingletonWithStrategyPattern` uses a `Logger` interface to handle logging. Different strategies like `ConsoleLogger` or `DatabaseLogger` can be injected into the Singleton, providing flexible behavior.

x??

---

#### RAII in C++
RAII stands for Resource Acquisition Is Initialization, which is a programming pattern used to manage resources like memory. The intent of RAII is not to reduce dependencies but to automate cleanup and encapsulate responsibility by ensuring that resources are acquired as soon as an object is created and released when the object is destroyed.
:p What is the primary purpose of RAII in C++?
??x
RAII's primary purpose in C++ is to ensure that resources are properly managed, meaning they are automatically released when objects go out of scope. This reduces the risk of memory leaks and other resource management issues by tying resource acquisition with object construction and release with object destruction.
```cpp
class File {
public:
    File(const std::string& path) : file_(fopen(path.c_str(), "r")) {
        if (file_ == NULL) {
            throw std::runtime_error("Failed to open file");
        }
    }

    ~File() {
        if (file_) {
            fclose(file_);
        }
    }

private:
    FILE* file_;
};
```
x??

---

#### Singleton Pattern
The singleton pattern is a design pattern that restricts the instantiation of a class to one "single" instance. The primary intent of using a singleton in C++ codebases can often lead to misuse and tight coupling, which are generally considered bad practices.
:p Why should you be suspicious if a Singleton is used for anything other than its intended purpose?
??x
Singletons can lead to tight coupling and make code harder to test and maintain. If a Singleton is used beyond simple global state management (like logging or configuration), it often indicates that the design is not adhering to principles such as dependency injection and SOLID.
```cpp
class Logger {
public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void log(const std::string& message) const {
        // Log message implementation
    }
private:
    Logger() {}  // Private constructor to prevent instantiation from outside the class
};
```
x??

---

#### Monostate Pattern
The monostate pattern allows any number of instances of a type but ensures that there is only one state for all instances. This differs from the Singleton pattern, where only one instance can exist.
:p What distinguishes the Monostate pattern from the Singleton pattern?
??x
The key difference between the Monostate and Singleton patterns is that the monostate allows multiple instances, whereas the Singleton restricts instantiation to a single object. Both ensure there's only one state shared among all instances, but Monostate achieves this by sharing a common data structure.
```cpp
class Logger {
public:
    void log(const std::string& message) const {
        // Log message implementation
    }

private:
    static std::map<std::string, int> states;
};
```
x??

---

#### Test Doubles in C++
Test doubles (mocks, stubs, spies) are objects that stand in for real components during testing. They allow you to control and verify interactions between different parts of your code.
:p What is the purpose of test doubles like mocks and stubs in software development?
??x
The primary purpose of test doubles such as mocks and stubs is to isolate units of code during testing, allowing developers to focus on specific pieces without dependencies interfering. This makes unit tests more reliable and easier to write and maintain.
```cpp
class MockLogger : public Logger {
public:
    MOCK_METHOD(void, log, (const std::string&));
};
```
x??

---

#### Template Method Design Pattern
The template method design pattern defines the skeleton of an algorithm in a method, deferring some steps to subclasses. This allows for flexible implementations where concrete methods can be overridden.
:p What is the purpose of the Template Method design pattern?
??x
The Template Method pattern provides a flexible structure by defining the basic algorithm's skeleton and allowing subclasses to define certain steps of that algorithm without changing its structure. This promotes code reuse and makes it easier to extend functionality.
```cpp
class AbstractClass {
protected:
    virtual void step1() = 0;
    virtual void step2() = 0;

public:
    void templateMethod() {
        // Define the basic skeleton of an algorithm
        step1();
        step2();
    }
};

class ConcreteClass : public AbstractClass {
private:
    void step1() override { /* Implementation */ }
    void step2() override { /* Implementation */ }
};
```
x??

---

#### Minimize Dependencies
Dependencies are a critical aspect of software design. Reducing dependencies is essential for maintaining and evolving your software over time.

Minimizing dependencies can be achieved through various design patterns that help separate concerns, manage abstraction, and encapsulate functionality.

:p How does minimizing dependencies impact the maintainability of software?
??x
Minimizing dependencies improves the maintainability of software by making it easier to understand, change, and test. When components are less dependent on each other, changes in one part of the system are less likely to affect other parts, reducing the risk of introducing bugs or breaking existing functionality.

For example, consider a `PaymentProcessor` class that directly depends on a `DatabaseConnection` class:
```cpp
class PaymentProcessor {
public:
    void processPayment() {
        DatabaseConnection db;
        // Process payment using database connection
    }
};
```
By introducing a dependency injection mechanism or using an interface for the database, we can decouple these classes and make them more modular.

```cpp
class PaymentProcessor {
private:
    IDatabaseConnection* _db;

public:
    void setDatabase(IDatabaseConnection* db) { _db = db; }
    void processPayment() {
        // Process payment using injected database connection
    }
};
```
x??

---

#### Separate Concerns
Separating concerns is a fundamental principle in software design. It involves breaking down complex systems into smaller, more manageable parts that handle specific responsibilities.

By separating concerns, you can enhance the readability and maintainability of your code. Each part of the system focuses on one aspect or responsibility, making it easier to understand and modify without affecting unrelated components.

:p How does separating concerns improve software design?
??x
Separating concerns improves software design by allowing different parts of the application to be developed, tested, and maintained independently. For example, in a game development project, you might separate concerns into modules such as UI, physics, networking, and AI. Each module handles its specific responsibilities, making it easier to isolate issues and update functionalities without disrupting other components.

Here’s an example using the Strategy design pattern, which separates concerns by allowing different behaviors (strategies) to be plugged in:

```cpp
class GameCharacter {
private:
    std::unique_ptr<Behavior> _behavior;

public:
    void setBehavior(std::unique_ptr<Behavior> behavior) { _behavior = std::move(behavior); }
    void performAction() { _behavior->execute(); }
};
```

In this case, the `GameCharacter` class delegates its specific actions (e.g., attacking or moving) to a strategy object (`_behavior`). This separation allows different behaviors to be easily swapped out without modifying the core character class.

x??

---

#### Prefer Composition Over Inheritance
Composition is often preferred over inheritance because it provides greater flexibility and modularity in software design. By using composition, you can achieve similar functionality with less coupling between classes.

Composing objects together can lead to more modular code that is easier to extend and test compared to deep class hierarchies created through inheritance.

:p Why should we prefer composition over inheritance?
??x
We should prefer composition over inheritance because it reduces tight coupling between classes, making the system more flexible and easier to maintain. Inheritance creates a rigid hierarchy where changes in one class can have cascading effects on derived classes. Composition allows us to build complex objects by combining simpler ones without imposing a fixed structure.

For example, consider a `Vehicle` class that could be extended using inheritance:
```cpp
class Vehicle {
public:
    virtual void drive() = 0;
};

class Car : public Vehicle {
public:
    void drive() override {
        // Specific car driving logic
    }
};
```

In this case, any change to the base `Vehicle` class would require modifications in all derived classes. Using composition, we can achieve similar functionality with more flexibility:

```cpp
class Engine;
class Car {
private:
    std::unique_ptr<Engine> _engine;

public:
    void setEngine(std::unique_ptr<Engine> engine) { _engine = std::move(engine); }
    void drive() {
        // Use the engine to drive the car
    }
};
```

By using composition, we can easily swap out different engines for our car without modifying the `Car` class or its hierarchy.

x??

---

#### Prefer Non-Intrusive Design
Non-intrusive designs allow you to add new functionality by adding code rather than modifying existing code. This approach enhances flexibility and maintainability because it avoids introducing side effects in other parts of the system.

Non-intrusive patterns like `Decorator`, `Adapter`, and others promote loose coupling, making your design more modular and easier to extend.

:p What is a non-intrusive design?
??x
A non-intrusive design allows you to add new functionality by adding code without modifying existing code. This approach enhances flexibility and maintainability because it avoids introducing side effects in other parts of the system. Non-intrusive designs promote loose coupling, making your design more modular and easier to extend.

For example, consider a `Logger` class that logs messages:

```cpp
class Logger {
public:
    void log(const std::string& message) { /* Log the message */ }
};
```

Instead of modifying this class every time we need new functionality, we can use composition (decorator pattern) to add additional behavior non-intrusively:

```cpp
class ExtendedLogger : public Logger {
private:
    std::unique_ptr<Logger> _innerLogger;

public:
    void setLogger(std::unique_ptr<Logger> logger) { _innerLogger = std::move(logger); }
    void log(const std::string& message) override {
        // Perform some additional logic
        _innerLogger->log(message);
    }
};
```

In this example, `ExtendedLogger` adds new functionality (additional logging logic) without modifying the original `Logger` class.

x??

---

#### Prefer Value Semantics Over Reference Semantics
Value semantics allow you to use values instead of pointers and references. This approach can simplify code and avoid common issues like null pointer exceptions, dangling pointers, and lifetime dependencies.

C++ is well-suited for value semantics due to its support for move semantics and value types.

:p Why should we prefer value semantics?
??x
We should prefer value semantics because they can simplify code and avoid common issues such as null pointer exceptions, dangling pointers, and lifetime dependencies. Value semantics allow you to work directly with values rather than references or pointers, which can make your code more straightforward and easier to reason about.

For example, consider a `Point` class that uses reference semantics:

```cpp
class Point {
public:
    int x, y;
};
```

If you pass around instances of this class by value, you might encounter issues like null pointer exceptions or dangling references. Using values instead can help mitigate these problems:

```cpp
struct Point { // Use C++17's std::variant for mixed types
    int x, y;
};

// Example function using value semantics
void drawPoint(const Point& p) {
    // Directly use the point without worrying about null or dangling references
}
```

In this example, `drawPoint` directly uses a `const` reference to the `Point`, avoiding potential issues related to null pointers or dangling references.

x??

---

