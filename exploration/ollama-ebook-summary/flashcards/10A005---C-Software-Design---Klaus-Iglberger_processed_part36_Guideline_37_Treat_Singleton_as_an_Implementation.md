# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 36)

**Starting Chapter:** Guideline 37 Treat Singleton as an Implementation Pattern Not a Design Pattern

---

---
#### Singleton Pattern Overview
In this chapter, we delve into the Singleton pattern, a design concept often criticized for its global nature but also recognized for its utility. The Singleton is a design pattern that restricts the instantiation of a class to one "single" instance. This makes it particularly useful in scenarios where exactly one object is needed to coordinate actions across the system.
The C++ Standard Library contains several Singleton-like instances, indicating that Singletons can be implemented effectively and beneficially.

:p What are the key points about the Singleton pattern discussed in this chapter?
??x
This question aims to test your understanding of the main topics covered in the text regarding the Singleton pattern. Key points include its restricted instantiation nature, utility for global state management, and the existence of Singleton-like instances within the C++ Standard Library.

```cpp
// Example of a simple Singleton class (Meyers' Singleton)
class Singleton {
private:
    static Singleton* instance;
    // Private constructor to prevent direct instantiation
    Singleton() {}

public:
    // Static method to access the single instance
    static Singleton& getInstance() {
        if (!instance) {
            instance = new Singleton();
        }
        return *instance;
    }

    // Example of a member function
    void doSomething() {
        // Perform some action
    }

private:
    ~Singleton(); // Private destructor to prevent deletion
};
```
x??

---
#### Treating Singleton as an Implementation Pattern, Not a Design Pattern
The text suggests that Singletons are often considered antipatterns due to their global nature. However, the chapter argues against treating Singleton as a design pattern and instead views it as an implementation pattern. This perspective emphasizes its use in specific scenarios where exactly one instance is required.

:p Why does the author suggest treating Singleton as an implementation pattern rather than a design pattern?
??x
The author suggests this because Singletons are typically used for situations requiring global state or coordinated behavior, such as managing resources or providing access to a single object. Treating it as an implementation detail focuses on its practical application while acknowledging potential drawbacks like tight coupling and global state issues.

```cpp
// Example of using Singleton within a class
class ResourceManager {
public:
    static Singleton& getResourceManager() {
        return Singleton::getInstance();
    }

private:
    // Private constructor and destructor to prevent direct instantiation and deletion
    ResourceManager() {}
    ~ResourceManager() {}
};
```
x??

---
#### Designing Singletons for Change and Testability
The text acknowledges that sometimes global aspects are necessary in code, making the Singleton pattern useful. However, it also highlights common issues like global state, strong dependencies, and reduced changeability and testability.

:p What are the main problems associated with using Singletons as described in the text?
??x
The main problems include:
- Global state: Singletons can lead to tightly coupled code that relies on a shared context.
- Strong, artificial dependencies: Code may become dependent on Singleton instances, making it harder to isolate components for testing.
- Impeded changeability and testability: Modifying or testing Singleton behavior is more challenging due to its single-instance nature.

```cpp
// Example of refactoring a Singleton for better testability
class Logger {
public:
    void log(const std::string& message) {
        // Log implementation
    }
};

// MockLogger can be used in tests instead of the real Logger singleton
class MockLogger : public Logger {
public:
    MOCK_METHOD1(log, void(const std::string&));
};
```
x??

---
#### Singleton Benefits with Changeability and Testability
The text argues that proper software design can combine the benefits of Singletons (e.g., global state management) with excellent changeability and testability. This suggests a balanced approach to using Singletons.

:p How does the text propose to combine the benefits of Singletons with good changeability and testability?
??x
The text proposes a balanced approach by integrating Singletons in a way that minimizes their drawbacks. It suggests designing them carefully to reduce global state dependencies, allowing for better isolation during testing. Additionally, using techniques like dependency injection can help manage Singleton instances more flexibly.

```cpp
// Example of using dependency injection with a Singleton-like pattern
class Service {
public:
    void useService(Logger& logger) {
        // Use the passed-in logger instance instead of a singleton
    }
};

// Testing by passing in a mock logger
TEST(ServiceTest, LogMessage) {
    MockLogger mockLogger;
    Service service;
    service.useService(mockLogger);
}
```
x??

---

#### Singleton Pattern Overview
Background context: The Singleton pattern is often misunderstood as a design pattern, but it is more accurately described as an implementation detail or anti-pattern due to its restrictions and potential issues. The intent of the Singleton pattern is to ensure that a class has only one instance and provide a global point of access to it.
:p What is the Singleton pattern intended to achieve?
??x
The Singleton pattern ensures that a class has only one instance and provides a global point of access to this instance, often referred to as the "Highlander" situation where there can be only one. This pattern is useful when you need to restrict instantiation to exactly one object.
??x

---

#### Meyers' Singleton Implementation
Background context: One popular implementation of the Singleton pattern is the Meyers' Singleton, which avoids using a global variable or static class members directly and instead uses private constructors and friend functions with static local variables for thread-safe initialization. This approach ensures that no two instances can be created.
:p How does Meyers' Singleton ensure there is only one instance?
??x
Meyers' Singleton ensures there is only one instance by using a private constructor and a public static function to return the single instance. The static local variable within this function is initialized in a thread-safe manner on its first access, and all subsequent calls return the same object.
```cpp
class Database final {
public:
    static Database& instance() 
    { 
        static Database db;  // Thread-safe initialization of the unique instance
        return db; 
    } 

    bool write( /*some arguments*/ );
    bool read( /*some arguments*/ ) const;
    // ... More database-specific functionality

private:
    Database() {}        // Private constructor to prevent instantiation from outside

    Database(const Database&) = delete;     // Deleted copy constructor
    Database& operator=(const Database&) = delete;  // Deleted assignment operator

    Database(Database&&) = delete;          // Deleted move constructor
    Database& operator=(Database&&) = delete; // Deleted move assignment operator
};
```
x??

---

#### C++17 Aggregates and Default Constructors
Background context: In C++, a class is considered an aggregate if it has only public or protected data members, no user-defined constructors (including the default constructor), and no virtual functions. Aggregate initialization allows value-initialization of objects, which can bypass private member initializations.
:p Why does Meyers' Singleton not use a default constructor?
??x
Meyers' Singleton does not use a default constructor to prevent value-initialization via braces ({}) in C++17. If the default constructor were present and defaulted, it would allow aggregate initialization, which could create an instance of the class even though the constructor is private.
```cpp
class Database  {
public:
    // ... As before

private:
    Database() = default;  // Compiler generated default constructor

    // ... As before
};

int main() {
    Database db;         // Does not compile: Default initialization
    Database db{};       // Works, since value initialization results in aggregate initialization, because Database is an aggregate type
}
```
x??

---

#### Thread-Safe Singleton Initialization
Background context: The Meyers' Singleton pattern ensures thread-safety by using a static local variable inside the instance() function. This guarantees that the variable is initialized only once and in a thread-safe manner.
:p How does the instance() function ensure thread safety?
??x
The instance() function ensures thread safety by initializing the static local variable "db" on its first access. The initialization is performed in a thread-safe way, meaning it happens only once even if multiple threads attempt to access the function simultaneously. Subsequent calls return the already initialized object.
```cpp
class Database final {
public:
    static Database& instance() 
    { 
        static Database db;  // Thread-safe initialization of the unique instance
        return db; 
    } 

    bool write( /*some arguments*/ );
    bool read( /*some arguments*/ ) const;
    // ... More database-specific functionality

private:
    Database() {}        // Private constructor to prevent instantiation from outside

    Database(const Database&) = delete;     // Deleted copy constructor
    Database& operator=(const Database&) = delete;  // Deleted assignment operator

    Database(Database&&) = delete;          // Deleted move constructor
    Database& operator=(Database&&) = delete; // Deleted move assignment operator
};
```
x??

---

#### Singleton Global Point of Access
Background context: The Singleton pattern provides a global point of access to the single instance through a static function. This function is responsible for returning the unique instance of the class.
:p What is the role of the instance() function in the Singleton pattern?
??x
The instance() function serves as the global point of access to the single instance of the Singleton class. It ensures that only one instance can exist and provides a way to obtain this instance, making it accessible throughout the application through a static reference.
```cpp
class Database final {
public:
    static Database& instance() 
    { 
        static Database db;  // Thread-safe initialization of the unique instance
        return db; 
    } 

    bool write( /*some arguments*/ );
    bool read( /*some arguments*/ ) const;
    // ... More database-specific functionality

private:
    Database() {}        // Private constructor to prevent instantiation from outside

    Database(const Database&) = delete;     // Deleted copy constructor
    Database& operator=(const Database&) = delete;  // Deleted assignment operator

    Database(Database&&) = delete;          // Deleted move constructor
    Database& operator=(Database&&) = delete; // Deleted move assignment operator
};
```
x??

#### Singleton Pattern and Design Patterns

Background context explaining the concept. A design pattern is a reusable solution to common software design problems that helps achieve specific goals or objectives. The provided text discusses whether the Singleton pattern qualifies as a true design pattern, focusing on its role in managing dependencies and abstractions.

If applicable, add code examples with explanations:
```cpp
class Database {
public:
    static Database& instance() {
        static Database db;
        return db;
    }

private:
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    // Database implementation details...
};
```
:p What is the Singleton pattern primarily targeted towards?
??x
The Singleton pattern is focused on restricting the number of instantiations to exactly one, rather than managing or reducing dependencies between software entities. This makes it an implementation detail rather than a design pattern aimed at software design.
x??

---

#### Properties of Design Patterns

Background context explaining the concept. The text outlines what constitutes a design pattern based on its name, intent, abstraction, and proven usage across different languages.

If applicable, add code examples with explanations:
```java
public class SingletonPattern {
    // Code for demonstrating properties of design patterns
}
```
:p What are the key properties that define a design pattern according to the text?
??x
The key properties that define a design pattern include having a name, carrying an intent, introducing an abstraction (through base classes or templates), and being proven over time. The Singleton pattern lacks these characteristics as it focuses on restricting instantiation rather than providing abstractions.
x??

---

#### Singleton Pattern as an Implementation Detail

Background context explaining the concept. The text argues that while the Singleton pattern is commonly used in various programming languages, its primary focus is on implementation details.

If applicable, add code examples with explanations:
```cpp
// Example of a Singleton class in C++
class Singleton {
private:
    static Singleton* instance;
    // private constructor to prevent direct instantiation

public:
    static Singleton& getInstance() {
        if (instance == nullptr) {
            instance = new Singleton();
        }
        return *instance;
    }

    ~Singleton();

private:
    Singleton() {}
};
```
:p Why might the Singleton pattern be listed as a design pattern despite not fitting traditional definitions?
??x
The Singleton pattern may still be considered a design pattern due to its common usage across different programming languages, even though it primarily focuses on implementation details rather than managing dependencies or introducing abstractions. This widespread use suggests that it is seen as a useful solution in certain contexts.

However, the text argues that calling it a design pattern could lead to confusion because it does not align with traditional definitions of software design patterns.
x??

---

#### Distinguishing Between Design Patterns and Implementation Details

Background context explaining the concept. The text introduces the term "implementation pattern" as a way to differentiate between design patterns that address dependencies and implementation details focused on managing instantiation.

If applicable, add code examples with explanations:
```java
// Example of an implementation detail in Java
public class SingletonImplementation {
    private static SingletonImplementation instance;

    private SingletonImplementation() {}

    public static SingletonImplementation getInstance() {
        if (instance == null) {
            synchronized (SingletonImplementation.class) {
                if (instance == null) {
                    instance = new SingletonImplementation();
                }
            }
        }
        return instance;
    }
}
```
:p How does the text suggest we should differentiate between design patterns and implementation details?
??x
The text suggests using the term "implementation pattern" to distinguish between patterns that address dependencies and decoupling (true design patterns) versus those focused on managing instantiation (like Singleton). This distinction is important for clear communication about software design concepts.
x??

---

