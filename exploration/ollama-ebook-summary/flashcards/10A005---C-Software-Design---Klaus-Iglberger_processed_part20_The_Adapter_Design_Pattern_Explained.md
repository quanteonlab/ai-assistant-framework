# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 20)

**Starting Chapter:** The Adapter Design Pattern Explained

---

#### Adapter Design Pattern
Background context explaining the concept. The Adapter design pattern is a classic GoF (Gang of Four) pattern aimed at standardizing interfaces and adding functionality non-intrusively to existing inheritance hierarchies.

The pattern involves creating an adapter class that wraps around an existing object with a different interface, allowing it to fit into another system's interface expectations. The key idea is to make classes work together even if their interfaces are incompatible initially.

In the provided context, you have a `Document` base class and its derived classes like `Word`. You need to integrate the `OpenPages` class from an external library which has different methods (`convertToBytes`) than what your system expects (e.g., `serialize`, `exportToJSON`). The Adapter pattern can help in this scenario.

:p How does the Adapter design pattern solve interface compatibility issues?
??x
The Adapter design pattern solves interface compatibility issues by wrapping around an object and providing a new interface that matches the client's expectations. This is achieved without modifying the existing class or breaking its encapsulation.

Here’s how it works with the provided example:

1. The `Pages` class acts as an adapter, inheriting from the `Document` base class.
2. It provides the required methods (`exportToJSON`) by calling corresponding functions on the `OpenPages` object internally.

```cpp
class Pages : public Document {
public:
    // Other necessary constructors and member variables

    void exportToJSON(/* parameters */) const override {
        exportToJSONFormat(pages, /* parameters */);
    }

private:
    OpenPages pages;  // The third-party OpenPages class instance wrapped by the adapter
};
```

In this example, `exportToJSON` in the `Document` interface is adapted to call `exportToJSONFormat` on an `OpenPages` object.

x??
---
#### UML Diagram for Adapter Pattern
UML diagram representation of the Adapter design pattern as described in the text. It includes the existing `Document` hierarchy and the new `Pages` class which acts as an adapter.

:p What does Figure 6-1 illustrate?
??x
Figure 6-1 illustrates the UML representation of the Adapter design pattern. Specifically, it shows how a `Pages` class acts as an adapter to integrate with the existing `Document` hierarchy by adapting its interface to match client expectations.

The diagram typically includes:

- The `Document` base class.
- Various derived classes like `Word`.
- The `Pages` class which is also derived from `Document`.
- The relationships and method mappings, highlighting how `Pages` adapts the `exportToJSON` method by internally calling `exportToJSONFormat`.

The diagram visually demonstrates the concept of adapting interfaces to ensure compatibility.

x??
---
#### Object Adapter
In this scenario, an object adapter is used to convert an existing class's interface into one that clients expect. It typically involves creating a new class that contains a reference to the existing class and provides the desired interface.

:p What type of adapter is being used in the provided example?
??x
The type of adapter being used in the provided example is an **Object Adapter**. This means that `Pages` acts as a wrapper around an instance of `OpenPages`. The `Pages` class contains an instance of `OpenPages` and provides methods from its interface (like `exportToJSON`) by delegating to the corresponding methods on the wrapped `OpenPages` object.

Here’s how this works in code:

```cpp
class Pages : public Document {
public:
    // Constructor that initializes the OpenPages object
    Pages(OpenPages& open_pages) : pages(open_pages) {}

    void exportToJSON(/* parameters */) const override {
        exportToJSONFormat(pages, /* parameters */);
    }

private:
    OpenPages& pages;  // Reference to the OpenPages instance
};
```

In this implementation, `Pages` is an adapter that adapts the interface of `OpenPages` to match the `Document` class's expectations.

x??
---

#### Adapter Design Pattern Overview
The Adapter design pattern is used to allow objects with incompatible interfaces to collaborate. It works by converting the interface of a class into another interface clients expect. The primary use case involves integrating third-party implementations or adapting existing APIs to fit new requirements without modifying the original code.

:p What is the main purpose of the Adapter design pattern?
??x
The main purpose of the Adapter design pattern is to allow objects with incompatible interfaces to collaborate by converting one interface into another that clients expect. This allows for non-intrusive integration, meaning you can add an adapter without altering the underlying implementation.
x??

---

#### Pages Class as a Concrete Example
The `Pages` class serves as a concrete example of adapting the `OpenPages` class to fit the `Document` interface by forwarding calls.

:p How does the `Pages` class adapt the `OpenPages` class to the `Document` interface?
??x
The `Pages` class adapts the `OpenPages` class to the `Document` interface by forwarding calls. For example, when `exportToJSON()` is called on a `Document`, it forwards the call to the `exportToJSONFormat()` function from `OpenPages`. Similarly, `serialize()` is forwarded to `convertToBytes()`. This non-intrusive nature allows easy integration without modifying the original implementation.
x??

---

#### Object Adapter vs. Class Adapter
There are two types of adapters: object adapter and class adapter.

:p What distinguishes an object adapter from a class adapter?
??x
An object adapter stores an instance of the wrapped type, whereas a class adapter inherits from it (if possible, non-publicly). The object adapter is generally more flexible because you can use it for all types within a hierarchy by storing a pointer to the base class. Class adapters inherit directly from the adapted type and implement the expected interface.
x??

---

#### Flexibility of Object Adapters
Object adapters offer greater flexibility compared to class adapters.

:p Why are object adapters considered more flexible than class adapters?
??x
Object adapters are more flexible because they store an instance of the wrapped type, allowing you to use them for all types within a hierarchy by storing a pointer to the base class. This provides significant flexibility. Class adapters inherit directly from the adapted type and can only be used with that specific type, limiting their applicability.
x??

---

#### Usage Context for Object Adapters
The `Pages` class is an example of an object adapter.

:p In what scenario would you use the Pages class as described in the text?
??x
You would use the `Pages` class to integrate a third-party implementation (e.g., `OpenPages`) into your existing hierarchy by adapting its interface without modifying it. This approach ensures non-intrusive integration and aligns with design principles like the Single-Responsibility Principle (SRP) and Open-Closed Principle (OCP).
x??

---

#### Code Example of Object Adapter
Here is a simplified example of an object adapter.

:p Provide a code snippet for an object adapter.
??x
```cpp
class Document {
public:
    virtual void exportToJSON() const = 0;
    virtual void serialize(ByteStream& bs) const = 0;
};

class OpenPages {
private:
    // Implementation details

public:
    void exportToJSONFormat(const OpenPages&, /*...*/);
    void convertToBytes(/*...*/);
};

// Object Adapter class
class Pages : public Document {
private:
    OpenPages pages;

public:
    void exportToJSON() const override {
        exportToJSONFormat(pages, /*...*/);
    }

    void serialize(ByteStream& bs) const override {
        pages.convertToBytes(/*...*/);
    }
};
```
x??

---

#### Code Example of Class Adapter
Here is a simplified example of a class adapter.

:p Provide a code snippet for a class adapter.
??x
```cpp
class Document {
public:
    virtual void exportToJSON() const = 0;
    virtual void serialize(ByteStream& bs) const = 0;
};

class OpenPages : public Document {
private:
    // Implementation details

public:
    void exportToJSONFormat(/*...*/);
    void convertToBytes(/*...*/);
};

// Class Adapter class
class Pages : public Document, private OpenPages {
public:
    void exportToJSON() const override {
        exportToJSONFormat(*this, /*...*/);
    }

    void serialize(ByteStream& bs) const override {
        this->convertToBytes(/*...*/);
    }
};
```
x??

---

#### Summary of Adapter Design Pattern
The Adapter design pattern allows you to integrate third-party implementations into your system without modifying their code. It supports both object and class adapters, with object adapters generally providing more flexibility.

:p Summarize the key points about the Adapter design pattern.
??x
The Adapter design pattern enables integration between incompatible interfaces by converting one interface into another that clients expect. Key points include:
- The `Pages` class serves as a concrete example of adapting the `OpenPages` class to fit the `Document` interface.
- Object adapters store an instance and provide greater flexibility compared to class adapters, which inherit directly from the adapted type.
- Both object and class adapters support non-intrusive integration, aligning with design principles like SRP and OCP.
x??

#### Adapter Design Pattern: Naming Conventions
Background context explaining the concept of naming conventions and their importance in design patterns. In particular, this card focuses on whether to name a class after the pattern it implements or keep a more generic name.

:p Should the class implementing the Adapter design pattern be named based on the pattern's name or can it have a generic name?
??x
In this case, naming the class `PagesAdapter` would communicate that you are using the Adapter pattern. However, if you find that the adapter nature is an implementation detail and not crucial for understanding the overall functionality of the code, using a more generic name like `PageLoader` or `PageManager` might be appropriate.

For example:
```cpp
class PageLoader : public VirtualBaseClass {
    // Implementation details here.
};
```

This approach allows developers to focus on the main functionality without being distracted by design pattern names. Ultimately, this decision should be made based on the specific context and how critical the Adapter pattern is to understanding the class's role in the code.

x??

---
#### Stack Abstract Base Class
Background context explaining the use of abstract base classes for defining interfaces. This card focuses on whether implementing a stack via an abstract base class makes sense from a performance perspective.

:p Do you suggest using an abstract base class for implementing a stack, and if so, why?
??x
While it is possible to define a stack interface using an abstract base class, doing so may introduce unnecessary overhead due to virtual function calls. In C++, this approach can be inefficient because of the performance cost associated with dynamic dispatch.

A more efficient implementation would use templates directly:
```cpp
template<typename T>
class Stack {
public:
    virtual ~Stack() = default;
    virtual void push(const T& value) = 0;
    virtual void pop() = 0;
    virtual bool empty() const = 0;
    virtual size_t size() const = 0;
};
```

However, for the purpose of demonstrating the Adapter pattern or when working within a design that mandates abstract interfaces, an abstract base class can be useful. In such cases, you might use a template with pointers to member functions (std::function) as shown in the Standard Library's container adaptors.

x??

---
#### VectorStack Implementation
Background context explaining how to implement a concrete adapter for a stack using `std::vector`. This card focuses on the implementation details of adapting an existing data structure to fit a specified interface.

:p How would you implement a stack using a vector as the underlying storage?
??x
To implement a stack using a vector, you can create a class that inherits from the abstract Stack base class and provides concrete implementations for its pure virtual functions. Here is an example:

```cpp
//---- <VectorStack.h> ----------------
#include "Stack.h"

template<typename T>
class VectorStack : public Stack<T> {
public:
    T& top() override { return vec_.back(); }
    bool empty() const override { return vec_.empty(); }
    size_t size() const override { return vec_.size(); }
    void push(const T& value) override { vec_.push_back(value); }
    void pop() override { vec_.pop_back(); }

private:
    std::vector<T> vec_;
};
```

This implementation ensures that the VectorStack class adheres to the Stack interface while leveraging the capabilities of `std::vector` for efficient stack operations.

x??

---
#### Performance Considerations with Virtual Functions
Background context explaining the performance implications of virtual functions in C++. This card focuses on why using abstract base classes can be less efficient than other approaches.

:p Why is implementing a stack via an abstract base class considered inefficient from a performance perspective?
??x
Implementing a stack via an abstract base class introduces overhead due to the need for virtual function calls. Each call involves a dynamic dispatch mechanism, which requires accessing the vtable (virtual table) at runtime. This can significantly impact performance, especially in tight loops or high-frequency operations.

A more efficient approach is to use templates directly, avoiding the need for virtual functions:
```cpp
template<typename T>
class Stack {
public:
    void push(const T& value) = 0;
    void pop() = 0;
    bool empty() const = 0;
    size_t size() const = 0;
};
```

For example, a stack implemented with `std::vector` using templates might look like this:

```cpp
template<typename T>
class VectorStack : public Stack<T> {
public:
    void push(const T& value) override { vec_.push_back(value); }
    void pop() override { if (!empty()) vec_.pop_back(); }
    bool empty() const override { return vec_.empty(); }
    size_t size() const override { return vec_.size(); }

private:
    std::vector<T> vec_;
};
```

Using templates allows for more efficient, type-safe operations without the overhead of virtual function calls.

x??

---

