# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 28)

**Starting Chapter:** The Prototype Design Pattern Explained

---

#### Prototype Design Pattern Intent and Background
The Prototype design pattern is one of the five creational design patterns identified by the Gang of Four. Its primary intent is to provide an abstract way of creating copies of some object, without specifying the exact class of object that will be copied.

The core idea behind this pattern is that you have a prototype instance (an original) and create new objects by copying this prototype. This avoids complex constructor calls or factory methods for creating objects.
:p What is the main intent of the Prototype design pattern?
??x
The main intent of the Prototype design pattern is to provide an abstract way of creating copies of some object, without specifying the exact class of object that will be copied.
x??

---

#### Prototype Design Pattern in Animal Base Class
In order to implement the prototype pattern, you need a base class with a pure virtual `clone` function. This ensures that derived classes must implement their own version of cloning.

The `Animal` base class is updated to include this functionality:
```cpp
class Animal {
public:
    virtual ~Animal() = default;
    virtual void makeSound() const = 0; // Pure virtual method for making sound

    virtual std::unique_ptr<Animal> clone() const = 0; // Prototype design pattern
};
```
:p How is the prototype pattern implemented in the `Animal` base class?
??x
The prototype pattern is implemented in the `Animal` base class by declaring a pure virtual `clone()` function. This ensures that any derived classes must implement their own version of cloning, providing an abstract way to create copies without knowing the exact type.
x??

---

#### Prototype Design Pattern in Sheep Derived Class
The `Sheep` class implements the prototype pattern by overriding the `clone` method.

Here is how it is done:
```cpp
class Sheep : public Animal {
public:
    explicit Sheep(std::string name) : name_{std::move(name)} {}

    void makeSound() const override { std::cout << "baa "; }

    std::unique_ptr<Animal> clone() const override; // Prototype design pattern

private:
    std::string name_;
};

std::unique_ptr<Animal> Sheep::clone() const {
    return std::make_unique<Sheep>(*this);
}
```
:p How does the `Sheep` class implement the prototype pattern?
??x
The `Sheep` class implements the prototype pattern by overriding the `clone()` method. This method returns a copy of the current object as an `std::unique_ptr<Animal>`. The implementation uses `std::make_unique` to create a new `Sheep` instance with the same state as the original.
x??

---

#### Dependency Inversion Principle (DIP)
The prototype design pattern aligns well with the Dependency Inversion Principle (DIP), which states that high-level modules should not depend on low-level modules, but both should depend on abstractions.

This is shown in the updated dependency graph:
```plaintext
+----------------+
|  Client        |
+--------+-------+
          |
          v
+-----------------+
|   Animal         |
+---+------------+
    |            |
    v            v
+--------------+ +-------------+
| Sheep        | | Dog          |
+--------------+ +-------------+
```
:p How does the prototype pattern support the Dependency Inversion Principle (DIP)?
??x
The prototype pattern supports DIP by ensuring that high-level modules depend on abstractions, not concrete implementations. In this case, clients can work with `Animal` instances and request clones from any derived class without knowing which specific animal type is being used.
x??

---

#### Use of Unique_ptr for Cloning
Using `std::unique_ptr<Animal>` for cloning ensures that the created object's lifetime is managed properly by the unique pointer.

Here is an example:
```cpp
std::unique_ptr<Animal> dolly = std::make_unique<Sheep>("Dolly");
dolly->makeSound();
```
:p Why is `std::unique_ptr` used in the cloning process?
??x
`std::unique_ptr` is used in the cloning process because it manages the memory and lifetime of the cloned object. This ensures that manual cleanup is not required, making the code safer and easier to maintain.
x??

---

#### Prototype Design Pattern vs. Slicing Problem
The prototype pattern avoids issues like slicing by ensuring that only complete objects are copied.

Slicing occurs when a subclass's data members are assigned to a base class pointer, resulting in loss of derived class information.

:p How does the prototype pattern prevent the slicing problem?
??x
The prototype pattern prevents the slicing problem by returning an `std::unique_ptr<Animal>` from the `clone` method. This ensures that the complete object is copied, preserving all its derived class-specific data and behavior.
x??

---

#### Prototype Design Pattern Overview
The Prototype design pattern is a classic object-oriented programming (OOP) solution for creating new objects based on an existing instance. This pattern avoids using constructors, making it ideal when you need to clone complex or resource-intensive objects.

In the context of the Sheep class, implementing `clone()` allows for the creation of exact copies without needing explicit constructor information.
:p How does the Prototype design pattern help in cloning objects?
??x
The Prototype design pattern helps by providing a way to create new instances based on existing ones. This is achieved through a clone() method which returns a copy of the current object, often using `std::make_unique` for unique ownership semantics.

Example:
```cpp
class Sheep {
public:
    std::string name;
    
    // Constructor and other methods
    
    virtual std::unique_ptr<Animal> clone() const {
        return std::make_unique<Sheep>(*this);
    }
};
```
x??

---
#### C++ Unique Pointer and Clone Method
C++11 introduced `std::unique_ptr` as a smart pointer for managing unique ownership. It ensures that only one unique owner exists at any time, which is particularly useful in scenarios like cloning objects.

In the Sheep class, the `clone()` method uses `std::make_unique<Sheep>(*this)` to create an exact copy of the object.
:p What does `std::make_unique` do in C++?
??x
`std::make_unique` is a utility function that creates and returns a `std::unique_ptr`. It ensures proper placement new syntax, which is required for objects with custom constructors or placement new expressions. This function helps avoid memory leaks by ensuring the unique ownership semantics of `std::unique_ptr`.

Example:
```cpp
#include <memory>

struct Sheep {
    std::string name;
    
    // Constructor and other methods
    
    virtual std::unique_ptr<Animal> clone() const {
        return std::make_unique<Sheep>(*this);
    }
};
```
x??

---
#### Comparison with `std::variant` for Copying
While the Prototype design pattern provides a flexible way to clone objects, modern C++ also offers `std::variant`. This can be used for value semantics and copying without needing a `clone()` method. However, it requires all possible types to be known in advance.

Example:
```cpp
#include <variant>

class Dog {};
class Cat {};
class Sheep {};

int main() {
    std::variant<Dog, Cat, Sheep> animal1{ /* ... */ };

    auto animal2 = animal1;  // Copy operation

    return EXIT_SUCCESS;
}
```
:p How does `std::variant` differ from the Prototype design pattern?
??x
`std::variant` is different from the Prototype design pattern because it is a value type that holds one of several types at any given time. It provides copy semantics, meaning you can directly copy an object without needing a custom `clone()` method. However, this comes with the constraint that all possible types must be known and specified in advance.

In contrast, the Prototype design pattern allows for dynamic cloning based on existing instances but requires implementing a `clone()` method.
x??

---
#### Value Semantics vs. Reference Semantics
Value semantics refer to objects where you can copy them as if they were simple values (integers, strings, etc.). This means that when you assign or pass by value, the object is duplicated.

Reference semantics, on the other hand, mean that an object's state is shared across multiple references, and changes made through one reference affect all others. In modern C++, `std::unique_ptr` and `std::variant` exemplify these concepts.

Example:
```cpp
class Sheep {
public:
    std::string name;

    Sheep(const Sheep& other) : name(other.name) {}  // Copy constructor

    Sheep(Sheep&& other) noexcept : name(std::move(other.name)) {}  // Move constructor
};

int main() {
    Sheep sheep1("Dolly");
    Sheep sheep2 = sheep1;  // Using copy semantics

    return EXIT_SUCCESS;
}
```
:p What is the difference between value and reference semantics in C++?
??x
Value semantics refer to objects that can be copied, allowing you to treat them as simple values. This means that when you assign or pass by value, a new object is created with its own state.

Reference semantics mean that an object's state is shared across multiple references, so changes made through one reference affect all others.

In C++, `std::unique_ptr` and `std::variant` embody these concepts differently. `std::unique_ptr` ensures unique ownership (value-like behavior), while `std::variant` supports value semantics for a closed set of types.
x??

---

#### Prototype Design Pattern - Value Semantics vs Reference Semantics

Background context: The Prototype design pattern is a creational design pattern that enables creating new objects by copying (cloning) existing instances. It’s particularly useful when dealing with complex object graphs or inheritance hierarchies where direct construction might be challenging.

The primary issue here is that the prototype-based approach relies on reference semantics, which means objects are copied rather than duplicated in value. This can lead to some drawbacks such as performance overhead due to indirection and memory fragmentation if dynamic allocation is used.

:p What are the key challenges with using the Prototype design pattern for creating abstract copies of complex objects?
??x
The main challenges include:
1. **Performance Overhead**: Indirection via pointers adds latency.
2. **Memory Fragmentation**: Dynamic memory allocation can lead to inefficient use of memory.
3. **Complexity in Implementation**: The need for a `clone()` function or virtual clone method increases complexity.

These issues arise because the pattern operates under reference semantics, meaning that cloning involves copying references rather than duplicating values directly.

??x
To mitigate these issues, one could consider using value semantics where applicable and avoid deep copies where shallow ones suffice.
x??

---

#### External Polymorphism Design Pattern

Background context: The External Polymorphism design pattern addresses the limitations of the Strategy pattern by further separating polymorphic behavior from the core objects. It enables treating non-polymorphic types as if they were polymorphic, allowing for a more modular and decoupled system.

The key idea is to introduce an external hierarchy that defines the required operations (e.g., `draw()`). This hierarchy then wraps around concrete shapes, providing a way to handle them uniformly without requiring them to inherit from a base class or implement any virtual functions directly.

:p How does the External Polymorphism design pattern achieve polymorphic behavior for non-polymorphic types?
??x
The external hierarchy introduces an abstract `ShapeConcept` that defines all the operations expected of shapes, such as drawing. The actual concrete shapes (`Circle`, `Square`, etc.) do not need to implement these operations directly; instead, they are wrapped by a `ShapeModel` class template instantiated for each specific shape type.

This separation ensures that:
- Concrete shapes remain decoupled from the polymorphic behavior.
- Operations like drawing can be isolated and managed separately.

The external hierarchy effectively decouples the core objects (shapes) from their behaviors (e.g., drawing), leading to a cleaner and more maintainable design.

??x
Here is an example of how this might look in code:
```cpp
class ShapeConcept {
public:
    virtual ~ShapeConcept() = default;
    virtual void draw(const Circle& circle, /*...*/) const = 0;
};

template <typename T>
class ShapeModel : public ShapeConcept {
private:
    T shape_;
public:
    explicit ShapeModel(T&& shape) : shape_(std::move(shape)) {}
    
    // Implementation of the required operations
    void draw(const Circle& circle, /*...*/) const override {
        // Logic to handle drawing using the wrapped shape
    }
};

// Usage example
Circle c(5.0);
ShapeConcept* concept = new ShapeModel<Circle>(std::move(c));
concept->draw(c, /*some arguments*/);
```
x??

---
#### Value Semantics vs Reference Semantics

Background context: Value semantics and reference semantics are two different approaches in C++ (and many other languages) for handling object instances. Value semantics means that objects are duplicated when copied or assigned, whereas reference semantics means they share the same instance.

In the context of the Prototype design pattern, value semantics would allow for more efficient copying since it duplicates values rather than references. However, this approach requires careful handling to ensure correctness and efficiency, especially with complex object graphs.

:p Why might one prefer value semantics over reference semantics in certain scenarios?
??x
Value semantics are preferable when:
1. **Efficiency**: Direct duplication of values is often faster than copying or moving references.
2. **Immutability**: Value types can be used more safely and predictably, especially in concurrent programming.
3. **Simplicity**: Abstracting away the need for deep copies and ensuring that modifications do not affect other instances.

However, value semantics come with their own challenges such as increased memory usage and the complexity of managing resource cleanup (e.g., RAII).

??x
For example, consider a `Point` class:
```cpp
class Point {
public:
    double x, y;

    // Copy constructor
    Point(const Point& other) : x(other.x), y(other.y) {}

    // Move constructor
    Point(Point&& other) noexcept : x(std::move(other.x)), y(std::move(other.y)) {}
};
```
x??

---

