# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 16)

**Starting Chapter:** Analyzing the Shortcomings of the Naive Solution

---

#### Dependency Injection through DrawStrategy

Background context: The current design allows for changing drawing behavior by providing a unique_ptr to a DrawStrategy. This is done via dependency injection, where objects are not created within a class but are passed as arguments.

:p How does the DrawStrategy allow for flexible drawing behaviors?
??x
The DrawStrategy pattern enables flexible drawing behaviors by allowing different strategies to be injected at runtime. This means you can easily switch between different drawing methods without changing the Shape classes directly. Each shape (Circle, Square) holds a unique_ptr to its specific DrawStrategy implementation.

```cpp
class Circle : public Shape {
public:
    explicit Circle(double radius, std::unique_ptr<DrawCircleStrategy> drawer)
        : radius_(radius), drawer_(std::move(drawer)) {}

    void draw(/*some arguments*/) const override {
        drawer_->draw(*this, /*some arguments*/);
    }

private:
    double radius_;
    std::unique_ptr<DrawCircleStrategy> drawer_;
};
```

x??

---

#### DrawStrategy for New Shapes

Background context: The current design introduces new DrawStrategies for each shape type to avoid coupling and allow easy addition of new shapes. However, this approach leads to cyclic dependencies between shapes and their strategies.

:p Why is it problematic to add a new shape like Triangle in the current implementation?
??x
Adding a new shape such as Triangle would require modifying the existing base class DrawStrategy, which affects all other shapes. This violates the Interface Segregation Principle (ISP), making the system less modular and harder to extend. Each time a new shape is added, every shape’s implementation needs recompilation due to their interdependencies.

```cpp
// Example of adding Triangle
class DrawTriangleStrategy {
public:
    virtual ~DrawTriangleStrategy() = default;
    virtual void draw(Triangle const& triangle, /*some arguments*/) const = 0;
};

class Triangle : public Shape {
public:
    explicit Triangle(double side, std::unique_ptr<DrawTriangleStrategy> drawer)
        : side_(side), drawer_(std::move(drawer)) {}

    void draw(/*some arguments*/) const override {
        drawer_->draw(*this, /*some arguments*/);
    }

private:
    double side_;
    std::unique_ptr<DrawTriangleStrategy> drawer_;
};
```

x??

---

#### Visitor Pattern Consideration

Background context: The current implementation of DrawStrategies resembles the Visitor pattern but falls short because it cannot easily add other operations or types. It also does not fulfill the Strategy design pattern's requirement to separate concerns.

:p Why is the current DrawStrategy implementation not a true Strategy pattern?
??x
The current DrawStrategy implementation is not a proper Strategy pattern because it tightly couples different shapes with their drawing strategies, requiring modifications to all shape classes whenever a new strategy is introduced. The Strategy pattern requires each concrete strategy class (DrawCircleStrategy, DrawSquareStrategy) to be separate and independent of the implementing class (Circle, Square).

```cpp
class DrawCircleStrategy {
public:
    virtual ~DrawCircleStrategy() = default;
    virtual void draw(Circle const& circle, /*some arguments*/) const = 0;
};

class Circle : public Shape {
public:
    explicit Circle(double radius, std::unique_ptr<DrawCircleStrategy> drawer)
        : radius_(radius), drawer_(std::move(drawer)) {}

    void draw(/*some arguments*/) const override {
        drawer_->draw(*this, /*some arguments*/);
    }

private:
    double radius_;
    std::unique_ptr<DrawCircleStrategy> drawer_;
};
```

x??

---

#### Template Strategy Class

Background context: To address the issues with cyclic dependencies and improve code reuse, using a template DrawStrategy class can be considered. This approach avoids the need for separate classes for each shape.

:p How does a template-based DrawStrategy help in this scenario?
??x
Using a template-based DrawStrategy helps by lifting the Strategy implementation into a higher level of abstraction. This reduces redundancy and allows easy extension without altering existing shape classes. Each concrete strategy can be specialized, adhering to the DRY (Don’t Repeat Yourself) principle.

```cpp
template<typename T>
class DrawStrategy {
public:
    virtual ~DrawStrategy() = default;
    virtual void draw(T const&) const = 0;
};

class OpenGLCircleStrategy : public DrawStrategy<Circle> {
public:
    explicit OpenGLCircleStrategy(/* Drawing related arguments */);
    void draw(Circle const& circle, /*...*/) const override;

private:
    // Drawing related data members
};
```

x??

---

#### Strategy Design Pattern: Overview
The Strategy design pattern is a behavioral design pattern that enables you to define a family of algorithms, encapsulate each one, and make them interchangeable. This pattern lets the algorithm vary independently from clients that use it.

Background context explaining the concept:
- The Strategy pattern allows for different behaviors to be swapped out at runtime.
- It is useful when you have multiple ways to implement an operation or function, and these need to be chosen at runtime based on certain conditions.

:p What is the main purpose of the Strategy design pattern?
??x
The main purpose of the Strategy design pattern is to decouple algorithmic behaviors from the entities that use them. This enables users to select the behavior dynamically without changing the object's structure.
x??

---

#### Strategy Design Pattern: Code Example
Background context explaining the concept:
- The provided code shows how to implement a strategy for drawing shapes using OpenGL.

:p How can you integrate multiple drawing strategies in your application?
??x
You can integrate multiple drawing strategies by creating unique pointers for each strategy and associating them with their respective shape objects. This allows you to change the drawing behavior of any shape at runtime without modifying its core functionality.
```cpp
#include <Circle.h>
#include <Square.h>
#include <OpenGLCircleStrategy.h>
#include <OpenGLSquareStrategy.h>
#include <memory>
#include <vector>

int main() {
    using Shapes = std::vector<std::unique_ptr<Shape>>;

    Shapes shapes{};

    // Creating some shapes, each one equipped with the corresponding OpenGL drawing strategy
    shapes.emplace_back(
        std::make_unique<Circle>(2.3, std::make_unique<OpenGLCircleStrategy>(/*...red...*/))
    );
    shapes.emplace_back(
        std::make_unique<Square>(1.2, std::make_unique<OpenGLSquareStrategy>(/*...green...*/))
    );
    shapes.emplace_back(
        std::make_unique<Circle>(4.1, std::make_unique<OpenGLCircleStrategy>(/*...blue...*/))
    );

    // Drawing all shapes
    for (auto const& shape : shapes) {
        shape->draw(/*some arguments*/);
    }

    return EXIT_SUCCESS;
}
```
x??

---

#### Strategy Design Pattern: Comparison with Visitor Pattern
Background context explaining the concept:
- The text compares and contrasts the Strategy pattern with the Visitor pattern.

:p What is the main difference between the Strategy and Visitor design patterns?
??x
The main difference between the Strategy and Visitor design patterns lies in their focus:

- **Strategy**: Allows adding new operations easily but makes it difficult to add new shape types.
- **Visitor**: Enables easy addition of new shapes while making it challenging to add new operations.

In the context provided, the Strategy pattern is used for drawing operations, while the Visitor pattern would be more suitable if you needed to perform different operations on multiple objects without changing their structure.
x??

---

#### Strategy Design Pattern: Strengths and Weaknesses
Background context explaining the concept:
- The text discusses both the advantages and disadvantages of using the Strategy design pattern.

:p What are some potential downsides to using the Strategy design pattern?
??x
Some potential downsides include:

1. **Operation Implementation Still Part of Concrete Type**: Operations remain part of specific types, which limits adding new operations.
2. **Large Refactoring Required if Not Identified Early**: Identifying variation points early can prevent large refactors later on, but implementing everything with Strategy upfront could lead to overengineering.
3. **Performance Impact Using Base Classes**: Performance might be affected due to additional runtime indirection and manual memory management.
4. **Single Strategy for Single Operation**: A single strategy should handle one operation or a small group of cohesive functions; otherwise, it violates the SRP (Single Responsibility Principle).

These issues highlight the importance of carefully considering when and how to apply the Strategy pattern based on your specific use case.
x??

---

#### Strategy Design Pattern: Multiple Strategies
Background context explaining the concept:
- The text discusses scenarios where multiple strategies are needed for different operations.

:p How can you implement multiple strategies for different operations in the same class?
??x
You can implement multiple strategies by creating separate strategy classes for each operation and then using dependency injection or template-based approaches to switch between them. For example, adding a `serialize` function:

```cpp
class Circle : public Shape {
public:
    explicit Circle(
        double radius,
        std::unique_ptr<DrawCircleStrategy> drawer,
        std::unique_ptr<SerializeCircleStrategy> serializer
        // potentially more strategy-related arguments
    )
        : radius_(radius),
          drawer_(std::move(drawer)),
          serializer_(std::move(serializer))
    {
        // Checking that the given radius is valid and that the given std::unique_ptrs are not nullptrs
    }

    void draw(/*some arguments*/) const override;
};
```

Here, `Circle` can use different strategies for drawing and serializing.
x??

---

#### Strategy Design Pattern: Performance Considerations
Background context explaining the concept:
- The text mentions performance impacts of using Strategy with base classes.

:p What are potential performance issues when implementing Strategy patterns with base classes?
??x
Potential performance issues include:

1. **Runtime Indirection**: Additional virtual function calls introduce runtime overhead.
2. **Memory Fragmentation and Manual Allocations**: Frequent allocations can lead to memory fragmentation, especially if `std::make_unique()` is used extensively.
3. **Indirection Due to Pointers**: Multiple pointers increase the number of indirections in your code.

These issues can be mitigated by using templates instead of base classes, which eliminate some of these overheads but require careful consideration based on specific use cases.
x??

---

#### Strategy Design Pattern Overview
Explanation: The Strategy design pattern is a behavioral design pattern that allows an algorithm’s behavior to be selected at runtime. It decouples an algorithm from the client that uses it, allowing different algorithms to be used interchangeably.

:p What is the primary purpose of the Strategy design pattern?
??x
The Strategy design pattern enables you to define a family of algorithms, encapsulate each one, and make them interchangeable. This allows clients to use algorithms without being dependent on their implementation details.
x??

---

#### Multiple Strategy Instances per Class
Explanation: The example provided shows how using multiple strategy instances in a class can lead to larger instance sizes due to the overhead of pointers.

:p How does using multiple `std::unique_ptr` strategies affect the Circle class's performance and size?
??x
Using multiple `std::unique_ptr` strategies increases the size of the Circle class due to additional pointer indirection. It also incurs a runtime overhead since calls through these pointers are slower than direct function calls.

For example, if you have two different drawing strategies for the Circle:
```cpp
class DrawCircleStrategy1 {};
class DrawCircleStrategy2 {};

class Circle {
public:
    explicit Circle(double radius, std::unique_ptr<DrawCircleStrategy> drawer) : radius_(radius), drawer_(std::move(drawer)) {}

private:
    double radius_;
    std::unique_ptr<DrawCircleStrategy> drawer_;
};
```
The overhead comes from the extra memory and runtime cost due to pointer indirection.

To improve performance, you can use template parameters as shown in the example:
```cpp
template <typename DrawCircleStrategy>
class Circle : public Shape {
public:
    explicit Circle(double radius, DrawCircleStrategy drawer) : radius_(radius), drawer_(drawer) {}

private:
    double radius_;
    DrawCircleStrategy drawer_; // Could possibly be omitted if stateless.
};
```
x??

---

#### Policy-Based Design
Explanation: The policy-based design is a form of static polymorphism using templates. It allows you to inject behavior into class templates, making the code more flexible and easier to extend.

:p How does policy-based design differ from runtime Strategy pattern?
??x
Policy-based design uses templates for injecting behavior at compile time, whereas the runtime Strategy pattern uses pointers or references to dynamically change behaviors during execution.

For example:
```cpp
namespace std {
template <typename T, typename Deleter = std::default_delete<T>>
class unique_ptr;
}

// Using a template parameter instead of a pointer
template <typename DrawCircleStrategy>
class Circle : public Shape {
public:
    explicit Circle(double radius, DrawCircleStrategy drawer) : radius_(radius), drawer_(drawer) {}

private:
    double radius_;
    DrawCircleStrategy drawer_; // Could possibly be omitted if stateless.
};
```
x??

---

#### Standard Library Algorithms and Strategy Pattern
Explanation: The `std::partition` and `std::sort` algorithms in the C++ standard library use the Strategy pattern to allow external behavior injection.

:p How do `std::partition` and `std::sort` utilize the Strategy design pattern?
??x
Both `std::partition` and `std::sort` take a predicate function or comparator as an argument, which allows injecting specific behaviors at runtime. For example:
```cpp
namespace std {
template <typename ForwardIt, typename UnaryPredicate>
constexpr ForwardIt partition(ForwardIt first, ForwardIt last, UnaryPredicate p);

template <typename RandomIt, typename Compare>
constexpr void sort(RandomIt first, RandomIt last, Compare comp);
}
```
The `UnaryPredicate` for `std::partition` and the `Compare` for `std::sort` allow you to specify how elements should be ordered or partitioned without modifying the algorithms themselves.

This separation of concerns makes these functions more versatile and allows them to work with different data types and orderings.
x??

---

#### Flexibility in Cleanup
Explanation: The `std::unique_ptr` class template uses a template parameter for its deleter, allowing flexible resource management behavior.

:p How does the `std::unique_ptr` example demonstrate policy-based design?
??x
The `std::unique_ptr` demonstrates policy-based design by using a template parameter to inject cleanup behavior. You can specify how memory should be deallocated when the pointer is destroyed.

Example:
```cpp
namespace std {
template <typename T, typename Deleter = std::default_delete<T>>
class unique_ptr;

// Specialization for array types
template <typename T, typename Deleter>
class unique_ptr<T[], Deleter>;
}
```
Here, you can specify a custom deleter to perform cleanup in different ways:
```cpp
std::unique_ptr<int[]> ptr(new int[10], [](int* p) {
    delete[] p; // Custom array deleter
});
```
This flexibility shows how policy-based design can be applied to manage resources differently.
x??

---

#### Limitations of Strategy Pattern
Explanation: The example discusses the limitations and trade-offs of using multiple strategy instances per class.

:p What are some potential disadvantages of using multiple `std::unique_ptr` strategies in a class?
??x
Using multiple `std::unique_ptr` strategies can lead to larger object sizes due to pointer indirection, which can affect performance. Additionally, you lose the flexibility to change strategies at runtime and might end up with many different classes instead of a single one.

For example:
```cpp
class DrawCircleStrategy1 {};
class DrawCircleStrategy2 {};

class Circle {
public:
    explicit Circle(double radius, std::unique_ptr<DrawCircleStrategy> drawer) : radius_(radius), drawer_(std::move(drawer)) {}

private:
    double radius_;
    std::unique_ptr<DrawCircleStrategy> drawer_;
};
```
This approach is less flexible and might result in more complex code if you need to change the strategy at runtime.

To address these issues, policy-based design using template parameters can provide a better balance between flexibility and performance.
x??

---

