# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 14)

**Starting Chapter:** Performance Benchmarks

---

#### Concept: std::variant for Shape Variants
Background context explaining how `std::variant` can be used to represent different types of shapes without the need for a common base class or visitor pattern. This allows for more flexible and non-intrusive code.

:p How does `std::variant` allow for representing multiple shape types?
??x
`std::variant` is a C++17 feature that allows storing one of several possible data types in a single variable. For shapes, we can use `std::variant` to store different round or angular shapes without needing a common base class or complex visitor patterns.

```cpp
#include <variant>
#include <iostream>

// Round Shape variant
using RoundShape = std::variant<Circle, Ellipse>;

// Angular Shape variant
using AngularShape = std::variant<Square, Rectangle>;
```

x??

---

#### Concept: Performance of std::variant vs Classic Visitor Pattern
Background context explaining the performance comparison between `std::variant` and classic visitor pattern. The benchmarks show that `std::variant` is more efficient due to its simpler memory layout and fewer indirections.

:p How does the benchmark compare the performance of `std::variant` against the classic visitor design pattern?
??x
The benchmark results showed that using `std::variant` with both `std::visit()` and `std::get_if()` was faster than implementing a classic visitor design pattern. The `std::variant` solutions had lower overhead due to better memory layout and fewer indirections.

Table 4-2: Benchmark results for different Visitor implementations
| Visitor implementation     | GCC 11.1 | Clang 11.1 |
|---------------------------|---------|----------|
| Classic Visitor design pattern | 1.6161 s | 1.8015 s |
| Object-oriented solution   | 1.5205 s | 1.1480 s |
| Enum solution              | 1.2179 s | 1.1200 s |
| std::variant (with std::visit()) | 1.1992 s | 1.2279 s |
| std::variant (with std::get_if()) | 1.0252 s | 0.6998 s |

x??

---

#### Concept: Memory Layout and Indirections
Background context explaining the memory layout benefits of `std::variant` compared to other solutions like enums or classic visitor patterns.

:p How does the memory layout of shapes using `std::variant` compare with other solutions?
??x
Using `std::variant`, all shape objects are stored contiguously in memory, which is cache-friendly. This contrasts with other approaches such as enums or classic visitor patterns, where additional indirections can occur.

```cpp
// Using std::variant
std::vector<RoundShape> roundShapes;
roundShapes.push_back(std::make_shared<Circle>());
roundShapes.push_back(std::make_shared<Ellipse>());

for (auto& shape : roundShapes) {
    if (std::holds_alternative<Circle>(shape)) {
        // Access Circle's members or methods
    }
}
```

x??

---

#### Concept: Translate Operation vs Draw Operation
Background context explaining why a translate operation was chosen over an expensive draw operation for benchmarking.

:p Why did the author choose to use a translate operation instead of a draw operation in the benchmarks?
??x
The author chose to update the center point by random vectors (translate operation) rather than using an expensive `draw()` operation. This is because the cheaper translate operation better highlights the intrinsic overhead of different solutions, such as indirections and virtual function call overhead.

```cpp
// Pseudocode for translating shapes
for (int i = 0; i < 25000; ++i) {
    for (auto& shape : allShapes) {
        int dx = rand() % 10 - 5;
        int dy = rand() % 10 - 5;
        translate(shape, dx, dy);
    }
}
```

x??

---

#### Concept: Quality of Performance Benchmarks
Background context explaining the subjective nature of performance benchmarks and their limitations.

:p Why are the benchmark results described as qualitative values rather than quantitative truths?
??x
The benchmark results should be considered qualitative values that point in the right direction, not absolute truth. They are based on specific conditions (e.g., 8-core Intel Core i7 with macOS Big Sur) and may vary depending on the environment, compiler flags, and hardware.

```cpp
// Pseudocode for running benchmarks
void runBenchmark() {
    auto start = std::chrono::high_resolution_clock::now();
    // Perform operations
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    return duration;
}
```

x??

---

#### Performance Considerations of std::variant

Background context explaining that performance is a critical aspect when choosing between different implementation strategies. It highlights how `std::get_if()` and `std::visit()` can have varying levels of efficiency, with some compilers producing slower code for certain scenarios.

:p How does the choice between using `std::get_if()` and `std::visit()` affect performance?
??x
The choice affects performance significantly. For example, both GCC and Clang produce much slower code when using `std::visit()`, which might be due to the fact that `std::visit()` is not yet fully optimized.

```cpp
// Example usage of std::get_if()
if (auto p = std::get_if<int>(variant); p) {
    // Use *p here
}

// Example usage of std::visit()
std::visit([](auto&& arg) { /* handle the argument */ }, variant);
```
x??

---

#### Advantages and Disadvantages of std::variant

Background context on how `std::variant` is similar to the Visitor design pattern but operates within procedural programming, emphasizing the importance of dealing with a closed set of types.

:p What are some advantages of using `std::variant`?
??x
Advantages include:
- Simplifying code significantly.
- Reducing boilerplate code.
- Encapsulating complex operations and making maintenance easier.

```cpp
// Example usage of std::variant for simplification
std::variant<int, double, std::string> value = 42;
if (auto p = std::get_if<int>(&value); p) {
    // Use *p here
}
```
x??

---

#### Adding New Types to `std::variant`

Background context that while `std::variant` provides a flexible solution for operations, adding new types can be problematic due to the closed set nature of variants.

:p What are the challenges when adding new types to `std::variant`?
??x
Challenges include:
- Needing to update the variant itself, which might trigger recompilation.
- Having to add or modify the operator() for each alternative type.
- Compilers will complain if necessary operators are missing but won't provide clear error messages.

```cpp
// Example of adding a new type
std::variant<int, double, std::string, MyNewType> value = 42;
if (auto p = std::get_if<MyNewType>(&value); p) {
    // Use *p here
}
```
x??

---

#### Storing Different Sized Types in `std::variant`

Background context that storing types of different sizes can lead to performance issues due to space inefficiency.

:p What are the implications of storing differently sized types within a single `std::variant`?
??x
Implications include:
- Potential waste of memory if one type is much larger than others.
- A possible solution is to use pointers or proxy objects, but this introduces indirection and its associated performance costs.

```cpp
// Example usage with different sizes
std::variant<int, std::string> value = "Hello";
if (auto p = std::get_if<std::string>(&value); p) {
    // Use *p here
}
```
x??

---

#### Revealing Implementation Details Through `std::variant`

Background context that while `std::variant` is a runtime abstraction, the types stored within it are still visible and can lead to physical dependencies.

:p How might revealing implementation details through `std::variant` impact code maintenance?
??x
Impact includes:
- Physical dependencies on variant: modifying one of the alternative types requires recompiling any dependent code.
- Possible solution is using pointers or proxy objects, but this impacts performance due to increased indirection.

```cpp
// Example usage showing visibility issues
std::variant<int, std::string> value = "Hello";
if (auto p = std::get_if<std::string>(&value); p) {
    // Use *p here
}
```
x??

---

#### std::variant for Visitor Pattern

This guideline discusses using `std::variant` to implement a Visitor pattern, emphasizing its non-intrusive nature and advantages over traditional object-oriented Visitor solutions.

:p How does `std::variant` help in implementing the Visitor design pattern?

??x
`std::variant` allows creating abstractions on the fly without modifying existing classes. It's used to represent values that can be one of a set of types, which is beneficial for the Visitor design pattern where we want to perform operations on different types dynamically.

Here’s how you might use `std::variant` in C++:
```cpp
#include <variant>

// Example shape and visitor structures using std::variant

struct Circle {};
struct Square {};

using Shape = std::variant<Circle, Square>;

class Visitor {
public:
    virtual void visit(Circle const&) {}
    virtual void visit(Square const&) {}

    // You can use `std::visit` to apply the appropriate function based on the shape type.
};

// Usage example
void processShape(Shape shape, Visitor& visitor) {
    std::visit(visitor, shape);
}
```

x??

---

#### Acyclic Visitor Pattern

The Acyclic Visitor pattern addresses the cyclic dependency issue in the traditional Visitor design pattern by breaking it. It uses abstract base classes for visitors and specific operation classes to avoid circular dependencies.

:p What is the key difference between the traditional GoF Visitor and the Acyclic Visitor?

??x
In the Acyclic Visitor, the `Visitor` class is split into an `AbstractVisitor` and visitor-specific classes (e.g., `CircleVisitor`, `SquareVisitor`). These specific visitors inherit from both the abstract base visitor class and a shape-specific visitor class. This structure breaks the cyclic dependency problem.

Here’s how it might be structured:
```cpp
//---- <AbstractVisitor.h> ----------------
class AbstractVisitor {
public:
    virtual ~AbstractVisitor() = default;
};

//---- <CircleVisitor.h> ----------------
#include "AbstractVisitor.h"

class CircleVisitor : public AbstractVisitor {
public:
    void visit(Circle const&) {}
};

//---- <SquareVisitor.h> ----------------
#include "AbstractVisitor.h"

class SquareVisitor : public AbstractVisitor {
public:
    void visit(Square const&) {}
};
```

x??

---

#### Performance Considerations in Acyclic Visitor

The Acyclic Visitor pattern helps with cyclic dependencies but might have performance implications due to the overhead of multiple base classes and virtual functions.

:p What are some key differences between the traditional Visitor and the Acyclic Visitor design patterns?

??x
Key differences include:
- **Dependencies**: Traditional Visitor has a cyclic dependency (Visitor depends on Shapes, Shapes depend on Visitor). The Acyclic Visitor breaks this cycle by using abstract visitor classes.
- **Flexibility**: In the Acyclic Visitor, visitors can opt-in or out of specific operations based on their inheritance from shape-specific visitor classes.

Example of an operation that supports circles but not squares:
```cpp
class Operation {
public:
    void execute(CircleVisitor& circleVisitor) {
        std::visit(circleVisitor, shape); // Only CircleVisitor will be called.
    }

    void execute(SquareVisitor& squareVisitor) {} // Optional: Not implemented here.
};
```

x??

---

#### Acyclic Visitor Design Pattern Overview
Background context: In C++, implementing shape-specific visitor base classes as class templates allows for a cleaner separation of concerns. This approach is part of the Visitor design pattern, which enables adding new operations to existing data structures without modifying their classes.

:p What does the acyclic visitor design pattern address in terms of dependencies?
??x
The acyclic visitor design pattern addresses the cyclic dependency issue that typically arises when implementing the Visitor design pattern directly. Normally, a shape class would need to know about its specific visitor implementations, leading to tight coupling and circular dependencies.
x??

---

#### Implementation of Draw Visitor Class
Background context: The `Draw` visitor is designed to handle both circle and square shapes by inheriting from multiple base classes. This allows the high-level architecture to remain decoupled from the concrete shape types.

:p How does the `Draw` class implement support for Circle and Square shapes?
??x
The `Draw` class implements support for Circle and Square shapes by inheriting from three base classes: `AbstractVisitor`, `Visitor<Circle>`, and `Visitor<Square>`. This structure ensures that the high-level architecture is not dependent on the concrete shape types.

```cpp
class Draw : public AbstractVisitor, public Visitor<Circle>, public Visitor<Square>
{
public:
    void visit(Circle const& c) const override {
        // Implementing the logic for drawing a circle
    }
    void visit(Square const& s) const override {
        // Implementing the logic for drawing a square
    }
};
```
x??

---

#### Circle Class with Accept Function
Background context: The `Circle` class has been modified to include an `accept` function that uses dynamic_cast to determine if the visitor supports circles. This change breaks the cyclic dependency and introduces a level of indirection.

:p How does the `Circle::accept` function work?
??x
The `Circle::accept` function works by performing a dynamic_cast from the given visitor (`AbstractVisitor const& v`) to `Visitor<Circle>`. If successful, it calls the corresponding visit method on that type. This approach breaks the cyclic dependency and allows for better separation of concerns.

```cpp
class Circle : public Shape {
public:
    explicit Circle(double radius)
        : radius_(radius) {
        // Checking that the given radius is valid
    }

    void accept(AbstractVisitor const& v) override {
        if (auto const* cv = dynamic_cast<Visitor<Circle> const*>(&v)) {
            cv->visit(*this);
        }
    }

    double radius() const { return radius_; }
    Point center() const { return center_; }

private:
    double radius_;
    Point center_{};
};
```
x??

---

#### Concerns with Dynamic Cast
Background context: While the use of dynamic_cast in `Circle::accept` is necessary to break the cyclic dependency, it introduces potential issues related to architecture integrity.

:p What are the downsides of using dynamic_cast in the `Circle::accept` function?
??x
The primary downside of using dynamic_cast in the `Circle::accept` function is that it can lead to runtime errors if not used carefully. Specifically, a bad use of dynamic_cast could break an architectural boundary by allowing code at a high level to interact with elements from a low-level implementation.

```cpp
void accept(AbstractVisitor const& v) override {
    if (auto const* cv = dynamic_cast<Visitor<Circle> const*>(&v)) {
        // Safe cast and proceed with the visit method
    } else {
        // Handle the case where the visitor does not support circles
    }
}
```
x??

---

#### Acyclic Visitor vs Cyclic Visitor Performance

Background context explaining the concept. The text discusses the performance implications of using an Acyclic Visitor compared to a Cyclic Visitor. It mentions that while an Acyclic Visitor has architectural advantages, its runtime can be significantly slower due to cross-casts and virtual function calls.

:p What is the primary reason for the poor performance of the Acyclic Visitor in this context?
??x
The primary reason for the poor performance of the Acyclic Visitor is the use of dynamic_casts that involve cross-casting between different branches of an inheritance hierarchy, followed by a virtual function call. These operations are significantly more costly than simple downcasts.
```cpp
// Example of a problematic cross-cast in C++
class Base {};
class Derived1 : public Base {};
class Derived2 : public Base {};

Derived1* derived1 = dynamic_cast<Derived1*>(dynamic_cast<Base*>(new Derived2()));
```
x??

---

#### Performance Disadvantage of Acyclic Visitor

Background context explaining the concept. The text emphasizes that while an Acyclic Visitor offers architectural benefits, its performance is notably worse than other visitor implementations like Cyclic Visitors and std::variant solutions.

:p Why does the Acyclic Visitor have a runtime penalty compared to other visitor implementations?
??x
The Acyclic Visitor has a runtime penalty because it involves cross-casts followed by virtual function calls. These operations are significantly more costly than simple downcasts or direct type conversions, leading to a much higher runtime cost.

Table 4-3 provides performance results comparing different visitor implementations:
```markdown
| Visitor implementation | GCC        | Clang       |
|------------------------|------------|-------------|
| Acyclic Visitor        | 14.3423 s  | 7.3445 s    |
| Cyclic Visitor         | 1.6161 s   | 1.8015 s    |
| Object-oriented solution| 1.5205 s  | 1.1480 s    |
| Enum solution          | 1.2179 s  | 1.1200 s    |
| std::variant (with std::visit()) | 1.1992 s | 1.2279 s   |
| std::variant (with std::get())      | 1.0252 s | 0.6998 s   |
```

The Acyclic Visitor's runtime is almost one order of magnitude higher than the Cyclic Visitor and other solutions.
x??

---

#### Architectural Advantages of Acyclic Visitor

Background context explaining the concept. The text discusses that an Acyclic Visitor has significant architectural advantages, such as avoiding virtual functions at a certain level in the architecture.

:p How does the use of an Acyclic Visitor affect the overall architecture?
??x
The use of an Acyclic Visitor allows for avoiding virtual functions at a certain low-level of the architecture. This avoids breaking the high-level design by inserting lower-level knowledge into it, which can be beneficial from an architectural standpoint.

However, this architectural benefit comes with a significant runtime penalty due to cross-casts and virtual function calls.
```cpp
// Example where Acyclic Visitor is used
class Base {};
class Derived1 : public Base {};
class Derived2 : public Base {};

void process(Base* obj) {
    if (Derived1* d1 = dynamic_cast<Derived1*>(obj)) {
        // Process Dervied1 specific logic
    } else if (Derived2* d2 = dynamic_cast<Derived2*>(obj)) {
        // Process Dervied2 specific logic
    }
}
```
x??

---

#### Performance Results Table

Background context explaining the concept. The text includes a performance results table comparing different visitor implementations.

:p What does Table 4-3 reveal about the runtime differences between Acyclic Visitor and Cyclic Visitor?
??x
Table 4-3 shows that the Acyclic Visitor has an almost one order of magnitude higher runtime compared to the Cyclic Visitor. For example, with GCC, the Acyclic Visitor takes around 14 seconds, while the Cyclic Visitor only takes about 1.6 seconds.

```markdown
| Implementation | GCC   | Clang    |
|---------------|-------|----------|
| Acyclic Visitor| 14.3423 s | 7.3445 s |
| Cyclic Visitor | 1.6161 s | 1.8015 s |
```
x??

---

#### Understanding the Performance Impact

Background context explaining the concept. The text emphasizes that while an Acyclic Visitor is architecturally interesting, its poor performance might make it less desirable in practical applications.

:p What does Guideline 18 advise regarding the use of Acyclic Visitors?
??x
Guideline 18 advises to be aware of the significant performance disadvantages of using an Acyclic Visitor. While the Acyclic Visitor has architectural advantages, the performance impact can be a strong argument for choosing alternative solutions like Cyclic Visitors or std::variant implementations.
```cpp
// Example implementation comparison
class Base {};
class Derived1 : public Base {};
class Derived2 : public Base {};

// Acyclic Visitor example (poor performance)
void process(Base* obj) {
    if (Derived1* d1 = dynamic_cast<Derived1*>(obj)) {
        // Process Dervied1 specific logic
    } else if (Derived2* d2 = dynamic_cast<Derived2*>(obj)) {
        // Process Dervied2 specific logic
    }
}

// Cyclic Visitor example (better performance)
void process(Base& obj) {
    if (dynamic_cast<Derived1*>(&obj)) {
        // Process Dervied1 specific logic
    } else if (dynamic_cast<Derived2*>(&obj)) {
        // Process Dervied2 specific logic
    }
}
```
x??

---

#### Team Dynamics and Office Politics

Background context: This excerpt discusses the potential impact of a proposed design decision on team dynamics. The author suggests that while an individual might not get angry about the change, it could lead to exclusion from social events like barbecues. 

:p How might a team member react to a design decision that they are neutral or ambivalent about?
??x
A team member who is neutral or ambivalent about a design decision might experience some unhappiness but likely would not get angry. However, there is a risk of being excluded from social events such as the next team barbecue.

??x
The potential impact on team cohesion and individual relationships due to minor changes in project dynamics.
```java
// Example code for a hypothetical scenario where a new design decision might affect team interactions
public class TeamDecision {
    public void updateBarbecueList(boolean includePerson) {
        // Logic to add or remove a person from the barbecue list based on their reaction to a decision
        if (!includePerson) {
            System.out.println("Sorry, you're not invited to this barbecue!");
        } else {
            System.out.println("You're all set for the team barbecue!");
        }
    }
}
```
x??

---

#### Design Patterns: Elements of Reusable Object-Oriented Software

Background context: This excerpt mentions a book by Erich Gamma et al. on design patterns, which is a key reference in software development. The book covers various patterns used to solve common problems in object-oriented programming.

:p What does the Design Patterns: Elements of Reusable Object-Oriented Software book cover?
??x
The Design Patterns: Elements of Reusable Object-Oriented Software by Erich Gamma et al. is a seminal work that provides solutions and patterns for solving common design issues in software development, specifically focusing on object-oriented approaches.

??x
It covers various design patterns such as the Visitor pattern, which is discussed further in this context.
```java
// Example of using the Visitor pattern (pseudocode)
public class Element {
    public void accept(Visitor v) { 
        v.visit(this); // Delegate to the visitor's visit method with the element object
    }
}

class ConcreteElement extends Element {
    // Concrete implementation details
}
```
x??

---

#### Naming Conventions in Design Patterns

Background context: This excerpt discusses naming conventions, specifically mentioning the `accept()` method from the Visitor pattern. It advises against renaming unless absolutely necessary to avoid confusion.

:p What is the significance of using a design pattern's name for methods?
??x
Using the exact names from established design patterns, such as `accept()`, helps maintain consistency and readability in code. This practice makes it easier for other developers familiar with these patterns to understand the intent behind your implementation without needing extensive documentation.

??x
Renaming can lead to confusion if the new name does not clearly convey the same purpose.
```java
// Example of using accept() method from Visitor pattern (pseudocode)
public class ConcreteElement {
    public void accept(Visitor v) {
        // Implementation of the visitor's operation on this element
        v.visit(this);
    }
}
```
x??

---

#### Design for Change

Background context: The excerpt emphasizes the importance of designing code to be easily changeable. This principle, known as "Don't Repeat Yourself" (DRY), suggests that repeated logic should be extracted into a single function or class.

:p Why is it important to design with change in mind?
??x
Designing for change ensures that when modifications are needed, the changes can be made more efficiently and reliably. By encapsulating common code in reusable functions or classes, you minimize the risk of introducing bugs through multiple updates across different parts of the codebase.

??x
It promotes maintainability by centralizing logic.
```java
// Example of DRY principle (pseudocode)
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    // Reusing add() instead of duplicating its implementation
    public int multiplyByThree(int num) {
        return this.add(this.add(num, num), num);
    }
}
```
x??

---

#### Random Number Generation and Performance

Background context: The excerpt mentions the use of random number generation in performance tests. It notes that while creating random numbers is not particularly expensive on certain machines, it can still impact performance under certain conditions.

:p Why might std::make_unique() be preferable to special-purpose allocation schemes?
??x
std::make_unique() encapsulates a call to `new`, making memory management safer and more consistent across different parts of the program. While it may introduce some overhead compared to specialized allocators, the added safety features often outweigh this cost.

??x
Memory fragmentation can be reduced with std::make_unique(), but special allocators might offer performance benefits in specific scenarios.
```cpp
// Example using std::make_unique (C++)
#include <memory>

std::unique_ptr<int> ptr = std::make_unique<int>(10);
```
x??

---

#### Open Source Implementations of Variant

Background context: The excerpt discusses alternative open-source implementations of the `variant` type, such as those provided by Boost and Abseil. These alternatives can offer additional insights into how to implement similar types.

:p What are some alternative implementations of variant?
??x
Some alternative implementations of variant include those provided by libraries like Boost (which offers two variants), Abseil, and the implementation by Michael Park. Exploring these can provide valuable insights into different design approaches.

??x
Understanding multiple implementations can help choose the best fit for a specific project.
```cpp
// Example using Boost's variant (C++)
#include <boost/variant.hpp>

using MyVariant = boost::variant<int, std::string>;
```
x??

---

#### Design Patterns: Bridge and Acyclic Visitor

Background context: This excerpt introduces two more design patterns—Bridge and Acyclic Visitor. It mentions that the author will cover the Bridge pattern in detail but not the Acyclic Visitor due to limited space.

:p What is another design pattern mentioned besides Proxy?
??x
Another design pattern mentioned, besides Proxy, is the Bridge pattern. The Acyclic Visitor pattern from Robert C. Martin’s book Agile Software Development: Principles, Patterns, and Practices is also referenced, though it will not be covered in this text due to space constraints.

??x
The Bridge pattern focuses on separating an object's interface from its implementation.
```java
// Example of the Bridge pattern (pseudocode)
public abstract class Implementor {
    public void operation() { }
}

public class ConcreteImplementorA extends Implementor {
    @Override
    public void operation() {
        // Concrete implementation A
    }
}
```
x??

---

#### Ownership and Abstractions

Background context: This excerpt discusses the importance of understanding ownership in abstractions, differentiating between high-level and low-level abstractions.

:p What does the term "high level" mean in relation to abstraction?
??x
In this context, "high level" refers to abstractions that are more abstract and less focused on specific details. They provide a broader view or higher-level perspective of the system or problem domain.

??x
High-level abstractions typically offer simpler interfaces but might hide implementation details.
```java
// Example of high-level abstraction (pseudocode)
public interface DataProcessor {
    void process();
}
```
x??

---

