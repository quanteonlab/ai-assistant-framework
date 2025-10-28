# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 15)

**Starting Chapter:** Guideline 19 Use Strategy to Isolate How Things Are Done

---

#### Strategy Design Pattern Overview
The Strategy design pattern is a behavioral design pattern that enables selecting and using a family of algorithms dynamically. The pattern allows you to encapsulate each algorithm behind an interface or abstract class, making it easy to change the behavior at runtime without modifying client code.

:p What is the purpose of the Strategy design pattern?
??x
The Strategy design pattern provides a flexible way to define a set of behaviors that can be used interchangeably. It helps in isolating how things are done by encapsulating algorithms behind an interface, making it easier to switch between different strategies at runtime without altering the client code.

For example, consider implementing drawing behavior for shapes where you might want to change the rendering engine (e.g., from OpenGL to DirectX) without changing existing code.
??x
---

#### Inheritance vs. Composition in C++
Inheritance is a powerful mechanism that allows classes to inherit properties and behaviors from parent classes, but it can lead to tight coupling between classes and may not always be flexible.

Composition, on the other hand, involves creating objects out of smaller parts. This approach promotes loose coupling and reusability.

:p Why might you prefer composition over inheritance?
??x
You might prefer composition over inheritance when you want more flexibility in changing how objects are constructed or when you need to avoid tight coupling between classes. Composition allows for easier maintenance and extension of the codebase since components can be replaced or extended without affecting other parts of the system.

For example, instead of inheriting from a base class that has many responsibilities, you might prefer to have a composition relationship where each object holds references to objects that perform specific tasks.
??x
---

#### Command Design Pattern Overview
The Command design pattern is another behavioral pattern used to encapsulate a request as an object. This allows you to parameterize methods with different requests, queue or log requests, and support undoable operations.

:p What does the Command design pattern enable?
??x
The Command design pattern enables you to parameterize methods with different requests by encapsulating them into objects (commands). It supports various actions like queuing commands for later execution, logging, undo/redo capabilities, and more.

For example, you can create a `DrawCommand` that contains the instructions needed to draw a shape. This command can be executed immediately or stored in a history stack for future use.
??x
---

#### Guideline 21: Use Command to Isolate What Things Are Done
The Command pattern helps in isolating what things are done by encapsulating actions into objects. Each command object represents an operation that can be performed and can be parameterized with data.

:p How does the Command design pattern help isolate actions?
??x
By using the Command pattern, you can encapsulate operations (actions) as separate objects, which makes it easier to manage and manipulate these actions independently of other parts of your application. This allows for better decoupling between the sender of a command and the receiver.

For example, in our 2D graphics tool, each shape's drawing operation could be represented by a `DrawCommand` object that encapsulates the necessary parameters and logic.
??x
---

#### Value Semantics vs. Reference Semantics
Value semantics refers to objects whose values are immutable or can be copied without affecting the original value. Reference semantics involves objects where changing one reference also changes the original.

:p Why might you prefer value semantics over reference semantics?
??x
You might prefer value semantics when you want to ensure that an object's state cannot be accidentally modified by other parts of your program, which can lead to more predictable and safer code. Value semantics are particularly useful in situations where immutability is desired or necessary.

For example, in C++, using `std::string` instead of `char*` often leads to better safety because string operations do not modify the original object but return new objects with updated values.
??x
---

#### Applying Value Semantics to Strategy and Command Patterns
By applying value semantics, you can implement the Strategy and Command patterns more effectively. For instance, using `std::function` in C++ allows for flexible and safe handling of functions as first-class citizens.

:p How can you apply value semantics to the Strategy pattern?
??x
You can apply value semantics to the Strategy pattern by implementing strategies as `std::function<void()>`, which are essentially function objects that encapsulate a piece of behavior. This approach provides flexibility and safety, as changes to one strategy do not affect others.

For example:
```cpp
#include <functional>
#include <iostream>

class Context {
public:
    void executeStrategy(std::function<void()> strategy) {
        // Execute the strategy passed in
        strategy();
    }
};

// A simple strategy
void drawCircle() { std::cout << "Drawing a circle." << std::endl; }

int main() {
    Context context;
    context.executeStrategy(drawCircle);
    return 0;
}
```
??x
---

#### Single-Responsibility Principle (SRP) Violation
Background context: The current implementation of drawing shapes violates the SRP because it tightly couples shape classes with specific drawing implementations, making future changes difficult and intrusive. This is problematic if the tool needs to support multiple graphic libraries in the future.
:p What problem does the SRP violation cause in the current implementation?
??x
The SRP violation causes two main issues: 1) It's not easy to change how shapes are drawn because drawing logic is embedded directly into `Circle` and `Square`. Any changes require modifying these classes, which can be intrusive. 2) If you want to support multiple graphic libraries (e.g., OpenGL, Metal, Vulkan), you would need to modify the behavior in multiple unrelated places.
??x
---

#### Inheritance Hierarchy for Drawing with OpenGL
Background context: To address the SRP violation and make drawing flexible across different libraries, a new approach was proposed where `OpenGLCircle` and `OpenGLSquare` inherit from their respective base classes (`Circle` and `Square`). This allows implementing drawing logic in specific derived classes without affecting the base class.
:p How does the proposed solution with inheritance hierarchy address the SRP violation?
??x
The proposed solution addresses the SRP violation by moving the drawing implementation to derived classes, making it easier to support different graphic libraries. For example:
```cpp
//---- <Circle.h> ----------------
#include <Shape.h>
class Circle : public Shape {
public:
    // ... No implementation of the draw() member function anymore
};

//---- <OpenGLCircle.h> ----------------
#include <Circle.h>
class OpenGLCircle : public Circle {
public:
    explicit OpenGLCircle(double radius) : Circle(radius) {}
    void draw(/*some arguments*/) const override;
};
```
This way, `Circle` and `Square` can focus on being simple geometric primitives without knowing how they are drawn. The drawing logic is encapsulated in derived classes like `OpenGLCircle`.
??x
---

#### Drawbacks of Extending Inheritance Hierarchy for Different Libraries
Background context: While the new solution with inheritance hierarchy works for drawing, it introduces challenges when adding new functionality such as serialization. Adding a `serialize()` function similarly needs to be implemented differently for each library, leading to complex and artificial class names like `OpenGLProtobufCircle` or `MetalBoostSerialSquare`.
:p What are the main drawbacks of extending the inheritance hierarchy with different libraries?
??x
The main drawbacks include:
1. **Complex Class Names**: New classes with names like `OpenGLProtobufCircle`, `MetalBoostSerialSquare`, etc., become cumbersome and less readable.
2. **Complex Hierarchy**: Adding more functionality requires deepening the hierarchy, leading to a complex structure that can be hard to maintain.
3. **Reuse Issues**: Reusing implementation details across different derived classes becomes difficult because each class needs to implement similar logic independently.

For example, it's challenging to reuse OpenGL code between `OpenGLProtobufCircle` and `OpenGLBoostSerialCircle`.
??x
---

#### Alternative Design Solution: Strategy Pattern
Background context: The proposed inheritance hierarchy approach is not scalable for multiple functionalities. A better solution might involve using the Strategy pattern where a strategy object handles the drawing or serialization logic, making it easier to switch between different implementations without changing class hierarchies.
:p How can the Strategy pattern be applied to address the issues with the current design?
??x
The Strategy pattern can address these issues by defining an interface for operations and allowing subclasses to provide specific implementations. Here’s a simplified example:
```cpp
//---- <DrawingStrategy.h> ----------------
class DrawingStrategy {
public:
    virtual ~DrawingStrategy() = default;
    virtual void draw(const Shape& shape) const = 0;
};

//---- <OpenGLDrawingStrategy.h> ----------------
#include "DrawingStrategy.h"
class OpenGLDrawingStrategy : public DrawingStrategy {
public:
    void draw(const Circle& circle) const override {
        // OpenGL code to draw a circle
    }
    void draw(const Square& square) const override {
        // OpenGL code to draw a square
    }
};

//---- <Shape.h> ----------------
#include "DrawingStrategy.h"
class Shape {
private:
    DrawingStrategy* strategy_;
public:
    void setDrawingStrategy(DrawingStrategy* strategy) { strategy_ = strategy; }
    void draw() const {
        if (strategy_) {
            strategy_->draw(*this);
        }
    }
};
```
With this pattern, you can easily switch between different drawing strategies without changing the `Shape` class or creating numerous subclasses.
??x
---

#### Strategy Design Pattern Intent
Background context explaining the intent of the Strategy design pattern. The pattern aims to define a family of algorithms, encapsulate each one, and make them interchangeable. This allows for changing how different shapes are drawn without modifying existing code.

:p What is the main goal of the Strategy design pattern?
??x
The main goal of the Strategy design pattern is to enable the algorithm to vary independently from its clients by encapsulating each algorithm as a separate class that can be swapped out at runtime. This promotes flexibility and adheres to the Single Responsibility Principle (SRP) and Open-Closed Principle (OCP).

```cpp
// Example pseudo-code for implementing part of the Strategy pattern in C++
class DrawStrategy {
public:
    virtual ~DrawStrategy() = default;
    virtual void draw(const Circle& circle, /*some arguments*/) const = 0;
    virtual void draw(const Square& square, /*some arguments*/) const = 0;
};
```
x??

---

#### UML Representation of Strategy Design Pattern
Explanation about the UML diagram for the Strategy design pattern. It shows a base class `DrawStrategy` with pure virtual functions for drawing different shapes.

:p What does the UML diagram typically show for the Strategy design pattern?
??x
The UML diagram for the Strategy design pattern typically illustrates a base class named `DrawStrategy` that defines pure virtual functions for drawing specific types of shapes, such as circles and squares. This separation allows for interchangeable strategies without changing the shape classes.

```uml
Class "DrawStrategy" {
    + draw(circle: Circle, /*some arguments*/) const = 0;
    + draw(square: Square, /*some arguments*/) const = 0;
}

Class "Circle"
Class "Square"

Interface "DrawStrategy" <<base>>
```
x??

---

#### DrawStrategy Base Class Implementation
Explanation about the implementation of the `DrawStrategy` base class. It includes a virtual destructor and pure virtual functions for drawing shapes.

:p What is included in the `DrawStrategy` base class?
??x
The `DrawStrategy` base class includes a virtual destructor to ensure proper cleanup when derived classes are destroyed, and two pure virtual functions: `draw(const Circle&, /*some arguments*/) const` and `draw(const Square&, /*some arguments*/) const`. These functions are meant to be implemented by derived classes to provide specific drawing logic.

```cpp
// Example implementation of the DrawStrategy base class in C++
class DrawStrategy {
public:
    virtual ~DrawStrategy() = default; // Virtual destructor for proper cleanup

    virtual void draw(const Circle& circle, /*some arguments*/) const = 0;
    virtual void draw(const Square& square, /*some arguments*/) const = 0;
};
```
x??

---

#### Shape Class and DrawStrategy Usage
Explanation about how the `Shape` class and `DrawStrategy` are used in practice. The `Shape` class remains unchanged but uses a pointer to `DrawStrategy` for drawing.

:p How does the `Shape` class utilize the `DrawStrategy`?
??x
The `Shape` class utilizes the `DrawStrategy` by storing a `std::unique_ptr<DrawStrategy>` member variable that points to an instance of a derived `DrawStrategy` class. This allows each shape (e.g., Circle, Square) to use its own drawing strategy without altering the `Shape` base class.

```cpp
// Example usage in the Shape and Circle classes in C++
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw(/*some arguments*/) const = 0; // Pure virtual function
};

class Circle : public Shape {
public:
    explicit Circle(double radius, std::unique_ptr<DrawStrategy> drawer)
        : radius_(radius), drawer_(std::move(drawer)) {}

    void draw(const std::string& /*some arguments*/) const override {
        drawer_->draw(*this, /*some arguments*/);
    }

private:
    double radius_;
    std::unique_ptr<DrawStrategy> drawer_;
};
```
x??

---

#### Benefits of the Strategy Design Pattern
Explanation about the benefits of using the Strategy design pattern. It allows for changing drawing algorithms without modifying existing classes and enables new drawing strategies to be easily added.

:p What are the main advantages of the Strategy design pattern?
??x
The main advantages of the Strategy design pattern include:
1. **Encapsulation**: The drawing logic is encapsulated within separate `DrawStrategy` classes.
2. **Flexibility**: Drawing algorithms can be changed independently without modifying existing code.
3. **Open-Closed Principle (OCP)**: New strategies can be introduced without altering existing code.
4. **SRP Compliance**: Each class has a single responsibility, making the system more maintainable.

```cpp
// Example of adding a new drawing strategy in C++
class CustomDrawStrategy : public DrawStrategy {
public:
    void draw(const Circle& circle, /*some arguments*/) const override {
        // Custom drawing logic for circles
    }

    void draw(const Square& square, /*some arguments*/) const override {
        // Custom drawing logic for squares
    }
};
```
x??

---

