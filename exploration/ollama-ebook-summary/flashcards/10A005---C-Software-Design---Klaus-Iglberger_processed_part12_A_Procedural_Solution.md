# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 12)

**Starting Chapter:** A Procedural Solution

---

---
#### Dynamic Polymorphism Overview
Dynamic polymorphism allows objects of derived classes to replace base class objects. This concept is often realized through virtual functions and inheritance.

In C++, this can be achieved using function overriding, where a derived class provides a specific implementation for a method defined in its base class.

:p What is dynamic polymorphism?
??x
Dynamic polymorphism in C++ refers to the ability of an object created from a derived class to replace an object created from a base class. It enables virtual functions that allow overridden methods in derived classes to be called through pointers or references to base classes. This provides flexibility and reusability by allowing different objects (derived types) to respond to the same method call in different ways.

```cpp
// Example of dynamic polymorphism using inheritance and virtual functions

class Shape {
public:
    virtual void draw() = 0; // Pure virtual function
};

class Circle : public Shape {
public:
    void draw() override { /* Draw a circle */ }
};

Shape* shapePtr;
shapePtr = new Circle(); // Polymorphic behavior
shapePtr->draw();
```

x??
---
#### Enumerated Types in C++
Enumerations are used to define a set of named integer constants. In the provided code, `ShapeType` is an enumeration that defines types for shapes.

:p What is the purpose of using enums (enumerations) in this context?
??x
Enums are used to define named constants within a specified range. Here, `ShapeType` enumerates different shape types like circle and square. This helps maintain type safety by ensuring only valid shape types can be assigned or checked against.

```cpp
enum ShapeType {
    circle,
    square
};
```

x??
---
#### Virtual Destructor in C++
A virtual destructor is a special member function declared as `virtual` that ensures the correct destruction order when deleting an object through a base class pointer. This is crucial for polymorphic behavior to ensure all derived classes are properly destroyed.

:p What is the role of a virtual destructor in dynamic polymorphism?
??x
The role of a virtual destructor in dynamic polymorphism is to ensure proper cleanup of objects, especially when dealing with derived class objects that have been created through base class pointers. If a virtual destructor is not defined, only the destructor of the immediate base class will be called during deletion, potentially leading to memory leaks or resource management issues.

```cpp
class Shape {
public:
    virtual ~Shape() = default; // Virtual default destructor
};
```

x??
---
#### Constructor Initialization in C++
In the provided code, constructors are used for initializing members. The constructor of `Circle` initializes its base class and member variables.

:p How does the constructor initialize a derived class like `Circle`?
??x
The constructor of a derived class, such as `Circle`, can call the base class constructor using the colon (:) followed by the base class name and parameters. Additionally, it initializes its own data members.

```cpp
class Circle : public Shape {
public:
    explicit Circle(double radius) 
        : Shape(circle),  // Base class initialization
          radius_(radius),
          center_{}       // Default initialized point
    {}
};
```

x??
---
#### Member Function Overriding
In C++, a derived class can override the implementation of a member function defined in its base class by declaring it with the `override` keyword. This ensures that the overridden method is correctly implemented.

:p What is overriding and how does it work in the context of dynamic polymorphism?
??x
Overriding allows a derived class to provide a specific implementation for a member function that has been declared in one or more base classes. In the given example, `Circle` overrides the pure virtual function `draw()` from the `Shape` class.

```cpp
class Circle : public Shape {
public:
    void draw() override { /* Draw a circle */ }
};
```

x??
---

#### Square Class Implementation
Background context: The Square class is a derived class from Shape, implementing specific behavior for squares. It includes a constructor that initializes with a side length and provides methods to retrieve the side length and center point.

:p What does the Square class do?
??x
The Square class implements the concept of a square as a derived class from Shape. It has a constructor that takes a side length, ensuring it is valid, and provides methods to get the side length and center point.
```cpp
class Square : public Shape {
public:
    explicit Square(double side)
        : Shape(square),
          side_(side) // Initialize with side length
    { /* Validation of side */ }

    double side() const { return side_; }
    Point center() const { return center_; } // Center can be any corner

private:
    double side_;
    Point center_{};
};
```
x??

---

#### DrawSquare Function Implementation
Background context: The `draw` function is responsible for drawing a Square. It requires including necessary headers and implementing the logic to draw the square.

:p What does the `draw` function do?
??x
The `draw` function takes a const reference to a Square object and implements the logic to draw it using some graphics library. This function demonstrates type casting based on the enum value returned by `getType()` method of the Shape base class.
```cpp
#include <DrawSquare.h>
#include <Square.h>

void draw(Square const& s) {
    // Implement drawing logic for a square
}
```
x??

---

#### DrawAllShapes Function Implementation
Background context: The `drawAllShapes` function takes a vector of unique pointers to Shape objects and uses polymorphism to determine the type of each shape, then calls the appropriate draw function.

:p What does the `drawAllShapes` function do?
??x
The `drawAllShapes` function iterates over a collection of shapes represented by `std::unique_ptr<Shape>` objects. It uses a switch statement based on the type returned from `getType()` to determine which shape it is and then calls the appropriate drawing function (Circle or Square).

```cpp
#include <DrawAllShapes.h>
#include <Circle.h>
#include <Square.h>

void drawAllShapes(std::vector<std::unique_ptr<Shape>> const& shapes) {
    for (auto const& shape : shapes) {
        switch (shape->getType()) {
            case circle:
                draw(static_cast<Circle const&>(*shape)); // Draw as Circle
                break;
            case square:
                draw(static_cast<Square const&>(*shape)); // Draw as Square
                break;
        }
    }
}
```
x??

---

#### Main Function Implementation
Background context: The `main` function creates a vector of unique pointers to Shape objects, adding different shapes (circles and squares) to the vector. It then calls `drawAllShapes` to draw all the shapes.

:p What does the main function do?
??x
The `main` function demonstrates creating and drawing multiple types of shapes using polymorphism and RAII. It creates a vector of unique pointers to Shape objects, adding both Circle and Square instances, and then passes this collection to `drawAllShapes`.

```cpp
#include <Circle.h>
#include <Square.h>
#include <DrawAllShapes.h>
#include <memory>
#include <vector>

int main() {
    using Shapes = std::vector<std::unique_ptr<Shape>>;

    // Creating some shapes
    Shapes shapes;
    shapes.emplace_back(std::make_unique<Circle>(2.3));
    shapes.emplace_back(std::make_unique<Square>(1.2));
    shapes.emplace_back(std::make_unique<Circle>(4.1));

    // Drawing all shapes
    drawAllShapes(shapes);
    return EXIT_SUCCESS;
}
```
x??

---

#### Direct Dependency on Enumeration
Background context: The problem described is about a design issue where an enumeration directly influences multiple parts of the system, leading to recompilation issues when changes are made. This violates the Open-Closed Principle (OCP), which states that software entities should be open for extension but closed for modification.

:p What is the main issue highlighted in this scenario?
??x
The main issue is that any change to the `ShapeType` enumeration necessitates recompiling all dependent source files, including classes like `Circle`, `Square`, and functions such as `drawAllShapes()`. This violates the OCP because extending functionality (e.g., adding a new shape type) should not require modifying existing code.

```cpp
enum ShapeType {
    circle,
    square,
};
```
x??

#### Violation of Open-Closed Principle
Background context: The example demonstrates how extending an `ShapeType` enumeration leads to recompilation issues across multiple files, violating the OCP. This issue arises because the design is tightly coupled with the enumeration.

:p How does adding a new shape type violate the Open-Closed Principle (OCP)?
??x
Adding a new shape type (`triangle`) requires modifying the `ShapeType` enumeration and updating the `drawAllShapes()` function, which in turn necessitates recompiling all dependent source files. This violates the OCP because classes like `Circle` and `Square` should remain unchanged when extending functionality.

```cpp
enum ShapeType {
    circle,
    square,
    triangle, // New shape type added
};
```
x??

#### Impact of Enumerations on Classes
Background context: The problem illustrates how direct dependencies on enumerations affect class design. Any change to the enumeration impacts multiple classes and functions, leading to recompilation issues.

:p How does a change in `ShapeType` impact derived classes?
??x
A change in the `ShapeType` enumeration impacts derived classes such as `Circle` and `Square`, which depend on the base class `Shape`. Specifically, adding or modifying enumerators requires updating not only the switch statement but also the classes that use these enumerators directly. This leads to recompilation of all dependent files.

```cpp
class Circle : public Shape {
public:
    static constexpr ShapeType getType() { return circle; }
};
```
x??

#### Refactoring for Better Design
Background context: The solution involves refactoring to decouple the classes from the `ShapeType` enumeration, ensuring that extending functionality does not require recompiling dependent files.

:p What is a recommended approach to avoid recompilation issues when adding new shape types?
??x
A recommended approach is to use factory methods or polymorphism instead of direct dependency on enumerations. For example, using a factory method can create shapes based on their type without modifying existing classes. This ensures that extending the system with new shape types does not require recompiling dependent files.

```cpp
class ShapeFactory {
public:
    static std::unique_ptr<Shape> createShape(ShapeType type) {
        switch (type) {
            case circle: return std::make_unique<Circle>();
            case square: return std::make_unique<Square>();
            // Add more cases for new shapes
            default: throw std::invalid_argument("Invalid shape type");
        }
    }
};
```
x??

#### Open-Closed Principle and Extension
Background context: The OCP emphasizes that software entities should be open for extension but closed for modification. This means adding functionality to existing classes should not require changing their source code.

:p How does the original design violate the OCP?
??x
The original design violates the OCP because extending the system (e.g., adding a new shape type) requires modifying the `ShapeType` enumeration and the switch statement in the `drawAllShapes()` function. This means that existing classes like `Circle` and `Square` must be altered, leading to recompilation issues.

```cpp
void drawAllShapes(std::vector<std::unique_ptr<Shape>> const& shapes) {
    for (auto const& shape : shapes) {
        switch (shape->getType()) {
            case circle: 
                draw(static_cast<Circle const&>(*shape));
                break;
```
x??

---

#### Copy-and-Paste Code Pattern
Background context explaining the concern about explicit type handling and potential maintenance issues. Highlight that this pattern often arises due to convenience but can lead to significant problems in larger codebases.

:p What is a major flaw with using switch statements or if-else cascades for drawing different shapes?
??x
This approach leads to duplicated code, making it hard to maintain and update. For example, adding a new shape requires updating every switch statement or if-else cascade, increasing the risk of introducing bugs. Additionally, this explicit type handling makes the code unmaintainable as the codebase grows.

```cpp
switch (shape->type()) {
    case square:              draw( static_cast<Square const&>( *shape ) );              break;
    case triangle:                 draw( static_cast<Triangle const&>( *shape ) );              break;
}
```
x??

---

#### Object-Oriented Approach to Drawing Shapes
Background context explaining the benefits of using an object-oriented approach, such as easier addition of new types and cleaner code structure. Highlight the use of a base class with virtual functions for polymorphism.

:p What is the advantage of implementing drawing logic in derived classes instead of switch statements?
??x
Using derived classes allows for more flexible and maintainable code. By defining a `draw()` function in the base class, each shape can implement its own drawing logic without needing to update existing switch cases or if-else cascades.

```cpp
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0; // Pure virtual function for polymorphism
};

class Circle : public Shape {
public:
    explicit Circle(double radius) : radius_(radius) {}
    double radius() const { return radius_; }
    Point center() const { return center_; }
    void draw() const override;
private:
    double radius_;
    Point center_{}; // Default constructor for a point
};

class Square : public Shape {
public:
    explicit Square(double side) : side_(side) {}
    double side() const { return side_; }
    Point center() const { return center_; }
    void draw() const override;
private:
    double side_;
    Point center_{}; // Default constructor for a point
};
```
x??

---

#### Violation of Open-Closed Principle (OCP)
Background context explaining the importance of the OCP in software design, particularly regarding adding new operations without changing existing code. Highlight that while it eases addition of types, it can violate the OCP when adding functions.

:p How does an object-oriented approach affect the ability to add new operations?
??x
While the object-oriented approach simplifies adding new types, it can make it difficult to add new operations (like `serialize()`) without modifying existing code. The base class must be changed to introduce a new virtual function, and all derived classes need to implement this function even if they don't use it.

```cpp
void drawAllShapes(std::vector<std::unique_ptr<Shape>> const& shapes) {
    for (auto const& shape : shapes) {
        shape->draw(); // Polymorphic call to the appropriate draw method
    }
}
```
x??

---

#### Adding Operations in Object-Oriented Design
Background context explaining how adding operations can violate the Open-Closed Principle and the steps required to make a class open for extension but closed for modification.

:p Why is it challenging to add a `serialize()` function using the object-oriented approach?
??x
Introducing a new virtual function like `serialize()` in the base class requires modifying that class, which breaks the Open-Closed Principle. Additionally, all derived classes (like `Circle`, `Square`) must implement this new function even if they don't use it, leading to potential code bloat and maintenance issues.

```cpp
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0; // Draw implementation in derived classes
    virtual void serialize() const = 0; // New pure virtual function for serialization
};
```
x??

---

#### Object-Oriented vs. Procedural Solutions

Background context explaining the concept of object-oriented and procedural programming, including their strengths and weaknesses.

:p How does an object-oriented solution fare with respect to adding types versus operations?

??x
An object-oriented solution adheres well to the Open-Closed Principle (OCP) when it comes to adding new types. However, it struggles with extending operations once the class structure is defined. This means that while adding new shapes or derived classes can be done without altering existing code, adding a new operation would typically require modifying multiple classes.

For example:
```java
// Example of adding a new type in an object-oriented approach
abstract class Shape {
    public void draw() { /* base implementation */ }
}

class Circle extends Shape {
    @Override
    public void draw() { /* circle-specific drawing logic */ }
}
```

In contrast, operations can be added as separate functions or classes without modifying the existing hierarchy.

x??

---

#### Procedural vs. Object-Oriented Solutions

Background context explaining the procedural approach and its strengths compared to object-oriented programming, especially in terms of adding operations.

:p How does a procedural solution handle adding operations?

??x
A procedural solution is more flexible when it comes to adding new operations. New functions or classes can be created independently of any class hierarchy, making it easier to extend the functionality without modifying existing code structures.

For example:
```java
// Example of adding an operation in a procedural approach
void drawShape(Shape shape) {
    // generic drawing logic
}

// Adding a new operation (function)
void printArea(Shape shape) {
    if (shape instanceof Circle) {
        double area = Math.PI * ((Circle) shape).getRadius() * ((Circle) shape).getRadius();
        System.out.println("Area: " + area);
    }
}
```

This approach allows for easy addition of new operations without altering the existing class hierarchy.

x??

---

#### Open-Closed Principle (OCP)

Background context explaining the OCP and its importance in software design. Include how it applies to adding types versus operations.

:p What is the OCP, and how does it apply to both adding types and operations?

??x
The Open-Closed Principle states that "software entities (classes, modules, functions, etc.) should be open for extension but closed for modification." This means that classes should be designed in such a way that new functionality can be added without altering the existing code.

When considering OCP:
- **Adding Types**: You want to be able to add new types (e.g., new shapes) by extending or modifying the base class or introducing new derived classes.
- **Adding Operations**: You want to be able to add new operations (e.g., new drawing methods) without changing existing types.

For example, in an object-oriented solution:
```java
// Open for extension but closed for modification
abstract class Shape {
    public void draw() { /* base implementation */ }
}

class Circle extends Shape {
    @Override
    public void draw() { /* circle-specific drawing logic */ }
}
```

While in a procedural solution, adding new operations is easier:
```java
// Adding an operation without modifying existing classes
void printArea(Shape shape) {
    if (shape instanceof Circle) {
        double area = Math.PI * ((Circle) shape).getRadius() * ((Circle) shape).getRadius();
        System.out.println("Area: " + area);
    }
}
```

x??

---

#### Design Choice in Dynamic Polymorphism

Background context explaining the choice between fixing types and operations for dynamic polymorphism. Discuss the trade-offs involved.

:p What is the design choice when using dynamic polymorphism, and what are its implications?

??x
When designing software with dynamic polymorphism (like object-oriented programming), you have to choose whether to fix the number of operations or the number of types:

- **Fixing Operations**: This makes it easy to add new types but harder to add new operations. You can extend the class hierarchy without changing existing code, but adding methods requires modifying multiple classes.
  
  Example:
  ```java
  // Extending with new types
  abstract class Shape {
      public void draw() { /* base implementation */ }
  }

  class Square extends Shape {
      @Override
      public void draw() { /* square-specific drawing logic */ }
  }
  ```

- **Fixing Types**: This makes it easy to add operations but harder to add new types. You can introduce new methods without modifying existing classes, but adding new shapes requires restructuring the hierarchy.

  Example:
  ```java
  // Extending with new operations
  void drawShape(Shape shape) {
      // generic drawing logic
  }

  // Adding a new operation (function)
  void printArea(Shape shape) {
      if (shape instanceof Circle) {
          double area = Math.PI * ((Circle) shape).getRadius() * ((Circle) shape).getRadius();
          System.out.println("Area: " + area);
      }
  }
  ```

The choice depends on your project's requirements and how you expect the codebase to evolve.

x??

---

