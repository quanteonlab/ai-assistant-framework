# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 29)

**Starting Chapter:** Drawing of Shapes Revisited

---

#### External Polymorphism Design Pattern
External Polymorphism is a technique where an external hierarchy manages and delegates virtual function calls to the desired implementation. This approach separates concerns by extracting polymorphic behavior into a dedicated class or template, enhancing flexibility and maintainability.

This pattern can be particularly useful when dealing with complex systems that require multiple behaviors from similar objects, such as different shapes in a drawing application.
:p What is external polymorphism?
??x
External Polymorphism is a design pattern where an external hierarchy manages and delegates virtual function calls to the desired implementation. This approach separates concerns by extracting polymorphic behavior into a dedicated class or template.
??x

---

#### Separation of Concerns (SoC)
Separation of Concerns is a design principle that advocates for dividing software functionality into distinct, independent components, each responsible for a specific aspect.

This helps in creating modular and maintainable code where changes to one component do not necessarily affect others. It's particularly beneficial when managing complex behaviors like drawing, rotating, or serializing different types of shapes.
:p What is separation of concerns?
??x
Separation of Concerns (SoC) is a design principle that advocates for dividing software functionality into distinct, independent components, each responsible for a specific aspect. This helps in creating modular and maintainable code where changes to one component do not necessarily affect others.
??x

---

#### Single Responsibility Principle (SRP)
The Single Responsibility Principle states that every class should have only one reason to change.

In the context of the provided example, SRP acts as an enabler for the Open-Closed Principle (OCP). It ensures that classes are responsible for a single part of the functionality, making it easier to add new types without modifying existing code.
:p How does SRP enable OCP?
??x
The Single Responsibility Principle (SRP) states that every class should have only one reason to change. In the context of the provided example, SRP enables the Open-Closed Principle (OCP) by ensuring that classes are responsible for a single part of the functionality. This means you can easily add new types into the ShapeConcept hierarchy without modifying existing code.
??x

---

#### Open-Closed Principle (OCP)
The Open-Closed Principle states that software entities should be open for extension but closed for modification.

In the example, the `ShapeModel` class template allows adding new nonpolymorphic shape types easily as long as they fulfill all required operations. This adheres to OCP by preventing changes in existing code when adding new functionality.
:p What is the Open-Closed Principle?
??x
The Open-Closed Principle (OCP) states that software entities should be open for extension but closed for modification. In the example, the `ShapeModel` class template allows adding new nonpolymorphic shape types easily as long as they fulfill all required operations. This adheres to OCP by preventing changes in existing code when adding new functionality.
??x

---

#### ShapeConcept and ShapeModel Classes
The `ShapeConcept` class introduces a pure virtual function, `draw()`, representing the entire set of requirements for shapes. The `ShapeModel` class template implements this interface and allows for flexible polymorphic behavior by delegating drawing operations to an external hierarchy.

This separation ensures that shape classes (`Circle` and `Square`) remain simple and unaware of specific behaviors like drawing.
:p What is the purpose of the ShapeConcept and ShapeModel classes?
??x
The `ShapeConcept` class introduces a pure virtual function, `draw()`, representing the entire set of requirements for shapes. The `ShapeModel` class template implements this interface and allows for flexible polymorphic behavior by delegating drawing operations to an external hierarchy. This separation ensures that shape classes (`Circle` and `Square`) remain simple and unaware of specific behaviors like drawing.
??x

---

#### Circle and Square Classes
The reduced versions of the `Circle` and `Square` classes are basic geometric entities with no base class or virtual functions. They are completely nonpolymorphic, making them independent of any additional operations such as drawing.

This design allows for easy addition of new shape types that fulfill all required operations without modifying existing code.
:p What is the role of Circle and Square classes in this example?
??x
The reduced versions of the `Circle` and `Square` classes are basic geometric entities with no base class or virtual functions. They are completely nonpolymorphic, making them independent of any additional operations such as drawing. This design allows for easy addition of new shape types that fulfill all required operations without modifying existing code.
??x

---

#### DrawStrategy in ShapeModel
The `DrawStrategy` type alias in the `ShapeModel` class template is a function object (`std::function`) that represents the drawing logic for shapes.

It abstracts away the specific implementation details, allowing for flexible and extensible behavior.
:p What is the role of the DrawStrategy in the ShapeModel?
??x
The `DrawStrategy` type alias in the `ShapeModel` class template is a function object (`std::function`) that represents the drawing logic for shapes. It abstracts away the specific implementation details, allowing for flexible and extensible behavior.
??x

---
These flashcards cover key concepts from the provided text, ensuring familiarity with external polymorphism, separation of concerns, single responsibility principle, open-closed principle, and design patterns in class templates.

#### ShapeConcept Abstraction and LSP Compliance
Background context explaining that the `ShapeConcept` represents a classic abstraction adhering to the Liskov Substitution Principle (LSP), ensuring that objects of derived classes can be used interchangeably with their base class without breaking the expected behavior. This is crucial for maintaining the integrity and reliability of software designs.

:p What does the ShapeConcept represent in terms of design patterns?
??x
The `ShapeConcept` represents a classic abstraction designed to ensure adherence to the Liskov Substitution Principle (LSP). It provides a blueprint that derived classes, such as `Circle`, `Square`, etc., must follow. This ensures that objects of these derived classes can be used interchangeably with their base class `ShapeConcept` without causing any issues in behavior.
x??

---

#### ShapeModel Class Template and Instantiation
Background context explaining how the `ShapeModel` class template is instantiated for various shape types, such as Circle or Square, to provide polymorphic behavior. The template parameter `ShapeT` acts as a placeholder for these specific shapes.

:p How does the `ShapeModel` instantiate different shape objects?
??x
The `ShapeModel` class template instantiates different shape objects based on the template parameter `ShapeT`. For example, if `ShapeT` is `Circle`, an instance of the `Circle` class will be created. The `ShapeModel` acts as a wrapper that augments these specific shapes with required polymorphic behavior like the `draw()` function.
x??

---

#### Composition Over Inheritance
Background context explaining why composition is preferred over inheritance to avoid complex and tightly coupled class hierarchies.

:p Why is composition used in this design?
??x
Composition is used in this design because it allows for more flexible and decoupled code. Instead of inheriting from `ShapeConcept`, the `ShapeModel` contains an instance of a specific shape type, such as `Circle`. This approach adheres to "Guideline 20: Favor Composition over Inheritance," making the system easier to maintain and extend.
x??

---

#### Drawing Strategy Implementation
Background context explaining how `ShapeModel` implements drawing behavior through strategies stored in `std::function`.

:p How does `ShapeModel` handle drawing operations?
??x
`ShapeModel` handles drawing operations by storing an instance of a `DrawStrategy` within its composition. When the `draw()` function is triggered, it uses this strategy to perform the actual drawing operation. This approach decouples the implementation details of drawing from the `ShapeModel`, allowing for greater flexibility and easier maintenance.
x??

---

#### Flexibility in Drawing Implementation
Background context explaining that while `std::function` is used here, other approaches are also possible.

:p What alternative methods can be used to implement drawing within `ShapeModel`?
??x
While using `std::function` with a `DrawStrategy` is one approach, there are other flexible methods. For example, you could forward the drawing request to a member function of the shape type or to a free function. The key is ensuring that any type used to instantiate `ShapeModel` meets the necessary requirements for the specific drawing implementation.
x??

---

#### Inversion of Control with Free Functions
Background context explaining how using free functions can invert dependencies, similar to the Strategy pattern.

:p How does using a free function in `ShapeModel::draw()` achieve inversion of control?
??x
Using a free function in `ShapeModel::draw()` achieves inversion of control by decoupling the drawing logic from both the `ShapeModel` and the specific shape type. This allows for more flexible and loosely coupled code, similar to how the Strategy pattern works. The exact function can be chosen based on its name or signature, providing a powerful way to manage dependencies.
x??

---

#### Template-Based ShapeModel Implementation

This section discusses how `ShapeModel` acts as a templated version of initial shape classes, offering several benefits over traditional implementations. It helps to reduce boilerplate code and improves implementation by adhering to the DRY (Don't Repeat Yourself) principle.

:p What is the key advantage of using `ShapeModel` over initial shape classes?
??x
The key advantages include reducing boilerplate code and improving adherence to the DRY principle, as `ShapeModel` can be used across different shapes with minimal additional setup. Additionally, it allows for easy switching between runtime and compile-time Strategy implementations.

```cpp
template< typename ShapeT, typename DrawStrategy >
class ShapeModel : public ShapeConcept {
public:
    explicit ShapeModel(ShapeT shape, DrawStrategy drawer)
        : shape_{ std::move(shape) }, drawer_{ std::move(drawer) }
    {}

    void draw() const override { drawer_(shape_); }

private:
    ShapeT shape_;
    DrawStrategy drawer_;
};
```
x??

---

#### Default Drawer Implementation

A `DefaultDrawer` is provided as a default strategy that can be used when no specific drawing Strategy is defined. This simplifies the usage of `ShapeModel`.

:p How does the `DefaultDrawer` template work?
??x
The `DefaultDrawer` template provides a default implementation for the drawing behavior, allowing it to accept any type and call its `draw` method.

```cpp
struct DefaultDrawer {
    template< typename T >
    void operator()(T const& obj) const {
        draw(obj);
    }
};
```
x??

---

#### Compile-Time Strategy Implementation

The example illustrates how a compile-time Strategy can be passed to the `ShapeModel` class, which can replace the use of `std::function`.

:p What are the benefits of using compile-time Strategy over runtime?
??x
Using a compile-time Strategy with `ShapeModel` offers several benefits: reduced runtime indirections (improving performance), avoiding template arguments in all shape classes, and keeping the code DRY by limiting the modification to one place.

```cpp
template< typename ShapeT, typename DrawStrategy = DefaultDrawer >
class ShapeModel : public ShapeConcept {
public:
    explicit ShapeModel(ShapeT shape, DrawStrategy drawer = DefaultDrawer{})
        // ... as before

private:
    ShapeT shape_;
    DrawStrategy drawer_;
};
```
x??

---

#### Dependency Reduction and Polymorphism

The text highlights the reduction of dependencies through this design pattern and mentions that the approach combines runtime and compile-time polymorphism effectively.

:p How does combining runtime and compile-time Strategy benefit the implementation?
??x
Combining runtime and compile-time Strategy benefits the implementation by reducing dependencies, improving performance due to fewer runtime indirections, and adhering to the DRY principle without cluttering all shape classes with template arguments.

```cpp
// Example of how this could be visualized in a dependency graph (not shown)
// Shows a simplified view where ShapeModel is the central class that handles drawing strategies.
```
x??

---

#### Summary

This section provides an overview of `ShapeModel` and its benefits, including how it reduces boilerplate code, supports compile-time Strategy implementations, and minimizes dependencies.

:p What are the main objectives of using the `ShapeModel` design?
??x
The main objectives include reducing boilerplate code, supporting compile-time Strategy implementations to improve performance, maintaining adherence to the DRY principle by centralizing configuration in one place, and minimizing dependencies through a well-structured class hierarchy.

```cpp
// Code examples are provided above for illustration.
```
x??

---

#### External Polymorphism Design Pattern
This pattern allows for polymorphic behavior without requiring inheritance or composition between shape classes. Instead, it uses a template approach to integrate drawing strategies with specific shape implementations.

:p What is the external polymorphism design pattern used for?
??x
The external polymorphism design pattern is used to achieve polymorphism by using templates and separate draw strategy classes, rather than relying on inheritance or composition. This decouples the shape types from the drawing logic, allowing for easier addition of new shapes and drawing strategies without modifying existing code.

```cpp
// Example of how a Circle class might be implemented
class Circle {
public:
    explicit Circle(double radius) : m_radius(radius) {}
    
    // Other methods related to circle implementation
private:
    double m_radius;
};

// OpenGLDrawStrategy example
class OpenGLDrawStrategy {
public:
    explicit OpenGLDrawStrategy(/* Drawing related arguments */);
    void operator()(Circle const& circle) const;
    void operator()(Square const& square) const;

private:
    /* Drawing related data members, e.g. colors, textures, ... */
};
```
x??

---

#### ShapeModel and DrawStrategy Integration
The `ShapeModel` class template is used to integrate specific shapes with their corresponding drawing strategies. This integration happens at the lowest level of the architecture.

:p How does `ShapeModel` integrate shapes with drawing strategies?
??x
`ShapeModel` integrates specific shapes with their corresponding drawing strategies by being instantiated for a particular shape and draw strategy combination. The instantiation is done at runtime, allowing dynamic selection of both the shape type and its drawing behavior.

```cpp
// Example of ShapeModel template
template <typename TShape, typename TDrawStrategy>
class ShapeModel {
public:
    explicit ShapeModel(TShape& shape, const TDrawStrategy& drawStrategy) : m_shape(shape), m_drawStrategy(drawStrategy) {}

    void draw() const { m_drawStrategy(m_shape); }

private:
    TShape& m_shape;
    const TDrawStrategy& m_drawStrategy;
};
```
x??

---

#### Instantiation of Shapes with Draw Strategies
In the main function, specific instances of `ShapeModel` are created for each shape type and drawing strategy.

:p How are shapes instantiated in the main function?
??x
Shapes are instantiated using `std::make_unique` to encapsulate the instantiation within a smart pointer. This approach ensures resource management is handled automatically by the smart pointers.

```cpp
// Example of main function
int main() {
    using Shapes = std::vector<std::unique_ptr<ShapeConcept>>;

    using CircleModel = ShapeModel<Circle, OpenGLDrawStrategy>;
    using SquareModel = ShapeModel<Square, OpenGLDrawStrategy>;

    Shapes shapes{};

    // Creating some shapes with OpenGL drawing strategy
    shapes.emplace_back(std::make_unique<CircleModel>(Circle{2.3}, OpenGLDrawStrategy{/*...red...*/}));
    shapes.emplace_back(std::make_unique<SquareModel>(Square{1.2}, OpenGLDrawStrategy{/*...green...*/}));
    shapes.emplace_back(std::make_unique<CircleModel>(Circle{4.1}, OpenGLDrawStrategy{/*...blue...*/}));

    // Drawing all shapes
    for (auto const& shape : shapes) {
        shape->draw();
    }

    return EXIT_SUCCESS;
}
```
x??

---

#### Benefits of External Polymorphism
The approach using `ShapeModel` and separate draw strategies offers several benefits, including loose coupling, easier addition of new shapes, and adherence to SOLID principles.

:p What are the main advantages of using external polymorphism?
??x
The main advantages include:

1. **Loose Coupling:** Shapes and drawing behaviors are decoupled, making it easier to modify one without affecting the other.
2. **Simplicity in Shape Types:** Shape classes become simpler and nonpolymorphic.
3. **Easier Addition of New Shapes:** Adding new shapes or drawing strategies is straightforward since no changes are needed in existing code.
4. **No Inheritance Overhead:** No need for inheritance, reducing boilerplate code and adhering to the Open/Closed Principle (OCP).
5. **DRY Principle Adherence:** Drawing logic is implemented only once within `OpenGLDrawStrategy`.
6. **Adheres to DIP:** The `ShapeConcept` and `ShapeModel` work together to form a stable abstraction.
7. **Performance Improvement:** Fewer indirections due to template usage.

x??

---

#### Non-Intrusive Polymorphism
Even simple types like integers can be equipped with polymorphic behavior using this pattern, making it highly versatile.

:p How does external polymorphism allow non-intrusive addition of polymorphic behavior?
??x
External polymorphism allows adding polymorphic behavior to any type without modifying the original type. This is achieved by creating a `ShapeModel` for that type and providing the necessary drawing strategy.

```cpp
// Example of adding polymorphic behavior to an int
class IntModel : public ShapeConcept {
public:
    explicit IntModel(int value) : m_value(value) {}

    void draw() const override { 
        // Draw logic for integer, e.g., text rendering
    }

private:
    int m_value;
};

int main() {
    using Shapes = std::vector<std::unique_ptr<ShapeConcept>>;

    using IntModel = ShapeModel<IntModel, OpenGLDrawStrategy>;

    Shapes shapes{};

    shapes.emplace_back(std::make_unique<IntModel>(42));

    for (auto const& shape : shapes) {
        shape->draw();
    }

    return EXIT_SUCCESS;
}
```
x??

---

#### External Polymorphism Design Pattern
Background context: The External Polymorphism design pattern is a technique for introducing polymorphic behavior to non-polymorphic types without modifying their source code. This allows us to treat different types uniformly through a common interface, promoting loose coupling and abstraction.

:p What does the External Polymorphism design pattern allow?
??x
It allows adding polymorphic capabilities to existing, non-polymorphic types by creating an adapter or proxy that conforms to a common interface. This is achieved without modifying the original type's source code.
x??

---

#### Comparison Between External Polymorphism and Adapter Pattern
Background context: Both the External Polymorphism design pattern and the Adapter design pattern aim to make different types work together, but they do so in slightly different ways.

:p How are the External Polymorphism and Adapter design patterns similar?
??x
Both patterns enable treating non-polymorphic objects as if they were polymorphic by creating an adapter or wrapper that conforms to a common interface.
x??

---

#### Comparison Between External Polymorphism and Adapter Pattern (continued)
Background context: While both patterns standardize interfaces, the External Polymorphism design pattern specifically creates a new hierarchy for abstraction purposes, whereas the Adapter pattern adapts existing types to fit into pre-existing interfaces.

:p How do the External Polymorphism and Adapter design patterns differ?
??x
The External Polymorphism design pattern creates a new external hierarchy to abstract from non-polymorphic types. The Adapter design pattern focuses on adapting an object or function to match an existing interface.
x??

---

#### Example of Using `ShapeModel` with External Polymorphism
Background context: In the provided code snippet, we see how an `int` can be equipped with polymorphic behavior using a `ShapeModel<int>`.

:p How does the example code demonstrate external polymorphism?
??x
The example demonstrates that an `int` can be treated polymorphically by creating a `ShapeModel<int>` and calling its `draw()` method. This shows how to add polymorphism without modifying the `int` type directly, adhering to the External Polymorphism design pattern.

Code Example:
```cpp
#include <memory>

class Shape {
public:
    virtual void draw() = 0;
};

template<typename T>
class ShapeModel : public Shape {
private:
    T value;

public:
    ShapeModel(T val) : value(val) {}

    void draw() override {
        // Implementation of drawing the value, e.g., printing it
        std::cout << "Drawing: " << value << std::endl;
    }
};

int main() {
    auto shape = std::make_unique<ShapeModel<int>>(42);
    shape->draw();  // Polymorphic behavior through ShapeModel
    return EXIT_SUCCESS;
}
```
x??

---

#### Shortcomings of External Polymorphism Design Pattern
Background context: While the External Polymorphism design pattern is powerful, it comes with certain limitations that developers should be aware of.

:p What are some shortcomings of the External Polymorphism design pattern?
??x
One major shortcoming is that it does not fulfill the expectations of a clean and simple solution. It doesn't help to reduce pointers, minimize manual allocations, simplify inheritance hierarchies, or ease user code. This makes it less desirable in scenarios where these factors are critical.
x??

---

#### Summary: Adapting Non-Polymorphic Types
Background context: The External Polymorphism design pattern is useful for adding polymorphic capabilities to non-polymorphic types by creating an external hierarchy.

:p When should the External Polymorphism design pattern be considered?
??x
Consider using the External Polymorphism design pattern when you want to introduce polymorphism without modifying existing code, particularly when dealing with a set of related, non-polymorphic types.
x??

---

---

#### External Polymorphism and Its Limitations

External Polymorphism can introduce complexity when dealing with specific interfaces, as it may require explicit instantiation. This can make user code more cumbersome.

:p What is external polymorphism, and why might it increase the complexity of user code?
??x
External Polymorphism refers to a technique where different types are treated uniformly by their common interface or behavior. In the context provided, this means using templates and class hierarchies to handle various shapes or documents with uniform methods. However, the explicit instantiation required can make the user code more complex because it forces specific types to be known at compile time.

For example:
```cpp
template <typename ShapeT>
class ShapeModel {
public:
    void draw(ShapeT& shape) const;
private:
    ShapeT model_;
};

// Usage requires explicit instantiation
ShapeModel<Circle> circleModel;
circleModel.draw(circle);
```
x??

---

#### Polymorphism and Abstraction

Just as any other base class, a base class used in external polymorphism is still subject to the Interface Segregation Principle (ISP). This means that classes should not be forced to implement methods they do not use.

:p How does the Interface Segregation Principle (ISP) apply to external polymorphism?
??x
The Interface Segregation Principle (ISP) states that no client should be forced to depend on methods it does not use. In the provided example, `DocumentConcept` combines two interfaces (`JSONExportable` and `Serializable`) into one class hierarchy.

If only JSON export functionality is needed, forcing serialization might violate ISP because the exporting function may not need or know about `ByteStream`.

For instance:
```cpp
void exportDocument(DocumentConcept const& doc) {
    // This forces an artificial dependency on ByteStream
    doc.exportToJSON(/* pass necessary arguments */);
}
```
To adhere to ISP, these interfaces should be segregated into orthogonal aspects.

x??

---

#### Segregating Interfaces

Segregating interfaces can help in reducing the coupling and making the design more flexible. By separating concerns like JSON export and serialization, we can provide better abstraction and avoid artificial dependencies.

:p Why is it important to segregate interfaces when using external polymorphism?
??x
Segregating interfaces helps to reduce unnecessary complexity and coupling between classes. When you have a large class hierarchy with many methods, some of which are not relevant for certain use cases, it violates the Interface Segregation Principle (ISP). By separating these concerns into smaller, more focused interfaces, you can provide better abstraction and avoid forcing clients to depend on functionalities they do not need.

For example:
```cpp
class JSONExportable {
public:
    virtual ~JSONExportable() = default;
    virtual void exportToJSON(/*...*/) const = 0;
};

class Serializable {
public:
    virtual ~Serializable() = default;
    virtual void serialize(ByteStream& bs, /*...*/) const = 0;
};

template <typename DocumentT>
class DocumentModel : public JSONExportable, public Serializable {
public:
    // ...
    void exportToJSON(/*...*/) const override;
    void serialize(ByteStream& bs, /*...*/) const override;
private:
    DocumentT document_;
};
```
Now, functions that only need JSON export functionality can accept `JSONExportable` interfaces without being dependent on serialization.

x??

---

#### Example of Segregated Interfaces

In the example provided, separating concerns into `JSONExportable` and `Serializable` interfaces allows for better abstraction. Functions interested in specific functionalities (like just exporting to JSON) can now depend only on those interfaces, reducing unnecessary dependencies.

:p How does separating `DocumentConcept` into `JSONExportable` and `Serializable` help reduce coupling?
??x
Separating the `DocumentConcept` into two orthogonal interfaces (`JSONExportable` and `Serializable`) helps in reducing coupling by allowing clients to depend only on the functionalities they need. This approach adheres to the Interface Segregation Principle (ISP), which suggests that a client should not be forced to depend on methods it does not use.

For example, a function that only needs to export documents as JSON can now accept `JSONExportable` objects without being dependent on serialization:
```cpp
void exportDocument(JSONExportable const& exportable) {
    // This function only depends on the JSON export functionality.
    exportable.exportToJSON(/* pass necessary arguments */);
}
```
This separation makes the design more modular and easier to maintain, as it avoids forcing unnecessary dependencies.

x??

---

#### Duck Typing with External Polymorphism

Similar to duck typing in Python where an object is considered a duck if it quacks like one, external polymorphism allows treating types based on their behavior. This can sometimes lead to misuse or unexpected behaviors, similar to how pretending that an `int` is a `Shape` might not be semantically correct.

:p How does duck typing apply to the context of external polymorphism?
??x
Duck typing in Python and its application to external polymorphism both revolve around the idea that the type of an object matters only by its behavior. In C++, this means that if a type can provide the necessary methods, it can be used interchangeably with another type that provides the same or compatible methods.

However, just as duck typing in Python might lead to unexpected behaviors when types are not truly interchangeable (like pretending an `int` is a `Shape`), external polymorphism also risks introducing issues if interfaces are not well-defined or if types are incorrectly assumed to be compatible.

For example:
```cpp
class FakeDocument {
public:
    void exportToJSON(/*...*/) const;
};

void exportDocument(DocumentConcept const& doc) {
    // This forces an artificial dependency on ByteStream.
    doc.exportToJSON(/* pass necessary arguments */);
}

// Now, a `FakeDocument` can be mistakenly used in the same way as `DocumentModel`.
exportDocument(FakeDocument());
```
This misuse could lead to runtime errors or unexpected behaviors if the `FakeDocument` does not implement all required methods.

x??

---

#### Liskov Substitution Principle (LSP) and Shape Concept

Background context: The Liskov Substitution Principle states that objects of a superclass shall be replaceable with objects of its subclasses without breaking the application. In the provided text, this principle is applied to ensure that any shape class instantiated through `ShapeModel` behaves as expected.

:p What does the LSP require in terms of object substitution?
??x
The LSP requires that objects of a subclass can be substituted for their superclass without affecting the correctness of the program. This means that if Circle and Square are subclasses of Shape, then any operation that works with a Shape should also work correctly when the operation is performed on a Circle or Square.

In the context provided:
```cpp
class Shape {
public:
    virtual void draw() const = 0;
};

class Circle : public Shape {
public:
    void draw() const override { /* Draw circle */ }
};

class Square : public Shape {
public:
    void draw() const override { /* Draw square */ }
};
```
x??

---

#### Guideline 31: Use External Polymorphism for Nonintrusive Runtime Polymorphism

Background context: External Polymorphism allows you to add polymorphic behavior to non-polymorphic types without modifying their source code, thus achieving loose coupling. This is particularly useful when dealing with value types or types that cannot be modified due to external factors.

:p What does the guideline suggest regarding the use of External Polymorphism?
??x
The guideline suggests applying the External Polymorphism design pattern to enable polymorphic treatment of non-polymorphic types in a non-intrusive manner. This means adding virtual functions or interfaces to types that were originally not designed with polymorphism in mind, ensuring that such types can participate in polymorphic behavior without altering their original implementation.

For example:
```cpp
class ShapeModel {
public:
    void draw() const { model->draw(); }
private:
    std::unique_ptr<Shape> model;
};

// Usage
Circle circle;
Square square;

ShapeModel model1(circle);
ShapeModel model2(square);

model1.draw();
model2.draw();
```
x??

---

#### ABI Stability and `std::unique_ptr`

Background context: Application Binary Interface (ABI) stability is crucial in C++ to ensure that changes in the library do not break existing binaries. The use of `std::unique_ptr` affects copyability, as it does not support copying due to its move-only semantics.

:p What issues arise when using `std::unique_ptr` for an ElectricEngine class?
??x
Using `std::unique_ptr` for a class like `ElectricEngine` can render the class noncopyable because `std::unique_ptr` is designed to be movable but not copyable. This means that any class containing a `std::unique_ptr` will also be noncopyable unless it is explicitly handled through custom copy constructors and assignment operators.

For example, if you have:
```cpp
class ElectricEngine {
public:
    void start() { /* Engine starts */ }
};

// Using std::unique_ptr
std::unique_ptr<ElectricEngine> engine = std::make_unique<ElectricEngine>();
```
Switching to `std::unique_ptr` will make the class noncopyable, which might affect its usage in certain contexts.

x??

---

#### Adapter vs. External Polymorphism

Background context: Both the Adapter pattern and External Polymorphism involve adapting interfaces but serve different purposes. The Adapter pattern is used to make incompatible interfaces compatible by providing a wrapper interface that clients do not need to understand, whereas External Polymorphism aims to add polymorphic behavior to types without modifying their source code.

:p How does External Polymorphism differ from the Adapter pattern?
??x
External Polymorphism and the Adapter pattern both aim to adapt interfaces but serve different purposes. The key difference is:

- **Adapter**: It is used to make incompatible interfaces compatible by providing a wrapper interface that clients do not need to understand.
  
- **External Polymorphism**: It allows adding polymorphic behavior to non-polymorphic types without modifying their source code, achieving loose coupling.

For example:
```cpp
// Adapter Example
class OldInterface {
public:
    void oldMethod() { /* old implementation */ }
};

class NewAdapter : public NewInterface {
public:
    OldInterface* oldObj;
    NewAdapter(OldInterface& obj) : oldObj(&obj) {}
    void newMethod() override { oldObj->oldMethod(); }
};
```
Versus
```cpp
// External Polymorphism Example
class Shape {
public:
    virtual void draw() const = 0;
};

class OldType {};
std::unique_ptr<Shape> adapter(const OldType& obj) {
    return std::make_unique<OldTypeAdapter>(obj);
}

class OldTypeAdapter : public Shape {
public:
    OldType* oldObj;
    OldTypeAdapter(OldType& obj) : oldObj(&obj) {}
    void draw() const override { /* Draw the old type polymorphically */ }
};
```
x??

---

#### Summary of Key Concepts

Background context: The text emphasizes the importance of adhering to design principles like LSP and External Polymorphism for building robust, maintainable software. It also highlights the benefits of decoupling through non-intrusive solutions.

:p What key insights does the provided text convey?
??x
The text conveys several key insights:
- **LSP**: Objects should be replaceable with objects of their subclasses without breaking the application.
- **External Polymorphism**: Allows adding polymorphic behavior to types without modifying them, achieving loose coupling and reducing dependencies.
- **ABI Stability**: Important considerations when dealing with binary compatibility in C++.

These principles are essential for building flexible and maintainable software designs. The examples provided illustrate how these concepts can be applied in practice to achieve better code quality and flexibility.

x??

#### Rule of 5 and Copy Semantics
When implementing copy operations, you must manually define them to preserve semantics. The Rule of 5 states that if a class needs to manage resources like memory, it should explicitly implement move constructors and move assignment operators (to avoid unnecessary copying) and may need to delete or define the copy constructor, copy assignment operator, and destructor.
:p What is the Rule of 5?
??x
The Rule of 5 ensures that classes properly handle resource management by either deleting or defining custom implementations for the five special member functions: move constructor, move assignment operator, copy constructor, copy assignment operator, and destructor. If any of these are deleted, the other ones should also be defined to ensure consistent behavior.
```cpp
class Person {
public:
    // Constructor
    Person() = default;
    
    // Copy constructor
    Person(const Person& other) { /* ... */ }
    
    // Move constructor
    Person(Person&& other) noexcept { /* ... */ }
    
    // Copy assignment operator
    Person& operator=(const Person& other) { /* ... */ return *this; }
    
    // Move assignment operator
    Person& operator=(Person&& other) noexcept { /* ... */ return *this; }
    
    // Destructor
    ~Person() = default;
};
```
x??

---

#### Prefer Sticking to the Rule of 5
To avoid potential issues, prefer sticking to the Rule of 5. The move operations are typically expected to be `noexcept` due to Core Guideline C.66, but this might not always be possible.
:p Why is it recommended to stick with the Rule of 5?
??x
Sticking to the Rule of 5 helps ensure that your class behaves correctly and efficiently when dealing with resource management. By properly defining move constructors and move assignment operators, you can avoid unnecessary copies and potential issues like exception safety problems. The `noexcept` qualifier for move operations is recommended as it indicates that these operations are very unlikely to throw exceptions.
```cpp
class Person {
public:
    // Move constructor
    Person(Person&& other) noexcept { /* ... */ }

    // Move assignment operator
    Person& operator=(Person&& other) noexcept { /* ... */ return *this; }
};
```
x??

---

#### Unique_ptr and Noexcept Considerations
When using `std::unique_ptr` as a data member, the move operations are expected to be `noexcept`. However, this might not always be possible if you assume that some `std::unique_ptr` data member is never `nullptr`.
:p Why do move operations for classes containing `std::unique_ptr` need to be `noexcept`?
??x
Move operations for classes containing `std::unique_ptr` are expected to be `noexcept` because it indicates that the operation will not throw exceptions. This is important for performance and exception safety, as it allows the compiler to optimize certain operations. However, if you have a scenario where a `std::unique_ptr` can never be `nullptr`, this might lead to non-`noexcept` move operations, which could result in less efficient code.
```cpp
class Person {
public:
    // Move constructor
    Person(Person&& other) noexcept { /* ... */ }

    // Move assignment operator
    Person& operator=(Person&& other) noexcept { /* ... */ return *this; }
};
```
x??

---

#### Dynamic Allocation and Class Size
Dynamic allocation can significantly affect class size. The size of a `std::string` depends on the compiler implementation, which can vary widely.
:p How does dynamic allocation impact class size?
??x
Dynamic allocation can lead to significant increases in class size due to padding and alignment requirements imposed by the compiler. For example, a `std::string`'s size varies between different compilers and versions, affecting the overall size of classes that use it as a member.
```cpp
class Person1 {
public:
    std::string name;
    int year_of_birth;
};

// Example sizes with different compilers
int main() {
    // With Clang 11.1: total size = 6 * 24 (std::string) + 4 (int) + padding = 152 bytes
    // With GCC 11.1: total size = 6 * 32 (std::string) + 4 (int) + padding = 200 bytes
}
```
x??

---

#### Data-Oriented Design and Performance
To improve performance, consider arranging data based on usage patterns. For example, storing all `year_of_birth` values in a single static vector can optimize memory access.
:p How does data-oriented design help with performance?
??x
Data-oriented design focuses on organizing data to facilitate efficient memory access and reduce cache misses. By grouping related data together, you can improve locality of reference, which leads to better cache utilization and faster execution. For instance, storing all `year_of_birth` values in a single static vector reduces the number of memory accesses required to access these values.
```cpp
class Person2 {
public:
    int year_of_birth;
};

// Example data-oriented design optimization
std::vector<int> years_of_birth;

void processPersons(const std::vector<Person2>& persons) {
    for (const auto& person : persons) {
        const auto index = &person - &persons[0];
        years_of_birth[index] = person.year_of_birth;
    }
}
```
x??

---

#### Understanding the Rule of 5
The Rule of 5 ensures that classes properly handle resource management by either deleting or defining custom implementations for move constructors, move assignment operators, copy constructor, copy assignment operator, and destructor.
:p What is the Rule of 5?
??x
The Rule of 5 is a set of guidelines for managing resources in C++ classes. It states that if you manage resources like memory with smart pointers, you should define or delete the following special member functions: move constructor, move assignment operator, copy constructor, copy assignment operator, and destructor. This ensures proper resource management and avoids common pitfalls.
```cpp
class Person {
public:
    // Constructor
    Person() = default;
    
    // Copy constructor
    Person(const Person& other) { /* ... */ }
    
    // Move constructor
    Person(Person&& other) noexcept { /* ... */ }
    
    // Copy assignment operator
    Person& operator=(const Person& other) { /* ... */ return *this; }
    
    // Move assignment operator
    Person& operator=(Person&& other) noexcept { /* ... */ return *this; }
    
    // Destructor
    ~Person() = default;
};
```
x??

---

#### Raw Pointers and Non-Ownership
Raw pointers are nonowning, meaning they do not manage the lifetime of the pointed-to object. This can make it difficult to use language features like covariant return types.
:p What is a raw pointer and why might it be problematic?
??x
A raw pointer is a simple pointer that does not own or manage the memory it points to. This means you must ensure that the lifetime of the pointed-to object is properly managed elsewhere in your code. Using raw pointers can lead to issues like dangling pointers, double deletion, and memory leaks.

Using raw pointers with covariant return types might be problematic because the language feature requires ownership semantics, which are not provided by nonowning raw pointers.
```cpp
class Base {};
class Derived : public Base {};

// Incorrect usage of covariant return type with raw pointer
Base* clone() const { return new Derived(); }
```
x??

---

#### Template Method Pattern for Cloning
If you need to use a covariant return type in the `clone()` function, consider splitting it into a private virtual function returning a raw pointer and a public non-virtual function returning `std::unique_ptr`.
:p How can you use the Template Method pattern for cloning?
??x
You can use the Template Method pattern to split the `clone()` function into two parts: a private virtual function that returns a raw pointer, and a public non-virtual function that calls this private function and returns a `std::unique_ptr`. This allows you to maintain covariant return types while ensuring proper resource management.
```cpp
class Base {
protected:
    virtual void* doClone() const = 0;

public:
    std::unique_ptr<Base> clone() const {
        return std::unique_ptr<Base>(static_cast<Base*>(doClone()));
    }
};

class Derived : public Base {
private:
    void* doClone() const override { return new Derived(*this); }

public:
    // No need to define the public non-virtual function here
};
```
x??

---

#### External Polymorphism and Design Patterns
External Polymorphism is a design pattern that allows you to transparently extend concrete data types. It is often used in conjunction with Type Erasure patterns.
:p What is External Polymorphism?
??x
External Polymorphism is an object structural pattern that enables you to transparently extend the behavior of existing concrete data types without modifying their implementation. This is particularly useful when you want to add functionality or customize operations on a type without changing its source code.

External Polymorphism plays a major role in Type Erasure patterns, where it helps achieve dynamic dispatch and polymorphic behavior.
```cpp
template <typename T>
class Wrapper {
public:
    virtual void doSomething() const = 0;
};

// Example usage of External Polymorphism
void useWrapper(Wrapper<int>* wrapper) {
    // The function can work with any type that implements the Wrapper interface
    wrapper->doSomething();
}
```
x??

---

#### Data-Oriented Design and Paradigm
Data-oriented design focuses on organizing data to facilitate efficient memory access and reduce cache misses. It is particularly useful when performance is critical.
:p What is Data-Oriented Design?
??x
Data-Oriented Design (DOD) is a software engineering paradigm that emphasizes the organization of data in a way that optimizes memory access patterns and improves cache utilization. By grouping related data together, DOD can reduce cache misses, improve locality of reference, and generally enhance performance.

For example, storing all `year_of_birth` values from multiple `Person` objects in a single static vector can optimize memory access.
```cpp
class Person {
public:
    int year_of_birth;
};

// Example Data-Oriented Design optimization
std::vector<int> years_of_birth;

void processPersons(const std::vector<Person>& persons) {
    for (const auto& person : persons) {
        const auto index = &person - &persons[0];
        years_of_birth[index] = person.year_of_birth;
    }
}
```
x??

---

---
#### Type Erasure: Introduction and Core Concepts
This chapter introduces one of the most exciting modern C++ design patterns, type erasure. It combines two essential principles from the book—separation of concerns and value semantics—to provide a robust solution for dynamic polymorphism without the drawbacks of traditional inheritance hierarchies.

:p What is type erasure in C++, and how does it combine separation of concerns and value semantics?
??x
Type erasure allows you to treat different types as if they were the same type, enabling polymorphic behavior while maintaining clean code structures. It combines separation of concerns by separating the interface from the implementation details, ensuring that each part of the program focuses on its specific responsibilities. Value semantics ensure that instances can be copied and moved efficiently without worrying about ownership or state.

For example:
```cpp
// Example class hierarchy
class Base {
public:
    virtual void func() = 0;
};

class Derived1 : public Base {
public:
    void func() override { /* implementation */ }
};

class Derived2 : public Base {
public:
    void func() override { /* different implementation */ }
};
```

With type erasure, you can create a wrapper that hides the derived class type while still allowing polymorphic behavior.
??x
---

---
#### Basic Owning Type Erasure Implementation
This section will walk through implementing a basic owning type erasure pattern in C++. The idea is to encapsulate the dynamic dispatch mechanism within an opaque wrapper, making it easy to manage and ensuring that the wrapped object can be treated as if it were of a single base class.

:p How would you implement a basic owning type erasure using `std::unique_ptr`?
??x
To implement a basic owning type erasure, you could create a template class that wraps a `std::unique_ptr` to a base class and uses virtual functions for polymorphic behavior. Here's an example:

```cpp
#include <memory>

template<typename Base>
class TypeEraser {
private:
    std::unique_ptr<Base> p;

public:
    template<typename Derived>
    TypeEraser(std::unique_ptr<Derived>&& ptr) : p(std::move(ptr)) {}

    void callFunc() {
        p->func();
    }
};

// Usage
int main() {
    auto derived1 = std::make_unique<Derived1>();
    auto eraser1 = TypeEraser<Base>(std::move(derived1));

    auto derived2 = std::make_unique<Derived2>();
    auto eraser2 = TypeEraser<Base>(std::move(derived2));

    eraser1.callFunc(); // Calls Derived1's func
    eraser2.callFunc(); // Calls Derived2's func

    return 0;
}
```

In this example, `TypeEraser` is a template class that takes the base type as a parameter. It uses a `std::unique_ptr` to manage ownership of the derived object and provides a virtual function call mechanism.
??x
---

---
#### Optimization Potential with Type Erasure
This guideline explores how to optimize type erasure implementations, particularly by applying techniques like Small Buffer Optimization (SBO) and manual virtual dispatch. The goal is to reduce overhead while maintaining flexibility.

:p What is the Small Buffer Optimization (SBO), and why might you want to use it in a type-erased context?
??x
The Small Buffer Optimization (SBO) is an optimization technique where small amounts of data are stored directly within a class instead of being managed through pointers or dynamic memory allocation. This reduces overhead, such as the cost of heap allocations and pointer dereferencing.

In a type-erased context, you might want to use SBO if the derived objects have small enough buffers that can be stored inline with minimal performance penalty. For instance:

```cpp
class SmallBufferBase {
    char buffer[1024]; // Fixed-size buffer

public:
    virtual void func() { /* implementation */ }
};

// Usage in TypeEraser
template<typename Base>
class TypeErasedWithSBO : public Base {
private:
    using Base::func; // Inherit base class methods

public:
    // Use SBO to store small buffers inline
};
```

Here, `SmallBufferBase` includes a fixed-size buffer. The `TypeErasedWithSBO` class inherits from it and uses the SBO to store data efficiently.
??x
---

---
#### Setup Costs of Owning Type Erasure Wrappers
This section investigates the setup costs associated with owning type-erasure wrappers, including the trade-offs between value semantics and performance. It highlights that there is a cost in maintaining value semantics, which might be acceptable for some cases but not others.

:p What are the setup costs involved in using an owning type-erased wrapper?
??x
Using an owning type-erased wrapper involves several setup costs:
1. **Memory Allocation**: Each time you create an `std::unique_ptr`, there is a small overhead associated with memory allocation.
2. **Virtual Function Calls**: Virtual dispatch incurs additional overhead compared to direct function calls.
3. **Copy Construction and Assignment**: When copying or assigning type-erased objects, the underlying pointer must be managed.

For example:
```cpp
// Example setup cost in TypeEraser
TypeEraser<Base> eraser1(std::make_unique<Derived1>());
TypeEraser<Base> eraser2 = std::move(eraser1);

// Virtual function call overhead
eraser1.callFunc(); // This involves a virtual dispatch, which has some runtime cost.
```

These setup costs can be significant in performance-critical applications. However, they are usually acceptable in most general-purpose code where maintainability and ease of use outweigh minor performance penalties.
??x
---

