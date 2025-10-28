# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 30)

**Starting Chapter:** The History of Type Erasure

---

#### Type Erasure History and Importance
This concept highlights the historical context of type erasure, emphasizing its importance in modern C++ programming. The technique was first discussed by Kevlin Henney in 2000 but gained popularity through Sean Parent's presentation at GoingNative 2013.
:p What is the history and significance of Type Erasure as described in the text?
??x
Type Erasure was initially discussed by Kevlin Henney in a C++ Report article published in July-August 2000. It gained broader recognition through Sean Parent's presentation at GoingNative 2013, where he introduced it as a solution to inheritance problems. The technique is significant because it allows for value semantics while handling polymorphism, aligning with the principle of preferring non-intrusive solutions and minimizing dependencies.
x??

---
#### Type Erasure in Boost Libraries
Type Erasure has been used in various libraries like Boost, where it was implemented by Douglas Gregor to enhance functionality. It's particularly notable as an example of type erasure in practice.
:p How does Type Erasure manifest in the Boost libraries?
??x
In the Boost libraries, especially with `boost::function`, type erasure is employed to allow functions and function objects to be treated polymorphically without reference semantics. This means that different types can be stored in a single container, abstracting away their specific types.
```cpp
#include <functional>
#include <vector>

int main() {
    std::vector<std::function<int(int)>> vec;
    vec.push_back([](int x) { return x + 1; }); // Lambda function
    vec.push_back(&std::cout <<);                // Function pointer

    for (const auto& func : vec) {
        std::cout << func(5) << "\n"; // Polymorphic usage
    }
}
```
x??

---
#### Type Erasure in C++17 and Beyond
`std::any`, `std::shared_ptr`, and other standard library components utilize type erasure to provide value semantics for various applications, enhancing flexibility without losing the benefits of polymorphism.
:p Which modern C++ types use Type Erasure?
??x
Types like `std::any` from C++17 and `std::shared_ptr` use type erasure. For instance, `std::any` allows storing values of any type in a single variable:
```cpp
#include <any>
#include <cstdlib>

int main() {
    std::any a;
    a = 42; // Stores an int
    a = "hello"; // Replaces the int with a string

    if (std::holds_alternative<std::string>(a)) {
        const auto& str = std::get<std::string>(a);
        std::cout << str << "\n";
    }
}
```
`std::shared_ptr`, on the other hand, uses type erasure to store a deleter that is not part of its type:
```cpp
#include <memory>

int main() {
    std::shared_ptr<int> ptr = std::make_shared<int>(42);
    // Custom deleter can be used with std::shared_ptr as well.
}
```
x??

---
#### Benefits and Implementation of Type Erasure
Type erasure offers benefits such as value semantics, loose coupling, and non-intrusive solutions. It involves encapsulating polymorphic behavior in a way that avoids the pitfalls of traditional inheritance hierarchies.
:p What are some key benefits of using Type Erasure?
??x
Key benefits include:
- **Value Semantics**: Allows for more flexible and robust data handling without the need for complex reference semantics.
- **Loose Coupling**: Enhances modularity by separating interface from implementation details.
- **Non-Intrusive Solutions**: Avoids forced modifications to existing code, promoting cleaner design.

Type erasure typically involves creating a wrapper class that encapsulates an object of any type and provides a common interface for interaction. This approach minimizes dependencies between components, making the system more maintainable.
x??

---
#### Practical Example: std::any
`std::any` is used to store values of various types dynamically without knowing their exact type at compile time, providing a flexible yet safe polymorphic solution.
:p Provide an example using `std::any`.
??x
Example:
```cpp
#include <any>
#include <cstdlib>

int main() {
    std::any value;
    value = 42; // Store an integer
    value = "Hello"; // Store a string

    if (std::holds_alternative<std::string>(value)) { // Check and cast to string
        const auto& str = std::get<std::string>(value);
        std::cout << str;
    }
}
```
In this example, `std::any` can hold different types of values. The `std::holds_alternative` function is used to check the type of value stored, and `std::get` retrieves it.
x??

---
#### Practical Example: std::shared_ptr
`std::shared_ptr` uses type erasure internally by storing a deleter that manages the lifetime of the object pointed to. This allows for flexible memory management without embedding the deleter in the type system.
:p Provide an example using `std::shared_ptr`.
??x
Example:
```cpp
#include <memory>

int main() {
    // Custom deleter
    auto ptr = std::make_shared<int>(42, [](int* p) { delete p; });

    // Default deleters (uses default_delete<T>)
    auto default_ptr = std::shared_ptr<int>(new int(10));

    // The deleter is stored internally and not part of the type
}
```
In this example, `std::shared_ptr` can be customized with a custom deleter or use the default one. The deleter is stored separately from the managed pointer, demonstrating the power of type erasure in managing resources.
x??

---

#### Type Erasure: Overview and Intent
Background context explaining the concept. Type Erasure is a compound design pattern that combines External Polymorphism, Bridge, and optionally Prototype to create value-based abstractions for unrelated types with similar behavior.

:p What is the intent of Type Erasure?
??x
The intent of Type Erasure is to provide a value-based, non-intrusive abstraction for an extendable set of unrelated, potentially non-polymorphic types that exhibit the same semantic behavior. This means creating wrapper types that can represent multiple underlying types without requiring modifications to those types.
```cpp
// Example of Type Erasure with a simple interface and concrete implementation
class Shape {
public:
    virtual void draw() = 0;
};

class Circle : public Shape {
public:
    void draw() override { /* Draw circle */ }
};

class Square : public Shape {
public:
    void draw() override { /* Draw square */ }
};
```
x??

---

#### External Polymorphism
Background context explaining the concept. External Polymorphism is a key component of Type Erasure, enabling decoupling and non-intrusive runtime polymorphism.

:p What is the role of External Polymorphism in Type Erasure?
??x
External Polymorphism plays a crucial role in Type Erasure by providing a way to achieve decoupling and non-intrusiveness. It allows for an external, non-intrusive abstraction that supports various unrelated types with similar behavior without requiring modifications to those types.
```cpp
// Example of External Polymorphism using a Shape hierarchy
class Shape {
public:
    virtual void draw() = 0;
};

void processShape(const std::shared_ptr<Shape>& shape) {
    // Process the shape without knowing its exact type
}
```
x??

---

#### Bridge Design Pattern
Background context explaining the concept. The Bridge pattern is another key component in Type Erasure, enabling value semantics-based implementations.

:p How does the Bridge design pattern contribute to Type Erasure?
??x
The Bridge design pattern contributes to Type Erasure by separating an object's interface from its implementation. This separation allows for a value semantics-based implementation where the abstraction (e.g., `Shape`) is separated from the behavior it provides, enabling non-intrusive and value-based abstractions.
```cpp
// Example of using Bridge with Shape and Color in Type Erasure
class Shape {
public:
    virtual void draw() = 0;
};

class Color {
public:
    virtual void applyColor(Shape& shape) = 0;
};

class RedColor : public Color {
public:
    void applyColor(Shape& shape) override { /* Apply red color to the shape */ }
};
```
x??

---

#### Prototype Design Pattern
Background context explaining the concept. The Prototype pattern, if used, is optional and helps manage copy semantics in value-based abstractions.

:p When would you use the Prototype design pattern in Type Erasure?
??x
The Prototype design pattern can be used in Type Erasure when dealing with the copy semantics of value-based abstractions. It allows for creating copies of objects without requiring knowledge of their internal structure, making it easier to handle complex or non-trivial object copying.
```cpp
// Example using Prototype with Shape and Color in Type Erasure
class Shape {
public:
    virtual void draw() = 0;
};

class Color : public Prototype<Color> {
public:
    virtual void applyColor(Shape& shape) = 0;
};
```
x??

---

#### OwningShapeModel Class Template
Background context explaining the concept. The OwningShapeModel class template is a wrapper that encapsulates Type Erasure, providing value semantics and non-intrusive behavior.

:p What is the purpose of the OwningShapeModel class template?
??x
The OwningShapeModel class template serves as a wrapper around external hierarchies introduced by External Polymorphism. It provides value semantics and non-intrusive behavior by encapsulating types that may have different internal structures while maintaining consistent semantic behavior.
```cpp
// Example of OwningShapeModel in Type Erasure
template <typename T>
class OwningShapeModel : public Shape {
private:
    std::unique_ptr<T> ptr;

public:
    void draw() override {
        // Delegate drawing to the wrapped type
        if (ptr) {
            ptr->draw();
        }
    }

    // Other methods for managing the internal pointer
};
```
x??

---

#### Circle and Square Classes
Background context: The provided text introduces two simple classes, `Circle` and `Square`, which are unrelated and nonpolymorphic. These classes do not share a common base class or introduce any virtual functions.

:p What are the Circle and Square classes in this context?
??x
The `Circle` and `Square` classes represent geometric shapes with specific properties (radius for circle, side for square) but do not inherit from a common base class or introduce any polymorphic behavior. They have constructors to initialize their respective properties and getters to retrieve them.
```cpp
//---- <Circle.h> ----------------
class Circle {
public:
    explicit Circle(double radius)
        : radius_(radius) {}

    double radius() const { return radius_; }

private:
    double radius_;
};

//---- <Square.h> ----------------
class Square {
public:
    explicit Square(double side)
        : side_(side) {}

    double side() const { return side_; }

private:
    double side_;
};
```
x??

---

#### ShapeConcept and OwningShapeModel Classes
Background context: The `ShapeConcept` class is an abstract base class with pure virtual functions for drawing a shape and cloning the object. The `OwningShapeModel` template class inherits from `ShapeConcept`, holds a pointer to a specific shape, and uses a draw strategy.

:p What are the roles of `ShapeConcept` and `OwningShapeModel`?
??x
The `ShapeConcept` class acts as an abstract base class for shapes, defining pure virtual functions such as `draw()` and `clone()`. The `OwningShapeModel` template class implements these methods by holding a specific shape instance and using a draw strategy.

```cpp
//---- <Shape.h> ----------------
namespace detail {
class ShapeConcept {
public:
    virtual ~ShapeConcept() = default;
    virtual void draw() const = 0;
    virtual std::unique_ptr<ShapeConcept> clone() const = 0;

template<typename ShapeT, typename DrawStrategy>
class OwningShapeModel : public ShapeConcept {
public:
    explicit OwningShapeModel(ShapeT shape, DrawStrategy drawer)
        : shape_{std::move(shape)}, drawer_{std::move(drawer)} {}

    void draw() const override { drawer_(shape_); }

    std::unique_ptr<ShapeConcept> clone() const override {
        return std::make_unique<OwningShapeModel>(*this);
    }

private:
    ShapeT shape_;
    DrawStrategy drawer_;
};
}
```
x??

---

#### Namespace and Class Naming Changes
Background context: The `Circle`, `Square`, `ShapeConcept`, and `OwningShapeModel` classes have been moved to the `detail` namespace. This change suggests that these are implementation details intended for internal use rather than direct exposure.

:p Why were the class names changed, and why did you move them to a `detail` namespace?
??x
The class names were changed and moved to the `detail` namespace to indicate that they are part of an implementation detail. This change suggests that these classes should not be used directly by external code but are rather internal components serving specific purposes in the design.

Moving these classes to the `detail` namespace helps encapsulate their implementation, making them less accessible and more suitable for internal use without exposing them to external users.
x??

---

#### Prototype Design Pattern
Background context: The text discusses the introduction of a `clone()` function alongside an existing `draw()` function within the `ShapeConcept` class. This combination strongly suggests the use of the Prototype design pattern, where objects can be copied using a prototype.

:p What is the purpose of adding a `clone()` function in the `ShapeConcept` class?
??x
The primary purpose of adding a `clone()` function is to enable copying behavior for shapes, which aligns with the Prototype design pattern. This allows creating new instances based on existing objects without having to know their concrete classes.

```cpp
class ShapeConcept {
public:
    virtual void draw() const = 0;
    virtual std::unique_ptr<ShapeConcept> clone() const = 0; // Prototype pattern implementation
};
```
x??

---

#### Requirement for Draw and Clone Functions in `ShapeT` Types
Background context: The text mentions that while the names of functions like `draw()` and `clone()` are chosen by the implementation, the actual requirements on `ShapeT` types are defined by their implementations. This means that shapes must be drawable (`draw()`) and copyable (`clone()`).

:p What does the requirement for `ShapeT` types consist of?
??x
The requirement for `ShapeT` types consists of implementing both a `draw()` function to represent drawing behavior and a `clone()` function to enable copying. These functions ensure that any shape can be drawn and duplicated, which are essential behaviors for shapes in this context.

```cpp
class ShapeT {
public:
    virtual void draw() const = 0; // Must provide drawing behavior
    virtual std::unique_ptr<ShapeT> clone() const = 0; // Must provide copying functionality
};
```
x??

---

#### OwningShapeModel Class Implementation
Background context: The text describes `OwningShapeModel` as an implementation of the `ShapeConcept` class, which takes a concrete shape type and a drawing strategy. It implements `draw()` using the given drawing strategy and `clone()` to return an exact copy.

:p How does `OwningShapeModel` implement its `draw()` function?
??x
`OwningShapeModel` implements its `draw()` function by applying the provided drawing strategy to render the shape. This ensures that shapes can be drawn according to their specific behavior defined in the drawing strategy.

```cpp
class OwningShapeModel : public ShapeConcept {
private:
    std::unique_ptr<ConcreteShape> shape;
    DrawingStrategy* drawingStrategy;

public:
    OwningShapeModel(std::unique_ptr<ConcreteShape> shape, DrawingStrategy* drawingStrategy)
        : shape(std::move(shape)), drawingStrategy(drawingStrategy) {}

    void draw() const override {
        // Apply the drawing strategy to render the shape
        drawingStrategy->draw(*shape);
    }

    std::unique_ptr<ShapeConcept> clone() const override {
        return std::make_unique<OwningShapeModel>(std::move(shape), drawingStrategy);
    }
};
```
x??

---

#### Transition from External Polymorphism to Type Erasure
Background context: The text notes that while the current implementation is similar to external polymorphism, it's one step away from type erasure. This means moving from reference semantics to value semantics by wrapping `OwningShapeModel` in a value type.

:p What does transitioning from external polymorphism to type erasure entail?
??x
Transitioning from external polymorphism to type erasure involves changing the representation of shapes from using pointers or references (reference semantics) to using objects directly (value semantics). This is achieved by creating a wrapper around `OwningShapeModel` that handles instantiation, pointer management, and lifetime details automatically.

```cpp
class ShapeErasure {
private:
    std::unique_ptr<ShapeConcept> shape;

public:
    // Constructor: Instantiates OwningShapeModel with the given parameters
    ShapeErasure(std::unique_ptr<ConcreteShape> shape, DrawingStrategy* drawingStrategy)
        : shape(std::make_unique<OwningShapeModel>(std::move(shape), drawingStrategy)) {}

    // Draw method: Calls draw on the internal object
    void draw() const {
        shape->draw();
    }

    // Clone method: Returns a copy of this object, handling deep copying internally
    ShapeErasure clone() const {
        return *this; // In practice, would call shape->clone()
    }
};
```
x??

---

#### Bridge Design Pattern Application

This pattern is used to separate an object’s interface from its implementation so that the two can vary independently. The `Shape` class uses this approach by storing a polymorphic pointer to its concrete implementation details, allowing the types of shapes and drawing strategies to be decoupled.

:p What does the `Shape` class do in terms of type management?
??x
The `Shape` class manages the storage of a polymorphic object (using `std::unique_ptr`) that represents the actual shape. This separation ensures that the interface remains stable while different implementations can vary independently.
x??

---

#### Type Erasure

Type erasure occurs when a concrete implementation is stored behind an abstract base class pointer, effectively hiding the type from the user of the class.

:p How does the `Shape` class achieve type erasure?
??x
The `Shape` class uses a `std::unique_ptr<detail::ShapeConcept>` to store the actual shape. The `detail::OwningShapeModel` is instantiated with types `ShapeT` and `DrawStrategy`, but it is stored as an instance of `detail::ShapeConcept`. This means that outside the class, no concrete type information about `shapeT` or `drawStrategy` is visible.
x??

---

#### Clone Function for Copy Constructor

To implement the copy constructor in a way that works with polymorphic types, the `clone()` function must be available to create an exact copy of the object.

:p How does the `Shape` class handle the copy constructor?
??x
The `Shape` class uses the `clone()` method provided by the abstract base class (`detail::ShapeConcept`) to make a deep copy of the shape. This allows it to create a new `Shape` instance with an identical internal state without knowing the concrete type.

```cpp
Shape(const Shape& other) : pimpl_(other.pimpl_->clone()) {}
```

This line creates a new `pimpl_` using the `clone()` method from the current object's implementation.
x??

---

#### Copy and Swap Idiom

The copy and swap idiom is used to implement the assignment operator by swapping the state of the current object with that of a temporary copy.

:p How does the assignment operator in `Shape` work?
??x
The `Shape` class implements the assignment operator using the copy and swap idiom. It first creates a temporary copy, then swaps its internal pointer with that of the copied object. This ensures safe copying and swapping without undefined behavior.

```cpp
Shape& operator=(const Shape& other) {
    Shape copy(other);
    pimpl_.swap(copy.pimpl_);
    return *this;
}
```

This implementation first creates a `copy` of the current state, then swaps its internal pointer with that of the copied object. Finally, it returns itself to allow chained assignments.
x??

---

#### Destructor and Move Operations

The destructor is defaulted by `Shape`, while move operations are default because they can reuse the copy constructor and assignment operator.

:p What special handling does `Shape` need for its destructor and move operations?
??x
The destructor of `Shape` is declared as `default`. This means that it will use the default destructor implementation provided by the compiler. Similarly, both the move constructor and move assignment operator are defaulted. These are useful because they can rely on the copy constructor and assignment operator to handle the actual work.

```cpp
~Shape() = default;
Shape(Shape&&) = default;
Shape& operator=(Shape&&) = default;
```

These declarations ensure that default implementations are used, which in this case will delegate the work to the copy constructor and assignment operator.
x??

---

#### Copy Assignment Operator and Copy-and-Swap Idiom
The context is about implementing efficient and convenient copy assignment operators using the Copy-and-Swap idiom. This approach ensures that deep copies are made without direct pointer manipulation, reducing bugs and making the code more robust.

:p What is the primary advantage of using the Copy-and-Swap idiom for implementing a copy assignment operator?
??x
The primary advantage is ensuring that the object being assigned to is in a valid state after the operation. This approach swaps the contents with another temporary object, guaranteeing that no self-assignment bugs occur and deep copies are performed.

```cpp
class Shape {
public:
    // Copy constructor and copy assignment operator implementation using Copy-and-Swap idiom
    Shape(const Shape& other) : pimpl_(new Impl(*other.pimpl_)) {}
    Shape& operator=(Shape other) {  // Note: using the 'other' parameter for swap
        if (this != &other) {
            swap(*this, other);
            delete pimpl_;  // Clean up old data member
            pimpl_ = new Impl(std::move(other.pimpl_));
        }
        return *this;
    }

private:
    void swap(Shape& lhs, Shape& rhs) { std::swap(lhs.pimpl_, rhs.pimpl_); }  // Implementation of the swap function

    std::unique_ptr<Impl> pimpl_;  // Pointer to implementation details
};
```
x??

---

#### Hidden Friend Function for Drawing
The text introduces a hidden friend function called `draw()` within the `Shape` class. This function provides a convenient way to draw shapes without needing to expose complex logic.

:p What is the purpose of defining the `draw()` function as a hidden friend in the `Shape` class?
??x
The purpose is to encapsulate drawing behavior tightly with the `Shape` class while still providing external access to this functionality. This approach maintains internal encapsulation and leverages friendship for deeper integration without exposing implementation details.

```cpp
class Shape {
friend void draw(const Shape& shape);  // Declaration of the hidden friend function

// Implementation within the class body
private:
    void drawImpl() const;  // Private drawing method
};

void draw(const Shape& shape) {
    shape.drawImpl();  // Calls the private implementation
}
```
x??

---

#### Use of `Shape` Class for Abstraction
The context is about creating a generic abstraction (`Shape`) that can handle various concrete shapes while providing a consistent interface for drawing.

:p How does the `Shape` class simplify the process of handling different shape types?
??x
By using templates and type erasure, the `Shape` class abstracts away the complexities of different shape types. This allows users to treat any shape uniformly through the same interface without needing to manage pointers or worry about deep copies.

```cpp
template <typename T, typename F>
class Shape {
public:
    Shape(T obj, F drawer) : object_(std::move(obj)), drawFunc_(std::move(drawer)) {}

    void draw() const { drawFunc_(*this); }  // Delegate drawing to the provided function

private:
    T object_;  // The shape object
    F drawFunc_;  // Drawing function (lambda or functor)
};

int main() {
    Circle circle{3.14};
    auto drawer = [](const Circle& c) { /*...*/ };
    Shape<Circle, decltype(drawer)> shape(circle, drawer);
    shape.draw();
}
```
x??

---

#### Lambda and Type Erasure
The context is about using lambdas to bind drawing functionality with shapes in a type-erased manner.

:p How does the use of lambda functions simplify the creation of `Shape` instances?
??x
Using lambdas simplifies creating `Shape` instances by allowing you to encapsulate drawing logic within the same function where the shape object is defined. This reduces boilerplate code and makes the construction process more concise.

```cpp
int main() {
    Circle circle{3.14};
    auto drawingCircle = [] (const Circle& c) { /*...*/ };
    Shape<circle, decltype(drawingCircle)> shape(circle, drawingCircle);
    draw(shape);  // Assuming 'draw' is a function to invoke the drawing functionality
}
```
x??

---

