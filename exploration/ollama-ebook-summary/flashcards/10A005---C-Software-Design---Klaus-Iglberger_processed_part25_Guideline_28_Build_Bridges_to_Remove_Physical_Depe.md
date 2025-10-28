# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 25)

**Starting Chapter:** Guideline 28 Build Bridges to Remove Physical Dependencies

---

#### Bridge Design Pattern
Background context explaining the concept. The Bridge design pattern decouples an abstraction from its implementation so that the two can vary independently. This is achieved by separating an object's interface from its implementation.

In practice, it allows you to change the underlying implementation of a class without changing any clients of this class. It is useful when an application needs to support multiple implementations and should be easily extensible in the future.

If applicable, add code examples with explanations:
:p What is the Bridge design pattern?
??x
The Bridge design pattern decouples an object's interface from its implementation so that both can vary independently. This separation allows for better manageability and flexibility, especially when dealing with complex objects or systems where the interface and behavior need to be separated.

For example, consider a GUI application where you have different types of widgets (e.g., buttons, text boxes) that might support multiple platforms (Windows, Mac, Linux).

```java
// Pseudocode for Bridge Pattern
interface Widget {
    void paint();
}

class WindowsButton implements Widget {
    public void paint() {
        System.out.println("Painting a button on Windows platform.");
    }
}

class MacButton implements Widget {
    public void paint() {
        System.out.println("Painting a button on Mac platform.");
    }
}
```

x??

---

#### Pimpl Idiom (Pimpl)
Background context explaining the concept. The Pimpl idiom, also known as "pointer-to-implementor" or just "pimpl," is an implementation detail that encapsulates the data and behavior of a class. It aims to hide the internal state and implementation details from clients while still providing a clean public interface.

:p What is the Pimpl idiom?
??x
The Pimpl idiom is an implementation technique used in software development where the body of a class's member functions are hidden behind pointers or references, typically stored inside a separate "pimpl" (pointer to implementation) class. This approach encapsulates the internal data and behavior from public access, making it easier to change the implementation without affecting the interface.

Example:
```java
// Pseudocode for Pimpl Idiom
class Button {
    private Impl* pimpl;

    public:
        Button() : pimpl(new Impl()) {}
        ~Button() { delete pimpl; }
        
        void paint();
}

class Button::Impl {
    void paint() {
        // Implementation details here.
    }
}
```

x??

---

#### Prototype Design Pattern
Background context explaining the concept. The Prototype design pattern provides a way to create new objects by copying (cloning) existing objects. It is particularly useful when creating complex objects with many attributes and behaviors, as it avoids the complexity of manually constructing each object.

:p What is the Prototype design pattern?
??x
The Prototype design pattern allows you to copy an existing object to produce a new object while avoiding expensive construction processes or duplicating code. This is achieved by cloning (copying) an instance rather than creating a new one from scratch, making it particularly useful for complex objects.

Example:
```java
// Pseudocode for Prototype Pattern
class Button {
    private String text;
    
    public void setText(String text) { this.text = text; }
    public String getText() { return text; }

    // Clone method to create a copy of the object.
    public Button clone() {
        try {
            return (Button) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();  // Should never happen
        }
    }
}

// Usage
public class PrototypePatternExample {
    public static void main(String[] args) {
        Button button = new Button();
        button.setText("OK");
        
        Button clonedButton = button.clone();
        System.out.println(clonedButton.getText());  // Output: OK
    }
}
```

x??

---

#### External Polymorphism
Background context explaining the concept. External Polymorphism, also known as "Virtual Function Polymorphism" in C++, is a mechanism where the behavior of an object can be extended or modified without changing its interface. This is achieved through the use of virtual functions and late binding (dynamic dispatch).

In Java, this concept is often referred to as method overriding, which allows subclasses to provide their own implementation of methods inherited from superclasses.

:p What is External Polymorphism?
??x
External Polymorphism refers to a mechanism in object-oriented programming where the behavior of an object can be extended or modified dynamically at runtime. This is typically achieved through the use of virtual functions (in C++) or method overriding (in Java), which allows for late binding and polymorphic behavior.

Example in Java:
```java
// Example class structure using External Polymorphism
class Shape {
    public void draw() { System.out.println("Drawing a default shape."); }
}

class Circle extends Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a circle.");
    }
}

public class ExternalPolymorphismExample {
    public static void main(String[] args) {
        Shape shape = new Circle();
        shape.draw();  // Output: Drawing a circle.
        
        Shape anotherShape = new Shape();
        anotherShape.draw();  // Output: Drawing a default shape.
    }
}
```

x??

---

#### Bridge Design Pattern
Background context: The term "bridge" suggests connecting two things to bring them closer together. However, in software design, bridges are used to reduce physical dependencies and decouple pieces of functionality that need to work together but shouldn't know too many details about each other.

The given example shows an `ElectricCar` class that is tightly coupled with the `ElectricEngine` implementation, leading to several problems:
- Physical coupling: Including headers results in transitive dependencies.
- Visibility issues: All implementation details are visible to anyone who sees the ElectricCar class definition.

To solve these issues, a better approach would be to use an abstract class as an abstraction layer.
:p What is the main issue with the original implementation of `ElectricCar`?
??x
The main issue is that the `ElectricCar` class has direct knowledge and dependency on the concrete `ElectricEngine` implementation. This leads to physical coupling, where any change in the `ElectricEngine` header would affect the `ElectricCar` and potentially many more classes.
```cpp
// ElectricCar.h with ElectricEngine as a data member
class ElectricCar {
public:
    ElectricCar(/*maybe some engine arguments*/);
    void drive();
private:
    ElectricEngine engine_;  // Direct dependency on ElectricEngine
    // ... more car-specific data members (wheels, drivetrain, ...)
};
```
x??

---

#### Forward Declaration for Decoupling
Background context: The original implementation of `ElectricCar` was tightly coupled with the `ElectricEngine`. A better approach is to use a forward declaration and store a pointer to the engine. This reduces physical dependencies but still reveals implementation details.
:p How can you reduce physical coupling in `ElectricCar` without revealing the implementation details?
??x
You can reduce physical coupling by storing a pointer to `ElectricEngine` instead of using it as a data member directly. While this moves the header inclusion into the source file, it doesn't address the visibility issue where anyone still sees that `ElectricCar` depends on `ElectricEngine`.
```cpp
// ElectricCar.h with forward declaration and unique_ptr
#include <memory>
struct ElectricEngine;  // Forward declaration

class ElectricCar {
public:
    ElectricCar(/*maybe some engine arguments*/);
    void drive();
private:
    std::unique_ptr<ElectricEngine> engine_;  // Pointer to the engine
    // ... more car-specific data members (wheels, drivetrain, ...)
};
```
x??

---

#### Introducing an Abstract Class for Decoupling and Encapsulation
Background context: The forward declaration approach reduces physical dependencies but still reveals implementation details. To fully decouple and hide these details, introduce an abstract class `Engine`.
:p How can you further reduce the visibility of implementation details in `ElectricCar`?
??x
To further reduce the visibility of implementation details, introduce an abstract base class `Engine`. This allows `ElectricCar` to depend on a more general interface rather than concrete implementations. Changes to the engine will affect only classes that use this abstract interface.
```cpp
// Engine.h with abstract class definition
class Engine {
public:
    virtual ~Engine() = default;
    virtual void start() = 0;
    virtual void stop() = 0;  // More engine-specific functions can be added here
private:
    // ... (not visible to ElectricCar)
};
```
```cpp
// ElectricCar.h with Engine as a data member
#include <Engine.h>
#include <memory>

class ElectricCar {
public:
    void drive();
private:
    std::unique_ptr<Engine> engine_;  // Dependency on the abstract Engine interface
    // ... more car-specific data members (wheels, drivetrain, ...)
};
```
```cpp
// ElectricEngine.h with derived class definition
#include <Engine.h>

class ElectricEngine : public Engine {
public:
    void start() override;
    void stop() override;
private:
    // ... implementation details hidden from ElectricCar
};
```
x??

---

#### Bridge Design Pattern Overview
The Bridge design pattern is a classic GoF (Gang of Four) design pattern introduced in 1994. Its primary goal is to minimize physical dependencies by encapsulating implementation details behind an abstraction, enabling easy change and separation of concerns.

:p What is the main intent of the Bridge design pattern?
??x
The main intent of the Bridge design pattern is "Decouple an abstraction from its implementation so that the two can vary independently."
x??

---
#### Abstraction vs Implementation in ElectricCar and Engine Classes
In our example, `ElectricCar` represents the abstraction while `Engine` serves as the implementation. The purpose here is to ensure both components can change independently without affecting each other.

:p How do we illustrate the separation between abstraction and implementation using the Bridge pattern?
??x
We use a class hierarchy where `ElectricCar` (the abstraction) depends on an interface or base class `Engine` (the implementation). This way, changes in `ElectricCar` will not affect `Engine`, and vice versa.
x??

---
#### Car Base Class Implementation
The `Car` base class acts as the core of our Bridge design. It encapsulates the Bridge to associated engines via a pointer-to-implementation (`pimpl_`).

:p What is the role of the `Car` class in the Bridge pattern?
??x
The `Car` class serves as the abstraction layer that decouples the car's functionality from its engine implementation, allowing both to vary independently.
x??

---
#### Protected Constructor and pimpl Technique
In the `Car` class, a protected constructor is used to initialize with an engine. This technique ensures derived classes can specify the engine type.

:p Why is the `Car` constructor marked as protected?
??x
The `Car` constructor is marked as protected because it should only be called by derived classes (e.g., `ElectricCar`). This restricts how engines are set, maintaining encapsulation.
x??

---
#### ElectricCar Implementation Details
The `ElectricCar` class inherits from the `Car` base class and uses a unique pointer to initialize its engine.

:p How does the `ElectricCar` constructor relate to the `Car` base class?
??x
The `ElectricCar` constructor initializes the `Car` base class with an engine using `std::make_unique`, demonstrating how derived classes can specify engine types.
x??

---
#### Pimpl Idiom and Pointer-to-Implementation (pimpl)
The `pimpl_` pointer in the `Car` class is a common technique to hide implementation details, making the code more maintainable.

:p What does the term "pimpl" stand for?
??x
"Pimpl" stands for "pointer to implementation," which is used to encapsulate implementation details within a derived class, hiding them from the public interface.
x??

---
#### Bridge Pattern in Multiple Car Types
To reduce duplication and adhere to the DRY principle, we can generalize the Bridge pattern by introducing a `Car` base class.

:p How does generalizing the Bridge pattern with a `Car` class help?
??x
Generalizing the Bridge pattern with a `Car` base class allows us to apply similar decoupling strategies across multiple car types (e.g., electric, combustion), reducing code duplication and improving maintainability.
x??

---
#### Getters for Engine Access
The `getEngine()` member functions in the `Car` class provide controlled access to the engine implementation.

:p Why are the `getEngine()` methods declared as protected?
??x
The `getEngine()` methods are protected because they allow derived classes to access the engine, but not expose this functionality publicly. This maintains encapsulation and control over how engines are accessed.
x??

---

