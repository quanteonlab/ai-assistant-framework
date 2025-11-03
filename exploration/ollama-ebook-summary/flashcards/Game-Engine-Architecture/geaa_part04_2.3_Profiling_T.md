
#### Polymorphism

Background context explaining polymorphism. Polymorphism is a language feature that allows objects of different types to be manipulated through a common interface. This makes collections of diverse objects appear homogeneous, simplifying code that operates on these objects.

If applicable, add code examples with explanations.
:p What is the problem with using a switch statement for drawing shapes in `drawShapes` function?
??x
The main issue with using a switch statement to draw different types of shapes (e.g., Circle, Rectangle, Triangle) is that it requires the function to have knowledge about all possible shape types. This makes it difficult to add new shape types in the future because every place where the `Shape` type is used would need updating.

For example:
```cpp
void drawShapes(std::list<Shape*>& shapes) {
    std::list<Shape*>::iterator pShape = shapes.begin();
    std::list<Shape*>::iterator pEnd = shapes.end();
    for ( ; pShape != pEnd; ++pShape) {
        switch ((*pShape)->mType) {
            case CIRCLE: // draw shape as a circle
                break;
            case RECTANGLE: // draw shape as a rectangle
                break;
            case TRIANGLE: // draw shape as a triangle
                break;
            //...
        }
    }
}
```
This approach is inflexible and can be error-prone, especially if new shapes are added frequently.
x??

---

#### Virtual Functions in Polymorphism

Background context explaining virtual functions. A virtual function is used to achieve polymorphism in C++. When a base class has a virtual function, derived classes can override this function to provide specific implementations.

:p How does the `Shape` class enable polymorphic behavior for drawing shapes?
??x
The `Shape` class enables polymorphism by declaring a pure virtual function `Draw()`, which is implemented differently in derived classes. This allows different types of `Shape` objects to be drawn using the same method call, without needing to know their specific type.

Here's an example implementation:
```cpp
struct Shape {
    virtual void Draw() = 0; // pure virtual function
    virtual ~Shape() {}      // ensure derived destructors are virtual
};

struct Circle : public Shape {
    virtual void Draw() override {
        // draw shape as a circle
    }
};

struct Rectangle : public Shape {
    virtual void Draw() override {
        // draw shape as a rectangle
    }
};

struct Triangle : public Shape {
    virtual void Draw() override {
        // draw shape as a triangle
    }
};
```
The `drawShapes` function can now call the appropriate `Draw()` method for each object in the list:
```cpp
void drawShapes(std::list<Shape*>& shapes) {
    std::list<Shape*>::iterator pShape = shapes.begin();
    std::list<Shape*>::iterator pEnd = shapes.end();
    for ( ; pShape != pEnd; ++pShape) {
        (*pShape)->Draw(); // call virtual function
    }
}
```
This approach is more flexible and easier to maintain than using a switch statement.
x??

---

#### Composition

Background context explaining composition. Composition involves using smaller, interacting objects to accomplish a higher-level task. This pattern allows for greater flexibility and reusability of code.

:p What is the difference between composition and inheritance?
??x
Composition differs from inheritance in that it allows you to combine multiple classes to form new behavior without creating a hierarchical relationship. In contrast, inheritance involves inheriting methods and properties from one class into another, which can lead to tight coupling and complex hierarchies.

For example:
- **Inheritance:** `DerivedClass` inherits from `BaseClass`, gaining all its methods.
- **Composition:** A `Car` object has a `Engine` object and a `Wheel` object. The `Engine` and `Wheel` classes can be reused in other contexts, not just within the `Car` class.

This makes composition more flexible because you can change or replace parts of an object without affecting its overall structure.
x??

---

#### Composition vs Aggregation
Explanation of composition and aggregation, highlighting their differences and benefits. These relationships are "has-a" or "uses-a" rather than "is-a". They result in simpler and more focused classes, making them easier to test, debug, and reuse.

:p What is the difference between composition and aggregation?
??x
Composition creates a strong relationship where one class owns another and controls its lifecycle. Aggregation represents a weaker relationship with less control over the owned object's lifecycle. Both make classes simpler by focusing on specific responsibilities.
For example:
```java
// Composition: Engine has FuelTank
class Engine {
    private FuelTank fuelTank;

    public void start() {
        if (fuelTank.getFuelLevel() > 0) {
            // Start engine logic
        }
    }
}

// Aggregation: Window uses Rectangle for its shape
class Window {
    private Rectangle rectangle;

    public Window(Rectangle rectangle) {
        this.rectangle = rectangle;
    }

    public void draw() {
        rectangle.draw();
    }
}
```
x??

---

#### Design Patterns Overview
Explanation of what a design pattern is, including examples like Singleton and Iterator. These patterns are common solutions to recurring problems in software design.

:p What is a design pattern?
??x
A design pattern is a template for solving a specific problem that arises over and over again. Examples include:
- **Singleton**: Ensures one instance of a class.
- **Iterator**: Provides an efficient way to traverse collection elements without exposing their underlying implementation details.
```java
// Example Singleton Pattern
public class Singleton {
    private static Singleton instance;

    // Private constructor prevents instantiation from other classes
    private Singleton() {}

    public static synchronized Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }

    // Other methods can be added here
}
```
x??

---

#### Resource Acquisition Is Initialization (RAII)
Explanation of the RAII pattern, which binds resource acquisition and release to class constructors and destructors. This prevents accidental forgetting of resource management.

:p What is the RAII pattern?
??x
The RAII pattern ensures that resources are acquired in a constructor and released in a destructor or when the object goes out of scope. It encapsulates resource handling within an object’s lifecycle, preventing memory leaks or other resource management issues.
```java
// Example Janitor Class (RAII)
class FileJanitor {
    private String filename;

    public FileJanitor(String filename) {
        this.filename = filename;
        openFile(filename);
    }

    public void useFile() {
        // Use file logic here
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            closeFile();
        } finally {
            super.finalize();
        }
    }

    private void openFile(String filename) {
        // Open and manage the file
    }

    private void closeFile() {
        // Close and release the file resources
    }
}
```
x??

---

#### RAII Pattern Explanation
Background context: The Resource Acquisition Is Initialization (RAII) pattern is a programming idiom used to manage resources such as memory or file handles. It ensures that resources are properly managed and released when they are no longer needed, even if an exception occurs.

If applicable, add code examples with explanations:
:p What is the RAII pattern?
??x
The RAII pattern uses constructors to allocate resources and destructors to release them. This pattern ensures that resources are automatically cleaned up without needing explicit deallocation calls, making the code safer and easier to maintain.
```cpp
class AllocJanitor {
public:
    explicit AllocJanitor(mem::Context context) { mem::PushAllocator(context); }
    ~AllocJanitor() { mem::PopAllocator(); }
};
```
x??

---

#### C++ Standard Evolution
Background context: The C++ programming language has undergone several standardizations over the years, with each version aiming to improve the language by refining semantics and adding new features while removing problematic aspects. Here's a summary of some key versions:
- **C++98**: Established in 1998.
- **C++03**: Introduced in 2003 to address issues identified in C++98.
- **C++11 (formerly C++0x)**: Approved on August 12, 2011.

:p What are the key versions of the C++ standard?
??x
The key versions include:
- C++98: The first official C++ standard established by ISO in 1998.
- C++03: Introduced to address issues identified in C++98.
- C++11 (formerly C++0x): Approved on August 12, 2011.

This version introduced numerous features and improvements to the language.??x
x??

---

#### RAII Example Usage
Background context: The RAII pattern can be used effectively in scenarios where memory management needs to be handled automatically without manual cleanup code.

:p How does an AllocJanitor work?
??x
An `AllocJanitor` is a helper class that pushes the allocator onto a stack when constructed and pops it off the stack when destructed. This ensures that the allocator's context is properly managed, making memory allocation and deallocation safer.
```cpp
void f() {
    // do some work...
    // allocate temp buffers from single-frame allocator
    {
        AllocJanitor janitor(mem::Context::kSingleFrame);
        U8* pByteBuffer = new U8[SIZE];
        float* pFloatBuffer = new float[SIZE];
        // use buffers...
        // (NOTE: no need to free the memory because we used a single-frame allocator)
    } // janitor pops allocator when it drops out of scope
    // do more work...
}
```
x??

---

#### C++11 Features Overview
Background context: C++11 introduced numerous features to enhance the language, making it more powerful and easier to use. Some key features include:

- **Type-safe nullptr literal**: Replaces the `NULL` macro.
- **auto and decltype keywords for type inference**.

:p What new feature did C++11 introduce?
??x
C++11 introduced a type-safe `nullptr` literal, replacing the bug-prone `NULL` macro from the C language. This makes it safer to handle null pointers in C++.
```cpp
int* ptr = nullptr; // Safe initialization of a pointer with nullptr
```
x??

---

#### Rvalue References and Move Semantics
Background context: To optimize handling temporary objects, C++11 introduced rvalue references and move semantics. These features allow the compiler to efficiently transfer ownership of resources from temporaries to other variables or functions.

:p What are rvalue references and move semantics?
??x
Rvalue references enable efficient resource management by distinguishing between lvalues (variables that can be assigned) and rvalues (temporary objects). Move semantics allow these temporary objects to be moved instead of copied, which is more efficient. This feature helps in reducing unnecessary copying.
```cpp
struct MyResource {
    MyResource(MyResource&& other) { // move constructor
        // Transfer resources from 'other' to this instance
    }
};

MyResource obj1 = ...; // Create an object with temporary resource
MyResource obj2 = std::move(obj1); // Move the resource from obj1 to obj2, leaving obj1 in a valid but unspecified state
```
x??

---

#### C++ Standard Library Improvements
Background context: With each version of the C++ standard, improvements and additions are made to the standard library. For instance, C++11 introduced support for threading, improved smart pointers, and expanded generic algorithms.

:p What improvements did C++11 make to the standard library?
??x
C++11 enhanced the standard library with features such as:
- Support for threading (concurrent programming).
- Improved smart pointer facilities.
- An expanded set of generic algorithms.
```cpp
#include <thread>
#include <future>

void f() {
    std::thread t([]{ /* do some work */ }); // Create a thread to run the lambda function
    t.join(); // Wait for the thread to complete
}
```
x??

---

#### C++ Standard Evolution and Adoption

Background context: The text discusses the evolution of C++ standards, specifically mentioning C++11, C++14, and C++17. It highlights the importance of adopting new language features conservatively due to limitations in compiler support, switching costs, and risk versus reward.

:p Which C++ standard is mentioned as recently adopted by Naughty Dog for The Last of Us Part II?
??x
C++11 was the most advanced C++ standard adopted relatively recently by Naughty Dog for The Last of Us Part II. This decision was based on the cost and risks associated with switching from an older C++98 standard used in other projects like Uncharted 4: A Thief's End and Uncharted: The Lost Legacy.
x??

---

#### Compiler Support for New Standards

Background context: The text notes that not all compilers fully support new C++ standards, highlighting the case of LLVM/Clang which partially supports C++17 but has no support for draft C++2a.

:p What is an example provided in the text about compiler support for a new C++ standard?
??x
LLVM/Clang's support for C++17 is spread across Clang versions 3.5 through 4.0, and it currently has no support for the draft C++2a standard.
x??

---

#### Cost of Switching Standards

Background context: The text emphasizes that switching between standards incurs a non-zero cost, suggesting that game studios should decide on one standard and stick with it for an extended period.

:p What is highlighted as a significant consideration when adopting new C++ standards according to the text?
??x
The text highlights the need to consider the cost of switching from one C++ standard to another. Due to this cost, game studios should choose the most advanced standard they can support and stick with it for a reasonable length of time, such as the duration of one project.
x??

---

#### Risk vs Reward

Background context: The text mentions that not all language features are equally beneficial or appropriate, using `auto` in C++11 as an example. It discusses how some features have both benefits and downsides.

:p Which C++ keyword is discussed for its benefits and potential issues?
??x
The `auto` keyword in C++11 is discussed for its convenience in writing variables and functions but also mentions that it has associated negatives.
x??

---

#### Example of `auto` Keyword

Background context: The text provides an example of the `auto` keyword, explaining how it can make code more convenient to write.

:p How does the `auto` keyword help in making C++ code more convenient?
??x
The `auto` keyword allows you to automatically deduce the type of a variable based on its initializer. This can simplify your code by reducing redundant type specifications. For example:
```cpp
// Old way
int value = 10;

// New way with auto
auto value = 10;
```
x??

---

#### Naughty Dog's Coding Standards

Background context: The text describes how Naughty Dog has different coding standards for runtime and offline tools code, emphasizing a conservative approach to adopting new language features.

:p What does the text suggest about Naughty Dog’s approach to using C++ language features?
??x
Naughty Dog tends to take a conservative approach to adopting new C++ language features into their codebase. They have separate lists of approved language features for runtime and offline tools code, reflecting different standards and risks associated with each type of project.
x??

---

#### Coding Standards: Importance and Application
Background context explaining the importance of coding standards, especially for languages like C++ which are prone to abuse. The discussion highlights Naughty Dog's approach to using such standards.
:p Why is it important to have coding standards?
??x
Coding standards help improve code readability, understandability, and maintainability. They also prevent common errors by restricting the use of potentially problematic language features. For example, in C++, where there are many possibilities for abuse, a good set of coding standards can significantly enhance the quality of the codebase.
??x

---

#### Use of 'auto' Keyword
Explanation on when to use and not to use the `auto` keyword based on Naughty Dog's rules.
:p When should the `auto` keyword be used according to Naughty Dog's guidelines?
??x
The `auto` keyword can only be used in three specific scenarios: for iterators, within template definitions where no other approach works, or when using it improves code clarity and readability significantly. In all other cases, explicit type declarations are required.
??x

---

#### Template Metaprogramming Restrictions
Explanation of why complex template metaprogramming is restricted, even with exceptions.
:p Why does Naughty Dog restrict the use of template metaprogramming in their engine code?
??x
Naughty Dog restricts complex template metaprogramming to avoid difficult-to-read and sometimes non-portable code. The goal is to make it easier for programmers to jump into debugging any part of the codebase quickly, even if they are not familiar with that specific section.
??x

---

#### Interfaces in Coding Standards
Explanation on the importance of clean and well-documented interfaces.
:p What does "interfaces are king" mean in the context of coding standards?
??x
It means that the design of public interfaces (e.g., .h files) should be kept clean, simple, minimal, and easy to understand. Well-commented interfaces enhance readability and make it easier for others to use or maintain the code.
??x

---

#### Naming Conventions
Explanation on why intuitive names are crucial in coding standards.
:p Why is it important to use intuitive names according to Naughty Dog's coding standards?
??x
Using intuitive names that directly map to the purpose of a class, function, or variable helps in reducing confusion and making the code easier to understand. This practice requires spending time to identify good names upfront.
??x

---

#### Summary of Coding Standards
Summary of key points discussed about coding conventions and their importance.
:p What are the two main reasons for having coding standards?
??x
Coding standards exist to make the code more readable, understandable, and maintainable (first reason), and to prevent programmers from making common mistakes by limiting the use of certain language features (second reason).
??x

