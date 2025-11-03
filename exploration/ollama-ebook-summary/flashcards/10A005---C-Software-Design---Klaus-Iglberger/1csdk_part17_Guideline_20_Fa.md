# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 17)

**Starting Chapter:** Guideline 20 Favor Composition over Inheritance

---

#### Inheritance vs. Composition
Inheritance is often overvalued and misunderstood, especially when it comes to reusability and decoupling. While inheritance can simplify certain aspects of software design, it has significant limitations that make it less preferable compared to composition.

:p What are some key limitations of using inheritance?
??x
- Inheritance does not inherently promote code reuse in the way it is often perceived.
- It can create tight coupling between classes.
- It forces specific implementation details and constraints on derived classes, which may be suboptimal for certain applications.
- Overuse of inheritance leads to deep class hierarchies that are hard to manage.

Code examples:
```cpp
class Shape {
public:
    virtual ~Shape() = default;
    virtual void translate(/* some arguments */) = 0;
    virtual void rotate(/* some arguments */) = 0;
    virtual void draw(const /*some arguments*/) const = 0;
    virtual void serialize(const /*some arguments*/) const = 0;
};

void rotateAroundPoint(Shape& shape);
void mergeShapes(Shape& s1, Shape& s2);
void writeToFile(const Shape& shape);
void sendViaRPC(const Shape& shape);

// All these functions use the Shape abstraction
```
x??

---

#### Reusability Through Polymorphism
Inheritance is often thought to promote code reuse by allowing derived classes to inherit from a base class. However, true reusability comes from polymorphic usage of abstractions.

:p How does real reusability differ from perceived reusability in inheritance?
??x
Real reusability arises when a class can be used polymorphically, meaning that different concrete implementations share the same interface and can be treated interchangeably. This allows for reuse of functionality without tight coupling.

Code example:
```cpp
// A Shape base class with pure virtual functions
class Shape {
public:
    virtual ~Shape() = default;
    virtual void translate(/* some arguments */) = 0;
    virtual void rotate(/* some arguments */) = 0;
    virtual void draw(const /*some arguments*/) const = 0;
    virtual void serialize(const /*some arguments*/) const = 0;
};

// Functions that operate on Shapes
void rotateAroundPoint(Shape& shape);
void mergeShapes(Shape& s1, Shape& s2);
void writeToFile(const Shape& shape);
void sendViaRPC(const Shape& shape);

// These functions can be reused for all kinds of shapes because they work with the common interface.
```
x??

---

#### Coupling and Inheritance
Inheritance is often touted as a way to decouple software entities, but it actually creates coupling due to the implementation details that must be inherited.

:p What are some negative aspects of inheritance in terms of coupling?
??x
- Inheritance forces certain implementation details on derived classes, which can lead to suboptimal designs.
- It fixes function arguments and return types, limiting flexibility.
- Deep inheritance hierarchies can make the codebase harder to manage.
- Classes can become too closely tied to each other due to shared base class implementations.

Code example:
```cpp
// A classic Visitor pattern implementation where inheritance is used but can be problematic
class ShapeVisitor {
public:
    virtual void visitCircle(Circle& c) = 0;
    virtual void visitRectangle(Rectangle& r) = 0;
};

class Circle : public Shape, public ShapeVisitor {
public:
    // Must implement pure virtual functions from both base classes
    void translate(/* some arguments */);
    void rotate(/* some arguments */);
    void draw(const /*some arguments*/) const override;
    void serialize(const /*some arguments*/) const override;

    void visitCircle(Circle& c) override { /* implementation */ }
    void visitRectangle(Rectangle& r) {} // Not implemented, causing a compile error
};
```
x??

---

#### Composition Over Inheritance
Composition is often the better choice over inheritance because it promotes loose coupling and reusability through aggregation.

:p Why is composition considered superior to inheritance in many cases?
??x
- Composition allows for more flexible and decoupled designs.
- It enables extension by means of composition, allowing parts of an object to be replaced or extended independently.
- Compositional patterns like the Strategy pattern are often simpler and more maintainable than their inherited counterparts.

Code example:
```cpp
// Using a strategy pattern with composition
class Context {
public:
    void setStrategy(std::shared_ptr<IShapeStrategy> strategy) { _strategy = strategy; }
    void operation() { _strategy->execute(); }

private:
    std::shared_ptr<IShapeStrategy> _strategy;
};

interface IShapeStrategy {
    virtual void execute() = 0;
};

class CircleStrategy : public IShapeStrategy {
public:
    void execute() override { /* Circle-specific logic */ }
};

class RectangleStrategy : public IShapeStrategy {
public:
    void execute() override { /* Rectangle-specific logic */ }
};
```
x??

---

#### Peter Parker Principle
Inheritance can be a powerful feature, but it requires careful management to avoid misuse and overuse.

:p How does the "Peter Parker Principle" apply to inheritance?
??x
The "Peter Parker Principle" emphasizes that with great power comes great responsibility. Inheritance provides significant capabilities, but improper use or overuse can lead to complex, hard-to-maintain codebases. Developers must be cautious when using inheritance to ensure they are leveraging its benefits without causing unnecessary complications.

Code example:
```cpp
// Misusing inheritance by creating a deep class hierarchy
class Base {
public:
    virtual void method() = 0;
};

class Derived1 : public Base {
public:
    void method() override { /* implementation */ }
};

class Derived2 : public Derived1 {
public:
    void method() override { /* different implementation */ }
};
```
x??

---

#### Favor Composition Over Inheritance
Background context explaining the preference for composition over inheritance, focusing on tight coupling issues with inheritance and how composition can enable design patterns like Command. The Pragmatic Programmer advocates using "Has-A" relationships instead of "Is-A" when possible to improve code flexibility and maintainability.
:p What does the guideline suggest about the use of inheritance?
??x
The guideline suggests that inheritance is often overused and sometimes misused, leading to tight coupling between classes. Instead, it recommends favoring composition (using "Has-A" relationships) because it can provide more flexible and decoupled designs, which is beneficial for implementing design patterns like Command.
x??

---
#### Command Design Pattern Intent
Explanation of the Command pattern's intent, focusing on encapsulating a request as an object to support various operations such as queuing or logging requests, and enabling undoable operations. The goal is to abstract away the details of how work packages are executed and reversed.
:p What does the Command design pattern intend to achieve?
??x
The Command design pattern intends to encapsulate a request as an object so that you can parameterize clients with different requests, queue or log these requests, and support undoable operations. This abstraction allows for easier implementation of new kinds of work packages by recognizing variations in work package types.
x??

---
#### CalculatorCommand Base Class
Explanation of the `CalculatorCommand` base class provided in the text, detailing its purpose and how it enforces the implementation of `execute()` and `undo()` methods to define mathematical operations. The example code snippet shows a basic structure for this command pattern implementation.
:p What is the role of the `CalculatorCommand` base class?
??x
The role of the `CalculatorCommand` base class is to provide an abstract interface for implementing concrete commands that perform specific mathematical operations on integers. It enforces that derived classes implement both the `execute()` and `undo()` methods, allowing these commands to be executed and reversed as needed.
x??

---
#### Implementing Execute and Undo Methods
Explanation of how the `execute()` and `undo()` methods should function in a `CalculatorCommand` subclass, detailing their responsibilities. The example code snippet shows how these methods might be implemented for addition and subtraction operations.
:p How should the `execute()` and `undo()` methods work in a `CalculatorCommand` subclass?
??x
The `execute()` method should perform the actual mathematical operation (e.g., adding or subtracting from an integer), while the `undo()` method should reverse that operation. For example, if `execute()` adds 5 to an integer, `undo()` would subtract 5.
```cpp
class AddCommand : public CalculatorCommand {
public:
    int execute(int i) const override { return i + 5; }
    int undo(int i) const override { return i - 5; }
};

class SubtractCommand : public CalculatorCommand {
public:
    int execute(int i) const override { return i - 3; }
    int undo(int i) const override { return i + 3; }
};
```
x??

---
#### Invoker and Receiver
Explanation of the `Invoker` and `Receiver` classes in the Command pattern, detailing their roles and how they interact with commands. The example code snippet shows a possible implementation for an invoker that uses a command to perform a calculation.
:p What are the roles of the `Invoker` and `Receiver` in the Command pattern?
??x
The `Invoker` is responsible for invoking the appropriate `Command` object, while the `Receiver` carries out the actual work when the command is executed. The invoker receives the command from somewhere (e.g., user input) and then executes it by calling its `execute()` method on the receiver.
```cpp
class Invoker {
public:
    void setCommand(CalculatorCommand* cmd) { command = cmd; }
    void execute() { command->execute(currentValue); }

private:
    CalculatorCommand* command;
    int currentValue = 0;
};
```
x??

---
#### Concrete Command Implementation
Explanation of how a concrete `CalculatorCommand` subclass would be implemented to perform specific calculations, including the use of a receiver. The example code snippet demonstrates implementing a `SubtractCommand` that uses an integer receiver.
:p How would you implement a concrete `CalculatorCommand` for subtraction?
??x
To implement a `CalculatorCommand` for subtraction, you would create a subclass like `SubtractCommand`, which uses an integer as the receiver. This class overrides the `execute()` and `undo()` methods to perform the necessary calculations.
```cpp
class SubtractCommand : public CalculatorCommand {
private:
    int value;

public:
    SubtractCommand(int v) : value(v) {}

    int execute(int i) const override { return i - value; }
    int undo(int i) const override { return i + value; }
};
```
x??

---

#### Add and Subtract Command Classes
Background context explaining how `Add` and `Subtract` classes implement the `CalculatorCommand` interface to represent commands for a calculator. The execute() function performs an addition or subtraction, while the undo() function reverses the operation.

:p What is the purpose of the `Add` class in this example?
??x
The `Add` class represents an addition command in the calculator system. It implements the `execute()` method to add its operand to a given value and the `undo()` method to subtract the operand, effectively reversing the addition.

```cpp
class Add : public CalculatorCommand {
public:
    explicit Add(int operand) : operand_(operand) {}
    
    int execute(int i) const override {
        return i + operand_;
    }
    
    int undo(int i) const override {
        return i - operand_;
    }
private:
    int operand_{};
};
```
x??

---
#### Subtract Command Class
Background context explaining how `Subtract` class implements the `CalculatorCommand` interface to represent a subtraction command in the calculator system. The execute() function subtracts its operand from a given value, and the undo() function adds back the operand.

:p What is the purpose of the `Subtract` class in this example?
??x
The `Subtract` class represents a subtraction command in the calculator system. It implements the `execute()` method to subtract its operand from a given value and the `undo()` method to add the operand, effectively reversing the subtraction.

```cpp
class Subtract : public CalculatorCommand {
public:
    explicit Subtract(int operand) : operand_(operand) {}
    
    int execute(int i) const override {
        return i - operand_;
    }
    
    int undo(int i) const override {
        return i + operand_;
    }
private:
    int operand_{};
};
```
x??

---
#### Calculator Class
Background context explaining the role of the `Calculator` class, which manages a stack of operations and updates its state based on these operations. It supports computing results and undoing previous actions.

:p What is the function of the `Calculator` class in this example?
??x
The `Calculator` class manages a series of calculator commands, allowing it to compute results and undo operations. It uses a stack to keep track of executed commands and their corresponding values.

```cpp
class Calculator {
public:
    void compute(std::unique_ptr<CalculatorCommand> command);
    
    void undoLast();
    
    int result() const;
    
    void clear();

private:
    using CommandStack = std::stack<std::unique_ptr<CalculatorCommand>>;

    int current_{};
    
    CommandStack stack_;
};
```
x??

---
#### Compute Function
Background context explaining the `compute()` function in the `Calculator` class, which applies a command to update the calculator's state and stores it on the stack.

:p What does the `compute()` function do?
??x
The `compute()` function takes a `CalculatorCommand` instance, executes it, updates the current value, and pushes the command onto the stack. This allows the calculator to remember each operation for later undo operations.

```cpp
void Calculator::compute(std::unique_ptr<CalculatorCommand> command) {
    current_ = command->execute(current_);
    stack_.push(std::move(command));
}
```
x??

---
#### Undo Last Function
Background context explaining the `undoLast()` function in the `Calculator` class, which reverts the last executed command and updates the calculator's state.

:p What does the `undoLast()` function do?
??x
The `undoLast()` function pops the last command from the stack, undoes its effect by calling its `undo()` method, and updates the current value. This allows reverting to the previous state before the last operation was performed.

```cpp
void Calculator::undoLast() {
    if (stack_.empty()) return;
    
    auto command = std::move(stack_.top());
    stack_.pop();
    
    current_ = command->undo(current_);
}
```
x??

---
#### Result Function
Background context explaining the `result()` function in the `Calculator` class, which returns the current value after all operations have been computed.

:p What does the `result()` function return?
??x
The `result()` function returns the current state of the calculator after all commands have been applied. This is essentially the final result after performing a series of operations.

```cpp
int Calculator::result() const {
    return current_;
}
```
x??

---
#### Clear Function
Background context explaining the `clear()` function in the `Calculator` class, which resets the calculator to its initial state by setting the current value to zero and clearing the stack.

:p What does the `clear()` function do?
??x
The `clear()` function resets the calculator's state by setting the current value to 0 and clearing the command stack. This effectively undoes all previous operations and prepares the calculator for new computations.

```cpp
void Calculator::clear() {
    current_ = 0;
    CommandStack{}.swap(stack_); // Clearing the stack
}
```
x??

---

#### Command Design Pattern Overview
The Command design pattern allows you to encapsulate a request as an object, thereby allowing for different requests, queueing or logging them, and supporting undoable operations. This design promotes separation of concerns (SRP) by abstracting the actions into commands that can be executed or undone independently.
:p What is the primary purpose of the Command design pattern?
??x
The primary purpose of the Command design pattern is to encapsulate a request as an object, thereby allowing for different requests to be queued, logged, and supported with undoable operations. This promotes separation of concerns by abstracting actions into commands that can be executed or undone independently.
??x

---

#### Separation of Concerns (SRP) through Command
The Command design pattern helps adhere to the Single Responsibility Principle (SRP), where a class has one reason to change, i.e., it is responsible for just performing operations and their reverses. By decoupling requesters from actions, this principle ensures that changes in the command implementation do not affect the client code.
:p How does the Command pattern help adhere to the Single Responsibility Principle (SRP)?
??x
The Command pattern helps adhere to the Single Responsibility Principle (SRP) by encapsulating requests as objects. This means that a class is responsible for just performing operations and their reverses, without involving other responsibilities. Changes in command implementation do not affect the client code, ensuring that each class has only one reason to change.
??x

---

#### Open/Closed Principle (OCP)
By using the Command design pattern, you can add new commands without modifying existing classes, adhering to the Open/Closed Principle (OCP). This allows for extending a program's behavior by adding new commands, rather than changing existing code, which enhances maintainability and flexibility.
:p How does the Command design pattern support the Open/Closed Principle (OCP)?
??x
The Command design pattern supports the Open/Closed Principle (OCP) by allowing you to add new commands without modifying existing classes. You can extend a program's behavior by adding new command implementations, ensuring that the core logic remains unchanged and only the external interfaces need to be adjusted.
??x

---

#### Dependency Injection in Command
Properly assigning ownership of the `Command` base class to higher-level components ensures adherence to the Dependency Inversion Principle (DIP), where dependencies should not depend on details, but rather on abstractions. This promotes loose coupling and better testability.
:p How does dependency injection play a role in the Command design pattern?
??x
Dependency injection plays a role in the Command design pattern by ensuring that ownership of the `Command` base class is properly assigned to higher-level components. This adherence to the Dependency Inversion Principle (DIP) promotes loose coupling and better testability, as dependencies are inverted from concrete implementations to abstract interfaces.
??x

---

#### Thread Pool Example
The example provided shows how a thread pool can be used with commands derived from a `Command` base class to schedule tasks in parallel. This decouples the task scheduling mechanism from the actual work being performed by the threads.
:p How does the ThreadPool class utilize the Command design pattern?
??x
The ThreadPool class utilizes the Command design pattern by allowing you to schedule tasks via the `schedule()` function, which can be any task represented as a command derived from the `Command` base class. This decouples the task scheduling mechanism from the actual work being performed by the threads.
??x

---

#### std::for_each Example
The `std::for_each()` algorithm in the C++ Standard Library is an example of the Command design pattern, where you can specify a unary function to be executed on each element. This allows for flexible and reusable algorithms that can perform different actions based on the provided function.
:p How does `std::for_each()` demonstrate the Command design pattern?
??x
`std::for_each()` demonstrates the Command design pattern by allowing you to specify a unary function as the third argument, which is executed on each element. This flexibility enables different actions to be performed without modifying existing code, adhering to the OCP and SRP principles.
??x

---

#### Algorithms in C++ Standard Library
Background context: The C++ Standard Library includes a variety of algorithms that can operate on ranges of elements, such as vectors or arrays. These algorithms are implemented using design patterns like Strategy and Command to provide flexibility in how operations are performed and what is done.

:p What are some examples of algorithms in the C++ Standard Library?
??x
Examples include `std::for_each`, `std::partition`, and `std::sort`. These functions allow for flexible operation application on elements within a range.
x??

---
#### Strategy Design Pattern
Background context: The Strategy design pattern allows you to define a family of algorithms, encapsulate each one, and make them interchangeable. This enables the algorithm selection at runtime.

:p How does the `std::partition` function use the Strategy design pattern?
??x
The `std::partition` function uses the Strategy design pattern by allowing a predicate function (UnaryPredicate) to be passed in as an argument. This predicate determines how elements are partitioned into two groups, but it is the implementation of this predicate that defines the "how" part of the operation.
x??

---
#### Command Design Pattern
Background context: The Command design pattern encapsulates a request as an object, thereby allowing you to parameterize methods with different requests, queue or log requests, and support undoable operations.

:p How does `std::for_each` exemplify the Command design pattern?
??x
The `std::for_each` function allows for passing in a lambda or a function pointer that defines what operation should be performed on each element. This is akin to encapsulating a command (the action) to be executed, which aligns with the Command design pattern.
x??

---
#### Comparison Between Strategy and Command Patterns
Background context: Both the Strategy and Command patterns involve dynamic polymorphism, but they serve different intents: Strategy defines "how" something should be done, while Command specifies what should be done.

:p How do `std::partition` and `std::for_each` differ in their implementation details?
??x
While both can use function objects or lambda functions, `std::partition` focuses on defining a predicate for partitioning elements into two groups. In contrast, `std::for_each` allows you to define the operation that should be applied to each element.
x??

---
#### Implementation of Calculator Using Command Pattern
Background context: The provided example shows how to implement a calculator using the Command pattern by directly executing actions without dependency injection.

:p How does the `Calculator` class in the example use the Command design pattern?
??x
The `Calculator` class uses immediate evaluation rather than dependency injection. In this case, it computes the result immediately when the `compute` method is called, making it more aligned with the Command pattern where actions are executed directly.
x??

---
#### Implementation of Calculator Using Strategy Pattern
Background context: An alternative approach to implementing a calculator using the Strategy design pattern involves injecting a strategy object that defines how calculations should be performed.

:p How does the `Calculator` class in this example use the Strategy design pattern?
??x
The `Calculator` class uses dependency injection by accepting a `std::unique_ptr<CalculatorStrategy>` through its constructor or set method. This allows for different calculation strategies to be plugged into the calculator, demonstrating the Strategy design pattern.
x??

---

