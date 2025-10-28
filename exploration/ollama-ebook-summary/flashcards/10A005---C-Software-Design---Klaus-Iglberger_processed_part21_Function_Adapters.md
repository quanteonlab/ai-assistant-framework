# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 21)

**Starting Chapter:** Function Adapters

---

#### Adapter Design Pattern vs Strategy Pattern

Background context: The text compares the `std::stack`, `std::queue`, and `std::priority_queue` adaptors with the Adapters and Strategy design patterns. It highlights how these standard library classes adapt container types to stack, queue, and priority queue interfaces respectively.

:p How do `std::stack`, `std::queue`, and `std::priority_queue` relate to Adapter and Strategy design patterns?
??x
The `std::stack`, `std::queue`, and `std::priority_queue` are part of the C++ Standard Library that adapt container types (like `std::vector`, `std::list`, or `std::deque`) to a specific interface. These class templates wrap around the functionality provided by the given container, making them adaptable but not necessarily changing their core behavior. In contrast, Adapters and Strategy patterns are design patterns intended for behavioral flexibility.

```cpp
template< typename T, typename Container = std::deque<T> >
class stack {
public:
    void push(const T& value) { ... }
    T pop() { ... }
    // Other stack operations...
};
```

x??

---

#### Free Functions as Adapters

Background context: The text discusses how free functions like `std::begin` and `std::end` can serve as examples of the Adapter design pattern. These functions adapt a container’s iterator interface to the expected STL iterator interface.

:p How do `std::begin` and `std::end` serve as Adapters in the context of C++ Standard Library?
??x
Free functions like `std::begin` and `std::end` act as adapters by mapping from an available set of functions (e.g., custom iterators) to an expected STL iterator interface. They enable interoperability between different types of containers without modifying their underlying behavior.

```cpp
template< typename Range >
void traverseRange(Range const& range) {
    for (auto&& element : range) {  // Uses `std::begin` and `std::end`
        // ...
    }
}
```

x??

---

#### Argument-Dependent Lookup (ADL)

Background context: ADL is a mechanism in C++ that resolves function calls by considering the namespaces of the arguments. This ensures the correct overload is called, even if it resides in a user-specific namespace.

:p What is Argument-Dependent Lookup (ADL) and why is it relevant to free functions like `std::begin` and `std::end`?
??x
Argument-Dependent Lookup (ADL) is a feature of C++ that allows the compiler to consider additional namespaces when resolving function calls. This mechanism ensures that the correct overload of a function, such as `std::begin`, is called, even if it is defined in a user-specific namespace.

```cpp
// Example usage with ADL
namespace User {
    struct Range {};
    
    // Overload for User::Range
    auto begin(User::Range) -> int* { return nullptr; }
}

void traverseRange(User::Range range) {
    using std::begin;
    using std::end;
    for (auto&& element : range) {
        // ...
    }
}
```

x??

---

#### Adapter Design Pattern Overview
Background context explaining how the Adapter design pattern allows different objects to collaborate despite incompatible interfaces. This is often done by providing a wrapper object that adapts one interface into another expected interface.

The example provided uses an abstract `Duck` class with pure virtual functions `quack()` and `fly()`, which are implemented in derived classes like `MallardDuck`. Similarly, there's an abstract `Turkey` class with different interfaces (`gobble()` and `fly()`) that need to be adapted for use with the Duck interface.

:p What is the main issue highlighted by this example regarding the Adapter design pattern?
??x
The main issue highlighted is the potential violation of Liskov Substitution Principle (LSP) when adapting objects from one class hierarchy into another. Specifically, while a turkey can fly and make a similar sound to a duck, it does not fully implement the expected `quack()` behavior, leading to potential misuse in contexts where ducks are expected.

In the example:
- A `TurkeyAdapter` is created to adapt a `WildTurkey` object to act as if it were a `Duck`.
- The `give_concert` function attempts to call `quack()` on each duck, including the turkey adapter. This results in the turkey making a gobble sound instead of quacking, causing the concert to be a "musical disaster."

The example demonstrates that while adapters can help integrate different interfaces, they must be used judiciously to avoid violating expected behaviors and principles.
x??

---
#### Duck and Turkey Interfaces
Explanation of how the Duck and Turkey classes are structured with their respective virtual functions.

:p What are the key differences between the Duck and Turkey interfaces in terms of methods?
??x
The key differences between the Duck and Turkey interfaces lie in the specific methods they define:

- **Duck Interface**:
  - `quack()`: Expected to make a duck-like sound.
  - `fly()`: Expected to enable flight.

- **Turkey Interface**:
  - `gobble()`: Represents the typical turkey sound.
  - `fly()`: Indicates the ability to fly, albeit not as well as ducks.

These differences highlight that while both interfaces include a `fly()` method, their implementations and expected behaviors differ significantly. A turkey can fly but cannot quack, which violates expectations for duck behavior.
x??

---
#### TurkeyAdapter Class
Explanation of how the `TurkeyAdapter` class is implemented to adapt a `WildTurkey` object to behave like a `Duck`.

:p What does the `TurkeyAdapter` class do?
??x
The `TurkeyAdapter` class adapts a `WildTurkey` object to conform to the expected interface of a `Duck`. It overrides the `quack()` and `fly()` methods from the `Duck` class, mapping them to corresponding `gobble()` and `fly()` methods in the `WildTurkey` class.

```cpp
#include <memory>

class TurkeyAdapter : public Duck {
public:
    explicit TurkeyAdapter(std::unique_ptr<Turkey> turkey)
        : turkey_{std::move(turkey)} {}

    void quack() override { turkey_->gobble(); }
    void fly() override { turkey_->fly(); }

private:
    std::unique_ptr<Turkey> turkey_;
};
```

- `quack()` in the adapter calls `turkey_->gobble()` to simulate a duck-like sound.
- `fly()` remains unchanged as both ducks and turkeys can fly, though with different capabilities.

This class demonstrates how an existing object (turkey) is adapted to fit into another interface (duck).
x??

---
#### DuckChoir Function
Explanation of the `DuckChoir` function that iterates over a collection of ducks and calls their `quack()` method.

:p What does the `give_concert` function do in the provided code?
??x
The `give_concert` function is designed to call the `quack()` method on each element within a `DuckChoir`, which is a container holding pointers or unique pointers to ducks. This function exemplifies how duck typing can be applied, but also highlights potential issues when mixing incompatible types.

```cpp
#include <vector>
using DuckChoir = std::vector<std::unique_ptr<Duck>>;

void give_concert(DuckChoir const& duck_choir) {
    for (auto const& duck : duck_choir) {
        duck->quack();
    }
}
```

In the example:
- A `DuckChoir` contains multiple ducks, including a `WildTurkey` wrapped in a `TurkeyAdapter`.
- When `give_concert` is called with this mixed collection, it will call `duck->quack()` on each element. For turkeys adapted as ducks, this results in calling the `gobble()` method instead of `quack()`, leading to unexpected behavior.
x??

---
#### Liskov Substitution Principle (LSP) Violation
Explanation of how the Adapter design pattern can lead to violations of the Liskov Substitution Principle if not used carefully.

:p How does the example illustrate a potential violation of the Liskov Substitution Principle?
??x
The example illustrates a potential violation of the Liskov Substitution Principle (LSP) by showing that when adapting a `WildTurkey` using a `TurkeyAdapter`, it does not fully adhere to the expected behavior of a duck. Specifically:

- The `quack()` method in a duck is supposed to produce a characteristic quacking sound, but the adapter makes the turkey "quack" with its gobble sound.
- The `fly()` method in both ducks and turkeys can be similar in functionality, but the context expects different behaviors. For example, the adapted turkey might fly short distances, which is not the primary expectation for a duck.

These discrepancies indicate that while the adapter allows integration of the turkey into the duck hierarchy, it does so at the cost of violating expected interface behavior, thus potentially breaking LSP.
x??

---

#### Adapter Design Pattern: Object Adapters

Background context explaining the concept. The Adapter design pattern is used to adapt an existing interface of a class into another one that is required by the client. It allows objects with incompatible interfaces to collaborate. This pattern is useful for both dynamic and static polymorphism.

:p What is an object adapter in the Adapter design pattern?
??x
An object adapter is a structure that implements the target interface, which is what the client needs. The adapter contains a reference to an object of the adaptee class (the original class with the incompatible interface), and it forwards requests from the target interface to the adaptee's interface.

For example, consider you have a `LegacyPrinter` class with an `printText()` method and you want to adapt it so that it can be used as if it were implementing `Printable` interface:

```java
// Adaptee class (Legacy Printer)
class LegacyPrinter {
    void printText(String text) { 
        // Print logic here
    }
}

// Target Interface (Printable)
interface Printable {
    void printDocument(Document doc);
}
```

The adapter might look like this:
```java
// Object Adapter
class PrinterAdapter implements Printable {
    private LegacyPrinter legacyPrinter;

    public PrinterAdapter(LegacyPrinter legacyPrinter) {
        this.legacyPrinter = legacyPrinter;
    }

    @Override
    public void printDocument(Document doc) {
        // Convert Document to Text for printing
        String text = doc.getText();
        legacyPrinter.printText(text);
    }
}
```
x??

---

#### Adapter Design Pattern: Class Adapters

:p What is a class adapter in the Adapter design pattern?
??x
A class adapter is another form of adapting an interface by inheriting from both the adaptee and the target classes. This approach allows you to implement the target interface directly within the derived class, using the methods provided by the adaptee.

For example, if `LegacyPrinter` is a class that provides a method `printText()`, and we want to use it with an adapter that implements the `Printable` interface:

```java
// Adaptee class (Legacy Printer)
class LegacyPrinter {
    void printText(String text) { 
        // Print logic here
    }
}

// Target Interface (Printable)
interface Printable {
    void printDocument(Document doc);
}
```

A class adapter might look like this:
```java
// Class Adapter
class PrinterAdapter extends LegacyPrinter implements Printable {
    
    @Override
    public void printDocument(Document doc) {
        // Convert Document to Text for printing
        String text = doc.getText();
        super.printText(text); // Use the printText method from the LegacyPrinter class
    }
}
```

x??

---

#### Adapter vs Strategy Design Pattern

:p How do the Adapter and Strategy design patterns differ?
??x
The Adapter pattern is used when you want to make existing classes work with a new system. It adapts an interface by wrapping it in another interface that clients expect, allowing incompatible interfaces to collaborate.

On the other hand, the Strategy pattern provides a way to define a family of algorithms, encapsulate each one, and make them interchangeable. Strategies are used when you want different behaviors to be interchangeable, without modifying the client code.

For example:
- **Adapter**: If you have an old class `LegacyPrinter` that prints text and you need to use it as if it were part of a new system `Printable`, you would use an Adapter.
- **Strategy**: If you have multiple ways of printing (like different printer brands), each with its own behavior, you could encapsulate these behaviors using the Strategy pattern.

```java
// Example of Strategy Pattern
interface Printer {
    void print(String text);
}

class FastPrinter implements Printer {
    @Override
    public void print(String text) {
        // Fast printing logic here
    }
}

class SlowPrinter implements Printer {
    @Override
    public void print(String text) {
        // Slow printing logic here
    }
}
```
x??

---

#### LSP Violations in Adapter Pattern

:p What is the Liskov Substitution Principle (LSP) and how can it be violated when using the Adapter pattern?
??x
The Liskov Substitution Principle states that objects of a superclass shall be replaceable with objects of its subclasses without breaking the application.

A violation occurs if the adapter introduces behavior or state in the target interface that is not present in the adaptee's original interface. This can lead to unexpected outcomes when the adapter is used as part of an inheritance hierarchy where it is supposed to behave like a subclass but does more than what it should.

For example, consider the following scenario:
- `LegacyPrinter` prints text only.
- An `EnhancedAdapter` that wraps `LegacyPrinter` and extends its functionality to print images.

```java
// Adaptee class (Legacy Printer)
class LegacyPrinter {
    void printText(String text) { 
        // Print logic here
    }
}

// Target Interface (Printable)
interface Printable {
    void printDocument(Document doc);
}
```

The adapter might have a method for printing images, which violates LSP:
```java
// Violating LSP in Adapter
class EnhancedAdapter extends LegacyPrinter implements Printable {
    
    @Override
    public void printDocument(Document doc) {
        // Convert Document to Text and Image
        String text = doc.getText();
        if (doc.hasImage()) {
            // Print image logic here, which is not part of the original interface
        }
        super.printText(text);
    }
}
```

To avoid this violation, ensure that the adapter only provides what is necessary to adapt the existing behavior.

x??

---

#### Observer Design Pattern

Background context explaining the concept. The Observer design pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically. This decouples software entities from each other, making it easier to extend and change them independently.

:p What is the intent of the Observer design pattern?
??x
The intent of the Observer design pattern is "Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically."

In this pattern, a subject maintains a list of observers and notifies them whenever it changes. Observers can be added or removed dynamically without affecting the subject.

For example:
- A `MessageQueue` is the subject.
- Users who want to know when new messages arrive are attached as observers to the queue.

```java
// Observer interface
interface Observer {
    void update(String message);
}

// Subject interface (Abstract)
interface Subject {
    void addObserver(Observer o);
    void removeObserver(Observer o);
    void notifyObservers();
}
```

A concrete implementation might look like:
```java
class MessageQueue implements Subject {
    private List<Observer> observers = new ArrayList<>();

    public void addObserver(Observer observer) {
        observers.add(observer);
    }

    public void removeObserver(Observer observer) {
        observers.remove(observer);
    }

    public void notifyObservers(String message) {
        for (Observer o : observers) {
            o.update(message);
        }
    }

    // Add a new message to the queue
    public void addMessage(String message) {
        // Add logic to handle messages
        notifyObservers(message); // Notify all observers
    }
}
```

x??

---

