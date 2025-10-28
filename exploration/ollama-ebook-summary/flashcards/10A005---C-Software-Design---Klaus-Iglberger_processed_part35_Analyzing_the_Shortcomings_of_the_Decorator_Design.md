# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 35)

**Starting Chapter:** Analyzing the Shortcomings of the Decorator Design Pattern

---

#### Decorator Design Pattern Overview
Background context explaining the decorator pattern. The decorator pattern allows for flexible and dynamic feature enhancement by wrapping objects with additional behaviors without modifying their structure or interface.

The core idea is to introduce a new class hierarchy that wraps around existing classes, allowing for the addition of responsibilities in a non-intrusive manner.
:p What is the decorator pattern used for?
??x
The decorator pattern is used for adding functionalities dynamically to an object without altering its structure. This allows for flexibility and ease of maintenance by avoiding changes to the original class hierarchy.

For example:
```cpp
class Item {
public:
    virtual Money price() const = 0;
};

class DecoratedPriceStrategy : public Item {
private:
    std::unique_ptr<Item> item_;
public:
    explicit DecoratedPriceStrategy(std::unique_ptr<Item> item) : item_(std::move(item)) {}

    Money price() const override { return item_->price(); }
};
```
x??

---

#### Performance Considerations of Decorator
Explanation of the performance implications when using decorators extensively. Each level in a decorator hierarchy adds an extra virtual function call, which can lead to performance overhead.

:p What is the potential downside related to performance with decorators?
??x
The potential downside related to performance with decorators is that each additional layer (or decorator) introduces one more virtual function call. This can lead to a significant performance penalty in scenarios where many decorators are applied hierarchically, as it results in multiple levels of indirection.

For example:
```cpp
class DecoratorA : public Item {
private:
    std::unique_ptr<Item> item_;
public:
    explicit DecoratorA(std::unique_ptr<Item> item) : item_(std::move(item)) {}

    Money price() const override { return item_->price(); }
};

class DecoratorB : public DecoratorA {
public:
    Money price() const override { return DecoratorA::price() * 1.05; } // 5% increase
};
```
In this case, there are two virtual function calls: one for `DecoratorA` and another for the original `Item`.
x??

---

#### Combining Decorators Rationale
Explanation of why combining decorators in a nonsensical way can be problematic. The example given is wrapping a tax decorator around an already taxed item.

:p Why should you avoid combining certain decorators?
??x
You should avoid combining certain decorators because it can lead to logical inconsistencies or incorrect behavior. For instance, applying a `Taxed` decorator around another `Taxed` decorator would result in double taxation, which is not typically the intended outcome and could be dangerous depending on the scenario.

For example:
```cpp
class TaxDecorator : public DecoratorA {
public:
    Money price() const override { return DecoratorA::price() * 1.05; } // 5% tax
};
```
If you apply `TaxDecorator` to an item that is already taxed, it will apply the tax twice, leading to incorrect pricing.

To avoid such issues, separate concerns like taxes and discounts into different strategies.
x??

---

#### Strategy Pattern for Taxes
Explanation of using a strategy pattern for handling taxes instead of decorators. This approach can make interfaces easier to use correctly and harder to use incorrectly by enforcing proper usage.

:p Why might you prefer the strategy pattern over decorators for taxes?
??x
You might prefer the strategy pattern over decorators for taxes because it enforces correct usage through well-defined interfaces and strategies, making it more difficult to apply taxes inappropriately. By using a `TaxStrategy` class, you can clearly define how taxes should be applied without accidentally stacking them incorrectly.

For example:
```cpp
class TaxStrategy {
public:
    virtual ~TaxStrategy() = default;
    virtual Money applyTax(Money price) const = 0;
};

class SalesTaxStrategy : public TaxStrategy {
public:
    Money applyTax(Money price) const override { return price * 1.05; } // 5% sales tax
};
```
Using a `SalesTaxStrategy` ensures that only one type of tax is applied, and the logic for applying it is encapsulated in its implementation.
x??

---

#### TaxedItem Class Implementation
Explanation of how the `TaxedItem` class combines an item with a tax strategy. This approach avoids the pitfalls of decorators while still providing flexible taxation behavior.

:p How does the `TaxedItem` class manage taxes?
??x
The `TaxedItem` class manages taxes by combining an `Item` with a `TaxStrategy`. It provides methods to calculate both the net and gross prices, ensuring that taxes are applied correctly without stacking them incorrectly. This approach uses smart pointers for managing the lifetimes of its components.

For example:
```cpp
class TaxedItem {
public:
    explicit TaxedItem(std::unique_ptr<Item> item, std::unique_ptr<TaxStrategy> taxer)
        : item_(std::move(item)), taxer_(std::move(taxer)) {}

    Money netPrice() const { return item_->price(); }
    Money grossPrice() const { return taxer_->applyTax(item_->price()); }

private:
    std::unique_ptr<Item> item_;
    std::unique_ptr<TaxStrategy> taxer_;
};
```
The `netPrice()` method returns the original price without taxes, while `grossPrice()` calculates the price including the applied tax.
x??

---

#### Decorator Design Pattern Overview
Background context: The Decorator design pattern is a structural design pattern that allows behavior to be added to an individual object, either statically or dynamically, without affecting the behavior of other objects from the same class. It promotes loose coupling by allowing decorator classes to wrap around any object with a compatible interface.
:p What is the main purpose of using the Decorator design pattern?
??x
The primary purpose of using the Decorator design pattern is to add new functionality to existing objects dynamically and hierarchically, without modifying the original class. This is achieved by wrapping the original object with one or more decorator classes that provide additional behavior.
??x

---

#### Inheritance vs. Decorators
Background context: Inheritance can be used to extend a class's behavior, but it can also lead to tightly coupled code and may violate the Single Responsibility Principle (SRP). The Decorator design pattern offers a way to achieve similar results with less coupling by wrapping objects.
:p Why is inheritance rarely the answer when extending object behavior?
??x
Inheritance is rarely the answer because it can lead to class hierarchies that are difficult to manage, especially as more behaviors need to be added. Inheritance tightly couples derived classes to their base classes, making it harder to change or extend functionality without affecting other parts of the codebase. Additionally, it may violate the Single Responsibility Principle if a class is trying to handle too many responsibilities.
??x

---

#### Value-Based Compile-Time Decorator
Background context: The provided example demonstrates how to implement a compile-time decorator using C++ templates and concepts. This approach leverages static polymorphism to enable early type checking and optimization during compilation.
:p How does the provided code snippet illustrate a value-based compile-time decorator?
??x
The code snippet illustrates a value-based compile-time decorator by defining the `ConferenceTicket` class as a simple, non-inheritable value object. It uses templates and concepts to create decorators that wrap around instances of `ConferenceTicket`, allowing for hierarchical customization without modifying the original class.
```cpp
#include <Money.h>
#include <string>
#include <utility>

class ConferenceTicket {
public:
    ConferenceTicket(std::string name, Money price)
        : name_{ std::move(name) }
        , price_{ price } {}
    
    std::string const& name() const { return name_; }
    Money price() const { return price_; }

private:
    std::string name_;
    Money price_;
};
```
??x

---

#### Static Polymorphism vs. Dynamic Polymorphism
Background context: The text mentions that one approach to implementing a decorator is through static polymorphism, which uses templates and concepts for compile-time type safety, while the other uses dynamic polymorphism, leveraging runtime dispatch.
:p What are the key differences between static and dynamic polymorphism in the context of decorators?
??x
Static polymorphism (used in C++ with templates) allows for type-safe and optimized code at compile time. It enables early detection of type errors and can be more efficient because it avoids virtual function call overhead. Dynamic polymorphism, on the other hand, uses virtual functions to provide runtime dispatching, which is flexible but incurs some performance overhead due to dynamic lookup.
??x

---

#### Implementing a Value-Based Compile-Time Decorator
Background context: The implementation of the value-based compile-time decorator focuses on using templates and concepts for type safety. This approach allows for easy addition of new decorators without modifying existing classes.
:p How can we add a simple wrapper (decorator) to the `ConferenceTicket` class using C++20 concepts?
??x
To add a simple wrapper to the `ConferenceTicket` class, you can create a template decorator that accepts a `ConferenceTicket` and adds additional behavior. Here’s an example:
```cpp
template<typename T>
class Decorator {
public:
    explicit Decorator(T& obj) : obj_(obj) {}

    // Forwarding function calls
    Money price() const { return obj_.price(); }

private:
    T& obj_;
};

// Usage
ConferenceTicket ticket("Tech Conference", Money{100});
Decorator<ConferenceTicket> wrappedTicket(ticket);
```
??x

---

#### Summary of Design Patterns
Background context: The text highlights the differences between Decorator, Adapter, and Strategy patterns. Each pattern addresses different needs in software design, such as adding responsibilities (Decorator), adapting interfaces (Adapter), or selecting algorithms at runtime (Strategy).
:p What are some key differences between the Decorator, Adapter, and Strategy patterns?
??x
- **Decorator**: Used to add new functionality to an object dynamically without modifying its structure. It's hierarchical and allows for multiple layers of behavior.
- **Adapter**: Used to make existing interfaces compatible with other interfaces that are already in use by a client. It changes the interface of a class but doesn't modify the underlying implementation.
- **Strategy**: Used when you need to choose an algorithm or behavior at runtime from a family of algorithms without changing the calling code. Each strategy is encapsulated as a separate object and can be switched during execution.
??x

---

#### Introduction to Decorator Pattern Implementation

This section explains how the Decorator pattern is implemented using template classes and constraints. The implementation avoids direct inheritance by composition or non-public inheritance, making it more flexible and adhering to design principles.

:p What is a key characteristic of the new Decorator implementation described?
??x
A key characteristic of this implementation is that it uses templates with constraints (PricedItem) instead of traditional inheritance. This allows for greater flexibility in adding decorators without modifying existing classes.
??x

---

#### PricedItem Constraint

The PricedItem constraint ensures types used as Item parameters must have a `price()` method, enforcing semantic requirements.

:p What is the role of the PricedItem concept?
??x
The PricedItem concept acts as a template constraint that enforces certain behavior on the decorated item. Specifically, it requires any type passed to the Discounted or Taxed classes to define a `price()` member function. If not defined, compilation will fail.
??x

---

#### Implementation of Discounted Class Template

This class uses composition to wrap an Item and apply a discount.

:p How does the Discounted class template implement its functionality?
??x
The Discounted class template implements its functionality by storing an instance of the decorated item as a data member and applying a discount when `price()` is called. It is implemented using composition rather than inheritance, adhering to the guideline to "favor composition over inheritance."

```cpp
template< double discount, PricedItem Item >
class Discounted {
public:
    template< typename... Args >
    explicit Discounted(Args&&... args)
        : item_{ std::forward<Args>(args)... } {}

    Money price() const {
        return item_.price() * (1.0 - discount);
    }

private:
    Item item_;
};
```
??x

---

#### Implementation of Taxed Class Template

This class uses private inheritance to inherit from the decorated item and apply a tax.

:p How does the Taxed class template implement its functionality?
??x
The Taxed class template implements its functionality by privately inheriting from the decorated item and applying a tax when `price()` is called. This approach allows it to directly access the `price()` method of the base class.

```cpp
template< double taxRate, PricedItem Item >
class Taxed : private Item {
public:
    template< typename... Args >
    explicit Taxed(Args&&... args)
        : Item{ std::forward<Args>(args)... } {}

    Money price() const {
        return Item::price() * (1.0 + taxRate);
    }
};
```
??x

---

#### Benefits of the Implementation

This implementation provides flexibility, separation of concerns, and adherence to design principles like the Open-Closed Principle.

:p What benefits does this implementation provide?
??x
The implementation offers several benefits:
- **Flexibility**: New items can be easily added without modifying existing code.
- **Separation of Concerns**: The `PricedItem` constraint ensures that only types with a `price()` method can be decorated, promoting clean separation of concerns based on the Single-Responsibility Principle (SRP).
- **Adherence to Principles**:
  - **Open-Closed Principle (OCP)**: New decorators and items can be added without modifying existing code.
  - **Dependency Inversion Principle (DIP)**: High-level modules depend on abstractions, not concrete implementations.

These features make the implementation more maintainable and scalable.
??x

---

#### Compile-Time vs Runtime Decorator Implementation
Background context: This section discusses the performance comparison between a runtime decorator implementation and a compile-time template-based decorator implementation. The focus is on showing how the compile-time approach can provide significant performance improvements due to the lack of pointer indirections and potential for inlining.

:p What are the key differences between the classic object-oriented decorator implementation and the compile-time template-based decorator implementation discussed?
??x
The classic object-oriented decorator uses dynamic dispatch, involving virtual functions and pointers, which introduces overhead such as vtable lookups. In contrast, the compile-time approach leverages templates to create specialized classes at compile time, avoiding runtime overhead.

Code example to illustrate the classic runtime decorator:
```cpp
class ConferenceTicket {};
class Discounted : public ConferenceTicket {
    double discount;
public:
    Discounted(double d) : discount(d) {}
    virtual Money price() const override { return basePrice * (1 - discount); }
};

class Taxed : public Discounted {
    double taxRate;
public:
    Taxed(double t, ConferenceTicket& ticket) : Discounted(ticket), taxRate(t) {}
    virtual Money price() const override { return discountedPrice * (1 + taxRate); }
};
```
x??

---
#### Performance Comparison Results
Background context: The author presents performance results comparing the classic runtime decorator implementation with a compile-time template-based implementation. The results are normalized to show how much faster the compile-time solution is compared to the runtime solution.

:p What does Table 9-1 in the text indicate about the performance of the compile-time Decorator implementation?
??x
Table 9-1 shows that the compile-time Decorator implementation is significantly faster, taking only about 8% of the time required by the classic runtime implementation for both GCC and Clang. This indicates a performance improvement of more than one order of magnitude.

Example table from the text:
| Compiler       | Classic Decorator | Compile-time Decorator |
|----------------|-------------------|-----------------------|
| GCC            | 1.0               | 0.078067              |
| Clang          | 1.0               | 0.080313              |

x??

---
#### Template Parameters and Class Specialization
Background context: The compile-time approach uses template parameters to define different discount and tax rates, leading to the creation of specialized class instances at compile time.

:p How do templates allow for the implementation of varying discount and tax rates in the compile-time Decorator solution?
??x
Templates enable the specialization of classes based on template arguments. In the given example, `Discounted<0.2,ConferenceTicket>` and `Taxed<0.19,Discounted<0.2,ConferenceTicket>>` are specific instantiations that correspond to different discount and tax rates.

Example code:
```cpp
template <double Discount>
class Discounted {
    // implementation with Discount as a template parameter
};

template <double TaxRate, class Item>
class Taxed {
    // implementation with TaxRate as a template parameter
};
```

x??

---
#### Compilation Time Considerations
Background context: The compile-time solution comes with significant advantages in terms of performance but also presents challenges such as increased compile times and larger executables due to the creation of multiple specialized classes.

:p What are the potential limitations of using templates for implementing decorators at compile time?
??x
The primary limitations include:
- Increased compilation time because all template instantiations must be processed.
- Larger executable size since more code is generated, potentially leading to bloated binaries.
- Reduced runtime flexibility as the decorator combinations need to be defined at compile time.

Example of increased compilation time and binary size:
```cpp
// Many template instantiations result in additional compiled files and larger executables
using DiscountedConferenceTicket = Discounted<0.2, ConferenceTicket>;
using TaxedConferenceTicket = Taxed<0.19, ConferenceTicket>;
using TaxedDiscountedConferenceTicket = Taxed<0.19, Discounted<0.2, ConferenceTicket>>;
```

x??

---
#### Virtual Base Classes and Inheritance
Background context: The text mentions that there are only a few valid reasons to prefer non-public inheritance over composition, with one of them being the use of virtual base classes.

:p Why is it mentioned that virtual base classes can be used as a reason for preferring non-public inheritance in some scenarios?
??x
Virtual base classes allow sharing common base class functionality among derived classes. This is particularly useful when a base class needs to be shared by multiple derived classes, and you want to avoid duplicating code.

Example of using virtual base classes:
```cpp
class VirtualBase {
    // Common functionality
public:
    virtual void commonFunction() {}
};

class DerivedA : public virtual VirtualBase {};
class DerivedB : public virtual VirtualBase {};

// A class can derive from multiple virtual bases to share common functionality
```

x??

---

#### Type Erasure Implementation for Item Class
Background context explaining the concept. The given text discusses a way to implement the Decorator design pattern with value semantics using type erasure, specifically focusing on an `Item` class that wraps other items and allows dynamic modification of behavior at runtime.

The implementation uses a nested `Concept` base class and a template `Model` struct to achieve this. This approach leverages polymorphism through virtual functions while allowing for flexible addition of new types and price modifiers.

:p What is the purpose of using type erasure in the Item class?
??x
Type erasure is used to enable dynamic behavior by hiding the specific implementation details of the wrapped item, thus making it easy to add new kinds of items or modify their behavior (like applying price modifiers) without changing the Item class's interface.

The key components are:
- `Concept` base class: Defines the required interface (`price()` and `clone()`) that any concrete model must implement.
- `Model` template struct: Implements the specific functionality for a given type, forwarding calls to the wrapped item.

This design allows for flexible runtime behavior while maintaining a clean and extensible interface.
x??

---
#### Nested Concept Base Class
Background context explaining the concept. The `Concept` base class in the provided code defines the abstract requirements that any model of an `Item` must adhere to. This is crucial for enabling polymorphic behavior without knowing the concrete type at compile time.

:p What are the two virtual functions defined in the `Concept` base class?
??x
The two virtual functions defined in the `Concept` base class are:
- `virtual Money price() const = 0;`: Returns the price of the item.
- `virtual std::unique_ptr<Concept> clone() const = 0;`: Creates a copy of the current model.

These pure virtual functions ensure that any derived classes (like `Model`) provide these essential behaviors, making the class hierarchy flexible and extensible.
x??

---
#### Model Template Struct
Background context explaining the concept. The `Model` template struct is a concrete implementation of the `Concept` base class for a given type `T`. It provides the actual behavior by forwarding calls to the wrapped item's methods.

:p How does the `Model` struct implement the `price()` function?
??x
The `Model` struct implements the `price()` function by calling the `price()` method on the stored `item_` data member. This is done using a simple forwarding mechanism:

```cpp
Money price() const override {
    return item_.price();
}
```

This implementation ensures that any new type implementing the required interface can be used with the `Item` class.

The logic here is straightforward:
1. It takes the wrapped item's `price()` method and returns its result.
2. This allows for dynamic polymorphism, where different types of items can have their own pricing logic without changing the `Item` class.
x??

---
#### Templated Constructor in Item Class
Background context explaining the concept. The templated constructor in the `Item` class is a key feature that enables flexibility by allowing any type to be wrapped and decorated dynamically.

:p How does the templated constructor of the `Item` class work?
??x
The templated constructor of the `Item` class accepts a move-constructed item of any type `T`, wrapping it in a `Model` struct. This allows for dynamic creation and manipulation of items:

```cpp
template< typename T >
Item( T item )
    : pimpl_( std::make_unique<Model<T>>( std::move(item) ) ) {}
```

Here's the detailed logic:
1. It takes an item of type `T`.
2. It creates a new `Model` instance, passing the moved item to it.
3. The `pimpl_` pointer is assigned this new model.

This constructor ensures that any type can be wrapped and managed by the `Item` class, maintaining flexibility in terms of adding new types or modifying existing ones.
x??

---
#### Clone Functionality
Background context explaining the concept. The `clone()` function is essential for ensuring deep copying when creating copies of items at runtime. It's a part of the `Concept` base class and implemented by the `Model` template struct.

:p How does the `clone()` function in the `Item` class work?
??x
The `clone()` function creates a copy of the current model, which is essential for implementing deep copies when making copies of items:

```cpp
std::unique_ptr<Concept> clone() const override {
    return std::make_unique<Model<T>>(*this);
}
```

Here's how it works:
1. It calls `std::make_unique` to create a new `Model<T>` object.
2. The current model is passed as an argument, effectively cloning the state of the original item.

This ensures that when items are copied, all internal states and modifications are properly replicated, maintaining consistency across copies.
x??

---
#### Price Function Implementation
Background context explaining the concept. The `price()` function in the `Item` class provides a consistent interface for getting the price of an item, regardless of its underlying type.

:p How does the `price()` function in the `Item` class work?
??x
The `price()` function in the `Item` class delegates the actual pricing logic to the wrapped item through the `Concept` base class:

```cpp
Money price() const {
    return pimpl_->price();
}
```

Here's how it works:
1. It calls the `price()` method on the wrapped model (`pimpl_`).
2. This allows different types of items (e.g., with varying pricing logic) to be used interchangeably.

This design ensures that the interface remains simple and consistent while providing the necessary flexibility in underlying behavior.
x??

---

#### Type Erasure Decorator Implementation

This section describes how to implement a decorator pattern using type erasure, where classes like `Discounted` and `Taxed` are implemented without inheriting from any base class. This approach allows combining decorators flexibly.

:p What is the key advantage of implementing `Discounted` and `Taxed` with type erasure?
??x
The key advantage is that both `Discounted` and `Taxed` can be used as if they were normal items, without requiring them to inherit from a base class. This flexibility allows combining decorators arbitrarily using a wrapper class.

```cpp
class Discounted {
public:
    Discounted(double discount, Item item)
        : item_(std::move(item)), factor_(1.0 - discount) {}
    
    Money price() const {
        return item_.price() * factor_;
    }

private:
    Item item_;
    double factor_;
};

class Taxed {
public:
    Taxed(double taxRate, Item item)
        : item_(std::move(item)), factor_(1.0 + taxRate) {}

    Money price() const {
        return item_.price() * factor_;
    }

private:
    Item item_;
    double factor_;
};
```
x??

---
#### Example of Combining Decorators

The example provided demonstrates combining `Discounted` and `Taxed` decorators to create a complex pricing structure.

:p How does the main function demonstrate the combination of decorators?
??x
In the `main` function, a complex item is created by nesting `Discounted` within `Taxed`. The `ConferenceTicket` object is wrapped in `Discounted`, which itself is passed to `Taxed`.

```cpp
int main() {
    // 20 percent discount, 15 percent tax: (499*0.8)*1.15 = 459.08
    Item item(Taxed(0.15, Discounted(0.2, ConferenceTicket{"Core C++", 499.0})));

    Money const totalPrice = item.price();
    // ... return EXIT_SUCCESS;
}
```
x??

---
#### Performance Comparison

The performance of the type erasure decorator implementation is compared with other solutions to ensure it does not significantly degrade performance.

:p What are the performance results for the type erasure decorator implementation?
??x
The type erasure decorator implementation performs similarly to the classic runtime solution. The benchmark shows that the performance numbers are close, indicating that this approach is efficient and maintains good performance without manual memory management.

Table 9-2: Performance results for the Type Erasure Decorator implementation (normalized performance)

| GCC | Clang |
|-----|-------|
| 1.0 | 1.0   |
| 0.078067 | 0.080313 |
| 0.997510 | 0.971875 |

This table shows the normalized performance of different solutions, where a value close to 1 indicates similar performance.
x??

---

#### Performance Considerations of Type Erasure Solution
Performance can sometimes appear marginally better, but this is based on averages from many runs. Don’t over-emphasize these results, as there are other optimizations available for improving performance. 
:p What does the guideline suggest about the performance benefits of the Type Erasure solution?
??x
The guideline suggests that while a small performance improvement might be observed, it should not be heavily relied upon due to the variability in results across multiple runs. There are multiple ways to further optimize the Type Erasure solution as highlighted in "Guideline 33: Be Aware of the Optimization Potential of Type Erasure."
x??

---

#### Runtime Flexibility and Decorator Design Pattern
The Type Erasure solution offers significant runtime flexibility, allowing decisions about wrapping Items in Decorators at runtime based on various factors such as user input or computation results. This results in more versatile Item objects that can be stored together.
:p How does the Type Erasure solution enhance runtime flexibility?
??x
The Type Erasure solution enhances runtime flexibility by enabling dynamic decision-making for wrapping items with decorators, depending on runtime conditions like user inputs or computational outcomes. For instance:
```cpp
Decorator* item = decideBasedOnUserInput() ? new Wrapper(new ConcreteItem) : new ConcreteItem;
```
This flexibility allows storing different types of decorated objects in a single container.
x??

---

#### Compile-Time vs. Runtime Abstraction
Compile-time solutions generally outperform runtime ones but limit runtime flexibility and encapsulation, whereas runtime solutions offer more flexibility at the cost of performance.
:p What is the trade-off between compile-time and runtime abstraction?
??x
The trade-off involves choosing between better performance in compile-time solutions and increased runtime flexibility in runtime solutions. For example:
```cpp
// Compile-time approach (potentially faster but less flexible)
class Item {
    // ...
};

// Runtime approach (more flexible but potentially slower)
class DynamicItem {
    Decorator* wrappedItem;
    public:
    DynamicItem(Item& item) : wrappedItem(new Wrapper(item)) {}
};
```
The compile-time approach has fixed structures and is optimized, while the runtime approach allows dynamic changes at execution time.
x??

---

#### Value Semantics vs. Reference Semantics
Value semantics solutions are preferred over reference semantics because they ensure that modifications to one object do not affect others, thus reducing recompilation needs and improving encapsulation.
:p What does the guideline recommend regarding value semantics?
??x
The guideline recommends using value semantics to create simpler, more comprehensible user code. Value semantics help in achieving better compile times by encapsulating changes more strongly, thereby avoiding unnecessary recompilations:
```cpp
class Item {
    // ...
};

Item item1 = new Item();
Item item2 = item1;  // Copies the state rather than sharing a reference.
```
This approach ensures that each object maintains its own state, leading to cleaner and safer code practices.
x??

---

#### Strategy Pattern Implementation
The Strategy pattern can be implemented using null objects, which represent neutral behavior. This makes them suitable for implementing default strategies in your design patterns.
:p What is the role of a null object in strategy implementation?
??x
A null object serves as an entity with no or neutral behavior, making it useful when you want to provide a fallback or default strategy implementation:
```cpp
class NullStrategy : public Strategy {
public:
    void execute() override {}
};

class ConcreteStrategy : public Strategy {
    // ...
};
```
In this setup, the `NullStrategy` can be used as a placeholder, ensuring that no operation is performed when it's active.
x??

---

#### Curiously Recurring Template Pattern (CRTP)
For C++20 and earlier versions without concepts support, you might use CRTP to introduce static type categories. This pattern helps in achieving compile-time polymorphism.
:p What is the Curiously Recurring Template Pattern (CRTP)?
??x
The Curiously Recurring Template Pattern (CRTP) allows for compile-time polymorphism by having a derived class template parameterized with its base class:
```cpp
template <typename Derived>
class Base {
public:
    void doSomething() {
        static_cast<Derived*>(this)->specialBehavior();
    }
};

class Concrete : public Base<Concrete> {
public:
    void specialBehavior() override {}
};
```
This pattern ensures that `Base` can call member functions of the derived class at compile time, which is useful for implementing design patterns like decorators.
x??

---

#### Tax Calculation Example
The example of tax calculation in the text highlights the limitations of simple solutions, as they may not cover real-world complexities and could lead to incorrect calculations. 
:p What issue does the tax calculation example illustrate?
??x
The tax calculation example illustrates that simple solutions can be insufficient for practical applications due to their oversimplification. In reality, taxes are much more complex and prone to errors if not carefully implemented:
```cpp
class ConferenceTicket {
    float price;
public:
    ConferenceTicket(float p) : price(p) {}
    float price() { return price; }
};

// Incorrect tax application example
class Taxed<Ticket> : public Ticket {
private:
    static const float TAX_RATE = 0.19f;
public:
    Taxed(Ticket& t) : Ticket(t) {}
    float price() override { return Ticket::price() * (1 + TAX_RATE); }
};
```
This example shows how naive tax application might lead to inaccuracies, necessitating more robust and comprehensive implementations.
x??

---

