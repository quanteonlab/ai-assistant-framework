# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 34)

**Starting Chapter:** A Classic Implementation of the Decorator Design Pattern

---

#### Item Base Class and DecoratedItem
Background context: The `Item` base class is an abstraction for all possible items, defining a pure virtual function `price()` to query the price. The `DecoratedItem` class is a concrete implementation that stores another `Item` and provides protected methods to access it.

:p What is the purpose of the `DecoratedItem` class?
??x
The `DecoratedItem` class serves as a wrapper around another item, allowing for additional functionality such as applying discounts or taxes without modifying the original item's code. It implements the Decorator pattern by providing a flexible mechanism to add responsibilities dynamically.
```cpp
class DecoratedItem : public Item {
public:
    explicit DecoratedItem(std::unique_ptr<Item> item)
        : item_(std::move(item)) {
        if (!item_) { throw std::invalid_argument("Invalid item"); }
    }

protected:
    Item&       item() { return *item_; }
    Item const& item() const { return *item_; }

private:
    std::unique_ptr<Item> item_;
};
```
x??

---

#### CppBook and ConferenceTicket Implementations
Background context: `CppBook` and `ConferenceTicket` are specific implementations of the `Item` class. They provide a concrete implementation for the price() function, representing different types of items.

:p What do `CppBook` and `ConferenceTicket` represent?
??x
`CppBook` represents a C++ book with a title and a specified price. `ConferenceTicket` represents a conference ticket with a name and a specified price.
```cpp
class CppBook : public Item {
public:
    CppBook(std::string title, Money price)
        : title_{std::move(title)}, price_{price} {}
    std::string const& title() const { return title_; }
    Money price() const override { return price_; }

private:
    std::string title_;
    Money price_{};
};

class ConferenceTicket : public Item {
public:
    ConferenceTicket(std::string name, Money price)
        : name_{std::move(name)}, price_{price} {}
    std::string const& name() const { return name_; }
    Money price() const override { return price_; }

private:
    std::string name_;
    Money price_{};
};
```
x??

---

#### Discounted Class Implementation
Background context: The `Discounted` class is a decorator that applies a discount to the wrapped item. It computes the discounted price by multiplying the original item's price with a factor derived from the discount value.

:p How does the `Discounted` class apply a discount?
??x
The `Discounted` class applies a discount by computing a factor (1 - discount) and then using this factor to modify the wrapped item's price. If the constructor receives an invalid discount, it throws an exception.
```cpp
class Discounted : public DecoratedItem {
public:
    Discounted(double discount, std::unique_ptr<Item> item)
        : DecoratedItem(std::move(item)), factor_(1.0 - discount) {
        if (!std::isfinite(discount) || discount < 0.0 || discount > 1.0) {
            throw std::invalid_argument("Invalid discount");
        }
    }

    Money price() const override {
        return item().price() * factor_;
    }

private:
    double factor_{};
};
```
x??

---

#### Taxed Class Implementation
Background context: The `Taxed` class is similar to the `Discounted` class but applies a tax instead of a discount. It uses a similar approach to modify the price by applying a tax rate.

:p How does the `Taxed` class apply a tax?
??x
The `Taxed` class applies a tax by computing a factor (1 + taxRate) and then using this factor to modify the wrapped item's price. If the constructor receives an invalid tax rate, it throws an exception.
```cpp
class Taxed : public DecoratedItem {
public:
    Taxed(double taxRate, std::unique_ptr<Item> item)
        : DecoratedItem(std::move(item)), factor_(1.0 + taxRate) {
        if (!std::isfinite(taxRate) || taxRate < 0.0) {
            throw std::invalid_argument("Invalid tax");
        }
    }

    Money price() const override {
        return item().price() * factor_;
    }

private:
    double factor_{};
};
```
x??

---

#### Main Function Example
Background context: The `main` function demonstrates how to combine different decorators (Discounted and Taxed) with specific items (CppClass, ConferenceTicket). It shows the application of multiple decorators in a hierarchical manner.

:p How are multiple decorators applied in the main function?
??x
In the `main` function, multiple decorators are applied hierarchically. First, a `Taxed` decorator is used to apply a 7% tax to a `CppClass`. Then, another `Discounted` decorator is added with an 80% discount and a 19% tax on top of that for a `ConferenceTicket`.

```cpp
int main() {
    // 7 percent tax: 19*1.07 = 20.33
    std::unique_ptr<Item> item1(std::make_unique<Taxed>(0.07, 
        std::make_unique<CppBook>("Effective C++", 19.0)));

    // 20 percent discount, 19 percent tax: (999*0.8)*1.19 = 951.05
    std::unique_ptr<Item> item2(std::make_unique<Taxed>(0.19,
        std::make_unique<Discounted>(0.2, 
            std::make_unique<ConferenceTicket>("CppCon", 999.0))));

    Money const totalPrice1 = item1->price(); // Results in 20.33
    Money const totalPrice2 = item2->price(); // Results in 951.05

    return EXIT_SUCCESS;
}
```
x??

---

#### Taxed Decorator Example for CppBook and ConferenceTicket
In this example, we see how a decorator can be used to wrap additional functionality around an object. Specifically, we create a taxed version of a C++ book and a discounted and taxed conference ticket.

:p What is the process for creating a taxed C++ book using decorators?
??x
To create a taxed C++ book, you first instantiate a `CppBook` object. Then, you wrap it with a `Taxed` decorator that applies a 7% tax to the book's price. The resulting `item1` represents a taxed C++ book.

```cpp
// Assuming CppBook and Taxed classes are defined
CppBook cppBook;
Taxed taxedCppBook(cppBook, 0.07);
```
x??

---

#### Discounted Decorator for ConferenceTicket
This example demonstrates the use of a decorator to apply discounts to an object before applying additional decorators like taxes.

:p How is a discounted and taxed C++ conference ticket created using decorators?
??x
To create a discounted and taxed C++ conference ticket, you first instantiate a `ConferenceTicket` representing CppCon. Then, you wrap it with a `Discounted` decorator that applies a 20% discount to the ticket's price. Finally, another `Taxed` decorator is used to apply a 19% tax on top of the discounted ticket.

```cpp
// Assuming ConferenceTicket and decorators are defined
ConferenceTicket cppConTicket;
Discounted discountedTicket(cppConTicket, 0.20);
Taxed taxedDiscountedTicket(discountedTicket, 0.19);
```
x??

---

#### STL Allocator Decorator Example: std::pmr::monotonic_buffer_resource
This example illustrates the use of the decorator pattern in C++17's Standard Template Library (STL) allocators to create flexible memory management strategies.

:p How does `std::pmr::monotonic_buffer_resource` work?
??x
The `std::pmr::monotonic_buffer_resource` is a decorator that wraps another allocator. It uses an internal buffer and only dispenses chunks of this buffer when the underlying container (like `std::vector`) requests memory. If the buffer runs out, it throws a `std::bad_alloc` exception.

```cpp
// Example configuration with raw byte array
std::array<std::byte, 1000> raw;
std::pmr::monotonic_buffer_resource buffer{ raw.data(), raw.size(), std::pmr::null_memory_resource() };
std::pmr::vector<std::pmr::string> strings{ &buffer };
```
x??

---

#### CustomAllocator in Decorator Pattern
This example shows how to create a custom allocator that can be used as a decorator for other allocators, providing flexible memory management strategies.

:p How does the `CustomAllocator` class work?
??x
The `CustomAllocator` class is derived from `std::pmr::memory_resource` and acts as a decorator. It owns a pointer to another allocator (upstream allocator) and overrides the virtual functions required by C++17 allocators: `do_allocate`, `do_deallocate`, and `do_is_equal`.

```cpp
// CustomAllocator definition
class CustomAllocator : public std::pmr::memory_resource {
public:
    CustomAllocator(std::pmr::memory_resource* upstream)
        : upstream_{ upstream } {}

private:
    void* do_allocate(size_t bytes, size_t alignment) override;
    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override;
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

    std::pmr::memory_resource* upstream_{};
};
```
x??

---

#### CustomAllocator Overview
Background context explaining how CustomAllocators can be used to extend standard library behavior without modifying existing code. This involves understanding memory management and allocation strategies.

:p How is a `CustomAllocator` utilized in extending allocator behavior?
??x
A `CustomAllocator` allows you to customize the way memory is allocated and deallocated, providing an extensible interface for handling allocations. By constructing it with an existing allocator like `std::pmr::new_delete_resource()`, you can plug it into a custom resource manager like `std::pmr::monotonic_buffer_resource`. This non-intrusive approach enables you to extend the behavior of memory management without altering any other parts of your program.

```cpp
#include <CustomAllocator.h>

int main() {
    CustomAllocator custom_allocator{ std::pmr::new_delete_resource() };
    
    std::pmr::monotonic_buffer_resource buffer{ &custom_allocator };
}
```
x??

---

#### Difference Between Decorator and Adapter
Background context explaining the purpose and use cases of both design patterns, highlighting how they differ in their approach to modifying interfaces.

:p How do the Decorator and Adapter design patterns differ?
??x
The Decorator pattern allows for adding responsibilities or functionality to objects dynamically by wrapping them with additional objects. It preserves the original interface, focusing on extending capabilities without altering the existing interface structure. The Adapter pattern, on the other hand, changes an interface so it conforms to another expected interface but does not add any new functionality; its primary goal is to make two interfaces compatible.

For example:
```cpp
// Decorator Example
class Text {
public:
    virtual ~Text() = default;
    virtual void show() const = 0;
};

class Bold : public Text {
private:
    std::shared_ptr<Text> wrapped_text_;
public:
    Bold(std::shared_ptr<Text> text) : wrapped_text_(text) {}
    void show() const override { 
        // Add bold formatting
        std::cout << "<b>";
        wrapped_text_->show();
        std::cout << "</b>";
    }
};

// Adapter Example
class NewFormat {
public:
    void newShow() { /* ... */ }
};
class OldFormatAdapter : public Text {
private:
    std::unique_ptr<NewFormat> new_format_;
public:
    OldFormatAdapter(std::unique_ptr<NewFormat> format) : new_format_(std::move(format)) {}
    void show() const override {
        // Convert to old format
        std::cout << "<old>";
        new_format_->newShow();
        std::cout << "</old>";
    }
};
```
x??

---

#### Strategy Design Pattern
Background context explaining the purpose and use cases of the Strategy design pattern, emphasizing its ability to encapsulate algorithmic behavior.

:p How does the Strategy design pattern help in managing algorithms?
??x
The Strategy design pattern decouples an algorithm from the entity that uses it. It enables you to define a family of algorithms, encapsulate each one, and make them interchangeable within the same context. This allows for flexible and extensible code where different strategies can be easily swapped out based on requirements.

For example:
```cpp
class PriceStrategy {
public:
    virtual ~PriceStrategy() = default;
    virtual Money update(Money price) const = 0;
};

class DiscountedPriceStrategy : public PriceStrategy {
public:
    void update(Money price) const override {
        // Apply discount logic
        return (price - (price * 15 / 100));
    }
};
```
x??

---

#### Combining Decorator and Strategy Patterns
Background context explaining how combining the Decorator and Strategy patterns can leverage their strengths to enhance functionality.

:p How can you combine the Decorator and Strategy patterns?
??x
You can use the Decorator pattern to wrap a `PriceStrategy` object, allowing for fine-grained configuration of pricing logic. The `DecoratedPriceStrategy` class can add additional responsibilities or modify existing ones dynamically, while the `DiscountedPriceStrategy` provides specific implementation details encapsulated within its strategy.

For example:
```cpp
class DecoratedPriceStrategy : public PriceStrategy {
private:
    std::unique_ptr<PriceStrategy> priceModifier_;
public:
    DecoratedPriceStrategy(std::unique_ptr<PriceStrategy> modifier) : priceModifier_(std::move(modifier)) {}
    Money update(Money price) const override {
        // Apply base strategy
        Money modified_price = priceModifier_->update(price);
        // Add additional logic if needed
        return (modified_price - (modified_price * 10 / 100));
    }
};

class DiscountedPriceStrategy : public DecoratedPriceStrategy {
public:
    DiscountedPriceStrategy() : DecoratedPriceStrategy(std::make_unique<DiscountedPriceStrategy>()) {}
};
```
x??

---

