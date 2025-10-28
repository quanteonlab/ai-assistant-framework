# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 26)

**Starting Chapter:** The Pimpl Idiom

---

#### Pimpl Idiom Explanation
Background context: The Pimpl idiom is a technique used to hide implementation details and ensure binary stability (ABI) by separating the interface from the implementation. This separation allows the class's internal structure to change without affecting its users, provided that the public interface remains unchanged.
:p What is the Pimpl idiom and why is it useful?
??x
The Pimpl idiom involves encapsulating the private members of a class into an internal class (called Impl), while keeping only a pointer or smart pointer (usually std::unique_ptr) to this implementation in the original class. This approach ensures that any changes in the implementation do not affect the binary compatibility and ease of maintenance.
```cpp
// Example of Pimpl idiom in C++
class Person {
public:
    // Public interface methods
private:
    struct Impl;  // The internal implementation details are hidden here
    std::unique_ptr<Impl> pimpl_;  // Pointer to the implementation
};
```
x??

---

#### Implementation Details Hiding
Background context: By using the Pimpl idiom, all the private members and functions of a class are moved into an internal struct (Impl), while only keeping a smart pointer to this struct in the original class. This allows changes to the implementation details without affecting external users.
:p How does hiding implementation details work in the Pimpl idiom?
??x
In the Pimpl idiom, the private members and functions that are part of the internal implementation are moved into a nested struct (Impl). The original class then holds only a smart pointer to this Impl instance. Any changes made to the Impl struct will not affect the external interface as long as the public methods remain consistent.
```cpp
// Example of hiding implementation details in Person class
struct Person::Impl {
    std::string forename;
    std::string surname;
    // ... other data members and functions
};

class Person {
public:
    // Public methods and constructors
private:
    struct Impl;  // Hidden implementation detail
    std::unique_ptr<Impl> pimpl_;  // Pointer to the implementation details
};
```
x??

---

#### Constructor Implementation
Background context: In the Pimpl idiom, the constructor of the main class initializes a smart pointer (std::unique_ptr) with an instance of the internal struct. This involves dynamic memory allocation.
:p How does the constructor in the Pimpl idiom initialize the smart pointer?
??x
The constructor of the main class creates and initializes the smart pointer to an instance of the nested Impl struct using `std::make_unique()`. This function allocates memory for the Impl object on the heap and returns a std::unique_ptr managing that memory.
```cpp
// Constructor in Person class with Pimpl idiom
Person::Person()
    : pimpl_{ std::make_unique<Impl>() }  // Initialize the smart pointer
{}
```
x??

---

#### Destructor Implementation
Background context: Although `std::unique_ptr` handles most of the cleanup, it is still necessary to manually define the destructor for the main class. This is because if no custom destructor is provided, the compiler-generated destructor will include the destruction of the `std::unique_ptr`, which can cause issues when included in headers.
:p Why must we manually implement the destructor in the Pimpl idiom?
??x
Even though `std::unique_ptr` manages the memory and handles its cleanup, it is necessary to define a custom destructor for the main class. This is because if no custom destructor is provided, the compiler-generated destructor will include the destruction of the `std::unique_ptr`. Including this in headers can cause issues due to unnecessary header re-instantiations.
```cpp
// Custom destructor in Person class with Pimpl idiom
Person::~Person() = default;  // Define a custom destructor explicitly
```
x??

---

#### Copy Constructor and Assignment Operator
Background context: In the Pimpl idiom, it is essential to correctly handle copy and move semantics to ensure that the smart pointer to the Impl struct is properly copied or moved.
:p How are the copy constructor and assignment operator implemented in the Pimpl idiom?
??x
The copy constructor and assignment operator for the main class need to be implemented to correctly copy or move the `std::unique_ptr` to the Impl struct. This involves using `std::make_unique()` and `operator=` on `std::unique_ptr`.
```cpp
// Copy constructor in Person class with Pimpl idiom
Person::Person(Person const& other)
    : pimpl_{ std::make_unique<Impl>(*other.pimpl_) }  // Copy construction of smart pointer
{}

// Assignment operator in Person class with Pimpl idiom
Person& Person::operator=(Person const& other) {
    *pimpl_ = *other.pimpl_;  // Use assignment operator on the unique_ptr
    return *this;
}
```
x??

---

#### Access Function Implementation
Background context: Access functions (public methods) in the Pimpl idiom typically involve accessing members of the smart pointer to the Impl struct.
:p How are access functions implemented in the Pimpl idiom?
??x
Access functions in the Pimpl idiom simply dereference the `std::unique_ptr` and access the corresponding data members or functions from the Impl struct. This ensures that the implementation details remain hidden while providing a clean interface to the users of the class.
```cpp
// Access function in Person class with Pimpl idiom
int Person::year_of_birth() const {
    return pimpl_->year_of_birth;  // Access year_of_birth from Impl through pimpl_
}
```
x??

---

#### Handling Incomplete Types and std::unique_ptr
Background context explaining how `std::unique_ptr` interacts with incomplete types, specifically when using PIMPL (Pointer-to-implementation) idiom. The issue arises because `std::unique_ptr` requires a fully defined type to be able to call its member functions or operators.
:p How does the implementation of the Person class handle the use of `std::unique_ptr` and incomplete types?
??x
The solution involves declaring the destructor in the header file and defining it in the source file using `=default`. Additionally, implementing copy constructor and assignment operator. The move constructor uses `std::make_unique()` to ensure proper memory allocation.
```cpp
class Person {
public:
    ~Person() = default; // Declaration in header

private:
    std::unique_ptr<Impl> pimpl_; // Incomplete type used here
};

// Definition in source file
Person::~Person() {} // Default definition

// Copy constructor and assignment operator implementation
Person(const Person& other) : pimpl_(other.pimpl_->clone()) {}
Person& operator=(const Person& other) {
    if (this != &other) {
        pimpl_.reset(other.pimpl_->clone());
    }
    return *this;
}

// Move constructor implementation
Person(Person&& other) noexcept(false) : pimpl_{std::make_unique<Impl>(std::move(other.pimpl_))} {}
```
x??

---

#### Bridge Design Pattern vs. Strategy Design Pattern
Background context explaining the key difference between these two design patterns, focusing on how data members are initialized.
:p How can you differentiate between the Bridge and Strategy design patterns based on their implementation details?
??x
The primary difference lies in how behavior is configured: Strategy requires setting up behavior via a constructor or setter function from outside, while Bridge initializes it internally. For example:
```cpp
// Strategy Design Pattern
class DatabaseEngine {
public:
    virtual ~DatabaseEngine() = default;
};

class Database : public StrategyInterface { // Assuming StrategyInterface has `std::unique_ptr<DatabaseEngine>`
public:
    explicit Database(std::unique_ptr<DatabaseEngine> engine);
private:
    std::unique_ptr<DatabaseEngine> engine_;
};

// Bridge Design Pattern
class Database {
public:
    explicit Database();
private:
    std::unique_ptr<DatabaseEngine> pimpl_; // Implementation detail set internally
};

Database::Database() : pimpl_{std::make_unique<ConcreteDatabaseEngine>(/*some arguments*/)} {}
```
x??

---

#### PIMPL Idiom with std::unique_ptr
Background context explaining the use of `std::unique_ptr` in conjunction with PIMPL to hide implementation details. The key challenge is handling incomplete types when implementing functions that require them.
:p How does the PIMPL idiom work with `std::unique_ptr` and what are some considerations?
??x
PIMPL (Pointer-to-implementation) hides the implementation of a class by using a pointer to another class as a member. With `std::unique_ptr`, you must ensure that the type pointed to is fully defined at the point where it's used. This can be challenging in header files if the definition is not available.
```cpp
class Person {
public:
    ~Person(); // Declaration only

private:
    std::unique_ptr<Impl> pimpl_; // Implementation details hidden
};

// Definition in source file
Person::~Person() {}
```
x??

---

#### Database Example - Strategy Design Pattern
Background context explaining the difference between Strategy and Bridge patterns through a database example. The Strategy pattern is demonstrated where behavior can be configured from outside.
:p In the provided code snippet, which design pattern does the `Database` class implement?
??x
The `Database` class implements the Strategy design pattern because it takes a `std::unique_ptr<DatabaseEngine>` as an argument in its constructor and passes behavior configuration to this pointer. This allows for flexible behavior setup from outside.
```cpp
class DatabaseEngine {
public:
    virtual ~DatabaseEngine() = default;
};

class Database : public StrategyInterface { // Assuming StrategyInterface has `std::unique_ptr<DatabaseEngine>`
public:
    explicit Database(std::unique_ptr<DatabaseEngine> engine);
private:
    std::unique_ptr<DatabaseEngine> engine_;
};

// Example of the constructor
Database::Database(std::unique_ptr<DatabaseEngine> engine) : engine_{std::move(engine)} {}
```
x??

---

#### Database Example - Bridge Design Pattern
Background context explaining how the Bridge pattern differs from the Strategy pattern in terms of behavior setup and physical dependencies.
:p In another example, which design pattern does the `Database` class implement?
??x
The `Database` class implements the Bridge design pattern because it initializes its internal pointer to a concrete implementation (`ConcreteDatabaseEngine`) internally. This shows that the class is logically coupled with a specific implementation but physically decoupled via an abstraction.
```cpp
class Database {
public:
    explicit Database();
private:
    std::unique_ptr<DatabaseEngine> pimpl_;
};

// Example of the constructor
Database::Database() : pimpl_{std::make_unique<ConcreteDatabaseEngine>(/*some arguments*/)} {}
```
x??

---

#### Performance Overhead of Bridge Pattern
Bridge pattern decouples an abstraction from its implementation so that both can vary independently. However, this comes with certain performance costs due to additional indirection and other factors.

:p How does the Bridge design pattern introduce a performance penalty?
??x
The Bridge design pattern introduces several performance penalties:

1. **Indirection Penalty**: The use of a pimpl (Pointer-to-implement) pointer adds an extra level of indirection, which can make access to implementation details more expensive.
2. **Virtual Function Call Overhead**: If virtual functions are used for abstraction and implementation separation, each call incurs the overhead of a virtual function table lookup.
3. **Inlining Issues**: Even simple function calls accessing data members cannot be inlined due to the separation between interface and implementation, which can degrade performance.
4. **Dynamic Memory Allocation**: Every instance creation results in dynamic memory allocation, adding to the overhead.

C++/Java code examples:
```cpp
class Abstraction {
public:
    virtual void operation() = 0;
};

class RefinedAbstraction : public Abstraction {
private:
    std::unique_ptr<Implementation> pimpl_;
public:
    RefinedAbstraction() : pimpl_(std::make_unique<Implementation>()) {}

    void operation() override {
        pimpl_->doSomething();
    }
};
```
x??

---
#### Guideline 29: Be Aware of Bridge Performance Gains and Losses
It's important to measure the actual performance impact when using the Bridge pattern. The overhead can sometimes be negligible, especially if the underlying implementation is already expensive.

:p In what scenarios might the performance loss from the Bridge pattern be minimal?
??x
The performance loss from the Bridge pattern may not be significant in scenarios where:

- **Expensive Operations**: If the underlying operations are already costly (e.g., system calls), the additional overhead might be negligible.
- **Optimized Systems**: In well-optimized systems with high-performance requirements, careful benchmarking is necessary to determine if the benefits outweigh the costs.

:p How can one decide whether to use a Bridge pattern or not?
??x
One should decide whether to use a Bridge pattern on a case-by-case basis and back it up with performance benchmarks. Factors include:

- **Implementation Complexity**: If the implementation performs slow, expensive tasks.
- **Performance Requirements**: For systems where performance is critical, thorough testing can help determine if the overhead is acceptable.

:p What does "always depends" mean in this context?
??x
"Always depends" means that whether to use a Bridge pattern should be evaluated based on specific circumstances and requirements. It's not a one-size-fits-all solution; it needs to fit into the broader context of the project’s performance and complexity constraints.

:p How does memory overhead from pimpl pointer affect performance?
??x
Memory overhead due to the pimpl pointer can increase the overall memory usage, which might indirectly impact performance in terms of increased garbage collection or memory allocation/deallocation time. This overhead is especially noticeable when creating many objects.

:x??

---
#### Complexity Increase with Bridge Pattern
Using the Bridge pattern increases code complexity, particularly in the internal implementation. While user code remains simple and readable, internal code becomes more complex due to separation concerns.

:p How does the Bridge pattern affect code simplicity?
??x
The Bridge pattern improves modularity by separating the abstraction from its implementation, but at the cost of increased internal complexity. User-facing interfaces remain simple, while the internal implementation details become more intricate, making it harder to maintain and understand the codebase.

:x??

---
#### Indirect Access vs Direct Access
Indirect access through a pimpl pointer in Bridge pattern is less efficient compared to direct access, as it introduces an additional level of indirection which can slow down function calls.

:p What is the difference between indirect and direct access in the context of Bridge pattern?
??x
In the Bridge pattern, indirect access involves using a pimpl pointer to hide implementation details. This means accessing data or methods through this pointer rather than directly. Direct access would involve having member functions that operate on internal data members without any indirection.

Example:
```cpp
class Implementation {
public:
    void doSomething() { std::cout << "Doing something.\n"; }
};

// Indirect Access (Bridge Pattern)
class RefinedAbstraction : public Abstraction {
private:
    std::unique_ptr<Implementation> pimpl_;
public:
    RefinedAbstraction() : pimpl_(std::make_unique<Implementation>()) {}

    void operation() override {
        pimpl_->doSomething();
    }
};

// Direct Access
class SimpleAbstraction {
private:
    Implementation impl_;
public:
    void operation() { impl_.doSomething(); }
};
```

:x??

---

