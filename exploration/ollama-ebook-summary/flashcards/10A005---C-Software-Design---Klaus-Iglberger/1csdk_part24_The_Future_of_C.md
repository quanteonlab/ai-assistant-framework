# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 24)

**Starting Chapter:** The Future of CRTP A Comparison Between CRTP and C20 Concepts

---

#### CRTP vs C++20 Concepts: Compile-Time Polymorphism
Background context explaining that CRTP provides compile-time polymorphism, which is useful for static type abstractions but lacks runtime flexibility and can be restrictive. C++20 concepts offer a more flexible and nonintrusive alternative.
:p How does CRTP differ from C++20 concepts in terms of implementation and flexibility?
??x
CRTP (Curiously Recurring Template Pattern) requires derived classes to implement certain interfaces, making it less flexible but purely compile-time polymorphic. C++20 concepts, on the other hand, provide a more flexible and nonintrusive way to define requirements for types using templates.
```cpp
// Example of CRTP
template<typename T>
class Base {
public:
    void foo(T t) { /* implementation */ }
};

class Derived : public Base<Derived> {
    // must implement the interface
};
```
x??

---

#### C++20 Concepts: Defining Requirements for Types
Background context explaining how C++20 concepts can define requirements for types in a nonintrusive manner, allowing functions and operators to be restricted to specific type sets.
:p How can we use C++20 concepts to restrict the `operator<<` to only dense vector types?
??x
C++20 concepts allow us to define requirements for types using constraints. By defining a concept like `DenseVector`, we can constrain the `operator<<` to work only with types that meet these requirements.
```cpp
template<typename T>
concept DenseVector = requires (T t, size_t index) {
    { t.size() };
    { t[index] };
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
} &&
requires (const T t, size_t index) {
    { t[index] };
    { t.begin() } -> std::same_as<const typename T::iterator>;
    { t.end() } -> std::same_as<const typename T::iterator>;
};

template<DenseVector VectorT>
std::ostream& operator<<(std::ostream& os, const VectorT& vector) {
    // implementation
}
```
x??

---

#### Nonintrusive vs Intrusive Approaches with Concepts
Background context explaining that while concepts can be nonintrusive, some situations may require an intrusive approach. A tag class and type traits are introduced to achieve this.
:p How does the introduction of a tag class and type trait improve the concept-based solution?
??x
The introduction of a tag class and type trait allows for both nonintrusive and intrusive approaches to defining types that conform to certain requirements. This is achieved by allowing classes like `DynamicVector` to inherit from a tag class, while others can specialize the type trait.

```cpp
// Using tag class and type traits
struct DenseVectorTag {};

template<typename T>
concept DenseVector = requires (T t, size_t index) {
    // ... requirements ...
} && std::is_base_of_v<DenseVectorTag, T>;

template<typename T>
class DynamicVector : private DenseVectorTag {
    // implementation
};

// Type trait for StaticVector
template<typename T, size_t Size>
struct IsDenseVector : public std::true_type {};

template<typename T, size_t Size>
concept DenseVector = requires (T t, size_t index) {
    // ... requirements ...
} && IsDenseVector<T, Size>::value;
```
x??

---

#### SRP and Separation of Concerns
Background context explaining the principle of separating responsibilities to achieve cleaner designs. In this case, it involves using a type trait for compile-time checks.
:p How does applying the Single Responsibility Principle (SRP) help in this scenario?
??x
Applying the SRP helps by separating the concern of determining if a type is a dense vector into its own class (`IsDenseVector`). This allows `DenseVector` to focus solely on defining requirements without directly checking types, making the code cleaner and more maintainable.

```cpp
// Applying SRP with IsDenseVector
struct DenseVectorTag {};

template<typename T>
concept DenseVector = requires (T t, size_t index) {
    // ... requirements ...
} && std::is_base_of_v<DenseVectorTag, T>;

template<typename T>
class DynamicVector : private DenseVectorTag {
    // implementation
};

// Type trait for StaticVector
template<typename T, size_t Size>
struct IsDenseVector : public std::true_type {};

template<typename T, size_t Size>
concept DenseVector = requires (T t, size_t index) {
    // ... requirements ...
} && IsDenseVector<T, Size>::value;
```
x??

---

#### CRTP Drawbacks and Alternatives

Background context: The Template Method Pattern (CRTP) is a design pattern that uses inheritance and templates to achieve compile-time polymorphism. However, it has certain drawbacks such as the lack of a common base class, the quick spreading of template code, and the restriction to compile-time polymorphism. C++20 introduces concepts which provide an easier and nonintrusive alternative.

:p What are the main drawbacks of using CRTP?
??x
The main drawbacks of using CRTP include:
- Lack of a common base class: This makes it harder to implement certain functionalities that would be simpler if a common base class existed.
- Quick spreading of template code: When you use CRTP, derived classes can become heavily dependent on the base template, leading to code bloat and potentially making the design more complex than necessary.
- Restriction to compile-time polymorphism: While useful in some cases, it limits the flexibility that runtime polymorphism offers.

C++20 concepts offer a way to address these issues by providing a more powerful abstraction mechanism without the drawbacks of CRTP.
x??

---

#### Applying CRTP for Static Type Categories

Background context: The CRTP design pattern can be used to introduce static type categories, which help in creating distinct types with semantic meaning. This is useful when you want to enforce certain rules or operations on derived classes.

:p How can CRTP be used to introduce static type categories?
??x
CRTP can be used by defining a template class that acts as an abstract base class for other specialized classes. The derived class provides the actual implementation, and the base class ensures that only specific types are allowed.

For example:
```cpp
template<typename Derived>
class Base {
public:
    // Force the Derived type to provide implementation of this method
    static void func() {
        static_cast<Derived*>(nullptr)->funcImpl();
    }
protected:
    // Forbidden copy constructor and assignment operator
    Base(const Base&) = delete;
    Base& operator=(const Base&) = delete;

private:
    // Private default constructor
    Base() = default;
};

// Derived class must implement a function that the base class calls.
class Derived : public Base<Derived> {
public:
    void funcImpl() override {
        // Implementation here
    }
};
```

Here, `Base` acts as an abstract base class and enforces certain rules on its derived class. Only types that provide the implementation of `func()` can be used as derived classes.
x??

---

#### Using CRTP for Static Mixin Classes

Background context: CRTP can also be used to implement static mixin classes, which extend the functionality of a type without modifying it directly.

:p How does CRTP facilitate implementing static mixin classes?
??x
CRTP allows you to define a template class that provides additional methods or behaviors to its derived class. This approach avoids modifying the original base class and allows for adding new functionalities in a non-intrusive way.

For example:
```cpp
template <typename T>
class Logging {
public:
    static void log(const T& value) {
        std::cout << "Logging: " << value << "\n";
    }
};

// Use CRTP to add logging behavior to any class.
class MyClass : public Logging<MyClass> {
public:
    int value = 42;

    // Method that logs the value
    void printValue() {
        log(value); // Uses the static method from Logging mixin class
    }
};
```

Here, `MyClass` is extended with logging behavior through the CRTP pattern. The `Logging` template provides a `log` function that can be used by any type derived from it.
x??

---

#### Strong Type Motivation

Background context: The provided code snippet introduces the concept of `StrongType`, which is a class template that wraps another type to create distinct, strongly typed versions for specific purposes like distance measurements or names. This helps in preventing accidental misuse and ensures semantic correctness.

:p What is the purpose of using `StrongType` in this example?
??x
The purpose of using `StrongType` is to enforce strong typing, which means that even though different types (like `long`, `double`, etc.) can be used under the hood, they are treated as distinct types with specific meanings. This prevents accidental misuse and ensures semantic correctness.

For instance:
- You can define `Meter<long>` and `Kilometer<long>` to represent lengths in meters and kilometers respectively.
- These types cannot be combined incorrectly (e.g., adding a meter value to a kilometer value or concatenating surnames).

This strong typing helps prevent bugs that might arise from accidentally treating different types the same way, such as performing arithmetic operations on incompatible types.
x??

---

#### Problem with Direct Operator Overloading

Background context explaining why direct operator overloading for all strong types was problematic. The implementation would violate DRY (Don't Repeat Yourself) principle because it would require identical code for each type.

:p What is the issue with directly implementing an addition operator for multiple strong types like Meter, Kilometer, etc.?
??x
The main issue is that directly overloading the addition operator in the StrongType class template would lead to repeated code, one for each specific type (Meter, Kilometer, etc.). This violates the DRY principle because the implementation logic would be identical across multiple classes.

For example:
```cpp
template<typename T, typename Tag>
StrongType<T,Tag> operator+(const StrongType<T,Tag>& a, const StrongType<T,Tag>& b) {
    return StrongType<T,Tag>(a.get() + b.get());
}
```
This approach is not scalable and would result in redundancy if you have many different strong types.

x??

---

#### CRTP (Curiously Recurring Template Pattern)

Background context explaining the role of CRTP in providing operations to specific instantiations of a class template. CRTP allows adding operations selectively, making the code more flexible and maintainable.

:p How does CRTP solve the problem with direct operator overloading?
??x
CRTP solves the problem by allowing you to add operations selectively to specific instantiations of a class template without duplicating code. Instead of defining operators directly in the base `StrongType` class, we define mixin classes that provide the desired operations.

For example, using CRTP for addition:
```cpp
// Addable mixin class
template<typename Derived>
struct Addable {
    friend Derived& operator+=(Derived& lhs, const Derived& rhs) {
        lhs.get() += rhs.get();
        return lhs;
    }
    
    friend Derived operator+(const Derived& lhs, const Derived& rhs) {
        return Derived{lhs.get() + rhs.get()};
    }
};
```
In the `StrongType` class template, you can use this mixin:
```cpp
template<typename T, typename Tag>
struct StrongType : private Addable<StrongType<T,Tag>> {
    // implementation details here
};
```

This way, only specific instantiations of `StrongType`, such as `Meter` or `Kilometer`, will have the addition operator injected, while other instantiations like `Surname` won't.

x??

---

#### Mixing Addable with StrongType

Background context explaining how to integrate the `Addable` mixin class into the `StrongType` template. This integration ensures that only specific types can benefit from the added operations, maintaining type safety and avoiding unintended operations.

:p How do you integrate the `Addable` mixin class with the `StrongType` template?
??x
To integrate the `Addable` mixin class with the `StrongType` template, you define a base template for `StrongType` that privately inherits from the `Addable` template. This ensures that only specific types derived from `StrongType` can benefit from the operations provided by `Addable`.

Here is an example:
```cpp
// Addable mixin class
template<typename Derived>
struct Addable {
    friend Derived& operator+=(Derived& lhs, const Derived& rhs) {
        lhs.get() += rhs.get();
        return lhs;
    }
    
    friend Derived operator+(const Derived& lhs, const Derived& rhs) {
        return Derived{lhs.get() + rhs.get()};
    }
};

// StrongType template
template<typename T, typename Tag>
struct StrongType : private Addable<StrongType<T,Tag>> {
    // implementation details here
};
```

This setup ensures that `Meter` and other derived types can use the addition operator, while types like `Surname` cannot.

x??

---

#### Specific Example with Meter

Background context explaining how to apply the CRTP pattern in a specific example. This involves defining a `Meter` type using `StrongType` and ensuring it has the necessary operations without affecting unrelated types.

:p How do you define a `Meter` type that can use the addition operator, while ensuring other types like `Surname` cannot?
??x
To define a `Meter` type that can use the addition operator, you derive from `StrongType` which privately inherits from the `Addable` mixin. This ensures that only derived types of `StrongType`, such as `Meter`, will have access to the addition operator.

Here is an example:
```cpp
// Addable mixin class
template<typename Derived>
struct Addable {
    friend Derived& operator+=(Derived& lhs, const Derived& rhs) {
        lhs.get() += rhs.get();
        return lhs;
    }
    
    friend Derived operator+(const Derived& lhs, const Derived& rhs) {
        return Derived{lhs.get() + rhs.get()};
    }
};

// StrongType template
template<typename T, typename Tag>
struct StrongType : private Addable<StrongType<T,Tag>> {
    // implementation details here
};

// Using the Meter type with addition
using Meter = StrongType<long, struct MeterTag>;

// Example usage in main.cpp
int main() {
    auto const m1 = Meter{100};
    auto const m2 = Meter{50};
    auto const m3 = m1 + m2; // This compiles and results in 150 meters
}
```

In this setup, `Meter` can use the addition operator because it is derived from a `StrongType` that privately inherits from `Addable`. Other types like `Surname` cannot access these operators.

x??

---

#### CR TP Implementation Details
Background context explaining the concept. The implementation details of the CR TP (Class Template Partial) are provided, which involves adding mixins through template template parameters to selectively enhance specific StrongType instantiations.

:p What is the purpose of extending `StrongType` with variadic template template parameters?
??x
The purpose is to allow individual strong type instantiations to receive or omit specific skills. This is achieved by using template template parameters that can be specified as optional arguments when defining a new strong type.
```cpp
template< typename T, typename Tag, template<typename> class... Skills >
struct StrongType : private Skills<StrongType<T,Tag,Skills...>>...
```
x??

---

#### Selective Skill Addition
Background context explaining the concept. The example demonstrates how to add specific skills (e.g., Printable and Swappable) to certain strong type instantiations while omitting others.

:p How can we define `Meter` and `Kilometer` types with specific skills?
??x
We define `Meter` and `Kilometer` using the `StrongType` template, specifying the desired skills. For example:
```cpp
template< typename T >
using Meter = StrongType<T, struct MeterTag, Addable, Printable, Swappable>;

template< typename T >
using Kilometer = StrongType<T, struct KilometerTag, Addable, Printable, Swappable>;
```
This allows `Meter` and `Kilometer` to be addable, printable, and swappable. Meanwhile, `Surname` is defined without the `Addable` skill.
```cpp
using Surname = StrongType<std::string, struct SurnameTag, Printable, Swappable>;
```
x??

---

#### CR TP as an Implementation Detail
Background context explaining the concept. The example shows that while CR TP can be used to implement specific functionality, it does not act as a design pattern but rather as an implementation detail.

:p Why is `StrongType` not designed as a polymorphic base class?
??x
`StrongType` is not designed with a virtual or protected destructor, making it unsuitable for use as a polymorphic base class. Instead, it uses CR TP as a private base class to encapsulate specific functionality.
```cpp
template< typename T, typename Tag, template<typename> class... Skills >
struct StrongType : private Skills<StrongType<T,Tag,Skills...>>...
```
This approach ensures that the mixin classes (`Addable`, `Printable`, etc.) are used only as implementation details and do not provide a public interface for inheritance.
x??

---

#### Inheritance Usage in CR TP
Background context explaining the concept. The example illustrates how CR TP is used to implement specific behaviors (skills) selectively, adhering to the Liskov Substitution Principle (LSP).

:p How does CR TP use inheritance according to the LSP?
??x
CR TP uses inheritance to define a base class that represents the requirements and expected behavior of derived classes. For instance:
```cpp
template< typename T, typename Tag, template<typename> class... Skills >
struct StrongType : private Skills<StrongType<T,Tag,Skills...>>...
```
In this case, `Skills` are used to add specific functionalities (skills) to the `StrongType` instantiation, ensuring that derived types can be substituted for their base type without breaking the expected behavior.
x??

---

#### CRTP as a Design Pattern vs. Implementation Pattern
Background context: The text discusses how the Curiously Recurring Template Pattern (CRTP) can be used both as a design pattern and an implementation pattern, explaining their differences and contexts of usage.

:p What is CRTP, and in what way does it differ when implemented as a design pattern versus an implementation pattern?
??x
CRTP stands for Curiously Recurring Template Pattern. When implemented as a design pattern, the base class requires a virtual or protected destructor because derived classes directly access operations via pointers or references to the base class. This makes CRTP suitable for scenarios where you want to provide static mixin functionality and ensure proper cleanup.

When used as an implementation pattern, especially in C++20, CRTP does not need a virtual or protected destructor since it is only an internal detail of the implementation. This allows for more flexible and cleaner code design without exposing unnecessary details to the calling code.
??x
In this form, however, CRTP does not compete with C++20 concepts but remains valuable as a unique technique for static mixin classes.

C++11 Code Example (Design Pattern):
```cpp
template <typename T>
class Base {
protected:
    void doSomething() { /* implementation */ }
public:
    virtual ~Base() {} // Virtual destructor required for proper cleanup
};

class Derived : public Base<Derived> {
};
```

C++20 Code Example (Implementation Pattern):
```cpp
template <typename T>
class Base {
    void doSomething() { /* implementation */ }
};

class Derived : public Base<Derived> {
};
```
x??

---
#### Difference Between CRTP as a Design and Implementation Pattern
Background context: The text explains the distinction between using CRTP in its role as a design pattern (requiring virtual or protected destructors) versus an implementation pattern (not requiring them). This differentiation is crucial for understanding when to use CRTP in different scenarios.

:p How does CRTP differ based on whether it is used as a design pattern or an implementation pattern, and what are the implications of each usage?
??x
When CRTP is used as a design pattern:
- The base class provides virtual or protected destructors.
- Derived classes directly access operations via pointers or references to the base class.
- This makes CRTP useful for providing static mixin functionality where proper cleanup is necessary.

When CRTP is used as an implementation pattern in C++20:
- No need for virtual or protected destructors since it remains an internal detail of the implementation.
- It offers more flexibility and cleaner code design without exposing unnecessary details to the calling code.
??x
The implication is that while CRTP can be a valuable tool, its usage depends on whether you are designing the system's interface or just implementing specific functionalities.

Example:
```cpp
// Design Pattern: CRTP with virtual destructor
template <typename T>
class Base {
public:
    virtual ~Base() {} // Virtual destructor required for proper cleanup
};

class Derived : public Base<Derived> {
    void doSomething() { /* implementation */ }
};

// Implementation Pattern: CRTP without virtual destructor
template <typename T>
class Base {
private:
    void doSomething() { /* implementation */ }
};

class Derived : public Base<Derived> {
};
```
x??

---
#### GUIDELINE 27: USE CRTP FOR STATIC MIXIN CLASSES
Background context: The text introduces a specific guideline (GUIDELINE 27) recommending the use of CRTP for static mixin classes. It emphasizes understanding the difference between using CRTP as a design pattern and an implementation pattern.

:p What is the guideline regarding the use of CRTP, and how should one differentiate its usage in different contexts?
??x
The guideline (GUIDELINE 27) recommends using CRTP for static mixin classes, with a key differentiation:
- **Design Pattern:** Use when you need to provide an abstraction through CRTP. This typically requires virtual or protected destructors.
- **Implementation Pattern:** Use primarily as a technique for implementing functionalities without exposing implementation details to the calling code. No virtual or protected destructor is needed here.

Key points include:
- CRTP can be both a design and an implementation pattern, each with its specific use cases.
- Understanding these differences helps in choosing the right approach based on the requirements of your project.
??x
For example, if you are designing a framework where certain behaviors need to be mixed into derived classes, using CRTP as a design pattern might be appropriate. Conversely, for internal implementation details, CRTP can serve as an efficient and clean way to add functionalities without cluttering the interface.

Example:
```cpp
// Design Pattern: Abstraction through CRTP with virtual destructor
template <typename T>
class Base {
public:
    virtual void doSomething() { /* implementation */ }
    virtual ~Base() {} // Virtual destructor required for proper cleanup
};

class Derived : public Base<Derived> {
    void doSomething() override { /* implementation */ }
};

// Implementation Pattern: Static mixin functionality without virtual destructor
template <typename T>
class Base {
private:
    void doSomething() { /* implementation */ }
};

class Derived : public Base<Derived> {
};
```
x??

---

