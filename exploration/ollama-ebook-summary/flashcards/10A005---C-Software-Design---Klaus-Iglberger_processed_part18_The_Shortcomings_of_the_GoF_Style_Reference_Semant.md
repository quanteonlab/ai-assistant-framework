# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 18)

**Starting Chapter:** The Shortcomings of the GoF Style Reference Semantics

---

#### Understanding Command vs Strategy Patterns

Background context: The passage discusses how to differentiate between the Command and Strategy design patterns, particularly focusing on the presence of an `undo()` operation. It provides a detailed explanation of when each pattern might be appropriate.

:p What is the key distinguishing factor between Command and Strategy patterns according to the text?
??x
The presence or absence of an `undo()` operation is the key differentiator. If your action includes an `undo()` method, it suggests that you are dealing with the Command design pattern because this allows for rolling back actions that have been performed. Conversely, if there is no `undo()` operation, the code might be better suited to the Strategy pattern.

In pseudocode:
```pseudocode
class Command {
    perform() 
        // Perform action logic here
    
    undo()
        // Rollback logic here
}

class Strategy {
    perform() 
        // Perform action logic here without an undo capability
}
```
x??

---

#### Command Design Pattern

Background context: The text explains the Command design pattern, emphasizing its ability to decouple actions from their invokers and provide flexibility. It also notes that `undo()` is optional but necessary in some cases.

:p What is a key strength of the Command design pattern according to the passage?
??x
A key strength of the Command design pattern is that it allows for abstracting away implementation details, making it easier to add new actions without modifying existing code. This aligns with the Single Responsibility Principle (SRP) and Open/Closed Principle (OCP).

In C++:
```cpp
class Light {
public:
    void on() { std::cout << "Light is ON\n"; }
    void off() { std::cout << "Light is OFF\n"; }
};

class Command {
protected:
    Light* light;

public:
    Command(Light& l) : light(&l) {}

    virtual void execute() = 0;
    virtual void undo() {}
};
```
x??

---

#### Strategy Design Pattern

Background context: The text contrasts the Strategy design pattern with the Command pattern, highlighting that it focuses more on how something is done rather than what should be done.

:p What is a key difference between the Strategy and Command patterns according to the passage?
??x
The main difference lies in the `undo()` operation. If your action does not provide an `undo()` method because it is focused on "how" to perform the task or lacks necessary information, then you are most likely dealing with the Strategy pattern.

In pseudocode:
```pseudocode
class LightStrategy {
    function turnOn() 
        // How to turn the light ON
    
    function turnOff() 
        // How to turn the light OFF
}
```
x??

---

#### Performance Overhead in Command Pattern

Background context: The text mentions that one disadvantage of using a base class for the Command pattern is increased runtime performance overhead due to additional indirection. However, it also notes that this might be outweighed by the flexibility gained.

:p What is a potential drawback of implementing the Command design pattern via a base class?
??x
A potential drawback is the added runtime performance overhead due to the extra layer of indirection required when using a base class for Commands. This can slow down execution if not properly optimized.

In C++:
```cpp
class LightCommand : public Command {
public:
    void execute() override { light.on(); }
    void undo() override { light.off(); }
};
```
x??

---

#### Application of Command and Strategy Patterns

Background context: The text suggests that both design patterns are fundamental and useful in various scenarios, helping to isolate concerns and improve maintainability.

:p How can the Command pattern be effectively used according to the passage?
??x
The Command pattern should be applied when you want to abstract and encapsulate actions that may or may not need an undo mechanism. It helps decouple tasks from their invokers and allows for easier extension of functionality without modifying existing code.

Example usage in C++:
```cpp
class Light {
public:
    void on() { std::cout << "Light is ON\n"; }
    void off() { std::cout << "Light is OFF\n"; }
};

class ToggleCommand : public Command {
private:
    Light& light;

public:
    ToggleCommand(Light& l) : light(l) {}

    void execute() override { light.on(); }
    void undo() override { light.off(); }
};
```
x??

---

#### The GoF Style and Its Limitations
Background context: The Gang of Four (GoF) design patterns, including Strategy and Command, were introduced as object-oriented design patterns. They are built on inheritance hierarchies, which can lead to several performance and maintenance issues.

:p What is the primary issue with the GoF style in terms of modern C++?
??x
The GoF style primarily relies on reference semantics (or pointer semantics), leading to increased runtime overhead due to virtual function calls and additional memory fragmentation. This style also heavily uses inheritance, which can complicate code maintenance and introduce inefficiencies.
x??

---
#### Performance Issues with the GoF Style
Background context: The GoF design patterns use inheritance hierarchies, which increase runtime overhead through virtual functions and lead to issues like extra memory allocation, suboptimal cache usage, and increased compile-time complexity.

:p How does the use of virtual functions impact performance in C++?
??x
Virtual functions introduce a level of indirection that increases runtime overhead. The compiler cannot optimize as effectively with virtual function calls because they must be resolved at runtime, rather than at compile time. This can lead to slower execution and reduced code efficiency.

For example, consider the following simplified scenario:
```cpp
class Base {
public:
    virtual void operation() { std::cout << "Base"; }
};

class Derived : public Base {
public:
    void operation() override { std::cout << "Derived"; }
};
```
When a `virtual` function is called through an object pointer or reference, the runtime system must check the vtable (virtual table) to determine which implementation of the function should be executed. This extra step can slow down execution.
x??

---
#### Reference Semantics and Its Disadvantages
Background context: The GoF style relies heavily on pointers and references, leading to issues such as increased memory usage, runtime overhead, and suboptimal cache utilization.

:p Why is reference semantics considered problematic in modern C++?
??x
Reference semantics, or pointer semantics, are problematic because they often lead to performance bottlenecks due to frequent allocations and deallocations of small objects. Additionally, the use of pointers can cause memory fragmentation, making it harder for the compiler to optimize code.

For instance, consider a situation where many small polymorphic objects need to be allocated and deallocated frequently:
```cpp
class SmallObject {
public:
    void doSomething() { std::cout << "Doing something"; }
};

void useSmallObjects(std::vector<SmallObject> &objects) {
    for (auto &obj : objects) {
        obj.doSomething();
    }
}
```
The frequent allocation and deallocation of these small objects can lead to increased runtime overhead, memory fragmentation, and suboptimal cache usage.
x??

---
#### Example with `std::span` in C++
Background context: The GoF style uses dynamic polymorphism through inheritance hierarchies. However, modern C++ idioms often prefer value semantics over reference semantics for performance reasons.

:p What is the purpose of using `std::span` instead of a dynamic array or vector in the provided code?
??x
The purpose of using `std::span` is to provide a lightweight and efficient way to handle slices of arrays without owning the data. It allows you to work with existing contiguous sequences of elements without the overhead of dynamic allocation.

In contrast, when using dynamic arrays or vectors (`std::vector<int>`), you would encounter issues like:
```cpp
int main() {
    std::vector<int> v{ 1, 2, 3, 4 };
    std::vector<int> const w{ v }; // This will cause a compilation error because `w` cannot be copy-constructed from `v`.

    std::span<int> s{ v }; // Correct way to use std::span

    w[2] = 99; // Compilation error, as `w` is a constant reference
}
```
The `std::span` object does not own the data and only provides access to it. This makes it safer and more efficient than using dynamic arrays or vectors.
x??

---
#### Value Semantics vs Reference Semantics
Background context: Modern C++ emphasizes value semantics over reference semantics to improve performance, reduce memory overhead, and simplify code.

:p Why should modern C++ developers prefer value semantics?
??x
Modern C++ developers should prefer value semantics because it leads to better performance, reduced memory fragmentation, and easier-to-understand code. Value semantics allow for more efficient copying and moving of data, which can significantly improve the overall performance of programs.

For example, consider a scenario where you need to pass objects by value:
```cpp
class MyClass {
public:
    void doSomething(const std::vector<int>& v) { /* ... */ }
};

void useMyClass(MyClass obj) {
    // Passing by value is efficient and safe with value semantics.
}
```
In this case, the compiler can perform copy elision and other optimizations that are not possible with reference semantics.

Using value semantics also avoids issues like shallow copying or aliasing, which can be problematic in reference-based designs.
x??

---

#### std::span Overview and Const Qualifier Behavior
std::span is a template class provided by C++17 that represents an abstraction for an array. It can be used with various types of arrays, including built-in arrays, `std::array`, and `std::vector`, without coupling to any specific type. The primary use case of std::span is to provide a range-based access mechanism.

The print() function demonstrates the purpose of std::span by simply traversing elements within a span and printing them via `std::cout`.

:p What does std::span represent, and how can it be used with different types of arrays?
??x
std::span represents an abstraction for an array and can be used with various types such as built-in arrays, `std::array`, and `std::vector`. It provides a range-based access mechanism that is type-agnostic.
x??

---

#### Const Qualifier with std::span
When using const on a std::span, it does not prevent modifications of the underlying data if the span acts as a reference to an array or vector. The const qualifier only prevents modification of the span itself, not the elements it references.

:p What happens when you declare `std::span<int> const s{ v }` and try to modify elements through this span?
??x
Declaring `std::span<int> const s{ v }` makes the span object immutable but does not prevent modifications of the underlying array. The const qualifier only affects the span's internal state, not the data it references.
x??

---

#### Const Vector vs. std::span
A vector declared as `const` (e.g., `std::vector<int> const v`) prevents modification of its elements directly but allows for creation of a non-const span that can modify those elements.

:p How does a const vector differ from a const std::span created from the same vector?
??x
A const vector (`std::vector<int> const v`) makes it impossible to modify its elements, whereas creating a non-const span from this vector (`std::span<int> s{ v }`) allows for modification of the underlying array's elements.
x??

---

#### Example Code with std::span and Vector

```cpp
#include <iostream>
#include <vector>
#include <span>

void print(const std::span<int>& s) {
    for (int i : s) {
        std::cout << " " << i;
    }
}

int main() {
    std::vector<int> v = {1, 2, 3, 4};
    const std::vector<int> w{v}; // Copy of v
    std::span<int> s{v};         // Span referencing v

    s[2] = 99;  // Works because s is a reference

    print(s);   // Prints (1 2 99 4)

    return EXIT_SUCCESS;
}
```

:p Explain the behavior of modifying an element through `s` in the provided code.
??x
Modifying an element through `std::span<int> s{v}` works because `s` is a reference to the underlying vector `v`. The const qualifier on the span does not affect the underlying data, only the span itself. Thus, you can modify elements of the vector via the span.
x??

---

#### Different Behavior with Const Vector

```cpp
#include <iostream>
#include <vector>

void print(const std::vector<int>& v) {
    for (int i : v) {
        std::cout << " " << i;
    }
}

int main() {
    std::vector<int> v = {1, 2, 3, 4};
    const std::vector<int> w{v}; // Copy of v

    w[2] = 99;  // Compilation error: w is declared const
                // and cannot modify its elements.

    return EXIT_SUCCESS;
}
```

:p Why does `w[2] = 99` cause a compilation error in the provided code?
??x
The assignment `w[2] = 99` causes a compilation error because `const std::vector<int> w{v}` declares `w` as a constant vector. This means that its elements cannot be modified directly, leading to a compile-time error.
x??

---

#### Assignment to Vector

```cpp
std::vector<int> v = {1, 2, 3, 4};
v[2] = 99;  // Works.

print(v);   // Prints (1 2 99 4)

// Later in the program...
v = {5, 6, 7, 8, 9}; // Reassigning v with a new vector
s[2] = 99;           // Works because s is still referencing the original vector.
```

:p How does reassigning `v` affect the span `s` created from it?
??x
Reassigning `v` to `{5, 6, 7, 8, 9}` changes the contents of the vector. However, since `std::span<int> s{v}` references the original vector, modifying elements through `s` still works with the new values in `v`.
x??

---

#### Summary of std::span and Const Qualifiers

In summary, while const on a span (e.g., `std::span<int> const s{ v }`) does not prevent modification of underlying data, it affects only the span's internal state. On the other hand, const vectors (`std::vector<int> const v`) make their elements immutable.

:p What is the key difference between modifying an element through a non-const std::span and a const std::vector?
??x
The key difference is that modifying an element through a non-const std::span allows changes to the underlying data, whereas modifying an element through a const std::vector results in a compile-time error because its elements cannot be modified directly.
x??

---

#### Undefined Behavior Due to Reallocation

Background context: When modifying a `std::vector` that results in reallocation (changing the size), any nonowning reference or pointer, such as a `std::span`, may become invalid. This is because the vector's internal storage might be moved to a new location.

If you write through an invalidated reference/pointer after reallocation, undefined behavior occurs.

:p What happens if we use a `std::span` on a `std::vector` that has been resized?
??x
When a `std::vector` is resized and reallocated (e.g., due to inserting more elements), the memory location of its first element may change. If you have a `std::span` referring to the old memory location, writing through this span will access invalid memory, leading to undefined behavior.

For example:
```cpp
#include <vector>
#include <span>

int main() {
    std::vector<int> v{1, 2, 3};
    std::span<int> s(v);

    // Resizing the vector causes reallocation and changes the address of elements.
    v.push_back(4); // After this, s is invalid for writing.

    // This is undefined behavior because we are writing to an invalid memory location
    s[0] = 10; // Attempting to write through a span pointing to old vector memory
}
```
x??

---

#### Erase-Remove Idiom with `std::remove`

Background context: The erase-remove idiom involves using the `std::remove` algorithm followed by an erase operation. However, this can lead to undefined behavior if the value being removed is changed during its removal.

The problem occurs because `std::remove` uses a reference parameter for identifying elements to remove, which may change their values as they are being removed from the container.

:p How does the use of `const&` in `std::remove` potentially cause issues?
??x
Using `std::remove` with `const&` means that the value passed is not modified during the removal process. However, if elements in the vector change their values while being removed (e.g., due to some side effects), this can lead to incorrect behavior.

For example:
```cpp
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec{1, -3, 27, 42, 4, -8, 22, 42, 37, 4, 18, 9};
    
    // Find the position of the greatest element.
    auto pos = std::max_element(begin(vec), end(vec));
    int value_to_remove = *pos; // Get the greatest value.

    // Correct way to use erase-remove idiom:
    vec.erase(std::remove(begin(vec), end(vec), value_to_remove), end(vec));

    // Incorrect usage (using std::remove directly):
    auto iter = std::remove(begin(vec), end(vec), *std::max_element(begin(vec), end(vec)));
    vec.erase(iter, end(vec)); // This can cause undefined behavior.
}
```
x??

---

#### Correct Usage of `std::erase` and `std::remove`

Background context: The `std::erase` function also takes a constant reference to the value to be removed. To avoid issues like those described in the erase-remove idiom, you need to explicitly determine the element before passing it.

:p Why is directly using `*std::max_element` as an argument to `std::remove` problematic?
??x
Directly using `*std::max_element` as an argument to `std::remove` can cause issues because `std::remove` uses a reference parameter. If the element's value changes during removal, it will remove based on the new value instead of the original one.

For example:
```cpp
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec{1, -3, 27, 42, 4, -8, 22, 42, 37, 4, 18, 9};
    
    // Find the position of the greatest element.
    auto pos = std::max_element(begin(vec), end(vec));
    int value_to_remove = *pos; // Get the greatest value.

    // Correct way to use erase-remove idiom:
    vec.erase(std::remove(begin(vec), end(vec), value_to_remove), end(vec));

    // Incorrect usage (using std::remove directly):
    auto iter = std::remove(begin(vec), end(vec), *std::max_element(begin(vec), end(vec)));
    vec.erase(iter, end(vec)); // This can cause undefined behavior.
}
```
x??

---

#### Raw Pointers vs. Smart Pointers
Background context explaining that raw pointers do not manage ownership, while smart pointers take on this responsibility. The discussion highlights how smart pointers improve code semantics and management of resources but still have limitations.

:p What is the main difference between raw pointers and smart pointers in C++?
??x
Raw pointers represent a non-owning resource, meaning they do not manage the lifetime of the pointed-to object. In contrast, smart pointers (like `std::unique_ptr`, `std::shared_ptr`) take responsibility for managing the lifetime of the resource they point to.

Smart pointers help address issues related to memory management and resource handling, making code more maintainable.
x??

---

#### Value Semantics in C++
Background context explaining that value semantics is a concept where values (like objects) are treated as complete entities. In the context of containers like `std::vector`, copying creates deep copies, meaning the copied object has its own resources and does not share with the original.

:p What does value semantics mean in the context of C++?
??x
Value semantics in C++ refers to treating values (such as objects) as complete entities. When you copy a value, it results in a full copy where both the content and memory are duplicated. Containers like `std::vector` follow this principle by creating deep copies when they are copied.

For example:
```cpp
#include <vector>
#include <cassert>

int main() {
    std::vector<int> v1{ 1, 2, 3, 4, 5 };
    auto v2 = v1; // v2 is a complete copy of v1 with its own memory.
    
    assert(v1 == v2); // True
    assert(&v1[0] != &v2[0]); // True
    
    v2[2] = 99;
    assert(v1 != v2); // True
    
    const auto v3 = v1; // v3 is a const reference to the original data.
    
    // The following line would result in a compile-time error:
    // v3[2] = 99; // Compilation error. v3 is declared as const, preventing modification.
}
```
x??

---

#### Copy Construction vs. Const Objects
Background context explaining that when copying objects using value semantics, they create complete copies with their own resources. However, making an object `const` prevents any modifications to its state.

:p How does copy construction work for a `std::vector<int>` in C++?
??x
Copy construction for a `std::vector<int>` involves creating a new vector that is a full duplicate of the original, including allocating its own memory and copying all elements. This ensures both vectors operate independently.

Example:
```cpp
#include <vector>
#include <cassert>

int main() {
    std::vector<int> v1{ 1, 2, 3, 4, 5 };
    auto v2 = v1; // v2 is a full copy of v1 with its own memory.
    
    assert(v1 == v2); // True
    assert(&v1[0] != &v2[0]); // True
    
    v2[2] = 99;
    assert(v1 != v2); // True
}
```
x??

---

#### Const Objects and Value Semantics
Background context explaining that const objects declared as values are completely immutable, meaning no modifications can be made to their state.

:p How does a `const` object behave in value semantics?
??x
In value semantics, if an object is declared as `const`, it means the entire object is considered constant. Any attempt to modify its state will result in a compilation error.

For example:
```cpp
#include <vector>
#include <cassert>

int main() {
    std::vector<int> v1{ 1, 2, 3, 4, 5 };
    const auto v3 = v1; // v3 is a const reference to the original data.
    
    // The following line would result in a compile-time error:
    // v3[2] = 99; // Compilation error. v3 is declared as const, preventing modification.
}
```
x??

---

#### Performance Considerations of Value Semantics
Background context explaining that while value semantics improve code clarity and reduce bugs related to shared ownership, they can also have performance implications due to the creation of deep copies.

:p How might value semantics affect the performance of C++ programs?
??x
Value semantics, by ensuring full copying during operations like assignment or copying an object, may introduce overhead compared to reference semantics. However, this can lead to more predictable and easier-to-understand code with fewer bugs related to shared ownership issues.

Example:
```cpp
#include <vector>
#include <cassert>

int main() {
    std::vector<int> v1{ 1, 2, 3, 4, 5 };
    auto v2 = v1; // v2 is a full copy of v1 with its own memory.
    
    assert(v1 == v2); // True
    assert(&v1[0] != &v2[0]); // True
    
    v2[2] = 99;
    assert(v1 != v2); // True
    
    const auto v3 = v1; // v3 is a const reference to the original data.
    
    // The following line would result in a compile-time error:
    // v3[2] = 99; // Compilation error. v3 is declared as const, preventing modification.
}
```
x??

---

