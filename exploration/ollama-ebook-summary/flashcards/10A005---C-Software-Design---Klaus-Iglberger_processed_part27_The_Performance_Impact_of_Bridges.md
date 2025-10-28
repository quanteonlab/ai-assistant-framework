# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 27)

**Starting Chapter:** The Performance Impact of Bridges

---

#### Bridge Design Pattern Overview
Background context: The Bridge design pattern is a structural pattern that aims to decouple an abstraction from its implementation so that the two can vary independently. This separation helps in reducing physical dependencies, making code more flexible and maintainable.

:p What is the primary goal of using the Bridge design pattern?
??x
The primary goal of using the Bridge design pattern is to separate a software entity's interface from its implementation, allowing both to evolve independently.
x??

---

#### Implementing Person1 Class
Background context: The `Person1` class represents a straightforward implementation without any indirections or bridges. All data members are directly part of the class.

:p What is the structure and size of the `Person1` class?
??x
The `Person1` class has all its data members (six strings and one int) directly within the class definition. On a 64-bit machine, assuming Clang 11.1 and GCC 11.1, the total size is 152 bytes with Clang and 200 bytes with GCC.

```cpp
#include <string>

class Person1 {
public:
    // ...
private:
    std::string forename_;
    std::string surname_;
    std::string address_;
    std::string city_;
    std::string country_;
    std::string zip_;
    int year_of_birth_;
};
```
x??

---

#### Implementing Person2 Class with Pimpl Idiom
Background context: The `Person2` class uses the Pimpl (Pointer-to-Implementation) idiom to hide implementation details. This makes it easier to change the implementation without altering the public interface.

:p How does the `Person2` class structure differ from `Person1`?
??x
The `Person2` class uses a nested `Impl` struct and a `std::unique_ptr` to manage its private data members, effectively hiding their implementation details. The total size of `Person2` is much smaller than `Person1`, as the `Impl` struct's size is similar but it only occupies 8 bytes due to the pointer.

```cpp
#include <memory>

class Person2 {
public:
    explicit Person2(/*...various person arguments...*/);
    ~Person2();
    // ...

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

struct Person2::Impl {
    std::string forename;
    std::string surname;
    std::string address;
    std::string city;
    std::string country;
    std::string zip;
    int year_of_birth;
};
```
x??

---

#### Performance Impact of Indirection
Background context: The Bridge design pattern can introduce performance overhead due to indirections and virtual function calls. However, the actual impact varies based on factors like access patterns and compiler optimizations.

:p How does the indirection through `Person2` affect memory usage?
??x
The `Person2` class uses a pointer (via `std::unique_ptr`) to manage its implementation details, which significantly reduces the size of the `Person2` struct itself. On a 64-bit machine, assuming Clang 11.1 and GCC 11.1, `Person2` occupies only 8 bytes.

```cpp
#include <memory>

class Person2 {
public:
    explicit Person2(/*...various person arguments...*/);
    ~Person2();
    // ...

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

struct Person2::Impl {
    std::string forename;
    std::string surname;
    std::string address;
    std::string city;
    std::string country;
    std::string zip;
    int year_of_birth;
};
```
x??

---

#### Benchmarking Performance
Background context: To evaluate the performance impact of using a Bridge, we compare `Person1` and `Person2`. The benchmark creates two large vectors of `Person` instances to measure traversal times.

:p What is the purpose of this benchmark?
??x
The benchmark aims to compare the performance between direct access (using `Person1`) and indirect access through a bridge (using `Person2`). It measures how many times faster or slower `Person2` performs compared to `Person1`.

```cpp
#include <vector>
#include <algorithm>

// Person1 and Person2 implementations as described above

std::vector<Person1> createPersons(size_t count);
std::vector<Person2> createPersons(size_t count);

void benchmark() {
    auto persons1 = createPersons(25000);
    auto persons2 = createPersons(25000);

    for (int i = 0; i < 100; ++i) {
        auto oldestPerson1 = std::min_element(persons1.begin(), persons1.end(),
                                             [](const Person1& a, const Person1& b) { return a.year_of_birth_ < b.year_of_birth_; });
        auto oldestPerson2 = std::min_element(persons2.begin(), persons2.end(),
                                             [](const Person2& a, const Person2& b) { return a.pimpl_->year_of_birth_ < b.pimpl_->year_of_birth_; });

        // Process the results
    }
}
```
x??

---

#### Bridge Implementation Performance Impact

Background context: The passage discusses the performance implications of different implementation techniques, specifically focusing on a "Bridge" (Pimpl) idiom and its effect on performance. It highlights that while a complete Pimpl can introduce overhead, it may be possible to optimize performance by strategically placing data members.

:p What does the passage reveal about the impact of Bridge implementations on performance?
??x
The passage indicates that a full Bridge implementation incurs significant performance penalties (10-13% depending on the compiler). However, it also suggests that optimizing the placement of frequently used and infrequently used data members can improve performance by reducing memory overhead. This optimization involves using a "partial Pimpl" approach.

```cpp
// Example of Person3 with partial Pimpl
class Person3 {
public:
    explicit Person3(/*... various person arguments...*/);
    ~Person3();
    // ... other methods ...

private:
    std::string forename_;
    std::string surname_;
    int year_of_birth_;

    struct Impl;  // Forward declaration of the impl struct
    std::unique_ptr<Impl> pimpl_;  // Pointer to implementation details
};

struct Person3::Impl {
    std::string address;
    std::string city;
    std::string country;
    std::string zip;
};
```
x??

---

#### Optimizing Performance with Partial Pimpl

Background context: The passage demonstrates how optimizing the placement of data members can lead to improved performance. It specifically mentions that by separating frequently used and infrequently used data, the size of the class instance can be reduced, which can improve memory efficiency.

:p How does the partial Pimpl approach help in optimizing Person3's implementation?
??x
The partial Pimpl approach helps optimize Person3's implementation by separating frequently used data members from infrequently used ones. This reduces the overall size of the `Person3` class instance, leading to better memory utilization and potentially faster performance.

```cpp
// Example of Person3 with optimized placement of data members
class Person3 {
public:
    explicit Person3(/*... various person arguments...*/);
    ~Person3();
    // ... other methods ...

private:
    std::string forename_;  // Frequently used
    std::string surname_;   // Frequently used
    int year_of_birth_;     // Frequently used

    struct Impl {           // Struct for infrequently used data
        std::string address;
        std::string city;
        std::string country;
        std::string zip;
    };

    std::unique_ptr<Impl> pimpl_;
};

// Constructor and destructor implementation remains the same as before.
```
x??

---

#### Performance Measurement of Different Implementations

Background context: The passage provides performance metrics for various implementations, including a no Pimpl approach (Person1), complete Pimpl idiom (Person2), and partial Pimpl idiom (Person3). These measurements are normalized to Person1's performance to highlight the relative impact on speed.

:p What do the performance results in Table 7-2 indicate about different implementation techniques?
??x
The performance results in Table 7-2 show that while a complete Pimpl implementation incurs significant overhead, reducing the size of the `Person3` instance by separating frequently used data from infrequently used data can improve performance. Specifically, Person3 outperforms Person1 by approximately 6.5% for Clang and 14.0% for GCC.

```cpp
// Example of Table 7-2: Performance results
#include <iostream>

struct PerformanceResult {
    float gcc;
    float clang;
};

PerformanceResult getPerformanceResults() {
    PerformanceResult results = {0.8597, 0.9353};
    return results;
}

int main() {
    PerformanceResult results = getPerformanceResults();
    std::cout << "GCC Performance: " << results.gcc * 100 << "%" << std::endl;
    std::cout << "Clang Performance: " << results.clang * 100 << "%" << std::endl;
    return 0;
}
```
x??

---

#### Importance of Representative Benchmarks

Background context: The passage emphasizes the importance of using representative benchmarks to verify performance improvements or bottlenecks. It stresses that theoretical improvements might not always translate into practical benefits and should be validated through actual code testing.

:p Why is it important to use a representative benchmark for verifying performance gains?
??x
Using a representative benchmark is crucial because it ensures that any observed performance improvements are relevant to the actual usage scenario. Theoretical optimizations may not show significant results in real-world applications, so empirical validation through benchmarks based on actual data and code is essential.

```cpp
// Example of using a simple benchmark to measure performance
#include <chrono>
#include <iostream>

void benchmark(const char* name) {
    auto start = std::chrono::high_resolution_clock::now();
    // Code block representing the operation being benchmarked
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << name << " took: " << duration.count() * 1000 << "ms" << std::endl;
}

int main() {
    benchmark("Person3");
    return 0;
}
```
x??

---

#### Guidelines for Implementing Bridges

Background context: The passage concludes with guidelines for using bridges, emphasizing the need to be aware of performance gains and losses. It suggests that partial Pimpl can be beneficial in certain scenarios but should always be validated through benchmarks.

:p What are some key considerations when implementing a bridge according to the guidelines?
??x
When implementing a bridge (Pimpl), it is essential to consider several factors:

1. **Performance Impact**: Bridges generally introduce overhead, so their use must be justified by measurable performance benefits.
2. **Partial Pimpl**: Separating frequently used data from infrequently used data can reduce memory footprint and improve performance.
3. **Benchmarking**: Always validate the impact of changes using representative benchmarks based on actual code and data.

```cpp
// Example guideline for implementing a bridge
GUIDELINE 29: BE AWARE OF BRIDGE PERFORMANCE GAINS AND LOSSES

- Keep in mind that bridges can have a negative performance impact.
- Be aware that partial Pimpl can have a positive impact by separating frequently used from infrequently used data.
- Always confirm performance bottlenecks or improvements through representative benchmarks; do not rely solely on intuition.

```
x??

---

#### Prototype Pattern Overview
Background context explaining the problem: When copying an object, especially when using a base pointer to point to derived objects, default copy constructors and assignment operators may not work as expected. The Prototype pattern provides a solution by allowing you to clone existing objects through a prototype instance.

:p What is the primary issue addressed by the Prototype pattern in C++?
??x
The primary issue is that default copy constructors and assignment operators might fail when dealing with polymorphic types because they only operate on base class pointers, which cannot correctly create copies of derived classes. The solution involves using a clone method to return an exact copy of the object.
x??

---
#### Prototype Pattern in C++
Background context explaining how the Prototype pattern can be applied in C++ by defining a `clone` method that returns a new instance of the same type.

:p How does the Prototype pattern solve the problem of copying objects with polymorphic types?
??x
The Prototype pattern solves this issue by providing a clone method within each class, allowing it to create a deep copy of itself. This way, regardless of the base pointer type, you can always get an exact replica of the object.

Here is how you might implement the `clone` method in C++:

```cpp
//---- <Animal.h> ----------------
class Animal {
public:
    virtual ~Animal() = default;
    virtual void makeSound() const = 0;

    // Prototype pattern: declare a clone function
    virtual std::unique_ptr<Animal> clone() const = 0; // Returns a unique pointer to the new object

    // ... more animal-specific functions
};

//---- <Sheep.h> ----------------
class Sheep : public Animal {
public:
    explicit Sheep(std::string name) : name_{std::move(name)} {}

    void makeSound() const override;
    
    std::unique_ptr<Animal> clone() const override; // Implement the clone function

private:
    std::string name_;
};

//---- <Sheep.cpp> ----------------
void Sheep::makeSound() const {
    std::cout << "baa ";
}

std::unique_ptr<Animal> Sheep::clone() const {
    return std::make_unique<Sheep>(*this); // Return a unique pointer to the cloned object
}
```

x??

---
#### Example Usage of Prototype Pattern
Background context explaining how to use the `clone` method in the main function.

:p How do you use the `clone` method in C++?
??x
You can use the `clone` method by calling it on an existing object and storing the returned pointer. This will create a deep copy of the object, ensuring that all derived classes are also properly copied.

Here is how you might implement this in the main function:

```cpp
#include <Sheep.h>
#include <iostream>

int main() {
    // Create a sheep and make it sound
    Sheep mySheep("Fluffy");
    
    Animal* animalPointer = &mySheep;  // Base pointer to a derived object

    // Clone the object using the clone method from the prototype pattern
    std::unique_ptr<Animal> clonedSheep = std::static_pointer_cast<Animal>(std::make_unique<Sheep>(*dynamic_cast<const Sheep*>(animalPointer))).clone();

    // Now you can use the clonedSheep pointer to make it sound
    if (clonedSheep) {
        clonedSheep->makeSound();
    }

    return 0;
}
```

x??

---
#### Polymorphism and Cloning in Prototype Pattern
Background context explaining how polymorphism plays a crucial role in the Prototype pattern.

:p How does polymorphism relate to cloning in the Prototype pattern?
??x
Polymorphism is essential in the Prototype pattern because it allows you to use base class pointers or references to access derived class objects. When you call the `clone` method on a base class pointer, the actual type of the object determines which clone implementation is used. This ensures that you get an exact copy of the derived class object.

Here’s how polymorphism works with cloning in C++:

```cpp
//---- <Animal.h> ----------------
class Animal {
public:
    virtual ~Animal() = default;
    virtual void makeSound() const = 0;

    virtual std::unique_ptr<Animal> clone() const = 0; // Polymorphic function

    // ... more animal-specific functions
};

//---- <Sheep.h> ----------------
class Sheep : public Animal {
public:
    explicit Sheep(std::string name) : name_{std::move(name)} {}

    void makeSound() const override;

    std::unique_ptr<Animal> clone() const override; // Override the virtual clone method

private:
    std::string name_;
};

//---- <Sheep.cpp> ----------------
void Sheep::makeSound() const {
    std::cout << "baa ";
}

std::unique_ptr<Animal> Sheep::clone() const {
    return std::make_unique<Sheep>(*this); // Return a unique pointer to the cloned object
}
```

x??

---
#### Prototype Pattern and Polymorphic Copying
Background context explaining why direct copying using base class pointers fails and how cloning solves this issue.

:p Why does direct copying with base class pointers fail in C++?
??x
Direct copying with base class pointers fails because the copy constructor and assignment operator of the base class are called, which only make a shallow copy. Since the object is polymorphic (i.e., derived from an abstract or polymorphic base class), this approach cannot create a deep copy that includes all derived class members.

The Prototype pattern addresses this by providing a `clone` method in each class, ensuring that objects are deeply copied according to their actual type.

x??

---

