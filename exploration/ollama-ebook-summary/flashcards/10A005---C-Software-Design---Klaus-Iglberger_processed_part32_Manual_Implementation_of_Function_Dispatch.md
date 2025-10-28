# Flashcards: 10A005---C-Software-Design---Klaus-Iglberger_processed (Part 32)

**Starting Chapter:** Manual Implementation of Function Dispatch

---

#### Performance Optimization of Virtual Functions
Virtual functions introduce a performance overhead due to indirection through virtual function tables and vptrs. A virtual function call typically involves two indirections: one via the vptr, and another to fetch the actual function pointer from the table.

This can be costly compared to regular non-virtual function calls, which are direct and involve no such indirections.

:p How does a typical virtual function call work in C++?
??x
In a typical virtual function call, the process involves:

1. Accessing the vptr (virtual pointer) of an object.
2. Using the vptr to fetch the address of the virtual table associated with that type.
3. Finding the corresponding function pointer within the virtual table.
4. Calling the function using the obtained function pointer.

This means a virtual function call involves two levels of indirection, making it slower than regular function calls. Here is an example in pseudocode:
```cpp
// Example in Pseudocode
Function CallVirtualFunction(Object obj) {
    // Step 1: Access vptr via object
    VTable *vtable = (VTable *)*(Object.vptr);
    
    // Step 2: Fetch the function pointer from the virtual table
    void (*func_ptr)(void) = (void(*)(void))vtable[FunctionIndex];
    
    // Step 3: Call the function using the obtained function pointer
    func_ptr();
}
```
x??

---

#### Virtual Function Table and Indirections
Virtual functions use a virtual function table (vftable) to manage polymorphism. Each type with at least one virtual function has its own vftable, stored in every object of that type via an additional hidden data member called the vptr.

The vptr contains the address of the vftable for the specific type of the object, allowing dynamic dispatch through a series of indirections.

:p What is the role of vptr and virtual function tables in managing polymorphism?
??x
The vptr (virtual pointer) acts as an index to the appropriate vftable within each instance of a class that contains at least one virtual function. The vftable stores pointers to all the virtual functions defined for the type.

When you call a virtual function, here's what happens:
1. The vptr is accessed.
2. It points to the vftable specific to the object's type.
3. From the vftable, the actual function pointer corresponding to the called virtual function is fetched.
4. Finally, that function pointer is used to call the function.

This process involves two indirections: one via the vptr and another within the vftable itself:
```cpp
// Example in Pseudocode
Function CallVirtualFunction(Object obj) {
    VTable *vtable = (VTable *)*(Object.vptr); // Indirect through vptr to get vftable
    void (*func_ptr)(void) = (void(*)(void))vtable[FunctionIndex]; // Indirect again to get function pointer from the table
    func_ptr(); // Call the function using the obtained function pointer
}
```
x??

---

#### Trade Space for Speed in Virtual Dispatch
One way to reduce the performance overhead of virtual functions is by manually implementing dispatch, trading space for speed. This involves storing all virtual function pointers directly within the class.

By doing so, we avoid the need for vptr and virtual tables, reducing the number of indirections required during a call.

:p How can you optimize virtual function calls to reduce performance overhead?
??x
To optimize virtual function calls, you can manually implement dispatch by storing all virtual function pointers directly within the class. This approach reduces the number of indirections from two (one via vptr and one in the table) to just one.

Here’s how it works:
1. Store each virtual function pointer directly as a member variable.
2. Directly call these stored pointers when needed, bypassing the need for vptr or virtual tables.

For example, if you have a `Shape` class with several virtual functions, you can store their addresses and use them directly:
```cpp
class Shape {
public:
    void (*draw_)(void*);  // Pointer to draw function

private:
    // Other members...
};
```
In this setup, calling the `draw` function would be a direct call without needing to go through vptr and virtual tables.

```cpp
// Example in C++
Shape shape;
// Set up pointers to actual functions
shape.draw_ = &DrawSquare;

void DrawSquare(void* obj) {
    // Drawing logic for square
}
```
x??

---

#### Implementation Details of OwningModel Template
The `OwningModel` template is used as storage for a specific kind of shape and a drawing strategy. This reduces the external hierarchy to just this single class, which stores both the shape instance and its drawing strategy.

It removes the need for virtual inheritance or polymorphic behavior within the `Shape` class itself by using templates and raw pointers.

:p What is the purpose of the `OwningModel` template in this context?
??x
The `OwningModel` template serves as a storage mechanism for a specific kind of shape and its associated drawing strategy. It replaces the need for virtual inheritance or polymorphism within the `Shape` class itself by leveraging templates.

By using `OwningModel`, the system can handle different shapes with their respective drawing strategies without relying on virtual functions. This template-based approach allows for more explicit control over how shapes are stored and manipulated, providing a cleaner separation of concerns.

Here’s an example of how it might be used:
```cpp
template<typename ShapeT, typename DrawStrategy>
struct OwningModel {
    OwningModel(ShapeT value, DrawStrategy drawer)
        : shape_(std::move(value)), drawer_(std::move(drawer)) {}

    ShapeT shape_;
    DrawStrategy drawer_;

private:
    // Other members...
};
```
In this implementation, `OwningModel` can be instantiated with different types of shapes and drawing strategies, providing a flexible way to manage polymorphism without the overhead of virtual functions.

```cpp
// Example usage
Shape square;
DrawSquare draw_square{&square};

OwningModel<Shape, DrawStrategy> model(square, draw_square);
```
x??

---

#### Destruction Strategy Using Lambda Functions
This section discusses how to implement a strategy pattern using lambda functions instead of virtual functions, particularly focusing on the destruction part. The use of lambdas as function pointers helps maintain type safety and flexibility.

:p What is the purpose of the `DestroyOperation` in this context?
??x
The `DestroyOperation` serves as a way to configure the deleter for the `pimpl_` data member using a lambda function. This lambda acts similarly to a virtual destructor by defining how the `pimpl_` object should be deleted.

```cpp
// Example of initialization
destroy_( []( void* shapeBytes ) {
    using Model = OwningModel<ShapeT, DrawStrategy>;
    auto* const model = static_cast<Model*>(shapeBytes);
    delete model;
} );
```
x??

---

#### Drawing Strategy Using Lambda Functions
This part explains how the `DrawOperation` and `CloneOperation` are implemented as lambda functions to replace the virtual `draw()` and `clone()` functions.

:p What is the purpose of `DrawOperation` in this context?
??x
The `DrawOperation` is used to configure the function pointer for drawing. It takes a `void*` parameter, casts it to the appropriate model type, and then calls the `drawer_` method on the shape stored within that model.

```cpp
// Example of initialization
draw_( []( void* shapeBytes ) {
    using Model = OwningModel<ShapeT, DrawStrategy>;
    auto* const model = static_cast<Model*>(shapeBytes);
    (*model->drawer_)( model->shape_ );
} );
```
x??

---

#### Cloning Strategy Using Lambda Functions
This part discusses how the `CloneOperation` is used to create a new instance of the object using a lambda function.

:p What is the purpose of `CloneOperation` in this context?
??x
The `CloneOperation` is responsible for creating a clone of the `pimpl_` data member. It takes a `void*` parameter, casts it to the appropriate model type, and returns a new instance of that model by copying the existing one.

```cpp
// Example of initialization
clone_( []( void* shapeBytes ) -> void* {
    using Model = OwningModel<ShapeT, DrawStrategy>;
    auto* const model = static_cast<Model*>(shapeBytes);
    return new Model( *model );
} );
```
x??

---

#### Pimpl Idiom with Lambda Deleters
This section describes how the pimpl idiom is used in conjunction with lambda functions to manage object lifetimes.

:p How does the `pimpl_` data member's deleter work?
??x
The `pimpl_` data member's deleter works by using a lambda function that is implicitly convertible to a function pointer. This lambda acts as a custom deleter for `std::unique_ptr`, ensuring that when the `pimpl_` object is destroyed, it correctly handles the deletion of the model and its resources.

```cpp
// Example of pimpl_'s initialization with a lambda deleter
pimpl_( new OwningModel<ShapeT, DrawStrategy>( std::move(shape)                                                  , std::move(drawer) )            , []( void* shapeBytes ){
    using Model = OwningModel<ShapeT,DrawStrategy>;
    auto* const model = static_cast<Model*>(shapeBytes);
    delete model;
} );
```
x??

---

#### Template Constructor for Shape Class
This part explains the constructor of the `Shape` class that uses templates and lambdas to initialize various members.

:p How does the template constructor of the `Shape` class work?
??x
The template constructor of the `Shape` class takes a shape object and a drawing strategy, initializes the `pimpl_`, `draw_`, and `clone_` data members using lambda functions. These lambdas are used to set up custom behavior for deletion, drawing, and cloning.

```cpp
template< typename ShapeT, typename DrawStrategy >
Shape( ShapeT shape, DrawStrategy drawer )
    : pimpl_( new OwningModel<ShapeT,DrawStrategy>( std::move(shape)                                                  , std::move(drawer) )            , []( void* shapeBytes ){
        using Model = OwningModel<ShapeT,DrawStrategy>;
        auto* const model = static_cast<Model*>(shapeBytes);
        delete model;
    } )
    , draw_( []( void* shapeBytes ){
        using Model = OwningModel<ShapeT, DrawStrategy>;
        auto* const model = static_cast<Model*>(shapeBytes);
        (*model->drawer_)( model->shape_ );
    } )
    , clone_( []( void* shapeBytes ) -> void* {
        using Model = OwningModel<ShapeT, DrawStrategy>;
        auto* const model = static_cast<Model*>(shapeBytes);
        return new Model( *model );
    } )
{}
```
x??

---

#### Type Erasure Implementation Overview
Type erasure is a technique used to make type information generic and polymorphic. It allows for dynamic dispatch of methods based on the actual types involved, rather than relying on virtual inheritance hierarchies. This approach can provide performance benefits while maintaining flexibility.

Background context: The provided text discusses implementing type erasure in C++ with specific details about handling `Shape` objects using a `std::unique_ptr` to manage memory and lambdas for method dispatch. Type erasure ensures that the correct cleanup behavior is triggered based on the actual type of the object, making it type-safe.

:p What is type erasure, and how does it differ from traditional inheritance hierarchies in C++?
??x
Type erasure allows you to use a single interface or class for multiple types without explicitly defining an inheritance hierarchy. Instead, runtime dispatch mechanisms like lambdas are used to handle different actions based on the actual type of objects involved. This approach can be more efficient and flexible compared to traditional virtual function tables (vtables) in C++.

In contrast, with traditional inheritance hierarchies, you need to define a complete class hierarchy, which incurs overhead due to vtable lookups at runtime.
x??

---
#### Lambda-based Dispatch
Lambdas are used to dynamically determine the correct method implementation based on the actual type of the `OwningModel` object. This ensures that the appropriate cleanup and drawing actions are performed.

Background context: The text explains how lambdas can safely perform a static cast from `void*` to the correct pointer type, given the knowledge of the actual type at runtime. This is crucial for proper memory management and behavior triggering in polymorphic scenarios.

:p How does lambda-based dispatch work in this implementation?
??x
Lambda-based dispatch uses the fact that we know the exact type of the object at runtime due to type erasure. It can perform a static cast from `void*` to the correct pointer type, which is safe because the type is already determined. The lambdas then use this pointer to trigger the appropriate cleanup or drawing actions.

Example:
```cpp
// Assuming OwningModel is a template class with specific types ShapeT and DrawStrategy
auto draw_ = [this](void* ptr) {
    static_cast<OwningModel<ShapeT, DrawStrategy>*>(ptr)->draw();
};

auto clone_ = [this](void* ptr) {
    return new OwningModel<ShapeT, DrawStrategy>(*static_cast<OwningModel<ShapeT, DrawStrategy>*>(ptr));
};
```
Here, the lambdas capture `this` and use it to perform a static cast from `void*` to the correct pointer type before calling the appropriate method.
x??

---
#### Copy Constructor Implementation
The copy constructor is implemented using a "Copy-and-Swap Idiom" where a temporary object is created first, and then swap operations are used to ensure proper resource management.

Background context: The text outlines how the copy constructor should behave in this scenario. It creates a temporary `Shape` object with the copied data and uses swap to update the current object's state. This approach ensures that resources like pointers and unique_ptr are managed correctly during copying.

:p How is the copy constructor implemented in this code snippet?
??x
The copy constructor is implemented using the "Copy-and-Swap Idiom" as follows:

```cpp
Shape(const Shape& other)
    : pimpl_(clone_(other.pimpl_.get()), other.pimpl_.get_deleter())
    , draw_(other.draw_)
    , clone_(other.clone_) {
}
```

Here, `pimpl_`, `draw_`, and `clone_` are initialized by copying the corresponding members from another `Shape` object. The key is to use a temporary object with these values before updating the current object via swap operations.

Example:
```cpp
using std::swap;

Shape copy(other);  // Create a temporary Shape object
swap(pimpl_, copy.pimpl_);  // Swap pimpl_
swap(draw_, copy.draw_);    // Swap draw_
swap(clone_, copy.clone_);  // Swap clone_

return *this;  // Return the current object
```
This ensures that resources are managed correctly, and no double delete or dangling pointers occur.
x??

---
#### Move Operations Implementation
The move operations (move constructor and move assignment operator) use default implementations provided by the compiler.

Background context: The text mentions that for these operations, the default implementations can be used since they do not require any special handling. However, if necessary, custom implementations could handle specific optimizations or resource management tasks.

:p How are the move operations implemented in this code snippet?
??x
The move operations (move constructor and move assignment operator) use default implementations provided by the compiler as follows:

```cpp
Shape(Shape&&) = default;  // Default move constructor
Shape& operator=(Shape&&) = default;  // Default move assignment operator
```

These default implementations handle the necessary optimizations for moving resources without any additional code, ensuring efficiency.

Example:
No custom implementation is needed because the compiler-generated versions are sufficient.
x??

---
#### Performance Comparison with Different Implementations
The performance comparison shows how various strategies and optimizations affect the runtime of different implementations. The text provides a detailed table comparing the time taken by different approaches like object-oriented solutions, `std::function`, and type erasure.

Background context: The performance results indicate that the manual implementation of virtual functions with type erasure provides a significant improvement over traditional inheritance hierarchies and function objects. This is particularly evident in GCC where there's an 25% improvement compared to unoptimized implementations.

:p What are the performance implications of using type erasure in this context?
??x
Using type erasure in this context significantly improves performance by avoiding the overhead associated with virtual functions, such as vtable lookups. The manual implementation of virtual dispatch provides a way to achieve dynamic behavior while managing resources more efficiently.

For example, GCC shows a 25% improvement compared to unoptimized implementations, making it highly efficient for scenarios where type information is erased but still needed at runtime.

Performance table:
```
Table 8-3. Performance results
-----------------------
Type Erasure implementation | GCC 1 1.1 | Clang 1 1.1
Object-oriented solution    | 1.5205 s  | 1.1480 s
std::function              | 2.1782 s  | 1.4884 s
Manual implementation of std::function | 1.6354 s  | 1.4465 s
Classic Strategy           | 1.6372 s  | 1.4046 s
Type Erasure               | 1.5298 s  | 1.1561 s
Type Erasure (SBO)         | 1.3591 s  | 1.0348 s
Type Erasure (manual virtual dispatch) | 1.1476 s  | 1.1599 s
Type Erasure (SBO + manual virtual dispatch) | 1.2538 s  | 1.2212 s
```

This table demonstrates that type erasure with optimized implementations can provide substantial performance benefits.
x??

---

#### Type Erasure Performance Considerations
Type erasure can offer significant performance benefits by reducing indirections and enabling optimizations. However, it comes with its own set of challenges that need to be addressed.

:p What is a potential optimization gain from using type erasure?
??x
By separating the virtual behavior from the encapsulated behavior into value types, we can reduce the number of indirections and enable compiler optimizations, such as inlining, which can improve performance. This approach allows for more efficient memory management and reduced overhead compared to traditional pointer-to-base class polymorphism.
x??

---

#### Small Object Optimization (SBO) with Type Erasure
Using type erasure can help avoid expensive copy operations by implementing small object optimization techniques.

:p How does SBO help in the context of type erasure?
??x
Small object optimization (SBO) is beneficial for type erasure when dealing with small objects, as it avoids the need to allocate memory for each instance. Instead, small objects can be stored inline within their containers or value types, reducing overhead and improving performance.

Example:
```cpp
template<typename T>
struct SmallObject {
    T data;
    // Other members...
};

// Usage in type erasure implementation
SmallObject<Shape> shapeImpl;
```
x??

---

#### Reducing Indirections with Type Erasure
Implementing virtual dispatch manually can reduce the number of indirections, improving performance.

:p How does implementing virtual dispatch manually help in reducing overhead?
??x
By manually managing the dispatch mechanism instead of relying on virtual functions, we can eliminate some of the overhead associated with virtual calls. This approach allows for more direct and efficient function calls, which can significantly improve performance in certain scenarios.
```
// Example: Manually dispatched function call
void useShape(const Shape& shape) {
    draw(shape);
}

// Pseudocode for manual dispatch
template <typename T>
struct Dispatcher {
    void operator()(const T& obj) const { 
        // Directly invoke the method without virtual call overhead 
        invokeDirect(obj); 
    }
};

void invokeDirect(const Circle& circle) {
    // Direct function implementation
}
```
x??

---

#### Setup Costs of Owning Type Erasure Wrappers
Owning type erasure wrappers can introduce significant setup costs, making them less efficient for passing objects to functions.

:p What are the setup costs associated with owning type erasure wrappers?
??x
When using an owning type erasure wrapper, the cost of creating a temporary object and performing a memory allocation can be substantial. This is because the wrapper encapsulates both the behavior and state of the object, leading to additional overhead in terms of copying and allocating memory.

Example:
```cpp
// UseShape function with type erasure wrapper
void useShape(const Shape& shape) {
    draw(shape);
}

int main() {
    Circle circle{3.14};
    auto drawStrategy = [](const Circle& c){ /*...*/ };
    
    // Expensive temporary object creation and memory allocation
    useShape({circle, drawStrategy});
}
```
x??

---

#### Comparing Type Erasure with Inheritance Hierarchies
Type erasure can be an alternative to traditional inheritance hierarchies, offering flexibility but at the cost of setup overhead.

:p How does type erasure compare to using inheritance hierarchies?
??x
While type erasure provides a flexible way to achieve polymorphism without the rigid structure of multiple inheritance or class hierarchies, it comes with its own trade-offs. Specifically, type erasure can introduce significant setup costs due to temporary object creation and memory allocation, whereas traditional inheritance hierarchies are typically cheaper in terms of performance.

Example:
```cpp
// Inheritance hierarchy example
class Shape { /*...*/ };
class Circle : public Shape { /*...*/ };

void useShape(const Shape& shape) {
    // Virtual function call
    draw(shape);
}
```
x??

---

#### Lambda and std::function Performance Considerations
Using `std::function` as a function parameter can be expensive due to deep copying, unlike non-owning abstractions like `std::string_view`.

:p How does using `std::function` compare to `std::string_view` in terms of performance?
??x
`std::function` is an owning abstraction that performs a deep copy of the callable object. This can be expensive due to potential memory allocations and copying, especially for large functions or lambdas. In contrast, `std::string_view` and similar non-owning abstractions are cheap to copy because they consist only of a pointer and size, making them suitable as function parameters.

Example:
```cpp
#include <functional>

int compute(int i, int j, std::function<int(int, int)> op) {
    return op(i, j);
}

// Deep copying with std::function can be expensive
compute(17, 10, [offset=15](int x, int y) { 
    return x + y + offset; 
});
```
x??

---

#### Nonowning Type Erasure Implementation
Background context explaining the concept of type erasure and its importance in modern C++. Discuss how value semantics-based implementations provide beautiful abstractions but come with performance penalties. Mention that for some scenarios, a nonowning implementation is preferred to balance between abstraction and performance.
:p What is the purpose of the `ShapeConstRef` class?
??x
The `ShapeConstRef` class serves as a nonowning type erasure wrapper. It allows you to wrap references to shape objects along with drawing strategies without owning the underlying data, thus avoiding potential lifetime issues and improving performance. This implementation leverages manual virtual dispatch for efficient operations.

```cpp
#include <memory>

class ShapeConstRef {
public:
    template<typename ShapeT, typename DrawStrategy>
    ShapeConstRef(ShapeT& shape, DrawStrategy& drawer)
        : shape_{std::addressof(shape)}
        , drawer_{std::addressof(drawer)}
        , draw_{[](void const* shapeBytes, void const* drawerBytes) {
            auto const* shape = static_cast<ShapeT const*>(shapeBytes);
            auto const* drawer = static_cast<DrawStrategy const*>(drawerBytes);
            (*drawer)(*shape);
          }}
    {}

private:
    friend void draw(ShapeConstRef const& shape)
    {
        shape.draw_(shape.shape_, shape.drawer_);
    }

    using DrawOperation = void(void const*, void const*);

    void const* shape_{nullptr};
    void const* drawer_{nullptr};
    DrawOperation* draw_{nullptr};
};
```
x??

---
#### Template Constructor of ShapeConstRef
Background context explaining the template constructor used in `ShapeConstRef` and its role. Highlight how it takes references to a shape object and drawing strategy.

:p What does the template constructor of `ShapeConstRef` do?
??x
The template constructor of `ShapeConstRef` initializes the wrapper with references to a shape object and a drawing strategy. It captures these references in member variables, ensuring that they are not copied but referenced directly. This setup is crucial for nonowning type erasure.

```cpp
template<typename ShapeT, typename DrawStrategy>
ShapeConstRef(ShapeT& shape, DrawStrategy& drawer)
    : shape_{std::addressof(shape)}
    , drawer_{std::addressof(drawer)}
    , draw_{[](void const* shapeBytes, void const* drawerBytes) {
        auto const* shape = static_cast<ShapeT const*>(shapeBytes);
        auto const* drawer = static_cast<DrawStrategy const*>(drawerBytes);
        (*drawer)(*shape);
      }}
{}
```
x??

---
#### Friend Function `draw` in ShapeConstRef
Background context explaining the purpose of the friend function and how it is used for manual virtual dispatch.

:p What is the role of the `friend void draw(ShapeConstRef const& shape)` function?
??x
The `friend void draw(ShapeConstRef const& shape)` function serves as a way to manually invoke the drawing operation stored in the `draw_` member. By marking it as a friend, it can access private members like `draw_`, allowing for efficient dispatch of the drawing operation.

```cpp
friend void draw(ShapeConstRef const& shape)
{
    shape.draw_(shape.shape_, shape.drawer_);
}
```
x??

---
#### Draw Operation Function Pointer
Background context explaining how the function pointer is used to store the drawing logic and its role in type erasure.

:p How does the `DrawOperation` function pointer work in `ShapeConstRef`?
??x
The `DrawOperation` function pointer stores a lambda function that encapsulates the actual drawing logic. This allows for efficient, nonowning type erasure by using a simple function call to invoke the appropriate drawing strategy on the given shape.

```cpp
using DrawOperation = void(void const*, void const*);

void const* shape_{nullptr};
void const* drawer_{nullptr};
DrawOperation* draw_{nullptr};
```
x??

---
#### Handling Rvalues and Lifetimes
Background context discussing the handling of rvalues in `ShapeConstRef` and the trade-offs made to ensure safe lifetime management.

:p How does `ShapeConstRef` handle temporary objects?
??x
To handle temporary objects, `ShapeConstRef` takes its arguments by reference-to-non-const, preventing the passing of rvalues. However, this approach also means that it cannot protect against all possible lifetime issues with lvalues. By not allowing rvalues, the implementation ensures a safer handling of shape and drawing strategy references.

```cpp
// No explicit handling for temporaries here as it's already constrained by reference-to-non-const.
```
x??

---

