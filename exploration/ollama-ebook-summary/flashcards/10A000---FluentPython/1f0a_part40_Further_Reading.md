# Flashcards: 10A000---FluentPython_processed (Part 40)

**Starting Chapter:** Further Reading

---

#### Parameterized Decorator Implementation as Class

Background context: The provided text discusses an alternative approach to implementing parameterized decorators using a class. This method is considered more suitable for non-trivial decorators, but this example illustrates the basic idea.

Explanation of the concept:
- A decorator can be implemented as a class instead of a function.
- This makes it easier to handle complex scenarios and retain state across multiple calls or other advanced features.

:p How does implementing a parameterized clock decorator as a class differ from its function-based implementation?

??x
Implementing a parameterized clock decorator as a class involves creating a `clock` class that has an `__init__` method for setting up the decorator's parameters and a `__call__` method to make instances of this class callable. This allows for more flexibility, especially when dealing with stateful decorators.

Code example:
```python
import time

DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result} '

class clock:
    def __init__(self, fmt=DEFAULT_FMT):
        self.fmt = fmt
    
    def __call__(self, func):
        def clocked(*_args):
            t0 = time.perf_counter()
            _result = func(*_args)
            elapsed = time.perf_counter() - t0
            name = func.__name__
            args = ', '.join(repr(arg) for arg in _args)
            result = repr(_result)
            print(self.fmt.format(**locals()))
            return _result
        return clocked

@clock('{name}({args}) dt={elapsed:0.3f}s')
def snooze(seconds):
    time.sleep(seconds)

for i in range(3):
    snooze(.123)
```

This class-based decorator can be instantiated with different formats, making it more flexible for various use cases.

x??

---
#### Drop-In Replacement

Background context: The text mentions that the class-based implementation of a parameterized clock decorator is intended as a drop-in replacement for the function-based version presented earlier. This implies both implementations serve the same purpose but are structured differently.

:p What does "drop-in replacement" mean in this context?

??x
A drop-in replacement means that the new implementation (in this case, the class-based `clock` decorator) can be substituted directly for the old one without requiring any changes to the calling code. This implies both versions of the decorator should behave identically when applied to functions.

For example:
```python
# Function-based clock decorator from earlier examples
@clock('{name}({args}) dt={elapsed:0.3f} s')
def snooze(seconds):
    time.sleep(seconds)

# Class-based drop-in replacement for the same purpose
@clock.DEFAULT_FMT(clock)  # Assuming DEFAULT_FMT is set to match the format used in function version
def snooze(seconds):
    time.sleep(seconds)
```

In this context, `clock.DEFAULT_FMT` likely refers to a default formatting string that can be passed to the class-based clock decorator.

x??

---
#### Comparison of Function-Based and Class-Based Decorators

Background context: The text compares two different ways to implement a parameterized clock decorator—a function-based approach used in earlier examples and a class-based approach described later. This comparison highlights the trade-offs between simplicity and flexibility.

:p What are some advantages of implementing a parameterized clock decorator as a class?

??x
Advantages of implementing a parameterized clock decorator as a class include:

1. **Flexibility**: The class can handle more complex stateful operations, such as maintaining state across multiple calls or logging.
2. **Readability and Maintainability**: Code using classes might be easier to read and maintain due to the clear separation of concerns.
3. **State Retention**: Classes allow for retaining state between function calls, which is useful for decorators that need to remember information from previous invocations.

For instance:
```python
class clock:
    def __init__(self, fmt=DEFAULT_FMT):
        self.fmt = fmt
    
    def __call__(self, func):
        # Implementation details...
```

This class-based approach provides more room for future enhancements and makes the code easier to extend or modify.

x??

---
#### Strategy Design Pattern Application

Background context: The text mentions that the idea of registration decorators will be applied in an implementation of the Strategy design pattern in Chapter 10. This suggests a connection between decorator concepts and broader software design patterns.

:p How does the concept of registration decorators relate to the Strategy design pattern?

??x
Registration decorators are relevant to the Strategy design pattern because they allow for registering different strategies or behaviors dynamically. In Python, this can be achieved by using decorators that register functions as strategy implementations.

For example:
```python
# Simple registration decorator (simplified version)
def register_strategy(strategy):
    if not hasattr(registry, 'strategies'):
        registry.strategies = []
    registry.strategies.append(strategy)
    return strategy

@register_strategy
def strategy_a():
    pass

@register_strategy
def strategy_b():
    pass

registry = ModuleRegistry()
```

In the context of the Strategy pattern, registration decorators enable the dynamic addition and removal of strategies. This flexibility is crucial for implementing different algorithms or behaviors in a flexible way.

x??

---

#### Import Time vs Runtime
Import time refers to when a module is loaded and its contents are evaluated, while runtime means when the code within that module is executed. Understanding this difference helps in grasping how decorators behave at various stages of program execution.

:p What distinguishes import time from runtime?
??x
Import time occurs during the loading of a module where the code inside it is parsed but not yet executed. Runtime refers to when the actual execution of the imported code happens, which can be later after the initial import.

For example:
```python
import math

# Import time: The 'math' module is loaded and its contents are evaluated.
print(math.sqrt(4))  # Runtime: This line executes during runtime.
```
x??

---

#### Variable Scoping
Variable scoping in Python refers to how variables are accessed within the program. It involves lexical scoping, where each variable's scope is determined by where it is declared.

:p What does variable scoping refer to?
??x
Variable scoping determines which parts of a program can access a particular variable based on its declaration location. In Python, this typically follows lexical scoping rules, meaning the scope of variables is defined by their position in the code.

For example:
```python
def outer_function():
    x = 10  # This 'x' has local scope within the function.
    def inner_function():
        y = 20  # 'y' has a different scope within this nested function.
    print(x)  # Accessing 'x' is allowed here because it's in its scope.

outer_function()
```
x??

---

#### Closures
Closures allow functions to remember and access variables from their lexical scope even when they are executed outside that scope. They consist of a function object, free variables used by the function, and an environment recording where those variables were bound at closure creation time.

:p What is a closure in Python?
??x
A closure is a function object that remembers values in enclosing scopes even if they are not present in memory. It consists of three main components: 
1. A nested function.
2. Free variables used by the nested function from its enclosing scope.
3. An environment record of where those free variables were bound at closure creation time.

For example:
```python
def outer_function(x):
    def inner_function(y):
        return x + y  # 'x' is a free variable in this context.
    return inner_function

add_five = outer_function(5)
print(add_five(10))  # Output: 15
```
x??

---

#### Nonlocal Keyword
The `nonlocal` keyword allows for modifying variables from an enclosing scope without using the global declaration. This is particularly useful in nested functions where you need to modify a variable that exists outside their local scope but not at the module level.

:p What does the `nonlocal` keyword do?
??x
The `nonlocal` keyword indicates that a function should refer to an existing variable from its enclosing scope rather than creating a new one. It is used in nested functions where you need to modify a non-global variable defined outside the innermost scope of those functions.

For example:
```python
def outer():
    x = 10

    def inner():
        nonlocal x  # This allows 'x' to be modified.
        x = 20
    inner()
    print(x)  # Output: 20 (modified)

outer()
```
x??

---

#### Parameterized Decorators
Parameterized decorators are those that take arguments. They typically involve at least two nested functions and often use `functools.wraps` to preserve the metadata of the original function being decorated.

:p What is a parameterized decorator?
??x
A parameterized decorator is one that accepts additional parameters, allowing it to customize its behavior for different functions or scenarios. These decorators usually involve at least two nested functions: an outer wrapper and an inner wrapper. They often use `functools.wraps` to maintain the metadata of the original function being decorated.

For example:
```python
from functools import wraps

def decorator_factory(arg):
    def real_decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            print(f"Arg: {arg}")
            return func(*args, **kwargs)
        return wrapped
    return real_decorator

@decorator_factory("Hello")
def say_hello():
    return "Hello, world!"

print(say_hello())  # Output: Arg: Hello; Hello, world!
```
x??

---

#### Stacked Decorators
Stacked decorators refer to applying multiple decorators to a single function. The order of application is important because the first decorator applied is the last one that gets executed.

:p What are stacked decorators?
??x
Stacked decorators involve applying more than one decorator to the same function. The order in which they are written (from left to right) determines their execution order. Typically, the innermost decorator is the first one to be called and the outermost decorator is the last one that gets executed.

For example:
```python
def decorator1(func):
    def wrapper():
        print("Decorator 1")
        return func()
    return wrapper

def decorator2(func):
    def wrapper():
        print("Decorator 2")
        return func()
    return wrapper

@decorator1
@decorator2
def say_hello():
    return "Hello, world!"

say_hello()  # Output: Decorator 2; Decorator 1; Hello, world!
```
x??

---

#### Class-Based Decorators
Class-based decorators provide a way to create more readable and maintainable decorator implementations. They are especially useful for complex or sophisticated decorators.

:p What is a class-based decorator?
??x
A class-based decorator implements the `__call__` method in a class, allowing instances of that class to behave like functions when applied as decorators. This approach can make the code more readable and maintainable, particularly for complex or sophisticated decorators.

For example:
```python
class Logger:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f"Calling {self.func.__name__}")
        return self.func(*args, **kwargs)

@Logger
def say_hello():
    return "Hello, world!"

say_hello()  # Output: Calling say_hello; Hello, world!
```
x??

---

#### functools.cached_property
The `cached_property` decorator from the `functools` module allows for caching the result of a property method. This is useful when the value returned by the method doesn't change during the object's lifetime and can be cached to avoid redundant calculations.

:p What does `functools.cached_property` do?
??x
The `cached_property` decorator from the `functools` module caches the result of a property method so that it only needs to be computed once. This is particularly useful for methods whose return values don't change during the object's lifetime, thereby avoiding redundant calculations.

For example:
```python
from functools import cached_property

class MyClass:
    @cached_property
    def expensive_computation(self):
        print("Computing...")
        return 42

obj = MyClass()
print(obj.expensive_computation)  # Output: Computing...; 42
print(obj.expensive_computation)  # No "Computing..." because it's cached.
```
x??

---

#### functools.singledispatch
The `singledispatch` decorator from the `functools` module allows creating a single-dispatch generic function, where different methods can be defined for different argument types.

:p What is `functools.singledispatch`?
??x
The `singledispatch` decorator from the `functools` module enables creating a single-dispatch generic function. A single-dispatch generic function has multiple methods corresponding to different argument types, and it dynamically dispatches to the appropriate method based on the type of the first positional argument.

For example:
```python
from functools import singledispatch

@singledispatch
def print_value(x):
    raise NotImplementedError("No implementation for this type")

@print_value.register(int)
def _(x):
    print(f"Printing integer: {x}")

@print_value.register(str)
def _(x):
    print(f"Printing string: {x}")

print_value(42)  # Output: Printing integer: 42
print_value("Hello")  # Output: Printing string: Hello
```
x??

---

#### Metaprogramming and Decorators
Metaprogramming involves writing programs that can read, generate, or transform other programs. Decorators are a powerful metaprogramming technique in Python, allowing for dynamic behavior modification of functions.

:p What is the relationship between decorators and metaprogramming?
??x
Decorators are a key aspect of metaprogramming in Python because they enable dynamically modifying function behavior at runtime. By wrapping functions, decorators can add functionality or change how functions behave without altering their original code.

For example:
```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello, world!")

say_hello()  # Output: Something is happening before the function is called.; Hello, world!; Something is happening after the function is called.
```
x??

---

#### Python Decorator Library Wiki Page
The Python Decorator Library wiki page contains numerous examples of decorators and their usage. Although some techniques may be outdated, it remains a valuable resource for inspiration.

:p What does the Python Decorator Library wiki page contain?
??x
The Python Decorator Library wiki page houses many examples of decorators and their various uses. While some of the shown techniques might have been superseded by newer approaches, the page still offers excellent inspiration and guidance on decorator usage in different scenarios.

For example:
```python
from functools import wraps

def log_function_data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_function_data
def say_goodbye():
    print("Goodbye, world!")

say_goodbye()  # Output: Calling say_goodbye; Goodbye, world!
```
x??

---

#### Michele Simionato's decorator Package
Michele Simionato’s `decorator` package aims to simplify the usage of decorators for average programmers and showcase various non-trivial examples.

:p What is Michele Simionato's decorator package?
??x
Michele Simionato’s `decorator` package simplifies working with decorators, providing a more user-friendly approach. It includes utilities that make it easier to create and apply decorators, especially when dealing with complex or sophisticated use cases.

For example:
```python
from decorator import decorator

@decorator
def debug(func, *args, **kwargs):
    print(f"Calling {func.__name__}({', '.join(map(str, args))})")
    return func(*args, **kwargs)

@debug
def multiply(a, b):
    return a * b

multiply(3, 4)  # Output: Calling multiply(3, 4); 12
```
x??

---

#### wrapt Module
The `wrapt` module simplifies the implementation of decorators and dynamic function wrappers. It supports introspection and behaves correctly when further decorated or used as attribute descriptors.

:p What is the `wrapt` module?
??x
The `wrapt` module provides a simplified way to implement decorators, particularly useful for handling complex scenarios like introspection and proper behavior under various decorator combinations. It helps in creating robust decorators that work seamlessly with other decorators and maintain correct function metadata.

For example:
```python
from wrapt import decorator

@decorator
def debug(wrapped, instance, args, kwargs):
    print(f"Calling {wrapped.__name__}")
    return wrapped(*args, **kwargs)

class MyClass:
    @debug
    def method(self, x):
        return x * 2

mc = MyClass()
print(mc.method(10))  # Output: Calling method; 20
```
x??

#### Single-Dispatch Generic Functions
Background context: PEP 443 introduces single-dispatch generic functions, which provide a mechanism for function overloading based on the type of one argument. This concept is different from multi-dispatch where the dispatch is based on multiple arguments.

:p What are single-dispatch generic functions?
??x
Single-dispatch generic functions allow functions to be overloaded based on the type of one argument. The selection of the appropriate method for a function call depends only on the first argument, while other arguments do not affect the choice.
x??

---

#### Multiple-Dispatch Generic Functions
Background context: Guido van Rossum's blog post from March 2005 describes an implementation of generic functions (multimethods) using decorators. This code supports multiple-dispatch, which means the function dispatch is based on more than one positional argument.

:p What does multiple-dispatch in Python refer to?
??x
Multiple-dispatch refers to a situation where a function's method selection depends on more than one argument. In Guido van Rossum’s example, this involves using decorators to create generic functions that can handle different types of arguments.
x??

---

#### Reg Framework for Generic Functions
Background context: The Reg framework by Martijn Faassen provides a modern and production-ready implementation of multiple-dispatch generic functions in Python.

:p What is the Reg framework used for?
??x
The Reg framework is utilized to implement multiple-dispatch generic functions, offering a robust solution that is suitable for real-world applications.
x??

---

#### Free Variables Evaluation
Background context: In any language with first-class functions, free variables are evaluated based on their environment. Dynamic scope evaluates these variables in the function's invocation environment.

:p What is dynamic scope?
??x
Dynamic scope refers to an evaluation strategy where a function's free variables are looked up in the environment where the function is invoked, rather than at definition.
x??

---

#### Example with make_averager Function
Background context: The example shows how to implement and use the `make_averager` function, which maintains state through a closure.

:p How does the `make_averager` function work?
??x
The `make_averager` function works by maintaining an internal list `series` using a closure. Each call to the returned function updates this list and calculates the average based on its contents.
```python
def make_averager():
    series = []

    def averager(new_value):
        series.append(new_value)
        return sum(series) / len(series)

    return averager
```
x??

---

#### SOAP Box: Dynamic Scope vs. Closures
Background context: The discussion contrasts dynamic scope, where free variables are evaluated in the function's invocation environment, with closures, which encapsulate and maintain state.

:p What is the issue with using dynamic scope for functions?
??x
The issue with dynamic scope is that it exposes internal details of a function to its users. This can lead to unintended side effects or bugs if not managed carefully.
x??

---

#### LaTeX and Dynamic Scope
Background context: The text draws an analogy between dynamic scoping in Python and the use of variables in LaTeX, which also uses dynamic scope.

:p Why are variables in LaTeX confusing?
??x
Variables in LaTeX are confusing because they follow dynamic scope rules. This means their values can change unpredictably based on where they are used in the document, making them difficult to manage.
x??

---

#### Emacs Lisp and Dynamic Scope
Background context: The text mentions that Emacs Lisp uses dynamic scope by default.

:p What does dynamic binding mean in Emacs Lisp?
??x
Dynamic binding in Emacs Lisp means that variable values are looked up at runtime based on the environment where the function is invoked, rather than at definition.
x??

---

#### Lisp and Dynamic Scope Origins
Background context: The text discusses how John McCarthy's original design of Lisp used dynamic scope, leading to potential issues with higher-order functions.

:p Why did John McCarthy choose dynamic scope for Lisp?
??x
John McCarthy chose dynamic scope in Lisp because it was easier to implement at the time. However, this choice led to complexities, especially when dealing with higher-order functions.
x??

---

#### Paul Graham's Commentary on Dynamic Scope
Background context: The text references Paul Graham’s commentary explaining the dangers and complexities of dynamic scoping.

:p What did Paul Graham say about dynamic scope?
??x
Paul Graham highlighted that even in McCarthy's original paper, which introduced Lisp, there was an error due to the tricky nature of dynamic scoping.
x??

---

#### Dynamic Scope vs Lexical Scope
In 1960, McCarthy was not fully aware of the implications of dynamic scope. Dynamic scope remained in Lisp implementations for a surprisingly long time—until Sussman and Steele developed Scheme in 1975. Lexical scope does not complicate the definition of `eval` very much, but it may make compilers harder to write.
:p What is the difference between dynamic scope and lexical scope?
??x
Dynamic scope binds variables to the execution stack at runtime, making them accessible based on where they are called from in the call stack. Lexical scope, on the other hand, binds variables based on their location in the code, meaning that a variable's value is determined by the nearest enclosing scope where it was defined.
??x
```lisp
;; Example of dynamic scope (Common Lisp)
(defun foo ()
  (let ((x 10))
    (lambda () x)))

;; Example of lexical scope (Scheme)
(let ((x 20))
  (lambda () x))
```
x??

---

#### Python Decorators and Closures
Python decorators are a way to modify the behavior of functions or classes without permanently modifying them. Lexical scope complicates the implementation of languages with first-class functions, because it requires the support of closures. However, lexical scope makes source code easier to read.
:p What is a decorator in Python?
??x
A decorator in Python is a design pattern that allows a user to add new functionality to an existing object without modifying its structure. Decorators are a very powerful and useful tool in Python since they allow the modification of functions, methods, or classes without any permanent change to the source code.
??x
```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```
x??

---

#### Closures and Python Lambdas
Python lambdas did not provide closures until Python 2.2, which contributed to a bad name among functional-programming geeks in the blogosphere.
:p What is a closure in programming?
??x
A closure is a function object that has access to variables from its own scope, even when that function is called outside that scope. In other words, closures allow you to return functions that remember their lexical environment, which can be useful for creating private data or maintaining state.
??x
```python
def outer_function(x):
    def inner_function(y):
        return x + y
    return inner_function

closure = outer_function(10)
print(closure(5))  # Output: 15
```
x??

---

#### Python Class Decorators
Class decorators are a way to modify the behavior of classes or objects without modifying their structure permanently. They can be used to add functionality or change how classes work.
:p What is a class decorator in Python?
??x
A class decorator in Python is a function that takes a class as its input and returns a modified version of the class, which can include adding new methods, changing existing ones, or modifying attributes.
??x
```python
def my_class_decorator(cls):
    cls.new_attribute = "Hello!"
    return cls

@my_class_decorator
class MyClass:
    pass

print(MyClass.new_attribute)  # Output: Hello!
```
x??

---

#### Static Checking Tools and Python
Static checking tools, like Mypy, can sometimes discourage the use of dynamic features in Python by complaining about unused variables or multiple functions with the same name.
:p How do static checking tools handle dynamic features in Python?
??x
Static checking tools can be restrictive when dealing with Python's dynamic nature. For instance, Mypy may complain if you have multiple functions with the same name, as it enforces strict typing and cannot infer function overloading like some other languages.
??x
```python
# Example of a static checking complaint in Mypy
def func(x: int):
    return x + 10

def func(y: str):
    return y.upper()

# This will raise an error in Mypy because it sees multiple functions with the same name
```
x??

---

#### Python Decorator Design Pattern Implementation
Python decorators are implemented differently from the classic Decorator design pattern, but they can still be seen as analogous. The decorator function acts like a concrete Decorator subclass, and the inner function returns wraps the original function.
:p How do Python decorators relate to the classic Decorator design pattern?
??x
Python decorators can be seen as an implementation of the Decorator design pattern where the decorator function acts as a concrete Decorator subclass. The returned function (which is analogous to a component in the pattern) wraps the function to be decorated and conforms to its interface by accepting the same arguments.
??x
```python
def decorator_function(original_func):
    def wrapper(*args, **kwargs):
        print(f"Before {original_func.__name__} is called")
        result = original_func(*args, **kwargs)
        print(f"After {original_func.__name__} is called")
        return result
    return wrapper

@decorator_function
def my_function():
    print("Inside my_function")

my_function()
```
x??

---

#### Design Patterns and First-Class Functions

Background context: This section discusses how design patterns, traditionally implemented using classes, can be refactored to use functions as objects (first-class functions) in languages like Python. The primary goal is to reduce boilerplate code while maintaining readability and functionality.

:p What are the benefits of using first-class functions over traditional class-based approaches for implementing design patterns?
??x
The benefits include shorter and easier-to-read code, reduced boilerplate, and better alignment with functional programming paradigms. Using first-class functions can make the implementation more concise and flexible.
```python
def strategy(context):
    if context.condition:
        return lambda: action1()
    else:
        return lambda: action2()

context = {"condition": True}
action = strategy(context)
print(action())  # Calls action1()
```
x??

---

#### Strategy Pattern with Functions

Background context: The Strategy pattern allows selecting an algorithm at runtime. Traditionally, it involves creating multiple classes implementing a common interface or inheriting from a base class. In this section, we refactor the implementation using functions as objects.

:p How can you implement the Strategy pattern in Python using first-class functions?
??x
You define strategies as functions that encapsulate specific behavior and return a function to execute based on conditions.
```python
def strategy_a():
    print("Executing Strategy A")

def strategy_b():
    print("Executing Strategy B")

# Context chooses which strategy to use
context = "A"
if context == "A":
    action = strategy_a
else:
    action = strategy_b

action()  # Calls the chosen strategy
```
x??

---

#### Command Pattern Simplification

Background context: The Command pattern encapsulates a request as an object, thereby allowing parameterization of clients with queues, requests, and operations. Traditionally, it involves classes for commands and invokers. Here, we simplify by using functions directly.

:p How can the Command pattern be simplified in Python?
??x
In Python, you can use functions to encapsulate actions, which can then be passed around like any other object.
```python
def execute_command(command):
    command()

# Define concrete commands as functions
class Command:
    def __init__(self, function):
        self.function = function

    def execute(self):
        self.function()

execute_command(lambda: print("Executing a command"))
```
x??

---

#### Template Method Pattern

Background context: The Template Method pattern defines the skeleton of an algorithm in a method, allowing subclasses to redefine certain steps without changing the algorithm's structure. In Python, this can be achieved with functions and closures.

:p How does the Template Method pattern work in Python?
??x
In Python, you define a template function that includes a series of hooks where specific behavior can be implemented by subclasses or other objects.
```python
def template_method(step1, step2):
    print("Executing Step 1")
    step1()
    print("Executing Step 2")
    step2()

# Define steps as functions
def step1():
    print("Customized Step 1")

def step2():
    print("Default Step 2")

template_method(step1, step2)
```
x??

---

#### Visitor Pattern

Background context: The Visitor pattern allows adding new operations to existing class hierarchies without altering the classes. In dynamic languages like Python, this can be achieved more simply by using functions.

:p How is the Visitor pattern implemented in Python?
??x
In Python, you can directly pass a function as an argument to another function or method, allowing it to define custom behavior.
```python
class Element:
    def accept(self, visitor):
        visitor.visit(self)

def visit(element):
    print(f"Visited: {element}")

e = Element()
e.accept(visit)
```
x??

---

#### Strategy Pattern Overview
Background context: The Strategy pattern is a behavioral design pattern that allows an object to alter its behavior when its internal state changes. It provides a way to define a family of algorithms, encapsulate each one, and make them interchangeable. This is particularly useful in scenarios where you need different ways to compute something based on certain conditions.

If applicable, add code examples with explanations.
:p What is the Strategy pattern?
??x
The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. It allows the algorithm to vary independently from clients that use it.

Example class diagram (UML):
```
Context ----> Strategy
      |          |
Promotion  <---- FidelityPromo, BulkItemPromo, LargeOrderPromo
```

Code example:
```python
class Promotion(ABC):  # the Strategy: an abstract base class
    @abstractmethod
    def discount(self, order: Order) -> Decimal:
        pass

class FidelityPromo(Promotion):  # first Concrete Strategy
    "5 percent discount for customers with 1000 or more fidelity points"
    
    def discount(self, order: Order) -> Decimal:
        rate = Decimal('0.05')
        if order.customer.fidelity >= 1000:
            return order.total() * rate
        return Decimal(0)
```
x??

---
#### Context Class - Order
Background context: The `Order` class acts as the context in the Strategy pattern. It delegates its responsibilities to a pluggable strategy (a promotional discount algorithm) that can be switched at runtime based on the customer's criteria.

:p How does the `Order` class work?
??x
The `Order` class encapsulates an order and provides a service by delegating some computations to interchangeable components that implement alternative algorithms. It uses a `promotion` strategy, which is either selected or passed in during instantiation.

Code example:
```python
class Order(NamedTuple):  # the Context
    customer: Customer
    cart: Sequence[LineItem]
    promotion: Optional['Promotion'] = None

    def total(self) -> Decimal:
        totals = (item.total() for item in self.cart)
        return sum(totals, start=Decimal(0))

    def due(self) -> Decimal:
        if self.promotion is None:
            discount = Decimal(0)
        else:
            discount = self.promotion.discount(self)
        return self.total() - discount
```
x??

---
#### Concrete Strategy - FidelityPromo
Background context: The `FidelityPromo` class implements the `Promotion` interface and provides a specific promotional strategy based on customer's fidelity points.

:p What does the `FidelityPromo` class do?
??x
The `FidelityPromo` class provides a 5% discount for customers with 1000 or more fidelity points. If the customer has fewer than 1000 points, no discount is applied.

Code example:
```python
class FidelityPromo(Promotion):  # first Concrete Strategy
    "5 percent discount for customers with 1000 or more fidelity points"
    
    def discount(self, order: Order) -> Decimal:
        rate = Decimal('0.05')
        if order.customer.fidelity >= 1000:
            return order.total() * rate
        return Decimal(0)
```
x??

---
#### Concrete Strategy - BulkItemPromo
Background context: The `BulkItemPromo` class implements the `Promotion` interface and provides a specific promotional strategy based on the quantity of line items.

:p What does the `BulkItemPromo` class do?
??x
The `BulkItemPromo` class applies a 10% discount to each line item with 20 or more units in the same order. The total discount is calculated by summing up the discounted amounts for all such items.

Code example:
```python
class BulkItemPromo(Promotion):  # second Concrete Strategy
    "10 percent discount for each LineItem with 20 or more units"
    
    def discount(self, order: Order) -> Decimal:
        discount = Decimal(0)
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * Decimal('0.1')
        return discount
```
x??

---
#### Concrete Strategy - LargeOrderPromo
Background context: The `LargeOrderPromo` class implements the `Promotion` interface and provides a specific promotional strategy based on the number of distinct items in an order.

:p What does the `LargeOrderPromo` class do?
??x
The `LargeOrderPromo` class applies a 7% discount to orders with at least 10 distinct items. The total discount is calculated as a percentage of the order's total amount based on this criterion.

Code example:
```python
class LargeOrderPromo(Promotion):  # third Concrete Strategy
    "7 percent discount for orders with 10 or more distinct items"
    
    def discount(self, order: Order) -> Decimal:
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * Decimal('0.07')
        return Decimal(0)
```
x??

---

