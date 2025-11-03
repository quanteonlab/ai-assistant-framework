# Flashcards: 10A000---FluentPython_processed (Part 36)

**Starting Chapter:** Further Reading

---

#### Guido van Rossum's Perspective on Type Hints
Background context: In the provided text, Guido van Rossum discusses his views on type hints. He believes that while type hints are beneficial and have their place, they should not be mandatory for all Python code.

:p What is Guido van Rossum’s stance on type hints?
??x
Guido van Rossum believes in providing developers with the freedom to choose whether or not to use type hints based on the context. He emphasizes that while type hints can be useful, particularly when writing unit tests, they are not always necessary and should not become a moral obligation.

```python
def greet(name: str) -> None:
    print(f"Hello, {name}!")
```
x??

---

#### Importance of Testing vs. Type Hints
Background context: The text mentions that type hints should be used when unit tests are worth writing. This suggests a balance between the two tools, with each serving different purposes.

:p According to Bernát Gábor, in what situations are type hints not helpful?
??x
According to Bernát Gábor, type hints are not particularly useful during exploratory coding or rapid prototyping where testing is not immediately required. The use of tests and type hints becomes a hindrance in these scenarios.

```python
def add(a: int, b: int) -> int:
    return a + b

# Example usage without explicit tests
result = add(3, 4)
print(result)
```
x??

---

#### Type Hints for Unit Testing
Background context: Bernát Gábor suggests that type hints are beneficial when unit testing is relevant. This indicates that while not mandatory, they can improve code quality and maintainability.

:p What does Bernát Gábor suggest about the use of type hints?
??x
Bernát Gábor recommends using type hints whenever unit tests are worth writing. Type hints help in making the code more robust and easier to understand, particularly when dealing with complex functions that require validation through tests.

```python
def process_data(data: list[int]) -> dict[str, int]:
    result = {}
    for item in data:
        if isinstance(item, int):
            result[f"item_{len(result)}"] = item
    return result

# Example usage with a simple test
data = [10, 20, "not an integer"]
print(process_data(data))
```
x??

---

#### Resources for Type Hints
Background context: The text provides several resources and references for those interested in learning more about type hints. These include articles, guides, and official documentation.

:p What are some recommended resources for learning about Python type hints?
??x
Some recommended resources for learning about Python type hints include:
- Gábor Bernát's post "The state of type hints in Python"
- Geir Arne Hjelle's guide "Python Type Checking (Guide)"
- Hypermodern Python Chapter 4: Typing by Claudio Jolowicz, which covers runtime type checking validation
- Mypy documentation, particularly the tutorial and reference pages about Python typing

```python
# Example usage of a type hint from the Mypy documentation
from typing import List, Dict

def process_data(data: List[int]) -> Dict[str, int]:
    result = {}
    for item in data:
        if isinstance(item, int):
            result[f"item_{len(result)}"] = item
    return result

# Testing the function
data = [10, 20, "not an integer"]
print(process_data(data))
```
x??

---

#### Type Hints and Coding Style
Background context: The text discusses concerns about the impact of type hints on coding style. It suggests that while some APIs benefit from type hints, they may not be necessary in all situations.

:p What are the potential drawbacks mentioned regarding the use of type hints?
??x
The potential drawback mentioned is that an overreliance on type hints could alter Python's coding style and make it more rigid or verbose. This might detract from the flexibility and simplicity that made Python appealing to many developers in the first place.

```python
# Example function with a type hint
def greet(name: str) -> None:
    print(f"Hello, {name}!")

greet("World")
```
x??

---

#### Code Examples for Type Hints
Background context: The text includes various examples of functions and their usage to illustrate the application of type hints.

:p Provide an example function with type hints.
??x
Here is an example of a Python function using type hints:

```python
def add(a: int, b: int) -> int:
    return a + b

# Example usage
result = add(3, 4)
print(result)
```

This function defines two integer parameters and returns an integer. The type hints help in understanding the expected input and output types.
x??

---

#### Common Issues with Type Hints
Background context: The text mentions that there are "Common issues and solutions" available for those who want to dive deeper into using type hints effectively.

:p Where can one find common issues and solutions related to type hints?
??x
One can find common issues and solutions related to type hints in the Mypy documentation. This resource provides comprehensive guides and references on Python typing, including specific examples of typical problems and their resolutions.

```python
# Example function with a type hint from the Common issues and solutions guide
from typing import List

def append_to_list(lst: List[int], item: int) -> None:
    lst.append(item)

my_list = [1, 2, 3]
append_to_list(my_list, 4)
print(my_list)
```

This function demonstrates how to use type hints effectively when working with lists.
x??

---

#### Alternatives to Type Hints
Background context: The text also touches on the idea that not all coding scenarios require or benefit from type hints. It suggests using them selectively.

:p In what situations might one choose not to use type hints?
??x
One might choose not to use type hints when writing exploratory code, doing quick prototyping, or in cases where unit tests are not immediately required. Type hints can be a hindrance in these scenarios as they add unnecessary complexity and reduce flexibility.

```python
# Example of exploratory coding without type hints
def process_data(data):
    result = {}
    for item in data:
        if isinstance(item, int):
            result[f"item_{len(result)}"] = item
    return result

data = [10, 20, "not an integer"]
print(process_data(data))
```

In this example, type hints are not used as the primary focus is on rapid experimentation.
x??

---

#### Linguistic Relativity and Programming Languages
Background context explaining how different programming languages can influence problem-solving approaches. The example given is about moving from Applesoft BASIC to Elixir, where recursion became more natural due to idiomatic use.

:p How might a programmer's approach be influenced by the language they primarily use?
??x
A programmer's approach could be significantly influenced by their primary language of choice. For instance, working with a language that lacks direct support for certain features (like recursion in early BASIC) can limit how these concepts are considered or used. Conversely, being familiar with languages that promote and utilize specific patterns (like Elixir with functional programming paradigms) can make those approaches more natural and preferred even when coding in other languages.
x??

---

#### Impact of Type Hints on Library Design
Explanation of how strict type hint enforcement might affect the design decisions of libraries like `requests`. It highlights that libraries with Pythonic APIs are often less inclined to adopt typing systems due to perceived minimal value.

:p How might insisting on 100% type hint coverage in a popular library impact its design and popularity?
??x
Insisting on 100% type hint coverage could result in more complex and verbose API designs, which may detract from the ease of use and flexibility that users appreciate. Libraries like `requests`, known for their simplicity and power, might see their APIs become unnecessarily complicated if every function and argument was meticulously typed. This could potentially reduce user adoption as developers prefer simpler interfaces.

For example, consider a hypothetical type hint for `files` in `requests.request()`:
```python
Optional[
    Union[
        Mapping[
            basestring,
            Union[
                Tuple[basestring, Optional[Union[basestring, file]]],
                Tuple[basestring, Optional[Union[basestring, file]], Optional[basestring]],
                Tuple[basestring, Optional[Union[basestring, file]], Optional[basestring], Optional[Headers]]
            ]
        ],
        Iterable[
            Tuple[
                basestring,
                Union[
                    Tuple[basestring, Optional[Union[basestring, file]]],
                    Tuple[basestring, Optional[Union[basestring, file]], Optional[basestring]],
                    Tuple[basestring, Optional[Union[basestring, file]], Optional[basestring], Optional[Headers]]
                ]
            ]
        ]
    ]
]
```
This level of detail can make the API cumbersome and less user-friendly. Given that `requests` was designed to be easy to use, flexible, and powerful, adding such type hints might alter its design philosophy.
x??

---

#### Historical Decision Not to Include Type Hints
Explanation of why maintainers of `requests` decided not to include extensive type hints despite their popularity.

:p Why did the maintainers of `requests` decide against including full type hint coverage?
??x
The maintainers, particularly Cory Benfield, believed that libraries with Pythonic APIs are least likely to benefit from typing systems due to minimal value provided. They opted to prioritize ease of use and flexibility over strict type hinting, as seen in the popularity of `requests`, which has a clean, user-friendly API.

For example, when considering adding extensive type hints for `files`:
```python
def request(url: str, files: Optional[
    Union[
        Mapping[
            basestring,
            Union[
                Tuple[basestring, Optional[Union[basestring, file]]],
                Tuple[basestring, Optional[Union[basestring, file]], Optional[basestring]],
                Tuple[basestring, Optional[Union[basestring, file]], Optional[basestring], Optional[Headers]]
            ]
        ],
        Iterable[
            Tuple[
                basestring,
                Union[
                    Tuple[basestring, Optional[Union[basestring, file]]],
                    Tuple[basestring, Optional[Union[basestring, file]], Optional[basestring]],
                    Tuple[basestring, Optional[Union[basestring, file]], Optional[basestring], Optional[Headers]]
                ]
            ]
        ]
    ]] = None) -> Response:
```
This would make the API much more complex and less intuitive.

The maintainers decided not to spend time on type hints because they felt it was a low-value addition compared to maintaining the simplicity of the library.
x??

---

#### Type Hints in Python: Understanding the Trade-offs
Background context explaining why type hints exist and their benefits. The primary goal is to enhance readability and maintainability of code, particularly in large projects where manual inspection can be error-prone.

:p What are the trade-offs involved with using type hints in Python?
??x
Using type hints in Python comes with both advantages and disadvantages. On one hand, it improves code clarity and enables better static analysis tools to catch potential issues before runtime. However, it also introduces a learning curve for understanding how the type system works, requiring a one-time investment of time and effort. Additionally, there is an ongoing maintenance cost due to the need to update types as the project evolves. The downside includes losing some Python's dynamic capabilities like argument unpacking or metaprogramming.

```python
# Example with argument unpacking without type hints
def config(**settings):
    # Code using settings

config(a=1, b=2)

# Attempting to type check this would require spelling out each argument
from typing import TypedDict
class Settings(TypedDict):
    a: int
    b: int

def config(settings: Settings):  # This is how you might annotate it
    # Code using settings

config(a=1, b=2)  # This will need to be adjusted for type checking
```
x??

---

#### Dynamic vs Static Typing in Python and Java
Background context explaining the differences between dynamic and static typing. In dynamic languages like Python, types are checked at runtime or not checked at all (dynamically), whereas in statically typed languages like Java, types are checked at compile-time.

:p What is a key difference between how type checking works in Python and Java?
??x
In Python, the language is dynamically typed, meaning that variables can hold values of any type without needing to be declared with a specific type. This flexibility comes from runtime type checking where the type of an object is determined at execution time.

Java, on the other hand, is statically typed, which means that every variable must have a defined type before it's used in code, and this type remains fixed for its lifetime. Java enforces these types strictly at compile-time to ensure that operations like method calls or variable assignments are valid.

```java
// Python example with dynamic typing
def add(a, b):
    return a + b  # a and b can be of any type

add(1, '2')  # This is allowed in Python due to dynamic typing

// Java example with static typing
public class Add {
    public int add(int a, int b) {
        return a + b;  // Both parameters must be integers
    }
}
```
x??

---

#### Metaprogramming in Python
Background context explaining what metaprogramming is and why it's valuable. Metaprogramming refers to writing programs that write or manipulate other programs (or themselves).

:p Why are libraries using metaprogramming hard to annotate with type hints?
??x
Libraries using metaprogramming can be challenging to annotate because the dynamic nature of Python allows for code generation and runtime manipulation, which can obscure type information. Type checkers typically rely on explicit annotations that describe static structures, making it difficult to infer types accurately from dynamically generated code.

```python
# Example of metaprogramming in Python using a decorator
def log(f):
    def wrapper(*args, **kwargs):
        print(f"Calling {f.__name__} with args: {args}, kwargs: {kwargs}")
        return f(*args, **kwargs)
    return wrapper

@log
def greet(name):
    print(f'Hello, {name}')

greet('World')
```
x??

---

#### Optional Typing and PEP 544
Background context explaining the evolution of typing in Python through PEP 544. The `typing` module introduced optional type hints to provide more flexibility.

:p What does PEP 544 add to Python's typing system?
??x
PEP 544 introduces a way to define protocol classes, which can be used as a base for other classes to ensure that they implement certain methods or attributes. This feature provides a more expressive and flexible type hinting mechanism compared to traditional static types.

```python
from typing import Protocol

class SupportsAdd(Protocol):
    def add(self, x: int) -> int:
        ...

def process(x: SupportsAdd) -> int:
    return x.add(1)

# Example of using the protocol
class MyClass:
    def add(self, x: int) -> int:
        return x + 5

process(MyClass())
```
x??

---

#### Generics in Python vs Java
Background context explaining how generics work differently in Python and Java. In Python, `list` is a generic type that accepts any object, while in Java, list types were specific to Object until generics were introduced.

:p How do the concepts of "generic" and "specific" differ between Python and Java?
??x
In Python, `list` is considered generic because it can hold elements of any type. In contrast, before Java 1.5, all collections could only store objects of the `Object` class, making them specific in a way that they allowed no other types.

Java introduced generics with version 1.5 to provide more flexibility by allowing collection types to be parameterized with specific types at compile time:

```java
// Java before generics
List list = new ArrayList();
list.add("Hello");
list.add(42); // This will work but may cause runtime errors

// Java with generics
List<String> stringList = new ArrayList<>();
stringList.add("Hello");  // Compile-time error if you try to add a non-string type
```

In Python, this is handled differently:

```python
# Python list example (generic)
my_list = [1, "two", {"three": 3}]
for item in my_list:
    print(type(item))
```
x??

---

#### Type Hints and Constraints
Background context: In Python, type hints are used to indicate the expected types of function arguments or return values. However, defining a specific range for an integer (e.g., between 1 and 1000) or a specific format for strings (e.g., 3-letter airport codes) is not directly supported in standard typing annotations.

:p How can you define a type hint for a quantity that must be an integer between 1 and 1000?
??x
To define such a constraint, you would typically use additional checks within the function or method. Python's `typing` module provides ways to create custom types or protocols, but these do not directly support range constraints.

```python
from typing import Protocol

class ValidQuantity(Protocol):
    def __call__(self, value: int) -> bool:
        ...

def check_quantity(value: int) -> None:
    if not (1 <= value <= 1000):  # Check the constraint
        raise ValueError("Value must be between 1 and 1000")
```
x??

---
#### Duck Typing in Python
Background context: Duck typing is a principle in programming where you check for the presence of certain methods or properties, rather than relying on inheritance. Python supports this through its dynamic nature.

:p How does duck typing work in Python?
??x
Duck typing works by checking if an object has the required attributes or methods, without requiring explicit type declarations. For example:

```python
def greet(obj):
    # Duck typing: check for a greeting method
    if hasattr(obj, 'greet'):
        obj.greet()
    else:
        raise TypeError("Object does not have a greet method")

class Person:
    def greet(self):
        print("Hello!")

p = Person()
greet(p)  # This works because Person has a greet method.
```
x??

---
#### Protocols in Python
Background context: Starting with Python 3.8, the `typing.Protocol` can be used to define interfaces or contracts that classes should follow, enhancing type checking.

:p How does `typing.Protocol` work?
??x
`typing.Protocol` is a way to create protocols (abstract base classes) which define methods and attributes that classes should implement, without requiring inheritance. For example:

```python
from typing import Protocol

class Animal(Protocol):
    def speak(self) -> str:
        ...

class Dog:
    def speak(self) -> str:
        return "Woof!"

class Cat:
    def meow(self) -> str:
        return "Meow!"

def animal_sound(animal: Animal) -> None:
    print(animal.speak())

animal_sound(Dog())  # This works because Dog implements the speak method.
```
x??

---
#### Inheritance and Subtyping
Background context: Python uses subtyping (or duck typing) to determine whether an object can be used in a particular context. However, inheritance is often overused or not justified in simple examples.

:p Why might inheritance be overused in examples?
??x
Inheritance can lead to complex hierarchies and tight coupling between classes, making the code harder to maintain and extend. In simpler examples, Python's dynamic nature allows for more flexible solutions using duck typing rather than strict class-based inheritance.

For example:

```python
class Animal:
    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    pass  # This is overusing inheritance if we don't add any specific implementation.
```
x??

---
#### Type System Limitations in Python
Background context: The Python type system, while powerful for many use cases, has limitations. For instance, it cannot express constraints like the length of a string or the range of an integer.

:p Why can't Python's type hints define constraints like the length of a string?
??x
Python's type hints are primarily designed to provide static type checking and do not support complex constraints like string length directly. You would typically use runtime checks instead:

```python
def process_str(s: str) -> None:
    if len(s) == 3:  # Check the constraint at runtime
        print("Valid airport code")
    else:
        raise ValueError("Airport code must be exactly 3 characters long")

process_str("ABC")  # This works as expected.
```
x??

---
#### Virtual Subclasses in Python
Background context: In Python, some classes can be considered virtual subclasses even if they do not explicitly inherit from them. The `abc` module provides mechanisms for defining abstract base classes and checking for these virtual subclasses.

:p What is a virtual subclass?
??x
A virtual subclass is a class that does not explicitly inherit from another class but still satisfies the interface defined by an abstract base class (ABC). For example, Python's built-in `dict` class is considered a virtual subclass of `abc.MutableMapping`.

```python
from abc import ABC

class MutableMapping(ABC):
    pass

# dict is a virtual subclass:
issubclass(dict, MutableMapping)  # True
```
x??

---
#### Custom Type Definitions and Deprecations
Background context: The `typing` module in Python has been continually updated to address new features and deprecate older ones. This evolution reflects the growing needs of type checking in larger projects.

:p What did you contribute to the `typing` module documentation?
??x
I contributed dozens of deprecation warnings to the `typing` module documentation, helping users transition to newer types and practices while documenting the removal of older features.

```python
# Example of a deprecated type hint:
from typing import Dict  # New in Python 3.9: use `dict`
```
x??

---

#### Introduction to Decorators and Closures

Background context explaining decorators: In Python, a decorator is a function that takes another function and extends its behavior without explicitly modifying it. This concept allows for a more modular and cleaner code structure.

Relevant formulas or data: Not applicable as this is a conceptual explanation.

:p What are decorators in Python?
??x
Decorators in Python are functions that take another function as an argument, modify its behavior, and return the modified function. They allow us to wrap a function with another to add functionality without changing the original function's source code.
x??

---

#### Syntax of Decorators

Background context explaining decorator syntax: The `@` symbol is used to apply a decorator to a function definition. This syntax allows for clean and concise way to enhance functions.

:p How do you use decorators in Python?
??x
You use decorators by placing the `@decorator_name` above the function definition, followed by the function body.
```python
def deco(func):
    # decorator logic here

@deco
def target():
    print('running target()')
```
x??

---

#### Decorators vs. Functions

Background context explaining how to call decorators: A decorator is a callable that takes another function as an argument and returns it or replaces it with another function.

:p What is the difference between calling a decorator directly and using it as a syntax?
??x
Calling a decorator directly involves simply invoking the decorator function and passing the target function, whereas using the `@` syntax combines these steps in a single line.
```python
# Direct call
def deco(func):
    # decorator logic here

decorated_function = deco(target)

# Syntax with @
@deco
def target():
    print('running target()')
```
x??

---

#### Closures and Decorators

Background context explaining closures: A closure is a function that remembers the environment in which it was created. It captures variables from its enclosing scope, even if those scopes are no longer active.

:p What is a closure?
??x
A closure is a function object that has access to variables from its own scope, as well as the outer (enclosing) scopes, even when the function is called outside of those scopes.
x??

---

#### Nonlocal Keyword

Background context explaining `nonlocal`: The `nonlocal` keyword in Python allows for modifying variables in non-local enclosing scopes. It's used to indicate that a variable inside a nested function needs to refer to a non-global variable from an enclosing scope.

:p What is the purpose of the `nonlocal` keyword?
??x
The `nonlocal` keyword is used when you want to modify a variable that exists in an outer (but not global) scope within a nested function. This allows for creating closures where inner functions can access and modify variables from their enclosing scopes.
x??

---

#### Decorator Examples

Background context explaining decorator examples: We will explore simple and complex decorators, including parameterized ones.

:p How do you create a simple decorator?
??x
Creating a simple decorator involves defining a function that takes another function as an argument. This function (the decorator) can modify the behavior of the original function.
```python
def deco(func):
    def inner():
        print("Running inner() before target()")
        func()
        print("Running inner() after target()")
    return inner

@deco
def target():
    print('running target()')
```
x??

---

#### Caching Decorators

Background context explaining caching decorators: The `functools.cache` decorator is a simpler alternative to the traditional `lru_cache`. It automatically caches results of the decorated function.

:p What is `functools.cache`?
??x
`functools.cache` is a Python decorator that simplifies caching. It caches the return values of the functions it decorates, making repeated calls with the same arguments much faster.
```python
from functools import cache

@cache
def expensive_function(x):
    # Computationally intensive work here
    pass
```
x??

---

#### Parameterized Decorators

Background context explaining parameterized decorators: A decorator can be made more flexible by allowing it to accept parameters.

:p How do you implement a parameterized decorator?
??x
A parameterized decorator is implemented by adding arguments to the outer function (decorator) and storing those arguments in nonlocal variables. These variables are then used inside the inner functions.
```python
def param_deco(arg1, arg2):
    def deco(func):
        def inner():
            print(f"arg1: {arg1}, arg2: {arg2}")
            func()
        return inner
    return deco

@param_deco('value1', 'value2')
def target():
    print('running target()')
```
x??

---

#### Standard Library Decorators

Background context explaining standard library decorators: Python’s standard library includes several useful decorators like `@cache` and `@lru_cache`.

:p What are some built-in decorators in the standard library?
??x
The standard library provides several useful decorators, including:
- `functools.cache`: Caches return values.
- `functools.lru_cache`: Caches return values with a limit on the size of cache to prevent memory leaks.

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def expensive_function(x):
    # Computationally intensive work here
    pass
```
x??

---

#### Nonlocal Keyword in Decorators

Background context explaining `nonlocal` keyword usage: The `nonlocal` keyword is used when the decorator needs to modify variables from an enclosing scope.

:p Why do we use the `nonlocal` keyword in decorators?
??x
The `nonlocal` keyword is used in decorators to allow modification of non-global variables that are defined in an outer (enclosing) scope. This enables creating closures where inner functions can access and modify these variables.
```python
def outer():
    x = 'initial'
    
    def inner():
        nonlocal x  # Allows modifying x from the enclosing scope
        x = 'modified'
        print(x)
    
    return inner

outer()()
```
x??

---

#### Decorators Execution Order and Function Registration
Background context: In Python, decorators are a powerful mechanism that allows for modifying or extending the behavior of functions. A decorator is a function that takes another function as an argument (the decorated function) and extends its functionality without explicitly modifying it.

Example in the provided text shows how `@register` is used to decorate functions `f1()` and `f2()`. The decorated functions are registered into a list named `registry`. When the module containing these definitions is imported, the decorator runs immediately at import time and registers the decorated functions. However, if the module is executed directly (i.e., as the main script), then the decorated functions' behavior is triggered.

:p What happens when a Python module with decorators defined is imported?
??x
When a Python module containing decorators is imported, the decorators are applied to the decorated functions right away. This means that any code within the decorator's body (such as function registration) runs at import time rather than during script execution. In contrast, if the module is executed directly (e.g., `python3 registration.py`), then the code within the main block runs.

Example:
```python
import registration  # Imports the module containing decorators

# At this point, registry will hold references to f1 and f2 from decorator execution.
print(registration.registry)
```

This demonstrates that import time is when decorators run and modify the functions, but not necessarily when the decorated function itself is called.

x??

---
#### Function Decoration Process
Background context: The `@register` decorator in the provided code snippet performs two key operations:
1. It registers the decorated function into a list.
2. It prints information about which function is being registered.

The process of decoration involves defining the `register` function, which takes another function as an argument and modifies its behavior by adding it to a registry.

:p How does the `@register` decorator work in Python?
??x
The `@register` decorator works by:
1. Defining a `register` function that accepts a function (`func`) as an argument.
2. Inside the `register` function, it prints information about which function is being registered and appends the function to the `registry` list.
3. It returns the same function passed in (i.e., `return func`).

The logic can be seen below:
```python
def register(func):
    print(f'running register({func})')
    registry.append(func)
    return func
```

When a function is decorated with `@register`, Python will invoke this decorator at the time of decoration, not necessarily when the script is run. For example, in the provided code:
```python
@register
def f1():
    print('running f1()')
```
The above line registers `f1` to the `registry`. If you import `registration`, it will automatically register both `f1` and `f2`.

x??

---
#### Execution Flow of Decorators in Python Scripts vs Imports
Background context: In Python, decorators like `@register` run at import time. This means that when a module is imported (e.g., via `import registration`), the decorator functions are executed immediately, modifying the behavior of the decorated functions.

In contrast, if you execute the script directly (`python3 registration.py`), only the code in the main block runs after importing all modules and executing any decorators. This results in different behaviors depending on how the module is used (directly or imported).

:p What is the difference between running a Python script with decorated functions versus just importing them?
??x
When you run a Python script directly, the execution flow includes the following steps for decorators:
1. The `@register` decorator runs when the module is loaded.
2. Functions like `f1()` and `f2()` are registered into the `registry`.
3. After all decorators have executed, the code within the main function (`main()`) runs.

However, if you import the same module without running it directly:
```python
import registration
```
The decorator still runs at import time, but the functions inside `main()` do not execute unless explicitly called later in your script. 

Example of how decorators behave differently with direct execution vs import:
```python
if __name__ == '__main__':
    main()
```

In this case, only when you run `python3 registration.py` will `f1`, `f2`, and `f3` be executed.

x??

---

