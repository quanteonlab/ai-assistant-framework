# Flashcards: 10A000---FluentPython_processed (Part 35)

**Starting Chapter:** Callable

---

#### Sorted and Duck Typing
Background context: The `sorted` function in Python can work on any iterable as long as its elements support a `<` comparison. This means that if an object supports the less-than operation, it can be sorted even without explicitly implementing an ordering mechanism.

:p What does the `sorted` function require from its input to function correctly?
??x
The `sorted` function only requires that the elements of the iterable support the `<` operator for comparison. This allows objects that implement their own comparison logic (duck typing) to be sorted.
x??

---

#### Typing.Protocol and Static Duck Typing
Background context: The `typing.Protocol` from Python's `typing` module enables static duck typing, which means you can declare a protocol with specific methods required by the interface. This allows for more robust type checking without needing to subclass or implement interfaces explicitly.

:p How does `typing.Protocol` enable static duck typing?
??x
`typing.Protocol` enables static duck typing by allowing you to define an abstract base class that only declares method signatures and no concrete implementations. An object can be checked against this protocol at runtime if it has the required methods, thus providing a way to enforce interface compliance in a statically typed manner.

:p How would you use `typing.Protocol` to create a `Double` function?
??x
You can define a `Double` function that works with any type of objects as long as they support multiplication. Here’s an example:

```python
from typing import Protocol, TypeVar

class SupportsMul(Protocol):
    def __mul__(self, other: int) -> 'SupportsMul':
        ...

def double(obj: SupportsMul) -> SupportsMul:
    return obj * 2

```
In this case, `double` function will work as long as the object passed to it supports multiplication with an integer.

x??

---

#### Callable Type
Background context: The `Callable` type from Python's `typing` module is used for annotating functions that are expected to be called or passed around. It can specify a function signature, including parameters and return types.

:p How do you annotate a function that accepts a callback with the `Callable` type?
??x
You can use the `Callable` type hint to indicate that a parameter should be callable. Here is an example of annotating the `repl` function:

```python
from typing import Callable

def repl(input_fn: Callable[[Any], str] = input) -> None:
    # Function implementation here
```

In this case, `input_fn` is expected to accept any type and return a string.

:p How do you use `Callable` in the context of an automated testing function?
??x
You can define a function that accepts different implementations of a callable interface. For example:

```python
from typing import Callable

def test_repl(input_fn: Callable[[Any], str] = input) -> None:
    # Function implementation here
```

Here, `test_repl` uses the built-in `input()` function by default but can accept any other function that takes an `Any` parameter and returns a string.

x??

---

#### Overloaded Signatures with @typing.overload
Background context: The `@typing.overload` decorator allows you to declare multiple signatures for the same function. This is useful when a function has different types of parameters or return values under certain conditions.

:p How do you use `@typing.overload` to declare overloaded signatures?
??x
You can use the `@typing.overload` decorator to define multiple signatures for a function, which helps in making type hints more accurate and precise. Here is an example:

```python
from typing import overload

@overload
def process_data(data: int) -> str:
    ...

@overload
def process_data(data: str) -> list[str]:
    ...

def process_data(data):
    if isinstance(data, int):
        return f"Integer data: {data}"
    elif isinstance(data, str):
        return data.split()
```

In this example, `process_data` can accept either an integer or a string and return the appropriate type.

x??

---

#### Order Class Example
Background context: The `Order` class in the provided text uses optional parameters with specific types to allow flexibility. This is useful for creating flexible classes that can handle different input scenarios.

:p How does the `promotion` parameter in the `Order.__init__` method work?
??x
The `promotion` parameter in the `Order.__init__` method allows the class to accept a function that takes an `Order` object and returns a float. This parameter is optional, meaning it can be set to `None`, or it can be any callable of type `Callable[['Order'], float]`. 

Here’s how you might define it in code:

```python
from typing import Optional, Callable

class Order:
    def __init__(
        self,
        customer: Customer,
        cart: Sequence[LineItem],
        promotion: Optional[Callable[['Order'], float]] = None,
    ) -> None:
        self.customer = customer
        self.cart = cart
        self.promotion = promotion
```

In this example, `promotion` can be set to a function that calculates a discount for the order. If no promotion is provided, it defaults to `None`.

x??

#### Forward References and Annotations in Python

Background context: PEP 563 introduced support for forward references in annotations, enabling more flexible type hints. This was implemented in Python 3.7 but has seen delays in becoming default behavior.

:p What is the issue with using `Order` class in type hints before it's defined?
??x
The issue arises because Python reads definitions from top to bottom. If you try to use a class like `Order` in an annotation before defining it, the interpreter will not recognize it, leading to errors. PEP 563 addresses this by allowing forward references.

For example:
```python
# Incorrect usage: results in error due to forward reference issue
def process_order(order_id: Order): ...

class Order:
    pass
```

PEP 563 allows the above code to work correctly:
```python
from __future__ import annotations

# Correct usage with PEP 563
def process_order(order_id: 'Order'): ...

class Order:
    pass
```
x??

---

#### Callable[..., ReturnType] for Dynamic Signatures

Background context: When you need a type hint to match functions with dynamic signatures, you can use `Callable[..., ReturnType]` where the ellipsis (`...`) indicates that any number of arguments of any types are accepted.

:p How do you annotate a function with a dynamic signature using `Callable`?
??x
You would use `Callable[..., ReturnType]`. This notation allows for functions that accept any number and type of positional or keyword arguments. For instance, if you have a callback that can take variable arguments and return a specific type:

```python
from typing import Callable

def process_callback(callback: Callable[..., int]):
    # The callback could be called with any number and types of arguments
    result = callback(1, 2, "hello")
```

Here `Callable[..., int]` means the function can take any combination of arguments and return an integer.

x??

---

#### NoReturn Type Hint

Background context: `NoReturn` is a special type hint used to indicate that a function never returns. It’s commonly used in functions designed to raise exceptions or terminate execution, such as system exit handlers.

:p What does the `NoReturn` type hint signify?
??x
The `NoReturn` type hint signifies that a function will never return normally; it is typically used for functions that perform actions like raising an exception or terminating program execution. For example:

```python
from typing import NoReturn

def exit_program(status: object) -> NoReturn:
    # This function is designed to terminate the program
    import sys
    sys.exit(status)
```

Here, `exit_program` raises a `SystemExit` and thus never returns.

x??

---

#### Positional-Only Parameters and Variadic Parameters

Background context: Python 3.8 introduced the `/` syntax for declaring positional-only parameters. This means these parameters can only be passed positionally and not as keyword arguments. Variadic parameters use `*args` for positional and `**kwargs` for keyword.

:p How do you annotate a function with positional-only and variadic parameters?
??x
For annotating functions, use the `/` notation after the first required argument to indicate that all subsequent parameters are positional only:

```python
from typing import Optional

def tag(
    name: str,
    /,
    *content: str,
    class_: Optional[str] = None,
    **attrs: str
) -> str:
    # Function body
```

Here, `name` is a positional-only parameter, while `*content`, `class_`, and `**attrs` allow flexible argument handling.

x??

---

#### Limits of Type Hints

Background context: While type hints are useful for catching bugs early, they have limitations. They cannot fully replace the need for testing or ensure all business logic is correct. Static type checkers also have false positives and negatives.

:p What are some limitations of using static type checking in Python?
??x
Static type checking can introduce several limitations:
- **False Positives**: Tools may report errors that do not actually exist.
- **False Negatives**: Type checkers might miss actual bugs.
- **Expressiveness Loss**: Advanced features like argument unpacking or complex data constraints cannot be fully checked statically.
- **Tool Lag**: Type checkers often lag behind Python releases, leading to crashes on new features.

In general, type hints should be used as one of many tools in a CI pipeline, alongside testing and linting.

x??

#### Gradual Typing Concept
Background context: The idea of gradual typing refers to a system that supports both dynamically typed and statically typed programming. Python is an example of such a language, where you can gradually introduce type hints while keeping other parts of your codebase dynamically typed.

:p What is gradual typing in the context of Python?
??x
Gradual typing in Python allows developers to start writing code without explicit types (dynamically typed) and gradually add static type annotations. This hybrid approach combines the flexibility of dynamic typing with the safety benefits of static typing.
x??

---

#### Python's Type Hints
Background context: Python’s type hints provide a way to specify expected types for variables, function parameters, and return values. These hints help in catching errors early through static analysis tools like Mypy.

:p What are type hints in Python?
??x
Type hints in Python allow developers to specify the data types of variables, function arguments, and return values using special syntax. This feature helps improve code readability and can be used with static analyzers to catch potential bugs.
x??

---

#### Mypy Tool
Background context: MyPy is a popular static type checker for Python that helps ensure your code adheres to the type hints you provide.

:p What is MyPy?
??x
Mypy is a static type checker tool for Python. It uses the type hints provided in your code to check for potential errors and ensures that the types used match what is expected. This can be done while running or by using an integrated development environment (IDE) plugin.
x??

---

#### Annotated Function Example
Background context: The text mentions developing an annotated function guided by MyPy error reports.

:p How do you develop an annotated function with type hints?
??x
To develop an annotated function with type hints, you start by defining the expected types for parameters and return values. For example:
```python
def greet(name: str) -> str:
    return f"Hello, {name}"
```
Here, `greet` is a function that takes a string argument `name` and returns a string.

You can use Mypy to check this function by running `mypy your_script.py`. If there are any type mismatches or issues, MyPy will provide error reports.
x??

---

#### Python 2.7 and 3.x Compatibility
Background context: The text discusses how to write code that runs under both Python 2.7 and 3.x while using type hints.

:p How do you handle compatibility between Python 2.7 and 3.x in type hints?
??x
To ensure your code is compatible with both Python 2.7 and 3.x, you can use the `from __future__ import annotations` statement at the top of your script. This allows you to write type hints that are evaluated later in the code rather than being evaluated during import time, which works differently between Python 2 and 3.
```python
from __future__ import annotations

def greet(name: str) -> str:
    return f"Hello, {name}"
```
x??

---

#### Generic Classes and Variance
Background context: The text mentions generic classes and variance in the typing module.

:p What are generic classes in Python?
??x
Generic classes in Python allow you to create reusable components that can work with different data types. For example:
```python
from typing import Sequence

def first_item(seq: Sequence[T]) -> T:
    return seq[0]
```
Here, `Sequence[T]` is a generic type where `T` can be any type (like int, str, etc.). The function `first_item` takes a sequence of any type and returns the first item.
x??

---

#### Protocols in Python
Background context: Protocols are introduced as a way to enable static duck typing.

:p What are protocols in Python?
??x
Protocols in Python, available since Python 3.8, provide a way to define abstract base classes (ABCs) without requiring the implementation of all methods. They enable static duck typing by allowing you to specify the structure and behavior of objects without enforcing inheritance.
```python
from typing import Protocol

class SupportAdd(Protocol):
    def __add__(self, other: "SupportAdd") -> "SupportAdd":
        ...
```
Here, `SupportAdd` is a protocol that defines an abstract method `__add__`. Any class implementing this protocol must define the `__add__` method.
x??

---

#### Type Variables
Background context: Type variables are used in generic classes and type hints to provide more flexibility.

:p What are type variables in Python's typing module?
??x
Type variables in Python’s typing module allow you to create generic types that can accept any valid type as a parameter. For example:
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class MyGenericClass(Generic[T]):
    def __init__(self, value: T):
        self.value = value

g1 = MyGenericClass(5)
g2 = MyGenericClass("hello")
```
Here, `T` is a type variable that can represent any type. The class `MyGenericClass` is a generic class that can hold values of any type.
x??

---

#### Overloaded Signatures
Background context: Overloaded signatures are another feature in Python’s typing module.

:p What are overloaded signatures?
??x
Overloaded signatures allow you to define multiple function signatures with the same name but different parameter types or return types. MyPy uses this information to provide more precise error messages and type checking.
```python
from typing import overload

@overload
def greet(name: str) -> str:
    ...

@overload
def greet(name: None) -> str:
    ...

def greet(name=None):
    if name is None:
        return "Hello, Guest"
    else:
        return f"Hello, {name}"
```
Here, `greet` has two overloaded signatures. One takes a string and returns a string, while the other handles `None` and returns a default greeting.
x??

---

#### Union Type
Background context: The text mentions using `Union` to represent multiple possible types.

:p What is the `Union` type in Python’s typing module?
??x
The `Union` type in Python’s typing module allows you to specify that a variable can be of one or more related types. For example:
```python
from typing import Union

def process_value(value: Union[int, str]) -> None:
    if isinstance(value, int):
        print(f"Processing integer {value}")
    else:
        print(f"Processing string {value}")

process_value(5)
process_value("hello")
```
Here, `Union[int, str]` indicates that the variable `value` can be either an `int` or a `str`.
x??

---

#### Optional Type
Background context: The text introduces `Optional` to represent optional parameters.

:p What is the `Optional` type in Python’s typing module?
??x
The `Optional` type in Python’s typing module indicates that a variable might have no value at all, represented as `None`. It can be used for function arguments or return values. For example:
```python
from typing import Optional

def get_value() -> Optional[str]:
    if some_condition():
        return "Hello"
    else:
        return None

print(get_value())
```
Here, the function `get_value` can either return a string or `None`.
x??

---

#### NoReturn Type
Background context: The text discusses the `NoReturn` type to indicate functions that never return.

:p What is the `NoReturn` type in Python’s typing module?
??x
The `NoReturn` type in Python’s typing module indicates that a function will not return normally, such as by raising an exception or entering an infinite loop. For example:
```python
from typing import NoReturn

def raise_error() -> NoReturn:
    raise ValueError("This is an error")

try:
    raise_error()
except ValueError as e:
    print(e)
```
Here, the function `raise_error` is annotated with `NoReturn`, indicating it will not return normally.
x??

---

