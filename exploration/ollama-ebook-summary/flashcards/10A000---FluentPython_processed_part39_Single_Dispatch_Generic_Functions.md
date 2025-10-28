# Flashcards: 10A000---FluentPython_processed (Part 39)

**Starting Chapter:** Single Dispatch Generic Functions

---

#### functools.singledispatch Overview
`functools.singledispatch` is a decorator that allows creating single-dispatch generic functions. It provides a mechanism to define different implementations for the same function based on the type of the first argument.

In contrast to multiple-dispatch, where more arguments can determine which method gets called, singledispatch focuses solely on the type of the first argument. This is particularly useful when you want to extend functionality in a non-intrusive way, especially for third-party libraries or complex data structures.

:p How does `functools.singledispatch` work?
??x
`functools.singledispatch` works by decorating a base function with the same name. Each specialized implementation is then registered using the `.register()` method. At runtime, Python selects which specific implementation to use based on the type of the first argument.

Here’s how you can register different implementations:
```python
from functools import singledispatch

@singledispatch
def htmlize(obj: object) -> str:
    content = html.escape(repr(obj))
    return f'<pre>{content}</pre>'

@htmlize.register
def _(text: str) -> str:
    content = html.escape(text).replace(' ', '<br/>')
    return f'<p>{content}</p>'

# More registrations...
```

x??

---

#### Registration with @singledispatch
When using `functools.singledispatch`, you define the base function and then use `.register()` to add specialized implementations for different types.

:p How do you register a new implementation for a specific type?
??x
You can register a new implementation by decorating it with `@htmlize.register` (or `_`) where `htmlize` is the name of your dispatched function. The type hint in the parameter defines which type will trigger this implementation.

For example:
```python
from functools import singledispatch

@singledispatch
def htmlize(obj: object) -> str:
    content = html.escape(repr(obj))
    return f'<pre>{content}</pre>'

@htmlize.register
def _(text: str) -> str:
    content = html.escape(text).replace(' ', '<br/>')
    return f'<p>{content}</p>'
```

x??

---

#### Handling Concrete Types vs. ABCs
When registering types, you can choose to handle concrete types or abstract base classes (ABCs).

:p What is the difference between handling concrete types and ABCs?
??x
Handling concrete types means you directly register specific implementations for each type. However, using ABCs allows your function to support any type that implements that interface, making it more flexible.

For example:
```python
from functools import singledispatch
import numbers

@singledispatch
def htmlize(obj: object) -> str:
    content = html.escape(repr(obj))
    return f'<pre>{content}</pre>'

@htmlize.register
def _(text: str) -> str:
    content = html.escape(text).replace(' ', '<br/>')
    return f'<p>{content}</p>'

@htmlize.register(numbers.Integral)
def _(n: numbers.Integral) -> str:
    return f'<pre>{n} (0x{n:x})</pre>'
```

Using `numbers.Integral` here allows the function to handle not only `int` but also other types that inherit from it, like `bool`.

x??

---

#### Using Type Hints vs. Without
You can either use type hints in your registration or pass the type directly.

:p How do you register a function without using type hints?
??x
If you don’t want to (or cannot) add type hints, you can pass the type directly as an argument to `.register()`.

For example:
```python
from functools import singledispatch

@singledispatch
def htmlize(obj: object) -> str:
    content = html.escape(repr(obj))
    return f'<pre>{content}</pre>'

@htmlize.register(bool)
def _(x):
    return f'<pre>{x}</pre>'
```

This approach is compatible with Python versions 3.4 and later.

x??

---

#### Registering Multiple Types on the Same Implementation
You can register multiple types to handle different scenarios within a single implementation.

:p How do you register two or more types on the same function?
??x
You can stack `@register` decorators to handle multiple types in one function. For example:
```python
from functools import singledispatch

@singledispatch
def htmlize(obj: object) -> str:
    content = html.escape(repr(obj))
    return f'<pre>{content}</pre>'

@htmlize.register(float)
@htmlize.register(decimal.Decimal)
def _(x):
    frac = fractions.Fraction(x).limit_denominator()
    return f'<pre>{x} ({frac.numerator}/{frac.denominator})</pre>'
```

This allows the function to handle both `float` and `decimal.Decimal`.

x??

---

#### Example of Complex Registration
Here’s an example combining multiple concepts:

:p Provide a complete example of using `functools.singledispatch` for HTMLizing different types.
??x
```python
from functools import singledispatch
import html
import abc
import fractions
import decimal
import numbers

@singledispatch
def htmlize(obj: object) -> str:
    content = html.escape(repr(obj))
    return f'<pre>{content}</pre>'

@htmlize.register(str)
def _(text: str) -> str:
    content = html.escape(text).replace(' ', '<br/>')
    return f'<p>{content}</p>'

@htmlize.register(abc.Sequence)
def _(seq: abc.Sequence) -> str:
    inner = '</li><li>'.join(htmlize(item) for item in seq)
    return '<ul><li>' + inner + '</li></ul>'

@htmlize.register(numbers.Integral)
def _(n: numbers.Integral) -> str:
    return f'<pre>{n} (0x{n:x})</pre>'

@htmlize.register(bool)
def _(n: bool) -> str:
    return f'<pre>{n}</pre>'

@htmlize.register(fractions.Fraction)
def _(x):
    frac = fractions.Fraction(x)
    return f'<pre>{frac.numerator}/{frac.denominator}</pre>'

@htmlize.register(decimal.Decimal)
@htmlize.register(float)
def _(x) -> str:
    frac = fractions.Fraction(x).limit_denominator()
    return f'<pre>{x} ({frac.numerator}/{frac.denominator})</pre>'
```

This example covers a wide range of types, ensuring that each type gets its own specialized HTML representation.

x??

---

#### Single Dispatch Mechanism Overview
Background context: The singledispatch mechanism allows for registering specialized functions that handle different types. This is useful when you want to provide type-specific behavior without cluttering a single function or class with numerous if/elif blocks.

:p What are the key benefits of using singledispatch over traditional if/elif blocks?
??x
The key benefits include:
- Easier maintenance and readability: Each specialized function can be defined in its own module, making it easier to understand which functions handle specific types.
- Flexibility: New modules with new user-defined types can easily provide custom handling without modifying existing code.
- Code separation: Functions are modular and can be written by different developers or teams.

Example:
```python
from functools import singledispatch

@singledispatch
def process(value):
    raise NotImplementedError(f"Unsupported type {type(value)}")

@process.register
def _(value: int):
    print(f"Processing integer value: {value}")

@process.register
def _(value: str):
    print(f"Processing string value: {value}")
```
x??

---

#### Example of Using singledispatch with Custom Types
Background context: The example demonstrates how to use the `singledispatch` mechanism to handle different types, such as integers and strings.

:p How can you register specialized functions for custom types using singledispatch?
??x
You can register specialized functions using the `.register()` method or the decorator syntax. Here’s an example:

```python
from functools import singledispatch

@singledispatch
def custom_process(value):
    raise NotImplementedError(f"Unsupported type {type(value)}")

@custom_process.register
def _(value: int):
    print(f"Custom processing integer value: {value}")

@custom_process.register
def _(value: str):
    print(f"Custom processing string value: {value}")
```
x??

---

#### Register Decorator with Parameters
Background context: The `register` decorator can be modified to accept parameters, such as an optional `active` flag. This allows controlling whether a function is registered or not.

:p How does the register decorator factory work?
??x
The `register` decorator factory returns another decorator based on the provided parameters. Here’s how it works:

```python
registry = set()

def register(active=True):
    def decorate(func):
        nonlocal registry
        
        if active:
            registry.add(func)
        else:
            registry.discard(func)

        return func
    
    return decorate

@register(active=False)
def f1():
    print('Running f1()')

@register()
def f2():
    print('Running f2()')
    
def f3():
    print('Running f3()')
```
The `registry` is a set, which allows for efficient addition and removal of functions. The `active` parameter controls whether the function is added to or removed from the registry.

x??

---

#### Using the Parameterized Register Decorator
Background context: This example demonstrates how to use the modified `register` decorator factory with different parameters.

:p What are the differences between using a regular `@register()` and calling it as a function?
??x
When used with `@`, the decorator is applied directly. When called as a function, it returns an inner decorator that can be applied to functions:

```python
registry = set()

def register(active=True):
    def decorate(func):
        nonlocal registry
        
        if active:
            registry.add(func)
        else:
            registry.discard(func)

        return func
    
    return decorate

from registration_param import *

# Using @ syntax
@register(active=False)
def f1():
    print('Running f1()')

# Using function call syntax
f2_decorated = register()(lambda: 'running f2()')
```
The `@register()` syntax is more concise and commonly used. The function call syntax can be useful for conditional registration.

x??

---

#### Applying the Parameterized Register Decorator to Functions
Background context: This example shows how to add or remove functions from a registry using the parameterized register decorator.

:p How does the `registration_param` module demonstrate adding and removing functions?
??x
The `registration_param` module demonstrates adding and removing functions based on the active flag:

```python
from registration_param import *

# Using @ with different parameters
@register(active=False)
def f1():
    print('Running f1()')

@register()
def f2():
    print('Running f2()')

# Adding a function manually
f3 = register()(lambda: 'running f3()')
print(f'Registry -> {registry}')
```
The output shows that only `f2` is registered because `active=False` for `f1`, and `f3` is added through the function call syntax.

x??

---

#### Parameterized Decorators Overview
Parameterized decorators allow for more flexible and dynamic functionality by accepting parameters. They are typically used to add features that can be customized when applied to a function.

:p What is the purpose of parameterized decorators?
??x
The purpose of parameterized decorators is to provide flexibility in adding functionality to functions, allowing users to customize how this additional functionality behaves or looks without altering the original function's code directly. This is achieved by accepting parameters during their definition and application.
x??

---

#### Default Format String in `clock` Decorator
In Example 9-24, a default format string `DEFAULT_FMT` is defined which controls the output of the clocked function report.

:p What is the default format string used in the `clock` decorator?
??x
The default format string used in the `clock` decorator is:
```
'[{elapsed:0.8f}s] {name}({args}) -> {result}'
```

This string formats the output to include details such as elapsed time, function name, arguments passed, and the result of the function.
x??

---

#### Decorator Factory `clock`
The `clock` function in Example 9-24 acts as a decorator factory that returns another decorator named `decorate`.

:p How does the `clock` function work?
??x
The `clock` function works by accepting an optional format string argument. It then returns a decorator called `decorate`. This `decorate` takes a function as input and returns another function, `clocked`, which performs timing and formatting.

Here's how it works in detail:
```python
import time

DEFAULT_FMT = '[{elapsed:0.8f}s] {name}({args}) -> {result} '

def clock(fmt=DEFAULT_FMT):
    def decorate(func):
        def clocked(*_args):
            t0 = time.perf_counter()
            _result = func(*_args)
            elapsed = time.perf_counter() - t0
            name = func.__name__
            args = ', '.join(repr(arg) for arg in _args)
            result = repr(_result)
            print(fmt.format(**locals()))
            return _result
        return clocked
    return decorate

if __name__ == '__main__':
    @clock()
    def snooze(seconds):
        time.sleep(seconds)

    for i in range(3):
        snooze(.123)
```

The `decorate` function wraps the original function and measures its execution time, then prints it using the provided format string.
x??

---

#### Applying `clock` Without Arguments
When `clock()` is called without arguments as shown in Example 9-24, it applies a default format to the decorated function.

:p What happens when `clock()` is called without arguments?
??x
When `clock()` is called without any arguments, it returns the `decorate` function. This returned decorator is then applied to the functions that use it as a decorator. By default, it uses the `DEFAULT_FMT` string defined in the `clock` function.

Here's how it works:
```python
if __name__ == '__main__':
    @clock()
    def snooze(seconds):
        time.sleep(seconds)

    for i in range(3):
        snooze(.123)
```

In this example, `snooze` is decorated with the default format string.
x??

---

#### Customizing Format String
Example 9-25 demonstrates how to customize the output by passing a different format string to the `clock` decorator.

:p How can you customize the output in the `clock` decorator?
??x
You can customize the output of the `clock` decorator by passing a custom format string when applying the `clock` decorator. This allows you to control how the function name, arguments, and result are displayed.

Here's an example from Example 9-25:
```python
from clockdeco_param import clock

@clock('{name}: {elapsed} s')
def snooze(seconds):
    time.sleep(seconds)

for i in range(3):
    snooze(.123)
```

In this case, the format string `'{name}: {elapsed} s'` is used to display only the function name and elapsed time.
x??

---

#### Understanding `clocked` Function
The `clocked` function in the `decorate` method of the `clock` decorator wraps the original function and measures its execution time.

:p What does the `clocked` function do?
??x
The `clocked` function performs several actions:
1. It records the start time using `time.perf_counter()`.
2. It calls the decorated function with the provided arguments.
3. It calculates the elapsed time after the function completes.
4. It constructs and prints a formatted string using the provided format or default format if none is given.
5. It returns the result of the decorated function.

Here's the relevant code snippet:
```python
def clocked(*_args):
    t0 = time.perf_counter()
    _result = func(*_args)
    elapsed = time.perf_counter() - t0
    name = func.__name__
    args = ', '.join(repr(arg) for arg in _args)
    result = repr(_result)
    print(fmt.format(**locals()))
    return _result
```

The `clocked` function is designed to measure the execution time of the decorated function and print it formatted according to the specified format string.
x??

---

