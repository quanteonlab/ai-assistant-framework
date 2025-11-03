# Flashcards: 10A000---FluentPython_processed (Part 33)

**Starting Chapter:** Optional and Union types

---

#### Type Annotations and Inference in Python

Background context: The provided text discusses various aspects of type annotations, including explicit type hints, inferred types from type checkers, and the `Any` wildcard type. It also covers simple types, classes, abstract base classes (ABCs), and union types.

:p What are the implications of using `Any` in function arguments or return values?
??x
The use of `Any` allows a function to accept any type of argument or return any type without explicit annotation. This is useful for situations where the exact type cannot be determined, but it can make the code less readable and type checking less effective.

```python
def f3(p: Any) -> None:
    ...
o0 = object()
o1 = T1()
o2 = T2()
f3(o0)  # All OK: rule #2
f3(o2)  # Also OK: rule #2
```
x??

---

#### Subtype Consistency in Python

Background context: The text explains that `int` is considered consistent with `float`, and `float` is consistent with `complex`. This means an object of type `int` can be used where a `float` or `complex` is expected, even though they are not nominal subtypes.

:p How does the practicality beats purity principle apply to subtype consistency?
??x
The principle "practicality beats purity" implies that while `int`, `float`, and `complex` are direct subclasses of `object`, the operations defined on them allow `int` to be treated as a consistent type with both `float` and `complex`. This practical approach ensures smooth operation without enforcing strict subtype relationships.

```python
3.real  # 3.0
3.imag  # 0.0
```
x??

---

#### Optional and Union Types

Background context: The text describes how `Optional[T]` is a shorthand for `Union[T, None]`, meaning that an optional parameter can accept either the type `T` or `None`. It also mentions that union types like `Union[str, bytes]` allow multiple types to be accepted.

:p What does the function `ord(c: Union[str, bytes]) -> int:` illustrate about union types?
??x
The function `ord(c: Union[str, bytes]) -> int:` illustrates that a single parameter can accept either a string or bytes object and return an integer. This example demonstrates how union types can be used to define flexible function signatures.

```python
from typing import Union

def ord(c: Union[str, bytes]) -> int:
    ...
```
x??

---

#### Practical Use of Union Types

Background context: The text provides examples of functions that may return multiple types using `Union`, such as `parse_token(token: str) -> Union[str, float]`. However, it advises against creating too many union type functions since they can complicate the usage of returned values.

:p Why should one avoid returning `Union` types in function signatures unless necessary?
??x
Avoiding return `Union` types in favor of specific types when possible simplifies the code and makes it easier for users to handle the results. If a function returns a union type, the user needs to check the actual returned type at runtime to determine how to use it, which can be error-prone.

```python
from typing import Union

def parse_token(token: str) -> Union[str, float]:
    try:
        return float(token)
    except ValueError:
        return token
```
x??

---

#### Example of Type Inference

Background context: The text discusses how modern type checkers infer types from function calls and variable assignments. It provides an example where `len(s) * 10` results in a type inference for `x`.

:p How does the type checker infer the return type of `x = len(s) * 10`?
??x
The type checker infers that `x` will be an integer (`int`) because `len(s)` returns an integer, and multiplying it by `10` also results in an integer.

```python
x = len(s) * 10  # Type inferred as int
```
x??

---

#### Abstract Base Classes (ABCs)

Background context: The text mentions that ABCs are useful for type hints. It explains how a subclass is consistent with its superclasses but notes the practicality of treating `int` as consistent with `complex`.

:p How does being consistent-with affect type hints for subclasses?
??x
Being consistent-with means that if a class `C` is a subclass of another class `B`, it can be used wherever an instance of `B` would be expected. This principle allows more flexible type hinting and easier function signatures.

```python
class T1:
    pass

class T2(T1):
    pass

def f3(p: Any) -> None:
    o0 = object()
    o1 = T1()
    o2 = T2()
    
    f3(o0)  # All OK: rule #2
    f3(o1)  # Also OK: rule #2
    f3(o2)  # Still OK: rule #2
```
x??

---

#### Function Return Type Inference

Background context: The text covers the implicit return type `Any` for functions without explicit annotations and how this can be used in various scenarios.

:p How does the function `f4()` with an implicit return type `Any` work?
??x
The function `f4()` has a return type of `Any`, meaning it can return any type. This is often useful when the exact type is not known or determined at the time of writing the code.

```python
def f4():
    ...

o4 = f4()  # inferred type: Any
```
x??

---

#### Union Types in Python 3.10
Python 3.10 introduced a more concise syntax for specifying union types, making type hints shorter and easier to read. This feature allows you to write `str | float` instead of `Union[str, float]`, which is more readable and doesn’t require importing `typing.Union`.
:p How does Python 3.10 simplify the representation of union types?
??x
In Python 3.10, the syntax for union types has been simplified by allowing you to use a vertical bar (`|`) between type annotations instead of using `Union`. This means that if you want to indicate that a variable can be either an integer or a float, you can write it as `int | float` directly.
```python
x: int | float = 3.14
```
This approach is more concise and readable compared to writing `Union[int, float]`.
x??

---

#### Type Hints for Lists with Specific Types
In Python, you can use type hints to specify the exact types of elements in a list or other collection. Starting from Python 3.9, this is particularly useful for ensuring that functions return lists with homogeneous data.
:p How do you define a function that returns a list of strings using modern type hints?
??x
Starting from Python 3.9, you can use the following syntax to indicate that a function `tokenize` should return a list of strings:
```python
def tokenize(text: str) -> list[str]:
    return text.upper().split()
```
This specifies that the `text` parameter is expected to be a string and that the function returns a list where every item is a string.
x??

---

#### Type Hints for Older Python Versions (3.7-3.8)
For older versions of Python, such as 3.7 and 3.8, you need to use special import statements or workarounds to enable type hinting with built-in collections like `list`.
:p How do you annotate a function in Python 3.7 or 3.8 using type hints for lists?
??x
For Python versions from 3.7 to 3.8, you need to use the `__future__` import statement to enable the use of type annotations with built-in collections like `list`. Here's an example:
```python
from __future__ import annotations

def tokenize(text: str) -> list[str]:
    return text.upper().split()
```
This ensures that in your code, you can write `list[str]` as a type hint without causing errors. If you try to use this approach with Python 3.6 or earlier, it will not work.
x??

---

#### Generic Types and Collections
In Python, collections like lists, sets, etc., are generally considered heterogeneous. However, for better type safety and to align with object-oriented programming principles, these can be made generic by specifying the types of items they should handle.
:p What is a benefit of using generic types in Python?
??x
A key benefit of using generic types in Python is improved type safety and clarity. By declaring that a list or set will contain specific types of objects, you can catch more errors at development time rather than runtime. For example:
```python
from typing import List

def tokenize(text: str) -> List[str]:
    return text.upper().split()
```
This function `tokenize` explicitly states it returns a list of strings. If you try to append non-string items, type-checkers and IDEs can flag these as errors.
x??

---

#### Legacy Support for Collection Types
For Python versions 3.7 and earlier, the syntax for generic types required importing from the `typing` module or using special import statements like `__future__`. This is because these older versions of Python did not natively support this feature without additional code.
:p How do you write type hints for a list in Python 3.5-3.7?
??x
For Python versions 3.5 to 3.7, you need to use the `typing` module and its `List` class for writing type hints. Here’s an example:
```python
from typing import List

def tokenize(text: str) -> List[str]:
    return text.upper().split()
```
This code specifies that the function `tokenize` returns a list of strings. The `List[str]` syntax ensures that you are adhering to the type hint guidelines for older Python versions.
x??

---

#### Type Hinting Generics in Standard Collections
In recent versions of Python, particularly from 3.9 onwards, you can use simpler type hints for common collections like lists and sets without importing additional modules. This makes your code cleaner and more readable.
:p Which standard collection types support simple generic type hints?
??x
Starting from Python 3.9, the following collections support simple generic type hints directly:
```python
from typing import List

def tokenize(text: str) -> List[str]:
    return text.upper().split()
```
This example uses `List[str]` to specify that `tokenize` returns a list of strings. Other supported types include `set`, `deque`, and others, as shown in the provided table.
x??

---

#### Type Hinting with `array.array`
Type hinting for the `array.array` module is challenging due to its constructor's reliance on type codes (`typecode`) that determine the data type stored within. As of Python 3.10, there isn’t a straightforward way to provide accurate type hints.
:p Why can't you easily use simple type hints with `array.array`?
??x
The `array.array` module in Python uses a `typecode` parameter during its construction that determines the data type stored within the array (e.g., integers or floats). Due to this, it is difficult to provide accurate and useful type hints for these arrays using simple syntax. For example, an array with `typecode='B'` can only hold integer values from 0 to 255.
Currently, there isn’t a good way in Python’s static typing system to reflect these constraints accurately in the type hints. This limitation means that developers need to rely on other forms of documentation or runtime checks to ensure correct usage.
x??

---

#### Introduction to Type Hinting Generics in Python 3.9+
Background context: The process of improving generic type hints in Python started with introducing `from __future__ import annotations` for Python 3.7, which was followed by making this behavior default in Python 3.9. This was part of a multi-year plan to streamline and improve the usability of type hinting generics from the standard collections.
:p What key changes were introduced in Python 3.9 regarding generic types?
??x
In Python 3.9, `list[str]` works without needing to import `from __future__ import annotations`. This change made it easier for developers to use generic type hints directly without additional imports.
x??

---
#### Deprecation of Redundant Generic Types from typing Module
Background context: Starting with Python 3.9, redundant generic types in the `typing` module were deprecated. These deprecations will be formally removed five years after Python 3.9 was released, which could be in Python 3.14 (code-named Python Pi).
:p What happens when using deprecated generic types from the typing module?
??x
When using deprecated generic types like `typing.Tuple`, type checkers should flag these deprecations for programs targeting Python 3.9 or newer. However, since this is a change in behavior and not syntax, no deprecation warnings will be issued by the Python interpreter.
x??

---
#### Annotating Tuples as Records
Background context: To annotate tuples as records, use `tuple` with types specified inside brackets. This is useful when you have fixed-length tuples that act like named fields but are still just tuples under the hood.
:p How do you annotate a tuple of geographic coordinates using `tuple`?
??x
You can annotate a tuple of geographic coordinates by specifying it as `tuple[float, float]`. For example:
```python
def geohash(lat_lon: tuple[float, float]) -> str:
    return gh.encode(*lat_lon, PRECISION)
```
This annotation indicates that the function expects a tuple with two `float` values.
x??

---
#### Using NamedTuples for Tuples as Records with Named Fields
Background context: For tuples with many fields or specific types used in multiple places, use `typing.NamedTuple`. This provides type safety and additional methods like `_asdict()`.
:p How do you define a `Coordinate` using `NamedTuple`?
??x
You can define a `Coordinate` using `NamedTuple` as follows:
```python
from typing import NamedTuple

class Coordinate(NamedTuple):
    lat: float
    lon: float
```
This defines a tuple subclass with named fields, making it more type-safe and easier to understand.
x??

---
#### Annotating Tuples of Unspecified Length
Background context: To annotate tuples of unspecified length used as immutable lists, use `tuple[Type, ...]`. This allows for any number of elements of the specified type.
:p How do you annotate a tuple with any number of string elements?
??x
You can annotate a tuple with any number of string elements using `tuple[str, ...]`:
```python
from collections.abc import Sequence

def columnize(sequence: Sequence[str], num_columns: int = 0) -> list[tuple[str, ...]]:
    # Function implementation here
```
This indicates that the function returns a list of tuples, each containing any number of `str` elements.
x??

---

