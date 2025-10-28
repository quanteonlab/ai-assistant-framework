# Flashcards: 10A000---FluentPython_processed (Part 21)

**Starting Chapter:** Whats new in this chapter

---

#### Data Classes Overview
Background context explaining data classes. Data classes are simple classes that serve primarily as data containers, containing fields without additional methods or logic. They are useful for reducing boilerplate code and making code more readable.

:p What is a data class?
??x
Data classes are lightweight classes designed to store and manage data efficiently. They provide a way to create objects with minimal effort by focusing on storing fields rather than implementing complex behavior.
x??

---
#### Collections.namedtuple
Background context explaining the `collections.namedtuple` function. The `namedtuple` is one of the simplest ways to create a data class, available since Python 2.6. It allows creating classes that can store named fields.

:p What does `collections.namedtuple` do?
??x
`collections.namedtuple` creates a subclass of tuple with named fields. This function takes two arguments: the name of the class and a list or space-separated string of field names.
```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x)   # Output: 1
```
x??

---
#### Typing.NamedTuple
Background context explaining the `typing.NamedTuple`. Introduced in Python 3.5, this is an alternative to `collections.namedtuple` that allows for type hints on fields. It supports class syntax.

:p How does `typing.NamedTuple` differ from `namedtuple`?
??x
`typing.NamedTuple` extends the functionality of `namedtuple` by allowing you to specify types for each field using Python's type hinting system. Additionally, it supports class syntax, making it more flexible and easier to work with.

Example:
```python
from typing import NamedTuple

class Point(NamedTuple):
    x: int
    y: int

p = Point(1, 2)
print(p.x)   # Output: 1
```
x??

---
#### Dataclasses.dataclass
Background context explaining the `@dataclasses.dataclass` decorator. This is a more advanced and flexible way to create data classes, introduced in Python 3.7. It allows for more customization compared to `collections.namedtuple` and `typing.NamedTuple`.

:p What does `@dataclasses.dataclass` offer over other data class alternatives?
??x
The `@dataclasses.dataclass` decorator provides a more flexible and customizable way to create data classes by allowing you to add methods like `__init__`, `__repr__`, etc., while still keeping the focus on storing fields. It also supports default values, frozen attributes, and more.

Example:
```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int = 0
    y: int = 0

p = Point(1, 2)
print(p.x)   # Output: 1
```
x??

---

#### Classic Named Tuples
Background context: The classic `NamedTuple` is a factory function from the `collections` module that simplifies creating named tuples. It allows you to define classes with fields and automatically generates methods like `__init__`, `__repr__`, and `__eq__`.

:p What are the key differences between using `NamedTuple` in Python 3.6+ compared to its earlier versions?
??x
The key differences include enhanced readability, support for PEP 526 variable annotations, and the ability to override or add methods directly within a class statement.

Code example:
```python
from collections import namedtuple

Coordinate = namedtuple('Coordinate', 'lat lon')
```
In this example, `Coordinate` is created with fields `lat` and `lon`, and it automatically has useful methods like `__repr__` and `__eq__`.

x??

---

#### Typing.NamedTuple
Background context: `typing.NamedTuple` is a type annotation extension of the classic `NamedTuple`. It not only provides similar functionality but also adds type hints to each field, making your code more readable and maintainable.

:p How does `typing.NamedTuple` differ from the classic `NamedTuple` in terms of syntax?
??x
`typing.NamedTuple` allows you to specify types for fields using PEP 526 syntax or by passing a list of tuples where each tuple contains the field name and its type. Additionally, it can be used directly in class statements with type annotations.

Code example:
```python
import typing

Coordinate = typing.NamedTuple('Coordinate', [('lat', float), ('lon', float)])
```
This creates a `Coordinate` class with fields `lat` (of type `float`) and `lon` (also of type `float`).

x??

---

#### Data Class Introduction
Background context: Introduced in Python 3.7, data classes are a convenient way to create simple data containers with minimal boilerplate code. They automatically generate common methods like `__init__`, `__repr__`, `__eq__`, and more.

:p What is the main purpose of using data classes over traditional class definitions?
??x
The main purpose is to reduce redundancy in writing boilerplate code for initializing, representing, comparing, and hashing objects. Data classes handle these tasks automatically, making your code cleaner and more readable.

Example:
```python
from dataclasses import dataclass

@dataclass
class Coordinate:
    lat: float
    lon: float
```
This `Coordinate` class is created with automatic methods generated by the `dataclass` decorator, reducing the need to write these methods manually.

x??

---

#### TypedDict Introduction
Background context: While `typing.TypedDict` shares some syntax similarities with data classes and named tuples, it serves a different purpose. It's used for creating dictionaries where keys represent field names and values can be of any type.

:p How does `typing.TypedDict` differ from data classes?
??x
`typing.TypedDict` is not meant to create concrete classes that you can instantiate but rather to provide static typing hints for dictionaries or records. It focuses on ensuring type consistency in function parameters or variables, whereas data classes are designed to be instantiated objects with methods and attributes.

Example:
```python
from typing import TypedDict

class CoordinateDict(TypedDict):
    lat: float
    lon: float
```
This `CoordinateDict` class is a type hint for dictionaries that should contain keys 'lat' and 'lon', both of which are expected to be floats.

x??

---

#### Summary of Data Class Builders
Background context: The chapter covers various tools like classic `NamedTuple`, `typing.NamedTuple`, data classes, and `TypedDict`. Each tool has its strengths and is suitable for different scenarios based on the need for static typing, boilerplate reduction, or object instantiation.

:p What are some key differences between `collections.namedtuple` and `dataclasses.dataclass`?
??x
Key differences include:
- `NamedTuple` subclasses `tuple` and does not support instance modifications.
- `dataclass` can be instantiated and supports methods like `__init__`, `__repr__`, etc., making it more flexible for object-oriented design.

Example:
```python
from collections import namedtuple

Coordinate = namedtuple('Coordinate', 'lat lon')

# vs.

from dataclasses import dataclass

@dataclass
class Coordinate:
    lat: float
    lon: float
```
The `NamedTuple` version is immutable, while the `dataclass` version can be modified and supports additional methods.

x??

---

#### Coordinate Class Definition
The provided `Coordinate` class is a dataclass that stores geographical coordinates (latitude and longitude). The class uses Python's `@dataclass` decorator to simplify its definition, making it easier to work with object attributes and generating methods like `__init__`, `__repr__`, etc.
:p What does the `Coordinate` class do?
??x
The `Coordinate` class defines a geographical coordinate point using latitude (lat) and longitude (lon). It includes a custom string representation method that formats the coordinates as degrees, indicating North/South for latitude and East/West for longitude. This helps in easily displaying or logging geographical positions.
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float

    def __str__(self):
        ns = 'N' if self.lat >= 0 else 'S'
        we = 'E' if self.lon >= 0 else 'W'
        return f'{abs(self.lat):.1f}°{ns}, {abs(self.lon):.1f}° {we}'
```
x??

---

#### Dataclass vs NamedTuple
The text explains the differences between `dataclass`, `collections.namedtuple`, and `typing.NamedTuple` in Python, focusing on their usage, features, and behaviors.
:p What are the main differences between `dataclass`, `namedtuple`, and `NamedTuple`?
??x
- **Mutable Instances**: Both `namedtuple` and `NamedTuple` produce immutable instances by default. In contrast, a class created with `@dataclass` can be mutable unless the `frozen=True` argument is used.
- **Class Statement Syntax**: Only `NamedTuple` and `@dataclass` support the regular class statement syntax, allowing for easier addition of methods and docstrings.
- **Construct Dict Methods**: Both `namedtuple` variants provide an `_asdict()` method to convert instance fields into a dictionary. The `dataclasses` module offers `dataclasses.asdict(x)` as well.
- **Field Names and Defaults**: All three class builders allow accessing field names, but `NamedTuple` and `@dataclass` use different attributes (`_fields` vs. `__annotations__`). Field defaults can be accessed through `_field_defaults` for both, while `__annotations__` holds type hints in classes created with `@dataclass`.
- **Field Types**: `NamedTuple` and `@dataclass` store field names to their corresponding type annotations in the `__annotations__` class attribute.
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ExampleClass:
    a: int  # Type annotation for field 'a'
```
x??

---

#### Immutable vs Mutable Classes
The text discusses how classes generated by different builders (`namedtuple`, `NamedTuple`, and `@dataclass`) differ in their mutability.
:p How do the classes created by these builders differ in terms of mutability?
??x
- **`namedtuple` and `NamedTuple`**: These create immutable instances. Attempting to modify fields after instantiation will result in an error.
- **`@dataclass` with `frozen=True`**: This creates a class where all fields are read-only, throwing exceptions if you try to assign new values to them once the object is initialized.
- **`@dataclass` without `frozen=True`**: This results in mutable instances, allowing modifications after initialization. You can use the `replace()` method from the `dataclasses` module to create a new instance with updated fields.
```python
from dataclasses import dataclass, replace

@dataclass(frozen=False)
class ExampleClass:
    a: int
    
# To modify an object in place:
example = ExampleClass(a=5)
example.a = 10  # This works for mutable classes

# To create a new instance with updated fields:
new_example = replace(example, a=20)  # This creates a copy with 'a' set to 20
```
x??

---

#### Accessing Type Annotations
The text explains the differences in accessing type annotations between `NamedTuple` and `@dataclass`.
:p How do you access type annotations for fields in classes created by these builders?
??x
- **`NamedTuple` and `typing.NamedTuple`**: These classes have a `_fields` attribute containing a tuple of field names, and an `_field_types` dictionary mapping each field name to its type.
- **`@dataclass` with `frozen=True`**:
  - Use the `__annotations__` class attribute to get a dictionary of field names to their corresponding types. This is also available for mutable classes created without `frozen=True`.
  - Avoid reading directly from `__annotations__`, as it may not always contain accurate type information.
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ExampleClass:
    a: int
    b: float
    
# Accessing annotations with @dataclass
example_class = ExampleClass(1, 2.0)
print(example_class.__annotations__)
```
x??

---

