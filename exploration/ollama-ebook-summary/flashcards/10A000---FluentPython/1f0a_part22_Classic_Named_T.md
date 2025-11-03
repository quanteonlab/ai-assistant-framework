# Flashcards: 10A000---FluentPython_processed (Part 22)

**Starting Chapter:** Classic Named Tuples

---

#### Named Tuples Overview
Named tuples are factory functions for creating tuple subclasses with named fields. They provide a convenient way to create lightweight classes without having to define a full class statement.

Background context: The `namedtuple` function from the `collections` module allows you to create immutable instances with field names, which can be accessed by both name and index.
:p What are named tuples used for?
??x
Named tuples are used to create lightweight and easy-to-use data classes with named fields. They provide a way to access tuple elements via named attributes rather than positional indices, making the code more readable.
x??

---

#### Defining Named Tuples
You can define a named tuple using `namedtuple` from the `collections` module by providing a class name and a list of field names.

:p How do you define a named tuple?
??x
To define a named tuple, use the `namedtuple` function. The first argument is the class name as a string, and the second argument is a space-delimited string or an iterable of strings representing the field names.
```python
from collections import namedtuple

City = namedtuple('City', 'name country population coordinates')
```
x??

---

#### Using Named Tuples
Named tuples can be used similarly to regular tuples but with named access.

:p How do you create and use a named tuple?
??x
You create a named tuple instance by calling the `namedtuple` class as if it were a constructor. You provide the field values in order corresponding to the fields defined when creating the named tuple.
```python
tokyo = City('Tokyo', 'JP', 36.933, (35.689722 , 139.691667 ))
```
You can access the values using both positional indexing and by name:
```python
tokyo.population      # 36.933
tokyo.coordinates    # (35.689722, 139.691667)
tokyo[1]             # 'JP'
```
x??

---

#### Named Tuple Methods and Attributes
Named tuples offer several methods and attributes in addition to the inherited tuple methods.

:p What are some useful named tuple methods and attributes?
??x
Some useful named tuple methods and attributes include:
- `_fields`: A tuple containing the names of the fields.
- `_make(iterable)`: A class method that creates a new instance from an iterable.
- `_asdict()`: An instance method that returns a dictionary representation of the named tuple.

Example usage:
```python
City._fields  # ('name', 'country', 'population', 'coordinates')
delhi = City._make(delhi_data)
delhi._asdict()
# {'name': 'Delhi NCR', 'country': 'IN', 'population': 21.935, 
#  'location': Coordinate(lat=28.613889, lon=77.208889)}
```
x??

---

#### Default Values in Named Tuples
Named tuples can accept default values for fields using the `defaults` keyword-only argument.

:p How do you define a named tuple with default values?
??x
You define a named tuple with default values by including the `defaults` parameter. The `defaults` should be an iterable of N default values corresponding to the rightmost N fields.
```python
Coordinate = namedtuple('Coordinate', 'lat lon reference', defaults=['WGS84'])
```
Example usage:
```python
Coordinate(0, 0)  # Coordinate(lat=0, lon=0, reference='WGS84')
Coordinate._field_defaults  # {'reference': 'WGS84'}
```
x??

---

#### Hack: Injecting Methods into Named Tuples
You can add methods to a named tuple by defining them as class attributes.

:p How do you add a method to a named tuple?
??x
To add a method to a named tuple, define the function and then assign it to a class attribute. This approach is a hack because it doesn't provide the full class statement support.
```python
Card.suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)
def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    suit_value = card.suit_values[card.suit]
    return rank_value * len(card.suit_values) + suit_value
Card.overall_rank = spades_high
```
Example usage:
```python
lowest_card = Card('2', 'clubs')
highest_card = Card('A', 'spades')
lowest_card.overall_rank()  # 0
highest_card.overall_rank()  # 51
```
x??

---

#### Dynamic Class Method Attachment
Background context explaining the dynamic nature of attaching methods to classes in Python. This allows for flexibility and can be useful for quick implementations where class modifications are needed without changing the actual class definition.

:p How can a method be attached dynamically to an existing class in Python?
??x
You can attach a method to an existing class by defining it outside the class statement but still within the scope of the class. This is demonstrated with the `overall_rank` method, which works as expected despite not being defined inside the class definition.

```python
class Cards:
    def overall_rank(self):
        # Method implementation here
        pass

# Attaching a method to an instance or the class itself
Cards.overall_rank = overall_rank
```
x??

---

#### Using `typing.NamedTuple` for Type Annotations
Background context on using type annotations in Python classes, especially when working with complex data structures like coordinates. This helps in improving code readability and maintaining type consistency.

:p How can you create a `Coordinate` class with typed fields using `typing.NamedTuple`?
??x
You can create a `Coordinate` class with typed fields by using the `NamedTuple` from the `typing` module. Each field must be annotated with its corresponding data type, and default values can also be specified.

```python
from typing import NamedTuple

class Coordinate(NamedTuple):
    lat: float
    lon: float
    reference: str = 'WGS84'
```
x??

---

#### Differences Between `typing.NamedTuple` and `collections.namedtuple`
Background context on the differences between Python's built-in `namedtuple` and `NamedTuple` from the `typing` module. While both generate similar classes, there are some key distinctions that affect their use cases.

:p What are the main differences between `typing.NamedTuple` and `collections.namedtuple`?
??x
The main difference lies in the type annotations provided by `typing.NamedTuple`. When using `NamedTuple`, every instance field must be annotated with a type. Additionally, `NamedTuple` introduces an `__annotations__` class attribute which is useful for static analysis tools but has no effect at runtime.

In contrast, `collections.namedtuple` does not require type annotations and lacks the `__annotations__` attribute.

```python
# NamedTuple example
from typing import NamedTuple

class Coordinate(NamedTuple):
    lat: float
    lon: float
    reference: str = 'WGS84'

# Collections.namedtuple example
from collections import namedtuple

Coordinate = namedtuple('Coordinate', ['lat', 'lon', 'reference'], defaults=['WGS84'])
```
x??

---

#### Type Hints Overview
Type hints, also known as type annotations, provide a way to declare the expected types of function arguments, return values, variables, and attributes. They are primarily used for documentation and static analysis tools such as Mypy or IDEs like PyCharm.

:p What is the primary purpose of type hints?
??x
The primary purpose of type hints is to serve as documentation that can be verified by IDEs and type checkers. While they do not enforce type checking at runtime, they help in catching type-related errors during development through static analysis tools.
x??

---

#### Python Type Hints vs Runtime Enforcement
Type hints are not enforced by the Python bytecode compiler or interpreter. They have no impact on the runtime behavior of programs.

:p What does it mean when type hints are "not enforced" at runtime?
??x
It means that although you can define types for variables, arguments, and return values in your code using type hints, these type checks will only be performed by external tools like static analyzers or linters, not by the Python interpreter during execution.
x??

---

#### Example of Coordinate NamedTuple
The provided example demonstrates a `Coordinate` class defined with `typing.NamedTuple`, showing that even though you have specified types, they are not enforced at runtime.

:p What happens when you attempt to create an instance of `Coordinate` using incorrect types?
??x
When you try to instantiate `Coordinate` with non-float arguments (a string and `None`), the Python interpreter does not raise any errors or warnings. The `Coordinate` object is created with those values, but static analysis tools like Mypy will flag these as type errors.

Example Code:
```python
import typing

class Coordinate(typing.NamedTuple):
    lat: float
    lon: float

trash = Coordinate('Ni.', None)
```

Mypy Output:
```
mypy nocheck_demo.py
nocheck_demo.py:8: error: Argument 1 to "Coordinate" has incompatible type "str"; expected "float"
nocheck_demo.py:8: error: Argument 2 to "Coordinate" has incompatible type "None"; expected "float"
```

x??

---

#### Syntax of Variable Annotations in NamedTuple
Variable annotations in `typing.NamedTuple` and `@dataclass` use the syntax defined in PEP 526, which allows you to specify types for attributes.

:p What is the basic syntax for variable annotations in a class statement?
??x
The basic syntax for variable annotations in a class statement is as follows:
```python
var_name: some_type
```
For example, in a `Coordinate` NamedTuple, you would define it like this:

```python
class Coordinate(typing.NamedTuple):
    lat: float
    lon: float
```

Here, the type of `lat` and `lon` is annotated as `float`.

x??

---

#### Acceptable Types for Annotations
In the context of defining a data class, acceptable types for annotations include concrete classes (e.g., `str`, `FrenchDeck`) and parameterized collection types (e.g., `list[int]`, `tuple[str, float]`).

:p What are some examples of acceptable type hints when defining a data class?
??x
Some examples of acceptable type hints in the context of defining a data class include:

- Concrete classes: `str`, `int`, `FrenchDeck`
- Parameterized collection types: `list[int]`, `tuple[str, float]`

For instance:
```python
class Deck(typing.NamedTuple):
    cards: list[str]

class FrenchDeck(typing.NamedTuple):
    suits: tuple[str]
    ranks: dict[str, int]
```

x??

---

#### Variable Annotations and `typing.Optional`

Variable annotations can be used to declare the expected type of a variable. This is particularly useful with `Optional` from the `typing` module, which allows you to specify that a field can either hold a certain type or `None`. When defining such fields in a class or a dataclass, Python stores these annotations but does not enforce them at runtime.

:p What are variable annotations and how do they work with `Optional[str]`?
??x
Variable annotations allow you to declare the expected types of variables. With `Optional[str]`, it signifies that a field can either be a string (`str`) or `None`. When defined in a class, Python stores these as part of the class's metadata but does not enforce type checking at runtime.

For example:
```python
from typing import Optional

class ExampleClass:
    name: Optional[str] = None
```
In this case, `name` can be either `str` or `None`, and is stored in the `__annotations__` dictionary of the class. However, attempting to assign a non-`str` value directly to `name` will not result in an error at runtime.

x??

---

#### The Role of `__annotations__`

The `__annotations__` special attribute holds type hints declared using variable annotations. This dictionary is used by tools like `typing.NamedTuple` and the `@dataclass` decorator to enhance class definitions, even if the class definition itself does not use these annotations directly.

:p What does the `__annotations__` attribute store in a class with type hints?
??x
The `__annotations__` attribute stores type hints declared using variable annotations. These hints are stored as key-value pairs where keys are variable names and values are their corresponding types. For example, if you have:
```python
class ExampleClass:
    name: str = 'John'
```
Then `ExampleClass.__annotations__` would be:
```python
{'name': <class 'str'>}
```

However, the type hints in `__annotations__` do not affect attribute creation. Only those that are bound to values become class attributes. For instance, if you have:
```python
class ExampleClass:
    name: str = 'John'
    age: int
```
Only `name` would be a class attribute with the value `'John'`, while `age` is just an annotation without any default value.

x??

---

#### Plain Class vs. NamedTuple

In Python, you can use variable annotations in plain classes to document types, but these type hints are not enforced at runtime and do not create attributes by themselves. On the other hand, when using `typing.NamedTuple`, these annotations become part of the class structure as both attributes and their corresponding default values.

:p What is the difference between a plain class with variable annotations and a NamedTuple in terms of attribute creation?
??x
In a plain class with variable annotations, type hints are stored in the `__annotations__` dictionary but do not create actual attributes unless they are bound to values. For example:
```python
class DemoPlainClass:
    name: str = 'John'
    age: int

print(DemoPlainClass.__annotations__)
# {'name': <class 'str'>, 'age': <class 'int'>}
```
Here, `name` is a class attribute with the value `'John'`, but `age` is just an annotation without any default value. Attempting to use `DemoPlainClass.age` would result in an `AttributeError`.

In contrast, when using `typing.NamedTuple`, these annotations are stored in the `__annotations__` dictionary and also become part of the class as attributes with their corresponding default values:
```python
from typing import NamedTuple

class DemoNTClass(NamedTuple):
    name: str = 'John'
    age: int

print(DemoNTClass.__annotations__)
# {'name': <class 'str'>, 'age': <class 'int'>}
```
Here, both `name` and `age` are class attributes with their respective default values.

x??

---

#### NamedTuple vs DataClass: Instance Attributes
Background context explaining the difference between `NamedTuple` and `DataClass`. Highlight that both are used to create classes with a fixed set of attributes, but they handle these attributes differently.

:p How does `DemoNTClass` differ from `DemoDataClass` in terms of instance attribute handling?
??x
`DemoNTClass` uses descriptors for the `a` and `b` attributes. These descriptors behave like property getters, making them read-only for instances of `DemoNTClass`. The descriptor logic ensures that these attributes are derived from the tuple-like nature of the class, treating it as immutable.

On the other hand, `DemoDataClass` does not use descriptors for its instance attributes; instead, they are regular public attributes. This means you can set and get their values directly on instances.
x??

---
#### NamedTuple Class Attributes
Background context explaining that `NamedTuple` creates class-level attributes (`a`, `b`) with descriptor behavior.

:p What are the characteristics of `a` and `b` in `DemoNTClass`?
??x
In `DemoNTClass`, `a` and `b` are class-level attributes implemented as descriptors. These descriptors behave like property getters, making them read-only for instances of `DemoNTClass`. The descriptor logic ensures that these attributes are derived from the tuple-like nature of the class, treating it as immutable.
x??

---
#### DataClass Class Attributes
Background context explaining how `DataClass` handles instance and class-level attributes differently.

:p How does `DemoDataClass` handle its `a`, `b`, and `c` attributes?
??x
In `DemoDataClass`, `a` is an instance attribute controlled by a descriptor, making it read-only for instances. It has a default value of `<class 'int'>`. The `b` attribute also behaves as an instance attribute with a default float value (`1.1`). However, `c` is just a plain class-level attribute and does not get bound to the instances.

Additionally, `DemoDataClass` provides a custom docstring that includes annotations for its attributes.
x??

---
#### NamedTuple Constructor
Background context explaining how the constructor of `NamedTuple` works.

:p How does the construction of an instance of `DemoNTClass` work?
??x
The construction of `nt` (an instance of `DemoNTClass`) requires providing at least the `a` argument. The `b` argument has a default value of `1.1`, making it optional. When you attempt to set values for `nt.a`, `nt.b`, or any non-existent attribute like `nt.z`, you will receive an `AttributeError`.

Example:
```python
>>> nt = DemoNTClass(8)
>>> nt.a  # 8
>>> nt.b  # 1.1 (default value if not provided)
```
x??

---
#### DataClass Constructor and Annotations
Background context explaining the constructor of `DataClass` and its annotations.

:p How does the construction of an instance of `DemoDataClass` work, and what do the `__annotations__` contain?
??x
The construction of `DemoDataClass` also requires providing at least the `a` argument. The `b` argument has a default value of `1.1`, making it optional.

When you create an instance:
```python
>>> d = DemoDataClass(8)
```
The `__annotations__` attribute contains a dictionary with annotations for the attributes.
```python
>>> from demo_dc import DemoDataClass
>>> DemoDataClass.__annotations__
{'a': <class 'int'>, 'b': <class 'float'>}
```

The `__doc__` provides a custom docstring that reflects the class and its annotations:
```python
>>> DemoDataClass.__doc__
'DemoDataClass(a: int, b: float = 1.1)'
```
Attempting to access or modify `DemoDataClass.a` directly will raise an `AttributeError`, as it is only present in instances.
x??

---

