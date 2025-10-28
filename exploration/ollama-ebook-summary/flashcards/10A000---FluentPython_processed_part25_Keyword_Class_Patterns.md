# Flashcards: 10A000---FluentPython_processed (Part 25)

**Starting Chapter:** Keyword Class Patterns

---

#### Simple Class Patterns
Background context: Simple class patterns are used to match class instances by type. These patterns can be used as subpatterns within more complex pattern matching cases, such as those involving sequences and mappings. The syntax for these patterns resembles a constructor invocation.

Example of usage:
```python
match x:
    case float():
        do_something_with(x)
```
This pattern matches `x` if it is an instance of the `float` class. However, using just `case float:` without parentheses can lead to bugs because Python will interpret `float` as a variable.

:p Can you explain why `case float:` without parentheses might be problematic in pattern matching?
??x
Using `case float:` without parentheses matches any subject because Python sees `float` as a variable. This means it can match anything, which is not the intended behavior for class patterns. To correctly match instances of `float`, use `case float():`.

```python
match x:
    case float():
        do_something_with(x)
```
This ensures that only `x` instances of type `float` are matched.
x??

---

#### Keyword Class Patterns
Background context: Keyword class patterns allow you to match specific attributes of a class instance. They can be used in combination with simple class patterns and other types of patterns.

Example usage:
```python
import typing

class City(typing.NamedTuple):
    continent: str
    name: str
    country: str

cities = [
    City('Asia', 'Tokyo', 'JP'),
    City('Asia', 'Delhi', 'IN'),
    City('North America', 'Mexico City', 'MX'),
    City('North America', 'New York', 'US'),
    City('South America', 'São Paulo', 'BR'),
]

def match_asian_cities():
    results = []
    for city in cities:
        match city:
            case City(continent='Asia'):
                results.append(city)
    return results
```
This code matches and collects all Asian cities from the `cities` list.

:p How would you modify the pattern to collect the country names of Asian cities?
??x
To collect the country names, you can use a keyword class pattern with an additional variable for the country attribute:

```python
def match_asian_countries():
    results = []
    for city in cities:
        match city:
            case City(continent='Asia', country=cc):
                results.append(cc)
    return results
```
This will bind `cc` to the `country` attribute of each matched Asian city and collect these values into the `results` list.
x??

---

#### Keyword Class Patterns
Keyword class patterns allow for readable matching of classes based on their attributes. They work with any class that has public instance attributes and are very flexible, but can be verbose.

:p What is a keyword class pattern?
??x
A keyword class pattern allows you to match instances of a class based on specific attribute values using `match` statements in Python 3.10 or later. These patterns are more readable and flexible compared to positional patterns because they explicitly name the attributes involved.
```python
def match_asian_countries_pos():
    results = []
    for city in cities:
        match city:
            case City('Asia', _, country):
                results.append(country)
    return results
```
x??

---

#### Positional Class Patterns
Positional class patterns are a type of pattern matching that uses the `__match_args__` attribute to match instances based on their positional attributes. They require explicit support by the class, but can be more convenient in some cases.

:p What is a positional class pattern?
??x
A positional class pattern matches instances of a class based on the order and values of their public instance attributes defined in `__match_args__`. The `__match_args__` attribute lists the names of the attributes in the order they should be matched.
```python
def match_asian_cities_pos():
    results = []
    for city in cities:
        match city:
            case City('Asia'):
                results.append(city)
    return results
```
x??

---

#### `__match_args__` Attribute
The `__match_args__` attribute is a special class attribute that the `@dataclass`, `namedtuple`, and other class builders automatically create. It lists the names of the instance attributes in the order they will be used in positional patterns.

:p What does the `__match_args__` attribute do?
??x
The `__match_args__` attribute is a special class-level attribute that stores a tuple of strings representing the names of the public instance attributes of a dataclass. This attribute enables pattern matching based on the order and values of these attributes.
```python
>>> City.__match_args__
('continent', 'name', 'country')
```
x??

---

#### Combining Keyword and Positional Arguments in Patterns
You can use both keyword and positional arguments in patterns, even though not all instance attributes may be listed in `__match_args__`. This allows for more flexibility in matching instances.

:p Can you combine keyword and positional arguments in a pattern?
??x
Yes, you can combine keyword and positional arguments in a single pattern. This is useful when the class has many attributes but you only want to match based on some of them.
```python
def match_asian_countries_pos():
    results = []
    for city in cities:
        match city:
            case City('Asia', _, country):
                results.append(country)
    return results
```
x??

---

#### Data Class Builders Overview
Data class builders like `collections.namedtuple`, `typing.NamedTuple`, and `dataclasses.dataclass` generate data classes from descriptions provided as arguments to factory functions or from class statements with type hints.

:p What are data class builders?
??x
Data class builders are tools that help you create lightweight, simple data classes. They automatically add special methods like `__init__`, `__repr__`, and others based on the fields specified in the class definition.
```python
from dataclasses import dataclass

@dataclass
class City:
    continent: str
    name: str
    country: str
```
x??

---

#### Type Hints and Runtime Effects
Type hints, introduced with Python 3.6 via PEP 526, are used to annotate attributes in class statements. However, they have no effect at runtime; Python remains a dynamic language.

:p What is the role of type hints?
??x
Type hints provide static typing information that can be used by external tools like Mypy for static analysis and error detection. At runtime, Python treats these as comments and does not enforce any type checks.
```python
def add_numbers(a: int, b: int) -> int:
    return a + b  # Type hints are ignored at runtime
```
x??

---

#### Syntax for Variable Annotations
Variable annotations introduced in PEP 526 allow you to specify types directly in the class definition.

:p What is variable annotation?
??x
Variable annotations let you specify the expected type of attributes or parameters directly in the class definition. This helps with static analysis and documentation.
```python
from typing import List

class Person:
    name: str
    age: int
    friends: List[str]
```
x??

---

#### `default_factory` Option in Dataclasses
The `dataclasses.field()` function's `default_factory` option allows you to specify a callable that will be called when an instance of the class is created without providing a value for that attribute.

:p What does the `default_factory` option do?
??x
The `default_factory` option in `dataclasses.field()` provides a way to set default values dynamically. The function specified as `default_factory` is only called if no other value is provided during object initialization.
```python
from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int = 0
    friends: List[str] = field(default_factory=list)
```
x??

---

#### `typing.ClassVar` and `dataclasses.InitVar`
These are special pseudo-type hints that are useful in the context of data classes. `ClassVar` is used to annotate class-level variables, while `InitVar` is used for values passed during initialization.

:p What are `ClassVar` and `InitVar`?
??x
- `ClassVar`: Used to declare a variable that is intended to be used as a class attribute rather than an instance attribute.
- `InitVar`: Used in the constructor of data classes to accept arguments that should not be stored on the instance but can be used for initialization purposes.

```python
from dataclasses import InitVar, dataclass

@dataclass
class Config:
    config: ClassVar[str] = "default"
    param1: InitVar[int]
```
x??

---

#### Dublin Core Schema Example
The Dublin Core Metadata Initiative provides a metadata schema that can be used to describe resources. Using `dataclasses.fields`, you can iterate over the attributes of a resource instance in a custom `__repr__`.

:p How can you use dataclasses fields with the Dublin Core Schema?
??x
You can use `dataclasses.fields` to inspect and manipulate the fields of a class representing a Dublin Core metadata object. This can be useful for generating dynamic representations or performing validations.
```python
from dataclasses import asdict, field, fields

class DublinCoreResource:
    title: str = ""
    creator: List[str] = field(default_factory=list)
    date: str = ""

resource = DublinCoreResource()
for field in fields(resource):
    print(f"{field.name}: {getattr(resource, field.name)}")
```
x??

#### Data Class as a Code Smell
Background context: The use of data classes should be approached with caution, especially when they are used without any logic or business rules. This is because data and the functions that touch it should ideally be together within the same class. Overusing data classes can lead to a separation of concerns in object-oriented programming principles.

:p What does "data class as a code smell" refer to?
??x
When using data classes without any associated logic or business rules, it suggests that the class may have been improperly designed. Data and behavior should ideally be encapsulated within the same class for better maintainability and adherence to OOP principles.
x??

---

#### Why not use namedtuple or NamedTuple for Data Classes
Background context: While namedtuples are a useful tool for creating lightweight classes with only data fields, they lack some of the features that modern data classes provide. The key differences include methods and validation support.

:p What are the reasons to avoid using `namedtuple` or `NamedTuple` for implementing data classes?
??x
The main reasons to avoid using `namedtuple` or `NamedTuple` over a data class are:
1. **Methods**: Namedtuples do not allow defining instance methods.
2. **Validation**: Data classes can perform type validation and provide default values, which is not possible with namedtuples.
3. **Flexibility**: Data classes can be extended more easily with additional attributes or behaviors.

Example code showing the limitations of `namedtuple`:
```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

point = Point(1, 2)
# point.x = 5  # This would raise an error because tuples are immutable
```
x??

---

#### Inappropriate Use of Data Classes
Background context: Data classes should not be used where type validation or value validation is required. They are most suitable for simple data storage with minimal logic.

:p Where is it inappropriate to use data classes?
??x
It is inappropriate to use data classes in scenarios requiring:
1. **API compatibility with tuples or dictionaries**: When the interface needs to match exactly.
2. **Beyond PEP 484 and PEP 526 type validation**: If more complex validation logic is needed.
3. **Value validation or conversion**: Data classes do not support this out of the box.

Example illustrating inappropriate use:
```python
# Inappropriate usage example
class Spam:
    repeat: int = 99

spam = Spam()
print(spam.repeat)  # This will work, but no type checking is done at runtime.
```
x??

---

#### Attributes and Class Decorators
Background context: The introduction of `@dataclass` in Python 3.7 introduced a new way to define classes with data fields more concisely. However, it brought changes to how instance and class attributes are defined.

:p How does the use of `@dataclass` affect attribute definitions?
??x
The use of `@dataclass` affects attribute definitions by reversing the convention for declaring top-level attributes:
- Attributes declared at the top level with a type hint become instance attributes.
- Attributes without a type hint remain class attributes.

Example showing this behavior:
```python
from dataclasses import dataclass

@dataclass
class Spam:
    repeat: int  # This is an instance attribute
```
x??

---

#### Alternative Syntax for Classes
Background context: The author proposes an alternative syntax to make class definitions more readable and reduce the complexity introduced by PEP 526.

:p What is the proposed alternative syntax for classes?
??x
The proposed alternative syntax uses a `.prefix` for instance attributes, making it clear which are instance attributes and which are class attributes. Here’s how it would look:
```python
@dataclass
class HackerClubMember:
    .name: str
    .guests: list = field(default_factory=list)
    .handle: str = ''
    all_handles: ClassVar[set] = set()
```
This syntax avoids the exceptions and makes the code more readable.

x??

---

#### Guido van Rossum's Time Machine
Background context: Guido van Rossum is often credited with implementing new features quickly, as evidenced by his alleged "time machine" ability. Python has lacked a quick way to declare instance attributes in classes until the introduction of `@dataclass`.

:p What does the "Guido's time machine" metaphor imply about Python development?
??x
The metaphor implies that Guido van Rossum is able to implement features very quickly, almost as if he had a time machine. It highlights his role in shaping Python’s evolution with new language features and improvements.

Example of what might have been:
```python
# Hypothetical example of how class attribute declarations could look without `@dataclass`
class Point:
    x: int  # This would be a class attribute if not for `@dataclass`
```

x??

---

