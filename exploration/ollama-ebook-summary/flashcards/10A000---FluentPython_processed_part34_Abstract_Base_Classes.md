# Flashcards: 10A000---FluentPython_processed (Part 34)

**Starting Chapter:** Abstract Base Classes

---

#### Tokenize Function
Background context: The `tokenize` function is part of a larger program that processes Unicode characters and their names, creating an inverted index. It uses regular expressions to identify words within character names and returns these as uppercased tokens.

:p What does the `tokenize` function do?
??x
The `tokenize` function takes a string `text`, finds all word-like substrings using a regular expression, and yields each match in uppercase.
```python
import re

RE_WORD = re.compile(r'\w+')

def tokenize(text: str) -> Iterator[str]:
    """return iterable of uppercased words"""
    for match in RE_WORD.finditer(text):
        yield match.group().upper()
```
x??

---

#### Name Index Function
Background context: The `name_index` function creates an inverted index mapping Unicode character names to sets of characters. It uses the `tokenize` function to split these names into words and maps each word back to its corresponding character.

:p What does the `name_index` function do?
??x
The `name_index` function generates an inverted index where keys are word tokens derived from Unicode character names, and values are sets of characters that have those words in their names. It uses a generator expression to efficiently process and map each character within a given range.
```python
from collections.abc import Iterator

def name_index(start: int = 32, end: int = sys.maxunicode + 1) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for char in (chr(i) for i in range(start, end)):
        if name := unicodedata.name(char, ''):
            for word in tokenize(name):
                index.setdefault(word, set()).add(char)
    return index
```
x??

---

#### Generic Mappings and ABCs
Background context: The text discusses the use of generic mappings (like `dict`) and abstract base classes (ABCs) to improve code flexibility. It also mentions how to handle different versions of Python for type annotations.

:p What is the difference between using `abc.Mapping` versus `dict` in function signatures?
??x
Using `abc.Mapping` instead of `dict` in a function signature makes the function more flexible by allowing it to accept any object that implements the Mapping interface, such as `defaultdict`, `ChainMap`, or even custom mapping types. This approach aligns with Postel's law ("be liberal in what you accept"), ensuring broader compatibility while maintaining type safety.
x??

---

#### Type Hints and Generics
Background context: The text explains how to use generic types from the collections module and typing module for better type annotations, especially when working across different Python versions.

:p How can you handle generic collections like lists or dictionaries in function signatures?
??x
For generic collections like lists (`list`), dictionaries (`dict`), and sets (`set`), it's preferable to use abstract collection types from `collections.abc`, such as `Sequence`, `Mapping`, or `Set`. This approach is more flexible, allowing the caller to pass objects that implement the relevant interface, even if they are not concrete collections like `list` or `dict`.
```python
from typing import List, Mapping

def process_list(data: List[str]) -> str:
    return ', '.join(data)

def use_mapping(color_map: Mapping[str, int]) -> int:
    # Process color_map
    pass
```
x??

---

#### Postel's Law and Type Hints
Background context: The text discusses the importance of using `abc.Mapping` for flexibility in function signatures and how it relates to Postel's law.

:p How does Postel's law influence type hints?
??x
Postel's law advises being conservative in what you send and liberal in what you accept. In terms of Python type hints, this means providing more flexible types like `abc.Mapping` or `collections.abc.Sequence` instead of concrete types like `dict` or `list`. This approach allows the caller to pass compatible objects while maintaining robustness in your function.
x??

---

#### Num_columns Calculation
Background context: The text provides a method for calculating the number of rows and columns needed to arrange elements in a grid-like structure.

:p How is the num_columns variable calculated?
??x
The `num_columns` variable is calculated as the square root of the sequence length, rounded to the nearest integer. This helps determine the dimensions of a grid that can contain all items from the sequence.
```python
num_columns = round(len(sequence) ** 0.5)
```
x??

---

#### Divmod and Bool for Rows Calculation
Background context: The text describes how `divmod` and `bool` are used to calculate the number of rows in a grid.

:p How is the num_rows variable calculated?
??x
The `num_rows` and `reminder` variables are calculated using `divmod`, which returns both the quotient and remainder. If there's any reminder, one additional row is added by checking with `bool(reminder)`.
```python
num_rows, reminder = divmod(len(sequence), num_columns)
num_rows += bool(reminder)
```
x??

---

#### Generic Mappings Example
Background context: The text provides an example of using a generic mapping to create an inverted index for Unicode character names.

:p How is the `name2hex` function used?
??x
The `name2hex` function takes a string name and a mapping (like a dictionary) from names to hexadecimal values. It returns the hexadecimal value corresponding to the given name.
```python
from collections.abc import Mapping

def name2hex(name: str, color_map: Mapping[str, int]) -> str:
    # Example implementation
    return color_map.get(name, '0x0')
```
x??

#### Floating Point Types and Type Checking
Background context: The text discusses how floating point types, including NumPy-specific types like `float32` and `longdouble`, are used alongside Python's built-in numeric types. However, these types do not define any methods, making them unsuitable for static type checking tools like Mypy.
:p What is the issue with using ABCs (Abstract Base Classes) like `Number` from the `numbers` module in static type checking?
??x
The main issue is that the `Number` ABC does not define any methods. Therefore, a static type checker would reject code operations on values inferred to be of the `Number` type because no method implementations are available. This makes the `Number` ABC useless for practical type annotations.
```python
# Example: Incorrect annotation due to lack of method definitions in Number ABC
def incorrect_sum(values: numbers.Number) -> float:
    return sum(values)
```
x??

---

#### Iterable and Function Parameters
Background context: The text explains that using `Iterable` is a good practice for function parameter type hints, as it allows flexibility without hard-coding specific types. It provides an example from the standard library's `math.fsum`.
:p What is the purpose of using `Iterable` in function parameters?
??x
Using `Iterable` in function parameters provides flexibility by allowing any iterable collection to be passed to the function, rather than restricting it to a specific type like `list`. This makes the function more versatile and easier to use with different types of iterables.
```python
# Example using Iterable as parameter type hint
from typing import Iterable

def fsum(__seq: Iterable[float]) -> float:
    # Implementation details not shown
    pass
```
x??

---

#### Type Aliases for Readability
Background context: The text introduces the use of type aliases to improve code readability. A type alias is a simple, custom name given to an existing type.
:p What is a `FromTo` type alias in the provided example?
??x
A `FromTo` type alias is defined as `tuple[str, str]`, which simplifies the function signature for the `zip_replace` function by using a descriptive name instead of repeatedly writing out `tuple[str, str]`. This improves code readability and maintainability.
```python
# Example of defining and using a FromTo type alias
from typing import tuple

FromTo = tuple[str, str]

def zip_replace(text: str, changes: Iterable[FromTo]) -> str:
    for from_, to in changes:
        text = text.replace(from_, to)
    return text
```
x??

---

#### Positional-Only Parameters
Background context: The text mentions that positional-only parameters are indicated with a double underscore `__` prefix. This is part of the PEP 484 convention for marking such parameters.
:p What does the leading underscore in function parameters signify?
??x
The leading underscores in function parameter names, such as `__seq`, indicate that these parameters are intended to be used only by their position and not by name. This helps prevent potential conflicts with variable names in the calling context.
```python
# Example of positional-only parameter usage
def fsum(__seq: Iterable[float]) -> float:
    # Implementation details not shown
    pass
```
x??

---

#### `math.fsum` Function Example
Background context: The text provides an example using the `fsum` function from Python's standard library, which accepts an iterable of floating-point numbers.
:p How does the `math.fsum` function use `Iterable` in its parameter type hint?
??x
The `math.fsum` function uses `Iterable[float]` as a type hint for its first parameter. This allows any iterable collection (like lists or tuples) that contains elements of type `float` to be passed to the function.
```python
# Example of using Iterable in the fsum function from math module
from typing import Iterable

def fsum(__seq: Iterable[float]) -> float:
    # Implementation details not shown
    pass
```
x??

---

#### `zip_replace` Function Implementation
Background context: The text describes a function called `zip_replace` that takes two parameters: a string and an iterable of tuple pairs. It iterates over the iterable, replacing occurrences in the string based on the tuples.
:p How does the `zip_replace` function replace characters in the input text?
??x
The `zip_replace` function replaces characters or substrings in the input text by iterating over the provided iterable (`changes`). For each tuple `(from_, to)`, it calls the `replace` method of the string, replacing all occurrences of `from_` with `to`.
```python
# Example implementation of zip_replace function
def zip_replace(text: str, changes: Iterable[tuple[str, str]]) -> str:
    for from_, to in changes:
        text = text.replace(from_, to)
    return text
```
x??

---

#### TypeAlias in Python 3.10

Type aliases are used to assign names to complex type expressions, making them more readable and maintainable.

:p What is a TypeAlias and when should it be used?
??x
A `TypeAlias` is introduced in Python 3.10 to provide an easier way to create type aliases. It makes the assignments that create type aliases more visible and facilitates better type checking.

Example:
```python
from typing import TypeAlias

FromTo: TypeAlias = tuple[str, str]
```
Use it when you want a clear name for a complex type expression or to improve readability of your code.
x??

---

#### Iterable vs Sequence in Python

`Iterable` and `Sequence` are both abstract base classes (ABC) from the collections.abc module.

:p What is the difference between `Iterable` and `Sequence`?
??x
- `Iterable`: A class that implements either the __iter__ method or the __getitem__ method with a stop argument.
- `Sequence`: An iterable, but also supports indexing.

Example:
```python
from collections.abc import Iterable, Sequence

# Both are iterables but not all iterables are sequences.
for item in [1, 2, 3]:
    pass  # This is an iterable

length = len([1, 2, 3])  # This requires a sequence.
```
`Iterable` is best used as a parameter type because it covers a broader range of types. `Sequence`, on the other hand, should be preferred when you need to know that the input supports indexing and slicing.
x??

---

#### Parameterized Generics and TypeVar

Generics allow creating reusable code that works with multiple concrete types.

:p What is a parameterized generic in Python?
??x
A parameterized generic is a generic type written as `list[T]` where `T` is a type variable. This allows the function to be more flexible, reflecting on the result type based on the input type.

Example:
```python
from collections.abc import Sequence
from random import shuffle
from typing import TypeVar

T = TypeVar('T')

def sample(population: Sequence[T], size: int) -> list[T]:
    if size < 1:
        raise ValueError("size must be >= 1")
    result = list(population)
    shuffle(result)
    return result[:size]
```
Here, `T` can represent any type (e.g., `int`, `str`), and the function will return a list of that same type.
x??

---

#### TypeVar in Python

Type variables allow creating more generic functions by using placeholders for concrete types.

:p Why is `TypeVar` needed?
??x
`TypeVar` is used to introduce a variable name into the current namespace, allowing you to define type parameters in function signatures. This avoids deep changes in the interpreter and allows for flexible type hints.

Example:
```python
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar('T')

def sample(population: Iterable[T], size: int) -> list[T]:
    # Implementation details...
```
Without `TypeVar`, you would have to define the type parameter directly in the signature, which could lead to issues if you need to use it multiple times.
x??

---

#### Restricted TypeVar

Restricted TypeVars can be constrained to specific types.

:p How do we create a restricted type variable?
??x
You can restrict a `TypeVar` by providing additional positional arguments. This ensures that the type parameter is consistent with the specified types.

Example:
```python
from collections.abc import Iterable
from decimal import Decimal
from fractions import Fraction
from typing import TypeVar

NumberT = TypeVar('NumberT', float, Decimal, Fraction)

def mode(data: Iterable[NumberT]) -> NumberT:
    # Implementation details...
```
Here, `NumberT` can only be one of the specified types (float, Decimal, or Fraction).
x??

---

#### Bounded TypeVar

Bounded TypeVars set an upper boundary for acceptable types.

:p How does bounded `TypeVar` work?
??x
A bounded `TypeVar` restricts the type parameter to a specific base class. This ensures that the return type is consistent with the expected behavior of the function.

Example:
```python
from collections import Counter
from collections.abc import Iterable, Hashable
from typing import TypeVar

HashableT = TypeVar('HashableT', bound=Hashable)

def mode(data: Iterable[HashableT]) -> HashableT:
    pairs = Counter(data).most_common(1)
    if len(pairs) == 0:
        raise ValueError("no mode for empty data")
    return pairs[0][0]
```
Here, `HashableT` can be any type that is hashable or a subclass of `Hashable`.
x??

---

#### AnyStr in Python

`AnyStr` is a predefined TypeVar used for functions accepting either bytes or str.

:p What is the purpose of `AnyStr`?
??x
`AnyStr` is defined as `AnyStr = TypeVar('AnyStr', bytes, str)` and is used in functions that accept either `bytes` or `str` types. It simplifies type hinting for these dual-type scenarios.

Example:
```python
from typing import AnyStr

def echo(data: AnyStr) -> AnyStr:
    return data
```
Here, the function can handle both bytes and str inputs.
x??

---

