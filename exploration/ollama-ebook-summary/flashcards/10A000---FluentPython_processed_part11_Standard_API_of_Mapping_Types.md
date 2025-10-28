# Flashcards: 10A000---FluentPython_processed (Part 11)

**Starting Chapter:** Standard API of Mapping Types

---

#### Pattern Matching with Mapping Patterns
Background context: This section explains how pattern matching works with mapping patterns, where a match can succeed even if there are extra keys that do not appear in the pattern. The order of keys in the pattern is irrelevant, and the subject must already have the required keys for a match to succeed.
:p What happens when using mapping patterns in Python's pattern matching?
??x
Mapping patterns allow partial matches on dictionaries, meaning you can match specific keys without needing all keys present in the dictionary. Extra keys are ignored unless explicitly captured by `**` (double star), which gathers unmatched key-value pairs into a dictionary. This is different from sequence patterns where every element must be matched.
```python
food = {'category': 'ice cream', 'flavor': 'vanilla', 'cost': 199}
match food:
    case {'category': 'ice cream', **details}:
        print(f'Ice cream details: {details}')
```
The output will include all extra keys and their values as a dictionary.
x??

---

#### Handling Missing Keys in Pattern Matching
Background context: In pattern matching, the `d.get(key, sentinel)` method is used to handle missing keys. This means that key lookups do not automatically create new items; instead, they return `None` or a specified sentinel value if the key is missing.
:p How does Python's pattern matching handle missing keys in mappings?
??x
Python's pattern matching uses `d.get(key, sentinel)` internally for key lookups. This method checks if the key exists and returns its value; otherwise, it returns the provided sentinel (which cannot be a value that might occur in user data). Therefore, to handle missing keys automatically, you would need to use collections like `defaultdict` where items are created on the fly.
```python
from collections import defaultdict

# Using defaultdict to handle missing keys automatically
food = defaultdict(lambda: None, {'category': 'ice cream', 'flavor': 'vanilla'})
print(food['cost'])  # Outputs: None because 'cost' is not defined in the initial dictionary
```
x??

---

#### ABCs of Mapping Types
Background context: The `collections.abc` module provides abstract base classes (ABCs) for defining interfaces that other types should implement, such as `Mapping` and `MutableMapping`. These ABCs are useful for code that needs to support a wide variety of mapping types, including dictionaries.
:p What is the purpose of the `abc.Mapping` and `abc.MutableMapping` in Python?
??x
The `abc.Mapping` and `abc.MutableMapping` ABCs from the `collections.abc` module serve as interfaces for defining mappings and mutable mappings, respectively. They document standard methods that mapping types should implement. This allows your code to check if an object is a valid mapping using `isinstance(obj, abc.Mapping)`, ensuring compatibility with various dictionary-like objects.
```python
from collections.abc import Mapping

# Checking if an object is a valid mapping
my_dict = {}
if isinstance(my_dict, Mapping):
    print("This is a valid mapping.")
```
x??

---

#### Hashable Objects in Python

Hashability is a fundamental property of objects in Python, crucial for understanding how they can be used as keys in dictionaries or elements in sets. A hashable object must have a hash code that never changes during its lifetime and can be compared to other objects.

Relevant context:
- A hashable object has two methods: `__hash__()` and `__eq__()`.
- Objects are considered hashable if they return the same hash code for equivalent states.
- Hash codes may vary between Python versions, machine architectures, and due to security salts added during computation.

:p What is a hashable object in Python?
??x
A hashable object in Python is one that can be used as a key in a dictionary or an element in a set. It must have a consistent `__hash__()` method that returns the same value for equivalent states and a `__eq__()` method to compare objects.

It’s important because:
- The hash code of a hashable object should remain constant throughout its lifetime.
- Equivalent objects (those that return true from `__eq__()`) must have the same hash code.
??x
The answer with detailed explanations. Include code examples if relevant:

```python
class ExampleClass:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        # This method should ensure that equivalent objects return the same hash code
        return hash(self.value)

    def __eq__(self, other):
        # Ensure that comparing equal objects returns True
        if isinstance(other, ExampleClass):
            return self.value == other.value
        return False

# Example usage:
obj1 = ExampleClass(10)
obj2 = ExampleClass(10)

print(hash(obj1) == hash(obj2))  # Should print True
print(obj1 == obj2)  # Should print True
```
x??

---

#### Custom Mapping Classes in Python

Custom mapping classes allow you to create custom data structures that mimic the behavior of dictionaries. These can be useful for implementing complex data handling logic.

Relevant context:
- `collections.UserDict` is a good base class for creating custom mappings.
- Wrapping a dictionary using composition can also achieve similar results without subclassing built-in types directly.
- All concrete mapping classes in Python (like `dict`, `defaultdict`, and `OrderedDict`) are based on dictionaries, making them hashable if their keys are.

:p What is the difference between extending `collections.UserDict` and subclassing a concrete dictionary class?
??x
Extending `collections.UserDict` or wrapping a dictionary by composition can be preferred over directly subclassing a concrete dictionary class. This approach allows for more flexibility in implementing custom behavior while still benefiting from the underlying hash table implementation of dictionaries.

The key difference is:
- **UserDict**: Allows you to customize the internal logic without modifying the core `dict` structure, which remains immutable and highly optimized.
- **Subclassing Concrete Dicts**: Directly subclassing a concrete dictionary might lead to unexpected behavior due to the strict contract required by these classes (e.g., ensuring keys are hashable).

Code example:
```python
from collections import UserDict

class MyCustomMapping(UserDict):
    def __missing__(self, key):
        # Custom logic when accessing a non-existent key
        return f"Key {key} not found in mapping"
```
x??

---

#### Common Mapping Methods in Python

The common methods available for mappings like `dict`, `defaultdict`, and `OrderedDict` provide a rich API for handling keys, values, and items.

Relevant context:
- These methods include operations such as adding, removing, retrieving, updating, and iterating over items.
- The `collections` module offers variations of these basic mapping types to handle specific use cases (e.g., ensuring order with `OrderedDict`, providing default values with `defaultdict`).

:p List the common methods for mappings like `dict`, `defaultdict`, and `OrderedDict`.
??x
Common methods for mappings include:

- **Adding/Removing Items**: `clear()`, `pop(k, [default])`, `popitem()` (last item if not specified).
- **Retrieving Values**: `get(k, [default])`, `__getitem__(k)`.
- **Setting Values**: `__setitem__(k, v)`.
- **Updating Items**: `update(m, [**kwargs])`.
- **Viewing Items**: `items()`, `keys()`, `values()`.

Code example:
```python
from collections import defaultdict, OrderedDict

# Example with a default dictionary
d = defaultdict(int)
d['a'] += 10
print(d)  # defaultdict(<class 'int'>, {'a': 10})

# Example with an ordered dictionary
od = OrderedDict()
od['apple'] = 5
od['banana'] = 3
print(od)  # OrderedDict([('apple', 5), ('banana', 3)])
```
x??

---

#### Hash Code Computation in Python

Hash codes are used to provide a unique integer representation for objects that can be used as dictionary keys. Understanding how hash codes are computed is crucial for ensuring consistent behavior across operations.

Relevant context:
- The `hash()` function computes the hash code of an object.
- For user-defined types, the default implementation uses the object’s ID.
- Custom types need to implement both `__hash__()` and `__eq__()` methods correctly.

:p What is a hash code in Python?
??x
A hash code in Python is a unique integer representation of an object. It is used by dictionaries and sets to quickly find or retrieve values based on their keys. The hash code must remain consistent for equivalent objects throughout their lifetime, as it affects how they are stored and accessed.

:p How does the `hash()` function work with built-in types?
??x
The `hash()` function works differently depending on the type of object:

- For immutable types like numbers (`int`, `float`), strings (`str`), bytes, and tuples where all elements are hashable: The `hash()` function directly computes a unique integer.
- For mutable collections like lists or sets: These cannot be hashed because their state can change.

Example:
```python
print(hash(123))  # Hashes an integer
# print(hash([1, 2, 3]))  # Raises TypeError as list is not hashable
```
x??

---

These flashcards provide a structured way to familiarize yourself with the concepts of hashability and common mapping methods in Python.

#### Default Factory in `defaultdict`
Background context explaining that a default factory is used to instantiate the default value for a missing key. It's not a method but a callable attribute set by the user when creating a `defaultdict`.

:p What is a default factory in `defaultdict`?
??x
A default factory in `defaultdict` is a callable object (like a function or lambda) that determines what to return when a non-existent key is accessed. When the key does not exist, Python calls this callable and uses its return value as the default value for that key.
```python
from collections import defaultdict

def create_list():
    return []

d = defaultdict(create_list)
print(d['missing_key'])  # Output: []
```
x??

---

#### OrderedDict Popitem Behavior
Background context explaining how `OrderedDict.popitem` works and its behavior with the `last` keyword. Mention that this method removes either the first or last item based on the argument.

:p What does `OrderedDict.popitem(last=False)` do?
??x
`OrderedDict.popitem(last=False)` removes the first item inserted into the dictionary (FIFO). If `last=True`, it would remove the last item, but as of Python 3.10b3, this keyword is not supported in `OrderedDict` or its parent class `dict`.

```python
from collections import OrderedDict

od = OrderedDict()
od['a'] = 1
od['b'] = 2
od['c'] = 3

# Removes the first item ('a', 1)
print(od.popitem(last=False))  # Output: ('a', 1)

# Removing the last item would not work as of Python 3.10b3
# print(od.popitem(last=True))
```
x??

---

#### `update` Method Behavior with Mapping
Background context explaining how the `update` method in dictionaries works and its use of duck typing to handle different types of input.

:p How does the `update` method work with its first argument?
??x
The `update` method checks if the first argument (`m`) has a `keys` method. If it does, `update()` assumes that `m` is a mapping and uses those keys and values to update or add new entries. Otherwise, it iterates over `m`, treating each item as a (key, value) pair.

This behavior is an example of duck typing in Python, where the method checks if the object behaves like a dictionary by having a `keys` method.
```python
d = {'a': 1}
m = {'b': 2, 'c': 3}

# Assuming m has keys()
d.update(m)
print(d)  # Output: {'a': 1, 'b': 2, 'c': 3}

# Non-mapping object treated as (key, value) pairs
l = [(4, 5), (6, 7)]
d.update(l)
print(d)  # Output: {'a': 1, 'b': 2, 'c': 3, 4: 5, 6: 7}
```
x??

---

#### `setdefault` Method in Dictionaries
Background context explaining the use of `setdefault` to avoid redundant key lookups when updating a value in place.

:p What is the purpose of `dict.setdefault`?
??x
The purpose of `dict.setdefault` is to return the value for the specified key if it exists. If not, it inserts the key with a value returned by the callable provided (or default value) and returns that value. This method avoids redundant key lookups when updating values in place.

```python
d = {}
word = 'apple'
location = (1, 2)

# Using setdefault to handle missing keys
locations = d.setdefault(word, []).append(location)
print(d[word])  # Output: [Location(1, 2)]
```
x??

---

#### Example of `dict.get` vs. `setdefault`
Background context explaining the differences between using `get` and `setdefault` for dictionary operations.

:p How does `index0.py` use `dict.get` to handle missing keys?
??x
In `index0.py`, a script is shown that uses `dict.get` to fetch and update a list of word occurrences. However, it results in redundant key lookups because the value has to be fetched again when updating.

```python
index = {}
with open('zen.txt', encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in re.finditer(r'\w+', line):
            word = match.group()
            column_no = match.start() + 1
            location = (line_no, column_no)
            
            # This is ugly; coded like this to make a point
            occurrences = index.get(word, [])
            occurrences.append(location)
            index[word] = occurrences

for word in sorted(index, key=str.upper):
    print(word, index[word])
```
x??

---

#### Example of Using `setdefault` for Better Performance
Background context explaining the improvement provided by using `setdefault` to avoid redundant key lookups.

:p How does `index.py` use `dict.setdefault` more efficiently?
??x
In `index.py`, a script is shown that uses `dict.setdefault` to fetch and update a list of word occurrences in a single line. This avoids the redundant key lookup, making the code more efficient.

```python
import re

WORD_RE = re.compile(r'\w+')
index = {}

with open('zen.txt', encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            column_no = match.start() + 1
            location = (line_no, column_no)
            
            # Using setdefault to handle missing keys and update in one go
            index.setdefault(word, []).append(location)

for word in sorted(index, key=str.upper):
    print(f'{word}: {index[word]}')
```
x??

