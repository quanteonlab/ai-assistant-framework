# Flashcards: 10A000---FluentPython_processed (Part 13)

**Starting Chapter:** Dictionary views

---

#### Immutable Mappings
Background context explaining that mutable mapping types are provided by the standard library, but sometimes you need to prevent users from accidentally changing a mapping. A concrete use case involves hardware programming libraries like Pingo, where a board's pin representation should not be altered via software.

The `MappingProxyType` class from the `types` module can create read-only views of dictionaries, ensuring that changes cannot be made through it but reflecting any updates to the original dictionary.

:p How does `MappingProxyType` work in creating immutable mappings?
??x
`MappingProxyType` creates a read-only view of an existing mapping. You can access items from the mapped object, but you cannot modify them directly through this proxy. Any changes made to the original mapping are reflected in the proxy.

Example code:
```python
from types import MappingProxyType

d = {1: 'A'}
d_proxy = MappingProxyType(d)
print(d_proxy[1])  # Output: A
try:
    d_proxy[2] = 'x'  # This will raise a TypeError
except TypeError as e:
    print(e)  # Output: 'mappingproxy' object does not support item assignment

# Direct modification of the original dictionary is allowed
d[2] = 'B'
print(d_proxy)  # Output: mappingproxy({1: 'A', 2: 'B'})
```
x??

---
#### Dictionary Views
Background context explaining that `dict.keys()`, `dict.values()`, and `dict.items()` return views of the dictionary's keys, values, and items respectively. These views are read-only projections and provide efficient operations without unnecessary data duplication.

:p What are dictionary views used for?
??x
Dictionary views are used to perform high-performance operations on dictionaries without duplicating data. They offer a dynamic way to access parts of a dictionary, such as its keys, values, or items, and can reflect changes in the original dictionary dynamically.

Example code:
```python
d = dict(a=10, b=20, c=30)
values = d.values()
print(values)  # Output: dict_values([10, 20, 30])
print(len(values))  # Output: 3
print(list(values))  # Output: [10, 20, 30]
reversed_values = reversed(values)
print(reversed_values)  # Output: <dict_reversevalueiterator object at ...>
# Trying to subscript the view will raise an error
try:
    print(values[0])
except TypeError as e:
    print(e)  # Output: 'dict_values' object is not subscriptable
```
x??

---
#### Example of Using `MappingProxyType` in a Hardware Programming Scenario
Background context explaining how you can use `MappingProxyType` to create read-only mappings that reflect changes in the original dictionary dynamically. This is particularly useful in scenarios where hardware programming libraries need to ensure the integrity of pin configurations.

:p How could `MappingProxyType` be used in a hardware programming scenario?
??x
In a hardware programming scenario, you can use `MappingProxyType` to create read-only views of mappings that represent physical pins on a device. This prevents accidental modifications by clients of the API while still allowing them to interact with the dictionary.

For example, if you have a subclass of a board class that manages pin objects:
```python
from types import MappingProxyType

class Board:
    def __init__(self):
        self._pins = {1: 'A', 2: 'B'}
    
    @property
    def pins(self):
        return MappingProxyType(self._pins)

board = Board()
print(board.pins)  # Output: mappingproxy({1: 'A', 2: 'B'})
# Direct modifications to the original dictionary through `self._pins` are allowed
board._pins[3] = 'C'
print(board.pins)  # Output: mappingproxy({1: 'A', 2: 'B', 3: 'C'})
```
x??

---
#### Explanation of Dictionary Views Operations
Background context explaining the operations that can be performed on dictionary views, such as querying length and converting to lists.

:p What operations are supported by dictionary views?
??x
Dictionary views support several operations including:
- Querying the length using `len()`.
- Iterating over the items.
- Reversing the iteration using `reversed()`.

Example code demonstrating these operations:
```python
d = dict(a=10, b=20, c=30)
values = d.values()
print(len(values))  # Output: 3
print(list(values))  # Output: [10, 20, 30]
reversed_values = reversed(values)
print(reversed_values)  # Output: <dict_reversevalueiterator object at ...>
```
x??

---

#### Dictionary Views (dict_keys, dict_values, dict_items)
Background context: Dictionary views provide a dynamic view on dictionary contents. They are used to access values or keys directly and support operations like iteration and membership testing.

Dict\_values is the simplest dictionary view, implementing only `__len__`, `__iter__`, and `__reversed__`. Dict\_keys and dict\_items implement several set methods similar to the frozenset class. These views are not directly instantiable via code but can be obtained using `.keys()`, `.values()`, or `.items()` on a dictionary.

:p How do you create a `dict_values` object in Python?
??x
You cannot directly instantiate `dict_values`. Instead, use the `.values()` method of a dictionary.
```python
# Example
my_dict = {'a': 1, 'b': 2, 'c': 3}
values_view = my_dict.values()
```
x??

---

#### Memory Optimization in Dictionaries (PEP 412 - Key-Sharing Dictionary)
Background context: Dictionaries store key-value pairs using a hash table. This implementation allows for fast access but requires additional memory to maintain efficiency, typically keeping about one-third of the entries empty.

Since Python 3.3, PEP 412 introduced key-sharing optimizations in dictionaries. When a new instance of a class is created with the same attributes as an existing instance when `__init__` returns, both instances can share the same hash table for their `__dict__`, reducing memory usage.

:p How does Python optimize dictionary memory usage?
??x
Python optimizes dictionary memory usage by sharing common hash tables among instances of a class that have the same attribute names. This is achieved through PEP 412, which allows instances to share the same internal data structure for their `__dict__` attributes.
```python
class MyClass:
    def __init__(self):
        pass

# First instance creation
instance_1 = MyClass()

# Second instance creation with the same attribute names as the first instance
instance_2 = MyClass()  # shares the hash table of `__dict__`
```
x??

---

#### Set Operations (set and frozenset)
Background context: Sets are collections of unique elements. They support operations like union, intersection, difference, and symmetric difference through infix operators.

Set elements must be hashable; sets themselves are not hashable but frozensets are.

:p What is a set in Python?
??x
A set in Python is a collection of unique objects. It can perform various set operations such as union, intersection, difference, and symmetric difference.
```python
# Example
set1 = {'spam', 'eggs', 'bacon'}
set2 = {1, 2, 3}
union_set = set1 | set2  # Union operation
intersection_set = set1 & set2  # Intersection operation
difference_set = set1 - set2  # Difference operation
symmetric_difference_set = set1 ^ set2  # Symmetric difference operation
```
x??

---

#### Set Operations with Examples
Background context: Sets can be used to remove duplicates and perform various operations efficiently. Using sets can reduce line count, execution time, and make code easier to read.

:p How can you use sets to find the intersection between two sets?
??x
You can use the `&` operator to find the intersection between two sets.
```python
# Example
haystack = {'a', 'b', 'c', 'd'}
needles = {'c', 'd', 'e', 'f'}
found = len(haystack & needles)  # Counts how many elements in `needles` are also in `haystack`
```
x??

---

#### Set and Frozenset Basics
Background context: Sets and frozensets are built-in Python data structures used to store collections of unique, unordered elements. They provide efficient membership testing via hash tables. A set is mutable (elements can be added or removed), while a frozenset is immutable.

:p What are the main differences between sets and frozensets?
??x
Sets in Python are mutable collections that allow duplicate elements but maintain only one instance of each element due to their underlying hash table implementation. Frozensets, on the other hand, are immutable versions of sets; they cannot be modified once created, which makes them suitable for use as keys in dictionaries or as members of sets.

Frozensets can be used as keys because Python requires that dictionary keys must be immutable.
x??

---
#### Set Literals and Syntax
Background context: Set literals provide a concise way to define sets. In Python 3, set literals follow the syntax `{...}` for creating an unordered collection of unique elements. The empty set is represented by `set()`.

:p How do you create an empty set in Python?
??x
To create an empty set in Python, use `set()`. Writing `{}` creates an empty dictionary instead.
x??

---
#### Set and Frozenset Operations
Background context: Sets offer a rich API for performing operations such as union, intersection, difference, and symmetric_difference. These operations can be performed using methods or the corresponding operators.

:p How do you find the intersection of two sets in Python?
??x
You can use the `&` operator or the `.intersection()` method to find the common elements between two sets.
```python
# Using & operator
set1 = {1, 2, 3}
set2 = {3, 4, 5}
common_elements = set1 & set2

# Using intersection() method
common_elements_method = set1.intersection(set2)
```
x??

---
#### Set Comprehensions
Background context: Python introduced set comprehensions in version 2.7 to provide a concise way of constructing sets from iterable objects, similar to list and dictionary comprehensions.

:p What is a set comprehension?
??x
A set comprehension is a concise syntax for creating a set by iterating over an iterable object and applying a condition or transformation to each element.
```python
# Example of a set comprehension
numbers = [1, 2, 3, 4, 5]
squares = {x**2 for x in numbers}
```
x??

---
#### Membership Testing with Sets
Background context: Sets provide an efficient way to perform membership testing (checking if an element is part of the collection) due to their underlying hash table implementation.

:p How can you count occurrences of elements from one iterable in another using sets?
??x
To count occurrences, convert both iterables into sets and use intersection or `&` operator. For example:
```python
# Example 1: Using intersection
needles = [1, 2, 3]
haystack = [1, 2, 3, 4, 5]
found = len(set(needles) & set(haystack))

# Example 2: Using .intersection() method
found = len(set(needles).intersection(haystack))
```
x??

---

#### Unicode Character Search
Background context: The provided Python script demonstrates how to search for characters that have a specific word, such as "SIGN", in their Unicode names. This involves understanding character encoding and using Python's `unicodedata` module.

:p How can you use Python to find characters with the word "SIGN" in their Unicode names?
??x
You can use the `unicodedata.name()` function from the `unicodedata` module to check if a given character has the specific word in its name. Then, filter and collect these characters into a set.

```python
from unicodedata import name

# Find characters with 'SIGN' in their names within ASCII range (32-255)
characters = {chr(i) for i in range(32, 256) if 'SIGN' in name(chr(i), '')}
characters
```
x??

---

#### Sets Overview and Characteristics
Background context: The text describes the behavior and characteristics of Python sets, emphasizing their implementation as hash tables. This includes understanding how elements are added, ordered, and tested for membership.

:p What are some key characteristics of sets in Python?
??x
Sets in Python are implemented using a hash table, which makes them efficient for operations like membership testing but less so for maintaining order or accessing elements by index. They must contain hashable objects (with proper `__hash__` and `__eq__` methods) and ensure that each element is unique.

:p How do sets handle resizing when they become more than ⅔ full?
??x
When a set exceeds ⅔ of its capacity, Python may resize the underlying hash table to maintain efficiency. This involves reinserting elements in potentially new positions, which can change their order.

```python
# Example of a set changing order due to resizing
set1 = {1, 2, 3, 4}
print(set1)  # Initial set

# Adding more elements
for i in range(5, 10):
    set1.add(i)
    
print(set1)  # New set after adding more elements and possible resizing
```
x??

---

#### Set Operations Overview
Background context: The text explains the various operations that can be performed on sets, including union, intersection, difference, symmetric difference, and others. These operations are essential for manipulating and comparing collections of items.

:p What is the `union` operation in set theory and how is it implemented in Python?
??x
The `union` operation combines elements from multiple sets without duplicates. In Python, you can use the `|` operator or the `update()` method to achieve this.

```python
# Example of using union
set_a = {1, 2, 3}
set_b = {3, 4, 5}
union_set = set_a | set_b

print(union_set)  # Output: {1, 2, 3, 4, 5}
```
x??

---

#### In-place Operations on Sets
Background context: The text mentions that some operations (like `difference_update` and `intersection_update`) modify the target set in place. These operations are not available for immutable sets like `frozenset`.

:p What is an example of an in-place operation on a mutable set?
??x
An example of an in-place operation on a mutable set would be removing elements that are present in another set using `difference_update()`.

```python
# Example of difference_update
set_a = {1, 2, 3, 4}
set_b = {2, 4}

# Remove elements from set_a that are also in set_b
set_a.difference_update(set_b)

print(set_a)  # Output: {1, 3}
```
x??

---

#### Frozensets and In-place Operations
Background context: The text differentiates between mutable sets and immutable `frozenset` objects. While `frozenset` cannot be modified in place, it provides a hash value that can be used as a dictionary key.

:p How do you create an immutable set (frozenset) in Python?
??x
You can create an immutable set using the `frozenset()` constructor or directly by passing an iterable to it.

```python
# Example of creating a frozenset
immutable_set = frozenset([1, 2, 3])
print(immutable_set)  # Output: frozenset({1, 2, 3})
```
x??

---

#### Set Operations with Multiple Iterables
Background context: The text explains that set operations can take multiple iterables as arguments. It also mentions a syntax introduced in Python 3.5 for creating sets from multiple iterables.

:p How can you use the `{*iterable}` unpacking syntax to create a union of multiple collections?
??x
You can use the `*` operator before an iterable within set literals or function calls to unpack its elements into separate arguments, effectively creating a union of multiple collections and returning a new set.

```python
# Example of using * for set creation from multiple iterables
a = {1, 2}
b = {3, 4}
c = {5, 6}

union_set = {*a, *b, *c}
print(union_set)  # Output: {1, 2, 3, 4, 5, 6}
```
x??

---

#### Intersection of Sets
Sets theory provides a way to find common elements between two or more sets. This operation is denoted by \( S \cap Z \) and can be performed using Python's set operators.

:p What is the method to perform intersection on two mutable sets `s` and `z`?
??x
The intersection of two sets can be obtained in several ways:

1. **Using the `&` operator:**
   - Example: `s & z`
   - This returns a new set containing elements that are common to both `s` and `z`.

2. **Using the `__and__()` method:**
   - Example: `s.__and__(z)` or `z.__rand__(s)`
   - These methods provide the same functionality as using the `&` operator.

3. **Using the `intersection()` method:**
   - Example: `s.intersection(z)`
   - This returns a new set with elements common to both sets, updating no existing sets in place.

4. **In-place update using `__iand__`:**
   - Example: `s &= z`
   - This updates `s` to contain only the common elements of itself and `z`.

5. **Updating the target set with multiple iterables using `intersection_update()`:**
   - Example: `s.intersection_update(it)`
   - This method updates `s` by removing any elements not present in all sets from the iterable `it`.

??x
To illustrate how to use these methods, consider the following example:
```python
# Example usage of intersection and its methods
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Using & operator
result = set1 & set2
print(result)  # Output: {3, 4}

# Using __and__ method
result = set1.__and__(set2)
print(result)  # Output: {3, 4}

# Using intersection() method
result = set1.intersection(set2)
print(result)  # Output: {3, 4}

# In-place update using &= operator
set1 &= set2
print(set1)  # Output: {3, 4}

# Updating with multiple iterables
it = [7, 8]
set1.intersection_update(it)
print(set1)  # Output: set()
```
x??
---

