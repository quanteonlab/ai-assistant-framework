# Flashcards: 10A000---FluentPython_processed (Part 12)

**Starting Chapter:** Automatic Handling of Missing Keys. defaultdict Another Take on Missing Keys

---

#### Using `setdefault` for Handling Missing Keys

Background context: When you need to insert a value into a dictionary only when the key is not already present, using `dict.setdefault()` can be an efficient method. This approach minimizes the number of lookups needed compared to manually checking if the key exists and then assigning a default value.

If the key is missing, `setdefault` performs both the check and assignment in one operation: `my_dict.setdefault(key, [])`. If the key already exists, it simply returns the existing value without performing any additional operations. 

:p What does `dict.setdefault()` do when the key is not present?
??x
When the key is not present, `setdefault` inserts the key with a default value (which can be specified as an argument) and then returns that same value.
x??

---

#### Using `defaultdict` for Handling Missing Keys

Background context: If you need to provide a made-up or default value when looking up keys in a dictionary, using `collections.defaultdict` is a convenient solution. A `defaultdict` automatically creates items with a default value on demand whenever a missing key is searched.

:p How does `defaultdict` handle missing keys?
??x
`defaultdict` handles missing keys by creating the item with a default value when accessed for the first time. You specify the default factory function (e.g., `list`, `int`) that determines what the default value will be.
x??

---

#### Example of `defaultdict` in Practice

Background context: The example provided uses `defaultdict(list)` to create an index mapping words to their locations in a text file. This approach ensures that new keys are automatically initialized with an empty list, allowing values to be appended without needing to check for the key's existence first.

:p How does the code snippet initialize and use `defaultdict`?
??x
The code initializes a `defaultdict` with `list` as the default factory. It then iterates over each line of the file, finds words using regular expressions, and appends their location to the corresponding list in the dictionary.

```python
import collections
import re
import sys

# Define a regex for word matching
WORD_RE = re.compile(r'\w+')

# Create a defaultdict with `list` as default_factory
index = collections.defaultdict(list)

with open(sys.argv[1], encoding='utf-8') as fp:
    for line_no, line in enumerate(fp, 1):
        for match in WORD_RE.finditer(line):
            word = match.group()
            column_no = match.start() + 1
            location = (line_no, column_no)
            index[word].append(location)

# Display the sorted index
for word in sorted(index, key=str.upper):
    print(word, index[word])
```
x??

---

#### Difference Between `setdefault` and `defaultdict`

Background context: Both `setdefault` and `defaultdict` can handle missing keys, but they do so with different levels of complexity. `setdefault` is useful for simple cases where you need to insert a value only if the key does not exist, whereas `defaultdict` provides more flexibility by allowing the creation of items on demand based on custom logic.

:p What are the main differences between using `setdefault` and `defaultdict`?
??x
The main differences are:
- **Efficiency**: `setdefault` performs a single lookup if the key is missing, while manually checking with `if not in` might require two lookups (one for existence check and one for assignment).
- **Complexity**: `defaultdict` can handle more complex default value generation logic through its custom factory function.
x??

---

#### Default Factory and `__getitem__` Method
Default factories for data structures like defaultdict are only invoked to provide default values when using __getitem__. This method is specifically called during direct key lookups but not necessarily through other methods like get or contains.

:p What does the default factory of a defaultdict do?
??x
The default factory in a defaultdict is used to generate default values only for `__getitem__` calls, meaning it's invoked when directly accessing keys that don't exist. It has no effect on other methods such as `get` or `in`.
x??

---

#### Missing Key Handling with `__missing__`
To handle missing keys more gracefully, Python allows the definition of a custom `__missing__` method in subclasses of dict. This method is called by `__getitem__` whenever a key is not found, providing a way to return a default value without raising a KeyError.

:p How does the `__missing__` method work?
??x
The `__missing__` method allows you to define custom behavior when a key is missing from a dictionary. It is called by `__getitem__` whenever a non-existent key is accessed, but not by other methods like `get` or `in`.

For example:
```python
class CustomDict(dict):
    def __missing__(self, key):
        # Define what to do when the key is missing
        return f'Missing value for {key}'
```
x??

---

#### Converting Non-String Keys to Strings on Lookup
Sometimes it's useful to convert non-string keys into strings when looking them up. This can be achieved by implementing a custom `__missing__` method in a subclass of dict.

:p How does the `StrKeyDict0` class ensure that all keys are converted to strings?
??x
The `StrKeyDict0` class ensures that all keys accessed as strings, even if the input key is not. It checks if the key is already a string; if not, it converts it and looks up the value.

Example implementation:
```python
class StrKeyDict0(dict):
    def __missing__(self, key):
        # Check if key is a non-string type
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self.keys() or str(key) in self.keys()
```
x??

---

#### Key Test for `__contains__` Method
The `in` operator checks if a key is present in the dictionary. In custom dictionaries, you can override this behavior by implementing a custom `__contains__` method.

:p How does the `__contains__` method work in the `StrKeyDict0` class?
??x
In the `StrKeyDict0` class, the `__contains__` method checks if the key is present as-is or as a string. This ensures that both exact matches and their string representations are considered.

Example implementation:
```python
class StrKeyDict0(dict):
    def __missing__(self, key):
        # Check if key is already a string
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self.keys() or str(key) in self.keys()
```
x??

---

#### Handling Key Types with `isinstance`
In the context of the `__missing__` method, it's important to check if a key is already a string before trying to convert it. This prevents infinite recursion.

:p Why is there an `isinstance(key, str)` check in the `StrKeyDict0` class?
??x
The `isinstance(key, str)` check is necessary to avoid infinite recursion when converting non-string keys to strings. If this check is not present, trying to convert a key that does not result in an existing string could lead to the same conversion function being called repeatedly.

Example:
```python
class StrKeyDict0(dict):
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]
```
x??

---

#### StrKeyDict and __missing__
Background context: The provided text discusses a custom dictionary subclass, `StrKeyDict`, that handles string conversion for keys. This is useful to ensure consistent behavior when dealing with mixed-type keys.

:p What does the `__missing__` method do in `StrKeyDict`?
??x
The `__missing__` method is overridden to handle missing keys by converting the key to a string and then calling `__getitem__` on the dictionary. This ensures that any non-string key can be converted and searched as if it were a string.

```python
class StrKeyDict(dict):
    def __missing__(self, key):
        # Convert the key to a string and try to get its value
        return self[str(key)]
```
x??

---

#### Consistency with __contains__
Background context: The `__contains__` method is necessary for consistent behavior in checking membership. The text explains why a direct `k in d` check needs to be handled differently.

:p Why does the implementation of `__contains__` use `key in self.keys()` instead of `str(key) in self`?
??x
Using `key in self.keys()` avoids recursive calls because directly using `str(key) in self` would invoke `__contains__`, causing a potential infinite recursion. By explicitly checking keys, we bypass the attribute lookup for `.keys()`, ensuring efficient and non-recursive behavior.

```python
class StrKeyDict(dict):
    def __contains__(self, key):
        # Check if the key is in the dictionary's keys, avoiding recursive calls
        return str(key) in self.keys()
```
x??

---

#### OrderedDict Overview
Background context: The `OrderedDict` class was designed to maintain the order of items inserted. While Python 3.6+ built-in `dict` also maintains insertion order, `OrderedDict` is still useful for backward compatibility and specific functionalities.

:p What are some key differences between `OrderedDict` and regular `dict` as mentioned in the text?
??x
Key differences include:
- The equality operation for `OrderedDict` checks matching order.
- The `popitem()` method of `OrderedDict` has a different signature, allowing specification of which item is popped.
- `OrderedDict` includes an efficient `move_to_end()` method to reposition elements.

These features make `OrderedDict` suitable for scenarios requiring frequent reordering operations or maintaining the insertion order strictly.

x??

---

#### ChainMap Overview
Background context: A `ChainMap` instance combines multiple mappings, allowing a unified view of them. The lookup process searches through each mapping in the order provided until it finds the key.

:p What does `ChainMap` do and how is it used?
??x
A `ChainMap` combines multiple dictionaries or other mappings into a single unit. It allows for efficient lookups by searching each input mapping in the specified order until the key is found. This can be particularly useful in scenarios where you need to simulate nested scope access, such as variable lookup in programming languages.

```python
from collections import ChainMap

d1 = dict(a=1, b=3)
d2 = dict(a=2, b=4, c=6)

chain = ChainMap(d1, d2)
print(chain['a'])  # Output: 1
print(chain['c'])  # Output: 6
```
x??

---

#### Counter Overview
Background context: A `Counter` is a dictionary subclass for counting hashable objects. It stores elements as dictionary keys and their counts as dictionary values.

:p What does the `Counter` class do?
??x
The `Counter` class is used to count the occurrences of hashable objects in an iterable. It returns a dictionary-like object where each key is an element from the input, and its value is the number of times it appears.

```python
from collections import Counter

c = Counter('abcdaab')
print(c)  # Output: Counter({'a': 3, 'b': 2, 'c': 1, 'd': 1})
```
x??

#### Counter in Python
Background context: `collections.Counter` is a convenient way to count occurrences of hashable objects. It can be used as a multiset, where keys represent elements and their counts are the number of occurrences.

:p How does `Counter` work in counting instances of hashable objects?
??x
The `Counter` class from the `collections` module helps you to count the frequency of each element in an iterable. Each key in the `Counter` object represents a unique item, and its value is the count of that item.

```python
from collections import Counter

# Example usage: counting letters in words
ct = Counter('abracadabra')
print(ct)  # Output: Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

# Updating the counter with another string
ct.update('aaaaazzz')
print(ct)  # Output: Counter({'a': 10, 'z': 3, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
```
x??

---

#### Using Counter to Find the Most Common Elements
Background context: The `Counter` class provides methods like `most_common`, which returns a list of tuples containing the most common elements and their counts.

:p How can you use `Counter.most_common` to find the three most frequent items?
??x
You can use the `most_common` method on a `Counter` object to get an ordered list of the n most common items along with their counts. For example, if you want to find the 3 most common elements:

```python
# Example usage: finding the 3 most common elements
ct = Counter('abracadabra')
print(ct.most_common(3))  # Output: [('a', 5), ('b', 2), ('r', 2)]
```

This method is particularly useful for data analysis and can provide insights into the frequency distribution of items in a dataset.
x??

---

#### Shelve Module Overview
Background context: The `shelve` module allows you to store mappings (string keys and Python objects) persistently, using the pickle format. This is similar to a simple key-value database.

:p What does the `shelve` module provide for persistent storage?
??x
The `shelve` module in Python's standard library provides persistent storage of mappings from string keys to Python objects serialized with the `pickle` module. It allows you to save and load data as if it were a dictionary, but persistently on disk.

A `Shelf` object is created using `shelve.open()`, which returns an instance that behaves like a dictionary, storing key-value pairs in a file on disk. Keys must be strings, and values can be any Python objects that are pickleable.

Example usage:

```python
import shelve

# Open the shelf database (creates it if doesn't exist)
shelf_db = shelve.open('example.db')

# Store some data
shelf_db['key1'] = 'value1'
shelf_db['key2'] = 42

# Close the shelf when done
shelf_db.close()
```

You can use a `with` statement to ensure proper handling:

```python
with shelve.open('example.db') as shelf:
    shelf['key3'] = [1, 2, 3]
```
x??

---

#### Subclassing UserDict for Custom Mapping Types
Background context: The `UserDict` class is designed to be subclassed when you need a custom mapping type. It provides methods that are useful for implementing custom dictionary-like objects.

:p Why should you prefer subclassing `UserDict` over `dict` when creating a custom mapping?
??x
You should prefer subclassing `collections.UserDict` over directly using the built-in `dict` class because `UserDict` is designed to be extended. It has an internal dictionary that it manages, and methods like `__setitem__`, `__getitem__`, etc., can be overridden more easily without needing to deal with the internal implementation details of the built-in `dict`.

Here's a simple example of subclassing `UserDict`:

```python
from collections import UserDict

class StrKeyDict(UserDict):
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def __contains__(self, key):
        return key in self.data or str(key) in self.data

    def __setitem__(self, key, value):
        self.data[str(key)] = value
```

This class ensures that keys are always stored as strings.
x??

---

#### StrKeyDict Class Definition
This section introduces a custom dictionary subclass that handles non-string keys by converting them to strings. This helps avoid issues with non-string data types and provides more flexibility.

:p What is the purpose of the `StrKeyDict` class?
??x
The purpose of `StrKeyDict` is to create a dictionary where all keys are stored as strings, even if non-string keys are used during insertion or updates. This avoids unexpected behavior when dealing with mixed key types.
??x

---

#### __missing__ Method Implementation
The `__missing__` method is overridden in `StrKeyDict` to handle missing keys. It checks if the given key can be converted to a string and raises an error if it cannot.

:p How does the `__missing__` method work?
??x
The `__missing__` method works by checking if the provided key is of type `str`. If not, it raises a `KeyError`. Otherwise, it converts the key to a string and looks up the value in the dictionary. Here's how it might be implemented:

```python
class StrKeyDict(collections.UserDict):
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]
```
??x

---

#### __contains__ Method Implementation
The `__contains__` method checks if a given key is in the dictionary. It assumes that all keys are stored as strings and uses `self.data` for lookups.

:p How does the `__contains__` method simplify key checking?
??x
The `__contains__` method simplifies key checking by converting the provided key to a string before performing the lookup. This ensures that the dictionary can handle different types of keys without additional logic, making it more robust and easier to use.

```python
def __contains__(self, key):
    return str(key) in self.data
```
??x

---

#### __setitem__ Method Implementation
The `__setitem__` method converts any key to a string before storing it. This ensures that all keys are stored as strings, providing consistency.

:p How does the `__setitem__` method handle non-string keys?
??x
The `__setitem__` method handles non-string keys by converting them to strings before adding them to the dictionary. It uses `self.data[str(key)] = item` to store the value associated with the key, ensuring all keys are always of type `str`.

```python
def __setitem__(self, key, item):
    self.data[str(key)] = item
```
??x

---

#### Inheritance from UserDict
The `StrKeyDict` class inherits from `collections.UserDict`, which provides many useful methods and ensures that the dictionary behaves like a standard Python mapping.

:p What does inheriting from `UserDict` provide to `StrKeyDict`?
??x
Inheriting from `UserDict` provides `StrKeyDict` with the functionality of a full-fledged dictionary, including methods for updating, checking containment, and retrieving items. This abstraction simplifies coding by leveraging existing implementations.

```python
class StrKeyDict(collections.UserDict):
    pass  # Other methods go here
```
??x

---

#### Usage Examples
The provided `StrKeyDict` class can be used to create a dictionary that automatically converts non-string keys to strings, ensuring consistent key handling.

:p How might you use the `StrKeyDict` in practice?
??x
You can use the `StrKeyDict` by creating an instance and adding items with various types of keys. The keys will always be converted to strings internally:

```python
d = StrKeyDict()
d[1] = 'one'
print(d['1'])  # Outputs: one
```
??x

---

