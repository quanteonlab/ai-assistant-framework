# Flashcards: 10A000---FluentPython_processed (Part 2)

**Starting Chapter:** 1. The Python Data Model

---

#### Python Data Model Introduction
Python's consistency is one of its best qualities. After working with Python, you can make informed guesses about new features due to this consistency. However, if you come from another object-oriented language like Java or C++, certain syntax may seem odd at first.

For example, in Python, you use `len(collection)` instead of `collection.len()`. This is because the underlying mechanism that makes this work is part of the Python Data Model.

:p What is the key difference between using `len(collection)` and a method like `collection.len()` in Python?
??x
The key difference lies in how Python handles built-in functions and methods. In Python, `len(collection)` calls the special method `__len__` on the collection object internally. This means that to support functionality such as getting the length of an object, you need to define a custom implementation for the `__len__` method.

```python
class MyCollection:
    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)
```

In this example, when you use `len(my_collection)`, Python internally calls `my_collection.__len__()`. This allows your custom class to play well with built-in functions and syntax.

x??

---
#### Special Method Names
Special methods in Python are named with leading and trailing double underscores. These methods are called "magic" or "dunder" methods (derived from "double underscore").

:p Why do special method names have double underscores?
??x
Special method names have double underscores to distinguish them from regular instance variables and methods. This naming convention helps prevent collisions between user-defined attributes and built-in functionality.

For example, if you define a `len` attribute in your class:

```python
class MyObject:
    def __init__(self):
        self.len = 10

obj = MyObject()
print(len(obj))  # Raises TypeError: object of type 'MyObject' has no len()
```

If Python had used double underscores, the `len` method would have been named `__len__`, and this conflict could be avoided.

x??

---
#### Example: Implementing __getitem__
The `__getitem__` method is a special method that allows objects to support subscription operations like `obj[key]`.

:p How does Python interpret `my_collection[key]`?
??x
Python interprets `my_collection[key]` by calling the `__getitem__` method on the `my_collection` object with `key` as an argument.

```python
class MyCollection:
    def __init__(self):
        self.items = {}

    def __getitem__(self, key):
        return self.items.get(key, None)

# Usage
collection = MyCollection()
collection['item'] = 'value'
print(collection['item'])  # Prints: value

# Internally, Python calls:
# my_collection.__getitem__('item')
```

When you use `my_collection[key]`, the Python interpreter internally invokes `my_collection.__getitem__(key)`.

x??

---
#### Iteration Using __iter__ and __next__
To support iteration (including asynchronous iteration), a class must implement special methods like `__iter__` and `__next__`.

:p What is required to make an object iterable in Python?
??x
To make an object iterable, you need to define the `__iter__` method that returns an iterator. The iterator should have a `__next__` method (or `__anext__` for asynchronous iteration) that provides the next item when called.

```python
class MyIterator:
    def __init__(self, data):
        self.index = 0
        self.data = data

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        item = self.data[self.index]
        self.index += 1
        return item

# Usage
it = MyIterator([1, 2, 3])
print(next(it))  # Prints: 1
print(next(it))  # Prints: 2
```

When you create an iterator using `iter(MyIterator([1, 2, 3]))`, it returns the object itself. When you call `next()` on this iterator, it invokes the `__next__` method.

x??

---
#### Operator Overloading with __add__
Operator overloading allows you to define custom behavior for operators such as `+`.

:p How does Python handle the addition of two objects using `__add__`?
??x
Python uses the `__add__` special method to define how instances of your class should behave when the `+` operator is used.

```python
class MyNumber:
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        if isinstance(other, MyNumber):
            return MyNumber(self.value + other.value)
        else:
            raise TypeError("Can only add two MyNumbers")

# Usage
num1 = MyNumber(5)
num2 = MyNumber(3)
result = num1 + num2  # result is an instance of MyNumber with value 8

# Internally, Python calls:
# num1.__add__(num2)
```

In this example, when you use `num1 + num2`, Python internally invokes `num1.__add__(num2)`. This allows you to define custom behavior for the addition operation.

x??

---
#### String Representation with __repr__ and __str__
The `__repr__` method is used for creating an unambiguous representation of objects, while `__str__` provides a user-friendly string representation.

:p What are the differences between `__repr__` and `__str__`?
??x
`__repr__` should return a string that could be used to recreate the object. It’s often used in development or debugging.

```python
class MyObject:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"MyObject({self.value})"

# Usage
obj = MyObject(10)
print(repr(obj))  # Prints: MyObject(10)

# Internally, Python calls:
# obj.__repr__()
```

`__str__` should provide a user-friendly string representation. This is often used for displaying the object in a human-readable form.

```python
class MyObject:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Value: {self.value}"

# Usage
obj = MyObject(10)
print(str(obj))  # Prints: Value: 10

# Internally, Python calls:
# obj.__str__()
```

If both `__repr__` and `__str__` are defined, the `__str__` method is preferred for string representation when using `print()` or similar functions.

x??

---

#### Special Methods and Dunder Naming Conventions

Background context explaining special methods and their naming conventions. The term "magic method" is slang for special method, but we refer to them as "dunder" methods due to the double underscores before and after the method name (e.g., `__getitem__`). These methods are documented in Python's official documentation and provide a way to customize class behavior.

If applicable, add code examples with explanations:
```python
class FrenchDeck:
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in Card.suits
                       for rank in Card.ranks]
```
:p What are dunder methods and how do they differ from regular methods?
??x
Dunder methods (double underscore methods like `__getitem__`) allow you to customize class behavior by implementing specific functionalities. They provide a way to extend the core language features of Python, such as iteration, slicing, and comparison, without reinventing them.
x??

---

#### Implementing __len__ Method

Background context explaining how the `__len__` method can be used to return the length of an object, making it behave like any standard Python collection. The `__len__` method should return an integer representing the number of items in the container.

:p What does the `__len__` method do and why is it important?
??x
The `__len__` method returns the number of items in a collection, allowing the object to behave like any standard Python collection. It's crucial for making your class interoperable with functions that expect collections, such as `len()`.
x??

---

#### Implementing __getitem__ Method

Background context explaining how the `__getitem__` method allows you to access elements of an object using indexing. The method should take a single argument (the index) and return the item at that index.

:p How does the `__getitem__` method work, and what is its significance?
??x
The `__getitem__` method enables you to use indexing to access elements of your class. It's significant because it allows your object to support slicing, iteration, and other collection-like behaviors. For example:
```python
class FrenchDeck:
    def __getitem__(self, position):
        return self._cards[position]
```
x??

---

#### Using Collections API with FrenchDeck

Background context explaining how the `collections` module provides tools for working with collections, including the `abc.Collection` abstract base class introduced in Python 3.6. This class can be used to check if an object is a collection.

:p How does using `abc.Collection` help with checking if an object is a collection?
??x
Using `abc.Collection` from the `collections.abc` module helps you check whether your custom class conforms to the expected behavior of collections, such as being iterable and supporting slicing and indexing. For example:
```python
from collections.abc import Collection

deck = FrenchDeck()
isinstance(deck, Collection)  # Returns True if deck is a collection
```
x??

---

#### Random Choice with Python's Standard Library

Background context explaining how the `random.choice` function can be used to select a random element from a sequence. The `__getitem__` method in the `FrenchDeck` class allows using this function on instances of the class.

:p How does the `random.choice` function work, and why is it useful with the `FrenchDeck` class?
??x
The `random.choice` function selects a random element from a non-empty sequence. It's useful with the `FrenchDeck` class because implementing `__getitem__` allows you to use this function directly on instances of your class. For example:
```python
from random import choice

deck = FrenchDeck()
choice(deck)  # Returns a random card from the deck
```
x??

---

#### Slicing and Iteration with Special Methods

Background context explaining how implementing `__getitem__` makes an object iterable, supports slicing, and allows for other collection-like behaviors.

:p What are some benefits of implementing `__getitem__`, and how does it enable iteration and slicing?
??x
Implementing `__getitem__` enables your class to behave like a standard Python sequence. This means you can:
- Use indexing: `deck[0]`
- Support slicing: `deck[:3]` or `deck[12::13]`
- Be iterable: `for card in deck:`

For example, implementing `__getitem__` allows the `FrenchDeck` class to automatically support these behaviors.
x??

---

#### Sorting with Custom Functions

Background context explaining how you can sort collections using custom comparison functions. The `spades_high` function demonstrates a way to rank cards by their value and suit.

:p How does the `spades_high` function help in sorting card decks?
??x
The `spades_high` function helps sort card decks by first ranking them based on their rank (with aces being highest) and then by suit (spades, hearts, diamonds, clubs). This custom ranking allows you to sort the deck using Python's built-in `sorted()` function. For example:
```python
def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

for card in sorted(deck, key=spades_high):
    print(card)
```
x??

---

#### Inheritance and Composition

Background context explaining how the `FrenchDeck` class leverages composition to delegate work to a list object (`self._cards`). It also highlights that most of its functionality is not inherited but comes from special methods.

:p How does the `FrenchDeck` class leverage composition, and why is this approach beneficial?
??x
The `FrenchDeck` class uses composition by delegating much of its functionality to an internal list (`self._cards`). This means it can benefit from core language features (like iteration) and standard library functions without having to implement all the underlying logic. For example:
```python
class FrenchDeck:
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in Card.suits
                       for rank in Card.ranks]
    
    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]
```
x??

---

#### Special Methods Overview
Special methods are called by Python’s interpreter and not directly by you. They provide a way to emulate built-in types or implement custom behavior for operations such as addition, length calculation, etc.

:p What is a special method?
??x
A special method is a method in Python that has a double underscore at the beginning and end of its name (e.g., `__len__`). These methods are called by the interpreter when you use specific operators or functions, like `+`, `len()`, etc. Implementing these methods allows your custom objects to behave more like built-in types.

```python
class Example:
    def __len__(self):
        # This method is called when len(Example()) is invoked.
        pass
```
x??

---

#### Vector Addition and Special Methods
Special methods can be used to emulate numeric operations, such as addition. For vectors in a 2D space, we can use the `__add__` special method.

:p How do you implement vector addition using special methods?
??x
To implement vector addition, you can define the `__add__` special method in your class. This method will be called when the `+` operator is used on an instance of your class.

```python
class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        # Add corresponding elements and return a new Vector.
        return Vector(self.x + other.x, self.y + other.y)
```
x??

---

#### Absolute Value Calculation with Special Methods
The `__abs__` special method is used to calculate the magnitude of vectors or complex numbers.

:p How do you implement the absolute value for a vector?
??x
To calculate the magnitude (absolute value) of a vector, you can define the `__abs__` special method. This method returns the Euclidean distance from the origin in n-dimensional space.

```python
import math

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __abs__(self):
        # Calculate the magnitude using Pythagoras' theorem.
        return math.hypot(self.x, self.y)
```
x??

---

#### String Representation with Special Methods
The `__repr__` special method provides a string representation of an object. This is useful for debugging and logging.

:p How do you implement the string representation for a vector?
??x
To provide a string representation for a vector, you can define the `__repr__` special method. This method returns a string that describes how to recreate the object.

```python
class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __repr__(self):
        # Return a string representation of the vector.
        return f'Vector( {self.x}, {self.y} )'
```
x??

---

#### Scalar Multiplication with Special Methods
Scalar multiplication can be implemented using the `__mul__` special method.

:p How do you implement scalar multiplication for a vector?
??x
To perform scalar multiplication, you can define the `__mul__` special method. This method takes another object and returns a new vector that is scaled by this factor.

```python
class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __mul__(self, scalar):
        # Multiply each component of the vector by the scalar.
        return Vector(self.x * scalar, self.y * scalar)
```
x??

---

