# Flashcards: 10A000---FluentPython_processed (Part 49)

**Starting Chapter:** Further Reading

---

#### Generators and Custom Formats
Background context explaining how generators support custom formats. Emphasize that this is to fulfill the promise of a Vector class that can do everything a Vector2d did, and more. This involves understanding standard Python objects' behaviors.

:p What are generators used for in the context of creating a Vector class?
??x
Generators are used to implement custom formatting methods (like `__format__`) in a Vector class to support a custom format beyond what Vector2d can do.
??x

---

#### Infix Operators and OO Design
Background context explaining that infix operators will be implemented on the Vector class, making it behave more like standard Python objects. This is part of achieving a "Pythonic" look-and-feel.

:p What does implementing infix operators mean for the Vector class?
??x
Implementing infix operators means adding methods to make operations (like addition or multiplication) between two Vectors readable and intuitive using operator syntax, e.g., `vec1 + vec2` instead of `operator.add(vec1, vec2)`.

```python
class Vector:
    def __add__(self, other):
        # Implementation details for vector addition
```
??x

---

#### Interfaces and Inheritance
Background context explaining that before diving into operator overloading, the text will revisit organizing multiple classes with interfaces and inheritance. Reference Chapters 13 and 14.

:p What does revisiting interfaces and inheritance imply in this context?
??x
Revisiting interfaces and inheritance suggests exploring how different classes can interact or inherit from one another to build a more complex system, such as the Vector class.
??x

---

#### Further Reading on Special Methods
Background context explaining that special methods (like `__format__` and `__getitem__`) will be referenced in further reading. Note the relevance of these methods in the Vector example.

:p What are some additional resources for understanding the special methods used in the Vector class?
??x
Additional resources include "Further Reading" which references the Vector2d example from Chapter 11, as well as information on `__index__` and its use with `__getitem__` methods.
??x

---

#### New Features in Python 2.5
Background context explaining that new features like `__index__` are introduced to support `__getitem__` methods. Reference "What's New in Python 2.5" for more details.

:p What is the significance of the `__index__` method?
??x
The `__index__` method allows objects to be used in contexts where an integer index is expected, such as slicing operations.
??x

---

#### Protocols and Informal Interfaces
Background context explaining that protocols are informal interfaces, similar to those found in Smalltalk. Discuss the use of "a file-like object" as a protocol example.

:p How do protocols (informal interfaces) work in Python?
??x
Protocols in Python allow objects to behave like other types by implementing relevant methods, without being strictly enforced by the language.
??x

---

#### Evolving Protocols with Dynamic Typing
Background context explaining that established protocols can naturally evolve in languages using dynamic typing due to runtime type checking. Discuss how this applies to Ruby and Python.

:p Why are protocols particularly useful in dynamically typed languages like Python?
??x
Protocols are useful in dynamically typed languages because they allow objects to be flexible and adaptable, implementing only relevant parts of a protocol as needed.
??x

---

#### Emulating Built-In Types with Classes
Background context: The chapter discusses implementing classes that emulate built-in types, such as sequences. It emphasizes the importance of only emulating functionalities that make sense for the object being modeled to adhere to the KISS (Keep It Simple, Stupid) principle.
:p What is the main advice given regarding emulating built-in types in classes?
??x
The main advice is to only implement behaviors that are meaningful and practical for the specific object. For example, a sequence might handle individual element retrieval but not slicing operations if they don't apply logically to what the object represents.
x??

---

#### Using Protocols with `typing.Protocol`
Background context: The text mentions that using protocols can be beneficial when implementing type checks or ensuring certain behaviors are met in classes. It introduces `typing.Protocol` as a tool for this purpose, and notes that further discussion on protocols is reserved for Chapter 13.
:p How does the use of `typing.Protocol` differ from emulating built-in types?
??x
`typing.Protocol` is used to define abstract base classes with only type information (protocols), allowing you to specify what methods a class must implement without enforcing any implementation details. This can be useful for ensuring that certain methods are implemented and for static type checkers, but it doesn't enforce actual behavior in the same way as implementing built-in types.
x??

---

#### Duck Typing Origins
Background context: The text explains the concept of "duck typing," which is about allowing objects to behave like other objects even if they don't inherit from a common base class. It traces the term's origins back to Python discussions and mentions its popularization by the Ruby community.
:p Who helped popularize the term “duck typing,” and when did it first appear in Python?
??x
The term "duck typing" was popularized by the Ruby community, but it has been discussed in Python communities since at least 2000. An early example of its use can be found in a message to the Python-list by Alex Martelli on July 26, 2000.
x??

---

#### Customizing `__format__` for Vectors
Background context: The text discusses how the `__format__` method was implemented for vectors. It explains that unlike `__repr__`, which should always produce some output even if it's not ideal, `__format__` is intended to show the entire vector to end users.
:p How does the implementation of `__format__` differ from `__repr__`?
??x
The implementation of `__format__` for vectors allows them to display their full contents when formatted, whereas `__repr__` uses `reprlib` to limit output size for debugging and logging purposes. This ensures that users can see the entire vector representation without truncation.
x??

---

#### Implementing Limited Display with `*`
Background context: The text suggests a potential improvement to the `__format__` method by implementing a special format specifier (`*`) that disables the default limitation on the number of components displayed in vectors. This could help prevent accidental issues with very long displays.
:p How would you implement the `*` format specifier for vectors?
??x
You could implement it by checking if the format string ends with an asterisk (*) and, if so, disabling the size limitation. Here’s a simple implementation idea:

```python
class Vector:
    def __format__(self, fmt_spec):
        if fmt_spec.endswith('*'):
            return self._full_display()
        else:
            return self._limited_display(fmt_spec)

    def _full_display(self):
        # Return the full vector representation
        return f"Vector({', '.join(map(str, self))})"

    def _limited_display(self, fmt_spec):
        # Use reprlib for limited display if needed
        import reprlib
        return f"{reprlib.repr(self)}"
```

This way, users can use `*` to see the full vector representation.
x??

---

#### The Search for Pythonic Summarization
Background context: The text concludes by discussing what "Pythonic" means, noting that it's a subjective term often referring to using idiomatic Python. It emphasizes that there isn't one single answer but encourages users to adopt an "idiomatic Python" approach.
:p How can you determine if something is "Pythonic"?
??x
Determining if something is "Pythonic" involves adopting idiomatic practices in Python, such as leveraging built-in functions and libraries, using comprehensions, and writing code that aligns with common Python conventions. It’s more about the spirit of the language rather than strict rules.
x??

---

#### Pythonic Code Examples and Idiomatic Practices
Background context: The text discusses various approaches to solving a common problem—summing the second element of each sublist in a list of lists. It explores different coding styles, from functional programming using `reduce` and lambda functions to more explicit loop-based solutions and eventually to the built-in `sum()` function introduced in Python 2.3.

:p What is an example of a Pythonic code solution for summing elements in sublists?
??x
An idiomatic approach might involve using `functools.reduce`, but this can be seen as not optimal due to its complexity and readability issues:
```python
import functools

my_list = [[1, 2, 3], [40, 50, 60], [9, 8, 7]]
result = functools.reduce(lambda a, b: a + b[1], my_list, 0)
print(result)  # Output: 60
```
x??

---
#### Summing Elements with `functools.reduce`
Background context: The text highlights the use of `functools.reduce` to sum elements in sublists. However, it notes that this approach is not preferred due to its complexity and readability issues.
:p How does using `functools.reduce` for summing elements compare to other methods?
??x
Using `functools.reduce` can make the code harder to read and understand compared to simpler loop-based solutions or built-in functions. For example:
```python
import functools

my_list = [[1, 2, 3], [40, 50, 60], [9, 8, 7]]
result = functools.reduce(lambda a, b: a + b[1], my_list, 0)
```
This approach uses `reduce` and a lambda function to sum the second element of each sublist. While it is a functional programming technique, it may not be as Pythonic due to its complexity.

In contrast, using a simple loop or built-in functions like `sum` can make the code more readable:
```python
total = 0
for sub in my_list:
    total += sub[1]

result = total
```
x??

---
#### Python 2.3 Built-In Function: `sum`
Background context: The text mentions that the built-in function `sum` was introduced in Python 2.3 to simplify common operations like summing elements.
:p How does the introduction of `sum` impact code readability and functionality?
??x
The introduction of `sum` in Python 2.3 significantly improved code readability by providing a direct, built-in method for summing elements. For example:
```python
result = sum(sub[1] for sub in my_list)
```
This approach is more concise and easier to understand compared to using `functools.reduce`. It directly expresses the intention of summing the second element of each sublist.

Moreover, it handles edge cases like an empty list gracefully by returning 0:
```python
print(sum([]))  # Output: 0
```
x??

---
#### Python 2.4 Generator Expressions
Background context: The text highlights how generator expressions in Python 2.4 further improved the code's readability and performance.
:p What is an example of using a generator expression with `sum`?
??x
Generator expressions can be used within the `sum` function to provide more readable and efficient code:
```python
result = sum(sub[1] for sub in my_list)
```
This approach not only simplifies the code but also avoids potential issues with empty sequences, as `sum` defaults to 0 if no elements are present.

For example:
```python
print(sum([]))  # Output: 0
```
x??

---
#### Python 3 Changes and Idiomatic Practices
Background context: The text explains how the use of `functools.reduce` was reduced due to its complexity, leading it to be moved to the `functools` module in Python 3. However, it still has valid uses.
:p How did Python 3 change the way idiomatic practices are implemented?
??x
In Python 3, the use of `functools.reduce` became less common due to its complexity and readability issues. Instead, built-in functions like `sum` were preferred for simpler operations.

However, there are still cases where `reduce` can be useful, such as in implementing custom hash functions or when dealing with more complex operations that cannot be easily expressed with higher-level constructs.
```python
from functools import reduce

# Example of using reduce in Python 3
result = reduce(lambda a, b: a + b[1], my_list, 0)
```
x??

---
#### Vector.__hash__ Implementation
Background context: The text mentions that `functools.reduce` was used to implement the `Vector.__hash__` method in a way considered Pythonic.
:p How can `functools.reduce` be used to create a hash value for a vector?
??x
In the example provided, `functools.reduce` was used to create a hash value for a vector by summing up elements in a tuple:
```python
from functools import reduce

def __hash__(self):
    return reduce(lambda a, b: a + hash(b), self._vector, 0)
```
This approach ensures that the hash is computed as a combination of each element's hash value, making it more robust and Pythonic.

The `reduce` function iterates over each element in `_vector`, using a lambda to accumulate the hash values.
x??

---

#### Duck Typing
Duck typing is Python's default approach to typing, which has been present since its early days. It is based on the idea that "if it looks like a duck and quacks like a duck, then it probably is a duck." In programming terms, if an object implements all the necessary methods of another type, it can be treated as that type.

:p What does duck typing entail in Python?
??x
Duck typing involves treating objects based on their attributes and methods rather than their class inheritance. If an object has the required methods or properties, it can be used as if it were of a certain type.
```python
class Duck:
    def quack(self):
        print("Quack!")

def perform_quack(d: Duck) -> None:
    d.quack()

duck = Duck()
perform_quack(duck)
```
x??

---

#### Goose Typing
Goose typing is an approach supported by Abstract Base Classes (ABCs) since Python 2.6, which relies on runtime checks of objects against ABCs. This means that the methods defined in the ABC are used to define what it means for a class to be considered a "duck" or "goose." This can help ensure compatibility and provide more explicit type checking at runtime.

:p What is goose typing?
??x
Goose typing involves using Abstract Base Classes (ABCs) with predefined methods. These methods act as contracts that subclasses must implement. At runtime, the ABC checks whether a class has implemented these methods, ensuring it behaves correctly.
```python
from abc import ABC, abstractmethod

class Goose(ABC):
    @abstractmethod
    def honk(self) -> None:
        pass

def perform_honk(g: Goose) -> None:
    g.honk()

goose = Goose()
perform_honk(goose)
```
x??

---

#### Static Typing
Static typing is a traditional approach found in statically-typed languages like C and Java, where types are checked at compile time. In Python 3.5 and later, static typing was introduced through the `typing` module, which supports type hints but does not enforce them during runtime. External tools can check these types.

:p What is static typing?
??x
Static typing in programming means that the type of a variable or object must be declared before it is used, and this type cannot change over its lifetime. This approach enforces type checking at compile time rather than runtime.
```python
from typing import Protocol

class CanFly(Protocol):
    def fly(self) -> None:
        ...

def perform_flight(f: CanFly) -> None:
    f.fly()

class Bird:
    def fly(self) -> None:
        print("Flying...")

bird = Bird()
perform_flight(bird)
```
x??

---

#### Static Duck Typing
Static duck typing, introduced in Python 3.8 with `typing.Protocol`, is an approach inspired by Go's type system. It allows for defining protocols that specify a set of methods that must be implemented, and these can be checked statically.

:p What is static duck typing?
??x
Static duck typing involves using the `typing.Protocol` class to define interfaces or contracts. These protocols do not inherit from `ABC`, but they still enforce that classes implement certain methods. This allows for more precise type checking at both runtime and compile time.
```python
from typing import Protocol

class CanFly(Protocol):
    def fly(self) -> None:
        ...

def perform_flight(f: CanFly) -> None:
    f.fly()

class Bird:
    def fly(self) -> None:
        print("Flying...")

bird = Bird()
perform_flight(bird)
```
x??

---

#### Sequence Protocol in Python
Python has a robust data model that cooperates with essential dynamic protocols, such as sequences. The sequence protocol is formalized via Abstract Base Classes (ABCs) to ensure proper behavior when dealing with iterable objects.

:p What is a Sequence ABC in Python?
??x
The `Sequence` abstract base class from the `collections.abc` module defines a set of methods that classes can implement to behave like sequences. These include operations like indexing (`__getitem__`) and determining length (`__len__`). Classes implementing these methods can be treated as sequences by Python's built-in functions.
```python
from collections.abc import Sequence

class CustomSequence(Sequence):
    def __getitem__(self, index):  # Required method for indexing
        pass
    
    def __len__(self):  # Required method for determining length
        pass
```
x??

---

#### Iterability and `__getitem__` Method
Python's dynamic nature allows objects to be iterable even if they do not explicitly implement the `Iterable` protocol. Instead, Python checks if an object has a `__getitem__` method that can be used to iterate over it.

:p How does Python handle iteration on objects without implementing `__iter__`?
??x
Python uses the `__getitem__` method as a fallback for iteration. If an object provides this method and is subscriptable, Python will attempt to iterate over it by calling `__getitem__` with integer indexes starting from 0.

For example:
```python
class Vowels:
    vowels = 'aeiou'

    def __getitem__(self, index):
        return self.vowels[index]

vowels = Vowels()
for vowel in vowels:  # Python will call __getitem__ internally to iterate over vowels
    print(vowel)
```
x??

---

#### `in` Operator and Sequence Protocols
The `in` operator works with objects that implement the sequence protocol, even if they do not explicitly provide a `__contains__` method. This is because Python uses `__getitem__` in conjunction with a sequential scan to check for containment.

:p How does the `in` operator work on an object without `__contains__`?
??x
The `in` operator works by sequentially scanning the object using `__getitem__`. If the sequence protocol (which includes `__len__` and `__getitem__`) is implemented, Python can use these methods to check for the presence of items.

Example:
```python
class Vowels:
    vowels = 'aeiou'

    def __getitem__(self, index):
        return self.vowels[index]

vowels = Vowels()
print('a' in vowels)  # True, because 'a' is found using __getitem__
```
x??

---

#### `FrenchDeck` as a Sequence Example
The `FrenchDeck` class from the text provides an implementation of the sequence protocol. It implements both `__len__` and `__getitem__`, making it iterable.

:p How does `FrenchDeck` make use of the sequence protocol?
??x
`FrenchDeck` defines a deck of cards as a sequence by implementing the required methods of the sequence protocol:
- `__len__`: Returns the length of the deck.
- `__getitem__`: Allows indexing, enabling iteration over the deck.

Here is an example of how it works:
```python
class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()
    
    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                       for rank in self.ranks]
        
    def __len__(self):
        return len(self._cards)
    
    def __getitem__(self, position):
        return self._cards[position]

deck = FrenchDeck()
for card in deck:
    print(card)  # Iterates over the cards using __getitem__
```
x??

---

#### Inheritance and Protocol Implementation
The `Sequence` ABC provides a blueprint for implementing sequence behavior. However, not all classes need to inherit from it directly; they can implement the required methods (`__len__` and `__getitem__`) independently.

:p Can a class be considered as a sequence without inheriting from `abc.Sequence`?
??x
Yes, a class can be treated as a sequence by implementing `__len__` and `__getitem__`. These methods allow Python to treat the class as if it were a sequence. The `Sequence` ABC is just one way of ensuring proper behavior but not mandatory.

Example:
```python
from collections import abc

class Vowels(abc.Sequence):
    vowels = 'aeiou'

    def __getitem__(self, index):
        return self.vowels[index]
    
    def __len__(self):
        return len(self.vowels)

vowels = Vowels()
print(len(vowels))  # 5
print('a' in vowels)  # True
```
x??

---

