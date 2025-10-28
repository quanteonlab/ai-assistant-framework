# Flashcards: 10A000---FluentPython_processed (Part 27)

**Starting Chapter:** Deep and Shallow Copies of Arbitrary Objects

---

---
#### Shallow Copies vs. Deep Copies
Background context explaining the concept: In Python, when working with objects that contain other objects (like lists or dictionaries), shallow and deep copies are used to duplicate these structures. A shallow copy creates a new object but fills it with references to the original elements, whereas a deep copy duplicates both the object itself and all of its nested objects.

:p What is the key difference between shallow and deep copies?
??x
A shallow copy creates a new object but uses references for embedded objects, while a deep copy recursively creates a duplicate of the parent object including all nested objects. This means that in a shallow copy, changes to the original object's embedded objects can affect the copied object.
x??

---
#### Creating Shallow Copies
Background context explaining the concept: The `copy` module provides functions for creating shallow and deep copies of arbitrary objects. A common use case is when you want to modify an object without affecting the original.

:p How do you create a shallow copy using the `copy` module in Python?
??x
To create a shallow copy, you can use the `copy()` function from the `copy` module:
```python
import copy

original_obj = SomeClass()
shallow_copy = copy.copy(original_obj)
```
Here, `shallow_copy` is a shallow copy of `original_obj`, and any changes to the embedded objects in `original_obj` will be reflected in `shallow_copy`.
x??

---
#### Creating Deep Copies
Background context explaining the concept: Sometimes you need a complete duplication of an object including all its nested structures without sharing references. This can be achieved using the `deepcopy()` function from the `copy` module.

:p How do you create a deep copy using the `copy` module in Python?
??x
To create a deep copy, you use the `deepcopy()` function from the `copy` module:
```python
import copy

original_obj = SomeClass()
deep_copy = copy.deepcopy(original_obj)
```
Here, `deep_copy` is an independent duplicate of `original_obj`, and changes to any part of `original_obj` will not affect `deep_copy`.
x??

---
#### Example with Bus Class
Background context explaining the concept: The provided code demonstrates how shallow and deep copies work using a `Bus` class. The `Bus` class has methods for picking up and dropping off passengers, and its `passengers` attribute is a list.

:p What happens when you create a shallow copy of a bus object?
??x
When you create a shallow copy of a bus object (like `bus2` from the example), it shares references to the original passenger list. Therefore, changes made through one instance affect the other.

Example:
```python
import copy

bus1 = Bus(['Alice', 'Bill', 'Claire', 'David'])
bus2 = copy.copy(bus1)

# Dropping a passenger from bus1 affects bus2 because they share references.
bus1.drop('Bill')
print(bus2.passengers)  # Output: ['Alice', 'Claire', 'David']
```
x??

---
#### Cyclic References and Deep Copies
Background context explaining the concept: The `deepcopy()` function handles cyclic references gracefully, ensuring that all objects are copied properly.

:p How does `deepcopy()` handle cyclic references?
??x
`deepcopy()` manages cyclic references by keeping track of already copied objects to avoid infinite loops. It recursively copies each object, including nested ones, ensuring that no shared references cause issues with the copy.

Example:
```python
from copy import deepcopy

a = [10, 20]
b = [a, 30]
a.append(b)

# Using deepcopy to handle cyclic references.
c = deepcopy(a)
print(c)  # Output: [10, 20, [[...], 30]]
```
x??

---

#### Call by Sharing in Python
Call by sharing is a mode of parameter passing used in Python, similar to most object-oriented languages like JavaScript, Ruby, and Java. In this method, each formal parameter gets a copy of each reference from the arguments passed to the function. Inside the function, parameters become aliases for the actual arguments.
:p How does call by sharing work in Python?
??x
In Python, when you pass an argument to a function, a copy of the reference is made and given to the formal parameter. This means that any mutable object (like lists or dictionaries) passed as an argument can be modified inside the function because it shares the same memory location with the original object. However, the identity of the objects cannot be changed within the function.
```python
def f(a, b):
    a += b  # Modifies list 'a' in place

x = [1, 2]
y = [3, 4]
f(x, y)
print(x)  # Output: [1, 2, 3, 4]
```
x??

---

#### Example of Call by Sharing
Let's see how call by sharing affects the behavior of functions with different types of arguments.
:p What happens when a function receives numbers, lists, and tuples as parameters?
??x
When a function receives numbers, it cannot modify them because they are immutable. Lists and tuples can be modified if they are mutable and passed to the function, but their identity remains unchanged within the function.

For example:
```python
def f(a, b):
    a += b  # Only works with lists/tuples

x = 1
y = 2
f(x, y)
print(x)  # Output: 1 (unchanged)

a = [1, 2]
b = [3, 4]
f(a, b)
print(a)  # Output: [1, 2, 3, 4] (modified in place)

t = (10, 20)
u = (30, 40)
f(t, u)
print(t)  # Output: (10, 20, 30, 40) (unchanged)
```
x??

---

#### Mutable Types as Parameter Defaults
Using mutable objects for default parameters in Python can lead to unexpected behavior due to how function defaults are evaluated.
:p Why should one avoid using mutable types as parameter defaults?
??x
When you use a mutable type like a list or dictionary as a default value for a function parameter, it is evaluated only once when the function is defined. Any changes made to this default object will affect all subsequent calls of the function because they share the same reference.

For example:
```python
def f(a=[]):
    a.append('new element')
    return a

print(f())  # Output: ['new element']
print(f())  # Output: ['new element', 'new element'] (shared state)
```
x??

---

#### The HauntedBus Example
The `HauntedBus` class demonstrates the problem of using mutable default parameters.
:p How does the `HauntedBus` class exhibit shared state?
??x
The `HauntedBus` class uses a mutable list as a default parameter, leading to unintended behavior. When no `passengers` argument is provided, it defaults to an empty list. Each instance of `HauntedBus` shares the same reference to this list, causing all instances to modify the same passenger list.

Example:
```python
class HauntedBus:
    def __init__(self, passengers=[]):
        self.passengers = passengers

    def pick(self, name):
        self.passengers.append(name)

    def drop(self, name):
        self.passengers.remove(name)

bus1 = HauntedBus(['Alice', 'Bill'])
print(bus1.passengers)  # Output: ['Alice', 'Bill']

bus2 = HauntedBus()
bus2.pick('Charlie')
print(bus2.passengers)  # Output: ['Charlie']

bus3 = HauntedBus()
bus3.pick('Carrie')
print(bus3.passengers)  # Output: ['Carrie']

print(bus2 is bus3)  # Output: True
```
x??

---

#### Mutable Default Arguments
Background context explaining that mutable default arguments can lead to unexpected behavior because they are evaluated only once when the function is defined. This means any changes made to the argument will affect all instances where it was used, leading to shared state issues.

:p Explain why mutable defaults like lists or dictionaries should be avoided as parameters in functions.
??x
Mutable defaults such as lists or dictionaries should be avoided as parameters because they are evaluated only once when the function is defined. Any changes made to these arguments persist across multiple function calls, causing unintended side effects and sharing state between different parts of a program.

For example, using `[]` as a default value for a list parameter will make all functions that use this parameter share the same list. If one function modifies the list, it affects other instances where the list was used.
??x

:p Provide an example in Python demonstrating why mutable defaults can cause issues.
??x
```python
def modify_list(lst=[]):
    lst.append('new item')
    return lst

# Using the function multiple times will share the same default list
print(modify_list())  # Output: ['new item']
print(modify_list())  # Output: ['new item', 'new item']
```
Here, `modify_list` uses an empty list as a mutable default argument. Each call to `modify_list()` modifies the same shared list, leading to unexpected behavior.

To avoid this issue, always use immutable types like `None` and create new lists inside the function:
```python
def modify_list(lst=None):
    if lst is None:
        lst = []
    lst.append('new item')
    return lst

# Each call uses a fresh copy of the list
print(modify_list())  # Output: ['new item']
print(modify_list())  # Output: ['new item']
```
x??

---

#### Defensive Programming with Mutable Parameters
Background context explaining how defensive programming practices can prevent shared state issues in mutable parameters. It’s important to consider whether a function should modify the input parameter or work with a copy.

:p Why is it necessary to make copies of mutable arguments when they are passed into functions?
??x
When passing mutable arguments like lists or dictionaries, making copies ensures that each call to the function works with its own independent data. If no copy is made and the function modifies the argument directly, this change will affect all other calls that received the same original argument.

For example:
```python
def modify_dict(d={}):
    d['key'] = 'value'
    return d

print(modify_dict())  # Output: {'key': 'value'}
print(modify_dict())  # Output: {'key': 'value', 'key': 'value'}
```
Here, the function `modify_dict` uses an empty dictionary as a mutable default argument. Each call modifies the same shared dictionary.

To prevent this issue, always create a new copy of the mutable object:
```python
def modify_dict(d=None):
    if d is None:
        d = {}
    d['key'] = 'value'
    return d

print(modify_dict())  # Output: {'key': 'value'}
print(modify_dict())  # Output: {'key': 'value'}
```
x??

---

#### Aliasing and Twisted Bus Example
Background context explaining how the `TwilightBus` class inverts expectations by sharing its passenger list with clients, violating the principle of least astonishment.

:p How does the `TwilightBus` violate the "Principle of Least Astonishment"?
??x
The `TwilightBus` violates the "Principle of Least Astonishment" because it shares its passenger list with clients. When a client passes a list to the bus and then calls methods like `drop`, the elements are removed from both the bus's internal state and the original list.

For example:
```python
basketball_team = ['Sue', 'Tina', 'Maya', 'Diana', 'Pat']
bus = TwilightBus(basketball_team)
print(bus.drop('Tina'))  # Output: None (or similar)
print(basketball_team)  # Output: ['Sue', 'Maya', 'Diana']
```
Here, `basketball_team` and the bus share the same list. When a passenger is dropped from the bus, they are also removed from `basketball_team`.

To avoid this issue, create a copy of the list:
```python
def __init__(self, passengers=None):
    if passengers is None:
        self.passengers = []
    else:
        self.passengers = list(passengers)
```
This ensures that any modifications to the bus's passenger list do not affect the original list passed in.

x??

---

