# Flashcards: 10A000---FluentPython_processed (Part 3)

**Starting Chapter:** String Representation

---

#### Special Methods Overview
Background context explaining that special methods are used for operator overloading and other operations. These methods are not called directly but by Python's interpreter or certain built-in functions.

:p What is the purpose of implementing special methods like `__bool__`, `__add__`, and `__mul__` in a class?
??x
The primary purpose of implementing these special methods is to provide custom behavior when using operators such as +, * with objects of your class. These methods allow you to define how instances of your class should interact with other values or instances.

For example:
- The `__bool__` method defines the truth value of an object.
- The `__add__` and `__mul__` methods enable addition and multiplication operations on custom objects, respectively.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
```

x??

---

#### `__bool__` Method
Background context on how the `__bool__` method is used to determine the truth value of an object. In Python, non-empty collections and objects with a non-zero numeric value are considered true.

:p What does the `__bool__` method do in the provided Vector class?
??x
The `__bool__` method returns the boolean value of the vector based on its magnitude. It uses the `abs()` function to calculate the absolute value (magnitude) of the vector and then checks if it is non-zero.

```python
def __bool__(self):
    return bool(abs(self))
```

If the vector has a zero length, it returns `False`; otherwise, it returns `True`.

x??

---

#### `__add__` Method
Background context on how operator overloading works through special methods. The `__add__` method is used to define custom behavior for the + operator when applied to instances of your class.

:p How does the `__add__` method work in the Vector class?
??x
The `__add__` method in the Vector class defines how two vectors can be added together. It takes another vector as an argument, adds their respective components (x and y), and returns a new vector with the summed values.

```python
def __add__(self, other):
    x = self.x + other.x
    y = self.y + other.y
    return Vector(x, y)
```

This method ensures that when you use the `+` operator between two Vector objects, it performs component-wise addition and returns a new Vector.

x??

---

#### `__mul__` Method
Background context on how multiplication is handled with special methods. The `__mul__` method allows defining custom behavior for the * operator.

:p How does the `__mul__` method work in the Vector class?
??x
The `__mul__` method in the Vector class defines scalar multiplication of a vector. It takes a scalar value as an argument and multiplies each component (x, y) of the vector by that scalar, returning a new Vector.

```python
def __mul__(self, scalar):
    return Vector(self.x * scalar, self.y * scalar)
```

This method ensures that when you use the `*` operator between a Vector object and a scalar value, it performs element-wise multiplication and returns a new Vector.

x??

---

#### `__repr__` Method
Background context on how `__repr__` is used to get a string representation of an object for inspection. The `__repr__` method provides a readable format that can be used to recreate the object.

:p What does the `__repr__` method do in the Vector class?
??x
The `__repr__` method in the Vector class returns a string representation of the vector, which is useful for debugging and inspecting objects. This string should be unambiguous and allow recreating the object if possible.

```python
def __repr__(self):
    return f"Vector({self.x}, {self.y})"
```

The `__repr__` method ensures that when you print a Vector object or use it with `repr()`, you get a meaningful string like `Vector(1, 2)` instead of the default representation.

x??

---

#### `__str__` Method
Background context on how `__str__` is used to return a user-friendly string. The `__str__` method returns a string that is suitable for display to end users.

:p What does the `__str__` method do in the Vector class?
??x
The `__str__` method in the Vector class provides a user-friendly string representation of the vector, which is useful for displaying the object's state to an end user. By default, if you implement only `__repr__`, Python will use it as fallback for `__str__`.

```python
def __str__(self):
    return f"Vector({self.x}, {self.y})"
```

The `__str__` method ensures that when you print a Vector object or use the `print()` function, you get a readable string like `Vector(1, 2)`.

x??

---

#### Commutative Property in Multiplication
Background context on the commutative property of scalar multiplication. The provided implementation only allows multiplying a vector by a scalar but not vice versa, violating this property.

:p Why is it important to implement both `__mul__` and `__rmul__` for the Vector class?
??x
Implementing both `__mul__` and `__rmul__` ensures that scalar multiplication is commutative. The current implementation only supports multiplying a vector by a scalar, but not vice versa. This violates the commutative property of scalar multiplication, which states that the order of operands should not matter.

To fix this, you need to implement `__rmul__`, which allows for the reverse operation:

```python
def __rmul__(self, other):
    return self.__mul__(other)
```

This ensures that both `vector * 2` and `2 * vector` yield the same result.

x??

---

#### Boolean Value of a Custom Type

In Python, any object can be used in a boolean context. To determine whether an instance of a user-defined class is truthy or falsy, you need to implement special methods such as `__bool__` and `__len__`. By default, instances are considered truthy unless these methods are defined.

If `__bool__` is not implemented, Python will call `__len__`, and if that returns zero, it will return `False`; otherwise, it returns `True`.

:p What does the `__bool__` method do in Python?
??x
The `__bool__` method allows you to customize how an object behaves in a boolean context. It should return `True` or `False`, indicating whether the object is considered truthy or falsy.

For example, if you have a class representing a vector and want it to be considered false only when its magnitude is zero, you can implement `__bool__` as follows:

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __bool__(self):
        return bool(abs(self))
```

This implementation uses the `abs` function to get the magnitude of the vector and then converts it to a boolean using `bool`.

x??

---

#### Faster Implementation of Vector.__bool__

A more efficient way to implement the `__bool__` method is to directly check if any component of the vector is non-zero. This avoids computing the full magnitude.

:p How can you optimize the `__bool__` method for a vector class?
??x
You can optimize the `__bool__` method by checking if either the x or y component of the vector is non-zero, as this directly indicates whether the vector has any magnitude:

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __bool__(self):
        return bool(self.x or self.y)
```

This implementation avoids calling `abs`, computing squares, and taking the square root. It simply returns `True` if either component is non-zero, making it faster.

x??

---

#### Collection API

Python’s collection interfaces are defined by abstract base classes (ABCs) in the `collections.abc` module. These ABCs provide a common interface for collections like lists, dictionaries, and sets. Implementing these special methods allows your custom class to interact seamlessly with built-in functions and other collections.

:p What is the role of the Collection ABC in Python?
??x
The Collection ABC in Python unifies three essential interfaces that every collection should implement: `Iterable`, `Sized`, and `Contains`. These interfaces support operations like iteration, length determination, and membership testing. By implementing these special methods, your custom class can be used similarly to built-in collections.

For example, a class that implements the `__len__` method satisfies the `Sized` interface:

```python
class CustomCollection:
    def __len__(self):
        return len(self.data)
```

This allows you to use the `len` function on instances of your class.

x??

---

#### UML Class Diagram with Fundamental Collection Types

The provided text describes an UML class diagram that outlines the interfaces for fundamental collection types in Python. These include `Sequence`, `Mapping`, and `Set`, each with specific special methods.

:p What are the three main ABCs described in the Collection API section?
??x
The three main abstract base classes (ABCs) described in the Collection API section are:

1. **Collection**: Unifies the interfaces for `Iterable`, `Sized`, and `Contains`.
2. **Sequence**: Formalizes the interface of built-in collections like lists and strings.
3. **Mapping**: Implemented by dictionaries, defaultdicts, etc.

Each of these ABCs has specific special methods that need to be implemented to provide a complete collection interface:

- `__len__` for the `Sized` interface
- `__iter__` and `__next__` for the `Iterable` interface
- `__contains__` for the `Contains` interface

x??

---

#### Infix Operators in Collection ABCs

The UML class diagram also mentions that some special methods in the Set ABC implement infix operators, such as `&` which computes the intersection of sets.

:p How do you implement the intersection operation for a set using an operator?
??x
You can implement the intersection operation for a set by defining the `__and__` method. This method is used to compute the intersection of two sets and returns a new set containing elements that are common to both sets.

For example:

```python
class CustomSet:
    def __init__(self, items):
        self.items = set(items)

    def __and__(self, other):
        return CustomSet(self.items & other.items)
```

Here, the `__and__` method computes the intersection of two custom sets and returns a new instance of `CustomSet` containing the common elements.

x??

---

---
#### String/Bytes Representation Methods
Background context explaining how these methods allow custom objects to have a meaningful representation when printed or converted to strings. These are essential for debugging and logging.

:p Which method is used to provide a string that should be used for pretty-printing the object?
??x
The `__str__` method returns a string that is more readable, often used for end-user output like print statements.
```python
class MyObject:
    def __str__(self):
        return "This is my object"
```
x??

---
#### Conversion to Number Methods
Background context explaining how these methods allow objects to be converted into numbers. These are useful in scenarios where the object needs to participate in arithmetic operations.

:p Which method returns a complex number from an instance?
??x
The `__complex__` method returns a complex number.
```python
class MyNumber:
    def __complex__(self):
        return 3 + 4j
```
x??

---
#### Emulating Collections Methods
Background context explaining how these methods allow objects to emulate collections, such as lists or dictionaries. This is useful for creating custom iterable and subscriptable types.

:p Which method defines the length of a collection?
??x
The `__len__` method returns the number of items in the collection.
```python
class MyCollection:
    def __len__(self):
        return 5
```
x??

---
#### Iteration Methods
Background context explaining how these methods enable objects to be iterable, allowing them to be used with for-loops and other iteration constructs.

:p Which method is called to get the next item in a collection during iteration?
??x
The `__next__` method returns the next value from the iterator. In Python 3.5+, `__anext__` was introduced for asynchronous iterators.
```python
class MyIterator:
    def __iter__(self):
        return self

    def __next__(self):
        # Return the next item or raise StopIteration when done
        pass
```
x??

---
#### Callable Methods
Background context explaining how these methods enable objects to be called as functions.

:p Which method is used for object invocation, allowing an object to act like a function?
??x
The `__call__` method allows the object to be called as a function.
```python
class MyFunction:
    def __call__(self):
        print("Function was called")
```
x??

---
#### Context Management Methods
Background context explaining how these methods enable objects to act like context managers, allowing for use in with statements.

:p Which method is used to acquire resources when entering a context?
??x
The `__enter__` method acquires the necessary resources and returns them.
```python
class MyContextManager:
    def __enter__(self):
        print("Entering context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting context")
```
x??

---
#### Instance Creation and Destruction Methods
Background context explaining how these methods control the instantiation of objects and their cleanup.

:p Which method is called to create a new instance of a class?
??x
The `__new__` method creates a new instance and returns it. The `__init__` method initializes the new instance.
```python
class MyClass:
    def __new__(cls):
        print("Creating an instance")
        return super().__new__(cls)

    def __init__(self):
        print("Initializing instance")
```
x??

---
#### Attribute Management Methods
Background context explaining how these methods handle attribute access and modification, including dynamic attribute management.

:p Which method is used to get the value of an attribute?
??x
The `__getattr__` method returns the value for a missing attribute.
```python
class MyClass:
    def __getattr__(self, name):
        return f"Attribute {name} does not exist"
```
x??

---
#### Abstract Base Classes Methods
Background context explaining how these methods provide hooks into class construction and checking.

:p Which method is used to check if an instance is of a certain type?
??x
The `__instancecheck__` method checks if the given object is an instance of the class.
```python
class MyType:
    @classmethod
    def __instancecheck__(cls, other):
        return isinstance(other, cls)
```
x??

---
#### Class Metaprogramming Methods
Background context explaining how these methods allow for advanced customization and manipulation of classes at the metaclass level.

:p Which method is used to customize class creation?
??x
The `__init_subclass__` method allows custom behavior when a subclass is defined.
```python
class MyMeta(type):
    def __init_subclass__(cls, **kwargs):
        print("Subclassing occurred")
```
x??

---
#### Infix and Numerical Operator Methods
Background context explaining how these methods support the use of operators as special methods.

:p Which method supports matrix multiplication?
??x
The `__matmul__` method supports matrix multiplication.
```python
class MyMatrix:
    def __matmul__(self, other):
        # Perform matrix multiplication logic here
        pass
```
x??

---

