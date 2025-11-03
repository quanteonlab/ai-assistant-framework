# Flashcards: 10A000---FluentPython_processed (Part 44)

**Starting Chapter:** Complete Listing of Vector2d version 3

---

#### Vector2d Class Overview
Background context explaining the Vector2d class, its purpose, and how it was developed. The class is a two-dimensional vector with several special methods implemented to support various operations such as arithmetic, comparison, formatting, and hashing.

:p What is the Vector2d class in this context?
??x
The Vector2d class represents a two-dimensional vector with properties for `x` and `y` coordinates. It implements multiple special methods like `__init__`, `__iter__`, `__repr__`, `__str__`, etc., to support operations such as arithmetic, comparison, formatting, and hashing.
```python
class Vector2d:
    __match_args__  = ('x', 'y')
    
    typecode = 'd'
    
    def __init__(self, x, y):
        self.__x = float(x)
        self.__y = float(y)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y
    
    # Other methods like __iter__, __repr__, etc.
```
x??

---

#### Vector2d Special Methods - `__init__` and Properties
Background context explaining the initialization of a vector with `x` and `y` coordinates, and how properties are used to provide controlled access to these values.

:p What does the `__init__` method in Vector2d do?
??x
The `__init__` method initializes a new instance of the Vector2d class by setting the `x` and `y` attributes. It ensures that the coordinates are converted to float type for consistency.
```python
def __init__(self, x, y):
    self.__x = float(x)
    self.__y = float(y)
```
x??

---

#### Vector2d Special Methods - `__iter__`
Background context explaining how iteration over a vector works and the role of the `__iter__` method in providing an iterator.

:p How does the `__iter__` method work in Vector2d?
??x
The `__iter__` method is implemented to allow iteration over the vector's coordinates. It returns a generator that yields the x and y values one by one.
```python
def __iter__(self):
    return (i for i in (self.x, self.y))
```
x??

---

#### Vector2d Special Methods - `__repr__` and `__str__`
Background context explaining how objects are represented as strings using the `__repr__` and `__str__` methods.

:p What do the `__repr__` and `__str__` methods in Vector2d return?
??x
The `__repr__` method returns a string representation of the vector that can be used to recreate the object. The `__str__` method returns a readable string representation of the vector's coordinates.
```python
def __repr__(self):
    class_name = type(self).__name__
    return '{}({:.r}, {:.r})'.format(class_name, *self)

def __str__(self):
    return str(tuple(self))
```
x??

---

#### Vector2d Special Methods - `__bytes__`
Background context explaining how to serialize a vector object into bytes and the role of the `__bytes__` method.

:p What does the `__bytes__` method in Vector2d do?
??x
The `__bytes__` method returns a byte representation of the vector. It first converts the typecode (in this case, 'd' for double precision float) to bytes and then appends the serialized coordinates.
```python
def __bytes__(self):
    return (bytes([ord(self.typecode)]) + 
            bytes(array(self.typecode, self)))
```
x??

---

#### Vector2d Special Methods - `__eq__` and `__hash__`
Background context explaining how to compare vector objects for equality and generate unique hash values.

:p How does the `__eq__` method in Vector2d compare vectors?
??x
The `__eq__` method compares two vectors by comparing their coordinates as tuples. It returns `True` if both coordinates are equal, otherwise `False`.
```python
def __eq__(self, other):
    return tuple(self) == tuple(other)
```
x??

---

#### Vector2d Special Methods - `__hash__`
Background context explaining the role of hashing in equality checks and set operations.

:p What does the `__hash__` method in Vector2d do?
??x
The `__hash__` method returns a hash value for the vector. It combines the hashes of the x and y coordinates using the XOR operation to ensure unique identification.
```python
def __hash__(self):
    return hash(self.x) ^ hash(self.y)
```
x??

---

#### Vector2d Special Methods - `__abs__` and `__bool__`
Background context explaining how to get the magnitude of a vector and its truthiness.

:p What does the `__abs__` method in Vector2d do?
??x
The `__abs__` method returns the magnitude (or length) of the vector using the Euclidean distance formula.
```python
def __abs__(self):
    return math.hypot(self.x, self.y)
```
x??

---

#### Vector2d Special Methods - `__bool__`
Background context explaining how to determine if a vector is "truthy" based on its magnitude.

:p What does the `__bool__` method in Vector2d do?
??x
The `__bool__` method returns `True` if the vector has a non-zero length (i.e., not equal to zero), otherwise it returns `False`.
```python
def __bool__(self):
    return bool(abs(self))
```
x??

---

#### Vector2d Special Methods - `angle`
Background context explaining how to calculate the angle of a vector with respect to the x-axis.

:p What does the `angle` method in Vector2d do?
??x
The `angle` method calculates and returns the angle (in radians) that the vector makes with the positive x-axis using the `math.atan2` function.
```python
def angle(self):
    return math.atan2(self.y, self.x)
```
x??

---

#### Vector2d Special Methods - `__format__`
Background context explaining how to format a vector in various ways (Cartesian and polar coordinates).

:p What does the `__format__` method in Vector2d do?
??x
The `__format__` method formats the vector according to the specified format specification. It supports both Cartesian (`(x, y)`) and polar (`<r, theta>`) coordinate representations.
```python
def __format__(self, fmt_spec=''):
    if fmt_spec.endswith('p'):
        fmt_spec = fmt_spec[:-1]
        coords = (abs(self), self.angle())
        outer_fmt = '<{}, {}>'
    else:
        coords = self
        outer_fmt = '({}, {})'
    components = (format(c, fmt_spec) for c in coords)
    return outer_fmt.format(*components)
```
x??

---

#### Vector2d Special Methods - `frombytes`
Background context explaining how to create a vector from byte data.

:p What does the `frombytes` class method in Vector2d do?
??x
The `frombytes` class method creates a new instance of Vector2d from its byte representation. It first extracts the typecode and then deserializes the coordinates.
```python
@classmethod
def frombytes(cls, octets):
    typecode = chr(octets[0])
    memv = memoryview(octets[1:]).cast(typecode)
    return cls(*memv)
```
x??

#### Private and "Protected" Attributes in Python

Background context: In Python, there is no true private variable mechanism like in languages such as Java. The `__` prefix (two underscores) is a naming convention used to prevent accidental overwriting of attributes in subclasses. This feature is known as name mangling.

Name mangling works by prefixing the attribute with an underscore and the class name when accessed from outside the class, effectively making it less accessible but not truly private or immutable.

If you have a `Dog` class that uses a mood attribute internally without exposing it, and someone subclasses `Dog` to create a `Beagle` class and defines its own `mood` attribute, it could overwrite the original `mood` attribute used by methods from `Dog`. To prevent this, Python mangles the name of the private attributes.

:p What is the purpose of using double underscores (`__`) in Python variable names?

??x
Name mangling in Python is designed to prevent accidental access or clobbering of variables within subclasses. It works by prefixing the attribute with an underscore and the class name, making it harder for developers to accidentally overwrite these attributes when subclassing.

For example:
```python
class Dog:
    def __init__(self, mood):
        self.__mood = mood

class Beagle(Dog):
    # This would cause a conflict without mangling as Python would look for _Beagle__mood
```

x??

---

#### Name Mangled Private Attributes in Vector2d Class

Background context: In the `Vector2d` class, private attributes like `__y` and `__x` are mangled to `_Vector2d__y` and `_Vector2d__x`, respectively. This helps avoid conflicts with subclass names.

:p What happens when you access a mangled attribute directly in Python?

??x
When you access a mangled attribute directly, it appears as if the original private name was used, but internally, it is prefixed with an underscore and the class name. For example, `v1._Vector2d__x` returns the value of the x-coordinate without causing issues due to name clashes.

For instance:
```python
>>> v1 = Vector2d(3, 4)
>>> v1.__dict__
{'_Vector2d__y': 4.0, '_Vector2d__x': 3.0}
>>> v1._Vector2d__x
3.0
```

x??

---

#### Single Underscore Prefix as a Convention

Background context: In Python, using a single underscore (`_`) before an attribute name is a convention used to indicate that the attribute should not be accessed from outside the class. This practice is known as "protecting" attributes.

While this does not make the attribute truly private or immutable (as it can still be accessed and modified), it serves as a strong signal to other developers to respect this naming convention.

:p Why do some Python programmers prefer using a single underscore over double underscores?

??x
Some Python programmers prefer using a single underscore (`_`) for attributes because they believe that explicit naming conventions are clearer and more maintainable. Using `self._x` instead of `self.__x` makes the code easier to read and understand, as it does not involve mangled names.

For example:
```python
class MyClass:
    def __init__(self):
        self._my_attribute = 10

# This is preferred over using double underscores for clarity.
```

x??

---

#### Name Mangling vs. Conventions

Background context: Python's name mangling uses a double underscore (`__`) prefix to mangle attribute names, making them harder to accidentally overwrite in subclasses but not fully private or immutable. Some developers prefer to use the single underscore (`_`) as a "protected" attribute convention without the complexity of name mangling.

:p What are the criticisms of using Python's automatic name mangling?

??x
Critics argue that automatic name mangling is annoying and unnecessary because it obscures the true names of attributes. They suggest that explicit naming conventions, such as `self._x`, are clearer and more maintainable. Critics like Ian Bicking advocate for using single underscores (`_`) instead of double underscores to indicate "protected" attributes.

For instance:
```python
class MyClass:
    def __init__(self):
        self._my_attribute = 10

# This approach is preferred by some over name mangling.
```

x??

---

#### Vector2d Class Components and Immutability

Background context: In the `Vector2d` class, components like `__y` and `__x` are intended to be private, making them difficult to modify from outside. However, this does not make the vector truly immutable because there is no mechanism in Python to enforce immutability at the language level.

:p Why can't Vector2d components really be made private or immutable?

??x
Vector2d components like `__y` and `__x` are marked as "private" using double underscores, but this does not make them truly private or immutable. Python allows direct access to these attributes through name mangling (`_Vector2d__y`, `_Vector2d__x`). While the intent is for these attributes to be treated as internal details of the class, they can still be modified if necessary.

For example:
```python
class Vector2d:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

v1 = Vector2d(3, 4)
print(v1._Vector2d__x)  # Direct access to the private attribute is possible.
```

x??

---

#### Special Attribute: __slots__

Background context: The `__slots__` special attribute in Python classes can be used to control the internal storage of an object. By specifying a list of allowed attributes, you can save memory and ensure that only those attributes are stored.

:p What is the purpose of using the `__slots__` attribute?

??x
The `__slots__` attribute allows you to specify a list of instance variables that will be used by an instance of a class. By defining `__slots__`, you can control which attributes are allowed and save memory compared to using a full dictionary for each object.

For example:
```python
class Vector2d:
    __slots__ = ['_x', '_y']

    def __init__(self, x, y):
        self._x = x
        self._y = y

v1 = Vector2d(3, 4)
print(v1._x)  # Direct access to the protected attribute is possible.
```

x??

#### Memory Optimization with `__slots__`
In Python, by default, each instance of a class stores its attributes in a dictionary called `__dict__`. However, this can result in significant memory overhead. The `__slots__` mechanism is used to reduce this memory usage.

The `__slots__` attribute allows you to specify which attributes the instances of your class should have, and stores them in an array instead of a dictionary. This reduces memory consumption but limits flexibility.
:p What does the `__slots__` feature do in Python?
??x
The `__slots__` feature in Python restricts the storage mechanism for instance attributes to reduce memory usage by using an array instead of a dictionary, while limiting the number and type of attributes an instance can have.

This optimization is achieved by defining a class attribute named `__slots__`, which holds a tuple or list of strings representing the names of allowed attributes. Here's how it works:

```python
class Pixel:
    __slots__ = ('x', 'y')
```

When you create an instance of `Pixel` and try to add additional attributes, Python raises an `AttributeError`.
??x
The `__slots__` mechanism restricts the creation of new attributes on instances. For example:

```python
p = Pixel()
p.x = 10  # Allowed
p.y = 20  # Allowed

# Trying to set a new attribute:
p.color = 'red'  # Raises AttributeError: 'Pixel' object has no attribute 'color'
```

If `__slots__` is not defined, the instance will fall back to using `__dict__`:

```python
class Pixel:
    pass

p = Pixel()
p.x = 10  # Allowed as it uses __dict__
p.y = 20  # Also allowed
p.color = 'red'  # Also allowed
```

To extend this behavior to subclasses, you need to define `__slots__` in the subclass:
```python
class OpenPixel(Pixel):
    pass

op = OpenPixel()
op.x = 8  # Stored directly as it's defined in __slots__
op.color = 'green'  # Falls back to __dict__
```

This example shows that attributes defined in `__slots__` of the base class are stored directly, while those not in `__slots__` use `__dict__`.
??x
When defining a subclass and wanting it to also use `__slots__`, you must explicitly define it:

```python
class ColorPixel(Pixel):
    __slots__ = ('color',)

cp = ColorPixel()
cp.x = 2  # Allowed as 'x' is in __slots__
cp.color = 'blue'  # Also allowed

# Trying to add a new attribute:
cp.flavor = 'banana'  # Raises AttributeError: 'ColorPixel' object has no attribute 'flavor'
```

Using `__slots__` with an empty tuple in subclasses can also restrict the instance attributes:

```python
class OpenPixel(Pixel):
    __slots__ = ()

op = OpenPixel()
op.x = 8  # Fallback to __dict__
op.color = 'green'  # Also stored in __dict__
```

This ensures that only attributes explicitly listed in `__slots__` are stored directly.
??x
Using `__slots__` can be beneficial for memory optimization, but it limits the flexibility of adding new attributes. It's important to define `__slots__` carefully and understand its implications.

For example:
```python
class Pixel:
    __slots__ = ('x', 'y')

p = Pixel()
p.x = 10  # Allowed
p.y = 20  # Allowed

# Trying to add a new attribute:
p.color = 'red'  # Raises AttributeError: 'Pixel' object has no attribute 'color'
```

In this case, only `x` and `y` can be directly assigned without using the `__dict__`.
??x
The limitation is that once you define `__slots__`, adding or changing it later does not affect existing instances. The attributes defined in `__slots__` must be present when the class is created.
??x

---
---

