# Flashcards: 10A000---FluentPython_processed (Part 43)

**Starting Chapter:** Formatted Displays

---

#### Classmethod Decorator
Background context: The `classmethod` decorator is a Python-specific feature that allows you to define methods within a class that can operate on the class itself rather than on instances of the class. This is particularly useful for alternative constructors, utility functions related to the class, or any method where you need to use the class object in some way.

:p What is the `classmethod` decorator used for?
??x
The `classmethod` decorator is used to define methods that can operate on a class itself rather than instances of the class. It allows access to and manipulation of class-specific data without the need for an instance of the class. A common use case is creating alternative constructors, where you might want to create an object from a different source like bytes or a string.

Example code:
```python
@classmethod
def frombytes(cls, octets):
    typecode = chr(octets[0])
    memv = memoryview(octets[1:]).cast(typecode)
    return cls(*memv)
```
x??

---

#### Staticmethod Decorator
Background context: The `staticmethod` decorator is another Python-specific feature that creates a method that does not receive any special first argument, unlike instance methods and classmethods. It is similar to a regular function but lives within the class body. This can be useful for utility functions or methods where you don't need access to either the class or its instances.

:p What is the `staticmethod` decorator used for?
??x
The `staticmethod` decorator is used to define a method that does not receive any special first argument (neither an instance nor the class). This means it behaves like a standalone function but can be called as if it were part of the class. It's useful when you have a utility function that doesn't need access to either the class or its instances.

Example code:
```python
class Demo:
    @classmethod
    def klassmeth(*args):
        return args

    @staticmethod
    def statmeth(*args):
        return args
```
x??

---

#### `frombytes` Method in Vector2d Class
Background context: In the given text, a method named `frombytes` is introduced for the `Vector2d` class. This method allows you to create an instance of `Vector2d` from a binary sequence of bytes. The first byte of the sequence is used to determine the typecode (e.g., 'f' for float), and the rest of the sequence is converted into memoryview, which is then unpacked into the necessary arguments for constructing the `Vector2d`.

:p What does the `frombytes` class method do?
??x
The `frombytes` class method reads a binary sequence (octets) to construct a new instance of `Vector2d`. It first extracts the typecode from the first byte, then creates a memoryview from the remaining bytes and casts it according to the typecode. Finally, it uses this memoryview to initialize a new instance.

Example code:
```python
@classmethod
def frombytes(cls, octets):
    typecode = chr(octets[0])
    memv = memoryview(octets[1:]).cast(typecode)
    return cls(*memv)
```
x??

---

#### Comparison of `classmethod` and `staticmethod`
Background context: The text explains the differences between `classmethod` and `staticmethod`. While a classmethod receives the class as its first argument, allowing it to operate on the class itself or create new instances from different sources (like bytes), a staticmethod does not receive any special arguments and behaves like an ordinary function but can be called directly on the class.

:p How do classmethods and staticmethods differ?
??x
Classmethods are used when you need to access or modify the class state in some way. They take the class as their first argument (`cls`), making them suitable for creating alternative constructors, factory methods, etc. On the other hand, staticmethods do not receive any special arguments; they behave like regular functions but can be defined within a class body. Staticmethods are useful when you need to perform some utility task that is related to the class but doesn't depend on its state.

Example code:
```python
class Demo:
    @classmethod
    def klassmeth(*args):
        return args

    @staticmethod
    def statmeth(*args):
        return args

# Usage examples
print(Demo.klassmeth())      # (<class '__main__.Demo'>,)
print(Demo.klassmeth('spam'))  # (<class '__main__.Demo'>, 'spam')
print(Demo.statmeth())       # ()
print(Demo.statmeth('spam'))  # ('spam',)
```
x??

---

#### Usage of `memoryview` and Typecode
Background context: In the example provided, a typecode is extracted from the first byte of the input sequence. This typecode (e.g., 'f' for float) is used to cast the memoryview object into a format that can be passed to the constructor of `Vector2d`. The use of `memoryview` allows efficient handling and manipulation of binary data.

:p What role does `memoryview` play in the `frombytes` method?
??x
The `memoryview` is used to efficiently access and manipulate the byte sequence. After extracting the typecode from the first byte, a memoryview object is created over the remaining bytes. This memoryview is then cast according to the typecode, allowing it to be unpacked directly into the arguments needed by the constructor of `Vector2d`.

Example code:
```python
octets = b'\x01\x40\x80\x00\x00\x00\x00\x3f'  # Example bytes for Vector2d
typecode = chr(octets[0])  # 'f'
memv = memoryview(octets[1:]).cast(typecode)  # Memoryview of float values
```
x??

#### f-strings, format() and str.format()
Background context: The provided text explains how to use formatted strings (f-strings), the built-in `format()` function, and the string method `.format()` in Python. These methods are used for formatting output according to specified formats.

The `format_spec` is a part of the formatting specifier that controls the formatting behavior of the data being printed or displayed.
:p What is the format_spec in formatted strings?
??x
Format_spec is a part of the formatting specifier and it defines how the value should be formatted. It can include details such as precision, alignment, padding, and more.

For example:
```python
print(f"Number: {3.14159:.2f}")  # Number: 3.14
```
x??

---

#### Formatting Specifier in f-strings
Background context: The formatting specifier is a key component of formatted strings (f-strings) and the `format()` method. It determines how the value should be displayed.

The format_spec can include various characters such as precision, width, padding, etc.
:p What does the format_spec do in an f-string?
??x
The format_spec in an f-string defines the details of how to format a specific part of the output. This includes things like the number of decimal places, alignment, padding, and other formatting options.

For example:
```python
print(f"Value: {3.14159:08.2f}")  # Value: 0003.14
```
x??

---

#### Field Name in Replacement Fields
Background context: The replacement field syntax used in f-strings and `str.format()` includes a 'field_name' to the left of the colon, which can be an arbitrary expression. This is distinct from the format_spec to the right of the colon.

:p What is the purpose of the field_name part of the replacement field?
??x
The field_name part in the replacement field syntax serves as a placeholder for keyword arguments or positional indices used in the formatting function call. It can be an arbitrary expression that evaluates to a valid key or index, which corresponds to a value from the `format()` method's argument list.

For example:
```python
data = {"rate": 0.20746887966804978}
print(f"1 BRL = {data['rate']:0.2f} USD")  # 1 BRL = 0.21 USD
```
x??

---

#### Format Specification Mini-Language
Background context: The `__format__` method in Python allows classes to define their own formatting behavior using the format_spec argument. The `Format Specification Mini-Language` is a way to specify how an object should be formatted.

:p What is the Format Specification Mini-Language?
??x
The Format Specification Mini-Language is a set of rules that allow specifying how data should be formatted, such as precision, alignment, padding, and more. Each class can interpret the format_spec argument in its own way.

For example:
```python
class Currency:
    def __format__(self, format_spec):
        if 'f' in format_spec:  # Fixed-point notation
            return f"{self:.2f}"
        elif '%' in format_spec:  # Percentage notation
            return f"{self * 100}%"
brl = Currency()
print(f"Rate: {brl:0.4f}")  # Rate: 0.2075
```
x??

---

#### Built-in Types and Format Specification Mini-Language
Background context: The `Format Specification Mini-Language` can be extended to include custom formatting for built-in types such as `int` and `float`. For example, `int` supports base 2 and base 16 output with 'b' and 'x', while `float` supports fixed-point display with 'f' and percentage display.

:p What are some examples of format_specifiers for built-in types?
??x
Some examples of format_specifiers for built-in types include:

- For `int`: 
  - `'b'` for base 2 (binary)
  - `'x'` for base 16 (hexadecimal)

For example:
```python
print(format(42, 'b'))  # 101010
```

- For `float`: 
  - `'f'` for fixed-point display
  - `'%'` for percentage

For example:
```python
print(format(2 / 3, '.1 percent'))  # 66.7 percent
```
x??

---

#### Custom Format Specification for Vector2d
Background context: The `Vector2d` class needs a custom format method to handle both Cartesian and polar coordinate representations. This involves using Python's built-in `format()` function and understanding how it works with different format specifiers.

:p How does the `__format__` method in `Vector2d` work for displaying vector components?
??x
The `__format__` method formats each component of a `Vector2d` instance according to the provided format specifier. If no specifier is given, it defaults to formatting as Cartesian coordinates. For polar coordinates, if the specifier ends with 'p', it removes the 'p' and uses `abs(self)` for magnitude and `self.angle()` for the angle in radians.

Code example:
```python
import math

class Vector2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('p'):
            fmt_spec = fmt_spec[:-1]  # Remove 'p' from the specifier
            coords = (abs(self), self.angle())  # Get magnitude and angle for polar coordinates
            outer_fmt = '<{}, {}> '.format(*coords)  # Format as <r, θ>
        else:
            components = (self.x, self.y)
            outer_fmt = '({}, {}) '.format(*components)  # Default to Cartesian format
        return outer_fmt
    
    def angle(self):
        return math.atan2(self.y, self.x)  # Calculate the angle in radians

# Example usage
v1 = Vector2d(3, 4)
print(format(v1))  # (3.0, 4.0)
print(format(v1, '.2f'))  # (3.00, 4.00)
print(format(v1, '.3e'))  # (3.000e+00, 4.000e+00)
print(format(v1, 'p'))  # <5.0, 0.9272952180016122>
```
x??

---
#### Handling Different Format Specifiers
Background context: The `Vector2d` class uses Python's `format()` method to handle different format specifiers for both Cartesian and polar coordinates.

:p What happens when a non-empty format specifier is passed to the `__format__` method?
??x
When a non-empty format specifier is passed, it checks if the specifier ends with 'p'. If so, it formats the vector in polar coordinates. Otherwise, it uses the default Cartesian coordinate formatting.

Code example:
```python
v1 = Vector2d(3, 4)
print(format(v1, '.2f'))  # (3.00, 4.00)
```
x??

---
#### Implementation of `__format__` for Polar Coordinates
Background context: The `Vector2d` class needs to support polar coordinate formatting when the format specifier ends with 'p'. This requires calculating the magnitude and angle.

:p How does the `__format__` method handle polar coordinates?
??x
The `__format__` method checks if the format specifier ends with 'p'. If so, it removes this character and calculates the magnitude using `abs(self)` and the angle using `self.angle()`. It then formats these values as a tuple in the form `<r, θ>`.

Code example:
```python
import math

class Vector2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('p'):
            fmt_spec = fmt_spec[:-1]  # Remove 'p' from the specifier
            coords = (abs(self), self.angle())  # Get magnitude and angle for polar coordinates
            outer_fmt = '<{}, {}> '.format(*coords)  # Format as <r, θ>
        else:
            components = (self.x, self.y)
            outer_fmt = '({}, {}) '.format(*components)  # Default to Cartesian format
        return outer_fmt
    
    def angle(self):
        return math.atan2(self.y, self.x)  # Calculate the angle in radians

# Example usage
v1 = Vector2d(3, 4)
print(format(v1, 'p'))  # <5.0, 0.9272952180016122>
```
x??

---
#### Default Cartesian Coordinate Formatting
Background context: The `Vector2d` class uses default Cartesian coordinate formatting when no specific format specifier is provided.

:p What happens if no format specifier is passed to the `__format__` method?
??x
If no format specifier is passed, the `__format__` method defaults to Cartesian coordinate formatting. It simply returns a tuple containing the x and y components of the vector in string form.

Code example:
```python
v1 = Vector2d(3, 4)
print(format(v1))  # (3.0, 4.0)
```
x??

---

#### Making a Vector2d Hashable

In the provided context, the `Vector2d` class needs to be made hashable so that instances can be used as set elements or dictionary keys. This requires implementing both the `__hash__` and `__eq__` methods.

The current implementation allows modification of vector components directly:

```python
v1 = Vector2d(3, 4)
v1.x = 7  # Direct modification is possible, leading to unhashable objects.
```

To prevent this, the class needs to be made immutable. The `x` and `y` attributes should now be read-only properties.

:p How do you make a `Vector2d` instance immutable?
??x
By defining `x` and `y` as read-only properties using the `@property` decorator. This prevents direct modification of these attributes.
```python
class Vector2d:
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
```
x??

---

#### Implementing the `__hash__` Method

For a `Vector2d` instance to be hashable, we need to implement the `__hash__` method. This method should return an integer and ideally use the hashes of the object attributes used in the `__eq__` method.

The `__hash__` method documentation suggests using the bitwise XOR operator (`^`) to mix the hashes of the components.

:p What is the implementation of the `__hash__` method for `Vector2d`?
??x
The implementation involves returning the bitwise XOR of the hashes of `x` and `y`.

```python
def __hash__(self):
    return hash(self.x) ^ hash(self.y)
```
This ensures that objects with equal attributes have the same hash value, allowing them to be used in sets or as dictionary keys.
x??

---

#### Formatting Vector2d for User-Defined Types

The `Vector2d` class can also benefit from custom formatting. The provided text shows examples of how to format `Vector2d` instances using Python's formatted string literals (f-strings).

:p How do you format a `Vector2d` instance with angle brackets and rectangular coordinates?
??x
To format a `Vector2d` instance with angle brackets for outer format, you can use the following code:

```python
format(Vector2d(1, 1), '<{:.6f}, {:.3f}')
```

This will output:
```
'<1.4142135623730951, 0.7853981633974483>'
```

For rectangular coordinates with parentheses and a different format:

```python
format(Vector2d(1, 1), '({:.3ep}, {:.3ep})')
```

This will output:
```
'(1.414e+00, 7.854e-01)'
```
x??

---

#### Vector2d Class with Read-Only Properties

To make `Vector2d` instances immutable and hashable, the class needs to define read-only properties for `x` and `y`.

:p How do you define read-only properties in a Python class?
??x
By using the `@property` decorator. This allows defining methods that can be accessed like attributes but are not directly modifiable.

```python
class Vector2d:
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
```

These properties act as read-only views into the private attributes `__x` and `__y`.
x??

---

#### Vector2d Class with Hashable Instances

The `Vector2d` class needs to be made hashable for use in sets or dictionaries. This involves implementing both `__hash__` and `__eq__`.

:p How do you implement the `__hash__` method to ensure `Vector2d` instances are hashable?
??x
To make a `Vector2d` instance hashable, you need to define the `__hash__` method. This should return an integer that is consistent with the object's identity.

```python
def __hash__(self):
    return hash(self.x) ^ hash(self.y)
```

This uses the bitwise XOR operator (`^`) on the hashes of `x` and `y`, ensuring that equal vectors have the same hash value.
x??

---

#### Hashable Vectors
Background context: In Python, to make an object hashable (i.e., suitable for use as a dictionary key or set element), you need to define both `__hash__` and `__eq__` methods. The `__hash__` method returns the hash value of an instance, which is used by dictionaries and sets. It should be consistent across all instances that are considered equal.
:p What would be the expected behavior when using a hashable vector in a set?
??x
When you use a hashable vector in a set, Python will check for equality based on `__eq__` before adding or looking up elements. If two vectors are considered equal by their `__eq__` method, they will occupy the same slot in the set.
```python
v1 = Vector2d(3, 4)
v2 = Vector2d(3.0, 4.0)
print(set([v1, v2]))  # {Vector2d(x=3.0, y=4.0)}
```
x??

---

#### Read-Only Properties
Background context: Implementing read-only properties ensures that the value of an instance attribute cannot be changed once it is set. This is useful for creating immutable objects or ensuring data integrity.
:p How does implementing `__hash__` and `__eq__` correctly contribute to making a vector hashable?
??x
Implementing `__hash__` and `__eq__` methods allows vectors to be used as keys in dictionaries or elements in sets. The `__hash__` method provides the hash value, which is consistent for instances considered equal by `__eq__`. This ensures that if two vectors are considered equivalent, they will have the same hash value.
```python
class Vector2d:
    def __init__(self, x, y):
        self._x = x
        self._y = y
    
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y
    
    def __hash__(self):
        # Combining hash values to make it consistent and unique
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if isinstance(other, Vector2d):
            return (self.x == other.x) and (self.y == other.y)
```
x??

---

#### Positional Patterns in `match` Expression
Background context: Python 3.10 introduced the `match` statement for pattern matching. By default, positional patterns require exact matches with the constructor arguments. However, you can enable positional matching by defining a class attribute named `__match_args__`.
:p How do you make a Vector2d compatible with positional match patterns?
??x
To support positional patterns in the `match` expression, you need to define the `__match_args__` class attribute listing the instance attributes used for pattern matching.
```python
class Vector2d:
    __match_args__ = ('x', 'y')
    
    def __init__(self, x, y):
        self._x = x
        self._y = y

    # Other methods...

# Example usage with match statement
def positional_pattern_demo(v: Vector2d) -> None:
    match v:
        case Vector2d(0, 0):  # Match when both coordinates are zero
            print(f'{v.r} is null')
        case Vector2d(0):     # Match when only the x-coordinate is zero
            print(f'{v.r} is vertical')
        case Vector2d(_, 0):  # Match when only the y-coordinate is zero
            print(f'{v.r} is horizontal')
```
x??

---

