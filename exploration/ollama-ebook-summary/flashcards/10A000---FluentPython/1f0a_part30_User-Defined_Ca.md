# Flashcards: 10A000---FluentPython_processed (Part 30)

**Starting Chapter:** User-Defined Callable Types

---

#### Callable Objects and Instance Methods
Callable objects are Python objects that can be called like functions. To make an instance callable, one needs to define a `__call__` method within its class definition. This allows instances of the class to behave as if they were actual functions.

:p How does making a class instance callable work in Python?
??x
To make a class instance callable, you need to implement the `__call__` method within the class. This method should contain the logic that will be executed when the instance is called like a function. For example:

```python
class BingoCage:
    def __init__(self, items):
        self._items = list(items)
        random.shuffle(self._items)

    def pick(self):
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCage')

    def __call__(self):  # This makes the instance callable
        return self.pick()
```

In this example, `bingo = BingoCage(range(3))` creates an instance of `BingoCage`, and calling `bingo()` invokes its `__call__` method.

x??

---

#### Example: BingoCage Class Implementation
The `BingoCage` class is implemented to manage a shuffled list of items. It ensures that the internal state (list) contains unique elements from an input iterable by making a local copy during initialization. The `shuffle` function guarantees randomness, and the `pick` method pops an item from this list.

:p What does the `BingoCage` class do?
??x
The `BingoCage` class manages a shuffled list of items that can be picked randomly. It accepts any iterable during initialization, creates a local copy to avoid side effects on the input, shuffles the copied list for randomness, and allows picking one item at a time.

Here is an example implementation:

```python
import random

class BingoCage:
    def __init__(self, items):
        self._items = list(items)
        random.shuffle(self._items)  # Shuffles the internal list

    def pick(self):  # Picks and removes one item from the list
        try:
            return self._items.pop()
        except IndexError:
            raise LookupError('pick from empty BingoCage')

    def __call__(self):  # Makes the instance callable like a function
        return self.pick()
```

:p How does `BingoCage` ensure that the input iterable is not modified?
??x
To prevent modifying the original input iterable, the `__init__` method of the `BingoCage` class makes a local copy of the items during initialization. This ensures that any changes made to the instance do not affect the external list passed as an argument.

```python
class BingoCage:
    def __init__(self, items):
        self._items = list(items)  # Creates a local copy
```

:p How does `BingoCage` handle an empty list when calling its pick method?
??x
The `BingoCage` class handles the case of an empty list by raising a `LookupError`. When the `_items` list is empty and the `pick` method is called, it will raise this exception with the message 'pick from empty BingoCage'.

```python
class BingoCage:
    def pick(self):
        try:
            return self._items.pop()
        except IndexError:  # Handles the case of an empty list
            raise LookupError('pick from empty BingoCage')
```

x??

---

#### Callable Classes and Method `__call__`
Callable classes are instances that can be called using the function call syntax. This is achieved by defining a `__call__` method within the class.

:p How does Python recognize an object as callable?
??x
Python recognizes an object as callable if it has a `__call__` method defined in its class definition. The `callable()` built-in function can be used to check whether an object is callable, returning `True` if it is and `False` otherwise.

```python
obj = abs  # An example of a callable object (built-in function)
print(callable(obj))  # Output: True

obj2 = 'string'  # Not a callable object
print(callable(obj2))  # Output: False
```

:p What is the `__call__` method used for in Python classes?
??x
The `__call__` method is a special method in Python that allows an instance of a class to be called like a function. When you call an instance of a class as if it were a function, the `__call__` method is executed.

Example:

```python
class Example:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        print(f"Called with args: {args} and kwargs: {kwargs}")

# Using the class instance as a function
ex = Example()
ex(1, 2, x=3, y=4)  # Output: Called with args: (1, 2) and kwargs: {'x': 3, 'y': 4}
```

:p How can you use `callable()` to check if an object is callable?
??x
You can use the built-in `callable()` function to determine whether an object is callable. It returns `True` if the object has a `__call__` method and `False` otherwise.

```python
obj = abs  # A built-in function, which is callable
print(callable(obj))  # Output: True

obj2 = 'string'  # Not a callable object
print(callable(obj2))  # Output: False
```

x??

---

#### Example: Using `callable()` to Check Callability
The `callable()` built-in function can be used to check whether an object is callable. It returns `True` if the object has a `__call__` method and `False` otherwise.

:p How does one use the `callable()` function in Python?
??x
To determine if an object is callable, you can use the `callable()` built-in function. This function takes an object as an argument and returns `True` if the object is callable (i.e., has a `__call__` method) and `False` otherwise.

Example:

```python
print(callable(abs))  # Output: True
print(callable(str))  # Output: True
print(callable('Ni.'))  # Output: False
```

:p How does the `BingoCage` class use `callable()`?
??x
The `BingoCage` class uses `callable()` indirectly through its instances. While the class itself is not checked for callability, an instance of `BingoCage` can be used as a callable object because it has implemented the `__call__` method.

```python
bingo = BingoCage(range(3))
print(callable(bingo))  # Output: True
```

x??

---

#### Decorators and Callable Objects
Decorators are functions that modify or extend the behavior of other functions. They must be callable, and sometimes it is useful to remember state between calls (e.g., for memoization).

:p How does `__call__` relate to decorators?
??x
The `__call__` method in a class instance can make an object callable like a function, which is essential for implementing decorators. Decorators are themselves functions that modify the behavior of other functions and must be callable. By defining `__call__`, you allow instances of your decorator class to act as if they were normal functions.

Example:

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()  # Output: Something is happening before the function is called. Hello! Something is happening after the function is called.
```

:p How can `__call__` be used in decorators?
??x
The `__call__` method can be used to implement a decorator's behavior that needs to remember state across multiple calls. For example, memoization involves caching results of expensive function calls and reusing them when the same inputs occur again.

Example:

```python
class Memoize:
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

@Memoize
def expensive_function(x):
    print(f"Computing {x}")
    return x ** 2

print(expensive_function(4))  # Output: Computing 4 16
print(expensive_function(4))  # No "Computing" message, uses cached value
```

:x??

#### Positional and Keyword Parameters in Python Functions
Python functions offer a flexible parameter handling mechanism that allows for both positional and keyword arguments. This flexibility is achieved through the use of *args and **kwargs, which unpack iterables and mappings into separate arguments when calling a function.

:p How can you define parameters in a Python function to allow multiple positional arguments?
??x
You can use `*args` to capture any number of additional positional arguments as a tuple. For example:

```python
def tag(name, *content):
    # code here
```

This allows the `tag` function to accept an indefinite number of content elements as separate positional arguments.

x??

---

#### Keyword-Only Parameters in Python Functions
Keyword-only parameters are a feature introduced in Python 3 that enforces certain parameters to be passed only as keyword arguments. This is useful for ensuring clarity and preventing confusion with reserved keywords or parameter order.

:p How do you define a function with keyword-only parameters?
??x
You can use the `*` symbol before the parameter name in the function definition to indicate that all subsequent parameters must be specified by their names (keyword arguments). For example:

```python
def tag(name, *content, class_=None):
    # code here
```

In this case, `class_` is a keyword-only parameter and can only be passed as a named argument.

x??

---

#### Using **kwargs in Python Functions
The double asterisks (`**`) are used to capture any additional keyword arguments as a dictionary. This allows the function to accept an indefinite number of keyword arguments beyond those explicitly defined in its signature.

:p How do you use `**kwargs` in defining and calling a Python function?
??x
When defining a function, you can use `**kwargs` to allow it to accept any additional keyword arguments as a dictionary. For example:

```python
def tag(name, *content, **attrs):
    # code here
```

This allows the `tag` function to accept an indefinite number of attributes beyond those explicitly defined in its signature.

x??

---

#### Closures and Decorators (Chapter 9)
Closures are functions that have access to variables from their lexical scope even when they are executed outside that scope. Decorators, a special case of closures, modify or wrap the functionality of another function without permanently modifying it.

:p What is a closure in Python?
??x
A closure in Python is a function object that remembers values in enclosing scopes even if they are not present in memory. Closures can access variables from parent functions, which makes them very powerful for encapsulating state and behavior together.

Example:
```python
def outer_function(x):
    def inner_function(y):
        return x + y
    return inner_function

closure_example = outer_function(10)
print(closure_example(5))  # Output: 15
```

In this example, `inner_function` is a closure because it retains access to the variable `x` from its parent function even though `outer_function` has finished execution.

x??

---

#### Unpacking Iterables and Mappings in Python Functions
The single asterisk (`*`) before an argument in a function call unpacks an iterable into positional arguments. The double asterisks (`**`) unpack a mapping (like a dictionary) into keyword arguments.

:p How do you use `*` to unpack iterables when calling a Python function?
??x
You can use the single asterisk (`*`) before an argument in a function call to unpack an iterable and pass its elements as separate positional arguments. For example:

```python
def concat(*args, sep="/"):
    return sep.join(args)

print(concat("earth", "mars", "venus"))  # Output: earth/mars/venus
```

In this case, the `*args` in the function definition allows multiple arguments to be passed as a tuple, and they are unpacked into separate positional arguments when calling the function.

x??

---

#### Variable Positional Arguments (*args)
Background context: In Python, `*args` allows a function to accept a variable number of positional arguments. This is useful when you're not sure how many arguments will be passed to your function or if you want to pass multiple values as a single argument.

:p How do you define and use `*args` in a Python function?
??x
In Python, `*args` allows a function to accept a variable number of positional arguments. These are stored as a tuple within the function. Here is an example:
```python
def my_function(*args):
    for arg in args:
        print(arg)

# Calling the function with different numbers of arguments
my_function(1, 2, 3) # Output: 1, 2, 3
```
x??

---

#### Keyword-Only Arguments
Background context: Python allows defining functions where some or all parameters must be specified by keyword. This is useful for making sure that certain arguments are always provided with their intended names, preventing confusion.

:p How do you define a function in Python to accept keyword-only arguments?
??x
To define a function that accepts only keyword arguments after the `*` parameter, place a single `*` before the first parameter. Here is an example:
```python
def my_function(a, *, b):
    print(f"a: {a}, b: {b}")

# Valid call
my_function(1, b=2)

# Invalid calls
my_function(1, 2) # Raises TypeError
my_function(b=2) # Missing positional argument 'a'
```
x??

---

#### Positional-Only Parameters (/)
Background context: Starting from Python 3.8, user-defined functions can have positional-only parameters using the `/` syntax in the parameter list. This means that certain arguments must be passed positionally and not by keyword.

:p How do you define a function with positional-only parameters?
??x
Starting from Python 3.8, you can define a function with positional-only parameters using the `/` token. Here is an example of how to emulate the `divmod` built-in function:
```python
def divmod(a, b, /):
    return (a // b, a % b)

# Valid call
result = divmod(10, 4)
print(result) # Output: (2, 2)

# Invalid calls
divmod(a=10, b=4) # Raises TypeError
```
x??

---

