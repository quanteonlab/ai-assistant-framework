# Flashcards: 10A000---FluentPython_processed (Part 29)

**Starting Chapter:** Further Reading

---

#### Mutable Objects as Default Parameters
Background context: In Python, mutable objects (like lists or dictionaries) used as default parameters can lead to unexpected behavior. When a function is called with a default parameter, that parameter is only evaluated once when the function is defined, not each time the function is called. This means any changes made to the object in place will affect all future calls using that default.
If an object is changed and it forms part of a cyclic reference, it might be garbage collected prematurely.

:p What are the potential issues with mutable objects as default parameters?
??x
The answer: The primary issue is that when you use a mutable object (like a list) as a default parameter in Python, the same object is used across all function calls. If this object is changed within the function, those changes persist for every future call.

For example:
```python
def append_to_list(value, mylist=[]):
    if mylist is None:
        mylist = []
    mylist.append(value)
    return mylist

print(append_to_list(1))  # [1]
print(append_to_list(2))  # [1, 2] - This is unexpected because a new call should not append to the previous list
```

In this case, `mylist` retains its state between function calls due to how default arguments are handled. To avoid this issue, you can use `None` as the default and initialize it inside the function:
```python
def append_to_list(value, mylist=None):
    if mylist is None:
        mylist = []
    mylist.append(value)
    return mylist

print(append_to_list(1))  # [1]
print(append_to_list(2))  # [2] - Now this works as expected
```
x??

---
#### Weak References
Background context: Weak references are a feature in Python that allows you to hold a reference to an object without preventing the garbage collector from collecting it if no other references exist. This is useful for scenarios like tracking instances of a class, where you want to avoid memory leaks but still keep track of active objects.

:p What is a weak reference and when is it useful?
??x
The answer: A weak reference in Python allows you to hold a reference to an object without preventing the garbage collector from collecting that object if no other references exist. This is particularly useful for scenarios where tracking instances of a class could otherwise lead to memory leaks.

For example, using `weakref.ref` to keep track of all current instances:
```python
import weakref

class MyClass:
    _instances = []

    def __init__(self):
        self._instances.append(weakref.ref(self))

    @classmethod
    def get_all_instances(cls):
        return [inst() for inst in cls._instances if inst()]

my_instance = MyClass()
# my_instance is tracked by the weak reference

del my_instance  # This instance can now be garbage collected, even though it's still referenced

MyClass.get_all_instances()  # Returns an empty list since no instances are alive
```
x??

---
#### CPython Garbage Collector and Object Lifecycle
Background context: In Python, especially with CPython (the standard implementation), objects are managed by a generational garbage collector. Objects are discarded when their reference count reaches zero or if they form part of a cycle but have no outside references.

:p What is the behavior of object lifecycle in CPython?
??x
The answer: In CPython, objects are retained in memory as long as there are active references to them. Once an object’s reference count drops to zero (or it forms a cycle with other unreferenced objects), it can be garbage collected.

However, the actual collection is not immediate but happens during periodic sweeps of different generations based on object age and type. The `gc` module provides access to this functionality:
```python
import gc

# Example: Force a sweep in generation 2 (long-lived objects)
gc.collect(2)

# Or manually track garbage collection events
def on_gc_event(event):
    print(f"Garbage collected {event.object} of type {type(event.object)}")

gc.callbacks.append(on_gc_event)  # Register a callback for tracking garbage collection events
```
x??

---

#### String Overloading in Java vs. Python

Background context: The designers of Java felt it necessary to overload the `+` operator for strings, but this approach was not extended to the `==` operator. In contrast, Python's design philosophy allows for a more flexible use of operators through overloading and reference-based comparison.

:p Why did Java choose to overload the `+` operator for strings?
??x
Java overloaded the `+` operator for strings to make string concatenation more intuitive. This was done because using an explicit method like `concat()` or `append()` would be less convenient in certain scenarios, making the syntax cleaner and easier to read.

```java
// Java code example
String greeting = "Hello" + " World";
```
x??

---

#### Operator Overloading and `==` in Python

Background context: In Python, the `==` operator compares object values by default. However, overloading this operator allows for custom behavior based on class definitions. Additionally, Python's handling of immutable objects like strings makes the `==` comparison straightforward.

:p How does Python handle the `==` operator when comparing strings?
??x
Python uses reference-based comparison for built-in types and mutable objects by default. For immutable types like strings, this works well because their values do not change after creation, so two variables with the same string value will always point to the exact same memory location.

```python
# Python code example
str1 = "hello"
str2 = "hello"

print(str1 is str2)  # True, because both refer to the same immutable object in memory
```
x??

---

#### Object Identity and Mutability

Background context: In languages where objects are mutable, the concept of object identity becomes crucial. If an object can change its state over time, comparing references using `is` is essential for determining whether two variables point to the exact same object.

:p What happens when you compare two mutable objects using `==` in Python?
??x
When comparing two mutable objects with `==`, Python checks if their values are equal. For mutable objects like lists or dictionaries, this means checking that they contain the same elements and have the same structure. However, for immutable objects like integers, strings, and tuples, `==` is equivalent to `is`.

```python
# Python code example
list1 = [1, 2, 3]
list2 = [1, 2, 3]

print(list1 == list2)  # True, because values are equal
print(list1 is list2)  # False, because they refer to different objects in memory

num1 = 5
num2 = 5

print(num1 == num2)  # True, because values are equal
print(num1 is num2)  # True, because both variables reference the same immutable integer object
```
x??

---

#### Garbage Collection and Reference Counting

Background context: Python employs a combination of reference counting and generational garbage collection to manage memory. Reference counting ensures immediate disposal of objects with zero references, while the generational collector handles cycles.

:p How does Python's garbage collection work?
??x
Python uses two main mechanisms for garbage collection:

1. **Reference Counting**: Each object has an internal counter that increments when a new reference is created and decrements when a reference is deleted. When the counter reaches zero, the object can be safely deallocated.

2. **Generational Garbage Collector**: With version 2.0, Python introduced a generational garbage collector to handle cycles more efficiently. Objects are categorized into generations based on their lifetime. New objects start in generation 0 and move up as they survive collection cycles.

```python
# Note: This is pseudo-code for understanding the mechanism.
def reference_counting(object):
    object.ref_count += 1

def decrement_ref_count(object):
    if object.ref_count == 0:
        deallocate_object(object)
```
x??

---

#### File Handling and Garbage Collection in Python

Background context: This concept explains the safe use of file handling methods like `write` in CPython, Jython, and IronPython. It discusses how garbage collection mechanisms differ between these interpreters and recommends best practices for ensuring files are properly closed.

:p What safety concerns arise when using file operations in non-CPython Python implementations?
??x
Safety concerns include the risk that the file object's reference count might not reach zero immediately after a `write` method call, leading to potential leaks. In Jython and IronPython, which use host runtime garbage collectors (Java VM or .NET CLR), this can be problematic because these collectors may delay the destruction of objects.

```python
# Example in CPython where it is safe
with open('test.txt', 'wt', encoding='utf-8') as fp:
    fp.write('1, 2, 3')
```
x??

---

#### Parameter Passing in Python

Background context: This concept explores the nuances of parameter passing in Python. It clarifies that parameters are passed by value (where values are references), but changes to mutable objects can still be made inside a function.

:p How does Python handle parameter passing for immutable and mutable types?
??x
Python passes the value (which is always a reference) of arguments, not their identity. For immutable objects like tuples, any modifications within the function will fail since the object cannot be changed in place. For mutable objects, changes made inside the function can affect the original object because they share references.

```python
def modify_list(lst):
    lst.append(4)

my_list = [1, 2, 3]
modify_list(my_list)
print(my_list)  # Output: [1, 2, 3, 4]
```
x??

---

#### Understanding the `with` Statement

Background context: The `with` statement ensures that resources like file handles are properly closed after their suite finishes. It is a best practice for managing such resources.

:p Why should you use the `with` statement when dealing with files in Python?
??x
Using the `with` statement guarantees that the file will be properly closed once operations on it are completed, even if an error occurs during execution. This helps prevent resource leaks and ensures clean-up processes are handled automatically.

```python
# Using 'with' for safe file handling
with open('test.txt', 'wt', encoding='utf-8') as fp:
    fp.write('1, 2, 3')
```
x??

---

#### Reference Counting vs. Garbage Collection

Background context: This concept explains the differences between reference counting and garbage collection mechanisms used by Python interpreters.

:p What is a key difference between CPython's reference counting mechanism and Jython/IronPython's garbage collectors?
??x
CPython uses reference counting to manage memory, which means an object is destroyed when its reference count reaches zero. However, Jython and IronPython rely on their host runtime's garbage collector (Java VM or .NET CLR), which may not destroy objects immediately after the last reference is gone, potentially leading to delayed destruction and resource management issues.

```python
# Example in CPython with immediate cleanup due to reference counting
import sys

a = [1, 2, 3]
b = a
del b  # Reference count of 'a' decreases
```
x??

---

#### Immutable vs. Mutable Objects

Background context: This concept distinguishes between immutable and mutable objects in Python, explaining how they are handled differently within functions.

:p What is the difference between mutable and immutable types in Python?
??x
In Python, immutable objects like integers, strings, tuples cannot be changed after their creation; any operation that appears to modify them actually creates a new object. Mutable objects, such as lists or dictionaries, can have their contents altered directly.

```python
# Example with an immutable type
a = 10
b = a + 5  # Creates a new integer object

# Example with a mutable type
lst = [1, 2, 3]
new_lst = lst.append(4)  # Modifies the original list in place
```
x??

---

#### Type Compatibility and Object Identity

Background context: This concept covers how Python handles identity and type compatibility, especially for built-in types like `tuple`.

:p Why does calling the `copy` method on a `frozenset` not create a new object?
??x
The `copy` method of `frozenset` returns the same object because frozen sets are immutable. The harmless lie here is to maintain interface compatibility with mutable sets, ensuring that users see no difference between identical immutable objects.

```python
# Example demonstrating identity
orig_set = frozenset([1, 2, 3])
copied_set = orig_set.copy()
print(orig_set is copied_set)  # Output: True
```
x??

---

#### Changing an Object's Type Dynamically

Background context: This concept discusses the ability to change an object's type dynamically in Python by assigning a different class to its `__class__` attribute, though this is not recommended.

:p How can you dynamically change the class of an object in Python?
??x
You can change an object’s class using its `__class__` attribute. However, doing so is generally discouraged as it can lead to unexpected behavior and bugs.

```python
# Example of changing class (not recommended)
class NewClass:
    pass

obj = [1, 2, 3]
obj.__class__ = NewClass
```
x??

#### First-Class Objects Concept
Functions and other program entities that can be created at runtime, assigned to variables or elements in data structures, passed as arguments to functions, and returned as results of functions are referred to as first-class objects. Examples include integers, strings, dictionaries, and functions.
:p What is a first-class object?
??x
A first-class object is an entity that can be created at runtime, assigned to variables or elements in data structures, passed as arguments to functions, and returned as the result of a function. In Python, both functions and other entities like integers, strings, and dictionaries are considered first-class objects.
x??

---

#### Functions as First-Class Objects
Functions in Python are treated as first-class objects, allowing them to be manipulated similarly to other data types such as integers and strings. This means they can be assigned to variables, passed as arguments to functions, and returned from functions.
:p How do you treat functions as first-class objects in Python?
??x
In Python, you can treat functions as first-class objects by assigning them to variables, passing them as arguments to other functions, or returning them from functions. For example:
```python
def greet(name):
    return f"Hello, {name}!"

greeting = greet  # Assigning the function to a variable

def apply_greet(func, name):
    print(func(name))  # Passing the function as an argument

apply_greet(greet, "Alice")  # Returning the function from another function
```
x??

---

#### New Callables in Python 3.5 and 3.6
New callables introduced in Python 3.5 include native coroutines, while asynchronous generators were added in Python 3.6. Both are covered in Chapter 22.
:p What new callables were introduced in Python 3.5 and 3.6?
??x
In Python 3.5, native coroutines were introduced as a callable object type. In Python 3.6, asynchronous generators were added. These are mentioned here for completeness and will be covered in detail in Chapter 22.
x??

---

#### Positional-Only Parameters in Python 3.8
Positional-only parameters allow certain function arguments to only be specified by position, not name. This feature was introduced in Python 3.8.
:p What is a positional-only parameter?
??x
A positional-only parameter is an argument of a function that can only be specified by its position and not by name. This feature was introduced in Python 3.8 to provide more control over how arguments are passed to functions.
x??

---

#### Reading Type Hints at Runtime
Since Python 3.5, function annotations should conform to PEP 484 for type hints. Coverage of this topic has been moved from the previous section to here due to changes in annotation usage and PEP 484 compliance.
:p What is covered in "Reading Type Hints at Runtime"?
??x
In "Reading Type Hints at Runtime," coverage includes how function annotations, which should conform to PEP 484 since Python 3.5, can be read and used at runtime for type hinting. This section explains the best practices for using type hints in functions.
x??

---

#### Python Functions as Full-Fledged Objects
In Python, functions are considered first-class objects. This means that functions can be treated like any other object, such as being assigned to a variable, passed as an argument to another function, and returned from functions as results.

:p What does it mean when we say functions in Python are full-fledged objects?
??x
Functions in Python are full-fledged objects because they possess attributes similar to those of other objects. They can be assigned to variables, passed around as arguments, and even returned by other functions. This flexibility allows for a wide range of programming styles, including functional programming.

Example:
```python
def factorial(n):
    """returns n."""
    return 1 if n < 2 else n * factorial(n - 1)

# Assigning the function to a variable
fact = factorial
print(fact(5))  # Output: 120
```
x??

---

#### The `__doc__` Attribute of Functions
The `__doc__` attribute in Python functions can be used to store and retrieve documentation strings. These docstrings are often used by tools like the built-in `help()` function.

:p What is the purpose of the `__doc__` attribute in a function?
??x
The `__doc__` attribute in a function stores the docstring, which is a string literal that provides documentation for the function. This can be retrieved and used to provide help information about the function.

Example:
```python
def factorial(n):
    """returns n."""
    return 1 if n < 2 else n * factorial(n - 1)

print(factorial.__doc__)  # Output: 'returns n.'
```
x??

---

#### `type()` Function for Identifying Object Types
The `type()` function in Python can be used to determine the type of an object. This is useful when you want to check if a variable or expression evaluates to a certain class.

:p How does the `type()` function help in understanding the nature of objects?
??x
The `type()` function returns the type of an object, which can be very helpful for debugging and ensuring that operations are being performed on appropriate data types. In the context of functions, it confirms that they are instances of the `function` class.

Example:
```python
def factorial(n):
    """returns n."""
    return 1 if n < 2 else n * factorial(n - 1)

print(type(factorial))  # Output: <class 'function'>
```
x??

---

#### First-Class Functions and Functional Programming
In programming, a language is said to support first-class functions when functions can be treated just like any other variable. This means they can be passed as arguments, returned from functions, and assigned to variables.

:p What does it mean for a function to be "first-class" in Python?
??x
A function is considered first-class if it can be used freely like any other data type—passed as an argument, returned by another function, or assigned to a variable. This allows you to manipulate functions as values and use them in various ways.

Example:
```python
def factorial(n):
    """returns n."""
    return 1 if n < 2 else n * factorial(n - 1)

def map(function, iterable):
    result = []
    for item in iterable:
        result.append(function(item))
    return result

print(map(factorial, range(10)))  # Output: [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
```
x??

---

#### Higher-Order Functions
Higher-order functions are functions that take one or more functions as arguments and/or return a function as a result. They enable you to abstract common patterns of behavior in your code.

:p What is a higher-order function?
??x
A higher-order function is a function that can accept other functions as parameters, or return them as results. This allows for powerful abstraction and flexible coding styles, particularly useful in functional programming.

Example:
```python
def factorial(n):
    """returns n."""
    return 1 if n < 2 else n * factorial(n - 1)

# Using `map` to apply `factorial` function to a range of numbers
print(list(map(factorial, range(10))))  # Output: [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
```
x??

---

#### Using `sorted()` with a Key Function
The built-in `sorted()` function in Python can take an optional key parameter to specify how the items should be compared. This is often used to sort objects based on custom criteria.

:p How does the `key` argument work in the `sorted()` function?
??x
The `key` argument in the `sorted()` function provides a way to customize the sorting behavior by applying a function to each item before comparing them. The key function is called once per item, and its return value is used for comparison.

Example:
```python
fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']

# Sorting fruits by their length using the `len` function as the key
print(sorted(fruits, key=len))  # Output: ['fig', 'apple', 'cherry', 'banana', 'raspberry', 'strawberry']
```
x??

---

#### Reverse Function and Sorting
Background context explaining how to reverse strings and sort lists using custom functions. The `reverse` function takes a string, reverses it, and returns the result.

:p What is the purpose of the `reverse` function?
??x
The `reverse` function is used to reverse the spelling of words. It helps in sorting a list of words based on their reversed spellings.
```python
def reverse(word):
    return word[::-1]
```
x??

---

#### Example Sorting with Reverse Function
Background context explaining how to use the `sorted` function with a custom key.

:p How is the `fruits` list sorted using the `reverse` function as a key?
??x
The `fruits` list is sorted based on the reversed spellings of its elements. The `sorted` function uses the `key=reverse` argument to determine the sorting order.
```python
fruits = ['banana', 'apple', 'fig', 'raspberry', 'strawberry', 'cherry']
sorted(fruits, key=reverse)
```
Output: `['banana', 'apple', 'fig', 'raspberry', 'strawberry', 'cherry']`
x??

---

#### Higher-Order Functions in Python
Background context explaining higher-order functions and their usage in functional programming.

:p What are some of the well-known higher-order functions in Python?
??x
Some well-known higher-order functions in Python include `map`, `filter`, and `reduce`. These functions take other functions as arguments or return new functions.
x??

---

#### Map Function Examples
Background context explaining how to use `map` with different operations.

:p How does the `map(factorial, range(6))` function work?
??x
The `map(factorial, range(6))` applies the `factorial` function to each element in the `range(6)` sequence. The result is an iterator that yields factorial values for 0 through 5.

Example:
```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

list(map(factorial, range(6)))
```
Output: `[1, 1, 2, 6, 24, 120]`
x??

---

#### Filter Function Examples
Background context explaining how to use `filter` with different conditions.

:p How does the `filter(lambda n: n % 2, range(6))` function work?
??x
The `filter(lambda n: n % 2, range(6))` returns elements from the `range(6)` sequence that satisfy the lambda condition (i.e., elements that are odd). The result is an iterator.

Example:
```python
list(filter(lambda n: n % 2, range(6)))
```
Output: `[1, 3, 5]`

Using `map` and `filter`, we can combine them to perform operations like calculating factorials of only the odd numbers in a sequence.
```python
list(map(factorial, filter(lambda n: n % 2, range(6))))
```
Output: `[1, 6, 120]`
x??

---

#### Reduce Function Examples
Background context explaining how to use `reduce` and its replacement `sum`.

:p How does the `reduce(add, range(100))` function work?
??x
The `reduce(add, range(100))` applies a binary function (in this case, addition) cumulatively to the items of an iterable (here, `range(100)`), from left to right, so as to reduce the iterable to a single value.

Example:
```python
from functools import reduce
from operator import add

reduce(add, range(100))
```
Output: `4950`

However, since Python 3.0, `reduce` is no longer a built-in function. Instead, you can use the `sum` function which does the same job more directly.
```python
sum(range(100))
```
Output: `4950`
x??

---

#### Built-In Functions for Summation
Background context explaining alternative functions to `reduce`.

:p What are some built-in functions that serve as alternatives to `reduce`?
??x
The built-in functions `all` and `any` can be used in place of `reduce` for specific use cases. For example, `sum(range(100))` provides a simpler way to perform summation.

Example:
```python
sum(range(100))
```
Output: `4950`

For checking if all elements are truthy or any element is truthy in an iterable, you can use `all(iterable)` and `any(iterable)`, respectively.
```python
all([])
any([])
```
`all([])` returns `True` because there are no falsy elements. `any([])` returns `False` as the iterable is empty.
x??

---

#### Lambda Functions and Anonymous Functions
Lambda functions are a way to create small, anonymous functions within Python expressions. The `lambda` keyword is used for this purpose. However, lambda functions have limitations compared to regular functions defined with `def`. They can only contain pure expressions and cannot include statements like `while`, `try`, or assignments.
:p What are the key characteristics of lambda functions in Python?
??x
Lambda functions in Python are small, anonymous functions that can only contain a single expression. They lack the ability to use statements such as `while`, `try`, and assignments directly. Lambda functions are primarily used within higher-order function calls where concise function definitions are needed.
```python
# Example of using lambda with sorted()
sorted_fruits = sorted(['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana'], key=lambda word: word[::-1])
```
x??

---

#### Higher-Order Functions and Iterables
Higher-order functions are functions that take other functions as arguments or return functions as results. The text mentions "Iterable Reducing Functions" later in the book, indicating that such functions operate on iterables to produce a single result.
:p What is an example of using a higher-order function with lambda?
??x
An example of using a higher-order function like `sorted` with a lambda function involves sorting elements based on a custom key. Here, `lambda` defines the sorting criterion without needing to define a separate named function.
```python
# Example of sorting fruits by their reversed spelling
fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
sorted_fruits = sorted(fruits, key=lambda word: word[::-1])
print(sorted_fruits)  # Output: ['banana', 'apple', 'fig', 'raspberry', 'strawberry', 'cherry']
```
x??

---

#### Anonymous Functions and Lambda Syntax
Anonymous functions are essentially unnamed functions. In Python, `lambda` is used to define these functions inline within expressions. However, the syntax of `lambda` restricts it to single-expression bodies.
:p What limitations do lambda functions have in terms of their body?
??x
Lambda functions in Python can only contain a single expression and cannot include statements like loops (`while`), conditional blocks, or assignments (`=`). While new assignment expressions using `:=` can be used within lambdas, overly complex lambda functions are often better refactored into regular named functions defined with `def`.
```python
# Example of a simple lambda function that returns the length of a word
length_of_word = lambda word: len(word)
```
x??

---

#### Refactoring Lambda Functions
When faced with complex or hard-to-read lambda expressions, it is advisable to refactor them into named functions using `def`. This improves readability and maintainability.
:p What refactoring advice is given for complicated lambda expressions?
??x
For complicated lambda expressions that are difficult to read, the recommended procedure involves:
1. Writing a comment explaining what the lambda does.
2. Studying the comment and coming up with an appropriate name.
3. Converting the lambda into a named function using `def`.
4. Removing the original comment.

This process helps in making the code more understandable and maintainable.
```python
# Original hard-to-read lambda refactored as a def
original_lambda = lambda x: len(x) * 2 if 'a' in x else len(x)
refactored_def = def enhanced_length(word):
    if 'a' in word:
        return len(word) * 2
    else:
        return len(word)
```
x??

---

#### Callable Objects in Python
Callable objects include various types of functions, methods, and even instances of classes that have a `__call__` method.
:p What are callable objects in Python?
??x
Callable objects in Python refer to any object that can be called like a function. They include:
- User-defined functions (`def` or `lambda`)
- Built-in functions (e.g., `len`, `time.strftime`)
- Methods (like `dict.get`)
- Classes (invoked as if they were functions)
- Class instances with `__call__`
- Generator and async generator functions
Each callable can be invoked using the call operator (`()`).

```python
# Example of a class acting like a function
class CallableClass:
    def __init__(self, value):
        self.value = value

    def __call__(self, multiplier):
        return self.value * multiplier

callable_instance = CallableClass(5)
print(callable_instance(2))  # Output: 10
```
x??

