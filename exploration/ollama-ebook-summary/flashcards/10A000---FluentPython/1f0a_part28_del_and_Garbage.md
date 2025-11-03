# Flashcards: 10A000---FluentPython_processed (Part 28)

**Starting Chapter:** del and Garbage Collection

---

#### del Statement Overview
Background context explaining the `del` statement. The `del` statement is a statement, not a function, and it deletes references to objects rather than destroying objects directly.

:p What does the `del` statement do in Python?
??x
The `del` statement removes the reference to an object, but it doesn't destroy the object immediately. It only affects the object when there are no more references pointing to it.
```python
a = [1, 2]
b = a
del a
print(b)  # Output: [1, 2]
```
x??

---

#### Indirect Object Destruction via del
Explanation of how indirect destruction occurs. Objects may be destroyed by the garbage collector if they become unreachable due to `del` or other bindings.

:p How can an object be destroyed indirectly using `del`?
??x
An object can be destroyed indirectly when it becomes unreachable, which happens after all references are deleted and possibly after rebinding variables.

```python
a = [1, 2]
b = a
del a
b = [3]
# Now the list [1, 2] is unreferenced and may be garbage collected.
```
x??

---

#### __del__ Method Misconception
Explanation of the `__del__` method. It is not used for direct object disposal but rather to release external resources when an instance is about to be destroyed.

:p What is the purpose of the `__del__` method?
??x
The `__del__` method is a special method that is called by the Python interpreter when an object is about to be destroyed. It should not be used for direct disposal but for releasing external resources such as file handles or network connections.

```python
class Example:
    def __del__(self):
        print("Cleaning up resources.")
        
# The __del__ method will be called when the instance is destroyed.
```
x??

---

#### Reference Counting in CPython
Explanation of how reference counting works in CPython. Objects are destroyed immediately when their reference count reaches zero.

:p How does CPython handle object destruction using reference counting?
??x
In CPython, objects are destroyed immediately when their reference count reaches zero. This means that as soon as all references to an object are deleted or reassigned, the object's `__del__` method is called (if defined), and then it is freed.

```python
a = [1, 2]
b = a
del a
# b still points to [1, 2], so [1, 2] is not destroyed.
b = None
# Now the list [1, 2] has no references and can be garbage collected.
```
x??

---

#### Generational Garbage Collection in CPython 2.0+
Explanation of generational garbage collection added in CPython 2.0 to handle reference cycles.

:p How does generational garbage collection work in CPython 2.0+?
??x
Generational garbage collection in CPython 2.0 identifies groups of objects involved in reference cycles that may be unreachable even with outstanding references. This is useful for detecting circular references, which can prevent objects from being garbage collected otherwise.

```python
a = {1, 2, 3}
b = a
del a
# Even though 'a' was deleted, the set remains reachable through 'b'.
```
x??

---

#### Weak References and finalize
Explanation of weak references and how they can be used with `weakref.finalize`.

:p What is the purpose of using `weakref.finalize`?
??x
The `weakref.finalize` function allows you to register a callback that will be called when an object is garbage collected. This is useful for cleaning up resources or performing cleanup tasks.

```python
import weakref

s1 = {1, 2, 3}
def bye():
    print("...like tears in the rain.")

ender = weakref.finalize(s1, bye)
print(ender.alive)  # True
del s1
print(ender.alive)  # False after garbage collection
```
x??

---

#### Weak References
Weak references are a special type of reference in Python that do not prevent their target object from being garbage collected. This is useful in caching applications where you don't want the cached objects to keep other objects alive just because they are referenced by the cache.

:p What does a weak reference in Python ensure?
??x
A weak reference ensures that it does not affect the reference count of its target object, allowing the object to be garbage collected even if there are weak references pointing to it. This is particularly useful when you need to create caches or temporary objects without preventing them from being cleaned up by the garbage collector.

Example:
```python
import weakref

# Creating a regular object
obj = SomeClass()

# Creating a weak reference to obj
weak_ref_obj = weakref.ref(obj)

# The original object can still be accessed through the weak reference
print(weak_ref_obj())  # Output: <SomeClass instance>

# If no other references exist, obj will eventually be garbage collected
del obj

# Trying to access the weak reference after deletion shows None
print(weak_ref_obj())  # Output: None
```
x??

---

#### Tuple Reference Behavior
In Python, slicing a tuple or converting it into another tuple using `tuple()` does not create a new object but returns a reference to the same object. This behavior is different from lists and can be counterintuitive.

:p What happens when you use `t1[:]` or `tuple(t1)` on a tuple?
??x
When you use `t1[:]` or `tuple(t1)` on a tuple, it does not create a new copy but returns a reference to the same object. This means that both `t1` and the created tuple are pointing to the exact same memory location.

Example:
```python
t1 = (1, 2, 3)
t2 = t1[:]
t3 = tuple(t1)

print(t2 is t1)  # Output: True
print(t3 is t1)  # Output: True
```
x??

---

#### String Interning in Python
String interning is an optimization technique used by CPython where identical string literals share the same memory location. This can improve performance for operations involving frequent comparisons of strings.

:p How does string interning work in Python?
??x
String interning works such that identical string literals are stored only once in a special dictionary, and any references to these literals will point to this single instance. This optimization avoids unnecessary duplication of memory usage.

Example:
```python
s1 = 'ABC'
s2 = 'ABC'

print(s1 is s2)  # Output: True
```
x??

---

#### Immutable Objects in Python
Immutable objects like tuples, strings, and frozensets are optimized by CPython to share the same object when their content matches. However, this sharing does not apply to lists or mutable sequences.

:p What happens when you create a new tuple from an existing one?
??x
When you create a new tuple using `t1[:]` or `tuple(t1)` from an existing tuple `t1`, it does not make a copy but returns a reference to the same object. This means that both `t1` and the newly created tuple are referencing the exact same memory location.

Example:
```python
t1 = (1, 2, 3)
t2 = t1[:]
t3 = tuple(t1)

print(t2 is t1)  # Output: True
print(t3 is t1)  # Output: True
```
x??

---

#### Frozenset Reference Behavior
Similar to tuples and strings, a `frozenset` created from another set does not create a new object but returns a reference to the same object. However, slicing operations like `fs[:]` do not work with frozensets.

:p How does creating a `frozenset` from an existing set behave?
??x
Creating a `frozenset` from an existing set using `fs.copy()` will return a reference to the same object and not a new copy. This means that both the original set and the newly created frozenset are referencing the exact same memory location.

Example:
```python
fs = frozenset([1, 2, 3])
fs_copy = fs.copy()

print(fs is fs_copy)  # Output: True
```
x??

---

#### Immutable Objects and Their Value

Background context: In Python, understanding how immutable objects behave is crucial. An immutable object’s value cannot change over time, which means that once an immutable object is created, it remains unchanged.

:p What is a characteristic of immutable objects in Python?
??x
Immutable objects in Python are those whose values do not change after they are created. Examples include integers, strings, and tuples.
x??

---

#### Mutable Collections within Immutable Objects

Background context: Even though the value of an immutable object cannot be changed, mutable collections inside immutable objects can still change if they hold references to mutable items.

:p How does the immutability of a collection affect its contents?
??x
The immutability of a collection (like tuples) means that the identity and the reference to the collection itself do not change. However, if these collections contain references to mutable objects, those mutable objects can still be modified.
x??

---

#### The frozenset Class

Background context: `frozenset` is an immutable version of the built-in set class in Python. It contains only hashable elements and cannot be changed after it is created.

:p What distinguishes `frozenset` from other collections like sets?
??x
`frozenset` differs from sets because it is immutable, meaning its contents cannot change once it is created. However, since the elements in a frozenset must be hashable, they remain constant.
x??

---

#### Simple Assignment and Copying

Background context: In Python, simple assignment does not create copies of objects; instead, it creates new references to the same object.

:p What happens when you use simple assignment with immutable objects?
??x
When you use simple assignment with immutable objects, a new reference (alias) is created. The original and the newly assigned variable now point to the same object.
x??

---

#### Augmented Assignment

Background context: Augmented assignments (`+=`, `*=`) can create new objects for immutable objects but modify mutable objects in place.

:p What does augmented assignment do with immutable objects?
??x
With immutable objects, augmented assignment creates a new object. For example, if you use `a += 1` with an integer `a`, Python will create a new integer and reassign the variable.
x??

---

#### Rebinding

Background context: Rebinding is when you assign a new value to a variable that was previously bound to an immutable or mutable object.

:p How does rebinding work in Python?
??x
Rebinding occurs when you change what a variable refers to, effectively creating a new reference. If the previous object had no other references, it can be garbage collected.
x??

---

#### Function Parameters and Aliasing

Background context: In Python, function parameters are passed as aliases, meaning that functions can modify mutable objects received as arguments.

:p What happens when you pass a mutable object to a function in Python?
??x
When you pass a mutable object to a function, the function receives a reference to that object. If the function modifies the object, those changes will be reflected outside of the function because it operates on the same memory location.
x??

---

#### Summary

Background context: Every Python object has an identity, type, and value. Values can change, but identities are constant unless the variable is rebound.

:p What never changes in an immutable collection?
??x
In an immutable collection, the identities of the objects within never change. Only their values may change if they contain mutable items.
x??
---

