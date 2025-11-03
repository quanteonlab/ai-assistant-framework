# Flashcards: 10A000---FluentPython_processed (Part 8)

**Starting Chapter:** list.sort versus the sorted Built-In

---

#### Mutable vs Immutable Objects and Augmented Assignment
Background context: In Python, understanding mutable and immutable objects is crucial. This distinction affects operations like augmented assignment (`TOS += b`). When `TOS` (Top of Stack) refers to a mutable object, such as a list, the operation can be successful. However, if `TOS` points to an immutable object, like a tuple or string, the operation will fail.

Example:
```python
# Example 2-16
t = ('a', 'b')  # Immutable tuple
l = ['a', 'b']  # Mutable list

# Augmented assignment on mutable list: succeeds
l += ['c']
print(l)  # Output: ['a', 'b', 'c']

# Augmented assignment on immutable tuple: fails
t += ('d',)
```
:p What happens when you attempt to perform `TOS += b` if `TOS` refers to an immutable object?
??x
When `TOS` refers to an immutable object, such as a tuple or string, the augmented assignment operation will fail because immutable objects do not support in-place modification. The operation will raise an exception since Python cannot modify the contents of immutable objects.
```python
t = ('a', 'b')
# This will raise an exception
try:
    t += ('c',)
except TypeError as e:
    print(e)  # Output: can only concatenate tuple (not "tuple") to tuple
```
x??

---

#### In-Place vs New Object Creation with `list.sort()` and `sorted()`
Background context: Python's `list.sort()` method sorts a list in-place, returning `None`. The `sorted()` function creates a new sorted list. Understanding the differences is essential for managing memory and avoiding side effects.

:p What does `list.sort()` return?
??x
`list.sort()` returns `None`, indicating that it modifies the list in place without creating a new one.
```python
fruits = ['grape', 'raspberry ', 'apple', 'banana']
fruits.sort()
print(fruits)  # Output: ['apple', 'banana', 'grape', 'raspberry']
```
x??

---

#### Keyword Arguments `reverse` and `key` in `sorted()` and `list.sort()`
Background context: Both `list.sort()` and `sorted()` support optional keyword arguments like `reverse` and `key`. These allow for flexible sorting behavior, such as reverse ordering or custom key generation.

:p How does the `reverse` parameter affect the output of `sorted()`?
??x
The `reverse` parameter in `sorted()` reverses the order of the sorted list. If set to `True`, items are sorted in descending order; if `False` (the default), they are sorted in ascending order.
```python
fruits = ['grape', 'raspberry ', 'apple', 'banana']
print(sorted(fruits, reverse=True))  # Output: ['raspberry ', 'grape', 'banana', 'apple']
```
x??

---

#### Stability of the Sorting Algorithm
Background context: Python's sorting algorithm is stable. This means that if two items compare equal, their original order relative to each other is preserved.

:p What does it mean for a sorting algorithm to be "stable"?
??x
A stable sorting algorithm preserves the relative ordering of elements that compare equal. In other words, if two elements have the same value according to the comparator function (e.g., `key` in Python), their order relative to each other will remain unchanged from the original list.

Example:
```python
fruits = ['grape', 'raspberry ', 'apple', 'banana']
print(sorted(fruits, key=len))  # Output: ['grape', 'apple', 'banana', 'raspberry ']
```
Here, "grape" and "apple" are both 5 characters long. The stable sorting algorithm ensures that their original order is preserved, so "grape" appears before "apple".
x??

---

#### Arrays vs Lists
Background context explaining the differences between arrays and lists. Discuss the memory efficiency, speed of operations, and use cases for each type. Highlight that Python lists are flexible but can be memory-heavy for large numerical data.

:p What is a primary difference between using `array` and `list` in Python?

??x
Arrays are more efficient in terms of memory usage when dealing with large sequences of numbers because they store items as packed bytes, whereas lists store full-fledged objects. Additionally, arrays support direct operations like `.fromfile()` and `.tofile()`, making them ideal for handling binary data efficiently.

```python
from array import array

# Creating an array of double-precision floats
floats = array('d', (random() for i in range(10**7)))
```
x??

---

#### Binary File Operations with Arrays
Explanation on using `array.tofile()` and `array.fromfile()` to read from and write to binary files, emphasizing their speed and memory efficiency.

:p How can you save an array of floating-point numbers to a binary file?

??x
You can use the `array.tofile()` method to save an array of floating-point numbers or other numeric data types to a binary file. This is faster and more memory-efficient than writing each number as a string in a text file.

```python
floats = array('d', (random() for i in range(10**7)))
fp = open('floats.bin', 'wb')
floats.tofile(fp)
fp.close()
```
x??

---

#### Memory Efficiency of Arrays
Explanation on how arrays store data more compactly compared to lists, and the implications for handling large datasets.

:p Why might you prefer using `array` over a list when dealing with large numeric data?

??x
Using `array` is preferable when working with large datasets of numbers because it uses less memory by storing items as packed bytes rather than full-fledged objects. This makes operations like saving and loading data to/from binary files much faster.

For example, an array of 10 million floating-point numbers takes up 80 MB (8 bytes per float), whereas a list would take significantly more memory due to the overhead of each object.

```python
floats = array('d', (random() for i in range(10**7)))
```
x??

---

#### Operations on Arrays
Detailed explanation of methods available for arrays, including `.append()`, `.fromfile()`, and `.tobytes()`.

:p What are some key operations you can perform with an `array` object?

??x
Key operations on array objects include appending elements (`append`), reading from binary files (`fromfile`), and converting to bytes (`tobytes`). These methods provide efficient ways to manipulate and manage numerical data in Python.

```python
floats = array('d', (random() for i in range(10**7)))
fp = open('floats.bin', 'wb')
floats.tofile(fp)
fp.close()
```
x??

---

#### Deque vs List
Explanation of why a `deque` might be more efficient than a list when items are added or removed from both ends frequently.

:p In what scenarios would using a `deque` be preferable to a list?

??x
A `deque` is more efficient than a list for operations that add or remove items from the beginning and end of the sequence. Deques support fast appends and pops from both ends, making them suitable for use cases like queues.

```python
from collections import deque

d = deque(range(10))
print(d)  # deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```
x??

---

#### Set Membership Checking
Explanation of the benefits of using a `set` for membership checking in large datasets.

:p Why might you use a set to check if an item is present in a collection?

??x
Using a set can provide fast membership checking, especially when dealing with large collections. Sets are optimized for this type of operation and are more efficient than lists or tuples.

```python
my_set = {1, 2, 3, 4, 5}
if 3 in my_set:
    print("Item found")
```
x??

---

#### Summary of Mutable Sequence Types
Overview of mutable sequence types available in Python, including arrays and deques, and their advantages over lists.

:p What are some alternatives to using a list when working with sequences?

??x
Alternatives to using a list include `array` for numeric data that needs efficient memory usage, and `deque` for operations requiring fast appends and pops from both ends. Both provide more efficient storage and operations compared to standard lists in specific scenarios.

```python
from array import array
from collections import deque

# Example of using an array
floats = array('d', (random() for i in range(10**7)))

# Example of using a deque
d = deque(range(10))
```
x??

---

#### Array Sorting in Python 3.10
Background context: As of Python 3.10, the `array` module does not support in-place sorting like lists do with `list.sort()`. To sort an array, you must use the built-in `sorted()` function to create a new sorted array.

:p How can you sort an array using Python's built-in functions?
??x
To sort an array, use the `sorted()` function and assign it back to your variable. Here is how you can do it:

```python
import array

# Example array
a = array.array('i', [5, 3, 4, 6, 1, 2])

# Sorting using sorted()
sorted_a = array.array(a.typecode, sorted(a))

print(sorted_a)  # Output: array('i', [1, 2, 3, 4, 5, 6])
```
x??

---

#### Keeping an Array Sorted with `bisect.insort()`
Background context: To keep your array sorted while adding items, you can use the `bisect.insort()` function. This function maintains the order of elements in the list as new elements are added.

:p How does `bisect.insort()` help maintain a sorted array?
??x
`bisect.insort()` helps to insert an element into a sorted list while maintaining its sorted state. Here's how you can use it:

```python
import bisect
import array

# Example array
a = array.array('i', [1, 2, 3, 4, 5])

# Inserting a new value while keeping the array sorted
bisect.insort(a, 0)     # Inserts 0 at the correct position to keep the list sorted: [0, 1, 2, 3, 4, 5]
bisect.insort(a, 6)     # Inserts 6 at the end: [0, 1, 2, 3, 4, 5, 6]

print(a)  # Output: array('i', [0, 1, 2, 3, 4, 5, 6])
```
x??

---

#### Memory Views
Background context: `memoryview` is a built-in Python class that allows you to handle slices of arrays without copying the underlying memory. It was inspired by NumPy and provides shared-memory between different data structures.

:p What is the primary use case for `memoryview`?
??x
The primary use case for `memoryview` is handling large datasets efficiently by sharing memory between different data structures without first copying the data. This can be very useful when working with large arrays, images, databases, or other large binary data.

Here's an example of creating and manipulating a memory view:

```python
from array import array

octets = array('B', range(6))  # Create an array of bytes from 0 to 5

# Creating a memoryview on the array
m1 = memoryview(octets)

print(m1.tolist())  # Output: [0, 1, 2, 3, 4, 5]

# Casting the memory view to another shape
m2 = m1.cast('B', [2, 3])
m3 = m1.cast('B', [3, 2])

print(m2.tolist())  # Output: [[0, 1, 2], [3, 4, 5]]
print(m3.tolist())  # Output: [[0, 1], [2, 3], [4, 5]]

# Modifying the original array via a memory view
m2[1, 1] = 22
m3[1, 1] = 33

print(octets)  # Output: array('B', [0, 1, 2, 33, 22, 5])
```
x??

---
Each flashcard provides a detailed explanation and example code to help understand the concept better.

