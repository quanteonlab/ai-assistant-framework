# Flashcards: 10A000---FluentPython_processed (Part 7)

**Starting Chapter:** Slicing. Slice Objects

---

#### Slices and Ranges Exclude the Last Item
Background context explaining that Pythonic convention of excluding the last item in slices and ranges works well with zero-based indexing used in Python, C, and many other languages. This makes it easy to see the length of a slice or range when only the stop position is given, compute the length from start and stop positions, and split sequences without overlapping.

Code examples:
```python
l = [10, 20, 30, 40, 50, 60]
print(l[:2])       # split at index 2   -> [10, 20]
print(l[2:])       # from index 2 to the end -> [30, 40, 50, 60]
print(l[:3])       # split at index 3   -> [10, 20, 30]
print(l[3:])       # from index 3 to the end -> [40, 50, 60]
```

:p What is the significance of excluding the last item in slices and ranges?
??x
Excluding the last item in Python’s slicing notation allows for easy computation of sequence lengths and splits without overlapping. For instance, using `l[:3]` gives you the first three elements, while `l[3:]` starts from index 3 to the end.
x??

---
#### Slice Objects
Background context explaining that slices can be specified with a stride or step, which can also be negative for reverse order.

Code examples:
```python
s = 'bicycle'
print(s[::3])      # every third character -> 'bye'
print(s[::-1])     # reversed string         -> 'elcycib'
print(s[::-2])     # every second in reverse -> 'eccb'
```

:p What does `s[::3]` and `s[::-1]` do?
??x
`s[::3]` returns a slice of the string `s`, taking every third character starting from index 0. `s[::-1]` returns the reversed version of the string `s`.
x??

---
#### Slicing Syntax and Functionality
Background context explaining how slicing syntax works in Python, including examples with ranges.

Code example:
```python
l = [10, 20, 30, 40, 50, 60]
print(l[2:])       # slice from index 2 to the end -> [30, 40, 50, 60]
```

:p How does slicing work in Python with range notation?
??x
Slicing in Python works using the syntax `seq[start:stop]`. The stop parameter is exclusive, meaning it includes all elements from start up to but not including stop. For example, `l[2:]` starts at index 2 and goes to the end of the list.
x??

---
#### Slice Objects as Function Calls
Background context explaining that slices can be created using the slice function.

Code example:
```python
invoice = "... invoice data ..."
SKU = slice(0, 6)
print(invoice[SKU])   # prints "0....."
```

:p How are slice objects typically used in Python code?
??x
Slice objects are often used to define named ranges or sequences that can be repeatedly accessed throughout the code. They allow for more readable and maintainable coding practices by avoiding hardcoded indices.
x??

---
#### Assigning Names to Slices
Background context explaining how names can be assigned to slices, making them easier to use in complex operations.

Code example:
```python
invoice = "... invoice data ..."
SKU = slice(0, 6)
DESCRIPTION = slice(6, 40)
PRICE = slice(52, 55)
print(invoice[DESCRIPTION])   # prints "Pimoroni PiBrella"
```

:p How can you assign names to slices in Python?
??x
You can create a `slice` object and assign it to a variable name. This allows for more readable code where slice objects are used repeatedly, such as parsing flat-file data like invoices.
x??

---

---
#### Slicing and Indexing in Python
Background context explaining how slicing works in Python. Slices allow you to access a range of elements from a sequence such as lists or strings using start, stop, and step parameters. For example:
```python
my_list = [0, 1, 2, 3, 4]
print(my_list[1:4])  # prints [1, 2, 3]
```
:p What is the purpose of slices in Python?
??x
Slices allow you to access a range of elements from sequences such as lists or strings. They provide a flexible way to extract parts of data without creating new objects.
x??

---
#### Slice Objects and Their Usage
Explanation on how slice objects are created and used, including `DESCRIPTION`, `UNIT_PRICE`, etc., provided in the text.
:p How do you create and use slice objects?
??x
You can create slice objects using the built-in `slice` function. These objects allow you to specify start, stop, and step parameters for slicing sequences like lists or strings. For example:
```python
DESCRIPTION = slice(6, 40)
UNIT_PRICE = slice(40, 52)

# Usage in a loop
for item in line_items:
    print(item[UNIT_PRICE], item[DESCRIPTION])
```
x??

---
#### Multidimensional Slicing and Ellipsis
Explanation on multidimensional slicing using tuples of indices. Introduction to the `...` (ellipsis) token.
:p What is multidimensional slicing in Python?
??x
Multidimensional slicing allows you to slice arrays or matrices with multiple dimensions. It uses a tuple of slices for each dimension:
```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a[0:2, 1])  # prints array([2, 5])
```
The `...` (ellipsis) token is recognized by the Python parser and can be used as a placeholder for missing slices in multidimensional indexing:
```python
x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(x[0,...])  # prints array([[1, 2], [3, 4]])
```
x??

---
#### Using Slices for In-Place Modification
Explanation on how slices can be used to modify mutable sequences in place.
:p How can slices be used to modify sequences in Python?
??x
Slices can be used on the left-hand side of an assignment statement to modify sequences in place. For example:
```python
l = list(range(10))
l[2:5] = [20, 30]
print(l)  # prints [0, 1, 20, 30, 5, 6, 7, 8, 9]
```
This modifies the sequence without creating a new object. If an iterable is not provided on the right-hand side, you'll get a `TypeError`.
x??

---
#### Concatenation and Repetition with Sequences
Explanation of how `+` and `*` operators work for sequences in Python.
:p How do `+` and `*` work for sequences in Python?
??x
For sequences like lists or strings, the `+` operator performs concatenation:
```python
l = [1, 2, 3]
print(l * 5)  # prints [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
```
The `*` operator can be used to repeat the sequence:
```python
print(5 * 'abcd')  # prints 'abcdabcdabcdabcdabcd'
```
Both operators create a new object and do not modify existing sequences.
x??

---
#### Pitfalls of Using `*` with Lists of Lists
Explanation on why using `*` with lists to initialize lists of lists can lead to unexpected results.
:p Why should you be careful when initializing lists of lists in Python?
??x
Using `*` to initialize a list of lists can result in all sublists pointing to the same object, leading to unintended behavior:
```python
weird_board = [['_'] * 3] * 3
print(weird_board)  # prints [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
```
Modifying one sublist will affect all others because they reference the same underlying object. Use a list comprehension to avoid this issue:
```python
board = [['_'] * 3 for _ in range(3)]
print(board)  # prints [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]
```
x??

---
#### Augmented Assignment with Sequences
Explanation on how `+=` and `*=` work differently based on the mutability of the sequence.
:p How do `+=` and `*=` operators behave in Python for sequences?
??x
Augmented assignment operators like `+=` and `*=` have different behaviors depending on the type of sequence:
```python
l = [1, 2, 3]
l += [4, 5]  # Similar to l.extend([4, 5])
print(l)  # prints [1, 2, 3, 4, 5]

s = "hello"
s += " world"  # Creates a new string
print(s)  # prints "hello world"
```
For mutable sequences like lists, `+=` modifies the sequence in place. For immutable sequences like strings, it creates a new object.
x??

---

#### Tuple Multiplication and ID Changes
Background context: When a tuple is multiplied, Python creates a new tuple with repeated elements. The `id` of the initial and resulting tuples can be different due to memory optimization techniques.

Example code:
```python
t = (1, 2, 3)
print(id(t)) # Initial id
t *= 2
print(id(t)) # New id after multiplication
```

:p How does Python handle tuple multiplication in terms of ID changes?
??x
Python creates a new tuple with the repeated elements when using the `*=` operator on tuples. This can result in different memory locations (IDs) before and after the operation, reflecting that a new object is created.
x??

---

#### Tuple Assignment Puzzler
Background context: Tuples are immutable, meaning their elements cannot be changed directly. However, mutable elements within a tuple can still be modified.

Example code:
```python
t = (1, 2, [30, 40])
print(t[2] += [50, 60]) # Attempting to append to the list inside the tuple
```

:p What happens when you try to modify a mutable element within an immutable tuple?
??x
A `TypeError` is raised because tuples are immutable and do not support item assignment directly. However, modifying a mutable object (like a list) referenced by a tuple element can change that reference.

In this case:
```python
t = (1, 2, [30, 40])
print(t[2] += [50, 60]) # Raises TypeError and modifies the inner list
```
The output will be:
```
[30, 40, 50, 60]
```
But `t` itself remains unchanged in terms of ID, as it still references the same mutable object.
x??

---

#### Bytecode Explanation for += Operation on Tuples
Background context: The bytecode generated by Python provides insights into how operations like `+=` are handled. Understanding this can help explain why certain behaviors occur.

Example code:
```python
dis.dis('s[a] += b')
```

:p What does the bytecode reveal about the operation `t[2] += [50, 60]` in a tuple?
??x
The bytecode shows that Python attempts to perform an in-place addition (`INPLACE_ADD`) on the mutable list within the tuple. This results in a modification of the original list object because lists are mutable.

Bytecode details:
```python
dis.dis('s[a] += b ')
1           0 LOAD_NAME                0 (s)
            3 LOAD_NAME                1 (a)
            6 DUP_TOP_TWO
            7 BINARY_SUBSCR
           11 LOAD_NAME                2 (b)
           14 INPLACE_ADD
           15 ROT_THREE
           16 STORE_SUBSCR
           17 LOAD_CONST               0 (None)
           20 RETURN_VALUE
```

This shows that the operation attempts to add `b` to `s[a]`, but due to the nature of mutable lists, it modifies the list rather than raising an error.
x??

---

