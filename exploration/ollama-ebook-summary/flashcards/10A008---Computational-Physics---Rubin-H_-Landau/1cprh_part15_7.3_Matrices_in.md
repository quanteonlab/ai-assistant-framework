# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 15)

**Starting Chapter:** 7.3 Matrices in Python. 7.3.2 NumPy Matrices

---

#### Lists as Arrays in Python
Background context: In Python, lists are a built-in data structure that can hold sequences of arbitrary objects. While they share similarities with arrays from other programming languages, there are significant differences and unique features that make them versatile for various applications.

:p What is the primary difference between a list and an array in Python?
??x
A Python list is mutable (changeable) and dynamic in size, allowing elements to be added or removed after creation. An array typically requires specifying its size beforehand.
x??

---

#### Creating and Accessing Lists in Python
Background context: Lists can hold sequences of objects such as numbers or strings. They are indexed and accessed using square brackets.

:p How do you create a list in Python?
??x
You can create a list by placing elements within square brackets, separated by commas.
```python
L = [1, 2, 3]
```
x??

---

#### Accessing Elements of a List in Python
Background context: Lists are zero-indexed and allow for accessing individual elements using their index.

:p How do you access the first element of a list named `L`?
??x
You can access the first element by indexing it with `[0]`.
```python
print(L[0])
```
x??

---

#### Modifying Elements in a List in Python
Background context: Lists are mutable, meaning elements can be changed after their creation.

:p How do you change an element in a list?
??x
You can modify an element by assigning a new value to the desired index.
```python
L[0] = 5
```
This changes the first element of `L` to `5`.
x??

---

#### Iterating Over Elements in a List in Python
Background context: Lists can be iterated over using loops, such as for-loops.

:p How do you iterate over all elements in a list named `L`?
??x
You can use a for-loop to print each element of the list.
```python
for items in L:
    print(items)
```
This will loop through and print each item in the list `L`.
x??

---

#### Tuples in Python
Background context: Tuples are similar to lists but are immutable, meaning their elements cannot be changed after creation.

:p What is a tuple in Python?
??x
A tuple is an immutable sequence of objects that can hold any type of data. It is created using round parentheses.
```python
T = (1, 2, 3, 4)
```
Attempting to change an element will result in an error.
x??

---

#### Operations on Lists in Python
Background context: Lists support various operations such as concatenation, slicing, and appending.

:p What is the `append` method used for in lists?
??x
The `append` method adds a single item to the end of the list. For example:
```python
L.append(4)
```
This appends `4` to the existing list `L`.
x??

---

#### Length of a List in Python
Background context: The length of a list can be obtained using the built-in function `len()`.

:p How do you find out the number of elements in a list?
??x
You can use the `len()` function to determine the length of a list.
```python
n = len(L)
```
This assigns the number of elements in `L` to the variable `n`.
x??

---

#### Slicing Lists in Python
Background context: Lists support slicing, which allows you to access a portion of the list.

:p How do you slice a list from index 1 to 3?
??x
You can use slicing notation `[i:j]` to get a sublist.
```python
sublist = L[1:3]
```
This creates `sublist` containing elements from index `1` to `2`.
x??

---

#### Concatenating Lists in Python
Background context: Lists support concatenation, which combines two lists into one.

:p How do you concatenate two lists?
??x
You can use the `+` operator to concatenate two lists.
```python
new_list = L1 + L2
```
This creates a new list that is the combination of `L1` and `L2`.
x??

---

#### Importing NumPy
Background context: To use NumPy functions and features, you must first import the NumPy package into your Python program. This is a fundamental step to perform numerical operations efficiently.

:p How do you import NumPy at the beginning of a Python script?
??x
You can import all the functionality of NumPy by using the following line:
```python
from numpy import *
```
This allows you to call NumPy functions directly without prefixing them with `numpy.`. However, it is recommended to use `import numpy as np` for better readability and fewer typing errors.
x??

#### Creating a 1D Array
Background context: A one-dimensional array (vector) can be created using the `array()` function from NumPy. The elements are of the same type.

:p How do you create a 1D array in Python using NumPy?
??x
You can create a 1D array by passing a list to the `array` function:
```python
import numpy as np

vector1 = np.array([1, 2, 3, 4, 5])
```
This creates an array with elements `[1, 2, 3, 4, 5]`.
x??

#### Element-wise Operations on Arrays
Background context: NumPy arrays support element-wise operations such as addition and multiplication. These operations are applied to each corresponding element in the arrays.

:p What happens when you add two vectors using NumPy?
??x
When you add two vectors using `+`, NumPy performs an element-wise addition:
```python
vector1 = np.array([1, 2, 3, 4, 5])
vector2 = vector1 + vector1
```
The result is a new array where each element is the sum of corresponding elements in the original arrays. For example:
```python
>>> print(vector2)
[2 4 6 8 10]
```
x??

#### Scalar Multiplication with Arrays
Background context: You can multiply an array by a scalar value, which multiplies every element in the array by that scalar.

:p What happens when you multiply an array by a constant using NumPy?
??x
Multiplying an array by a constant performs element-wise multiplication:
```python
vector1 = np.array([1, 2, 3, 4, 5])
vector2 = 3 * vector1
```
The result is an array where each element is the product of the corresponding element in `vector1` and the scalar value. For example:
```python
>>> print(vector2)
[ 3 6 9 12 15]
```
x??

#### Creating a Matrix with NumPy
Background context: A matrix can be created using a 2D array of arrays. However, it's important to understand that this is not the same as a true mathematical matrix.

:p How do you create a 2D array (matrix) in Python using NumPy?
??x
You can create a 2D array by passing a list of lists to the `array` function:
```python
import numpy as np

matrix1 = np.array([[0, 1], [1, 3]])
```
This creates an array with two rows and two columns. For example:
```python
>>> print(matrix1)
[[0 1]
 [1 3]]
```
x??

#### Matrix Multiplication in NumPy
Background context: Unlike element-wise operations, matrix multiplication is performed using the `*` operator on 2D arrays.

:p What happens when you multiply a matrix by itself using `*`?
??x
Matrix multiplication in NumPy does not perform a true matrix product but rather an element-wise multiplication:
```python
matrix1 = np.array([[0, 1], [1, 3]])
result = matrix1 * matrix1
```
The result is another 2D array where each element is the product of corresponding elements. For example:
```python
>>> print(result)
[[0 1]
 [1 9]]
```
This output differs from the expected true matrix product.
x??

#### NumPy Array Dimensions and Types
Background context: A NumPy array can have up to 32 dimensions, but all elements must be of the same type. The `dtype` attribute is used to determine the data type.

:p How do you check the data type of an array in NumPy?
??x
To check the data type of an array, use the `dtype` attribute:
```python
import numpy as np

a = np.array([1, 2, 3, 4])
print(a.dtype)
```
The output will show the data type. For example, if all elements are integers, it will return `int32`. If there is a mix of types (including floats), it will return `float64`.

You can create an array with specific types as well:
```python
b = np.array([1.2, 2.3, 3.4])
print(b.dtype)
```
This will output `float64` since all elements are floating-point numbers.
x??

#### NumPy Array Shape
Background context: The shape of a NumPy array is a tuple indicating the size of each dimension.

:p How do you check the shape of an array in NumPy?
??x
To find out the shape of an array, use the `shape` attribute:
```python
import numpy as np

vector1 = np.array([1, 2, 3, 4, 5])
print(vector1.shape)
```
The output will be a tuple indicating the dimensions. For example, for `vector1`, it will return `(5,)`.

For a matrix with two rows and two columns:
```python
matrix1 = np.array([[0, 1], [1, 3]])
print(matrix1.shape)
```
This will return `(2, 2)`.
x??

---

#### Importing NumPy and Using `arange` and `reshape`
NumPy is a powerful library for numerical computations in Python. When working with arrays, it’s essential to know how to create and manipulate them efficiently. The `np.arange()` function generates a 1D array of evenly spaced values within a specified range. The `reshape()` method can be used to change the shape of an existing array.

:p How do you generate a 1D array using NumPy's `arange` function, and how can you reshape it into a 3x4 matrix?
??x
You start by importing NumPy as `np`. Then, use `np.arange(12)` to create a 1D array with elements ranging from 0 to 11. To reshape this array into a 3x4 matrix, you can call the `reshape` method on the resulting array, passing it the new shape `(3, 4)`. Here’s how:

```python
import numpy as np

# Create a 1D array with 12 elements from 0 to 11
one_dimensional_array = np.arange(12)

# Reshape the 1D array into a 3x4 matrix
reshaped_array = one_dimensional_array.reshape((3, 4))

print("Reshaped Array:")
print(reshaped_array)
```

x??

---

#### Transposing an Array with `.T` Method
In NumPy, transposing an array can be done using the `.T` method. This is particularly useful for changing the orientation of a matrix.

:p How do you transpose a 2D array in NumPy?
??x
Transposing an array means swapping its rows and columns. You can use the `.T` attribute or method to achieve this. Here's how:

```python
import numpy as np

# Create a sample 3x4 matrix
matrix = np.arange(12).reshape((3, 4))

# Transpose the matrix using .T
transposed_matrix = matrix.T

print("Original Matrix:")
print(matrix)
print("\nTransposed Matrix:")
print(transposed_matrix)
```

x??

---

#### Reshaping an Array into a Vector
Reshaping is useful for converting multi-dimensional arrays into single-dimensional or vice versa. In NumPy, you can reshape an array to have different dimensions.

:p How do you convert a 3x4 matrix into a vector of length 12 using NumPy?
??x
To transform a 3x4 matrix into a flat (vector) form with 12 elements, you can use the `reshape` method and specify the new shape as `(1, 12)` or simply `12` if it’s a single-dimensional vector. Here's how:

```python
import numpy as np

# Create a 3x4 matrix
matrix = np.arange(12).reshape((3, 4))

# Reshape into a vector of length 12
vector = matrix.reshape(12)

print("Original Matrix:")
print(matrix)
print("\nReshaped Vector:")
print(vector)
```

x??

---

#### Slicing an Array in Python
Slicing is a powerful feature that allows you to extract parts of arrays. In NumPy, slicing can be done using the colon `:` operator with start:stop:step.

:p How do you slice a 2D array in NumPy?
??x
In NumPy, slicing works similarly to list slicing but on multi-dimensional arrays. Here are some examples:

```python
import numpy as np

# Create a sample 3x4 matrix
matrix = np.arange(12).reshape((3, 4))

# Slice the first two rows of the matrix
first_two_rows = matrix[:2, :]

print("Original Matrix:")
print(matrix)
print("\nFirst Two Rows:")
print(first_two_rows)

# Slice columns 1-3 (not inclusive of 3) from all rows
columns_1_to_3 = matrix[:, 1:3]

print("\nColumns 1 to 3:")
print(columns_1_to_3)
```

x??

---

#### Compound Data Types in NumPy Arrays
NumPy arrays can contain elements of different types, including compound data like sub-arrays or even complex numbers. This feature makes NumPy versatile for various applications.

:p How do you create a NumPy array with compound data types?
??x
Creating an array with compound data types involves specifying the data type explicitly when using `np.array`. Here’s how to create an array of arrays and an array of complex numbers:

```python
import numpy as np

# Create an array of 3 sub-arrays
compound_array = np.array([[10, 20], [30, 40], [50, 60]])

print("Compound Array:")
print(compound_array)

# Check the shape and size of the compound array
print("\nShape:", compound_array.shape)
print("Size:", compound_array.size)
print("Data Type:", compound_array.dtype)

# Create an array with complex numbers
complex_array = np.array([[1, 2+2j], [3+2j, 4]], dtype=complex)

print("\nComplex Array:")
print(complex_array)
```

x??

---

#### Dot Product vs Element-by-Element Multiplication
For operations like matrix multiplication and element-wise (Hadamard) product, NumPy provides specific functions. The `dot` function computes the dot product of two arrays, whereas the `*` operator performs an element-by-element multiplication.

:p How do you perform a dot product and an element-by-element multiplication on matrices in NumPy?
??x
To compute the matrix or dot product, use the `np.dot()` function. For element-wise (Hadamard) multiplication, simply use the `*` operator between arrays:

```python
import numpy as np

# Create two 2D matrices for demonstration
matrix1 = np.array([[0, 1], [1, 3]])
matrix2 = np.array([[1, 2], [3, 4]])

# Compute the dot product of matrix1 and matrix2
dot_product = np.dot(matrix1, matrix2)

print("Dot Product:")
print(dot_product)

# Perform element-by-element multiplication (Hadamard product)
elementwise_multiplication = matrix1 * matrix2

print("\nElement-Wise Multiplication:")
print(elementwise_multiplication)
```

x??

#### NumPy Array Slicing and Broadcasting
Background context: NumPy arrays are handled similarly to scalar variables, which makes slicing and broadcasting very powerful. Slicing allows you to extract a range of elements from an array using indices separated by colons. Broadcasting is an operation that allows NumPy to expand the dimensions of smaller arrays to match larger ones for operations.

:p How does slicing work in NumPy?
??x
Slicing in NumPy works similarly to Python's list and tuple slicing, where two indices separated by a colon indicate a range. For example, `stuff[3:7]` will slice the array from index 3 to 6 (not including 7).

Here is an example of slicing:

```python
import numpy as np

# Create a NumPy array of zeros
stuff = np.zeros(10, dtype=float)

# Create another array with values in the range [0-4]
t = np.arange(4)

# Use slicing to assign values based on square root function
stuff[3:7] = np.sqrt(t + 1)
```

In this example, `stuff` starts as an array of zeros. The slice `stuff[3:7]` is assigned the result of applying the `sqrt` function to each element in `t + 1`.

x??

---

#### NumPy Array Broadcasting Example
Background context: Broadcasting allows values to be assigned to multiple elements via a single assignment statement, making operations on arrays more efficient. This example demonstrates how broadcasting works with a simple assignment.

:p What is an example of broadcasting in NumPy?
??x
Broadcasting allows you to perform operations between arrays of different shapes by expanding the dimensions of smaller arrays to match larger ones for operations. Here's an example:

```python
w = np.zeros(100, dtype=float)
w[:] = 23.7
```

In this code, `w` is a NumPy array of size 100 initialized with zeros. The line `w[:] = 23.7` broadcasts the scalar value 23.7 to all elements in the array.

x??

---

#### Solving Matrix Equations using NumPy
Background context: In NumPy, you can solve matrix equations such as $Ax = b $ using linear algebra functions provided by NumPy's `linalg` module. The `solve` function is used to find the solution vector$x$.

:p How do you solve a matrix equation using NumPy?
??x
To solve a matrix equation like $Ax = b$ in NumPy, you can use the `solve` function from the `numpy.linalg` package.

Here's an example:

```python
import numpy as np

# Define matrix A and vector b
A = np.array([[1, 2, 3], [22, 32, 42], [55, 66, 100]])
b = np.array([1, 2, 3])

# Solve the equation Ax = b for x
x = np.linalg.solve(A, b)

print('Solution:', x)
```

The `solve` function automatically handles matrix operations and returns the solution vector $x$.

x??

---

#### Matrix Inverse Calculation in NumPy
Background context: Another way to solve a matrix equation like $Ax = b $ is by calculating the inverse of matrix$A $(denoted as$ A^{-1}$) and then using it to find the solution. This can be done using `numpy.linalg.inv`.

:p How do you calculate the inverse of a matrix in NumPy?
??x
To calculate the inverse of a matrix in NumPy, you use the `inv` function from the `numpy.linalg` package.

Here is an example:

```python
import numpy as np

# Define matrix A and vector b
A = np.array([[1, 2, 3], [22, 32, 42], [55, 66, 100]])

# Test if the inverse of A is correct by multiplying it with A
print(np.dot(np.linalg.inv(A), A))
```

This will output an identity matrix, confirming that the `inv` function correctly computes $A^{-1}$.

To solve for $x$ using the inverse, you can do:

```python
# Solve the equation Ax = b for x using the inverse of A
x = np.dot(np.linalg.inv(A), b)
print('Solution:', x)
```

The solution `x` will be printed out.

x??

---

#### Numerical Inverse of a Matrix

Background context: Finding the numerical inverse of a matrix is crucial for solving systems of linear equations. The provided matrix $A $ can be inverted to find its numerical inverse, and then checking this inverse by verifying if$AA^{-1} = I$. This also helps in understanding the precision of the calculation.

:p Find the numerical inverse of the matrix $A = \begin{bmatrix} 4 & -2 & 1 \\ 3 & 6 & -4 \\ 2 & 1 & 8 \end{bmatrix}$.

??x
To find the numerical inverse, you can use a NumPy function such as `numpy.linalg.inv`. Here's how you might do it:

```python
import numpy as np

# Define matrix A
A = np.array([[4, -2, 1],
              [3, 6, -4],
              [2, 1, 8]])

# Calculate the inverse of A
A_inv = np.linalg.inv(A)

print('Numerical Inverse:', A_inv)
```

This code will give you the numerical inverse of matrix $A$.

To verify that this is indeed the correct inverse, check if multiplying $A$ by its inverse gives the identity matrix:

```python
# Check if AA^-1 = I
I = np.dot(A, A_inv)

print('Check Matrix (should be close to Identity):', I)
```

The result should be very close to the identity matrix. The discrepancy in decimal places will give you an idea of the precision.

x??

---

#### Solving Linear Equations

Background context: Given a matrix $A $ and vectors$b_1, b_2, b_3 $, we need to solve for the vector$ x $such that$ Ax = b_i$. This involves using NumPy's `numpy.linalg.solve` function.

:p Consider the same matrix $A $ as in the previous problem. Solve for the vectors$x_1, x_2, x_3$ corresponding to different right-hand side (RHS) vectors:

- $b_1 = \begin{bmatrix} 12 \\ -25 \\ 32 \end{bmatrix}$-$ b_2 = \begin{bmatrix} 4 \\ -10 \\ 22 \end{bmatrix}$-$ b_3 = \begin{bmatrix} 20 \\ -30 \\ 40 \end{bmatrix}$

??x
To solve the linear equations, you can use NumPy's `numpy.linalg.solve` function. Here is how:

```python
import numpy as np

# Define matrix A and vectors b1, b2, b3
A = np.array([[4, -2, 1],
              [3, 6, -4],
              [2, 1, 8]])

b1 = np.array([12, -25, 32])
b2 = np.array([4, -10, 22])
b3 = np.array([20, -30, 40])

# Solve for x1
x1 = np.linalg.solve(A, b1)
print('Solution for x1:', x1)

# Solve for x2
x2 = np.linalg.solve(A, b2)
print('Solution for x2:', x2)

# Solve for x3
x3 = np.linalg.solve(A, b3)
print('Solution for x3:', x3)
```

This code will output the solutions $x_1, x_2, x_3 $ corresponding to each$b_i$.

The expected solutions are:

- $x_1 = \begin{bmatrix} 1 \\ -2 \\ 4 \end{bmatrix}$-$ x_2 = \begin{bmatrix} 0.312 \\ -0.038 \\ 2.677 \end{bmatrix}$-$ x_3 = \begin{bmatrix} 2.319 \\ -2.965 \\ 4.79 \end{bmatrix}$

x??

---

#### Eigenvalues and Eigenvectors

Background context: The eigenvalue problem is a fundamental concept in linear algebra, where we find the eigenvalues and eigenvectors of a matrix $I $ such that$I\omega = \lambda\omega$. This helps in understanding the principal axes of a cube.

:p Solve for the eigenvalues and eigenvectors of the matrix:

$$I = \begin{bmatrix} 0.6667 & -0.25 \\ -0.25 & 0.6667 \end{bmatrix}$$??x
To solve the eigenvalue problem, you can use NumPy's `numpy.linalg.eig` function.

```python
import numpy as np

# Define matrix I
I = np.array([[2./3, -1./4],
              [-1./4, 2./3]])

# Solve for eigenvalues and eigenvectors
E_vals, E_vectors = np.linalg.eig(I)

print('Eigenvalues:', E_vals)
print('Eigenvector Matrix:', E_vectors)
```

This code will output the eigenvalues and eigenvectors of matrix $I$.

To verify that the equation $I\omega = \lambda\omega$ holds, you can check:

```python
# Extract first eigenvector
vec = np.array([E_vectors[0, 0], E_vectors[1, 0]])

# Compute LHS and RHS of eigenvalue equation
LHS = np.dot(I, vec)
RHS = E_vals[0] * vec

print('LHS - RHS:', LHS - RHS)
```

The result should be close to zero, indicating the correctness of the eigenvalues and eigenvectors.

x??

---

#### Double Roots in Eigenvalue Problems

Background context: When a matrix has double roots as its eigenvalues, it can lead to degenerate eigenvectors. This means that any linear combination of these eigenvectors is also an eigenvector corresponding to the same eigenvalue.

:p Find the eigenvalues and eigenvectors of the matrix:

$$A = \begin{bmatrix} -2 & 2 & -3 \\ 2 & 1 & -6 \\ -1 & -2 & 0 \end{bmatrix}$$

Verify that you obtain the eigenvalues $\lambda_1 = 5, \lambda_2 = \lambda_3 = -3 $. Verify also for the eigenvector corresponding to $\lambda_1 = 5 $ and verify if the eigenvectors for$\lambda = -3$ are degenerate.

??x
To find the eigenvalues and eigenvectors, you can use NumPy's `numpy.linalg.eig` function:

```python
import numpy as np

# Define matrix A
A = np.array([[-2, 2, -3],
              [2, 1, -6],
              [-1, -2, 0]])

# Solve for eigenvalues and eigenvectors
E_vals, E_vectors = np.linalg.eig(A)

print('Eigenvalues:', E_vals)
print('Eigenvector Matrix:', E_vectors)
```

This code will output the eigenvalues and eigenvectors of matrix $A$.

To verify that the eigenvalue $5$ has an associated eigenvector, you can check:

```python
# Extract eigenvector for lambda = 5
lambda_1_idx = np.where(np.isclose(E_vals, 5))[0][0]
vec_lambda_1 = E_vectors[:, lambda_1_idx]

print('Eigenvector corresponding to lambda=5:', vec_lambda_1)
```

For the eigenvalue $-3$, you can similarly check:

```python
# Extract eigenvectors for lambda = -3
lambda_2_idx = np.where(np.isclose(E_vals, -3))[0]

for idx in lambda_2_idx:
    print('Eigenvector corresponding to lambda=-3:', E_vectors[:, idx])
```

The eigenvalues obtained should be $5 $ and two copies of$-3 $. The eigenvectors for the double root $-3$ will show that they are linearly dependent, confirming degeneracy.

x??

---

#### Solving a System with Hilbert Matrix

Background context: A system of linear equations can often be solved using known matrices like the Hilbert matrix. This type of problem is common in numerical analysis and provides an opportunity to practice solving systems with well-known structures.

:p Solve for $x$ values in a system where:

$$[A_{ij}] = a = \begin{bmatrix} 1 & 1/2 & 1/3 & \cdots & 1/100 \\ 1/2 & 1/3 & 1/4 & \cdots & 1/101 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1/100 & 1/101 & 1/102 & \cdots & 1 \end{bmatrix}$$and$$[b_i] = b = \begin{bmatrix} 1 \\ 1/2 \\ 1/3 \\ \vdots \\ 1/100 \end{bmatrix}$$??x
To solve the system of linear equations with a Hilbert matrix and its first column vector, you can use NumPy's `numpy.linalg.solve` function:

```python
import numpy as np

# Define the Hilbert matrix A and vector b
n = 100
A = np.array([[1/(i+j) for j in range(1, n+1)] for i in range(1, n+1)])
b = np.array([1/i for i in range(1, n+1)])

# Solve for x
x = np.linalg.solve(A, b)

print('Solution vector:', x)
```

This code constructs the Hilbert matrix $A $ and the vector$b $, then solves for the vector$ x$.

The result will be a solution vector that approximates the values of $x $ satisfying$Ax = b$. Due to the nature of the Hilbert matrix, the solution can be quite sensitive to numerical precision issues.

x??

--- 

These flashcards cover key concepts in solving linear equations and eigenvalue problems using NumPy functions. Each card provides context, relevant code examples, and detailed explanations for better understanding. ---

