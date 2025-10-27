# Flashcards: 2A014 (Part 2)

**Starting Chapter:** 13-Basic Types Initialization and Immutability

---

#### Ranker Function Explanation
Background context: The ranker function sorts dictionary values and returns a list of top recommended item-identifiers based on these values. It leverages Python's built-in `sorted` function to sort by dictionary values, using reverse sorting for descending order.

:p What is the purpose of the ranker in this context?
??x
The purpose of the ranker is to provide a simple method for recommending items based on their popularity or frequency within categories. This can be useful in e-commerce platforms and video sites where top-rated content needs to be highlighted.
x??

---
#### JAX Array Immutability
Background context: JAX arrays, similar to NumPy arrays, have a fixed shape and data type but are immutable by design. This means that elements of the array cannot be changed once it is created.

:p What differentiates JAX arrays from typical Python lists?
??x
JAX arrays differ from typical Python lists in that they are immutable—once an array is created, its elements cannot be altered directly. Unlike mutable data structures, which allow for direct modifications (e.g., list[index] = new_value), JAX arrays require a new array to be created if changes are needed.
x??

---
#### JAX Array Initialization
Background context: In the provided example, a three-dimensional vector is initialized using `jnp.array`. The type of the elements and their shape are specified.

:p How is a JAX array initialized in this snippet?
??x
A JAX array is initialized using the `jnp.array` function. Here's how it works:
```python
import jax.numpy as jnp

# Initialize an array with float32 type
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
```
The `dtype` parameter specifies the data type of the elements in the array.
x??

---
#### JAX Array Shape and Data Types
Background context: The shape attribute provides information about the dimensions of a JAX array, and the `dtype` parameter determines the type of the elements.

:p How do you check the shape of a JAX array?
??x
To check the shape of a JAX array, you can use the `.shape` attribute. For example:
```python
print(x.shape)
```
This will output the dimensions of the array as a tuple, such as `(3,)` for a three-dimensional vector.
x??

---
#### Modifying JAX Arrays
Background context: Since JAX arrays are immutable, any attempt to modify an element directly results in a `TypeError`. Instead, new arrays must be created.

:p Why can't elements of a JAX array be modified directly?
??x
Elements of a JAX array cannot be modified directly because the array is immutable. This immutability ensures that functions applied to data do not have side effects, allowing for parallel processing and deterministic results.
To modify an element, you must create a new array with the desired changes:
```python
# Incorrect: Direct modification will result in TypeError
x[0] = 4.0

# Correct: Create a new array
new_x = jnp.array([4.0, x[1], x[2]], dtype=jnp.float32)
```
x??

---

#### Indexing and Slicing Arrays
Indexing and slicing are fundamental operations that allow us to access specific parts of an array. These operations follow a `start:end:stride` convention, where the first element indicates the start, the second indicates where to end (but not inclusive), and the stride specifies the number of elements to skip over.

:p How do you use indexing and slicing to print different parts of a matrix in JAX?
??x
To print different parts of a matrix using indexing and slicing:

- To print the whole matrix: `print(x)`
  
  ```python
  x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
  print(x) 
  ```

- To print the first row: `print(x[0])`
  
  ```python
  x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
  print(x[0])
  ```

- To print the last row: `print(x[-1])`
  
  ```python
  x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
  print(x[-1])
  ```

- To print the second column: `print(x[:, 1])`
  
  ```python
  x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
  print(x[:, 1])
  ```

- To print every other element: `print(x[::2, ::2])`
  
  ```python
  x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
  print(x[::2, ::2])
  ```

x??

---

#### Broadcasting in JAX
Broadcasting is a feature of JAX that allows binary operations (such as addition or multiplication) to be applied between tensors of different sizes. When the operation involves a tensor with an axis of size 1, this axis is duplicated to match the size of the larger tensor.

:p What happens when you perform broadcasting in JAX?
??x
When performing broadcasting in JAX:

- A scalar can be multiplied by a matrix directly:
  
  ```python
  x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)
  y = 2 * x
  print(y) 
  ```

- A vector can be multiplied with a matrix by duplicating its axes to match the shape of the matrix:
  
  - Reshaping the vector to (3,1):
    ```python
    vec = jnp.reshape(jnp.array([0.5, 1.0, 2.0]), [3, 1])
    y = vec * x
    print(y)
    ```

  - Reshaping the vector to (1,3):
    ```python
    vec = jnp.reshape(vec, [1, 3])
    y = vec * x
    print(y)
    ```

x??

---

#### Random Numbers in JAX
JAX handles random numbers differently from traditional NumPy because of its pure function philosophy. Instead of modifying a global state, it uses a key that can be split into subkeys for reproducibility.

:p How does JAX handle the generation and splitting of random keys?
??x
In JAX:

- A random key is created from a seed:
  
  ```python
  import jax.random as random
  key = random.PRNGKey(0)
  x = random.uniform(key, shape=[3, 3])
  print(x) 
  ```

- Keys can be split to generate more keys and subkeys for parallel operations:
  
  ```python
  key, subkey = random.split(key)
  x = random.uniform(key, shape=[3, 3])
  print(x)
  y = random.uniform(subkey, shape=[3, 3])
  print(y) 
  ```

x??

---

#### Just-in-Time Compilation (JIT) in JAX
Just-in-Time (JIT) compilation transforms code to be compiled just before it is run, allowing the same code to execute on different hardware such as CPUs, GPUs, and TPUs. This can significantly improve execution speed.

:p What does JIT compilation do in JAX?
??x
In JAX:

- JIT compilation allows the same code to be executed efficiently on various hardware:
  
  ```python
  import jax
  
  x = random.uniform(key, shape=[2048, 2048]) - 0.5

  def my_function(x):
      x = x @ x
      return jnp.maximum(0.0, x)

  %timeit my_function(x).block_until_ready() 
  # Output: ~302 ms per loop
  
  my_function_jitted = jax.jit(my_function)
  %timeit my_function_jitted(x).block_until_ready()
  # Output: ~294 ms per loop
  ```

JIT compilation can significantly improve performance, especially on GPU and TPU backends. However, the first execution may have some overhead due to initial compilation.

x??

---

#### User-Item Matrix Introduction
Background context: The user-item matrix is a fundamental concept in recommendation systems, used to represent ratings or preferences given by users for items. This matrix helps in understanding patterns and similarities between users and items, which can be utilized to make personalized recommendations.

:p What is the user-item matrix?
??x
The user-item matrix is a table that represents the ratings or preferences of users for different items. It's used to identify commonalities and patterns among users and items to generate recommendations.
x??

---
#### Binary Relationships Example
Background context: The text introduces an example where five friends rate four cheeses, ranging from 1 (dislike) to 4 (like). This example helps in understanding how relationships between users and items can be recorded.

:p What is the binary relationship mentioned for the cheese ratings?
??x
The binary relationship mentioned is a simple rating given by each friend (user) for each type of cheese (item), ranging from 1 to 4, where 1 indicates dislike and 4 indicates strong preference.
x??

---
#### Data Representation Using Tables
Background context: The example provides a table format to represent the cheese ratings. This method makes it easier to summarize the data and understand the preferences of each user for different items.

:p How is the cheese rating data represented in the table?
??x
The cheese rating data is represented in a table where rows correspond to users (friends) and columns correspond to items (cheeses). Each cell contains the rating given by a user for an item.
```plaintext
Cheese taster | Gouda | Chèvre | Emmentaler | Brie
A             | 5     | 4      | 4          | 1
B             | 2     | 3      | 3          | 4.5
C             | 3     | 2      | 3          | 4
D             | 4     | 4      | 5          | -
E             | 3     | -      | -         | -
```
x??

---
#### Heatmap Representation
Background context: The text explains using a heatmap to visualize the user-item matrix, making it easier to identify patterns and similarities among users and items.

:p How is the data represented in a heatmap?
??x
In a heatmap, each cell's color intensity represents the rating given by a user for an item. A higher value is typically shown with a darker or more intense color. The heatmap provides a visual representation of the ratings.
```python
import seaborn as sns
_ = np.nan
scores = np.array([[5, 4, 4, 1],
                   [2, 3, 3, 4.5],
                   [3, 2, 3, 4],
                   [4, 4, 5, _],
                   [3, _, _, _]])
sns.heatmap(
    scores,
    annot=True,
    fmt=".1f",
    xticklabels=['Gouda', 'Chèvre', 'Emmentaler', 'Brie'],
    yticklabels=['A','B','C','D','E']
)
```
x??

---
#### Differing Representations
Background context: The text discusses two types of data representations: dense and sparse. A dense representation contains a datum for each possibility, whereas a sparse representation includes only nontrivial observations.

:p What are the differences between dense and sparse representations?
??x
Dense representations include a datum for every possible combination of users and items, even if some ratings are null or zero. Sparse representations contain only the nontrivial observations, omitting null values or zeros.
```python
scores_dense = {
    ('A', 'Gouda'): 5,
    ('B', 'Chèvre'): 3,
    ('C', 'Emmentaler'): 3,
    ...
}
```
x??

---

