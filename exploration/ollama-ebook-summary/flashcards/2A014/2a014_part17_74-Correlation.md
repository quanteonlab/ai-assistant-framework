# Flashcards: 2A014 (Part 17)

**Starting Chapter:** 74-Correlation Mining

---

#### Simple Counting Recommender
Background context: The simplest feature type involves counting frequencies and pairwise frequencies to generate initial models. This approach is used as a basic framework for implementing a most-popular-item recommender (MPIR).

:p What does simple counting recommend?
??x
Simple counting recommends the item with the highest frequency of clicks or interactions, optimizing the click-through rate (CTR) by allocating all recommendations to the most popular item.
x??

---
#### Bayesian Approximation in Recommender Systems
Background context: The MPIR framework can be extended using a Bayesian approximation approach. This involves setting up a multiarmed bandit problem where rewards are determined based on prior distributions of CTR.

:p How is the reward function formulated for recommending items?
??x
The reward function Rx,c = ∑i ∈ ℐ ci * N * xi, where ci represents the prior distribution of CTR for each item i. The goal is to maximize this reward by optimizing the allocation plan x.
x??

---
#### Multiarmed Bandit Problem in Recommender Systems
Background context: The multiarmed bandit problem models a scenario with multiple choices (items) and seeks an optimal strategy over time to maximize rewards.

:p What does maximizing the reward function achieve?
??x
Maximizing the reward function Rx,c = ∑i ∈ ℐ ci * N * xi, where ci is the CTR prior for each item i and N is the number of recommendations, leads to allocating all recommendations to the most popular item in terms of CTR.
x??

---
#### Distributional Assumptions and Optimization
Background context: When there's a mismatch between confidence levels in different items' CTRs, distributional assumptions are used. Gamma and normal distributions can be applied for optimization.

:p How is the expected number of clicks computed?
??x
The expected number of clicks EN0 * x0p0 - q0 + N1 * x1p1 - q1 + q0N0 + q1N1, where p0 and p1 are the prior distributions for CTR in two time periods.
x??

---
#### Co-occurrence Matrix
Background context: The co-occurrence matrix is a multidimensional array of counts indicating how often items appear together. It's used to find correlations between items.

:p What does the co-occurrence matrix represent?
??x
The co-occurrence matrix represents the frequency of co-occurrences between pairs of items, useful for finding similarities and recommending related items.
x??

---
#### Higher-Order Co-occurrences
Background context: Extending the basic co-occurrence model to higher-order models can capture more complex interactions among items.

:p How can higher-order co-occurrences be computed?
??x
Higher-order co-occurrences can be computed by considering triples, quadruples, or more items. For example, Cℐ = ℐT * ℐ, where Cℐ is the matrix with rows and columns indexed by items.
x??

---
#### Conditional MPIR Recommendations
Background context: The conditional MPIR considers user interactions to provide recommendations based on the last item interacted with.

:p What does the conditional MPIR return?
??x
The conditional MPIR returns the max of the elements in the row corresponding to the last interacted item, xi. This is often represented as a basis vector qxi.
x??

---
#### Incidence Vector and User-Item Matrix
Background context: An incidence vector represents user interactions with items using one-hot encoding.

:p What is an incidence vector?
??x
An incidence vector is a binary vector representing each user's interaction with items, where the elements are 1 if the user-item pair has interacted.
x??

---

#### Co-occurrence Data Storage
Background context: The co-occurrence data for recommendation systems is often sparse, meaning it contains a large number of zeros. Storing such matrices directly can be inefficient and slow.

:p How do you store co-occurrence data efficiently?
??x
We typically use sparse matrix formats to handle the sparsity effectively. JAX supports BCOO (Batched Coordinate Format), which stores only non-zero elements along with their coordinates and values.
```python
# Example of BCOO storage in Python using a hypothetical function
indices = [[0, 1], [2, 3]]  # Coordinates of non-zero elements
values = [5, 7]             # Values at those coordinates
shape = (4, 4)              # Shape of the matrix

coo_matrix = create_bcoo(indices, values, shape)
```
x??

---

#### Pointwise Mutual Information (PMI)
Background context: PMI is used to normalize co-occurrence counts and measure how much more frequent a co-occurrence is than random chance. It's an estimator for word similarity in NLP.

:p What formula is used to compute the Pointwise Mutual Information (PMI)?
??x
The PMI between two items, $x_i $ and$x_j$, can be computed as follows:
$$\text{PMI}(x_i, x_j) = \log \left( \frac{\frac{C_\mathcal{I}_{x_i,x_j}}{\# \text{total interactions}}}{\frac{\# x_i \cdot \# x_j}{(\# \text{total interactions})^2}} \right)$$

Where:
- $C_\mathcal{I}_{x_i, x_j}$ is the co-occurrence count between $ x_i $ and $x_j$.
- $\# \text{total interactions}$ is the total number of interactions.
- $\# x_i $ and$\# x_j $ are the counts of$x_i $ and$x_j$ respectively.

This formula normalizes the co-occurrence count to account for the frequency of individual items.
x??

---

#### Jaccard Similarity
Background context: Jaccard similarity is a measure used in recommendation systems, particularly when dealing with binary interactions or incidence sets. It measures how much two users or items have interacted.

:p What is the formula for Jaccard Similarity?
??x
The Jaccard Similarity between two sets $A $ and$B$ (e.g., sets of items that two users have interacted with) can be defined as:
$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Where:
- $|A \cap B|$ is the number of elements common to both sets.
- $|A \cup B|$ is the total number of unique elements in both sets.

This formula gives a measure of similarity between two sets by considering the intersection and union of the sets.
x??

---

#### Cosine Similarity
Background context: Cosine similarity is another measure used for recommendation systems. It calculates the cosine of the angle between two vectors, which can handle multi-interaction scenarios where each interaction has a polarity.

:p How do you compute Cosine Similarity?
??x
The Cosine Similarity between two sets $A $ and$B$(e.g., incidence sets for users or items) is computed as:
$$\text{Cosim}(A, B) = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \cdot \sqrt{\sum_{i=1}^{n} b_i^2}}$$

Where:
- $a_i $ and$b_i$ are the interaction counts for each item in sets A and B respectively.

This formula normalizes the dot product of two vectors to give a similarity score between -1 and 1.
x??

---

#### User-Item Recommendation Using Jaccard Similarity
Background context: In recommendation systems, we can use various similarity measures to recommend items. Here, we discuss how to use Jaccard Similarity for user-item recommendations.

:p How do you compute the Jaccard Similarity between a user and an item?
??x
To compute the Jaccard Similarity between a user $y_u $ and an unseen item$x_i$, you can follow these steps:
1. Let $A = \{ x_j | (y_u, x_j) \text{ has been interacted with} \}$.
2. Let $B = \{ x_j | x_j \text{ is in the set of co-occurring items with } x_i \}$.
3. Calculate:
$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

This gives a measure of similarity between the user's interactions and the item's co-occurrences.
x??

---

#### Item-Item Recommendation Using Jaccard Similarity
Background context: Similarly to user-user or user-item recommendations, we can use Jaccard Similarity for item-item recommendations.

:p How do you compute the Jaccard Similarity between two items?
??x
To compute the Jaccard Similarity between two items $x_i $ and$x_j$, you need to:
1. Find the set of users who have interacted with both items: $A = \{ y_u | (y_u, x_i) \text{ and } (y_u, x_j) \text{ has been interacted with} \}$.
2. Calculate:
$$\text{Jaccard}(A) = \frac{|A|}{\# \text{total users}}$$

This gives a measure of the commonality in user interactions between two items.
x??

---

#### User-User Recommendation Using Jaccard Similarity
Background context: We can also use Jaccard Similarity for recommending items based on similar users.

:p How do you compute the Jaccard Similarity between two users?
??x
To compute the Jaccard Similarity between two users $y_u $ and$y_v$, you need to:
1. Find the set of interactions for both users: $A = \{ x_j | (y_u, x_j) \text{ has been interacted with} \}$ and $B = \{ x_j | (y_v, x_j) \text{ has been interacted with} \}$.
2. Calculate:
$$\text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

This gives a measure of the commonality in user interactions between two users.
x??

---

#### Latent Spaces
Latent spaces are lower-dimensional representations of high-dimensional data. Unlike direct feature representations, latent features do not correspond to any specific real value but are learned through a process. This is useful for capturing complex relationships between items and users more efficiently.

Latent vector generation often involves factorizing matrices into smaller matrices that capture the essence of the interactions without representing every single feature explicitly.
:p What are latent spaces and how do they differ from direct feature representations?
??x
Latent spaces represent data in a lower-dimensional form, which is learned through a process. They do not directly map to specific real-world features but help in capturing complex relationships more efficiently.

Direct feature representations, on the other hand, are high-dimensional and can correspond to actual attributes of items or users.
x??

---

#### Matrix Factorization
Matrix factorization involves breaking down a large matrix into smaller matrices that capture the underlying structure. This is particularly useful for recommendation systems where interaction data between users and items (e.g., ratings) can be represented in a lower-dimensional space.

The Singular Value Decomposition (SVD) is one common method used for this purpose.
:p What does SVD do, and how is it applied to matrix factorization?
??x
Singular Value Decomposition (SVD) decomposes a matrix into three matrices:$U $, $\Sigma $, and $ V^T$. This decomposition can be used to reduce the dimensionality of the original matrix by setting some singular values to zero, effectively approximating the matrix with lower rank.

Here is an example using SVD for matrix factorization:
```python
import numpy as np

# Example interaction matrix
a = np.array([
    [1, 0, 0 ,1],
    [1, 0, 0 ,0],
    [0, 1, 1, 0],
    [0, 1, 0, 0]
])

u, s, v = np.linalg.svd(a, full_matrices=False)
# Set the last two eigenvalues to 0
s[2:4] = 0

b = np.dot(u * s, v)

print(b)  # This is the newly reconstructed matrix.
```
x??

---

#### Information Bottleneck in Matrix Factorization
In matrix factorization, an information bottleneck occurs when the rank of the matrices (K) is much smaller than the dimensions of the original data. This forces the model to infer the missing values based on the learned latent factors.

The idea behind this is that by reducing the dimensionality, the system has fewer degrees of freedom and must generalize more effectively.
:p What is an information bottleneck in matrix factorization?
??x
An information bottleneck happens when the rank (K) of the matrices used in the factorization process is significantly smaller than the original dimensions (N or M). This forces the model to infer the missing values based on learned latent factors, essentially making it generalize more effectively.

For example, if you have a 4x4 matrix and reduce its rank to 2, the system must capture the essential information in only two dimensions, which can help in reconstructing the original data or predicting missing entries.
x??

---

#### Code Example for Matrix Factorization
The code provided demonstrates how SVD can be used to factorize a user-item interaction matrix into smaller matrices, which can then be recombined to approximate the original matrix.

Here is a more detailed explanation of the process:
1. Decompose the original matrix using SVD.
2. Set some singular values (eigenvalues) to zero to reduce the rank.
3. Recombine the matrices to get an approximation of the original matrix.
:p What does this code snippet demonstrate?
??x
This code snippet demonstrates how Singular Value Decomposition (SVD) can be used to factorize a user-item interaction matrix into smaller matrices and then recombine them to approximate the original matrix.

Here is the detailed breakdown:
```python
import numpy as np

# Define the interaction matrix 'a'
a = np.array([
    [1, 0, 0 ,1],
    [1, 0, 0 ,0],
    [0, 1, 1, 0],
    [0, 1, 0, 0]
])

# Perform SVD on matrix 'a'
u, s, v = np.linalg.svd(a, full_matrices=False)

# Set the last two singular values to zero
s[2:4] = 0

# Reconstruct the matrix using truncated SVD
b = np.dot(u * s, v)

print(b)  # This is the newly reconstructed matrix.
```
x??

---

#### Low-Rank Methods in Recommendation Systems
Low-rank methods are used to reduce the dimensionality of high-dimensional feature spaces, making recommendation systems more efficient and effective. By representing users and items as low-dimensional vectors, complex relationships can be captured while reducing computational complexity.

The goal is to generate latent features or embeddings that capture the essential information without explicitly representing every single feature.
:p Why use low-rank methods in recommendation systems?
??x
Low-rank methods are used in recommendation systems to reduce the dimensionality of high-dimensional feature spaces, making the system more efficient and effective. By representing users and items as low-dimensional vectors, complex relationships can be captured while reducing computational complexity.

This is particularly useful because it helps in generating more personalized recommendations by focusing on the most relevant features and reduces the curse of dimensionality associated with sparse data.
x??

---

