# Flashcards: 2A014 (Part 18)

**Starting Chapter:** 78-Dot Product Similarity

---

#### Matrix Rank and Low-Rank Approximation

Background context explaining that a matrix's rank is related to its ability to be represented by fewer dimensions. This concept is crucial for understanding how recommender systems work, especially when using techniques like singular value decomposition (SVD). The rank of an \(N \times M\) matrix is the minimum number of dimensions necessary to represent the vectors in the matrix.

In the given example:

```b = [[1.17082039  0.         0.         0.7236068  ]
 [0.7236068   0.         0.         0.4472136  ]
 [0.         1.17082039  0.7236068   0.        ]
 [0.         0.7236068   0.4472136   0.        ]]
```

The matrix has a rank of 4, but through low-rank approximation (where the rank is reduced to 2), we can capture essential patterns in user-item interactions.

:p What is the concept of matrix rank and how does it apply to recommender systems?
??x
Matrix rank refers to the minimum number of dimensions necessary to represent the vectors in a matrix. In the context of recommender systems, reducing the rank through techniques like SVD helps in capturing essential patterns with fewer parameters, making the system more efficient.

```java
// Pseudocode for SVD decomposition
public class SvdDecomposition {
    public void decompose(Matrix A) {
        // Perform SVD to get U, S, V matrices
        Matrix U = singularValueDecomposition(A).getU();
        Matrix Sigma = singularValueDecomposition(A).getSigma();
        Matrix V = singularValueDecomposition(A).getV();

        // Rank reduction can be done by keeping only top-k singular values and corresponding vectors
    }
}
```
x??

---

#### Dot Product Similarity in Recommender Systems

Background context explaining that the dot product provides a geometric interpretation of user-item similarity, which is crucial for recommendation systems. It captures alignment between preferences and characteristics.

:p What is dot product similarity and how does it relate to recommender systems?
??x
Dot product similarity measures the projection of one vector onto another, scaled by their magnitudes. In the context of recommendation systems, it helps in identifying items that align well with a user's preferences.

The formula for dot product similarity between vectors \(u\) and \(p\) is:
\[ u \cdot p = \|u\| \cdot \|p\| \cdot \cos(\theta) \]

Where:
- \( \|u\|\) and \( \|p\|\) are the magnitudes of vectors \(u\) and \(p\)
- \(\theta\) is the angle between them

In Java, this can be implemented as:

```java
public class DotProductSimilarity {
    public double calculate(Vector u, Vector p) {
        double dotProduct = u.dot(p);
        double magnitudeU = u.magnitude();
        double magnitudeP = p.magnitude();

        return dotProduct / (magnitudeU * magnitudeP);
    }
}
```

x??

---

#### Cosine Similarity in Recommender Systems

Background context explaining that cosine similarity is a normalized measure of the alignment between vectors, which is useful for recommendation systems. It ranges from -1 to 1 and provides a more meaningful metric compared to raw dot product.

:p What is cosine similarity and how does it differ from dot product similarity?
??x
Cosine similarity is derived directly from the dot product and normalizes the measure of alignment between two vectors, making it invariant to their magnitudes. The formula for cosine similarity is:

\[ \text{cosineSimilarity}(u, p) = \frac{u \cdot p}{\|u\| \cdot \|p\|} \]

This ensures that the similarity score ranges from -1 (completely dissimilar) to 1 (perfect alignment). In contrast, dot product can be influenced by the magnitudes of the vectors.

In Java, this can be implemented as:

```java
public class CosineSimilarity {
    public double calculate(Vector u, Vector p) {
        double dotProduct = u.dot(p);
        double magnitudeU = u.magnitude();
        double magnitudeP = p.magnitude();

        return (dotProduct / (magnitudeU * magnitudeP));
    }
}
```

x??

---

#### Geometric Interpretation of Dot Product in Recommendation Systems

Background context explaining that the geometric interpretation of the dot product helps in understanding how alignment between user preferences and item characteristics can be measured. Shorter angles indicate better alignment, leading to higher similarity scores.

:p How does the geometric interpretation of the dot product help in recommendation systems?
??x
The geometric interpretation of the dot product captures the alignment between user preferences (vector \(u\)) and item characteristics (vector \(p\)). The angle \(\theta\) between these vectors affects the cosine similarity score:

- Small angles (\(\theta < 45^\circ\)) indicate high alignment, leading to a higher similarity score.
- Large angles (\(\theta > 90^\circ\)) indicate low alignment or dissimilarity.

In recommendation systems, this helps in identifying items that are likely to be relevant and appealing to the user based on their preferences. For example:

```java
public class RecommendationSystem {
    public Item recommend(User u) {
        double highestSimilarity = 0;
        Item recommendedItem = null;

        for (Item item : allItems) {
            double similarityScore = calculateCosineSimilarity(u, item);
            if (similarityScore > highestSimilarity) {
                highestSimilarity = similarityScore;
                recommendedItem = item;
            }
        }

        return recommendedItem;
    }

    private double calculateCosineSimilarity(User u, Item i) {
        // Implementation of cosine similarity
        return (u.dot(i) / (u.magnitude() * i.magnitude()));
    }
}
```

x??

#### PMI and Co-occurrence Models
Background context: We discussed how PMI (Pointwise Mutual Information) can be used to generate similarity measures based on co-occurrences. PMI is defined as follows:
\[ \text{PMI}(x_i, x_j) = \log\left(\frac{P(x_i, x_j)}{P(x_i)P(x_j)}\right) \]
However, PMI itself does not provide a distance metric but can be used to determine high mutual information between items. Here, \( P(x_i) \) and \( P(x_j) \) are the probabilities of events \( x_i \) and \( x_j \), respectively.

We introduced co-occurrence models where the co-occurrence structure between two items is utilized to generate measures of similarity.

:p What is PMI used for in this context?
??x
PMI is used to determine high mutual information between an item in a cart and other items, effectively generating a measure of their similarity. It helps in making recommendations based on very high mutual information.
x??

---

#### Hellinger Distance
Background context: To quantify the difference between co-occurrence distributions, we can use the Hellinger distance as a proper distance metric. The Hellinger distance is defined as:
\[ H(P, Q) = 1 - \frac{1}{\sqrt{2}} \sum_{i} \sqrt{p_i q_i} \]
where \( P = p_i \) and \( Q = q_i \) are probability density vectors.

:p How is the Hellinger distance calculated?
??x
The Hellinger distance is calculated using the formula:
\[ H(P, Q) = 1 - \frac{1}{\sqrt{2}} \sum_{i} \sqrt{p_i q_i} \]
where \( p_i \) and \( q_i \) are the probabilities of events in distributions P and Q, respectively. It measures the distributional distance between two probability density vectors.
x??

---

#### Kullback–Leibler Divergence
Background context: While Hellinger distance is a useful measure, another popular method for measuring differences between distributions is the Kullback-Leibler (KL) divergence, which is described in a Bayesian sense as the amount of surprise in seeing distribution P when expecting distribution Q.

:p What is KL divergence?
??x
Kullback–Leibler (KL) divergence measures the difference between two probability distributions. It quantifies the amount of information lost if one uses \( Q \) to approximate \( P \). The formula for KL divergence is:
\[ D_{KL}(P \| Q) = \sum_i p_i \log\left(\frac{p_i}{q_i}\right) \]
However, it is not a proper distance metric because it is asymmetric.
x??

---

#### Hellinger Distance and Discrete Distributions
Background context: The Hellinger distance effectively measures the 2-norm measure theoretic distance between distributions. It naturally generalizes to discrete distributions.

:p Why is Hellinger distance preferred over KL divergence?
??x
Hellinger distance is preferred over KL divergence because it is a proper distance metric, being symmetric and smooth. This makes it more suitable for various applications where a true distance measurement is required.
x??

---

#### Matrix Factorization (MF) via SVD
Background context: To reduce the dimensionality of our recommender problem, we can use matrix factorization techniques like Singular Value Decomposition (SVD). The key idea is to decompose an \( N \times M \) matrix into two smaller matrices \( U_{N \times d} \) and \( V_{d \times M} \).

:p What is SVD used for in recommendation systems?
??x
SVD is used to reduce the dimensionality of the recommender problem by factorizing the rating or interaction matrix. It decomposes a large matrix into smaller matrices, making it easier to handle high-dimensional data.
x??

---

#### Factorization and Latent Space
Background context: By factorizing a matrix \( A \) as \( U V^T \), where \( U \) is of size \( N \times d \) and \( V \) is of size \( d \times M \), we can represent the original data in a lower-dimensional latent space. This helps in reducing the complexity of the recommendation model.

:p How does factorization help in recommendation systems?
??x
Factorization, such as using SVD, helps reduce the dimensionality of the recommender problem by approximating the original matrix with two smaller matrices \( U \) and \( V \). This allows for a more efficient representation of users and items in a lower-dimensional latent space.
x??

---

#### Singular Value Decomposition (SVD)
Background context: SVD is a matrix factorization technique that decomposes any real-valued matrix into three separate matrices: a left singular matrix, a diagonal matrix of eigenvalues, and a right singular matrix. The columns and rows of the singular matrices are eigenvectors, and the values in the diagonal matrix are the eigenvalues.

:p What is SVD and how does it decompose a matrix?
??x
SVD decomposes a real-valued matrix into three separate matrices: a left singular matrix (U), a diagonal matrix of eigenvalues (Σ), and a right singular matrix (V^T). The columns and rows of the singular matrices are eigenvectors, and the values in the diagonal matrix are the eigenvalues.
??x

---

#### Matrix Factorization (MF)
Background context: MF decomposes a user-item matrix into two matrices representing user preferences and item characteristics. This allows for generating personalized recommendations by matching users' preferences with items.

:p What is Matrix Factorization (MF) used for?
??x
Matrix Factorization (MF) decomposes the user-item interaction matrix into two lower-dimensional matrices: one representing user preferences and the other representing item characteristics. By matching these representations, MF can generate personalized recommendations.
??x

---

#### Challenges with Matrix Factorization
Background context: When dealing with MF, several challenges arise such as sparsity of the matrix, varying number of non-zero elements per vector, computational complexity, and issues with full-rank methods requiring imputation.

:p What are some common challenges in using Matrix Factorization?
??x
Challenges include:
- The user-item interaction matrix being sparse and often non-negative or binary.
- Varying numbers of non-zero entries across item vectors.
- High computational complexity due to factorizing matrices.
- Difficulty with full-rank methods, which require imputation that can be complex.
??x

---

#### Alternating Least Squares (ALS)
Background context: ALS is an optimization method used in MF where the matrix is alternately updated between two factors. It significantly reduces the number of computations compared to optimizing all parameters simultaneously.

:p What is ALS and how does it work?
??x
Alternating Least Squares (ALS) optimizes the factorization process by alternating updates between one factor matrix at a time, instead of updating both matrices simultaneously. This method dramatically reduces computational complexity.
Algorithm:
```python
# Pseudocode for ALS
def ALS(U, V, D, η):
    while not converged:
        # Update U
        U = update_U(V, D, U, η)
        # Update V
        V = update_V(U, D, V, η)
```
??x

---

#### Distance Between Matrices
Background context: The distance between two matrices can be calculated using various methods. Commonly used distances include observed mean squared error and cross-entropy loss.

:p How is the distance between two matrices typically measured in MF?
??x
The distance between two matrices can be measured using:
1. Observed Mean Squared Error (MSE):
   \[
   MSE = \frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (A_{ij} - U_i V_j)^2
   \]
   where \( |\Omega| \) is the number of non-zero entries.
2. Cross-Entropy Loss, useful when dealing with single nonzero entries per vector:
   \[
   CE = -(r_{ij} \log(U_i V_j) + (1 - r_{ij}) \log(1 - U_i V_j))
   \]
   where \( r_{ij} \) is the observed rating.
??x

---

