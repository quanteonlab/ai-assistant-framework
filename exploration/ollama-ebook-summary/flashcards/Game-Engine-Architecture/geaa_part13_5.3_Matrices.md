# Flashcards: Game-Engine-Architecture_processed (Part 13)

**Starting Chapter:** 5.3 Matrices

---

#### Pseudovectors and Transformations
Background context: In 3D game programming, understanding pseudovectors is important because they require specific handling when changing coordinate system handedness. A pseudovector transforms differently from a regular vector under reflection or handedness changes.

:p What are pseudovectors in the context of 3D game programming?
??x
Pseudovectors, also known as axial vectors, are vectors that transform like vectors but change sign when the coordinate system is reflected (handedness is changed). This property makes them useful for representing quantities that have a direction associated with their orientation, such as angular momentum. When changing handedness in a game, it's necessary to properly transform pseudovectors to ensure consistent behavior.

```java
// Example of transforming a pseudovector v when changing handedness
Vector3 v = ...; // original vector
Vector3 transformedV = -v; // flip the sign to preserve orientation under handedness change
```
x??

---

#### Linear Interpolation (LERP)
Background context: Linear interpolation is used in game programming for smooth transitions between two points or vectors. It helps in animations and interpolating positions over time.

:p What is linear interpolation (LERP) and how does it work?
??x
Linear interpolation, often abbreviated as LERP, finds an intermediate point between two known points along a line segment. The operation is defined by the formula:
$$\mathbf{L} = \text{LERP}(\mathbf{A}, \mathbf{B}, b) = (1 - b)\mathbf{A} + b\mathbf{B}$$where $0 \leq b \leq 1$ is a scalar parameter that determines the position along the line segment.

Geometrically, LERP finds the point on the line between points A and B at distance b from A. For example:
```java
public class LinearInterpolation {
    public static Vector3 lerp(Vector3 A, Vector3 B, float b) {
        return new Vector3(
            (1 - b) * A.x + b * B.x,
            (1 - b) * A.y + b * B.y,
            (1 - b) * A.z + b * B.z
        );
    }
}
```
This method provides a smooth transition between two vectors. For instance, if you want to animate an object from point A to B over time, LERP can be used to calculate the position at any given time.

x??

---

#### Matrices in 3D Transformations
Background context: Matrices are essential for representing linear transformations such as translation, rotation, and scaling in 3D space. They allow us to manipulate points and vectors efficiently using matrix multiplication.

:p What is a matrix and how is it used in game programming?
??x
A matrix is a rectangular array of m x n scalars arranged in rows and columns. In the context of 3D game programming, matrices are used to represent linear transformations like translation, rotation, and scaling. Matrices can be thought of as grids of numbers enclosed in square brackets.

For example, a 3x3 matrix M might look like this:
$$M = \begin{bmatrix}
    m_{11} & m_{12} & m_{13} \\
    m_{21} & m_{22} & m_{23} \\
    m_{31} & m_{32} & m_{33}
\end{bmatrix}$$

Rows and columns of a 3x3 matrix can be considered as 3D vectors. If all the row and column vectors are unit magnitude, the matrix is called an orthogonal or orthonormal matrix, which represents pure rotations.

Transformation matrices, specifically 4x4 matrices, can represent arbitrary 3D transformations including translations, rotations, and scaling. These are crucial for game development as they allow us to apply complex transformations efficiently.

To transform a point or vector using a matrix, we perform matrix multiplication. For example:
```java
public class MatrixTransform {
    public static Vector3 transform(Vector3 v, float[][] M) {
        return new Vector3(
            (M[0][0] * v.x + M[1][0] * v.y + M[2][0] * v.z),
            (M[0][1] * v.x + M[1][1] * v.y + M[2][1] * v.z),
            (M[0][2] * v.x + M[1][2] * v.y + M[2][2] * v.z)
        );
    }
}
```
This method applies the transformation matrix to a 3D vector.

x??

---

#### Affine Transformation Matrix
Affine matrices are 4×4 transformation matrices used to preserve parallelism and relative distance ratios but not necessarily absolute lengths and angles. They can perform combinations of rotation, translation, scaling, and shear.

:p What is an affine matrix?
??x
An affine matrix is a type of transformation matrix that can combine operations such as rotation, translation, scaling, and shearing while preserving the parallelism of lines and relative distances between points but not necessarily absolute lengths and angles.
x??

---

#### Matrix Multiplication
Matrix multiplication involves combining two matrices to produce another matrix that performs both transformations. If A and B are transformation matrices, their product P = AB is also a transformation matrix.

:p What does the product of two transformation matrices represent?
??x
The product of two transformation matrices represents a new transformation matrix that combines both original transformations. For example, if one matrix scales and another rotates, their product will perform both scaling and rotation.
x??

---

#### Matrix Product Calculation
To calculate the matrix product P = AB, we take dot products between the rows of A and the columns of B. Each dot product becomes one component of the resulting matrix.

:p How is a matrix product calculated?
??x
A matrix product P = AB is calculated by taking dot products between the rows of matrix A and the columns of matrix B. The result is a new matrix where each element is computed as the dot product of corresponding row from A and column from B.
For example, if $A $ and$B$ are 3×3 matrices:
$$P = AB = \begin{pmatrix}
P_{11} & P_{12} & P_{13} \\
P_{21} & P_{22} & P_{23} \\
P_{31} & P_{32} & P_{33}
\end{pmatrix} = \begin{pmatrix}
A_{row1} \cdot B_{col1} & A_{row1} \cdot B_{col2} & A_{row1} \cdot B_{col3} \\
A_{row2} \cdot B_{col1} & A_{row2} \cdot B_{col2} & A_{row2} \cdot B_{col3} \\
A_{row3} \cdot B_{col1} & A_{row3} \cdot B_{col2} & A_{row3} \cdot B_{col3}
\end{pmatrix}$$x??

---

#### Matrix Multiplication Order
Matrix multiplication is not commutative, meaning $AB \neq BA$. The order in which matrices are multiplied affects the resulting transformation.

:p Why does matrix multiplication order matter?
??x
Matrix multiplication order matters because it affects the sequence of transformations. For example, multiplying a rotation followed by scaling results in different transformations compared to scaling followed by rotating.
x??

---

#### Concatenation and Transformation Order
Concatenating transformation matrices means applying them sequentially. The product of multiple matrices represents the combined effect of all individual transformations.

:p What does concatenation mean in matrix multiplication?
??x
Concatenation in matrix multiplication refers to combining multiple transformation matrices so that they are applied in sequence. If $A, B,$ and $ C $ represent three consecutive transformations, their product $(ABC)$ represents the combined effect of all these transformations.
x??

---

#### Row Vectors vs Column Vectors
Points and vectors can be represented as row or column matrices depending on the convention used. The choice affects how matrix multiplication is performed.

:p How do you choose between row vectors and column vectors?
??x
The choice between row vectors and column vectors depends on the context. In this book, we use row vectors for simplicity because they align with the left-to-right order of transformations in English.
However, both choices are valid. The key difference is that multiplying a 1×n row vector by an n×n matrix requires the row to appear to the left, while multiplying an n×n matrix by an n×1 column vector requires the column to appear to the right.
x??

---

#### Transformations with Row Vectors
When using row vectors for transformations, they "read" from left to right. The closest transformation matrix is applied first.

:p How do you read transformations when using row vectors?
??x
When using row vectors, transformations are read from left to right. This means the first matrix in the sequence (closest to the vector) is applied first. For example, if $v $ is a row vector and matrices$A, B, C$ represent transformations:
$$v' = (((vA)B)C)$$

Here,$v $ is transformed by$A $, then the result is transformed by$ B $, and finally by$ C$.
x??

---

#### Transformations with Column Vectors
When using column vectors for transformations, they "read" from right to left. The last matrix in the sequence (farthest from the vector) is applied first.

:p How do you read transformations when using column vectors?
??x
When using column vectors, transformations are read from right to left. This means the last matrix in the sequence (farthest from the vector) is applied first. For example, if $v $ is a column vector and matrices$A, B, C$ represent transformations:
$$v' = (C^T(B^T(A^Tv^T)))$$

Here,$v $ is transformed by$A $, then the result is transformed by$ B $, and finally by$ C$.
x??

---

---
#### Column Vector vs. Row Vector
Background context explaining the difference between column vectors and row vectors, including how they are used in matrix multiplication.

:p What is the difference between column vectors and row vectors?
??x
Column vectors have elements arranged vertically (n×1), while row vectors have elements arranged horizontally (1×n). When using column vectors, you need to transpose all matrices shown in this book because vector-matrix multiplications are written with the vector on the right of the matrix. 

For example, if a vector $\mathbf{v}$ is represented as a column vector:
```java
// Column Vector (Java representation)
double[] v = {1, 2, 3};
```
and you want to multiply it by a matrix $\mathbf{M}$, the multiplication would be written as $\mathbf{M} \cdot \mathbf{v}$.

However, if row vectors are used instead:
```java
// Row Vector (Java representation)
double[] v = {1, 2, 3};
```
the matrix-vector multiplication would be written as $\mathbf{v} \cdot \mathbf{M}^T $, where $\mathbf{M}^T $ is the transpose of$\mathbf{M}$.
x??

---
#### Identity Matrix
Explanation of what an identity matrix is, including its properties and representation.

:p What is an identity matrix?
??x
An identity matrix is a square matrix that yields the same matrix when multiplied by any other matrix. It is usually represented by the symbol $I$. The identity matrix has 1’s along the diagonal and 0’s everywhere else. For example, for a 3×3 identity matrix:
```java
// Identity Matrix (Java representation)
double[][] I = {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1}
};
```
The properties of the identity matrix are $AI = IA \rightarrow A$.

For a 3×3 identity matrix:
```java
double[][] I3x3 = {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1}
};
```
Multiplying any matrix $A $ by the identity matrix$I $ will yield the same matrix$A$.
x??

---
#### Matrix Inversion
Explanation of what a matrix inverse is and how it works.

:p What is a matrix inverse?
??x
The inverse of a matrix $A $(denoted as $ A^{-1}$) is another matrix that undoes the effects of matrix $ A$. For example, if $ A$rotates objects by 37 degrees about the z-axis, then $ A^{-1}$will rotate by -37 degrees about the z-axis. If a matrix scales objects to be twice their original size, then its inverse $ A^{-1}$ will scale objects to be half their size.

The property of an inverse matrix is that when multiplied by the original matrix or vice versa, it results in the identity matrix:
```java
double[][] A = { /* some values */ };
double[][] AI = MatrixInverse(A); // Function to compute the inverse

// Verify multiplication with the identity matrix
double[][] result = multiplyMatrix(A, AI);
assert Arrays.deepEquals(result, I); // Assuming 'I' is the 3x3 identity matrix

result = multiplyMatrix(AI, A);
assert Arrays.deepEquals(result, I);
```
However, not all matrices have inverses. Affine transformations (combinations of pure rotations, translations, scales, and shears) do have inverses.

The inverse can be found using methods like Gaussian elimination or lower-upper (LU) decomposition.
x??

---
#### Transposition
Explanation of what transposition is and how it works with matrices.

:p What is matrix transposition?
??x
Transposition involves reflecting the entries of a matrix across its diagonal. In other words, rows become columns and vice versa. For example, for a 3×3 matrix:
```java
// Original Matrix (Java representation)
double[][] M = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};

// Transposed Matrix (Java representation)
double[][] MT = transposeMatrix(M); // Function to transpose the matrix

// Transpose Logic
for(int i = 0; i < M.length; i++) {
    for(int j = 0; j < i; j++) { // Only transposing up to the diagonal
        double temp = M[i][j];
        M[i][j] = M[j][i];
        M[j][i] = temp;
    }
}
```
The transpose is useful, especially when dealing with orthonormal matrices (pure rotations), where the inverse of such a matrix is exactly equal to its transpose. This can save computational resources.

When moving data between libraries that use different conventions for vectors (column vs row), transposition will be necessary.
x??

---
#### Homogeneous Coordinates
Explanation of homogeneous coordinates and their application in 2D rotation using matrices.

:p What are homogeneous coordinates?
??x
Homogeneous coordinates are a method used to represent points in $n+1 $ dimensions as an$n $-dimensional vector. In the context of 2D rotations, a 2×2 matrix can be used to rotate a point by an angle $\phi$. The rotation is achieved through the following transformation:
```java
// Rotation Matrix (Java representation)
double[][] R = {
    {Math.cos(phi), -Math.sin(phi)},
    {Math.sin(phi), Math.cos(phi)}
};

// Original Vector (Column Vector, Java representation)
double[] v = {1, 2};
```
To rotate the vector $r $ through an angle$\phi$:
```java
// Rotated Vector Calculation
double[][] M = R; // Rotation matrix
double[] rPrime = multiplyMatrix(M, v); // Multiply rotation matrix by vector

// Function to multiply a matrix by a vector (Java representation)
public double[] multiplyMatrix(double[][] A, double[] v) {
    int n = A.length;
    double[] result = new double[n];
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < v.length; j++) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}
```
This method allows representing transformations such as rotation, translation, and scaling in a unified manner.
x??

---

#### 3D Rotations and Translations Using Matrices

Background context explaining the concept. In 3D graphics, rotations are commonly represented using 3×3 matrices due to their simplicity and efficiency.

In the given example:
$$\begin{bmatrix} r'_{x} \\ r'_{y} \\ z \end{bmatrix} = \begin{bmatrix} r_{x} & r_{y} & r_{z} \end{bmatrix} \cdot \begin{bmatrix} \cos\phi & \sin\phi & 0 \\ -\sin\phi & \cos\phi & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

This is a rotation about the z-axis by an angle $\phi$.

:q Can a 3×3 matrix be used to represent translations in 3D space?
??x
No, because translating a point requires adding the components of translation $t $ to the corresponding components of the point$r$ individually. Matrix multiplication involves both multiplication and addition of matrix elements, which cannot achieve the necessary summation form needed for translations.

Here's why:
- Consider the matrix $\begin{bmatrix} A & B \\ C & D \end{bmatrix}$ and a column vector $r = \begin{bmatrix} r_x \\ r_y \\ r_z \end{bmatrix}$.
- The result of multiplying this with $r $ would involve terms like$A*r_x + B*r_y + C*r_z $, which cannot directly achieve the form$(r_x + t_x)$ needed for translation.

Therefore, translations must be handled separately from rotations and scales using a different representation.
x??

---

#### 4×4 Matrix for Combining Transformations

Background context explaining the concept. In 3D graphics, a combination of transformations (translation, rotation, scaling) is often required. A 4×4 matrix can handle all these operations effectively.

A 4×4 transformation matrix in homogeneous coordinates looks like this:
$$T = \begin{bmatrix} 
1 & 0 & 0 & t_x \\ 
0 & 1 & 0 & t_y \\ 
0 & 0 & 1 & t_z \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

Where $t_x, t_y, t_z$ are the translation components.

:q How can a 4×4 matrix represent translations in homogeneous coordinates?
??x
A 4×4 matrix with a fourth column containing the translation components (e.g.,$[0, 0, 1, 0]$ for rotation and scale, and $[t_x, t_y, t_z, 1]$ for translation) can be used to represent translations. By setting the fourth element of the position vector $r$ to 1 (i.e., writing it in homogeneous coordinates), we can achieve the desired sums when multiplying by the matrix.

For example:
$$\begin{bmatrix} r_x \\ r_y \\ r_z \\ 1 \end{bmatrix} \cdot \begin{bmatrix} 
1 & 0 & 0 & t_x \\ 
0 & 1 & 0 & t_y \\ 
0 & 0 & 1 & t_z \\ 
0 & 0 & 0 & 1 
\end{bmatrix} = \begin{bmatrix} (r_x + t_x) \\ (r_y + t_y) \\ (r_z + t_z) \\ 1 \end{bmatrix}$$

This results in a vector with the translation applied.

x??

---

#### Transforming Direction Vectors

Background context explaining the concept. When transforming vectors in 3D space, points and direction vectors are treated differently due to their nature.

For points (position vectors), both rotation and translation components of the matrix are applied.
- Example:
$$\begin{bmatrix} r_x \\ r_y \\ r_z \\ 1 \end{bmatrix} \cdot M = \text{(Transformed Point)}$$

For direction vectors, only the rotation component is applied. The translation part does not affect direction since it would change magnitude.

:q How are direction vectors transformed differently from points in 3D space?
??x
Direction vectors are treated differently from points because they do not inherently have a translation component. Applying a translation to a direction vector would alter its magnitude, which is generally undesirable.

In homogeneous coordinates:
- Points use $w = 1$.
- Direction vectors use $w = 0$.

This ensures that only the rotation part of the transformation matrix affects direction vectors:

$$\begin{bmatrix} v_x \\ v_y \\ v_z \\ 0 \end{bmatrix} \cdot M = \begin{bmatrix} (v_x \times U_{1x}) + (v_y \times U_{2x}) + (v_z \times U_{3x}) \\ (v_x \times U_{1y}) + (v_y \times U_{2y}) + (v_z \times U_{3y}) \\ (v_x \times U_{1z}) + (v_y \times U_{2z}) + (v_z \times U_{3z}) \\ 0 \end{bmatrix}$$

Here, the $w = 0$ component ensures that no translation is applied.

x??

---

#### Converting Homogeneous Coordinates to Non-Homogeneous

Background context explaining the concept. In some cases, it's necessary to convert a vector in homogeneous coordinates back to non-homogeneous (3D) coordinates.

:q How can a point in homogeneous coordinates be converted to 3D (non-homogeneous) coordinates?
??x
A point in homogeneous coordinates can be converted to its 3D (non-homogeneous) form by dividing the $x $, $ y $, and$ z $components by the$ w$component:
$$\begin{bmatrix} x \\ y \\ z \\ w \end{bmatrix} = \frac{1}{w} \begin{bmatrix} x \\ y \\ z \end{bmatrix}$$

For example, if a point is represented as:
$$\begin{bmatrix} 2 \\ 3 \\ 4 \\ 2 \end{bmatrix}$$

The non-homogeneous form would be:
$$\begin{bmatrix} 1 \\ 1.5 \\ 2 \end{bmatrix}$$

This conversion effectively normalizes the point to its original 3D coordinates.

x??

---

#### Homogeneous Coordinates and W-Component
Background context explaining the concept of homogeneous coordinates and why points and vectors are treated differently. Specifically, the w-component for a point is set to 1, while it is set to 0 for a vector.

:p Why do we set the w-component to different values for points and vectors?
??x
The w-component is set to 1 for points because dividing by 1 (which effectively means no change) does not alter the coordinates of the point. For vectors, setting the w-component to 0 implies that any attempt at translation would result in an undefined form, as division by zero is not defined.

For example:
- A point in homogeneous coordinates:$[x_p, y_p, z_p, 1]$- A vector in homogeneous coordinates:$[x_v, y_v, z_v, 0]$ This treatment helps distinguish between points and vectors mathematically. The point at infinity is a special case where the w-component can be 0 but only for certain transformations like rotation.

---
#### Affine Transformation Matrices
Background context explaining affine transformation matrices and how they are composed of translation, rotation, scale, and shear operations.

:p What is an affine transformation matrix in 3D space?
??x
An affine transformation matrix in 3D space can be created by concatenating transformations such as pure translations, rotations, scales, and shears. These atomic transformations form the building blocks for more complex transformations.

The general form of an affine 4×4 transformation matrix is:
$$M_{\text{affine}} = \begin{bmatrix}
U & t \\
0 & 1
\end{bmatrix}$$where $ U $is a 3×3 upper triangular matrix representing the rotation and/or scale,$ t$ is a 1×3 translation vector, and the last row ensures that when multiplying by homogeneous coordinates, we get back to a valid 4D point.

For example:
```java
public class AffineTransform {
    private double[] U; // 3x3 matrix elements
    private double[] t; // 1x3 translation vector

    public AffineTransform(double[] u, double[] t) {
        this.U = u;
        this.t = t;
    }

    public Vector4D transform(Vector4D point) {
        return new Vector4D(
            U[0] * point.x + U[1] * point.y + U[2] * point.z + t[0],
            U[3] * point.x + U[4] * point.y + U[5] * point.z + t[1],
            U[6] * point.x + U[7] * point.y + U[8] * point.z + t[2],
            1.0 // Homogeneous coordinate
        );
    }
}
```
x??

---
#### Translation Matrix in Affine Transformations
Background context explaining the specific case of translation within affine transformations, including how it affects homogeneous coordinates.

:p What is a translation matrix and how does it work?
??x
A translation matrix translates a point by adding its 1×3 translation vector $t$ to the point's coordinates. This operation ensures that points are moved in space while vectors remain unaffected due to their zero w-component.

The general form of a translation matrix is:
$$T = \begin{bmatrix}
I & t \\
0 & 1
\end{bmatrix}$$where $ I $ is the identity matrix and $ t$ is the translation vector. When applied, it performs the transformation:
$$[x', y', z', 1] = [x, y, z, 1] \cdot T$$

For example, a translation by $(tx, ty, tz)$:
```java
public class TranslationMatrix {
    private double[] t;

    public TranslationMatrix(double tx, double ty, double tz) {
        this.t = new double[]{tx, ty, tz};
    }

    public Vector4D applyTranslation(Vector4D point) {
        return new Vector4D(
            point.x + t[0],
            point.y + t[1],
            point.z + t[2],
            1.0 // Homogeneous coordinate
        );
    }
}
```
x??

---
#### Rotation Matrices in Affine Transformations
Background context explaining the specific forms of rotation matrices around different axes and their application.

:p What are the rotation matrices for each axis in 3D space?
??x
Rotation matrices for each axis in 3D space follow a specific form. Each matrix rotates points about an axis by a given angle $\theta$.

- Rotation about the x-axis:
$$R_x(\phi) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos(\phi) & -\sin(\phi) & 0 \\
0 & \sin(\phi) & \cos(\phi) & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$- Rotation about the y-axis:
$$

R_y(q) = \begin{bmatrix}
\cos(q) & 0 & -\sin(q) & 0 \\
0 & 1 & 0 & 0 \\
\sin(q) & 0 & \cos(q) & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$- Rotation about the z-axis:
$$

R_z(\gamma) = \begin{bmatrix}
\cos(\gamma) & -\sin(\gamma) & 0 & 0 \\
\sin(\gamma) & \cos(\gamma) & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

These matrices allow for the rotation of points in 3D space around their respective axes.

For example:
```java
public class RotationMatrix {
    private double angle; // Angle in radians

    public RotationMatrix(double angle, String axis) {
        this.angle = angle;
        if (axis.equals("x")) {
            matrix = new double[][]{
                {1, 0, 0, 0},
                {0, Math.cos(angle), -Math.sin(angle), 0},
                {0, Math.sin(angle), Math.cos(angle), 0},
                {0, 0, 0, 1}
            };
        } else if (axis.equals("y")) {
            matrix = new double[][]{
                {Math.cos(angle), 0, -Math.sin(angle), 0},
                {0, 1, 0, 0},
                {Math.sin(angle), 0, Math.cos(angle), 0},
                {0, 0, 0, 1}
            };
        } else if (axis.equals("z")) {
            matrix = new double[][]{
                {Math.cos(angle), -Math.sin(angle), 0, 0},
                {Math.sin(angle), Math.cos(angle), 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}
            };
        }
    }

    public Vector4D applyRotation(Vector4D point) {
        // Apply rotation matrix to the point
        return new Vector4D(
            point.x * matrix[0][0] + point.y * matrix[1][0] + point.z * matrix[2][0] + matrix[3][0],
            point.x * matrix[0][1] + point.y * matrix[1][1] + point.z * matrix[2][1] + matrix[3][1],
            point.x * matrix[0][2] + point.y * matrix[1][2] + point.z * matrix[2][2] + matrix[3][2],
            1.0 // Homogeneous coordinate
        );
    }
}
```
x??

---

#### Scaling Matrices
Scaling matrices are used to scale a point $\mathbf{r} = [x, y, z]^T$ by factors along each axis. The matrix for scaling is given as:
$$\mathbf{r_S} = \begin{bmatrix}
x & y & z & 1 \\
\end{bmatrix} 
\begin{bmatrix}
s_x & 0 & 0 & 0 \\
0 & s_y & 0 & 0 \\
0 & 0 & s_z & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} 
= \begin{bmatrix}
s_x x \\
s_y y \\
s_z z \\
1
\end{bmatrix} = [s_x r_x, s_y r_y, s_z r_z, 1]^T$$

Here $s_x, s_y, s_z$ are the scaling factors along the x, y, and z axes respectively.

:p How does a scaling matrix affect a point in 3D space?
??x
A scaling matrix affects each coordinate of a point by multiplying it with a corresponding scaling factor. For example, if you have a point $[x, y, z]$ and you apply a scaling matrix where the factors are $ s_x, s_y, s_z $, then the new coordinates become $[s_x x, s_y y, s_z z]$.

For instance, if we want to scale a point by 2 along the x-axis, 3 along the y-axis and leave the z-axis unchanged (i.e., scaling factor is 1 for z), the matrix will be:

```java
// Scaling Matrix
Matrix4f scaleMatrix = new Matrix4f();
scaleMatrix.m00 = 2; // s_x = 2
scaleMatrix.m11 = 3; // s_y = 3
scaleMatrix.m22 = 1; // s_z = 1

// Point before scaling
Vector3f point = new Vector3f(1, 2, 3);

// Applying the scale matrix to the point
point.mulLocal(scaleMatrix); // Result: [2, 6, 3]
```

x??

---

#### Inverting a Scaling Matrix
To invert a scaling matrix, you simply replace $s_x, s_y,$ and $s_z$ with their reciprocals. This effectively undoes the original scaling.

For example, if we have a scaling matrix:

$$S = \begin{bmatrix}
2 & 0 & 0 & 0 \\
0 & 3 & 0 & 0 \\
0 & 0 & 4 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

The inverse scaling matrix would be:
$$

S^{-1} = \begin{bmatrix}
1/2 & 0 & 0 & 0 \\
0 & 1/3 & 0 & 0 \\
0 & 0 & 1/4 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$:p How is an inverse scaling matrix calculated?
??x
An inverse scaling matrix is computed by taking the reciprocal of each non-zero diagonal element in the original scaling matrix. If $S = [s_x, s_y, s_z]$, then $ S^{-1} = [1/s_x, 1/s_y, 1/s_z]$.

For example, given a scaling matrix with factors 2, 3, and 4 along the x, y, and z axes respectively:

```java
// Original Scaling Matrix
Matrix4f scaleMatrix = new Matrix4f();
scaleMatrix.m00 = 2; // s_x = 2
scaleMatrix.m11 = 3; // s_y = 3
scaleMatrix.m22 = 4; // s_z = 4

// Inverting the scaling matrix by taking reciprocals of each non-zero diagonal element
Matrix4f inverseScaleMatrix = new Matrix4f();
inverseScaleMatrix.setIdentity(); // Initialize to identity matrix
inverseScaleMatrix.m00 = 1 / scaleMatrix.m00; // s_x = 2, so 1/2
inverseScaleMatrix.m11 = 1 / scaleMatrix.m11; // s_y = 3, so 1/3
inverseScaleMatrix.m22 = 1 / scaleMatrix.m22; // s_z = 4, so 1/4

// Resulting inverse scaling matrix:
// S^-1 = [0.5, 0, 0, 0]
//        [0, 1/3, 0, 0]
//        [0, 0, 1/4, 0]
//        [0, 0, 0, 1]
```

x??

---

#### Uniform vs. Nonuniform Scaling
Uniform scaling occurs when the same scale factor is applied to all three axes (i.e., $s_x = s_y = s_z$). This means that shapes remain similar in form under uniform scaling, such as spheres remaining spherical.

Non-uniform scaling, where each axis has a different scaling factor ($s_x \neq s_y \neq s_z$), results in non-similar transformations. For example, a sphere would become an ellipsoid.

:p What is the difference between uniform and nonuniform scaling?
??x
Uniform scaling involves applying the same scale factor to all axes (i.e., $s_x = s_y = s_z$). This type of transformation preserves the shape's proportions, ensuring that geometric properties like angles remain consistent. For instance, a sphere remains spherical under uniform scaling.

Non-uniform scaling uses different scale factors along each axis ($s_x \neq s_y \neq s_z$), which can distort shapes into non-similar forms. A typical example is how a sphere might be transformed into an ellipsoid.

For uniform scaling:

```java
// Example of uniform scaling with factor 2
Matrix4f scaleMatrix = new Matrix4f();
scaleMatrix.m00 = 2; // s_x = 2, same for s_y and s_z
scaleMatrix.m11 = 2;
scaleMatrix.m22 = 2;

// Applying this to a point [x, y, z] will result in [2x, 2y, 2z]
```

For non-uniform scaling:

```java
// Example of non-uniform scaling with factors 2, 3, and 1
Matrix4f scaleMatrix = new Matrix4f();
scaleMatrix.m00 = 2; // s_x = 2
scaleMatrix.m11 = 3; // s_y = 3
scaleMatrix.m22 = 1; // s_z = 1

// Applying this to a point [x, y, z] will result in [2x, 3y, z]
```

x??

---

#### Concatenating Uniform Scale and Rotation Matrices
When the uniform scale matrix $S_u $ is concatenated with a rotation matrix$R $, the order of multiplication does not matter (i.e.,$ S_uR = RS_u$). This property only holds for uniform scaling.

:p What happens when you concatenate a uniform scale matrix with a rotation matrix?
??x
When a uniform scale matrix $S_u $ is concatenated with a rotation matrix$R $, the order of multiplication does not affect the final result. Mathematically, this means that both$ S_uR $and$ RS_u$ yield the same transformation.

For example, consider:

```java
// Uniform Scale Matrix with factor 2
Matrix4f scaleMatrix = new Matrix4f();
scaleMatrix.m00 = 2;
scaleMatrix.m11 = 2;
scaleMatrix.m22 = 2;

// Rotation Matrix (simple rotation around z-axis by π/4)
Matrix4f rotationMatrix = new Matrix4f().rotateZ(Math.PI / 4);

// Concatenating scale and rotation
Matrix4f combinedScaleRotation = scaleMatrix.mulLocal(rotationMatrix);
Matrix4f combinedRotationScale = rotationMatrix.mulLocal(scaleMatrix);

// Both matrices should be identical if the operation is valid
```

In this example, both $S_uR $ and$RS_u$ produce the same transformation matrix.

x??

---

#### 4×3 Matrices in Game Programming
Game programmers often use 4×3 affine matrices to save memory because they only store the necessary values for transformations. The rightmost column of a 4×4 matrix is always `[0, 0, 0, 1]`, and game libraries frequently omit this redundant information.

:p Why do game engines use 4×3 matrices instead of full 4×4 matrices?
??x
Game engines use 4×3 affine matrices to save memory because they eliminate the redundancy in a 4×4 matrix. In a 4×4 matrix, the rightmost column is always `[0, 0, 0, 1]`, which means this information does not need to be stored and can be implied.

Using 4×3 matrices reduces storage requirements, making them more efficient for large-scale applications such as games that handle many transformations per frame. However, it also means that the full transformation properties are not fully represented in a single matrix; additional operations might require explicit use of the missing column to ensure correctness.

For example, when working with translations and rotations, using 4×3 matrices can save memory compared to using full 4×4 matrices:

```java
// Example of a 4x3 affine transformation matrix for translation
Matrix4f affineMatrix = new Matrix4f();
affineMatrix.m03 = 2; // Translation along x-axis by 2 units
affineMatrix.m13 = 3; // Translation along y-axis by 3 units

// Applying this to a point [x, y, z] results in [x+2, y+3, z, 1]
```

While 4×3 matrices are more memory-efficient, they do not provide all the same transformation properties as full 4×4 matrices.

x??

---

#### Coordinate Spaces
Coordinate spaces are used to describe the position and orientation of objects relative to a particular reference frame. In games, common coordinate spaces include model space (object space), world space, and view space.

- **Model Space**: Positions of vertices in a mesh are defined relative to the object's local origin.
- **World Space**: Positions are described relative to an absolute global coordinate system.
- **View Space**: Positions are transformed into the camera’s coordinate system for rendering purposes.

:p What is model space?
??x
Model space, also known as object space or local space, refers to a coordinate system where the positions of vertices in a mesh are defined relative to the object's local origin. This means that transformations applied directly affect the object itself without considering any external reference frames.

For example, if you have an object represented by its model matrix $M$, and you want to transform its vertices from model space to world space, you would multiply each vertex position by the model matrix:

```java
// Model Matrix representing the transformation in model space
Matrix4f modelMatrix = new Matrix4f(); // Example initialization

// Vertex position in model space
Vector3f vertexModelSpace = new Vector3f(1, 2, 3);

// Transforming to world space
vertexWorldSpace.mulLocal(modelMatrix);
```

In this example, `vertexWorldSpace` will contain the transformed position relative to the world coordinate system.

x??

---

#### Model-Space Origin and Axes
Background context explaining that the model-space origin is typically placed at a central location within an object, such as its center of mass or between the feet for a humanoid character. The axes are aligned to natural directions on the model, often labeled as front, up, and left/right.

:p What is the typical placement of the model-space origin?
??x
The model-space origin is usually placed at a central location within an object such as its center of mass or between the feet for a humanoid character. This ensures that transformations are centered around a meaningful point.
x??

---

#### Model-Space Axes Directions
Background context explaining the intuitive directions given to the axes, like front, up, and left/right. These names help in defining rotations and orientations.

:p What are the typical names for model-space axes?
??x
The typical names for model-space axes are:
- Front: Points in the direction that the object naturally travels or faces.
- Up: Points towards the top of the object.
- Left/Right: Points toward the left or right side of the object, depending on whether your game engine uses left-handed or right-handed coordinates.

In a right-handed coordinate system, front is typically assigned to the positive z-axis, up to the positive y-axis, and left to the positive x-axis (F=k, U=j, L=i). In some engines, +x could be front, and +z could be right (F=i, R=k, U=j).
x??

---

#### Euler Angles in Model-Space
Background context explaining how Euler angles (pitch, yaw, roll) are defined using the model-space basis vectors for clarity.

:p How are pitch, yaw, and roll defined in terms of the model-space basis vectors?
??x
Pitch is rotation about the left or right axis (L or R).
Yaw is rotation about the up axis (U).
Roll is rotation about the front axis (F).

In a right-handed coordinate system with F=k, U=j, L=i:
- Pitch: Rotation about the x-axis (left or right)
- Yaw: Rotation about the y-axis (up)
- Roll: Rotation about the z-axis (front)

:p How are pitch, yaw, and roll defined in terms of model-space basis vectors?
??x
- Pitch is rotation about the left or right axis (L).
- Yaw is rotation about the up axis (U).
- Roll is rotation about the front axis (F).

In a typical right-handed coordinate system:
```java
public class EulerAngleRotations {
    public void applyPitch(float angle, Vector3f axis) { // Rotation about L
        // Apply pitch rotation logic here
    }

    public void applyYaw(float angle, Vector3f axis) {  // Rotation about U
        // Apply yaw rotation logic here
    }

    public void applyRoll(float angle, Vector3f axis) { // Rotation about F
        // Apply roll rotation logic here
    }
}
```
x??

---

#### World Space in Game Engines
Background context explaining the role of world space as a fixed coordinate system that ties all objects together into one cohesive virtual world.

:p What is world space and its significance?
??x
World space is a fixed coordinate system where positions, orientations, and scales of all objects in the game world are expressed. It helps tie all individual objects together into a cohesive virtual world.
The origin location in world space can be arbitrary but is often placed near the center of the playable game space to minimize precision loss when coordinates grow large.

:p What are the common conventions for axes orientation in world space?
??x
Common conventions for axes orientation in world space include:
- y-up: The y-axis goes up and the x-axis goes right.
- z-up: The z-axis is oriented vertically, often used as an extension of 2D conventions found in most mathematics textbooks.

The choice between these depends on the engine’s requirements but consistency is key throughout the engine implementation.
x??

---

#### Aircraft Transformation in 3D Space

Background context: This section explains how an aircraft's position and orientation can be transformed from model space to world space. It describes a scenario where an aircraft is rotated about the y-axis and translated, resulting in specific coordinates for its wingtip.

:p What happens when the left wingtip of a Lear jet at (5, 0, 0) in model space is rotated by 90 degrees about the world-space y-axis and then translated to (–25, 50, 8)?

??x
When the aircraft is rotated by 90 degrees about the y-axis, its new coordinates for the left wingtip change from (5, 0, 0) in model space to (0, 0, -5) in world space. However, since the origin of the aircraft has been translated to (–25, 50, 8), we need to apply this translation to the new coordinates. The final position of the left wingtip is thus calculated as:

$$(0 + (-25), 0 + 50, -5 + 8) = (-25, 50, 3)$$

This transformation accounts for both the rotation and translation required to represent the aircraft in world space.

??x
```java
public class Aircraft {
    public void transform(double[] modelSpaceOrigin, double[] rotationAngle, double[] translationVector, double[] pointInModelSpace) {
        // Assuming 90 degrees rotation about y-axis, we only change z to -5
        double[] transformedPoint = {pointInModelSpace[1], pointInModelSpace[2], -pointInModelSpace[0]};
        
        // Apply the translation vector
        transformedPoint[0] += modelSpaceOrigin[0];
        transformedPoint[1] += modelSpaceOrigin[1];
        transformedPoint[2] += modelSpaceOrigin[2];
        
        System.out.println("Transformed Point: (" + transformedPoint[0] + ", " + transformedPoint[1] + ", " + transformedPoint[2] + ")");
    }
    
    public static void main(String[] args) {
        double[] modelSpaceOrigin = {-25, 50, 8};
        double rotationAngleY = Math.PI / 2; // 90 degrees
        double[] translationVector = {0, 0, 0}; // Placeholder for actual vector
        
        Aircraft aircraft = new Aircraft();
        aircraft.transform(modelSpaceOrigin, rotationAngleY, translationVector, new double[]{5, 0, 0});
    }
}
```
x??

---

#### View Space (Camera Space) Definition

Background context: The text explains the concept of view space or cameraspace and its use in computer graphics. It provides an explanation of how this coordinate system is fixed to the camera's perspective.

:p What defines a view space, also known as camera space?

??x
A view space (or camera space) is a coordinate frame that is fixed relative to the camera. The origin of the view space is typically placed at the focal point of the camera. This means any transformation or object position in this space can be directly related to the camera's perspective.

:p How does the orientation and direction differ between left-handed and right-handed view spaces?

??x
In a left-handed view space, the z-axis increases in the direction that the camera is facing (which is typically the positive direction), while in a right-handed view space, the z-coordinate decreases as the camera faces forward. 

The difference lies in how the depth information is represented:
- In a left-handed system, z-values increase into the screen.
- In a right-handed system, z-values decrease into the screen.

This distinction affects how objects are positioned and rendered based on their distance from the camera.

??x
```java
public class Camera {
    private boolean handedness; // true for left-handed, false for right-handed

    public void setViewSpaceOrientation(boolean handedness) {
        this.handedness = handedness;
    }

    public boolean getViewSpaceOrientation() {
        return this.handedness;
    }
}
```
x??

---

#### Coordinate Space Hierarchies in 3D Graphics

Background context: The text discusses the hierarchical nature of coordinate spaces, explaining that every coordinate system is a child of another, forming a tree structure. World space is described as the root node.

:p What does it mean when coordinate frames are relative to each other?

??x
Coordinate frames are relative because they provide positional and orientational information for objects in 3D space only with respect to some other set of axes or reference frame. For example, if an object's position in one coordinate system (say model space) is described as (5, 0, 0), this makes sense only when you have a specific origin point and orientation defined for that space.

:p How does the concept of a hierarchy apply to coordinate spaces?

??x
The hierarchy of coordinate spaces means that every coordinate system in 3D graphics has a parent. The world space is at the root of this hierarchy, meaning all other coordinate systems are derived relative to it. For instance, if an aircraft is described in its local model space and then translated or rotated into world space, these transformations can be seen as moving through the hierarchy from model space to world space.

:p What is the significance of world space being the root?

??x
World space serves as the top-level coordinate system that provides a universal reference point for all objects in the 3D scene. Every other object's position and orientation are described relative to this global coordinate frame, making it easier to manage transformations and interactions between multiple objects.

??x
```java
public class CoordinateSystem {
    private String name;
    private double[] origin; // Origin of the coordinate system
    private double[] parentOrigin; // Reference to the parent's origin
    
    public CoordinateSystem(String name) {
        this.name = name;
        this.origin = new double[]{0, 0, 0}; // Default origin at (0,0,0)
        this.parentOrigin = null; // Initially no parent
    }
    
    public void setParent(CoordinateSystem parent) {
        this.parentOrigin = parent.getOrigin();
    }

    public double[] getOrigin() {
        return this.origin;
    }
}
```
x??

---

#### Change of Basis Matrix Concept
Background context: A change of basis matrix transforms points and directions from a child coordinate system (C) to its parent coordinate system (P). This transformation is crucial for understanding how positions and orientations are represented across different coordinate systems. The matrix $M_{C.P}$ indicates the transformation from child space to parent space.

:p What is the purpose of a change of basis matrix in game development?
??x
A change of basis matrix is used to convert coordinates and directions from one coordinate system (child space) to another (parent space). This transformation is essential for aligning objects or components within a scene where different parts might be defined relative to their local parent systems. It helps maintain consistency across hierarchies of objects in 3D scenes.
x??

---
#### Formula for Change of Basis Matrix
Background context: The change of basis matrix $M_{C.P}$ is constructed from the translation vector $ t_C $, and unit vectors $ i_C, j_C, k_C$. These vectors are expressed in parent coordinates.

Formula:
$$MC.P = 
\begin{pmatrix}
i_{Cx} & i_{Cy} & i_{Cz} & 0 \\
j_{Cx} & j_{Cy} & j_{Cz} & 0 \\
k_{Cx} & k_{Cy} & k_{Cz} & 0 \\
t_{Cx} & t_{Cy} & t_{Cz} & 1
\end{pmatrix}$$:p How is the change of basis matrix constructed?
??x
The change of basis matrix $M_{C.P}$ is constructed using the translation vector $ t_C $, and unit vectors $ i_C, j_C, k_C$. The unit vectors represent the axes of the child coordinate system in parent space coordinates. The matrix has four columns: three for the unit vectors (forming a 3x3 rotation part) and one for the translation.

```java
public static float[] buildChangeOfBasisMatrix(Vector3f iC, Vector3f jC, Vector3f kC, Vector3f tC) {
    // Constructing the matrix with row-major format
    float[] matrix = new float[16];
    matrix[0] = iC.x;  // iCx
    matrix[4] = iC.y;  // iCy
    matrix[8] = iC.z;  // iCz
    matrix[12] = 0;

    matrix[1] = jC.x;  // jCx
    matrix[5] = jC.y;  // jCy
    matrix[9] = jC.z;  // jCz
    matrix[13] = 0;

    matrix[2] = kC.x;  // kCx
    matrix[6] = kC.y;  // kCy
    matrix[10] = kC.z; // kCz
    matrix[14] = 0;

    matrix[3] = tC.x;  // tCx
    matrix[7] = tC.y;  // tCy
    matrix[11] = tC.z; // tCz
    matrix[15] = 1;
    return matrix;
}
```
x??

---
#### Rotation Matrix Example
Background context: A simple example of a rotation by an angle $\theta$ about the z-axis can be represented by:

$$\text{rotate\_z}(\theta, g) =
\begin{pmatrix}
\cos(g) & -\sin(g) & 0 & 0 \\
\sin(g) & \cos(g) & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}$$:p How does the rotation matrix for a z-axis rotation look?
??x
The rotation matrix for a z-axis rotation by an angle $\theta$ is given by:
$$\text{rotate\_z}(\theta, g) =
\begin{pmatrix}
\cos(g) & -\sin(g) & 0 & 0 \\
\sin(g) & \cos(g) & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}$$

This matrix rotates vectors in the x-y plane while keeping z and t (translation) components unchanged.

For instance, if $\theta = g$, then:

$$i_C = [\cos(g) \sin(g) 0]$$
$$j_C = [-\sin(g) \cos(g) 0]$$

When these vectors are plugged into the change of basis matrix formula with $k_C = [0, 0, 1]$, it matches the rotation matrix.

```java
public static float[] buildRotationMatrixZ(float angle) {
    // Constructing the rotation matrix for z-axis
    float cosG = (float)Math.cos(angle);
    float sinG = (float)Math.sin(angle);

    float[] matrix = new float[16];
    matrix[0] = cosG;
    matrix[1] = -sinG;
    matrix[2] = 0;

    matrix[4] = sinG;
    matrix[5] = cosG;
    matrix[6] = 0;

    matrix[8] = 0;
    matrix[9] = 0;
    matrix[10] = 1;

    return matrix;
}
```
x??

---
#### Scaling of Child Axes
Background context: Scaling the child coordinate system is achieved by scaling the unit basis vectors appropriately. If a child space is scaled up or down, this affects the lengths of the $i_C $, $ j_C $, and$ k_C$ vectors.

:p How does scaling affect the child axes in a change of basis matrix?
??x
Scaling the child coordinate system changes the length of the unit basis vectors. For example, if the child space is scaled up by a factor of 2, then the unit basis vectors $i_C $, $ j_C $, and$ k_C$ will be twice as long.

If we scale up the axes, their new lengths would be:
$$i_C' = [i_C.x * scalingFactor, i_C.y * scalingFactor, i_C.z * scalingFactor]$$
$$j_C' = [j_C.x * scalingFactor, j_C.y * scalingFactor, j_C.z * scalingFactor]$$
$$k_C' = [k_C.x * scalingFactor, k_C.y * scalingFactor, k_C.z * scalingFactor]$$

The change of basis matrix would then be:
$$

M_{C.P} =
\begin{pmatrix}
i_{Cx}' & i_{Cy}' & i_{Cz}' & 0 \\
j_{Cx}' & j_{Cy}' & j_{Cz}' & 0 \\
k_{Cx}' & k_{Cy}' & k_{Cz}' & 0 \\
t_{Cx} & t_{Cy} & t_{Cz} & 1
\end{pmatrix}$$```java
public static float[] scaleChangeOfBasisMatrix(float scalingFactor, Vector3f tC) {
    // Scaling the basis vectors
    Vector3f iCScaled = new Vector3f(iCx * scalingFactor, iCy * scalingFactor, iCz * scalingFactor);
    Vector3f jCScaled = new Vector3f(jCx * scalingFactor, jCy * scalingFactor, jCz * scalingFactor);
    Vector3f kCScaled = new Vector3f(kCx * scalingFactor, kCy * scalingFactor, kCz * scalingFactor);

    // Constructing the scaled matrix
    float[] matrix = new float[16];
    matrix[0] = iCScaled.x;
    matrix[4] = iCScaled.y;
    matrix[8] = iCScaled.z;
    matrix[12] = 0;

    matrix[1] = jCScaled.x;
    matrix[5] = jCScaled.y;
    matrix[9] = jCScaled.z;
    matrix[13] = 0;

    matrix[2] = kCScaled.x;
    matrix[6] = kCScaled.y;
    matrix[10] = kCScaled.z;
    matrix[14] = 0;

    matrix[3] = tC.x;
    matrix[7] = tC.y;
    matrix[11] = tC.z;
    matrix[15] = 1;
    return matrix;
}
```
x??

---
#### Extracting Unit Basis Vectors from a Matrix
Background context: Given an affine transformation matrix, we can extract the unit basis vectors $i_C $, $ j_C $, and$ k_C$ by isolating specific rows of the matrix. This is useful for extracting information like orientation without needing to explicitly calculate it.

:p How can we extract the child-space basis vectors from a given affine transformation matrix?
??x
To extract the unit basis vectors $i_C $, $ j_C $, and$ k_C$ from an affine transformation matrix, you simply need to look at the appropriate rows of the matrix. For example:

- The first three elements of the third row give $k_C$.
- The first three elements of the second row give $j_C$.
- The first three elements of the first row give $i_C$.

For a transformation matrix:
$$M = 
\begin{pmatrix}
a & b & c & 0 \\
d & e & f & 0 \\
g & h & i & 0 \\
j & k & l & 1
\end{pmatrix}$$

The basis vectors are:
- $i_C$ is [a, b, c]
- $j_C$ is [d, e, f]
- $k_C$ is [g, h, i]

```java
public static Vector3f extractK(Vector3f kC, float[] matrix) {
    // Extracting the k basis vector from the 3rd row of the matrix
    return new Vector3f(matrix[2], matrix[6], matrix[10]);
}
```
x??

---

#### Transforming Coordinate Systems vs Vectors

Background context: Matrices can be used to transform points and vectors from one coordinate system (child space) to another (parent space). The fourth row of a transformation matrix $M_{C.P}$ contains the translation vector $t_C$, which represents how the child's axes are translated relative to the world space. This can also be visualized as transforming the parent’s coordinate axes into the child’s axes, effectively reversing the direction in which points and vectors move.

:p How does a transformation matrix $M_{C.P}$ transform both points and coordinate axes?

??x
A transformation matrix $M_{C.P}$ transforms points and directions (vectors) from the child space to the parent space. Conversely, it can also be seen as transforming the parent’s coordinate axes into the child's axes. This is because moving a point 20 units in one direction with fixed axes is equivalent to moving the axes 20 units in the opposite direction while keeping the point fixed.

```java
// Example of transforming a vector v from child space to parent space using matrix M_C.P
Vector3D vChild = new Vector3D(1, 2, 3);
Matrix4x4f MC_P = getTransformationMatrix(); // Assume this is the transformation matrix from child to parent
Vector4D vParent = MC_P.transform(vChild); // Transform vector vChild into vParent in parent space

// To transform axes instead of points (reverse process)
Vector4D axisX = new Vector4D(1, 0, 0, 0);
Vector4D transformedAxisX = MC_P.transpose().inverse().transform(axisX);
```
x??

---

#### Transformation Conventions

Background context: In the book, it is conventionally agreed that transformations apply to vectors rather than coordinate axes. Additionally, vectors are represented as rows instead of columns. This makes matrix multiplication readable from left to right and ensures meaningful cancellation in sequences.

:p How does choosing transformation conventions impact how you work with matrices?

??x
Choosing conventions such as applying transformations to vectors over coordinate axes and representing vectors as rows impacts the readability and ease of working with matrices. It allows for straightforward reading of matrix multiplications from left to right, enabling intuitive cancellations like $r_D = r_A \cdot M_{A.B} \cdot A.D$.

```java
// Example of vector transformation following chosen conventions
Vector3D pointA = new Vector3D(1, 2, 3);
Matrix4x4f MA_B = getTransformationMatrix(); // Transformation matrix from space A to B
Vector4D pointB = MA_B.transform(rowVector(pointA)); // Transforming point in row vector form

// Ensuring correct cancellation and readability
Matrix4x4f MC_D = getAnotherTransformationMatrix();
Vector3D result = rowVector(pointB).multiply(MC_D); // Readable from left to right
```
x??

---

#### Transforming Normal Vectors

Background context: Normal vectors are special because they not only need to be of unit length but must also remain perpendicular to the surface or plane they represent. When transforming a normal vector, it should ideally preserve its length and orthogonality.

:p Why is it necessary to use the inverse transpose when transforming a normal vector?

??x
It is essential to use the inverse transpose ($(M_{A.B}^{-1})^T $) of the transformation matrix when transforming normal vectors to ensure that both their length and perpendicularity properties are maintained. Directly using $ M_{A.B}$ might alter these properties, making it necessary to use its inverse transpose.

```java
// Example of normal vector transformation
Vector3D normal = new Vector3D(1, 0, 0); // Normal vector in space A
Matrix4x4f MA_B = getTransformationMatrix(); // Transformation matrix from space A to B
Vector3D transformedNormal = (MA_B.transpose().inverse()).transform(normal);
```
x??

---

#### Angle Preservation and Transformation Matrices
Background context: The text discusses how transformation matrices (specifically, MA.B) preserve angles between surfaces and vectors when they only contain uniform scaling. However, non-uniform scaling or shear can distort these angles, necessitating the use of the inverse transpose matrix to maintain perpendicularity.
:p What happens to the angles between surfaces and vectors if the matrix contains non-uniform scale or shear?
??x
When the matrix MA.B includes non-uniform scaling or shear (non-orthogonal transformations), the angles between surfaces and vectors are not preserved when moving from space A to space B. A vector that was normal to a surface in space A will not necessarily be perpendicular to that same surface in space B.
x??

---

#### Storing Matrices in Memory - Approach 1
Background context: The text explains two methods for storing matrices in memory within C/C++: contiguous storage of vectors (each row as a single vector) and strided storage of vectors (each column as a single vector). Approach 1, which stores vectors contiguously, is described.
:p How does approach 1 store matrix elements?
??x
In approach 1, the matrix elements are stored such that each row contains all four components of one vector. This means that the four values for each vector (iC, jC, kC, tC) are stored contiguously in memory.
```c
float M[4][4];
M[0][0]=ix;  // ix component of first vector
M[0][1]=iy;
M[0][2]=iz;
M[0][3]=0.0f;  // w-component, set to 0 for vectors

// Similarly for other rows...
```
x??

---

#### Storing Matrices in Memory - Approach 2
Background context: The text provides an alternative approach (2) for storing matrices, where each column contains one vector. This method is sometimes necessary when performing fast matrix-vector multiplies using SIMD microprocessors.
:p How does approach 2 store matrix elements?
??x
In approach 2, the matrix elements are stored such that each column contains all four components of one vector. This means that vectors are stored in columns rather than rows.
```c
float M[4][4];
M[0][0]=ix;  // ix component of first vector (column)
M[1][0]=jx;
M[2][0]=kx;
M[3][0]=tx;

// Similarly for other columns...
```
x??

---

#### Matrix Storage in Game Engines
Background context: The text mentions that most game engines store matrices using approach 1, where vectors are stored contiguously within the rows of a two-dimensional C/C++ array. This method is chosen because it matches row vector matrix equations.
:p How do most game engines typically store matrices?
??x
Most game engines store matrices by storing vectors contiguously in memory with each row representing one vector. For example, a 4x4 matrix might look like this:
```c
float M[4][4];
M[0][0]=ix; // ix component of first vector (row)
M[0][1]=iy;
M[0][2]=iz;
M[0][3]=0.0f;

// Similarly for other rows...
```
This layout ensures that vectors are stored as contiguous blocks, which can be useful when accessing and manipulating individual vectors.
x??

---

#### Matrix Storage Layout in Debuggers
Background context: The text provides a specific example of how a 4x4 matrix is laid out in memory using approach 1. This layout is important for understanding how matrices are accessed and manipulated both in code and during debugging.
:p What does the debugger display show about the matrix storage?
??x
The debugger displays a two-dimensional array where each row corresponds to one vector, with components stored contiguously:
```
M[][] [0] [0] ix   // First row, first element (ix)
[1] iy
[2] iz
[3] 0.0000

// Similarly for other rows...
```
This layout helps in understanding and debugging matrix operations by showing the direct memory addresses of each vector component.
x??

---

