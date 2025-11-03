# Flashcards: Game-Engine-Architecture_processed (Part 44)

**Starting Chapter:** 5.3 Matrices

---

#### Pseudovectors and Hand-ness
Background context explaining that some vectors are pseudovectors, which transform differently when changing hand-ness. This is important for game programmers to handle transformations properly.
:p What is a pseudovector, and why is it significant for game programmers?
??x
A pseudovector is a type of vector quantity that changes sign under an improper rotation (such as a reflection). In the context of game programming, this means that when you change the hand-ness of your coordinate system (e.g., switch from left-handed to right-handed), certain vectors need to be transformed differently. For example, in a left-handed coordinate system, a vector might point one way, but in a right-handed system, it will point the opposite direction.
This distinction is important because improper rotations can affect how objects are oriented and transformed in 3D space.

If you're working with cross products or other operations that involve pseudovectors, such as calculating torque or angular velocity, you need to ensure proper handling of these vectors when switching coordinate systems.
x??

---

#### Linear Interpolation (LERP)
Background context explaining the need for linear interpolation between points in game programming for smooth animations. The operation is a simple mathematical function that finds an intermediate point between two known points.
:p What is linear interpolation (LERP), and how does it work?
??x
Linear interpolation, or LERP, is used to find a vector that lies at a certain percentage along the line segment connecting two vectors \( \mathbf{A} \) and \( \mathbf{B} \). The formula for linear interpolation between points \( \mathbf{A} \) and \( \mathbf{B} \), where \( b \) is a scalar between 0 and 1, is given by:

\[ L = \text{LERP}(\mathbf{A}, \mathbf{B}, b) = (1 - b)\mathbf{A} + b\mathbf{B} \]

In component form for 3D vectors:
\[ L_x = (1 - b)A_x + bB_x \]
\[ L_y = (1 - b)A_y + bB_y \]
\[ L_z = (1 - b)A_z + bB_z \]

Here, \( b \) represents the fraction of the distance from point \( A \) to point \( B \). For example, if \( b = 0.5 \), then \( L \) is exactly halfway between \( \mathbf{A} \) and \( \mathbf{B} \).
x??

---

#### Matrices Overview
Background context explaining that matrices are used in game programming to represent linear transformations like translation, rotation, and scaling.
:p What is a matrix, and why are they useful in game development?
??x
A matrix is a rectangular array of scalars arranged into rows and columns. In the context of 3D graphics, matrices are particularly useful because they can compactly represent various types of linear transformations such as translation, rotation, and scaling.

Matrices are typically written with entries enclosed in square brackets, where subscripts \( r \) and \( c \) denote the row and column indices, respectively. For example:
\[ M = \begin{bmatrix} 
M_{11} & M_{12} & M_{13} \\
M_{21} & M_{22} & M_{23} \\
M_{31} & M_{32} & M_{33}
\end{bmatrix} \]

For a 3x3 matrix, each row and column can be thought of as 3D vectors. When all rows and columns are unit length and orthogonal to each other (i.e., their dot product is zero), the matrix is called an orthonormal matrix. Such matrices represent pure rotations.

A 4x4 matrix can represent arbitrary 3D transformations, including translations, rotations, and scaling changes. These matrices are known as transformation matrices and are essential for game development.
x??

---

#### Special Orthogonal Matrices
Background context explaining the properties of special orthogonal matrices (also called orthonormal matrices) which represent pure rotations.
:p What is a special orthogonal matrix, and how does it differ from other matrices?
??x
A special orthogonal matrix, also known as an orthonormal matrix, is a square matrix where all rows and columns are unit vectors and are mutually orthogonal. This means that each row and column has a magnitude of 1, and any two different rows (or columns) are perpendicular to each other.

The defining property of such matrices is:
\[ M^T M = I \]
where \( M^T \) is the transpose of matrix \( M \), and \( I \) is the identity matrix. This ensures that the transformation represented by the matrix preserves lengths and angles, making it suitable for representing pure rotations without scaling or shearing.

In contrast, other matrices might represent more complex transformations like scaling or projection, where rows and columns are not necessarily unit vectors and may not be orthogonal.
x??

---

#### Transformation Matrices
Background context explaining that 4x4 transformation matrices can represent arbitrary 3D transformations including translation, rotation, and scaling.
:p What is a transformation matrix, and how does it differ from other matrices?
??x
A transformation matrix is a specific type of 4x4 matrix used in 3D graphics to represent various transformations such as translations, rotations, and scalings. These matrices are crucial for game development because they allow for the combination of multiple transformations into a single operation.

The general form of a 4x4 transformation matrix can be written as:
\[ \begin{bmatrix} 
R & T \\
0^T & 1
\end{bmatrix} \]
where \( R \) is a 3x3 rotation sub-matrix and \( T \) is a translation vector. The bottom row ensures that the matrix behaves correctly with homogeneous coordinates.

Transformation matrices differ from other types of matrices (like special orthogonal matrices or projection matrices) in that they can handle translations, which are not possible with just rotations or scalings alone.
x??

---

#### Affine Transformation Matrix
Affine transformation matrices are used to apply combinations of operations like rotation, translation, scaling, and shearing. These transformations preserve parallelism but not necessarily absolute lengths and angles.

:p What is an affine matrix?
??x
An affine matrix is a 4×4 transformation matrix that combines operations such as rotation, translation, scaling, and shear into a single matrix. It preserves the parallelism of lines and relative distances between points but does not preserve absolute lengths or angles.
x??

---
#### Matrix Multiplication in Affine Transformations
Matrix multiplication for affine transformations allows combining multiple transformations (like scaling and rotating) into one operation.

:p How is matrix multiplication used in affine transformations?
??x
In affine transformations, the product of two matrices \(P = AB\) results in another transformation matrix that applies both transformations. For example, if \(A\) is a scale matrix and \(B\) is a rotation matrix, the resulting matrix \(P\) will both scale and rotate points or vectors.

For 3×3 matrices:
\[ P = \begin{pmatrix}
p_{11} & p_{12} & p_{13} \\
p_{21} & p_{22} & p_{23} \\
p_{31} & p_{32} & p_{33}
\end{pmatrix} = \begin{pmatrix}
A_{row1} \cdot B_{col1} & A_{row1} \cdot B_{col2} & A_{row1} \cdot B_{col3} \\
A_{row2} \cdot B_{col1} & A_{row2} \cdot B_{col2} & A_{row2} \cdot B_{col3} \\
A_{row3} \cdot B_{col1} & A_{row3} \cdot B_{col2} & A_{row3} \cdot B_{col3}
\end{pmatrix} \]

Matrix multiplication is not commutative, i.e., \(AB \neq BA\).

```java
public class Matrix {
    public double[][] multiply(double[][] a, double[][] b) {
        if (a[0].length != b.length) throw new IllegalArgumentException("Inner dimensions must match");
        
        double[][] result = new double[a.length][b[0].length];
        
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < b.length; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }
}
```
x??

---
#### Dot Product in Matrix Multiplication
To calculate the product of two matrices, you perform dot products between their rows and columns.

:p What is a dot product in matrix multiplication?
??x
A dot product involves multiplying corresponding elements from the row of one matrix with the column of another and summing those products. For example, to get \(P_{11}\) in the resulting 3×3 matrix:
\[ P_{11} = A_{row1} \cdot B_{col1} = a_{11}b_{11} + a_{12}b_{21} + a_{13}b_{31} \]

The overall product \(P\) is formed by repeating this process for all rows and columns.

```java
public class Matrix {
    public double dotProduct(double[] row, double[] col) {
        double result = 0;
        for (int i = 0; i < row.length; i++) {
            result += row[i] * col[i];
        }
        return result;
    }
}
```
x??

---
#### Row and Column Vectors in Affine Transformations
Vectors can be represented as either row vectors or column vectors, affecting the order of matrix multiplication.

:p How do we represent points and vectors in matrices for affine transformations?
??x
Points and vectors can be represented as 1×n row matrices (for row vectors) or n×1 column matrices (for column vectors). The choice between row and column vectors is arbitrary but affects the order of matrix multiplication:
- For a row vector \(v_1 = [3 \, 4 \, -1]\), multiplying by an n×n matrix \(M\):
  \[ v'_{1 \times n} = v_{1 \times n} M_{n \times n} \]
- For a column vector \(v_2 = \begin{pmatrix} 3 \\ 4 \\ -1 \end{pmatrix}\), multiplying by an n×n matrix \(M\):
  \[ v'_{n \times 1} = M_{n \times n} v_{n \times 1} \]

If multiple transformations are applied, the order is reversed for column vectors compared to row vectors. For example:
- Row vector: \(v' = (((vA)B)C)\)
- Column vector: \(v' = (C^T(B^T(A^Tv^T)))\)

The matrix closest to the vector is applied first.
x??

---
#### Concatenation of Transformation Matrices
Concatenating multiple transformation matrices results in a single matrix that applies all transformations in order.

:p What does concatenation mean in the context of matrix multiplication for transformations?
??x
Concatenation, or chaining together, involves multiplying multiple transformation matrices to form one matrix. The resulting matrix performs all the original transformations in the order they are multiplied.

For example:
\[ P = AB \]
If \(A\) is a rotation and \(B\) is a scale matrix, \(P\) will first rotate then scale.

```java
public class Transformation {
    public double[][] concatenate(double[][] A, double[][] B) {
        return multiply(A, B);
    }
}
```
x??

---

---
#### Vector-Matrix Multiplication Convention

Background context: Understanding whether vectors are row or column vectors is crucial for correct matrix operations. This can be determined by how vector-matrix multiplications are written.

:p How do you determine if a game engine uses column vectors?
??x
To determine the convention used in your game engine, check how vector-matrix multiplications are represented. If vectors are multiplied on the left of matrices (e.g., \( \mathbf{v}M \)), then the engine likely uses row vectors. Conversely, if vectors are multiplied on the right of matrices (e.g., \( M\mathbf{v} \)), the engine probably uses column vectors.
x??

---
#### Identity Matrix

Background context: The identity matrix is a fundamental concept in linear algebra and transformations, often denoted by \( I \). It has 1’s along the diagonal and 0’s elsewhere. Multiplying any matrix with the identity matrix yields the same original matrix.

:p What is an identity matrix?
??x
An identity matrix is a square matrix that, when multiplied by another matrix, results in the same matrix. It's represented by \( I \) and has 1’s along the diagonal and 0’s everywhere else. For example, for a 3x3 identity matrix:
```plaintext
I_3 = | 1 0 0 |
      | 0 1 0 |
      | 0 0 1 |
```
It holds that \( AI = IA = A \).
x??

---
#### Matrix Inversion

Background context: The inverse of a matrix, denoted as \( A^{-1} \), is another matrix that when multiplied by the original matrix, results in the identity matrix. Not all matrices have inverses; only specific types like affine transformations do.

:p What is the concept of matrix inversion?
??x
Matrix inversion refers to finding a matrix \( B = A^{-1} \) such that \( AB = BA = I \). For example, if a rotation matrix \( A \) rotates objects by 37 degrees about the z-axis, its inverse \( A^{-1} \) will rotate them back by -37 degrees. Similarly, scaling by 2 and then inversely scaling by 0.5 would return the original size.

If the matrix multiplication involves a sequence of matrices (e.g., \( ABC \)), the inverse is found as \( (ABC)^{-1} = C^{-1}B^{-1}A^{-1} \).

Code example for finding an inverse in Java using a simple pseudocode:
```java
public class MatrixInverse {
    public static double[][] inverse(double[][] matrix) {
        // Pseudocode to calculate the inverse of a matrix
        return inverseMatrix;
    }
}
```
x??

---
#### Transposition

Background context: The transpose of a matrix, denoted \( M^T \), is obtained by swapping its rows and columns. It's useful for operations like finding the inverse of an orthonormal (pure rotation) matrix.

:p What is transposition?
??x
Transposition involves reflecting the entries of the original matrix across its diagonal, such that rows become columns and vice versa. For example:
```plaintext
M = | a b c |
    | d e f |
    | g h i |

MT = | a d g |
      | b e h |
      | c f i |
```
Transposition is particularly useful because the inverse of an orthonormal matrix (pure rotation) is equal to its transpose. Additionally, it's easier to transpose a matrix than to find its inverse generally.

Code example for transposing a 3x3 matrix in Java:
```java
public class MatrixTranspose {
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposedMatrix = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposedMatrix[j][i] = matrix[i][j];
            }
        }
        return transposedMatrix;
    }
}
```
x??

---
#### Homogeneous Coordinates

Background context: In computer graphics, homogeneous coordinates extend the concept of a 2D vector into 3D by adding an extra coordinate. This allows for representing both points and directions (vectors) in a unified manner.

:p What are homogeneous coordinates?
??x
Homogeneous coordinates are a way to represent points and vectors in \( n \)-dimensional space using \( n+1 \) dimensions. A 2x2 matrix can represent a rotation in two dimensions by incorporating an extra coordinate. For example, rotating a vector \( r = (r_x, r_y) \) through an angle of \( \phi \) degrees (counterclockwise) is represented as:
```plaintext
[r' x' y'] = [r_x r_y] * | cos(ϕ) sin(ϕ) |
                         | -sin(ϕ) cos(ϕ) |
```
This unified representation simplifies many operations in computer graphics, including handling both points and directions.

Code example for applying a 2D rotation matrix using homogeneous coordinates:
```java
public class RotationMatrix {
    public static double[][] rotate(double angleDegrees) {
        // Convert degrees to radians
        double radian = Math.toRadians(angleDegrees);
        
        // Create the rotation matrix
        double[][] matrix = {
            {Math.cos(radian), -Math.sin(radian)},
            {Math.sin(radian),  Math.cos(radian)}
        };
        return matrix;
    }
}
```
x??

#### 3D Rotations and Translation Matrices
Background context: In 3D game development, rotations can be represented using a 3x3 matrix. However, translations cannot be directly represented using just a 3x3 matrix due to the nature of matrix multiplication.

Matrix for 2D rotation around z-axis:
```plaintext
[r' x r' y r' z] = [r_x r_y r_z]^T * 
                    |cos(ϕ) sin(ϕ) 0    |
                    |sin(ϕ) cos(ϕ) 0    |
                    |0      0       1   |
```
:p Can a 3x3 matrix be used to represent translations?
??x
No, a 3x3 matrix cannot be used to represent translations because the result of translating a point \( \mathbf{r} \) by a translation vector \( \mathbf{t} \) requires adding the components of \( \mathbf{t} \) to those of \( \mathbf{r} \) individually: 
\[ \mathbf{r'} = (\mathbf{r} + \mathbf{t}) \]

Matrix multiplication involves multiplications and additions of matrix elements, so it is not possible to arrange the components of \( \mathbf{t} \) within a 3x3 matrix such that multiplying it with column vector \( \mathbf{r} \) yields sums like \( (r_x + t_x) \).

:x??
---

#### Using 4x4 Matrices for Translation
Background context: To incorporate translations, a 4x4 matrix is used in homogeneous coordinates. This approach ensures that both rotations and translations can be combined into one matrix operation.

Example of a 4x4 translation matrix:
```plaintext
[r x r y r z 1]^T * 
                |1   0    0    0    |
                |0   1    0    0    |
                |0   0    1    0    |
                |t_x t_y t_z 1     | = [r' x r' y r' z 1]^T
```
:p How is a translation represented in a 4x4 matrix?
??x
In a 4x4 matrix, the translation vector \( \mathbf{t} \) is placed in the bottom row. The fourth element of the point vector (usually called \( w \)) is set to 1:
\[ \begin{bmatrix}
r_x \\
r_y \\
r_z \\
1
\end{bmatrix} \times 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
t_x & t_y & t_z & 1
\end{bmatrix} = 
\begin{bmatrix}
r'_x \\
r'_y \\
r'_z \\
1
\end{bmatrix} \]

This ensures that the resulting vector has a \( w \) component of 1, which is necessary for further transformations.

:x??
---

#### Homogeneous Coordinates and Points vs. Directions
Background context: In 3D game development, points (position vectors) and direction vectors are treated differently when transforming them using matrices. Points involve both rotation and scaling, while directions only involve rotation.

:p How do you represent a point and a direction in homogeneous coordinates?
??x
In homogeneous coordinates:
- Points have their \( w \) component equal to 1.
- Directions have their \( w \) component equal to 0.

For example, the vector \( \mathbf{v} \) with \( w = 0 \):
\[ 
\begin{bmatrix}
v_x \\
v_y \\
v_z \\
0
\end{bmatrix}
\]

When this vector is multiplied by a transformation matrix that includes translation:
```plaintext
[v_x v_y v_z 0]^T * 
                |1   0    0    0    |
                |0   1    0    0    |
                |0   0    1    0    |
                |t_x t_y t_z 1     | = [v'x v'y v'z 0]^T
```

The \( w \) component of the resulting vector remains 0, effectively ignoring any translation.

:x??
---

#### Converting Homogeneous to Non-Homogeneous Coordinates
Background context: To convert a point in homogeneous coordinates back to non-homogeneous (three-dimensional) coordinates, divide each of the x, y, and z components by the w component.

:p How can you convert a point from homogeneous coordinates to non-homogeneous coordinates?
??x
To convert a point from homogeneous coordinates \([x\ y\ z\ w]\) back to three-dimensional coordinates, divide each of the \( x, y, \) and \( z \) components by the \( w \) component:
\[ 
\begin{bmatrix}
x \\
y \\
z \\
w
\end{bmatrix} \rightarrow 
\begin{bmatrix}
\frac{x}{w} \\
\frac{y}{w} \\
\frac{z}{w}
\end{bmatrix}
\]

This process ensures that the point is represented correctly in three-dimensional space.

:x??
---

#### Homogeneous Coordinates and w-component

Background context: In 4D homogeneous space, a point's \(w\)-component is set to 1, while a vector's \(w\)-component is set to 0. Dividing by \(w=1\) does not affect the coordinates of a point, but dividing by \(w=0\) would yield infinity.

:p What is the significance of setting the \(w\)-component of a point and a vector differently in homogeneous space?
??x
Setting the \(w\)-component to 1 for points ensures that division operations do not alter their coordinates. For vectors, setting \(w=0\) allows them to represent directions without being affected by translations. This distinction is crucial because it enables mathematical transformations such as rotations and translations to be performed on homogeneous coordinates.

```java
// Example of a point in 4D homogeneous space
Point4D p = new Point4D(1, 2, 3, 1);
// Example of a vector in 4D homogeneous space
Vector4D v = new Vector4D(4, 5, 6, 0);
```
x??

---

#### Transformation Matrices

Background context: Any affine transformation matrix can be created by concatenating matrices representing pure translations, rotations, and scale operations. These transformations are essential for manipulating geometric objects in 3D space.

:p How is an affine 4×4 transformation matrix partitioned?
??x
An affine 4×4 transformation matrix is partitioned into four components:

- The upper 3×3 matrix \(U\) represents rotation and/or scaling.
- A 1×3 translation vector \(t\).
- A 3×1 zero vector \([0, 0, 0]^T\).
- A scalar 1 in the bottom-right corner.

The partitioned form is:

\[ M_{affine} = [U_{3 \times 3} \; 0_{3 \times 1}; t^T_1 \; 1] \]

Where:
- \(U_{3 \times 3}\) contains rotation and scaling information.
- \(t_1\) is the translation vector.
- The zero vector ensures that only points are translated, not vectors.

```java
// Example of partitioning an affine transformation matrix
Matrix4x4 mat = new Matrix4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    3.0f, 4.0f, 5.0f, 1.0f
);
```
x??

---

#### Translation in Homogeneous Space

Background context: A translation matrix moves a point by adding the translation vector \(t\) to the original point \(r\). In homogeneous space, this is achieved by appending 1 at the bottom-right corner.

:p What is the formula for translating a point in homogeneous coordinates?
??x
The formula for translating a point \([r_1]\) by a vector \(t\) is:

\[ [r'_{1 \times 3} 1] = [r_{1 \times 3} 1][ I_{3 \times 3} 0_{3 \times 1}; t^T_{1 \times 3} 1] \]

Which simplifies to:

\[ [r'_{1 \times 3} 1] = [(r + t) 1] \]

In code, this can be represented as:

```java
// Translating a point in homogeneous space
Vector3D r = new Vector3D(1.0f, 2.0f, 3.0f);
Vector3D t = new Vector3D(4.0f, 5.0f, 6.0f);

Matrix4x4 translationMat = new Matrix4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    t.x(), t.y(), t.z(), 1.0f
);

Vector4D rTranslated = translationMat * new Vector4D(r.getX(), r.getY(), r.getZ(), 1);
```
x??

---

#### Rotation in Homogeneous Space

Background context: Pure rotation matrices are used to rotate points about the coordinate axes without scaling or translating them. These matrices have a specific form and include sine and cosine values.

:p What is the matrix representation for rotating a point around the x-axis by an angle \(\phi\)?
??x
The matrix representing a rotation of a point around the x-axis by an angle \(\phi\) is:

\[ \text{rotate}_x(r, \phi) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\phi & -\sin\phi & 0 \\
0 & \sin\phi & \cos\phi & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} \]

In code, this can be represented as:

```java
// Rotating a point around the x-axis by angle phi
Matrix4x4 rotateX = new Matrix4x4(
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, (float)Math.cos(phi), -(float)Math.sin(phi), 0.0f,
    0.0f, (float)Math.sin(phi), (float)Math.cos(phi), 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
);

Vector4D rRotated = rotateX * new Vector4D(r.getX(), r.getY(), r.getZ(), 1);
```
x??

---

#### Scaling Matrices
Background context explaining scaling matrices. The provided formula shows how a point \( \mathbf{r} = [x, y, z, 1]^T \) is scaled by factors \( s_x \), \( s_y \), and \( s_z \) along the x-, y-, and z-axes respectively.

:p What are scaling matrices used for?
??x
Scaling matrices are utilized to scale a point in 3D space along the three axes independently. The transformation matrix allows us to resize objects without changing their shapes, which is crucial in many graphics applications.
??? 

Code example with explanation:
```java
public class ScalingMatrix {
    public double sx; // scaling factor for x-axis
    public double sy; // scaling factor for y-axis
    public double sz; // scaling factor for z-axis

    public ScalingMatrix(double sx, double sy, double sz) {
        this.sx = sx;
        this.sy = sy;
        this.sz = sz;
    }

    public static void scalePoint(double[] point, double sx, double sy, double sz) {
        point[0] *= sx; // Scale x-coordinate
        point[1] *= sy; // Scale y-coordinate
        point[2] *= sz; // Scale z-coordinate
    }
}
```
????

---

#### Inverting Scaling Matrices
Background context explaining how to invert a scaling matrix. The inverse of the scaling matrix involves substituting \( s_x \), \( s_y \), and \( s_z \) with their reciprocals, i.e., \( 1/s_x \), \( 1/s_y \), and \( 1/s_z \).

:p How do you invert a scaling matrix?
??x
To invert a scaling matrix, replace the scaling factors \( s_x \), \( s_y \), and \( s_z \) with their reciprocals. This effectively reverses the scaling effect.
????

---

#### Uniform vs Non-Uniform Scaling
Background context explaining uniform and non-uniform scaling. In uniform scaling, all axes are scaled by the same factor. In contrast, non-uniform scaling changes the size along different axes independently.

:p What is the difference between uniform and non-uniform scaling?
??x
In uniform scaling, a scale matrix scales an object equally in all three dimensions (x, y, z). Non-uniform scaling, on the other hand, allows for different scaling factors along each axis. Under uniform scaling, spheres remain spherical, but under non-uniform scaling, they become ellipsoidal.
????

---

#### Affine 4×3 Matrices
Background context explaining affine 4×3 matrices and their usage in game programming. These matrices are often used to represent transformations without the fourth column, saving memory.

:p What is an affine 4×3 matrix?
??x
An affine 4×3 matrix represents a transformation using only three columns for a point in 3D space. The rightmost column always contains \([0, 0, 0, 1]^T\). This form is frequently used in game math libraries to save memory.
????

---

#### Coordinate Spaces
Background context explaining coordinate spaces and their usage in computer graphics. Coordinate systems represent points relative to a set of axes.

:p What are coordinate spaces?
??x
Coordinate spaces, or frames, represent sets of axes that define the position and orientation of objects in 3D space. They help in transforming points and vectors by applying matrices.
????

---

#### Model Space
Background context explaining model space (or object/local space) used in tools like Maya and 3D Studio MAX.

:p What is model space?
??x
Model space, also known as object space or local space, is the coordinate system where a triangle mesh's vertices are defined relative to a Cartesian coordinate system. This is typically how objects are created and manipulated within modeling software.
????

---

#### Position Vectors in Different Coordinate Axes
Background context explaining position vectors and their representation in different coordinate systems.

:p How do position vectors change when represented in different coordinate axes?
??x
Position vectors change numerically depending on the coordinate system used. A point can be represented differently based on which set of axes is chosen, reflecting its position relative to those axes.
????

---

#### Transformations for Rigid Objects
Background context explaining how transformations are applied to rigid objects using vertex transformation.

:p How are rigid objects transformed in computer graphics?
??x
Rigid objects are typically represented as collections of points. Applying a transformation matrix to all vertices of the object results in the same transformation being applied uniformly across the entire object.
????

---

#### Order of Matrix Multiplication for Uniform Scale and Rotation
Background context explaining that the order of multiplication does not matter when concatenating uniform scale and rotation matrices.

:p What is the significance of the order of multiplication for uniform scale and rotation matrices?
??x
For uniform scaling, the order of matrix multiplication with a rotation matrix does not affect the result. This property holds true only for uniform scaling.
????

---

#### Summary of Coordinate Spaces in Games
Background context summarizing common coordinate spaces used in games.

:p What are some common coordinate spaces used in games and computer graphics?
??x
Common coordinate spaces include model space, world space, view space, and screen space. Each represents a different perspective on the same 3D scene.
????

---

#### Model Space Origin and Directions

Background context explaining that model space is a local coordinate system centered on an object. It's often aligned with natural directions like front, up, and left/right axes to facilitate intuitive rotations and transformations.

:p What are the common names for the axes in model space?
??x
In model space, the axes are commonly named as follows:
- Front: This axis points in the direction that the object naturally travels or faces.
- Up: This axis points towards the top of the object.
- Left/Right: These axes point to the left or right side of the object, depending on whether your game engine uses a left-handed or right-handed coordinate system.

For right-handed coordinates, you might assign:
```java
F = k; // Front is +z-axis
L = i; // Left is +x-axis
U = j; // Up is +y-axis
```
Or for left-handed coordinates:
```java
F = i; // Front is +x-axis
R = k; // Right is +z-axis
U = j; // Up is +y-axis
```

x??

---

#### World Space

Background context explaining that world space is a fixed coordinate system in which all object positions, orientations, and scales are expressed. It allows for the entire game world to be coherently managed.

:p What is the significance of world space?
??x
World space serves as a global reference frame within the game engine where all objects' transformations (positions, orientations, scales) are described relative to this fixed coordinate system. This helps in managing the game world uniformly and ensures that all objects can be positioned and oriented consistently.

For example, if you want to move an object from one location to another in the game world, you would do so using operations in world space rather than local model space.

x??

---

#### Euler Angles

Background context explaining that Euler angles (pitch, yaw, roll) are used to describe an aircraft's orientation. They provide a way to define rotations around specific axes relative to a set of basis vectors.

:p How are pitch, yaw, and roll defined in terms of the basis vectors?
??x
In the context of Euler angles:
- Pitch is rotation about the left or right axis (L/R), which corresponds to rotating around the x-axis.
- Yaw is rotation about the up axis (U), corresponding to a rotation around the y-axis.
- Roll is rotation about the front axis (F), equivalent to a rotation around the z-axis.

If you're using right-handed coordinates, these would map as:
```java
pitch = rotation around L; // +x-axis
yaw   = rotation around U; // +y-axis
roll  = rotation around F; // +z-axis
```
For left-handed systems, they might be:
```java
pitch = rotation around R; // -x-axis
yaw   = rotation around U; // +y-axis
roll  = rotation around F; // +z-axis
```

x??

---

---
#### Aircraft Rotation and Translation in World Space
Background context: The passage explains how an aircraft's position and orientation can be transformed from model space to world space. It covers rotation around the y-axis and translation of the origin.

:p What is the transformation process for a Lear jet with its left wingtip at (5, 0, 0) in model space?
??x
The transformation involves rotating the aircraft by 90 degrees about the world-space y-axis and then translating the model-space origin to (–25, 50, 8). After these transformations, the left wingtip's position in world space is at (–25, 50, 3).

Explanation:
1. **Rotation**: Rotating by 90 degrees about the y-axis changes the aircraft’s orientation.
2. **Translation**: The origin of the model space is moved to (–25, 50, 8), which affects all points in the model.

C/Java code example for a simplified transformation process:
```java
public class AircraftTransform {
    public static void transformAircraft(double[] position) {
        // Assume rotation by 90 degrees about y-axis and translation to (25, 50, 8)
        double[] rotatedAndTranslatedPosition = new double[3];
        
        // Rotate around the y-axis
        rotatedAndTranslatedPosition[0] = -position[2]; // x' = -z
        rotatedAndTranslatedPosition[1] = position[1];   // y' = y (remains unchanged)
        rotatedAndTranslatedPosition[2] = position[0];   // z' = x
        
        // Translate the origin to (-25, 50, 8)
        rotatedAndTranslatedPosition[0] += -25; 
        rotatedAndTranslatedPosition[1] += 50;
        rotatedAndTranslatedPosition[2] += 8;
        
        System.out.println("Transformed position: " + rotatedAndTranslatedPosition[0] + ", "
                           + rotatedAndTranslatedPosition[1] + ", " + rotatedAndTranslatedPosition[2]);
    }
}
```
x??

---
#### View Space (Camera Space)
Background context: The passage explains the concept of view space or cameraspace, which is fixed to the camera. It mentions that the origin of view space is at the focal point of the camera and discusses axis orientation schemes like y-up and z-forward.

:p What is the significance of the y-up convention in view space?
??x
The y-up convention in view space allows z-coordinates to represent depths into the screen, making it easier to interpret depth information. This is typical for most 3D engines and APIs.

Explanation:
- In a y-up coordinate system, the positive y-axis points upwards.
- The camera faces towards negative z, meaning that moving forward in world space corresponds to increasing z-values in view space.

C/Java code example (pseudocode) for setting up a view matrix with y-up convention:
```java
public class ViewMatrixSetup {
    public static void setupViewMatrix(double eyepointX, double eyepointY, double eyepointZ,
                                       double lookatX, double lookatY, double lookatZ,
                                       double upVectorX, double upVectorY, double upVectorZ) {
        // Calculate the view direction vector
        double[] viewDirection = new double[3];
        viewDirection[0] = lookatX - eyepointX;
        viewDirection[1] = lookatY - eyepointY;
        viewDirection[2] = lookatZ - eyepointZ;
        
        // Normalize the direction vector
        double length = Math.sqrt(viewDirection[0]*viewDirection[0] + 
                                  viewDirection[1]*viewDirection[1] + 
                                  viewDirection[2]*viewDirection[2]);
        for (int i = 0; i < 3; i++) {
            viewDirection[i] /= length;
        }
        
        // Calculate the right and up vectors
        double[] rightVector = new double[3];
        rightVector[1] = -upVectorX * viewDirection[2] + upVectorZ * viewDirection[0];
        rightVector[2] = -upVectorX * viewDirection[0] - upVectorZ * viewDirection[1];
        
        // Normalize the right vector
        length = Math.sqrt(rightVector[1]*rightVector[1] + 
                           rightVector[2]*rightVector[2]);
        for (int i = 1; i <= 2; i++) {
            rightVector[i] /= length;
        }
        
        double[] upVector = new double[3];
        upVector[0] = viewDirection[1] * rightVector[2] - viewDirection[2] * rightVector[1];
        upVector[1] = -viewDirection[0] * rightVector[2] + viewDirection[2] * rightVector[0];
        upVector[2] = -viewDirection[0] * rightVector[1] + viewDirection[1] * rightVector[0];
        
        // Normalize the up vector
        length = Math.sqrt(upVector[0]*upVector[0] + 
                           upVector[1]*upVector[1] + 
                           upVector[2]*upVector[2]);
        for (int i = 0; i < 3; i++) {
            upVector[i] /= length;
        }
        
        // Construct the view matrix
        double[] viewMatrix = new double[4][4];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 3; col++) {
                if (row == 2) {
                    viewMatrix[row][col] = -viewDirection[col];
                } else if (row == 3) {
                    viewMatrix[row][col] = 1.0;
                } else {
                    viewMatrix[row][col] = rightVector[(col + 1) % 3];
                }
            }
        }
    }
}
```
x??

---
#### Change of Basis in Coordinate Systems
Background context: The passage explains the concept of change of basis, which involves converting an object's position, orientation, and scale from one coordinate system to another.

:p How does a change of basis work in games and computer graphics?
??x
In games and computer graphics, changing the basis means converting an object’s position, orientation, and scale from one coordinate system (like model space) into another (such as world space). This is useful for defining relative positions and orientations within a hierarchy of coordinate systems.

Explanation:
- Coordinate frames are hierarchical; each frame has a parent.
- World space has no parent and serves as the root.
- Objects can be transformed using rotation, translation, and scaling matrices to move them from one basis (coordinate system) to another.

C/Java code example for applying a change of basis transformation:
```java
public class ChangeOfBasis {
    public static void transformObject(double[] originalPosition, double[][] transformationMatrix,
                                       double[] newCoordinates) {
        // Assuming the transformation matrix is already defined and valid
        
        // Apply the transformation matrix to the object's position
        for (int i = 0; i < 3; i++) {
            newCoordinates[i] = 0;
            for (int j = 0; j < 3; j++) {
                newCoordinates[i] += originalPosition[j] * transformationMatrix[i][j];
            }
        }
    }
}
```
x??

---

#### Change of Basis Matrix Definition
Background context: A change of basis matrix is used to transform points and directions from a child coordinate system (C) to its parent coordinate system (P). This transformation is denoted as \( M_{C.P} \).

Relevant formulas:
\[ P_P = P_C \cdot M_{C.P} \]
Where:
- \( P_P \) is the position vector in the parent space.
- \( P_C \) is the position vector in the child space.

Matrix form of \( M_{C.P} \):
\[ M_{C.P} = \begin{bmatrix}
i_{Cx} & i_{Cy} & i_{Cz} & 0 \\
j_{Cx} & j_{Cy} & j_{Cz} & 0 \\
k_{Cx} & k_{Cy} & k_{Cz} & 0 \\
t_{Cx} & t_{Cy} & t_{Cz} & 1
\end{bmatrix} \]

Explanation: The matrix includes the unit basis vectors (i, j, k) in parent space and the translation vector \( t_C \).

:p What is the transformation equation for changing from child to parent space?
??x
The position vector in the parent space can be found by multiplying the position vector in the child space with the change of basis matrix:
\[ P_P = P_C \cdot M_{C.P} \]
x??

---

#### Unit Basis Vectors and Rotation
Background context: The unit basis vectors (i, j, k) form the upper 3x3 submatrix of \( M_{C.P} \), which is a rotation matrix. If child space rotates by an angle g about the z-axis with no translation, specific vector forms can be derived.

Relevant formulas:
\[ \text{Rotation matrix: } \text{rotate}_z(r,g) = 
\begin{bmatrix}
\cos(g) & -\sin(g) & 0 & 0 \\
\sin(g) & \cos(g) & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix} 
\]

Explanation: In a simple example, if the z-axis is rotated by an angle \( g \), then:
- The iC vector in parent space is \( [cos(g) \; sin(g) \; 0] \).
- The jC vector in parent space is \( [-sin(g) \; cos(g) \; 0] \).

:p What are the unit basis vectors of a child coordinate system that rotates by an angle g about the z-axis?
??x
In this case, the iC and jC vectors can be derived as:
- \( i_C = [cos(g), sin(g), 0] \)
- \( j_C = [-sin(g), cos(g), 0] \)

These vectors form part of the upper 3x3 submatrix of the change of basis matrix.
x??

---

#### Scaling Child Axes
Background context: Scaling the child coordinate system involves scaling the unit basis vectors. If the axes are scaled by a factor, the basis vectors become non-unit length.

Relevant formulas:
\[ \text{If } i_C, j_C, k_C \text{ are of unit length and scaled up by a factor of 2, then:} \]
\[ i_C = [2i_{Cx}, 2i_{Cy}, 2i_{Cz}] \]
\[ j_C = [2j_{Cx}, 2j_{Cy}, 2j_{Cz}] \]
\[ k_C = [2k_{Cx}, 2k_{Cy}, 2k_{Cz}] \]

Explanation: The scaling factor is applied to each component of the basis vectors.

:p How does scaling affect the unit basis vectors in a child coordinate system?
??x
Scaling affects the unit basis vectors by multiplying each component with the scaling factor. For example, if a scale factor of 2 is used:
\[ i_C = [2i_{Cx}, 2i_{Cy}, 2i_{Cz}] \]
\[ j_C = [2j_{Cx}, 2j_{Cy}, 2j_{Cz}] \]
\[ k_C = [2k_{Cx}, 2k_{Cy}, 2k_{Cz}] \]

This scales the vectors by the factor, making them non-unit length.
x??

---

#### Extracting Basis Vectors from a Matrix
Background context: Given any affine 4x4 transformation matrix, the child-space basis vectors \( i_C, j_C, k_C \) can be extracted. This is useful in games where coordinate systems are transformed.

Relevant formulas:
\[ \text{For an affine transformation matrix } M_{C.P}: \]
- The vector \( k_C \) (representing the z-axis direction in child space) can be found by extracting the third row of \( M_{C.P} \).

Explanation: If we know that the positive z-axis always points in the direction that an object is facing, then the extracted vector will already be normalized and ready to use.

:p How can you extract the unit basis vectors from a given affine 4x4 transformation matrix?
??x
To extract the unit basis vectors from a given affine 4x4 transformation matrix \( M_{C.P} \):
- Extract the third row of \( M_{C.P} \) for \( k_C \).
- The resulting vector will be normalized and can be used as the z-axis direction in child space.

For example, if the model-to-world transform matrix is given:
\[ M_{C.P} = 
\begin{bmatrix}
m_{11} & m_{12} & m_{13} & t_x \\
m_{21} & m_{22} & m_{23} & t_y \\
m_{31} & m_{32} & m_{33} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix}
\]
The vector \( k_C \) can be extracted as:
\[ k_C = [m_{31}, m_{32}, m_{33}] \]

This vector is normalized and ready to represent the z-axis direction in child space.
x??

---

#### Transforming Coordinate Systems versus Vectors
Background context: The matrix \(MC.P\) transforms points and directions from child space into parent space. Its fourth row contains the translation of the child coordinate axes relative to world-space axes. This can be visualized as transforming the parent coordinate axes into the child axes.

:p How does a transformation matrix like \(MC.P\) affect both points and coordinate axes?
??x
A transformation matrix like \(MC.P\), which transforms vectors from child space to parent space, also reverses this action when dealing with coordinate axes. For instance, moving a point 20 units to the right with fixed axes is equivalent to moving the axes 20 units to the left while keeping the point fixed. This concept is illustrated in Figure 5.22.

```java
public class CoordinateTransform {
    public void transformPoint(float[] parentAxes, float[] childPoint) {
        // Logic to apply transformation from child space to parent space on a point
    }
    
    public void transformAxes(float[] childAxes, float[] parentAxes) {
        // Reverse logic to apply transformation from parent space to child space on axes
    }
}
```
x??

---

#### Change of Basis
Background context: Figure 5.21 illustrates how the basis changes when the child coordinate axes are rotated by an angle \(g\) relative to the parent coordinate system.

:p How does a change in basis affect point and vector transformations?
??x
A change in basis affects both points and vectors, but the direction of transformation depends on whether you view it from the perspective of moving the object or moving the coordinate system. For example, if you move a point 20 units to the right with fixed axes, this is equivalent to moving the axes 20 units left while keeping the point fixed.

```java
public class BasisTransform {
    public void transformPoint(float[] parentAxes, float[] childPoint) {
        // Logic to apply transformation from child space to parent space on a point
    }
    
    public void transformBasis(float[] parentAxes, float[] childBasis) {
        // Reverse logic to apply transformation from parent space to child space on basis vectors
    }
}
```
x??

---

#### Transforming Normal Vectors
Background context: A normal vector is special because it must remain perpendicular to the associated surface or plane. When transforming a normal vector, care must be taken to maintain both its length and perpendicularity properties.

:p How are normal vectors transformed differently compared to regular vectors?
??x
Normal vectors require a transformation using the inverse transpose of the matrix used for point or non-normal vector transformations. If a 3×3 matrix \(MA.B\) rotates a point from space A to space B, then a normal vector \(n\) should be transformed from space A to space B via \((M^{-1}_{A.B})^T\).

```java
public class NormalVectorTransform {
    public void transformNormal(float[] normal, float[] transformationMatrix) {
        // Logic to apply inverse transpose of the matrix on a normal vector
        float[] invertedTranspose = invertAndTranspose(transformationMatrix);
        multiply(normal, invertedTranspose);
    }
    
    private float[] invertAndTranspose(float[] matrix) {
        // Invert and transpose logic for the given matrix
    }
    
    private void multiply(float[] vector, float[] matrix) {
        // Matrix multiplication to apply transformation on a normal vector
    }
}
```
x??

---

#### Inverse Transpose for Normal Vectors
Background context: When a matrix \( \mathbf{M}_{A.B} \) represents transformations that include non-uniform scaling or shear (i.e., it is not orthogonal), angles between surfaces and vectors are no longer preserved. The inverse transpose of the matrix, denoted as \( (\mathbf{M}_{A.B})^T^{-1} \), corrects this distortion to maintain perpendicularity of normal vectors with their corresponding surfaces.

Explanation: For uniform scaling and no shear, the angles remain consistent between spaces A and B. However, for non-uniform scaling or shear transformations, a vector that was originally perpendicular (or normal) in space A may not be so after transformation into space B. The inverse transpose operation ensures that these normal vectors are restored to their correct orientation.

:p Why is the inverse transpose necessary?
??x
The inverse transpose is necessary because it ensures that normal vectors remain perpendicular to surfaces even when subjected to transformations that include non-uniform scaling or shear, which otherwise distort angles and orientations.
x??

---

#### Storing Matrices in C/C++ Memory Layout (Row-Vectors)
Background context: In the C and C++ languages, two-dimensional arrays are commonly used to store matrices. Each vector can be stored either contiguously in memory (rows of vectors) or strided in memory (columns of vectors).

Explanation: The row-vector storage approach stores each column of a matrix as a sequence of elements, making it easy to access individual columns by indexing the appropriate rows. This is particularly useful for matching with row vector matrix equations.

:p How are matrices stored using the row-vector approach?
??x
Matrices are stored such that vectors are contiguous in memory (stored row-wise). For example, in a 4x4 matrix \( \mathbf{M} \), each row contains one complete vector. This layout is convenient for accessing individual vectors and aligns with row vector matrix equations.
```cpp
float M[4][4]; // A 4x4 matrix stored as rows of vectors
// Example storage:
M[0][0]=ix;   // ix, iy, iz form the first vector (row)
M[1][0]=jx;
M[2][0]=kx;
M[3][0]=tx;

// And so on for other columns.
```
x??

---

#### Matrix Storage Layout in Game Engines
Background context: In game engines, matrices are typically stored using a specific layout to facilitate efficient vector-matrix multiplication operations. This is often referred to as the row-vector storage approach.

Explanation: Row vectors are stored contiguously within rows of a two-dimensional C/C++ array, making it easy to interpret each row as a single vector. This setup is beneficial for matching with row vector matrix equations and aligning well with how most game engines perform operations on matrices.

:p How does the row-vector storage layout work in game engines?
??x
In game engines, matrices are stored such that vectors are contiguous within rows of a two-dimensional C/C++ array. Each element in a row corresponds to one component of a vector. For example:
```cpp
float M[4][4]; // A 4x4 matrix stored as rows of vectors
M[0][0]=ix;    // ix, iy, iz form the first vector (row)
M[0][1]=iy;
M[0][2]=iz;
M[0][3]=0.0f;

// Similarly for other vectors:
M[1][0]=jx;
M[1][1]=jy;
M[1][2]=jz;
M[1][3]=0.0f;

// And so on.
```
This layout ensures that each row is a complete vector, simplifying access and matching with row vector matrix equations.
x??

---

#### Two-dimensional Array Contiguity in Memory
Background context: Understanding how elements of a two-dimensional array are laid out in memory helps optimize data access and manipulation. In C/C++, the first subscript (row) varies slower than the second (column), making columns vary faster as you move through memory.

Explanation: This layout means that accessing consecutive elements within a row is more efficient, but jumping to different rows requires additional addressing overhead.

:p How does the memory layout of a two-dimensional array in C/C++ work?
??x
In C/C++, two-dimensional arrays are laid out such that each element is stored contiguously in memory. The first index (row) varies slower than the second index (column). This means accessing consecutive elements within a row is efficient, but jumping to different rows requires additional addressing.

For example:
```cpp
float m[4][4]; // 2D array with 4 rows and 4 columns
// Memory layout will be like this:

// Row 0: [m[0][0], m[0][1], m[0][2], m[0][3]]
// Row 1: [m[1][0], m[1][1], m[1][2], m[1][3]]
// Row 2: [m[2][0], m[2][1], m[2][2], m[2][3]]
// Row 3: [m[3][0], m[3][1], m[3][2], m[3][3]]

float* pm = &m[0][0]; // Starting address of the array
ASSERT(&pm[4] == &m[1][0]); // Jumping to next row (4 elements)
ASSERT(&pm[5] == &m[1][1]); // Moving to next column within same row
```
x??

---

#### Strided Memory Layout for Matrix Storage
Background context: When working with matrices, especially in vector-enabled SIMD microprocessors, there is a need to optimize memory access patterns. One approach is to store matrix elements such that columns are contiguous.

Explanation: This striding storage layout allows for efficient SIMD operations where multiple elements from the same column can be processed together, leveraging parallel processing capabilities of modern CPUs.

:p Why would one use a strided memory layout?
??x
A strided memory layout is used when performing fast matrix-vector multiplications on vector-enabled (SIMD) microprocessors. This layout stores each column of the matrix as a contiguous sequence in memory, allowing for efficient SIMD operations that can process multiple elements from the same column simultaneously.

For example:
```cpp
float m[4][4]; // 2D array with 4 rows and 4 columns
// Strided storage (each column is a separate block):
m[0][0] = ix;   // First element of the first column
m[1][0] = jx;
m[2][0] = kx;
m[3][0] = tx;

m[0][1] = iy;   // Second element of the first column, etc.
m[1][1] = jy;
m[2][1] = ky;
m[3][1] = ty;

// And so on for other columns.
```
This layout allows for efficient SIMD operations by ensuring that elements from the same column are stored contiguously in memory, enabling parallel processing of these elements.
x??

