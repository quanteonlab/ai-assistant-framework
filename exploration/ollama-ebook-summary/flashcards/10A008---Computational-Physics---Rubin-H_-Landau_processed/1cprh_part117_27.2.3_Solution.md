# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 117)

**Starting Chapter:** 27.2.3 Solution via Linear Equations

---

#### Finite Element Method Overview
The finite element method (FEM) is a numerical technique used to solve partial differential equations (PDEs). It is particularly useful for problems with irregular domains or highly varying conditions. FEM offers flexibility and can be applied to various types of PDEs.

:p What does the finite element method offer in terms of problem-solving?
??x
FEM provides a flexible approach to solving complex problems by breaking down the domain into smaller, manageable elements. This allows for solutions that are accurate even when dealing with irregular shapes or varying conditions.
x??

---

#### Weak Formulation of PDEs
To formulate the weak form of the partial differential equation (PDE), we start with the strong form and integrate it over the entire domain.

:p How do you derive the weak form of a PDE?
??x
The weak form is derived by multiplying the strong form of the PDE by an approximate trial function \(\phi(x)\) and integrating over the entire domain. This process helps in relaxing the requirements for the solution, making it more practical to solve numerically.

```math
\int_a^b dx U''(x) \phi(x) = -4\pi \int_a^b dx \rho(x) \phi(x)
```

By integrating by parts, we get:
```math
\int_a^b dx U''(x) \phi(x) = -4\pi \int_a^b dx \rho(x) \phi'(x) + (U'(x) \phi(x))|_a^b
```

Since the trial function vanishes at the boundaries, we get:
```math
-4\pi \int_a^b dx U''(x) \phi(x) = 4\pi \int_a^b dx \rho(x) \phi'(x)
```

This is the weak form of the PDE.
x??

---

#### Galerkin Spectral Decomposition
The approximate solution to the weak form is obtained by expanding the solution within each element using basis functions.

:p What are the steps involved in the Galerkin spectral decomposition?
??x
1. The domain is divided into elements, and a trial function \(U(x) \approx \sum_{j=0}^{N-1} \alpha_j \phi_j(x)\) is assumed.
2. The solution is expanded using basis functions \(\phi_i\).
3. The coefficients \(\alpha_j\) are determined by matching the solutions on each element.

The solution reduces to finding the expansion coefficients, which can be done through a Galerkin method:
```math
\sum_{j=0}^{N-1} \int_a^b dx \phi_j(x) \frac{d}{dx}(\phi_i''(x)) = -4\pi \int_a^b dx \rho(x) \phi_i'(x)
```

This leads to a system of linear equations.
x??

---

#### Solution via Linear Equations
The solution involves determining the expansion coefficients by substituting the expansions into the weak form.

:p How does one solve for the unknown coefficients in FEM?
??x
By substituting the expansion \(U(x) \approx \sum_{j=0}^{N-1} \alpha_j \phi_j(x)\) into the weak form, we get a system of linear equations:
```math
\sum_{j=0}^{N-1} \int_a^b dx \frac{d}{dx}(\phi_j(x)) \cdot \frac{d}{dx}(\phi_i(x)) = -4\pi \int_a^b dx \rho(x) \phi_i'(x)
```

This can be written in matrix form:
```math
A y = b
```
where \(y\) is a vector of the unknown coefficients, and \(A\) and \(b\) are known matrices.

For hat functions, we get:
```math
A_{ij} = \int_a^b dx \phi_i'(x) \cdot \phi_j'(x)
```

The resulting matrix \(A\) is tridiagonal.
x??

---

#### Element Matrices and Vectors
The element matrices are constructed from the integrals over derivatives of basis functions.

:p How do you construct the stiffness matrix for FEM?
??x
For hat functions, the derivatives are easy to compute analytically:
```math
\frac{d\phi_i}{dx} = \begin{cases}
0 & \text{if } x < x_{i-1} \text{ or } x > x_{i+1}, \\
\frac{x - x_{i-1}}{h_{i-1}} & \text{if } x_{i-1} \leq x \leq x_i, \\
-\frac{x_i - x}{h_i} & \text{if } x_i \leq x \leq x_{i+1}.
\end{cases}
```

The integrals are computed as:
```math
\int_{x_i}^{x_{i+1}} dx (\phi'_i)^2 = \frac{1}{h_{i-1}} + \frac{1}{h_i}, \\
\int_{x_i}^{x_{i+1}} dx \phi'_i \cdot \phi'_{i+1} = -\frac{1}{h_i}.
```

These integrals form the tridiagonal matrix \(A\) and the vector \(b\).
x??

--- 

Each flashcard covers a specific aspect of FEM, providing context and explanations to aid in understanding the concepts. The code examples help illustrate the logical steps involved in solving problems using FEM. ---

#### FEM Solution for 1D Problems
Background context: The Finite Element Method (FEM) is used to solve partial differential equations by discretizing a domain into smaller elements. In this case, we focus on 1D problems where the domain is discretized using linear basis functions.

Formulas and explanations:
- The general solution can be expanded as \( U(x) = \sum_{j=0}^{N-1} \alpha_j \phi_j(x) + U_a \phi_N(x) \), where \( \phi_0, \ldots, \phi_{N-1} \) are basis functions that vanish at the endpoints, and \( \phi_N \) is a particular solution satisfying boundary conditions.
- The matrix equation to solve becomes \( Ay = b' \).

:p What does the general form of the 1D FEM solution look like?
??x
The general form of the 1D FEM solution includes both the basis functions and a particular solution that satisfies the boundary conditions:
\[ U(x) = \sum_{j=0}^{N-1} \alpha_j \phi_j(x) + U_a \phi_N(x). \]
Here, \( \phi_0, \ldots, \phi_{N-1} \) are basis functions that vanish at the endpoints, and \( \phi_N \) is a specific solution satisfying the boundary conditions.
x??

---

#### Imposing Boundary Conditions in 1D FEM
Background context: For accurate solutions, it's crucial to impose boundary conditions correctly. In this section, we discuss how to handle both Dirichlet and Neumann boundary conditions using basis functions.

Formulas and explanations:
- For a boundary condition at \( x = a \), the solution is adjusted by adding a particular solution.
- The modified matrix equation becomes \( Ay = b' \).

:p How does the 1D FEM solution incorporate boundary conditions?
??x
The 1D FEM solution incorporates boundary conditions by adjusting the general form of the solution:
\[ U(x) - U_a \phi_0(x) + U_b \phi_N(x). \]
This adjustment ensures that the solution satisfies the given boundary conditions at \( x = a \) and \( x = b \).
x??

---

#### Linear Algebra for FEM Solutions
Background context: Once the basis functions are defined, we need to solve linear equations using efficient methods from a linear algebra library.

Formulas and explanations:
- The matrix equation is typically of the form \( Ay = b' \), where \( A \) is a sparse matrix.
- For 1D problems with \( N \) elements, the number of calculations varies approximately as \( N^2 \).

:p What is the typical form of the linear algebra problem solved in FEM?
??x
The typical form of the linear algebra problem solved in FEM for 1D problems is:
\[ Ay = b'. \]
Here, \( A \) is a sparse matrix, and solving this equation gives us the coefficients \( y \), which are then used to find the solution \( U(x) \).
x??

---

#### 2D FEM Triangulation
Background context: In 2D problems, the domain is decomposed into triangular elements, each of which can be numbered. The vertices and nodes within these triangles also need to be identified.

Formulas and explanations:
- Each triangle in the mesh is numbered from 1 to \( N \).
- Each vertex of a triangle is numbered counter-clockwise from 1 to 3.
- Nodes are numbered based on where lines intersect, typically from 1 to \( M \).

:p How does the 2D FEM triangulation work?
??x
In 2D FEM triangulation:
- The domain is decomposed into triangular elements, each numbered from 1 to \( N \).
- Each triangle has its vertices numbered counter-clockwise from 1 to 3.
- Nodes are numbered based on where lines intersect, typically from 1 to \( M \).

This process ensures that the solution can be accurately represented and solved using FEM techniques.
x??

---

#### Stiffness Matrix in 2D FEM
Background context: The stiffness matrix is a key component in 2D FEM problems. It represents the system of equations derived from the weak form of the PDE.

Formulas and explanations:
- The stiffness matrix \( A \) for 2D elements can be triangular, depending on the basis functions used.
- The load vector \( b \) is computed using integrations.

:p How does the stiffness matrix in 2D FEM typically look?
??x
The stiffness matrix in 2D FEM typically looks like this:
\[ A = \begin{bmatrix}
A_{0,0} & \cdots & A_{0,N-1} \\
\vdots & \ddots & \vdots \\
0 & \cdots & A_{N-1,N-1} \\
0 & 0 & \cdots & 1
\end{bmatrix}. \]
This triangular structure is due to the nature of the basis functions used in FEM.
x??

---

#### Solving Linear Equations in 2D FEM
Background context: After setting up the stiffness matrix and load vector, the linear equation \( Ay = b' \) needs to be solved. This involves numerical methods from a linear algebra library.

Formulas and explanations:
- The solution \( y \) is found using efficient algorithms.
- The global error can be computed using the formula provided in the text.

:p How is the solution for 2D FEM obtained?
??x
The solution for 2D FEM is obtained by solving the linear equation \( Ay = b' \). This involves:
1. Setting up the stiffness matrix \( A \) and load vector \( b' \).
2. Using efficient algorithms from a linear algebra library to solve for \( y \).

This step is crucial as it provides the coefficients used to construct the numerical solution.
x??

---

#### Piecewise-Quadratic Functions in FEM
Background context: To improve accuracy, higher-order basis functions such as piecewise-quadratic functions can be used instead of linear ones.

Formulas and explanations:
- Piecewise-quadratic functions provide a better fit for the solution, leading to more accurate results.
- The process involves adjusting the basis functions and re-solving the system equations.

:p How do piecewise-quadratic functions enhance the FEM solution?
??x
Piecewise-quadratic functions enhance the FEM solution by providing a higher-order approximation of the true solution. This leads to:
1. More accurate representation of the solution.
2. Improved convergence properties for the numerical solution.

By using quadratic basis functions, we can better capture the behavior of the physical system being modeled.
x??

---

#### 2D Capacitor Problem
Background context: The problem involves solving Laplace's equation in a 2D domain to model an electrostatic capacitor with specific charge distributions.

Formulas and explanations:
- The solution involves setting up the domain, triangulating it, and applying boundary conditions.
- The electric potential \( U(x, y) \) is computed using FEM techniques.

:p How does the 2D FEM solve for the electric potential in a capacitor?
??x
The 2D FEM solves for the electric potential in a capacitor by:
1. Setting up the domain and triangulating it.
2. Applying appropriate boundary conditions, including charge distributions at specific points.
3. Solving Laplace's equation using piecewise-quadratic basis functions to obtain accurate results.

This process models the electrostatic behavior of the capacitor with high precision.
x??

---

---
#### Finite Element Method for 1D Laplace's Equation
Background context: The provided Python script solves the one-dimensional Laplace’s equation using the finite element method. This involves setting up a matrix and vector to represent the discrete form of the partial differential equation, applying boundary conditions, and solving the resulting system.

The problem is defined over an interval with specific boundary conditions. The weak formulation leads to a linear system \( A \mathbf{u} = \mathbf{b} \), where \( A \) is the stiffness matrix, and \( \mathbf{b} \) represents external forces or sources.

:p What is the main goal of this script?
??x
The main goal is to solve the 1D Laplace's equation using the finite element method by setting up a linear system and solving it numerically. This involves creating matrices, applying boundary conditions, and finding the potential distribution in the domain.
x??

---
#### Setting Up the Stiffness Matrix \( A \)
Background context: The stiffness matrix \( A \) is constructed based on the elemental contributions from each finite element. Each element's contribution to the global stiffness matrix is calculated using a specific formula that depends on the coordinates and derivatives of the basis functions.

:p How is the stiffness matrix \( A \) set up in this script?
??x
The stiffness matrix \( A \) is set up by iterating over all elements, computing the local contributions, and summing them into the global stiffness matrix. Each element's contribution involves calculating a submatrix for its vertices and then adding it to the appropriate entries in the global matrix.

```python
# Pseudocode for setting up the stiffness matrix A
for e in range(1, Ne):
    x21 = x[node[e, 2]] - x[node[e, 1]]
    x31 = x[node[e, 3]] - x[node[e, 1]]
    x32 = x[node[e, 3]] - x[node[e, 2]]
    x13 = x[node[e, 1]] - x[node[e, 3]]
    y12 = y[node[e, 1]] - y[node[e, 2]]
    y21 = y[node[e, 2]] - y[node[e, 1]]
    y31 = y[node[e, 3]] - y[node[e, 1]]
    y23 = y[node[e, 2]] - y[node[e, 3]]
    J = x21 * y31 - x31 * y21
    
    # Evaluate A matrix, element vector ge
    A[1, 1] = -(y23 * y23 + x32 * x32) / (2 * J)
    A[1, 2] = -(y23 * y31 + x32 * x13) / (2 * J)
    # and so on for other elements...
```
x??

---
#### Applying Boundary Conditions
Background context: The boundary conditions are crucial as they define the constraints at specific points in the domain. In this script, Dirichlet boundary conditions are applied by modifying the stiffness matrix \( A \) and the right-hand side vector \( b \). These changes ensure that certain degrees of freedom are fixed to prescribed values.

:p How are the boundary conditions applied in this script?
??x
Boundary conditions are imposed by setting specific entries in the stiffness matrix \( A \) and the right-hand side vector \( b \) to zero, effectively removing those degrees of freedom. For Dirichlet conditions, rows and columns corresponding to fixed nodes are set to zero except for the diagonal entry which is set to 1, making sure that only the boundary condition value modifies the solution.

```python
# Pseudocode for applying boundary conditions
for i in range(1, Tnebc):
    for j in range(1, Nn + 1):
        if j == Ebcnod[i]:
            b[j] = b[j] - A[j, Ebcnod[i]] * Ebcval[i]
            A[Ebcnod[i], :] = 0
            A[:, Ebcnod[i]] = 0
            A[Ebcnod[i], Ebcnod[i]] = 1
            b[Ebcnod[i]] = Ebcval[i]
```
x??

---
#### Solving the Linear System
Background context: After setting up and applying the boundary conditions, a linear system \( A \mathbf{u} = \mathbf{b} \) is solved using numerical methods. In this script, the `linalg.solve` function from NumPy is used to find the solution vector \( \mathbf{u} \), which represents the potential values at each node in the domain.

:p What linear algebra method is used to solve the system?
??x
The linear system \( A \mathbf{u} = \mathbf{b} \) is solved using NumPy's `linalg.solve` function, which internally uses efficient numerical algorithms to find the solution vector \( \mathbf{u} \).

```python
# Solving the linear system using linalg.solve
V = linalg.solve(A, b)
```
x??

---
#### Interpolating Potential on a Grid
Background context: Once the potential values at each node are obtained, these need to be interpolated onto a grid for visualization. The script uses a weighted sum of nodal values based on their contributions from adjacent elements.

:p How is the potential value interpolated over the domain?
??x
The potential value at any point in the domain is interpolated using a linear combination of nodal values, where each node's contribution depends on its distance and orientation relative to the point. The weighted sum is calculated for every grid cell based on the areas formed by the nodes.

```python
# Pseudocode for interpolating potential Vgrid
for i in range(1, 11):
    for j in range(1, 11):
        for e in range(0, Ne):
            x2p = x[node[e, 2]] - X[i, j]
            x3p = x[node[e, 3]] - X[i, j]
            y2p = y[node[e, 2]] - Y[i, j]
            y3p = y[node[e, 3]] - Y[i, j]
            A1 = 0.5 * abs(x2p * y3p - x3p * y2p)
            # Calculate other areas and contributions...
            Vgrid[i, j] += N1 * V[node[e, 1]] + N2 * V[node[e, 2]] + N3 * V[node[e, 3]]
```
x??

---

