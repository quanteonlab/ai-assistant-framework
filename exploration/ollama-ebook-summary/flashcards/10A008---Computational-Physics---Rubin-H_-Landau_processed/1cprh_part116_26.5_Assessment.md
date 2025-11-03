# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 116)

**Starting Chapter:** 26.5 Assessment and Exploration

---

#### Vorticity and Stream Function Relationship

:p Explain the relationship between vorticity \( \omega \) and the stream function \( u \).
??x
The relationship between vorticity \( \omega \) and the stream function \( u \) is derived from vector calculus identities. Starting with the definition of vorticity as the curl of velocity, we have:

\[ w = \nabla \times \mathbf{v} \]

For a 2D flow where the velocity has only \( z \)-components that do not vary with \( z \), the divergence of velocity is zero (\( \nabla \cdot \mathbf{u} = 0 \)). Therefore, using vector identities:

\[ w = \nabla \times ( \nabla \times u ) = \nabla(\nabla \cdot u) - \nabla^2 u = -\nabla^2 u \]

This simplifies to the basic relation between \( u \) and \( w \):

\[ \nabla^2 u = -w \]

This equation is analogous to Poisson's equation in electrostatics but for fluid dynamics.

??x
---

#### Vorticity Form of Navier–Stokes Equation

:p How is the vorticity form of the Navier-Stokes equation obtained?
??x
The vorticity form of the Navier-Stokes equation is derived by taking the curl of the velocity form. This involves operating on both sides with \( \nabla \times \):

\[ \nu \nabla^2 w = [(\nabla \times u) \cdot \nabla]w \]

This results in two simultaneous partial differential equations (PDEs) that need to be solved:

1. For the stream function \( u \):
   \[ \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = -w \]

2. For vorticity \( w \):
   \[ \nu \left( \frac{\partial^2 w}{\partial x^2} + \frac{\partial^2 w}{\partial y^2} \right) = \frac{\partial u}{\partial y} \frac{\partial w}{\partial x} - \frac{\partial u}{\partial x} \frac{\partial w}{\partial y} \]

These equations resemble a mixture of Poisson's equation and the wave equation.

??x
---

#### Vorticity Difference Equation on a Grid

:p How are the vorticity difference equations implemented on an \( N_x \times N_y \) grid?
??x
To implement the vorticity difference equations on an \( N_x \times N_y \) grid with uniform spacing \( h \):

\[ x = i\Delta x = ih, \quad y = j\Delta y =jh, \quad i=0,\dots,N_x, \quad j=0,\dots,N_y \]

Using central difference approximations for the Laplacians of \( u \) and \( w \), we get:

\[ \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2} \]

For the product of first derivatives:

\[ \frac{\partial u}{\partial y} \frac{\partial w}{\partial x} \approx \frac{u_{i,j+1} - u_{i,j-1}}{2h} \cdot \frac{w_{i+1,j} - w_{i-1,j}}{2h} \]

The difference vorticity Navier-Stokes equation for \( u \) is:

\[ u_{i,j} = \frac{1}{4}(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} + h^2 w_{i,j}) \]

The difference vorticity equation for \( w \) is:

\[ w_{i,j} = \frac{1}{4}(w_{i+1,j} + w_{i-1,j} + w_{i,j+1} + w_{i,j-1}) - \frac{R}{16} \left( [u_{i,j+1} - u_{i,j-1}] \times [w_{i+1,j} - w_{i-1,j}] - [u_{i+1,j} - u_{i-1,j}] \cdot [w_{i,j+1} - w_{i,j-1}] \right) \]

Here, \( R = \frac{1}{\nu} \left( \frac{V_0 h}{\nu} \right) \), known as the grid Reynolds number.

??x
---

#### Relaxation Algorithm for Vorticity and Stream Function

:p How does the relaxation algorithm work for vorticity and stream function?
??x
The relaxation algorithm iteratively updates the values of \( u \) and \( w \) to converge to a solution. The process is separated into two functions: one for relaxing the stream function \( u \) and another for relaxing the vorticity \( w \).

For the stream function:

```python
def relax_stream_function(u, N_x, N_y, h, R):
    # Update u using its difference equation
    for i in range(1, N_x - 1):
        for j in range(1, N_y - 1):
            u[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] + h**2 * w[i,j]) / 4
```

For the vorticity:

```python
def relax_vorticity(w, N_x, N_y, h, R):
    # Update w using its difference equation
    for i in range(1, N_x - 1):
        for j in range(1, N_y - 1):
            numerator = (w[i+1,j] + w[i-1,j] + w[i,j+1] + w[i,j-1])
            denominator = 4
            cross_term = ((u[i,j+1] - u[i,j-1]) * (w[i+1,j] - w[i-1,j]) - 
                          (u[i+1,j] - u[i-1,j]) * (w[i,j+1] - w[i,j-1]))
            w[i,j] = numerator / denominator - R * cross_term / 16
```

??x
---

#### Simulation Parameters and Boundary Conditions

:p What are the initial parameters for setting up a simulation of fluid flow around a beam?
??x
For setting up an initial simulation, you can start with:

- Beam length \( L = 8h \)
- Beam height \( H = h \)
- Reynolds number \( R = 0.1 \)
- Intake velocity \( V_0 = 1 \)

During debugging, keep the grid size small, e.g., \( N_x = 24 \) and \( N_y = 70 \).

??x
---

#### Convergence of the Algorithm

:p How do you determine the number of iterations necessary for convergence in the relaxation algorithm?
??x
To explore the convergence of the algorithm:

1. Print out the iteration number and values upstream, above, and downstream from the beam.
2. Determine the number of iterations necessary to obtain three-place convergence for successive relaxation (\( \omega = 1 \)).
3. Determine the number of iterations necessary to obtain three-place convergence for successive over-relaxation (\( \omega \approx 1.3 \)).

Use this number as a baseline for future calculations.

??x
---

#### Beam’s Horizontal Placement

:p How can you observe the development of standing waves due to beam placement?
??x
Change the horizontal placement of the beam so that it allows observation of the undisturbed current entering from the left, and then developing into a standing wave. This may require increasing the size of your simulation volume to see the effect of all boundary conditions.

??x
---

#### Visualization of Stream Function and Vorticity

:p How can you create surface plots including contours of the stream function \( u \) and vorticity \( w \)?
??x
To create surface plots with contour lines for both the stream function \( u \) and vorticity \( w \):

1. Use a plotting library like Matplotlib in Python.
2. Plot the surfaces using `plot_surface` or `contourf` functions.

For example:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming u and w are 2D arrays of size N_x x N_y
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface plot for stream function u
ax.plot_surface(X, Y, u, cmap='viridis')
plt.show()

# Contour plot for vorticity w
plt.contourf(X, Y, w, levels=20)
plt.colorbar()
plt.show()
```

These visualizations help in understanding the behavior of the fluid around the beam.

??x
---

#### Region for Fish to Rest

:p Is there a region behind the beam where a big fish can rest?
??x
The simulation results show that there is indeed a region behind the beam where the flow is less turbulent. This area, known as the "vortex core" or "wake", provides a relatively calmer environment where a big fish could rest.

??x
---

---
#### Velocity as a Vector
Background context explaining that velocity is a vector with two components, and these individual components are interesting to visualize. A vector plot works well for visualization purposes.

:p What does the term "velocity" refer to in this context?
??x
Velocity refers to the speed and direction of fluid flow over a beam, which is represented as a vector with two components: one along the x-axis (streamwise) and another along the y-axis. A vector plot can help visualize these components effectively.
x??

---
#### Exploring Changes in Reynolds Number \( R \)
Background context explaining that increasing the Reynolds number \( R \) changes the flow pattern, starting from \( R = 0 \) and gradually increasing while watching for numeric instabilities. To overcome numerical instabilities, reducing the size of the relaxation parameter \( \omega \) is suggested.

:p How does changing the Reynolds number \( R \) affect fluid flow?
??x
Increasing the Reynolds number \( R \) affects the flow pattern significantly. For small \( R \) values, the flow around the beam remains smooth and attached to the surface. However, as \( R \) increases beyond a certain threshold, the flow may separate from the back edge of the beam, leading to the formation of a small vortex. This transition can be observed by incrementally increasing \( R \) while monitoring for numerical instabilities, which can be mitigated by adjusting the relaxation parameter \( \omega \).
x??

---
#### Determining Flow Behind a Circular Rock
Background context explaining that one needs to determine the flow behind a circular rock in the stream. This involves understanding how boundary conditions and the shape of objects affect fluid dynamics.

:p How would you determine the flow behavior behind a circular object placed in a stream?
??x
To determine the flow behavior behind a circular object (rock) in a stream, you need to simulate the fluid dynamics around it using appropriate numerical methods such as solving the Navier-Stokes equations. This involves setting up boundary conditions that accurately represent the rock's shape and analyzing the resulting velocity and pressure fields.

:p How might boundary conditions affect this simulation?
??x
Boundary conditions significantly influence the flow simulation results, especially near solid objects like a circular rock. Different boundary conditions can lead to different flow behaviors, such as changes in velocity profiles or pressure distributions. By exploring various boundary conditions at the outlet downstream of the rock, you can determine which ones produce realistic and stable results.
x??

---
#### Pressure Variation Around the Beam
Background context explaining that one needs to verify how the pressure varies around a beam for small Reynolds numbers \( R \) but separates from the back edge for large \( R \).

:p What is the expected behavior of pressure variation around a beam as \( R \) changes?
??x
For small Reynolds numbers \( R \), the flow around the beam remains smooth, and there are no significant pressure variations. As \( R \) increases beyond a critical value, the flow may separate from the back edge of the beam, leading to the formation of vortices and increased pressure gradients. This behavior can be observed by monitoring the pressure distribution around the beam as \( R \) is incremented.
x??

---
#### Numerical Relaxation Method
Background context explaining that the code uses a relaxation method to solve the Navier-Stokes equations for flow around a plate, involving boundary conditions and iterative updates.

:p What does the function `relax()` do in this context?
??x
The function `relax()` implements a numerical relaxation method to update the stream function \( u \) and vorticity \( w \) fields iteratively. This method involves solving the Navier-Stokes equations by relaxing the values of these fields at each grid point, ensuring that they converge towards an accurate solution.

:p Can you explain the logic behind the `relax()` function?
??x
The `relax()` function updates both the stream function \( u \) and vorticity \( w \) using a relaxation method. For the stream function:
```python
r1 = omega * ((u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] + h*h*w[i,j]) / 4 - u[i,j])
u[i, j] += r1
```
This updates the value of \( u \) at grid point \((i, j)\) based on its neighbors and the vorticity field. For the vorticity:
```python
a1 = w[i+1, j] + w[i-1,j] + w[i,j+1] + w[i,j-1]
a2 = (u[i,j+1] - u[i,j-1]) * (w[i+1,j] - w[i-1,j])
a3 = (u[i+1,j] - u[i-1,j]) * (w[i,j+1] - w[i,j-1])
r2 = omega * ((a1 - (R/4.) * (a2 - a3)) / 4. - w[i,j])
w[i, j] += r2
```
This updates the vorticity at grid point \((i, j)\) based on its neighbors and the stream function. The relaxation parameter \( \omega \) controls how much each field is updated in one iteration.

:p How does the `relax()` function handle boundary conditions?
??x
The `relax()` function handles boundary conditions by defining specific updates for regions where fluid boundaries or no-slip conditions are applied. For example, at the inlet and outlet:
```python
# Inlet (fluid surface)
u[1, j] = u[0, j]
w[0, j] = 0.
# Outlet (fluid surface)
u[Nxmax, j] = u[Nxmax-1, j]
w[Nxmax, j] = w[Nxmax-1, j]
```
These boundary conditions ensure that the flow properties are correctly set at the boundaries to maintain physical consistency.

:p How does the `beam()` function modify the stream and vorticity fields?
??x
The `beam()` function modifies the stream and vorticity fields based on the geometry of a beam. It updates the fields along the sides, front, and back of the beam:
```python
# Sides
w[IL, j] = -2 * u[IL-1, j] / (h*h)
w[IL+T, j] = -2 * u[IL + T + 1, j] / (h*h)
# Front and back of the beam
for i in range(IL, IL+T+1):
    w[i, H-1] = -2 * u[i, H] / (h*h);
    u[IL, j] = 0.
    u[IL+T, j] = 0.
    u[i, H] = 0;
```
These updates ensure that the flow properties are adjusted according to the presence of the beam.

x??

---

#### Finite Element Method (FEM) Overview
Finite element method is a numerical technique used to solve partial differential equations (PDEs). It involves dividing the domain into smaller, manageable elements and approximating the solution within each element. This approach allows for more accurate solutions compared to finite difference methods in complex geometries.
:p What is FEM and how does it differ from finite differences?
??x
FEM is a numerical technique that patches together approximate solutions on small finite elements to obtain the full solution, whereas finite differences approximate derivatives directly. FEM is generally faster but requires more setup effort due to its complexity.
x??

---

#### Potential Between Two Metal Plates (Analytic Solution)
The problem involves determining the electric potential between two conducting plates with a uniform charge density \(\rho(x)\) and different potentials at the boundaries. The relation between charge density \(\rho(x)\) and potential \(U(x)\) is given by Poisson’s equation.
:p What is the analytic solution for the potential between two metal plates?
??x
The potential \(U(x)\) changes only in the x-direction, and thus the PDE becomes an ODE:
\[
\frac{d^2 U(x)}{dx^2} = -4\pi \rho(x)
\]
Given that \(\rho(x) = 1/4\pi\) for \(0 < x < 1\), we have:
\[
\frac{d^2 U(x)}{dx^2} = -1
\]
The solution to this ODE, subject to the Dirichlet boundary conditions \(U(a=0) = 0\) and \(U(b=1) = 1\), is:
\[
U(x) = -x(3-x)
\]

No code examples are necessary for the analytic solution.
x??

---

#### Problem Setup: Two Metal Plates
The problem involves two conducting plates with a uniform charge density \(\rho(x)\) between them. The lower plate is at potential \(U_a\), and the upper plate is at potential \(U_b\). We need to determine the electric potential in this region.
:p What are the key parameters and conditions for the two metal plates problem?
??x
The key parameters and conditions include:
- Distance between the plates: \(b - a\)
- Lower plate potential: \(U_a\)
- Upper plate potential: \(U_b\)
- Uniform charge density between the plates: \(\rho(x) = 1/4\pi\)

The Dirichlet boundary conditions are:
\[
U(a=0) = U_a, \quad U(b=1) = U_b
\]

These conditions help in formulating the PDE and solving for \(U(x)\).
x??

---

#### Finite Differences vs. FEM
Finite differences approximate derivatives directly on a grid, while finite elements divide the domain into smaller patches and solve the problem within each patch.
:p What are the main differences between finite difference methods and finite element methods?
??x
The main differences are:
- **Finite Differences**: Approximates derivatives using simple formulas at discrete points. Faster execution but may be less accurate for complex geometries.
- **Finite Elements**: Solves the PDE in small, manageable elements. More accurate for complex geometries but requires more setup and computation.

For practical implementation, FEM is often used via specialized packages like FiPy in Python.
x??

---

#### Implementing FEM with FiPy
FiPy is a Python package that implements finite element methods to solve partial differential equations. It automates much of the setup process for complex geometries.
:p How can we use FiPy to implement the two metal plates problem?
??x
To implement the problem using FiPy, you would typically:
1. Define the geometry and initial conditions.
2. Set up the equation and boundary conditions.
3. Solve the PDE.

Here is a simplified example of how this might be implemented in Python with FiPy:

```python
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, Viewer

# Define grid size and spacing
nx, ny = 100, 100
xMin, xMax = 0., 1.
yMin, yMax = 0., 1.

# Create a 2D grid
mesh = Grid2D(nx=nx, ny=ny, dx=xMax - xMin, dy=yMax - yMin)

# Define the variable (potential U)
U = CellVariable(name="Potential", mesh=mesh)

# Set initial and boundary conditions
U.constrain(0., mesh.facesLeft)
U.constrain(1., mesh.facesRight)

# Define the PDE equation: Laplace's Equation in 2D
eq = DiffusionTerm(coeff=-1.) == U

# Solve the equation
for step in range(100):  # Number of time steps
    eq.solve(var=U)

# View the solution
viewer = Viewer(U, viewerName="Potential")
```

In this example:
- A 2D grid is created.
- Initial and boundary conditions are set.
- The PDE equation (Laplace’s Equation) is defined and solved iteratively.
x??

