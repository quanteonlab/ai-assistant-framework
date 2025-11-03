# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 108)

**Starting Chapter:** 21.4 Alternate Capacitor Problems

---

#### Relaxation and Overrelaxation Methods

Background context: The Jacobi method is a simple approach to solving partial differential equations (PDEs) by iteratively updating potential values on a grid. However, it can be slow due to its lack of symmetry between iterations. The Gauss-Seidel method improves upon this by using the latest available values during each iteration, which generally leads to faster convergence. Successive Overrelaxation (SOR) further accelerates the process by adding an amplification factor \(\omega\) to the correction term.

:p What is the difference between the Jacobi and Gauss-Seidel methods?
??x
The Jacobi method updates all potential values simultaneously using the old values, whereas the Gauss-Seidel method uses the most recent updated values during each iteration. This makes the Gauss-Seidel method more efficient by reducing the lag in information flow.
x??

---
#### Successive Overrelaxation (SOR)

Background context: SOR is an iterative technique that modifies the basic Jacobi or Gauss-Seidel methods to achieve faster convergence. It introduces a parameter \(\omega\) which scales the correction term \(r_i, j\), aiming for even quicker convergence when properly tuned.

:p How does Successive Overrelaxation (SOR) modify the update formula?
??x
The SOR method modifies the basic Jacobi or Gauss-Seidel update rule by adding an amplification factor \(\omega\) to the correction term \(r_i, j\):
\[ U_{new}^{i,j} = U_{old}^{i,j} + \omega r^{i,j}, \]
where
\[ r^{i,j} = \frac{1}{4}[U_{old}^{i+1,j} + U_{new}^{i-1,j} + U_{old}^{i,j+1} + U_{new}^{i,j-1}] - U_{old}^{i,j}. \]

The value of \(\omega\) can range from 1 to 2, with values greater than 1 leading to overrelaxation and potentially faster convergence.
x??

---
#### Capacitor Problems

Background context: Realistic capacitor problems involve non-uniform electric fields due to finite dimensions and edge effects. Numerical simulations can model these scenarios by solving Poisson's or Laplace's equations under specific boundary conditions.

:p How would you set up the potential for a simple parallel-plate capacitor in a grounded box?
??x
For a simple parallel-plate capacitor, assume thin conductive sheets maintained at 100V and -100V. Since these are conductors, they must be equipotential surfaces. The simulation should solve Laplace's equation with fixed voltage plates on the top and bottom:
```java
// Pseudocode for setting up potential values
for each point (i, j) in the grid {
    if (point is at the top plate) U[i,j] = 100;
    else if (point is at the bottom plate) U[i,j] = -100;
    else U[i,j] = solveLaplaceEquation(U);
}
```
x??

---
#### Edge Effects and Fringe Fields

Background context: In real capacitors, electric fields vary near the edges due to edge effects and extend beyond the capacitor boundaries. These fringe fields can be modeled by solving Poisson's equation with appropriate boundary conditions.

:p How would you model a realistic capacitor with finite-width plates?
??x
To model a realistic capacitor with finite-width plates, solve Poisson's equation in the region including the plates:
\[ \nabla^2 U(x,y) = -\frac{\rho}{\epsilon_0}, \]
where \(\rho\) is the charge density on the plates. Outside this region, use Laplace's equation.

Experiment with different values of \(\rho\) to achieve a potential similar to that shown in Figure 21.5.
x??

---
#### Charge Distribution on Finite-Thickness Plates

Background context: For capacitors with finite-thickness conducting plates, charges redistribute themselves due to the non-uniform electric field. This can be modeled by solving Laplace's equation and then using it to determine the charge density.

:p How would you model the charge distribution on finite conducting plates?
??x
To model the charge distribution on finite conducting plates:
1. Solve Laplace’s equation for \(U(x, y)\) with appropriate boundary conditions (e.g., 100V at top and -100V at bottom).
2. Substitute \(U(x,y)\) into Poisson's equation to determine the charge density \(\rho\):
\[ \rho = \epsilon_0 \left( \nabla^2 U(x, y) \right). \]
x??

---
#### Arbitrary Boundary Conditions

Background context: The numerical solution can be applied to arbitrary boundary conditions. Exploring triangular and sinusoidal boundaries helps in understanding how different geometries affect the field distribution.

:p How would you model a triangular boundary condition?
??x
For a triangular boundary, define:
\[ U(x) = \begin{cases} 
200 \frac{x}{w}, & x \leq w/2 \\
100(1 - \frac{x}{w}), & x \geq w/2 
\end{cases}. \]
This defines a piecewise linear function that can be used as an initial boundary condition for the simulation.
x??

---
#### Square Conductors and Electric Fields

Background context: Designing equipment with square conductors involves solving the electric field within a grounded inner box and outer grounded box to prevent sparking. The goal is to determine where the field is most intense.

:p How would you model the electric field between two square conductors?
??x
To model the electric field between two square conductors:
1. Set up the boundary conditions for a small metal box at 100V and a larger grounded one.
2. Solve for the potential \(U(x, y)\) using the appropriate method (e.g., SOR).
3. Plot the potential and equipotential surfaces.
4. Sketch in the electric field lines to deduce where the field is most intense.

Modify the program to satisfy these boundary conditions:
```java
// Pseudocode for setting up potential values
for each point (i, j) in the grid {
    if (point inside small box) U[i,j] = 100;
    else U[i,j] = solveLaplaceEquation(U);
}
```
x??

---
#### Cracked Cylindrical Capacitor

Background context: Modeling a cracked cylindrical capacitor with an inner and outer cylinder connected by a crack involves determining how the small crack affects the field configuration. This requires placing both cylinders within a large grounded box to ensure a unique solution.

:p How would you model the effect of a small crack in a cylindrical capacitor?
??x
To model the effect of a small crack in a cylindrical capacitor:
1. Place an inner cylinder at -100V and an outer cylinder at 100V.
2. Introduce a small crack to connect them.
3. Ensure both cylinders are within a large, grounded box to maintain uniqueness.
4. Solve Laplace's equation for the potential \(U(x, y)\) in this configuration.

This setup helps determine how the field is affected by the presence of the crack.
x??

---

#### Modifying LaplaceLine.py for Capacitor Simulation
Background context: The provided `LaplaceLine.py` script solves Laplace's equation in a square domain and visualizes the potential. To adapt it for simulating capacitors, we need to modify initial conditions, boundary conditions, and possibly the relaxation algorithm.

:p How should you start modifying `LaplaceLine.py` to find the electric potential for a capacitor?
??x
To start, you should:
1. Set appropriate boundary conditions to represent the two plates of the capacitor.
2. Modify the initial condition U(i,Nmax) = 99 to reflect the top plate's voltage.

For example:
- Set `V[0,k]` and `V[Nmax-1,k]` to 100 V (top and bottom plates).
- Keep other boundaries at 0 V.

You can use a simple loop to set these values:

```python
# Modify initial conditions for the capacitor
for k in range(Nmax):
    V[0, k] = 100.0  # Top plate
    V[Nmax - 1, k] = 0.0  # Bottom plate
```

x??

---

#### Iterative Solution and Convergence
Background context: The iterative relaxation method used in `LaplaceLine.py` needs to be refined for better convergence and accuracy. This involves checking the potential changes and stopping once a certain threshold is met.

:p How do you implement an iterative solution with convergence testing?
??x
To implement this, use a loop that iterates until the change in potential along the diagonal is below a specified tolerance:

```python
trace = 0
threshold = 1e-4

while trace > threshold:
    for i in range(1, Nmax - 1):
        for j in range(1, Nmax - 1):
            V[i, j] = 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1])
    
    # Calculate the trace to check for convergence
    trace = sum(abs(V[i, i]) for i in range(Nmax))
```

If `trace` is less than the threshold, the loop will break, indicating that the solution has converged.

x??

---

#### Successive Overrelaxation (SOR) Method
Background context: The SOR method can accelerate convergence by adjusting the relaxation parameter \(\omega\). This involves finding a good value of \(\omega\) through trial and error to double the speed of the algorithm.

:p How do you implement the SOR technique in `LaplaceLine.py`?
??x
To implement the SOR technique, modify the iteration formula by introducing an over-relaxation parameter:

```python
omega = 1.5  # Choose a value between 1 and 2 for convergence; start with trial values

for i in range(1, Nmax - 1):
    for j in range(1, Nmax - 1):
        V[i, j] = (1 - omega) * 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1]) + omega * V[i, j]
```

You can experiment with different values of \(\omega\) to find the optimal one that doubles the convergence rate.

x??

---

#### Visualization of Equipotential and Electric Field Lines
Background context: After obtaining the potential distribution, you need to visualize it as equipotential lines. The electric field \(E = -\nabla U\) can be derived using central differences for better visualization.

:p How do you compute the electric field from the potential?
??x
To compute the electric field, use finite difference approximations:

```python
Ex[i, j] = (V[i + 1, j] - V[i - 1, j]) / (2 * Δ)
Ey[i, j] = (V[i, j + 1] - V[i, j - 1]) / (2 * Δ)
```

Where \(\Delta\) is the grid spacing.

To visualize these fields, you can use arrows or lines:

```python
from matplotlib.quiver import quiver

# Plot electric field vectors
E = np.stack([Ex, Ey], axis=-1)
p.quiver(X, Y, E[:, :, 0], E[:, :, 1], color='blue')
```

x??

---

#### Realistic Capacitor Simulation
Background context: For a more realistic capacitor, reduce the plate separation to about \( \frac{1}{10} \) of the plate length. This will make the electric field more condensed and uniform between the plates.

:p How do you modify `LaplaceLine.py` for a realistic capacitor?
??x
To simulate a realistic capacitor with reduced plate separation, adjust the initial conditions to reflect the smaller distance:

```python
# Reduce plate separation (example)
Nmax = 100
Delta = 1.0 / Nmax

for k in range(Nmax):
    V[0, k] = 100.0  # Top plate
    V[int(0.9 * Nmax), k] = 0.0  # Bottom plate (at approximately 9/10 of the length)
```

This adjustment will ensure that the electric field is more condensed and uniform between the plates.

x??

---

#### Comparison with Analytic Solution
Background context: Compare your numerical solution to the analytic one for a parallel-plate capacitor given by \( U(x, y) = V_0 \frac{y}{d} \), where \( d \) is the plate separation. Note that high precision may require summing many terms.

:p How do you compare the numerical and analytical solutions?
??x
To compare the numerical solution with the analytic one:

1. Calculate the analytic potential at each grid point.
2. Plot both potentials side by side for comparison.

Example of calculating the analytic potential in Python:

```python
# Define parameters
V0 = 100
d = Nmax / 5

U_analytic = V0 * (Y / d)

# Plot both solutions
p.plot_surface(X, Y, U_numerical, label='Numerical')
p.plot_surface(X, Y, U_analytic, label='Analytic', color='green')
```

This will allow you to visually and numerically compare the accuracy of your numerical solution.

x??

---

#### Parabolic Heat Equation

Background context: The heat equation describes how temperature distributes over time within a material. For a one-dimensional bar, it is given by:
\[
\frac{\partial T(x,t)}{\partial t} = \frac{K}{C \rho} \frac{\partial^2 T(x,t)}{\partial x^2}
\]
where \( K \) is thermal conductivity, \( C \) is specific heat, and \( \rho \) is density.

:p What is the parabolic heat equation for a one-dimensional bar?
??x
The parabolic heat equation describes how temperature distributes over time within a material in one dimension:
\[
\frac{\partial T(x,t)}{\partial t} = \frac{K}{C \rho} \frac{\partial^2 T(x,t)}{\partial x^2}
\]
This equation is used to model the flow of heat through the bar, where \( K \) is thermal conductivity, \( C \) is specific heat capacity, and \( \rho \) is density.
x??

---

#### Analytic Solution via Separation of Variables

Background context: The solution can be obtained by assuming a product form for the temperature function:
\[
T(x,t) = X(x)\Theta(t)
\]
This leads to two ordinary differential equations (ODEs).

:p What is the analytic solution approach for the heat equation?
??x
The analytic solution involves assuming that the temperature function separates into spatial and temporal parts:
\[
T(x,t) = X(x)\Theta(t)
\]
Substituting this into the heat equation results in two ODEs, one for each part. These are solved to find specific functions \( X(x) \) and \( \Theta(t) \), leading to a general solution.
x??

---

#### Leapfrog Time Stepping Algorithm

Background context: The leapfrog algorithm is used for time-stepping problems. It uses forward and central differences for approximating derivatives.

:p What is the leapfrog algorithm?
??x
The leapfrog algorithm is an explicit method for solving PDEs by discretizing both space and time. For the heat equation, it uses a forward difference for the time derivative and a central difference for the spatial second derivative.
\[
T(x,t+\Delta t) - T(x,t) = \frac{K}{C \rho} \left[ T(x+\Delta x, t) + T(x-\Delta x, t) - 2T(x,t) \right] \Delta t
\]
This is rearranged to:
\[
T_{i,j+1} = T_{i,j} + \eta [T_{i+1,j} + T_{i-1,j} - 2T_{i,j}]
\]
where \( \eta = \frac{K \Delta t}{C \rho (\Delta x)^2} \).

:p What is the formula for the leapfrog algorithm?
??x
The formula for the leapfrog algorithm in the context of the heat equation is:
\[
T_{i,j+1} = T_{i,j} + \eta [T_{i+1,j} + T_{i-1,j} - 2T_{i,j}]
\]
where \( \eta = \frac{K \Delta t}{C \rho (\Delta x)^2} \).

This formula is used to step the temperature forward in time using known values from an earlier time and adjacent spatial points.
x??

---

#### Von Neumann Stability Analysis

Background context: The stability of numerical solutions can be analyzed through von Neumann's method. For a linear equation, the solution form after \( j \) steps is:
\[
T_{i,j} = \zeta(k)^j e^{ik \Delta x}
\]
where \( k \) and \( \zeta(k) \) are unknown wave vector and amplification factor.

:p What is von Neumann's stability analysis used for?
??x
Von Neumann's stability analysis is used to determine the conditions under which a numerical solution to a PDE converges. For linear equations, it involves assuming that after \( j \) time steps, the approximate solution has the form:
\[
T_{i,j} = \zeta(k)^j e^{ik \Delta x}
\]
The stability condition requires that the amplification factor \( |\zeta(k)| < 1 \).

This analysis helps ensure that numerical solutions do not diverge and provide a reliable approximation to the true solution.
x??

---

#### Initial and Boundary Conditions

Background context: The initial conditions for the problem are given as:
\[
T(x, t=0) = 100^\circ C
\]
and boundary conditions at both ends of the bar:
\[
T(0,t) = T(L,t) \equiv 0^\circ C
\]

:p What are the initial and boundary conditions for the aluminum bar?
??x
The initial and boundary conditions for the aluminum bar are:
- Initial condition: The entire bar is initially at \( 100^\circ C \).
- Boundary conditions: Both ends of the bar are in contact with ice, so their temperature remains \( 0^\circ C \) at all times.

These conditions ensure that heat flows from the hot interior to the cold boundaries.
x??

---

