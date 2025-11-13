# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 115)

**Starting Chapter:** 26.2 Flow Through Parallel Plates

---

#### Incompressibility Condition and Continuity Equation
Background context explaining the concept. The continuity equation is a fundamental principle in fluid dynamics that expresses the equality of inflow and outflow, known as the condition of incompressibility. This means that the divergence of the velocity field is zero: $\nabla \cdot \mathbf{v} = 0$.
:p What does the continuity equation express?
??x
The continuity equation expresses the conservation of mass in a fluid flow, ensuring that the amount of fluid entering a volume equals the amount leaving it. In this case, because the problem is steady-state and involves incompressible fluid (water), the time derivative of density vanishes, simplifying to $\nabla \cdot \mathbf{v} = 0$.
x??

---
#### Navier-Stokes Equation Simplification
Background context explaining the concept. The Navier-Stokes equation describes the velocity changes resulting from pressure and viscous forces. For steady-state flow of an incompressible fluid, several terms can be simplified or ignored.
:p What are the key simplifications made to the Navier-Stokes equation for this problem?
??x
For a steady-state flow of an incompressible fluid, we set all time derivatives of velocity to zero and ignore z-dependence because the stream is much wider than the plate width. This leads to simplified partial differential equations (PDEs):
$$\frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} = 0,$$and$$\nu \left( \frac{\partial^2 v_x}{\partial x^2} + \frac{\partial^2 v_x}{\partial y^2} \right) = v_x \frac{\partial v_x}{\partial x} + v_y \frac{\partial v_x}{\partial y} + \frac{1}{\rho} \frac{\partial P}{\partial x},$$
$$\nu \left( \frac{\partial^2 v_y}{\partial x^2} + \frac{\partial^2 v_y}{\partial y^2} \right) = v_x \frac{\partial v_y}{\partial x} + v_y \frac{\partial v_y}{\partial y} + \frac{1}{\rho} \frac{\partial P}{\partial y}.$$x??

---
#### Boundary Conditions for Parallel Plates
Background context explaining the concept. In this problem, we consider flow through parallel plates with specific boundary conditions to solve for the velocity profile.
:p What are the boundary conditions at the inlet of the integration domain?
??x
At the inlet, where fluid enters the integration domain with a horizontal velocity $V_0$, the boundary conditions are:
$$v_x = V_0, \quad v_y = 0.$$

This means the fluid enters with only horizontal velocity components and no vertical components.
x??

---
#### Laminar Flow and Streamlines
Background context explaining the concept. The flow through parallel plates is expected to be laminar if the plate separation $H $ and length$L $ are small compared to the stream size, and if the kinematic viscosity$\nu $ is sufficiently large or the velocity$V_0$ is not too high.
:p What defines laminar flow in this context?
??x
Laminar flow is defined as a smooth flow where fluid elements move along smooth paths that do not close on themselves. In this case, if the plates are thin and the flow far upstream of them is unaffected, we can model the flow using simplified equations for steady-state, incompressible, laminar flow between parallel plates.
x??

---
#### Parabolic Velocity Profile
Background context explaining the concept. For a fluid flowing through parallel plates, under certain conditions, there exists an analytic solution known as a parabolic velocity profile.
:p What is the equation of the parabolic velocity profile for this problem?
??x
The parabolic velocity profile for flow between parallel plates is given by:
$$\rho \nu v_x(y) = \frac{1}{2} \frac{\partial P}{\partial x} (y^2 - yH).$$

This equation describes how the velocity $v_x $ varies linearly with the distance$y$ from the plate surface, creating a parabolic profile.
x??

---
#### Symmetry Boundary Condition
Background context explaining the concept. The flow between parallel plates is symmetric about the centerline plane. This symmetry allows us to simplify the problem by considering only half of the domain and applying appropriate boundary conditions.
:p What does the symmetry assumption imply for the velocity at the centerline?
??x
The symmetry assumption implies that the velocity components are zero on the centerline, which acts as a plane of symmetry:
$$v_x = v_y = 0 \quad \text{at} \quad y = \frac{H}{2}.$$

This ensures that any flow variation above the centerline is mirrored below it.
x??

---
#### Outlet Boundary Condition
Background context explaining the concept. At the outlet, where fluid leaves the integration domain, we can assume either a physical condition or an idealized one based on the problem's nature. In this case, we assume the outlet acts as a physical gap with zero pressure.
:p What boundary condition is applied at the outlet of the stream?
??x
At the outlet, the fluid exits into a space where it returns to its unperturbed state. Assuming a physical outlet, the water pressure equals zero:
$$

P = 0.$$

This implies that any flow leaving the domain will have no pressure constraint and can freely exit.
x??

---

#### Fluid Hydrodynamics Context
Background context: The fluid dynamics problem described involves a scenario where fluid flows through a narrow gap between two plates, and the velocity of the fluid does not change perpendicular to the outlet. This situation is typical when modeling flow conditions such as those found at the end of a garden hose or similar geometries.

:p What are the conditions for the fluid hydrodynamics in this context?
??x
The conditions include zero pressure gradient along the direction normal to the outlet, and no velocity component perpendicular to the plates. Mathematically:
$$

P = 0, \quad \frac{\partial v_x}{\partial x} = \frac{\partial v_y}{\partial x} = 0.$$

Additionally, due to symmetry about the y=0 plane, there is no flow through this plane, implying:
$$v_y = 0, \quad \frac{\partial v_y}{\partial y} = 0.$$

These conditions arise from the assumption that plates are along streamlines and are negligibly thin. This ensures all streamlines are parallel to the plates, the water surface, and hence $v_y$ is zero everywhere.

x??

---

#### Symmetry Plane Condition
Background context: The flow symmetry about the y=0 plane implies no components of velocity perpendicular to this plane. Since the plates are along the streamlines and negligibly thin, all streamlines must be parallel to both the plates and the water surface.

:p What does the condition $v_y = 0$ imply for fluid flow?
??x
The condition $v_y = 0$ implies that there is no vertical component of velocity in the flow. This means the fluid enters horizontally and the plates do not change the vertical direction of the flow, maintaining symmetry about the centerline.

x??

---

#### Navier–Stokes Difference Equation for Velocity Components
Background context: The Navier-Stokes equations are discretized to solve for velocity components on a grid using finite differences. This is done in two dimensions with spacing $h$ in both directions.

:p What is the central-difference approximation used for the Navier-Stokes equation?
??x
The central-difference approximation uses finite differences to express derivatives. For example, the continuity equation and momentum equations are transformed into difference forms:
$$v(x)_{i+1,j} - v(x)_{i-1,j} + v(y)_{i,j+1} - v(y)_{i,j-1} = 0,$$and$$v(x)_{i+1,j} + v(x)_{i-1,j} + v(x)_{i,j+1} + v(x)_{i,j-1} - 4v(x)_{i,j} = \frac{h^2}{2} (v(x)_{i+1,j} - v(x)_{i-1,j}) + \frac{h^2}{2} (v(y)_{i,j+1} - v(y)_{i,j-1}) + h^2 \left(\frac{P_{i+1,j} - P_{i-1,j}}{4}\right).$$

For the y-component:
$$v(y)_{i+1,j} + v(y)_{i-1,j} + v(y)_{i,j+1} + v(y)_{i,j-1} - 4v(y)_{i,j} = \frac{h^2}{2} (v(x)_{i+1,j} - v(x)_{i-1,j}) + \frac{h^2}{2} (v(y)_{i,j+1} - v(y)_{i,j-1}) + h^2 \left(\frac{P_{i,j+1} - P_{i,j-1}}{4}\right).$$

Since $v_y = 0 $, the equation simplifies to solving for$ v(x)$.

x??

---

#### Simplifying the Equation
Background context: Given that the y-component of velocity is zero, the Navier-Stokes difference equations simplify significantly. The key equation now focuses on the x-component.

:p What simplified form does the Navier–Stokes equation take when $v_y = 0$?
??x
The simplified form of the equation for $v(x)$ becomes:
$$4v(x)_{i,j} = v(x)_{i+1,j} + v(x)_{i-1,j} + v(x)_{i,j+1} + v(x)_{i,j-1} - h^2 \left( v(x)_{i+1,j} - v(x)_{i-1,j} \right) - h^2 \left( v(y)_{i,j+1} - v(y)_{i,j-1} \right) - h^2 \left( P_{i+1,j} - P_{i-1,j} \right).$$

Given $v_y = 0$ and the central-difference approximations, we get:
$$4v(x)_{i,j} = v(x)_{i+1,j} + v(x)_{i-1,j} + v(x)_{i,j+1} + v(x)_{i,j-1} - h^2 \left( v(x)_{i+1,j} - v(x)_{i-1,j} \right) - h^2 \left( P_{i+1,j} - P_{i-1,j} \right).$$x??

---

#### Relaxation Method for Solving Navier-Stokes Equation
Background context: The relaxation method is an iterative approach used to solve the Navier-Stokes equation, similar to methods used for solving Laplace's equation. This technique involves updating values of velocity components by adding corrections (residuals) until convergence is achieved.
Relevant formulas and explanations:
- Update rule:$v(x)_{i,j} = v(x)_{i,j} + r_{i,j}$- Residual calculation:
$$r_{i,j} = \frac{1}{4} \left\{
    v(x)_{i+1,j} + v(x)_{i-1,j} + v(x)_{i,j+1} + v(x)_{i,j-1}
    - h^2 v(x)_{i,j} 
  \right.$$$$- \left. \frac{h}{2} \left[ P_{i+1,j} - P_{i-1,j} \right]
  - \frac{h}{2} \left[ v(y)_{i,j+1} - v(y)_{i,j-1} \right]
  - h^2 [P_{i+1,j} - P_{i-1,j}]
  \right\}
  - v(x)_{i,j}.$$:p What is the update rule for the relaxation method in solving Navier-Stokes equations?
??x
The update rule for the relaxation method involves adding a residual $r_{i,j}$ to the current value of velocity components:
$$v(x)_{i,j} = v(x)_{i,j} + r_{i,j}$$where the residual is calculated using a combination of neighboring values and pressure differences.
x??

---

#### Successive Overrelaxation (SOR)
Background context: SOR is an extension of the standard relaxation method that accelerates convergence by including an amplifying factor $\omega$. The formula for SOR updates the velocity components more aggressively compared to simple relaxation.
Relevant formulas and explanations:
- SOR update rule:
  $$v(x)_{i,j} = v(x)_{i,j} + \omega r_{i,j}$$
- Standard relaxation ($\omega = 1 $) vs. Accelerated convergence ($\omega > 1$):
  - Overrelaxation: $\omega \geq 1 $- Underrelaxation:$\omega < 1$

:p How does the SOR method differ from the standard relaxation method?
??x
The SOR method differs from the standard relaxation method by introducing an amplifying factor $\omega$ that speeds up convergence. The update rule is:
$$v(x)_{i,j} = v(x)_{i,j} + \omega r_{i,j}$$where $\omega > 1$ for overrelaxation, improving the speed of convergence.
x??

---

#### Numerical Solution Implementation
Background context: Implementing numerical solutions to Navier-Stokes equations involves using programming techniques like successive relaxation or SOR. The problem setup includes grid parameters and boundary conditions as specified in the text.
Relevant formulas and explanations:
- Grid setup:$\nu = 1 \, m^2/s $, $\rho = 103 \, kg/m^3 $- Boundary conditions:$ v(x) = 3j/20 (1 - j/40)$,$ v(y) = 0$

:p How would you implement the numerical solution for Navier-Stokes equations in a Python program?
??x
To implement the numerical solution, you can modify or write your own code to solve the Navier-Stokes equation using successive relaxation. The key steps are:
1. Define arrays for velocity components $vx[Nx, Ny]$ and $vy[Nx, Ny]$.
2. Implement grid parameters: $Nx = 400 $, $ Ny = 40 $,$ h = 1$.
3. Set up initial conditions based on given formulas.
4. Use iterative methods like relaxation or SOR to update the velocity components until convergence.

Example pseudocode:
```python
import numpy as np

# Define grid parameters
Nx, Ny = 400, 40
h = 1
omega = 1.2  # Example value for overrelaxation

# Initialize arrays for vx and vy
vx = np.zeros((Nx, Ny))
vy = np.zeros((Ny, Nx))

# Set initial conditions
for j in range(Ny):
    for i in range(Nx):
        if j < 40:
            vx[i, j] = 3 * (1 - j / 40) / 20

# Iterative relaxation or SOR method
tolerance = 1e-6
max_iterations = 1000
for iteration in range(max_iterations):
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            r_x = ...  # Calculate residual for vx
            r_y = ...  # Calculate residual for vy
            
            vx[i, j] += omega * r_x
            vy[j, i] += omega * r_y
    
    if np.linalg.norm(r) < tolerance:
        break

print(f"Number of iterations: {iteration}")
```

x??

---

#### Vorticity Form in Fluid Dynamics
Background context: The vorticity form of the Navier-Stokes equations allows for a more straightforward solution by casting the problem into simpler equations involving stream functions and vorticity. This simplification helps in understanding the flow patterns, especially in 2D flows.
Relevant formulas and explanations:
- Stream function $u(x)$:
  - Velocity components: 
    $$v_x = \frac{\partial u}{\partial y}, \quad v_y = -\frac{\partial u}{\partial x}$$- Vorticity field $ w(x)$:
  - Defined as:
    $$w = \nabla \times v$$:p What is the stream function and how does it relate to velocity components?
??x
The stream function $u(x)$ relates to velocity components in 2D flows through the following relations:
$$v_x = \frac{\partial u}{\partial y}, \quad v_y = -\frac{\partial u}{\partial x}$$

These equations allow us to determine the velocity field from the stream function, which is particularly useful for visualizing streamline patterns.
x??

---

#### Vorticity and Fluid Dynamics
Background context: Vorticity measures how much a fluid's velocity curls or rotates. It helps in understanding flow behavior, especially around objects like beams, by providing local information on angular velocity vectors.
Relevant formulas and explanations:
- Vorticity definition:
$$w = \nabla \times v(x)$$- For 2D flows with no $ z$-component of velocity:
  $$w_z = \left( \frac{\partial v_y}{\partial x} - \frac{\partial v_x}{\partial y} \right)$$:p What is vorticity and how is it related to fluid flow?
??x
Vorticity measures the local rotation or curl of a fluid's velocity. It is defined as the curl of the velocity vector:
$$w = \nabla \times v(x)$$

In 2D flows, this simplifies to:
$$w_z = \frac{\partial v_y}{\partial x} - \frac{\partial v_x}{\partial y}$$
It helps in understanding rotational behavior and is useful for analyzing fluid dynamics around objects like beams.
x??

---

