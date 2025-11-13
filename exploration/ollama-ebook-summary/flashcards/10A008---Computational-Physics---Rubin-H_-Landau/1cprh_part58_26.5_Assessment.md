# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 58)

**Starting Chapter:** 26.5 Assessment and Exploration

---

#### Vorticity and Stream Function Relationship
The vorticity $\boldsymbol{\omega}$ of a flow is related to the stream function $u$ by:
$$\boldsymbol{\omega} = \nabla \times \mathbf{v} = \nabla \times (\nabla \times \mathbf{u}) = \nabla(\nabla \cdot \mathbf{u}) - \nabla^2\mathbf{u},$$where $\mathbf{u}$ is the velocity field. For flows with only a z-component that does not vary with $ z $, and no sources, the divergence of the velocity $\nabla \cdot \mathbf{u} = 0$. This leads to:
$$\nabla^2 u = -\boldsymbol{\omega}.$$

This equation is analogous to Poisson's equation in electrostatics but now describes the relationship between vorticity and stream function.

:p What does the vorticity form of the Navier–Stokes equation describe?
??x
The vorticity form of the Navier–Stokes equation relates the vorticity $\boldsymbol{\omega}$ to the stream function $u$. It shows how changes in the velocity field can be linked through vorticity and stream function. The key relationship is given by:
$$\nabla^2 u = -\boldsymbol{\omega}.$$

This equation describes the coupling between the vorticity and the stream function, making it easier to analyze certain types of flows.

x??

---

#### Vorticity Form of Navier–Stokes Equation
Starting from the velocity form, taking the curl yields:
$$\nu \nabla^2 \boldsymbol{\omega} = [(\nabla \times \mathbf{u}) \cdot \nabla] \boldsymbol{\omega}.$$

This equation is coupled with the Poisson-like equation for $u$:
$$\nabla^2 u = -\boldsymbol{\omega}.$$

In 2D, where $\mathbf{u}$ and $\boldsymbol{\omega}$ have only z-components:
$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = -\omega,$$
$$\nu \left( \frac{\partial^2 \omega}{\partial x^2} + \frac{\partial^2 \omega}{\partial y^2} \right) = \frac{\partial u}{\partial y} \frac{\partial \omega}{\partial x} - \frac{\partial u}{\partial x} \frac{\partial \omega}{\partial y}.$$:p What are the two simultaneous nonlinear elliptic PDEs that describe vorticity and stream function in 2D?
??x
The two simultaneous nonlinear elliptic PDEs for $u $(stream function) and $\boldsymbol{\omega}$(vorticity) in 2D are:
$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = -\omega,$$and$$\nu \left( \frac{\partial^2 \omega}{\partial x^2} + \frac{\partial^2 \omega}{\partial y^2} \right) = \frac{\partial u}{\partial y} \frac{\partial \omega}{\partial x} - \frac{\partial u}{\partial x} \frac{\partial \omega}{\partial y}.$$

These equations describe the relationship between vorticity and stream function, making it easier to analyze certain types of flows.

x??

---

#### Vorticity Difference Equation on a Grid
The difference equation for $u $ and$\boldsymbol{\omega}$ on an $Nx \times Ny$ grid is derived using central differences:
$$u_{i,j} = \frac{1}{4}(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} + h^2 \omega_{i,j}),$$and$$\omega_{i,j} = \frac{1}{4}(\omega_{i+1,j} + \omega_{i-1,j} + \omega_{i,j+1} + \omega_{i,j-1}) - R \frac{1}{16} \left\{ [u_{i,j+1} - u_{i,j-1}] \times [\omega_{i+1,j} - \omega_{i-1,j}] - [u_{i+1,j} - u_{i-1,j}] \times [\omega_{i,j+1} - \omega_{i,j-1}] \right\},$$where$$

R = \frac{1}{\nu V_0 h / \nu}.$$:p How are $ u $ and $\boldsymbol{\omega}$ discretized on a grid?
??x
On a grid, the stream function $u $ and vorticity$\boldsymbol{\omega}$ are discretized using central differences. For example:
$$u_{i,j} = \frac{1}{4}(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} + h^2 \omega_{i,j}),$$and$$\omega_{i,j} = \frac{1}{4}(\omega_{i+1,j} + \omega_{i-1,j} + \omega_{i,j+1} + \omega_{i,j-1}) - R \frac{1}{16} \left\{ [u_{i,j+1} - u_{i,j-1}] \times [\omega_{i+1,j} - \omega_{i-1,j}] - [u_{i+1,j} - u_{i-1,j}] \times [\omega_{i,j+1} - \omega_{i,j-1}] \right\}.$$

These equations allow the computation of $u $ and$\boldsymbol{\omega}$ at each grid point based on their neighbors. The parameter $R$ is a relaxation factor that depends on the Reynolds number, grid spacing, and velocity scale.

x??

---

#### Boundary Conditions Implementation
Implementing boundary conditions for stream function $u $ and vorticity$\boldsymbol{\omega}$ requires careful handling:
- **Inlet (F):**$\frac{\partial u}{\partial x} = 0 $, $\omega = 0 $- **Surface (G):**$\frac{\partial u}{\partial y} = V_0 $, $\omega = 0 $- **Outlet (H):**$\frac{\partial u}{\partial x} = 0 $, $\frac{\partial \omega}{\partial x} = 0 $:p How are the boundary conditions implemented for $ u $and$\boldsymbol{\omega}$?
??x
Boundary conditions for stream function $u $ and vorticity$\boldsymbol{\omega}$ are implemented as follows:

- **Inlet (F):** No-slip condition, no flow through the wall:
  - $\frac{\partial u}{\partial x} = 0 $-$\omega = 0$- **Surface (G):** Inflow boundary condition for velocity:
  -$\frac{\partial u}{\partial y} = V_0 $-$\omega = 0$- **Outlet (H):** Far-field condition, assuming no flow out of the domain:
  -$\frac{\partial u}{\partial x} = 0 $-$\frac{\partial \omega}{\partial x} = 0$ These conditions ensure that the flow behavior at boundaries is consistent with physical expectations.

x??

---

#### Relaxation Algorithm for Convergence
The relaxation algorithm iteratively solves for $u $ and$\boldsymbol{\omega}$ until convergence:
- Print iteration number and values of $u$ upstream, above, and downstream from the beam.
- Determine the number of iterations needed to achieve three-place convergence with successive relaxation ($\omega = 1$).

:p How is the relaxation algorithm used to find convergence?
??x
The relaxation algorithm iteratively updates $u $ and$\boldsymbol{\omega}$ until a converged solution is reached. For example, if we use $\omega = 1$ for successive relaxation:

- Print out the iteration number and values of $u$:
  - Upstream from the beam
  - Above the beam
  - Downstream from the beam

- Determine the number of iterations needed to achieve three-place convergence.

This process ensures that the solution reaches a stable state where small changes in the variables do not significantly alter their values.

x??

---

#### Simulation Parameters and Boundary Conditions
Simulate with $L = 8h $, $ H = h $,$ R = 0.1 $,$ V_0 = 1 $. Use a grid of$ Nx = 24 $ and $ Ny = 70$.

:p What are the simulation parameters for the vorticity form of the Navier–Stokes equation?
??x
The simulation parameters for the vorticity form of the Navier–Stokes equation include:
- Beam size: $L = 8h $- Height:$ H = h $- Reynolds number:$ R = 0.1 $- Intake velocity:$ V_0 = 1 $The grid dimensions are set to$ Nx = 24 $and$ Ny = 70$.

These parameters help in setting up the initial conditions for the simulation, ensuring that the flow behavior is consistent with the given physical setup.

x??

---

#### Standing Wave Development
Change the beam's horizontal placement to observe the development of a standing wave. This may require increasing the size of the simulation volume to see all boundary effects clearly.

:p How does changing the beam’s horizontal placement affect the flow?
??x
Changing the beam's horizontal placement can alter the flow pattern, leading to the development of standing waves behind the beam. By observing the flow, one can notice that as the beam moves, the undisturbed current entering from the left develops into a stable wave structure.

This effect is more pronounced when the simulation volume is large enough to capture the full extent of boundary interactions and the developing standing wave.

x??

---

#### Surface Plots for Stream Function and Vorticity
Make surface plots including contours of $u $(stream function) and $\boldsymbol{\omega}$ (vorticity).

:p How can one create surface plots to visualize stream function $u $ and vorticity$\boldsymbol{\omega}$?
??x
Surface plots for the stream function $u $ and vorticity$\boldsymbol{\omega}$ can be created using contour plotting techniques. These plots help in visualizing the flow patterns:

- **Stream Function $u$:** Contours show regions of high and low velocity.
- **Vorticity $\boldsymbol{\omega}$:** Contours indicate areas where vortices form.

By creating these surface plots, one can gain insights into how the fluid moves around the beam and the formation of vortices behind it.

x??

---

#### Region for a Fish to Rest
Determine if there is a region where a big fish could rest behind the beam based on the simulation results.

:p Is there a region where a big fish could rest behind the beam?
??x
Based on the simulation results, one can identify regions of low velocity or areas with minimal flow disturbance. These regions might provide a suitable place for a big fish to rest. By analyzing the stream function $u $ and vorticity$\boldsymbol{\omega}$, you can pinpoint such locations.

For example, behind the beam where the flow is calm, there may be a region where a fish could find shelter.

x??

---

#### Visualization of Fluid Velocity
Make several visualizations showing the fluid velocity throughout the simulation region.

:p How can one visualize the fluid velocity in the simulation region?
??x
To visualize the fluid velocity in the simulation region:

1. **Streamlines:** Show streamlines to indicate the path of fluid particles.
2. **Vector Fields:** Use vector fields to represent both $u $ and$v$ components of the velocity.

By creating these visualizations, you can gain a comprehensive understanding of how the flow behaves in different parts of the simulation domain.

x??

--- 

These flashcards cover key concepts from the provided text, focusing on the relationship between stream function and vorticity, numerical methods for solving PDEs, boundary conditions, and visualization techniques. Each card includes context, relevant formulas, and explanations to aid understanding. --- 

Note: The code snippets in this example are simplified representations of what one might see or use as guidance. Actual implementations would depend on the specific programming environment and libraries used (e.g., NumPy for Python).