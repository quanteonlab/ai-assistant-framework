# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 46)

**Starting Chapter:** Chapter 19 General Relativity. 19.1 Einsteins Field Equations

---

#### Einstein's Field Equations
Background context explaining Einstein’s field equations, which describe how matter and energy distort spacetime. These equations are crucial for understanding general relativity.

:p What is the significance of Einstein's field equations in general relativity?
??x
Einstein's field equations are fundamental to general relativity as they connect the geometry of spacetime with its dynamics through the presence of mass, energy, and other forms of stress-energy. These equations describe how matter and energy curve spacetime, leading to gravitational effects.

The equation is given by:
$$R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{\kappa}{c^4} T_{\mu\nu},$$where $ R_{\mu\nu}$ is the Ricci curvature tensor,$ R $is the scalar curvature,$ g_{\mu\nu}$ is the metric tensor,$\Lambda $ is the cosmological constant, and$\frac{\kappa}{c^4} = 8\pi G/c^4 \approx 2.077 \times 10^{-43} N^{-1}$.

The term on the left-hand side describes the curvature of spacetime, while the right-hand side is related to the stress-energy content.
x??

---

#### Metric Tensor
Background context explaining how the metric tensor $g_{\mu\nu}$ defines the path length between two points in spacetime.

:p What does the metric tensor represent and provide an example of its form?
??x
The metric tensor $g_{\mu\nu}$ represents a way to calculate distances in curved spacetime. In general relativity, it provides the necessary information to compute the arclength between two points. For instance, in spherical polar coordinates, the arclength is given by:

$$ds^2 = -dt^2 + d\ell^2 + r^2(d\theta^2 + \sin^2\theta d\phi^2).$$

This example uses the metric tensor to describe the path length for a specific coordinate system.
x??

---

#### Christoffel Symbols
Background context explaining how Christoffel symbols $\Gamma_{\mu}^{\alpha\beta}$ are derived from the metric tensor and used in calculating curvatures.

:p How are Christoffel symbols calculated?
??x
Christoffel symbols are computed using the metric tensor $g_{\mu\nu}$. The formula to calculate them is:
$$\Gamma_{\mu}^{\alpha\beta} = \frac{1}{2}g^{\alpha\lambda}(\partial_\lambda g_{\mu\beta} + \partial_\beta g_{\mu\lambda} - \partial_\mu g_{\lambda\beta}).$$

This formula involves summing over repeated indices and using the inverse metric tensor $g^{\alpha\lambda}$.

Example:
Given a metric tensor, compute the Christoffel symbols for it. For example, in spherical coordinates, you would use this formula to find specific Christoffel symbols like $\Gamma_{\theta\phi}^\theta $ or$\Gamma_{\theta\phi}^\phi$.
x??

---

#### Ricci Curvature Tensor
Background context explaining the Ricci curvature tensor $R_{\mu\nu}$, which is derived from the Christoffel symbols and provides a measure of spacetime curvature.

:p What is the formula for calculating the Ricci curvature tensor?
??x
The Ricci curvature tensor $R_{\mu\nu}$ can be calculated using the Christoffel symbols. The formula is:
$$R_{\mu\nu} = \partial_\nu \Gamma^\alpha_{\mu\alpha} - \partial_\mu \Gamma^\alpha_{\nu\alpha} + \Gamma^\alpha_{\beta\mu}\Gamma^\beta_{\alpha\nu} - \Gamma^\alpha_{\beta\nu}\Gamma^\beta_{\alpha\mu}.$$

This tensor is a contracted form of the Riemann curvature tensor and provides a measure of how spacetime curves.

Example:
Given Christoffel symbols, compute $R_{\mu\nu}$ for specific coordinates.
x??

---

#### Scalar Curvature
Background context explaining the scalar curvature $R$, which summarizes the Ricci curvature tensor into a single value.

:p How is the scalar curvature defined?
??x
The scalar curvature $R$ is derived from the Ricci curvature tensor and can be calculated as:
$$R = g^{\mu\nu}R_{\mu\nu}.$$

This formula sums over all components of the Ricci tensor using the inverse metric tensor.

Example:
Given a specific Ricci tensor, compute the scalar curvature for that spacetime.
x??

---

#### Stress-Energy Tensor
Background context explaining the stress-energy tensor $T_{\mu\nu}$, which describes the source of spacetime curvature and is related to matter and energy distribution.

:p What components of the stress-energy tensor are relevant in general relativity?
??x
In general relativity, the stress-energy tensor $T_{\mu\nu}$ has several important components. The time-time component represents relativistic energy density due to mass and electromagnetic fields:
$$T_{00} = \frac{\rho_E}{c^2} + \frac{1}{2}\left(\frac{1}{\epsilon_0} E^2 + \frac{1}{\mu_0} B^2\right),$$where $\rho_E $ is the energy density, and$E $ and$B$ are electric and magnetic fields respectively.

Other components relate to stress (pressure in a specific direction) and shear stress due to momentum flux across surfaces.
x??

---

#### Geodesic Equation
Background context explaining the geodesic equation, which describes the motion of freely falling particles in spacetime.

:p What is the geodesic equation?
??x
The geodesic equation describes how massive particles move in curved spacetime. It is given by:
$$\frac{d^2 x^\mu}{ds^2} = -\Gamma_{\mu}^{\alpha\beta} \frac{dx^\alpha}{ds} \frac{dx^\beta}{ds},$$where $ s $ is the scalar proper time and $\Gamma_{\mu}^{\alpha\beta}$ are Christoffel symbols.

For a specific coordinate system, this equation can be written explicitly using an explicit time coordinate:
$$\frac{d^2 x^\mu}{dt^2} = -\Gamma_{\mu}^{\alpha\beta} \frac{dx^\alpha}{dt} \frac{dx^\beta}{dt} + \Gamma_0^{\alpha\beta} \frac{dx^\alpha}{dt} \frac{dx^\beta}{dt}.$$

Example:
Solve the geodesic equation for a free-falling particle in spherical coordinates.
x??

---

#### Geodesic Equation and its Applications

General context: The geodesic equation is a fundamental concept in General Relativity, describing the shortest path that a particle can take in spacetime. This form of the geodesic equation is used for numerical computations.

:p What does the geodesic equation describe in terms of acceleration and force?
??x
The geodesic equation describes the acceleration ($\frac{d^2 x^\mu}{ds^2}$) of a test particle moving through spacetime. It can be analogized to Newton’s second law,$ F = ma$, but instead of force, it incorporates the geometry of spacetime as described by Christoffel symbols (Γ). The equation is given by:

$$\frac{d^2 x^\mu}{ds^2} + \Gamma^\mu_{\alpha \beta} \frac{dx^\alpha}{ds}\frac{dx^\beta}{ds} = 0$$

For non-relativistic motion, the terms quadratic and cubic in velocity can be neglected, leading to a simpler form:
$$\frac{d^2 x_i}{dt^2} \approx - \Gamma^i_{00}$$

This simplified form is similar to Galileo's hypothesis that all particles have the same acceleration in a uniform gravitational field.

:x??
---
#### Calculating the Riemann and Ricci Tensors

Background context: The Riemann tensor quantifies the curvature of spacetime, while the Ricci tensor is derived from it by contraction. These tensors are essential for understanding the geometry of curved spacetime described by General Relativity.

:p How can we calculate the Riemann tensor using geodesics?
??x
To find the Riemann tensor, consider two infinitesimally close geodesics and their relative acceleration. The relative acceleration is given by:
$$\frac{d^2 n^\alpha}{d \tau^2} = 0$$

This derivative acts on the basis vectors, requiring knowledge of the Christoffel symbols (Γ). By substituting into the expression for the Riemann tensor, we get:
$$

R^\alpha_{\mu\nu\beta} = \frac{\partial \Gamma^\alpha_{\nu\beta}}{\partial x^\mu} - \frac{\partial \Gamma^\alpha_{\mu\beta}}{\partial x^\nu} + \Gamma^\alpha_{\gamma\beta}\Gamma^\gamma_{\mu\nu} - \Gamma^\alpha_{\gamma\beta}\Gamma^\gamma_{\mu\nu}$$

This can be simplified as:
$$

R^\alpha_{\mu\nu\beta} = \frac{\partial \Gamma^\alpha_{\nu\beta}}{\partial x^\mu} - \frac{\partial \Gamma^\alpha_{\mu\beta}}{\partial x^\nu} + \Gamma^\alpha_{\gamma\beta}\Gamma^{\gamma}_{\mu\nu} - \Gamma^\alpha_{\nu\gamma}\Gamma^{\gamma}_{\mu\beta}$$:p How do we extract the Ricci tensor from the Riemann tensor?
??x
The Ricci tensor is obtained by contracting the Riemann tensor:
$$

R_{\mu\nu} = R^\alpha_{\mu\alpha\nu}$$

In simpler terms, it sums over one of the upper and lower indices. The Ricci scalar $R $ can then be found as a contraction of the Ricci tensor with the metric tensor$g^{\mu\nu}$:

$$R = g^{\mu\nu} R_{\mu\nu}$$:p What is an example of a Schwarzschild solution and how do we approach calculating tensors for it?
??x
The Schwarzschild metric describes the geometry outside a spherical mass. For this case, we can use SymPy or similar symbolic manipulation tools to calculate Christoffel symbols, Riemann tensor, and Ricci tensor.

```python
# Pseudocode Example

import sympy as sp

def calculate_tensors():
    # Define spacetime coordinates
    t, r, theta, phi = sp.symbols('t r theta phi')
    
    # Schwarzschild metric components
    gtt, grr, gthth, gphiphi = (1 - 2*GM/c**2*r), -(1 - 2*GM/c**2/r)**(-1), -r**2, -r**2 * sp.sin(theta)**2
    
    # Define metric tensor
    g = [[gtt, 0, 0, 0],
         [0, grr, 0, 0],
         [0, 0, gthth, 0],
         [0, 0, 0, gphiphi]]
    
    # Calculate Christoffel symbols
    christoffel_symbols = calculate_christoffel(g)
    
    # Calculate Riemann tensor
    riemann_tensor = calculate_riemann(christoffel_symbols)
    
    # Contract to get Ricci tensor and scalar
    ricci_tensor, ricci_scalar = contract_tensors(riemann_tensor, g)

```
:x??
---
#### Event Horizons

Background context: In the Schwarzschild metric, the event horizon is a boundary in spacetime where distances become singular. This singularity can be understood by analyzing the proper distance $ds$.

:p What defines an event horizon and how do we find it for a black hole with mass M?
??x
The event horizon of a black hole is defined as the radius $r_h = 2GM/c^2$ where distances become singular. This can be found by setting up the Schwarzschild metric:

$$ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right) dt^2 + \frac{dr^2}{1 - \frac{2GM}{c^2 r}} + r^2 d\theta^2 + r^2 \sin^2(\theta) d\phi^2$$

To find the event horizon, we set $1 - \frac{2GM}{c^2 r} = 0$:

$$r_h = \frac{2GM}{c^2}$$:p How do we verify an approximate solution to the deflection angle of light near a massive object?
??x
To verify an approximate solution for the deflection angle, consider the nonlinear ODE:
$$\left(\frac{du}{d\phi}\right)^2 = 1 - u^2 - \frac{2M}{R} (1 - u^3)$$where $ u = R/r$. The solution can be verified by comparing it to the known approximate formula for the deflection angle:

$$\phi \approx \frac{4GM}{c^2 r}$$:p How do we numerically solve the ODE for light deflection and compare with an analytic approximation?
??x
To solve the ODE numerically, we can use a simple Euler method or Runge-Kutta methods. Given initial conditions $u(\phi = 0) \approx 1/R $ and$\frac{du}{d\phi} \approx 0 $, we can integrate to find $ r(\phi)$.

Here is a pseudocode example:

```python
def solve_ode(phi_max):
    # Initial conditions
    u_initial = 1 / R
    du_dphi_initial = 1e-6
    
    # ODE solver setup
    step_size = phi_max / 10000
    current_phi = 0
    current_u = u_initial
    current_du_dphi = du_dphi_initial
    
    while current_phi < phi_max:
        # Euler method for simplicity
        current_du_dphi += (1 - current_u**2 - (2*M/R)*(1 - current_u**3)) * step_size
        current_u += current_du_dphi * step_size
        current_phi += step_size
    
    return r(current_phi)
```

By comparing the numerical solution with the analytic approximation, we can validate our methods.

:x??
---
#### Gravitational Lensing

Background context: Gravitational lensing is a phenomenon where light from distant objects is bent due to the gravitational field of massive objects. This effect can be modeled using geodesic equations in curved spacetime.

:p How do we model the deflection of light around a massive object like a star?
??x
To model the deflection of light around a star, we use the Schwarzschild metric with appropriate transformations. The geodesic equation for the inverse radial distance $u = 1/r$ is:

$$\frac{d^2 u}{d\phi^2} = -3GMu^2 + u$$:p How do we solve this ODE numerically and plot the trajectory of light?
??x
To solve the ODE for the deflection angle, we can use a numerical solver. Given initial conditions $u(\phi=0) \approx 1/R $ and$du/d\phi = 0 $, we can integrate to find$ r(\phi)$.

Here is an example in Python:

```python
# LensGravity.py

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lensing_ode(u, phi, M, R):
    # u = 1/r, G=1 for simplicity
    du_dphi = -3 * M * u**2 + u
    return [du_dphi]

M = 28 * 1.989e30   # Mass of the star in kg
R = 10**6           # Initial distance from the star

# Initial conditions
u_initial = 1 / R
phi_initial = 0

# Time span (in units of phi)
time_span = np.linspace(0, np.pi, 1000)

solution = odeint(lensing_ode, [u_initial], time_span, args=(M/R,))

# Convert to r and plot the trajectory
r = 1 / solution
x = r * np.sin(time_span)
y = -r * np.cos(time_span)

plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gravitational Lensing Trajectory')
plt.show()
```

This code numerically integrates the ODE and plots the trajectory of light deflected by a star.

:x??
--- 

These flashcards cover key concepts in General Relativity, including geodesic equations, tensor calculations, event horizons, and gravitational lensing. Each card provides context, formulas, and examples to aid understanding.

#### Plotting Effective Potential
Background context explaining how to plot the effective potential $V_{\text{eff}}(r')$ and its significance. The formula given is:
$$V_{\text{eff}}(r') = -\frac{G M}{r'} + \frac{\ell'^2}{2 r'^2} - \frac{G M \ell'^2}{r'^3}$$where $ G $is the gravitational constant,$\ell'$ is the angular momentum per unit rest mass, and $M$ is the star's mass.

:p Plot $V_{\text{eff}}(r')$ versus $r'$ for $\ell = 4.3$.
??x
To plot the effective potential, we substitute $\ell' = 4.3$ into the formula:
$$V_{\text{eff}}(r') = -\frac{G M}{r'} + \frac{(4.3)^2}{2 r'^2} - \frac{G M (4.3)^2}{r'^3}$$

We can use Python to plot this function:

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 1  # gravitational constant, set to 1 for simplicity
M = 1  # mass of the star, also set to 1
l_prime = 4.3

r_prime_values = np.linspace(0.1, 50, 400)  # range from a small value to large values of r'

# Calculate Veff
V_eff = -G * M / r_prime_values + (l_prime**2) / (2 * r_prime_values**2) - G * M * l_prime**2 / r_prime_values**3

# Plotting the effective potential
plt.figure()
plt.plot(r_prime_values, V_eff)
plt.title('Effective Potential for $\ell = 4.3$')
plt.xlabel('$r\'$')
plt.ylabel('$V_{\text{eff}}(r\')$')
plt.grid(True)
plt.show()
```
x??

---

#### Effect of Energy on Orbits
Background context explaining how the effective potential affects orbits and the significance of energy in determining orbit characteristics.

:p Describe how the orbits within this potential change with different energies.
??x
The orbits are influenced by the total energy $E $, which is a sum of kinetic and potential terms. The effective potential determines the stable, unstable, and circular orbits based on the value of $ E$.

- For low $E$: Orbits can be highly elliptical or even spiral in.
- For moderate $E$: Stable orbits (elliptical) are possible.
- For high $E$: Orbits tend to be hyperbolic or parabolic.

The specific details depend on the balance between the kinetic and potential energies. The effective potential has maxima and minima, which indicate different types of orbits:
- Maxima: Unstable orbits (small perturbations lead to rapid divergence).
- Minima: Stable orbits (perturbations lead to oscillatory motion around the equilibrium).

The circular orbit exists at a specific radius where $dV_{\text{eff}}/dr = 0 $ and$d^2 V_{\text{eff}}/dr^2 > 0$.
x??

---

#### Finding Maximum and Minimum of Effective Potential
Background context explaining how to find the critical points of the effective potential.

:p At what values of $r'$ does the effective potential have a maximum and minimum?
??x
To find the maxima and minima, we take the first derivative of $V_{\text{eff}}(r')$ with respect to $r'$ and set it to zero:
$$\frac{d V_{\text{eff}}}{d r'} = -\frac{G M}{r'^2} + \frac{\ell'^4}{r'^3} + 3 G M \ell'^2 / r'^4 = 0$$

Solving this equation numerically for specific values of $G $,$ M $, and$\ell'$ gives the critical points.

For example, using Python to solve it:

```python
from scipy.optimize import fsolve

# Define Veff and its derivative
def V_eff_derivative(r_prime):
    return -G * M / r_prime**2 + (l_prime**4) / r_prime**3 + 3 * G * M * l_prime**2 / r_prime**4

# Initial guess for the root
initial_guess = [1, 50]

# Find roots
roots = fsolve(V_eff_derivative, initial_guess)

print(f"Critical points: {roots}")
```
x??

---

#### Circular Orbit Existence
Background context explaining how to determine if a circular orbit exists and its stability.

:p At what value of $r'$ does a circular orbit exist?
??x
To find the radius for a circular orbit, we set the effective potential's first derivative to zero:
$$\frac{d V_{\text{eff}}}{d r'} = -\frac{G M}{r'^2} + \frac{\ell'^4}{r'^3} + 3 G M \ell'^2 / r'^4 = 0$$

And the second derivative should be positive to ensure stability:
$$\frac{d^2 V_{\text{eff}}}{d r'^2} > 0$$

For $\ell' = 4.3$, solving these equations numerically will give us the radius of a circular orbit.

```python
from scipy.optimize import fsolve

def V_eff_derivative(r_prime):
    return -G * M / r_prime**2 + (l_prime**4) / r_prime**3 + 3 * G * M * l_prime**2 / r_prime**4

# Initial guess for the root
initial_guess = [1]

# Find roots
roots = fsolve(V_eff_derivative, initial_guess)

print(f"Radius for circular orbit: {roots[0]}")
```
x??

---

#### Range of $r'$ Values
Background context explaining how to determine the range of values that occur for $\ell' = 4.3$.

:p Determine the range of $r'$ values that occur for $\ell' = 4.3$.
??x
The range of $r'$ values can be determined by analyzing the behavior of the effective potential. We plot the effective potential and identify regions where it is positive, indicating stable orbits.

From Figure 19.4, we see that the circular orbits occur at approximately $r' \approx 20$.

For a more precise range, we solve for roots numerically:

```python
def V_eff(r_prime):
    return -G * M / r_prime + (l_prime**2) / (2 * r_prime**2) - G * M * l_prime**2 / r_prime**3

# Find roots where V_eff = 0
roots = fsolve(V_eff, [1, 50])

print(f"Range of stable orbits: {roots}")
```
x??

---

#### Numerical Exploration of Orbits
Background context explaining how to use energy conservation and the ODE derived from it to numerically explore orbits.

:p Use your ODE solver to explore various orbits corresponding to different initial conditions and energies.
??x
To explore orbits, we use the equation for angular momentum $u' = M / r'$ and solve the second-order differential equation:
$$\frac{d^2 u'}{d \phi^2} = -u' + \frac{G M}{\ell'^2} (1 + 3 G M u')$$

The initial conditions are derived from the energy integral. We use a numerical solver to plot orbits.

```python
def orbit_equation(phi, y, G, M, l_prime):
    u_prime = y[0]
    d2u_dphi2 = -u_prime + (G * M) / (l_prime**2) * (1 + 3 * G * M * u_prime)
    return [d2u_dphi2]

# Initial conditions
E = ...  # energy value, e.g., from Figure 19.4

# Convert to initial condition for y0
y0 = [l_prime**2 / (2 * E)]

# Solve ODE
from scipy.integrate import solve_ivp
sol = solve_ivp(orbit_equation, [phi_min, phi_max], y0, args=(G, M, l_prime), t_eval=np.linspace(phi_min, phi_max, 1000))

# Plotting the orbit
plt.figure()
plt.plot(sol.y[0] * l_prime / (2 * E), sol.t)
plt.title('Orbit for given energy')
plt.xlabel('$u\'$')
plt.ylabel('$\phi$')
plt.grid(True)
plt.show()
```
x??

---

#### Investigating Angular Momentum and Orbits
Background context explaining how changes in angular momentum affect orbits.

:p Investigate the effect of gradually decreasing the angular momentum $\ell'$.
??x
Decreasing the angular momentum $\ell'$ affects the shape and stability of the orbit. As $\ell'$ decreases, the circular orbit radius increases because the effective potential barrier becomes less pronounced.

For example, if we start with a specific energy $E $ corresponding to an initial$\ell' = 4.3 $, we can vary $\ell'$ and observe how orbits change:

```python
def plot_orbits_multiple_angular_momentum(G, M, E_values, l_prime_values):
    for l_prime in l_prime_values:
        y0 = [l_prime**2 / (2 * E)]
        sol = solve_ivp(orbit_equation, [phi_min, phi_max], y0, args=(G, M, l_prime), t_eval=np.linspace(phi_min, phi_max, 1000))
        plt.plot(sol.y[0] * l_prime / (2 * E), sol.t)
    plt.title('Orbits for different angular momenta')
    plt.xlabel('$u\'$')
    plt.ylabel('$\phi$')
    plt.grid(True)
    plt.show()

l_prime_values = [4.3, 3.5, 2.8]
plot_orbits_multiple_angular_momentum(G, M, E, l_prime_values)
```
x??

---

#### Minimum Effective Potential and Orbits
Background context explaining how to find orbits corresponding to the minimum of the effective potential.

:p Choose an energy that corresponds to the minimum in the effective potential and plot nearby orbits. Examine the sensitivity of these orbits to the choice of initial conditions.
??x
To find orbits near the minimum of the effective potential, we first identify the radius where $V_{\text{eff}}$ has a minimum. This is typically done by solving:
$$\frac{d V_{\text{eff}}}{d r'} = 0$$

For a specific $\ell'$, this gives us a critical point. The minimum of the effective potential can be numerically found and used to determine stable orbits.

```python
def find_minimal_energy(G, M, l_prime):
    # Solve for the radius where V_eff has a minimum
    def V_eff_derivative(r_prime):
        return -G * M / r_prime**2 + (l_prime**4) / r_prime**3 + 3 * G * M * l_prime**2 / r_prime**4

    initial_guess = [10]
    minimal_radius = fsolve(V_eff_derivative, initial_guess)[0]

    # Calculate the energy corresponding to this radius
    V_min = -G * M / minimal_radius + (l_prime**2) / (2 * minimal_radius**2) - G * M * l_prime**2 / minimal_radius**3

    return minimal_radius, V_min

minimal_radius, E_min = find_minimal_energy(G, M, l_prime)

# Plot nearby orbits
def plot_nearby_orbits(E, r0, l_prime):
    y0 = [l_prime**2 / (2 * E)]
    sol = solve_ivp(orbit_equation, [phi_min, phi_max], y0, args=(G, M, l_prime), t_eval=np.linspace(phi_min, phi_max, 1000))
    plt.plot(sol.y[0] * l_prime / (2 * E), sol.t)
plot_nearby_orbits(E_min, minimal_radius, l_prime)

plt.title('Nearby orbits for minimal energy')
plt.xlabel('$u\'$')
plt.ylabel('$\phi$')
plt.grid(True)
plt.show()
```
x??

