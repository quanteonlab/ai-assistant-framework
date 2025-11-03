# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 100)

**Starting Chapter:** 16.4.3.1 Hard Disk Scattering

---

#### Chaotic Pendulum Analysis
Background context: The chaotic behavior of a driven pendulum, particularly when it has one-, three-, and five-cycle structures, involves analyzing its Fourier components to understand the periodic behaviors and transitions between them. The natural frequency \(\omega_0\) and driving frequency \(\omega\) play crucial roles in determining these behaviors.

:p What is the goal of this analysis?
??x
The goal is to identify the Fourier components present in different cycle structures (one-, three-, and five-cycle) of a chaotic pendulum, deduce major frequencies, and compare results with theoretical expectations.
x??

---

#### Wavelet Analysis for Chaotic Pendulum
Background context: Wavelets are more suitable than traditional Fourier analysis for signals that have transient or non-stationary components. For the chaotic pendulum, wavelets can provide better insights into the temporal sequence of various frequency components.

:p How does wavelet analysis differ from Fourier analysis in this context?
??x
Wavelet analysis is preferred over Fourier analysis because it provides localized information about both time and frequency, making it more effective for analyzing transient behaviors like those seen in chaotic pendulum dynamics. It can help discern the temporal sequence of various components.
x??

---

#### Double Pendulum Dynamics
Background context: The double pendulum system has two coupled degrees of freedom and can exhibit complex chaotic behavior even without external driving forces. The Lagrangian formulation is used to derive its equations of motion.

:p What are the key differences between a simple pendulum and a double pendulum?
??x
Key differences include:
- A double pendulum has two coupled degrees of freedom (\(\theta_1\) and \(\theta_2\)).
- It can exhibit chaotic behavior, even without external driving forces.
- The Lagrangian formulation accounts for the coupling between the two motions.
x??

---

#### Bifurcation in Double Pendulum
Background context: Bifurcations in the double pendulum system can be observed through phase space trajectories and bifurcation diagrams. These show how the number of dominant frequencies changes with varying parameters, leading to fractal structures.

:p What does a bifurcation diagram for the double pendulum illustrate?
??x
A bifurcation diagram illustrates how the instantaneous angular velocity of the lower pendulum changes as a function of a parameter (e.g., mass of the upper pendulum). It shows the transition between different types of motion and can reveal fractal structures due to the complex dynamics.
x??

---

#### Billiards in Dynamical Systems
Background context: A mathematical billiard involves a particle moving freely until it hits a boundary wall, where it undergoes specular reflection. Different shapes of confining tables (square, circular, Sinai, stadium) exhibit different behaviors and can display chaotic motion.

:p What are the key features of a billiard system?
??x
Key features include:
- A particle moves in straight lines between collisions.
- Collisions with boundaries result in specular reflections.
- The shape of the boundary table affects the dynamics (e.g., periodic or chaotic behavior).
- Billiards are Hamiltonian systems with no energy loss and can display chaos.
x??

---

#### Multiple Scattering Centers
Background context: The scattering of a projectile from a force center is typically continuous. However, when multiple internal scatterings occur due to complex potential structures, it leads to complex behaviors.

:p What factors influence the behavior of projectiles in multiple scattering centers?
??x
Factors include:
- Internal structure of the potential.
- Multiple internal scatterings leading to complex dynamics.
- Transition from continuous processes to discrete interactions.
x??

---

#### Hard Disk Scattering
Background context: The problem involves simulating point particles scattering elastically from stationary hard disks on a 2D billiard table. Different configurations of disks (one, two, or three) create periodic and possibly chaotic behaviors in particle trajectories. The goal is to explore how these disks can lead to complex dynamical systems.
:p What modifications are needed to simulate the scattering of particles from one, two, or three hard disks?
??x
The task involves modifying an existing program that simulates scattering from a four-peaked Gaussian potential (Section 13.3.1) to use point particles scattered elastically by hard disks. The key is to replace the Gaussian potential with disk-based scattering.

To handle this numerically, you can model the disks as infinite potentials or very large finite barriers. This requires implementing collision detection logic where a particle's trajectory changes upon hitting a disk boundary. You will need to plot trajectories showing both typical and unusual behaviors, including those resulting from multiple scatterings.
```python
# Pseudocode for modifying the program

def update_position_and_velocity(x, y, vx, vy):
    # Collision detection with disks
    if is_collision_with_disk(x, y):
        adjust_vx_vy(vx, vy)

def is_collision_with_disk(x, y):
    # Check if (x, y) is within a disk's boundary
    for disk in disks:
        if distance((x, y), disk.center) < R:
            return True
    return False

def adjust_vx_vy(vx, vy):
    # Change velocity components after collision
    vx = -vx  # Reflecting the x-component of velocity
    vy = -vy  # Reflecting the y-component of velocity

# Example usage in a simulation loop
for t in range(num_steps):
    update_position_and_velocity(x[t], y[t], vx[t], vy[t])
```
x??

---

#### Trajectory and Phase Space Plots
Background context: After setting up the disk scattering model, you need to generate trajectories for particles scattered by one, two, or three disks. This includes plotting both position-time and phase space plots to visualize the particle motion.

Phase space plots specifically are crucial as they can reveal underlying chaotic behaviors.
:p What should be plotted for different scattering scenarios?
??x
For each configuration (one, two, or three disks), you need to plot several trajectories of particles scattered from these disks. These include both typical and unusual trajectories, particularly those involving back-angle scattering where the particle bounces multiple times before escaping.

Additionally, you should create phase space plots using position-velocity pairs to analyze the motion further:
```python
# Example code for plotting trajectories

import matplotlib.pyplot as plt

def plot_trajectories(x_list, y_list):
    fig, ax = plt.subplots()
    ax.plot(x_list, y_list)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    plt.show()

# Example usage
plot_trajectories(x_positions, y_positions)

# Example code for phase space plots

def plot_phase_space(x_list, v_x_list):
    fig, ax = plt.subplots()
    ax.plot(x_list, v_x_list)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('v_x(t)')
    plt.show()

plot_phase_space(x_positions, vx_list)
```
x??

---

#### Discontinuities in Scattering Angle
Background context: You need to determine the scattering angle as a function of impact parameter and then plot the discontinuities in this relationship. This involves detailed analysis of how particles scatter off disks and the resulting changes in their trajectory angles.
:p What is the task for determining and plotting the discontinuities in d𝜃/db?
??x
The task requires computing the scattering angle \(\theta\) as a function of the impact parameter \(b\), where \(b = y_0 - y_c\) (the vertical distance from the center of the disk to the initial position of the particle). You need to determine this for each particle that exits the interaction region with minimal energy loss, i.e., when the potential energy divided by total energy is less than or equal to 10^-10.

Once \(\theta\) is calculated as a function of \(b\), you can plot \(\sigma(\theta) = \left| \frac{d\theta}{db} \right|\) versus \(b \sin(\theta)\). This will help identify any discontinuities or sharp changes in the scattering angle.

Here's how to implement this:
```python
def calculate_scattering_angle(x0, y0, vx0, vy0):
    # Simulate particle motion and return final position (x_final, y_final)
    x_final, y_final = simulate_motion(x0, y0, vx0, vy0)

    # Calculate scattering angle using atan2 function
    theta = np.arctan2(vy_final, vx_final)
    return theta

def plot_discontinuities(impact_parameters, angles):
    b_values = [b for b in impact_parameters]
    dtheta_db = [(angles[i+1] - angles[i]) / (b_values[i+1] - b_values[i]) if i < len(b_values) - 1 else 0
                 for i in range(len(b_values))]
    
    # Plot |dθ/db| vs b sin(θ)
    plt.plot([b * np.sin(theta) for b, theta in zip(b_values, angles)], abs(dtheta_db))
    plt.xlabel('|b sin(θ)|')
    plt.ylabel('|\u03C4(\u03B8)/db|')
    plt.title('Discontinuities in Scattering Angle')
    plt.show()
```
x??

---

#### Lorenz Attractors
Background context: The Lorenz attractor is a classic example of chaotic behavior arising from a simplified atmospheric convection model. By simulating the system using differential equations, you can observe complex and unpredictable patterns.
:p What are the steps to simulate the Lorenz attractor?
??x
The task involves setting up an ODE solver to handle three simultaneous differential equations representing the Lorenz attractor:

1. \( \dot{x} = \sigma (y - x) \)
2. \( \dot{y} = x (\rho - z) - y \)
3. \( \dot{z} = -\beta z + xy \)

Where \(\sigma, \rho,\) and \(\beta\) are parameters with specific values.

You need to ensure the solver uses a sufficiently small step size for accurate results and avoid numerical errors.
```python
from scipy.integrate import solve_ivp
import numpy as np

# Define the Lorenz system of equations
def lorenz(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        -beta * z + x * y
    ]

# Define initial conditions and parameters
initial_conditions = [0.1, 0.1, 15.0]
sigma = 10.0
rho = 28.0
beta = 8 / 3

# Time span for the integration
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Solve the ODE system
sol = solve_ivp(lorenz, t_span, initial_conditions, args=(sigma, rho, beta), t_eval=t_eval)

# Plot x vs. y and z vs. t
plt.figure()
plt.plot(sol.y[0], sol.y[1])
plt.title('x vs y')
plt.show()

plt.figure()
plt.plot(t_eval, sol.y[2])
plt.title('z vs time')
plt.show()
```
x??

---

#### Lorenz Attractor in Phase Space
Background context: The phase space plots of the Lorenz attractor provide a visual representation of how trajectories evolve over time. These plots can reveal the characteristic "butterfly" shape associated with chaos.
:p What are the steps to create 2D and 3D phase space plots for the Lorenz attractor?
??x
To create the phase space plots, you need to plot the state variables \(x\), \(y\), and \(z\) against each other. The 2D phase space plots will show how these variables evolve with respect to each other over time.

For example:
1. Plot \(z(t)\) vs. \(x(t)\)
2. Plot \(y(t)\) vs. \(x(t)\) and \(y(t)\) vs. \(z(t)\)
3. Create a 3D plot of \(x(t)\), \(y(t)\), and \(z(t)\)

Here's how to implement this:
```python
from mpl_toolkits.mplot3d import Axes3D

# Plot z(t) vs x(t)
plt.figure()
plt.plot(sol.y[0], sol.y[2])
plt.title('z vs x')
plt.xlabel('x(t)')
plt.ylabel('z(t)')
plt.show()

# Plot y(t) vs x(t)
plt.figure()
plt.plot(sol.y[0], sol.y[1])
plt.title('y vs x')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.show()

# Plot y(t) vs z(t)
plt.figure()
plt.plot(sol.y[2], sol.y[1])
plt.title('y vs z')
plt.xlabel('z(t)')
plt.ylabel('y(t)')
plt.show()

# 3D plot of x(t), y(t), and z(t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2])
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
plt.title('3D Lorenz Attractor')
plt.show()
```
x??

---

#### Sensitivity to Parameters
Background context: The chaotic behavior of the Lorenz attractor is highly sensitive to initial conditions and parameters. This property can be exploited to explore how small changes in parameters lead to different dynamical behaviors.
:p How can you check if the given parameters lead to chaotic solutions?
??x
To verify that the given parameters (\(\sigma=10\), \(\beta=\frac{8}{3}\), and \(\rho=28\)) indeed produce chaotic behavior, you need to make small changes to these parameters and observe if it eventually leads to different solutions.

Here’s a step-by-step approach:
1. Run the simulation with the given parameters.
2. Make minor adjustments to one of the parameters (e.g., change \(\sigma\) by 0.1).
3. Re-run the simulation with the new parameter values.
4. Compare the results and check if they differ significantly.

For instance, you can modify \(\sigma\) slightly:
```python
# Change sigma by a small amount
new_sigma = 10.1

# Solve the ODE system with the new parameters
sol_new = solve_ivp(lorenz, t_span, initial_conditions, args=(new_sigma, rho, beta), t_eval=t_eval)

# Compare the solutions
plt.figure()
plt.plot(sol.y[2], sol.y[0], label='Original')
plt.plot(sol_new.y[2], sol_new.y[0], label='New sigma=10.1', linestyle='--')
plt.legend()
plt.title('Comparison of z vs x with original and new sigma')
plt.show()
```
x??

#### Van der Pol Oscillator
Background context: The van der Pol oscillator is a nonlinear oscillator that exhibits self-sustained oscillations. It is described by the differential equation:
\[ \frac{d^2 x}{dt^2} + \mu(x^2 - x_0^2) \frac{dx}{dt} + \omega_0^2 x = 0. \]
The term \( \mu(x^2 - x_0^2) \frac{dx}{dt} \) represents a nonlinear damping force, which is dependent on the amplitude of the oscillator.

:p Explain why (16.25) describes an oscillator with x-dependent damping.
??x
This equation describes an x-dependent damping because the term \( \mu(x^2 - x_0^2) \frac{dx}{dt} \) includes a nonlinear factor that depends on the amplitude of the oscillation \( x \). The coefficient \( \mu \) scales the damping force, and when \( x = x_0 \), the damping vanishes. This results in a more complex behavior compared to linear systems.
x??

---

#### Phasespace Plots for Van der Pol Oscillator
Background context: Phasespace plots are useful tools for visualizing the behavior of dynamical systems. For the van der Pol oscillator, we plot \( \dot{x} \) (the time derivative of \( x \)) versus \( x \).

:p Create phasespace plots \( \dot{x}(t) \) versus \( x(t) \).
??x
To create these plots, you would numerically integrate the van der Pol equation using a method like Runge-Kutta. The plot will show how the state of the system evolves over time in phase space.

Example code snippet:
```python
import numpy as np
from scipy.integrate import solve_ivp

# Define the van der Pol oscillator function
def vdp_oscillator(t, X, mu, omega0):
    x, dx = X
    dXdt = [dx, -mu * (x**2 - 1) * dx - omega0**2 * x]
    return dXdt

# Initial conditions and parameters
x0 = 0.5
dx0 = 0
t_span = (0, 50)
params = {'mu': 3, 'omega0': 1}

# Solve the differential equation
sol = solve_ivp(vdp_oscillator, t_span, [x0, dx0], args=(params['mu'], params['omega0']))

# Plot x(t) and \dot{x}(t)
import matplotlib.pyplot as plt

plt.plot(sol.y[0], sol.y[1])
plt.xlabel('x')
plt.ylabel('\dot{x}')
plt.title('Phasespace plot of van der Pol Oscillator')
plt.show()
```
x??

---

#### Duffing Oscillator
Background context: The Duffing oscillator is a driven, damped nonlinear system. It is described by the differential equation:
\[ \frac{d^2 x}{dt^2} = -2\gamma \frac{dx}{dt} - \alpha x - \beta x^3 + F \cos(\omega t). \]
This model includes both linear and cubic damping terms, as well as a driving force.

:p Modify your ODE solver to solve (16.26).
??x
To modify the ODE solver for the Duffing oscillator equation, you can use a numerical integration method like Runge-Kutta. Here’s an example using `solve_ivp` from SciPy:

```python
from scipy.integrate import solve_ivp

# Define the Duffing oscillator function
def duffing_oscillator(t, X, gamma, alpha, beta, F, omega):
    x, dx = X
    dXdt = [dx, -2*gamma*dx - alpha*x - beta*x**3 + F*np.cos(omega*t)]
    return dXdt

# Initial conditions and parameters
x0 = 0.009
dx0 = 0
t_span = (0, 200)
params = {'gamma': 0.2, 'alpha': 1.0, 'beta': 0.2, 'F': 4.0, 'omega': 1.0}

# Solve the differential equation
sol = solve_ivp(duffing_oscillator, t_span, [x0, dx0], args=(params['gamma'], params['alpha'], params['beta'], params['F'], params['omega']))

# Plot x(t)
import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Duffing Oscillator Response')
plt.show()
```
x??

---

#### Period Three Solution for Duffing Oscillator
Background context: The period three solution is a specific behavior that can emerge in the Duffing oscillator under certain parameter conditions.

:p Search for period-three solutions like those in Figure 16.15.
??x
To search for period-three solutions, you need to adjust the parameters of the Duffing oscillator and observe the response over multiple cycles. In this case, we use the specific parameters that are known to produce a period-three solution:
```python
# Parameters corresponding to a period-three solution
params = {'gamma': 0.2, 'alpha': 1.0, 'beta': 0.2, 'F': 0.2, 'omega': 1.0}

# Solve for 100 cycles first to eliminate transients
sol = solve_ivp(duffing_oscillator, (0, 300), [x0, dx0], args=(params['gamma'], params['alpha'], params['beta'], params['F'], params['omega']))

# Then plot the phase space
plt.plot(sol.y[0][-100:], sol.y[1][-100:])
plt.xlabel('x(t)')
plt.ylabel('\dot{x}(t)')
plt.title('Phase Space Plot for Duffing Oscillator')
plt.show()
```
This code snippet will help you identify if a period-three solution is present. If the plot shows a pattern consistent with three cycles, then it indicates a period-three behavior.

If no such pattern is observed, adjust parameters and re-run until a clear period-three cycle is visible.
x??

---

#### Ueda Oscillator
Background context: The Ueda oscillator is another form of a forced Duffing oscillator. It is described by the same equation but with specific parameter values:
\[ \frac{d^2 x}{dt^2} = -2\gamma \frac{dx}{dt} - \alpha x - \beta x^3 + F \cos(\omega t). \]
For modeling an Ueda oscillator, \( \omega = 1.0 \) and \( \alpha = 0 \).

:p Change your parameters to \(\omega=1\) and \(\alpha=0\).
??x
To model the Ueda oscillator with the specified parameters, you can update the parameter values in your Duffing oscillator function:
```python
# Parameters for Ueda Oscillator
params = {'gamma': 0.2, 'alpha': 0, 'beta': 0.2, 'F': 4.0, 'omega': 1.0}

# Solve the differential equation with new parameters
sol = solve_ivp(duffing_oscillator, t_span, [x0, dx0], args=(params['gamma'], params['alpha'], params['beta'], params['F'], params['omega']))

# Plot x(t)
plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Ueda Oscillator Response')
plt.show()
```
This code will simulate the Ueda oscillator with the updated parameters.
x??

---

#### Square Billiard Table
Background context: The square billiard table is a simple model of a particle bouncing around within a boundary. It helps illustrate concepts like trajectory and reflection in classical mechanics.

:p What is the question about this concept?
??x
The question is about understanding how to simulate motion on a square billiard table, including reflections off the walls.
x??

---

#### Code for Square Billiard Table
Background context: The provided code simulates the motion of a particle inside a square billiard table. The simulation includes handling collisions with the boundaries and updating the position accordingly.

:p Explain the logic in the given code snippet for the square billiard table.
??x
The code simulates the motion of a ball on a square billiard table by tracking its position and velocity over time, adjusting based on collisions at the edges. The key steps are:

1. Set initial conditions: Position \( (Xo, Yo) \), initial velocity \( v \).
2. Define the boundaries.
3. Use a loop to update the ball's position over discrete time steps.

If the ball hits a boundary, its direction is reversed.

Code:
```python
dt = 0.01; Xo = -90.; Yo = -5.4; v = vector(13., 13.1)
r0 = r = vector(Xo, Yo); eps = 0.1; Tmax = 500; tp = 0
scene = display(width=500, height=500, range=120, background=color.white, foreground=color.black)

table = curve(pos=[(-100, -100, 0), (100, -100, 0), (100, 100, 0), (-100, 100, 0), (-100, -100, 0)])
ball = sphere(pos=(Xo, Yo, 0), color=color.red, radius=3, make_trail=True)

for t in range(0, Tmax, dt):
    rate(5000)
    tp = tp + dt
    r = r0 + v * tp

    if (r.x >= 100 or r.x <= -100):  # Right and left walls
        v = vector(-v.x, v.y, 0)

    if (r.y >= 100 or r.y <= -100):  # Top and bottom walls
        v = vector(v.x, -v.y, 0)
    
    r0 = vector(r.x, r.y, 0)
    tp = 0

    ball.pos = r
```
x??

---

