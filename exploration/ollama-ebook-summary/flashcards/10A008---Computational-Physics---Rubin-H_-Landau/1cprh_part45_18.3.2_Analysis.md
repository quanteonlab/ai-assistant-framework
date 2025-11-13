# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 45)

**Starting Chapter:** 18.3.2 Analysis

---

#### Verlet Algorithm Overview
Background context: The Verlet algorithm is a method used in molecular dynamics (MD) simulations to integrate Newton's equations of motion. It uses a central-difference approximation for the second derivative to update positions and velocities simultaneously.

:p What is the main idea behind the Verlet algorithm?
??x
The Verlet algorithm provides an efficient way to simulate the motion of particles by updating their positions based on forces, without needing explicit velocity values until later steps.
```java
// Pseudocode for basic Verlet integration
for each particle i in system {
    // Compute acceleration Fi at current position ri
    Fi = computeAcceleration(ri);
    
    // Update position to the future state using previous and current positions
    ri(t+h) = 2 * ri(t) - ri(t-h) + h^2 * Fi;
}
```
x??

---

#### Velocity-Verlet Algorithm Details
Background context: The velocity-Verlet algorithm is an improved version of the Verlet algorithm, providing more stability. It uses a forward-difference approximation to update both positions and velocities simultaneously.

:p What distinguishes the velocity-Verlet algorithm from the basic Verlet algorithm?
??x
The velocity-Verlet algorithm updates velocities using a forward difference approximation, which incorporates information about forces at future time steps, thus providing better stability and accuracy compared to the basic Verlet algorithm.
```java
// Pseudocode for Velocity-Verlet integration
for each particle i in system {
    // Compute acceleration Fi at current position ri
    Fi = computeAcceleration(ri);
    
    // Update velocity using forces from previous and current time steps
    vi(t+h) = vi(t) + h * (Fi(t) + Fi(t+h)) / 2;
    
    // Update position to the future state using updated velocities
    ri(t+h) = ri(t) + h * vi(t) + h^2 * Fi(t) / 2;
}
```
x??

---

#### Implementation and Exercises for MD Simulations
Background context: The provided code snippets (MD1D.py, MD2D.py, MDpBC.py) demonstrate basic implementations of one-dimensional, two-dimensional, and periodic boundary condition molecular dynamics simulations using the velocity-Verlet algorithm.

:p What is the purpose of these implementation exercises?
??x
The purpose is to familiarize users with running and visualizing 1D and 2D molecular dynamics simulations. Users will learn how to initialize particles, apply periodic boundary conditions, and observe their behavior over time.
```python
# Example pseudocode for initializing and simulating a particle in MD2D.py
def init_particles():
    # Place particles at lattice sites of simple cubic structure

def run_simulation(steps):
    for step in range(steps):
        update_forces()
        update_positions_and_velocities()
```
x??

---

#### Periodic Boundary Conditions (PBC)
Background context: PBCs are essential in MD simulations to prevent particles from escaping the simulation box. The potential is cutoff beyond a certain radius, and interactions with periodic images of atoms are considered.

:p How do you implement periodic boundary conditions in an MD simulation?
??x
To implement PBCs, first update particle positions using the Verlet algorithm or similar method. Then check for image particles that might have crossed the box boundaries, adjust their positions accordingly, and calculate forces between these updated positions.
```python
# Example pseudocode for applying periodic boundary conditions
def apply_PBC(position):
    if position > box_length:
        position -= box_length * round(position / box_length)
    elif position < 0:
        position += box_length * (1 + round(-position / box_length))
    return position
```
x??

---

#### Time-Averaged Energy Calculation
Background context: The total, kinetic, and potential energies in an MD system change over time as particles equilibrate. Time-averaging these energies provides insights into the thermal behavior of the system.

:p How do you calculate time-averaged energies for an equilibrated system?
??x
Time-averaged energies are calculated by averaging the total energy (KE + PE) over a sufficient number of simulation steps.
```python
def calculate_time_averaged_energy(steps):
    total_energy_sum = 0
    for step in range(steps):
        current_energy = kinetic_energy() + potential_energy()
        total_energy_sum += current_energy
    return total_energy_sum / steps
```
x??

---

#### Temperature Comparison and Relation
Background context: The initial and final temperatures of an MD system can be compared to understand the thermal dynamics. Changes in temperature during equilibration provide insights into energy transfer.

:p How do you compare the final and initial temperatures after running an MD simulation?
??x
To compare the final and initial temperatures, run a simulation at a given initial temperature, allow it to equilibrate, and then check the final temperature. You may observe that the system reaches thermal equilibrium at a different temperature.
```python
def compare_initial_final_temperatures(initial_temp, steps):
    # Run MD simulation with initial temperature
    simulate_system(initial_temp, steps)
    
    # Retrieve or calculate final temperature from simulation data
    final_temperature = get_final_temperature()
    
    return (initial_temp, final_temperature)
```
x??

---

#### Root-Mean-Square Displacement Analysis
Background context: RMS displacement is a measure of how much particles move over time. It helps understand the dynamics and diffusion behavior in MD simulations.

:p What is the formula for calculating root-mean-square displacement?
??x
The RMS displacement is calculated using:
$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (r_i(t) - r_i(0))^2 }$$where $ N $ is the number of particles, and $ r_i(t)$are their positions at time $ t$.
```python
def calculate_rms_displacement(positions):
    sum_of_squares = 0
    for pos in positions:
        sum_of_squares += (pos - initial_position) ** 2
    
    return math.sqrt(sum_of_squares / len(positions))
```
x??

---

#### Diffusion Simulation with MD
Background context: Lighter molecules tend to diffuse more quickly than heavier ones. Using a Lennard-Jones potential and periodic boundary conditions, you can simulate this behavior in an MD system.

:p How do you generalize the velocity-Verlet algorithm for particles of different masses?
??x
Generalize by incorporating mass into the acceleration calculation:
$$\text{acceleration} = \frac{\text{force}}{\text{mass}}$$

In pseudocode, this would look like:
```python
def compute_acceleration(position, mass):
    force = calculate_force(position)
    return force / mass
```
x??

--- 
Note: The code examples are simplified for clarity and may need adjustments to fit specific programming languages or contexts.

#### MD Program for 16 Particles

Background context: This concept involves implementing a molecular dynamics simulation using velocity-Verlet algorithm to simulate 16 particles in a 2D box with periodic boundary conditions (PBCs). The goal is to study particle distribution, velocity distribution, and heat capacity.

:p What are the steps to modify an existing MD program for simulating 16 particles in a 2D box?

??x
The task involves extending the existing MD simulation code to count the number of particles on the right-hand side (RHS) of the box after each time step, updating a histogram with these counts, and comparing the results with theoretical probabilities. Here's an outline:

1. **Modify the simulation loop**:
   - After each time step, check if any particle has crossed the boundary due to periodic conditions.
   - Increment counters for particles on the RHS.

2. **Update histograms**:
   - Create a histogram that records the number of times `Nrhs` values occur.
   - Calculate and plot the probability distribution using equation (18.23).

3. **Comparison with theoretical results**:
   - Compare your simulation results with those in Figure 18.7, which are generated by running MDpBC.py.

:p How would you implement the counting of particles on the RHS?

??x
In each time step after updating particle positions and velocities using velocity-Verlet algorithm, check if a particle's position has crossed the boundary due to PBCs. If so, increment the count for that particle being in the RHS. Here is an example pseudocode:

```java
for (Particle p : particles) {
    // Update position using velocity-Verlet
    p.updatePosition();
    
    // Apply periodic boundary conditions
    applyPBC(p);
    
    if (p.isInRHS()) {
        count++;
    }
}
```

x??

---

#### Histogram of Number of Particles on RHS

Background context: This concept involves creating a histogram to show the distribution of particles crossing the box's right-hand side, and comparing it with the theoretical probability.

:p How would you create a histogram showing the number of times `Nrhs` values occur?

??x
To create this histogram, iterate through your simulation data and increment counts based on the number of particles found in the RHS at each snapshot. Use matplotlib or similar plotting library to visualize the histogram:

```python
import matplotlib.pyplot as plt

# Assuming `rhs_counts` is a list of Nrhs values from the simulation
plt.hist(rhs_counts, bins=range(min(rhs_counts), max(rhs_counts) + 2))
plt.xlabel('Number of particles on RHS')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Particles on RHS')
plt.show()
```

x??

---

#### Probability Distribution for Finding Nrhs Particles

Background context: This involves calculating and plotting the probability distribution of finding a specific number `Nrhs` of particles in the right-hand side (RHS) using equation 18.23.

:p How would you calculate and plot the probability distribution?

??x
Calculate the binomial coefficient $C(n)$ for each possible value of `n` (number of particles on RHS). Then, use these values to compute probabilities and plot them:

```python
import numpy as np

# Number of particles in the box
N = 16
# Possible Nrhs range
Nrange = range(0, N+1)

probabilities = []
for n in Nrange:
    Cn = binom_coeff(n)  # Calculate binomial coefficient
    Pn = (Cn * 2**(-N)) / np.math.factorial(n)
    probabilities.append(Pn)

plt.plot(Nrange, probabilities)
plt.xlabel('Number of particles on RHS')
plt.ylabel('Probability')
plt.title('Probability Distribution for Finding Nrhs Particles on RHS')
plt.show()
```

x??

---

#### Velocity Distribution

Background context: This involves determining the velocity distribution of 16 particles by creating a histogram and ensuring that it resembles a normal distribution over time.

:p How would you create a histogram to determine the velocity distribution?

??x
Create histograms for particle velocities in each step and update them as the simulation progresses. Use matplotlib or similar libraries:

```python
import matplotlib.pyplot as plt

# Assuming `velocities` is a list of 16 velocity vectors (3D)
velocity_histogram = np.histogram([v[0] for v in velocities], bins=50)  # x-component of velocity
plt.hist(velocity_histogram[0], bins=50)
plt.xlabel('Velocity')
plt.ylabel('Frequency')
plt.title('Velocity Distribution')
plt.show()
```

x??

---

#### Heat Capacity Calculation

Background context: This involves computing and plotting the heat capacity at a constant volume,$C_V = \frac{\partial E}{\partial T}$, as a function of temperature for 16 particles in a box.

:p How would you compute the heat capacity at a constant volume?

??x
Compute the average total energy for multiple initial conditions and temperatures. Use numerical differentiation to find the derivative with respect to temperature:

```python
temperatures = np.linspace(0.5, 20, 10)  # Example temperature range

for T in temperatures:
    energies = []  # List to store energy at each step for current temperature
    
    # Simulate particles with initial speed v0 and random directions
    for _ in range(10):  # Repeat simulation 10 times for averaging
        # Initialize velocities randomly
        velocities = [random_velocity(v0) for _ in range(16)]
        
        E, T_calculated = calculate_energy_and_temperature(velocities)
        energies.append(E)
    
    average_energy = np.mean(energies)
    dE_dT = (average_energy - previous_average_energy) / (T - previous_T)

plt.plot(temperatures, [dE_dT for _ in range(len(temperatures))])
plt.xlabel('Temperature')
plt.ylabel('Heat Capacity C_V')
plt.title('Heat Capacity at Constant Volume')
plt.show()
```

x??

---

#### Effect of a Projectile

Background context: This concept involves simulating the effect of a projectile hitting a group of particles and observing its impact on the particle distribution.

:p How would you simulate the effect of a projectile hitting a group of particles?

??x
Simulate the collision by introducing a moving projectile particle with a high velocity into the system. Update the positions and velocities of all particles after each time step to account for collisions:

```java
// Assume `projectile` is the projectile particle and `particles` is the list of 16 particles

for (int t = 0; t < simulationSteps; t++) {
    // Move the projectile
    projectile.updatePosition();
    
    // For each particle, check collision with the projectile
    for (Particle p : particles) {
        if (collides(p, projectile)) {
            updateVelocity(p, projectile);
        }
    }
}
```

x??

---

#### Molecular Dynamics Simulations Overview
Molecular dynamics (MD) simulations are used to study the motion of atoms and molecules over time. These simulations combine Newton's equations of motion with molecular mechanics force fields, allowing the simulation of physical processes at the atomic scale.

In this context, MD is used in a 1D system where particles interact through pairwise forces. The objective is to understand how temperature affects the total energy and heat capacity of the system.

:p What are the key components involved in performing an MD simulation as described in the provided code?
??x
The key components include:
- Defining initial positions and velocities for atoms.
- Implementing a force calculation function that accounts for interactions between particles within a cutoff distance.
- Updating particle positions and velocities over time using integration methods like Verlet or Velocity Verlet.

The code also handles periodic boundary conditions to ensure the system behaves as if it were in an infinite space. For instance, when a particle moves beyond the boundary of the simulation box, its position is adjusted accordingly.

C/Java pseudocode:
```python
# Pseudocode for MD Simulation
def initialize():
    set initial positions and velocities

def calculate_forces(t):
    compute forces between atoms within cutoff distance

def update_positions_and_velocities(dt):
    # Verlet or Velocity-Verlet method to update positions and velocities
    pass

def main_simulation_loop(time_steps):
    for each time step:
        calculate forces at previous and current time steps
        update positions using calculated forces
        handle periodic boundary conditions
        compute kinetic energy
```
x??

---
#### Initial Conditions in MD Simulation
Setting the initial conditions is crucial as it initializes the state of the system. The provided code sets up an initial position for each atom based on its index and assigns random velocities scaled by the square root of the initial temperature.

:p How are the initial positions and velocities set up in the 1D molecular dynamics simulation?
??x
Initial positions and velocities are setup as follows:
- Positions: Initially, atoms are placed linearly along a line segment.
- Velocities: Each atom is assigned a random velocity which is scaled by $\sqrt{T}$ where $T$ is the initial temperature.

This ensures that the distribution of velocities approximates a Maxwell-Boltzmann distribution at the given temperature.

C/Java pseudocode:
```python
def initialize_positions_and_velocities():
    # Initialize positions based on index
    for i in range(0, L):
        x[i] = i * dx  # Assuming dx is some distance increment

    # Assign random velocities scaled by sqrt(T)
    for i in range(0, Natom):
        vx[i] = twelveran() * sqrt(Tinit)

def twelveran():
    s = 0.0
    for _ in range(12):
        s += random.random()
    return (s / 12) - 0.5
```
x??

---
#### Force Calculation in MD Simulation
The force calculation is a critical component of the simulation as it determines how particles interact with each other and influences their motion.

:p How are forces between atoms calculated in the provided code?
??x
Forces between atoms are calculated using a Lennard-Jones potential, which is commonly used to model interatomic interactions. The provided code has an incomplete implementation but follows these steps:
1. Iterate over all pairs of atoms.
2. Calculate the distance squared ($r^2$).
3. Check if the distance is within the cutoff radius.
4. Compute the force using the Lennard-Jones potential function.

The energy due to interactions between atoms is also calculated and accumulated in $PE$(potential energy).

C/Java pseudocode:
```python
def calculate_forces(t, PE):
    for i in range(0, Natom - 1):
        for j in range(i + 1, Natom):
            dx = x[i] - x[j]
            if abs(dx) > 0.5 * L:
                dx -= sign(L, dx)
            r2 = dx ** 2
            if r2 < r2cut:
                invr2 = 1 / r2
                wij = 48 * (invr2 ** 3 - 0.5) * invr2 ** 3
                fijx = wij * invr2 * dx
                fx[i][t] += fijx
                fx[j][t] -= fijx
    PE += 4 * (invr2 ** 3) * ((invr2 ** 3) - 1)
```
x??

---
#### Time Evolution in MD Simulation
The time evolution of the system is carried out by updating the positions and velocities of atoms over discrete time steps. The provided code uses a simple Velocity Verlet algorithm for this purpose.

:p How does the simulation update particle positions and velocities over time?
??x
Particle positions and velocities are updated using the following steps:
1. Calculate forces at previous and current time steps.
2. Update positions based on velocities and half of the force contribution from the previous step.
3. Apply periodic boundary conditions to ensure particles stay within the simulation box.
4. Update velocities by adding contributions from both previous and current forces.

C/Java pseudocode:
```python
def update_positions_and_velocities(dt):
    for i in range(0, Natom):
        PE = calculate_forces(t1, PE)
        x[i] += dt * (vx[i] + 0.5 * dt * fx[i][t1])
        if x[i] <= 0:
            x[i] += L
        elif x[i] >= L:
            x[i] -= L
        atoms[i].pos = (2 * x[i] - 7, 0)  # Linear transform to plot

    PE = calculate_forces(t2, PE)
    for i in range(0, Natom):
        vx[i] += 0.5 * dt * (fx[i][t1] + fx[i][t2])
```
x??

---
#### Energy Calculation in MD Simulation
Energy is a critical quantity to monitor during the simulation as it provides insights into the system's behavior.

:p How are kinetic and potential energies calculated in the provided code?
??x
Kinetic and potential energies are calculated as follows:
- Kinetic energy: Sum of $\frac{1}{2} m v^2$ for each particle.
- Potential energy: Sum of interaction energies between pairs of particles within a cutoff radius.

The kinetic energy is used to compute the temperature, while the potential energy is directly plotted over time.

C/Java pseudocode:
```python
def calculate_energies():
    KE = 0.0
    for i in range(0, Natom):
        KE += (vx[i] * vx[i]) / 2

    PE = 0.0
    PE = calculate_forces(t1, PE)
```
x??

---

#### Periodic Boundary Conditions (PBC) in Molecular Dynamics Simulation
Periodic boundary conditions are a common approach used in simulating systems with finite size, such as molecules or atoms. The idea is to treat the simulation box as if it were infinite by assuming that particles exiting one side of the box re-enter on the opposite side. This ensures that interactions between particles are considered over the entire system.

In this code snippet, PBCs are implemented using periodic boundary conditions in a 2D space:
```python
if x[i] <= 0.: x[i] = x[i] + L
if x[i] >= L: x[i] = x[i] - L
if y[i] <= 0.: y[i] = y[i] + L
if y[i] >= L: y[i] = y[i] - L
```
:p How are periodic boundary conditions applied in this code?
??x
Periodic boundary conditions (PBCs) are enforced by adjusting the position of atoms that cross the boundaries of the simulation box. If an atom's x-coordinate is less than or equal to 0, it wraps around to the right edge at `L`. Similarly, if its y-coordinate is less than or equal to 0, it wraps around to the top edge at `L`. The same logic applies for positions greater than or equal to `L`.

The periodic boundary condition ensures that interactions between atoms are considered as if they were in an infinite system. This is done by effectively "copying" the simulation box infinitely many times.
x??

---
#### Force Calculation in Molecular Dynamics Simulation
Force calculation is a critical step in molecular dynamics simulations, especially when using potential functions like the Lennard-Jones potential. The code snippet provided shows how forces are calculated between pairs of atoms based on their distance.

The interaction energy `wij` and force components `fijx` and `fijy` are computed as follows:
```python
if(r2 < r2cut):
    if(r2 == 0.): 
        r2 = 0.0001
    invr2 = 1./r2
    wij = 48.*(invr2**3 - 0.5) * invr2**3
    fijx = wij * invr2 * dx
    fijy = wij * invr2 * dy
```
:p What is the logic for force calculation in this molecular dynamics simulation?
??x
The force between two atoms `i` and `j` is calculated based on their distance. If the distance squared (`r2`) is less than a cutoff value (`r2cut`), the interaction energy `wij` is computed using the Lennard-Jones potential formula:
$$wij = 48 \cdot (invr^3 - 0.5) \cdot invr^3$$where $ invr = \frac{1}{\sqrt{r2}}$.

The force components in the x and y directions are then calculated as:
$$fijx = wij \cdot invr \cdot dx$$
$$fijy = wij \cdot invr \cdot dy$$

This ensures that the forces accurately reflect the attractive and repulsive interactions between atoms.
x??

---
#### Potential Energy Calculation
Potential energy calculation is essential for understanding the total potential energy of a system in molecular dynamics. The code snippet demonstrates how to calculate potential energy using the `Forces` function, which also accounts for kinetic energy.

The function `PE = Forces(t1 , w, PE, 1)` calculates potential energy and returns it if `PEorW == 1`, otherwise it returns a weight value.
:p How is potential energy calculated in this molecular dynamics simulation?
??x
Potential energy is calculated using the `Forces` function, which iterates through all pairs of atoms to compute pairwise interactions based on the Lennard-Jones potential. The total potential energy (`PE`) is updated by summing up the contributions from each pair.

The function also considers kinetic energy if required:
```python
for i in range(0, Natom):
    KE = KE + (vx[i] * vx[i] + vy[i] * vy[i]) / 2.0

PE = Forces(t1 , w, PE, 1)
```

If `PEorW == 1`, the function returns the potential energy; otherwise, it returns a weight value (`w`).
x??

---
#### Time Evolution of a System
Time evolution in molecular dynamics is handled by updating positions and velocities at each time step. The code snippet illustrates a simple Euler integration method to update positions and velocities.

The position updates are performed as:
```python
for i in range(0, Natom):
    x[i] = x[i] + h * (vx[i] + 0.5 * fx[i][t1])
    y[i] = y[i] + h * (vy[i] + 0.5 * fy[i][t1])

if x[i] <= 0.: 
    x[i] = x[i] + L
if x[i] >= L: 
    x[i] = x[i] - L
if y[i] <= 0.: 
    y[i] = y[i] + L
if y[i] >= L: 
    y[i] = y[i] - L
```
:p How is the position updated in this molecular dynamics simulation?
??x
The position of each atom is updated using a simple Euler integration method. At each time step, the new position is calculated as:
$$x'[i] = x[i] + h \cdot (vx[i] + 0.5 \cdot fx[i][t1])$$
$$y'[i] = y[i] + h \cdot (vy[i] + 0.5 \cdot fy[i][t1])$$

Here, `h` is the time step size, and `fx[i][t1]`, `fy[i][t1]` are the forces at the previous half time step.

After updating the positions, periodic boundary conditions are enforced to ensure that atoms wrap around the simulation box if they cross its boundaries.
x??

---
#### Force Updates for Time Evolution
Force updates in molecular dynamics are crucial for accurate time evolution. The code snippet shows how forces are updated during each time step:
```python
for i in range(0, Natom):
    vx[i] = vx[i] + 0.5 * (fx[i][t1] + fx[i][t2])
    vy[i] = vy[i] + 0.5 * (fy[i][t1] + fy[i][t2])

w = Forces(t2, w, PE, 2)
```
:p How are forces updated in this molecular dynamics simulation?
??x
Forces are updated by averaging the force contributions from two time steps. This is done to ensure numerical stability and accuracy:
$$vx[i] = vx[i] + 0.5 \cdot (fx[i][t1] + fx[i][t2])$$
$$vy[i] = vy[i] + 0.5 \cdot (fy[i][t1] + fy[i][t2])$$

After updating the velocities, the potential energy is recalculated using the `Forces` function with a different parameter setting (`PEorW == 2`), which updates only the forces and returns the weight value.
x??

---
#### Energy Averages in Molecular Dynamics
Energy averages are computed over multiple time steps to ensure statistical equilibrium. The code snippet demonstrates how energy averages (kinetic, potential, total) are calculated:
```python
avKE = avKE + KE
avPE = avPE + PE
t += 1

Pavg = avP / t
eKavg = avKE / t
ePavg = avPE / t
Tavg = ePavg / Natom
```
:p How are energy averages computed in this molecular dynamics simulation?
??x
Energy averages are computed by incrementally updating the total kinetic and potential energies at each time step. The average values are calculated as follows:
$$\text{avKE} = \text{avKE} + KE$$
$$\text{avPE} = \text{avPE} + PE$$

After completing a full cycle, the averages are computed by dividing the total energies by the number of time steps (`t`):
$$

Pavg = \frac{\text{avP}}{t}$$
$$eKavg = \frac{\text{avKE}}{t}$$
$$ePavg = \frac{\text{avPE}}{t}$$

The temperature is then calculated as the average potential energy per atom:
$$

Tavg = \frac{ePavg}{Natom}$$

This process ensures that the simulation reaches a statistically stable state.
x??

---

---
#### Atom Positioning and Visualization
Background context: The provided text describes a method for positioning atoms within a simulation box, visualizing their positions, setting initial velocities, and computing forces between them. This is typically used in molecular dynamics simulations.

:p How are atoms positioned initially?
??x
Atoms are positioned randomly within the defined boundaries of the simulation box using a uniform random distribution. The position `x` for an atom is calculated as:
```python
x = (L - Ratom) * random.random() - L + Ratom
```
This ensures that each atom's initial position is within the allowed range, considering the radius of the atoms.

```python
y = 2 * (L - Ratom) * random.random() - L + Ratom
```
After calculating `x` and `y`, a sphere representing an atom is created at these coordinates.
??x
---
#### Atom Visualization in Code
Background context: The code snippet creates visual spheres to represent atoms within the simulation. These spheres help in understanding the spatial distribution of atoms during the simulation.

:p What does this line of code do?
```python
Atom = Atom + [sphere(pos=(x, y), radius=Ratom, color=col)]
```
??x
This line adds a new atom (represented by a sphere) at position `(x, y)` with a given radius `Ratom` and a specified color to the list of atoms. This helps in visualizing the initial setup of the simulation.
??x
---
#### Initial Velocity Assignment
Background context: The code assigns initial velocities to each atom based on a random angle selection.

:p How are initial velocities assigned?
??x
Initial velocities for an atom are computed using a polar coordinate system approach, where:
```python
theta = 2 * pi * random.random() # Select angle 0 <= theta <= 2*pi
vx = pref * cos(theta) # x component velocity
vy = pref * sin(theta) # y component velocity
```
Here, `pref` is a predefined constant that sets the initial speed. The velocities are randomly oriented within the range of $0 $ to$2\pi$.

:p How are these velocities added to the simulation?
??x
These velocities are appended to a list:
```python
vel.append((vx, vy))
```
This step ensures that each atom's velocity is recorded and used in subsequent force calculations.
??x
---
#### Force Calculation Between Atoms
Background context: The text describes how forces between atoms are computed using the Lennard-Jones potential. This interaction helps in determining the dynamics of the system.

:p How are forces calculated between atoms?
??x
Forces between two atoms are calculated based on their relative positions and the Lennard-Jones potential:
```python
def forces(fr):
    for i in range(0, Natom - 1):
        for j in range(i + 1, Natom):
            dr = pos[i] - pos[j] # relative position

            if abs(dr[0]) > L: # smallest distance or image
                dr[0] = dr[0] - sign(2 * L, dr[0])

            if abs(dr[1]) > L:
                dr[1] = dr[1] - sign(2 * L, dr[1])

            r2 = mag2(dr) # squared distance

            if abs(r2) < Ratom: # to avoid 0 denominator
                r2 = Ratom

            invr2 = 1. / r2
            fij = invr2 * factor * 48. * (invr2 ** 3 - 0.5) * invr2 ** 3
            fr[i] = fij * dr + fr[i]
            fr[j] = -fij * dr + fr[j]

    return fr
```
:p What does this function do?
??x
This function calculates the forces between pairs of atoms using the Lennard-Jones potential. The force calculation involves:
1. Determining the relative position `dr` between two atoms.
2. Adjusting for periodic boundary conditions to ensure the closest image is considered.
3. Calculating the squared distance and its inverse.
4. Applying the Lennard-Jones formula to find the force component.

:p How are forces updated over time using Velocity Verlet integration?
??x
The forces are used in a Velocity Verlet algorithm for updating positions and velocities:
```python
for t in range(0, 1000):
    Nrhs = 0 # begin 0 each time

    for i in range(0, Natom):
        fr = forces(fr)
        
        dpos = pos[i]
        if dpos[0] <= -L: 
            pos[i] = [dpos[0] + 2 * L, dpos[1]] # x periodic BC
        elif dpos[0] >= L:
            pos[i] = [dpos[0] - 2 * L, dpos[1]]

        if dpos[1] <= -L: 
            pos[i] = [dpos[0], dpos[1] + 2 * L] # y periodic BC
        elif dpos[1] >= L:
            pos[i] = [dpos[0], dpos[1] - 2 * L]

        if dpos[0] > 0 and dpos[0] < L: 
            Nrhs += 1

        fr2 = forces(fr)
        v[i] = v[i] + 0.5 * h * h * (fr[i] + fr2[i]) # velocity Verlet
        pos[i] = pos[i] + h * v[i] + 0.5 * h * h * fr[i]
```
:p What is the purpose of this loop?
??x
This loop updates positions and velocities for each atom over time using the Velocity Verlet method, which is a numerical integration scheme to solve Newton's equations of motion.

The `forces` function is called twice within one step of the loop to ensure consistency in force calculations, reflecting the update rule:
$$v(t + h) = v(t) + 0.5h(f(t) + f(t + h))$$and$$x(t + h) = x(t) + hv(t + h/2) + 0.5hf(t + h/2)$$

This approach ensures accurate trajectory updates while handling periodic boundary conditions.
??x
---

