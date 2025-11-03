# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 111)

**Starting Chapter:** Chapter 24 Quantum Wave Packets and EM Waves. 24.1 TimeDependent Schrodinger Equation

---

#### Time-Dependent Schrödinger Equation Introduction
In this problem, an electron is confined within a region of size comparable to an atom. The initial state of the electron has both defined momentum and position, requiring a quantum mechanical solution using the time-dependent Schrödinger equation.

The initial wave function for the electron is modeled as:
\[
\psi(x,t=0) = \exp\left[-\frac{1}{2} \left(\frac{x-5}{\sigma_0}\right)^2\right] e^{ik_ox}
\]
where \( \hbar = 1 \).

The wave function is a superposition of a Gaussian localized in space and a plane wave representing the defined momentum. However, this wave packet is neither an eigenstate of position nor momentum.

:p What does the initial wave function represent for the electron?
??x
The initial wave function represents an electron that starts with both a specific position (centered at \( x = 5 \)) and a well-defined momentum (\( k_o \)). This state is described by a Gaussian function in space multiplied by a plane wave.

This setup ensures the wave function has both spatial localization and momentum, which are necessary to accurately model the initial conditions of an electron under confinement.
x??

---

#### Time-Dependent Schrödinger Equation Formulation
The time-dependent Schrödinger equation needs to be solved for this scenario. The Hamiltonian \( \tilde{H} \) is given by:
\[
i \frac{\partial \psi(x,t)}{\partial t} = -\frac{1}{2m} \frac{\partial^2 \psi(x,t)}{\partial x^2} + V(x)\psi(x,t)
\]
For simplicity, \( 2m = 1 \) and \( \hbar = 1 \).

Since the initial wave function is complex, both real and imaginary parts need to be handled separately.

:p What equation describes the time-dependent Schrödinger equation for this scenario?
??x
The time-dependent Schrödinger equation for this scenario is given by:
\[
i \frac{\partial \psi(x,t)}{\partial t} = -\frac{1}{2m} \frac{\partial^2 \psi(x,t)}{\partial x^2} + V(x)\psi(x,t)
\]
where \( 2m = 1 \) and \( \hbar = 1 \).

This equation governs the evolution of the wave function over time, accounting for both spatial derivatives and potential energy terms.
x??

---

#### Real and Imaginary Parts of Wave Function
To handle complex wave functions, separate equations are derived for the real (\( R(x,t) \)) and imaginary (\( I(x,t) \)) parts:
\[
\psi(x,t) = R(x,t) + iI(x,t)
\]
The resulting partial differential equations (PDEs) for \( R \) and \( I \) are:
\[
\frac{\partial R(x,t)}{\partial t} = -\frac{\partial^2 I(x,t)}{\partial x^2} + V(x)I(x,t)
\]
\[
\frac{\partial I(x,t)}{\partial t} = +\frac{\partial^2 R(x,t)}{\partial x^2} - V(x)R(x,t)
\]

These equations describe the evolution of each component separately, ensuring a complete solution for \( \psi(x,t) \).

:p What are the partial differential equations (PDEs) describing the real and imaginary parts of the wave function?
??x
The PDEs describing the real (\( R \)) and imaginary (\( I \)) parts of the wave function are:
\[
\frac{\partial R(x,t)}{\partial t} = -\frac{\partial^2 I(x,t)}{\partial x^2} + V(x)I(x,t)
\]
\[
\frac{\partial I(x,t)}{\partial t} = +\frac{\partial^2 R(x,t)}{\partial x^2} - V(x)R(x,t)
\]

These equations ensure that the real and imaginary components evolve according to their respective PDEs, reflecting the complex nature of the wave function.
x??

---

#### Time Propagation Using Split Times
To propagate the initial wave function through time, algorithms with split times are used. This involves solving for \( R(x,t) \) and \( I(x,t) \) at slightly differing moments.

:p What method is used to handle the complex nature of the wave function over time?
??x
The method used to handle the complex nature of the wave function over time involves propagating the initial wave function using split times. This means solving for the real (\( R(x,t) \)) and imaginary (\( I(x,t) \)) parts at slightly differing moments.

This approach allows for accurate propagation by handling the separate evolution of each component, ensuring a comprehensive solution.
x??

---

#### Visualization of Probability Density
The probability density as a function of time and space is visualized to understand how the electron spreads out over time. For example:
- Figure 24.1 shows the probability density in a square well, illustrating the spread and collision with walls.
- Figure 24.2 depicts the same for a one-dimensional harmonic oscillator.

These visualizations help comprehend the dynamics of the wave packet over time.

:p What are Figures 24.1 and 24.2 used to illustrate?
??x
Figures 24.1 and 24.2 are used to illustrate the evolution of the probability density for an electron in a confining potential:

- **Figure 24.1** shows how the probability density changes over time and space within a square well, demonstrating the initial state starting on the left, spreading out, and colliding with the walls.
- **Figure 24.2** depicts the same process for an electron in a one-dimensional harmonic oscillator potential well, providing both a conventional surface plot and a color visualization.

These figures help visualize the spread and behavior of the wave packet over time.
x??

---

#### Split-Time Algorithm for Probability Conservation

Background context: The Schrödinger equation can be solved using both implicit and explicit methods. However, conserving probability \( \int_{-\infty}^{+\infty} dx \rho(x,t) = 1 \) at high precision is a challenge. An explicit method that ensures high probability conservation involves solving for the real and imaginary parts of the wave function at staggered times.

Relevant formulas:

- Real part equation:
  \[
  R(x, t + \frac{1}{2}\Delta t) = R(x, t - \frac{1}{2}\Delta t) + [4\alpha + V(x)\Delta t]I(x, t) - 2\alpha[I(x+\Delta x, t) + I(x-\Delta x, t)]
  \]
  
- Imaginary part equation:
  \[
  I(x, t + \frac{1}{2}\Delta t) = I(x, t - \frac{1}{2}\Delta t) + [4\alpha + V(x)\Delta t]R(x, t) - 2\alpha[R(x+\Delta x, t) + R(x-\Delta x, t)]
  \]

- Probability density:
  \[
  \rho(t) = 
    \begin{cases} 
      R^2(t) + I(t + \frac{1}{2}\Delta t)I(t - \frac{1}{2}\Delta t), & \text{for integer } t \\
      I^2(t) + R(t + \frac{1}{2}\Delta t)R(t - \frac{1}{2}\Delta t), & \text{for half-integer } t
    \end{cases}
  \]

:p What is the split-time algorithm and how does it ensure probability conservation?
??x
The split-time algorithm involves solving for the real part \( R \) at times \( 0, \Delta t, 2\Delta t, \ldots \), and the imaginary part \( I \) at staggered times \( \frac{1}{2}\Delta t, \frac{3}{2}\Delta t, \ldots \). This approach uses Taylor expansions to update these parts:

- For the real part:
  \[
  R_{n+1}^i = R_n^i - 2\alpha[I_{n}^{i+1} + I_{n}^{i-1}] + 4\alpha I_n^i
  \]
  
- For the imaginary part:
  \[
  I_{n+1}^i = I_n^i + 2\alpha[R_{n}^{i+1} + R_{n}^{i-1}] - 4\alpha R_n^i
  \]

These updates ensure that probability conservation is maintained to a high level, even though the algorithm does not exactly conserve it. The error in probability conservation is typically two orders of magnitude lower than the wave function itself.

:p How do you implement the split-time algorithm for solving the Schrödinger equation?
??x
To implement the split-time algorithm:

1. Define arrays `psr[751,2]` and `psi[751,2]` for real and imaginary parts of the wave function.
2. Set up initial values based on given parameters:
   - \(\sigma_0 = 0.5\)
   - \(\Delta x = 0.02\)
   - \(k_o = 17\pi\)
   - \(\Delta t = \frac{1}{2}(\Delta x)^2\)

3. Use equation (24.1) to initialize the wave packet:
   - Set `psr[j,1]` for all j at \(t=0\) and `psi[j,1]` for half time steps.

4. Ensure boundary conditions by setting \(\rho[1]\) and \(\rho[751]\) to 0 due to infinite walls.

5. Update the wave function in discrete steps using equations (24.8) and (24.9).

6. Replace present wave packets with future ones across all space.

7. Verify probability conservation by integrating over space at various times.

:p How do you check for probability conservation?
??x
To check for probability conservation, compute the integral of the probability density over all space:

\[
\int_{-\infty}^{+\infty} dx \rho(x,t)
\]

This should remain constant or change minimally with time. If significant changes are observed, adjust step sizes.

:p Why do collisions with walls cause wave packets to broaden and break up?
??x
Collisions with the walls cause the wave packet to spread out (broaden) and eventually disintegrate because:

- When a Gaussian wave packet is confined within a harmonic oscillator potential, it initially maintains its form due to the smooth potential.
- However, when confined in an infinite square well with abrupt boundaries, each collision at these boundaries causes discontinuities that disrupt the localized nature of the wave packet.

These disruptions spread out the wave function over time, leading to broadening and potentially breakup into smaller components.

#### 2D Harmonic Oscillator Wavepacket Motion
Background context: The problem involves determining the motion of a Gaussian wavepacket within a 2D harmonic oscillator potential. The initial localization is given by Eq. (24.14), and the potential energy function \(V(x,y) = 0.3(x^2 + y^2)\).

The classical analog of this system exhibits chaotic behavior, but for quantum mechanics, we use wavepackets to model particles. The goal is to observe how the Gaussian wavepacket evolves over time within the harmonic oscillator well.

:p Determine the motion of a Gaussian wavepacket in a 2D harmonic oscillator potential.
??x
To determine the motion, we need to solve the time-dependent Schrödinger equation for a Gaussian wavepacket in this potential. The initial condition is given by Eq. (24.14), and the Hamiltonian operator \( \tilde{H} = -\left(\frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}\right) + V(x,y)\).

The wavepacket can be translated forward in time using Eq. (24.18):
\[ \psi_{n+1}^{i,j} = U(\Delta t) \psi_n^{i,j}, \]
where \(U(t) = e^{-i\tilde{H}t}\), and the evolution operator for a small time step is:
\[ \psi_{n+1} = \psi_{n-1} + (e^{-i\tilambda \Delta t} - e^{i\tilambda \Delta t}) \psi_n. \]

Here, \(\lambda = \frac{\Delta t}{2 (\Delta x)^2}\).

The second derivative in the Schrödinger equation can be approximated using a Taylor expansion:
\[ \frac{\partial^2 \psi}{\partial x^2} \approx -\frac{1}{2} (\psi_{i+1,j} + \psi_{i-1,j} - 2 \psi_{i,j}). \]

Substituting these into the Schrödinger equation results in a discrete update formula for the wavepacket.

```java
// Pseudocode for updating the wavepacket
for (int n = 0; n < total_time_steps; n++) {
    // Update real and imaginary parts of the wavefunction
    for (int i = 1; i < grid_size_x - 1; i++) {
        for (int j = 1; j < grid_size_y - 1; j++) {
            R[i][j] += 2 * ((4*alpha + 0.5*dt*V[i][j]) * I[i][j] - alpha * (I[i+1][j] + I[i-1][j] + I[i][j+1] + I[i][j-1]));
            I[i][j] -= 2 * ((4*alpha + 0.5*dt*V[i][j]) * R[i][j] + alpha * (R[i+1][j] + R[i-1][j] + R[i][j+1] + R[i][j-1]));
        }
    }
}
```
x??

---

#### Young’s Single-Slit Experiment
Background context: This experiment involves a Gaussian wavepacket of width 3 passing through a slit of width 5, to observe the quantum interference pattern.

The initial condition is given by Eq. (24.14), and we need to simulate how this wavepacket behaves as it passes through the slit.

:p Simulate a Gaussian wavepacket passing through a single slit.
??x
To simulate the Gaussian wavepacket passing through a single slit, we start with an initial wavefunction described by:
\[ \psi(x,y,t=0) = e^{ik_0 x}e^{ik_0 y} e^{-\frac{(x-x_0)^2}{2\sigma^2}} e^{-\frac{(y-y_0)^2}{2\sigma^2}}, \]
where \(k_0\) is the momentum, and \(x_0, y_0, \sigma\) are parameters describing the initial localization.

When this wavepacket passes through a slit of width 5, it will exhibit diffraction effects due to its wave nature. The resulting pattern can be calculated by propagating the wavefunction using the time-dependent Schrödinger equation and observing the interference pattern at the other side of the slit.

```java
// Pseudocode for simulating single-slit experiment
for (int n = 0; n < total_time_steps; n++) {
    // Update real and imaginary parts of the wavefunction using the time-dependent Schrödinger equation
    for (int i = 1; i < grid_size_x - 1; i++) {
        for (int j = 1; j < grid_size_y - 1; j++) {
            R[i][j] += 2 * ((4*alpha + 0.5*dt*V[i][j]) * I[i][j] - alpha * (I[i+1][j] + I[i-1][j] + I[i][j+1] + I[i][j-1]));
            I[i][j] -= 2 * ((4*alpha + 0.5*dt*V[i][j]) * R[i][j] + alpha * (R[i+1][j] + R[i-1][j] + R[i][j+1] + R[i][j-1]));
        }
    }

    // Propagate the wavepacket through the slit
    for (int i = 0; i < grid_size_x; i++) {
        if (i >= position_of_slit) { // Position of the slit
            // Apply boundary conditions or propagate normally
            R[i][j] += ...;
            I[i][j] -= ...;
        }
    }
}
```
x??

---

#### Square Billiard Wavepacket Motion
Background context: The problem involves determining the motion of an initial Gaussian wavepacket confined to a square billiards table. We need to compare classical and quantum trajectories.

The initial condition is given by Eq. (24.14), and we must simulate both classical and quantum dynamics over time, examining how many reflections it takes for the wavepacket to lose all traces of classical trajectories.

:p Simulate the motion of a Gaussian wavepacket in a square billiard.
??x
To simulate the motion of a Gaussian wavepacket in a square billiard table, we start with an initial condition:
\[ \psi(x,y,t=0) = e^{ik_0 x}e^{ik_0 y} e^{-\frac{(x-x_0)^2}{2\sigma^2}} e^{-\frac{(y-y_0)^2}{2\sigma^2}}, \]
where \(k_0, x_0, y_0, \sigma\) are parameters.

The goal is to observe the classical motion first and then compare it with the quantum dynamics. We need to compute:

1. **Classical Motion**: Simulate the trajectory of a particle bouncing off the walls of the square billiard table.
2. **Quantum Dynamics**: Propagate the wavepacket using the time-dependent Schrödinger equation.

For both cases, we can use the same update formula for the wavefunction:
\[ \psi_{n+1} = \psi_{n-1} + (e^{-i\tilambda \Delta t} - e^{i\tilambda \Delta t}) \psi_n. \]

We then compare the number of reflections required for the wavepacket to lose all traces of classical trajectories.

```java
// Pseudocode for simulating square billiard motion
for (int n = 0; n < total_time_steps; n++) {
    // Update real and imaginary parts of the wavefunction using the time-dependent Schrödinger equation
    for (int i = 1; i < grid_size_x - 1; i++) {
        for (int j = 1; j < grid_size_y - 1; j++) {
            R[i][j] += 2 * ((4*alpha + 0.5*dt*V[i][j]) * I[i][j] - alpha * (I[i+1][j] + I[i-1][j] + I[i][j+1] + I[i][j-1]));
            I[i][j] -= 2 * ((4*alpha + 0.5*dt*V[i][j]) * R[i][j] + alpha * (R[i+1][j] + R[i-1][j] + R[i][j+1] + R[i][j-1]));
        }
    }

    // Check for reflections and update wavefunction accordingly
    if (i == 0 || i == grid_size_x - 1) {
        // Handle boundary conditions
        R[i][j] = ...;
        I[i][j] = ...;
    }
}
```
x??

---

#### Three Disks Scattering System
Background context: The problem involves examining the scattering of a Gaussian wavepacket from various configurations of fixed hard disks, with potential quantum chaotic behavior.

The initial condition is given by Eq. (24.22), and we need to simulate how this wavepacket scatters off different disk configurations.

:p Simulate scattering from three fixed hard disks.
??x
To simulate the scattering from three fixed hard disks, we start with an initial Gaussian wavepacket:
\[ \psi(x,y,t=0) = e^{ik_0 x}e^{ik_0 y} e^{-A(x-x_0)^2 - A(y-y_0)^2}, \]
where \(k_0, x_0, y_0, A\) are parameters.

The goal is to observe the scattering patterns and check for quantum chaotic behavior by varying disk size, wavepacket momentum, and initial position. For three disks:

1. **Single Disk**: Produce a surface plot of the probability density \(z(x,y)\) over time.
2. **Two Disks**: Vary parameters to obtain multiple scatterings and look for trapped orbits.
3. **Three Disks**: Extend to observe many multiple scatterings.

```java
// Pseudocode for simulating three disks scattering
for (int n = 0; n < total_time_steps; n++) {
    // Update real and imaginary parts of the wavefunction using the time-dependent Schrödinger equation
    for (int i = 1; i < grid_size_x - 1; i++) {
        for (int j = 1; j < grid_size_y - 1; j++) {
            R[i][j] += 2 * ((4*alpha + 0.5*dt*V[i][j]) * I[i][j] - alpha * (I[i+1][j] + I[i-1][j] + I[i][j+1] + I[i][j-1]));
            I[i][j] -= 2 * ((4*alpha + 0.5*dt*V[i][j]) * R[i][j] + alpha * (R[i+1][j] + R[i-1][j] + R[i][j+1] + R[i][j-1]));
        }
    }

    // Check for scattering and update wavefunction accordingly
    if (disk1_collision || disk2_collision) {
        R[i][j] = ...;
        I[i][j] = ...;
    }
}
```
x??

--- 

These flashcards cover the key concepts in the provided text, focusing on different aspects of wavepacket dynamics and scattering systems. Each card provides context, relevant formulas, and code examples to facilitate understanding.

