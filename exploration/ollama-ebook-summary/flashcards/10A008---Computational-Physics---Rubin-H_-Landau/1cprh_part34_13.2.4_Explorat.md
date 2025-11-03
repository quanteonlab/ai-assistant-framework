# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 34)

**Starting Chapter:** 13.2.4 Explorations

---

#### Logarithmic Derivative Continuity Condition
Background context: For probability and current to be continuous at \( x = x_m \), both \(\psi(x)\) and its first derivative, \(\psi'(x)\), must be continuous there. Requiring the logarithmic derivative, defined as \(\frac{\psi'(x)}{\psi(x)}\), to be continuous encapsulates these continuity conditions into a single condition.
:p What is the importance of the logarithmic derivative in ensuring the continuity of wave functions?
??x
The logarithmic derivative helps ensure that both \(\psi(x)\) and \(\psi'(x)\) are continuous at \( x = x_m \). This approach avoids dealing with two separate conditions, making it simpler to handle. The ratio \(\frac{\psi'(x)}{\psi(x)}\) is used because it remains independent of the normalization constant.

```python
def log_derivative(psi_prime, psi):
    return psi_prime / psi

# Example usage:
psi_prime = 2 * x - 1  # First derivative of a wave function
psi = x**2 + 3*x + 1  # Wave function
log_derivative_value = log_derivative(psi_prime, psi)
```
x??

#### Good Initial Guess for Ground-State Energy
Background context: The ground-state energy is often sought after using numerical methods. A good initial guess can significantly improve the efficiency of finding the eigenvalue.
:p What is a recommended starting value for the ground-state energy?
??x
A good starting value for the ground-state energy should be slightly above the bottom of the well, \( E > -V_0 \). This ensures that we start within a reasonable range where bound states might exist.

```python
initial_guess = -65  # in MeV, as an example
```
x??

#### Mismatch and Energy Adjustment
Background context: When integrating wave functions from the left and right sides of \( x_m \), they may not match exactly due to numerical approximations. The mismatch is quantified using the logarithmic derivative.
:p How do you measure the mismatch between left and right wave functions?
??x
The mismatch between the left and right wave functions at \( x = x_m \) can be measured by calculating the difference in their logarithmic derivatives:

\[ \Delta(E, x) = \frac{\psi_L'(x)}{\psi_L(x)} - \frac{\psi_R'(x)}{\psi_R(x)} \]

Where:
- \(\psi_L(x)\) and \(\psi_R(x)\) are the left and right wave functions respectively.
- The denominator is included to avoid large or small numbers.

```python
def log_derivative(psi_prime, psi):
    return psi_prime / psi

# Example usage for calculating Δ(E)
left_log_derivative = log_derivative(left_wave_function_prime, left_wave_function)
right_log_derivative = log_derivative(right_wave_function_prime, right_wave_function)
delta_E = left_log_derivative - right_log_derivative
```
x??

#### Numerov Algorithm for Solving Schrödinger Equation
Background context: The Numerov algorithm is a specialized numerical method used to solve second-order differential equations without the first derivative term. It provides higher accuracy and efficiency compared to general methods.
:p What is the Numerov algorithm, and why is it useful?
??x
The Numerov algorithm is a specialized fourth-order method for solving second-order differential equations, particularly suitable when there is no first derivative term in the equation. This makes it especially useful for solving the Schrödinger equation.

The key steps of the Numerov algorithm involve:
1. Using Taylor series expansions to approximate the second derivative.
2. Rewriting the Schrödinger equation using these approximations.
3. Applying a specific operator to eliminate higher-order terms and achieve high accuracy.

```python
def numerov(psi_prev, psi_curr, h):
    # Numerov algorithm formula
    return 2 * (1 - 5/12 * h**2 * k2(x)) * psi_curr \
           - (1 + h**2 / 12 * k2(x - h)) * psi_prev \
           / (1 + h**2 * k2(x) / 12)

# Example usage:
psi_next = numerov(psi_current, psi_previous, h)
```
x??

#### Bisection Algorithm for Finding Eigenvalues
Background context: The bisection method can be combined with an ODE solver to find eigenvalues efficiently. It repeatedly narrows the search range until convergence is achieved.
:p How do you implement a bisection algorithm to find eigenvalues?
??x
To implement a bisection algorithm for finding eigenvalues, follow these steps:
1. Define the initial step size \( h \).
2. Implement a method to calculate the matching function \( \Delta(E, x) \) as a function of energy.
3. Use this method within the bisection algorithm to search for the energy at which \( \Delta(E, x=2) \) vanishes.

Example code:
```python
def find_eigenvalue(h):
    initial_energy = -65  # Initial guess in MeV
    tolerance = 1e-4

    while True:
        # Calculate Δ(E)
        delta_E = calculate_matching_function(initial_energy, h)

        if abs(delta_E) < tolerance:
            break

        if delta_E > 0:
            initial_energy += 0.5 * h
        else:
            initial_energy -= 0.5 * h

    return initial_energy

# Example usage:
eigenvalue = find_eigenvalue(0.04)
print(f"Eigenvalue: {eigenvalue} MeV")
```
x??

#### Iteration Scheme for Solving ODEs

Background context: The problem involves solving an ordinary differential equation (ODE) to find the wave function and potential. The goal is to determine if the solution is a ground state or excited state, and whether it's even or odd about the origin.

:p What is the iteration scheme in this context used for?
??x
The iteration scheme is used to solve the ODE numerically using methods like Numerov or Runge-Kutta (rk4). The scheme iterates until a certain number of iterations are reached, with a warning if it fails. This helps ensure that we find accurate solutions.
```java
// Pseudocode for Iteration Scheme
void iterateOdeSolver(double initialEnergyGuess) {
    int maxIterations = 1000; // Example value
    double tolerance = 1e-6; // Tolerance to check convergence

    for (int i = 0; i < maxIterations; i++) {
        // Solve ODE with current energy guess
        solveOde(initialEnergyGuess);
        
        // Check if solution meets convergence criteria
        if (checkConvergence()) break;

        // If not converged, adjust initial energy and continue
        initialEnergyGuess += 0.1; // Example adjustment

        // Print warning if maximum iterations reached without convergence
        if (i == maxIterations - 1) {
            System.out.println("Iteration scheme failed to converge.");
        }
    }
}
```
x??

---

#### Plotting Wave Function and Potential

Background context: To visualize the solution, we need to plot both the wave function and potential on the same graph. Since they have different scales, one axis will be scaled appropriately.

:p How do you plot the wave function and potential together?
??x
To plot the wave function and potential together, first normalize them such that their ranges are comparable or scale one of them. Use a common x-axis for both functions and set up separate y-axes if necessary.
```java
// Pseudocode for Plotting
void plotWaveFunctionAndPotential(double[] xPoints, double[] waveFunctionValues, double[] potentialValues) {
    // Normalize values if needed
    normalizeValues(waveFunctionValues);
    
    // Set up graph with two y-axes or scale one axis appropriately
    setupGraphWithTwoYAxes();
    
    // Plot the wave function and potential on the same x-axis
    plot(xPoints, waveFunctionValues, "Wave Function", Color.BLUE);
    plot(xPoints, potentialValues, "Potential Energy", Color.RED);
}

void normalizeValues(double[] values) {
    double min = Arrays.stream(values).min().getAsDouble();
    for (int i = 0; i < values.length; i++) {
        values[i] -= min;
    }
}
```
x??

---

#### Identifying Ground State and Excited States

Background context: By counting the nodes in the wave function, we can determine if the solution is a ground state or excited state. The ground state must be even about the origin.

:p How do you identify if the solution is a ground state or an excited state?
??x
To identify whether the solution is a ground state or an excited state, count the number of nodes in the wave function:
- If there are no nodes, it is a ground state.
- If there are one or more nodes, it is an excited state.

Additionally, check if the wave function is even about the origin. The ground state must be even (symmetric).

```java
// Pseudocode for Identifying States
boolean isEvenWaveFunction(double[] xPoints, double[] waveFunctionValues) {
    int size = xPoints.length;
    boolean symmetric = true;

    for (int i = 0; i < size / 2; i++) {
        if (waveFunctionValues[i] != waveFunctionValues[size - 1 - i]) {
            symmetric = false;
            break;
        }
    }

    return symmetric;
}

int countNodes(double[] xPoints, double[] waveFunctionValues) {
    int nodeCount = 0;

    for (int i = 1; i < xPoints.length - 1; i++) {
        if ((waveFunctionValues[i] * waveFunctionValues[i + 1]) < 0) { // Change in sign indicates a node
            nodeCount++;
        }
    }

    return nodeCount;
}

void identifyState(double[] xPoints, double[] waveFunctionValues) {
    int nodes = countNodes(xPoints, waveFunctionValues);
    
    if (nodes == 0) {
        System.out.println("Ground state: Even function");
    } else {
        System.out.println("Excited state with " + nodes + " nodes");
    }
}
```
x??

---

#### Ground State Energy Bar in Potential

Background context: In Figure 13.1a, a horizontal line within the potential indicates the energy of the ground state relative to the potential’s depth.

:p How do you represent the ground state energy bar on the graph?
??x
To add a horizontal line representing the ground state energy bar on the graph:

```java
// Pseudocode for Adding Ground State Energy Bar
void addGroundStateEnergyBar(double[] xPoints, double groundStateEnergy) {
    // Find y-coordinate of the potential at which to draw the ground state bar
    double minPotential = Arrays.stream(potentialValues).min().getAsDouble();
    double energyLineY = groundStateEnergy - minPotential;

    // Draw a horizontal line on the graph for the ground state energy
    drawHorizontalLine(xPoints, 0, xPoints.length - 1, energyLineY);
}

void drawHorizontalLine(double[] xPoints, int start, int end, double y) {
    for (int i = start; i <= end; i++) {
        // Plot a point at each x-coordinate on the line
        plot(xPoints[i], y, "o", Color.BLACK, 1);
    }
}
```
x??

---

#### Finding Excited States

Background context: By increasing the initial energy guess and searching for excited states, we ensure that the wave function is continuous and count nodes to identify each state.

:p How do you search for excited states?
??x
To find excited states:

```java
// Pseudocode for Searching for Excited States
void searchForExcitedStates(double[] xPoints) {
    double initialEnergyGuess = -1; // Start with an arbitrary negative energy value

    while (true) {
        solveOde(initialEnergyGuess);
        
        if (!checkWaveFunctionContinuity()) {
            System.out.println("Wave function not continuous at this energy.");
            break;
        }

        int nodes = countNodes(xPoints, waveFunctionValues);

        if (nodes == 0) { // Ground state
            break; // No more excited states to find
        } else if (nodes > 0) {
            addExcitedStateEnergyBar(xPoints, initialEnergyGuess);
            System.out.println("Excited state with " + nodes + " nodes at energy: " + initialEnergyGuess);
        }

        initialEnergyGuess += 0.1; // Increase the energy guess
    }
}

void addExcitedStateEnergyBar(double[] xPoints, double groundStateEnergy) {
    double minPotential = Arrays.stream(potentialValues).min().getAsDouble();
    double energyLineY = groundStateEnergy - minPotential;

    drawHorizontalLine(xPoints, 0, xPoints.length - 1, energyLineY);
}
```
x??

---

#### Comparing Numerov and RK4 Methods

Background context: To compare the results obtained using both the Numerov and RK4 methods, we need to solve the ODEs with these different numerical schemes.

:p How do you compare the results of Numerov and RK4 methods?
??x
To compare the results:

1. Implement both Numerov and RK4 methods.
2. Solve the same set of differential equations using both methods.
3. Compare the wave functions, energy levels, and computational time taken by each method.

```java
// Pseudocode for Comparing Methods
void compareNumerovRk4(double[] xPoints) {
    // Numerov Method
    double[] numerovWaveFunction = solveOdeUsingNumerov(xPoints);
    
    // RK4 Method
    double[] rk4WaveFunction = solveOdeUsingRK4(xPoints);

    // Plot both wave functions on the same graph to visually compare them
    plot(xPoints, numerovWaveFunction, "Numerov Wave Function", Color.BLUE);
    plot(xPoints, rk4WaveFunction, "RK4 Wave Function", Color.RED);

    // Compare computational time (pseudo-code)
    long startNum = System.currentTimeMillis();
    solveOdeUsingNumerov(xPoints);
    long endNum = System.currentTimeMillis();
    
    long startRk4 = System.currentTimeMillis();
    solveOdeUsingRK4(xPoints);
    long endRk4 = System.currentTimeMillis();

    double numerovTime = (endNum - startNum) / 1000.0;
    double rk4Time = (endRk4 - startRk4) / 1000.0;

    System.out.println("Numerov Time: " + numerovTime + " seconds");
    System.out.println("RK4 Time: " + rk4Time + " seconds");
}
```
x??

---

#### Adding Energy Bars for Excited States

Background context: As excited states are found, add corresponding energy bars to the graph.

:p How do you add an energy bar for each excited state?
??x
To add an energy bar for each excited state:

```java
// Pseudocode for Adding Energy Bar for Each Excited State
void addExcitedStateEnergyBar(double[] xPoints, double groundStateEnergy) {
    double minPotential = Arrays.stream(potentialValues).min().getAsDouble();
    double energyLineY = groundStateEnergy - minPotential;

    drawHorizontalLine(xPoints, 0, xPoints.length - 1, energyLineY);
}

void drawHorizontalLine(double[] xPoints, int start, int end, double y) {
    for (int i = start; i <= end; i++) {
        // Plot a point at each x-coordinate on the line
        plot(xPoints[i], y, "o", Color.BLACK, 1);
    }
}
```
x??

---

#### Determining Wave Function Continuity

Background context: Ensuring that the wave function is continuous helps in identifying valid solutions for excited states.

:p How do you check if the wave function is continuous?
??x
To check if the wave function is continuous:

```java
// Pseudocode for Checking Wave Function Continuity
boolean checkWaveFunctionContinuity(double[] xPoints, double[] waveFunctionValues) {
    int size = waveFunctionValues.length;

    // Check continuity by ensuring no sudden jumps in value
    for (int i = 0; i < size - 1; i++) {
        if (Math.abs(waveFunctionValues[i] - waveFunctionValues[i + 1]) > 1e-6) {
            return false;
        }
    }

    return true;
}
```
x??

---

#### Atan2 Function for Angle Calculation

Background context: The `atan2` function is used to calculate the angle of a vector in the correct quadrant.

:p How do you use atan2 to find the angle?
??x
To use `atan2` to find the angle:

```java
// Pseudocode for Using Atan2 Function
double calculateAngle(double y, double x) {
    return Math.atan2(y, x);
}

void plotTheta(double[] xPoints, double[] yPoints) {
    // Calculate angles at each point
    double[] thetaValues = new double[yPoints.length];
    for (int i = 0; i < yPoints.length; i++) {
        thetaValues[i] = calculateAngle(yPoints[i], xPoints[i]);
    }

    // Plot the angles on a graph
    plot(xPoints, thetaValues, "Theta Values", Color.GREEN);
}
```
x??

---

#### Solving ODE for Numerov Method

Background context: The Numerov method is used to solve second-order linear differential equations with high accuracy.

:p How do you implement the Numerov method to solve an ODE?
??x
To implement the Numerov method, follow these steps:

1. Define the initial conditions and step size.
2. Use the recurrence relation specific to the Numerov method for solving the ODE.

```java
// Pseudocode for Solving Ode Using Numerov Method
double[] solveOdeUsingNumerov(double[] xPoints) {
    int numPoints = xPoints.length;
    double h = (xPoints[numPoints - 1] - xPoints[0]) / (numPoints - 1);
    
    // Initial conditions and initial wave function value
    double[] waveFunctionValues = new double[numPoints];
    waveFunctionValues[0] = 1.0; // Example initial condition
    
    for (int i = 1; i < numPoints - 1; i++) {
        double y_i_minus_1 = waveFunctionValues[i - 1];
        double y_i_plus_1 = waveFunctionValues[i + 1];
        
        double k1 = ...; // Calculate k1 from the ODE
        double k2 = ...; // Calculate k2 from the ODE
        
        waveFunctionValues[i] += h * (3 / 10.0 * k1 - 7 / 5.0 * y_i_minus_1 + 19 / 10.0 * y_i_plus_1);
    }
    
    return waveFunctionValues;
}
```
x??

---

#### Solving ODE for RK4 Method

Background context: The Runge-Kutta (RK4) method is a widely used numerical method to solve ordinary differential equations.

:p How do you implement the RK4 method to solve an ODE?
??x
To implement the RK4 method, follow these steps:

1. Define the initial conditions and step size.
2. Use the fourth-order Runge-Kutta formula for solving the ODE.

```java
// Pseudocode for Solving Ode Using RK4 Method
double[] solveOdeUsingRK4(double[] xPoints) {
    int numPoints = xPoints.length;
    double h = (xPoints[numPoints - 1] - xPoints[0]) / (numPoints - 1);
    
    // Initial conditions and initial wave function value
    double[] waveFunctionValues = new double[numPoints];
    waveFunctionValues[0] = 1.0; // Example initial condition
    
    for (int i = 1; i < numPoints - 1; i++) {
        double k1 = ...; // Calculate k1 from the ODE
        double k2 = ...; // Calculate k2 from the ODE
        double k3 = ...; // Calculate k3 from the ODE
        double k4 = ...; // Calculate k4 from the ODE
        
        waveFunctionValues[i] += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    }
    
    return waveFunctionValues;
}
```
x??

---

#### Sine Wave Function Example

Background context: For demonstration purposes, we can use a sine function to represent the wave function in some examples.

:p How do you plot a simple sine wave as an example?
??x
To plot a simple sine wave:

```java
// Pseudocode for Plotting a Simple Sine Wave
void plotSineWave(double[] xPoints) {
    double[] sineValues = new double[xPoints.length];
    
    // Generate sine values
    for (int i = 0; i < xPoints.length; i++) {
        sineValues[i] = Math.sin(xPoints[i]);
    }
    
    // Plot the sine wave
    plot(xPoints, sineValues, "Sine Wave", Color.BLUE);
}
```
x??

--- 

(Note: The `plot` function and other utility functions are assumed to be part of a plotting library or framework, such as JavaFX or JFreeChart.)

#### RK4 Method for Simultaneous ODEs
Background context explaining the concept. The Runge-Kutta method of order 4 (RK4) is a numerical technique to solve ordinary differential equations (ODEs). For projectile motion with drag, we need to simultaneously solve two coupled first-order ODEs: one for the horizontal position \(x(t)\), and another for the vertical position \(y(t)\).
:p What is the primary method used to solve simultaneous ODEs in this context?
??x
The Runge-Kutta method of order 4 (RK4) is applied. The ODEs are:
\[
\frac{dx}{dt} = v_x, \quad \frac{dy}{dt} = v_y
\]
Where \(v_x\) and \(v_y\) are the velocity components in the x and y directions respectively.
```java
public void rk4(double[] dydt, double t, double h, double[] y) {
    double k1[], k2[], k3[], k4[];
    // Calculate k1 to k4 values using the derivative function f(t,y)
    for (int i = 0; i < dydt.length; i++) {
        k1[i] = h * dydt[i];
        // Similarly, calculate k2, k3, and k4
        y[i] += (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]) / 6.0;
    }
}
```
x??

---

#### Initial Conditions for Projectile Motion with Drag
Background context explaining the concept. The initial conditions are crucial in ensuring that the numerical solution is accurate and reliable. For projectile motion, we start from an initial position \( (b, y_{\infty}) \) where the potential energy is small compared to the total energy.
:p What are the typical initial conditions for solving projectile motion with drag?
??x
The initial conditions typically include a horizontal position \( b \), and a vertical position \( y_{\infty} \) such that:
\[
\left| PE(y_{\infty}) \right| / E \leq 10^{-10}
\]
where \( PE \) is the potential energy, and \( E \) is the total energy. Good starting parameters are provided as:
- Mass \( m = 0.5 \)
- Initial vertical velocity \( v_y(0) = 0.5 \)
- Initial horizontal velocity \( v_x(0) = 0.0 \)
- Small change in \( b \): \( \Delta b = 0.05 \), with \( -1 \leq b \leq 1 \).
x??

---

#### Trajectory Plotting
Background context explaining the concept. Trajectories of a projectile with drag are plotted to observe their behavior under different conditions. These include usual and unusual behaviors such as back-angle scattering, where multiple scatterings are required.
:p What is the objective in plotting trajectories for projectile motion?
??x
The primary objective is to visualize the trajectories of projectiles for both usual and unusual behaviors. Specifically, focus on trajectories where back-angle scattering occurs, indicating significant multiple scatterings.
```java
for (double b = -1; b <= 1; b += 0.05) {
    double[] x = new double[numPoints];
    double[] y = new double[numPoints];
    
    // Use RK4 method to solve ODEs for each initial condition b
    // Store the resulting trajectory in arrays x and y
}
```
x??

---

#### Phase Space Trajectories
Background context explaining the concept. Phase space trajectories are plotted to analyze the dynamics of projectile motion with drag by examining both position and velocity components over time.
:p What is a phase space plot for projectile motion?
??x
A phase space plot shows \([ x(t), ̇x(t) ]\) and \([ y(t), ̇y(t) ]\). These differ from bound state trajectories as they capture the full dynamics, including velocity components over time.
```java
for (double b = -1; b <= 1; b += 0.05) {
    double[] x, y;
    
    // Use RK4 method to solve ODEs for each initial condition b
    // Plot phase space trajectories using the velocity components dx/dt and dy/dt
}
```
x??

---

#### Determining Scattering Angle
Background context explaining the concept. The scattering angle is determined by analyzing the velocity components of the projectile after it has left the interaction region, where \( \frac{PE}{E} \leq 10^{-10} \).
:p How do you determine the scattering angle for a projectile?
??x
The scattering angle \( \theta = \text{atan2}(V_x, V_y) \) is determined by calculating the velocity components of the scattered particle after it has left the interaction region. The condition \( \frac{PE}{E} \leq 10^{-10} \) ensures that the projectile is far enough from the interaction region.
```java
double scatteringAngle = Math.atan2(Vx, Vy);
```
x??

---

#### Identifying Discontinuities in d𝜃/db
Background context explaining the concept. Discontinuities in \( \frac{d\theta}{db} \) and thus \( \sigma(\theta) \) can be identified by analyzing characteristic features of trajectories.
:p What characteristics lead to discontinuities in the scattering angle?
??x
Discontinuities in \( \frac{d\theta}{db} \) are caused by specific trajectory characteristics, such as sharp turns or sudden changes due to multiple scatterings. These discontinuities affect the differential cross-section \( \sigma(\theta) \).
```java
for (double b = -1; b <= 1; b += 0.05) {
    double[] x = new double[numPoints];
    double[] y = new double[numPoints];
    
    // Use RK4 method to solve ODEs for each initial condition b
    // Analyze trajectory characteristics to identify discontinuities in dθ/db
}
```
x??

---

#### Time Delay and Impact Parameter
Background context explaining the concept. The time delay \( T(b) \) is computed as a function of the impact parameter \( b \). High oscillatory regions are identified by plotting \( T(b) \) on a semilog plot, and finer scales are used for detailed analysis.
:p What is time delay in projectile motion?
??x
Time delay \( T(b) \) measures the increase in travel time through an interaction region due to interactions with the potential. High oscillatory regions indicate complex behavior, often requiring finer scale simulations by setting \( b \approx b / 10 \).
```java
double[] impactParams = new double[numBValues];
for (int i = 0; i < impactParams.length; i++) {
    double b = impactParams[i];
    // Use RK4 to simulate projectile motion for each b value and compute T(b)
}
```
x??

---

#### Attractive Potential with Drag
Background context explaining the concept. Simulations are run for both attractive and repulsive potentials, varying energy levels below and above \( V_{max} = \exp(-2) \). The goal is to determine if there is a physics explanation for balls appearing to "fall out of the sky."
:p What is the impact of an attractive potential on projectile motion?
??x
An attractive potential may cause the projectile to be retained or significantly altered in its trajectory. Detailed simulations are needed to observe the behavior, including back-angle scattering and multiple interactions.
```java
public void simulateProjectile(double[] y0, double k) {
    // Use RK4 method with attractive potential force F(f)
}
```
x??

---

#### Ball Trajectories Without Air Resistance
Background context explaining the concept. The trajectories of a projectile are compared when air resistance is ignored versus when it is included. The analytical solution for frictionless motion provides a baseline.
:p How do ball trajectories differ with and without air resistance?
??x
With air resistance, balls can appear to "fall out of the sky" at the end of their trajectory due to drag effects. Analytical solutions show that air resistance modifies the parabolic path, making it asymmetric and leading to unusual behavior.
```java
public double y(double t) {
    return V0 * Math.sin(theta) * t - 0.5 * g * t * t;
}

public double x(double t) {
    return V0 * Math.cos(theta) * t;
}
```
x??

---

#### Modifying RK4 Program for Friction
Background context explaining the concept. The Runge-Kutta method is modified to solve ODEs with friction, using \( n \) values of 1, 3/2, and 2 to model different velocities.
:p How do you modify the RK4 program for projectile motion with drag?
??x
The RK4 program is adapted to solve the coupled ODEs:
\[
\frac{dx}{dt} = v_x, \quad \frac{dy}{dt} = v_y
\]
Where \( n \) represents different models of air resistance. For low velocities (\( n=1 \)), medium (\( n=3/2 \)), and high (\( n=2 \)) velocities, appropriate values of \( k \) are adjusted to ensure the initial friction force is consistent.
```java
public void rk4Friction(double[] dydt, double t, double h, double[] y) {
    // Adjust equations for friction based on n value
}
```
x??

