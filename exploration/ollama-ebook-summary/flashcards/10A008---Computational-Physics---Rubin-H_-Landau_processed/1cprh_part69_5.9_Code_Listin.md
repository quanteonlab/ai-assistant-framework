# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 69)

**Starting Chapter:** 5.9 Code Listings

---

#### Monte Carlo Integration for 10D Problems
Monte Carlo integration is a statistical technique to estimate definite integrals. The method involves generating random points within the domain of integration and then averaging the function values at these points. For high-dimensional problems, this can be computationally intensive.

The provided text suggests using a built-in random-number generator to perform 10D Monte Carlo integration. The objective is to conduct 16 trials and take the average as your answer. Different sample sizes \( N = 2, 4, 8, \ldots, 8192 \) are used, and relative error versus \( 1/\sqrt{N} \) is plotted to check for linear behavior.

:p How can we conduct 10D Monte Carlo integration with a random-number generator?
??x
To conduct 10D Monte Carlo integration:
1. Generate random points in the 10-dimensional space.
2. Evaluate the function at each point.
3. Compute the average of these evaluations.
4. Repeat this process for multiple trials (16 in this case) and take their average as the final answer.

For example, if you are integrating a function \( f \) over the domain \( [a_1, b_1] \times [a_2, b_2] \times \ldots \times [a_{10}, b_{10}] \), each random point \( (x_1, x_2, \ldots, x_{10}) \) is generated with coordinates uniformly distributed within the respective intervals. The integral can be approximated as:
\[ \int_{[a_1,b_1]}\cdots\int_{[a_{10},b_{10}]} f(x_1, x_2, \ldots, x_{10}) dx_1 dx_2 \cdots dx_{10} \approx V \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_1^i, x_2^i, \ldots, x_{10}^i) \]
where \( V = (b_1 - a_1)(b_2 - a_2) \cdots (b_{10} - a_{10}) \).

This process is repeated 16 times and the average result is taken as the final answer.
x??

---
#### Linear Behavior of Relative Error
The relative error in Monte Carlo integration decreases with \( \sqrt{N} \). For high-dimensional integrations, plotting the relative error versus \( 1/\sqrt{N} \) should show a linear behavior.

:p How does plotting relative error versus \( 1/\sqrt{N} \) help us understand the accuracy of our integration?
??x
Plotting the relative error versus \( 1/\sqrt{N} \) helps to check if the Monte Carlo method follows the expected convergence rate, which is theoretically \( O(1/\sqrt{N}) \). If the behavior is linear, it suggests that the method is performing as expected.

For example, consider the relative error \( E \):
\[ E = \left| \frac{\text{MC integral} - \text{True integral}}{\text{True integral}} \right| \]
When plotted against \( 1/\sqrt{N} \), a linear trend indicates that the method is converging as expected. This can be verified by generating multiple samples and observing if the error decreases proportionally to \( 1/\sqrt{N} \).

To visualize this, you would generate several values of \( N \) (e.g., powers of 2 up to 8192), compute the Monte Carlo integral for each \( N \), calculate the relative error, and plot these points. If the plot is linear, it confirms the accuracy of the method.
x??

---
#### Multidimensional Integration
Monte Carlo integration can be extended to multidimensional problems by generating random points in the multidimensional space.

:p How does Monte Carlo integration work for 10D problems?
??x
For a 10D problem, Monte Carlo integration involves:
1. Generating \( N \) random points uniformly distributed in the 10-dimensional domain.
2. Evaluating the function at each of these points.
3. Averaging these evaluations to approximate the integral.

The exact formula for a 10D integral is:
\[ \int_{[a_1, b_1]}\cdots\int_{[a_{10}, b_{10}]} f(x_1, x_2, \ldots, x_{10}) dx_1 dx_2 \cdots dx_{10} \approx V \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_1^i, x_2^i, \ldots, x_{10}^i) \]
where \( V = (b_1 - a_1)(b_2 - a_2) \cdots (b_{10} - a_{10}) \).

The process is repeated for multiple trials to get an average estimate of the integral.
x??

---
#### Relative Error Calculation
Relative error is calculated by comparing the estimated integral from Monte Carlo integration with the true value.

:p How do we calculate the relative error in Monte Carlo integration?
??x
The relative error \( E \) is calculated as:
\[ E = \left| \frac{\text{MC integral} - \text{True integral}}{\text{True integral}} \right| \]

For example, if you have a true integral value of 10 and your Monte Carlo estimate is 9.8, the relative error would be:
\[ E = \left| \frac{9.8 - 10}{10} \right| = 0.2 \text{ or } 20\% \]

This can be computed for multiple trials to get an average relative error.
x??

---
#### Linear Relationship Between Relative Error and \( 1/\sqrt{N} \)
Plotting the relative error versus \( 1/\sqrt{N} \) helps in understanding the convergence behavior of Monte Carlo integration.

:p Why do we plot relative error versus \( 1/\sqrt{N} \)?
??x
We plot relative error versus \( 1/\sqrt{N} \) to check if the Monte Carlo method converges at the expected rate, which is theoretically \( O(1/\sqrt{N}) \). A linear relationship indicates that as \( N \) increases, the relative error decreases proportionally.

For example:
- Generate multiple values of \( N \).
- Compute the Monte Carlo integral for each \( N \).
- Calculate the relative error.
- Plot the relative error against \( 1/\sqrt{N} \).

If the plot is linear, it confirms that the method is converging as expected. This helps in assessing the accuracy and efficiency of the integration process.
x??

---
#### Code Example for Monte Carlo Integration
The provided code example demonstrates how to perform a simple one-dimensional Monte Carlo integration using random sampling.

:p Can you explain the code snippet provided for one-dimensional Monte Carlo integration?
??x
Sure! The code performs a simple one-dimensional Monte Carlo integration:

```python
import random

def fx(x):
    return x * sin(x) * sin(x)

# Plot function
N = 100
graph = display(width=500, height=500, title='vonNeumann Rejection Int')
xsinx = curve(x=list(range(0, N)), color=color.yellow, radius=0.5)
pts = label(pos=(-60, -60), text='points=', box=0)
inside = label(pos=(30, -60), text='accepted=', box=0)
arealbl = label(pos=(-65, 60), text='area=', box=0)
areanal = label(pos=(30, 60), text='analytical=', box=0)
zero = label(pos=(-85, -48), text='0', box=0)
five = label(pos=(-85, 50), text='5', box=0)
twopi = label(pos=(90, -48), text='2pi', box=0)

def plotfunc():
    incr = 2.0 * pi / N
    for i in range(0, N):
        xx = i * incr
        xsinx.x[i] = ((80.0 / pi) * xx - 80)
        xsinx.y[i] = 20 * fx(xx) - 50

box = curve(pos=[(-80, -50), (-80, 50), (80, 50), (80, -50), (-80, -50)], color=color.white)
plotfunc()

area = 2.0 * pi * 5.0
analyt = pi ** 2

genpts = points(size=2)
for i in range(1, N):
    x = 2.0 * pi * random.random()
    y = 5 * random.random()
    xp = x * 80.0 / pi - 80
    yp = 20.0 * y - 50
    pts.text = f'points={i:4d}'
    
    if y <= fx(x):
        inside.text = f'accepted={j:4d}'
        genpts.append(pos=(xp, yp), color=color.cyan)
        j += 1
    else:
        genpts.append(pos=(xp, yp), color=color.green)

boxarea = 2.0 * pi * 5.0
area = boxarea * j / (N - 1)
arealbl.text = f'analytical={percent8.5f}' % analyt
areanal.text = f'area= {percent8.5f}' % area
```

This code:
1. Defines the function \( f(x) \).
2. Sets up a graphical display to plot the function.
3. Generates points within the specified range and evaluates the function at these points.
4. Compares the evaluated function values with the y-axis to decide if the point is below or above the curve, thus determining acceptance.
5. Updates labels for the number of accepted points and the estimated area.

The relative error can be calculated based on this estimated area compared to the analytical solution.
x??

--- 
#### Multidimensional Integration using Python
The example uses a simple one-dimensional function but can be extended to higher dimensions by generating multidimensional random points.

:p How would you extend this code for 10D Monte Carlo integration?
??x
Extending the code for 10D Monte Carlo integration involves:
1. Generating 10-dimensional random points.
2. Evaluating a 10D function at these points.
3. Averaging the results to estimate the integral.

Here's an example of how you could modify the code:

```python
import random

def f(x):
    # Define your 10D function here
    return sum(xi ** 2 for xi in x) / 10  # Example: Average of squares

N = 10000  # Number of samples
dim = 10   # Dimensionality of the problem
total_area = (2 * pi) ** dim  # Volume of the domain
sum_f = 0.0

for _ in range(N):
    point = [random.uniform(-pi, pi) for _ in range(dim)]  # Generate a 10D random point
    sum_f += f(point)

integral_estimate = total_area * (sum_f / N)
print(f"Estimated integral: {integral_estimate}")

# Optionally, plot the results or perform error analysis as needed.
```

This code:
1. Defines a simple 10D function \( f(x_1, x_2, \ldots, x_{10}) \).
2. Generates random points in 10-dimensional space.
3. Evaluates the function at each point and sums these values.
4. Averages the sum to estimate the integral.

The volume of the domain is calculated as \( (2\pi)^{10} \) for simplicity, but you should adjust this based on your actual integration bounds.
x??

--- 
#### Code Example for High-Dimensional Integration
The provided code can be adapted to handle high-dimensional integrals by adjusting the dimensionality and sample size.

:p How would you modify the provided code to perform 10D Monte Carlo integration?
??x
To adapt the provided code for 10D Monte Carlo integration, follow these steps:
1. Adjust the function to accept a 10-tuple.
2. Generate random points in 10-dimensional space.
3. Evaluate the function at each point.
4. Sum and average the results.

Here's an example:

```python
import random

def f(x):
    # Define your 10D function here
    return sum(xi ** 2 for xi in x) / 10  # Example: Average of squares

N = 10000  # Number of samples
dim = 10   # Dimensionality of the problem
sum_f = 0.0

for _ in range(N):
    point = [random.uniform(-pi, pi) for _ in range(dim)]  # Generate a 10D random point
    sum_f += f(point)

integral_estimate = (2 * pi) ** dim * (sum_f / N)
print(f"Estimated integral: {integral_estimate}")
```

This code:
1. Defines the function \( f(x_1, x_2, \ldots, x_{10}) \).
2. Sets the number of samples and dimensionality.
3. Generates random 10D points.
4. Evaluates the function at each point and sums these values.
5. Averages the sum to estimate the integral.

The volume of the domain is calculated as \( (2\pi)^{10} \) for simplicity, but you should adjust this based on your actual integration bounds.
x??

--- 
#### Relative Error Plotting
Plotting relative error against \( 1/\sqrt{N} \) helps in understanding the convergence behavior.

:p How would you plot the relative error against \( 1/\sqrt{N} \) for multiple trials?
??x
To plot the relative error against \( 1/\sqrt{N} \), follow these steps:
1. Perform multiple Monte Carlo integrations with increasing sample sizes.
2. Calculate the relative error for each trial.
3. Plot the relative errors on a graph.

Here’s an example in Python:

```python
import numpy as np

def f(x):
    return x * np.sin(x)

N_values = [2**i for i in range(1, 15)]  # Sample sizes from 2 to 2^14
relative_errors = []

for N in N_values:
    sum_f = 0.0
    
    for _ in range(N):
        x = random.uniform(-np.pi, np.pi)
        sum_f += f(x)
    
    integral_estimate = (2 * np.pi) * (sum_f / N)
    true_integral = -1  # Example: True value of the integral over [-pi, pi]
    relative_error = abs((integral_estimate - true_integral) / true_integral)
    relative_errors.append(relative_error)

# Plotting
import matplotlib.pyplot as plt

N_sqrt = np.sqrt(N_values)
plt.plot(1 / N_sqrt, relative_errors, marker='o')
plt.xlabel('1/sqrt(N)')
plt.ylabel('Relative Error')
plt.title('Convergence of Monte Carlo Integration')
plt.grid(True)
plt.show()
```

This code:
1. Defines the function \( f(x) \).
2. Sets a range of sample sizes.
3. Performs Monte Carlo integration for each size.
4. Calculates the relative error for each trial.
5. Plots the relative errors against \( 1/\sqrt{N} \).

By observing if the plot is linear, you can verify the convergence behavior and accuracy of the method.
x??

--- 
#### Code Example for Plotting Relative Error
The provided code demonstrates how to calculate and plot the relative error for multiple trials of Monte Carlo integration.

:p Can you explain the process of plotting relative error against \( 1/\sqrt{N} \) in more detail?
??x
Certainly! Here’s a detailed explanation of the steps involved in calculating and plotting the relative error for multiple trials of Monte Carlo integration:

### Steps to Calculate Relative Error

1. **Define the Function:**
   - Define the function you want to integrate.

2. **Set Sample Sizes:**
   - Create an array of sample sizes, starting from a small value (e.g., \( N = 2 \)) and increasing exponentially (e.g., up to \( N = 2^{14} \)).

3. **Perform Monte Carlo Integration for Each Sample Size:**
   - For each sample size \( N \), perform the integration by generating random points and evaluating the function at these points.
   - Sum the evaluated function values and average them to get an estimate of the integral.

4. **Calculate True Value (if known):**
   - Calculate or know the true value of the integral for comparison.

5. **Compute Relative Error:**
   - For each sample size, compute the relative error using the formula:
     \[
     E = \left| \frac{\text{MC integral} - \text{True integral}}{\text{True integral}} \right|
     \]

### Steps to Plot Relative Error

1. **Compute \( 1/\sqrt{N} \):**
   - For each sample size, compute the value of \( 1/\sqrt{N} \).

2. **Store Relative Errors:**
   - Store the computed relative errors for plotting.

3. **Plotting:**
   - Use a plot to visualize the relationship between \( 1/\sqrt{N} \) and the relative error.
   - The plot helps in assessing if the method is converging as expected.

### Example Code

Here’s an example code that illustrates these steps:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function to integrate (example: f(x) = x * sin(x))
def f(x):
    return x * np.sin(x)

# Set sample sizes
N_values = [2**i for i in range(1, 15)]  # Sample sizes from 2 to 2^14

relative_errors = []

for N in N_values:
    sum_f = 0.0
    
    for _ in range(N):
        x = random.uniform(-np.pi, np.pi)
        sum_f += f(x)
    
    integral_estimate = (2 * np.pi) * (sum_f / N)
    
    # True value of the integral over [-pi, pi] (example: -1)
    true_integral = -1.0
    relative_error = abs((integral_estimate - true_integral) / true_integral)
    relative_errors.append(relative_error)

# Compute 1/sqrt(N) for plotting
N_sqrt = np.sqrt(np.array(N_values))

# Plotting the results
plt.plot(1 / N_sqrt, relative_errors, marker='o')
plt.xlabel('1/sqrt(N)')
plt.ylabel('Relative Error')
plt.title('Convergence of Monte Carlo Integration')
plt.grid(True)
plt.show()
```

### Explanation

- **Step 1:** The function `f(x)` is defined. This example uses \( f(x) = x \sin(x) \).
- **Step 2:** Sample sizes are set using a list comprehension, starting from \( N = 2 \) to \( N = 2^{14} \).
- **Step 3:** For each sample size, random points are generated and the function is evaluated. The integral estimate is computed by averaging these values.
- **Step 4:** The true value of the integral is set (in this example, it's known to be -1). Relative errors are calculated for each trial.
- **Step 5:** The relative errors are plotted against \( 1/\sqrt{N} \).

By observing if the plot shows a linear relationship, you can verify that the method is converging as expected. This helps in understanding the accuracy and efficiency of the Monte Carlo integration method.
x??

#### Bisection Search Overview
Background context explaining the bisection search algorithm. This technique is used to find a value of \( x \) for which \( f(x) \approx 0 \). The algorithm works by repeatedly dividing an interval in half and narrowing down the location of the root.

The bisection method is reliable but relatively slow. It requires knowing an initial interval where the function changes sign, meaning that \( f(a) < 0 \) and \( f(b) > 0 \), or vice versa. The algorithm then narrows this interval by evaluating the function at the midpoint of the current interval.

:p What is the bisection search algorithm used for?
??x
The bisection search algorithm is used to find a root (value of \( x \)) where \( f(x) \approx 0 \). It works by repeatedly dividing an interval in half and narrowing down the location of the root based on sign changes.
x??

---

#### Bisection Search Algorithm Steps
Explanation: The bisection search algorithm starts with an initial interval \([a, b]\) where the function \( f(x) \) changes sign. It then iteratively divides this interval in half and narrows down to a smaller subinterval that contains the root.

:p How does the bisection algorithm work?
??x
The bisection algorithm works by starting with an initial interval \([a, b]\) where \( f(a) \) and \( f(b) \) have opposite signs. It then repeatedly halves this interval, evaluating the function at the midpoint to determine which half contains the root.
x??

---

#### Bisection Search Pseudocode
Explanation: The following pseudocode outlines the logic of the bisection algorithm.

```pseudocode
function bisectionSearch(f, a, b, tolerance):
    if f(a) * f(b) >= 0:
        return "No root found in this interval"
    
    while (b - a) / 2.0 > tolerance:
        c = (a + b) / 2.0
        if f(c) == 0:
            return c
        else if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2.0
```

:p What is the pseudocode for implementing the bisection search algorithm?
??x
The pseudocode for implementing the bisection search algorithm is as follows:

```pseudocode
function bisectionSearch(f, a, b, tolerance):
    if f(a) * f(b) >= 0:
        return "No root found in this interval"
    
    while (b - a) / 2.0 > tolerance:
        c = (a + b) / 2.0
        if f(c) == 0:
            return c
        else if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2.0
```
This algorithm checks the initial interval and then iteratively narrows it down based on the function evaluations.
x??

---

#### Bisection Search Example in Python
Explanation: The following example demonstrates how to implement the bisection search algorithm in Python.

```python
def f(x):
    return 10 - x * (x / 2)

def bisection_search(f, a, b, tol=1e-5):
    if f(a) * f(b) >= 0:
        print("No root found in this interval")
        return None
    
    while (b - a) > tol:
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2

# Example usage
result = bisection_search(f, -3, 5)
print("Root found at:", result)
```

:p How would you implement the bisection search algorithm in Python?
??x
The implementation of the bisection search algorithm in Python is as follows:

```python
def f(x):
    return 10 - x * (x / 2)

def bisection_search(f, a, b, tol=1e-5):
    if f(a) * f(b) >= 0:
        print("No root found in this interval")
        return None
    
    while (b - a) > tol:
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2

# Example usage
result = bisection_search(f, -3, 5)
print("Root found at:", result)
```
This Python implementation checks the initial interval and iteratively narrows it down based on function evaluations until the desired tolerance is met.
x??

---

#### Bisection Search Applied to Quantum Bound States
Explanation: The provided text discusses applying the bisection search algorithm to find the energies of bound states in a 1D square well potential. The goal is to solve for \( E_B \) using transcendental equations.

The problem involves solving for even and odd wave functions within a square well potential defined by:
\[ V(x) = \begin{cases} 
-10 & \text{for } |x| \leq 1 \\
0 & \text{for } |x| > 1
\end{cases} \]

The energies \( E_B < 0 \) are solutions to the transcendental equations:
\[ \sqrt{10 - E_B} \tan(\sqrt{10 - E_B}) = \sqrt{E_B} \quad (\text{even}), \]
\[ \sqrt{10 - E_B} \cot(\sqrt{10 - E_B}) = \sqrt{E_B} \quad (\text{odd}). \]

:p How is the bisection search algorithm applied to find bound state energies in a 1D square well?
??x
The bisection search algorithm is applied to find bound state energies \( E_B < 0 \) for both even and odd wave functions within a 1D square well potential. The algorithm solves transcendental equations that determine the energy levels by narrowing down intervals where the function changes sign.
x??

---

#### Bisection Search with Different Potential Depths
Explanation: The provided text also discusses how changing the depth of the potential affects the number of bound states and their energies.

:p How does changing the potential depth affect the number of bound states in a 1D square well?
??x
Changing the potential depth (e.g., from 10 to 20 or 30) affects the number of bound states and their energies. A deeper potential well leads to more discrete energy levels, as the particles have less room to move and thus experience stronger confinement.
x??

---

#### Bisection Method Overview
The bisection method is a simple root-finding technique that works by repeatedly bisecting an interval and then selecting a subinterval in which a root must lie for further processing. It starts with two points, `plus` and `minus`, where the function values have opposite signs (indicating there's at least one root between them). The method continues to halve the interval until it finds the root within a specified precision.

:p What is the basic idea behind the bisection method?
??x
The basic idea of the bisection method is to repeatedly bisect an interval and select subintervals in which the function changes sign, thus narrowing down the location of the root. This process continues until the value of \(f(x)\) is less than a predefined level of precision or a large number of subdivisions occur.
x??

---
#### Implementation Steps for Bisection Method
The bisection method involves evaluating the function at the midpoint of an interval and adjusting the interval based on the sign change. This process is repeated until the root is found within a desired accuracy.

:p How does the pseudocode for the bisection method look like?
??x
```pseudocode
function bisection(f, plus, minus, tol):
    while (plus - minus) / 2 > tol:
        x = (plus + minus) / 2
        if f(plus) * f(x) < 0:
            minus = x
        else:
            plus = x
    return (plus + minus) / 2
```
x??

---
#### Handling Singularities in Bisection Method
The tangent function \( \tan(\theta) \) has singularities, so the original equation can be transformed to avoid these issues. For instance, the given equation can be rewritten as:
\[ f(E) = \sqrt{10 - E}B \tan\left(\sqrt{10 - E}\right) - \sqrt{E B} = 0 \]
This form may not handle singularities well around \( \sqrt{10 - E} = n\pi \), where \( n \) is an integer. An alternative, equivalent equation can be used:
\[ f(E) = \sqrt{E} \cot\left(\sqrt{10 - E}\right) - \sqrt{10 - E} = 0 \]

:p How can the original equation be transformed to avoid singularities?
??x
The original equation can be transformed by using an equivalent form that avoids singularities:
\[ f(E) = \sqrt{E} \cot\left(\sqrt{10 - E}\right) - \sqrt{10 - E} = 0. \]
This form has different singularities, which are easier to handle numerically.
x??

---
#### Newton-Raphson Method Overview
The Newton-Raphson method is an iterative technique for finding the roots of a function by approximating the function with its tangent line at each iteration and solving for the root where this tangent crosses the x-axis. The key idea is to start from an initial guess \( x_0 \) and refine it using:
\[ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}. \]
This method converges much faster than the bisection method, but it requires the derivative of the function.

:p What is the formula for the Newton-Raphson method?
??x
The formula for the Newton-Raphson method is:
\[ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}. \]
This process iteratively refines the guess until a root is found within a specified precision.
x??

---
#### Central Difference Approximation in Newton-Raphson
In cases where an analytical derivative is not available, a numerical approximation can be used. The forward difference formula for the derivative is:
\[ f'(x) \approx \frac{f(x + \delta x) - f(x)}{\delta x}. \]
For simplicity, central differences could also be used but would require more function evaluations.

:p How does one approximate the derivative in Newton-Raphson using a forward difference?
??x
The derivative can be approximated using the forward difference formula:
\[ f'(x) \approx \frac{f(x + \delta x) - f(x)}{\delta x}. \]
Here, \( \delta x \) is a small change in \( x \).
x??

---
#### Example of Central Difference Approximation
In Listing 6.2, the program `NewtonCD.py` implements the Newton-Raphson method using a central difference approximation for the derivative:
```python
def newton_raphson(f, dfdx, x0, tol):
    while True:
        fx = f(x0)
        if abs(fx) < tol: 
            break
        dfdx_x0 = (f(x0 + 1e-6) - f(x0)) / 1e-6  # Central difference approximation
        dx = -fx / dfdx_x0
        x0 += dx
    return x0
```

:p What does the `NewtonCD.py` program do?
??x
The `NewtonCD.py` program implements the Newton-Raphson method using a central difference approximation for the derivative. It iteratively refines an initial guess until the function value is within a specified tolerance.
x??

---

