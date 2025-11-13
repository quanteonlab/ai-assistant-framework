# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 11)

**Starting Chapter:** 5.9 Code Listings

---

#### Monte Carlo Integration Overview
Background context: Monte Carlo integration is a numerical method for estimating definite integrals using random sampling. It's particularly useful when traditional methods like Simpson's rule or Gaussian quadrature are difficult to apply due to high dimensions or complex functions.

:p What is Monte Carlo integration?
??x
Monte Carlo integration uses random sampling to estimate the value of an integral. The process involves generating points within the domain of integration and using these points to approximate the area under a curve.
x??

---

#### 10D Monte Carlo Integration with Random Sampling
Background context: Performing Monte Carlo integration in higher dimensions (such as 10D) requires careful sampling techniques due to the curse of dimensionality. The goal is to estimate the integral by averaging over many trials.

:p How can you perform a 10D Monte Carlo integration using random numbers?
??x
To perform a 10D Monte Carlo integration, generate points randomly within the 10-dimensional space and evaluate the function at these points. Repeat this process for multiple trials (e.g., 16) and take the average as the final result.

```python
import numpy as np

def monte_carlo_integration(f, dim, N, trials):
    error_sum = 0
    for _ in range(trials):
        points = np.random.rand(N, dim)
        integral = np.mean([f(point) for point in points])
        error_sum += (integral - f(np.ones(dim)))**2
    
    average_error = error_sum / trials
    return average_error

# Example usage:
def integrand(x): 
    return x[0]**2 + 2*x[1]**3

result = monte_carlo_integration(integrand, 10, 8192, 16)
print(result)
```
x??

---

#### Variance Reduction in Monte Carlo Integration
Background context: Variance reduction techniques aim to improve the accuracy of Monte Carlo integration by reducing the variance of the random samples. This is crucial for functions with rapid variations.

:p What are variance reduction techniques in Monte Carlo integration?
??x
Variance reduction techniques involve transforming the integrand into a function that has a smaller variance, making it easier to integrate accurately. This can be achieved by constructing a simpler function close to the original one or by using importance sampling.

```python
def variance_reduction(f, g, w):
    integral = 0
    for _ in range(1000):  # Number of trials
        x = np.random.rand()
        integral += (f(x) - g(x)) * w(x)
    
    return integral

# Example usage:
def f(x): 
    return x**2 + np.sin(5*x)

def g(x):
    return x**3

def w(x):
    return 1 / (g(x) - f(x))

result = variance_reduction(f, g, w)
print(result)
```
x??

---

#### Importance Sampling in Monte Carlo Integration
Background context: Importance sampling is a method to improve the efficiency of Monte Carlo integration by focusing on regions where the integrand has higher values. This technique involves sampling from a distribution that matches the shape of the integrand.

:p What is importance sampling?
??x
Importance sampling is a variance reduction technique in Monte Carlo integration where samples are drawn from a probability distribution that gives more weight to important regions, thus reducing variance and improving accuracy.

```python
def importance_sampling(f, q, w, N):
    sample = np.random.normal(size=N)
    weights = [w(x) for x in sample]
    integral = sum([f(x) * w(x) / q.pdf(x) for x in sample]) / N
    
    return integral

# Example usage:
from scipy.stats import norm
import numpy as np

def f(x):
    return x**2 + 2*np.sin(3*x)

q = norm(loc=0, scale=1)
w = lambda x: q.pdf(x) * (x**2 + 2*np.sin(3*x))

result = importance_sampling(f, q, w, 10000)
print(result)
```
x??

---

#### Graphical Representation of Monte Carlo Integration
Background context: The provided code demonstrates a graphical approach to Monte Carlo integration using the von Neumann rejection method. This method involves plotting the function and randomly generating points within a bounding box.

:p How does the graphically represented Monte Carlo integration work?
??x
The method generates random points within a predefined area (usually a rectangle) and counts how many fall under the curve of the integrand. The ratio of accepted points to total points gives an estimate of the integral's value.

```python
import numpy as np

def monte_carlo_integration_graphically(f, min_val, max_val, N):
    count = 0
    for _ in range(N):
        x = np.random.uniform(min_val, max_val)
        y = f(x) + np.random.uniform(0, f(x))
        
        if y <= f(x):
            count += 1
    
    area = (max_val - min_val) * f(max_val)
    estimated_integral = area * count / N
    return estimated_integral

# Example usage:
def f(x): 
    return x**2 + np.sin(5*x)

result = monte_carlo_integration_graphically(f, 0, 10, 10000)
print(result)
```
x??

---

#### Gaussian Quadrature vs. Monte Carlo Integration
Background context: Gaussian quadrature is a deterministic method for numerical integration that uses weighted sums of function values at specific points. In contrast, Monte Carlo methods rely on random sampling.

:p How does Gaussian quadrature differ from Monte Carlo integration?
??x
Gaussian quadrature approximates the integral by evaluating the integrand at specific points (nodes) and multiplying these evaluations by weights. These nodes are chosen to maximize the accuracy of the approximation for polynomials up to a certain degree. In contrast, Monte Carlo integration uses random sampling across the entire domain.

```python
from scipy.integrate import quad

def gaussian_quadrature(f, a, b):
    result, error = quad(f, a, b)
    return result

# Example usage:
def f(x): 
    return x**2 + 2*np.sin(3*x)

result = gaussian_quadrature(f, 0, 10)
print(result)
```
x??

---

#### Summary of Concepts
---
#### Key Points Recap
- **Monte Carlo Integration**: Uses random sampling to estimate integrals.
- **Variance Reduction Techniques**: Reduce variance by transforming the function or using importance sampling.
- **Importance Sampling**: Focuses on regions where the integrand has higher values.
- **Graphical Monte Carlo**: Visually represents the process of estimating an integral.

:p What are the key concepts covered in this flashcard set?
??x
The key concepts covered include:
- The basics of Monte Carlo integration and its application.
- Variance reduction techniques such as importance sampling.
- Graphical representation of Monte Carlo methods through rejection sampling.
- Comparison with deterministic methods like Gaussian quadrature.

These concepts help in understanding different ways to approach numerical integration, especially in complex or high-dimensional scenarios.
x??

#### Trial-and-Error Searching Overview
Background context: This chapter introduces techniques for solving equations via trial-and-error search methods. These methods are particularly useful when well-defined algorithms leading to definite outcomes are not available, and involve making internal decisions on what steps to follow during the search process.

:p What is the main focus of this section?
??x
The main focus is on using trial-and-error searching techniques for solving equations and fitting curves to data. This includes methods like bisection search.
x??

---

#### Quantum Bound States I
Background context: The problem involves finding the energies of a particle bound within a 1D square well. The potential $V(x)$ is defined as:
$$V(x) = \begin{cases} 
-10, & \text{for } |x| \leq a \\
0, & \text{for } |x| > a
\end{cases}$$

The energies of the bound states $E_B < 0$ are solutions to transcendental equations given by:
$$\sqrt{10 - E_B} \tan(\sqrt{10 - E_B}) = \sqrt{E_B} \quad (\text{even}),$$
$$\sqrt{10 - E_B} \cot(\sqrt{10 - E_B}) = \sqrt{E_B} \quad (\text{odd}).$$:p What are the transcendental equations used to find the bound states energies in a 1D square well?
??x
The transcendental equations used to find the bound state energies in a 1D square well are:
$$\sqrt{10 - E_B} \tan(\sqrt{10 - E_B}) = \sqrt{E_B} \quad (\text{even}),$$
$$\sqrt{10 - E_B} \cot(\sqrt{10 - E_B}) = \sqrt{E_B} \quad (\text{odd}).$$x??

---

#### Bisection Search Algorithm
Background context: The bisection search algorithm is a reliable but slow trial-and-error method that finds roots of functions by repeatedly dividing intervals in half and selecting subintervals where the function changes sign.

:p What is the basis of the bisection algorithm?
??x
The basis of the bisection algorithm involves starting with an interval $[x_-, x_+]$ where $f(x_-)$ and $f(x_+)$ have opposite signs. The algorithm repeatedly divides this interval in half, choosing the subinterval where the function changes sign until a root is found within a desired precision.

```python
def bisection(f, a, b, tol):
    if f(a) * f(b) > 0:
        print("f(a) and f(b) do not have opposite signs")
        return None
    
    c = a
    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0
        if f(c) == 0:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return c
```
x??

---

#### Bisection Search in Practice: Finding Bound States Energies
Background context: Given the function $\sqrt{10 - E_B} \tan(\sqrt{10 - E_B}) - \sqrt{E_B} = 0 $ for even wave functions, and$\sqrt{10 - E_B} \cot(\sqrt{10 - E_B}) - \sqrt{E_B} = 0$ for odd wave functions. The algorithm starts with an interval where the function changes sign.

:p How do you apply the bisection search to find bound state energies?
??x
To apply the bisection search, start by finding an initial interval $[a, b]$ where $ f(a) < 0 $ and $ f(b) > 0 $. For example:
1. Evaluate $f(0)$ and $f(10)$:
   - If $f(0) = 3 $(positive) and $ f(10) = -2 $(negative), then choose the interval$[0, 10]$.
   
2. Compute the midpoint $c = (a + b) / 2 $. Check if $ f(c)$ is close to zero or changes sign.

3. Repeat this process:
   - If $f(a) * f(c) < 0 $, then the root lies between $ a $and$ c$.
   - Otherwise, it lies between $c $ and$b$.

4. Continue until $|b - a|$ is less than the desired tolerance.

Example in Python:
```python
def bisection(f, a, b, tol):
    if f(a) * f(b) > 0:
        print("f(a) and f(b) do not have opposite signs")
        return None
    
    c = a
    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0
        if abs(f(c)) < tol:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    return c

# Example function for even wave functions
def f_even(x):
    return (10 - x) ** 0.5 * tan((10 - x) ** 0.5) - (x ** 0.5)

bound_state_energy_even = bisection(f_even, 0, 10, 1e-6)
```
x??

---

#### Bisection Method Overview
Background context: The bisection method is a root-finding algorithm that repeatedly bisects an interval and then selects a subinterval in which a root must lie for further processing. It works by checking if the signs of function values at both ends of an interval are opposite, indicating a root within this interval.

:p What is the basic idea behind the bisection method?
??x
The basic idea is to repeatedly halve the interval where the root might exist based on the sign changes of the function. This ensures that if $f(a) \cdot f(b) < 0$, there must be at least one zero in the interval [a, b]. 
```python
def bisection(f, a, b, tol):
    plus = b
    minus = a
    x = (plus + minus) / 2
    
    while abs(f(x)) > tol:
        if f(plus) * f(x) > 0:
            plus = x
        else:
            minus = x
        
        x = (plus + minus) / 2

    return x
```
x??

---

#### Bisection Example Problem Setup
Background context: For the given function $f(E) = \sqrt{10 - E} \tan(\sqrt{10 - E}) - \sqrt{E}$, plotting or creating a table can help identify approximate values at which $ f(EB) = 0$. This step is crucial for determining initial intervals.

:p How should you approach identifying zeros of the function?
??x
First, plot or create a table of the function to observe where it crosses zero. For instance:
```python
import matplotlib.pyplot as plt
import numpy as np

def f(E):
    return np.sqrt(10 - E) * np.tan(np.sqrt(10 - E)) - np.sqrt(E)

E = np.linspace(0, 10, 400)
plt.plot(E, f(E))
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
```
x??

---

#### Bisection Algorithm Implementation
Background context: Implementing the bisection algorithm involves repeatedly bisecting intervals and checking for sign changes. The process continues until the function value is below a certain precision level or a maximum number of iterations are reached.

:p Write pseudocode to implement the bisection method.
??x
```pseudocode
function bisection(f, initialInterval, tolerance, maxIterations):
    (a, b) = initialInterval  # Interval [a, b]
    
    for i from 1 to maxIterations:
        c = (a + b) / 2  # Midpoint of the interval
        
        if f(c) == 0 or (b - a) < tolerance:
            return c  # Return the root
        
        elif f(a) * f(c) > 0:  # Sign does not change
            a = c
        else:  # Sign changes
            b = c
    
    return "Failed to converge"
```
x??

---

#### Handling Tan Function Singularities
Background context: The function $f(E)$ involves the tangent function, which has singularities. These can cause numerical issues and need careful handling.

:p How can you handle the singularities of the tan function?
??x
One way is to use an equivalent form of the equation that avoids these singularities. For example:
```python
def g(E):
    return np.sqrt(E) * np.cot(np.sqrt(10 - E)) - np.sqrt(10 - E)
```
Using $\tan(x) = 1 / \cot(x)$, we can rewrite the function to avoid singularities.
x??

---

#### Newton-Raphson Method Overview
Background context: The Newton-Raphson method is another root-finding algorithm that uses tangent lines to approximate the roots. It starts with an initial guess and iteratively improves this guess until a certain precision level is reached.

:p What is the basic idea behind the Newton-Raphson method?
??x
The basic idea is to use the slope of the function at a point to find a better approximation for the root. The method uses the tangent line at the current estimate to approximate where the function crosses the x-axis.
```python
def newton_raphson(f, df_dx, initial_guess, tol, max_iter):
    x = initial_guess
    
    for i in range(max_iter):
        fx = f(x)
        
        if abs(fx) < tol:
            return x
        
        dx = -fx / df_dx(x)
        x += dx
    
    return "Failed to converge"
```
x??

---

#### Newton-Raphson Method Implementation with Derivatives
Background context: The Newton-Raphson method requires the derivative of the function. This can be calculated analytically or approximated numerically.

:p How does the Newton-Raphson method update its guess?
??x
The method updates the guess by adding the negative ratio of the function value to the derivative at that point:
```python
def newton_raphson(f, df_dx, initial_guess, tol, max_iter):
    x = initial_guess
    
    for i in range(max_iter):
        fx = f(x)
        
        if abs(fx) < tol:
            return x
        
        dx = -fx / df_dx(x)
        x += dx
    
    return "Failed to converge"
```
Here, $\Delta x = -\frac{f(x)}{f'(x)}$.
x??

---

#### Central Difference Approximation of Derivative
Background context: For complex functions or when the derivative is not easily obtainable, a numerical approximation using central difference can be used.

:p How does the central difference method approximate the derivative?
??x
The central difference method approximates the derivative as follows:
```python
def df_dx(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```
This provides a more accurate estimate of the derivative than the forward difference.
x??

---

