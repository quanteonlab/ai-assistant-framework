# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 68)

**Starting Chapter:** 5.4.1 Mapping Gaussian Points

---

#### Simpson's Rule vs Trapezoid Rule

Background context: The text compares Simpson's rule and the trapezoidal rule for numerical integration. It highlights that Simpson's rule requires fewer points and has less error compared to the trapezoidal rule. Additionally, it mentions that with appropriate algorithms like Simpson's rule, one can achieve an error close to machine precision.

:p How does Simpson's rule compare to the trapezoid rule in terms of accuracy?
??x
Simpson's rule is more accurate than the trapezoidal rule because it approximates the function using a quadratic polynomial between each pair of points. This allows for better fitting, especially when the function has curvature. The error term for Simpson's rule is proportional to $h^4 $, while for the trapezoidal rule, it is proportional to $ h^2$.

For example, if we have two intervals, using Simpson's rule:
- Trapezoidal rule: $E_{trap} = -\frac{(b-a)^3}{12n^2}f''(\xi)$- Simpson's rule:$ E_{simp} = -\frac{(b-a)^5}{180n^4}f^{(4)}(\xi)$

Simpson's rule provides a smaller error for the same number of points.

```java
public class TrapezoidalAndSimpson {
    public static double trapezoidRule(double[] y, int N) {
        // Implementation of the trapezoidal rule.
        return (y[0] + 2 * sum(y, 1, N - 1) + y[N]) / 2.0;
    }

    public static double simpsonRule(double[] y, int N) {
        // Implementation of Simpson's rule.
        return (y[0] + 4 * sum(y, 1, N - 1, true) + y[N]) / 3.0;
    }

    private static double sum(double[] array, int start, boolean multiplyBy4) {
        double result = 0.0;
        for (int i = start; i < array.length; i += 2) {
            result += array[i];
        }
        if (multiplyBy4) result *= 4;
        return result;
    }
}
```
x??

---

#### Romberg's Extrapolation

Background context: The text explains how to use known functional dependence of the error on interval size $h$ to reduce integration error. For simple algorithms like trapezoid and Simpson’s rules, there are analytic estimates for the error terms.

:p How can you apply Romberg’s extrapolation to improve numerical integration accuracy?
??x
Romberg's extrapolation is a technique that uses known functional dependence of the error on interval size $h $ to reduce integration errors. Specifically, if the error has an expansion with a leading term proportional to$h^2$, we can use the values at different intervals and combine them in a way to eliminate this leading error term.

For example, if $A(h) \approx \int_a^b f(x) dx + \alpha h^2 + \beta h^4 + \ldots$:
- For interval size $h/2 $, we have $ A(h/2) \approx \int_a^b f(x) dx + \alpha (h/2)^2 + \beta (h/2)^4 + \ldots$.

To eliminate the leading error term, we can combine these values:
$$A(h) \approx 4A(h/2) - A(h).$$

This formula effectively cancels out the $h^2$ term.

```java
public class RombergsExtrapolation {
    public static double romberg(double[] y, int N) {
        // Initial estimate with trapezoidal rule.
        double R01 = (y[0] + 2 * sum(y, 1, N - 1) + y[N]) / 3.0;
        
        for (int i = 1; i < y.length; i++) {
            // Compute intermediate approximations.
            double t = 4 * R01 - y[i];
            
            // Update the Romberg table.
            if (i > 1) {
                y[i] = (y[i-1] + t) / 2.0;
            } else {
                y[i] = t;
            }
        }
        
        return y[y.length - 1]; // Return last entry in the table, which is the best approximation.
    }

    private static double sum(double[] array, int start) {
        double result = 0.0;
        for (int i = start; i < array.length; i += 2) {
            result += array[i];
        }
        return result;
    }
}
```
x??

---

#### Integration with Equal-Interval Rules

Background context: The text discusses various equal-interval rules such as trapezoid, Simpson’s, and three-eighths rules. These methods are used to approximate the integral over a given range by dividing it into smaller intervals.

:p How do you determine the number of elementary intervals in an integration problem using equal-interval rules?
??x
To determine the number of elementary intervals $N $ in an integration problem, you can sum up the weights for any rule. The sum should be equal to$h \times N = b - a $, where$ h $is the interval size and$ b-a$ is the total range.

For example:
- Trapezoidal Rule: Sum of weights is 1.
- Simpson’s Rule: Sum of weights is 2 (it uses two intervals internally).

```java
public class EqualIntervalRules {
    public static int determineN(double a, double b, double[] weights) {
        // Calculate the number of intervals N such that h * N = b - a.
        double h = (b - a);
        return sum(weights) == 1 ? (int) Math.round(h / a) : (int) Math.round((b - a) / (h * 2));
    }

    private static int sum(double[] array) {
        int result = 0;
        for (double weight : array) {
            result += weight;
        }
        return result;
    }
}
```
x??

---

#### Gaussian Quadrature

Background context: The text introduces the concept of Gaussian quadrature, which is a method to approximate an integral by choosing specific points and weights that make the integration exact if the function $g(x)$ were a polynomial of degree up to $2N-1$.

:p What is Gaussian quadrature, and why is it useful?
??x
Gaussian quadrature is a numerical integration technique where the number of points and their associated weights are chosen such that the method integrates exactly polynomials up to a certain degree. This means if $g(x)$ is a polynomial of degree less than or equal to $2N-1$, the integral will be computed exactly.

This approach often yields higher accuracy compared to simpler methods like trapezoidal and Simpson's rules for the same number of points, especially when dealing with smooth functions. The specific choice of points and weights depends on the weighting function used.

For example, in ordinary Gaussian (Gauss-Legendre) integration:
$$\int_{-1}^{1} f(x) dx = \sum_{i=0}^N w_i f(x_i),$$where $ x_i $ are the roots of the Legendre polynomial and $ w_i$ are the weights related to derivatives.

```java
public class GaussianQuadrature {
    public static double gaussLegendre(int N, Function<Double, Double> func) {
        // Gauss-Legendre integration with four-point rule as an example.
        double[] xi = {-0.906179845938664, -0.538469310105683,
                       0.538469310105683, 0.906179845938664};
        double[] wi = {0.236926885056189, 0.478628670499366,
                       0.478628670499366, 0.236926885056189};
        
        double result = 0;
        for (int i = 0; i < N; i++) {
            result += wi[i] * func.apply(xi[i]);
        }
        return result;
    }
}
```
x??

---

#### Gaussian Quadrature Mapping for [−1, 1] to [a, b]
Background context: When performing numerical integration on a general interval $[a, b]$, it is common to map the interval back to $[-1, 1]$ where standard Gaussian quadrature rules apply. The transformation involves scaling and shifting operations.

The mapping formula for uniform scaling from $[-1, 1]$ to $[a, b]$ is given by:
$$x_i = \frac{b+a}{2} + \frac{b-a}{2} y_i,$$where $ y_i $ are the Gaussian points in $[-1, 1]$, and the weights transform as:
$$w_i' = w'_i \cdot \frac{b-a}{2}.$$:p How is a general interval [a, b] transformed to [-1, 1] for Gaussian quadrature?
??x
The transformation involves scaling and shifting operations. Specifically:

- The midpoint of the interval $[a, b]$ is calculated as:
$$\text{midpoint} = \frac{b+a}{2}.$$- Each Gaussian point $ y_i $ in the range $[-1, 1]$ is mapped to a corresponding point $x_i$ in $[a, b]$ using the formula:
$$x_i = \text{midpoint} + \frac{b-a}{2} y_i.$$- The weights for the transformed points are adjusted by multiplying them with $\frac{b-a}{2}$:
  $$w_i' = w'_i \cdot \frac{b-a}{2}.$$

This transformation ensures that the Gaussian quadrature rules, which are valid on $[-1, 1]$, can be applied to any general interval $[a, b]$.
x??

---

#### Derivation of Gaussian Quadrature
Background context: The goal is to derive a numerical integration rule using Gaussian points and weights. We start by expressing the integral in terms of a polynomial divided by Legendre polynomials.

Given:
$$q(x) = \frac{f(x)}{P_N(x)},$$where $ P_N(x)$is the Legendre polynomial of order $ N$, we can decompose $ f(x)$:
$$f(x) = q(x) P_N(x) + r(x),$$with $ r(x)$being a polynomial of degree at most $ N-1$.

:p What is the key idea behind deriving Gaussian quadrature?
??x
The key idea involves using Legendre polynomials to decompose the function. By expressing $f(x)$ as:
$$q(x) = \frac{f(x)}{P_N(x)},$$we can separate it into two parts: a polynomial $ r(x)$ and the product of a polynomial with the Legendre polynomial.

The integral then becomes:
$$\int_{-1}^{1} f(x) dx = \int_{-1}^{1} q(x) P_N(x) dx + \int_{-1}^{1} r(x) dx.$$

Since $P_N(x)$ is orthogonal to any polynomial of degree less than or equal to $N$, the first term vanishes, and we are left with:
$$\int_{-1}^{1} f(x) dx = \int_{-1}^{1} r(x) dx.$$

This integral can be approximated using a standard quadrature rule because $r(x)$ is of degree at most $N-1$.

By choosing the Gaussian points to be the roots of $P_N(x)$, we ensure that:
$$q(x_i) P_N(x_i) = 0,$$for each root $ x_i $. This simplifies the integral, leading us to a quadrature rule with exactness for polynomials up to degree$2N-1$.
x??

---

#### Monte Carlo Integration Technique
Background context: Monte Carlo integration is a probabilistic method that uses random sampling to approximate integrals. It is particularly useful when dealing with high-dimensional integrals or complex functions.

:p What is the basic principle of Monte Carlo integration?
??x
The basic principle of Monte Carlo integration involves using random points to estimate the integral of a function over a given domain. The idea is based on the mean value theorem:
$$I = \int_a^b f(x) dx = (b-a) \langle f \rangle,$$where $\langle f \rangle $ is the average value of$f(x)$ over the interval $[a, b]$.

To approximate this integral using Monte Carlo:
1. Generate a sequence of random points $x_i $ uniformly distributed in$[a, b]$.
2. Evaluate $f(x_i)$ at these points.
3. The sample mean is computed as:
$$\langle f \rangle \approx \frac{1}{N} \sum_{i=1}^N f(x_i).$$4. Finally, the integral approximation is given by:
$$

I \approx (b-a) \cdot \left( \frac{1}{N} \sum_{i=1}^N f(x_i) \right).$$:p How does the Monte Carlo integration rule work for a general interval [a, b]?
??x
The Monte Carlo integration works by leveraging random sampling to approximate integrals. Given an interval $[a, b]$, we generate $ N$uniform random points $ x_i$. The function values at these points are averaged, and the result is scaled by the length of the interval:

1. Generate $N $ random points$x_i $ in the interval$[a, b]$.
2. Evaluate $f(x_i)$ for each point.
3. Compute the sample mean:
$$\langle f \rangle = \frac{1}{N} \sum_{i=1}^N f(x_i).$$4. The integral approximation is then given by:
$$

I \approx (b-a) \cdot \langle f \rangle.$$

This method relies on the law of large numbers, where increasing $N$ improves the accuracy of the estimate.
x??

---

#### Implementing Gaussian Quadrature in Python
Background context: The provided code example `vonNeuman.py` demonstrates how to implement Gaussian quadrature using Python. The function `gauss` generates points and weights for a specified number of nodes.

:p How does the `gauss` function generate Gaussian points and their corresponding weights?
??x
The `gauss` function in the `vonNeuman.py` code generates Gaussian points and their corresponding weights as follows:

1. **Generate Nodes**: It uses a predefined set of nodes (Gaussian points) for different numbers of nodes.
2. **Compute Weights**: The weights are calculated based on these nodes using known formulas.

Here is a simplified version of the `gauss` function:
```python
def gauss(n):
    # Predefined Gaussian nodes and weights for n = 1, 2, ..., 8
    nodes_and_weights = [
        ([-0.774596669241483, -0.000000000000000], [0.555555555555556, 0.555555555555556]),
        # More nodes and weights...
    ]
    
    if n > len(nodes_and_weights):
        raise ValueError("Too many nodes requested")
    
    return nodes_and_weights[n-1]
```

The function returns a tuple containing the nodes and their corresponding weights. For example, for $n=2$, it returns:
```python
([-0.774596669241483, -0.000000000000000], [0.555555555555556, 0.555555555555556])
```

This function can be used in other applications where Gaussian quadrature is needed.
x??

---

#### Error Analysis for Numerical Integration
Background context: To understand the accuracy of different numerical integration methods, we need to analyze their error behavior. The provided exercise involves comparing the errors of the trapezoidal rule, Simpson's rule, and Monte Carlo integration.

:p How do you compute the relative error for numerical integration?
??x
The relative error is computed by comparing the numerical result with the exact value:
$$\epsilon = \left| \frac{\text{numerical} - \text{exact}}{\text{exact}} \right|.$$

For example, if we are integrating $e^{-t}$ from 0 to 1 and the exact value is known to be:
$$N(1) = \int_0^1 e^{-t} dt = 1 - e^{-1},$$the relative error for a numerical method can be calculated as follows:

```python
def compute_relative_error(numerical, exact):
    return abs((numerical - exact) / exact)

# Example usage:
exact_value = 1 - math.exp(-1)
numerical_values = [trapezoidal_rule(0, 1, n), simpson_rule(0, 1, n), monte_carlo_integration(0, 1, n)]
relative_errors = [compute_relative_error(num_val, exact_value) for num_val in numerical_values]
```

:p How do you plot the relative error versus number of samples $N$ to analyze the power-law behavior?
??x
To plot the relative error $\epsilon $ versus the number of samples$N$, follow these steps:

1. **Compute Numerical Results**: For each value of $N$, compute the numerical integration result using different methods (trapezoidal rule, Simpson's rule, Monte Carlo).
2. **Calculate Relative Errors**: Compute the relative errors for each method.
3. **Plot the Data**: Use a log-log plot to visualize how $\epsilon $ scales with$N$.

Here is an example of Python code to perform these steps:

```python
import numpy as np
import matplotlib.pyplot as plt

def integrate_trapezoidal(a, b, N):
    # Implement trapezoidal rule here
    pass

def integrate_simpson(a, b, N):
    # Implement Simpson's rule here
    pass

def monte_carlo_integration(a, b, N):
    # Implement Monte Carlo integration here
    pass

# Exact value for e^{-t} from 0 to 1
exact_value = 1 - np.exp(-1)

N_values = [2, 10, 20, 40, 80, 160]
errors_trapezoidal = []
errors_simpson = []
errors_monte_carlo = []

for N in N_values:
    numerical_trapezoidal = integrate_trapezoidal(0, 1, N)
    errors_trapezoidal.append(compute_relative_error(numerical_trapezoidal, exact_value))
    
    numerical_simpson = integrate_simpson(0, 1, N)
    errors_simpson.append(compute_relative_error(numerical_simpson, exact_value))
    
    numerical_monte_carlo = monte_carlo_integration(0, 1, N)
    errors_monte_carlo.append(compute_relative_error(numerical_monte_carlo, exact_value))

# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(N_values, errors_trapezoidal, marker='o', label='Trapezoidal')
plt.loglog(N_values, errors_simpson, marker='s', label='Simpson')
plt.loglog(N_values, errors_monte_carlo, marker='^', label='Monte Carlo')

# Power-law fit
for name, errors in zip(['trapezoidal', 'simpson', 'monte_carlo'], [errors_trapezoidal, errors_simpson, errors_monte_carlo]):
    logN = np.log10(N_values)
    logE = np.log10(errors)
    slope, intercept = np.polyfit(logN, logE, 1)  # Linear fit
    print(f"Power-law for {name}: E ~ N^{slope}")

plt.xlabel('Number of Samples (log scale)')
plt.ylabel('Relative Error (log scale)')
plt.title('Error vs Number of Samples')
plt.legend()
plt.grid(True)
plt.show()
```

This code computes the relative errors and plots them on a log-log scale, allowing you to observe the power-law behavior.
x??

--- 

These flashcards cover key concepts in numerical integration techniques including Gaussian quadrature mapping, Monte Carlo integration, error analysis, and implementation details. Each card provides context and explanations along with relevant Python code examples where applicable. ---

#### Monte Carlo Integration Error Analysis
Background context explaining how the relative error in Monte Carlo integration decreases as $1/\sqrt{N}$, where $ N$ is the number of samples. This applies even when the points are distributed over multiple dimensions.

:p How does the error in Monte Carlo integration behave with increasing dimensions?
??x
The error in Monte Carlo integration decreases as $1/\sqrt{N}$ regardless of the number of dimensions, meaning that for large dimensions, it can become more accurate compared to traditional schemes. However, using Monte Carlo as a D-dimensional integral directly requires many samples per dimension, which increases with the dimensionality.

```java
// Pseudocode to illustrate the concept
public class ErrorAnalysis {
    public static void main(String[] args) {
        int N = 1000; // Number of samples
        double errorEstimate = 1.0 / Math.sqrt(N);
        System.out.println("Estimated error: " + errorEstimate);
    }
}
```
x??

---

#### High-Dimensional Integration Challenge
Background context explaining the difficulty in performing high-dimensional integrations, such as a 36-dimensional integral for small atoms like magnesium.

:p Why is it challenging to calculate properties of small atoms using Monte Carlo integration?
??x
Calculating properties of small atoms with many electrons requires integrating over many dimensions. For example, a 36-dimensional integral (12 electrons * 3 coordinates) would require evaluating the integrand at $64^{36} \approx 10^{65}$ points. This is computationally infeasible even for very fast computers due to the astronomical number of evaluations needed.

```java
// Pseudocode for high-dimensional integration challenge
public class HighDimIntegration {
    public static void main(String[] args) {
        int pointsPerDim = 64;
        int dims = 3 * 12; // For 12 electrons, each with 3 coordinates
        long evaluationsNeeded = (long) Math.pow(pointsPerDim, dims);
        System.out.println("Evaluations needed: " + evaluationsNeeded);
        double seconds = evaluationsNeeded / 1e6; // Assuming a million evaluations per second
        System.out.println("Time required: " + seconds + " seconds");
    }
}
```
x??

---

#### Monte Carlo vs. Traditional Integration Methods in High Dimensions
Background context explaining that while traditional methods (like Simpson’s rule) become less accurate with increasing dimensions, Monte Carlo integration can remain competitive or even more accurate for high-dimensional integrals.

:p How does the error of Monte Carlo integration compare to traditional integration methods as the number of dimensions increases?
??x
For high dimensions, traditional methods such as Simpson's rule become less effective because they require a large number of points per dimension, which increases with the dimensionality. In contrast, Monte Carlo integration maintains its $1/\sqrt{N}$ error rate regardless of the number of dimensions. Therefore, for dimensions greater than about 3-4, Monte Carlo can be more accurate.

```java
// Pseudocode to compare error rates
public class ErrorComparison {
    public static void main(String[] args) {
        int N = 1000; // Number of samples
        double monteCarloError = 1.0 / Math.sqrt(N);
        double simpsonsError = 1.0 / Math.pow(N, 4); // Simplified for demonstration

        System.out.println("Monte Carlo error: " + monteCarloError);
        System.out.println("Simpson's rule error (approx): " + simpsonsError);
    }
}
```
x??

---

#### Implementation of Monte Carlo Integration in High Dimensions
Background context explaining the implementation of 10-dimensional Monte Carlo integration and how to generalize mean value integration for multiple dimensions.

:p How would you implement a 10-dimensional Monte Carlo integral?
??x
To perform a 10-dimensional Monte Carlo integration, we need to randomly sample points within each dimension and approximate the integral by averaging the function values at these points. For example, to evaluate $\int_0^1 dx_1 \int_0^1 dx_2 ... \int_0^1 dx_{10} (x_1 + x_2 + ... + x_{10})^2 $, we would generate 10 random points in the range [0,1] for each dimension and compute the average of $ f(x_1, x_2, ..., x_{10}) = (x_1 + x_2 + ... + x_{10})^2$ at these points.

```java
// Pseudocode for 10D Monte Carlo integration
public class TenDimensionalIntegration {
    public static void main(String[] args) {
        int N = 1000; // Number of samples per dimension
        double sum = 0.0;
        
        for (int i = 0; i < N; i++) {
            double xiSum = 0.0;
            for (int j = 0; j < 10; j++) {
                double xj = Math.random(); // Random point in [0,1]
                xiSum += xj;
            }
            sum += Math.pow(xiSum, 2);
        }

        double result = sum / N;
        System.out.println("Monte Carlo integral result: " + result);
    }
}
```
x??

---

#### Generalization of Mean Value Integration
Background context explaining how mean value integration can be generalized to multiple dimensions by picking random points in a multidimensional space and using the average function values.

:p How does mean value integration generalize to many dimensions?
??x
Mean value integration can be extended to higher dimensions by sampling points uniformly from the hypercube defined by the integration limits. The integral is then approximated by averaging the function values at these points, scaled appropriately by the volume of the domain. For example, in 2D:$\int_a^b dx \int_c^d dy f(x,y) \approx (b-a)(d-c)\frac{1}{N} \sum_{i=1}^N f(x_i, y_i)$, where $ x_i$and $ y_i$ are random points in the intervals [a,b] and [c,d], respectively.

```java
// Pseudocode for 2D mean value integration
public class TwoDimensionalMeanValue {
    public static void main(String[] args) {
        int N = 1000; // Number of samples
        double a = 0.0, b = 1.0;
        double c = 0.0, d = 1.0;
        
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            double xi = Math.random() * (b - a) + a;
            double yi = Math.random() * (d - c) + c;
            sum += f(xi, yi); // Function to evaluate
        }
        
        double result = (b - a) * (d - c) * (sum / N);
        System.out.println("Mean value integration result: " + result);
    }

    public static double f(double x, double y) {
        return (x + y) * (x + y); // Example function
    }
}
```
x??

