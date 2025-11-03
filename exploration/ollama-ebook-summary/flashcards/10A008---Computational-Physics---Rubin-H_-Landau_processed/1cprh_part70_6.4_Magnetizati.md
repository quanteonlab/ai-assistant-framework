# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 70)

**Starting Chapter:** 6.4 Magnetization Search

---

#### Newton-Raphson Algorithm Basics
Background context explaining the Newton-Raphson algorithm. The algorithm is used to find roots of a function \( f(x) = 0 \). It requires an initial guess and iteratively improves it using the formula:
\[ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \]

If the derivative vanishes at the root, the algorithm fails as division by zero occurs. This can lead to issues like infinite loops or guessing out of bounds.

:p What is a common issue with the Newton-Raphson algorithm when the initial guess is not accurate?
??x
When the initial guess is inaccurate and the function has a local extremum (where the derivative vanishes), the algorithm might fail due to division by zero, leading to an infinite loop or incorrect guesses. This happens because \( x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \) cannot be computed if \( f'(x_n) = 0 \).
x??

---

#### Backtracking in Newton-Raphson
Background context on the failure modes of the Newton-Raphson algorithm. If a step leads to an increase in function magnitude, backtracking can help.

:p How does backtracking work to address issues with the Newton-Raphson method?
??x
Backtracking involves reducing the step size when a guess increases the function's value. For example:
- If \( x_0 + \Delta x \) increases \( f(x) \), try a smaller step like \( x_0 + \frac{\Delta x}{2} \).
- Continue halving until \( f(x) \) decreases, ensuring convergence.

Here is a pseudocode snippet to illustrate the backtracking logic:
```pseudocode
function backtrack(x0, dx):
    current_x = x0 + dx
    while f(current_x) > f(x0):  # Check if step increased function value
        dx /= 2  # Halve the step size
        current_x = x0 + dx
    return current_x
```
x??

---

#### Bisection vs. Newton-Raphson Comparison
Background context on comparing different root-finding methods, specifically bisection and Newton-Raphson.

:p How can you compare solutions from the Newton-Raphson algorithm with those from the bisection method?
??x
To compare, apply both methods to find roots of an equation like \( f(x) = 0 \). Use the bisection method first for its robustness, then switch to Newton-Raphson once close to the root. Compare results in terms of accuracy and computational efficiency.

Example code comparing two methods:
```python
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        x = x0 - f(x0) / df(x0)
        if abs(f(x)) < tol:
            return x
    return None

def bisection(f, a, b, tol=1e-6):
    while (b - a) > tol:
        c = (a + b) / 2.0
        if f(c) == 0 or (b - a) < tol:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return None

# Example usage
f = lambda x: x**2 - 4
df = lambda x: 2*x
x_newton = newton_raphson(f, df, 3)
x_bisect = bisection(f, 1, 3)

print("Newton-Raphson root:", x_newton)
print("Bisection root:", x_bisect)
```
x??

---

#### Magnetization Problem Setup
Background context on the magnetization problem. The system involves spins in an external magnetic field \( B \), with Boltzmann distribution determining spin states.

:p What is the formula for the number of particles in the lower and upper energy states?
??x
The formulas are:
- Lower state: \( N_L = \frac{N e^{\mu B / (k_B T)}}{e^{\mu B / (k_B T)} + e^{-\mu B / (k_B T)}} \)
- Upper state: \( N_U = \frac{N e^{-\mu B / (k_B T)}}{e^{\mu B / (k_B T)} + e^{-\mu B / (k_B T)}} \)

Where:
- \( N \) is the total number of particles,
- \( \mu \) is the magnetic moment per particle,
- \( k_B \) is Boltzmann's constant,
- \( T \) is temperature.

These formulas are derived from the Boltzmann distribution.
x??

---

#### Magnetization in Terms of Reduced Variables
Background context on simplifying the magnetization problem using reduced variables. Introduce reduced magnetization and temperature to simplify calculations.

:p How do you express the magnetization in terms of reduced variables?
??x
The magnetization \( M(T) \) is expressed as:
\[ m(t) = \tanh\left( \frac{m}{t} \right) \]
where:
- \( t = T / T_c \),
- \( T_c = N \mu^2 \lambda / k_B \).

The reduced magnetization \( m \) and temperature \( t \) help in numerical solutions.

To find the magnetization, use:
\[ M(T) = \frac{N \mu}{1 + e^{(\mu B - 2 J S)(T_c/T - 1)}} \]
where \( B \) is the external magnetic field.
x??

---

#### Implementing Backtracking in Newton-Raphson
Background context on enhancing the Newton-Raphson method with backtracking.

:p How does implementing backtracking improve the Newton-Raphson algorithm?
??x
Backtracking improves by halving steps when they increase function value, ensuring convergence. If \( f(x_0 + \Delta x) > f(x_0) \), reduce step size like \( x_0 + \frac{\Delta x}{2} \). Continue halving until a decrease is found.

Example pseudocode:
```pseudocode
function newton_raphson_backtrack(f, df, x0):
    dx = 1.0
    while True:
        x_next = x0 - f(x0) / df(x0)
        if f(x_next) < f(x0):  # Check for improvement
            break
        dx /= 2
        x0 += dx
    return x_next + dx
```
x??

---

#### Magnetization Problem: Finding Roots
Background context on solving the magnetization problem using root-finding methods.

:p How do you find the reduced magnetization \( m \) for a given temperature?
??x
Find the zero of \( f(m, t) = m - \tanh\left( \frac{m}{t} \right) \). Use numerical methods like bisection or Newton-Raphson.

Example code using the bisection method:
```python
def f(m, t):
    return m - math.tanh(m / t)

def bisect_root(f, a, b, tol=1e-6):
    while (b - a) > tol:
        c = (a + b) / 2.0
        if f(c) == 0 or (b - a) < tol:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return None

t_value = 0.5
root = bisect_root(f, 0, 2, tol=1e-6)
print("Reduced magnetization at t =", t_value, "is", root)
```
x??

---

#### Data Fitting Introduction
Background context: Data fitting is a crucial skill for scientists, as it involves finding the best fit of a theoretical function to experimental data. This process often uses least-squares methods to account for experimental noise and ensure meaningful results.

:p What is the main goal of data fitting?
??x
The main goal of data fitting is to determine the best-fit parameters of a theoretical function that describes experimental data, accounting for any noise present in the measurements.
x??

---

#### Interpolation between Data Points
Background context: Interpolation involves estimating values between known data points. A common method is polynomial interpolation using Lagrange's formula.

:p What does polynomial interpolation aim to achieve?
??x
Polynomial interpolation aims to estimate the value of a function at intermediate points between known data values by fitting a polynomial that passes through these points.
x??

---

#### Lagrange Interpolation Formula
Background context: The Lagrange interpolation formula provides a way to fit an \( (n-1) \)-degree polynomial through \( n \) given points. It is defined as:

\[ g(x) ≃ g_1\lambda_1(x) + g_2\lambda_2(x) + ... + g_n\lambda_n(x) \]
where
\[ \lambda_i(x) = \frac{(x - x_1)(x - x_2)...(x - x_{i-1})(x - x_{i+1})...(x - x_n)}{x_i - x_1}(x - x_1)x_i - x_2 ... (x - x_n)(x_i - x_n) \]

:p What is the Lagrange interpolation formula used for?
??x
The Lagrange interpolation formula is used to fit a polynomial of degree \( n-1 \) that passes through \( n \) given data points, allowing us to estimate values between these points.
x??

---

#### Example of Lagrange Interpolation
Background context: An example illustrates how the Lagrange formula can be applied to find a polynomial that fits multiple data points.

:p How is the polynomial determined using Lagrange interpolation?
??x
Using Lagrange interpolation, we determine the coefficients of the polynomial by evaluating it at specific points. For instance, given four points \( (0, -12), (1, -12), (2, -24), (4, -60) \), a third-degree polynomial is found that fits these values.

```java
// Example code to calculate the Lagrange polynomial for 4 points
public class LagrangeInterpolation {
    public static double lagrangePolynomial(double x, double[] xValues, double[] yValues) {
        int n = xValues.length;
        double result = 0.0;

        for (int i = 0; i < n; i++) {
            double term = yValues[i];
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    term *= (x - xValues[j]) / (xValues[i] - xValues[j]);
                }
            }
            result += term;
        }
        return result;
    }

    public static void main(String[] args) {
        double[] xValues = {0, 1, 2, 4};
        double[] yValues = {-12, -12, -24, -60};
        System.out.println(lagrangePolynomial(3.5, xValues, yValues)); // Example evaluation
    }
}
```
x??

---

#### Least-Squares Fitting
Background context: Least-squares fitting is a statistical method used to find the best-fit parameters of a theoretical function that minimizes the sum of the squares of the residuals (differences between observed and predicted values).

:p What does least-squares fitting aim to achieve?
??x
Least-squares fitting aims to find the parameters of a theoretical function that provide the closest fit to experimental data by minimizing the sum of the squares of the differences between observed and predicted values.
x??

---

#### Application of Least-Squares Fitting in Neutron Scattering Data
Background context: In neutron scattering, cross sections are measured at discrete energy points. A least-squares fitting approach can be used to find the best-fit parameters for a theoretical function describing these data.

:p How is the least-squares method applied to fit a theoretical model?
??x
The least-squares method is applied by adjusting the parameters of a theoretical function (e.g., \( f(E) = fr (E - Er)^2 + \Gamma^2 / 4 \)) to minimize the sum of the squares of the differences between observed and predicted cross sections. This provides the best-fit values for unknown parameters like \( fr, Er, \) and \( \Gamma \).
x??

---

#### Interpolation vs. Least-Squares Fitting
Background context: While interpolation directly fits a function through data points, least-squares fitting finds the best fit in a statistical sense, potentially not passing through all data points.

:p What is the main difference between interpolation and least-squares fitting?
??x
The main difference between interpolation and least-squares fitting lies in their objectives. Interpolation aims to find a function that passes through every data point, while least-squares fitting seeks to minimize the overall error by finding a best-fit curve that may not pass through all points.
x??

---

#### Global vs. Local Fits
Background context: In global fits, a single function is used for the entire dataset, whereas in local fits, different functions are used over smaller intervals.

:p What distinguishes a global fit from a local fit?
??x
A global fit uses a single function to represent the entire dataset, while a local fit uses multiple functions (polynomials) over smaller intervals. Global interpolation may show non-physical behavior between data points if the assumed function is incorrect.
x??

---

#### Lagrange Interpolation Overview
Lagrange interpolation is a method used to construct a polynomial that passes through all given data points. The general form of the Lagrange polynomial for \(n\) points \((x_i, y_i)\) is:

\[
P(x) = \sum_{i=0}^{n-1} L_i(x) y_i
\]

where

\[
L_i(x) = \prod_{\substack{0 \leq j \leq n-1 \\ j \neq i}} \frac{x - x_j}{x_i - x_j}
\]

This polynomial is guaranteed to pass through each of the \(n\) given points. For example, in a 9-point Lagrange interpolation (8th degree polynomial), it would fit all nine data points.

:p What is Lagrange interpolation used for?
??x
Lagrange interpolation is used to construct a polynomial that passes exactly through all given data points. It's particularly useful when you need an exact fit and the order of the polynomial can be determined by the number of data points.
x??

---

#### Eight-Point Polynomial Fitting Using Lagrange Interpolation
The problem at hand involves fitting an 8th-degree polynomial to nine noisy experimental neutron scattering data points using Lagrange interpolation. This means \(n = 9\) and you need to create a 9-point Lagrange interpolation.

:p How do you fit the entire experimental spectrum with one polynomial using Lagrange interpolation?
??x
To fit the entire experimental spectrum with an 8th-degree polynomial using Lagrange interpolation, you would use all nine data points. The general form of the Lagrange polynomial is:

\[
P(x) = \sum_{i=0}^{8} L_i(x) y_i
\]

where

\[
L_i(x) = \prod_{\substack{0 \leq j \leq 8 \\ j \neq i}} \frac{x - x_j}{x_i - x_j}
\]

Here, \(y_i\) are the experimental data values at points \(x_i\). For each new point \(x\) where you want to interpolate, you would compute:

```java
public class LagrangeInterpolation {
    public double lagrangePolynomial(double[] xi, double[] yi, double x) {
        int n = xi.length;
        double result = 0.0;
        
        for (int i = 0; i < n; i++) {
            double li = 1.0;
            
            // Compute L_i(x)
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    li *= (x - xi[j]) / (xi[i] - xi[j]);
                }
            }
            
            result += yi[i] * li;
        }
        
        return result;
    }
}
```

This code calculates the Lagrange polynomial value at point \(x\).

x??

---

#### Resonance Energy and Full Width at Half Maximum
After fitting the data with an 8th-degree polynomial, you can analyze it to find resonance energy \(E_r\) (the peak position) and full width at half maximum \(\Gamma\). These values are crucial for understanding the behavior of the resonant cross section.

:p How do you determine the resonance energy and full width at half maximum using the fitted polynomial?
??x
To determine the resonance energy \(E_r\) and full width at half maximum \(\Gamma\):

1. **Resonance Energy (\(E_r\))**: Find the peak position of the polynomial, which is where the derivative changes from positive to negative.
2. **Full Width at Half Maximum (FWHM, \(\Gamma\))**: Calculate the \(x\) values where the polynomial value equals half its maximum.

Here's a simplified approach:

```java
public class CrossSectionAnalysis {
    public double findResonanceEnergy(double[] xi, double[] yi) {
        // Fit the data with an 8th-degree polynomial and find the peak position.
        double max = Double.MIN_VALUE;
        int peakIndex = -1;
        
        for (int i = 0; i < yi.length; i++) {
            if (yi[i] > max) {
                max = yi[i];
                peakIndex = i;
            }
        }
        
        return xi[peakIndex]; // Approximate resonance energy
    }

    public double findGamma(double[] xi, double[] yi) {
        double max = Double.MIN_VALUE;
        int peakIndex = -1;
        
        for (int i = 0; i < yi.length; i++) {
            if (yi[i] > max) {
                max = yi[i];
                peakIndex = i;
            }
        }

        double halfMax = max / 2.0;
        
        // Find the x-values where the polynomial value equals half its maximum.
        List<Double> gammaXValues = new ArrayList<>();
        for (int i = 0; i < xi.length; i++) {
            if (yi[i] > halfMax) {
                gammaXValues.add(xi[i]);
            }
        }

        // Calculate FWHM
        return Math.abs(gammaXValues.get(1) - gammaXValues.get(0));
    }
}
```

x??

---

#### Three-Point Lagrange Interpolation for Resonant Cross Section
For a more realistic use, three-point Lagrange interpolation can be used to fit the cross-sectional data in 5-MeV steps. This avoids the wild oscillations that high-degree polynomials might introduce.

:p How do you perform three-point Lagrange interpolation for each interval?
??x
To perform three-point Lagrange interpolation for each interval of 5 MeV:

1. **Divide the spectrum into intervals**: Each interval will span 5 MeV.
2. **Calculate the polynomial**: For each \(x\) value in an interval, use the three points around it to calculate a 2nd-degree polynomial.

Example code:

```java
public class ThreePointLagrange {
    public double threePointLagrange(double x, double[] xi, double[] yi) {
        int n = xi.length;
        
        if (n != 3) {
            throw new IllegalArgumentException("Exactly 3 points required for 2nd-degree polynomial.");
        }
        
        // Lagrange basis polynomials
        double L0 = ((x - xi[1]) * (x - xi[2])) / ((xi[0] - xi[1]) * (xi[0] - xi[2]));
        double L1 = ((x - xi[0]) * (x - xi[2])) / ((xi[1] - xi[0]) * (xi[1] - xi[2]));
        double L2 = ((x - xi[0]) * (x - xi[1])) / ((xi[2] - xi[0]) * (xi[2] - xi[1]));
        
        return yi[0] * L0 + yi[1] * L1 + yi[2] * L2;
    }
}
```

This code calculates the value of a 2nd-degree Lagrange polynomial at \(x\).

x??

---

#### Extrapolation Using Lagrange Interpolation
Extrapolating with high-order polynomials can lead to serious systematic errors, as they rely heavily on the assumed function form. However, it's an interesting exercise.

:p How do you extrapolate using the programs written for Lagrange interpolation?
??x
To extrapolate using Lagrange interpolation, follow these steps:

1. **Fit the data with a high-degree polynomial**: Use all available points to fit an \(n\)-degree polynomial.
2. **Predict values outside the range**: Use the fitted polynomial to predict values at \(x\) outside the given range.

Example code for extrapolation:

```java
public class Extrapolation {
    public double extrapolate(double x, double[] xi, double[] yi) {
        int n = xi.length;
        
        // Create a Lagrange polynomial object and use it to fit the data.
        LagrangeInterpolation interpolator = new LagrangeInterpolation();
        double interpolatedValue = interpolator.lagrangePolynomial(xi, yi, x);
        
        return interpolatedValue; // Extrapolated value
    }
}
```

This code extrapolates values using a previously fitted polynomial.

x??

---

#### Cubic Spline Interpolation Overview
Cubic spline interpolation is a method that fits piecewise cubic polynomials between data points while ensuring the first and second derivatives are continuous. This results in smooth curves, making it ideal for real-world applications where noise is present.

:p What is cubic spline interpolation used for?
??x
Cubic spline interpolation is used to fit data with smooth, piecewise cubic polynomials that ensure continuity of the function, its first derivative, and its second derivative (spline). It provides a more stable and visually pleasing fit compared to high-degree polynomial fitting.
x??

---

#### Cubic Spline Fitting Implementation
In cubic spline interpolation, each interval \([x_i, x_{i+1}]\) is fitted with a 3rd-degree polynomial that ensures continuity of the function and its first and second derivatives at the knots.

:p How do you implement cubic spline fitting?
??x
To implement cubic spline fitting:

1. **Set up the system of equations**: Ensure \(n-1\) intervals, resulting in \((n-1)\) third-degree polynomials.
2. **Solve for coefficients**: Use linear algebra to solve for the coefficients that ensure continuity.

Example code:

```java
public class CubicSpline {
    public void fitCubicSplines(double[] xi, double[] yi) {
        int n = xi.length;
        
        // Initialize arrays
        double[][] A = new double[n][n];
        double[] B = new double[n-1];
        double[] C = new double[n-1];
        double[] D = new double[n-1];
        
        // Set up the system of equations
        for (int i = 0; i < n-1; i++) {
            A[i][i] = 2 * (xi[i+1] - xi[i]);
            if (i > 0) {
                A[i][i-1] = (xi[i+1] - xi[i]) / 3;
            }
            if (i < n-2) {
                A[i][i+1] = (xi[i+2] - xi[i+1]) / 3;
            }
            
            B[i] = yi[i];
            C[i] = (yi[i+1] - yi[i]) / (xi[i+1] - xi[i]);
        }
        
        // Solve the system of equations
        double[] Z = solveLinearSystem(A, B);
        
        for (int i = 0; i < n-2; i++) {
            D[i] = C[i+1] - C[i];
            D[i] /= (xi[i+2] - xi[i]);
        }
    }

    // Helper method to solve linear system
    private double[] solveLinearSystem(double[][] A, double[] b) {
        int n = b.length;
        
        for (int k = 0; k < n-1; k++) {
            double temp = A[k+1][k] / A[k][k];
            A[k+1][k] = 0.0;
            A[k+1][k+1] -= temp * A[k][k+1];
            b[k+1] -= temp * b[k];
        }
        
        double[] x = new double[n];
        for (int k = n-1; k >= 0; k--) {
            x[k] = (b[k] - A[k][k+1] * x[k+1]) / A[k][k];
        }
        
        return x;
    }
}
```

This code sets up and solves the system of equations for cubic spline fitting.

x??

