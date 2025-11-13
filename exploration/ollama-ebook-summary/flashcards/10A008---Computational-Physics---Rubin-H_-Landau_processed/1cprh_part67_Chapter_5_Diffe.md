# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 67)

**Starting Chapter:** Chapter 5 Differentiation and Integration. 5.1 Differentiation Algorithms. 5.2.1 Second Derivatives

---

#### Forward Difference Algorithm
Background context explaining the forward difference algorithm. The forward difference algorithm is used to approximate the derivative of a function using its values at neighboring points. It uses the Taylor series expansion to derive the formula for numerical differentiation.

The formula derived from the Taylor series expansion is:
$$y(t+h) = y(t) + h \frac{dy}{dt} + \frac{h^2}{2!} \frac{d^2y}{dt^2} + \cdots$$

From this, we can approximate the derivative as follows:
$$\frac{dy(t)}{dt} \approx \frac{y(t+h) - y(t)}{h}$$

This approximation is known as the forward difference algorithm. The error in this approximation comes from the higher-order terms in the Taylor series.

:p What does the forward difference algorithm approximate, and what formula does it use?
??x
The forward difference algorithm approximates the derivative of a function at a point by using the values of the function at that point and a nearby point. The formula used is:
$$\frac{dy(t)}{dt} \approx \frac{y(t+h) - y(t)}{h}$$where $ h$ is a small step size.

This approximation can be written in pseudocode as follows:

```python
def forward_difference_derivative(y, t, h):
    return (y(t + h) - y(t)) / h
```

x??

---

#### Central Difference Algorithm
Background context explaining the central difference algorithm. The central difference algorithm is an improved method for numerical differentiation compared to the forward difference. It uses both a point and its symmetric neighbor to estimate the derivative, which reduces the error.

The formula derived from the Taylor series expansion is:
$$y(t + h/2) = y(t) + \frac{h}{2} \frac{dy}{dt} + \frac{h^2}{8} \frac{d^2y}{dt^2} + \cdots$$
$$y(t - h/2) = y(t) - \frac{h}{2} \frac{dy}{dt} + \frac{h^2}{8} \frac{d^2y}{dt^2} - \cdots$$

By subtracting these two expansions, the error terms containing even powers of $h$ cancel out:
$$y(t + h/2) - y(t - h/2) = h \frac{dy}{dt} + \frac{h^3}{24} \frac{d^3y}{dt^3} + \cdots$$

Therefore, the central difference approximation for the derivative is:
$$\frac{dy(t)}{dt} \approx \frac{y(t + h/2) - y(t - h/2)}{h}$$

The error in this method is $O(h^2)$, making it more accurate than the forward difference.

:p What does the central difference algorithm approximate, and what formula does it use?
??x
The central difference algorithm approximates the derivative of a function by using both points symmetrically around the point of interest. The formula used is:
$$\frac{dy(t)}{dt} \approx \frac{y(t + h/2) - y(t - h/2)}{h}$$

This method is more accurate than the forward difference because it reduces the error terms containing even powers of $h$.

Here’s a pseudocode implementation:

```python
def central_difference_derivative(y, t, h):
    return (y(t + h / 2) - y(t - h / 2)) / h
```

x??

---

#### Extrapolated Difference Algorithm
Background context explaining the extrapolated difference algorithm. This is an advanced method for numerical differentiation that combines multiple approximations to reduce error even further.

The central-difference approximation using a half-step back and forward:
$$\frac{dy(t)}{dt} \approx \frac{y(t + h/4) - y(t - h/4)}{h / 2} = 2 \cdot \frac{y(t + h/4) - y(t - h/4)}{h}$$

The central-difference approximation using a quarter-step:
$$\frac{dy(t, h / 2)}{dt} \approx \frac{y(t + h/8) - y(t - h/8)}{h / 2} = 2 \cdot \frac{y(t + h/8) - y(t - h/8)}{h}$$

The extended difference algorithm combines these two approximations to eliminate quadratic and linear terms:
$$\frac{dy(t)}{dt} \approx \frac{4 \cdot D_{cd}(y, t, h / 2) - D_{cd}(y, t, h)}{3}$$where $ D_{cd}$ represents the central-difference algorithm.

The error in this method is further reduced to higher-order terms. If $h = 0.4 $ and the fifth derivative of$y$ is approximately 1, there will be only one non-zero term left after combining both approximations.

:p What does the extrapolated difference algorithm do, and what formula does it use?
??x
The extrapolated difference algorithm uses a combination of central-difference approximations with different step sizes to reduce error further. Specifically, it combines the central-difference approximation using half-steps and quarter-steps:

1. The first approximation:
$$\frac{dy(t)}{dt} \approx 2 \cdot \frac{y(t + h/4) - y(t - h/4)}{h}$$2. The second approximation:
$$\frac{dy(t, h / 2)}{dt} \approx 2 \cdot \frac{y(t + h/8) - y(t - h/8)}{h}$$

The combined formula for the derivative is:
$$\frac{dy(t)}{dt} \approx \frac{4 \cdot D_{cd}(y, t, h / 2) - D_{cd}(y, t, h)}{3}$$where $ D_{cd}$ represents the central-difference algorithm.

This method reduces the error to higher-order terms. For example, if $h = 0.4 $ and the fifth derivative of$y$ is approximately 1, there will be only one non-zero term left after combining both approximations.

Here’s a pseudocode implementation:

```python
def extrapolated_difference_derivative(y, t, h):
    # Central-difference approximation using half-step (h/2)
    D_cd_half = (y(t + h / 4) - y(t - h / 4)) / (h / 2)
    
    # Central-difference approximation using quarter-step (h/8)
    D_cd_quarter = (y(t + h / 8) - y(t - h / 8)) / (h / 2)
    
    return (4 * D_cd_half - D_cd_quarter) / 3
```

x??

---

#### Central Difference Method for Second Derivatives
Background context: The central difference method is used to approximate second derivatives, which are crucial for determining force from position measurements as described by Newton's second law. The formula given is:
$$\frac{d^2y(t)}{dt^2} \bigg|_{cd} \approx \frac{y(t+h/2) - y(t-h/2)}{h}$$

This method reduces subtractive cancellation compared to the direct application of first differences. However, it may still introduce rounding errors when $h$ is very small.
:p How does the central difference method for second derivatives approximate the acceleration?
??x
The central difference method approximates the second derivative by moving forward and backward from the point of interest by half the step size $h$. This approach helps reduce subtractive cancellation:
$$\frac{d^2y(t)}{dt^2} \bigg|_{cd} \approx \frac{y(t+h/2) - y(t-h/2)}{h}$$:p What is the formula for calculating the second derivative using central differences?
??x
The formula for calculating the second derivative using central differences is:
$$\frac{d^2y(t)}{dt^2} \bigg|_{cd} \approx \frac{y(t+h/2) - y(t-h/2)}{h}$$

This method provides a balance between numerical accuracy and computational complexity.
x??

---
#### Extrapolated Central Difference Method
Background context: The extrapolated central difference method aims to improve the accuracy of second derivative approximation by combining values at different step sizes. This reduces subtractive cancellation but requires additional computations.
:p What is the formula for the extrapolated central difference method?
??x
The extrapolated central difference method uses a combination of first differences:
$$\frac{d^2y(t)}{dt^2} \bigg|_{cd} \approx \frac{8(y(t+h/4) - y(t-h/4)) - (y(t+h/2) - y(t-h/2))}{3h}$$

This formula combines values at $h/4, h/2,$ and $h$ to reduce subtractive cancellation.
:p What is the formula for the extrapolated central difference method?
??x
The formula for the extrapolated central difference method is:
$$\frac{d^2y(t)}{dt^2} \bigg|_{cd} \approx \frac{8(y(t+h/4) - y(t-h/4)) - (y(t+h/2) - y(t-h/2))}{3h}$$

This method aims to reduce subtractive cancellation by using multiple step sizes.
x??

---
#### Forward Difference Method for Second Derivatives
Background context: The forward difference method is another approach to approximating second derivatives. However, it tends to be less accurate due to the larger error term compared to central differences. The formula given is:
$$\frac{d^2y(t)}{dt^2} \bigg|_{fd} \approx \frac{y(t+h) - 2y(t) + y(t-h)}{h^2}$$

This method uses forward and backward differences, but it can introduce more rounding errors for small $h$.
:p What is the formula for calculating the second derivative using the forward difference method?
??x
The formula for calculating the second derivative using the forward difference method is:
$$\frac{d^2y(t)}{dt^2} \bigg|_{fd} \approx \frac{y(t+h) - 2y(t) + y(t-h)}{h^2}$$

This method can be less accurate than central differences due to the larger error term.
:p What is the formula for the forward difference method?
??x
The formula for the forward difference method is:
$$\frac{d^2y(t)}{dt^2} \bigg|_{fd} \approx \frac{y(t+h) - 2y(t) + y(t-h)}{h^2}$$

This method introduces a larger error term and can be less accurate for small step sizes.
x??

---
#### Evaluating Numerical Differentiation Error
Background context: The approximation errors in numerical differentiation decrease with smaller step sizes $h$, but rounding errors increase. The optimal step size balances these two errors, leading to the best approximation when:
$$\epsilon_{ro} \approx \epsilon_{app}$$

The forward and central difference methods have different error behaviors, with central differences generally providing better accuracy.
:p How do you determine the optimal step size for numerical differentiation?
??x
To determine the optimal step size $h $ for numerical differentiation, we balance the application error$\epsilon_{app}$ and rounding error $\epsilon_{ro}$:
$$\epsilon_{ro} \approx \frac{\epsilon_m}{h}, \quad \epsilon_{app} = y''h^2/2$$

Setting these equal gives:
$$h^3 = 24\epsilon_m / |y'''|$$

For double precision,$\epsilon_m \approx 10^{-15}$:
$$h = (24 \cdot 10^{-15})^{1/3}/|y'''| \approx 3 \times 10^{-5} \text{ for } y = \cos(t)$$

This step size minimizes the total error.
:p What is the optimal step size for numerical differentiation?
??x
The optimal step size $h$ for numerical differentiation, given by:
$$h = (24 \cdot 10^{-15})^{1/3}/|y'''| \approx 3 \times 10^{-5}$$

This value balances the application error and rounding errors.
x??

---
#### Implementing Numerical Differentiation in Code
Background context: The code examples provided demonstrate how to implement forward, central, and extrapolated difference methods for numerical differentiation. These methods are crucial for accurate force calculations from position data.
:p Write a program to calculate the second derivative of $\cos(t)$ using the central-difference algorithms (5.15) and (5.16).
??x
```c++
#include <iostream>
#include <cmath>

double y(double t) {
    return cos(t);
}

double fd_derivative(double t, double h) {
    return (y(t + h) - 2 * y(t) + y(t - h)) / (h * h);
}

double cd_derivative(double t, double h) {
    return (y(t + h / 2) - y(t - h / 2)) / h;
}

int main() {
    double t = 0.1; // Example time point
    double h = M_PI / 10; // Initial step size
    
    std::cout << "h: " << h << std::endl;
    
    while (true) {
        double fd_val = fd_derivative(t, h);
        double cd_val = cd_derivative(t, h);
        
        std::cout << "fd derivative: " << fd_val << ", error: " << fabs(fd_val + sin(t)) << std::endl; // Relative error
        std::cout << "cd derivative: " << cd_val << ", error: " << fabs(cd_val + sin(t)) << std::endl; // Relative error
        
        if (h < 1e-8) break;
        
        h /= 2; // Reduce step size
    }
    
    return 0;
}
```
This code calculates the second derivative of $\cos(t)$ using forward and central difference methods, printing out the derivative values and their relative errors as the step size decreases.
x??

---
#### Assessment for Numerical Differentiation
Background context: The assessment requires implementing various numerical differentiation techniques to approximate derivatives of functions. This involves reducing the step size until the error equals machine precision $\epsilon_m$.
:p Implement forward-, central-, and extrapolated-difference algorithms to differentiate the function $\cos(t)$ at specific points.
??x
```c++
#include <iostream>
#include <cmath>

double y(double t) {
    return cos(t);
}

// Forward difference derivative
double fd_derivative(double t, double h) {
    return (y(t + h) - y(t)) / h;
}

// Central difference derivative
double cd_derivative(double t, double h) {
    return (y(t + h / 2) - y(t - h / 2)) / h;
}

// Extrapolated central difference derivative
double ec_derivative(double t, double h) {
    return (8 * (y(t + h / 4) - y(t - h / 4)) - (y(t + h / 2) - y(t - h / 2))) / (3 * h);
}

int main() {
    std::cout << "t: 0.1" << std::endl;
    for (double t = 0.1; t <= 100; t *= 10) {
        double h = M_PI / 10; // Initial step size
        
        while (true) {
            double fd_val = fd_derivative(t, h);
            double cd_val = cd_derivative(t, h);
            double ec_val = ec_derivative(t, h);
            
            std::cout << "t: " << t << ", h: " << h << std::endl;
            std::cout << "fd derivative: " << fd_val << ", error: " << fabs(fd_val + sin(t)) << std::endl; // Relative error
            std::cout << "cd derivative: " << cd_val << ", error: " << fabs(cd_val + sin(t)) << std::endl; // Relative error
            std::cout << "ec derivative: " << ec_val << ", error: " << fabs(ec_val + sin(t)) << std::endl; // Relative error
            
            if (h < 1e-8) break;
            
            h /= 2; // Reduce step size
        }
    }
    
    return 0;
}
```
This code calculates the derivatives of $\cos(t)$ at points $t = 0.1, 1.0, 100$ using forward, central, and extrapolated difference methods, printing out the derivative values and their relative errors as the step size decreases.
x??

---

#### Trapezoid Rule Overview
Background context explaining the trapezoid rule. The trapezoid rule uses evenly spaced points to approximate an integral by breaking it into trapezoids, each with a height equal to the average of its endpoints and a width $h$.

The formula for the trapezoid rule is:
$$\int_a^b f(x) \, dx \approx \frac{h}{2} \left( f_1 + 2\sum_{i=2}^{N-1} f_i + f_N \right),$$where $ h = \frac{b - a}{N-1}$.

:p What is the trapezoid rule and how does it approximate an integral?
??x
The trapezoid rule approximates the area under a curve by dividing it into trapezoids. Each trapezoid's height is the average of its endpoints, and the width is $h = \frac{b - a}{N-1}$.

```java
public class TrapezoidalIntegration {
    public static double integrate(double a, double b, int N, Function<Double, Double> f) {
        double h = (b - a) / (N - 1);
        double sum = 0.5 * (f.apply(a) + f.apply(b));
        
        for (int i = 2; i < N; i++) {
            sum += f.apply(a + (i - 1) * h);
        }
        
        return sum * h;
    }
}
```
x??

---
#### Simpson’s Rule Overview
Background context explaining Simpson's rule. Simpson's rule uses parabolic arcs to approximate the integral within each interval, providing a more accurate approximation compared to the trapezoid rule.

The formula for Simpson's rule is:
$$\int_{a}^{b} f(x) \, dx \approx \frac{h}{3} \left( f_1 + 4\sum_{i=2,4,\ldots,N-1} f_i + 2\sum_{i=3,5,\ldots,N-2} f_i + f_N \right),$$where $ h = \frac{b - a}{N}$and $ N$ must be even.

:p What is Simpson's rule and how does it approximate an integral?
??x
Simpson's rule approximates the area under a curve by fitting parabolic arcs between points. Each interval contributes to the sum with weights that depend on whether the point is at the endpoints or midpoints of intervals, providing higher accuracy than the trapezoid rule.

```java
public class SimpsonsIntegration {
    public static double integrate(double a, double b, int N, Function<Double, Double> f) {
        if (N % 2 != 0) {
            throw new IllegalArgumentException("Number of points must be even for Simpson's rule.");
        }
        
        double h = (b - a) / N;
        double sum = f.apply(a) + f.apply(b);
        
        int k = 1;
        while (k < N) {
            sum += (2 * (k % 4 == 0 ? 2 : 4)) * f.apply(a + k * h);
            k++;
        }
        
        return sum * h / 3.0;
    }
}
```
x??

---
#### Simple Integration Error Estimates
Background context explaining the error estimates for equal-spacing rules like trapezoid and Simpson's rule, which involve Taylor series expansions around the midpoint of the integration interval.

For the trapezoid and Simpson’s rules:
$$\epsilon_t = O\left( \frac{(b-a)^3}{N^2} f'' \right),$$
$$\epsilon_s = O\left( \frac{(b-a)^5}{N^4} f^{(4)} \right).$$

The relative error $\epsilon$ is given by:
$$\epsilon_t, \epsilon_s = \frac{\epsilon_t, \epsilon_s}{f}.$$:p What are the error estimates for trapezoid and Simpson’s rules?
??x
For the trapezoid rule, the approximation error is proportional to $\left( \frac{(b-a)^3}{N^2} f'' \right)$, while for Simpson's rule, it is proportional to $\left( \frac{(b-a)^5}{N^4} f^{(4)} \right)$.

These estimates show that increasing the complexity of the integration method (from trapezoid to Simpson’s) reduces the error with a higher inverse power of N but also introduces higher derivative terms.

```java
public class ErrorEstimate {
    public static double estimateError(double b, double a, int N, Function<Double, Double> fPrimePrime, Function<Double, Double> f4) {
        return (Math.pow((b - a), 3) / Math.pow(N, 2)) * fPrimePrime.apply(0.5 * (a + b));
    }
    
    public static double simpsonErrorEstimate(double b, double a, int N, Function<Double, Double> f4) {
        return (Math.pow((b - a), 5) / Math.pow(N, 4)) * f4.apply(0.5 * (a + b));
    }
}
```
x??

---
#### Round-Off Error in Integration
Background context explaining the model for round-off error in integration assuming that after N steps, the relative round-off error is random and of the form $\epsilon_{\text{ro}} \approx \sqrt{N} \epsilon_m $, where $\epsilon_m$ is the machine precision.

To find the optimal number of points $N$ to minimize total error (approximation + round-off):
$$\epsilon_{\text{tot}} = \epsilon_{\text{ro}} + \epsilon_{\text{app}},$$where we approximate that the two errors are equal:
$$\sqrt{N} \epsilon_m \approx f'' \frac{(b-a)^3}{N^2},$$for trapezoid rule, and$$\sqrt{N} \epsilon_m \approx f^{(4)} \frac{(b-a)^5}{N^4},$$for Simpson's rule.

:p What is the model for round-off error in integration?
??x
The model for round-off error in integration assumes that after N steps, the relative round-off error is $\epsilon_{\text{ro}} \approx \sqrt{N} \epsilon_m $, where $\epsilon_m $ is the machine precision. To find the optimal number of points$N$ to minimize total error (approximation + round-off):
$$\sqrt{N} \epsilon_m = f'' \frac{(b-a)^3}{N^2},$$for trapezoid rule, and$$\sqrt{N} \epsilon_m = f^{(4)} \frac{(b-a)^5}{N^4},$$for Simpson's rule.

```java
public class OptimizeIntegration {
    public static int optimizeN(double b, double a, Function<Double, Double> fPrimePrime, Function<Double, Double> f4) {
        double epsilonM = 1e-15; // Machine precision for double
        double NTrap = Math.pow(epsilonM / (fPrimePrime.apply((a + b) / 2) * Math.pow((b - a), 3)), 2.0 / 5.0);
        
        int NSimpson = (int) Math.pow(epsilonM / (f4.apply((a + b) / 2) * Math.pow((b - a), 5)), 2.0 / 9.0);
        
        return NTrap > NSimpson ? NSimpson : NTrap;
    }
}
```
x??

--- 
#### Optimal Number of Points for Trapezoid and Simpson’s Rule
Background context explaining how to determine the optimal number of points $N$ for trapezoid and Simpson's rules based on the error estimates and round-off error model.

For small intervals and well-behaved functions, Simpson's rule should converge more rapidly and be more accurate than the trapezoid rule due to its higher inverse power of N in the error term.

:p What is the optimal number of points for trapezoid and Simpson’s rules?
??x
The optimal number of points $N$ for trapezoid and Simpson's rules can be determined by balancing the approximation error with the round-off error. For small intervals and well-behaved functions, we find:

For trapezoid rule:
$$N \approx 10^6,$$resulting in a relative round-off error of $\epsilon_{\text{ro}} \approx 10^{-12}$.

For Simpson's rule:
$$N \approx 2^{154},$$resulting in a relative round-off error of $\epsilon_{\text{ro}} \approx 5 \times 10^{-14}$.

```java
public class OptimizePoints {
    public static int trapezoidOptimalN(double epsilonM) {
        return (int) Math.pow(epsilonM, -2.0 / 5.0);
    }
    
    public static long simpsonOptimalN(double epsilonM) {
        return (long) Math.pow(epsilonM, -2.0 / 9.0);
    }
}
```
x?? 

--- 
#### Summary of Trapezoid and Simpson’s Rules
Background context summarizing the key differences between trapezoid and Simpson's rules in terms of their error estimates, approximation accuracy, and optimal number of points.

Trapezoid rule uses linear approximations (trajectories) to estimate the integral, while Simpson's rule fits parabolas. The former has a quadratic error term, whereas the latter has a quartic term, making it more accurate for smooth functions but computationally intensive due to its higher requirement for $N$.

:p What are the key differences between trapezoid and Simpson’s rules?
??x
The key differences between trapezoid and Simpson's rules lie in their error terms and approximation accuracy:

- Trapezoid rule uses linear approximations, resulting in an error term proportional to $\left( \frac{(b-a)^3}{N^2} f'' \right)$.
- Simpson's rule fits parabolic arcs, yielding a more accurate error term proportional to $\left( \frac{(b-a)^5}{N^4} f^{(4)} \right)$.

Trapezoid is simpler and faster but less accurate for smooth functions. Simpson’s method provides better accuracy with higher computational cost due to its requirement for an even number of points.

```java
public class Summary {
    public static void compareMethods(double a, double b, Function<Double, Double> fPrimePrime, Function<Double, Double> f4) {
        int NTrap = 1000000; // Trapezoid optimal N
        long NSimpson = (long) Math.pow(epsilonM / (f4.apply((a + b) / 2) * Math.pow((b - a), 5)), 2.0 / 9.0); // Simpson's optimal N
        
        System.out.println("Trapezoid: " + NTrap);
        System.out.println("Simpson's: " + NSimpson);
    }
}
```
x??
--- 

These flashcards cover the key concepts of trapezoid and Simpson’s rules, their error estimates, and optimal number of points for integration. Each card includes background context, formulas, code examples, and detailed explanations to aid in understanding and application.

