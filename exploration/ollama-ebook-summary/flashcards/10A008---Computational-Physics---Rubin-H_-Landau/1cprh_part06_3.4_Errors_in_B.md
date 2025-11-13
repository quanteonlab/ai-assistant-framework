# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 6)

**Starting Chapter:** 3.4 Errors in Bessel Functions

---

#### Total Error Calculation for Measurement

Background context: In measurement, the total error is a combination of random and systematic errors. The formula provided gives an approximation of the total error $\epsilon_{\text{tot}}$ which combines random ($\epsilon_{\text{ro}}$) and proportional ($\epsilon_{\text{app}}$) uncertainties.

The total error equation:
$$\epsilon_{\text{tot}} = \epsilon_{\text{ro}} + \epsilon_{\text{app}} \approx 2N^4 + \sqrt{N}\epsilon_m$$

To find the number of points for minimum error, we take the derivative of $\epsilon_{\text{tot}}$ with respect to $N$ and set it to zero:
$$\frac{d\epsilon_{\text{tot}}}{dN} = 0 \Rightarrow N^{9/2} \Rightarrow N \approx 67 \Rightarrow \epsilon_{\text{tot}} \approx 9 \times 10^{-7}$$:p What is the formula for total error in this context?
??x
The formula given is:
$$\epsilon_{\text{tot}} = \epsilon_{\text{ro}} + \epsilon_{\text{app}} \approx 2N^4 + \sqrt{N}\epsilon_m.$$

This combines the random and proportional errors in a measurement context.

x??

---

#### Double-Precision Calculation Error Estimation

Background context: This section discusses estimating the error for double-precision calculations. The example provided uses trigonometric functions, specifically sine calculation, to demonstrate precision issues with finite precision arithmetic.

:p Estimate the error for a double-precision calculation.
??x
The question is about determining how errors propagate in double-precision calculations, especially when dealing with trigonometric functions like sine. This involves understanding the limitations of floating-point arithmetic and the need for careful algorithm design to minimize errors.

x??

---

#### Summation of Power Series

Background context: The summation of a power series to approximate functions such as $\sin(x)$ is discussed. A finite number of terms in the series are used, and the challenge is deciding when to stop summing based on desired accuracy.

The series for $\sin(x)$:
$$\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$$

One approach to stopping the summation is to use the last term as a criterion for convergence:
$$\left|\frac{\text{nth term}}{\text{sum}}\right| < 10^{-8}$$:p What is the criterion used to stop summing in this series?
??x
The criterion to stop summing in this series is when the absolute value of the nth term divided by the accumulated sum is less than $10^{-8}$:
$$\left|\frac{\text{nth term}}{\text{sum}}\right| < 10^{-8}$$x??

---

#### Algorithmic Implementation for $\sin(x)$ Background context: This section details an algorithm to compute $\sin(x)$ using a power series. The goal is to achieve relative error less than $10^{-8}$.

The pseudocode given:
```plaintext
term = x, sum = x, eps = 10^(-8)
do
    term = -term * x * x / (2n-1) / (2*n-2)
    sum = sum + term
while abs(term/sum) > eps
```

:p What is the pseudocode for computing $\sin(x)$?
??x
The pseudocode provided for computing $\sin(x)$ using a power series to achieve relative error less than $10^{-8}$ is:
```plaintext
term = x, sum = x, eps = 10^(-8)
do
    term = -term * x * x / (2n-1) / (2*n-2)
    sum = sum + term
while abs(term/sum) > eps
```

x??

---

#### Precision and Convergence of the Algorithm

Background context: The convergence of the algorithm for computing $\sin(x)$ is discussed, along with considerations to avoid issues like overflows and unnecessary computational expense.

:p Explain why using $2n-1 $ and$x^{2n-1}$ directly can be problematic.
??x
Using $2n-1 $ and$x^{2n-1}$ directly in the algorithm can lead to overflow problems because both terms individually can become very large, even though their quotient might be small. This can cause computational errors.

x??

---

#### Error Behavior of Series Summation

Background context: The error behavior for the series summation is discussed, noting that it does not accumulate randomly as in more complex computations. Instead, there are predictable patterns in the accumulation of errors.

:p Describe the error behavior for the series summation.
??x
The error behavior for the series summation is such that because the process is simple and correlated, round-off errors do not accumulate randomly as they might in a more complicated computation. The error increases and decreases predictably with the number of terms added, showing patterns similar to those seen in Figure 3.3.

x??

---

#### Specular Reflection

Background context: This section discusses specular reflection within circular mirrors, where light rays reflect according to the law of reflection, effectively leading to infinite internal reflections if no absorption occurs.

The equation for angle after each reflection:
$$\theta_{\text{new}} = \theta_{\text{old}} + 2\phi$$:p What is the equation for the new angle after a reflection?
??x
The equation for the new angle after a reflection in a circular mirror, given an initial angle $\phi$, is:
$$\theta_{\text{new}} = \theta_{\text{old}} + 2\phi.$$x??

---

#### Experimentally Determining Series Convergence

Background context: The section describes how to experimentally determine when a series starts losing accuracy and no longer converges by increasing the value of $x$ incrementally.

:p How can you determine when the series for $\sin(x)$ starts to lose accuracy?
??x
To determine when the series for $\sin(x)$ starts to lose accuracy, you can incrementally increase $x$ from 1 to 10 and then from 10 to 100 using a program that implements the pseudocode provided. By observing the results, you can identify at what point the series begins to lose precision.

x??

---

#### Graphing Error vs. Number of Terms

Background context: The section mentions creating graphs of error versus the number of terms for different values of $x$. These graphs should show similar behavior to those in Figure 3.3.

:p What is the objective of graphing the error versus the number of terms?
??x
The objective of graphing the error versus the number of terms is to visualize how the accuracy of the series summation changes as more terms are added for different values of $x$. This helps in understanding the convergence and stability of the algorithm.

x??

---

#### Using Trigonometric Identities

Background context: The use of trigonometric identities can help improve the accuracy of the $\sin(x)$ calculation, especially for large values of $x$ where direct summation might fail to converge properly.

:p How can you use the identity $\sin(x + 2n\pi) = \sin(x)$?
??x
To use the identity $\sin(x + 2n\pi) = \sin(x)$, you can reduce large values of $ x$by subtracting multiples of $2\pi$. This helps ensure that the series summation converges to the correct answer, even for very large $ x$.

x??

---

#### Tolerance Level and Machine Precision

Background context: The section discusses setting a tolerance level lower than machine precision and observing its effect on the conclusions drawn from the calculations.

:p What happens when you set your tolerance level to a number smaller than machine precision?
??x
When you set your tolerance level to a number smaller than machine precision, the calculation may not be able to achieve the desired accuracy due to inherent limitations in floating-point representation. This can lead to inaccurate results and affect the reliability of the algorithm.

x??

---

#### Light Ray Trajectories on Reflecting Mirrors

Background context: The problem involves determining the path of a light ray reflecting off a mirror. This is an application of geometric optics where the behavior of light rays can be analyzed using simple trigonometric relationships and periodic properties.

:p Determine the path followed by a light ray for a perfectly reflecting mirror.
??x
To determine the path, consider the initial angle $\phi $ and how the ray reflects off the mirror. The key is to understand that adding or subtracting$2\pi $ to the angle$\theta$ does not change its location on the circle.

If $\frac{\phi}{\pi}$ is a rational number, i.e.,$\frac{\phi}{\pi} = \frac{n}{m}$, the ray will form a geometric figure due to periodicity. 

For example:
- If $\phi = 0$, the light ray initially travels along the positive x-axis and reflects symmetrically.
- For different initial angles, the path can be traced by calculating the reflection angle using Snell's law or simple trigonometric transformations.

Here is a Python pseudocode to trace the light trajectories:

```python
import math

def reflect_ray(initial_angle, n_steps):
    theta = initial_angle  # Initial angle in radians
    for _ in range(n_steps):
        # Reflecting off mirror using Snell's law simplification (for simplicity)
        reflected_angle = -theta + 2 * math.pi  # Mirror reflection logic
        
        print(f"Step {_+1}, Angle: {reflected_angle}")

# Example usage:
reflect_ray(0, 5)  # Tracing the path for 5 steps
```

This code reflects a ray off the mirror and prints out the angle at each step.

x??

---

#### Accumulating Round-off Errors

Background context: The problem highlights the issue of accumulating round-off errors in numerical calculations. These errors can limit the accuracy of computations, especially with many steps involved. The example uses Python's `round` function to demonstrate how significant relative errors can accumulate when using finite precision arithmetic.

:p Explain the significance of using fewer places of precision in computational results.
??x
Using fewer decimal places or significant figures (e.g., rounding) can lead to significant relative errors accumulating over multiple steps, especially in complex calculations with many iterations. This is because small round-off errors at each step can compound and affect the final result.

For instance:

```python
initial_value = 1.234567890123456789
rounded_value_3 = round(initial_value, 3)
print(rounded_value_3)  # Output: 1.235

rounded_value_6 = round(initial_value, 6)
print(rounded_value_6)  # Output: 1.234568
```

Here, rounding `initial_value` to three decimal places results in `1.235`, and when rounded to six decimal places, it becomes `1.234568`. The difference between these values is significant relative to the original value.

These accumulated errors can lead to substantial discrepancies in final computations, limiting the accuracy of numerical methods like those used for spherical Bessel functions.

x??

---

#### Spherical Bessel Functions

Background context: Spherical Bessel functions $j_l(x)$ are solutions to the differential equation given by:

$$x^2 f''(x) + 2x f'(x) + [x^2 - l(l+1)] f(x) = 0.$$

These functions are related to the more general Bessel function of the first kind $J_n(x)$. The spherical Bessel functions have applications in various physical problems, such as expanding plane waves into spherical partial waves.

:p Explain the relationship between spherical Bessel functions and Bessel functions.
??x
The spherical Bessel functions $j_l(x)$ are related to the Bessel function of the first kind $J_n(x)$ by:

$$j_l(x) = \sqrt{\frac{\pi}{2 x}} J_{l+\frac{1}{2}}(x).$$

This relationship allows us to use properties and values of Bessel functions in calculations involving spherical Bessel functions. For instance, the explicit forms for $l=0 $ and$l=1$ are:
$$j_0(x) = \frac{\sin x}{x},$$
$$j_1(x) = \frac{\sin x}{x^2} - \frac{\cos x}{x}.$$

These functions can be used in physical problems, such as expanding a plane wave into spherical partial waves.

x??

---

#### Numerical Recursion for Spherical Bessel Functions

Background context: The problem describes using recursion relations to compute the values of spherical Bessel functions efficiently. These recursive formulas are given by:
$$j_{l+1}(x) = \frac{2l+1}{x} j_l(x) - j_{l-1}(x),$$
$$j_{l-1}(x) = \frac{2l+1}{x} j_l(x) - j_{l+1}(x).$$

These relations permit rapid computation of the spherical Bessel functions for a fixed $x $ and all$l$.

:p Explain the upward recursion relation for computing spherical Bessel functions.
??x
The upward recursion relation is used to compute higher-order spherical Bessel functions from lower ones. Starting with known values (e.g., $j_0(x)$ and $j_1(x)$), we can use the formula:

$$j_{l+1}(x) = \frac{2l+1}{x} j_l(x) - j_{l-1}(x).$$

This relation allows us to compute $j_{l+1}(x)$ from $j_l(x)$ and $j_{l-1}(x)$, making it an efficient method for calculating the entire set of spherical Bessel functions.

For example, if we know $j_0(x) = \frac{\sin x}{x}$ and $j_1(x) = \frac{\sin x}{x^2} - \frac{\cos x}{x}$, we can use:

$$j_2(x) = \frac{3}{x} j_1(x) - j_0(x).$$

This process is repeated to compute higher-order functions.

x??

---

#### Numerical Errors in Bessel Functions
Background context: When computing Bessel functions numerically, even though we start with a pure $j_l(x)$, the computer's lack of precision can introduce errors due to the admixture of $ n_l(x)$. This is because both $ j_l$and $ n_l$ satisfy the same differential equation, leading to similar recurrence relations. When the numerical value of $ n_l(x)$becomes much larger than that of $ j_l(x)$, even a small admixture can lead to significant errors.

:p What is Miller's device for reducing numerical errors in Bessel functions?
??x
Miller's device involves using downward recursion starting at a large value $l = L $. This avoids subtractive cancellation by taking the small values of $ j_{l+1}(x)$and $ j_l(x)$, producing a larger $ j_{l-1}(x)$through addition. While the error may still behave like a Neumann function, the actual magnitude of the error decreases quickly as we move downward to smaller $ l$ values.

The relative value will be accurate, but the absolute value needs normalization based on the known value:
$$j_N^l(x) = \frac{j_c^l(x)}{j_{c0}(x)/j_0^{\text{anal}}(x)}$$

Where $j_0^{\text{anal}}(x) = \sin x / x$.

:x??

---

#### Downward Recursion in Bessel Functions
Background context: The downward recursion method for computing Bessel functions starts at a large value of $l$ and moves downwards, avoiding subtractive cancellation. This method normalizes the computed values to ensure accurate relative values.

:p What is the formula used for normalization during downward recursion?
??x
The normalization formula ensures that while the absolute value might not be exact due to initial arbitrary values, the relative values are correct:
$$j_N^l(x) = \frac{j_c^l(x)}{j_{c0}(x)/j_0^{\text{anal}}(x)}$$

Where $j_0^{\text{anal}}(x) = \sin x / x$.

This normalization is crucial for maintaining the accuracy of relative values, especially when starting with arbitrary initial values.

:x??

---

#### Comparison of Upward and Downward Recursion
Background context: Comparing upward and downward recursion methods helps in assessing their stability and accuracy. Both methods can give similar answers for certain values of $x$, but understanding their convergence and stability is important.

:p How do you compare the results from upward and downward recursion?
??x
To compare the results, print out the relative difference between the values obtained using both methods:
$$\text{Relative Difference} = \frac{|j_{\text{up}}^l - j_{\text{down}}^l|}{|j_{\text{up}}^l| + |j_{\text{down}}^l|}$$

This helps in evaluating the convergence and stability of each method.

:x??

---

#### Stability and Convergence
Background context: The stability and convergence of Bessel function calculations are crucial for accurate results. Using downward recursion starting from a large $l = L $ can help reduce numerical errors, but both methods need to be tested for different values of$x$.

:p What is the reason that certain values of $x$ might give similar answers using both upward and downward recursions?
??x
For certain values of $x $, both upward and downward recursions may give similar answers because the admixture of small errors in numerical computations can be minimized when the functions are well-behaved. Additionally, for some specific ranges of $ x$, the relative differences between the functions might not be significant.

:x??

---

#### Implementation: Bessel.py
Background context: The provided code snippet is a Python implementation using Visual Python (VP) to determine spherical Bessel functions via downward recursion. Modifying this code to include upward recursion can help in comparing both methods.

:p What does the `down` function do in the Bessel.py code?
??x
The `down` function calculates the spherical Bessel function $j_l(x)$ using downward recursion, starting from a large value $l = L$. It initializes the array and uses recurrence relations to compute the values step by step.

```python
def down(x, n, m):
    # Method down, recurs downward
    j = zeros((start + 2), float)
    j[m+1] = j[m] = 1.0  # Start with anything
    for k in range(m, 0, -1):
        j[k-1] = ((2 * k + 1) / x) * j[k] - j[k+1]
    scale = (sin(x)/x) / j[0]  # Scale solution to known j[0]
    return j[n] * scale
```

:x??

