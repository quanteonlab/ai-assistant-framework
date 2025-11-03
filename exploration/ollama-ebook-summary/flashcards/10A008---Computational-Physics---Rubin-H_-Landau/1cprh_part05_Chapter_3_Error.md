# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 5)

**Starting Chapter:** Chapter 3 Errors and Uncertainties. 3.1 Types of Errors

---

#### Blunders or Bad Theory
Background context: Errors can be introduced by human mistakes, such as typos, running incorrect programs, or flawed reasoning. These errors are not influenced by computational precision but rather by human factors.

:p What is an example of a blunder?
??x
An example could include entering the wrong data file into a program or accidentally using the wrong program to solve a problem.
x??

---

#### Random Errors
Background context: These errors arise from unpredictable events, such as fluctuations in electronics, cosmic rays, or power interruptions. They are inherent and cannot be controlled but can be managed through reproducibility checks.

:p How do random errors affect computational results?
??x
Random errors can make a result unreliable over time because they increase the likelihood of incorrect outcomes as the computation runs longer.
x??

---

#### Approximation Errors
Background context: These errors occur when simplifying mathematical models to make them computable. Examples include replacing infinite series with finite sums, approximating infinitesimals with small values, and using constant approximations for variable functions.

:p What is an example of an approximation error?
??x
An example is the Taylor series expansion of \(\sin(x)\), where:
\[ \sin(x) = \sum_{n=1}^{\infty} \frac{(-1)^{n-1}}{(2n-1)!} x^{2n-1} \]
This infinite series can be approximated by a finite sum, say \(N\):
\[ \sin(x) \approx \sum_{n=1}^{N} \frac{(-1)^{n-1}}{(2n-1)!} x^{2n-1} + O(x^{N+1}) \]
The approximation error is the difference between the actual series and the finite sum.
x??

---

#### Round-off Errors
Background context: These errors arise from using a finite number of digits to store floating-point numbers. They are analogous to measurement uncertainties in experiments.

:p What is an example illustrating round-off errors?
??x
An example is storing \(\frac{1}{3}\) and \(\frac{2}{3}\) with four decimal places:
\[ 1/3 = 0.3333 \]
\[ 2/3 = 0.6667 \]
When performing a simple calculation like \(2(1/3) - 2/3\):
```python
# Python code example
result = 2 * (1/3) - 2/3
print(result)
```
The result is:
\[ 2(1/3) - 2/3 = 0.6666 - 0.6667 = -0.0001 \neq 0 \]
x??

---

#### Subtractive Cancelation and its Impact on Accuracy

Subtractive cancelation occurs when two nearly equal numbers are subtracted, leading to significant loss of precision. This is a common issue in numerical computations where exact values are approximated by finite-precision arithmetic.

The error in the result can be modeled as:
\[ a_c \approx a(1 + \epsilon_a) \]
where \( \epsilon_a \) is the relative error due to machine precision, which we assume to be of the order of machine epsilon (\(\epsilon_m\)).

If we apply this to subtraction:
\[ a = b - c \Rightarrow a_c \approx b(1 + \epsilon_b) - c(1 + \epsilon_c) \]
\[ a_c \approx b + \frac{b}{a}(\epsilon_b - \epsilon_c) \]

This expression shows that the error in \(a\) is a weighted average of the errors in \(b\) and \(c\), with potential magnification due to large values.

:p What happens when we subtract two nearly equal numbers in a calculation?
??x
When we subtract two nearly equal numbers, the result can be significantly affected by the precision limitations. The relative error in the result is not just the sum of the individual errors but can be amplified if one number is much larger than the other.

For example, consider:
```java
double a = 1000000;
double b = 999999;
double c = 1;

double result = (b - c) / a; // This operation could lead to large errors due to subtractive cancelation.
```
x??

---

#### Quadratic Equation Solutions and Subtractive Cancelation

The quadratic equation \(ax^2 + bx + c = 0\) has solutions:
\[ x_{1,2} = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

However, when \(b^2 \gg 4ac\), the square root term and its preceding term nearly cancel out, leading to significant loss of precision.

:p How does subtractive cancelation affect the solution of a quadratic equation?
??x
Subtractive cancelation affects the solution by causing large errors when the discriminant (\(b^2 - 4ac\)) is much larger than \(4ac\). This results in the subtraction of nearly equal numbers, leading to significant loss of precision.

To illustrate:
```java
public class QuadraticSolutions {
    public static double[] solveQuadratic(double a, double b, double c) {
        double discriminant = Math.sqrt(b * b - 4 * a * c);
        return new double[]{(-b + discriminant) / (2 * a), (-b - discriminant) / (2 * a)};
    }
}
```
The code above calculates the roots of the quadratic equation. If \(b^2 \gg 4ac\), the root calculation can be problematic due to subtractive cancelation.

x??

---

#### Alternating Series Summation and Subtractive Cancelation

When summing alternating series, especially those with large terms that nearly cancel out, significant errors can occur if not handled carefully. For example:
\[ S(1) = \sum_{n=1}^{2N} (-1)^{n-1} \frac{n}{n+1} \]

Summing even and odd values separately might lead to unnecessary subtractive cancelation.

:p How does the summation of an alternating series with large terms contribute to error?
??x
The summation of an alternating series, especially when large terms nearly cancel out, can introduce significant errors if handled improperly. This is because subtraction of nearly equal numbers can amplify any existing errors due to finite precision arithmetic.

For instance:
```java
public class AlternatingSeries {
    public static double sumAlternatingSeries(int N) {
        double sum1 = 0;
        for (int n = 1; n <= 2 * N; n++) {
            sum1 += Math.pow(-1, n - 1) * n / (n + 1);
        }
        return sum1;
    }
}
```
This code sums the series directly. However, this approach can lead to significant errors due to subtractive cancelation when terms are large and nearly equal.

x??

---

#### Numerical Summation of Series

When summing a series numerically, different methods can yield varying results due to the order in which terms are added or subtracted. For example:
\[ S(1) = \sum_{n=1}^{2N} (-1)^{n-1} \frac{n}{n+1} \]
\[ S(2) = -\sum_{n=1}^{N} \frac{2n-1}{2n} + \sum_{n=1}^{N} \frac{2n}{2n+1} \]
\[ S(3) = \sum_{n=1}^{N} \frac{1}{2n(2n+1)} \]

These methods can yield different numerical results, especially when dealing with alternating signs and large terms.

:p How do different summation techniques affect the accuracy of a series?
??x
Different summation techniques can significantly affect the accuracy of a series due to issues like subtractive cancelation. For example:
- Summing all terms directly (\(S(1)\)) may lead to significant errors when large alternating terms nearly cancel out.
- Separating even and odd terms separately (\(S(2)\)) might still involve unnecessary subtraction, leading to potential errors.
- Combining the series analytically (\(S(3)\)) can eliminate these issues.

For instance:
```java
public class SeriesSummation {
    public static double sumSeries1(int N) {
        double sum = 0;
        for (int n = 1; n <= 2 * N; n++) {
            sum += Math.pow(-1, n - 1) * n / (n + 1);
        }
        return sum;
    }

    public static double sumSeries2(int N) {
        double evenSum = 0, oddSum = 0;
        for (int n = 1; n <= N; n++) {
            evenSum -= (2 * n - 1) / (2 * n);
            oddSum += (2 * n) / (2 * n + 1);
        }
        return evenSum + oddSum;
    }

    public static double sumSeries3(int N) {
        double sum = 0;
        for (int n = 1; n <= N; n++) {
            sum += 1.0 / (2 * n * (2 * n + 1));
        }
        return sum;
    }
}
```
These methods illustrate how different approaches can yield varying results, with \(S(3)\) being the most accurate due to avoiding unnecessary subtractive cancelation.

x??

---

#### Summation of Simple Series and Subtractive Cancelation

Summing simple series like:
\[ S_{up} = \sum_{n=1}^{N} \frac{1}{n}, \quad S_{down} = \sum_{n=N}^{1} \frac{1}{n} \]

Can lead to different numerical results due to the order of terms and subtractive cancelation.

:p How does the order in which terms are summed affect the accuracy?
??x
The order in which terms are summed can significantly affect the accuracy due to issues like subtractive cancelation. Summing from small to large or vice versa can lead to different numerical results because subtraction of nearly equal numbers amplifies any existing errors.

For example:
```java
public class SimpleSeriesSummation {
    public static double sumUp(int N) {
        double sum = 0;
        for (int n = 1; n <= N; n++) {
            sum += 1.0 / n;
        }
        return sum;
    }

    public static double sumDown(int N) {
        double sum = 0;
        for (int n = N; n >= 1; n--) {
            sum += 1.0 / n;
        }
        return sum;
    }
}
```
These methods show that the order can affect accuracy, with \(S_{down}\) generally being more precise due to avoiding unnecessary subtractive cancelation.

x??

---

#### Round-Off Error from Division
Background context: When performing division on two numbers, represented in a computer, there can be round-off errors due to finite precision. The formula given approximates how these errors accumulate.

:p How does error arise from a single division of two computer-represented numbers?
??x
The error arises because the computer cannot represent all real numbers precisely. In equation (3.14), \( \frac{a}{b} = 1 + \epsilon_b - \epsilon_c \) where \( \epsilon_b \) and \( \epsilon_c \) are small errors due to finite precision. Ignoring higher-order terms, the total relative error in the division is approximately \( | \epsilon_b| + | \epsilon_c| \). This same rule applies to multiplication.

No code examples needed for this concept.
x??

---

#### Error Propagation from Functions
Background context: The basic rule of error propagation involves adding uncertainties when evaluating a function. For small errors, the relative change in the function's value can be approximated using its derivative.

:p How is the uncertainty in the evaluation of a general function \( f(x) \) estimated?
??x
The uncertainty in \( f(x) \) evaluated at \( x_c \) can be estimated by first-order Taylor expansion:
\[ \Delta f = f(x) - f(x_c) \approx \frac{df}{dx} f(x_c) (x - x_c). \]
For the function \( f(x) = \sqrt{1 + x} \), its derivative is:
\[ \frac{df}{dx} = \frac{1}{2\sqrt{1+x}}. \]
Thus, the relative error becomes:
\[ \Delta f \approx \frac{1}{2\sqrt{1+x}} (x - x_c). \]

For \( x = \pi/4 \) and an assumed fourth-place error in \( x \), we get a similar relative error of about \( 1.5 \times 10^{-4} \).

No code examples needed for this concept.
x??

---

#### Accumulation of Round-Off Errors
Background context: When performing calculations with many steps, round-off errors can accumulate and be modeled as a random walk. The total distance \( R \) covered in \( N \) steps is approximately:
\[ R \approx \sqrt{N r^2}. \]
Similarly, the total relative error after \( N \) calculation steps each with machine precision error \( \epsilon_m \), on average, accumulates as:
\[ \epsilon_{ro} \approx \sqrt{N \epsilon_m}. \]

:p How does round-off error accumulate in a long sequence of calculations?
??x
Round-off errors can be modeled as a random walk. The total relative error after \( N \) steps each with machine precision error \( \epsilon_m \), on average, is:
\[ \epsilon_{ro} \approx \sqrt{N \epsilon_m}. \]
This means the round-off error grows slowly and randomly with \( N \). If errors in each step are uncorrelated, this model accurately predicts their accumulation.

No code examples needed for this concept.
x??

---

#### Convergence of Algorithms
Background context: The performance of algorithms is crucial in computational physics. Both algorithmic errors (decreasing as a power law) and round-off errors (growing slowly but randomly) need to be considered.

:p How do you determine the best number of steps in an algorithm?
??x
To determine the best number of steps, compare the approximation error \( \epsilon_{app} \approx \alpha N^{-\beta} \) with the round-off error \( \epsilon_{ro} \approx \sqrt{N \epsilon_m} \). The total error is:
\[ \epsilon_{tot} = \epsilon_{app} + \epsilon_{ro}. \]
The optimal number of steps occurs when these two errors are equal, i.e., \( N^{5/2} \propto 4 \epsilon_m \).

For double precision (where \( \epsilon_m \approx 10^{-15} \)), the minimum total error occurs at:
\[ N \approx 10^99. \]

No code examples needed for this concept.
x??

---

#### Analyzing Numerical Integration Errors
Background context: In numerical integration, such as Simpson's rule, understanding how errors behave is crucial to determine the number of points required for desired precision.

:p How do you analyze the relative error in numerical integration using a log-log plot?
??x
To analyze the relative error in numerical integration, use a log-log plot. For example, with Simpson's rule, the relative error \( \epsilon_{app} \) should show rapid decrease for small \( N \). Beyond this region, round-off errors start to dominate.

Plotting \( \log_{10}\left|\frac{A(N) - A(2N)}{A(2N)}\right| \) versus \( \log_{10}(N) \) helps identify the convergence region and the level of precision.

No code examples needed for this concept.
x??

---

#### Example of Different Errors
Background context: Analyzing both approximation and round-off errors can help in optimizing algorithms. For instance, if the approximation error is \( \epsilon_{app} = \frac{1}{N^2} \) and the round-off error is \( \epsilon_{ro} = \sqrt{N \epsilon_m} \), their sum determines the overall error.

:p How do you find the optimal number of steps for an algorithm given both approximation and round-off errors?
??x
Given:
\[ \epsilon_{app} \approx \frac{1}{N^2}, \quad \epsilon_{ro} \approx \sqrt{N \epsilon_m}. \]
The total error is:
\[ \epsilon_{tot} = \frac{1}{N^2} + \sqrt{N \epsilon_m}. \]

To minimize this, take the derivative with respect to \( N \):
\[ \frac{d\epsilon_{tot}}{dN} = -\frac{2}{N^3} + \frac{\epsilon_m^{1/2}}{2\sqrt{N}} = 0. \]
Solving for \( N \) gives:
\[ N^{5/2} = 4 \epsilon_m, \quad N = (4 \epsilon_m)^{2/5}. \]

For double precision (\( \epsilon_m \approx 10^{-15} \)):
\[ N = (4 \times 10^{-15})^{2/5} \approx 10^99. \]

The minimum total error is approximately \( 4 \times 10^{-6} \).

No code examples needed for this concept.
x??

--- 

These flashcards cover the key concepts from the provided text, each focusing on a specific aspect of numerical errors and their analysis in algorithms.

