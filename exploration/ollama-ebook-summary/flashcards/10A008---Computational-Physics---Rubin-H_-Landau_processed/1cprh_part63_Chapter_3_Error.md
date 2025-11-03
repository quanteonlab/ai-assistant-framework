# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 63)

**Starting Chapter:** Chapter 3 Errors and Uncertainties. 3.1 Types of Errors

---

#### Blunders or Bad Theory
Background context explaining the concept of blunders or bad theory. These errors are introduced by humans and can include typographical errors, running incorrect programs, or flawed reasoning. They do not stem from computational limitations but rather human mistakes.

:p What is an example of a blunder or bad theory error?
??x
Examples of blunders or bad theory errors include entering the wrong data file, using an incorrect program, or having logical flaws in your reasoning (theory). For instance, if you mistakenly input `2 + 2 = 5` instead of `2 + 2 = 4`, this would be a typographical error.
x??

---

#### Random Errors
Background context explaining the concept of random errors. These are imprecisions caused by external events like fluctuations in electronics or cosmic rays, which cannot be controlled and increase with running time.

:p What is an example of a random error?
??x
Random errors can occur due to uncontrollable factors such as fluctuations in electronic components or cosmic rays interfering with the computation. For instance, if you are measuring something using a device that experiences voltage fluctuations due to electromagnetic interference from nearby devices, this could cause random variations in your results.
x??

---

#### Approximation Errors
Background context explaining approximation errors and their sources. These arise from simplifying mathematical problems so they can be solved computationally, such as replacing infinite series with finite sums or using constants instead of variable functions.

:p What is an example of an approximation error?
??x
Approximation errors occur when simplifications are made to mathematical models for computational feasibility. For example, the Taylor series expansion of \( \sin(x) \) is given by:
\[ \sin(x) = \sum_{n=1}^{\infty} \frac{(-1)^{n-1}}{(2n-1)!} x^{2n-1} \]
However, in practice, it may be approximated as:
\[ \sin(x) \approx \sum_{n=1}^{N} \frac{(-1)^{n-1}}{(2n-1)!} x^{2n-1} + O\left(\frac{x^{2N+1}}{(2N+1)!}\right) \]
where \( N \) is the number of terms used, and \( O\left(\frac{x^{2N+1}}{(2N+1)!}\right) \) represents the approximation error. If \( x \) and \( N \) are close in value, this error can be large.
x??

---

#### Round-off Errors
Background context explaining round-off errors and their origins from finite precision storage of floating-point numbers.

:p What is an example of a round-off error?
??x
Round-off errors arise due to the limited number of digits used to store floating-point numbers. For instance, if your computer stores only four decimal places, it would represent \( \frac{1}{3} \) as 0.3333 and \( \frac{2}{3} \) as 0.6667. Performing a simple calculation like:
\[ 2\left(\frac{1}{3}\right) - \frac{2}{3} = 2(0.3333) - 0.6667 = 0.6666 - 0.6667 = -0.0001 \neq 0 \]
shows that the result is not exactly zero, even though it should be.
x??

---
Each flashcard covers a different type of error in computational physics, providing clear explanations and examples for each concept.

#### Subtractive Cancelation Overview
Background context explaining subtractive cancelation. It occurs when two nearly equal numbers are subtracted, leading to significant errors in the result. The error is a weighted average of the relative errors in the operands and can be magnified by large factors.

:p What is subtractive cancelation?
??x
Subtractive cancelation is an error that occurs when performing subtraction with approximately represented numbers on computers. It happens because the difference between two nearly equal numbers leaves only the least significant parts, which are prone to rounding errors.
In C/Java, this can be illustrated by:
```java
float a = 100.0f;
float b = 99.999f;
float c = a - b; // This may result in an error due to subtractive cancelation
```
x??

---

#### Theoretical Analysis of Subtractive Cancelation
An analytical approach to understanding the impact of subtractive cancelation on calculations.

:p How does subtractive cancelation affect the accuracy of results?
??x
Subtractive cancelation affects the accuracy by introducing significant errors when two nearly equal numbers are subtracted. This is because the least significant parts of both numbers contribute most to the error, which can be magnified if the operands are large.
The formula provided in the text shows how the relative error in the result increases with:
\[ \text{error} = 1 + \frac{\epsilon_b - \epsilon_c}{b/a} \]
where \( \epsilon_b \) and \( \epsilon_c \) are the relative errors of b and c, respectively.
x??

---

#### Quadratic Equation Solutions
Exploring how subtractive cancelation affects solutions to quadratic equations.

:p How can subtractive cancelation affect the solutions to a quadratic equation?
??x
Subtractive cancelation in solving quadratic equations occurs when \(b^2 \gg 4ac\). The standard form of the solution:
\[ x_{1,2} = -\frac{b \pm \sqrt{b^2 - 4ac}}{2a} \]
can lead to large relative errors if \(b\) is much larger than both \(2a\sqrt{c}\) and \(2a\sqrt{c}\), causing the square root term to nearly cancel with the linear term.
This can be demonstrated by:
```java
public class QuadraticEquation {
    public static void main(String[] args) {
        float a = 1.0f;
        float b = 1000000.0f;
        float c = 1.0f;
        
        float discriminant = (float)Math.sqrt(b*b - 4*a*c);
        float x1 = (-b + discriminant) / (2*a);
        float x2 = (-b - discriminant) / (2*a);
        
        System.out.println("x1: " + x1);
        System.out.println("x2: " + x2);
    }
}
```
x??

---

#### Series Summation and Alternating Signs
Exploring the effect of subtractive cancelation in series summation, particularly with alternating signs.

:p How does subtractive cancelation affect the summation of a series with alternating signs?
??x
Subtractive cancelation can significantly impact the accuracy of summing a series with alternating signs. For example, consider:
\[ S = \sum_{n=1}^{2N} (-1)^n \frac{n}{n+1} \]
Summing even and odd terms separately or directly combining the series can mitigate this effect.

In C/Java, this can be demonstrated by:
```java
public class AlternatingSeries {
    public static void main(String[] args) {
        int N = 1000;
        double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        
        for (int n = 1; n <= 2*N; n++) {
            sum1 += Math.pow(-1, n) * n / (n + 1);
        }
        
        int evenTerms = N/2;
        int oddTerms = N - evenTerms;
        
        for (int i = 0; i < evenTerms; i++) {
            sum2 -= (2*i+1) / (2*i+2);
        }
        
        for (int j = 0; j < oddTerms; j++) {
            sum3 += (2*j + 2) / (2*j + 3);
        }
        
        System.out.println("Sum of all terms: " + sum1);
        System.out.println("Combined even and odd sums: " + (sum2 + sum3));
    }
}
```
x??

---

#### Numerical Summation Techniques
Exploring different summation techniques to avoid subtractive cancelation.

:p How can we avoid the effects of subtractive cancelation in numerical summation?
??x
To avoid subtractive cancelation, one can rearrange or combine series terms. For example:
\[ S(1) = \sum_{n=1}^{N} (-1)^n n / (n+1) \]
can be rewritten as:
\[ S(3) = \sum_{n=1}^{N} 1 / ((2n)(2n+1)) \]
which avoids the subtraction of nearly equal terms and can provide more accurate results.

In C/Java, this can be demonstrated by:
```java
public class SummationTechniques {
    public static void main(String[] args) {
        int N = 1000;
        
        double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        
        for (int n = 1; n <= N; n++) {
            sum1 += Math.pow(-1, n) * n / (n + 1);
        }
        
        double totalSum = 0.0;
        for (int n = 1; n <= N; n++) {
            totalSum += 1.0 / ((2*n)*(2*n+1));
        }
        
        System.out.println("Direct summation: " + sum1);
        System.out.println("Combined series summation: " + totalSum);
    }
}
```
x??

---

#### Summation of Simple Series
Exploring the numerical issues in summing simple series.

:p How can we avoid subtractive cancelation when summing a simple series?
??x
To avoid subtractive cancelation, it is important to rearrange or combine terms. For example:
\[ S_{up} = \sum_{n=1}^{N} 1/n \]
and
\[ S_{down} = \sum_{n=N}^{1} 1/n \]
both give the same analytical result but can have different numerical results due to round-off errors.

In C/Java, this can be demonstrated by:
```java
public class SimpleSeries {
    public static void main(String[] args) {
        int N = 100;
        
        double sumUp = 0.0, sumDown = 0.0;
        
        for (int n = 1; n <= N; n++) {
            sumUp += 1.0 / n;
        }
        
        for (int n = N; n >= 1; n--) {
            sumDown += 1.0 / n;
        }
        
        System.out.println("Sum from up: " + sumUp);
        System.out.println("Sum from down: " + sumDown);
    }
}
```
x??

#### Round-Off Error in Division

Background context explaining how round-off errors arise from division operations. The formula given is:
\[ c = \frac{a}{b} \Rightarrow ac = bc(1 + \epsilon_b) (c(1 + \epsilon_c)) \]

This simplifies to:
\[ a = 1 + \epsilon_b (1 - \epsilon_c) \approx 1 + |\epsilon_b| + |\epsilon_c|. \]

Here, we have ignored the very small \(\epsilon^2\) terms and added the absolute values of the errors. This same rule applies for multiplication.

:p How does round-off error propagate in a simple division operation?
??x
The round-off error propagates by adding the absolute values of individual errors in the operands. For example, if we divide \(a\) by \(b\), the error \(\epsilon_a\) in \(a\) and \(\epsilon_b\) in \(b\) combine to give an overall relative error.

For a division:
\[ c = \frac{a}{b} \Rightarrow ac \approx bc(1 + \epsilon_b)(1 - \epsilon_c) \approx 1 + |\epsilon_b| + |\epsilon_c|. \]

This means the total relative error is approximately the sum of individual errors.
x??

---

#### General Model for Error Propagation in Functions

The text extends the basic rule to functions \(f(x)\). The key formula given is:
\[ \Delta f = f(x) - f(x_c) \approx \frac{df}{dx} f(x_c)(x - x_c) \]

Specifically, for the function:
\[ f(x) = \sqrt{1 + x}, \quad \frac{df}{dx} = \frac{1}{2\sqrt{1 + x}}. \]

This leads to an approximation of the error:
\[ \Delta f \approx \frac{1}{2\sqrt{1 + x_c}} (x - x_c) = \frac{x - x_c}{2(1 + x)}. \]

If we evaluate this expression for \(x = \pi/4\) and assume an error in the fourth place, a similar relative error of about \(1.5 \times 10^{-4}\) is obtained.

:p How does the general model estimate the error in evaluating a function?
??x
The general model estimates the error in evaluating a function by approximating the change in the function value due to small changes in the input. For a function \(f(x)\), the error \(\Delta f\) can be estimated using the derivative of the function:
\[ \Delta f \approx \frac{df}{dx} f(x_c)(x - x_c). \]

For example, for \(f(x) = \sqrt{1 + x}\):
\[ \frac{df}{dx} = \frac{1}{2\sqrt{1 + x}}, \]
and the error approximation is:
\[ \Delta f \approx \frac{x - x_c}{2(1 + x)}. \]

This method helps in understanding how small changes in input values affect the function's output.
x??

---

#### Round-Off Error Accumulation

The text provides a useful model for approximating how round-off errors accumulate over multiple steps of a calculation. It uses the analogy of a random walk to describe the error propagation.

The key formula given is:
\[ R \approx \sqrt{N} \epsilon_m, \]
where \(R\) is the total relative error after \(N\) steps and \(\epsilon_m\) is the machine precision error per step.

:p How does round-off error accumulate over multiple steps in a calculation?
??x
Round-off errors accumulate over multiple steps of a calculation by following a random walk model. Each step introduces an error proportional to the machine precision \(\epsilon_m\). After \(N\) steps, the total relative error can be approximated as:
\[ \epsilon_{ro} \approx \sqrt{N} \epsilon_m. \]

This formula indicates that the overall error increases with the square root of the number of steps.

For example, if a calculation involves 1000 steps and each step introduces an error of \(10^{-6}\), the total relative error would be approximately:
\[ \epsilon_{ro} \approx \sqrt{1000} \times 10^{-6} \approx 0.032\%. \]

This shows that even small errors can accumulate significantly over many steps.
x??

---

#### Algorithmic and Round-Off Error in Algorithms

The text distinguishes between algorithmic error, which decreases with the number of terms used, and round-off error, which increases due to machine precision limits.

Key formulas given are:
\[ \epsilon_{app} \approx \alpha N^{-\beta}, \]
and
\[ \epsilon_{ro} \approx \sqrt{N} \epsilon_m. \]

The total error is the sum of these two types:
\[ \epsilon_{tot} = \epsilon_{app} + \epsilon_{ro}. \]

The goal is to find \(N\) such that the algorithmic and round-off errors balance, minimizing the total error.

:p How do you determine the best number of steps for an algorithm?
??x
To determine the best number of steps for an algorithm, you need to balance the algorithmic error and the round-off error. The general form is:
\[ \epsilon_{tot} = \alpha N^{-\beta} + \sqrt{N} \epsilon_m. \]

You want to find \(N\) such that these two errors are comparable. Typically, this involves plotting:
\[ \log_{10} \left| \frac{A(N) - A(2N)}{A(2N)} \right| \]
versus \(\log_{10} N\).

If the error decreases rapidly at first but then starts to increase due to round-off errors, you can find the optimal \(N\) where these two errors balance.

For example:
```java
public class ErrorAnalysis {
    public static void main(String[] args) {
        double N = 10;
        while (true) {
            double error = Math.abs((A(N) - A(2 * N)) / A(2 * N));
            if (error > threshold) {
                // Increase N to reduce round-off errors
                N *= 2;
            } else {
                // Decrease N to minimize algorithmic errors
                break;
            }
        }
    }

    private static double A(double N) {
        // Implementation of the function approximation and error calculations
    }
}
```
x??

---

#### Example Error Calculation

The text provides an example where both types of errors are known:
\[ \epsilon_{app} \approx \frac{1}{N^2}, \quad \epsilon_{ro} \approx \sqrt{N} \epsilon_m. \]

Combining these, the total error is:
\[ \epsilon_{tot} = \frac{1}{N^2} + \sqrt{N} \epsilon_m. \]

To minimize this total error, solve for \(N\):
\[ \frac{d \epsilon_{tot}}{dN} = -\frac{2}{N^3} + \frac{1}{2\sqrt{N} \epsilon_m} = 0. \]
This simplifies to:
\[ N^{5/2} = 4 \epsilon_m, \]

For double-precision (\(\epsilon_m \approx 10^{-15}\)):
\[ N^{5/2} \approx 4 \times 10^{-15}, \]
giving \(N \approx 10^99\).

:p How do you calculate the optimal number of steps for an algorithm when both approximation and round-off errors are known?
??x
To find the optimal number of steps \(N\) when both approximation error and round-off error are known, follow these steps:

Given:
\[ \epsilon_{app} \approx \frac{1}{N^2}, \quad \epsilon_{ro} \approx \sqrt{N} \epsilon_m. \]

The total error is:
\[ \epsilon_{tot} = \frac{1}{N^2} + \sqrt{N} \epsilon_m. \]

To minimize this, take the derivative with respect to \(N\) and set it to zero:
\[ \frac{d \epsilon_{tot}}{dN} = -\frac{2}{N^3} + \frac{1}{2 \sqrt{N} \epsilon_m} = 0. \]

Solving this equation gives:
\[ N^{5/2} = 4 \epsilon_m. \]

For double-precision (\(\epsilon_m \approx 10^{-15}\)):
\[ N^{5/2} \approx 4 \times 10^{-15}, \]
so \(N \approx 10^99\).

Thus, the optimal number of steps is about \(10^99\) to balance both types of errors.
x?? 

--- 
Note: The values and specific calculations can vary based on actual precision levels used.

