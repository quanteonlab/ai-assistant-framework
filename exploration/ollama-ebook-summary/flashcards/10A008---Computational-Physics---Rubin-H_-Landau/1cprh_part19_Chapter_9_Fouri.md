# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 19)

**Starting Chapter:** Chapter 9 Fourier Analyses. 9.1 Fourier Series

---

#### Fourier Series Introduction
Background context explaining the concept. The text discusses expanding solutions of nonlinear oscillators into a series of sinusoidal functions (Fourier series). A periodic function can be expressed as a sum of sine and cosine terms with frequencies that are integer multiples of the fundamental frequency.

:p What is the purpose of using Fourier series in analyzing nonlinear oscillators?
??x
The purpose of using Fourier series in analyzing nonlinear oscillators is to decompose the complex periodic motion into simpler harmonic components. This allows for easier analysis and understanding of the system's behavior, especially when the initial transient states have died out.

---

#### Fourier Series Representation
Relevant formulas include expressing a periodic function \( y(t) \):
\[ y(t) = a_0 + \sum_{n=1}^{\infty}(a_n \cos n\omega t + b_n \sin n\omega t). \]

This equation represents the signal as a sum of pure tones with frequencies that are multiples of the fundamental frequency.

:p What is the general form of a Fourier series for a periodic function?
??x
The general form of a Fourier series for a periodic function \( y(t) \) is:
\[ y(t) = a_0 + \sum_{n=1}^{\infty}(a_n \cos n\omega t + b_n \sin n\omega t). \]
This representation decomposes the signal into its harmonic components, where each term represents a sine or cosine wave with frequency \( n\omega \).

---

#### Fourier Series Coefficients
The coefficients \( a_n \) and \( b_n \) are determined by multiplying both sides of the series equation by \( \cos(n\omega t) \) or \( \sin(n\omega t) \), integrating over one period, and then projecting to find each coefficient.

Relevant formulas:
\[ (a_n bn) = \frac{2}{T} \int_{0}^{T} y(t) (\cos n\omega t \text{ or } \sin n\omega t) dt. \]

:p How are the coefficients \( a_n \) and \( b_n \) calculated in a Fourier series?
??x
The coefficients \( a_n \) and \( b_n \) in a Fourier series are calculated by integrating the product of the function \( y(t) \) and either \( \cos(n\omega t) \) or \( \sin(n\omega t) \) over one period. The formulas for determining these coefficients are:
\[ (a_n bn) = \frac{2}{T} \int_{0}^{T} y(t) (\cos n\omega t \text{ or } \sin n\omega t) dt, \]
where \( \omega = \frac{2\pi}{T} \).

---

#### Periodic Functions and Fourier Series
Background context: A periodic function can be expanded into a series of harmonic functions with frequencies that are multiples of the fundamental frequency. This is possible due to Fourier's theorem, which states that any single-valued periodic function with only a finite number of discontinuities can be represented by such a series.

:p What does Fourier’s theorem state?
??x
Fourier’s theorem states that any single-valued periodic function with only a finite number of discontinuities can be represented as a sum of sine and cosine functions, i.e., it can be expanded into a Fourier series. This means that the behavior of such functions over time can be approximated by adding together waves of different frequencies.

---

#### Application to Nonlinear Oscillators
Background context: The text discusses applying Fourier series to analyze periodic but non-sinusoidal motions resulting from nonlinear oscillators like those in highly anharmonic potentials or perturbed harmonic oscillators. The analysis helps in understanding the behavior of such systems by breaking down complex motion into simpler, more manageable components.

:p How can Fourier series be used to analyze highly anharmonic oscillators?
??x
Fourier series can be used to analyze highly anharmonic oscillators by decomposing their periodic but non-sinusoidal motions into a sum of sinusoidal functions. This approach allows for the study and analysis of complex behaviors such as sawtooth-like motion, which would otherwise be difficult to understand using simple linear methods.

---

#### Fourier Series in Nonlinear Systems
Background context: In nonlinear systems, the "steady-state" behavior may jump among multiple configurations. Fourier series can help analyze this by providing a spectral representation that shows how much of each frequency is present in the system's response over time.

:p Why might one use Fourier series to analyze nonlinear systems?
??x
One uses Fourier series to analyze nonlinear systems because it helps identify and quantify the presence of various frequencies in the system’s response. This is particularly useful when the steady-state behavior jumps among multiple configurations, as Fourier analysis can reveal how much each frequency contributes to the overall motion.

---

#### Summary of Concepts
This summary consolidates the key points discussed about Fourier series, including their application to nonlinear oscillators and the process of decomposing periodic functions into simpler harmonic components. It emphasizes the importance of understanding both the theoretical underpinnings and practical applications of Fourier analysis in computational physics.

:p What are the main topics covered in this section?
??x
The main topics covered in this section include:
- Introduction to Fourier series and their application to nonlinear oscillators.
- The general form of a Fourier series for periodic functions.
- Calculation of Fourier coefficients using integration techniques.
- Fourier’s theorem and its applicability to single-valued periodic functions with finite discontinuities.
- Use of Fourier analysis in understanding the behavior of complex, non-sinusoidal motions.

---

#### Example Code for Calculating Fourier Coefficients
Background context: Implementing the calculation of Fourier coefficients involves integrating the function over one period. This can be done using numerical integration methods or analytical methods if possible.

:p Provide an example of calculating Fourier coefficients in code.
??x
Here is a simple example of how to calculate Fourier coefficients \( a_n \) and \( b_n \) for a given periodic function \( y(t) \):

```java
public class FourierCoefficients {
    public static double[] calculateFourierCoefficients(double[] y, int N, double T) {
        // N is the number of samples in one period
        // T is the period length

        double[] coefficients = new double[2 * N + 1]; // Array to store an and bn

        for (int n = 0; n <= N; n++) {
            // Calculate a_n
            double an = (2.0 / T) * sumOverOnePeriod(y, n, Math.PI * 2 * n / T);
            coefficients[n] = an;

            // Calculate b_n
            double bn = (2.0 / T) * sumOverOnePeriod(y, n, -Math.PI * 2 * n / T);
            coefficients[N + n + 1] = bn;
        }

        return coefficients;
    }

    private static double sumOverOnePeriod(double[] y, int n, double omega) {
        // Sum the function over one period
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += y[i] * Math.cos(omega * i);
        }
        return sum;
    }
}
```

This code calculates the Fourier coefficients by integrating the function over one period using a simple summation method.

x??

#### Sawtooth Function Definition
Background context: The sawtooth function is described mathematically as \( y(t) = \begin{cases} t/T, & \text{for } 0 \leq t \leq T/2 \\ t - T /T, & \text{for } T/2 \leq t \leq T \end{cases} \). This function is clearly periodic, nonharmonic, and discontinuous. However, it can be represented more simply by shifting the signal to the left.
:p What defines a sawtooth function?
??x
A sawtooth function is defined piecewise over its period \(T\):
- For \(0 \leq t \leq T/2\), \(y(t) = t / (T/2)\).
- For \(T/2 \leq t \leq T\), \(y(t) = t - T / (T/2)\).

This function is periodic with period \(T\) and has sharp corners, making it challenging to approximate using a few Fourier components.
x??

---
#### Odd Function Properties
Background context: For an odd function \( y(-t) = -y(t) \), all the cosine coefficients (\(a_n\)) are zero. Only half of the integration range is needed to determine the sine coefficients (\(b_n\)).
:p What happens with Fourier series for an odd function?
??x
For an odd function, the Fourier series will have only sine terms because \( a_n = 0 \). The formula to calculate the sine coefficients \( b_n \) simplifies due to symmetry:
\[ b_n = \frac{4}{T} \int_{0}^{T/2} y(t) \sin(n\omega t) dt. \]
x??

---
#### Even Function Properties
Background context: For an even function \( y(-t) = y(t) \), all the sine coefficients (\(b_n\)) are zero. Only half of the integration range is needed to determine the cosine coefficients (\(a_n\)).
:p What happens with Fourier series for an even function?
??x
For an even function, the Fourier series will have only cosine terms because \( b_n = 0 \). The formula to calculate the cosine coefficients \( a_n \) simplifies due to symmetry:
\[ a_n = \frac{4}{T} \int_{0}^{T/2} y(t) \cos(n\omega t) dt. \]
x??

---
#### Fourier Coefficients for Sawtooth Function
Background context: The sawtooth function can be represented as \( y(t) = t / (T/2) \), which is an odd function. Therefore, the Fourier series will only have sine terms.
:p How do you calculate the Fourier coefficients for a sawtooth function?
??x
For the sawtooth function \( y(t) = t / (T/2) \):
- The \(a_n\) coefficients are zero because it is an odd function.
- The \(b_n\) coefficients can be calculated as:
\[ b_n = \frac{4}{T} \int_{0}^{T/2} y(t) \sin(n\omega t) dt, \]
where \( \omega = 2\pi / T \).

The specific form of the sawtooth function makes these integrals computable.
x??

---
#### Average Value and a0
Background context: The average value of the periodic function over one period is given by \(a_0/2\) in its Fourier series representation. For the sawtooth function, \(a_0 = 2 \langle y(t) \rangle\).
:p What does \(a_0\) represent for a periodic function?
??x
The coefficient \(a_0\) represents the average value of the function over one period. Specifically, for the sawtooth function:
\[ a_0 = 2 \langle y(t) \rangle, \]
where \( \langle y(t) \rangle \) is the average value of \(y(t)\) over one period.
x??

---
#### Integration Range Simplification
Background context: For odd functions like the sawtooth, only half the integration range is needed to determine the Fourier coefficients because the function is symmetric about the origin. Similarly, for even functions, only half the range is required due to symmetry.
:p Why do we need to integrate over only half the period in some cases?
??x
We only need to integrate over half the period when dealing with odd or even functions due to their symmetry:
- For an odd function \(y(t) = -y(-t)\), the integral from \(-T/2\) to \(0\) is the negative of the integral from \(0\) to \(T/2\). Thus, we only need to integrate over half the period.
- For an even function \(y(t) = y(-t)\), the integrals from \(-T/2\) to \(0\) and from \(0\) to \(T/2\) are identical. Therefore, integrating over half the period is sufficient.

This simplification reduces computational complexity while maintaining the accuracy of the Fourier series.
x??

---

#### Sawtooth Function Fourier Series Summation

Background context: The text discusses summing the Fourier series for a sawtooth function up to different orders and observing its behavior over two periods. The sawtooth function has discontinuities, which can lead to overshooting as described by Gibbs phenomenon.

:p What is the Fourier series summation for the sawtooth function up to order N=20?

??x
The Fourier series summation for the sawtooth function up to order \(N = 20\) involves summing the coefficients \(a_n\) and \(b_n\). For a sawtooth function, these coefficients are:

\[a_n = -\frac{4}{n^2 \pi^2} \quad \text{for } n \neq 0, \text{ even or odd}\]
\[b_n = 0 \]

For \(N=20\), the summation involves terms from \(n=1\) to \(20\).

The series is:

\[ y(t) = \sum_{n=1}^{20} b_n \sin(n \omega t) + a_1 \sin(\omega t) - \sum_{n=3,5,...,19} \frac{4}{n^2 \pi^2} \cos((n-1) \omega t) \]

Since \(b_n = 0\) for all \(n\), the series simplifies to:

\[ y(t) = -\sum_{n=3,5,...,19} \frac{4}{(n-1)^2 \pi^2} \cos((n-1) \omega t) + a_1 \sin(\omega t) \]

Where \(a_1\) is the first coefficient.

For \(N = 20\):

\[ y(t) = -\frac{4}{9 \pi^2} \cos(2 \omega t) - \frac{4}{25 \pi^2} \cos(4 \omega t) - \cdots - \frac{4}{361 \pi^2} \cos(18 \omega t) + a_1 \sin(\omega t) \]

:p How does the series at \(N=20\) behave around discontinuities?

??x
At \(N=20\), the Fourier series of the sawtooth function overshoots by about 9% on either side of the discontinuity, a phenomenon known as Gibbs phenomenon. This occurs because the series tries to approximate the sharp corners of the discontinuous function with a finite number of terms.

:p How can you verify that the series gives the mean value at points of discontinuity?

??x
To verify that the series gives the mean value at points of discontinuity, we need to check if the average value of the series equals the midpoint of the jump in the sawtooth function. For a sawtooth function that goes from 0 to \(2\pi\) over one period:

- At \(t = T/4\), the function jumps from 0 to \(\pi\).
- The mean value should be \((0 + \pi) / 2 = \pi / 2\).

For the Fourier series, we sum up the series and evaluate it at the discontinuity points. The average value is:

\[ y(t) = -\sum_{n=3,5,...,19} \frac{4}{(n-1)^2 \pi^2} + a_1 \sin(\omega t) \]

At \(t = T/4\), since the sine terms average to zero over one period:

\[ y(T/4) = -\sum_{n=3,5,...,19} \frac{4}{(n-1)^2 \pi^2} \]

The series converges to a value that is close to the mean of the discontinuity.

:p How can you plot the results over two periods?

??x
To plot the results over two periods using Python and `FourierMatplot.py`:

```python
# Import necessary libraries
import numpy as np
from FourierMatplot import *  # Assuming FourierMatplot is a custom library

# Define the function to be plotted
def sawtooth(t):
    T = 1  # Period of the function
    return t % (2 * T) - T if 0 < t % T <= T else 0

# Sum the Fourier series up to N=20
N = 20
a_n = [-4 / (n**2 * np.pi**2) for n in range(1, N+1, 2)]  # Only odd terms
b_n = [0] * N  # No sine terms

# Generate time points
t = np.linspace(-T, 3*T, 1000)

# Compute the Fourier series approximation
y_approx = sum(a * np.sin(n * omega * t) for n, a in enumerate(a_n))

# Plot the original and approximate functions over two periods
plot_sawtooth_and_approx(t, sawtooth, y_approx)
```

:p What is the Gibbs phenomenon?

??x
The Gibbs phenomenon refers to the overshoot that occurs at points of discontinuity when approximating a discontinuous function using a finite number of terms from its Fourier series. Despite the approximation converging to the true value in an average sense, it will overshoot near the discontinuities by about 9% for large \(N\).

:p Half-Wave Function Fourier Series Summation

??x
The half-wave function is periodic but nonharmonic and continuous with discontinuous derivatives. The Fourier series summation up to order \(N=20\) can be computed using:

\[ y(t) = \frac{1}{2} \sin(\omega t) - \frac{2}{3\pi} \cos(2\omega t) - \frac{2}{15\pi} \cos(4\omega t) + \cdots \]

:p How does the Fourier series of the half-wave function converge?

??x
The Fourier series for the half-wave function converges well due to its continuous nature and absence of sharp corners. This allows it to be accurately represented by a finite number of terms, unlike the sawtooth function which exhibits overshooting.

---
---

