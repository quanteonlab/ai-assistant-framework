# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 77)

**Starting Chapter:** Chapter 9 Fourier Analyses. 9.1 Fourier Series

---

#### Fourier Series Basics
Fourier series are used to expand periodic functions into a sum of sines and cosines. The function $y(t)$ can be expressed as:
$$y(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos(n\omega t) + b_n \sin(n\omega t) \right)$$where $\omega = \frac{2\pi}{T}$, and $ T$ is the period of the function.

:p What is the Fourier series representation for a periodic function?
??x
The Fourier series represents any periodic function as a sum of sines and cosines with frequencies that are integer multiples of the fundamental frequency. The coefficients $a_n $ and$b_n$ determine the contribution of each harmonic component.
```java
public class FourierSeries {
    public void computeCoefficients(double[] y, double T) {
        int N = y.length;
        double a0 = (2/T) * sum(y); // Compute a_0
        for (int n = 1; n < N; n++) {
            double an = (2/T) * sum(x -> Math.cos(n*2*Math.PI*x/T)*y[x]);
            double bn = (2/T) * sum(x -> Math.sin(n*2*Math.PI*x/T)*y[x]);
            // Store an and bn
        }
    }

    private double sum(double[] y) {
        return Arrays.stream(y).sum();
    }

    private double sum(MathFunction f, double[] y) {
        return Arrays.stream(y).mapToDouble(x -> f.apply(x)).sum();
    }
}
```
x??

---

#### Fourier Series Coefficient Calculation
The coefficients $a_n $ and$b_n$ are calculated by integrating the product of the function and the cosine or sine functions over one period. The formula is:
$$a_n = \frac{2}{T} \int_0^T y(t) \cos(n\omega t) dt, \quad b_n = \frac{2}{T} \int_0^T y(t) \sin(n\omega t) dt$$where $\omega = \frac{2\pi}{T}$.

:p How are the coefficients $a_n $ and$b_n$ calculated for a Fourier series?
??x
The coefficients $a_n $ and$b_n $ are computed by multiplying both sides of the Fourier series equation by$\cos(n\omega t)$ or $\sin(n\omega t)$, integrating over one period, and then solving for $ a_n$and $ b_n$.

```java
public class FourierCoefficients {
    public void calculateCoefficients(double[] y, double T) {
        int N = y.length;
        double a0 = (2/T) * sum(y);
        for (int n = 1; n < N; n++) {
            double an = (2/T) * integrate(x -> Math.cos(n*2*Math.PI*x/T)*y[x]);
            double bn = (2/T) * integrate(x -> Math.sin(n*2*Math.PI*x/T)*y[x]);
            // Store an and bn
        }
    }

    private double sum(double[] y) {
        return Arrays.stream(y).sum();
    }

    private double integrate(MathFunction f, double[] y) {
        return Arrays.stream(y).mapToDouble(x -> f.apply(x)).sum() * (2/T);
    }
}
```
x??

---

#### Short-Time Fourier Transform Overview
The Short-Time Fourier Transform (STFT) is used to analyze the spectral content of non-stationary signals, which means it provides a time-frequency representation. Unlike the full Fourier transform, STFT analyzes small segments of the signal at different times.

:p What does the Short-Time Fourier Transform provide?
??x
The Short-Time Fourier Transform provides a time-frequency analysis of non-stationary signals by analyzing small segments of the signal over different time windows. This allows for detailed examination of how the frequency content changes with time.
```java
public class STFT {
    public Complex[][] applySTFT(double[] signal, int windowSize) {
        int N = signal.length;
        Complex[][] stft = new Complex[N - windowSize][windowSize];
        
        for (int i = 0; i <= N - windowSize; i++) {
            double[] segment = Arrays.copyOfRange(signal, i, i + windowSize);
            // Apply Fourier Transform to the segment
            // Store result in stft[i]
        }
        
        return stft;
    }
}
```
x??

---

#### Quantum Fourier Transform
The Quantum Fourier Transform (QFT) is a quantum computing version of the Discrete Fourier Transform. It transforms a state vector $| \psi \rangle$ from the computational basis to the frequency domain.

:p What is the Quantum Fourier Transform?
??x
The Quantum Fourier Transform (QFT) is a quantum algorithm that transforms a quantum state in the computational basis to a superposition of states representing frequencies. This transformation is crucial for many quantum algorithms, such as Shor's algorithm for integer factorization.
```java
public class QuantumFourierTransform {
    public Complex[] applyQFT(Complex[] psi) {
        int N = psi.length;
        
        // Apply Hadamard gates and controlled rotations to transform the state
        for (int i = 0; i < N; i++) {
            psi = hadamard(psi, i);
            
            for (int j = 1; j <= i; j++) {
                psi = controlledRotation(psi, i, j);
            }
        }
        
        return psi;
    }

    private Complex[] hadamard(Complex[] psi, int qubitIndex) {
        // Apply Hadamard gate to the qubit
        return psi; // Pseudocode for simplicity
    }

    private Complex[] controlledRotation(Complex[] psi, int controlQubit, int targetQubit) {
        // Apply controlled rotation based on the state of the control qubit
        return psi; // Pseudocode for simplicity
    }
}
```
x??

---

#### Nonlinear Oscillations and Fourier Series
In nonlinear oscillators, such as those described by $V(x) = \frac{1}{2} k x^2 (1 - 2/3 \alpha x)$, the motion is periodic but not necessarily harmonic. The solution can be expanded in a Fourier series to analyze its spectral content.

:p How are nonlinear oscillations analyzed using Fourier series?
??x
Nonlinear oscillations are analyzed by expanding their solutions into a Fourier series, which decomposes the motion into sinusoidal components with varying amplitudes and phases. This helps in understanding the periodic nature of the motion even when it is not purely harmonic.
```java
public class NonlinearOscillator {
    public double[] expandInFourierSeries(double[] x) {
        int N = x.length;
        double[] fourierCoefficients = new double[N];
        
        for (int n = 0; n < N; n++) {
            // Calculate coefficients a_n and b_n
            fourierCoefficients[n] = calculateCoefficient(x, n);
        }
        
        return fourierCoefficients;
    }

    private double calculateCoefficient(double[] x, int n) {
        int N = x.length;
        double integral = 0;
        
        for (int i = 0; i < N; i++) {
            integral += Math.cos(2 * Math.PI * i / N) * x[i];
        }
        
        return (2/N) * integral;
    }
}
```
x??

---

#### Fourier Series Basics
Background context explaining the concept of Fourier series, including its relevance to periodic functions and their decomposition into sinusoidal components. The Fourier spectrum shows how frequencies are distributed within a function, with coefficients $b_n$ decreasing as frequency increases.

:p What is the Fourier series used for in analyzing periodic functions?
??x
The Fourier series decomposes periodic functions into a sum of sine and cosine waves, allowing us to analyze and understand their frequency content. This is useful in many fields such as signal processing, physics, and engineering.
x??

---
#### Symmetry in Fourier Series
Background on how symmetry can simplify the calculation of Fourier coefficients. For an odd function, all $a_n $ coefficients are zero, and only half the integration range is needed to determine$b_n $. For an even function, all $ b_n $coefficients are zero, and similarly, only half the range is used for$ a_n$.

:p How can symmetry in a periodic signal simplify Fourier series calculations?
??x
Symmetry allows us to focus on specific types of coefficients. If a function is odd (i.e., $y(-t) = -y(t)$), all $ a_n$coefficients are zero, and we only need to compute the $ b_n$coefficients over half the period. Similarly, if the function is even ($ y(-t) = y(t)$), all $ b_n$coefficients are zero, and we only integrate over half the period for $ a_n$.

For example, if a signal is odd:
```java
bn = (4/T) * ∫[0 to T/2] y(t) * sin(nωt) dt
```
x??

---
#### Sawtooth Function Analysis
Description of the sawtooth function and its mathematical representation. It is periodic, nonharmonic, and discontinuous but can be simplified by shifting it.

:p How is a sawtooth function mathematically represented?
??x
A sawtooth function $y(t)$ is defined as:
$$y(t) = \begin{cases} 
t/T & \text{for } 0 \leq t \leq T/2 \\
t - T/T & \text{for } T/2 \leq t \leq T
\end{cases}$$

This function can be simplified by shifting it to:
$$y(t) = \frac{t}{T/2} \text{ for } -T/2 \leq t \leq T/2.$$

The general shape of the sawtooth function is periodic and nonharmonic, but its symmetry allows for simpler Fourier series representation. However, many components are needed to accurately represent sharp corners.

x??

---
#### Odd Function Properties
Explanation of how an odd function simplifies Fourier series calculations by setting all $a_n$ coefficients to zero and reducing the integration range.

:p What happens when a function is odd in terms of its Fourier series?
??x
For an odd function, defined as $y(-t) = -y(t)$, the Fourier series contains only sine terms ($ b_n $) because cosine terms (even functions) are orthogonal to the odd function over any symmetric interval. This means all$ a_n$ coefficients are zero.

The integration range can be halved, simplifying the calculation:
$$b_n = \frac{4}{T} \int_{0}^{T/2} y(t) \sin(n\omega t) \, dt.$$

This property is useful for reducing computational complexity in signal processing and analysis.

x??

---
#### Even Function Properties
Explanation of how an even function simplifies Fourier series calculations by setting all $b_n$ coefficients to zero and reducing the integration range.

:p What happens when a function is even in terms of its Fourier series?
??x
For an even function, defined as $y(-t) = y(t)$, the Fourier series contains only cosine terms ($ a_n $) because sine terms (odd functions) are orthogonal to the even function over any symmetric interval. This means all$ b_n$ coefficients are zero.

The integration range can be halved, simplifying the calculation:
$$a_n = \frac{4}{T} \int_{0}^{T/2} y(t) \cos(n\omega t) \, dt.$$

This property is useful for reducing computational complexity in signal processing and analysis.

x??

---
#### Sawtooth Function Simplification
Explanation of the sawtooth function's simplification through symmetry and its Fourier series representation.

:p How does the sawtooth function simplify when considering symmetry?
??x
The sawtooth function can be simplified by leveraging its odd symmetry. By shifting the function, we get:
$$y(t) = \frac{t}{T/2} \text{ for } -T/2 \leq t \leq T/2.$$

This simplification is useful because it reduces the complexity of the Fourier series calculation. Although many components are needed to accurately represent sharp corners, the symmetry allows us to focus on either half of the period.

x??

---
#### Fourier Series Components for Sharp Corners
Explanation of why many components are required in the Fourier series representation of functions with sharp corners like sawtooth waves.

:p Why do we need many Fourier series components for a function with sharp corners?
??x
Functions with sharp corners, such as the sawtooth wave, cannot be accurately represented by just a few sine and cosine terms. This is because the Fourier series expansion aims to approximate the function in the least squares sense over one period.

To capture the discontinuities and sharp transitions, many more high-frequency components are necessary. These components contribute to the detailed structure of the function, allowing for a better approximation near the corners.

x??

#### Fourier Series for Sawtooth Function

Background context explaining the concept. The sawtooth function is a periodic, nonharmonic, and continuous function with discontinuous derivatives. Its Fourier series can be used to approximate it.

Formula:
$$y(t) = 2 \frac{1}{\pi} \left( -\frac{\sin(\omega t)}{1} + \frac{\sin(3 \omega t)}{3} - \frac{\sin(5 \omega t)}{5} + \cdots \right)$$:p What is the Fourier series representation of a sawtooth function?
??x
The Fourier series for a sawtooth function can be represented as an infinite sum of sine terms:
$$y(t) = 2 \frac{1}{\pi} \sum_{n=1,3,5,\ldots}^{\infty} \left( -\frac{\sin((2n-1)\omega t)}{2n-1} \right).$$

This series includes only odd harmonics.

The code to sum the Fourier series up to a given order $N$ is as follows:
```python
import numpy as np

def sawtooth_fourier_series(t, omega, N):
    y = 0.0
    for n in range(1, N+1):
        if n % 2 == 1:  # Only odd harmonics
            term = (2 / (n * np.pi)) * (-np.sin(n * omega * t))
            y += term
    return y

# Example usage:
t_values = np.linspace(-T/2, T/2, 400)
omega_value = 1.0  # Frequency of the signal
N_value = 20       # Order up to which series is summed
y_values = [sawtooth_fourier_series(t, omega_value, N_value) for t in t_values]
```
x??

---

#### Gibbs Phenomenon

Background context explaining the concept. The Gibbs phenomenon refers to the overshoot and undershoot at a discontinuity when approximating functions using Fourier series.

:p What is the Gibbs phenomenon?
??x
The Gibbs phenomenon describes the overshoot that occurs at points of discontinuity in a function's approximation by its Fourier series. Specifically, as more terms are added to the series, the overshoot at these points remains and does not diminish but converges to a constant fraction (approximately 9%) of the jump.

The code to visualize this is similar to summing the series:
```python
import matplotlib.pyplot as plt

def plot_fourier_sawtooth(t_values, omega_value, N_values):
    y_actual = np.where(t_values > 0, np.sin(omega_value * t_values), 0)
    
    for N in N_values:
        y_approx = sawtooth_fourier_series(t_values, omega_value, N)
        
        plt.figure()
        plt.plot(t_values, y_approx, label=f'N={N}')
        plt.plot(t_values, y_actual, label='Actual')
        plt.xlabel('t')
        plt.ylabel('y(t)')
        plt.legend()

# Example usage:
T = 2 * np.pi
omega_value = 1.0
N_values = [2, 4, 10, 20]
plot_fourier_sawtooth(np.linspace(-T/2, T/2, 400), omega_value, N_values)
plt.show()
```
x??

---

#### Fourier Series for Half-Wave Function

Background context explaining the concept. The half-wave function is a periodic, nonharmonic, and continuous function with discontinuous derivatives that lacks sharp corners compared to a sawtooth function.

Formula:
$$y(t) = \begin{cases} 
\sin(\omega t), & 0 < t < T/2 \\
0, & T/2 < t < T 
\end{cases}$$

The Fourier series for the half-wave function is:
$$y(t) = \frac{1}{2} \sin(\omega t) + \sum_{n=1}^{\infty} \left( -\frac{2}{\pi (2n-1)^2} \cos((2n-1)\omega t) \right).$$:p What is the Fourier series representation of a half-wave function?
??x
The Fourier series for the half-wave function includes both sine and cosine terms:
$$y(t) = \frac{1}{2} \sin(\omega t) - \frac{2}{\pi (3)} \cos(2\omega t) - \frac{2}{\pi (5)} \cos(4\omega t) + \cdots.$$

The code to sum the series up to a given order $N$ is:
```python
def half_wave_fourier_series(t, omega, N):
    y = 0.0
    y += 1 / 2 * np.sin(omega * t)
    
    for n in range(3, N + 1, 2):  # Only odd harmonics except the first one
        term = - (2 / (n * np.pi)) * np.cos(n * omega * t)
        y += term
    
    return y

# Example usage:
t_values = np.linspace(-T/2, T/2, 400)
omega_value = 1.0
N_value = 20
y_values = [half_wave_fourier_series(t, omega_value, N_value) for t in t_values]
```
x??

---

#### Fourier Transform

Background context explaining the concept. The Fourier transform is used to analyze non-periodic functions by representing them as a sum of sinusoidal functions.

Formula:
$$y(t) = \int_{-\infty}^{\infty} d\omega Y(\omega) e^{i\omega t} / \sqrt{2\pi}.$$
$$

Y(\omega) = \int_{-\infty}^{\infty} dt e^{-i\omega t} y(t) / \sqrt{2\pi}.$$:p What are the formulas for Fourier transform and its inverse?
??x
The formulas for the Fourier transform and its inverse are:
$$y(t) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} d\omega Y(\omega) e^{i\omega t},$$
$$

Y(\omega) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} dt y(t) e^{-i\omega t}.$$

The code to compute the Fourier transform and inverse is as follows:
```python
import numpy as np

def fourier_transform(y, omega):
    return (1 / np.sqrt(2 * np.pi)) * np.trapz(y * np.exp(-1j * omega), dx=0.01)

def inverse_fourier_transform(Y, t):
    return (1 / np.sqrt(2 * np.pi)) * np.trapz(Y * np.exp(1j * 2 * np.pi * t), dx=0.01)
```
x??

---

#### Dirac Delta Function

Background context explaining the concept. The Dirac delta function is a generalized function that can be used to represent impulses and has applications in physics.

Formula:
$$\int_{-\infty}^{\infty} d\omega e^{i(\omega - \omega_0)t} = 2\pi \delta(\omega - \omega_0).$$:p What is the Dirac delta function?
??x
The Dirac delta function $\delta(\omega - \omega_0)$ is a generalized function defined such that:
$$\int_{-\infty}^{\infty} d\omega e^{i(\omega - \omega_0)t} = 2\pi \delta(\omega - \omega_0).$$

The Dirac delta function is not well-behaved in the sense of conventional functions, but it is extremely useful in theoretical physics. It can be thought of as a function that has an infinite value at $\omega_0$ and zero elsewhere, with the area under its curve being 1.

:p How does the Dirac delta function relate to the Fourier transform?
??x
The Dirac delta function appears in the context of the Fourier transform when converting between the time domain and frequency domain. Specifically, it helps establish the identity:
$$Y(\omega) = \int_{-\infty}^{\infty} d\omega' \delta(\omega - \omega') Y(\omega').$$

This relationship is crucial for ensuring consistency in the Fourier transform pair.

:p How can we use the Dirac delta function to prove the equivalence of Fourier series and transforms?
??x
To prove the equivalence, we start by substituting the inverse Fourier transform into the forward Fourier transform:
$$

Y(\omega) = \int_{-\infty}^{\infty} dt e^{-i\omega t} y(t),$$
$$y(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega' Y(\omega') e^{i\omega' t}.$$

Substituting $Y(\omega)$ into the inverse Fourier transform:
$$y(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega' \left( \int_{-\infty}^{\infty} dt e^{-i\omega t} y(t) \right) e^{i\omega' t}.$$

By changing the order of integration and using the identity:
$$\int_{-\infty}^{\infty} d\omega e^{i(\omega - \omega')t} = 2\pi \delta(\omega - \omega'),$$we get:
$$y(t) = \int_{-\infty}^{\infty} dt' y(t') \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega e^{i\omega (t-t')} = y(t).$$

This shows that the Fourier transform and its inverse are consistent with each other.

:p How can we create a semilog plot of the squared modulus of the Fourier transform?
??x
To create a semilog plot of the squared modulus of the Fourier transform $|Y(\omega)|^2$, follow these steps:
```python
import matplotlib.pyplot as plt

def plot_power_spectrum(Y, omega_values):
    power = np.abs(Y)**2
    
    plt.figure()
    plt.semilogy(omega_values, power)
    plt.xlabel(r'$\omega$')
    plt.ylabel('Power Spectrum (|Y($\omega$)|^2)')
    plt.title('Power Spectrum of the Signal')

# Example usage:
omega_values = np.linspace(-10 * np.pi, 10 * np.pi, 400)
Y_values = fourier_transform(y(t), omega_values) / len(omega_values)  # Normalized
plot_power_spectrum(Y_values, omega_values)
plt.show()
```
x??

---

