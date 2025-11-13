# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 22)

**Starting Chapter:** 9.7 FFT Assessment

---

#### Bit Reversal and FFT Input Reordering
Background context: The Fourier transforms are produced in an order corresponding to the bit-reversed order of numbers. This suggests that processing data in a bit-reversed order (e.g., 0, 4, 2, 6 for 8 points) will result in correctly ordered output. The number 3 appears here because it is the power of 2 giving the number of data; specifically, $2^3 = 8$. For an FFT algorithm to produce transforms in the proper order, input data must be reshuffled into bit-reversed order.

:p How does bit reversal help in ordering the output for FFTs?
??x
Bit reversal helps by ensuring that the output from the FFT is ordered correctly according to the bit-reversed indices of the input. This means that if you process your data points in a specific order derived from their binary representation (e.g., 0, 4, 2, 6 for an 8-point DFT), the resulting Fourier coefficients will be in the correct sequence. 

For example, consider the bit-reversed order of indices for 16 points:
- Binary: 000 (0), 010 (2), 100 (4), 110 (6), 001 (1), 011 (3), 101 (5), 111 (7)
- Bit-reversed: 000, 100, 010, 110, 001, 101, 011, 111

This reordering ensures that the output is in the proper frequency order. 

In code, this can be implemented by reversing the binary digits of each index.
```java
public void bitReverse(int n, int[] arr) {
    for (int i = 0; i < n / 2; ++i) {
        if (arr[i] != -1 && arr[n - 1 - i] == -1) { // Ensure no double assignment
            int temp = arr[i];
            arr[i] = arr[n - 1 - i];
            arr[n - 1 - i] = temp;
        }
    }
}
```
x??

---

#### Butterfly Operations in FFTs
Background context: The butterfly operation is a key component of the Fast Fourier Transform (FFT) algorithm. It involves combining pairs of complex numbers to produce new values that are used in subsequent stages of the transform. For an N-point FFT, where $N = 2^k $, the number of butterfly operations required is $\log_2(N)$.

:p What is a butterfly operation and how many times does it need to be applied for an 8-point FFT?
??x
A butterfly operation combines pairs of complex numbers using a specific formula, typically involving multiplications by roots of unity (twiddle factors). For an 8-point FFT ($N = 2^3 $), the number of butterfly operations needed is $\log_2(8) = 3$.

In each stage of the FFT, the input data are processed in pairs to produce new values that form the output. The process involves multiple stages where the number of operations per stage decreases logarithmically.

Here's a simplified pseudocode for a single butterfly operation:
```java
for (int i = 0; i < N / 2; ++i) {
    int evenIdx = i * 2;
    int oddIdx = evenIdx + 1;

    // Store the original values before transformation.
    double reEven = data[evenIdx].real;
    double imEven = data[evenIdx].imag;
    double reOdd = data[oddIdx].real;
    double imOdd = data[oddIdx].imag;

    // Calculate the twiddle factors and apply them to odd index components.
    double phaseShiftReal = -2 * M_PI * i / N;  // Phase shift
    double phaseShiftImag = 0;                  // For simplicity, assuming imaginary part is zero

    double reOddTwisted = reOdd * cos(phaseShiftReal) + imOdd * sin(phaseShiftReal);
    double imOddTwisted = -reOdd * sin(phaseShiftReal) + imOdd * cos(phaseShiftReal);

    // Update the odd index components with the transformed values.
    data[oddIdx].real = reEven - reOddTwisted;
    data[oddIdx].imag = imEven - imOddTwisted;

    data[evenIdx].real = reEven + reOddTwisted;
    data[evenIdx].imag = imEven + imOddTwisted;
}
```
x??

---

#### Handling Data Points that are Not Powers of 2
Background context: In practical applications, the number of input data points might not always be a power of 2. To make it so, you can concatenate some of the initial data to the end of your input until a power of 2 is obtained. Since a Discrete Fourier Transform (DFT) is periodic, this just starts the period slightly earlier.

:p How do you handle cases where the number of input data points is not a power of 2?
??x
When the number of input data points is not a power of 2, you can pad your input with additional values to make it so. This padding ensures that the FFT algorithm can be applied correctly and efficiently. Since the DFT is periodic, this does not change the fundamental properties of the signal being transformed; it merely starts the period earlier or later.

For example, if you have 10 data points, you would pad them with 6 additional zeros (to make a total of 16 points) before applying the FFT. Here's how you might do this in code:
```java
public void padData(double[] input) {
    int paddingSize = 2;
    while ((input.length & (paddingSize - 1)) != 0) {
        paddingSize *= 2;
    }

    double[] paddedInput = new double[paddingSize];
    System.arraycopy(input, 0, paddedInput, 0, input.length);
    
    // Fill the rest with zeros
    for (int i = input.length; i < paddingSize; ++i) {
        paddedInput[i] = 0;
    }

    return paddedInput;
}
```
x??

---

#### Python Implementation of FFT
Background context: The provided text mentions a Python implementation of an FFT algorithm, which is easier to follow than the original Fortran IV version. This implementation processes $N = 2^n$ data points using complex numbers and bit-reversal ordering.

:p How does the Python FFT implementation in Listing 9.3 work?
??x
The Python FFT implementation processes $N = 2^n$ data points by first assigning complex numbers to the data points, then reordering them via bit reversal, and finally performing butterfly operations. Here’s a high-level overview of how it works:

1. **Complex Number Assignment**: Each of the $N$ data points is assigned a complex number.
2. **Bit Reversal Ordering**: The input data are reordered according to their bit-reversed indices.
3. **Butterfly Operations**: Butterfly operations are performed in stages until all outputs are computed.

Here's a simplified version of how you might implement this in Python:
```python
import numpy as np

def bit_reversal_order(N, x):
    n = N.bit_length() - 1
    order = [0] * N
    for i in range(N):
        reversed_i = int('{:0{n}b}'.format(i, n=n)[::-1], 2)
        order[i] = reversed_i
    return np.array(order)

def fft(x):
    N = len(x)
    n = N.bit_length()
    
    # Assign complex numbers to data points
    ym = [complex(m, m) for m in range(N)]
    
    # Bit reversal ordering
    order = bit_reversal_order(2**n, np.arange(N))
    x = [ym[i] for i in order]
    
    # Perform butterfly operations (simplified)
    stages = int(np.log2(N))
    for s in range(stages):
        length = 2 ** s
        for k in range(N // (2 * length)):
            for j in range(length):
                w = np.exp(-2j * np.pi * j / N)
                u = x[2 * k * length + j]
                v = x[2 * k * length + j + length] * w
                x[2 * k * length + j] = u + v
                x[2 * k * length + j + length] = u - v
    
    return np.array(x)

# Example usage:
data = [0, 1, 2, 3, 4, 5, 6, 7]
result = fft(data)
print(result)
```
x??

---

#### Discrete Fourier Transform (DFT) Using Complex Numbers

Background context: The DFT is a fundamental tool used to analyze signals by transforming them from the time domain to the frequency domain. When using complex numbers, each input signal can be represented as a combination of sine and cosine functions. The formula for DFT is given by:
$$X[k] = \sum_{n=0}^{N-1} x[n]e^{-j2\pi kn/N}$$where $ x[n]$ is the time-domain signal,$ X[k]$is the frequency-domain representation, and $ k$ranges from 0 to $ N-1$.

The provided code demonstrates how to compute the DFT using complex numbers. The `DFT` function iterates over each point in the input signal and calculates the corresponding frequency domain values by summing the product of the time-domain values with complex exponentials.

:p What is the purpose of the `f` method in the given code?

??x
The `f` method generates a time-domain signal based on predefined mathematical expressions. In this case, it creates a signal that consists of three sinusoidal components: a cosine term and two sine terms with different frequencies.

```python
def Signal(y):
    h = twopi / N
    x = 0.
    for i in range(0, N+1):
        y[i] = 30 * cos(x) + 60 * sin(2 * x) + 120 * sin(3 * x)
        SignalCurve.plot(pos=(x, y[i]))
        x += h
```
x??

---

#### Discrete Fourier Transform (DFT) Using Real Numbers

Background context: Similar to the previous DFT implementation using complex numbers, this code computes the DFT for a real-valued signal. However, it only uses real arithmetic by leveraging the properties of sine and cosine functions.

The formula for computing the imaginary part of the DFT is:

$$X[k]_{\text{imag}} = - \sum_{n=0}^{N-1} x[n]\sin(2\pi kn/N)$$

This method avoids complex arithmetic by separating real and imaginary parts, making it more efficient for certain hardware implementations.

:p What is the purpose of the `fourier` method in this code?

??x
The `fourier` method computes the DFT's imaginary part for a given signal using only real arithmetic. It iterates over each frequency bin and calculates the sum of products between the time-domain signal values and sine functions, which are scaled by the appropriate frequency factors.

```python
def fourier(dftimag):
    for n in range(0, Np):
        imag = 0.
        for k in range(0, N):
            imag += signal[k] * sin((twopi * k * n) / N)
        dftimag[n] = -imag * sq2pi
```
x??

---

#### Fast Fourier Transform (FFT)

Background context: The FFT is an efficient algorithm to compute the DFT of a sequence. It reduces the complexity from $O(N^2)$ to $O(N \log N)$ by exploiting the symmetry and periodicity properties of complex exponentials.

The provided `fft` function implements the Cooley-Tukey algorithm, which recursively splits the DFT into smaller DFTs. The function reorders input data in a bit-reversed manner and performs butterfly operations to combine results from lower frequency bins.

:p What is the primary purpose of the FFT algorithm?

??x
The primary purpose of the FFT algorithm is to efficiently compute the Discrete Fourier Transform (DFT) of a sequence by reducing computational complexity. It achieves this by leveraging the symmetry and periodicity properties of complex exponentials, allowing for recursive decomposition of larger DFTs into smaller ones.

```python
def fft(N, Switch):
    y = zeros(2 * (N + 4), float)
    Y = zeros((N + 3, 2), float)

    # Bit-reversal permutation
    for i in range(1, n + 1, 2):
        j = i - m
        if j < 0:
            break
        y[j], y[j + 1] = y[i], y[i + 1]
    
    # FFT butterfly operations
    istep = 2 * mmax
    theta = 6.2831853 / (Switch * mmax)
    sinth = math.sin(theta / 2.0)
    wstpr = -2.0 * sinth ** 2
    wstpi = math.sin(theta)
    wr = 1.0
    wi = 0.0

    for min range(1, mmax + 1, 2):
        for i in range(m, n + 1, istep):
            j = i + mmax
            tempr = wr * y[j] - wi * y[j + 1]
            tempi = wr * y[j + 1] + wi * y[j]
            y[j], y[j + 1] = y[i] - tempr, y[i + 1] - tempi
            y[i], y[i + 1] = y[i] + tempr, y[i + 1] + tempi

            wr = wr * wstpr - wi * wstpi + wr
            wi = wi * wstpr + tempr * wstpi + wi
        mmax //= 2
```
x??

---

#### Wavelet Analysis Introduction
Wavelet analysis is a technique used to analyze signals that change over time. Unlike Fourier analysis, which provides frequency information but lacks temporal resolution, wavelets offer both frequency and time localization, making them suitable for non-stationary signals like those with varying frequencies over time.
:p What does wavelet analysis provide in terms of signal analysis?
??x
Wavelet analysis provides both frequency and time localization. Unlike Fourier analysis, which gives a frequency spectrum but lacks temporal resolution, wavelets can pinpoint when specific frequencies occur within the signal.
x??

---

#### Signal Example for Wavelet Analysis
The example signal given is:
$$y(t) = \begin{cases} 
\sin(2\pi t), & \text{for } 0 \leq t \leq 2, \\
5\sin(2\pi t) + 10\sin(4\pi t), & \text{for } 2 \leq t \leq 8, \\
2.5\sin(2\pi t) + 6\sin(4\pi t) + 10\sin(6\pi t), & \text{for } 8 \leq t \leq 12.
\end{cases}$$:p What is the signal used to demonstrate wavelet analysis?
??x
The signal used to demonstrate wavelet analysis is a piecewise function that changes its frequency content over time:
$$y(t) = \begin{cases} 
\sin(2\pi t), & \text{for } 0 \leq t \leq 2, \\
5\sin(2\pi t) + 10\sin(4\pi t), & \text{for } 2 \leq t \leq 8, \\
2.5\sin(2\pi t) + 6\sin(4\pi t) + 10\sin(6\pi t), & \text{for } 8 \leq t \leq 12.
\end{cases}$$x??

---

#### Wavelet Functions
Wavelets are functions that are localized in both time and frequency. They can be mathematically defined as:
$$\psi(t) = e^{i\omega_0t - \frac{t^2}{2\sigma^2}} = (\cos(\omega_0 t) + i\sin(\omega_0 t))e^{-\frac{t^2}{2\sigma^2}},$$where $\psi(t)$ is the Morlet wavelet with $\omega_0$ being a frequency parameter and $\sigma$ controlling its width.
:p What is an example of a wavelet function provided in the text?
??x
An example of a wavelet function provided in the text is the Morlet wavelet, which has the following definition:
$$\psi(t) = e^{i\omega_0t - \frac{t^2}{2\sigma^2}} = (\cos(\omega_0 t) + i\sin(\omega_0 t))e^{-\frac{t^2}{2\sigma^2}},$$where $\omega_0 $ is the frequency parameter and$\sigma$ controls its width.
x??

---

#### Wavelet Generation
Wavelets can be generated by scaling and translating a mother wavelet. For instance, a Morlet wavelet:
$$\psi(t) = e^{i\omega_0t - \frac{t^2}{2\sigma^2}},$$can generate daughter wavelets through different values of $ a $(scaling factor) and$ b$(translation parameter).
:p How are mother wavelets transformed into daughter wavelets?
??x
Mother wavelets can be transformed into daughter wavelets by scaling and translating them. For example, the Morlet wavelet:
$$\psi(t) = e^{i\omega_0t - \frac{t^2}{2\sigma^2}},$$can generate different daughter wavelets using different values of $ a $(scaling factor) and$ b$ (translation parameter).
x??

---

#### Time Localization
Wavelets are localized in time, meaning they exist for only short periods. This property allows them to capture both the frequency content and when it occurs within the signal.
:p Why is time localization important in wavelet analysis?
??x
Time localization is important in wavelet analysis because it allows capturing both the frequency content of a signal and when specific frequencies occur. Unlike Fourier analysis, which provides only frequency information but lacks temporal resolution, wavelets can pinpoint the time instances at which certain frequencies appear.
x??

---

#### Example Wavelets
Four possible mother wavelets are:
- Morlet (real part)
- Mexican hat
- Daub4 e6
- Haar

These wavelets are generated by scaling and translating their respective mother functions. The daughter wavelets provide a set of basis functions that can be used for signal analysis.
:p What are some examples of mother wavelets mentioned?
??x
Some examples of mother wavelets mentioned are:
- Morlet (real part)
- Mexican hat
- Daub4 e6
- Haar

These mother wavelets, when scaled and translated, generate a set of daughter wavelets that serve as basis functions for signal analysis.
x??

---

