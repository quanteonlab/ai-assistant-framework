# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 21)

**Starting Chapter:** 9.4.4 Digital Filters Windowed Sinc Filters

---

#### Delay-Line Filter Concept
Background context: In digital signal processing, delay-line filters are used to create specific frequency responses through time delays. Each tap on a delay line provides a delayed version of the input signal, which is then scaled and summed up to form the total response function.

Formula:
$$h(t) = \sum_{n=0}^{N} c_n \delta(t - n\tau)$$where $ c_n $ are scaling factors, and $\delta(t)$ represents an impulse at time $t$.

Frequency domain transfer function:
$$H(\omega) = \sum_{n=0}^{N} c_n e^{-in\omega\tau}$$:p What is the delay-line filter used for in digital signal processing?
??x
The delay-line filter is used to create a specific frequency response by using time delays and scaling factors. Each tap on the delay line represents an impulse that is delayed by a certain amount of time, which can be combined to produce a desired output.
x??

---

#### Digital Filter: Windowed Sinc Filters
Background context: Windowed sinc filters are used to separate different bands of frequencies in a signal. They are popular because they can effectively remove high-frequency noise while preserving low-frequency signals.

Formula for the sinc function:
$$\int_{-\infty}^{\infty} d\omega e^{-i\omega t} \text{rect}(\omega) = \frac{\sin(\pi t / 2)}{\pi t / 2}$$

Filter response in time domain:
$$h[i] = \frac{\sin(2\pi \omega_c (i - M/2))}{\pi (i - M/2)}, \quad 0 \leq t \leq M$$where $ M $ is the number of points, and $\omega_c$ is the cutoff frequency.

:p What is the sinc filter used for in digital signal processing?
??x
The sinc filter is used to separate different bands of frequencies in a signal by filtering out high-frequency components. It helps reduce noise by removing high-frequency signals while preserving low-frequency signals.
x??

---

#### Rectangular Function and Its Fourier Transform
Background context: The rectangular function $\text{rect}(\omega)$ is constant within a finite frequency interval, representing an ideal low-pass filter that passes all frequencies below a cutoff frequency $\omega_c$ and blocks higher frequencies.

Formula for the rect function:
$$\text{rect}\left( \frac{\omega}{2\omega_c} \right) = \begin{cases} 1 & \text{if } |\omega| \leq 1/2 \\ 0 & \text{otherwise} \end{cases}$$

Fourier transform of the sinc function:
$$\int_{-\infty}^{\infty} d\omega e^{-i\omega t} \text{rect}(\omega) = \frac{\sin(\pi t / 2)}{\pi t / 2}$$

:p What is the Fourier transform of a rectangular pulse?
??x
The Fourier transform of a rectangular pulse in the frequency domain results in a sinc function in the time domain. Specifically, the Fourier transform of rect($\omega$) is given by:
$$\int_{-\infty}^{\infty} d\omega e^{-i\omega t} \text{rect}(\omega) = \frac{\sin(\pi t / 2)}{\pi t / 2}$$x??

---

#### Gibb's Overshoot and Windowing
Background context: In practice, using a sinc function as a filter results in Gibbs overshoot, where the response has rounded corners and oscillations beyond the cutoff frequency. To mitigate this, window functions such as Hamming windows are applied to smooth out the truncation.

Hamming window formula:
$$w[i] = 0.54 - 0.46 \cos\left( \frac{2\pi i}{M} \right)$$

Combined filter kernel with Hamming window:
$$h[i] = \frac{\sin[2\pi \omega_c (i - M/2)]}{\pi (i - M/2)} \left[ 0.54 - 0.46 \cos\left( \frac{2\pi i}{M} \right) \right]$$:p What is Gibbs overshoot in the context of digital filters?
??x
Gibbs overshoot refers to the phenomenon where a sinc function filter, when used directly as a filter kernel, results in oscillations and ripples around the cutoff frequency. This is due to the abrupt truncation of the sinc function.
x??

---

#### Sinc Filter for Noise Reduction
Background context: The sinc filter is a type of low-pass filter used to reduce noise while preserving the signal. It works by allowing signals below a certain cutoff frequency $\omega_c$ to pass through, and attenuating higher frequencies.

The sinc function has the form:
$$h[n] = \frac{\sin(\pi n / M)}{\pi n / M}$$where $ M $ is half of the filter's time length. The cutoff frequency $\omega_c $ should be a fraction of the sampling rate, and the timelength$M$ determines the bandwidth over which the filter changes from 1 to 0.

:p How does the sinc filter reduce noise in signals?
??x
The sinc filter reduces noise by allowing low-frequency components (close to DC or the fundamental frequency) to pass through while attenuating high-frequency noise. This is achieved by designing a window function that has a sharp transition at the cutoff frequency.
x??

---

#### Fast Fourier Transform (FFT)
Background context: The FFT algorithm is an efficient method for computing the Discrete Fourier Transform (DFT). It reduces the number of operations from $N^2 $ to approximately$N \log_2 N$, significantly speeding up computations.

The DFT in compact form:
$$Y_n = \frac{1}{\sqrt{2\pi N}} \sum_{k=1}^{N} Z_n k y_k, \quad Z=e^{-2\pi i/N}, \quad n=0,1,\dots,N-1$$where $ Z $ is complex, and both $ n $ and $ k $ range from 0 to $ N-1$.

:p What is the time complexity of a direct DFT computation?
??x
The time complexity of a direct DFT computation is $O(N^2)$.
x??

---

#### Butterfly Operation in FFT
Background context: The butterfly operation is a key component in implementing the FFT algorithm. It takes pairs of complex numbers and combines them to produce new values, utilizing symmetries in the data.

:p What is the purpose of the butterfly operation in the FFT?
??x
The purpose of the butterfly operation is to efficiently compute the DFT by leveraging symmetry and reducing the number of multiplications required.
x??

---

#### Simplified DFT Computations with Symmetry
Background context: For a specific case, such as $N = 8 $, we can simplify the computations using the symmetry in the powers of $ Z$.

Example for $N = 8$:
$$Y_0 = Z_0 y_0 + Z_1 y_1 + Z_2 y_2 + Z_3 y_3 + Z_4 y_4 + Z_5 y_5 + Z_6 y_6 + Z_7 y_7$$
$$

Y_1 = Z_0 y_0 + Z_1 y_1 + Z_2 y_2 + Z_3 y_3 - Z_4 y_4 - Z_5 y_5 - Z_6 y_6 - Z_7 y_7$$:p How do we simplify the DFT computations using symmetry?
??x
By exploiting the periodicity and symmetry of $Z $, we can reduce the number of multiplications. For instance, for $ N = 8 $, only four unique powers of$ Z $ are used: $ Z_0, Z_1, Z_2, Z_3$. This allows us to rewrite the DFT as a series of sums and differences.
x??

---

#### Butterfly Operation Visualization
Background context: The butterfly operation regroups terms into sums and differences, reducing the number of complex multiplications.

:p What is the butterfly operation in the FFT?
??x
The butterfly operation restructures the computations by combining pairs of $y $ values into new values using symmetry. It reduces the number of required multiplications to approximately$N \log_2 N$.

For example, for $N = 8$:
$$Y_0 = Z_0 (y_0 + y_4) + Z_0 (y_1 + y_5) + Z_0 (y_2 + y_6) + Z_0 (y_3 + y_7)$$x??

---

#### Butterfly Operation in FFT
Background context: The butterfly operation is a fundamental step in the Fast Fourier Transform (FFT) algorithm. It reduces the number of required multiplications and additions by reusing intermediate results.

:p Describe the basic structure of the butterfly operation in an FFT.
??x
The butterfly operation involves taking two input elements, typically denoted as $y_p $ and$y_q $, and transforming them into two new output values: $(y_p + Z y_q)$ and $(y_p - Z y_q)$. The complex number $ Z$ is a twiddle factor which rotates the phase of the second input by an angle corresponding to its position in the FFT.

This operation can be visualized as:

```plaintext
yp yp
Z yp + Zyq yp – Zyq
```

:p Explain the butterfly operation for two consecutive inputs.
??x
For two consecutive inputs $y_p $ and$y_q$, the butterfly operation computes:
1. $y_p + Z y_q $2.$ y_p - Z y_q $Where$ Z = e^{-j2\pi k/N}$ is a complex number that represents a phase shift.

:p Provide an example of how a single butterfly operation works.
??x
Consider two input elements $y_0 $ and$y_1 $. After applying the butterfly operation with $ Z = e^{-j2\pi \cdot 1/8} = Z_4$:

```plaintext
y0 y1
Z0 y0 + Z0 y1 y0 – Z0 y1
```

The outputs are:
- $Y_0 = (y_0 + y_1)$-$ Y_1 = (y_0 - Z_0 y_1)$:p What is the significance of bit reversal in FFT?
??x
Bit-reversal refers to rearranging the input data based on their binary representation. For example, if we have 8 inputs labeled from 0 to 7, after bit-reversing, the order changes to 0, 4, 2, 6, 1, 5, 3, 7.

Bit reversal is crucial because it ensures that the data are correctly ordered for subsequent butterfly operations. This reordering helps in efficiently computing the FFT by reducing the number of required multiplications and additions.

:p How does the bit-reversal affect the processing order in an FFT?
??x
The bit-reversal process affects the processing order such that the first half of the elements (0, 2, 4, 6) are placed before the second half (1, 3, 5, 7). This reordering ensures that the outputs from the butterfly operations are computed in a specific sequence.

:p Show an example of bit-reversal for 8 input data.
??x
For 8 input data elements numbered as 0 through 7:

- Binary representation: 0 (000), 1 (001), 2 (010), 3 (011), 4 (100), 5 (101), 6 (110), 7 (111)
- Bit-reversal: 0 (000), 4 (100), 2 (010), 6 (110), 1 (001), 5 (101), 3 (011), 7 (111)

:p How does the FFT algorithm reduce the number of multiplications compared to a direct DFT?
??x
The FFT reduces the number of required multiplications by exploiting the symmetry and periodicity properties of complex exponentials. For an 8-point FFT, it requires only 24 multiplications compared to 64 in the original DFT formula.

:p Illustrate how the total number of multiplications is reduced from 64 to 24 for 8 points.
??x
In a straightforward DFT, we would need $N^2 = 8^2 = 64$ complex multiplications. However, an FFT algorithm reduces this by reusing intermediate results:

- For the first butterfly: 8 multiplications by $Z_0$- Second butterfly: 8 multiplications
- And so on...

Total: 24 multiplications.

:p Explain how to implement a modified FFT.
??x
A modified FFT transforms the eight input data into eight transforms, but arranges the output in numerical order. This can be achieved by first performing bit-reversal and then applying butterfly operations:

```plaintext
y7 y3 y5 y1 y6 y2 y4 y0
```

After processing:
- $Y_0 = (y_0 + y_4) + (y_2 + y_6)$-$ Y_1 = (y_0 – y_4) + Z_2(y_2 – y_6) + Z_1(y_1 – y_5) + Z_3(y_3 – y_7)$

:p How does the output order differ between a standard FFT and the modified FFT?
??x
In a standard FFT, the outputs are in bit-reversed order (0, 4, 2, 6, 1, 5, 3, 7). In contrast, in the modified FFT, the outputs are in numerical order (0, 1, 2, 3, 4, 5, 6, 7).

:p Summarize the key differences between a standard and modified FFT.
??x
- **Standard FFT**: Outputs are bit-reversed for efficient computation. This requires an initial bit-reversal step but reduces complexity in subsequent steps.
- **Modified FFT**: Outputs are directly in numerical order, which simplifies post-processing but may require additional ordering steps.

---
---

