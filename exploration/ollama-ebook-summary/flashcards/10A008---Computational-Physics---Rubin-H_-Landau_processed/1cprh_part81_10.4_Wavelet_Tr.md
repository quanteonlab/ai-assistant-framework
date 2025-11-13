# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 81)

**Starting Chapter:** 10.4 Wavelet Transforms

---

#### Wave Packet and Uncertainty Principle
Wave packets are collections of waves with varying frequencies that form a pulse. The Fourier transform of such wave packets results in pulses in the frequency domain. The relationship between time width $\Delta t $ and frequency width$\Delta \omega$ is fundamental.
:p What is the relationship between the time width and frequency width of a wave packet?
??x
The time width $\Delta t $ and frequency width$\Delta \omega$ are related by the uncertainty principle:
$$\Delta t \Delta \omega \geq 2\pi.$$

This means that as the signal becomes more localized in time (smaller $\Delta t $), it becomes less localized in frequency (larger $\Delta \omega$).
x??

---

#### Simple Wave Packet Example
A simple example of a wave packet is given by a sine wave oscillating for $N$ periods:
$$y(t) = \begin{cases} 
\sin(\omega_0 t), & |t| < \frac{N\pi}{\omega_0}, \\
0, & |t| > \frac{N\pi}{\omega_0},
\end{cases}$$where $\omega_0 = \frac{2\pi}{T}$.
The time width of this wave packet is:
$$\Delta t = NT = \frac{N^2}{2\omega_0}.$$:p What is the Fourier transform of the simple wave packet example given?
??x
The Fourier transform of the wave packet can be calculated as follows:
$$

Y(\omega) = -i\sqrt{\frac{2\pi}{1}} \int_{-N\pi/\omega_0}^{0} \sin(\omega_0 t) e^{-i\omega t} dt + i\sqrt{\frac{2\pi}{1}} \int_{0}^{N\pi/\omega_0} \sin(\omega_0 t) e^{-i\omega t} dt.$$

This results in:
$$

Y(\omega) = \frac{(\omega_0+\omega)\sin[(\omega_0-\omega)N\pi/\omega_0] - (\omega_0-\omega)\sin[(\omega_0+\omega)N\pi/\omega_0]}{\sqrt{2\pi}(\omega^2_0-\omega^2)}.$$

The transform peaks at $\omega = \omega_0 $, with the frequency width $\Delta \omega \approx \frac{\omega_0}{N}$.
x??

---

#### Wave Packet Exercise
Consider three wave packets:
$$y_1(t) = e^{-t^2/2}, \quad y_2(t) = \sin(8t)e^{-t^2/2}, \quad y_3(t) = (1-t^2)e^{-t^2/2}.$$

For each, estimate the time width $\Delta t $ and frequency width$\Delta \omega$.
:p What is the general procedure for estimating the time width of a wave packet?
??x
To estimate the time width $\Delta t $, one can use the full width at half-maxima (FWHM) of $|y(t)|$. This involves finding the points where the absolute value of the signal drops to half its maximum value.
For frequency width $\Delta \omega $, similarly, find the FWHM of $|Y(\omega)|$.
x??

---

#### Short-Time Fourier Transforms
Short-time Fourier transforms (STFT) are used when traditional Fourier analysis cannot handle time-varying signals effectively. The signal is "chopped up" into segments over time, and a Fourier transform is applied to each segment.
:p What is the purpose of using short-time Fourier transforms?
??x
The purpose of using STFT is to analyze signals that change with time by breaking them down into smaller segments and applying Fourier analysis to each. This allows for better handling of non-stationary signals, which have varying frequency content over time.
x??

---

#### Concept of Overlap in Fourier Components
In the context of signal processing, the basis functions used in Fourier transforms are often constant in amplitude across all times, leading to significant overlap and correlated information among different components. This is problematic for data compression because it requires storing more redundant information.
:p Why is there a need to adjust the amount of information stored dependent on the desired quality of the reconstructed signal?
??x
There is a need to adjust the amount of information stored based on the desired quality of the reconstructed signal to optimize storage space and achieve lossless or lossy compression. By removing repeated elements and appropriate Fourier components, one can store less redundant information while maintaining acceptable reconstruction quality.
x??

---

#### Wavelet Analysis Introduction
Wavelets provide an effective approach to data compression, with standards like JPEG2000 based on wavelet analysis. They offer better localization in both time and frequency compared to traditional Fourier transforms.
:p How do wavelets differ from traditional Fourier transforms?
??x
Wavelets differ from traditional Fourier transforms by providing better time-frequency localization. While Fourier transforms are global and treat the entire signal, wavelets allow for localized analysis over different scales, making them more suitable for non-stationary signals with varying frequency content over time.
x??

---

#### Short-Time Fourier Transform
Background context explaining the short-time Fourier transform, including the mathematical formulation and its purpose. The formula for the short-time Fourier transform is given as:
$$Y(\text{ST})(\omega, \tau) = \int_{-\infty}^{+\infty} dt e^{i\omega t} \sqrt{\frac{2}{\pi}} w(t - \tau) y(t).$$

Here, the values of the translation time τ correspond to different locations of the window function over the signal.

:p What is the purpose of the short-time Fourier transform?
??x
The short-time Fourier transform aims to analyze signals in both time and frequency domains simultaneously. By translating a window function $w(t - \tau)$ across the signal, it captures local spectral information at different times. This approach allows for a more detailed analysis compared to traditional Fourier transforms, which only provide global frequency information.
x??

---

#### Wavelet Transform
Background context explaining wavelet transforms, their similarity to short-time Fourier transforms, and key differences such as using basis functions that are localized in time.

:p What is the key difference between wavelet transforms and short-time Fourier transforms?
??x
The key difference lies in the basis functions used. While short-time Fourier transforms use $e^{i\omega t}$ as their basis functions, wavelet transforms use wavelets or wave packets $\psi_{s,\tau}(t)$ that are localized both in time and frequency.
x??

---

#### Generating Wavelet Basis Functions
Explanation of how to generate a family of wavelet basis functions using a mother wavelet. The provided example uses the function:
$$\Psi(t) = \sin(8t)e^{-\frac{t^2}{2}}.$$

From this, daughter wavelets are generated as:
$$\psi_{s,\tau}(t) = \frac{1}{\sqrt{s}} \Psi\left(\frac{t - \tau}{s}\right) = \frac{1}{\sqrt{s}} \sin\left(8\frac{(t-\tau)}{s}\right)e^{-\frac{(t-\tau)^2}{2s^2}}.$$:p How do we generate daughter wavelets from a mother wavelet?
??x
We generate daughter wavelets by scaling and translating the mother wavelet $\Psi(t)$. For example, given the mother wavelet:
$$\Psi(t) = \sin(8t)e^{-\frac{t^2}{2}},$$daughter wavelets are created using the transformation:
$$\psi_{s,\tau}(t) = \frac{1}{\sqrt{s}} \Psi\left(\frac{t - \tau}{s}\right).$$

This involves scaling $\tau $ by a factor of$s $ and translating it by$\tau$.
x??

---

#### Wavelet Transform Equations
Explanation of the wavelet transform equations, including the forward and inverse transforms.

:p What are the forward and inverse wavelet transform equations?
??x
The forward wavelet transform is given by:
$$Y(s, \tau) = \frac{1}{\sqrt{s}} \int_{-\infty}^{+\infty} dt \Psi^* \left(\frac{t - \tau}{s}\right) y(t).$$

And the inverse wavelet transform is:
$$y(t) = \frac{1}{C} \int_{-\infty}^{+\infty} d\tau \int_0^\infty ds \psi^*_{s, \tau}(t) \frac{s^{3/2}}{Y(s, \tau)}.$$

Here $C$ is a normalization constant that depends on the wavelet used.
x??

---

#### Mother Wavelet Requirements
Explanation of the requirements for a mother wavelet and their significance.

:p What are the general requirements for a mother wavelet?
??x
The general requirements for a mother wavelet $\Psi(t)$ are:
1. Real-valued:$\Psi(t)$ must be real.
2. Zero-mean oscillation: The average value of $\Psi(t)$ over all time should be zero, i.e.,$\int_{-\infty}^{+\infty} \Psi(t) dt = 0$.
3. Localized in time: It must decay rapidly as $|t| \to \infty $, i.e., $\Psi(|t| \to \infty) \to 0$, and be square-integrable, meaning the integral of its squared magnitude over all time is finite.
4. Vanishing low powers of t moments: The first few moments should vanish, ensuring the transform is more sensitive to details than general shapes.
x??

---

#### Application in Signal Analysis
Explanation of how wavelet transforms are used in analyzing signals like chirps.

:p How do wavelets help analyze a chirp signal?
??x
Wavelets help by analyzing the signal at multiple scales. For a chirp signal $y(t) = \sin(60t^2)$, wavelets can capture both time and frequency details:
1. At low scales (small s values, higher frequencies), wavelets provide high-resolution analysis.
2. At high scales (large s values, lower frequencies), wavelets offer low-resolution analysis.

As the scale increases, fewer detailed features of the time signal remain visible, but the overall shape or gross features become clearer.
x??

---

#### Understanding Wavelet Transforms

Background context explaining wavelet transforms. This involves analyzing a signal by breaking it down into components at different scales and times, using a mother wavelet that is shifted and scaled to match features of the signal.

:p What are wavelet transforms used for?
??x
Wavelet transforms are used to analyze signals in both time and frequency domains simultaneously, providing a multi-resolution analysis. This allows for detailed examination of localized phenomena within the signal.
x??

---

#### Continuous Wavelet Transforms

Background context explaining continuous wavelet transforms (CWT). Unlike discrete wavelet transforms, CWT involves evaluating the similarity between the signal and the mother wavelet at every possible scale and time shift.

:p What is a key difference between continuous and discrete wavelet transforms?
??x
The key difference is that in CWT, the transform is performed over all possible scales and shifts of the mother wavelet, whereas in DWT (discrete wavelet transform), only specific scales are used.
x??

---

#### Implementing Wavelets

Background context on implementing different types of mother wavelets such as Morlet, Mexican Hat, and Haar.

:p How do you calculate a Morlet wavelet?
??x
To calculate a Morlet wavelet, use the following formula:
$$\psi(t) = \frac{1}{\sqrt{\pi f_b}} e^{j 2 \pi f_c t} e^{-t^2 / (f_b)}$$where $ f_c $ is the center frequency and $ f_b$ is the bandwidth.

```python
import numpy as np

def morlet_wavelet(t, f_c=5, f_b=1):
    return (1/np.sqrt(np.pi*f_b)) * np.exp(1j*2*np.pi*f_c*t) * np.exp(-t**2 / (f_b))
```
x??

---

#### Applying Wavelets to Signals

Background context on applying wavelet transforms to various signals, including sine waves and non-stationary signals.

:p How would you apply a wavelet transform to a pure sine wave?
??x
To apply a wavelet transform to a pure sine wave $y(t) = \sin(2\pi t)$, use the Morlet wavelet or another mother wavelet. The process involves shifting and scaling the wavelet across the signal's time domain and calculating the overlap at each point.

```python
def continuous_wavelet_transform(signal, wavelet):
    # Pseudocode for CWT
    result = []
    for t in range(len(signal)):
        overlap = np.sum(wavelet(t) * signal)
        result.append(overlap)
    return result
```
x??

---

#### Inverse Wavelet Transform

Background context on the inverse transform and its importance in reconstructing the original signal.

:p How do you invert a wavelet transform?
??x
To invert a wavelet transform, use the CWT formula:
$$y(t) = \int_{-\infty}^{\infty} X(\tau, s) \overline{\psi(s - t / s)} d\tau$$where $ X(\tau, s)$is the wavelet coefficients and $\psi$ is the mother wavelet.

```python
def inverse_wavelet_transform(wavelet_coeffs, wavelet):
    # Pseudocode for Inverse CWT
    signal = np.zeros_like(wavelet_coeffs)
    for t in range(len(signal)):
        recon_signal = 0
        for tau in range(len(wavelet_coeffs)):
            recon_signal += wavelet_coeffs[tau] * np.conj(wavelet(tau - t / s))
        signal[t] = recon_signal
    return signal
```
x??

---

#### Wavelet Spectrogram Analysis

Background context on analyzing signals using the continuous wavelet transform to observe frequency content over time.

:p What does a wavelet spectrum show?
??x
A wavelet spectrum shows how the frequency content of a signal changes with respect to time. It provides insights into non-stationary signals where the frequency composition may vary over different intervals.
x??

---

#### Schematic Representation

Background context on the schematic representation showing the process of analyzing a signal using a continuous wavelet transform.

:p Describe the steps in performing a wavelet transformation.
??x
The steps involve:
1. Analyzing the signal by evaluating its overlap with a narrow wavelet at the beginning.
2. Successively shifting the wavelet over the length of the signal and evaluating overlaps.
3. Expanding the wavelet after covering the entire signal and repeating the analysis.

```python
def perform_wavelet_transformation(signal, wavelet):
    # Pseudocode for Wavelet Transformation
    result = []
    for t in range(len(signal)):
        overlap = np.sum(wavelet(t) * signal)
        result.append(overlap)
    return result
```
x??

---

#### Discrete Wavelet Transform

Background context on the discrete wavelet transform, which makes optimal discrete choices for scale and time translation parameters.

:p How does a discrete wavelet transform differ from a continuous wavelet transform?
??x
A discrete wavelet transform (DWT) uses specific discrete scales and shifts rather than evaluating the mother wavelet over all possible scales and shifts as in CWT. This results in a more computationally efficient process.
x??

---

