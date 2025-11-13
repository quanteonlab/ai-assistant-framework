# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 24)

**Starting Chapter:** 10.5 Discrete Wavelet Transforms

---

#### Discrete Wavelet Transforms (DWT)
Background context: DWT is a method used to analyze time signals that are measured at discrete times. Unlike continuous wavelet transforms, DWT deals with discrete values of scaling and translation parameters, making it suitable for practical applications where data is often sampled at discrete intervals.
Relevant formulas:
$$\Psi\left[\frac{t - k2^j}{2^j}\right] = \psi_{j,k}(t) \sqrt{\frac{1}{2^j}}$$where $ s = 2^j, \tau = \frac{k}{2^j}$, and $ j, k$ are integers.
The DWT is defined as:
$$Y_{j,k} \approx \sum_m \psi_{j,k}(t_m)y(t_m)$$:p What does the discrete wavelet transform (DWT) evaluate?
??x
The DWT evaluates transforms using discrete values for scaling and translation parameters. It is used when time signals are measured at discrete times, allowing for a more practical approach to signal analysis.
x??

---

#### Orthonormality of Basis Functions in DWT
Background context: The orthonormal basis functions in the DWT ensure that each wavelet has unit energy and is independent of other basis functions. This leads to flexible data storage by ensuring low correlation between different transform components.
Relevant formulas:
$$\int_{-\infty}^{\infty} dt\, \psi_{j,k}^*(t) \psi_{j',k'}(t) = \delta_{jj'} \delta_{kk'}$$where $\delta_{m,n}$ is the Kronecker delta function.

:p What does the orthonormality of basis functions in DWT imply?
??x
The orthonormality implies that each wavelet has unit energy and is independent of other basis functions, leading to low correlation between different transform components. This results in efficient data storage.
x??

---

#### Sampling Strategy in DWT
Background context: When applying the DWT, it's important to sample the input signal at discrete times determined by integers $j $ and$k$. The number of steps required to cover each major feature should be sufficient for desired precision. A rule of thumb is to start with 100 steps per interval.
Relevant formulas: None.

:p How do we determine the sampling strategy in DWT?
??x
We determine the sampling strategy by sampling the input signal at discrete times determined by integers $j $ and$k$. The number of steps should be sufficient to cover each major feature, with a rule of thumb being 100 steps per interval.
x??

---

#### Uncertainty Principle in DWT
Background context: Just as in Fourier transforms, the uncertainty principle places constraints on the time intervals and frequency intervals. Specifically, the product of the widths of the wave packet and its Fourier transform must be at least $2\pi$.
Relevant formulas:
$$\Delta\omega \cdot \Delta t \geq 2\pi$$:p How does the uncertainty principle apply to DWT?
??x
The uncertainty principle in DWT places constraints on the time intervals and frequency intervals, ensuring that the product of their widths is at least $2\pi$. This means higher-resolution frequency components require more sampling points in time.
x??

---

#### Multiresolution Analysis (MRA) for DWT
Background context: MRA is a technique used to compute DWT without explicit integration. It uses a pyramid algorithm that samples the signal at finite times and passes it through a series of filters, each representing a digital version of a wavelet. The process does not compute explicit integrals but instead relies on convolutions with filter response functions.
Relevant code example:
```python
def DWT_pyramid_algorithm(signal, j):
    # Initialize pyramid levels
    for level in range(j):
        # Apply low-pass and high-pass filters (subband coding)
        low_pass_filter = apply_low_pass_filter()
        high_pass_filter = apply_high_pass_filter()

        signal = convolve(signal, low_pass_filter)  # Low pass
        subbands = decompose_signal_into_subbands(low_pass_filter_output, high_pass_filter)

    return subbands
```

:p What is the multiresolution analysis (MRA) technique in DWT?
??x
The MRA technique in DWT uses a pyramid algorithm to sample the signal at finite times and passes it through filters representing digital wavelets. The process avoids explicit integration by using convolutions with filter response functions.
x??

---

#### Discrete Wavelet Transforms (DWT)
Discrete Wavelet Transforms decompose a signal into smooth information stored in low-frequency components and detailed information stored in high-frequency components. This is done using a series of filters that change the scale and resolution of the input signal.

:p What are the key steps involved in performing a Discrete Wavelet Transform (DWT)?
??x
The DWT process involves several key steps:
1. Applying filter matrices to the input data.
2. Decimating or downsampling the output by half.
3. Repeating this process until only two coefficients remain for each high and low frequency component.

Code example in pseudocode:

```pseudocode
function performDWT(signal, filters)
    smooth = applyFilter(signal, lowPassFilter)
    detail = applyFilter(signal, highPassFilter)
    decimatedSmooth = downsample(smooth)
    decimatedDetail = downsample(detail)

    if length(decimatedSmooth) > 2:
        return performDWT(decimatedSmooth, filters), performDWT(decimatedDetail, filters)
    else:
        return smooth, detail
```
x??

---

#### Low-Pass and High-Pass Filters in DWT
In the context of DWT, low-pass and high-pass filters are used to decompose a signal into its frequency components. Low-pass filters retain the lower frequencies (smooth information) while high-pass filters capture the higher frequencies (detailed information).

:p How do low-pass and high-pass filters contribute to the Discrete Wavelet Transform?
??x
Low-pass and high-pass filters in DWT are essential for decomposing a signal into its constituent frequency components. Low-pass filters allow only lower-frequency components of the input signal to pass through, effectively smoothing out the data, while high-pass filters allow higher-frequency components (details) to pass through.

The filter matrices are applied successively to downsampled versions of the input vector:

```pseudocode
function applyFilter(vector, filterMatrix)
    transformed = filterMatrix * vector
    return transformed
```

Example application:

```pseudocode
lowPassFiltered = applyFilter(signal, lowPassFilterMatrix)
highPassFiltered = applyFilter(signal, highPassFilterMatrix)
```
x??

---

#### Pyramid Scheme in DWT
The pyramid scheme is a method for decomposing an input signal into multiple levels of detail and approximation using filter banks. It involves repeated filtering and downsampling to produce a multi-resolution analysis.

:p What are the key steps in implementing the pyramid algorithm for Discrete Wavelet Transform?
??x
Implementing the pyramid algorithm for DWT involves several key steps:
1. Apply the filter matrix (low-pass or high-pass) to the entire input signal.
2. Downsample the output by half.
3. Repeat the process until only two coefficients remain.

Pseudocode example:

```pseudocode
function pyramidDWT(signal, lowPassFilterMatrix, highPassFilterMatrix)
    smooth = applyFilter(signal, lowPassFilterMatrix)
    detail = applyFilter(signal, highPassFilterMatrix)

    decimatedSmooth = downsample(smooth)
    decimatedDetail = downsample(detail)

    if length(decimatedSmooth) > 2:
        lowerSmooth, lowerDetail = pyramidDWT(decimatedSmooth, lowPassFilterMatrix, highPassFilterMatrix)
        return lowerSmooth, lowerDetail, detail
    else:
        return smooth, detail
```
x??

---

#### Decimation and Subsampling in DWT
Decimation or subsampling involves filtering the output by a factor of 2. This reduces the number of values needed to represent the remaining signal while maintaining its key features.

:p What is the role of decimation in Discrete Wavelet Transform?
??x
The role of decimation (or subsampling) in DWT is to reduce the number of data points required to represent the decomposed signal by a factor of 2. This process helps in achieving a multi-resolution representation where lower levels have fewer samples but still capture important features.

Example:

```pseudocode
function downsample(vector)
    return vector[::2]  # Every second element from the original vector
```
x??

---

#### Processing Chirp Signal with DWT
A practical example of applying DWT to a chirp signal involves filtering and downsampling. The chirp function y(t) = sin(60t^2) is sampled, and then passed through multiple levels of filters to decompose it into different frequency components.

:p How does the processing of a chirp signal with Daub4 wavelets work in DWT?
??x
Processing a chirp signal with Daub4 wavelets involves filtering the signal through a series of low-pass and high-pass filters at different resolutions. The signal is first filtered by a single low-pass and high-pass filter, then downsampled by half. This process is repeated recursively until only two coefficients remain for each level.

Example:

```pseudocode
def processChirpSignal(chirpSignal):
    levels = 5  # Number of processing levels

    for _ in range(levels):
        smooth, detail = applyFilter(chirpSignal, lowPassFilterMatrix)
        decimatedSmooth, decimatedDetail = downsample(smooth), downsample(detail)

        if len(decimatedSmooth) > 2:
            chirpSignal = decimatedSmooth
        else:
            break

    return smooth, detail

# Filter matrices and downsampling functions are predefined
```
x??

--- 

These flashcards cover the key concepts in DWT with detailed explanations and examples. Each card focuses on a specific aspect of the process to aid understanding.

#### Discrete Wavelet Transforms (DWT)
Background context: The discrete wavelet transform (DWT) decomposes a signal into different frequency subbands, where each subband is further processed. This process involves dilating and analyzing the signal at multiple stages to capture both high-frequency details and low-frequency smooth components.
:p What are the main steps involved in the Discrete Wavelet Transform?
??x
The DWT process includes multiple stages of filtering and downsampling:
1. **Initial Analysis**: The input signal is filtered using a low-pass filter (L) and a high-pass filter (H).
2. **Downsampling**: The resulting coefficients are downsampled by 2, retaining only half the number of coefficients.
3. **Repeat Stages**: This process is repeated on the lower frequency part obtained from the previous stage until two coefficients per filter remain.

The inverse DWT reconstructs the original signal using an upward process where the filtered coefficients are upsampled and reprocessed with both low-pass and high-pass filters to recover all N values of the original signal.
x??

---
#### Low-Pass Filter (L) and High-Pass Filter (H)
Background context: The low-pass filter $L $ and high-pass filter$H$ are represented by a set of coefficients. These filters are used to decompose an input signal into smooth and detail components, respectively.

The filters are defined as follows:
$$L = [c_0 + c_1, c_2 + c_3]$$
$$

H = [c_3 - c_2, c_1 - c_0]$$

Where $c_0, c_1, c_2,$ and $c_3$ are the filter coefficients.

:p How do the low-pass and high-pass filters act on an input vector?
??x
The low-pass filter $L$ acts as a smoothing operation that outputs a weighted average of the input signal elements. For example:
$$L = [c_0, c_1, c_2, c_3]$$
$$

L \times \begin{bmatrix} y_0 \\ y_1 \\ y_2 \\ y_3 \end{bmatrix} = c_0y_0 + c_1y_1 + c_2y_2 + c_3y_3$$

The high-pass filter $H$ acts as a detail extraction operation that outputs weighted differences of the input signal elements. For example:
$$H = [c_3 - c_2, c_1 - c_0]$$
$$

H \times \begin{bmatrix} y_0 \\ y_1 \\ y_2 \\ y_3 \end{bmatrix} = (c_3y_0 - c_2y_1) + (c_1y_2 - c_0y_3)$$

The result of these operations is a set of coefficients representing the smooth and detail parts of the input signal.
x??

---
#### Orthogonality Condition for Wavelet Filters
Background context: For the wavelet transform to be orthogonal, the filter matrix must satisfy an orthogonality condition. This ensures that the transformation can reversibly reconstruct the original signal.

The orthogonality condition is expressed as:
$$\begin{bmatrix} c_0 & c_1 & c_2 & c_3 \\ c_3 - c_2 & c_1 - c_0 & 0 & 0 \\ c_2 & c_3 & c_0 & c_1 \\ c_1 - c_0 & 0 & c_3 - c_2 & 0 \end{bmatrix} \times \begin{bmatrix} c_0 & 3 + \sqrt{3}/(4\sqrt{2}) & 3 - \sqrt{3}/(4\sqrt{2}) & 1 - \sqrt{3}/(4\sqrt{2}) \\ 1 - \sqrt{3}/(4\sqrt{2}) & c_0 & c_1 & c_2 \\ 3 - \sqrt{3}/(4\sqrt{2}) & 1 - \sqrt{3}/(4\sqrt{2}) & c_0 & c_1 \\ \sqrt{3}/(4\sqrt{2}) & 3 + \sqrt{3}/(4\sqrt{2}) & -c_0 & -c_1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$:p What is the orthogonality condition for wavelet filters?
??x
The orthogonality condition ensures that the wavelet transform matrix is invertible, allowing perfect reconstruction of the original signal. This is achieved by satisfying the following equation:
$$\begin{bmatrix} c_0 & c_1 & c_2 & c_3 \\ c_3 - c_2 & c_1 - c_0 & 0 & 0 \\ c_2 & c_3 & c_0 & c_1 \\ c_1 - c_0 & 0 & c_3 - c_2 & 0 \end{bmatrix} \times \begin{bmatrix} c_0 & 3 + \sqrt{3}/(4\sqrt{2}) & 3 - \sqrt{3}/(4\sqrt{2}) & 1 - \sqrt{3}/(4\sqrt{2}) \\ 1 - \sqrt{3}/(4\sqrt{2}) & c_0 & c_1 & c_2 \\ 3 - \sqrt{3}/(4\sqrt{2}) & 1 - \sqrt{3}/(4\sqrt{2}) & c_0 & c_1 \\ \sqrt{3}/(4\sqrt{2}) & 3 + \sqrt{3}/(4\sqrt{2}) & -c_0 & -c_1 \end{bmatrix} = I$$

Where $I$ is the identity matrix. This condition ensures that the filter matrix and its inverse are well-defined, enabling accurate reconstruction of the original signal.
x??

---
#### Construction of Multi-Scale Wavelets
Background context: The multi-scale wavelets are constructed by placing the row versions of $L $ and$H$ along the diagonal, with successive pairs displaced two columns to the right. This results in a larger filter matrix that can process multiple elements at once.

For example, for 8 elements:
$$\begin{bmatrix} c_0 & c_1 & c_2 & c_3 & 0 & 0 & 0 & 0 \\ c_3 - c_2 & c_1 - c_0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & c_0 & c_1 & c_2 & c_3 & 0 & 0 \\ 0 & 0 & c_3 - c_2 & c_1 - c_0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & c_0 & c_1 & c_2 & c_3 \\ 0 & 0 & 0 & 0 & c_3 - c_2 & c_1 - c_0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & c_0 & c_1 \\ 0 & 0 & 0 & 0 & 0 & 0 & c_3 - c_2 & c_1 - c_0 \end{bmatrix} \times \begin{bmatrix} y_0 \\ y_1 \\ y_2 \\ y_3 \\ y_4 \\ y_5 \\ y_6 \\ y_7 \end{bmatrix}$$:p How are multi-scale wavelets constructed for a larger number of elements?
??x
Multi-scale wavelets are constructed by placing the row versions of $L $ and$H$ along the diagonal, with successive pairs displaced two columns to the right. For example, for 8 elements:
$$\begin{bmatrix} c_0 & c_1 & c_2 & c_3 & 0 & 0 & 0 & 0 \\ c_3 - c_2 & c_1 - c_0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & c_0 & c_1 & c_2 & c_3 & 0 & 0 \\ 0 & 0 & c_3 - c_2 & c_1 - c_0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & c_0 & c_1 & c_2 & c_3 \\ 0 & 0 & 0 & 0 & c_3 - c_2 & c_1 - c_0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & c_0 & c_1 \\ 0 & 0 & 0 & 0 & 0 & 0 & c_3 - c_2 & c_1 - c_0 \end{bmatrix} \times \begin{bmatrix} y_0 \\ y_1 \\ y_2 \\ y_3 \\ y_4 \\ y_5 \\ y_6 \\ y_7 \end{bmatrix}$$

This construction ensures that the wavelet analysis can handle multiple elements simultaneously, providing a more comprehensive decomposition of the signal.
x??

---
#### Inverse Wavelet Transform
Background context: The inverse wavelet transform reconstructs the original signal from its transformed coefficients. This involves upsampling and reprocessing with both low-pass and high-pass filters to recover all N values of the original signal.

:p How does the inverse wavelet transform work?
??x
The inverse wavelet transform works by upsampled and reprocessing the filtered coefficients using both low-pass and high-pass filters. For example, given a set of transformed coefficients:
$$\begin{bmatrix} s_0 \\ d_1 \\ s_2 \\ d_3 \\ s_4 \\ d_5 \\ s_6 \\ d_7 \end{bmatrix}$$

The process involves upsampled and reprocessing each pair with the appropriate filter to reconstruct the original signal. The details (d) are combined with the smoothed parts (s) at different stages to recover all N values of the original signal.
x??

---
#### Daubechies Wavelets
Background context: The Daubechies wavelets, specifically Daub4, are constructed by applying the inverse transform to a vector where only one element is set to 1 and all others to 0. This process results in a set of wavelet functions that can be used for detailed analysis.

:p How are Daubechies wavelets constructed?
??x
Daubechies wavelets, specifically Daub4, are constructed by applying the inverse transform to a vector where only one element is set to 1 and all others to 0. This process results in a set of wavelet functions that can be used for detailed analysis.

For example:
- To obtain $y_{6}(t)$, apply the inverse transform to a vector with a 1 in the 7th position and zeros elsewhere.
- The sum of Daub4 e10 and Daub4 1e58 wavelets, each corresponding to different scale and time displacements, can be visualized as:
$$y_{10}(t) + y_{58}(t)$$

These wavelet functions capture both the smooth and detailed features of the input signal.
x??

---
#### Time Dependencies of Daubechies Wavelets
Background context: The time dependencies of Daubechies wavelets, specifically Daub4, are visualized in Figure 10.12. These wavelets have different scales and time positions.

:p What do the time dependencies of Daubechies wavelets represent?
??x
The time dependencies of Daubechies wavelets represent their temporal spread and scale variations. For example:
- The left side of Figure 10.12 shows the e6 wavelet, which has been found to be particularly effective in wavelet analyses.
- The right side of Figure 10.12 shows the sum of Daub4 e10 and Daub4 1e58 wavelets, each corresponding to different scale and time displacements.

These visualizations help understand how the wavelets capture both high-frequency details and low-frequency smooth components at various temporal scales.
x??

