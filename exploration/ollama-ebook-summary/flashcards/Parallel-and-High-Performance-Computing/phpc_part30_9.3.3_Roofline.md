# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 30)

**Starting Chapter:** 9.3.3 Roofline performance model for GPUs

---

---
#### Babel STREAM Benchmark for NVIDIA GPUs
Background context: The Babel STREAM Benchmark is a tool used to measure memory bandwidth on different GPU architectures and programming languages. This specific example uses CUDA to test an NVIDIA V100 GPU.

:p What are the steps to run the Babel STREAM Benchmark for an NVIDIA GPU using CUDA?
??x
The steps include cloning the BabelStream repository, configuring it with a CUDA makefile, and running the benchmark:

```bash
git clone git@github.com:UoB-HPC/BabelStream.git
make -f CUDA.make ./cuda-stream
```

This will yield results for various memory bandwidth operations on the GPU. Each operation (Copy, Mul, Add, Triad, Dot) measures its performance in MBytes/sec and provides minimum, maximum, and average values.

The output example includes:

```plaintext
Function    MBytes/sec  Min (sec)   Max         Average
Copy        800995.012  0.00067     0.00067     0.00067 
Mul         796501.837  0.00067     0.00068     0.00068
Add         838993.641  0.00096     0.00097     0.00096 
Triad       840731.427  0.00096     0.00097     0.00096
Dot         866071.690  0.00062     0.00063     0.00063
```

x??

---
#### Babel STREAM Benchmark for AMD GPUs (OpenCL)
Background context: The Babel STREAM Benchmark can also be used to test an AMD GPU, specifically the AMD Vega 20. This involves configuring and running OpenCL-based benchmarks.

:p How do you set up and run the Babel STREAM Benchmark on an AMD GPU using OpenCL?
??x
The process for setting up and running the benchmark on an AMD GPU includes editing the `OpenCL.make` file to include paths to the necessary header files and libraries, then compiling and executing:

```bash
make -f OpenCL.make ./ocl-stream
```

For the AMD Vega 20, the output shows slightly lower memory bandwidth compared to the NVIDIA V100:

```plaintext
Function    MBytes/sec  Min (sec)   Max         Average 
Copy        764889.965  0.00070     0.00077     0.00072  
Mul         764182.281  0.00070     0.00076     0.00072 
Add         764059.386  0.00105     0.00134     0.00109 
Triad       763349.620  0.00105     0.00110     0.00108
Dot         670205.644  0.00080     0.00088     0.00083 
```

x??

---
#### Roofline Performance Model for GPUs (Babel STREAM)
Background context: The Roofline model is used to analyze and predict the performance limits of a system based on memory bandwidth and flop performance. This concept applies similarly to both CPUs and GPUs, helping understand their operational efficiency.

:p How does the roofline performance model apply to measuring GPU performance with Babel STREAM?
??x
The Roofline model helps in understanding the performance limits by plotting the FLOPs/Byte (arithmetic intensity) against GFLOP/s. For GPUs, it involves testing various operations and visualizing their performance on a graph.

For instance, when running the Babel STREAM benchmark using the Empirical Roofline Toolkit, you can generate roofline plots for different GPU architectures like NVIDIA V100 and AMD Vega 20:

```bash
git clone https://bitbucket.org/berkeleylab/cs-roofline-toolkit.git
cd cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0
cp Config/config.voltar.uoregon.edu Config/config.V100_gpu
# Edit the configuration file for V100 GPU details.
./ert Config/config.V100_gpu

# Repeat similar steps for AMD Vega 20.
```

This process generates detailed plots that illustrate the theoretical and actual performance limits, showing where operations are memory bound or compute bound.

Example of roofline plot output:

```plaintext
Figure 9.5 Roofline plots for NVIDIA V100 and AMD Vega 20:
NVIDIA V100: 
   - DRAM bandwidth = 793.1 GB/s
   - L1 cache bandwidth = 2846.3 GB/s

AMD Vega 20:
   - DRAM bandwidth = 744.0 GB/s
   - L1 cache bandwidth = 2082.7 GB/s
```

x??

---

#### Mixbench Performance Tool Overview
Background context: The mixbench tool is a performance model that helps identify the best GPU for a specific workload by plotting compute rate versus memory bandwidth. This visual representation aids in selecting the most suitable GPU based on application requirements.

:p What is the primary purpose of using the mixbench tool?
??x
The primary purpose of using the mixbench tool is to determine the best GPU for a given workload by comparing its peak performance characteristics (compute rate and memory bandwidth) against the arithmetic intensity and memory bandwidth limits of the application. This helps in selecting the most efficient GPU based on the specific needs of the application.
x??

---
#### Setting Up Mixbench
Background context: The mixbench tool can be set up to run benchmarks for different GPU devices, requiring CUDA or OpenCL installations. This setup involves cloning the repository, modifying the Makefile, and configuring paths.

:p How do you install and configure the mixbench tool?
??x
To install and configure the mixbench tool, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/ekondis/mixbench.git
```

2. Navigate to the directory and edit the Makefile:
```bash
cd mixbench;
edit Makefile
```

3. Fix the path to the CUDA or OpenCL installations.

4. Set the executables to build, which can be done by running the following command (override the path with your installation):
```bash
make CUDA_INSTALL_PATH=<path>
```

5. Run either of these commands:
```bash
./mixbench-cuda-ro
./mixbench-ocl-ro
```
x??

---
#### Understanding Roofline Plot
Background context: The mixbench tool plots the compute rate in GFlops/sec with respect to memory bandwidth in GB/sec, identifying the peak flop and bandwidth capabilities of GPU devices. This plot helps understand if an application is memory-bound or compute-bound.

:p What does a roofline plot show in terms of GPU performance?
??x
A roofline plot shows the relationship between the compute rate (in GFlops/sec) and memory bandwidth (in GB/sec) for different GPU devices. The plot highlights the peak flop and bandwidth capabilities, indicating whether an application is more limited by memory or computation.

If a GPU device point is above the application line, it means the application is memory-bound. If below the line, the application is compute-bound.
x??

---
#### Identifying Application Boundaries
Background context: The plot helps identify if applications are memory-bound or compute-bound based on their arithmetic intensity and memory bandwidth requirements.

:p How can you determine if an application is memory-bound or compute-bound?
??x
You can determine if an application is memory-bound or compute-bound by comparing its arithmetic intensity (flops/load) to the GPU's performance characteristics plotted on a roofline plot. If the application's line intersects above the GPU device point, it is memory-bound; below indicates compute-bound.

For example:
- A typical 1 flop/word application would be compared against GPUs like the GeForce GTX 1080Ti.
x??

---
#### Example of GPU Performance Points
Background context: The roofline plot also includes specific points for different GPU devices and applications, showing where they intersect based on their performance capabilities.

:p What does a typical 1 flop/word application look like in relation to GPUs?
??x
A typical 1 flop/word application would be compared against GPUs. For instance, the GeForce GTX 1080Ti built for the graphics market intersects above the typical 1 flop/word line on the roofline plot, indicating it is memory-bound.

This means that if the application's line intersects at a point above the GPU device (GeForce GTX 1080Ti) in the plot, the performance will be limited by memory bandwidth.
x??

---

#### PCI Bus Overview
Background context explaining the role of the PCI bus in data transfer between CPU and GPU. It is critical for performance, especially for applications involving significant data transfer.
:p What does the PCI bus facilitate in terms of data transfer?
??x
The PCI bus facilitates the communication and data transfer between the CPU and the GPU. This is crucial because all data communication between a dedicated GPU and the CPU occurs over this bus, making it a key component affecting overall application performance.
x??

---

#### Theoretical Bandwidth Calculation for PCI Bus
Explanation of how to calculate the theoretical bandwidth using the formula provided in the text. Note that the transfer rate is usually reported in GT/s (GigaTransfers per second).
:p How do you calculate the theoretical bandwidth of a PCI bus?
??x
To calculate the theoretical bandwidth of a PCI bus, use the following formula:
\[ \text{Theoretical Bandwidth (GB/s)} = \text{Lanes} \times \text{TransferRate (GT/s)} \times \text{OverheadFactor(Gb/GT)} \times \frac{\text{byte}}{8\text{ bits}} \]

Where:
- **Lanes** is the number of PCIe lanes.
- **TransferRate (GT/s)** is the maximum transfer rate per lane, which depends on the generation of the PCI bus.
- **OverheadFactor(Gb/GT)** is due to an encoding scheme used for data integrity. For Gen1 and Gen2, it's 80%, while from Gen3 onward, it’s approximately 98.46%.

For example:
```java
// Assuming a Gen3 system with 16 lanes and transfer rate of 8 GT/s
double theoreticalBandwidth = 16 * 8 * 0.9846 * (1/8);
```
x??

---

#### Determining the Number of PCIe Lanes
Explanation on how to determine the number of PCIe lanes using tools like `lspci` and `dmidecode`.
:p How can you find out the number of PCIe lanes in your system?
??x
You can determine the number of PCIe lanes by using command-line utilities such as `lspci` or `dmidecode`. For example, with `lspci`, you can run:
```bash
$ lspci -vmm | grep "PCI bridge" -A2
```
This will show information about the PCI bridges and indicate the number of lanes. Alternatively, `dmidecode` provides similar information:
```bash
$ dmidecode | grep "PCI"
```
For example, an output might include `(x16)` which indicates 16 lanes.
x??

---

#### Maximum Transfer Rate Determination
Explanation on how to find out the maximum transfer rate for a PCI bus by looking at the link capacity in `lspci` output.
:p How do you determine the maximum transfer rate of your PCI bus?
??x
To determine the maximum transfer rate, you can use the `lspci` command with verbose output:
```bash
$ sudo lspci -vvv | grep -E 'PCI|LnkCap'
```
This will provide information such as "Link Capacity: Port #2, Speed 8GT/s". The speed in GT/s indicates the maximum transfer rate per lane.
x??

---

#### Overhead Factor in PCI Bus Data Transfer
Explanation on how overhead affects data transfer efficiency and the different factors for Gen1, Gen2, and Gen3+ generations.
:p What is the impact of overhead factor on PCI bus data transfer?
??x
The overhead factor impacts the effective bandwidth of data transfer across the PCI bus. For older generations (Gen 1 and Gen 2), an overhead of 20% reduces the effective bandwidth to 80%. Starting with Gen3, this overhead drops significantly to approximately 1.54%, making the theoretical bandwidth nearly equal to the nominal transfer rate.

For example:
- **Gen1**: \( \text{OverheadFactor} = 1 - 0.20 = 0.80 \)
- **Gen3 and above**: \( \text{OverheadFactor} = 1 - 0.0154 = 0.9846 \)

This means Gen3+ systems can achieve nearly the full theoretical bandwidth.
x??

---

