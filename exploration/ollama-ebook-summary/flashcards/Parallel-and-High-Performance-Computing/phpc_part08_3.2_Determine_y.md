# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 8)

**Starting Chapter:** 3.2 Determine your hardware capabilities Benchmarking. 3.2.1 Tools for gathering system characteristics

---

#### Determining Hardware Capabilities: Benchmarking

Background context explaining the concept. To characterize the hardware for performance, various metrics are used such as FLOPs/s (floating-point operations per second), data transfer rate between memory levels (GB/s), and energy usage (Watts). These help in understanding the theoretical peak performance of different components.

Empirical measurements using micro-benchmarks like STREAM can provide insights into real-world performance. Theoretical models give an upper bound, while empirical measurements are closer to actual operating conditions.

:p What are some metrics used to characterize hardware for performance?
??x
Some key metrics include the floating-point operations per second (FLOPs/s), data transfer rate between memory levels (GB/s), and energy usage (Watts). These help in understanding the theoretical peak performance of different components.
x??

---

#### Tools for Gathering System Characteristics

The lstopo program from hwloc package provides a graphical view of hardware on your system. It is bundled with most MPI distributions.

:p What tool can be used to get a graphical view of hardware on your system?
??x
The `lstopo` program from the `hwloc` package can be used to get a graphical view of hardware on your system.
x??

---

#### Installing Cairo and HWLOC

Instructions for installing Cairo v1.16.0 and HWLOC v2.1.0a1-git are provided.

:p How do you install Cairo v1.16.0?
??x
To install Cairo v1.16.0, follow these steps:
1. Download Cairo from https://www.cairographics.org/releases/
2. Configure it using the following commands:
   ```
   ./configure --with-x --prefix=/usr/local
   make
   make install
   ```
x??

---

#### Installing HWLOC

Instructions for installing HWLOC v2.1.0a1-git are provided.

:p How do you install HWLOC v2.1.0a1-git?
??x
To install HWLOC v2.1.0a1-git, follow these steps:
1. Clone the hwloc package from Git: https://github.com/open-mpi/hwloc.git
2. Configure it using the following commands:
   ```
   ./configure --prefix=/usr/local
   make
   make install
   ```
x??

---

#### Hardware Topology Example

The lstopo command outputs a graphical view of hardware, such as for a Mac laptop.

:p What does the `lstopo` command output?
??x
The `lstopo` command outputs a graphical view of the hardware on your system. For example, it can show the NUMA nodes, processors, and levels of cache.
x??

---

#### Probing Hardware Details

Linux commands like `lscpu`, Windows commands like `wmic`, and Mac commands like `sysctl` or `system_profiler` are useful for probing hardware details.

:p What Linux command provides a consolidated report on CPU information?
??x
The `lscpu` command in Linux outputs a consolidated report of the information from the `/proc/cpuinfo` file. It helps determine the number of processors, processor model, cache sizes, and clock frequency.
x??

---

#### Vector Instruction Sets

Information on vector instruction sets like AVX2 can be found using commands such as `lscpu`.

:p What does the lscpu command output for a specific CPU?
??x
The `lscpu` command outputs information about a specific CPU, including the number of processors, processor model, cache sizes, and clock frequency. It also provides flags indicating available vector instruction sets like AVX2.
For example:
```text
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                4
On-line CPU(s) list:   0-3
Thread(s) per core:    1
Core(s) per socket:    4
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 65
Model name:            Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz
Stepping:              4
CPU MHz:               3194.078
BogoMIPS:              6388.15
Hypervisor vendor:     VMware
Virtualization type:   full
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              8192K
NUMA node0 CPU(s):     0-3
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts pcrs
```
This output shows that the AVX2 and various forms of the SSE vector instruction set are available.
x??

---

---
#### CPU Theoretical Flops Calculation
This section describes how to calculate the theoretical maximum floating-point operations per second (FLOPS) for a processor. It considers hyper-threading and turbo boost features.

:p How do we calculate the theoretical FLOPS of a mid-2017 MacBook Pro with an Intel Core i7-7920HQ processor?
??x
To calculate the theoretical FLOPS, we use the formula: 
\[ \text{FT} = C_v \times f_c \times I_c \]
Where:
- \( C_v \) is the number of virtual cores (considering hyperthreads),
- \( f_c \) is the clock rate,
- \( I_c \) is the operations per cycle, including FMA instructions.

For a dual-core processor with 2 hyperthreads (4 physical cores):
\[ C_v = Ch \times HT = (4 \text{ physical cores} \times 2 \text{ hyperthreads}) = 8 \]
The turbo boost clock rate \( f_c \) is 3.7 GHz, and the operations per cycle \( I_c \) can be calculated as:
\[ I_c = \frac{\text{Vector Width (VW)}}{\text{Word Size (Wbits)}} \times FMA \]
For a 256-bit vector unit with a word size of 64 bits:
\[ I_c = \left(\frac{256}{64}\right) \times 2 = 8 \]

Thus, the theoretical maximum flops is:
\[ FT = (8 \text{ virtual cores}) \times (3.7 \text{ GHz}) \times (8 \text{ Flops/Cycle}) = 236.8 \text{ GFLOPS} \]
x??

---
#### Memory Bandwidth Calculation
This section explains how to calculate the theoretical memory bandwidth of a system's main memory, considering factors like memory channels and socket configurations.

:p How do we calculate the theoretical memory bandwidth for a system with dual-socket motherboards?
??x
To calculate the theoretical memory bandwidth (BT), use the formula:
\[ BT = MTR \times Mc \times Tw \times Ns \]
Where:
- \( MTR \) is the data transfer rate in millions of transfers per second (MT/s),
- \( Mc \) is the number of memory channels,
- \( Tw \) is the memory transfer width in bits,
- \( Ns \) is the number of sockets.

For a dual-socket motherboard:
- The memory transfer width \( Tw \) is 64 bits, and since there are 8 bits per byte, 8 bytes are transferred.
- If DDR memory is used, the data transfer rate (MTR) can be derived from the clock rate. DDR memory performs transfers at both ends of the cycle for two transactions per cycle. This means that if the memory bus clock rate is \( x \text{ MHz} \), the MTR would be \( 2x \).
- If there are 4 memory channels and 1 socket:
\[ BT = (MTR \times Mc \times Tw \times Ns) \]
For example, with a transfer rate of 3200 MT/s, 4 memory channels, 64 bits transfer width, and 1 socket:
\[ BT = (3200 \text{ MT/s} \times 4 \times 8/8 \times 1) = 12800 \text{ MB/s} \]

Thus, the theoretical memory bandwidth for this configuration is 12800 MB/s.
x??

---
#### Cache Hierarchy in Modern Systems
This section provides background on how cache hierarchies have evolved to manage data access in modern CPUs.

:p What is the significance of the memory hierarchy and its evolution over time?
??x
The memory hierarchy has grown deeper with the addition of multiple levels of cache, designed to bridge the speed gap between processing units and main memory. Modern processors use a multi-level cache system (L1, L2, L3) to store frequently accessed data closer to the CPU cores.

The general formula for calculating theoretical memory bandwidth is:
\[ BT = MTR \times Mc \times Tw \times Ns \]
Where:
- \( MTR \) is the data transfer rate in millions of transfers per second (MT/s),
- \( Mc \) is the number of memory channels,
- \( Tw \) is the memory transfer width in bits,
- \( Ns \) is the number of sockets.

For example, a system with 3200 MT/s DDR memory, 4 memory channels, and 64-bit transfer width:
\[ BT = (3200 \text{ MT/s} \times 4 \times 8/8 \times 1) = 12800 \text{ MB/s} \]

This evolution helps in improving overall system performance by reducing the latency and increasing the throughput of data access.
x??

---

#### Memory Channels and Bandwidth
Background context explaining how memory channels work, their impact on bandwidth, and why replacing all modules is necessary when adding more. The theoretical and achievable memory bandwidth are discussed.

:p What is the effect of having two memory channels (Mc) on a system's performance?
??x
Having two memory channels allows for better bandwidth, as data can be read from both channels simultaneously, doubling the effective bandwidth compared to a single channel setup. This improvement comes at the cost of complexity; you cannot simply add more memory by inserting another module without replacing existing ones with larger modules.

For example, if each channel supports 2133 MT/s (Megatransfers per second) and there are two channels:

```plaintext
BT = 2133 MT/s × 2 channels × 8 bytes × 1 socket = 34,128 MiB/s or 34.1 GiB/s.
```

This formula calculates the theoretical bandwidth based on the memory transfer rate, number of channels, and data width.

x??

---

#### Memory Latency
Explanation about how long it takes to retrieve a single byte from different cache levels up to main memory, with a focus on the concept of memory latency. The impact of accessing contiguous data is also discussed.

:p What is memory latency?
??x
Memory latency refers to the time required for the first byte of data from each level of memory hierarchy (L1, L2, L3, or main memory) to be retrieved by the CPU. For a single byte, this can range from 4 cycles in a CPU register, up to 75 cycles in L1 cache, 10 cycles in L2 cache, and 400 cycles in main memory.

To optimize performance, data is loaded in chunks called cache lines (typically 64 bytes or 8 doubles) rather than one byte at a time. This reduces the number of times data needs to be accessed from slower memory levels.

x??

---

#### Empirical Measurement of Bandwidth
Explanation on how empirical measurements are used to determine real-world memory bandwidth, distinguishing between theoretical and practical performance. Mention specific tools like STREAM Benchmark and Empirical Roofline Toolkit.

:p What is the difference between theoretical and measured memory bandwidth?
??x
Theoretical memory bandwidth represents the maximum potential speed at which data can be transferred from main memory into the CPU, based on factors such as clock rate, channel count, and bus width. However, practical performance is often lower due to overhead in the memory hierarchy.

For instance, a 2017 MacBook Pro with LPDDR3-2133 memory has a theoretical bandwidth of:

```plaintext
BT = 2133 MT/s × 2 channels × 8 bytes × 1 socket = 34,128 MiB/s or 34.1 GiB/s.
```

Empirical measurements use tools like the STREAM Benchmark to measure actual performance. For this system, empirical measurements showed a bandwidth of about 22 GiB/s.

x??

---

#### Cache Lines and Data Transfer
Explanation on why loading data in cache lines (64 bytes or 8 doubles) is more efficient than loading one byte at a time, detailing the impact on performance.

:p Why is loading data in cache lines important?
??x
Loading data in cache lines improves performance significantly because it reduces the number of times slower memory levels need to be accessed. When contiguous data is loaded into cache lines (typically 64 bytes or 8 doubles), nearby values can be reused without needing to access main memory multiple times.

For example, if a single byte is requested and not in any cache, it would take around 50 cycles/double to load from main memory. However, loading an entire cache line at once (64 bytes) allows the CPU to reuse data more efficiently, reducing the total number of memory accesses and improving overall performance.

x??

---

#### Roofline Model
Explanation on how the roofline model integrates both memory bandwidth limit and peak flop rate into a single plot, distinguishing its use from traditional theoretical models.

:p What is the purpose of the roofline model?
??x
The roofline model combines both the memory bandwidth limit and the peak floating-point capability (FLOPS) into one plot. It helps in understanding the limits of performance for different types of computations and can identify whether a system is limited by memory bandwidth or computational speed.

For example, if you are doing big calculations out of main memory, the model would show that performance is bottlenecked by the memory bandwidth. The roofline model uses regions to illustrate each performance limit:

- The line represents peak flop rate.
- The region above this line indicates memory-bound workloads where memory bandwidth limits performance.
- The region below this line shows compute-bound workloads where computational speed limits performance.

x??

#### The STREAM Benchmark Overview
Background context: The STREAM benchmark is a suite of memory bandwidth and cache performance benchmarks. It measures how fast a computer can move data between main memory and CPU registers or caches, using various arithmetic operations.

:p What is the purpose of the STREAM Benchmark?
??x
The purpose of the STREAM Benchmark is to measure the maximum rate at which data can be loaded from main memory into CPU registers or caches. This is done by performing different arithmetic operations on large arrays.
x??

---

#### Different Variants in STREAM Benchmark
Background context: The STREAM benchmark includes four variants: copy, scale, add, and triad. These each perform a specific type of operation to measure the bandwidth under different conditions.

:p What are the four variants in the STREAM Benchmark?
??x
The four variants in the STREAM Benchmark are:
1. Copy: \(a(i) = b(i)\)
2. Scale: \(a(i) = q * b(i)\)
3. Add: \(a(i) = b(i) + c(i)\)
4. Triad: \(a(i) = b(i) + q * c(i)\)

Each variant measures the bandwidth under different conditions of arithmetic operations.
x??

---

#### Measuring Bandwidth with STREAM Benchmark
Background context: Jeff Hammond's version of the McCalpin STREAM Benchmark is used to measure bandwidth on a CPU. The benchmark involves running the code and analyzing the results for each operation.

:p How do you use the STREAM Benchmark to measure bandwidth?
??x
To use the STREAM Benchmark to measure bandwidth, follow these steps:
1. Clone the Git repository: `git clone https://github.com/jeffhammond/STREAM.git`
2. Edit the makefile and change the compile line to include optimization flags:
   ```makefile
   make -O3 -march=native -fstrict-aliasing -ftree-vectorize -fopenmp \
   -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20
   ```
3. Run the executable: `./stream_c.exe`

The results provide the best rate in MB/s for each operation, which can be used to determine the maximum bandwidth.
x??

---

#### Results from 2017 Mac Laptop Example
Background context: The STREAM Benchmark results on a specific hardware (2017 Mac Laptop) are provided. These results help in determining the empirical value of maximum bandwidth.

:p What were the results for the 2017 Mac Laptop?
??x
The results for the 2017 Mac Laptop from the STREAM Benchmark are as follows:
- Copy: Best Rate = 22,086.5 MB/s, Avg Time = 0.060570 s, Min Time = 0.057954 s, Max Time = 0.062090 s
- Scale: Best Rate = 16,156.6 MB/s, Avg Time = 0.081041 s, Min Time = 0.079225 s, Max Time = 0.082322 s
- Add: Best Rate = 16,646.0 MB/s, Avg Time = 0.116622 s, Min Time = 0.115343 s, Max Time = 0.117515 s
- Triad: Best Rate = 16,605.8 MB/s, Avg Time = 0.117036 s, Min Time = 0.115622 s, Max Time = 0.118004 s

The best bandwidth can be selected from these results as the empirical value of maximum bandwidth.
x??

---

#### Roofline Model Concept
Background context: The roofline model is a graphical representation that shows the relationship between arithmetic intensity and floating-point operations per second (FLOPS) on a computer. It helps in understanding where an application's performance bottlenecks are.

:p What is the roofline model?
??x
The roofline model is a graphical tool used to visualize the relationship between memory bandwidth, FLOPS, and arithmetic intensity on a computer. The model consists of a horizontal line representing maximum theoretical FLOPS (memory bandwidth) and a sloped line that represents achievable FLOPS as a function of arithmetic intensity.

For high arithmetic intensity, where there are many floating-point operations compared to the data loaded, the maximum theoretical FLOPS is the limit. As arithmetic intensity decreases, memory load times dominate, reducing the achievable FLOPS.
x??

---

#### Applying Roofline Model
Background context: The roofline model can be applied to a CPU or GPU by determining the maximum theoretical FLOPS and plotting it against different levels of arithmetic intensity.

:p How do you apply the roofline model?
??x
To apply the roofline model, follow these steps:
1. Determine the maximum theoretical FLOPS (memory bandwidth) using the STREAM Benchmark results.
2. Plot a horizontal line on a graph representing this maximum FLOPS.
3. For different levels of arithmetic intensity, plot points showing achievable FLOPS. As arithmetic intensity decreases, the slope of the line decreases.

The resulting plot will show a characteristic "roofline" shape, indicating where the application performance is limited by memory bandwidth or FLOPS.
x??

---

---

#### Installing Software on Macs

Background context: This section covers the installation process for `gnuplot v4.2`, `Python v3.0`, and the GCC compiler to replace the default compiler, which is necessary for running the Roofline Toolkit effectively.

:p What are the steps needed to install `gnuplot v4.2` and `Python v3.0` on a Mac using a package manager?

??x
The first step involves installing `gnuplot v4.2` and `Python v3.0` using Homebrew, a popular package manager for macOS.

To do this:
1. Open the Terminal application.
2. Install Homebrew by running the following command in the terminal: 
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/main/install.sh)"
   ```
3. Once Homebrew is installed, use it to install `gnuplot v4.2` and Python 3 using:
   ```bash
   brew install gnuplot python@3.0
   ```

For replacing the default compiler with GCC, you can install it via Homebrew as well:
```bash
brew install gcc
```
x??

---

#### Cloning Roofline Toolkit

Background context: This section explains how to clone the Roofline Toolkit from a Git repository and navigate through its directory structure.

:p How do you clone the Roofline Toolkit from Git?

??x
To clone the Roofline Toolkit from Git, use the following command in your terminal:
```bash
git clone https://bitbucket.org/berkeleylab/cs-roofline-toolkit.git
```
This command clones the repository into a local directory on your machine.

Next, navigate to the cloned directory using `cd`:
```bash
cd cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0
```

To copy the configuration file, use the following command:
```bash
cp Config/config.madonna.lbl.gov.01 Config/MacLaptop2017
```
x??

---

#### Editing Configuration File

Background context: This section describes how to edit a configuration file for running empirical roofline measurements.

:p How do you edit the `Config/MacLaptop2017` file?

??x
To edit the `Config/MacLaptop2017` file, use any text editor. For instance, using the `nano` editor:
```bash
nano Config/MacLaptop2017
```

You can then make changes to parameters such as the number of MPI ranks and threads, compiler flags, and other settings that define how the Roofline Toolkit will run its tests.

For example, you might need to adjust the `ERT_MPI_PROCS` and `ERT_OPENMP_THREADS` values based on your system configuration.
x??

---

#### Running Tests

Background context: This section describes running empirical roofline measurements using the Roofline Toolkit's command-line interface.

:p How do you run empirical tests with the Roofline Toolkit?

??x
To run empirical tests, navigate to the directory containing the configuration file and execute:
```bash
./ert Config/MacLaptop2017
```
This command runs the tests based on the settings defined in `Config/MacLaptop2017`. The results will be saved in a subdirectory named `Results.MacLaptop.01`, which contains various output files, including a PostScript file that can be used to visualize the roofline plot.

You can then view the results using a tool like `gv` or by opening the `.ps` file with Ghostscript:
```bash
gv MacLaptop2017/Run.001/roofline.ps & 
```
x??

---

#### Viewing Results

Background context: This section explains how to interpret and view the empirical roofline measurement results.

:p How do you view the results of the empirical roofline measurements?

??x
To view the results, navigate to the `Results.MacLaptop.01` directory where the Roofline Toolkit stores its output files:
```bash
cd MacLaptop2017/Run.001/
```

You can use a tool like Ghostscript or any PostScript viewer to open and view the roofline plot, located in `roofline.ps`. To open it with Ghostscript directly from the terminal:
```bash
gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -o roofline.pdf roofline.ps 
```
This command converts the PostScript file into a PDF for easy viewing.

Alternatively, you can use any PostScript viewer such as `gv` (Graphviz Viewer):
```bash
gv roofline.ps &
```

The resulting plot will show the empirical measurements of performance limits and help in understanding the actual hardware capabilities.
x??

---

---
#### Machine Balance Calculation
Background context: The machine balance is a critical concept that helps understand how efficiently the hardware utilizes its computational power and memory bandwidth. It is calculated by dividing the floating-point operations per second (FLOPs) by the memory bandwidth.

The theoretical machine balance \( MB_T \) can be computed using:
\[ MB_T = \frac{FT}{BT} = \frac{236.8 \, \text{GFlops/s}}{34.1 \, \text{GiB/s} \times (8 \, \text{bytes/word})} \]
where \( FT \) is the theoretical peak floating-point operations per second and \( BT \) is the memory bandwidth in GiB/s.

Empirically, it can be calculated using:
\[ MB_E = \frac{FE}{BE} = \frac{264.4 \, \text{GFlops/s}}{22 \, \text{GiB/s} \times (8 \, \text{bytes/word})} \]

:p How do you calculate the machine balance?
??x
The machine balance is calculated by dividing the floating-point operations per second (FLOPs) by the memory bandwidth. It helps determine whether your application is flop-bound or bandwidth-bound.

For theoretical calculation:
\[ MB_T = \frac{236.8 \, \text{GFlops/s}}{34.1 \times 8 \, \text{GiB/s}} \approx 56 \, \text{Flops/word} \]

For empirical calculation:
\[ MB_E = \frac{264.4 \, \text{GFlops/s}}{22 \times 8 \, \text{GiB/s}} \approx 96 \, \text{Flops/word} \]
x??

---
#### Roofline Model
Background context: The roofline model is a graphical tool that illustrates the theoretical and achievable performance of an application. It helps in understanding the efficiency of your code by visualizing the relationship between floating-point operations (FLOPs) and memory bandwidth.

In the provided graph, the horizontal line represents the maximum FLOPs, while the sloped lines represent the various levels of cache and DRAM bandwidths.

:p What does the roofline model help you understand?
??x
The roofline model helps visualize the theoretical peak performance limits and actual achievable performance by showing the relationship between floating-point operations (FLOPs) and memory bandwidth. It assists in identifying whether an application is limited by FLOPs or memory bandwidth, which can guide optimizations.

In the provided graph:
- The horizontal line represents the maximum FLOPs.
- The sloped lines represent different levels of cache and DRAM bandwidths.
x??

---
#### Call Graph Analysis
Background context: Call graphs provide a visual representation of subroutine dependencies in an application. They highlight hot spots, which are routines that consume significant execution time.

Tools like Valgrind’s cachegrind can generate call graphs, combining information about hot spots and subroutine dependencies for deeper analysis.

:p How do you use call graphs to analyze your application?
??x
Call graphs help identify performance bottlenecks by visualizing the execution flow of subroutines. They highlight routines that consume significant time (hot spots) and show how these subroutines depend on each other.

To generate a call graph using Valgrind’s cachegrind, you would run:
```bash
valgrind --tool=cachegrind ./your_application
```
After execution, the `cachegrind.out.x` file will contain the profiling data. Using tools like KCacheGrind or other visualization tools can help analyze this data to find hot spots and dependencies.

For example, if you run:
```bash
kcachegrind cachegrind.out.x
```
It opens an interface where you can explore the call graph and see execution times for each routine.
x??

---

#### Profiling and Performance Analysis
Background context: This section describes how to use profiling tools like Callgrind from Valgrind suite for performance analysis. The objective is to identify bottlenecks and optimize code through techniques such as parallelization with OpenMP and vectorization.

:p What are the key steps involved in using Callgrind for generating a call graph of an application?
??x
The key steps involve installing necessary tools, downloading and building the mini-app CloverLeaf, running Valgrind with specific options to generate a call graph file, and visualizing it using QCacheGrind.

Example code:
```sh
# Step 1: Install Valgrind and KcacheGrind or QCacheGrind
sudo apt-get install valgrind kcachegrind

# Step 2: Clone CloverLeaf repository
git clone --recursive https://github.com/UK-MAC/CloverLeaf.git

# Step 3: Build the serial version of CloverLeaf
cd CloverLeaf/CloverLeaf_Serial
make COMPILER=GNU IEEE=1 C_OPTIONS="-g -fno-tree-vectorize" OPTIONS="-g -fno-tree-vectorize"

# Step 4: Run Valgrind with Callgrind tool
cp InputDecks/clover_bm256_short.in clover.in
edit clover.in and change cycles from 87 to 10
valgrind --tool=callgrind -v ./clover_leaf

# Step 5: Visualize the call graph using QCacheGrind
qcachegrind
```
x??

---

#### Call Graph and Its Visualization
Background context: The call graph generated by Callgrind provides insights into how different functions are called during program execution. Understanding this can help identify performance bottlenecks.

:p What is a call stack, and how does it work?
??x
A call stack is a chain of routines that calls the present location in the code. When a routine calls another subroutine, it pushes its address onto the stack. At the end of the routine, the program pops the address off the stack as it returns to the prior calling routine. This hierarchical structure helps track function invocations and their return paths.

Explanation:
```java
public class Example {
    public void methodA() {
        System.out.println("Method A");
        methodB();
    }

    public void methodB() {
        System.out.println("Method B");
    }
}
```
In the above example, if `methodA` is called and then calls `methodB`, the call stack would first push `methodA`'s address onto the stack, execute its body (prints "Method A"), and then push `methodB`'s address onto the stack. Once `methodB` finishes executing, both addresses are popped off the stack in reverse order.

x??

---

#### Visualizing Call Graph with QCacheGrind
Background context: QCacheGrind is a tool that visualizes call graphs generated by Callgrind. It provides detailed information on how functions are called and consumed time.

:p How do you load a specific Callgrind output file into QCacheGrind?
??x
To load a specific `callgrind.out` file into QCacheGrind, follow these steps:
1. Start QCacheGrind with the command: `qcachegrind`
2. Right-click on "Call Graph" in the main window.
3. Select "Load" and choose the `callgrind.out.XXX` file generated by Callgrind.

Example:
```sh
# Start QCacheGrind
qcachegrind

# Load a specific callgrind output file into QCacheGrind
qcachegrind callgrind.out.1234567890  # Replace with the actual filename
```

x??

---

#### CloverLeaf Performance Study
Background context: The example uses CloverLeaf to understand performance optimizations by comparing serial and parallel versions of a similar application.

:p What is the purpose of profiling the serial version of CloverLeaf?
??x
The purpose of profiling the serial version of CloverLeaf is to identify where time is spent in the code, which can help guide decisions on how to optimize it. Specifically, this step aims to understand the performance characteristics and potential bottlenecks before applying parallelization techniques like OpenMP and vectorization.

Explanation:
```sh
# Build the serial version
make COMPILER=GNU IEEE=1 C_OPTIONS="-g -fno-tree-vectorize" OPTIONS="-g -fno-tree-vectorize"

# Run Valgrind with Callgrind tool for profiling
valgrind --tool=callgrind -v ./clover_leaf

# Load the output file into QCacheGrind to visualize the call graph
qcachegrind callgrind.out.1234567890  # Replace with the actual filename
```

x??

---

#### Inclusive Timing Measurement
Inclusive timing measures the total time spent within a function, including its child functions. This is useful for understanding the overall contribution of a routine to the entire runtime.

:p What does inclusive timing measure?
??x
Inclusive timing measures the total run time of a function, including the time taken by all its called subroutines. This provides an accurate picture of how much of the overall execution time is spent within that specific function.
x??

---

#### Call Hierarchy and Run Time Analysis
The call hierarchy in profiling tools shows the routines calling each other and their respective run times. Understanding the call hierarchy helps identify where most of the runtime is spent.

:p What does a call hierarchy reveal about an application's performance?
??x
A call hierarchy reveals which functions are called by others, and how much time is spent in these functions. This helps pinpoint inefficient or heavily used routines that might benefit from optimization.
x??

---

#### CloverLeaf Performance Profiling with Intel Advisor
Intel Advisor can be used to profile applications and generate roofline models, providing insights into arithmetic intensity and computational rates.

:p How does Intel Advisor help in performance profiling?
??x
Intel Advisor helps by collecting runtime data on an application, analyzing it using the roofline model, which separates computation from memory access efficiency. This allows developers to identify bottlenecks related to both floating-point operations and memory usage.
x??

---

#### Roofline Model in Action
The roofline model visualizes computational limits and performance, showing the theoretical maximum performance based on arithmetic intensity and hardware capabilities.

:p What is the purpose of using the roofline model?
??x
The purpose of using the roofline model is to understand the relationship between computational throughput (FLOPS) and memory bandwidth. It helps in identifying whether an application is limited by computation or memory access, guiding optimization strategies.
x??

---

#### Generating Roofline for CloverLeaf Mini-App
Steps include building the OpenMP version of CloverLeaf, running it with Intel Advisor, setting up the executable, and analyzing the roofline to determine arithmetic intensity.

:p How do you generate a roofline analysis using Intel Advisor for CloverLeaf?
??x
To generate a roofline analysis using Intel Advisor for CloverLeaf:
1. Clone and build the OpenMP version of CloverLeaf.
2. Run the application in the Intel Advisor tool.
3. Set the executable to clover_leaf and configure the project directory.
4. Start Survey Analysis, then Roofline Analysis.
5. Load the run data and view performance results.

Code Example:
```bash
# Clone and build CloverLeaf OpenMP version
git clone --recursive https://github.com/UK-MAC/CloverLeaf.git
cd CloverLeaf/CloverLeaf_OpenMP
make COMPILER=INTEL IEEE=1 C_OPTIONS="-g -xHost" OPTIONS="-g -xHost"
```
x??

---

#### Viewing Roofline Analysis Results
The roofline analysis summary provides details on arithmetic intensity and floating-point computational rate, helping to assess the efficiency of an application.

:p What information does the roofline analysis provide?
??x
The roofline analysis provides summaries such as arithmetic intensity (FLOPS/byte or FLOPS/word) and floating-point computational rates. These metrics help in assessing whether an application is memory-bound or compute-bound, guiding further optimization efforts.
x??

---

#### Summary Statistics from Intel Advisor
Summary statistics include metrics like arithmetic intensity and floating-point performance rate.

:p What are the key summary statistics generated by Intel Advisor?
??x
Key summary statistics generated by Intel Advisor include:
- Arithmetic intensity (FLOPS/byte or FLOPS/word)
- Floating-point computational rate in GFLOPS/s

These stats help understand the efficiency of the application's computation and memory access patterns.
x??

---

---
#### Lik- Wid (Like I Knew What I’m Doing) Tool
Background context: Lik- wid is a command-line tool for performance analysis and optimization, authored by Treibig, Hager, and Wellein at the University of Erlangen-Nuremberg. It runs on Linux systems and leverages hardware counters through the machine-specific registers (MSR) module.
:p What does Lik- wid stand for?
??x
Lik- wid stands for "Like I Knew What I'm Doing," indicating its utility in performance analysis without requiring extensive knowledge of low-level system details.
x??

---
#### MSR Module and Modprobe
Background context: The MSR (Machine-Specific Registers) module must be enabled to use Lik- wid effectively. This is done by running the command `sudo modprobe msr`.
:p How do you enable the MSR module on a Linux system?
??x
You enable the MSR module on a Linux system by executing the following command:
```sh
sudo modprobe msr
```
This command loads the module necessary for Lik- wid to access hardware counters.
x??

---
#### Roofline Plot and Performance Analysis
Background context: The roofline plot provides a visual representation of an application's performance relative to theoretical peak. In this specific case, the roofline plot shows that the CloverLeaf mini-app is bandwidth limited and far left of the compute-bound region, indicating poor arithmetic intensity.
:p What does it mean if an application is shown as bandwidth limited in a roofline plot?
??x
If an application is shown as bandwidth limited in a roofline plot, it means that the performance is constrained by memory bandwidth rather than computational power. This suggests that optimizing for better arithmetic intensity (more flops per byte) could significantly improve performance.
x??

---
#### Arithmetic Intensity Calculation
Background context: Arithmetic intensity measures the amount of computation relative to data movement. It's often used in performance analysis to identify bottlenecks.
:p How is arithmetic intensity calculated?
??x
Arithmetic intensity is calculated by dividing the total floating-point operations (FLOPs) by the total bytes of memory accessed. For double precision:
```plaintext
Arithmetic Intensity = FLOPs / Bytes
```
In this case, for CloverLeaf using double precision:
```plaintext
Operational Intensity = 41274 MFLOPs/sec / 123319.9692 MB/s = 0.33 FLOPs/byte
```
This value indicates that the application is not memory-bound, as it requires less than one floating-point operation per byte of data.
x??

---
#### Lik- Wid Profiling Command
Background context: The `likwid-perfctr` command allows profiling and performance analysis using hardware counters. It can be configured to measure various metrics such as runtime, clock frequency, energy usage, etc.
:p How do you run the Lik- wid profiler for CloverLeaf?
??x
To run the Lik- wid profiler for CloverLeaf on a Skylake Gold_6152 processor, use the following command:
```sh
likwid-perfctr -C 0-87 -g MEM_DP ./clover_leaf
```
This command runs CloverLeaf with CPU cores 0 to 87 and gathers memory-related data.
x??

---

#### LIKWID Marker Initialization and Usage
Background context: LIKWID is a performance analysis tool that can be used to measure specific sections of code using markers. This allows for detailed profiling and performance evaluation.

:p How do you initialize and use LIKWID markers in your code?
??x
To initialize and use LIKWID markers, follow these steps:

1. Insert the initialization line at the start of the section you want to profile:
   ```c
   LIKWID_MARKER_INIT;
   ```
2. Initialize each thread if needed (though typically not necessary for single-threaded regions):
   ```c
   LIKWID_MARKER_THREADINIT;
   ```
3. Register a performance counter name and start it before your code block:
   ```c
   LIKWID_MARKER_REGISTER("Compute");
   LIKWID_MARKER_START("Compute");
   ```

4. Execute the section of code you want to measure.

5. Stop the marker after executing the code:
   ```c
   LIKWID_MARKER_STOP("Compute");
   ```
6. Close all markers once your profiling is complete:
   ```c
   LIKWID_MARKER_CLOSE;
   ```

These steps help isolate and analyze specific performance aspects of your application.
x??

---

#### Generating Roofline Plots with Python
Background context: A roofline plot visually represents the theoretical hardware performance limits by plotting the arithmetic intensity against the computation rate. It is a powerful tool for understanding performance bottlenecks.

:p How can you generate a custom roofline plot using Python?
??x
To generate a custom roofline plot, follow these steps:

1. Clone and install the necessary scripts:
   ```bash
   git clone https://github.com/cyanguwa/nersc-roofline.git
   cd nersc-roofline/
   Plotting
   ```

2. Modify `data.txt` with your performance data.

3. Run the plotting script:
   ```python
   python plot_roofline.py data.txt
   ```

This command will generate a roofline plot based on the provided data, allowing you to visualize theoretical and actual performance metrics.
x??

---

#### Energy Savings Calculation for Parallel vs Serial Runs
Background context: Calculating energy savings between parallel and serial runs is crucial for understanding efficiency improvements. This involves comparing the total energy consumed in both scenarios.

:p How do you calculate the energy savings from a parallel run compared to a serial run?
??x
To calculate the energy savings, use the following formula:

\[
\text{Energy Savings} = \frac{\text{Energy in Serial Run} - \text{Energy in Parallel Run}}{\text{Energy in Serial Run}}
\]

For example:
- Serial Energy: 212747.7787
- Parallel Energy: 151590.4909

The energy savings calculation would be:

\[
\frac{(212747.7787 - 151590.4909)}{212747.7787} = \frac{61157.2878}{212747.7787} = 0.287
\]

This results in a 28.7% energy savings.
x??

---

#### Jupyter Notebook for Performance Characterization
Background context: Jupyter notebooks are interactive documents that can contain both code and rich text elements, making them ideal for dynamic data analysis and visualization.

:p How do you set up and run a Jupyter notebook to characterize the hardware platform?
??x
To set up and run a Jupyter notebook for hardware characterization:

1. Install Python3 using your package manager:
   ```bash
   brew install python3
   ```

2. Use pip to install necessary packages:
   ```bash
   pip install numpy scipy matplotlib jupyter
   ```

3. Download the Jupyter notebook from GitHub:
   ```
   https://github.com/EssentialsofParallelComputing/Chapter3
   ```

4. Open and run the Jupyter notebook named `HardwarePlatformCharacterization.ipynb`.

5. Change the hardware settings in the first section to match your platform.

6. Run all cells in the notebook to perform calculations and generate plots.

This setup allows you to dynamically calculate theoretical performance characteristics and plot them using matplotlib.
x??

---

#### Theoretical Hardware Characteristics
Background context: Understanding theoretical hardware characteristics helps in setting realistic benchmarks for actual application performance.

:p How do you run calculations for theoretical hardware characteristics in a Jupyter notebook?
??x
To run calculations for theoretical hardware characteristics in the Jupyter notebook:

1. Change the hardware settings at the beginning of `HardwarePlatformCharacterization.ipynb` to reflect your platform.
2. Run all cells in the notebook to perform these calculations.

For example, the notebook might calculate metrics such as peak memory bandwidth and FLOPS per second.

```python
# Example calculation cell
peak_memory_bandwidth = 100  # Gbps
flops_per_second = 50e9      # GFLOPs/s

print(f"Peak Memory Bandwidth: {peak_memory_bandwidth} GBps")
print(f"FLOPS per Second: {flops_per_second}")
```

These cells will dynamically update based on the hardware settings you input.
x??

---

#### Plotting Arithmetic Intensity and Performance
Background context: Visualizing arithmetic intensity (KI) against performance helps in identifying bottlenecks and optimizing applications.

:p How do you plot arithmetic intensity and performance using matplotlib in a Jupyter notebook?
??x
To plot arithmetic intensity and performance using matplotlib in a Jupyter notebook:

1. Download the plotting script from the GitHub repository.
2. Modify `data.txt` with your measured performance data.
3. Run the following Python code to generate the roofline plot:
   ```python
   import matplotlib.pyplot as plt

   KI = [0.5, 0.6, 0.7]  # Example arithmetic intensities
   performance = [10, 20, 30]  # Corresponding performance rates in GFLOPs/s
   
   plt.plot(KI, performance)
   plt.xlabel('Arithmetic Intensity (KI)')
   plt.ylabel('Performance (GFLOPs/s)')
   plt.title('Roofline Plot')
   plt.show()
   ```

This code will plot the roofline and help you visualize how your application's performance compares to theoretical limits.
x??

---

#### Intel Software Development Emulator (SDE) for Performance Analysis
Background context: SDE is a tool that provides detailed performance information, including arithmetic intensity, which can be used alongside other tools like LIKWID.

:p How does the Intel Software Development Emulator (SDE) work?
??x
The Intel Software Development Emulator (SDE) works by running your application in an emulator environment. It generates extensive data on operations, cache behavior, and performance metrics.

To use SDE:

1. Install the SDE package from Intel.
2. Run your application with the SDE:
   ```bash
   sde -o output_dir -- /path/to/your/app
   ```

This command will run your application in an emulator mode, collecting detailed information that can be used to calculate arithmetic intensity.

Example:
```c
// Example C code snippet
void compute(int size) {
    for (int i = 0; i < size; ++i) {
        // Some computation here
    }
}
```

When run with SDE, it provides insight into the operations performed and their impact on performance.
x??

---

#### Intel VTune Performance Tool
Background context: Intel VTune is another powerful tool for collecting detailed performance data, which can be used to analyze arithmetic intensity.

:p How does Intel VTune work?
??x
Intel VTune works by profiling your application in various ways. It collects data such as memory access patterns, CPU events, and computation details.

To use VTune:

1. Install the Parallel Studio package, which includes VTune.
2. Run a profiling session:
   ```bash
   vtune -collect hotspots -- /path/to/your/app
   ```

This command will run your application while collecting data on performance hotspots, including arithmetic intensity.

Example:
```c
// Example C code snippet
void compute(int size) {
    for (int i = 0; i < size; ++i) {
        // Some computation here
    }
}
```

VTune provides detailed reports and visualizations of this data, helping you understand where optimizations can be made.
x??

---

