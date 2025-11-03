# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 56)

**Starting Chapter:** 17.4 Benchmarks and mini-apps A window into system performance

---

#### HPCToolkit Overview
HPCToolkit is a powerful, open-source profiler developed by Rice University. It uses hardware performance counters to measure application performance and presents detailed data through graphical user interfaces. HPCToolkit is sponsored by the DOE Exascale Computing Project for extreme-scale computing.
:p What does HPCToolkit use to measure performance?
??x
HPCToolkit uses hardware performance counters to measure performance. These counters provide detailed metrics on how your application is utilizing various hardware resources, such as CPU cycles, cache misses, and memory bandwidth.
x??

---

#### HPCToolkit GUI Components
The HPCToolkit graphical user interfaces (GUIs) include `hpcviewer` for code-level perspective and `hpctraceviewer` for a time trace of code execution. These tools help in understanding performance bottlenecks at various levels of granularity.
:p What are the two main components of HPCToolkit's GUI?
??x
The two main components of HPCToolkit's GUI are:
1. `hpcviewer`, which provides a code-level perspective on performance data.
2. `hpctraceviewer`, which presents a time trace of code execution.

These tools help in understanding the performance bottlenecks at different levels of detail.
x??

---

#### Installing HPCToolkit
HPCToolkit can be installed using Spack, a package manager for scientific software, with the command:
```
spack install hpctoolkit
```
:p How is HPCToolkit installed?
??x
HPCToolkit can be installed using the Spack package manager with the following command:

```bash
spack install hpctoolkit
```

This command installs HPCToolkit, making it available for use in profiling applications.
x??

---

#### Open|SpeedShop Overview
Open|SpeedShop is another detailed profiler that supports MPI, OpenMP, and CUDA. It offers both a graphical user interface and a command-line interface. The tool runs on the latest high-performance computing systems due to DOE funding.
:p What does Open|SpeedShop support?
??x
Open|SpeedShop supports:
- MPI (Message Passing Interface)
- OpenMP (Open Multi-Processing)
- CUDA (Compute Unified Device Architecture)

These capabilities make it a versatile profiler for applications using parallel and GPU technologies.
x??

---

#### Installing Open|SpeedShop
Open|SpeedShop can be installed with Spack using the command:
```
spack install openspeedshop
```
:p How is Open|SpeedShop installed?
??x
Open|SpeedShop can be installed with Spack using the following command:

```bash
spack install openspeedshop
```

This command installs Open|SpeedShop, making it available for profiling applications that require detailed performance analysis.
x??

---

#### TAU Overview
TAU (Tool to support Analysis of Programs) is a freely available profiling tool developed primarily at the University of Oregon. It has a graphical user interface that simplifies its use and is widely used in high-performance computing applications.
:p What is TAU?
??x
TAU, or Tool to Support Analysis of Programs, is a freely available profiling tool developed primarily at the University of Oregon. It offers a simple and intuitive graphical user interface, making it easy for users to understand and utilize its capabilities.

TAU is extensively used in high-performance computing applications due to its ease of use and effectiveness.
x??

---

#### TAU Installation
TAU can be installed using Spack with the command:
```
spack install tau
```
:p How is TAU installed?
??x
TAU can be installed using Spack with the following command:

```bash
spack install tau
```

This command installs TAU, making it available for use in profiling applications.
x??

---

#### Benchmarking and Mini-Apps Overview
Benchmarking is used to measure system performance, while mini-apps focus on specific application areas. Benchmarks like Linpack, STREAM, Random, NAS Parallel Benchmarks, HPCG, and HPC Challenge are useful for evaluating different aspects of system performance.
:p What is the primary purpose of benchmarks in assessing system performance?
??x
Benchmarks provide a standardized way to measure the performance characteristics of a computing system. They help identify strengths and weaknesses, allowing developers to optimize applications accordingly.
x??

---
#### Linpack Benchmark
Linpack is used for measuring the Top 500 High Performance Computers list. It evaluates a system's floating-point arithmetic operations by solving a large dense system of linear equations.
:p What does the Linpack benchmark primarily measure?
??x
The Linpack benchmark measures the performance of a computer in solving a large dense system of linear equations, which is often used to rank high-performance computers on the Top 500 list.
x??

---
#### STREAM Benchmark
The STREAM benchmark evaluates memory bandwidth and cache performance by copying data through various operations like store, add, copy, scale, and swap. A sample version can be found in a Git repository.
:p What does the STREAM benchmark measure?
??x
The STREAM benchmark measures the system's memory bandwidth and cache performance using a series of basic arithmetic operations on large datasets.
x??

---
#### Random Benchmark
Random accesses data from scattered locations to evaluate random memory access performance. It is useful for applications that frequently read or write data in non-contiguous memory regions.
:p What does the Random benchmark measure?
??x
The Random benchmark measures the system's ability to perform random memory accesses, which is crucial for applications that require frequent and unpredictable data accesses.
x??

---
#### NAS Parallel Benchmarks
NAS Parallel Benchmarks are a set of NASA benchmarks first released in 1991. They are widely used in research and include some of the most heavily used benchmarks.
:p What is the significance of the NAS Parallel Benchmarks?
??x
The NAS Parallel Benchmarks are significant because they provide standardized tests for evaluating high-performance computing systems, supporting extensive use in academic and industrial research.
x??

---
#### HPCG Benchmark
HPCG is a new conjugate gradient benchmark that serves as an alternative to Linpack. It gives a more realistic performance benchmark for current algorithms and hardware configurations.
:p What distinguishes the HPCG benchmark from other benchmarks?
??x
The HPCG benchmark differs from others like Linpack by offering a more realistic representation of modern computational tasks, focusing on conjugate gradient methods that better reflect current algorithms and hardware.
x??

---
#### HPC Challenge Benchmark
The HPC Challenge is a composite benchmark that evaluates multiple aspects of high-performance computing systems, including memory bandwidth, latency, and parallelism.
:p What does the HPC Challenge benchmark evaluate?
??x
The HPC Challenge benchmark evaluates various performance aspects of high-performance computing systems, such as memory bandwidth, latency, and parallel performance.
x??

---
#### DOE Mini-Apps Overview
DOE laboratories have developed mini-apps to help hardware designers and application developers understand how best to utilize exascale computers. These include proxy apps for performance characteristics and research mini-apps for algorithmic exploration.
:p What are the main purposes of DOE mini-apps?
??x
DOE mini-apps serve two primary purposes: as proxy applications that capture the performance characteristics of larger systems, and as research tools to explore algorithms on new architectures.
x??

---
#### LULESH Proxy Application
LULESH is a proxy application for explicit Lagrangian shock hydrodynamics, widely studied by vendors and academic researchers. It represents unstructured mesh representations in explicit simulations.
:p What does the LULESH proxy application simulate?
??x
The LULESH proxy application simulates explicit Lagrangian shock hydrodynamics using an unstructured mesh representation, providing insights into complex fluid dynamics problems.
x??

---
#### MACSio Proxy Application
MACSio is a scalable I/O test mini-app that helps evaluate input/output performance in high-performance computing environments.
:p What does the MACSio proxy application focus on?
??x
The MACSio proxy application focuses on evaluating the scalability and efficiency of I/O operations in high-performance computing systems, ensuring robust data handling capabilities.
x??

---
#### ExaMiniMD Proxy Application
ExaMiniMD is a proxy application for particle and molecular dynamics codes. It helps evaluate performance characteristics related to such simulations.
:p What does the ExaMiniMD proxy application represent?
??x
The ExaMiniMD proxy application represents particle and molecular dynamics simulations, providing insights into the performance of algorithms used in these complex scientific areas.
x??

---
#### PICSARlite Proxy Application
PICSARlite is a mini-app for electromagnetic particle-in-cell simulations. It helps evaluate performance characteristics relevant to such computations.
:p What does the PICSARlite proxy application simulate?
??x
The PICSARlite proxy application simulates electromagnetic particle-in-cell systems, providing insights into the performance of algorithms used in these specific scientific computations.
x??

---

#### Valgrind Memcheck: Memory Error Detection Tool
Valgrind is an open-source memory debugging tool that can detect various types of memory errors, including uninitialized memory and memory leaks. It works well with GCC compilers and provides useful reports on issues found during execution.

:p What does Valgrind Memcheck help identify in a program?
??x
Valgrind Memcheck helps identify several types of memory errors such as out-of-bound access (fence-post checkers can catch these), uninitialized memory, and memory leaks. It is particularly effective because it offers comprehensive reports that aid developers in understanding the source of issues.

```bash
mpirun -n 4 valgrind \
--suppressions=$MPI_DIR/share/openmpi/openmpi-valgrind.supp <my_app>
```
x??

---

#### Dr. Memory for Memory Issues
Dr. Memory is another tool used to detect and debug memory errors in programs. It offers a simpler interface compared to Valgrind, making it easier to use but still very effective.

:p How does Dr. Memory help in detecting memory issues?
??x
Dr. Memory helps by identifying various types of memory issues such as uninitialized variables and memory leaks during program execution. Its ease of use makes it accessible for developers who might not have extensive experience with complex tools like Valgrind.

```bash
drmemory -- <executable_name>
```
x??

---

#### Example Code with Dr. Memory
The provided code demonstrates a simple example where the `jmax` variable is used before being initialized, leading to an uninitialized memory read error. Additionally, there is a potential memory leak issue due to un-freed allocated memory.

:p What are the issues identified by Dr. Memory in the given C code?
??x
Dr. Memory identifies two main issues:
1. Uninitialized memory: The `jmax` variable is used before being initialized.
2. Memory leaks: Dynamic memory allocated using `malloc` is not freed, leading to potential memory leaks.

```c
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int j, imax, jmax;

    // first allocate a column of pointers of type pointer to double
    double **x = (double **) malloc(jmax * sizeof(double *));
    
    // now allocate each row of data
    for (j=0; j<jmax; j++) {
        x[j] = (double *)malloc(imax * sizeof(double));
    }

    return 0;
}
```
The `jmax` variable is not initialized, and the allocated memory using `x` is never freed.

---
#### Fixing Issues with Dr. Memory
To fix the issues identified by Dr. Memory, it is necessary to initialize `jmax` and ensure all dynamically allocated memory is properly freed after use.

:p How can you modify the code to avoid uninitialized memory errors?
??x
You need to initialize the variable `jmax` before using it:

```c
int main(int argc, char *argv[]) {
    int j, imax = 10, jmax = 5; // Initialize imax and jmax

    double **x = (double **) malloc(jmax * sizeof(double *));
    
    for (j=0; j<jmax; j++) {
        x[j] = (double *)malloc(imax * sizeof(double));
    }

    // Free the allocated memory
    for (j = 0; j < jmax; j++) {
        free(x[j]);
    }
    free(x);

    return 0;
}
```
x??

---

#### Dr. Memory Report and Analysis
The report generated by Dr. Memory indicates specific issues like an uninitialized read of `jmax` on line 11, a memory leak for the variable `x`, and other potential errors.

:p What does the Dr. Memory report indicate about the code?
??x
The Dr. Memory report highlights:
- An uninitialized read error where `jmax` is used before being initialized.
- A memory leak issue because dynamically allocated memory using `x` was not freed properly.

These reports help in understanding and fixing specific issues, ensuring robust application behavior.

---
#### Running Dr. Memory on the Code
The process involves setting up Dr. Memory, downloading it from a GitHub repository, compiling the example code, and then running it to see the error reports.

:p How do you set up and run Dr. Memory for the given C code?
??x
Here are the steps to set up and run Dr. Memory on the provided C code:

1. **Clone the Repository**:
   ```bash
   git clone --recursive https://github.com/EssentialsofParallelComputing/Chapter17
   ```

2. **Navigate to the Directory**:
   ```bash
   cd Chapter17
   ```

3. **Build the Example Code**:
   ```bash
   make
   ```

4. **Run Dr. Memory**:
   ```bash
   drmemory -- memoryexample
   ```

This process will produce a detailed report that helps in identifying and fixing memory issues.

x??

