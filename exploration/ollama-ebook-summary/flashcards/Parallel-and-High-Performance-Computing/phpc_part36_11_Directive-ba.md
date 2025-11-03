# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 36)

**Starting Chapter:** 11 Directive-based GPU programming

---

#### Parallelism on GPUs vs CPUs
Background context explaining the parallelism requirements for GPU and CPU. The GPU needs thousands of independent work items to utilize its architecture effectively, while the CPU only requires tens.

:p How does parallelism differ between GPU and CPU?
??x
The GPU is designed with a large number of cores that can perform computations in parallel, ideally in the thousands. This aligns well with data-parallel tasks where the same operation needs to be applied to many elements simultaneously, such as image processing or matrix operations. In contrast, CPUs have fewer cores but are optimized for sequential execution and handling complex control flow.

```java
// Example of a simple loop that could benefit from parallelism on GPU
for (int i = 0; i < largeArray.length; i++) {
    // Process each element in the array
}
```
x??

---

#### OneAPI Toolkit
Background context about oneAPI toolkit, including its components like Intel GPU driver, compilers, and tools. The provided link offers more information.

:p What is oneAPI toolkit?
??x
OneAPI toolkit is a suite of software development tools designed for programming GPUs from Intel and other devices. It includes drivers, compilers, and tools that support various GPU architectures. You can download it from the provided URL: <https://software.intel.com/oneapi>.

```c
// Example initialization code to use oneAPI components in C
#include "oneapi/dpl/vector"
int main() {
    using namespace oneapi::dpl;
    // Use oneAPI libraries for GPU programming
}
```
x??

---

#### GPU vs CPU Processing Time Calculation
Background context involving an image classification application with given processing times and number of images.

:p In problem 1, would a GPU system be faster than the CPU?
??x
To determine if the GPU system is faster, we need to compare the total time taken by both systems. For the GPU:

- Transfer: 5 ms per file * 2 (to and from) = 10 ms
- Process: 5 ms
- Total = 10 + 5 = 15 ms

For the CPU:
- Process: 100 ms per image * 1,000,000 images = 100,000,000 ms or 11.57 days (assuming it can handle all 1 million images in parallel)
- Transfer time is negligible here as the CPU handles all processing locally.

Since the GPU system processes the data much faster than the CPU system over a large dataset, it would be significantly faster.

```java
// Pseudocode to illustrate comparison
public void processImages(int[] images) {
    long startTime = System.currentTimeMillis();
    
    // Assume transfer and processing on GPU are done here
    for (int i = 0; i < images.length; i++) {
        // Process each image on the GPU
    }
    
    long endTime = System.currentTimeMillis();
    long gpuTime = endTime - startTime;
    
    // For CPU, just measure process time
    long cpuTime = 100 * images.length;
    
    if (gpuTime < cpuTime) {
        System.out.println("GPU is faster");
    } else {
        System.out.println("CPU is faster or comparable");
    }
}
```
x??

---

#### PCI Bus Transfer Impact
Background context on the impact of different generations of PCI bus transfer speeds.

:p How does changing the PCI bus generation affect GPU performance?
??x
Changing the PCI bus generation can significantly reduce transfer times, impacting overall processing time. For example, a Gen4 PCI bus would have faster data transfer rates compared to a third-generation bus, potentially reducing transfer times by 2-3 times or more.

With the Gen5 PCI bus:
- Transfer time could be even lower, further improving performance.
- The impact is significant if large amounts of data need to be transferred repeatedly between GPU and CPU.

```java
// Pseudocode for comparing different bus speeds
public void measureTransferTime(int[] images) {
    long startTime = System.currentTimeMillis();
    
    // Measure transfer time on third-gen PCI
    transfer(images);
    long gpuTimeThirdGen = System.currentTimeMillis() - startTime;
    
    // Reset start time and switch to Gen4/Gen5
    startTime = System.currentTimeMillis();
    transfer(images);  // Assume optimized transfer with faster bus
    long gpuTimeGen4 = System.currentTimeMillis() - startTime;
    
    if (gpuTimeGen4 < gpuTimeThirdGen) {
        System.out.println("Gen4 PCI improves performance");
    } else {
        System.out.println("Third-gen still faster or comparable");
    }
}
```
x??

---

#### 3D Application on Discrete GPU
Background context about running a 3D application with specific memory constraints.

:p What size 3D application can be run on a discrete GPU?
??x
To determine the size of a 3D application, consider the GPU's memory capacity and how much is needed per cell. For example, if you need to store 4 double-precision variables per cell:

- Assume half of the GPU memory for temporary arrays.
- Calculate the total number of cells that can be stored.

For single precision:
- Double the capacity as single precision requires less memory per variable.

```java
// Pseudocode to calculate application size on a discrete GPU
public int getMaxCells(int gpuMemory, double variablesPerCell) {
    // Half of the memory is used for temporary arrays
    int usableMemory = (int)(gpuMemory * 0.5);
    
    // Calculate maximum number of cells based on available memory and precision
    if ("double") {
        return usableMemory / (4 * variablesPerCell);
    } else if ("single") {
        return usableMemory / (2 * variablesPerCell);
    }
}
```
x??

---

#### OpenMP vs. OpenACC
Background context about the history and current status of both directive-based languages.

:p What is the history of OpenMP and OpenACC?
??x
OpenMP was first released in 1997 to simplify parallel programming on CPUs. However, it focused more on new CPU capabilities initially. To address GPU accessibility, a group of compiler vendors (Cray, PGI, CAPS, and NVIDIA) joined forces in 2011 to release the OpenACC standard. This provided an easier pathway for GPU programming using pragmas.

OpenMP has since added its own support for GPUs through the OpenMP Architecture Review Board (ARB). The OpenACC standard is now more mature with broader compiler support, while OpenMP is gaining traction as a long-term solution.

```java
// Example of an OpenMP directive
public void parallelLoop(int[] array) {
    #pragma omp parallel for
    for (int i = 0; i < array.length; i++) {
        // Parallel loop body
    }
}
```
x??

---

#### Directive-Based GPU Programming with Examples
Background context on using pragmas or directives to port code to GPUs.

:p What are the key concepts of directive-based GPU programming?
??x
Directive-based GPU programming uses pragmas to direct the compiler to generate GPU code. This approach simplifies the process by allowing developers to add annotations directly in their source code, making it easier to exploit parallelism on both CPUs and GPUs.

Key concepts include:
- Separation of computational loop body from index set.
- Using pragmas for memory allocation and kernel invocation.
- Asynchronous work queues to overlap communication and computation.

Here’s an example using OpenACC:

```c
// Example OpenACC code snippet
#include <openacc.h>

void processArray(double* data, int size) {
    #pragma acc kernels copyin(data[0:size]) async(1)
    for (int i = 0; i < size; i++) {
        // Process each element in the array
        data[i] *= 2;
    }
}
```
x??

---

#### OpenACC and OpenMP Overview
Background context: The text discusses how to use directives and pragmas (specifically, OpenACC) to offload work to a GPU. This allows developers to leverage GPU power without significant changes to their application code.

:p What are OpenACC and OpenMP used for in GPU programming?
??x
OpenACC and OpenMP are used as directive-based languages to allow users to offload computationally intensive tasks from the CPU to the GPU, thereby utilizing the parallel processing capabilities of GPUs. This is achieved by adding specific directives or pragmas within the application code.

For example:
```c
#pragma acc kernels
void compute(int *data) {
    // Compute kernel logic here
}
```
x??

---

#### Steps for Implementing OpenACC on a GPU

:p What are the three steps to implement a GPU port using OpenACC?
??x
1. Move the computationally intensive work to the GPU, which necessitates data transfers between CPU and GPU.
2. Reduce the data movement between the CPU and GPU by optimizing memory usage.
3. Tune the size of the workgroup, number of workgroups, and other kernel parameters to enhance performance.

For example:
```c
#pragma acc kernels copyin(a) copyout(b)
void compute(int *a, int *b) {
    // Computation logic here
}
```
x??

---

#### Offloading Work to a GPU

:p How does offloading work to the GPU affect data transfers and application performance?
??x
Offloading work to the GPU causes data transfers between the CPU and GPU, which can initially slow down the application. However, this is necessary because the GPU can process tasks much faster than the CPU.

To reduce the impact of these data transfers, optimize memory usage by allocating data on the GPU if it will only be used there.
x??

---

#### Example Makefiles for Compiling OpenACC Code

:p What are some flags and configurations needed to compile an OpenACC code using PGI or GCC compilers?
??x
For PGI Compiler:
```makefile
CFLAGS:= -g -O3 -c99 -alias=ansi -Mpreprocess -acc -Mcuda -Minfo=accel
```

For GCC Compiler:
```makefile
CFLAGS:= -g -O3 -std=c99 -fopenacc -Xpreprocessor -DACC_OFFLOAD
```
These flags are set to provide detailed compiler information, optimize the code, and ensure proper handling of OpenACC directives.

Make sure to use the most recent version of GCC as it supports more features.
x??

---

#### OpenACC Compilers Overview

:p Which compilers support OpenACC and which are notable?
??x
Several compilers support OpenACC:
- PGI (commercial with community edition available for free)
- GCC (versions 7, 8 implement 2.0a; version 9 implements 2.5; development branch supports 2.6)
- Cray (commercial, only available on Cray systems)

PGI is the most mature and widely used option.
x??

---

#### Running Examples Without a GPU

:p What happens if you don't have an appropriate GPU but still want to try OpenACC examples?
??x
If you do not have an appropriate GPU, you can still run OpenACC examples on your CPU. The performance will be different from running it on a GPU, but the basic code should remain functional.

For PGI Compiler:
```bash
pgaccelinfo
```
This command provides information about the system and whether it is set up correctly for OpenACC.
x??

---

---
#### OpenACC Compilation Flags for GCC
Background context explaining the compilation flags required to enable OpenACC support on GCC. The `CFLAGS` variable is defined with various options, including `-g`, `-O3`, and `-std=gnu99`. These flags optimize and debug the code but also need specific flags like `-fopenacc` to parse OpenACC directives.

The `fopt-info-optimized-omp` flag enables detailed feedback on optimized code for debugging purposes. The `fstrict-aliasing` and `foptimize-sibling-calls` options can improve performance by allowing better optimization of function calls, but they can also increase the complexity of code generation due to stricter aliasing rules.

:p What are the GCC compilation flags required to enable OpenACC support?
??x
To enable OpenACC on GCC, you need to include specific compiler flags in your `CFLAGS`. The essential flags are:
```makefile
CFLAGS:= -g -O3 -std=gnu99 -fstrict-aliasing -fopenacc \                               -fopt-info-optimized-omp
```
These flags optimize and debug the code while enabling OpenACC directives. The `-g` flag includes debugging symbols, `-O3` enables aggressive optimizations, and `-std=gnu99` sets the C standard to GNU99.

The `-fopenacc` flag is crucial as it activates the parsing of OpenACC directives for GCC. The `fopt-info-optimized-omp` flag provides detailed feedback on optimized code.
x??

---
#### OpenACC Compilation Flags for PGI
Background context explaining that different compilers require specific flags to enable OpenACC support, and for PGI (Portland Group), the relevant flags are `-acc -Mcuda`. These flags tell the compiler to generate CUDA-compatible code.

The `Minfo=accel` flag provides detailed feedback on the use of accelerator directives. The `alias=ansi` flag allows less strict pointer aliasing checks, which can enable more aggressive optimizations but may require careful handling by developers.

:p What are the PGI compilation flags for OpenACC?
??x
For PGI (Portland Group), you need to include specific compiler flags in your `CFLAGS`. These are:
```makefile
CFLAGS:= -acc -Mcuda \Minfo=accel -alias=ansi
```
The `-acc` flag enables OpenACC support, while the `-Mcuda` option generates CUDA-compatible code. The `Minfo=accel` flag gives detailed feedback on the use of accelerator directives, which is useful for debugging and optimization purposes. The `alias=ansi` flag allows the compiler to make less strict aliasing checks, enabling more aggressive optimizations but requiring careful handling by developers.
x??

---
#### OpenACC Compilation Flags for Cray
Background context explaining that the Cray compiler has OpenACC support enabled by default. However, you can disable it using `-hnoacc`. Additionally, the `_OPENACC` macro is important as it indicates which version of OpenACC your compiler supports.

You can check the version by comparing `_OPENACC == yyyymm`, where `yyyymm` represents the date of implementation for each version (e.g., 201111 for Version 1.0).

:p What are the Cray-specific compilation flags related to OpenACC?
??x
For Cray, OpenACC is enabled by default, but you can disable it using the `-hnoacc` flag if necessary. Additionally, the `_OPENACC` macro is crucial as it indicates which version of OpenACC your compiler supports. You can check the version by comparing `_OPENACC == yyyymm`, where `yyyymm` represents the date of implementation for each version.

For example:
```makefile
# To disable OpenACC on Cray
CFLAGS:= -hnoacc

# To check the version of OpenACC support
ifneq ($(findstring 201111,$(_OPENACC)),)
    # Version 1.0 is supported
endif
```
x??

---
#### `kernels` Pragma for Compiler Autoparallelization
Background context explaining that the `kernels` pragma allows auto-parallelization of a code block by the compiler, often used to get feedback on sections of code.

The formal syntax includes optional clauses like `data clause`, `kernel optimization`, `async clause`, and `conditional`. The `data clause` is used for specifying data movement between host and device. The `kernel optimization` allows specifying details such as the number of threads or vector length.

:p What is the purpose of the `kernels` pragma in OpenACC?
??x
The purpose of the `kernels` pragma in OpenACC is to allow auto-parallelization of a code block by the compiler, often used to get feedback on sections of code. The formal syntax for the `kernels` pragma from the OpenACC 2.6 standard is:
```c
#pragma acc kernels [ data clause | kernel optimization | async clause | conditional ]
```
Where:
- **Data Clauses** - `copy`, `copyin`, `copyout`, `create`, `no_create`, `present`, `deviceptr`, `attach`, and `default(none|present)`
- **Kernel Optimizations** - `num_gangs`, `num_workers`, `vector_length`, `device_type`, `self`
- **Async Clauses** - `async`, `wait`
- **Conditional** - `if`

For example, to use the `kernels` pragma with a data clause and kernel optimization:
```c
#pragma acc kernels copyin(a) num_gangs(16)
{
    // Code block to be parallelized
}
```
x??

---

---
#### OpenACC `kernels` Directive Overview
OpenACC provides directives to specify which parts of a program should be executed on the GPU. The `#pragma acc kernels` directive is used to parallelize loops and other code blocks, allowing automatic distribution of work among multiple processing units.

In this example, the compiler is using `#pragma acc kernels` around for-loops to indicate that these sections can potentially be offloaded to the GPU.
:p What does the `#pragma acc kernels` directive do?
??x
The `#pragma acc kernels` directive indicates to the OpenACC runtime system and compiler that the enclosed code should be executed on a GPU if available. The compiler will attempt to parallelize the loop or block of code within this directive, distributing tasks among multiple cores on the GPU for improved performance.
```c
#pragma acc kernels
for (int i=0; i<nsize; i++) {
    c[i] = a[i] + scalar*b[i];
}
```
x?
---

#### Compiler Feedback and Parallelization Challenges
The compiler's feedback provides insights into its decision-making process regarding the parallelization of code. In this example, several loop carried dependences are noted as reasons why some loops cannot be parallelized automatically.

Loop carry depends occur when a value written in one iteration affects the calculation in subsequent iterations.
:p According to the compiler output, what issues prevent automatic parallelization?
??x
The compiler feedback indicates that there are complex loop-carried dependencies between arrays `a`, `b`, and `c`. Specifically:
- Loop carried dependence of `a` and `b` prevents parallelization.
- Loop carried backward dependence of `c` also impacts vectorization.

These dependencies imply that the values from one iteration depend on results computed in previous iterations, making it difficult for the compiler to parallelize the loops automatically without additional directives or manual intervention.
```c
// Example problematic loop
#pragma acc kernels
for (int i=0; i<nsize; i++) {
    c[i] = a[i] + scalar * b[i];
}
```
x?
---

#### Implicit Copy Operations and Data Management
OpenACC generates implicit copy operations to transfer data between host memory and device memory. These copies are necessary for arrays that need to be shared between the CPU and GPU.

In this example, `#pragma acc kernels` implicitly adds copy operations for arrays `a`, `b`, and `c`.
:p What are implicit copy operations in OpenACC?
??x
Implicit copy operations in OpenACC automatically handle data transfer between host (CPU) memory and device (GPU) memory. When the `#pragma acc kernels` directive is used, the compiler adds necessary copy operations to ensure that arrays and other variables are correctly transferred.

For instance:
- `Generating implicit copyout(b[:20000000],a[:20000000])`: This indicates data will be copied from host memory (CPU) to device memory (GPU).
- `Generating implicit copyin(b[:20000000],a[:20000000])`: This indicates data will be copied from device memory (GPU) back to host memory (CPU).

These operations are managed automatically by the compiler, simplifying the process of moving data between different memory spaces.
```c
// Example implicit copy
#pragma acc kernels
for (int i=0; i<nsize; i++) {
    c[i] = a[i] + scalar * b[i];
}
```
x?
---

#### Compiler’s Decision on Loop Parallelization
The compiler's output provides information about which loops it can or cannot parallelize based on the current code and data dependencies.

In this example, the loop in the `stream triad` is marked as serial (`#pragma acc loop seq`), indicating that it could not be parallelized.
:p Based on the compiler feedback, what does "Complex loop carried dependence of a->,b-> prevents parallelization" mean?
??x
This message indicates that there are complex data dependencies between arrays `a` and `b` within the loop. These dependencies make it difficult for the compiler to determine how to safely distribute the work among multiple threads or processing units on the GPU.

In other words, if modifying an element in array `a` or `b` affects a future iteration of the same loop, parallelizing this loop could lead to incorrect results due to data races and inconsistent state. Therefore, the compiler decides not to parallelize it to avoid potential issues.
```c
// Example problematic loop marked as serial
#pragma acc kernels
for (int i=0; i<nsize; i++) {
    c[i] = a[i] + scalar * b[i];
}
```
x?
---

---
#### Restrict Attribute Usage
In the context of OpenACC programming, adding a `restrict` attribute to pointers is essential for the compiler to optimize memory access. The `restrict` keyword indicates that a pointer points exclusively to one piece of data and that no other pointer will modify this data through the same address.
:p What does the `restrict` attribute do in C/C++?
??x
The `restrict` attribute helps the compiler understand that different pointers point to non-overlapping memory regions, which can lead to better optimization by allowing the compiler to make certain assumptions about memory access patterns. This is particularly useful when working with parallel code to avoid false sharing issues.
```c
double* restrict a = malloc(nsize * sizeof(double));
double* restrict b = malloc(nsize * sizeof(double));
double* restrict c = malloc(nsize * sizeof(double));
```
x??

---
#### OpenACC Loop Directives
OpenACC provides various loop directives to guide the compiler on how to parallelize and optimize loops. The `loop` directive can be specified with different clauses, such as `auto`, `independent`, or `seq`.
:p What are the possible values for the `loop` directive in OpenACC?
??x
The `loop` directive in OpenACC can take several values including:
- **auto**: Let the compiler analyze and decide on parallelization.
- **independent**: Indicate that the loop iterations can be executed independently of each other, allowing parallel execution.
- **seq**: Explicitly indicate that the loop should not be parallelized.

Example usage:
```cpp
#pragma acc kernels loop independent
```
x??

---
#### Parallel Loop Pragma in OpenACC
The `parallel` and `loop` pragmas together allow for more fine-grained control over parallelization. The `parallel` pragma opens a parallel region, while the `loop` pragma distributes work within that region.
:p What is the purpose of using the `parallel loop` pragma in OpenACC?
??x
Using the `parallel loop` pragma gives developers more explicit control over parallelization. It allows you to specify exactly which loops should be executed in parallel and can provide additional directives for optimization, such as gang or vector instructions.

Example usage:
```cpp
#pragma acc parallel loop independent
```
x??

---
#### Compiler Output Analysis
The compiler output shows feedback about data transfers between host and GPU memory. These messages help identify potential optimizations needed to improve performance.
:p What does the compiler feedback in bold signify in the OpenACC code?
??x
The bolded feedback in the compiler output indicates important information such as data transfers between host and device memory. For example, messages like "Generating implicit copyout" or "Loop is parallelizable" provide insights into the optimization process.

Example:
```plaintext
main:      15, Generating implicit copyout(a[:20000000],b[:20000000]) [if not already present]
16, Loop is parallelizable          Generating Tesla code
```
x??

---

---
#### Parallel Loop Construct Overview
The parallel loop construct is used to indicate that a loop should be executed in parallel on the available hardware. Unlike some other constructs, it uses the independent clause by default, meaning each iteration can be processed independently of others.

:p What is the default behavior for the parallel loop directive?
??x
The default behavior for the parallel loop directive is to use the `independent` clause, allowing iterations to run in parallel without requiring explicit dependencies. This contrasts with other constructs that might have different defaults.
x??

---
#### OpenACC Directive Syntax for Parallel Loops
The syntax for using a parallel loop construct with OpenACC includes specifying the data regions and any additional clauses such as `gang`, `vector`, etc., which define how the iterations are grouped and executed.

:p How does the OpenACC directive for a parallel loop look in code?
??x
Here's an example of how to use the OpenACC parallel loop directive:

```c
#pragma acc parallel loop
for (int i=0; i<nsize; i++) {
   // Loop body
}
```

The `parallel loop` directive indicates that the loop should be executed in parallel. The compiler can add additional optimizations like vectorization or gang execution based on the hardware and data dependencies.
x??

---
#### Example of Using Parallel Loops in Stream Triad

:p How is a parallel loop added to the stream triad example?
??x
In the provided code, a parallel loop is inserted into the kernel that performs the stream triad operation. The `parallel loop` directive is used twice: once for initializing arrays and again for performing the actual computation.

Here's how it looks in the code:

```c
#pragma acc parallel loop
for (int i=0; i<nsize; i++) {
   a[i] = 1.0;
   b[i] = 2.0;
}

// Stream triad loop
#pragma acc parallel loop
for (int i=0; i<nsize; i++) {
   c[i] = a[i] + scalar*b[i];
}
```

The compiler outputs show that the loops are being parallelized and optimized for execution on the GPU.
x??

---
#### Reducing Data Movement with Parallel Loops

:p How does adding a reduction clause to a parallel loop affect data movement?
??x
Adding a reduction clause to a parallel loop can help reduce unnecessary data movement. The `reduction(+:summer)` clause ensures that the accumulation of `summer` is done correctly across iterations, reducing the need for frequent data transfers.

```c
#pragma acc parallel loop reduction(+:summer)
for (int ic=0; ic<ncells ; ic++) {
   if (celltype[ic] == REAL_CELL) {
      summer += H[ic]*dx[ic]*dy[ic];
   }
}
```

This approach helps in optimizing the data movement, making the code more efficient.
x??

---
#### Explanation of Compiler Output for Parallel Loops

:p What does the compiler output indicate about parallel loop optimization?
??x
The compiler output indicates that the loops are being optimized for execution on the GPU. The `#pragma acc` directives show how iterations are being grouped (gang) and vectorized.

For example, in the stream triad code:
```
15, #pragma acc loop gang, vector(128)
24, #pragma acc loop gang, vector(128)
```

These lines indicate that the loops are being parallelized using a combination of `gang` and `vector` groups. The compiler also handles implicit data movement:
```c
Generating implicit copyout(a[:20000000],b[:20000000])
Generating implicit copyin(b[:20000000],a[:20000000])
```

These lines show that the compiler is managing data movement between host and device.
x??

---
#### Performance Impact of Data Movement

:p How does data movement impact the performance of parallelized code?
??x
Data movement can significantly slow down the performance of parallelized code, especially when large amounts of data need to be transferred between the CPU and GPU. The compiler-generated code attempts to optimize this by reducing unnecessary transfers using constructs like reductions.

For example:
```c
#pragma acc parallel loop reduction(+:summer)
```

This construct helps in managing shared variables efficiently, thereby reducing the overhead of frequent data movement.
x??

---

