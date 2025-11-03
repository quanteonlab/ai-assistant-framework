# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 19)

**Starting Chapter:** 7.3.1 Loop level OpenMP Vector addition example

---

#### Loop-Level OpenMP for Simple Speedup
Background context: This section explains how loop-level OpenMP is used when modest speedup is needed. It involves placing `parallel for` pragmas or `parallel do` directives before key loops to introduce parallelism with minimal effort. The goal here is not to achieve maximum performance but rather a quick way to parallelize existing code.

:p What is the primary purpose of using loop-level OpenMP in this scenario?
??x
The primary purpose of using loop-level OpenMP when modest speedup is needed is to quickly and easily introduce parallelism into the application by placing `parallel for` pragmas or `parallel do` directives before key loops. This approach minimizes effort compared to more complex high-level strategies.

```cpp
// Example C code
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    // loop body
}
```
x??

---

#### High-Level OpenMP for Better Parallel Performance
Background context: This section introduces a different approach to OpenMP, called high-level OpenMP, which focuses on the entire system rather than just loop parallelism. It aims at achieving better performance by considering memory systems, kernels, and hardware in a top-down design strategy.

:p What is the main difference between standard loop-level OpenMP and high-level OpenMP?
??x
The main difference between standard loop-level OpenMP and high-level OpenMP lies in their approach to parallelism. Standard OpenMP starts from the bottom-up by applying parallel constructs at the loop level, whereas high-level OpenMP takes a top-down design strategy that considers the entire system, including memory systems, kernels, and hardware.

```cpp
// Example C code (high-level)
// This is more about system-wide optimization rather than a specific loop
#pragma omp parallel 
{
    // System-wide parallelization logic
}
```
x??

---

#### MPI + OpenMP for Extreme Scalability
Background context: This section discusses combining OpenMP with Message Passing Interface (MPI) to achieve extreme scalability in applications. It mentions the use of threading within one memory region, commonly a Non-Uniform Memory Access (NUMA) region, and how this can be used to supplement distributed memory parallelism.

:p How does using OpenMP on a small subset of processes help in achieving extreme scalability?
??x
Using OpenMP on a small subset of processes helps achieve extreme scalability by adding another level of parallel implementation within the node or NUMA region. This approach restricts threading to regions where all memory accesses have the same cost, thereby avoiding some complexity and performance traps associated with OpenMP.

```cpp
// Example C code (MPI + OpenMP)
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
if (rank % 2 == 0) { // Only even ranks use OpenMP
    #pragma omp parallel
    {
        // OpenMP parallel region
    }
}
```
x??

---

#### Standard Loop-Level OpenMP Examples
Background context: This section provides examples of using standard loop-level OpenMP to introduce parallelism in applications. It is a starting point for learning the basics and can be used as a foundation before moving on to more complex high-level strategies.

:p What are some key benefits of using standard loop-level OpenMP?
??x
Key benefits of using standard loop-level OpenMP include:
- Quick and easy introduction of parallelism with minimal effort.
- Reduces thread race conditions compared to other approaches.
- Acts as the first step when introducing thread parallelism to an application.

```cpp
// Example C code (loop-level)
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    // loop body
}
```
x??

---

#### High-Level OpenMP Implementation Model and Methodology
Background context: This section delves into the methodology of high-level OpenMP, which is designed to extract maximum performance by considering the entire system. It outlines a step-by-step method for achieving better scalability compared to loop-level parallelism.

:p What are some key steps in implementing high-level OpenMP?
??x
Key steps in implementing high-level OpenMP include:
1. Understanding and optimizing memory systems.
2. Addressing kernel performance issues.
3. Considering hardware-specific optimizations.
4. Using `parallel` pragmas at a higher level to manage threads and resources more efficiently.

```cpp
// Example C code (high-level)
#pragma omp parallel
{
    // System-wide optimization logic
}
```
x??

---

#### Hybrid MPI + OpenMP for Modest Thread Counts
Background context: This section discusses using OpenMP in conjunction with MPI, particularly focusing on hybrid implementations where threading is used within one memory region. It mentions the use of two-to-four hyperthreads per processor to enhance performance.

:p What are some key benefits of a hybrid MPI + OpenMP implementation?
??x
Key benefits of a hybrid MPI + OpenMP implementation include:
- Enhancing performance by using OpenMP on small thread counts.
- Avoiding complex memory access patterns and performance traps.
- Utilizing NUMA regions for efficient memory access.

```cpp
// Example C code (MPI + OpenMP)
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
if (rank % 2 == 0) { // Only even ranks use OpenMP
    #pragma omp parallel
    {
        // OpenMP parallel region
    }
}
```
x??

---
#### Parallel Regions and Pragmas
Parallel regions are initiated by inserting pragmas around blocks of code that can be divided among independent threads (e.g., do loops, for loops). OpenMP relies on the OS kernel for its memory handling.

:p What is a parallel region in OpenMP?
??x
A parallel region in OpenMP refers to a block of code enclosed within specific pragmas, such as `#pragma omp parallel`, that allows multiple threads to execute simultaneously. These regions are used to distribute work among threads, enabling parallel execution.
```c
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    // thread-specific code
}
```
x??

---
#### Shared and Private Variables in OpenMP
Each variable within a parallel construct can be either shared or private. Shared variables are accessible to all threads, while private variables have unique copies per thread.

:p How does the scope of a variable affect its behavior in an OpenMP parallel region?
??x
In an OpenMP parallel region, the scope of a variable determines whether it is shared among all threads (shared) or has individual copies for each thread (private). Shared variables allow multiple threads to access and modify them, while private variables ensure that each thread operates on its own copy.
```c
// Example with private scope
#pragma omp parallel for private(i)
for (int i = 0; i < n; ++i) {
    // thread-specific code
}

// Example with shared scope
#pragma omp parallel for shared(a, b)
for (int i = 0; i < n; ++i) {
    a[i] += b[i];
}
```
x??

---
#### Memory Model and Synchronization in OpenMP
OpenMP has a relaxed memory model. Each thread has its own temporary view of memory to avoid overhead from frequent synchronization with main memory. This requires explicit barriers or flush operations for synchronization.

:p What is the purpose of barriers and flushes in OpenMP?
??x
Barriers and flushes in OpenMP are used to synchronize threads when they need to reconcile their local views of memory with the global state. Barriers ensure that all threads reach a checkpoint before proceeding, while flushes force memory updates between threads. These operations help maintain data consistency but introduce overhead.
```c
// Example using barrier
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    // thread-specific code
}
#pragma omp barrier

// Example using flush operation
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    // thread-specific code
}
#pragma omp flush(i)
```
x??

---
#### Vector Addition Example
The vector addition example demonstrates the use of OpenMP work-sharing directives, implied variable scope, and memory placement by the OS. This helps in understanding the interaction between these components.

:p What does the vector addition example illustrate?
??x
The vector addition example illustrates how to parallelize a simple operation using OpenMP pragmas. It shows how `#pragma omp parallel for` is used to distribute iterations of a loop among threads, and it highlights the importance of variable scoping (private vs. shared) in ensuring correct execution.
```c
#include <omp.h>
void vector_add(double *a, double *b, double *result, int n) {
    #pragma omp parallel for private(i)
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}
```
x??

---

#### Vector Addition Optimization: First Touch Concept
Background context: In vector addition, optimizing memory allocation can significantly improve performance. The "first touch" concept involves initializing arrays such that they are allocated close to the threads that will work with them, minimizing memory access latency.

:p What is the first touch optimization in the context of vector addition?
??x
The first touch optimization ensures that array elements are allocated near the thread that will be working with them during their first access. This placement can reduce memory access time and improve parallel performance by aligning data access patterns more closely with execution threads.
??x

---

#### Vector Addition Optimization: Code Example
Code example in C demonstrating the concept of initializing arrays within a parallel loop.

:p How does the following code ensure efficient first touch for array initialization?
```c
#include <omp.h>
#include "timer.h"

#define ARRAY_SIZE 80000000

int main(int argc, char *argv[]) {
    #pragma omp parallel >> Spawn threads >>
       if (omp_get_thread_num() == 0)
          printf("Running with %d thread(s)", omp_get_num_threads());
       Implied Barrier
       Implied Barrier

    struct timespec tstart;
    double time_sum = 0.0;

    #pragma omp parallel for
    for (int i=0; i<ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }
       Implied Barrier
       Implied Barrier

    // Perform vector addition
    vector_add(c, a, b, ARRAY_SIZE);

    cpu_timer_start(&tstart);
    time_sum += cpu_timer_stop(tstart);

    printf("Runtime is %lf msecs", time_sum);
}
```
??x
In the provided code, the `a` and `b` arrays are initialized within the parallel loop using `#pragma omp parallel for`. This ensures that each thread initializes its segment of the array during the first access. The operating system will likely allocate these segments of memory near the threads, thus reducing the latency when they later read from or write to these arrays.

The `vector_add` function then performs the vector addition operation, and due to the efficient allocation strategy, the memory for `a` and `b` is close to the executing thread.
??x

---

#### Vector Addition: Memory Allocation in Main vs. Parallel Loops
Background context: In the initial implementation of vector addition, array elements are first touched by the main thread during initialization before being used in parallel computations. This can lead to suboptimal memory allocation and increased latency.

:p How does moving the array initialization inside a `#pragma omp parallel for` affect memory allocation?
??x
By placing the array initialization within a `#pragma omp parallel for`, each thread initializes its segment of the array during the first access. The operating system will allocate this memory close to the executing thread, reducing memory latency.

In contrast, in the initial code (Listing 7.7), the main thread allocates all elements before any parallel computation starts, leading to possible suboptimal memory placement.
??x

---

#### Vector Addition: Improved Performance Through First Touch
Background context: The first touch optimization aims to allocate array data close to the threads that will be working with them during their first access. This reduces memory latency and improves performance.

:p What is the primary benefit of using `#pragma omp parallel for` for array initialization in vector addition?
??x
The primary benefit of using `#pragma omp parallel for` for array initialization in vector addition is to ensure that each thread allocates its segment of the array during the first access. This allows the operating system to place this memory closer to the executing threads, reducing memory latency and improving overall performance.

By placing the data close to where it will be used, this optimization can lead to better cache utilization and reduced contention for shared resources.
??x

---

#### Vector Addition: Difference Between Initial and Optimized Code
Background context: The initial code initializes arrays in the main thread before parallel execution. The optimized code distributes initialization within a `#pragma omp parallel for` loop.

:p How does the placement of array initialization affect memory allocation and performance?
??x
Placing array initialization inside a `#pragma omp parallel for` loop affects memory allocation by ensuring that each thread allocates its segment during the first access. This allows the operating system to place this memory closer to the executing threads, reducing memory latency.

In contrast, initializing arrays in the main thread before any parallel execution can lead to less optimal memory placement, as all elements are allocated together and may not be close to the threads that will use them.
??x

#### OpenMP Vector Addition Example
Background context: This example demonstrates how to use OpenMP to parallelize a vector addition operation, highlighting the performance improvement through thread spawning and loop distribution.

:p What is the purpose of using `#pragma omp parallel for` in the vector_add function?
??x
The purpose of using `#pragma omp parallel for` is to distribute the iterations of the loop among multiple threads. This directive allows OpenMP to automatically spawn a team of threads, each executing one or more iterations of the loop based on the number of available threads.

C/Java code example:
```c
void vector_add(double *c, double *a, double *b, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i]; // Each thread will compute one element of the result array.
    }
}
```

---
#### NUMA and Memory Access Costs
Background context: Non-Uniform Memory Access (NUMA) refers to computer systems in which memory access times depend on where data is located relative to the processing unit. In a NUMA system, threads that are assigned to specific nodes have faster access to local memory.

:p What does the `numactl` and `numastat` commands reveal about the Skylake Gold 6152 CPU?
??x
The `numactl` and `numastat` commands provide information about NUMA configuration, including memory distance metrics. For the Skylake Gold 6152 CPU, these commands can show that accessing remote memory incurs a performance penalty of approximately a factor of two compared to local memory.

Example output from `numastat`:
```
Node distances:
   from node 0   to node 0:   10 ms
   from node 0   to node 1:   21 ms
```

:p How can NUMA configuration affect performance on the Skylake Gold 6152 CPU?
??x
NUMA configuration significantly impacts performance because accessing remote memory is slower than local memory. For example, a factor of two decrease in performance when accessing remote memory compared to local memory on the Skylake Gold 6152 CPU highlights that applications should optimize for first touch policies and preferentially use local resources.

---
#### Stream Triad Benchmark Example
Background context: The Stream Triad benchmark is used to measure the performance of vector operations. This example demonstrates how OpenMP can be used to parallelize the triad operation, which involves three sequential vector operations (copy, add, scale).

:p What are the key differences between the vector addition and stream triad examples?
??x
The key differences lie in the complexity of the operations performed and the number of iterations. The vector addition example focuses on a simple element-wise addition of two vectors stored in arrays `a` and `b`, with results stored in array `c`. In contrast, the stream triad involves multiple iterations and additional operations like scaling.

C/Java code example:
```c
int main(int argc, char *argv[]) {
    #pragma omp parallel if (omp_get_thread_num() == 0)
    printf("Running with %d thread(s)\n", omp_get_num_threads());

    struct timeval tstart;
    double scalar = 3.0, time_sum = 0.0;

    #pragma omp parallel for
    for (int i = 0; i < STREAM_ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }
}
```

:p How does the OpenMP directive `#pragma omp parallel` work in the main function of the stream triad example?
??x
The `#pragma omp parallel` directive creates a team of threads to execute the enclosed code. The thread that is responsible for the initial execution (i.e., `omp_get_thread_num() == 0`) prints out the number of threads, and then each thread executes the loop defined after the directive.

```c
int main(int argc, char *argv[]) {
    #pragma omp parallel if (omp_get_thread_num() == 0)
    printf("Running with %d thread(s)\n", omp_get_num_threads());
```

---
#### NUMA First Touch Optimization
Background context: First touch optimization is a technique where the initial access to memory by threads is designed to minimize remote memory accesses. This can significantly improve performance in NUMA systems.

:p How does first touch optimization help in NUMA configurations?
??x
First touch optimization helps by ensuring that each thread initializes its local data before accessing it, thus reducing the chances of remote memory requests and improving overall performance. For example, in a NUMA system, initializing arrays on their respective nodes can prevent threads from having to access slower remote memory.

:p What is the significance of the `time_sum` variable in the provided vector addition code?
??x
The `time_sum` variable accumulates the total execution time across multiple iterations or runs. It helps in measuring and comparing performance before and after optimizations, such as first touch optimizations.

```c
double scalar = 3.0, time_sum = 0.0;
```

---
#### NUMA Node Terminology
Background context: In the provided text, it is noted that `numactl` terminology differs from the standard definition used in this document. Specifically, each NUMA region is considered a "node" by `numactl`, whereas "node" typically refers to separate distributed memory systems.

:p What is the difference between the NUMA terminology used in `numactl` and the definitions provided in the text?
??x
The term "node" in `numactl` specifically refers to each NUMA region within a system. In contrast, this document uses "node" to denote separate distributed memory systems such as different computers or trays in a rack-mounted setup.

:p Why is it important to know the specific terminology when working with NUMA configurations?
??x
Knowing the specific terminology is crucial because it helps avoid confusion and ensures that optimizations are applied correctly. For instance, understanding that each NUMA region can be treated as an independent memory domain allows developers to better manage resource allocation and memory access patterns.

---

---
#### OpenMP for Parallelizing Loops
Background context: The provided text explains how to use OpenMP to parallelize a loop. OpenMP is an API that supports multi-platform shared memory multiprocessing programming. It allows developers to write multithreaded C or C++ programs using compiler intrinsics and pragmas.

:p What is the role of the `#pragma omp parallel for` in the provided code?
??x
The `#pragma omp parallel for` directive is used to distribute iterations of a loop among multiple threads. This allows for parallel execution, improving performance by utilizing all available cores or processors.

```c
#pragma omp parallel for // Spawns threads and distributes work
for (int i=0; i<STREAM_ARRAY_SIZE; i++){
    c[i] = a[i] + scalar*b[i];
}
```
x??

---
#### OpenMP Synchronization
Background context: The `#pragma omp parallel` directive in the provided code creates a team of threads. Each thread can perform its own tasks independently, but they need to be synchronized when accessing shared resources.

:p How does the `#pragma omp masked` directive work in the stencil example?
??x
The `#pragma omp masked` directive is used within an OpenMP parallel region to mask certain iterations or parts of a loop from being executed by some threads. This can help with memory placement and better utilization of caches, especially when using techniques like "first touch."

```c
#pragma omp parallel // Spawns threads
#pragma omp masked
printf("Running with %d thread(s) ",omp_get_num_threads());
```
x??

---
#### Stencil Operation Implementation
Background context: The stencil operation is a common numerical technique used in various scientific computations. It involves computing the value of each cell based on its neighbors.

:p What does the `#pragma omp parallel for` directive do in the stencil example?
??x
The `#pragma omp parallel for` directive is used to distribute iterations of a loop among multiple threads, allowing for parallel execution. In the provided code, it is applied twice: once for initializing the arrays and again for performing the stencil computation.

```c
#pragma omp parallel for // Spawns threads and distributes work
for (int j = 0; j < jmax; j++){
   for (int i = 0; i < imax; i++){
       xnew[j][i] = 0.0;
       x[j][i] = 5.0;
   }
}
```
x??

---
#### First Touch and Memory Placement
Background context: The first touch technique is a memory optimization strategy where the code is designed to ensure that each thread "first touches" its own data, reducing contention on shared resources and improving cache performance.

:p Why is an `#pragma omp parallel for` inserted at line 17 in the stencil example?
??x
The `#pragma omp parallel for` directive at line 17 is used to create a team of threads that will execute the print statement. By masking these iterations, it ensures that only certain threads perform the print operation, which can help with proper memory placement and cache utilization.

```c
#pragma omp parallel for // Spawns threads and distributes work
```
x??

---
#### Loop-Level OpenMP Implementation Details
Background context: The provided code demonstrates a loop-level OpenMP implementation where each thread is responsible for processing a portion of the data. The `#pragma omp parallel for` directive is used to distribute iterations among threads.

:p What is the purpose of the `#pragma omp parallel for` in the initialization and stencil computation loops?
??x
The `#pragma omp parallel for` directive distributes the iterations of the loop among multiple threads, allowing for parallel execution. This improves performance by leveraging all available cores or processors.

For initialization:
```c
#pragma omp parallel for // Spawns threads and distributes work
for (int j = 0; j < jmax; j++){
   for (int i = 0; i < imax; i++){
       xnew[j][i] = 0.0;
       x[j][i] = 5.0;
   }
}
```

For stencil computation:
```c
#pragma omp parallel for // Spawns threads and distributes work
for (int j = 1; j < jmax-1; j++){
   for (int i = 1; i < imax-1; i++){
       xnew[j][i]=(x[j][i] + x[j][i-1] + x[j][i+1] + 
                   x[j-1][i] + x[j+1][i])/5.0;
   }
}
```
x??

---

