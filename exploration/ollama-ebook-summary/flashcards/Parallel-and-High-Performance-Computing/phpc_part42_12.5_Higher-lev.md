# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 42)

**Starting Chapter:** 12.5 Higher-level languages for performance portability. 12.5.1 Kokkos A performance portability ecosystem

---

#### GPU Kernel Capture Mechanism

Background context: In SYCL, capturing variables for use within a kernel or lambda is crucial. This mechanism allows you to pass data from the host (CPU) to the device (GPU) during computation. The way you capture variables can significantly impact performance and correctness.

:p How do you capture variables in a SYCL kernel using lambdas?
??x
In SYCL, you can capture variables by value or reference when defining your lambda functions inside the kernel body. Capturing by value means that each variable is copied into the local scope of the lambda, while capturing by reference means that the lambda uses the original variable's memory location.

For example:
```cpp
Queue.submit([&nsize, &scalar, &dev_a, &dev_b, &dev_c](Sycl::handler& CommandGroup)
{
    // Use dev_a, dev_b, and dev_c within this scope.
});
```

If you want to capture by value, the syntax is slightly different:
```cpp
Queue.submit([=] (Sycl::handler& CommandGroup) 
{
    // Use nsize, scalar, dev_a, dev_b, and dev_c directly from their current state.
});
```
x??

---

#### StreamTriad Example in SYCL

Background context: The stream triad example demonstrates how to perform a simple matrix operation using SYCL. This involves three arrays (a, b, c) where each element of `c` is the sum of elements from `a` and `b`, scaled by a scalar value.

:p What does the StreamTriad function do in the provided SYCL code?
??x
The StreamTriad function performs a stream triad operation: it updates an array `c` such that each element `c[i]` is computed as `a[i] + scalar * b[i]`. This example showcases parallel processing on both CPU and GPU using SYCL.

Code Example:
```cpp
Kokkos::parallel_for(nsize, KOKKOS_LAMBDA (const int i) {
    c[i] = a[i] + scalar * b[i];
});
```

Explanation: The `KOKKOS_LAMBDA` keyword is used to define a lambda function that operates in parallel over the range `[0, nsize)`. This lambda function updates each element of array `c` based on the corresponding elements from arrays `a` and `b`, scaled by `scalar`.

```cpp
for (int i=0; i<nsize && icount < 10; i++) {
    if (c[i] != 1.0 + 3.0*2.0) { // Check for correctness
        cout << "Error with result c[" << i << "]=" << c[i] << endl;
        icount++;
    }
}
```

This loop checks the correctness of the results by comparing them against a known value.
x??

---

#### Kokkos Execution Spaces

Background context: Kokkos is an ecosystem that provides performance portability across different hardware architectures. It supports various execution spaces, enabling developers to write code once and run it on multiple platforms.

:p What are some of the execution spaces provided by Kokkos?
??x
Kokkos provides several named execution spaces:

- Serial Execution (`Kokkos::Serial`)
  - Enabled with `Kokkos_ENABLE_SERIAL`
  
- Multi-threaded Execution (`Kokkos::Threads` / `Kokkos::OpenMP`)
  - Enabled with `Kokkos_ENABLE_PTHREAD` or `Kokkos_ENABLE_OPENMP`

- CUDA Execution (`Kokkos::Cuda`)
  - Enabled with `Kokkos_ENABLE_CUDA`
  
- HPX (High Performance Parallelism)
  - Enabled with `Kokkos_ENABLE_HPX`
  
- ROCm
  - Enabled with `Kokkos_ENABLE_ROCm`

These execution spaces allow the same code to be compiled and run on different hardware platforms without modifications.

Example CMake flags:
```cmake
-DKokkos_ENABLE_SERIAL=On
-DKokkos_ENABLE_PTHREAD=On
-DKokkos_ENABLE_OPENMP=On
-DKokkos_ENABLE_CUDA=On
-DKokkos_ENABLE_HPX=On
-DKokkos_ENABLE_ROCm=On
```
x??

---

#### Kokkos CMake Configuration

Background context: Setting up a project to use Kokkos requires configuring the build process with specific CMake flags. These configurations allow for cross-platform and multi-backend support.

:p How do you configure Kokkos with OpenMP backend using CMake?
??x
To configure Kokkos with an OpenMP backend using CMake, follow these steps:

1. Clone the Kokkos repository.
2. Create a build directory and navigate to it.
3. Run `cmake` with the appropriate flags.

Example command:
```sh
mkdir build && cd build
cmake ../kokkos -DKokkos_ENABLE_OPENMP=On
```

Then, configure and compile the project using CMake:
```sh
export Kokkos_DIR=${HOME}/Kokkos/lib/cmake/Kokkos
cmake ..
make
```

Ensure that `Kokkos_DIR` is set correctly to point to the location of the generated CMake configuration files.

Example setup environment variables:
```sh
export OMP_PROC_BIND=true
export OMP_PLACES=threads
```
x??

---

#### Kokkos Initialization and Finalization
Kokkos is a performance portable framework that encapsulates flexible multi-dimensional array allocations. It starts with `Kokkos::initialize()` to set up the execution space, such as threads, and ends with `Kokkos::finalize()`. This ensures proper resource management across different architectures.

:p What do Kokkos::initialize() and Kokkos::finalize() do?
??x
These functions are used to initialize and finalize resources for the Kokkos runtime environment. They start up the necessary execution space (e.g., threads) when called, and properly clean up these resources when `Kokkos::finalize()` is invoked.

```cpp
#include <Kokkos_Core.hpp>

int main() {
    // Initialize Kokkos resources.
    Kokkos::initialize();

    // Application code here...

    // Finalize Kokkos resources to release any associated memory or other resources.
    Kokkos::finalize();
}
```
x??

---

#### Kokkos Views and Memory Spaces
Kokkos views allow for flexible multi-dimensional array allocations, which can be switched depending on the target architecture. This includes different data orders for CPU versus GPU operations.

:p What is a Kokkos View used for?
??x
A Kokkos View is used to create multidimensional arrays with flexible layouts that can adapt to different execution spaces (CPU or GPU). It allows specifying memory spaces like `HostSpace`, `CudaSpace`, etc., and layout options such as `LayoutLeft` or `LayoutRight`.

```cpp
// Example of using a Kokkos::View for 1D array allocation.
Kokkos::View<double*, Kokkos::DefaultExecutionSpace> view_name("Name", size);
```
x??

---

#### Parallel For Pattern in Kokkos
Kokkos supports parallel execution through patterns like `parallel_for`. This pattern is used to execute a kernel function across all elements of the data array in parallel.

:p What does the `parallel_for` pattern do?
??x
The `parallel_for` pattern executes a kernel function in parallel over all elements of the data array. It uses lambdas for specifying the kernel, which Kokkos handles with readable syntax via macros like `KOKKOS_LAMBDA`.

```cpp
// Example using parallel_for.
KOKKOS_LAMBDA(int i) { ... }  // Lambda used to define the kernel.

// Kernel execution pattern.
Kokkos::parallel_for(num_elements, KOKKOS_LAMBDA(int i) {
    // Operation on each element of the array...
});
```
x??

---

#### RAJA for Performance Portability
RAJA is a performance portable framework that aims to minimize disruptions to existing codes. It supports various backends like OpenMP, CUDA, and TBB.

:p What does RAJA offer in terms of portability?
??x
RAJA offers a simpler and easier-to-adopt approach for achieving performance portability across different architectures (like CPUs and GPUs). It uses lambdas extensively, making it adaptable to both CPU and GPU environments. RAJA can be built with support for OpenMP, CUDA, TBB, etc.

```cpp
// Example of using RAJA.
using Policy = RAJA::KernelPolicy<RAJA::seq_exec>;  // Or other execution policies.

void kernel_function(int i) {
    // Operation on each element...
}

RAJA::forall<nbytes>(Policy{}, 0, nbytes, KOKKOS_LAMBDA(auto i) { 
    // RAJA lambda syntax.
});
```
x??

---

#### RAJA CMake Integration
RAJA can be integrated into projects using CMake. The project needs to find and link the necessary RAJA libraries.

:p How is RAJA set up in a CMake project?
??x
RAJA can be set up in a CMake project by including `find_package` commands for RAJA and OpenMP, then linking the required libraries. This setup ensures that RAJA is properly integrated into the build process.

```cmake
# CMakeLists.txt example.
cmake_minimum_required(VERSION 3.0)
project(StreamTriad)

find_package(Raja REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(StreamTriad StreamTriad.cc)
target_link_libraries(StreamTriad PUBLIC RAJA)
set_target_properties(StreamTriad PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
```
x??

---

#### Raja and Performance Portability
Background context explaining how RAJA enables performance portability across different hardware platforms. Mention that it uses a domain-specific language (DSL) to express algorithms for various target devices.

:p What is RAJA used for?
??x
RAJA is used for enabling performance portability across different hardware platforms by expressing algorithms in a domain-specific language (DSL) that can be compiled and executed on various targets, including CPUs, GPUs, and other accelerators.
x??

---

#### C++ Lambda Function Usage in Raja
Explanation of how lambda functions are utilized within RAJA's `forall` construct to perform computations. Mention the importance of this approach for expressing parallelism.

:p How is a computation loop expressed using Raja's `forall`?
??x
A computation loop can be expressed using Raja's `forall` by defining a lambda function that performs the desired operation on each element within a range. This allows RAJA to generate optimized code for different hardware backends.
```cpp
RAJA::forall<RAJA::omp_parallel_for_exec>(            RAJA::RangeSegment(0, nsize),[=](int i){     c[i] = a[i] + scalar * b[i]; });
```
x??

---

#### Integrated Build and Run Script for Raja Stream Triad
Explanation of the script provided to build and run the Raja stream triad example. Highlight the key steps involved in setting up RAJA, including installing it and building the example code.

:p What does the setup script for Raja do?
??x
The setup script for Raja builds and installs RAJA on a specified directory by first creating a temporary build folder, running CMake to configure the build, and then making and installing the software. It then builds and runs the stream triad example code.
```bash
# Example of the setup script steps
export INSTALL_DIR=`pwd`/build/Raja
mkdir -p build/Raja_tmp && cd build/Raja_tmp
cmake ../../Raja_build -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
make -j
install && cd .. && rm -rf Raja_tmp

cmake ..
make
./StreamTriad
```
x??

---

#### Additional Reading and Resources for GPU Programming Languages
Explanation of the various resources available for learning different GPU programming languages. Mention popular books, websites, and communities for each language.

:p What are some additional resources for learning CUDA?
??x
For learning CUDA, you can start with NVIDIA's Developer website at https://developer.nvidia.com/cuda-zone, where extensive guides on installing and using CUDA are available. Additionally, the book by David B. Kirk and W. Hwu Wen-Mei, "Programming massively parallel processors: a hands-on approach" (Morgan Kaufmann, 2016), is a valuable reference.
x??

---

#### Exercises for GPU Programming Languages
Explanation of exercises provided to help gain practical experience with various GPU programming languages like CUDA, HIP, SYCL, and RAJA.

:p What exercise involves changing the host memory allocation in the CUDA stream triad example?
??x
The first exercise involves changing the host memory allocation in the CUDA stream triad example to use pinned memory. This change aims to see if performance improvements can be observed by leveraging CPU caching mechanisms.
x??

---

#### Kokkos and SYCL for Performance Portability
Explanation of single-source performance portability languages like Kokkos and SYCL, highlighting their advantages for running applications on a variety of hardware platforms.

:p What is the advantage of using single-source performance portability languages?
??x
The main advantage of using single-source performance portability languages like Kokkos and SYCL is that they allow you to write portable code that can be compiled and executed on different hardware backends without the need for multiple code versions. This approach simplifies maintenance and ensures better compatibility across various platforms.
x??

---

#### Sum Reduction Example with CUDA
Explanation of the sum reduction example using CUDA, emphasizing the importance of careful design in GPU kernel programming.

:p What is the objective of the sum reduction example?
??x
The objective of the sum reduction example is to demonstrate how to efficiently compute a sum over an array on a GPU. It highlights the importance of careful design and optimization techniques when implementing algorithms for parallel architectures.
x??

---

#### HIP (Heterogeneous-compute Interface for Portability) Conversion
Explanation of converting CUDA code to HIP, emphasizing interoperability between different hardware backends.

:p How can the CUDA reduction example be converted to use HIP?
??x
The objective is to convert the CUDA reduction example into a HIP version. This involves modifying the kernel code and host functions to work with the HIP API, which allows for portability across AMD GPUs.
x??

---

#### SYCL Example Initialization on GPU Device
Explanation of initializing arrays on the GPU device using SYCL.

:p How can the SYCL example in Listing 12.20 be modified to initialize arrays on the GPU?
??x
To modify the SYCL example in Listing 12.20, you would need to allocate memory on the GPU and initialize the arrays directly on the device. This involves using SYCL commands for memory allocation and data transfer.
```cpp
cl::sycl::buffer<int, 1> buf_a(a, nsize);
cl::sycl::buffer<int, 1> buf_b(b, nsize);
cl::sycl::queue q;
q.submit([=](cl::sycl::handler &cgh) {
    auto acc_a = buf_a.get_access<cl::sycl::access::mode::write>(cgh);
    auto acc_b = buf_b.get_access<cl::sycl::access::mode::write>(cgh);
    // Initialize arrays on the device
});
```
x??

---

#### RAJA Example with Raja:forall Syntax
Explanation of converting initialization loops in a RAJA example to use `Raja:forall` syntax.

:p How can the initialization loops in Listing 12.24 be converted to use `Raja:forall`?
??x
To convert the initialization loops in Listing 12.24, you would replace the manual loop with a call to `RAJA::forall`. This involves defining a lambda function that performs the assignment and using it within the `RAJA::RangeSegment`.
```cpp
RAJA::forall<RAJA::omp_parallel_for_exec>(            RAJA::RangeSegment(0, nsize),[=](int i){     c[i] = a[i] + scalar * b[i]; });
```
x??

---

