# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 31)

**Starting Chapter:** 9.4.2 A benchmark application for PCI bandwidth

---

---
#### PCI Bandwidth Overview
Background context explaining how data is transferred between CPU and GPU using PCI. Theoretical peak bandwidth can be calculated by considering the number of lanes, transfer rate, and overhead factors.

:p What is the formula for calculating theoretical PCI bandwidth?
??x
The formula for calculating theoretical PCI bandwidth involves multiplying the number of lanes by the transfer rate per lane (in gigatransfers per second or GT/s) and then applying an overhead factor to convert to gigabytes per second (GB/s). The formula can be expressed as:

\[ \text{Theoretical Bandwidth (GB/s)} = (\text{Number of Lanes}) \times (\text{Transfer Rate (GT/s)}) \times (\text{Overhead Factor}) \times \frac{\text{Byte}}{\text{Bit (8)}} \]

For a Gen3 x16 PCI system, with 16 lanes and an 8 GT/s transfer rate, the overhead factor is typically around 0.985.

```c
float calculateTheoreticalBandwidth(int numLanes, float transferRateGTs, float overheadFactor) {
    return numLanes * transferRateGTs * overheadFactor / 8.0f;
}
```
x??

---
#### Host to Device Memory Transfer Pseudocode
Explanation of the CUDA-based code that allocates and copies data from the CPU (host) to the GPU (device).

:p What is the purpose of `cudaMallocHost` in this context?
??x
The purpose of `cudaMallocHost` is to allocate memory for data on the host side. This function allocates pinned memory, which means the memory can be accessed more efficiently by both the CPU and the GPU.

```c
cudaError_t status = cudaMallocHost((void**)&x_host, N * sizeof(float));
if (status != cudaSuccess) {
    printf("Error allocating pinned host memory");
}
```

This ensures that the data allocated in `x_host` can be accessed directly by both the CPU and the GPU without any additional overhead.

??x
The function returns a pointer to the newly allocated memory, which is then used for data transfers. If there's an error (indicated by `status` not equaling `cudaSuccess`), it prints an error message.
x??

---
#### Micro-Benchmark Experiment Setup
Explanation of how to set up and run experiments to measure PCI bandwidth.

:p How does the micro-benchmark application measure PCI bandwidth?
??x
The micro-benchmark application measures PCI bandwidth by repeatedly copying data from the CPU (host) to the GPU (device) and timing these operations. The average time for 1,000 copies is used to calculate the transfer rate and then the achieved bandwidth.

```c
void Host_to_Device_Pinned(int N, double *copy_time) {
    float *x_host, *x_device;
    struct timespec tstart;

    cudaError_t status = cudaMallocHost((void**)&x_host, N * sizeof(float));
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory");

    cudaMalloc((void **)&x_device, N * sizeof(float));

    cpu_timer_start(&tstart);

    for(int i = 1; i <= 1000; i++) {
        cudaMemcpy(x_device, x_host, N * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();

    *copy_time = cpu_timer_stop(tstart) / 1000.0;

    cudaFreeHost(x_host);
    cudaFree(x_device);
}
```

The function `Host_to_Device_Pinned` allocates memory, starts timing, performs the data transfer in a loop, and then stops timing to calculate the average time per transfer.

??x
The application uses CUDA functions like `cudaMallocHost`, `cudaMalloc`, `cudaMemcpy`, and `cudaDeviceSynchronize` to manage memory allocation, data transfer, and synchronization. The timing is captured using `cpu_timer_start` and `cpu_timer_stop`.
x??

---
#### Calculating Bandwidth from Copy Time
Explanation of how bandwidth is calculated from the time it takes to copy an array.

:p How is the achieved bandwidth calculated in the micro-benchmark?
??x
The achieved bandwidth is calculated by dividing the number of bytes transferred (which depends on the data type) by the total time taken for multiple transfers. Specifically, if `N` elements are copied and each element has 4 bytes (as they are floats), then the number of bytes (`byte_size`) is \(4 \times N\). The average transfer time over 1,000 trials is used to compute the bandwidth.

```c
for(int j=0; j<n_experiments; j++) {
    array_size = 1;
    for(int i=0; i<max_array_size; i++) {
        Host_to_Device_Pinned(array_size, &copy_time);
        double byte_size = 4.0 * array_size;
        bandwidth[j][i] = byte_size / (copy_time * 1024.0 * 1024.0 * 1024.0);
        array_size = array_size * 2;
    }
}
```

Here, `byte_size` is calculated as \(4 \times \text{array\_size}\), and the bandwidth in gigabytes per second (GB/s) is computed by dividing this value by the total time taken (`copy_time`) and converting to gigabytes.

??x
The code calculates the achieved bandwidth for different array sizes, where `byte_size` accounts for the number of bytes transferred, and `copy_time` records the average transfer time. The bandwidth is then calculated as \( \frac{\text{byte\_size}}{\text{time (in seconds)}} \).

```c
double byte_size = 4.0 * array_size;
bandwidth[j][i] = byte_size / (copy_time * 1024.0 * 1024.0 * 1024.0);
```
x??

---

---
#### Pinned and Pageable Memory
Background context: The difference between pinned and pageable memory affects how data is transferred to the GPU. Pinned memory cannot be paged out of RAM, while pageable memory can. This impacts the performance and efficiency of GPU transfers.

:p What is the main difference between pinned and pageable memory in the context of GPU memory allocation?
??x
Pinned memory refers to memory that cannot be swapped out to disk (paged out), allowing it to be directly accessed by the GPU without making a copy. Pageable memory, on the other hand, can be paged out to disk, which means it must first be copied into pinned memory before being sent to the GPU.

Code example:
```java
// Allocating pinned and pageable memory (pseudo-code)
Pointer pinnedMemory = cudaMallocHost((void**)&hostPtrPinned, size);
Pointer pageableMemory = cudaMallocHost((void**)&hostPtrPageable, size);

// Pinned memory can be directly used by CUDA kernels
cudaMemcpyAsync(devicePtr, hostPtrPinned, size, cudaMemcpyHostToDevice);

// Pageable memory needs to be copied into pinned memory first
cudaMemcpy(hostPtrPinned, pageableMemory, size, cudaMemcpyHostToHost);
cudaMemcpyAsync(devicePtr, hostPtrPinned, size, cudaMemcpyHostToDevice);
```
x??

---
#### GPU Theoretical and Empirical Bandwidth Comparison
Background context: The theoretical peak bandwidth of a GPU can be significantly higher than the empirical (measured) bandwidth. This difference is due to various factors such as system limitations and real-world performance.

:p What does the theoretical peak bandwidth represent, and how does it compare with the measured bandwidth in practice?
??x
The theoretical peak bandwidth represents the maximum data transfer rate a GPU can achieve under ideal conditions without any overheads. In practice, the empirical (measured) bandwidth is often much lower due to real-world constraints like memory bus limitations, system bottlenecks, and other factors.

Graph example:
```
Array size (B)
02468101214
102 103 104 105 106 107 108
Pinned benchmark Pageable benchmark Gen2 theoretical peak Gen1 theoretical peakGen3 theoretical peak
```
In the provided graph, you can see that while the theoretical peaks for different generations of GPUs are high, the measured bandwidth (pinned and pageable) is significantly lower, especially when compared to the theoretical maximums.

x??

---
#### Multi-GPU Platforms and MPI
Background context: In multi-GPU systems, the use of Message Passing Interface (MPI) is often necessary due to the distributed nature of the GPUs. Each MPI rank can be assigned to a single GPU for data parallelism or multiple ranks can share a GPU.

:p What are some common configurations in multi-GPU platforms?
??x
Common configurations include:
- A single MPI rank driving each GPU.
- Multiple MPI ranks multiplexing their work on a single GPU.

This approach is particularly useful for applications requiring high levels of parallelism and data parallelism across multiple GPUs. However, efficient multiplexing requires optimized software as early implementations faced performance issues due to inefficiencies in handling concurrent access to the same GPU resources.

x??

---
#### Data Transfer Optimization Across Networks
Background context: Efficient data transfer between GPUs across networks is crucial for applications that require inter-GPU communication. The standard method involves multiple copies and stages, which can be inefficient. Newer technologies like NVIDIA GPUDirect® allow for direct data transfers over the PCI bus without these intermediate steps.

:p What are some steps involved in the standard data transfer process?
??x
The standard data transfer process includes:
1. Copying data from one GPU to host processor.
   - a. Move the data across the PCI bus to the processor.
   - b. Store the data in CPU DRAM memory.
2. Sending data via an MPI message to another processor.
   - a. Stage the data from CPU memory to the processor.
   - b. Move the data across the PCI bus to the network interface card (NIC).
   - c. Store the data from the processor to CPU memory.
3. Copying data from the second processor to the second GPU.
   - a. Load the data from CPU memory to the processor.
   - b. Send the data across the PCI bus to the GPU.

GPUDirect® and similar technologies reduce this complexity by allowing direct message passing between GPUs, minimizing unnecessary data copying steps.

x??

---
#### Optimizing Data Movement with GPUDirect
Background context: Direct transfer capabilities like NVIDIA GPUDirect® or AMD's DirectGMA enable more efficient communication between GPUs. By bypassing the CPU memory, these methods significantly reduce the overhead and improve overall performance.

:p How does GPUDirect® optimize data movement?
??x
GPUDirect® optimizes data movement by allowing direct message passing between GPUs. Instead of transferring data to and from the CPU's DRAM, it transfers pointers directly over the PCI bus, reducing the number of intermediate steps required for data transfer.

Example:
```java
// Pseudo-code illustrating GPUDirect communication
Pointer sourceGPU = ...;
Pointer destinationGPU = ...;

// Standard method (with CPU involvement)
cudaMemcpy(hostPtrPinnedSource, sourceGPU, size, cudaMemcpyDeviceToHost);
MPI_Send(hostPtrPinnedSource, size, MPI_FLOAT, destRank, tag, MPI_COMM_WORLD);
MPI_Recv(hostPtrPinnedDestination, size, MPI_FLOAT, srcRank, tag, MPI_COMM_WORLD, &status);
cudaMemcpy(destinationGPU, hostPtrPinnedDestination, size, cudaMemcpyHostToDevice);

// GPUDirect method (direct GPU-to-GPU transfer)
sourceGPU.sendTo(destinationGPU); // Direct GPU communication
```
In the example above, standard methods involve CPU involvement, while GPUDirect allows for direct GPU-to-GPU transfers.

x??

#### NVLink and Infinity Fabric Introduction
NVIDIA introduced NVLink to replace traditional GPU-to-GPU and GPU-to-CPU connections. This was particularly significant with their Volta series of GPUs, which started with P100 and V100 models. In addition, AMD's Infinity Fabric aimed to speed up data transfers between various components.
:p What is the main purpose of NVLink?
??x
NVLink aims to enhance the performance of GPU architectures by providing a faster communication channel compared to traditional PCI bus connections, especially for large applications and machine learning workloads on smaller clusters. It achieves this with data transfer rates that can reach up to 300 GB/sec.
x??

---

#### Concept: CPU vs GPU Performance Comparison
The text provides a comparison between the performance of CPUs and GPUs using Roofline plots, which illustrate theoretical peak performance limits for both architectures. For many applications, these peak values are not always achieved in practice.
:p How do Roofline plots help in comparing CPU and GPU performance?
??x
Roofline plots provide a visual representation of an application's performance by dividing the plot into regions representing different performance limitations: floating-point calculation limits (the roofline) and memory bandwidth limits. By overlaying these plots, one can understand where the bottlenecks lie for both CPUs and GPUs.
x??

---

#### Reducing Time-to-Solution with GPU Acceleration
The example uses Cloverleaf as a proxy application to compare the performance of different systems: an Intel Ivybridge system, a Skylake Gold 6152 system, and a V100 GPU system. The goal is to demonstrate how much faster GPUs can be in reducing the time-to-solution for long-running applications.
:p How does porting Cloverleaf from a CPU system to a V100 GPU reduce the run time?
??x
Porting Cloverleaf to a V100 GPU significantly reduces the run time. For instance, running 500 cycles on an Ivybridge system took 615.1 seconds (1.23 seconds per cycle) for a total of 171 hours or about 7 days and 3 hours. On a Skylake system with 36 cores, it took 273.3 seconds (0.55 seconds per cycle) for the same cycles, reducing the run time to 76.4 hours or roughly 3 days and 4 hours. When running on a V100 GPU, it only took 59.3 seconds (0.12 seconds per cycle), significantly cutting down the total runtime to just 16.5 hours.
x??

---

#### Example: CPU Replacement with Skylake Gold 6152
The example compares an Intel Ivybridge system with a Skylake Gold 6152 system, showing how increasing the number of cores and processors can improve performance but not as much as using GPUs.
:p What is the difference between running Cloverleaf on an Ivybridge system versus a Skylake system?
??x
Running Cloverleaf on an Ivybridge system took 615.1 seconds for 500 cycles, averaging 1.23 seconds per cycle and totaling about 171 hours or 7 days and 3 hours. On the Skylake system with 36 cores, it reduced to 273.3 seconds (0.55 seconds per cycle) for the same number of cycles, cutting down the total run time to 76.4 hours or about 3 days and 4 hours.
x??

---

#### Example: GPU Replacement with V100
The example demonstrates how a V100 GPU can significantly reduce the run time compared to both CPU systems by utilizing CUDA for parallel processing.
:p How does running Cloverleaf on a V100 GPU compare to other CPU systems?
??x
Running Cloverleaf on a V100 GPU using CUDA showed a dramatic improvement in performance. The run time was only 59.3 seconds (0.12 seconds per cycle) for the same 500 cycles, reducing the total runtime to just 16.5 hours. This is 4.6 times faster than the Skylake system and 10.4 times faster than the original Ivybridge system.
x??

---

