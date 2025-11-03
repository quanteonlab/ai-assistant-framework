# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 48)

**Starting Chapter:** 14.3 Thread affinity with OpenMP

---

---
#### Thread Affinity Overview
Thread affinity is crucial for optimizing applications using OpenMP by ensuring threads are tied to specific locations that minimize memory latency and maximize bandwidth. This helps in maintaining locality of reference, which is essential for performance.

:p What is thread affinity in the context of OpenMP?
??x
Thread affinity in OpenMP refers to the technique of binding or associating threads with specific processor cores or sockets to optimize performance by reducing memory access latency and improving data locality.
x??

---
#### OMP_PLACES Environment Variable
The `OMP_PLACES` environment variable controls where threads can be scheduled. It allows specifying a place such as "sockets", "cores", or "threads". Additionally, you can provide an explicit list of places to further constrain thread placement.

:p What does the `OMP_PLACES` environment variable control in OpenMP?
??x
The `OMP_PLACES` environment variable controls where threads can be scheduled within a program. It specifies the place (e.g., sockets, cores, or threads) that limits the scheduler's freedom to move threads around. Setting this appropriately helps in maintaining thread locality and improving performance.
x??

---
#### OMP_PROC_BIND Environment Variable
The `OMP_PROC_BIND` environment variable manages how threads are bound to processors. It offers options like `close`, `spread`, `primary`, and `true` or `false`. The default setting is `true`.

:p What does the `OMP_PROC_BIND` environment variable do in OpenMP?
??x
The `OMP_PROC_BIND` environment variable determines how threads are bound to processors. Setting it to `close` keeps threads close together, while `spread` distributes them across available processors. `primary` schedules threads on the main processor, and `true` or `false` controls whether threads can be moved at all.
x??

---
#### Example Vector Addition with Affinity
The provided example code demonstrates vector addition using OpenMP and shows how to use `OMP_PLACES` and `OMP_PROC_BIND`. It includes a function to report thread placement.

:p How is the affinity of threads managed in the provided vector addition example?
??x
In the provided vector addition example, the affinity of threads is managed through the `OMP_PLACES` and `OMP_PROC_BIND` environment variables. These variables control where threads can be scheduled and how they are bound to processors. The example shows setting these variables to demonstrate different configurations.
x??

---
#### Code Example for Vector Addition
Here’s a code snippet from the provided example, demonstrating the use of OpenMP directives and thread placement reporting.

:p What does this C code do?
??x
This C code demonstrates vector addition using OpenMP and includes functions to report thread placement. It uses `OMP_PLACES` and `OMP_PROC_BIND` to control thread affinity and placement.
```c
#include <stdio.h>
#include <time.h>
#include "timer.h"
#include "omp.h"
#include "place_report_omp.h"

#define ARRAY_SIZE 80000000

static double a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

void vector_add(double *c, double *a, double *b, int n);

int main(int argc, char *argv[]) {
    #ifdef VERBOSE
        place_report_omp();
    #endif
    struct timespec tstart;
    double time_sum = 0.0;

    #pragma omp parallel 
    { 
        #pragma omp for 
        for (int i=0; i<ARRAY_SIZE; i++) {
            a[i] = 1.0;
            b[i] = 2.0;
        }

        #pragma omp masked
        cpu_timer_start(&tstart);
        vector_add(c, a, b, ARRAY_SIZE);
        #pragma omp masked
        time_sum += cpu_timer_stop(tstart);
    } // end of omp parallel

    printf("Runtime is %lf msecs", time_sum);
}

void vector_add(double *c, double *a, double *b, int n) {
    #pragma omp for 
    for (int i=0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```
x??

---

---
#### OpenMP Reporting Routine Overview
This section discusses the placement reporting routine used to gather and report thread affinity information using OpenMP. The routine is designed to be easily toggled on or off via an `#ifdef` directive, making it flexible for various testing scenarios.

:p What does the `place_report_omp` function do?
??x
The `place_report_omp` function reports the number of threads being used and their placement settings using OpenMP. It also prints out each thread's core affinity and its assigned socket (or processing place).

```c
void place_report_omp(void) {
    #pragma omp parallel
    { 
        if (omp_get_thread_num() == 0){
            printf("Running with %d thread(s)", omp_get_num_threads());
            int bind_policy = omp_get_proc_bind();
            switch (bind_policy) {
                case omp_proc_bind_false:
                    printf(" proc_bind is false ");
                    break;
                // other cases...
            }
            printf(" proc_num_places is %d", omp_get_num_places());
        } 
    }

    int socket_global[144];
    char clbuf_global[144][7 * CPU_SETSIZE];

    #pragma omp parallel
    { 
        int thread = omp_get_thread_num();
        cpu_set_t coremask;
        char clbuf[7 * CPU_SETSIZE];
        memset(clbuf, 0, sizeof(clbuf));
        sched_getaffinity(0, sizeof(coremask), &coremask);
        cpuset_to_cstr(&coremask, clbuf);
        strcpy(clbuf_global[thread],clbuf);
        socket_global[omp_get_thread_num()] = omp_get_place_num();
        #pragma omp barrier
        #pragma omp master
        for (int i=0; i<omp_get_num_threads(); i++){
            printf("Hello from thread %d: (core affinity = %s) OpenMP socket is %d", 
                i, clbuf_global[i], socket_global[i]);
        }  
    }
}
```

x??
---

#### CPU Bit Mask Conversion Function
This function converts a CPU bit mask to a C string, detailing which cores are active. It iterates through each core and constructs a descriptive string based on contiguous sets of active cores.

:p What does the `cpuset_to_cstr` function do?
??x
The function takes a `cpu_set_t` structure representing a set of active CPUs and converts it to a human-readable format in a C-string. It identifies contiguous ranges of CPU IDs that are marked as active within the mask, then formats these into strings like "0-1" or "3", which are appended to the output string.

```c
static char *cpuset_to_cstr(cpu_set_t *mask, char *str) {
    char *ptr = str;
    int i, j, entry_made = 0;

    for (i = 0; i < CPU_SETSIZE; i++) { 
        if (CPU_ISSET(i, mask)) {  
            int run = 0;
            entry_made = 1;
            for (j = i + 1; j < CPU_SETSIZE; j++) {
                if (CPU_ISSET(j, mask)) run++;
                else break;
            }
            if (run) 
                sprintf(ptr, "%d,", i);
            else if (run == 1) { 
                sprintf(ptr, "%d,%d,", i, i + 1);  
                i++; 
            } else { 
                sprintf(ptr, "%d-%d,", i, i + run); 
                i += run; 
            }
            while (*ptr != '\0') ptr++;
        } 
    }

    ptr -= entry_made;
    *ptr = 0;
    return str;
}
```
x??

---

#### Affinity Binding and Placement Settings
This section discusses how to control the affinity of threads using environment variables in OpenMP. By setting `OMP_PLACES=cores` and `OMP_PROC_BIND=close`, you can pin threads to specific hardware cores, optimizing performance.

:p What are the effects of setting `OMP_PLACES=cores` and `OMP_PROC_BIND=close`?
??x
Setting `OMP_PLACES=cores` informs OpenMP that thread placement should be done based on cores. Setting `OMP_PROC_BIND=close` ensures that threads are bound to nearby cores, reducing latency and improving performance.

The output shows that the threads are now pinned to specific virtual cores within a single hardware core, leading to a 25% reduction in computation time from 0.0221 ms to 0.0166 ms compared to previous settings where threads could run on any processor.

```sh
export OMP_PLACES=cores
export OMP_PROC_BIND=close
./vecadd_opt3
```
x??

---

#### Querying OpenMP Settings and Thread Affinity
This example demonstrates how to query the current OpenMP settings for thread placement and affinity. The output reveals that with no environment variables set, threads can run on any virtual core from 0 to 87.

:p What does running `./vecadd_opt3` without setting environment variables show?
??x
Running `./vecadd_opt3` without setting environment variables results in the following output:

```
The core affinity allows the thread to run on any of the 88 virtual cores.
proc_bind is false
proc_num_places is 0
Hello from thread 0: (core affinity = 0-87)
OpenMP socket is -1
...
Hello from thread 43: (core affinity = 0-87)
OpenMP socket is -1
0.022119
```

This indicates that the threads are not pinned to any specific cores, and can run on any of the virtual cores ranging from 0 to 87.

```sh
export OMP_NUM_THREADS=44
./vecadd_opt3
```
x??

---

#### Automating Exploration with Multiple Threads
This section explains how to automate the exploration of different OpenMP settings for varying numbers of threads, which can help in optimizing performance by adjusting thread placement and binding based on system characteristics.

:p How can you automate the exploration of OpenMP settings?
??x
To automate the exploration of OpenMP settings, you can use scripts that vary the number of threads and change environment variables like `OMP_PLACES` and `OMP_PROC_BIND`. By running such a script, you can observe how different settings affect performance.

Example steps:

1. Create a build directory and navigate to it.
2. Configure and make your application with verbose options enabled.
3. Run the program with different numbers of threads and environment variable configurations.

```sh
mkdir build && cd build
cmake -DCMAKE_VERBOSE=on ..
make

export OMP_NUM_THREADS=44
./vecadd_opt3
```

By varying `OMP_PLACES` and `OMP_PROC_BIND`, you can test different placements and bindings to find the optimal configuration for your specific workload.
x??

---

---
#### Directory and Build Process
The provided script involves building a CMake project, creating a directory for the build, configuring it with CMake, compiling it using `make`, and then running specific tests to measure performance. This setup is crucial for understanding how different thread counts and placement settings affect program performance.

:p What does the initial command sequence do?
??x
The commands `mkdir build && cd build` create a directory named "build" and switch into that directory, preparing it for building the project from source code located in another part of the file system. Then, `cmake ..` runs CMake to configure the build with the appropriate settings, typically found in a parent directory (indicated by ".."). Finally, `make` compiles the source files according to the configuration generated by CMake.

```sh
# Example shell commands for building and configuring
mkdir build && cd build
cmake ..
make
```
x??

---
#### Performance Script: `calc_avg_stddev()`
The script contains a function named `calc_avg_stddev()` which calculates the average runtime of multiple trials and their standard deviation. This is essential to understand how different settings affect program performance.

:p What does the `calc_avg_stddev()` function do?
??x
The `calc_avg_stddev()` function processes input data, typically from timing results, to calculate the mean (average) and standard deviation of the runtime values over multiple trials. This helps in quantifying the variability and consistency of performance across different configurations.

```bash
# Example pseudocode for calc_avg_stddev()
function calc_avg_stddev(inputData)
    sum = 0.0
    sum2 = 0.0
    count = 0
    for each value in inputData
        sum += value
        sum2 += (value * value)
        count += 1
    end

    avg = sum / count
    std_dev = sqrt((sum2 - (sum*sum)/count) / count)
    print "Number of trials: ", count, "avg: ", avg, "std dev: ", std_dev
end function
```
x??

---
#### Performance Script: `conduct_tests()`
The script also includes a `conduct_tests()` function that runs performance tests multiple times to generate statistical data about runtime. This function helps in analyzing the impact of different thread counts and placement settings.

:p What does the `conduct_tests()` function do?
??x
The `conduct_tests()` function runs timing tests for a specific executable (in this case, `./vecadd_opt3`) ten times to collect runtime data. It calculates the average runtime and standard deviation using the `calc_avg_stddev()` function and processes this information.

```bash
# Example pseudocode for conduct_tests()
function conduct_tests(exec_string)
    time_val = array of size 10
    foo = ""
    for i from 1 to 10
        result = system(exec_string)
        time_val[i] = result
        foo += " " + result
    end

    calc_avg_stddev(foo)
end function
```
x??

---
#### Executing Performance Tests
The script sets the environment variables `OMP_NUM_THREADS`, `OMP_PLACES`, and `OMP_PROC_BIND` to test different thread counts, core placements, and binding settings. It then runs performance tests with these configurations and compares their speedup against a single-threaded baseline.

:p How does the script set up its test conditions?
??x
The script sets up multiple test scenarios by configuring environment variables for OpenMP such as `OMP_NUM_THREADS`, `OMP_PLACES`, and `OMP_PROC_BIND`. It then runs performance tests with various thread counts, core placements, and binding settings. The objective is to measure the speedup of running a program (`./vecadd_opt3`) under different conditions.

```bash
# Example setup code for testing
exec_string="./vecadd_opt3"
conducted_tests(exec_string)

THREAD_COUNT="88 44 22 16 8 4 2 1"
for my_thread_count in $THREAD_COUNT
do
    unset OMP_PLACES
    unset OMP_PROC_BIND
    export OMP_NUM_THREADS=$my_thread_count

    conducted_tests(exec_string)

    PLACES_LIST="threads cores sockets"
    BIND_LIST="true false close spread primary"

    for my_place in $PLACES_LIST
    do
        for my_bind in $BIND_LIST
        do
            export OMP_NUM_THREADS=$my_thread_count
            export OMP_PLACES=$my_place
            export OMP_PROC_BIND=$my_bind

            conducted_tests(exec_string)
        done
    done
done
```
x??

---
#### Performance Analysis Results
The results show that the program generally performs best with 44 threads, and hyperthreading does not provide significant benefits. The `close` setting for thread binding limits memory bandwidth until more than 44 threads are used, but at full 88 threads, it offers the highest performance.

:p What were the key findings of the performance analysis?
??x
The key findings indicate that the program performs optimally with a specific number of threads (44), and hyperthreading does not significantly enhance performance. The `close` binding setting for threads limits memory bandwidth when used until more than 44 threads are active, but it provides the best overall performance at 88 threads due to full utilization.

```bash
# Key findings from the analysis
The program is fastest with all settings and only 44 threads.
Hyperthreading does not help in general.
Thread binding `close` setting shows a limited memory bandwidth effect up to 44 threads but provides better performance at 88 threads by fully utilizing both sockets.
```
x??

---

#### Hyperthreading Impact on Memory-Bound Kernels
Background context: The analysis indicates that hyperthreading does not significantly benefit simple memory-bound kernels, but it also doesn’t introduce a noticeable penalty. This means for applications where memory access is the bottleneck, using multiple threads per core (hyperthreading) might be unnecessary.

:p What impact does hyperthreading have on simple memory-bound kernels?
??x
Hyperthreading typically does not provide significant benefits to simple memory-bound kernels but also doesn't harm performance noticeably.
x??

---

#### Multi-Socket Memory-Bound Kernels and Process Affinity
Background context: For memory-bandwidth-limited kernels operating across multiple sockets (NUMA domains), it is important to utilize both sockets effectively. The text suggests that not showing the results of setting `OMP_PROC_BIND` to `primary` is due to its potential to significantly slow down programs.

:p What is the recommended approach for multi-socket applications with memory-bound kernels?
??x
For multi-socket applications, ensure that both sockets are utilized by keeping processes spread across different cores. Avoid settings like `OMP_PROC_BIND=primary` as it can degrade performance.
x??

---

#### OpenMPI Process Affinity and Placement Settings
Background context: Applying process affinity in MPI (specifically with OpenMPI) helps maintain memory bandwidth and cache performance, preventing the operating system from migrating processes to different cores. The text discusses using tools like `ompi_info`, `mpiexec`, and `srun` for managing process placement.

:p What tool is recommended for setting process affinity in OpenMPI?
??x
The recommended tools for setting process affinity in OpenMPI include `ompi_info`, `mpiexec`, and `srun`. These tools help in managing the placement of processes to optimize performance.
x??

---

#### Impact of OMP_PROC_BIND=spread on VecAdd Speedup
Background context: The text mentions that using `OMP_PROC_BIND=spread` can boost parallel scaling by about 50 percent for the `VecAdd` application. This setting helps distribute threads across multiple cores, optimizing performance.

:p How does the `omp_proc_bind=spread` setting affect `VecAdd` speedup?
??x
The `omp_proc_bind=spread` setting improves the parallel scalability of the `VecAdd` application by spreading the threads across different cores, leading to a 50 percent increase in speedup compared to other settings.
x??

---

#### Thread and Core Placement Strategies
Background context: The graph in Figure 14.4 illustrates how varying thread and core placement strategies can affect parallel scaling for an `omp_proc_bind=spread` setting. The text suggests that spreading processes across cores (Threads spread, Cores spread) is generally more beneficial than keeping them close together.

:p What does the term "threads spread cores spread" imply in process placement?
??x
"Threads spread cores spread" implies a strategy where both threads and cores are distributed across multiple processor cores to maximize parallelism and optimize resource utilization.
x??

---

#### OpenMPI Default Process Placement
Background context: When using OpenMPI, the process placement and affinity are not left to the kernel scheduler but are specified by default. The default settings depend on the number of processes involved:
- Processes ≤ 2: Bind to core
- Processes > 2: Bind to socket
- Processes > processors: Bind to none

Sometimes, HPC centers might set other defaults such as always binding to cores. This policy can be suitable for most MPI jobs but may cause issues with applications using both OpenMP threading and MPI because all threads will be bound to a single processor, leading to serialization.

:p What is the default process placement in OpenMPI when the number of processes is less than or equal to 2?
??x
The default process placement in OpenMPI when there are 2 or fewer processes is binding each process to a core.
x??

---

#### OpenMPI Process Placement and Affinity Control
Background context: Recent versions of OpenMPI offer extensive tools for managing process placement and affinity. Using these, you can achieve performance gains that depend on how the operating system's process scheduler optimizes placement.

:p How does the operating system scheduler typically optimize placement?
??x
The operating system scheduler is generally optimized for general computing tasks such as word processing and spreadsheets rather than parallel applications. As a result, it may not always provide optimal placement for MPI jobs.
x??

---

#### Distributing Processes Across Multi-Node Jobs in OpenMPI
Background context: When running an application that requires more memory than a single node can provide (e.g., 32 MPI ranks with half a terabyte of memory), you need to distribute the processes across multiple nodes. This example demonstrates how to do this using OpenMPI.

:p How many nodes are needed for 32 MPI ranks if each node has 128 GiB of memory and the application requires half a terabyte (512 GiB)?
??x
To support 32 MPI ranks with half a terabyte of memory, you would need at least 4 nodes because each node only provides 128 GiB. Therefore, four nodes are required to meet the memory requirement.

For this example, we use 32 processes across multiple nodes.
x??

---

#### Using `mpirun` Command for Process Placement
Background context: You can control process placement and binding using options with the `mpirun` command in OpenMPI. This allows you to distribute processes more effectively across available hardware.

:p How do you launch 32 processes using `mpirun`?
??x
You can launch 32 processes using the `mpirun` command as follows:
```bash
mpirun -n 32 ./your_program
```
Here, `-n 32` specifies that 32 processes should be launched.
x??

---

#### Placement Reporting Tool in MPI Applications
Background context: The code provided includes a simple placement reporting tool to demonstrate how you can track process placement and affinity within an MPI application. This tool uses the `place_report_mpi()` function.

:p What is the purpose of the `place_report_mpi()` function?
??x
The `place_report_mpi()` function's purpose is to report on the placement and core affinity of each MPI process. It initializes MPI, reports the hostname and core affinity for each rank, and then finalizes MPI.
```c
void place_report_mpi(void) {
    int rank;
    cpu_set_t coremask;
    char clbuf[7 * CPU_SETSIZE], hnbuf[64];

    memset(clbuf, 0, sizeof(clbuf));
    memset(hnbuf, 0, sizeof(hnbuf));

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    gethostname(hnbuf, sizeof(hnbuf));
    sched_getaffinity(0, sizeof(coremask), &coremask);
    cpuset_to_cstr(&coremask, clbuf);
    printf("Hello from rank %d on %s (core affinity = %s)", rank, hnbuf, clbuf);
}
```
x??

---

#### Example of Process Placement Across Nodes
Background context: The provided example demonstrates how to distribute 32 MPI ranks across nodes with a specific configuration. Each node has two sockets with Intel Broadwell CPUs and 128 GiB of memory.

:p How are the processes distributed in this multi-node job?
??x
In the given setup, each node has:
- Two sockets
- Intel Broadwell (E5-2695) CPUs: 18 hardware cores per CPU
- Hyperthreading providing 36 virtual processors per socket

Given there are 32 processes to run, and assuming an even distribution across nodes with sufficient memory on each node, you would distribute the processes as follows:
```plaintext
Node 0: Processes 0 - 15 (36 cores available)
Node 1: Processes 16 - 31 (36 cores available)
```
Each process is bound to a specific core or socket based on the configuration.
x??

---

#### Performance Gain from Custom Placement
Background context: Recent versions of OpenMPI provide tools for custom placement, which can lead to performance gains. These gains depend on how well the scheduler optimizes placement.

:p What potential benefit can be gained by optimizing process placement using OpenMPI?
??x
Optimizing process placement with OpenMPI can yield a performance gain, typically in the range of 5-10%, but it can also result in much larger benefits. The exact gain depends on how well the scheduler is configured and optimized for the specific workload.
x??

---

#### Node Affinity and NUMA Awareness
In distributed computing, particularly with MPI (Message Passing Interface), understanding node affinity and NUMA (Non-Uniform Memory Access) awareness is crucial for optimizing performance. The default settings of OpenMPI bind more than two ranks to a single socket. This can lead to memory allocation failures if the application's requirements exceed the available memory on a single node.

:p What does the output in Figure 14.5 and 14.6 indicate about the placement of MPI processes?
??x
The output shows that when using `mpirun -n 32 --npernode 8`, the processes are distributed across four nodes (cn328 to cn331), with each node handling eight ranks. The core affinity for each rank is set to its respective NUMA region, confirming that the processes are spread out while maintaining affinity within a socket.

```bash
# Example of using mpirun with --npernode
mpirun -n 32 --npernode 8 ./MPIAffinity | sort -n -k 4
```
x??

---

#### Core Affinity and Memory Allocation
Core affinity in MPI applications can significantly impact performance by influencing how processes are mapped to physical cores. If the application's memory requirements exceed the available RAM on a single node, it may fail during memory allocation.

:p How does spreading out the processes across multiple nodes help address memory allocation issues?
??x
Spreading out the processes across multiple nodes helps because if the total memory requirement of the application exceeds the capacity of one node, distributing the ranks ensures that enough overall memory is available. In this case, by running 32 MPI processes (8 per node), the application can utilize the combined memory resources of four nodes.

```bash
# Example command to spread out processes across multiple nodes
mpirun -n 32 --npernode 8 ./MPIAffinity | sort -n -k 4
```
x??

---

#### NUMA Region and Socket Affinity
NUMA regions are typically aligned with the sockets of a node. Setting affinity to a socket can optimize data locality, which is important for performance in distributed computing.

:p What does setting the core affinity to the NUMA region mean in this context?
??x
Setting the core affinity to the NUMA region means that each MPI process is bound to specific cores within a particular NUMA domain. In this case, the output indicates that ranks 0-31 are spread across four nodes, with each rank's core affinity corresponding to either the first or second half of the available cores on their respective node.

```bash
# Example command to set socket affinity
mpirun -n 32 --npersocket 4 ./MPIAffinity | sort -n -k 4
```
x??

---

#### Alternative Placement Strategy
Another approach is to use `--npersocket` instead of `--npernode`. This can help in scenarios where processes need to communicate more frequently with their nearest neighbors, as adjacent ranks might be placed on the same NUMA domain.

:p How does using --npersocket 4 differ from --npernode 8?
??x
Using `--npersocket 4` means that four MPI processes are allocated per socket. This can be beneficial if communication between processes is frequent, as it ensures that processes communicating with each other are closer in terms of memory locality.

```bash
# Example command to set process placement based on sockets
mpirun -n 32 --npersocket 4 ./MPIAffinity | sort -n -k 4
```
x??

---

#### Binding MPI Processes to Cores
Background context: In this scenario, we are exploring how to control where MPI processes run on a multi-core system by binding them to specific hardware resources. The `--bind-to` option in `mpirun` allows us to specify the granularity of process placement and affinity. For example, using `--bind-to core`, we bind each process to a specific core.
:p What does the `--bind-to core` option do in MPI applications?
??x
This option binds each MPI process to a specific hardware core, which can help optimize performance by reducing cache contention and improving data locality. In our example, using this binding method ensures that processes are assigned to cores 0-17 and 36-53.
```shell
mpirun -n 32 --npersocket 4 --bind-to core ./MPIAffinity | sort -n -k 4
```
x??

---

#### Binding MPI Processes to Hyperthreads
Background context: Continuing from the previous example, we now explore binding processes to hyperthreads. This method ensures that each process runs on a single virtual core (hyperthread), which can further reduce cache contention and improve performance.
:p What does the `--bind-to hwthread` option do in MPI applications?
??x
This option binds each MPI process to a specific hardware thread or hyperthread, ensuring that processes run on only one virtual core. In our example, using this binding method results in each process being assigned to a single location, as shown in Figure 14.9.
```shell
mpirun -n 32 --npersocket 4 --bind-to hwthread ./MPIAffinity | sort -n -k 4
```
x??

---

#### NUMA Binding and Placement
Background context: Non-Uniform Memory Access (NUMA) refers to a system where memory access times depend on the location of the data relative to the processing element. Binding processes to specific NUMA nodes can help reduce inter-process communication latency.
:p How does binding MPI processes to NUMA regions work?
??x
Binding MPI processes to NUMA regions ensures that processes are placed in such a way that they share memory and resources within the same NUMA node, reducing inter-process communication latency. In our example, using `--bind-to numa` results in four adjacent ranks being on the same NUMA region.
```shell
mpirun -n 32 --npersocket 4 --bind-to numa ./MPIAffinity | sort -n -k 4
```
x??

---

#### MPI Process Placement and Memory Management
Background context: Proper placement of MPI processes can significantly impact performance by managing memory usage effectively. In our example, we aim to use only four out of the eighteen processor cores on each socket to allocate more memory for each MPI rank.
:p Why do you want to limit the number of cores used in an MPI process?
??x
Limiting the number of cores used helps optimize memory usage per MPI rank. By using fewer cores, there is more available memory for each rank, which can improve overall performance and reduce contention among processes. In our example, we use four out of eighteen processor cores on each socket.
```shell
mpirun -n 32 --npersocket 4 --bind-to core ./MPIAffinity | sort -n -k 4
```
x??

---

#### Reporting MPI Process Bindings
Background context: The `--report-bindings` option in `mpirun` provides detailed information about the binding of each MPI process. This can be useful for debugging and understanding how processes are distributed across hardware resources.
:p What does the `--report-bindings` option do?
??x
The `--report-bindings` option reports the bindings of each MPI process, providing a detailed view of where processes are placed on hardware resources such as cores and sockets. This can help in debugging and optimizing performance by ensuring proper resource utilization.

Example output:
```
Hello from rank 0, on cn328. (core affinity = 0) Hello from rank 1, on cn328. (core affinity = 36)
```

```shell
mpirun -n 32 --npersocket 4 --bind-to hwthread --report-bindings ./MPIAffinity
```
x??

---

