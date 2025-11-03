# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 50)

**Starting Chapter:** 14.6.2 Using likwid-pin An affinity tool in the likwid tool suite

---

---
#### Using `mpirun_distrib.sh` to Set MPI Affinity
Background context: The provided text explains how to use a custom script, `mpirun_distrib.sh`, to run an MPI application with affinity set on specific cores. This is done by binding the processes to specific hardware resources using the `hwloc-bind` command.
:p How does `mpirun_distrib.sh` ensure that the MPI processes are bound to specific cores?
??x
The script `mpirun_distrib.sh` uses `mpirun -np 1 hwloc-bind core:<core_numbers> ./MPIAffinity`. For example, `./mpirun_distrib.sh "1 22" ./MPIAffinity` sets the affinity for the application to run on cores 1 and 22. The `hwloc-bind` command is used to specify the exact hardware resources (in this case, cores) that each process will use.
```bash
#!/bin/bash
APP="$2"
CORES="$1"

mpirun -np 1 hwloc-bind core:$CORES $APP
```
x??

---
#### Using `likwid-pin` for OpenMP Pinning
Background context: The text discusses using the `likwik-pin` tool to set affinity for both MPI and OpenMP applications. It specifically covers how to pin threads in an OpenMP application.
:p How does `likwik-pin` handle pinning threads in an OpenMP application?
??x
The `likwik-pin` tool can be used to manually specify the hardware cores on which OpenMP threads will run by using the `-c S0:0-21@S1:0-21 ./vecadd_opt3` command. This binds 22 threads per socket, ensuring that each thread is assigned to a specific core within those sockets.
```bash
export OMP_NUM_THREADS=44
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

./vecadd_opt3
```
However, the `likwik-pin` tool can achieve the same pinning without setting these environment variables. It automatically determines the number of threads from the pin set lists and places them accordingly.
```bash
likwid-pin -c S0:0-21@S1:0-21 ./vecadd_opt3
```
x??

---
#### Using `likwik-mpirun` for MPI Pinning
Background context: The text explains how to use the `likwik-mpirun` tool, which is an extension of `mpirun` and provides similar functionality but with additional features. This is used to set affinity for MPI applications.
:p How does `likwik-mpirun` ensure that MPI ranks are pinned to specific cores?
??x
The `likwik-mpirun` tool pinns the MPI ranks directly to the hardware cores by default, without requiring any additional options. For example, running `likwid-mpirun -n 44 ./MPIAffinity` will distribute 44 MPI ranks across 44 available cores.
```bash
likwid-mpirun -n 44 ./MPIAffinity | sort -n -k 4
```
This command runs the MPI application `MPIAffinity` with 44 ranks, and each rank is bound to a specific core. The output includes detailed placement reports showing which threads are running on which cores.
x??

---
#### Understanding Pinning Behavior in `likwik-pin`
Background context: The text discusses how `likwik-pin` handles pinning based on the number of threads defined by the user versus the number of available processors in the specified pin sets. It explains that if there are more threads than processors, the tool wraps around to distribute them.
:p What happens when the number of threads exceeds the number of available processors?
??x
If you set `OMP_NUM_THREADS` or specify more threads via a pin set list than there are available processors, `likwik-pin` will wrap the thread placement around on the available processors. For instance, if 45 threads are requested but only 44 cores are available, the first 44 threads will be placed as expected, and the 45th thread will start over from core 0.
```bash
likwid-pin -c S0:0-21@S1:0-21 ./vecadd_opt3
```
x??

---
#### Exploring `likwik-pin` Syntax for Processor Sets
Background context: The text provides an overview of the syntax used by `likwik-pin` to define processor sets and how to use these sets to set affinity. It explains different numbering schemes like physical numbering, node-level numbering, socket-level numbering, etc.
:p How do you specify a pinning set using the `-c` option in `likwik-pin`?
??x
The `-c` option in `likwik-pin` allows specifying processor sets by using various numbering schemes. For example:
```bash
-omp_places=threads: likwik-pin -c S0:0-21@S1:0-21 ./vecadd_opt3
```
This command uses socket-level numbering (`S`) to pin 22 threads on each of the two sockets (0 and 1). The `@` symbol is used to concatenate multiple sets.
x??

---
#### Interpreting Placement Reports
Background context: The text mentions that running an example application with `-DCMAKE_VERBOSE` option provides a detailed placement report, showing how OpenMP has placed and pinned threads. However, it also notes that the same placement can be achieved using `likwik-pin` without setting specific environment variables.
:p How does the output of `likwik-pin` compare to that of an OpenMP application?
??x
The output from `likwik-pin` shows a detailed placement report similar to what would be generated by an OpenMP application. In both cases, the threads are pinned to specific cores based on the pin set lists provided. The text confirms that setting environment variables like `OMP_NUM_THREADS` is not necessary when using `likwik-pin`.
```bash
 likwid-pin -c S0:0-21@S1:0-21 ./vecadd_opt3
```
This command runs the application with threads pinned to specific cores, providing the same placement and pinning results as would be seen from an OpenMP application.
x??

---

#### Setting Affinities in Your Executable
Background context: The text discusses embedding pinning logic into an executable to simplify process placement and affinity management. This approach can be more user-friendly than using complex mpirun commands, as it integrates affinity settings directly into the application.

:p How can you set affinities within your executable?
??x
You can use libraries like QUO (which is built on top of hwloc) to programmatically set and change process affinities. For example, in a C or Java program, you would query hardware information, determine optimal core bindings, and then apply these settings.

Example using the QUO library involves querying hardware resources and setting affinity policies:

```c
// Pseudocode for setting CPU affinity with QUO
#include <hwloc.h>

int main() {
    hwloc_topology_t topology;
    hwloc_obj_type_t type;

    // Initialize topology
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    // Query the core resources
    type = hwloc_get_type_by_name(topology, "core");
    if (type == HWLOC_OBJ_TYPE_NOTFOUND) {
        fprintf(stderr, "Error: cannot find 'core' type.\n");
        return -1;
    }

    int socket0_cores[] = { /* list of cores on socket 0 */ };
    hwloc_cpuset_setsocket_mask(socket0_cores, topology, 0);

    // Set the affinity
    if (hwloc_set_cpuset_affinity(topology, NULL) != 0) {
        fprintf(stderr, "Error setting CPU affinity.\n");
        return -1;
    }

    // Continue with your application logic

    hwloc_topology_destroy(topology);
    return 0;
}
```

x??

---

#### Using likwid-mpirun for Process Placement
Background context: The text demonstrates how to use `likwid-mpirun` to distribute MPI ranks across available hardware cores. This tool provides a convenient way to manage process placement without manual command-line arguments.

:p How do you use `likwid-mpirun` to distribute MPI ranks?
??x
You can use `likwid-mpirun` with specific options to distribute MPI ranks effectively. For instance, the following command distributes 22 ranks across the first 22 hardware cores on socket 0:

```bash
likwid-mpirun -n 22 ./MPIAffinity | sort -n -k 4
```

Adding `-nperdomain S:11` ensures that 11 ranks are placed on each socket, which is useful when you need to account for NUMA (Non-uniform Memory Access) considerations.

```bash
likwid-mpirun -n 22 -nperdomain S:11 ./MPIAffinity | sort -n -k 4
```

This command pinns the ranks in numeric order, as shown by the placement report.

x??

---

#### Run Time Affinity Management with QUO
Background context: The text introduces the QUO library for setting and modifying affinity at runtime. This is particularly useful when applications call libraries that use both MPI ranks and OpenMP threads, requiring dynamic adjustments to process binding.

:p What does the QUO library enable in terms of run-time affinity management?
??x
The QUO library allows you to dynamically set and change CPU affinities during program execution. It leverages hwloc for hardware topology information and provides an easy-to-use interface for managing process bindings.

Example initialization using QUO:

```c
// Pseudocode for initializing QUO in C
#include <quo.h>

int main() {
    // Initialize QUO context
    quo_context_t ctx = quo_init();

    // Set core affinity (example)
    hwloc_obj_type_t type;
    int socket0_cores[] = { /* list of cores on socket 0 */ };
    quo_set_affinity(ctx, "core", socket0_cores);

    // Continue with your application logic

    quo_destroy(ctx);
    return 0;
}
```

x??

---

#### QUO Library and hwloc Integration
Background context: The text explains that the QUO library integrates well with hwloc to manage process affinities. This integration allows for detailed hardware topology queries and setting of bindings.

:p How does the QUO library integrate with hwloc?
??x
The QUO library integrates with hwloc by providing a high-level interface for querying hardware resources and setting affinity policies. It uses hwloc's functionality to determine the topology, such as cores and sockets, and then applies these insights to set appropriate affinities.

Example of using QUO and hwloc together:

```c
// Pseudocode for integrating QUO with hwloc
#include <hwloc.h>
#include <quo.h>

int main() {
    // Initialize hwloc topology
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    // Initialize QUO context
    quo_context_t ctx = quo_init();

    // Query core resources using hwloc
    int socket0_cores[] = { /* list of cores on socket 0 */ };
    hwloc_obj_type_t type;
    type = hwloc_get_type_by_name(topology, "core");
    if (type == HWLOC_OBJ_TYPE_NOTFOUND) {
        fprintf(stderr, "Error: cannot find 'core' type.\n");
        return -1;
    }
    hwloc_cpuset_setsocket_mask(socket0_cores, topology, 0);

    // Set affinity using QUO
    quo_set_affinity(ctx, "core", socket0_cores);

    // Continue with your application logic

    quo_destroy(ctx);
    hwloc_topology_destroy(topology);
    return 0;
}
```

x??

---

#### Initializing QUO Context and Getting System Information
Background context: This concept involves initializing the QUO (Quiet Uninterrupted Operation) context, which is crucial for managing process bindings. The function `QUO_create` initializes the context, and subsequent calls to `QUO_id`, `QUO_nqids`, `QUO_ncores`, and `QUO_obj_type_t` retrieve system information.

:p What are the steps involved in initializing the QUO context and retrieving system information?
??x
The function `QUO_create(&qcontext, MPI_COMM_WORLD)` initializes the QUO context using the MPI world communicator. Then, we use several calls to get the number of nodes (`nnoderanks`), node rank (`noderank`), core count (`ncores`), and other relevant information.

```c
QUO_context qcontext;
MPI_Init(&argc, &argv);
QUO_create(&qcontext, MPI_COMM_WORLD);
MPI_Comm_size(MPI_COMM_WORLD, &nranks);  // Get total number of ranks
MPI_Comm_rank(MPI_COMM_WORLD, &rank);    // Get rank of current process
QUO_id(qcontext, &noderank);              // Get node rank
QUO_nqids(qcontext, &nnoderanks);         // Get number of nodes
QUO_ncores(qcontext, &ncores);            // Get core count per node
```
x??

---

#### Reporting Default Bindings
Background context: After initializing the QUO context and getting system information, this step reports the default process bindings. This is useful for understanding how processes are initially bound to cores before any modifications.

:p What does reporting the default bindings show?
??x
Reporting the default bindings shows the initial state of process bindings on the hardware cores. This provides a baseline or starting point to understand and compare with the modified bindings after adjustments.

```c
if (rank == 0) {
    printf(" Default binding for MPI processes \");
}
place_report_mpi();
```
x??

---

#### Synchronizing Processes During Bindings
Background context: Process synchronization is crucial when changing process affinities. The `SyncIt` function uses an MPI barrier and a micro sleep to ensure that all processes are in sync before making changes.

:p How does the `SyncIt` function synchronize processes?
??x
The `SyncIt` function synchronizes processes by first getting the rank of the current process using `MPI_Comm_rank`. It then uses `MPI_Barrier` to ensure that all processes have reached this point. Finally, it introduces a delay (`usleep`) proportional to the process's rank.

```c
void SyncIt(void) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);      // Ensure all processes reach this point
    usleep(rank * 1000);               // Introduce a delay proportional to rank
}
```
x??

---

#### Binding Processes to Hardware Cores Using QUO
Background context: The `QUO_bind_push` and `QUO_auto_distrib` functions are used to bind processes to hardware cores. This step changes the bindings from sockets (default) to the core level, ensuring optimal performance.

:p What does the function `QUO_auto_distrib` do?
??x
The `QUO_auto_distrib` function automatically distributes processes based on the specified object type (`tres`) and maximum members per resource (`max_members_per_res`). It returns the actual number of members assigned to each resource (`work_member`).

```c
QUO_bind_push(qcontext, QUO_BIND_PUSH_PROVIDED, QUO_OBJ_CORE, noderank);
QUO_auto_distrib(qcontext, tres, max_members_per_res, &work_member);
```
x??

---

#### Reporting Bindings After Modifications
Background context: After modifying the bindings with `QUO_auto_distrib`, it is essential to report the new process bindings. This helps in verifying that the changes were applied correctly and provides insights into the new binding configuration.

:p What does reporting the bindings after modifications show?
??x
Reporting the bindings after modifications shows how processes are now bound to hardware cores. This comparison with the initial default bindings helps in assessing the effectiveness of the changes made by `QUO_auto_distrib`.

```c
if (rank == 0) {
    printf(" Processes should be pinned to the hw cores \");
}
place_report_mpi();
```
x??

---

#### Freeing QUO Context and Finalizing MPI
Background context: After all modifications are done, it is necessary to free the QUO context and finalize the MPI environment. This ensures that all resources are properly released.

:p What does freeing the QUO context and finalizing MPI accomplish?
??x
Freeing the QUO context (`QUO_free(qcontext)`) releases any allocated resources used by the QUO system. Finalizing MPI (`MPI_Finalize()`) cleans up all MPI-related resources, ensuring that no memory leaks or resource conflicts occur.

```c
QUO_free(qcontext);
MPI_Finalize();
```
x??

---

#### Initialization of QUO Context
Background context: The process initializes the QUO (Affinity Manager) context for managing process bindings and affinity settings within MPI and OpenMP regions.

:p How is the QUO context initialized in the provided code?
??x
The `QUO_create` function is called to create a new QUO context using the MPI communicator `MPI_COMM_WORLD`. This sets up the environment for managing affinity settings.
```c
QUO_create(&qcontext, MPI_COMM_WORLD);
```
x??

---

#### Node Information Reporting
Background context: The node information report provides details about the rank and number of ranks on a specific node.

:p What does the function `node_info_report` do in the provided code?
??x
The `node_info_report` function reports the current node information, including the rank of the process (`noderank`) and the total number of nodes (`nnoderanks`). This is used to understand the distribution of processes across nodes.
```c
node_info_report(qcontext, &noderank, &nnoderanks);
```
x??

---

#### Synchronization Function `SyncIt`
Background context: The `SyncIt` function is a placeholder for synchronization logic within the MPI region.

:p What is the purpose of the `SyncIt` function in the provided code?
??x
The `SyncIt` function serves as a placeholder to ensure that processes are synchronized before and after setting process bindings. It is called at key points to maintain coherence between different affinity settings.
```c
void SyncIt() {
    // Placeholder for synchronization logic
}
```
x??

---

#### Pushing Bindings with QUO
Background context: The `QUO_bind_push` function pushes new bindings onto the stack, allowing dynamic changes in process binding policies.

:p How does the `QUO_bind_push` function push core bindings for a specific node rank?
??x
The `QUO_bind_push` function is used to bind the current MPI process to a specific core on its node. The first call binds it to the hardware core, and the second call expands the binding to the entire node.
```c
QUO_bind_push(qcontext, QUO_BIND_PUSH_PROVIDED, QUO_OBJ_CORE, noderank);
```
x??

---

#### Auto-Distribution of MPI Ranks
Background context: The `QUO_auto_distrib` function automatically distributes and binds processes across available resources.

:p What does the `QUO_auto_distrib` function do in the provided code?
??x
The `QUO_auto_distrib` function is used to distribute and bind MPI ranks to hardware cores. It takes the number of members per resource (`max_members_per_res`) as an argument, ensuring that processes are spread across available resources.
```c
QUO_auto_distrib(qcontext, QUO_OBJ_SOCKET, max_members_per_res, &work_member);
```
x??

---

#### Expanding Bindings for OpenMP Region
Background context: The code snippet shows how bindings can be expanded to cover the entire node when entering an OpenMP region.

:p How does the code expand the bindings from core-level to whole-node binding in the provided example?
??x
The `QUO_bind_push` function is used with the `QUO_BIND_PUSH_OBJ` option and a socket ID of `-1`, which expands the current process's cpuset to cover all available resources on the node.
```c
QUO_bind_push(qcontext, QUO_BIND_PUSH_OBJ, QUO_OBJ_SOCKET, -1);
```
x??

---

#### Reverting to Initial Bindings
Background context: The `QUO_bind_pop` function is used to revert process bindings to their initial state after entering an OpenMP region.

:p How does the code ensure that MPI bindings are restored in the provided example?
??x
The `QUO_bind_pop` function pops off the current bindings and restores them to the previous settings, allowing processes to return to their initial bindings.
```c
QUO_bind_pop(qcontext);
```
x??

---

#### Reporting Process Affinities
Background context: The code snippet includes functions for reporting process affinities at different stages.

:p What is the purpose of `place_report_mpi_quo` and `place_report_mpi_omp` in the provided code?
??x
The `place_report_mpi_quo` function reports the current bindings for MPI processes, while `place_report_mpi_omp` reports the bindings for OpenMP threads. These functions help in monitoring how process affinities change between different regions.
```c
void place_report_mpi_quo(QUO_context *qcontext) {
    // Code to report MPI affinities
}

void place_report_mpi_omp() {
    // Code to report OpenMP affinities
}
```
x??

---

#### Finalization and Cleanup
Background context: The code snippet includes cleanup logic for freeing resources after the execution is complete.

:p What does the `QUO_free` function do in the provided code?
??x
The `QUO_free` function frees the QUO context, releasing any allocated resources. This is called at the end of the application to ensure proper resource management.
```c
QUO_free(qcontext);
```
x??

---

#### Process Placement and Bindings in MPI, OpenMP, and MPI+OpenMP
Background context: The handling of process placement and bindings is a relatively new but crucial topic in parallel programming. This involves understanding how processes are assigned to specific cores or nodes in hardware architectures, which can significantly impact performance.

This area is particularly important as it influences the efficiency and scalability of parallel applications running on high-performance computing (HPC) systems. The handling methods vary between MPI, OpenMP, and their combination, each offering unique features and capabilities.

:p What are process placement and bindings in HPC contexts?
??x
Process placement refers to how processes or threads are assigned to specific cores or nodes within a hardware architecture. Bindings determine the association of tasks (processes or threads) with particular processors or memory locations, optimizing resource utilization and performance.

Bindings can be set at various levels: process, thread, or task level. For example, in MPI+OpenMP applications, both MPI processes and OpenMP threads can have specific binding requirements to optimize load balancing and reduce contention.
x??

---

#### Affinity Explorations
Background context: Affinity is a key concept in parallel computing that controls the scheduling of processes or threads on specific hardware resources. This is especially relevant for optimizing performance on HPC systems.

Affinity can be set using various tools and libraries such as MPI, OpenMP, and hwloc. These tools help manage how tasks are placed and bound to hardware resources, ensuring efficient execution and minimizing inter-task communication overhead.

:p What are some references recommended for exploring affinity?
??x
Some key references include:

- Y. He, B. Cook, et al., “Preparing NERSC users for Cori, a Cray XC40 system with Intel many integrated cores” in Concurrency Computat: Pract Exper., 2018; 30:e4291 (https://doi.org/10.1002/cpe.4291).
- Argonne National Laboratory’s “Affinity on Theta,” available at https://www.alcf.anl.gov/support-center/theta/affinity-theta.
- NERSC’s "Process and Thread Affinity," found at https://docs.nersc.gov/jobs/affinity/.

These resources provide insights into how affinity can be used to optimize performance in HPC environments.

x??

---

#### OpenMP: Beyond the Common Core
Background context: OpenMP is a widely-used shared-memory parallel programming API that allows for multi-threading within a single process. It offers various mechanisms for task scheduling and resource management, including affinity settings.

The `mpirun` command from OpenMPI provides options to control process placement, which can be further customized using the Portable Hardware Locality (hwloc) library.

:p What are some resources for exploring advanced OpenMP features?
??x
- T. Mattson and H. He, "OpenMP: Beyond the common core," available at https://mng.bz/aK47.
- The man page for `mpirun` in OpenMPI can be found at https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php.

These resources offer detailed information on advanced features of OpenMP, including how to manage process placement and affinity settings.

x??

---

#### Portable Hardware Locality (hwloc)
Background context: hwloc is a standalone package that provides a universal hardware interface for most MPI implementations and many other parallel programming software applications. It helps in managing process placements and thread bindings efficiently.

The hwloc library can be used with both OpenMPI and MPICH, making it a versatile tool for optimizing the performance of parallel applications across different environments.

:p What is Portable Hardware Locality (hwloc)?
??x
Portable Hardware Locality (hwloc) is a standalone package that provides a hardware interface to manage process placements and thread bindings. It works with various MPI implementations like OpenMPI and MPICH, making it a universal tool for optimizing parallel applications on diverse hardware architectures.

:p How does hwloc help in managing process placements?
??x
hwloc helps by providing detailed information about the underlying hardware architecture (e.g., cores, sockets, NUMA nodes). This information is used to make informed decisions about where processes and threads should be placed to optimize performance. For example, it can ensure that data locality is maintained, reducing inter-node communication overhead.

:p What are some resources for exploring hwloc?
??x
- The main page of the hwloc project at https://www.open-mpi.org/projects/hwloc/.
- B. Goglin, “Understanding and managing hardware affinities with Hardware Locality (hwlooc),” presented at High Performance and Embedded Architecture and Compilation (HiPEAC) 2013, available at http://mng.bz/gxYV.

These resources provide comprehensive information on how to use hwloc effectively for affinity management in parallel applications.
x??

---

#### likwid Suite of Tools
Background context: The likwid suite is well-regarded for its simplicity and good documentation. It offers tools for performance monitoring and benchmarking, which can be particularly useful when exploring affinity settings.

The suite includes various utilities that help developers understand the performance characteristics of their code and optimize it accordingly.

:p What are some resources for exploring the likwid suite?
??x
- University of Erlangen-Nuremberg’s performance monitoring and benchmarking suite at https://github.com/RRZE-HPC/likwik/wiki.
- A conference presentation about the QUO library, which gives a more complete overview and philosophy behind it: S. Gutiérrez et al., “Accommodating Thread-Level Heterogeneity in Coupled Parallel Applications,” available at https://github.com/lanl/libquo/blob/master/docs/slides/gutierrez-ipdps17.pdf (2017 International Parallel and Distributed Processing Symposium, IPDPS17).

These resources provide detailed information on how to use the likwid suite effectively for performance analysis and optimization.
x??

---

---
#### Discovering Hardware Characteristics and Running Tests

Background context: To optimize system performance, it is crucial to understand the hardware characteristics of your devices. This involves running various tests using a specific script to gather data on how the hardware behaves under different conditions.

:p What did you discover about optimizing your system's use after running the test suite?

??x
After running the test suite with the provided script (Listing 14.1), we discovered key insights into our system's performance characteristics, such as cache utilization, memory bandwidth usage, and CPU efficiency. These findings help in making informed decisions about task placement and resource allocation.

For example:
```bash
# Sample command to run the test script
./run_tests.sh
```

This process helps identify bottlenecks and optimize parallel tasks for maximum performance.
x??

---
#### Vector Addition Optimization with Pythagorean Formula

Background context: The vector addition (vecadd_opt3.c) example in Section 14.3 was modified to include more floating-point operations using the Pythagorean formula \( c[i] = \sqrt{a[i]*a[i] + b[i]*b[i]} \). This change affects how tasks are placed and bound to cores.

:p How did changing the vector addition kernel to use the Pythagorean formula affect your results?

??x
Changing the vector addition kernel to use the Pythagorean formula significantly impacted performance. The new operations required more computational resources, leading to different CPU cache utilization patterns and potentially affecting the ideal core bindings.

For example:
```c
// Original vecadd_opt3.c
for (i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
}

// Modified version using Pythagorean formula
for (i = 0; i < N; i++) {
    c[i] = sqrt(a[i]*a[i] + b[i]*b[i]);
}
```

The results showed that the new operations required more cache space and potentially longer execution times, affecting placement decisions.
x??

---
#### MPI Example with Vector Addition and Pythagorean Formula

Background context: For the MPI example in Section 14.4, the vector addition kernel was included, and a scaling graph generated for performance analysis. The kernel was then replaced with the Pythagorean formula to study further.

:p How did replacing the vector addition kernel with the Pythagorean formula change your results?

??x
Replacing the vector addition kernel with the Pythagorean formula led to different scaling behavior in the MPI example. The new operations required more complex computations, which affected the communication and computation balance between processes.

For example:
```c
// Original MPI vector add
for (i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
}

// Modified version using Pythagorean formula
for (i = 0; i < N; i++) {
    c[i] = sqrt(a[i]*a[i] + b[i]*b[i]);
}
```

The results showed that the new operations required more global communication and local computation, changing the ideal process distribution.
x??

---
#### Combining Vector Addition and Pythagorean Formula

Background context: To maximize data reuse, the vector addition and Pythagorean formula were combined in a single routine. This approach aimed to leverage common variables to reduce overhead.

:p How did combining vector addition and Pythagorean formula impact your study?

??x
Combining vector addition with the Pythagorean formula allowed for better data reuse, potentially reducing memory access times and improving overall performance. The new routine:
```c
// Combined routine
for (i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
    d[i] = sqrt(a[i]*a[i] + b[i]*b[i]);
}
```

This approach helped in reducing the number of memory accesses and improving cache utilization, leading to better performance but with more complex tasks per loop iteration.
x??

---
#### Setting Placement and Affinity within Applications

Background context: Managing process placement and affinity is crucial for optimizing parallel application performance. Tools exist to set these parameters within applications dynamically.

:p How can you set the placement and affinity within an application?

??x
To set the placement and affinity within an application, you can use specific environment variables or library functions provided by the runtime system. For example, in OpenMP, you can use `omp_set_num_threads` to specify the number of threads and `omp_get_thread_num` for thread management.

For example:
```c
#include <omp.h>

int main() {
    // Set the number of threads explicitly
    omp_set_num_threads(4);

    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

    return 0;
}
```

This approach allows fine-grained control over thread distribution, ensuring optimal use of hardware resources.
x??

---

---
#### PBS Scheduler Overview
Background context: The Portable Batch System (PBS) is a batch scheduling system for high-performance computing clusters. It originated at NASA and was released as open source under the name OpenPBS in 1998. Commercial versions, such as PBS Professional by Altair and PBS/TORQUE by Adaptive Computing Enterprises, are also available with support contracts.
:p What is the PBS scheduler?
??x
The PBS scheduler is a batch scheduling system for managing jobs on high-performance computing clusters. It offers job submission, job management, resource allocation, and monitoring capabilities to ensure efficient use of cluster resources.

---
#### Slurm Scheduler Overview
Background context: The Simple Linux Utility for Resource Management (Slurm) is another batch scheduling system that originated at Lawrence Livermore National Laboratory in 2002. It has been widely adopted due to its simplicity and flexibility.
:p What is the Slurm scheduler?
??x
The Slurm scheduler is a simple resource management tool for Linux clusters, designed to manage jobs and allocate resources efficiently. It provides job submission, scheduling, and monitoring functionalities.

---
#### Customizations in Batch Schedulers
Background context: Both PBS and Slurm can be customized with plugins or add-ins that provide additional functionality, support for special workloads, and improved scheduling algorithms.
:p How can batch schedulers like PBS and Slurm be customized?
??x
Batch schedulers like PBS and Slurm can be customized using plugins or add-ins to enhance their functionality. These customizations can include adding support for specific applications, implementing new scheduling policies, and improving performance metrics.

---
#### Management of High-Performance Clusters
Background context: As the number of users on high-performance computing clusters increases, it becomes necessary to manage the system to ensure efficient job execution and prevent conflicts.
:p Why is management important in high-performance computing?
??x
Management is crucial in high-performance computing because it helps maintain order and efficiency in resource allocation. Without proper management, multiple jobs could collide, leading to slow performance and potential job crashes.

---
#### Portability of Batch Scripts
Background context: While batch scripts are useful for managing jobs on clusters, they can require customization for each system due to variations in scheduler implementations.
:p What challenges arise with the portability of batch scripts?
??x
The portability of batch scripts is a challenge because different systems may have varying scheduler implementations that require specific configurations. Customization is often necessary to ensure that batch scripts work effectively on different clusters.

---
#### Queue and Policy Management
Background context: Batch schedulers like PBS and Slurm allow for the establishment of queues and policies, which can be used to allocate resources according to predefined rules.
:p What are queues and policies in batch schedulers?
??x
Queues and policies in batch schedulers are mechanisms that help manage resource allocation. Queues group jobs with similar characteristics, while policies define how these jobs are prioritized and allocated resources.

---
#### Example of PBS Job Submission
Background context: PBS provides a way to submit jobs through the `qsub` command, which can be used to specify job requirements such as wall time, memory, and node count.
:p How does one submit a job using PBS?
??x
To submit a job using PBS, you use the `qsub` command with appropriate options. For example:
```bash
qsub -l nodes=1:ppn=8,walltime=03:00:00 my_script.sh
```
This command requests one node with 8 processors per node (ppn) and a wall time of 3 hours.

---
#### Example of Slurm Job Submission
Background context: Slurm also allows job submission through the `sbatch` command, specifying resource requirements such as nodes, processors, and time.
:p How does one submit a job using Slurm?
??x
To submit a job using Slurm, you use the `sbatch` command with appropriate options. For example:
```bash
sbatch --nodes=1 --ntasks-per-node=8 --time=03:00:00 my_script.sh
```
This command requests one node with 8 tasks per node and a time limit of 3 hours.

---
#### Summary of Batch Schedulers
Background context: Both PBS and Slurm offer similar functionalities for managing high-performance computing clusters, but they differ in implementation details.
:p What are the key differences between PBS and Slurm?
??x
The key differences between PBS and Slurm include their origins, implementations, and support options. PBS has a longer history and offers more mature features with commercial support, while Slurm is simpler and more flexible.

---

