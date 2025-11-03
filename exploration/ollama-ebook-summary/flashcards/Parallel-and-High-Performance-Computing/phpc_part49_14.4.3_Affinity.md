# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 49)

**Starting Chapter:** 14.4.3 Affinity is more than just process binding The full picture

---

#### Process Affinity and MPI
Background context: In parallel computing, process affinity allows you to control where processes run on a multi-core processor. This is particularly useful for optimizing performance by ensuring that processes remain close to specific hardware components, thus reducing latency. The `taskset` and `numactl` commands are commonly used tools on Linux systems to set this binding.

:p What is the main purpose of process affinity in MPI?
??x
The primary goal of process affinity in MPI is to optimize performance by ensuring that processes run on specific CPU cores or nodes, thereby reducing inter-process communication latency. This can be achieved using `taskset` and `numactl` commands.
x??

---

#### Affinity for Parallel Programming
Background context: In parallel programming, especially with MPI, you need to consider the placement of multiple ranks across available processors. The objective is not just binding processes but also placing them in a way that optimizes performance.

:p What are the additional considerations for process placement in parallel programming compared to single-process scenarios?
??x
In parallel programming using MPI, besides binding individual processes, there are several additional considerations such as:
- Mapping (placement of processes)
- Order of ranks (which ranks should be close together)
- Binding (affinity or tying a process to specific locations)

These factors can significantly impact performance by ensuring that data and communication are optimized.
x??

---

#### Placement of Processes
Background context: Proper placement of processes across available cores is crucial for optimizing the execution time and reducing latency. Tools like `taskset` and `numactl` allow you to specify which core each process should run on.

:p How can you use `taskset` to bind a process to specific cores?
??x
You can use `taskset` to bind a process to specific cores by specifying the bitmask of the desired CPUs. For example, if you want to bind process 1234567890 to core 2 and core 3 on a system where cores are numbered from 0, you would use:
```sh
taskset -c 2,3 <PID>
```
This command sets the process's CPU affinity to run only on cores 2 and 3.

x??

---

#### Order of Ranks
Background context: The order in which ranks are placed can affect data locality and communication efficiency. Proper ordering ensures that processes that communicate frequently are close together.

:p What is the importance of placing closely interacting ranks next to each other?
??x
Placing closely interacting ranks next to each other is important because it maximizes data locality, reducing the time needed for inter-process communication. This arrangement can significantly improve performance by minimizing latency and bandwidth requirements.
x??

---

#### Binding (Affinity)
Background context: Process binding or affinity allows you to control where a process runs on multi-core processors. This is essential for optimizing performance in parallel computing.

:p What does process binding accomplish in the context of MPI?
??x
Process binding in MPI accomplishes the task of ensuring that each rank runs on a specified core, thereby reducing communication latency and improving overall performance by keeping processes close to their data and minimizing cross-core communication.
x??

---

#### Example Placement Report
Background context: The example provided shows an extensive placement report from `mpirun` with the `--report-bindings` option. It demonstrates how ranks are bound to specific cores.

:p What does the output of the `--report-bindings` option show?
??x
The output of the `--report-bindings` option in `mpirun` shows where each rank is bound to specific cores, helping you understand and verify the placement and binding settings. Each line indicates a rank's binding with details such as socket number, core number, and hardware thread.
x??

---

#### Summary
This flashcard series covers the importance of process affinity in MPI, including how it affects performance through proper placement, ordering, and binding processes to specific cores or nodes.

#### Mapping Processes to Processors or Other Locations
Background context: In parallel programming, mapping processes to processors is crucial for efficient resource utilization and performance. The `--map-by` option in OpenMPI allows you to specify how processes are distributed across hardware resources such as slots, hwthreads (hardware threads), cores, sockets, numa nodes, or entire nodes.

The default behavior of the `--map-by` option is to use `socket`, but other options can provide more control. For example:
- `--map-by slot`: Processes are mapped based on a list of available slots.
- `--map-by hwthread`: Processes are mapped based on hardware threads.
- `--map-by core`: Processes are mapped based on cores.

The `ppr` (processes per resource) option allows specifying the number of processes per hardware resource, providing more flexibility. For example:
```bash
mpirun --map-by ppr:8:node
```
This command maps 8 processes to each node in a round-robin fashion.

:p How does the `--map-by` option work in OpenMPI for mapping processes to processors?
??x
The `--map-by` option in OpenMPI allows you to specify how processes are distributed across hardware resources like slots, hwthreads, cores, sockets, numa nodes, or entire nodes. By default, it uses `socket`, but you can use other options such as `slot`, `hwthread`, `core`, etc., for finer control over process mapping.

For example:
- `--map-by slot` maps processes based on a list of available slots.
- `--map-by hwthread` maps processes based on hardware threads.
- `--map-by core` maps processes based on cores.

The `ppr` (processes per resource) option allows specifying the number of processes per hardware resource, providing more flexibility. For instance:
```bash
mpirun --map-by ppr:8:node
```
This command maps 8 processes to each node in a round-robin fashion.
x??

---

#### Block Size for Ordering MPI Ranks
Background context: Controlling the ordering of MPI ranks is essential when processes need to communicate frequently with their neighbors. The block size parameter can be used to group processes together, which can help reduce communication overhead.

:p How does the block size affect the ordering of MPI ranks?
??x
The block size in MPI rank mapping helps control how processes are ordered on physical processors. If adjacent MPI ranks often communicate with each other, placing them close physically reduces the cost of inter-process communication.

For example, if you have 16 processes and want to group every four processes together (a block size of 4), you might use:
```bash
mpirun --map-by ppr:4:node
```
This command maps 4 processes to each node in a round-robin fashion, grouping them into blocks.

Using the appropriate block size can optimize communication patterns and overall performance.
x??

---

#### Using `--cpu-list` for Explicit Process Mapping
Background context: The `--cpu-list` option allows you to specify an explicit list of processor numbers to map processes onto. This is useful when you need precise control over which physical cores or threads are used by your MPI processes.

:p How does the `--cpu-list` option work in OpenMPI?
??x
The `--cpu-list` option in OpenMPI allows you to explicitly specify a list of logical processor numbers to map processes onto. This provides fine-grained control over process placement and can be useful for specific performance tuning scenarios.

For example, if you have 4 MPI ranks and want them placed on specific cores (0, 1, 2, 3), you could use:
```bash
mpirun --cpu-list 0,1,2,3
```
This command binds the processes to these specific logical processors and ensures they are mapped onto those cores.

Using `--cpu-list` can be particularly useful in scenarios where you need to avoid certain hardware resources or ensure that processes run on specific threads.
x??

---

---
#### Affinity and MPI/OpenMP Distribution
MPI applications can benefit from distributing processes across different hardware resources, such as sockets, cores, or nodes. OpenMP applications allow for fine-grained control over thread placement within a process. The `--rank-by` option provides additional control over how MPI ranks are mapped to hardware components.

Using the command:
```
--rank-by ppr:n:[slot | hwthread | core | socket | numa | node]
```
Or the more general `--rankfile <filename>` can help in better placing processes. However, fine-tuning these settings may only provide marginal performance improvements and is generally not necessary for most applications.

Binding MPI processes to specific hardware resources using:
```
--bind-to [slot | hwthread | core | socket | numa | node]
```
With the default setting of `core` being sufficient in many cases. For hybrid MPI/OpenMP applications, the affinity settings need careful consideration since child processes inherit their parent's affinity.

:p What does the `--rank-by` option allow you to do?
??x
The `--rank-by` option allows for more detailed control over how MPI ranks are distributed across hardware components like sockets, cores, or nodes. It provides a way to specify the placement of processes based on different criteria such as slot (hyperthreads), core, socket, numa, and node.

For example:
```
--rank-by ppr:24:socket
```
This would distribute 24 MPI ranks across all available sockets.

```c
// Example usage in a script or command line argument list
int main() {
    // Command setup
    char* cmd = "--rank-by ppr:24:socket";
}
```
x??

---
#### Fine-Tuning Process Placement and Affinity
Fine-tuning process placement can sometimes yield small performance improvements, but it is generally not necessary for most applications. The `--bind-to` option allows binding processes to specific hardware components like slots (hyperthreads), cores, sockets, numa domains, or nodes.

For instance:
```
--bind-to core
```
This binds the process to a single core, while:
```
--bind-to hwthread
```
Binds the process more tightly to hyperthreads on a core.

:p What is the default setting for `--bind-to` when launching an MPI application with more than two processes?
??x
The default setting for `--bind-to` in MPI applications running with more than two processes is `socket`. This means that by default, each process will be bound to a socket (a group of cores sharing a memory domain).

For example:
```c
// Using the default binding to sockets in a script or command line argument list
int main() {
    // Command setup
    char* cmd = "--bind-to socket";
}
```
x??

---
#### Affinity for Hybrid MPI and OpenMP Applications
In hybrid MPI/OpenMP applications, setting affinity correctly can be challenging because child processes inherit the affinity settings of their parent. For instance, if you set `npersocket 4 --bind-to core` and launch two threads (OpenMP), they will share only two logical processor locations per core.

:p How do OpenMP threads inherit the affinity settings from their MPI parent process?
??x
OpenMP threads inherit the affinity settings of their MPI parent process. This means that if you set a specific binding policy for the MPI processes, such as `--bind-to core`, then each thread created by OpenMP will also follow this binding rule.

For example:
```c
// Inheriting affinity in an MPI+OpenMP hybrid application
int main() {
    // MPI and OpenMP setup
    char* mpi_cmd = "--bind-to core";
    char* omp_cmd = "-threads 4"; // Launching four threads

    // Command setup
    char* cmd = "mpirun " + std::string(mpi_cmd) + " -npersocket 4 " + std::string(omp_cmd);
}
```
x??

---
#### Custom Placement Reporting for Hybrid MPI and OpenMP Applications
The `place_report_mpi_omp` function in the provided code snippet customizes the placement report to include information relevant to hybrid MPI and OpenMP applications. It uses OpenMP directives to gather detailed information about thread placement, socket binding, and core affinity.

:p What does the `place_report_mpi_omp` function do?
??x
The `place_report_mpi_omp` function provides a detailed placement report for hybrid MPI and OpenMP applications. It prints out information such as the number of threads, the process binding policy, the number of places available, and the core affinity for each thread.

Here is an example of how this function works:
```c
void place_report_mpi_omp(void) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int socket_global[144];
    char clbuf_global[144][7 * CPU_SETSIZE];

#pragma omp parallel
    { 
        if (omp_get_thread_num() == 0 && rank == 0) {
            printf("Running with %d thread(s)", omp_get_num_threads());
            int bind_policy = omp_get_proc_bind();
            switch (bind_policy) {
                // Various cases for binding policies
            }
            printf("proc_num_places is %d", omp_get_num_places());
        }

        int thread = omp_get_thread_num();
        cpu_set_t coremask;
        char clbuf[7 * CPU_SETSIZE], hnbuf[64];
        memset(clbuf, 0, sizeof(clbuf));
        memset(hnbuf, 0, sizeof(hnbuf));
        gethostname(hnbuf, sizeof(hnbuf));
        sched_getaffinity(0, sizeof(coremask), &coremask);
        cpuset_to_cstr(&coremask, clbuf);
        strcpy(clbuf_global[thread], clbuf);
        socket_global[omp_get_thread_num()] = omp_get_place_num();
        #pragma omp barrier
        #pragma omp master
        for (int i = 0; i < omp_get_num_threads(); i++) {
            printf("Hello from rank %02d, thread %02d, on %s " 
                    "(core affinity = %2s) OpenMP socket is %2d",
                   rank, i, hnbuf, clbuf_global[i], socket_global[i]);
        }
    }
}
```
x??

---

---
#### Stream Triad Code Compilation and Execution
Background context: The provided text describes how to compile and run a stream triad code on a Skylake Gold processor. The code is designed for parallel processing using OpenMP and MPI.

The compilation command involves creating a build directory, running CMake with verbose output enabled, and then compiling the code:

```bash
mkdir build && cd build
./cmake -DCMAKE_VERBOSE=1 ..
make
```

:p What are the steps to compile the stream triad code?
??x
To compile the stream triad code, you need to follow these steps:
1. Create a `build` directory and navigate into it.
2. Run CMake with verbose output enabled using the command: `./cmake -DCMAKE_VERBOSE=1 ..`.
3. Finally, execute the make command to compile the source files.

This process sets up the build environment and compiles the code with detailed logging for debugging purposes.
x??

---
#### Code Execution Layout
Background context: The text explains how to layout the MPI ranks on a Skylake Gold processor using OpenMP threads and hardware cores. The goal is to achieve good memory bandwidth by distributing processes across NUMA domains.

:p How do you configure the environment variables for running the stream triad code with two threads per rank?
??x
To configure the environment variables, you need to set `OMP_NUM_THREADS` to 2 and enable OpenMP thread binding using `OMP_PROC_BIND=true`. Additionally, use the `mpirun` command with specific placement constraints.

Here are the commands:
```bash
export OMP_NUM_THREADS=2
export OMP_PROC_BIND=true
mpirun -n 44 --map-by socket ./StreamTriad
```

These settings ensure that each MPI rank is spread across sockets and that two OpenMP threads are placed on hyperthreads of a hardware core.
x??

---
#### Placement Report Output Interpretation
Background context: The text discusses the output from the placement report, which shows how MPI ranks are distributed across NUMA domains.

:p What does the round-robin distribution pattern in the output indicate?
??x
The round-robin distribution pattern indicates that MPI ranks are placed across NUMA domains in a balanced manner. Specifically, every second rank is assigned to a different socket, ensuring even memory access and potentially better performance due to reduced contention on shared memory.

This distribution helps in maintaining good bandwidth from main memory while allowing the scheduler to move processes freely within their respective NUMA domains.
x??

---
#### Advanced Affinity Constraints
Background context: The text explains how to use advanced affinity constraints with the `--map-by` option. This allows for more precise control over process placement on hardware cores.

:p How do you spread MPI ranks across sockets while ensuring each socket has a specified number of processes?
??x
To spread MPI ranks across sockets, you can use the `--map-by ppr:N:socket:PE=N` option with specific parameters. For instance, to place 22 MPI ranks per socket:

```bash
mpirun -n 44 --map-by ppr:22:socket:PE=1 ./StreamTriad
```

This command places processes in a specified pattern across sockets while binding each rank's threads to hardware cores. The `PE=1` parameter specifies that one physical core can have two virtual processors (threads).

Here, for rank 0 and 1:
- Rank 0 gets the first hardware core with virtual processors 0 and 44.
- Rank 1 gets the next hardware core with virtual processors 22 and 66.

This ensures processes are spread out and threads remain together on their respective cores.
x??

---

#### Background on Affinity Settings
MPI ranks and OpenMP threads are pinned to specific hardware cores and hyperthreads. This ensures efficient parallel processing, reducing communication costs by keeping related tasks close.

:p What is the purpose of setting affinity for MPI and OpenMP?
??x
The purpose of setting affinity for MPI and OpenMP is to optimize performance by ensuring that threads and processes are scheduled on appropriate physical resources, thereby minimizing context switching and inter-processor communication overhead. This improves overall efficiency in distributed computing environments.
x??

---

#### Determining Logical Processors Available
Logical processors (threads) per core and sockets can be determined using the `lscpu` command.

:p How do you determine the number of logical processors available?
??x
The number of logical processors available is determined by executing the `lscpu` command and parsing its output. Specifically, the following lines are used:
```bash
LOGICAL_PES_AVAILABLE=`lscpu | grep '^CPU(s):' | cut -d':' -f 2`
```

This line extracts the total number of logical processors from the `lscpu` output.
x??

---

#### Setting OpenMP Environment Variables
Environment variables like `OMP_PROC_BIND`, `OMP_PLACES`, and `OMP_CPU_BIND` are unset to allow dynamic binding.

:p Why are certain OpenMP environment variables unset?
??x
Certain OpenMP environment variables, such as `OMP_PROC_BIND`, `OMP_PLACES`, and `OMP_CPU_BIND`, are unset because they might interfere with the desired affinity settings. This allows the system to dynamically bind threads to cores without manual configuration.
x??

---

#### Calculating Variables for MPI Run Command
Variables like `HW_PES_PER_PROCESS`, `MPI_RANKS`, and `PES_PER_SOCKET` are calculated based on available resources.

:p What variables need to be calculated before running an MPI command?
??x
Several variables need to be calculated before running an MPI command:
- `HW_PES_PER_PROCESS`: Number of hardware processor elements (HPEs) per process.
- `MPI_RANKS`: Total number of MPI ranks needed.
- `PES_PER_SOCKET`: Number of processors per socket.

These are calculated using the following formulas and commands from the script:
```bash
THREADS_PER_CORE=`lscpu | grep '^Thread(s) per core:' | cut -d':' -f 2`
LOGICAL_PES_AVAILABLE=`lscpu | grep '^CPU(s):' | cut -d':' -f 2`
SOCKETS_AVAILABLE=`lscpu | grep '^Socket(s):' | cut -d':' -f 2`

HW_PES_PER_PROCESS=$((${OMP_NUM_THREADS} / ${THREADS_PER_CORE}))
MPI_RANKS=$((${LOGICAL_PES_AVAILABLE} / ${OMP_NUM_THREADS}))
PES_PER_SOCKET=$((${MPI_RANKS} / ${SOCKETS_AVAILABLE}))
```
x??

---

#### Running MPI Jobs with Affinity Settings
The `mpirun` command is configured to run jobs based on the calculated values.

:p How do you run an MPI job with specific affinity settings?
??x
An MPI job can be run with specific affinity settings using the `mpirun` command. The script constructs a string that includes necessary parameters:
```bash
RUN_STRING="mpirun -n ${MPI_RANKS} --map-by ppr:${PES_PER_SOCKET}:socket:PE=${HW_PES_PER_PROCESS} ./StreamTriad ${POST_PROCESS}"
echo ${RUN_STRING}
eval ${RUN_STRING}
```

This command runs the `StreamTriad` application with the specified number of ranks and mapping.
x??

---

#### Testing Different Numbers of Threads
The script tests various numbers of threads that divide evenly into the number of processors.

:p What does the script do to test different numbers of OpenMP threads?
??x
The script tests different numbers of OpenMP threads by iterating through a list of thread counts. For each count, it sets the `OMP_NUM_THREADS` variable and calculates other necessary values:
```bash
THREAD_LIST_FULL="2 4 11 22 44"
for num_threads in ${THREAD_LIST_FULL}
do
    export OMP_NUM_THREADS=${num_threads}}
    HW_PES_PER_PROCESS=$((${OMP_NUM_THREADS} / ${THREADS_PER_CORE}))
    MPI_RANKS=$((${LOGICAL_PES_AVAILABLE} / ${OMP_NUM_THREADS}))
    PES_PER_SOCKET=$((${MPI_RANKS} / ${SOCKETS_AVAILABLE}))

    RUN_STRING="mpirun -n ${MPI_RANKS} --map-by ppr:${PES_PER_SOCKET}:socket:PE=${HW_PES_PER_PROCESS} ./StreamTriad ${POST_PROCESS}"
    echo ${RUN_STRING}
    eval ${RUN_STRING}
done
```

This loop ensures that the script runs `StreamTriad` with different thread configurations, verifying the affinity settings.
x??

---

#### Affinity: Full Stream Triad Example
Background context explaining how thread and MPI rank combinations affect performance. The example focuses on bandwidth from main memory with little work or MPI communication, limiting hybrid MPI and OpenMP benefits.
:p What is the significance of using 88 processes in the full stream triad example?
??x
In the full stream triad example, using 88 processes helps in testing various combinations of thread sizes and MPI ranks that divide evenly into this number. This setup ensures a balanced distribution of work among threads and ranks without overcomplicating the test with too many variables.
x??

---

#### Affinity: Larger Simulations
Background context discussing how affinity benefits larger simulations by reducing buffer memory requirements, consolidating domains, reducing ghost cell regions, minimizing processor contention, and utilizing underutilized components like vector units.
:p In what scenarios would you expect to see significant benefits from using hybrid MPI and OpenMP in large-scale simulations?
??x
In large-scale simulations, the use of hybrid MPI and OpenMP can provide significant benefits by:
- Reducing MPI buffer memory requirements.
- Creating larger domains that consolidate and reduce ghost cell regions.
- Minimizing contention for processors on a node through better workload distribution.
- Utilizing vector units and other processor components more effectively when they are underutilized.
x??

---

#### Affinity: Controlling from the Command Line
Background context explaining the need to control affinity manually, especially in applications without built-in options. Introduces tools like hwloc and likwid for this purpose.
:p What is the purpose of using command-line tools like hwloc and likwid?
??x
The primary purpose of using command-line tools like hwloc and likwik is to manually control processor affinity when your MPI or parallel application lacks built-in options. These tools help in binding processes close to important hardware components such as graphics cards, network ports, and storage devices.
x??

---

#### Using hwloc-bind to Assign Affinity
Background context on the use of `hwloc-bind` for specifying hardware locations where processes should be bound. Explains how to launch an application with specific core bindings.
:p How do you use `hwloc-bind` to bind a process to a specific hardware location?
??x
To use `hwloc-bind`, prefix your application command with `hwloc-bind` and specify the hardware location where you want the processes to be bound. For example, to run an application on core 2:
```bash
hwloc-bind core:2 my_application
```
This binds the process to core 2.
x??

---

#### Example of Binding Processes Using hwloc-bind
Background context providing a detailed shell script example for binding MPI processes using `hwloc-bind`.
:p How can you create a general-purpose mpirun command with binding using a shell script?
??x
To create a general-purpose mpirun command with binding, you can use the following shell script:
```bash
#!/bin/sh
PROC_LIST=$1
EXEC_NAME=$2
OUTPUT="mpirun "
for core in ${PROC_LIST}
do
    OUTPUT="$OUTPUT -np 1"
    OUTPUT="${OUTPUT} hwloc-bind core:${core}"
    OUTPUT="${OUTPUT} ${EXEC_NAME} :"
done
OUTPUT=$(echo ${OUTPUT} | sed -e 's/:$/ /')
eval ${OUTPUT}
```
This script initializes the `mpirun` command, appends MPI rank launches with binding, and ensures proper formatting of the command.
x??

---

#### lstopo Command Example
Background context explaining how to use `lstopo` to visualize hardware core assignments. The example shows launching multiple instances of `lstopo`.
:p How can you launch multiple processes on different cores using `hwloc-bind` and `lstopo`?
??x
To launch multiple processes on different cores, you can use the following command:
```bash
for core in $(hwloc-calc --intersect core --sep " " all); do hwloc-bind core:${core} lstopo --no-io --pid 0 & done
```
This command uses `hwloc-calc` to get a list of hardware cores, then binds each process to these cores using `hwloc-bind`. Each instance of `lstopo` is launched on the specified core.
x??

---

