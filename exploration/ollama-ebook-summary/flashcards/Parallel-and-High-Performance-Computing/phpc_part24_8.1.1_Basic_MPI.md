# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 24)

**Starting Chapter:** 8.1.1 Basic MPI function calls for every MPI program

---

---
#### Message Passing Interface (MPI)
Background context explaining MPI. It is a standard for parallel computing, allowing programs to run on multiple nodes and facilitate communication between processes through message passing.
:p What is MPI?
??x
MPI is a standard used in high-performance computing that enables parallel processing by distributing tasks across multiple compute nodes and facilitating the exchange of data between these nodes via messages.
x??

---
#### Basic Structure of an MPI Program
Explanation of how to structure a minimal MPI program. It starts with `MPI_Init` at the beginning and ends with `MPI_Finalize`.
:p What is the basic structure of an MPI program?
??x
An MPI program typically follows this structure:
```c
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI
    // Program logic here
    MPI_Finalize();         // Finalize MPI
    return 0;
}
```
x??

---
#### Compilation and Execution of MPI Programs
Explanation on how to compile and run MPI programs. Common compiler wrappers are mentioned.
:p How do you compile and run an MPI program?
??x
To compile and run an MPI program, follow these steps:
- **Compilation:** Use appropriate compilers like `mpicc`, `mpiCC`, or `mpif90` based on the language being used (C/C++, C++, Fortran).
```bash
mpicxx -o my_program my_program.cpp  // For C++
```
- **Execution:** Use a parallel launcher like `mpirun` to specify the number of processes.
```bash
mpirun -np <number_of_processes> ./my_program.x
```
Common alternatives for `mpirun` are `mpiexec`, `aprun`, or `srun`.
x??

---
#### MPI_Init and MPI_Finalize
Explanation on the purpose and usage of these functions.
:p What do `MPI_Init` and `MPI_Finalize` do?
??x
`MPI_Init` initializes the MPI environment, allowing processes to communicate with each other. It must be called at the beginning of the program before any MPI calls are made.

`MPI_Finalize` cleans up resources used by MPI and should be called when the program is exiting.
```c
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI
    // Program logic here
    MPI_Finalize();         // Finalize MPI
    return 0;
}
```
x??

---

#### MPI Initialization and Finalization
Background context: MPI programs are typically initiated and concluded using specific functions. `MPI_Init` is called at the beginning to initialize the MPI environment, while `MPI_Finalize` terminates it. The arguments from the main routine must be passed through `argc` and `argv`, which usually represent the command-line arguments of the program.
:p What function initializes the MPI environment?
??x
The `MPI_Init` function is used to initialize the MPI environment. It takes two arguments: pointers to `argc` and `argv`, which are typically set by the operating system when a program starts, providing information about the command-line parameters passed to the application.

```c
iret = MPI_Init(&argc, &argv);
```

x??

---

#### Process Rank and Number of Processes
Background context: After initializing the MPI environment, it is often necessary to know the rank of the process within its communicator (typically `MPI_COMM_WORLD`) and the total number of processes. This information is crucial for distributing tasks and coordinating communication among processes.
:p How can you determine the rank of a process?
??x
The rank of a process can be determined using the function `MPI_Comm_rank`. This function requires the communicator as its first argument, which is often `MPI_COMM_WORLD`, and returns the rank in an integer variable.

```c
iret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
```

x??

---

#### Compiler Wrappers for Simplified Builds
Background context: To simplify building MPI applications without explicitly knowing about libraries and their locations, compiler wrappers such as `mpicc`, `mpicxx`, and `mpifort` can be used. These tools handle the necessary compile flags internally.
:p What are MPI compiler wrappers?
??x
MPI compiler wrappers (like `mpicc` for C, `mpicxx` for C++, and `mpifort` for Fortran) are tools that simplify building MPI applications. They automatically include the necessary libraries and set appropriate compile flags without requiring manual configuration.

```c
// Example usage of mpicc
mpicc -o my_mpi_program my_mpi_source.c
```

x??

---

#### Basic MPI Function Calls
Background context: The fundamental operations in an MPI program involve initialization (`MPI_Init`), termination (`MPI_Finalize`), and obtaining information about the communicator (like process rank and number of processes).
:p What are basic MPI function calls?
??x
Basic MPI function calls include `MPI_Init`, `MPI_Finalize`, and functions like `MPI_Comm_rank` and `MPI_Comm_size`. These are essential for initializing and finalizing an MPI program, as well as querying information about the communicator.

```c
iret = MPI_Init(&argc, &argv);
iret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
iret = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
iret = MPI_Finalize();
```

x??

---

#### Communicators in MPI
Background context: A communicator in MPI is a group of processes that can communicate with each other. The default communicator `MPI_COMM_WORLD` includes all the processes involved in an MPI job.
:p What is the purpose of communicators in MPI?
??x
The purpose of communicators in MPI is to define groups of processes that can exchange messages and synchronize their actions. The default communicator, `MPI_COMM_WORLD`, includes all processes participating in a parallel job.

```c
iret = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
```

x??

---

#### Process Definition
Background context: In the context of MPI, a process is an independent unit of computation that has its own memory space and can communicate with other processes through messages.
:p What defines a process in MPI?
??x
A process in MPI is defined as an independent unit of computation that owns a portion of memory and controls resources in user space. It can initiate computations and send/receive messages to/from other processes.

```c
// Example pseudocode for a process in MPI
void main() {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of this process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the total number of processes
    // Process-specific computations here...
    MPI_Finalize();
}
```

x??

#### MPI Compiler Command Line Options
Background context explaining that the command line options for `mpicc`, `mpicxx`, and `mpifort` differ based on the MPI implementation. Specifically, we will discuss OpenMPI and MPICH.

:p What are the different command-line options provided by `man mpicc` for MPI implementations like OpenMPI and MPICH?
??x
The `man mpicc` command provides specific command-line options for initializing MPI with `mpicc`, `mpicxx`, or `mpifort`. For example, in OpenMPI, you can use:

```sh
mpicc --showme:compile --showme:link
```

For MPICH, the equivalent commands would be:

```sh
mpicc -show -compile_info -link_info
```

These options help users understand how to compile and link their MPI programs. They provide insights into the compiler and linker flags required for MPI.
x??

---

#### Parallel Startup Commands
Background context explaining that parallel processes in MPI are typically started using a special command like `mpirun` or `mpiexec`. However, there is no standardization across implementations.

:p List some of the startup commands used for running an MPI program.
??x
The startup commands for running an MPI program can vary depending on the MPI implementation. Commonly used ones include:

- `mpirun -n <nprocs>`
- `mpiexec -n <nprocs>`

Other variations might include:
- `aprun`
- `srun`

These commands typically take a `-n` or `-np` option to specify the number of processes (`<nprocs>`).

The exact options and their usage can vary between different MPI implementations.
x??

---

#### Minimum Working Example (MWE) of an MPI Program
Background context explaining that the MWE demonstrates the basic structure of an MPI program, including initialization, communication, and finalization.

:p What is the purpose of the `MPI_Init` function in a typical MPI program?
??x
The `MPI_Init` function initializes the MPI environment. It must be called before any other MPI functions can be used within your program. This function sets up the necessary context for parallel execution.

```c
int MPI_Init(int *argc, char ***argv);
```

It takes two arguments:
- `*argc`: A pointer to an integer that stores the number of command-line arguments.
- `**argv`: A double pointer to a string array containing the command-line arguments.

:p What does the `MPI_Comm_rank` function do in an MPI program?
??x
The `MPI_Comm_rank` function retrieves the rank of the calling process within its communicator. In most cases, this communicator is `MPI_COMM_WORLD`, which contains all processes involved in the parallel job.

```c
int MPI_Comm_rank(MPI_Comm comm, int *rank);
```

It takes:
- `comm`: The communicator (usually `MPI_COMM_WORLD`).
- `rank`: A pointer to an integer that will receive the rank of the process.

:p What does the `MPI_Comm_size` function do in an MPI program?
??x
The `MPI_Comm_size` function retrieves the number of processes in a given communicator. This is typically used with `MPI_COMM_WORLD`, which contains all processes involved in the parallel job.

```c
int MPI_Comm_size(MPI_Comm comm, int *size);
```

It takes:
- `comm`: The communicator (usually `MPI_COMM_WORLD`).
- `size`: A pointer to an integer that will receive the number of processes.

:p What is the purpose of `MPI_Finalize` in a typical MPI program?
??x
The `MPI_Finalize` function cleans up and terminates the MPI environment. It ensures that all resources are properly released, and it waits for all processes to reach this point before exiting. This is crucial for proper cleanup after parallel execution.

```c
int MPI_Finalize(void);
```

:p What does the line `printf("Rank %d of %d ", rank, nprocs);` do in the example program?
??x
This line prints a message that includes the rank of each process and the total number of processes. The format string `"Rank %d of %d "` is used to print:

- `%d`: The integer value stored in `rank`.
- `%d`: The integer value stored in `nprocs`.

:p How does one compile an MPI program using a simple makefile?
??x
To compile an MPI program, you can use a simple makefile that specifies the correct compiler wrapper. For example:

```makefile
MinWorkExampleMPI.c: MinWorkExampleMPI.c Makefile
    mpicc MinWorkExampleMPI.c -o MinWorkExampleMPI

clean:
    rm -f MinWorkExampleMPI MinWorkExampleMPI.o
```

This makefile uses `mpicc` to compile the C file into an executable. The `-o` flag specifies the output filename.

:p How can one use CMake to build the example program?
??x
Using CMake to build an MPI program involves creating a `CMakeLists.txt` file that sets up the project and finds the MPI library:

```cmake
cmake_minimum_required(VERSION 2.8)
project(MinWorkExampleMPI)

# Require MPI for this project:
find_package(MPI REQUIRED)
add_executable(MinWorkExampleMPI MinWorkExampleMPI.c)
```

This CMake script ensures that the MPI libraries are found and linked correctly during compilation.
x??

---

