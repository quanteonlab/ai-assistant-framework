# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 58)

**Starting Chapter:** 17.7 Bug-busters Debuggers to exterminate those bugs. 17.7.1 TotalView debugger is widely available at HPC sites. 17.7.3 Linux debuggers Free alternatives for your local development needs

---

#### TotalView Debugger Overview
Background context: The TotalView debugger is a powerful and easy-to-use tool widely available at high-performance computing (HPC) sites. It supports extensive features for debugging applications written in languages such as C, C++, Fortran, Python, and others that use MPI and OpenMP threading. Additionally, it provides support for debugging NVIDIA GPUs using CUDA.

:p What is TotalView?
??x
TotalView is a debugger used in HPC environments to help developers identify and fix bugs in their applications. It supports various programming languages like C, C++, Fortran, Python, among others, and offers robust features for both sequential and parallel debugging, including support for MPI and OpenMP.

```shell
# Invoking TotalView with an application
totalview mpirun -a -n 4 <my_application>
```
x??

#### DDT Debugger Overview
Background context: ARM DDT (Dynamic Debugging Technology) is another popular commercial debugger used at HPC sites. It supports extensive features for debugging applications that use MPI and OpenMP, as well as some support for CUDA code. The DDT debugger offers a user-friendly graphical interface and includes remote debugging capabilities.

:p What is DDT?
??x
DDT (Dynamic Debugging Technology) is a commercial debugger used in HPC environments to assist developers with finding and resolving bugs. It supports multiple programming standards such as MPI, OpenMP, and CUDA, and provides an intuitive graphical user interface for easier navigation during the debugging process.

```shell
# Invoking DDT with an application
ddt <my_application>
```
x??

#### TotalView Usage Example
Background context: TotalView can be invoked by prefixing the command line with `totalview` followed by additional arguments. The `-a` flag indicates that the rest of the arguments are to be passed directly to the application.

:p How do you invoke TotalView with an MPI application?
??x
You would use the following command format to invoke TotalView with a parallel (MPI) application:
```shell
totalview mpirun -a -n 4 <my_application>
```
Here, `mpirun` is used to launch the MPI program with 4 processes. The `-a` flag tells TotalView that the subsequent arguments should be passed directly to the application.

x??

#### DDT Usage Example
Background context: DDT can also be invoked by prefixing the command line with `ddt`. It provides a graphical user interface and supports remote debugging, allowing developers to run the application on remote HPC systems while controlling it from their local system.

:p How do you invoke DDT for an MPI application?
??x
You would use the following command format to invoke DDT with an MPI application:
```shell
ddt <my_application>
```
Here, `ddt` is used to start the debugger session. The `<my_application>` part should be replaced with the actual name or path of your MPI application.

x??

#### TotalView and HPC Sites
Background context: At high-performance computing sites, TotalView is widely available and supports leading HPC systems, including those that use MPI and OpenMP for parallel programming. It also has some support for CUDA debugging, making it a valuable tool for developers working on complex applications.

:p Where can you find TotalView tutorials at an HPC site?
??x
At high-performance computing sites, such as Lawrence Livermore National Laboratory (LLNL), TotalView tutorials are available. You can access them via the following link:
```
https://computing.llnl.gov/tutorials/totalview/
```

Additionally, detailed information is available on the TotalView website at: 
```
https://totalview.io
```

x??

#### DDT and HPC Sites
Background context: Similarly, ARM DDT is another popular debugger used in HPC sites. It supports MPI and OpenMP for debugging parallel applications and has some support for CUDA code. The DDT debugger also offers remote debugging capabilities.

:p Where can you find an introduction to DDT at an HPC site?
??x
At high-performance computing sites, such as the Texas Advanced Computing Center (TACC), there is a good introduction to DDT available. You can access it via this link:
```
https://www.tacc.utexas.edu/research-support/software-installations/ddt
```

For more detailed information, you can visit the official DDT website at:
```
https://www.ddt-debugger.com/
```

x??

#### ARM DDT Debugger Tutorials
Background context: The ARM DDT debugger is a powerful tool for debugging applications on ARM-based systems. It can be used with various tools and environments, such as those provided by the Texas Advanced Computing Center (TACC).

:p What are some resources available for learning about the ARM DDT Debugger?
??x
The ARM DDT Debugger tutorials from TACC provide an in-depth understanding of how to use this debugger effectively. You can find these resources at https://portal.tacc.utexas.edu/tutorials/ddt and learn about setting up your environment, using the debugger commands, and debugging various types of applications.

```bash
# Example command to access ARM DDT tutorials
wget https://portal.tacc.utexas.edu/tutorials/ddt
```
x??

---

#### ARM DDT (ARM Forge)
Background context: The ARM DDT Debugger is part of a suite called ARM Forge, which includes several tools for software development and debugging. Understanding the features and usage of ARM Forge can enhance your debugging capabilities.

:p What is ARM Forge?
??x
ARM Forge is a collection of development tools provided by ARM that includes the ARM DDT Debugger among others. These tools are designed to support various aspects of software development, including debugging, profiling, and performance tuning on ARM-based systems. The key feature of ARM Forge is its comprehensive set of tools tailored for developers working with ARM architectures.

```bash
# Example command to access ARM Forge documentation
wget https://www.arm.com/products/development-tools/server-and-hpc/forge/ddt
```
x??

---

#### Linux Debuggers: GDB
Background context: GDB, the GNU Debugger, is a widely used tool for debugging applications on Linux systems. While it has a command-line interface that can be complex to use initially, there are various graphical user interfaces (GUIs) and higher-level tools built on top of GDB to make debugging easier.

:p What is the basic command to run GDB on a serial application?
??x
The basic command to start debugging a serial application with GDB is:
```bash
gdb <my_application>
```
This command launches GDB, which then attaches to your application. You can use various commands within GDB such as `run`, `step`, and `break` to control the execution of your program.

x??

---

#### cgdb - The Curses-based Debugger
Background context: cgdb is a curses-based interface for GDB that provides a more user-friendly experience compared to GDB's command-line interface. It offers features similar to those in popular text editors, making it easier to navigate and debug code.

:p What does the `cgdb` tool do?
??x
The `cgdb` tool is a curses-based debugger that runs on top of GDB, providing an interactive environment for debugging C programs. Its interface is designed with ease of use in mind, offering features such as line number navigation, source code highlighting, and breakpoints management.

To launch `cgdb`, you can use:
```bash
mpirun -np 4 xterm -e gdb ./<my_application>
```
This command runs multiple GDB sessions in separate terminals. However, the `cgdb` tool simplifies this process by providing a curses-based interface directly within your terminal.

x??

---

#### DDD - DataDisplayDebugger
Background context: DDD is a graphical debugger that provides an easy-to-use interface for debugging applications on Linux systems. It offers advanced features like data visualization and network performance optimizations, making it suitable for complex debugging tasks.

:p What is the purpose of `DDD`?
??x
The purpose of DDD (DataDisplayDebugger) is to provide users with a graphical user interface for debugging their applications. Unlike GDB's command-line interface, DDD offers a more intuitive way to navigate through code and inspect variables.

To start `DDD`, you can use the command:
```bash
ddd --debugger cuda-gdb
```
This launches DDD with `cuda-gdb` as its backend debugger. You can then interact with your application via a graphical interface, making it easier to analyze complex data structures and manage breakpoints.

x??

---

#### CUDA-GDB - A Debugger for NVIDIA GPU
Background context: CUDA-GDB is a command-line debugger based on GDB that enables debugging applications written in CUDA, which are used for parallel processing on NVIDIA GPUs. This tool significantly enhances the debugging capabilities of developers working with GPU-accelerated applications.

:p What is `CUDA-GDB`?
??x
`CUDA-GDB` is a debugger specifically designed for CUDA applications running on NVIDIA GPUs. It extends GDB's functionality to support debugging tasks related to GPU programming, including memory management and thread synchronization issues.

To use `CUDA-GDB`, you can launch it with DDD:
```bash
ddd --debugger cuda-gdb
```
This command integrates the powerful features of DDD with the CUDA-specific capabilities provided by `cuda-gdb`.

x??

---

#### ROC GDB - A Debugger for AMD GPUs
Background context: ROC GDB is a debugger part of the AMD ROCm initiative, which offers support for debugging applications using AMD GPUs. It is built on top of GDB and provides initial support for AMD's GPU architectures.

:p What does `ROC GDB` offer?
??x
`ROC GDB` (Radeon Open Compute Debugger) is a debugger designed for AMD GPUs, providing an interface similar to GDB but with specific optimizations for AMD hardware. It supports debugging applications that use the ROCm framework.

To start `ROC GDB`, you can use:
```bash
darshan-job-summary.pl <darshan log file>
```
This command runs the analysis tool on a Darshan log file, generating detailed reports about your application's I/O operations and performance metrics.

x??

---

#### Darshan - An HPC I/O Characterization Tool
Background context: Darshan is a profiling tool designed for high-performance computing (HPC) applications. It characterizes filesystem usage by tracking I/O patterns and other relevant metrics, helping developers optimize their file operations and improve application performance.

:p What does `Darshan` do?
??x
`Darshan` is an HPC I/O characterization tool that measures and reports on the I/O operations performed by high-performance computing applications. It provides detailed insights into how your application interacts with the filesystem, including read/write patterns and metadata usage.

To install Darshan in your home directory:
```bash
wget ftp://ftp.mcs.anl.gov/pub/darshan/releases/darshan-3.2.1.tar.gz
tar -xvf darshan-3.2.1.tar.gz
```
This command downloads the Darshan distribution and extracts it to your local machine.

x??

#### POSIX and MPI-IO Profiling
POSIX stands for Portable Operating System Interface, which is a standard for portability of system-level functions such as regular filesystem operations. In our modified test, we focused on MPI-IO parts by turning off all verification and standard IO operations to isolate the performance of the MPI-IO components.

:p What is POSIX used for in the context of file operations?
??x
POSIX provides a set of standards that ensure portability across different operating systems for common tasks like file I/O. In our test, we used it to maintain consistency in how filesystem operations were handled.
x??

---

#### MPI-IO Write vs Read Performance
In our tests using the NFS filesystem, we observed that an MPI-IO write operation was slightly slower than a read operation. This is because writing metadata (information about file location, permissions, access times) is inherently serial and incurs additional overhead.

:p Why might an MPI-IO write be slower than an MPI-IO read?
??x
An MPI-IO write can be slower due to the necessity of writing metadata, which involves a serial operation that adds extra overhead. The read operation does not require this additional step.
x??

---

#### Darshan I/O Profiling Tool
Darshan is an HPC (High Performance Computing) tool designed for profiling and characterizing I/O operations. It supports both POSIX and MPI-IO operations and can provide detailed insights into file access patterns.

:p What is Darshan used for in the context of high-performance computing?
??x
Darshan is a tool used to profile and characterize I/O operations in HPC environments, providing metrics on file access patterns and performance issues.
x??

---

#### Package Managers Overview
Package managers are tools that simplify software installation on various systems. They manage software packages across different distributions and can keep the system more stable and up-to-date by handling dependencies automatically.

:p What is a package manager used for?
??x
A package manager simplifies software installation, updates, and dependency management on operating systems like Linux or macOS.
x??

---

#### Package Managers in Linux
Linux relies heavily on package managers. Common formats include Debian (.deb) and RPM (Red Hat Package Manager). These packages can be installed using tools like `apt` for Debian-based distributions or `yum`/`dnf` for Red Hat-based systems.

:p What are common package manager formats for Linux?
??x
Common package manager formats for Linux include `.deb` for Debian-based systems and `.rpm` for Red Hat-based systems.
x??

---

#### Homebrew vs MacPorts on macOS
Homebrew and MacPorts are two prominent package managers for macOS. They allow the installation of many open-source tools, but recent changes in macOS have led to some compatibility issues.

:p What are the main package managers for macOS?
??x
The main package managers for macOS are Homebrew and MacPorts.
x??

---

#### Example of Using Homebrew (Code)
To install a package using Homebrew on macOS, you can use the following command:
```sh
# Install a package with Homebrew
brew install <package_name>
```

:p How do you install a package using Homebrew?
??x
You can install a package using Homebrew by running the `brew install` command followed by the name of the package.
x??

---

