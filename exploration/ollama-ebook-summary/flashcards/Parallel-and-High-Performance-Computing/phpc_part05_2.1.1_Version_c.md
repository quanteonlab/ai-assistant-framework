# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 5)

**Starting Chapter:** 2.1.1 Version control Creating a safety vault for your parallel code

---

---
#### Planning Steps for a Parallel Project
The chapter emphasizes the importance of a structured approach to parallelizing applications. It outlines a workflow that helps developers plan, prepare, and implement parallelism incrementally.

:p What are the main steps involved in planning a parallel project?
??x
The main steps involve setting up version control, developing a test suite for the application, cleaning up existing code, and preparing the team and the application for rapid development. Each step is crucial to ensure that the parallelization process is manageable and successful.

For example, setting up version control (like Git) helps track changes and revert them if necessary:
```bash
git init
git add .
git commit -m "Initial setup"
```

x??
---

#### Version Control and Team Development Workflows
Version control systems are essential for tracking changes in the codebase. They allow developers to manage different versions of their code, making it easier to revert or merge changes.

:p What is version control, and why is it important for parallel projects?
??x
Version control is a system that records changes to a file or set of files over time so that you can recall specific versions later. It is crucial because it helps manage different versions of the codebase, making it easier to revert changes if something goes wrong during the parallelization process.

Example using Git:
```bash
git clone <repository-url>
# Make modifications and commit them
git add .
git commit -m "Parallelization step 1"
```

x??
---

#### Understanding Performance Capabilities and Limitations
Understanding the performance characteristics of your application is critical before attempting to parallelize it. This includes knowing where bottlenecks are and what hardware limitations might affect performance.

:p What should developers understand about their application’s performance before starting a parallelization project?
??x
Developers need to understand the current performance capabilities and limitations of their application, including identifying potential bottlenecks and assessing hardware constraints that could limit performance gains from parallelism. This understanding helps in making informed decisions on where and how to apply parallel techniques.

For example, profiling tools can be used to identify slow parts:
```java
public class Profiler {
    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        // Code snippet that might need optimization
        long duration = System.currentTimeMillis() - start;
        System.out.println("Duration: " + duration + " ms");
    }
}
```

x??
---

#### Developing a Plan to Parallelize a Routine
The workflow involves repeating four steps (Profile, Plan, Commit, Implement) to incrementally parallelize an application. This approach is particularly suited for agile project management techniques.

:p What are the four main steps in the suggested parallel development workflow?
??x
The four main steps in the suggested parallel development workflow are:
1. **Profile**: Identify areas of the code that can be parallelized.
2. **Plan**: Develop a plan to parallelize these routines.
3. **Commit**: Implement small, incremental changes and commit them.
4. **Implement**: Continue implementing the plan with frequent tests.

Example pseudocode for profiling and planning:
```java
public class ParallelizationWorkflow {
    public static void main(String[] args) {
        // Profile step
        profileRoutine();
        
        // Plan step
        planParallelRoutines();
        
        // Commit step
        commitChanges();
        
        // Implement step
        implementPlan();
    }
    
    private static void profileRoutine() {
        // Code to identify bottlenecks and areas for parallelization
    }
    
    private static void planParallelRoutines() {
        // Code to develop a detailed plan for parallelism
    }
    
    private static void commitChanges() {
        // Code to implement small, incremental changes
    }
    
    private static void implementPlan() {
        // Code to fully implement the plan with testing
    }
}
```

x??
---

#### Incremental Parallelization in Agile Projects
Implementing parallelism in small increments is beneficial as it allows for easy troubleshooting and rollbacks. This approach fits well within agile project management techniques.

:p Why should developers implement parallelism in small increments rather than all at once?
??x
Developers should implement parallelism in small increments to ensure that any issues can be easily identified and resolved, minimizing the risk of a complete failure during the implementation phase. Incremental changes also allow for continuous testing and validation, ensuring that the application's functionality is maintained.

Example of committing changes:
```bash
git commit -am "Parallelized step 1 with tests"
```

x??
---

#### Determining Computing Resources Capabilities
Background context: Before starting a project, it's crucial to assess the computing resources available and understand their limitations. This involves benchmarking the system to determine its compute capabilities.

:p What is the first step in preparing for developing parallel applications?
??x
The first step in preparing for developing parallel applications is determining the capabilities of the computing resources available. This includes understanding the hardware limitations, such as CPU speed, memory capacity, and I/O throughput, which can impact performance.

```java
// Example of a simple benchmarking function to measure system performance
public class Benchmark {
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        
        // Perform some intensive computation here
        
        long endTime = System.currentTimeMillis();
        double timeTaken = (endTime - startTime);
        System.out.println("Time taken: " + timeTaken + " ms");
    }
}
```
x??

---

#### Application Profiling
Background context: Understanding the application's demands is essential for identifying bottlenecks and optimizing performance. Profiling helps in understanding how the application uses computational resources, particularly focusing on computationally intensive kernels.

:p What does profiling an application help you understand?
??x
Profiling an application helps you understand its operational characteristics, such as which sections of code are most resource-intensive (compute-bound), what parts consume significant time or resources, and where potential performance bottlenecks exist. Identifying these "expensive" computational kernels is critical for optimization.

```java
// Example of a profiling tool usage in Java using VisualVM
public class ProfilerExample {
    public static void main(String[] args) throws Exception {
        ProcessBuilder process = new ProcessBuilder("jvisualvm", "--trace-start");
        process.start();
        
        // Run the target application to be profiled here
        
        // Stop tracing and analyze the results in VisualVM interface
    }
}
```
x??

---

#### Parallelizing Routines
Background context: After profiling, you identify which parts of your code are computationally intensive (kernels) and plan how these sections can be parallelized. This involves breaking down tasks that can run concurrently to improve performance.

:p What is the process of identifying computational kernels in an application?
??x
Identifying computational kernels in an application involves analyzing the code to find sections that are both computationally intensive and conceptually self-contained. These kernels often represent opportunities for parallelization, as they can be executed independently or concurrently without affecting other parts of the program.

```java
// Example pseudocode for identifying a kernel
public void identifyKernels() {
    List<Kernel> kernels = new ArrayList<>();
    
    for (int i = 0; i < numberOfOperations; i++) {
        if (isComputationallyIntensive(i) && isSelfContained(i)) {
            Kernel kernel = new Kernel();
            // Set up the kernel properties
            kernels.add(kernel);
        }
    }
}
```
x??

---

#### Committing Changes to Version Control
Background context: Once the parallelization and optimization are complete, changes need to be committed to a version control system. This ensures that modifications are tracked and can be reverted if necessary.

:p What is the importance of committing changes to a version control system?
??x
Committing changes to a version control system is essential for maintaining a history of modifications, allowing you to track changes over time and revert to previous versions if needed. This practice facilitates collaboration among team members and helps in managing the project's evolution more effectively.

```bash
# Example command sequence for committing changes using Git
git add .
git commit -m "Parallelized kernel X and optimized performance"
git push origin master
```
x??

---

#### Ensuring Code Quality with Modularity
Background context: Good code is modular, meaning it is composed of independent subroutines or functions that have well-defined inputs and outputs. This approach helps in making the code easy to modify and extend.

:p What does modularity mean in the context of coding?
??x
Modularity means implementing kernels as independent subroutines or functions with clear and well-defined input and output parameters. Each module should be self-contained, performing a specific task without relying on other modules' internal implementation details.

```java
// Example of a modular function for a kernel operation
public class KernelModule {
    public int process(int data) {
        // Perform the computation here
        return result;
    }
}
```
x??

---

#### Ensuring Code Portability
Background context: Code portability ensures that your application can be compiled and run on multiple platforms. This is crucial for adapting to changing hardware and software environments.

:p Why is code portability important?
??x
Code portability is important because it allows your application to be compiled and executed across different platforms, including various operating systems and hardware architectures. Ensuring portability helps in maintaining flexibility and reducing dependency on specific development environments.

```cpp
// Example of a portable function using #ifdef for conditional compilation
void process(int data) {
#ifdef WINDOWS
    // Windows-specific code here
#else
    // Unix/Linux-specific code here
#endif
}
```
x??

---

#### Version Control Overview
Background context: The importance of version control is highlighted when working on parallelism tasks, as it allows for recovery from broken or problematic code versions. This is particularly crucial in scenarios where rapid changes and small commits are common.

:p What is version control and why is it important?
??x
Version control is a system that records changes to a file or set of files over time so that you can recall specific versions later. It helps manage different versions of source code during software development, ensuring that developers can revert to previous working states if needed. This is especially critical in parallelism tasks where small and frequent changes occur.

In the context of our ash plume model project, version control ensures that there is a record of every change made by each developer, allowing for easier collaboration and troubleshooting when issues arise.

For example, suppose you have multiple developers working on the same codebase. Version control helps them keep track of their contributions and revert to previous versions if necessary.
??x

---

#### Pull Request Model
Background context: The pull request (PR) model involves changes being posted for review by other team members before they are committed to the main repository. This process is often used to ensure quality checks and collective code ownership.

:p What is a pull request model in version control?
??x
A pull request model is a workflow where developers submit their changes for review by other members of the development team before those changes are merged into the main branch or repository. This approach promotes code quality, collaborative coding practices, and ensures that all team members have visibility into ongoing changes.

For example:
```java
// Pseudocode illustrating a pull request process
public class PullRequest {
    public void sendPullRequest(String commitMessage) {
        // Send the PR to the team for review
        System.out.println("Sending pull request with message: " + commitMessage);
    }
}
```
The `sendPullRequest` method simulates sending a pull request, which would be reviewed by other developers before merging.

??x

---

#### Push Model
Background context: The push model allows developers to make commits directly to the repository without prior review. This is often used in scenarios where rapid and frequent changes are common.

:p What is the push model in version control?
??x
The push model of version control allows developers to commit their changes directly to the main repository without prior review by other team members. This approach can be more suitable for tasks that require quick iteration and small, frequent commits.

For example:
```java
// Pseudocode illustrating a push model process
public class PushModel {
    public void commitChanges(String commitMessage) {
        // Commit changes directly to the repository without a review
        System.out.println("Committing changes with message: " + commitMessage);
    }
}
```
The `commitChanges` method simulates committing changes directly, bypassing any intermediate review process.

??x

---

#### Git as a Version Control System
Background context: Git is the most common distributed version control system and is recommended for managing parallel code development. It allows multiple repository databases and is advantageous in open-source projects or remote work environments.

:p What is Git and why is it important?
??x
Git is a distributed version control system that enables developers to manage changes to their source code across multiple branches, facilitating collaboration among team members. It supports a decentralized approach where each developer has a complete copy of the repository, allowing for flexibility in working conditions like remote environments.

For example:
```java
// Pseudocode illustrating Git commands
public class GitCommands {
    public void initializeRepository() {
        // Initialize a new git repository
        System.out.println("Initializing git repository...");
    }

    public void commitCode(String commitMessage) {
        // Commit changes to the local repository
        System.out.println("Committing code with message: " + commitMessage);
    }
}
```
The `initializeRepository` and `commitCode` methods demonstrate basic Git commands for setting up a repository and committing changes.

??x

---

#### Commit Messages
Background context: Commit messages provide detailed information about the changes made, helping developers understand why certain modifications were necessary. They are crucial for maintaining clarity in the codebase.

:p What is the importance of commit messages?
??x
Commit messages are essential for documenting the purpose and reasoning behind each change made to the codebase. They help maintain a clear history that can be reviewed by team members, improving transparency and facilitating collaboration.

For example:
```java
// Example of a detailed commit message in Git
public class CommitMessage {
    public void createCommit(String message) {
        // Create a git commit with a detailed message
        System.out.println("Creating commit: " + message);
    }
}
```
The `createCommit` method simulates creating a commit with a detailed message that explains the changes.

??x

---

#### Frequent Commits
Background context: Regular commits are recommended to avoid losing work or encountering issues later. This practice helps in maintaining a clean and understandable code history, making it easier to revert to previous states if necessary.

:p Why is frequent committing important?
??x
Frequent committing is crucial because it prevents loss of work and makes it easier to identify the source of issues when they arise. By committing regularly, developers can maintain a clear and organized version history, which is essential for debugging and collaboration.

For example:
```java
// Pseudocode illustrating frequent commits
public class FrequentCommits {
    public void commitAfterEachChange(String changeDescription) {
        // Commit after each small change
        System.out.println("Committing: " + changeDescription);
    }
}
```
The `commitAfterEachChange` method demonstrates committing after making a small change, ensuring that the code history is well-documented and easy to trace.

??x

---

#### Version Control Commit Messages
Version control systems like Git are essential for managing changes to software projects. A commit message is a tool used to document and communicate the purpose of your recent changes. Properly structured commit messages can significantly improve team collaboration, especially when multiple developers are working on the same codebase.

In general, a good commit message includes two parts: a summary line and a body.
- The **summary** (first line) should be concise and descriptive, summarizing what was changed.
- The **body** provides additional context such as why the change was made and any related details.
If your project uses an issue tracking system like Jira or GitHub Issues, it's good practice to include the issue number in the summary for reference.

Good commit messages not only summarize changes but also explain the reasoning behind them. This makes it easier for other developers to understand the purpose of the change without having to go through the code diff.

:p What is the structure of a good commit message?
??x
A good commit message typically follows this structure:
- **Summary line**: A concise description of what was changed (first line).
- **Body**: Additional details about why and how the changes were made.
If you are using an issue tracking system, include the issue number in the summary.

Example:
```
[Issue #21] Fixed the race condition in the OpenMP version of the blur operator.
The race condition was causing non-reproducible results amongst GCC, Intel, and PGI compilers. 
To fix this, an OMP BARRIER was introduced to force threads to synchronize just before calculating
the weighted stencil sum. Confirmed that the code builds and runs with GCC, Intel, and PGI compilers
and produces consistent results.
```
x??

---

#### Test Suites in Parallel Applications
A test suite is a collection of tests designed to verify that different parts of an application are working correctly, particularly useful for ensuring robustness and reliability. In parallel applications, where the order of operations can be altered due to scheduling and threading, it's crucial to have comprehensive test suites.

Test suites help identify whether changes in code lead to unintended side effects or bugs. They ensure that your software behaves as expected even when executed on different hardware configurations (e.g., different numbers of processors).

:p What is a test suite in the context of parallel applications?
??x
A test suite in the context of parallel applications is a set of problems designed to exercise parts of an application and guarantee that related code segments still work correctly. These tests are essential for catching bugs early, especially when dealing with the complexities introduced by parallelism.

Test suites help maintain consistency across different hardware configurations and compilers, ensuring reproducibility of results.
x??

---

#### Understanding Changes in Results Due to Parallelism
Parallel computing can introduce variations in numerical results due to differences in the order of operations. While these changes are often minor, they can become significant when comparing outputs from single-threaded runs versus multi-threaded runs.

These discrepancies must be carefully managed and understood because they can mask or reveal issues with parallel code implementation. For example, race conditions and deadlocks might only occur under certain hardware configurations, making them hard to detect in a testing environment.

:p How does parallelism affect the results of numerical computations?
??x
Parallel computing affects numerical computations by inherently changing the order of operations. This can lead to small differences in the results due to variations in how tasks are scheduled and executed across multiple threads or processors.

These changes need to be understood and managed, as they can mask real bugs or introduce new ones if not properly accounted for.
x??

---

#### Krakatau Scenario Test for Validated Results
In some projects, simulation codes generate validated results that are compared against experimental or real-world data. These simulations are considered valuable when their outputs match known data sets.

The Krakatau scenario is an example where a wave simulation application generates results that can be verified against historical data from the 1883 eruption of Krakatoa. Validated results ensure that the code produces accurate and reliable output, which is crucial for the credibility of the research or product.

:p What are validated results in simulations?
??x
Validated results in simulations refer to simulation outputs that are compared against experimental or real-world data. These comparisons serve as a form of quality assurance, ensuring that the software performs accurately under known conditions.

For instance, in wave simulation applications, validated results can be compared to historical data from events like the 1883 eruption of Krakatoa. This process helps verify the correctness and reliability of the code.
x??

---
#### Compiler Variations and Parallelism
Background context: In the given scenario, two different compilers (GCC and Intel) are used for development and production. The variations in output due to these differences can include compiler version changes, hardware changes, compiler optimizations, and differences in the order of operations due to parallelism.
:p How do variations in compilers affect the program's output?
??x
Compiler differences such as versions or types (GCC vs Intel) can lead to slight variations in the program’s output. This is because different compilers may optimize code differently and handle parallelization in unique ways, affecting the final results.

For example:
- The same source code might be optimized differently by GCC and Intel compilers.
- Parallel processing can yield varying results due to differences in how tasks are distributed among processors.
??x
The variations are acceptable as long as they do not significantly impact the overall correctness or performance of the application. For critical applications, it is recommended to validate the output across different compiler versions and configurations.

```cpp
// Example C++ code snippet
int main() {
    int waveHeight = 4;
    double totalMass = 293548;

    // Parallel processing might yield slightly different results due to compiler optimizations.
    #pragma omp parallel for
    for (int i = 0; i < 1000000; ++i) {
        waveHeight += i % 2;
        totalMass += i * 0.001; // Small increment for demonstration
    }

    printf("Wave Height: %.8f, Total Mass: %.4f\n", waveHeight, totalMass);
}
```
x??

---
#### Numerical Diff Utilities
Background context: When comparing numerical outputs across different runs or compilers, tools like `numdiff` and `ndiff` can be used to identify small differences. These utilities allow specifying a tolerance for acceptable variation.
:p What are some numerical diff utilities available for comparing program outputs?
??x
Numerical diff utilities such as `numdiff` from nongnu.org/numdiff/ and `ndiff` from www.math.utah.edu/~beebe/software/ndiff/ can be used to compare the outputs of different runs or compilers. These tools help in identifying small numerical differences within a specified tolerance.

For example, using `numdiff`:
```bash
numdiff -s 1e-6 output_file1.txt output_file2.txt
```
This command compares two files with a tolerance of 1e-6 and suppresses insignificant differences.
??x
These utilities are particularly useful in scientific computing where small numerical discrepancies can be significant. By setting an appropriate tolerance, you can ensure that minor variations due to different compiler versions or hardware configurations do not falsely indicate errors.

```bash
// Example command using numdiff
numdiff -s 1e-6 result_gcc.txt result_intel.txt > diff_report.txt
```
This command generates a report of differences between `result_gcc.txt` and `result_intel.txt`, allowing you to see which values differ beyond the specified tolerance.
x??

---
#### HDF5 and NetCDF for Data Comparison
Background context: HDF5 and NetCDF are binary data formats used in scientific computing. These formats allow storing large datasets efficiently. Tools like `h5diff` can be used to compare files with a certain numeric tolerance, ensuring that small numerical differences do not indicate errors.
:p How can HDF5 or NetCDF files be compared for numerical differences?
??x
HDF5 and NetCDF files can be compared using the `h5diff` utility. This tool allows comparing two files and reporting the differences above a specified numeric tolerance.

For example, to compare two HDF5 files:
```bash
h5diff -t 1e-6 file1.h5 file2.h5 > diff_report.txt
```
This command compares `file1.h5` and `file2.h5`, considering numerical differences greater than the tolerance of \(1 \times 10^{-6}\) as significant.

Similarly, NetCDF files can be compared using a similar approach:
```bash
ncdiff -t 1e-6 file1.nc file2.nc > diff_report.txt
```
This command compares `file1.nc` and `file2.nc`, considering numerical differences greater than the tolerance of \(1 \times 10^{-6}\) as significant.
??x
Using these tools, you can ensure that small numerical variations do not indicate actual errors. This is particularly useful in parallel computing where minor discrepancies might arise due to different compiler optimizations or hardware configurations.

```bash
// Example command using h5diff and ncdiff
h5diff -t 1e-6 result_gcc.h5 result_intel.h5 > diff_report.txt
ncdiff -t 1e-6 result_gcc.nc result_intel.nc > diff_report.txt
```
These commands generate reports that highlight significant differences, helping in understanding the impact of compiler and hardware variations.
x??

---

#### Introduction to CMake and CTest
Background context explaining that CMake is a configuration system used for generating platform-specific build files, and it integrates well with testing frameworks like CTest. CMake simplifies the process of building applications on different platforms by adapting generated makefiles.

CTest is part of the CMake suite and provides an easy way to write test cases as sequences of commands and integrate them into a project.

:p What are CMake and CTest used for?
??x
CMake is a cross-platform open-source build system that generates native build files. It simplifies the configuration process by adapting makefiles to different systems and compilers. CTest, on the other hand, is part of the CMake suite and provides a testing framework for running tests as part of the build process.

```cmake
# Example CMakeLists.txt snippet
enable_testing()
add_test(name_of_test executable_name arguments)
```
x??

---

#### Enabling Testing with CMake
The `enable_testing()` command must be included in the CMakeLists.txt file to enable testing capabilities. This sets up the framework for running tests.

:p How do you enable testing in a CMake project?
??x
To enable testing, you include the line `enable_testing()` in your CMakeLists.txt file. This step is crucial as it initializes the necessary infrastructure within CMake to support test execution.

```cmake
# Example CMakeLists.txt snippet for enabling tests
enable_testing()
```
x??

---

#### Adding Tests with CTest
The `add_test` command adds a specific test case to your project by specifying the name of the test and the executable along with any necessary arguments. This integrates the test directly into the build process.

:p How do you add a test using CTest?
??x
To add a test, you use the `add_test` command in your CMakeLists.txt file. You specify the name of the test, the executable to run, and any required arguments for that executable.

```cmake
# Example CMakeLists.txt snippet for adding tests
add_test(name_of_test executable_name arguments)
```
x??

---

#### Invoking Tests with CTest
You can invoke tests using commands like `make test` or simply `ctest`. Additionally, you can select specific tests by using a regular expression with the `-R` option. For example, `ctest -R pattern` will run all tests matching the given pattern.

:p How do you run individual tests in CMake?
??x
You can run individual tests by specifying a regular expression with the `-R` flag when invoking `ctest`. This command allows running only the tests that match the specified pattern. For example, to run all tests related to "mpi", use:

```bash
$ ctest -R mpi
```
x??

---

#### Example of CTest and ndiff Usage
The provided text describes an example where a simple test is created using both C and MPI programs. The `TimeIt.c` program uses the `clock_gettime` function to measure elapsed time, while the `MPITimeIt.c` uses MPI to achieve parallel execution.

:p What are the two source files used in this testing example?
??x
The two source files used in this testing example are:

- `TimeIt.c`: A simple C program that measures the elapsed time using a sleep function.
- `MPITimeIt.c`: An MPI program that measures the elapsed time in parallel.

```c
// Example TimeIt.c snippet
#include <unistd.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char *argv[]) {
    struct timespec tstart, tstop, tresult;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    sleep(10); // Sleep for 10 seconds
    clock_gettime(CLOCK_MONOTONIC, &tstop);
    tresult.tv_sec = (tstop.tv_sec - tstart.tv_sec);
    tresult.tv_usec = (tstop.tv_nsec - tstart.tv_nsec) * 1e-3; // Convert nanoseconds to microseconds
    printf("Elapsed time is %f secs\n", (double)tresult.tv_sec + (double)tresult.tv_usec / 1000000.0);
}
```

```c
// Example MPITimeIt.c snippet
#include <unistd.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int mype;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
}
```
x??

---

#### Testing Prerequisites
The text mentions that specific tools and versions are required to run the example provided. These include OpenMPI 4.0.0 for MPI, CMake 3.13.3 (which includes CTest), GCC version 8, ndiff for comparing output files.

:p What tools and versions are needed for this testing example?
??x
For running the example in the text, you need:

- OpenMPI 4.0.0: For message passing interface support.
- CMake 3.13.3 (includes CTest): To manage project configuration and build processes.
- GCC version 8: As the default compiler on macOS; it can be different on Ubuntu.

Additional tools include:
- ndiff: A tool for comparing output files, which is installed manually from source code available at https://www.math.utah.edu/~beebe/software/ndiff/.

```bash
# Installation commands for dependencies (example)
$ brew install openmpi cmake gcc # On macOS with Homebrew
$ sudo apt-get install libopenmpi-dev cmake g++ # On Ubuntu with Synaptic
```
x??

---

---
#### Timing Program for Demonstration
The context is a simple timing program used to demonstrate testing functionalities. The program starts by initializing MPI and gets the rank of the current process, then measures the elapsed time using `MPI_Wtime()` before and after a sleep function call.

:p What does the provided C code snippet do?
??x
The code initializes MPI, measures the time taken for a 10-second sleep operation, and prints out the elapsed time from the first processor. It then finalizes MPI.
```c
#include <mpi.h>
#include <stdio.h>
#include <unistd.h> // For sleep function

int main() {
    double t1 = MPI_Wtime();
    sleep(10);             // Sleep for 10 seconds
    double t2 = MPI_Wtime();

    if (mype == 0)         // Assuming mype is defined elsewhere to get the rank
        printf("Elapsed time is %f secs \n", t2 - t1);

    MPI_Finalize();
    return 0;
}
```
x??

---
#### Test Script for Comparing Output Files
This script runs multiple instances of a parallel and serial application, compares their outputs, and sets test statuses based on the differences.

:p What does `ndiff --relative-error` do in this context?
??x
The command `ndiff --relative-error 1.0e-4 run1.out run2.out` is used to compare two files (`run1.out` and `run2.out`) with a relative error tolerance of \(1 \times 10^{-4}\). It returns an exit status code indicating if the difference between the two files falls within the specified tolerance.
x??

---
#### CMakeLists.txt for Building Applications
This section outlines how to use CMake to build the necessary executables and add them to a test suite.

:p What does `enable_testing()` do in a CMake project?
??x
`enable_testing()` is used in CMake projects to enable the testing framework. This allows you to define tests that can be run using tools like CTest, which helps in managing and running multiple test cases.
```cmake
# Enable testing support
enable_testing()
```
x??

---
#### Adding Test Files to CTest Suite
The provided script uses a loop to find all `.ctest` files and add them as test cases to the suite.

:p How does the `file(GLOB ...)` command work in this context?
??x
The `file(GLOB TESTFILES RELATIVE \"${CMAKE_CURRENT_SOURCE_DIR}\" \"*.ctest\")` command gathers all the files ending with `.ctest` relative to the current source directory and assigns them to the variable `TESTFILES`. This is used later to add these test files as CTest tests.
```cmake
# Find all .ctest files in the current directory
file(GLOB TESTFILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.ctest")
```
x??

---
#### CMake Custom Target for Clean-Up
This section explains how to create a custom target to clean up build artifacts.

:p What is the purpose of `add_custom_target(distclean ...)`?
??x
The command `add_custom_target(distclean COMMAND rm -rf CMakeCache.txt CMakeFiles Testing cmake_install.cmake)` creates a custom target named `distclean` that, when executed, removes all the build-related files like `CMakeCache.txt`, `CMakeFiles`, and other artifacts such as `Testing` directory and `cmake_install.cmake`.
```cmake
# Create a custom clean-up target
add_custom_target(distclean
    COMMAND rm -rf CMakeCache.txt CMakeFiles Testing cmake_install.cmake)
```
x??

---

---
#### Running Tests and Managing Output
Background context explaining how to run tests using `mkdir`, `cd`, `cmake`, `make` commands. Mention that `ctest --output-on-failure` can be used to get detailed output for failed tests.

:p How do you run tests in a CMake project?
??x
To run tests in a CMake project, follow these steps:

1. Create a build directory: `mkdir build`
2. Change into the build directory: `cd build`
3. Configure and generate build files: `cmake ..` (the dot indicates the parent directory)
4. Build the project: `make`
5. Run tests using `make test` or simply `ctest`

To get detailed output for failed tests, use `ctest --output-on-failure`.

```bash
mkdir build && cd build
cmake ..
make
make test  # or ctest --output-on-failure
```
x??

---
#### Test Results and Comparison
Background context explaining the importance of comparing outputs between runs to detect changes in the application. Mention storing a gold standard file for comparison.

:p What do you get from running tests on your project?
??x
When you run tests, you typically get a report showing the percentage of tests passed or failed along with detailed output for any failures. For example:

```
Running tests...
Test project /Users/brobey/Programs/RunDiff 
Start 1: mpitest.ctest 1/1 Test #1: mpitest.ctest .................... Passed   30.24 sec
100 percent tests passed, 0 tests failed out of 1 Total Test time (real) =  30.24 sec

This test is based on the sleep function and timers.
```

To store a gold standard file for comparison, you can use tools like `ctest` or write your own scripts to save expected outputs from successful runs.

```bash
ctest --output-on-failure
```
x??

---
#### Custom Commands in CMake
Background context explaining custom commands added in the CMake script. Mention the importance of updating gold standards when new versions are correct.

:p What does a `distclean` command do in CMake?
??x
A `distclean` command in CMake is typically used to clean up any generated files that were created during testing or other operations, ensuring a fresh build environment for each run. This helps maintain the integrity of your tests and ensures that no leftover artifacts from previous builds interfere with current ones.

For example:

```cmake
add_custom_command(TARGET test COMMAND distclean)
```

This command tells CMake to execute `distclean` whenever the target `test` is built, effectively cleaning up any generated files before starting a new build.
x??

---
#### Code Coverage with GCC
Background context explaining code coverage and how to measure it using GCC. Mention that high code coverage is important but more critical for parts of the code being parallelized.

:p How do you generate code coverage statistics with GCC?
??x
To generate code coverage statistics with GCC, follow these steps:

1. Compile your source files with additional flags: `-fprofile-arcs` and `-ftest-coverage`.
2. Run the instrumented executable on a series of tests.
3. Use `gcov` to analyze the coverage.

For CMake projects, you might need to add an extra `.c` extension to handle file naming conventions added by CMake:

```bash
gcc -fprofile-arcs -ftest-coverage source.c -o instrumented_executable
./instrumented_executable

# Run gcov for each file:
gcov <source>.c
```

This will produce a `*.gcov` file containing line-by-line coverage data, showing how many times each line was executed.

For CMake projects:

```bash
gcov CMakeFiles/stream_triad.dir/stream_triad.c.c
```
x??

---
#### Different Kinds of Tests
Background context explaining various types of tests and their purposes. Include a brief description of regression tests, unit tests, continuous integration tests, and commit tests.

:p What are the different kinds of testing systems?
??x
There are several types of testing systems used in software development:

- **Regression Tests**: Run at regular intervals to ensure that no new changes have introduced bugs or broken existing functionality. Typically run nightly or weekly.
  
- **Unit Tests**: Small, isolated tests that verify individual functions or methods work correctly during the development process.

- **Continuous Integration Tests**: Automatically triggered by commits to a code repository. They help catch issues early in the development cycle.

- **Commit Tests**: A small set of tests that can be run quickly from the command line before making commits to ensure changes are stable and functional.

All these types of testing should be used together to provide comprehensive coverage.
x??

---

#### Parallel Application Development Importance
Background context: The importance of planning for parallelization is highlighted to ensure that bugs are detected early, reducing debugging time and effort. This is particularly crucial when working with large-scale systems involving thousands of processors.
:p Why is it important to plan for parallelization in the development process?
??x
Early detection of bugs through unit testing can save significant amounts of time and resources, especially during long runtime tasks. In a scenario where you are running a program across 1,000 processors, debugging at any later stage would be far more complex and time-consuming.
```bash
# Example command to run commit tests using Bash script
./TimeIt
```
x??

---

#### Unit Testing in Parallel Code Development
Background context: Unit testing is an essential part of the development process. It helps in identifying issues early, making it easier to resolve them. Test-driven development (TDD) involves creating tests before writing the actual code.
:p What is the significance of unit testing and TDD in parallel code development?
??x
Unit testing ensures that individual components of a program work correctly. TDD promotes a mindset where developers write tests first, ensuring that the final product meets all requirements. This approach helps in maintaining high-quality code by catching issues early.
```python
# Example of a simple unit test using Python's unittest module
import unittest

class TestMyFunction(unittest.TestCase):
    def test_my_function(self):
        self.assertEqual(my_function(), expected_output)

if __name__ == '__main__':
    unittest.main()
```
x??

---

#### Commit Tests in Development Workflow
Background context: Commit tests are critical for continuous integration and maintaining code quality. They run before a developer commits changes to the repository, ensuring that all committed code is functional.
:p What are commit tests and why are they important?
??x
Commit tests ensure that every change made to the codebase passes certain checks before it is merged into the main branch. This helps in preventing broken builds and ensures the stability of the codebase. Developers can run these tests easily from the command line or through a makefile.
```bash
# Example command to run commit tests using CMake and CTest
make commit_tests
```
x??

---

#### Continuous Integration Tests
Background context: Continuous integration (CI) tests are automatically triggered when changes are committed to the repository. These tests provide an additional layer of quality assurance, ensuring that all code is reliable.
:p How do continuous integration tests contribute to a project?
??x
Continuous integration tests run automatically and can be configured using various tools like Jenkins, Travis CI, GitLab CI, or CircleCI. They act as a safeguard against committing bad code and help in maintaining the overall quality of the software.
```bash
# Example command to run CI tests with Jenkins
jenv build-job my-project
```
x??

---

#### Regression Testing
Background context: Regression testing is performed after making changes to ensure that existing functionality continues to work as expected. These tests are typically run overnight and can be extensive, covering all aspects of the application.
:p What is regression testing and when is it used?
??x
Regression testing is crucial for ensuring that new code changes do not break existing features. It is often automated and runs nightly or periodically. Tools like cron jobs are commonly used to schedule these tests at specific times.
```bash
# Example cron job entry for running regression tests
0 2 * * * /path/to/regression-tests.sh >> /path/to/logfile.log 2>&1
```
x??

---

#### CMake and CTest Integration in Workflow
Background context: Using CMake and CTest to manage testing workflows can simplify the process of setting up and running different types of tests. This integration helps in maintaining a consistent development environment.
:p How does integrating CMake and CTest enhance the testing workflow?
??x
CMake is used for building and managing projects, while CTest provides robust test execution capabilities. Together, they enable developers to create detailed test plans and run various tests efficiently. For example, commit tests can be integrated into the build process to ensure that every code change passes these tests.
```cmake
# Example of adding a custom target in CMakeLists.txt for running tests
add_custom_target(run_tests COMMAND ctest)
```
x??

---

---
#### Uninitialized Memory
Background context: Uninitialized memory is a common issue where variables are accessed before being properly initialized. This can lead to unpredictable behavior because the variable holds indeterminate values left over from previous operations or hardware.

:p What happens if a variable is accessed without being initialized?
??x
When a variable is accessed without being initialized, it contains whatever value was previously stored in that memory location. This could be random data left over from previous computations, which can lead to bugs and unexpected behavior.
x??

---
#### Memory Overwrites
Background context: Memory overwriting occurs when a program writes data into a memory location not owned by the variable. Common examples include writing past the bounds of an array or string.

:p What is an example of memory overwrite?
??x
An example of memory overwrite is writing to an index that is beyond the allocated size of an array or string, leading to incorrect values being stored and possibly overwriting adjacent memory.
x??

---
#### Using Valgrind for Memory Checking
Background context: Valgrind is a powerful tool used to detect memory issues in programs. It operates at the machine code level, intercepting every instruction to check for various types of errors.

:p How do you run Valgrind on an MPI job?
??x
To run Valgrind on an MPI job, insert the `valgrind` command after `mpirun` and before your executable name. For example: 
```
mpirun -n 2 valgrind <./myapp>
```

x??

---
#### Memcheck Tool in Valgrind
Background context: The Memcheck tool is the default tool in the Valgrind suite, intercepting instructions to check for memory errors and generating diagnostics.

:p What does Memcheck do during execution?
??x
Memcheck intercepts every instruction during program execution and checks it for various types of memory errors. It generates diagnostics at the start, during, and end of the run, which can slow down the program by an order of magnitude.
x??

---
#### Example Code for Valgrind Memory Errors
Background context: The provided code snippet is intended to demonstrate how Valgrind identifies memory issues like invalid writes and uninitialized values.

:p What are some key parts of Valgrind's output in this example?
??x
Key parts of the Valgrind output include:
- `Invalid write of size 4`: Indicates an attempt to write to a location that does not exist.
- `Conditional jump or move depends on uninitialized value(s)`: Points out that a condition relies on data from an uninitialized variable.

Example code and output:
```c
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int ipos, ival;
    int *iarray = (int *) malloc(10*sizeof(int));
    if (argc == 2) ival = atoi(argv[1]);
    for (int i = 0; i<=10; i++) { iarray[i] = ipos; }
    for (int i = 0; i<=10; i++) {
        if (ival == iarray[i]) ipos = i;
    }
}
```
Run with `valgrind --leak-check=full ./test`:
```
==14324== Invalid write of size 4
==14324==    at 0x400590: main (test.c:7)
==14324== Conditional jump or move depends on uninitialized value(s)
```

x??

---

