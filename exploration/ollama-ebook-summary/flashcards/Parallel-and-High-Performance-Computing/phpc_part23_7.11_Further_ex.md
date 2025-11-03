# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 23)

**Starting Chapter:** 7.11 Further explorations

---

#### Task-Based Parallel Strategy Overview
Background context: The task-based parallel strategy allows dividing work into separate tasks that can be assigned to individual processes. This approach is more natural for many algorithms compared to traditional thread management techniques. OpenMP has supported this since version 3.0 and continues to improve in subsequent releases.

:p What is the task-based parallel strategy, and why is it useful?
??x
The task-based parallel strategy divides a problem into separate tasks that can be executed independently by different threads. This approach simplifies parallelization because tasks can be dynamically created and managed by the runtime system or programmer. It's particularly useful for algorithms where natural divisions exist (e.g., recursive algorithms, data processing pipelines).

This method reduces overhead associated with thread management and synchronization compared to thread-based approaches.

```c
// Example code snippet in C using OpenMP
#include <omp.h>

double PairwiseSumByTask(double* restrict var, long ncells) {
    double sum;
    #pragma omp parallel // Start of the parallel region
    {
        #pragma omp task
        {
            // Perform tasks here
        }
    }
}
```
x??

---

#### Recursive Data Splitting in Task-Based Summation
Background context: In the provided code, the algorithm recursively splits an array into smaller halves during a downward sweep. This process continues until each sub-array has a length of 1. During the upward sweep, pairs are summed.

:p How does the recursive data splitting work in the task-based summation?
??x
The recursive data splitting works by dividing the input array into two halves repeatedly until each segment contains only one element. In the upward phase, these single-element segments are paired and their values are summed together.

Here’s a detailed explanation with code:

```c
// Pseudocode for the recursive split and sum process
void pairwiseSumRecursive(double* arr, long start, long end) {
    if (end - start <= 1) { // Base case: single element or empty segment
        return;
    }

    long mid = (start + end) / 2; // Find midpoint

    // Spawn tasks for the left and right segments
    #pragma omp task shared(arr)
    pairwiseSumRecursive(arr, start, mid);

    #pragma omp task shared(arr)
    pairwiseSumRecursive(arr, mid, end);
}

int main() {
    double array[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    long size = sizeof(array) / sizeof(double);

    // Start the parallel region
    #pragma omp parallel
    {
        #pragma omp task
        pairwiseSumRecursive(array, 0, size);
    }
}
```

In this example, tasks are spawned in a parallel region to handle different segments of the array.

x??

---

#### Upward Sweep for Pairwise Summation
Background context: After the downward sweep where arrays are recursively split into halves, the upward sweep begins. This phase involves summing pairs of elements from the recursive splits, starting with single-element arrays and working back up to larger sums.

:p What happens during the upward sweep in pairwise summation?
??x
During the upward sweep, the algorithm combines the results from the downward split by pairing up the smallest segments (single elements) and summing their values. This process continues until all pairs are combined into a single total sum. Essentially, it reconstructs the original array's sums by combining smaller segment sums.

Here’s how this can be implemented:

```c
// Pseudocode for upward sweep
double PairwiseSumByTask(double* restrict var, long ncells) {
    double sum;
    #pragma omp parallel // Start of the parallel region
    {
        #pragma omp task
        {
            if (ncells <= 1) { // Base case: single element or empty segment
                return var[0];
            }

            long mid = ncells / 2; // Find midpoint

            // Recursive sum for left and right halves
            double leftSum = PairwiseSumByTask(var, mid);
            double rightSum = PairwiseSumByTask(&var[mid], ncells - mid);

            return leftSum + rightSum; // Sum of both halves
        }
    }
}

// Example usage in a main function
int main() {
    double array[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    long size = sizeof(array) / sizeof(double);

    // Start the parallel region
    #pragma omp parallel
    {
        #pragma omp task
        sum = PairwiseSumByTask(array, size);
    }

    // Collect or print results as needed
}
```

In this code, tasks are recursively spawned to handle each segment of the array. The upward phase combines these segments back into a single sum.

x??

---

---
#### Parallel Task Execution using OpenMP
Background context: The provided code snippet demonstrates a parallel algorithm that uses OpenMP tasks to compute the sum of an array in a divide-and-conquer manner. This approach is common in algorithms like merge sort, where the problem space is recursively divided into smaller sub-problems.

:p What does `#pragma omp masked` do in this context?
??x
In this context, `#pragma omp masked` is not used as shown; it typically allows tasks to be masked (i.e., conditionally executed) based on a specific predicate. However, the actual task execution here is handled by recursive function calls and OpenMP task directives.
??
---
#### Recursive Task Launching
Background context: The `PairwiseSumBySubtask` function recursively divides an array into smaller subarrays until each element can be processed individually (i.e., when the subarray size is 1). Then, it merges results from these subtasks using OpenMP tasks.

:p How does the `PairwiseSumBySubtask` function work?
??x
The function works by first checking if the subarray size (`nsize`) is 1. If so, it returns the single value directly. Otherwise, it recursively divides the array into two halves and launches two tasks to compute the sum of each half. After both tasks complete, their results are summed up.

```c
double PairwiseSumBySubtask(double* restrict var, long nstart, long nend) {
    long nsize = nend - nstart;
    if (nsize == 1){
        return(var[nstart]);
    }
    
    #pragma omp task shared(x) mergeable final(nsize > 10)
    x = PairwiseSumBySubtask(var, nstart, nstart + nmid);
    
    #pragma omp task shared(y) mergeable final(nsize > 10)
    y = PairwiseSumBySubtask(var, nend - nmid, nend);

    #pragma omp taskwait
    return(x+y);
}
```

The `#pragma omp task` directives create tasks to compute the sums of the left and right halves. The `mergeable final(nsize > 10)` clause ensures that tasks can be merged if they are small enough.
??
---
#### Task Synchronization with Taskwait
Background context: After launching multiple tasks, it's necessary to ensure all tasks have completed before proceeding. This is handled using the `#pragma omp taskwait` directive.

:p What does `#pragma omp taskwait` do in this code snippet?
??x
`#pragma omp taskwait` ensures that the current thread waits for all launched tasks to complete before continuing with the next line of code. This prevents race conditions and ensures correct computation of results.
??
---
#### Performance Tuning in Task-Based Algorithms
Background context: Achieving good performance with OpenMP task-based algorithms requires careful tuning, such as controlling the number of threads spawned and keeping task granularity reasonable.

:p What are some key considerations for achieving good performance with OpenMP task-based algorithms?
??x
Key considerations include:
- Limiting the number of threads to avoid excessive overhead.
- Ensuring tasks have a reasonable amount of work (granularity) to prevent too many small tasks being created.
- Using appropriate scheduling policies and directives like `mergeable final(nsize > 10)` to manage task dependencies and merging.

For example, setting a threshold in the `final` directive helps avoid creating too many small tasks that could degrade performance due to frequent context switching.
??
---

---
#### High-Level OpenMP Overview
High-level OpenMP refers to more sophisticated and efficient techniques of implementing OpenMP, designed for better performance on current and upcoming many-core architectures. These implementations are often developed by researchers and can significantly enhance application speed-up compared to basic OpenMP constructs.

:p What is high-level OpenMP?
??x
High-level OpenMP involves advanced techniques that improve the efficiency and effectiveness of parallel programming using OpenMP. These advancements help in achieving better performance on modern many-core architectures, such as Intel's Knights Landing processors. By refining how tasks are distributed among threads and managing synchronization more effectively, these implementations can lead to substantial improvements in application performance.

??x
---

---
#### Vector Add Example with High-Level OpenMP
The vector add example is a common task used to demonstrate basic OpenMP concepts. High-level OpenMP involves enhancing this implementation for better performance by optimizing loop structures, reducing synchronization overheads, and improving thread management.

:p Convert the vector add example in Listing 7.8 into a high-level OpenMP version.
??x
To convert the vector add example into a high-level OpenMP version, we need to focus on minimizing synchronization and maximizing parallel efficiency. Here’s an enhanced version:

```c
void vecAdd(double *a, double *b, double *c, int n) {
    #pragma omp parallel for reduction(+:c)
    for (int i = 0; i < n; ++i) {
        c[i] += a[i] + b[i];
    }
}
```

Explanation:
- The `reduction` clause is used to automatically handle the addition of values within each thread and accumulate the result in variable `c`.
- This reduces explicit synchronization, making the code more efficient.

??x
---

---
#### Maximum Value Calculation with OpenMP
Finding the maximum value in an array using OpenMP involves parallelizing a loop. By adding appropriate pragmas, you can enable parallel execution of the loop to improve performance.

:p Write a routine to get the maximum value in an array and add an OpenMP pragma for thread parallelism.
??x
Here is a routine that finds the maximum value in an array using OpenMP:

```c
double maxArray(double *arr, int n) {
    double max = arr[0];
    #pragma omp parallel shared(arr, n, max)
    {
        int idx;
        double localMax;

        #pragma omp single
        {
            // Ensure the initial value of `max` is assigned to all threads
            for (int i = 1; i < n; ++i) {
                if (arr[i] > max) max = arr[i];
            }
        }

        idx = omp_get_thread_num();
        localMax = arr[idx];

        #pragma omp barrier

        // Find the maximum of all threads' localMax values
        for (int i = 0; i < n / omp_get_num_threads(); ++i) {
            int tid = idx + i * omp_get_num_threads();
            if (tid >= n) break;
            if (arr[tid] > localMax) localMax = arr[tid];
        }

        // Reduce all threads' localMax to the global max
        #pragma omp single nowait
        {
            if (localMax > max) max = localMax;
        }
    }
    return max;
}
```

Explanation:
- The `#pragma omp parallel` directive creates a team of threads that will execute concurrently.
- The `#pragma omp single` and `#pragma omp barrier` ensure that all threads execute the critical section once before proceeding.
- `localMax` is used to store each thread's maximum value, which is later reduced to find the overall maximum.

??x
---

---
#### High-Level OpenMP Reduction Example
The reduction example involves using a high-level approach to perform a reduction operation in parallel. This typically involves minimizing synchronization and optimizing loop structures for better performance.

:p Write a high-level OpenMP version of the reduction from the previous exercise.
??x
Here is a high-level OpenMP version of the reduction routine:

```c
double maxArrayReduction(double *arr, int n) {
    double max = arr[0];
    #pragma omp parallel shared(arr, n)
    {
        // Each thread computes its local maximum
        int idx = omp_get_thread_num();
        double localMax = arr[idx];

        // Reduce all threads' localMax to the global max
        for (int i = 0; i < n / omp_get_num_threads(); ++i) {
            int tid = idx + i * omp_get_num_threads();
            if (tid >= n) break;
            if (arr[tid] > localMax) localMax = arr[tid];
        }

        // Use the nowait clause to minimize synchronization
        #pragma omp single nowait
        if (localMax > max) {
            max = localMax;
        }
    }
    return max;
}
```

Explanation:
- The `#pragma omp parallel` directive initializes a thread team.
- Each thread computes its own maximum value using `localMax`.
- The `#pragma omp single nowait` ensures minimal synchronization by reducing only when necessary.

??x
---

