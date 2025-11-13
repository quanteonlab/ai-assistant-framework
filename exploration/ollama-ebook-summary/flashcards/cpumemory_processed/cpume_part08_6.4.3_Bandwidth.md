# Flashcards: cpumemory_processed (Part 8)

**Starting Chapter:** 6.4.3 Bandwidth Considerations

---

#### Memory Bandwidth Considerations for Parallel Programs
Background context: When using many threads, even without cache contention, memory bandwidth can become a bottleneck. Each processor has a maximum bandwidth to memory shared by all cores and hyper-threads on that processor. This limitation can affect performance, especially in scenarios with large working sets.
:p What is the primary concern regarding memory bandwidth when running multiple threads?
??x
The primary concern is that the available memory bandwidth may become a limiting factor for parallel programs, even if there is no cache contention between threads.
x??

---
#### Bus Speed and Processor Core Count
Background context: Increasing the Front Side Bus (FSB) speed can help improve memory bandwidth. As more cores are added to processors, FSB speeds often increase as well. However, this alone might not be sufficient for programs that use large working sets and are sufficiently optimized.
:p How does increasing the FSB speed affect a processor's performance?
??x
Increasing the FSB speed can enhance memory bandwidth, allowing better performance in scenarios where there is high demand for data access from main memory. This improvement is particularly beneficial when dealing with large working sets that cannot be fully cached.
x??

---
#### Detecting Bus Contention on Core 2 Processors
Background context: Core 2 processors provide specific events to measure FSB contention, such as the NUS_BNR_DRV event which counts cycles a core has to wait because the bus is not ready. These tools can help identify when bus utilization is high.
:p How does the NUS_BNR_DRV event work in detecting FSB contention?
??x
The NUS_BNR_DRV event on Core 2 processors counts the number of cycles a core must wait due to an unready bus, indicating that the bus is heavily utilized. This can help identify cases where memory access operations take longer than usual.
x??

---
#### Improving Memory Bandwidth Utilization
Background context: To improve memory bandwidth utilization, several strategies can be employed, including upgrading hardware or optimizing program code and thread placement. The scheduler in the kernel typically assigns threads based on its own policy but may not fully understand the specific workload demands.
:p What are some strategies to address limited memory bandwidth?
??x
Strategies include buying faster computers with higher FSB speeds and faster RAM modules, possibly even local memory. Additionally, optimizing the program code to minimize cache misses and repositioning threads more effectively on available cores can help utilize memory bandwidth better.
x??

---
#### Scheduler Behavior in Memory Bandwidth Management
Background context: By default, the kernel scheduler assigns threads based on its own policies but may not be aware of specific workload demands. Cache miss information can provide some insight but is often insufficient for making optimal thread placements.
:p How does the kernel scheduler handle thread placement by default?
??x
The kernel scheduler typically assigns threads based on its own policies and tries to avoid moving threads from one core to another when possible, even though it may not fully understand the specific workload demands. Cache miss information can provide some insight but is often insufficient for making optimal thread placements.
x??

---

#### Memory Bus Usage Inefficiency
Background context: When two threads on different cores access the same data set, it can lead to inefficiencies. Each core might read the same data from memory separately, causing higher memory bus usage and decreased performance.

:p What is a situation that can cause big memory bus usage?
??x
A situation where two threads are scheduled on different processors (or cores in different cache domains) and they use the same data set.
x??

---

#### Efficient Scheduling
Background context: Proper scheduling of threads to cores with shared data sets can reduce memory bus usage. By placing threads that share data on the same core, the data can be read from memory only once.

:p How does efficient scheduling affect memory bus usage?
??x
Efficient scheduling reduces memory bus usage by ensuring that threads accessing the same data set are placed on the same cores, thereby reducing redundant reads from memory.
x??

---

#### Thread Affinity
Background context: Thread affinity allows a programmer to specify which core(s) a thread can run on. This is useful in optimizing performance but may cause idle cores if too many threads are assigned exclusively to a few cores.

:p What is thread affinity?
??x
Thread affinity is the ability to assign a thread to one or more specific cores, ensuring that the scheduler runs the thread only on those cores.
x??

---

#### Scheduling Interface: `sched_setaffinity`
Background context: The kernel does not have insight into data use by threads, so programmers must ensure efficient scheduling. The `sched_setaffinity` interface allows setting the core(s) a thread can run on.

:p How is thread affinity set using C code?
??x
Thread affinity is set using the `sched_setaffinity` function in C. This function requires specifying the process ID, size of the CPU set, and the bitmask for the cores.
```c
#include <sched.h>
#define _GNU_SOURCE

int sched_setaffinity(pid_t pid, size_t size, const cpu_set_t *cpuset);
```
The `pid` parameter specifies which process’s affinity should be changed. The caller must have appropriate privileges to change the affinity.

x??

---

#### Scheduling Interface: `sched_getaffinity`
Background context: Similar to setting thread affinity, the `sched_getaffinity` interface retrieves the core(s) a thread is currently assigned to.

:p How is current thread affinity retrieved using C code?
??x
Current thread affinity can be retrieved using the `sched_getaffinity` function in C. This function requires specifying the process ID, size of the CPU set, and a buffer for the bitmask.
```c
#include <sched.h>
#define _GNU_SOURCE

int sched_getaffinity(pid_t pid, size_t size, cpu_set_t *cpuset);
```
The `pid` parameter specifies which process’s affinity should be queried. The function fills in the bitmask with the scheduling information of the selected thread.

x??

---

#### CPU Set Operations
Background context: The `cpu_set_t` type and associated macros are used to manipulate core sets, allowing precise control over thread placement.

:p How do you initialize a `cpu_set_t` object?
??x
A `cpu_set_t` object is initialized using the `CPU_ZERO` macro. This clears all bits in the set, effectively setting it to an empty state.
```c
#include <sched.h>

// Initialize cpu_set_t object
CPU_ZERO(&cpuset);
```
This operation must be performed before setting or clearing specific cores.

x??

---

#### CPU Set Operations (continued)
Background context: Once initialized, individual cores can be added or removed from the set using `CPU_SET` and `CPU_CLR`.

:p How do you add a core to a `cpu_set_t` object?
??x
To add a core to a `cpu_set_t` object, use the `CPU_SET` macro. This sets the bit for a specific core in the bitmask.
```c
#include <sched.h>

// Add core 2 (assuming CPU numbering starts at 0)
CPU_SET(2, &cpuset);
```
x??

---

#### CPU Set Operations (continued)
Background context: To remove a core from a `cpu_set_t` object, use the `CPU_CLR` macro.

:p How do you remove a core from a `cpu_set_t` object?
??x
To remove a core from a `cpu_set_t` object, use the `CPU_CLR` macro. This clears the bit for a specific core in the bitmask.
```c
#include <sched.h>

// Remove core 3 (assuming CPU numbering starts at 0)
CPU_CLR(3, &cpuset);
```
x??

---

#### CPU Set Operations (continued)
Background context: To check if a specific core is included in the set, use the `CPU_ISSET` macro.

:p How do you check if a core is part of a `cpu_set_t` object?
??x
To check if a specific core is part of a `cpu_set_t` object, use the `CPU_ISSET` macro. This returns non-zero if the bit for the specified core is set.
```c
#include <sched.h>

// Check if core 1 (assuming CPU numbering starts at 0) is in the set
if(CPU_ISSET(1, &cpuset)) {
    // Core 1 is part of the set
}
```
x??

---

#### CPU Set Operations (continued)
Background context: To count the number of cores selected in a `cpu_set_t` object, use the `CPU_COUNT` function.

:p How do you count the number of cores selected in a `cpu_set_t` object?
??x
To count the number of cores selected in a `cpu_set_t` object, use the `CPU_COUNT` macro. This returns the number of bits set to 1 in the bitmask.
```c
#include <sched.h>

// Count the number of selected cores
int count = CPU_COUNT(&cpuset);
```
x??

---

---
#### CPU Set Handling Macros
This section explains how to handle dynamic CPU sets using macros provided by the GNU C Library. These macros allow for flexible and dynamically sized CPU set management, which is crucial for programs that need to adapt to different system configurations.

:p What are the macros used for handling dynamically sized CPU sets?
??x
The macros include `CPU_ALLOC_SIZE`, `CPU_ALLOC`, and `CPU_FREE`. The first macro determines the size of a `cpu_set_t` structure needed for a given number of CPUs, while the second allocates memory for such a structure. Finally, the third frees the allocated memory.

Code example:
```c
#define _GNU_SOURCE
#include <sched.h>

size_t requiredSize = CPU_ALLOC_SIZE(count);
void *cpuset = CPU_ALLOC(requiredSize);

// Use cpuset...

CPU_FREE(cpuset);
```
x??
---

#### Logical Operations on CPU Sets
This section describes macros that perform logical operations (AND, OR, XOR) on `cpu_set_t` structures. These operations are useful for managing and manipulating sets of CPUs.

:p What is the purpose of the logical operation macros defined in this section?
??x
The purpose of these macros is to provide a way to manipulate CPU set objects using standard logical operators such as AND, OR, and XOR. These operations can be used to combine or compare different sets of CPUs.

Code example:
```c
#define _GNU_SOURCE
#include <sched.h>

cpu_set_t destset;
CPU_AND_S(setsize, destset, cpuset1, cpuset2);
```
x??
---

#### sched_getcpu Interface
This section introduces the `sched_getcpu` function, which returns the index of the CPU on which a process is currently running. This can be useful for identifying where a process or thread is executing.

:p What does the `sched_getcpu` interface return?
??x
The `sched_getcpu` interface returns the index of the CPU on which the calling process is currently running. However, due to the nature of scheduling, this value may not always be 100% accurate as the thread might have been moved between the time the result was returned and when it returns to user level.

Code example:
```c
#include <sched.h>
int cpuIndex = sched_getcpu();
```
x??
---

#### sched_getaffinity Interface
This section discusses the `sched_getaffinity` function, which retrieves the set of CPUs a process or thread is allowed to run on. This information can be useful for determining the affinity mask and ensuring that threads are restricted to certain CPU sets.

:p What does the `sched_getaffinity` interface return?
??x
The `sched_getaffinity` interface returns a `cpu_set_t` structure containing the set of CPUs on which the process or thread is allowed to run. This information can be useful for managing and controlling the execution environment of threads.

Code example:
```c
#include <sched.h>
cpu_set_t cpuset;
int rc = sched_getaffinity(pid, sizeof(cpu_set_t), &cpuset);
```
x??
---

#### Linux CPU Hot-Plugging and Thread Affinity
Background context: Linux supports CPU hot-plugging, allowing CPUs to be added or removed from a system while it is running. This capability also affects how threads are assigned to cores through CPU affinity settings. In multi-threaded programs, individual threads do not have POSIX-defined process IDs, thus traditional functions for setting and getting process-level CPU affinity cannot be used.

Relevant interfaces introduced in the text allow for setting and getting thread-specific CPU affinity:
- `pthread_setaffinity_np`: Sets the CPU affinity of a given thread.
- `pthread_getaffinity_np`: Gets the current CPU affinity settings of a given thread.
- `pthread_attr_setaffinity_np`: Sets the CPU affinity attribute at thread creation time.
- `pthread_attr_getaffinity_np`: Gets the CPU affinity attribute for a thread.

These functions take a thread handle and a `cpu_set_t` structure to specify the allowed CPUs. 

:p What are the main differences between traditional process-level CPU affinity functions (PID-based) and thread-level CPU affinity functions?
??x
The primary difference lies in the fact that POSIX does not define a process ID for individual threads, whereas processes do have unique PIDs. Therefore, functions like `sched_setaffinity` operate on process IDs, while `pthread_setaffinity_np` works with thread handles. Additionally, setting CPU affinity at the thread level can be advantageous as it influences scheduling decisions earlier in the thread's lifecycle.

:p How does one set the CPU affinity for a thread using `pthread_setaffinity_np`?
??x
To set the CPU affinity of a thread, you would use the `pthread_setaffinity_np` function. You need to provide the thread identifier (`th`), the size of the affinity mask, and a pointer to the `cpu_set_t` structure that defines the allowed CPUs.

```c
#include <pthread.h>
#include <sched.h>

// Example usage:
int result = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), &cpuset);

if (result == 0) {
    printf("Affinity set successfully\n");
} else {
    perror("Failed to set affinity");
}
```

x??

#### NUMA Programming Overview
Background context: NUMA (Non-Uniform Memory Access) introduces different costs when accessing different parts of the address space. Unlike uniform memory access, where all pages are created equal, in NUMA, the cost of accessing a page can vary based on which node it is located.

:p What is NUMA and how does it differ from uniform memory access?
??x
NUMA (Non-Uniform Memory Access) is a system design that allows for different costs when accessing different parts of the address space. In contrast to uniform memory access, where all pages are treated equally, NUMA systems have varying costs depending on which node the data resides in.

:p How does NUMA affect cache and memory optimization strategies?
??x
NUMA affects cache and memory optimization by introducing differing costs for accessing different parts of the address space. This means that optimizing for cache sharing to enhance bandwidth becomes more complex because the cost of accessing a specific page can vary based on its location. Programmers need to consider not just memory locality but also the physical placement of data across nodes.

:p What is an example scenario where NUMA might be beneficial?
??x
An example scenario where NUMA might be beneficial is when two threads work on separate data sets and are scheduled on different cores, reducing cache contention and improving overall performance. By placing these threads on non-sharing cores, the system can minimize the number of page faults and improve memory locality.

:x??

#### Thread Affinity in Multi-Threaded Programs
Background context: In multi-threaded programs, individual threads do not have a process ID as defined by POSIX, making it challenging to apply traditional CPU affinity functions. The `pthread_setaffinity_np` function is introduced to set the affinity of a thread directly.

:p How does the `pthread_setaffinity_np` function differ from its process-level counterpart?
??x
The `pthread_setaffinity_np` function differs from the process-level counterpart (like `sched_setaffinity`) in that it operates on thread handles rather than process IDs. This allows for more granular control over CPU affinity settings, as each thread can have its own specific CPU constraints.

:p How does setting affinity early with `pthread_attr_setaffinity_np` impact scheduling?
??x
Setting the affinity of a thread at creation time using `pthread_attr_setaffinity_np` impacts scheduling by allowing threads to be scheduled from the start on specific sets of CPUs. This can be advantageous, especially for optimizing cache sharing and memory page locality.

:x??

#### NUMA vs. Uniform Memory Access
Background context: In a NUMA system, the cost of accessing different parts of the address space varies based on the node where data is located. Unlike uniform memory access (UMA), where all pages are treated equally, NUMA introduces differing costs for memory access.

:p What is the primary difference between NUMA and UMA in terms of memory access?
??x
The primary difference between NUMA and UMA lies in how memory access costs vary based on the node. In a NUMA system, accessing data from a different node incurs higher costs compared to local nodes. Conversely, in UMA systems, all memory is treated uniformly with no significant variation in access costs.

:p How can programmers optimize cache sharing in a NUMA environment?
??x
In a NUMA environment, optimizing cache sharing involves setting the affinity of threads so that they do not share the same core or cache level. This reduces contention and improves overall performance by minimizing page faults and maximizing memory locality.

:x??

#### Cache Hierarchy and Thread Affinity
Background context: As the number of cores per processor increases, managing caches becomes more complex. Threads on cores sharing a cache can collaborate faster than those not sharing a cache.

:p How does increasing core density affect cache management?
??x
Increasing core density in processors introduces hierarchical cache structures. Placing threads on cores that share higher-level caches (like L2 or L3) is crucial for optimizing performance, as it reduces cache contention and improves data locality.

:p What is the significance of NUMA support libraries in multi-core scheduling?
??x
NUMA support libraries provide tools to manage thread affinity across multiple cores, ensuring optimal placement on shared cache levels. These libraries help programmers determine affinity masks without hardcoding system details or diving into low-level filesystems, making it easier to write efficient and scalable applications.

:x??

---
#### Memory Policy Concept
Memory policy definitions allow processes to control where their memory is allocated on a NUMA system. The Linux kernel supports several policies that can be applied at different levels: task, VMA (Virtual Memory Area), and default system-wide.

:p What are the types of memory policies supported by the Linux kernel?
??x
The Linux kernel supports four main memory policies:
- MPOL_BIND: Allocates memory only from a given set of nodes. If this is not possible, allocation fails.
- MPOL_PREFERRED: Prefers allocating memory from the specified nodes; if it cannot be allocated locally, other nodes are considered.
- MPOL_INTERLEAVE: Allocates memory equally from the specified nodes using an offset or counter mechanism.
- MPOL_DEFAULT: Uses the default policy for a region. If no specific policy is set, this defaults to allocating local memory.

This hierarchy allows flexibility in managing memory allocation based on the needs of different processes and regions within them.
x??

---
#### Memory Policy Hierarchy
Memory policies form a hierarchical structure where each level can influence memory allocation decisions:
1. VMA (Virtual Memory Area) Policy: Specific to a particular address space region.
2. Task Policy: Applies to all allocations for threads in the same task.
3. System Default Policy: Used when no specific policy is set, defaulting to local node allocation.

:p How does the memory policy hierarchy work?
??x
The memory policy hierarchy works as follows:
- If an address is covered by a VMA policy, that policy is used.
- If there's no VMA policy for a specific address, the task policy is applied.
- If neither a VMA nor task policy is present, the system default policy is used.

The system default policy allocates memory locally to the thread requesting it. Each process typically does not provide explicit policies; instead, they inherit from their parent processes or follow the default behavior of allocating memory on the local node.
x??

---
#### set_mempolicy Function
The `set_mempolicy` function is used to set the task policy for a specific thread in Linux. This allows controlling how memory allocations are handled by specifying nodes and allocation modes.

:p How does one use the `set_mempolicy` function?
??x
To use the `set_mempolicy` function, you need to include `<numaif.h>` and call it with appropriate parameters:

```c
#include <numaif.h>

long set_mempolicy(int mode, unsigned long *nodemask, unsigned long maxnode);
```

- `mode`: Must be one of the MPOL_* constants (e.g., MPOL_BIND).
- `nodemask`: Specifies which memory nodes should be used for future allocations. This is a bitmask where each bit represents a node.
- `maxnode`: The number of bits in the nodemask.

Example usage:
```c
#include <numaif.h>
#include <stdio.h>

int main() {
    unsigned long mask = 1 << 0 | 1 << 2; // Nodes 0 and 2 are preferred
    int result = set_mempolicy(MPOL_PREFERRED, &mask, 3);
    if (result == -1) {
        perror("set_mempolicy");
        return 1;
    }
    printf("Memory policy set to preferred nodes: %lx\n", mask);
    return 0;
}
```

This example sets the memory policy for a thread such that it prefers allocating memory from nodes 0 and 2. If `mode` is `MPOL_DEFAULT`, no specific nodes need to be provided.
x??

---
#### Migration Policies
The `mbind` system call allows more granular control over memory binding by specifying which pages should be bound to specific nodes.

:p How does the `mbind` function work?
??x
The `mbind` function is used for finer-grained control of memory binding:

```c
#include <numaif.h>

int mbind(unsigned long start, size_t len, int mode, unsigned long *nodemask, unsigned long maxnode, unsigned long flags);
```

- `start`: The starting address of the memory region.
- `len`: The length of the memory region.
- `mode`: The binding mode (e.g., MPOL_BIND).
- `nodemask`: A bitmask specifying nodes to bind pages to.
- `maxnode`: The number of bits in the nodemask.

Example usage:
```c
#include <numaif.h>
#include <stdio.h>

int main() {
    unsigned long mask = 1 << 0 | 1 << 2; // Nodes 0 and 2 are preferred
    int result = mbind(0, 4096, MPOL_BIND, &mask, 3, 0);
    if (result == -1) {
        perror("mbind");
        return 1;
    }
    printf("Pages bound to nodes: %lx\n", mask);
    return 0;
}
```

This example binds a memory region starting at address `0` with size `4096` bytes to nodes 0 and 2.
x??

---
#### move_pages Function
The `move_pages` function is used for migrating pages between different nodes.

:p How does the `move_pages` function work?
??x
The `move_pages` function migrates selected memory pages from one set of nodes to another:

```c
#include <numaif.h>

int move_pages(pid_t pid, unsigned long start, size_t len, int flags, struct move_pages_args *args);
```

- `pid`: The process ID whose pages are being migrated.
- `start`: The starting address in the process's virtual memory.
- `len`: The length of the memory region to be considered for migration.
- `flags`: Flags controlling the operation (e.g., `MPOL_MF_MOVE`, `MPOL_MF_MUST_MIGRATE`).
- `args`: An argument structure specifying details about the migration.

Example usage:
```c
#include <numaif.h>
#include <stdio.h>

int main() {
    struct move_pages_args args = {0};
    int result = move_pages(1234, 0x7fffffffe000, 4096, MPOL_MF_MOVE | MPOL_MF_MUST_MIGRATE, &args);
    if (result == -1) {
        perror("move_pages");
        return 1;
    }
    printf("Pages moved successfully\n");
    return 0;
}
```

This example migrates pages starting at virtual address `0x7fffffffe000` with a length of `4096` bytes from process ID `1234`.
x??

---

