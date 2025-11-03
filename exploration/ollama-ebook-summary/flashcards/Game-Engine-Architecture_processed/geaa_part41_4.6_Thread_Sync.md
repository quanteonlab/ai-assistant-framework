# Flashcards: Game-Engine-Architecture_processed (Part 41)

**Starting Chapter:** 4.6 Thread Synchronization Primitives

---

#### Mutexes Overview
Mutexes are a type of thread synchronization primitive that allows critical operations to be made atomic. A mutex can exist in one of two states: unlocked or locked.

:p What is a mutex and what are its primary functions?
??x
A mutex is an operating system object designed to provide mutual exclusion for critical sections of code. It ensures that only one thread can execute a specific section of code at any given time, making those operations atomic relative to other threads. Mutexes achieve this by allowing only the holding thread to unlock them.

The primary functions of a mutex include:
- **create() or init()**: Creates the mutex.
- **destroy()**: Destroys the mutex.
- **lock() or acquire()**: Locks the mutex on behalf of the calling thread, putting it to sleep if the lock is currently held by another thread.
- **try_lock() or try_acquire()**: Attempts to lock the mutex without blocking; returns immediately if the lock cannot be acquired.
- **unlock() or release()**: Releases the lock on the mutex.

```cpp
// Example C++ code for using a mutex
#include <iostream>
#include <mutex>

std::mutex m;

void criticalSection() {
    m.lock();
    // Critical section of code that must be executed atomically
    std::cout << "Critical section is executing.\n";
    m.unlock();
}
```
x??

---

#### Mutex States and Operations
A mutex can be in one of two states: unlocked or locked. The most important property of a mutex is its mutual exclusion behavior, ensuring only one thread can hold the lock at any given time.

:p What are the states of a mutex and what does it guarantee?
??x
A mutex has two states:
- **Unlocked**: No thread holds the lock.
- **Locked (or acquired)**: One thread currently holds the lock.

The primary guarantee of a mutex is mutual exclusion, meaning that only one thread can hold the lock on the mutex at any given time. This ensures that critical sections of code are executed atomically with respect to other threads.

Example usage in C++:
```cpp
// Example C++ code demonstrating mutex states and operations
#include <iostream>
#include <mutex>

std::mutex m;

void func() {
    if (m.try_lock()) {  // Try to acquire the lock without blocking
        std::cout << "Lock acquired.\n";
        m.unlock();  // Release the lock after use
    } else {
        std::cout << "Failed to acquire lock.\n";
    }
}
```
x??

---

#### Mutex Lock and Unlock Functions
Mutexes provide functions like `lock()` or `acquire()`, which block until the mutex is available, and `unlock()` or `release()`, which release the mutex.

:p What are the locking and unlocking functions in a mutex?
??x
The locking and unlocking functions in a mutex include:
- **lock() or acquire()**: Blocks the calling thread until it can successfully obtain the lock. If another thread holds the lock, this function will put the thread to sleep.
- **try_lock() or try_acquire()**: Attempts to lock the mutex without blocking. It returns immediately if the lock cannot be acquired.
- **unlock() or release()**: Releases the lock on the mutex.

Example usage in C++:
```cpp
// Example C++ code demonstrating lock and unlock functions
#include <iostream>
#include <mutex>

std::mutex m;

void threadFunction() {
    while (true) {
        if (m.try_lock()) {  // Try to acquire the lock without blocking
            std::cout << "Lock acquired.\n";
            // Critical section of code
            m.unlock();  // Release the lock after use
        } else {
            std::cout << "Failed to acquire lock.\n";
        }
    }
}
```
x??

---

#### Mutex in Different States
When a mutex is locked by a thread, it enters a non-signaled state. When the thread releases the lock, the mutex becomes signaled. If other threads are waiting for the mutex, signaling it will wake one of these waiting threads.

:p What happens when a mutex is locked and released by a thread?
??x
When a mutex is locked by a thread, it enters a non-signaled state:
- Only the holding thread can lock the mutex.
- Other threads attempting to acquire the lock will block until the mutex becomes available.

When the holding thread releases the lock, the mutex enters a signaled state:
- The kernel selects one of the waiting threads and wakes it up to proceed with its execution.
- If multiple threads are waiting, typically only one is awakened, but this can vary depending on the operating system's implementation.

Example usage in C++:
```cpp
// Example C++ code demonstrating mutex states and signaling
#include <iostream>
#include <mutex>

std::mutex m;

void waitThread() {
    while (true) {
        if (m.try_lock()) {  // Try to acquire the lock without blocking
            std::cout << "Lock acquired.\n";
            m.unlock();  // Release the lock after use
        } else {
            std::cout << "Failed to acquire lock, waiting...\n";
        }
    }
}
```
x??

---

#### Mutex and Context Switches
Using mutexes involves a kernel call which can lead to context switches. These context switches are expensive and can cost upwards of 1000 clock cycles.

:p What is the overhead associated with using mutexes?
??x
The use of mutexes often involves a significant overhead due to their interaction with the operating system's kernel:
- Mutex operations require a kernel call, which necessitates a context switch into protected mode.
- Context switches are expensive and can cost upwards of 1000 clock cycles.

This high cost is one of the reasons why some concurrent programmers prefer alternative methods like implementing their own atomicity or using lock-free programming techniques to improve efficiency.

Example context for understanding overhead:
```cpp
// Example C++ code showing the potential overhead in a loop
#include <iostream>
#include <mutex>

std::mutex m;

void expensiveFunction() {
    while (true) {
        if (!m.try_lock()) {  // Non-blocking lock attempt
            std::cout << "Busy waiting...\n";
        } else {
            // Critical section
            m.unlock();
        }
    }
}
```
x??

---

---
#### POSIX Mutexes
POSIX threads provide a way to manage shared resources using mutexes, which are used for synchronization between threads. Mutexes ensure that only one thread can access a shared resource at any given time.

In our example, we use a counter `g_count` and protect it with a mutex `g_mutex`.

:p How do you implement a basic increment operation on a shared counter using POSIX mutexes?
??x
To increment the counter atomically, you first lock the mutex to ensure exclusive access. After modifying the counter, unlock the mutex to release it for other threads.

```c
#include <pthread.h>
int g_count = 0;
pthread_mutex_t g_mutex;

inline void IncrementCount() {
    pthread_mutex_lock(&g_mutex); // Lock the mutex to gain exclusive access.
    ++g_count; // Increment the shared counter.
    pthread_mutex_unlock(&g_mutex); // Unlock the mutex, allowing other threads to proceed.
}
```
x??

---
#### C++11 Mutexes
C++11 introduced `std::mutex` as part of the standard library. This provides a similar functionality to POSIX mutexes but with a more streamlined API.

In our example, we use a shared counter `g_count` and protect it with a `std::mutex`.

:p How do you implement an atomic increment operation on a shared counter using C++11 mutexes?
??x
To atomically increment the counter, you lock the mutex to ensure exclusive access. After modifying the counter, unlock the mutex.

```cpp
#include <mutex>
int g_count = 0;
std::mutex g_mutex;

inline void IncrementCount() {
    g_mutex.lock(); // Lock the mutex to gain exclusive access.
    ++g_count; // Increment the shared counter.
    g_mutex.unlock(); // Unlock the mutex, allowing other threads to proceed.
}
```
x??

---
#### Windows Mutexes
In Windows, a mutex is an opaque kernel object that can be used for synchronization. A mutex must be initialized and destroyed properly.

We use a handle `g_hMutex` to manage a shared counter `g_count`.

:p How do you implement an atomic increment operation on a shared counter using Windows mutexes?
??x
To atomically increment the counter, you first wait for the mutex to become signaled (locked) and then increment the counter. After modifying the counter, release the mutex.

```cpp
#include <windows.h>
int g_count = 0;
HANDLE g_hMutex;

inline void IncrementCount() {
    if (WaitForSingleObject(g_hMutex, INFINITE) == WAIT_OBJECT_0) { // Wait for the mutex to become available.
        ++g_count; // Increment the shared counter.
        ReleaseMutex(g_hMutex); // Unlock the mutex after use.
    } else {
        // Handle failure case where the mutex is not available.
    }
}
```
x??

---
#### Critical Sections
Critical sections are a synchronization mechanism provided by Windows that offer a lower-cost alternative to traditional mutexes. They allow for finer-grained locking and can be shared between threads.

We initialize, use, and destroy a critical section in our example.

:p How do you implement an atomic increment operation on a shared counter using Windows critical sections?
??x
To atomically increment the counter, you first enter the critical section to gain exclusive access. After modifying the counter, leave the critical section.

```cpp
#include <windows.h>
int g_count = 0;
InitializeCriticalSection(&g_hMutex); // Initialize the critical section.
DeleteCriticalSection(&g_hMutex); // Clean up when done using it.

inline void IncrementCount() {
    EnterCriticalSection(&g_hMutex); // Enter the critical section to gain exclusive access.
    ++g_count; // Increment the shared counter.
    LeaveCriticalSection(&g_hMutex); // Leave the critical section after use.
}
```
x??

---

#### Critical Section Implementation
Background context explaining how critical sections are used for thread synchronization. Mention that `EnterCriticalSection` and `LeaveCriticalSection` functions are non-blocking but provide exclusive access to a shared resource by blocking other threads.

:p How is an atomic increment implemented using Windows critical sections?
??x
An atomic increment can be implemented using the following code:
```cpp
#include <windows.h>

int g_count = 0;
CRITICAL_SECTION g_critsec;

inline void IncrementCount() {
    EnterCriticalSection(&g_critsec); // Acquire the critical section
    ++g_count;                       // Atomically increment the count
    LeaveCriticalSection(&g_critsec); // Release the critical section
}
```
The `EnterCriticalSection` function attempts to lock a critical section, and if it cannot be acquired, it will spin in a loop waiting for the critical section to become available. Once the lock is obtained, the atomic increment operation can proceed without interference from other threads.

x??

---
#### Spin Lock Mechanism
Background context explaining how Windows implements non-blocking locking mechanisms using spin locks. Mention that when a thread attempts to enter a locked critical section, it first tries to acquire the lock by spinning (busy-waiting) before resorting to yielding execution if necessary.

:p What mechanism does Windows use for acquiring a critical section atomically?
??x
Windows uses a spin lock mechanism to achieve non-blocking locking. When a thread attempts to enter a locked critical section, it first tries to acquire the lock in an inexpensive manner by spinning (busy-waiting) until the critical section becomes available.

If the thread cannot acquire the lock after a certain period of time or if other conditions are met, it will yield execution to allow other threads to run. This approach is cheaper than using a regular mutex because it avoids context switching into the kernel.

```cpp
// Pseudocode for attempting to enter a critical section with spin lock
void TryEnterCriticalSection(CRITICAL_SECTION *critsec) {
    while (true) { // Spin until the critical section becomes available
        if (TryLock(critsec)) {
            break; // Successfully acquired the lock, exit the loop
        }
        YieldProcessor(); // If unable to acquire, yield execution
    }
}
```
x??

---
#### Producer-Consumer Problem with Condition Variables
Background context explaining the producer-consumer problem and how it can be solved using condition variables. Highlight that a common approach is to use a global Boolean variable as a signaling mechanism between threads.

:p How does a producer thread signal readiness of data in the producer-consumer problem?
??x
In the producer-consumer problem, the producer thread signals readiness by setting a global Boolean variable `g_ready` to `true`. This informs the consumer that new data is available for consumption. Here's an example using POSIX threads:

```cpp
Queue g_queue;
pthread_mutex_t g_mutex;
bool g_ready = false;

void* ProducerThread(void*) {
    while (true) {
        pthread_mutex_lock(&g_mutex); // Lock to ensure mutual exclusion
        ProduceDataInto(&g_queue);
        g_ready = true;               // Signal that data is ready
        pthread_mutex_unlock(&g_mutex); // Unlock the mutex
        pthread_yield();              // Yield CPU time to allow consumer to run
    }
    return nullptr;
}
```
x??

---
#### Consumer Thread Behavior in Producer-Consumer Problem
Background context explaining how a consumer thread waits for data and consumes it. Emphasize that using polling can lead to unnecessary busy-waiting, which is inefficient.

:p How does the consumer thread handle waiting for data from the producer?
??x
The consumer thread uses a `while` loop to continuously check if the `g_ready` variable indicates that new data is available. If not, it will keep spinning (busy-waiting) until the producer signals readiness:

```cpp
void* ConsumerThread(void*) {
    while (true) {
        pthread_mutex_lock(&g_mutex); // Lock to ensure mutual exclusion
        const bool ready = g_ready;   // Check if data is ready
        pthread_mutex_unlock(&g_mutex);
        
        if (ready) break;             // If ready, exit the loop
        
        // Polling: keep spinning until ready
    }
    
    // Consume the data once it's ready
    pthread_mutex_lock(&g_mutex);
    ConsumeDataFrom(&g_queue);
    g_ready = false;
    pthread_mutex_unlock(&g_mutex);
}
```
x??

---
#### Alternatives to Mutexes and Critical Sections
Background context explaining that some operating systems provide alternative synchronization primitives such as futexes. Mention Linux's `futex` implementation, which acts somewhat like a critical section.

:p What is a futex and how does it differ from Windows' critical sections?
??x
A futex (short for "fast userspace mutex") is an alternative to traditional kernel-level mutexes provided by some operating systems, such as Linux. Unlike standard mutexes that involve a context switch into the kernel, futexes allow threads to wait in user space without involving the kernel.

The main difference between futexes and Windows' critical sections is their scope and implementation:
- **Critical Sections**: Are specific to Windows and are used for synchronization within a process.
- **Futexes**: Can be shared across processes, making them more flexible but also more complex to implement compared to Windows' critical sections.

Here's an example of how futexes might be used in Linux:

```c
#include <linux/futex.h>

int g_count = 0;

void IncrementCount() {
    struct futex_waiter waiter;
    
    // Wait for the lock
    while (futex_wait(&g_count, 0) != EWOULDBLOCK) {
        // Busy-wait if the count is not ready
    }
    
    // Critical section code here
    
    // Wake up any waiting threads
    futex_wake(&g_count, 1);
}
```
x??

---

#### Condition Variable (CV) Overview
Condition variables are kernel objects that help manage thread synchronization by queuing sleeping threads and providing a mechanism to wake them up. They combine a wait queue with a signaling mechanism, which is useful for scenarios where one thread produces data and another consumes it.

:p What is the primary purpose of a condition variable in concurrent programming?
??x
Condition variables are used to synchronize producer-consumer threads or any other scenario where threads need to wait until certain conditions are met before proceeding. They allow threads to be notified when specific events occur, such as new data being available for consumption.
x??

---
#### Creating and Initializing Condition Variables
To use condition variables effectively, they must first be created and initialized properly.

:p How do you create and initialize a condition variable in C?
??x
In C, a condition variable can be created using `pthread_cond_t`. You typically declare it as a global variable or pass it as an argument to your functions. Here's how you would define and initialize one:

```c
#include <pthread.h>

// Declare the condition variable globally
pthread_cond_t g_cv;

// In your initialization function, create and initialize the condition variable.
int initConditionVariable() {
    pthread_cond_init(&g_cv, NULL); // Initialize with default attributes
    return 0;
}
```
x??

---
#### Destroying Condition Variables
After use, it's important to clean up by destroying the condition variable.

:p How do you destroy a condition variable in C?
??x
In C, a condition variable should be destroyed using `pthread_cond_destroy()` after its use. This function releases any resources associated with the condition variable and ensures that no further operations can be performed on it.

```c
#include <pthread.h>

// In your cleanup function or at program exit, destroy the condition variable.
int destroyConditionVariable() {
    pthread_cond_destroy(&g_cv); // Destroy the condition variable
    return 0;
}
```
x??

---
#### Producer Thread Using Condition Variable
The producer thread updates the shared state and signals the consumer to proceed when data is ready.

:p What does the producer thread do with a condition variable?
??x
The producer thread updates the shared state (e.g., setting `g_ready` to true) and signals the consumer that it can now proceed. This is done using atomic operations in conjunction with the mutex to ensure safety.

```c
void* ProducerThreadCV(void*) {
    // Keep producing data forever...
    while (true) {
        pthread_mutex_lock(&g_mutex); // Lock the mutex before making changes

        // Produce new data into the queue
        ProduceDataInto(&g_queue);

        // Set the ready flag to true and signal the consumer
        g_ready = true;
        pthread_cond_signal (&g_cv);

        pthread_mutex_unlock(&g_mutex); // Unlock the mutex after signaling

        // Producer can continue its work or sleep briefly for simulation
    }
    return nullptr;
}
```
x??

---
#### Consumer Thread Using Condition Variable
The consumer thread waits on a condition variable until data is ready, then consumes it.

:p What does the consumer thread do with a condition variable?
??x
The consumer thread waits in a loop until the producer signals that data is available. It checks if `g_ready` is true and uses `pthread_cond_wait()` to block itself until notified by the producer.

```c
void* ConsumerThreadCV(void*) {
    // Keep consuming data forever...
    while (true) {
        pthread_mutex_lock(&g_mutex); // Lock the mutex before making changes

        // Wait for the producer to signal that data is ready
        while (!g_ready) { 
            pthread_cond_wait (&g_cv, &g_mutex); // Go to sleep until notified
        }

        // Consume the data from the queue
        ConsumeDataFrom(&g_queue);

        g_ready = false; // Reset the ready flag

        pthread_mutex_unlock(&g_mutex); // Unlock the mutex after consuming
    }
    return nullptr;
}
```
x??

---

#### Mutex Slight of Hand Technique
Mutexes are used to ensure mutual exclusion when accessing shared resources. In some scenarios, a mutex might be unlocked after a thread is put into a sleeping state and reacquired when the thread wakes up. This technique helps manage thread states efficiently.

:p Explain how a mutex can perform a "slight of hand" in managing threads.
??x
A mutex typically locks a resource to prevent concurrent access by multiple threads. When a thread needs to wait, the kernel might unlock the mutex and put the thread to sleep. Later, when the condition wakes up the thread, the kernel relocks the mutex so that it is ready for the thread's next operation.

For example:
```c
// Pseudocode for Mutex Slight of Hand
void producer() {
    while (true) {
        // Perform some work and update g_ready flag
        if (update_g_ready_flag()) {
            pthread_mutex_unlock(&mutex);
            sleep(1);  // Simulate thread going to sleep
            pthread_mutex_lock(&mutex);  // Lock the mutex again before proceeding
        }
    }
}

void consumer() {
    while (true) {
        pthread_cond_wait(&condition, &mutex);  // Wait until g_ready is true
        if (g_ready) {
            use_resource();  // Use the resource safely now that it's ready
            reset_g_ready_flag();
        }
    }
}
```
x??

---

#### Polling for Condition Variable in Consumer Thread
Even when using condition variables, a thread might be awoken spuriously by the kernel. Therefore, threads often check the condition variable multiple times to ensure they are not interrupted prematurely.

:p Why does the consumer thread still use a while loop despite using a condition variable?
??x
The `pthread_cond_wait()` function can return for reasons other than the condition becoming true (e.g., spurious wakeups). Thus, the consumer thread must continue polling in a loop until it actually detects that `g_ready` is true.

Example:
```c
// Pseudocode for Consumer Thread with Polling Loop
void consumer() {
    while (!g_ready) {  // Check g_ready multiple times to avoid spurious wakeups
        pthread_cond_wait(&condition, &mutex);
    }
    use_resource();
}
```
x??

---

#### Semaphores and Atomic Counters
A semaphore acts as an atomic counter that can be used to manage concurrent access to resources. Unlike a mutex, which allows only one thread at a time, a semaphore can allow multiple threads to acquire it simultaneously.

:p How does a semaphore differ from a mutex?
??x
While both semaphores and mutexes are synchronization mechanisms, they serve different purposes:
- **Mutex**: Ensures mutual exclusion for shared resources. Only one thread can hold the lock at any given time.
- **Semaphore**: Manages access to a pool of resources where multiple threads can acquire it simultaneously as long as there are available "slots".

Example C code with a semaphore:
```c
// Pseudocode for Initializing and Using a Semaphore
sem_t sem;

void init_semaphore(int count) {
    sem_init(&sem, 0, count); // Initialize semaphore with an initial count
}

void acquire_resource() {
    sem_wait(&sem); // Decrement the counter; block if count is zero
}

void release_resource() {
    sem_post(&sem); // Increment the counter to open up a "slot"
}
```
x??

---

#### Binary Semaphores and Mutexes
A binary semaphore has an initial value of 1, allowing for mutual exclusion similar to a mutex. However, it can be more flexible in scenarios where multiple threads might need to access a resource concurrently but not all at once.

:p What is the key difference between a binary semaphore and a regular mutex?
??x
Both a binary semaphore and a mutex ensure that only one thread accesses a critical section of code at any given time. However, a binary semaphore can be used in scenarios where multiple threads might need to access a resource but not all simultaneously:
- **Mutex**: Exclusive access; allows exactly one thread.
- **Binary Semaphore**: Can allow more than one thread (up to the initial count), providing finer control over concurrent access.

Example C code using a binary semaphore:
```c
// Pseudocode for Using a Binary Semaphore
sem_t sem;

void producer() {
    while (true) {
        acquire_resource();  // Lock resource with sem_wait
        // Perform work
        release_resource();  // Unlock resource with sem_post
    }
}
```
x??

---

---
#### Mutex vs Binary Semaphore
Mutexes and binary semaphores are both synchronization objects but serve different purposes. A mutex allows only one thread to access a resource at a time, and it can be unlocked by the same thread that locked it. In contrast, a binary semaphore has a counter that can be incremented and decremented by different threads, allowing for more flexible signaling between threads.

:p What is the key difference between a mutex and a binary semaphore?
??x
The key difference lies in how they handle locking and unlocking:
- A mutex can only be unlocked by the thread that locked it.
- A binary semaphore's counter can be incremented and decremented by different threads, meaning any thread can unlock it after it has been locked.

This difference makes semaphores more versatile for signaling between threads. For example, in a producer-consumer scenario, one semaphore (g_semUsed) signals when the consumer should proceed, while another (g_semFree) signals when the producer is ready to add data.
x??

---
#### Producer-Consumer Example with Semaphores
In this example, semaphores are used to manage the state of a shared buffer between a producer and a consumer. The `g_semUsed` semaphore tracks how many items are being consumed by the consumer, while `g_semFree` tracks how much space is available in the buffer.

:p How do you implement the producer thread using semaphores?
??x
The producer thread uses `sem_wait(&g_semFree)` to ensure there is enough free space before adding an item to the queue. After adding the item, it uses `sem_post(&g_semUsed)` to notify the consumer that there is new data available.

```cpp
void* ProducerThreadSem(void*) {
    while (true) {
        Item item = ProduceItem();
        sem_wait(&g_semFree);  // Wait until there's room in the queue.
        AddItemToQueue(&g_queue, item);
        sem_post(&g_semUsed);  // Notify consumer that data is ready.
    }
    return nullptr;
}
```
x??

---
#### Semaphore Implementation
A semaphore can be implemented using a mutex, a condition variable, and an integer counter. The `Take()` method waits until the count reaches zero, then decrements it, while the `Give()` method increments the count and wakes up waiting threads if necessary.

:p How do you implement the `Take()` function for a semaphore?
??x
The `Take()` function uses a mutex to ensure that the operations are atomic. It first locks the mutex, checks if the counter is zero, and waits until it's signaled by another thread using `pthread_cond_wait()`. If the count becomes non-zero, it decrements the counter.

```cpp
void Take() {
    pthread_mutex_lock(&m_mutex);
    while (m_count == 0) 
        pthread_cond_wait(&m_cv, &m_mutex);  // Wait until count is > 0.
    --m_count;
    pthread_mutex_unlock(&m_mutex);
}
```
x??

---
#### Windows Event Objects
Windows provides a mechanism called an event object that can be used to signal between threads. An event is initially created in the nonsignaled state, meaning no thread waits on it by default. Threads can go to sleep using `WaitForSingleObject()`, and another thread can wake them up by calling `SetEvent()`.

:p How do you implement the producer thread using Windows events?
??x
The producer thread uses `WaitForSingleObject(&g_hFree)` to wait until there is free space in the queue before adding an item. After adding the item, it calls `SetEvent(&g_hUsed)` to signal that data is ready for consumption.

```cpp
void* ProducerThreadEv(void*) {
    while (true) {
        Item item = ProduceItem();
        WaitForSingleObject(&g_hFree);  // Wait until there's room in the queue.
        AddItemToQueue(&g_queue, item);
        SetEvent(&g_hUsed);  // Notify consumer that data is ready.
    }
    return nullptr;
}
```
x??

---

