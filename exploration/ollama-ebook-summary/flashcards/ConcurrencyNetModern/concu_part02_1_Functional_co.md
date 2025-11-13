# Flashcards: ConcurrencyNetModern_processed (Part 2)

**Starting Chapter:** 1 Functional concurrency foundations

---

#### Moore's Law and CPU Speed Limits
Background context explaining the concept. Moore predicted that the density and speed of transistors would double every 18 months before reaching a maximum speed beyond which technology couldn't advance. Progress continued for almost 50 years, but now single-core CPUs have nearly reached the limit due to physical constraints like the speed of light.
The fundamental relationship between circuit length (CPU physical size) and processing speed means that shorter circuits require smaller and fewer switches, thus increasing transmission speed. The speed of light is a constant at about $3 \times 10^8$ meters per second.
:p What are the key points in Moore's Law regarding CPU speed?
??x
Moore's Law predicted a doubling of transistor density every 18 months, but as technology advanced over decades, CPUs hit physical limits. The speed of light serves as an absolute limit for data propagation, meaning that even if you increase clock speeds, signals can't travel faster than the speed of light.
Code examples:
```java
public class LightSpeedLimit {
    public static double calculatePropagationDistance(double speedOfLight, double nanoseconds) {
        return (speedOfLight * 1e-9) * nanoseconds;
    }
}
```
x??

---

#### CPU Performance Limitations
Background context explaining the concept. The single-processor CPU has nearly reached its maximum speed due to physical limitations like the speed of light and heat generation from energy dissipation.
As CPUs approach these limits, creating smaller chips was the primary approach for higher performance. However, high frequencies in small chip sizes introduce thermal issues, as power in a switching transistor is roughly $frequency^2$.
:p Why are single-core CPUs nearing their maximum speed?
??x
Single-core CPUs are nearing their maximum speed due to physical constraints such as the speed of light and heat generation from energy dissipation. Smaller chips can increase processing speed by reducing circuit length, but high frequencies in small chip sizes introduce thermal issues, increasing power consumption exponentially.
Code examples:
```java
public class ThermalIssues {
    public static double calculatePower(double frequency) {
        return Math.pow(frequency, 2);
    }
}
```
x??

---

#### Introduction to Concurrency and Multicore Processing
Background context explaining the concept. With single-core CPU performance improvement stagnating, developers are adapting by moving into multicore architecture and developing software that supports concurrency.
The processor revolution has brought parallel programming models into mainstream use, allowing for more efficient computing through multiple cores.
:p What is the main reason developers are shifting to multicore processing?
??x
Developers are shifting to multicore processing because single-core CPU performance improvement has stagnated. Multicore architecture allows for better performance by distributing tasks across multiple cores.
```java
public class MulticoreExample {
    public static void distributeTasks(int[] data, int threads) {
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        List<Future<Long>> results = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            Future<Long> result = executor.submit(() -> performTask(data[i]));
            results.add(result);
        }
        // Process results
    }

    private static long performTask(int value) {
        return value * value;
    }
}
```
x??

---

#### Concurrency vs. Parallelism vs. Multithreading
Background context explaining the concept. This section differentiates between concurrency, parallelism, and multithreading.
Concurrency involves running multiple operations in such a way that they appear to execute simultaneously but are not necessarily executed at the same time. Parallelism refers to executing tasks concurrently on multiple processors or cores. Multithreading is a specific implementation of concurrent execution within a single process.
:p What distinguishes concurrency from parallelism?
??x
Concurrency involves running multiple operations in such a way that they appear to execute simultaneously, but are not necessarily executed at the same time. Parallelism refers to executing tasks concurrently on multiple processors or cores. For example:
```java
public class ConcurrencyExample {
    public static void main(String[] args) {
        Thread thread1 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Thread 1: " + i);
                try { Thread.sleep(1000); } catch (InterruptedException e) {}
            }
        });

        Thread thread2 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Thread 2: " + i);
                try { Thread.sleep(1000); } catch (InterruptedException e) {}
            }
        });

        thread1.start();
        thread2.start();
    }
}
```
In this example, both threads appear to run concurrently but are not necessarily executed at the same time.
x??

---

#### Avoiding Common Pitfalls in Concurrency
Background context explaining the concept. This section covers common pitfalls when writing concurrent applications such as race conditions and deadlocks.
:p What is a common pitfall in concurrency?
??x
A common pitfall in concurrency is the risk of race conditions, where the output depends on the sequence or timing of uncontrollable events. For example:
```java
public class RaceConditionExample {
    private int count = 0;

    public void increment() {
        count++;
    }

    public int getCount() { return count; }
}
```
If multiple threads call `increment` and `getCount` in a non-atomic manner, the final value of `count` may not be accurate.
x??

---

#### Sharing Variables Between Threads
Background context explaining the concept. This section discusses the challenges of sharing variables between threads and how to address them using synchronization mechanisms like locks or atomic operations.
:p How can you safely share a variable between threads?
??x
You can safely share a variable between threads by using synchronization mechanisms such as locks or atomic operations. For example, using a synchronized block:
```java
public class ThreadSafeExample {
    private int count = 0;

    public synchronized void increment() {
        count++;
    }

    public synchronized int getCount() { return count; }
}
```
This ensures that only one thread can execute the `increment` method at a time, preventing race conditions.
x??

---

#### Functional Paradigm for Concurrency
Background context explaining the concept. This section introduces the functional paradigm as a way to develop concurrent programs by avoiding shared mutable state and side effects.
:p What is the advantage of using the functional paradigm in concurrency?
??x
The advantage of using the functional paradigm in concurrency is that it avoids shared mutable state and side effects, making code easier to reason about and less prone to race conditions. For example:
```java
public class FunctionalConcurrencyExample {
    public int process(int input) {
        return input * 2;
    }
}
```
This function `process` operates on inputs without changing any external state, making it thread-safe.
x??

---

#### Introduction to Concurrency and Functional Paradigm

Background context: This section introduces the importance of concurrency and functional programming in modern software development. It explains how traditional concurrent programs face challenges, especially with shared state and race conditions. The benefits of using a functional paradigm are highlighted as it provides a way to handle concurrency more safely by avoiding side effects and mutable data structures.

:p What is the main focus of this chapter?
??x
The main focus is on understanding why concurrency is valued in programming models and how the functional paradigm can help in writing correct concurrent programs.
x??

---

#### Traditional vs. Functional Paradigm

Background context: This section contrasts traditional programming paradigms with functional ones, emphasizing that traditional approaches often struggle with concurrency due to shared state and mutable data. In contrast, functional programming encourages immutable data structures and avoids side effects, making it easier to handle concurrency.

:p What are the main challenges of writing concurrent programs in a traditional paradigm?
??x
The main challenges include managing shared state, avoiding race conditions, and dealing with thread safety issues due to mutable data and global variables.
x??

---

#### Benefits of Functional Programming for Concurrency

Background context: The chapter explains how functional programming can simplify concurrent programming by leveraging immutable data structures and pure functions. This approach reduces the likelihood of bugs related to shared state.

:p How does functional programming help in handling concurrency?
??x
Functional programming helps in handling concurrency by encouraging the use of immutable data structures, which can be safely passed between threads without worrying about race conditions or shared state. Pure functions also avoid side effects, making it easier to reason about and test concurrent programs.
x??

---

#### Asynchronous Operations with Task Parallel Library

Background context: This section covers how to combine asynchronous operations with the Task Parallel Library (TPL) in .NET applications. It explains how TPL can be used to perform parallel tasks without blocking the main thread, improving application responsiveness.

:p What is the Task Parallel Library (TPL)?
??x
The Task Parallel Library (TPL) is a framework for managing concurrent and parallel programming in .NET. It provides a set of classes that simplify the creation, scheduling, and management of tasks.
x??

---

#### Concurrency Models with Functional Paradigm

Background context: This section introduces various concurrency models that adopt functional paradigms such as functional, asynchronous, event-driven, and message-passing (with agents and actors). These models are designed to handle complex concurrent issues in a declarative way.

:p What are some concurrency models discussed in this chapter?
??x
The chapter discusses several concurrency models including functional, asynchronous, event-driven, and message-passing with agents and actors.
x??

---

#### Building High-Performance Concurrent Systems

Background context: This section explains how to build high-performance concurrent systems using the functional paradigm. It emphasizes the importance of immutable data structures and pure functions in achieving performance and reliability.

:p How does functional programming contribute to building high-performance concurrent systems?
??x
Functional programming contributes to building high-performance concurrent systems by promoting the use of immutable data structures, which can be shared safely between threads without locks or synchronization mechanisms. Pure functions ensure that operations are deterministic and side-effect-free, making it easier to reason about and optimize performance.
x??

---

#### Declarative Style for Asynchronous Computations

Background context: This section introduces how to express and compose asynchronous computations in a declarative style using functional programming concepts. It explains the benefits of this approach in terms of simplicity and readability.

:p What is declarative programming, and why is it useful for asynchronous computations?
??x
Declarative programming is about specifying what computation should be done without describing the steps that must be taken to carry out the computation. For asynchronous computations, this means focusing on the intended outcome rather than the process. It enhances readability and maintainability by abstracting away implementation details.
x??

---

#### Data-Parallel Programming

Background context: This section covers how data-parallel programming can be used to accelerate sequential programs in a pure fashion. It explains the principles of applying functional programming techniques to parallelize computations.

:p What is data-parallel programming, and how does it differ from traditional parallel programming?
??x
Data-parallel programming involves executing the same operation on multiple chunks of data simultaneously. It differs from traditional parallel programming by focusing on operations that can be applied in a distributed or parallel manner without explicit thread management.
x??

---

#### Reactive Programming with Rx

Background context: This section introduces reactive programming and event-based programs using Rx-style event streams. It explains how to use Rx to handle asynchronous events and transform data streams.

:p What is reactive programming, and what are some key benefits of using Rx?
??x
Reactive programming is about responding to changes in data or events in a way that's both responsive and efficient. Using Rx (Reactive Extensions), you can observe sequences of values and apply transformations on these sequences in an event-driven manner. Key benefits include simplified handling of asynchronous data streams, improved responsiveness, and easier composition of asynchronous operations.
x??

---

#### Functional Concurrent Collections

Background context: This section discusses the use of functional concurrent collections for building lock-free multi-threaded programs. It explains how immutable structures can be used to avoid locking and achieve thread safety.

:p What are some benefits of using functional concurrent collections?
??x
The benefits include improved scalability, reduced contention between threads, and automatic management of state, which simplifies the implementation of concurrent algorithms.
x??

---

#### Sequential Programming
Sequential programming involves accomplishing tasks step-by-step, where one task must be completed before moving on to the next. This method is easy to understand and reduces the likelihood of errors but can be inefficient because it leaves resources idle during waiting periods.

:p What is sequential programming?
??x
Sequential programming refers to a method of executing tasks in a linear, ordered sequence where each step is completed before the next one begins. For instance, in making a cappuccino, the barista must grind the coffee, brew the coffee, steam and froth the milk, and combine them sequentially.
```java
public class CappuccinoMaker {
    public void makeCappuccino() {
        // Step 1: Grind Coffee
        System.out.println("Grinding Coffee...");
        // Simulate grinding time
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        // Step 2: Brew Coffee
        System.out.println("Brewing Coffee...");
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        // Step 3: Steam Milk
        System.out.println("Steaming Milk...");
        try {
            Thread.sleep(750);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        // Step 4: Froth Milk
        System.out.println("Frothing Milk...");
        try {
            Thread.sleep(750);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        // Step 5: Combine Coffee and Milk
        System.out.println("Combining Coffee and Milk...");
    }
}
```
x??

---

#### Concurrent Programming
Concurrent programming allows the execution of multiple tasks at the same time, giving an illusion of multitasking. In reality, only one task is executed at a time due to resource sharing.

:p How does concurrent programming differ from sequential programming?
??x
Concurrent programming involves executing multiple tasks simultaneously or in parallel, which appears to be multitasking but is actually achieved by switching between tasks rapidly. For example, a barista can start brewing coffee while grinding the next batch of beans and steaming milk for another customer.

:p What is an example of concurrent programming?
??x
An example is a barista who prepares multiple cappuccinos simultaneously. The barista starts brewing one drink while grinding coffee for another order and frothing milk for yet another customer.
```java
public class Barista {
    public void prepareCappuccinos() {
        // Simultaneously perform tasks in an interleaved manner
        new Thread(() -> {
            System.out.println("Grinding Coffee...");
            try { Thread.sleep(500); } catch (InterruptedException e) { e.printStackTrace(); }
        }).start();
        
        new Thread(() -> {
            System.out.println("Brewing Coffee...");
            try { Thread.sleep(1000); } catch (InterruptedException e) { e.printStackTrace(); }
        }).start();
        
        new Thread(() -> {
            System.out.println("Steaming Milk...");
            try { Thread.sleep(750); } catch (InterruptedException e) { e.printStackTrace(); }
        }).start();
    }
}
```
x??

---

#### Parallel Programming
Parallel programming executes multiple tasks simultaneously on different cores, improving application performance and throughput.

:p What is parallel programming?
??x
Parallel programming involves executing multiple tasks in parallel across different cores or processors. This method aims to fully utilize available computational resources by running tasks concurrently. For instance, hiring a second barista in a coffee shop allows them to work in parallel on separate stations, increasing overall cappuccino production speed.

:p How can we implement parallelism in Java?
??x
Parallelism in Java can be implemented using multi-threading or other concurrency libraries like Fork/Join framework. For example, two baristas working simultaneously at different coffee stations:

```java
public class Barista {
    public void prepareCappuccinos() throws InterruptedException {
        // Simultaneously perform tasks on separate threads
        Thread t1 = new Thread(() -> {
            System.out.println("Grinding Coffee for Barista 2...");
            try { Thread.sleep(500); } catch (InterruptedException e) { e.printStackTrace(); }
        });
        
        Thread t2 = new Thread(() -> {
            System.out.println("Brewing Coffee for Barista 1...");
            try { Thread.sleep(1000); } catch (InterruptedException e) { e.printStackTrace(); }
        });
        
        t1.start();
        t2.start();
    }
}
```
x??

---

#### Concurrency vs. Parallelism
Concurrency and parallelism are related but distinct concepts in programming. Concurrency refers to the ability to run several programs or parts of a program simultaneously, while parallelism involves executing tasks concurrently on multiple cores.

:p What is the difference between concurrency and parallelism?
??x
Concurrent programming allows for running multiple tasks at the same time with an illusion of multitasking. However, in practice, only one task runs at any given time due to resource sharing. Parallelism, on the other hand, involves executing tasks concurrently across multiple cores or processors.

:p Can concurrency be achieved without parallelism?
??x
Yes, concurrency can be achieved through techniques like context switching (preemptive multitasking) where a single-core CPU rapidly switches between different threads, giving the appearance of running multiple tasks at once. However, this does not utilize all available computational resources and is limited by the number of cores.

:p How do we ensure efficient parallelism in a multi-core environment?
??x
Efficient parallelism requires proper thread management and task distribution across multiple cores to maximize resource utilization. This can be achieved using tools like Java's Fork/Join framework, which manages tasks and ensures load balancing.
```java
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class ParallelTask extends RecursiveAction {
    @Override
    protected void compute() {
        // Perform task logic here
    }
    
    public static void main(String[] args) {
        ForkJoinPool pool = new ForkJoinPool();
        pool.invoke(new ParallelTask());
    }
}
```
x??

---

#### Multitasking vs. Concurrent Programming
Multitasking is a broader term used in operating systems to manage the execution of multiple processes or tasks, while concurrent programming specifically refers to executing parts of a program simultaneously.

:p What does multitasking refer to in computer science?
??x
Multitasking in computer science involves managing and scheduling multiple processes or tasks so that they appear to execute concurrently. This can be achieved through context switching by the operating system, which rapidly switches between running programs.

:p How is concurrent programming different from traditional multitasking?
??x
Concurrent programming focuses on executing parts of a program simultaneously within a single application, whereas traditional multitasking involves managing multiple applications or processes across an entire system.
```java
public class MultitaskingExample {
    public static void main(String[] args) {
        // Simulate two tasks running in parallel using context switching
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Task 1: " + i);
                try { Thread.sleep(100); } catch (InterruptedException e) { e.printStackTrace(); }
            }
        });
        
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                System.out.println("Task 2: " + i);
                try { Thread.sleep(100); } catch (InterruptedException e) { e.printStackTrace(); }
            }
        });
        
        t1.start();
        t2.start();
    }
}
```
x??

---

