# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 2)

**Starting Chapter:** 1 Why parallel computing

---

#### Why Parallel Computing?

Background context: In today's world, various applications require extensive and efficient use of computing resources. These applications include scientific modeling, artificial intelligence (AI), machine learning, and more.

:p What is the primary reason for the growing importance of parallel computing?
??x
Parallel computing becomes important because it allows many operations to be executed simultaneously at a single instance in time, thereby handling larger problems and datasets faster. This is crucial as typical applications often leave much of the compute capability untapped.

Example: For modeling megafires or tsunamis, running simulations ten, a hundred, or even a thousand times faster can significantly improve response times and accuracy.
```java
// Pseudocode for parallel simulation execution
public void runParallelSimulations(int numSimulations) {
    List<Future<SimulationResult>> futures = new ArrayList<>();
    ExecutorService executor = Executors.newFixedThreadPool(numSimulations);

    for (int i = 0; i < numSimulations; i++) {
        Future<SimulationResult> future = executor.submit(new SimulationTask());
        futures.add(future);
    }

    // Wait for all simulations to complete
    for (Future<SimulationResult> future : futures) {
        try {
            future.get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }
}
```
x??

---

#### What is Parallel Computing?

Background context: Parallel computing involves executing many operations simultaneously. It requires effort from the programmer to identify and expose potential parallelism, properly leverage resources for simultaneous execution.

:p Define what parallel computing is.
??x
Parallel computing is the practice of identifying and exposing parallelism in algorithms, expressing this in software, and understanding the costs, benefits, and limitations of the chosen implementation. The goal is to enhance performance by executing operations concurrently.

Example: In a simple algorithm like sorting an array, you could sort elements in parallel if no dependencies exist between them.
```java
// Pseudocode for parallel sorting using multiple threads
public void parallelSort(int[] arr) {
    int threadCount = Runtime.getRuntime().availableProcessors();
    List<Future<Integer>> futures = new ArrayList<>();
    ExecutorService executor = Executors.newFixedThreadPool(threadCount);

    // Divide array into chunks and sort each chunk in parallel
    for (int i = 0; i < arr.length / threadCount; i++) {
        Future<Integer> future = executor.submit(new SortChunkTask(arr, i * threadCount, Math.min((i + 1) * threadCount, arr.length)));
        futures.add(future);
    }

    // Wait for all threads to complete
    for (Future<Integer> future : futures) {
        try {
            future.get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }
}
```
x??

---

#### Potential Parallelism

Background context: Identifying and exposing potential parallelism is crucial in developing parallel applications. It involves certifying that operations can be conducted in any order as system resources become available.

:p What does it mean by "potential parallelism" or "concurrency"?
??x
Potential parallelism, also known as concurrency, means identifying parts of an algorithm where operations can be executed independently and safely without dependencies. This allows the programmer to leverage multiple cores or threads to perform these operations simultaneously.

Example: In a checkout process at a supermarket, unloading items from a basket and scanning them can be done concurrently because there is no dependency between these tasks.
```java
// Pseudocode for concurrent item processing in a checkout line
public void processCheckout(Item[] items) {
    List<Future<Item>> futures = new ArrayList<>();
    ExecutorService executor = Executors.newFixedThreadPool(items.length);

    // Scan each item and add to future results
    for (Item item : items) {
        Future<Item> future = executor.submit(new ScanTask(item));
        futures.add(future);
    }

    // Collect all scanned items from futures
    List<Item> scannedItems = futures.stream().map(future -> {
        try {
            return future.get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }).collect(Collectors.toList());

    // Process collected items further, if needed.
}
```
x??

---

#### Practical Examples of Parallelism

Background context: The text provides various practical examples where parallel computing can be applied to solve complex problems more efficiently. These include scenarios like modeling natural disasters, virus spread analysis, and more.

:p Give an example scenario from the text that demonstrates the application of parallel computing.
??x
One example is the modeling of megafires. By using parallel computing, fire crews and public safety officials can run simulations faster to assist in firefighting efforts and provide timely information to the public. Parallel execution allows for the rapid processing of large datasets related to fire spread, weather conditions, etc.

Example: Simulating a megafire with multiple threads could involve dividing the fire area into zones and simulating each zone's behavior independently.
```java
// Pseudocode for parallel megafire simulation
public void simulateMegafire(FireArea fireArea) {
    List<Future<MegafireSimulationResult>> futures = new ArrayList<>();
    ExecutorService executor = Executors.newFixedThreadPool(fireArea.getNumZones());

    // Simulate each zone in parallel
    for (int i = 0; i < fireArea.getNumZones(); i++) {
        Future<MegafireSimulationResult> future = executor.submit(new ZoneSimulationTask(fireArea, i));
        futures.add(future);
    }

    // Wait for all simulations to complete and collect results
    List<MegafireSimulationResult> results = futures.stream().map(future -> {
        try {
            return future.get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }).collect(Collectors.toList());

    // Use the collected results for decision-making or further analysis.
}
```
x??

---

#### Why Parallel Computing Matters

Parallel computing has become increasingly important due to the plateauing of serial performance improvements. As hardware designs have reached their limits for clock frequency, power consumption, and miniaturization, increasing computational cores is a new approach to enhance performance.

:p What factors have limited advancements in serial processing speed?
??x
The key factors that have limited advancements in serial processing speed are:

- **Miniaturization Limits**: The physical constraints of semiconductor technology.
- **Clock Frequency Plateauing**: Increasing clock speeds beyond a certain point results in higher power consumption and heat issues, leading to diminishing returns.
- **Power Consumption**: Higher clock frequencies require more energy, which can lead to overheating and reduced performance.

x??

---

#### Trends in CPU Performance

The trends show that from 1970 to 2005, there was a steady increase in single-thread performance, CPU clock frequency, and power consumption. However, after 2005, the number of cores started increasing while clock frequency and power consumption plateaued.

:p What does Figure 1.2 illustrate about trends in CPU performance?
??x
Figure 1.2 illustrates that from 1970 to 2005, there was a steady increase in single-thread performance, CPU clock frequency, and power consumption. However, around 2005, the number of cores began to rise while clock frequency and power consumption stabilized. This shift indicates a move towards parallel processing as the primary means for increasing performance.

x??

---

#### Hyperthreading

Hyperthreading is a feature introduced by Intel that allows a single physical core to appear as two logical cores to the operating system (OS). It achieves this by interleaving work between two instruction queues and the hardware logic units.

:p What is hyperthreading and how does it work?
??x
Hyperthreading is an Intel technology that allows a single physical core to functionally appear as two cores. This works by having two instruction queues that interleave instructions, allowing both threads to be processed in parallel on the same core's hardware logic units.

:p Provide pseudocode for hyperthreading functionality.
??x
```java
class Core {
    InstructionQueue queue1;
    InstructionQueue queue2;

    public void executeThreads() {
        Thread threadA = new Thread();
        Thread threadB = new Thread();

        while (!threadA.isComplete() || !threadB.isComplete()) {
            // Interleave work between the two instruction queues
            if (queue1.hasNextInstruction()) {
                queue1.nextInstruction().execute();
            }
            if (queue2.hasNextInstruction()) {
                queue2.nextInstruction().execute();
            }

            threadA.updateState();  // Update state of Thread A
            threadB.updateState();  // Update state of Thread B
        }
    }
}
```
x??

---

#### Vector Processors

Vector processors execute multiple instructions simultaneously. The width in bits of the vector processor (vector unit) specifies the number of instructions to be executed concurrently.

:p Explain vector processors and their functionality.
??x
Vector processors are designed to execute multiple instructions at once, leveraging a wider bit-width for the vector unit. For example, a 256-bit-wide vector unit can handle four 64-bit (doubles) or eight 32-bit (single-precision) instructions simultaneously.

:p Calculate the parallelism of a hypothetical system with 16 cores, hyperthreading, and a 256-bit wide vector unit.
??x
Given:
- 16 physical cores
- Hyperthreading: 2 threads per core = 32 logical cores
- 256-bit wide vector unit (can execute 4 double or 8 single precision instructions at once)

Calculation of parallelism:
```
16 cores × 2 hyperthreads × (256 bit-wide vector unit) / (64-bit double)
= 16 × 2 × (256/64)
= 32 × 4
= 128-way parallelism

This means the system can execute 128 parallel paths while a serial application only uses about 0.8% of this potential.
```

x??

---

#### Example: Performance Calculation

An example demonstrates that with a 16-core CPU, hyperthreading, and a 256-bit wide vector unit, a serial program using one core and no vectorization only utilizes 0.8% of the processor's theoretical capability.

:p What is the percentage of processing power used by a serial application on this system?
??x
A serial application running on a 16-core CPU with hyperthreading (2 threads per physical core) and a 256-bit wide vector unit uses only about 0.8% of its theoretical processing capability.

Calculation:
```
16 cores × 2 hyperthreads × (256 bit-wide vector unit) / (64-bit double)
= 32 logical cores × (256/64)
= 32 × 4
= 128 parallelism

Since there is only one serial path, the percentage of processing power used by a serial application is:
(1 / 128) × 100% ≈ 0.78125%, or approximately 0.8%
```

x??

---

#### Speedup and Performance Analysis
Parallel computing can significantly reduce an application’s run time, often referred to as speedup. This is a primary goal of parallel programming.

:p What is speedup and how does it relate to parallel computing?
??x
Speedup measures the factor by which running an algorithm in parallel reduces its execution time compared to running it sequentially. It is calculated using the formula:

\[ \text{Speedup} = \frac{\text{Time taken by serial program}}{\text{Time taken by parallel program}} \]

For example, if a serial program takes 10 seconds and a parallel version of the same program takes 2 seconds to run on multiple cores, then the speedup would be:

\[ \text{Speedup} = \frac{10}{2} = 5 \]

This indicates that the parallel version is five times faster than the serial version.

In practice, theoretical speedup can often be higher due to scalability and efficiency issues. However, achieving high practical speedup requires careful design and optimization of the application for parallel execution.
x??

---

#### Larger Problem Sizes with More Compute Nodes
Parallel computing allows you to handle larger problem sizes by utilizing more compute nodes. The key idea is that the amount of available computational resources determines what can be accomplished.

:p How does parallel computing enable handling larger problems?
??x
Parallel computing enables handling larger problems by distributing tasks across multiple processors or nodes, thereby increasing the total computational capacity available for a given application. For instance, if you have an algorithm that needs to process a large dataset, running it on more cores or nodes can significantly increase the amount of data processed.

Consider an example where a serial program processes 10GB of data in one hour:

- If you run this same program on two parallel processors, each processing half the data, theoretically, it could be completed in about 30 minutes.
- With four parallel processors, the time would halve again to around 15 minutes.

Thus, by increasing the number of compute nodes (processors), you can process larger datasets more efficiently and achieve better scalability.

Code Example:
```java
public class ParallelDataProcessor {
    public static void main(String[] args) {
        int totalData = 10_000_000_000; // 10GB in bytes
        int numProcessors = Runtime.getRuntime().availableProcessors();

        long startTime = System.currentTimeMillis();
        
        // Parallel processing logic here

        long endTime = System.currentTimeMillis();
        double timeTaken = (endTime - startTime) / 1000.0;
        System.out.println("Time taken: " + timeTaken + " seconds");
    }
}
```
x??

---

#### Energy Efficiency by Doing More with Less
Energy efficiency in parallel computing refers to the ability to perform more tasks using less power, which is increasingly important as devices become smaller and more energy-conscious.

:p How does parallel computing contribute to energy efficiency?
??x
Parallel computing contributes to energy efficiency by enabling applications to run faster or use fewer resources, thereby reducing overall power consumption. This is particularly significant in mobile devices and remote sensors where battery life and power usage are critical concerns.

One way this works is through the use of more efficient processors like GPUs that can handle specific tasks with lower power requirements compared to traditional CPUs. By offloading work from the main CPU to these specialized parallel processors, overall energy consumption can be reduced.

Example: Suppose a multimedia application running on a mobile device uses 120W for 24 hours in serial mode:
\[ \text{Energy Usage} = 120 \, \text{W} \times 24 \, \text{hours} = 2880 \, \text{kWhrs} \]

By using a GPU with a thermal design power of 300W, the energy usage could be reduced to:
\[ \text{Energy Usage} = 300 \, \text{W} \times 24 \, \text{hours} = 720 \, \text{kWhrs} \]

In this example, even though the application might take longer to run on the GPU, the overall energy savings could be significant.

Code Example:
```java
public class EnergyEfficiencyCalculator {
    public static void main(String[] args) {
        double powerUsageSerial = 120; // in W
        double timeHours = 24;
        
        double energyUsageSerial = powerUsageSerial * timeHours;

        double powerUsageParallelGPU = 300; // in W
        double energyUsageParallelGPU = powerUsageParallelGPU * timeHours;
        
        System.out.println("Energy usage (serial): " + energyUsageSerial);
        System.out.println("Energy usage (parallel GPU): " + energyUsageParallelGPU);
    }
}
```
x??

---

#### Cost Reduction through Parallel Computing
Parallel computing can reduce the actual monetary cost of running applications by optimizing resource utilization and reducing power consumption.

:p How does parallel computing help in reducing costs?
??x
Parallel computing helps in reducing costs primarily through two mechanisms:

1. **Reduced Energy Consumption**: By utilizing more efficient processors or accelerator devices like GPUs, the overall energy usage can be minimized. This is crucial for applications running on battery-powered devices and remote sensors.

2. **Optimized Resource Utilization**: Parallel processing can distribute tasks across multiple cores or nodes, thereby reducing the need for high-end hardware and minimizing idle time.

For instance, if a serial program uses 120W of power over 24 hours:
\[ \text{Energy Usage} = 120 \, \text{W} \times 24 \, \text{hours} = 2880 \, \text{kWhrs} \]

By using a GPU with a thermal design power (TDP) of 300W to achieve the same results in less time:
\[ \text{Energy Usage} = 300 \, \text{W} \times 24 \, \text{hours} = 720 \, \text{kWhrs} \]

Here, even though the TDP is higher for the GPU, the reduced run time can lead to significant cost savings over a long period.

Code Example:
```java
public class CostCalculator {
    public static void main(String[] args) {
        double powerUsageSerial = 120; // in W
        double timeHours = 24;
        
        double energyUsageSerial = powerUsageSerial * timeHours;

        double powerUsageParallelGPU = 300; // in W
        double energyUsageParallelGPU = powerUsageParallelGPU * (timeHours / 2); // Assuming half the run time

        System.out.println("Energy usage (serial): " + energyUsageSerial);
        System.out.println("Energy usage (parallel GPU): " + energyUsageParallelGPU);
    }
}
```
x??

---

