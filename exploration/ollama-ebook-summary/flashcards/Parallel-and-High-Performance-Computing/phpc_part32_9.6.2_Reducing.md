# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 32)

**Starting Chapter:** 9.6.2 Reducing energy use with GPUs

---

#### Energy Costs for Parallel Applications
Background context: As computing systems advance, energy costs are becoming a significant concern. The cost of operating computers and their associated infrastructure has increased significantly over time. This is especially relevant as we approach exascale computing where power requirements need to be tightly managed.

The energy consumption for an application can be estimated using the formula:
\[ \text{Energy} = (N \, \text{Processors}) \times (R \, \text{Watts/Processor}) \times (T \, \text{hours}) \]

:p What is the formula used to estimate the energy consumption of an application?
??x
The formula for estimating energy consumption is:
\[ \text{Energy} = (N \, \text{Processors}) \times (R \, \text{Watts/Processor}) \times (T \, \text{hours}) \]
This helps in understanding how much power an application consumes over a specific period.

x??

---

#### Comparing GPU and CPU Systems
Background context: The text compares the energy consumption between a GPU-based system and a CPU-based system for a 10 TB/sec bandwidth application. It highlights that while GPUs typically have higher thermal design power (TDP) compared to CPUs, they can reduce overall run time or require fewer processors.

:p How does the text compare the energy costs of a GPU system versus a CPU system?
??x
The text compares the energy costs by using the given specifications:
- NVIDIA V100: 12 GPUs at $11,000 each and 300 watts per GPU.
- Intel Skylake Gold 6152: 45 processors (CPUs) at $3,800 each and 140 watts per CPU.

The energy consumption for one day is calculated as:
\[ \text{Energy} = (N \, \text{Processors}) \times (R \, \text{Watts/Processor}) \times (T \, \text{hours}) \]

For the GPU system: 
\[ 12 \times 300 \times 24 = 86.4 \, \text{kWhrs} \]
For the CPU system:
\[ 45 \times 140 \times 24 = 151.2 \, \text{kWhrs} \]

The GPU system consumes less energy than the CPU system.

x??

---

#### Reducing Energy Usage in Applications
Background context: The text discusses strategies to reduce energy usage by focusing on application parallelism and efficient resource utilization. It mentions that achieving significant energy cost reductions requires applications to expose sufficient parallelism and efficient use of device resources.

:p What are some strategies mentioned for reducing energy costs through GPU accelerators?
??x
Some strategies include:
1. Exposing sufficient parallelism in the application.
2. Efficiently utilizing the GPU’s resources.
3. Reducing run time by leveraging the GPU's high processing capabilities.

For instance, running an application on 12 GPUs might reduce the overall energy consumption compared to using 45 fully subscribed CPU processors for the same amount of time.

x??

---

#### Example of Plotting Power and Utilization
Background context: The text refers to plotting power and utilization for a V100 GPU as part of examining energy consumption. This involves measuring how much power is used under different levels of utilization.

:p How can you plot the power and utilization for a V100 GPU?
??x
Plotting power and utilization for a V100 GPU involves monitoring the GPU's power draw while varying its workload. Here is an example in pseudocode:
```pseudocode
function plotPowerUtilization(gpu):
    for each level of utilization from 0 to 100:
        set gpu utilization to current level
        measure current power consumption
        store (utilization, power) pair
    draw graph with utilization on x-axis and power on y-axis
```

This helps in understanding the relationship between GPU workload and energy consumption.

x??

---

#### TDP and Energy Consumption Calculation for CPU
Background context: This concept explains how to calculate energy consumption based on Thermal Design Power (TDP) of processors. The TDP is a specification that defines the maximum amount of power the processor can consume at full load, which helps in estimating its energy usage.

:p How do you calculate the estimated energy usage for an application running on multiple CPUs?
??x
To estimate the energy usage, you need to multiply the number of processors by their TDP and then by the duration of the run. The formula is:
\[ \text{Energy} = (\text{Number of Processors}) \times (\text{TDP per Processor in W}) \times (\text{Duration in Hours}) \]

For example, if you use 45 Intel’s 22 core Xeon Gold 6152 processors for 24 hours:
\[ \text{Energy} = (45) \times (140 \, \text{W}) \times (24 \, \text{hrs}) = 151.2 \, \text{kWhrs} \]

This calculation helps in understanding the energy consumption of the application.
x??

---
#### TDP and Energy Consumption Calculation for GPU
Background context: This concept explains how to calculate energy consumption based on Thermal Design Power (TDP) of GPUs. The TDP is a specification that defines the maximum amount of power the GPU can consume at full load, which helps in estimating its energy usage.

:p How do you calculate the estimated energy usage for an application running on multiple GPUs?
??x
To estimate the energy usage, you need to multiply the number of GPUs by their TDP and then by the duration of the run. The formula is:
\[ \text{Energy} = (\text{Number of GPUs}) \times (\text{TDP per GPU in W}) \times (\text{Duration in Hours}) \]

For example, if you use 12 NVIDIA Tesla V100 GPUs for 24 hours and each has a maximum TDP of 300 W:
\[ \text{Energy} = (12) \times (300 \, \text{W}) \times (24 \, \text{hrs}) = 86.4 \, \text{kWhrs} \]

This calculation helps in understanding the energy consumption of the application and comparing it with CPU-only versions.
x??

---
#### Monitoring GPU Power Consumption
Background context: This concept explains how to monitor the power consumption of GPUs using tools like `nvidia-smi dmon`. The data collected can be used to understand the power usage patterns over time, which is crucial for optimizing energy efficiency.

:p How do you use nvidia-smi to collect performance metrics including power and GPU utilization?
??x
You can use the following command before running your application:
```sh
nvidia-smi dmon -i 0 --select pumct -c 65 --options DT --filename gpu_monitoring.log &
```

- `-i 0`: Queries GPU device 0.
- `--select pumct`: Selects power [p], utilization [u], memory usage [m], clocks [c], PCI throughput [t].
- `-c 65`: Collects 65 samples. Default time is 1 second.
- `--options DT`: Prepends monitoring data with date in YYYMMDD format and time in HH:MM::SS format, respectively.
- `--filename <name>`: Writes output to the specified filename.

This command runs in the background so you can run your application simultaneously.
x??

---
#### Plotting Power and Utilization Data
Background context: This concept explains how to plot power consumption and GPU utilization data using Python and matplotlib. The collected data helps in visualizing energy usage over time, which is useful for optimization purposes.

:p How do you read and process the monitoring log file to plot the power and utilization data?
??x
You can use the following code snippet to read the log file, process it, and plot the data:

```python
import matplotlib.pyplot as plt
import numpy as np
import re

gpu_power = []
gpu_time = []
sm_utilization = []

# Collect the data from the file, ignore empty lines
with open('gpu_monitoring.log', 'r') as data:
    count = 0
    energy = 0.0
    nominal_energy = 0.0
    for line in data:
        if re.match('^2019', line):
            line = line.rstrip(" ")
            dummy, dummy, dummy, gpu_power_in, dummy, dummy, sm_utilization_in, \
                dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy = line.split()
            if (float(sm_utilization_in) > 80):
                gpu_power.append(float(gpu_power_in))
                sm_utilization.append(float(sm_utilization_in))
                gpu_time.append(count)
                count += 1
                energy += float(gpu_power_in)*1.0
                nominal_energy += 300.0*1.0

print(energy, "watts-secs", simps(gpu_power, gpu_time))
print(nominal_energy, "watts-secs", " ratio ", energy/nominal_energy*100.0)

plt.figure()
ax1 = plt.subplot()
ax1.plot(gpu_time, gpu_power, 'o', linestyle='-', color='red')
ax1.fill_between(gpu_time, gpu_power, color='orange')
ax1.set_xlabel('Time (secs)', fontsize=16)
ax1.set_ylabel('Power Consumption (watts)', fontsize=16, color='red')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(gpu_time, sm_utilization, 'o', linestyle='-', color='green')
ax2.set_ylabel('GPU Utilization (percent)', fontsize=16, color='green')

plt.tight_layout()
plt.savefig("power.pdf")
plt.savefig("power.svg")
plt.savefig("power.png", dpi=600)
plt.show()
```

This script reads the log file and processes it to plot power consumption and GPU utilization over time. The plot helps in visualizing energy usage patterns.
x??

---

---
#### GPU Power Consumption and Utilization
Background context: The provided text discusses the power consumption and utilization of GPUs, noting that even at full utilization (100 percent), the actual power usage is significantly lower than the nominal specification. For a V100 GPU, the power rate was found to be around 61 percent of its nominal power specification. At idle, it consumed only about 20 percent of the nominal amount.
:p What does the text indicate about the actual power consumption of GPUs compared to their nominal specifications?
??x
The text indicates that the actual power consumption of GPUs is significantly lower than their nominal specifications, especially when not fully utilized. For instance, a V100 GPU operates at 61 percent of its nominal power specification even under full utilization and consumes only about 20 percent of the nominal amount at idle.
x??

---
#### Real Power Usage Rate vs Nominal Specification
Background context: The text explicitly mentions that GPUs consume less power than their nominal specifications, with a specific example given for a V100 GPU. This difference is highlighted through integration under the curve method to calculate energy usage.
:p How does integrating the area under the power curve help in calculating energy usage?
??x
Integrating the area under the power curve helps in calculating the actual energy usage over time. For instance, using Python's `scipy.integrate.simps` function can approximate this integration accurately by summing up trapezoidal areas under the curve.
```python
import numpy as np
from scipy import integrate

# Example data for power consumption over time (in watts)
time = np.linspace(0, 60, 120)  # Time in seconds
power_consumption = np.sin(time / 5.0) * 100 + 200  # Power in watts

# Integrate to get energy usage
energy_usage = integrate.simps(power_consumption, time)
print(f"Energy Usage: {energy_usage} Joules")
```
x??

---
#### Energy Consumption and Utilization Example
Background context: The provided text gives an example of integrating the area under a power curve for the CloverLeaf problem running on a V100 GPU. It shows how to calculate energy usage by integrating the power consumption over time.
:p How can we calculate the energy usage from the given power data?
??x
To calculate the energy usage, you need to integrate the power consumption over time. Using Python's `scipy.integrate.simps` function:
```python
import numpy as np
from scipy import integrate

# Example data for power consumption over time (in watts)
time = np.linspace(0, 60, 120)  # Time in seconds
power_consumption = [x for x in range(1040)]  # Power in watts over the given time period

# Integrate to get energy usage
energy_usage = integrate.simps(power_consumption, time)
print(f"Energy Usage: {energy_usage} Joules")
```
The `integrate.simps` function uses the Simpson's rule to approximate the integral of the power consumption data.
x??

---
#### CPU Power Consumption and Utilization
Background context: The text mentions that CPUs also consume less power than their nominal specifications, although not by as great a percentage as GPUs. This is an additional observation to the main focus on GPU power consumption.
:p What does the text indicate about the power consumption of CPUs compared to their nominal specifications?
??x
The text indicates that CPUs also consume less power than their nominal specifications, but likely not by as significant a percentage as GPUs do. For example, at full utilization, the power rate might be around 61 percent for GPUs and possibly a similar or slightly lower percentage for CPUs.
x??

---
#### Parallel Efficiency and Energy Savings
Background context: The text discusses the trade-offs between running multiple jobs on different numbers of processors and the associated energy savings. It introduces Amdahl's Law to explain how parallel efficiency decreases as more processors are added, but also mentions potential cost savings from reduced run times.
:p How does adding more processors affect parallel efficiency according to Amdahl’s law?
??x
Amdahl's Law states that the maximum speedup achievable by using multiple processors is limited by the parts of the program that cannot be parallelized. The formula for Amdahl's Law is:
\[ S(p) = \frac{1}{(1 - P + \frac{P}{p})} \]
where \( S(p) \) is the speedup, \( p \) is the number of processors, and \( P \) is the fraction of the program that can be parallelized. As more processors are added, the improvement in run time decreases because a larger portion of the code must still be executed sequentially.
x??

---
#### Example Scenario for Energy Savings
Background context: The text provides an example scenario where 100 jobs need to be run on either 20 or 40 processors. It calculates the total runtime and discusses how increasing the number of processors can reduce energy consumption by reducing overall job run time, but also increases cost due to fixed costs.
:p How does running more parallel jobs affect the total run time according to the example given?
??x
Running more parallel jobs generally reduces the total run time because the number of tasks processed concurrently is higher. For instance, in the provided example:
- Running 10 jobs at a time on 20 processors results in a total run time of 100 hours.
- Running 5 jobs at a time on 40 processors with an 80% parallel efficiency results in a reduced run time (6.25 hours), showing the potential for significant energy savings due to shorter overall job completion times.
x??

---

#### Optimizing GPU Usage for Large Job Suites

Background context: The example illustrates that optimizing runtime for a large suite of jobs often involves using less parallelism. This is because, as more processors are added, the efficiency decreases due to overheads and bottlenecks.

:p How does the use of fewer processors impact the optimization of a large job suite?
??x
Using fewer processors can be more efficient when optimizing runtime for a large suite of jobs because the increase in parallelism starts to introduce overhead that reduces overall efficiency. This is especially true if the tasks are not perfectly divisible or have significant setup and teardown times.
x??

---

#### Cost Optimization with Cloud Computing

Background context: The example discusses how cloud computing services can be used to optimize costs by choosing appropriate hardware based on workload demands.

:p How do cloud computing services help in reducing costs for applications that are memory bound?
??x
For memory-bound applications, you can use GPUs with lower flop-to-load ratios at a lower cost. This is because the primary bottleneck is memory access, and such GPUs might offer better performance per dollar compared to high-flop but low-memory-perf devices.
x??

---

#### Speedup and Parallel Efficiency Calculation

Background context: The example provides a detailed calculation of speedup and parallel efficiency using a specific formula.

:p How do you calculate the new time \( T_{new} \) when doubling the number of processors?
??x
To calculate the new time \( T_{new} \), you use the speedup equation:
\[ S = \frac{T_{base}}{T_{new}} \]
Given that the parallel efficiency is 80%, and the processor multiplier (Pmult) is a factor of 2, we can find the speedup \( S \):
\[ S = 0.8 \times Pmult = 0.8 \times 2 = 1.6 \]

Then:
\[ T_{new} = \frac{T_{base}}{S} = \frac{10}{1.6} = 6.25 \text{ hours} \]
x??

---

#### Total Suite Run Time Calculation

Background context: The example calculates the total run time for a suite of jobs with multiple parallel processes.

:p How do you calculate the total suite run time?
??x
To calculate the total suite run time, multiply the new run time per job by the number of jobs in the suite. For instance:
\[ \text{Total Suite Time} = T_{new} \times \left(\frac{\text{Number of Jobs}}{\text{Processes per Job}}\right) \]
Given that there are 100 jobs and each process handles 5 jobs, we have:
\[ \text{Total Suite Time} = 6.25 \text{ hours/job} \times \left(\frac{100}{5}\right) = 125 \text{ hours} \]
x??

---

#### Preemptible Resources in Cloud Computing

Background context: The example mentions the use of preemptible resources, which can significantly reduce costs but come with less serious deadlines.

:p How do preemptible resources affect cost optimization?
??x
Preemptible resources allow you to use cheaper hardware that might be shut down at any time. This is suitable for workloads where deadlines are not strict and downtime can be accommodated without major issues, thus providing a substantial reduction in costs.
x??

---

#### Cloud Computing Hardware Flexibility

Background context: The example highlights the flexibility of cloud computing in terms of accessing various hardware types.

:p How does cloud computing provide more options compared to on-site resources?
??x
Cloud computing provides access to a wider variety of hardware than is typically available on-site. This allows for better matching of hardware to specific workloads, optimizing performance and cost efficiency.
x??

---

#### Lack of Parallelism in GPUs
Background context: GPUs are highly parallel processors designed for graphics workloads. The effectiveness of GPU computing depends on having a high degree of parallelism, as not all computational tasks can benefit from this architecture.

:p What does "With great power comes great need for parallelism" imply about the use of GPUs?
??x
This phrase highlights that for GPUs to be effective, the computation workload must have a significant amount of parallelizable operations. If the task at hand lacks inherent parallelism, such as sequential or highly dependent tasks, GPUs may not provide substantial performance benefits.

Example: A simple loop processing each element in an array sequentially would not benefit much from GPU acceleration.
```java
for (int i = 0; i < n; i++) {
    result[i] = processElement(array[i]);
}
```
x??

---

#### Irregular Memory Access in GPUs
Background context: CPUs and GPUs both struggle with irregular memory access patterns. This means that accessing memory in a non-contiguous or unpredictable way can be inefficient for both types of processors.

:p What does the second law of GPGPU programming state, and why is it relevant?
??x
The second law of GPGPU programming states that "CPUs also struggle with this." This highlights that GPUs are not immune to inefficiencies in memory access. Since many algorithms require non-contiguous or unpredictable memory access, GPUs cannot always provide the expected performance benefits.

Example: Accessing an array where each element depends on its index and the value of another random element.
```java
for (int i = 0; i < n; i++) {
    result[i] += array[i] * array[randomIndex];
}
```
x??

---

#### Thread Divergence in GPUs
Background context: GPUs use SIMD (Single Instruction, Multiple Data) and SIMT (Single Instruction, Multiple Threads) architectures. While small amounts of branching can be handled, large variations in branch paths can lead to inefficiencies.

:p What is thread divergence, and how does it impact GPU performance?
??x
Thread divergence occurs when threads on a GPU take different execution paths within the same block. This can happen due to conditional statements (branches) that result in some threads taking one path while others take another. Large amounts of divergence can significantly reduce GPU efficiency.

Example: A branch statement where each thread has a unique condition.
```java
if (condition[i]) {
    // path 1
} else {
    // path 2
}
```
x??

---

#### Dynamic Memory Requirements in GPUs
Background context: GPUs allocate memory on the CPU, which can limit algorithms that require dynamic memory management. This is because memory allocation must be done outside of the GPU's processing environment.

:p How do dynamic memory requirements impact GPU performance?
??x
Dynamic memory requirements pose a challenge for GPUs because they typically allocate and manage memory through the CPU. If an algorithm requires memory sizes or patterns that are determined dynamically during execution, this can lead to significant overhead, reducing overall efficiency.

Example: An algorithm that allocates memory based on runtime conditions.
```java
int size = computeSize();
malloc(size); // Hypothetical GPU memory allocation function
```
x??

---

#### Recursive Algorithms in GPUs
Background context: Recursion is limited on GPUs due to stack space constraints. However, some algorithms can still be implemented using iterative techniques.

:p How do GPUs handle recursive algorithms?
??x
GPUs have limited stack resources and often do not support recursion directly. While deep recursion can lead to stack overflow errors, some algorithms that require recursion can be adapted into iterative solutions or optimized for use with the available stack space.

Example: Converting a recursive function to an iterative one.
```java
// Recursive version
void recurse(int n) {
    if (n == 0) return;
    // do something
    recurse(n - 1);
}

// Iterative version
void iterate(int n) {
    while (n > 0) {
        // do something
        n--;
    }
}
```
x??

---

#### GPU Architecture Evolution and Innovations
Background context: GPUs have evolved beyond their original purpose of graphics processing to support a wide range of applications, including machine learning and general computation. Continuous innovation is necessary to keep up with the changing demands of these fields.

:p Why are continuous developments in GPU architecture important?
??x
Continuous developments in GPU architecture are crucial because they enable better performance for diverse workloads. As new applications arise and existing ones evolve, so must the hardware and software capabilities of GPUs to support them effectively.

Example: The integration of machine learning frameworks into modern GPU designs.
```java
// Example pseudo-code using a hypothetical ML library
Model model = loadModel("path/to/model");
output = model.predict(inputData);
```
x??

---

#### STREAM Benchmark for Memory Bandwidth Testing
Background context: The STREAM benchmark is used to test the achievable memory bandwidth of many-core processors. It measures how efficiently different parallel programming models can utilize memory.

:p What does the STREAM benchmark measure, and why is it important?
??x
The STREAM benchmark measures the maximum sustainable memory bandwidth that a system can achieve by copying data between different types of memory operations (read-only, write-only, read-write). This is important for evaluating the efficiency of various parallel programming models in utilizing memory resources.

Example: Running the STREAM benchmark on a GPU and CPU.
```java
// Pseudo-code for running the STREAM benchmark
double[] array = new double[size];
long startTime, endTime;

startTime = System.nanoTime();
for (int i = 0; i < iterations; i++) {
    // Perform read-only operations
}
endTime = System.nanoTime();
long timeTaken = endTime - startTime;
double bandwidth = size * 8.0 / (timeTaken / 1e9); // 8 bytes per double

// Repeat for other memory operations
```
x??

---

#### Roofline Model for GPU Performance Analysis
Background context: The Roofline model is a visual tool that helps analyze and predict the performance of algorithms on different architectures by showing the theoretical limits of arithmetic intensity.

:p What is the Roofline model, and how can it be used to understand GPU performance?
??x
The Roofline model provides a visual representation of the relationship between computational intensity (FLOPs per byte) and execution time. It helps in understanding where an algorithm falls on the performance spectrum relative to theoretical limits, aiding in optimization strategies for different architectures.

Example: Plotting a simple algorithm on the Roofline model.
```java
// Pseudo-code for plotting FLOPS/Byte vs Performance
double flopsPerByte = computeFlopsPerByte();
double timeTaken = measureExecutionTime();

plotPoint(flopsPerByte, 1.0 / (timeTaken * 1e9)); // Time in seconds
```
x??

