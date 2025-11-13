# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 32)

**Starting Chapter:** Chapter 20 Tales of Tails A Case Study of Real-World Workloads. 20.2 UNIX Process Lifetime Measurements

---

#### CPU Load Balancing in Networks of Workstations (NOW Project)
Background context explaining the concept. The idea is to balance CPU loads by migrating jobs from heavily loaded workstations to more lightly loaded ones within a network. This technique aims to improve system performance and resource utilization.

:p What does CPU load balancing aim to achieve in a Network of Workstations?
??x
CPU load balancing aims to improve overall system performance and resource utilization by distributing the workload evenly across multiple machines in the network, thus ensuring no single machine is overly burdened.
x??

---

#### Types of Process Migration
Background context explaining the two types of process migration used for load balancing. There are two primary methods: migrating newborn jobs (initial placement or remote execution) and migrating active running processes.

:p What are the two types of process migrations in load balancing techniques?
??x
The two types of process migrations are:
1. Migration of newborn jobs only – also called initial placement or remote execution.
2. Migration of jobs that are already active (running) – also referred to as active process migration.
x??

---

#### Job Size and Age Definitions
Background context explaining the definitions related to job size, age, lifetime, and remaining lifetime. These terms describe different aspects of a job's characteristics.

:p What does the term "job’s size" refer to in this context?
??x
The term "job’s size" refers to the total CPU requirement of the job.
x??

---

#### Exponential Distribution in Job Lifetimes
Background context explaining why there was a belief that UNIX job lifetimes were exponentially distributed, and its implications. The common wisdom suggested that all jobs had the same remaining lifetime regardless of their current age.

:p What is the implication of UNIX job lifetimes being exponentially distributed?
??x
The implication of UNIX job lifetimes being exponentially distributed is that they exhibit a constant failure rate. This means all jobs have the same remaining lifetime and the same probability of requiring another second of CPU, irrespective of their current age. Since newborn jobs and older (active) jobs have the same expected remaining lifetime, it made sense to migrate only newborn jobs due to their lower migration costs.
x??

---

#### Measuring Job Lifetimes
Background context explaining the method used to measure job lifetimes. The author collected data on millions of jobs across various machines over an extended period.

:p How did the author collect and plot the distribution of job lifetimes?
??x
The author collected the CPU lifetimes of millions of jobs from a wide range of different machines, including instructional, research, and administrative machines, over many months. The distribution was measured using a log-log plot to better visualize the decreasing rate at which jobs remain active as their age increases.

To measure the distribution, they plotted the fraction of jobs whose size exceeds x for all jobs whose size is greater than 1 second. They then created a log-log plot of this data, where the bumpy line represented the measured distribution and the straight line was the best-fit curve.

The author used Figure 20.1 to show the raw data on a standard scale and Figure 20.2 to visualize it more easily using a log-log plot.
x??

---

#### Identifying Non-Exponential Distribution
Background context explaining how the exponential distribution was tested against the actual measured data, showing that it did not fit an Exponential distribution.

:p How can you tell that job lifetimes are not exponentially distributed?
??x
For an Exponential distribution, the fraction of jobs remaining should drop by a constant factor with each unit increase in x (constant failure rate). However, in Figure 20.1, the fraction of jobs remaining decreases by a slower and slower rate as we increase x, indicating a decreasing failure rate.

To see this more clearly, consider that if the distribution were exponential:
- Half of the jobs make it to 2 seconds.
- Half of those that made it to 2 seconds would then make it to 4 seconds.
- Half of those that made it to 4 seconds would then make it to 8 seconds.

However, in reality, this pattern is not observed. Instead, the fraction of remaining jobs decreases more gradually as time increases, suggesting a non-exponential distribution.

To confirm this, the author created a log-log plot (Figure 20.2) and compared it with the best-fit Exponential distribution (Figure 20.3), showing that the measured data did not follow an exponential curve.
x??

---

#### Pareto Distribution Properties

Background context explaining the concept. The provided text discusses a specific type of distribution, known as the Pareto distribution, which is observed in various real-world workloads such as job lifetimes on UNIX systems. The distribution is characterized by a power-law behavior and has certain properties that make it interesting for modeling.

The probability that a job's lifetime exceeds $x$ seconds given that its lifetime exceeds 1 second can be expressed as:
$$P\{Job size > x | Job size > 1\} = \frac{1}{x}.$$

This implies the following distribution function:
$$

F(x) = 1 - \frac{1}{x^{\alpha}}, \quad x \geq 1,$$where $\alpha$ ranges from approximately 0.8 to 1.2 across different machines.

The failure rate (or hazard function) of the Pareto distribution is given by:
$$r(x) = \frac{f(x)}{F(x)} = \frac{\alpha x^{-\alpha-1}}{(1 - x^{-\alpha})} = \frac{\alpha}{x}, \quad x \geq 1.$$

Notice that the failure rate decreases with $x$, making it a decreasing failure rate (DFR) distribution.

:p What is the failure rate of the Pareto distribution?
??x
The failure rate of the Pareto distribution is given by:
$$r(x) = \frac{\alpha}{x}.$$

This indicates that older jobs have a higher probability of surviving another second, as their failure rate decreases with time. This can be visualized in C/Java code as follows:

```java
public class FailureRate {
    private double alpha;

    public FailureRate(double alpha) {
        this.alpha = alpha;
    }

    public double getFailureRate(double x) {
        return alpha / x;
    }
}
```

x??

---

#### Mean and Variance for Pareto Distribution with $\alpha \leq 1 $ Background context explaining the concept. The provided text discusses how to calculate the mean and variance of a Pareto distribution when$\alpha \leq 1$. For such values, these moments are infinite, which has implications for modeling job lifetimes.

For a Pareto distribution with $0 < \alpha \leq 1$:
$$E[Lifetime] = \infty.$$

The second and higher moments of the lifetime are also infinite:
$$

E[ith moment of Lifetime] = \infty, \quad i=2,3,\ldots.$$:p For a Pareto distribution with $\alpha \leq 1$, what is the mean and variance?
??x
For $0 < \alpha \leq 1$:
- The expected lifetime (mean) is infinite:
  $$E[Lifetime] = \infty.$$- Higher moments of the lifetime are also infinite, implying that both the second moment (variance) and higher-order moments do not exist in finite form.

x??

---

#### Mean and Variance for Pareto Distribution with $\alpha > 1 $ Background context explaining the concept. The text states that when$\alpha > 1$, both the expected lifetime and the expected remaining lifetime are finite, but higher moments of the lifetime remain infinite.

For a Pareto distribution with $\alpha > 1$:
- The expected lifetime (mean) is finite:
  $$E[Lifetime] = \int_1^\infty x f(x) dx < \infty.$$- The expected remaining lifetime given an age of $ a$ is also finite, but higher moments are still infinite.

:p For a Pareto distribution with $\alpha > 1$, what changes in the mean and variance?
??x
For $\alpha > 1$:
- Both the expected lifetime (mean) and the expected remaining lifetime are finite:
  $$E[Lifetime] = \int_1^\infty x f(x) dx < \infty.$$- Higher moments of the lifetime, such as the second moment (variance), remain infinite.

x??

---

#### Probability of a Job Living Beyond Age $b $ Given it Has Survived to Age$a $ Background context explaining the concept. The provided text explains how to calculate the probability that a job with CPU age$ a $ will survive to a CPU age $ b $, where $ b > a$.

For a Pareto distribution with $\alpha = 1$:
$$P\{Life > b | Life \geq a, a > 1\} = \frac{a}{b}.$$

This means that if we consider all the jobs currently of age 1 second, half of them will live to an age of at least 2 seconds. Similarly:
- The probability that a job of age 1 second uses more than $T$ seconds of CPU is given by:
$$P\{Life > T | Life \geq 1\} = \frac{1}{T}.$$- The probability that a job of age $ T $ seconds lives to be at least $2T$ seconds old is:
$$P\{Life \geq 2T | Life \geq T, T > 0\} = \frac{T}{2T} = \frac{1}{2}.$$:p Under the Pareto distribution with $\alpha = 1 $, what is the probability that a job of CPU age $ a $ lives to CPU age $ b$?
??x
For a Pareto distribution with $\alpha = 1$:
$$P\{Life > b | Life \geq a, a > 1\} = \frac{a}{b}.$$

This means that the probability of a job surviving from age $a $ to age$b $ is directly proportional to the ratio of the initial age$ a $ to the final age $b$.

x??

---

#### Bounded Pareto Distribution

Background context explaining the concept. The provided text discusses how real-world data often exhibit finite minimum and maximum values, which cannot be accurately modeled by an unbounded Pareto distribution. To address this, a bounded Pareto distribution is introduced.

A Bounded Pareto distribution has the density function:
$$f(x) = \frac{\alpha k^\alpha x^{-\alpha-1}}{1 - (k/p)^{\alpha}}, \quad for \; k \leq x \leq p,$$where $0 < \alpha < 2$.

The factor $\frac{k^\alpha}{1 - (k/p)^\alpha}$ is a normalization constant ensuring that the integral of the density function between $ k $ and $p$ equals 1.

:p What is the Bounded Pareto distribution?
??x
A Bounded Pareto distribution has the following properties:
- It models real-world data with finite minimum and maximum values.
- The density function is defined as:
$$f(x) = \frac{\alpha k^\alpha x^{-\alpha-1}}{1 - (k/p)^{\alpha}}, \quad for \; k \leq x \leq p,$$where $0 < \alpha < 2$.
- The normalization factor ensures that the integral of the density function between $k $ and$p$ equals 1.

x??

---

#### Decreasing Failure Rate (DFR) Property of Pareto Distribution
Background context: The decreasing failure rate (DFR) property means that as more CPU has been used so far, the job is expected to use even more CPU. This characteristic implies a certain level of continuity in job usage over time.

:p What does the DFR property imply about older jobs?
??x
The DFR property implies that older jobs have higher expected remaining lifetimes. Consequently, it might be beneficial to migrate older jobs because they are likely to continue using significant amounts of CPU resources for longer periods.
x??

---

#### Heavy-Tail Property in Pareto Distribution
Background context: The heavy-tail property indicates that a small fraction of the largest jobs can comprise a large portion (often more than 50%) of the total system load. This means that focusing on a few big jobs can significantly reduce overall load.

:p What does the heavy-tail property suggest about job migration?
??x
The heavy-tail property suggests that it might be sufficient to migrate only the largest 1% of jobs, as they often account for about half of the total system load. However, determining which specific jobs are "old enough" to migrate can be challenging due to the infinite moments in some Pareto distributions.
x??

---

#### Bounded Pareto Distribution
Background context: While the standard Pareto distribution does not have an upper bound on job sizes, the bounded Pareto distribution introduces such a constraint. This means that while it still exhibits heavy-tailed properties, there is a limit to how large jobs can get.

:p How do bounded and unbounded Pareto distributions differ?
??x
Bounded and unbounded Pareto distributions differ in their upper limits for job sizes. The standard (unbounded) Pareto distribution allows for arbitrarily large jobs, which can lead to infinite moments. In contrast, the bounded Pareto distribution restricts job sizes, making its moments finite but still exhibiting heavy-tailed characteristics.
x??

---

#### Active Process Migration Benefits
Background context: Active process migration involves moving running processes from one machine to another to balance CPU load. The DFR property suggests that older jobs have higher expected remaining lifetimes and thus might be worth migrating.

:p Why does the DFR property suggest migrating older jobs?
??x
The DFR property indicates that older jobs are likely to continue using significant amounts of CPU resources for longer periods. Migrating these old jobs can spread their load over time, potentially reducing overall system slowdowns.
x??

---

#### Pareto Distribution in Real-World Applications
Background context: The Pareto distribution is observed in various real-world scenarios including file sizes at websites, internet topologies, IP flow packet counts, and more. It often describes phenomena where a few large items dominate the total load.

:p Provide an example of how Pareto distribution applies to web traffic.
??x
In web traffic, the Pareto distribution can be applied by observing that a small fraction (1%) of IP flows contain most of the data transmitted. By rerouting only 1% of these large flows, significant load balancing can be achieved.
x??

---

#### SYNC Project for Web Servers
Background context: The SYNC project aimed to improve web server performance by favoring requests for smaller files over larger ones. This approach leverages the heavy-tail property of file sizes.

:p How does the SYNC project use the Pareto distribution?
??x
The SYNC project uses the heavy-tail property of web file sizes, which means that although short requests are favored, they collectively make up a small portion of the total load (less than half). By favoring shorter requests, long requests are not significantly impacted.
x??

---

#### TCP Flow Scheduling Using DFR Property
Background context: Ernst Biersack et al. extended the SYNC project's findings to TCP flow scheduling by exploiting the DFR property of Pareto distributions.

:p How does the DFR property apply to TCP flow scheduling?
??x
The DFR property allows for identifying flows with shorter remaining durations, which can be prioritized without significantly impacting longer flows. This approach helps in balancing load across connections.
x??

---

#### Central Limit Theorem and Pareto Distribution
Background context: There is ongoing research into proving a similar theorem to the Central Limit Theorem but specifically for the Pareto distribution, explaining its ubiquity.

:p Why is there interest in understanding the origin of the Pareto distribution?
??x
There is significant interest in understanding why the Pareto distribution appears so frequently in nature and human-created systems. Proving a similar theorem (like the Central Limit Theorem) for the Pareto distribution would help explain its prevalence.
x??

---

