# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 32)

**Starting Chapter:** Chapter 20 Tales of Tails A Case Study of Real-World Workloads. 20.2 UNIX Process Lifetime Measurements

---

#### CPU Load Balancing Overview
Background context: In the mid-1990s, an important area of research was CPU load balancing in Network of Workstations (N.O.W.). The goal is to migrate CPU-bound jobs from heavily loaded workstations to more lightly loaded ones. This process can improve system efficiency but comes with costs.

:p What is the main idea behind CPU load balancing?
??x
The main idea is to balance the computational load across multiple machines in a network by moving CPU-bound jobs between them, aiming for better overall system performance and resource utilization.
x??

---

#### Types of Migration
Background context: There are two types of migration used in load balancing techniques - initial placement (migrating newborn jobs) and active process migration (moving already running processes). The common belief was that migrating active processes is costly.

:p What are the two types of migration discussed?
??x
The two types of migration are:
1. Migration of newborn jobs only, also called initial placement or remote execution.
2. Migration of active jobs - referred to as active process migration.
x??

---

#### Job Size and Age Definitions
Background context: The terms "job size," "age," "lifetime," and "remaining lifetime" have specific meanings in this context. Understanding these definitions is crucial for evaluating the effectiveness of job migrations.

:p What do we mean by a job's size, age, and remaining lifetime?
??x
- Job **size** refers to its total CPU requirement.
- Job **age** means its total CPU usage thus far.
- A job’s **lifetime** (same as size) is its total CPU requirement.
- The **remaining lifetime** of a job is its remaining CPU requirement.

These definitions are important for assessing whether migrating a job would be beneficial.
x??

---

#### Exponential Distribution Assumption
Background context: In the 1990s, it was widely believed that UNIX job lifetimes followed an exponential distribution. This assumption had significant implications for load balancing strategies.

:p What is the implication of jobs having exponentially distributed lifetimes?
??x
The implication is that all jobs have the same expected remaining lifetime and a constant failure rate (i.e., each second of CPU usage has the same probability). Therefore, newborn and older jobs were considered to have equal value in terms of migration benefits.

Since newborn jobs are cheaper to migrate compared to active ones, it was generally accepted that only newborn jobs should be migrated.
x??

---

#### Measuring Job Lifetimes
Background context: The author measured job lifetimes to validate the exponential distribution assumption. This involved collecting data on millions of jobs from different machines over a period.

:p What did you measure and why?
??x
The author measured the CPU lifetimes of thousands of jobs across various machines to determine if their lifetimes followed an exponential distribution as commonly believed.

This was done because the common wisdom was that jobs had exponentially distributed lifetimes, but there was skepticism about this assumption.
x??

---

#### Log-Log Plot Analysis
Background context: The author used a log-log plot to analyze the data and compare it with the expected exponential distribution. This revealed that the actual job lifetime distribution did not follow an exponential pattern.

:p How can you tell the measured distribution is not Exponential?
??x
For an Exponential distribution, the fraction of jobs remaining should decrease by a constant factor for each unit increase in x (constant failure rate). However, in Figure 20.1, the fraction decreases at a slower rate as x increases (decreasing failure rate).

Specifically:
- At CPU age 1 second: Half of the jobs last another second.
- Of those that make it to 2 seconds, half make it to 4 seconds.
- Of those that make it to 4 seconds, half make it to 8 seconds.

This pattern indicates a non-Exponential distribution. To visualize this more clearly, the log-log plot in Figure 20.2 was used, showing the actual data as a bumpy line and the best-fit Exponential curve.
x??

---

#### Non-Exponential Distribution
Background context: The actual job lifetime distribution did not follow an exponential pattern but rather decreased at a slower rate with increasing CPU age.

:p How does the measured distribution look on a log-log plot?
??x
The measured distribution, when plotted on a log-log scale (Figure 20.2), shows a bumpy line that fits less well with the straight line representing the best-fit Exponential distribution (Figure 20.3).

This visual comparison clearly demonstrates that the job lifetime is not exponentially distributed.
x??

---

#### Pareto Distribution Definition
Background context explaining the Pareto distribution. The definition provided is \( F(x) = P\{X > x\} = \frac{1}{x^\alpha}, \, x \geq 1 \), where \(0 < \alpha < 2\). This distribution is used to model phenomena like job lifetimes in real-world workloads.

:p What is the definition of the Pareto distribution?
??x
The Pareto distribution is defined such that for a random variable \(X\), the probability that \(X\) exceeds some value \(x\) is given by \( F(x) = P\{X > x\} = \frac{1}{x^\alpha}, \, x \geq 1 \). The parameter \(\alpha\) controls the shape of the distribution.

---
#### Failure Rate of Pareto Distribution
Explaining how to derive the failure rate function for the Pareto distribution and that it shows a decreasing failure rate (DFR).

:p What is the failure rate \(r(x)\) of the Pareto distribution?
??x
The failure rate \(r(x)\) of the Pareto distribution can be derived as follows:
\[ F(x) = P\{X > x\} = \frac{1}{x^\alpha}, \, x \geq 1 \]
\[ f(x) = \frac{dF(x)}{dx} = \alpha x^{-\alpha-1}, \, x \geq 1 \]
\[ r(x) = \frac{f(x)}{F(x)} = \frac{\alpha x^{-\alpha-1}}{\frac{1}{x^\alpha}} = \alpha x^{-1} = \frac{\alpha}{x} \]

Since \(r(x) = \frac{\alpha}{x}\) decreases with \(x\), the Pareto distribution has a decreasing failure rate (DFR).

---
#### Mean and Variance for α ≤ 1
Explanation of the mean and variance for the Pareto distribution when \(\alpha \leq 1\).

:p What are the mean and variance for a Pareto distribution with \(\alpha \leq 1\)?
??x
For \(\alpha \leq 1\):
- The expected lifetime \(E[Lifetime]\) is infinite.
- Higher moments of the lifetime (for \(i = 2, 3, \ldots\)) are also infinite.

This means that for values of \(\alpha \leq 1\), it is impossible to calculate a finite mean or variance for the distribution.

---
#### Mean and Variance for α > 1
Explanation of the mean and variance when \(\alpha > 1\).

:p What changes in mean and variance when \(\alpha\) is above 1?
??x
For \(\alpha > 1\):
- The expected lifetime \(E[Lifetime]\) becomes finite.
- Higher moments of the lifetime are still infinite, but the first moment (mean) is now finite.

This indicates that for \(\alpha > 1\), we can calculate a finite mean and variance, even though higher-order moments remain infinite.

---
#### Probability of Job Age
Explanation of how to calculate the probability that a job of CPU age \(a\) lives to CPU age \(b\).

:p What is the probability that a job of CPU age \(a\) lives to CPU age \(b\) for a Pareto distribution with \(\alpha = 1\)?
??x
For a Pareto distribution with \(\alpha = 1\), the probability that a job of CPU age \(a\) lives to CPU age \(b\) is given by:
\[ P\{Life > b | Life \geq a, a > 1\} = \frac{a}{b} \]

This means:
- Half of all jobs currently of age 1 sec will live to age ≥2 sec.
- The probability that a job of age 1 sec uses more than \(T\) sec of CPU is \(\frac{1}{T}\).
- The probability that a job of age \(T\) sec lives to be age ≥2\(T\) sec is \(\frac{1}{2}\).

---
#### Bounded Pareto Distribution
Explanation and definition of the bounded Pareto distribution, which models finite moments in measured data.

:p What is the Bounded Pareto distribution?
??x
The Bounded Pareto distribution is a version of the Pareto distribution that is truncated on both ends to account for finite minimum and maximum job lifetimes. Its density function is defined as:
\[ f(x) = \frac{\alpha x^{-\alpha-1} \cdot k^\alpha}{1 - (k/p)^\alpha}, \, \text{for } k \leq x \leq p \]
where \(0 < \alpha < 2\) and \(k \leq x \leq p\) are the lower and upper bounds.

This distribution ensures that all moments of the distribution are finite, making it suitable for modeling real-world data with bounded lifetimes.

#### Decreasing Failure Rate (DFR) Property of Pareto Distribution
Background context: The decreasing failure rate (DFR) property of the Pareto distribution states that as more CPU has been used, the remaining lifetime of a job increases. This implies older jobs have higher expected remaining lifetimes.

:p What does the DFR property of the Pareto distribution tell us about migrating older jobs?
??x
The DFR property suggests that older jobs might still need significant computing resources in the future. Thus, migrating these old jobs can be beneficial as their long remaining lifetime can help distribute the load more evenly across the system.

```java
public class JobMigration {
    public void migrateOldJobs() {
        for (Job job : getOldestJobs()) {
            if (job.isDFRConditionMet()) { // Check based on usage and age
                migrate(job);
            }
        }
    }

    private boolean isDFRConditionMet(Job job) {
        return job.getUsageTime() > threshold && job.ageInSystem >= oldJobAgeThreshold;
    }

    private void migrate(Job job) {
        targetMachine.run(job);
        sourceMachine.removeJob(job);
    }
}
```
x??

---

#### Heavy-Tail Property of Pareto Distribution
Background context: The heavy-tail property indicates that a small fraction of very large jobs contribute significantly to the total system load. Specifically, for a Pareto distribution with α=1.1, the largest 1 percent of jobs contain about 12% of the total load.

:p What does the heavy-tail property tell us?
??x
The heavy-tail property suggests that it might be sufficient to focus on migrating or managing only the top 1 percent of very large jobs, as they contribute most of the system load. However, determining which jobs are "old enough" for migration based on their size and cost is complex due to infinite moments.

```java
public class JobManager {
    public void manageHeavyJobs() {
        List<Job> heavyJobs = getTopPercentOfLargestJobs(1);
        for (Job job : heavyJobs) {
            if (isMigrationWorthy(job)) { // Determine based on size and cost
                migrate(job);
            }
        }
    }

    private boolean isMigrationWorthy(Job job) {
        return job.getSize() > threshold && job.getCostToMigrate() < maxMigrationCost;
    }

    private void migrate(Job job) {
        targetMachine.run(job);
        sourceMachine.removeJob(job);
    }
}
```
x??

---

#### Bounded Pareto Distribution
Background context: While the standard Pareto distribution can have infinite variance, the bounded Pareto distribution has an upper limit on job size. Despite this bound, it still exhibits the heavy-tail property but with some differences in its parameters.

:p How does the bounded Pareto distribution differ from the standard Pareto distribution?
??x
The bounded Pareto distribution differs from the standard one by having a finite maximum value for job size, which affects its variance and heaviness of tails. While it still has decreasing failure rate (DFR) property, the heavy-tail is more constrained compared to an unbounded Pareto.

```java
public class BoundedParetoJobSize {
    public double getProbAboveThreshold(double threshold) {
        return 1 - Math.pow(threshold / maxLimit, alpha);
    }

    public double expectedLoad() {
        if (alpha > 2) return Double.POSITIVE_INFINITY; // Infinite variance
        else return (maxLimit * (alpha - 1)) / alpha;
    }
}
```
x??

---

#### Active Process Migration for CPU Load Balancing
Background context: Active process migration involves moving running processes to better utilize available resources. For Pareto-distributed workloads, migrating the largest jobs can significantly reduce overall system slowdown.

:p How does active process migration benefit from the heavy-tail property of Pareto distributions?
??x
Active process migration benefits by targeting only the 1 percent largest jobs for migration, as they carry most of the load. This approach minimizes the number of migrations required while effectively balancing the workload across machines.

```java
public class LoadBalancer {
    public void balanceLoad() {
        List<Job> largestJobs = getTopPercentOfLargestJobs(1);
        for (Job job : largestJobs) {
            if (shouldMigrate(job)) { // Determine based on cost and size
                migrate(job);
            }
        }
    }

    private boolean shouldMigrate(Job job) {
        return job.getSize() > threshold && job.getCostToMigrate() < maxMigrationCost;
    }

    private void migrate(Job job) {
        targetMachine.run(job);
        sourceMachine.removeJob(job);
    }
}
```
x??

---

#### Pareto Distributions in Real-World Workloads
Background context: Various real-world workloads, such as web file sizes and network flows, often follow a Pareto distribution. This property helps in understanding and optimizing system performance.

:p What are some examples of distributions following the heavy-tail property?
??x
Examples include file sizes at websites (α≈1.1), IP flow packet sizes, wireless session times, phone call durations, human wealth, forest fire damage, and earthquake occurrences. These follow a Pareto distribution with varying α values.

```java
public class ParetoExample {
    public void analyzeWorkload() {
        // Analyze file size distribution at websites
        double alpha = 1.1;
        double[] sizes = analyzeFileSizeDistribution(alpha);

        // Use DFR and other properties to optimize system performance
        for (double size : sizes) {
            if (size > threshold && isMigrationWorthy(size)) {
                migrateJob(size);
            }
        }
    }

    private boolean isMigrationWorthy(double size) {
        return size > threshold && getMigrationCost(size) < maxMigrationCost;
    }

    private void migrateJob(double size) {
        // Migrate job to better balance load
    }
}
```
x??

---

