# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 48)

**Starting Chapter:** 31.2 Non-Preemptive Priority

---

#### Non-Preemptive Priority M/G/1 Queue Overview
This section discusses the performance of a non-preemptive priority queueing system where jobs are classified into different priorities. The server's utilization (\(\rho\)) is required to be less than 1, and the average time in the system for each job class \(k\) (E[T(k)]) is derived.
:p What does this section focus on?
??x
This section focuses on analyzing the performance of a non-preemptive priority queueing system where jobs are classified into different priorities. It derives formulas for the average time in the system (\(E[T(k)]\)) for each job class \(k\) and compares these with an FCFS (First-Come, First-Served) system.
x??

#### Derivation of Time in Queue for Priority 1 Jobs
The formula to derive the time in queue (\(E[TQ(1)]\)) for jobs of priority 1 is given. It involves considering both the job currently being served and all jobs already in the queue of higher or equal priorities.
:p What is \(E[TQ(1)]\)?
??x
\(E[TQ(1)]\) is derived by considering two components: (i) the time spent waiting for the current job if one is present, and (ii) the expected service times of all jobs already in the queue of higher or equal priorities.
The formula given is:
\[ E[TQ(1)] = \rho \cdot E[Se] + E[NQ(1)] \cdot E[S1] \]
where \(E[Se]\) is the average service time, and \(E[NQ(1)]\) is the expected number of jobs in queue.
x??

#### Derivation of Time in Queue for Priority 2 Jobs
The formula to derive the time in queue (\(E[TQ(2)]\)) for jobs of priority 2 involves additional terms compared to priority 1, including waiting for all jobs of priorities 1 and 2 already in the queue and those that arrive while the new job is waiting.
:p What does \(E[TQ(2)]\) account for?
??x
\(E[TQ(2)]\) accounts for the time spent by a priority 2 job queuing, which includes:
(i) Waiting for the current job if one is being served,
(ii) All jobs of priorities 1 and 2 already in queue,
(iii) All jobs of priorities 1 that arrive while this new job is waiting (not in service).
The formula given is:
\[ E[TQ(2)] = \rho \cdot E[Se] + E[NQ(1)] \cdot E[S1] + E[NQ(2)] \cdot E[S2] + E[TQ(2)] \cdot \lambda_1 E[S1] \]
This can be simplified and iteratively solved to find the expression for \(E[TQ(2)]\).
x??

#### General Derivation of Time in Queue for Priority k Jobs
The general formula for time in queue (\(E[TQ(k)]\)) for jobs of priority \(k\) is derived using induction. It includes contributions from waiting behind jobs of higher or equal priorities and those arriving after the tagged job.
:p What is the general formula for \(E[TQ(k)]\)?
??x
The general formula for \(E[TQ(k)]\) in a non-preemptive M/G/1 priority queue is:
\[ E[TQ(k)] = \frac{\rho E[Se]}{(1 - \sum_{i=1}^{k} \rho_i)(1 - \sum_{i=1}^{k-1} \rho_i)} \]
where \(E[Se]\) represents the average service time, and \(\rho_i\) is the utilization due to jobs of priority \(i\).
This formula accounts for both waiting behind higher or equal priority jobs and those arriving after.
x??

#### Comparison Between NP-Priority and FCFS Queues
The formulas for \(E[TQ(k)]\) in a non-preemptive priority queue (NP-Priority) and a First-Come, First-Served (FCFS) queue are compared. The key difference lies in the denominator, which accounts for different waiting components.
:p How does the NP-Priority formula differ from the FCFS formula?
??x
The main difference is that in the non-preemptive priority queue:
\[ E[TQ(k)] = \frac{\rho E[Se]}{(1 - \sum_{i=1}^{k} \rho_i)(1 - \sum_{i=1}^{k-1} \rho_i)} \]
The squared denominator accounts for the additional waiting due to jobs of higher or equal priorities and those arriving after. This differs from FCFS, where:
\[ E[TQ] = \frac{\rho E[Se]}{1 - \rho} \]
Here, the single term in the denominator only accounts for waiting behind jobs already in queue.
x??

#### Analysis of High-Priority Job Performance
For high-priority jobs (low \(k\)), the non-preemptive priority system (\(E[TQ(k)]_{NP-Priority}\)) is compared to FCFS. The squared denominator in NP-Priority leads to lower expected waiting times for high-priority jobs.
:p How does NP-Priority handle high-priority jobs better than FCFS?
??x
For low \(k\) (high priority), the term \(\sum_{i=1}^{k} \rho_i\) is much smaller than \(\rho\). Thus, the squared denominator in \(E[TQ(k)]_{NP-Priority}\) leads to a lower expected waiting time compared to FCFS. This advantage holds even if some jobs of higher priority classes arrive after the tagged job.
\[ E[TQ(k)]_{NP-Priority} < E[TQ]_{FCFS} \]
This is because the squared term in the denominator for NP-Priority accounts for both current and future arrivals more effectively.
x??

---

#### Non-Preemptive Priority (NP-Priority) Scheduling

Background context: In NP-Priority scheduling, jobs are assigned a priority based on their size, and this policy is non-preemptive. The formula provided shows how to calculate the expected waiting time \(E[TQ]_{NP-Priority}\).

Given that:
\[ E[TQ]_{NP-Priority} = \sum_{k=1}^{n} p_k \cdot E[TQ(k)] \]

Where \(p_k\) is the fraction of jobs in class k, and \(E[TQ(k)]\) is the expected waiting time for a job of class k.

The formula for \(E[TQ(k)]_{NP-Priority}\) involves the squared term and the summation over different classes:
\[ E[TQ]_{NP-Priority} = \frac{n}{\sum_{k=1}^{n} p_k \cdot \frac{E[S^2]}{2E[S]} \cdot \frac{1}{1 - \sum_{i=1}^{k-1} \rho_i}}{\frac{1}{1 - \sum_{i=1}^{k} \rho_i}} \]

:p How do you calculate the expected waiting time for a job in NP-Priority scheduling?
??x
The expected waiting time \(E[TQ]_{NP-Priority}\) is calculated by summing up the weighted contributions from each priority class. For each class k, the contribution to the total waiting time involves the squared size of jobs and the probabilities associated with different classes.

```java
public class NP.Priority {
    public double expectedWaitingTime(List<Double> sizes, List<Double> arrivalRates) {
        int n = sizes.size();
        double ETSquared = // calculate E[S^2] based on job sizes;
        double ES = // calculate E[S] based on job sizes;
        double totalSum = 0;
        for (int k = 1; k <= n; k++) {
            double pk = arrivalRates.get(k-1); // fraction of jobs in class k
            double sumRho = 0;
            for (int i = 1; i < k; i++) {
                sumRho += rho[i];
            }
            totalSum += pk * ETSquared / (2 * ES) * (1.0 / (1 - sumRho)) / (1 - sumRho);
        }
        return totalSum;
    }
}
```
x??

---

#### Shortest-Job-First (SJF)

Background context: SJF is a non-preemptive scheduling policy where the server processes jobs in order of their size. This means smaller jobs have higher priority.

:p If your goal is to minimize mean response time, which type of job should have higher priority?
??x
The small ones. This is because shorter jobs generally complete faster and can be dispatched more quickly, reducing the average waiting time for all jobs.

```java
public class SJF {
    public void processJobs(List<Double> sizes) {
        // Logic to sort jobs based on size and then process them
        Collections.sort(sizes);
        int processed = 0;
        double totalWaitTime = 0;
        for (double size : sizes) {
            totalWaitTime += processed * size; // Add wait time of each job
            processed++;
        }
    }
}
```
x??

---

#### SJF Performance Analysis

Background context: To analyze the performance of SJF, we can model it as an infinite number of priority classes where smaller jobs have higher priorities. The expected waiting time for a job \(E[TQ(x)]_{SJF}\) is derived from the NP-Priority formulas.

:p How can you derive the expected waiting time for a job in SJF?
??x
The expected waiting time for a job of size x in SJF is given by:
\[ E[TQ(x)]_{SJF} = \frac{\rho E[S^2]}{2E[S]} \cdot \left(\frac{1}{1 - \int_0^{x} tf(t)dt}\right)^2 \]

And the total expected waiting time for all jobs is:
\[ E[TQ]_{SJF} = \int_0^{n_x} f(x) dx \cdot \frac{\rho E[S^2]}{2E[S]} \cdot \left(\frac{1}{1 - \int_0^{x} tf(t)dt}\right)^2 \]

Where:
- \(\rho\) is the load composed of jobs of size 0 to x.
- \(f(x)\) is the job size distribution.

```java
public class SJFAnalysis {
    public double expectedWaitingTime(double size, List<Double> sizes, List<Double> arrivalRates) {
        double rho = lambda * integral(0, size, t -> t * f(t)) / integral(0, size, t -> f(t));
        double ETSquared = // calculate E[S^2] based on job sizes;
        double ES = // calculate E[S] based on job sizes;
        return (rho * ETSquared) / (2 * ES) * (1.0 / (1 - rho)) * (1.0 / (1 - integral(0, size, t -> t * f(t))));
    }
}
```
x??

---

#### Load and Job Size Distribution

Background context: The load \(\rho_x\) is defined as the fraction of jobs with sizes less than or equal to x. This helps in understanding the distribution of job sizes.

:p Define \(\rho_x\) for a job size x.
??x
\(\rho_x = \frac{\lambda}{\int_0^x t f(t) dt}\)

Where:
- \(\lambda\) is the arrival rate of jobs.
- \(f(x)\) is the probability density function of job sizes.

The term \(\rho_x\) represents the load composed of jobs with sizes up to x, and it's used in comparing SJF with FCFS.

```java
public class LoadDistribution {
    public double rhoX(double size, List<Double> arrivalRates, List<Double> sizes) {
        return arrivalRates.get(0) / integral(0, size, t -> t * getDensity(t));
    }
}
```
x??

---

#### SJF vs. FCFS Scheduling
Background context explaining the comparison between Shortest Job First (SJF) and First Come First Serve (FCFS) scheduling policies, particularly in non-preemptive scenarios with size-based policies.

The expected time in queue \(E[TQ(x)]\) for both policies can be expressed as:
- For SJF: \(E[TQ(x)]SJF = \rho E[S^2] / 2E[S] \cdot (1 - \rho_x)^2\)
- For FCFS: \(E[TQ(x)]FCFS = \rho E[S^2] / 2E[S] \cdot (1 - \rho)\)

Where \(\rho\) is the load factor, and \(\rho_x\) is typically much less than \(\rho\).

:p How does SJF compare to FCFS in terms of expected time in queue for small jobs?
??x
SJF generally performs better for small jobs because it considers the size of the job. When there are many short jobs, SJF can complete them quickly by always starting with the smallest remaining job. In contrast, FCFS might have a larger \(E[S^2]\) term due to occasional large jobs that can delay smaller ones.

For small jobs (small x), \(E[TQ(x)]SJF\) should be lower than \(E[TQ(x)]FCFS\). However, for very large jobs, SJF's performance may degrade because of the squared factor in the denominator. If the service time distribution is heavy-tailed, SJF might only perform worse on extremely large jobs.

```java
public class JobQueue {
    double rhoX; // Load factor for small jobs
    double rho;  // Overall load factor
    
    public double expectedTimeInQueueSJF() {
        return rho * Math.pow(expectedServiceTime(), 2) / (2 * expectedServiceTime()) * Math.pow(1 - rhoX, 2);
    }
    
    public double expectedTimeInQueueFCFS() {
        return rho * Math.pow(expectedServiceTime(), 2) / (2 * expectedServiceTime()) * (1 - rho);
    }
}
```
x??

---

#### SJF Performance with Heavy-Tailed Distributions
Background context explaining the impact of heavy-tailed job size distributions on the performance of Shortest Job First (SJF).

In a system where job sizes follow a heavy-tailed distribution, the variance \(E[S^2]\) is significantly large. This means that even small jobs can get stuck behind much larger ones, leading to longer waiting times.

:p How does heavy-tailed job size distribution affect SJF's performance for small jobs?
??x
Even with heavy tails, small jobs are still likely to be impacted because they may get delayed by large jobs. The variance in job sizes means that the system can experience significant delays even when processing smaller tasks.

The presence of a few very large jobs can cause significant backlogs and increase the overall waiting time for all jobs, including small ones. This is why SJF might not be optimal in such environments.
x??

---

#### Preemption and Checkpointing
Background context explaining the importance of preemption in scheduling policies, especially when it's not naturally available.

Preemption allows a scheduler to interrupt running tasks to prioritize smaller or more urgent jobs. When preemption isn't available, checkpointing can be used as an alternative method where a job is saved at regular intervals and restarted from the last saved state if interrupted.

:p Why might preemptive policies be better than SJF in systems with heavy-tailed distributions?
??x
Preemptive policies are more effective because they allow the scheduler to interrupt longer-running jobs to make way for smaller, higher-priority jobs. This is crucial when job sizes follow a heavy-tailed distribution, as it helps ensure that short jobs can still get processed promptly.

Without preemption or checkpointing, SJF might not perform well due to the delay caused by large jobs.
x??

---

#### TAGS Policy
Background context explaining the Time-Activated Guarded Scheduling (TAGS) policy, which involves killing and restarting long-running jobs when a queue of short jobs is waiting.

The idea behind TAGS is that killing a running job can prevent it from blocking shorter jobs, thereby improving overall system responsiveness. This policy is particularly useful in environments where job sizes are highly variable and unpredictable.

:p Why might the TAGS policy be beneficial in systems with high job size variability?
??x
TAGS is beneficial because it ensures that short jobs don't get delayed by long-running ones. By periodically killing and restarting longer jobs, the system can ensure that short jobs get a chance to run more frequently, improving overall response times.

This approach is especially useful when the distribution of job sizes is heavy-tailed, as it helps mitigate the impact of large jobs on the system's performance.
x??

---

#### Priority Scheduling for Small Jobs
Background context explaining the optimal scheduling policy for an M/G/1 system with non-preemptive priority scheduling and two job classes (S: small; L: large).

The goal is to minimize the mean waiting time over all jobs by prioritizing smaller jobs, which have a shorter expected service time.

:p How can we prove that class S jobs should get priority in an M/G/1 system?
??x
To prove that class S jobs should get priority, we compare the expected waiting times for both policies (S has priority and L has priority).

For non-preemptive priority scheduling:
- \(E[TQ]NP-Priority(S has priority)\)
  \[
  = \frac{\lambda_S E[S^2]}{2} + \rho_L \frac{\lambda_L E[L^2]}{2}
  \]
- \(E[TQ]NP-Priority(L has priority)\)
  \[
  = \frac{\lambda_L E[L^2]}{2} + \rho_S \frac{\lambda_S E[S^2]}{2}
  \]

Since \(E[SS] < E[SL]\), it follows that:
- If S has priority: \(\frac{\lambda_S E[S^2]}{2} < \frac{\lambda_L E[L^2]}{2}\)
- If L has priority: \(\frac{\lambda_L E[L^2]}{2} > \frac{\lambda_S E[S^2]}{2}\)

Thus, giving S jobs priority minimizes the mean waiting time over all jobs.
x??

---

