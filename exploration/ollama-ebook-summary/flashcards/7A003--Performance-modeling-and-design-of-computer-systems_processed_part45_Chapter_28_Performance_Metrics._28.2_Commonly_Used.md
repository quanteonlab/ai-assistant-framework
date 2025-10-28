# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 45)

**Starting Chapter:** Chapter 28 Performance Metrics. 28.2 Commonly Used Metrics for Single Queues

---

---
#### Mean Response Time and Mean Waiting Time
Background context: The traditional performance metrics used to evaluate scheduling policies include mean response time (E[T]) and mean waiting time or "wasted" time, also known as mean delay or mean queuing time (E[TQ]). E[T] is defined as the average time a job spends in the system, while E[TQ] is the average time a job spends waiting in the queue before service starts.
:p What does E[T] represent?
??x
Mean response time represents the total time a job spends from arrival to completion. It includes both the time spent waiting in the queue and the actual service time.

E[T] = E[S] + E[TQ]

where E[S] is the mean service time, and E[TQ] is the mean waiting time.
x??
---

---
#### Mean Number in System and Queue
Background context: The mean number of jobs in the system (E[N]) and the mean number of jobs in the queue (E[NQ]) are additional metrics used to evaluate scheduling policies. These metrics help understand the load on the system and the efficiency of resource utilization.
:p What is E[N]?
??x
Mean number in system, denoted as E[N], represents the average number of jobs present in the system at any given time. This includes both the number of jobs being served and those waiting in the queue.

E[N] = λ * E[W]

where λ is the arrival rate, and E[W] is the mean waiting time.
x??
---

---
#### Evaluating Scheduling Algorithm Benefits
Background context: The chapter discusses evaluating scheduling algorithms based on traditional performance metrics. Specifically, it examines how improvements in mean waiting time (E[TQ]) translate to benefits in overall response time (E[T]).
:p If an algorithm improves E[TQ] by a factor of 100, does this necessarily improve E[T] by the same factor?
??x
No, the improvement in E[T] is not necessarily comparable. The relationship between E[T], E[S], and E[TQ] needs to be considered. If E[S] > E[TQ], then even a significant reduction in E[TQ] might yield only a minor reduction in E[T].

For example:
- Suppose E[S] = 10 units, and E[TQ] is improved by a factor of 100.
- The initial value of E[T] could be 20 (E[S] + E[TQ]).
- After the improvement: if E[TQ] reduces to 0.1, then E[T] = 10.1, which is only a 50% reduction compared to the original E[T].

To better understand this, consider the formula:
E[T] = E[S] + E[TQ]

If E[S] > E[TQ], reducing E[TQ] by a large factor may not significantly affect E[T].
x??
---

#### Work in System and Device Utilization
Background context: When analyzing a single queue, two important metrics are work in system (remaining tasks to complete) and device utilization (fraction of time that the device is busy). These metrics can help understand how different scheduling policies affect system performance.

:p If two scheduling policies result in the same work in system and server utilization over all time, do they necessarily have the same mean response time? Explain with an example.
??x
No. The answer here involves understanding that while "work in system" and "device utilization" are consistent across different policies, these metrics alone do not dictate the mean response time.

Consider two work-conserving policies A and B:
- Policy A serves the shortest available job first, resulting in a few large jobs being left.
- Policy B serves the longest available job first, leaving many more small jobs behind.

By Little's Law, which states \( E[N] = \lambda E[T] \) where \( N \) is the number of items in system and \( T \) is the average time spent per item, we can see that:

- For policy B, since there are many more small jobs, \( E[N] \) would be higher.
- Therefore, by Little's Law, \( E[T] \) for B would also be higher.

The key point is:
```java
// Example logic in pseudocode to illustrate the concept
public class SchedulingPolicies {
    public double calculateMeanResponseTime(double lambda, double arrivalRate, double workInSystem) {
        // Using Little's Law: E[N] = λ * E[T]
        // Where E[N] is higher for policy B due to more small jobs.
        return workInSystem / (arrivalRate - lambda);
    }
}
```
x??

---

#### Slowdown of a Job
Background context: The slowdown of a job is defined as its response time divided by the size of the job. This metric helps in understanding how well jobs are being handled relative to their sizes.

:p Why is mean slowdown preferable to mean response time?
??x
Mean slowdown is preferable because it directly correlates with the desired outcome—smaller jobs should have smaller response times, and larger jobs should have proportionally larger response times. A low mean slowdown ensures that no job has a significantly higher ratio of response time to its size.

The objective here is to maintain fair treatment of different-sized jobs:
```java
// Example logic in pseudocode to calculate mean slowdown
public class JobSlowdown {
    public double calculateMeanSlowdown(double totalResponseTime, int totalJobSize) {
        // Mean slowdown is the average ratio of response time to job size.
        return (totalResponseTime / totalJobSize);
    }
}
```
x??

---

#### Tail Behavior of Response Time
Background context: Understanding the tail behavior of response times is crucial for setting Service Level Agreements (SLAs), as it helps in ensuring that extreme delays are kept under control. The probability that a response time exceeds some level \( x \) is denoted as \( P{T > x} \).

:p Why does knowing mean slowdown being low tell us anything about the maximum slowdown?
??x
Knowing the mean slowdown gives insight into how many jobs can have significantly higher than average slowdowns (and thus, high response times). If the mean slowdown is 2:
- Fewer than half of the jobs can have a slowdown greater than or equal to 3.
- Fewer than one-fourth of the jobs can have a slowdown at least 5.

This means that by keeping the mean slowdown low, we limit the number of jobs with very high response times relative to their size:
```java
// Example logic in pseudocode to estimate job counts based on mean slowdown
public class SlowdownAnalysis {
    public double calculateFractionHighSlowdown(double meanSlowdown) {
        // Fewer than 1/(n-1) fraction of jobs can have a slowdown of at least n.
        return 1 / (meanSlowdown - 1);
    }
}
```
x??

---

#### Starvation/Fairness Metrics
Background context: As mean slowdown becomes popular, researchers are concerned about potential policies that might achieve low mean slowdown by starving large jobs. The Shortest-Remaining-Processing-Time (SRPT) policy is an example where small jobs get treated well at the expense of delaying larger ones.

:p What performance metric can tell us if jobs are being starved?
??x
To determine if jobs are being starved, one should look at mean slowdown as a function of job size. For instance:
- "What is the expected slowdown of jobs of size \( x \)? "
- "What is the expected slowdown for the maximum job size? "
- "What is the expected slowdown for jobs in the 99th percentile of the job size distribution?"

A scheduling policy P can be deemed "unfair" if:
```java
// Example logic in pseudocode to compare expected slowdowns
public class FairnessAnalysis {
    public boolean checkFairness(double meanSlowdownPS, double meanSlowdownP) {
        // Compare E[Slowdown(x)] under policy P with PS.
        return (meanSlowdownP > meanSlowdownPS);
    }
}
```
x??

---

#### Deriving Performance Metrics
Background context: For various scheduling policies in the M/G/1 queue, we typically derive \( E[T] \) (mean time in system) and \( E[T(x)] \) (mean time in system for a job of size x). To find mean slowdown given these metrics:
- First, derive \( E[Slowdown(x)] = \frac{E[T(x)]}{x} \).
- Then, use this to get the overall mean slowdown: \( E[Slowdown] = \int \frac{E[Slowdown(x)] f_S(x)}{dx} \).

:p How can we derive mean slowdown given \( E[T] \) and \( E[T(x)] \)?
??x
To derive mean slowdown, follow these steps:
1. Derive the mean slowdown for a job of size \( x \):
   - \( E[Slowdown(x)] = \frac{E[T(x)]}{x} \)
2. Use this to get overall mean slowdown:
   - \( E[Slowdown] = \int \frac{E[Slowdown(x)] f_S(x)}{dx} \)

Where \( f_S(x) \) is the job size distribution.

```java
// Example logic in pseudocode for deriving mean slowdown
public class SlowdownDerivation {
    public double calculateMeanSlowdown(double[] E_Tx, int[] sizes, double[] fS_x) {
        // Derive E[Slowdown(x)] and then overall E[Slowdown]
        double total = 0;
        for (int i = 0; i < E_Tx.length; i++) {
            total += (E_Tx[i] / sizes[i]) * fS_x[i];
        }
        return total;
    }
}
```
x??

#### FCFS vs LCFS Scheduling Policies
FCFS (First-Come, First-Served) and LCFS (Last-Come, First-Served) are non-preemptive policies that do not use job size information. In an M/G/1 queue setting, they serve jobs based on their arrival order or the reverse of it.
:p When would one use LCFS?
??x
LCFS is used when arriving jobs can be easily accessed from a stack (e.g., last job to arrive). It ensures that new arrivals are served immediately once space becomes available.
x??

---

#### FCFS Service Order Analysis
The M/G/1 queue with FCFS service order has a specific embedded DTMC formulation. This helps in understanding the behavior of jobs over time, particularly at departure points.
:p What is the embedded DTMC formulation for an M/G/1/FCFS queue?
??x
For the M/G/1/FCFS queue, we consider the number of jobs in the system at the time of each departure. Let \( \{X_i, i \geq 0\} \) be a sequence representing these states. The transition probability is given by:
\[ P_{ij} = \text{Probability that when leaving state } i, \text{ we next go to state } j = \frac{\lambda^{j-i+1} e^{-\lambda x}}{(j-i+1)!} f_S(x) dx \]
where \( f_S(x) \) is the service time distribution. The limiting probability \( \pi_i \) specifies the fraction of jobs that leave behind \( i \) jobs.
x??

---

#### LCFS Service Order Analysis
LCFS also follows a similar embedded DTMC formulation as FCFS, making it equivalent in terms of the number of jobs in the system and mean response time. This is due to PASTA (Poisson Arrivals See Time Averages).
:p How does the argument for M/G/1/LCFS differ from M/G/1/FCFS?
??x
The argument for M/G/1/LCFS remains identical because any non-preemptive service order that doesn't use job size information results in the same behavior. This is due to the fact that both policies effectively treat jobs symmetrically over time.
x??

---

#### Non-Preemptive Policies and Job Size Ignorance
Non-preemptive policies like FCFS, LCFS, and RANDOM do not use any job size information. They are work-conserving but handle jobs based on their arrival or random selection.
:p Why is job size ignored in non-preemptive scheduling?
??x
Ignoring job size prevents the policy from affecting the distribution of the number of arrivals during a service time. If sizes were used, it would change how arrivals occur, influencing the system dynamics.
x??

---

#### Variance of Response Time
While FCFS, LCFS, and RANDOM have the same mean response time, their variances differ significantly. LCFS can lead to very high response times due to potential long waits for the system to clear before servicing a new job.
:p Why do different policies like FCFS, LCFS, and RANDOM have different Var(T)?
??x
The variance of response times varies because LCFS may result in long waiting periods if it has to wait for the system to become empty. In contrast, FCFS serves jobs closer to their arrival time, leading to lower variability.
x??

---

#### Laplace Transform of Waiting Time for LCFS
For M/G/1/LCFS, we can use the Laplace transform of waiting times to derive its variance. This involves understanding the busy periods started by job sizes and using these to compute the expected waiting time.
:p How do you determine Var(T)LCFS?
??x
To find \( \tilde{T}_{\text{LCFS}}(s) \), we use the Laplace transform of the excess service time, denoted as \( \tilde{S}_e(s) \). The waiting time is given by:
\[ \tilde{T}_{\text{LCFS}}(s|busy) = \frac{\tilde{S}_e(s)}{s + \lambda - \frac{\lambda}{\tilde{B}(s)}} \]
where \( \tilde{S}_e(s) \) and \( \tilde{B}(s) \) are the Laplace transforms of the excess service time and busy period, respectively.
x??

---

