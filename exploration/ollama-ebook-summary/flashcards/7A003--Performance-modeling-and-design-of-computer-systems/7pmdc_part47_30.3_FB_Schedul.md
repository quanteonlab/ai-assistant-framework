# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 47)

**Starting Chapter:** 30.3 FB Scheduling

---

#### PLCFS vs PS: Preemption and Performance

Background context explaining how PLCFS works compared to PS. The performance metrics such as mean slowdown, preemptions, and wasted time are discussed.

:p What is the expected number of times a tagged job will be interrupted under PLCFS?

??x
The expected number of times our tagged job with size $x $ gets interrupted is given by$\lambda x $, where $\lambda$ is the arrival rate. This is because each job creates two preemptions—when it arrives and when it departs.

The formula for wasted time under PLCFS is:
$$E[Wasted-Time (x)] = \lambda x \cdot E[S]$$

Where $S $ is the expected length of an interruption, and$1 - \rho $ is the utilization factor. The total expected completion time$E[T(x)]$ for a job under PLCFS is:
$$E[T(x)] = x + \frac{\lambda x E[S]}{1 - \rho} = \frac{x}{1 - \rho}$$

The slowdown for a job of size $x$ under PLCFS is:
$$E[Slowdown (x)] = 1$$

C/Java code to illustrate the concept:
```java
public class Job {
    double size;
    double arrivalRate; // λ
    double utilization; // ρ

    public double expectedPreemptionTimes() {
        return this.arrivalRate * this.size;
    }

    public double expectedCompletionTime() {
        return this.size / (1 - this.utilization);
    }
}
```

x??

---

#### FB Scheduling and Job Age

Background context explaining how FB scheduling uses job age to prioritize jobs. The concept of "transformer glasses" is introduced, which transforms the size of larger jobs to $x$ when evaluating the remaining work in the system.

:p How does FB scheduling decide which job gets CPU time?

??x
FB scheduling gives priority to the job with the lowest CPU age. If several jobs have the same lowest age, they share the CPU using PS (Processor Sharing).

The algorithm can be described as:
1. The job with the lowest CPU age receives the CPU.
2. If multiple jobs have the same lowest age, these jobs share the CPU using PS.

C/Java code to illustrate this concept:
```java
public class Job {
    int size;
    long age;

    public void checkForService() {
        // Logic to determine if a job should get CPU time based on its age.
        if (age == minAge()) {
            // Get CPU time using PS.
        }
    }

    private int minAge() {
        // Find the minimum age among all jobs in the queue.
        return ...;
    }
}
```

x??

---

#### Derivation of E[T(x)]FB

Background context explaining how to derive $E[T(x)]$ for FB scheduling, using concepts like job size transformation and remaining work.

:p What is $E[T(x)]$ under FB scheduling?

??x
The expected completion time $E[T(x)]$ for a job of size $x$ under FB scheduling can be derived as follows:

1. The job itself:$x$ units.
2. Expected remaining work in the system when the job arrives, assuming all jobs have service requirements no more than $x$.
3. Expected work due to new arrivals while the job is in the system.

The formula for $E[T(x)]$ under FB scheduling is:
$$E[T(x)]FB = x + \frac{\lambda E[S^2_x]}{2(1 - \rho_x)} + \lambda E[T(x)]FBE[Sx]$$

This can be simplified to:
$$

E[T(x)]FB = x + \frac{\lambda E[S^2_x]}{2(1 - \rho_x)} + \frac{\rho_x}{1 - \rho_x}E[T(x)]FB$$

By solving for $E[T(x)]FB$:
$$E[T(x)]FB (1 - \rho_x) = x + \frac{\lambda E[S^2_x]}{2(1 - \rho_x)}$$
$$

E[T(x)]FB = x(1 - \rho_x) + \frac{1}{2} \frac{\lambda E[S^2_x]}{(1 - \rho_x)^2}$$

C/Java code to illustrate the concept:
```java
public class Job {
    double size;
    double arrivalRate; // λ
    double utilization; // ρ

    public double expectedCompletionTimeFB() {
        return this.size * (1 - this.utilization) + 0.5 * (this.arrivalRate * Math.pow(this.expectedServiceSize(), 2)) / Math.pow(1 - this.utilization, 2);
    }

    private double expectedServiceSize() {
        // This is a placeholder for the actual computation of E[Sx].
        return ...;
    }
}
```

x??

---

#### DFR and FB Scheduling Performance

Background context explaining how jobs with decreasing failure rate (DFR) can benefit from FB scheduling, as younger jobs have lower remaining service times.

:p How does the expected completion time under FB compare to PS for DFR job size distributions?

??x
For a job size distribution with DFR, younger jobs (jobs that have been in the system longer and thus have had more service) are expected to have lower remaining service times. Therefore, the expected completion time $E[T]$ under FB scheduling is less than under PS:
$$E[T]_{FB} < E[T]_{PS}$$

This result can be proven formally as stated in [189].

x??

---

---

#### FB and PS Scheduling Policies
Background context: The problem discusses the comparison between Foreground-Background (FB) scheduling and Processor-Sharing (PS) under different job size distributions. FB favors younger jobs, while PS treats all jobs equally.

:p What is the primary difference in performance between FB and PS when the job size distribution follows an Exponential distribution?
??x
In an Exponential job size distribution, E[T]FB = E[T]PS because age is independent of remaining time. However, FB provides a slight preference to shorter jobs due to their smaller expected original size, which reduces E[Slowdown ].

```java
// Pseudocode for calculating mean response times under FB and PS
public class Scheduler {
    public double calculateMeanResponseTime(double load) {
        return load / (1 - load); // M/M/1/FCFS or M/M/1/PS with same load
    }
}
```
x??

---

#### Mean Response Time Formulas
Background context: This section lists various formulas for calculating the mean response time in different queueing models. The formulas are derived from specific conditions and distributions.

:p Match each of the following 12 expressions to one of the given formulas.
??x
- (1) E[T]M/G/1/FCFS : (d)
- (2) E[T]M/G/1/PS : (c)
- (3) E[T]M/G/1/LCFS : (f)
- (4) E[T]M/G/1/PLCFS : (g)
- (5) E[T]M/M/1/FCFS : (b)
- (6) E[T]M/M/1/PS : (a)
- (7) E[T]M/M/1/FB : (e)
- (8) ρ : (d)
- (9) E[B]M/G/1/FCFS : (c)
- (10) E[B]M/M/1/FCFS : (b)
- (11) E[Se] : (f)
- (12) E[W]M/G/1/FCFS : (e)

```java
// Pseudocode for matching formulas to expressions
public class FormulaMatcher {
    public String matchExpression(String expression) {
        // Logic to map each expression to its formula
        return "Formula";
    }
}
```
x??

---

#### FB versus PS under Exponential Workloads
Background context: The problem asks to prove that the mean response times for both FB and PS are equal when the job size distribution is Exponential.

:p Prove formally that E[T]FB = E[T]PS for an M/G/1 server with Exponential job size distribution.
??x
For an Exponential distribution, age and remaining time are independent. Thus, biasing towards younger jobs (as FB does) does not affect the mean response time since the expected service requirement is the same regardless of the job's arrival order.

Mathematically:
- E[T]M/G/1/FCFS = 1/(λ(1 - ρ))
- Since age and remaining time are independent for Exponential, this holds true for both FB and PS.

```java
// Pseudocode for proving equality
public class ResponseTimeProof {
    public double proveEquality() {
        double lambda = 0.8; // Load factor
        return 1 / (lambda * (1 - lambda)); // Mean response time formula
    }
}
```
x??

---

#### Starvation under FB
Background context: The problem explores the impact of FB on starvation, especially for large jobs.

:p Compare the mean slowdown of a job in the 90th percentile under FB and under PS.
??x
Under FB, smaller jobs are favored, which can lead to better overall performance but may increase the slowdown for larger jobs due to potential priority inversion. The exact comparison requires detailed computation using symbolic math tools.

```java
// Pseudocode for computing mean slowdown
public class SlowdownComparison {
    public double computeMeanSlowdown(double percentile) {
        // Use symbolic math package to calculate E[Slowdown ]FB and E[Slowdown ]PS
        return 0; // Placeholder value
    }
}
```
x??

---

#### M/G/1/FB Transform
Background context: The problem asks for the derivation of the transform for response time under FB.

:p Derive the Laplace transform for time in system under M/G/1/FB and use it to determine the first two moments of response time.
??x
Deriving the Laplace transform involves considering the interruption patterns of jobs and their contributions. For an M/G/1/FB, the key is recognizing that smaller jobs are served faster.

```java
// Pseudocode for deriving Laplace transform
public class FBTransform {
    public double deriveLaplaceTransform() {
        // Use symbolic math package to derive the transform
        return 0; // Placeholder value
    }
}
```
x??

---

#### Database Performance Analysis
Background context: The problem models a database system with an M/M/1/PS queue and load-dependent service rates.

:p Solve for the mean response time under Bianca’s M/M/1/PS system.
??x
To solve this, model Bianca’s database as an M/M/1/PS queue. Given λ = 0.9 and μ(n) from Figure 30.5, use a Markov chain approach to calculate the mean response time.

```java
// Pseudocode for solving mean response time
public class DatabaseResponseTime {
    public double solveMeanResponseTime() {
        // Use Markov chain to model the system and calculate mean response time
        return 0; // Placeholder value
    }
}
```
x??

---

#### Hybrid FCFS/PS Architecture
Background context: The problem explores Bianca’s hybrid architecture combining FCFS with PS.

:p Compute the mean response time for Bianca’s new architecture.
??x
Bianca's architecture limits concurrent transactions to 4, then queues remaining jobs in a FCFS queue. The mean response time can be computed by considering both the PS and FCFS components.

```java
// Pseudocode for computing mean response time with hybrid architecture
public class HybridQueue {
    public double computeMeanResponseTime() {
        // Use Markov chain to model the system and calculate mean response time
        return 0; // Placeholder value
    }
}
```
x??

---

#### Limited Processor-Sharing System Intuition
Background context: The problem suggests increasing the MPL in a limited PS system if job sizes are highly variable.

:p Explain Varun’s suggestion for increasing the MPL when job size distribution is highly variable.
??x
Increasing the MPL can help balance the trade-off between fairness and performance. When job sizes vary widely, allowing more multiprogramming levels (MPL) ensures that large jobs get adequate service, reducing their slowdown compared to a strict FCFS system.

```java
// Pseudocode for explaining Varun’s suggestion
public class PSIntuition {
    public String explainVarunsSuggestion() {
        return "Increasing MPL increases the chances of larger jobs being served promptly, mitigating their slowdown.";
    }
}
```
x??

#### Non-Preemptive Priority Queueing
Non-preemptive priority queueing is used when a job cannot be stopped once it has started running. This model ensures that jobs of higher priority are serviced before those of lower priority, but once a job starts execution, no other job can interrupt its service.
:p What is non-preemptive priority queueing?
??x
Non-preemptive priority queueing refers to a scheduling mechanism where a job cannot be interrupted after it has started executing. Higher priority jobs will preempt the server when available, ensuring that they are processed before lower priority ones. However, once a higher-priority job starts running, no other job can interrupt its service.
x??

---

#### Preemptive Priority Queueing
In contrast to non-preemptive priority queueing, preemptive priority queueing allows for the interruption of a currently executing job if a higher priority job arrives. This ensures that high-priority tasks are given immediate attention without waiting for the current task to complete.
:p What is preemptive priority queueing?
??x
Preemptive priority queueing allows a running job to be interrupted and replaced by a higher-priority job whenever such a job arrives. This mechanism prioritizes high-priority tasks over lower-priority ones, ensuring that critical tasks are handled promptly.
x??

---

#### M/G/1 Priority Queue Model
The M/G/1 priority queue model divides jobs into multiple priority classes based on their importance or urgency. Each class has its own arrival rate and service time distribution. The server serves the highest-priority non-empty queue first, ensuring that higher-priority jobs are processed before lower-priority ones.
:p What is an M/G/1 priority queue?
??x
An M/G/1 priority queue model consists of multiple priority classes where:
- Each class has its own Poisson arrival rate λk and service time distribution with moments E[Sk] and E[S2 k].
- The server always serves the job at the head of the highest-priority non-empty queue.
This ensures that higher-priority jobs are processed before lower-priority ones.
x??

---

#### Average Number of Jobs in Queue
The average number of jobs in the queue for a specific priority class $k$ can be calculated using the formula:
$$E[NQ(k)] = \frac{\rho_k}{1 - \rho_k}$$where $\rho_k = \lambda_k \cdot E[S_k]$ is the traffic intensity.
:p What is the average number of jobs in queue for priority class k?
??x
The average number of jobs in queue for a specific priority class $k$ can be calculated using the formula:
$$E[NQ(k)] = \frac{\rho_k}{1 - \rho_k}$$where $\rho_k = \lambda_k \cdot E[S_k]$ is the traffic intensity.
This formula helps in understanding how many jobs, on average, are waiting to be served at any given time for class $k$.
x??

---

#### Average Time in Queue
The average time a job spends in queue before being serviced can be calculated using Little’s Law:
$$E[TQ(k)] = E[NQ(k)] \cdot E[S_k]$$where $ E[NQ(k)]$is the average number of jobs in the queue for priority class $ k$, and $ E[S_k]$ is the expected service time.
:p What is the average time a job spends in queue?
??x
The average time a job spends in queue before being serviced can be calculated using Little’s Law:
$$E[TQ(k)] = E[NQ(k)] \cdot E[S_k]$$where $ E[NQ(k)]$is the average number of jobs in the queue for priority class $ k$, and $ E[S_k]$ is the expected service time.
This formula helps in understanding the delay experienced by jobs waiting to be serviced, providing insights into system performance.
x??

---

