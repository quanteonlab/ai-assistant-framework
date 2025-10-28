# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 47)

**Starting Chapter:** 30.3 FB Scheduling

---

#### PLCFS vs PS
Background context explaining the difference between PLCFS and PS. The key point is that both policies have the same expected performance but PLCFS has fewer preemptions, leading to less wasted time.
:p What is E[Slowdown (x)] for both PLCFS and PS?
??x
For both PLCFS and PS, the mean slowdown \( E[\text{Slowdown}(x)] = \frac{1}{1 - \rho} \). This means that regardless of the job size, the average time a job spends in the system is the same under these two policies. The performance metrics are identical despite the different mechanisms for handling preemptions.
x??

---

#### PLCFS Preemption Count
Background context explaining how many preemptions occur per job under PLCFS and PS. PLCFS has fewer preemptions compared to PS due to its simpler mechanism of preemption on arrival and departure only.
:p How many preemptions does each job experience in PLCFS?
??x
Each job experiences exactly 2 preemptions in PLCFS: one when it arrives and another when it departs. This is significantly fewer than the multiple preemptions that can occur per job under PS, especially with small quantum sizes.
x??

---

#### FB Scheduling Overview
Background context explaining the need for a scheduling policy that gives preference to smaller jobs based on age. The goal is to achieve lower slowdowns for smaller jobs and potentially reduce the mean slowdown overall.
:p Why is it nice if we could somehow get lower slowdowns for the smaller jobs?
??x
It would allow us to drop our mean slowdown while maintaining good performance for smaller jobs, which are often more critical or time-sensitive. By giving preference to younger jobs (those with less remaining CPU demand), we can ensure that shorter jobs complete faster and do not have to wait as long for service.
x??

---

#### FB Scheduling Algorithm
Background context explaining the Generalized Foreground-Background (FB) scheduling algorithm, which uses job age to prioritize smaller jobs. The key idea is that younger jobs (those with less remaining CPU demand) are given more attention.
:p What is the rule for allocating CPU under FB?
??x
The CPU is allocated to the job with the lowest CPU age. If multiple jobs have the same lowest CPU age, they share the CPU using PS. This ensures that smaller jobs, which tend to be younger and thus have less remaining demand, are prioritized.
x??

---

#### Example of FB Scheduling
Background context explaining how the state of the system changes over time under FB scheduling with a given arrival sequence.
:p What is the completion time for each job in the given example?
??x
- The size 1 job leaves at time 3.
- The size 2 job leaves at time 5.
- The size 3 job leaves at time 6.

This can be visualized as follows:

```plaintext
time:   0  1  2  3  4  5  6
size:    3  2  1

Completion times:
Size 1: 3 (leaves)
Size 2: 5 (leaves)
Size 3: 6 (leaves)
```
x??

---

#### Mathematical Exercise for FB Scheduling
Background context explaining the transformation of job sizes in the derivation of \( E[T(x)] \) under FB scheduling. The goal is to compute the expected remaining work by transforming job sizes.
:p What is the formula for \( E[S_x] \)?
??x
The expected size \( E[S_x] \) under the transformed distribution is given by:
\[ E[S_x] = \int_0^x y f(y) dy + x (1 - F(x)) \]
where \( F(x) \) is the cumulative distribution function of job sizes, and \( f(y) \) is the probability density function.
x??

---

#### Expression for Expected Remaining Work
Background context explaining how to compute the expected remaining work under FB scheduling by considering both initial state and new arrivals during processing.
:p How do you derive \( E[T(x)] \) in FB scheduling?
??x
The derivation of \( E[T(x)] \) involves summing up:
1. The size of job \( x \).
2. The expected remaining work when job \( x \) arrives, considering jobs as having a service requirement no more than \( x \).
3. The expected work due to new arrivals while job \( x \) is in the system.

The formula for \( E[T(x)] \) under FB scheduling is:
\[ E[T(x)]_{\text{FB}} = x + \frac{\lambda E[S^2_x]}{2(1 - \rho_x)} + \lambda E[T(x)]_{\text{FB}} E[S_x] \]

Simplifying, we get:
\[ E[T(x)]_{\text{FB}} (1 - \rho_x) = x + \frac{\lambda E[S^2_x]}{2(1 - \rho_x)} \]
x??

---

#### FB Scheduling and Exponential Workloads
Background context explaining the concept. The problem states that for an M/G/1 server with an Exponential job size distribution, both Foreground-Background (FB) and Processor-Sharing (PS) have equal mean response times under certain conditions.

:p Why would the slowdown under FB be strictly smaller than under PS, even though their mean response times are equal?
??x
The answer is that FB scheduling gives a slight preference to shorter jobs. Even with an Exponential job size distribution, which means age and remaining time are independent, biasing towards younger (smaller age) jobs still provides some advantage in expected original size. This subtle preference for short jobs under FB improves the E[Slowdown] compared to PS.

```java
// Pseudocode example to illustrate the idea:
public class JobScheduler {
    public double calculateExpectedResponseTime(double load, DistributionType jobSizeDistribution) {
        if (jobSizeDistribution == Exponential) {
            return (load / 1 - load); // Mean response time for both FB and PS with Exponential distribution
        } else {
            // Other calculations for different distributions
        }
    }

    public double calculateExpectedSlowdown(double load, DistributionType jobSizeDistribution) {
        if (jobSizeDistribution == Exponential) {
            return (load / 1 - load); // Slowdown is the same as response time in this scenario
        } else {
            // Other calculations for different distributions
        }
    }
}
```
x??

---

#### FB vs. PS under Exponential Workloads
The problem suggests that for an M/G/1 server with an Exponential job size distribution, both Foreground-Background (FB) and Processor-Sharing (PS) have the same mean response time.

:p How can we prove formally that the mean response times under FB and PS are equal when the job size distribution is Exponential?
??x
To prove this, consider the properties of the Exponential distribution. For an Exponential distribution, the remaining service time is independent of age (i.e., jobs of all ages have the same expected remaining service time). Therefore, biasing towards younger jobs does not provide a significant advantage in terms of reducing the mean response time.

The key insight is that both FB and PS will essentially handle jobs in a way that their expected remaining service times are equal due to the independence property of Exponential distribution. This leads to the equality of mean response times under both policies when the job size distribution is Exponential.

```java
// Pseudocode for proving the equality:
public class ResponseTimeAnalysis {
    public double getMeanResponseTimeExponential(double lambda, double mu) {
        return 1 / (mu - lambda); // Mean response time for M/M/1 queue with Exponential distribution
    }

    public void proveEquality() {
        double load = 0.8;
        double serviceRate = 3; // Example values, assume these are the same for FB and PS

        // Calculate mean response times:
        double meanResponseTimeFB = getMeanResponseTimeExponential(load * serviceRate, serviceRate);
        double meanResponseTimePS = getMeanResponseTimeExponential(load * serviceRate, serviceRate);

        System.out.println("E[T]FB: " + meanResponseTimeFB);
        System.out.println("E[T]PS: " + meanResponseTimePS);
    }
}
```
x??

---

#### Starvation under FB Scheduling
The problem considers an M/G/1 server with load ρ=0.8 and different job size distributions, comparing the mean slowdown of large jobs between Foreground-Background (FB) and Processor-Sharing (PS).

:p What is the first percentile where a job does worse under FB than under PS?
??x
The answer involves analyzing the relative performance of small and large jobs under both policies. Since FB favors smaller jobs, it can lead to larger jobs experiencing higher slowdowns compared to PS. The specific percentile can be determined by comparing the expected slowdown for jobs in different percentiles.

```java
// Pseudocode example:
public class JobPerformanceAnalysis {
    public double calculateExpectedSlowdown(double load, DistributionType jobSizeDistribution, int percentile) {
        // Implementation based on the given distributions and policies
        if (load == 0.8 && jobSizeDistribution == Exponential) {
            return (load / 1 - load); // Example calculation for mean slowdown
        }
        return -1; // Placeholder value
    }

    public int findPercentileWhereFBWorseThanPS() {
        double[] percentiles = {90, 95, 99};
        for (int p : percentiles) {
            double slowdownFB = calculateExpectedSlowdown(0.8, Exponential, p);
            double slowdownPS = calculateExpectedSlowdown(0.8, Exponential, p);

            if (slowdownFB > slowdownPS) {
                return p;
            }
        }
        return -1; // No such percentile found
    }
}
```
x??

---

#### PLCFS Analysis
The problem develops a clearer understanding of Preemptive-Less-Attained-Service (PLCFS) by determining the Laplace transform for time in system and then using it to determine the first 2 moments of response time.

:p What is the Laplace transform for time in system under PLCFS?
??x
The Laplace transform for time in system under PLCFS, \(\tilde{T}(s)\), can be determined by considering the response time of a job and how many times it gets interrupted. This involves breaking down the problem into simpler components: first finding the response time \(T(x)\) for jobs of size \(x\), then determining the number of interruptions and their contribution.

```java
// Pseudocode example:
public class PLCFSAnalysis {
    public double getLaplaceTransformPLCFS(double s, double arrivalRate, double serviceRate) {
        // Example calculation based on the given parameters
        return 1 / (s * (1 - (arrivalRate / serviceRate))); // Simplified Laplace transform for M/M/1 queue
    }

    public void calculateFirstTwoMoments() {
        double s = 0.5; // Example value
        double arrivalRate = 2;
        double serviceRate = 3;

        double laplaceTransform = getLaplaceTransformPLCFS(s, arrivalRate, serviceRate);
        System.out.println("Laplace Transform: " + laplaceTransform);

        // Using the transform to find first two moments (mean and variance)
    }
}
```
x??

---

#### M/G/1/FB Response Time
The chapter derived the mean response time for FB. The problem asks to derive the transform of response time using similar arguments.

:p How can we derive the transform of response time under M/G/1/FB?
??x
To derive the transform of response time under M/G/1/FB, follow a similar approach as used in deriving the mean response time for FCFS and PS. The key is to consider the time a job spends in the system and how this is affected by the service interruptions and arrivals.

```java
// Pseudocode example:
public class FBResponseTime {
    public double getTransformFB(double s, double arrivalRate, DistributionType jobSizeDistribution) {
        // Implementation based on the given distributions and policies
        if (jobSizeDistribution == Exponential) {
            return 1 / (s * (1 - (arrivalRate / (3000)))); // Example calculation for M/G/1/FB with Exponential distribution
        }
        return -1; // Placeholder value
    }

    public void deriveTransform() {
        double s = 0.5; // Example value
        double arrivalRate = 2;
        DistributionType jobSizeDistribution = Exponential;

        double transformFB = getTransformFB(s, arrivalRate, jobSizeDistribution);
        System.out.println("Transform: " + transformFB);
    }
}
```
x??

---

#### M/M/1/PS with Load-Dependent Service Rate
The problem models a database system as an M/M/1/PS queue with load-dependent service rate and asks for the intuition behind increasing the MPL.

:p What is the intuition behind Varun's suggestion to increase the MPL?
??x
Varun suggests increasing the MPL when the job size distribution is highly variable because this can help balance the system. Even though a higher MPL reduces the overall service rate, it allows the system to better handle jobs of varying sizes. This can reduce the variability in response times and improve overall performance.

```java
// Pseudocode example:
public class DatabaseSystem {
    public void analyzeMPLImpact(double jobSizeVariability) {
        if (jobSizeVariability > 10) { // Example threshold for high variability
            System.out.println("Increase MPL to more than 4");
        } else {
            System.out.println("Keep current MPL settings");
        }
    }

    public void simulateSystem() {
        double jobSizeVariability = 20; // Example value indicating high variability

        analyzeMPLImpact(jobSizeVariability);
    }
}
```
x??

---

#### Limited Processor-Sharing System
The problem provides references for analyzing the limited Processor-Sharing system in Figure 3.6.

:p What is the intuition behind Bianca's FCFS/PS architecture?
??x
Bianca’s FCFS/PS architecture combines elements of both First-Come-First-Served (FCFS) and Processor-Sharing (PS). It helps balance between fairness and efficiency by using FCFS for initial job handling and PS to manage service rates. This hybrid approach can provide better performance in systems with highly variable job sizes.

```java
// Pseudocode example:
public class HybridSystem {
    public void analyzeHybridPerformance(double jobSizeVariability) {
        if (jobSizeVariability > 10) { // Example threshold for high variability
            System.out.println("Use FCFS/PS hybrid approach");
        } else {
            System.out.println("Use pure PS or FCFS depending on the scenario")
        }
    }

    public void simulateSystem() {
        double jobSizeVariability = 20; // Example value indicating high variability

        analyzeHybridPerformance(jobSizeVariability);
    }
}
```
x??

#### Non-Preemptive Priority Queueing
Background context: In non-preemptive priority queueing, once a job starts running, it cannot be interrupted even if a higher-priority job arrives. This is common in scenarios where jobs are time-sensitive or resource-intensive and should not be interrupted.
:p What is non-preemptive priority queueing?
??x
In non-preemptive priority queueing, a job that has started execution runs until completion without being interrupted by a higher-priority job. This model is suitable for tasks that must run to completion and cannot be paused or restarted once they begin.
x??

---

#### Preemptive Priority Queueing
Background context: In preemptive priority queueing, the current job in service can be interrupted if a higher-priority job arrives, allowing the higher-priority job to take over. This is useful for interactive jobs that require immediate attention and can be paused or resumed without loss of data.
:p What is preemptive priority queueing?
??x
In preemptive priority queueing, the server preempts the current job in service if a higher-priority job arrives, allowing the new job to start execution immediately. This ensures that high-priority jobs receive immediate attention and do not wait for lower-priority jobs to complete.
x??

---

#### M/G/1 Priority Queue Model
Background context: The M/G/1 priority queue model is used to analyze systems with multiple priority classes. Each class has its own Poisson arrival process and service time distribution, allowing the system to handle different types of jobs efficiently.
:p What is an M/G/1 priority queue?
??x
The M/G/1 priority queue is a queuing model where:
- 'M' stands for Markovian (Poisson) job arrivals
- 'G' represents general service time distribution
- '1' indicates there is one server

This model divides arriving jobs into n priority classes, with class 1 being the highest and class n the lowest. Each class has a Poisson arrival rate λk = λ·pk, where pk is the proportion of jobs in that class.
x??

---

#### Priority Classes in M/G/1 Model
Background context: In the M/G/1 priority queue model, jobs are divided into multiple classes based on their priority levels. The server always serves from the highest non-empty priority class to ensure efficient use of resources and prioritize critical tasks.
:p How are jobs prioritized in an M/G/1 priority queue?
??x
In an M/G/1 priority queue with n priority classes, jobs are divided into separate queues based on their priority levels. The server always serves from the highest non-empty queue, ensuring that higher-priority jobs are attended to first.
Example: If there are 3 priority classes (class 1, class 2, and class 3), the server will:
- First serve from class 1 if it is not empty
- Then move on to class 2 if class 1 is empty but class 2 is not
- Finally, consider class 3 as a last resort
x??

---

#### Average Number of Jobs in Queue (E[NQ(k)])
Background context: The average number of jobs in the queue for each priority class can be calculated using queuing theory formulas. This metric helps understand the congestion and performance of the system.
:p What is E[NQ(k)]?
??x
The expected number of jobs in the queue for a job of priority k, denoted as \(E[NQ(k)]\), can be determined using queuing theory. It represents the average number of jobs of that priority level waiting to be served.

Formula: 
\[E[NQ(k)] = \frac{\lambda_k E[S^2]}{\mu - \lambda_k}\]
where:
- \(\lambda_k\) is the arrival rate for class k
- \(E[S]\) and \(E[S^2]\) are the first and second moments of the service time distribution, respectively
- \(\mu\) is the service rate

Example: If \(\lambda_1 = 0.5\), \(E[S] = 2\), \(E[S^2] = 4\), and \(\mu = 3\):
\[E[NQ(1)] = \frac{0.5 \times 4}{3 - 0.5} = \frac{2}{2.5} = 0.8\]
x??

---

#### Average Time in Queue (E[TQ(k)])
Background context: The average time a job spends waiting in the queue before being served, denoted as \(E[TQ(k)]\), is an important metric for understanding the performance and efficiency of the system.
:p What is E[TQ(k)]?
??x
The expected time a job spends in the queue before being served, denoted as \(E[TQ(k)]\), can be calculated using queuing theory. This metric provides insight into the average waiting time for jobs of each priority level.

Formula:
\[E[TQ(k)] = \frac{1}{\mu - \lambda_k}\]
where:
- \(\lambda_k\) is the arrival rate for class k
- \(\mu\) is the service rate

Example: If \(\lambda_2 = 0.3\) and \(\mu = 4\):
\[E[TQ(2)] = \frac{1}{4 - 0.3} = \frac{1}{3.7} \approx 0.27\]
x??

---

