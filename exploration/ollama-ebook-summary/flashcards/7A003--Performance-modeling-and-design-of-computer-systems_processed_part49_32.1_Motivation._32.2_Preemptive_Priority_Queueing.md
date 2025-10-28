# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 49)

**Starting Chapter:** 32.1 Motivation. 32.2 Preemptive Priority Queueing

---

#### Non-Preemptive vs. Preemptive Policies
Background context: The chapter discusses differences between non-preemptive and preemptive scheduling policies, focusing on how they handle job sizes and variability. Non-preemptive policies do not interrupt jobs once started, whereas preemptive policies can pause a job to start another with higher priority or smaller size.
:p What is the key difference between non-preemptive and preemptive scheduling policies?
??x
Non-preemptive policies cannot be interrupted, while preemptive policies can suspend an in-progress job to handle a more urgent or smaller task. This makes preemptive policies potentially more efficient under highly variable job sizes but less straightforward.
x??

---

#### Motivation for Preemptive Policies
Background context: The text highlights that non-preemptive policies suffer from E[S2] factors due to waiting for the excess of jobs in service, which is problematic with high variability in job sizes. In contrast, preemptive policies like PS and PLCFS are generally better but have limitations.
:p What discourages us about non-preemptive scheduling policies under highly variable job size distributions?
??x
Non-preemptive policies suffer from an E[S2] term in their mean response time calculations, which arises because they wait for the excess of jobs currently being serviced. This issue is more pronounced with high variability in job sizes.
x??

---

#### Preemptive Priority Queueing Overview
Background context: The chapter introduces preemptive priority queueing where higher-priority jobs can interrupt lower-priority ones. It uses a network analogy to explain how different packet streams compete for resources based on their priorities.
:p How does preemptive priority queueing differ from non-preemptive queueing?
??x
In preemptive priority queueing, if a job of higher priority arrives, the current job is paused and the higher-priority job starts service. This differs from non-preemptive policies where jobs must complete without interruption.
x??

---

#### Preemption in Network Streams
Background context: The text describes an application scenario involving multiple packet streams with different priorities sharing a communication link. Only one stream can be active at any time, and higher-priority streams preempt the current stream's service.
:p Can you explain the network scenario described?
??x
In this scenario, various packet streams compete for a shared communication link. When a new high-priority stream starts, it preempts the ongoing lower-priority stream until completion. Only one stream can use the link at any moment.
x??

---

#### Mean Time in System (E[T(k)]) Calculation
Background context: The chapter outlines how to calculate E[T(k)]P-Priority for jobs of priority k in a preemptive system, considering the three components: current job service, waiting time before arrival, and total work required from all classes.
:p How is E[T(k)] calculated in a preemptive priority queueing system?
??x
E[T(k)]P-Priority is computed by summing up the expected service time of the current job, the expected waiting time until the next job arrives, and the total remaining work from other classes. This approach accounts for all work required before a job leaves.
x??

---

#### Components of E[T(k)]
Background context: The calculation involves three key components: the remaining service time of the current job, the expected wait time until another job of higher priority arrives, and the sum of the moments E[Sk] from all classes. This breakdown helps in understanding the overall system behavior.
:p What are the three main components of E[T(k)]?
??x
The three main components are:
1. Remaining service time of the current job.
2. Expected wait time until a higher-priority job arrives.
3. Sum of the moments \(E[S_k]\) for all classes.
These components help in accurately estimating the mean time in system for a given priority class.
x??

#### Component (2) of E[T(k)] Calculation
Background context: In a preemptive priority queue, we need to determine the expected time required to complete service on all jobs of priority 1 through k already in the system when our arrival walks in. This is component (2) of \(E[T(k)]\). The key difference from non-preemptive systems is that jobs may have been partially worked on.

:p How do we compute component (2) for a preemptive priority queue?
??x
We cannot use the same approach as in non-preemptive queues, where you add up the expected number of jobs in each class weighted by the mean job size. Instead, we need to consider the remaining work due to only jobs of priority 1 through k.

The computation involves considering the total expected remaining work if the system had only arrivals of classes 1 through k and using a work-conserving scheduling order such as FCFS (First-Come-First-Served).

Formula:
\[
(2) = \frac{\lambda}{\sum_{i=1}^k p_i E[S_i]} \left(\frac{1 - \sum_{i=1}^k \rho_i}{\sum_{i=1}^k \rho_i}\right)
\]

Simplified:
\[
(2) = \sum_{i=1}^k \rho_i \frac{E[S_i^2]}{2 E[S_i] (1 - \sum_{i=1}^k \rho_i)} = \frac{\lambda}{1 - \sum_{i=1}^{k-1} \rho_i} + \sum_{i=1}^k \rho_i \frac{E[S_i^2]}{2 E[S_i] (1 - \sum_{i=1}^k \rho_i)}
\]

x??

---

#### Component (3) of E[T(k)] Calculation
Background context: Component (3) of \(E[T(k)]\) is the expected total service time required for all jobs of priority 1 through k-1 that arrive before our arrival. This can be simplified as follows:
\[
(3) = \frac{\sum_{i=1}^{k-1} E[T(i)] \cdot \lambda_i \cdot E[S_i]}{E[T(k)] \sum_{i=1}^{k-1} \rho_i} = \frac{E[T(k)]}{\sum_{i=1}^{k-1} \rho_i}
\]

:p How is component (3) computed?
??x
Component (3) is simplified to:
\[
(3) = \frac{\sum_{i=1}^{k-1} E[T(i)] \cdot \lambda_i \cdot E[S_i]}{E[T(k)] \sum_{i=1}^{k-1} \rho_i}
\]

This formula represents the expected total service time for all jobs of priority 1 through k-1, given by the weighted sum of expected times and considering the mean service times.

x??

---

#### Total Expected Time in System (E[T(k)]P-Priority)
Background context: The total expected time in system \(E[T(k)]_{\text{P-Priority}}\) for a job of class k in a preemptive priority queue is the sum of three components:
1. \(E[S_k]\): Mean service time for a job of priority class k.
2. Component (2) as described above, representing the expected remaining work due to jobs of priorities 1 through k.
3. Component (3), representing the expected total service time required for all jobs of priority 1 through k-1.

Formula:
\[
E[T(k)]_{\text{P-Priority}} = E[S_k] + \sum_{i=1}^k \rho_i \frac{E[S_i^2]}{2 E[S_i] (1 - \sum_{i=1}^k \rho_i)} + \frac{E[T(k)]}{\sum_{i=1}^{k-1} \rho_i}
\]

Simplified:
\[
E[T(k)]_{\text{P-Priority}} = E[S_k] + \frac{\lambda}{1 - \sum_{i=1}^{k-1} \rho_i} + \sum_{i=1}^k \rho_i \frac{E[S_i^2]}{(1 - \sum_{i=1}^k \rho_i) E[S_i]}
\]

:p What is the formula for \(E[T(k)]P-Priority\)?
??x
The total expected time in system for a job of class k under preemptive priority can be computed using the following expression:
\[
E[T(k)]_{\text{P-Priority}} = E[S_k] + \frac{\lambda}{1 - \sum_{i=1}^{k-1} \rho_i} + \sum_{i=1}^k \rho_i \frac{E[S_i^2]}{(1 - \sum_{i=1}^k \rho_i) E[S_i]}
\]

This expression accounts for the mean service time, the remaining work due to jobs of priorities 1 through k, and the total service time required for jobs of priority 1 through k-1.

x??

---

#### Residence Time in Preemptive Priority Queues
Background context: In preemptive queues, the residence time is different from the service time. The residence time includes interruptions where the job may be interrupted by higher-priority jobs and then resumed later. The residence time can be divided into two components:
1. Waiting time (Wait): Time until the job starts serving.
2. Residence time (Res): Time from when the job first receives some service until it leaves the system.

:p What is the difference between residence time and service time in preemptive priority queues?
??x
In preemptive priority queues, the residence time is longer than the service time because it includes all interruptions where the job may be interrupted by higher-priority jobs and then resumed later. The service time only accounts for the actual processing time of the job.

:p How can we interpret \(E[T(k)]P-Priority\)?
??x
The expression \(E[T(k)]_{\text{P-Priority}} = E[S_k] + \frac{\lambda}{1 - \sum_{i=1}^{k-1} \rho_i} + \sum_{i=1}^k \rho_i \frac{E[S_i^2]}{(1 - \sum_{i=1}^k \rho_i) E[S_i]}\) can be interpreted as the total expected time in the system for a job of class k under preemptive priority. It breaks down into:
- The mean service time \(E[S_k]\),
- An adjustment term accounting for the remaining work due to jobs of priorities 1 through k,
- And an additional term representing the total service time required for lower-priority jobs.

x??

---

#### First Term Explanation
Background context: The first term \( E[S_k] \left(1 - \sum_{i=1}^{k-1} \rho_i\right) \) represents the mean residence time of a job of class \( k \), denoted as \( E[\text{Res}(k)] \). It calculates the expected length of a busy period started by a job of size \( E[S_k] \), where only jobs of classes 1 through \( k-1 \) are allowed in this busy period.

Formula: 
\[ E[\text{Res}(k)] = E[S_k] \left(1 - \sum_{i=1}^{k-1} \rho_i\right) \]

:p What does the first term \( E[S_k] \left(1 - \sum_{i=1}^{k-1} \rho_i\right) \) represent in a preemptive priority queueing system?
??x
The first term represents the mean residence time of the job of class \( k \), which is equivalent to the expected length of a busy period initiated by a job of size \( E[S_k] \). This busy period only includes interruptions from jobs of classes 1 through \( k-1 \).

```java
// Pseudocode for calculating the mean residence time
public double calculateMeanResidenceTime(double Sk, List<Double> rho) {
    int k = rho.size() + 1; // Class index starts from 1 in the given context
    double sumRho = 0;
    for (int i = 1; i < k - 1; i++) {
        sumRho += rho.get(i);
    }
    return Sk * (1 - sumRho);
}
```
x??

---

#### Second Term Explanation
Background context: The second term in equation \(32.1\) is defined as:
\[ E[\text{Wait}(k)] = \frac{\sum_{i=1}^{k} \rho_i E[S_i^2]} {2E[S_k] (1 - \sum_{i=1}^{k-1} \rho_i)} \left( \frac{1}{1 - \sum_{i=1}^{k} \rho_i} \right) \]

This term represents the mean time until a job of priority \( k \) first receives service. It is almost identical to \( E[\text{TQ}(k)] \) for non-preemptive priority, but accounts only for jobs of class 1 through \( k \).

Formula: 
\[ E[\text{Wait}(k)] = \frac{\sum_{i=1}^{k} \rho_i E[S_i^2]} {2E[S_k] (1 - \sum_{i=1}^{k-1} \rho_i)} \left( \frac{1}{1 - \sum_{i=1}^{k} \rho_i} \right) \]

:p What does the second term in equation \(32.1\) represent?
??x
The second term represents the mean time until a job of priority \( k \) first receives service. It accounts only for jobs of class 1 through \( k \), unlike the non-preemptive case, which considers all classes.

```java
// Pseudocode for calculating the waiting time
public double calculateWaitingTime(List<Double> rho, List<Double> E_Squared, double Sk) {
    int k = rho.size() + 1; // Class index starts from 1 in the given context
    double sumRho = 0;
    for (int i = 1; i < k - 1; i++) {
        sumRho += rho.get(i);
    }
    double numerator = 0;
    for (int i = 1; i <= k; i++) {
        numerator += rho.get(i-1) * E_Squared.get(i-1);
    }
    return numerator / (2 * Sk * (1 - sumRho)) * (1.0 / (1 - sumRho));
}
```
x??

---

#### Preemptive vs Non-preemptive Comparison
Background context: In the case of non-preemptive priority and Shortest Job First (SJF), a high-priority job does not necessarily perform well due to the variability in job sizes. However, in preemptive priority, this is different because a high-priority job only sees variability from lower-priority jobs.

:p Is a high-priority job in a preemptive priority queue guaranteed to have better performance than in non-preemptive priority, even with high variability in job sizes?
??x
Yes, a high-priority job in a preemptive priority queue is more likely to have better performance. This is because the mean residence time and waiting time depend only on the first \( k \) classes of jobs (where \( k \) is the priority level). Thus, higher-priority jobs see less variability compared to non-preemptive systems where all job sizes contribute.

```java
// Pseudocode for comparing preemptive vs. non-preemptive priority
public double calculatePerformancePreemptive(double Sk, List<Double> rho, List<Double> E_Squared) {
    int k = rho.size() + 1;
    double sumRhoNonPreemptive = 0;
    for (int i = 1; i <= k; i++) {
        sumRhoNonPreemptive += rho.get(i-1);
    }
    
    // Preemptive performance
    double preemptiveMeanResidenceTime = Sk * (1 - sumRhoNonPreemptive + rho.get(k-1));
    double preemptiveWaitingTime = calculateWaitingTime(rho, E_Squared, Sk);
    
    // Non-preemptive performance for comparison
    double nonPreemptiveWaitingTime = calculateNonPreemptiveWaitingTime(rho, E_Squared);
    
    return preemptiveMeanResidenceTime + preemptiveWaitingTime < nonPreemptiveWaitingTime;
}

public double calculateNonPreemptiveWaitingTime(List<Double> rho, List<Double> E_Squared) {
    // Similar to the waiting time calculation but without truncation
}
```
x??

---

#### Non-preemptive Priority Comparison
Background context: In a non-preemptive priority system, the excess due to job size variability is contributed by all classes. This means that high-priority jobs see more of this variability than in preemptive systems.

:p How does the non-preemptive priority system differ from the preemptive system in terms of job size variability?
??x
In a non-preemptive priority system, the excess due to job size variability is contributed by all classes. This means that high-priority jobs see more variability compared to preemptive systems, where only lower-priority jobs affect their service times.

```java
// Pseudocode for calculating non-preemptive performance
public double calculateNonPreemptivePerformance(double E_Squared_k) {
    int k = rho.size() + 1;
    double sumRho = 0;
    for (int i = 1; i <= k; i++) {
        sumRho += rho.get(i-1);
    }
    
    return (rho.get(k-1) * E_Squared_k) / (2 * E_Sk * (1 - sumRho)) * (1.0 / (1 - sumRho));
}
```
x??

---

#### PSJF Mean Response Time Analysis (Approach 1)
Background context: The mean response time of the Preemptive-Shortest-Job-First (PSJF) policy can be analyzed by utilizing results for scheduling with preemptive priority classes, where job size defines its class. The formula involves taking limits as the number of classes approaches infinity to derive a specific expression.

Relevant formulas:
\[ E[T(x)]_{\text{PSJF}} = \frac{x}{1 - \rho_x + \lambda \int_0^x tf(t)t^2 dt (1 - \rho_x)^2} \]
where \( f(t) \) is the probability density function of job size, and \( \rho_x \) is defined as:
\[ \rho_x = \frac{\lambda}{\int_0^x t f(t) dt} \]

:p How do we derive the mean response time for PSJF using the first approach?
??x
To derive the mean response time for PSJF, we use the preemptive priority class formula and take limits as the number of classes goes to infinity. We start with the preemptive priority response time for a job of size \( k \):
\[ E[T(k)]_{\text{P-Priority}} = \frac{E[S_k]}{1 - \sum_{i=1}^{k-1}\rho_i + \lambda^2 \sum_{i=1}^k p_i E[S_i]^2 (1 - \sum_{i=1}^{k-1}\rho_i)(1 - \sum_{i=1}^k\rho_i)} \]
By imagining an infinite number of classes and substituting the job size \( x \) for all jobs, we get:
\[ E[T(x)]_{\text{PSJF}} = \frac{x}{1 - \rho_x + \lambda \int_0^x t f(t)t^2 dt (1 - \rho_x)^2} \]
where \( \rho_x = \frac{\lambda}{\int_0^x t f(t) dt} \).

---
#### PSJF Mean Response Time Analysis (Approach 2)
Background context: The mean response time for PSJF can also be derived by breaking down the response time into waiting time and residence time. We then analyze each component separately.

:p What is \( E[\text{Res}(x)]_{\text{PSJF}} \)?
??x
The expected residence time, \( E[\text{Res}(x)]_{\text{PSJF}} \), for a job of size \( x \) in the PSJF policy represents the duration of a busy period started by this job. Only jobs of size less than or equal to \( x \) are relevant:
\[ E[\text{Res}(x)]_{\text{PSJF}} = x (1 - \rho_x) \]
where \( \rho_x \) is the fraction of the load made up by jobs of size less than \( x \).

---
#### PSJF Mean Waiting Time Analysis
Background context: The waiting time in PSJF can be viewed as the length of a busy period started by a phantom job with work equivalent to the system's relevant portion.

:p Can we think of \( E[\text{Wait}(x)]_{\text{PSJF}} \) as a busy period duration?
??x
Yes, when a job of size \( x \) arrives, it sees some relevant work. The waiting time \( E[\text{Wait}(x)]_{\text{PSJF}} \) can be thought of as the length of a busy period started by a phantom job with this relevant work. Only jobs of size less than or equal to \( x \) are considered in this busy period:
\[ E[\text{Wait}(x)]_{\text{PSJF}} = E[W_x]_{\text{PSJF}} (1 - \rho_x) \]
where \( W_x \) is the work seen by a job of size \( x \).

---
#### Work in System for PSJF
Background context: The work seen by a job of size \( x \), denoted as \( E[W_x]_{\text{PSJF}} \), can be viewed as the time-in-queue in an FCFS system where only jobs of size less than or equal to \( x \) are allowed.

:p What is \( E[W_x]_{\text{PSJF}} \)?
??x
The expected work seen by a job of size \( x \), \( E[W_x]_{\text{PSJF}} \), in the PSJF policy, can be interpreted as:
\[ E[W_x]_{\text{PSJF}} = \lambda F(x) E[S^2_x] / 2 (1 - \rho_x) \]
where \( S_x \) is the size of a job with size less than or equal to \( x \), and its density function is given by \( f(t) F(x) \). Here, \( F(x) \) is the cumulative distribution function up to size \( x \).

---
#### Summary of PSJF Analysis
Background context: The PSJF policy analysis involves understanding how jobs are scheduled based on their original sizes and breaking down response time into waiting and residence times.

:p Summarize the key points about analyzing the mean response time for PSJF.
??x
Key points in analyzing the mean response time for PSJF include:
1. **Approach 1**: Using preemptive priority class formulas with limits as the number of classes approaches infinity.
2. **Approach 2**: Breaking down response time into waiting and residence times, understanding that \( E[\text{Res}(x)]_{\text{PSJF}} = x (1 - \rho_x) \).
3. **Waiting Time**: Interpreting it as a busy period duration for relevant work.
4. **Work in System**: Calculating the expected work seen by a job of size \( x \), which can be equated to an FCFS system's time-in-queue with only smaller jobs.

---
#### Transformer Glasses Concept
Background context: The "transformer glasses" concept helps visualize how PSJF makes jobs larger than a certain size invisible, similar to FB but in a different way.

:p Explain the transformer glasses analogy for PSJF.
??x
In PSJF, "transformer glasses" make any job larger than \( x \) invisible. This is akin to seeing only relevant work for jobs of size less than or equal to \( x \). For example:
```java
public class TransformerGlasses {
    private final double x;
    public TransformerGlasses(double x) {
        this.x = x;
    }
    
    public boolean isVisible(Job job) {
        return job.size <= x;
    }
}
```
This analogy helps in understanding the scheduling decisions made by PSJF.

#### Preemptive, Size-Based Policies (PSJF)
Background context: The given text discusses the preemptive shortest job first (PSJF) scheduling policy for an M/G/1 queue. This policy schedules jobs based on their size and allows preemption when a smaller job arrives.

Formula:
\[ E[T(x)]_{\text{PSJF}} = x \frac{1 - \rho_x}{1 - \rho_x} + \frac{\lambda}{2(1 - \rho_x)^2 \int_0^x t^2 f(t) dt} \]

This formula is similar to (32.4), indicating the expected response time for a job of size \( x \).

:p What is the formula for the expected response time \( E[T(x)]_{\text{PSJF}} \)?
??x
The formula given in the text describes the expected response time as:
\[ E[T(x)]_{\text{PSJF}} = x \frac{1 - \rho_x}{1 - \rho_x} + \frac{\lambda}{2(1 - \rho_x)^2 \int_0^x t^2 f(t) dt} \]

Here, \( \rho_x \) is the load made up of jobs of size ≤ \( x \), and \( \lambda \) is the arrival rate. The term involving the integral accounts for the impact of job sizes on the response time.

---

#### Laplace Transform Analysis of PSJF
Background context: The text aims to derive the Laplace transform of the response time for an M/G/1/PSJF queue by conditioning on job size.

Formula:
\[ \tilde{T}(s) = \int_0^\infty \tilde{T}(x)(s) f(x) dx \]

Where \( \tilde{T}(x)(s) = \tilde{\text{Wait}}(x)(s) \cdot \tilde{\text{Res}}(x)(s) \).

:p What is the formula for the Laplace transform of the response time?
??x
The Laplace transform of the response time \( T(x) \) can be derived by integrating over all job sizes:
\[ \tilde{T}(s) = \int_0^\infty \tilde{T}(x)(s) f(x) dx \]

Where \( \tilde{T}(x)(s) \) is further decomposed into the product of the Laplace transforms of waiting time and residence time for a job of size \( x \).

---

#### Determining \( \tilde{\text{Wait}}(x)(s) \)
Background context: The waiting time \( \text{Wait}(x) \) denotes the duration of a busy period started by any job of size ≤ \( x \), and this is crucial for calculating the response time.

Formula:
\[ \tilde{\text{Wait}}(x)(s) = e^{-sx} \cdot \hat{\text{A}}_x(x / \tilde{\text{B}}_x(s)) \]

:p What is the formula for \( \tilde{\text{Wait}}(x)(s) \)?
??x
The Laplace transform of the waiting time \( \text{Wait}(x) \) is given by:
\[ \tilde{\text{Wait}}(x)(s) = e^{-sx} \cdot \hat{\text{A}}_x(x / \tilde{\text{B}}_x(s)) \]

Here, \( \hat{\text{A}}_x \) represents the number of arrivals of size ≤ \( x \) during time period \( y \), and \( \tilde{\text{B}}_x(s) \) is the Laplace transform of the busy period duration for jobs of size ≤ \( x \).

---

#### Determining \( \tilde{\text{Res}}(x)(s) \)
Background context: The residence time \( \text{Res}(x) \) denotes the total busy period started by a job of exact size \( x \), and this is essential for calculating the response time.

Formula:
\[ \tilde{\text{Res}}(x)(s) = e^{-sx} \cdot (1 - \tilde{\text{B}}_x(s)) / (\lambda_x s + 1 - \tilde{\text{B}}_x(s)) \]

:p What is the formula for \( \tilde{\text{Res}}(x)(s) \)?
??x
The Laplace transform of the residence time \( \text{Res}(x) \) is given by:
\[ \tilde{\text{Res}}(x)(s) = e^{-sx} \cdot (1 - \tilde{\text{B}}_x(s)) / (\lambda_x s + 1 - \tilde{\text{B}}_x(s)) \]

Here, \( \lambda_x \) is the arrival rate of jobs of size ≤ \( x \), and \( \tilde{\text{B}}_x(s) \) is the Laplace transform of the busy period duration for jobs of size ≤ \( x \).

---

#### Determining \( \tilde{\text{W}}_x(s) \)
Background context: The work in system \( W_x \) made up by jobs of size ≤ \( x \) is crucial for calculating the waiting time.

Formula:
\[ \tilde{\text{W}}_x(s) = (1 - \rho_x) s / (\lambda_x \tilde{\text{S}}_x(s) - \lambda_x + s) \]

:p What is the formula for \( \tilde{\text{W}}_x(s) \)?
??x
The Laplace transform of the work in system \( W_x \), which consists of jobs of size ≤ \( x \), can be calculated as:
\[ \tilde{\text{W}}_x(s) = (1 - \rho_x) s / (\lambda_x \tilde{\text{S}}_x(s) - \lambda_x + s) \]

Here, \( \rho_x \) is the load made up of jobs of size ≤ \( x \), and \( \tilde{\text{S}}_x(s) \) is the Laplace transform of the arbitrary job size distribution.

---

