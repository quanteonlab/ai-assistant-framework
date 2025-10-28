# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 46)

**Starting Chapter:** 29.2 Readings. Chapter 30 Preemptive Non-Size-Based Policies

---

#### LCFS Waiting Time Transform
Background context explaining the transformation of waiting times under LCFS. The provided equation for \( \widetilde{TLCFS}_Q(s) \) is derived from the system's state and arrival rate parameters, showing how the expected queueing time can be expressed in terms of system idle and busy states.

:p What is the expression for the transform \( \widetilde{TLCFS}_Q(s) \) under LCFS?
??x
The expression for the transform \( \widetilde{TLCFS}_Q(s) \) under LCFS is given by:
\[ \widetilde{TLCFS}_Q(s) = (1-\rho) \cdot \widetilde{TLCFS}_Q(s|\text{idle}) + \rho \cdot \widetilde{TLCFS}_Q(s|\text{busy}) \]
Where \( \rho \) is the traffic intensity, and \( s \), \( \lambda \), and \( B(s) \) are system parameters related to service rate and busy period distribution. The transform helps in deriving moments of waiting times.

x??

---

#### Mean Waiting Time under LCFS
Explanation of how to derive the mean queueing time for LCFS without using transforms, by conditioning on whether an arrival finds the system idle or busy.

:p How can we derive the mean waiting time \( E[TQ]_{\text{LCFS}} \) for LCFS by conditioning?
??x
To derive the mean waiting time \( E[TQ]_{\text{LCFS}} \), we condition on whether an arrival finds the system idle or busy:
- If the arrival finds the system idle, its waiting time is 0.
- If the arrival finds the system busy, it waits for one service completion plus the remaining part of the busy period.

The formula can be written as:
\[ E[TQ]_{\text{LCFS}} = (1-\rho) \cdot 0 + \rho \left( \frac{\lambda}{s} + \frac{1 - B(s)}{\lambda s} \right) \]
Where \( \rho \), \( s \), and \( B(s) \) are defined as the traffic intensity, service rate, and busy period distribution respectively.

x??

---

#### Second Moment of Waiting Time under LCFS
Explanation of deriving the second moment of waiting time for LCFS and comparing it with FCFS.

:p How is the second moment of waiting time \( E[T^2_Q]_{\text{LCFS}} \) derived, and what is its relationship to FCFS?
??x
The second moment of waiting time under LCFS is given by:
\[ E[T^2_Q]_{\text{LCFS}} = \frac{\lambda E[S^3]}{3(1-\rho)^2} + \frac{(\lambda E[S^2])^2}{2(1-\rho)^3} \]
This expression can be compared with the second moment of waiting time under FCFS:
\[ E[T^2_Q]_{\text{FCFS}} = \frac{\lambda E[S^3]}{3(1-\rho)} + \frac{(\lambda E[S^2])^2}{2(1-\rho)^2} \]
The relationship between the two is that:
\[ E[T^2_Q]_{\text{LCFS}} = E[T^2_Q]_{\text{FCFS}} \cdot \frac{1}{1-\rho} \]

x??

---

#### Mean Slowdown Comparison
Explanation of how different scheduling policies compare in terms of mean slowdown for an M/G/1 queue.

:p How do FCFS, LCFS, and RANDOM scheduling policies compare in terms of mean slowdown?
??x
For an M/G/1 queue under the M/M/1 model, all three scheduling disciplines (FCFS, LCFS, and RANDOM) have the same distribution of the number of jobs in the system. However, they differ in their mean slowdown:

- **FCFS**: The mean slowdown is higher due to the nature of first-come-first-served.
- **LCFS**: The mean slowdown can be lower or similar, depending on job sizes and arrivals.
- **RANDOM**: The mean slowdown can vary but often lies between FCFS and LCFS.

The exact comparison can be proven by analyzing the service times and arrival processes under each policy. For instance, for FCFS and LCFS:
\[ E[S]_{\text{FCFS}} = E[S]_{\text{LCFS}} \]
But their slowdowns differ based on job sizes and arrival patterns.

x??

---

#### Processor-Sharing (PS) Concept
Background context: In Chapter 30, we explore three preemptive scheduling policies that do not rely on job size or priority class. One of these policies is Processor-Sharing (PS). PS allows short jobs to time-share with all other jobs in the system immediately upon arrival, which can reduce their waiting times compared to non-preemptive policies.

:p Why are short jobs not affected by long ones under PS?
??x
Short jobs benefit from immediate time-sharing with all jobs in the system. They do not have to wait for long jobs to finish before starting execution. This is because each job gets a small quantum of CPU time, and new arrivals can start sharing the CPU immediately.
The processor shares its resources among all currently executing jobs, ensuring that short jobs can complete their service faster.

```java
public class ProcessorSharing {
    private double quantum; // Small time slice for scheduling

    public void scheduleJobs(double[] jobSizes) {
        // Simulate time-sharing with a small quantum
        for (double size : jobSizes) {
            // Each job gets a quantum of CPU time to run
            while (size > 0 && !allJobsFinished()) {
                size -= getQuantum();
            }
        }
    }

    private double getQuantum() {
        return 0.1; // Example small quantum value
    }

    private boolean allJobsFinished() {
        // Logic to check if all jobs are finished
        return true;
    }
}
```
x??

---

#### Motivation Behind PS
Background context: The motivation for Processor-Sharing is to address the issue of high mean response times in non-preemptive policies due to variability in job sizes. Non-preemptive policies like FCFS can lead to short jobs being delayed significantly behind long jobs.

:p Why does PS not suffer from high mean response times caused by variable job sizes?
??x
PS mitigates this issue by allowing short jobs to immediately share the CPU with other jobs upon arrival, regardless of their size. This means that short jobs do not have to wait for long jobs to complete, reducing their average waiting time and overall delay.

In contrast, in non-preemptive policies like FCFS, a short job must wait until all longer jobs before it are completed, which can lead to very high mean response times when there is significant variability in job sizes.

```java
public class NonPreemptiveFCFS {
    public double calculateMeanResponseTime(double[] jobSizes) {
        // Calculate the mean response time assuming FCFS scheduling
        return (jobSizes.length * 1.5); // Example calculation, actual formula depends on E[S2]
    }
}
```
x??

---

#### Comparison with FCFS in PS
Background context: While Processor-Sharing can be beneficial for short jobs by allowing immediate access to the CPU, it is not universally better than First-Come-First-Served (FCFS) scheduling. There are specific arrival sequences where FCFS outperforms PS.

:p Provide an example of an arrival sequence where PS performs worse than FCFS in terms of mean response time and mean slowdown.
??x
Consider two jobs both arriving at time 0, with each job having a size of 1:
- For FCFS: 
  - \(E[T]_{FCFS} = 1.5\)
  - \(E[\text{Slowdown}]_{FCFS} = 1.5\)

- For PS:
  - \(E[T]_{PS} = 2\)
  - \(E[\text{Slowdown}]_{PS} = 2\)

In this example, both the mean response time and mean slowdown are higher for PS compared to FCFS.

```java
public class ExampleArrivalSequence {
    public void compareSchedulingPolicies() {
        double fcfsMeanResponseTime = 1.5; // Example value
        double psMeanResponseTime = 2;      // Example value

        if (psMeanResponseTime > fcfsMeanResponseTime) {
            System.out.println("PS performs worse than FCFS.");
        } else {
            System.out.println("PS performs better than or equal to FCFS.");
        }
    }
}
```
x??

---

#### Stochastic Setting Outperformance
Background context: Even though PS does not outperform FCFS on every arrival sequence, we need to determine if M/G/1/PS can still provide an advantage over M/G/1/FCFS in a stochastic setting.

:p Can we claim that M/G/1/PS outperforms M/G/1/FCFS with respect to expected response time in a stochastic setting?
??x
In a stochastic setting, it is not accurate to definitively state that M/G/1/PS will always outperform M/G/1/FCFS. The performance of PS versus FCFS can vary depending on the characteristics of job sizes and arrival patterns.

However, PS tends to perform better for systems with highly variable job sizes because short jobs can immediately start using the CPU without waiting behind long jobs. In contrast, FCFS might result in significant delays for short jobs if many longer jobs are already in the queue.

```java
public class StochasticSetting {
    public boolean isMg1PsBetterThanFcfs(double[] jobSizes) {
        double psResponseTime = calculatePSResponseTime(jobSizes);
        double fcfsResponseTime = calculateFCFSResponseTime(jobSizes);

        return (psResponseTime < fcfsResponseTime);
    }

    private double calculatePSResponseTime(double[] jobSizes) {
        // PS response time calculation logic
        return 2; // Example value
    }

    private double calculateFCFSResponseTime(double[] jobSizes) {
        // FCFS response time calculation logic
        return 1.5; // Example value
    }
}
```
x??

#### Definition of Job Age
The age of a job is defined as the total service it has received so far. By definition, \(0 \leq \text{age}(j) \leq \text{size}(j)\), where \(\text{age}(j)\) denotes the age of job \(j\) and \(\text{size}(j)\) denotes the (original) size of job \(j\).
:p What is the definition of a job's age in an M/G/1/PS system?
??x
The age of a job is the total service it has received so far. This means that for any given job, its age will be between 0 and its original job size.
x??

---

#### Distribution of Job Ages under FCFS
Under First-Come-First-Served (FCFS) policy, jobs in queue all have an age of 0, while the job currently in service has an age distributed according to the equilibrium distribution. The probability density function for this is given by:
\[ f_e(x) = \frac{F(x)}{E[S]} \]
Where \(S\) denotes job size, and \(f(·)\) is the job size p.d.f.
:p How are the ages of jobs distributed under FCFS in an M/M/1/FCFS system?
??x
Under FCFS, all jobs in queue have age 0. The job currently being served has its age (and any excess service time) distributed according to the equilibrium distribution with probability density function \( f_e(x) = \frac{F(x)}{E[S]} \).
x??

---

#### Distribution of Job Ages under PS
Under Processor-Sharing (PS), all jobs are worked on simultaneously. An arrival sees every job through an Inspection Paradox, which means that all jobs in the system have i.i.d. ages distributed according to the equilibrium distribution.
:p How are the ages of jobs distributed under PS in an M/G/1/PS system?
??x
Under PS, all jobs have i.i.d. ages and these ages follow the equilibrium distribution defined by \( f_e(x) = \frac{F(x)}{E[S]} \).
x??

---

#### Response Time and Slowdown for PS System
In an M/G/1/PS system, every job has the same expected slowdown given as:
\[ E[T(x)]_{M/G/1/PS} = \frac{x}{1-\rho} \]
where \( x \) is the size of the job and \( \rho \) is the utilization factor.
:p What is the expected response time for a job in an M/G/1/PS system as a function of its size?
??x
The expected response time for a job in an M/G/1/PS system, given its size \( x \), is:
\[ E[T(x)]_{M/G/1/PS} = \frac{x}{1-\rho} \]
This means that the slowdown (response time divided by job size) is constant and equal to \( \frac{1}{1-\rho} \).
x??

---

#### Departure Process in M/G/1/PS System
The departure process from an M/G/1/PS system is a Poisson process with rate \( \lambda \). This implies that the inter-departure times are exponentially distributed.
:p What can we say about the departure process of jobs in an M/G/1/PS system?
??x
The departure process in an M/G/1/PS system is a Poisson process with rate \( \lambda \). Therefore, the inter-departure times between any two successive departures are exponentially distributed.
x??

---

#### Equilibrium Distribution in M/G/1/PS and M/M/1/FCFS
The distribution of the number of jobs in an M/G/1/PS system is identical to that in an M/M/1/FCFS system. This means that both systems have the same mean response time and mean number of jobs.
:p Why can we say that the M/G/1/PS and M/M/1/FCFS systems are equivalent regarding their performance?
??x
The M/G/1/PS and M/M/1/FCFS systems are equivalent in terms of performance because they have identical distributions for the number of jobs in the system. This implies that both systems share the same mean response time and mean number of jobs.
x??

---

#### Squared Coefficient of Variation and PS
The squared coefficient of variation \( C^2_G \) is greater than 1 if the M/G/1/PS system outperforms the M/G/1/FCFS system. This means that PS is better when job size variability is high.
:p Under what condition does the M/G/1/PS system perform better in expectation compared to the M/G/1/FCFS system?
??x
The M/G/1/PS system performs better in expectation than the M/G/1/FCFS system exactly when the squared coefficient of variation \( C^2_G \) is greater than 1. This indicates that PS is more effective for systems with high job size variability.
x??

---

#### Expected Number of Jobs in System with Size x to x+h

Background context: The question asks about expressing the expected number of jobs in a system with original sizes between \(x\) and \(x + h\). It highlights that the original job size distribution is given by \(f(\cdot)\), but the job sizes in the system have a different probability density function (pdf) due to preemptive service.

:p Can we express the expected number of jobs in the system with size between \(x\) and \(x + h\) as \(E[N] f(x) h + o(h)\)?
??x
No, because the job sizes in the system are processed differently compared to their original arrival sizes. The pdf of job sizes in the system (\(fsys(\cdot)\)) is not necessarily the same as \(f(\cdot)\), especially considering that small jobs are finished more quickly.

To find \(fsys(w)\), we condition on the job’s age:
\[
fsys(w) = \int_0^w fsys(w | \text{job has age } x) \cdot P\{\text{job has age } x\} dx
= \int_0^w fsys(w | \text{job has age } x) \cdot f_e(x) dx,
\]
where \(f_e(x)\) is the exponential distribution for job arrival times. This simplifies to:
\[
fsys(w) = f(w) \cdot \frac{w}{E[S]},
\]
where \(E[S]\) is the mean service time.

Using this, we can find the expected number of jobs in the system with sizes between \(x\) and \(x + h\):
\[
E[\text{Number of jobs in system with size } (x, x + h)] = E[N] \cdot fsys(x) \cdot h + o(h),
= \frac{\rho}{1 - \rho} \cdot \frac{x \cdot f(x)}{E[S]} \cdot h + o(h).
\]
The rate of arrivals of jobs into the system with size between \(x\) and \(x + h\) is:
\[
E[\text{Rate of arrivals of jobs with size } (x, x + h)] = \lambda \cdot f(x) \cdot h + o(h).
\]

Applying Little’s Law, we find the expected time in the system for jobs with original sizes between \(x\) and \(x + h\):
\[
E[T(x)] = \frac{\rho}{1 - \rho} \cdot x \cdot f(x),
\]
which completes the proof of Theorem 30.4.

x??

#### Intuition Behind (30.2)

Background context: Equation (30.2) shows how \(fsys(w)\), the pdf for job sizes in the system, relates to the original job size distribution \(f(w)\).

:p Explain the intuition behind equation (30.2).
??x
In equation (30.2):
\[
fsys(w) = f(w) \cdot \frac{w}{E[S]}.
\]
The factor that multiplies \(f(w)\) is \(\frac{w}{E[S]}\), which is greater than 1 for large jobs and less than 1 for small jobs. This indicates that the probability of finding a job with size \(w\) in the system increases as \(w\) gets larger, compared to what it would be under the original distribution.

For small values of \(w\), \(\frac{w}{E[S]}\) is small, reducing the likelihood of having such jobs. Conversely, for large values of \(w\), this factor becomes significant, making larger jobs more likely in the system.

x??

#### Response Time and Busy Period Length under M/G/1/PS

Background context: The text discusses how the expected response time for a job of size \(x\) in an M/G/1/PS queue is equal to the mean length of a busy period started by a job of size \(x\), both of which are given by \(\frac{x}{1 - \rho}\).

:p What is the intuition behind the expression \(\frac{x}{1 - \rho}\) for response time under M/G/1/PS?
??x
The expression \(\frac{x}{1 - \rho}\) represents the expected time a job of size \(x\) spends in the system. The factor \(1 - \rho\) is the utilization factor, which captures how busy the server is on average.

When an arrival sees \(\frac{\rho}{1 - \rho}\) jobs in the system (which is the steady-state number of jobs in the system), each job contributes to slowing down new arrivals. The total slowdown for a job of size \(x\) is proportional to the number of jobs in the system plus one, i.e., \(\frac{\rho}{1 - \rho} + 1 = \frac{1}{1 - \rho}\).

Thus, any arrival should take:
\[
E[T(x)] = x \cdot \frac{1}{1 - \rho}
\]
time to leave the system.

x??

#### Fairness of Processor-Sharing Scheduling

Background context: The text explains that under processor-sharing (PS) scheduling, all jobs experience the same slowdown regardless of their size. This is in contrast to non-preemptive and non-size-based policies where smaller jobs might have higher mean slowdown compared to larger jobs.

:p Why is processor-sharing referred to as fair scheduling?
??x
Processor-sharing (PS) is called fair scheduling because it ensures that all jobs experience the same relative slowdown, independent of their size. This is in contrast to other policies like Shortest Remaining Time First (SRPT), where smaller jobs might be starved and thus have higher mean slowdown compared to larger jobs.

By ensuring equal slowdown for all jobs, PS promotes fairness among different job sizes. The metric used to evaluate this fairness involves comparing the mean slowdown of large jobs under SRPT with that under PS, which is always lower or equal due to the fair distribution of service.

x??

#### M/G/1/PS Queue Analysis

Background context: The text discusses the analysis of an M/G/1/PS queue, highlighting its simplicity and beauty in terms of expected slowdown for jobs of size \(x\).

:p What are some other expressions that also have the form \(\frac{x}{1 - \rho}\)?
??x
The expression \(\frac{x}{1 - \rho}\) appears in multiple contexts related to the M/G/1/PS queue. Specifically, it represents both the expected response time for a job of size \(x\) and the mean length of a busy period started by a job of size \(x\).

For instance:
- The expected response time for a job of size \(x\) is \(\frac{x}{1 - \rho}\).
- The expected length of a busy period initiated by a job of size \(x\) is also \(\frac{x}{1 - \rho}\).

These two concepts are related but not identical, as the busy period can be longer than the response time due to additional factors like job arrivals during the busy period.

x??

#### Transform Equation for M/G/1/FCFS

Background context: The text mentions a transform equation for waiting time in the M/G/1/FCFS queue, which is given by:
\[
\widetilde{T}_{Q, \text{FCFS}}(s) = \frac{1 - \rho}{1 - \rho/s} \cdot \widetilde{S}(s),
\]
where \(\widetilde{S}(s)\) is the Laplace transform of the service time.

:p What is the form of the waiting time (delay) in the M/G/1/FCFS queue?
??x
The waiting time (delay) in the M/G/1/FCFS queue, when transformed using the Laplace domain, is given by:
\[
\widetilde{T}_{Q, \text{FCFS}}(s) = \frac{1 - \rho}{1 - \rho/s} \cdot \widetilde{S}(s).
\]
This equation captures how the waiting time varies with the service time and the utilization factor \(\rho\).

x??

---

#### Expressing (30.3) as a Summation
Background context: The expression (30.3) is being converted into a summation form to simplify its analysis, particularly for the M/G/1/FCFS queue.

:p How can we express (30.3) as a sum?
??x
We can express (30.3) as:
\[
/tildewiderTQFCFS(s)=( 1−ρ)∞/\sum_{k=0}/parenleftBig \frac{ρ}{/tildewiderSe(s)}/parenrightBig^k =∞/\sum_{k=0}(1−ρ)\left(\frac{ρ}{/tildewiderSe(s)}\right)^k.
\]
This transformation helps in understanding the delay transform more clearly and facilitates further analysis.

---
#### Meaning of /parenleftBig /tildewiderSe(s)/parenrightBigk
Background context: The term \(\left(\frac{\tilde{S}_e(s)}{ρ}\right)^k\) represents the Laplace transform of the sum of \(k\) i.i.d. instances of \(\tilde{S}_e\), where \(\tilde{S}_e(s)\) is the Laplace transform of the service time.

:p What does /parenleftBig /tildewiderSe(s)/parenrightBigk represent?
??x
The term \(\left(\frac{\tilde{S}_e(s)}{ρ}\right)^k\) represents the Laplace transform of the sum of \(k\) i.i.d. instances of \(\tilde{S}_e\). Here, \(\tilde{S}_e(s)\) is the Laplace transform of a single service time in an M/G/1 queue.

---
#### Deriving /tildewiderTQFCFS(s)
Background context: The delay transform for the M/G/1/FCFS system can be derived using the concept of stationary work and Poisson arrivals. This involves understanding that the expected work content is equal to the expected delay experienced by a Poisson arrival.

:p How do we derive /tildewiderTQFCFS(s)?
??x
The derivation starts by noting that \(W_{\text{FCFS}}\) (the stationary work in an M/G/1/FCFS system) equals the expected delay of a Poisson arrival under FCFS. This is equivalent to the expected work content \(\tilde{W}_{\text{PS}}(s)\) in an M/G/1/PS system, since both are work-conserving.

Thus:
\[
\tildewider{TQFCFS(s)} = \tildewider{W_{\text{FCFS}}(s)} = \tildewider{W_{\text{PS}}(s)} = \sum_{k=0}^{\infty} \tildewider{W_{\text{PS}}(s|\text{arrival sees } k \text{ jobs})} \cdot P(\text{arrival sees } k \text{ jobs})
\]
\[
= \sum_{k=0}^{\infty} \left(\frac{\tilde{S}_e(s)}{\rho}\right)^k \cdot (\rho (1-\rho))^k
\]

---
#### Preemptive-LCFS Policy Description
Background context: The Preemptive-LCFS policy is a scheduling algorithm where an arriving job immediately preempts the currently running job, resuming its service only after all subsequent jobs are completed.

:p What is the Preemptive-LCFS (PLCFS) policy?
??x
The Preemptive-LCFS policy works as follows: whenever a new arrival enters the system, it preemptively interrupts the currently running job. The preempted job resumes service only when all jobs arriving after its interruption are completed.

---
#### Performance of LCFS Policy Recap
Background context: The non-preemptive LCFS (Last-Come-First-Served) policy was discussed earlier and found to have poor performance, especially with highly variable job sizes, performing identically to FCFS in many scenarios.

:p What is the performance of the (non-preemptive) LCFS policy?
??x
The non-preemptive LCFS policy had identical performance to FCFS, making it unsuitable for systems where job size variability is high. This poor performance stems from its tendency to delay smaller jobs behind larger ones, leading to suboptimal use of resources.

---
#### Theorem 30.6 - Preemptive-LCFS Performance
Background context: A theorem is presented that describes the expected completion time and slowdown for jobs in a Preemptive-LCFS (PLCFS) system. This theorem states that both measures are equal to \(\frac{x}{1-\rho}\), where \(x\) is the job size and \(\rho\) is the utilization factor.

:p What does Theorem 30.6 state about PLCFS performance?
??x
Theorem 30.6 asserts that for a Preemptive-LCFS system, the expected completion time \(E[T(x)]_{\text{PLCFS}} = \frac{x}{1-\rho}\) and the expected slowdown \(E[\text{Slowdown}(x)]_{\text{PLCFS}} = \frac{1}{1-\rho}\).

---
#### Key Observation for PLCFS
Background context: An important observation in analyzing Preemptive-LCFS is that once a job is interrupted, it will not resume service until all jobs arriving after its interruption are completed. This behavior affects the total time a job spends waiting or executing.

:p What key observation supports the analysis of PLCFS?
??x
A critical observation for PLCFS is that an interrupted job will only get back the processor when all subsequent jobs complete, which implies that the delay until resumption is equivalent to the length of a busy period in an M/G/1 queue. Thus, the expected time until the job gets back the processor is given by:
\[
E[\text{Time until job gets back processor}] = E[\text{Length of busy period}] = \frac{E[S]}{1-\rho},
\]
where \(S\) represents the service time.

---
#### Deriving Expected Time in PLCFS
Background context: To derive the expected completion time for a tagged job under Preemptive-LCFS, we leverage the fact that once interrupted, the job waits until all subsequent jobs are completed. This is analogous to the busy period length in an M/G/1 queue.

:p How do we calculate the expected time a job spends waiting or executing under PLCFS?
??x
The expected completion time for a tagged job \(E[T(x)]_{\text{PLCFS}}\) can be derived as follows:

Since the job will not resume until all subsequent jobs are completed, this is equivalent to the length of a busy period in an M/G/1 queue. The mean length of such a busy period is given by:
\[
E[\text{Time until job gets back processor}] = E[\text{Length of busy period}] = \frac{E[S]}{1-\rho}.
\]

This result holds because the service order does not affect the mean length of the busy period in an M/G/1 system, as long as it is work-conserving.

