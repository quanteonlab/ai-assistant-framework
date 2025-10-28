# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 50)

**Starting Chapter:** 32.5 Exercises

---

#### PSJF Scheduling Policy
Background context: The provided text discusses the response time calculation for jobs under a Preemptive Shortest Job First (PSJF) scheduling policy, combining equations from the previous sections. It mentions that the response time \( \tilde{T}(x)_{\text{PSJF}}(s) \) is derived using the Laplace transform of wait time and service time for jobs of size \( x \).

:p What is the formula for the Laplace transform of response time under PSJF scheduling?
??x
The formula for the Laplace transform of response time under PSJF scheduling is given by:
\[
\tilde{T}(x)_{\text{PSJF}}(s) = \frac{\tilde{W}(x)(s)}{\tilde{R}(x)(s)} = \left(1 - \rho_x\right)\left(s + \lambda x - \frac{\lambda x}{\tilde{B}_x(s)}\right)e^{-x(s + \lambda x - \frac{\lambda x}{\tilde{B}_x(s)})} / \left(s + \lambda x - \frac{\lambda x}{\tilde{S}_x}\right)
\]
where:
- \( \tilde{W}(x)(s) \): Laplace transform of wait time.
- \( \tilde{R}(x)(s) \): Laplace transform of service time.
- \( \rho_x \): Utilization factor for jobs of size \( x \).
- \( \lambda \): Arrival rate.

This formula combines the concepts from equations (32.9), (32.8), and (32.7).

x??

---
#### Warmup: Preemptive Priority Queue
Background context: The exercise focuses on an M/M/1 queue with preemptive priority classes, where jobs arrive in different classes with rates \( \lambda_i \) and have exponential job sizes.

:p What is the simple expression for the mean response time of class k?
??x
For an M/M/1 queue with preemptive priority classes and jobs arriving at rate \( \lambda_k \), the mean response time can be derived using Little's Law:
\[
E[T] = E[N] / \mu
\]
where \( E[N] \) is the mean number of jobs in the system, and \( \mu \) is the service rate. The exact expression for \( E[T_k] \), the mean response time for class k, depends on the specific job sizes and arrival rates but can be simplified by considering the preemptive priority nature.

x??

---
#### cμ-Rule
Background context: The cμ-rule is a scheduling policy that prioritizes jobs based on their holding cost \( c_i \) multiplied by their mean service time \( \mu_i \). The rule states that this policy is optimal for minimizing operational costs under certain conditions.

:p What does the cμ-Rule say about mean response time when all costs are equal?
??x
When all costs are equal (\( c_i = c \)), the cμ-rule simplifies to a standard shortest job first (SJF) policy. Therefore, the mean response time for any class under this condition would be optimized by always scheduling the smallest jobs next.

x??

---
#### Work Sum Inequality
Background context: The cμ-Rule involves proving that the cμ-policy minimizes the total work in the system and translates this into a cost minimization problem. A key step is showing an inequality involving the expected total work for different policies.

:p Explain why the following work sum inequality holds for all policies \( \pi \):
\[
\sum_{i=1}^{j} E[W_{cμ_i}] \leq \sum_{i=1}^{j} E[W_{π_i}]
\]
??x
The inequality states that for any policy \( π \), the expected total work under the cμ-policy is less than or equal to the expected total work under policy \( π \). This can be explained through sample-path arguments, considering the nature of the cμ-policy which gives priority to lower cost jobs first.

x??

---
#### Identity Proof
Background context: The cμ-Rule involves proving an identity that helps in translating the work results into a cost minimization problem. The identity is used to show that the cost under the cμ-policy is less than or equal to the cost under any other policy.

:p Prove the following simple identity:
\[
\sum_{i=1}^{n} a_i b_i = \sum_{i=1}^{n} (a_i - a_{i+1}) \sum_{j=1}^{i} b_j
\]
??x
The proof involves manipulating the summation to break it down into smaller parts and using telescoping series properties:
\[
\sum_{i=1}^{n} a_i b_i = \sum_{i=1}^{n} (a_i - a_{i+1}) \sum_{j=1}^{i} b_j
\]
This can be derived by expanding and simplifying the right-hand side:
\[
\sum_{i=1}^{n} (a_i - a_{i+1}) \sum_{j=1}^{i} b_j = \sum_{i=1}^{n-1} (a_i - a_{i+1}) \left(\sum_{j=1}^{i} b_j + \sum_{j=i+1}^{n} b_j\right) + a_n \sum_{j=1}^{n} b_j
\]
Simplifying this, we get:
\[
a_1 b_1 + (a_2 - a_3) (b_1 + b_2) + \cdots + (a_{n-1} - a_n) (b_1 + \cdots + b_{n-1}) + a_n b_n
\]
Which simplifies to:
\[
a_1 b_1 + a_2 b_2 + \cdots + a_n b_n = \sum_{i=1}^{n} a_i b_i
\]

x??

---
#### Cost Minimization Proof
Background context: The final step in proving the cμ-rule involves translating the work results into cost minimization. This requires showing that the operational costs under the cμ-policy are less than or equal to those under any other policy.

:p Prove that \( \text{Cost}(cμ) = \sum_{i=1}^{n} c_i E[N_{cμ_i}] \leq \sum_{i=1}^{n} c_i E[N_{π_i}] = \text{Cost}(π) \) for all policies \( π \).
??x
To prove this, we first translate the number of jobs in the system into work:
\[
E[W_{π_i}] = E[N_{π_i}] / μ_i \implies E[N_{π_i}] = μ_i E[W_{π_i}]
\]
Then, using equation (32.11):
\[
\sum_{i=1}^{n} c_i E[N_{cμ_i}] = \sum_{i=1}^{n} c_i \left(\mu_i - \sum_{j=i+1}^{n} a_j b_j\right) = \sum_{i=1}^{n} c_i \left(\mu_i - \sum_{j=1}^{i-1} (c_j - c_{j+1}) \sum_{k=j+1}^{i} b_k\right)
\]
Applying the inequality from part (b):
\[
\sum_{i=1}^{n} c_i E[N_{cμ_i}] \leq \sum_{i=1}^{n} c_i E[N_{π_i}]
\]

x??

#### SRPT Overview
SRPT stands for Shortest-Remaining-Processing-Time scheduling. Under this policy, at all times the server is working on that job with the shortest remaining processing time. This policy is preemptive, meaning a new arrival will preempt the current job serving if the new arrival has a shorter remaining processing time.
:p Can you explain SRPT in your own words?
??x
SRPT selects the job to be served based on its remaining processing time. When a new job arrives, it can interrupt an ongoing job if the new job's remaining processing time is less than that of the current job. This makes sure shorter jobs are given priority as they age.
x??

---
#### Response Time Analysis in M/G/1 Setting
The response time for SRPT in the M/G/1 setting includes both waiting time and residence time. The formula for the expected response time \( E[T(x)] \) is:
\[ E[T(x)] = E[Wait(x)] + E[Res(x)] \]
where
\[ E[Wait(x)] = \frac{\lambda^2}{\int_0^x t^2 f(t) dt} + \frac{\lambda^2 x^2 (1 - F(x))}{(1 - \rho_x)^2} \]
and 
\[ E[Res(x)] = \int_0^x \frac{dt}{1 - \rho_t}, \quad \text{with } \rho_x = \frac{\lambda}{\int_0^x t f(t) dt}. \]

:p What is the formula for \( E[T(x)] \)?
??x
The expected response time \( E[T(x)] \) in SRPT is given by:
\[ E[T(x)] = E[Wait(x)] + E[Res(x)], \]
where \( E[Wait(x)] \) and \( E[Res(x)] \) are the waiting time and residence time, respectively.
x??

---
#### Residence Time Understanding
The term representing mean residence time for a job of size \( x \) under SRPT is:
\[ E[Res(x)] = \int_0^x \frac{dt}{1 - \rho_t}. \]

:p Why does the residence time in SRPT increase as the job ages?
??x
In SRPT, a job’s "priority" increases over time. Therefore, once a job has started service, its effective slowdown factor should depend on its remaining service requirement \( t \) and be related to the load of all jobs with smaller sizes. As a job ages, it encounters more and smaller jobs in the system, causing it to take longer to complete.
x??

---
#### Waiting Time Analysis
The waiting time for SRPT is given by:
\[ E[Wait(x)] = \frac{\lambda^2}{\int_0^x t^2 f(t) dt} + \frac{\lambda^2 x^2 (1 - F(x))}{(1 - \rho_x)^2}. \]

:p What does the second term in \( E[Wait(x)] \) represent?
??x
The second term, \( \frac{\lambda^2 x^2 (1 - F(x))}{(1 - \rho_x)^2} \), represents the contribution of jobs with sizes greater than \( x \) to the waiting time. It suggests that larger jobs contribute more significantly as they are still in the system and have a higher remaining processing time.
x??

---
#### SRPT vs PSJF
The response time for SRPT can be compared to PSJF (Shortest Job First). The waiting time expression for SRPT, when ignoring the second term:
\[ E[Wait(x)] = \frac{\lambda^2}{\int_0^x t^2 f(t) dt} + \frac{\lambda^2 x^2 (1 - F(x))}{(1 - \rho_x)^2}, \]
resembles that of PSJF, where only jobs with size \( \leq x \) contribute to the waiting time.
:p How does SRPT's waiting time compare to PSJF?
??x
SRPT’s waiting time expression is similar to PSJF but includes an additional term representing the contribution from larger jobs. This means that in SRPT, all job sizes contribute to the waiting time of a job, not just those smaller than or equal to \( x \).
x??

---
#### FB Scheduling Comparison
The numerator of \( E[Wait(x)]_{SRPT} \) is similar to \( E[S^2_x] \), used in FB (Fairness-Based) scheduling. The formula for the numerator in SRPT waiting time:
\[ \lambda^2 E[S^2_x], \]
is analogous to FB's approach.
:p How does SRPT’s waiting time expression compare to FB?
??x
SRPT’s waiting time expression has a similar numerator structure to FB, where \( \lambda^2 E[S^2_x] \) represents the expected contribution from all job sizes. However, the denominator involves \( \rho_x \), as in PSJF, because only jobs of size \( \leq x \) are allowed to enter the busy period.
x??

---

#### SRPT Waiting Time Derivation Overview
Background context: The text discusses the precise derivation of the SRPT (Shortest Remaining Processing Time) waiting time, focusing on how work found by an arrival affects its waiting time. This involves understanding the work seen by an arrival before it starts running and the busy periods associated with this work.
:p What is \( W_{SRPT}^x \) in the context of SRPT?
??x
\( W_{SRPT}^x \) represents the work found in the system that is "relevant" to an arriving job of size \( x \), meaning the work that runs before the arrival of size \( x \) starts running.
x??

---
#### Work Found by Arrival of Size x (WSRP Tx)
Background context: The SRPT algorithm considers two types of jobs when determining the work found by an arrival of size \( x \): type a and type b. Type a includes jobs that are in the system with original size \( \leq x \), while type b includes jobs originally larger than \( x \) but now reduced to size \( \leq x \).
:p How many type b jobs can there be?
??x
There can be at most one job of type b. Furthermore, no more type b jobs will enter the system until the arrival of size \( x \) has left the system entirely.
x??

---
#### Work Type a and b Analysis
Background context: The analysis for work made up of both type a and type b jobs involves breaking down the queueing system into two parts: the queue part (type a jobs only) and the server part (type a and type b jobs). This allows treating type b jobs as having priority over type a jobs.
:p Why does the analysis consider the queue and server parts separately?
??x
The queue and server parts are considered separately to simplify the analysis. By treating type b jobs as arriving directly into the server, we ensure they never enter the queue part, allowing us to use FCFS principles for type a jobs in the queue.
x??

---
#### Tagged Job Argument for Type a Arrivals
Background context: To determine the mean delay \( E[T_Q] \) for a type a arrival, a tagged job argument is used. This involves calculating the expected number of type a jobs in the queue and their service time.
:p How is the mean delay \( E[T_Q] \) calculated?
??x
The mean delay \( E[T_Q] \) for a type a arrival is calculated by considering the number of type a jobs in the queue, denoted as \( N_Q \), and their expected remaining service times. The formula uses the probability that an arriving job finds a busy server, which is \( \rho_x \), and the expected excess service time.
x??

---
#### Detailed Calculation for Mean Delay
Background context: Using a tagged-job argument, we can calculate the mean delay \( E[T_Q] \) by considering the number of type a jobs in the queue and their expected remaining service times. This involves integrating job size distributions to find the fraction of time the server is busy.
:p What formula represents the mean delay for a type a arrival?
??x
The mean delay \( E[T_Q] \) can be calculated using the formula:
\[ E[T_Q] = \frac{\lambda E[S_x]}{1 - \rho_x} \cdot \frac{E[S^2_x]}{2E[S_x]} \]
Where \( \lambda \) is the arrival rate, \( E[S_x] \) is the expected job size, and \( \rho_x = \lambda E[S_x] \).
x??

---
#### Example Code for Calculation
Background context: The calculation involves integrating over the job size distribution to find the fraction of time the server is busy.
:p Provide an example code snippet in Java to calculate \( E[T_Q] \)?
??x
```java
public class SRPTWaitingTime {
    public static double calculateE_TQ(double lambda, double[] sDistribution) {
        double rho = lambda * expectedJobSize(sDistribution);
        double E_S2_x = 0;
        
        for (int i = 0; i < sDistribution.length; i++) {
            E_S2_x += sDistribution[i] * Math.pow(i, 2);
        }
        
        return (lambda * expectedJobSize(sDistribution) / (1 - rho)) * (E_S2_x / (2 * expectedJobSize(sDistribution)));
    }

    private static double expectedJobSize(double[] sDistribution) {
        // Calculate the expected job size from the distribution
        return 0;
    }
}
```
x??

---
#### Conclusion on SRPT Waiting Time
Background context: The derivation of the SRPT waiting time involves understanding how work found by an arrival affects its waiting time, broken down into type a and b jobs. By treating type b jobs as having priority over type a jobs, we can use FCFS principles for simpler analysis.
:p What is the key takeaway from this section?
??x
The key takeaway is that the SRPT waiting time involves analyzing both types of jobs (a and b) found by an arrival, where type b jobs are treated with high priority to ensure they never enter the queue part. This allows using FCFS principles for simpler analysis while accurately calculating the waiting time.
x??

---

