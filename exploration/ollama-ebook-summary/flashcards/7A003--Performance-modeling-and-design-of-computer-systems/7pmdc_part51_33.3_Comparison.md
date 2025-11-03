# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 51)

**Starting Chapter:** 33.3 Comparisons with Other Policies

---

#### Comparison of SRPT and Other Policies
Background context: This section discusses the comparison between Shortest Remaining Processing Time First (SRPT) and other policies such as PSJF (Processor Sharing with Job Size First). The objective is to understand how different policies affect job waiting times and overall mean response time.

:p What does the expression for \( E[Wait(x)] \) under SRPT represent?
??x
The expression for \( E[Wait(x)] \) represents the expected wait time for a job of size \( x \) under SRPT. The formula provided relates this to WSRPT (Weighted Shortest Remaining Processing Time), and it uses the load factor \( \rho_x \).

Formula: 
\[ E[Wait(x)]_{SRPT} = E[WSRPT_x] = \frac{\lambda}{2} \int_0^x t^2 f(t) \, dt + F(x) \cdot x^2 (1-\rho_x)^2 \]

Here, \( f(t) \) is the probability density function of job sizes up to \( x \), and \( F(x) \) is the cumulative distribution function. The term involving integration accounts for the waiting time due to jobs smaller than or equal to size \( x \).

:p What are some immediate observations about SRPT's response time formula?
??x
Immediate observations include that the response time for a job of size \( x \) is not influenced by the variance of the entire job size distribution but just by the part up to size \( x \). The response time also does not depend on the entire load, but rather only jobs of size less than or equal to \( x \).

:p Why do small jobs perform well under SRPT?
??x
Small jobs (smaller \( x \)) perform well because they are prioritized quickly as they start receiving service. Under SRPT, a job gains priority as it receives more service, allowing smaller jobs to progress through the system faster.

:p How does SRPT compare with PSJF in terms of waiting time?
??x
SRPT has greater waiting time compared to PSJF due to the extra \( x^2 \) term in the numerator. In contrast, SRPT's residence time is better because a job only needs to wait for smaller jobs relative to its remaining service requirement.

:p What does Lemma 33.1 state about SRPT and FB?
??x
Lemma 33.1 states that in an M/G/1 system, for all \( x \) and for all \( \rho \), the expected response time under SRPT is less than or equal to the expected response time under Fair-Band (FB).

:p How does SRPT compare with FB?
??x
SRPT and FB are complements. In SRPT, a job gains priority as it receives more service, whereas in FB, a job has highest priority when it first enters and loses priority over time.

:p What is the key difference between SRPT and FB in terms of response time?
??x
The key difference lies in how jobs gain or lose priority. In SRPT, smaller remaining service requirements get higher priority, while in FB, larger initial job sizes have a head start but their priority diminishes over time.

:p How does SRPT perform compared to other policies overall?
??x
SRPT performs better overall due to its lower mean waiting time and better response times for small jobs. The performance is evaluated using nested integrals of the formulas provided, showing that SRPT outperforms other policies in terms of average response time.

:p What scheduling policies are compared in this section?
??x
The comparison includes Shortest Remaining Processing Time First (SRPT), Processor Sharing with Job Size First (PSJF), FCFS (First-Come-First-Served), and SJF (Shortest Job First) as well as their fair versions. The focus is on understanding how different policies affect job waiting times.

:p What type of distribution is used for the job sizes in the evaluations?
??x
A Weibull distribution with mean 1 and \( C_2 = 0.1 \) is used to evaluate the formulas and compare different scheduling policies.

:p How are the mean response time formulas evaluated?
??x
Mathematica is used to evaluate the mean response time formulas for various scheduling policies, providing a visual comparison in terms of load and variance of job sizes.

:p What does Figure 33.1 show?
??x
Figure 33.1 shows the mean response time as a function of load for different M/G/1 policies including SRPT, PSJF, FCFS, SJF, and their fair versions (PLCFSSRPTFB).

:x

---

--- 

#### Key Formulas for Response Time

:p What is the formula for \( E[T(x)]_{SRPT} \)?
??x
The formula for \( E[T(x)]_{SRPT} \) is:
\[ E[T(x)]_{SRPT} = \frac{\lambda}{2} \int_0^x t^2 f(t) \, dt + \frac{\lambda}{2} x^2 (1 - F(x)) \left( 1 - \rho_x \right)^2 + \int_0^x \frac{dt}{1 - \rho_t} \]

:p What is the formula for \( E[T(x)]_{FB} \)?
??x
The formula for \( E[T(x)]_{FB} \) is:
\[ E[T(x)]_{FB} = x (1 - \rho_x) + \frac{1}{2\lambda} \mathbb{E}\left[ S_x^2 \right] (1 - \rho_x)^2 \]

:p How does SRPT perform better in terms of mean response time?
??x
SRPT performs better because the integral term and the service completion term contribute to a lower overall waiting time compared to PSJF. The extra \( x^2(1-\rho_x) \) term ensures that smaller jobs are favored, reducing their waiting time.

:p How is the performance of SRPT compared with PSJF quantitatively?
??x
The comparison between SRPT and PSJF shows that SRPT has a lower mean response time due to its lower integral term. The extra \( x^2 \) term in the numerator ensures that SRPT waits for fewer smaller jobs, leading to better performance.

:p What is the benefit of SRPT over FB?
??x
SRPT beats FB on every job size \( x \), as shown by Lemma 33.1. This is because SRPT has a lower mean waiting time and lower mean residence time compared to FB for all job sizes.

:x

---

--- 

#### Evaluation with Mathematica

:p How is the mean response time evaluated using Mathematica?
??x
The mean response time formulas are evaluated using Mathematica, which provides numerical integration and evaluation of complex expressions. This allows a visual comparison across different policies in terms of load and variance of job sizes.

:p What does Figure 33.2 show?
??x
Figure 33.2 shows the mean response time as a function of \( C_2 \), where \( C_2 \) is a measure of the variance of the job size distribution, for different M/G/1 policies including SRPT, PSJF, FCFS, SJF, and their fair versions (PLCFSSRPTFB).

:p What type of distribution is used in Figure 33.1?
??x
In Figure 33.1, a Weibull job size distribution with mean 1 and \( C_2 = 0.1 \) is used to evaluate the formulas and compare different scheduling policies.

:x

--- 

---

#### Variance of Job Size Distribution

:p How does the variance of the job size distribution affect SRPT's performance?
??x
The variance of the job size distribution influences SRPT's performance, but only up to a certain point. The formula for \( E[T(x)]_{SRPT} \) shows that it is primarily affected by the part of the distribution up to size \( x \).

:p Why are small jobs favored in SRPT?
??x
Small jobs are favored because they gain priority as they receive service, and their remaining service time decreases. This allows them to progress through the system more quickly compared to larger jobs.

:x

--- 

---

#### Numerical Evaluation with Mathematica

:p What tools are used for numerical evaluation of response times in this section?
??x
Mathematica is used for numerical evaluation of response times, providing a platform to integrate complex expressions and compare different policies across various load conditions and job size distributions.

:p How does SRPT perform relative to other policies in terms of mean response time?
??x
SRPT performs better than other policies like PSJF because it has lower waiting times due to the integral term and service completion term. The extra \( x^2 \) term ensures that smaller jobs are favored, reducing their overall wait.

:x

--- 

---

#### Overall Mean Response Time

:p How is the overall mean response time calculated?
??x
The overall mean response time, \( E[T] \), is calculated as a weighted integral of \( E[T(x)] \) over all job sizes \( x \). This involves evaluating the formulas for different policies and integrating them to get an average response time.

:p What does Figure 33.1 reveal about policy performance?
??x
Figure 33.1 reveals that SRPT outperforms other policies in terms of mean response time, especially under high load conditions where small jobs benefit significantly from the prioritization provided by SRPT.

:x

--- 

---

#### SJF Policy Performance under High Load

Background context: The Shortest Job First (SJF) policy is known to perform well for high load scenarios due to a specific term in its mean response time formula. This term, \(1 - \rho x\), significantly improves its performance compared to policies like PS, where the denominator only includes \(1 - \rho\).

:p Why does SJF start performing better than other policies under high loads?
??x
The 1−ρx term in the denominator of E[TQ(x)]SJF helps it perform much better under high load conditions. This is because as ρ (load) increases, the system becomes busier, and this term accounts for the increased probability that a job will get some service even if it starts very late.

```java
// Pseudocode to illustrate SJF policy's response time formula impact
public double EResponseTimeSJF(double x, double rho) {
    return (1 - Math.pow(rho * x, 2)) / (1 - rho);
}
```
x??

---

#### PS Policy Invariance to Job Size Variability

Background context: The Processor-Sharing (PS) policy remains unaffected by the variability of job sizes. This property is useful in scenarios where the exact size distribution might be uncertain but still ensures fair service to all jobs.

:p Why is the PS = PLCFS line flat in Figure 33.2?
??x
The PS = PLCFS line appears flat because these policies are invariant to the variability (C2) of the job size distribution, meaning their performance does not change with different levels of C2.

```java
// Pseudocode demonstrating PS policy's response time formula
public double EResponseTimePS(double x, double rho) {
    return 1 / (1 - rho);
}
```
x??

---

#### FB Policy Performance under Low Variability

Background context: The Fair-Balance (FB) scheduling policy requires a job size distribution with Decreasing Failure Rate (DFR). Higher DFR is associated with higher C2, which can make the FB policy perform poorly when variability in job sizes is low.

:p Why do policies like FB look worse for low values of C2?
??x
For the Fair-Balance (FB) policy to perform well, it requires a job size distribution with decreasing failure rate. However, this requirement means that higher DFR is coupled with higher C2. When C2 is low, the variability in job sizes is high, which makes FB less effective compared to other policies.

```java
// Pseudocode for checking if a given job size distribution has DFR property
public boolean hasDFR(double lambda, double alpha) {
    return alpha < 1;
}
```
x??

---

#### SRPT Fairness Considerations

Background context: While the Shortest Remaining Processing Time (SRPT) scheduling policy is optimal in terms of mean response time and second moment, it is rarely used due to concerns about long job starvation. The All-Can-Win theorem states that every job prefers SRPT over PS under certain conditions.

:p Why might people be concerned about using SRPT?
??x
People fear that SRPT could cause long jobs to "starve" because large jobs get low priority under SRPT and may not receive adequate service, leading to higher response times compared to fair policies like Processor-Sharing (PS).

```java
// Pseudocode for SRPT scheduling logic
public class SRPTScheduler {
    public Job serveNextJob() {
        // Logic to select the job with the shortest remaining processing time
        return findShortestRemainingTimeJob();
    }
}
```
x??

---

#### Preference of Mr. Max for PS or SRPT

Background context: In an M/G/1 queue with a Bounded Pareto job size distribution and load ρ=0.9, it is initially intuitive that the largest job (Mr. Max) would prefer Processor-Sharing (PS). However, analysis shows that almost all jobs prefer SRPT to PS by significant factors.

:p Which policy does Mr. Max prefer in the M/G/1 queue with Bounded Pareto job size distribution?
??x
Surprisingly, Mr. Max and almost all other jobs prefer the Shortest Remaining Processing Time (SRPT) scheduling policy over Processor-Sharing (PS). The expected slowdown under SRPT is lower for large jobs compared to PS.

```java
// Pseudocode for comparing E[Slowdown] of two policies
public double compareESlowdown(double x, double rho) {
    // Assuming the All-Can-Win theorem and Mathematica calculations
    return (SRPTExpectedSlowdown(x, rho) < PSExpectedSlowdown(x, rho)) ? SRPT : PS;
}
```
x??

---

#### All-Can-Win Theorem

Background context: The All-Can-Win theorem states that for an M/G/1 queue with ρ<0.5, every job prefers SRPT over PS in expectation. This theorem applies to various job size distributions, including the Bounded Pareto distribution.

:p What is the significance of the All-Can-Win theorem?
??x
The All-Can-Win theorem signifies that under light load conditions (ρ<0.5), every job benefits from SRPT scheduling because it ensures that no job will experience a worse slowdown compared to PS, even for large jobs.

```java
// Pseudocode for the All-Can-Win proof logic
public boolean allCanWin(double rho) {
    return rho < 0.5;
}
```
x??

---

#### Mathematical Derivation for Fairness and SRPT Policy

Background context: The provided excerpt discusses a mathematical derivation to show that seemingly "unfair" policies like Shortest Remaining Processing Time (SRPT) can outperform fair policies like Processor Sharing (PS) in expectation, on every job size. This involves complex inequalities and integrals related to the policies' performance metrics.

:p What inequality needs to be shown to demonstrate the counterintuitive nature of fairness?
??x
To show that SRPT can outperform PS in expectation for any job size, we need to prove the following inequality:

\[ \frac{\lambda^2}{\int_{0}^{x} t^2 f(t) dt} + \frac{\lambda^2 x^2 (1 - F(x))}{(1 - \rho_x)^2} \leq \frac{\lambda x^2 (1 - F(x))}{1 - \rho} + \frac{\lambda}{\int_{0}^{x} t^2 f(t) dt} \cdot \frac{1}{1 - \rho}. \]

This inequality needs to hold for the policy comparison. Here, \( \lambda \), \( x \), \( F(x) \), and \( \rho_x \) are parameters related to job arrival rates, job sizes, cumulative distribution functions, and utilization factors.

:x??

---

#### Utilization Factor (\(\rho\)) and Fairness

Background context: The text mentions that the proof technique involves showing an inequality under certain assumptions about the utilization factor \(\rho\), which is less than \(1/2\). This helps in understanding why seemingly "unfair" policies can outperform fair ones.

:p What condition must be true for the inequality to hold?
??x
The condition that must be true for the inequality to hold is:

\[ 2(1 - \rho_x)^2 > 1 - \rho. \]

Given that \( \rho > \rho_x \) and under the theorem assumption that \( \rho < 1/2 \), it follows that \( \rho_x < 1/2 \). Therefore, dividing both sides by \( 1 - \rho_x \), we see that this inequality is true when \( \rho_x < 1/2 \).

:x??

---

#### Integration and Probability

Background context: The text involves integrals related to the probability distribution of job sizes. Specifically, it deals with expressions involving \( f(t) \) (the probability density function) and \( F(x) \) (the cumulative distribution function). These are essential in understanding the performance metrics of different scheduling policies.

:p What do \( f(t) \) and \( F(x) \) represent?
??x
\( f(t) \) represents the probability density function, which gives the relative likelihood for a continuous random variable to take on a given value. On the other hand, \( F(x) \) is the cumulative distribution function, representing the probability that a random variable takes on a value less than or equal to \( x \).

These functions are crucial in the derivation as they help in calculating expected values and probabilities associated with job sizes.

:x??

---

#### Fairness vs. Performance

Background context: The main idea is to demonstrate how fairness does not always lead to optimal performance, using SRPT as an example. This involves complex mathematical derivations to compare different scheduling policies based on their performance metrics.

:p Why does the text claim that fairness can be counterintuitive?
??x
The text claims that fairness can be counterintuitive because a seemingly "unfair" policy like SRPT can outperform fair policies like PS in expectation for any job size. This is due to complex mathematical inequalities and integrals that show the superior performance of SRPT under certain conditions.

:x??

---

#### Theorem Assumptions

Background context: The derivation relies on specific assumptions, such as \( \rho < 1/2 \), which ensures that the inequality holds true. These assumptions are critical for the validity of the proof.

:p What is a key assumption made in the theorem?
??x
A key assumption made in the theorem is that the utilization factor \( \rho \) is less than \( 1/2 \).

This assumption simplifies the mathematical derivations and ensures that the inequality holds, demonstrating the counterintuitive nature of fairness in scheduling policies.

:x??

---

