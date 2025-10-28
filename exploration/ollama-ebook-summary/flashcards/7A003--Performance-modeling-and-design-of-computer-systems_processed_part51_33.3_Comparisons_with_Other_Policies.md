# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 51)

**Starting Chapter:** 33.3 Comparisons with Other Policies

---

#### SRPT vs WSRPT Comparison
Background context explaining that this concept discusses the comparison between Shortest Remaining Processing Time (SRPT) and Weighted Shortest Remaining Processing Time (WSRPT) policies. The expression for \(E[T_Q]\) represents WSRPT, which is then compared to SRPT using equation (33.1).

:p What does the expression for \(E[T_Q]\) represent in this context?
??x
The expression for \(E[T_Q]\) represents the expected waiting time under WSRPT.
x??

---

#### Response Time Formula for SRPT
Background context explaining that the response time formula for Shortest Remaining Processing Time (SRPT) is given, and it helps understand how job sizes influence response times.

:p What is the formula for the response time \(E[T(x)]_{SRPT}\)?
??x
The formula for the response time under SRPT is:
\[ E[T(x)]_{SRPT} = \frac{\lambda^2}{2} \int_0^x t^2 f(t) dt + \frac{\lambda x^2 (1 - F(x))}{(1 - \rho_x)^2} + \int_0^x \frac{dt}{1 - \rho_t}. \]
x??

---

#### SRPT vs PSJF Comparison
Background context explaining that Shortest Remaining Processing Time First (SRPT) has a different waiting time compared to Priority Scheduling based on Job Fairness (PSJF), with SRPT having an additional \(x^2\) term.

:p How does the response time for SRPT compare to PSJF?
??x
The response time for SRPT is greater than that for PSJF due to the extra \(x^2\) term in the numerator. However, SRPT has a better residence time because it only needs to wait behind jobs smaller than its current remaining service requirement, whereas PSJF requires waiting behind all jobs smaller than the original size.
x??

---

#### SRPT vs FB Comparison
Background context explaining that First Come First Served (FCFS) and Shortest Remaining Processing Time (SRPT) are complements in terms of their priority systems. SRPT gives more priority to jobs as they receive service, while FB starts with high priority but loses it over time.

:p How does SRPT compare to Feedback (FB)?
??x
Lemma 33.1 shows that on every job size \(x\), SRPT beats FB. This is because the mean waiting time and residence time are lower under SRPT compared to FB.
x??

---

#### Evaluation of Scheduling Policies
Background context explaining that the overall mean response time for various scheduling policies has been evaluated using Mathematica, with results given in Figures 33.1 and 33.2.

:p How have the formulas for E[T] been evaluated?
??x
The formulas for \(E[T]\) were evaluated using Mathematica to understand how these policies compare with respect to overall mean response time.
x??

---

#### Weibull Job Size Distribution
Background context explaining that a Weibull job size distribution is used in evaluating the performance of different scheduling policies, with a specific mean and variance.

:p What type of job size distribution was used for evaluation?
??x
A Weibull job size distribution was used for evaluation, defined by a mean of 1 and \(C_2 = \frac{1}{10}\).
x??

---

#### Mean Response Time vs Load
Background context explaining that the mean response time as a function of load is given in Figure 33.1.

:p How does the mean response time change with increasing load?
??x
The mean response time increases with increasing load, as shown in Figure 33.1 for various scheduling policies.
x??

---

#### Mean Response Time vs Variance
Background context explaining that the mean response time as a function of variance is given in Figure 33.2.

:p How does the mean response time change with increasing variance?
??x
The mean response time increases with increasing variance, as shown in Figure 33.2 for various scheduling policies.
x??

---

#### SJF vs PS and SRPT Performance under High Load

Background context: The Shortest Job First (SJF) policy performs poorly at low loads due to high variability \(C_2\), but its performance improves significantly at high loads, often outperforming other policies like Processor Sharing (PS). This is because the term \((1-\rho x)\) in the denominator of \(E[T_Q(x)]_{SJF}\) helps it better under high load.

:p Why does SJF suddenly start looking much better compared to PS and SRPT at high loads?
??x
The SJF policy's performance improves significantly under high loads because the term \((1-\rho x)\) in its denominator helps mitigate the impact of variability. At low loads, this term is less significant due to lower utilization (\(\rho\)). However, as load increases, the effectiveness of this term becomes more pronounced, providing a larger improvement for SJF compared to other policies like PS.

```java
public class LoadImpact {
    public double sjfPerformance(double rho, double x) {
        // Denominator in E[T_Q(x)]_SJF includes (1 - ρx)
        return 1.0 / ((1 - rho * x));
    }
}
```
x??

---

#### PS Invariance to Job Size Variability

Background context: Processor Sharing (PS) scheduling policies are invariant to the variability of the job size distribution, meaning their performance remains consistent regardless of how variable the job sizes are.

:p Why is the line for PS = PLCFS flat in Figure 33.2?
??x
The line for PS = PLCFS is flat because PS policies do not change their behavior or performance as the variability \(C_2\) of the job size distribution changes. This means that regardless of how variable the job sizes are, the performance of these policies remains constant.

```java
public class PSInvariance {
    public double psPerformance(double rho) {
        // PS performance is independent of C2
        return 1.0 / (1 - rho);
    }
}
```
x??

---

#### FB Policy Performance under Low Variability

Background context: The Fair Bank (FB) policy performs worse for low variability \(C_2\) because it requires a Decreasing Failure Rate (DFR) property, which is closely related to high \(C_2\). Lower \(C_2\) values mean the job sizes are less variable and thus FB does not perform as well.

:p Why do policies like FB look worse for low C2?
??x
Policies like Fair Bank (FB) require a Decreasing Failure Rate (DFR) property to perform well. However, DFR is more prevalent at higher variability \(C_2\), meaning lower \(C_2\) values indicate less variability in job sizes. Consequently, FB policies do not benefit as much from the reduced variability and thus perform worse.

```java
public class FBPolicy {
    public double fbPerformance(double c2) {
        // FB performance degrades with lower C2 due to lack of DFR property
        return 100.0 / (c2 + 1);
    }
}
```
x??

---

#### SRPT Fairness and Job Size Ignorance

Background context: The Shortest Remaining Processing Time First (SRPT) policy, while optimal for mean response time, is rarely used due to the concern of long jobs being ignored. However, the All-Can-Win theorem states that every job prefers SRPT over PS in expectation under certain conditions.

:p Why might people be concerned about SRPT causing "long jobs to starve"?
??x
People are concerned about SRPT causing "long jobs to starve" because, although SRPT is optimal for mean response time, it may not provide the same fairness. Long jobs always have low priority under SRPT until they start receiving service, which can delay their completion compared to other policies like Processor Sharing (PS), where long and short jobs are treated more equally.

```java
public class SRPTStarvation {
    public double srptPerformance(double x) {
        // Simulate SRPT performance for a job of size x
        return 10.0 / Math.sqrt(x);
    }
}
```
x??

---

#### M/G/1 Queue - Mr. Max's Preference

Background context: In an M/G/1 queue, the Bounded Pareto distribution is used to model job sizes. The All-Can-Win theorem suggests that even large jobs prefer SRPT over PS in expectation under certain conditions.

:p Which policy does Mr. Max (the largest job) prefer between PS and SRPT for a load \(\rho = 0.9\)?
??x
Mr. Max, the largest job with size \(x = 10^{10}\), prefers SRPT over PS in expectation. Despite large jobs being ignored under SRPT until they start receiving service, once they begin to receive service, they gain priority over other jobs. This ensures that even the largest job will have a period of high priority.

```java
public class MaxJobPreference {
    public double psPerformance(double x) {
        // Simulate PS performance for a large job size x
        return 100.0 / (x + 1);
    }
    
    public double srptPerformance(double x) {
        // Simulate SRPT performance for a large job size x
        return Math.log(x) / 3;
    }
}
```
x??

---

#### All-Can-Win Theorem

Background context: The All-Can-Win theorem states that every job prefers SRPT to PS in expectation under certain conditions, specifically when the load \(\rho < 1/2\).

:p Why does the All-Can-Win theorem hold for all M/G/1 queues with \(\rho < 1/2\)?
??x
The All-Can-Win theorem holds because even large jobs prefer SRPT to PS in expectation. While large jobs are ignored under SRPT until they start receiving service, once they begin to receive service, their priority increases significantly. This ensures that every job, regardless of size, benefits more from SRPT than PS when the load is less than 1/2.

```java
public class AllCanWin {
    public double allCanWin(double rho) {
        // Check if the theorem holds for a given ρ
        return (rho < 0.5) ? 1 : 0;
    }
}
```
x??

---

#### Fairness vs Performance in Scheduling Policies

Background context: The provided text discusses a counterintuitive result in scheduling policies, where an "unfair" policy like Shortest Remaining Processing Time (SRPT) can outperform a fair policy like Proportional Share (PS) in terms of performance. This result is based on mathematical analysis and involves some inequalities related to the fairness parameter \( \rho \) and the job size distribution.

:p What does the text imply about the relationship between fairness and performance in scheduling policies?
??x
The text implies that an "unfair" policy, such as SRPT, can outperform a fair policy like PS in expectation for every job size. This result is counterintuitive because it suggests that a scheduling policy designed with less consideration of fairness can still provide better overall system performance.
x??

---

#### Mathematical Inequalities and Fairness

Background context: The text provides several inequalities to establish the relationship between \( \rho \), which represents some measure of fairness, and the performance of the scheduling policies. These inequalities involve terms like \( x(\rho - \rho_x) = \frac{\lambda}{\int_{x}^{\infty} t f(t) dt} > \lambda x^2 (1 - F(x)) \), where \( x \) is a job size, and \( \rho_x \) is the fairness parameter for jobs of size \( x \).

:p What inequalities are used to show that SRPT can outperform PS?
??x
The inequalities used include:
- \( x(\rho - \rho_x) = \frac{\lambda}{\int_{x}^{\infty} t f(t) dt} > \lambda x^2 (1 - F(x)) \)
- The inequality that needs to be shown is: 
  \[
  \frac{\lambda^2}{\int_0^{x} t^2 f(t) dt} + \lambda^2 x^2 (1 - F(x))(1 - \rho_x)^2 \leq \lambda x^2 (1 - F(x)) (1 - \rho)
  \]
- Further simplification leads to:
  \[
  2(1 - \rho_x)^2 > 1 - \rho
  \]
x??

---

#### Fairness Parameter ρ

Background context: The fairness parameter \( \rho \) is a measure of how fairly the system allocates resources. It is assumed that \( \rho < \frac{1}{2} \). The text shows that for SRPT to outperform PS, it suffices to prove certain inequalities involving \( \rho_x \), which are derived from the properties of the job size distribution and fairness.

:p What does the condition \( 2(1 - \rho_x)^2 > 1 - \rho \) imply about the fairness parameter?
??x
The condition \( 2(1 - \rho_x)^2 > 1 - \rho \) implies that for SRPT to outperform PS, the difference between 1 and the fairness parameter \( \rho_x \) should be large enough relative to \( \rho \). Given that \( \rho < \frac{1}{2} \), if \( \rho_x < \frac{1}{2} \), then this inequality is satisfied.

This result highlights that even a slightly "unfair" policy can perform better than a fair one, which is counterintuitive in the context of resource allocation.
x??

---

#### Conclusion on Fairness and Performance

Background context: The text concludes by stating that fairness is not always necessary for optimal performance. An unfair policy like SRPT can outperform a fair policy like PS for every job size when certain mathematical conditions are met.

:p What does the conclusion about fairness imply for practical scheduling?
??x
The conclusion implies that in practical scenarios, it might be beneficial to use scheduling policies that prioritize performance over strict fairness criteria. This finding challenges the common belief that fairness is always desirable and can lead to improved system efficiency even if some jobs receive less favorable treatment.

In practice, this means that while fairness may be an important consideration in certain contexts, there are cases where optimizing for performance might provide better overall outcomes.
x??

---

