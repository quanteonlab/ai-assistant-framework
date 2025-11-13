# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 7)

**Starting Chapter:** 5.1 Convergence

---

#### Definition of Convergence for Sequences of Numbers

Background context: In high school, we learned about the convergence of a sequence of numbers. A sequence $\{a_n : n=1,2,...\}$ converges to $b$ as $n \to \infty$, denoted as $ a_n \to b$or equivalently,$\lim_{n \to \infty} a_n = b $. This means that for any given degree of convergence $\epsilon > 0 $, there exists an index point $ n_0(\epsilon)$such that beyond this point, all elements of the sequence are within $\epsilon$ of $b$.

:p What does the definition of convergence for a sequence of numbers mean?
??x
The definition states that a sequence converges to $b $ if, for any given$\epsilon > 0 $, one can find an index point $ n_0(\epsilon)$such that all elements in the sequence beyond this point are within $\epsilon$ distance from $b$. Mathematically:
$$\forall \epsilon > 0, \exists n_0(\epsilon) : |a_n - b| < \epsilon \text{ for all } n > n_0(\epsilon)$$x??

---

#### Definition of Convergence for Random Variables

Background context: When dealing with random variables, we need a similar definition to that of sequences. A sequence of random variables $\{Y_n : n=1,2,...\}$ converges almost surely (a.s.) or with probability 1 (w.p.1) to $\mu$ if the probability of sample paths where the limit does not equal $\mu$ is zero.

:p What does $P\{\omega: \lim_{n \to \infty}|Y_n(\omega)-\mu| > k\}$ represent?
??x
This represents the probability of sample paths that behave badly, meaning for each such path $\omega $, the limit of the sequence $\{Y_n(\omega): n=1,2,...\}$ is not $\mu$ or does not exist. Formally:
$$P\{\omega: \lim_{n \to \infty}|Y_n(\omega)-\mu| > k\} = 0$$x??

---

#### Almost Sure Convergence

Background context: Almost sure convergence occurs when, on almost all sample paths, the sequence of random variables will start behaving well and continue to behave well after some point. Specifically, for any $k > 0 $, the probability that the limit does not converge to $\mu$ is zero.

:p Describe what almost sure convergence means.
??x
Almost sure convergence means that on a set of sample paths with total probability 1, the sequence of random variables converges to $\mu$. Formally:
$$\forall k > 0, P\{\omega: \lim_{n \to \infty}|Y_n(\omega) - \mu| > k\} = 0$$

This implies that almost all sample paths $\omega $ have the property that the sequence$\{Y_n(\omega): n=1,2,...\}$ converges to $\mu$.

An illustration might be:
```plaintext
Y(ω2) Y(ω1)
Y(ω3) Y(ω4)

n μ

Figure 5.1. Illustration of the concept of almost sure convergence.
```
Here, after some point, all sample paths behave well.

x??

---

#### Convergence in Probability

Background context: Convergence in probability is another way to describe how a sequence of random variables $\{Y_n : n=1,2,...\}$ converges to $\mu$. It states that for any $ k > 0$, the probability that the difference between $ Y_n$and $\mu$ exceeds $ k $ approaches zero as $n$ goes to infinity.

:p What does $P\{\omega: |Y_n(\omega)-\mu| > k\}$ represent in the context of convergence in probability?
??x
This represents the probability that a given sample path behaves badly for the random variable $Y_n $, meaning $ Y_n(\omega)$deviates from $\mu$ by more than $k$.

Formally:
$$P\{\omega: |Y_n(\omega)-\mu| > k\}$$

This is a number between 0 and 1.

x??

---

#### Comparison Between Almost Sure Convergence and Convergence in Probability

Background context: Both almost sure convergence and convergence in probability describe how sequences of random variables converge, but they are not the same. Almost sure convergence implies that for all $k > 0 $, with a probability of 1, the sequence converges to $\mu$ beyond some point.

:p How many badly behaving sample paths can there be according to almost sure convergence?
??x
There can be uncountably many such bad paths, each occurring with a probability zero and summing to a measure of zero. For example:
- Considering sequences made up of "red cars" (110) and "blue cars" (101), any sequence containing twice as many 1's as 0's is considered bad.
- There are uncountably many such sequences, demonstrating that there can be an infinite number of badly behaving paths.

x??

---

#### Example of Convergence in Probability

Background context: If $Y_n $ represents the average of the first$n $ coin flips for a fair coin (where each flip is either 0 or 1 with equal probability), we expect the sequence to converge to$\frac{1}{2}$.

:p What do we expect the sequence $\{Y_n(\omega): n=1,2,...\}$ to converge to if $Y_n$ represents the average of the first $n$ coin flips?
??x
We expect the sequence to converge to $\frac{1}{2}$, as each flip is equally likely to be 0 or 1.

x??

---

#### Limitations of Convergence in Probability

Background context: While convergence in probability is useful, it does not guarantee that all sample paths behave well. There can always be some sample paths where the sequence does not converge.

:p Why can't we say that the convergence in probability holds for all sample paths?
??x
There are always some sample paths (e.g., 1111...) that do not average to $\frac{1}{2}$ no matter how far out we look. However, these paths have a total measure of zero.

x??

---

#### Measure-Theoretic Concepts

Background context: The concepts discussed here are foundational for understanding more advanced topics in probability theory and stochastic processes. While the explanations provided cover key points, deeper study through books like Halmos’s [80] can provide a comprehensive understanding.

:p How many badly behaving sample paths are there according to convergence definitions?
??x
There can be uncountably many such bad paths, each occurring with a probability zero and summing to a measure of zero. This is because the sequences can behave differently on different paths, but the overall probability remains low.

x??

--- 

Each flashcard provides detailed explanations while focusing on one key concept at a time. The questions are designed to reinforce understanding rather than pure memorization.

#### Limit of a Sequence of Constants
Background context: In probability theory, understanding how a sequence converges to a limit is crucial. For a constant sequence $a_n = c $, the concept of convergence helps us understand if these constants approach some value as $ n$ becomes large.

:p How can we define the convergence in probability using the limit of a sequence of constants?
??x
For any $\epsilon > 0 $, there exists an $ N $such that for all$ n > N $, the probability that the absolute difference between$ Y_n $and$\mu $ exceeds$\epsilon$ is less than some small value. Formally, this can be stated as:
$$\forall k>0, \forall \epsilon_1>0, \exists n_0(\epsilon_1) \text{ s.t. } \forall n>n_0(\epsilon_1), P\{\omega : |Y_n(\omega) - \mu| > k\} < \epsilon_1.$$

This definition replaces the traditional limit of a sequence with one that works in terms of probability.
x??

---

#### Comparison Between Almost Sure Convergence and Convergence in Probability
Background context: These two types of convergence are fundamental concepts in probability theory, each with distinct properties and implications.

:p Which is stronger: almost sure convergence or convergence in probability?
??x
Almost sure convergence implies convergence in probability. Intuitively, if a sequence $Y_n $ converges almost surely to$\mu $, it means that for "almost every" sample path (with probability 1), the sequence will eventually get arbitrarily close to $\mu$ and stay there.

The formal intuition is that given almost sure convergence, each sample path will behave correctly from some point onward. As $n$ increases, fewer and fewer paths deviate significantly, making it likely that the mass of such bad paths decreases. However, this requires a rigorous proof.
x??

---

#### Convergence in Probability but Not Almost Surely
Background context: Understanding when a sequence can converge in probability without converging almost surely is important for grasping the nuances between these two types of convergence.

:p How might a sequence $\{Y_n\}$ converge in probability but not almost surely?
??x
Even if $\{Y_n\}$ converges in probability, it could still be that no single sample path behaves well after some point. For example, each path may spike occasionally; these spikes get further apart as $n$ increases. Thus, for no sample path $\omega$, does the sequence converge. However, for any fixed $ n$, the fraction of paths where $ Y_n$is far from $\mu$ becomes smaller as $n$ grows.

This can be visualized as follows: 
- For each path, there might be occasional deviations (spikes).
- These spikes get further apart.
- As $n$ increases, fewer and fewer paths exhibit such behavior. 

In summary, while the overall probability of deviation decreases with larger $n $, no single path converges to $\mu$. This is illustrated in Figure 5.2 where sample paths have occasional "bad" behaviors that do not align with convergence.
x??

---

#### Weak Law of Large Numbers
Background context: The weak law of large numbers (WLLN) provides a probabilistic framework for understanding the behavior of averages of i.i.d. random variables.

:p What does the Weak Law of Large Numbers state?
??x
The WLLN states that if $X_1, X_2, \ldots $ are independent and identically distributed (i.i.d.) random variables with mean$E[X]$, then the sample mean $ Y_n = \frac{S_n}{n}$converges in probability to $ E[X]$. Formally:
$$Y_n \xrightarrow{P} E[X], \text{ as } n \to \infty.$$

This is shorthand for:
$$\forall k > 0, \lim_{n \to \infty} P\{|Y_n - E[X]| > k\} = 0.$$

In simpler terms, the probability that the difference between $Y_n $ and the mean$E[X]$ exceeds any positive constant $k$ goes to zero as $n$ increases.
x??

---

#### Strong Law of Large Numbers
Background context: The strong law of large numbers (SLLN) provides a deterministic framework for understanding the behavior of averages of i.i.d. random variables.

:p What does the Strong Law of Large Numbers state?
??x
The SLLN states that if $X_1, X_2, \ldots $ are independent and identically distributed (i.i.d.) random variables with mean$E[X]$, then the sample mean $ Y_n = \frac{S_n}{n}$converges almost surely to $ E[X]$. Formally:
$$Y_n \xrightarrow{\text{a.s.}} E[X], \text{ as } n \to \infty.$$

This is shorthand for:
$$\forall k > 0, P\left\{ \lim_{n \to \infty} |Y_n - E[X]| \geq k \right\} = 0.$$

In simpler terms, almost every sample path of the sequence will eventually get arbitrarily close to $E[X]$ and stay there as $n$ increases.
x??

---

#### Strong Law for Bernoulli Random Variables
Background context: When dealing with i.i.d. Bernoulli random variables (which take values 0 or 1), understanding the SLLN helps in analyzing the long-term behavior of such sequences.

:p What does the Strong Law say about a sequence of i.i.d. Bernoulli random variables?
??x
For a sequence of i.i.d. Bernoulli random variables $X_i $ with mean$\frac{1}{2}$, the strong law says that for "almost every" sample path, if we average the outcomes far enough along the path, we will get convergence to $\frac{1}{2}$ from that point onward.

Even though there might be uncountably many paths that do not behave this way, the mass of such bad paths is zero when compared to all well-behaved sample paths.
x??

---

---
#### Tim and Enzo's Simulation Approaches
Tim, who viewed the world as a timeline, used one long sequence of coin flips to simulate the queue over an extended period. He recorded the number of jobs in the system every second for millions of samples and then took the average of these samples.

Enzo approached this from a more 2-dimensional perspective. He ran 1,000 shorter simulations each lasting 1,000 seconds. At time $t = 1,000$, he sampled the number of jobs in the system for each simulation and then averaged these values over all his experiments.

:p Who is "right"? Tim or Enzo?
??x
Both Tim and Enzo are correct but from different perspectives. Tim's approach calculates the time average by observing a single long sample path, while Enzo's method calculates the ensemble average by averaging over multiple short sample paths. Both methods provide valid ways to estimate the expected number of jobs in the system.

Tim's time average is given by:
$$N_{\text{Time Avg}} = \lim_{t \to \infty} \frac{\int_0^t N(v) dv}{t}$$

Enzo's ensemble average is given by:
$$

N_{\text{Ensemble Avg}} = \lim_{t \to \infty} E[N(t)] = \sum_{i=0}^\infty i p_i$$where $ p_i = \lim_{t \to \infty} P\{N(t) = i\}$.

In practice, both methods converge to the same value as $t$ approaches infinity.
x??

---
#### Time Average Definition
The time average of a process is defined as the limit of the integral of the number of jobs in the system divided by time over an infinite period. It captures the long-term behavior of a single sample path.

Formally, it is given by:
$$N_{\text{Time Avg}}(ω) = \lim_{t \to ∞} \frac{\int_0^t N(v, ω) dv}{t}$$where $ N(v, ω)$represents the number of jobs in the system at time $ v$under sample path $ω$.

:p What is the definition of the time average?
??x
The time average of a process is defined as the limit of the integral of the number of jobs in the system over an infinite period divided by time. This approach focuses on observing one long sequence to determine the average behavior.

Mathematically, it is given by:
$$N_{\text{Time Avg}}(ω) = \lim_{t \to ∞} \frac{\int_0^t N(v, ω) dv}{t}$$where $ N(v, ω)$represents the number of jobs in the system at time $ v$under sample path $ω$.
x??

---
#### Ensemble Average Definition
The ensemble average is defined as the limit of the expected value of the number of jobs in the system over an infinite period. It involves averaging multiple short simulations to capture the long-term behavior.

Formally, it is given by:
$$N_{\text{Ensemble Avg}} = \lim_{t \to ∞} E[N(t)] = \sum_{i=0}^\infty i p_i$$where $ p_i = \lim_{t \to ∞} P\{N(t) = i\}$, which represents the mass of sample paths with value $ i$at time $ t$.

:p What is the definition of the ensemble average?
??x
The ensemble average is defined as the limit of the expected number of jobs in the system over an infinite period. It involves averaging multiple short simulations to determine the long-term behavior.

Mathematically, it is given by:
$$N_{\text{Ensemble Avg}} = \lim_{t \to ∞} E[N(t)] = \sum_{i=0}^\infty i p_i$$where $ p_i = \lim_{t \to ∞} P\{N(t) = i\}$, which represents the mass of sample paths with value $ i$at time $ t$.
x??

---

#### Time Average vs Ensemble Average

Background context: The passage explains two ways to measure the average number of jobs in a system—time average and ensemble average. The time average focuses on a single process over an extended period, while the ensemble average considers all possible processes.

:p What is the difference between time average and ensemble average?
??x
The time average refers to calculating the average number of jobs by monitoring a single instance (sample path) over a long period, whereas the ensemble average involves considering the expected value across all possible instances or sample paths. The time average gives you an insight into how one specific process behaves over time, while the ensemble average provides an overall expectation based on all potential scenarios.
??x

---

#### Single Server Queue Example

Background context: The passage uses a single server queue as an example to illustrate both time and ensemble averages.

:p In the context of the single server queue, what does N(ν,ω) represent?
??x
N(ν,ω) represents the number of jobs in the system at time ν for a specific sample path ω. This is used to calculate the time average.
??x

---

#### Time Average Calculation

Background context: The passage provides an example calculation for the time average number of jobs in the system.

:p How do you calculate the time average number of jobs in the system?
??x
The time average number of jobs in the system is calculated by summing the number of jobs at each time step and dividing by the total number of time steps. For instance, if N(0) = 0, N(1) = 1, N(2) = 2, N(3) = 3, and N(4) = 2, then the time average is (0 + 1 + 2 + 3 + 2) / 5 = 8/5.

Example calculation:
```java
public class TimeAverageCalculation {
    public double calculateTimeAverage(int[] jobCountPerStep) {
        int totalJobs = 0;
        for (int jobs : jobCountPerStep) {
            totalJobs += jobs;
        }
        return (double) totalJobs / jobCountPerStep.length;
    }
}
```
??x

---

#### Ensemble Average Concept

Background context: The passage explains the concept of ensemble average, which considers all possible sequences and their probabilities.

:p What is the significance of the ensemble average in queueing theory?
??x
The ensemble average represents the expected number of jobs in the system when considering all possible sample paths (sequences) over time. It provides a broader perspective on the system's behavior by accounting for variability, making it more reliable than focusing on a single instance.
??x

---

#### Steady State Concept

Background context: The passage introduces the idea of steady state, which is crucial in understanding ensemble averages.

:p What does "steady state" mean in the context of queueing theory?
??x
In queueing theory, steady state refers to a point in time where the effects of initial conditions have dissipated, and the system's behavior is consistent over long periods. At steady state, the probabilities of having different numbers of jobs in the system stabilize.
??x

---

#### Steady State Probability Calculation

Background context: The passage discusses how to calculate expected values at any given time by considering all possible sequences.

:p How do you compute E[N(t)] for a specific time t?
??x
To compute E[N(t)], you consider all possible states the system can be in at time t and their corresponding probabilities. For example, if at time 1 there's a probability p0 that the system is empty and a probability p1 that it contains one job, then E[N(1)] = 0 * p0 + 1 * p1.

Example calculation:
```java
public class SteadyStateProbability {
    public double calculateExpectedJobs(int[] states, double[] probabilities) {
        double expectedValue = 0;
        for (int i = 0; i < states.length; i++) {
            expectedValue += states[i] * probabilities[i];
        }
        return expectedValue;
    }
}
```
??x

#### Time Average vs. Ensemble Average: Concepts
Background context explaining the difference between time average and ensemble average, focusing on the practical implications for Tim (time average) and Enzo (ensemble average).

:p What is the primary difference between a time average and an ensemble average?
??x
Tim measures the time average, which involves averaging over time along a single sample path. Enzo measures the ensemble average, which involves averaging over many different sample paths at a specific point in time.

Ensemble average:
$$\text{N}_{\text{Ensemble}} = \frac{1}{N} \sum_{i=1}^{N} Y(\omega_i)$$

Time average along a single path:
$$\text{N}_{\text{Time Avg}} = \lim_{T \to \infty} \frac{1}{T} \int_0^T Y(t) dt$$x??

---

#### Ergodic System: Definition and Intuition
Background context explaining the concept of an ergodic system, including its defining properties.

:p What is an ergodic system, and what are the key conditions required for a system to be considered ergodic?
??x
An ergodic system is one that is positive recurrent, aperiodic, and irreducible. These terms ensure that:
- **Irreducibility:** The process can transition from any state to any other state.
- **Positive Recurrence:** Every state will be visited infinitely often with finite mean recurrence times.

For example, in a queueing system, the process must be able to empty (state 0) and fill up repeatedly. This ensures that different starting conditions do not significantly affect long-term behavior.

x??

---

#### Restart Concept: Intuition
Background context explaining what it means for a process to probabilistically restart itself and its importance in ergodic systems.

:p What does it mean for the system to "probabilistically restart itself"?
??x
In an ergodic system, particularly positive recurrent ones, if the number of jobs in the queue returns to zero (empty state), this event acts as a reset. The process can be considered as starting anew from the empty state. For instance, in a queueing model where a job is added and removed with probabilities $p $ and$q$ respectively, once the system empties, the next arrival of a new job will start the cycle again.

x??

---

#### Equivalence of Time Average and Ensemble Average
Background context explaining under what conditions the time average equals the ensemble average, leading to Theorem 5.9.

:p Under what conditions does the time average equal the ensemble average according to Theorem 5.9?
??x
According to Theorem 5.9, for an ergodic system, the time average along a single path converges to the ensemble average with probability one as $T$ approaches infinity. This equivalence holds because in an irreducible and positive recurrent system, states are revisited infinitely often with finite mean times between visits.

x??

---

#### Example of Positive Recurrence
Background context explaining how positive recurrence works through an example, such as a queueing model.

:p How can we demonstrate the concept of positive recurrence using a queueing model?
??x
Consider a simple queue where jobs arrive and are processed. If the system has a finite mean time between arrivals and departures (i.e., it empties in finite time), then over an infinite run, the number of jobs will return to zero infinitely often with a finite average time between each emptying.

For example:
```java
public class QueueModel {
    private int numJobs;
    
    public void tick() {
        if (numJobs > 0) {
            // Process job
            numJobs--;
        } else {
            // New job arrives
            numJobs++;
        }
    }
}
```
In this model, the system will empty and fill up repeatedly over time, demonstrating positive recurrence.

x??

---

#### Aperiodicity Condition
Aperiodicity ensures that the system state (number of jobs) is not tied to a particular time step. If the state depends on the parity of the time step, it can bias results based on when the observation stops.
:p What does aperiodicity ensure in a system?
??x
Aperiodicity ensures that the system's state transitions are not periodic or deterministic with respect to time steps. This prevents biases in the long-term behavior and averages observed at specific times. For example, if the number of jobs in the system is always 0 for even time steps and 1 for odd time steps, the average will be skewed based on when the observation stops.
x??

---

#### Ergodicity and Averages
Ergodic systems have the property that their time averages equal their ensemble averages. This is due to the Strong Law of Large Numbers (SLLN) which states that the sample mean converges almost surely to the expected value as the number of samples grows large.
:p Why does ergodicity imply that the time average equals the ensemble average?
??x
Ergodic systems allow us to use either a single long run or multiple short runs with averaging. The Strong Law of Large Numbers (SLLN) guarantees that the sample mean, calculated over a sufficiently long period from a single system, will converge almost surely to the expected value of the process, which is equivalent to the ensemble average when considering many independent realizations.
x??

---

#### Time and Ensemble Averages in Simulation
Time averages are obtained by running one simulation for an extended duration. Ensemble averages involve multiple simulations started at different times or with different initial conditions.
:p What are time and ensemble averages used for in simulation?
??x
Time averages use a single long simulation to estimate the average behavior over time, while ensemble averages aggregate results from multiple independent runs of the system. Both methods aim to estimate the same steady-state behavior but differ in their computational approach.
x??

---

#### Importance of Long Run Times
Long run times are crucial because they ensure that initial conditions do not significantly affect the observed behavior. This is necessary to reach a steady state where the system's properties become stable and representative of its long-term characteristics.
:p Why is running simulations for a "long" time important?
??x
Running simulations for a long time ensures that the transient effects due to initial conditions have dissipated, leading to more accurate estimates of the system's steady-state behavior. This helps in obtaining reliable averages without being influenced by initial biases or fluctuations.
x??

---

#### Average Time in System
The average time in the system can be defined from either a single long run (time average) or multiple short runs with averaging over all jobs (ensemble average).
:p How is the average time in the system calculated?
??x
The average time in the system can be calculated as follows:
- For a single long run:$T_{\text{Time Avg}} = \lim_{t \to \infty} \frac{\sum_{i=1}^{A(t)} T_i}{A(t)}$, where $ A(t)$is the number of arrivals by time $ t$and $ T_i $ is the time in system for the $i$-th arrival.
- For multiple short runs: $T_{\text{Ensemble}} = \lim_{i \to \infty} E[T_i]$, where $ E[T_i]$is the average time in system of the $ i$-th job, averaged over all sample paths.
x??

---

#### Markov's Inequality
Markov’s Inequality provides a way to bound the probability that a non-negative random variable exceeds some positive threshold. This inequality is particularly useful in proving other probabilistic limits.

If $X$ is a non-negative random variable, then:
$$P\{X > t\} \leq \frac{E[X]}{t}, \quad \forall t > 0.$$

This means that the probability of observing a value greater than $t $ for a non-negative random variable$X $ is at most the expected value of$ X $ divided by $t$.

:p State Markov's Inequality.
??x
Markov’s Inequality states that if $X$ is a non-negative random variable, then:
$$P\{X > t\} \leq \frac{E[X]}{t}, \quad \forall t > 0.$$x??

#### Chebyshev's Inequality
Chebyshev’s Inequality extends the concept of Markov’s Inequality to random variables with finite mean and variance, providing a bound on the probability that a random variable deviates from its mean by more than a certain amount.

Let $Y $ be a random variable with finite mean$E[Y]$ and finite variance $\sigma^2_Y$. Then:
$$P\{|Y - E[Y]| \geq t\} \leq \frac{\sigma^2_Y}{t^2}.$$

This inequality gives an upper bound on the probability that a random variable deviates from its mean by more than $t$.

:p State Chebyshev's Inequality.
??x
Chebyshev’s Inequality states that for any random variable $Y $ with finite mean$E[Y]$ and finite variance $\sigma^2_Y$:
$$P\{|Y - E[Y]| \geq t\} \leq \frac{\sigma^2_Y}{t^2}.$$x??

#### Weak Law of Large Numbers
The Weak Law of Large Numbers (WLLN) is a fundamental theorem in probability theory and statistics, which describes the convergence of the sample mean to the expected value as the number of trials increases.

For independent and identically distributed (i.i.d.) random variables $X_1, X_2, X_3, \ldots $ with finite mean$E[X]$ and finite variance $\sigma^2$, the WLLN states:
$$\lim_{n \to \infty} P\left( \left| \frac{S_n}{n} - E[X] \right| > \epsilon \right) = 0,$$where $ S_n = \sum_{i=1}^n X_i$.

:p State the Weak Law of Large Numbers.
??x
The Weak Law of Large Numbers (WLLN) states that for i.i.d. random variables $X_1, X_2, X_3, \ldots $ with finite mean$E[X]$ and finite variance $\sigma^2$:
$$\lim_{n \to \infty} P\left( \left| \frac{S_n}{n} - E[X] \right| > \epsilon \right) = 0,$$where $ S_n = \sum_{i=1}^n X_i$.
x??

---

These flashcards cover the key concepts of Markov’s Inequality, Chebyshev’s Inequality, and the Weak Law of Large Numbers. Each card provides a clear statement of the concept along with relevant formulas and explanations to aid in understanding.

