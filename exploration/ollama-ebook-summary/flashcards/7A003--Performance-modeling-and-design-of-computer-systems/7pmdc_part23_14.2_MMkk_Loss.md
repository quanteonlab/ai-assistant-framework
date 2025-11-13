# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 23)

**Starting Chapter:** 14.2 MMkk Loss System

---

#### CTMC and Time-Reversibility

CTMC (Continuous-Time Markov Chain) is a stochastic process where the states change over time according to certain transition rates. A key property of some CTMCs is time-reversibility, meaning that the system behaves the same way when run forward or backward in time.

The balance equations for the CTMC are given by:

$$\pi_i q_{ij} = \pi_j q_{ji}, \forall i, j$$

If this condition holds true, then the stationary distribution (π) can be used to solve the balance equations directly.

:p What is an example of a CTMC that is not time-reversible?
??x
An example of a CTMC that is not time-reversible would be a chain where there's an arc from state i to state j labeled with transition rate qij, but no arc from state j to state i. In this case, the rate of going from state i to state j is πiqij, while the rate of going from state j to state i is zero.

This implies that transitions are only in one direction, breaking the time-reversibility condition.
x??

---

#### Birth-Death Processes

A birth-death process is a special type of CTMC where each state transition changes by exactly one unit. Specifically:

- From state $i $ to$i+1$: This is called a "birth."
- From state $i+1 $ to$i$: This is called a "death."

The key characteristic of birth-death processes is that the balance equations are simpler due to this linear transition pattern.

:p Are all birth-death processes time-reversible?
??x
Yes, all birth-death processes are time-reversible. To prove this, observe that during any period of time $t $, the number of transitions from state $ i $to$ i+1 $is within 1 of the number of transitions from state$ i+1 $to state$ i $. This is because you cannot repeat a transition from$ i $to$ i+1 $without first returning to state$ i $, and this return can only happen through a transition from$ i+1 $back to$ i$.

Thus, the long-run rate of transitions (number of transitions divided by time) from state $i $ to state$i+1 $ is equal to the rate of transitions from$ i+1 $ to state $i$. This ensures that the time-reversibility condition holds.

:p Can you provide an example of a birth-death process?
??x
A classic example of a birth-death process is the M/M/1 queue, where jobs arrive according to a Poisson process with rate $\lambda $, and service times are exponentially distributed with rate $\mu$.

In this model:
- The state transitions from $i $ to$i+1$ represent an arrival (birth).
- The state transitions from $i+1 $ to$i$ represent a departure (death).

The balance equations for such processes can be solved using the time-reversibility approach, leading to the steady-state probabilities.

:p How is time-reversibility useful in solving CTMCs?
??x
Time-reversibility simplifies the solution of CTMCs by providing a direct way to derive the limiting distribution. By setting up and solving the time-reversibility equations, one can often find the stationary distribution without having to solve the more complex balance equations directly.

For example, in an M/M/1 queue, we guess that $\pi_i = \left(\frac{\lambda}{\mu}\right)^i \cdot \pi_0 $. Substituting this into the time-reversibility equation confirms its correctness. Finally, by ensuring that the sum of all probabilities equals 1, we determine $\pi_0$ and thus the full stationary distribution.

:p What are insensitivity results in queueing theory?
??x
Insensitivity results in queueing theory refer to situations where certain performance measures (like blocking probability) depend only on the mean of a parameter rather than its specific distribution. For example, in an M/M/k/k system, the blocking probability $P_{\text{block}}$ depends solely on the ratio $\frac{\lambda}{\mu}$, not on the actual service time or interarrival time distributions.

This is significant because it means that the key performance metrics are robust to variations in the underlying distribution, making these results highly valuable for practical applications where exact distributions might be hard to determine.
x??

---

#### M/M/k/k Loss System

The M/M/k/k loss system models scenarios where a queue has $k $ servers and only allows up to$k $ jobs at any time. Arrivals follow a Poisson process with rate$\lambda $, and service times are exponentially distributed with rate$\mu$. If all servers are busy, an arrival is lost.

:p What should the state space be for this system?
??x
The state space for the M/M/k/k loss system represents the number of busy servers in the system. The states range from 0 to $k$, where:
- State 0: No jobs are being served.
- State $i $(1 ≤ i < k): Exactly $ i$ jobs are being served.
- State $k $: All $ k$ servers are occupied.

The CTMC for this system is shown in Figure 14.2, with transitions labeled by their respective rates $\lambda $ and$\mu$.

:p How do we determine the blocking probability?
??x
To determine the blocking probability $P_{\text{block}}$, which is the fraction of jobs that are lost when all servers are busy, we model the M/M/k/k loss system as a CTMC. The key idea is to solve the time-reversibility equations for the steady-state probabilities.

By solving these equations and summing up the probabilities from state 0 to $k$, we get:

$$P_{\text{block}} = \frac{\left(\frac{\lambda}{\mu}\right)^k / k!}{1 + \sum_{i=1}^{k} \frac{\left(\frac{\lambda}{\mu}\right)^i}{i!}}$$

This is known as the Erlang-B formula. By using the Poisson distribution, we can remember this result more easily:
$$

P_{\text{block}} = e^{-\lambda/\mu} \cdot \left(\frac{\lambda/\mu}{k!}\sum_{i=0}^{k} \frac{(λ/μ)^i}{i!}\right)$$:p What is the significance of insensitivity results in queueing theory?
??x
Insensitivity results are significant because they show that certain performance measures, such as blocking probability $P_{\text{block}}$ in an M/M/k/k system, depend only on the mean service and arrival rates ($\lambda $ and$\mu$), rather than their specific distributions. This means that even if the underlying distribution of service times or interarrival times is not known precisely, one can still accurately predict important queueing behavior.

For example, in an M/M/k/k system, $P_{\text{block}} = e^{-λ/μ} \cdot \frac{(λ/μ)^k}{k!} / \sum_{i=0}^{k} (λ/μ)^i / i!$, which only depends on the ratio $\lambda/\mu$.

This is particularly useful in practical applications where exact distributions are difficult to determine, as it allows for robust and reliable predictions.

#### System Utilization (ρ)
Background context: The system utilization, denoted by $\rho $, is a measure of how busy the servers are. In an M/M/k system, it is defined as $\rho = \frac{\lambda k}{\mu}$, where $\lambda $ is the arrival rate and $k\mu$ represents the total service capacity.

:p What is the definition of system utilization in an M/M/k queueing system?
??x
The system utilization $\rho $ for an M/M/k system is given by$\rho = \frac{\lambda k}{\mu}$. This measures the average load on the system, indicating how often servers are busy. It's derived from dividing the total arrival rate by the total service capacity.
x??

---

#### Resource Requirement (R)
Background context: The resource requirement $R $ is defined as the expected number of jobs in service and can also be interpreted as the minimum number of servers needed to maintain stability, given by$R = \frac{\lambda}{\mu}$.

:p What does the variable R represent in an M/M/k system?
??x
In an M/M/k system, $R $ represents the expected number of jobs in service and can also be viewed as the minimum number of servers required to keep the system stable. It is calculated as$R = \frac{\lambda}{\mu}$, where $\lambda $ is the arrival rate and $\mu$ is the service rate per server.
x??

---

#### Probability that an Arrival Has to Queue (PQ)
Background context: The probability that an arriving job has to queue, denoted by $P_Q$, is the probability that all servers are busy when a new job arrives. This can be expressed as:
$$P_Q = \sum_{i=k}^{\infty} \pi_i (i - k)$$where $\pi_i $ represents the stationary probabilities of having$i$ jobs in the system.

:p What is the formula for calculating $P_Q$, and what does it represent?
??x
The probability that an arriving job has to queue, $P_Q$, can be calculated using:
$$P_Q = \sum_{i=k}^{\infty} \pi_i (i - k)$$

This represents the probability that all servers are busy when a new job arrives and must join the queue.
x??

---

#### Erlang-C Formula
Background context: The famous Erlang-C formula is used to find $P_Q $, which is the blocking probability. It relates directly to the system utilization $\rho$ as:
$$P_Q = (1 - \rho) \frac{(\frac{k \rho}{1 - \rho})^k}{k!} \cdot \frac{\frac{k \rho}{(1 - \rho)^2}}{1 - \rho}$$:p What is the Erlang-C formula and what does it calculate?
??x
The Erlang-C formula for calculating $P_Q$, the probability that an arriving job has to queue, is given by:
$$P_Q = (1 - \rho) \frac{\left(\frac{k \rho}{1 - \rho}\right)^k}{k!} \cdot \frac{\frac{k \rho}{(1 - \rho)^2}}{1 - \rho}$$

This formula helps in determining the blocking probability, i.e., the likelihood that an arriving job will have to wait because all servers are busy.
x??

---

#### Expected Number of Jobs in Queue (E[NQ])
Background context: The expected number of jobs in queue $E[N_Q]$ is calculated as:
$$E[N_Q] = \frac{P_Q \rho}{1 - \rho}$$:p What is the formula for calculating $ E[N_Q]$?
??x
The expected number of jobs in queue $E[N_Q]$ can be calculated using the formula:
$$E[N_Q] = \frac{P_Q \rho}{1 - \rho}$$

This measures the average number of jobs waiting in the queue.
x??

---

#### Expected Time in Queue (E[TQ])
Background context: The expected time a job spends in the queue $E[T_Q]$ is given by:
$$E[T_Q] = \frac{E[N_Q]}{\lambda} = \frac{P_Q \rho}{\lambda (1 - \rho)}$$:p What is the formula for calculating $ E[TQ]$?
??x
The expected time a job spends in the queue $E[T_Q]$ can be calculated using:
$$E[T_Q] = \frac{E[N_Q]}{\lambda} = \frac{P_Q \rho}{\lambda (1 - \rho)}$$

This measures the average time a job waits before being served.
x??

---

#### Expected Total Time in System (E[T])
Background context: The expected total time a job spends in the system $E[T]$ is:
$$E[T] = E[T_Q] + \frac{1}{\mu} = \frac{P_Q \rho}{\lambda (1 - \rho)} + \frac{1}{\mu}$$:p What is the formula for calculating $ E[T]$?
??x
The expected total time a job spends in the system $E[T]$ can be calculated using:
$$E[T] = E[T_Q] + \frac{1}{\mu} = \frac{P_Q \rho}{\lambda (1 - \rho)} + \frac{1}{\mu}$$

This measures the average time a job spends from arrival to departure, including both queueing and service times.
x??

---

#### Expected Number of Jobs in System (E[N])
Background context: The expected number of jobs in the system $E[N]$ is:
$$E[N] = P_Q k + R$$where $ R$ is the resource requirement.

:p What is the formula for calculating $E[N]$?
??x
The expected number of jobs in the system $E[N]$ can be calculated using:
$$E[N] = P_Q k + R$$

This measures the average total number of jobs present in the system, including both those being served and those in queue.
x??

---

#### FDM vs M/M/1 Comparison
Background context: This concept compares Frequency-Division Multiplexing (FDM) and an $M/M/1 $ system in terms of their mean response time. The objective is to determine which configuration provides better performance, considering the load$\rho = \frac{\lambda}{k\mu}$.

Relevant formulas:
$$E[T]_{\text{FDM}} = \frac{1}{\mu - \frac{\lambda}{k}} = k \left( \frac{k\mu - \lambda}{k\mu} \right)$$
$$

E[T]_{M/M/1} = \frac{1}{k\mu - \lambda}$$

Explanation: In the FDM configuration, traffic is split into $k $ separate channels. Each job experiences a$k $ times higher arrival rate but also a $ k$ times higher service rate compared to the M/M/1 system. The mean response time in both systems is directly influenced by this factor.

:p Which system has a lower mean response time between FDM and M/M/1?
??x
The M/M/1 system has a lower mean response time because each job experiences $k $ times higher arrival rate but also$k$ times higher service rate, leading to an overall better performance in terms of mean response time.

```java
// Pseudocode for calculating the mean response times
public class ServerPerformance {
    public static double calculateE_T_FDM(double k, double lambda, double mu) {
        return 1 / (mu - lambda / k);
    }

    public static double calculateE_T_MMX1(double k, double lambda, double mu) {
        return 1 / (k * mu - lambda);
    }
}
```
x??

---

#### M/M/1 vs M/M/k Comparison
Background context: This concept compares the $M/M/1 $ and$M/M/k $ systems in terms of their mean response time. The objective is to understand how the load$\rho = \frac{\lambda}{k\mu}$ affects the performance in both configurations.

Relevant formulas:
$$E[T]_{M/M/k} = \frac{1}{\lambda \cdot PQ \cdot \rho(1 - \rho)} + \frac{1}{\mu}$$where $ PQ $ is the probability an arrival is forced to queue, and $\rho = \frac{\lambda}{k\mu}$.

Explanation: In the M/M/k system, traffic is lumped together but service capacity is split. The mean response time in this configuration depends on the load factor $\rho $ and the probability$PQ$ that an arrival has to wait due to queueing.

:p How does the M/M/1 system compare with the M/M/k system?
??x
The M/M/1 system generally outperforms the M/M/k system in terms of mean response time because it does not require queuing and the service capacity is not split, leading to a lower mean response time. The exact performance difference depends on the load factor $\rho $ and the probability$PQ$.

```java
// Pseudocode for calculating the mean response times in M/M/k system
public class ServerPerformance {
    public static double calculateE_T_MMXK(double lambda, double mu, double k) {
        double rho = lambda / (k * mu);
        // Assuming PQ is a function of rho and other factors
        double PQ = someQueueingProbabilityFunction(rho);
        return 1 / (lambda * PQ * rho * (1 - rho)) + 1 / mu;
    }
}
```
x??

---

