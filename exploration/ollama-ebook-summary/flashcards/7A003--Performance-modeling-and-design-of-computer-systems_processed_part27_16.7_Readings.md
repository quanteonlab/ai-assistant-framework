# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 27)

**Starting Chapter:** 16.7 Readings

---

#### Performance Comparison of Two Systems

Background context: The performance comparison involves two systems where each system has a different routing configuration with Poisson arrivals and M/M/1 queue characteristics. The goal is to understand which configuration might have better overall performance based on the given formulas.

:p Which of these systems in Figure 16.9 has better performance?
??x
Both systems have the same performance because their expected number of jobs \(E[N]\) can be expressed as:
\[ E[N] = \frac{\rho_1}{1-\rho_1} + \frac{\rho_2}{1-\rho_2} \]
where \(\rho_1 = \frac{\lambda}{3}\) and \(\rho_2 = \frac{\lambda}{6}\).

This is derived from the properties of M/M/1 queues under Poisson arrivals, where \(E[N] = \frac{\rho_i}{1-\rho_i}\) for each server. Since both configurations have the same arrival rates and service rates per node in isolation, they will exhibit identical performance.

:p What does this imply about the systems?
??x
This implies that the performance of both systems is identical because the expected number of jobs at any point in time can be calculated using the M/M/1 queue formula independently for each server. The configuration with different routing paths and service rates per node does not change the overall system performance when considering only Poisson arrivals.

---

#### Acyclic Network Performance

Background context: For an acyclic network of servers, we use Burke’s theorem to determine the limiting probabilities of the number of jobs at each server. This involves understanding that each server can be treated as an M/M/1 queue in isolation and using parts (1) and (2) of Burke’s theorem.

:p What does Burke's theorem state for this context?
??x
Burke’s theorem states two key points:
1. The arrival process into a server is a Poisson process.
2. The number of jobs at different servers are independent given the arrival processes.

Thus, for \(k\) servers, the probability that there are \(n_1\) jobs at server 1, \(n_2\) jobs at server 2, ..., and \(n_k\) jobs at server k is:
\[ \pi_{n_1, n_2, \ldots, n_k} = P\{n_1 \text{ jobs at server 1}\} \cdot P\{n_2 \text{ jobs at server 2}\} \cdots P\{n_k \text{ jobs at server k}\} = \rho_{n_1} (1 - \rho_1) \cdot \rho_{n_2} (1 - \rho_2) \cdots \rho_{n_k} (1 - \rho_k). \]

:p How do we calculate the probability of having \(N_i = n_i\) jobs at server \(i\)?
??x
The probability of having \(N_i = n_i\) jobs at server \(i\) is given by:
\[ P\{N_i = n_i\} = \rho_{n_i} (1 - \rho_i) \]
where \(\rho_i = \frac{\lambda}{\mu_i}\), and \(\mu_i\) is the service rate for server \(i\).

For example, if we have a server with arrival rate \(\lambda\) and service rate \(\mu\):
\[ P\{N_i = n_i\} = \left( \frac{\lambda}{\mu} \right)^{n_i} \frac{(1 - \frac{\lambda}{\mu})^{n_i}}{n_i!}. \]

:p How does this formula relate to the performance of each server?
??x
This formula relates directly to the performance as it describes the probability distribution of jobs at each server in an M/M/1 queue. The term \(\rho = \frac{\lambda}{\mu}\) is the traffic intensity, which determines how full a server is likely to be. A lower \(\rho\) means fewer jobs and better performance.

:p How can we use this formula in practice?
??x
In practice, you can use this formula to predict the number of jobs at each server and understand the system's stability. For example:

```java
public class QueuePerformance {
    private double lambda; // Arrival rate
    private double mu;     // Service rate

    public QueuePerformance(double lambda, double mu) {
        this.lambda = lambda;
        this.mu = mu;
    }

    public double probabilityOfNJobs(int n) {
        double rho = lambda / mu;
        return Math.pow(rho, n) * (1 - rho) / factorial(n);
    }

    private double factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
}
```

This code calculates the probability of having \(N_i = n\) jobs at a server given the arrival and service rates. The `probabilityOfNJobs` method uses the formula derived from Burke’s theorem to compute the desired probabilities.

---

#### Time-Reversibility and Burke's Theorem Overview
Time-reversible chains have properties that make them interesting in queueing theory. The provided formula is for a specific Markov chain state transition probability.

:p What does the given equation represent?
??x
The given equation represents the calculation of the stationary distribution for a time-reversible Markov chain with \( k \) states, denoted by \( n_1, n_2, ..., n_k \).

Specifically:
\[ P\{N_1 = n_1\} = \sum_{n_2,n_3,...,n_k} \pi_{n_1,n_2,...,n_k} = \sum_{n_2,n_3,...,n_k} \rho^{n_1}_1 (1-\rho_1) \rho^{n_2}_2 (1-\rho_2)\cdots \rho^{n_k}_k (1-\rho_k) = \rho^{n_1}_1 (1-\rho_1). \]

This equation shows that the stationary probability of being in state \( n_1 \) is directly proportional to the arrival rate and inversely related to the sum of all service rates.

x??

---

#### Time-Reversibility Properties
The text mentions two properties for time-reversible chains: cycle product equality and path traversal equality. These properties are crucial for understanding the behavior of reversible Markov chains.

:p Prove that for any time-reversible CTMC, the product of transition rates along a cycle equals the reverse order.
??x
To prove this property, consider a finite subset of states \( S = \{j_1, j_2, ..., j_n\} \) and the cycle \( j_1 \to j_2 \to ... \to j_n \to j_1 \).

The product of transition rates in one direction is:
\[ q_{j_1,j_2} \cdot q_{j_2,j_3} \cdots q_{j_{n-1},j_n} \cdot q_{j_n, j_1}. \]

By time-reversibility, the reverse cycle's transition rates product must be equal:
\[ q_{j_1,j_n} \cdot q_{j_n,j_{n-1}} \cdots q_{j_2,j_1} = q_{j_1,j_2} \cdot q_{j_2,j_3} \cdots q_{j_{n-1},j_n} \cdot q_{j_n, j_1}. \]

This equality holds for any time-reversible chain.

x??

---

#### Burke’s Theorem for Finite Queues
Burke's theorem is typically applied to M/M/1 queues and states that the departure process from such a queue follows a Poisson distribution. However, the finite capacity \( k \) introduces complexity.

:p Does an M/M/1/k single-server queue with finite capacity follow Burke’s theorem?
??x
For an M/M/1/k queue:

- **Time-Reversibility**: The M/M/1 queue is time-reversible.
- **Applicability of Burke's Theorem**: In a standard M/M/1 queue, the departure process is Poisson. However, with finite capacity \( k \), the system might not maintain all properties required by Burke's theorem.

The key issue arises because in a finite capacity system, some customers may be lost if the server is busy when an arrival occurs, which can affect the departure process.

Therefore, while the M/M/1/k queue is time-reversible, Burke's theorem does not directly apply due to the possibility of loss and non-Poisson departures.

x??

---

#### Jackson Network Definition
Background context explaining the concept. A Jackson network is a very general form of queueing network with \(k\) servers, each having its own unbounded queue and serving jobs on a First-Come-First-Served (FCFS) basis. Each server has a service rate \(\mu_i\). Jobs arrive at each server according to a Poisson process with rate \(r_i\), and the routing of jobs is probabilistic based on probabilities \(P_{ij}\).

The response time of a job is defined as the total time from when it arrives in the network until it leaves, including any multiple visits to servers. The arrival rates \(\lambda_i\) at each server are computed using equations (17.1) and (17.2).
:p What is the total rate at which jobs leave server \(j\)?
??x
\(\lambda_j\) is both the total rate at which jobs enter server \(j\) and at which they leave server \(j\). Jobs can exit or move to another server with probabilities given by \(P_{ji}\) or stay in server \(j\) (i.e., \(P_{jj} = 1 - \sum_i P_{ij}\)).
x??

---
#### Total Arrival Rate into Each Server
The total arrival rate into each server is the sum of outside arrivals and internal transitions. Specifically, for server \(i\), we have:
\[ \lambda_i = r_i + \sum_j \lambda_j P_{ji} \]

Equivalently, we can write it as:
\[ \lambda_i (1 - P_{ii}) = r_i + \sum_{j \neq i} \lambda_j P_{ji} \]

:p What is the formula for the total arrival rate into server \(i\)?
??x
The total arrival rate into server \(i\) is given by:
\[ \lambda_i (1 - P_{ii}) = r_i + \sum_{j \neq i} \lambda_j P_{ji} \]
This equation accounts for both external arrivals and internal transitions.
x??

---
#### Arrival Process in Acyclic Networks vs. Cyclic Networks
For acyclic networks, the arrival process into each server can be considered as a Poisson process due to the simplification that jobs do not cycle back through servers.

However, in cyclic networks like those shown in Figures 17.2 and 17.3, this assumption no longer holds because of feedback or circular routing which can affect the independence of inter-arrival times.
:p Is the arrival process into each server still a Poisson process if the network is not acyclic?
??x
No, the arrival process into each server is not necessarily a Poisson process when the network is not acyclic. This is because feedback or circular routing can cause the inter-arrival times to be dependent, violating one of the key properties of a Poisson process.
x??

---
#### Example Illustrating Non-Poisson Arrival Process
Consider Figure 17.3 where an arrival at time \(t\) makes it much more likely that there will be another arrival in a short interval \((t, t+\epsilon)\) due to the very low arrival rate \(\lambda\). This dependence violates the independence of inter-arrival times required for a Poisson process.
:p Why is the merging argument incorrect?
??x
The merging argument is incorrect because it assumes that the two Poisson processes being merged are independent. However, in cyclic networks like the one shown, the feedback loop creates dependencies between arrivals, making the total arrival process non-Poisson.
x??

---

