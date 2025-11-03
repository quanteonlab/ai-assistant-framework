# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 27)

**Starting Chapter:** 16.7 Readings

---

#### Poisson Arrival Process and M/M/1 Queues

Background context: In Figure 16.9, we see two systems with servers that have different service rates but receive a common Poisson arrival process. The performance can be analyzed using the properties of M/M/1 queues.

:p What is the key performance metric for evaluating the systems in Figure 16.9?
??x
Both systems have the same expected number of jobs \(E[N]\) given by:
\[ E[N] = \rho_1 \frac{1}{1 - \rho_1} + \rho_2 \frac{1}{1 - \rho_2} \]
where \(\rho_1 = \lambda / 3\) and \(\rho_2 = \lambda / 6\). The performance is determined by the utilization factors of each server.
??x

---

#### Burke’s Theorem Application in Acyclic Networks

Background context: Figure 16.10 illustrates an acyclic network with multiple servers, where each server can be treated independently due to the probabilistic routing.

:p According to Burke's theorem, what type of process is the arrival into each server?
??x
According to part (1) of Burke’s theorem, the arrival process into each server is a merged and/or split Poisson process. Therefore, each server in isolation can be modeled as an M/M/1 queue.
??x

---

#### Limiting Probabilities in Acyclic Networks

Background context: Using Burke's theorem, we can determine the independence of job numbers at different servers in an acyclic network.

:p How are the limiting probabilities calculated for such a network?
??x
The limiting probabilities \(\pi_{n_1, n_2, ..., n_k}\) can be expressed as:
\[ \pi_{n_1, n_2, ..., n_k} = P\{n_1 \text{ jobs at server 1}\} \cdot P\{n_2 \text{ jobs at server 2}\} \cdots P\{n_k \text{ jobs at server k}\} \]
\[ = \rho_{n_1}^1 (1 - \rho_1) \cdot \rho_{n_2}^2 (1 - \rho_2) \cdots \rho_{n_k}^k (1 - \rho_k) \]
where \(\rho_i\) is the utilization factor for server \(i\).
??x

---

#### Probability of Jobs at a Server

Background context: The probability \(P\{N_1 = n_1\}\) needs to be determined in an acyclic network with \(k\) servers.

:p What is the formula for calculating \(P\{N_1 = n_1\}\)?
??x
The probability that there are \(n_1\) jobs at server 1, given by:
\[ P\{N_1 = n_1\} = \rho_{n_1}^1 (1 - \rho_1) \]
where \(\rho_1 = \lambda / s_1\) and \(s_1\) is the service rate of server 1.
??x

---
Each flashcard covers a different aspect of the provided text, ensuring comprehensive understanding without pure memorization.

---
#### Time-Reversibility and Burke’s Theorem
Background context explaining the concept. Time-reversibility is a property of certain stochastic processes, where the process can be run backward without altering its statistical properties. For continuous-time Markov chains (CTMCs), time-reversibility implies that the transition rates between states are symmetric when the chain is in equilibrium.
:p What does time-reversibility imply for a CTMC?
??x
Time-reversibility implies that the product of the transition rates along any cycle involving states in a finite subset S equals the product of the same cycle in reverse order. Additionally, the rate of traversing any path equals the rate of traversing the same path in the reverse direction.
For example:
- For states \( j_1, j_2, \ldots, j_n \in S \):
  \[
  q_{j_1,j_2} \cdot q_{j_2,j_3} \cdot \ldots \cdot q_{j_n,j_1} = q_{j_1,j_n} \cdot q_{j_n,j_{n-1}} \cdot \ldots \cdot q_{j_2,j_1}
  \]
- For any state \( j \) and path from \( j \) to another state:
  \[
  \pi_j \cdot q_{j, j_n} \cdot q_{j_n, j_{n-1}} \cdot \ldots \cdot q_{j_2, j_1} = \pi_{j_1} \cdot q_{j_1, j_2} \cdot q_{j_2, j_3} \cdot \ldots \cdot q_{j_n, j}
  \]
x??

---
#### Burke’s Theorem
Burke’s theorem states that in an M/M/1 queue, the departure process is a Poisson process with the same rate as the arrival process. This theorem extends to more general queueing networks under certain conditions.
:p Can you state Burke’s theorem?
??x
In an M/M/1 queue, the departure process is a Poisson process with the same rate as the arrival process.
For example:
- In an M/M/1 queue with arrival rate \(\lambda\) and service rate \(\mu\), if the system is stable (i.e., \(\lambda < \mu\)), then the departures form a Poisson process with rate \(\lambda\).
x??

---
#### Exact Throughput for Closed System Performance
The exact throughput \(X\) in a closed queueing network can be calculated using the service rates and routing probabilities. The mean response time (excluding think times) can also be computed.
:p How would you compute the exact throughput for a closed system performance?
??x
For a closed system with \(N\) users, where each user has a service rate \(\mu_i\) and a think time \(Z\), the exact throughput \(X\) is given by:
\[ X = N \cdot (1 - P_0) / \sum_{i=1}^{N} r_i \]
where \(P_0\) is the probability that no users are in the system, and \(r_i\) is the routing probability from user \(i\).

For example, for a network with 3 users:
- Service rates: \(\mu_1 = 1\), \(\mu_2 = 2\), \(\mu_3 = 2\)
- Routing probabilities: \(r_1 = r_2 = r_3 = 0.5\)

The throughput can be computed using the above formula.
x??

---
#### Asymptotic Throughput for High N
The asymptotic throughput for high \(N\) in a closed system performance can be approximated using operational analysis from Chapter 7, which often involves fluid or diffusion approximations.
:p How would you approximate the throughput for large \(N\) in a closed system?
??x
For large \(N\), the exact throughput formula can be approximated using operational laws. This typically involves analyzing the behavior of the system as \(N \to \infty\).

For example, if the service rates and routing probabilities are known:
- Use fluid or diffusion approximations to derive an asymptotic expression for throughput.
x??

---
#### Chip Manufacturing Plant
In a chip manufacturing plant, wafers pass through three stations with Poisson arrivals and exponential service times. The mean time from arrival until completion can be derived by analyzing the queueing network.
:p How would you derive the mean time for wafer processing in a chip manufacturing plant?
??x
For a chip manufacturing plant with Poisson arrivals and exponential service times:
- Each station has two workers serving a single queue, with service rates \(\mu_1 = 1\), \(\mu_2 = 2\), and \(\mu_3 = 3\).
- Wafers arrive according to a Poisson process with rate \(\lambda = 1\).

The mean time from arrival until chip creation can be derived by analyzing the queueing network. This involves calculating the average waiting times at each station.

For example:
```java
public class WaferProcessing {
    public static double meanTime(double lambda, double[] mu) {
        // Calculate waiting times and total time
        return 1.0 / (lambda - sum(mu)) + sum(mu);
    }

    private static double sum(double[] rates) {
        double sum = 0;
        for (double rate : rates) {
            sum += 1.0 / rate;
        }
        return sum;
    }
}
```
x??

---
#### Square-Root Stafﬁng in Chip Manufacturing
The square-root staffing rule is used to determine the minimum number of servers needed such that fewer than 20% of wafers experience any delay.
:p How would you apply the square-root stafﬁng rule for wafer processing?
??x
To apply the square-root stafﬁng rule:
- Assume wafers arrive according to a Poisson process with rate \(\lambda = 10,000\) wafers per second.
- Service rates at each station are \(\mu_1 = 1\), \(\mu_2 = 2\), and \(\mu_3 = 3\).

The rule suggests:
\[ k^* = \sqrt{\lambda} + z_{0.95} \cdot \sqrt{\frac{\lambda}{n}} \]
where \(z_{0.95}\) is the Z-score for 95th percentile, and \(n\) is the number of parallel servers.

For example:
```java
public class SquareRootStafﬁng {
    public static int calculateServers(double lambda, double[] mu) {
        // Calculate the minimum number of servers needed
        return (int) Math.sqrt(lambda) + 1.645 * Math.sqrt(lambda / 2);
    }
}
```
x??

---
#### Alternative Views of Time-Reversibility
Time-reversibility involves proving properties about cycles and paths in a CTMC.
:p Prove that for any time-reversible CTMC, the product of transition rates along any cycle equals the same product in reverse order.
??x
For any states \(j_1, j_2, \ldots, j_n\) in a finite subset \(S\):
\[ q_{j_1,j_2} \cdot q_{j_2,j_3} \cdot \ldots \cdot q_{j_n,j_1} = q_{j_1,j_n} \cdot q_{j_n,j_{n-1}} \cdot \ldots \cdot q_{j_2,j_1} \]

This can be proven by using the detailed balance equations, which state:
\[ \pi_i q_{i,j} = \pi_j q_{j,i} \]
for all states \(i\) and \(j\).

Thus, for a cycle:
\[ \prod_{k=1}^{n} q_{j_k, j_{k+1}} = \prod_{k=n}^{1} q_{j_k, j_{k-1}} \]

Where the indices are taken modulo \(n\).
x??

---
#### Burke’s Theorem for Finite Queues
Burke’s theorem is considered for an M/M/1 queue with finite capacity.
:p Is the M/M/1/k single-server queue time-reversible?
??x
The M/M/1 queue with finite capacity \(k\) (M/M/1/k) is not time-reversible because the transition rates are not symmetric when considering the finite capacity. In a time-reversible system, the departure process must match the arrival process exactly, which is not the case for queues with finite capacity.
x??

---

#### Jackson Network Definition
A Jackson network is a general architecture of queueing networks where there are \(k\) servers, each with its own unbounded queue. Jobs at a server are served according to FCFS (First-Come-First-Served) order. The service rate for the \(i\)-th server is an exponential distribution with rate \(\mu_i\). Each server may receive arrivals from both inside and outside the network.

External arrivals into the \(i\)-th server follow a Poisson process with rate \(r_i\). Jobs are routed probabilistically between servers; if a job completes at server \(i\), it can be transferred to another server \(j\) with probability \(P_{ij}\) or exit the system with probability \(P_{i,out} = 1 - \sum_j P_{ij}\).

The response time of a job is defined as the total time from when the job arrives at the network until it exits. For each server \(i\), the total arrival rate, \(\lambda_i\), includes both internal and external arrivals.
:p What is the total rate at which jobs leave server \(j\)?
??x
The total rate at which jobs leave server \(j\) is given by \(\lambda_j\). This rate accounts for both jobs leaving due to service completion (i.e., from any other servers that might have routed to it) and direct arrivals.
x??

---
#### Arrival Process into Each Server
For a Jackson network, the arrival process into each server can be complex. While in acyclic networks, we saw that the arrival process is a Poisson process, this is not always true for non-acyclic (cyclic) networks.

In a cyclic network like Figure 17.2:
- An M/M/1 queue has Poisson arrivals.
- Departures from an M/M/1 are also Poisson with the same rate due to Burke’s theorem.
- Some fraction, \(1 - p\), of these departures leave the system, and a portion \(p\) gets fed back into the server.

The feedback process can disrupt the independence required for a Poisson process. For example, in Figure 17.3 with very low arrival rates:
- The time between arrivals is typically high.
- If an arrival happens at time \(t\), it's more likely to see another soon due to the low rate.
- This violates the independent increments property of a Poisson process.

:p Is the arrival process into each server in a cyclic network still a Poisson process?
??x
No, the arrival process into each server is not necessarily a Poisson process if the network has cycles. The feedback and correlated arrivals violate the independence required for a Poisson process.
x??

---
#### Merging Non-Independent Poisson Processes
In the example of Figure 17.3 with very low \(\lambda\):
- If an arrival happens at time \(t\), it is more likely to see another soon due to the high inter-arrival times, violating independence.

The incorrect argument merges two Poisson processes but assumes they are independent:
- Departures from M/M/1 are Poisson of rate \(\lambda\) by Burke’s theorem.
- Some fraction gets fed back and merged with external arrivals.

However, these processes were not independent, so their merge is not a Poisson process. This highlights the importance of careful analysis when dealing with feedback in queueing networks.
:p Why does merging two Poisson processes not necessarily result in another Poisson process?
??x
Merging two non-independent Poisson processes does not result in a Poisson process because independence is crucial for maintaining the Poisson property. The merge of dependent Poisson processes does not preserve the independent increments property required for a Poisson distribution.
x??

---
#### Implications for Solving Jackson Networks
Given that arrival processes can be complex, solving Jackson networks involves:
- Calculating \(\lambda_i\) using (17.1) or equivalently (17.2).
- Noting that if the network is cyclic, the arrival process into each server may not follow a Poisson distribution due to feedback and correlated arrivals.

This complexity necessitates more sophisticated methods for solving Jackson networks.
:p How do we handle non-Poisson arrival processes in Jackson networks?
??x
Non-Poisson arrival processes in Jackson networks can be handled by directly solving the equations (17.1) or (17.2) to find \(\lambda_i\). For cyclic networks, this often requires numerical methods due to the lack of simple Poisson properties.
x??

---

