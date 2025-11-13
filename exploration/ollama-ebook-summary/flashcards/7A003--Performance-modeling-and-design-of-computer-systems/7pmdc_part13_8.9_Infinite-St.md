# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 13)

**Starting Chapter:** 8.9 Infinite-State Stationarity Result

---

#### Repair Facility Problem with Cost
Background context: The repair facility problem involves a Markov chain representing whether a machine is working (W) or broken (B). Given transition probabilities and costs, we need to find the annual repair bill. 
The relevant equations from the problem are:
- πW = 0.95 * πW + 0.4 * πB
- πB = 0.05 * πW + 0.6 * πB
- πW + πB = 1

:p What do you notice about the first two equations?
??x
These equations are linearly dependent, as shown by their identical structure and coefficients. The second equation can be derived from the first one by rearranging terms.
```java
// Example code to solve for πW and πB
public class RepairFacility {
    public static void main(String[] args) {
        double piW = 8 / 9; // Calculated value of πW
        double piB = 1 / 9; // Calculated value of πB
        
        System.out.println("πW: " + piW);
        System.out.println("πB: " + piB);
    }
}
```
x??

---

#### Umbrella Problem with General Probability p
Background context: The umbrella problem involves determining the probability that a professor gets soaked given the daily probability of rain, p. We need to derive the limiting probabilities for general p using stationary equations.
The relevant stationary equations are:
- π0 = π2 * (1 - p)
- π1 = π1 * (1 - p) + π2 * p
- π2 = π0 * 1 + π1 * p
- π0 + π1 + π2 = 1

:p What is the fraction of days the professor gets soaked if the probability of rain is p=0.6?
??x
The professor gets wet when she has zero umbrellas and it is raining, which corresponds to π0 * p. Given p = 0.6, we calculate:
π0·p = (1 - p) / (3 - p) * p = (1 - 0.6) / (3 - 0.6) * 0.6 = 0.4 / 2.4 * 0.6 = 0.1
Thus, the professor gets soaked 10% of the days.
```java
// Example code to calculate π0 for p=0.6
public class UmbrellaProblem {
    public static void main(String[] args) {
        double p = 0.6;
        double pi0 = (1 - p) / (3 - p);
        double soakedProbability = pi0 * p; // Probability of getting soaked
        
        System.out.println("Soaked probability: " + soakedProbability);
    }
}
```
x??

---

#### Infinite-State DTMCs
Background context: Infinite-state Discrete-Time Markov Chains (DTMCs) are common in scenarios where the state space is unbounded, such as modeling systems with an unlimited number of customers or jobs.
For infinite-state DTMCs, the limiting probability distribution πj can be defined using:
- πj = lim n→∞ Pn ij
- ∑∞ j=0 πj = 1

:p What does the statement "Given an infinite-state DTMC, let πj = lim n→∞ Pn ij > 0" imply?
??x
It implies that πj is the limiting probability of being in state j as time approaches infinity. This means that for each state j, there is a non-zero probability that the system will be in state j in the long run.
```java
// Example code to simulate an infinite-state DTMC (pseudocode)
public class InfiniteStateDTMC {
    public static void main(String[] args) {
        double p = 0.6; // Probability of rain
        
        // Simulate states over time, assuming πj > 0 for all j
        double pi0 = (1 - p) / (3 - p);
        System.out.println("π0: " + pi0);
    }
}
```
x??

---

#### Stationary Distribution in Infinite-State DTMCs
Background context: For both finite and infinite-state DTMCs, if the limiting distribution exists, it is equivalent to the stationary distribution. The theorem states that for an infinite-state DTMC:
- πj = lim n→∞ Pn ij > 0 (limiting probability of state j)
- ∑∞ i=0 πi = 1 (normalization condition)

:p What does the theorem state about the relationship between the limiting and stationary distributions in infinite-state DTMCs?
??x
The theorem states that for an infinite-state DTMC, if a limiting distribution exists, then it is also a stationary distribution. Moreover, no other stationary distribution can exist.
```java
// Example code to verify stationarity (pseudocode)
public class StationaryDistribution {
    public static void main(String[] args) {
        double p = 0.6; // Probability of rain
        
        // Calculate πj for infinite-state DTMC
        double pi0 = (1 - p) / (3 - p);
        
        System.out.println("π0: " + pi0); // Verify that it satisfies the stationary condition
    }
}
```
x??

---

#### Proof of Stationary Distribution in Infinite-State DTMCs
Background context: The proof involves showing two things:
1. That {πj, j=0, 1, 2,...} is a stationary distribution.
2. Any stationary distribution must be equal to the limiting distribution.

:p What does the first part of the proof show?
??x
The first part shows that {πj, j=0, 1, 2,...} is a stationary distribution by demonstrating that πj = lim n→∞ Pn+1 ij.
```java
// Example code to prove stationarity (pseudocode)
public class StationaryProof {
    public static void main(String[] args) {
        double p = 0.6; // Probability of rain
        
        // Calculate πj for infinite-state DTMC
        double pi0 = (1 - p) / (3 - p);
        
        System.out.println("π0: " + pi0); // Verify that it satisfies the stationary condition
    }
}
```
x??

---

#### Proof of Uniqueness of Stationary Distribution in Infinite-State DTMCs
Background context: The second part of the proof involves proving that any stationary distribution must be equal to the limiting distribution.
:p What does the second part of the proof prove?
??x
The second part proves that any stationary distribution π′ must equal the limiting distribution πj. This is done by showing that for any j, π′j = πj.

```java
// Example code to verify uniqueness (pseudocode)
public class UniquenessProof {
    public static void main(String[] args) {
        double p = 0.6; // Probability of rain
        
        // Calculate πj for infinite-state DTMC
        double pi0 = (1 - p) / (3 - p);
        
        System.out.println("π0: " + pi0); // Verify that it satisfies the stationary condition and uniqueness
    }
}
```
x??

---

#### Stationary Equations and Limiting Probability Distribution for DTMCs

Background context: In discrete-time Markov chains (DTMCs), the stationary equations are used to find the limiting probability distribution vector $\pi $. This is done by solving an infinite number of equations, each representing a state's long-term probability. The key idea is that as $ n $approaches infinity, the probability of being in state$ j $converges to$\pi_j$.

Formulas and explanations:
- $\pi_j = \sum_{i=0}^{\infty} P_{ij} \pi_i $- To prove that $\pi_j$ is bounded above and below by itself, we use the sandwich theorem: 
  -$\sum_{i=0}^{M} P_{ij} \pi_i \leq \pi_j \leq \sum_{i=0}^{M} P_{ij} \pi_i + \sum_{i=M+1}^{\infty} \pi_i $- As $ M $ approaches infinity, the bounds become tighter and converge to $\pi_j$.

:p What is the main theorem or concept being discussed in this section?
??x
The main theorem discusses how to find the limiting probability distribution vector $\pi$ for a DTMC by solving an infinite number of stationary equations. It uses the sandwich theorem to prove that the solution converges to the actual probabilities.
x??

---

#### Unbounded Queue Example

Background context: The example provided deals with an unbounded queue system where jobs can arrive and depart at each time step according to certain probabilities. The goal is to determine the average number of jobs in the system using a DTMC model.

Formulas and explanations:
- Transition probability matrix $P$ for the infinite states (0, 1, 2, ...):
  -$ P = \begin{pmatrix} 
    1-r & r & 0 & 0 & \cdots \\
    s(1-r-s) & 1-r-s & s & 0 & \cdots \\
    0 & sr & 1-r-s & s & \cdots \\
    \vdots & \vdots & \vdots & \vdots & \ddots 
    \end{pmatrix} $- Stationary equations:
  -$\pi_0 = \pi_0(1-r) + \pi_1s $-$\pi_1 = \pi_0r + \pi_1(1-r-s) + \pi_2s $-$\pi_i = r/s \cdot \pi_{i-1}$:p What is the structure of the transition probability matrix for this unbounded queue example?
??x
The transition probability matrix $P $ has a specific structure where each row transitions to itself with probability$1-r $ and to neighboring states with probabilities dependent on$r $, $ s$. The matrix extends infinitely, making it difficult to handle using traditional methods.
x??

---

#### Solving Stationary Equations

Background context: When dealing with an infinite number of stationary equations in a DTMC, the approach is to express $\pi_i $ in terms of$\pi_0$. This involves repeatedly substituting expressions and observing patterns.

Formulas and explanations:
- Expressing $\pi_1 $ from$\pi_0$:
  - $\pi_1 = r/s \cdot \pi_0 $- Substituting $\pi_1 $ into the second equation to get$\pi_2$:
  - $\pi_2 = (r/s)^2 \cdot \pi_0$- Generalizing this for all states:
  -$\pi_i = (r/s)^i \cdot \pi_0 $:p How do you express $\pi_1 $ in terms of$\pi_0$?
??x
We express $\pi_1 $ as$\pi_1 = r/s \cdot \pi_0$.
x??

---

#### Determining $\pi_0 $ Background context: Once we have the general form for$\pi_i $, the next step is to determine $\pi_0 $ by using the normalization condition$\sum_{i=0}^{\infty} \pi_i = 1$.

Formulas and explanations:
- Sum of probabilities:
  - $\pi_0(1 + r/s + (r/s)^2 + (r/s)^3 + \cdots) = 1$
  - This is a geometric series with sum: 
    - $\pi_0 \cdot \frac{1}{1 - r/s} = 1 $:p How do you determine the value of$\pi_0$?
??x
To determine $\pi_0$, we use the normalization condition:
- $\pi_0 \cdot (1 + r/s + (r/s)^2 + (r/s)^3 + \cdots) = 1$- This is a geometric series with sum: 
  -$\pi_0 \cdot \frac{1}{1 - r/s} = 1 $- Therefore,$\pi_0 = 1 / (1 - r/s)$.
x??

---

#### Average Number of Jobs at the Server

Background context: In queueing theory, specifically in an M/M/1 queue (a single server queue where both arrival and service times are exponentially distributed), we can calculate the average number of jobs at the server using the stationary distribution. The formula derived is $E[N] = \frac{\rho}{1 - \rho}$.

:p What is the formula for calculating the average number of jobs at the server in an M/M/1 queue?
??x
The formula for calculating the average number of jobs at the server, denoted as $E[N]$, in an M/M/1 queue is given by:
$$E[N] = \frac{\rho}{1 - \rho}$$where $\rho $ is defined as the traffic intensity and can be expressed as the ratio of the arrival rate ($\lambda $) to the service rate ($\mu$):
$$\rho = \frac{\lambda}{\mu}$$

This formula helps us understand how the system load affects the average number of jobs present in the queue.
x??

---

#### Limiting Distribution for Program Analysis Problem

Background context: Given a Markov chain representing a program analysis problem, we can determine its limiting distribution by solving stationary equations. The stationary equations are:
$$\pi_i = \sum_{j} \pi_j P(j \rightarrow i)$$where $\pi_i $ is the limiting probability of state$i $, and$ P(j \rightarrow i)$represents the transition probability from state $ j$to state $ i$.

:p How would you solve for the stationary distribution, (πC, πM, πU), given a specific Markov chain?
??x
To determine the limiting distribution, $(\pi_C, \pi_M, \pi_U)$, by solving the stationary equations, follow these steps:

1. Write down the stationary equations for each state.
2. Solve the system of linear equations to find the values of $\pi_C $, $\pi_M $, and $\pi_U$.

For example, if we have a transition matrix $P$ with states C (correct), M (uncertain), and U (undefined), the stationary distribution must satisfy:
$$\pi_C = \pi_C p_{CC} + \pi_M p_{MC} + \pi_U p_{UC}$$
$$\pi_M = \pi_C p_{CM} + \pi_M p_{MM} + \pi_U p_{UM}$$
$$\pi_U = \pi_C p_{CU} + \pi_M p_{MU} + \pi_U p_{UU}$$

Additionally, the probabilities must sum to 1:
$$\pi_C + \pi_M + \pi_U = 1$$

Solving these equations will give us the limiting distribution.
x??

---

#### Powers of Transition Matrix

Background context: For any finite-state transition matrix $P $, it is important to understand that raising $ P $ to a power $ n$ maintains the property that each row sums to 1. This is crucial in understanding long-term behavior and stability of Markov chains.

:p Prove that for any integer $n $, $ P^n$ maintains the property that each row sums to 1.
??x
To prove that for any integer $n $, $ P^n $ maintains the property that each row sums to 1, we can use induction on $ n$.

**Base Case:**
For $n = 0$:
$$P^0 = I$$where $ I $ is the identity matrix. Each row of $ I$ clearly sums to 1.

**Inductive Step:**
Assume that for some integer $k $, $ P^k $ has each row summing to 1. We need to show that $ P^{k+1}$ also has this property.

By definition:
$$P^{k+1} = P^k \cdot P$$

Consider the first row of $P^{k+1}$:
The element in the first row and $i $-th column of $ P^{k+1}$ is given by:
$$(P^{k+1})_{1i} = \sum_{j=1}^n (P^k)_{1j} P_{ji}$$

By the induction hypothesis, each row of $P^k$ sums to 1. Therefore:
$$\sum_{i=1}^n (P^{k+1})_{1i} = \sum_{i=1}^n \left( \sum_{j=1}^n (P^k)_{1j} P_{ji} \right) = \sum_{j=1}^n (P^k)_{1j} \sum_{i=1}^n P_{ji}$$

Since each row of $P$ sums to 1:
$$\sum_{i=1}^n P_{ji} = 1$$

Thus:
$$\sum_{i=1}^n (P^{k+1})_{1i} = \sum_{j=1}^n (P^k)_{1j} \cdot 1 = \sum_{j=1}^n (P^k)_{1j} = 1$$

This shows that the first row of $P^{k+1}$ sums to 1. By a similar argument, all rows of $P^{k+1}$ sum to 1.

Hence, by induction, each row of $P^n $ sums to 1 for any integer$n$.
x??

---

#### Doubly Stochastic Matrix

Background context: A doubly stochastic matrix is one where the entries in each row and column sum up to 1. For a finite-state Markov chain with a doubly stochastic transition matrix, we can deduce that its stationary distribution must be uniform.

:p What can you prove about the stationary distribution of this Markov chain?
??x
For a finite-state Markov chain whose limiting probabilities exist and whose transition matrix is doubly stochastic (i.e., each row and column sums to 1), the stationary distribution $\pi$ must be uniform. This means that:
$$\pi_i = \frac{1}{n}$$where $ n$ is the number of states.

Proof:
Consider a doubly stochastic matrix $P $. By definition, for all rows $ i$:
$$\sum_{j=1}^n P_{ij} = 1$$

And for each column $j$:
$$\sum_{i=1}^n P_{ij} = 1$$

The stationary distribution $\pi$ satisfies:
$$\pi_i = \sum_{j=1}^n \pi_j P_{ji}$$

For the stationary distribution to be consistent with the doubly stochastic property, consider the following:

Summing both sides of the stationary equation over all states $i$:
$$\sum_{i=1}^n \pi_i = \sum_{i=1}^n \left( \sum_{j=1}^n \pi_j P_{ji} \right)$$

By the definition of doubly stochastic, this simplifies to:
$$\sum_{i=1}^n \pi_i = \sum_{j=1}^n \pi_j \cdot 1 = \sum_{j=1}^n \pi_j = 1$$

Thus:
$$\sum_{i=1}^n \pi_i = 1$$

Given that the matrix is doubly stochastic, each row and column sums to 1. This implies that every state has an equal contribution to the stationary distribution. Therefore, the only uniform solution is:
$$\pi_i = \frac{1}{n}$$for all $ i$.
x??

---

