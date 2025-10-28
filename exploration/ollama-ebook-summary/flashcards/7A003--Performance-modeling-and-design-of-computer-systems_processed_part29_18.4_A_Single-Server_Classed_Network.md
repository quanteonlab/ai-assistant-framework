# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 29)

**Starting Chapter:** 18.4 A Single-Server Classed Network

---

#### Notation for Classed Jackson Networks
Background context: This section introduces notations and modeling for classed Jackson networks, which handle multiple job classes within a queueing network. We define various probabilities and rates to describe the movement of jobs between servers and their service rates.

:p What are the key symbols used in defining the arrival and departure rates in classed Jackson networks?
??x
The key symbols include:
- \(r_i\): Arrival rate to server \(i\) from outside the network.
- \(r_i(c)\): Arrival rate of class \(c\) jobs to server \(i\) from outside the network.
- \(\lambda_i\): Total arrival rate to server \(i\), considering both inside and outside arrivals.
- \(\lambda_i(c)\): Total arrival rate of class \(c\) jobs to server \(i\).
- \(\mu_i\): Service rate at server \(i\).
- \(\rho_i = \frac{\lambda_i}{\mu_i}\): Utilization of server \(i\).

These symbols help in defining the flow and processing of different job classes within a network.
x??

---

#### Deriving Total Arrival Rate λj
Background context: In classed Jackson networks, we need to determine the total arrival rate \(\lambda_j\) into server \(j\). This is derived from both external arrivals and internal routing.

:p Can you derive the formula for \(\lambda_j(c)\), the arrival rate of class \(c\) jobs into server \(j\)?
??x
The formula to derive \(\lambda_j(c)\) is given by:
\[
\lambda_j(c) = r_j(c) + \sum_{i=1}^{k} \sum_{c'=1}^{\ell} \lambda_i(c') P(c' \rightarrow c)_i j
\]

Here, \(r_j(c)\) represents the external arrival rate of class \(c\) jobs to server \(j\). The summation terms represent the contribution from other servers and classes. \(P(c' \rightarrow c)_i j\) is the probability that a job at server \(i\) of class \(c'\) moves to server \(j\) and becomes a class \(c\) job.

To get \(\lambda_j\), we sum over all classes:
\[
\lambda_j = \sum_{c=1}^{\ell} \lambda_j(c)
\]

This formula helps in calculating the total arrival rate considering both external inputs and internal routing.
x??

---

#### States of the Continuous-Time Markov Chain (CTMC)
Background context: For a continuous-time Markov chain (CTMC) representation, we need to define states that capture not just the number of jobs but also their classes at each server.

:p How are the states defined in the CTMC for classed Jackson networks?
??x
The state \(z\) is defined as:
\[
z = (z_1, z_2, \ldots, z_k)
\]
where \(z_i = (c(1)_i, c(2)_i, \ldots, c(n_i)_i)\). Here, \(n_i\) denotes the number of packets at server \(i\), and \(c(j)_i\) is the class of the \(j\)th job at server \(i\).

In simpler terms:
- For a single server, if there are 3 jobs with classes (1, 2, 1) in order from head to tail, the state would be \((1, 2, 1)\).
- If multiple servers are considered, each server has its own state defined similarly.

This definition ensures that we capture not just the number of packets but also their class information.
x??

---

#### Single-Server Classed Network
Background context: To understand the basics of classed networks, let's consider a single-server scenario. This helps in grasping how different classes of jobs are processed and interact within a simple network.

:p What is the state probability \(\pi(c(1)_1, c(2)_1, \ldots, c(n_1)_1)\) for a classed M/M/1 queue with single server?
??x
The limiting probability \(\pi(c(1)_1, c(2)_1, \ldots, c(n_1)_1)\) can be guessed as follows:
\[
\pi(c(1)_1, c(2)_1, \ldots, c(n_1)_1) = \frac{\lambda_1}{c(1)_1} \cdot \frac{\lambda_1}{c(2)_1} \cdots \frac{\lambda_1}{c(n_1)_1} \cdot \mu_1^{n_1} \cdot (1 - \rho_1)
\]
where:
- \(\rho_1 = \frac{\lambda_1}{\mu_1}\) is the utilization of server 1.
- This formula considers both the class-specific arrival rates and the service rate.

The key idea here is that the probability depends on the specific classes of the jobs in the queue, not just their number. The term \(1 - \rho_1\) represents the stability condition for the system.
x??

---

#### Balance Equations
Background context: To ensure the correctness of our state probabilities, we need to verify they satisfy balance equations. These equations equate the rates at which the system leaves and enters a particular state.

:p How do you prove that the guessed \(\pi(c(1)_1, c(2)_1, \ldots, c(n_1)_1)\) satisfies the balance equations?
??x
To verify that the guessed \(\pi\) satisfies the balance equations, we equate the rate of leaving a state with the rate of entering it. For example, for a state \((c(1)_1, c(2)_1, \ldots, c(n_1)_1)\):

Leaving Rate:
- Occurs due to an arrival or departure.
- The rate can be calculated as: 
\[
n_1 \left(\frac{\lambda_1}{c(1)_1} + \sum_{i=1}^{k} \sum_{c'=1}^{\ell} \lambda_i(c') P(c' \rightarrow c(1))_i 1\right)
\]

Entering Rate:
- Occurs due to arrivals from other servers or external sources.
- The rate can be calculated as: 
\[
r_j + \sum_{i=1}^{k} \sum_{c'=1}^{\ell} \lambda_i(c') P(c' \rightarrow c(1))_i 1
\]

By equating these rates and simplifying, we confirm that the guessed probabilities satisfy the balance equations.
x??

---

#### State Transition and Arrival/Departure Rates
Background context: The passage describes state transitions within a classed network of queues, focusing on arrival and departure rates. It discusses how an arriving job can join or leave based on its class and the current state.

:p What are the two ways in which a new job can enter the system according to the text?
??x
The two ways are:
1. An arrival of class \(c(n_1)^1\) joins the end of the queue when we are in state \(\left( c^{(1)}_1, c^{(2)}_1, ..., c^{(n_1-1)}_1 \right)\).
2. A departure occurs from the head of the queue where the job is of class \(c\), and a new job of an unspecified class arrives.
x??

---
#### Rate Equations for Departure
Background context: The text provides rate equations to show that the limiting probabilities follow equation (18.2) in the M/M/1 classed queue scenario.

:p How are the rates of departure calculated according to the text?
??x
The rates of departure are calculated as follows:
\[ \text{Rate Leave} = \pi\left( c^{(1)}_1, c^{(2)}_1, ..., c^{(n_1-1)}_1 \right) \cdot \lambda_1 / \left( c(n_1)^1 \right) + \sum_{c} \pi(c, c^{(1)}_1, c^{(2)}_1, ..., c^{(n_1)}_1) \cdot \mu_1. \]
By substituting the guess from equation (18.2), we simplify this to:
\[ \text{Rate Leave} = \pi\left( c^{(1)}_1, c^{(2)}_1, ..., c^{(n_1-1)}_1 \right) \cdot \lambda_1 / \left( c(n_1)^1 \right) + \sum_{c} \lambda_1(c) \cdot \mu_1. \]
x??

---
#### Product Form Theorem for Classed Queues
Background context: The theorem states that the classed network of queues with \(k\) servers has a product form, and provides specific formulas for the probability distribution.

:p What is the formula given in Theorem 18.1 for the classed network of queues?
??x
The formula given is:
\[ \pi(z_1, z_2, ..., z_k) = \frac{1}{k} \prod_{i=1}^k P(\text{state at server } i \text{ is } z_i), \]
where \(z_i = \left( c^{(1)}_i, c^{(2)}_i, ..., c^{(n_i)_i} \right)\) and each server behaves like an M/M/1 classed queue.
x??

---
#### Local Balance Concept
Background context: The passage uses the concept of local balance to verify that the guess satisfies the conditions for state transitions.

:p What does A represent in the context of the local balance equations?
??x
A represents the rate at which leave state \(z\) due to arrival from outside:
\[ A = \pi(z_1, z_2, ..., z_k) \cdot \mu_i. \]
This is the total rate at which jobs enter the network from external sources.
x??

---
#### Equilibrium Conditions
Background context: The text explains how the rates of entering and leaving a state must balance in an equilibrium system.

:p How are Bi and B/prime i related for all 1 ≤ i ≤ k according to the text?
??x
For all \(1 \leq i \leq k\), \(B_i = B'_{i,out}\) as:
\[ B_i = \pi(z_1, z_2, ..., z_k) \cdot \lambda_i / (c(n_1)^1) + \sum_{c} \lambda_i(c) \cdot \mu_i. \]
x??

---
#### Product Form Verification
Background context: The passage verifies the product form by summing over all possible states and classes for each server.

:p How is \(P\{\text{state at server } i \text{ is } z_i\}\) calculated?
??x
\[ P\{\text{state at server } i \text{ is } z_i\} = (1 - \rho_i) \cdot \lambda_i / \left( c^{(1)}_i \right) \cdots \lambda_i / \left( c^{(n_i)_i} \right) \mu_{n_i}^i \times \prod_{j \neq i}^\infty \sum_{n_j=0} (1 - \rho_j) \cdot \lambda_{n_j}^j \mu_{n_j}^j. \]
Simplifying, this gives:
\[ P\{\text{state at server } i \text{ is } z_i\} = (1 - \rho_i) \cdot \lambda_i / \left( c^{(1)}_i \right) \cdots \lambda_i / \left( c^{(n_i)_i} \right) \mu_{n_i}^i. \]
x??

---
#### Corollary for Job Distribution
Background context: The corollary provides the distribution of jobs in each queue, similar to what was proved for the unclassed Jackson network.

:p What is the formula given for the probability of having \(n_i\) jobs at server \(i\)?
??x
The formula given is:
\[ P\{\text{Distribution of jobs is } (n_1, n_2, ..., n_k)\} = \prod_{i=1}^k P\{n_i \text{ jobs at server } i\} = \prod_{i=1}^k \rho_i^{n_i} (1 - \rho_i). \]
This is identical to what was proved for the unclassed Jackson network.
x??

---

#### Distribution of Jobs in Classed Networks

In a classed network of queues, jobs are distributed among servers according to their classes. Each server can be in one of several states, and the distribution of jobs depends on these states.

The probability of having \( n_1 \) jobs at server 1, \( n_2 \) jobs at server 2, ..., and \( n_k \) jobs at server k is given by:

\[ P\{ \text{Distribution of jobs is } (n_1, n_2, \ldots, n_k) \} = \sum_{c(1)^1...c(n_1)^1,..., c(1)^k...c(n_k)^k} P\left\{\begin{array}{l}
\text{state at server 1 is } z_1 \\
\text{state at server 2 is } z_2 \\
\vdots \\
\text{state at server k is } z_k
\end{array}\right\} \]

This can be simplified to:

\[ P\{ \text{Distribution of jobs is } (n_1, n_2, \ldots, n_k) \} = \prod_{i=1}^k \sum_{c(1)^i...c(n_i)^i} \lambda_i \left(\frac{\lambda_i}{c(1)^i}\right) \cdots \left(\frac{\lambda_i}{c(n_i)^i}\right) \mu_i^{n_i} (1-\rho_i) \]

Where:
- \( \lambda_i \): Total arrival rate into server i of class c packets.
- \( \rho_i \): Load at server i, given by \( \rho_i = \frac{\lambda_i}{\mu_i} \).
- \( \mu_i \): Service rate of server i.

:p What is the probability distribution of jobs in a classed network?
??x
The answer involves calculating the probabilities for each state of each server. This is done using the formula derived from the theorem, which takes into account the arrival and service rates of each class at every server.
```java
public double probabilityDistribution(int n1, int n2, int[] lambda, int[] mu) {
    double prob = 1;
    for (int i = 0; i < lambda.length; i++) {
        prob *= calculateServerProbability(n1 + n2, lambda[i], mu[i]);
    }
    return prob;
}

private double calculateServerProbability(int totalJobs, int lambda, int mu) {
    double rho = (double) lambda / mu;
    // Simplified calculation for the probability of having specific jobs in a server
    double probability = Math.pow(rho, totalJobs) * (1 - rho);
    return probability;
}
```
x??

---

#### Connection-Oriented ATM Network Example

In this example, we are dealing with a connection-oriented network where packets follow specific routes. The objective is to determine the expected time \( E[T] \) for packets on route 2.

:p How can we express the problem of determining \( E[T] \) for packets on route 2 as a classed network?
??x
We need to associate each packet with its respective class based on the routes it takes. Each route has specific arrival rates and service rates at different servers, which help in defining the transition probabilities between servers.

For example:
- Route 1: Poisson (3 pkts/sec)
- Route 2: Poisson (4 pkts/sec)
- Route 3: Poisson (5 pkts/sec)
- Route 4: Poisson (6 pkts/sec)

The class definitions for the packets on each route are as follows:
For class 1, arriving at server 1 with rate 3; transitioning to servers 2, 3, and out.
For class 2, arriving at server 1 with rate 4; transitioning to servers 3, 4, and out.
For class 3, arriving at server 2 with rate 5; transitioning to servers 3, 4, and out.
For class 4, arriving at server 2 with rate 6; only out.

The total arrival rates into each server can be calculated as:
- Server 1: \( \lambda_1(1) = 3 \), \( \lambda_1(2) = 4 \)
- Server 2: \( \lambda_2(1) = 5 \)
- Server 3: \( \lambda_3(1) = 3 + 4 = 7 \), \( \lambda_3(2) = 5 \), \( \lambda_3(3) = 6 \)
- Server 4: \( \lambda_4(2) = 5 + 6 = 11 \)

The load at each server can then be calculated as:
- \( \rho_1 = \frac{7}{10} \)
- \( \rho_2 = \frac{8}{10} \)
- \( \rho_3 = \frac{18}{20} \)
- \( \rho_4 = \frac{9}{10} \)

:p How do we determine the arrival rates into each server?
??x
We calculate the total arrival rate into each server by summing up the individual class arrival rates. For instance, to find \( \lambda_3(1) \), we sum the arrivals at server 3 from both server 1 (class 1 and class 2).

For all servers:
- Server 1: \( \lambda_1 = \lambda_1(1) + \lambda_1(2) = 7 \)
- Server 2: \( \lambda_2 = \lambda_2(1) + \lambda_2(3) = 8 \)
- Server 3: \( \lambda_3 = \lambda_3(1) + \lambda_3(2) + \lambda_3(3) + \lambda_3(4) = 18 \)
- Server 4: \( \lambda_4 = \lambda_4(2) + \lambda_4(3) = 9 \)

:p How do we determine the load at each server?
??x
The load at each server is determined by dividing the total arrival rate into that server by its service rate. This gives us the utilization factor \( \rho_i \).

For all servers:
- Server 1: \( \rho_1 = \frac{7}{10} \)
- Server 2: \( \rho_2 = \frac{8}{10} \)
- Server 3: \( \rho_3 = \frac{18}{20} \)
- Server 4: \( \rho_4 = \frac{9}{10} \)

:p What is the expected time for packets on route 2?
??x
The expected time \( E[T] \) can be derived from the utilization factors and service rates. However, in this example, it's more about setting up the classed network correctly.

Given the arrival and service rates, we can use queuing theory formulas to determine \( E[T] \), but for simplicity, let’s assume we have already set up the network as described.
```java
public double expectedTime(int lambda2, int mu4) {
    double rho4 = (double) lambda2 / mu4;
    return 1.0 / (mu4 * (1 - rho4));
}
```
x??

---

#### Expected Time for Route 2 Packets
We are determining the expected time that packets spend on route 2, which involves calculating the expected number of packets at each server and applying Little's Law. The expected time a packet spends at each server can be determined by first finding the expected number of packets at each server.
:p What is the expected time for packets on route 2?
??x
The total expected time for packets on route 2 is calculated as the sum of the times spent at each server visited in that route. For this specific example, we calculate:
\[ E[T_{\text{route 2}}] = E[T_1] + E[T_3] + E[T_4] \]
Given the values from the text:
- \(E[T_1] = \frac{7}{3} \times \frac{7}{7} = \frac{1}{3}\) sec
- \(E[T_3] = 1\) sec (since there are no other packets at server 2)
- \(E[T_4] = 1\) sec

Adding these together:
\[ E[T_{\text{route 2}}] = \frac{1}{3} + 1 + 1 = \frac{7}{3} \approx 2.33 \text{ sec} \]
However, the text states it as \(11/6\) seconds.
x??

---

#### Distribution of Job Classes
The problem involves calculating the probability that there are exactly \(s\) jobs of class 1 and \(t\) jobs of class 2 at a server using Theorem 18.1. This theorem provides a formula for such probabilities, which can be simplified as shown.
:p What is the right factor in equation (18.5)?
??x
The right factor in equation (18.5) represents the probability that there are \(s + t\) jobs at server \(i\). It follows from the theorem's expression and simplification steps provided in the text.
x??

---

#### Left Factor in Equation (18.5)
The left factor of equation (18.5) is derived to represent a conditional probability given that there are \(s + t\) jobs total at server \(i\). It accounts for the specific combination of job types.
:p What is the left factor in equation (18.5)?
??x
The left factor represents the probability that out of \(s + t\) jobs, exactly \(s\) are of type 1 and \(t\) are of type 2 at server \(i\). This can be simplified to:
\[ \frac{s+t \choose s} \left(\frac{\lambda_i(1)}{\lambda_i}\right)^s \left(\frac{\lambda_i(2)}{\lambda_i}\right)^t (1 - \rho_i) \]
This factor is derived from the theorem and represents the conditional probability given that there are \(s + t\) jobs total at server \(i\).
x??

---

#### CPU-Bound and I/O-Bound Jobs
In this example, we have two types of jobs: CPU-bound and I/O-bound. Each type has different arrival rates and behavior upon leaving or returning to the system.
:p What is the expected time in system for CPU-bound jobs?
??x
To find the expected time in the system for CPU-bound jobs, we need to consider their arrival rate, service rate at the CPU, and the probabilities of each job's fate after service. The CPU device has an exponential service rate of 2 jobs/sec, and CPU-bound jobs arrive at a rate of 0.2 jobs/sec.
Using the provided data:
- Probability of leaving: \(P_{\text{leave}} = 0.3\)
- Probability of returning to CPU queue: \(P_{\text{return to CPU}} = 0.65\)
- Probability of going to I/O queue and back: \(P_{\text{to I/O, then back to CPU}} = 0.05\)

First, calculate the effective arrival rate at the CPU considering all paths:
\[ \lambda_{\text{CPU-bound}} = 0.2 + 0.65 \times 0.05 \]
Then, use the Little's Law to find the expected time in system:
\[ E[T] = \frac{\lambda}{\mu} \]

Given that the service rate \(\mu = 2\) jobs/sec and effective arrival rate can be calculated as above.
x??

---

#### Average Number of CPU-Bound Jobs at the CPU
For this part, we need to determine the average number of CPU-bound jobs at the CPU using Little's Law.
:p What is the average number of CPU-bound jobs at the CPU?
??x
The average number of CPU-bound jobs at the CPU can be calculated using Little's Law:
\[ E[N] = \lambda W \]
where \(W\) is the expected time in system for a CPU-bound job.

From the previous calculation, we need to find the effective arrival rate and then use it to determine the time spent in the system. Given the service rate at the CPU (\(\mu = 2\)) and the total effective arrival rate:
\[ \lambda_{\text{CPU-bound}} = 0.2 + 0.65 \times 0.05 \]
Calculate \(W\) and then use it to find \(E[N]\).
x??

---

#### Expected Time in System for CPU-bound Jobs (Method 1)
Background context: The expected time in system for CPU-bound jobs is calculated using a recursive approach considering different paths through the network. This involves understanding routing probabilities and how jobs traverse between devices.

Relevant equations:
- \( E\left[ TC \right] = 0.3E[T|leaves after visiting 1 ] + 0.65E[T|loops back to 1 ] + 0.05E[T|loops back to 1 via 2 ] \)
- The equation expands into a recursive form that involves \( E\left[ TC \right] \) itself.

:p What is the expected time in system for CPU-bound jobs using Method 1?
??x
The expected time in system for CPU-bound jobs can be found by solving the recursive equation. By substituting and simplifying, we get:
\[ E\left[ TC \right] = 3.117 \]

This value is derived from the recursive relationships between different paths of the job routing.
x??

---

#### Expected Time in System for CPU-bound Jobs (Method 2)
Background context: An alternative method to calculate the expected time in system involves using Little's Law and considering the virtual times spent at each device.

Relevant equations:
- \( E\left[ TC \right] = E\left[ VC_{1} \right] \cdot E[T_1] + E\left[ VC_{2} \right] \cdot E[T_2] \)
Where \( E\left[ VC_i \right] \) are the expected virtual times at device 1 and 2.

:p How do you calculate the expected time in system for CPU-bound jobs using Method 2?
??x
Using Little's Law, we can express the expected time in system as a sum of weighted average times spent at each device. First, solve for \( E\left[ VC_{1} \right] \) and \( E\left[ VC_{2} \right] \):
\[ E\left[ VC_1 \right] = 1 + 0.65E\left[ VC_1 \right] + 1.0E\left[ VC_2 \right] \]
\[ E\left[ VC_2 \right] = 0.05E\left[ VC_1 \right] \]

Solving these, we get:
- \( E\left[ VC_1 \right] = 3 \)
- \( E\left[ VC_2 \right] = 1 \)

Then, use the equation for expected time in system:
\[ E\left[ TC \right] = 3E[T_1] + 1E[T_2] \]

Given that \( E[T_1] \) and \( E[T_2] \) are known from previous calculations, we can compute the total expected time.
x??

---

#### Average Number of CPU-bound Jobs at the CPU
Background context: The average number of CPU-bound jobs at server 1 is derived using the traffic intensity ratio (ρ) and considering the proportion of CPU-bound jobs.

Relevant equations:
- \( E\left[ N_{C1} \right] = \frac{\rho_1}{1 - \rho_1} \cdot \frac{\lambda_{C1}}{\lambda_{C1} + \lambda_{I1}} \)
Where \( \rho_1 = \frac{\lambda_1}{\mu_1} \) and \( \lambda_1 = \lambda_{C1} + \lambda_{I1} \).

:p How do you find the average number of CPU-bound jobs at server 1?
??x
The average number of CPU-bound jobs at server 1 can be found by using the traffic intensity ratio (ρ) and the proportion of CPU-bound jobs:
\[ E\left[ N_{C1} \right] = \frac{\rho_1}{1 - \rho_1} \cdot \frac{\lambda_{C1}}{\lambda_{C1} + \lambda_{I1}} \]

Given \( \rho_1 = 0.3663 \) and the values of λ and μ, we can compute:
\[ E\left[ N_{C1} \right] = \frac{0.3663}{1 - 0.3663} \cdot \frac{2/3}{2/3 + 5/76} \]

Solving this gives the expected number of CPU-bound jobs at server 1.
x??

---

#### Explanation for Expected Number of Jobs Formula
Background context: The formula to find the expected number of jobs at a given device involves considering both the traffic intensity and the proportion of specific job types.

Relevant equations:
- \( E\left[ N_{C1} \right] = E\left[ \text{Number of jobs at CPU} \right] \cdot p \)
Where \( p \) is the fraction of those jobs that are CPU-bound jobs.

:p Why do we use this formula to find the expected number of CPU-bound jobs?
??x
The formula uses the traffic intensity (ρ1) and the proportion of CPU-bound jobs to calculate the expected number of jobs. It leverages the fact that the expected number of jobs is directly proportional to the traffic intensity and the proportion of CPU-bound jobs.

Given:
\[ E\left[ N_{C1} \right] = \frac{\rho_1}{1 - \rho_1} \cdot \frac{\lambda_{C1}}{\lambda_{C1} + \lambda_{I1}} \]

This formula is derived from understanding the relationship between traffic intensity and job distribution.
x??

---

