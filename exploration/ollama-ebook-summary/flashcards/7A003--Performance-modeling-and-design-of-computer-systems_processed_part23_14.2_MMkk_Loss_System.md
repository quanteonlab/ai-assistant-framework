# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 23)

**Starting Chapter:** 14.2 MMkk Loss System

---

#### Time-Reversibility of CTMCs

Background context: A Continuous-Time Markov Chain (CTMC) is time-reversible if the rates of transitions between states are symmetric, i.e., πiqij = πjqji for all states \(i\) and \(j\). This property ensures that the chain looks statistically similar when run forwards or backwards in time. If a CTMC is not time-reversible, it means these transition rates are not equal.

:p Can you provide an example of a CTMC that is not time-reversible?
??x
Consider a chain where there is an arc from state \(i\) to state \(j\) labeled with rate \(q_{ij}\), but no arc from state \(j\) to state \(i\). In this case, the rate of going from state \(i\) to state \(j\) is \(\pi_i q_{ij}\), but the rate of going from state \(j\) to state \(i\) is zero. This non-reversibility implies that the chain cannot be time-reversible because it does not satisfy πiqij = πjqji for all states.

---

#### M/M/1 Birth-Death Process

Background context: The M/M/1 queueing system, which stands for Markovian arrivals and exponential service times with one server, is a birth-death process. In such processes, the rate of transitions between adjacent states (states \(i\) to \(i+1\)) are proportional to the rates at which events occur.

:p Are all birth-death processes time-reversible?
??x
Yes, all birth-death processes are time-reversible. The proof relies on the observation that during any period of time \(t\), the number of transitions from state \(i\) to \(i+1\) is within 1 of the number of transitions from \(i+1\) to \(i\). This is because you cannot transition directly from \(i\) to \(i+1\) without first returning to state \(i\). Therefore, long-run rates are equal for opposite transitions. Time-reversibility equations can be used to find limiting distributions if solutions exist.

---

#### M/M/k/k Loss System

Background context: The M/M/k/k loss system models scenarios where there is a fixed number of servers and any arrival finding all servers busy is lost. It's applicable in various fields such as telephony and network management.

:p What should the state space be for an M/M/k/k queueing system?
??x
The state space represents the number of busy servers in the system. For an M/M/k/k loss system, states range from 0 to \(k\), where:
- State 0: No servers are busy.
- State 1: One server is busy.
- ...
- State k: All \(k\) servers are busy.

This state space captures all possible scenarios of the number of active servers in the system.

---

#### Time-Reversibility Equations for M/M/k/k

Background context: To determine the limiting probabilities \(\pi_i\) for the states in an M/M/k/k queueing system, time-reversibility equations are used. These equations ensure that the chain looks statistically similar when run forwards or backwards.

:p How can we solve the time-reversibility equations to find the limiting probabilities?
??x
To solve the time-reversibility equations, we make a guess for \(\pi_i\) and verify it:
- Assume: \(\pi_i = \left( \frac{\lambda}{\mu} \right)^i / i! \cdot \pi_0\)

We can verify this by substituting back into the time-reversibility equation for \(\pi_i\):
\[ \pi_{i-1}\lambda = \pi_i 2\mu \]

Finally, we determine \(\pi_0\) such that:
\[ \sum_{i=0}^{k} \pi_i = 1 \]
This yields:
\[ \pi_0 = \frac{1}{\sum_{i=0}^{k} \left( \frac{\lambda}{\mu} \right)^i / i! } \]

Thus, the limiting probabilities are:
\[ \pi_i = \frac{\left( \frac{\lambda}{\mu} \right)^i / i! } {\sum_{j=0}^{k} \left( \frac{\lambda}{\mu} \right)^j / j! } \]

The blocking probability \(P_{block}\) is the probability that an arrival finds all servers busy, given by:
\[ P_{block} = \pi_k = \frac{\left( \frac{\lambda}{\mu} \right)^k / k! } {\sum_{j=0}^{k} \left( \frac{\lambda}{\mu} \right)^j / j! } \]

This is known as the Erlang-B formula.

---

#### Erlang-B Formula and Poisson Distribution

Background context: The Erlang-B formula calculates the blocking probability in an M/M/k/k system, which can be related to the Poisson distribution. This formula shows that the blocking probability depends only on the mean service rate \(\lambda/\mu\), not the specific service time distribution.

:p Can you derive the Erlang-B formula by relating it to the Poisson distribution?
??x
To derive the Erlang-B formula, we start from the fact that the number of busy servers follows a Poisson distribution with parameter \(\lambda/\mu\):
\[ P(X = k) = e^{-\frac{\lambda}{\mu}} \left( \frac{\lambda}{\mu} \right)^k / k! \]

The blocking probability \(P_{block}\) is the probability that all servers are busy:
\[ P_{block} = P(X = k) = e^{-\frac{\lambda}{\mu}} \left( \frac{\lambda}{\mu} \right)^k / k! \]

Alternatively, using the sum of probabilities up to \(k\):
\[ P_{block} = \frac{e^{-\frac{\lambda}{\mu}} \left( \frac{\lambda}{\mu} \right)^k / k! } {1 + \sum_{j=0}^{k-1} \frac{\left( \frac{\lambda}{\mu} \right)^j } {j! }} = P(X \leq k) \]

This is equivalent to:
\[ Pblock = e^{-\frac{\lambda}{\mu}} \cdot \frac{(\frac{\lambda}{\mu})^k / k! } {\sum_{j=0}^{k} (\frac{\lambda}{\mu})^j / j! } \]

Thus, the Erlang-B formula is:
\[ Pblock = e^{-\frac{\lambda}{\mu}} \cdot \frac{(\frac{\lambda}{\mu})^k / k! } {\sum_{j=0}^{k} (\frac{\lambda}{\mu})^j / j! } \]

This derivation connects the Erlang-B formula to the Poisson distribution, highlighting its independence from the specific service time distribution.

#### System Utilization Definition
The system utilization, denoted by \(\rho\), is defined for an M/M/k queueing system as \(\rho = \frac{\lambda}{k\mu}\). Here, \(\lambda\) is the arrival rate into the system in jobs/sec and \(k\mu\) represents the total service capacity of the system in jobs/sec.

:p What is the formula to calculate the system utilization for an M/M/k queue?
??x
The system utilization \(\rho\) for an M/M/k system is calculated as:
\[
\rho = \frac{\lambda}{k\mu}
\]
where \(\lambda\) is the arrival rate and \(k\mu\) is the total service capacity of the system.

x??

---

#### Expected Number of Busy Servers
The expected number of busy servers, denoted by \(R\), in an M/M/k system can be calculated as:
\[
R = \frac{\lambda}{\mu}
\]
This is derived from considering that each server is busy with probability \(\rho = \frac{\lambda}{k\mu}\) and there are \(k\) servers.

:p What is the formula for calculating the expected number of busy servers in an M/M/k system?
??x
The expected number of busy servers \(R\) in an M/M/k system can be calculated as:
\[
R = \frac{\lambda}{\mu}
\]
This result comes from recognizing that each server has a utilization rate of \(\rho = \frac{\lambda}{k\mu}\), and with \(k\) servers, the expected number busy is given by this fraction.

x??

---

#### Probability an Arriving Job Has to Queue
The probability that an arriving job has to queue, denoted as \(P_Q\), for an M/M/k system can be derived using the Erlang-C formula:
\[
P_Q = \frac{\left(\frac{k\rho}{1 - \rho}\right)^k \cdot \rho^0}{\sum_{i=0}^{k-1} \left(\frac{k\rho}{1 - \rho}\right)^i i! + \frac{(k\rho)^k}{(1 - \rho) k!}}
\]
where \(\rho = \frac{\lambda}{k\mu}\).

:p What is the formula for calculating the probability that an arriving job has to queue in an M/M/k system?
??x
The probability \(P_Q\) that an arriving job has to queue in an M/M/k system can be calculated using the Erlang-C formula:
\[
P_Q = \frac{\left(\frac{k\rho}{1 - \rho}\right)^k \cdot \rho^0}{\sum_{i=0}^{k-1} \left(\frac{k\rho}{1 - \rho}\right)^i i! + \frac{(k\rho)^k}{(1 - \rho) k!}}
\]
where \(\rho = \frac{\lambda}{k\mu}\).

x??

---

#### Expected Number in the Queue
The expected number of jobs in the queue, \(E[N_Q]\), for an M/M/k system can be derived as:
\[
E[N_Q] = P_Q \cdot \frac{\rho}{1 - \rho}
\]

:p What is the formula to calculate the expected number of jobs in the queue for an M/M/k system?
??x
The expected number of jobs in the queue \(E[N_Q]\) for an M/M/k system can be calculated using:
\[
E[N_Q] = P_Q \cdot \frac{\rho}{1 - \rho}
\]
where \(P_Q\) is the probability that an arriving job has to queue, and \(\rho\) is the system utilization.

x??

---

#### Expected Time in Queue
The expected time in the queue, \(E[T_Q]\), can be calculated as:
\[
E[T_Q] = \frac{1}{\lambda} E[N_Q] = \frac{P_Q \cdot \rho}{\lambda (1 - \rho)}
\]

:p What is the formula to calculate the expected time in the queue for an M/M/k system?
??x
The expected time in the queue \(E[T_Q]\) for an M/M/k system can be calculated as:
\[
E[T_Q] = \frac{1}{\lambda} E[N_Q] = \frac{P_Q \cdot \rho}{\lambda (1 - \rho)}
\]
where \(E[N_Q]\) is the expected number of jobs in the queue and \(P_Q\) is the probability that an arriving job has to queue.

x??

---

#### Expected Total Time
The total expected time, \(E[T]\), spent by a job can be derived as:
\[
E[T] = E[T_Q] + \frac{1}{\mu} = \frac{P_Q \cdot \rho}{\lambda (1 - \rho)} + \frac{1}{\mu}
\]

:p What is the formula to calculate the expected total time spent by a job in an M/M/k system?
??x
The expected total time \(E[T]\) spent by a job in an M/M/k system can be calculated as:
\[
E[T] = E[T_Q] + \frac{1}{\mu} = \frac{P_Q \cdot \rho}{\lambda (1 - \rho)} + \frac{1}{\mu}
\]
where \(E[T_Q]\) is the expected time in the queue and \(\frac{1}{\mu}\) is the service time.

x??

---

#### Expected Total Number
The total expected number of jobs, \(E[N]\), can be derived as:
\[
E[N] = E[N_Q] + E[\text{number being served}] = P_Q \cdot \rho (1 - \rho) + k\rho
\]

:p What is the formula to calculate the expected total number of jobs in an M/M/k system?
??x
The expected total number of jobs \(E[N]\) in an M/M/k system can be calculated as:
\[
E[N] = E[N_Q] + E[\text{number being served}] = P_Q \cdot \rho (1 - \rho) + k\rho
\]
where \(E[N_Q]\) is the expected number of jobs in the queue, and \(k\rho\) is the expected number of busy servers.

x??

---

#### FDM vs M/M/1 Comparison
Background context explaining the concept. The problem involves comparing the mean response times of two server organizations: Frequency-Division Multiplexing (FDM) and M/M/1 under the same load conditions.

:p Which configuration is better for minimizing mean response time between FDM and M/M/1?
??x
The M/M/1 system has a lower mean response time compared to FDM. This is because in the M/M/1 setup, each job experiences both a higher arrival rate and a higher service rate due to the single server handling all jobs simultaneously. Mathematically, this can be shown by comparing their respective formulas for mean response times.

Formula for Mean Response Time under FDM:
\[ E[T]_{\text{FDM}} = \frac{1}{\mu - \lambda k} = \frac{k}{k\mu - \lambda} \]

Formula for Mean Response Time under M/M/1:
\[ E[T]_{\text{M/M/1}} = \frac{1}{k\mu - \lambda} \]

Since \( E[T]_{\text{FDM}} = k \times E[T]_{\text{M/M/1}} \), the M/M/1 system is k times better than FDM.
??x

---

#### M/M/k Comparison
Background context explaining the concept. This involves understanding how the mean response time changes in an M/M/k configuration under the same load conditions.

:p How does the M/M/1 system compare with the M/M/k system?
??x
In the M/M/k setup, the traffic is lumped together but the service capacity is split among k servers. The mean response time for an M/M/k system can be given by:
\[ E[T]_{\text{M/M/k}} = \frac{1}{\lambda \cdot P_Q \cdot \rho (1 - \rho) + \frac{\mu}{k}} \]
where \( \rho = \frac{\lambda k \mu}{k} \), and \( P_Q \) is the probability that an arrival has to queue.

For comparison, M/M/1 involves a single server handling all jobs. The mean response time for M/M/1 can be derived from:
\[ E[T]_{\text{M/M/1}} = \frac{1}{k\mu - \lambda} \]

The exact comparison would depend on the value of \( P_Q \), but generally, as k increases (i.e., more servers are available to handle the load), the mean response time can potentially decrease due to better utilization and lower queuing probabilities.

To understand this concept, consider that with more servers in M/M/k, jobs may experience shorter queues and thus a faster service. However, the exact improvement depends on \( P_Q \).
??x

