# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 21)

**Starting Chapter:** Chapter 13 MM1 and PASTA. 13.1 The MM1 Queue

---

#### M/M/1 Queueing System Overview
Background context: The simplest queueing model consists of a single server with Exponentially distributed service times and Poisson-distributed interarrival times. This system is denoted as M/M/1, where "M" stands for memoryless (Exponential distribution) in both the arrival process and service times, and the first slot describes the number of servers (1 in this case).
:p What is an M/M/1 queueing system?
??x
An M/M/1 queueing system consists of a single server where customers arrive according to a Poisson process with rate λ and are served with Exponential service times having mean 1/μ. The term "M" indicates that both the arrival and service processes follow an exponential distribution, implying memoryless properties.
x??

---

#### Birth-Death Process
Background context: In an M/M/1 system, the states of the queue form a birth-death process where state transitions only occur between consecutive states. The rate at which the system leaves state $j $ is denoted as$μ_j $, and the rate at which it enters state $ j+1 $ is denoted as $λ_j$.
:p What does the term "birth" represent in a birth-death process?
??x
In a birth-death process, the term "birth" represents an increase in the number of customers in the system. In the context of M/M/1, this corresponds to new customer arrivals.
x??

---

#### Balance Equations for State 1
Background context: To find the steady-state probabilities $π_j $ for each state$j$, balance equations are used. These equate the rate at which the system leaves a state with the rate at which it enters that state.
:p What is the balance equation for state 1?
??x
The balance equation for state 1 in an M/M/1 queueing system is given by:
$$π_1(λ + μ) = π_0 λ + π_2 μ$$

This equation balances the rate at which customers leave state 1 (both to state 0 and to state 2) with the rate at which they enter state 1.
x??

---

#### General Form of Steady-State Probabilities
Background context: The steady-state probabilities $π_j $ for states in an M/M/1 queueing system are derived by assuming a general form and then solving balance equations. This involves finding expressions for$π_j$ that satisfy the balance equations.
:p What is the assumed form of $π_i$ for state i?
??x
The assumed form of $π_i$ for state i in an M/M/1 queueing system is:
$$π_i = \left(\frac{λ}{μ}\right)^i (1 - ρ)$$where $ρ = \frac{λ}{μ}$ is the server utilization.
x??

---

#### Determining π₀
Background context: The value of $π_0$ must be determined so that the sum of all probabilities equals 1. This involves solving a geometric series.
:p How do you determine the value of $π_0$?
??x
To determine $π_0$, we use the normalization condition:
$$\sum_{i=0}^{\infty} π_i = 1$$

Given that $π_i = \left(\frac{λ}{μ}\right)^i (1 - ρ)$, we have:
$$\sum_{i=0}^{\infty} \left(\frac{λ}{μ}\right)^i (1 - ρ) = 1$$

This simplifies to:
$$(1 - ρ) \sum_{i=0}^{\infty} \left(\frac{λ}{μ}\right)^i = 1$$

The sum of the infinite geometric series is:
$$\sum_{i=0}^{\infty} \left(\frac{λ}{μ}\right)^i = \frac{1}{1 - \frac{λ}{μ}} = \frac{1}{1 - ρ}$$

Thus, we get:
$$(1 - ρ) \cdot \frac{1}{1 - ρ} = 1$$

Therefore,$$π_0 = \frac{1}{1 - ρ}$$x??

---

#### Mean Number of Customers in the System
Background context: The mean number of customers $E[N]$ can be derived by conditioning on the state. This involves summing over all states, weighted by their probabilities.
:p How do you calculate the mean number of customers $E[N]$?
??x
The mean number of customers $E[N]$ in an M/M/1 queueing system is calculated as:
$$E[N] = \sum_{i=0}^{\infty} i π_i$$

Substituting $π_i = \left(\frac{λ}{μ}\right)^i (1 - ρ)$:
$$E[N] = \sum_{i=1}^{\infty} i \left(\frac{λ}{μ}\right)^i (1 - ρ)$$

This can be simplified using the formula for the sum of a geometric series:
$$

E[N] = \rho + 2ρ(1 - ρ)\left(\frac{\lambda}{\mu}\right) + 3ρ(1 - ρ)\left(\frac{\lambda}{\mu}\right)^2 + \ldots$$

Recognizing this as the derivative of a geometric series:
$$

E[N] = \rho \sum_{i=0}^{\infty} i \left(\frac{λ}{μ}\right)^i (1 - ρ) = \rho \cdot \frac{d}{dρ} \left( \sum_{i=0}^{\infty} \left(\frac{λ}{μ}\right)^i (1 - ρ) \right)$$
$$

E[N] = \rho \cdot \frac{d}{dρ} \left( \frac{1 - ρ}{1 - \frac{λ}{μ}} \right) = \rho \cdot \frac{1}{\left(1 - \frac{λ}{μ}\right)^2} = \frac{ρ^2}{1 - ρ}$$x??

---

#### M/M/1 Queue Mean Number of Customers

Background context: The M/M/1 queue is a fundamental model for single-server queuing systems where arrivals follow a Poisson process and service times are exponentially distributed. The utilization factor $\rho $ represents the ratio of arrival rate$\lambda $ to service rate$\mu $. A key property of this system is the expected number of customers in the system, which can be derived using the formula $ E[N] = \frac{\rho}{1 - \rho}$.

:p What does the equation for the mean number of customers in an M/M/1 queue represent?
??x
The equation $E[N] = \frac{\rho}{1 - \rho}$ represents the expected number of customers in the system, including both those being served and waiting. This relationship highlights how increasing the utilization factor $\rho$ can dramatically affect the mean number of customers.
x??

---

#### Variance of Number of Customers

Background context: The variance of the number of customers in an M/M/1 queue is given by $Var(N) = \frac{\rho}{(1 - \rho)^2}$. This measure provides insight into the variability or spread of the customer count around its mean.

:p What does the formula for the variance of the number of customers represent?
??x
The formula $Var(N) = \frac{\rho}{(1 - \rho)^2}$ represents the variance in the number of customers present in the M/M/1 queue. It shows that as $\rho$ increases, the variance grows more sharply than the mean, indicating a higher likelihood of having significantly more or fewer customers than the average.
x??

---

#### Little’s Law and Mean Response Time

Background context: Little's Law is a fundamental principle stating that the mean number of items in a system $E[N]$ equals the arrival rate $\lambda$ multiplied by the mean time an item spends in the system $E[T]$. For an M/M/1 queue, this can be expressed as:
$$E[N] = \frac{\rho}{1 - \rho}$$and$$

E[T] = \frac{1}{\mu - \lambda}$$:p How do we use Little's Law to find the mean time in system?
??x
Using Little's Law, we can find the mean time an item spends in the system by calculating $E[T] = \frac{E[N]}{\lambda}$. Given that for an M/M/1 queue,$ E[N] = \frac{\rho}{1 - \rho}$, it follows that:
$$E[T] = \frac{1}{\mu - \lambda}$$

This relationship shows how the mean time in the system is inversely related to the difference between the service rate and arrival rate.
x??

---

#### Impact of Increasing Arrival and Service Rates

Background context: If both the arrival rate $\lambda $ and service rate$\mu $ are increased by a factor$k $, the utilization $\rho$ remains unchanged, but throughput is increased. The mean number of customers in the system also stays constant, while the mean response time decreases proportionally.

:p What happens to the throughput when both arrival and service rates are increased proportionally?
??x
When both arrival rate $\lambda $ and service rate$\mu $ are increased by a factor$k $, the throughput is increased by the same factor $ k$. This is because:
$$X_{new} = k \cdot X_{old}$$where $ X$ represents the throughput.

This result explains why increasing both arrival and service rates can accommodate more traffic with less delay per packet.
x??

---

#### Statistical Multiplexing vs. Frequency-Division Multiplexing

Background context: In statistical multiplexing (SM), multiple independent Poisson streams are merged into a single stream, modeled as an M/M/1 queue. In frequency-division multiplexing (FDM), these streams remain separate but share the transmission capacity equally.

:p How does the mean response time compare between statistical and frequency-division multiplexing?
??x
The mean response time for statistical multiplexing is given by $E[T_{SM}] = \frac{1}{\mu - \lambda}$. This means that the mean time in the system for SM is simply the reciprocal of the difference between the service rate and arrival rate.

In comparison, frequency-division multiplexing (FDM) would involve analyzing each stream separately and summing their individual contributions to the overall response time. However, due to the independent nature of Poisson processes, the combined effect on mean response time for SM is equivalent.
x??

---

#### Frequency-Division Multiplexing (FDM) vs. Statistical Multiplexing

Background context: The text discusses frequency-division multiplexing (FDM) and statistical multiplexing, focusing on their differences and use cases.

:p Why would one ever use FDM?

??x
Frequency-Division Multiplexing guarantees a specific service rate to each stream, which is not possible with Statistical Multiplexing. Additionally, merging regular streams into irregular ones can introduce variability that might be problematic for applications requiring low delay variability, such as voice or video.

---

#### PASTA (Poisson Arrivals See Time Averages)

Background context: The concept of PASTA deals with the relationship between the state of the system seen by an arrival and its long-run average state. It is particularly useful in simulations to determine the fraction of time that the system has a certain number of jobs.

:p Is $a_n = p_n$?

??x
No, according to Claim 13.1, $a_n $(the probability that an arrival sees $ n $jobs) is not necessarily equal to$ p_n $(the limiting probability that there are$ n $jobs in the system). However, the average time spent by a job in the system (response time) and the probability of response time exceeding$ x$ are defined as the same for both arrivals and departures.

---

#### Proof of PASTA

Background context: The proof of PASTA involves showing that the probability an arrival sees $n $ jobs ($a_n $) is equal to the limiting probability that there are$ n $jobs in the system ($ p_n$), under a Poisson arrival process.

:p Is $a_n = d_n$?

??x
Yes, according to Claim 13.2, when customers arrive one at a time and are served one at a time, then $a_n = d_n $. This is because both arrivals seeing $ n $jobs and departures leaving behind$ n$ jobs happen an equal number of times.

---

#### Example of Uniform Interarrival Times with Deterministic Service Times

Background context: The example provided illustrates why the proof of PASTA would not hold for a uniform interarrival process with deterministic service times. In such cases, knowing $N(t)$(number of jobs in the system at time $ t$) affects whether there will be an arrival in the next $\delta$ seconds.

:p Why wouldn’t this proof go through for the example of Uniform interarrival times and Deterministic service times?

??x
In a scenario where interarrival times are uniformly distributed between 1 and 2, and service times are deterministic (equal to 1), knowing $N(t)$ affects whether there will be an arrival in the next $\delta$ seconds. Specifically, if $N(t) = 1$, then there won't be an arrival in the next $\delta$ seconds. This dependency breaks the independence required for the proof of PASTA.

---

#### Independence Assumption

Background context: The independence assumption is crucial for the proof of PASTA. It ensures that knowing when an arrival occurs tells us nothing about $N(t)$ and vice versa.

:p Why might we need to make the further assumption (stated in the footnote to Claim 13.3) that the interarrival times and service times are independent?

??x
The independence assumption is necessary because if interarrival times and service times are not independent, as in a perverse scenario where the service time of the $n $ th arrival equals half the interarrival time between packets$n $ and$n+1$, then an arrival finding the system empty would be contradictory. This hypothetical situation highlights why independence is required for PASTA to hold.

---

#### Application of PASTA in Simulation

Background context: The application of PASTA in simulations allows us to determine the mean number of jobs in a system by observing arrivals, as Poisson arrivals see time averages.

:p How can PASTA be useful in system simulations?

??x
PASTA is useful in system simulations because it helps estimate the state of the system from an arrival's perspective. By tracking the fraction of arrivals that witness $n$ jobs, we can approximate the long-run average number of jobs in the system without needing to simulate every job individually.

---
---

