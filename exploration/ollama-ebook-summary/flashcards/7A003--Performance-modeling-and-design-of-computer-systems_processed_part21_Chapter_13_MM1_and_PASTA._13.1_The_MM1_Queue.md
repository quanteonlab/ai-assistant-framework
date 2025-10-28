# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 21)

**Starting Chapter:** Chapter 13 MM1 and PASTA. 13.1 The MM1 Queue

---

#### Rate of Transitions Leaving State 1 to Go to State 2
Background context: In an M/M/1 queue, the rate of transitions leaving state 1 (which means having one customer) is determined by the arrival rate λ. The system moves from one state to a higher state with this rate.

:p What is the rate of transitions leaving state 1?
??x
The rate of transitions leaving state 1 is π₁λ, where π₁ is the probability that there is exactly one customer in the system.
x??

---

#### Balance Equations for M/M/1 CTMC
Background context: The balance equations are used to find the steady-state probabilities πᵢ for each state i in an M/M/1 queue. These equations ensure that the rate at which the system leaves a state is equal to the rate at which it enters the state.

:p What are the full balance equations for the M/M/1 CTMC?
??x
The full balance equations are:
- For state 0: π₀λ = π₁μ → π₁ = (λ/μ)π₀
- For state 1: π₁(λ + μ) = π₀λ + π₂μ → π₂ = (λ/μ)²π₀
- For state i: πᵢ(λ + μ) = πᵢ₋₁λ + πᵢ₊₁μ

We guess that πᵢ = (λ/μ)ⁱπ₀.
x??

---

#### Determining π₀ for the M/M/1 CTMC
Background context: To find the steady-state probabilities, we need to normalize them so their sum equals 1. This is done by solving the equation ∑ₐᵦ₌₀⁰̣∞ πᵢ = 1.

:p How do you determine π₀ in an M/M/1 CTMC?
??x
We set up the normalization condition:
\[ \sum_{i=0}^{\infty} \pi_i = 1 \]

Substituting our guess for πᵢ into this equation, we get:
\[ \sum_{i=0}^{\infty} \left(\frac{\lambda}{\mu}\right)^i \pi_0 = 1 \]

This is a geometric series with the sum given by:
\[ \pi_0 \cdot \frac{1}{1 - \frac{\lambda}{\mu}} = 1 \]

Solving for π₀, we get:
\[ \pi_0 = 1 - \frac{\lambda}{\mu} = 1 - \rho \]
where ρ = λ/μ is the server utilization.

x??

---

#### Server Utilization and Stability Condition
Background context: The server utilization ρ measures how busy the server is. For the system to be stable, it must not grow without bound. This requires that the arrival rate λ does not exceed the service rate μ.

:p What is the condition for stability in an M/M/1 queue?
??x
The stability condition for an M/M/1 queue is that ρ < 1, which means λ < μ.
x??

---

#### Mean Number of Customers in the System
Background context: The mean number of customers N in the system can be derived by summing over all states i, weighted by their probabilities πᵢ.

:p How do you derive the mean number of customers E[N] in an M/M/1 queue?
??x
The mean number of customers \( E[N] \) is given by:
\[ E[N] = \sum_{i=0}^{\infty} i \pi_i \]

Since \( \pi_0 = 1 - \rho \), we only need to sum from state 1 onwards:
\[ E[N] = \sum_{i=1}^{\infty} i \pi_i = (1 - \rho) \sum_{i=1}^{\infty} i \left(\frac{\lambda}{\mu}\right)^i \]

This can be simplified using the formula for the sum of a series:
\[ E[N] = \rho(1 - \rho) \cdot \frac{d}{d\rho} \sum_{i=0}^{\infty} \left(\frac{\lambda}{\mu}\right)^i \]

The derivative simplifies to:
\[ E[N] = \rho(1 - \rho) \cdot \frac{d}{d\rho} \left( \frac{1}{1 - \rho} \right) = \rho(1 - \rho) \cdot \frac{1}{(1 - \rho)^2} = \frac{\rho}{1 - \rho} \]

x??

---

#### Mean Number of Customers in M/M/1 Queue

Background context: In an M/M/1 queue, the mean number of customers in the system \(E[N]\) is given by \(\frac{\rho}{(1-\rho)}\), where \(\rho\) is the traffic intensity (arrival rate divided by service rate).

Figure 13.3 plots this relationship between \(E[N]\) and \(\rho\). For small values of \(\rho\) (less than 0.5 or even 0.6), the increase in the mean number of customers is minimal, but as \(\rho\) increases beyond these points, the mean number can rise sharply.

:p What does Figure 13.3 illustrate about \(E[N]\) and \(\rho\)?
??x
The figure illustrates that for small values of \(\rho\), the increase in the expected number of customers is minimal. However, as \(\rho\) approaches or exceeds certain thresholds (around 0.5 to 0.6), the mean number of customers in the system starts increasing rapidly.
x??

---

#### Variance of Number of Customers in M/M/1 Queue

Background context: The variance of the number of customers in an M/M/1 queue is given by \(\text{Var}(N) = \frac{\rho}{(1-\rho)^2}\), which grows even more sharply than the mean number of jobs.

:p What is the formula for the variance of \(N\) in an M/M/1 queue?
??x
The formula for the variance of \(N\) in an M/M/1 queue is \(\text{Var}(N) = \frac{\rho}{(1-\rho)^2}\).
x??

---

#### Mean and Expected Time in System Using Little's Law

Background context: The mean time in the system \(E[T]\) and the expected number of customers \(E[N]\) are related by Little’s Law. Specifically, for an M/M/1 queue:

\[ E[T] = \frac{E[N]}{\lambda} = \frac{1}{\mu - \lambda} \]

Similarly, the mean time in the queue \(E[TQ]\) is given by:

\[ E[TQ] = E[T] - \frac{1}{\mu} = \frac{\rho}{\mu(1-\rho)} \]

:p What does Little's Law provide for the relationship between \(E[T]\), \(E[N]\), and \(\lambda\)?
??x
Little's Law provides that the mean time in the system \(E[T]\) is equal to the expected number of customers \(E[N]\) divided by the arrival rate \(\lambda\):

\[ E[T] = \frac{E[N]}{\lambda} \]

This can also be expressed as:

\[ E[T] = \frac{1}{\mu - \lambda} \]
x??

---

#### Impact of Increasing Arrival and Service Rates Proportionally

Background context: If the arrival rate \(\lambda\) and service rate \(\mu\) in an M/M/1 system are increased by a factor \(k\), then:

\[ \rho_{\text{new}} = \frac{\lambda_{\text{new}}}{\mu_{\text{new}}} = \frac{k\lambda}{k\mu} = \rho_{\text{old}} \]

Thus, the system utilization remains unchanged. The throughput is increased by a factor of \(k\), and the mean number of jobs in the system stays the same.

The expected time in the system also decreases proportionally:

\[ E[T_{\text{new}}] = \frac{1}{k(\mu - \lambda)} = \frac{E[T_{\text{old}}]}{k} \]

:p How is the mean response time affected when both arrival and service rates are increased by a factor of \(k\)?
??x
The mean response time decreases by a factor of \(k\) when both the arrival rate and service rate are increased by a factor of \(k\).

This can be seen from:

\[ E[T_{\text{new}}] = \frac{1}{k(\mu - \lambda)} = \frac{E[T_{\text{old}}]}{k} \]
x??

---

#### Statistical Multiplexing vs. Frequency-Division Multiplexing

Background context: In the case of \(m\) independent Poisson packet streams, each with an arrival rate of \(\frac{\lambda}{m}\) packets per second and an exponentially distributed service time with mean \(\mu\), we analyze two multiplexing techniques:

1. **Statistical Multiplexing (SM):** Merges all the streams into one stream.
2. **Frequency-Division Multiplexing (FDM):** Keeps the \(m\) streams separate and divides the transmission capacity into \(m\) equal portions.

:p What is the expected time in the system for Statistical Multiplexing (SM)?
??x
The expected time in the system for Statistical Multiplexing (SM) is given by:

\[ E[T_{\text{SM}}] = \frac{1}{\mu - \lambda} \]
x??

---

#### Frequency-Division Multiplexing vs. Statistical Multiplexing

Background context: The passage explains the differences between frequency-division multiplexing (FDM) and statistical multiplexing, focusing on the guarantees they offer regarding service rates.

:p Why would one use FDM over statistical multiplexing?
??x
Frequency-Division Multiplexing guarantees a specific service rate to each stream. Statistical multiplexing cannot provide such a guarantee. In scenarios where applications require low variability in delay (e.g., voice or video), merging streams can introduce more variability, making FDM preferable.

```java
// Example of setting up FDM channels
public class FdmChannel {
    private int allocatedBandwidth; // Bandwidth assigned to each stream

    public void allocateChannel(int bandwidth) {
        this.allocatedBandwidth = bandwidth;
    }

    public int getAllocatedBandwidth() {
        return this.allocatedBandwidth;
    }
}
```
x??

---

#### PASTA - Poisson Arrivals See Time Averages

Background context: The passage introduces the concept of "Poisson Arrivals See Time Averages" (PASTA) and explains how it applies to systems with a Poisson arrival process.

:p What does PASTA state?
??x
PASTA states that an = pn when the arrivals follow a Poisson process, meaning the fraction of arrivals that see n jobs in the system is equal to the long-run fraction of time that there are n jobs in the system.

```java
// Example simulation logic for tracking job counts over time
public class PastaSimulation {
    private int currentJobCount; // Current number of jobs in the system

    public void simulate() {
        while (true) { // Simulate indefinitely
            updateJobCount(); // Update based on arrivals and departures
            trackAvgJobsInSystem(currentJobCount); // Track average over time
            sleepForSimulationTick(); // Move to next simulation tick
        }
    }

    private void updateJobCount() {
        // Logic for updating job count based on new arrivals or departures
    }

    private void trackAvgJobsInSystem(int currentCount) {
        // Logic for tracking the average number of jobs seen by an arrival
    }

    private void sleepForSimulationTick() {
        // Sleep to simulate passage of time in simulation
    }
}
```
x??

---

#### Relationship Between \(a_n\) and \(p_n\)

Background context: The text discusses why \(a_n = p_n\) does not always hold true, especially for non-Poisson arrival processes.

:p Why is \(a_n \neq p_n\) in general?
??x
In general, \(a_n \neq p_n\) because the fraction of arrivals that see n jobs (an) and the long-run fraction of time with n jobs (pn) are not necessarily equal. This inequality arises from the variability introduced when merging streams or using non-Poisson arrival processes.

```java
// Example logic to simulate job counts over time for a non-Poisson process
public class NonPoissonSimulation {
    private int currentJobCount; // Current number of jobs in the system

    public void simulate() {
        while (true) { // Simulate indefinitely
            updateJobCount(); // Update based on arrivals and departures
            trackAvgJobsInSystem(currentJobCount); // Track average over time
            sleepForSimulationTick(); // Move to next simulation tick
        }
    }

    private void updateJobCount() {
        // Logic for updating job count based on new arrivals or departures (non-Poisson)
    }

    private void trackAvgJobsInSystem(int currentCount) {
        // Logic for tracking the average number of jobs seen by an arrival
    }

    private void sleepForSimulationTick() {
        // Sleep to simulate passage of time in simulation
    }
}
```
x??

---

#### Detailed Proof of PASTA

Background context: The text provides a detailed proof showing that \(a_n = p_n\) for Poisson processes.

:p Why does the PASTA proof not hold for uniform arrival processes with deterministic service times?
??x
The PASTA proof assumes independence between arrivals and job counts, which fails when interarrival times are uniformly distributed and service times are deterministic. In such a scenario, knowing N(t) affects whether there will be an arrival in the next δ seconds.

```java
// Example logic to simulate non-Poisson process with uniform interarrivals and deterministic service
public class NonPoissonUniformServiceSimulation {
    private int currentJobCount; // Current number of jobs in the system

    public void simulate() {
        while (true) { // Simulate indefinitely
            updateJobCount(); // Update based on arrivals and departures
            trackAvgJobsInSystem(currentJobCount); // Track average over time
            sleepForSimulationTick(); // Move to next simulation tick
        }
    }

    private void updateJobCount() {
        // Logic for updating job count with uniform interarrivals and deterministic service times
    }

    private void trackAvgJobsInSystem(int currentCount) {
        // Logic for tracking the average number of jobs seen by an arrival
    }

    private void sleepForSimulationTick() {
        // Sleep to simulate passage of time in simulation
    }
}
```
x??

---

#### Independence Assumption in PASTA

Background context: The text explains why the independence assumption between interarrival times and service times is necessary for PASTA.

:p Why might we need the further assumption that interarrival times and service times are independent?
??x
We need this assumption because Poisson arrivals do not guarantee such independence. For example, if service times depend on interarrival times (as in the hypothetical scenario), an arrival's occurrence can influence the state of the system differently than what PASTA assumes.

```java
// Example logic to illustrate dependent interarrivals and service times
public class DependentInterArrivalServiceSimulation {
    private int currentJobCount; // Current number of jobs in the system

    public void simulate() {
        while (true) { // Simulate indefinitely
            updateJobCount(); // Update based on arrivals and departures
            trackAvgJobsInSystem(currentJobCount); // Track average over time
            sleepForSimulationTick(); // Move to next simulation tick
        }
    }

    private void updateJobCount() {
        // Logic for updating job count with dependent interarrivals and service times
    }

    private void trackAvgJobsInSystem(int currentCount) {
        // Logic for tracking the average number of jobs seen by an arrival
    }

    private void sleepForSimulationTick() { // Sleep to simulate passage of time in simulation
    }
}
```
x??

---

