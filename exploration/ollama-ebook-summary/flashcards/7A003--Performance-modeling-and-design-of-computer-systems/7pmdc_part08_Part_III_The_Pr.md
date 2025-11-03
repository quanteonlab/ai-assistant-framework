# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 8)

**Starting Chapter:** Part III The Predictive Power of Simple Operational Laws What-If Questions and Answers

---

#### Operational Laws and Their Importance
Background context: The text introduces operational laws as a powerful tool for analyzing system behavior. These laws are simple, exact, and "distribution independent," meaning they do not depend on specific distributions of job service requirements or interarrival times but only on their means. This makes them very popular among system builders.

:p What is the significance of distribution independence in operational laws?
??x
Distribution independence means that these laws can be applied universally to any system or part of a system, as long as certain statistical measures (like mean values) are known. This flexibility allows for accurate predictions without needing detailed information about job sizes or arrival times.

Example: If you know the average service time and arrival rate, you can use operational laws to estimate the performance of a system, regardless of how the individual jobs vary in size.
x??

#### Little's Law
Background context: Little’s Law is one of the most important operational laws discussed. It relates the mean number of jobs in any system (L) to the mean response time experienced by arrivals (W), expressed as \( L = \lambda W \).

:p What does Little's Law relate?
??x
Little's Law relates the mean number of jobs in a system (L) to the mean response time experienced by arrivals (W). The relationship is given by \( L = \lambda W \), where \(\lambda\) represents the arrival rate.
x??

#### Application of Operational Laws
Background context: Operational laws are particularly useful for "what-if" questions, such as determining which system modification would be more beneficial. For example, deciding whether increasing CPU speed or I/O device speed is more advantageous.

:p How do operational laws help in answering "what-if" questions?
??x
Operational laws provide a framework to analyze the impact of changes on system performance without detailed modeling. By understanding how different parameters affect response time and throughput, you can determine which modifications are likely to yield better results.

Example: If increasing CPU speed by a factor of 2 decreases the service time (\(1/\mu\)), while increasing I/O device speed by a factor of 3 only slightly reduces interarrival times or increases service rates, Little's Law helps evaluate whether the overall response time and throughput improve significantly.
x??

#### Asymptotic Bounds
Background context: Asymptotic bounds are used to analyze system behavior under extreme conditions. Specifically, they provide insights into how systems behave as the multiprogramming level approaches infinity or 1.

:p What do asymptotic bounds help determine?
??x
Asymptotic bounds help determine the long-term performance of a system as the number of processes (multiprogramming level) becomes very large or very small. This is useful for understanding the limits and trade-offs in system design.
x??

#### Proving Bounds Using Operational Laws
Background context: Chapter 7 focuses on using operational laws to prove asymptotic bounds, such as mean response time and throughput, for closed systems.

:p How does one use operational laws to prove bounds?
??x
By combining multiple operational laws, one can derive formulas that describe system behavior under various conditions. For example, Little's Law combined with other operational principles can help prove how changes in parameters like service times or arrival rates affect the overall performance metrics of a closed system.

Example: Using operational laws, you might derive an expression for mean response time \( W \) as a function of multiprogramming level \( n \), and then take limits to understand its behavior at very high or low values of \( n \).
x??

--- 

These flashcards cover key concepts from the provided text related to operational laws, Little's Law, and their applications in system analysis.

#### Little's Law for Open Systems
Background context explaining the concept. Little's Law states that the average number of jobs \(E[N]\) in a system is equal to the product of the average arrival rate into the system \(\lambda\) and the mean time jobs spend in the system \(E[T]\). Mathematically, this can be expressed as:
\[ E[N] = \lambda E[T] \]

This law applies to both open and closed systems and holds true regardless of assumptions about the arrival process, service time distributions, network topology, or service order. The setup for Little's Law involves considering a system with jobs arriving at an average rate \(\lambda\) and spending some amount of time \(T\) in the system.

:p What does Little's Law state?
??x
Little's Law states that the average number of jobs in a system is equal to the product of the average arrival rate into the system and the mean time jobs spend in the system.
x??

---
#### Ergodic Open Systems
Background context explaining the concept. An ergodic open system refers to systems where, over an extended period, all possible states are visited and the long-term behavior of the system is predictable based on the steady-state probabilities.

The theorem for Little's Law in such a system can be expressed as:
\[ E[N] = \lambda E[T] \]
where \(E[N]\) is the expected number of jobs in the system, \(\lambda\) is the average arrival rate into the system, and \(E[T]\) is the mean time jobs spend in the system.

:p What does ergodicity imply about open systems?
??x
Ergodicity implies that over an extended period, all possible states are visited and the long-term behavior of the system can be analyzed using steady-state probabilities.
x??

---
#### Application of Little's Law
Background context explaining the concept. The application of Little's Law involves leveraging known quantities (such as \(E[N]\) or \(\lambda\)) to find unknowns (\(E[T]\)) in queueing systems.

:p How can we use Little's Law to find unknown values in a system?
??x
We can use Little's Law by rearranging the formula. For example, if you know \(E[N]\) and \(\lambda\), you can find \(E[T]\) as follows:
\[ E[T] = \frac{E[N]}{\lambda} \]
Similarly, if you know \(E[T]\) and \(\lambda\), you can find \(E[N]\):
\[ E[N] = \lambda E[T] \]

This law is particularly useful in network analysis and system design.
x??

---
#### Open System Setup
Background context explaining the concept. The setup for Little's Law involves a system with arrivals at an average rate \(\lambda\), departures, and jobs spending time \(T\) in the system.

The diagram provided (Figure 6.1) shows:
- Arrivals (rate \(\lambda\))
- Departures
- Time in system, \(T\)

:p What elements are involved in the setup for Little's Law?
??x
The elements involved in the setup for Little's Law include arrivals at an average rate \(\lambda\), departures from the system, and the time jobs spend in the system, denoted as \(T\).
x??

---
#### Open Systems Example
Background context explaining the concept. To illustrate the use of Little's Law, consider a simple example where you have a queueing system with an arrival rate \(\lambda = 10\) jobs per minute and the average time spent in the system is \(E[T] = 2\) minutes.

:p Calculate \(E[N]\) using Little's Law.
??x
Given \(\lambda = 10\) jobs/minute and \(E[T] = 2\) minutes, we can calculate \(E[N]\) as follows:
\[ E[N] = \lambda E[T] = 10 \times 2 = 20 \text{ jobs} \]

Thus, the expected number of jobs in the system is 20.
x??

---
#### Open Systems and Markov Chains
Background context explaining the concept. When studying Markov chains, many techniques are used to compute \(E[N]\). Applying Little's Law will then immediately yield \(E[T]\).

:p How does applying Little's Law help when studying Markov chains?
??x
Applying Little's Law helps by providing a straightforward way to find \(E[T]\) once \(E[N]\) and \(\lambda\) are known. If you have computed \(E[N]\) using techniques from Markov chains, you can directly use the formula:
\[ E[T] = \frac{E[N]}{\lambda} \]
This simplifies the process of finding the mean time jobs spend in the system.
x??

#### Little's Law Intuition
Background context explaining the intuitive understanding of Little’s Law. Consider a fast-food restaurant where E[T] (average time spent by a customer) and E[N] (average number of customers waiting for service) are inversely proportional. A faster service means lower E[T], while fewer seats result in lower E[N].
:p What is an intuitive way to remember the relationship between E[T] and E[N]?
??x
E[T] should be directly proportional to E[N]. This can be understood by imagining a single FCFS queue where a customer sees E[N] jobs upon arrival, and each job takes approximately 1/λ time to complete on average.
x??

---
#### Little's Law for Open Systems - Statement via Time Averages
Background context explaining the statement of Little’s Law for open systems in terms of time averages. This involves defining λ as the limit of the number of arrivals by time t divided by t, and X as the limit of the number of system completions (departures) by time t divided by t.
:p What is the formal statement of Little's Law for open systems using time averages?
??x
For any system where NTime Avg, TTime Avg, λ, and X exist and where λ = X, then NTime Avg = λ · TTime Avg. This equality relates the time-average number in the system to the product of the arrival rate (λ) and the time-average time in the system.
x??

---
#### Little's Law Proof for Open Systems
Background context explaining the proof of Little’s Law for open systems by considering the area under a graph representing arrivals and departures. The proof involves summing up times spent in the system and comparing it with the number of jobs in the system at any moment.
:p How is Little's Law proved for an open system?
??x
The proof uses the area A formed by rectangles representing time spent in the system. By summing horizontally, we get ∑ Ti ≤ ∫ N(s) ds ≤ ∑ Ti where Ti are times in the system and N(s) represents jobs in the system at any moment s. Dividing by t throughout and taking limits as t → ∞, we get TTime Avg · X = NTime Avg.
x??

---
#### Utilization Law
Background context explaining the concept of device utilization or load. Given a single device with arrival rate λ and service rate μ where λ < μ, the long-run fraction (ρ) that the device is busy can be derived using Little’s Law.
:p How does Little's Law help prove the Utilization Law?
??x
By defining the system as just the service facility without the queue, we find that the number of jobs in this "system" is always 0 or 1. The long-run fraction ρi = λi / μi represents both the busy time and the limiting probability that the device is busy.
x??

---
#### Corollary for Time in Queue
Background context explaining Little's Law application to the time spent in queues. This involves defining NQ as the number of jobs in queue and TQ as the time jobs spend in queues, similar to the main theorem but focusing on queuing times.
:p How can we prove the corollary related to time in queue?
??x
The proof is similar to Theorem 6.3, where Ti are replaced by TQ(i), representing the time the ith arrival spends in queues. The logic involves summing up wasted time and comparing it with the number of jobs in the system at any moment.
x??

---
#### Difference Between Time Averages and Ensemble Averages
Background context explaining the distinction between time averages and ensemble averages, particularly under ergodicity assumptions. This helps in applying Little’s Law to systems where the long-term behavior is consistent across all paths.
:p What does it mean for a system to be ergodic?
??x
Ergodicity means that the time average converges to the ensemble average with probability 1. In simpler terms, on almost every sample path, the observed averages will match the theoretical (ensemble) averages over all possible paths in the system.
x??

---
#### Application of Little's Law to Systems with Service Orders and Multiple Servers
Background context explaining how Little’s Law applies regardless of service order or number of servers. This involves understanding that the law holds for any system configuration, be it FCFS or other scheduling policies.
:p Does Little's Law depend on the specific service order or number of servers?
??x
No, Little’s Law does not depend on the specific service order (e.g., FCFS) or the number of servers. The proof shows that the relationship between average time in system and average number in system holds regardless of these factors.
x??

---

