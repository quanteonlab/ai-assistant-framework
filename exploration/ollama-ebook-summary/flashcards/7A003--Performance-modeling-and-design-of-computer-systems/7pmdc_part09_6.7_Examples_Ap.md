# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 9)

**Starting Chapter:** 6.7 Examples Applying Littles Law

---

#### Little's Law for Closed Systems - General Overview
Little’s Law applies to closed systems where there are no exogenous arrivals, and jobs generate themselves within the system. It states that the number of jobs (N) in a system over time is equal to the throughput (X) times the mean time each job spends in the system (TTime Avg). The formula can be expressed as \( N = X \cdot T_{\text{TimeAvg}} \).

:p Can you explain Little's Law for closed systems?
??x
Little’s Law for closed systems states that the number of jobs in a system over time is equal to the throughput (X) times the mean time each job spends in the system (\(T_{\text{TimeAvg}}\)). This can be expressed as \( N = X \cdot T_{\text{TimeAvg}} \). In simpler terms, it means that the average number of jobs in a closed system is equal to the rate at which jobs leave (throughput) multiplied by the time each job spends in the system.
x??

---

#### Little's Law for "Red" Jobs
Little’s Law can be applied specifically to subsets of jobs within a system. For "red" jobs, the same law holds but applies only to those specific jobs.

:p Can we apply Little’s Law just to "red" jobs?
??x
Yes, Little's Law can be applied to just "red" jobs. The formula is \( E[\text{Number of red jobs in system}] = \lambda_{\text{red}} \cdot E[\text{Time spent by red jobs in the system}] \). Here, \(\lambda_{\text{red}}\) represents the average arrival rate of "red" jobs, and \(E[\text{Time spent by red jobs in the system}]\) is the mean time these specific jobs spend within the system.
x??

---

#### Throughput Law for Closed Systems
The Throughput Law (also known as Response Time Law) states that the throughput (\(X\)) of a closed system can be calculated using \( X = \frac{N}{E[R] - E[Z]} \), where \( N \) is the number of users, \( E[R] \) is the expected response time, and \( E[Z] \) is the expected think time.

:p What is the Throughput Law for closed systems?
??x
The Throughput Law (or Response Time Law) for a closed system states that the throughput (\(X\)) can be calculated as \( X = \frac{N}{E[R] - E[Z]} \). This formula relates the number of users, the expected response time, and the expected think time to determine the overall throughput.
x??

---

#### Example 1: Interactive System with N=10 Users
An interactive system has 10 users. The expected think time is \(E[Z] = 5\) seconds, and the expected response time is \(E[R] = 15\) seconds.

:p What is the throughput of this system?
??x
Using Little's Law for closed systems, we can calculate the throughput (\(X\)) as follows:
\[ N = X \cdot E[T] = X(E[Z] + E[R]) \]
Given \(N = 10\), \(E[Z] = 5\) seconds, and \(E[R] = 15\) seconds, we get:
\[ 10 = X(5 + 15) \]
\[ X = \frac{10}{20} = 0.5 \text{ jobs/sec} \]

The throughput of the system is 0.5 jobs per second.
x??

---

#### Example 2: Disk System with Throughput and Service Time
In a more complex interactive system, disk 3 has a throughput (\(X_{\text{disk3}} = 40\) requests/sec) and an average service time (\(E[S_{\text{disk3}}] = 0.0225\) sec). The average number of jobs in the system consisting of disk 3 and its queue is 4 (\(E[N_{\text{disk3}}] = 4\)).

:p What is the utilization of disk 3?
??x
The utilization (\(\rho_{\text{disk3}}\)) can be calculated using:
\[ \rho_{\text{disk3}} = X_{\text{disk3}} \cdot E[S_{\text{disk3}}] \]
Substituting the given values:
\[ \rho_{\text{disk3}} = 40 \cdot 0.0225 = 0.9 \text{ or } 90\% \]

The utilization of disk 3 is 90 percent.
x??

---

#### Example 2: Disk System - Mean Queueing Time
Continuing with the same setup, the mean time spent queueing plus serving at disk 3 (\(E[T_{\text{disk3}}]\)) and the mean queueing time (\(E[TQ_{\text{disk3}}]\)) need to be calculated.

:p What is the mean time spent queueing at disk 3?
??x
The mean total time spent by a job at disk 3 is given by:
\[ E[T_{\text{disk3}}] = \frac{E[N_{\text{disk3}}]}{X_{\text{disk3}}} = \frac{4}{40} = 0.1 \text{ sec} \]

The mean queueing time (\(E[TQ_{\text{disk3}}]\)) can be found by subtracting the service time from the total time:
\[ E[TQ_{\text{disk3}}] = E[T_{\text{disk3}}] - E[S_{\text{disk3}}] = 0.1 \text{ sec} - 0.0225 \text{ sec} = 0.0775 \text{ sec} \]

The mean queueing time at disk 3 is 0.0775 seconds.
x??

---

#### Example 2: Disk System - Number of Requests Queued
Continuing with the same setup, the number of requests queued at disk 3 (\(E[NQ_{\text{disk3}}]\)) needs to be calculated.

:p What is the mean number of requests queued at disk 3?
??x
The mean number of requests in the queue can be found by subtracting the number of jobs being served from the total number of jobs:
\[ E[NQ_{\text{disk3}}] = E[N_{\text{disk3}}] - \rho_{\text{disk3}} = 4 - 0.9 = 3.1 \text{ requests} \]

Alternatively, this can be calculated as:
\[ E[NQ_{\text{disk3}}] = E[TQ_{\text{disk3}}] \cdot X_{\text{disk3}} = 0.0775 \cdot 40 = 3.1 \text{ requests} \]

The mean number of requests queued at disk 3 is 3.1.
x??

---

#### Example 2: System Throughput Calculation
Given the throughput (\(X\)) and average think time (\(E[Z]\)), and knowing that \( E[R] = N / X - E[Z] \), we can calculate the system throughput.

:p How to find the system throughput using only one equation?
??x
To find the system throughput, we can apply Little's Law to the thinking region of the system. The throughput (\(X\)) is still \(X\), and the mean time spent in the thinking region is \(E[Z]\).

\[ E[N_{\text{thinking}}] = X \cdot E[Z] = 0.5 \cdot 5 = 2.5 \]

This equation shows that the number of ready users (not thinking) is 7.5, and we can solve for \(X\) and \(E[R]\):
\[ E[R] = N / X - E[Z] = 10 / X - 5 \]
Given \(N = 10\), solving gives:
\[ 2.5 = 10 / X - 5 \]
\[ 7.5 = 10 / X \]
\[ X = 10 / 7.5 = 0.5 \text{ jobs/sec} \]

The system throughput is 0.5 jobs per second.
x??

--- 

Each flashcard covers a specific aspect of the provided text, ensuring that all key concepts are explained and understood in detail.

#### Little's Law and Operational Laws
Background context: Little’s Law states that the average number of jobs in a system (E[N]) is equal to the arrival rate (λ) multiplied by the average time a job spends in the system (E[T]): \( E[N] = \lambda \cdot E[T] \). Other operational laws, such as the Forced Flow Law, are also discussed.

:p What does Little's Law state?
??x
Little's Law states that the average number of jobs in a system (\( E[N] \)) is equal to the arrival rate (λ) multiplied by the average time a job spends in the system (E[T]): \( E[N] = \lambda \cdot E[T] \). This law helps in understanding the relationship between the number of items in a queue, the rate at which they arrive, and their average waiting time.
x??

---
#### Forced Flow Law
Background context: The Forced Flow Law relates system throughput to the throughput of an individual device. It states that the system throughput (X) is equal to the sum of the product of the number of visits to a device per job (V_i) and the throughput at that device (X_i): \( X = \sum_{i} V_i \cdot X_i \).

:p What does the Forced Flow Law state?
??x
The Forced Flow Law states that for every system completion, there are on average \( E[Vi] \) completions at device i. Hence, the rate of completions at device i is \( E[Vi] \) times the rate of system completions.

Formally: If we observe the system for a large period t, then:
\[ X = \lim_{t \to \infty} \frac{C(t)}{t} = \sum_{i} E[V_i] \cdot X_i \]
where \( C(t) \) is the number of system completions during time t.

This law can be explained using a single device within a larger system. The visit ratio (Vi) represents the average number of times a job visits device i.
x??

---
#### Example Calculations Using Operational Laws
Background context: The example demonstrates calculating visit ratios and applying operational laws to find mean response times in systems.

:p What is the visit ratio for Disk b if given the network shown?
??x
Given the system where \( Ca = Ccpu \cdot 80/181 \), \( Cb = Ccpu \cdot 100/181 \), and \( C = Ccpu \cdot 1/181 \). The visit ratio for Disk b can be calculated as:
\[ E[Vb] = E[Vcpu] \cdot \frac{100}{181} \]

By solving the system of equations, we get:
\[ E[Vcpu] = 181 \]
\[ E[Va] = 80 \]
\[ E[Vb] = 100 \]
x??

---
#### Combining Operational Laws: Simple Example
Background context: The example uses operational laws to calculate the mean response time in an interactive system with multiple devices.

:p What is the mean response time, \( E[R] \), for a system with given characteristics?
??x
Given:
- Number of terminals (N) = 25
- Average think time (E[Z]) = 18 seconds
- Average visits to disk per interaction (E[Vdisk]) = 20
- Disk utilization (\( \rho_{disk} \)) = 30%
- Average service time per visit (E[Sdisk]) = 0.025 seconds

To find the mean response time \( E[R] \):
1. Calculate system throughput:
\[ X = \frac{X_{disk}}{E[V_{disk}]} = \frac{\rho_{disk} \cdot E[S_{disk}]}{E[V_{disk}]} = \frac{0.3 \cdot 0.025}{20/181} = 0.6 \text{ interactions/sec} \]

2. Calculate mean response time:
\[ E[R] = N \cdot X - E[Z] = 25 \cdot 0.6 - 18 = 23.7 \text{ sec} \]
x??

---
#### Combining Operational Laws: Harder Example
Background context: The harder example involves a more complex system with a memory queue and multiple devices.

:p What is the average amount of time that elapses between getting a memory partition and completing the interaction?
??x
The average amount of time spent in the central subsystem can be calculated as:
\[ E[\text{Time in central subsystem}] = E[\text{Response Time}] - E[\text{Time to get memory}] \]

First, calculate the response time using Little's Law:
\[ E[Response Time] = N \cdot X - E[Z] = 23 \cdot 0.45 - 21 = 30.11 \text{ sec} \]

Next, calculate the expected time to get memory:
\[ E[\text{Time to get memory}] = \frac{E[N_{getting memory}]}{X} = \frac{11.65}{0.45} = 25.88 \text{ sec} \]

Finally, the average amount of time spent in the central subsystem is:
\[ E[\text{Time in central subsystem}] = 30.11 - 25.88 = 4.23 \text{ sec} \]
x??

---
#### Calculating CPU Utilization
Background context: This example involves calculating the utilization of a CPU given system throughput and average service demand per visit.

:p What is the CPU utilization?
??x
The CPU utilization can be calculated using:
\[ \rho_{cpu} = X_{cpu} \cdot E[S_{cpu}] = X \cdot E[V_{cpu}] \cdot E[S_{cpu}] \]
Given:
- System throughput (X) = 0.45 interactions/sec
- Average visits to the CPU per interaction (E[Vcpu]) = 3
- Average service demand per visit to the CPU (E[Scpu]) = 0.21 seconds

Calculate the utilization:
\[ \rho_{cpu} = 0.45 \cdot 3 \cdot 0.21 = 0.28 \]
x??

---

#### Total Service Demand per Job Visit (D)
Background context: The total service demand on device \(i\) for all visits of a single job is denoted as \(D_i\). This can be defined as the ratio of the virtual visit time \(V_i\) to the sum of the service times \(S(j)_i\) required by each visit \(j\) of the job to server \(i\).
Formula: 
\[ D_i = \frac{V_i}{\sum_{j=1}^{S(i)} S(j)_i} \]
:p How would you determine \(E[D_i]\) in practice?
??x
To determine \(E[D_i]\), observe the total busy time at device \(i\) during an observation period, denoted as \(B_i\). Also, count the number of system completions during this observation period, denoted as \(C\). The expected value \(E[D_i]\) can be estimated using these measurements:
\[ E[D_i] = \frac{B_i}{C} \]
:p
The formula to determine \(E[D_i]\) involves measuring the busy time and the number of system completions. Explain why this method is practical.
??x
This method is practical because it avoids tracking individual job visits, which can be complex if devices are shared among multiple jobs. By observing the overall busy time and counting completions, we can estimate \(E[D_i]\) without needing to follow each job's interactions with the device.

---
#### Bottleneck Law (ρi)
Background context: The Bottleneck Law is a key principle for understanding the utilization of devices in a system. It relates the arrival rate \(X\) of jobs per second to the expected total service demand per visit \(E[D_i]\).
Formula:
\[ \rho_i = X \cdot E[Di] \]
:p
What does the equation \(\rho_i = X \cdot E[Di]\) represent in terms of device utilization?
??x
The equation represents the utilization (\(\rho_i\)) of a device \(i\) as the product of the arrival rate of jobs per second (\(X\)) and the expected total service demand per visit to that device (\(E[D_i]\)). Intuitively, \(\rho_i\) indicates how much time the device is busy handling work from arriving jobs.
:p
Explain the intuition behind the Bottleneck Law: why does \(X \cdot E[Di]\) represent the utilization of a device?
??x
The law suggests that each job arrival into the system contributes an amount of work \(E[D_i]\) to device \(i\). Since the arrival rate is \(X\) jobs per second, the total busy time for device \(i\) out of every second is given by:
\[ X \cdot E[Di] \]
This formula effectively measures how much of each second the device spends handling work from incoming jobs.

---
#### Proof of Bottleneck Law
Background context: The proof involves using Little’s law and assuming that the number of visits a job makes to a device is independent of its service demand.
Formula:
\[ \rho_i = X \cdot E[Si] = X \cdot E[V_i] \cdot E[S_i] = X \cdot E[D_i] \]
:p
How does the proof show that \(X \cdot E[Di]\) represents the utilization of device \(i\)?
??x
The proof starts with the understanding that the arrival rate \(X\) multiplied by the expected service time \(E[S_i]\) gives the expected total service demand per job (\(E[V_i] \cdot E[S_i]\)). Since \(D_i = V_i / S(i)\), we can rewrite this as:
\[ X \cdot E[Si] = X \cdot (E[V_i] \cdot E[S_i]) = X \cdot E[D_i] \]
Thus, the utilization (\(\rho_i\)) of device \(i\) is given by the product of the arrival rate and the expected total service demand per visit.

---
#### Practical Determination of Utilization
Background context: The practical determination of utilization involves measuring the busy time at a device during an observation period and counting system completions.
Formula:
\[ \rho_i = X \cdot E[D_i] = X \cdot Bi / C \]
:p
How would you practically determine the utilization (\(\rho_i\)) using observations?
??x
Practically, to determine the utilization (\(\rho_i\)), measure the busy time at device \(i\) over a long observation period (denoted as \(B_i\)). Also, count the number of system completions during this same period (denoted as \(C\)). The utilization can then be calculated as:
\[ \rho_i = X \cdot Bi / C \]
where \(X\) is the arrival rate of jobs per second.

