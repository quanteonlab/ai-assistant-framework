# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 2)

**Starting Chapter:** 2.3 Classification of Queueing Networks. 2.5 More Metrics Throughput and Utilization

---

#### Open Queueing Networks
Background context: Open queueing networks have external arrivals and departures. They can be modeled using various routing schemes, including probabilistic and non-probabilistic routes.

:p What are some examples of open queueing networks?
??x
Open queueing networks include:
- Single-server systems (Figure 2.2)
- Networks with probabilistic routing (Figure 2.3)
- Networks with non-probabilistic routing (Figure 2.4)

These networks can model scenarios such as packet flows in the Internet or manufacturing processes.

---
#### Network of Queues with Probabilistic Routing
Background context: In a network with probabilistic routing, packets receive external arrivals and internal arrivals from other servers. The probability of routing to another server depends on the current state (e.g., class of the packet).

:p What does Figure 2.3 illustrate?
??x
Figure 2.3 illustrates a network of queues with probabilistic routing where:
- Server \(i\) receives external arrivals at rate \(r_i\).
- A job finishing service at server \(i\) has a probability \(p_{ij}\) of being routed to server \(j\).

:p How can packet classes affect routing?
??x
Packet classes can influence the routing probabilities, meaning not all packets follow the same route. This could be useful in modeling internet traffic where packet class might depend on source and destination IP addresses.

---
#### Network of Queues with Non-Probabilistic Routing
Background context: In a network with non-probabilistic routing, all jobs follow a predetermined path. This is often used to model systems like CPU-to-disk operations (Figure 2.4).

:p What does Figure 2.4 illustrate?
??x
Figure 2.4 illustrates a network of queues with non-probabilistic routing where:
- Jobs arrive from an external source.
- The job follows a fixed route through the system, e.g., CPU to Disk 1 to Disk 2 and back.

:p How is throughput related to response time?
??x
Throughput is not directly related to minimizing response time. For instance, in Figure 2.6:
- System with lower processing rate (μ) but higher arrival rate (λ) may have a lower response time due to reduced queue length.
- Throughput \(X\) is the rate of job completions and depends on both λ and μ.

---
#### Utilization and Throughput
Background context: Utilization (\(\rho\)) measures how busy a device is, while throughput (X) measures the rate at which jobs are completed. In single-server systems, \(\rho = \frac{\lambda}{\mu}\).

:p What is the relationship between utilization and throughput in a single-server system?
??x
In a single-server system:
- Throughput \(X\) is given by \(X = \rho \cdot \mu\).
- Utilization \(\rho\) can be derived as \(\rho = \frac{\lambda}{\mu}\).

Using these, the throughput \(X\) simplifies to:
\[ X = \left( \frac{\lambda}{\mu} \right) \cdot \mu = \lambda. \]

This shows that in a single-server system with multiple servers, the overall throughput is independent of individual service rates.

---
#### Throughput Law
Background context: The Utilization Law states that device throughput \(X_i\) can be expressed as:
\[ X_i = \rho_i \cdot E[S], \]
where \(\rho_i\) is the utilization and \(E[S]\) is the mean service time.

:p What does the Utilization Law state?
??x
The Utilization Law states that the throughput (\(X_i\)) of a device in a queueing system can be expressed as:
\[ X_i = \rho_i \cdot E[S], \]
where \(\rho_i\) is the fraction of time the server is busy, and \(E[S]\) is the mean service time.

This relationship helps understand how utilization affects throughput. If the service rate increases but does not change the arrival or average service times, the throughput remains constant.

---
#### Single-Server Network with Finite Buffer
Background context: A single-server network can have a finite buffer capacity (Figure 2.5). Jobs that arrive when all server space is occupied are dropped.

:p What happens in a single-server network with finite buffer?
??x
In a single-server network with finite buffer, any arriving job that finds the system full is dropped. The performance metrics such as \(E[N]\), \(E[T]\), and \(E[TQ]\) need to account for this loss probability.

:p How does throughput relate to arrival rate in a system?
??x
Throughput (X) in a single-server system is independent of the service rate (\(\mu\)) but depends on the arrival rate (\(\lambda\)). Specifically, if \(\rho = \frac{\lambda}{\mu}\), then:
\[ X = \lambda. \]

This implies that increasing the processing speed does not increase throughput; it only affects response time and queue length.

---

#### Throughput in Probabilistic Networks of Queues
Background context: In a probabilistic network of queues, we consider the throughput (X) which is the number of jobs that can be processed per unit time. For server \(i\), the throughput \(X_i\) is equal to the total arrival rate \(\lambda_i\) into server \(i\). However, to determine \(\lambda_i\), we need to solve simultaneous equations where \(\lambda_i = r_i + \sum_j \lambda_j P_{ji}\).
:p What is the system throughput \(X\) in Figure 2.3?
??x
The system throughput \(X\) for Figure 2.3 is given by \(X = \sum_i r_i\). This equation sums up the rates of arrivals into each server, considering that because we assume a stable system, for large \(t\), the number of arrivals during \(t\) is approximately equal to the number of completions during \(t\).
x??

---

#### Throughput in Non-Probabilistic Networks
Background context: In networks with non-probabilistic routing, like those in Figure 2.4 and 2.5, the throughput can be analyzed differently due to fixed arrival rates. For instance, if there are two servers Disk1 and Disk2, \(X_{Disk1} = 3\lambda\) and \(X_{Disk2} = 2\lambda\). Here \(\lambda\) represents the total arrival rate.
:p What is the throughput \(X\) in Figure 2.4?
??x
The throughput \(X\) for Figure 2.4 is given by \(X = \lambda\).
x??

---

#### Throughput with Finite Buffer
Background context: In a network with finite buffer capacity, such as shown in Figure 2.5, the throughput \(X\) can be determined using the utilization factor \(\rho\), where \(X = \rho \mu\). Here, \(\rho\) is less than or equal to 1 because some jobs might get dropped due to buffer limitations.
:p What is the throughput \(X\) in Figure 2.5?
??x
The throughput \(X\) for Figure 2.5 is given by \(X = \rho \mu\). Here, \(\rho\) is the utilization factor which measures how busy the system is and can be determined through stochastic analysis.
x??

---

#### Closed Network Throughput: Interactive Systems
Background context: In an interactive system like the one shown in Figure 2.10, the throughput \(X\) is related to the response time (R) rather than a simple arrival rate. The goal is to minimize the average response time \(E[R]\). Here, \(\mu\) represents the service rate at each server.
:p What is the throughput \(X\) for an interactive system?
??x
The throughput \(X\) for an interactive system can be defined as \(X = \mu\), where \(\mu\) is the average service rate. However, due to think times and job submission constraints, \(E[R]\) (response time) differs from simple arrival rates.
x??

---

#### Throughput in Closed Network: Batch Systems
Background context: In a batch system like the one shown in Figure 2.12, all users submit jobs simultaneously, and each user can have multiple requests. The throughput \(X\) is typically high as long as there are no buffer limitations. The goal is to maximize job completion within a fixed time.
:p What is the throughput \(X\) for a batch system?
??x
The throughput \(X\) in a batch system is the number of jobs that can be processed per second, assuming there are always \(N\) jobs in the central subsystem due to predetermined and fixed multiprogramming level (MPL).
x??

---

#### Throughput in Closed Systems: Single Server Example
Background context: In a closed network with a single server as shown in Figure 2.13, the throughput is given by the service rate \(\mu\) of that single server. This differs from open systems where throughput can be independent of the service rate.
:p What is the throughput \(X\) for a single-server closed network?
??x
The throughput \(X\) for a single-server closed network in Figure 2.13 is given by \(X = \mu\). This means that the number of jobs processed per second is equal to the service rate of the server.
x??

---

#### Mean Response Time and Throughput in Closed Systems

Background context: In closed systems, the mean response time (E[R]) is a significant metric. For a closed batch system with N jobs, E[T] = N/μ, where μ is the service rate. The relationship between X (throughput) and E[R] is inversely related.

:p What is the throughput in terms of job service rates for a closed network?
??x
In a closed network, the throughput \(X\) depends on the service rates \(\mu_i\). If we double all the service rates \(\mu_i\) while keeping the number of jobs \(N\) constant, the throughput \(X\) changes. This is because the completion rate at the server affects the overall system throughput.
x??

---

#### Tandem Servers Closed Network

Background context: In a tandem servers closed network (as shown in Figure 2.14), the mean response time and throughput are interrelated but not always as straightforward as in simple models.

:p Why might the previous answer about throughput not be universally correct?
??x
The previous answer about throughput is correct only if we know that the slower server is always busy, which is not necessarily true. For example, with \(N=1\), it's clear that the slower server doesn't have to be busy all the time.

Even for \(N=2\), the slower server might still be idle sometimes because service rates are averages and job sizes can vary.
x??

---

#### Throughput in Open Systems

Background context: In open systems, throughput (\(X\)) is independent of individual service rates \(\mu_i\). Doubling all service rates does not change \(X\), but response time (E[R]) and throughput are unrelated.

:p What does the throughput depend on in an open system?
??x
In an open system, the throughput \(X\) depends on the arrival rate \(\lambda\) and the service rates \(\mu_i\). Doubling all the \(\mu_i\) while keeping \(\lambda\) constant will not change the throughput \(X\), as long as \(\lambda < \sum \mu_i\).
x??

---

#### Modeling Single-Server Queues

Background context: For a single-server queue, obtaining the mean job size \(E[S]\) can be challenging. Direct measurement methods might not accurately reflect the actual system behavior.

:p How do you practically obtain \(E[S]\) for a single-server system?
??x
To obtain \(E[S]\), it's better to use the service rate \(\mu\). For an open system, increase the arrival rate \(\lambda\) until the completion rate levels off. The level-off value is the service rate \(\mu\), and \(E[S] = 1/\mu\).

```java
public class Server {
    private double lambda; // arrival rate
    private double mu; // service rate

    public void measureServiceRate() {
        for (lambda = 0.1; ; lambda += 0.1) { // incrementing lambda
            if (isSystemStable(lambda)) {
                mu = calculateCompletionRate(lambda);
                break;
            }
        }
        E[S] = 1 / mu;
    }

    private boolean isSystemStable(double lambda) {
        // logic to check stability
    }

    private double calculateCompletionRate(double lambda) {
        // logic to calculate completion rate based on lambda
    }
}
```
x??

---

#### Differences Between Closed and Open Networks

Background context: The behavior of closed and open networks differs significantly. In closed systems, throughput changes with service rates, while in open systems, throughput is independent of individual service rates but depends on the arrival rate.

:p How do throughput and response time relate differently in closed versus open networks?
??x
In closed networks, higher throughput corresponds to lower average response times. However, in open networks, throughput (\(X\)) is independent of \(\mu_i\) but can be affected by the arrival rate (\(\lambda\)). The response time (E[R]) and throughput are not related in open systems.
x??

---

#### Modeling with IBM Blade Server

Background context: Modeling a server as a single-server queue involves understanding job sizes. Direct measurement might not accurately represent the system's behavior due to varying conditions.

:p How can you measure the mean job size \(E[S]\) for an IBM blade server?
??x
To measure \(E[S]\), use the service rate \(\mu\). For an open system, incrementally increase \(\lambda\) until stability is achieved. The completion rate at that point gives \(\mu\), and then \(E[S] = 1/\mu\).

```java
public class BladeServer {
    private double lambda; // arrival rate
    private double mu; // service rate

    public void measureJobSize() {
        for (lambda = 0.1; ; lambda += 0.1) { // incrementing lambda
            if (isSystemStable(lambda)) {
                mu = calculateCompletionRate(lambda);
                break;
            }
        }
        E[S] = 1 / mu;
    }

    private boolean isSystemStable(double lambda) {
        // logic to check stability based on arrival rate and service rate
    }

    private double calculateCompletionRate(double lambda) {
        // logic to calculate completion rate based on lambda and mu
    }
}
```
x??

---

#### Throughput in Closed Systems

Background context: In closed systems, the throughput \(X\) depends on the service rates \(\mu_i\). Doubling all \(\mu_i\) while keeping \(N\) constant changes \(X\).

:p What happens to the throughput when you double all service rates in a closed system?
??x
Doubling all service rates \(\mu_i\) in a closed system with \(N\) jobs results in an increase in throughput \(X\), as the completion rate at each server increases. The exact change depends on the new balance of work among servers.
x??

---

#### Scheduling Orders and Slowdown

Background context: Different scheduling policies can affect both response time and slowdown. Shortest-Processing-Time-First (SRPT) is claimed to minimize mean response time, but its impact on mean slowdown is less clear.

:p How does SRPT scheduling policy compare with FCFS in terms of mean slowdown?
??x
SRPT generally minimizes mean response time, but the effect on mean slowdown can vary. For some workloads, it might reduce mean slowdown, while for others, it may not.

```java
public class Scheduler {
    public double calculateMeanSlowdown(SchedulingPolicy policy) {
        // logic to calculate mean slowdown based on scheduling policy and job sizes
    }
}
```
x??

---

#### Variability in Service Time

Background context: In closed systems, variability in service times can significantly affect response time. The relationship between closed and open systems is not straightforward.

:p How does variability in service time affect closed systems?
??x
Variability in service times can increase the average response time in closed systems due to the queuing behavior. Closed systems might have different performance characteristics compared to their open counterparts.
x??

---

#### Scheduling Policies

Background context: Different scheduling policies (e.g., FCFS, SJF) can affect mean response time and other performance metrics.

:p How does SRPT compare with other scheduling policies in terms of minimizing response time?
??x
SRPT is designed to minimize mean response time by always serving the job with the currently shortest remaining processing time. Other policies like FCFS do not necessarily achieve this goal.
x??

---

