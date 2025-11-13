# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 11)

**Starting Chapter:** 7.5 Comparison of Closed and Open Networks. 7.6 Readings. 7.7 Exercises

---

#### Outside Arrival Rates - Open Networks
Background context: The text discusses how the asymptotic bounds for closed networks do not directly apply to open networks. In an open network, jobs arrive from outside and can leave at any time after service completion. The main difference is that the utilization $X$(the fraction of time a device is busy) in an open network is constrained by both its processing capacity and the arrival rate.

:p What are the limitations when applying closed network asymptotic bounds to open networks?
??x
In open networks, the asymptotic bounds derived for closed networks do not directly apply because the utilization $X $ in an open systems is bounded by$\frac{1}{D_{\text{max}}}$, where $ D_{\text{max}}$ is the maximum service time. This means that even if a device has high processing capacity, its actual utilization will be limited by the incoming job rate and not necessarily reach the asymptotic bound derived for closed networks.

For example, in an open network with average service times:
- If jobs require 3 seconds on average to complete,$D_{\text{max}} = 3$ seconds.
- Thus, the maximum utilization $X \leq \frac{1}{3}$, regardless of the arrival rate or processing speed.

This limitation means that bounds derived for closed networks may not accurately predict performance in open systems unless the outside arrival rate is high enough to bring the system close to its asymptotic regime.

x??

---
#### Open versus Closed Systems - Server Speed Increase
Background context: The text presents a scenario where a server’s speed is doubled and asks whether this change results in significant improvement for both closed and open networks. It involves understanding how load balancing and throughput are affected differently by such modifications.

:p In the given network, does doubling the speed of one server result in significant improvement? Explain using operational laws.
??x
Doubling the speed of a single server can lead to significant improvements in the mean response time for an open network but not necessarily for a closed network. This is because in an open system, load balancing and reducing bottlenecks are key factors.

**Open Network Scenario:**
- Assume servers alternate processing jobs deterministically.
- Doubling the speed of Server 2 will decrease its busy time significantly, leading to faster service completion times.
- For example, if both servers process jobs at a constant rate, doubling one server’s speed can halve the overall processing time.

**Closed Network Scenario:**
- In a closed network, all jobs are eventually processed by the system and do not leave.
- Speeding up only one device may not significantly affect the overall response time as the system is constrained by the slower device's capacity.

To illustrate:
```java
public class ServerSpeedTest {
    public static void main(String[] args) {
        double r1 = 0.3; // Arrival rate to server 1
        double E[S1] = 0.1; // Service time at server 1 in seconds
        double E[S2] = 0.05; // Service time at server 2 in seconds (original)
        
        System.out.println("Original Mean Response Time: " + calculateMeanResponseTime(r1, E[S1], E[S2]));
        
        E[S2] *= 2; // Doubling the service rate of Server 2
        System.out.println("Improved Mean Response Time: " + calculateMeanResponseTime(r1, E[S1], E[S2]));
    }
    
    public static double calculateMeanResponseTime(double r, double E[S1], double E[S2]) {
        return (r / (1 - (0.3 * E[S1] + 0.7 * E[S2])));
    }
}
```
x??

---
#### Modifying a Closed Interactive System
Background context: Marty is running his database as a closed system and wants to improve throughput by purchasing additional CPU or disk capacity. The goal is to determine which modification will provide the most significant performance improvement.

:p Which device should Marty buy (CPU or Disk) to increase throughput, given equal costs?
??x
Given that both devices are equally priced, Marty should choose the one with the highest utilization and the largest potential for improvement in throughput.

From the measurements:
- Number of CPU completions: 300
- Number of disk completions: 400

This indicates higher CPU utilization (300 out of total jobs) compared to disk utilization (400 out of total jobs). However, the actual CPU busy time is lower than the disk busy time:
- CPU busy time: 600 seconds
- Disk busy time: 1200 seconds

Doubling the CPU speed will have a more significant impact on throughput because it reduces the CPU's busy time and can better balance the load.

To determine the optimal split, we need to consider the current utilization:
```java
public class ModifyingSystem {
    public static void main(String[] args) {
        double rCCPU = 300; // Number of CPU completions
        double rCdisk = 400; // Number of disk completions
        double rBCPU = 600;  // CPU busy time in seconds
        double rBdisk = 1200; // Disk busy time in seconds
        
        System.out.println("CPU Utilization: " + (rCCPU / rBCPU));
        System.out.println("Disk Utilization: " + (rCdisk / rBdisk));
    }
}
```

The disk utilization is higher but the CPU's busy time is lower, indicating that increasing the CPU speed will provide a more significant throughput improvement.

x??

---
#### Proportional Power - Machine Speed and Power
Background context: In power distribution systems, the speed of a machine is proportional to the power allocated. The goal is to maximize system throughput by optimally dividing the total power budget between two machines based on their processing probabilities.

:p What choice for dividing power $W $ and picking$p$ will maximize the throughput in a closed batch system with two servers?
??x
To maximize throughput, we need to balance the load across both servers. The optimal strategy is to allocate power such that the effective service rates of both machines are equalized.

Given:
- Total power budget:$W $- Number of jobs routed to server 1:$ pN $- Number of jobs routed to server 2:$(1-p)N $ Let's denote the speed of machine 1 as$w_1 $ and machine 2 as$w_2$. The throughput is maximized when:
$$w_1 \cdot p = w_2 \cdot (1 - p)$$

If $w_1 = w_2 $, then$ p = 0.5$.

For general $w_1 $ and$w_2$:
- Allocate power such that the effective speeds are equal.
- This can be achieved by solving:
$$\frac{W}{w_1} \cdot p = \frac{W}{w_2} \cdot (1 - p)$$

Solving for $p$:
$$p = \frac{w_2}{w_1 + w_2}$$
$$1 - p = \frac{w_1}{w_1 + w_2}$$

Thus, the optimal strategy is to divide power and choose routing probabilities such that:
- $p = \frac{\text{speed of slower machine}}{\text{sum of speeds}}$- This ensures balanced load distribution.

x??

---

#### Discrete-Time Markov Chains (DTMCs)
Background context: Discrete-time Markov chains are used to model systems where the state changes at discrete time points. They are particularly useful for analyzing processes that have memoryless properties, meaning the future behavior depends only on the current state and not on the sequence of events that preceded it.

:p What are DTMCs used for in modeling systems?
??x
DTMCs are used to model systems where the state changes at discrete time points. They allow us to analyze processes with memoryless properties, focusing solely on the current state without considering past states.
x??

#### The Markovian Property
Background context: The Markovian property ensures that future behavior depends only on the current state and not on the sequence of events that preceded it. This is crucial for DTMCs as it simplifies analysis by reducing dependencies.

:p What does the Markovian property ensure in a system?
??x
The Markovian property ensures that the future behavior of a system depends only on its current state, making the analysis more tractable and simpler.
x??

#### Introduction to Discrete-Time Markov Chains (Chapter 8)
Background context: Chapter 8 introduces DTMCs and explains the Markovian property. It covers foundational concepts necessary for understanding how to model systems with discrete states changing at discrete time points.

:p What is covered in Chapter 8?
??x
Chapter 8 covers the introduction of DTMCs, explaining the Markovian property and foundational concepts needed to model systems with discrete state changes over discrete time points.
x??

#### Google’s PageRank Algorithm (Chapter 10)
Background context: Chapter 10 discusses real-world applications of DTMCs in computing. One notable example is Google's PageRank algorithm, which uses a Markov chain to rank web pages based on their importance.

:p What does Chapter 10 introduce regarding real-world examples?
??x
Chapter 10 introduces the application of DTMCs through real-world examples, including Google’s PageRank algorithm, which ranks web pages by modeling links between them as a Markov chain.
x??

#### Complex Discrete-Time Markov Chains (Chapter 10)
Background context: Chapter 10 also covers more complex DTMCs that occur naturally and demonstrates how generating functions can be used to solve such systems.

:p What additional topics does Chapter 10 cover?
??x
Chapter 10 covers more complex real-world examples of DTMCs, including the use of generating functions to solve them.
x??

#### Continuous-Time Markov Chains (CTMCs)
Background context: CTMCs model systems where state changes occur continuously over time. They are an extension of DTMCs but allow for transitions that can happen at any point in time.

:p What is a key difference between DTMCs and CTMCs?
??x
A key difference is that CTMCs model continuous-time processes, whereas DTMCs handle discrete-time events.
x??

#### Markovian Property of Exponential Distribution (Chapter 11)
Background context: Chapter 11 discusses the Markovian property as it applies to exponential distributions and Poisson processes. These properties make them particularly suitable for CTMCs.

:p What specific distributions are discussed in relation to CTMCs in Chapter 11?
??x
Chapter 11 discusses the Exponential distribution and the Poisson process, which have the Markovian property, making them suitable for use with CTMCs.
x??

#### Transition of Concepts from DTMCs to CTMCs (Chapter 12)
Background context: Chapter 12 shows how to translate concepts learned in DTMCs to CTMCs, providing a bridge between these two types of Markov chains.

:p What is the main focus of Chapter 12?
??x
The main focus of Chapter 12 is to demonstrate how concepts from DTMCs can be applied and adapted to understand CTMCs.
x??

#### Analyzing M/M/1 Queues with CTMC Theory (Chapter 13)
Background context: Chapter 13 applies CTMC theory to analyze the M/M/1 single-server queue, covering key properties such as the PASTA property.

:p What does Chapter 13 focus on analyzing?
??x
Chapter 13 focuses on applying CTMC theory to analyze an M/M/1 single-server queue and covers properties like the PASTA (Poisson Arrivals See Time-Averages) property.
x??

#### Application of CTMCs in Multi-Server Systems (Part V)
Background context: CTMCs are extensively used in Part V to model multi-server systems, providing a powerful tool for analyzing complex queueing networks.

:p What is the significance of using CTMCs in Part V?
??x
Using CTMCs in Part V allows for detailed analysis of multi-server systems, offering a robust method to understand and optimize these complex queueing networks.
x??

---

#### Closed Systems
Background context: For closed systems, we can approximate and bound the values of throughput,$X $, and the expected response time, $ E[R]$. The approximations developed are independent of the distribution of service times but require that the system is closed. When the multiprogramming level $ N$is much higher than $ N^*$, we have a tight bound on $ X$and $ E[R]$. Also, when $ N = 1$, we have a tight bound. However, for intermediate values of $ N$, we can only approximate $ X$and $ E[R]$.

:p What are the conditions under which closed systems allow tight bounds on throughput and expected response time?
??x
When the multiprogramming level $N $ is much higher than a critical value$N^*$ or when $ N = 1 $, we can achieve tight bounds on the throughput $ X$and the expected response time $ E[R]$. For intermediate values of $ N$, only approximations are possible.

---
#### Open Systems
Background context: In open systems, it is more challenging to derive performance metrics such as the mean number of jobs $E[N_i]$ at a server in a queueing network. We cannot calculate $E[T]$(mean response time) without knowing $ E[N]$, which we do not yet know how to compute.

:p What are the limitations when analyzing open systems?
??x
In open systems, it is difficult to derive performance metrics like mean number of jobs at a server or mean response time because we cannot calculate these metrics without knowing $E[N]$ (mean number of jobs in the system), which is unknown. This makes analysis more complex compared to closed systems.

---
#### Markov Chain Analysis
Background context: Markov chain analysis is a powerful tool for deriving performance metrics such as the mean number of jobs at each server and their full distribution. It can be applied not only to queueing networks but also to more complex systems, provided certain distributions (Exponential or Geometric) are used.

:p What makes Markov chains particularly useful in analyzing queueing systems?
??x
Markov chain analysis is useful because it enables us to determine the mean number of jobs at each server and their full distribution. This is especially true when service times and interarrival times follow Exponential or Geometric distributions, which have a memoryless property.

---
#### Memoryless Property (Exponential Distribution)
Background context: The Exponential distribution has the Markovian property (memoryless property), meaning that the remaining time until an event occurs (like service completion or job arrival) is independent of how long we have waited so far. This property allows for exact modeling of certain queueing systems.

:p What does it mean when a distribution is said to be "memoryless"?
??x
When a distribution is memoryless, the remaining time until an event occurs is independent of how much time has already passed without that event occurring. For example, in Exponential distributions, the time left for service completion or job arrival does not depend on how long the current wait has been.

---
#### Non-Markovian Workloads
Background context: While some systems can be modeled using Markov chains with memoryless properties (Exponential or Geometric), other distributions do not have this property. However, these non-memoryless distributions can often be approximated by mixtures of Exponential distributions, which still allows for analysis through Markov chain methods.

:p Can non-Markovian workloads be analyzed using Markov chains?
??x
Yes, even though some workload distributions do not have the memoryless property (are non-Markovian), they can often be approximated by mixtures of Exponential distributions. This allows for analysis through Markov chain methods, although with potentially less accuracy compared to exact models.

---
#### Summary
Background context: The text discusses the limitations and capabilities of Markov chains in analyzing both closed and open queueing systems. It highlights that while certain distributions (Exponential or Geometric) enable precise modeling, other non-Markovian distributions can still be approximated for analysis.

:p What are the key points covered in this section about Markov chain analysis?
??x
Key points include the limitations of analyzing open systems, the usefulness of Markov chains with memoryless properties (Exponential or Geometric) for exact modeling, and how non-Markovian distributions can be approximated by mixtures of Exponential distributions to facilitate analysis.

---

