# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 36)

**Starting Chapter:** 22.7 Readings

---

#### Network of PS Servers with Probabilistic Routing
Background context: This concept describes a network of PS (Processor Sharing) servers where jobs are routed probabilistically. The system has Poisson outside arrivals and general Coxian service times. The main result is that such networks can have a product form solution, similar to Jackson networks but with different distributions.

Relevant formulas:
- \( P\{n \text{jobs at server 1}\} = (1-\rho_a)\rho_a^n \)
- \( P\{m \text{jobs at server 2}\} = (1-\rho_b)\rho_b^m \)
- The joint probability: \( P\{n \text{jobs at server 1, } m \text{jobs at server 2}\} = \rho_a^n(1-\rho_a)(1-\rho_b)\rho_b^m = P\{n \text{jobs at server 1}\} \cdot P\{m \text{jobs at server 2}\} \)

Explanation: The insensitivity property allows the distribution of jobs in each queue to be calculated independently, making analysis simpler.

:p What is a key feature of PS servers that affects the routing and job behavior?
??x
PS servers have all jobs inside the gray bubble (server) at all times, with no queue outside. Jobs move through phases independently.
x??

---

#### Insensitivity Property in PS Networks
Background context: This concept explains why only the mean service time is relevant for networks of PS servers, allowing a simpler product form solution.

Relevant formulas:
- \( \rho_i = \frac{\lambda_i E[S_i]}{1} \)
- General network formula: 
  \[
  P\left\{ (n_1, n_2, ..., n_k) \text{jobs at each queue}\right\} = \prod_{i=1}^{k} P\{n_i \text{jobs at server } i\}
  \]
  where \( P\{n_i \text{jobs at server } i\} = \rho_i^n (1-\rho_i) \)

Explanation: The mean service time is the only parameter needed, as other details are averaged out by the nature of PS processing.

:p Why is the insensitivity property significant in analyzing networks with PS servers?
??x
The insensitivity property allows us to ignore job size distribution specifics and focus on mean service times, simplifying analysis. Only the average service time affects the queue length distribution.
x??

---

#### Differences Between FCFS and PS Networks
Background context: This concept highlights the key differences between FCFS (First Come First Served) and PS networks in terms of job behavior and state space.

Relevant formulas:
- N/A

Explanation: In FCFS servers, only one job is processed at a time within the server's gray bubble, while others wait outside. This restricts job movement compared to PS servers where jobs are always inside and move independently.

:p Why do networks with FCFS servers not exhibit product form solutions like those of PS networks?
??x
FCFS servers process jobs in strict order, leading to a different state space and restricted job behavior. Jobs cannot be processed simultaneously, unlike in PS queues.
x??

---

#### General Network of PS Servers
Background context: This concept extends the understanding of PS networks beyond two servers, explaining the product form solution for multiple server configurations.

Relevant formulas:
- \( P\left\{ (n_1, n_2, ..., n_k) \text{jobs at each queue}\right\} = \prod_{i=1}^{k} P\{n_i \text{jobs at server } i\} \)
  where \( P\{n_i \text{jobs at server } i\} = \rho_i^n (1-\rho_i) \)

Explanation: The same analysis applies to networks with multiple PS servers, maintaining the product form solution for each queue.

:p How does the general network of PS servers maintain a product form solution?
??x
The product form solution holds because jobs in each server act independently, allowing individual probabilities to be multiplied together. Each queue's state can be analyzed separately.
x??

---

#### BCMP Theorem Overview
Background context: This section discusses the application of the method of phases, specifically how it applies to networks with time-sharing (PS) servers using the BCMP theorem. It mentions that under certain conditions, an M/Cox/1/PS queue can behave like an M/M/1/FCFS queue and that a network of such queues has product form.
:p What is the BCMP theorem and its significance in queueing networks?
??x
The BCMP theorem provides a way to analyze complex queuing networks where service times are Coxian, meaning they have multiple phases. It states that under certain conditions, these networks can be decomposed into simpler M/Cox/1/PS queues, which can then be analyzed individually. This decomposition simplifies the analysis of the entire network.
x??

---

#### Product Form Networks
Background context: The text explains how product form solutions extend beyond PS and PLCFS (Probabilistic Last Come First Serve) scheduling disciplines to include quasi-reversible queues, allowing for a broader class of networks with product form solutions. Key references are provided for further reading.
:p What is the significance of quasi-reversible queues in extending product form solutions?
??x
Quasi-reversible queues extend the applicability of product form solutions by allowing more general queueing disciplines than just PS or PLCFS. This means that a wider range of network configurations can be analyzed using these simplified methods, providing flexibility and greater practical utility.
x??

---

#### M/BP/1/PS Queue Simulation
Background context: The exercise asks to simulate an M/BP/1/PS queue with bounded Pareto service times (BP) under two different values of α. The goal is to measure the mean response time in both cases.
:p What do you expect the mean response time to be for M/BP/1/PS queues with different values of α?
??x
For an M/BP/1/PS queue, the expected mean response time increases as the value of α decreases. This is because a smaller α means longer service times, which will increase the mean response time.
To determine this analytically or through simulation:
- Simulate with α=1.5: The system has shorter service times and should have a lower mean response time.
- Simulate with α=2.9: The system has longer service times and should have a higher mean response time.
:x??

---

#### Tandem Network of PS Servers
Background context: This section discusses the local balance equations for a tandem network of PS servers, providing a specific guess for the limiting probabilities that need to be verified.
:p How can you prove that the given guess satisfies the local balance equations?
??x
To verify that the given guess satisfies the local balance equations, we need to check if the guessed probabilities πn1,n2,m1,m2 satisfy Bi = B' for i = 0, 1, 2, 3, 4. Here’s a step-by-step approach:

- **Guess**:
  \[
  \pi_{n_1, n_2, m_1, m_2} = \binom{n_1 + n_2}{n_1} \rho_1^{n_1} \rho_2^{n_2} / \left(\binom{m_1 + m_2}{m_1} \rho_3^{m_1} \rho_4^{m_2}\right) \pi_{0, 0, 0, 0}
  \]
  where:
  - \( \rho_1 = \lambda / \mu_1 \)
  - \( \rho_2 = \lambda p / \mu_2 \)
  - \( \rho_3 = \lambda / \mu_3 \)
  - \( \rho_4 = \lambda q / \mu_4 \)

- **Local Balance Equations**:
  Check the balance equations Bi = B' for each state.

Example code to verify these equations (pseudocode):
```java
public class TandemNetwork {
    private double rho1, rho2, rho3, rho4;
    private double pi0;

    public TandemNetwork(double lambda, double mu1, double mu2, double mu3, double mu4, double p, double q) {
        this.rho1 = lambda / mu1;
        this.rho2 = (lambda * p) / mu2;
        this.rho3 = lambda / mu3;
        this.rho4 = (lambda * q) / mu4;
    }

    public boolean verifyLocalBalance() {
        // Implement verification logic here
        return true; // Placeholder, actual implementation needed
    }
}
```

The key is to ensure that the balance equations hold by comparing the input and output rates for each state.
:x??

#### The Inspection Paradox
Background context: We introduce the Inspection Paradox by asking two questions. One pertains to a bus arrival scenario where you arrive at a random time and want to know how long you expect to wait for a bus. This question is followed by considering different distributions of inter-arrival times between buses, while maintaining an average mean.

:p What is the question about the Inspection Paradox in the context of bus arrivals?
??x
The question asks: If buses arrive every 10 minutes on average and you arrive at a random time, how long can you expect to wait for the next bus? Additionally, does your answer change if we use different distributions for the inter-arrival times between buses (with the mean still being 10 minutes)?
x??

---

#### The M/G/1 Queue
Background context: We discuss an M/G/1 queue where jobs arrive according to a Poisson process with rate λ and have general service time distributions. The system assumes First-Come-First-Served (FCFS) service discipline.

:p What is the definition of an M/G/1 queue?
??x
An M/G/1 queue consists of a single server and a queue where:
- Jobs arrive according to a Poisson process with rate λ.
- Service times are generally distributed, denoted by the random variable S with E[S] = 1/μ.

This setup assumes First-Come-First-Served (FCFS) service order unless otherwise stated.
x??

---

#### Tagged Job Technique for Mean Time in Queue
Background context: We use a tagged job technique to derive the mean time in queue (TQ) for an M/G/1 system. The technique involves tagging an arbitrary arrival and analyzing their experience in the queue.

:p What is TQ in the context of the tagged job technique?
??x
TQ represents the time spent by an arrival in the queue. It can be broken down into two parts:
- Unfinished work that the arrival witnesses in the system.
- This includes unfinished work in the queue and at the server.

Mathematically, TQ is given by:
\[ E[TQ] = \frac{E[\text{Unfinished work in queue}]}{1 - \rho} + (Time\ -avg\ probability\ server\ busy) \cdot E[Se] \]

Where \( Se \) is the remaining service time of a job in service, given that there is some job in service.
x??

---

#### Formula for Mean Time in Queue
Background context: We derive a formula for the mean time in queue (TQ) using the tagged job technique. This involves understanding the expectations and utilizing properties of Poisson processes.

:p What is the formula for E[TQ] derived from the tagged job technique?
??x
The formula for the expected time in queue \(E[TQ]\) is given by:
\[ E[TQ] = \frac{ρ}{1 - ρ} \cdot E[Se] \]

Where:
- \( ρ \) is the traffic intensity (load factor), defined as \( λ / μ \).
- \( E[Se] \) is the expected remaining service time given that there is a job in service.
x??

---

#### Example: M/M/1 Queue
Background context: We provide an example to illustrate the application of the tagged job technique to an M/M/1 queue, where both arrival and service times are exponentially distributed.

:p For an M/M/1 queue, what is \( E[Se] \)?
??x
For an M/M/1 queue, since the service time S is Exponentially distributed with mean 1/μ, we have:
\[ E[Se] = \frac{1}{μ} \]

Thus, the expected time in queue for the tagged job is:
\[ E[TQ] = \frac{ρ}{1 - ρ} \cdot \frac{1}{μ} \]
x??

---

#### Example: M/D/1 Queue
Background context: We provide an example to illustrate the application of the tagged job technique to a deterministic service time (M/D/1) queue.

:p For an M/D/1 queue, what is \( E[Se] \)?
??x
For an M/D/1 queue, since the service time S is Deterministic and equal to 1/μ, the remaining service time of a job in service is uniformly distributed between 0 and 1/μ. Therefore:
\[ E[Se] = \frac{1}{2} \cdot \frac{1}{μ} = \frac{1}{2μ} \]

Thus, the expected time in queue for the tagged job is:
\[ E[TQ] = \frac{ρ}{1 - ρ} \cdot \frac{1}{2μ} \]
x??

---

#### Example: M/Ek/1 Queue
Background context: We provide an example to illustrate the application of the tagged job technique to a service time with Erlang-k distribution (M/Ek/1) queue.

:p For an M/Ek/1 queue, what is \( E[Se] \)?
??x
For an M/Ek/1 queue, where the service time has an Erlang-k distribution, the remaining service time of a job in service is uniformly distributed between 0 and k/μ. On average, the job will be at the middle stage, leaving ceil(k+1)/2 stages left to complete.

Thus:
\[ E[Se] = \left\lceil \frac{k + 1}{2} \right\rceil \cdot \frac{1}{kμ} \]

The expected time in queue for the tagged job is then:
\[ E[TQ] = \frac{ρ}{1 - ρ} \cdot \left\lceil \frac{k + 1}{2} \right\rceil \cdot \frac{1}{kμ} \]
x??

---

#### Example: M/H2/1 Queue
Background context: We provide an example to illustrate the application of the tagged job technique to a service time with Hyperexponential distribution (M/H2/1) queue. This involves using the Renewal-Reward theorem.

:p For an M/H2/1 queue, how is \( E[Se] \) derived?
??x
For an M/H2/1 queue, where the service time has a Hyperexponential distribution with two phases, we use the Renewal-Reward Theorem to compute \(E[Se]\). This theorem allows us to find the expected value of the remaining service time given that there is some job in service.

To compute \( E[Se] \) exactly for any random variable S, we need to apply the Renewal-Reward theorem.
x??

---

