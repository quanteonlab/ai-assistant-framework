# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 19)

**Starting Chapter:** 11.6 Merging Independent Poisson Processes. 11.7 Poisson Splitting

---

#### Exponential Distribution and Poisson Process Approximation
Background context: In a Poisson process, each δ-size interval has approximately 1 event with probability $\lambda\delta + o(\delta)$, where events occur at rate $\lambda $. As $\delta \to 0 $, the number of events $ N(t)$in time $ t$ can be approximated as a Binomial distribution, which converges to a Poisson distribution.

:p What does each δ-size interval approximate in terms of event occurrence?
??x
Each δ-size interval approximates having 1 event with probability $\lambda\delta + o(\delta)$, and otherwise having 0 events. This is an approximation that holds as the size of the intervals, δ, approaches zero.
x??

---

#### Merging Independent Poisson Processes
Background context: When merging two independent Poisson processes, each with rates $\lambda_1 $ and$\lambda_2 $, the merged process becomes a single Poisson process with rate $\lambda_1 + \lambda_2$.

:p What is the result of merging two independent Poisson processes?
??x
Merging two independent Poisson processes results in a single Poisson process with the combined rate, which is the sum of the individual rates. Specifically, if Process 1 has events at rate $\lambda_1 $ and Process 2 has events at rate$\lambda_2 $, the merged process will have events at rate $\lambda_1 + \lambda_2$.
x??

---

#### Poisson Splitting
Background context: Given a single Poisson process with rate $\lambda $, where each event is classified as "type A" with probability $ p $ and "type B" with probability $1-p $, the type A events form a Poisson process with rate $ p\lambda $, and the type B events form a Poisson process with rate$(1-p)\lambda$. These two processes are independent.

:p What happens when each event in a Poisson process is classified as "type A" or "type B"?
??x
When each event in a Poisson process is classified as either "type A" with probability $p $ or "type B" with probability$1-p $, the type A events form their own independent Poisson process with rate $ p\lambda $, and the type B events also form an independent Poisson process with rate$(1-p)\lambda$.

To understand why, consider that in a time period of length $t $, the number of type A events is distributed as $ N_A(t) \sim \text{Poisson}(\lambda p t)$, and the number of type B events is distributed as $ N_B(t) \sim \text{Poisson}(\lambda (1-p) t)$.

The joint probability can be computed using:
$$P\{N_A(t) = n, N_B(t) = m\} = e^{-\lambda t} \binom{n+m}{n} p^n (1-p)^m (\lambda t)^{n+m} / (n+m)!$$which simplifies to the product of individual Poisson probabilities:
$$

P\{N_A(t) = n\} \cdot P\{N_B(t) = m\} = e^{-p\lambda t} \frac{(p\lambda t)^n}{n!} \cdot e^{-(1-p)\lambda t} \frac{((1-p)\lambda t)^m}{m!}$$x??

---

#### Poisson Splitting Intuition
Background context: The Poisson splitting theorem can be understood by analogy with the Geometric distribution. In a sequence of coin flips (with bias $p $), type A events are identified as "heads" and occur at rate $\lambda p$. Type B events, corresponding to "tails," have their own independent process.

:p How does Poisson splitting relate to Geometric distributions?
??x
Poisson splitting relates to the Geometric distribution through an analogy. In a sequence of coin flips with bias $p $, where each event in the original Poisson process is classified as type A (heads) or type B (tails), we can think of flipping a biased coin repeatedly. Type A events occur when both the "first" coin flip and the "second" coin flip come up heads, which corresponds to a single coin with success probability $\lambda p$. This means that the interarrival times between type A events are distributed as Exponential(λp).

The Geometric distribution describes the number of trials needed for the first success in repeated Bernoulli trials. Here, it helps us understand why the interarrival times between type A events follow an Exponential distribution with rate $\lambda p$.
x??

---

#### Poisson Process Independence
Background context: This section discusses how to prove that two Poisson processes are independent and form separate Poisson processes with their own rates. The key idea is using the joint probability of events in both processes.

:p What does it mean for NA(t) and NB(t) to be independent Poisson processes?
??x
To show independence, we need to demonstrate that the joint probability $P\{NA(t)=n, NB(t)=m\}$ can be expressed as the product of the individual probabilities. This is done by summing over all possible values of m in the equation provided.

The derivation uses properties of Poisson processes and their joint distribution:
$$P\{NA(t)=n, NB(t)=m\} = e^{-\lambda t} p (\lambda t p)^n \frac{n!}{n!} \times e^{-\lambda t (1-p)} (1 - p) (\lambda t (1 - p))^m \frac{m!}{m!}$$

This simplifies to:
$$

P\{NA(t)=n, NB(t)=m\} = e^{-\lambda t} p (\lambda t p)^n \times e^{-\lambda t (1-p)} (1 - p) (\lambda t (1 - p))^m$$
$$= e^{-\lambda t p} (\lambda t p)^n \times e^{-\lambda t (1-p)} (\lambda t (1 - p))^m$$

Thus, the joint probability is the product of individual probabilities:
$$

P\{NA(t)=n\} \cdot P\{NB(t)=m\} = e^{-\lambda t p} (\lambda t p)^n \times e^{-\lambda t (1-p)} (\lambda t (1 - p))^m$$

This shows that the processes are independent.
x??

---

#### Uniformity of Poisson Process Events
Background context: Given one event in a Poisson process, it is equally likely to have occurred at any point within the time interval.

:p What does Theorem 11.9 state about events occurring in a Poisson process?
??x
Theorem 11.9 states that if one event of a Poisson process occurs by time t, then this event is equally likely to have occurred anywhere in the interval $[0,t]$.

This can be shown using conditional probability:
$$P\{T_1 < s | N(t) = 1\} = \frac{P\{T_1 < s \text{ and } N(t) = 1\}}{P\{N(t) = 1\}}$$

Given that exactly one event occurs in $[0,t]$:
$$P\{1 \text{ event in } [0,s] \text{ and } 0 \text{ events in } (s, t)\} = e^{-\lambda t} (\lambda s)$$
$$

P\{1 \text{ event in } [0,t]\} = e^{-\lambda t} \lambda t$$

Thus:
$$

P\{T_1 < s | N(t) = 1\} = \frac{e^{-\lambda t} \lambda s}{e^{-\lambda t} \lambda t} = \frac{s}{t}$$

This means the event is uniformly distributed in $[0,t]$.
x??

---

#### Exponential Distribution Memorylessness
Background context: The memoryless property of an exponential distribution implies that the probability of an event occurring within a time interval, given it has not occurred yet, does not depend on how much time has already passed.

:p What does "memorylessness" mean in the context of the exponential distribution?
??x
Memorylessness means that for an exponentially distributed random variable $X \sim \text{Exp}(\lambda)$, the probability of an event occurring within a time interval given it hasn't occurred yet is independent of how much time has passed. Specifically, the conditional expectation $ E[X | X > 10]$ can be calculated in two ways:

1. Integrating the conditional PDF:
$$E[X | X > 10] = \int_{10}^{\infty} x f_X(x) dx$$where $ f_X(x) = \lambda e^{-\lambda x}$.

2. Using memorylessness directly:
$$E[X | X > 10] = 10 + E[X] = 10 + \frac{1}{\lambda}$$

Both methods yield the same result: the expected additional time is simply the mean of the exponential distribution, plus the initial interval.
x??

---

#### Doubling Exponential Distribution
Background context: If job sizes are exponentially distributed with rate $\mu$ and all double, we need to determine the new distribution.

:p How does doubling the size of exponentially distributed jobs affect their distribution?
??x
Doubling the size of exponentially distributed jobs changes the parameter of the exponential distribution. Originally, if $X \sim \text{Exp}(\mu)$, then the expected value is $\frac{1}{\mu}$. If we double each job size, let the new random variable be $ Y = 2X$.

The cumulative distribution function (CDF) of $Y$ is:
$$F_Y(y) = P(Y \leq y) = P(2X \leq y) = P(X \leq \frac{y}{2}) = 1 - e^{-\mu \frac{y}{2}}$$

This shows that the new distribution of job sizes,$Y$, is still exponentially distributed but with a halved rate parameter:
$$Y \sim \text{Exp}\left(\frac{\mu}{2}\right)$$

The mean and variance also adjust accordingly:
Mean:$E[Y] = \frac{1}{\frac{\mu}{2}} = \frac{2}{\mu}$ Variance:$\text{Var}[Y] = \left(\frac{2}{\mu}\right)^2 = \frac{4}{\mu^2}$ x??

---

#### Failure Rate of Exponential Distribution
Background context: The failure rate is a measure of how likely an item is to fail per unit time. For the exponential distribution, it is constant.

:p Prove that for the exponential distribution with rate $\lambda $, the failure rate $ r(t) = f(t) / F(t)$ is constant.
??x
For an exponential distribution with rate $\lambda$:
- The probability density function (PDF): $f(t) = \lambda e^{-\lambda t}$- The cumulative distribution function (CDF):$ F(t) = 1 - e^{-\lambda t}$ The failure rate is given by:
$$r(t) = \frac{f(t)}{F(t)} = \frac{\lambda e^{-\lambda t}}{1 - e^{-\lambda t}}$$

For small values of $t $, the term$ e^{-\lambda t} \approx 1 $for large$\lambda t$. Thus:
$$r(t) \approx \frac{\lambda}{1 - (1 - e^{-\lambda t})} = \frac{\lambda}{e^{-\lambda t}} = \lambda$$

This shows that the failure rate $r(t)$ is constant and equal to $\lambda$ for all $t$.

Additionally, we can prove it directly:
$$r(t) = \lim_{dt \to 0} \frac{P(t < T < t+dt)}{1 - F(t)} = \lim_{dt \to 0} \frac{\lambda dt}{e^{-\lambda t}} = \lambda$$

Thus, the failure rate is constant for exponential distributions.
x??

---

#### Poisson Process with Known Events
Background context: Given that $N$ green packets arrived during a second in a Poisson process, we can calculate expected values and probabilities related to yellow packets.

:p What is the expected number of yellow packets arriving if 100 green packets arrived in a previous second?
??x
Given that each packet has a probability of 5% (or $p = 0.05$) of being "green" and 95% of being "yellow", we can find the expected number of yellow packets.

If 100 green packets have arrived, then:
$$\text{Number of total packets} = \frac{\text{Number of green packets}}{p} = \frac{100}{0.05} = 2000$$

The expected number of yellow packets is:
$$

E[\text{yellow packets}] = (1 - p) \times \text{total packets} = 0.95 \times 2000 = 1900$$

So, the expected number of yellow packets arriving in that second is 1900.
x??

---

#### Conditional Distribution Given Minimum
Background context: If $X $ and$Y $ are independent exponential random variables with rates$\lambda_X $ and$\lambda_Y $, the minimum $ Z = \min(X, Y)$has a known distribution. We need to find the conditional distribution of $ X$given $ X < Y$.

:p Prove that if $X $ and$Y $ are independent exponential random variables with rates$\lambda_X $ and$\lambda_Y $, then$ P\{X > t | X < Y\} = P\{Z > t\}$.
??x
Given:
- $X \sim \text{Exp}(\lambda_X)$-$ Y \sim \text{Exp}(\lambda_Y)$We need to show that the conditional distribution of $ X$given $ X < Y $ is the same as the distribution of the minimum $Z = \min(X, Y)$.

The event $X < Y $ means we are only interested in values where$X $ is less than$ Y $. For a fixed $ t$, we need to find:
$$P\{X > t | X < Y\} = \frac{P\{X > t, X < Y\}}{P\{X < Y\}}$$

The probability that $X $ is greater than$t $ and less than $ Y$ is:
$$P\{X > t, X < Y\} = \int_{0}^{\infty} \left( \int_{x}^{\infty} f_X(x) f_Y(y) dy \right) dx$$

Where $f_X(x) = \lambda_X e^{-\lambda_X x}$ and $f_Y(y) = \lambda_Y e^{-\lambda_Y y}$:
$$P\{X > t, X < Y\} = \int_{0}^{\infty} \left( \int_{x}^{\infty} \lambda_X e^{-\lambda_X x} \lambda_Y e^{-\lambda_Y y} dy \right) dx$$
$$= \int_{0}^{\infty} \lambda_X e^{-\lambda_X x} \left[ -e^{-\lambda_Y y} \right]_x^\infty dx$$
$$= \int_{0}^{\infty} \lambda_X e^{-\lambda_X x} e^{-\lambda_Y x} dx$$
$$= \lambda_X e^{-\lambda_X t} \int_{t}^{\infty} e^{-(\lambda_X + \lambda_Y) x} dx$$
$$= \lambda_X e^{-\lambda_X t} \left[ -\frac{e^{-(\lambda_X + \lambda_Y)x}}{\lambda_X + \lambda_Y} \right]_t^\infty$$
$$= \frac{e^{-\lambda_X t}}{\lambda_X + \lambda_Y}$$

The probability that $X < Y$ is:
$$P\{X < Y\} = 1 - P\{Y < X\} = 1 - \int_{0}^{\infty} \left( \int_{y}^{\infty} f_X(x) f_Y(y) dx \right) dy$$
$$= 1 - \frac{\lambda_Y}{\lambda_X + \lambda_Y}$$

Thus:
$$

P\{X > t | X < Y\} = \frac{\frac{e^{-\lambda_X t}}{\lambda_X + \lambda_Y}}{1 - \frac{\lambda_Y}{\lambda_X + \lambda_Y}} = e^{-\lambda_X t}$$

This is the same as $P\{Z > t\}$ where $ Z $ follows an exponential distribution with rate $\lambda_X + \lambda_Y$.

Therefore, we have:
$$P\{X > t | X < Y\} = P\{Z > t\}$$x??

--- 
These flashcards cover key concepts in the provided text. Each card focuses on a specific aspect and includes relevant background information, formulas, and explanations to aid understanding. The questions are designed to test comprehension rather than pure memorization. 
--- 

Note: The code examples are not directly applicable for these theoretical concepts but could be used to illustrate practical applications if needed.

#### Definition of Discrete-Time Markov Chain (DTMC)
Background context: The provided text introduces a definition of a Discrete-Time Markov Chain (DTMC) and lists its three properties:
1. Transitions are always made at discrete time steps,$n = 0, 1, 2,...$.
2. The past does not matter; only the present state matters.
3. Transition probabilities are stationary.

:p What is the definition of a Discrete-Time Markov Chain (DTMC)?
??x
A DTMC is a stochastic process $\{X_n, n=0,1,2,...\}$ where $X_n$ denotes the state at discrete time step $n$. The key properties are:
- Transitions occur only at discrete time steps.
- Future states depend only on the present state (Markovian Property).
- Transition probabilities are stationary and independent of time.

Example code to simulate a simple DTMC might look like this:

```java
public class SimpleDTMC {
    private int[] states;
    
    public void transition(int currentState, double[][] transitionMatrix) {
        // Simulate the next state based on current state and transition matrix
        Random random = new Random();
        double r = random.nextDouble();  // Generate a random number between 0 and 1.
        
        for (int i = 0; i < transitionMatrix[currentState].length; i++) {
            if (r <= cumulativeProb(transitionMatrix[currentState], i)) {
                states[n] = i;
                break;
            }
        }
    }
    
    private double cumulativeProb(double[] row, int i) {
        // Calculate the cumulative probability
        return Arrays.stream(row).limit(i + 1).sum();
    }
}
```
x??

---

#### Definition of Continuous-Time Markov Chain (CTMC)
Background context: The text defines a Continuous-Time Markov Chain (CTMC) as a continuous-time stochastic process $\{X(t), t\geq0\}$ with properties analogous to DTMCs but allowing for transitions at any time.

:p What is the definition of a Continuous-Time Markov Chain (CTMC)?
??x
A CTMC is defined as a continuous-time stochastic process $\{X(t), t\geq 0\}$ such that:
- The past does not matter; only the current state matters.
- Transition probabilities are stationary, independent of time.

Formally, for any $s,t\geq0 $ and states$i,j$:
$$P\{X(s+t)=j|X(s)=i, X(u)=x(u), 0\leq u \leq s\} = P\{X(t+s)=j|X(s)=i\} = P_{ij}(t)$$where $ P_{ij}(t)$is the probability of transitioning from state $ i$to state $ j $ in time $t$.

Example code for generating an exponential random variable (time until transition):

```java
public class ExponentialGenerator {
    public static double generateExponential(double rate) {
        // Generate an exponentially distributed random number with given rate
        return -Math.log(1.0 - Math.random()) / rate;
    }
}
```
x??

---

#### Memorylessness of Transition Time (τi)
Background context: The transition time $\tau_i $ from state$i $ is memoryless, meaning the probability of leaving state$ i $ within the next $ t $ seconds depends only on the current state and not on how long it has been in state $i$.

:p What does the memorylessness property imply for τi?
??x
The memorylessness property implies that $\tau_i$ is exponentially distributed. This means:
$$P\{\tau_i > t + s | \tau_i > s\} = P\{\tau_i > t\}$$for any $ s, t \geq 0$.

This property allows us to define a CTMC where the time spent in state $i $ before transitioning is exponentially distributed with rate$\nu_i $, and the next state depends only on the transition probabilities $ p_{ij}$independent of the time spent in state $ i$.

Example code for simulating exponential distribution:

```java
public class ExponentialSimulation {
    public static double simulateTime(double rate) {
        // Simulate an exponentially distributed random variable with given rate
        return -Math.log(1.0 - Math.random()) / rate;
    }
}
```
x??

---

#### View 1 of CTMC
Background context: VIEW 1 of a CTMC is defined as follows:
- The process stays in state $i $ for an exponentially distributed time$\tau_i $ with rate$\nu_i$.
- When transitioning from state $i $, the next state $ j $is chosen with probability$ p_{ij}$.

:p What is VIEW 1 of a CTMC?
??x
VIEW 1 of a CTMC describes it as:
- The process stays in state $i $ for an exponentially distributed time$\tau_i $ with rate$\nu_i$.
- When transitioning from state $i $, the next state $ j $is chosen independently with probability$ p_{ij}$.

This view emphasizes that transitions are made at random times determined by exponential distributions, and the next state is chosen based on fixed transition probabilities.

Example code for simulating VIEW 1 of a CTMC:

```java
public class CTCMSimulation {
    private double[] rates; // Transition rates from each state to any other state.
    
    public void simulateCTMC(double initialState) {
        while (currentState != -1) { // -1 indicates termination or absorbing state
            double time = simulateExponential(rates[currentState]);
            
            // Determine next state based on transition probabilities
            int nextState = determineNextState();
            
            currentState = nextState;
        }
    }
    
    private double simulateExponential(double rate) {
        return -Math.log(1.0 - Math.random()) / rate;
    }
    
    private int determineNextState() {
        // Logic to determine the next state based on transition probabilities
        Random random = new Random();
        double r = random.nextDouble();
        
        for (int i = 0; i < transitionProbabilities.length; i++) {
            if (r <= cumulativeSum) {
                return i;
            }
        }
    }
}
```
x??

---

#### View 2 of CTMC
Background context: VIEW 2 of a CTMC is described as follows:
- The time to leave state $i $ and transition to another state$j $ is exponentially distributed with rate$\nu_i p_{ij}$.
- The minimum of these exponential times determines the next state.

:p What is VIEW 2 of a CTMC?
??x
VIEW 2 of a CTMC describes it as:
- The time until transitioning from state $i $ to another state$j $ is exponentially distributed with rate$\nu_i p_{ij}$.
- The minimum of these exponential times determines the next state.

This view highlights that transitions are determined by the earliest of several exponential random variables, each associated with a different possible transition and its probability.

Example code for simulating VIEW 2 of a CTMC:

```java
public class CTCMSimulation {
    private double[][] rates; // Rates from each state to any other state.
    
    public void simulateCTMC(double initialState) {
        while (currentState != -1) { // -1 indicates termination or absorbing state
            double minTime = Double.MAX_VALUE;
            
            for (int j = 0; j < states.length; j++) {
                if (rates[currentState][j] > 0 && rates[currentState][j] * probabilities[currentState][j] < minTime) {
                    minTime = rates[currentState][j] * probabilities[currentState][j];
                    nextState = j;
                }
            }
            
            double timeUntilTransition = simulateExponential(1.0 / (minTime));
            
            currentState = nextState;
        }
    }
    
    private double simulateExponential(double rate) {
        return -Math.log(1.0 - Math.random()) / rate;
    }
}
```
x??

---

#### Modeling a Single-Server Network as CTMC
Background context: The example provided models a single-server network using a CTMC where:
- The state is the number of jobs in the system.
- Jobs arrive according to a Poisson process with rate $\lambda$.
- Service demand follows an exponential distribution with rate $\mu$.

:p How can a single-server network be modeled as a CTMC?
??x
A single-server network can be modeled as a CTMC by considering:
- States: The number of jobs in the system.
- Arrival process: Jobs arrive according to a Poisson process with rate $\lambda$.
- Service time: Each job's service demand follows an exponential distribution with rate $\mu$.

This model captures the dynamics of arrivals and departures, allowing for transitions between states based on these processes.

Example code to simulate this CTMC:

```java
public class SingleServerCTMC {
    private int currentState; // Number of jobs in the system.
    
    public void simulateSingleServerCTMC(double lambda, double mu) {
        while (currentState != -1) { // -1 indicates termination or absorbing state
            double arrivalTime = simulatePoisson(lambda);
            double serviceTime = simulateExponential(mu);
            
            if (arrivalTime < serviceTime) {
                currentState += 1; // Arrival happens first, increase number of jobs.
            } else {
                currentState -= 1; // Service completes before next arrival, decrease number of jobs.
            }
        }
    }
    
    private double simulatePoisson(double rate) {
        double u = Math.random();
        return (-Math.log(1 - u) / rate);
    }
    
    private double simulateExponential(double rate) {
        return -Math.log(1.0 - Math.random()) / rate;
    }
}
```
x?? 

--- 
This format can be used to create multiple flashcards covering the key concepts in the provided text, ensuring that each card focuses on a single question with detailed explanations and relevant code examples where applicable.

