# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 12)

**Starting Chapter:** 8.3 Examples of Finite-State DTMCs

---

#### Discrete-Time Markov Chains (DTMCs) vs. Continuous-Time Markov Chains (CTMCs)
Background context: DTMCs and CTMCs are two types of stochastic processes used to model systems over time. The primary difference between them is that DTMCs operate in discrete-time steps, while CTMCs can model events happening at any point in continuous time.
:p What is the key difference between Discrete-Time Markov Chains (DTMCs) and Continuous-Time Markov Chains (CTMCs)?
??x
The key difference is that in a DTMC, events occur only at discrete time steps, whereas in a CTMC, events can happen continuously over time. This makes CTMCs more suitable for modeling systems where events can occur at any moment.
x??

---

#### Definition of Discrete-Time Markov Chains (DTMCs)
Background context: A DTMC is defined as a stochastic process $\{X_n, n=0,1,2,...\}$ where $X_n$ denotes the state at time step $n$. The key properties are stationarity and the Markovian property. Stationarity ensures that transition probabilities do not change over time, while the Markovian property states that future states depend only on the current state.
:p What is the definition of a Discrete-Time Markov Chain (DTMC)?
??x
A Discrete-Time Markov Chain (DTMC) $\{X_n, n=0,1,2,...\}$ is defined such that for any $n \geq 0$ and states $i, j$, the transition probability from state $ i$to state $ j$at time $ n+1$given the present state is:
$$P(X_{n+1} = j | X_n = i, X_{n-1} = i_{n-1}, ..., X_0 = i_0) = P(X_{n+1} = j | X_n = i) = P_{ij}$$where $ P_{ij}$is the transition probability from state $ i$to state $ j$ and does not depend on past states.
x??

---

#### Transition Probability Matrix
Background context: The transition probability matrix, denoted by $P $, for a DTMC has entries $ P_{ij}$representing the probability of moving to state $ j$in one step from state $ i$. This matrix is crucial in understanding how states transition over time.
:p What is the transition probability matrix and what does it represent?
??x
The transition probability matrix, denoted by $P $, for a DTMC is an $(M \times M)$ matrix where each entry $P_{ij}$ represents the probability of moving from state $i$ to state $j$ in one step. The key property is that:
$$\sum_{j=1}^{M} P_{ij} = 1, \forall i$$

This ensures that given a current state $i$, the sum of probabilities of transitioning to any other state must be 1.
x??

---

#### Repair Facility Problem
Background context: This problem involves a machine that can either be working or broken. The states and transition probabilities are given explicitly in this example, illustrating how to model real-world scenarios using DTMCs.
:p Describe the DTMC for the repair facility problem.
??x
The DTMC has two states: "Working" (W) and "Broken" (B). The transition probability matrix is:
$$P = \begin{bmatrix} 0.95 & 0.05 \\ 0.40 & 0.60 \end{bmatrix}$$where $ P_{ij}$is the probability of moving from state $ i$to state $ j$. For example, the probability that a machine transitions from "Working" (W) to "Broken" (B) in one step is 0.05.
x??

---

#### Umbrella Problem
Background context: The umbrella problem involves an absent-minded professor who has two umbrellas and needs to decide whether to take one when it rains. This example helps illustrate how DTMCs can model scenarios with probabilistic decisions.
:p What is the state space for the umbrella problem?
??x
The state space for the umbrella problem consists of three states: 0 umbrellas available, 1 umbrella available, and 2 umbrellas available. The transition probability matrix $P$ captures the probabilities of moving between these states:
$$P = \begin{bmatrix} 0 & 1-p & p \\ 1-p & 0 & p \\ 0 & 1-p & p \end{bmatrix}$$where $ p$ is the probability that it rains during a commute. This matrix reflects the logic of whether an umbrella is taken or left based on its availability and weather conditions.
x??

---

#### Program Analysis Problem
Background context: The program analysis problem involves tracking different types of instructions in a program, which can be modeled using DTMCs to understand their behavior over time.
:p What are the states for the program analysis problem?
??x
The states for the program analysis problem track the types of instructions available at any given point in time. Specifically, there are three states: "CPU instruction" (C), "Memory instruction" (M), and "User interaction instruction" (U).
x??

---

#### n-Step Transition Probabilities
Background context: The transition probability matrix $P $ represents the probabilities of moving from one state to another in a single step. When we consider the$n $-step transition probabilities, denoted as$ P^n_{ij}$, it gives the probability of transitioning from state $ i$to state $ j$ in exactly $ n$ steps.
:p What does $P^n_{ij}$ represent?
??x
$P^n_{ij}$ represents the probability of moving from state $ i $ to state $ j $ in exactly $ n $ steps. This is calculated as the $(i,j)$-th entry of the matrix obtained by multiplying the transition probability matrix $ P$with itself $ n$ times.
```java
// Pseudo-code for calculating P^n_ij
Matrix multiply(Matrix P, int n) {
    if (n == 1) return P; // Base case: n-step is just the original transition matrix
    Matrix result = multiply(P, n / 2);
    result = result * result; // Square the matrix
    if (n % 2 != 0) result = result * P; // If n is odd, one more multiplication by P is needed
    return result;
}
```
x??

---

#### Umbrella Problem Example
Background context: The umbrella problem involves a transition probability matrix $P $, and we compute the probabilities over multiple steps using $ P^n $. For example, with$ p = 0.4$, the initial transition matrix is given as:
$$P = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 0.6 & 0.4 \\ 0.6 & 0.4 & 0 \end{bmatrix}$$

The problem asks us to consider the states of having no umbrellas, one umbrella, and two umbrellas.
:p What is the meaning of $P^5$ in this context?
??x $P^5 $ represents the probability distribution after five steps (days), starting from any initial state. Each entry in the matrix$P^5$ gives the probability of transitioning to a particular state in five steps.
```java
// Example Java code for computing P^n using a simple matrix multiplication approach
Matrix multiply(Matrix A, Matrix B) {
    // Perform matrix multiplication logic here
}

Matrix computePn(int n) {
    Matrix initialP = new Matrix(3, 3, new double[][]{
        {0.0, 0.0, 1.0},
        {0.0, 0.6, 0.4},
        {0.6, 0.4, 0.0}
    });
    
    if (n == 1) return initialP; // Base case
    Matrix result = multiply(initialP, computePn(n - 1)); // Recursively multiply the matrix
    return result;
}
```
x??

---

#### Repair Facility Problem Example
Background context: The repair facility problem uses a general transition probability matrix $P$:
$$P = \begin{bmatrix} 1-a & a & b \\ b & 1-b & a \\ a & b & 1-b \end{bmatrix}$$

We need to find the $n $-step transition probabilities and observe the behavior as $ n$ approaches infinity.
:p What does the matrix $P^n $ approach as$n$ becomes very large?
??x
As $n $ becomes very large, the matrix$P^n$ converges to a steady-state matrix where all rows are the same. The limiting probabilities can be found by observing that each row approaches a common vector:
$$\lim_{n \to \infty} P^n = \begin{bmatrix} \frac{a+b}{2(a+b)} & \frac{a+b}{2(a+b)} & \frac{a+b}{2(a+b)} \\ \frac{a+b}{2(a+b)} & \frac{a+b}{2(a+b)} & \frac{a+b}{2(a+b)} \\ \frac{a+b}{2(a+b)} & \frac{a+b}{2(a+b)} & \frac{a+b}{2(a+b)} \end{bmatrix} = \begin{bmatrix} \frac{1}{2} & \frac{1}{2} & 0 \\ \frac{1}{2} & \frac{1}{2} & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

This indicates that in the long run, states are equally likely to be visited.
```java
// Pseudo-code for finding steady-state probabilities
Matrix findSteadyState(double a, double b) {
    Matrix steadyState = new Matrix(3, 3, new double[][]{
        {0.5, 0.5, 0},
        {0.5, 0.5, 0},
        {0, 0, 1}
    });
    return steadyState;
}
```
x??

---

#### Limiting Probabilities
Background context: The limiting probabilities represent the long-term behavior of a discrete-time Markov chain (DTMC). As $n $ approaches infinity, the entries in$P^n$ approach these values. For example, if we start with state 0 and want to find the probability of being in state 1 after many steps, this is given by the corresponding entry in the steady-state matrix.
:p What does the limit as $n$ approaches infinity represent?
??x
The limit as $n$ approaches infinity represents the long-term or limiting probabilities of being in each state of a DTMC. These are the probabilities that describe the behavior of the system over an infinite amount of time, assuming the initial conditions are followed for all steps.
```java
// Pseudo-code for finding limiting probabilities
Matrix findLimitingProbabilities(Matrix P) {
    // Implement logic to compute steady-state probabilities using power method or other algorithms
    return steadyStateMatrix;
}
```
x??

---

#### Concept of Stationary Distribution and Limiting Probability
The concept revolves around understanding how, as $n \to \infty $, the probabilities of being in any state stabilize to a certain distribution. This is denoted by $\pi_j = \lim_{n \to \infty} P^n_{ij}$, where $ P$is the transition probability matrix and $ i, j$ are states.

The stationary distribution $\vec{\pi} = (\pi_0, \pi_1, ..., \pi_{M-1})$ satisfies the equation $\vec{\pi} \cdot P = \vec{\pi}$ where $\sum_{i=0}^{M-1} \pi_i = 1$.

:p What does the stationary distribution represent in a Markov Chain?
??x
The stationary distribution represents the limiting probabilities of being in each state as $n \to \infty$ and is independent of the initial state. This means that if we start with any distribution, after many transitions, the system's state probability will stabilize to this stationary distribution.
x??

---

#### Concept of Compressed Umbrella Problem
In the umbrella problem, where the states are based on the number of umbrellas available at a location and whether it is raining or not, the concept of symmetry in states 1 and 2 can be observed. If we consider only having 0 or 1 versus 2 umbrellas, the transition matrix becomes symmetric.

:p Why does the limiting probability of having 1 umbrella equal that of having 2 umbrellas?
??x
This is due to the symmetry in the problem when considering states with respect to 1 vs. 2 umbrellas. By compressing the state space and noticing that the transitions are equivalent, the probabilities of being in these states stabilize to the same value.
x??

---

#### Concept of Stationary Equations for Markov Chains
The stationary equations for a finite-state discrete-time Markov chain (DTMC) involve finding a probability distribution $\vec{\pi}$ such that $\vec{\pi} \cdot P = \vec{\pi}$, where $ P$ is the transition matrix. This means that if the system starts in this distribution, it will remain in the same distribution after any number of transitions.

:p How do we determine $\pi_j = \lim_{n \to \infty} P^n_{ij}$?
??x
To determine the stationary probability $\pi_j $, we can solve the stationary equations. Specifically, for a finite-state DTMC with $ M $states, find a vector$\vec{\pi} = (\pi_0, \pi_1, ..., \pi_{M-1})$ such that $\sum_{i=0}^{M-1} \pi_i = 1$ and $\pi_j = \sum_{k=0}^{M-1} \pi_k P_{kj}$ for all $j$.

Here's the logic in pseudocode:
```pseudocode
function findStationaryDistribution(P):
    // Initialize pi vector with equal probabilities (for simplicity)
    let pi = [1/M, 1/M, ..., 1/M]
    
    // Iterate until convergence or max iterations reached
    for each iteration do:
        newPi = []
        for j from 0 to M-1 do:
            sum = 0
            for k from 0 to M-1 do:
                sum += pi[k] * P[k][j]
            newPi.append(sum)
        
        // Check convergence or update pi
        if closeEnough(newPi, pi):
            return newPi
        else:
            pi = newPi
    
    return pi
```
x??

---

#### Concept of Theorem for Stationary and Limiting Distribution Equivalence
For a finite-state DTMC, the theorem states that the stationary distribution obtained by solving the stationary equations is unique and represents the limiting probabilities if these exist.

:p How does the theorem relate the limiting distribution to the stationary distribution?
??x
The theorem says that for a finite-state DTMC with $M $ states, if$\pi_j = \lim_{n \to \infty} P^n_{ij}$ exists, then this $\vec{\pi}$ is also a stationary distribution and no other stationary distribution can exist. This means the limiting probabilities are exactly the same as the unique stationary distribution.

The proof involves showing that:
1. The limiting distribution satisfies the stationary equations.
2. Any stationary distribution must be equal to the limiting distribution.

This equivalence helps in determining the long-term behavior of the Markov chain without having to compute many transition steps.
x??

---

#### Concept of Steady State for Markov Chains
A Markov Chain is said to be stationary or in steady state if it has a unique stationary distribution $\vec{\pi}$ and the initial state is chosen according to these stationary probabilities.

:p What does "stationary" mean for a Markov chain?
??x
For a Markov chain, being "stationary" means that once the system reaches this state (where probabilities of being in any state are given by $\vec{\pi}$), it remains there indefinitely. This is achieved if the initial distribution matches $\vec{\pi}$, ensuring no matter the starting point, after many transitions, the probability distribution stabilizes to $\vec{\pi}$.
x??

---

