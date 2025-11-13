# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 20)

**Starting Chapter:** 12.2 Solving CTMCs

---

#### Time to Next Event in CTMCs
Background context explaining that events in a Continuous-Time Markov Chain (CTMC) are arrivals and departures. The time until the next arrival $X_A \sim Exp(\lambda)$ and the time until the next departure $X_D \sim Exp(\mu)$. The time to leave state $ i$, regardless of how long we have been in that state, is given by:
$$\tau_i \sim Exp(\lambda + \mu)$$:p What does $\tau_i$ represent?
??x $\tau_i $ represents the time until we leave state$i $. It follows an exponential distribution with a rate of$\lambda + \mu$, which is the combined rate of arrivals and departures.
x??

---

#### Transition Probabilities in CTMCs
Background context explaining that the probability of leaving state $i $ to state$i+1 $ when$X_A < X_D$ is:
$$P(X_A < X_D) = \frac{\lambda}{\lambda + \mu}$$:p What is the probability that an arrival occurs before a departure?
??x
The probability that an arrival occurs before a departure is given by:
$$

P(X_A < X_D) = \frac{\lambda}{\lambda + \mu}$$

This means that out of all possible transitions, the fraction of time we expect to observe an arrival first is $\frac{\lambda}{\lambda + \mu}$.
x??

---

#### Limiting Probabilities in CTMCs
Background context explaining that if a DTMC can be used to model a CTMC by approximating it with small steps $\delta $, the solution to the original CTMC equals the solution to the equivalent DTMC as $\delta \to 0$.

:p How do we determine the limiting probabilities $\pi_j$ in a CTMC?
??x
To determine the limiting probabilities $\pi_j $ in a CTMC, we can use a Discrete-Time Markov Chain (DTMC) that approximates the CTMC by making transitions at every small step$\delta $. As $\delta \to 0$, the solution to the original CTMC equals the solution to this DTMC.

The key steps involve solving balance equations for the equivalent DTMC. For a simple example, consider the following balance equation:
$$\pi_0 (\lambda \delta + o(\delta)) = \pi_1 (\mu \delta + o(\delta))$$

Dividing by $\delta $ and taking the limit as$\delta \to 0$:
$$\frac{\pi_0}{\lambda} = \frac{\pi_1}{\mu}$$

Thus:
$$\pi_1 = \frac{\lambda}{\mu} \pi_0$$

For state 1, we have the equation:
$$\pi_1 (\lambda + \mu) = \pi_0 \lambda + \pi_2 \mu$$

Using the previous result for $\pi_1$:
$$\pi_2 = \left(\frac{\lambda}{\mu}\right)^2 \pi_0$$

The limiting probabilities are then determined by normalizing these values so that they sum to 1.
x??

---

#### Transition Probabilities in DTMC Approximation
Background context explaining the approximation of a CTMC using a DTMC with small steps $\delta$. The transition probabilities are derived from flipping two coins simultaneously at each step.

:p How do we model the transitions in the equivalent DTMC?
??x
We can model the transitions by flipping two coins simultaneously every $\delta$-step. One coin represents arrivals, and the other represents departures. If a "flip" occurs:
- With probability $\lambda \delta (1 - \mu \delta)$, an arrival happens with no departure.
- With probability $(1 - \lambda \delta) \mu \delta$, a departure happens with no arrival.
- With probability $\lambda \delta \mu \delta$, both an arrival and a departure happen.
- With probability $1 - (\text{all of the above})$, nothing happens.

As $\delta \to 0$, these probabilities simplify to:
$$P(\text{arrival, no departure}) = \lambda \delta (1 - \mu \delta) + o(\delta)$$
$$

P(\text{departure, no arrival}) = (1 - \lambda \delta) \mu \delta + o(\delta)$$
$$

P(\text{both}) = \lambda \delta \mu \delta + o(\delta)$$
$$

P(\text{nothing}) = 1 - (\lambda \delta + \mu \delta) + o(\delta)$$

This setup is equivalent to a DTMC with transition probabilities:
```java
public class TransitionModel {
    public void update(double lambda, double mu) {
        // Update the transition probabilities based on small steps delta
        double pArrival = lambda * (1 - mu);
        double pDeparture = (1 - lambda) * mu;
        double pBoth = lambda * mu;
        double pNothing = 1 - (lambda + mu);
    }
}
```
x??

---

#### Solving for Limiting Probabilities in CTMC
Background context explaining that the limiting probabilities can be found by solving balance equations derived from the equivalent DTMC.

:p How do we solve for the limiting probabilities $\pi_j$ using the equivalent DTMC?
??x
To find the limiting probabilities $\pi_j$, we use the balance equations derived from the equivalent DTMC. For example, consider a simple two-state system where:
$$\pi_0 (\lambda + o(\delta)) = \pi_1 (\mu + o(\delta))$$

Dividing by $\delta $ and taking the limit as$\delta \to 0$:
$$\frac{\pi_0}{\lambda} = \frac{\pi_1}{\mu}$$

Thus:
$$\pi_1 = \frac{\lambda}{\mu} \pi_0$$

For state 1, we have the equation:
$$\pi_1 (\lambda + \mu) = \pi_0 \lambda + \pi_2 \mu$$

Using the previous result for $\pi_1$:
$$\pi_2 = \left(\frac{\lambda}{\mu}\right)^2 \pi_0$$

The limiting probabilities are then determined by normalizing these values so that they sum to 1.
x??

---

#### Transition to Continuous-Time Markov Chains (CTMCs)
Background context: The transition from a Discrete-Time Markov Chain (DTMC) to a CTMC involves understanding how rates of transitions can be converted into probabilities. This process helps in approximating the behavior of continuous-time processes using discrete steps.

:p What is the relationship between rate and probability in the context of transitioning from a DTMC to a CTMC?
??x
The relationship lies in the approximation where we convert exponential rates (λ, μ) to transition probabilities within small time intervals. Specifically, for small δ:
- Rate leaving state 2:$\lambda \delta + o(\delta)$- Rate entering state 2:$\mu \delta + o(\delta)$ This conversion allows us to derive balance equations similar to those in a DTMC.
??x
The answer is that rates (λ, μ) are approximated as probabilities within small time intervals. For example:
$$\pi_2 (\lambda \delta + o(\delta)) = \pi_1 (\lambda \delta + o(\delta)) + \pi_3 (\mu \delta + o(\delta))$$

This simplifies to:
$$\pi_2 \lambda = \pi_1 \lambda + \pi_3 \mu$$

Thus, the balance equations for the DTMC are derived from these approximations.
??x

---

#### Balance Equations in CTMC
Background context: The balance equations derived from the approximation of rates to probabilities help us understand how states transition over time. These equations resemble the balance conditions of a DTMC.

:p What do the balance equations (12.1), (12.2), and (12.3) represent?
??x
The balance equations are:
$$\pi_0 (\lambda) = \pi_1 (\mu)$$
$$\pi_1 (\lambda + \mu) = \pi_0 (\lambda) + \pi_2 (\mu)$$
$$\pi_2 (\lambda + \mu) = \pi_1 (\lambda) + \pi_3 (\mu)$$

These equations ensure that the rates of leaving a state equal the rates of entering it, maintaining balance in the system.
??x
The answer is that these balance equations represent the condition where the rate at which states are left equals the rate at which they are entered. They are derived from approximating exponential transition rates (λ, μ) to probabilities over small time intervals.

---

#### Generalization and Interpretation of Balance Equations
Background context: The method for converting a CTMC into a DTMC by using balance equations can be generalized to any CTMC. This involves modeling the CTMC with discrete steps and solving for limiting probabilities.

:p How does this method work for generalizing from CTMC to DTMC?
??x
The process involves:
1. Starting with a general CTMC.
2. Considering a single state $i$ and modeling it in a DTMC framework.
3. Approximating the exponential rates (λ, μ) as probabilities over small time intervals.
4. Writing out balance equations for the DTMC.

These balance equations will yield the same limiting probabilities as the original CTMC.
??x
The answer is that the method involves:
1. Starting with a general CTMC and choosing an arbitrary state $i$.
2. Approximating exponential rates (λ, μ) to probabilities over small time intervals in a DTMC framework.
3. Writing out balance equations for the DTMC.

For example, if we are in state $i $, on most δ-step transitions, we return to state $ i $. This exactly models sitting in state$ i$ for a while before transitioning:
```java
public class StateModel {
    public void transition(double lambda, double mu, double delta) {
        // Transition logic based on probabilities and rates
        if (Math.random() < 1 - lambda * delta - mu * delta) {
            stayInStateI();
        } else if (Math.random() < 1 - mu * delta) {
            moveFromStateIToJ();
        } else {
            moveToStateK();
        }
    }

    private void stayInStateI() { /* Logic to stay in state I */ }
    private void moveFromStateIToJ() { /* Logic to transition from I to J */ }
    private void moveToStateK() { /* Logic to transition to K */ }
}
```
??x
The answer is that the method involves converting exponential rates (λ, μ) into probabilities within small time intervals (δ), modeling this in a DTMC framework, and solving for balance equations. This yields the same limiting probabilities as the original CTMC.
??x

---

#### Transition to Continuous-Time Markov Chains (CTMC)
In practice, we do not always need to translate a CTMC into a discrete-time Markov chain (DTMC) with δ-steps. We can directly derive balance equations for the CTMC and solve them for the limiting probabilities πi.

:p What is the primary advantage of working directly with continuous-time Markov chains?
??x
By working directly with CTMCs, we avoid the complexity and potential inaccuracies introduced by translating to a discrete-time framework, making it easier to derive and solve balance equations.
x??

---

#### Balance Equations in CTMCs
The balance equations for a CTMC are derived from the principle that the rate at which jobs leave state j equals the rate at which they enter state j. The standard notation is:

$$π_jν_j = \sum_{i} π_i q_{ij}$$where:
- $π_j$ is the limiting probability of being in state j.
- $ν_j$ is the transition rate out of state j.
- $q_{ij}$ is the transition rate from state i to state j.

:p What do the balance equations represent in a CTMC?
??x
The balance equations represent the equality between the total rate at which jobs leave state j and the total rate at which jobs enter state j. This ensures that there is no net flow of probability into or out of state j in the long run.
x??

---

#### Interpreting Balance Equations for CTMCs
The left-hand side (LHS) of the balance equation represents the product of the limiting probability $π_j$ and the transition rate out of state j, νj. The right-hand side (RHS) is a sum over all states i, where each term represents the product of the limiting probability of being in state i and the transition rate from state i to state j.

:p What does the left-hand side of the balance equation represent?
??x
The left-hand side of the balance equation represents the total rate at which transitions leave state j. It is calculated as $π_j \cdot ν_j $, where $π_j $ is the limiting probability of being in state j, and$ν_j$ is the transition rate out of state j.
x??

---

#### Interpreting Balance Equations for CTMCs (continued)
The ith term on the RHS represents the product of the limiting probability $π_i $ and the transition rate$q_{ij}$, which is the rate at which transitions from state i to state j occur. The sum over all states i on the RHS gives the total rate at which transitions enter state j.

:p What does each term in the summand of the right-hand side (RHS) represent?
??x
Each term in the summand of the RHS represents the rate at which transitions leave state i to go to state j. It is calculated as $π_i \cdot q_{ij}$, where $π_i $ is the limiting probability of being in state i, and $q_{ij}$ is the transition rate from state i to state j.
x??

---

#### Summary Theorem for CTMCs
For an irreducible CTMC with πi’s that satisfy the balance equations:
$$π_jν_j = \sum_{i} π_i q_{ij}$$and$$\sum_{i} π_i = 1$$the πi's are the limiting probabilities for the CTMC, and the CTMC is ergodic.

:p What does the Summary Theorem state about the limiting probabilities of an irreducible CTMC?
??x
The Summary Theorem states that if there exist πi’s such that they satisfy both the balance equations $π_jν_j = \sum_{i} π_i q_{ij}$ and the normalization condition $\sum_{i} π_i = 1$, then these πi's are the limiting probabilities for the CTMC, and the CTMC is ergodic.
x??

---

#### Converting a CTMC to a DTMC
The provided figure (Figure 12.10) shows a simple CTMC with states 1, 2, and 3, and transition rates λ31, λ12, λ21, and λ32.

:p How can we model the given CTMC as a DTMC?
??x
To convert the CTMC to a DTMC, we introduce a small time step δ. The rate of transitions between states in the DTMC will be $\frac{λ_{ij}}{\delta}$. For example, for state 1, the transition rates would be:

$$p_{12} = \frac{λ_{12}}{\delta}, \quad p_{13} = \frac{λ_{13}}{\delta}$$

Similarly, for other states. The balance equations in the DTMC can then be derived and taken to the limit as δ → 0 to obtain the balance equations for the original CTMC.
x??

---

#### Potential Pitfall: Balance vs Stationary Equations
For a CTMC, the balance equations yield the limiting probabilities directly. However, stationary equations are meaningless unless they are first translated into a DTMC.

:p What is the difference between balance equations and stationary equations in the context of CTMCs?
??x
Balance equations for CTMCs give the limiting probabilities directly, while stationary equations for CTMCs do not have a meaningful interpretation until the CTMC is translated into a DTMC. The stationary equations for a CTMC are equivalent to the balance equations only after such a translation.
x??

---

