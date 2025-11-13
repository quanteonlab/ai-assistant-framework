# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 15)

**Starting Chapter:** 9.5 Time Averages

---

#### Ergodicity and Irreducibility
Background context explaining the concept. In a Discrete-Time Markov Chain (DTMC), ergodicity is a desirable property that simplifies analysis. Specifically, if a DTMC is both irreducible and aperiodic, it can be shown that it has a unique stationary distribution which also serves as its limiting probability distribution.
:p What does the summary theorem tell us about determining whether our DTMC's limiting probability distribution?
??x
The summary theorem states that we do not need to determine positive recurrence; instead, checking for irreducibility and aperiodicity suffices. Once these conditions are met, solving the stationary equations yields both the stationary distribution and the limiting probability distribution.
x??

---

#### Periodic Chains and Their Solutions
Background context explaining the concept. When dealing with DTMCs that are not irreducible or periodic, the stationary equations may still have solutions, but their interpretation differs from when the chain is positive recurrent and irreducible.
:p What happens to the solution of the stationary equations in a periodic chain?
??x
In a periodic chain, if the stationary equations do yield a solution, it does not represent the limiting probability distribution. Instead, it represents the long-run time-average fraction of time spent in each state. This is different from the limiting probability distribution.
x??

---

#### Time Averages and Long-Run Behavior
Background context explaining the concept. The time average fraction $p_j $ of time spent in state$j $ can be defined as the limit of the ratio of the number of times the Markov chain enters state$ j $ by time $ t $, to $ t$. This is an important measure for understanding long-run behavior.
:p How is $p_j$ defined?
??x $p_j $ is defined as the time-average fraction of time that the Markov chain spends in state$j$ and can be expressed as:
$$p_j = \lim_{t \to \infty} \frac{N_j(t)}{t}$$where $ N_j(t)$is the number of times the Markov chain enters state $ j$by time $ t$.
x??

---

#### Positive Recurrence and Irreducibility
Background context explaining the concept. For a positive recurrent, irreducible DTMC, Theorem 9.28 provides strong guarantees about the convergence of the time averages to the limiting probabilities.
:p What does Theorem 9.28 tell us for a positive recurrent and irreducible Markov chain?
??x
Theorem 9.28 states that for a positive recurrent and irreducible Markov chain, with probability 1:
$$p_j = \lim_{t \to \infty} \frac{N_j(t)}{t} = \frac{1}{m_{jj}}$$where $ m_{jj}$is the mean number of time steps between visits to state $ j$. This theorem ensures that the time averages converge to the limiting probability $π_j$ and also provides a way to compute it.
x??

---

#### Corollary 9.29 - Ergodic DTMC
Background context explaining the concept. A corollary to Theorem 9.28, specifically for ergodic (irreducible and aperiodic) Markov chains, relates time averages to limiting probabilities in a straightforward manner.
:p What does Corollary 9.29 state about an ergodic DTMC?
??x
For an ergodic DTMC:
$$p_j = \pi_j = \frac{1}{m_{jj}}$$where $ p_j $ is the time-average fraction of time spent in state $ j $, and$π_j$ is the limiting probability. This corollary essentially connects the long-run behavior described by time averages to the stationary distribution.
x??

---

#### Summation of Limiting Probabilities
Background context explaining the concept. The fact that the sum of all limiting probabilities must equal 1 for a Markov chain is derived from the properties of ergodic chains and their convergence.
:p What does Corollary 9.30 state about the limiting probabilities in an ergodic DTMC?
??x
Corollary 9.30 states that for an ergodic DTMC, the sum of all limiting probabilities must equal 1:
$$\sum_{j=0}^{\infty} π_j = 1$$

This is derived from the fact that $p_j = π_j $ and the time averages$p_j$ are defined such that they sum to 1 over all states.
x??

---

#### Strong Law of Large Numbers (SLLN)
Background context explaining the concept. The SLLN provides a foundational result for understanding the convergence of time averages in sequences of independent, identically distributed random variables.
:p What does Theorem 9.31 state about the sequence of random variables $X_1, X_2, \ldots$?
??x
Theorem 9.31 (SLLN) states that for a sequence of independent and identically distributed (i.i.d.) random variables $X_1, X_2, \ldots $ each with mean$E[X]$, the average:
$$S_n = \frac{1}{n} \sum_{i=1}^n X_i$$converges to $ E[X]$with probability 1 as $ n$ approaches infinity.
x??

---

#### Renewal Process
Background context explaining the concept. A renewal process is a stochastic process where the inter-event times are i.i.d. random variables, each drawn from a distribution $F$. This concept is fundamental in understanding the long-term behavior of certain systems.
:p What is a renewal process?
??x
A renewal process is any process for which the times between events (inter-arrival times) are independent and identically distributed (i.i.d.) random variables with some common distribution $F$. For example, if we consider a sequence of arrivals where each inter-event time follows the same distribution, this forms a renewal process.
x??

---

#### Renewal Theorem
The Renewal Theorem states that for a renewal process, the long-run average number of events per unit time converges to 1/E[X] almost surely as t approaches infinity. Here, E[X] is the expected value of the inter-renewal times.
:p What does the Renewal Theorem state about the long-run behavior of a renewal process?
??x
The Renewal Theorem states that for a renewal process with mean inter-renewal time $E[X]$, the ratio of the number of renewals to time converges almost surely to $1/E[X]$ as time $t$ approaches infinity. Mathematically, this is expressed as:
$$\lim_{t \to \infty} \frac{N(t)}{t} = \frac{1}{E[X]} \text{ with probability 1.}$$

This means that over a long period of time, the average number of renewals per unit time approaches $\frac{1}{E[X]}$.
x??

---

#### Proof of Renewal Theorem
The proof involves applying the Strong Law of Large Numbers (SLLN) to show that both upper and lower bounds on the renewal process converge almost surely to $E[X]$. Specifically, it shows:
$$S_{N(t)} / N(t) \to E[X] \text{ as } t \to \infty$$and$$(S_{N(t)} + 1) / (N(t) + 1) \to E[X] \text{ as } t \to \infty.$$:p How does the proof of the Renewal Theorem use SLLN?
??x
The proof uses the Strong Law of Large Numbers (SLLN), which states that for a sequence of independent and identically distributed random variables $X_1, X_2, \ldots $ with mean$E[X]$:
$$\frac{\sum_{i=1}^n X_i}{n} \to E[X] \text{ almost surely as } n \to \infty.$$

In the context of the Renewal Theorem:
- Let $S_n = \sum_{i=1}^n X_i$ be the sum of inter-renewal times up to the nth renewal.
- By SLLN, for large $t$:
$$\frac{S_{N(t)}}{N(t)} \to E[X] \text{ almost surely.}$$- Similarly,$$\frac{S_{N(t)} + 1}{N(t) + 1} \to E[X] \text{ almost surely.}$$

These two expressions sandwich the ratio $N(t)/t$, leading to:
$$\frac{N(t)}{t} \to \frac{1}{E[X]} \text{ almost surely.}$$x??

---

#### Ergodic Markov Chain and Limiting Probabilities
For an ergodic (irreducible and positive recurrent) Markov chain, the limiting probability $\pi_i$ of being in state i is the long-run proportion of time that the process spends in state i.
:p What does the limiting probability $\pi_i$ represent for an ergodic Markov chain?
??x
The limiting probability $\pi_i$ represents the long-run proportion of time that a stationary and ergodic (irreducible and positive recurrent) Markov chain spends in state i. This is formally defined as:
$$\pi_i = \lim_{t \to \infty} P(X_t = i)$$where $ X_t$ is the state at time t.
x??

---

#### Transition Rates
For a Markov chain, the rate of transitions out of state i can be calculated as $\sum_j \pi_i P_{ij}$, and the rate of transitions into state i from any other state j is given by $\sum_j \pi_j P_{ji}$. The stationary equations then relate these rates.
:p What does $\sum_j \pi_i P_{ij}$ represent in a Markov chain?
??x
The expression $\sum_j \pi_i P_{ij}$ represents the total rate of transitions out of state i. This includes both direct exits from state i and any self-loops (transitions that return to state i).
x??

---

#### Stationary Equations and Transition Rates
For an ergodic Markov chain, the stationary probabilities $\pi_i$ satisfy the equation:
$$\pi_i = \sum_{j \neq i} \pi_j P_{ji} + \pi_i P_{ii}.$$

Simplifying, we get:
$$\sum_j \pi_i P_{ij} = \sum_j \pi_j P_{ji},$$which means the total rate of transitions out of state i equals the total rate of transitions into state i.
:p Why is it true that $\sum_j \pi_i P_{ij} = \sum_j \pi_j P_{ji}$ for an ergodic Markov chain?
??x
This equality holds because in a long run, every departure from state i must be balanced by some arrival into state i. The total rate of departures (outgoing transitions) from state i is the sum $\sum_j \pi_i P_{ij}$, and the total rate of arrivals (incoming transitions) to state i is the sum $\sum_j \pi_j P_{ji}$. Since the system reaches a steady state, these rates must be equal.
x??

---

#### Stationary Equations Simplified
The stationary equations can also be simplified by ignoring self-loops:
$$\sum_{j \neq i} \pi_i P_{ij} = \sum_{j \neq i} \pi_j P_{ji}.$$:p How are the stationary equations rewritten to ignore self-loops?
??x
The stationary equations can be rewritten to ignore self-loops by removing $\pi_i P_{ii}$ from both sides of the equation:
$$\sum_{j \neq i} \pi_i P_{ij} = \sum_{j \neq i} \pi_j P_{ji}.$$

This simplified form highlights that the rate of transitions out of state i is equal to the rate of transitions into state i, excluding self-loops.
x??

---

#### Balance Equations and Ergodicity Theory

Background context: In ergodicity theory, balance equations are used to equate the rates at which we leave one state and enter another. They help in finding the stationary distribution of a Markov chain by simplifying the process compared to regular stationary equations.

Balance equations can be applied to both single states and sets of states. For a set of states $S $ and its complement$S^c $, balance equations are used to equate the flux (rate of transitions) from$ S $to$ S^c $ with the flux from $ S^c $ to $ S$.

:p Why does it make sense that the total flux from $S $ to$S^c $ should equal that from$ S^c $ to $S$?
??x
The equality of fluxes makes sense because every transition out of state $S $ must eventually lead back into states in$S $, necessitating an equal number of transitions into$ S $from outside$ S$.

---
#### Time-Reversibility Theorem

Background context: The time-reversibility theorem simplifies the process of solving for stationary distributions by providing a condition under which the Markov chain is time-reversible. Aperiodic, irreducible Markov chains can be checked against this condition to determine if they are time-reversible.

The theorem states that given an aperiodic and irreducible Markov chain with transition probabilities $P_{ij}$, there exist constants $ x_i$such that for all $ i$ and $ j$:
$$x_i P_{ij} = x_j P_{ji}$$:p According to the time-reversibility theorem, what does it mean if a Markov chain is time-reversible?
??x
If a Markov chain is time-reversible, then the limiting probabilities can be determined by $x_i = \pi_i $, where $ x_i $ satisfies the condition $ x_i P_{ij} = x_j P_{ji}$. This means that the backward and forward transitions between states are balanced.

---
#### Periodic or Irreducible Chains

Background context: The properties of aperiodicity and irreducibility in Markov chains affect the existence and uniqueness of the stationary distribution. If a chain is both aperiodic and irreducible, then there exists a unique limiting distribution given by solving the stationary equations.

If a chain is not irreducible or is periodic, these conditions need to be checked separately as they can lead to non-unique or no solutions for the stationary distribution.

:p How do the properties of aperiodicity and irreducibility impact the solution of the stationary distribution in Markov chains?
??x
Aperiodicity ensures that the chain does not get stuck in cycles, while irreducibility guarantees that all states are reachable from any other state. Together, they ensure the existence of a unique limiting distribution described by solving the stationary equations.

---
#### Application of Time-Reversibility

Background context: For a given Markov chain, first attempt to apply time-reversibility equations $x_i P_{ij} = x_j P_{ji}$ and normalization condition $\sum x_i = 1$. If successful, this directly gives the limiting probabilities $\pi_i = x_i$.

If not, revert to solving regular stationary or balance equations.

:p How can one determine if a Markov chain is time-reversible using the provided method?
??x
To determine if a Markov chain is time-reversible, attempt to find constants $x_i $ such that$x_i P_{ij} = x_j P_{ji}$ for all states $ i $ and $ j $. If these constants can be found satisfying the normalization condition $\sum x_i = 1$, then the chain is time-reversible. This simplifies finding the stationary distribution.

---
#### Example: Three Types of Equations

Background context: The example illustrates how different sets of equations (regular stationary, balance, and time-reversibility) can be used to determine the limiting probabilities $\pi_i$. Time-reversibility often provides a simpler method but may not always apply.

The Markov chain in Figure 9.5 has three types of equations:
1. Regular stationary: $\pi_i = \pi_{i-1} p + \pi_r i + q $2. Balance:$\pi_i (1-r) = \pi_{i-1} p + \pi_{i+1} q $3. Time-reversibility:$\pi_i p = \pi_{i+1} q$

:p How does the example in Figure 9.5 differentiate between regular stationary, balance, and time-reversibility equations?
??x
The example differentiates by showing that while regular stationary equations are complex to solve, balance equations are a bit simpler but still messy, whereas time-reversibility equations provide a much simpler solution for determining $\pi_i$. This highlights the utility of applying time-reversibility when possible.

#### Periodicity in Markov Chains
In a discrete-time Markov chain (DTMC), states can be periodic, meaning that they return to themselves only at certain intervals. The period of a state $i $, denoted by $\pi(i)$, is defined as the greatest common divisor (GCD) of all positive integers $ n$such that there exists a path from state $ i$back to itself with length exactly $ n$.
:p What does the concept of periodicity in Markov chains refer to?
??x
Periodicity in a Markov chain refers to the fact that a state returns to itself only after certain intervals, which are determined by its period. The period is the greatest common divisor (GCD) of all positive integers $n $ such that there exists a path from the state back to itself with length exactly$n$. This means that states can return to themselves in multiple steps and not just at every step.
x??

---

#### Periodicity Lemma
The lemma states that in an irreducible DTMC, all states have the same period. The proof involves showing that if two communicating states have different periods, a contradiction arises due to the properties of paths between these states.
:p What does Lemma 9.35 state about periodicity in irreducible Markov chains?
??x
Lemma 9.35 states that in an irreducible DTMC, all states have the same period. The proof involves assuming two communicating states with different periods and deriving a contradiction based on path lengths and greatest common divisors (GCDs).
The key steps are:
1. If states $i $ and$j $ communicate, where state$ i $ has period $ p $ and state $ j $ has period $ q $, there exist paths from $ i$to $ j$of length $ d_1$and from $ j$to $ i$of length $ d_2$.
2. The combined path forms a loop back to state $i $ with length$d_1 + d_2 $. Hence, the period$ p $divides$(d_1 + d_2)$.
3. Considering any loop from $j $ back to$j $ of length$ x $, and following it with paths from $ i$to $ j$and then from $ j$to $ i$, a journey of length $ d_1 + d_2 + x$ is formed.
4. This shows that $p $ divides$(d_1 + d_2 + x)$.
5. Subtracting the previous two steps, we get that $p $ divides$x$.
6. Since this holds for any loop from $j $ to$j $,$ p$ must divide the GCD of all such loops.
7. By symmetry and similar arguments, it follows that $q $ also divides the same GCD, leading to$p = q$.

The lemma proves that in an irreducible DTMC, all states share the same period.
x??

---

#### Positive Recurrence for Periodic Chains
In Theorem 9.36, it is shown that for a periodic and positive-recurrent chain (with finite period), there exists a unique stationary distribution $\vec{\pi}$ which represents the long-run time-average proportion of time spent in each state.
:p What does Theorem 9.36 assert about periodic chains?
??x
Theorem 9.36 states that for an irreducible, positive-recurrent DTMC with finite period $d < \infty $, there exists a unique stationary distribution $\vec{\pi}$ such that:
- It satisfies the stationary equations:$\vec{\pi} \cdot P = \vec{\pi}$- The sum of all probabilities equals 1:$\sum_i \pi_i = 1 $ This stationary distribution$\vec{\pi}$ represents the long-run time-average proportion of time spent in each state. This theorem is crucial because it ensures that for periodic chains, there exists a unique solution to the stationary equations and this solution accurately reflects the long-term behavior of the chain.
x??

---

#### Labeling States in Periodic Chains
The states are labeled according to their residue classes. This labeling helps in understanding how states interact over time and simplifies the analysis of the periodicity.
:p How do we label states in a periodic Markov chain?
??x
In a periodic Markov chain, states can be grouped into residue classes based on their periods. Each state $i $ is labeled according to its period$p$. This labeling helps in understanding the long-term behavior and simplifying the analysis.

For example, if a state has a period of 2, it means that this state will return to itself every two steps. By grouping states into residue classes, we can analyze how these states interact over multiple cycles.

This labeling is particularly useful because:
1. It allows us to focus on one representative state from each residue class.
2. The behavior of the chain can be studied by examining transitions within and between these residue classes.

This approach simplifies the analysis and helps in proving properties such as the existence and uniqueness of the stationary distribution for periodic chains.
x??

---

#### Proof Outline for Periodic Chains
The proof involves several steps:
1. Define a convenient way to label states based on their periods.
2. Prove that the time-average distribution is a stationary distribution.
3. Show that any stationary distribution equals the time-average distribution.

This outline helps in systematically proving Theorem 9.36 for periodic chains.
:p What does the proof outline for Theorem 9.36 cover?
??x
The proof outline for Theorem 9.36 covers the following steps:
1. **Labeling States**: Group states into residue classes based on their periods to simplify analysis.
2. **Time-Average Distribution as Stationary**: Prove that the distribution of time averages is a stationary distribution.
3. **Equality of Any Stationary Distribution and Time-Average Distribution**: Show that any stationary distribution must equal the time-average distribution.

This systematic approach ensures a rigorous proof of the theorem, demonstrating the existence, uniqueness, and correctness of the stationary distribution for periodic chains.
x??

---

#### Knight's Residue Classes on a Chessboard
Background context explaining that for a knight, each move alternates between black and white squares, creating a periodic pattern. The state space can be divided into residue classes based on this periodicity.

:p How many residue classes does the knight have on a chessboard?
??x
The knight has 2 residue classes: one for the black squares (0) and another for the white squares (1).

Explanation:
Since the knight alternates between black and white squares, it can return to its original square in exactly 2 moves. This means that all states are classified into two categories based on their color.

```java
// Pseudocode to illustrate state transitions
public class KnightOnChessboard {
    public void moveKnight(int currentState) {
        if (currentState == 0) { // Current state is a black square
            // Possible next states: white squares
        } else if (currentState == 1) { // Current state is a white square
            // Possible next states: black squares
        }
    }
}
```
x??

---

#### Transition Matrix $P$ Structure
Background context explaining that the transition matrix can be partitioned based on residue classes, and each row corresponds to transitions from one residue class to another.

:p What is the structure of the transition matrix $P$ after relabeling states into their respective residue classes?
??x
The transition matrix $P$ has a block diagonal form where rows and columns are grouped by residue classes. Specifically:

```
P = [
 [A0,1 0 0 ... 0]
 [0 A1,2 0 ... 0]
 [0 0 A2,3 ... 0]
 ...
 [0 0 0 ... Ad-1,0]
]
```

Here,$A_{i,i+1}$ is a stochastic matrix representing the probability of transitioning from class $i$ to class $i+1$.

```java
// Pseudocode for transition matrix structure
public class TransitionMatrix {
    public void constructTransitionMatrix() {
        // Initialize the d x d transition matrix
        double[][] P = new double[d][d];
        
        // Fill in the matrices A_{i, i+1}
        for (int i = 0; i < d - 1; i++) {
            P[i][i + 1] = 1.0; // Assuming direct transitions
        }
        P[d-1][0] = 1.0; // Last row transition back to first class
    }
}
```
x??

---

#### Properties of $A_{i, i+1}$ Matrices
Background context explaining that each matrix $A_{i, i+1}$ is stochastic because the entire matrix $P$ is stochastic.

:p Are all elements in matrices $A_{i,i+1}$ positive?
??x
Not necessarily. There may not be a direct connection between every element of vector $i $ and vector$(i+1)$, meaning some entries can be zero.

Explanation:
The positivity of the elements depends on whether there is a direct transition path from state $i $ to state$(i+1)$ in one step. Since transitions may involve multiple steps, not all paths are guaranteed to exist directly.

```java
// Pseudocode for checking positive elements in A_i,i+1
public class DirectTransitions {
    public boolean hasDirectTransition(int i, int j) {
        // Check if there is a direct transition from state i to state j
        return (i != j); // Simplified example; actual logic depends on the problem
    }
}
```
x??

---

#### Form of $P^d $ Background context explaining that raising the matrix$P $ to the power$d $ results in a specific form, where each block$D_{i,i}$ is composed of products of transition matrices.

:p What does $P^d $ look like and how can it be expressed in terms of the$A_{i,i+1}$ matrices?
??x
The matrix $P^d$ has a diagonal form where:

```
P^d = [
 [D0,0 0 0 ... 0]
 [0 D1,1 0 ... 0]
 [0 0 D2,2 ... 0]
 ...
 [0 0 0 ... D_{d-1,d-1}]
]
```

The diagonal elements $D_{i,i}$ are given by:
$$D_{i,i} = A_{i, i+1} \cdot A_{i+1, i+2} \cdot \ldots \cdot A_{i+d-1, i}$$

Explanation:
This form arises because each transition matrix $A_{i,i+1}$ represents the probability of moving from one residue class to another in exactly 1 step. Raising $P$ to the power $d$ effectively multiplies these probabilities over $d$ steps.

```java
// Pseudocode for calculating D_i,i
public class DiagonalBlockCalculation {
    public double calculateDiagonalBlock(int i) {
        double result = 1.0; // Initialize product
        int j = i;
        while (j < d) { 
            result *= transitionMatrices[j][i];
            j++;
        }
        return result;
    }
}
```
x??

---

#### Properties of $D_{i,i}$ Background context explaining that $D_{i,i}$ is stochastic and represents the probability of moving between states within a residue class in exactly $d$ steps.

:p Is $D_{i,i}$ stochastic? What does it represent? And are the properties irreducibility, aperiodicity, and positive recurrence true for $D_{i,i}$?
??x
Yes, $D_{i,i}$ is stochastic because it is the product of stochastic matrices. It represents the probability of moving from any state in residue class $i$ to any other state within that same class after exactly $d$ steps.

Properties:
- **Irreducibility**: Since $P $ is irreducible, and all paths between states in class$i $ involve lengths that are multiples of$d $, the submatrix $ D_{i,i}$ is also irreducible.
- **Aperiodicity**: The period of each residue class is 1 because there exists a path from any state to itself in exactly $d$ steps, which is a multiple of the class's period.
- **Positive Recurrence**: Because every state can be reached from any other state within its own class and all elements are positive (or at least non-zero),$D_{i,i}$ is positive recurrent.

Explanation:
The irreducibility follows from the fact that transitions between classes respect the periodicity, ensuring that we can always return to our starting state after a multiple of $d $ steps. The aperiodicity comes from the ability to move directly within the same class in exactly$d$ steps. Positive recurrence is guaranteed by the existence of non-zero (stochastic) entries in all blocks.

```java
// Pseudocode for checking properties of D_i,i
public class PropertiesOfDiagonalBlocks {
    public boolean checkIrreducibility() {
        // Check if there exists a path between any two states within the same class after d steps
        return true; // Assumption based on problem context
    }
    
    public boolean checkAperiodicity() {
        // Check if the period is 1 (i.e., we can move directly in exactly d steps)
        return true;
    }
    
    public boolean checkPositiveRecurrence() {
        // Check if all elements are positive (or non-zero)
        return true; // Assumption based on problem context
    }
}
```
x??

---

#### Definition of Time-Average Distribution
Background context explaining the time-average distribution and its relation to periodic chains. The formula for the distribution is given as:
$$\vec{p} = (p_{01}, p_{02}, p_{03}, ..., p_{(d-1)1}, p_{(d-1)2}, p_{(d-1)3}, ...)$$where$$\sum_{i=0}^{d-1} \sum_{j} p_{ij} = 1$$and $ p_{ij}$represents the long-run proportion of time spent in state $ ij$.

:p What do we know about $\sum_j p_{ij}$?
??x
Since vector $i $ is only visited once every d steps,$\sum_j p_{ij} = \frac{1}{d}$.
The explanation is that if the chain visits each state in vector $i $ only once every$d $ steps on average, then the sum of the long-run proportions spent in any single state within vector$i $ over these$d $ steps must be $\frac{1}{d}$.

---
#### Definition of q/vectori
Background context explaining the relationship between $\vec{p}$ and $\vec{q}$. The definition is given as:
$$\vec{q}_i = d \cdot \vec{p}_i$$:p What does $\vec{q}_i$ represent?
??x $\vec{q}_i $ represents the time-average proportion of time spent in each state of vector$i $ when observing the chain every$d $ steps. It essentially captures how often we visit states within vector $ i$ over those observations.

---
#### Stationary Distribution for Di,i
Background context explaining the relation between stationary distributions and the matrices $D_i,i$. The equations provided are:
$$\vec{q}_i \cdot D_{i,i} = \vec{q}_i$$and$$\sum_j q_{ij} = 1$$:p What does the equation $\vec{q}_i \cdot D_{i,i} = \vec{q}_i$ imply?
??x
This equation implies that $\vec{q}_i $ is a stationary distribution for the matrix$D_{i,i}$, which represents the probability transitions within states of vector $ i$. Since $ D_{i,i}$ is ergodic, it has a unique stationary distribution equal to both its limiting and time-average distributions.

---
#### Transition Matrix P
Background context explaining how to relate $\vec{q}_i $ back to the original transition matrix$P $. The logic involves transforming from $ D_i,i+1 $matrices back to$ P$using the relation:
$$\vec{r} = \vec{q}_i \cdot A_{i,i+1}$$where$$\sum_j r_j = 1$$:p What can be said about the sum of elements in vector $\vec{r}$?
??x
Since the elements of $\vec{q}_i $ sum to 1 and$A_{i,i+1}$ is a stochastic matrix (preserving sums), the sum of the elements in $\vec{r}$ must also be 1.

---
#### Uniqueness of Stationary Distribution
Background context explaining why $(\vec{p}_0, \vec{p}_1, ..., \vec{p}_{d-1})$ is a stationary distribution for $P$. The reasoning involves showing that:
$$(\vec{q}_0, \vec{q}_1, ..., \vec{q}_{d-1}) \cdot P = (\vec{q}_0, \vec{q}_1, ..., \vec{q}_{d-1})$$:p What does this imply about the stationary distribution?
??x
This implies that $(\vec{p}_0, \vec{p}_1, ..., \vec{p}_{d-1})$ is a stationary distribution for $P$, meaning it satisfies the stationary equations of the original chain with transition matrix $ P$.

---
#### Summary Theorem for Periodic Chains
Background context explaining the theorem that given an irreducible DTMC with period $d < \infty $, if a stationary distribution $\vec{\pi}$ exists, then the chain must be positive recurrent.

:p What does this theorem state?
??x
The theorem states that if an irreducible discrete-time Markov chain (DTMC) has a finite period and a stationary distribution exists, then the chain is positive recurrent. This means that the chain will return to any given state infinitely often with probability 1 over time.

---
#### Proof for Positive Recurrence
Background context explaining the proof's structure and key steps in showing positive recurrence using residue classes and matrices $D_{i,i}$.

:p What does the proof argue about the irreducibility and aperiodicity of $D_{i,i}$?
??x
The proof argues that while the original matrix $P $ might not be positive recurrent, by partitioning states into residue classes and analyzing the matrices$D_{i,i}$, it can show that each $ D_{i,i}$is irreducible and aperiodic. This step-by-step analysis helps in concluding that if $\vec{\pi} \cdot P = \vec{\pi}$, then $\vec{\pi}$ must be the unique stationary distribution for $P$.

