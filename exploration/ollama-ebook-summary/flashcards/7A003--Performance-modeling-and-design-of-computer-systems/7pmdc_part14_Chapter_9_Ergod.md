# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 14)

**Starting Chapter:** Chapter 9 Ergodicity Theory. 9.2 Finite-State DTMCs

---

#### Time Averages vs. Ensemble Averages
Background context: The provided text discusses the difference between time averages and ensemble averages within the context of DTMCs (Discrete-Time Markov Chains). Time averages, denoted as \( p_j \), are defined as the long-run fraction of time spent in state \( j \) over one infinite sample path. Ensemble averages, represented by \( \pi_j \), are probabilities computed from all possible sample paths of length \( n \) as \( n \) approaches infinity.

:p What is the difference between a time average and an ensemble average in the context of DTMCs?
??x
In the context of DTMCs:
- A **time average** (\( p_j \)) refers to the long-run fraction of time spent in state \( j \), averaged over one infinite sample path. It is calculated as:
  \[
  p_j = \lim_{n \to \infty} \frac{1}{n} \sum_{k=0}^{n-1} I(X_k = j)
  \]
  where \( I(X_k = j) \) is an indicator function that equals 1 if the system is in state \( j \) at time \( k \).

- An **ensemble average** (\( \pi_j \)) refers to the limiting probability of being in state \( j \), averaged over all possible sample paths. It is computed as:
  \[
  \pi_j = \lim_{n \to \infty} (P^n)_{ij}
  \]
  where \( P \) is the transition matrix and \( (P^n)_{ij} \) represents the probability of transitioning from state \( i \) to state \( j \) in exactly \( n \) steps.

The key difference lies in how these averages are computed:
- Time averages consider a single path over an infinitely long time.
- Ensemble averages consider all possible paths, which gives a broader view of the system's behavior.

C/Java code is not directly applicable here but can be used to simulate and estimate these values. For example:

```java
public class MarkovChainSimulation {
    private double[][] transitionMatrix;
    private int steps;

    public MarkovChainSimulation(double[][] transitionMatrix, int steps) {
        this.transitionMatrix = transitionMatrix;
        this.steps = steps;
    }

    // Simulate the long-run fraction of time spent in each state (time average)
    public void simulateTimeAverage() {
        // Implement logic to run simulations and estimate p_j
    }

    // Estimate ensemble average by calculating P^n * initial_state_vector
    public double[] calculateEnsembleAverage(double initialStateVector) {
        return matrixPower(transitionMatrix, steps).multiply(initialStateVector);
    }

    private double[][] matrixPower(double[][] matrix, int power) {
        // Implement matrix multiplication and exponentiation logic here
    }
}
```

x??

---

#### Example of Periodic Chain
Background context: The text provides an example of a two-state transition matrix where the limiting distribution does not exist due to periodicity. This means that in such chains, states are visited only at specific intervals.

:p Provide an example of a valid two-state transition matrix for which \( \pi_j \) (limiting distribution) does not exist.
??x
An example of a valid two-state transition matrix for which the limiting distribution \( \pi_j \) does not exist is:
\[
P = \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\]
In this chain, state 0 transitions to state 1 and vice versa every time step. Therefore, a given state is only visited every other time step, making the limiting distribution non-existent.

To illustrate:
- If you start in state 0, you will alternate between states 0 and 1: 0 -> 1 -> 0 -> 1 ...
- The long-run fraction of time spent in each state does not converge to a single value; hence \( \pi_j \) is undefined for this matrix.

x??

---

#### Mean Time Between Visits
Background context: This concept delves into understanding the average time spent between visits to a particular state. It relates closely to the limiting distribution and provides insights into the behavior of the Markov chain.

:p How does the mean time between visits to state \( j \) relate to the limiting probability \( \pi_j \)?
??x
The mean time between visits to state \( j \), often denoted as \( m_j \), is related to the limiting probability \( \pi_j \). Specifically, it can be derived from the properties of the stationary distribution and reflects how frequently the system revisits a particular state in the long run.

To find the mean time between visits (\( m_j \)), consider:
\[
m_j = 1 / \pi_j
\]
This relationship implies that if \( \pi_j \) is high, then the mean time between visits to state \( j \) is low, and vice versa. This means states with higher limiting probabilities are visited more frequently.

For example, in a simple chain:
- If \( \pi_0 = 0.4 \), the mean time between visits to state 0 would be \( m_0 = 1 / 0.4 = 2.5 \) steps.
- This indicates that on average, the system revisits state 0 every 2.5 steps.

x??

---

#### Existence of Limiting Distribution
Background context: The text introduces the question of when a limiting distribution exists for finite-state Markov chains and provides an example to illustrate this concept. Periodic chains are highlighted as cases where such distributions do not exist.

:p Under what conditions does the limiting distribution exist for a finite-state DTMC?
??x
The limiting distribution exists for a finite-state DTMC if the chain is both irreducible (all states can be visited from any state) and aperiodic. A periodic chain, where states are only visited at specific intervals, will not have a well-defined limiting distribution.

For example, consider:
\[
P = \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\]
This matrix is periodic because it cycles between the two states without converging to any fixed probability. Therefore, no \( \pi_j \) exists for this matrix.

On the other hand, a non-periodic chain that can transition freely and return to all states will have a well-defined limiting distribution.

x??

---

#### Time Average and Stationary Distribution
The time average spent in each state of a Discrete-Time Markov Chain (DTMC) is represented by \( p_j \), which can be determined based on the transition probabilities. For a valid DTMC, these averages must exist. In simple cases, like the one mentioned:
:p What does \( p_0 = 1/2 \) and \( p_1 = 1/2 \) indicate in this context?
??x
This indicates that the chain spends half of its time in state 0 and the other half in state 1. This is a simple example where the stationary distribution is straightforward to determine.
x??

---

#### Existence of Stationary Distribution
The existence of a stationary distribution can be verified by solving a set of equations derived from setting up \( \pi \cdot P = \pi \), where \( \pi \) is the stationary vector and \( P \) is the transition matrix. In the given example:
:p Does this chain have a stationary distribution, and if so, what is it?
??x
Yes, the stationary distribution exists and can be found by solving:
\[ \pi_0 = \pi_1 \]
\[ \pi_1 = \pi_0 \]
\[ \pi_0 + \pi_1 = 1 \]

Solving these equations gives \( \pi = (1/2, 1/2) \), indicating that the chain spends equal time in both states.
x??

---

#### Periodicity of a Transition Matrix
Periodicity is determined by finding the greatest common divisor (GCD) of the set of integers \( n \) for which \( P^n_{jj} > 0 \). For periodicity, this GCD must be greater than 1. In the given example:
:p Is the matrix provided an example of a periodic transition matrix? Explain why or why not.
??x
Yes, the provided matrix is periodic because it has a cycle (for instance, from state 3 to state 0 and back). The states do not allow direct transitions that would make all periods equal to 1, indicating periodicity.
x??

---

#### Aperiodicity and Irreducibility
Aperiodicity means that for any state \( j \), the GCD of the set of integers \( n \) such that \( P^n_{jj} > 0 \) is 1. Irreducibility requires that all states communicate with each other, meaning there is a path from one state to another.
:p Why are aperiodicity and irreducibility necessary for limiting probabilities?
??x
Aperiodicity ensures that the chain does not get stuck in cycles of fixed length, while irreducibility ensures that every state can be reached from any other state. Together, they guarantee that the limiting distribution is unique and independent of the starting state.
x??

---

#### Identity Matrix as Not Irreducible
The identity matrix is an example where states are not connected, making it non-irreducible.
:p Can you provide a simple transition matrix that is not irreducible?
??x
Yes, the identity matrix \( I \) is a simple example. It consists of diagonal entries being 1 and all other entries being 0, indicating no transitions between different states:
```
I = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
```
x??

---

#### Limiting Probabilities Existence
For a finite-state DTMC to have limiting probabilities that exist, are positive, sum to 1, and are independent of the starting state, it must be both aperiodic and irreducible.
:p Do aperiodicity and irreducibility alone guarantee the existence of a limiting distribution?
??x
Yes, according to Theorem 9.4, for a finite-state DTMC with an aperiodic and irreducible transition matrix \( P \), as \( n \to \infty \), \( P^n \) converges to a limiting matrix where all rows are the same vector \( \vec{\pi} \). This vector has positive components that sum to 1, ensuring the existence of the limiting distribution.
x??

---

#### Background on Matrix Multiplication and Convergence
Context: We are dealing with a matrix \(P\) that is used to transform vectors, where each row of \(P\) sums to 1. The goal is to show how repeated multiplication by \(P\) leads to convergence of the vector components to be equal.

:p Explain the process of multiplying a vector by the matrix \(P\).
??x
When we multiply a vector \(\mathbf{v}\) by the matrix \(P\), each element in the resulting vector is a weighted average of all elements of the original vector. This averaging brings the components closer together, reducing the difference between the maximum and minimum values.

:p What does the expression \(M_n - m_n \leq (1-2s)(M_{n-1} - m_{n-1})\) represent?
??x
This inequality represents the relationship between the differences in the maximum and minimum components of successive vectors after multiplying by matrix \(P\). It shows that the difference decreases over time, where \(s\) is the smallest element in \(P\).

:p Why does the argument fail when \(s = 0\)?
??x
The argument fails because if \(s = 0\), then \((1 - 2s) = 1\). In this case, there is no reduction in the difference between the maximum and minimum components of the vectors. This means that without a reduction factor, the components might not necessarily converge.

:p How can we fix the issue when \(s = 0\)?
??x
We need to ensure that all elements of the matrix \(P^n\) are positive for sufficiently large \(n\). Given that \(P\) is aperiodic and irreducible, there exists some \(n_0\) such that for all \(n \geq n_0\), every element in \(P^n\) is positive. This ensures that the transformation continues to average out components effectively.

:p What does it mean for a matrix \(P\) to be aperiodic and irreducible?
??x
Aperiodicity means there exists an integer \(k\) such that all elements of \(P^k\) are positive, indicating no fixed cycles. Irreducibility means that from any state \(i\), we can reach any other state \(j\) in some number of steps, ensuring connectivity across the matrix.

:p What does it mean for a vector to have its components converge?
??x
The convergence of vector components means that as we repeatedly apply the transformation by matrix \(P\), the difference between the maximum and minimum values in the vector decreases, eventually making all components equal. This is due to the averaging effect of multiplying by \(P\) multiple times.

---
---

#### Existence of Limiting Probabilities for aperiodic, irreducible, finite-state Markov chains
In this section, we establish that for any aperiodic and irreducible finite-state Markov chain with a transition matrix \( P \), the limiting probabilities exist. This involves showing that as \( n \) approaches infinity, \( P^n \) has all positive elements.

:p What does it mean to have all positive elements in a Markov chain's transition matrix?
??x
The transition matrix \( P^n \) will eventually contain only positive values for sufficiently large \( n \), ensuring that the probability of transitioning between any two states is non-zero. This property is crucial because it means every state can be reached from any other state after a certain number of steps.
x??

---
#### Defining \( n_0(i, j) \)
To ensure that \( P^n \) has all positive elements for large enough \( n \), we define \( n_0(i, j) \) such that there is a path of length at least \( n_0(i, j) \) from state \( i \) to state \( j \). Given irreducibility and the fact that there is always a path of length \( x \) from any state \( i \) to itself (\( i = i \)), we can use these paths to find such an \( n_0(i, j) \).

:p How do you define \( n_0(i, j) \)?
??x
We define \( n_0(i, j) = n_0(i, i) + x \), where \( n_0(i, i) \) is the minimum number of steps needed to return to state \( i \) starting from \( i \), and \( x \) is the length of a path from state \( i \) to state \( j \).
x??

---
#### Defining \( P' \)
After defining \( n_0(i, j) \) for all pairs \( (i, j) \), we find the maximum value among these definitions. We then define \( P' = P^{n_0} \). This matrix will have the property that when raised to any power \( k \geq 1 \), it remains positive.

:p What is the role of \( n_0 \) in defining \( P' \)?
??x
\( n_0 \) represents the maximum number of steps needed for the transition matrix \( P \) and its powers to ensure all elements are positive. By setting \( P' = P^{n_0} \), we guarantee that \( (P')^k \) will have all positive entries, maintaining the ergodic property.
x??

---
#### Mean Time between Visits to a State
The mean time between visits to state \( j \), denoted as \( m_{jj} \), is defined as the expected number of steps needed to first return to state \( j \). This quantity is related to the limiting probability \( \pi_j \) of being in state \( j \).

:p What does \( m_{jj} \) represent?
??x
\( m_{jj} \) represents the expected number of time steps required for a Markov chain to first return to state \( j \), starting from the same state. It is related to the limiting probability \( \pi_j \) which gives the long-term proportion of time spent in state \( j \).
x??

---
#### Theorem 9.6: Relationship between \( m_{jj} \) and \( \pi_j \)
For an irreducible, aperiodic finite-state Markov chain with transition matrix \( P \), the mean time between visits to state \( j \), \( m_{jj} \), is given by:

\[
m_{jj} = \frac{1}{\pi_j}
\]

where \( \pi_j \) is the limiting probability of being in state \( j \).

:p What is the relationship between \( m_{jj} \) and \( \pi_j \)?
??x
The mean time between visits to state \( j \), denoted as \( m_{jj} \), is inversely proportional to the limiting probability \( \pi_j \). As \( n \) approaches infinity, the proportion of time spent in state \( j \) (given by \( \pi_j \)) and the expected number of steps between visits to state \( j \) are reciprocally related.
x??

---
#### Matrix Notation for Mean Time Between Visits
We define a matrix \( M \) where each entry \( m_{ij} \) represents the expected number of time steps needed to first get to state \( j \), given we are currently at state \( i \). We can express this using matrices as follows:

\[
M = E + PN
\]

where:
- \( E \) is a matrix with all entries 1.
- \( P \) is the transition matrix.
- \( N \) is a matrix where \( N_{ij} = m_{ij} \) for \( i \neq j \), and zero otherwise.

This representation helps in deriving expressions for \( m_{jj} \).

:p What does the matrix equation \( M = E + PN \) represent?
??x
The matrix equation \( M = E + PN \) represents a way to express the expected number of steps between visits to different states using matrix notation. Here, \( E \) is a matrix filled with ones, representing constant 1 values across all entries, and \( N \) captures the specific mean time to visit other states.

By solving this equation, we can derive expressions for \( m_{jj} \):
```java
// Pseudocode for solving M = E + PN
Matrix M = Matrix::create(E + P * N);
```
x??

#### Limiting Distribution and Time Averages
Background context: For a finite-state Markov chain, when the limiting distribution exists, it is equal to the unique stationary distribution. Additionally, the probability of visiting state \( j \) over time (denoted as \( p_j \)) can be related to the mean recurrence time \( m_{jj} \), where \( p_j = \frac{1}{m_{jj}} \).

Formulas: 
- Limiting distribution: \( \pi_j \)
- Time spent in state \( j \): \( p_j \)
- Mean recurrence time between visits to state \( j \): \( m_{jj} \)

The relationship is given by:
\[ p_j = \lim_{t \to \infty} \frac{\text{number of times in state } j \text{ during } t \text{ steps}}{t} \]

:p What is the relationship between the limiting distribution \( \pi_j \) and the time spent in state \( j \), assuming the limiting distribution exists?
??x
The limiting distribution \( \pi_j \) is equal to the fraction of time that the Markov chain spends in state \( j \) along a given sample path, i.e., \( p_j = \pi_j \). This relationship holds with probability 1 for almost every sample path.

Explanation: 
- The theorem states that if the limiting distribution exists, then \( p_j = \frac{1}{m_{jj}} \), and since \( m_{jj} = \sum_{t=0}^{\infty} t P_{jj}(t) \), we have \( p_j = \pi_j \).
- Intuitively, if the average time between visits to state \( j \) is \( m_{jj} \), then during a long period of time \( t \), we visit state \( j \) approximately \( \frac{t}{m_{jj}} \) times. Hence, the proportion of time spent in state \( j \) is \( p_j = \frac{1}{m_{jj}} \).

```java
public class MarkovChain {
    // Method to calculate pi_j and p_j for a given state j
    public double calculatePiAndP(int stateJ, int[] meanRecurrenceTimes) {
        return 1.0 / meanRecurrenceTimes[stateJ]; // Assuming m_{jj} is in the array
    }
}
```
x??

---

#### Infinite-State Markov Chains
Background context: In infinite-state Markov chains, reasoning about the chain's behavior becomes more complex than with finite-state chains. The concepts of aperiodicity and irreducibility still apply but have different implications.

C/Java code or pseudocode is not directly relevant here as we are focusing on definitions and properties.

:p Which of the three DTMCs shown in Figure 9.1 are aperiodic and irreducible?
??x
All of them are aperiodic and irreducible.
Explanation: 
- Aperiodicity means that the greatest common divisor (gcd) of all return times is 1.
- Irreducibility means that there is a non-zero probability of transitioning from any state to any other state.

```java
public class MarkovChain {
    // Method to check if a DTMC is aperiodic and irreducible
    public boolean isAperiodicAndIrreducible() {
        return true; // Since all are periodic, this example assumes they meet the criteria.
    }
}
```
x??

---

#### Positive Recurrence, Transience, Null Recurrence
Background context: In infinite-state Markov chains, the concepts of positive recurrence, transience, and null recurrence are used to describe different behaviors of the chain. These terms relate to the existence and properties of limiting distributions.

C/Java code or pseudocode is not directly relevant here as we are focusing on definitions and examples.

:p Which of the chains in Figure 9.1 have a limiting distribution?
??x
The first chain has a well-defined limiting probability for each state, which sums to 1. The second and third chains do not have a limiting distribution; their limiting probabilities sum to 0.
Explanation: 
- Positive recurrence means that the expected return time is finite.
- Transience means that the probability of returning to any state is less than 1.
- Null recurrence means that the expected return time is infinite, but the chain still visits each state infinitely often.

```java
public class MarkovChain {
    // Method to check if a DTMC has a limiting distribution
    public boolean hasLimitingDistribution() {
        return true; // Only for the first chain.
    }
}
```
x??

---

#### Problem with Transient and Null Recurrent Chains
Background context: The second chain (transient) can be viewed as an ocean with states drifting away, while the third chain (null recurrent) suggests visits but not guaranteed returns.

C/Java code or pseudocode is not directly relevant here as we are focusing on explanations of concepts.

:p Intuitively, what is the problem with the second and third chains in Figure 9.1?
??x
The main issues with these chains are related to their recurrence properties:
- For the second chain (transient), there is a drift away from state 1 (or any chosen "shore"), making it uncertain whether we will return infinitely often.
- For the third chain (null recurrent), while visits seem likely, the time between returns can be arbitrarily large, so it may not happen within any finite period.

Explanation: 
- Transient chains have a non-zero probability of never returning to some states.
- Null recurrent chains visit each state infinitely often but do not guarantee a finite return time.

```java
public class MarkovChain {
    // Method to explain the problem with transient and null recurrent chains
    public void explainProblems() {
        System.out.println("Transient chain: Drift away from shore makes it uncertain if we return infinitely often.");
        System.out.println("Null recurrent chain: Visits seem likely but may take an arbitrarily large time, not guaranteed within any finite period.");
    }
}
```
x??

#### Recurrent vs Transient States
Recurrent states are those from which the Markov chain returns to the state infinitely often with probability 1. Transient states, on the other hand, have a finite number of visits before being left forever.

:p What is the definition of recurrent and transient states?
??x
In a Markov chain, a state \(j\) is recurrent if \(f_j = 1\), meaning it will be visited infinitely often with probability 1. A state \(j\) is transient if \(f_j < 1\), meaning the number of visits to this state is finite.

```java
// Pseudocode for checking if a state is recurrent or transient
public class StateChecker {
    private double[] transitionMatrix; // Transition probabilities

    public boolean isRecurrent(int state) {
        return computeRecurrenceProbability(state) == 1;
    }

    public boolean isTransient(int state) {
        return computeRecurrenceProbability(state) < 1;
    }

    private double computeRecurrenceProbability(int state) {
        // Implement logic to calculate recurrence probability
        return transitionMatrix[state][state];
    }
}
```
x??

---

#### Geometric Distribution of Transient State Visits
The number of visits to a transient state \(j\) follows a geometric distribution with mean \(\frac{1}{1-f_j}\), where \(f_j\) is the probability that the chain starting in state \(j\) will return to it.

:p What is the distribution of the number of visits to a transient state?
??x
The number of visits to a transient state \(j\) follows a geometric distribution with mean \(\frac{1}{1-f_j}\). Here, \(f_j\) is the probability that the Markov chain starting in state \(j\) returns to state \(j\).

```java
// Pseudocode for calculating expected number of visits to a transient state
public class VisitCounter {
    private double fJ; // Probability of returning to state j

    public double expectedVisits() {
        return 1.0 / (1 - fJ);
    }
}
```
x??

---

#### Infinite Visits in Recurrent States
For recurrent states, the Markov chain returns infinitely often with probability 1.

:p What is the behavior of a Markov chain starting from a recurrent state?
??x
A Markov chain starting from a recurrent state \(i\) will visit state \(i\) an infinite number of times with probability 1. This means that if you start in state \(i\), it is guaranteed to return to state \(i\) infinitely many times.

```java
// Pseudocode for simulating visits in a recurrent state
public class RecurrentStateSimulator {
    private double[][] transitionMatrix; // Transition probabilities

    public void simulateVisits(int initialState) {
        int current_state = initialState;
        while (true) { // Simulate infinite number of steps
            current_state = getNextState(current_state);
            System.out.println("Visited state: " + current_state);
        }
    }

    private int getNextState(int currentState) {
        return ThreadLocalRandom.current().nextInt(transitionMatrix.length); // Simplified logic for now
    }
}
```
x??

---

#### Expected Number of Visits in Finite Steps
The expected number of visits to a state \(i\) over \(s\) steps, starting from state \(i\), is given by \(\frac{s}{\sum_{n=0}^{s} P^n_{ii}}\).

:p What formula calculates the expected number of visits to a state in finite steps?
??x
The expected number of visits to a state \(i\) over \(s\) steps, starting from state \(i\), is given by:
\[ E[\# \text{visits to } i \text{ in } s \text{ steps} | X_0 = i] = \frac{s}{\sum_{n=0}^{s} P^n_{ii}} \]
where \(P^n_{ii}\) is the probability of being at state \(i\) after \(n\) steps starting from state \(i\).

```java
// Pseudocode for calculating expected number of visits in finite steps
public class ExpectedVisitsCalculator {
    private double[][] transitionMatrix; // Transition probabilities

    public double expectedVisits(int initialState, int steps) {
        double sum = 0;
        for (int n = 0; n <= steps; n++) {
            sum += computeProbability(initialState, initialState, n);
        }
        return steps / sum;
    }

    private double computeProbability(int fromState, int toState, int steps) {
        // Implement logic to calculate the probability of being in 'toState' after 'steps' starting from 'fromState'
        return transitionMatrix[fromState][toState];
    }
}
```
x??

---

#### Infinite vs Finite Visits in Transient and Recurrent States
Recurrent states have infinite visits with probability 1, whereas transient states have finite visits.

:p What are the differences between recurrent and transient states in terms of visits?
??x
In a Markov chain:
- **Recurrence**: A state is recurrent if it is visited infinitely often with probability 1. The sum \(\sum_{n=0}^{\infty} P^n_{ii}\) diverges, indicating infinite visits.
- **Transience**: A state is transient if the number of visits to this state is finite. The sum \(\sum_{n=0}^{\infty} P^n_{jj}\) converges, indicating a finite number of visits.

```java
// Pseudocode for checking convergence of visit probabilities
public class VisitProbabilityChecker {
    private double[][] transitionMatrix; // Transition probabilities

    public boolean isRecurrent(int state) {
        return checkConvergence(state);
    }

    public boolean isTransient(int state) {
        return !checkConvergence(state);
    }

    private boolean checkConvergence(int state) {
        double sum = 0;
        for (int n = 0; ; n++) { // Infinite loop
            sum += computeProbability(state, state, n);
            if (sum > Double.MAX_VALUE / 2.0) break; // Check for divergence
        }
        return sum == Double.POSITIVE_INFINITY;
    }

    private double computeProbability(int fromState, int toState, int steps) {
        // Implement logic to calculate the probability of being in 'toState' after 'steps' starting from 'fromState'
        return transitionMatrix[fromState][toState];
    }
}
```
x??

---

#### Communicating States and Recurrence
If a state \(i\) is recurrent and communicates with another state \(j\), then \(j\) must also be recurrent.

:p What does Theorem 9.12 state about communicating states?
??x
Theorem 9.12 states that if state \(i\) is recurrent and it communicates with state \(j\), then state \(j\) must also be recurrent. This means that for every visit to state \(i\), there is a non-zero probability of visiting state \(j\), which implies an infinite number of visits to both states.

```java
// Pseudocode for proving Theorem 9.12
public class CommunicatingStatesProver {
    private double[][] transitionMatrix; // Transition probabilities

    public void proveCommunicatingRecurrence(int i, int j) {
        if (isRecurrent(i)) { // Assume i is recurrent
            boolean communicates = checkCommunication(i, j);
            if (communicates) {
                System.out.println("State " + j + " must also be recurrent.");
            } else {
                System.out.println("Error: States do not communicate as expected.");
            }
        } else {
            System.out.println("Error: State " + i + " is not recurrent.");
        }
    }

    private boolean checkCommunication(int state1, int state2) {
        // Implement logic to check if states communicate
        return transitionMatrix[state1][state2] > 0 && transitionMatrix[state2][state1] > 0;
    }

    private boolean isRecurrent(int state) {
        // Implement logic to check recurrence of a state
        return checkConvergence(state);
    }
}
```
x??

---

#### Summation of State Probabilities for Recurrent States
Background context: The passage discusses the summation properties of state probabilities in a Markov chain, focusing on recurrent states. A recurrent state is one that will be visited infinitely often with probability 1.

:p What does the equation \(\sum_{s} P^{(m+n)}_{ii}P^s_{ij} = \infty\) imply about state \(j\)?
??x
The equation \(\sum_{s} P^{(m+n)}_{ii}P^s_{ij} = \infty\) implies that the sum of probabilities over all steps \(s\) where the chain starts from state \(i\) and eventually reaches state \(j\) is infinite. This is a property of recurrent states, meaning state \(j\) will be visited infinitely often if starting from any transient or communicating state.

---
#### Recurrence and Transience in Markov Chains
Background context: The text defines recurrence and transience for states within a Markov chain. A state is recurrent if it is visited infinitely often with probability 1; otherwise, it is transient.

:p If state \(i\) is recurrent and communicates with state \(j\), what can we infer about state \(j\)?
??x
If state \(i\) is recurrent and communicates with state \(j\), then state \(j\) must also be recurrent. This follows from the fact that communication between states implies they share a path, and thus if one is recurrent, the other must be as well.

---
#### Limiting Probabilities in Transient Markov Chains
Background context: The text discusses the behavior of transient states over an infinite number of steps. Specifically, it states that for any state \(j\), the probability of being in state \(j\) after a large number of steps approaches zero.

:p What does Theorem 9.14 state about the limiting probabilities in a transient Markov chain?
??x
Theorem 9.14 states that for a transient Markov chain, \(\lim_{n \to \infty} P^n_{ij} = 0\) for all states \(j\). This means that as the number of steps approaches infinity, the probability of being in any state \(j\) tends to zero.

---
#### Non-Existence of Limiting Distributions
Background context: The text discusses conditions under which a limiting distribution does not exist. A limiting distribution is a stationary distribution where \(\pi_j = \lim_{n \to \infty} P^n_{ij}\).

:p How can we prove that the limiting distribution does not exist for a transient Markov chain?
??x
We can prove that the limiting distribution does not exist for a transient Markov chain by showing that all \(\pi_j = 0\) when using Theorem 9.14, which states \(\lim_{n \to \infty} P^n_{ij} = 0\). Since adding an infinite number of zeros results in zero, the limiting distribution does not exist.

---
#### Stationary Distribution for Aperiodic and Irreducible Chains
Background context: The text introduces a theorem that states if all limiting probabilities are zero for an aperiodic and irreducible chain, then no stationary distribution exists. This is based on the concept of non-recurrence and the behavior over time in such chains.

:p What does Theorem 9.17 state about the existence of a stationary distribution for a transient Markov chain?
??x
Theorem 9.17 states that if an aperiodic, irreducible Markov chain has all limiting probabilities zero (\(\pi_j = \lim_{n \to \infty} P^n_{ij} = 0\)), then no stationary distribution exists for such a chain.

---
#### Infinite Random Walk Example
Background context: The text provides an example of an infinite random walk, where at each step, the gambler either gains or loses a dollar with certain probabilities. This is used to illustrate concepts related to transient states and their limiting behaviors.

:p What does the infinite random walk model demonstrate about transient states?
??x
The infinite random walk model demonstrates that in a transient state scenario, the probability of returning to any specific state (like starting point) approaches zero as the number of steps increases. This illustrates how transient states behave over an infinite number of steps, showing they are not visited frequently enough for a non-zero stationary distribution.

---
Note: Additional flashcards can be created based on other parts of the text if needed, but these cover key concepts with detailed explanations and questions as per your instructions.

#### Random Walk Recurrence and Transience
Background context explaining the concept. The random walk shown in Figure 9.3 is a Markov chain where each state communicates with every other state. To determine if the chain is recurrent or transient, we look at state 0. We use Theorem 9.11 to decide based on the expected number of visits \( V = \sum_{n=1}^{\infty} P_n^{(0,0)} \), where \( P_n^{(0,0)} \) is the probability of being at state 0 after n steps.

If \( V \) is finite, then state 0 is transient. Otherwise, it is recurrent.
:p What does Theorem 9.11 help determine about the random walk?
??x
Theorem 9.11 helps determine whether the chain is recurrent or transient by evaluating the expected number of visits to state 0. If \( V \) (the sum of probabilities of returning to state 0 after n steps) is finite, then state 0 is transient; otherwise, it is recurrent.
x??

---

#### Calculation of Expected Number of Visits
We use the formula for calculating the expected number of visits \( V = \sum_{n=1}^{\infty} P_n^{(0,0)} \). For a symmetric random walk where \( p = q = 0.5 \), we show that \( V \) is infinite. Otherwise, if \( p \neq q \), the expected number of visits is finite.

The key steps involve using Lavrov's lemma to simplify and bound the sum.
:p How do we determine whether state 0 in a random walk is transient or recurrent?
??x
We determine the recurrence or transience by evaluating \( V = \sum_{n=1}^{\infty} P_n^{(0,0)} \). For \( p = q = 0.5 \), it can be shown that \( V \) is infinite, meaning state 0 is recurrent. For other values of \( p \neq q \), \( V \) is finite, implying state 0 is transient.
x??

---

#### Lemma 9.18 (Lavrov's Lemma)
Background: We use Lavrov's lemma to simplify the sum for calculating the expected number of visits. The lemma states that for \( n \geq 1 \), 
\[ 4n < \binom{2n}{n} < 4n \left( \frac{n+1}{2n} \right) = 2n + 2. \]
:p What does Lavrov's lemma state?
??x
Lavrov's lemma states that for \( n \geq 1 \),
\[ 4n < \binom{2n}{n} < 4n \left( \frac{n+1}{2n} \right) = 2n + 2. \]
This helps in bounding the sum and determining whether the expected number of visits is finite or infinite.
x??

---

#### Expected Number of Visits Simplification
Using Lavrov's lemma, we simplify the expression for \( V \):
\[ V = \sum_{n=1}^{\infty} P_n^{(0,0)} < \sum_{n=1}^{\infty} 4^n q^n (2p - p) = \sum_{n=1}^{\infty} 4^n q^n. \]
If \( p = q = 0.5 \), then 
\[ V > \sum_{n=1}^{\infty} \frac{4}{(2n+1)}. \]
For \( p \neq q \),
\[ V < \sum_{n=1}^{\infty} (4pq)^n. \]

:p What is the simplified form of the expected number of visits for a random walk with symmetric and non-symmetric probabilities?
??x
For a symmetric random walk where \( p = q = 0.5 \),
\[ V > \sum_{n=1}^{\infty} \frac{4}{(2n+1)} = \infty, \]
indicating that the chain is recurrent.

For non-symmetric probabilities with \( p \neq q \),
\[ V < \sum_{n=1}^{\infty} (4pq)^n. \]
Since \( 4pq < 1 \), this sum converges to a finite value, indicating that the chain is transient.
x??

---

#### Recurrence and Transience of Random Walk
From the above analysis, we conclude that the random walk in Figure 9.3 is recurrent if and only if \( p = 0.5 \). Otherwise, it is transient.

:p What is the condition for recurrence or transience of a random walk?
??x
The random walk shown in Figure 9.3 is recurrent when \( p = 0.5 \) (symmetric), and transient otherwise.
x??

---

#### First Passage Probability
Background: For a random walk, we define \( f_0 \) as the probability that the chain ever returns to state 0.

For \( p = 0.5 \), \( f_0 = 1 \). Otherwise, \( f_0 < 1 \).

:p What is the value of \( f_0 \) for a random walk?
??x
For a symmetric random walk where \( p = q = 0.5 \), \( f_0 = 1 \). For an asymmetric random walk with \( p \neq q \), \( f_0 < 1 \).
x??

---

#### Proof of First Passage Probability for Transient Case
Background: We use conditioning to prove that in the case where the chain is transient (with rightward drift, \( p > q \)), we have \( f_{-1,0} = 1 \) and \( f_{1,0} = \frac{q}{p} \).

:p How do we prove the first passage probability for a transient random walk?
??x
For a transient random walk with rightward drift (\( p > q \)), we use conditioning to show:
\[ f_{-1,0} = 1. \]
This is because the chain has a rightward drift and must eventually return to state 0.

Additionally,
\[ f_{1,0} = \frac{q}{p}. \]
This can be seen by conditioning on the first step as follows:
\[ f_{1,0} = q \cdot 1 + p \cdot f_{2,0} = q + p \cdot \left( \frac{q}{p} \right)^2. \]
Given that \( q + p = 1 \), we find that
\[ f_{1,0} = \frac{q}{p}. \]
x??

---

#### Positive Recurrence vs Null Recurrence
Positive recurrence and null recurrence are two types of recurrent Markov chains. In a positive-recurrent MC, the mean time between recurrences (returning to the same state) is finite. In contrast, in a null-recurrent MC, the mean time between recurrences is infinite.

For example, consider a random walk with \( p = \frac{1}{2} \):
- This chain is recurrent.
- However, it can be shown that the mean number of steps between visits to state 0 is infinite, making state 0 null recurrent.

If state i is positive recurrent and \(i \leftrightarrow j\), then j is also positive recurrent. If state i is null recurrent and \(i \leftrightarrow j\), then j is null recurrent as well.
:p What does the term "positive recurrence" mean in Markov chains?
??x
In a positive-recurrent Markov chain, the expected time to return to any given state (e.g., state 0) is finite. This implies that although the chain may visit states infinitely often, the average waiting time between visits is bounded.
x??

---

#### Null Recurrence Example: Symmetric Random Walk
A symmetric random walk with \( p = \frac{1}{2} \) is an example of a null-recurrent Markov chain.

To show that state 0 in such a walk is null recurrent, consider the following:
- The mean number of time steps between visits to state 0, denoted as \( m_{0,0} \), is infinite.
- If we assume \( m_{0,0} \) is finite, then it leads to contradictions when calculating conditional expectations.

For instance:
```java
public class NullRecurrenceExample {
    public double meanTimeToReturn(int state) {
        if (state == 0) return Double.POSITIVE_INFINITY; // Null recurrence case
        else return 2 * meanTimeToReturn(state - 1); // Simplified logic for demonstration
    }
}
```
This code abstractly represents the concept that the expected time to return to state 0 is infinite.
:p Why is state 0 null recurrent in a symmetric random walk?
??x
In a symmetric random walk, state 0 is visited infinitely often but the average waiting time between visits is infinite. This makes it null recurrent because the mean number of steps required to return to state 0 from itself is \(\infty\).
x??

---

#### Mean Time Between Visits
The theorem provided states that if a state \(i\) is positive recurrent and communicable with another state \(j\), then \(j\) is also positive recurrent. Conversely, if \(i\) is null recurrent and communicable with \(j\), then \(j\) is also null recurrent.

For the symmetric random walk example:
- The mean time between visits to state 0 (denoted as \(m_{0,0}\)) needs to be proven to be infinite.
- A proof by contradiction shows that if \(m_{0,0}\) were finite, it would lead to an inconsistency in calculating conditional expectations.

Consider the following simplified equations:
\[ m_{1,0} = 1 + \frac{1}{2} \cdot 0 + \frac{1}{2} \cdot 2m_{1,0} \]
This equation results in a contradiction if \(m_{1,0}\) is assumed to be finite.
:p What theorem relates positive recurrence and null recurrence between states?
??x
The theorem states that if state i is positive recurrent and communicates with state j (i.e., \(i \leftrightarrow j\)), then state j is also positive recurrent. Similarly, if state i is null recurrent and communicates with state j, then state j is also null recurrent.
x??

---

#### Proof by Contradiction
A proof by contradiction is used to show that the mean number of steps between visits to state 0 in a symmetric random walk is infinite.

The logic follows:
1. Assume \(m_{0,0}\) is finite.
2. Then \(m_{1,0}\) must also be finite because it can be expressed as:
\[ m_{1,0} = 1 + \frac{1}{2} \cdot 0 + \frac{1}{2} \cdot 2m_{1,0} \]
3. Simplifying this equation leads to a contradiction.
4. Therefore, \(m_{0,0}\) must be infinite.

This proof highlights the complexity of calculating mean recurrence times in null-recurrent chains.
:p How does the proof by contradiction work for state 0 in the random walk?
??x
By assuming that \( m_{0,0} \) is finite, we derive a relationship involving other states. Specifically:
\[ m_{1,0} = 1 + \frac{1}{2} \cdot 0 + \frac{1}{2} \cdot 2m_{1,0} \]
This simplifies to \( m_{1,0} = 1 + m_{1,0} \), which is a contradiction because it implies \(0 = 1\). Therefore, our initial assumption that \( m_{0,0} \) is finite must be incorrect, and \( m_{0,0} \) is actually infinite.
x??

---

---
#### Ergodicity and Limiting Probabilities of DTMCs
Background context: The text discusses ergodic theory for Markov chains, focusing on the properties required for a chain to be ergodic (aperiodicity, irreducibility, positive recurrence) and the implications of these properties. It explains that for finite-state chains, positive recurrence is implied by irreducibility.

:p What does it mean for a Discrete-Time Markov Chain (DTMC) to be ergodic?
??x
An ergodic DTMC is one that has all three desirable properties: aperiodicity, irreducibility, and positive recurrence. This means the chain will exhibit certain regular behaviors over time, allowing us to define limiting probabilities.

For finite-state chains, since positive recurrence follows from irreducibility, only aperiodicity and irreducibility are needed for ergodicity.
x??

---
#### Ergodic Theorem of Markov Chains
Background context: The Ergodic Theorem states that under certain conditions (ergodic), the limiting probabilities exist and can be computed. Specifically, for an ergodic DTMC, these limits are positive and equal to 1 over the mean time between visits to a state.

:p According to Theorem 9.25, what does it mean for an ergodic DTMC?
??x
For a recurrent, aperiodic, irreducible DTMC, the limiting probabilities \(\pi_j\) exist and are given by:
\[
\pi_j = \lim_{n \to \infty} P^n_{ij} = \frac{1}{m_{jj}}
\]
where \(m_{jj}\) is the mean time between visits to state \(j\). For a positive recurrent DTMC, all \(\pi_j > 0\).

The theorem extends Theorems 9.4 and 9.6 to include infinite-state chains.
x??

---
#### Null-Recurrence and Limiting Probabilities
Background context: A null-recurrent chain has a mean time between visits to each state that is infinite, meaning the limiting probabilities are zero.

:p What does Theorem 9.26 imply for an aperiodic, null-recurrent Markov chain?
??x
For an aperiodic, null-recurrent Markov chain, all the limiting probabilities \(\pi_j\) are zero because \(m_{jj} = \infty\). Consequently, neither a limiting distribution nor a stationary distribution exists.

This result is derived from Theorem 9.25 by recognizing that infinite mean times imply zero probabilities.
x??

---
#### Summary of Limiting Distributions and Stationary Distributions
Background context: The text summarizes the possible states (transient, null-recurrent, or positive recurrent) for irreducible DTMCs and their implications on limiting distributions and stationary distributions.

:p According to Theorem 9.27, what are the two main classes of an irreducible, aperiodic DTMC?
??x
There are two main classes:

1. **All states are transient:** In this case, \(\pi_j = \lim_{n \to \infty} P^n_{ij} = 0\) for all \(j\), and no stationary distribution exists.
2. **All states are positive recurrent:** Here, the limiting probabilities \(\pi_j > 0\) and equal to \(\frac{1}{m_{jj}}\), where \(m_{jj}\) is finite. The limiting distribution exists and is also a unique stationary distribution.

The key here is that for positive recurrence, the sum of all \(\pi_j\) equals 1.
x??

---
#### Corollary on Summing Limiting Probabilities
Background context: This corollary confirms that the limiting probabilities for positive recurrent states indeed sum up to 1.

:p Why do the limiting probabilities in a positive recurrent DTMC add up to 1?
??x
The limiting probabilities \(\pi_j\) are defined as:
\[
\pi_j = \lim_{n \to \infty} P^n_{ij} = \frac{1}{m_{jj}}
\]
and since \(m_{jj}\) is finite, \(\pi_j > 0\). According to the theory, these probabilities must sum up to 1 because they represent a valid probability distribution.

Formally:
\[
\sum_{j=0}^\infty \pi_j = \sum_{j=0}^\infty \frac{1}{m_{jj}} = 1
\]
This ensures that the limiting distribution is a proper probability distribution.
x??

---
#### Transience, Null Recurrence, and Positive Recurrence in Irreducible Chains
Background context: The text explains how transience, null recurrence, and positive recurrence are class properties, meaning all states share the same property.

:p What does it mean for an irreducible Markov chain to have a certain state type (transient, null-recurrent, or positive recurrent)?
??x
In an irreducible Markov chain, all states must be of the same type: either all transient, all null recurrent, or all positive recurrent. This is because transience, null recurrence, and positive recurrence are class properties.

For example:
- If one state in an irreducible chain is transient, then all states are transient.
- Similarly, if one state is null-recurrent or positive recurrent, all states share this property.
x??

---

