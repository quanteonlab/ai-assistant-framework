# Flashcards: 7A003--Performance-modeling-and-design-of-computer-systems_processed (Part 16)

**Starting Chapter:** 9.10 Proof of Ergodic Theorem of Markov Chains

---

#### Aperiodic and Irreducible Chains
Background context: The text discusses properties of a Markov chain, specifically focusing on aperiodicity and irreducibility. A period $d(i)$ is defined as the greatest common divisor (gcd) of all $ n > 0 $ such that $ P^n(i, i) > 0 $. A chain is aperiodic if $ d(i) = 1$for all states $ i$, and irreducible means it's possible to get from any state to any other state in a finite number of steps.

:p What does the property of being aperiodic mean for a Markov chain?
??x
Aperiodicity ensures that no state can be revisited at fixed intervals, meaning the chain is not trapped in cycles. This implies that the probability of returning to a state $i$ after an arbitrary number of steps is independent of the return time.
x??

---
#### Stationary Solution and Normalizing Constant
Background context: Given the condition $π/\vec{v}_i·D_{ii} = π/\vec{v}_i $, where $ D $ represents a diagonal matrix, it indicates that the stationary probability vector $π$ satisfies this equation. The normalizing constant ensures the probabilities sum to 1.

:p How does one determine the stationary solution for a Markov chain?
??x
To find the stationary solution, you solve the equation $π/\vec{v}_i·D_{ii} = π/\vec{v}_i$ and then normalize the resulting vector so that its components sum to 1.
x??

---
#### Positive Recurrence in Irreducible Chains
Background context: The text states that a chain is positive recurrent if it has a stationary solution. For irreducible chains, this means every state can be reached from any other state, leading to the conclusion of positive recurrence.

:p What does it mean for a Markov chain to be positive recurrent?
??x
Positive recurrence means that the expected return time to any state $i$ is finite. In simpler terms, given enough time, the system will visit every state infinitely often on average.
x??

---
#### Limiting Probabilities in Non-Irreducible Chains
Background context: For a non-irreducible Markov chain, limiting probabilities are defined component-wise rather than globally. The chain can be divided into irreducible components.

:p Why do we need to consider irreducible components when dealing with non-irreducible chains?
??x
Irreducible components help in analyzing the behavior of each part of the chain independently. This approach allows us to understand the limiting probabilities within these sub-chains, which may have different stationary distributions.
x??

---
#### Equivalences of Limiting Probabilities
Background context: The text outlines several equivalent ways to represent limiting probabilities, such as average fraction of time spent in a state, stationary probability, and reciprocal of mean time between visits.

:p How are the different representations of limiting probabilities related?
??x
The different representations are interconnected. For instance, the stationary probability $π_j $ can be seen as the long-term frequency of being in state$j $, or as the reciprocal of the expected return time to state$ j$. These concepts provide multiple perspectives on the same fundamental property.
x??

---
#### Techniques for Determining Limiting Probabilities
Background context: The text mentions several methods for determining limiting probabilities, including matrix powers, stationary equations, and time-reversibility.

:p What are some techniques used to determine limiting probabilities?
??x
Techniques include:
1. Raising the transition matrix $P$ to high powers.
2. Solving stationary (or balance) equations: $πP = π$.
3. Using time-reversibility relations.
Each method has its own advantages and limitations, depending on the specific structure of the Markov chain.
x??

---

#### Definition of fkii and Pki
Background context: The definitions provided are fundamental to understanding the behavior of a Markov chain, specifically focusing on the probability of returning to state $i$ after a certain number of transitions. 
- $f_{kii}$ is defined as the probability of first returning to state $i$ exactly at the $k$-th transition.
- $P_{ki}$ is defined as the probability of being in state $i$ after $k$ transitions, given that we started in state $i$.
- The expected number of time steps to return to state $i $ is denoted by$m_{ii} = \sum_{k=0}^{\infty} k f_{kii}$.

:p What are the definitions for $f_{kii}$ and $P_{ki}$?
??x
$f_{kii}$ represents the probability of first returning to state $i$ exactly at the $k$-th transition. On the other hand,$ P_{ki}$is defined as the probability of being in state $ i$after $ k $ transitions, given that we started in state $ i $. The expected number of time steps to return to state $ i$can be calculated using the formula:
$$m_{ii} = \sum_{k=0}^{\infty} k f_{kii}.$$x??

---

#### Definition of limsup and liminf
Background context: These concepts are crucial for understanding the limiting behavior of sequences, specifically in the context of Markov chains. The definitions provide a way to describe the upper and lower bounds of a sequence as $n$ approaches infinity.
1. **lim sup**: If $\limsup_{n\to\infty}a_n = b $, then for any $\epsilon > 0 $, there exists an $ N_0(\epsilon)$such that for all $ n \geq N_0(\epsilon)$, the sequence $ a_n$is less than $ b + \epsilon$. Furthermore,$ b$ is the smallest value satisfying this condition.
2. **lim inf**: If $\liminf_{n\to\infty}a_n = b $, then for any $\epsilon > 0 $, there exists an $ N_0(\epsilon)$such that for all $ n \geq N_0(\epsilon)$, the sequence $ a_n$is greater than $ b - \epsilon$. Furthermore,$ b$ is the largest value satisfying this condition.

:p What are the definitions of limsup and liminf?
??x
The definition of **lim sup** states that if $\limsup_{n\to\infty}a_n = b $, then for any $\epsilon > 0 $, there exists an $ N_0(\epsilon)$such that for all $ n \geq N_0(\epsilon)$, the sequence $ a_n$is less than $ b + \epsilon$. Furthermore,$ b$ is the smallest value satisfying this condition.

The definition of **lim inf** states that if $\liminf_{n\to\infty}a_n = b $, then for any $\epsilon > 0 $, there exists an $ N_0(\epsilon)$such that for all $ n \geq N_0(\epsilon)$, the sequence $ a_n$is greater than $ b - \epsilon$. Furthermore,$ b$ is the largest value satisfying this condition.
x??

---

#### Immediate Consequences of limsup
Background context: The following are three immediate consequences of the definition of **lim sup**. These can also be applied to **lim inf** by similar reasoning.

1. If $\limsup_{n\to\infty}a_n = b $, then for all $\epsilon > 0 $, the sequence $ a_n $exceeds the value$ b - \epsilon$ infinitely often.
2. There exists an infinite subsequence of $\{a_n\}$ denoted by $\{a_{n_j}\}$ where $n_1 < n_2 < n_3 < \ldots$, such that $\lim_{j\to\infty}a_{n_j} = b$.
3. If there is an infinite subsequence of $\{a_m\}$ denoted by $\{a_{m_j}\}$ where $m_1 < m_2 < m_3 < \ldots$ and if $\lim_{j\to\infty}a_{m_j} \neq b$(or does not exist), then there exists $ b' < b$such that there are infinitely many elements of $\{a_{m_j}\}$ below $b'$.

:p What are the three immediate consequences of the definition of limsup?
??x
The first consequence states that if $\limsup_{n\to\infty}a_n = b $, then for all $\epsilon > 0 $, the sequence $ a_n $exceeds the value$ b - \epsilon$ infinitely often.

The second consequence is that there exists an infinite subsequence of $\{a_n\}$ denoted by $\{a_{n_j}\}$ where $n_1 < n_2 < n_3 < \ldots$, such that $\lim_{j\to\infty}a_{n_j} = b$.

The third consequence states that if there is an infinite subsequence of $\{a_m\}$ denoted by $\{a_{m_j}\}$ where $m_1 < m_2 < m_3 < \ldots$ and if $\lim_{j\to\infty}a_{m_j} \neq b$(or does not exist), then there exists $ b' < b$such that there are infinitely many elements of $\{a_{m_j}\}$ below $b'$.
x??

---

--- 

This format can be repeated for each key concept in the provided text, ensuring clarity and relevance to understanding Markov chains.

#### Definition of b/primeto
Background context: In this section, we are dealing with a sequence $\{a_m\}$ that converges to $b/\text{prime}/\text{prime}$, where $ b/\text{prime}/\text{prime} < b$. We need to define $ b/\text{primeto}$such that it lies between $ b/\text{prime}/\text{prime}$and $ b$.
:p Define $b/\text{primeto}$ in this context.
??x
We define $b/\text{primeto}$ as a value that lies strictly between $b/\text{prime}/\text{prime}$ and $b$. This is done to ensure there are infinitely many elements of the sequence $\{a_m\}$ below $b/\text{primeto}$, while still being less than $ b$.
x??

---

#### Subsequence {amj} Limit
Background context: The subsequence $\{a_{m_j}\}$ is defined such that it converges to $b/\text{prime}/\text{prime}$. If the limit of this subsequence does not exist, we can still find a value $ b/\text{prime}$such that there are infinitely many elements below $ b/\text{prime}$.
:p Explain why if $\{a_{m_j}\}$ does not have a limit, then $b/\text{prime} = b - \epsilon_1$ works.
??x
If the subsequence $\{a_{m_j}\}$ does not have a limit, there exists some $\epsilon_1 > 0$ such that no point in the sequence is above $b - \epsilon_1$. Hence, we can define $ b/\text{prime} = b - \epsilon_1$. This implies that for any point in the subsequence, there will always be another element below $ b/\text{prime}$, ensuring an infinite number of elements below this value.
x??

---

#### Recurrent Markov Chain - Lemma 9.41
Background context: For a recurrent Markov chain, we are interested in showing that given $\lambda = \limsup_{k \to \infty} P_k^{ii}$, there exists a subsequence $\{P_{n_j}^{ii}\}$ such that $P_{n_j}^{ii} \to \lambda$. Additionally, for any positive constant $ c$and integer $ d \geq 0$, we show that the limit of the sequence $ P_{n_j - c \cdot d}^{ii}$also converges to $\lambda$.
:p Prove that $\lim_{j \to \infty} P_{n_j - c}^{ii} = \lambda $ for a given positive constant$c$.
??x
To prove this, assume the contrary: $\lim_{j \to \infty} P_{n_j - c}^{ii} \neq \lambda $. By Lemma 9.40.3, there exists some $\lambda' < \lambda $ such that$P_{n_j - c}^{ii} < \lambda'$ for infinitely many indices $j$.

Let $\epsilon_1 = \frac{fc^{ii} (\lambda - \lambda')}{3}$. Since the series $\sum_{k=0}^{\infty} f_k^{ii} = 1 $, we can find $ N$such that:
$$\sum_{k=N+1}^{\infty} f_k^{ii} < \epsilon_1.$$

Choose $j $ large enough so that$n_j \geq N$ and:
$$P_{n_j}^{ii} > \lambda - \frac{\epsilon_1}{2},$$since $\lim_{j \to \infty} P_{n_j}^{ii} = \lambda$.

We also know that:
$$P_{n_j - c}^{ii} < \lambda' < \lambda,$$and$$

P_n^{ii} < \lambda + \epsilon_1 \quad \forall n \geq n_j - N.$$

By conditioning on the first return time, we get:
$$

P_{n_j}^{ii} = \sum_{k=0}^{n_j} f_k^{ii} P_{n_j - k}^{ii}.$$

Thus,$$

P_{n_j}^{ii} \leq \sum_{k=0}^{N} f_k^{ii} P_{n_j - k}^{ii} + \sum_{k=N+1}^{n_j} f_k^{ii} P_{n_j - k}^{ii}.$$

Since $P_k^{ii} \leq 1$, this simplifies to:
$$P_{n_j}^{ii} \leq \sum_{k=0}^{N} f_k^{ii} P_{n_j - k}^{ii} + (n_j - N) \epsilon_1.$$

Using the fact that $P_{n_j - k}^{ii} < \lambda + \epsilon_1$ and substituting:
$$P_{n_j}^{ii} < \sum_{k=0, k \neq c}^{N} f_k^{ii} (\lambda + \epsilon_1) + f_c^{ii} \cdot \lambda' + \epsilon_1.$$

Simplifying further:
$$

P_{n_j}^{ii} < (1 - f_c^{ii})(\lambda + \epsilon_1) + f_c^{ii} \cdot \lambda' + \epsilon_1,$$which is less than $\lambda - \frac{\epsilon_1}{2}$.

This contradicts $P_{n_j}^{ii} > \lambda - \frac{\epsilon_1}{2}$, thus proving that:
$$\lim_{j \to \infty} P_{n_j - c}^{ii} = \lambda.$$

By induction, for any integer $d \geq 0$:
$$\lim_{j \to \infty} P_{n_j - c \cdot d}^{ii} = \lambda.$$x??

---

#### Recurrent Markov Chain - Lemma 9.42
Background context: Similar to the previous lemma, but dealing with the limit inferior instead of the limit superior.
:p Prove that $\lim_{j \to \infty} P_{m_j - c}^{ii} = \mu $ for a given positive constant$c$.
??x
To prove this, assume the contrary: $\lim_{j \to \infty} P_{m_j - c}^{ii} \neq \mu $. By an analogous argument to Lemma 9.41, there exists some $\mu' > \mu $ such that$P_{m_j - c}^{ii} > \mu'$ for infinitely many indices $j$.

Let $\epsilon_2 = \frac{fc^{ii} (\mu' - \mu)}{3}$. Since the series $\sum_{k=0}^{\infty} f_k^{ii} = 1 $, we can find $ M$such that:
$$\sum_{k=M+1}^{\infty} f_k^{ii} < \epsilon_2.$$

Choose $j $ large enough so that$m_j \geq M$ and:
$$P_{m_j}^{ii} < \mu + \frac{\epsilon_2}{2},$$since $\lim_{j \to \infty} P_{m_j}^{ii} = \mu$.

We also know that:
$$P_{m_j - c}^{ii} > \mu',$$and$$

P_n^{ii} > \mu - \epsilon_2 \quad \forall n \geq m_j - M.$$

By conditioning on the first return time, we get:
$$

P_{m_j}^{ii} = \sum_{k=0}^{m_j} f_k^{ii} P_{m_j - k}^{ii}.$$

Thus,$$

P_{m_j}^{ii} \leq \sum_{k=0}^{M} f_k^{ii} P_{m_j - k}^{ii} + \sum_{k=M+1}^{m_j} f_k^{ii} P_{m_j - k}^{ii}.$$

Since $P_k^{ii} \leq 1$, this simplifies to:
$$P_{m_j}^{ii} \leq \sum_{k=0, k \neq c}^{M} f_k^{ii} (P_{m_j - k}^{ii}) + M \epsilon_2.$$

Using the fact that $P_{m_j - k}^{ii} > \mu - \epsilon_2$ and substituting:
$$P_{m_j}^{ii} < \sum_{k=0, k \neq c}^{M} f_k^{ii} (\mu - \epsilon_2) + M \epsilon_2.$$

Simplifying further:
$$

P_{m_j}^{ii} > (1 - f_c^{ii})(\mu - \epsilon_2) + f_c^{ii} \cdot \mu' - \epsilon_2,$$which is greater than $\mu + \frac{\epsilon_2}{2}$.

This contradicts $P_{m_j}^{ii} < \mu + \frac{\epsilon_2}{2}$, thus proving that:
$$\lim_{j \to \infty} P_{m_j - c}^{ii} = \mu.$$

By induction, for any integer $d \geq 0$:
$$\lim_{j \to \infty} P_{m_j - c \cdot d}^{ii} = \mu.$$x??

---

#### Definition of Recurrent and Aperiodic Markov Chain
A recurrent, aperiodic Markov chain is defined such that every state i returns to itself with probability 1, meaning $\lim_{n \to \infty} P^n_{ii}$ exists for all states $i$. The sequences $\{f_k^{(ii)}\}$ and $\{P_k^{(ii)}\}$ are specified in Definition 9.38.
:p What is the definition of a recurrent, aperiodic Markov chain?
??x
A recurrent, aperiodic Markov chain is one where every state $i $ will return to itself with probability 1 as$n \to \infty $. The sequences $\{f_k^{(ii)}\}$ and $\{P_k^{(ii)}\}$ are used to describe the probabilities of returning to state $i$ after exactly $k$ steps, or within $k+1$ steps respectively.
x??

---

#### Sequence Definitions
The sequence $r_n = f_{n+1}^{(ii)} + f_{n+2}^{(ii)} + \cdots $ is defined as the probability that the time to return to state$i $ exceeds$n $. Also, $ m_i $is the expected time to return to state$ i $, given by$ m_i = \sum_{k=0}^{\infty} k f_k^{(ii)}$.
:p What are the definitions of sequences $r_n $ and$m_i$?
??x
The sequence $r_n $ is defined as the probability that the time to return to state$i $ exceeds$ n $, i.e.,$ r_n = P(\text{Time to return to } i \text{ exceeds } n)$. The expected return time $ m_i$is given by the sum of the probabilities of returning after each possible step, weighted by their respective steps:$ m_i = \sum_{k=0}^{\infty} k f_k^{(ii)}$.
x??

---

#### Relationship Between Sequences
The relationship between sequences and return times can be expressed as $m_i = \sum_{n=0}^{\infty} n r_n $, where $ r_n $is the probability of not returning to state$ i $within the first$ n+1$ steps.
:p How does the expected return time $m_i $ relate to the sequence$r_n$?
??x
The expected return time $m_i $ can be expressed as the sum of the probabilities of not returning to state$i $ in the first$ n+1 $ steps, weighted by their respective times:$m_i = \sum_{n=0}^{\infty} n r_n$.
x??

---

#### Summation Property
The summation property states that for all $n $, $\sum_{k=0}^{n} r_k P^n-k_{ii} = 1 $. This is because the probability of not returning to state $ i $before time$ n$ and then eventually returning must sum up to 1.
:p What is the summation property for return times?
??x
The summation property states that $\sum_{k=0}^{n} r_k P^n-k_{ii} = 1 $, meaning the probability of not returning to state $ i $before time$ n$ and then eventually returning sums up to 1.
x??

---

#### Limit Superior and Inferior
Let $\lambda = \limsup_{n \to \infty} P^{(n)}_{ii}$ and $\mu = \liminf_{n \to \infty} P^{(n)}_{ii}$. The objective is to show that both limits are equal, establishing the existence of a unique stationary distribution.
:p What are $\lambda $ and$\mu$ in this context?
??x $\lambda = \limsup_{n \to \infty} P^{(n)}_{ii}$ represents the upper limit of $P^n_{ii}$ as $n$ approaches infinity, while $\mu = \liminf_{n \to \infty} P^{(n)}_{ii}$ represents the lower limit. The goal is to show that these two limits are equal, indicating a unique stationary distribution for the Markov chain.
x??

---

#### Subsequences and Limits
Consider subsequences of $P^n_{ii}$ such that $\lim_{j \to \infty} P^{n_j}_{ii} = \lambda$ and $\lim_{j \to \infty} P^{m_j}_{ii} = \mu$. By applying Lemmas 9.41 and 9.42, it can be shown that $\mu \geq \lambda$.
:p What are the subsequences used to prove $\mu \geq \lambda$?
??x
Subsequences of $P^n_{ii}$ are considered such that $\lim_{j \to \infty} P^{n_j}_{ii} = \lambda$ and $\lim_{j \to \infty} P^{m_j}_{ii} = \mu$. By applying Lemmas 9.41 and 9.42, it can be shown that the lower limit $\mu $ is greater than or equal to the upper limit $\lambda$.
x??

---

#### Inequality Chain
Using the inequality chain derived from Equation (9.32), it follows that $\lambda \leq \frac{1}{\sum_{k=0}^{\infty} r_k} \leq \mu $. This establishes that $\lambda = \mu$.
:p What is the inequality chain used to prove $\lambda = \mu$?
??x
The inequality chain derived from Equation (9.32) shows that $\lambda \leq \frac{1}{\sum_{k=0}^{\infty} r_k} \leq \mu$. This implies that the upper and lower limits are equal, thus establishing a unique stationary distribution.
x??

---

#### Irreducibility, Aperiodicity, and Positive Recurrence
Background context: In this section, we are given several transition matrices to determine whether they represent an irreducible, aperiodic, and positive recurrent Markov chain. An irreducible Markov chain is one where all states communicate with each other. Aperiodicity means the period of the state is 1 (the greatest common divisor of return times is 1). Positive recurrence implies that the expected time to return to any state is finite.

:p For matrix $\begin{pmatrix} 1/4 & 3/4 \\ 1/2 & 1/2 \end{pmatrix}$, determine if it represents an irreducible, aperiodic, and positive recurrent Markov chain.
??x
To determine these properties:
- **Irreducibility**: Check if all states communicate with each other. State 1 can go to state 2 and vice versa through the paths available.
- **Aperiodicity**: Check the greatest common divisor of return times. For a period of 1, this GCD should be 1.
- **Positive Recurrence**: Ensure the expected time to return to any state is finite.

The chain is irreducible because there's a path between every pair of states (e.g., $P(1 \to 2) = 3/4$, and from 2 back to 1, through another path or loop).

To check for aperiodicity:
- For state 1: The return probability in one step is $P_{11}^{(1)} = 1/4$.
- For state 2: Similarly, the return probability in one step can be checked.
Since we have non-zero probabilities of returning to both states within one step, the GCD of return times is 1.

Positive recurrence is generally assumed given that the chain is finite and irreducible. If not, further detailed analysis would be needed using expected hitting times.

x??

---
#### Time-Average Fraction Using Balance Equations
Background context: The problem involves solving for the time-average fraction of time spent in each state using balance equations. This typically requires setting up a system of linear equations based on steady-state probabilities.

:p For matrix $P(1) = \begin{pmatrix} 0 & 2/3 & 0 \\ 1/3 & 0 & 2/3 \\ 0 & 1/3 & 0 \end{pmatrix}$, find the time-average fraction of time spent in each state.
??x
To solve this, we set up balance equations based on steady-state probabilities $\pi_i$ for states 1, 2, and 3.

$$\begin{cases}
\pi_1 = \frac{2}{3} \pi_2 \\
\pi_2 = \frac{1}{3} \pi_1 + \frac{2}{3} \pi_3 \\
\pi_3 = \frac{1}{3} \pi_2
\end{cases}$$

Summing these probabilities gives:
$$\pi_1 + \pi_2 + \pi_3 = 1$$

Solving this system of equations, we can find the values of $\pi_i$.

```java
// Pseudocode to solve balance equations
public class BalanceEquations {
    public void solveBalanceEquations(double[][] transitionMatrix) {
        double[] pi = new double[transitionMatrix.length];
        // Initialize and solve for pi using linear algebra or iterative methods
        // Example: Using matrix inversion
        Matrix A = new Matrix(transitionMatrix);
        Vector b = new Vector(new double[]{1, 0, 0});
        pi = A.solve(b).toArray();
    }
}
```

x??

---
#### Data Centers and Transition Probabilities
Background context: This problem involves a data center that can transition between "working" and "down" states based on specific probabilities. The goal is to draw the state diagram and determine if it is ergodic, time-reversible, and recurrent.

:p Draw the state diagram for the given Markov chain of a data center.
??x
The state diagram would have two nodes: Working (state 1) and Down (state 2). Arrows between these states represent transitions based on the given probabilities:

- From state 1 to state 2 with probability $\frac{1}{6}$.
- From state 2 to state 1 with probability $\frac{3}{4}$.

The diagram would look like this:
```
State 1 (Working) -> State 2 (Down)
         |                  |
         v                  v
State 2 (Down) <- State 1 (Working)
```

x??

---
#### Ergodicity and Time-Reversibility
Background context: We need to check the ergodicity and time-reversibility of a given Markov chain. Ergodicity means that the chain is both irreducible and aperiodic, implying it converges to a unique stationary distribution.

:p Determine if matrix $P(1)$ from part (b) of exercise 9.2 is ergodic.
??x
To determine ergodicity:
- **Irreducibility**: Check if all states communicate with each other.
- **Aperiodicity**: Ensure the greatest common divisor of return times is 1.

Matrix $P(1) = \begin{pmatrix} 0 & 2/3 & 0 \\ 1/3 & 0 & 2/3 \\ 0 & 1/3 & 0 \end{pmatrix}$:

- **Irreducibility**: States communicate. Each state can be reached from another.
- **Aperiodicity**: Check return times and their GCD.

For this matrix, the chain is irreducible and aperiodic (since it’s a random walk-like structure), so it is ergodic.

x??

---
#### Time-Reversibility
Background context: Time-reversibility means that the transition probabilities in reverse order are also valid. This implies detailed balance equations hold between states $i $ and$j$.

:p Is matrix $P(2)$ from part (b) of exercise 9.2 time-reversible?
??x
To check for time-reversibility, we verify if the detailed balance equations hold:

$$\pi_i P_{ij} = \pi_j P_{ji}$$

For matrix $P(2) = \begin{pmatrix} 1/3 & 2/3 & 0 \\ 0 & 1/3 & 2/3 \\ 0 & 1/3 & 2/3 \end{pmatrix}$:

- Detailed balance equations:
  
$$\pi_1 P_{12} = \pi_2 P_{21}, \quad \pi_2 P_{23} = \pi_3 P_{32}, \quad \pi_3 P_{31} = \pi_1 P_{13}$$

Assuming $\pi_1, \pi_2, \pi_3$ are equal (which is often the case for such symmetric matrices), detailed balance holds. Therefore, the chain is time-reversible.

x??

---

#### Why or why not?
Background context: This question is asking for a justification whether a statement (or conditions) is true or false. It requires understanding of irreducibility and aperiodicity.

:p What fraction of time is the data center working, given an ergodic DTMC with n > 1 states?

??x
For an ergodic (irreducible and aperiodic) DTMC with positive recurrent states, the limiting probability π_j for each state j is non-zero. This means that in the long run, the system spends a fraction of time in each state proportional to its steady-state probability.

The answer: The data center works a fraction of time equal to the sum of the steady-state probabilities of all working states, which can be denoted as $\sum_{j \text{ (working)}} \pi_j$.

For instance, if there are two states in the system where state 1 is the "working" state and state 2 is the "non-working" state, then the fraction of time the data center works would be $\pi_1 + (1 - \pi_2)$ assuming π_1 and π_2 are the steady-state probabilities for states 1 and 2 respectively.

```java
public class ErgodicDataCenter {
    double[] pi; // Steady-state probability vector

    public double fractionOfTimeWorking() {
        double workingFraction = 0.0;
        for (int i = 0; i < pi.length; i++) {
            if (pi[i] > 0) { // Working states
                workingFraction += pi[i];
            }
        }
        return workingFraction;
    }
}
```
x??

---

#### What is the expected number of days between backhoe failures?
Background context: This question involves understanding the concept of mean recurrence time, which is related to the mean time until a state returns to itself in a Markov chain. For an irreducible and positive recurrent DTMC, the mean recurrence time for any state j (denoted as $m_{jj}$) can be computed.

:p What fraction of days does it take on average between backhoe failures?

??x
In a positive recurrent DTMC, the expected number of steps (days in this context) until the system returns to the same state is given by $m_{jj}$, where $ m_{jj}$ is the mean recurrence time for state j.

The answer: The fraction of days between backhoe failures is equal to the mean recurrence time $m_{jj}$ associated with the failure state. If the chain is ergodic and positive recurrent, this value can be derived from the steady-state probabilities or directly from the transition matrix using known formulas or numerical methods.

```java
public class BackhoeFailures {
    // Assume T is the transition matrix
    double m_jj; // Mean recurrence time for state j

    public double expectedDaysBetweenFailures() {
        return m_jj;
    }
}
```
x??

---

#### Prove or disprove Sherwin’s conjecture
Background context: The conjecture $m_{jj} \leq m_{ji} + m_{ij}$ is about the mean recurrence time in a Markov chain. This involves understanding the concept of hitting times and return times.

:p Prove or disprove Sherwin’s conjecture, where $m_{jj}$ denotes the mean number of steps to get from state i to j given we’re in state j.

??x
To prove or disprove Sherwin's conjecture $m_{jj} \leq m_{ji} + m_{ij}$, let's analyze it step-by-step:

1. **Mean Recurrence Time**: The mean recurrence time $m_{jj}$ is the expected number of steps to return to state j given that we are currently in state j.

2. **Hitting Times**: 
   - Let $h_{ij}$ be the hitting time from state i to state j, which is the expected number of steps to go from i to j.
   - From any state j, the chain can either return immediately (if it's already at j), or move to another state and eventually return.

3. **Analysis**:
   - Consider the path from j back to j directly: This takes $m_{jj}$ steps on average.
   - Alternatively, consider moving from j to i and then from i back to j: The first part of this journey is $h_{ji}$, and the second part is $ m_{ij}$.

By linearity of expectation:
$$m_{jj} = \mathbb{E}[T_{return}]$$where $ T_{return}$can be either directly returning (which takes $ m_{jj}$ steps on average) or by going through state i first.

Thus, the mean time to return from j to j is at most the sum of the mean time to go from j to i and then from i back to j:
$$m_{jj} \leq h_{ji} + m_{ij}$$

Since $h_{ji}$ can be 0 (if state i is not accessible), we have:
$$m_{jj} \leq m_{ji} + m_{ij}$$

Therefore, Sherwin's conjecture is true.

```java
public class MarkovChainAnalysis {
    // Assume T is the transition matrix
    double m_jj; // Mean recurrence time for state j
    double h_ji; // Hitting time from i to j

    public boolean checkConjecture() {
        return m_jj <= h_ji + m_ij;
    }
}
```
x??

---

#### Time-average fraction of time spent in each state
Background context: This problem involves finding the long-term (time-averaged) probability of being in each state, which is given by the steady-state probabilities. The system tends to revert to a specific position over time.

:p Given the bidirectional chain for pricing, what does your position tend to revert to?

??x
The position tends to revert to a state where the expected number of shares owned or owed is zero due to the balancing nature of the Markov process. Since you set prices such that with probability p (for being long) and q = 1 - p (for being short), your position will fluctuate but in the long run, it tends towards zero.

The answer: The bidirectional chain for pricing ensures a balanced trade-off between buying and selling. Over time, the system will revert to a state where you have no net shares, i.e., position 0.

```java
public class PricingModel {
    double p; // Probability of next trade being a buy when long

    public int reversionPosition() {
        return 0;
    }
}
```
x??

---

#### Expected (absolute value) size of the position
Background context: This involves calculating the expected absolute value of the position over time, which can be derived from the properties of the Markov chain.

:p What is the expected (absolute value) size of your position?

??x
The expected (absolute value) size of the position is determined by the probabilities and the structure of the bidirectional chain. Since you are long with probability p and short with probability q = 1 - p, the absolute size will be influenced by these probabilities.

Given that the chain has equal chances to move up or down symmetrically around zero, the expected absolute value can be derived as follows:
$$

E[|X_n|] = p \cdot |X_{n+1} + 1| + (1 - p) \cdot |X_{n+1} - 1|$$

For simplicity, in steady-state, the expected absolute size of your position is:
$$

E[|X|] = \frac{2p(1-p)}{p + q} = 1$$

The answer: The expected (absolute value) size of your position over time is 1.

```java
public class ExpectedPositionSize {
    double p; // Probability of next trade being a buy when long

    public double expectedAbsolutePosition() {
        return 2 * p * (1 - p);
    }
}
```
x??

---

#### Expected number of minutes until k consecutive failures
Background context: This problem involves understanding the concept of run length in probability theory, where we are interested in the expected time to observe a sequence of $k$ consecutive failures.

:p Derive the Markov chain for this problem and find the expected number of minutes until there are k consecutive failures.

??x
To derive the Markov chain for this problem, we can define states based on the number of consecutive failures. Let state $S_i $ represent having exactly$i $ consecutive failures (where$0 \leq i < k $), and state $ F_k $represent having$ k$ consecutive failures.

The transition probabilities are as follows:
- From state $S_0 $, a failure occurs with probability p, leading to state $ S_1$.
- From state $S_i $ where$1 \leq i < k $, both a success and a failure can occur: a success leads back to $ S_0 $, and a failure leads to$ S_{i+1}$.
- State $F_k $ is an absorbing state, meaning once we reach$k$ consecutive failures, the process stops.

The expected number of minutes until k consecutive failures in state j (where j = 0) can be represented by the following equations:
$$E_0 = 1 + pE_0 + qE_1$$
$$

E_i = 1 + pE_{i-1} + qE_0 \quad \text{for } i=1,2,\ldots,k-1$$

Solving these equations step-by-step:
1.$E_k = 0$(absorbing state)
2.$E_{k-1} = 1 + p(0) + qE_0 = 1 + qE_0$3. Continue this for each i down to 0.

Finally, solving the system of equations gives us:
$$E_0 = \frac{1}{p^k} - 1$$

The answer: The expected number of minutes until k consecutive failures is $\frac{1}{p^k} - 1$.

```java
public class ConsecutiveFailures {
    double p; // Probability of a failure

    public int expectedMinutesForKConsecutiveFailures(int k) {
        return (int)(1.0 / Math.pow(p, k) - 1);
    }
}
```
x??

---

#### Walks on Undirected Weighted Graphs
Background context: This problem involves understanding how to represent and analyze walks in undirected weighted graphs, where the weights on edges indicate the strength or likelihood of moving between nodes.

:p Consider any undirected graph with weights. Explain how to find a walk from node i to j based on the edge weights.

??x
In an undirected weighted graph, each edge (i, j) has a weight $w_{ij}$, which can be thought of as the probability or strength of moving from node i to node j.

To find a walk from node i to node j:
1. **Identify Paths**: List all possible paths between nodes i and j.
2. **Calculate Path Weights**: For each path, calculate the product of edge weights along that path.
3. **Select Optimal Path**: Choose the path with the highest combined weight (or probability).

The answer: To find a walk from node i to node j based on the edge weights, one can use algorithms like Dijkstra’s or Bellman-Ford for finding shortest paths in weighted graphs. However, if we are interested in walks rather than just the shortest path, we would typically use a method that considers all possible walks and their combined weights.

```java
public class WeightedGraphWalks {
    private Map<Integer, List<int[]>> graph; // Adjacency list representation

    public double findOptimalPathWeight(int i, int j) {
        Queue<int[]> queue = new PriorityQueue<>((a, b) -> Double.compare(b[1], a[1])); // Max-heap based on weight
        Set<Integer> visited = new HashSet<>();
        
        // Initialize the graph traversal with starting node and path weight 0
        queue.offer(new int[]{i, 0.0});
        
        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            if (visited.contains(current[0])) continue;
            
            visited.add(current[0]);
            for (int[] neighbor : graph.getOrDefault(current[0], new ArrayList<>())) {
                int nextNode = neighbor[0];
                double weight = neighbor[1];
                
                // Calculate the combined path weight
                queue.offer(new int[]{nextNode, current[1] + weight});
            }
        }
        
        return -1; // If no path found, return -1 or an appropriate value
    }
}
```
x??

---

#### Long-run Proportion of Time in State i (Ergodicity Theory)
Background context: In a stochastic process, particularly one that is ergodic, the long-run proportion of time spent in each state can be determined using stationary equations. However, for this problem, we'll focus on the time-reversibility equations to find the long-run proportion of time.

:p What is the long-run proportion of time that a particle is in state i?
??x
The long-run proportion of time that the particle is in state $i $ can be found by solving the time-reversibility equation. In an ergodic Markov chain, this proportion is given by the stationary distribution$\pi_i$, which satisfies:

$$\pi_j = \sum_{i} \pi_i P_{ij}$$where $ P_{ij}$is the transition probability from state $ i$to state $ j $. The stationary distribution can be guessed and verified through normalization, ensuring that $\sum_{i} \pi_i = 1$.

??x
The answer with detailed explanations.
To find the long-run proportion of time a particle spends in state $i $, we need to solve for the stationary distribution $\pi_i$ using the equation:
$$\pi_j = \sum_{i} \pi_i P_{ij}$$

Here,$P_{ij}$ is defined as:
$$P_{ij} = \frac{w_{ij}}{\sum_{j} w_{ij}}$$

This represents the probability of transitioning from state $i $ to state$j $. By guessing a solution for$\pi_i$ and verifying it satisfies the above equation, we can determine the long-run proportion.

```java
public class StationaryDistribution {
    private double[] pi; // stationary distribution

    public void computeStationary(double[][] transitionMatrix) {
        int n = transitionMatrix.length;
        pi = new double[n];

        // Guess a solution for π (uniform distribution as an initial guess)
        Arrays.fill(pi, 1.0 / n);

        boolean converged = false;

        while (!converged) {
            double[] nextPi = new double[n];
            for (int i = 0; i < n; i++) {
                nextPi[i] = 0;
                for (int j = 0; j < n; j++) {
                    nextPi[i] += pi[j] * transitionMatrix[j][i];
                }
            }

            converged = true;

            for (int i = 0; i < n && converged; i++) {
                if (Math.abs(pi[i] - nextPi[i]) > 1e-6) {
                    converged = false;
                }
                pi[i] = nextPi[i];
            }
        }
    }
}
```

The code iteratively updates the distribution until it converges. The final values of $\pi_i $ represent the long-run proportion of time the particle spends in state$i$.

x??

---

#### Irreducibility and Aperiodicity for Different Chess Pieces
Background context: We need to analyze the Markov chains corresponding to the movement of different chess pieces on an 8×8 board. We'll determine if these processes are irreducible (a single component) and aperiodic.

:p Is the Markov chain for a king's random moves on an 8×8 board irreducible?
??x
The Markov chain for a king's random moves on an 8×8 board is irreducible. This is because, from any state (any square), the king can potentially move to any other state within a finite number of steps. The kings' movement allows it to reach every square from any starting position in a finite number of moves.

??x
The answer with detailed explanations.
Yes, the Markov chain for a king's random moves on an 8×8 board is irreducible because:

- From any given state (square), the king can move to any adjacent square (including diagonally) in one step.
- There are no absorbing states or isolated components where the king cannot move from.

This means that the graph of all possible states is fully connected, making it an irreducible chain. 

:p Is the Markov chain for a bishop's random moves on an 8×8 board irreducible?
??x
The Markov chain for a bishop's random moves on an 8×8 board is also irreducible. This is because, from any state (any square), the bishop can potentially move to any other state that shares the same color as its current square within a finite number of steps.

:p Is the Markov chain for a knight's random moves on an 8×8 board irreducible?
??x
The Markov chain for a knight's random moves on an 8×8 board is also irreducible. This is because, from any state (any square), the knight can potentially move to any other state within a finite number of steps.

:p Is the corresponding Markov chain for a king's random moves on an 8×8 board aperiodic?
??x
The Markov chain for a king's random moves on an 8×8 board is aperiodic. This means that, from any given state, it is possible to return to that state in varying numbers of steps.

:p Is the corresponding Markov chain for a bishop's random moves on an 8×8 board aperiodic?
??x
The Markov chain for a bishop's random moves on an 8×8 board is also aperiodic. This means that, from any given state (any square of the same color), it is possible to return to that state in varying numbers of steps.

:p Is the corresponding Markov chain for a knight's random moves on an 8×8 board aperiodic?
??x
The Markov chain for a knight's random moves on an 8×8 board is also aperiodic. This means that, from any given state (any square), it is possible to return to that state in varying numbers of steps.

:x??

---

#### Expected Time for King to Return to Corner Using Time-Reversibility
Background context: We need to calculate the expected time for the king to return to its starting corner using the concept of time-reversibility, which simplifies the calculation significantly.

:p Calculate the expected time for the king to return to the corner.
??x
Using time-reversibility, we can simplify the calculation of the expected return time for the king. The key insight is that the expected return time from any state in a symmetric and irreducible Markov chain can be calculated using the hitting times.

The answer with detailed explanations.
Given the symmetry of the 8×8 board and the random nature of the moves, we can leverage the concept of time-reversibility to simplify the calculation. The expected return time $E_T$ for the king to return to its starting corner can be found by considering the properties of the Markov chain:
$$E_T = \sum_{i=1}^{n} P_i T_i$$where $ P_i $ is the probability of being in state $ i $ and $ T_i$ is the hitting time from that state. For a symmetric random walk on an 8×8 board, this simplifies significantly.

Using the properties of irreducibility and symmetry:
$$E_T = 64$$

This means the expected return time for the king to return to its starting corner is 64 steps.

:x??

---

#### Positive Recurrence of Threshold Queue
Background context: We need to argue that the Markov chain representing a threshold queue, where $T=3$, is both aperiodic and positive recurrent. This involves understanding the behavior of the states in the chain over time.

:p Argue that the Markov chain for the threshold queue is aperiodic.
??x
The Markov chain for the threshold queue with $T=3$ is aperiodic because there are no cycles with odd periods. In other words, from any state, it is possible to return to that state in both even and odd numbers of steps.

:p Argue that the Markov chain for the threshold queue is positive recurrent.
??x
The Markov chain for the threshold queue with $T=3$ is positive recurrent because every state has a finite expected return time. This means that, starting from any state, the expected number of steps to return to that state is finite.

:x??

---

#### Symmetric Random Walk and Mean Return Time
Background context: We need to prove that for a symmetric random walk on an 8×8 board, the mean time between visits to state 0 is infinite. This involves understanding the combinatorial proof using Catalan numbers.

:p What is $m_{00}$ in the context of the symmetric random walk?
??x
In the context of the symmetric random walk on an 8×8 board,$m_{00}$, which denotes the mean time between visits to state 0, is infinite. This means that while we are certain to return to state 0 infinitely many times, the expected number of steps between each visit is infinite.

:p How can we use Catalan numbers to prove $m_{00} = \infty$?
??x
To prove $m_{00} = \infty$, we need to consider the properties of paths in a symmetric random walk. The key insight here is that while every path must return to state 0 infinitely many times, the expected time between visits can be shown to diverge.

Using Catalan numbers, which count certain types of lattice paths, we can show that the number of ways to reach and return from state 0 grows faster than any polynomial. This implies that the mean time $m_{00}$ is infinite.

:x??

#### Characterizing Middle Steps in a Random Walk
Background context: We are dealing with a random walk where T00 = n, and we need to understand the behavior of the middle $n-2 $ steps. The Catalan number$C(k)$ represents the number of valid sequences (strings) of length $2k$ containing $k$0's and $ k$1's such that no prefix contains more 0's than 1's. This can be useful in characterizing certain paths or events in a random walk.

:p How does the Catalan number help characterize the middle $n-2$ steps in a random walk where T00 = n?
??x
The Catalan number helps by providing a combinatorial framework to count valid sequences that meet specific conditions. For instance, if we consider the path of the random walk from state 0 to state 0 after $n $ steps, the middle$n-2$ steps can be modeled as a sequence where each step is either a move right (R) or left (L), and no prefix of this sequence has more L's than R's. The number of such valid sequences is given by a Catalan number.

For example, if we have 10 steps in total (T00 = 10), the middle 8 steps can be characterized using $C(4)$ because each step sequence must balance between right and left moves without ever going below zero. The formula for the Catalan number is:
$$C(k) = \frac{1}{k+1} \binom{2k}{k}$$

This helps in understanding that the middle steps are constrained by the same combinatorial rules as the overall path, ensuring no prefix has more 0's (left moves) than 1's (right moves).

```java
// Pseudo-code to compute Catalan number for k = 4
public class CatalanNumber {
    public static long catalan(int n) {
        if (n <= 1) return 1;
        
        long res = 0;
        for (int i = 0; i < n; i++) {
            res += catalan(i) * catalan(n - 1 - i);
        }
        return res / (n + 1); // Dividing by (n+1) to avoid overflow
    }
}
```
x??

---

#### Expressing $P\{T_{0,0} = n\}$ Using Catalan Number

Background context: The probability that a random walk returns to state 0 after exactly $n $ steps can be expressed using the Catalan number. Given that$P\{T_{0,0} = n\} = P\{T_{0,0} = n \mid \text{First step is right}\}$, we need to use the properties of Catalan numbers to derive this expression.

:p How can $P\{T_{0,0} = n\}$ be expressed in terms of a Catalan number?
??x
The probability $P\{T_{0,0} = n\}$ for a random walk returning to state 0 after exactly $n$ steps can be expressed using the Catalan number. Specifically, if $n$ is even (since it must involve an equal number of left and right moves), then:
$$P\{T_{0,0} = n\} = \frac{C(n/2)}{2^{n}}$$

This formula comes from understanding that for the walk to return to 0 after $n $ steps, exactly half of these steps must be rights (1's) and the other half lefts (0's), forming a valid sequence as per the definition of Catalan numbers. The factor of$2^n $ accounts for all possible sequences of length$n$.

For example, if $n = 6$:
$$P\{T_{0,0} = 6\} = \frac{C(3)}{2^6} = \frac{\frac{1}{4} \binom{6}{3}}{64} = \frac{5}{128}$$```java
// Pseudo-code to compute the probability using Catalan number
public class RandomWalkProbability {
    public static double randomWalkProb(int n) {
        if (n % 2 != 0) return 0; // Only even steps are possible
        int k = n / 2;
        long catalanNum = catalan(k);
        return (double) catalanNum / Math.pow(2, n);
    }

    public static long catalan(int n) {
        if (n <= 1) return 1;

        long res = 0;
        for (int i = 0; i < n; i++) {
            res += catalan(i) * catalan(n - 1 - i);
        }
        return res / (n + 1); // Dividing by (n+1) to avoid overflow
    }
}
```
x??

---

#### Lower Bound on $P\{T_{0,0} = n\}$ Background context: Using the lower bound derived from the Catalan number and Lemma 9.18, we can show that $m_{0,0} = \infty$. The key idea is to use the fact that the probability of returning after exactly $ n$steps decreases as $ n$ increases.

:p How do you derive a lower bound on $P\{T_{0,0} = n\}$ using the Catalan number and Lemma 9.18?
??x
To derive a lower bound on $P\{T_{0,0} = n\}$, we use the properties of Catalan numbers to understand that:

$$P\{T_{0,0} = n\} \geq \frac{C(n/2)}{2^n}$$where $ C(k) = \frac{1}{k+1} \binom{2k}{k}$. By Lemma 9.18, which likely states that the Catalan number has a lower bound in terms of exponential decay or factorial growth, we can infer that:

$$C(n/2) \geq \frac{(2^{n/2})^2}{4(n/2) + 1} = \frac{4^{n/2}}{2n + 1}$$

Thus,$$

P\{T_{0,0} = n\} \geq \frac{\frac{4^{n/2}}{2n + 1}}{2^n} = \frac{1}{(2n+1)2^{n/2 - n}} = \frac{1}{(2n+1)\sqrt{2^n}}$$

This lower bound shows that the probability of returning to state 0 after exactly $n $ steps is always positive and decreases as$n $ increases. This implies that the expected time between visits,$m_{0,0}$, must be infinite because a finite mean would contradict this lower bound.

```java
// Pseudo-code for calculating the lower bound of probability
public class LowerBoundProbability {
    public static double lowerBoundProb(int n) {
        if (n % 2 != 0) return 0; // Only even steps are possible
        int k = n / 2;
        long catalanNum = catalan(k);
        return (double) catalanNum / Math.pow(2, n);
    }

    public static long catalan(int n) {
        if (n <= 1) return 1;

        long res = 0;
        for (int i = 0; i < n; i++) {
            res += catalan(i) * catalan(n - 1 - i);
        }
        return res / (n + 1); // Dividing by (n+1) to avoid overflow
    }
}
```
x??

---

#### Stopping Times and Wald's Equation

Background context: A stopping time $N$ is a random variable that depends on the past but not on future observations. Wald's equation relates the expected value of a sum to the expected number of terms in the sum, under certain conditions.

:p Is the total time until we see 5 heads (or 5 consecutive heads) a stopping time in a sequence of coin flips?
??x
Yes, the total time until we see 5 heads is a stopping time. This is because it only depends on past observations and does not depend on future outcomes. For example, if we have flipped some number of coins and have seen 4 heads so far, then the next head will complete our count, making the stopping time well-defined.

However, the time until we see 5 consecutive heads is not a stopping time because it depends on future outcomes. We cannot determine when this event happens based solely on past observations; we need to know all future flips as well.

```java
// Pseudo-code for checking if a sequence has reached 5 heads
public class HeadsStoppingTime {
    public static boolean checkFor5Heads(List<String> coinFlips) {
        int count = 0;
        for (String flip : coinFlips) {
            if ("H".equals(flip)) count++;
            else count = 0;
            if (count == 5) return true;
        }
        return false;
    }
}
```
x??

---

#### Gambler's Stopping Time in a Game

Background context: In a game where a gambler starts with zero dollars and has an equal chance of winning or losing one dollar per round, the stopping time is defined as the number of rounds until the gambler is 2 dollars ahead.

:p Write a mathematical expression for the stopping time $N$ in terms of a sum.
??x
The stopping time $N$, which represents the number of games until the gambler reaches 2 dollars, can be expressed as:

$$N = \sum_{i=1}^{n} X_i$$where $ X_i $ is an indicator random variable that takes value 1 if the gambler wins the $ i$-th game and 0 otherwise. The sum essentially counts the number of games until the total winnings reach 2 dollars.

```java
// Pseudo-code for calculating the stopping time N
public class GamblerStoppingTime {
    public static int calculateStoppingTime(List<String> outcomes) {
        int wins = 0;
        int rounds = 0;
        for (String outcome : outcomes) {
            if ("W".equals(outcome)) wins++;
            else wins--;
            rounds++;
            if (wins == 2) return rounds; // Stopping condition
        }
        return -1; // Gambler never reaches 2 dollars
    }
}
```
x??

---

#### Expectation of Sum Involving i.i.d. Random Variables

Background context: Given $X_i $ as independent and identically distributed random variables, we need to understand the expectation of their sum up to a positive integer-valued stopping time$Y$.

:p What do we know about $E\left[\sum_{i=1}^{Y} X_i \right]$?
??x
For i.i.d. random variables $X_i $ and a stopping time$Y $ that is independent of the$ X_i $'s, the expectation of their sum up to $ Y$can be expressed using Wald's equation:
$$E\left[\sum_{i=1}^{Y} X_i \right] = E[Y]E[X_1]$$

This result follows from the linearity of expectation and the fact that the sum is split into two expectations, one for the number of terms $Y $ and another for the average value of each term$X_1$.

```java
// Pseudo-code to illustrate Wald's equation
public class WaldEquation {
    public static double waldEquation(int EY, double EX) {
        return EY * EX;
    }
}
```
x??

---

#### Proving $m_{11} = \infty$ Using Wald's Equation

Background context: The mean time between visits to state 1 in a symmetric random walk is denoted as $m_{11}$. To show that $ m_{11} = \infty$, we use the properties of stopping times and Wald's equation.

:p Prove that $m_{11} > 0.5m_{01}$ and explain why it suffices to show $m_{01} = \infty$ to prove $m_{11} = \infty$.
??x
To prove that $m_{11} > 0.5m_{01}$, we need to understand the relationship between the mean times in a symmetric random walk.

Given:
- $T_{01}$ is the first hitting time from state 0 to state 1.
- $T_{11}$ is the return time to state 1 after starting at state 1.

In a symmetric random walk, we know that:
$$m_{11} = E[T_{11}]$$and$$m_{01} = E[T_{01}]$$

Using Wald's equation and properties of stopping times, we can establish the relationship between these two expectations. Specifically, because $T_{11}$ is at least as large as half of $T_{01}$ plus some additional time (due to symmetry), it follows that:
$$m_{11} > 0.5m_{01}$$

To show that $m_{11} = \infty $, it suffices to show that$ m_{01} = \infty$. This is because if the time to go from state 0 to state 1 is infinite, then the return time from any other state must also be infinite due to symmetry and the nature of the random walk.

```java
// Pseudo-code to illustrate the relationship between m11 and m01
public class RandomWalkTime {
    public static double m11GreaterThanHalfM01(double m01) {
        return 0.5 * m01; // Simplified representation
    }
}
```
x??

---

#### Expected Value of a Sum with i.i.d. Variables

Background context: Given $X_i $ as i.i.d. random variables and a stopping time$N$, Wald's equation helps us find the expected value of their sum.

:p Use Wald's equation to relate the expected value of a sum up to a stopping time to the expected number of terms and the expected value of each term.
??x
Wald's equation states that if $X_i $ are i.i.d. random variables with finite expectation$E[X_i] = \mu $, and$ N$is an integrable stopping time, then:
$$E\left[\sum_{i=1}^{N} X_i \right] = E[N] \cdot E[X_1]$$

This equation holds because the sum of the random variables up to a stopping time can be split into two parts: the expected number of terms $E[N]$ and the average value of each term $E[X_1]$.

```java
// Pseudo-code for applying Wald's equation
public class WaldEquationApplication {
    public static double waldSumExpectation(int EY, double EX) {
        return EY * EX;
    }
}
```
x??

#### Recurrence and Transience of Markov Chain
In a Markov chain, states can be classified as either recurrent (a state that is visited infinitely often with probability 1) or transient (a state that is not visited infinitely often with positive probability). The classification depends on the expected number of visits to each state. If the expected number of returns to any state $i$ is infinite, then the chain is recurrent; otherwise, it is transient.

:p For what values of $p$ is the Markov chain recurrent and for which is it transient?
??x
For a Markov chain with states 0, 1, 2, etc., where $q = 1 - p$, we can determine recurrence or transience by examining the expected number of visits to each state. Specifically:

- The chain will be recurrent if the probability of returning to any state is 1.
- The chain will be transient if there is a positive probability that the process will never return to some states.

For this Markov chain, if $p > \frac{1}{2}$, then state 0 is recurrent. If $ p < \frac{1}{2}$, state 0 is transient.
??x
To understand why, consider the expected number of visits. For a simple random walk where each step has an equal probability of moving left or right (or in this case, up or down based on $p $ and$q $), if$ p > \frac{1}{2}$, the process is more likely to return to 0 than not. Conversely, for $ p < \frac{1}{2}$, the process has a positive probability of never returning.

For example, in a simple random walk:
```java
public class RandomWalk {
    public static void simulate(int steps) {
        int position = 0;
        for (int i = 0; i < steps; i++) {
            if (Math.random() > 0.5) { // 50% chance to move left or right
                position--;
            } else {
                position++;
            }
        }
        System.out.println("Final Position: " + position);
    }
}
```
:p What is the probability $f_0$ of ever returning to state 0 if started at state 0?
??x
If the chain is transient, the probability $f_0 = P(\text{Ever return to state 0 given start there})$ can be derived using various methods. For a simple random walk where $p < q$, the probability of ever returning to state 0 is:
$$f_0 = \begin{cases} 
1 & \text{if } p > \frac{1}{2}, \\
\frac{1 - (q/p)^0}{1 - (q/p)} & \text{if } p < q.
\end{cases}$$

For $p < q$ and starting at state 0, the probability of ever returning to state 0 is:
$$f_0 = \frac{1}{1 + \left(\frac{q}{p}\right)^n}, \text{ as } n \to \infty.$$??x
This can be understood by considering the long-term behavior of the random walk. If $p < q$, then the process is more likely to move away from state 0, making it less probable that it will ever return.
??x

---

#### Time to Return to State 0 in Transient Markov Chain
For a transient Markov chain where $p < q $, we can derive the expected time $ E[T_{00}]$ to go from state 0 back to state 0.

:p What is the expected time $E[T_{00}]$ to return to state 0 if starting at state 0?
??x
The expected time to return to state 0, given that the chain is transient and $p < q$, can be derived using the properties of the Markov chain. Specifically:
$$E[T_{00}] = \frac{1}{f_0},$$where $ f_0$ is the probability of ever returning to state 0.

Given that for a random walk with $p < q$:
$$f_0 = \frac{1 - (q/p)^n}{1 - (q/p)}, \text{ as } n \to \infty,$$we can find:
$$

E[T_{00}] = \frac{1 - (q/p)}{(q/p) - 1}.$$

For example, if $p = 0.4 $ and$q = 0.6$:
```java
public class ExpectedReturnTime {
    public static double calculateExpectedReturnTime(double p, double q) {
        return (1 - (q / p)) / ((q / p) - 1);
    }
}
```
:p What does this tell us about $\pi_0 = \lim_{n \to \infty} P_n(0,0)$?
??x
This tells us that if the chain is transient, then:
$$\pi_0 = \lim_{n \to \infty} P_n(0,0) = 0.$$

The stationary probability $\pi_0 $ of being in state 0 as$n \to \infty$ is zero because the process will eventually leave and not return to state 0 with positive probability.
??x
This can be understood by noting that for a transient state, the chain will spend only a finite amount of time in that state before moving to another. Therefore, its long-term stationary probability is zero.

---

#### Residue Classes in Periodic Chains
Residue classes partition states into groups based on their periodicity and connectivity. For an irreducible DTMC with period $d$, we can define residue classes as follows:

- State $i$ has residue class 0.
- For any other state $j $, its residue class is the length of the shortest path from $ i $to$ j $modulo$ d$.

:p Show that the notion of residue classes is well-defined by proving that the lengths of any two paths from $i \to j $ are equivalent modulo$d$.
??x
To show that the notion of residue classes is well-defined, we need to prove that for any pair of states $i $ and$j $, all shortest paths between them have lengths that are congruent modulo$ d$.

Given an irreducible DTMC with period $d$:
- There exists a path from state $i $ to state$j $ that is equivalent to the length of its residue class modulo$d$.
- Any two such paths will have lengths that differ by a multiple of $d $, meaning their lengths are congruent modulo $ d$.

Thus, the residue classes are well-defined.
??x
This can be proven using the properties of irreducibility and periodicity. Since the chain is irreducible, there exists at least one path between any two states, and since it has period $d $, all such paths will have lengths that differ by multiples of $ d$.
??x

---

#### Positive Recurrence for Finite-State DTMCs
For a finite-state, irreducible DTMC, we can prove the theorem that all states are positive recurrent using class properties:

- Null recurrence is a class property.
- Positive recurrence is a class property.

:p Prove that in a finite state, irreducible DTMC, all states are positive recurrent.
??x
To prove this, we use the following steps:
1. **Class Properties**: If $i $ is null recurrent and communicates with$j $, then$ j $must also be null recurrent. Similarly, if$ i $is positive recurrent and communicates with$ j $, then$ j$ must also be positive recurrent.
2. **Finite States**: In a finite state space, every state can communicate with itself (self-loops).
3. **Irreducibility**: The chain is irreducible, meaning there is a path from any state to any other state.

Given these properties:
- If we start at any state $i$, it must either be positive recurrent or null recurrent.
- Since the chain is finite and irreducible, if one state is null recurrent, all states would have to be null recurrent due to communication.
- However, in a finite system, this leads to contradiction because there are only finitely many states, and infinite visits imply recurrence.

Therefore, all states must be positive recurrent.
??x
This can be understood by recognizing that in a finite state space, the chain cannot have cycles of null recurrent states without violating the irreducibility or finiteness constraints. Thus, every state must be positive recurrent to ensure finite expected return times for all states.

---

#### Time Reversibility and Finite-State DTMCs
For a finite-state, irreducible DTMC with $p < q$, we can determine if the chain is time-reversible by examining its stationary probabilities.

:p Is the chain time reversible?
??x
To determine if the Markov chain is time-reversible, we need to check if the detailed balance equations hold:
$$\pi_i P_{ij} = \pi_j P_{ji},$$for all states $ i $ and $ j $, where$\pi_i$ are the stationary probabilities.

Given that $p < q$, the chain is transient, implying it does not satisfy detailed balance due to the unbalanced transition probabilities. Thus:
- The chain is **not** time-reversible.
??x
This can be understood by noting that for a Markov chain to be time-reversible, the product of the stationary probability and the transition probability must be equal in both directions between all states. Given $p < q$, this condition does not hold, indicating non-time-reversibility.

---

---

#### Importance of Backlinks Not Being Equal
Background context: In Google’s PageRank algorithm, backlinks are used to determine the importance of a web page. However, not all backlinks should be considered equally important.

:p Why would counting all backlinks equally not be a good measure of a page's importance?
??x
Counting all backlinks equally does not account for the quality or significance of each link. A link from a popular and authoritative site (e.g., Yahoo) is more valuable than a link from an obscure personal blog.
x??

---

#### Trickability of Citation Counting System
Background context: The citation counting system can be manipulated by creating many dummy pages that all point to one target page, thereby inflating its importance.

:p Why is the citation counting scheme easy to fool?
??x
The system can be easily fooled because you can create a clique of dummy pages, each pointing to your main page. This increases the number of backlinks but does not necessarily increase the overall relevance or quality.
x??

---

#### Recursive Definition of Page Rank
Background context: Google's solution was to define page rank recursively, stating that a page has high rank if the sum of the ranks of its backlinks is high.

:p How does the recursive definition help us figure out the rank of a page?
??x
The recursive definition translates into solving balance equations. Each state (page) πj is equal to the average of the states pointing into it, i.e., πj = ∑(πiPij)/n. This means that for a page to have high limiting probability, its backlinks must also have high probabilities.
x??

---

#### Creation of DTMC for Web Pages
Background context: Google uses a Markov chain to model web surfing behavior, where each state represents a web page and transitions represent clicking from one page to another.

:p What are the steps in creating a DTMC transition diagram for web pages?
??x
1. Create states corresponding to each web page.
2. Draw arrows between states if there is a link from one page to another.
3. Assign probabilities based on the number of outgoing links: if page i has k outgoing links, then each probability is 1/k.

Example:
```java
public class PageRankModel {
    private Map<String, List<String>> linkGraph;
    
    public void buildTransitionDiagram(Map<String, List<String>> linkGraph) {
        this.linkGraph = linkGraph;
        
        for (String page : linkGraph.keySet()) {
            int numOutLinks = linkGraph.get(page).size();
            
            // Assign transition probabilities
            for (String destPage : linkGraph.get(page)) {
                setTransitionProbability(page, destPage, 1.0 / numOutLinks);
            }
        }
    }
    
    private void setTransitionProbability(String from, String to, double prob) {
        // Implement logic to update the transition probability matrix
    }
}
```
x??

---

#### Handling Dead Ends and Spider Traps
Background context: In real web graphs, dead ends or spider traps can cause issues where limiting probabilities do not converge properly.

:p Why is a dead end or spider trap problematic for PageRank?
??x
A dead end (no outgoing links) or spider trap (self-loop with no other exits) causes the Markov chain to get stuck. The solution does not match our intuitive understanding of web surfing, where some pages are still important despite having limited connectivity.
x??

---

#### Google’s Solution for Dead Ends and Spider Traps
Background context: To handle dead ends and spider traps, Google introduced a “tax” on each page that redistributes importance to other pages.

:p How does the 30 percent tax work in the DTMC?
??x
Each existing transition is multiplied by 70%. Additionally, for each state s in an n-state chain, we add transitions of weight 30%/n from state s to every other state. This ensures that no page gets trapped and all states have a chance to receive importance.

Example:
```java
public class PageRankModel {
    private Map<String, List<String>> linkGraph;
    
    public void applyTax(double taxRate) {
        for (String page : linkGraph.keySet()) {
            int numOutLinks = linkGraph.get(page).size();
            
            // Apply 70% reduction to existing transitions
            for (String destPage : linkGraph.get(page)) {
                setTransitionProbability(page, destPage, (1 - taxRate) / numOutLinks);
            }
            
            // Add tax transitions
            double taxProb = taxRate / numOutLinks;
            for (String destPage : linkGraph.keySet()) {
                setTransitionProbability(page, destPage, taxProb); // including self-links
            }
        }
    }
}
```
x??

---

