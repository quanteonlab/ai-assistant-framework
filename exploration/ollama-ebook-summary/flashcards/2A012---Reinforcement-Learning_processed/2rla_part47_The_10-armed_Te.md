# Flashcards: 2A012---Reinforcement-Learning_processed (Part 47)

**Starting Chapter:** The 10-armed Testbed

---

#### Sample-Average Action-Value Estimation Method
Background context explaining how action values are estimated using sample averages. The formula is provided to show the process of averaging rewards for an action.

:p What is the sample-average method used for estimating action values?
??x
The sample-average method estimates the true value \( q^*(a) \) of an action by taking the average of all the rewards received when that action was chosen up to time step \( t-1 \).

\[
Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i A_i=a}{\sum_{i=1}^{t-1} I(A_i=a)}
\]

Where:
- \( R_i \): The reward received at time step \( i \)
- \( A_i \): The action taken at time step \( i \)
- \( I(\cdot) \): An indicator function that returns 1 if the condition is true, and 0 otherwise

By the law of large numbers, as the number of samples increases, \( Q_t(a) \) converges to the true value \( q^*(a) \).
x??

---

#### Greedy Action Selection Method
Background context on how actions are selected using the greedy approach. The formula for selecting an action is given.

:p How does the greedy action selection method work?
??x
The greedy action selection method chooses one of the actions with the highest estimated value \( Q_t(a) \). If there are multiple such actions, a tie-breaking rule (e.g., random selection among them) is used. The formal definition is:

\[
A_t = \arg\max_a Q_t(a)
\]

Where:
- \( A_t \): Action selected at time step \( t \)
x??

---

#### Near-Greedy or \(\epsilon\)-Greedy Action Selection Method
Explanation of the \(\epsilon\)-greedy method, which balances exploration and exploitation by selecting a greedy action most of the time but occasionally choosing random actions.

:p What is an \(\epsilon\)-greedy action selection method?
??x
The \(\epsilon\)-greedy method selects one of the actions with the highest estimated value \( Q_t(a) \) with probability \( 1 - \epsilon \). With a small probability \( \epsilon \), it randomly chooses among all actions. The formal definition is:

\[
A_t = 
\begin{cases} 
\arg\max_a Q_t(a) & \text{with probability } 1-\epsilon \\
a & \text{uniformly at random from the set of actions otherwise}
\end{cases}
\]

Where:
- \( A_t \): Action selected at time step \( t \)
x??

---

#### 10-Armed Testbed
Background context on the testbed used to compare different action-value methods. It is described as a suite of randomly generated k-armed bandit problems with k=10.

:p What is the 10-armed testbed?
??x
The 10-armed testbed is a set of 2000 randomly generated 10-armed bandit problems. In each problem, action values \( q^*(a) \) are selected from a normal distribution with mean zero and unit variance. The actual rewards for each action are then sampled from a normal distribution centered around the true value \( q^*(a) \). This setup is used to numerically compare different learning methods over 1000 time steps, repeating this process for 2000 independent runs.

For example:
- Each run involves selecting actions and recording rewards.
- The performance of a method can be measured as the average reward per step after 1000 steps.
x??

---

#### Performance Comparison on 10-Armed Testbed
Comparison between greedy methods and \(\epsilon\)-greedy methods on the 10-armed testbed. The sample-average technique is used to estimate action values.

:p How did the different methods perform on the 10-armed testbed?
??x
On the 10-armed testbed, the performance of greedy and \(\epsilon\)-greedy methods was compared using the sample-average technique for estimating action values. The results showed that while the greedy method improved faster initially, it achieved a lower long-term average reward per step (about 1) compared to the optimal possible value (around 1.55). In contrast, \(\epsilon\)-greedy methods, especially with small \(\epsilon\) like \(0.01\), performed better by ensuring that all actions were sampled sufficiently.

For instance:
- A greedy method selected the action with the highest estimated value.
- An \(\epsilon\)-greedy method selected an action with probability \( 1 - \epsilon \) or a random action otherwise.
x??

---

---

#### Exploration vs. Exploitation Dilemma
In reinforcement learning, particularly in multi-armed bandit problems, there's a trade-off between exploring different actions to gather more information and exploiting the currently known best action to maximize reward.

The greedy method always exploits the current best-known action but can get stuck with suboptimal choices if initial samples of the optimal action are disappointing. `-greedy methods balance exploration and exploitation by randomly selecting non-greedy actions a fraction of the time, which helps in finding the optimal action more reliably over many trials.
:p How does the `-greedy method address the exploration vs. exploitation dilemma?
??x
The `-greedy method addresses this dilemma by using an -probability (0 < <= 1) to randomly select non-greedy actions. This ensures that while the agent exploits known better options with probability (1-), it also explores suboptimal actions with probability . This balance helps in finding the optimal action more effectively.
x??

---

#### Long-term Performance of Exploration Strategies
The performance of `-greedy methods versus the greedy method can vary depending on the task characteristics, such as reward variance and whether the environment is stationary or nonstationary.

In a stationary environment with low noise (small reward variance), the greedy method might perform well because it quickly learns the optimal action. However, in environments with high noise or when the true values of actions change over time, `-greedy methods are generally better at exploring and eventually finding the best action.
:p Which exploration strategy is more likely to perform better in a nonstationary environment?
??x
In a nonstationary environment, -greedy methods are more likely to outperform the greedy method. This is because they continue to explore even when the optimal action has changed over time, which helps ensure that no new best actions are missed.
x??

---

#### Bandit Example: `k`-Armed Bandit Problem
Consider a k-armed bandit problem with k=4 actions (1, 2, 3, and 4). The `-greedy action selection method is applied, along with sample-average action-value estimates. Given initial estimates of Q(a) = 0 for all a.

The sequence of actions and rewards is as follows: A1= 1, R1= -1, A2= 2, R2= 1, A3= 2, R3= -2, A4= 2, R4= 2, A5= 3, R5= 0.
:p On which time steps did the `-case definitely occur?
??x
The `-case definitely occurred on actions where a non-greedy action was selected. Since the greedy action at each step is based on the current estimates of Q(a), and assuming initial estimates are all zero, the `-case would have definitely happened when selecting A2 or A3 instead of A1 in the first two steps.

For example:
- At time 2: A2= 2 (non-greedy) - must be due to -case.
- At time 4: A3= 2 (non-greedy) - must be due to -case.
x??

---

#### Cumulative Performance of Exploration Strategies
In the comparison shown in Figure 2.2, the `-greedy method with =0.1 generally performs better than the greedy method over the long run by finding the optimal action earlier and more reliably.

However, for very deterministic environments with zero reward variance, the greedy method might perform best as it can quickly find and stick to the optimal action.
:p Which method is expected to perform best in a highly deterministic environment?
??x
In a highly deterministic environment (zero reward variance), the greedy method would likely perform best. This is because once an action’s value is accurately estimated, the greedy method will always choose this action, leading to rapid convergence and sustained optimal performance.
x??

---

#### Incremental Implementation of Action-Value Methods
To efficiently update average estimates in a computationally efficient manner with constant memory, we use incremental formulas.

For example, given Qn (action-value estimate after n-1 selections) and Rn (the nth reward), the new estimate Qn+1 can be computed using:
Qn+1 = 1/n * (Rn + (n-1)*Qn)

This update rule is of a form commonly used in reinforcement learning.
:p What is the formula for updating the action-value estimates incrementally?
??x
The formula for updating the action-value estimates incrementally is:
Qn+1 = 1/n * (Rn + (n-1)*Qn)
This formula updates the average reward based on the latest observation while maintaining constant memory usage.

Here's a pseudocode example of this update rule:
```java
for (int n = 1; n <= N; n++) {
    double Rn = getRewardForAction(a); // Get the nth reward for action a
    Qn += (Rn - Qn) / n; // Update the estimate using the incremental formula
}
```
x??

---

#### Step-size Parameter and Sample Averages

Background context: The step-size parameter, denoted by \(\alpha\), is crucial for updating the average reward estimates in multi-armed bandit problems. In the stationary case, sample averages are used to estimate action values. However, in nonstationary environments, a constant step-size parameter can be employed to give more weight to recent rewards.

:p What is the role of the step-size parameter \(\alpha\) in updating the average reward estimates?
??x
The step-size parameter \(\alpha\) controls how much weight is given to new observations versus previous estimates. In nonstationary environments, a constant \(\alpha\) can ensure that more recent rewards have a higher influence on the updated estimates.
```java
// Pseudocode for updating Q(a) with a constant step-size parameter alpha
for each action a:
    Q[a] = Q[a] + alpha * (reward - Q[a])
```
x??

---

#### Exponential Recency-Weighted Average

Background context: In nonstationary environments, the use of a constant step-size parameter \(\alpha\) can result in an exponential recency-weighted average. This method adjusts the weights on past rewards so that more recent rewards have higher influence.

:p How does the constant step-size parameter \(\alpha\) affect the averaging process in a nonstationary environment?
??x
The constant step-size parameter \(\alpha\) (where \(0 < \alpha \leq 1\)) ensures that more recent rewards are given higher weight compared to older rewards. This is achieved through an exponential decay, where the influence of each past reward decreases exponentially as the time since its observation increases.

For example, the update rule for a new sample average can be written as:
\[ Q_{n+1} = \alpha (R_n - Q_n) + Q_n \]

This results in \(Q_{n+1}\) being a weighted sum of past rewards and the initial estimate:
\[ Q_{n+1} = \sum_{i=1}^{n} (\alpha(1-\alpha)^{n-i}) R_i + (1-\alpha)^n Q_1 \]

The weights decay exponentially according to \( (1 - \alpha)^{n-i} \), ensuring that more recent rewards have a higher impact.
```java
// Pseudocode for updating Q(a) with a constant step-size parameter alpha
for each action a:
    R = bandit(a)
    Q[a] = Q[a] + alpha * (R - Q[a])
```
x??

---

#### Convergence Conditions for Step-size Parameters

Background context: For the sample-average method and other methods using variable step-size parameters, certain conditions must be met to ensure convergence. These conditions are derived from stochastic approximation theory.

:p What are the two conditions required for a sequence of step-size parameters \(\alpha_n(a)\) to ensure convergence with probability 1?
??x
For a sequence of step-size parameters \(\alpha_n(a)\), the following two conditions must be met:

1. The sum of the step-size parameters over all steps should diverge:
\[ \sum_{n=1}^{\infty} \alpha_n(a) = \infty \]

2. The sum of the squares of the step-size parameters should converge to a finite value:
\[ \sum_{n=1}^{\infty} \alpha_n^2(a) < \infty \]

These conditions ensure that steps are large enough initially to overcome initial conditions and random fluctuations, but small enough eventually to guarantee convergence.
```java
// Pseudocode for checking the convergence conditions
for each step n:
    if (sum of alpha_n(a) from 1 to n diverges AND sum of alpha_n^2(a) from 1 to n converges):
        converge = true
```
x??

---

#### Experiment on Nonstationary Problems

Background context: To demonstrate the difficulties that sample-average methods face in nonstationary environments, an experiment can be conducted using a modified version of the 10-armed testbed where action values change over time. This involves comparing the performance of an action-value method using sample averages with another method using a constant step-size parameter.

:p How would you design and conduct an experiment to show the challenges faced by sample-average methods in nonstationary environments?
??x
To demonstrate the challenges faced by sample-average methods in nonstationary environments, follow these steps:

1. **Experiment Setup**:
   - Use a modified 10-armed testbed where all action values \(q^*(a)\) start equal and then change independently over time.
   - Add a normally distributed increment with mean zero and standard deviation of 0.01 to each action value at every step.

2. **Comparison**:
   - Implement an action-value method using sample averages, which update the estimates based on past rewards without any step-size parameter adjustment.
   - Implement another action-value method using a constant step-size parameter \(\alpha = 0.1\), where each reward influences the estimate according to this fixed rate.

3. **Parameter Settings**:
   - Set the exploration parameter \(\epsilon\) (or \(\epsilon = 1 - \alpha\)) to \(0.1\).
   - Run both methods for a large number of steps, say 10,000 steps, and record their performances.

4. **Plotting**:
   - Prepare plots similar to Figure 2.2 showing the average rewards over time for each method.
   - Analyze how well each method adapts to changes in the environment and compare their long-term performance.

By running this experiment, you can observe how sample-average methods struggle with nonstationary environments due to their reliance on older data, while constant step-size parameter methods better adapt to recent changes.
```java
// Pseudocode for designing the experiment
for each step n:
    // Update action values using normal increment
    q_star[a] += normally distributed increment
    
    // Select an action with epsilon-greedy policy
    if random() < 0.1: 
        a = choose_random_action()
    else:
        a = argmax(Q[a])
    
    // Get the reward from the selected action
    R = bandit(a)
    
    // Update Q values using sample averages and constant step-size parameter
    Q_sample[a] += alpha * (R - Q_sample[a])
    Q_const[a] += 0.1 * (R - Q_const[a])
```
x??

