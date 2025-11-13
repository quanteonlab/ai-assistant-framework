# Flashcards: 2A012---Reinforcement-Learning_processed (Part 74)

**Starting Chapter:** Off-policy Traces with Control Variates

---

#### Generalizing Recursive Equations to Truncated Versions

Background context: The text discusses generalizing recursive equations for off-policy traces with control variates. Specifically, it deals with the truncation of these equations and their application to importance sampling.

:p How do you generalize the three recursive equations to their truncated versions?

??x
To generalize the recursive equations to their truncated versions, we focus on approximating $G_{s,t}$ using sums of state-based TD errors. The original equation for the non-truncated return is:
$$G_{s,t} = \gamma^t (R_{t+1} + \pi(St+1) - V(St)) + \gamma^t V(St)$$

However, in practice, we often use a truncated version of this return. The truncated version can be approximated by:
$$

G_{s,t} \approx V(St) + \sum_{k=t}^{h-1} \delta_k \prod_{i=t+1}^k (1-\gamma)$$

Where:
- $V(St)$ is the approximate value function.
- $\delta_k = R_{t+k+1} + \pi(S_{t+k+1}) - V(S_{t+k+1})$ is the TD error.

For state-based returns, this generalizes to:
$$G_{s,t} = \gamma^t (R_{t+1} + \pi(St+1) - v(St)) + \gamma^t v(St)$$

The truncated version of this return can be approximated as:
$$

G_{s,t} \approx v(St) + \sum_{k=t}^{h-1} \delta_k \prod_{i=t+1}^k (1-\gamma)$$

Where:
- $v(St)$ is the approximate state value function.
- $\delta_k = R_{t+k+1} + \pi(S_{t+k+1}) - v(S_{t+k+1})$ is the TD error.

For simplicity, consider the case of $h=0$, and use the notation:

$$V_k = v(S_k)$$

The truncated version becomes exact if the value function does not change over time. This can be proven by showing that the sum of the errors from time $t $ to$h-1$ converges to zero.

??x
The answer is derived from understanding the approximation process and its convergence properties under constant value functions.
```java
// Example pseudo-code for updating approximate values
for (int t = 0; t < h - 1; t++) {
    delta[t] = R[t + 1] + pi(S[t + 1]) - V[S[t + 1]];
    V[S[t]] += alpha * delta[t] * prod(1 - gamma, t + 1, k);
}
```
x??

---

#### Off-Policy Return with Control Variates

Background context: The text discusses incorporating importance sampling into off-policy methods using control variates. It generalizes the return for state-based and action-based cases.

:p How is the generalized $\epsilon$-return for off-policy methods defined?

??x
The generalized $\epsilon$-return for off-policy methods, in the case of state-based returns, is defined as:

$$G_{s,t} = \rho_t (R_{t+1} + \pi(S_{t+1}) - v(S_{t})) + \rho_t v(S_{t})$$

Where:
- $\rho_t = \frac{\pi(A_t|S_t)}{b(A_t|S_t)}$ is the importance sampling ratio.
- $R_{t+1}$ is the next reward.
- $\pi(S_{t+1})$ and $v(S_{t})$ are the value under the target policy and behavior policy, respectively.

The truncated version of this return can be approximated as:
$$G_{s,t} \approx v(S_t) + \sum_{k=t}^{h-1} \delta_k (1 - \rho_t) \prod_{i=t+1}^k (1-\gamma)$$

Where:
- $\delta_k = R_{t+k+1} + \pi(S_{t+k+1}) - v(S_{t+k+1})$.

For simplicity, consider the case of $h=0$:

$$G_{s,t} = v(S_t) + (R_{t+1} + \pi(S_{t+1}) - v(S_{t+1}))$$

This approximation becomes exact if the value function does not change. The update for this is:
$$

V(S_t) = V(S_t) + \alpha (\rho_t G_{s,t} - r(\pi, S_t))$$

Where:
- $\alpha$ is the learning rate.
- $r(\pi, S_t)$ is the actual reward.

??x
The answer involves understanding how importance sampling and control variates are used to approximate off-policy returns in a truncated manner.
```java
// Pseudo-code for updating state value using off-policy TD update
for (int t = 0; t < h - 1; t++) {
    delta[t] = R[t + 1] + pi(S[t + 1]) - V[S[t + 1]];
    V[S[t]] += alpha * rho * (delta[t] / prod(1 - gamma, t + 1, k)) - r(pi, S[t]);
}
```
x??

---

#### Off-Policy Traces for Action Values

Background context: The text discusses the off-policy traces and their application to action values. It extends these concepts from state-based returns to action-based ones.

:p How is the generalized $\epsilon$-return defined for action values?

??x
The generalized $\epsilon$-return for action values, in the case of off-policy methods, is defined as:

$$G_{a,t} = R_{t+1} + \rho_{t+1} \left( (1 - \rho_{t+1}) \bar{V}_{t}(S_{t+1}) + \rho_{t+1} (\gamma G_{a, t+1} - q(S_{t+1}, A_{t+1})) \right)$$

Where:
- $R_{t+1}$ is the next reward.
- $\bar{V}_{t}(S_{t+1}) = E_{\pi}[R_{t+2} + \gamma V(S_{t+2}) | S_{t+1}]$.
- $q(S_{t+1}, A_{t+1})$ is the action-value function.
- $\rho_{t+1} = \frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}$.

The truncated version of this return can be approximated as:

$$G_{a,t} \approx q(S_t, A_t) + \sum_{k=t}^{h-1} \delta_k (1 - \rho_t) \prod_{i=t+1}^k (1-\gamma)$$

Where:
- $\delta_k = R_{t+k+1} + \bar{V}_{t+k+1} - q(S_{t+k+1}, A_{t+k+1})$.

For simplicity, consider the case of $h=0$:

$$G_{a,t} = q(S_t, A_t) + (R_{t+1} + \bar{V}_{t+1} - q(S_{t+1}, A_{t+1}))$$

This approximation becomes exact if the value function does not change. The update for this is:
$$

Q(S_t, A_t) = Q(S_t, A_t) + \alpha (\rho_t G_{a,t} - r(\pi, S_t))$$

Where:
- $\alpha$ is the learning rate.
- $r(\pi, S_t)$ is the actual reward.

??x
The answer involves understanding how importance sampling and control variates are used to approximate off-policy returns in a truncated manner for action values.
```java
// Pseudo-code for updating action value using off-policy TD update
for (int t = 0; t < h - 1; t++) {
    delta[t] = R[t + 1] + barV(S[t + 1]) - Q(S[t + 1], A[t + 1]);
    Q(S[t], A[t]) += alpha * rho * (delta[t] / prod(1 - gamma, t + 1, k)) - r(pi, S[t]);
}
```
x??

---

#### Off-Policy TD Update for State and Action Values

Background context: The text discusses the off-policy TD update for state and action values using importance sampling and control variates. It focuses on the logic behind these updates.

:p How is the off-policy TD update derived from the generalized $\epsilon$-return?

??x
The off-policy TD update can be derived from the generalized $\epsilon$-return equations for both state-based and action-based values. For state-based returns, the update rule is:

$$V(S_t) = V(S_t) + \alpha (\rho_t G_{s,t} - r(\pi, S_t))$$

Where:
- $\rho_t = \frac{\pi(A_t|S_t)}{b(A_t|S_t)}$ is the importance sampling ratio.
- $G_{s,t} \approx v(S_t) + (R_{t+1} + \pi(S_{t+1}) - v(S_{t+1}))$.
- $r(\pi, S_t)$ is the actual reward.

For action values, the update rule is:

$$Q(S_t, A_t) = Q(S_t, A_t) + \alpha (\rho_t G_{a,t} - r(\pi, S_t))$$

Where:
- $\bar{V}_{t}(S_{t+1}) = E_{\pi}[R_{t+2} + \gamma V(S_{t+2}) | S_{t+1}]$.
- $G_{a,t} \approx q(S_t, A_t) + (R_{t+1} + \bar{V}_{t+1} - q(S_{t+1}, A_{t+1}))$.

The derivation involves substituting the generalized return into the TD update formula and simplifying it.

??x
The answer is derived from understanding how the importance sampling ratio and control variates are incorporated into the off-policy TD update rules.
```java
// Pseudo-code for updating state value using off-policy TD update
for (int t = 0; t < h - 1; t++) {
    delta[t] = R[t + 1] + pi(S[t + 1]) - V[S[t + 1]];
    V[S[t]] += alpha * rho * (delta[t] / prod(1 - gamma, t + 1, k)) - r(pi, S[t]);
}

// Pseudo-code for updating action value using off-policy TD update
for (int t = 0; t < h - 1; t++) {
    delta[t] = R[t + 1] + barV(S[t + 1]) - Q(S[t + 1], A[t + 1]);
    Q(S[t], A[t]) += alpha * rho * (delta[t] / prod(1 - gamma, t + 1, k)) - r(pi, S[t]);
}
```
x??

--- 

These flashcards cover the key concepts in the provided text, focusing on understanding and deriving the off-policy TD updates for state and action values. Each card includes relevant formulas, explanations, and examples to aid in comprehension. ---

#### Eligibility Traces and Monte Carlo Methods
Eligibility traces make irrevocable updates as a trajectory unfolds, whereas true Monte Carlo methods would make no update for a trajectory if any action within it has zero probability under the target policy. All these methods still bootstrap because their targets depend on current value estimates.
:p What is the key difference between eligibility traces and true Monte Carlo methods in terms of making updates?
??x
Eligibility traces update based on a trajectory, while true Monte Carlo methods do not update if any action within the trajectory has zero probability under the target policy. This means that eligibility traces make irrevocable updates as a trajectory unfolds.
x??

---
#### Provisional Weights in PTD(λ) and PQ(λ)
Recently proposed methods use provisional weights to keep track of updates which may need to be retracted or emphasized depending on later actions taken. These are called PTD(λ) for state and PQ(λ) for state-action versions.
:p What is the role of provisional weights in the context of eligibility traces?
??x
Provisional weights are used to manage updates that might need adjustment based on subsequent actions. They help achieve an exact equivalence between on-policy and off-policy methods, making these algorithms more robust.
x??

---
#### Off-Policy Learning Challenges with Eligibility Traces
Off-policy learning involves two main challenges: correcting the expected value of targets and dealing with the distribution of updates. While eligibility traces handle the first part, they do not address the second.
:p What are the two parts of the off-policy learning challenge addressed by eligibility traces?
??x
The two parts of the off-policy learning challenge are:
1. Correcting for the expected value of the targets.
2. Dealing with the distribution of updates.
Eligibility traces effectively handle the first part but not the second.
x??

---
#### Watkins’s Q(λ) and Backup Diagram
Watkins's Q(λ) decays its eligibility traces as usual until a non-greedy action is taken, at which point it cuts them to zero. The backup diagram for Watkins's Q(λ) ends updates either with the end of the episode or the first non-greedy action.
:p What does Watkins’s Q(λ) do when a non-greedy action is encountered?
??x
When a non-greedy action is encountered, Watkins’s Q(λ) cuts its eligibility traces to zero. This behavior ensures that updates are only made for actions consistent with the current policy.
x??

---
#### Tree-Backup(λ) and Its Backup Diagram
Tree-Backup(λ), or TB(λ), generalizes tree-backup to eligibility traces. The backup diagram includes weighted tree-backup updates of each length, depending on the bootstrapping parameter λ.
:p How does Tree-Backup(λ) generalize tree-backup?
??x
Tree-Backup(λ) extends tree-backup by incorporating eligibility traces. It weights the tree-backup updates according to the bootstrapping parameter λ, allowing for more flexible and off-policy learning while retaining the benefits of tree-backup.
x??

---
#### Detailed Equations for Tree-Backup(λ)
The detailed equations for TB(λ) involve weighted tree-backup updates based on action values. The formula accounts for both immediate rewards and discounted future returns.
:p What is the key equation for calculating Gt in Tree-Backup(λ)?
??x
The key equation for calculating Gt in Tree-Backup(λ) is:
$$G_{a,t} = R_{t+1} + \lambda t+1 \left[ (1 - \lambda t+1)\bar{V}_t(S_{t+1}) + \lambda t+1 \sum_{a' \neq A_{t+1}} \pi(a'|S_{t+1}) q(S_{t+1}, a', w_t) + \pi(A_{t+1}|S_{t+1}) G_{A,t+1} \right]$$

This equation captures the weighted sum of immediate rewards and discounted future returns.
x??

---
#### Eligibility Trace Update for Tree-Backup(λ)
The eligibility trace update involves target-policy probabilities of selected actions. The formula is:
$$z_t = \lambda t \pi(A_t|S_t) z_{t-1} + r q(S_t, A_t, w_t)$$:p What is the eligibility trace update for Tree-Backup(λ)?
??x
The eligibility trace update for Tree-Backup(λ) is:
$$z_t = \lambda t \pi(A_t|S_t) z_{t-1} + r q(S_t, A_t, w_t)$$

This formula updates the eligibility traces based on the target-policy probabilities of selected actions and the reward.
x??

---

#### Double Expected Sarsa and Eligibility Traces
Background context: The text discusses extending algorithms like Double Expected Sarsa to use eligibility traces. This extension is necessary for achieving stability under off-policy training, especially with powerful function approximators.

:p How might Double Expected Sarsa be extended to include eligibility traces?
??x
The extension of Double Expected Sarsa (ES) to use eligibility traces would involve incorporating a mechanism that allows the algorithm to track which states and actions were visited over time. This tracking helps in updating parameters more effectively, even when using off-policy data.

To implement this:
1. Maintain an eligibility trace vector for each state-action pair.
2. Update these vectors based on eligibility criteria related to visitation.
3. Use the accumulated values of these traces during parameter updates.

For example, if ES uses a Q-value update rule like:

```java
w = w + alpha * delta * (x - w^T x)
```

Where $\delta $ is the TD error, and$x$ is the feature vector. With eligibility traces, this becomes more complex but allows for more nuanced updates.

```java
// Pseudocode for ES with Eligibility Traces
for each state-action pair (s,a):
    z[s][a] = gamma * lambda * z[s][a]
    if (s,a) was visited:
        delta = r + gamma * max Q(s', a') - Q(s, a)
        w = w + alpha * delta * x
```

x??

---

#### GTD(λ) Algorithm for State-Values
Background context: The Generalized Temporal Difference (GTD) algorithm with eligibility traces is presented as an off-policy method. It aims to learn state-value estimates even when following a different policy, using linear function approximation.

:p What is the update rule for the GTD(λ) algorithm?
??x
The update rule for the GTD(λ) algorithm involves two key components: a parameter vector $w $ and a vector of eligibility traces$z$.

The update rules are as follows:

1. Update the eligibility trace:
   ```java
   z_t = gamma * lambda * z_{t-1} + delta_t * x_t
   ```

2. Update the weight vector:
   ```java
   w_{t+1} = w_t + alpha_t * (delta_t * z_t - v_t) * x_t
   ```

Where:
- $\alpha_t$ is the step size for updates.
- $delta_t$ is the TD error, calculated as: 
   ```java
   delta_t = R_{t+1} + gamma * v_t - v_t
   ```
- $z_t $ and$z_{t-1}$ are eligibility traces.

If initialized with $v_0 = 0$, the algorithm iteratively updates these values to learn state-value estimates from off-policy data.

x??

---

#### GQ(λ) Algorithm for Action-Values
Background context: The Gradient-TD (GTD) algorithm extended to action-values is introduced. This method aims to learn action-value functions using eligibility traces, which can be used as a control algorithm when the target policy is $\epsilon$-greedy.

:p What is the update rule for the GQ(λ) algorithm?
??x
The update rule for the GQ(λ) algorithm involves updating the parameter vector $w $ and the eligibility trace vector$z$.

1. Update the eligibility trace:
   ```java
   z_t = gamma * lambda * z_{t-1} + delta_t * bar_x_t
   ```

2. Update the weight vector:
   ```java
   w_{t+1} = w_t + alpha_t * (delta_t * z_t - v_t) * bar_x_t
   ```

Where:
- $\alpha_t$ is the step size for updates.
- $delta_t$ is the TD error, calculated as: 
   ```java
   delta_t = R_{t+1} + gamma * v_t - v_t
   ```
- $bar_x_t$ is the average feature vector under the target policy:
   ```java
   bar_x_t = sum_{a in A} pi(a|S_t) * x(S_t, a)
   ```

If initialized with $v_0 = 0$, the algorithm iteratively updates these values to learn action-value estimates from off-policy data.

x??

---

#### HTD(λ) Algorithm
Background context: The Hybrid TD (HTD) algorithm combines aspects of GTD and TD algorithms, providing a strict generalization for off-policy learning. It includes two sets of weights and eligibility traces, ensuring it matches TD(λ) when the policies are identical.

:p What is the update rule for HTD(λ)?
??x
The HTD(λ) update rule involves updating both a weight vector $w $ and an additional set of eligibility traces$z$.

1. Update the primary weight vector:
   ```java
   w_{t+1} = w_t + alpha * (delta * z - v_t) * x_t
   ```

2. Update the secondary weight vector:
   ```java
   v_{t+1} = v_t + beta * (delta * (z - b_z) - delta_v) * x_t
   ```

Where:
- $alpha $ and$beta$ are step size parameters.
- $delta$ is the TD error: 
   ```java
   delta = R_{t+1} + gamma * v_t - v_t
   ```
- $z $ and$b_z$ are eligibility traces:
   ```java
   z_t = gamma * lambda * z_{t-1} + delta_t * x_t
   b_z_t = delta_t * (z_{t-1} - b_z_{t-1}) / (lambda + 1)
   ```

If initialized with $v_0 = 0$, the algorithm iteratively updates these values to learn state-value estimates from off-policy data.

x??

---

#### Emphatic TD(λ) Algorithm
Background context: The Emphatic-TD (ETD) algorithm is extended to eligibility traces, providing strong off-policy convergence guarantees while allowing flexible bootstrapping.

:p What does the ETD(λ) algorithm do?
??x
The ETD(λ) algorithm extends the one-step Emphatic-TD algorithm by incorporating eligibility traces. This extension allows for more flexible bootstrapping and retains strong off-policy convergence guarantees.

The key updates involve:
1. Updating the primary weight vector $w$:
   ```java
   w_{t+1} = w_t + alpha * (delta * z - v_t) * x_t
   ```

2. Updating the eligibility traces for both the behavior and target policies:
   ```java
   z_{t+1} = gamma * lambda * z_t + delta * bar_x_t
   b_z_{t+1} = delta * (z_t - b_z_t) / (lambda + 1)
   ```

Where:
- $alpha$ is the step size.
- $delta$ is the TD error: 
   ```java
   delta = R_{t+1} + gamma * v_t - v_t
   ```
- $bar_x_t$ is the average feature vector under the target policy.

The algorithm ensures that when all $\lambda_t = 1$, it reduces to the one-step ETD algorithm, providing a balance between stability and flexibility in off-policy learning.

x??

---

