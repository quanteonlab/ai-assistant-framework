# Flashcards: 2A012---Reinforcement-Learning_processed (Part 58)

**Starting Chapter:** n-step Off-policy Learning

---

#### n-step Return of Sarsa
Background context: The n-step return for Sarsa is defined as $G_{t:t+n} = Q_t(S_t, A_t) + \sum_{k=t+1}^{\min(t+n, T)} [R_k + \gamma Q_k(S_k, A_k) - Q_{k-1}(S_{k-1}, A_{k-1})]$. This formula accounts for the return over $ n$ steps using a combination of immediate rewards and estimated future values.
:p How is the n-step return defined in Sarsa?
??x
The n-step return $G_{t:t+n}$ is computed by summing the current action-value estimate at time $ t $, followed by discounted immediate rewards and the difference between estimated future value and previous action-value estimates over $ n$ steps. 
```python
def n_step_return(S_t, A_t, Q, R, gamma):
    G = Q(S_t, A_t)
    for k in range(t+1, min(t+n, T)+1):
        if k < T:
            G += gamma ** (k - t) * (R[k] + Q(S_k, A_k) - Q(S_{k-1}, A_{k-1}))
    return G
```
x??

---

#### Expected Sarsa n-step Return
Background context: In the case of Expected Sarsa, the n-step return is defined as $G_{t:t+n} = R_{t+1} + \cdots + R_{t+n} + \bar{V}_{t+n-1}(S_{t+n})$, where $\bar{V}_{t+n-1}(s) = \sum_a \pi(a|s) Q_{t+n-1}(s, a)$. This formula accounts for the sum of immediate rewards and the expected approximate value under the target policy.
:p How is the n-step return defined in Expected Sarsa?
??x
The n-step return $G_{t:t+n}$ in Expected Sarsa includes the cumulative immediate rewards from time $ t+1 $ to $ t+n $, plus the expected approximate value of the state at time $ t+n$. 
```python
def expected_n_step_return(S_t, A_t, Q, R, gamma, target_policy):
    G = 0
    for k in range(t+1, min(t+n, T)+1):
        if k < T:
            G += gamma ** (k - t) * R[k]
    bar_V = sum(target_policy(S_k) * Q(S_k, A_k) for A_k in actions)
    return G + bar_V
```
x??

---

#### n-step O↵-policy Learning
Background context: In o↵-policy learning, the goal is to learn the value function of one policy $\pi $, while following another behavior policy $ b $. The importance sampling ratio$\rho_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$ is used to adjust the weight of returns based on the relative probability of taking actions under different policies.
:p What update rule does n-step o↵-policy Sarsa follow?
??x
The update rule for n-step o↵-policy Sarsa includes weighting the return by the importance sampling ratio $\rho_{t:t+n-1}$:
```python
def off_policy_n_step_sarsa(S_t, A_t, Q, R, gamma, behavior_policy, target_policy):
    G = 0
    for k in range(t+1, min(t+n, T)+1):
        if k < T:
            G += gamma ** (k - t) * R[k]
    rho = product([target_policy(S_k)[A_k] / behavior_policy(S_k)[A_k] for A_k in actions])
    Q[S_t][A_t] += alpha * rho * (G - Q[S_t][A_t])
```
x??

---

#### Importance Sampling Ratio
Background context: The importance sampling ratio $\rho_{t:t+n-1} = \prod_{k=t}^{t+n-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$ is used to adjust the weight of returns when the behavior policy $ b $ and target policy $\pi$ differ. In the on-policy case, where $\pi = b$, this ratio is always 1.
:p What is the importance sampling ratio in o↵-policy learning?
??x
The importance sampling ratio $\rho_{t:t+n-1}$ measures the relative probability of taking actions under the target policy $\pi$ compared to the behavior policy $b$:
```python
def importance_sampling_ratio(S_t, A_t, S_k, A_k, behavior_policy, target_policy):
    return (target_policy(S_k)[A_k] / behavior_policy(S_k)[A_k])
```
x??

---

#### n-step TD with O↵-policy Learning
Background context: The update for n-step TD with o↵-policy learning involves weighting the importance sampling ratio $\rho_{t:t+n-1}$ to adjust the return based on the relative probability of taking actions under different policies. This is particularly useful when following a more exploratory policy.
:p How does the update rule for n-step TD with o↵-policy learning work?
??x
The update rule for n-step TD with o↵-policy learning involves weighting the importance sampling ratio to adjust the return:
```python
def off_policy_n_step_td(V, S_t, R, gamma, behavior_policy, target_policy):
    G = 0
    for k in range(t+1, min(t+n, T)+1):
        if k < T:
            G += gamma ** (k - t) * R[k]
    rho = product([target_policy(S_k)[A_k] / behavior_policy(S_k)[A_k] for A_k in actions])
    V[S_t] += alpha * rho * (G - V[S_t])
```
x??

#### n-step Bootstrapping for Oﬄine Policies
Background context explaining the concept. In Chapter 7, we discuss how to adapt n-step bootstrapping to off-policy learning, particularly focusing on the oﬄine version of Expected Sarsa. The key idea is that importance sampling is used with one less factor compared to the standard n-step algorithm.
:p What does the update for oﬄine n-step Expected Sarsa use?
??x
The update uses an importance sampling ratio of $\theta_{t+1:t+n-1}$ instead of $\theta_{t+1:t+n}$, and it employs the expected version of the n-step return.
x??

---

#### Per-decision Importance Sampling with Control Variates for Oﬄine Policies
Context: Section 7.4 introduces a more sophisticated approach to oﬄine policy learning using per-decision importance sampling, control variates, and recursive returns. It presents an advanced method that addresses variance issues in the standard n-step TD methods.
:p How does the n-step return at horizon $h $ change when following a behavior policy$\beta $ that is not the same as the target policy$\pi$?
??x
The return changes to include a control variate, which helps stabilize updates by ensuring the expected value of the update remains unaffected. The updated formula for the n-step return at horizon $h$ is:
$$G_{t:h} = \theta_t (R_{t+1} + \gamma^k G_{t+1:h}) + (1 - \theta_t) V_{h-1}(S_{t+1})$$where $\theta_t = \frac{\pi(A_t|S_t)}{\beta(A_t|S_t)}$.
x??

---

#### Pseudocode for Oﬄine Policy Prediction Algorithm
Context: The section provides pseudocode to implement the oﬄine policy prediction algorithm.
:p Write the pseudocode for the oﬄine policy state-value prediction algorithm described in this section.
??x
```python
function off_policy_n_step_predict(s, a, R, G, n, b, pi):
    # s: current state, a: action taken, R: reward received, 
    # G: returns, n: number of steps, b: behavior policy, pi: target policy
    
    for t in range(len(R)):
        G[t] = 0
        if t + n < len(s) and t + n < T:
            G[t] += R[t+1]
            next_state = s[t+1]
            next_action = a[t+1]
            G[t] += pi(next_action|next_state) * (G[t+1])
            G[t] += (1 - pi(next_action|next_state)) * V(next_state)
        else:
            # End of episode or horizon
            break
    return G
```
x??

---

#### Action Values with Control Variate for Oﬄine Policies
Context: The section extends the concept to action values, introducing a control variate that ensures the expected update remains stable.
:p Write the formula for the n-step return at horizon $h$ when using oﬄine policy and control variates.
??x
The n-step return at horizon $h$ with control variate is:
$$G_{t:h} = R_{t+1} + \gamma \theta_{t+1} (G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1})) + V_{h-1}(S_{t+1})$$where $\theta_t = \frac{\pi(A_t|S_t)}{\beta(A_t|S_t)}$.
x??

---

#### Control Variate Does Not Change Expected Value
Context: This exercise aims to prove that the control variate does not change the expected value of the return.
:p Prove that the control variate in equation (7.13) does not change the expected value of the return.
??x
The control variate is a term added to ensure the expected update remains stable:
$$G_{t:h} = \theta_t (R_{t+1} + \gamma G_{t+1:h}) + (1 - \theta_t) V_h(S_h)$$

Since $\theta_t$ has an expected value of 1 and is uncorrelated with the estimate, the control variate does not change the expected return.
x??

---

#### Pseudocode for Oﬄine Policy Action-Value Prediction
Context: The section provides pseudocode to implement the oﬄine policy action-value prediction algorithm.
:p Write the pseudocode for the oﬄine policy action-value prediction algorithm described in this section.
??x
```python
function off_policy_n_step_predict_action_values(s, a, R, G, n, b, pi):
    # s: current state, a: action taken, R: reward received, 
    # G: returns, n: number of steps, b: behavior policy, pi: target policy
    
    for t in range(len(R)-1):  # Note the -1 to avoid out-of-bound errors
        if t + n < len(s):
            G[t] = R[t+1]
            next_state = s[t+1]
            next_action = a[t+1]
            G[t] += pi(next_action|next_state) * (G[t+1])
            G[t] -= pi(next_action|next_state) * Q(next_state, next_action)
        else:
            # End of episode or horizon
            break
    return G
```
x??

---

#### General oﬄine Policy n-step Return with Control Variate
Context: This exercise aims to show that the general (oﬄine policy) version of the n-step return can still be written as a sum of state-based TD errors.
:p Show that the general (oﬄine policy) version of the n-step return can still be written exactly and compactly as the sum of state-based TD errors if the approximate state value function does not change.
??x
The general oﬄine policy n-step return with control variate can be expressed as:
$$G_{t:h} = R_{t+1} + \gamma \theta_{t+1} (G_{t+1:h} - Q_h(S_t, A_t)) + V_h(S_t)$$where $ V_h(S_t)$ is the approximate state value function. This can be seen as a sum of state-based TD errors if the value function remains constant.
x??

---

#### Action Version of oﬄine Policy n-step Return with Expected Sarsa
Context: The section extends this to action values, using the control variate in Expected Sarsa.
:p Repeat the above exercise for the action version of the oﬄine policy n-step return and the Expected Sarsa TD error.
??x
The action version of the oﬄine policy n-step return with control variate can be expressed as:
$$G_{t:h} = R_{t+1} + \gamma \theta_{t+1} (G_{t+1:h} - Q_h(S_t, A_t)) + V_h(S_{t+1})$$

This formula reduces to the Expected Sarsa TD error when $h < T$:
$$TD = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$x??

---

#### Programming Exercise for Oﬄine Policy Prediction
Context: The final exercise aims to demonstrate the efficiency of oﬄine policy prediction using (7.13) and (7.2).
:p Devise a small oﬄine policy prediction problem and show that the algorithm using (7.13) and (7.2) is more data-efficient than the simpler algorithm using (7.1) and (7.9).
??x
Create a simple grid-world environment where an agent learns to navigate from start to goal, following a behavior policy $\beta $ but aiming for target policy$\pi$. Implement both algorithms and compare their performance by observing how quickly they converge with less data.
```python
# Example pseudocode
def test_off_policy_prediction():
    env = GridWorldEnv()
    b = BehaviorPolicy(env)
    pi = TargetPolicy(env)
    off_policy_timesteps = 1000  # Number of timesteps for oﬄine policy learning
    on_policy_timesteps = 2000  # Number of timesteps for on-policy learning
    
    off_policy_values, _ = off_policy_n_step_predict(env.start_state(), b, pi, n=5, T=off_policy_timesteps)
    on_policy_values, _ = on_policy_n_step_predict(env.start_state(), pi, n=5, T=on_policy_timesteps)
    
    # Compare the number of timesteps to reach a certain value of Q
```
x??

---

#### 3-Step Tree Backup Update Diagram
Background context: The 3-step tree-backup algorithm is an off-policy learning method that does not use importance sampling. It extends the idea of a backup diagram by incorporating all possible actions at each level, rather than just following the actual action taken.
:p Describe the 3-step tree-backup update process using the provided diagram.
??x
The 3-step tree-backup algorithm uses a diagram to visualize the updates for state-action values. It extends traditional backup diagrams by considering all possible actions at each step, weighted by their probabilities under the target policy π.

For example, in the given diagram:
- The central spine represents three states and rewards.
- Actions that were not selected from any state are considered using their respective probabilities.
- Each action node's value is updated based on its probability of being chosen by the policy π and the values of all its child nodes.

Here’s a simplified version of how this works in practice:

```plaintext
St, At, At+1, Rt+1, St+1, At+2, Rt+2, St+2, At+3, Rt+3, St+3
```
The update involves:
- Considering the actual action taken (At+1) and its probability.
- Weighting the values of all non-selected actions at each subsequent level by their probabilities.

This process forms a tree structure where each leaf node contributes to the target value based on its probability under π. The actual action At+1 does not contribute directly but affects the weights of other nodes.
x??

---

#### n-Step Tree Backup Algorithm
Background context: The n-step tree-backup algorithm is designed for off-policy learning without using importance sampling, extending Q-learning and Expected Sarsa to multi-step updates. It uses a recursive definition to calculate the target value for each state-action pair.

Relevant formula: 
$$G_{t:t+n} = R_{t+1} + \sum_{a_6=At+1} \pi(a|S_{t+1}) Q_{n-1}(S_{t+1}, a) + \pi(At+1|S_{t+1}) G_{t+1:t+n}$$:p What is the target value $ G_{t:t+n}$ in the n-step tree-backup algorithm?
??x
The target value $G_{t:t+n}$ in the n-step tree-backup algorithm is calculated as:
$$G_{t:t+n} = R_{t+1} + \sum_{a_6=At+1} \pi(a|S_{t+1}) Q_{n-1}(S_{t+1}, a) + \pi(At+1|S_{t+1}) G_{t+1:t+n}$$

This formula considers the immediate reward $R_{t+1}$, the values of all non-selected actions at state $ S_{t+1}$, and the target value from subsequent states.

For example, if $n=3$:
$$G_{t:t+3} = R_{t+1} + \sum_{a_6=At+1} \pi(a|S_{t+1}) Q_{2}(S_{t+1}, a) + \pi(At+1|S_{t+1}) (R_{t+2} + \sum_{a_6=At+2} \pi(a|S_{t+2}) Q_{1}(S_{t+2}, a))$$

This target value is used to update the action-value function $Q_n(S_t, A_t)$.

```java
for (int t = 0; t < T-1; ++t) {
    int n = // some predefined or calculated value;
    double G = R[t+1];
    for (int k = 0; k < n-1 && t+k+2 < T; ++k) {
        if (A[t+k+1] != A[t+k+2]) {
            G += π(A[t+k+2]|S[t+k+2]) * Q[t+k+2];
        }
    }
    double delta = G - Q[t][A[t]];
    Q[t][A[t]] += α * delta;
}
```
x??

---

#### Detailed Equations for n-Step Tree Backup
Background context: The n-step tree-backup algorithm has a detailed recursive definition and an update rule that generalizes the one-step return used in Expected Sarsa.

Relevant formulas:
1. One-step return:
$$G_{t:t+1} = R_{t+1} + \sum_{a_6=At+1} \pi(a|S_{t+1}) Q(S_{t+1}, a)$$2. Two-step tree-backup return:
$$

G_{t:t+2} = R_{t+1} + \sum_{a_6=At+1} \pi(a|S_{t+1}) Q_{1}(S_{t+1}, a) + \pi(At+1|S_{t+1}) (R_{t+2} + \sum_{a_6=At+2} \pi(a|S_{t+2}) Q(S_{t+2}, a))$$3. General n-step tree-backup return:
$$

G_{t:t+n} = R_{t+1} + \sum_{a_6=At+1} \pi(a|S_{t+1}) Q_{n-1}(S_{t+1}, a) + \pi(At+1|S_{t+1}) G_{t+1:t+n}$$:p Explain the general recursive definition of the n-step tree-backup return.
??x
The general recursive definition of the n-step tree-backup return is:
$$

G_{t:t+n} = R_{t+1} + \sum_{a_6=At+1} \pi(a|S_{t+1}) Q_{n-1}(S_{t+1}, a) + \pi(At+1|S_{t+1}) G_{t+1:t+n}$$

This equation considers the immediate reward $R_{t+1}$, the values of all non-selected actions at state $ S_{t+1}$, and the target value from subsequent states. The term $\pi(At+1|S_{t+1}) G_{t+1:t+n}$ accounts for the contribution from the action actually taken.

For example, if $n=3$:
$$G_{t:t+3} = R_{t+1} + \sum_{a_6=At+1} \pi(a|S_{t+1}) Q_2(S_{t+1}, a) + \pi(At+1|S_{t+1}) (R_{t+2} + \sum_{a_6=At+2} \pi(a|S_{t+2}) Q(S_{t+2}, a))$$

This recursive formula ensures that the update rule considers not just the immediate reward and value of actions, but also their values at future steps, forming a tree-like structure.

```java
double G = R[t + 1];
for (int k = 0; k < n - 1 && t + k + 2 < T; ++k) {
    if (A[t + k + 1] != A[t + k + 2]) {
        G += π(A[t + k + 2]|S[t + k + 2]) * Q[t + k + 2];
    }
}
```
x??

---

#### Tree-Backup Update Weights
Background context: In the tree-backup update, weights are assigned to action nodes based on their probabilities of being selected under the target policy π. These weights determine how much each node contributes to the overall target value.

Relevant formulas:
$$\text{Weight for } a = \pi(a|S_{t+1}) \cdot \prod_{i=t+2}^{t+n-1} \pi(A_i|S_i)$$:p How are the weights assigned in the tree-backup update?
??x
In the tree-backup update, the weights for each action node $a $ at level$k $(where $ k = t + 1, t + 2, \ldots, t + n - 1$) are assigned based on their probabilities under the target policy π. The weight of an action node is calculated as:
$$\text{Weight for } a = \pi(a|S_{t+1}) \cdot \prod_{i=t+2}^{t+n-1} \pi(A_i|S_i)$$

This means that each leaf node contributes to the target value with a weight proportional to its probability of being selected under π. For example, at level $t + 1$, all non-selected actions contribute with weights:
$$\text{Weight for } a_0 = \pi(a_0|S_{t+1})$$

For the action actually taken at each step, it does not directly contribute but affects the weight of the next-level nodes. Specifically, if $a_{t+1}$ is the actual action:
$$\text{Weight for } a_{t+2} = \pi(a_{t+1}|S_{t+1}) \cdot \pi(a_{t+2}|S_{t+2})$$

This process forms a tree structure where each node's value is updated based on its probability and the values of all child nodes.

```java
for (int k = 0; k < n - 1 && t + k + 2 < T; ++k) {
    if (A[t + k + 1] != A[t + k + 2]) {
        G += π(A[t + k + 2]|S[t + k + 2]) * Q[t + k + 2];
    }
}
```
x??

---

#### C/Java Pseudocode for n-Step Tree Backup
Background context: The pseudocode for the n-step tree-backup algorithm is provided, detailing how to update action values based on the target value $G_{t:t+n}$.

Relevant formulas:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [ G_{t:t+n} - Q(S_t, A_t)]$$:p Provide a detailed pseudocode for the n-step tree-backup algorithm.
??x
Here is the pseudocode for the n-step tree-backup algorithm:

```java
// Initialize Q values and set hyperparameters α (learning rate)
double[] Q = new double[numStates * numActions];
for (int t = 0; t < T - 1; ++t) {
    int n = // some predefined or calculated value;
    double G = R[t + 1];
    for (int k = 0; k < n - 1 && t + k + 2 < T; ++k) {
        if (A[t + k + 1] != A[t + k + 2]) {
            G += π(A[t + k + 2]|S[t + k + 2]) * Q[S[t + k + 2]];
        }
    }
    double delta = G - Q[S[t]][A[t]];
    Q[S[t]][A[t]] += α * delta;
}
```

Explanation:
- The algorithm iterates through each time step $t $ up to$T-1$.
- For each step, it calculates the n-step target value $G_{t:t+n}$ by considering the immediate reward and recursively adding the values of non-selected actions.
- The update rule then adjusts the action-value function $Q(S_t, A_t)$ based on the difference between the target value $G_{t:t+n}$ and its current value.

This pseudocode ensures that the algorithm updates the Q-values in a manner consistent with the n-step tree-backup method, taking into account both immediate rewards and future values.
x??

