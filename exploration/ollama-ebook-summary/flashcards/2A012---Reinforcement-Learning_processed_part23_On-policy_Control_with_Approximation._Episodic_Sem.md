# Flashcards: 2A012---Reinforcement-Learning_processed (Part 23)

**Starting Chapter:** On-policy Control with Approximation. Episodic Semi-gradient Control

---

#### Episodic Semi-gradient Control
Episodic semi-gradient control extends the ideas of semi-gradient prediction methods to action values, allowing for parametric approximation. In this method, the approximate action-value function \(\hat{q}^{\pi}(s, a; w)\) is represented as a parameterized functional form with weight vector \(w\). The update rule for the weights \(w\) in the semi-gradient Sarsa algorithm can be expressed as:
\[ w_{t+1} = w_t + \alpha \left( U_t - \hat{q}(s, a; w_t) \right) \nabla_w \hat{q}(s, a; w_t) \]

:p What is the update rule for semi-gradient Sarsa in episodic control?
??x
The update rule for semi-gradient Sarsa involves adjusting the weight vector \(w\) based on the difference between the target value \(U_t\) and the current prediction \(\hat{q}(s, a; w_t)\), weighted by the gradient of the action-value function with respect to the weights. This ensures that the predicted values align better with actual returns.
```java
// Pseudocode for semi-gradient Sarsa update rule
public void updateWeights(double[] wt, double alpha, double Ut, double[] gradQ) {
    int weightLength = gradQ.length;
    double[] newWt = Arrays.copyOf(wt, weightLength);
    
    // Update weights based on the difference and gradient
    for (int i = 0; i < weightLength; i++) {
        newWt[i] += alpha * (Ut - wt[i]) * gradQ[i];
    }
}
```
x??

---

#### Mountain Car Task Example
The Mountain Car task is a classic example used to illustrate continuous control tasks. In this problem, the goal is to drive an underpowered car up a steep mountain road. The action space consists of three possible actions: full throttle forward (+1), full throttle reverse (−1), and zero throttle (0). The state space includes position \(x_t\) and velocity \(\dot{x}_t\).

The dynamics of the system are described by:
\[ x_{t+1} = \text{bound}\left( x_t + \dot{x}_t + 1 \right) - \frac{2}{30} \cos(3 x_t), \]
where bound operation enforces \( -1.2 \leq x_{t+1} \leq 0.5 \) and \(-0.07 \leq \dot{x}_{t+1} \leq 0.07\).

:p What are the dynamics of the Mountain Car task?
??x
The dynamics of the Mountain Car task involve updating the position \(x_{t+1}\) based on the current position and velocity, with a cosine term to simulate the effect of gravity. The position is constrained between \(-1.2\) and \(0.5\), and the velocity is bounded between \(-0.07\) and \(0.07\).
```java
// Pseudocode for updating state in Mountain Car task
public double updatePosition(double x, double dx) {
    double nextX = x + dx - (2/30.0) * Math.cos(3*x);
    return Math.max(-1.2, Math.min(nextX, 0.5));
}
```
x??

---

#### Tile Coding Feature Vector
For the Mountain Car task, tile coding is used to convert continuous state and action variables into a discrete feature vector. The position \(x\) and velocity \(\dot{x}\) are mapped to a grid of tiles, where each tile covers an 8th of the bounded distance in both dimensions. The indices for the active tiles are determined using the `IHT` algorithm.

:p How is the state-action feature vector created for the Mountain Car task?
??x
The state-action feature vector for the Mountain Car task is created by applying tile coding to the continuous state and action variables. Each pair of state \(s\) and action \(a\) is mapped to a set of binary features, which are then combined linearly with the parameter vector \(w\). The indices for active tiles are obtained using an `IHT` algorithm.
```java
// Pseudocode for creating feature vector using tile coding
public int[] getTileIndices(double x, double dx, Action action) {
    int iht = IHT(4096);
    return tiles(iht, 8, [8*x/(0.5+1.2), 8*dx/(0.07+0.07)], action);
}
```
x??

---

#### Average Reward Formulation
In the continuing case of control problems, the traditional discounted reward formulation needs to be replaced with an average-reward formulation due to the presence of function approximation. This new formulation involves using differential value functions and can lead to different optimal policies compared to the discounted reward case.

:p Why is the average-reward formulation necessary in the continuing case?
??x
The average-reward formulation is necessary in the continuing case because it accounts for long-term rewards that cannot be captured by a simple discounting mechanism, especially when function approximation is used. It allows the algorithm to converge to policies that optimize the long-term average reward rather than just the immediate discounted returns.
```java
// Pseudocode for transitioning to average-reward formulation
public double calculateAverageReward(double[] valueFunction) {
    int length = valueFunction.length;
    double sum = 0;
    
    // Calculate average reward over a large number of episodes or steps
    for (int i = 0; i < length; i++) {
        sum += valueFunction[i];
    }
    return sum / length;
}
```
x??

---

#### Episodic Semi-gradient Sarsa Pseudocode
The episodic semi-gradient Sarsa algorithm follows the general pattern of on-policy GPI (Gradient Policy Improvement). It uses a parameterized action-value function and updates it based on observed returns. The policy is improved by following an \(\epsilon\)-greedy strategy, where actions are selected according to the current estimates.

:p What is the pseudocode for the episodic semi-gradient Sarsa algorithm?
??x
The pseudocode for the episodic semi-gradient Sarsa algorithm involves initializing weights \(w\) and then iteratively updating them based on observed returns. The policy is improved using \(\epsilon\)-greedy action selection.
```java
// Pseudocode for Episodic Semi-gradient Sarsa Algorithm
public void semiGradientSarsa(double[] w, double alpha, double epsilon) {
    // Initialize weights arbitrarily (e.g., to 0)
    
    while (true) {
        Episode e = initializeEpisode(); // Get initial state and action
        
        while (!episodeIsTerminated(e)) { // For each step in the episode
            takeAction(e); // Take an action based on current policy
            
            double reward = getReward(e); // Observe reward
            
            updateState(e); // Update state for next time step
            
            if (stateIsTerminal(e.newState)) {
                w += alpha * (reward - predictValue(w, e.state, e.action)) *
                     gradientOfPredictedValue(w, e.state, e.action);
            } else {
                double qNext = predictValue(w, e.newState, getActionForNewState());
                w += alpha * (reward + gamma * qNext - predictValue(w, e.state, e.action)) *
                     gradientOfPredictedValue(w, e.state, e.action);
            }
        }
    }
}
```
x??

---

#### Semi-gradient n-step Sarsa Overview
Semi-gradient n-step Sarsa extends the tabular form of semi-gradient Sarsa to a function approximation setting. The update target is an n-step return, which generalizes from its tabular form (Equation 7.4) to a function approximation form as shown in Equation 10.4.
:p What is Semi-gradient n-step Sarsa?
??x
Semi-gradient n-step Sarsa uses an n-step return as the update target in the semi-gradient Sarsa update equation, allowing for more efficient updates by combining information from multiple time steps. This method improves learning efficiency and can achieve better asymptotic performance.
---
#### Update Equation of Semi-gradient n-step Sarsa
The update rule for the weights \( w \) in semi-gradient n-step Sarsa is given by Equation 10.5, which involves using an n-step return to adjust the weights based on the current state and action.
:p What is the update equation for Semi-gradient n-step Sarsa?
??x
The weight vector \( w \) is updated as follows:
\[ w_{t+n} = w_{t+n-1} + \alpha [G_t:t+n - \hat{q}(S_t, A_t, w_{t+n-1})] r(S_t, A_t, w_{t+n-1}) \]
where \( G_t:t+n \) is the n-step return defined as:
\[ G_t:t+n = \sum_{i=\tau+1}^{\min(\tau+n, T)} R_i + \gamma^n \hat{q}(S_{\tau+n}, A_{\tau+n}, w_{t+n}) \]
for \( t < T - n \) and \( G_t:t+n = G_t \) if \( t \geq T - n \).
---
#### Pseudocode for Episodic Semi-gradient n-step Sarsa
The pseudocode provided outlines the algorithm for estimating \( \hat{q}^{\pi} \) or \( q^* \) using a differentiable action-value function parameterization.
:p Can you provide the pseudocode for Episodic Semi-gradient n-step Sarsa?
??x
```plaintext
Episodic semi-gradient n-step Sarsa for estimating hat{q}^{pi} or q*
Input: a differentiable action-value function parameterization \hat{q}:S × A × R^d → R
Input: a policy pi (if estimating q^{pi})
Algorithm parameters: step size alpha > 0, small \epsilon > 0, a positive integer n
Initialize value-function weights w in R^d arbitrarily (e.g., w=0)
All store and access operations (St, At, and Rt) can take their index mod n+1
Loop for each episode:
    Initialize and store S_0 = terminal
    Select and store an action A_0 ∼ pi(·|S_0) or \epsilon-greedy w.r.t. hat{q}(S_0, ·, w)
    T-1 Loop for t=0, 1, 2,...:
        | If t < T, then:
            | Take action A_t
            | Observe and store the next reward as R_{t+1} and the next state as S_{t+1}
            | If S_{t+1} is terminal, then: Goto T
            | else: Select and store A_{t+1} ∼ pi(·|S_{t+1}) or \epsilon-greedy w.r.t. hat{q}(S_{t+1}, ·, w)
            | \tau <- t - n + 1 (time whose estimate is being updated)
            | If \tau >= 0:
                | G <- sum_{i=\tau+1}^{min(\tau+n, T)} R_i
                | If \tau + n < T: G <- G + gamma^n * hat{q}(S_{\tau+n}, A_{\tau+n}, w)
            | w <- w + alpha [G - hat{q}(S_{\tau}, A_{\tau}, w)] * r(S_{\tau}, A_{\tau}, w)
    Until \tau = T-1
```
---
#### Performance of Semi-gradient n-step Sarsa on Mountain Car Task
The performance of semi-gradient n-step Sarsa, particularly with an optimal intermediate level of bootstrapping (n=4), was demonstrated to outperform one-step and larger values of n. The results showed faster learning at \( n=8 \) compared to \( n=1 \).
:p What does the figure 10.3 indicate about Semi-gradient n-step Sarsa on Mountain Car?
??x
Figure 10.3 illustrates that semi-gradient n-step Sarsa, with a well-chosen step size (α = 0.5/8 for \( n=1 \) and α = 0.3/8 for \( n=8 \)), performs better than one-step Sarsa on the Mountain Car task at \( n=8 \). The algorithm learns faster and achieves a better asymptotic performance with an intermediate level of bootstrapping.
---
#### Effect of Step Size (α) and n on Learning
The study showed that the choice of step size α and the number of steps n significantly affect the rate of learning. An optimal balance is achieved when \( n=4 \), indicating an intermediate level of bootstrapping.
:p What does Figure 10.4 reveal about the effect of α and n?
??x
Figure 10.4 demonstrates that the performance of semi-gradient n-step Sarsa with tile-coding function approximation on the Mountain Car task is most effective when \( n=4 \), representing an intermediate level of bootstrapping. The results show a higher rate of learning for this choice, with standard errors ranging from 0.5 to about 4 across different values of n.
---
#### Exercise 10.1: Monte Carlo Methods in Sarsa
Monte Carlo methods involve waiting for the end of an episode before updating the value function based on all sampled returns. However, these are not covered explicitly in this chapter due to their focus on gradient-based methods like Sarsa and Expected Sarsa.
:p Why is pseudocode for Monte Carlo methods not provided?
??x
Monte Carlo methods involve sampling full episodes before updating the value function, which can be computationally expensive compared to gradient-based methods. Since this chapter focuses on semi-gradient methods that provide a balance between Monte Carlo sampling and bootstrapping, it may not explicitly cover Monte Carlo updates due to their different nature.
---
#### Exercise 10.2: Semi-gradient One-step Expected Sarsa
Semi-gradient one-step Expected Sarsa involves using the expected value of future rewards based on a single step ahead.
:p Give pseudocode for semi-gradient one-step Expected Sarsa for control.
??x
```plaintext
Semi-gradient one-step Expected Sarsa for Control:
Input: differentiable action-value function parameterization \hat{q}:S × A × R^d → R
Algorithm parameters: step size alpha > 0, a small epsilon > 0
Initialize value-function weights w in R^d arbitrarily (e.g., w=0)
Loop for each episode:
    Initialize and store S_0 = terminal state
    Select and store an action A_0 ∼ pi(·|S_0) or \epsilon-greedy w.r.t. hat{q}(S_0, ·, w)
    T-1 Loop for t=0, 1, ...:
        Take action A_t
        Observe the next reward R_{t+1} and state S_{t+1}
        Select and store A_{t+1} ∼ pi(·|S_{t+1}) or \epsilon-greedy w.r.t. hat{q}(S_{t+1}, ·, w)
        Update the weight vector as:
            w <- w + alpha * [R_{t+1} + \gamma * (hat{q}(S_{t+1}, A_{t+1}, w) - hat{q}(S_t, A_t, w))] * hat{q}(S_t, A_t, w)
    Until the episode ends
```
---
#### Exercise 10.3: Standard Errors in Large n vs Small n
The higher standard errors at large values of \( n \) compared to small values are due to overfitting and instability in the estimates. Smaller values of \( n \) reduce variance but increase bias, while larger values of \( n \) decrease bias but can increase variance.
:p Why do results have higher standard errors at large n than at small n?
??x
Higher standard errors at large values of \( n \) are due to the increased risk of overfitting and instability in the estimates. As \( n \) increases, the algorithm relies more on predictions far into the future, which can be less accurate. Conversely, smaller values of \( n \) reduce variance but introduce bias, leading to more stable performance with lower standard errors.
---

#### Average Reward Setting for Continuing Tasks
In this setting, we address Markov decision problems (MDPs) where there is no inherent termination condition; interactions between an agent and environment continue indefinitely. Unlike the discounted setting which discounts future rewards to a lower value, the average reward setting does not discount future rewards.

The quality of a policy \( \pi \) is defined as the long-term average reward it generates:
\[ r(\pi) = \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^{h} E[R_t | S_0, A_0:A_{t-1} \sim \pi] \]
or equivalently,
\[ r(\pi) = \lim_{t \to \infty} E[R_t | S_0, A_0:A_{t-1} \sim \pi]. \]

This can also be expressed as:
\[ r(\pi) = \sum_s \mu_\pi(s) \sum_a \pi(a|s) \sum_{s',r} p(s',r | s,a)r, \]
where \( \mu_\pi(s) \) is the steady-state distribution defined as:
\[ \mu_\pi(s) = \lim_{t \to \infty} Pr(S_t = s | A_0:A_{t-1} \sim \pi). \]

This assumption about the MDP, known as ergodicity, ensures that starting from any state and following policy \( \pi \), the long-term expectation of being in a state depends only on the policy and transition probabilities. Policies are compared based on their average reward per time step.

:p What is the definition of the quality of a policy in terms of the average reward?
??x
The quality of a policy \( \pi \) is defined as the limit of the average total reward received over time, under that policy:
\[ r(\pi) = \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^{h} E[R_t | S_0, A_0:A_{t-1} \sim \pi]. \]
This measures the long-term average reward generated by following policy \( \pi \).
x??

---

#### Steady-State Distribution
The steady-state distribution \( \mu_\pi(s) \) is a key concept in the average reward setting. It represents the probability of being in state \( s \) after a long period, assuming actions are chosen according to the policy \( \pi \).

Formally,
\[ \mu_\pi(s) = \lim_{t \to \infty} Pr(S_t = s | A_0:A_{t-1} \sim \pi). \]

It also satisfies the equation:
\[ \sum_s \mu_\pi(s) \sum_a \pi(a|s) p(s',r | s,a) = \mu_\pi(s'). \]
This equation ensures that the distribution remains unchanged over time if actions are chosen according to \( \pi \).

:p What is the definition of the steady-state distribution in terms of policy \( \pi \)?
??x
The steady-state distribution \( \mu_\pi(s) \) for a policy \( \pi \) is defined as the probability of being in state \( s \) after many time steps, assuming actions are chosen according to \( \pi \). It satisfies:
\[ \mu_\pi(s) = \lim_{t \to \infty} Pr(S_t = s | A_0:A_{t-1} \sim \pi). \]
It also ensures that the distribution is invariant under the policy, as shown by the equation:
\[ \sum_s \mu_\pi(s) \sum_a \pi(a|s) p(s',r | s,a) = \mu_\pi(s'). \]
x??

---

#### Differential Return and Value Functions
In the average reward setting, returns are defined differently. The differential return \( G_t \) is:
\[ G_t = R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + \cdots. \]

This represents the difference between the actual rewards and the long-term average reward.

The corresponding value functions are known as differential value functions, defined as:
\[ v_\pi(s) = E_\pi[G_t | S_t = s] \]
and
\[ q_\pi(s,a) = E_\pi[G_t | S_t = s, A_t = a]. \]

These have their own Bellman equations, which are similar to but slightly different from the ones we've seen before.

:p What is the definition of differential return in the average reward setting?
??x
The differential return \( G_t \) in the average reward setting is defined as the difference between actual rewards and the long-term average reward:
\[ G_t = R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + \cdots. \]
This captures the deviation of each reward from the expected average reward.
x??

---

#### Bellman Equations for Differential Value Functions
The Bellman equations for differential value functions are similar to those for discounted or undiscounted returns, but with slight modifications:
\[ v_\pi(s) = E_\pi[R_{t+1} - r(\pi) + G_t | S_t = s] \]
and
\[ q_\pi(s,a) = E_\pi[R_{t+1} - r(\pi) + G_t | S_t = s, A_t = a]. \]

These equations capture the value of being in state \( s \) or taking action \( a \) at time \( t \), considering both immediate rewards and future rewards relative to the average reward.

:p What are the Bellman equations for differential value functions?
??x
The Bellman equations for differential value functions are:
\[ v_\pi(s) = E_\pi[R_{t+1} - r(\pi) + G_t | S_t = s] \]
and
\[ q_\pi(s,a) = E_\pi[R_{t+1} - r(\pi) + G_t | S_t = s, A_t = a]. \]

These equations express the value functions in terms of both immediate rewards and future rewards relative to the long-term average reward \( r(\pi) \).
x??

---

#### Removing Characters and Adjusting Rewards for Average Reward Setting

Background context explaining the concept of removing specific characters () and adjusting rewards to calculate the average reward. The formulas provided detail how state values \(v_\pi(s)\), action-state values \(q_\pi(s, a)\), optimal state value \(v^\ast(s)\), and optimal action-state value \(q^\ast(s, a)\) are adjusted.

Formula for state value:
\[ v_\pi(s)=\sum_a \pi(a|s) \sum_r \sum_{s'} p(s',r|s, a)( r - r(\pi)+v_\pi(s') ) \]

Formula for action-state value:
\[ q_\pi(s, a)=\sum_r \sum_{s'} p(s',r|s, a)\left( r - r(\pi) + \sum_{a'} \pi(a'|s')q_\pi(s',a') \right) \]

Optimal state and action-state values:
\[ v^\ast(s) = \max_a \sum_r \sum_{s'} p(s',r|s, a)( r - \max_\pi r(\pi)+v^\ast(s') ) \]
\[ q^\ast(s, a)=\sum_r \sum_{s'} p(s',r|s, a)\left( r - \max_\pi r(\pi) + \max_a q^\ast(s',a) \right) \]

Additional differential forms of the two TD errors:
\[ \delta_t = R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, w_t) - \hat{v}(S_t, w_t), (10.10) \]
\[ \delta_t' = R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t), (10.11) \]

Where \(\bar{R}_t\) is an estimate of the average reward \(r(\pi)\).

The differential version of semi-gradient Sarsa uses:
\[ w_{t+1} = w_t + \alpha \delta_t q(S_t, A_t, w_t), (10.12) \]

where \(\delta_t\) is given by (10.11).

:p How do state and action values change in the average reward setting?
??x
In the average reward setting, both state values \(v_\pi(s)\) and action-state values \(q_\pi(s, a)\) are adjusted to account for the difference between the received reward and the true average reward. This adjustment helps in making decisions that are not solely based on immediate rewards but also consider long-term outcomes.

For state values:
\[ v_\pi(s)=\sum_a \pi(a|s) \sum_r \sum_{s'} p(s',r|s, a)( r - r(\pi)+v_\pi(s') ) \]

Here, \(r(\pi)\) represents the average reward under policy \(\pi\), and the term \(r - r(\pi)\) adjusts for the difference between the actual received reward and the expected long-term reward.

For action-state values:
\[ q_\pi(s, a)=\sum_r \sum_{s'} p(s',r|s, a)\left( r - r(\pi) + \sum_{a'} \pi(a'|s')q_\pi(s',a') \right) \]

The differential form of the TD errors is used to update the weights in algorithms like semi-gradient Sarsa and Q-learning:
```java
// Example pseudocode for updating weights using differential TD error
w[t+1] = w[t] + alpha * (R[t+1] - mean_reward[t] + v_hat(S[t+1], w[t]) - v_hat(S[t], w[t]))
```

x??

---

#### Differential Semi-Gradient Sarsa Algorithm

Background context explaining the differential version of semi-gradient Sarsa algorithm. The objective is to use a differentiable action-value function parameterization and adjust the weights based on the difference between actual and expected rewards.

Formula for weight update:
\[ w_{t+1} = w_t + \alpha \delta_t q(S_t, A_t, w_t), (10.12) \]

where \(\delta_t\) is given by:
\[ \delta_t = R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t), (10.10) \]

Here, \(\bar{R}_t\) is an estimate of the average reward.

:p What is the weight update rule for differential semi-gradient Sarsa?
??x
The weight update rule for differential semi-gradient Sarsa involves adjusting the weights based on the difference between the actual and expected rewards. The update is performed using the TD error \(\delta_t\), which captures this difference:

\[ w_{t+1} = w_t + \alpha \delta_t q(S_t, A_t, w_t) \]

Where:
- \(w_t\) are the current weights.
- \(\alpha\) is the learning rate.
- \(\delta_t = R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t)\)

The term \(R_{t+1} - \bar{R}_t\) represents the difference between the actual reward and the average reward estimate at time step \(t\). The term \(\hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t)\) captures how well the current action-value function approximates the target.

This update rule ensures that the algorithm adapts to both immediate rewards and long-term outcomes by incorporating the difference between expected and actual rewards.

```java
// Pseudocode for differential semi-gradient Sarsa
w[t+1] = w[t] + alpha * (R[t+1] - mean_reward[t] + q_hat(S[t+1], A[t+1], w[t]) - q_hat(S[t], A[t], w[t]))
```

x??

---

#### Differential Version of Q-Learning

Background context explaining the differential version of Q-learning. This involves adjusting the weights based on the difference between actual and expected rewards.

Formula for weight update:
\[ w_{t+1} = w_t + \alpha \delta_t q(S_t, A_t, w_t), (10.12) \]

where \(\delta_t\) is given by:
\[ \delta_t = R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t), (10.11) \]

Here, \(\bar{R}_t\) is an estimate of the average reward.

:p How does differential Q-learning update its weights?
??x
In the differential version of Q-learning, the weights are updated based on the difference between actual and expected rewards. The weight update rule is similar to that used in differential semi-gradient Sarsa:

\[ w_{t+1} = w_t + \alpha \delta_t q(S_t, A_t, w_t) \]

where:
- \(w_t\) are the current weights.
- \(\alpha\) is the learning rate.
- \(\delta_t = R_{t+1} - \bar{R}_t + \hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t)\)

The term \(R_{t+1} - \bar{R}_t\) represents the difference between the actual reward and the average reward estimate at time step \(t\). The term \(\hat{q}(S_{t+1}, A_{t+1}, w_t) - \hat{q}(S_t, A_t, w_t)\) captures how well the current action-value function approximates the target.

This update rule ensures that the algorithm adapts to both immediate rewards and long-term outcomes by incorporating the difference between expected and actual rewards.

```java
// Pseudocode for differential Q-learning
w[t+1] = w[t] + alpha * (R[t+1] - mean_reward[t] + q_hat(S[t+1], A[t+1], w[t]) - q_hat(S[t], A[t], w[t]))
```

x??

---

#### Example of a Markov Reward Process with Ring States

Background context explaining the example of a Markov reward process where states form a ring. The process involves state transitions going deterministically around the ring, and rewards are received based on specific conditions.

The ring consists of three states \(A\), \(B\), and \(C\). State transitions go from one to another in a deterministic manner: \(A \rightarrow B \rightarrow C \rightarrow A\). The reward is +1 upon arrival at state \(A\) and 0 otherwise.

:p What are the differential values of the three states in this example?
??x
In the given Markov reward process with a ring of three states (\(A\), \(B\), \(C\)), the differential value (or average reward) for each state can be calculated by considering the long-term behavior and rewards received.

- For State \(A\):
  - If in state \(A\), the only transition is to state \(B\) with a reward of +1.
  - The differential value of \(A\) would consider that once entering \(A\), it will stay there infinitely on average, thus the differential value for \(A\) can be set as 1 (since we get an immediate reward of +1 upon arrival at \(A\)).

- For State \(B\):
  - If in state \(B\), it transitions to state \(C\).
  - The differential value of \(B\) would also need to account for the long-term average, and since entering \(B\) does not provide an immediate reward but rather moves towards state \(A\), the differential value of \(B\) is expected to be less than that of \(A\).

- For State \(C\):
  - If in state \(C\), it transitions back to state \(A\).
  - Similarly, the differential value of \(C\) would also need to consider the long-term average and the fact that entering \(C\) does not provide an immediate reward but moves towards state \(A\).

In summary:
- Differential Value for State \(A\) = 1
- Differential Values for States \(B\) and \(C\) will be less than 1, likely closer to 0.5 due to their cyclic nature.

Thus, the differential values of the states are approximately:
- \(A \approx 1\)
- \(B \approx 0.5\)
- \(C \approx 0.5\)

x??

---

#### Access-Control Queuing Task Example

Background context explaining an access-control queuing task where customers with different priorities arrive at a single queue and are assigned to servers based on their priority.

The objective is to serve customers by providing them the highest possible reward, which depends on their priority (1, 2, 4, or 8).

:p What rewards do customers of each priority level receive?
??x
In the access-control queuing task, customers arrive at a single queue with different priorities and are served by one of ten servers. The rewards for each customer depend on their priority as follows:

- Customers of Priority 1: Receive a reward of 1.
- Customers of Priority 2: Receive a reward of 2.
- Customers of Priority 3: Receive a reward of 4.
- Customers of Priority 4: Receive a reward of 8.

The queue never empties, and the priorities of customers in the queue are randomly distributed. If a server is busy and no free servers are available, a customer will be rejected with a reward of zero.

Thus, the rewards for each priority level are:
- Priority 1: Reward = 1
- Priority 2: Reward = 2
- Priority 3: Reward = 4
- Priority 4: Reward = 8

x??

---

#### Differential Semi-Gradient Sarsa Algorithm
Background context: The text describes a scenario where customers with varying priorities are queued up, and decisions must be made on whether to accept or reject them based on the number of free servers. This is modeled as an MDP (Markov Decision Process) problem, specifically using differential semi-gradient Sarsa for decision-making.

The algorithm used here is **Semi-Gradient Sarsa** with a step-size parameter \(\alpha = 0.01\), eligibility trace parameter \(\lambda = 0.01\), and exploration rate \(\epsilon = 0.1\). The goal is to maximize the long-term reward without discounting.

:p What is the semi-gradient Sarsa algorithm used for in this context?
??x
The semi-gradient Sarsa algorithm is employed here to make decisions on accepting or rejecting customers based on their priority and the availability of free servers, aiming to optimize the long-term reward.
x??

---

#### Policy and Value Function from Semi-Gradient Sarsa
Background context: Figure 10.5 illustrates the policy and value function obtained after running semi-gradient Sarsa for 2 million steps. The policy dictates whether a customer should be accepted or rejected based on the number of free servers and their priority, while the value function estimates the expected future rewards.

:p What does Figure 10.5 show?
??x
Figure 10.5 shows the policy and value function resulting from running differential semi-gradient Sarsa for 2 million steps in the access-control queuing task.
x??

---

#### Deterministic Reward Sequence MDP
Background context: Exercise 10.7 describes an MDP where rewards are deterministic, forming a repeating sequence of +1 and 0.

Formula: The average reward \( R \) is defined as:
\[
R = \lim_{N \to \infty} \frac{1}{N} \sum_{t=0}^{N-1} r_t
\]

:p What is the average reward for this MDP?
??x
The average reward for this MDP, which produces a deterministic sequence of +1 and 0, is \( R = \frac{1}{2} \).
x??

---

#### Value Function with Different Starting Points
Background context: Exercise 10.7 further explores the value function when starting from different initial states in an MDP that alternates between +1 and 0 rewards.

Formula for the value of a state \( s \) under policy \(\pi\):
\[
v_\pi(s) = \lim_{N \to \infty} E_\pi \left[ \frac{1}{N} \sum_{t=0}^{N-1} R_t \mid S_0 = s \right]
\]

:p What are the values of states A and B in this MDP?
??x
The value of state A, which starts with a +1 reward, is \( v_\pi(A) = 1 \), while the value of state B, which starts with a 0 reward, is \( v_\pi(B) = 0.5 \).
x??

---

#### Update Mechanism for Average Reward
Background context: Exercise 10.8 discusses the difference between updating the average reward using \( R_{t+1} - \bar{R}_t \) versus just \( R_{t+1} \).

Formula:
\[
\bar{R}_{t+1} = \bar{R}_t + \alpha_t (R_{t+1} - \bar{R}_t)
\]

:p Why is using \( \Delta t = R_{t+1} - \bar{R}_t \) better than just \( R_{t+1} \)?
??x
Using \( \Delta t = R_{t+1} - \bar{R}_t \) helps in stabilizing the learning process and reducing oscillations. It provides a more accurate update by considering the difference from the current average, leading to better convergence.
x??

---

#### Ring MRP Example for Average Reward
Background context: The ring MRP of three states is mentioned as an example where the estimate of the average reward should tend towards 1/3.

:p What does the formula in equation (10.13) suggest?
??x
The formula in equation (10.13) suggests that the value of a state can be defined as:
\[
v_\pi(s) = \lim_{N \to \infty} E_\pi \left[ \frac{1}{N} \sum_{t=1}^{N} R_t \mid S_0 = s \right]
\]
This formula provides a way to compute the value of states in non-ergodic MDPs by considering the limit of the average reward over time.
x??

---

