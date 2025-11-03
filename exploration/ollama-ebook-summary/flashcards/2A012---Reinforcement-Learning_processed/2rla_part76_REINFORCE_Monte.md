# Flashcards: 2A012---Reinforcement-Learning_processed (Part 76)

**Starting Chapter:** REINFORCE Monte Carlo Policy Gradient

---

#### Policy Gradient Theorem for Episodic Case

Background context: The policy gradient theorem provides a way to compute the gradient of performance with respect to the policy parameter without involving the derivative of the state distribution. This is crucial for applying stochastic gradient ascent (13.1) in reinforcement learning.

Relevant formulas:
\[ \frac{\partial J(\theta)}{\partial \theta} = \sum_{s, a} \mu(s) q_\pi(s, a) r_\pi(a|s, \theta), \]
where \(J(\theta)\) is the performance measure with respect to parameter vector \(\theta\).

:p What does the policy gradient theorem provide in terms of computing gradients?
??x
The policy gradient theorem provides an expression for the gradient of the performance measure with respect to the policy parameters without needing to compute derivatives involving state distributions. This allows us to use stochastic gradient ascent methods effectively.
x??

---

#### REINFORCE Algorithm

Background context: The REINFORCE algorithm is a Monte Carlo policy gradient method that updates the policy parameter based on sampled returns from episodes.

Relevant formulas:
\[ \frac{\partial J(\theta)}{X s\mu(s)X aq_\pi(s, a)r_\pi(a|s, \theta)} = E_\pi \left[ q_\pi(S_t, A_t) r_\pi(A_t | S_t, \theta) \right], \]
where \(G_t\) is the return at time step \(t\).

:p What is the basic idea behind the REINFORCE algorithm?
??x
The REINFORCE algorithm updates the policy parameter based on the product of a return and the gradient of the probability of taking the action actually taken. This update direction ensures that actions leading to high returns are more likely in future episodes.
x??

---

#### REINFORCE Update Equation

Background context: The REINFORCE update equation is derived from the policy gradient theorem and involves sampling returns directly.

Relevant formulas:
\[ \theta_{t+1} = \theta_t + \alpha G_t r_\pi(A_t | S_t, \theta_t) / \pi(A_t | S_t, \theta_t). \]

:p What does the REINFORCE update equation look like?
??x
The REINFORCE update equation is:
\[ \theta_{t+1} = \theta_t + \alpha G_t r_\pi(A_t | S_t, \theta_t) / \pi(A_t | S_t, \theta_t). \]
This equation adjusts the policy parameter in a direction that increases the probability of actions leading to high returns.
x??

---

#### REINFORCE Pseudocode

Background context: The pseudocode for REINFORCE provides a step-by-step implementation of the algorithm.

Relevant code:
```java
REINFORCE: Monte-Carlo Policy-Gradient Control (episodic)
Input: a differentiable policy parameterization π(a|s, θ)
Algorithm parameter: step size α > 0
Initialize policy parameter θ ∈ Rd (e.g., to 0)
Loop forever (for each episode):
    Generate an episode S0, A0, R1,..., ST-1, AT-1, RT, following π(·|·,θ)
    Loop for each step of the episode t = 0,1,...,T - 1:
        Gt ← PT k=t+1 (Rk) 
        θt+1 ← θt + α Gt rlnπ(At | St, θt)
```

:p What is the REINFORCE pseudocode?
??x
The REINFORCE pseudocode involves generating episodes and updating the policy parameter based on sampled returns:
```java
REINFORCE: Monte-Carlo Policy-Gradient Control (episodic)
Input: a differentiable policy parameterization π(a|s, θ)
Algorithm parameter: step size α > 0
Initialize policy parameter θ ∈ Rd (e.g., to 0)
Loop forever (for each episode):
    Generate an episode S0, A0, R1,..., ST-1, AT-1, RT, following π(·|·,θ)
    Loop for each step of the episode t = 0,1,...,T - 1:
        Gt ← PT k=t+1 (Rk) 
        θt+1 ← θt + α Gt rlnπ(At | St, θt)
```
x??

---

#### Eligibility Vector

Background context: The eligibility vector is a key component in the REINFORCE update rule.

Relevant code:
```java
rlnπ(At | St, θ) = rπ(At | St, θ) / π(At | St, θ)
```

:p What is an eligibility vector?
??x
An eligibility vector is a component in the REINFORCE update rule that represents the gradient of the probability of taking the action actually taken divided by the probability of taking that action. It is used to determine the direction in parameter space that increases the likelihood of repeating the action.
x??

---

#### Discounted Case

Background context: The algorithms can be extended to handle discounted returns, but the non-discounted case is considered here.

Relevant formulas:
\[ \theta_{t+1} = \theta_t + \alpha G_t r_\pi(A_t | S_t, \theta_t) / \pi(A_t | S_t, \theta_t). \]

:p How does the REINFORCE algorithm handle discounted returns?
??x
The REINFORCE algorithm can be adjusted to handle discounted returns by including a discount factor \(\gamma\):
\[ G_t = \sum_{k=t}^{T-1} \gamma^{k-t+1} R_k. \]
However, for simplicity and focus on the main ideas, the non-discounted case is often considered.
x??

---

#### Performance of REINFORCE

Background context: The performance of REINFORCE can be demonstrated through simulations on specific environments.

Relevant data:
Figure 13.1 shows the total reward per episode approaching the optimal value with a good step size in the short-corridor gridworld example.

:p How does REINFORCE perform in practice?
??x
In practice, REINFORCE performs well by gradually improving the policy parameters to achieve higher returns over episodes, as shown in Figure 13.1 for the short-corridor gridworld. With an appropriate step size, the total reward per episode can approach the optimal value of the start state.
x??

---

#### REINFORCE Method Overview
Background context: REINFORCE is a stochastic gradient method used for policy gradient methods. It updates the policy parameter to improve performance by moving in the direction of the performance gradient. The update rule for REINFORCE without a baseline is given as:
\[ \theta_{t+1} = \theta_t + \alpha G_t r(\pi(a|s, \theta_t)) \]
where \( G_t \) is the discounted return from time step \( t \).

:p What does the REINFORCE method do?
??x
REINFORCE updates the policy parameters to improve performance by moving in the direction of the performance gradient. It is a Monte Carlo method and may be slow due to high variance.
x??

---

#### Eligibility Vector for Softmax Policy
Background context: For a policy parameterized using softmax with linear action preferences, the eligibility vector \( \mathcal{E}(s,a) \) can be calculated as:
\[ r\ln \pi(a|s,\theta) = x(s,a) \sum_{b} \pi(b|s,\theta) x(s,b) \]

:p Prove that the eligibility vector is given by the formula provided.
??x
Given the policy parameterization using softmax with linear action preferences, we can derive the eligibility vector as follows:
1. Start from the definition of the log probability: \( r\ln \pi(a|s,\theta) = x(s,a) - \sum_{b} \pi(b|s,\theta) x(s,b) \).
2. Simplify to get the final form: \( r\ln \pi(a|s,\theta) = x(s,a) \sum_{b} \pi(b|s,\theta) x(s,b) \).

This derivation uses the properties of softmax and linearity.
x??

---

#### Policy Gradient Theorem with Baseline
Background context: The policy gradient theorem can be extended to include a comparison with an arbitrary baseline \( b(s) \). This is represented by:
\[ \frac{\partial J(\theta)}{\partial \theta} = \sum_{s,\pi} \mu(s)\sum_a [q_\pi(s,a) - b(s)] r\pi(a|s, \theta) \]

:p Explain the policy gradient theorem with a baseline.
??x
The policy gradient theorem with a baseline \( b(s) \) modifies the update rule to:
\[ \nabla J(\theta) = \sum_{s} \mu(s) \sum_a [q_\pi(s,a) - b(s)] r\pi(a|s, \theta) \]
where \( q_\pi(s,a) \) is the action value function. The baseline can be any function and helps in reducing variance.

This update rule results in a new version of REINFORCE that includes a general baseline:
\[ \theta_{t+1} = \theta_t + \alpha [G_t - b(S_t)] r\pi(A_t|S_t, \theta_t) \]

The baseline can help reduce the variance and speed up learning.
x??

---

#### REINFORCE with Baseline Algorithm
Background context: REINFORCE with a baseline uses a learned state-value function \( \hat{v}(s,\omega) \) to estimate the value of states. The algorithm updates both policy parameters and state-value weights.

:p Describe the pseudocode for REINFORCE with baseline.
??x
```python
# Pseudocode for REINFORCE with Baseline (Episodic)
def REINFORCE_with_Baseline():
    Initialize policy parameter θ to 0
    Initialize state-value weights ω to 0
    For each episode:
        Generate an episode following π(·|·, θ)
        For each step in the episode:
            Calculate G_t = ∑_{k=t+1}^{T} r_k
            Update state-value weights: ω = ω + α_ω * [G_t - v_hat(S_t, ω)]
            Update policy parameters: θ = θ + α_θ * [G_t - v_hat(S_t, ω)] * ln(π(A_t|S_t, θ))
```

This algorithm uses two step sizes \( \alpha_\theta \) and \( \alpha_\omega \). The state-value function is updated based on the difference between the discounted return and its estimate.
x??

---

#### Importance of Baseline in REINFORCE
Background context: A baseline can significantly reduce the variance in updates, making learning faster. For example, in gradient bandits, a simple average reward acts as a baseline.

:p Why is a baseline important in REINFORCE?
??x
A baseline helps reduce the variance of the policy update by subtracting a component that does not change with actions. This can be particularly useful when all actions have similar values, making it harder to distinguish between them.

For instance, if using a linear state-value function as a baseline:
\[ \theta_{t+1} = \theta_t + \alpha [G_t - v(S_t, w)] r\pi(A_t|S_t, \theta_t) \]
where \( v(S_t, w) \) is the estimated value of the state.

Using such a baseline can significantly improve learning speed by reducing noise in the gradient estimates.
x??

---

#### Actor–Critic Methods Overview
Actor–Critic methods combine aspects of policy gradient and value-based methods. They learn both a policy (actor) and a state-value function, but the latter is used as a critic for bootstrapping rather than directly estimating values. This approach helps in reducing variance compared to plain REINFORCE.

:p What are actor–critic methods and how do they differ from REINFORCE?
??x
Actor–Critic methods are a type of reinforcement learning algorithm that combines elements of policy gradient and value-based methods. Unlike REINFORCE, which uses only the return (full or one-step) to update the policy, actor–critic methods use an additional state-value function as a critic for bootstrapping. This means the value estimate is updated based on the estimated values from future states, reducing variance in learning.

In REINFORCE with baseline:
```java
// Pseudocode for REINFORCE with baseline
for each episode {
    S = initial state
    while not terminal {
        A = choose action according to policy(·|S)
        S', R = take action A and observe result
        G = sum of rewards from current state onwards
        w += stepSize * (G - v(S, w))
        S = S'
    }
}
```
In actor–critic methods:
```java
// Pseudocode for one-step Actor-Critic method
for each episode {
    S = initial state
    while not terminal {
        A = choose action according to policy(·|S)
        S', R = take action A and observe result
        G = R + v(S', w)  // One-step return
        w += stepSize * (G - v(S, w))
        ✓ += stepSize * G * log(policy(A|S))
        S = S'
    }
}
```
x??

---

#### REINFORCE with Baseline
REINFORCE with a baseline uses an approximate state-value function to reduce the variance of policy gradients. This method does not directly use the value function for bootstrapping but instead employs it as a baseline to stabilize learning.

:p What is REINFORCE with a baseline and how does it differ from plain REINFORCE?
??x
REINFORCE with a baseline uses an approximate state-value function (ˆv(s, w)) to reduce the variance in policy gradients. The update rule for the policy parameter (✓) becomes:
```java
// Pseudocode for REINFORCE with baseline
for each episode {
    S = initial state
    while not terminal {
        A = choose action according to policy(·|S)
        S', R = take action A and observe result
        G = sum of discounted rewards from current state onwards
        ✓ += stepSize * (G - v(S, w)) * log(policy(A|S))
        S = S'
    }
}
```
In contrast, plain REINFORCE updates the policy parameter as follows:
```java
// Pseudocode for plain REINFORCE
for each episode {
    S = initial state
    while not terminal {
        A = choose action according to policy(·|S)
        S', R = take action A and observe result
        G = sum of discounted rewards from current state onwards
        ✓ += stepSize * G * log(policy(A|S))
        S = S'
    }
}
```
The key difference is the use of a baseline (v(S, w)) in REINFORCE with baseline to reduce the variance.

x??

---

#### One-Step Actor–Critic Method Details
One-step actor–critic methods update both the policy and state-value function based on one step ahead predictions. They are fully online and incremental, avoiding the complexities of eligibility traces.

:p What is a one-step actor–critic method and how does it update parameters?
??x
A one-step actor–critic method updates the policy (✓) and state-value function (ˆv(s, w)) based on one step ahead predictions. The policy parameter (✓) is updated using:
```java
// Update rule for policy parameter in one-step actor-critic
✓ += stepSize * G * log(policy(A|S))
```
Where G is the one-step return, and the state-value function (ˆv(s, w)) is used as a baseline to update the policy. The state-value function parameters (w) are updated using:
```java
// Update rule for state-value function parameter in one-step actor-critic
w += stepSize * (G - ˆv(S, w))
```
The complete algorithm is given by the pseudocode provided earlier.

x??

---

#### Actor–Critic with Eligibility Traces
Actor–Critic methods with eligibility traces use separate eligibility traces for the policy and state-value function to account for temporal dependencies. This approach provides a more flexible way of bootstrapping values over multiple time steps.

:p What is actor–critic method with eligibility traces and how does it update parameters?
??x
Actor–Critic methods with eligibility traces use eligibility traces to account for temporal dependencies in policy and state-value function updates. The policy parameter (✓) and state-value function (ˆv(s, w)) are updated using separate eligibility traces (z✓, zw).

The policy parameter is updated as:
```java
// Update rule for policy parameter with eligibility traces
✓ += stepSize * z✓ * log(policy(A|S))
```
And the state-value function parameters as:
```java
// Update rule for state-value function parameter with eligibility traces
w += stepSize * zw
```
The complete algorithm is given by the pseudocode provided earlier.

x??

---

