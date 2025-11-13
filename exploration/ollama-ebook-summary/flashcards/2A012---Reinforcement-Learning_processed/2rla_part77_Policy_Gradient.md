# Flashcards: 2A012---Reinforcement-Learning_processed (Part 77)

**Starting Chapter:** Policy Gradient for Continuing Problems

---

#### Continuing Problems Definition and Performance Measure
Background context: In continuing problems, episodes do not have natural boundaries. The performance of a policy is measured by its average rate of reward per time step, denoted as $J(\theta)$.

Relevant formula:
$$J(\theta) = \lim_{t \rightarrow \infty} \frac{1}{t} E[R_t | S_0, A^0:t-1 \sim \pi]$$

Explanation: This formula indicates that the performance is defined as the limit of the average discounted reward over time. The policy $\pi $ is parameterized by$\theta$.

:p What does the performance measure for continuing problems look like?
??x
The performance measure is given by the long-term average rate of reward per time step, which captures the expected total reward normalized by the number of steps.
x??

---

#### Steady-State Distribution and Policy Gradient Context
Background context: For a continuing problem, we need to define the steady-state distribution $\mu(s)$, which describes the probability that the system is in state $ s$when following policy $\pi$.

Relevant formulas:
$$\mu(s) = \lim_{t \rightarrow \infty} P(S_t = s | A^0:t-1 \sim \pi)$$and$$\sum_s \mu(s) \sum_a \pi(a|s, \theta) p(s'|s, a) = \mu(s')$$

Explanation: The steady-state distribution $\mu(s)$ is the long-term probability of being in state $ s $, assuming the system follows policy $\pi $. This condition ensures that if actions are chosen according to $\pi$, the distribution remains unchanged over time.

:p What is the role of the steady-state distribution in a continuing problem?
??x
The steady-state distribution $\mu(s)$ represents the long-term probability of being in state $ s $ when following policy $\pi$. It ensures that if actions are chosen according to $\pi$, the system's state distribution remains stable over time.
x??

---

#### Actor-Critic Algorithm for Continuing Problems
Background context: The actor-critic algorithm is used to optimize policies in continuing problems by updating both a policy and a value function. This method uses eligibility traces to update parameters smoothly.

Relevant pseudocode:
```python
Actor-Critic with Eligibility Traces (continuing), for estimating πθ ⇡ π* 
Input: a differentiable policy parameterization π(a|s, θ) 
Input: a differentiable state-value function parameterization ˆv(s, w)
Algorithm parameters: γ ∈ [0,1], αθ ∈ [0,1], αw > 0, βθ > 0, βR > 0
Initialize R_hat = 0 (e.g., to 0) 
Initialize state-value weights w and policy parameter θ 
Initialize S (e.g., to s0)
z_w = 0 (d-component eligibility trace vector) 
z_θ = 0 (d0-component eligibility trace vector)

Loop forever (for each time step):
    A ~ π(·|S, θ) 
    Take action A, observe S0, R
    R_hat = R_hat + αR (ˆv(S0, w) - ˆv(S, w)) + R
    z_w = γ z_w + R ˆv(S, w)
    z_θ = z_θ + αθ log(π(A|S, θ))
    w = w + βw (z_w)
    θ = θ + βθ (z_θ)
    S = S0
```

Explanation: This pseudocode outlines the actor-critic algorithm for continuing problems. The policy $\pi $ and state-value function$ˆv$ are updated based on eligibility traces to ensure smooth convergence.

:p What does this pseudocode illustrate?
??x
This pseudocode illustrates the actor-critic algorithm for continuing problems, which updates both a policy and a value function using eligibility traces. The policy is adjusted based on log-policy gradients, while the state-value function is updated based on differences in predicted values.
x??

---

#### Policy Gradient Theorem in Continuing Problems
Background context: The policy gradient theorem provides a way to compute the gradient of the performance measure with respect to the policy parameters $\theta$ for continuing problems.

Relevant formula:
$$r J(\theta) = \sum_s \mu(s) \sum_a \pi(a|s, \theta) q_\pi (s, a) + \sum_s \mu(s) \sum_a \pi(a|s, \theta) \sum_{s'} p(s'|s,a) r \frac{\partial}{\partial \theta} v_\pi(s') - r \frac{\partial}{\partial \theta} v_\pi(s)$$

Explanation: This formula relates the gradient of the performance measure $J(\theta)$ to the sum over all states and actions, weighted by the steady-state distribution.

:p What does this theorem state?
??x
This theorem states that the gradient of the performance measure $J(\theta)$ for a policy in a continuing problem can be computed as the sum over all states and actions, weighted by the steady-state distribution. It connects the policy's parameters $\theta$ to the value functions and the state-action values.
x??

---

#### Forward and Backward View Equations
Background context: The forward and backward view equations remain the same in continuing problems.

Relevant formulas:
- Forward view equation:$r J(\theta) = \sum_s \mu(s) \sum_a \pi(a|s, \theta) q_\pi (s, a) + \sum_s \mu(s) \sum_a \pi(a|s, \theta) \sum_{s'} p(s'|s,a) r \frac{\partial}{\partial \theta} v_\pi(s') - r \frac{\partial}{\partial \theta} v_\pi(s)$- Backward view equation:$ r J(\theta) = \sum_s \mu(s) \sum_a \pi(a|s, \theta) q_\pi (s, a) + \sum_s \mu(s) \sum_a \pi(a|s, \theta) \sum_{s'} p(s'|s,a) r \frac{\partial}{\partial \theta} v_\pi(s') - r \frac{\partial}{\partial \theta} v_\pi(s)$Explanation: These equations provide a way to compute the gradient of the performance measure $ J(\theta)$.

:p What are these equations used for?
??x
These equations are used to compute the gradient of the performance measure $J(\theta)$ in continuing problems, allowing the optimization of policies based on their long-term average reward.
x??

---

#### Gaussian Policy Parameterization

Background context explaining the concept. The problem deals with continuous action spaces where actions are chosen from a normal (Gaussian) distribution. This method is used to avoid having to compute probabilities for each possible action, which can be computationally expensive or infeasible when dealing with an infinite number of actions.

The probability density function for a normal distribution is given by:
$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$where $\mu $ and$\sigma^2$ are the mean and variance of the normal distribution respectively.

To parameterize a policy for continuous actions, the action $a $ is drawn from a normal distribution with parameters that depend on the state$s $ and a set of parameters$\theta$. The policy can be defined as:
$$\pi(a|s; \theta) = \frac{1}{\sqrt{2\pi}\sigma(s;\theta)} e^{-\frac{(a-\mu(s;\theta))^2}{2\sigma^2(s;\theta)}}$$where $\mu(s; \theta)$ and $\sigma(s; \theta)$ are the mean and standard deviation of the normal distribution, respectively.

The parameters $\mu(s; \theta)$ and $\sigma(s; \theta)$ can be approximated using function approximators. For instance:
$$\mu(s; \theta_{\mu}) = \theta_{\mu}^T x_\mu(s)$$
$$\sigma(s; \theta_{\sigma}) = e^{\theta_{\sigma}^T x_\sigma(s)}$$where $ x_\mu(s)$and $ x_\sigma(s)$ are state feature vectors.

:p What is the Gaussian policy parameterization?
??x
The Gaussian policy parameterization involves using a normal distribution to model continuous actions. The parameters of this distribution, specifically the mean ($\mu $) and standard deviation ($\sigma $), are approximated by function approximators that depend on the current state $ s $and some learned parameters$\theta_{\mu}$ and $\theta_{\sigma}$.

The policy can be defined as:
$$\pi(a|s; \theta) = \frac{1}{\sqrt{2\pi}\sigma(s;\theta)} e^{-\frac{(a-\mu(s;\theta))^2}{2\sigma^2(s;\theta)}}$$

In practice, the mean and standard deviation are parameterized as follows:
$$\mu(s; \theta_{\mu}) = \theta_{\mu}^T x_\mu(s)$$
$$\sigma(s; \theta_{\sigma}) = e^{\theta_{\sigma}^T x_\sigma(s)}$$

Where $x_\mu(s)$ and $x_\sigma(s)$ are state feature vectors.

---
#### Eligibility Vector for Gaussian Policy

Background context explaining the concept. In policy gradient methods, the eligibility vector is used to accumulate gradients over time to update the parameters of the policy. For a Gaussian policy parameterization, the eligibility vector has two parts: one for the mean and another for the standard deviation.

Given the policy:
$$\pi(a|s; \theta) = \frac{1}{\sqrt{2\pi}\sigma(s;\theta)} e^{-\frac{(a-\mu(s;\theta))^2}{2\sigma^2(s;\theta)}}$$

The eligibility vectors are defined as follows:

For the mean:
$$r_{ln\pi}(a|s, \theta_\mu) = r_\pi(a|s, \theta_\mu) - \frac{1}{\sigma(s; \theta)^2} a \left( a - \mu(s; \theta) \right) x_\mu(s)$$

For the standard deviation:
$$r_{ln\pi}(a|s, \theta_\sigma) = r_\pi(a|s, \theta_\sigma) + \frac{\left( a - \mu(s; \theta) \right)^2}{\sigma^2(s; \theta)} x_\sigma(s)$$

Where $r_\pi(a|s, \theta)$ is the discounted return.

:p What are the parts of the eligibility vector for a Gaussian policy?
??x
The eligibility vector for a Gaussian policy parameterization has two main parts: one for updating the mean and another for updating the standard deviation. Specifically:

For the mean:
$$r_{ln\pi}(a|s, \theta_\mu) = r_\pi(a|s, \theta_\mu) - \frac{1}{\sigma(s; \theta)^2} a \left( a - \mu(s; \theta) \right) x_\mu(s)$$

For the standard deviation:
$$r_{ln\pi}(a|s, \theta_\sigma) = r_\pi(a|s, \theta_\sigma) + \frac{\left( a - \mu(s; \theta) \right)^2}{\sigma^2(s; \theta)} x_\sigma(s)$$

Where:
- $r_\pi(a|s, \theta)$ is the discounted return.
- $\mu(s; \theta)$ and $\sigma(s; \theta)$ are the mean and standard deviation of the Gaussian policy respectively.

This split helps in updating the parameters separately for better convergence.

---
#### Bernoulli Logistic Unit

Background context explaining the concept. The Bernoulli-logistic unit is a stochastic neuron-like unit used in some Artificial Neural Networks (ANNs). It outputs either 0 or 1 based on the input and a learned parameter $\theta$.

The probability $P_t$ of outputting 1 can be expressed as:
$$P_t = \pi(1|S_t, \theta_t) = \frac{1}{1 + e^{-\theta^T x(S_t)}}$$where $ x(S_t)$is the input feature vector and $\theta$ are the weights.

:p What is the Bernoulli-logistic unit?
??x
The Bernoulli-logistic unit, also known as a stochastic neuron-like unit in some ANNs, outputs either 0 or 1 based on its input $x(S_t)$ and learned parameters $\theta$. The probability of outputting 1 is given by the logistic function:
$$P_t = \pi(1|S_t, \theta_t) = \frac{1}{1 + e^{-\theta^T x(S_t)}}$$

This unit helps introduce stochasticity into neural networks, allowing for probabilistic outputs.

#### Policy Gradient Methods Overview
Policy gradient methods are a set of reinforcement learning techniques that directly learn and update policy parameters to improve performance. Unlike action-value methods, which rely on action values, these methods directly optimize policies by adjusting their parameters based on an estimate of the gradient of performance with respect to these parameters.

These methods have several advantages:
- They can output specific probabilities for taking actions.
- They can handle exploration and approach deterministic policies asymptotically.
- They are suitable for continuous action spaces.
- They simplify representation in certain scenarios where policies are easier to parameterize than value functions.

The Policy Gradient Theorem provides a theoretical foundation, offering an exact formula for how performance is affected by policy parameters without involving derivatives of the state distribution. This theorem supports REINFORCE and similar methods.

:p What key characteristic distinguishes policy gradient methods from action-value methods?
??x
Policy gradient methods learn and update policies directly rather than learning action values first.
x??

---

#### Policy Gradient Theorem
The Policy Gradient Theorem (PGT) provides a formula to estimate how changes in the policy parameter affect performance. It states that the gradient of the expected return with respect to the parameters of the policy can be estimated using samples from that policy.

Mathematically, it is expressed as:
$$\nabla_\theta J(\pi_\theta) \approx \mathbb{E}_{s_t \sim \mu, a_t \sim \pi_\theta} [G_t \nabla_\theta \log \pi_\theta(a_t | s_t)]$$where $ J(\pi_\theta)$is the expected return, and $\pi_\theta$ represents the policy parameterized by $\theta$.

:p What does the Policy Gradient Theorem provide in reinforcement learning?
??x
The Policy Gradient Theorem provides a formula for estimating the gradient of performance with respect to policy parameters.
x??

---

#### REINFORCE Method
REINFORCE is a simple yet effective algorithm that follows directly from the Policy Gradient Theorem. It updates policy parameters by taking steps proportional to the product of the observed return and the log probability of the taken actions.

Pseudocode:
```python
for each episode:
    rollout = run_policy()
    G = calculate_return(rollout)
    for t in range(len(rollout)):
        grad_log_pi_t = compute_gradient_of_log_prob(rollout[t])
        policy_gradient = G * grad_log_pi_t
        update_parameters(policy_gradient)
```

:p What is the main advantage of using REINFORCE over other methods?
??x
REINFORCE provides a straightforward way to learn policies directly, avoiding the complexities involved in estimating value functions.
x??

---

#### Actor-Critic Methods Overview
Actor-critic methods combine elements of both policy gradients and temporal difference learning. They split the problem into two components: the actor (which determines actions) and the critic (which evaluates those actions).

The actor learns a policy based on the feedback from the critic, while the critic provides an estimate of the value function to help guide the learning process.

:p How do actor-critic methods differentiate themselves from pure policy gradient methods?
??x
Actor-critic methods incorporate a critic component that evaluates actions, providing more structured guidance compared to purely policy-based methods.
x??

---

#### The Critic and Actor Roles
In actor-critic methods:
- **Critic**: Evaluates the quality of actions taken by the actor. It provides a numerical value or advantage function for each action.
- **Actor**: Determines how to take actions based on the feedback from the critic.

:p What are the roles of the "critic" and "actor" in actor-critic methods?
??x
The critic evaluates the quality of actions, while the actor decides which actions to take based on this evaluation.
x??

---

#### Baseline in Policy Gradient Methods
Adding a baseline (such as an estimated state value function) to REINFORCE can reduce variance without introducing bias. This is done by subtracting the expected return of the baseline from the return.

:p How does adding a baseline help in policy gradient methods?
??x
Adding a baseline reduces the variance of the gradient estimates, making the learning process more stable and efficient.
x??

---

#### Deterministic Policy Gradients
Deterministic policy gradients address continuous action spaces by directly optimizing deterministic policies. This approach simplifies training by avoiding the need for stochastic exploration.

:p What is the key feature of deterministic policy gradients?
??x
Deterministic policy gradients optimize actions deterministically, making them suitable for problems with continuous action spaces.
x??

---

#### Ongoing Research in Policy Gradients
Recent research has explored various extensions and improvements to policy gradient methods. These include natural-gradient methods, deterministic policy gradients, off-policy methods, and entropy regularization.

:p What are some recent developments in the field of policy gradient methods?
??x
Recent developments include natural-gradient methods, deterministic policy gradients, off-policy methods, and entropy regularization.
x??

---

#### Actor-Critic Methods Applications
Actor-critic methods have been successfully applied to complex tasks such as acrobatic helicopter autopilots and in the development of advanced AI systems like AlphaGo.

:p What are some major applications of actor-critic methods?
??x
Major applications include acrobatic helicopter autopilots and projects like AlphaGo.
x??

---

#### Early Work on Policy Gradients
Policy gradient methods were among the earliest studied in reinforcement learning, with significant contributions from researchers such as Witten (1977), Barto, Sutton, and Anderson (1983), and Sutton (1984).

:p Who are some early contributors to policy gradient research?
??x
Early contributors include Witten (1977), Barto, Sutton, and Anderson (1983), and Sutton (1984).
x??

---

