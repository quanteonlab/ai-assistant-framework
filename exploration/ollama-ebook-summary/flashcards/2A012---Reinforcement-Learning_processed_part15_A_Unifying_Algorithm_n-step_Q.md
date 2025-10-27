# Flashcards: 2A012---Reinforcement-Learning_processed (Part 15)

**Starting Chapter:** A Unifying Algorithm n-step Q

---

#### n-step Q(λ) Algorithm Overview
The algorithm unifies three different kinds of action-value algorithms: n-step Sarsa, n-step Tree Backup, and n-step Expected Sarsa. It allows for a continuous variation between sampling (as in Sarsa) and expectation (as in tree-backup). This is achieved by introducing a parameter λ that controls the degree of sampling on each step.

:p What is the n-step Q(λ) algorithm designed to unify?
??x
The n-step Q(λ) algorithm unifies three different action-value algorithms: n-step Sarsa, n-step Tree Backup, and n-step Expected Sarsa. It allows for a smooth transition between fully sampling actions (as in Sarsa) and considering the expectation over all possible actions (as in tree-backup). The parameter λ controls this balance.

---
#### Step-by-Step Implementation of n-step Q(λ)
The algorithm uses the n-step return formula to update action values. For each step t, it decides whether to sample or use expectations based on λ.

:p How does the n-step Q(λ) algorithm handle sampling and expectation?
??x
In n-step Q(λ), at each step \(t\), a decision is made between sampling an action (as in Sarsa) and considering the expected value over all actions. This decision is controlled by the parameter λ, which ranges from 0 to 1. If \(\lambda = 1\), it fully samples the action; if \(\lambda = 0\), it only considers the expected value.

The n-step return for Q(λ) can be written as:
\[ G_{t:h} = R_{t+1} + \lambda \left[ \theta_t (G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1})) + (1 - \theta_t) \bar{V}_{h-1}(S_{t+1}) \right] \]
where \( \theta_t = 1 \) if a sample is taken, and \( \theta_t = 0 \) otherwise. The term \(\bar{V}_{h-1}(S_{t+1})\) represents the expected value.

Example pseudocode for updating Q(λ):
```python
for t in range(T-2, -1, -1): # T is the terminal state
    if random() < λ:  # Decide to sample or not based on λ
        action = choose_action(S[t+1])  # Sample an action
        G[t] = R[t+1] + (G[t+1] if t < T-2 else 0) - Q[S[t], A[t]]
    else:
        G[t] = R[t+1] + λ * expected_value(S[t+1])  # Use expectation

Q[S[τ], A[τ]] += α * (G[τ] - Q[S[τ], A[τ]])
```

x??

---
#### n-step Tree Backup Algorithm
The tree-backup algorithm updates the value of a state-action pair by considering all possible actions at each step. It does not sample action values but uses the expected value directly.

:p How is the tree-backup algorithm different from Sarsa in terms of updating Q-values?
??x
Unlike Sarsa, which samples an action and follows it to get the return, the tree-backup algorithm considers all possible actions at each step. It updates the state-action value by summing up expected values for all actions and states encountered during a trajectory.

The n-step tree backup update can be written as:
\[ G_{t:h} = R_{t+1} + \lambda \left[ \theta_t (G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1})) + (1 - \theta_t) \bar{V}_{h-1}(S_{t+1}) \right] \]
where for tree-backup, the term involving \( G_{t+1:h} \) is not present since it doesn't sample an action. The update simplifies to:
\[ Q(S_t, A_t) = Q(S_t, A_t) + \alpha [R_{t+1} + \lambda \bar{V}(S_{t+1}) - Q(S_t, A_t)] \]

x??

---
#### n-step Sarsa Algorithm
n-step Sarsa updates the value of a state-action pair based on the actual action taken. It follows the chosen action to get the return and uses this information to update the Q-value.

:p How does n-step Sarsa differ from tree-backup in its approach to updating Q-values?
??x
n-step Sarsa differs from tree-backup by sampling actions as they are taken during the episode. For each step, it updates the state-action value based on the actual action chosen and follows this action to get the return.

The update for n-step Sarsa can be written as:
\[ Q(S_t, A_t) = Q(S_t, A_t) + \alpha [R_{t+1} + \lambda (G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1}))] \]
where \( G_{t+1:h} \) is the n-step return from step \( t+1 \).

Example pseudocode for updating n-step Sarsa:
```python
for t in range(T-2, -1, -1):
    action = choose_action(S[t+1])
    G[t] = R[t+1] + (G[t+1] if t < T-2 else 0) - Q[S[t], A[t]]
Q[S[τ], A[τ]] += α * G[τ]
```

x??

---
#### n-step Expected Sarsa Algorithm
Expected Sarsa updates the value of a state-action pair by considering all possible actions and their probabilities. It is similar to tree-backup but takes an expectation over all possible actions.

:p How does expected Sarsa differ from n-step Sarsa in terms of action selection?
??x
In expected Sarsa, instead of sampling a single action as in n-step Sarsa, it considers the expected value over all possible actions. The update equation for expected Sarsa includes a summation over all possible actions with their probabilities.

The update can be written as:
\[ Q(S_t, A_t) = Q(S_t, A_t) + \alpha [R_{t+1} + \lambda (G_{t+1:h} - \sum_a \pi(a|S_{t+1}) Q_{h-1}(S_{t+1}, a))] \]

x??

---
#### n-step Bootstrapping and Tree Backup
n-step tree backup updates the value of a state-action pair by considering all possible actions at each step. It does not sample action values but uses the expected value directly.

:p What is the key difference between n-step tree backup and n-step Sarsa in their approach to updating Q-values?
??x
The key difference lies in how they handle action selection during updates:
- **Tree Backup**: Uses a fixed policy (often \(\pi\)) to determine actions at each step without sampling. It sums the expected values of all possible actions.
- **Sarsa**: Samples an actual action taken during the episode and follows it to get the return.

Example pseudocode for tree-backup update:
```python
for t in range(T-2, -1, -1):
    G[t] = R[t+1] + λ * expected_value(S[t+1])  # No sampling, uses expected value directly
Q[S[τ], A[τ]] += α * (G[τ] - Q[S[τ], A[τ]])
```

x??

---

#### n-step Temporal-Difference Methods Overview
This chapter introduces a range of methods that combine elements of both one-step temporal-difference (TD) and Monte Carlo techniques. These methods involve bootstrapping future rewards, states, or actions up to \(n\) steps into the future. The key idea is that these methods typically outperform either extreme, offering a balanced approach.
:p What are n-step TD methods?
??x
n-step TD methods are learning algorithms that look ahead to the next \(n\) rewards, states, and actions before updating their estimates. They provide a compromise between one-step TD methods and Monte Carlo methods by incorporating an intermediate amount of bootstrapping. This balance often results in better performance than either technique alone.
??x
---

#### 4-step Backup Diagrams
The two diagrams summarize the key n-step methods introduced. One diagram shows state-value updates for \(n\)-step TD with importance sampling, while the other displays action-value updates for \(n\)-step Q(π).
:p What do the 4-step backup diagrams represent?
??x
The first diagram represents the update rule for state-values in \(n\)-step TD methods using importance sampling. The second diagram illustrates the update rule for action-values (\(Q(\pi)\)) in \(n\)-step TD methods, which generalizes Expected Sarsa and Q-learning.
??x
---

#### Time Delay and Computation
These methods require a delay of \(n\) time steps before updating their estimates because all future events needed are only known after these steps. Additionally, they entail more computation per time step compared to one-step methods due to the need for multiple steps ahead in each update.
:p What is the main drawback of n-step TD methods?
??x
The primary drawback of n-step TD methods is the delay before updating estimates and increased computational requirements per time step because they look ahead \(n\) steps. This makes them slower and more resource-intensive compared to one-step methods.
??x
---

#### Memory Requirements
To implement these methods, it's necessary to record states, actions, rewards, and sometimes other variables over the last \(n\) time steps. While this improves performance, it increases memory usage, which can be a significant factor in practical applications.
:p What additional resource requirement do n-step TD methods have?
??x
N-step TD methods require more memory to store states, actions, rewards, and possibly other variables from the last \(n\) time steps. This is necessary for accurate updates but adds to the overall resource demand of the algorithm.
??x
---

#### Multi-Step TD Methods Implementation
In Chapter 12, we will explore how multi-step TD methods can be implemented with minimal memory and computational complexity using eligibility traces. However, even then, there will still be some additional computation beyond one-step methods.
:p What is the future direction for implementing n-step TD methods?
??x
The future direction involves implementing n-step TD methods with minimal memory and computational complexity through the use of eligibility traces. While this can reduce resource demands, some additional computation over one-step methods will still be required.
??x
---

#### Importance Sampling in n-step Methods
One approach to off-policy learning is based on importance sampling. This method updates estimates using weighted samples from a different behavior policy than the target policy. However, it may suffer from high variance if the policies are very different.
:p What does one approach for off-policy learning involve?
??x
Importance sampling is an approach used in off-policy learning that updates estimates based on weighted samples from a behavior policy different from the target policy. While conceptually simple, it can have high variance if the policies differ significantly.
??x
---

#### Tree-Backup Updates
Another method for off-policy learning, tree-backup updates, extends Q-learning to the multi-step case with stochastic target policies. It avoids importance sampling but may only span a few steps even when \(n\) is large if the target and behavior policies are substantially different.
:p What does the other approach for off-policy learning involve?
??x
Tree-backup updates are an approach that extends Q-learning to handle multi-step cases with stochastic target policies without using importance sampling. However, its effectiveness can be limited if the target and behavior policies differ significantly, as it may only span a few steps even when \(n\) is large.
??x
---

#### Models and Planning
Background context: In reinforcement learning, a model of the environment is used to predict its response to actions. This can be either a distribution model that provides all possible outcomes with their probabilities or a sample model that randomly samples one outcome according to these probabilities.

:p What are the two types of models discussed in the chapter?
??x
- Distribution models: These produce a description of all possibilities and their probabilities.
- Sample models: These provide just one possibility, sampled according to the probabilities.
x??

---
#### Planning as a Search Through State Space
Background context: State-space planning involves searching through states for an optimal policy or path to a goal. Actions cause transitions from state to state, and value functions are computed over these states.

:p How does state-space planning differ from plan-space planning?
??x
- State-space planning: It searches the state space for an optimal policy or path.
- Plan-space planning: It searches the space of plans where operators transform one plan into another. However, this method is not extensively covered in the chapter due to its difficulty in application to stochastic sequential decision problems.
x??

---
#### Dynamic Programming and Value Functions
Background context: Dynamic programming methods use value functions as a key intermediate step towards improving policies. These values are computed by applying backup operations (or updates) to simulated experience.

:p What is the common structure shared by all state-space planning methods according to the chapter?
??x
All state-space planning methods involve:
1. Computing value functions as an intermediate step.
2. Using these value functions to update policies through backups or updates applied to simulated experience.
The common structure can be diagrammed as follows: values -> backups -> models -> simulated experience -> policy -> updates -> backups.
x??

---
#### Monte Carlo and Temporal-Difference Methods
Background context: The chapter discusses the integration of model-based methods (dynamic programming) with model-free methods (Monte Carlo, temporal-difference). Earlier chapters presented these separately but now focus on how they can be unified.

:p How are Monte Carlo and temporal-difference methods typically described in earlier parts of the book?
??x
In earlier parts, Monte Carlo and temporal-difference methods were presented as distinct alternatives. The chapter aims to unify them using n-step methods.
x??

---
#### Unifying Model-Based and Model-Free Methods
Background context: The goal is to integrate model-based methods (which rely on planning) with model-free methods (which primarily rely on learning). Both types of methods compute value functions, but they do so in different ways.

:p What are the two primary components that distinguish model-based from model-free reinforcement learning methods?
??x
- Model-based: Rely heavily on planning.
- Model-free: Primarily rely on learning.
Both methods involve computing value functions and using backups or updates to update these values based on simulated experience.
x??

---
#### Simulation of Experience Using Models
Background context: Models can be used to simulate the environment. A sample model produces a single transition, while a distribution model provides all possible transitions weighted by their probabilities.

:p How do models produce simulated experience?
??x
- Sample Model: Produces one possible state and reward based on the probability distribution.
- Distribution Model: Provides all possible states and rewards along with their probabilities of occurrence.
This is done to mimic or simulate actual environment interactions, which can be used for planning and learning.
x??

---
#### Planning in Artificial Intelligence
Background context: There are two approaches to planning in AI: state-space planning and plan-space planning. The chapter focuses on state-space planning.

:p What is the difference between state-space planning and plan-space planning?
??x
- State-Space Planning: Search through states for an optimal policy or path.
- Plan-Space Planning: Search through plans, where operators transform one plan into another. It includes methods like evolutionary algorithms and partial-order planning.
The chapter does not delve deeply into plan-space planning due to its complexity in handling stochastic decision problems.
x??

---

