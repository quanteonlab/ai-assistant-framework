# Flashcards: 2A012---Reinforcement-Learning_processed (Part 51)

**Starting Chapter:** Policy Improvement

---

#### Value Function and Policy Evaluation

Background context: The value function \( v_\pi(s) \) for a policy \( \pi \) is defined as the expected return starting from state \( s \) under that policy. The update rule for the value function during iterative policy evaluation is given by:

\[ v_{k+1}(s) = E[R_{t+1} + \gamma v_k(S_{t+1}) | S_t = s, A_t = \pi(s)] \]

Where \( R_{t+1} \) is the reward at time step \( t+1 \), and \( \gamma \) is the discount factor. For an undiscounted task (\( \gamma = 1 \)), this simplifies to:

\[ v_{k+1}(s) = E[R_{t+1} + v_k(S_{t+1}) | S_t = s, A_t = \pi(s)] \]

:p What is the value function \( v_\pi(s) \)?
??x
The value function \( v_\pi(s) \) represents the expected cumulative return starting from state \( s \) and following policy \( \pi \) thereafter. It quantifies how good it is to start in a particular state and follow a specific policy.

In iterative policy evaluation, we update our estimate of the value function iteratively using the Bellman expectation equation until convergence.
x??

---

#### Action-Value Function and Its Approximation

Background context: The action-value function \( q_\pi(s,a) \) for a given state-action pair is defined as the expected return starting from state \( s \), taking action \( a \), and then following policy \( \pi \). It can be expressed using the Bellman equation:

\[ q_\pi(s, a) = E[R_{t+1} + v_\pi(S_{t+1}) | S_t = s, A_t = a] \]

For an undiscounted task (\( \gamma = 1 \)):

\[ q_\pi(s, a) = E[R_{t+1} + v_\pi(S_{t+1}) | S_t = s, A_t = a] \]

:p What is the action-value function \( q_\pi(s,a) \)?
??x
The action-value function \( q_\pi(s,a) \) for a given state-action pair represents the expected return starting from state \( s \), taking action \( a \), and then following policy \( \pi \). It quantifies how good it is to take a specific action in a particular state under the current policy.
x??

---

#### Policy Improvement Theorem

Background context: Given a policy \( \pi \) with value function \( v_\pi(s) \), we can determine if changing the policy at any state would improve the expected return. Specifically, for all states \( s \):

\[ q_\pi(s, \pi(s)) \geq v_\pi(s) \]

If strict inequality holds, a new policy \( \pi' \) that is greedy with respect to \( v_\pi \) will strictly outperform \( \pi \).

The policy improvement theorem states:

- If \( q_\pi(s, \pi'(s)) > v_\pi(s) \), then the new policy \( \pi' \) is better than the original policy \( \pi \).
- The greedy policy defined by:
  \[ \pi'(s) = \arg\max_a q_\pi(s, a) \]
  satisfies this condition and thus guarantees an improvement.

:p What does the policy improvement theorem state?
??x
The policy improvement theorem states that if changing the action in a given state \( s \) to another action \( a \) (where \( q_\pi(s, a) > v_\pi(s) \)) improves the expected return, then using this new greedy policy will yield a better overall policy.

Formally:
- For any deterministic policies \( \pi \) and \( \pi' \), if for all states \( s \):
  \[ q_\pi(s, \pi'(s)) > v_\pi(s) \]
  Then the new policy \( \pi' \) is strictly better than the original policy \( \pi \).

- If there are ties in \( q_\pi(s, a) \), each maximizing action can be given a probability according to some apportioning scheme.
x??

---

#### Policy Improvement for Stochastic Policies

Background context: For stochastic policies \( \pi \) that specify probabilities of taking actions \( a \) in state \( s \), the policy improvement theorem still applies. The greedy policy is defined as:

\[ \pi'(s) = \arg\max_a q_\pi(s, a) \]

Where ties are broken arbitrarily.

The process ensures that the new policy \( \pi' \) will be at least as good as, and often better than, the original policy \( \pi \).

:p What is the stochastic version of the greedy policy?
??x
For stochastic policies, the greedy policy for improvement is defined as:

\[ \pi'(s) = \arg\max_a q_\pi(s, a) \]

Where each action that achieves the maximum value in \( q_\pi(s, a) \) can be given a probability according to some apportioning scheme. This ensures that if there are multiple actions with the same maximum value, they share the probability of being selected.

This stochastic greedy policy guarantees an improvement over the original policy.
x??

--- 

These flashcards cover key concepts from the provided text and should help in understanding their context and implications for reinforcement learning and policy evaluation. Each card focuses on a single concept to ensure clarity and ease of recall during study sessions.

#### Policy Iteration Overview
Policy iteration involves alternating between policy evaluation and policy improvement steps to find an optimal policy. Each iteration guarantees a strict improvement over the previous one, unless it is already optimal.

The process can be summarized as follows:
1. **Initialization**: Start with arbitrary policies for all states.
2. **Policy Evaluation Loop**: Evaluate the current policy iteratively until convergence.
3. **Policy Improvement**: Improve the policy based on the evaluated value function.

:p What is the primary goal of policy iteration?
??x
The primary goal of policy iteration is to find an optimal policy by alternately evaluating and improving policies starting from an arbitrary initial policy. Each step guarantees a strict improvement, leading to convergence in a finite number of iterations.
x??

---

#### Policy Evaluation Step
In each iteration, the value function for the current policy needs to be computed iteratively until it converges.

:p What is the purpose of the policy evaluation loop?
??x
The purpose of the policy evaluation loop is to compute the state-value function \( V^\pi \) for a given policy \( \pi \). This involves iteratively updating the value of each state based on the expected future rewards under that policy until convergence.
x??

---

#### Policy Improvement Step
Policy improvement step checks if there exists an action in the current state that can strictly improve the value function. If such actions exist, the policy is updated.

:p What happens during the policy improvement step?
??x
During the policy improvement step, for each state, we find the action that maximizes the expected future reward under the current value function. If this action differs from the current policy, the policy is updated to use this new action.
x??

---

#### Example: Jack’s Car Rental Problem
This example demonstrates how policy iteration works in a practical scenario involving car rental locations.

:p How does the policy iteration algorithm find an optimal policy for Jack's car rental problem?
??x
The policy iteration algorithm finds an optimal policy by iteratively improving policies starting from an initial arbitrary policy. It alternates between evaluating the current policy to update state-value functions and then improving the policy based on these values. The process continues until no further improvements are possible, indicating optimality.
x??

---

#### Iteration Process in Policy Iteration
The iteration process involves multiple steps of evaluation and improvement.

:p What is the basic structure of the policy iteration algorithm?
??x
The basic structure of the policy iteration algorithm consists of two main phases: 
1. **Policy Evaluation**: Iteratively update state values \( V(s) \) for all states until convergence.
2. **Policy Improvement**: Check if any actions can improve the value function and update the policy accordingly.

If no further improvements are possible, the current policy is optimal.
x??

---

#### Code Example for Policy Iteration
Here’s a simplified pseudocode example to illustrate how policy iteration works:

```pseudocode
function PolicyIteration() {
    V <- initialize_value_function()
    policy_stable <- false
    while not policy_stable do
        # Policy Evaluation
        policy_stable <- true
        for each state s in states do
            old_action <- get_policy(s)
            v <- evaluate_value_function(V, s, policy)
            if |v - V[s]| > epsilon then
                policy_stable <- false
        
        # Policy Improvement
        new_policy <- argmax_a(sum_{s',r} P(s', r|s, a) * (r + V[s']))
        if new_policy != old_action then
            policy <- new_policy
    end while
    return V, policy
}
```

:p How does the pseudocode for policy iteration work?
??x
The pseudocode for policy iteration works by first initializing the value function and a policy. It then enters a loop where it alternates between evaluating the current policy to update state values and improving the policy based on these updated values.

1. **Policy Evaluation**: For each state, iteratively update its value until no significant changes are observed.
2. **Policy Improvement**: Check if any actions can improve the value function and update the policy accordingly.

The process continues until the policy stabilizes, indicating that an optimal policy has been found.
x??

---

#### Policy Iteration and Convergence Issues

Policy iteration often converges quickly, but has a subtle bug where it may never terminate if policies continually switch between equally good ones. This is acceptable for pedagogical purposes but not practical.

:p What modification can be made to ensure convergence in policy iteration?
??x
One way to ensure convergence is by breaking the cycle of switching between policies that are equally good. A common approach is to add a small perturbation or a tie-breaking rule to prefer one policy over another, ensuring the algorithm makes progress and eventually converges.

For example, you could implement a rule where if two policies have equal value function estimates, the previous policy is preferred:

```java
if (oldPolicy == currentPolicy) {
    // Use the old policy again
} else {
    // Proceed with the new policy
}
```

x??

---

#### Policy Iteration for Action Values

The concept of policy iteration can be extended to action values. The goal is to compute q⇤, which is similar to v⇤ but considers actions.

:p How would you define policy iteration for action values?
??x
Policy iteration for action values involves iteratively improving policies based on the estimated action values (q-values). Here’s a pseudocode outline:

1. Initialize q-values arbitrarily.
2. Evaluate each state-action pair:
   - Update the q-value using the Bellman optimality equation: 
     \[ q(s, a) = \sum_{s', r} p(s', r|s, a) [r + \gamma v(s')] \]
3. Improve policy based on q-values:
   - For each state \( s \), set the action to maximize the q-value: 
     \[ \pi'(s) = \arg\max_a q(s, a) \]

4. Repeat steps 2 and 3 until convergence.

Here is a pseudocode implementation:

```java
// Initialize q-values
for each state s in S do {
    for each action a in A(s) do {
        q[s][a] = random_value();
    }
}

// Policy iteration loop
while not converged do {
    // Policy evaluation (update q-values)
    for each state s in S do {
        for each action a in A(s) do {
            q[s][a] = sum over all transitions of [r + gamma * V(next_state)];
        }
    }

    // Policy improvement
    policy_stable = true;
    for each state s in S do {
        old_action = current_policy[s];
        new_action = argmax_a[q(s, a)];
        if (new_action != old_action) {
            policy_stable = false;
            current_policy[s] = new_action;
        }
    }

    // Check convergence
    if policy_stable { break; }
}
```

x??

---

#### -Soft Policies

An \(-soft\) policy ensures that the probability of selecting each action in each state is at least \(\frac{\epsilon}{|A(s)|}\).

:p How would you modify steps 3, 2, and 1 of the v⇤ policy iteration algorithm for an \(-soft\) policy?
??x
For an \(-soft\) policy, we need to ensure that every action in each state has a minimum probability. This affects the policy evaluation (step 2) and improvement (step 3), but step 1 can remain unchanged.

- **Step 1: Initialization**: Keep as is.
- **Step 2: Policy Evaluation**:
   - When updating q-values, ensure that actions with higher values are selected with a probability proportional to \(\frac{\epsilon}{|A(s)|}\).

- **Step 3: Policy Improvement**:
   - For each state \(s\), compute the maximum q-value and select an action based on the soft policy.
   - Adjust the probabilities of actions such that all actions in \(s\) have a probability at least \(\frac{\epsilon}{|A(s)|}\).

Example pseudocode for step 2 (policy evaluation) adjustment:

```java
for each state s in S do {
    for each action a in A(s) do {
        if (random() < epsilon / A(s).size()) {
            q[s][a] = some_random_value();
        } else {
            // Update using Bellman optimality equation
            q[s][a] = sum over all transitions of [r + gamma * V(next_state)];
        }
    }
}
```

x??

---

#### Gambler’s Problem

The gambler's problem is an example where the optimal policy can be solved via value iteration. The state space and actions are defined as follows:
- State: Capital \(s \in {1, 2, ..., 99}\)
- Actions: Stakes \(a \in {0, 1, ..., \min(s, 100 - s)}\)
- Reward: +1 when goal is reached; otherwise, 0

:p What is the optimal policy for the gambler's problem?
??x
The optimal policy in the gambler’s problem involves a non-monotonic betting strategy. For low capital levels (e.g., less than \( \frac{99}{2} \)), the gambler bets all available money. Conversely, when the capital is close to 100, the gambler bets just enough to reach or exceed 100.

This policy ensures that the gambler maximizes his chances of reaching the goal without risking more than necessary at any point.

Example:
- If the gambler has $50, he should bet all $50 in one flip.
- However, if the gambler has $51, he should not risk losing a big chunk by betting too much. Instead, he bets just enough to have a higher probability of reaching 100.

The exact threshold values and strategies can vary based on the discount factor \( \gamma \) and the probability \( p_h \).

x??

---

