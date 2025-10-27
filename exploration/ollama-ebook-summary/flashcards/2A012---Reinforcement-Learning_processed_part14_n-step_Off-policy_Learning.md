# Flashcards: 2A012---Reinforcement-Learning_processed (Part 14)

**Starting Chapter:** n-step Off-policy Learning

---

#### n-step Bootstrapping Formula for Sarsa
Sarsa is an on-policy temporal difference (TD) learning method that updates action-value estimates using a single step of experience. The n-step return for Sarsa can be written exactly in terms of a novel TD error, as shown below:
\[ G_{t:t+n} = Q_t(S_t, A_t) + \min(n+1, T-t-1)\sum_{k=t}^{t+n-1} \left[R_{k+1} + \gamma^k\left( Q_k(S_{k+1}, A_{k+1}) - Q_{t-1}(S_t, A_t) \right) \right] \]
:p What is the n-step return formula for Sarsa?
??x
The n-step return \( G_{t:t+n} \) in Sarsa can be computed as:
\[ G_{t:t+n} = Q_t(S_t, A_t) + \min(n+1, T-t-1)\sum_{k=t}^{t+n-1} \left[R_{k+1} + \gamma^k\left( Q_k(S_{k+1}, A_{k+1}) - Q_{t-1}(S_t, A_t) \right) \right] \]
This formula takes into account the immediate rewards and discounted future rewards up to \( n \) steps ahead. It uses the current action-value estimate at time \( t \), and updates it based on the actual returns observed over the next \( n \) steps.
x??

---

#### Expected Sarsa n-step Return
Expected Sarsa is a variant of Sarsa that incorporates exploration by considering all possible actions from the future state. The n-step return for Expected Sarsa can be defined as:
\[ G_{t:t+n} = R_{t+1} + \sum_{k=2}^{n} \gamma^k R_{t+k} + \bar{V}_{t+n-1}(S_{t+n}) \]
where \( \bar{V}_{t+n-1}(S) \) is the expected approximate value of state \( S \), given by:
\[ \bar{V}_{t+n-1}(S) = \sum_{a} \pi(a|S) Q_{t+n-1}(S, a) \]
:p What is the n-step return formula for Expected Sarsa?
??x
The n-step return \( G_{t:t+n} \) in Expected Sarsa can be computed as:
\[ G_{t:t+n} = R_{t+1} + \sum_{k=2}^{n} \gamma^k R_{t+k} + \bar{V}_{t+n-1}(S_{t+n}) \]
where \( \bar{V}_{t+n-1}(S) \) is the expected approximate value of state \( S \), given by:
\[ \bar{V}_{t+n-1}(S) = \sum_{a} \pi(a|S) Q_{t+n-1}(S, a) \]
This formula incorporates the average action values over all possible actions under the target policy.
x??

---

#### n-step Off-policy Learning
Off-policy learning in reinforcement learning refers to learning the value function for one policy while following another policy. In n-step methods, returns are constructed over \( n \) steps and adjusted based on the relative probability of taking those actions. The importance sampling ratio is used to weight the return.
The update rule for off-policy n-step Sarsa can be written as:
\[ V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha \cdot \frac{P(S_0, A_0; \pi)}{P(S_0, A_0; b)} [G_{t:t+n} - V_{t+n-1}(S_t)] \]
where \( P(S_0, A_0; \pi) \) is the probability of taking action \( A_0 \) in state \( S_0 \) under policy \( \pi \), and \( P(S_0, A_0; b) \) is the same for behavior policy \( b \).
:p What is the update rule for off-policy n-step Sarsa?
??x
The update rule for off-policy n-step Sarsa can be written as:
\[ V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha \cdot \frac{P(S_0, A_0; \pi)}{P(S_0, A_0; b)} [G_{t:t+n} - V_{t+n-1}(S_t)] \]
where \( P(S_0, A_0; \pi) \) is the probability of taking action \( A_0 \) in state \( S_0 \) under policy \( \pi \), and \( P(S_0, A_0; b) \) is the same for behavior policy \( b \). This formula adjusts the value based on the importance sampling ratio to account for the difference between the target and behavior policies.
x??

---

#### n-step Off-policy Sarsa Algorithm
The pseudocode for off-policy n-step Sarsa includes an importance sampling ratio to weight returns. The algorithm updates action-value estimates by considering both the actual return and the target policy's expected value.
:p What is the pseudocode for off-policy n-step Sarsa?
??x
```pseudocode
// Off-policy n-step Sarsa Algorithm

Input: behavior policy b, target policy π, step size α, positive integer n
Initialize Q(s, a) arbitrarily, for all s ∈ S, a ∈ A
Initialize π to be greedy with respect to Q or as a fixed policy
Algorithm parameters: step size α ∈ (0, 1], a positive integer n

Loop for each episode:
    Initialize and store S₀ ≠ terminal
    Select and store an action A₀ ∼ b(·|S₀)
    T = 1
    Loop for t=0,1,...:
        If t < T, then: 
            Take action A_t
            Observe and store the next reward as R_{t+1} and the next state as S_{t+1}
            If S_{t+1} is terminal, then end this episode; else continue.
            Select and store an action A_{t+1} ∼ b(·|S_{t+1})
            If t + n < T, then: 
                G = R_{t+1} + γ^n * Q(S_{t+n}, A_{t+n}) - Q(S_t, A_t)
                V_t+n(S_t) = V_{t+n-1}(S_t) + α * [G - V_{t+n-1}(S_t)]
        If π is being learned, ensure that π(·|S_t) is greedy w.r.t. Q
Until T = T-1
```
This pseudocode outlines the steps for updating action values using off-policy n-step Sarsa while adjusting for the difference between target and behavior policies through importance sampling.
x??

#### n-step Bootstrapping Overview
Background context: The chapter discusses an oﬄine version of n-step Expected Sarsa, which updates using a similar formula to n-step Sarsa but with adjusted importance sampling. This is because in Expected Sarsa, all possible actions are considered in the final state's value estimation.
:p What is the key difference between the oﬄine n-step Expected Sarsa and the standard n-step Sarsa?
??x
The key difference lies in the use of importance sampling ratios. In oﬄine n-step Expected Sarsa, the importance sampling ratio uses \( \pi_{t+1:t+n-1} \) instead of \( \pi_{t+1:t+n} \). This adjustment is made because in Expected Sarsa, all possible actions are considered at the last state, so the action actually taken has no effect.
x??

---

#### Per-decision Methods with Control Variates
Background context: The text introduces a more sophisticated approach to oﬄine multi-step methods by using per-decision importance sampling ideas. This involves an alternate, oﬄine definition of the n-step return that avoids shrinking estimates due to zero weights when actions are not selected.
:p How does the control variate in the oﬄine n-step return (Equation 7.13) help in maintaining the estimate's stability?
??x
The control variate helps by ensuring that even if an action has a zero importance sampling weight, it does not result in a zero target. Instead, the target remains equal to the current estimate, thus no change occurs. This is achieved through the term \( (1 - \pi_t) V_{h-1}(S_h) \), which ensures that when \( \pi_t = 0 \), the update does not shrink but remains stable.
x??

---

#### oﬄine Action-value Prediction
Background context: For action values, the n-step return differs because the first action is being learned and thus must be fully weighted. The control variate here adjusts the importance sampling to account for actions taken after the first one.
:p How does the oﬄine n-step action value update formula (Equation 7.14) address the issue of the initial action?
??x
The oﬄine n-step action value update formula addresses the initial action by ensuring it is fully weighted, even if unlikely under the target policy. The first part \( R_{t+1} + \pi_{t+1} [G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1})] \) ensures that the initial action's impact is captured, while the control variate term \( \bar{V}_{h-1}(S_{t+1}) \) helps in stabilizing the estimate.
x??

---

#### Control Variates and Expected Updates
Background context: The text explains how control variates do not change the expected value of the return. This is because the importance sampling ratio has an expected value of one, making the control variate term's expected value zero.
:p How does the use of a control variate ensure that the expected update remains unchanged?
??x
The use of a control variate ensures the expected update remains unchanged by leveraging the fact that \( \pi_t \) (the importance sampling ratio) has an expected value of one. Since it is uncorrelated with the estimate, the control variate term's expected value is zero. This means that the overall update rule retains its expected value, stabilizing the learning process.
x??

---

#### Example Programming Exercise
Background context: The text mentions a programming exercise where one needs to show that using (7.13) and (7.2) results in more data-efficient updates compared to simpler methods.
:p How can you demonstrate that an oﬄine prediction algorithm using n-step returns with control variates is more data-efficient?
??x
To demonstrate this, you would implement the learning rules for both the simple method using \( G_{t+1:t+n} \) and the more advanced method using (7.13). By running simulations or experiments, you can observe that the algorithm using control variates converges faster or requires fewer samples to achieve similar performance.
```java
public class PredictionAlgorithm {
    public void train() {
        // Implement training logic with n-step returns and control variates
    }
}
```
x??

---
These flashcards cover key aspects of oﬄine reinforcement learning methods, providing a framework for understanding their implementation and benefits.

#### n-step Tree Backup Algorithm Overview
The 3-step tree-backup update is an o↵-policy learning method without using importance sampling, which extends the idea of a backup diagram. Each node in the tree contributes to the target for updating based on its probability under the target policy π.

:p What does the 3-step tree-backup update represent?
??x
The 3-step tree-backup update represents an o↵-policy learning algorithm that updates Q-values without using importance sampling, extending the backup diagram concept. It considers all possible actions and their probabilities to form the target for updating Q-values.
x??

---

#### Tree Backup Update Formula Derivation
The tree-backup update formula is derived from the idea of considering the entire tree of action values. The target value Gt:t+n is defined as a sum that includes rewards, estimated future Q-values, and the probability of actions under the policy π.

:p What is the general recursive definition of the n-step tree backup return?
??x
The general recursive definition of the n-step tree-backup return (Gt:t+n) is given by:
\[ G_{t:t+n} = R_{t+1} + \sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q_{n-1}(S_{t+1}, a) + \pi(A_{t+1}|S_{t+1}) G_{t+1:t+n} \]
This equation is valid for \( t < T - 1 \), and the case for \( n = 1 \) uses the expected SARSA return:
\[ G_{t:t+1} = R_{t+1} + \sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q(S_{t+1}, a) \]

x??

---

#### n-step Tree Backup Algorithm Pseudocode
The pseudocode for the n-step tree-backup algorithm involves updating action values using the n-step return target. The update rule is similar to that of n-step Sarsa but considers all possible actions and their probabilities.

:p Provide the pseudocode for the n-step tree backup algorithm.
??x
```pseudocode
for each episode:
    initialize Q-values randomly or with zeros
    
    for each step t < T - 1:
        # Perform the update using the n-step return target
        Qt+n(St, At) = Qt+n-1(St, At) + α [Gt:t+n - Qt+n-1(St, At)]
        
        # Ensure other state-action pairs remain unchanged
        for all s, a such that s ≠ St or a ≠ At:
            Qt+n(s, a) = Qt+n-1(s, a)
```
x??

---

#### Explanation of n-step Tree Backup Target
The target value Gt:t+n is the expected sum of rewards and discounted future Q-values, considering actions not selected but with their probabilities. This ensures that all possible paths from the state-action pair (St, At) are considered.

:p What does the tree-backup return formula account for in its calculation?
??x
The tree-backup return formula accounts for the expected sum of rewards and discounted future Q-values, considering actions not selected but with their probabilities. Specifically, it includes:
- The immediate reward \( R_{t+1} \)
- The estimated Q-values of unselected actions at each level, weighted by their policy probability
- The recursive call to the next step’s target value for the selected action

This comprehensive approach ensures that all possible paths from (St, At) are considered in the update.
x??

---

#### Tree Backup Update Weights
The weights for the leaf nodes in the tree-backup update are proportional to their probabilities under the policy π. The actions taken directly do not contribute but influence the updates of unselected actions.

:p How are the weights assigned to each action node during the tree backup update?
??x
During the tree backup update, each action node's weight is assigned based on its probability under the target policy π. Specifically:
- For a non-selected first-level action \( a \), it contributes with a weight of \( \pi(a|S_{t+1}) \).
- For a selected action \( A_{t+1} \), it does not contribute at all.
- For a second-level action \( a_0 \) that was not taken, its contribution is weighted by \( \pi(A_{t+1}|S_{t+1})\pi(a_0|S_{t+2}) \).
- This continues recursively for higher levels.

The weight essentially propagates the influence of selected actions down the tree.
x??

---

#### n-step Tree Backup Algorithm Target Calculation
The target value \( G_t:t+n \) is calculated by considering the immediate reward and discounted future rewards and Q-values, including all possible actions at each level with their policy probabilities. This approach ensures that the update considers a broad range of potential outcomes.

:p How do you calculate the n-step tree-backup return for a given step t?
??x
The n-step tree-backup return \( G_t:t+n \) is calculated as follows:
\[ G_{t:t+n} = R_{t+1} + \sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q_{n-1}(S_{t+1}, a) + \pi(A_{t+1}|S_{t+1}) G_{t+1:t+n} \]

For the first step (n=1), it simplifies to:
\[ G_{t:t+1} = R_{t+1} + \sum_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q(S_{t+1}, a) \]

This formula accounts for the immediate reward and all possible future actions, weighted by their probabilities.
x??

---

