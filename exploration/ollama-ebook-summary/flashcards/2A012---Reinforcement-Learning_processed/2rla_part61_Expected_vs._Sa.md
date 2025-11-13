# Flashcards: 2A012---Reinforcement-Learning_processed (Part 61)

**Starting Chapter:** Expected vs. Sample Updates

---

#### Expected vs. Sample Updates
Expected updates consider all possible events that might happen, while sample updates consider a single sampled event. In this context, we are focusing on one-step updates for approximating value functions $q^{\pi}$,$ v^{\pi}$,$ q^{*}$, and $ v^{*}$.

:p What is the difference between expected and sample updates in the context of reinforcement learning?
??x
Expected updates compute an exact estimate by considering all possible events, whereas sample updates use a single sampled event. This means that expected updates are more computationally intensive but yield lower variance estimates due to their reliance on all possible transitions.

Example:
- Expected update: 
$$Q(s, a) = \sum_{s', r} p(s', r | s, a) [r + \max_{a'} Q(s', a')]$$- Sample update (Q-learning):
$$

Q(s, a) = Q(s, a) + \alpha [R + \max_{a'} Q(S', a') - Q(s, a)]$$where $ p(s', r | s, a)$is the probability of transitioning to state $ s'$with reward $ r$from taking action $ a $ in state $s$.

x??

---
#### Computational Requirements for Updates
The computation required by update operations is often dominated by the number of state-action pairs. For discrete states and actions, the expected update requires evaluating each possible next state, while the sample update evaluates only one.

:p How does computational efficiency compare between expected and sample updates?
??x
Expected updates are more computationally intensive because they consider all possible transitions from a given state-action pair. Sample updates are cheaper as they only evaluate one transition per update step. The relative cost can be quantified by the branching factor $b$, which represents the number of possible next states.

Example:
- Expected Update Cost: 
$$\text{Cost} = b \times \text{evaluation time for } Q(s', a')$$- Sample Update Cost:
$$\text{Cost} = 1 \times \text{evaluation time for } Q(s', a')$$

Given the same computational budget, sample updates can be performed more frequently than expected updates.

x??

---
#### Estimation Error Analysis
The estimation error of expected and sample updates depends on the branching factor $b$. For large problems with many state-action pairs, it is often better to perform multiple sample updates rather than a single expected update due to time constraints.

:p How does the estimation error differ between expected and sample updates?
??x
Expected updates are free from sampling error but are computationally expensive. Sample updates introduce variance due to their reliance on random samples, making them cheaper but potentially less accurate. The trade-off is that for a large branching factor $b$, many sample updates can achieve a similar accuracy to one expected update.

Example:
- For a small branching factor ($b = 1$), both types of updates are identical.
- For a large branching factor ($b > 1$), the difference becomes significant. Expected updates yield lower variance but higher computational cost, while sample updates have higher variance but lower computation time per update.

x??

---
#### Practical Considerations in Planning
In practice, it is often more beneficial to perform multiple sample updates over fewer expected updates due to limited computational resources. Prioritized sweeping always uses expected updates for its accuracy, even in stochastic environments.

:p In what scenarios are sample updates preferable to expected updates?
??x
Sample updates are preferable when computation time is a limiting factor and high variance can be tolerated. Expected updates provide more accurate estimates but require significantly more computation, especially with large branching factors. Sample updates can make progress faster by performing many updates in less time.

Example:
```java
// Pseudocode for updating Q-values using sample method (Q-learning)
public void updateQ(double alpha, double reward, State nextState) {
    double target = reward + maxActionValue(nextState); // Find the maximum Q-value of the next state actions
    qValue[state, action] += alpha * (target - qValue[state, action]); // Update the current Q-value
}
```

x??

---

#### Comparison of Expected and Sample Updates
Background context explaining the comparison between expected updates and sample updates. In this scenario, we consider a problem where there are $b$ successor states that are equally likely to occur, with an initial error in the value estimate of 1.

If the values at the next states are assumed correct, then an expected update can reduce the error to zero upon completion. For sample updates, the error is reduced according to:
$$q_b^t = \frac{b}{t}$$where $ t $ is the number of sample updates performed (assuming sample averages with $\alpha = 1/t$).

The key observation is that for moderately large $b $, the error falls dramatically even after a tiny fraction of $ b$ updates. This means that many state-action pairs can have their values improved significantly in the same time it would take to perform one expected update.
:p How do expected and sample updates differ in this scenario?
??x
Expected updates reduce the error completely by considering all possible next states, while sample updates reduce the error based on a fraction of these states. For large branching factors $b$, many state-action pairs can be updated effectively with sample updates in less time than it would take to perform one expected update.
x??

---

#### Trajectory Sampling
Background context explaining trajectory sampling and its advantages over exhaustive sweeps. In dynamic programming, the classical approach is to perform sweeps through the entire state (or state–action) space, updating each state once per sweep. However, this can be problematic for large tasks because it may not be feasible to complete even one sweep.

Exhaustive sweeps distribute updates equally across all states, even those that are irrelevant or visited infrequently. In contrast, trajectory sampling distributes updates according to the on-policy distribution, which is easier to generate by interacting with the model and following the current policy.
:p How does trajectory sampling differ from exhaustive sweeps?
??x
Trajectory sampling focuses updates where they are most needed, rather than equally distributing them across all states as in exhaustive sweeps. This approach can be more efficient because it avoids unnecessary computations on less relevant parts of the state space.
x??

---

#### On-Policy Distribution of Updates
Background context explaining the use of on-policy distribution for generating experience and updating values. The on-policy distribution is generated by interacting with the model according to the current policy, making it easier than explicitly representing and using the on-policy distribution.

In episodic tasks, one starts in a start state (or according to the starting-state distribution) and simulates until the terminal state. In continuing tasks, one starts anywhere and keeps simulating indefinitely. This approach is efficient because it simulates explicit individual trajectories and performs updates at the states or state-action pairs encountered along the way.
:p Why might on-policy focusing be beneficial?
??x
On-policy focusing can be beneficial because it ignores vast, uninteresting parts of the state space, potentially leading to faster learning. However, it could also be detrimental by repeatedly updating the same old parts of the space.
x??

---

#### Empirical Evaluation of Update Distribution
Background context explaining an experiment comparing uniform and on-policy updates. The experiment used one-step expected tabular updates and evaluated two cases: uniform updates cycling through all state-action pairs, and on-policy updates simulating episodes starting in the same state.

Results were obtained for tasks with 1000 and 10,000 states, branching factors of 1, 3, and 10. The quality of policies found was plotted as a function of expected updates completed. On-policy focusing resulted in faster initial planning but eventually slowed down the process.
:p What were the key findings from this experiment?
??x
On-policy focusing initially improved planning speed by concentrating on relevant states, but it slowed down long-term planning. This effect was more pronounced for smaller branching factors and larger state spaces.
x??

---

---
#### Real-time Dynamic Programming (RTDP)
Real-time dynamic programming is an on-policy trajectory-sampling version of value-iteration, closely related to conventional policy iteration. RTDP updates state values based on expected tabular value-iteration updates as defined by formula 4.10.

:p What does the term "real-time dynamic programming" refer to?
??x
Real-time dynamic programming refers to a method that updates state values in real or simulated trajectories, using the same policy for both planning and execution, allowing for adaptive learning in an environment.
x??

---
#### On-policy Trajectory Sampling vs. Unfocused Approach
When dealing with many states and a small branching factor, on-policy trajectory sampling can be more effective than an exhaustive unfocused approach. In the long run, focusing on commonly occurring states may not provide additional value since their values are already correct.

:p Why might on-policy trajectory sampling be advantageous in large problems?
??x
On-policy trajectory sampling can be advantageous because it allows the algorithm to focus on relevant and frequently visited states, which have incorrect or uncertain values. This reduces unnecessary computations on irrelevant states, making the learning process more efficient.
x??

---
#### Scallop Effect in Graphs
Graphs showing scalloped early portions, particularly for b=1 and uniform distribution, might indicate that the algorithm is overfitting to a small set of frequently visited states.

:p Why do you think some graphs in Figure 8.8 seem to be scalloped in their early portions?
??x
The scalloped shape suggests that the algorithm is focusing too heavily on certain frequent states at the beginning, leading to overshooting or overfitting these states' values, while other states are not adequately sampled.
x??

---
#### Programming Exercise 8.8: Replicate RTDP Experiment
Replicating the experiment from Figure 8.8 and trying it with b=3 can provide insights into how varying the branching factor affects learning performance.

:p What is the objective of replicating the experiment in Exercise 8.8?
??x
The objective is to understand the impact of changing the branching factor (b) on the performance of RTDP by comparing results from b=1 and b=3, which can reveal how exploration versus exploitation trade-offs affect learning.
x??

---
#### Asynchronous DP Algorithms
Asynchronous DP algorithms update state values in any order, using whatever values are available at that moment. This is different from systematic sweeps where states are updated in a fixed sequence.

:p What distinguishes asynchronous DP algorithms from traditional DP methods?
??x
Async-DP algorithms do not follow a predefined schedule to update states but instead use the most recently available information for updates, which can lead to more efficient learning and better exploration of the state space.
x??

---
#### Relevant vs. Irrelevant States
In RTDP, if trajectories start from designated start states, irrelevant states (those unreachable under any policy) do not need attention as they are not part of the prediction or control problem.

:p How does RTDP handle irrelevant states?
??x
RTDP can skip updating values for irrelevant states because these states cannot be reached by the given policy from any of the start states. This approach saves computational resources and focuses on relevant states that contribute to the problem's solution.
x??

---

#### Exploring Starts and RTDP
Exploring starts is a technique that can be used to initialize value iteration algorithms, including RTDP. For episodic tasks, RTDP acts as an asynchronous value-iteration algorithm for discounted finite MDPs (and certain undiscounted cases). Unlike prediction problems, it's generally necessary to keep updating states until convergence to an optimal policy.
:p What is the role of exploring starts in RTDP?
??x
Exploring starts help initialize the values of states before running RTDP. This ensures that initial state values are not all zero or arbitrary, which can improve the performance and correctness of value iteration algorithms by providing more realistic starting points. 
```java
// Pseudocode for setting up exploring starts in RTDP
public void setupExploringStarts(State[] startStates) {
    for (State state : startStates) {
        setValue(state, randomValue()); // Set initial values based on exploration
    }
}
```
x??

---

#### Convergence of RTDP with Exploring Starts
For episodic tasks, particularly those where every episode ends in an absorbing goal state generating zero rewards, RTDP converges to optimal policies for both discounted and undiscounted finite MDPs. However, unlike prediction problems, it is essential to continue updating states until convergence if the optimal policy is crucial.
:p How does RTDP ensure convergence to an optimal policy?
??x
RTDP ensures convergence by continuously updating state values based on greedy actions at each step of a trajectory. The key conditions are that all non-goal state transitions yield negative rewards, initial values meet or exceed their optimal values (often set to zero), and the algorithm starts from exploring states with positive values.
```java
// Pseudocode for RTDP update process
public void performRTDPUpdate(State currentState) {
    Action action = getGreedyAction(currentState);
    State nextState = applyAction(action, currentState);
    double expectedReward = calculateExpectedReward(nextState);
    setValue(currentState, getValue(currentState) + alpha * (expectedReward - getValue(currentState)));
}
```
x??

---

#### Conditions for RTDP Convergence
RTDP converges to an optimal policy under specific conditions: initial values of goal states are zero, at least one policy guarantees reaching a goal state with probability one from any start state, all non-goal transitions have strictly negative rewards, and initial values are equal or greater than their optimal values.
:p What conditions must be met for RTDP to converge?
??x
For RTDP to converge, the following conditions must hold: 
1. Initial value of every goal state is zero.
2. There exists at least one policy that ensures a goal state will be reached with probability one from any start state.
3. All rewards for transitions from non-goal states are strictly negative.
4. All initial values are equal to or greater than their optimal values (often set to zero).

These conditions ensure the algorithm's convergence and accuracy in finding an optimal policy.
```java
// Pseudocode checking RTDP convergence conditions
public boolean checkConvergenceConditions(State[] startStates, State[] goalStates) {
    for (State state : goalStates) {
        if (getValue(state) != 0) return false; // Condition 1 violated
    }
    for (State state : startStates) {
        if (!canReachGoalFrom(state)) return false; // Condition 2 violated
    }
    for (Transition transition : nonGoalTransitions()) {
        if (transition.getReward() >= 0) return false; // Condition 3 violated
    }
    return true; // All conditions met, RTDP can converge
}
```
x??

---

#### Application of RTDP in Racetrack Problem
The racetrack problem is an example of a stochastic optimal path problem. RTDP can be applied to such problems by setting up the environment and updating values based on trajectories generated from the current state.
:p How does RTDP apply to the racetrack problem?
??x
RTDP applies to the racetrack problem by treating it as a stochastic optimal path problem where the goal is to minimize the cost of reaching the finish line. RTDP updates value estimates for states along trajectories, ensuring convergence to an optimal policy under certain conditions.
```java
// Pseudocode for applying RTDP in Racetrack
public void applyRTDPToRacetrack(RaceTrack track) {
    setupExploringStarts(track.getStartStates());
    while (!convergenceConditionsMet()) {
        State current = getRandomStartState();
        while (current != null) {
            Action action = getGreedyAction(current);
            State next = applyAction(action, current);
            updateValue(current, next);
            current = next;
        }
    }
}
```
x??

---

#### Problem Description
The problem involves an agent learning to drive a car around turns and cross a finish line as quickly as possible while staying on track. The state space is potentially infinite, but only 9,115 states are reachable from start states by any policy. There are 599 relevant states that can be reached via some optimal policy.

:p What problem does the agent need to solve?
??x
The agent needs to learn how to drive a car around turns and cross a finish line as quickly as possible while staying on track. This involves navigating from start states (all zero-speed states on the starting line) to goal states (reaching the finish line within the track boundaries).
x??

---
#### State Space and Convergence Criteria
The state space consists of 9,115 reachable states, with only 599 relevant ones being optimal. Convergence in conventional DP is determined when the maximum change in a state value over a sweep is less than $10^{-4}$. RTDP converges when the average time to cross the finish line stabilizes.

:p How are the methods judged to have converged?
??x
Conventional DP converges when the maximum change in a state value over a sweep is less than $10^{-4}$. For RTDP, convergence occurs when the average time to cross the finish line appears to stabilize. This means that after 20 episodes, the average time does not significantly change.
x??

---
#### Conventional DP vs. RTDP
Conventional DP uses value iteration with exhaustive sweeps of the state set using the Gauss-Seidel version (approximately twice as fast as Jacobi). It required an average of 28 sweeps to converge and had over 250,000 updates.

:p What are the key differences between conventional DP and RTDP?
??x
Conventional DP uses value iteration with exhaustive state set sweeps, updating values one state at a time using the most recent values. It required an average of 28 sweeps to converge and had over 250,000 updates.

RTDP is on-policy trajectory sampling, requiring only 4000 episodes but more than 127,600 updates. RTDP updates only the current state's value on each step.
x??

---
#### State Updates in Conventional DP
Conventional DP updated every state multiple times, with 31.9% of states being updated over 10 times.

:p What is the percentage of states that were updated more than 10 times in conventional DP?
??x
In conventional DP, 31.9% of states were updated more than 10 times during convergence.
x??

---
#### State Updates in RTDP
RTDP required fewer updates compared to conventional DP. Only 98.45% of states were updated at least 10 times.

:p What percentage of states were updated 10 or less times in RTDP?
??x
In RTDP, 98.45% of states were updated 10 or fewer times during convergence.
x??

---
#### Convergence Time and Updates in RTDP
RTDP required an average of 4000 episodes to converge.

:p How many episodes did it take for RTDP to converge?
??x
It took an average of 4000 episodes for RTDP to converge.
x??

---
#### Performance Comparison
Both methods produced similar policies, averaging between 14 and 15 steps to cross the finish line. However, RTDP required fewer updates than conventional DP.

:p What was the performance comparison between Conventional DP and RTDP?
??x
Both methods produced similar policies, with an average of around 14-15 steps to cross the finish line. RTDP required only about half as many updates compared to conventional DP.
x??

---

