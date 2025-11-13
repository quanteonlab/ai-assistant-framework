# Flashcards: 2A012---Reinforcement-Learning_processed (Part 86)

**Starting Chapter:** AlphaGo

---

#### AlphaGo's Innovation and Approach
AlphaGo, developed by DeepMind, significantly advanced the application of Monte Carlo Tree Search (MCTS) through a novel method called asynchronous policy and value MCTS (APV-MCTS). This approach combines elements from both reinforcement learning and supervised learning to enhance its performance in the game of Go. 

The core idea is to use deep convolutional artificial neural networks (ANNs) for guiding the selection of actions during MCTS while leveraging a learned value function to evaluate states.

:p What does APV-MCTS stand for, and what are its key components?
??x
APV-MCTS stands for Asynchronous Policy and Value Monte Carlo Tree Search. Its key components include:
1. **SL-policy Network**: A 13-layer deep convolutional ANN that predicts moves based on a database of nearly 30 million human expert Go games.
2. **Value Network**: Another 13-layer deep convolutional ANN that provides estimated values for board positions, used to evaluate states beyond the rollout returns.

This combination allows for more informed decision-making during the search process, integrating both policy and value information effectively.

x??

---

#### AlphaGo's MCTS Implementation
In contrast to traditional MCTS, which selects unexplored edges based on stored action values, APV-MCTS uses a probabilistic approach guided by the SL-policy network. This network predicts moves with high confidence for each position in the game.

:p How does APV-MCTS differ from basic MCTS in selecting actions?
??x
APV-MCTS differs from traditional MCTS by using probabilities supplied by a 13-layer deep convolutional ANN called the SL-policy network to choose which action edge to explore next. Unlike basic MCTS, where edges are selected based on stored action values, APV-MCTS relies on the probabilistic predictions of the policy network.

The logic can be illustrated with pseudocode:
```java
function selectAction(node) {
    // Get probabilities from SL-policy network
    probabilities = slPolicyNetwork.predictProbabilities(node);
    
    // Select an action based on these probabilities
    selectedAction = randomChoice(probabilities);
    
    return selectedAction;
}
```

This approach ensures that the search is directed towards more promising moves according to expert human knowledge, enhancing the efficiency and effectiveness of the MCTS process.

x??

---

#### Value Network in AlphaGo
The value network in AlphaGo plays a crucial role in evaluating states. It outputs an estimated value for each board position using a 13-layer deep convolutional ANN. The formula for updating the value of a state node $s$ combines both rollout returns and learned value functions.

:p What is the formula used to update the value function in APV-MCTS?
??x
The formula used to update the value function in APV-MCTS for a newly added node $s$ is:
$$v(s) = (1 - \alpha)v_\text{✓}(s) + \alpha G$$where $ G $ is the return from the rollout and $\alpha$ controls the mixing of the values derived from both methods.

This formula ensures that the value function $v(s)$ is a blend of the current estimate $v_\text{✓}(s)$ from the learned value network and the actual outcome $G$ from simulations. This hybrid approach helps in refining the model over time by incorporating real outcomes while leveraging prior knowledge.

x??

---

#### Role of Rollouts
In AlphaGo, rollouts are simulated games played with both players using a fast rollout policy provided by a simple linear network trained via supervised learning. These rollouts help to evaluate newly added nodes by providing a quick estimate of the outcome from that position.

:p How does AlphaGo use rollouts in its MCTS process?
??x
AlphaGo uses rollouts as part of its APV-MCTS evaluation strategy. Rollouts are fast simulations where both players play with a simple linear network trained to mimic expert player behavior. During each MCTS iteration, after actions are selected by the SL-policy network, the system runs these rollouts to estimate the outcome from the current position.

The logic can be described as follows:
```java
function performRollout(node) {
    // Initialize board state from node
    boardState = node.getState();
    
    // Simulate a game with both players using the rollout policy
    while (gameNotOver(boardState)) {
        action = randomActionFromLegalActions(boardState);
        boardState = applyAction(action, boardState);
    }
    
    return calculateOutcome(boardState); // 1 for win, -1 for loss, 0 for draw
}
```

These rollouts provide a quick and efficient way to assess the quality of moves, integrating them into the MCTS framework alongside the learned value function.

x??

---

---
#### Value Network Training Process
Background context: The DeepMind team divided the training process of the value network for AlphaGo into two stages to tackle the complexity of the game Go. In the first stage, they used reinforcement learning (RL) to create a policy network. In the second stage, they employed Monte Carlo policy evaluation using self-play games generated by this RL policy network.

:p What was the two-stage approach taken by DeepMind for training AlphaGo's value network?
??x
In the first stage, they created a best possible policy using reinforcement learning (RL) to train an RL policy network. This network was initialized with weights from a supervised learning (SL) policy network and further improved through policy-gradient RL. In the second stage, Monte Carlo policy evaluation was used on data generated by self-play games, where moves were selected based on the policy network.

```java
// Pseudocode for initializing the first stage
public void initializePolicyNetwork() {
    // Initialize weights from SL policy network
    weights = finalWeightsOfSLPolicy;
    // Further improvement through RL
}
```
x??

---
#### Policy Network Accuracy and Evaluation
Background context: The team trained a 13-layer policy network (SL policy network) using data from the KGS Go Server. This network achieved an accuracy of 57.0% when considering all input features, and 55.7% with only raw board position and move history as inputs.

:p What was the accuracy of the SL policy network?
??x
The accuracy of the SL policy network was 57.0 percent using all input features and 55.7 percent using just raw board position and move history, which was significantly better than state-of-the-art methods at the time (44.4%).

```java
// Pseudocode for evaluating the accuracy of the SL policy network
public double evaluateAccuracy(List<Position> positions) {
    int correctPredictions = 0;
    for (Position pos : positions) {
        if (predictedMove(pos).equals(expertMove(pos))) {
            correctPredictions++;
        }
    }
    return (double) correctPredictions / positions.size();
}
```
x??

---
#### Reinforcement Learning of Policy Networks
Background context: The second stage involved using policy gradient reinforcement learning to further improve the policy network. This was done by playing games between the current RL policy network and a randomly selected previous iteration.

:p How did DeepMind train the RL policy network in the second stage?
??x
DeepMind trained the RL policy network through policy gradient reinforcement learning, where weights were updated at each time step to maximize expected outcome. The process involved playing games between the current RL policy network $p_{\rho}$ and a randomly selected previous iteration of the same network.

```java
// Pseudocode for updating weights in RL training
public void updateWeights(double learningRate, double outcome) {
    // Calculate gradient based on outcome
    Gradient = calculateGradient(outcome);
    // Update weights using stochastic gradient ascent
    weights += learningRate * Gradient;
}
```
x??

---

---

#### Neural Network Training Pipeline and Architecture

Background context: The text describes the training pipeline for AlphaGo, a neural network system designed to play Go. This includes both policy networks (which predict moves) and value networks (which predict game outcomes), trained using self-play data and reinforcement learning.

:p What is the structure of the neural network used in AlphaGo?
??x
The architecture involves multiple convolutional layers for processing board positions, followed by a policy network that outputs move probabilities and a value network that predicts game outcomes. Here’s a simplified version:

```python
class NeuralNetwork:
    def __init__(self):
        self.policy_network = PolicyNetwork()
        self.value_network = ValueNetwork()

    def train(self, data_set):
        # Train policy network with supervised learning (SL)
        self.policy_network.train_sl(data_set)

        # Initialize RL policy to SL and improve via reinforcement
        rl_policy = self.policy_network
        while not converged:
            new_data_set = self.play_self_games(rl_policy)
            self.value_network.train(new_data_set)
            rl_policy.improve_with_policy_gradient()

    def play_self_games(self, policy):
        # Play games using the current policy and collect self-play data
        return collected_data

    def evaluate_value(self, position):
        # Evaluate a given position with the value network
        return self.value_network.predict(position)

```
x??

---

#### Policy Networks Training

Background context: Policy networks are trained to predict moves based on board positions. The text mentions that policy networks were periodically evaluated using AlphaGo against itself.

:p How does the training process for policy networks work in the provided pipeline?
??x
The training involves initial supervised learning (SL) to learn from human expert moves, followed by reinforcement learning (RL) where the network improves its performance through self-play games. Here’s a simplified version:

```python
class PolicyNetwork:
    def train_sl(self, data_set):
        # Train using human expert positions
        for position, move in data_set:
            self.update_parameters(position, move)

    def improve_with_policy_gradient(self):
        # Improve the policy network via reinforcement learning
        while not converged:
            new_data_set = play_self_games(self)
            for position, outcome in new_data_set:
                self.update_parameters(position, outcome)

def play_self_games(policy_network):
    # Play games using the current policy and collect data
    return collected_data

```
x??

---

#### Value Network Training

Background context: The value network predicts the expected outcome of a game based on board positions. It was trained to improve the accuracy of predictions over time.

:p How is the value network trained in AlphaGo?
??x
The value network is trained using regression, where it predicts the outcome (win/lose) of self-play games generated by the policy network. Here’s a simplified version:

```python
class ValueNetwork:
    def train(self, data_set):
        for position, outcome in data_set:
            prediction = self.predict(position)
            loss = compute_loss(prediction, outcome)
            self.update_parameters(loss)

def predict(self, position):
    # Predict the outcome of the given position
    return predicted_value

def update_parameters(self, loss):
    # Adjust parameters based on the computed loss
    pass

```
x??

---

#### Evaluation Accuracy Comparison

Background context: The text compares the evaluation accuracy of the value network against different policies and rollouts. This helps in understanding which policy is more accurate at predicting game outcomes.

:p How does the text compare the performance of the value network with various rollout policies?
??x
The comparison involves evaluating positions using a single pass through the value network or by averaging multiple rollout outcomes (using uniform random, fast rollout policy, SL policy, or RL policy). The mean squared error between predicted values and actual game outcomes is plotted to assess accuracy.

```python
def evaluate_value_network():
    errors = []
    for position in sampled_positions:
        true_outcome = get_true_game_outcome(position)
        
        value_network_prediction = vθ(position)
        rollout_predictions = [pπ(position), pσ(position), pρ(position)]
        mean_rollout_prediction = sum(rollout_predictions) / len(rollout_predictions)

        errors.append(mean_squared_error(value_network_prediction, true_outcome))
        errors.append(mean_squared_error(mean_rollout_prediction, true_outcome))

    return errors

def mean_squared_error(predicted_value, actual_value):
    # Compute the MSE between predicted and actual values
    return (predicted_value - actual_value) ** 2

```
x??

---

#### RL Policy Network Performance Against Pachi14

Background context: The RL policy network was tested against a strong open-source Go program, Pachi14. The results showed that even without search, the RL policy network outperformed Pachi14 in most games.

:p What were the results of testing the RL policy network against Pachi14?
??x
The RL policy network won 85 percent of the games against Pachi14, which is ranked at an amateur dan level on KGS and executes 100,000 simulations per move. This indicates that the RL approach significantly outperformed traditional Monte Carlo methods in this context.

```java
public class PolicyNetworkTest {
    public static void main(String[] args) {
        PolicyNetwork rlPolicy = new PolicyNetwork();
        Pachi14 pachi = new Pachi14();

        int totalGames = 100;
        int wins = 0;

        for (int i = 0; i < totalGames; i++) {
            GameResult result = playGame(rlPolicy, pachi);
            if (result == WIN) {
                wins++;
            }
        }

        double winRate = (double) wins / totalGames * 100;
        System.out.println("Win rate against Pachi: " + winRate + "%");
    }

    private static GameResult playGame(PolicyNetwork player1, PolicyNetwork player2) {
        // Simulate a game between the two players
        return WIN; // or LOSE or DRAW based on logic
    }
}
```
x??

---

#### AlphaGo's Evaluation Method
AlphaGo used a combination of a value network and rollouts to evaluate game states. The parameter ⌘ controlled how these two components were mixed, with values between 0 and 1.

:p What did the parameter ⌘ control in AlphaGo?
??x
The parameter ⌘ controlled the mixing of game state evaluations produced by the value network and by rollouts. When ⌘=0, only the value network was used; when ⌘=1, evaluation relied solely on rollouts. A value between 0 and 1 combined both methods.
x??

---
#### AlphaGo's Success with ⌘ = 0.5
AlphaGo achieved its best results with ⌘ set to 0.5, indicating that the combination of the value network and rollouts was crucial for its performance.

:p Why did setting ⌘ to 0.5 result in better play?
??x
Setting ⌘ to 0.5 balanced the use of the value network and rollouts effectively. The value network could evaluate high-performance policies, while the rollouts provided precision evaluations for specific game states. This combination outperformed using either component alone.
x??

---
#### AlphaGo Zero's Self-Play Reinforcement Learning
AlphaGo Zero learned entirely through self-play reinforcement learning without any human data or guidance beyond basic game rules.

:p How did AlphaGo Zero learn?
??x
AlphaGo Zero used a form of policy iteration, interleaving policy evaluation with improvement. It relied solely on raw board positions and used MCTS throughout its self-play process to select moves.
x??

---
#### Difference Between AlphaGo and AlphaGo Zero in MCTS Usage
While AlphaGo used MCTS for live play after learning, AlphaGo Zero applied MCTS continuously during the self-play reinforcement learning process.

:p How did AlphaGo use MCTS differently from AlphaGo Zero?
??x
AlphaGo used MCTS selectively for live play, whereas AlphaGo Zero employed MCTS throughout its entire training process. In AlphaGo, MCTS was not part of the initial learning phase but was introduced later to improve decision-making during actual gameplay.
x??

---
#### AlphaGo Zero's MCTS Process
Each iteration of AlphaGo Zero’s MCTS ran simulations ending at leaf nodes instead of terminal positions, using a deep convolutional network for guidance.

:p How did AlphaGo Zero perform its MCTS?
??x
AlphaGo Zero conducted MCTS iterations that ended at leaf nodes in the search tree. Each iteration was guided by a deep convolutional network, which provided an estimate of winning probability (v) and move probabilities (p). This method allowed for more precise evaluations without running full game simulations.
x??

---
#### AlphaGo Zero's Network Output
AlphaGo Zero’s network produced both a scalar value (win probability) and a vector of move probabilities.

:p What did the network output in AlphaGo Zero?
??x
The network generated two outputs: a scalar value, v, which estimated the win probability for the current player, and a vector, p, containing the probabilities for each possible stone placement on the board, plus pass or resign moves.
x??

---
#### Importance of Simulation Results in MCTS
AlphaGo Zero used simulations to refine move probabilities and improve its policy.

:p How did AlphaGo Zero use simulation results?
??x
Each MCTS iteration in AlphaGo Zero returned new move probabilities (denoted as policies ⇡i) after conducting numerous simulations. These simulations helped refine the move probabilities, leading to improved overall strategy.
x??

---

#### Monte-Carlo Tree Search (MCTS) Execution
Background context: In AlphaGo Zero, a Monte Carlo Tree Search (MCTS) is executed to select moves during self-play. The MCTS helps explore promising moves and compute search probabilities that guide subsequent actions.

:p How does the MCTS work in AlphaGo Zero?
??x
The MCTS works by iteratively building a tree of possible moves from the current position, evaluating them using simulations, and adjusting the policy based on these evaluations. It balances exploration (exploring unvisited or underexplored nodes) and exploitation (focusing on promising nodes).

```pseudocode
function MonteCarloTreeSearch(rootNode):
    for i in range(numIterations):
        node = selectBestNode(rootNode)
        result = simulateGameFrom(node)
        backpropagate(result, node)
```

x??

---

#### Neural Network Architecture and Training Process
Background context: The neural network in AlphaGo Zero takes raw board positions as input, processes them through convolutional layers, and outputs a probability distribution over moves and an estimate of the current player's win probability.

:p What is the role of the neural network in AlphaGo Zero?
??x
The neural network serves to predict move probabilities and game outcomes by processing raw board positions. It updates its parameters based on self-play data to improve its policy (move selection) and value function (win probability).

```python
def trainNeuralNetwork(data):
    for batch in data:
        inputs, moves, winners = batch
        outputs = network(inputs)
        loss = calculateLoss(outputs, moves, winners)
        optimizer.minimize(loss)
```

x??

---

#### Self-Play Reinforcement Learning Process
Background context: AlphaGo Zero plays against itself to generate training examples. Each game generates a sequence of board positions and corresponding moves and outcomes.

:p How does the self-play reinforcement learning process work in AlphaGo Zero?
??x
AlphaGo Zero plays many games against itself, generating sequences of board positions, moves, and outcomes. These are used as training data for both improving its policy network and value function by adjusting parameters to better match MCTS search probabilities and game winners.

```pseudocode
for i in range(numGames):
    boardPositions, moves, winners = playGame()
    updatePolicyNetwork(boardPositions, moves)
    updateValueFunction(boardPositions, winners)
```

x??

---

#### Policy Vector and MCTS Search Probabilities
Background context: The policy vector $p$ represents the move probabilities output by the neural network. These are compared to the search probabilities from MCTS to adjust the network's parameters.

:p How do the policy vector $p$ and search probabilities relate in AlphaGo Zero?
??x
The policy vector $p$ guides the Monte Carlo Tree Search (MCTS) exploration, with moves being selected based on these probabilities. The neural network's policy vector is updated to better match the MCTS-generated search probabilities, ensuring that the model learns from the most promising actions.

```pseudocode
function updatePolicy(network, monteCarloProbabilities):
    for i in range(len(monteCarloProbabilities)):
        network.setParameter(i, monteCarloProbabilities[i])
```

x??

---

#### Win Probability Estimation and Error Minimization
Background context: The value function $v$ estimates the probability of winning from each position. It is trained to minimize error between its predictions and actual game outcomes.

:p How does the neural network estimate win probabilities in AlphaGo Zero?
??x
The value function $v$ outputs a scalar representing the estimated probability of the current player winning. The network is trained to reduce the difference between this predicted value and the actual game winner, thereby improving its accuracy over time.

```pseudocode
def trainValueFunction(data):
    for batch in data:
        inputs, wins = batch
        predictions = network(inputs)
        loss = calculateLoss(predictions, wins)
        optimizer.minimize(loss)
```

x??

---

#### Markov State and Game Repetition
Background context: In Go, the current board position is not a Markov state because of rules against repeated moves and compensation points for the player who did not make the first move.

:p Why can't the current board position be considered a Markov state in Go?
??x
The current board position cannot be a Markov state because it does not fully encapsulate all relevant information due to Go's specific rules. For instance, moves are not allowed to repeat, and there is compensation for the player who did not get the first move. These factors make future states dependent on past sequences of moves rather than just the current board.

x??

---

#### Network Architecture Overview
Background context explaining the network's architecture. The network is described as "two-headed," with one head producing move probabilities and the other an estimate of win probability.

:p What was the structure of the network before it split into two heads?
??x
The network consisted of 41 convolutional layers, each followed by batch normalization, and with skip connections to implement residual learning. These layers were followed by a split into two heads: one head for producing move probabilities (362 output units) and another for estimating the win probability (one output unit).
x??

---
#### Move Probability Output
Background context explaining how the network produced move probabilities.

:p How did the network produce move probabilities?
??x
The network's first head, after the split, fed into 362 output units to produce move probabilities $p$ for each of the 192+1 possible stone placements (plus pass).
x??

---
#### Win Probability Output
Background context explaining how the network estimated win probability.

:p How did the network estimate the win probability?
??x
The network's second head, after the split, fed into a single output unit to produce an estimate of the scalar $v$, representing the probability that the current player would win from the current board position.
x??

---
#### Training Process Overview
Background context explaining how the network was trained.

:p How was the network initially trained?
??x
The network started with random weights and was trained using stochastic gradient descent (with momentum, regularization, and step-size parameter decreasing over time). It used batches of examples sampled uniformly at random from the most recent 500,000 games of self-play. Extra noise was added to encourage exploration.
x??

---
#### Evaluation During Training
Background context explaining how the network's policy was evaluated during training.

:p How did the researchers evaluate the network’s performance during training?
??x
At periodic checkpoints (every 1,000 training steps), the network’s current policy output by running Monte Carlo Tree Search (MCTS) for 400 games against the current best policy. If the new policy won a sufficient margin, it became the best policy used in subsequent self-play.
x??

---
#### Training Iterations and Time
Background context explaining the training process duration.

:p How long did the DeepMind team train AlphaGo Zero?
??x
The DeepMind team trained AlphaGo Zero over 4.9 million games of self-play, which took about 3 days. Each move was selected by running MCTS for 1,600 iterations, taking approximately 0.4 seconds per move.
x??

---
#### Batch Size and Updates
Background context explaining the batch size and update process.

:p What were the details of the training batches?
??x
The network’s weights were updated over 700,000 batches, each consisting of 2,048 board configurations.
x??

---
#### Comparison with AlphaGo
Background context explaining how AlphaGo Zero was compared against previous versions of AlphaGo.

:p How did they compare AlphaGo Zero to previous versions of AlphaGo?
??x
They ran tournaments with AlphaGo Zero playing against the version that defeated Fan Hui and the version that defeated Lee Sedol. They used Elo ratings to evaluate performance, finding significant differences in these ratings.
x??

---
#### Elo Ratings Comparison
Background context explaining the use of Elo ratings.

:p What were the Elo ratings for AlphaGo Zero and previous versions?
??x
The Elo ratings were 4,308 for AlphaGo Zero, 3,144 for the version that defeated Fan Hui, and 3,739 for the version that defeated Lee Sedol. These ratings indicated that AlphaGo Zero was much stronger.
x??

---
#### Tournament Results
Background context explaining the tournament results.

:p What were the results of the match between AlphaGo Zero and the previous AlphaGo version?
??x
In a match of 100 games, AlphaGo Zero defeated the exact version of AlphaGo that had defeated Lee Sedol in all 100 games.
x??

---
#### Supervised Learning Comparison
Background context explaining how AlphaGo Zero was compared with a supervised learning player.

:p How did AlphaGo Zero compare to a supervised-learning player?
??x
The supervised-learning player initially played better and was better at predicting human expert moves, but its performance declined after training AlphaGo Zero for a day. This suggested that AlphaGo Zero had discovered strategies superior to those learned by the supervised approach.
x??

---

#### AlphaGo Zero Overview
Background context explaining how AlphaGo Zero revolutionized AI by achieving superhuman performance through pure reinforcement learning. It discovered novel move sequences and defeated previous versions of AlphaGo, demonstrating its problem-solving capabilities.
:p What is AlphaGo Zero's significance in AI research?
??x
AlphaGo Zero was a groundbreaking achievement as it demonstrated that superhuman performance can be achieved purely through reinforcement learning without any human data or features. It started with random weights and learned to play Go by self-play, discovering novel move sequences and defeating previous versions of AlphaGo.
x??

---

#### Reinforcement Learning in AlphaGo Zero
Explanation of how AlphaGo Zero used reinforcement learning (RL) to improve its performance over time through self-play games. Mention the use of a neural network (ANN) for policy optimization and Monte Carlo Tree Search (MCTS) for decision-making.
:p How did AlphaGo Zero utilize RL?
??x
AlphaGo Zero utilized reinforcement learning by playing millions of self-play games against itself, continuously improving its performance. It used a large ANN to optimize the policy and MCTS to make decisions during gameplay. Starting from random weights, it achieved an Elo rating of 5,185.
x??

---

#### AlphaZero: General Reinforcement Learning
Description of AlphaZero as a general RL algorithm that surpassed previous versions in games like Go, chess, and shogi without any domain-specific knowledge. Emphasize the broader applicability of such algorithms in various domains.
:p What distinguishes AlphaZero from other reinforcement learning programs?
??x
AlphaZero is distinct because it does not rely on any specific domain knowledge or human data; instead, it uses general RL techniques to outperform previous versions in games like Go, chess, and shogi. This approach highlights its potential for broader application across different problem domains.
x??

---

#### Personalized Web Services with Reinforcement Learning
Explanation of how personalized web services can enhance user satisfaction by recommending content based on user profiles inferred from online activity. Use the concept of a contextual bandit to maximize user engagement through click-through rate optimization.
:p How does reinforcement learning improve personalized web services?
??x
Reinforcement learning enhances personalized web services by adjusting recommendations in response to user feedback, optimizing for actions like clicks or views. Using methods like contextual bandits, it can maximize the total number of user clicks by considering individual user contexts and content features.
x??

---

#### Contextual Bandit Problem
Explanation of the contextual bandit problem as a specific type of reinforcement learning where decisions are made based on context to optimize an objective, such as maximizing user clicks. Mention the importance of context in personalizing services.
:p What is the contextual bandit problem?
??x
The contextual bandit problem is a specific type of reinforcement learning used to make optimal decisions in real-time by leveraging context. It aims to maximize an objective, like user clicks, while considering individual features and content. This approach allows for personalized service delivery based on user preferences.
x??

---

#### A/B Testing in Personalized Web Services
Explanation of A/B testing as a method for comparing two versions of a website to determine user preference without personalization. Mention its limitations and how contextual bandits address these issues.
:p What is A/B testing, and why is it limited?
??x
A/B testing involves showing different versions (A and B) of a web page to users and observing which one performs better in terms of user preferences or actions. However, this method is non-associative, meaning it does not personalize content delivery based on individual user data.
x??

---

#### Applying Contextual Bandits for Personalization
Explanation of how contextual bandits can be used to personalize web services by incorporating context and optimizing for user engagement metrics like click-through rates.
:p How do contextual bandits enable personalized service delivery?
??x
Contextual bandits enable personalized service delivery by considering individual user contexts and content features, aiming to optimize user engagement through actions such as clicks or views. This approach contrasts with non-personalized methods like A/B testing by adapting recommendations in real-time based on user data.
x??

---

#### Contextual Bandit Algorithm for Webpage Personalization
Background context explaining the concept. The objective was to maximize click-through rate (CTR), which is defined as the ratio of total number of clicks all users make on a webpage to the total number of visits to the page. Their contextual bandit algorithm improved over standard non-associative bandit algorithms by 12.5 percent.
:p What is CTR in the context of web personalization?
??x
CTR, or Click-Through Rate, refers to the ratio of users who click on a particular link (or ad) out of those who viewed it. In this case, it's the total number of clicks all users make on a webpage divided by the total number of visits to that page.
??x

---

#### Policies Derived from Contextual Bandit Formulation
Background context explaining the concept. Policies derived from contextual bandit formulations are greedy in nature, treating each visit as if made by a new visitor uniformly sampled from the population.
:p What does it mean when policies are described as "greedy" in the context of web personalization?
??x
Greedy policies only consider immediate benefits and do not take into account long-term effects or user behavior patterns. They treat each visit independently, assuming users are new visitors every time they come back to the website.
??x

---

#### Long-Term Interaction Policies vs Greedy Policies
Background context explaining the concept. Policies that leverage long-term interactions with users can improve overall click-through rates over repeated visits compared to greedy policies which only consider immediate actions.
:p How do longer-term interaction policies differ from greedy policies?
??x
Longer-term policies consider the history and behavior of individual users, potentially guiding them through a sequence of steps (e.g., sales funnel) before making offers. Greedy policies offer immediate benefits but may not capitalize on long-term user engagement or behavioral patterns.
??x

---

#### Example: Displaying Ads for Buying a Car
Background context explaining the concept. The example contrasts a greedy policy that offers an immediate discount with a longer-term policy that gradually presents information leading to eventual sales.
:p What is the key difference between a greedy policy and a long-term policy in the car buying scenario?
??x
A greedy policy immediately offers a discount, while a long-term policy builds interest through multiple visits by providing information on favorable financing terms, service quality, etc., eventually leading to a sale. The long-term policy aims for sustained engagement and repeated interactions.
??x

---

#### Off-Policy Evaluation in Adobe Marketing Cloud Experiments
Background context explaining the concept. The research team at Adobe aimed to evaluate new policies using data collected under existing ones to reduce deployment risk while ensuring high confidence in performance predictions.
:p What is off-policy evaluation in this context?
??x
Off-policy evaluation involves estimating the performance of a new policy (the target policy) using data collected from different or older policies (the behavior policies). This method helps assess potential risks before actual deployment, providing insights into expected long-term impacts.
??x

---

#### Importance of High Confidence in Off-Policy Evaluation
Background context explaining the concept. The research team needed to ensure high confidence in their off-policy evaluation results to minimize risk associated with deploying novel policies.
:p Why is high confidence important in off-policy evaluation?
??x
High confidence ensures that the predicted performance of a new policy is reliable, reducing the risk of deploying ineffective or harmful strategies. This approach helps maintain trust and efficiency in marketing campaigns by validating potential improvements before widespread implementation.
??x

#### Greedy Optimization Algorithm
Background context explaining the concept. The greedy optimization algorithm aimed at maximizing only the probability of immediate clicks, using a mapping estimated as a function of user features learned via supervised learning with random forests (RF). This method did not consider long-term effects and had challenges due to sparse rewards and high variance.
:p What is the primary goal of the greedy optimization algorithm?
??x
The primary goal of the greedy optimization algorithm was to maximize only the probability of immediate clicks by selecting offers based on predicted click probabilities. The algorithm used a mapping estimated from user features via supervised learning with RF, but it did not take into account long-term effects or future user behavior.
??x

---

#### Life-time Value (LTV) Optimization
Background context explaining the concept. LTV optimization aimed to improve the number of clicks users made over multiple visits by considering the long-term value of recommendations using an MDP formulation and reinforcement learning via fitted Q iteration (FQI). The algorithm used RF for function approximation due to its scalability.
:p What was the main objective of the life-time value (LTV) optimization approach?
??x
The main objective of the LTV optimization approach was to improve the total number of clicks users made over multiple visits by considering long-term user behavior and preferences. This method used reinforcement learning with FQI, where RF was employed for function approximation due to its effectiveness in handling high-dimensional data.
??x

---

#### Data Set Details
Background context explaining the concept. The study utilized two large data sets from the banking industry, each containing interactions between customers and website offers. These data sets were used for training and testing both greedy optimization and LTV optimization algorithms.
:p What are the details of the two data sets used in the experiments?
??x
The two data sets used in the experiments contained:
1. Approximately 200,000 interactions from a month with 7 possible offers.
2. Over 4 million interactions involving 12 possible offers.

These data sets included various customer features such as time since last visit, number of visits, geographic location, interests, and demographic information.
??x

---

#### -Greedy Policy Implementation
Background context explaining the concept. The greedy optimization algorithm implemented an $-greedy policy that selected with probability \(1-$ the offer predicted by the RF to have the highest click probability, and otherwise selected other offers uniformly at random.
:p How was the$-greedy policy defined in the greedy optimization approach?
??x
The \(-greedy policy in the greedy optimization approach was defined as follows:
- With probability \(1-$, select the offer predicted by the RF algorithm to have the highest click probability.
- Otherwise, select other offers uniformly at random.

This policy balanced exploration and exploitation by leveraging the RF model's predictions while still allowing for some randomness.
??x

---

#### Fitted Q Iteration (FQI)
Background context explaining the concept. The LTV optimization approach used batch-mode reinforcement learning with fitted Q iteration (FQI), which is a variant of fitted value iteration adapted to Q-learning and suitable for large-scale applications due to its use of RF.
:p What is the key algorithm used in the life-time value (LTV) optimization approach?
??x
The key algorithm used in the LTV optimization approach was fitted Q iteration (FQI), which is a batch-mode reinforcement learning method. FQI is a variant of fitted value iteration adapted for Q-learning, making it suitable for large-scale applications due to its scalability with RF algorithms.

FQI works by iteratively improving an action-value function approximation until convergence. The algorithm uses RF for function approximation, leveraging its effectiveness in handling high-dimensional data and avoiding overfitting.
??x

---

#### Ongoing Policy Evaluation
Background context explaining the concept. For FQI's non-monotonic convergence, Theocharous et al. used o\-n-policy evaluation with a validation training set to keep track of the best policy produced by FQI. The final policy for testing was based on this best policy.
:p How did Theocharous et al. ensure they had the best policy during the LTV optimization process?
??x
To address the non-monotonic convergence of FQI, Theocharous et al. used o\-n-policy evaluation with a validation training set to track the best policy produced by FQI. This method involved evaluating each new policy against previous policies using the validation data and selecting the best one.

The final policy for testing was based on this best policy, ensuring that the chosen approach maximized long-term user engagement and click-through rates.
??x

---

#### Metrics Used
Background context explaining the concept. The study used two metrics to evaluate the performance of both greedy optimization and LTV optimization: CTR (Click-Through Rate) and LTV (Life-time Value). These metrics differed in how they measured overall effectiveness over multiple visits.
:p What metrics were used to measure the performance of the policies?
??x
The metrics used to measure the performance of the policies were:
1. **CTR (Click-Through Rate)**: Defined as the total number of clicks divided by the total number of visits.
2. **LTV (Life-time Value)**: Defined as the total number of clicks divided by the total number of visitors.

These metrics distinguished between individual website visitors, providing a more comprehensive view of user engagement and overall effectiveness.
??x

---

#### RF Algorithm for Function Approximation
Background context explaining the concept. Random forests (RF) were used for function approximation in both greedy optimization and LTV optimization to handle high-dimensional data effectively and avoid overfitting or noise issues.
:p What role did the random forest (RF) algorithm play in the experiments?
??x
The random forest (RF) algorithm played a crucial role in the experiments by serving as the primary method for function approximation:
- In greedy optimization, RF was used to estimate click probabilities based on user features.
- For LTV optimization, RF was also employed to approximate the action-value function in FQI.

This choice of RF allowed for effective handling of high-dimensional data and robust performance across different scales of interaction data.
??x

