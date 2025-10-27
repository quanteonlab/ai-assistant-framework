# Flashcards: 2A014 (Part 31)

**Starting Chapter:** 125-Order-Two Markov Chain

---

#### Sequential Recommendations
Background context explaining sequential recommendations. These models aim to predict users' next actions based on their recent ordered list of interactions, going beyond simple pairwise relationships between potential recommendations and historical interactions.

:p What is the main focus of sequential recommendation models?
??x
Sequential recommendation models focus on predicting users’ next actions by considering the sequential interactions in the past, often involving combinations of interactions among three or more items. This approach goes beyond simple pairwise relationships to capture higher-order dependencies.
x??

---
#### Markov Chains
Background context explaining Markov chains and their role in modeling temporal dependencies between items. A first-order Markov chain models future states based solely on the current state, while higher-order chains consider a set of previous states.

:p What is a Markov chain used for in sequential recommendation?
??x
A Markov chain is used to model the probability of transitioning from one state to another, given the current state, without considering the sequence of preceding events. In sequential recommendations, it helps capture the temporal dependencies between items by treating each state as an item and transition probabilities as the likelihood of a user interacting with a certain item after the current one.
x??

---
#### First-Order Markov Chain
Background context explaining first-order Markov chains and their effectiveness in capturing short-term, item-to-item transition patterns. It is effective for predicting immediate next actions based on recent interactions.

:p How does a first-order Markov chain work in recommending items?
??x
A first-order Markov chain models the future state depending solely on the current state. This approach is useful for capturing short-term, item-to-item transition patterns. For example, if you are watching episodes of Succession, the next episode can be predicted based on the current one without considering earlier interactions.

Code Example:
```java
public class FirstOrderMarkovChain {
    private Map<String, List<String>> transitionMap;

    public FirstOrderMarkovChain(Map<String, List<String>> data) {
        this.transitionMap = data;
    }

    public String predictNext(String currentItem) {
        if (!transitionMap.containsKey(currentItem)) {
            return null; // No data available
        }
        return transitionMap.get(currentItem).get(0); // Assuming only one next item for simplicity
    }
}
```
x??

---
#### Higher-Order Markov Chains
Background context explaining higher-order Markov chains and their ability to model richer user behavior by considering multiple previous states.

:p How do higher-order Markov chains differ from first-order ones?
??x
Higher-order Markov chains differ from first-order ones by considering a set of previous states, rather than just the current state. This approach provides a richer model of user behavior and can better capture complex patterns in user interactions over longer periods.
x??

---
#### Transformer Architectures for Sequential Data Modeling
Background context explaining how transformer architectures have shown superior performance for modeling sequential data due to their efficiency and effectiveness at handling long-range sequences.

:p Why are transformer architectures used in sequential recommendation?
??x
Transformer architectures are used in sequential recommendations because they can efficiently handle long-range dependencies, making them suitable for capturing complex patterns in user interactions over extended periods. Transformers excel at parallelization, which enhances computational efficiency.
x??

---

#### Order-Two Markov Chain
An order-two Markov chain models the probability of a state based on the previous two states. This is useful for scenarios where the current state depends not only on the immediate past but also on further history.

The transition probabilities are represented as \(P_{S_t, S_{t-1}, S_{t-2}}\), indicating the probability of moving from one set of states to another.
:p What does an order-two Markov chain model?
??x
An order-two Markov chain models the probability distribution of a state based on the previous two states. It is used when the current state depends not only on the immediate past but also on the state before that. The transition probabilities are given by \(P_{S_t, S_{t-1}, S_{t-2}}\), where \(S_t\) is today's state, \(S_{t-1}\) is yesterday’s state, and \(S_{t-2}\) is the day before yesterday's state.
x??

---

#### Transition Probabilities in Order-Two Markov Chain
The transition probabilities for an order-two Markov chain can be represented using a three-dimensional tensor. For example:
\[ PSS,S = 0.7 \]
\[ PCS,S = 0.2 \]
\[ PRS,S = 0.1 \]

These represent the probability of transitioning from sunny (S) to sunny given that it was sunny and then cloudy (S, S, C).

:p What are transition probabilities in an order-two Markov chain?
??x
Transition probabilities in an order-two Markov chain are the chances of moving from one state to another based on the previous two states. These are represented using a three-dimensional tensor where each element \(P_{S_t, S_{t-1}, S_{t-2}}\) gives the probability of transitioning from state at time \(t\), given the state at times \(t-1\) and \(t-2\).
x??

---

#### Visualizing Transition Probabilities
These transition probabilities can be visualized in a three-dimensional cube. The first two dimensions represent today’s state and yesterday's state, while the third dimension represents tomorrow’s state.

:p How are transition probabilities typically visualized?
??x
Transition probabilities in an order-two Markov chain are typically visualized using a three-dimensional tensor or cube. Each axis of the cube corresponds to one of the states: two axes represent today's and yesterday's states, and the other axis represents the state at time \(t+1\). This visualization helps in understanding the probability distribution across these states.
x??

---

#### Estimating Transition Probabilities
The transition probabilities can be estimated from historical data by counting the number of times each transition occurs and dividing by the total number of transitions.

:p How are transition probabilities typically estimated?
??x
Transition probabilities in an order-two Markov chains are estimated using historical data. This involves counting the occurrences of specific state sequences (e.g., S, S, C) over time and then normalizing these counts to get the probability of transitioning from one state to another given the previous two states.
```java
public class ProbabilityEstimator {
    private Map<String, Integer> countMap;
    
    public void addTransition(String currentState, String previousState1, String previousState2) {
        String key = currentState + "," + previousState1 + "," + previousState2;
        countMap.put(key, countMap.getOrDefault(key, 0) + 1);
    }
    
    public double getProbability(String currentState, String previousState1, String previousState2) {
        String key = currentState + "," + previousState1 + "," + previousState2;
        return (double) countMap.getOrDefault(key, 0) / totalTransitionsCount;
    }
}
```
x??

---

#### Markov Decision Process (MDP)
A more advanced approach is the Markov decision process (MDP), which extends the basic Markov chain by incorporating actions and rewards. It can be used in recommender systems where each action represents a recommendation, and the reward is based on user feedback.

:p What is a Markov decision process (MDP)?
??x
A Markov Decision Process (MDP) is an extension of the basic Markov chain that includes actions and rewards. In the context of a recommender system, actions can represent recommendations, and rewards can be user responses to these recommendations. The goal is to learn a policy that maximizes the expected cumulative reward.

An MDP is defined by a tuple \( (S, A, P, R) \), where:
- \( S \) is the set of states
- \( A \) is the set of actions
- \( P \) is the state transition probability matrix
- \( R \) is the reward function

For example, in a movie recommender system:
- States (S): Genres watched by users (e.g., Comedy, Drama, Action)
- Actions (A): Movies that can be recommended (e.g., Movie 1, 2, 3, 4, 5)
- Transition probabilities (P): Likelihood of transitioning from one state to another given an action
- Rewards (R): User feedback after a recommendation

x??

---

#### Sequential Recommendation Systems Overview
Background context: This section introduces sequential recommendation systems, highlighting how they differ from traditional methods by focusing on user interactions within a session rather than explicit user IDs. The goal is to capture and leverage the sequence of actions for better recommendations.
:p What are key differences between traditional and session-based recommendation systems?
??x
Session-based recommendations operate over anonymous user sessions that are often short, allowing them to model user behavior without relying on long-term user profiles. Traditional methods rely on explicit user IDs to build interest profiles, which can be less efficient due to the variability in user motivations across different sessions.
x??

---

#### Recurrent Neural Networks (RNNs) for Sequential Recommendations
Background context: RNNs are designed to recognize patterns in sequences of data and maintain a form of memory by feeding outputs back into the network. This is crucial for tasks like language modeling, where each word depends on previous words.
:p What mechanism allows RNNs to effectively process sequential data?
??x
RNNs maintain an internal state that gets updated at each time step with information from previous steps. At each time step, an input (like a word in a sentence) is processed, and the network updates its internal state before producing an output.
```java
// Pseudocode for RNN processing
public void processInput(String input) {
    // Update internal state based on input
    currentState = updateState(currentState, input);
    
    // Produce output (e.g., next word prediction)
    String output = predictNextWord(currentState);
}
```
x??

---

#### GRU4Rec for Session-Based Recommendations
Background context: GRU4Rec is an application of RNNs to session-based recommendations. It models sessions as sequences of items and predicts the next item in a sequence.
:p How does GRU4Rec use RNNs for session-based recommendations?
??x
GRU4Rec treats each user session as a sequence of items, where it uses an RNN to predict the next item based on all previous items in the session. This approach allows the model to leverage the entire history of interactions within a session.
```java
// Pseudocode for GRU4Rec prediction
public Item predictNextItem(List<Item> session) {
    // Update hidden state with each item in the sequence
    HiddenState hiddenState = updateHiddenState(hiddenState, session);
    
    // Predict next item based on current hidden state
    Item predictedItem = predictNextItemFromState(hiddenState);
}
```
x??

---

#### CNN for Sequential Recommendations: CosRec
Background context: CosRec uses a Convolutional Neural Network (CNN) to handle sequential recommendations. It captures sequence information through pairwise transitions and concatenates these with user embeddings.
:p How does CosRec use CNNs in sequential recommendation?
??x
CosRec encodes sequences by collecting embedding vectors for each item in the sequence, creating an L×D matrix. Adjacent row pairs are concatenated to form a three-tensor that is passed through a 2D CNN, yielding a vector that is combined with user embeddings and fed through a fully connected layer.
```java
// Pseudocode for CosRec processing
public Vector processSequence(List<Item> sequence) {
    // Create L×D matrix from item embeddings
    Matrix embeddings = createEmbeddingsMatrix(sequence);
    
    // Form three-tensor by concatenating adjacent row pairs
    ThreeTensor tensor = formThreeTensor(embeddings);
    
    // Pass through 2D CNN and fully connected layer
    Vector output = applyCNNAndFC(tensor, userVector);
}
```
x??

---

