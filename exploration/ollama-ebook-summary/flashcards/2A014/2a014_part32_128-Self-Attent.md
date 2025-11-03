# Flashcards: 2A014 (Part 32)

**Starting Chapter:** 128-Self-Attentive Sequential Recommendation

---

#### Positional Embeddings
Positional embeddings are a critical component of transformer models, as they allow the model to learn about the position of each token in the input sequence. Unlike traditional embedding techniques that focus on the semantic meaning of words alone, positional embeddings help maintain the order and context within sequences.

These embeddings are learned during training and added to the word embeddings to provide positional information. For example, if we have a sentence with 5 tokens, the model will learn 5 distinct positional embeddings corresponding to each position in the sequence.

:p What is the role of positional embeddings in transformers?
??x
Positional embeddings serve to incorporate the order and context within sequences that are processed by transformer models. They are learned during training and added to word embeddings to help the model understand the relative positions of tokens, which is crucial for tasks like machine translation or text generation.

Example: If we have a sentence "The cat sat on the mat", the positional embeddings would encode the position of each token (e.g., The - 1, cat - 2, sat - 3, etc.) to help the model understand that "cat" follows "The".

---
#### Self-Attention Layer
Self-attention is a mechanism in transformers where every element in the sequence gets to attend to all other elements. This allows the model to weigh the importance of different tokens based on their context within the sequence.

Formally, self-attention can be defined as:

\[ \text{self\_attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V \]

Where \( Q \), \( K \), and \( V \) are the query, key, and value matrices respectively.

:p What is self-attention in transformers?
??x
Self-attention in transformers allows each element of a sequence to attend to all other elements. It computes an attention score for every pair of tokens based on their queries and keys, and then combines these scores with values (e.g., word embeddings) to compute a weighted sum that represents the context of the token.

Example:
Given a sentence "The cat sat on the mat", if we have query \( Q \), key \( K \), and value \( V \) matrices for each token:

\[ 
Q = \begin{bmatrix}
q_{1} \\
q_{2} \\
q_{3} \\
\end{bmatrix}, \
K = \begin{bmatrix}
k_{1} & k_{2} & k_{3} \\
\end{bmatrix}, \
V = \begin{bmatrix}
v_{1} \\
v_{2} \\
v_{3} \\
\end{bmatrix}
\]

The attention scores \( S \) can be calculated as:

\[ 
S = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)
\]

Where \( d_k \) is the dimension of keys.

Then, the output is computed as:

\[ 
O = SV
\]

The attention mechanism helps the model capture long-range dependencies and contextual information within the sequence.
??x

---
#### Skip-Addition Mechanism
Skip-addition in transformers refers to adding the input vector back to the final output after passing through self-attention or feed-forward layers. This is a form of residual connection that helps mitigate vanishing gradients and allows deeper network architectures.

:p What is skip-addition in transformers?
??x
Skip-addition, also known as residual connections, involves adding the original input vector back to the output after processing it through a layer (such as self-attention or feed-forward). This mechanism helps stabilize training and enables the model to learn more complex functions by stacking multiple layers.

Example:
Assume \( X \) is the input tensor, and \( F(X) \) represents a combination of self-attention and feed-forward layers. The output after skip-addition would be:

\[ 
Y = X + F(X)
\]

This addition allows the gradient to flow more smoothly through deeper layers.

:p How does skip-addition work in transformers?
??x
Skip-addition works by adding the original input vector \( X \) back to the final output of a layer, such as self-attention or feed-forward. This helps stabilize training and enables the model to capture long-range dependencies effectively.

Example:
Given an input tensor \( X \), after passing through self-attention and feed-forward layers:

\[ 
Y = F(X)
\]

The skip-addition operation adds the original input back to this output:

\[ 
Z = X + Y
\]

This mechanism helps in mitigating vanishing gradients, making it easier to train deeper networks.

---
#### Feed-Forward Layer
A feed-forward layer in transformers is a simple multilayer perceptron (MLP) that processes each position independently. It typically consists of two linear transformations with an activation function between them.

Formally:

\[ 
\text{feed\_forward}(x) = \sigma(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2)
\]

Where \( W_1 \), \( W_2 \), and biases \( b_1 \), \( b_2 \) are learnable parameters, and \( \sigma \) is the activation function (e.g., ReLU or GeLU).

:p What is a feed-forward layer in transformers?
??x
A feed-forward layer in transformers is a simple multilayer perceptron that processes each position independently. It consists of two linear transformations with an activation function between them, typically followed by a residual connection.

Example:
Consider the following feed-forward layer implementation:

```java
public class FeedForwardLayer {
    private LinearLayer linear1;
    private ActivationLayer relu; // or GeLU
    private LinearLayer linear2;

    public FeedForwardLayer(int inputSize, int hiddenSize) {
        this.linear1 = new LinearLayer(inputSize, hiddenSize);
        this.relu = new ActivationLayer(hiddenSize); // ReLU activation
        this.linear2 = new LinearLayer(hiddenSize, 1); // Output size is the same as input size
    }

    public Tensor feedForward(Tensor input) {
        Tensor hidden = linear1.forward(input);
        Tensor activated = relu.forward(hidden);
        return linear2.forward(activated);
    }
}
```

:p How does a feed-forward layer work in transformers?
??x
A feed-forward layer in transformers processes each position independently using two linear transformations with an activation function (e.g., ReLU or GeLU) between them. It typically follows the structure:

\[ 
\text{feed\_forward}(x) = \sigma(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2)
\]

Where \( W_1 \), \( W_2 \) are weight matrices, and \( b_1 \), \( b_2 \) are biases. The activation function (e.g., ReLU or GeLU) introduces nonlinearity.

Example:
```java
public class FeedForwardLayer {
    private LinearLayer linear1;
    private ActivationLayer relu; // or GeLU
    private LinearLayer linear2;

    public FeedForwardLayer(int inputSize, int hiddenSize) {
        this.linear1 = new LinearLayer(inputSize, hiddenSize);
        this.relu = new ActivationLayer(hiddenSize); // ReLU activation
        this.linear2 = new LinearLayer(hiddenSize, 1); // Output size is the same as input size
    }

    public Tensor feedForward(Tensor input) {
        Tensor hidden = linear1.forward(input);
        Tensor activated = relu.forward(hidden);
        return linear2.forward(activated);
    }
}
```

This structure allows the model to learn complex transformations of each position independently, contributing to the overall functionality of the transformer architecture.
??x

---

#### Self-Attention Mechanism Overview
Self-attention is a mechanism where each element in a sequence considers every other element. This is achieved by learning four weight matrices per head, typically denoted as Q (query), K (key), O (output), and V (value). The attention process involves the following steps:
1. Compute the query matrix \( QE^\prime \) from the input vectors using the query weights.
2. Compute the key matrix \( KE^\prime \) from the positional embeddings.
3. Form the attention matrix by computing the dot product between the query and key matrices: \( A = QE^\prime \cdot (KE^\prime)^T \).
4. Apply a softmax function to each row of the attention matrix to get the attention vector.

:p What is the self-attention mechanism?
??x
The self-attention mechanism allows every element in a sequence to consider all other elements, facilitating interactions between different parts of the sequence. This process involves learning four weight matrices per head (Q, K, O, V) and computing an attention matrix through dot products followed by applying a softmax function.

```java
public class SelfAttention {
    public void computeAttentionVectors(float[][] inputs, float[][] positionalEmbeddings) {
        // Compute query vectors
        float[][] Q = ...; // Weighted input vectors
        
        // Compute key vectors from positional embeddings
        float[][] K = ...; // Weighted positional embeddings
        
        // Form the attention matrix A by computing dot products of Q and K
        float[][] A = computeAttentionMatrix(Q, K);
        
        // Apply softmax to get the attention vector
        float[] attentionVector = applySoftmax(A);
    }
    
    private float[][] computeAttentionMatrix(float[][] Q, float[][] K) {
        int seqLength = Q.length;
        float[][] result = new float[seqLength][seqLength];
        for (int i = 0; i < seqLength; i++) {
            for (int j = 0; j < seqLength; j++) {
                result[i][j] = dotProduct(Q[i], K[j]);
            }
        }
        return result;
    }
    
    private float[] applySoftmax(float[][] A) {
        // Softmax function applied row-wise
        ...
    }
}
```
x??

---
#### Heads and Weight Matrices in Self-Attention
In self-attention, the weight matrices are often referred to as Q (query), K (key), O (output), and V (value). These matrices are used to transform input vectors into query, key, output, and value vectors. The heads are in a 1-to-1 correspondence with the sequence length.

:p How many weight matrices are typically learned per head in self-attention?
??x
Typically, four weight matrices (Q, K, O, V) are learned per head in self-attention. These matrices transform input vectors into query, key, output, and value vectors respectively.

```java
public class AttentionHead {
    private float[] Q;
    private float[] K;
    private float[] O;
    private float[] V;

    public void initializeWeights(float[][] inputs) {
        // Initialize weight matrices for the heads
        ...
    }
}
```
x??

---
#### SASRec Model Overview
SASRec is a transformer model designed for sequential recommendation tasks. It predicts the next user interaction from past interactions in an autoregressive manner, meaning it only allows attention to earlier positions in the sequence.

:p What is SASRec and how does it work?
??x
SASRec is a transformer-based model used for sequential recommendation tasks. It works by predicting the next user interaction based on past interactions in an autoregressive manner, allowing self-attention to attend only to earlier positions in the sequence. This means that the model respects causality, ensuring that future items are not considered when making predictions about the current or past items.

```java
public class SASRec {
    public void predictNextInteraction(float[][] historySequence) {
        // Predict next interaction using autoregressive self-attention mechanism
        ...
    }
}
```
x??

---
#### BERT4Rec Model Overview
BERT4Rec is an improvement over SASRec, inspired by the Bidirectional Encoder Representations from Transformers (BERT). It trains a bidirectional masked sequential model to predict masked items in the user-interaction sequence.

:p What distinguishes BERT4Rec from SASRec?
??x
BERT4Rec improves upon SASRec by training a bidirectional masked sequential model. This means that it can look at both past and future positions within the sequence, unlike SASRec which is autoregressive and only considers earlier positions. The bidirectional nature of BERT4Rec allows for more comprehensive learning of sequences.

```java
public class BERT4Rec {
    public void predictMaskedItems(float[][] maskedSequence) {
        // Predict masked items using a bidirectional self-attention mechanism
        ...
    }
}
```
x??

---

#### BERT4Rec vs. SASRec
Background context: The BERT4Rec model is an extension of the transformer architecture applied to sequential recommendation tasks, where it uses masked items for training. It outperforms SASRec but requires more computational resources and time due to its complexity.

:p What are the key differences between BERT4Rec and SASRec in terms of performance and resource requirements?
??x
BERT4Rec outperforms SASRec in terms of accuracy on sequential recommendation tasks by leveraging masked items for training, which allows it to capture long-term dependencies effectively. However, this comes at a cost: BERT4Rec is more computationally intensive and requires significantly more time to train compared to SASRec.

Bert4rec Example Training Process:
```python
# Pseudocode for BERT4Rec training
def bert4rec_train(model, data):
    # Masked item prediction
    masked_items = mask_items(data)
    
    # Forward pass through the model
    outputs = model(masked_items)
    
    # Compute loss and backpropagation
    loss = compute_loss(outputs, targets)
    model.backward(loss)
    
    # Update weights
    optimizer.step()
```
x??

---

#### Recency Sampling
Background context: To address the inefficiencies of autoregressive and masked training paradigms in sequential recommendation tasks, Petrov and Macdonald propose a recency-based sampling approach. This method aims to balance between capturing recent interactions while maintaining computational efficiency.

:p What is the primary goal of recency sampling in sequential recommendation systems?
??x
The primary goal of recency sampling is to improve training efficiency by giving more recent interactions higher chances of being sampled, thus balancing the need for capturing recent user behavior with reduced computational costs. It uses an exponential function that interpolates between masking-based and autoregressive sampling.

Recency Sampling Pseudocode:
```python
# Function to sample recency-based items
def sample_recency(sequence):
    # Calculate probabilities based on recency
    recency_probs = calculate_recency_probabilities(sequence)
    
    # Sample item based on probabilities
    sampled_item = np.random.choice(sequence, p=recency_probs)
    
    return sampled_item

# Example of calculating recency probabilities
def calculate_recency_probabilities(sequence):
    # Higher recent interactions have higher probability
    timestamps = [item['timestamp'] for item in sequence]
    max_time = max(timestamps)
    min_time = min(timestamps)
    
    def prob_func(time_diff):
        return np.exp(-time_diff / (max_time - min_time))
    
    recency_probs = [prob_func(max_time - t) for t in timestamps]
    total_prob = sum(recency_probs)
    normalized_probs = [p / total_prob for p in recency_probs]
    
    return normalized_probs
```
x??

---

#### PinnerFormer: Merging Static and Sequential Models
Background context: Pinterest's personalized recommendation system, PinnerFormer, integrates user historical actions into an end-to-end embedding model to predict both short-term and long-term user activities. It uses a transformer-based architecture on sequential data and combines it with GNN-based embeddings for improved feature richness.

:p What is the key difference between traditional sequential modeling tasks and PinnerFormer?
??x
The key difference between traditional sequential modeling tasks and PinnerFormer lies in their prediction horizons and objectives. Traditional models focus on predicting the immediate subsequent action, whereas PinnerFormer aims to predict extended future user activities over a window of 14 days following the embedding's generation.

PinnerFormer Example Training Process:
```python
# Pseudocode for training PinnerFormer
def train_pinnerformer(model, data):
    # Generate embeddings from historical actions
    user_embeddings = model.generate_embeddings(data)
    
    # Define future prediction target window (e.g., 14 days)
    target_window = 14
    
    # Train the model to predict future interactions within the target window
    for embedding, user in zip(user_embeddings, data['users']):
        positive_future_interactions = get_positive_future_interactions(user, target_window)
        dense_all_action_loss = compute_dense_all_action_loss(embedding, positive_future_interactions)
        
        optimizer.zero_grad()
        loss = dense_all_action_loss
        loss.backward()
        optimizer.step()
```
x??

---

