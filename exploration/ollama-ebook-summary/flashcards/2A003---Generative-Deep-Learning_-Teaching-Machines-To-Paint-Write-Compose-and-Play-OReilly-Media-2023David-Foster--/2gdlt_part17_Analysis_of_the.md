# Flashcards: 2A003---Generative-Deep-Learning_-Teaching-Machines-To-Paint-Write-Compose-and-Play-OReilly-Media-2023David-Foster--_processed (Part 17)

**Starting Chapter:** Analysis of the LSTM

---

#### Temperature Parameter in Text Generation
Background context: The temperature parameter influences how deterministic or stochastic the word sampling process is during text generation. A lower temperature makes the selection more deterministic (choosing the highest probability), while a higher temperature increases randomness, making it less deterministic.

:p What does the temperature parameter control in the text generation process?
??x
The temperature parameter controls the level of randomness in selecting the next word during text generation. It affects how the model samples from the probability distribution:
- A lower temperature (e.g., 0) makes the selection more deterministic, choosing the highest-probability word.
- A higher temperature (e.g., 1) makes each word more likely to be chosen according to its probability.

This is achieved by scaling the probabilities with a power operation and then normalizing them: 
```python
probs = probs ** (1 / temperature)
probs = probs / np.sum(probs)
```
x??

---

#### TextGenerator Callback Function
Background context: The `TextGenerator` callback function is designed to generate text at the end of each training epoch. It converts words into tokens and uses a sampling method based on temperature to produce new text.

:p What is the purpose of the `TextGenerator` class?
??x
The `TextGenerator` class serves to generate text using the trained LSTM model after each training epoch. Its primary function involves converting input prompts into sequences, generating new tokens based on the model's output probabilities, and appending these tokens back to the prompt.

Here is a simplified version of how it works:
```python
class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {word: index for index, word in enumerate(index_to_word)}

    def sample_from(self, probs, temperature):
        # Adjust probabilities based on temperature and normalize them
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        # Convert the prompt into tokens
        start_tokens = [self.word_to_index.get(x, 1) for x in start_prompt.split()]
        
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y = self.model.predict(x)
            
            # Sample the next token based on temperature
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            
            info.append({'prompt': start_prompt, 'word_probs': probs})
            start_tokens.append(sample_token)
            
            start_prompt += f' {self.index_to_word[sample_token]}'

        print(f"generated text: {start_prompt}")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("recipe for", max_tokens=100, temperature=1.0)
```
x??

---

#### Sampling Process in Text Generation
Background context: The sampling process involves converting the output probabilities of the LSTM model into a token selection that generates new text. This process is stochastic and can be controlled by adjusting the temperature.

:p How does the sampling process work in generating text?
??x
The sampling process works by taking the output probabilities from the LSTM model, scaling them according to a given temperature, normalizing the result, and then choosing a token based on these probabilities. The higher the temperature, the more random the selection; the lower the temperature, the more deterministic it becomes.

Here is an example of how the sampling process works in code:
```python
def sample_from(self, probs, temperature):
    # Adjust probabilities based on temperature and normalize them
    probs = probs ** (1 / temperature)
    probs = probs / np.sum(probs)
    
    return np.random.choice(len(probs), p=probs), probs
```
The `sample_from` method takes the model's output probabilities, scales them by the inverse of the temperature, and then normalizes these values to get a probability distribution. It uses this distribution to select the next token using `np.random.choice`.

x??

---

#### Text Generation Process Overview
Background context: The text generation process involves feeding an existing sequence of words into the LSTM model, which predicts the following word based on its learned patterns. This new word is then appended to the sequence and the process repeats.

:p What is the basic process for generating text using an LSTM model?
??x
The basic process for generating text using an LSTM model starts by feeding a sequence of existing words into the network. The network predicts the next word in the sequence based on its learned patterns from the training data. This predicted word is then appended to the original sequence, and the process repeats until a certain maximum length is reached or a stop condition (like a token with value 0) is met.

Here is an example of how this process works:
```python
def generate(self, start_prompt, max_tokens, temperature):
    # Convert the prompt into tokens
    start_tokens = [self.word_to_index.get(x, 1) for x in start_prompt.split()]
    
    sample_token = None
    info = []
    while len(start_tokens) < max_tokens and sample_token != 0:
        x = np.array([start_tokens])
        y = self.model.predict(x)
        
        # Sample the next token based on temperature
        sample_token, probs = self.sample_from(y[0][-1], temperature)
        
        info.append({'prompt': start_prompt, 'word_probs': probs})
        start_tokens.append(sample_token)
        
        start_prompt += f' {self.index_to_word[sample_token]}'

    print(f"generated text: {start_prompt}")
    return info
```
This method converts the initial prompt into tokens, predicts and appends new tokens based on temperature until a maximum length is reached or a stop condition occurs.

x??

---

#### Contextual Understanding of Recipe Instructions
Background context explaining how the model selects appropriate verbs based on the recipe title. The model uses an LSTM to generate instructions that are suitable for specific ingredients, such as "preheat" for roasted vegetables and "mix" for ice cream.

:p How does the model determine the initial verb in recipe instructions?
??x
The model determines the initial verb by leveraging contextual information from the preceding title. It selects verbs like "preheat," "prepare," or "heat" for roasted vegetables, indicating a need to pre-temperature settings. For ice cream, it chooses verbs related to mixing and combining ingredients.

For example:
```java
if (title.contains("vegetables")) {
    // Select appropriate verbs based on the context of vegetables
}
else if (title.contains("ice cream")) {
    // Select appropriate verbs based on the context of ice cream
}
```
x??

---

#### Temperature Parameter Influence on Generation Diversity
Explanation of how the temperature parameter affects the model's generation process. Lower temperatures result in less diverse outputs because tokens are more heavily weighted toward higher probabilities.

:p How does the temperature parameter affect the model’s output diversity?
??x
The temperature parameter influences the softmax function, which determines the probability distribution over possible next tokens. A lower temperature makes the probabilities of tokens less uniform, leading to a higher likelihood of selecting the most probable token. This results in fewer diverse outputs when the temperature is low.

For example:
```java
public double[] applyTemperature(double temp) {
    List<Double> logprobs = getLogProbs(); // Get log-probabilities from model output
    return softmax(logprobs, temp); // Apply temperature scaling to probabilities
}
```
x??

---

#### Basic LSTM Model Limitations
Explanation of the basic LSTM model's limitations in generating realistic text and handling semantic meaning.

:p What are some limitations of the basic LSTM model?
??x
The basic LSTM model struggles with understanding semantic meanings and can generate ingredients that do not work well together. For instance, it might suggest an unusual combination like "sour Japanese potatoes, pecan crumbs, and sorbet." This limitation arises because LSTMs primarily focus on generating text in a given style without deeper semantic understanding.

To address this issue, the model needs to have better memory of previous content and a deeper understanding of how words should be grouped together.

x??

---

#### Stacked Recurrent Networks
Explanation of using multiple LSTM layers to learn more complex features from text. The idea is to pass hidden states from one layer as input data to another, allowing for the learning of deeper features.

:p What is the benefit of using stacked LSTM layers in a network?
??x
Using stacked LSTM layers allows the model to learn more complex and abstract features from the text by processing it through multiple layers. Each additional LSTM layer can capture higher-level abstractions that are not apparent in single-layer models.

For example, a two-layer RNN architecture:
```java
public class StackedRNN {
    private List<LSTM> layers;

    public void forwardPropagate(List<Double> input) {
        for (LSTM layer : layers) {
            input = layer.computeHiddenStates(input);
        }
    }

    // Each LSTM layer processes the hidden states from the previous layer.
}
```
x??

---

#### Model Architecture of Stacked RNN
Illustration of how multiple LSTM layers work together in a stacked recurrent network.

:p How does a multilayer RNN process inputs?
??x
In a multilayer RNN, each LSTM layer processes the hidden states from the previous layer. The first LSTM layer receives the input sequence and produces its own set of hidden states. These hidden states are then fed into the second LSTM layer as input data, allowing it to learn deeper features.

For example:
```java
public class MultilayerRNN {
    private List<LSTM> layers;

    public List<Double> processSequence(List<Double> input) {
        for (LSTM layer : layers) {
            input = layer.process(input);
        }
        return input;
    }
}
```
x??

---

---
#### Stacked LSTM Layer Overview
Stacked Long Short-Term Memory (LSTM) layers are used to build deep RNN architectures. The provided table shows that there are three LSTM layers stacked, each with 128 units.

The input shape is `(None, None)` indicating variable length sequences of vectors, and the embedding layer converts these into a higher-dimensional space.

The total number of parameters for the model is 2,538,832, all of which are trainable. The layers are as follows:
1. **InputLayer**: Converts input to `(None, None, 100)`.
2. **Embedding Layer**: Transforms each sequence element into a dense vector of size 100.
3. **First LSTM Layer**: Processes the embedded sequences and outputs another sequence with 128 units.
4. **Second LSTM Layer**: Again processes the output from the first LSTM, this time also producing a sequence with 128 units.
5. **Dense Layer**: Converts the final sequence to a vector of size `total_words`, using softmax activation for probability distribution over words.

:p What is the structure and parameter count of the model described?
??x
The model has three stacked LSTM layers, each with 128 units, followed by an embedding layer that converts inputs into 100-dimensional vectors. The total number of parameters in the model is 2,538,832.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=embedding_size, input_length=None))
model.add(LSTM(units=n_units, return_sequences=True))
model.add(LSTM(units=n_units, return_sequences=True))
model.add(Dense(total_words, activation='softmax'))

# Example of model summary
model.summary()
```
x??

---
#### Gated Recurrent Unit (GRU) Introduction
Gated Recurrent Units (GRUs) are another type of RNN layer. Unlike LSTMs, GRUs use reset and update gates instead of forget and input gates. They also have no cell state or output gate; only a hidden state that is updated.

:p What are the key differences between LSTM and GRU?
??x
The key differences between LSTM and GRU include:
1. **Gates**: LSTMs use forget and input gates, whereas GRUs use reset and update gates.
2. **Cell State**: LSTMs have a cell state that can carry information over long sequences, while GRUs do not have this concept; they directly update the hidden state based on inputs.
3. **Activation Steps**: In GRUs, there are four main steps for updating the hidden state: reset gate creation, applying the reset gate to the previous hidden state and current input, generating a new cell state, and then calculating the final updated hidden state.

x??

---
#### GRU Process Explanation
The process in a single GRU cell involves several key steps. These are:
1. **Reset Gate**: The concatenation of the hidden state from the previous time step (`ht-1`) and the current input embedding (`xt`) is used to create the reset gate. This operation uses weights `Wr` followed by a sigmoid activation function.
2. **Applying Reset Gate**: The reset gate (a vector with length equal to the number of units) determines how much of the previous hidden state should be carried forward into the new calculation for the cell's beliefs.

:p What is the role and mechanism of the reset gate in GRU?
??x
The reset gate in GRU plays a crucial role by determining which parts of the previous hidden state `ht-1` are to be passed on to the current time step. It does this through the following steps:
1. Concatenate `ht-1` and `xt`, where `xt` is the current input embedding.
2. This concatenated vector is used as input for a dense layer with weights `Wr`.
3. The output of this dense layer undergoes a sigmoid activation, resulting in a vector `rt` (reset gate) that contains values between 0 and 1.

This mechanism allows the model to decide which parts of the previous state should be discarded or retained based on the current input.
```python
# Pseudocode for creating the reset gate
def create_reset_gate(ht_minus_1, xt):
    concatenated = tf.concat([ht_minus_1, xt], axis=-1)
    W_r = tf.Variable(tf.random.normal(shape=(units * 2, units)))
    r_t = tf.sigmoid(tf.matmul(concatenated, W_r))
    return r_t
```
x??

---
#### GRU Hidden State Update
After the reset gate is applied, it is used to update the hidden state. This involves:
1. **Applying Reset Gate**: Multiplying `ht-1` by the reset gate vector.
2. **Generating a New Belief Vector (`ht`)**: Concatenating `ht-1`, `xt`, and applying another dense layer with weights `W` to produce a new belief vector, which has values between -1 and 1.

:p How is the hidden state updated in GRU?
??x
The update of the hidden state in GRU involves:
1. **Applying Reset Gate**: The previous hidden state `ht-1` is multiplied by the reset gate `rt`, resulting in a vector that retains parts of the previous hidden state.
2. **Generating New Beliefs (`ht`)**: Concatenating `ht-1 * rt` and `xt`, this concatenated vector passes through another dense layer with weights `W` and a tanh activation function to produce the new hidden state `ht`.

This process ensures that the model can adaptively decide which parts of the previous state are relevant for the current computation.
```python
# Pseudocode for updating the hidden state in GRU
def update_hidden_state(ht_minus_1, xt, rt):
    h_tilde = tf.tanh(tf.matmul(tf.concat([ht_minus_1 * rt, xt], axis=-1), W))
    h_t = (1 - rt) * ht_minus_1 + rt * h_tilde
    return h_t
```
x??

---

