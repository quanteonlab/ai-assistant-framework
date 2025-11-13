# Flashcards: 2A005---Learn-Generative-AI-with-PyTorch-Manning-2024-Mark-Liu--_processed (Part 29)

**Starting Chapter:** 12 Training a Transformer to generate text

---

#### Scaled-Down GPT Model Architecture
Background context: In this chapter, a scaled-down version of the GPT model is constructed to make it trainable on regular computers. The original GPT-2XL model has 1.5 billion parameters and 48 decoder blocks with an embedding dimension of 1,600. To achieve a manageable parameter count, we reduce the number of decoder blocks to 3 and decrease the embedding dimension to 256, resulting in approximately 5 million parameters.

:p What are the key differences between the scaled-down GPT model and the original GPT-2XL model?
??x
The scaled-down GPT model has significantly fewer parameters (approximately 5 million) compared to the original GPT-2XL (1.5 billion). Specifically, it uses only 3 decoder blocks instead of 48, and its embedding dimension is reduced from 1,600 to 256.
x??

---

#### Training Data Selection
Background context: The training data for the scaled-down GPT model consists of three novels by Ernest Hemingway. These choices are made to ensure sufficient length and variation in the text while keeping the training practical.

:p Why were specific novels by Ernest Hemingway chosen as the training material?
??x
The novels by Ernest Hemingway were selected because they provide a substantial amount of text, allowing for effective learning, yet their style is relatively consistent. This selection helps the model to learn and mimic Hemingway’s distinctive writing style without overwhelming computational resources.
x??

---

#### Tokenization Process
Background context: Text data needs to be transformed into a format that can be processed by neural networks. In this case, the text is tokenized (split into words), and each unique word is mapped to an index using a dictionary.

:p How is the input sequence prepared for training in the GPT model?
??x
The input sequence is prepared by shifting it one token to the right. For example, if the input sequence is [10, 20, 30], the output would be [20, 30, 40] where each index represents a word in the dictionary. This process forces the model to predict the next word in a sentence based on the current token and all previous tokens.
x??

---

#### Training Epochs Determination
Background context: The number of training epochs is crucial for balancing model performance with overfitting. Too few epochs might result in underfitting, while too many could lead to overfitting.

:p How are the optimal number of epochs determined for training the GPT model?
??x
The optimal number of epochs is often determined by a balance between minimizing cross-entropy loss and avoiding overfitting. In this case, 40 epochs were chosen as they provide a good balance. The model was saved at 10-epoch intervals to evaluate its performance in generating coherent text without copying passages from the training material.
x??

---

#### Text Generation with Autoregressive Approach
Background context: After training, the GPT model can generate text autoregressively, meaning it predicts the next word based on the current and previous words.

:p How is text generated using an autoregressive approach in the GPT model?
??x
Text generation uses an autoregressive approach where the model takes a sequence of 128 indexes as input and predicts the next token (word) at each step. The process starts with a seed sequence, and for each new word, it uses the entire context up to that point to predict the next word.
```python
# Pseudocode for text generation
def generate_text(model, start_sequence):
    current_sequence = start_sequence
    generated_text = ""
    
    while len(generated_text) < desired_length:
        # Predict the next token given the current sequence
        next_token_index = model.predict(current_sequence)
        
        # Append the predicted token to the generated text
        generated_text += model.tokenizer.decode(next_token_index)
        
        # Shift the sequence by one and append the new token
        current_sequence = np.append(current_sequence[1:], next_token_index)
    
    return generated_text
```
x??

---

#### Architecture of a GPT Model for Text Generation
GPT models, including the one we are constructing, follow a specific architecture that allows them to generate coherent text. This model is based on the Transformer architecture with an emphasis on self-attention mechanisms and feed-forward networks.

:p What is the key feature of the GPT model being constructed in this chapter?
??x
The key feature of the GPT model being constructed in this chapter is its architecture, which follows a similar design to the GPT-2 models but is significantly smaller. It includes three decoder layers with an embedding dimension of 256, four attention heads for causal self-attention, and a maximum sequence length of 128. The model's structure ensures that it can be trained on standard computing resources.

```java
public class GPTModel {
    private int embeddingDimension;
    private int decoderLayers;
    private List<DecoderLayer> layers;

    public GPTModel(int embeddingDimension, int decoderLayers) {
        this.embeddingDimension = embeddingDimension;
        this.decoderLayers = decoderLayers;
        // Initialize the model architecture here
    }
}
```
x??

---

#### Hyperparameters of the GPT Model
Hyperparameters are crucial for defining the behavior and performance of a machine learning model. In our case, we need to determine several key hyperparameters such as sequence length, embedding dimension, number of decoder blocks, and dropout rates.

:p What are some important hyperparameters when building a GPT model?
??x
Important hyperparameters when building a GPT model include:
- **Embedding Dimension**: The size of the vector that represents each token. In our case, it is set to 256.
- **Number of Decoder Layers**: The number of decoder layers in the model, which we have chosen as 3 for this task.
- **Sequence Length**: The maximum length of a sequence, which we set to 128.

These hyperparameters significantly influence both the quality of the output and the speed of training. For instance, a longer sequence can capture more context but increases the complexity and computational requirements.

```java
public class HyperParameters {
    private int embeddingDim;
    private int decoderLayers;
    private int seqLength;

    public HyperParameters(int embeddingDim, int decoderLayers, int seqLength) {
        this.embeddingDim = embeddingDim;
        this.decoderLayers = decoderLayers;
        this.seqLength = seqLength;
    }
}
```
x??

---

#### Training Data Preparation
To train the GPT model, we need to prepare the training data by tokenizing the raw text and converting it into sequences of integers. Each unique word or token in the input text is assigned a different integer.

:p How do you tokenize and convert raw text into indexes for the GPT model?
??x
Tokenization involves breaking down the raw text into individual tokens (words, punctuation, etc.). We then assign each unique token an integer index to feed it into the model. This process ensures that the input data is structured in a way that can be processed by the Transformer.

Here’s how you might implement this in Java:

```java
public class Tokenizer {
    private Map<String, Integer> vocabulary;

    public Tokenizer(List<String> trainingText) {
        // Create vocabulary map from unique tokens in the text
        this.vocabulary = createVocabulary(trainingText);
    }

    private Map<String, Integer> createVocabulary(List<String> trainingText) {
        Map<String, Integer> vocabMap = new HashMap<>();
        int index = 0;
        for (String token : trainingText) {
            if (!vocabMap.containsKey(token)) {
                vocabMap.put(token, index++);
            }
        }
        return vocabMap;
    }

    public int getIndex(String token) {
        return vocabulary.getOrDefault(token, 0); // Default to 0 if token not found
    }
}
```
x??

---

#### Training the GPT Model
Once we have prepared the training data, the next step is to train the model. This involves feeding batches of sequences into the model and adjusting the weights through backpropagation.

:p What are the steps involved in training a GPT model?
??x
Training a GPT model involves several key steps:
1. **Prepare Training Data**: Tokenize the raw text, convert it into indexes, and split into sequences.
2. **Initialize Model Architecture**: Set up the architecture with the chosen hyperparameters (embedding dimension, decoder layers, sequence length).
3. **Train the Model**: Feed batches of sequences to the model, adjust weights through backpropagation, and optimize the loss function.

Here’s a simplified version of how you might train the model in Java:

```java
public class GPTTrainer {
    private GPTModel model;
    private Tokenizer tokenizer;

    public GPTTrainer(GPTModel model, Tokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
    }

    public void train(List<List<Integer>> batches, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (List<Integer> batch : batches) {
                // Forward pass
                List<Integer> predictions = model.predict(batch);

                // Backward pass and weight adjustment
                model.adjustWeights(predictions);
            }
        }
    }
}
```
x??

---

#### Generating Text Using the Model
After training, the GPT model can generate text based on a prompt. The process involves converting the input text into indexes, feeding it to the model, and iteratively predicting the next token.

:p How does the GPT model generate text from a given prompt?
??x
The GPT model generates text from a given prompt by following these steps:
1. **Tokenize Prompt**: Convert the input text (prompt) into a sequence of indexes.
2. **Predict Next Token**: Use the model to predict the next token in the sequence based on the current token and all previous tokens.
3. **Iterate Prediction**: Repeat the prediction process for each subsequent token until the generated sequence reaches a predefined length or end-of-sequence marker.

Here’s how you might implement this in Java:

```java
public class TextGenerator {
    private GPTModel model;
    private Tokenizer tokenizer;

    public TextGenerator(GPTModel model, Tokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
    }

    public String generateText(String prompt, int maxLength) {
        List<Integer> inputSequence = tokenizer.getIndex(prompt);
        List<Integer> generatedTokens = new ArrayList<>();
        
        for (int token : inputSequence) {
            generatedTokens.add(token);
            if (generatedTokens.size() >= maxLength) break;
        }
        
        while (generatedTokens.size() < maxLength) {
            int nextTokenIndex = model.predictNextToken(generatedTokens.subList(generatedTokens.size() - 256, generatedTokens.size()));
            generatedTokens.add(nextTokenIndex);
        }

        return tokenizer.decodeIndexSequence(generatedTokens);
    }
}
```
x??

---

#### Differences Between GPT and GPT-2 Models
There are several differences between the GPT model being constructed in this chapter and the larger GPT-2 models. These include variations in hyperparameters such as embedding dimension, number of layers, sequence length, and vocabulary size.

:p How do the GPT and GPT-2 models differ?
??x
The GPT and GPT-2 models differ primarily in their hyperparameters:
- **Embedding Dimension**: GPT has an embedding dimension of 256, while GPT-2 can range from 768 to 1,600.
- **Number of Decoder Layers**: GPT uses three decoder layers, whereas GPT-2 models use between 12 and 48 layers.
- **Sequence Length**: GPT’s maximum sequence length is 128, compared to up to 1,024 in GPT-2.
- **Vocabulary Size**: GPT has a vocabulary size of 10,600, whereas GPT-2 can have a much larger vocabulary.

These differences affect the model's complexity and training requirements. A smaller GPT model like ours is more suitable for real-world applications where computational resources are limited.

```java
public class ModelComparison {
    private int embeddingDimGpt;
    private int decoderLayersGpt;
    private int seqLengthGpt;
    private int vocabSizeGpt;

    public ModelComparison() {
        this.embeddingDimGpt = 256; // GPT
        this.decoderLayersGpt = 3; // GPT
        this.seqLengthGpt = 128;   // GPT
        this.vocabSizeGpt = 10600; // GPT
    }
}
```
x??

---

#### Text Tokenization and Indexing
Background context: The process of converting raw text into a numerical form is essential for training machine learning models. In this case, we are using word-level tokenization to break down the text from three Hemingway novels into tokens (words or punctuation marks) and assigning each token a unique index.

:p What is the purpose of tokenizing and indexing in the context of training a GPT model?
??x
The purpose is to convert raw text data into a numerical form that can be processed by the machine learning model. By tokenizing, we break down the text into meaningful units (words or punctuation marks), which are then indexed with unique integers. This allows us to represent the input and target sequences as sequences of integers.

```python
# Example code for tokenization and indexing
import re

def tokenize_and_index(text):
    tokens = re.findall(r'\b\w+\b|[.,!?;]', text)  # Tokenize using regex
    token_to_index = {token: idx for idx, token in enumerate(set(tokens))}
    indexed_tokens = [token_to_index[token] for token in tokens]
    
    return indexed_tokens, token_to_index

# Example usage
text = "the old man and the sea"
tokens, token_to_index = tokenize_and_index(text)
print("Tokens:", tokens)
print("Token to Index Mapping:", token_to_index)
```
x??

---

#### Sequence Length for Training Data
Background context: The sequence length is a crucial hyperparameter in training text generation models. In this case, we are setting the maximum sequence length to 128 indexes per sequence. This choice allows capturing long-range dependencies while keeping the model manageable.

:p What is the rationale behind choosing a sequence length of 128 for the GPT model?
??x
The rationale behind choosing a sequence length of 128 is to balance between capturing long-range dependencies and maintaining a manageable model size. A longer sequence can capture more context, but it also increases computational complexity and memory requirements.

```python
# Example code demonstrating sequence creation
def create_sequences(indexed_tokens):
    sequences = []
    for i in range(len(indexed_tokens) - 128):
        seq = indexed_tokens[i:i+128]
        if len(seq) == 128:
            sequences.append(seq)
    
    return sequences

# Example usage
indexed_tokens = [0, 17, 16, 2, 0, 102, ...]  # Assume this is the indexed tokens list
sequences = create_sequences(indexed_tokens)
print("Number of sequences:", len(sequences))
```
x??

---

#### Training Process Overview
Background context: The training process for a GPT model involves feeding input sequences into the model and using the output to calculate loss, which guides the learning process. This is an iterative process that aims to minimize the cross-entropy loss between predicted probabilities and ground truth.

:p What are the main steps involved in the training process of a GPT model?
??x
The main steps involved in the training process of a GPT model include:

1. **Tokenize & Index**: Convert raw text into tokens and assign unique indices.
2. **Create Input Sequences**: Break down the indexed text into sequences of fixed length (e.g., 128).
3. **Generate Output Sequences**: Shift the input sequence by one token to predict the next token.
4. **Model Prediction**: Use the GPT model to predict the next token in the shifted sequence.
5. **Calculate Cross-Entropy Loss**: Compare the predicted probabilities with the ground truth labels and calculate the loss.
6. **Backpropagation and Optimization**: Update the model parameters based on the calculated loss.

```python
# Example code for training steps
def train_step(input_sequences, model):
    outputs = []  # Placeholder for actual output sequences
    
    for seq in input_sequences:
        predicted_token = model.predict(seq[:-1])  # Predict next token
        true_next_token = seq[1:]  # Ground truth next token
        loss = cross_entropy_loss(predicted_token, true_next_token)  # Calculate loss
        outputs.append(true_next_token)
        
        # Backpropagation and optimization steps would follow here

# Example usage (simplified for illustration)
input_sequences = [[0, 17, 16, ...], [16, 2, 0, 102, ...]]  # Assume these are input sequences
train_step(input_sequences, model)
```
x??

---

#### Cross-Entropy Loss Calculation
Background context: Cross-entropy loss is a common metric used to evaluate the performance of probabilistic classifiers. In this case, it helps measure how well the GPT model predicts the next token in the sequence.

:p How is cross-entropy loss calculated for a single output and ground truth pair?
??x
Cross-entropy loss is calculated using the formula:

$$\text{loss} = -\sum_{i=1}^{n} y_i \log(p_i)$$where $ y_i $ is the true probability (1 if the token matches, 0 otherwise), and $ p_i$ is the predicted probability.

```python
import torch

def cross_entropy_loss(predicted_probs, ground_truth):
    # Convert ground truth to one-hot encoding
    one_hot = torch.zeros_like(predicted_probs)
    one_hot.scatter_(1, ground_truth.unsqueeze(1), 1)
    
    # Calculate log probabilities and compute loss
    log_probabilities = torch.log(predicted_probs + 1e-8)  # Add small epsilon for numerical stability
    loss = -torch.sum(one_hot * log_probabilities)
    
    return loss

# Example usage (simplified for illustration)
predicted_probs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])  # Predicted probabilities
ground_truth = torch.tensor([2, 1])  # Ground truth indices
loss = cross_entropy_loss(predicted_probs, ground_truth)
print("Cross-entropy loss:", loss.item())
```
x??

---

#### Concept: Input and Output Sequences for Training
Background context explaining how input and output sequences are created for training. The input sequence is shifted one token to the right to serve as the output, creating pairs of (input, output) used as training data.

:p How do we create the input and output sequences for training in this example?
??x
In this process, we take a sentence like "the old man and the sea" and break it into tokens. For training, we use "the old man and the" as the input sequence (x) and "old man and the sea" as the output sequence (y). During each time step, the model predicts the next token based on the current context.

For instance:
- In the first time step, "the" is used to predict "old."
- In the second time step, "the old" is used to predict "man," and so forth.
??x
This approach helps the model learn the relationships between tokens in a sequence. The training data consists of pairs (x, y), where x is the shifted input and y is the actual next token.

```python
# Example pseudocode for creating input-output sequences
def create_sequences(sentence):
    tokens = sentence.split()
    inputs = []
    outputs = []
    for i in range(len(tokens) - 1):
        inputs.append(tokens[i:i+1])
        outputs.append(tokens[i+1:i+2])
    return inputs, outputs

sentence = "the old man and the sea"
inputs, outputs = create_sequences(sentence)
print("Inputs:", inputs)
print("Outputs:", outputs)
```
x??

---

#### Concept: Training Process Overview
Background context explaining the training process, including forward passes, loss computation, and parameter adjustment.

:p What is the general outline of the training process for the GPT model?
??x
The training process involves several steps:
1. **Forward Pass**: Feed the input sequence (x) through the GPT model.
2. **Prediction**: The model makes a prediction based on its current parameters.
3. **Loss Computation**: Compute the cross-entropy loss by comparing the predicted next tokens with the actual output from step 3.
4. **Parameter Adjustment**: Adjust the model's parameters to minimize the cross-entropy loss.

This process is repeated through many iterations, and the trained model is saved after every 10 epochs.

```python
# Example pseudocode for training steps
def train_gpt(model, inputs, outputs, epochs=40):
    for epoch in range(epochs):
        # Forward pass
        predictions = model(inputs)
        
        # Loss computation
        loss = compute_loss(predictions, outputs)
        
        # Backward pass and parameter adjustment
        adjust_parameters(model, loss)

# Example of computing cross-entropy loss (pseudocode)
def compute_cross_entropy_loss(predictions, targets):
    loss = 0.0
    for i in range(len(targets)):
        p = predictions[i]
        t = targets[i]
        # Cross-entropy formula: -sum(t * log(p))
        loss -= t * np.log(p)
    return loss

# Example of adjusting parameters (pseudocode)
def adjust_parameters(model, loss):
    model.optimize(loss)
```
x??

---

#### Concept: Tokenizing Text for Training
Background context explaining the importance of tokenization and indexing text before feeding it into a GPT model. Emphasizes word-level tokenization due to its simplicity and efficiency.

:p What is the first step in preparing text data for training with a GPT model?
??x
The first step involves breaking down the text into individual tokens (words) and creating an index for each token. This process converts raw text into integers, which can be processed by deep neural networks. Word-level tokenization is chosen due to its simplicity and efficiency in handling text data.

```python
# Example pseudocode for tokenizing text
def tokenize_text(text):
    # Split the text into words (tokens)
    tokens = text.split()
    
    # Create a dictionary mapping each unique token to an index
    vocab = {token: idx for idx, token in enumerate(set(tokens))}
    
    # Map tokens to indices
    indexed_tokens = [vocab[token] for token in tokens]
    
    return indexed_tokens

# Example usage
text = "the old man and the sea"
indexed_text = tokenize_text(text)
print("Indexed Text:", indexed_text)
```
x??

---

#### Loading and Cleaning Text Files
Background context: The process of loading and cleaning raw text files for training a GPT model involves several steps. These include downloading specific text files from a repository, removing irrelevant information, and ensuring that the text is properly formatted.

:p How do you load and clean the text file for "The Old Man and the Sea"?
??x
To load and clean the text file for "The Old Man and the Sea," you first need to open the file in read mode. The next step involves converting the text into a list of characters, checking each character for double quotes, and replacing them with appropriate opening or closing quotes where necessary.

Here is the code snippet that performs this operation:

```python
with open("files/OldManAndSea.txt", "r", encoding='utf-8-sig') as f:
    text = f.read()
text = list(text)
for i in range(len(text)):
    if text[i] == '\"':
        if text[i + 1] == ' ' or text[i + 1] == '\n':
            text[i] = '\"'
    elif text[i] == "'":
        if text[i - 1] == ' ':
            text[i] = "'"
text = "".join(text)
```

This code first reads the entire file and converts it into a list of characters. Then, it iterates through each character to identify double quotes that need to be converted. If a double quote is followed by a space or line break, it changes it to a closing quote; otherwise, it keeps it as an opening quote.

x??

---

#### Combining Text Files
Background context: After cleaning the individual text files, the next step involves combining them into a single file that can be used for training the GPT model. This step ensures that all necessary texts are included and properly formatted.

:p How do you combine the cleaned text from three novels?
??x
To combine the text from "The Old Man and the Sea," "A Farewell to Arms," and "For Whom the Bell Tolls" into a single file, you need to read each of these files separately, concatenate their content with appropriate spacing, and then write this combined text to a new file.

Here is the code snippet that performs this operation:

```python
with open("files/ToWhomTheBellTolls.txt", "r", encoding='utf-8-sig') as f:
    text1 = f.read()
with open("files/FarewellToArms.txt", "r", encoding='utf-8-sig') as f:
    text2 = f.read()
text += " " + text1 + " " + text2
with open("files/ThreeNovels.txt", "w", encoding='utf-8-sig') as f:
    f.write(text)
print(text[:250])
```

This code first reads the content of each file and stores it in variables `text1` and `text2`. It then concatenates these texts with spaces in between. Finally, it writes this combined text to a new file named "ThreeNovels.txt" and prints out the first 250 characters for verification.

x??

---

#### Tokenizing Text
Background context: Before feeding the text into the GPT model, it is crucial to tokenize the text. This process involves breaking down the raw text into smaller units (tokens) that can be processed by the model. In this case, the text is broken down into individual characters, which are then used for training.

:p How do you load up and break the text from "The Old Man and the Sea" into individual characters?
??x
To tokenize the text from "The Old Man and the Sea," you need to open the file in read mode, read its content, convert it into a list of characters, and then process these characters as needed. Here is how you can do it:

```python
with open("files/OldManAndSea.txt", "r", encoding='utf-8-sig') as f:
    text = f.read()
text = list(text)
# Further processing of text goes here (e.g., changing quotes to opening or closing quotes).
```

This code snippet opens the file in read mode, reads its content, and converts it into a list of characters. This step is essential for further processing such as replacing straight quotes with appropriate opening or closing quotes.

x??

---

These flashcards cover the key concepts involved in preparing text data for training a GPT model, including loading, cleaning, and tokenizing text files.

#### Reading Text from Multiple Novels
Background context: This section describes how to read and combine text from three novels by Ernest Hemingway. The combined text is stored locally for verification purposes, ensuring that the generated content can be compared against the original.

:p How do you read and save the text from multiple Hemingway novels?
??x
The process involves reading the text from each novel, combining them into a single file, and saving it locally. This step ensures that the combined text is stored for later verification to ensure no direct copying was done.
```python
# Example pseudocode
def read_and_save_novels():
    # Read text from second and third novels
    with open('novel2.txt', 'r') as f:
        novel2_text = f.read()
    
    with open('novel3.txt', 'r') as f:
        novel3_text = f.read()
    
    # Combine the texts
    combined_text = original_novel1_text + novel2_text + novel3_text
    
    # Save the combined text to a file
    with open('ThreeNovels.txt', 'w') as f:
        f.write(combined_text)
```
x??

---

#### Tokenizing Text of Hemingway Novels
Background context: This section explains how to prepare the text by tokenizing it. The goal is to clean and tokenize the combined text from three novels, ensuring better model training.

:p What steps are involved in cleaning and tokenizing the text?
??x
The steps include converting the entire text to lowercase, replacing line breaks with spaces, adding spaces around punctuation marks, and counting unique tokens.

```python
# Example pseudocode for cleaning and tokenizing
text = "your combined text here"
text = text.lower()  # Convert all characters to lowercase
chars = set(text)
punctuations = [i for i in chars if not i.isalpha() and not i.isdigit()]  # Identify punctuation

for x in punctuations:
    text = text.replace(f"{x}", f" {x} ")

text_tokenized = text.split()
unique_tokens = set(text_tokenized)
print(len(unique_tokens))  # Output the number of unique tokens
```
x??

---

#### Adding UNK Token for Unknown Tokens
Background context: The inclusion of a special "UNK" token is crucial to handle unknown words in the prompt. This ensures that the model can process any new word not included in its vocabulary.

:p Why do you need to add an "UNK" token?
??x
The "UNK" (Unknown) token is necessary because it allows the model to handle and process words that were not present during training. Without this token, if a prompt contains a new or unknown word, the program would crash. By including "UNK", the model can map such tokens to an index, preventing crashes.

```python
# Example pseudocode for adding UNK token
from collections import Counter

word_counts = Counter(text_tokenized)
words = sorted(word_counts, key=word_counts.get, reverse=True)  # Sort by frequency
words.append("UNK")  # Add the "UNK" token to the list of words
text_length = len(text_tokenized)

# Print or use these words for further processing
print(words)
```
x??

---

#### Mapping Tokens to Indexes
Background context: After tokenizing and cleaning the text, mapping tokens to indexes helps in preparing the data for training. This involves counting the frequency of each token and assigning unique indices.

:p How do you map tokens to their respective indexes?
??x
Mapping tokens to indexes involves using a `Counter` from the `collections` module to count the occurrences of each word. Then, sort these words by frequency (in descending order) and add "UNK" as an additional token.

```python
from collections import Counter

word_counts = Counter(text_tokenized)
words = sorted(word_counts, key=word_counts.get, reverse=True)

# Add the "UNK" token to handle unknown words
words.append("UNK")

# Print or use these words for further processing
print(words)  # Output will be a list of tokens including "UNK"
```
x??

---

