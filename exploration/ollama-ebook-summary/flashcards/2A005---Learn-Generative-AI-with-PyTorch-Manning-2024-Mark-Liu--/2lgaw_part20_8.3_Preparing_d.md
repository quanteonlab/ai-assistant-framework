# Flashcards: 2A005---Learn-Generative-AI-with-PyTorch-Manning-2024-Mark-Liu--_processed (Part 20)

**Starting Chapter:** 8.3 Preparing data to train the LSTM model. 8.3.1 Downloading and cleaning up the text

---

#### Downloading and Cleaning Text Data
Background context: This section explains how to download and clean text data for training an LSTM model. The primary text used is from "Anna Karenina," which will be downloaded, cleaned, and tokenized.

:p How do you load and display the first 20 words of the "Anna Karenina" text file?
??x
To load and display the first 20 words, use the following code:
```python
with open("files/anna.txt", "r") as f:
    text = f.read()
    words = text.split(" ")
print(words[:20])
```
This script reads the text from a file named `anna.txt`, splits it into individual words based on spaces, and prints out the first 20 of them. This helps in understanding the dataset's structure.
x??

---

#### Tokenizing Text Data
Background context: After loading the data, the next step is to tokenize the text into individual tokens and then create a dictionary that maps each token to an index.

:p How do you clean up the downloaded text data?
??x
To clean up the text data, use the following code:
```python
clean_text = text.lower().replace("\n", " ")
clean_text = clean_text.replace("-", " ")
for x in [",.:;?.$()/_& percent*@'`\""]:
    clean_text = clean_text.replace(f"{x}", f" {x} ")

clean_text = clean_text.replace('"', ' " ')
text = clean_text.split()
```
This code converts all characters to lowercase, replaces line breaks and hyphens with spaces, adds a space around punctuation marks and special characters. This ensures that the text is in a consistent format suitable for tokenization.
x??

---

#### Creating Dictionaries
Background context: After cleaning up the text data, we create dictionaries that map tokens to indices and vice versa.

:p How do you create mappings between unique tokens and their indexes?
??x
To create these mappings, use the following code:
```python
from collections import Counter

word_counts = Counter(text)
words = sorted(word_counts, key=word_counts.get, reverse=True)

text_length = len(text)
num_unique_words = len(words)
print(f"the text contains {text_length} words")
print(f"there are {num_unique_words} unique tokens")

word_to_int = {v: k for k, v in enumerate(words)}
int_to_word = {k: v for k, v in enumerate(words)}

print({k: v for k, v in word_to_int.items() if k in words[:10]})
print({k: v for k, v in int_to_word.items() if v in words[:10]})
```
This code first counts the occurrences of each unique token using `Counter`. Then it creates a list of these tokens sorted by frequency. Finally, it generates two dictionaries: one mapping from tokens to their indexes and another from indexes back to tokens.
x??

---

#### Understanding Unique Tokens
Background context: The next step is understanding the unique tokens in the text data.

:p What are the most frequent 10 tokens in "Anna Karenina"?
??x
The most frequent 10 tokens, based on the provided code, are:
```python
[',', '.', 'the', '"', 'and', 'to', 'of', 'he', "'", 'a']
```
These tokens represent commas, periods, and common words like "the," "and," "to," etc. The most frequent token is a comma (,), followed by the period (.).
x??

---

#### Using Dictionaries for Indexing
Background context: Once we have the dictionaries, they can be used to convert text data into a numerical format suitable for training an LSTM model.

:p How do you use the `word_to_int` and `int_to_word` dictionaries?
??x
You can use these dictionaries as follows:
```python
# Example usage of word_to_int and int_to_word
print(word_to_int["the"])  # prints: 2
print(int_to_word[5])      # prints: 'to'
```
The `word_to_int` dictionary allows you to get the index corresponding to a token, while `int_to_word` lets you retrieve the token from an index. This is crucial for converting text into numerical data and vice versa during training.
x??

---

#### Summary
---

#### Token Indexing
Background context: In natural language processing, tokenizing involves breaking down text into individual words or tokens. After tokenization, each unique word is assigned an index to facilitate machine learning models like LSTM networks.

:p What are dictionaries `word_to_int` and `int_to_word` used for in the given text?
??x
The `word_to_int` dictionary assigns a unique integer index to each token (word) in the text. The `int_to_word` dictionary does the reverse, translating an index back into its corresponding token.

For example:
```python
# Example of using dictionaries
word = "the"
index = word_to_int[word]  # This gets the index for 'the'
token = int_to_word[index]  # This converts index to the original token
```

x??

---

#### Converting Text to Indexes
Background context: After tokenizing and indexing, the entire text is converted into a list of indexes. This step transforms the raw text data into numerical form that can be understood by machine learning models.

:p How does the text convert to indexes?
??x
The text is converted to indexes using the `word_to_int` dictionary, where each word is replaced with its corresponding index value in a new list called `wordidx`.

For example:
```python
text = "chapter 1 happy families are all alike ; every unhappy family is unhappy in its own way . everything was"
wordidx = [word_to_int[w] for w in text]
```

x??

---

#### Creating Training Pairs (x, y)
Background context: In the context of training language models like LSTM networks, a common approach is to create pairs of input sequences (`x`) and their corresponding target outputs (`y`). These are used to train the model to predict the next word in a sequence based on the previous words.

:p How does the code block in Listing 8.2 create the training data?
??x
The code creates training data by generating pairs of inputs (x) and targets (y). Each input is a sequence of 100 indexes, and each target `y` is the next index after the corresponding input sequence.

Here’s how it works:
```python
seq_len = 100
xys = []
for n in range(0, len(wordidx) - seq_len - 1):
    xys.append((wordidx[n:n+seq_len], wordidx[n+seq_len]))
```

This means that for every index `n` from 0 to the length of `wordidx` minus `seq_len` and one (to ensure enough data), a pair `(x, y)` is created where:
- `x` is a sequence of indexes from `wordidx[n:n+seq_len]`
- `y` is the next index in `wordidx`, i.e., `wordidx[n+seq_len]`

This process ensures that each input sequence has a corresponding target output for training.

x??

---

#### Training Data Batch Size
Background context: The choice of sequence length (`seq_len`) when creating training data can affect model performance and training speed. Common choices include 90, 100, or 110 tokens per sequence, but the exact value should be chosen based on experimentation to balance between capturing long-range dependencies and maintaining efficient training.

:p What factors influence the choice of `seq_len` when creating training data?
??x
The choice of `seq_len` influences both model performance and training efficiency. A smaller `seq_len` can lead to a faster but less informative sequence, while a larger `seq_len` can capture long-range dependencies more effectively but may slow down training due to increased complexity.

For instance:
- **90 tokens**: Faster training but limited context.
- **100 tokens**: Balanced choice, often used in practice.
- **110 tokens**: More context and potentially better model performance but slower training.

The exact value should be determined through experimentation based on the specific task requirements.

x??

---

#### Sliding Window Technique
Background context: The sliding window technique is a common method for generating sequence data where each input sequence (`x`) consists of a fixed number of tokens, and the target output (`y`) is the next token after `x`.

:p How does the sliding window technique work in creating training pairs?
??x
The sliding window technique works by taking sequences of a fixed length (e.g., 100 indexes) from the text and using the next index as the target. This process repeats, moving the window one step at a time until the end of the text is reached.

For example:
```python
seq_len = 100
xys = []
for n in range(0, len(wordidx) - seq_len - 1):
    xys.append((wordidx[n:n+seq_len], wordidx[n+seq_len]))
```

Here’s the breakdown:
- `n` iterates from 0 to `len(wordidx) - seq_len - 1`.
- For each `n`, a pair `(x, y)` is created where:
  - `x` is a sequence of indexes from `wordidx[n:n+seq_len]`.
  - `y` is the next index in `wordidx`, i.e., `wordidx[n+seq_len]`.

This approach ensures that every possible sequence of length `seq_len` has an associated target, allowing the model to learn patterns and dependencies effectively.

x??

---

#### Word Embedding Layer
Background context explaining how the `torch.nn.Embedding` layer works. It is a trainable lookup table that maps integer indexes to dense, continuous vector representations (embeddings). When you create an instance of `torch.nn.Embedding()`, you need to specify two main parameters: `num_embeddings`, the size of the vocabulary (total number of unique tokens), and `embedding_dim`, the size of each embedding vector.
:p How does the `torch.nn.Embedding` class work?
??x
The `torch.nn.Embedding` class creates a matrix or lookup table where each row corresponds to an integer index mapping to a dense, continuous vector representation. Initially, these embeddings are randomly initialized but are learned and updated during training through backpropagation.
```python
vocab_size = len(word_to_int)
embedding_dim = 128
embed_layer = nn.Embedding(vocab_size, embedding_dim)
```
x??

---

#### LSTM Layer in WordLSTM Model
Explanation of the LSTM layer within the `WordLSTM` model. The LSTM layer processes elements of a sequence in a sequential manner and stacks three LSTMs together to form a stacked LSTM.
:p What is the purpose of the LSTM layer in the `WordLSTM` model?
??x
The LSTM layer processes elements of a sequence sequentially, allowing the model to capture temporal dependencies within sequences. By stacking three LSTMs, the last two take the output from the previous LSTM as input, enhancing the model's ability to understand long-term dependencies.
```python
n_layers = 3
lstm_layer = nn.LSTM(input_size=self.input_size,
                     hidden_size=self.n_embed,
                     num_layers=n_layers,
                     dropout=self.drop_prob,
                     batch_first=True)
```
x??

---

#### Forward Method in WordLSTM Model
Explanation of the forward method within the `WordLSTM` model. It involves embedding the input tokens, passing them through an LSTM layer to produce output and hidden states, and then applying a linear transformation.
:p What does the `forward` method do in the `WordLSTM` model?
??x
The `forward` method embeds the input tokens into dense vectors using the embedding layer. It then passes these embeddings through the LSTM layer to generate outputs and hidden states. Finally, it applies a linear transformation to produce the final output logits.
```python
def forward(self, x, hc):
    embed = self.embedding(x)
    x, hc = self.lstm(embed, hc)
    x = self.fc(x)
    return x, hc
```
x??

---

#### init_hidden Method in WordLSTM Model
Explanation of the `init_hidden` method. It initializes the hidden state with zeros to be used when making predictions on the first element in the sequence.
:p What is the purpose of the `init_hidden` method?
??x
The `init_hidden` method initializes the hidden state with zeros, which is used as an initial condition when the model starts processing the first token in a sequence. This ensures that there are no biases introduced from previous sequences or batches.
```python
def init_hidden(self, n_seqs):
    weight = next(self.parameters()).data
    return (weight.new(self.n_layers,
                       n_seqs,
                       self.n_embed).zero_(),
            weight.new(self.n_layers,
                       n_seqs,
                       self.n_embed).zero_())
```
x??

---

#### Training the WordLSTM Model
Explanation of how to train the `WordLSTM` model using the training data. It involves creating batches, initializing the model and optimizer, and iterating over the dataset to update the model's weights.
:p How do you train the `WordLSTM` model?
??x
To train the `WordLSTM` model, you create a DataLoader for batching the input data, initialize the model and an optimizer (like Adam), and then iterate over the dataset. During each epoch, you process batches of input-output pairs to update the model's weights using backpropagation.
```python
from torch.optim import Adam

# Initialize model and optimizer
model = WordLSTM()
optimizer = Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    for batch_xys in loader:
        x, y = map(lambda t: t.to(device), batch_xys)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.permute(0, 2, 1), y)
        loss.backward()
        optimizer.step()
```
x??

---

