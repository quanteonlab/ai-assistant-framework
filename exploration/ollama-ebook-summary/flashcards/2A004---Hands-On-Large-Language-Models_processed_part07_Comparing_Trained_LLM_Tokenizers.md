# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 7)

**Starting Chapter:** Comparing Trained LLM Tokenizers

---

#### Subword Tokenizers and Byte-Level Representations

Background context explaining the concept. In natural language processing, subword tokenization is a method where words are represented as sequences of smaller units called subwords. This can be particularly useful for handling out-of-vocabulary (OOV) words or rare words in multilingual scenarios. Some subword tokenizers use bytes as tokens to represent characters they cannot otherwise encode.

:p What is the role of bytes in some subword tokenizers?
??x
Bytes are used as a fallback mechanism when a subword tokenizer encounters characters it can't represent using its primary vocabulary. For example, GPT-2 and RoBERTa tokenizers include bytes in their vocabulary to handle these cases but do not use them for everything; they only fall back to bytes for a subset of characters.

```
Example: 
tokenizer = AutoTokenizer.from_pretrained('gpt2')
token_ids = tokenizer("Hello 🌟", add_special_tokens=False).input_ids
print(tokenizer.decode(token_ids))
```

x??

---

#### Tokenization Factors

Background context explaining the concept. The way a text is tokenized can significantly impact model performance, and this depends on several factors: the tokenization method, special tokens used for initialization, and the dataset used to train the tokenizer.

:p What are the three major factors that determine the tokens within a tokenizer?
??x
The three major factors are:
1. **Tokenization Method**: How words or characters are broken down into smaller units.
2. **Parameters and Special Tokens**: The specific tokens used for initialization, like beginning and end of sentence tokens.
3. **Dataset Used for Training**: The data on which the tokenizer is trained.

x??

---

#### Tokenizers for Different Languages

Background context explaining the concept. Comparing different tokenizers can help us understand how these factors change their behavior. For instance, some newer tokenizers have adjusted their methods to improve model performance in various languages and scenarios.

:p How do tokenizers handle different types of text as mentioned?
??x
Tokenizers handle different types of text including:
- **Capitalization**: Different tokenizers might tokenize capital letters differently.
- **Languages other than English**: Tokenizers like BERT and RoBERTa support multiple languages but handle them based on their training data.
- **Emojis**: Emojis are often treated as special tokens or subwords depending on the tokenizer.
- **Programming code with keywords and whitespaces**: Specialized tokenizers for code generation models can tokenize code more effectively, handling indentation and keywords.

Example:
```python
from transformers import AutoTokenizer

text = "English and CAPITALIZATION 🎵鸟 show_tokens False None elif == >= else: two tabs:\" \" Three tabs: \"   \" 12.0*50=600 \""

tokenizers_to_compare = [
    ('bert-base-uncased', 'BERT base model (uncased)'),
    ('roberta-base', 'RoBERTa base model'),
    # Add more tokenizers to compare
]

for tokenizer_name, description in tokenizers_to_compare:
    print(f"### {description}")
    show_tokens(text, tokenizer_name)
```

x??

---

#### Special Tokens

Background context explaining the concept. Special tokens are unique tokens that serve specific roles other than representing text content. These include tokens for the beginning and end of a sentence or paragraph.

:p What are some examples of special tokens?
??x
Examples of special tokens include:
- **[UNK]**: An unknown token used when the tokenizer encounters an out-of-vocabulary word.
- **[SEP]**: A separator token that is useful in tasks requiring two inputs, like text classification.
- **[CLS]**: A classification token often used as input for models where a single output needs to be generated from the entire sequence.

x??

---

#### BERT Tokenizer Example

Background context explaining the concept. The BERT tokenizer is an example of how special tokens are utilized in practice. BERT uses a specific set of special tokens to help with tasks like classification and sentence pair modeling.

:p How does BERT handle tokenization?
??x
BERT handles tokenization by using WordPiece, which breaks down words into subwords or individual characters if necessary. It includes several special tokens:
- **[UNK]**: Represents unknown or out-of-vocabulary words.
- **[CLS]** and **[SEP]**: Used for tasks that require separating two sentences or marking the start of a sentence.

Example code to tokenize text with BERT:
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "Hello world [SEP] How are you?"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
```

x??

---

#### Reranking Concept
Background context: In natural language processing, reranking is a technique often used after generating candidate outputs from an NLP model. The purpose of reranking is to improve the quality and relevance of generated text by reordering or adjusting the probabilities of different hypotheses based on additional criteria.

:p What is reranking in the context of NLP?
??x
Reranking is a process where candidate outputs from a language model are reordered or their probabilities adjusted based on additional criteria, improving the relevance and quality of generated text.
x??

---

#### Pad Token [PAD]
Background context: The pad token ([PAD]) is used to ensure that all input sequences have the same length by padding them with tokens. This is necessary because many models require a fixed-length input sequence.

:p What is the purpose of the pad token in NLP models?
??x
The pad token is used to pad unused positions in the model's input, ensuring that all input sequences are of a certain fixed length.
x??

---

#### CLS Token [CLS]
Background context: The classification ([CLS]) token is used for tasks like sentence or document classification. It typically appears at the beginning of the sequence and helps the model understand the overall context.

:p What is the purpose of the cls_token in NLP models?
??x
The cls_token serves as a special classification token placed at the start of the input sequence to help the model classify the entire sequence.
x??

---

#### Mask Token [MASK]
Background context: The mask token ([MASK]) is used during training to hide certain tokens and predict them, allowing the model to learn better generalization and contextual understanding.

:p What is the purpose of the mask_token in NLP models?
??x
The mask_token is used to hide certain tokens during training so that the model can predict these tokens, enhancing its ability to understand context.
x??

---

#### BERT Tokenizer Example
Background context: The BERT tokenizer processes text into subtokens and adds special tokens like [CLS] and [SEP]. It supports both cased and uncased versions.

:p What are the key differences between BERT's cased and uncased tokenizers?
??x
The main difference is that the cased version preserves capitalization, whereas the uncased version converts all text to lowercase. The cased version also encodes "CAPITALIZATION" as eight subtokens, while the uncased version does so in four.
x??

---

#### GPT-2 Tokenizer Example
Background context: The GPT-2 tokenizer uses Byte Pair Encoding (BPE) and retains newline breaks and capitalization.

:p What is unique about the GPT-2 tokenizer compared to BERT?
??x
GPT-2 retains newline breaks and preserves capitalization. It encodes "CAPITALIZATION" as four tokens, whereas BERT encodes it as eight.
x??

---

#### Special Tokens in Both Tokenizers
Background context: Both BERT and GPT-2 use special utility tokens like [CLS] for classification and [SEP] to separate sentences.

:p What are the roles of [CLS] and [SEP] tokens?
??x
[CLS] is used for classification tasks, typically at the start of a sequence. [SEP] separates different segments in sequences, such as two sentences.
x??

---

#### WordPiece vs BPE
Background context: BERT uses WordPiece tokenization, while GPT-2 uses Byte Pair Encoding (BPE).

:p What are the differences between WordPiece and BPE?
??x
WordPiece splits words into subwords based on their frequency. BPE merges common character sequences to create new tokens.
x??

---

#### Tokenized Text Example
Background context: The provided text demonstrates how different tokenizers handle text differently, including special cases like unknown tokens ([UNK]), tabs, and numbers.

:p How does the BERT tokenizer handle "capitalization"?
??x
The BERT tokenizer handles "capitalization" by encoding it as eight subtokens: capital  ##ization.
x??

---

#### Tokenized Text Example (GPT-2)
Background context: The provided text also shows how GPT-2 tokenizes similar text, including handling capitalization and newline breaks.

:p How does the GPT-2 tokenizer handle "capitalization"?
??x
The GPT-2 tokenizer handles "capitalization" by encoding it as four tokens.
x??

---

#### Tokenization and Whitespace Handling

Background context: In tokenization, whitespace characters play a crucial role. Different models handle whitespaces differently to optimize performance on specific tasks.

:p How does tokenizing whitespaces affect model performance?
??x
Whitespace handling can significantly impact model performance. For instance, using fewer tokens for multiple consecutive spaces simplifies the model's task but can lead to issues with understanding context and maintaining proper indentation in code. 
```python
# Example: Code snippet with different whitespace representations
code = "def function(param):\n    return param + 1\n"
```
x??

---

#### Tokenization of Special Characters

Background context: The use of special characters, such as emojis or Chinese characters, can affect how a model processes and generates text. Some models may replace these with an unknown token to simplify the task.

:p How do special characters like emojis and Chinese characters typically get handled in tokenized text?
??x
Special characters often get replaced by an unknown token (<unk>) because not all tokens are designed to handle such diverse inputs efficiently.
```python
# Example: Tokenization of a sentence with special characters
text = "Hello 🌟, how are you?"
tokens = tokenizer.tokenize(text)
```
x??

---

#### SentencePiece and Flan-T5

Background context: The Flan-T5 model uses the SentencePiece method for tokenization. This approach is effective but has limitations when dealing with new or rare tokens.

:p What does the Flan-T5 use for its tokenization?
??x
Flan-T5 uses the SentencePiece method, which supports BPE and unigram language models.
```python
# Example: Tokenization using SentencePiece
import sentencepiece as spm

model = spm.SentencePieceProcessor()
model.load("flant5_model.spiece")
text = "Hello, how are you?"
tokens = model.encode_as_pieces(text)
print(tokens)
```
x??

---

#### BPE in GPT-4 Tokenization

Background context: The GPT-4 model uses Byte Pair Encoding (BPE) for tokenization. This method is known for its efficiency and ability to handle a wide range of input types.

:p How does the GPT-4 model tokenize text?
??x
The GPT-4 model uses BPE for tokenization, allowing it to efficiently process a large vocabulary size with around 100,000 special tokens.
```python
# Example: Tokenization using BPE
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(tokens)
```
x??

---

#### Fill-in-the-Middle Tokens in LLMs

Background context: Some language models use fill-in-the-middle tokens to improve the ability of the model to generate coherent text by considering future text during training.

:p What is the purpose of fill-in-the-middle (FTM) tokens?
??x
The purpose of FTM tokens is to enable the model to consider the future context when generating text, leading to more coherent and accurate completions.
```python
# Example: Using FTM tokens for text generation
def generate_text(prompt):
    # Assume some function that generates text using FTM tokens
    generated_text = model.generate(prompt + " —", max_length=50)
    return generated_text

prompt = "The weather is nice today, and"
print(generate_text(prompt))
```
x??

---

#### GPT-4 Tokenizer Differences from GPT-2
Background context: The GPT-4 tokenizer has several differences compared to its predecessor, GPT-2. These differences include how it handles whitespace, code-specific tokens, and word representations.

:p How does the GPT-4 tokenizer represent whitespace?
??x
The GPT-4 tokenizer represents a sequence of four spaces as a single token and introduces specific tokens for up to 83 consecutive whitespaces. This specialization is due to its focus on handling code more effectively.
x??

---

#### StarCoder2 Tokenization Method
Background context: StarCoder2, a 15-billion parameter model, uses byte pair encoding (BPE) for tokenization and focuses on generating code. It introduces special tokens like —<filename>— and —<reponame>— to manage code contexts across different files in the same repository.

:p What are some of the special tokens used by StarCoder2?
??x
StarCoder2 uses several special tokens such as —<filename>—, —<reponame>—, and —<gh_stars>— to help manage code contexts. These tokens are crucial for identifying code that spans different files within the same repository.
x??

---

#### Tokenization of Numbers in StarCoder2
Background context: Unlike GPT-2, where numbers like 871 are represented as two separate tokens (8 and 71), StarCoder2 assigns each digit its own token to better represent numbers and mathematics.

:p How does StarCoder2 tokenize the number 600?
??x
In StarCoder2, the number 600 is tokenized into individual digits: 6, 0, 0. This approach ensures that numbers are represented more distinctly, reducing potential confusion in the model.
x??

---

#### Galactica Special Tokens for Scientific Knowledge
Background context: The Galactica model, designed for scientific knowledge, uses special tokens to enhance its understanding of specific elements like citations, reasoning, and amino acid sequences.

:p What are some special tokens used by Galactica?
??x
Galactica includes several special tokens such as —[START_REF]—, —[END_REF]—, and —<work>— for handling references, reasoning, and specific scientific terms. These tokens help the model better understand the nuances of scientific text.
x??

---

#### Example Galactica Tokenization
Background context: The Galactica model tokenizes scientific texts to improve its understanding of specialized content like citations and amino acid sequences.

:p Can you provide an example of a citation in Galactica's tokenized format?
??x
Sure, here’s an example of a citation in Galactica's tokenized format:
```
Recurrent neural net — works, long short-term memory [START_REF]Long Short-Term Memory, Hochreiter[END_REF]
```
In this case, the reference "Hochreiter" is wrapped within the special tokens —[START_REF]— and —[END_REF]—.
x??

---

#### Chain-of-Thought Reasoning in Galactica
Background context: The Galactica model includes a token called —<work>— that is used for chain-of-thought reasoning, which helps the model follow logical steps in problem-solving.

:p What role does the —<work>— token play in Galactica?
??x
The —<work>— token in Galactica serves to facilitate chain-of-thought reasoning. It allows the model to break down complex problems into step-by-step processes, making it easier to follow logical sequences of thought.
x??

---

#### Byte Pair Encoding (BPE) Overview
Background context explaining BPE, a method used to create subword units from bytes. It is a popular technique for tokenizing natural language text. BPE works by repeatedly merging the most frequent byte pairs until it reaches a specified vocabulary size.

:p What is Byte Pair Encoding (BPE)?
??x
Byte Pair Encoding (BPE) is an algorithmic method used in tokenizer models to generate subword units from bytes, aiming to create a finite set of token types. It operates by iteratively merging the most frequent byte pairs until it reaches a predetermined vocabulary size.
x??

---

#### Tokenization Methods Comparison
Background context comparing different tokenizers and their unique features. The text discusses several tokenizers: BERT base model (uncased), BERT base model (cased), GPT-2, FLAN-T5, GPT-4, StarCoder, and Phi-3/Llama 2.

:p How do the tokenizers described in the text differ?
??x
The tokenizers discussed vary in their handling of special characters, whitespace, tabs, and numbers. For instance:
- BERT base model (uncased) does not preserve uppercase letters.
- BERT base model (cased) preserves uppercase letters but uses special tokens like `##` to indicate subwords.
- GPT-2, FLAN-T5, and StarCoder do not use `##` for subword indication.
- Phi-3/Llama 2 introduces several chat-related special tokens such as `<|user|>`, `<|assistant|>`, and `<|system|>`.

Additionally, some tokenizers treat whitespace differently, with StarCoder being unique in handling sequences of two tabs as a single token (`\t\t`).
x??

---

#### Handling Special Characters
Background context on how different tokenizers handle special characters like `##` for subword tokens. The text mentions BERT base model (cased) and GPT-2, which use the `##` symbol.

:p How do BERT base model (cased) and GPT-2 handle special characters?
??x
BERT base model (cased) uses the `##` symbol to indicate that a subword should be prefixed to the previous word. For example:
```
CA ##PI  ##TA  ##L  ##I  ##Z  ##AT  ##ION
```

GPT-2, on the other hand, does not use any special symbols for subwords and relies solely on context.
x??

---

#### White Space Handling
Background context explaining how different tokenizers handle whitespace. The text mentions that some tokenizers treat whitespaces uniformly, while others have specific handling for tabs.

:p How do different tokenizers handle whitespace?
??x
Different tokenizers handle whitespace differently:
- BERT base model (cased) and GPT-2 use a uniform tokenization approach where sequences of spaces are represented as single tokens.
- StarCoder is unique in its handling of tabs, treating two consecutive tab characters (`\t\t`) as a single token.

This difference affects how these models tokenize text that contains multiple whitespaces or tabs.
x??

---

#### Chat Tokens
Background context on the introduction of special chat-related tokens like `<|user|>`, `<|assistant|>`, and `<|system|>` by Phi-3/Llama 2. These are used to indicate the roles in a conversation.

:p What special tokens does Phi-3 (and Llama 2) introduce for chat scenarios?
??x
Phi-3 (and Llama 2) introduces several special tokens specifically designed for chat scenarios:
```
<|user|> — User input token.
<|assistant|> — Assistant response token.
<|system|> — System or environment information token.
```

These tokens help in distinguishing different roles and context within a conversation, enhancing the model's ability to handle natural language conversations more effectively.
x??

---

#### Tokenization Examples
Background context providing examples of how various tokenizers tokenize specific phrases. The text includes examples like "english  and  capital" and shows how different models handle these cases.

:p What are some tokenization examples provided in the text?
??x
Here are some tokenization examples from the text:
- BERT base model (uncased):
  ```
  [CLS] english and capital ##ization [UNK] show _ tokens false none eli ##f = = > = else : two tab ##s : " " three tab ##s : " " 12 . 0 * 50 = 600 [SEP]
  ```

- BERT base model (cased):
  ```
  [CLS] English and CA ##PI ##TA ##L ##I ##Z ##AT ##ION [UNK] show _ tokens false none el if == >= else : two tabs : " " three tabs : " " 12 . 0 * 50 = 600 [SEP]
  ```

- GPT-2:
  ```
  English and CAP ITAL IZ ATION show _ t ok ens False None elif == >= else : two tabs : " " Three tabs : " " 12. 0 * 50 = 600 FLAN-T5
  ```

- StarCoder:
  ```
  English and CAPITAL IZATION show _ tokens False None elif == >= else : two tabs : " " Three tabs : " " 12 . 0 * 50 = 600
  ```

These examples illustrate how different tokenizers handle capitalization, special characters, and whitespaces.
x??

---

#### Tokenization Methods Overview
Tokenization methods refer to algorithms used by tokenizers to break down texts into tokens. Popular methods include Byte Pair Encoding (BPE). Each method has its own way of determining which sequences of characters should form a single token.

:p What are some common tokenization methods?
??x
Common tokenization methods include Byte Pair Encoding (BPE), SentencePiece, and WordPiece. These methods differ in how they decide the most appropriate set of tokens to represent a dataset.
x??

---

#### Initialization Parameters for Tokenizers
These parameters are crucial as they influence the behavior and performance of the tokenizer during both training and inference phases.

:p What parameters might an LLM designer need to configure when initializing a tokenizer?
??x
An LLM designer needs to configure several parameters, such as:
- Vocabulary size: How many unique tokens to keep in the model's vocabulary.
- Special tokens: Tokens like beginning of text (e.g., <s>), end of text, padding, unknown, etc.

Example relevant code snippet:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased',
                                          additional_special_tokens=['<work>', '[START_REF]'])
```
x??

---

#### Domain-Specific Tokens
Certain tokenizers include domain-specific tokens to better represent the context of specific use cases.

:p What are examples of special tokens that might be added for a particular task?
??x
Examples of special tokens include:
- `<s>`: Beginning of text.
- `</s>`: End of text.
- `[PAD]`: Padding token.
- `[UNK]`: Unknown token.
- `[CLS]`: Classification token.
- Masking token.

For example, in a domain-specific model like Galactica, additional tokens such as `<work>` and `[START_REF]` are used to better handle the specific context of that domain.
x??

---

#### Impact of Tokenization on Different Domains
The choice of tokenizer can significantly affect how well a language model performs on different types of text.

:p How does tokenization impact models differently based on their training data?
??x
Tokenization methods, even with the same parameters, can behave differently depending on the specific dataset they were trained on. For instance:
- Text-focused tokenizers might tokenize indentations in code as separate tokens.
- Code-focused tokenizers may handle these better by treating indents as part of a function definition.

Example:

```python
# Example tokenizer output for text vs. code
text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
code_tokenizer = AutoTokenizer.from_pretrained('github-codebert')

text_input = "def add_numbers(a, b):"
print(text_tokenizer.tokenize(text_input))

code_input = "def add_numbers(a, b):"
print(code_tokenizer.tokenize(code_input))
```
x??

---

#### Capitalization Handling
Handling capitalization can vary between languages and affects how tokens are processed.

:p How do tokenizers typically handle capitalization in English text?
??x
In English, common practices include converting everything to lowercase to save space. However, name capitalization often carries useful information that might be lost by converting everything to lower case.

Example:

```python
# Lowercase conversion
text = "This Is A Test"
lowercased_text = text.lower()
print(lowercased_text)  # Output: this is a test

# Capitalization handling
def handle_capitalization(text):
    tokens = tokenizer.tokenize(text)
    for i, token in enumerate(tokens):
        if any(c.isupper() for c in token):  # Special case for names like "John"
            tokens[i] = token.capitalize()
    return ' '.join(tokens)

print(handle_capitalization("This Is A Test"))  # Output: This is a test
```
x??

---

#### Tokenizer Training and Domain-Specific Data
The choice of tokenizer can significantly influence model performance, especially when dealing with domain-specific data.

:p How does the training dataset affect the behavior of tokenizers?
??x
Tokenizers are trained on specific datasets to optimize their vocabulary for that particular type of text. For example:
- A code-focused tokenizer may treat indentation differently compared to a general-purpose text tokenizer.
- Multilingual tokenizers are optimized for handling multiple languages and their unique characters.

Example:

```python
# Training a tokenizer on code data
from transformers import BytePairTokenizer

code_tokenizer = BytePairTokenizer.from_pretrained('github-codebert')
code_input = "def add_numbers(a, b):"
tokens = code_tokenizer.tokenize(code_input)
print(tokens)  # Output: ['de', 'f', 'add_nu', '[UNK]', 'b', ':']
```
x??

---

#### Tokenization and Token Embeddings Overview
Background context explaining tokenization as breaking down texts into smaller units (tokens) which are then represented numerically. This process is crucial for language models to understand and generate text.

:p What is tokenization, and why is it important?
??x
Tokenization involves splitting a sequence of characters into words or tokens. It's the first step in preparing text data for input into machine learning models like language models. This process helps in understanding the structure of sentences and making them amenable to numerical processing.
x??

---
#### Token Embeddings and Language Models
Explanation on how token embeddings are used as numeric representations for tokens to help language models understand patterns in text.

:p How do token embeddings assist language models?
??x
Token embeddings transform each word or token into a vector of numbers, allowing the model to capture semantic meanings. These vectors enable the model to recognize and generate coherent sentences by understanding the relationships between words.
x??

---
#### Pretrained Language Models and Their Tokenizers
Explanation on how pretrained language models are linked with specific tokenizers.

:p Why can't a pretrained language model use a different tokenizer without retraining?
??x
A pretrained language model is closely tied to its tokenizer because it has learned embeddings for each token in the vocabulary of that particular tokenizer. Changing the tokenizer would mean using an entirely new set of tokens, which requires reinitializing and potentially retraining the entire model.
x??

---
#### Contextualized Word Embeddings with Language Models
Explanation on how language models generate different embeddings for words based on their context.

:p How do contextualized word embeddings differ from static token embeddings?
??x
Contextualized word embeddings create a vector representation of each token that changes depending on its context within the sentence. This contrasts with static token embeddings, which remain constant regardless of context.
x??

---
#### DeBERTa and Its Use in Token Embeddings
Explanation on how DeBERTa models are used for generating contextualized word embeddings.

:p How is the DeBERTa model used to generate embeddings?
??x
The DeBERTa model processes input tokens through its layers, producing contextualized embeddings that reflect the semantic meaning of each token based on its context within a sentence. These embeddings help in tasks like named-entity recognition and text summarization.
x??

---
#### Generating Contextualized Embeddings with Code
Explanation on using DeBERTa for generating embeddings and interpreting the results.

:p How do we generate contextualized embeddings using DeBERTa?
??x
To generate contextualized embeddings with DeBERTa, you first load a tokenizer and model. Then, tokenize your input text, pass it through the model to get embeddings, and inspect these embeddings to understand their context-dependent nature.

```python
from transformers import AutoModel, AutoTokenizer

# Load a tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# Tokenize the sentence
tokens = tokenizer('Hello world', return_tensors='pt')

# Process the tokens to get embeddings
output = model(**tokens)[0]

# Inspect the output shape and individual token embeddings
print(output.shape)
for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))
```
x??

---

